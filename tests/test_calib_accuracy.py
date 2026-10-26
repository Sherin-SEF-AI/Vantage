"""Calibration accuracy by Monte Carlo against independent references.

The installation check's joint solve (core.joint.joint_solve) on the 6-camera sample, boards laid by hand: simulated
corners (0.5 px noise, the boards off their plan by the placement prior, the cameras off theirs by a mounting error) are
solved exactly as the installation check solves them (each camera's own PnP, then the joint solve). Every answer is
compared with a reference solve of the same problem written here from scratch (own parameterisation, Gauss–Newton with
a central-difference Jacobian, iteratively reweighted for the Huber loss, run to convergence), and the spread of the
answers with the covariance (JᵀJ)⁻¹ of that reference. The joint solve used to stop short of its optimum (an
iterative trust-region solver crawling along the board priors' shallow directions): 38 of 40 trials, 1.2–3.3× the
spread. A corrupted board is still set aside.

The lidar–camera calibration's planned 1σ (core.lidar_calib.predicted_sigma) against the spread of its own solve
(core.lidar_calib._solve) on simulated corners and lidar points of the planned board poses, and the solve's bias.
"""
import math
import os

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np
import pytest

from vantage.core import samples
from vantage.core.install import BoardObs, board_outline, inner_corners, solve_pose
from vantage.core.lens import resolve
from vantage.core.mathutil import BODY_TO_OPTICAL, rot_z, rot_zyx

PX = 0.5                 # corner noise, px: the joint solve's own weight, so (JᵀJ)⁻¹ is the estimator's covariance
SB_MM, SB_DEG = 20.0, 1.0        # boards laid by hand (the installation check's placement prior)
HUBER = 3.0                      # the joint solve's Huber scale, in units of PX


def _exp(w):
    th = float(np.linalg.norm(w))
    if th < 1e-15:
        return np.eye(3)
    k = np.asarray(w, float) / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K


def _log(R):
    th = math.acos(max(-1.0, min(1.0, (float(np.trace(R)) - 1) / 2)))
    if th < 1e-12:
        return np.zeros(3)
    return th / (2 * math.sin(th)) * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])


class _Rig:
    """The sample's six cameras and eight floor boards, and which camera sees which board whole (in the image, in
    front, nothing of the vehicle in the way, squares of 8 px or more)."""

    def __init__(self):
        from vantage.core.analysis import Engine
        self.p = samples.tigor_sample_project()
        eng = Engine()
        vg = eng.vehicle_geom(self.p)
        self.cams = list(self.p.variant.cameras)
        self.targets = list(self.p.targets)
        self.intr = [resolve(self.p.module_for(c), self.p.lens_for(c)) for c in self.cams]
        self.R = [rot_zyx(c.yaw, c.pitch, c.roll) for c in self.cams]
        self.P = [c.pos().astype(float) for c in self.cams]
        self.W = [inner_corners(t).reshape(-1, 3) for t in self.targets]
        self.ctr = [np.array([t.x, t.y, 0.0]) for t in self.targets]
        self.seen = []
        for ci in range(len(self.cams)):
            cube = eng.cube(self.P[ci], vg)
            for bi, t in enumerate(self.targets):
                Q = np.vstack([self.W[bi], board_outline(t)])
                d = (Q - self.P[ci]) @ (BODY_TO_OPTICAL @ self.R[ci].T).T
                u, v, _, ok = self.intr[ci].project_dirs(d)
                dist = np.linalg.norm(Q - self.P[ci], axis=1)
                if not ok.all() or (d[:, 2] / dist < 0.05).any():
                    continue
                if not (cube.lookup(Q - self.P[ci])[0] > dist - (0.02 + 0.01 * dist)).all():
                    continue
                G = np.stack([u, v], -1)[:len(self.W[bi])].reshape(t.rows - 1, t.cols - 1, 2)
                if min(np.linalg.norm(np.diff(G, axis=a), axis=-1).min() for a in (0, 1)) < 8:
                    continue
                self.seen.append((ci, bi))

    def uv(self, ci, R, pos, Q):
        u, v, _, _ = self.intr[ci].project_dirs((Q - pos) @ (BODY_TO_OPTICAL @ R.T).T)
        return np.stack([u, v], -1)

    def laid(self, bi, b):
        return (self.W[bi] - self.ctr[bi]) @ rot_z(b[2]).T + self.ctr[bi] + np.array([b[0], b[1], 0.0])

    def shapes(self):
        return {c.uid: (it.height, it.width) for c, it in zip(self.cams, self.intr)}

    def capture(self, rng, placed=True):
        """True poses (the plan off by up to 1° and 10 mm), true board placements (drawn from the prior) and the noisy
        corners: ({uid: [BoardObs]}, [(camera, board, uv)], true poses, true boards (nb, 3))."""
        nb = len(self.targets)
        truth = []
        for c, P in zip(self.cams, self.P):
            d = rng.uniform(-1, 1, 3)
            truth.append((rot_zyx(c.yaw + d[0], c.pitch + d[1], c.roll + d[2]), P + rng.uniform(-0.01, 0.01, 3)))
        boards = (np.c_[rng.normal(0, SB_MM / 1000, nb), rng.normal(0, SB_MM / 1000, nb),
                        rng.normal(0, math.radians(SB_DEG), nb)] if placed else np.zeros((nb, 3)))
        return self.recapture(rng, truth, boards)

    def recapture(self, rng, truth, boards):
        """Noisy corners of given true poses and board placements: (obs, data, truth, boards)."""
        obs, data = {}, []
        for ci, bi in self.seen:
            uv = self.uv(ci, *truth[ci], self.laid(bi, boards[bi])) + rng.normal(0, PX, (len(self.W[bi]), 2))
            rays, _ = self.intr[ci].unproject(uv[:, 0], uv[:, 1])
            t = self.targets[bi]
            obs.setdefault(self.cams[ci].uid, []).append(BoardObs(t.uid, t.name, self.W[bi], rays, uv, uv.copy(),
                                                                  np.zeros((4, 2))))
            data.append((ci, bi, uv))
        return obs, data, truth, boards

    def own_solves(self, obs):
        """Each camera's own PnP (the boards where the plan puts them): where the installation check starts the joint
        solve."""
        sh = self.shapes()
        return {c.uid: solve_pose(self.p, c, obs[c.uid], sh[c.uid])[:2] for c in self.cams if c.uid in obs}


class _Reference:
    """The joint problem stated independently: camera i at (R_i exp(w_i), p_i + t_i) about its true pose, board b moved
    by (dx, dy, dyaw) about its planned centre; corners weighted 1/PX, boards by their prior, the Huber loss beyond
    HUBER."""

    def __init__(self, rig, data, truth, placed=True):
        self.rig, self.data, self.truth, self.placed = rig, data, truth, placed
        self.nc, self.nb = len(rig.cams), (len(rig.targets) if placed else 0)
        self.prior = np.tile([1000 / SB_MM, 1000 / SB_MM, 1 / math.radians(SB_DEG)], self.nb)

    def x_of(self, poses, boards):
        x = np.zeros(6 * self.nc + 3 * self.nb)
        for ci, ((R0, p0), (R, p)) in enumerate(zip(self.truth, poses)):
            x[6 * ci:6 * ci + 6] = np.r_[_log(R0.T @ np.asarray(R, float)), np.asarray(p, float) - p0]
        if self.placed:
            x[6 * self.nc:] = np.asarray(boards, float).ravel()
        return x

    def resid(self, x):
        out = []
        for ci, bi, uv in self.data:
            R0, p0 = self.truth[ci]
            b = x[6 * self.nc + 3 * bi:6 * self.nc + 3 * bi + 3] if self.placed else np.zeros(3)
            pr = self.rig.uv(ci, R0 @ _exp(x[6 * ci:6 * ci + 3]), p0 + x[6 * ci + 3:6 * ci + 6], self.rig.laid(bi, b))
            out.append(((pr - uv) / PX).ravel())
        out.append(x[6 * self.nc:] * self.prior)
        return np.concatenate(out)

    def jac(self, x, h=1e-6):
        J = np.zeros((len(self.resid(x)), len(x)))
        for k in range(len(x)):
            d = np.zeros(len(x))
            d[k] = h
            J[:, k] = (self.resid(x + d) - self.resid(x - d)) / (2 * h)
        return J

    def weights(self, r):
        """Iteratively reweighted least squares for the Huber loss: √w per residual, w = min(1, HUBER / |r|). The joint
        solve applies the loss to every residual, the boards' priors too (a board laid far off pulls on its prior
        linearly)."""
        return np.sqrt(np.minimum(1.0, HUBER / np.maximum(np.abs(r), 1e-12)))

    def cost(self, x):
        """The Huber cost (the joint solve's objective): r² inside HUBER, 2·HUBER·|r| − HUBER² beyond."""
        a = np.abs(self.resid(x))
        return float(np.sum(np.where(a <= HUBER, a ** 2, 2 * HUBER * a - HUBER ** 2)))

    def solve(self, x):
        """Gauss–Newton on the reweighted problem to convergence: the Huber optimum. Returns (x, (JᵀJ)⁻¹ at it)."""
        x = np.array(x, float)
        for _ in range(40):
            r = self.resid(x)
            w = self.weights(r)
            J = self.jac(x) * w[:, None]
            step = np.linalg.lstsq(J, -r * w, rcond=None)[0]
            x = x + step
            if np.max(np.abs(step)) < 1e-9:            # a millionth of the smallest σ; round-off is ~1e-10
                break
        J = self.jac(x)
        return x, np.linalg.inv(J.T @ J)


@pytest.fixture(scope='module')
def rig():
    return _Rig()


@pytest.fixture
def solver_calls(monkeypatch):
    """Every least_squares call the joint solve makes: (loss, status, nfev)."""
    import scipy.optimize
    real = scipy.optimize.least_squares
    calls = []

    def spy(*a, **kw):
        sol = real(*a, **kw)
        calls.append((kw.get('loss', 'linear'), int(sol.status), int(sol.nfev)))
        return sol
    monkeypatch.setattr(scipy.optimize, 'least_squares', spy)
    return calls


def _joint_x(ref, rig, jr):
    poses = [jr.poses[c.uid] for c in rig.cams]
    fix = {b.uid: b for b in jr.boards}
    boards = np.array([[fix[t.uid].dx_mm / 1000, fix[t.uid].dy_mm / 1000, math.radians(fix[t.uid].dyaw_deg)]
                       if t.uid in fix else [0.0, 0.0, 0.0] for t in rig.targets])
    return ref.x_of(poses, boards)


def test_the_rig_ties_every_camera_to_its_neighbours(rig):
    """The simulated layout is the sample's: every camera sees two or three boards whole, every board but T1 and T8
    (straight ahead, straight behind) is shared by two cameras."""
    tags = [c.tag for c in rig.cams]
    per_cam = {t: sum(1 for ci, _ in rig.seen if tags[ci] == t) for t in tags}
    assert all(n >= 2 for n in per_cam.values()), per_cam
    shared = {rig.targets[b].name for b in {bi for _, bi in rig.seen} if sum(1 for _, x in rig.seen if x == b) >= 2}
    assert len(shared) >= 5, shared


def test_joint_solve_reaches_its_optimum_and_its_spread_matches_the_covariance(rig, solver_calls):
    """40 simulated installation checks with the boards laid by hand. Each joint answer equals the independent
    reference optimum (to 1 % of a σ; the joint solve used to leave a median χ² of 55 on the table), every
    least_squares call stops on its tolerances (not on max_nfev), and the answers scatter as (JᵀJ)⁻¹ says: per camera
    the mean squared Mahalanobis distance of its 6-axis error within 4 standard errors of 6 (it was 11–49), all six
    cameras together within 4 standard errors of 36, every axis's σ within 4.5 standard errors of the prediction (it
    was up to 3.3×)."""
    from vantage.core.joint import joint_solve
    n = 40
    rng = np.random.default_rng(20261026)
    nc = len(rig.cams)
    errs, d2_cam, d2_all, pred = [], [], [], []
    for trial in range(n):
        obs, data, truth, boards = rig.capture(rng)
        solver_calls.clear()
        jr = joint_solve(rig.p, rig.cams, obs, rig.own_solves(obs), rig.shapes(), SB_MM, SB_DEG, pixel_sigma=PX)
        assert jr.ok and not jr.rejected and len(jr.poses) == nc
        assert solver_calls and all(st > 0 for _, st, _ in solver_calls), solver_calls
        ref = _Reference(rig, data, truth)
        xr, C = ref.solve(ref.x_of(truth, boards))
        xj = _joint_x(ref, rig, jr)
        sd = np.sqrt(np.diag(C))
        assert np.max(np.abs(xj - xr) / sd) < 0.01, (trial, np.max(np.abs(xj - xr) / sd))
        assert ref.cost(xj) - ref.cost(xr) < 1e-3, trial
        e_j = xj - ref.x_of(truth, boards)
        errs.append(e_j)
        pred.append(np.diag(C))
        d2_cam.append([e_j[6 * i:6 * i + 6] @ np.linalg.solve(C[6 * i:6 * i + 6, 6 * i:6 * i + 6], e_j[6 * i:6 * i + 6])
                       for i in range(nc)])
        Cc = C[:6 * nc, :6 * nc]
        d2_all.append(e_j[:6 * nc] @ np.linalg.solve(Cc, e_j[:6 * nc]))
    d2_cam = np.mean(d2_cam, axis=0)
    assert np.all(np.abs(d2_cam - 6) < 4 * math.sqrt(12 / n)), d2_cam
    assert abs(np.mean(d2_all) - 6 * nc) < 4 * math.sqrt(12 * nc / n), np.mean(d2_all)
    E = np.array(errs)
    ratio = np.sqrt(np.mean(E ** 2, axis=0) / np.mean(pred, axis=0))
    band = 4.5 / math.sqrt(2 * n)
    assert np.all(np.abs(np.log(ratio)) < band), (ratio.min(), ratio.max())


def test_joint_solve_reaches_its_optimum_with_the_boards_exact(rig, solver_calls):
    """Boards taken as exactly where planned (solve_boards=False): the cameras alone, still the reference optimum."""
    from vantage.core.joint import joint_solve
    rng = np.random.default_rng(7)
    for trial in range(8):
        obs, data, truth, boards = rig.capture(rng, placed=False)
        solver_calls.clear()
        jr = joint_solve(rig.p, rig.cams, obs, rig.own_solves(obs), rig.shapes(), pixel_sigma=PX, solve_boards=False)
        assert all(st > 0 for _, st, _ in solver_calls), solver_calls
        ref = _Reference(rig, data, truth, placed=False)
        xr, C = ref.solve(np.zeros(6 * len(rig.cams)))
        xj = _joint_x(ref, rig, jr)
        assert np.max(np.abs(xj - xr) / np.sqrt(np.diag(C))) < 0.01, trial
        assert all(b.dx_mm == 0 and b.dy_mm == 0 and b.dyaw_deg == 0 for b in jr.boards)
        if trial == 0:
            # a placement tolerance of 0 (batch --board-tol 0) pins the boards: the same answer, not a failure
            j0 = joint_solve(rig.p, rig.cams, obs, rig.own_solves(obs), rig.shapes(), 0.0, 0.0, pixel_sigma=PX)
            assert j0.poses, j0.message
            assert np.max(np.abs(_joint_x(ref, rig, j0) - xr) / np.sqrt(np.diag(C))) < 0.05


@pytest.mark.parametrize('how', ['reversed', 'turned a quarter', 'mirrored'])
def test_a_mismatched_grid_is_set_aside_and_the_rest_solved_to_the_optimum(rig, how):
    """Camera F's view of the board it shares with ML (T2) matched the wrong way round, a quarter turn off, or
    mirrored: the joint solve sets exactly that observation aside, and its answer is the reference optimum without it.
    A square grid looks the same turned a quarter or half round: a solve free to turn the board that far explained
    F's view by turning T2 and set aside ML's view instead (the turn is now bounded to the placement's reach)."""
    from vantage.core.joint import joint_solve
    rng = np.random.default_rng(11)
    obs, data, truth, boards = rig.capture(rng)
    init = rig.own_solves(obs)                   # the camera's own solve has already set the bad board aside
    ci = [c.tag for c in rig.cams].index('F')
    victim = next(k for k, (c, b, _) in enumerate(data) if c == ci and rig.targets[b].name == 'T2')
    t = rig.targets[data[victim][1]]
    o = next(o for o in obs[rig.cams[ci].uid] if o.name == 'T2')
    g = o.image_pts.reshape(t.rows - 1, t.cols - 1, 2)
    o.image_pts = {'reversed': g[::-1, ::-1], 'turned a quarter': np.rot90(g), 'mirrored': g[:, ::-1]}[how] \
        .reshape(-1, 2).copy()
    jr = joint_solve(rig.p, rig.cams, obs, init, rig.shapes(), SB_MM, SB_DEG, pixel_sigma=PX)
    assert [(t_, b) for t_, b, _ in jr.rejected] == [('F', 'T2')]
    assert all(x.name != 'T2' for x in jr.used[rig.cams[ci].uid])
    ref = _Reference(rig, [d for k, d in enumerate(data) if k != victim], truth)
    xr, C = ref.solve(ref.x_of(truth, boards))
    assert np.max(np.abs(_joint_x(ref, rig, jr) - xr) / np.sqrt(np.diag(C))) < 0.01


def test_a_board_laid_well_off_its_place_is_found(rig):
    """T5 laid 70 mm and 2.5° off its place (3.5σ, 2.5σ), the others laid with care (half the prior): the joint
    solve names it first among the moved boards and places it where the reference does, within 3.5σ of the truth (the
    Huber loss on the prior lets the corners move it)."""
    from vantage.core.joint import joint_solve
    rng = np.random.default_rng(3)
    bi = next(i for i, t in enumerate(rig.targets) if t.name == 'T5')
    found = []
    for trial in range(5):
        obs, data, truth, boards = rig.capture(rng)
        boards *= 0.5
        a = rng.uniform(0, 2 * math.pi)
        boards[bi] = (0.07 * math.cos(a), 0.07 * math.sin(a), math.radians(2.5))
        obs, data, _, _ = rig.recapture(rng, truth, boards)
        jr = joint_solve(rig.p, rig.cams, obs, rig.own_solves(obs), rig.shapes(), SB_MM, SB_DEG, pixel_sigma=PX)
        assert not jr.rejected and jr.moved and jr.moved[0].name == 'T5'
        b = next(b for b in jr.boards if b.name == 'T5')
        ref = _Reference(rig, data, truth)
        xr, C = ref.solve(ref.x_of(truth, boards))
        assert np.max(np.abs(_joint_x(ref, rig, jr) - xr) / np.sqrt(np.diag(C))) < 0.01
        k = 6 * len(rig.cams) + 3 * bi
        err = np.array([b.dx_mm / 1000, b.dy_mm / 1000, math.radians(b.dyaw_deg)]) - boards[bi]
        found.append(np.abs(err) / np.sqrt(np.diag(C)[k:k + 3]))
    assert np.max(found) < 3.5, found


# =========================================================================== the lidar–camera calibration
@pytest.fixture(scope='module')
def lidar_pair():
    """The data rig's camera F and roof lidar L1, the calibration's own pose plan, and per pose the board in the optical
    frame and the lidar beams that meet it (cast here from the lidar model's beam grid)."""
    from vantage.core import lidar_calib as LC, sensors as S
    from vantage.core.analysis import Engine
    p = samples.tigor_sample_project()
    p.active_uid = next(v for v in p.variants if v.sensors).uid
    cam = p.variant.by_tag('F')
    lid = next(s for s in p.variant.sensors if s.tag == 'L1')
    st = LC.LidarCalibSettings.from_dict({'camera': 'F', 'lidar': 'L1'})
    poses, _ = LC.plan_poses(p, cam, lid, st, Engine())
    m = p.sensor_model_for(lid)
    intr = resolve(p.module_for(cam), p.lens_for(cam))
    W2O = BODY_TO_OPTICAL @ rot_zyx(cam.yaw, cam.pitch, cam.roll).T
    T0 = LC.plan_T(cam, lid)
    R0, t0 = T0[:3, :3], T0[:3, 3]
    Wb, Hb = st.board_size()
    D = S.directions(m.elevations(), S.scan_azimuths(m)).reshape(-1, 3)            # lidar frame
    boards = []
    for bp in poses:
        Rb, cb = W2O @ bp.R, W2O @ (bp.centre - cam.pos())
        n = Rb[:, 2]
        Dc = D @ R0.T
        with np.errstate(divide='ignore', invalid='ignore'):
            s = np.where(np.abs(Dc @ n) > 1e-9, float(n @ (cb - t0)) / (Dc @ n), -1.0)
        q = (t0 + Dc * s[:, None] - cb) @ Rb
        on = (s > m.range_min_m) & (s < m.range_m) & (np.abs(q[:, 0]) <= Wb / 2) & (np.abs(q[:, 1]) <= Hb / 2)
        boards.append((Rb, cb, D[on], s[on], float(np.sqrt(np.mean((Dc[on] @ n) ** 2)))))
    return dict(p=p, cam=cam, lid=lid, st=st, poses=poses, intr=intr, T0=T0, boards=boards,
                model=LC.model_corners(st), WH=(Wb, Hb))


def _lidar_errors(lp, trials, seed, px=0.3):
    """Simulated captures of the plan (corners with px noise through the lens model, the camera's own PnP of each
    board, lidar ranges with the settings' noise along the beams) solved by lidar_calib._solve (edges and checker
    off): the lidar's pose error (x y z mm, roll pitch yaw deg, vehicle frame) per trial."""
    from vantage.core import lidar_calib as LC
    from vantage.core.mathutil import euler_zyx, wrap180
    rng = np.random.default_rng(seed)
    intr, model, T0, st = lp['intr'], lp['model'], lp['T0'], lp['st']
    sr = st.range_sigma_mm / 1000.0
    R0, t0 = T0[:3, :3], T0[:3, 3]
    Rl0, pl0 = LC.lidar_pose(lp['cam'], T0)
    y0, p0, r0 = euler_zyx(Rl0)
    out = []
    for trial in range(trials):
        obs = []
        for k, (Rb, cb, D, s, cos_rms) in enumerate(lp['boards']):
            u, v, _, _ = intr.project_dirs(model @ Rb.T + cb)
            uv = np.stack([u, v], -1) + rng.normal(0, px, (len(model), 2))
            rays, _ = intr.unproject(uv[:, 0], uv[:, 1])
            R, c, rms, cov = LC._board_pnp(intr, 1.0, 1.0, model, rays, uv)
            pts = D * (s + rng.normal(0, sr, len(s)))[:, None]
            nl = R0.T @ Rb[:, 2]
            lb = LC.LidarBoard(pts, nl, float(nl @ pts.mean(0)), sr * cos_rms, np.zeros((0, 3)), np.zeros((0, 3)), 0)
            obs.append(LC.PoseObs(f'pose{k + 1:02d}', cam=LC.CamBoard(uv, rays, R, c, rms, cov), lid=lb))
        Ts = LC._T(_exp(rng.normal(0, math.radians(0.3), 3)) @ R0, t0 + rng.normal(0, 0.01, 3))
        sol = LC._solve(intr, model, *lp['WH'], obs, Ts, use_edges=False, seed=trial, st=st, use_checker=False)
        assert sol['status'] > 0, sol['status']                 # converged, not stopped by max_nfev
        Rl, pl = LC.lidar_pose(lp['cam'], sol['T'])
        y, p_, r = euler_zyx(Rl)
        out.append([*((pl - pl0) * 1000.0), wrap180(r - r0), p_ - p0, wrap180(y - y0)])
    return np.array(out)


LIDAR_KEYS = ('x_mm', 'y_mm', 'z_mm', 'roll_deg', 'pitch_deg', 'yaw_deg')


def test_lidar_camera_prediction_matches_the_solves_spread_without_bias(lidar_pair):
    """predicted_sigma (each board plane as the lidar's points and as the camera's corners measure it) against 150
    simulated calibrations solved by the calibration's own least squares: every axis's spread within 4.5 standard
    errors of the prediction (it took the camera as good to 1 mm: z 3.4 mm predicted, 16.8 mm the spread), and no
    axis biased beyond 3.5 standard errors of its mean (the solve measured each point's distance along the board's
    normal, the range noise's sideways part taken as signal: x −3.3 mm, 10.8 standard errors, z +3.8 mm)."""
    from vantage.core import lidar_calib as LC
    lp = lidar_pair
    pred = LC.predicted_sigma(lp['p'], lp['cam'], lp['lid'], lp['poses'], lp['st'])
    n = 150
    E = _lidar_errors(lp, n, seed=2026)
    sd = E.std(0, ddof=1)
    band = 4.5 / math.sqrt(2 * (n - 1))
    for k, key in enumerate(LIDAR_KEYS):
        assert abs(math.log(sd[k] / pred[key])) < band, (key, sd[k], pred[key])
        assert abs(E[:, k].mean()) < 3.5 * sd[k] / math.sqrt(n), (key, E[:, k].mean(), sd[k])
    assert pred['z_mm'] > 10                    # the camera's distance to each board limits z: centimetres, not 3 mm


def test_camera_plane_covariance_matches_the_board_pnp(lidar_pair):
    """camera_plane_cov (the corners' noise through the lens model to each board's plane) against 400 noisy corner
    sets of the farthest planned board solved by the calibration's own board PnP: the plane's offset along its normal
    and its two tilts scatter as predicted, and the offset is centimetres at 5–8 m (the plan used to take 1 mm)."""
    from vantage.core import lidar_calib as LC
    lp = lidar_pair
    k = int(np.argmax([b.distance for b in lp['poses']]))
    bp = lp['poses'][k]
    cam, intr, model = lp['cam'], lp['intr'], lp['model']
    W2O = BODY_TO_OPTICAL @ rot_zyx(cam.yaw, cam.pitch, cam.roll).T
    C = LC.camera_plane_cov(intr, W2O, cam.pos().astype(float), bp.centre[None], bp.R[None], model, 0.3)[0]
    Rb, cb = lp['boards'][k][:2]
    n, u, v = Rb[:, 2], Rb[:, 0], Rb[:, 1]
    rng = np.random.default_rng(9)
    out = []
    for _ in range(400):
        pu, pv, _, _ = intr.project_dirs(model @ Rb.T + cb)
        uv = np.stack([pu, pv], -1) + rng.normal(0, 0.3, (len(model), 2))
        rays, _ = intr.unproject(uv[:, 0], uv[:, 1])
        R, c, _, _ = LC._board_pnp(intr, 1.0, 1.0, model, rays, uv)
        w = _log(R @ Rb.T)                                      # the turn about the board's centre, optical frame
        out.append([n @ (c - cb), v @ w, -(u @ w)])
    E = np.array(out)
    ratio = E.std(0, ddof=1) / np.sqrt(np.diag(C))
    assert np.all(np.abs(np.log(ratio)) < 4.5 / math.sqrt(2 * 399)), ratio
    corr = np.corrcoef(E.T)
    pc = C / np.sqrt(np.outer(np.diag(C), np.diag(C)))
    assert np.allclose(corr, pc, atol=0.15), (corr, pc)
    assert math.sqrt(C[0, 0]) > 0.01 and bp.distance > 5
