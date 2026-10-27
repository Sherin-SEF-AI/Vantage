"""Checks of the calibration models: the unit lens calibration, the installation check's joint extrinsic solve against
an independent reference solve, the calibration observability against Monte Carlo solves with the real machinery,
and the lidar–camera calibration."""
from __future__ import annotations

import math

import numpy as np

from .suite import Outcome, check, figure, series

PX = 0.5                       # corner noise, px: the joint solve's own weight, so (JᵀJ)⁻¹ is its covariance
SB_MM, SB_DEG = 20.0, 1.0      # boards laid by hand: the installation check's placement prior
HUBER = 3.0                    # the joint solve's Huber scale, in units of PX


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


# ============================================================================ unit lens calibration
@check('calib.unit_recovery', 'calib', 'A unit calibration recovers a simulated unit\'s lens',
       'RMS pixel distance, over the field out to 65° off the axis, between the lens fitted by calib.calibrate from 16 '
       'synthetic board views (rendered through a unit 1.5 % off the library focal length with an offset principal '
       'point, 2 grey levels of noise, half resolution) and that unit\'s true lens; the library lens must be at least '
       '5 times further off, the focal length within 0.3 %',
       'the simulated unit\'s own intrinsics (the images are rendered from them)', 1.0, 'px',
       refs=('zhang2000', 'kannala2006', 'opencv'), cost='study')
def _unit(ctx):
    from ..core import calib
    from ..core.lens import builtin_lenses, builtin_modules, resolve
    m, wide = builtin_modules()[0], builtin_lenses()[0]
    truth = calib.simulated_unit(m, wide, seed=3)
    views = calib.board_views(m, wide, truth, (9, 6), 60.0, n=16, seed=1)
    r = calib.calibrate(views, (9, 6), 60.0, m, wide)
    if not r.ok:
        raise ValueError(f'the calibration failed: {r.message}')
    rec = r.record
    s = m.width_px / rec['width']
    fit = resolve(m, calib.unit_lens(wide, m, rec))
    true = resolve(m, calib.unit_lens(wide, m, truth))
    field = math.radians(65)
    err = calib.model_difference(fit, true, max_theta=field)
    nominal = calib.model_difference(resolve(m, wide), true, max_theta=field)
    fx_err = abs(rec['fx'] * s - truth['fx']) / truth['fx']
    return Outcome(err['rms_px'], {'fit_rms_px': err['rms_px'], 'fit_max_px': err['max_px'],
                                   'library_rms_px': nominal['rms_px'], 'reprojection_rms_px': r.rms,
                                   'fx_rel_error': fx_err, 'fx_true_px': truth['fx'], 'views': len(views)},
                   ok=nominal['rms_px'] > 5 * err['rms_px'] and fx_err < 3e-3)


# ============================================================================ joint extrinsic solve
class _Rig:
    """The sample's six cameras and eight floor boards, and which camera sees which board whole (in the image, in
    front, nothing of the vehicle in the way, squares of 8 px or more)."""

    def __init__(self, ctx):
        from ..core.install import board_outline, inner_corners
        from ..core.lens import resolve
        from ..core.mathutil import BODY_TO_OPTICAL, rot_zyx
        self.p = ctx.sample()
        eng = ctx.engine
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
        from ..core.mathutil import BODY_TO_OPTICAL
        u, v, _, _ = self.intr[ci].project_dirs((Q - pos) @ (BODY_TO_OPTICAL @ R.T).T)
        return np.stack([u, v], -1)

    def laid(self, bi, b):
        from ..core.mathutil import rot_z
        return (self.W[bi] - self.ctr[bi]) @ rot_z(b[2]).T + self.ctr[bi] + np.array([b[0], b[1], 0.0])

    def shapes(self):
        return {c.uid: (it.height, it.width) for c, it in zip(self.cams, self.intr)}

    def capture(self, rng, placed=True, noise=PX):
        """True poses (the plan off by up to 1° and 10 mm), true board placements (drawn from the prior when placed)
        and the corners with noise: ({uid: [BoardObs]}, [(camera, board, uv)], truth, boards (nb, 3))."""
        from ..core.mathutil import rot_zyx
        nb = len(self.targets)
        truth = []
        for c, P in zip(self.cams, self.P):
            d = rng.uniform(-1, 1, 3)
            truth.append((rot_zyx(c.yaw + d[0], c.pitch + d[1], c.roll + d[2]), P + rng.uniform(-0.01, 0.01, 3)))
        boards = (np.c_[rng.normal(0, SB_MM / 1000, nb), rng.normal(0, SB_MM / 1000, nb),
                        rng.normal(0, math.radians(SB_DEG), nb)] if placed else np.zeros((nb, 3)))
        from ..core.install import BoardObs
        obs, data = {}, []
        for ci, bi in self.seen:
            uv = self.uv(ci, *truth[ci], self.laid(bi, boards[bi])) + rng.normal(0, noise, (len(self.W[bi]), 2))
            rays, _ = self.intr[ci].unproject(uv[:, 0], uv[:, 1])
            t = self.targets[bi]
            obs.setdefault(self.cams[ci].uid, []).append(BoardObs(t.uid, t.name, self.W[bi], rays, uv, uv.copy(),
                                                                  np.zeros((4, 2))))
            data.append((ci, bi, uv))
        return obs, data, truth, boards

    def own_solves(self, obs):
        """Each camera's own PnP (the boards where the plan puts them): where the installation check starts."""
        from ..core.install import solve_pose
        sh = self.shapes()
        return {c.uid: solve_pose(self.p, c, obs[c.uid], sh[c.uid])[:2] for c in self.cams if c.uid in obs}


class _Reference:
    """The joint problem stated independently: camera i at (R_i exp(w_i), p_i + t_i) about a reference pose, board b
    moved by (dx, dy, dyaw) about its planned centre; corners weighted 1/PX, boards by their prior, the Huber loss
    beyond HUBER on every residual; solved by Gauss–Newton (iteratively reweighted, central-difference Jacobian) to
    convergence."""

    def __init__(self, rig, data, ref_poses, placed=True):
        self.rig, self.data, self.ref, self.placed = rig, data, ref_poses, placed
        self.nc, self.nb = len(rig.cams), (len(rig.targets) if placed else 0)
        self.prior = np.tile([1000 / SB_MM, 1000 / SB_MM, 1 / math.radians(SB_DEG)], self.nb)

    def x_of(self, poses, boards):
        x = np.zeros(6 * self.nc + 3 * self.nb)
        for ci, ((R0, p0), (R, p)) in enumerate(zip(self.ref, poses)):
            x[6 * ci:6 * ci + 6] = np.r_[_log(R0.T @ np.asarray(R, float)), np.asarray(p, float) - p0]
        if self.placed:
            x[6 * self.nc:] = np.asarray(boards, float).ravel()
        return x

    def resid(self, x):
        out = []
        for ci, bi, uv in self.data:
            R0, p0 = self.ref[ci]
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

    def cost(self, x):
        a = np.abs(self.resid(x))
        return float(np.sum(np.where(a <= HUBER, a ** 2, 2 * HUBER * a - HUBER ** 2)))

    def solve(self, x):
        x = np.array(x, float)
        for _ in range(40):
            r = self.resid(x)
            w = np.sqrt(np.minimum(1.0, HUBER / np.maximum(np.abs(r), 1e-12)))
            J = self.jac(x) * w[:, None]
            step = np.linalg.lstsq(J, -r * w, rcond=None)[0]
            x = x + step
            if np.max(np.abs(step)) < 1e-9:
                break
        J = self.jac(x)
        return x, np.linalg.inv(J.T @ J)


def _joint_x(ref, rig, jr):
    poses = [jr.poses[c.uid] for c in rig.cams]
    fix = {b.uid: b for b in jr.boards}
    boards = np.array([[fix[t.uid].dx_mm / 1000, fix[t.uid].dy_mm / 1000, math.radians(fix[t.uid].dyaw_deg)]
                       if t.uid in fix else [0.0, 0.0, 0.0] for t in rig.targets])
    return ref.x_of(poses, boards)


@check('joint.reference_optimum', 'joint', 'The joint solve reaches the optimum of its least-squares problem',
       'largest difference, in units of each parameter\'s 1σ, between the installation check\'s joint solve '
       '(joint.joint_solve from each camera\'s own PnP) and an independent Gauss–Newton solve of the same Huber '
       'problem, for simulated captures of the sample\'s six cameras and eight floor boards (corners with 0.5 px '
       'noise, boards laid off their plan by the placement prior, cameras off theirs by up to 1° and 10 mm); the '
       'Huber cost must not be higher than the reference\'s',
       'the same problem written out from scratch (own parameterisation, IRLS Gauss–Newton, central differences), '
       'solved to convergence', 0.01, 'σ', refs=('triggs2000', 'huber1964'), cost='study')
def _joint_ref(ctx):
    from ..core.joint import joint_solve
    rig = _Rig(ctx)
    rng = ctx.rng('joint.reference_optimum')
    n = 1 if ctx.quick else 3
    worst, costs, rows = 0.0, [], []
    for trial in range(n):
        obs, data, truth, boards = rig.capture(rng)
        jr = joint_solve(rig.p, rig.cams, obs, rig.own_solves(obs), rig.shapes(), SB_MM, SB_DEG, pixel_sigma=PX)
        if not jr.ok or len(jr.poses) != len(rig.cams):
            raise ValueError(f'the joint solve did not solve every camera: {jr.message}')
        ref = _Reference(rig, data, truth)
        xr, C = ref.solve(ref.x_of(truth, boards))
        xj = _joint_x(ref, rig, jr)
        sd = np.sqrt(np.diag(C))
        e = float(np.max(np.abs(xj - xr) / sd))
        dc = ref.cost(xj) - ref.cost(xr)
        rows.append({'max_sigma': e, 'cost_above_reference': dc, 'rejected': len(jr.rejected)})
        worst = max(worst, e)
        costs.append(dc)
    return Outcome(worst, {'trials': rows, 'cameras': len(rig.cams), 'boards': len(rig.targets),
                           'observations': len(rig.seen)}, ok=max(costs) < 1e-3)


@check('joint.exact_recovery', 'joint', 'The joint solve recovers known poses from exact corners',
       'largest error of the camera poses the joint solve gives from noise-free corners of the sample\'s boards, '
       'boards where planned, every camera mounted off its plan by up to 1° and 10 mm, in degrees (rotation) and '
       'millimetres (position)',
       'the injected poses the corners were rendered from', 1e-3, 'degrees or mm', refs=('triggs2000',),
       cost='study')
def _joint_exact(ctx):
    from ..core.joint import joint_solve
    rig = _Rig(ctx)
    rng = ctx.rng('joint.exact_recovery')
    obs, data, truth, boards = rig.capture(rng, placed=False, noise=0.0)
    jr = joint_solve(rig.p, rig.cams, obs, rig.own_solves(obs), rig.shapes(), pixel_sigma=PX, solve_boards=False)
    rot, pos = [], []
    for c, (R, p) in zip(rig.cams, truth):
        Rj, pj = jr.poses[c.uid]
        rot.append(math.degrees(np.linalg.norm(_log(np.asarray(R).T @ np.asarray(Rj)))))
        pos.append(float(np.linalg.norm(np.asarray(pj) - p)) * 1000)
    return Outcome(max(max(rot), max(pos)), {'rotation_error_deg': [round(x, 9) for x in rot],
                                             'position_error_mm': [round(x, 9) for x in pos]})


# ============================================================================ observability
def _ob_settings(**kw):
    from ..core import observability as OB
    d = {'fusion': False, 'lidar_pairs': False, 'suggest': False}
    d.update(kw)
    return OB.ObservabilitySettings.from_dict(d)


def _band(n):
    """4.5 standard errors of log σ̂ for a σ estimated from n trials (1/√(2(n−1))), as a log ratio: symmetric, so an
    empirical σ above or below the prediction is held to the same number of standard errors."""
    return 4.5 / math.sqrt(2.0 * max(n - 1, 1))


def _mahalanobis_ok(groups, n):
    """Every group's (a camera's, a pair's) mean squared Mahalanobis distance within 4 standard deviations of its
    degrees of freedom (a χ²_k mean over n trials has standard deviation √(2k/n), per axis √(2/(kn)))."""
    worst = 0.0
    for g in groups:
        for s in g.values():
            if math.isfinite(s['mahalanobis2']):
                worst = max(worst, abs(s['mahalanobis2'] / s['dof'] - 1.0) / (4.0 * math.sqrt(2.0 / (s['dof'] * n))))
    return worst <= 1.0, worst


def _ratios(groups):
    return [a['ratio'] for g in groups for s in g.values() for a in s['axes'].values()]


def _mc_figure(groups, title):
    pred, emp, labels = [], [], []
    for g in groups:
        for name, s in g.items():
            for k, a in s['axes'].items():
                if a.get('predicted') and a.get('empirical'):
                    pred.append(a['predicted'])
                    emp.append(a['empirical'])
                    labels.append(f'{name} {k}')
    if not pred:
        return None
    return figure(title, 'predicted 1σ (deg or mm)', 'Monte Carlo 1σ (deg or mm)',
                  [series('axes', pred, emp, kind='points')], diag=True)


@check('observability.cameras', 'observability', 'Predicted camera σ against the joint solve\'s spread',
       'largest |log(empirical σ / predicted σ)| over every camera\'s and camera pair\'s six axes, the predicted σ of '
       'the observability study (Fisher information through the lens model) against the spread of the joint solve\'s '
       'answers to 200 simulated captures (the sample\'s six cameras, boards exact, 0.4 px corner noise); the '
       'tolerance is 4.5 standard errors of a σ estimated from the trials, and each camera\'s and pair\'s mean '
       'squared Mahalanobis distance must lie within 4 standard deviations of its degrees of freedom',
       'Monte Carlo solves with core.joint.joint_solve, the machinery that solves real captures', 0.2255, 'log ratio',
       refs=('kay1993', 'huber1967'), cost='study')
def _ob_cams(ctx):
    from ..core import observability as OB
    p = ctx.sample()
    res = OB.run(p, ctx.engine, _ob_settings(boards_placed=False, pixel_sigma=0.4))
    n = 60 if ctx.quick else 200
    mc = OB.monte_carlo(res, trials=n, seed=ctx.settings.seed % 100000, parts=('cameras',))
    c = mc['cameras']
    r = np.array(_ratios((c['cameras'], c['pairs'])))
    band = _band(c['solved'])
    maha, mw = _mahalanobis_ok((c['cameras'], c['pairs']), c['solved'])
    return Outcome(float(np.max(np.abs(np.log(r)))), {
        'trials': n, 'solved': c['solved'], 'stalled': c['stalled'], 'median_ratio': float(np.median(r)),
        'min_ratio': float(r.min()), 'max_ratio': float(r.max()), 'band_log': band,
        'mahalanobis_worst_in_bands': mw, 'module_agreement': bool(mc['agree']),
        'cameras': {k: {'mahalanobis2_per_dof': v['mahalanobis2'] / v['dof']} for k, v in c['cameras'].items()}},
        _mc_figure((c['cameras'], c['pairs']), 'Camera pose 1σ: Monte Carlo against the prediction'),
        ok=maha and c['stalled'] == 0, tolerance=band)


@check('observability.boards_placed', 'observability', 'Predicted σ with boards laid by hand',
       'largest |log(empirical σ / predicted σ)| over cameras F and ML and their pair, boards laid off their plan by '
       'draws of the placement σ (the joint solve\'s answers carried to the optimum of its least squares), 100 '
       'trials, tolerance 4.5 standard errors',
       'Monte Carlo solves with core.joint.joint_solve', 0.3198, 'log ratio', refs=('kay1993', 'huber1967'),
       cost='study')
def _ob_placed(ctx):
    from ..core import observability as OB
    p = ctx.sample()
    for cam in p.variant.cameras:
        cam.visible = cam.tag in ('F', 'ML')
    res = OB.run(p, ctx.engine, _ob_settings(boards_placed=True))
    n = 30 if ctx.quick else 100
    mc = OB.monte_carlo(res, trials=n, seed=(ctx.settings.seed + 1) % 100000, parts=('cameras',))
    c = mc['cameras']
    opt = c['optimum']
    r = np.array(_ratios((opt['cameras'], opt['pairs'])))
    band = _band(c['solved'])
    maha, mw = _mahalanobis_ok((opt['cameras'], opt['pairs']), c['solved'])
    return Outcome(float(np.max(np.abs(np.log(r)))), {'trials': n, 'solved': c['solved'],
                                                      'median_ratio': float(np.median(r)), 'band_log': band,
                                                      'mahalanobis_worst_in_bands': mw,
                                                      'module_agreement': bool(mc['agree'])},
                   _mc_figure((opt['cameras'], opt['pairs']),
                              'Boards laid by hand: Monte Carlo against the prediction'),
                   ok=maha and c['solved'] >= n - 2, tolerance=band)


@check('observability.radar', 'observability', 'Radar pose σ from corner reflectors',
       'relative difference of the radar\'s x σ from σ_r / √n and of its (y, yaw) covariance from the inverse of the '
       'range–azimuth Fisher information Σ [[1/r², 1/r], [1/r, 1]] / σ_az² for three reflectors straight ahead; the '
       'Monte Carlo spread of a least-squares solve must agree (ratios 0.8–1.25 over 200 trials)',
       'the Fisher information of range and azimuth measurements, written out in the check; Monte Carlo',
       1e-4, 'relative', refs=('kay1993', 'knott2004'), cost='study')
def _ob_radar(ctx):
    from ..core import observability as OB
    p = ctx.sample()
    v = next(v for v in p.variants if v.sensors)
    p.active_uid = v.uid
    rd1 = next(s for s in v.sensors if s.tag == 'RD1')
    rs = [5.0, 9.0, 16.0]
    refl = [{'name': f'CR{i + 1}', 'x': rd1.x + r, 'y': rd1.y, 'z': rd1.z, 'edge_mm': 150.0} for i, r in enumerate(rs)]
    st = _ob_settings(reflectors=refl, reflector_sigma_mm=0.0, radar_range_sigma_m=0.1, radar_az_sigma_deg=0.3)
    Q = np.array([[r['x'], r['y'], r['z']] for r in refl])
    C = OB._radar_cov(rd1, Q, st)
    sa = math.radians(0.3)
    I = sum(np.array([[1 / r ** 2, 1 / r], [1 / r, 1.0]]) for r in rs) / sa ** 2
    Cy = np.linalg.inv(I)
    e = max(abs(math.sqrt(C[0, 0]) - 0.1 / math.sqrt(3)) / (0.1 / math.sqrt(3)),
            float(np.max(np.abs(C[1:, 1:] - Cy) / np.abs(Cy))))
    res = OB.run(p, ctx.engine, st)
    mc = OB.monte_carlo(res, trials=80 if ctx.quick else 200, seed=(ctx.settings.seed + 3) % 100000,
                        parts=('radar',))
    ratios = [a['ratio'] for a in mc['radar']['radars']['RD1']['axes'].values()]
    return Outcome(e, {'x_sigma_m': math.sqrt(C[0, 0]), 'mc_ratios': [round(x, 3) for x in ratios],
                       'module_agreement': bool(mc['agree'])},
                   ok=all(0.8 < x < 1.25 for x in ratios))


@check('observability.lidar_pair', 'observability', 'Predicted lidar–camera σ against the calibration\'s spread',
       'largest |log(empirical σ / predicted σ)| over the six axes of the data rig\'s camera F – lidar L1 pair, the '
       'study\'s prediction against the spread of the lidar–camera calibration\'s own solve on simulated noisy '
       'corners and lidar points of the planned board poses, 60 trials, tolerance 4.5 standard errors',
       'Monte Carlo solves with core.lidar_calib._solve', 0.4142, 'log ratio', refs=('kay1993', 'zhang2004'),
       cost='study')
def _ob_lidar(ctx):
    from ..core import observability as OB
    p = ctx.sample()
    v = next(v for v in p.variants if v.sensors)
    p.active_uid = v.uid
    for c in v.cameras:
        c.visible = c.tag == 'F'
    for s in v.sensors:
        s.visible = s.tag == 'L1'
    res = OB.run(p, ctx.engine, _ob_settings(lidar_pairs=True))
    if len(res.lidar_pairs) != 1:
        raise ValueError('the F–L1 pair was not planned')
    n = 20 if ctx.quick else 60
    mc = OB.monte_carlo(res, trials=n, seed=(ctx.settings.seed + 2) % 100000, parts=('lidar',))
    s = mc['lidar']['pairs']['F–L1']
    r = np.array([a['ratio'] for a in s['axes'].values()])
    band = _band(n)
    maha, mw = _mahalanobis_ok((mc['lidar']['pairs'],), n)
    return Outcome(float(np.max(np.abs(np.log(r)))), {'trials': n, 'ratios': r.round(3).tolist(), 'band_log': band,
                                                      'mahalanobis_worst_in_bands': mw,
                                                      'module_agreement': bool(mc['agree']),
                                                      'predicted': res.lidar_pairs[0]['sigma']},
                   _mc_figure((mc['lidar']['pairs'],), 'Lidar–camera 1σ: Monte Carlo against the prediction'),
                   ok=maha, tolerance=band)


# ============================================================================ lidar–camera calibration
@check('lidar_calib.closed_form', 'lidar_calib', 'Plane-to-plane alignment recovers a known transform',
       'largest element difference between the transform the closed-form plane alignment gives for four exact board '
       'planes (seen by the camera and, through a known T_camera_lidar, by the lidar) and that T; two planes must '
       'not fix it',
       'the transform the planes were generated with', 1e-9, 'matrix element', refs=('unnikrishnan2005', 'zhang2004'))
def _lc_closed(ctx):
    from ..core import lidar_calib as LC
    T_true = LC._T(LC._exp([0.02, -0.05, 0.3]), np.array([0.3, -1.2, 2.1]))
    rng = ctx.rng('lidar_calib.closed_form')
    obs = []
    for n_dir in ([0.2, 0.1, -1.0], [-0.5, 0.0, -1.0], [0.1, 0.6, -1.0], [0.4, -0.4, -1.0]):
        n = np.asarray(n_dir) / np.linalg.norm(n_dir)
        c = -n * rng.uniform(4, 8) + rng.normal(0, 0.5, 3)
        u = np.cross(n, [0, 1.0, 0])
        u /= np.linalg.norm(u)
        R = np.stack([u, np.cross(n, u), n], 1)
        cam = LC.CamBoard(np.zeros((1, 2)), np.zeros((1, 3)), R, c, 0.1, np.eye(6))
        Pc = c + (rng.uniform(-0.4, 0.4, (200, 1)) * R[:, 0] + rng.uniform(-0.3, 0.3, (200, 1)) * R[:, 1])
        Pl = (Pc - T_true[:3, 3]) @ T_true[:3, :3]
        nl, dl = LC._fit_plane(Pl)
        lid = LC.LidarBoard(Pl, nl, dl, 0.0, np.zeros((0, 3)), np.zeros((0, 3)), 0)
        obs.append(LC.PoseObs('p', cam=cam, lid=lid))
    T = LC._closed_form(obs)
    return Outcome(float(np.abs(T - T_true).max()), {'planes': 4}, ok=LC._closed_form(obs[:2]) is None)


@check('lidar_calib.recovery', 'lidar_calib', 'An injected lidar mounting error is recovered',
       'position error (mm) of the lidar L1 the lidar–camera calibration recovers from synthetic captures (camera '
       'images through the real lens, lidar scans with range noise) of its six planned board poses with camera F, '
       'L1 mounted 15, −10, 8 mm and 0.6°, −0.4°, 0.3° off its plan; the rotation error must be under 0.08° and the '
       'error within 3 of the solve\'s own 1σ',
       'the injected mounting error the captures were simulated with', 3.0, 'mm',
       refs=('zhang2004', 'unnikrishnan2005'), cost='study')
def _lc_recovery(ctx):
    from ..core import lidar_calib as LC
    p = ctx.sample()
    p.active_uid = next(v for v in p.variants if v.sensors).uid
    err = dict(err_x_mm=15.0, err_y_mm=-10.0, err_z_mm=8.0, err_yaw_deg=0.6, err_pitch_deg=-0.4, err_roll_deg=0.3)
    st = LC.LidarCalibSettings.from_dict(dict(err, camera='F', lidar='L1', seed=4))
    cam = p.variant.by_tag('F')
    lid = next(s for s in p.variant.sensors if s.tag == 'L1')
    poses, _ = LC.plan_poses(p, cam, lid, st, ctx.engine)
    res = LC.run(p, ctx.engine, st, poses=poses)
    e = LC._delta(cam, res.T, LC.plan_T(cam, LC.true_lidar(lid, st)))
    pos = float(e['distance_mm'])
    sg = res.sigma
    within = pos <= 3 * max(float(sg['position_mm']), 1e-9) and e['rotation_deg'] <= 3 * max(sg['angle_deg'], 1e-9)
    return Outcome(pos, {'position_error_mm': pos, 'rotation_error_deg': float(e['rotation_deg']),
                         'sigma_position_mm': float(sg['position_mm']), 'sigma_angle_deg': float(sg['angle_deg']),
                         'poses_used': len(res.used)},
                   ok=e['rotation_deg'] < 0.08 and bool(within))
