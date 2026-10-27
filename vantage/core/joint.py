"""Joint extrinsic solve: every camera's pose and every board's placement on the floor, in one least-squares problem.

Solving each camera on its own trusts the boards to lie exactly where the plan put them. On a real line they are
laid by hand, a centimetre or two off; each camera then absorbs the error of the boards it sees, and two cameras that
share a board disagree about where it is: the stitched view breaks at the seam. The joint solve lets each board move
on the floor (x, y and yaw, held near the plan by a prior of board_sigma) and asks every camera that sees it to agree.

The seam consistency it reports does not depend on the plan at all: for each corner seen by two or more cameras,
each camera's ray is intersected with the floor; the spread of those floor points is what a viewer of the stitched
image sees as a double or broken line.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .lens import resolve
from .mathutil import BODY_TO_OPTICAL, euler_zyx, rot_z
from .project import Project

HUBER_SCALE = 3.0       # corner residuals beyond 3 × pixel_sigma count linearly (a stray corner pulls less)
FD_STEP = 1e-6          # central-difference step of the Jacobian (rad, m)
BOARD_REACH = 10.0      # how many placement σ a board may move in the solve
MAX_BOARD_TURN_DEG = 30.0       # and never further round than this (a square grid repeats every 90°)


@dataclass
class BoardFix:
    uid: str
    name: str
    dx_mm: float
    dy_mm: float
    dyaw_deg: float
    cams: list                  # tags of the cameras that see it

    @property
    def shift_mm(self):
        return math.hypot(self.dx_mm, self.dy_mm)


@dataclass
class JointResult:
    poses: dict                              # uid -> (R_body_to_world, position)
    rms_px: dict                             # uid -> reprojection RMS with the moved boards
    boards: list = field(default_factory=list)       # [BoardFix]
    seam_before_mm: float = float('nan')     # median floor disagreement of shared corners, per-camera poses
    seam_after_mm: float = float('nan')
    seam_before_p90_mm: float = float('nan')     # 90th percentile
    seam_after_p90_mm: float = float('nan')
    shared_corners: int = 0
    message: str = ''
    ok: bool = False
    rejected: list = field(default_factory=list)     # [(camera tag, board name, rms px)] observations set aside
    used: dict = field(default_factory=dict)         # uid -> [BoardObs] the solve kept
    moved: list = field(default_factory=list)        # [BoardFix] beyond the placement tolerance, worst first


def _rotvec_to_matrix(w):
    th = float(np.linalg.norm(w))
    if th < 1e-12:
        return np.eye(3)
    k = w / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K


def floor_points(R, pos, rays):
    """Floor (z = 0) intersections of optical-frame rays from a camera at (R, pos); NaN for rays above the horizon."""
    W2O = BODY_TO_OPTICAL @ R.T
    d = rays @ W2O                         # optical -> world
    with np.errstate(divide='ignore', invalid='ignore'):
        t = np.where(d[:, 2] < -1e-6, -pos[2] / d[:, 2], np.nan)
    return pos[:2] + d[:, :2] * t[:, None]


def seam_spread(obs: dict, poses: dict):
    """(median, 90th percentile, n) floor disagreement in metres of corners seen by two or more cameras.
    obs: uid -> [BoardObs]; poses: uid -> (R, pos)."""
    pts = {}
    for uid, lst in obs.items():
        if uid not in poses:
            continue
        R, p = poses[uid]
        for o in lst:
            fp = floor_points(np.asarray(R, float), np.asarray(p, float), o.rays)
            for i, q in enumerate(fp):
                if np.isfinite(q).all():
                    pts.setdefault((o.target_uid, i), []).append(q)
    spreads = []
    for q in pts.values():
        if len(q) >= 2:
            Q = np.array(q)
            spreads.append(float(np.max(np.linalg.norm(Q[:, None] - Q[None], axis=-1))))
    if not spreads:
        return float('nan'), float('nan'), 0
    return float(np.median(spreads)), float(np.percentile(spreads, 90)), len(spreads)


def joint_solve(project: Project, cams: list, obs: dict, init: dict, img_shapes: dict, board_sigma_mm=20.0,
                board_sigma_deg=1.0, pixel_sigma=0.5, solve_boards=True) -> JointResult:
    """cams: cameras to solve; obs: uid -> [BoardObs]; init: uid -> (R, pos) starting poses (e.g. each camera's own
    solve); img_shapes: uid -> image shape the observations are in. A board observation that no placement of the board
    explains (a mis-detected grid) is set aside and listed in JointResult.rejected."""
    from scipy.optimize import least_squares
    cams = [c for c in cams if obs.get(c.uid) and c.uid in init]
    res = JointResult({}, {})
    if not cams:
        res.message = 'No camera has board observations.'
        return res
    targets = {t.uid: t for t in project.targets}
    seen = {o.target_uid for c in cams for o in obs[c.uid] if o.target_uid in targets}
    board_ids = [t.uid for t in project.targets if t.uid in seen]       # the project's order, never the uids'
    bidx = {u: i for i, u in enumerate(board_ids)}
    nc, nb = len(cams), len(board_ids) if solve_boards else 0
    W2O0, P0, intr, scale = [], [], [], []
    for c in cams:
        R, p = init[c.uid]
        W2O0.append(BODY_TO_OPTICAL @ np.asarray(R, float).T)
        P0.append(np.asarray(p, float))
        it = resolve(project.module_for(c), project.lens_for(c))
        intr.append(it)
        h, w = img_shapes[c.uid][:2]
        scale.append((w / it.width, h / it.height))
    # observation blocks: (camera index, board index, world corners (N,3), image points (N,2), BoardObs)
    allblocks = []
    for ci, c in enumerate(cams):
        for o in obs[c.uid]:
            if o.target_uid in bidx:
                allblocks.append((ci, bidx[o.target_uid], np.asarray(o.world, float), np.asarray(o.image_pts, float), o))
    centres = {u: np.array([targets[u].x, targets[u].y, 0.0]) for u in board_ids}
    # a tolerance of 0 (or junk) pins the boards to a micrometre and a ten-thousandth of a degree rather than failing
    sig_b = (board_sigma_mm if board_sigma_mm > 1e-3 else 1e-3) / 1000.0
    sig_y = math.radians(board_sigma_deg if board_sigma_deg > 1e-4 else 1e-4)

    def unpack(x):
        cp = x[:6 * nc].reshape(nc, 6)
        bp = x[6 * nc:].reshape(nb, 3) if nb else np.zeros((len(board_ids), 3))
        return cp, bp

    def board_world(bi, W, bp):
        dx, dy, dyaw = bp[bi]
        c = centres[board_ids[bi]]
        return (W - c) @ rot_z(dyaw).T + c + np.array([dx, dy, 0.0])

    def block_px(x, blk):
        cp, bp = unpack(x)
        ci, bi, W, uv, _ = blk
        Rw = _rotvec_to_matrix(cp[ci, :3]) @ W2O0[ci]
        d = (board_world(bi, W, bp) - (P0[ci] + cp[ci, 3:])) @ Rw.T
        u, v, _, _ = intr[ci].project_dirs(d)
        sx, sy = scale[ci]
        return np.stack([u * sx, v * sy], -1) - uv

    prior = np.tile([1 / sig_b, 1 / sig_b, 1 / sig_y], nb)
    # a board laid by hand lies within a few σ of its place: bounds keep the solve away from the placements a square
    # grid cannot tell apart (turned a quarter or half round), where a mis-matched grid in one camera would otherwise be
    # explained by turning the board and moving the other cameras that see it
    reach = np.tile([BOARD_REACH * sig_b, BOARD_REACH * sig_b, min(BOARD_REACH * sig_y, math.radians(MAX_BOARD_TURN_DEG))],
                    nb)
    bounds = (np.r_[np.full(6 * nc, -np.inf), -reach], np.r_[np.full(6 * nc, np.inf), reach])

    def solve(blocks, x0):
        n_par = 6 * nc + 3 * nb
        spans, r0 = [], 0
        for b in blocks:
            spans.append((r0, r0 + 2 * len(b[2])))
            r0 += 2 * len(b[2])
        n_res = r0 + 3 * nb

        def residuals(x):
            out = [(block_px(x, b) / pixel_sigma).ravel() for b in blocks]
            if nb:
                out.append(x[6 * nc:] * prior)
            return np.concatenate(out)

        def jacobian(x):
            # a block's pixels depend only on its camera's 6 and its board's 3 parameters: central differences of that
            # block alone, into a dense matrix (a few hundred rows by 6 per camera + 3 per board)
            J = np.zeros((n_res, n_par))
            for (a, e), b in zip(spans, blocks):
                ci, bi = b[0], b[1]
                cols = [*range(6 * ci, 6 * ci + 6), *(range(6 * nc + 3 * bi, 6 * nc + 3 * bi + 3) if nb else ())]
                for k in cols:
                    d = np.zeros(n_par)
                    d[k] = FD_STEP
                    J[a:e, k] = ((block_px(x + d, b) - block_px(x - d, b)) / (2 * FD_STEP * pixel_sigma)).ravel()
            if nb:
                J[r0:, 6 * nc:] = np.diag(prior)
            return J
        # The trust-region steps are solved exactly (SVD of the dense Jacobian). With a sparse Jacobian least_squares
        # solves them iteratively (LSMR), which barely moves along the shallow directions the board priors hold (every
        # camera and board shifting together): the solve then ran into max_nfev short of its optimum, its answers
        # 1.2–3.3× wider than the noise allows (round 6 Monte Carlo: 38 of 40 trials, a median χ² of 55 left; every
        # solve took all 600 evaluations). Plain least squares first: Gauss–Newton converges from far off; then the
        # Huber loss from there (a few steps) for the odd stray corner. Tight tolerances: some directions (a camera
        # that sees two boards side on) are shallow, and a loose stop leaves the answer depending on the start.
        common = dict(jac=jacobian, bounds=bounds, method='trf', tr_solver='exact', x_scale='jac', ftol=1e-12,
                      xtol=1e-12, gtol=1e-12)
        x1 = least_squares(residuals, x0, loss='linear', max_nfev=200, **common).x
        return least_squares(residuals, x1, loss='huber', f_scale=HUBER_SCALE, max_nfev=100, **common)

    blocks = list(allblocks)
    x = np.zeros(6 * nc + 3 * nb)
    rejected = []
    try:
        rounds = max(3, len(allblocks) // 3) + 1         # at most a third of the observations can be set aside
        for it in range(rounds):
            # after a rejection start again from the cameras' own solves: the previous answer was pulled by the outlier
            # and can sit in another minimum
            sol = solve(blocks, np.zeros_like(x) if rejected else x)
            x = sol.x
            if it == rounds - 1:
                break
            rms = [float(np.sqrt(np.mean(np.sum(block_px(x, b) ** 2, axis=1)))) for b in blocks]
            lim = max(3.0, 4.0 * float(np.median(rms)))
            # drop the worst offender only (the others may have been pulled off by it), but never a camera's last board
            order = [k for k in np.argsort(rms)[::-1] if rms[k] > lim and
                     sum(1 for b in blocks if b[0] == blocks[k][0]) > 1]
            if not order:
                break
            worst = int(order[0])
            rejected.append((cams[blocks[worst][0]].tag, blocks[worst][4].name, rms[worst]))
            blocks.pop(worst)
    except Exception as e:
        res.message = f'Joint solve failed: {e}'
        return res
    cp, bp = unpack(x)
    for ci, c in enumerate(cams):
        Rw = _rotvec_to_matrix(cp[ci, :3]) @ W2O0[ci]
        R = (BODY_TO_OPTICAL.T @ Rw).T
        res.poses[c.uid] = (R, P0[ci] + cp[ci, 3:])
        e = [np.linalg.norm(block_px(x, b), axis=1) for b in blocks if b[0] == ci]
        res.rms_px[c.uid] = float(np.sqrt(np.mean(np.concatenate(e) ** 2))) if e else float('nan')
    res.rejected = rejected
    kept = {}
    for ci, _, _, _, o in blocks:
        kept.setdefault(cams[ci].uid, []).append(o)
    res.used = kept
    tags = {c.uid: c.tag for c in cams}
    for bi, u in enumerate(board_ids):
        seen = [tags[cams[ci].uid] for ci, bj, _, _, _ in blocks if bj == bi]
        dx, dy, dyaw = (bp[bi] if nb else (0.0, 0.0, 0.0))
        res.boards.append(BoardFix(u, targets[u].name, float(dx * 1000), float(dy * 1000), float(math.degrees(dyaw)), seen))
    b_med, b_p90, n = seam_spread(kept, {c.uid: init[c.uid] for c in cams})
    a_med, a_p90, _ = seam_spread(kept, res.poses)
    res.seam_before_mm, res.seam_before_p90_mm = b_med * 1000, b_p90 * 1000
    res.seam_after_mm, res.seam_after_p90_mm = a_med * 1000, a_p90 * 1000
    res.shared_corners = n
    res.ok = sol.status >= 0
    # a board further out than its placement tolerance is worth re-laying; the estimate is good to about a centimetre
    moved = sorted([b for b in res.boards if b.shift_mm > board_sigma_mm or abs(b.dyaw_deg) > board_sigma_deg],
                   key=lambda b: -b.shift_mm)
    res.moved = moved
    res.message = (f'{nc} cameras, {len(board_ids)} boards, {n} shared corners'
                   + (f' · seams agree to {res.seam_after_mm:.1f} mm (per camera {res.seam_before_mm:.1f} mm)'
                      if n else ' · no board is seen by two cameras, so the cameras are not tied together')
                   + (' · off their planned place: ' + ', '.join(f'{b.name} {b.shift_mm:.0f} mm {b.dyaw_deg:+.1f}°'
                                                                  for b in moved[:3])
                      + (f' and {len(moved) - 3} more' if len(moved) > 3 else '') if moved else '')
                   + (f' · set aside: ' + ', '.join(f'{b} in {t}' for t, b, _ in rejected) if rejected else ''))
    return res


def pose_dict(R, pos) -> dict:
    y, pi, r = euler_zyx(np.asarray(R, float))
    return {'x': round(float(pos[0]), 4), 'y': round(float(pos[1]), 4), 'z': round(float(pos[2]), 4), 'yaw': round(y, 3),
            'pitch': round(pi, 3), 'roll': round(r, 3)}


def report(jr: JointResult) -> dict:
    """The joint solve's part of an installation report."""
    return {'seam_median_mm': {'own_solves': round(jr.seam_before_mm, 2), 'joint': round(jr.seam_after_mm, 2)},
            'seam_p90_mm': {'own_solves': round(jr.seam_before_p90_mm, 2), 'joint': round(jr.seam_after_p90_mm, 2)},
            'shared_corners': jr.shared_corners,
            'boards': [{'name': b.name, 'seen_by': b.cams, 'dx_mm': round(b.dx_mm, 1), 'dy_mm': round(b.dy_mm, 1),
                        'dyaw_deg': round(b.dyaw_deg, 3)} for b in jr.boards],
            'set_aside': [{'camera': t, 'board': b, 'rms_px': round(e, 2)} for t, b, e in jr.rejected],
            'message': jr.message}
