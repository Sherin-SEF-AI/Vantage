"""Vehicle load cases: the body (and every camera on it) sits lower and pitches when loaded."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .mathutil import rot_y

# example values; replace with the suspension data of the vehicle
DEFAULT_CASES = [('Kerb (as designed)', 0.0, 0.0), ('Driver only', 8.0, 6.0), ('4 occupants', 18.0, 32.0),
                 ('Full load + boot', 22.0, 55.0)]


@dataclass
class LoadResult:
    name: str
    front_mm: float
    rear_mm: float
    pitch_deg: float
    blind_m2: float
    min_density: float
    multi_pct: float
    delta_blind: float
    ground_shift_cm: float      # worst displacement of the stitched ground image (calibrated at kerb weight)
    seam_ghost_cm: float        # worst disagreement between the two cameras at a seam
    status: str


def body_pose(front_mm, rear_mm, wheelbase):
    """Rigid body motion for axle drops (mm, positive = lower). Pitch positive = nose down, about the rear axle."""
    theta = math.atan2((front_mm - rear_mm) / 1000.0, max(wheelbase, 0.5))
    return rot_y(theta), np.array([0.0, 0.0, -rear_mm / 1000.0]), math.degrees(theta)


def cases_of(project):
    lc = project.analysis.load_cases
    return [tuple(c) for c in lc] if lc else list(DEFAULT_CASES)


def _reground(cam_pos, P_world, R, t):
    """Where a kerb-calibrated stitcher puts the true ground point P when the body moved by (R, t):
    the camera (fixed to the body) sees P along a body-frame ray that the stitcher intersects with z = 0."""
    Pb = (P_world - t) @ R
    d = Pb - cam_pos
    with np.errstate(divide='ignore', invalid='ignore'):
        s = -cam_pos[2] / d[:, 2]
    return cam_pos + d * s[:, None], s > 0


def evaluate(engine, project, cases=None, cell=None) -> list[LoadResult]:
    cases = cases or cases_of(project)
    wc = project.vehicle.wheel_centres
    wb = abs(wc['FL'][0] - wc['RL'][0]) if 'FL' in wc and 'RL' in wc else 2.6
    base = engine.analyse(project, cell=cell)
    cams = {c.uid: c for c in project.variant.cameras}
    g = base.grid
    # ground samples: each ROI cell with its best camera, and points along every seam seen by both cameras
    jj, ii = np.nonzero(g.roi & (base.best >= 0))
    Pc = np.stack([g.xs[ii], g.ys[jj], np.zeros(len(ii))], 1)
    best = base.best[jj, ii]
    seam_pts = []
    cx, cy = base.centre
    for sm in base.seams:
        a = math.radians(sm.angle)
        rr = np.arange(0.2, 8.0, g.cell)
        X, Y = cx + rr * math.cos(a), cy + rr * math.sin(a)
        ia = base.cam_index(sm.a)
        ib = base.cam_index(sm.b)
        for x, y in zip(X, Y):
            cc = g.cell_of(x, y)
            if cc and g.roi[cc[1], cc[0]] and base.cams[ia].vis[cc[1], cc[0]] and base.cams[ib].vis[cc[1], cc[0]]:
                seam_pts.append((x, y, sm.a, sm.b))
    out = []
    for name, f, r in cases:
        R, t, pitch = body_pose(float(f), float(r), wb)
        res = engine.analyse(project, cell=cell, body_pose=(R, t)) if (f or r) else base
        m = res.metrics
        shift = 0.0
        for k, ce in enumerate(base.cams):
            sel = best == k
            if not sel.any() or ce.uid not in cams:
                continue
            Q, ok = _reground(cams[ce.uid].pos(), Pc[sel], R, t)
            if ok.any():
                shift = max(shift, float(np.max(np.linalg.norm(Q[ok, :2] - Pc[sel][ok, :2], axis=1))))
        ghost = 0.0
        for x, y, ua, ub in seam_pts:
            P = np.array([[x, y, 0.0]])
            qa, oka = _reground(cams[ua].pos(), P, R, t)
            qb, okb = _reground(cams[ub].pos(), P, R, t)
            if oka[0] and okb[0]:
                ghost = max(ghost, float(np.linalg.norm(qa[0, :2] - qb[0, :2])))
        d = m['blind_m2'] - base.metrics['blind_m2']
        ghost_cm, shift_cm = ghost * 100, shift * 100
        status = 'good' if ghost_cm < 2 and d < 0.01 else ('warning' if ghost_cm < 5 and d < 0.05 else 'serious')
        out.append(LoadResult(name, float(f), float(r), pitch, m['blind_m2'], m['min_density'], m['multi_pct'], d,
                              shift_cm, ghost_cm, status))
    return out
