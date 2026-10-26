"""Vehicle load cases: the body (and every camera and sensor on it) sits lower, pitches and rolls when loaded.

Two kinds of case go through evaluate():
- typed drops, (name, front drop mm, rear drop mm) from analysis.load_cases (DEFAULT_CASES are made-up examples when
  none are set): the body pitches about the rear axle and drops by the rear figure; the cameras are calibrated at kerb.
- CornerCase, from the suspension (core.suspension): heave, pitch and roll of the body from a static solve of the load
  items on the four corners, with the pose the cameras were calibrated at (the calibration load) as the reference.

For each case the ground analysis is run again with the body moved (blind area, density), and two image figures say
what a stitcher calibrated at the reference load does with the moved body: the ground shift (worst displacement of a
ground point in the stitched view) and the seam ghost (worst disagreement of the two cameras at a seam). device_moves
gives every camera's and sensor's move relative to the road (mm and degrees), which the calibration has to absorb, and
horizon_moves how far the horizon moves in each camera's image (rows and degrees). The cases are evaluated on flat
ground whatever road core.terrain sets for the coverage analysis: their figures are about the body's attitude.

Pose convention: a body point p (vehicle frame at kerb) moves to R p + t. Drops are positive down. Pitch is positive
nose down (right-hand about +y, as a camera's pitch), roll positive right side down (right-hand about +x, ISO 8855).
The rotations are about axes through the rear-axle centre on the ground (the vehicle origin), so the drop of a point
on the ground plane at (x, y) is heave + x tan(pitch) - y tan(roll) to first order.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .mathutil import rot_x, rot_y, rot_zyx, euler_zyx, wrap180

# example values; replace with the suspension data of the vehicle (or use the suspension tab: core.suspension)
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
    ground_shift_cm: float      # worst displacement of the stitched ground image (calibrated at the reference load)
    seam_ghost_cm: float        # worst disagreement between the two cameras at a seam
    status: str
    roll_deg: float = 0.0       # positive = right side lower
    heave_mm: float = 0.0       # drop at the rear-axle centre
    corners: dict = field(default_factory=dict)    # 'FL' -> drop from kerb, mm
    devices: list = field(default_factory=list)    # device_moves() relative to the reference
    bump: list = field(default_factory=list)       # corners resting on their bump stop
    source: str = 'typed'       # typed | suspension
    reference: str = 'kerb'     # the load the cameras are calibrated at


@dataclass
class CornerCase:
    """A load case from the suspension: the body's attitude from kerb and the attitude the cameras are calibrated at."""
    name: str
    heave_mm: float = 0.0       # drop at the rear-axle centre from kerb, positive = lower
    pitch_deg: float = 0.0      # positive = nose down
    roll_deg: float = 0.0       # positive = right side lower
    corners: dict = field(default_factory=dict)    # 'FL' -> drop from kerb, mm
    reference: tuple | None = None                 # (heave_mm, pitch_deg, roll_deg) at calibration; None = kerb
    reference_name: str = 'kerb'
    bump: list = field(default_factory=list)


def pose(heave_mm, pitch_deg=0.0, roll_deg=0.0):
    """(R, t) of the body for a heave (drop at the rear-axle centre, mm), pitch and roll (degrees)."""
    return rot_zyx(0.0, float(pitch_deg), float(roll_deg)), np.array([0.0, 0.0, -float(heave_mm) / 1000.0])


def body_pose(front_mm, rear_mm, wheelbase, roll_deg=0.0):
    """Rigid body motion for axle drops (mm, positive = lower). Pitch positive = nose down, about the rear axle; an
    optional roll (degrees, positive = right side lower) about the vehicle's x axis on the ground. Returns
    (R, t, pitch_deg); with no roll exactly the two-drop pose of earlier versions."""
    theta = math.atan2((front_mm - rear_mm) / 1000.0, max(wheelbase, 0.5))
    R = rot_y(theta) if not roll_deg else rot_y(theta) @ rot_x(math.radians(float(roll_deg)))
    return R, np.array([0.0, 0.0, -rear_mm / 1000.0]), math.degrees(theta)


def plane_of_drops(drops: dict, xy: dict):
    """Least-squares plane through corner drops (mm) at corner positions (x, y in m): (heave_mm, pitch_deg, roll_deg,
    rms residual mm). Four drops of a rigid body lie on a plane (residual 0); typed ones may not."""
    keys = [k for k in drops if k in xy]
    if len(keys) < 3:
        raise ValueError('a plane needs the drops of at least three corners')
    A = np.array([[1.0, xy[k][0], xy[k][1]] for k in keys])
    d = np.array([float(drops[k]) for k in keys])
    sol, *_ = np.linalg.lstsq(A, d, rcond=None)
    w, b, c = (float(v) for v in sol)
    res = d - A @ sol
    return w, math.degrees(math.atan(b / 1000.0)), -math.degrees(math.atan(c / 1000.0)), float(np.sqrt(np.mean(res ** 2)))


def relative(R, t, R0, t0):
    """The motion that takes the body from pose (R0, t0) to pose (R, t), both from kerb: (dR, dt) with
    R p + t = dR (R0 p + t0) + dt."""
    dR = R @ R0.T
    return dR, t - dR @ t0


def turn_deg(R) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, (float(np.trace(R)) - 1.0) / 2.0))))


def device_moves(project, R, t, R0=None, t0=None, variant=None) -> list[dict]:
    """How every visible camera and sensor moves relative to the road between the reference pose (R0, t0; default
    kerb) and the pose (R, t): its position change (vehicle frame at the reference, mm) and the change of its own yaw,
    pitch and roll as the project states them (degrees), plus the body's whole rotation (turn_deg, the same for every
    device: the body is rigid, so the devices do not move relative to each other)."""
    from .sensors import rotation as sensor_rotation
    v = variant or project.variant
    R0 = np.eye(3) if R0 is None else np.asarray(R0, float)
    t0 = np.zeros(3) if t0 is None else np.asarray(t0, float)
    R, t = np.asarray(R, float), np.asarray(t, float)
    turn = turn_deg(R @ R0.T)
    devs = [('camera', c, rot_zyx(c.yaw, c.pitch, c.roll)) for c in v.cameras if c.visible]
    devs += [(s.kind, s, sensor_rotation(s)) for s in v.sensors if s.visible]
    out = []
    for kind, o, Rd in devs:
        p = o.pos()
        d = ((R @ p + t) - (R0 @ p + t0)) * 1000.0
        e0, e1 = euler_zyx(R0 @ Rd), euler_zyx(R @ Rd)
        out.append({'uid': o.uid, 'tag': o.tag, 'kind': kind, 'dx_mm': float(d[0]), 'dy_mm': float(d[1]),
                    'dz_mm': float(d[2]), 'move_mm': float(np.linalg.norm(d)),
                    'd_yaw_deg': wrap180(e1[0] - e0[0]), 'd_pitch_deg': e1[1] - e0[1],
                    'd_roll_deg': wrap180(e1[2] - e0[2]), 'turn_deg': turn})
    return out


def facing(R_cam) -> str:
    """Which way a camera looks, from its optical axis's heading: forward (within 45° of +x), rear, left or right."""
    d = np.asarray(R_cam, float)[:, 0]
    az = math.degrees(math.atan2(d[1], d[0]))
    if abs(az) <= 45.0:
        return 'forward'
    if abs(az) >= 135.0:
        return 'rear'
    return 'left' if az > 0 else 'right'


def horizon_at(intr, W2O, delta_deg=2.0):
    """Where the horizon (the level directions of the road plane) crosses a camera's image at its centre column:
    {'v': row px, 'elev_deg': its angle above the optical axis in the image's vertical plane, 'tilt_deg': the slope of
    the horizon there (positive = its right side lower in the image), 'in_view'}; None when the camera is rolled so
    far that its image columns are level (the horizon then runs along the centre column). W2O: world -> optical."""
    n = np.asarray(W2O, float) @ np.array([0.0, 0.0, 1.0])     # the world's up in the optical frame
    o = np.array([0.0, n[2], -n[1]])                            # level (o·n = 0) and in the centre column (x = 0)
    no = float(np.linalg.norm(o))
    if no < 1e-9:
        return None
    o /= no
    if o[2] < 0 or (o[2] == 0 and o[1] > 0):
        o = -o                                                  # the one in front of the camera
    u, v, _, ok = intr.project_dirs(o[None])
    side = np.cross(n / max(float(np.linalg.norm(n)), 1e-12), o)
    a = math.radians(delta_deg)
    pts = np.stack([o * math.cos(a) + side * math.sin(a), o * math.cos(a) - side * math.sin(a)])
    us, vs, _, _ = intr.project_dirs(pts)
    if us[1] < us[0]:
        us, vs = us[::-1], vs[::-1]
    tilt = math.degrees(math.atan2(float(vs[1] - vs[0]), float(us[1] - us[0])))
    return {'v': float(v[0]), 'elev_deg': math.degrees(math.atan2(-o[1], o[2])), 'tilt_deg': tilt,
            'in_view': bool(ok[0])}


def horizon_moves(project, R, R0=None, variant=None) -> list[dict]:
    """How the horizon moves in every visible camera's image when the body turns from the reference attitude R0
    (default level) to R (body -> world rotations; a heave or shift does not move the horizon): per camera 'uid',
    'tag', 'slot', 'facing', 'v_ref'/'v' (row px), 'dv_px' (positive = down the image), 'd_deg' (positive = the horizon
    rises in the view, as when the nose dips), 'd_tilt_deg' (the horizon's slope change), 'in_view' (at both)."""
    from .lens import resolve
    from .mathutil import BODY_TO_OPTICAL
    v = variant or project.variant
    R = np.asarray(R, float)
    R0 = np.eye(3) if R0 is None else np.asarray(R0, float)
    out = []
    for c in v.cameras:
        if not c.visible:
            continue
        intr = resolve(project.module_for(c), project.lens_for(c))
        Rc = rot_zyx(c.yaw, c.pitch, c.roll)
        a = horizon_at(intr, BODY_TO_OPTICAL @ (R0 @ Rc).T)
        b = horizon_at(intr, BODY_TO_OPTICAL @ (R @ Rc).T)
        if a is None or b is None:
            continue
        out.append({'uid': c.uid, 'tag': c.tag, 'slot': c.slot, 'facing': facing(Rc), 'v_ref': a['v'], 'v': b['v'],
                    'dv_px': b['v'] - a['v'], 'elev_ref_deg': a['elev_deg'], 'd_deg': b['elev_deg'] - a['elev_deg'],
                    'd_tilt_deg': wrap180(b['tilt_deg'] - a['tilt_deg']), 'in_view': a['in_view'] and b['in_view']})
    return out


def cases_of(project):
    lc = project.analysis.load_cases
    return [tuple(c) for c in lc] if lc else list(DEFAULT_CASES)


def _reground(cam_pos, P_world, R, t):
    """Where a stitcher calibrated with the camera at cam_pos puts the true ground point P when the body then moved by
    (R, t): the camera (fixed to the body) sees P along a body-frame ray that the stitcher intersects with z = 0."""
    Pb = (P_world - t) @ R
    d = Pb - cam_pos
    with np.errstate(divide='ignore', invalid='ignore'):
        s = -cam_pos[2] / d[:, 2]
    return cam_pos + d * s[:, None], s > 0


def _samples(base):
    """Ground samples of an analysis: each ROI cell with its best camera, and points along every seam seen by both
    cameras."""
    g = base.grid
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
    return Pc, best, seam_pts


def evaluate(engine, project, cases=None, cell=None, progress=None, cancel=None) -> list[LoadResult] | None:
    """Ground figures per case: typed (name, front mm, rear mm) tuples and CornerCase objects, in any mix. Returns
    None when cancelled, [] without a ground analysis (no vehicle or no region of interest)."""
    cases = cases or cases_of(project)
    wc = project.vehicle.wheel_centres
    wb = abs(wc['FL'][0] - wc['RL'][0]) if 'FL' in wc and 'RL' in wc else 2.6
    base = engine.analyse(project, cell=cell, terrain=False)
    if base is None:
        return []
    cams = {c.uid: c for c in project.variant.cameras}
    refs = {None: (base, np.eye(3), np.zeros(3), _samples(base))}

    def reference(key):
        if key not in refs:
            R0, t0 = pose(*key)
            ra = engine.analyse(project, cell=cell, body_pose=(R0, t0), terrain=False)
            refs[key] = (ra, R0, t0, _samples(ra))
        return refs[key]

    out = []
    for n, case in enumerate(cases):
        if cancel is not None and cancel():
            return None
        if isinstance(case, CornerCase):
            key = tuple(float(v) for v in case.reference) if case.reference is not None else None
            if key is not None and not any(key):
                key = None
            ref, R0, t0, (Pc, best, seam_pts) = reference(key)
            R, t = pose(case.heave_mm, case.pitch_deg, case.roll_deg)
            same = (case.heave_mm, case.pitch_deg, case.roll_deg) == (key or (0.0, 0.0, 0.0))
            res = ref if same else engine.analyse(project, cell=cell, body_pose=(R, t), terrain=False)
            dR, dt = relative(R, t, R0, t0)
            cam_pos = {u: R0 @ c.pos() + t0 for u, c in cams.items()}
            name, pitch = case.name, case.pitch_deg
            corners = {k: float(v) for k, v in case.corners.items()}
            f = (corners.get('FL', 0.0) + corners.get('FR', 0.0)) / 2
            r = (corners.get('RL', 0.0) + corners.get('RR', 0.0)) / 2
            extra = dict(roll_deg=case.roll_deg, heave_mm=case.heave_mm, corners=corners, bump=list(case.bump),
                         source='suspension', reference=case.reference_name)
        else:
            name, f, r = case
            f, r = float(f), float(r)
            ref, R0, t0, (Pc, best, seam_pts) = refs[None]
            R, t, pitch = body_pose(f, r, wb)
            res = engine.analyse(project, cell=cell, body_pose=(R, t), terrain=False) if (f or r) else base
            dR, dt = R, t
            cam_pos = {u: c.pos() for u, c in cams.items()}
            extra = dict(heave_mm=r, corners={'FL': f, 'FR': f, 'RL': r, 'RR': r})
        m = res.metrics
        shift = 0.0
        for k, ce in enumerate(ref.cams):
            sel = best == k
            if not sel.any() or ce.uid not in cams:
                continue
            Q, ok = _reground(cam_pos[ce.uid], Pc[sel], dR, dt)
            if ok.any():
                shift = max(shift, float(np.max(np.linalg.norm(Q[ok, :2] - Pc[sel][ok, :2], axis=1))))
        ghost = 0.0
        for x, y, ua, ub in seam_pts:
            P = np.array([[x, y, 0.0]])
            qa, oka = _reground(cam_pos[ua], P, dR, dt)
            qb, okb = _reground(cam_pos[ub], P, dR, dt)
            if oka[0] and okb[0]:
                ghost = max(ghost, float(np.linalg.norm(qa[0, :2] - qb[0, :2])))
        d = m['blind_m2'] - ref.metrics['blind_m2']
        ghost_cm, shift_cm = ghost * 100, shift * 100
        status = 'good' if ghost_cm < 2 and d < 0.01 else ('warning' if ghost_cm < 5 and d < 0.05 else 'serious')
        out.append(LoadResult(name, f, r, pitch, m['blind_m2'], m['min_density'], m['multi_pct'], d,
                              shift_cm, ghost_cm, status, devices=device_moves(project, R, t, R0, t0), **extra))
        if progress is not None:
            progress((n + 1) / len(cases))
    return out
