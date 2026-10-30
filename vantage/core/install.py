"""Installation check: find the floor calibration boards in each camera's image, solve the installed pose
and compare it with the plan.

Each board is cut out of the fisheye image as a rectified pinhole view (exact lens model), the chessboard
corners are found there with OpenCV, turned back into bearing rays in the camera's optical frame and fed to
PnP together with the boards' known floor positions.

When glare, a half-hidden board or a large mounting error defeats the detection, the board corners can be picked by
hand instead (solve_from_picks): four or more picked corners give the whole pose, two or three the rotation with the
position as planned, and the picked pose then seeds the automatic detection again.
"""
from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field

import numpy as np

from .lens import resolve, Intrinsics
from .mathutil import rot_zyx, euler_zyx, BODY_TO_OPTICAL, wrap180
from .project import Project, Camera, Target

PATCH = 640


@dataclass
class InstallSettings:
    pos_tol_mm: float = 10.0
    ang_tol_deg: float = 1.0
    min_boards: int = 1
    joint: bool = True                 # solve all cameras together with the board placement (core.joint)
    board_sigma_mm: float = 20.0       # how far a board may plausibly lie from its planned place
    board_sigma_deg: float = 1.0


@dataclass
class BoardObs:
    target_uid: str
    name: str
    world: np.ndarray            # (N,3) inner corners on the floor
    rays: np.ndarray             # (N,3) unit bearings in the optical frame
    image_pts: np.ndarray        # (N,2) corner positions in the captured image (pixels of that image)
    predicted: np.ndarray        # (N,2) where the plan put them (same image pixels)
    outline: np.ndarray          # (4,2) predicted board outline in image pixels
    picked: bool = False         # corners picked by hand (board outline corners), not detected


@dataclass
class PoseResult:
    uid: str
    tag: str
    boards: list = field(default_factory=list)
    n_corners: int = 0
    rms_px: float = float('nan')
    pos: np.ndarray | None = None
    yaw: float = float('nan')
    pitch: float = float('nan')
    roll: float = float('nan')
    dpos_mm: np.ndarray | None = None
    dang_deg: np.ndarray | None = None
    status: str = 'none'          # good | warning | serious | none
    message: str = ''
    single: tuple | None = None   # (R, pos) of the camera's own solve when the joint solve replaced it
    joint: bool = False
    detected: list = field(default_factory=list)   # every board found, incl. ones the camera's own solve set aside
    rot_err_deg: float = float('nan')                # angle of the rotation from the planned to the measured pose
    method: str = 'detected'      # detected | picked+detected (picks seeded the detection) | picked | picked-rotation
    picked: int = 0               # picked corners the solve used
    pick_err_px: list = field(default_factory=list)  # [(target uid, corner, px)] each pick's distance to the solve
    img_px: float = 1.0           # image pixels per sensor pixel, at least 1 (how far off a pick may be)


def inner_corners(t: Target) -> np.ndarray:
    """World coordinates (rows-1, cols-1, 3) of a board's inner corners, row-major from the (0, 0) corner."""
    w, h = t.size_m
    sq = t.square_mm / 1000
    c, s = math.cos(math.radians(t.yaw)), math.sin(math.radians(t.yaw))
    i = np.arange(1, t.cols)
    j = np.arange(1, t.rows)
    U, V = np.meshgrid(i * sq - w / 2, j * sq - h / 2)
    x = t.x + U * c - V * s
    y = t.y + U * s + V * c
    return np.stack([x, y, np.zeros_like(x)], -1)


def board_outline(t: Target) -> np.ndarray:
    w, h = t.size_m
    c, s = math.cos(math.radians(t.yaw)), math.sin(math.radians(t.yaw))
    pts = [(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)]
    return np.array([[t.x + u * c - v * s, t.y + u * s + v * c, 0.0] for u, v in pts])


def _w2o(R):
    return BODY_TO_OPTICAL @ R.T


def _scale(intr, img_shape):
    h, w = img_shape[:2]
    return w / intr.width, h / intr.height


def _project(intr, R, pos, P, sx, sy):
    d = (np.asarray(P, float) - pos) @ _w2o(R).T
    u, v, th, ok = intr.project_dirs(d)
    return np.stack([u * sx, v * sy], -1), ok


def _virtual_basis(c):
    z = c / np.linalg.norm(c)
    x = np.array([1.0, 0, 0]) - z * z[0]
    if np.linalg.norm(x) < 1e-3:
        x = np.array([0, 1.0, 0]) - z * z[1]
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.stack([x, y, z], 1)        # columns: virtual axes in the optical frame


def rectify(img, intr, rays_world_dirs, sx, sy, n=PATCH, fill=0.72):
    """Pinhole patch centred on the given optical-frame rays, the board filling `fill` of it. Returns (patch, Rv, f, n)."""
    import cv2
    c = rays_world_dirs.mean(axis=0)
    Rv = _virtual_basis(c)
    loc = rays_world_dirs @ Rv           # rays in virtual coordinates
    half = np.max(np.abs(loc[:, :2] / loc[:, 2:3]))
    f = (n / 2) * fill / max(half, 1e-3)
    xs = (np.arange(n) + 0.5 - n / 2) / f
    X, Y = np.meshgrid(xs, xs)
    rv = np.stack([X, Y, np.ones_like(X)], -1).reshape(-1, 3)
    ro = rv @ Rv.T
    u, v, th, ok = intr.project_dirs(ro)
    mx = np.where(ok, u * sx - 0.5, -1).reshape(n, n).astype(np.float32)
    my = np.where(ok, v * sy - 0.5, -1).reshape(n, n).astype(np.float32)
    patch = cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return patch, Rv, f, n


_warm = threading.Lock()
_warmed = False


def _warm_up():
    """OpenCV's SB chessboard detector returns slightly different corners on its very first call in a process than on
    every later one (same image, same flags), which made a check depend on what ran before it. One throw-away call on a
    synthetic board makes every real call repeatable."""
    global _warmed
    with _warm:
        if _warmed:
            return
        import cv2
        sq = 24
        b = ((np.indices((7 * sq, 7 * sq)) // sq).sum(0) % 2 * 200 + 30).astype(np.uint8)
        img = np.full((7 * sq + 40, 7 * sq + 40), 230, np.uint8)
        img[20:-20, 20:-20] = b
        flags = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY | cv2.CALIB_CB_NORMALIZE_IMAGE
        cv2.setRNGSeed(SB_SEED)                     # its result is thrown away, but no unseeded draw anywhere
        cv2.findChessboardCornersSB(cv2.GaussianBlur(img, (3, 3), 0.8), (6, 6), flags=flags)
        _warmed = True


SB_SEED = 0x5EED
_sb_lock = threading.Lock()


def find_sb(gray, pattern, flags):
    """cv2.findChessboardCornersSB, repeatable. The detector draws on OpenCV's global random generator, so with
    OpenCV 5 the same image gave other corners (at worst a wrong grid) depending on how many random numbers earlier
    calls had used: each call starts the generator from the same seed, one call at a time (the seed is global)."""
    import cv2
    if not _warmed:
        _warm_up()
    with _sb_lock:
        cv2.setRNGSeed(SB_SEED)
        return cv2.findChessboardCornersSB(gray, pattern, flags=flags)


def _find(patch, pattern):
    import cv2
    flags = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY | cv2.CALIB_CB_NORMALIZE_IMAGE
    for pat in (pattern, (pattern[1], pattern[0])):
        ok, corners = find_sb(patch, pat, flags)
        if ok:
            return corners.reshape(pat[1], pat[0], 2)
        ok, corners = cv2.findChessboardCorners(patch, pat, flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
        if ok:
            corners = cv2.cornerSubPix(patch, corners, (5, 5), (-1, -1),
                                       (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.01))
            return corners.reshape(pat[1], pat[0], 2)
    return None


def _match_grid(det, pred):
    """Reorder the detected (r,c,2) grid to the predicted (R,C,2) grid over the 8 grid symmetries."""
    best = None
    cands = []
    g = det
    for k in range(4):
        r = np.rot90(g, k)
        cands += [r, r[:, ::-1]]
    for cnd in cands:
        if cnd.shape != pred.shape:
            continue
        err = float(np.mean(np.linalg.norm(cnd - pred, axis=-1)))
        if best is None or err < best[0]:
            best = (err, cnd)
    return best


def detect_boards(img, project: Project, cam: Camera, targets=None, pose=None) -> list[BoardObs]:
    """Find every board the camera should see. img: 2-D uint8 (any resolution of the sensor's aspect).
    pose: (R_body_to_world, position) to predict the boards with; the planned pose when None."""
    import cv2
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    intr = resolve(project.module_for(cam), project.lens_for(cam))
    sx, sy = _scale(intr, img.shape)
    R0, p0 = (rot_zyx(cam.yaw, cam.pitch, cam.roll), cam.pos()) if pose is None else \
        (np.asarray(pose[0], float), np.asarray(pose[1], float))
    W2O = _w2o(R0)
    out = []
    for t in (project.targets if targets is None else targets):
        W = inner_corners(t)
        flat = W.reshape(-1, 3)
        pred_img, ok = _project(intr, R0, p0, flat, sx, sy)
        outl = board_outline(t)
        outl_img, ok2 = _project(intr, R0, p0, outl, sx, sy)
        if not (ok.all() and ok2.all()):
            continue
        rays = (np.vstack([flat, outl]) - p0) @ W2O.T
        rays /= np.linalg.norm(rays, axis=1, keepdims=True)
        if (rays[:, 2] < 0.05).any():
            continue
        # a tight patch first (best corner accuracy); a wider one finds a board a large mounting error moved
        m = None
        for fill in (0.72, 0.45):
            patch, Rv, f, n = rectify(img, intr, rays, sx, sy, fill=fill)
            det = _find(patch, (t.cols - 1, t.rows - 1))
            if det is None:
                continue
            pr = rays[:len(flat)] @ Rv
            # patch pixel j samples the ray (j + 0.5 - n/2) / f; OpenCV puts pixel centres on integers
            pred_patch = (pr[:, :2] / pr[:, 2:3] * f + n / 2 - 0.5).reshape(t.rows - 1, t.cols - 1, 2)
            m = _match_grid(det, pred_patch)
            if m is not None and m[0] <= n * 0.15:
                break
            m = None
        if m is None:
            continue
        det = m[1].reshape(-1, 2)
        rv = np.c_[(det - n / 2 + 0.5) / f, np.ones(len(det))]
        ro = rv @ Rv.T
        ro /= np.linalg.norm(ro, axis=1, keepdims=True)
        u, v, th, okd = intr.project_dirs(ro)
        out.append(BoardObs(t.uid, t.name, flat, ro, np.stack([u * sx, v * sy], -1), pred_img, outl_img))
    return out


def solve_pose(project: Project, cam: Camera, obs: list[BoardObs], img_shape, init=None):
    """Installed pose from board observations: (R_body_to_world, position, rms reprojection px).
    init: (R, position) starting guess; the planned pose when None."""
    import cv2
    W = np.concatenate([o.world for o in obs])
    rays = np.concatenate([o.rays for o in obs])
    img_pts = np.concatenate([o.image_pts for o in obs])
    norm = rays[:, :2] / rays[:, 2:3]
    R0, p0 = (rot_zyx(cam.yaw, cam.pitch, cam.roll), cam.pos()) if init is None else \
        (np.asarray(init[0], float), np.asarray(init[1], float))
    W2O = _w2o(R0)
    rvec0, _ = cv2.Rodrigues(W2O)
    tvec0 = (-W2O @ p0).reshape(3, 1)
    K = np.eye(3)
    ok, rvec, tvec = cv2.solvePnP(W.astype(np.float64), norm.astype(np.float64), K, None, rvec0.copy(), tvec0.copy(),
                                  useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        raise RuntimeError('PnP did not converge')
    rvec, tvec = cv2.solvePnPRefineLM(W.astype(np.float64), norm.astype(np.float64), K, None, rvec, tvec)
    Rw2o, _ = cv2.Rodrigues(rvec)
    pos = (-Rw2o.T @ tvec).ravel()
    R = (BODY_TO_OPTICAL.T @ Rw2o).T
    intr = resolve(project.module_for(cam), project.lens_for(cam))
    sx, sy = _scale(intr, img_shape)
    rep, ok = _project(intr, R, pos, W, sx, sy)
    rms = float(np.sqrt(np.mean(np.sum((rep - img_pts) ** 2, axis=1))))
    return R, pos, rms


def board_rms(project, cam, obs: list[BoardObs], img_shape, pose) -> list[float]:
    intr = resolve(project.module_for(cam), project.lens_for(cam))
    sx, sy = _scale(intr, img_shape)
    out = []
    for o in obs:
        rep, ok = _project(intr, pose[0], pose[1], o.world, sx, sy)
        out.append(float(np.sqrt(np.mean(np.sum((rep - o.image_pts) ** 2, axis=1)))))
    return out


def _reject_outliers(project, cam, obs, img_shape, pose):
    """Keep the largest set of boards that agree on one pose (a clipped, mis-detected or moved board does not).
    Candidates are the joint pose and each board's own pose. A board agrees with a pose when it reprojects nearly as
    well as under its own best pose: corners of a board far out in the fisheye are noisier, so the bar is per board."""
    if len(obs) <= 1:
        return list(obs)
    intr = resolve(project.module_for(cam), project.lens_for(cam))
    px = max(0.5, img_shape[1] / intr.width)
    cands, own = [pose], []
    for k, o in enumerate(obs):
        try:
            P = solve_pose(project, cam, [o], img_shape, init=pose)[:2]
            cands.append(P)
            own.append(board_rms(project, cam, [o], img_shape, P)[0])
        except Exception:
            own.append(math.inf)
    tol = [1.25 * r + 0.35 * px for r in own]
    best = None
    for P in cands:
        r = board_rms(project, cam, obs, img_shape, P)
        inl = [i for i, v in enumerate(r) if v <= tol[i]]
        key = (len(inl), -sum(r[i] for i in inl))
        if best is None or key > best[0]:
            best = (key, inl)
    keep = best[1] or list(range(len(obs)))
    return [obs[i] for i in keep]


def check_camera(project: Project, cam: Camera, img, st: InstallSettings | None = None) -> PoseResult:
    st = st or InstallSettings()
    res = PoseResult(cam.uid, cam.tag)
    try:
        obs = detect_boards(img, project, cam)
    except Exception as e:      # malformed image etc.
        res.message = f'Could not read the image: {e}'
        return res
    res.boards = obs
    res.detected = list(obs)
    res.n_corners = int(sum(len(o.world) for o in obs))
    if len(obs) < st.min_boards:
        res.status = 'serious'
        res.message = 'No calibration board found. Check that the planned boards are on the floor and in view.'
        return res
    try:
        R, pos, rms = solve_pose(project, cam, obs, img.shape)
        R, pos, rms = _refine(project, cam, img, obs, R, pos, res)
    except Exception as e:
        res.status = 'serious'
        res.message = f'Pose solve failed: {e}'
        return res
    _grade(res, cam, R, pos, rms, st)
    return res


def _refine(project, cam, img, obs, R, pos, res, keep=()):
    """The boards were cut out where the plan predicts them; with a large mounting error that clips them. Detect again
    around the solved pose, drop boards that disagree with the rest and solve again (twice). keep: picked boards, kept
    in every solve until the detection finds the board; they stay out of the vote on outliers, where a board of three
    picked corners would fit its own pose exactly and win. Fills res.boards, res.detected and res.n_corners."""
    for _ in range(2):
        again = {o.target_uid: o for o in detect_boards(img, project, cam, pose=(R, pos))}
        base = [o for o in obs if not o.picked]
        have = {o.target_uid for o in base}
        merged = [again.get(o.target_uid, o) for o in base] + [o for u, o in again.items() if u not in have]
        # keep every board for the joint solve: one that disagrees here may just lie off its planned place
        res.detected = [o for o in res.detected if o.target_uid not in {m.target_uid for m in merged}] + merged
        obs = _reject_outliers(project, cam, merged, img.shape, (R, pos)) if merged else []
        obs += [o for o in keep if o.target_uid not in {b.target_uid for b in merged}]
        R, pos, rms = solve_pose(project, cam, obs, img.shape, init=(R, pos))
    res.boards = obs
    res.n_corners = int(sum(len(o.world) for o in obs))
    return R, pos, rms


# ----------------------------------------------------------------------------- corners picked by hand
def planned_corners(project: Project, cam: Camera, img_shape, targets=None, pose=None) -> list:
    """Where each board's four outline corners fall in the image under the plan (or pose): [(target uid, name,
    (4,2) image pixels, (4,) in view)], for the boards with at least one corner in view. The corners are numbered as
    board_outline numbers them, which is how picks name them."""
    intr = resolve(project.module_for(cam), project.lens_for(cam))
    sx, sy = _scale(intr, img_shape)
    R0, p0 = (rot_zyx(cam.yaw, cam.pitch, cam.roll), cam.pos()) if pose is None else \
        (np.asarray(pose[0], float), np.asarray(pose[1], float))
    out = []
    for t in (project.targets if targets is None else targets):
        uv, ok = _project(intr, R0, p0, board_outline(t), sx, sy)
        if ok.any():
            out.append((t.uid, t.name, uv, ok))
    return out


def picked_obs(project: Project, cam: Camera, picks: dict, img_shape) -> list[BoardObs]:
    """Board observations from picked corners. picks: {target uid: {corner index 0-3: (u, v) image pixels}}."""
    intr = resolve(project.module_for(cam), project.lens_for(cam))
    sx, sy = _scale(intr, img_shape)
    R0, p0 = rot_zyx(cam.yaw, cam.pitch, cam.roll), cam.pos()
    targets = {t.uid: t for t in project.targets}
    out = []
    for tu, corners in picks.items():
        t = targets.get(tu)
        ks = sorted(int(k) for k in corners if 0 <= int(k) < 4) if t is not None else []
        if not ks:
            continue
        uv = np.array([corners[k] if k in corners else corners[str(k)] for k in ks], float).reshape(-1, 2)
        rays, _ = intr.unproject(uv[:, 0] / sx, uv[:, 1] / sy)
        rays = rays / np.linalg.norm(rays, axis=1, keepdims=True)
        world = board_outline(t)[ks]
        outl, _ = _project(intr, R0, p0, board_outline(t), sx, sy)
        pred, _ = _project(intr, R0, p0, world, sx, sy)
        out.append(BoardObs(t.uid, t.name, world, rays, uv, pred, outl, picked=True))
    return out


def picks_to_rows(project: Project, picks: dict) -> list:
    """{target uid: {corner: (u, v)}} -> [{'board': name, 'corner': k, 'x': u, 'y': v}] (the report's and
    --install-picks' form; boards by name, corners 0-3 as board_outline numbers them)."""
    names = {t.uid: t.name for t in project.targets}
    return [{'board': names[tu], 'corner': int(k), 'x': round(float(uv[0]), 2), 'y': round(float(uv[1]), 2)}
            for tu, cs in picks.items() if tu in names for k, uv in sorted(cs.items(), key=lambda kv: int(kv[0]))]


def picks_from_rows(project: Project, rows) -> tuple[dict, list]:
    """[{'board', 'corner', 'x', 'y'}] -> ({target uid: {corner: (u, v)}}, [what could not be used])."""
    by_name = {t.name.lower(): t.uid for t in project.targets}
    out, bad = {}, []
    for r in rows or []:
        try:
            tu = by_name.get(str(r['board']).lower())
            k = int(r['corner'])
            uv = (float(r['x']), float(r['y']))
        except (KeyError, TypeError, ValueError):
            bad.append(f'not a picked corner: {r!r}')
            continue
        if tu is None:
            bad.append(f'no board named {r["board"]!r} in the project')
        elif not 0 <= k < 4:
            bad.append(f'{r["board"]}: corner {k} is not 0-3')
        elif not all(math.isfinite(v) for v in uv):
            bad.append(f'{r["board"]}: corner {k} has no position')
        else:
            out.setdefault(tu, {})[k] = uv
    return out, bad


def _rotation_from_rays(cam: Camera, obs: list[BoardObs]):
    """Rotation only (Kabsch) with the camera at its planned position: (R_body_to_world, position)."""
    p0 = cam.pos()
    W = np.concatenate([o.world for o in obs]) - p0
    W /= np.linalg.norm(W, axis=1, keepdims=True)
    D = np.concatenate([o.rays for o in obs])
    U, _, Vt = np.linalg.svd(W.T @ D)                       # W2O maps world directions onto the rays
    S = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    W2O = Vt.T @ S @ U.T
    return (BODY_TO_OPTICAL.T @ W2O).T, p0


def _non_degenerate(obs) -> bool:
    """Four or more picked points that span the floor (not all on one line), enough for the whole pose."""
    W = np.concatenate([o.world for o in obs])[:, :2]
    if len(W) < 4:
        return False
    sv = np.linalg.svd(W - W.mean(axis=0), compute_uv=False)
    return sv[-1] > 0.05 * max(sv[0], 1e-9)


def solve_from_picks(project: Project, cam: Camera, img, picks: dict, st: InstallSettings | None = None) -> PoseResult:
    """The installed pose from board corners picked by hand, for a camera the detection fails on (glare, a board
    partly hidden, a mounting error too large for the planned cut-outs). Four or more picks that span the floor give
    the whole pose, two or three the rotation with the position as planned. The picked pose then seeds the automatic
    detection: boards found around it replace their picks (their corners are far more precise), the others' picks
    stay in the solve."""
    st = st or InstallSettings()
    res = PoseResult(cam.uid, cam.tag)
    obs = picked_obs(project, cam, picks, img.shape)
    n = int(sum(len(o.world) for o in obs))
    if n < 2:
        res.status = 'serious'
        res.message = 'Pick at least two board corners (four or more for the position too).'
        return res
    try:
        full = _non_degenerate(obs)
        if full:
            R, pos, rms = solve_pose(project, cam, obs, img.shape)
        else:
            R, pos = _rotation_from_rays(cam, obs)
        # boards found around the picked pose; with those also found around the plan's, which can cut a board out
        # better than a pose from a few clicks (or, under a large error, clip it): the second wins only when it
        # detects more boards and fits about as well
        seeded = detect_boards(img, project, cam, pose=(R, pos))
        both = seeded + [o for o in detect_boards(img, project, cam) if o.target_uid not in {f.target_uid for f in seeded}]
        px = max(0.5, img.shape[1] / resolve(project.module_for(cam), project.lens_for(cam)).width)
        best = None
        for found in [f for f in (seeded, both) if f][:1 if len(both) == len(seeded) else 2]:
            r = PoseResult(cam.uid, cam.tag, detected=list(found))
            keep = [o for o in obs if o.target_uid not in {f.target_uid for f in found}]
            P = solve_pose(project, cam, found + keep, img.shape, init=(R, pos))
            P = _refine(project, cam, img, found, P[0], P[1], r, keep=keep)
            n_det = sum(not o.picked for o in r.boards)
            if best is None or (n_det > best[0] and P[2] <= 2 * best[2][2] + 0.3 * px) or \
                    (n_det == best[0] and P[2] < best[2][2]):
                best = (n_det, r, P)
        if best is not None:
            _, r, (R, pos, rms) = best
            res.boards, res.detected, res.n_corners = r.boards, r.detected, r.n_corners
            res.method = 'picked+detected'
        else:
            res.boards, res.detected = obs, list(obs)
            res.n_corners = n
            if full:
                res.method = 'picked'
            else:
                res.method = 'picked-rotation'
                intr = resolve(project.module_for(cam), project.lens_for(cam))
                sx, sy = _scale(intr, img.shape)
                W = np.concatenate([o.world for o in obs])
                uv = np.concatenate([o.image_pts for o in obs])
                rep, _ = _project(intr, R, pos, W, sx, sy)
                rms = float(np.sqrt(np.mean(np.sum((rep - uv) ** 2, axis=1))))
    except Exception as e:
        res.status = 'serious'
        res.message = f'Pose solve from the picked corners failed: {e}'
        return res
    res.picked = int(sum(len(o.world) for o in res.boards if o.picked))
    res.pick_err_px = _pick_errors(project, cam, obs, img.shape, (R, pos))
    res.img_px = max(1.0, img.shape[1] / resolve(project.module_for(cam), project.lens_for(cam)).width)
    _grade(res, cam, R, pos, rms, st)
    return res


def _method_note(res: PoseResult) -> str:
    """What the picked corners did, for the message ('' for a camera solved from detected boards alone)."""
    if res.method == 'detected':
        return ''
    n = sum(1 for _ in res.pick_err_px)
    note = {'picked+detected': ' · picked corners found the boards',
            'picked': f' · from {n} picked corners (less certain)',
            'picked-rotation': f' · rotation from {n} picked corners, position as planned (pick four or more for it)'
            }[res.method]
    worst = max((e for *_, e in res.pick_err_px), default=0.0)
    if worst > 15 * res.img_px:
        note += f' · a pick is {worst:.0f} px off the solve: check it is the right corner'
    return note


def _pick_errors(project, cam, obs, img_shape, pose) -> list:
    intr = resolve(project.module_for(cam), project.lens_for(cam))
    sx, sy = _scale(intr, img_shape)
    targets = {t.uid: t for t in project.targets}
    out = []
    for o in obs:
        outl = board_outline(targets[o.target_uid])
        for w, uv in zip(o.world, o.image_pts):
            k = int(np.argmin(np.linalg.norm(outl - w, axis=1)))
            rep, _ = _project(intr, pose[0], pose[1], w[None], sx, sy)
            out.append((o.target_uid, k, float(np.linalg.norm(rep[0] - uv))))
    return out


def _grade(res: PoseResult, cam: Camera, R, pos, rms, st: InstallSettings):
    """Fill the measured pose, the deltas against the plan and the verdict."""
    yaw, pitch, roll = euler_zyx(R)
    res.pos, res.yaw, res.pitch, res.roll, res.rms_px = pos, yaw, pitch, roll, rms
    res.dpos_mm = (pos - cam.pos()) * 1000
    res.dang_deg = np.array([wrap180(yaw - cam.yaw), pitch - cam.pitch, wrap180(roll - cam.roll)])
    # judge the angle by the rotation between plan and measurement: Euler differences blow up as pitch nears 90°
    dR = rot_zyx(cam.yaw, cam.pitch, cam.roll).T @ np.asarray(R, float)
    res.rot_err_deg = math.degrees(math.acos(max(-1.0, min(1.0, (float(np.trace(dR)) - 1) / 2))))
    worst_p = float(np.max(np.abs(res.dpos_mm)))
    worst_a = res.rot_err_deg
    if worst_p <= st.pos_tol_mm and worst_a <= st.ang_tol_deg:
        res.status = 'good'
        res.message = 'Within tolerance'
    elif worst_p <= 3 * st.pos_tol_mm and worst_a <= 3 * st.ang_tol_deg:
        res.status = 'warning'
        res.message = f'Off by {worst_p:.0f} mm / {worst_a:.1f}°: adjust the bracket or keep it as built'
    else:
        res.status = 'serious'
        res.message = f'Off by {worst_p:.0f} mm / {worst_a:.1f}°: re-mount'
    if len(res.boards) == 1 and res.method in ('detected', 'picked+detected'):
        res.message += ' · one board only (less certain)'
    res.message += _method_note(res)


def check_all(project: Project, cams, images: dict, st: InstallSettings | None = None, progress=None, picks=None):
    """Check every camera that has an image: ({uid: PoseResult}, JointResult or None).
    With st.joint the per-camera poses are refined together with the board placement (core.joint).
    picks: {camera uid: {target uid: {corner: (u, v)}}}: a camera with picks is solved from them (solve_from_picks)."""
    st = st or InstallSettings()
    results = {}
    cams = [c for c in cams if c.uid in images]
    for i, c in enumerate(cams):
        if (picks or {}).get(c.uid):
            r = solve_from_picks(project, c, images[c.uid], picks[c.uid], st)
        else:
            r = check_camera(project, c, images[c.uid], st)
        r.img_w = images[c.uid].shape[1]
        results[c.uid] = r
        if progress:
            progress((i + 1) / (len(cams) + (1 if st.joint else 0)), f'Checked {c.tag}: {len(r.boards)} board(s)')
    jr = None
    # a rotation from two or three picks has no measured position to refine: it stays out of the joint solve
    solved = [c for c in cams if results[c.uid].pos is not None and results[c.uid].method != 'picked-rotation']
    if st.joint and len(solved) >= 2:
        from .joint import joint_solve
        obs = {c.uid: results[c.uid].detected or results[c.uid].boards for c in solved}
        init = {c.uid: (rot_zyx(results[c.uid].yaw, results[c.uid].pitch, results[c.uid].roll), results[c.uid].pos)
                for c in solved}
        shapes = {c.uid: images[c.uid].shape for c in solved}
        jr = joint_solve(project, solved, obs, init, shapes, st.board_sigma_mm, st.board_sigma_deg)
        if jr.poses:
            for c in solved:
                r = results[c.uid]
                R, pos = jr.poses[c.uid]
                r.single = init[c.uid]
                r.joint = True
                r.boards = jr.used.get(c.uid, r.boards)
                r.n_corners = int(sum(len(o.world) for o in r.boards))
                _grade(r, c, R, pos, jr.rms_px[c.uid], st)
        if progress:
            progress(1.0, jr.message)
    return results, jr

