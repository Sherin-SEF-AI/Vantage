"""Installation check: find the floor calibration boards in each camera's image, solve the installed pose
and compare it with the plan.

Each board is cut out of the fisheye image as a rectified pinhole view (exact lens model), the chessboard
corners are found there with OpenCV, turned back into bearing rays in the camera's optical frame and fed to
PnP together with the boards' known floor positions.
"""
from __future__ import annotations

import math
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


@dataclass
class BoardObs:
    target_uid: str
    name: str
    world: np.ndarray            # (N,3) inner corners on the floor
    rays: np.ndarray             # (N,3) unit bearings in the optical frame
    image_pts: np.ndarray        # (N,2) corner positions in the captured image (pixels of that image)
    predicted: np.ndarray        # (N,2) where the plan put them (same image pixels)
    outline: np.ndarray          # (4,2) predicted board outline in image pixels


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


def _find(patch, pattern):
    import cv2
    flags = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY | cv2.CALIB_CB_NORMALIZE_IMAGE
    for pat in (pattern, (pattern[1], pattern[0])):
        ok, corners = cv2.findChessboardCornersSB(patch, pat, flags=flags)
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
            pred_patch = (pr[:, :2] / pr[:, 2:3] * f + n / 2).reshape(t.rows - 1, t.cols - 1, 2)
            m = _match_grid(det, pred_patch)
            if m is not None and m[0] <= n * 0.15:
                break
            m = None
        if m is None:
            continue
        det = m[1].reshape(-1, 2)
        rv = np.c_[(det - n / 2) / f, np.ones(len(det))]
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
    res.n_corners = int(sum(len(o.world) for o in obs))
    if len(obs) < st.min_boards:
        res.status = 'serious'
        res.message = 'No calibration board found. Check that the planned boards are on the floor and in view.'
        return res
    try:
        R, pos, rms = solve_pose(project, cam, obs, img.shape)
        # the boards were cut out where the plan predicts them; with a large mounting error that clips them.
        # Detect again around the solved pose, drop boards that disagree with the rest and solve again.
        for _ in range(2):
            again = {o.target_uid: o for o in detect_boards(img, project, cam, pose=(R, pos))}
            have = {o.target_uid for o in obs}
            obs = [again.get(o.target_uid, o) for o in obs] + [o for u, o in again.items() if u not in have]
            obs = _reject_outliers(project, cam, obs, img.shape, (R, pos))
            R, pos, rms = solve_pose(project, cam, obs, img.shape, init=(R, pos))
        res.boards = obs
        res.n_corners = int(sum(len(o.world) for o in obs))
    except Exception as e:
        res.status = 'serious'
        res.message = f'Pose solve failed: {e}'
        return res
    yaw, pitch, roll = euler_zyx(R)
    res.pos, res.yaw, res.pitch, res.roll, res.rms_px = pos, yaw, pitch, roll, rms
    res.dpos_mm = (pos - cam.pos()) * 1000
    res.dang_deg = np.array([wrap180(yaw - cam.yaw), pitch - cam.pitch, wrap180(roll - cam.roll)])
    worst_p = float(np.max(np.abs(res.dpos_mm)))
    worst_a = float(np.max(np.abs(res.dang_deg)))
    if worst_p <= st.pos_tol_mm and worst_a <= st.ang_tol_deg:
        res.status = 'good'
        res.message = 'Within tolerance'
    elif worst_p <= 3 * st.pos_tol_mm and worst_a <= 3 * st.ang_tol_deg:
        res.status = 'warning'
        res.message = f'Off by {worst_p:.0f} mm / {worst_a:.1f}°: adjust the bracket or keep it as built'
    else:
        res.status = 'serious'
        res.message = f'Off by {worst_p:.0f} mm / {worst_a:.1f}°: re-mount'
    if len(obs) == 1:
        res.message += ' · one board only (less certain)'
    return res
