"""Knock detection on live camera streams: a camera that has been knocked (its mount slipped or bent) points somewhere
else from then on. Two light measurements per analysed frame, each a step test against the camera's own baseline.

1. Direction of travel. An analysed frame and the frame about pair_s before it: corners spread over the view (at most
   features_per_cell in each cell of a 10 x 8 grid, so one textured patch such as the road cannot decide alone),
   tracked with pyramidal Lucas-Kanade forward and backward (kept when the two agree within 0.5 px and the patch
   changed no more than the median one: glints and flicker go), give pairs of rays through the lens table (only rays
   within max_theta_deg of the axis). Driving straight, the camera moves without turning, so each pair of rays spans a
   plane through the direction of motion; the direction is found by RANSAC over pairs of these planes and refined by
   reweighted least squares (translation_from_rays). That direction is the vehicle's forward axis seen from the camera:
   a rotation of the camera moves it by the same angle about the two axes across it (a roll about the direction of
   travel does not move it). A pair is used only when the features moved enough (min_parallax_px: a standing vehicle
   has no direction) and at least min_inlier_share of the tracks fit one straight motion (turning adds a rotation that
   does not fit, and so does a view full of moving traffic). Why not the essential matrix: with the five-point solver
   (rotation free) the sideways direction trades off against a small yaw and scattered 3-15° pair to pair on the CARLA
   drives; fixing the rotation to none brought the scatter to 0.5-2°. Every camera sees the same direction of
   travel in the vehicle frame (through its planned pose), so with three cameras or more each camera is judged against
   the median of the others (RigConsensus): what moves them all, the body pitching under braking or load or a crab
   angle, cancels; a knocked camera alone departs from it. Without that, the end of a CARLA drive (the car braking for
   a pedestrian) read as a 0.9° knock on one camera.
2. The own body. A camera that sees part of the vehicle (a mirror camera its flank and wheel, a rear camera the
   bumper) sees it still while the world moves past. Over the first baseline_s the pixels whose image gradient keeps
   its direction and strength (the coherence |Σg|/Σ|g| of the gradient vectors over the frames at least
   body_coherence, and the mean gradient strong) are the body's edges: a moving scene's gradients point every way and
   cancel, a reflection on paint moves, the body's own outline stays. The lens circle's edge is left out (it does not
   move with the camera). Each later frame's gradient image is matched to them by masked normalised cross-correlation
   (sub-pixel peak), giving the body's image shift (dx, dy) in pixels, converted to degrees with the lens's local
   scale. It measures every rotation axis that moves the body edge in the image, but only for cameras with enough
   body edge in view (body_min_share).

The step test (StepDetector): the baseline is the median of the first baseline_s of estimates (at least 10), the scatter
its median absolute deviation (σ = 1.4826 MAD, at least 0.3° for the travel direction); the current value the median
of the last window_s (at least 5 estimates). A knock is a difference of at least threshold_deg (body_threshold_deg for
the body edge, whose scatter is a tenth of the travel direction's) that is at least z_min standard errors of the
window median (σ·1.2533/√n) and stays for hold_s. Medians make a single bad pair harmless; the hold makes a short
excursion (a pothole, a curve the gate let through) harmless. A knock stays flagged until the watchdog restarts or the
baseline is reset: the camera needs re-calibration.

Limits: the travel direction assumes straight driving between the two frames (the inlier gate), a static world for
most features (moving traffic is outliers to RANSAC) and, for the consensus, that most cameras were not knocked; it is
noisier at low speed and with little texture, and a camera's sideways axis is noisier when its view of the direction
of travel is near the edge of the image (the front and rear cameras of the CARLA drive scattered 2-6°, too much to
see 2°; the mirror and side cameras 0.3-1.3°). A rotation about the direction of travel itself does not move it. The
body method needs the body in view and its edges lit the same way as when learned (a low sun sweeping over the body
can move a shadow edge), and measures the body edge's image shift as one rigid 2-D shift, which over a fisheye's rim
reads 10-35 % low. Neither measures translation. This is a watchdog's alarm, not a calibration:
vantage.core.online_calib estimates the new extrinsics from a recorded drive.
"""
from __future__ import annotations

import collections
import math

import numpy as np

from .config import RadialLens


def _basis(ref):
    """Two unit vectors across ref (for angles about the axes across it)."""
    ref = ref / np.linalg.norm(ref)
    a = np.array([0.0, 1.0, 0.0]) if abs(ref[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = np.cross(a, ref)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(ref, e1)
    return ref, e1, e2


def angles_about(d, ref) -> np.ndarray:
    """(2,) degrees: the direction d relative to ref, about the two axes across ref."""
    r, e1, e2 = _basis(np.asarray(ref, float))
    d = np.asarray(d, float)
    return np.degrees([math.atan2(d @ e1, d @ r), math.atan2(d @ e2, d @ r)])


def translation_from_rays(r1, r2, thresh=2e-3, iters=200, seed=0):
    """The camera's displacement direction (unit, optical frame) between two frames of a camera that moved without
    turning, from matched unit rays (N, 3): each pair spans a plane through the direction (its normal n = r1 x r2), so
    the direction is perpendicular to every n. RANSAC over pairs of normals (drawn in proportion to their parallax),
    then iteratively reweighted least squares (Huber) on the inliers: the eigenvector of the smallest eigenvalue of
    sum w n nᵀ. The residual of a pair is the angle of r2 from the plane through the direction and r1 (radians),
    thresh its inlier bound. The sign is the one rays move away from (forward motion). Returns (d, inliers,
    inlier share, rms residual) or None."""
    r1, r2 = np.asarray(r1, float), np.asarray(r2, float)
    n = np.cross(r1, r2)
    nn = np.linalg.norm(n, axis=1)
    good = nn > 1e-9
    r1, r2, n, nn = r1[good], r2[good], n[good], nn[good]
    N = len(n)
    if N < 8:
        return None
    rng = np.random.default_rng(seed)

    def cross(a, b):                                     # np.cross without its axis handling (hot path)
        return np.stack([a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1], a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2],
                         a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]], -1)

    def resid(d):
        c = cross(d[..., None, :], r1)
        return np.abs(np.sum(r2 * c, -1)) / np.maximum(np.sqrt(np.sum(c * c, -1)), 1e-12)
    w = nn / nn.sum()
    pairs = rng.choice(N, (iters, 2), p=w)
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    D = cross(n[pairs[:, 0]], n[pairs[:, 1]])
    L = np.sqrt(np.sum(D * D, 1))
    D = D[L > 1e-12] / L[L > 1e-12, None]
    if not len(D):
        return None
    counts = (resid(D) < thresh).sum(1)                 # every hypothesis at once: (hypotheses, pairs)
    best = D[int(np.argmax(counts))]
    if best is None:
        return None
    d = best
    for _ in range(6):
        e = resid(d)
        h = np.where(e < thresh, 1.0, thresh / np.maximum(e, 1e-12)) * (e < 3 * thresh)
        _, V = np.linalg.eigh((n * h[:, None]).T @ n)
        d2 = V[:, 0]
        d = d2 if d2 @ d >= 0 else -d2
    # Gauss-Newton on the angular residuals themselves (the eigenvector minimises an algebraic error that weighs
    # pairs by their parallax), a step kept only when it lowers the robust cost
    def signed(d):
        c = cross(d[None], r1)
        return np.sum(r2 * c, -1) / np.maximum(np.sqrt(np.sum(c * c, -1)), 1e-12)

    def cost(r):                                       # Huber, flat beyond 3 thresh (outliers count the same)
        a = np.minimum(np.abs(r), 3 * thresh)
        return float(np.sum(np.where(a < thresh, 0.5 * a * a, thresh * (a - 0.5 * thresh))))
    r = signed(d)
    c0 = cost(r)
    for _ in range(8):
        _, e1, e2 = _basis(d)
        h = np.where(np.abs(r) < thresh, 1.0, thresh / np.maximum(np.abs(r), 1e-12)) * (np.abs(r) < 3 * thresh)
        eps = 1e-6
        J = np.stack([(signed((d + eps * e) / np.linalg.norm(d + eps * e)) - r) / eps for e in (e1, e2)], -1)
        try:
            step = -np.linalg.solve((J * h[:, None]).T @ J + 1e-12 * np.eye(2), (J * h[:, None]).T @ r)
        except np.linalg.LinAlgError:
            break
        for scale in (1.0, 0.5, 0.25):
            dn = d + scale * (step[0] * e1 + step[1] * e2)
            dn /= np.linalg.norm(dn)
            rn = signed(dn)
            cn = cost(rn)
            if cn < c0:
                break
        else:
            break
        moved = np.hypot(*step) * scale
        d, r, c0 = dn, rn, cn
        if moved < 1e-9:
            break
    e = resid(d)
    inl = e < thresh
    if inl.sum() < 3:
        return None
    a1 = np.arccos(np.clip(r1[inl] @ d, -1, 1))
    a2 = np.arccos(np.clip(r2[inl] @ d, -1, 1))
    if np.median(a2 - a1) < 0:
        d = -d
    return d, int(inl.sum()), float(inl.mean()), float(np.sqrt(np.mean(e[inl] ** 2)))


def bucket(p, shape, rows=8, cols=10, per=8):
    """At most per of the points p (N, 2), strongest first (the order goodFeaturesToTrack returns), in each cell of a
    rows x cols grid over the image: spread over the whole view so no one textured patch (a road surface) decides
    the direction alone."""
    h, w = shape
    cell = (np.minimum(p[:, 1] * rows // h, rows - 1) * cols + np.minimum(p[:, 0] * cols // w, cols - 1)).astype(int)
    order = np.argsort(cell, kind='stable')
    c = cell[order]
    first = np.searchsorted(c, c, side='left')
    rank = np.arange(len(c)) - first                    # 0 for a cell's strongest point, 1 for the next …
    return p[np.sort(order[rank < per])]


def encode_mask(mask) -> dict:
    """A boolean image as {'w', 'h', 'png'} (base64 PNG): how the configuration carries the planned body."""
    import base64
    import cv2
    ok, buf = cv2.imencode('.png', np.asarray(mask, np.uint8) * 255)
    return {'w': int(mask.shape[1]), 'h': int(mask.shape[0]), 'png': base64.b64encode(buf.tobytes()).decode('ascii')}


def decode_mask(d) -> np.ndarray | None:
    import base64
    import cv2
    try:
        img = cv2.imdecode(np.frombuffer(base64.b64decode(d['png']), np.uint8), cv2.IMREAD_GRAYSCALE)
    except Exception:                                  # noqa: BLE001 - a broken mask is no mask
        return None
    return None if img is None else img > 127


class StepDetector:
    """A persistent step in a 2-vector series (degrees), against the series' own baseline."""

    def __init__(self, baseline_s, window_s, hold_s, threshold, z_min, floor=0.02, min_base=10, min_window=5):
        self.baseline_s, self.window_s, self.hold_s = baseline_s, window_s, hold_s
        self.threshold, self.z_min, self.floor = threshold, z_min, floor
        self.min_base, self.min_window = int(min_base), int(min_window)   # estimates a baseline / a window needs
        self.reset()

    def reset(self):
        self.t0 = None
        self.base_vals = []
        self.baseline = None
        self.sigma = None
        self.recent = collections.deque(maxlen=400)
        self.since = None              # when the current exceedance began
        self.knock = None              # {'t', 'size', 'vector', 'z'} once confirmed
        self.last = None               # the latest evaluation

    def add(self, t, v) -> dict | None:
        v = np.asarray(v, float)
        if self.t0 is None:
            self.t0 = t
        if self.baseline is None:
            self.base_vals.append(v)
            if t - self.t0 >= self.baseline_s and len(self.base_vals) >= self.min_base:
                B = np.array(self.base_vals)
                self.baseline = np.median(B, 0)
                mad = np.median(np.abs(B - self.baseline), 0)
                self.sigma = np.maximum(1.4826 * mad, self.floor)
            return None
        self.recent.append((t, v))
        win = np.array([x for (tt, x) in self.recent if tt >= t - self.window_s])
        n = len(win)
        if n < self.min_window:
            return None
        m = np.median(win, 0)
        diff = m - self.baseline
        size = float(np.linalg.norm(diff))
        se = self.sigma * 1.2533 / math.sqrt(n)
        z = float(np.linalg.norm(diff / se))
        over = size >= self.threshold and z >= self.z_min
        if over:
            self.since = t if self.since is None else self.since
        else:
            self.since = None
        held = over and t - self.since >= self.hold_s
        if held and self.knock is None:
            self.knock = {'t': float(t), 'onset_t': float(self.since), 'size': size, 'vector': diff.tolist(), 'z': z}
        self.last = {'t': float(t), 'size': size, 'z': z, 'vector': diff.tolist(), 'n': n, 'over': over,
                     'held': held}
        return self.last

    @property
    def state(self) -> str:
        if self.baseline is None:
            return 'learning'
        return 'knock' if self.knock else ('step' if self.last and self.last['over'] else 'ok')


class BodyEdge:
    """The own body's static edges: learned, then matched (image shift in analysis pixels)."""

    def __init__(self, lens: RadialLens | None, st: dict, plan_dir=None, expected=None):
        self.lens, self.st = lens, st
        self.plan_dir = plan_dir        # the direction of travel (optical frame): edges along the flow are the world's
        self.expected = expected        # (h, w) bool: where the plan puts the body, or None
        self.n = 0
        self.sum = self.gx = self.gy = self.gsum = None
        self.ready = False
        self.absent = ''               # why there is no body estimate ('' while learning or when ready)
        self.size = None

    @staticmethod
    def _grad(g):
        import cv2
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        return cv2.magnitude(gx, gy)

    def learn(self, g):
        import cv2
        f = g.astype(np.float32)
        if self.sum is None:
            self.size = g.shape
            self.sum, self.gx, self.gy, self.gsum = (np.zeros_like(f) for _ in range(4))
        if g.shape != self.size:
            return
        gx = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3)
        self.sum += f
        self.gx += gx
        self.gy += gy
        self.gsum += cv2.magnitude(gx, gy)
        self.n += 1

    def finish(self, deg_per_px):
        import cv2
        if self.n < 5:
            self.absent = 'too few frames to learn the body'
            return
        h, w = self.size
        mean = self.sum / self.n
        gmean = self.gsum / self.n
        coh = np.hypot(self.gx, self.gy) / np.maximum(self.gsum, 1e-6)
        circle = np.ones((h, w), bool)
        if self.lens is not None:
            circle = self.lens.inside(w, h, self.lens.max_theta - math.radians(4.0))
        edge = (coh >= self.st['body_coherence']) & (gmean >= self.st['body_min_gradient']) & circle
        if self.plan_dir is not None and self.lens is not None:
            # driving straight, a world edge that runs along the flow (a lane line, a kerb, a far roof line) also
            # stays put in the image; the body's do not run along the flow in general: those along it are left out
            fl = self._flow_dirs(w, h)
            gxm, gym = self.gx / self.n, self.gy / self.n
            cosang = np.abs(gxm * fl[..., 0] + gym * fl[..., 1]) / np.maximum(np.hypot(gxm, gym), 1e-6)
            edge &= cosang >= self.st['body_flow_cos']
        if self.expected is not None:
            # near where the plan puts the body (the real body and the plan's differ by a little)
            ex = cv2.resize(self.expected.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
            r = int(math.ceil(self.st['body_plan_margin_deg'] / max(deg_per_px, 1e-6)))
            ex = cv2.dilate(ex, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
            edge &= ex > 0
        edge = cv2.morphologyEx(edge.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)) > 0
        share = float(edge.mean())
        self.share = share
        self.sum = self.gx = self.gy = self.gsum = None    # the learning buffers are not needed any more
        if share < self.st['body_min_share']:
            self.absent = f'no body in view (static edges cover {share * 100:.2f} % of the image)'
            return
        ys, xs = np.nonzero(edge)
        S = int(math.ceil(self.st['body_search_deg'] / max(deg_per_px, 1e-6)))
        S = max(3, min(S, 64))
        y0, y1 = max(0, ys.min() - 2), min(h, ys.max() + 3)
        x0, x1 = max(0, xs.min() - 2), min(w, xs.max() + 3)
        self.box = (x0, y0, x1, y1)
        self.S = S
        self.tmpl = self._grad(mean.astype(np.float32))[y0:y1, x0:x1].copy()
        m = cv2.dilate(edge.astype(np.uint8), np.ones((3, 3), np.uint8))[y0:y1, x0:x1]
        self.mask = m.astype(np.float32)
        th, tw = self.tmpl.shape
        if th >= 16 and tw >= 16 and S >= 4:
            self.tmpl_h = cv2.resize(self.tmpl, (tw // 2, th // 2), interpolation=cv2.INTER_AREA)
            mh = cv2.resize(self.mask, (tw // 2, th // 2), interpolation=cv2.INTER_AREA)
            self.mask_h = (mh > 0).astype(np.float32)
        else:
            self.tmpl_h = self.mask_h = None
        self.centre = ((x0 + x1) / 2.0, (y0 + y1) / 2.0)
        self.ready = True

    def _flow_dirs(self, w, h):
        """(h, w, 2) unit image directions in which a static point moves at each pixel when the camera moves along
        plan_dir (away from the focus of expansion), through the lens table."""
        uu, vv = np.meshgrid(np.arange(w, dtype=float), np.arange(h, dtype=float))
        r, th = self.lens.unproject(uu.ravel(), vv.ravel(), w, h)
        d = self.plan_dir / np.linalg.norm(self.plan_dir)
        away = (r @ d)[:, None] * r - d[None]                 # a ray moves away from d as the camera advances
        away /= np.maximum(np.linalg.norm(away, axis=1, keepdims=True), 1e-9)
        q = r + 1e-3 * away
        u2, v2, _ = self.lens.project(q, w, h)
        f = np.stack([u2 - uu.ravel(), v2 - vv.ravel()], -1)
        f /= np.maximum(np.linalg.norm(f, axis=1, keepdims=True), 1e-12)
        return np.nan_to_num(f).reshape(h, w, 2)

    def match(self, g):
        """(dx, dy, score) in analysis pixels, or None. Coarse to fine: masked normalised cross-correlation of the
        gradient images at half resolution over the whole search, then at full resolution within 2 px of the coarse
        peak, with a parabola through the peak for the sub-pixel part."""
        import cv2
        if not self.ready or g.shape != self.size:
            return None
        x0, y0, x1, y1 = self.box
        S = self.S
        pad = S + 1
        big = cv2.copyMakeBorder(g, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        crop = big[y0:y1 + 2 * pad, x0:x1 + 2 * pad].astype(np.float32)        # the box grown by S + 1 each way
        search = self._grad(crop)[1:-1, 1:-1]                                 # the box grown by S
        if self.tmpl_h is not None:
            sh = cv2.resize(search, (search.shape[1] // 2, search.shape[0] // 2), interpolation=cv2.INTER_AREA)
            if sh.shape[0] < self.tmpl_h.shape[0] or sh.shape[1] < self.tmpl_h.shape[1]:
                return None
            rh = np.nan_to_num(cv2.matchTemplate(sh, self.tmpl_h, cv2.TM_CCORR_NORMED, mask=self.mask_h))
            _, _, _, (cx, cy) = cv2.minMaxLoc(rh)
            ex, ey = 2 * cx, 2 * cy                                           # full-resolution offset of the peak
        else:
            ex, ey = S, S
        th, tw = self.tmpl.shape
        lo_x, lo_y = max(0, ex - 2), max(0, ey - 2)
        hi_x, hi_y = min(2 * S, ex + 2), min(2 * S, ey + 2)
        sub = search[lo_y:hi_y + th, lo_x:hi_x + tw]
        res = np.nan_to_num(cv2.matchTemplate(sub, self.tmpl, cv2.TM_CCORR_NORMED, mask=self.mask))
        _, best, _, (bx, by) = cv2.minMaxLoc(res)
        if best < self.st['body_min_score']:
            return None
        X, Y = lo_x + bx, lo_y + by                                           # offset in the full search
        if X in (0, 2 * S) or Y in (0, 2 * S):
            return None                               # at the edge of the search: beyond body_search_deg

        def sub_px(a, b, c):
            den = a - 2 * b + c
            return 0.0 if abs(den) < 1e-9 else float(np.clip(0.5 * (a - c) / den, -0.5, 0.5))
        fx = sub_px(res[by, bx - 1], res[by, bx], res[by, bx + 1]) if 0 < bx < res.shape[1] - 1 else 0.0
        fy = sub_px(res[by - 1, bx], res[by, bx], res[by + 1, bx]) if 0 < by < res.shape[0] - 1 else 0.0
        return X + fx - S, Y + fy - S, float(best)


class RigConsensus:
    """The direction of travel every camera of the rig sees, in the vehicle frame (angles about +x, degrees): each
    camera's median over the last window_s, and the median of those over the other cameras."""

    def __init__(self, window_s: float):
        import threading
        self.window_s = float(window_s)
        self.series = {}
        self.lock = threading.Lock()

    def add(self, tag, t, ang):
        with self.lock:
            self.series.setdefault(tag, collections.deque(maxlen=400)).append((float(t), np.asarray(ang, float)))

    def others(self, tag, t, min_cameras=2, min_estimates=3):
        with self.lock:
            meds = []
            for k, q in self.series.items():
                if k == tag:
                    continue
                win = [a for (tt, a) in q if t - self.window_s <= tt <= t + 1e-9]
                if len(win) >= min_estimates:
                    meds.append(np.median(np.array(win), 0))
        return np.median(np.array(meds), 0) if len(meds) >= min_cameras else None


class KnockMonitor:
    """Both measurements and their step tests for one camera. feed() takes an analysis-size grey frame (uint8) and the
    full stream size; it returns what it measured."""

    def __init__(self, cam: dict, st: dict, seed: int = 0):
        self.tag = cam['tag']
        self.st = st
        self.lens = RadialLens(cam['lens']) if cam.get('lens') else None
        plan = cam.get('travel_dir_optical')
        self.plan_dir = np.asarray(plan, float) / np.linalg.norm(plan) if plan is not None else None
        self.seed = int(seed)
        self.R_vo = None                  # optical -> vehicle rotation at the planned pose
        pose = cam.get('pose')
        if isinstance(pose, dict) and all(k in pose for k in ('yaw', 'pitch', 'roll')):
            from .inject import BODY_TO_OPTICAL, rot_zyx
            self.R_vo = rot_zyx(pose['yaw'], pose['pitch'], pose['roll']) @ BODY_TO_OPTICAL.T
        self.consensus = None             # a RigConsensus shared by the rig's cameras (set by the watchdog)
        self.relative = None              # whether the travel step is judged against the other cameras
        mk = lambda thr, floor: StepDetector(st['baseline_s'], st['window_s'], st['hold_s'], thr, st['z_min'], floor)
        self.travel = mk(st['threshold_deg'], 0.3)            # the solver's own scatter on a good pair
        self.body_steps = mk(st['body_threshold_deg'], 0.01)
        exp = None
        if cam.get('body_mask'):
            exp = decode_mask(cam['body_mask'])
        self.body = BodyEdge(self.lens, st, self.plan_dir, exp)
        self.ref = None                   # the travel direction the angles are taken about
        self.t0 = None
        self.rays_mask = None
        self.analysed = 0
        self.travel_used = 0
        self.last_travel = None           # the latest direction (deg about ref) and its offset from the plan
        self.last_body = None
        self.size_full = None             # the stream's size (W, H)
        self.reject = collections.Counter()

    def reset(self):
        self.__init__({'tag': self.tag, 'lens': self.lens.to_dict() if self.lens else None,
                       'travel_dir_optical': self.plan_dir.tolist() if self.plan_dir is not None else None},
                      self.st, self.seed)

    def _mask(self, w, h):
        if self.rays_mask is None or self.rays_mask.shape != (h, w):
            import cv2
            m = (self.lens.inside(w, h, math.radians(self.st['max_theta_deg'])) if self.lens is not None
                 else np.ones((h, w), bool))
            self.rays_mask = cv2.erode(m.astype(np.uint8) * 255, np.ones((5, 5), np.uint8))
        return self.rays_mask

    def feed(self, g, t, full_w, full_h, prev=None) -> dict:
        """g: the analysed frame (grey, analysis size), t its time; prev: (grey, t) of the frame just before it in the
        stream (the travel direction is measured between these two: close in time, so the near ground has not moved
        too far to track)."""
        out = {}
        self.analysed += 1
        if self.t0 is None:
            self.t0 = t
        h, w = g.shape
        self.size_full = (full_w, full_h)
        sx = full_w / w                                      # stream pixels per analysis pixel
        learning = t - self.t0 < self.st['baseline_s']
        # ---- the body
        if learning:
            self.body.learn(g)
        elif not self.body.ready and not self.body.absent:
            dpp = self.lens.deg_per_px(0.5 * self.lens.max_theta, full_w) * sx if self.lens else 0.1
            self.body.finish(dpp)
        if self.body.ready and not learning:
            m = self.body.match(g)
            if m is not None:
                dx, dy, score = m
                vec = (self._body_deg(dx, dy, sx, full_w, full_h) if self.lens is not None
                       else np.array([dx, dy]) * 0.1)
                self.last_body = {'t': t, 'dx_px': dx * sx, 'dy_px': dy * sx, 'deg': vec.tolist(), 'score': score}
                out['body'] = self.body_steps.add(t, vec)
        # ---- the direction of travel
        if self.lens is not None and prev is not None and prev[0].shape == g.shape and t > prev[1]:
            out['travel'] = self._travel(prev[0], g, t, full_w, full_h, sx)
        return out

    def _body_deg(self, dx, dy, sx, W, H) -> np.ndarray:
        """The body edge's shift (dx, dy analysis pixels) as an angle, a 2-vector in degrees along the shift: the
        median over the edge pixels of the angle between a pixel's ray and the ray of the pixel it moved to, through
        the lens table (a fisheye's scale shrinks towards the rim, where a mirror camera sees the body, and differs
        along and across the radius)."""
        key = (sx, W, H)
        if getattr(self, '_bpix', None) is None or self._bpix[0] != key:
            x0, y0, _, _ = self.body.box
            yy, xx = np.nonzero(self.body.mask > 0)
            if len(xx) > 400:
                k = np.linspace(0, len(xx) - 1, 400).astype(int)
                xx, yy = xx[k], yy[k]
            cx, cy = (xx + x0).astype(float), (yy + y0).astype(float)
            d0, _ = self.lens.unproject((cx + 0.5) * sx - 0.5, (cy + 0.5) * sx - 0.5, W, H)
            self._bpix = (key, cx, cy, d0)
        _, cx, cy, d0 = self._bpix
        n = math.hypot(dx, dy)
        if n < 1e-9:
            return np.zeros(2)
        d1, _ = self.lens.unproject((cx + dx + 0.5) * sx - 0.5, (cy + dy + 0.5) * sx - 0.5, W, H)
        a = np.degrees(np.arccos(np.clip(np.sum(d0 * d1, 1), -1, 1)))
        a = a[np.isfinite(a)]
        if not len(a):
            return np.array([dx, dy]) * 0.1
        return np.array([dx, dy]) / n * float(np.median(a))

    def _travel(self, g0, g1, t, W, H, sx):
        import cv2
        st = self.st
        mask = self._mask(g0.shape[1], g0.shape[0])
        if self.body.ready:
            x0, y0, x1, y1 = self.body.box
            mask = mask.copy()
            bm = np.zeros_like(mask)
            bm[y0:y1, x0:x1] = (self.body.mask > 0).astype(np.uint8) * 255
            mask[cv2.dilate(bm, np.ones((9, 9), np.uint8)) > 0] = 0
        p0 = cv2.goodFeaturesToTrack(g0, 1500, 0.005, 5, mask=mask, blockSize=7)
        if p0 is None or len(p0) < 20:
            self.reject['few features'] += 1
            return None
        p0 = bucket(p0.reshape(-1, 2), g0.shape, per=int(st['features_per_cell'])).reshape(-1, 1, 2)
        p0 = np.ascontiguousarray(p0, np.float32)
        lk = dict(winSize=(15, 15), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))
        p1, s1, err = cv2.calcOpticalFlowPyrLK(g0, g1, p0, None, **lk)
        pb, sb, _ = cv2.calcOpticalFlowPyrLK(g1, g0, p1, None, **lk)
        ok = (s1.ravel() == 1) & (sb.ravel() == 1) & (np.linalg.norm((pb - p0).reshape(-1, 2), axis=1) < 0.5)
        if ok.sum() >= 20:              # patches that changed more than the median one (glints, flicker): dropped
            e = err.ravel()
            ok &= e <= np.median(e[ok])
        a, b = p0.reshape(-1, 2)[ok], p1.reshape(-1, 2)[ok]
        if len(a) < 15:
            self.reject['few tracks'] += 1
            return None
        flow = np.linalg.norm(b - a, axis=1)
        if np.median(flow) < st['min_parallax_px']:
            self.reject['standing'] += 1
            return None
        to_stream = lambda x: (x + 0.5) * sx - 0.5                     # analysis -> stream pixel coordinates
        r0, th0 = self.lens.unproject(to_stream(a[:, 0]), to_stream(a[:, 1]), W, H)
        r1, th1 = self.lens.unproject(to_stream(b[:, 0]), to_stream(b[:, 1]), W, H)
        lim = math.radians(st['max_theta_deg'])
        keep = np.isfinite(th0) & np.isfinite(th1) & (th0 < lim) & (th1 < lim)
        if keep.sum() < 15:
            self.reject['few rays'] += 1
            return None
        f_an = self.lens.f_px * W / self.lens.width / sx               # focal length in analysis pixels
        res = translation_from_rays(r0[keep], r1[keep], thresh=1.0 / max(f_an, 1.0), seed=self.seed * 100003 +
                                    self.analysed)
        if res is None:
            self.reject['no direction'] += 1
            return None
        d, inl, share, rms = res
        if inl < 15 or share < st['min_inlier_share']:
            self.reject['turning or traffic'] += 1          # the flow is not one straight motion through a still world
            return None
        if self.ref is None:
            self.ref = self.plan_dir if self.plan_dir is not None else d
        if d @ self.ref < 0:
            d = -d                                                      # reversing: the same axis
        ang = angles_about(d, self.ref)
        vs_plan = (float(math.degrees(math.acos(min(1.0, abs(float(d @ self.plan_dir))))))
                   if self.plan_dir is not None else None)
        self.last_travel = {'t': t, 'deg': ang.tolist(), 'inliers': inl, 'inlier_share': share,
                            'rms_deg': math.degrees(rms), 'vs_plan_deg': vs_plan}
        if self.R_vo is None or self.consensus is None:
            self.travel_used += 1
            return self.travel.add(t, ang)
        # the direction of travel in the vehicle frame through the planned pose; every camera sees the same one, so
        # what the others see now (their median) takes out what moves them all: the body pitching under braking or
        # load, a crab angle; a knocked camera alone departs from it
        v = self.R_vo @ d
        av = angles_about(v, np.array([1.0, 0.0, 0.0]))
        self.consensus.add(self.tag, t, av)
        cons = self.consensus.others(self.tag, t)
        self.relative = True
        if cons is None:                                  # the others have too few estimates yet
            self.reject['no consensus'] += 1
            return None
        self.last_travel['vs_rig_deg'] = (av - cons).tolist()
        self.travel_used += 1
        return self.travel.add(t, av - cons)

    def status(self) -> dict:
        """What the watchdog reports for this camera's knock check. A knocked method's size is its latest window
        median's distance from the baseline (it settles once the window holds only frames after the knock); the
        camera's size is the travel direction's when that method fired, else the body edge's."""
        learning = self.travel.baseline is None and self.body_steps.baseline is None
        k = self.travel.knock or self.body_steps.knock
        which = None
        if self.travel.knock and self.body_steps.knock:
            which = 'travel direction and body edge'
        elif self.travel.knock:
            which = 'travel direction'
        elif self.body_steps.knock:
            which = 'body edge'
        size = lambda s: None if not s.knock else round(float((s.last or s.knock)['size']), 3)
        tsz, bsz = size(self.travel), size(self.body_steps)
        rays = None
        if self.body.ready and self.lens is not None and self.size_full:
            W, H = self.size_full
            self._body_deg(0.0, 0.0, W / self.body.size[1], W, H)       # the edge pixels' rays
            d0 = self._bpix[3]
            d0 = d0[np.all(np.isfinite(d0), 1)]
            if len(d0):
                rays = [[round(float(x), 5) for x in r] for r in d0[np.linspace(0, len(d0) - 1, min(32, len(d0)))
                                                                    .astype(int)]]
        return {
            'state': 'knock' if k else ('learning' if learning else 'ok'),
            'method': which,
            'size_deg': tsz if tsz is not None else bsz,
            'at_s': None if not k else min(s.knock['t'] for s in (self.travel, self.body_steps) if s.knock),
            'onset_s': None if not k else min(s.knock['onset_t'] for s in (self.travel, self.body_steps) if s.knock),
            'travel': {'state': self.travel.state, 'size_deg': tsz, 'estimates': self.travel_used,
                       'against': None if self.relative is None else ('the other cameras' if self.relative
                                                                       else 'its own baseline'),
                       'last': self.last_travel, 'step': self.travel.last,
                       'baseline_sigma_deg': None if self.travel.sigma is None else
                       [round(float(x), 4) for x in self.travel.sigma], 'rejected': dict(self.reject)},
            'body': {'state': ('absent' if self.body.absent else self.body_steps.state), 'size_deg': bsz,
                     'why': self.body.absent, 'share': getattr(self.body, 'share', None), 'last': self.last_body,
                     'step': self.body_steps.last, 'edge_rays': rays,
                     'baseline_sigma_deg': None if self.body_steps.sigma is None else
                     [round(float(x), 4) for x in self.body_steps.sigma]},
            'analysed': self.analysed,
        }
