"""Research maps for perception and fusion work: what a BEV network's feature grid gets from each camera, how well two
overlapping cameras triangulate, how many camera pixels each lidar point has around it, and a click-to-pin probe that
reads out what every camera sees of an object at a ground point.

All in the vehicle frame (ISO 8855, metres, rear-axle origin on the ground); every camera is the project's real lens
(core.lens.Intrinsics) and is hidden by the body, the rig and the housings through the engine's occlusion cube maps
(the same visibility test as the coverage analysis, analysis.eval_points). Pure functions: nothing here changes the
project.

- bev_feature_map: a square BEV grid (BEVFormer / Lift-Splat-Shoot / OccNet style, 200 × 200 cells of 0.5 m for
  ±50 m by default). Per cell and camera: whether the camera sees the cell centre, and the cell's footprint in pixels
  (its 4 corners projected through the lens, shoelace area); divided by stride² that is the number of backbone feature
  cells (1/16 or 1/32 feature maps) the BEV cell can draw on. Where the sum over cameras is under one feature cell the
  cell is ray-starved: a far-field BEV pillar no feature reaches.
- stereo_depth_sigma: on ground cells two or more cameras see, the depth uncertainty of triangulating the point from
  each pair of rays, σ = √((σθ₁ r₁)² + (σθ₂ r₂)²) / sin γ, with σθᵢ = σ_px / (pixels per radian of camera i's lens at
  that image point) and γ the angle between the rays at the point; the best pair's σ, σ / range and baseline. A binary
  overlap map does not tell a 3 cm baseline from a 1.8 m one: this does (σ ∝ 1 / baseline at range).
- lidar_camera_association: along each lidar's ground rings, the lidar's own point density on the ground (one point
  per azimuth step × ring spacing) against the best camera's pixel density there: camera pixels per lidar point.
- probe_readout: a pedestrian cylinder, a car-sized box or an upright ChArUco board at a ground point; per camera
  whether it sees it, the rays to it, its bounding box in pixels, the ray's incidence to the ground normal, the lens's
  dr/dθ there and the worst ground shift of the base point if the camera's mounting turns by drift_deg.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .analysis import camera_frames, eval_points
from .lens import dr_dtheta, resolve

SECTORS = ('front', 'front-left', 'left', 'rear-left', 'rear', 'rear-right', 'right', 'front-right')


# ---------------------------------------------------------------- shared helpers
def _cams(project, variant=None):
    v = variant or project.variant
    return [c for c in v.cameras if c.visible]


def _intr(project, cam):
    return resolve(project.module_for(cam), project.lens_for(cam))


def _cube(engine, project, cam, result=None):
    if result is not None and cam.uid in getattr(result, 'cubes', {}):
        return result.cubes[cam.uid]
    vg = engine.vehicle_geom(project)
    return engine.cube(cam.pos(), vg)


def _project(cam, intr, P):
    """World points (..., 3) -> pixel (u, v), theta (rad), in-image flag, through the camera's lens."""
    P = np.asarray(P, float)
    shp = P.shape[:-1]
    _, W2O = camera_frames(cam)
    opt = (P.reshape(-1, 3) - cam.pos()) @ W2O.T
    u, v, th, ok = intr.project_dirs(opt)
    return u.reshape(shp), v.reshape(shp), th.reshape(shp), ok.reshape(shp)


def _quad_area(u, v):
    """Shoelace area (px²) of quadrilaterals given as (..., 4) corner coordinates in order."""
    x, y = u, v
    a = (x[..., 0] * y[..., 1] - x[..., 1] * y[..., 0] + x[..., 1] * y[..., 2] - x[..., 2] * y[..., 1] +
         x[..., 2] * y[..., 3] - x[..., 3] * y[..., 2] + x[..., 3] * y[..., 0] - x[..., 0] * y[..., 3])
    return 0.5 * np.abs(a)


def ground_px_per_m2(cam, intr, P, h=0.05, z=0.0):
    """Pixels per m² of ground at vehicle-frame points P (K, 2 or 3) in this camera: a square of side h about each point
    projected through the lens, its area in px² over h². No visibility test (see eval_points)."""
    P = np.asarray(P, float).reshape(-1, P.shape[-1] if np.ndim(P) > 1 else 3)
    x, y = P[:, 0], P[:, 1]
    zz = P[:, 2] if P.shape[1] > 2 else np.full(len(P), z)
    d = h / 2
    C = np.stack([np.stack([x - d, y - d, zz], -1), np.stack([x + d, y - d, zz], -1),
                  np.stack([x + d, y + d, zz], -1), np.stack([x - d, y + d, zz], -1)], axis=1)   # (K, 4, 3)
    u, v, _, _ = _project(cam, intr, C)
    return _quad_area(u, v) / (h * h)


def px_per_rad(intr, theta):
    """The lens's angular sampling at off-axis angle theta (rad): √(radial · tangential) pixels per radian, the same
    measure as Intrinsics.density_px_per_cm (that figure × 100 at 1 m)."""
    return intr.density_px_per_cm(np.asarray(theta, float), 1.0) * 100.0


def _footprint_centre(project):
    m = project.vehicle.mesh
    if m is None or m.n_tris == 0:
        return np.zeros(2), (np.zeros(2), np.zeros(2))
    P = m.vertices
    lo, hi = P[:, :2].min(0), P[:, :2].max(0)
    return (lo + hi) / 2, (lo, hi)


# ---------------------------------------------------------------- 1 · BEV feature grid
@dataclass
class BevMap:
    """A BEV grid's camera support. Arrays are (n, n), row j = y, column i = x, cell centres at x0 + (i + ½) cell."""
    x0: float
    y0: float
    cell_m: float
    n: int
    extent_m: float              # half-width of the grid about the vehicle's footprint centre
    stride: int                  # backbone downsampling (16 = 1/16 feature map)
    rays: np.ndarray             # cameras that see the cell centre
    feats: np.ndarray            # backbone feature cells the BEV cell covers, summed over cameras
    best_feats: np.ndarray       # the best single camera's
    best: np.ndarray             # index into tags of that camera (-1: none)
    starved: np.ndarray          # feats < 1 outside the vehicle's footprint: no feature reaches the cell
    body: np.ndarray             # under the vehicle (its plan bounding box)
    tags: list
    per_cam: dict = field(default_factory=dict)    # tag -> feats (n, n) float32
    ms: float = 0.0

    @property
    def xs(self):
        return self.x0 + (np.arange(self.n) + 0.5) * self.cell_m

    @property
    def ys(self):
        return self.y0 + (np.arange(self.n) + 0.5) * self.cell_m

    def summary(self) -> dict:
        out = ~self.body
        tot = max(int(out.sum()), 1)
        r = np.hypot(*np.meshgrid(self.xs - (self.x0 + self.extent_m), self.ys - (self.y0 + self.extent_m)))
        far = {}
        for lim in (10, 25, 50):
            ring = out & (r <= lim)
            far[f'starved_within_{lim}m_pct'] = round(100.0 * float((self.starved & ring).sum()) / max(int(ring.sum()), 1), 2)
        return {'cells': self.n * self.n, 'cell_m': self.cell_m, 'stride': self.stride, 'extent_m': self.extent_m,
                'seen_pct': round(100.0 * float(((self.rays > 0) & out).sum()) / tot, 2),
                'starved_pct': round(100.0 * float(self.starved.sum()) / tot, 2),
                'multi_pct': round(100.0 * float(((self.rays >= 2) & out).sum()) / tot, 2),
                'median_feats_seen': float(np.median(self.feats[(self.rays > 0) & out])) if ((self.rays > 0) & out).any()
                else 0.0, **far, 'ms': round(self.ms, 1)}


def bev_feature_map(project, engine, extent_m=50.0, cell_m=0.5, stride=16, result=None, variant=None) -> BevMap:
    """The BEV grid's camera support (BevMap): per 0.5 m cell (default) of a square ±extent_m about the vehicle's
    footprint centre, which cameras see its centre (in the image, in front of the lens, not hidden by the body, rig or
    housings: analysis.eval_points with the engine's cube maps) and how many backbone feature cells of size
    stride × stride pixels its footprint covers (4 corners through the lens, shoelace area / stride²).

    Approximation: a cell counts for a camera when its centre is visible; its whole footprint is then counted, also a
    part hidden by the body or outside the image (at the edge of a footprint by at most one cell)."""
    import time
    t0 = time.perf_counter()
    cell_m = float(cell_m)
    n = max(int(round(2 * extent_m / cell_m)), 1)
    c, (lo, hi) = _footprint_centre(project)
    x0, y0 = float(c[0] - n * cell_m / 2), float(c[1] - n * cell_m / 2)
    xs = x0 + (np.arange(n) + 0.5) * cell_m
    ys = y0 + (np.arange(n) + 0.5) * cell_m
    X, Y = np.meshgrid(xs, ys)
    P = np.stack([X, Y, np.zeros_like(X)], -1).reshape(-1, 3)
    d = cell_m / 2
    corners = np.stack([P + [-d, -d, 0], P + [d, -d, 0], P + [d, d, 0], P + [-d, d, 0]], axis=1)   # (K, 4, 3)
    body = ((X >= lo[0]) & (X <= hi[0]) & (Y >= lo[1]) & (Y <= hi[1]))
    cams = _cams(project, variant)
    s2 = float(stride) ** 2
    rays = np.zeros(n * n, np.int16)
    feats = np.zeros(n * n, np.float32)
    best_f = np.zeros(n * n, np.float32)
    best = np.full(n * n, -1, np.int16)
    per = {}
    engine.vehicle_geom(project, variant)
    for k, cam in enumerate(cams):
        intr = _intr(project, cam)
        vis, _, _, _ = eval_points(P, cam, intr, _cube(engine, project, cam, result))
        vis &= ~body.ravel()
        f = np.zeros(n * n, np.float32)
        if vis.any():
            u, v, _, _ = _project(cam, intr, corners[vis])
            f[vis] = (_quad_area(u, v) / s2).astype(np.float32)
        rays += vis
        feats += f
        upd = f > best_f
        best_f = np.where(upd, f, best_f)
        best = np.where(upd, k, best)
        per[cam.tag] = f.reshape(n, n)
    feats2 = feats.reshape(n, n)
    return BevMap(x0, y0, cell_m, n, float(extent_m), int(stride), rays.reshape(n, n), feats2, best_f.reshape(n, n),
                  best.reshape(n, n), (feats2 < 1.0) & ~body, body, [c.tag for c in cams], per,
                  (time.perf_counter() - t0) * 1000)


# ---------------------------------------------------------------- 2 · stereo triangulation
@dataclass
class StereoMap:
    """Triangulation depth uncertainty on a ground grid (row j = y, column i = x). NaN where fewer than two cameras
    see the cell."""
    x0: float
    y0: float
    cell: float
    sigma_m: np.ndarray          # the best pair's 1σ position uncertainty along its worst direction, metres
    sigma_pct: np.ndarray        # sigma_m / mean range of the pair, %
    baseline_m: np.ndarray       # that pair's baseline (distance between the cameras)
    angle_deg: np.ndarray        # the angle between its two rays at the point
    pair: np.ndarray             # index into pairs (-1: none)
    pairs: list                  # [(tag a, tag b)]
    sigma_px: float = 0.5

    def summary(self) -> dict:
        ok = np.isfinite(self.sigma_m)
        if not ok.any():
            return {'cells': 0}
        cnt = np.bincount(self.pair[ok], minlength=len(self.pairs))
        return {'cells': int(ok.sum()), 'median_sigma_m': float(np.median(self.sigma_m[ok])),
                'median_sigma_pct': float(np.median(self.sigma_pct[ok])),
                'worst_sigma_m': float(np.max(self.sigma_m[ok])),
                'pairs': [{'pair': f'{a}·{b}', 'cells': int(cnt[i]),
                           'baseline_m': round(float(np.median(self.baseline_m[ok & (self.pair == i)])), 3),
                           'median_sigma_m': round(float(np.median(self.sigma_m[ok & (self.pair == i)])), 4)}
                          for i, (a, b) in enumerate(self.pairs) if cnt[i]],
                'sigma_px': self.sigma_px}


def pair_sigma(Ca, Cb, P, pxrad_a, pxrad_b, sigma_px=0.5):
    """1σ triangulation uncertainty (m) of points P (K, 3) seen from camera centres Ca, Cb with angular sampling
    pxrad_a, pxrad_b (px/rad at the points' image positions): √((σθa ra)² + (σθb rb)²) / sin γ. Also (ra, rb, γ rad)."""
    da = P - np.asarray(Ca, float)
    db = P - np.asarray(Cb, float)
    ra = np.linalg.norm(da, axis=-1)
    rb = np.linalg.norm(db, axis=-1)
    cg = np.clip(np.sum(da * db, axis=-1) / np.maximum(ra * rb, 1e-12), -1.0, 1.0)
    g = np.arccos(cg)
    sa = sigma_px / np.maximum(pxrad_a, 1e-9)
    sb = sigma_px / np.maximum(pxrad_b, 1e-9)
    sig = np.hypot(sa * ra, sb * rb) / np.maximum(np.sin(g), 1e-9)
    return sig, ra, rb, g


def stereo_depth_sigma(project, engine, result=None, sigma_px=0.5, grid=None, z=None, variant=None) -> StereoMap:
    """Per ground cell two or more cameras see (the analysis ROI grid result.grid, or grid: a RoiGrid; visibility from
    result.cams when given, else eval_points), the best camera pair's triangulation uncertainty (StereoMap): every pair
    that sees the cell, σ = √((σθ₁ r₁)² + (σθ₂ r₂)²) / sin γ with σθᵢ = sigma_px / (the lens's px/rad at the point's
    image position), the smallest σ kept with its pair, baseline and ray angle."""
    if result is None and grid is None:
        result = engine.analyse(project, variant)
    g = grid if grid is not None else result.grid
    zz = float(result.z if (z is None and result is not None) else (z or 0.0))
    cams = _cams(project, variant)
    shape = (g.ny, g.nx)
    Pfull = g.points(zz).reshape(-1, 3)
    vis = []
    for cam in cams:
        if result is not None and grid is None and result.cam_index(cam.uid) >= 0:
            vis.append(result.cams[result.cam_index(cam.uid)].vis.ravel() & g.roi.ravel())
        else:
            engine.vehicle_geom(project, variant)
            vv, _, _, _ = eval_points(Pfull, cam, _intr(project, cam), _cube(engine, project, cam, result))
            vis.append(vv & g.roi.ravel() & ~g.body.ravel())
    V = np.array(vis) if vis else np.zeros((0, len(Pfull)), bool)
    multi = V.sum(0) >= 2 if len(V) else np.zeros(len(Pfull), bool)
    idx = np.nonzero(multi)[0]
    P = Pfull[idx]
    pr = []
    for cam in cams:
        intr = _intr(project, cam)
        _, _, th, _ = _project(cam, intr, P)
        pr.append(px_per_rad(intr, th))
    best_s = np.full(len(idx), np.inf)
    best_p = np.full(len(idx), -1, np.int32)
    best_b = np.full(len(idx), np.nan)
    best_g = np.full(len(idx), np.nan)
    best_r = np.full(len(idx), np.nan)
    pairs = []
    for a in range(len(cams)):
        for b in range(a + 1, len(cams)):
            both = V[a, idx] & V[b, idx]
            if not both.any():
                continue
            Ca, Cb = cams[a].pos(), cams[b].pos()
            s, ra, rb, gm = pair_sigma(Ca, Cb, P[both], pr[a][both], pr[b][both], sigma_px)
            k = len(pairs)
            pairs.append((cams[a].tag, cams[b].tag))
            w = np.nonzero(both)[0]
            better = s < best_s[w]
            ww = w[better]
            best_s[ww] = s[better]
            best_p[ww] = k
            best_b[ww] = float(np.linalg.norm(Ca - Cb))
            best_g[ww] = np.degrees(gm[better])
            best_r[ww] = 0.5 * (ra[better] + rb[better])
    full = lambda a, fill=np.nan: _scatter(len(Pfull), idx, a, fill).reshape(shape)    # noqa: E731
    sig = np.where(np.isfinite(best_s), best_s, np.nan)
    return StereoMap(g.x0, g.y0, g.cell, full(sig), full(100.0 * sig / np.maximum(best_r, 1e-9)), full(best_b),
                     full(best_g), _scatter(len(Pfull), idx, best_p, -1).reshape(shape).astype(np.int32), pairs,
                     float(sigma_px))


def _scatter(n, idx, vals, fill):
    vals = np.asarray(vals)
    out = np.full(n, fill, dtype=float if isinstance(fill, float) else vals.dtype)
    out[idx] = vals
    return out


# ---------------------------------------------------------------- 3 · lidar–camera association
def lidar_camera_association(project, engine, result=None, az_step=2.0, variant=None) -> list:
    """Per visible lidar of the variant, along its ground rings (beams that meet the ground within range, not stopped
    by the vehicle): the lidar's ground point density (points/m²: one point per its real azimuth step × the radial
    spacing to the neighbouring rings) and the best visible camera's ground pixel density there (px/m²), and their
    ratio, camera pixels per lidar point. Rings are sampled every az_step degrees.

    [{'tag', 'model', 'points' (K, 3), 'ring' (K,) channel index, 'lidar_pts_m2', 'cam_px_m2', 'ratio', 'camera'
      (K,) best camera tag or '', 'sectors': [{'sector', 'points', 'median_ratio', 'seen_pct'}]}]"""
    from . import sensors as S
    v = variant or project.variant
    vg = engine.vehicle_geom(project, variant)
    cams = _cams(project, variant)
    intrs = [_intr(project, c) for c in cams]
    out = []
    centre, _ = _footprint_centre(project)
    for s in v.sensors:
        if s.kind != 'lidar' or not s.visible:
            continue
        m = project.sensor_model_for(s)
        pos = s.pos().astype(float)
        D = S.beam_dirs(s, m, az_step)                                  # (C, N, 3)
        C, N = D.shape[:2]
        with np.errstate(divide='ignore', invalid='ignore'):
            t = np.where(D[..., 2] < -1e-6, -pos[2] / D[..., 2], np.inf)
        ok = np.isfinite(t) & (t <= m.range_m) & (t >= m.range_min_m)
        cube = engine.cube(pos, vg) if vg is not None else None
        if cube is not None and ok.any():
            depth = cube.lookup(D.reshape(-1, 3))[0].reshape(C, N).astype(float)
            depth[depth <= 0.15] = np.inf                                # its own mount
            ok &= ~(depth < t - 0.05)
        G = pos + D * np.where(ok, t, 0.0)[..., None]                   # ground points
        rho = np.hypot(G[..., 0] - pos[0], G[..., 1] - pos[1])
        rho = np.where(ok, rho, np.nan)
        # radial spacing to the neighbouring rings (channels are lowest first: rho grows with the channel)
        with np.errstate(invalid='ignore'):
            up = np.vstack([rho[1:] - rho[:-1], np.full((1, N), np.nan)])
            dn = np.vstack([np.full((1, N), np.nan), rho[1:] - rho[:-1]])
            up, dn = np.abs(up), np.abs(dn)
            dr = np.where(np.isnan(up), dn, np.where(np.isnan(dn), up, 0.5 * (up + dn)))
        arc = rho * math.radians(max(m.h_res, 1e-6))
        dens = 1.0 / np.maximum(dr * arc, 1e-9)                          # points / m²
        sel = ok & np.isfinite(dr)
        Pk = G[sel]
        ring = np.nonzero(sel)[0]
        lid = dens[sel]
        best = np.zeros(len(Pk))
        btag = np.array([''] * len(Pk), dtype=object)
        for cam, intr in zip(cams, intrs):
            vis, _, _, _ = eval_points(Pk, cam, intr, _cube(engine, project, cam, result))
            if not vis.any():
                continue
            a = np.zeros(len(Pk))
            a[vis] = ground_px_per_m2(cam, intr, Pk[vis])
            upd = a > best
            best = np.where(upd, a, best)
            btag[upd] = cam.tag
        ratio = best / np.maximum(lid, 1e-9)
        az = np.degrees(np.arctan2(Pk[:, 1] - centre[1], Pk[:, 0] - centre[0]))
        from .perception import _sector_of
        sec = _sector_of(az) if len(az) else np.zeros(0, int)
        sectors = []
        for i, name in enumerate(SECTORS):
            msk = sec == i
            seen = msk & (best > 0)
            sectors.append({'sector': name, 'points': int(msk.sum()),
                            'median_ratio': float(np.median(ratio[seen])) if seen.any() else 0.0,
                            'seen_pct': round(100.0 * float(seen.sum()) / max(int(msk.sum()), 1), 1)})
        out.append({'tag': s.tag, 'model': m.name, 'points': Pk, 'ring': ring, 'lidar_pts_m2': lid, 'cam_px_m2': best,
                    'ratio': ratio, 'camera': btag, 'sectors': sectors})
    return out


# ---------------------------------------------------------------- 4 · probe
PROBES = {'pedestrian': 'a pedestrian: a cylinder 0.5 m across and 1.75 m tall',
          'box': 'a car: a box 4.5 × 1.8 × 1.5 m along the vehicle axes',
          'board': 'a ChArUco board 0.8 × 0.6 m standing upright (centre 0.8 m high), facing the vehicle'}


def probe_points(kind, point_xy, facing_xy=(0.0, 0.0)) -> tuple[np.ndarray, np.ndarray]:
    """Sample points of a probe at a ground point (its base centre): (points (K, 3), base point (3,))."""
    x, y = float(point_xy[0]), float(point_xy[1])
    if kind == 'pedestrian':
        a = np.linspace(0, 2 * math.pi, 16, endpoint=False)
        zs = np.linspace(0.0, 1.75, 6)
        P = [(x + 0.25 * math.cos(t), y + 0.25 * math.sin(t), z) for z in zs for t in a] + [(x, y, 1.75)]
    elif kind == 'box':
        hx, hy, h = 2.25, 0.9, 1.5
        g = np.linspace(-1, 1, 5)
        P = []
        for zz in (0.0, h / 2, h):
            for gx in g:
                for gy in g:
                    if abs(gx) == 1 or abs(gy) == 1 or zz == h:
                        P.append((x + gx * hx, y + gy * hy, zz))
    elif kind == 'board':
        n = np.array([facing_xy[0] - x, facing_xy[1] - y], float)
        n = n / max(np.linalg.norm(n), 1e-9)
        t = np.array([-n[1], n[0]])
        P = [(x + t[0] * a, y + t[1] * a, 0.8 + b) for a in np.linspace(-0.4, 0.4, 5) for b in np.linspace(-0.3, 0.3, 4)]
    else:
        raise ValueError(f'probe kind {kind!r}: one of {", ".join(PROBES)}')
    return np.array(P, float), np.array([x, y, 0.0])


def _rodrigues(axis, ang):
    k = np.asarray(axis, float)
    k = k / max(np.linalg.norm(k), 1e-12)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * K @ K


def drift_shift_cm(C, R, P, drift_deg):
    """Ground shift (cm) of the ground point P seen from camera centre C (body->world rotation R) when the camera's
    mounting turns by drift_deg about its own pitch (lateral), yaw (vertical) and roll (forward) axes, the pixel kept:
    {'pitch', 'yaw', 'roll', 'worst'}. The ray turned by the same angle meets the ground elsewhere; inf if it no longer
    reaches the ground."""
    C, P = np.asarray(C, float), np.asarray(P, float)
    d = P - C
    out = {}
    for name, axis in (('pitch', R[:, 1]), ('yaw', R[:, 2]), ('roll', R[:, 0])):
        worst = 0.0
        for sgn in (1.0, -1.0):
            d2 = _rodrigues(axis, sgn * math.radians(drift_deg)) @ d
            if d2[2] >= -1e-9:
                worst = math.inf
                continue
            s = (P[2] - C[2]) / d2[2]
            Q = C + d2 * s
            worst = max(worst, float(np.linalg.norm(Q - P)) * 100.0)
        out[name] = worst
    out['worst'] = max(out.values())
    return out


def probe_readout(project, engine, point_xy, kind='pedestrian', drift_deg=0.2, result=None, variant=None) -> dict:
    """What every camera sees of a probe (PROBES) whose base is at ground point point_xy (vehicle frame, m):
    {'kind', 'point' (3,), 'points' (K, 3), 'cameras': [{'tag', 'uid', 'sees', 'visible_points', 'rays' [(C, Q)],
    'bbox_px' (W, H), 'bbox' (u0, v0, u1, v1), 'range_m', 'incidence_deg', 'dr_dtheta_px_per_rad', 'drift_cm'
    {'pitch', 'yaw', 'roll', 'worst'}}], 'seen_by' [tags]}. A camera sees the probe when some sample points are in its
    image and not hidden (eval_points); incidence is the base ray's angle to the ground normal; dr/dθ is the lens's
    radial sampling at the probe's image point; drift_cm is drift_shift_cm at the base point."""
    centre, _ = _footprint_centre(project)
    Pp, base = probe_points(kind, point_xy, centre)
    engine.vehicle_geom(project, variant)
    rows = []
    for cam in _cams(project, variant):
        intr = _intr(project, cam)
        vis, _, _, _ = eval_points(Pp, cam, intr, _cube(engine, project, cam, result))
        C = cam.pos()
        R, _ = camera_frames(cam)
        row = {'tag': cam.tag, 'uid': cam.uid, 'sees': bool(vis.any()), 'visible_points': int(vis.sum()),
               'rays': [], 'bbox_px': (0.0, 0.0), 'bbox': None, 'range_m': float(np.linalg.norm(base - C)),
               'incidence_deg': None, 'dr_dtheta_px_per_rad': None, 'drift_cm': None}
        if vis.any():
            u, v, th, _ = _project(cam, intr, Pp[vis])
            u0, u1, v0, v1 = float(u.min()), float(u.max()), float(v.min()), float(v.max())
            row['bbox'] = (u0, v0, u1, v1)
            row['bbox_px'] = (round(u1 - u0, 1), round(v1 - v0, 1))
            Q = Pp[vis]
            keep = {int(np.argmin(u)), int(np.argmax(u)), int(np.argmin(v)), int(np.argmax(v)),
                    int(np.argmin(Q[:, 2])), int(np.argmax(Q[:, 2]))}
            row['rays'] = [(C.copy(), Q[i].copy()) for i in sorted(keep)]
            d = base - C
            row['incidence_deg'] = round(math.degrees(math.acos(min(abs(d[2]) / max(np.linalg.norm(d), 1e-9), 1.0))), 2)
            _, _, thc, _ = _project(cam, intr, Q.mean(axis=0)[None])
            row['dr_dtheta_px_per_rad'] = round(float(dr_dtheta(max(float(thc[0]), 1e-4), intr.f_mm, intr.projection,
                                                                 intr.kb) / intr.pixel_mm), 1)
            row['drift_cm'] = {k: round(x, 2) for k, x in drift_shift_cm(C, R, base, drift_deg).items()}
        rows.append(row)
    return {'kind': kind, 'about': PROBES[kind], 'point': base, 'points': Pp, 'drift_deg': float(drift_deg),
            'cameras': rows, 'seen_by': [r['tag'] for r in rows if r['sees']]}


__all__ = ['BevMap', 'StereoMap', 'bev_feature_map', 'stereo_depth_sigma', 'pair_sigma', 'lidar_camera_association',
           'probe_readout', 'probe_points', 'drift_shift_cm', 'ground_px_per_m2', 'px_per_rad', 'PROBES']
