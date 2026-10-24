"""3D blind volume: where around the vehicle something can stand unseen by every camera, not just on the ground.

The ROI columns (the analysis' region of interest on the ground) are sampled at several heights, and every point gets
the same visibility test as the ground analysis: inside a camera's image circle and sensor, and not behind the body
(the camera's occlusion cube map).  From that:

    blind volume        the space inside the ROI up to the chosen height that no camera sees, m³ and %
    per level           blind area at each height
    detect height       per ground cell, the lowest height a camera sees there: the smallest object standing on the
                        cell that is still detected
    objects             an object of height h standing on a cell is detected when any camera sees any part of it
                        (any sample of its column from the ground up to h); the area where it is not, per object class
    rose                for 72 azimuth sectors around the body: how far from the body something can still hide
                        (per level and per object class)
    blind_surface()     a smooth closed surface around the blind space (naive surface nets) for 3D views and export
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from .analysis import eval_points
from .lens import resolve

DEFAULT_OBJECTS = (('Pole 0.5 m', 0.5), ('Child 0.9 m', 0.9), ('Adult 1.8 m', 1.8))
SECTORS = 72
CHUNK = 250_000             # points per eval_points call


@dataclass
class VolumeSettings:
    height: float = 2.0         # m, top of the analysed space
    dz: float = 0.05            # m, level spacing
    cell: float = 0.0           # m, column spacing on the ground (0 = the analysis cell, at least 5 cm)

    def to_dict(self):
        return {'height': self.height, 'dz': self.dz, 'cell': self.cell}

    @staticmethod
    def from_dict(d):
        d = d or {}
        s = VolumeSettings()
        for k in ('height', 'dz', 'cell'):
            try:
                v = float(d.get(k, getattr(s, k)))
            except (TypeError, ValueError):
                continue
            if math.isfinite(v):
                setattr(s, k, v)
        s.height = min(max(s.height, 0.1), 6.0)
        s.dz = min(max(s.dz, 0.02), s.height)
        s.cell = 0.0 if s.cell <= 0 else min(max(s.cell, 0.02), 0.5)
        return s


@dataclass
class VolumeResult:
    x0: float
    y0: float
    cell: float
    nx: int
    ny: int
    z: np.ndarray               # (nz,) level heights (centres of the slabs)
    dz: float
    height: float
    roi: np.ndarray             # (ny, nx) bool, the analysed columns
    body: np.ndarray            # (ny, nx) bool, the vehicle's footprint
    dist: np.ndarray            # (ny, nx) float, distance from the body (as the ROI measures it)
    count: np.ndarray           # (nz, ny, nx) int8, cameras seeing each point (0 outside the ROI)
    centre: tuple               # body centre the rose is measured around
    outline: list               # body outline polylines (ground plane)
    objects: list               # [(name, height m)]
    object_seen: dict           # name -> (ny, nx) bool: an object of that class standing there is seen
    level_blind_m2: np.ndarray  # (nz,)
    reach_level: np.ndarray     # (nz, SECTORS) m, farthest blind point from the body per sector (0 = none)
    reach_object: dict          # name -> (SECTORS,) m, farthest undetected object position per sector
    detect_h: np.ndarray        # (ny, nx) m, lowest seen sample height per column (inf: none up to the height, nan: off ROI)
    metrics: dict
    per_cam_m3: dict = field(default_factory=dict)     # uid -> volume seen by that camera
    variant_uid: str = ''

    @property
    def blind(self) -> np.ndarray:
        """(nz, ny, nx) bool: inside the ROI and seen by no camera."""
        return self.roi[None] & (self.count == 0)

    @property
    def xs(self):
        return self.x0 + (np.arange(self.nx) + 0.5) * self.cell

    @property
    def ys(self):
        return self.y0 + (np.arange(self.ny) + 0.5) * self.cell

    @property
    def sector_deg(self):
        """Centre azimuth of each rose sector, degrees (0 = forward, 90 = left)."""
        return -180.0 + (np.arange(SECTORS) + 0.5) * 360.0 / SECTORS

    def level_index(self, z: float) -> int:
        return int(np.clip(np.round((z - self.z[0]) / self.dz), 0, len(self.z) - 1))

    def object_undetected_m2(self, name) -> float:
        return float((self.roi & ~self.object_seen[name]).sum() * self.cell ** 2)


def _sector_of(res_xy, centre):
    az = np.degrees(np.arctan2(res_xy[1] - centre[1], res_xy[0] - centre[0]))
    return np.clip(((az + 180.0) / (360.0 / SECTORS)).astype(int), 0, SECTORS - 1)


def analyse_volume(project, engine, res=None, height=2.0, dz=0.05, cell=None, objects=DEFAULT_OBJECTS,
                   progress=None, cancel=None) -> VolumeResult | None:
    """Visibility of the space above the ROI, level by level.  Returns None when cancelled or there is no vehicle."""
    t0 = time.perf_counter()
    height = float(height)
    dz = float(min(max(dz, 0.01), height))
    if not cell:
        cell = max(res.cell if res is not None else project.analysis.cell_m, 0.05)
    grid = engine.roi_grid(project, float(cell))
    if grid is None:
        return None
    vg = engine.vehicle_geom(project)
    nz = max(1, int(round(height / dz)))
    zs = (np.arange(nz) + 0.5) * dz
    roi = grid.roi
    jj, ii = np.nonzero(roi)
    xs, ys = grid.xs[ii], grid.ys[jj]
    n = len(ii)
    count = np.zeros((nz, grid.ny, grid.nx), np.int8)
    if res is not None:
        centre = tuple(res.centre)
    elif grid.body.any():
        bx = grid.xs[grid.body.any(axis=0)]
        centre = (float((bx.min() + bx.max()) / 2), 0.0)
    else:
        centre = (0.0, 0.0)
    cams = [c for c in project.variant.cameras if c.visible]
    per_cam = {}
    a3 = grid.cell ** 2 * dz
    total = max(len(cams) * nz * n, 1)
    done = 0
    for cam in cams:
        intr = resolve(project.module_for(cam), project.lens_for(cam))
        cube = engine.cube(cam.pos(), vg)
        seen_n = 0
        k0 = 0
        step = max(1, CHUNK // max(n, 1))
        while k0 < nz:
            if cancel is not None and cancel():
                return None
            k1 = min(nz, k0 + step)
            kk = np.arange(k0, k1)
            P = np.empty((len(kk) * n, 3))
            P[:, 0] = np.tile(xs, len(kk))
            P[:, 1] = np.tile(ys, len(kk))
            P[:, 2] = np.repeat(zs[kk], n)
            vis = eval_points(P, cam, intr, cube)[0].reshape(len(kk), n)
            count[k0:k1, jj, ii] += vis.astype(np.int8)
            seen_n += int(vis.sum())
            done += len(kk) * n
            if progress:
                progress(done / total)
            k0 = k1
        per_cam[cam.uid] = seen_n * a3
    blind = roi[None] & (count == 0)
    a2 = grid.cell ** 2
    level_blind = blind.sum(axis=(1, 2)) * a2
    roi_cells = int(roi.sum())
    blind_m3 = float(blind.sum() * a3)
    roi_m3 = roi_cells * a2 * nz * dz
    # objects: seen when any sample of the column up to the object's height is seen
    seen_up_to = np.logical_or.accumulate(count > 0, axis=0)
    obj_seen, obj_reach = {}, {}
    sec = _sector_of((grid.xs[None, :].repeat(grid.ny, 0), grid.ys[:, None].repeat(grid.nx, 1)), centre)
    for name, h in objects:
        k = int(np.clip(np.searchsorted(zs, float(h), side='right') - 1, 0, nz - 1))
        seen = roi & seen_up_to[k]
        obj_seen[name] = seen
        und = roi & ~seen
        r = np.zeros(SECTORS)
        np.maximum.at(r, sec[und], grid.dist[und])
        obj_reach[name] = r
    first = np.argmax(count > 0, axis=0)
    detect_h = np.where(roi, np.where((count > 0).any(axis=0), zs[first], np.inf), np.nan)
    reach = np.zeros((nz, SECTORS))
    for k in range(nz):
        b = blind[k]
        if b.any():
            np.maximum.at(reach[k], sec[b], grid.dist[b])
    metrics = {
        'blind_m3': blind_m3, 'roi_m3': float(roi_m3), 'blind_pct': 100.0 * blind_m3 / roi_m3 if roi_m3 else 0.0,
        'roi_m2': float(roi_cells * a2), 'height': float(nz * dz), 'dz': dz, 'cell': float(grid.cell), 'levels': nz,
        'points': int(nz * n), 'cameras': len(cams), 'seconds': time.perf_counter() - t0,
        'ground_blind_m2': float(level_blind[0]),
        'never_seen_m2': float((roi & np.isinf(detect_h)).sum() * a2),
        'worst_level_z': float(zs[int(np.argmax(level_blind))]), 'worst_level_m2': float(level_blind.max()),
        # the height up to which at most 1 % of the ROI is blind at every level
        'covered_to_m': float(zs[np.argmax(over)] - dz / 2) if (over := level_blind > 0.01 * roi_cells * a2).any()
        else float(nz * dz),
        'objects': {name: {'height': float(h), 'undetected_m2': float((roi & ~obj_seen[name]).sum() * a2),
                           'reach_m': float(obj_reach[name].max())} for name, h in objects},
    }
    return VolumeResult(grid.x0, grid.y0, grid.cell, grid.nx, grid.ny, zs, dz, float(nz * dz), roi.copy(), grid.body.copy(),
                        grid.dist.copy(), count, centre, list(grid.outline), list(objects), obj_seen, level_blind, reach,
                        obj_reach, detect_h, metrics, per_cam, project.variant.uid)


# ---------------------------------------------------------------------------------------------- surface
# the 12 edges of a unit cube as pairs of corner offsets (dz, dy, dx)
_CORNERS = np.array([[a, b, c] for a in (0, 1) for b in (0, 1) for c in (0, 1)])
_EDGES = [(i, j) for i in range(8) for j in range(i + 1, 8) if np.abs(_CORNERS[i] - _CORNERS[j]).sum() == 1]


def surface_nets(inside: np.ndarray, smooth: int = 4, blur: float = 0.0):
    """Naive surface nets on a boolean volume (z, y, x).  Returns (vertices in index coordinates (z, y, x) of the
    sample points, quads (Q, 4) with the outside on their right-hand normal side).  The volume is padded, so the
    surface is closed.  blur > 0 runs it on a Gaussian-blurred field (sigma in samples) with the vertices placed where
    the field crosses 0.5, which smooths gentle ramps the level steps would otherwise terrace (one-sample specks may
    vanish); smooth = relaxation passes, each vertex kept inside its cell."""
    f = np.pad(np.asarray(inside, np.float32), 1)
    if blur > 0:
        from scipy import ndimage
        f = ndimage.gaussian_filter(f, blur, mode='constant')
    s = f > 0.5
    Z, Y, X = s.shape
    cz, cy, cx = Z - 1, Y - 1, X - 1
    corner = [s[dz:dz + cz, dy:dy + cy, dx:dx + cx] for dz, dy, dx in _CORNERS]
    value = [f[dz:dz + cz, dy:dy + cy, dx:dx + cx] for dz, dy, dx in _CORNERS]
    acc = np.zeros((cz, cy, cx, 3), np.float32)
    cnt = np.zeros((cz, cy, cx), np.int16)
    for a, b in _EDGES:
        x = corner[a] != corner[b]
        if x.any():
            fa, fb = value[a][x], value[b][x]
            t = np.clip((0.5 - fa) / np.where(np.abs(fb - fa) > 1e-6, fb - fa, 1.0), 0.0, 1.0)[:, None]
            acc[x] += _CORNERS[a] + t * (_CORNERS[b] - _CORNERS[a])
            cnt[x] += 1
    active = cnt > 0
    idx = np.full((cz, cy, cx), -1, np.int64)
    nv = int(active.sum())
    idx[active] = np.arange(nv)
    kk, jj, ii = np.nonzero(active)
    verts = np.stack([kk, jj, ii], 1).astype(np.float32) + acc[active] / cnt[active][:, None]
    lo = np.stack([kk, jj, ii], 1).astype(np.float32)
    quads = []
    # x edges: points (k, j, i) - (k, j, i+1); cubes (k-1|k, j-1|j, i); quad in the y-z plane, normal +x when inside at i
    for axis in range(3):
        a = s[:, :, :-1] != s[:, :, 1:] if axis == 2 else (s[:, :-1, :] != s[:, 1:, :] if axis == 1 else s[:-1] != s[1:])
        k, j, i = np.nonzero(a)
        ins = s[k, j, i]
        if axis == 2:      # normal x: corners (j-1,k-1) (j,k-1) (j,k) (j-1,k): +y then +z, y × z = +x
            q = np.stack([idx[k - 1, j - 1, i], idx[k - 1, j, i], idx[k, j, i], idx[k, j - 1, i]], 1)
        elif axis == 1:    # normal y: +z then +x, z × x = +y
            q = np.stack([idx[k - 1, j, i - 1], idx[k, j, i - 1], idx[k, j, i], idx[k - 1, j, i]], 1)
        else:              # normal z: +x then +y, x × y = +z
            q = np.stack([idx[k, j - 1, i - 1], idx[k, j - 1, i], idx[k, j, i], idx[k, j, i - 1]], 1)
        q[~ins] = q[~ins][:, ::-1]
        quads.append(q)
    quads = np.concatenate(quads) if quads else np.zeros((0, 4), np.int64)
    quads = quads[(quads >= 0).all(axis=1)]
    if smooth and nv and len(quads):
        from scipy import sparse
        e = np.concatenate([quads[:, [0, 1]], quads[:, [1, 2]], quads[:, [2, 3]], quads[:, [3, 0]]])
        A = sparse.coo_matrix((np.ones(len(e) * 2, np.float32), (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])),
                              shape=(nv, nv)).tocsr()
        A.data[:] = 1.0
        deg = np.maximum(np.asarray(A.sum(axis=1)).ravel(), 1)
        for _ in range(int(smooth)):
            verts = np.clip(np.asarray(A @ verts) / deg[:, None], lo, lo + 1.0).astype(np.float32)
    # back to the unpadded sample index space
    return verts - 1.0, quads


def _pool(b, f):
    """Coarser boolean volume: blocks of f x f columns (and f levels when there are enough), blind when half is."""
    fz = f if b.shape[0] >= 4 * f else 1
    nz, ny, nx = (-(-b.shape[0] // fz)) * fz, (-(-b.shape[1] // f)) * f, (-(-b.shape[2] // f)) * f
    pad = np.zeros((nz, ny, nx), np.float32)
    pad[:b.shape[0], :b.shape[1], :b.shape[2]] = b
    m = pad.reshape(nz // fz, fz, ny // f, f, nx // f, f).mean(axis=(1, 3, 5))
    return m >= 0.5, fz


def blind_surface(vr: VolumeResult, smooth=True, caps=True, pool=1, top=None, body_walls=True, blur=0.0):
    """Surface around the blind space in world coordinates: (vertices (N, 3) float32, faces (M, 3) int32), outward
    normals by the right-hand rule.  smooth: True (4 relaxation passes) or a number of passes.  caps=False drops the faces lying on the ground and body_walls=False the walls
    against the vehicle's footprint (for drawing: both only z-fight with the ground and the body); pool > 1 builds it
    from a coarser volume (fewer faces); top = number of levels from the ground to include (a cut-away: the blind
    space up to that height, closed by a lid there); blur > 0 smooths it further (see surface_nets)."""
    b, cell, dz = vr.blind, vr.cell, vr.dz
    zmax = vr.height
    if top is not None:
        top = int(min(max(top, 1), len(vr.z)))
        b, zmax = b[:top], top * vr.dz
    if pool > 1:
        b, fz = _pool(b, int(pool))
        cell, dz = cell * pool, dz * fz
    v, q = surface_nets(b, (4 if smooth is True else int(smooth)) if smooth else 0, blur)
    if not len(q):
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int32)
    P = np.empty((len(v), 3), np.float32)
    P[:, 0] = vr.x0 + (v[:, 2] + 0.5) * cell
    P[:, 1] = vr.y0 + (v[:, 1] + 0.5) * cell
    P[:, 2] = np.clip((v[:, 0] + 0.5) * dz, 0.0, zmax)
    F = np.concatenate([q[:, [0, 1, 2]], q[:, [0, 2, 3]]]).astype(np.int32)
    if not caps:
        F = F[(P[F][:, :, 2] > 1e-3).any(axis=1)]
    if not body_walls and vr.body.any() and len(F):
        # a vertex borders the body when a footprint cell is among the 2 x 2 columns around it
        fx, fy = (P[:, 0] - vr.x0) / vr.cell - 0.5, (P[:, 1] - vr.y0) / vr.cell - 0.5
        near = np.zeros(len(P), bool)
        for ox in (0, 1):
            for oy in (0, 1):
                ii = np.clip(np.floor(fx).astype(int) + ox, 0, vr.nx - 1)
                jj = np.clip(np.floor(fy).astype(int) + oy, 0, vr.ny - 1)
                near |= vr.body[jj, ii]
        T = P[F]
        n = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
        steep = np.abs(n[:, 2]) < 0.7 * np.maximum(np.linalg.norm(n, axis=1), 1e-12)
        F = F[~(near[F].all(axis=1) & steep)]
    return P, F


# ---------------------------------------------------------------------------------------------- export
def to_doc(vr: VolumeResult, project=None) -> dict:
    """JSON-ready summary (schema vantage.blind_volume/1)."""
    r = lambda x, d=3: round(float(x), d)      # noqa: E731
    m = vr.metrics
    doc = {'schema': 'vantage.blind_volume/1',
           'note': 'Space inside the ROI, from the ground up to the height, that no camera sees (same visibility test as '
                   'the ground analysis: lens model, sensor, occlusion by the body, rig and housings). An object is '
                   'detected when any camera sees any part of its column from the ground up to its height. Reach: '
                   'farthest such point from the body per azimuth sector (0 deg = forward, 90 deg = left).',
           'height_m': r(vr.height), 'level_step_m': r(vr.dz), 'cell_m': r(vr.cell),
           'roi_m2': r(m['roi_m2']), 'roi_m3': r(m['roi_m3']), 'blind_m3': r(m['blind_m3']), 'blind_pct': r(m['blind_pct'], 2),
           'never_seen_m2': r(m['never_seen_m2']), 'covered_to_m': r(m['covered_to_m']),
           'levels': [{'z_m': r(z), 'blind_m2': r(a)} for z, a in zip(vr.z, vr.level_blind_m2)],
           'objects': [{'name': n, 'height_m': r(h), 'undetected_m2': r(m['objects'][n]['undetected_m2']),
                        'reach_m': r(m['objects'][n]['reach_m'])} for n, h in vr.objects],
           'rose': {'sector_deg': [r(a, 1) for a in vr.sector_deg],
                    'objects': {n: [r(x) for x in vr.reach_object[n]] for n, _ in vr.objects},
                    'levels': {f'{z:.2f}': [r(x) for x in vr.reach_level[k]] for k, z in enumerate(vr.z)}},
           'centre': [r(vr.centre[0]), r(vr.centre[1])]}
    if project is not None:
        doc['project'] = project.name
        doc['variant'] = project.variant.name
        tags = {c.uid: c.tag for c in project.variant.cameras}
        doc['cameras'] = [{'tag': tags.get(u, u), 'seen_m3': r(v)} for u, v in vr.per_cam_m3.items()]
    return doc


def to_obj(vr: VolumeResult, name='blind_volume') -> str:
    P, F = blind_surface(vr)
    lines = [f'# Vantage blind volume: {vr.metrics["blind_m3"]:.3f} m3 up to {vr.height:.2f} m (ISO 8855 frame, metres)',
             f'o {name}']
    lines += [f'v {x:.4f} {y:.4f} {z:.4f}' for x, y, z in P]
    lines += [f'f {a + 1} {b + 1} {c + 1}' for a, b, c in F]
    return '\n'.join(lines) + '\n'
