"""Coverage engine: ROI grid, per-camera visibility and pixel density, metrics and seams.

The ground is flat unless the project has a road (core.terrain, analysis.terrain): then the ground points sit on it,
the car stands on it and the road itself can hide ground from a camera; without one the analysis is unchanged.

What blocks a camera: the vehicle body (simplified to the occluder), the rig parts that occlude, and the housings of
the variant's sensors and other cameras (core.housings, composited into each camera's cube map by core.occlusion).
The vehicle geometry the engine hands out for a variant carries that variant's housings; its version is the body and
rig's, so what depends on the body alone (the ROI, the renderer's shadow and sky maps) is kept while a sensor moves.
"""
from __future__ import annotations

import dataclasses
import math
import threading
import time
from dataclasses import dataclass, field

import numpy as np

from .contour import iso_segments, chain
from .lens import resolve, Intrinsics
from .mathutil import rot_zyx, BODY_TO_OPTICAL
from .mesh import Mesh
from . import housings as H
from .occlusion import OccluderGeom, CubeCache, HousedOccluder
from .rig import merge, rig_key, rig_mesh
from .project import Project, Variant, Camera, ROI

HIST_EDGES = np.array([0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 1e9], float)


# ------------------------------------------------------------------ vehicle geometry
@dataclass
class VehicleGeom:
    version: int
    mesh: Mesh                  # the body with the rig parts (surfaces to mount on, mount checks)
    occ_mesh: Mesh              # the simplified body with the rig parts: what blocks the cameras
    occ: OccluderGeom
    centre: tuple
    bbox: tuple                 # the body alone
    max_dev: float
    body_occ_mesh: Mesh | None = None   # the simplified body alone: its footprint (ROI), drawing
    housings: object = None             # core.housings.HousingSet composited into the cube maps (None: body + rig)
    base: object = None                 # the same geometry without housings (None: this is it)

    @property
    def body(self) -> Mesh:
        return self.body_occ_mesh if self.body_occ_mesh is not None else self.occ_mesh

    @property
    def static(self) -> 'VehicleGeom':
        """The body and rig alone (no housings)."""
        return self.base if self.base is not None else self

    def with_housings(self, hs) -> 'VehicleGeom':
        """This geometry with a variant's housings blocking the cameras (same version, meshes and occluder triangles)."""
        st = self.static
        return dataclasses.replace(st, occ=HousedOccluder(st.occ, hs), housings=hs, base=st)


_vg_counter = [0]


def prepare_vehicle(mesh: Mesh, occluder_target: int = 8000, rig=None, body=None) -> VehicleGeom:
    """The body's occluder (simplified to occluder_target triangles) and, with rig parts, the rig merged in: the rig
    blocks cameras and carries them, but the ROI footprint and the vehicle's size are the body's. body: an already
    simplified body (mesh, max deviation) to reuse when only the rig changed. The sensor and camera housings are not
    merged here: Engine.vehicle_geom adds a variant's to the result (VehicleGeom.with_housings)."""
    _vg_counter[0] += 1
    if body is not None:
        body_occ, dev = body
    else:
        occ_faces = mesh.occluder_faces()
        base = mesh.subset(occ_faces) if not occ_faces.all() else mesh
        if base.n_tris > occluder_target:
            body_occ, dev = base.decimate(occluder_target)
        else:
            body_occ, dev = base, 0.0
    rm = rig_mesh(rig) if rig else None
    if rm is not None:
        mesh = merge(mesh, rm)
        keep = rm.occluder_faces()
        occ_mesh = merge(body_occ, rm.subset(keep)) if keep.any() else body_occ
    else:
        occ_mesh = body_occ
    glass_idx = [i for i, p in enumerate(occ_mesh.parts) if 'not mountable' in p.role or p.name.lower().startswith('glass')]
    glass = np.isin(occ_mesh.face_part, glass_idx)
    occ = OccluderGeom(occ_mesh.tri, occ_mesh.normals, glass, version=_vg_counter[0])
    lo, hi = body_occ.bbox
    return VehicleGeom(_vg_counter[0], mesh, occ_mesh, occ, ((lo[0] + hi[0]) / 2, 0.0), (lo, hi), dev, body_occ)


# ------------------------------------------------------------------ ROI
@dataclass
class RoiGrid:
    x0: float
    y0: float
    cell: float
    nx: int
    ny: int
    roi: np.ndarray
    body: np.ndarray
    dist: np.ndarray
    boundary: list
    outline: list
    area: float
    settings_key: tuple = ()

    @property
    def xs(self):
        return self.x0 + (np.arange(self.nx) + 0.5) * self.cell

    @property
    def ys(self):
        return self.y0 + (np.arange(self.ny) + 0.5) * self.cell

    def cell_of(self, x, y):
        i = int((x - self.x0) // self.cell)
        j = int((y - self.y0) // self.cell)
        if 0 <= i < self.nx and 0 <= j < self.ny:
            return i, j
        return None

    def points(self, z=0.0):
        X, Y = np.meshgrid(self.xs, self.ys)
        return np.stack([X, Y, np.full_like(X, z)], axis=-1)


def compute_roi(vg: VehicleGeom, roi: ROI, cell: float) -> RoiGrid:
    from scipy import ndimage

    lo, hi = vg.bbox
    if roi.mode == 'polygon' and len(roi.polygon) >= 3:
        P = np.array(roi.polygon, float)
        xmin, ymin = min(P[:, 0].min(), lo[0]) - 0.3, min(P[:, 1].min(), lo[1]) - 0.3
        xmax, ymax = max(P[:, 0].max(), hi[0]) + 0.3, max(P[:, 1].max(), hi[1]) + 0.3
    else:
        m = roi.distance + 0.25
        xmin, ymin, xmax, ymax = lo[0] - m, lo[1] - m, hi[0] + m, hi[1] + m
    nx = int(math.ceil((xmax - xmin) / cell))
    ny = int(math.ceil((ymax - ymin) / cell))
    x0, y0 = xmin, ymin
    body = vg.body.plan_mask(x0, y0, cell, nx, ny)
    src = body.copy()
    if roi.measured_from == 'bbox' and body.any():
        jj, ii = np.nonzero(body)
        src = np.zeros_like(body)
        src[jj.min():jj.max() + 1, ii.min():ii.max() + 1] = True
    if roi.corners == 'square':
        dist = ndimage.distance_transform_cdt(~src, metric='chessboard').astype(float) * cell
    else:
        dist = ndimage.distance_transform_edt(~src) * cell
    if roi.mode == 'polygon' and len(roi.polygon) >= 3:
        inside = _poly_mask(np.array(roi.polygon, float), x0, y0, cell, nx, ny)
        region = inside
        boundary = [np.vstack([np.array(roi.polygon, float), np.array(roi.polygon[:1], float)])]
    else:
        region = dist <= roi.distance + 1e-9
        boundary = chain(iso_segments(dist, roi.distance, x0, y0, cell), tol=1e-5)
    roi_mask = region & ~body if roi.exclude_under_body else region
    sdf = np.where(body, -ndimage.distance_transform_edt(body) * cell, dist)
    outline = chain(iso_segments(-sdf, -cell * 0.5, x0, y0, cell), tol=1e-5)
    area = float(roi_mask.sum()) * cell * cell
    return RoiGrid(x0, y0, cell, nx, ny, roi_mask, body, dist, boundary, outline, area)


def _poly_mask(P, x0, y0, cell, nx, ny):
    from PyQt6.QtGui import QImage, QPainter, QPolygonF, QColor
    from PyQt6.QtCore import QPointF, Qt
    img = QImage(nx, ny, QImage.Format.Format_Grayscale8)
    img.fill(0)
    p = QPainter(img)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QColor(255, 255, 255))
    p.drawPolygon(QPolygonF([QPointF((x - x0) / cell, (y - y0) / cell) for x, y in P]))
    p.end()
    ptr = img.constBits()
    ptr.setsize(img.sizeInBytes())
    return (np.frombuffer(ptr, np.uint8).reshape(ny, img.bytesPerLine())[:, :nx] > 127).copy()


# ------------------------------------------------------------------ per-camera evaluation
@dataclass
class CamEval:
    uid: str
    tag: str
    slot: int
    vis: np.ndarray            # (ny,nx) bool
    dens: np.ndarray           # (ny,nx) float32, 0 where not visible
    focus: np.ndarray          # (ny,nx) bool
    field: np.ndarray          # (ny,nx) float32 margin (positive = visible)
    segments: np.ndarray       # footprint outline segments (S,2,2) world
    border: list = field(default_factory=list)   # projected image border on the ground (polylines)


def camera_frames(cam: Camera):
    R = rot_zyx(cam.yaw, cam.pitch, cam.roll)       # body -> world
    W2O = BODY_TO_OPTICAL @ R.T                     # world -> optical
    return R, W2O


def eval_points(P: np.ndarray, cam: Camera, intr: Intrinsics, cube, R_override=None, pos_override=None):
    """Visibility, density, focus and margin for world points P (K,3)."""
    pos = cam.pos() if pos_override is None else pos_override
    if R_override is None:
        _, W2O = camera_frames(cam)
    else:
        W2O = BODY_TO_OPTICAL @ R_override.T
    d = P - pos
    dist = np.linalg.norm(d, axis=1)
    opt = d @ W2O.T
    u, v, theta, ok = intr.project_dirs(opt)
    depth, _ = cube.lookup(d)
    tol = 0.02 + 0.01 * dist
    clear = depth > dist - tol
    vis = ok & clear
    dens = np.where(vis, intr.density_px_per_cm(theta, dist), 0.0).astype(np.float32)
    focus = (dist >= intr.focus_near) & (dist <= intr.focus_far)
    m = intr.margin(u, v, theta)
    m = np.where(intr.projection == 'rectilinear', np.where(opt[:, 2] > 0, m, -1.0), m)
    occ_m = np.clip((np.where(np.isfinite(depth), depth, 1e3) - (dist - tol)) / np.maximum(dist, 0.3), -1, 1)
    fieldv = np.minimum(m, occ_m * 3).astype(np.float32)
    return vis, dens, focus, fieldv


def ground_border(cam: Camera, intr: Intrinsics, far: float, samples: int = 36):
    """Image border projected onto the ground plane (polyline, clipped to `far` metres)."""
    R, W2O = camera_frames(cam)
    O2W = W2O.T
    w, h = intr.width, intr.height
    t = np.linspace(0, 1, samples, endpoint=False)
    us = np.concatenate([t * w, np.full(samples, w), (1 - t) * w, np.zeros(samples)])
    vs = np.concatenate([np.zeros(samples), t * h, np.full(samples, h), (1 - t) * h])
    d_opt, theta = intr.unproject(us, vs)
    valid = theta <= intr.max_theta
    dw = d_opt @ O2W.T
    pos = cam.pos()
    pts = []
    for dd, ok in zip(dw, valid):
        if not ok:
            pts.append(None)
            continue
        if dd[2] < -1e-6:
            s = -pos[2] / dd[2]
            g = pos + dd * s
            hd = math.hypot(g[0] - pos[0], g[1] - pos[1])
            if hd <= far:
                pts.append(g)
                continue
        hl = math.hypot(dd[0], dd[1]) or 1.0
        pts.append(np.array([pos[0] + dd[0] / hl * far, pos[1] + dd[1] / hl * far, 0.0]))
    lines, cur = [], []
    for p in pts + pts[:1]:
        if p is None:
            if len(cur) > 1:
                lines.append(np.array(cur))
            cur = []
        else:
            cur.append(p)
    if len(cur) > 1:
        lines.append(np.array(cur))
    return lines


# ------------------------------------------------------------------ results
@dataclass
class Seam:
    a: str
    b: str
    angle: float          # degrees, azimuth from the body centre
    auto: bool
    density: float


@dataclass
class AnalysisResult:
    grid: RoiGrid
    cams: list
    best: np.ndarray
    best_dens: np.ndarray
    count: np.ndarray
    metrics: dict
    per_cam: dict
    hist: tuple
    seams: list
    sector: np.ndarray
    blend: np.ndarray
    centre: tuple
    ms: float
    z: float = 0.0
    cubes: dict = field(default_factory=dict)
    obstacles: dict = field(default_factory=dict)
    variant_uid: str = ''
    cell: float = 0.05
    terrain: object = None      # core.terrain.Terrain the ground followed (None: flat ground)

    def cam_index(self, uid):
        for i, c in enumerate(self.cams):
            if c.uid == uid:
                return i
        return -1

    def query(self, x, y):
        """Hover info at a ground point: list of (cam, density, range, off-axis deg)."""
        g = self.grid.cell_of(x, y)
        if g is None:
            return None
        i, j = g
        out = []
        for c in self.cams:
            if c.vis[j, i]:
                out.append((c, float(c.dens[j, i])))
        return {'cell': (i, j), 'in_roi': bool(self.grid.roi[j, i]), 'under_body': bool(self.grid.body[j, i]),
                'cams': sorted(out, key=lambda t: -t[1]), 'best': int(self.best[j, i])}


class Engine:
    """Stateful analysis engine; keeps vehicle geometry, ROI grid and cube maps cached.

    One engine serves the UI thread and the analysis worker at once: the UI asks about the live project while the
    worker analyses a snapshot, which can hold a different car (just after an import or an undo).  So the geometry in
    use (``vg``, what ``cube()`` looks into) is per thread, built geometries are shared, and the caches are locked.
    The geometry in use also carries the housings of the variant it was asked for (vehicle_geom(project, variant)),
    so cube() looks into the body, the rig and that variant's housings."""

    GEOMS_KEPT = 3
    HOUSED_KEPT = 8

    def __init__(self):
        self.cubes = CubeCache(96)
        self._geoms = {}                    # key -> VehicleGeom, the last few (each holds its mesh, so ids stay unique)
        self._glock = threading.RLock()
        self._tl = threading.local()
        self._latest = (None, None)         # (key, geometry) set last by any thread, for a thread that never asked
        self._rois = {}
        self._housed = {}                   # (geometry key, housing key) -> the geometry with those housings

    def _current(self):
        """(key, geometry) in use on this thread, read as one pair."""
        return getattr(self._tl, 'cur', None) or self._latest

    @property
    def vg(self) -> VehicleGeom | None:
        return self._current()[1]

    def _use(self, key, vg):
        self._tl.cur = self._latest = (key, vg)

    def share(self) -> 'Engine':
        """An engine for another worker: the same cube maps and built geometries, its own ROI and current geometry."""
        e = Engine()
        e.cubes, e._geoms, e._glock, e._housed = self.cubes, self._geoms, self._glock, self._housed
        e._latest = self._latest
        return e

    def vehicle_geom(self, project: Project, variant: Variant | None = None) -> VehicleGeom | None:
        """The vehicle geometry (body, rig) with the housings of variant (default: the active one) blocking the
        cameras; made the one in use on this thread. Without housings (switched off or none) the plain geometry."""
        m = project.vehicle.mesh
        if m is None or m.n_tris == 0:
            return None
        rig = [r for r in getattr(project, 'rig', ()) if r.visible]
        key = (id(m), m.n_tris, project.vehicle.occluder_target, tuple(p.role for p in m.parts),
               id(m.face_zone), tuple((z.name, z.kind) for z in m.zones),      # zones are copied into the occluder
               rig_key(rig))
        st = H.settings_of(project)
        hk = H.key(project, variant, st)
        full = (key, hk)
        cur_key, cur = self._current()
        if full == cur_key:
            return cur
        with self._glock:
            vg = self._geoms.pop(key, None)
            if vg is None:
                same = next((g for k, g in self._geoms.items() if k[:-1] == key[:-1]), None)  # only the rig changed
                vg = prepare_vehicle(m, project.vehicle.occluder_target, rig,
                                     (same.body, same.max_dev) if same is not None else None)
            self._geoms[key] = vg                                               # most recent last
            while len(self._geoms) > self.GEOMS_KEPT:
                self._geoms.pop(next(iter(self._geoms)))
            if hk is not None:
                hv = self._housed.pop(full, None)
                if hv is None or hv.base is not vg:
                    hs = H.housing_set(project, variant, st, hk)
                    hv = vg.with_housings(hs) if hs is not None else vg
                self._housed[full] = hv
                while len(self._housed) > self.HOUSED_KEPT:
                    self._housed.pop(next(iter(self._housed)))
                vg = hv
        self._use(full, vg)
        return vg

    def roi_grid(self, project: Project, cell=None, variant: Variant | None = None) -> RoiGrid | None:
        vg = self.vehicle_geom(project, variant)
        if vg is None:
            return None
        cell = cell or project.analysis.cell_m
        r = project.roi
        key = (vg.version, cell, r.mode, round(r.distance, 4), r.measured_from, r.corners, r.exclude_under_body,
               tuple(map(tuple, r.polygon)))
        with self._glock:
            grid = self._rois.pop(key, None)
            if grid is None:
                grid = compute_roi(vg, r, cell)
            self._rois[key] = grid
            while len(self._rois) > self.GEOMS_KEPT:
                self._rois.pop(next(iter(self._rois)))
        return grid

    def cube(self, pos, vg: VehicleGeom | None = None):
        """The occlusion cube map at pos: the body and rig, with the housings of the geometry in use (vg, else this
        thread's) composited in, the housing anchored at pos left out."""
        return self.cubes.get((vg or self.vg).occ, pos)

    # --------------------------------------------------------------
    def analyse(self, project: Project, variant: Variant | None = None, cell=None, z: float | None = None,
                cameras=None, progress=None, body_pose=None, terrain=None) -> AnalysisResult | None:
        """body_pose = (R, t): the vehicle body (and its cameras) moved rigidly in the world, e.g. a load case.
        terrain: the road the ground follows (core.terrain.Terrain); None: the project's road when it has one switched
        on (analysis.terrain), else flat ground; False: flat ground. On a road the ground points sit on it, the car
        stands on it (its stance, after body_pose) and the road itself can hide the ground from a camera."""
        t0 = time.perf_counter()
        variant = variant or project.variant
        grid = self.roi_grid(project, cell, variant)
        if grid is None:
            return None
        vg = self.vehicle_geom(project, variant)        # the variant's housings block its cameras
        if z is None:
            z = {'ground': 0.0, 'pole05': 0.5, 'pole10': 1.0, 'child': 0.9}.get(project.analysis.obstacle_mode, 0.0)
        cams = [c for c in (cameras if cameras is not None else variant.cameras) if c.visible]
        ter = None
        if terrain is not False:
            from . import terrain as TR
            ter = terrain if terrain is not None else TR.active(project)
        P = grid.points(z).reshape(-1, 3)
        Pw = None
        if ter is not None:
            Pw = P.copy()
            Pw[:, 2] += ter.height(Pw[:, 0], Pw[:, 1])                    # on the road
            P = Pw
            body_pose = ter.pose(body_pose)                              # the car standing on it
        if body_pose is not None:
            Rb, tb = body_pose
            P = (P - np.asarray(tb, float)) @ np.asarray(Rb, float)      # world ground -> body frame
        shape = (grid.ny, grid.nx)
        evals = []
        cubes = {}
        for k, cam in enumerate(cams):
            intr = resolve(project.module_for(cam), project.lens_for(cam))
            cube = self.cube(cam.pos(), vg)
            cubes[cam.uid] = cube
            vis, dens, focus, fld = eval_points(P, cam, intr, cube)
            if ter is not None:                                          # the road itself in the way
                C = cam.pos() if body_pose is None else \
                    np.asarray(body_pose[0], float) @ cam.pos() + np.asarray(body_pose[1], float)
                clear = ter.clear(C, Pw, on_surface=z <= 1e-9)
                vis = vis & clear
                fld = np.where(clear, fld, np.minimum(fld, -0.2)).astype(np.float32)
            vis = vis.reshape(shape) & ~grid.body
            dens = np.where(vis, dens.reshape(shape), 0).astype(np.float32)
            fld = fld.reshape(shape)
            fld = np.where(grid.body, -1.0, fld)
            from scipy import ndimage
            # light smoothing so outlines follow the footprint, not the 5 cm cell steps
            segs = iso_segments(ndimage.gaussian_filter(np.clip(fld, -0.3, 0.3), 0.9), 0.0, grid.x0, grid.y0, grid.cell)
            # keep segments inside the ROI neighbourhood
            if len(segs):
                mid = segs.mean(axis=1)
                ii = np.clip(((mid[:, 0] - grid.x0) / grid.cell).astype(int), 0, grid.nx - 1)
                jj = np.clip(((mid[:, 1] - grid.y0) / grid.cell).astype(int), 0, grid.ny - 1)
                segs = segs[grid.dist[jj, ii] <= (project.roi.distance if project.roi.mode == 'offset' else 1e9) + grid.cell]
            ce = CamEval(cam.uid, cam.tag, cam.slot, vis, dens, focus.reshape(shape), fld, segs)
            ce.border = ground_border(cam, intr, project.roi.distance + 2.5)
            evals.append(ce)
            if progress:
                progress((k + 1) / max(len(cams), 1) * 0.8)
        res = self._aggregate(project, variant, grid, evals, cams, z)
        res.cubes = cubes
        res.ms = (time.perf_counter() - t0) * 1000
        res.variant_uid = variant.uid
        res.cell = grid.cell
        res.obstacles = self.obstacle_visibility(project, cams, cubes, body_pose, ter)
        res.terrain = ter
        return res

    def _aggregate(self, project, variant, grid, evals, cams, z) -> AnalysisResult:
        roi = grid.roi
        a = grid.cell ** 2
        n = len(evals)
        if n:
            D = np.stack([e.dens for e in evals])          # (n, ny, nx)
            V = np.stack([e.vis for e in evals])
            F = np.stack([e.focus for e in evals])
            count = V.sum(axis=0).astype(np.int8)
            best = np.where(count > 0, np.argmax(D, axis=0), -1).astype(np.int8)
            best_d = D.max(axis=0)
        else:
            D = np.zeros((0,) + roi.shape, np.float32)
            V = np.zeros((0,) + roi.shape, bool)
            F = V
            count = np.zeros(roi.shape, np.int8)
            best = np.full(roi.shape, -1, np.int8)
            best_d = np.zeros(roi.shape, np.float32)
        roi_n = max(int(roi.sum()), 1)
        covered = roi & (count > 0)
        blind = roi & (count == 0)
        mind = project.analysis.min_density
        tgt = project.analysis.target_density
        in_focus = covered & (np.take_along_axis(F, np.maximum(best, 0)[None].astype(np.int64), 0)[0] if n else covered)
        metrics = {
            'roi_area': grid.area, 'cells': int(roi.sum()), 'blind_m2': float(blind.sum() * a),
            'covered_pct': 100.0 * covered.sum() / roi_n, 'multi_pct': 100.0 * (roi & (count >= 2)).sum() / roi_n,
            'min_density': float(best_d[covered].min()) if covered.any() else 0.0,
            'in_focus_pct': 100.0 * in_focus.sum() / roi_n,
            'min_ok_pct': 100.0 * (roi & (best_d >= mind)).sum() / roi_n,
            'target_ok_pct': 100.0 * (roi & (best_d >= tgt)).sum() / roi_n,
            'cell': grid.cell, 'z': z,
        }
        per = {}
        for i, e in enumerate(evals):
            mine = roi & V[i]
            unique = mine & (count == 1)
            others = np.delete(D, i, axis=0).max(axis=0) if n > 1 else np.zeros_like(best_d)
            if_rm_blind = float((roi & (count - V[i].astype(np.int8) == 0)).sum() * a)
            if_rm_below = float((roi & (others < mind)).sum() * a)
            bestmine = roi & (best == i)
            ov = {}
            for j, f in enumerate(evals):
                if j != i:
                    s = float((mine & V[j]).sum() * a)
                    if s > 0:
                        ov[f.uid] = s
            per[e.uid] = {'seen_m2': float(mine.sum() * a), 'unique_m2': float(unique.sum() * a), 'overlaps': ov,
                          'min_dens': float(D[i][bestmine].min()) if bestmine.any() else 0.0,
                          'best_m2': float(bestmine.sum() * a),
                          'if_removed_blind': if_rm_blind, 'if_removed_below': if_rm_below}
        vals = best_d[covered]
        areas, _ = np.histogram(vals, bins=HIST_EDGES)
        hist = (HIST_EDGES, areas.astype(float) * a, float(blind.sum() * a))
        centre = (float((grid.xs[grid.body.any(axis=0)].min() + grid.xs[grid.body.any(axis=0)].max()) / 2)
                  if grid.body.any() else 0.0, 0.0)
        seams, sector, blend = compute_seams(grid, evals, variant, centre, D, V)
        return AnalysisResult(grid, evals, best, best_d, count, metrics, per, hist, seams, sector, blend, centre, 0.0, z)

    # --------------------------------------------------------------
    def obstacle_visibility(self, project, cams, cubes, body_pose=None, terrain=None):
        """body_pose: the body's pose in the world (on a road: its stance included); terrain: the road the obstacles
        stand on (core.terrain.Terrain; None: flat ground)."""
        out = {}
        for ob in project.obstacles:
            zs = np.linspace(0.05, ob.height, 8)
            P = np.stack([np.full_like(zs, ob.x), np.full_like(zs, ob.y), zs], 1)
            if terrain is not None:
                P[:, 2] += float(terrain.height(ob.x, ob.y))
            Pw = P
            if body_pose is not None:       # obstacles stand in the world; the cameras and occluders are in the body frame
                P = (P - np.asarray(body_pose[1], float)) @ np.asarray(body_pose[0], float)
            seen = {}
            for cam in cams:
                intr = resolve(project.module_for(cam), project.lens_for(cam))
                vis, dens, _, _ = eval_points(P, cam, intr, cubes[cam.uid])
                if terrain is not None:
                    C = cam.pos() if body_pose is None else \
                        np.asarray(body_pose[0], float) @ cam.pos() + np.asarray(body_pose[1], float)
                    vis = vis & terrain.clear(C, Pw, on_surface=False)
                if vis.any():
                    seen[cam.uid] = (float(vis.mean()), float(dens[vis].max()))
            out[ob.uid] = seen
        return out

    def eval_single(self, project, cam: Camera, grid: RoiGrid, z=0.0, cell_points=None, R_override=None, pos=None):
        """Visibility/density of one camera on the ROI cells (flattened), for studies."""
        intr = resolve(project.module_for(cam), project.lens_for(cam))
        P = cell_points if cell_points is not None else grid.points(z)[grid.roi]
        cube = self.cube(cam.pos() if pos is None else pos)
        vis, dens, focus, _ = eval_points(P, cam, intr, cube, R_override=R_override, pos_override=pos)
        return vis, dens


# ------------------------------------------------------------------ seams and stitching
def _azimuth(grid, centre):
    X, Y = np.meshgrid(grid.xs - centre[0], grid.ys - centre[1])
    return np.arctan2(Y, X), np.hypot(X, Y)


def compute_seams(grid, evals, variant, centre, D, V):
    n = len(evals)
    sector = np.full((grid.ny, grid.nx), -1, np.int8)
    blend = np.zeros((grid.ny, grid.nx), bool)
    if n == 0:
        return [], sector, blend
    az_cells, rad = _azimuth(grid, centre)
    best = np.argmax(D, axis=0) if n else None
    cover = V.any(axis=0)
    azs = []
    for i, e in enumerate(evals):
        m = grid.roi & cover & (best == i)
        if m.sum() > 4:
            w = D[i][m]
            ang = np.arctan2((np.sin(az_cells[m]) * w).sum(), (np.cos(az_cells[m]) * w).sum())
        else:
            cam = variant.cam(e.uid)
            ang = math.atan2(cam.y - centre[1], cam.x - centre[0]) if cam else 0.0
        azs.append(ang)
    order = list(np.argsort(azs))
    seams = []
    if n == 1:
        sector[grid.roi] = 0
        return seams, sector, blend
    rmax = float(rad[grid.roi].max()) if grid.roi.any() else 5.0
    radii = np.arange(0.3, rmax + grid.cell, grid.cell)
    angles_out = []
    for k in range(n):
        ia, ib = order[k], order[(k + 1) % n]
        a0 = azs[ia]
        span = (azs[ib] - a0) % (2 * math.pi)
        if span < 1e-3:
            span = 2 * math.pi / n
        key = f'{evals[ia].uid}|{evals[ib].uid}'
        manual = variant.seams.get(key)
        if manual is not None:
            ang = math.radians(manual)
            auto = False
        else:
            cand = a0 + span * np.linspace(0.08, 0.92, 41)
            X = centre[0] + np.cos(cand)[:, None] * radii[None]
            Y = centre[1] + np.sin(cand)[:, None] * radii[None]
            ii = ((X - grid.x0) / grid.cell).astype(int)
            jj = ((Y - grid.y0) / grid.cell).astype(int)
            ok = (ii >= 0) & (ii < grid.nx) & (jj >= 0) & (jj < grid.ny)
            ii, jj = np.clip(ii, 0, grid.nx - 1), np.clip(jj, 0, grid.ny - 1)
            inroi = ok & grid.roi[jj, ii]
            both = V[ia][jj, ii] & V[ib][jj, ii] & inroi
            dmin = np.minimum(D[ia][jj, ii], D[ib][jj, ii])
            frac = both.sum(1) / np.maximum(inroi.sum(1), 1)
            dsc = np.where(both, np.clip(dmin, 0, 10), 0).sum(1) / np.maximum(inroi.sum(1), 1) / 10
            score = frac + 0.25 * dsc - 0.02 * np.abs(np.linspace(-1, 1, len(cand)))
            ang = float(cand[int(np.argmax(score))])
            auto = True
        angles_out.append(ang)
        # density along the seam
        X = centre[0] + math.cos(ang) * radii
        Y = centre[1] + math.sin(ang) * radii
        ii = np.clip(((X - grid.x0) / grid.cell).astype(int), 0, grid.nx - 1)
        jj = np.clip(((Y - grid.y0) / grid.cell).astype(int), 0, grid.ny - 1)
        inr = grid.roi[jj, ii]
        dd = np.maximum(D[ia][jj, ii], D[ib][jj, ii])
        dd = np.where(V[ia][jj, ii] & V[ib][jj, ii], np.minimum(D[ia][jj, ii], D[ib][jj, ii]), dd)
        sd = float(np.median(dd[inr])) if inr.any() else 0.0
        seams.append(Seam(evals[ia].uid, evals[ib].uid, math.degrees(math.atan2(math.sin(ang), math.cos(ang))), auto, sd))
    # sectors: camera order[k+1] owns [seam_k, seam_{k+1})
    for k in range(n):
        s0 = angles_out[k]
        s1 = angles_out[(k + 1) % n]
        span = (s1 - s0) % (2 * math.pi)
        rel = (az_cells - s0) % (2 * math.pi)
        own = rel < span
        sector[own] = order[(k + 1) % n]
    half = variant.blend_m / 2
    for k, ang in enumerate(angles_out):
        X, Y = np.meshgrid(grid.xs - centre[0], grid.ys - centre[1])
        along = X * math.cos(ang) + Y * math.sin(ang)
        perp = np.abs(-X * math.sin(ang) + Y * math.cos(ang))
        blend |= (along > 0) & (perp < half)
    sector = np.where(grid.roi | grid.body, sector, -1).astype(np.int8)
    return seams, sector, blend


# ------------------------------------------------------------------ surface helpers
def surface_frame(mesh: Mesh, p):
    """Closest surface point, outward normal and two tangents (horizontal first) at p."""
    q, fi, d = mesh.closest_point(np.asarray(p, float), mesh.occluder_faces())
    n = mesh.normals[fi] if fi >= 0 else np.array([0.0, 0.0, 1.0])
    t1 = np.cross(np.array([0.0, 0.0, 1.0]), n)
    if np.linalg.norm(t1) < 1e-6:
        t1 = np.array([1.0, 0.0, 0.0])
    t1 /= np.linalg.norm(t1)
    t2 = np.cross(n, t1)
    return q, n, t1, t2, fi


def slide_on_surface(mesh: Mesh, cam: Camera, direction, s: float):
    """Move a camera s metres along the body surface in `direction` (tangent), re-snapped with its standoff."""
    q, n, _, _, fi = surface_frame(mesh, cam.pos())
    target = q + np.asarray(direction, float) * s
    o = target + n * 0.3
    t, f = mesh.raycast(o[None], -n[None], faces_mask=mesh.occluder_faces())
    if f[0] < 0:
        return cam.pos() + np.asarray(direction, float) * s
    hit = o - n * t[0]
    nn = mesh.normals[f[0]]
    if nn @ n < 0:
        nn = -nn
    return hit + nn * (cam.standoff_mm / 1000.0)


# ------------------------------------------------------------------ helpers for the UI
def describe_location(x, y, project: Project) -> str:
    """Rough body-relative description of a ground point (for worst-case tables and issues)."""
    meas = project.vehicle.measured()
    xf, xr = meas.get('front_x', 3.0), meas.get('rear_x', -0.8)
    hw = meas.get('Width', 1.7) / 2
    wc = project.vehicle.wheel_centres
    side = 'left' if y > 0 else 'right'
    if x > xf - 0.05:
        if abs(y) < hw * 0.6:
            return 'below the front bumper' if x - xf < 0.25 else 'ahead of the front bumper'
        return f'front-{side} corner'
    if x < xr + 0.05:
        if abs(y) < hw * 0.6:
            return 'below the rear bumper' if xr - x < 0.25 else 'behind the rear bumper'
        return f'rear-{side} corner'
    for k, v in wc.items():
        if abs(x - v[0]) < 0.45 and np.sign(y) == np.sign(v[1]):
            return f"{side} {'front' if k[0] == 'F' else 'rear'} wheel arch"
    return f'{side} sill, beside the doors'
