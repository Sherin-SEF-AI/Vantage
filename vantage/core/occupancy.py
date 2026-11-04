"""3D occupancy-grid support: per voxel of an Occ3D / OpenOccupancy-style grid, how much the rig gives a 3D occupancy
network (OccNet, SurroundOcc, FB-OCC, BEVFormer-Occ) to work with there:

- camera rays: which cameras see the voxel's centre (inside the lens, not hidden by the vehicle, the rig or a housing:
  analysis.eval_points with the engine's occlusion cube maps, as the coverage analysis), and how many backbone feature
  cells of stride × stride pixels the voxel covers in each: its projected area a² (|n_x| + |n_y| + |n_z|) (the
  silhouette of a cube of side a seen along the unit ray n) times (pixels per radian / range)², over stride². Under one
  feature cell summed over the cameras the voxel is feature-starved;
- lidar returns: the beam × azimuth samples of each spinning lidar's sweep that pass through the voxel (its channels'
  elevations in the voxel's elevation span, its azimuth span over the lidar's h_res), or for a solid-state /
  non-repetitive lidar its points per sweep spread evenly over its field; zero where the vehicle stops the ray (the
  cube at the lidar, its own mount within 15 cm left out), beyond its range, or under the ground. Under one sample per
  sweep the voxel is lidar-sparse;
- stereo: the best camera pair's 1σ triangulation depth uncertainty at the voxel centre (research_maps.pair_sigma,
  0.5 px), NaN where fewer than two cameras see it.

The grid (defaults: Occ3D-nuScenes, x, y ∈ [−40, 40] m, z ∈ [−1, 5.4] m, 0.4 m voxels: 200 × 200 × 16) is in the
vehicle frame (ISO 8855: x forward, y left, z up, metres) centred on the rear-axle centre on the ground (centre=
'rear_axle', the nuScenes ego frame's origin and axes, which Occ3D uses) or on the vehicle's plan footprint
(centre='footprint'). Arrays are (nz, ny, nx): index [k, j, i] is the voxel with centre (x0 + (i + ½) a, y0 +
(j + ½) a, z0 + (k + ½) a). The ground is the plane z = 0: voxels whose centre is below it see nothing.

Free-space assumption: nothing but the vehicle stops a ray (no other objects, no occupancy in front of the voxel).
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field

import numpy as np

from .analysis import eval_points
from .research_maps import _cams, _cube, _footprint_centre, _intr, _project, pair_sigma, px_per_rad

SCHEMA = 'vantage.occupancy_density/1'
BANDS = ((-1.0, 0.0), (0.0, 1.0), (1.0, 2.0), (2.0, 5.4))
RINGS = ((0.0, 10.0), (10.0, 20.0), (20.0, 30.0), (30.0, 40.0))


@dataclass
class VoxelDensity:
    x0: float
    y0: float
    z0: float
    voxel_m: float
    nx: int
    ny: int
    nz: int
    stride: int
    centre: str
    cam_rays: np.ndarray            # (nz, ny, nx) cameras that see the voxel centre
    feats: np.ndarray               # backbone feature cells covered, summed over the cameras
    best_feats: np.ndarray          # the best single camera's
    lidar_pts: np.ndarray           # lidar samples through the voxel per sweep, summed over the lidars
    sigma_m: np.ndarray             # best pair's depth sigma (m), NaN with fewer than two cameras
    body: np.ndarray                # voxel centre inside the vehicle's box
    below: np.ndarray               # voxel centre under the ground
    tags: list = field(default_factory=list)
    lidars: list = field(default_factory=list)
    ms: float = 0.0

    @property
    def shape(self):
        return (self.nz, self.ny, self.nx)

    def centres(self):
        """(xs, ys, zs) of the voxel centres along each axis."""
        a = self.voxel_m
        return (self.x0 + (np.arange(self.nx) + 0.5) * a, self.y0 + (np.arange(self.ny) + 0.5) * a,
                self.z0 + (np.arange(self.nz) + 0.5) * a)

    @property
    def free(self):
        """The voxels that count: above the ground, not inside the vehicle."""
        return ~self.body & ~self.below

    @property
    def starved(self):
        return self.free & (self.feats < 1.0)

    @property
    def sparse(self):
        return self.free & (self.lidar_pts < 1.0)

    def summary(self) -> dict:
        xs, ys, zs = self.centres()
        Z, Y, X = np.meshgrid(zs, ys, xs, indexing='ij')
        R = np.maximum(np.abs(X - (self.x0 + self.nx * self.voxel_m / 2)), np.abs(Y - (self.y0 + self.ny * self.voxel_m / 2)))
        free = self.free

        def stats(m):
            n = int(m.sum())
            if not n:
                return {'voxels': 0}
            seen = m & (self.cam_rays > 0)
            s = self.sigma_m[m & np.isfinite(self.sigma_m)]
            return {'voxels': n,
                    'camera_seen_pct': round(100.0 * float(seen.sum()) / n, 2),
                    'feature_starved_pct': round(100.0 * float((m & (self.feats < 1.0)).sum()) / n, 2),
                    'lidar_sparse_pct': round(100.0 * float((m & (self.lidar_pts < 1.0)).sum()) / n, 2),
                    'stereo_pct': round(100.0 * float(len(s)) / n, 2),
                    'median_sigma_m': float(np.median(s)) if len(s) else None,
                    'median_feats_where_seen': float(np.median(self.feats[seen])) if seen.any() else 0.0,
                    'median_lidar_pts_where_hit': (float(np.median(self.lidar_pts[m & (self.lidar_pts > 0)]))
                                                   if (m & (self.lidar_pts > 0)).any() else 0.0)}
        bands = [{'z_m': list(b), **stats(free & (Z >= b[0]) & (Z < b[1]))} for b in BANDS]
        rings = [{'range_m': list(r), 'range_is': 'max(|x − xc|, |y − yc|): square rings about the grid centre',
                  **stats(free & (R >= r[0]) & (R < r[1]))} for r in RINGS]
        return {'grid': {'shape': list(self.shape), 'voxel_m': self.voxel_m, 'x0': self.x0, 'y0': self.y0,
                         'z0': self.z0, 'stride': self.stride, 'centre': self.centre},
                'all': stats(free), 'height_bands': bands, 'range_rings': rings, 'cameras': self.tags,
                'lidars': self.lidars, 'ms': round(self.ms, 1)}


def voxel_area_px(cam, intr, P, a):
    """Projected area (px²) in the camera of axis-aligned cubes of side a centred at P (K, 3): a² (|n_x| + |n_y| +
    |n_z|) for the unit ray n, times (pixels per radian / range)². No visibility test."""
    d = np.asarray(P, float) - cam.pos()
    r = np.linalg.norm(d, axis=-1)
    n = d / np.maximum(r, 1e-9)[..., None]
    _, _, th, _ = _project(cam, intr, P)
    pr = px_per_rad(intr, th)
    return a * a * np.abs(n).sum(axis=-1) * (pr / np.maximum(r, 1e-6)) ** 2


def lidar_samples(s, model, pos, P, a, cube=None):
    """Expected lidar samples per sweep passing through axis-aligned cubes of side a centred at P (K, 3) from a lidar
    at pos: spinning, the channels whose elevation lies in the cube's elevation span times its azimuth span over
    h_res; otherwise the points per sweep spread evenly over the h_fov × v_fov. Zero beyond range, outside the field,
    or where the vehicle (cube, own mount within 15 cm left out) stops the ray first."""
    from . import sensors as S
    P = np.asarray(P, float)
    q = S.to_sensor(s, P)                               # sensor frame
    rh = np.hypot(q[:, 0], q[:, 1])
    r = np.linalg.norm(q, axis=1)
    az = np.degrees(np.arctan2(q[:, 1], q[:, 0]))
    el = np.degrees(np.arctan2(q[:, 2], np.maximum(rh, 1e-9)))
    d = P - pos
    phi = np.arctan2(d[:, 1], d[:, 0])                  # the cube's axes are the vehicle's
    wh = a * (np.abs(np.cos(phi)) + np.abs(np.sin(phi)))      # horizontal width seen from the lidar
    elr = np.radians(el)
    dphi = np.degrees(wh / np.maximum(rh, 1e-6))
    delv = np.degrees((a * np.cos(elr) + wh * np.abs(np.sin(elr))) / np.maximum(r, 1e-6))
    rng = float(model.range_max_m or model.range_m or 100.0)
    ok = (r <= rng) & (r >= float(model.range_min_m or 0.0))
    hf = float(model.h_fov or 360.0)
    if hf < 359.0:
        ok &= np.abs(az) <= hf / 2
    if model.scan == 'spinning' and model.channels:
        els = np.sort(np.asarray(model.elevations(), float))
        lo = np.searchsorted(els, el - delv / 2, side='left')
        hi = np.searchsorted(els, el + delv / 2, side='right')
        ch = (hi - lo).astype(float)
        n = ch * dphi / max(float(model.h_res or 0.2), 1e-6)
    else:
        vmin, vmax = float(model.v_min), float(model.v_max)
        ok &= (el >= vmin) & (el <= vmax)
        per_sweep = float(model.points_per_s or 0.0) / max(float(model.rate_hz or 10.0), 1e-6)
        n = per_sweep * dphi * delv / max(hf * max(vmax - vmin, 1e-6), 1e-6)
    if cube is not None and ok.any():
        depth, _ = cube.lookup(d[ok])
        depth = np.asarray(depth, float)
        depth[depth <= 0.15] = np.inf
        blocked = depth < r[ok] - 0.05
        okk = ok.copy()
        okk[np.nonzero(ok)[0][blocked]] = False
        ok = okk
    return np.where(ok, n, 0.0)


def voxel_density(project, engine, extent_m=40.0, z_range=(-1.0, 5.4), voxel_m=0.4, stride=16, result=None,
                  centre='rear_axle', sigma_px=0.5, variant=None) -> VoxelDensity:
    """The occupancy grid's camera, lidar and stereo support (VoxelDensity), computed slice by slice in z."""
    t0 = time.perf_counter()
    if centre not in ('rear_axle', 'footprint'):
        raise ValueError('centre must be "rear_axle" or "footprint"')
    a = float(voxel_m)
    nx = ny = max(int(round(2 * extent_m / a)), 1)
    nz = max(int(round((z_range[1] - z_range[0]) / a)), 1)
    fc, (lo, hi) = _footprint_centre(project)
    c = (0.0, 0.0) if centre == 'rear_axle' else (float(fc[0]), float(fc[1]))
    x0, y0, z0 = c[0] - nx * a / 2, c[1] - ny * a / 2, float(z_range[0])
    xs = x0 + (np.arange(nx) + 0.5) * a
    ys = y0 + (np.arange(ny) + 0.5) * a
    zs = z0 + (np.arange(nz) + 0.5) * a
    X, Y = np.meshgrid(xs, ys)
    m = project.vehicle.mesh
    top = float(m.vertices[:, 2].max()) if m is not None and m.n_tris else 0.0
    cams = _cams(project, variant)
    intrs = [_intr(project, cam) for cam in cams]
    cubes = [_cube(engine, project, cam, result) for cam in cams]
    vg = engine.vehicle_geom(project, variant)
    v = variant or project.variant
    lids = [(sn, project.sensor_model_for(sn)) for sn in v.sensors if sn.kind == 'lidar' and sn.visible]
    lcubes = [engine.cube(sn.pos().astype(float), vg) if vg is not None else None for sn, _ in lids]
    s2 = float(stride) ** 2
    shp = (nz, ny, nx)
    cam_rays = np.zeros(shp, np.int16)
    feats = np.zeros(shp, np.float32)
    best_f = np.zeros(shp, np.float32)
    lidar = np.zeros(shp, np.float32)
    sigma = np.full(shp, np.nan, np.float32)
    body = np.zeros(shp, bool)
    below = np.zeros(shp, bool)
    in_box = (X >= lo[0]) & (X <= hi[0]) & (Y >= lo[1]) & (Y <= hi[1])
    for k, z in enumerate(zs):
        P = np.stack([X, Y, np.full_like(X, z)], -1).reshape(-1, 3)
        below[k] = z < -1e-6
        body[k] = in_box & (z <= top)
        live = ~(below[k] | body[k]).ravel()
        if not live.any():
            continue
        Pl = P[live]
        vis_all, prad_all = [], []
        f_sum = np.zeros(len(Pl), np.float32)
        f_best = np.zeros(len(Pl), np.float32)
        nr = np.zeros(len(Pl), np.int16)
        for cam, intr, cube in zip(cams, intrs, cubes):
            vis, _, _, _ = eval_points(Pl, cam, intr, cube)
            f = np.zeros(len(Pl), np.float32)
            pr = np.zeros(len(Pl))
            if vis.any():
                f[vis] = (voxel_area_px(cam, intr, Pl[vis], a) / s2).astype(np.float32)
                _, _, th, _ = _project(cam, intr, Pl[vis])
                pr[vis] = px_per_rad(intr, th)
            vis_all.append(vis)
            prad_all.append(pr)
            f_sum += f
            f_best = np.maximum(f_best, f)
            nr += vis
        # the best pair's triangulation sigma
        sg = np.full(len(Pl), np.inf)
        for i in range(len(cams)):
            for j in range(i + 1, len(cams)):
                both = vis_all[i] & vis_all[j]
                if not both.any():
                    continue
                s_, _, _, _ = pair_sigma(cams[i].pos(), cams[j].pos(), Pl[both], prad_all[i][both], prad_all[j][both],
                                         sigma_px)
                sg[both] = np.minimum(sg[both], s_)
        lp = np.zeros(len(Pl))
        for (sn, mdl), lc in zip(lids, lcubes):
            lp += lidar_samples(sn, mdl, sn.pos().astype(float), Pl, a, lc)
        for arr, vals in ((cam_rays, nr), (feats, f_sum), (best_f, f_best), (lidar, lp),
                          (sigma, np.where(np.isfinite(sg), sg, np.nan))):
            flat = arr[k].reshape(-1)
            flat[live] = vals
    return VoxelDensity(float(x0), float(y0), z0, a, nx, ny, nz, int(stride), centre, cam_rays, feats, best_f, lidar,
                        sigma, body, below, [cam.tag for cam in cams], [sn.tag for sn, _ in lids],
                        (time.perf_counter() - t0) * 1000)


def export_occupancy(project, engine, out_dir, png=True, **kw) -> list[str]:
    """Write occupancy_density.npz (cam_rays, feats, best_feats, lidar_pts, sigma_m, body, below; (nz, ny, nx)),
    occupancy_density.json (the grid, conventions, summary) and, with png, one image per height band (the
    feature-starved and lidar-sparse share of each column, false colour). Returns the paths."""
    os.makedirs(out_dir, exist_ok=True)
    vd = voxel_density(project, engine, **kw)
    npz = os.path.join(out_dir, 'occupancy_density.npz')
    np.savez_compressed(npz, cam_rays=vd.cam_rays, feats=vd.feats, best_feats=vd.best_feats, lidar_pts=vd.lidar_pts,
                        sigma_m=vd.sigma_m, body=vd.body, below=vd.below)
    meta = {'schema': SCHEMA, 'project': project.name, 'variant': project.variant.name,
            'frame': 'ISO 8855 vehicle frame (x forward, y left, z up, metres), the rear-axle centre on the ground',
            'index': '[k, j, i] = voxel centred at (x0 + (i + 0.5) a, y0 + (j + 0.5) a, z0 + (k + 0.5) a)',
            'fields': {'cam_rays': 'cameras that see the voxel centre', 'feats': 'backbone feature cells covered, '
                       'summed over cameras (projected cube area / stride²)', 'best_feats': 'best single camera',
                       'lidar_pts': 'lidar samples through the voxel per sweep, summed over lidars',
                       'sigma_m': 'best camera pair 1σ triangulation depth uncertainty (m), NaN < 2 cameras',
                       'body': 'inside the vehicle box', 'below': 'centre under the ground'},
            'assumptions': ['free space: nothing but the vehicle stops a ray', 'flat ground at z = 0',
                            'a voxel counts for a camera when its centre is visible', f'sigma at {0.5} px'],
            'summary': vd.summary()}
    js = os.path.join(out_dir, 'occupancy_density.json')
    with open(js, 'w', encoding='utf-8') as fh:
        json.dump(meta, fh, indent=1)
    paths = [npz, js]
    if png:
        import cv2
        xs, ys, zs = vd.centres()
        for b in BANDS:
            sel = (zs >= b[0]) & (zs < b[1]) & (zs >= 0)
            if not sel.any():
                continue
            fr = vd.free[sel]
            n = np.maximum(fr.sum(axis=0), 1)
            starv = (vd.starved[sel].sum(axis=0) / n)
            sparse = (vd.sparse[sel].sum(axis=0) / n)
            img = np.zeros((vd.ny, vd.nx, 3), np.uint8)
            img[..., 2] = np.round(255 * starv)                  # red: feature-starved share of the column
            img[..., 1] = np.round(255 * (1 - sparse))           # green: lidar-covered share
            img[..., 0] = np.round(255 * np.isfinite(vd.sigma_m[sel]).mean(axis=0))   # blue: stereo share
            img = np.flipud(img)                                  # row 0 = max y
            img = cv2.resize(img, (vd.nx * 3, vd.ny * 3), interpolation=cv2.INTER_NEAREST)
            p = os.path.join(out_dir, f'occupancy_z{b[0]:+.0f}_{b[1]:+.0f}.png'.replace('+', 'p').replace('-', 'm'))
            cv2.imwrite(p, img)
            paths.append(p)
    return paths
