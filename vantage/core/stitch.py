"""Bird's-eye stitching: per-camera remap tables + blend weights, and the stitched image itself.

The tables are what a surround-view pipeline on the target (e.g. cv2.remap / NPP remap on the Jetson)
needs: for every output pixel on the ground, where to sample each camera and with what weight.
Ownership follows the analysis seams; a linear blend of `blend_m` is applied across each seam, and
pixels the owner cannot see fall back to the camera with the best pixel density.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field

import numpy as np

from .lens import resolve
from .mathutil import rot_zyx, BODY_TO_OPTICAL
from .project import Project, Camera


@dataclass
class StitchLUT:
    x0: float
    y0: float
    res: float
    nx: int
    ny: int
    uids: list
    tags: list
    sensor: dict                       # uid -> (w, h)
    map_x: dict                        # uid -> (ny, nx) float32, sensor pixel coords, -1 = no sample
    map_y: dict
    weight: dict                       # uid -> (ny, nx) float32, sums to 1 where covered
    body: np.ndarray                   # (ny, nx) bool, under the vehicle
    covered: np.ndarray                # (ny, nx) bool
    poses: dict = field(default_factory=dict)   # uid -> (yaw, pitch, roll, x, y, z) used

    def world_xy(self):
        xs = self.x0 + (np.arange(self.nx) + 0.5) * self.res
        ys = self.y0 + (self.ny - np.arange(self.ny) - 0.5) * self.res     # row 0 = max y (image up = +y)
        return np.meshgrid(xs, ys)


def build_lut(project: Project, engine, result, res=0.02, poses=None, extent=None, blend_m=None) -> StitchLUT:
    """poses: optional uid -> (R_body_to_world, pos) (e.g. measured by the installation check)."""
    variant = project.variant_by_uid(result.variant_uid) or project.variant
    cams = [c for c in variant.cameras if c.visible and result.cam_index(c.uid) >= 0]
    g = result.grid
    if extent is None:
        extent = (g.x0, g.y0, g.x0 + g.nx * g.cell, g.y0 + g.ny * g.cell)
    x0, y0, x1, y1 = extent
    nx = int(round((x1 - x0) / res))
    ny = int(round((y1 - y0) / res))
    lut = StitchLUT(x0, y0, res, nx, ny, [c.uid for c in cams], [c.tag for c in cams], {}, {}, {}, {},
                    np.zeros((ny, nx), bool), np.zeros((ny, nx), bool))
    X, Y = lut.world_xy()
    P = np.stack([X.ravel(), Y.ravel(), np.zeros(X.size)], 1)
    # under-body mask from the analysis grid (nearest cell)
    gi = np.clip(((X - g.x0) / g.cell).astype(int), 0, g.nx - 1)
    gj = np.clip(((Y - g.y0) / g.cell).astype(int), 0, g.ny - 1)
    lut.body = g.body[gj, gi]
    vis, dens = {}, {}
    has_body = engine.vehicle_geom(project) is not None
    for c in cams:
        intr = resolve(project.module_for(c), project.lens_for(c))
        if poses and c.uid in poses:
            R, pos = poses[c.uid]
            R, pos = np.asarray(R, float), np.asarray(pos, float)
        else:
            R, pos = rot_zyx(c.yaw, c.pitch, c.roll), c.pos()
        W2O = BODY_TO_OPTICAL @ R.T
        d = P - pos
        dist = np.linalg.norm(d, axis=1)
        u, v, th, ok = intr.project_dirs(d @ W2O.T)
        if has_body:
            depth, _ = engine.cube(pos).lookup(d)
            ok &= depth > dist - (0.02 + 0.01 * dist)
        ok &= ~lut.body.ravel()
        lut.sensor[c.uid] = (intr.width, intr.height)
        lut.map_x[c.uid] = np.where(ok, u, -1).reshape(ny, nx).astype(np.float32)
        lut.map_y[c.uid] = np.where(ok, v, -1).reshape(ny, nx).astype(np.float32)
        vis[c.uid] = ok.reshape(ny, nx)
        dens[c.uid] = np.where(ok, intr.density_px_per_cm(th, dist), 0).reshape(ny, nx)
        from .mathutil import euler_zyx
        yaw, pitch, roll = euler_zyx(R)
        lut.poses[c.uid] = (yaw, pitch, roll, float(pos[0]), float(pos[1]), float(pos[2]))
    W = {c.uid: np.zeros((ny, nx), np.float32) for c in cams}
    cx, cy = result.centre
    az = np.arctan2(Y - cy, X - cx)
    seams = [s for s in result.seams if s.a in W and s.b in W]
    if seams:
        angs = [math.radians(s.angle) for s in seams]
        n = len(seams)
        for k in range(n):
            s0, s1 = angs[k], angs[(k + 1) % n]
            span = (s1 - s0) % (2 * math.pi) or 2 * math.pi
            own = ((az - s0) % (2 * math.pi)) < span
            W[seams[k].b][own] = 1.0
        half = (variant.blend_m if blend_m is None else blend_m) / 2
        if half > 0:
            for k, s in enumerate(seams):
                a = angs[k]
                Xc, Yc = X - cx, Y - cy
                along = Xc * math.cos(a) + Yc * math.sin(a)
                d = -Xc * math.sin(a) + Yc * math.cos(a)          # + = ccw side (camera b)
                zone = (along > 0) & (np.abs(d) < half)
                wb = np.clip(0.5 + d / (2 * half), 0, 1)
                W[s.a][zone] = (1 - wb[zone])
                W[s.b][zone] = wb[zone]
    elif cams:
        W[cams[0].uid][:] = 1.0
    for u in W:
        W[u] *= vis[u]
    tot = sum(W.values()) if W else np.zeros((ny, nx))
    # holes: owner cannot see -> best density camera that can
    hole = (tot < 1e-6) & ~lut.body
    if cams:
        D = np.stack([dens[c.uid] for c in cams])
        best = np.argmax(D, 0)
        anyv = D.max(0) > 0
        for i, c in enumerate(cams):
            m = hole & anyv & (best == i)
            W[c.uid][m] = 1.0
        tot = sum(W.values())
    with np.errstate(invalid='ignore', divide='ignore'):
        for u in W:
            W[u] = np.where(tot > 0, W[u] / np.where(tot > 0, tot, 1), 0).astype(np.float32)
    lut.weight = W
    lut.covered = tot > 0
    return lut


def stitch(lut: StitchLUT, images: dict, body_rgb=(38, 38, 36), empty_rgb=(20, 20, 20)) -> np.ndarray:
    """images: uid -> uint8 image (grey or RGB, any resolution with the sensor's aspect). Returns RGB uint8."""
    import cv2
    out = np.zeros((lut.ny, lut.nx, 3), np.float32)
    for uid in lut.uids:
        img = images.get(uid)
        if img is None:
            continue
        if img.ndim == 2:
            img = np.repeat(img[..., None], 3, axis=2)
        h, w = img.shape[:2]
        sw, sh = lut.sensor[uid]
        mx = np.where(lut.map_x[uid] >= 0, lut.map_x[uid] * (w / sw) - 0.5, -1).astype(np.float32)
        my = np.where(lut.map_y[uid] >= 0, lut.map_y[uid] * (h / sh) - 0.5, -1).astype(np.float32)
        smp = cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0).astype(np.float32)
        out += smp * lut.weight[uid][..., None]
    out[~lut.covered] = empty_rgb
    out[lut.body] = body_rgb
    return np.clip(out, 0, 255).astype(np.uint8)


def lut_files(lut: StitchLUT, project: Project | None = None, variant=None) -> list[tuple[str, bytes]]:
    """[(file name, bytes)]: bev_lut.json, bev_lut.npz, <tag>_map.bin (float32 x,y interleaved, row-major),
    <tag>_weight.bin (uint8 0..255) and body_mask.bin (uint8)."""
    import io
    files = []
    arrays = {'body_mask': lut.body.astype(np.uint8), 'covered': lut.covered.astype(np.uint8)}
    cams = []
    for uid, tag in zip(lut.uids, lut.tags):
        safe = ''.join(ch if ch.isalnum() or ch in '-_' else '_' for ch in tag) or 'cam'
        mp = np.stack([lut.map_x[uid], lut.map_y[uid]], -1).astype(np.float32)
        wt = np.round(lut.weight[uid] * 255).astype(np.uint8)
        arrays[f'{safe}_map'] = mp
        arrays[f'{safe}_weight'] = wt
        files.append((f'{safe}_map.bin', mp.tobytes()))
        files.append((f'{safe}_weight.bin', wt.tobytes()))
        y, pi, r, x, yy, z = lut.poses.get(uid, (0, 0, 0, 0, 0, 0))
        cams.append({'tag': tag, 'map': f'{safe}_map.bin', 'weight': f'{safe}_weight.bin',
                     'sensor_width': lut.sensor[uid][0], 'sensor_height': lut.sensor[uid][1],
                     'pose': {'x': round(x, 4), 'y': round(yy, 4), 'z': round(z, 4), 'yaw': round(y, 3), 'pitch': round(pi, 3),
                              'roll': round(r, 3)}})
    files.append(('body_mask.bin', lut.body.astype(np.uint8).tobytes()))
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    files.append(('bev_lut.npz', buf.getvalue()))
    meta = {
        'schema': 'vantage.bev_lut/1',
        'project': project.name if project else '',
        'variant': (variant or project.variant).name if project else '',
        'frame': 'ISO 8855, rear-axle origin on the ground; x forward, y left, z up',
        'width': lut.nx, 'height': lut.ny, 'resolution_m_per_px': lut.res,
        'extent_m': {'x0': lut.x0, 'y0': lut.y0, 'x1': lut.x0 + lut.nx * lut.res, 'y1': lut.y0 + lut.ny * lut.res},
        'image_axes': 'column 0 = x0 (rear), row 0 = y1 (left side at the top); pixel centres at +0.5',
        'map_format': 'float32, height x width x 2 (x, y) in SENSOR pixel coordinates (pixel centre = i + 0.5); -1 where the '
                      'camera does not contribute. Scale by (image_width / sensor_width) for other resolutions and subtract 0.5 '
                      'for cv2.remap.',
        'weight_format': 'uint8, height x width, 0..255; weights of all cameras sum to 255 where covered',
        'cameras': cams,
        'usage_python': 'out = sum(cv2.remap(img[c], mapx[c], mapy[c], cv2.INTER_LINEAR) * w[c] / 255)',
    }
    files.insert(0, ('bev_lut.json', json.dumps(meta, indent=1).encode('utf-8')))
    return files


def export_lut(lut: StitchLUT, out_dir: str, project: Project | None = None, variant=None) -> list[str]:
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for name, data in lut_files(lut, project, variant):
        p = os.path.join(out_dir, name)
        with open(p, 'wb') as fh:
            fh.write(data)
        paths.append(p)
    return paths
