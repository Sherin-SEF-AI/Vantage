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


def warp(lut: StitchLUT, images: dict) -> tuple[dict, dict]:
    """Each camera's image resampled onto the ground grid: ({uid: float32 (ny, nx, 3)}, {uid: bool valid})."""
    import cv2
    out, valid = {}, {}
    for uid in lut.uids:
        img = images.get(uid)
        if img is None:
            continue
        if img.ndim == 2:
            img = np.repeat(img[..., None], 3, axis=2)
        h, w = img.shape[:2]
        sw, sh = lut.sensor[uid]
        ok = lut.map_x[uid] >= 0
        mx = np.where(ok, lut.map_x[uid] * (w / sw) - 0.5, -1).astype(np.float32)
        my = np.where(ok, lut.map_y[uid] * (h / sh) - 0.5, -1).astype(np.float32)
        out[uid] = cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0).astype(np.float32)
        valid[uid] = ok
    return out, valid


def overlap_means(lut: StitchLUT, warped: dict, valid: dict, step=2) -> list:
    """[(uid_a, uid_b, n, mean_a (3,), mean_b (3,))] over the ground both cameras see (well exposed pixels only)."""
    uids = [u for u in lut.uids if u in warped]
    sl = (slice(None, None, step), slice(None, None, step))
    free = ~lut.body[sl]
    out = []
    for i, a in enumerate(uids):
        for b in uids[i + 1:]:
            m = valid[a][sl] & valid[b][sl] & free
            if m.sum() < 50:
                continue
            A, B = warped[a][sl][m], warped[b][sl][m]
            la, lb = A.mean(1), B.mean(1)
            ok = (la > 8) & (la < 247) & (lb > 8) & (lb < 247)
            if ok.sum() < 50:
                continue
            out.append((a, b, int(ok.sum()), A[ok].mean(0), B[ok].mean(0)))
    return out


def estimate_gains(lut: StitchLUT, images: dict = None, warped=None, sigma_n=10.0, sigma_g=0.5) -> dict:
    """Per-camera, per-channel gains that make overlapping cameras agree (Brown & Lowe gain compensation):
    minimise sum N_ab ((g_a I_ab - g_b I_ba)^2 / sigma_n^2 + (1 - g)^2 / sigma_g^2). Returns {uid: (3,) gains}."""
    if warped is None:
        warped, valid = warp(lut, images)
    else:
        warped, valid = warped
    uids = [u for u in lut.uids if u in warped]
    idx = {u: i for i, u in enumerate(uids)}
    n = len(uids)
    pairs = overlap_means(lut, warped, valid)
    g = np.ones((n, 3))
    for ch in range(3):
        A = np.zeros((n, n))
        bvec = np.zeros(n)
        for a, b, N, ma, mb in pairs:
            i, j = idx[a], idx[b]
            Ia, Ib = ma[ch], mb[ch]
            A[i, i] += N * (Ia * Ia / sigma_n ** 2 + 1 / sigma_g ** 2)
            A[j, j] += N * (Ib * Ib / sigma_n ** 2 + 1 / sigma_g ** 2)
            A[i, j] -= N * Ia * Ib / sigma_n ** 2
            A[j, i] -= N * Ia * Ib / sigma_n ** 2
            bvec[i] += N / sigma_g ** 2
            bvec[j] += N / sigma_g ** 2
        for i in range(n):
            if A[i, i] == 0:
                A[i, i], bvec[i] = 1.0, 1.0
        g[:, ch] = np.linalg.solve(A, bvec)
    g = np.clip(g, 0.4, 2.5)
    return {u: g[idx[u]].astype(np.float32) for u in uids}


def overlap_mismatch(lut: StitchLUT, warped, gains=None) -> float:
    """Mean absolute grey-level difference between overlapping cameras (after gains when given)."""
    w, valid = warped
    tot, num = 0.0, 0
    for a, b, N, ma, mb in overlap_means(lut, w, valid):
        ga = gains[a] if gains else 1.0
        gb = gains[b] if gains else 1.0
        tot += N * float(np.mean(np.abs(ma * ga - mb * gb)))
        num += N
    return tot / num if num else 0.0


def _fill(img, ok, levels):
    """Pull-push fill of the pixels a camera does not see, so its Laplacian pyramid has no edge there."""
    import cv2
    w = ok.astype(np.float32)
    ims, ws = [img * w[..., None]], [w]
    for _ in range(levels):
        ims.append(cv2.pyrDown(ims[-1]))
        ws.append(cv2.pyrDown(ws[-1]))
    out = ims[-1] / np.maximum(ws[-1], 1e-6)[..., None]
    for l in range(levels - 1, -1, -1):
        up = cv2.pyrUp(out, dstsize=(ims[l].shape[1], ims[l].shape[0]))
        own = ims[l] / np.maximum(ws[l], 1e-6)[..., None]
        out = np.where((ws[l] > 1e-3)[..., None], own, up)
    return out


def _multiband(lut: StitchLUT, warped: dict, valid: dict, bands: int) -> np.ndarray:
    """Laplacian-pyramid blend: fine detail switches sharply at the seam (no double edges), brightness blends over
    2^bands pixels (no visible step)."""
    import cv2
    uids = [u for u in lut.uids if u in warped]
    stack = np.stack([lut.weight[u] for u in uids])
    own = np.argmax(stack, 0)
    has = stack.max(0) > 0
    acc = wacc = None
    for k, u in enumerate(uids):
        m = ((own == k) & has).astype(np.float32)
        if not m.any():
            continue
        G = [_fill(warped[u], valid[u], bands)]
        M = [m]
        for _ in range(bands):
            G.append(cv2.pyrDown(G[-1]))
            M.append(cv2.pyrDown(M[-1]))
        L = [G[l] - cv2.pyrUp(G[l + 1], dstsize=(G[l].shape[1], G[l].shape[0])) for l in range(bands)] + [G[-1]]
        if acc is None:
            acc = [np.zeros_like(x) for x in L]
            wacc = [np.zeros(x.shape[:2], np.float32) for x in L]
        for l in range(bands + 1):
            acc[l] += L[l] * M[l][..., None]
            wacc[l] += M[l]
    if acc is None:
        return np.zeros((lut.ny, lut.nx, 3), np.float32)
    acc = [a / np.maximum(w, 1e-6)[..., None] for a, w in zip(acc, wacc)]
    out = acc[-1]
    for l in range(bands - 1, -1, -1):
        out = cv2.pyrUp(out, dstsize=(acc[l].shape[1], acc[l].shape[0])) + acc[l]
    return out


def stitch(lut: StitchLUT, images: dict, body_rgb=(38, 38, 36), empty_rgb=(20, 20, 20), gains=None, blend='linear',
           bands=5) -> np.ndarray:
    """images: uid -> uint8 image (grey or RGB, any resolution with the sensor's aspect). Returns RGB uint8.
    gains: {uid: scalar or (3,)} exposure gains, 'auto' to estimate them from the overlaps, None for none.
    blend: 'linear' (the remap tables' weights) or 'multiband'."""
    warped, valid = warp(lut, images)
    if isinstance(gains, str):
        gains = estimate_gains(lut, warped=(warped, valid))
    if gains:
        warped = {u: w * np.asarray(gains.get(u, 1.0), np.float32) for u, w in warped.items()}
    if blend == 'multiband' and warped:
        out = _multiband(lut, warped, valid, bands)
    else:
        out = np.zeros((lut.ny, lut.nx, 3), np.float32)
        for uid, smp in warped.items():
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
        cams.append({'tag': tag, 'tag_safe': safe, 'map': f'{safe}_map.bin', 'weight': f'{safe}_weight.bin',
                     'sensor_width': lut.sensor[uid][0], 'sensor_height': lut.sensor[uid][1],
                     'pose': {'x': round(x, 4), 'y': round(yy, 4), 'z': round(z, 4), 'yaw': round(y, 3), 'pitch': round(pi, 3),
                              'roll': round(r, 3)}})
    files.append(('body_mask.bin', lut.body.astype(np.uint8).tobytes()))
    txt = ['# vantage.bev_lut/1 manifest for the reference stitcher (stitcher/bev_stitch.cpp); see bev_lut.json',
           f'width {lut.nx}', f'height {lut.ny}', f'resolution {lut.res:.6g}',
           f'extent {lut.x0:.6g} {lut.y0:.6g} {lut.x0 + lut.nx * lut.res:.6g} {lut.y0 + lut.ny * lut.res:.6g}']
    txt += [f'camera {c["tag_safe"]} {c["map"]} {c["weight"]} {c["sensor_width"]} {c["sensor_height"]}' for c in cams]
    txt.append('body body_mask.bin')
    for c in cams:
        c.pop('tag_safe')
    files.append(('bev_lut.txt', ('\n'.join(txt) + '\n').encode('utf-8')))
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


RUNTIME_FILES = ('bev_stitch.cpp', 'CMakeLists.txt', 'bev_stitch.py', 'README.md')


def runtime_files() -> list[tuple[str, bytes]]:
    """The reference stitcher sources ([(name, bytes)]), to sit in a stitcher/ folder next to the remap tables."""
    here = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'runtime')
    out = []
    for n in RUNTIME_FILES:
        with open(os.path.join(here, n), 'rb') as fh:
            out.append((n, fh.read()))
    return out


def export_lut(lut: StitchLUT, out_dir: str, project: Project | None = None, variant=None) -> list[str]:
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for name, data in lut_files(lut, project, variant):
        p = os.path.join(out_dir, name)
        with open(p, 'wb') as fh:
            fh.write(data)
        paths.append(p)
    return paths
