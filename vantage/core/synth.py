"""Synthetic camera captures: what a mounted camera would record of the floor, the calibration boards, the body and
the rig's own sensor and camera housings.

Used to test the installation check and the stitching pipeline end to end, and as a stand-in when no real
captures are loaded. Pure numpy; the lens model, occlusion (body, rig and housings: the engine's cube maps) and
target layout are the same as the analysis. Housings are drawn as dark plastic and metal (core.housings.GREY / RGB)
lit by the occlusion module's light, so they read as the rig's hardware rather than the body.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .housings import GREY as HOUSING_GREY, RGB as HOUSING_RGB, tone as housing_tone
from .lens import resolve, Intrinsics
from .mathutil import rot_zyx, BODY_TO_OPTICAL
from .project import Project, Camera, Target

BOARD_MARGIN = 0.05        # white border around a printed board, metres
NEAR_CLIP = 0.05           # the lens protrudes from its module: body surface closer than this is not seen (housings
                           # of other sensors and cameras are, however close)

_rays = {}


def pixel_rays(intr: Intrinsics, w: int, h: int):
    """Optical-frame unit rays and theta for every pixel of a w x h image of this sensor."""
    key = (intr.width, intr.height, intr.pixel_mm, intr.f_mm, intr.projection, tuple(intr.kb), intr.cx, intr.cy,
           intr.aspect, w, h)
    if key in _rays:
        return _rays[key]
    us = (np.arange(w) + 0.5) / w * intr.width
    vs = (np.arange(h) + 0.5) / h * intr.height
    U, V = np.meshgrid(us, vs)
    d, th = intr.unproject(U.ravel(), V.ravel())
    if len(_rays) > 8:
        _rays.clear()
    _rays[key] = (d, th)
    return d, th


def _checker(gx, gy, targets, fw):
    """Board intensity (0..1) at ground points; NaN where no board. fw = pixel footprint in metres (antialiasing)."""
    out = np.full(gx.shape, np.nan)
    for t in targets:
        c, s = math.cos(math.radians(t.yaw)), math.sin(math.radians(t.yaw))
        lx = (gx - t.x) * c + (gy - t.y) * s
        ly = -(gx - t.x) * s + (gy - t.y) * c
        w, h = t.size_m
        sq = t.square_mm / 1000
        inside = (np.abs(lx) <= w / 2 + BOARD_MARGIN) & (np.abs(ly) <= h / 2 + BOARD_MARGIN)
        if not inside.any():
            continue
        u = (lx[inside] + w / 2) / sq
        v = (ly[inside] + h / 2) / sq
        f = np.maximum(fw[inside] / sq, 1e-4)
        du = np.abs(u - np.round(u))
        dv = np.abs(v - np.round(v))
        sign = np.where((np.floor(u) + np.floor(v)) % 2 == 0, -1.0, 1.0)     # dark square at the (0, 0) corner
        edge = np.clip(np.minimum(du, dv) / f, 0, 1)
        val = 0.5 + 0.5 * sign * edge
        in_board = (u >= 0) & (u <= t.cols) & (v >= 0) & (v <= t.rows)
        # soften the outer board edge too
        bu = np.minimum(u, t.cols - u)
        bv = np.minimum(v, t.rows - v)
        border = np.clip(np.minimum(bu, bv) / f + 0.5, 0, 1)
        val = np.where(in_board, val * border + (1 - border), 1.0)
        cur = out[inside]
        out[inside] = np.where(np.isnan(cur), val, np.minimum(cur, val))
    return out


def render_capture(project: Project, engine, cam: Camera, w=None, h=None, pose=None, targets=None, seed=0,
                   noise=2.0, rgb=False, exposure=1.0, vignette=0.0, world=None):
    """Grey (or RGB) image a camera records. pose = (R_body_to_world, position) of the *installed* camera
    (defaults to the planned pose), so a capture can include a deliberate mounting error. exposure scales the
    brightness (auto-exposure differs between cameras); vignette darkens towards the image circle's edge by that
    fraction. world: a World (e.g. parking_world()) renders an RGB parking scene instead of the bare calibration
    floor."""
    if world is not None:
        return render_world(project, engine, cam, world, w, h, pose, seed, noise, exposure, vignette)
    intr = resolve(project.module_for(cam), project.lens_for(cam))
    w = w or intr.width
    h = h or intr.height
    R = rot_zyx(cam.yaw, cam.pitch, cam.roll) if pose is None else np.asarray(pose[0], float)
    pos = cam.pos() if pose is None else np.asarray(pose[1], float)
    W2O = BODY_TO_OPTICAL @ R.T
    d_opt, th = pixel_rays(intr, w, h)
    dw = d_opt @ W2O
    valid = th <= intr.max_theta
    dz = dw[:, 2]
    with np.errstate(divide='ignore', invalid='ignore'):
        tg = np.where(dz < -1e-6, -pos[2] / dz, np.inf)
    ground = valid & np.isfinite(tg) & (tg < 60)
    body_d = np.full(len(dw), np.inf)
    shade = np.zeros(len(dw), np.uint8)
    hid = np.full(len(dw), -1, np.int16)
    cube = None
    if engine.vehicle_geom(project) is not None:
        cube = engine.cube(pos)
        body_d, shade, hid = cube.lookup_ids(dw)
    near = np.isfinite(body_d) & (body_d < np.where(ground, tg, np.inf))
    house = valid & near & (hid >= 0)
    body = valid & near & (hid < 0) & (body_d > NEAR_CLIP)
    gv = ground & ~body & ~house
    img = np.full(len(dw), 40.0)                         # outside the image circle
    img[valid] = 150.0                                   # sky / walls
    tgf = np.where(gv, tg, 0.0)
    gx = pos[0] + dw[:, 0] * tgf
    gy = pos[1] + dw[:, 1] * tgf
    # floor: mid grey with a faint large-scale pattern so stitching seams are visible
    floor = 118 + 10 * np.sin(gx * 2.1) * np.sin(gy * 1.7)
    img[gv] = floor[gv]
    # pixel footprint on the ground (metres): range x angular pixel size, stretched by incidence
    inc = np.clip(np.abs(dz), 0.08, 1.0)
    pix_ang = intr.pixel_mm / np.maximum(1e-6, intr.f_mm) * (intr.width / w)
    fw = np.where(gv, tg * pix_ang / inc, 1.0)
    tl = project.targets if targets is None else targets
    if tl:
        b = _checker(gx[gv], gy[gv], tl, fw[gv])
        sel = ~np.isnan(b)
        idx = np.nonzero(gv)[0][sel]
        img[idx] = 25 + 210 * b[sel]
    img[body] = 55 + 0.45 * shade[body]
    if house.any():
        grey = np.array([HOUSING_GREY.get(h.what, 0.15) for h in cube.housings])
        img[house] = 255.0 * grey[hid[house]] * housing_tone(shade[house])
    if exposure != 1.0 or vignette:
        img = img * exposure * (1 - vignette * np.clip(th / intr.max_theta, 0, 1) ** 2)
    if noise:
        rng = np.random.default_rng(seed)
        img = img + rng.normal(0, noise, img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8).reshape(h, w)
    if rgb:
        return np.repeat(img[..., None], 3, axis=2)
    return img


# ---------------------------------------------------------------- a parking scene
@dataclass
class World:
    """A synthetic surrounding: a parking row with bay markings, parked cars (boxes), pillars and bollards
    (vertical cylinders) and a garage wall all round. Positions are world metres; colours 0..1 RGB."""
    centre: tuple = (1.3, 0.0)
    markings: list = field(default_factory=list)    # [(x0, y0, x1, y1, width, dash)] dash 0 = solid, else period m
    boxes: list = field(default_factory=list)       # [(cx, cy, yaw_deg, length, width, height, rgb)]
    cylinders: list = field(default_factory=list)   # [(x, y, radius, height, rgb, banded)]
    wall_r: float = 11.0
    wall_h: float = 3.2
    boards: bool = False                            # also draw the project's calibration boards
    sun: tuple = (0.32, 0.42, 0.85)


def parking_world(project: Project, boards=False) -> World:
    """The ego car in a bay of a parking row, a driving aisle in front, parked cars either side and across the aisle,
    a concrete pillar, a bollard near the rear corner, the project's obstacles, and a garage wall."""
    meas = project.vehicle.measured() if project.vehicle.mesh is not None else {}
    cx = (meas.get('front_x', 3.2) + meas.get('rear_x', -0.8)) / 2
    length = meas.get('Length', 4.0)
    bay_w, bay_l = 2.7, max(5.2, length + 1.0)
    rear, front = cx - bay_l / 2, cx + bay_l / 2
    aisle = 6.0
    lw = 0.12
    marks = []
    for k in range(-3, 5):
        y = (k - 0.5) * bay_w
        marks.append((rear, y, front, y, lw, 0.0))                                   # bay sides, this row
        marks.append((front + aisle, y, front + aisle + bay_l, y, lw, 0.0))          # across the aisle
    marks.append((rear, -3.5 * bay_w, rear, 4.5 * bay_w, lw, 0.0))                  # end line behind the row
    marks.append((front + aisle, -3.5 * bay_w, front + aisle, 4.5 * bay_w, lw, 0.0))
    marks.append((front + aisle / 2, -3.5 * bay_w, front + aisle / 2, 4.5 * bay_w, 0.10, 1.2))  # aisle centre, dashed
    boxes = [(cx - 0.1, bay_w, 0.0, 4.4, 1.82, 1.48, (0.16, 0.24, 0.42)),
             (cx + 0.2, -2 * bay_w, 0.0, 4.6, 1.86, 1.52, (0.58, 0.12, 0.10)),
             (front + aisle + bay_l / 2, 0.4, 0.0, 4.5, 1.84, 1.50, (0.80, 0.80, 0.78)),
             (front + aisle + bay_l / 2 + 0.2, -bay_w - 0.2, 3.0, 4.3, 1.80, 1.62, (0.20, 0.20, 0.21))]
    cyl = [(front + 0.9, -1.5 * bay_w, 0.32, 3.2, (0.70, 0.69, 0.66), True),          # pillar with hazard band
           (rear - 0.55, 0.95, 0.07, 0.9, (0.95, 0.74, 0.10), False)]                 # bollard by the rear corner
    for ob in project.obstacles:
        cyl.append((ob.x, ob.y, max(0.02, ob.radius), max(0.05, ob.height), (0.92, 0.45, 0.12), False))
    return World(centre=(cx, 0.0), markings=marks, boxes=boxes, cylinders=cyl, boards=boards)


def _seg_cover(px, py, segs, fw):
    """Antialiased coverage 0..1 of floor points by painted line segments."""
    cov = np.zeros(len(px))
    for x0, y0, x1, y1, wdt, dash in segs:
        dx, dy = x1 - x0, y1 - y0
        L2 = dx * dx + dy * dy
        t = np.clip(((px - x0) * dx + (py - y0) * dy) / max(L2, 1e-12), 0, 1)
        d = np.hypot(px - (x0 + t * dx), py - (y0 + t * dy))
        c = np.clip((wdt / 2 - d) / np.maximum(fw, 1e-4) + 0.5, 0, 1)
        if dash:
            s = t * math.sqrt(L2)
            ph = np.abs((s % dash) - dash * 0.25) - dash * 0.25        # <0 on the painted half
            c *= np.clip(-ph / np.maximum(fw, 1e-4) + 0.5, 0, 1)
        cov = np.maximum(cov, c)
    return cov


def _ray_box(o, d, box):
    """Entry distance and outward normal (world) of rays into a yawed box standing on the floor."""
    cx, cy, yaw, L, W, H = box[:6]
    c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    ox, oy = (o[0] - cx) * c + (o[1] - cy) * s, -(o[0] - cx) * s + (o[1] - cy) * c
    dx, dy = d[:, 0] * c + d[:, 1] * s, -d[:, 0] * s + d[:, 1] * c
    lo = np.array([-L / 2, -W / 2, 0.0])
    hi = np.array([L / 2, W / 2, H])
    oo = (ox, oy, o[2])
    dd = (dx, dy, d[:, 2])
    tmin = np.full(len(d), -np.inf)
    tmax = np.full(len(d), np.inf)
    axis = np.zeros(len(d), int)
    sign = np.zeros(len(d))
    with np.errstate(divide='ignore', invalid='ignore'):
        for k in range(3):
            inv = 1.0 / dd[k]
            t1, t2 = (lo[k] - oo[k]) * inv, (hi[k] - oo[k]) * inv
            near = np.minimum(t1, t2)
            far = np.maximum(t1, t2)
            par = np.abs(dd[k]) < 1e-12
            inside = (oo[k] >= lo[k]) & (oo[k] <= hi[k])
            near = np.where(par, np.where(inside, -np.inf, np.inf), near)
            far = np.where(par, np.where(inside, np.inf, -np.inf), far)
            better = near > tmin
            axis = np.where(better, k, axis)
            sign = np.where(better, np.where(t1 < t2, -1.0, 1.0), sign)
            tmin = np.maximum(tmin, near)
            tmax = np.minimum(tmax, far)
    hit = (tmax >= tmin) & (tmin > 1e-4)
    nl = np.zeros((len(d), 3))
    nl[np.arange(len(d)), axis] = sign
    n = np.stack([nl[:, 0] * c - nl[:, 1] * s, nl[:, 0] * s + nl[:, 1] * c, nl[:, 2]], 1)
    return np.where(hit, tmin, np.inf), n


def _ray_cyl(o, d, x, y, r, h):
    """Entry distance and normal of rays into a vertical cylinder (side or top cap) on the floor."""
    ox, oy = o[0] - x, o[1] - y
    a = d[:, 0] ** 2 + d[:, 1] ** 2
    b = 2 * (ox * d[:, 0] + oy * d[:, 1])
    c = ox * ox + oy * oy - r * r
    disc = b * b - 4 * a * c
    with np.errstate(divide='ignore', invalid='ignore'):
        t = (-b - np.sqrt(np.maximum(disc, 0))) / (2 * np.maximum(a, 1e-12))
        z = o[2] + t * d[:, 2]
        side = (disc > 0) & (t > 1e-4) & (z >= 0) & (z <= h)
        tc = (h - o[2]) / d[:, 2]
        cx_, cy_ = ox + tc * d[:, 0], oy + tc * d[:, 1]
        cap = (o[2] > h) & (d[:, 2] < 0) & (cx_ * cx_ + cy_ * cy_ <= r * r) & (tc > 1e-4)
    ts = np.where(side, t, np.inf)
    tcap = np.where(cap, tc, np.inf)
    use_cap = tcap < ts
    tt = np.where(side, t, 0.0)
    n = np.stack([ox + tt * d[:, 0], oy + tt * d[:, 1], np.zeros(len(d))], 1)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
    n[use_cap] = (0, 0, 1)
    return np.minimum(ts, tcap), n, z


def render_world(project: Project, engine, cam: Camera, world: World, w=None, h=None, pose=None, seed=0, noise=2.0,
                 exposure=1.0, vignette=0.0) -> np.ndarray:
    """RGB uint8 image of the parking scene through this camera (same lens model, occlusion and pose handling as
    render_capture)."""
    intr = resolve(project.module_for(cam), project.lens_for(cam))
    w = w or intr.width
    h = h or intr.height
    R = rot_zyx(cam.yaw, cam.pitch, cam.roll) if pose is None else np.asarray(pose[0], float)
    pos = cam.pos() if pose is None else np.asarray(pose[1], float)
    d_opt, th = pixel_rays(intr, w, h)
    img = np.zeros((len(th), 3))
    img[:] = 0.16                                                   # outside the image circle
    idx = np.nonzero(th <= intr.max_theta)[0]
    d = (d_opt[idx] @ (BODY_TO_OPTICAL @ R.T))
    n = len(d)
    sun = np.asarray(world.sun, float)
    sun = sun / np.linalg.norm(sun)
    best = np.full(n, np.inf)
    mat = np.zeros(n, int)          # 0 sky, 1 floor, 2 box, 3 cylinder, 4 wall, 5 body, 6 a sensor or camera housing
    nrm = np.zeros((n, 3))
    col = np.zeros((n, 3))
    # floor
    with np.errstate(divide='ignore', invalid='ignore'):
        tg = np.where(d[:, 2] < -1e-6, -pos[2] / d[:, 2], np.inf)
    sel = tg < best
    best[sel], mat[sel] = tg[sel], 1
    # wall around the scene (seen from inside)
    ox, oy = pos[0] - world.centre[0], pos[1] - world.centre[1]
    a = d[:, 0] ** 2 + d[:, 1] ** 2
    b = 2 * (ox * d[:, 0] + oy * d[:, 1])
    c = ox * ox + oy * oy - world.wall_r ** 2
    with np.errstate(divide='ignore', invalid='ignore'):
        tw = (-b + np.sqrt(np.maximum(b * b - 4 * a * c, 0))) / (2 * np.maximum(a, 1e-12))
    zw = pos[2] + tw * d[:, 2]
    tw = np.where((tw > 0) & (zw >= 0) & (zw <= world.wall_h) & (a > 1e-9), tw, np.inf)
    sel = tw < best
    best[sel], mat[sel] = tw[sel], 4
    wall_n = -np.stack([ox + tw * d[:, 0], oy + tw * d[:, 1], np.zeros(n)], 1) / world.wall_r
    nrm[sel] = wall_n[sel]
    for k, bx in enumerate(world.boxes):
        t, nb = _ray_box(pos, d, bx)
        sel = t < best
        best[sel], mat[sel] = t[sel], 2
        nrm[sel] = nb[sel]
        zb = pos[2] + t * d[:, 2]
        base = np.asarray(bx[6], float)
        top = nb[:, 2] > 0.5
        glass = (zb > 0.6 * bx[5]) & ~top
        tyre = zb < 0.32
        cc = np.where(glass[:, None], [0.10, 0.12, 0.14], np.where(tyre[:, None], [0.07, 0.07, 0.07], base))
        cc = np.where(top[:, None], base * 1.1, cc)
        col[sel] = cc[sel]
    for x, y, r, hh, rgb, banded in world.cylinders:
        t, nc, zc = _ray_cyl(pos, d, x, y, r, hh)
        sel = t < best
        best[sel], mat[sel] = t[sel], 3
        nrm[sel] = nc[sel]
        cc = np.tile(np.asarray(rgb, float), (n, 1))
        if banded:
            band = (zc > 0.25) & (zc < 1.05)
            stripe = ((zc * 6 + np.arctan2(nc[:, 1], nc[:, 0]) * 1.5) % 2.0) < 1.0
            cc = np.where((band & stripe)[:, None], [0.95, 0.76, 0.08], np.where(band[:, None], [0.08, 0.08, 0.08], cc))
        col[sel] = cc[sel]
    # the ego body, its rig and the housings (cube depth map, as the analysis sees it)
    if engine.vehicle_geom(project) is not None:
        cube = engine.cube(pos)
        bd, shade, hid = cube.lookup_ids(d)
        sel = np.isfinite(bd) & (bd < best) & ((bd > NEAR_CLIP) | (hid >= 0))
        body = sel & (hid < 0)
        best[body], mat[body] = bd[body], 5
        col[body] = (0.18 + 0.72 * shade[body, None] / 255.0) * np.array([0.86, 0.87, 0.88])
        house = sel & (hid >= 0)
        if house.any():
            rgb = np.array([HOUSING_RGB.get(h.what, (0.14, 0.14, 0.15)) for h in cube.housings])
            best[house], mat[house] = bd[house], 6
            col[house] = rgb[hid[house]] * housing_tone(shade[house])[:, None]
    P = pos + d * np.where(np.isfinite(best), best, 0)[:, None]
    # shading
    lam = np.clip(np.sum(nrm * sun, axis=1), 0, 1)
    for m in (2, 3):
        s = mat == m
        col[s] *= (0.50 + 0.55 * lam[s])[:, None]
    s = mat == 4
    if s.any():
        az = np.arctan2(P[s, 1] - world.centre[1], P[s, 0] - world.centre[0]) * world.wall_r
        z = P[s, 2]
        conc = 0.62 + 0.04 * np.sin(az * 3.1) * np.sin(z * 7.3)
        joint = (np.abs(((az + 1.25) % 2.5) - 1.25) < 0.025) | (np.abs(z - 2.4) < 0.02)
        stripe = (z > 0.85) & (z < 1.15)
        c = np.stack([conc, conc * 0.98, conc * 0.94], 1)
        c[joint] *= 0.7
        c[stripe] = [0.86, 0.66, 0.10]
        col[s] = c * (0.80 + 0.25 * lam[s])[:, None]
    s = mat == 1
    if s.any():
        gx, gy = P[s, 0], P[s, 1]
        inc = np.clip(np.abs(d[s, 2]), 0.06, 1.0)
        pix = intr.pixel_mm / intr.f_mm * (intr.width / w)
        fw = best[s] * pix / inc
        # asphalt: a faint large-scale mottle, plus fine grain that fades out where a pixel covers several grains
        grain = np.clip(1 - fw * 12, 0, 1)
        asphalt = (0.34 + 0.012 * np.sin(gx * 2.3 + gy * 1.9) + 0.008 * np.sin(gx * 0.7 - gy * 1.1)
                   + 0.018 * grain * np.sin(gx * 41.0) * np.sin(gy * 37.0))
        cov = _seg_cover(gx, gy, world.markings, fw)
        c = np.stack([asphalt, asphalt, asphalt * 1.03], 1) * (1 - cov[:, None]) + cov[:, None] * np.array([0.90, 0.90, 0.86])
        if world.boards and project.targets:
            bv = _checker(gx, gy, project.targets, fw)
            ok = ~np.isnan(bv)
            c[ok] = (0.10 + 0.82 * bv[ok])[:, None]
        # shadows of the boxes and cylinders on the floor (rays towards the sun)
        lit = np.ones(len(gx))
        Pf = np.stack([gx, gy, np.full(len(gx), 1e-3)], 1)
        sd = np.tile(sun, (len(gx), 1))
        for bx in world.boxes:
            t, _ = _ray_box_many(Pf, sd, bx)
            lit = np.where(np.isfinite(t), np.minimum(lit, 0.55), lit)
        for x, y, r, hh, _, _ in world.cylinders:
            t = _ray_cyl_many(Pf, sd, x, y, r, hh)
            lit = np.where(np.isfinite(t), np.minimum(lit, 0.6), lit)
        col[s] = c * lit[:, None]
    s = mat == 0
    if s.any():
        el = np.clip(d[s, 2], 0, 1)
        col[s] = np.array([0.80, 0.84, 0.88]) * (1 - el[:, None]) + np.array([0.55, 0.66, 0.80]) * el[:, None]
    out = img
    out[idx] = col
    out = out * 255.0
    if exposure != 1.0 or vignette:
        out = out * exposure * (1 - vignette * np.clip(th / intr.max_theta, 0, 1) ** 2)[:, None]
    if noise:
        out = out + np.random.default_rng(seed).normal(0, noise, out.shape)
    return np.clip(out, 0, 255).astype(np.uint8).reshape(h, w, 3)


def _ray_box_many(P, d, box):
    """_ray_box for rays with individual origins (P (K,3))."""
    cx, cy, yaw, L, W, H = box[:6]
    c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    ox, oy = (P[:, 0] - cx) * c + (P[:, 1] - cy) * s, -(P[:, 0] - cx) * s + (P[:, 1] - cy) * c
    dx, dy = d[:, 0] * c + d[:, 1] * s, -d[:, 0] * s + d[:, 1] * c
    lo, hi = (-L / 2, -W / 2, 0.0), (L / 2, W / 2, H)
    tmin = np.zeros(len(P))
    tmax = np.full(len(P), np.inf)
    with np.errstate(divide='ignore', invalid='ignore'):
        for oo, dd, a, b in ((ox, dx, lo[0], hi[0]), (oy, dy, lo[1], hi[1]), (P[:, 2], d[:, 2], lo[2], hi[2])):
            t1, t2 = (a - oo) / dd, (b - oo) / dd
            par = np.abs(dd) < 1e-12
            inside = (oo >= a) & (oo <= b)
            near = np.where(par, np.where(inside, -np.inf, np.inf), np.minimum(t1, t2))
            far = np.where(par, np.where(inside, np.inf, -np.inf), np.maximum(t1, t2))
            tmin = np.maximum(tmin, near)
            tmax = np.minimum(tmax, far)
    hit = tmax >= tmin
    return np.where(hit, tmin, np.inf), None


def _ray_cyl_many(P, d, x, y, r, h):
    ox, oy = P[:, 0] - x, P[:, 1] - y
    a = d[:, 0] ** 2 + d[:, 1] ** 2
    b = 2 * (ox * d[:, 0] + oy * d[:, 1])
    c = ox * ox + oy * oy - r * r
    disc = b * b - 4 * a * c
    with np.errstate(divide='ignore', invalid='ignore'):
        t = (-b - np.sqrt(np.maximum(disc, 0))) / (2 * np.maximum(a, 1e-12))
        z = P[:, 2] + t * d[:, 2]
    return np.where((disc > 0) & (t > 1e-4) & (z >= 0) & (z <= h), t, np.inf)
