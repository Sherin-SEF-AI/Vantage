"""Synthetic camera captures: what a mounted camera would record of the floor, the calibration boards and the body.

Used to test the installation check and the stitching pipeline end to end, and as a stand-in when no real
captures are loaded. Pure numpy; the lens model, occlusion and target layout are the same as the analysis.
"""
from __future__ import annotations

import math

import numpy as np

from .lens import resolve, Intrinsics
from .mathutil import rot_zyx, BODY_TO_OPTICAL
from .project import Project, Camera, Target

BOARD_MARGIN = 0.05        # white border around a printed board, metres
NEAR_CLIP = 0.05           # the lens protrudes from its housing: body surface closer than this is not seen

_rays = {}


def pixel_rays(intr: Intrinsics, w: int, h: int):
    """Optical-frame unit rays and theta for every pixel of a w x h image of this sensor."""
    key = (intr.width, intr.height, intr.f_mm, intr.projection, tuple(intr.kb), intr.cx, intr.cy, intr.aspect, w, h)
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
                   noise=2.0, rgb=False, exposure=1.0, vignette=0.0):
    """Grey (or RGB) image a camera records. pose = (R_body_to_world, position) of the *installed* camera
    (defaults to the planned pose), so a capture can include a deliberate mounting error. exposure scales the
    brightness (auto-exposure differs between cameras); vignette darkens towards the image circle's edge by that
    fraction."""
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
    if engine.vehicle_geom(project) is not None:
        body_d, shade = engine.cube(pos).lookup(dw)
    body = valid & np.isfinite(body_d) & (body_d > NEAR_CLIP) & (body_d < np.where(ground, tg, np.inf))
    gv = ground & ~body
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
    if exposure != 1.0 or vignette:
        img = img * exposure * (1 - vignette * np.clip(th / intr.max_theta, 0, 1) ** 2)
    if noise:
        rng = np.random.default_rng(seed)
        img = img + rng.normal(0, noise, img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8).reshape(h, w)
    if rgb:
        return np.repeat(img[..., None], 3, axis=2)
    return img
