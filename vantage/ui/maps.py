"""Coverage maps as RGBA images on the ROI grid (shared by the 3D ground texture and the plan views)."""
from __future__ import annotations

import math

import numpy as np
from PyQt6.QtGui import QImage, QColor

from .theme import THEME, SEQ, REDUNDANCY, DIVERGING, STATUS, CAMERA


def _rgb(hexs):
    c = QColor(hexs)
    return np.array([c.red(), c.green(), c.blue()], np.float32)


def to_qimage(rgba: np.ndarray) -> QImage:
    rgba = np.ascontiguousarray(rgba, np.uint8)
    h, w = rgba.shape[:2]
    img = QImage(rgba.data, w, h, w * 4, QImage.Format.Format_RGBA8888)
    return img.copy()


def _up(a, k):
    return np.repeat(np.repeat(a, k, axis=0), k, axis=1)


def _flip(a):
    return a[::-1]


def blank(result, k):
    g = result.grid
    return np.zeros((g.ny * k, g.nx * k, 4), np.float32)


def _over(dst, rgb, alpha):
    """Alpha-composite rgb (3,) or (H,W,3) with alpha (H,W) over dst (H,W,4 float 0..255 / alpha 0..1)."""
    a = alpha[..., None]
    da = dst[..., 3:4]
    out_a = a + da * (1 - a)
    rgb = np.broadcast_to(rgb, dst[..., :3].shape)
    dst[..., :3] = np.where(out_a > 0, (rgb * a + dst[..., :3] * da * (1 - a)) / np.maximum(out_a, 1e-6), 0)
    dst[..., 3:4] = out_a
    return dst


def hatch(mask_img, k, period=6, width=2):
    H, W = mask_img.shape
    jj, ii = np.mgrid[0:H, 0:W]
    stripes = ((ii + jj) % period) < width
    return stripes & mask_img


def add_blind(img, result, k, alpha_fill=0.35):
    g = result.grid
    blind = _flip(g.roi & (result.count == 0))
    m = _up(blind, k)
    if not m.any():
        return img
    red = _rgb(STATUS['critical'])
    _over(img, red, m * alpha_fill)
    _over(img, red, hatch(m, k).astype(np.float32) * 0.85)
    return img


def add_shadow(img, result, k, strength=0.35):
    g = result.grid
    from scipy import ndimage
    body = _flip(g.body).astype(np.float32)
    soft = ndimage.gaussian_filter(body, sigma=max(1.0, 0.12 / g.cell))
    soft = np.clip(soft * 1.4, 0, 1) * strength
    _over(img, np.zeros(3, np.float32), _up(soft, k))
    return img


def footprints(result, k=4, alpha=0.12, isolate=None):
    img = blank(result, k)
    g = result.grid
    roi = _flip(g.roi)
    for c in result.cams:
        if isolate and c.uid not in isolate:
            continue
        m = _up(_flip(c.vis) & roi, k).astype(np.float32)
        _over(img, _rgb(THEME.cam_hex(c.slot)), m * alpha)
    return img


def coverage(result, k=4, alpha=0.42):
    img = blank(result, k)
    g = result.grid
    roi = _flip(g.roi)
    best = _flip(result.best)
    for i, c in enumerate(result.cams):
        m = _up((best == i) & roi, k).astype(np.float32)
        _over(img, _rgb(THEME.cam_hex(c.slot)), m * alpha)
    return img


def density(result, k=4, lo=3.0, hi=16.0, alpha=1.0):
    g = result.grid
    v = _flip(result.best_dens)
    roi = _flip(g.roi) & (v > 0)
    t = (np.log(np.maximum(v, lo)) - math.log(lo)) / (math.log(hi) - math.log(lo))
    idx = np.clip(np.round(t * (len(SEQ) - 1)), 0, len(SEQ) - 1).astype(int)
    pal = np.array([_rgb(s) for s in SEQ])
    rgb = pal[idx]
    img = np.zeros((g.ny, g.nx, 4), np.float32)
    img[..., :3] = rgb
    img[..., 3] = roi * alpha
    return _up(img, k)


def redundancy(result, k=4, alpha=1.0):
    g = result.grid
    cnt = _flip(result.count)
    roi = _flip(g.roi)
    pal = np.array([_rgb(s) for s in REDUNDANCY])
    idx = np.clip(cnt - 1, 0, 2)
    img = np.zeros((g.ny, g.nx, 4), np.float32)
    img[..., :3] = pal[idx]
    img[..., 3] = (roi & (cnt > 0)) * alpha
    return _up(img, k)


def delta(res_a, res_b, k=4, span=4.0):
    """Δ best-camera density (b - a) on a's grid, diverging palette."""
    g = res_a.grid
    d = _flip(res_b.best_dens - res_a.best_dens)
    roi = _flip(g.roi)
    pal = np.array([_rgb(s) for s in DIVERGING])
    if THEME.name == 'light':
        pal[3] = _rgb('#e1e0d9')
    t = np.clip(d / span, -1, 1)
    idx = np.clip(np.round((t + 1) / 2 * (len(pal) - 1)), 0, len(pal) - 1).astype(int)
    img = np.zeros((g.ny, g.nx, 4), np.float32)
    img[..., :3] = pal[idx]
    img[..., 3] = roi * 1.0
    return _up(img, k)


def bev(result, k=4, alpha=0.30, grid_m=0.5):
    """Stitched bird's-eye: each cell tinted by the camera that owns its sector; blind cells hatched."""
    g = result.grid
    img = blank(result, k)
    sec = _flip(result.sector)
    roi = _flip(g.roi)
    for i, c in enumerate(result.cams):
        m = _up((sec == i) & roi, k).astype(np.float32)
        _over(img, _rgb(THEME.cam_hex(c.slot)), m * alpha)
    # sector owner cannot see the cell -> hole in the stitched image
    if len(result.cams):
        V = np.stack([c.vis for c in result.cams])
        own = np.where(result.sector >= 0, result.sector, 0)
        seen = np.take_along_axis(V, own[None].astype(np.int64), 0)[0]
        hole = _flip(g.roi & (result.sector >= 0) & ~seen & (result.count > 0))
        if hole.any():
            m = _up(hole, k)
            _over(img, _rgb(STATUS['warning']), hatch(m, k).astype(np.float32) * 0.7)
    # metric grid
    H, W = img.shape[:2]
    step = grid_m / g.cell * k
    xs = (np.arange(W) + 0.5) / k * g.cell + g.x0
    ys = g.y0 + (g.ny * k - np.arange(H) - 0.5) / k * g.cell
    gx = (np.abs(((xs / grid_m) + 0.5) % 1 - 0.5) * grid_m) < g.cell / k * 0.6
    gy = (np.abs(((ys / grid_m) + 0.5) % 1 - 0.5) * grid_m) < g.cell / k * 0.6
    gl = (gx[None, :] | gy[:, None]) & _up(roi, k)
    _over(img, np.array([255, 255, 255], np.float32), gl.astype(np.float32) * 0.10)
    return img


def stitched(sres, k=4) -> np.ndarray:
    """Objects standing in the stitched bird's-eye view (core.stitched_objects): how much of the object the view
    shows, pale (none) to dark blue (all of it); vanishing at a seam orange, cut off at the view's edge dotted,
    blind red."""
    from ..core.stitched_objects import map_rgba
    rgb = lambda h: tuple(int(v) for v in _rgb(h))
    return map_rgba(sres, k, SEQ, vanish_rgb=rgb(STATUS['serious']), blind_rgb=rgb(STATUS['critical']))


def stitched_zones(sres, k=4) -> np.ndarray:
    """Over the bird's-eye view's sector tints: where a standing object vanishes at a seam (orange, hatched) or is cut
    off at the view's edge (dotted)."""
    H, W = sres.ny * k, sres.nx * k
    img = np.zeros((H, W, 4), np.float32)
    van = _up(_flip(sres.vanish), k)
    cut = _up(_flip(sres.cut), k)
    jj, ii = np.mgrid[0:H, 0:W]
    _over(img, _rgb(STATUS['serious']), van * 0.45)
    _over(img, np.array([250, 250, 250], np.float32), (van & (((ii + jj) % 6) < 2)).astype(np.float32) * 0.9)
    _over(img, _rgb(THEME.tok['ink']), (cut & (ii % 4 == 1) & (jj % 4 == 1)).astype(np.float32) * 0.7)
    out = np.clip(img, 0, 255)
    out[..., 3] = np.clip(img[..., 3] * 255, 0, 255)
    return out.astype(np.uint8)


def stitched_extent(sres):
    return (sres.x0, sres.y0, sres.x0 + sres.nx * sres.cell, sres.y0 + sres.ny * sres.cell)


def targets_redundancy(result, k=4):
    return redundancy(result, k)


def make(result, mode, k=4, isolate=None, shadow=True, blind=True):
    if result is None:
        return None
    if mode == 'density':
        img = density(result, k)
    elif mode == 'redundancy':
        img = redundancy(result, k)
    elif mode == 'coverage':
        img = coverage(result, k)
    elif mode == 'bev':
        img = bev(result, k)
    elif mode == 'footprints':
        img = footprints(result, k, isolate=isolate)
    else:
        img = blank(result, k)
    if blind:
        add_blind(img, result, k)
    if shadow:
        add_shadow(img, result, k)
    out = np.clip(img, 0, 255)
    out[..., 3] = np.clip(img[..., 3] * 255, 0, 255)
    return out.astype(np.uint8)


def extent(result):
    g = result.grid
    return (g.x0, g.y0, g.x0 + g.nx * g.cell, g.y0 + g.ny * g.cell)
