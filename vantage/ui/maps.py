"""Coverage maps as RGBA images on the ROI grid (shared by the 3D ground texture and the plan views), and the vector
outlines the plan view draws over them (density iso-lines, blind and below-minimum regions).

Continuous fields (pixel density) are interpolated bilinearly between cell centres; values never leak across the ROI
edge or into blind cells because the field is first extended outwards from the valid cells (nearest valid value) and
then masked. Categorical fields (best camera, redundancy) keep crisp cell boundaries."""
from __future__ import annotations

import math
import weakref

import numpy as np
from PyQt6.QtGui import QImage, QColor

from . import colormaps
from .theme import THEME, STATUS


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


def _smooth_up(a, k):
    """Bilinear upsampling by k that keeps cell centres in place (texel centre (t + 0.5) / k - 0.5 in cells)."""
    if k == 1:
        return np.asarray(a, np.float32)
    import cv2
    a = np.ascontiguousarray(a, np.float32)
    h, w = a.shape
    return cv2.resize(a, (w * k, h * k), interpolation=cv2.INTER_LINEAR)


def _filled(field, valid):
    """field with every invalid cell replaced by its nearest valid cell's value, so that interpolation near the ROI
    edge, the body and blind cells never mixes in values from outside the data."""
    if valid.all() or not valid.any():
        return field
    from scipy import ndimage
    idx = ndimage.distance_transform_edt(~valid, return_distances=False, return_indices=True)
    return field[tuple(idx)]


def _flip(a):
    return a[::-1]


def blank(result, k):
    g = result.grid
    return np.zeros((g.ny * k, g.nx * k, 4), np.float32)


def _over(dst, rgb, alpha):
    """Alpha-composite rgb (3,) or (H,W,3) with alpha (H,W) over dst (H,W,4 float 0..255 / alpha 0..1), in place.
    Only the texels with alpha > 0 are touched (hatches and tints cover a small part of the image)."""
    sel = np.asarray(alpha) > 0
    if not sel.any():
        return dst
    a = np.asarray(alpha, np.float32)[sel][:, None]
    d = dst[sel]
    da = d[:, 3:4]
    out_a = a + da * (1 - a)
    src = np.asarray(rgb, np.float32)
    src = src if src.ndim == 1 else src[sel]
    d[:, :3] = (src * a + d[:, :3] * da * (1 - a)) / np.maximum(out_a, 1e-6)
    d[:, 3:4] = out_a
    dst[sel] = d
    return dst


def hatch(mask_img, k, period=6, width=2, mirror=False):
    """Diagonal stripes (45°, or 135° with mirror) inside a texel mask."""
    H, W = mask_img.shape
    jj, ii = np.mgrid[0:H, 0:W]
    stripes = (((ii - jj) if mirror else (ii + jj)) % period) < width
    return stripes & mask_img


def _mask_up(mask, k, smooth):
    """Upsampled cell mask: crisp cells, or (smooth) cut along the half-way line between cell centres."""
    return _smooth_up(mask.astype(np.float32), k) >= 0.5 if smooth and k > 1 else _up(mask, k)


def add_blind(img, result, k, alpha_fill=0.35, smooth=False):
    g = result.grid
    blind = _flip(g.roi & (result.count == 0))
    if not blind.any():
        return img
    m = _mask_up(blind, k, smooth)
    red = _rgb(STATUS['critical'])
    _over(img, red, m * alpha_fill)
    _over(img, red, hatch(m, k).astype(np.float32) * 0.85)
    return img


def add_shadow(img, result, k, strength=0.35):
    """Soft contact shadow of the body (for the 3D ground)."""
    g = result.grid
    from scipy import ndimage
    body = _flip(g.body).astype(np.float32)
    soft = ndimage.gaussian_filter(body, sigma=max(1.0, 0.12 / g.cell))
    soft = np.clip(soft * 1.4, 0, 1) * strength
    _over(img, np.zeros(3, np.float32), np.clip(_smooth_up(soft, k), 0, 1))
    return img


def footprints(result, k=4, alpha=0.12, isolate=None):
    """Each camera's ground footprint as a light tint of its colour (built per cell, then upsampled)."""
    g = result.grid
    img = blank(result, 1)
    roi = _flip(g.roi)
    for c in result.cams:
        if isolate and c.uid not in isolate:
            continue
        _over(img, _rgb(THEME.cam_hex(c.slot)), (_flip(c.vis) & roi).astype(np.float32) * alpha)
    return _up(img, k)


def coverage(result, k=4, alpha=0.42):
    """ROI cells tinted by their best camera (crisp cells)."""
    g = result.grid
    img = blank(result, 1)
    best = _flip(result.best).astype(int)
    m = _flip(g.roi) & (best >= 0)
    if len(result.cams):
        pal = np.array([_rgb(THEME.cam_hex(c.slot)) for c in result.cams], np.float32)
        img[m, :3] = pal[best[m]]
        img[m, 3] = alpha
    return _up(img, k)


_field_cache = [lambda: None, None]


def density_field(result):
    """(natural log of the best-camera density, extended outwards from the valid cells; valid mask) on the grid
    (row 0 = lowest y). Valid = in the ROI and seen by a camera. The last result's field is kept (the map, its
    iso-lines and its below-minimum outline all start from it), by weak reference."""
    if _field_cache[0]() is result:
        return _field_cache[1]
    g = result.grid
    valid = g.roi & (result.count > 0) & (result.best_dens > 0)
    f = np.log(np.maximum(result.best_dens, 1e-6)).astype(np.float32)
    try:
        ref = weakref.ref(result)
    except TypeError:
        ref = lambda: None                                  # noqa: E731 (not weak-referenceable: no caching)
    _field_cache[:] = [ref, (_filled(f, valid), valid)]
    return _field_cache[1]


def density(result, k=4, scale=None, alpha=1.0, smooth=True, flag_below=True, hatch_below=False):
    """Best-camera pixel density on the colormaps density scale (log, from the minimum density). Smooth: bilinear in
    log density between cell centres, edges cut half-way between cells. flag_below: ground below the minimum in the
    neutral grey (hatched too with hatch_below, for views that cannot draw the plan view's vector hatch)."""
    sc = scale or colormaps.density_scale()
    f, valid = density_field(result)
    f, valid = _flip(f), _flip(valid)
    lf = _smooth_up(f, k) if smooth and k > 1 else _up(f, k)
    if smooth and k > 1:
        # the ROI's outer edge and the body edge follow the smooth boundary the cell staircase approximates (a light
        # blur of the mask before cutting at one half); blind cells stay cut exactly half-way between cell centres,
        # where the blind hatch outline runs
        from scipy import ndimage
        g = result.grid
        roi_s = ndimage.gaussian_filter(_flip(g.roi).astype(np.float32), 0.7)
        m = (_smooth_up(roi_s, k) >= 0.5) & ~_mask_up(_flip(g.roi & (result.count == 0)), k, True)
    else:
        m = _up(valid, k)
    l0, l1 = math.log(sc.lo), math.log(sc.hi)
    idx = np.clip(np.rint((lf - l0) / (l1 - l0) * (len(sc.lut) - 1)), 0, len(sc.lut) - 1).astype(np.intp)
    img = np.zeros(lf.shape + (4,), np.float32)
    img[..., :3] = sc.lut[idx]
    img[..., 3] = m * alpha
    if flag_below:
        below = lf < l0 - 1e-5
        img[below, :3] = colormaps.hex_rgb(sc.below_hex) * 255
        if hatch_below:             # coarse and quiet: it is seen at grazing angles in the 3D view
            hb = hatch(below & m, k, period=max(6, 4 * k), width=max(2, k), mirror=True)
            _over(img, colormaps.hex_rgb(sc.hatch_hex) * 255, hb.astype(np.float32) * 0.35)
    return img


def redundancy(result, k=4, alpha=1.0):
    """Cameras seeing each ROI cell, 1 / 2 / 3+ on the ordered blue steps (crisp cells; 0 = blind, added by
    add_blind or drawn by the plan view)."""
    g = result.grid
    cnt = _flip(result.count)
    roi = _flip(g.roi)
    pal = np.array([colormaps.hex_rgb(h) * 255 for h in colormaps.redundancy()], np.float32)
    idx = np.clip(cnt.astype(int) - 1, 0, 2)
    img = np.zeros((g.ny, g.nx, 4), np.float32)
    img[..., :3] = pal[idx]
    img[..., 3] = (roi & (cnt > 0)) * alpha
    return _up(img, k)


def delta(res_a, res_b, k=4, span=4.0):
    """Δ best-camera density (b - a, px/cm) on a's grid, diverging scale clipped at ±span (float RGBA, alpha 0..1)."""
    g = res_a.grid
    d = _flip(res_b.best_dens - res_a.best_dens)
    roi = _flip(g.roi)
    lut = colormaps.diverging().astype(np.float32)
    t = np.clip(d / span, -1, 1)
    idx = np.clip(np.rint((t + 1) / 2 * (len(lut) - 1)), 0, len(lut) - 1).astype(np.intp)
    img = np.zeros((g.ny, g.nx, 4), np.float32)
    img[..., :3] = lut[idx]
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
    shows on the theme's blue ramp (none: nearest the background, all of it: farthest from it); vanishing at a seam
    orange, cut off at the view's edge dotted, blind red."""
    from ..core.stitched_objects import map_rgba
    rgb = lambda h: tuple(int(v) for v in _rgb(h))
    return map_rgba(sres, k, stitched_palette(), vanish_rgb=rgb(STATUS['serious']), blind_rgb=rgb(STATUS['critical']))


def stitched_palette():
    """Steps for 'share of the object shown', none … all: the theme's density ramp (more = more contrast)."""
    return colormaps.DENSITY['light' if THEME.name == 'light' else 'dark']


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


def _aa_up(mask, k):
    """Upsampled cell mask with an anti-aliased edge (alpha 0..1) along the half-way line between cell centres: no
    staircase of whole cells at the rim."""
    if k == 1:
        return mask.astype(np.float32)
    s = _smooth_up(mask.astype(np.float32), k)
    return np.clip((s - 0.5) * k + 0.5, 0.0, 1.0)


def soften_rim(img, result, k):
    """Cut the map's outer rim (the analysis domain's edge, holes such as the body kept) along a smooth anti-aliased
    line instead of whole cells, in place."""
    g = result.grid
    from scipy import ndimage
    dom = ndimage.binary_fill_holes(_flip(g.roi | g.body))
    if dom.all() or not dom.any():
        return img
    img[..., 3] *= np.minimum(_aa_up(dom, k), 1.0)
    return img


def make(result, mode, k=4, isolate=None, shadow=True, blind=True, thresholds=None, smooth=True):
    """RGBA uint8 image (row 0 = max y) of a heatmap mode on the result's grid, k texels per cell.
    thresholds: (minimum, target) density in px/cm; with them the density map starts at the minimum and marks ground
    below it (grey, hatched when blind=True); without them it keeps a fixed 3–16 px/cm scale and flags nothing.
    blind: hatch blind ROI cells into the image (the plan view draws its own vector hatch instead)."""
    if result is None:
        return None
    sm = smooth and mode == 'density'
    if mode == 'density':
        sc = colormaps.density_scale(*thresholds) if thresholds else colormaps.density_scale()
        img = density(result, k, sc, smooth=sm, flag_below=thresholds is not None, hatch_below=blind)
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
        add_blind(img, result, k, smooth=sm)
    if shadow:
        add_shadow(img, result, k)
    if mode in ('footprints', 'blank', 'coverage', 'redundancy', 'density'):
        soften_rim(img, result, k)
    out = np.clip(img, 0, 255)
    out[..., 3] = np.clip(img[..., 3] * 255, 0, 255)
    return out.astype(np.uint8)


# ------------------------------------------------------------------ vector outlines for the plan view
def mask_outline(mask, grid):
    """Closed outlines (list of (n, 2) world polylines) of a cell mask, running half-way between cell centres (corners
    cut), for filling with a pattern (odd-even rule)."""
    from ..core.contour import iso_segments, chain
    if not mask.any():
        return []
    m = np.pad(mask.astype(np.float32), 1)
    c = grid.cell
    return chain(iso_segments(m, 0.5, grid.x0 - c, grid.y0 - c, c), tol=c * 1e-3)


def blind_outline(result):
    """Outlines of the blind ROI cells (no camera sees them)."""
    return mask_outline(result.grid.roi & (result.count == 0), result.grid)


def unique_outline(result, uid):
    """Outlines of the ROI ground that only camera uid sees (what the 'seen only by this camera' bars measure)."""
    i = result.cam_index(uid)
    if i < 0:
        return []
    return mask_outline(result.grid.roi & result.cams[i].vis & (result.count == 1), result.grid)


def valid_outline(result):
    """Outlines of the seen ROI ground (in the ROI, seen by a camera): the clip for below_min_outline."""
    return mask_outline(density_field(result)[1], result.grid)


def below_min_outline(result, minv):
    """Closed outlines of the ground whose interpolated best-camera density is below minv (the grey of the density
    map). They follow the density's own iso-line and run past the data at the grid's edge: clip them with
    valid_outline (blind cells and the ROI edge cut them half-way between cell centres, like the map)."""
    from ..core.contour import iso_segments, chain
    if not minv or minv <= 0:
        return []
    f, valid = density_field(result)
    ind = f - math.log(minv)
    if not (ind[valid] < 0).any():
        return []
    g = result.grid
    # a ring of edge values, then a ring above the level: loops close one cell outside the grid
    ind = np.pad(np.pad(ind, 1, mode='edge'), 1, constant_values=max(1.0, float(ind.max()) + 1.0))
    c = g.cell
    return chain(iso_segments(-ind, 0.0, g.x0 - 2 * c, g.y0 - 2 * c, c), tol=c * 1e-3)


def density_contours(result, levels):
    """Iso-lines of the best-camera density at each level (px/cm) -> {level: [(n, 2) world polylines]}, only where
    the ground is in the ROI and seen (not along the ROI edge or around blind cells)."""
    from ..core.contour import iso_segments, chain
    g = result.grid
    f, valid = density_field(result)
    out = {}
    for lv in levels:
        if not lv or lv <= 0:
            continue
        segs = iso_segments(f, math.log(lv), g.x0, g.y0, g.cell)
        if len(segs):
            mid = segs.mean(axis=1)
            ii = np.clip(((mid[:, 0] - g.x0) / g.cell).astype(int), 0, g.nx - 1)
            jj = np.clip(((mid[:, 1] - g.y0) / g.cell).astype(int), 0, g.ny - 1)
            segs = segs[valid[jj, ii]]
        out[lv] = chain(segs, tol=g.cell * 1e-3) if len(segs) else []
    return out


def extent(result):
    g = result.grid
    return (g.x0, g.y0, g.x0 + g.nx * g.cell, g.y0 + g.ny * g.cell)
