"""Per-unit lens calibration.

Every physical camera differs a little from its lens datasheet: the focal length by a percent or two, the principal
point by tens of pixels. A unit calibration is that camera's own intrinsics, measured from checkerboard images
(Kannala-Brandt, the cv::fisheye model) or read from a calibration file made elsewhere (OpenCV YAML/XML, ROS
camera_info, Kalibr camchain, JSON). It is stored on the camera (Camera.calib) and replaces the library lens wherever
the lens model is used: the analysis, synthetic captures, the installation check, the remap tables and the exports.

Records use OpenCV pixel coordinates (pixel centre at integer i) at the resolution the calibration was made at;
unit_lens() converts them to the sensor's own coordinates (pixel centre at i + 0.5, full sensor resolution).
"""
from __future__ import annotations

import json
import math
import os
import re
import time
from dataclasses import dataclass, field, replace

import numpy as np

from .lens import Intrinsics, Lens, Module, resolve

SCHEMA = 'vantage.unit_calib/1'
KEYS = ('width', 'height', 'fx', 'fy', 'cx', 'cy', 'k')
_cache: dict = {}


# ---------------------------------------------------------------- records
def make_record(width, height, fx, fy, cx, cy, k, rms_px=0.0, serial='', source='', views=0, date=None) -> dict:
    k = [float(x) for x in list(k)[:4]]
    return {'model': 'kannala_brandt', 'width': int(width), 'height': int(height), 'fx': float(fx), 'fy': float(fy),
            'cx': float(cx), 'cy': float(cy), 'k': k + [0.0] * (4 - len(k)), 'rms_px': float(rms_px),
            'serial': str(serial or ''), 'source': str(source or ''), 'views': int(views),
            'date': date or time.strftime('%Y-%m-%d')}


def clean_record(rec: dict) -> dict:
    """A record with only the stored keys (drops import hints such as 'hint' and 'file')."""
    keep = ('model', 'width', 'height', 'fx', 'fy', 'cx', 'cy', 'k', 'rms_px', 'serial', 'source', 'views', 'date')
    return {k: rec[k] for k in keep if k in rec}


def unit_lens(base: Lens, module: Module, rec: dict) -> Lens:
    """The library lens with a unit calibration applied, at the module's full resolution."""
    key = (base.id, base.name, base.focal_mm, base.max_half_deg, tuple(base.sheet_fov), base.f_number, base.focus_near_m,
           base.focus_far_m, module.width_px, module.height_px, module.pixel_um, json.dumps(rec, sort_keys=True))
    hit = _cache.get(key)
    if hit is not None:
        return hit
    sx = module.width_px / float(rec['width'])
    sy = module.height_px / float(rec['height'])
    fx, fy = float(rec['fx']) * sx, float(rec['fy']) * sy
    cx, cy = (float(rec['cx']) + 0.5) * sx, (float(rec['cy']) + 0.5) * sy
    pm = module.pixel_um * 1e-3
    tag = rec.get('serial') or 'unit'
    lens = replace(base, name=f'{base.name} · {tag}', label='unit', focal_mm=fx * pm, projection='kannala_brandt',
                   kb=tuple(float(x) for x in rec['k'][:4]), pp_offset_px=(cx - module.width_px / 2, cy - module.height_px / 2),
                   aspect=fy / fx, measured=[], fit_rms_px=float(rec.get('rms_px', 0.0)), builtin=False)
    if len(_cache) > 256:
        _cache.clear()
    _cache[key] = lens
    return lens


def nominal_record(module: Module, lens: Lens) -> dict:
    """The library lens expressed as a record (sensor resolution, OpenCV pixel coordinates)."""
    from .exporters import kb_coefficients
    intr = resolve(module, lens)
    k, _ = kb_coefficients(intr)
    return make_record(intr.width, intr.height, intr.f_px, intr.f_px * intr.aspect, intr.cx - 0.5, intr.cy - 0.5, k,
                       source='library', date='')


def problems(rec: dict, module: Module, base: Lens | None = None) -> list[str]:
    """Reasons a record does not fit this camera (empty when it looks right)."""
    out = []
    missing = [k for k in KEYS if k not in rec]
    if missing:
        return [f'missing {", ".join(missing)}']
    w, h = float(rec['width']), float(rec['height'])
    if w <= 0 or h <= 0 or rec['fx'] <= 0 or rec['fy'] <= 0:
        return ['resolution and focal length must be positive']
    if abs(w / h - module.width_px / module.height_px) > 0.005 * module.width_px / module.height_px:
        out.append(f'{int(w)} × {int(h)} does not have the {module.width_px} × {module.height_px} sensor\'s aspect '
                   '(a cropped or binned mode?)')
    if abs(rec['fy'] / rec['fx'] - 1) > 0.03:
        out.append(f'fy/fx = {rec["fy"] / rec["fx"]:.3f}; expected square pixels')
    s = module.width_px / w
    off = math.hypot((rec['cx'] + 0.5) * s - module.width_px / 2, (rec['cy'] + 0.5) * module.height_px / h - module.height_px / 2)
    if off > 0.08 * module.width_px:
        out.append(f'principal point is {off:.0f} px from the sensor centre')
    if base is not None:
        f_nom = resolve(module, base).f_px
        f_rec = rec['fx'] * s
        if abs(f_rec / f_nom - 1) > 0.12:
            out.append(f'focal length {f_rec:.0f} px is {100 * (f_rec / f_nom - 1):+.0f} % from {base.name} ({f_nom:.0f} px)')
    return out


def fatal_problems(rec: dict, module: Module) -> list[str]:
    """The problems that make a record unusable for this camera (as opposed to ones worth a look)."""
    out = [m for m in problems(rec, module) if m.startswith(('missing', 'resolution and')) or 'aspect' in m]
    return out


def model_difference(a: Intrinsics, b: Intrinsics, n: int = 48, max_theta: float | None = None) -> dict:
    """How far apart two lens models put the same ray, over b's image: max / rms pixels and max degrees.
    Rays are taken from b (the measured unit) and projected with a (e.g. the nominal lens); max_theta (radians)
    limits the comparison to the field a calibration covered."""
    us = (np.arange(n) + 0.5) / n * b.width
    vs = (np.arange(n) + 0.5) / n * b.height
    U, V = np.meshgrid(us, vs)
    d, th = b.unproject(U.ravel(), V.ravel())
    keep = th <= min(b.max_theta, math.radians(89.5) if b.projection == 'rectilinear' else b.max_theta,
                     max_theta if max_theta else math.inf)
    d, U, V = d[keep], U.ravel()[keep], V.ravel()[keep]
    if not len(d):
        return {'max_px': 0.0, 'rms_px': 0.0, 'max_deg': 0.0, 'centre_px': 0.0}
    ua, va, _, _ = a.project_dirs(d)
    e = np.hypot(ua - U, va - V)
    da, _ = a.unproject(U, V)
    ang = np.degrees(np.arccos(np.clip(np.sum(da * d, axis=1), -1, 1)))
    centre = np.hypot(a.cx - b.cx, a.cy - b.cy)
    return {'max_px': float(e.max()), 'rms_px': float(np.sqrt(np.mean(e ** 2))), 'max_deg': float(ang.max()),
            'centre_px': float(centre)}


def compare_rows(module: Module, base: Lens, rec: dict) -> list[tuple]:
    """[(parameter, nominal, unit, difference text)] for display."""
    nom = nominal_record(module, base)
    s = module.width_px / rec['width']
    u = dict(rec, fx=rec['fx'] * s, fy=rec['fy'] * module.height_px / rec['height'],
             cx=(rec['cx'] + 0.5) * s - 0.5, cy=(rec['cy'] + 0.5) * module.height_px / rec['height'] - 0.5)
    rows = []
    for k, lab, dp in (('fx', 'fx px', 2), ('fy', 'fy px', 2), ('cx', 'cx px', 2), ('cy', 'cy px', 2)):
        d = u[k] - nom[k]
        extra = f' ({100 * d / nom[k]:+.2f} %)' if k in ('fx', 'fy') else ''
        rows.append((lab, f'{nom[k]:.{dp}f}', f'{u[k]:.{dp}f}', f'{d:+.{dp}f}{extra}'))
    for i in range(4):
        rows.append((f'k{i + 1}', f'{nom["k"][i]:+.5f}', f'{u["k"][i]:+.5f}', f'{u["k"][i] - nom["k"][i]:+.5f}'))
    return rows


# ---------------------------------------------------------------- board detection
def board_points(pattern, square_mm) -> np.ndarray:
    """(N, 3) board-frame coordinates (metres) of the inner corners, row-major, pattern = (columns, rows)."""
    cols, rows = pattern
    sq = square_mm / 1000.0
    X, Y = np.meshgrid(np.arange(cols) * sq, np.arange(rows) * sq)
    return np.stack([X.ravel(), Y.ravel(), np.zeros(X.size)], 1)


def _gray(img):
    import cv2
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img


def _sb(gray, pattern, exhaustive=False):
    import cv2
    from .install import find_sb      # repeatable: the same corners whatever ran before
    flags = cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_ACCURACY | (cv2.CALIB_CB_EXHAUSTIVE if exhaustive else 0)
    ok, c = find_sb(gray, pattern, flags)
    return c.reshape(-1, 2).astype(np.float64) if ok else None


# The board pre-check: a cheap test that a frame can hold a chessboard at all, so a frame with none is turned away
# in tens of milliseconds instead of running the detector's searches (seconds to minutes on an empty, noisy frame).
# It looks for the board's inner corners as saddle points of the image (Hessian determinant < 0) on an image pyramid
# and keeps those that (1) stand above the noise, (2) have some contrast (the saddle's response over that of an ideal
# sharp corner per grey level of local mean brightness, so independent of exposure and gain: a sharp black-on-white
# print gives 0.8–1.8; blur, haze and flare lower it), (3) have two nearby saddles of the opposite polarity (dark and
# bright squares swapped, as at the next corner along an edge) in two different directions, and (4) form, linked to
# such neighbours, one connected two-dimensional group. A frame passes when some pyramid level has a group of at
# least PRE_FRAC of the board's inner corners. The thresholds are heuristics chosen to keep the boards the detectors
# find (rendered lens and lidar boards through the library lenses and the sample's cameras: distance, tilt, fisheye
# edge, blur, noise, exposure, gamma, haze, clipping; of 419 found, one blurred by ~0.4 of a square was turned away)
# with margin, which lets many cluttered board-free scenes through; for those the groups' places at least narrow the
# searches: a rectified view that holds no group's worth of corners is skipped (board_groups, groups_in_view). The
# measurements are summarised in tests/test_board_reject.py.
PRE_SIGMA = 1.0          # px at each pyramid level: the derivative scale
PRE_NOISE_K = 8.0        # a saddle's response must exceed this many robust σ of the image's mixed derivative
PRE_CONTRAST = 0.15      # minimum contrast (2)
PRE_FRAC = 0.5           # the group must hold this share of the board's inner corners
PRE_MAX_SIDE = 2000      # px: larger frames are looked at downscaled to this
_pre_unit: dict = {}


def _pre_derivs(f, sigma):
    import cv2
    b = cv2.GaussianBlur(f, (0, 0), sigma)
    return (cv2.Sobel(b, cv2.CV_32F, 2, 0, ksize=3), cv2.Sobel(b, cv2.CV_32F, 0, 2, ksize=3),
            cv2.Sobel(b, cv2.CV_32F, 1, 1, ksize=3))


def _pre_unit_response(sigma) -> float:
    """Saddle response per grey level of step at the centre of an ideal, sharp chessboard corner."""
    if sigma not in _pre_unit:
        n = 64
        y, x = np.mgrid[0:n, 0:n] + 0.5 - n / 2
        f = (np.sign(x) * np.sign(y) * 50 + 100).astype(np.float32)
        a, c, b = _pre_derivs(f, sigma)
        _pre_unit[sigma] = float(np.sqrt(np.maximum(b * b - a * c, 0)).max()) / 100.0
    return _pre_unit[sigma]


def _lattice_group(f, contrast=PRE_CONTRAST, sigma=PRE_SIGMA, k=10, near=2.0, sep_deg=25.0, d_min=3.0,
                   min_group=6, spread=0.5, groups=None) -> int:
    """Size of the largest two-dimensional group of chessboard-like saddle points in one pyramid level f (float32
    grey); 0 when there is none. groups: a list that receives every such group's points (n, 2), level pixels."""
    import cv2
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree
    a, c, b = _pre_derivs(f, sigma)
    r = np.sqrt(np.maximum(b * b - a * c, 0))
    noise = float(np.median(np.abs(b))) * 1.4826 + 1e-6
    ys, xs = np.nonzero((r >= cv2.dilate(r, np.ones((5, 5), np.uint8))) & (r > PRE_NOISE_K * noise))
    if len(xs) < min_group:
        return 0
    mu = cv2.GaussianBlur(f, (0, 0), 3.0)[ys, xs]
    keep = r[ys, xs] / _pre_unit_response(sigma) / np.maximum(mu, 4.0) >= contrast
    xs, ys = xs[keep], ys[keep]
    n = len(xs)
    if n < min_group:
        return 0
    A, B, C = (m[ys, xs].astype(np.float64) for m in (a, b, c))
    phi = 0.5 * np.arctan2(2 * B, A - C)                     # each saddle's direction of positive curvature
    xy = np.c_[xs, ys].astype(np.float64)
    d, idx = cKDTree(xy).query(xy, k=min(k + 1, n))
    d, idx = d[:, 1:], idx[:, 1:]
    v = xy[idx] - xy[:, None]
    ang = np.arctan2(v[..., 1], v[..., 0])
    cp, sp = np.cos(phi), np.sin(phi)
    # opposite polarity: the neighbour's curvature along this saddle's positive direction is negative
    ok = (A[idx] * (cp * cp)[:, None] + 2 * B[idx] * (cp * sp)[:, None] + C[idx] * (sp * sp)[:, None] < 0) & \
        (d >= d_min)
    dn = np.min(np.where(ok, d, np.inf), 1)
    ok &= d <= near * dn[:, None]
    rows = np.arange(n)
    a0 = ang[rows, np.argmin(np.where(ok, d, np.inf), 1)]
    other = ok & (np.abs((ang - a0[:, None] + math.pi / 2) % math.pi - math.pi / 2) >= math.radians(sep_deg))
    good = ok.any(1) & other.any(1)
    link = ok & good[:, None] & good[idx]
    ii, kk = np.nonzero(link)
    if len(ii) < min_group:
        return 0
    _, lab = connected_components(coo_matrix((np.ones(len(ii)), (ii, idx[ii, kk])), shape=(n, n)), directed=False)
    lab = np.where(good, lab, -1)
    sizes = np.bincount(lab[lab >= 0])
    best = 0
    for g in np.nonzero(sizes >= min_group)[0]:
        if sizes[g] <= best and groups is None:
            continue
        P = xy[lab == g]
        step = float(np.median(d[ii, kk][lab[ii] == g]))
        minor = math.sqrt(max(float(np.linalg.eigvalsh(np.cov((P - P.mean(0)).T))[0]), 0.0))
        if minor >= spread * step:                           # two-dimensional, not a chain along one edge
            best = max(best, len(P))
            if groups is not None:
                groups.append(P)
    return best


def board_lattice(img, need: int | None = None, max_side=PRE_MAX_SIDE, groups=None) -> int:
    """Largest group of chessboard-like corners in the image over its pyramid levels (see PRE_*), coarse levels
    first; with need given, stops at the first level that reaches it, unless groups (a list) is given: it then
    receives every group of at least need points, as OpenCV pixel coordinates of img (n, 2)."""
    import cv2
    g = _gray(img)
    h, w = g.shape[:2]
    if min(h, w) < 48:
        return 0
    f = g.astype(np.float32)
    s = min(1.0, max_side / max(h, w))
    if s < 1.0:
        f = cv2.resize(f, (max(int(round(w * s)), 1), max(int(round(h * s)), 1)), interpolation=cv2.INTER_AREA)
    levels = [f]
    while min(levels[-1].shape) >= 96:
        p = levels[-1]
        levels.append(cv2.resize(p, (p.shape[1] // 2, p.shape[0] // 2), interpolation=cv2.INTER_AREA))
    best = 0
    for f in reversed(levels):
        if min(f.shape) < 48:
            continue
        found = [] if groups is not None else None
        best = max(best, _lattice_group(f, groups=found))
        if found:
            k = np.array([w / f.shape[1], h / f.shape[0]])
            groups += [(P + 0.5) * k - 0.5 for P in found if len(P) >= (need or 0)]
        elif need is not None and best >= need:
            break
    return best


def board_groups(img, pattern) -> list:
    """The image's groups of chessboard-like corners large enough for a board of pattern = (columns, rows): a list of
    (n, 2) OpenCV pixel coordinates, empty when board_possible is False. A board the detector finds anywhere lies
    round one of them."""
    cols, rows = pattern
    out = []
    board_lattice(img, int(math.ceil(PRE_FRAC * cols * rows)), groups=out)
    return out


def groups_in_view(rays, Rv, tan_half, pattern) -> bool:
    """Whether a virtual pinhole view (axes Rv, columns in the optical frame; half-width tan_half) holds at least
    half of a board's worth (PRE_FRAC × corners / 2) of the points of one group (rays: list of (n, 3) unit rays)."""
    need = 0.5 * PRE_FRAC * pattern[0] * pattern[1]
    for r in rays:
        loc = r @ Rv
        z = np.maximum(loc[:, 2], 1e-9)
        inside = (loc[:, 2] > 0) & (np.abs(loc[:, 0] / z) <= tan_half) & (np.abs(loc[:, 1] / z) <= tan_half)
        if inside.sum() >= need:
            return True
    return False


def group_rays(groups, intr: Intrinsics, sx, sy) -> list:
    """board_groups' points (pixels of an image scaled sx, sy from the sensor) as unit rays in the optical frame."""
    out = []
    for P in groups:
        d, _ = intr.unproject((P[:, 0] + 0.5) / sx, (P[:, 1] + 0.5) / sy)
        d = np.asarray(d, np.float64)
        ok = np.all(np.isfinite(d), axis=1)
        out.append(d[ok] / np.linalg.norm(d[ok], axis=1, keepdims=True))
    return out


def board_possible(img, pattern) -> bool:
    """False when the image cannot hold a chessboard of pattern = (columns, rows) inner corners: the pre-check run
    before the detector's searches. It never decides where the board is; when it passes the detector runs as before."""
    cols, rows = pattern
    need = int(math.ceil(PRE_FRAC * cols * rows))
    return board_lattice(img, need) >= need


def _view_dirs():
    """Directions of the virtual pinhole views tried when the board is not found in the raw fisheye image."""
    out = [(0.0, 0.0)]
    for off in (38.0, 66.0):
        for az in range(0, 360, 45):
            out.append((off, float(az)))
    return out


def find_board(img, pattern, intr: Intrinsics | None = None) -> np.ndarray | None:
    """Inner corners (N, 2) of a chessboard in a camera image, in OpenCV pixel coordinates of that image; None when
    not found. pattern = (columns, rows) of inner corners. With the lens model (intr) given, a board the raw fisheye
    image distorts too much is looked for in rectified pinhole views and its corners mapped back exactly. A frame the
    pre-check (board_possible) finds no chessboard-like corners in is turned away first."""
    import cv2
    from .install import _virtual_basis
    g = _gray(img)
    if not board_possible(g, pattern):
        return None
    c = _sb(g, pattern)
    if c is not None:
        return c
    if intr is None:
        return _sb(g, pattern, exhaustive=True)
    h, w = g.shape
    sx, sy = w / intr.width, h / intr.height
    n = 900
    half = math.radians(48)
    f = (n / 2) / math.tan(half)
    rays = group_rays(board_groups(g, pattern), intr, sx, sy)
    for off, az in _view_dirs():
        t, a = math.radians(off), math.radians(az)
        z = np.array([math.sin(t) * math.cos(a), math.sin(t) * math.sin(a), math.cos(t)])
        Rv = _virtual_basis(z)
        if not groups_in_view(rays, Rv, math.tan(half), pattern):     # no board-like corners in this view
            continue
        xs = (np.arange(n) + 0.5 - n / 2) / f
        X, Y = np.meshgrid(xs, xs)
        rv = np.stack([X, Y, np.ones_like(X)], -1).reshape(-1, 3)
        ro = rv @ Rv.T
        u, v, th, ok = intr.project_dirs(ro)
        mx = np.where(ok, u * sx - 0.5, -1).reshape(n, n).astype(np.float32)
        my = np.where(ok, v * sy - 0.5, -1).reshape(n, n).astype(np.float32)
        patch = cv2.remap(g, mx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        if not board_possible(patch, pattern):
            continue
        det = _sb(patch, pattern)
        if det is None:
            continue
        rv = np.c_[(det - n / 2 + 0.5) / f, np.ones(len(det))]
        ro = rv @ Rv.T
        u, v, _, okd = intr.project_dirs(ro)
        if not okd.all():
            continue
        return np.stack([u * sx - 0.5, v * sy - 0.5], -1)
    return None


# ---------------------------------------------------------------- calibration
@dataclass
class CalibView:
    name: str
    size: tuple                       # (w, h)
    corners: np.ndarray | None = None
    rms: float = float('nan')
    used: bool = False
    note: str = ''


@dataclass
class CalibResult:
    record: dict | None
    views: list
    pattern: tuple
    square_mm: float
    width: int = 0
    height: int = 0
    rms: float = float('nan')
    coverage: np.ndarray | None = None     # (rows, cols) corner counts over the image
    coverage_frac: float = 0.0             # share of the image-circle cells with corners
    ok: bool = False
    message: str = ''
    advice: list = field(default_factory=list)


def _initial(intr: Intrinsics, w, h):
    from .exporters import kb_coefficients
    s = w / intr.width
    t = h / intr.height
    K = np.array([[intr.f_px * s, 0, intr.cx * s - 0.5], [0, intr.f_px * intr.aspect * t, intr.cy * t - 0.5], [0, 0, 1]])
    k, _ = kb_coefficients(intr)
    return K, np.array(k, float).reshape(4, 1)


def _fisheye(obj, pts, size, K0, D0):
    import cv2
    # OpenCV 5 dropped the fisheye-only flags for the general CALIB_* ones, with different values
    fl = (lambda n: getattr(cv2.fisheye, n)) if hasattr(cv2.fisheye, 'CALIB_FIX_SKEW') else (lambda n: getattr(cv2, n))
    flags = fl('CALIB_USE_INTRINSIC_GUESS') | fl('CALIB_RECOMPUTE_EXTRINSIC') | fl('CALIB_FIX_SKEW')
    crit = (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 200, 1e-10)
    o = [x.reshape(1, -1, 3).astype(np.float64) for x in obj]      # (1, N, 3): OpenCV 5 rejects (N, 1, 3) here
    p = [x.reshape(1, -1, 2).astype(np.float64) for x in pts]
    rms, K, D, rv, tv = cv2.fisheye.calibrate(o, p, size, K0.copy(), D0.copy(), flags=flags, criteria=crit)
    errs = []
    for oo, pp, r, t in zip(o, p, rv, tv):
        proj, _ = cv2.fisheye.projectPoints(oo, r, t, K, D)
        errs.append(np.linalg.norm(proj.reshape(-1, 2) - pp.reshape(-1, 2), axis=1))
    return K, D, errs


def calibrate(images, pattern, square_mm, module: Module, lens: Lens, progress=None, serial='', source='') -> CalibResult:
    """Unit calibration from checkerboard images. images: [(name, uint8 image)], all the same size.
    pattern = (columns, rows) of inner corners. lens: the nominal lens (initial guess and rectified search)."""
    res = CalibResult(None, [], tuple(pattern), float(square_mm))
    if not images:
        res.message = 'No images.'
        return res
    h, w = images[0][1].shape[:2]
    res.width, res.height = w, h
    sensor = module.width_px / module.height_px
    if abs(w / h - sensor) > 0.005 * sensor:
        res.message = (f'The images are {w} × {h}, not the {module.width_px} × {module.height_px} sensor\'s aspect: a crop '
                       'or another sensor mode moves the principal point, so the result would not fit this camera. '
                       'Capture at the full field (any resolution with the sensor\'s aspect).')
        return res
    intr = resolve(module, lens)
    obj = board_points(pattern, square_mm)
    found = []
    for i, (name, img) in enumerate(images):
        v = CalibView(name, (img.shape[1], img.shape[0]))
        res.views.append(v)
        if img.shape[:2] != (h, w):
            v.note = f'{img.shape[1]} × {img.shape[0]}, not {w} × {h}'
        else:
            try:
                v.corners = find_board(img, pattern, intr)
            except Exception as e:      # unreadable image
                v.note = str(e)
            if v.corners is None and not v.note:
                v.note = 'board not found'
            elif v.corners is not None:
                found.append(v)
        if progress:
            progress((i + 1) / len(images) * 0.8, f'{name}: ' + ('board found' if v.corners is not None else v.note))
    if len(found) < 4:
        res.message = f'The board was found in {len(found)} image(s); at least 4 are needed, 10 or more spread over the image is better.'
        return res
    K0, D0 = _initial(intr, w, h)
    use = list(found)
    K = D = None
    for attempt in range(5):
        try:
            K, D, errs = _fisheye([obj] * len(use), [v.corners for v in use], (w, h), K0, D0)
        except Exception as e:
            # an ill-conditioned view (board nearly edge-on or tiny) stops cv::fisheye; drop the worst by spread
            spread = [np.ptp(v.corners, axis=0).prod() for v in use]
            if len(use) <= 4:
                res.message = f'Calibration did not converge: {str(e).splitlines()[-1][:120]}'
                return res
            use.pop(int(np.argmin(spread)))
            continue
        per = [float(np.sqrt(np.mean(e ** 2))) for e in errs]
        for v, r in zip(use, per):
            v.rms = r
        med = float(np.median(per))
        bad = [i for i, r in enumerate(per) if r > max(3 * med, 1.0)]
        if not bad or len(use) - len(bad) < 4 or attempt == 4:
            break                       # the result always comes from a solve over exactly the views kept
        for i in sorted(bad, reverse=True):
            use[i].note = f'dropped: {per[i]:.2f} px'
            use.pop(i)
        K0, D0 = K, D
    if K is None:
        res.message = 'Calibration did not converge.'
        return res
    for v in use:
        v.used = True
    allerr = np.concatenate([np.linalg.norm(_reproject(v, obj, K, D) - v.corners, axis=1) for v in use])
    res.rms = float(np.sqrt(np.mean(allerr ** 2)))
    res.record = make_record(w, h, K[0, 0], K[1, 1], K[0, 2], K[1, 2], D.ravel(), res.rms, serial, source, len(use))
    pts = np.concatenate([v.corners for v in use])
    gy, gx = 8, 10
    cov, _, _ = np.histogram2d(pts[:, 1], pts[:, 0], bins=(gy, gx), range=((0, h), (0, w)))
    res.coverage = cov
    # cells inside the image circle (the measured model)
    ui = resolve(module, unit_lens(lens, module, res.record))
    cu = (np.arange(gx) + 0.5) / gx * ui.width
    cv_ = (np.arange(gy) + 0.5) / gy * ui.height
    CU, CV = np.meshgrid(cu, cv_)
    _, th = ui.unproject(CU.ravel(), CV.ravel())
    inside = (th <= ui.max_theta * 0.95).reshape(gy, gx)
    res.coverage_frac = float(((cov > 0) & inside).sum() / max(1, inside.sum()))
    res.ok = True
    res.advice = []
    if res.rms > 0.6:
        res.advice.append(f'RMS {res.rms:.2f} px is high: check focus, motion blur and that the board is flat.')
    if res.coverage_frac < 0.55:
        res.advice.append(f'Corners cover {100 * res.coverage_frac:.0f} % of the image: add views with the board near '
                          'the edges and corners, where the distortion is strongest.')
    if len(use) < 8:
        res.advice.append(f'{len(use)} views used: 10 to 20 give a steadier result.')
    res.message = (f'{len(use)} of {len(images)} views · RMS {res.rms:.3f} px · corners cover '
                   f'{100 * res.coverage_frac:.0f} % of the image')
    if progress:
        progress(1.0, res.message)
    return res


def _reproject(v, obj, K, D):
    import cv2
    ok, rvec, tvec = _pose_fisheye(v.corners, obj, K, D)
    proj, _ = cv2.fisheye.projectPoints(obj.reshape(-1, 1, 3), rvec, tvec, K, D)
    return proj.reshape(-1, 2)


def _pose_fisheye(corners, obj, K, D):
    import cv2
    und = cv2.fisheye.undistortPoints(corners.reshape(-1, 1, 2).astype(np.float64), K, D)
    ok, rvec, tvec = cv2.solvePnP(obj.astype(np.float64), und.reshape(-1, 2), np.eye(3), None,
                                  flags=cv2.SOLVEPNP_ITERATIVE)
    if ok:
        rvec, tvec = cv2.solvePnPRefineLM(obj.astype(np.float64), und.reshape(-1, 2), np.eye(3), None, rvec, tvec)
    return ok, rvec, tvec


# ---------------------------------------------------------------- synthetic calibration sets
def simulated_unit(module: Module, lens: Lens, seed=0, f_pct=1.5, pp_px=15.0, k_pct=1.0, aspect_pct=0.1, serial=None) -> dict:
    """A plausible unit-to-unit variation of the library lens, as a record at sensor resolution."""
    from .exporters import kb_coefficients
    intr = resolve(module, lens)
    k, _ = kb_coefficients(intr)
    rng = np.random.default_rng(seed)
    fx = intr.f_px * (1 + rng.uniform(-f_pct, f_pct) / 100)
    fy = fx * (1 + rng.uniform(-aspect_pct, aspect_pct) / 100)
    cx = intr.cx - 0.5 + rng.uniform(-pp_px, pp_px)
    cy = intr.cy - 0.5 + rng.uniform(-pp_px, pp_px)
    kk = [float(x) * (1 + rng.uniform(-k_pct, k_pct) / 100) for x in k]
    return make_record(intr.width, intr.height, fx, fy, cx, cy, kk, serial=serial or f'SIM{seed:04d}', source='simulated')


def _random_poses(intr: Intrinsics, pattern, square_mm, n, rng, w, h):
    """Board poses spread over the whole image: a target pixel is drawn (more often near the edges, where the
    distortion is strongest), the board is centred on its ray at a distance that fits it, then tilted and spun."""
    from .mathutil import rot_x, rot_y, rot_z
    cols, rows = pattern
    sq = square_mm / 1000
    bw, bh = (cols + 1) * sq, (rows + 1) * sq
    obj = board_points(pattern, square_mm) - np.array([(cols - 1) * sq / 2, (rows - 1) * sq / 2, 0])
    corners = np.array([[-bw / 2, -bh / 2, 0], [bw / 2, -bh / 2, 0], [bw / 2, bh / 2, 0], [-bw / 2, bh / 2, 0]])
    lim = min(intr.max_theta - math.radians(3), math.radians(84))
    m = 0.015 * intr.width
    out = []
    tries = 0
    while len(out) < n and tries < n * 600:
        tries += 1
        if not out:
            u, v = intr.width / 2, intr.height / 2
        elif rng.uniform() < 0.6:                       # outer band
            e = rng.uniform(0.04, 0.2)
            side = rng.integers(4)
            t = rng.uniform(0, 1)
            u, v = [(e * intr.width, t * intr.height), ((1 - e) * intr.width, t * intr.height),
                    (t * intr.width, e * intr.height), (t * intr.width, (1 - e) * intr.height)][side]
        else:
            u, v = rng.uniform(0.1, 0.9) * intr.width, rng.uniform(0.1, 0.9) * intr.height
        d, th = intr.unproject(np.array([u]), np.array([v]))
        d = d[0]
        if th[0] > lim:
            continue
        half = math.radians(rng.uniform(9, 24))         # angular half-size of the board
        dist = (bw / 2) / math.tan(half)
        C = d * dist
        zb = -d
        xb = np.cross([0, 1.0, 0], zb)
        if np.linalg.norm(xb) < 1e-6:
            xb = np.array([1.0, 0, 0])
        xb /= np.linalg.norm(xb)
        yb = np.cross(zb, xb)
        Rb = np.stack([xb, yb, zb], 1) @ rot_z(rng.uniform(-math.pi, math.pi)) @ rot_x(rng.uniform(-0.6, 0.6)) @ \
            rot_y(rng.uniform(-0.6, 0.6))
        allp = np.vstack([obj @ Rb.T + C, corners @ Rb.T + C])
        dirs = allp / np.linalg.norm(allp, axis=1, keepdims=True)
        pu, pv, pth, ok = intr.project_dirs(dirs)
        if not ok.all() or (pth > lim).any():
            continue
        if (pu < m).any() or (pu > intr.width - m).any() or (pv < m).any() or (pv > intr.height - m).any():
            continue
        if abs(float(Rb[:, 2] @ (-C / np.linalg.norm(C)))) < math.cos(math.radians(62)):
            continue
        out.append((Rb, C))
    return out


def render_board(intr: Intrinsics, Rb, C, pattern, square_mm, w, h, seed=0, noise=2.0) -> np.ndarray:
    """Grey image of a chessboard (board-to-optical rotation Rb, centre C in metres) seen through intr."""
    from .synth import pixel_rays, _checker
    from .project import Target
    cols, rows = pattern
    d, th = pixel_rays(intr, w, h)
    n = Rb[:, 2]
    with np.errstate(divide='ignore', invalid='ignore'):
        t = (C @ n) / (d @ n)
    hit = (th <= intr.max_theta) & (t > 0) & np.isfinite(t)
    P = d * np.where(hit, t, 0)[:, None] - C
    lx, ly = P @ Rb[:, 0], P @ Rb[:, 1]
    inc = np.clip(np.abs(d @ n), 0.1, 1.0)
    pix = intr.pixel_mm / intr.f_mm * (intr.width / w)
    fw = np.where(hit, t * pix / inc, 1.0)
    tgt = Target(name='cal', x=0.0, y=0.0, yaw=0.0, rows=rows + 1, cols=cols + 1, square_mm=square_mm)
    img = np.full(len(d), 40.0)
    inside = th <= intr.max_theta
    # a softly textured room
    az = np.arctan2(d[:, 1], d[:, 0])
    img[inside] = 120 + 18 * np.sin(az[inside] * 3) * np.cos(th[inside] * 5)
    b = np.full(len(d), np.nan)
    b[hit] = _checker(lx[hit], ly[hit], [tgt], fw[hit])
    sel = ~np.isnan(b)
    img[sel] = 25 + 210 * b[sel]
    if noise:
        img += np.random.default_rng(seed).normal(0, noise, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8).reshape(h, w)


def board_views(module: Module, lens: Lens, rec: dict | None, pattern=(9, 6), square_mm=60.0, n=16, seed=0,
                scale=0.5, noise=2.0, progress=None) -> list:
    """Synthetic calibration set: [(name, image)] of a hand-held board spread over the image of a camera whose true
    intrinsics are rec (the library lens when None)."""
    truth = unit_lens(lens, module, rec) if rec else lens
    intr = resolve(module, truth)
    w, h = int(round(intr.width * scale)), int(round(intr.height * scale))
    rng = np.random.default_rng(seed)
    poses = _random_poses(intr, pattern, square_mm, n, rng, w, h)
    out = []
    for i, (Rb, C) in enumerate(poses):
        out.append((f'view_{i + 1:02d}', render_board(intr, Rb, C, pattern, square_mm, w, h, seed=seed * 101 + i, noise=noise)))
        if progress:
            progress((i + 1) / len(poses), f'Rendered view {i + 1} of {len(poses)}')
    return out


# ---------------------------------------------------------------- files
def _kb_model(model: str) -> bool:
    return (model or '').lower().replace('-', '_') in ('fisheye', 'kannala_brandt', 'equidistant', 'kb4', 'kb', '')


def _rec_from(K, D, w, h, model='', hint='', serial='', path='') -> dict:
    K = np.asarray(K, float).reshape(3, 3)
    D = [] if D is None else np.asarray(D, float).ravel().tolist()
    if not _kb_model(model):
        raise ValueError(f'distortion model “{model}” is not fisheye (Kannala-Brandt / equidistant)')
    if not model and len(D) != 4:
        # cv::calibrateCamera files name no model and carry 5+ pinhole coefficients (k1 k2 p1 p2 k3 ...)
        raise ValueError(f'no distortion model named and {len(D)} coefficients: looks like a pinhole calibration '
                         '(a fisheye one has exactly 4)')
    if len(D) < 4:
        raise ValueError(f'{len(D)} distortion coefficients; a fisheye model has 4')
    if not (w and h):
        raise ValueError('image size missing')
    r = make_record(w, h, K[0, 0], K[1, 1], K[0, 2], K[1, 2], D[:4], serial=serial,
                    source=os.path.basename(path) if path else 'import')
    r['hint'] = str(hint or '')
    r['file'] = path
    return r


def _fs_array(n):
    if n.isSeq():
        # a flat list, or a list of rows ([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        parts = [_fs_array(n.at(i)).ravel() if n.at(i).isSeq() else np.array([n.at(i).real()]) for i in range(n.size())]
        return np.concatenate(parts) if parts else np.zeros(0)
    m = n.mat()
    if m is None:
        raise ValueError(f'{n.name()} is not a matrix')
    return np.asarray(m, float)


def _opencv_storage(path) -> list:
    import cv2
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise ValueError('not an OpenCV FileStorage file')
    try:
        def node(*names):
            for n in names:
                x = fs.getNode(n)
                if not x.empty():
                    return x
            return None
        Kn = node('camera_matrix', 'K', 'cameraMatrix', 'intrinsic_matrix')
        Dn = node('distortion_coefficients', 'D', 'dist_coeffs', 'distCoeffs')
        if Kn is None:
            raise ValueError('camera_matrix missing')
        K = _fs_array(Kn)
        D = _fs_array(Dn) if Dn is not None else None
        wn, hn = node('image_width', 'width'), node('image_height', 'height')
        w, h = (int(wn.real()) if wn is not None else 0), (int(hn.real()) if hn is not None else 0)
        sizen = node('image_size', 'resolution')
        if (not w or not h) and sizen is not None:
            sz = _fs_array(sizen).ravel()
            w, h = int(sz[0]), int(sz[1])
        mn = node('distortion_model', 'model')
        model = mn.string() if mn is not None else ''
        fm = node('fisheye_model')                   # the OpenCV calibration sample writes this flag instead
        if not model and fm is not None:
            model = 'fisheye' if int(fm.real()) else 'pinhole'
        name = node('camera_name', 'name', 'tag', 'camera')
        sn = node('serial', 'serial_number')
        return [_rec_from(K, D, w, h, model, name.string() if name is not None else '',
                          sn.string() if sn is not None else '', path)]
    finally:
        fs.release()


def _mat(x):
    if isinstance(x, dict) and 'data' in x:
        return x['data']
    return x


def _from_mapping(d: dict, path: str, hint='') -> list:
    """Records from a parsed YAML/JSON mapping: Vantage, ROS camera_info, Kalibr camchain or generic K/D."""
    if not isinstance(d, dict):
        return []
    if d.get('schema') == SCHEMA:
        out = []
        for c in d.get('cameras', []):
            r = dict(make_record(**{k: c['calib'][k] for k in ('width', 'height', 'fx', 'fy', 'cx', 'cy', 'k')}),
                     **{k: c['calib'][k] for k in ('rms_px', 'serial', 'source', 'views', 'date') if k in c['calib']})
            r['hint'] = c.get('tag', '')
            r['file'] = path
            out.append(r)
        return out
    cams = [(k, v) for k, v in d.items() if re.fullmatch(r'cam\d+', str(k)) and isinstance(v, dict)]
    if cams:                                                    # Kalibr camchain
        out = []
        for k, c in cams:
            if c.get('camera_model', 'pinhole') != 'pinhole':
                raise ValueError(f'{k}: camera model “{c.get("camera_model")}” is not supported (pinhole + equidistant)')
            fu, fv, pu, pv = c['intrinsics'][:4]
            w, h = c['resolution'][:2]
            K = [[fu, 0, pu], [0, fv, pv], [0, 0, 1]]
            out.append(_rec_from(K, c.get('distortion_coeffs'), w, h, c.get('distortion_model', ''),
                                 c.get('rostopic', '') or k, '', path))
        return out
    if 'camera_matrix' in d or 'K' in d or 'k' in d and isinstance(d.get('k'), list) and len(d['k']) == 9:
        K = _mat(d.get('camera_matrix', d.get('K', d.get('k'))))
        D = _mat(d.get('distortion_coefficients', d.get('D', d.get('d', d.get('dist_coeffs')))))
        w = d.get('image_width', d.get('width'))
        h = d.get('image_height', d.get('height'))
        if (not w or not h) and isinstance(d.get('resolution'), (list, tuple)):
            w, h = d['resolution'][:2]
        return [_rec_from(K, D, w, h, d.get('distortion_model', ''), d.get('camera_name', d.get('tag', hint)),
                          d.get('serial', ''), path)]
    if all(k in d for k in ('fx', 'fy', 'cx', 'cy')) and ('width' in d or 'image_width' in d):
        D = d.get('k') if isinstance(d.get('k'), list) else [d.get(f'k{i}', 0.0) for i in range(1, 5)]
        K = [[d['fx'], 0, d['cx']], [0, d['fy'], d['cy']], [0, 0, 1]]
        return [_rec_from(K, D, d.get('width', d.get('image_width')), d.get('height', d.get('image_height')),
                          d.get('model', d.get('distortion_model', '')), d.get('tag', d.get('camera', hint)),
                          d.get('serial', ''), path)]
    out = []
    for key in ('cameras', 'camera'):                           # Vantage exports and similar lists
        v = d.get(key)
        if isinstance(v, list):
            for c in v:
                if isinstance(c, dict):
                    sub = c.get('intrinsics') if isinstance(c.get('intrinsics'), dict) else c
                    sub = dict(sub)
                    sub.setdefault('tag', c.get('tag', c.get('name', '')))
                    try:
                        out += _from_mapping(sub, path, sub['tag'])
                    except (KeyError, ValueError):
                        pass
    return out


def read_file(path: str) -> list:
    """Unit calibrations in a file: [record + 'hint' (camera tag, name or topic) + 'file']."""
    ext = os.path.splitext(path)[1].lower()
    with open(path, 'rb') as fh:
        head = fh.read(64)
    if ext == '.xml' or head.lstrip().startswith(b'%YAML:1.0') or head.lstrip().startswith(b'%YAML 1.0'):
        recs = _opencv_storage(path)
    elif ext == '.json':
        with open(path, encoding='utf-8') as fh:
            recs = _from_mapping(json.load(fh), path)
    else:
        import yaml
        with open(path, encoding='utf-8') as fh:
            recs = _from_mapping(yaml.safe_load(fh), path)
    if not recs:
        raise ValueError('no fisheye calibration found')
    stem = os.path.splitext(os.path.basename(path))[0]
    for r in recs:
        if not r.get('hint'):
            r['hint'] = stem
        r['stem'] = stem
    return recs


def _tokens(s: str) -> list:
    return [t for t in re.split(r'[^a-z0-9]+', (s or '').lower()) if t]


def match(records: list, cams: list) -> tuple[dict, list]:
    """Assign records to cameras by tag, then by serial. Returns ({uid: record}, unmatched records)."""
    def contains(src, tag):
        a, b = _tokens(src), _tokens(tag)
        return bool(b) and any(a[k:k + len(b)] == b for k in range(len(a) - len(b) + 1))

    scores = []
    for i, r in enumerate(records):
        for c in cams:
            tag = c.tag.lower()
            s = 0
            for src, w in ((r.get('hint', ''), 3), (r.get('stem', ''), 2)):
                if src.lower() == tag:
                    s = max(s, 20 * w)
                elif contains(src, tag):
                    s = max(s, 10 * w + len(_tokens(tag)))      # 'FL_1' beats 'FL' in 'calib_FL_1'
            if r.get('serial') and c.calib and c.calib.get('serial') == r['serial']:
                s = max(s, 70)
            if s:
                scores.append((s, i, c.uid))
    scores.sort(key=lambda t: -t[0])
    used_r, used_c, out = set(), set(), {}
    for s, i, uid in scores:
        if i in used_r or uid in used_c:
            continue
        used_r.add(i)
        used_c.add(uid)
        out[uid] = records[i]
    return out, [r for i, r in enumerate(records) if i not in used_r]


def to_json(project, cams) -> str:
    """The unit calibrations of these cameras as a Vantage JSON file (read back by read_file)."""
    rows = [{'tag': c.tag, 'module': c.module_id, 'lens': c.lens_id, 'calib': clean_record(c.calib)} for c in cams if c.calib]
    return json.dumps({'schema': SCHEMA, 'project': project.name,
                       'convention': 'OpenCV pixel coordinates (pixel centre at integer i) at width x height; '
                                     'cv::fisheye (Kannala-Brandt) k1..k4',
                       'cameras': rows}, indent=1)


def to_opencv_yaml(rec: dict, tag: str = '') -> str:
    """One unit calibration as an OpenCV FileStorage YAML (cv::fisheye)."""
    def mat(rows, cols, vals):
        return (f'!!opencv-matrix\n   rows: {rows}\n   cols: {cols}\n   dt: d\n   data: [ '
                + ', '.join(repr(float(v)) for v in vals) + ' ]')
    def q(t):
        return str(t).replace('\\', '\\\\').replace('"', '\\"')
    K = [rec['fx'], 0, rec['cx'], 0, rec['fy'], rec['cy'], 0, 0, 1]
    return ('%YAML:1.0\n---\n'
            f'camera_name: "{q(tag)}"\nserial: "{q(rec.get("serial", ""))}"\nimage_width: {int(rec["width"])}\n'
            f'image_height: {int(rec["height"])}\ndistortion_model: fisheye\n'
            f'camera_matrix: {mat(3, 3, K)}\n'
            f'distortion_coefficients: {mat(1, 4, rec["k"])}\n'
            f'rms_px: {float(rec.get("rms_px", 0))!r}\n'
            f'calibration_date: "{q(rec.get("date", ""))}"\n')
