"""Checks of the camera models: lens projection, ground coverage and pixel density, occlusion by the body, the
perception range and the critical-target pixels and envelopes."""
from __future__ import annotations

import math

import numpy as np
from scipy.spatial.transform import Rotation

from .suite import Outcome, check, figure, rel_err, series

PROJ = ('rectilinear', 'equidistant', 'equisolid', 'stereographic', 'orthographic')
# the half angle each base projection is checked to (rectilinear and orthographic end at 90°)
PROJ_REACH = {'rectilinear': 85.0, 'equidistant': 175.0, 'equisolid': 175.0, 'stereographic': 175.0,
              'orthographic': 89.0}


# ============================================================================ helpers
def _library():
    from ..core.lens import builtin_lenses, builtin_modules
    mods = {m.id: m for m in builtin_modules()}
    return [(ln, mods.get(ln.module_id) or next(iter(mods.values()))) for ln in builtin_lenses()]


def _cone_dirs(rng, n, max_deg, min_deg=0.0):
    """n unit directions about +z, uniform in solid angle between min_deg and max_deg off the axis."""
    c0, c1 = math.cos(math.radians(min_deg)), math.cos(math.radians(max_deg))
    ct = rng.uniform(c1, c0, n)
    st = np.sqrt(1 - ct * ct)
    ph = rng.uniform(0, 2 * math.pi, n)
    return np.c_[st * np.cos(ph), st * np.sin(ph), ct]


def _rot(yaw, pitch, roll):
    """Body -> vehicle rotation of the documented convention (intrinsic Z-Y-X, pitch positive down, right-handed about
    +y), built with scipy rather than the app's own rot_zyx."""
    return Rotation.from_euler('ZYX', [yaw, pitch, roll], degrees=True).as_matrix()


def _optical_axes(yaw, pitch, roll):
    """Columns: the optical frame's x (image right), y (image down) and z (the optical axis) in the vehicle frame,
    from the documented convention: the camera body's +x is the optical axis, image right is body −y, image down
    body −z."""
    R = _rot(yaw, pitch, roll)
    return np.stack([-R[:, 1], -R[:, 2], R[:, 0]], axis=1)


def box_mesh(lo, hi):
    """A closed axis-aligned box, 12 triangles wound outward."""
    from ..core.mesh import Mesh
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    V = np.array([[x, y, z] for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)], float)
    F = [[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6], [0, 2, 6],
         [0, 6, 4], [1, 5, 7], [1, 7, 3]]
    return Mesh(V, np.array(F))


class RayCaster:
    """Exact ray casting against triangles (M, 3, 3), written for the checks: Möller–Trumbore on the ray–triangle pairs
    whose triangle's bounding sphere the ray passes through."""

    def __init__(self, tri):
        self.tri = np.asarray(tri, float)
        self.v0 = self.tri[:, 0]
        self.e1 = self.tri[:, 1] - self.v0
        self.e2 = self.tri[:, 2] - self.v0
        self.c = self.tri.mean(axis=1)
        self.r = np.linalg.norm(self.tri - self.c[:, None], axis=2).max(axis=1) + 1e-9

    def first_hit(self, O, D, t_max=None):
        """For rays O + t·D (D unit, K × 3; O one point or K points): the nearest t > 1e-9 of a hit, inf where none
        (and, with t_max (K,), none before t_max)."""
        D = np.asarray(D, float).reshape(-1, 3)
        O = np.broadcast_to(np.asarray(O, float).reshape(-1, 3), D.shape)
        out = np.full(len(D), np.inf)
        tm = None if t_max is None else np.broadcast_to(np.asarray(t_max, float), (len(D),))
        step = max(1, 4_000_000 // max(len(self.tri), 1))
        for a in range(0, len(D), step):
            o, d = O[a:a + step], D[a:a + step]
            w = self.c[None] - o[:, None]                                   # (k, M, 3)
            s = np.einsum('kmi,ki->km', w, d)
            perp2 = np.einsum('kmi,kmi->km', w, w) - s * s
            k, m = np.nonzero((perp2 <= self.r[None] ** 2) & (s > -self.r[None]))
            if not len(k):
                continue
            dk, ok_ = d[k], o[k]
            p = np.cross(dk, self.e2[m])
            det = np.einsum('ij,ij->i', p, self.e1[m])
            good = np.abs(det) > 1e-14
            inv = np.where(good, 1.0 / np.where(good, det, 1.0), 0.0)
            sv = ok_ - self.v0[m]
            u = np.einsum('ij,ij->i', sv, p) * inv
            q = np.cross(sv, self.e1[m])
            v = np.einsum('ij,ij->i', dk, q) * inv
            t = np.einsum('ij,ij->i', q, self.e2[m]) * inv
            hit = good & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 1e-9)
            if tm is not None:
                hit &= t < tm[a:a + step][k]
            if hit.any():
                np.minimum.at(out[a:a + step], k[hit], t[hit])
        return out


# ============================================================================ lens
@check('lens.inverse', 'lens', 'Image height inverts back to the angle',
       'largest |θ − θ(r(θ))| over each projection\'s field, every base projection and every built-in lens\'s '
       'Kannala–Brandt curve',
       'the identity: θ(r) is the inverse of r(θ) (closed form, or Newton\'s method for Kannala–Brandt)',
       1e-6, 'degrees', refs=('kannala2006', 'miyamoto1964'))
def _lens_inverse(ctx):
    from ..core.lens import r_of_theta, theta_of_r, resolve
    worst, rows = 0.0, []
    for proj in PROJ:
        th = np.radians(np.linspace(0.01, PROJ_REACH[proj], 2000))
        e = float(np.degrees(np.max(np.abs(theta_of_r(r_of_theta(th, 3.0, proj), 3.0, proj) - th))))
        rows.append((proj, e))
        worst = max(worst, e)
    for ln, m in _library():
        it = resolve(m, ln)
        th = np.linspace(1e-4, it.max_theta, 2000)
        e = float(np.degrees(np.max(np.abs(theta_of_r(r_of_theta(th, it.f_mm, 'kannala_brandt', it.kb), it.f_mm,
                                                       'kannala_brandt', it.kb) - th))))
        rows.append((ln.id, e))
        worst = max(worst, e)
    return Outcome(worst, {'per_model_deg': {k: v for k, v in rows}, 'samples': 2000})


@check('lens.derivative', 'lens', 'Radial sampling dr/dθ',
       'largest relative difference of dr/dθ from a central finite difference of r(θ), every base projection and '
       'built-in lens',
       'a central difference of r(θ) with step 10⁻⁶ rad', 1e-6, 'relative', refs=('kannala2006',))
def _lens_derivative(ctx):
    from ..core.lens import r_of_theta, dr_dtheta, resolve
    h = 1e-6
    worst, rows = 0.0, {}
    cases = [(p, 3.0, p, (0, 0, 0, 0), math.radians(PROJ_REACH[p] - 1)) for p in PROJ]
    for ln, m in _library():
        it = resolve(m, ln)
        cases.append((ln.id, it.f_mm, 'kannala_brandt', it.kb, it.max_theta))
    for name, f, proj, kb, top in cases:
        th = np.linspace(0.01, top, 500)
        num = (r_of_theta(th + h, f, proj, kb) - r_of_theta(th - h, f, proj, kb)) / (2 * h)
        e = float(np.max(rel_err(dr_dtheta(th, f, proj, kb), num)))
        rows[name] = e
        worst = max(worst, e)
    return Outcome(worst, {'per_model': rows})


@check('lens.opencv_pinhole', 'lens', 'Rectilinear lens against OpenCV\'s pinhole projection',
       'largest pixel distance between Intrinsics.project_dirs and cv2.projectPoints for directions over the image, '
       'with a pixel aspect of 1.004 and an offset principal point',
       'OpenCV cv2.projectPoints (no distortion); OpenCV pixel centres are at integers, Vantage\'s at i + ½',
       1e-6, 'px', refs=('opencv', 'hartley2004'))
def _lens_opencv_pinhole(ctx):
    import cv2
    from ..core.lens import Lens, Module, resolve
    m = Module(id='m', width_px=1920, height_px=1536, pixel_um=3.0)
    ln = Lens(id='p', focal_mm=4.0, projection='rectilinear', pp_offset_px=(12.3, -7.9), aspect=1.004)
    it = resolve(m, ln)
    d = _cone_dirs(ctx.rng('lens.opencv_pinhole'), 4000, 40.0)
    u, v, _, ok = it.project_dirs(d)
    d, u, v = d[ok], u[ok], v[ok]
    fx = it.f_px
    K = np.array([[fx, 0, it.cx - 0.5], [0, fx * it.aspect, it.cy - 0.5], [0, 0, 1]], float)
    uv, _ = cv2.projectPoints(d.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K, np.zeros(5))
    err = np.hypot(uv[:, 0, 0] - (u - 0.5), uv[:, 0, 1] - (v - 0.5))
    return Outcome(float(err.max()), {'points': int(len(d)), 'f_px': fx, 'aspect': it.aspect,
                                      'principal_point_px': [it.cx, it.cy]})


@check('lens.opencv_fisheye', 'lens', 'Kannala–Brandt lenses against OpenCV\'s fisheye model',
       'largest pixel distance between Intrinsics.project_dirs and cv2.fisheye.projectPoints (D = k1…k4) for every '
       'built-in lens, directions up to 88° off the axis',
       'OpenCV cv2.fisheye.projectPoints: θ_d = θ(1 + k1θ² + k2θ⁴ + k3θ⁶ + k4θ⁸), the same model',
       1e-6, 'px', refs=('opencv', 'kannala2006'))
def _lens_opencv_fisheye(ctx):
    import cv2
    from ..core.lens import resolve
    rng = ctx.rng('lens.opencv_fisheye')
    worst, rows = 0.0, {}
    for ln, m in _library():
        it = resolve(m, ln)
        d = _cone_dirs(rng, 1500, min(88.0, math.degrees(it.max_theta)))
        u, v, _, ok = it.project_dirs(d)
        d, u, v = d[ok], u[ok], v[ok]
        K = np.array([[it.f_px, 0, it.cx - 0.5], [0, it.f_px * it.aspect, it.cy - 0.5], [0, 0, 1]], float)
        uv, _ = cv2.fisheye.projectPoints(d.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K,
                                          np.asarray(it.kb, float).reshape(4, 1))
        e = float(np.hypot(uv[:, 0, 0] - (u - 0.5), uv[:, 0, 1] - (v - 0.5)).max())
        rows[ln.id] = e
        worst = max(worst, e)
    return Outcome(worst, {'per_lens_px': rows})


@check('lens.opencv_undistort', 'lens', 'Pixel to ray against OpenCV\'s fisheye undistortion',
       'largest angle between Intrinsics.unproject\'s ray and the ray cv2.fisheye.undistortPoints gives for the same '
       'pixel, every built-in lens, pixels up to 75° off the axis',
       'OpenCV cv2.fisheye.undistortPoints (iterated to 10⁻¹⁴)', 1e-4, 'degrees', refs=('opencv', 'kannala2006'))
def _lens_opencv_undistort(ctx):
    import cv2
    from ..core.lens import resolve
    rng = ctx.rng('lens.opencv_undistort')
    worst, rows = 0.0, {}
    crit = (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 100, 1e-14)
    for ln, m in _library():
        it = resolve(m, ln)
        d = _cone_dirs(rng, 800, min(75.0, math.degrees(it.max_theta) - 0.5))
        u, v, _, ok = it.project_dirs(d)
        u, v = u[ok], v[ok]
        ray, _ = it.unproject(u, v)
        K = np.array([[it.f_px, 0, it.cx - 0.5], [0, it.f_px * it.aspect, it.cy - 0.5], [0, 0, 1]], float)
        pts = np.c_[u - 0.5, v - 0.5].reshape(-1, 1, 2)
        try:
            n = cv2.fisheye.undistortPoints(pts, K, np.asarray(it.kb, float).reshape(4, 1), criteria=crit)
        except TypeError:                                   # an OpenCV without the criteria argument
            n = cv2.fisheye.undistortPoints(pts, K, np.asarray(it.kb, float).reshape(4, 1))
        cv = np.c_[n[:, 0, 0], n[:, 0, 1], np.ones(len(u))]
        cv /= np.linalg.norm(cv, axis=1, keepdims=True)
        ang = np.degrees(np.arccos(np.clip(np.sum(cv * ray, axis=1), -1, 1)))
        rows[ln.id] = float(ang.max())
        worst = max(worst, rows[ln.id])
    return Outcome(worst, {'per_lens_deg': rows})


@check('lens.builtin_fov', 'lens', 'Built-in lenses reproduce e-con\'s stated field of view',
       'largest difference between a built-in lens model\'s horizontal, vertical and diagonal field (limited by its '
       'image circle) and the field e-con states for it',
       'the FOV e-con Systems publishes for each lens (H, V, D)', 0.5, 'degrees', refs=('econ',))
def _lens_builtin_fov(ctx):
    from ..core.lens import model_fov
    rows, worst, names, vals = {}, 0.0, [], []
    for ln, m in _library():
        fov = model_fov(m, ln, circle=True)
        diffs = {a: round(f - s, 4) for f, s, a in zip(fov, ln.sheet_fov, 'HVD') if s}
        e = max((abs(x) for x in diffs.values()), default=0.0)
        rows[ln.id] = {'model_deg': [round(x, 3) for x in fov], 'stated_deg': list(ln.sheet_fov), 'diff_deg': diffs}
        names.append(ln.name if ln.name not in names else ln.id)
        vals.append(e)
        worst = max(worst, e)
    fig = figure('Largest FOV difference per built-in lens', 'lens', 'model − stated, |°|',
                 [series('largest |difference|', np.arange(len(vals)), vals, kind='bar')], kind='bars',
                 hline=0.5, hline_label='tolerance 0.5°', categories=names)
    return Outcome(worst, {'per_lens': rows}, fig)


def _patch_density(it, theta, rng_m=1.0, side_m=0.001):
    """px/cm on a small square facing the camera at off-axis angle theta: its corners projected through the lens and
    the pixel area taken by the shoelace formula."""
    d = np.array([math.sin(theta) * math.cos(0.7), math.sin(theta) * math.sin(0.7), math.cos(theta)])
    e1 = np.cross(d, [0.0, 0.0, 1.0] if abs(d[2]) < 0.9 else [1.0, 0.0, 0.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(d, e1)
    h = side_m / 2
    C = np.array([rng_m * d + a * h * e1 + b * h * e2 for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
    u, v, _, _ = it.project_dirs(C)
    area = 0.5 * abs(np.dot(u, np.roll(v, -1)) - np.dot(v, np.roll(u, -1)))
    return math.sqrt(area) / (side_m * 100.0)


@check('lens.density', 'lens', 'Pixel density on a target facing the camera',
       'largest relative difference between Intrinsics.density_px_per_cm and the square root of the pixel area a '
       '1 mm square facing the camera at 1 m covers (its corners projected through the lens), from the axis to the '
       'edge of each lens\'s field',
       'the projected area of a small square, by the shoelace formula', 1e-3, 'relative',
       refs=('kannala2006', 'johnson1958'))
def _lens_density(ctx):
    from ..core.lens import Lens, Module, resolve
    m = Module(id='m', width_px=1920, height_px=1536, pixel_um=3.0)
    cases = [(ln.id, resolve(mm, ln)) for ln, mm in _library()[:4]]
    cases.append(('rectilinear 4 mm, aspect 1.004', resolve(m, Lens(id='r', focal_mm=4.0, projection='rectilinear',
                                                                    aspect=1.004))))
    cases.append(('equidistant 1.5 mm', resolve(m, Lens(id='e', focal_mm=1.5, projection='equidistant',
                                                         max_half_deg=95.0))))
    worst, rows, figs = 0.0, {}, []
    for name, it in cases:
        top = min(math.degrees(it.max_theta) - 0.5, 88.0 if it.projection == 'rectilinear' else 120.0)
        ths = np.radians(np.linspace(0.5, top, 25))
        got = np.array([float(it.density_px_per_cm(t, 1.0)) for t in ths])
        ref = np.array([_patch_density(it, t) for t in ths])
        e = rel_err(got, ref)
        rows[name] = float(e.max())
        worst = max(worst, rows[name])
        if len(figs) < 4:
            figs.append(series(name, np.degrees(ths), got))
    fig = figure('Pixel density at 1 m against the angle off the axis', 'angle off the optical axis, °',
                 'px/cm at 1 m', figs)
    return Outcome(worst, {'per_lens': rows}, fig)


# ============================================================================ coverage on a flat plate
def _plate_project(lens, cam, polygon, body=((-9.0, -0.3, 0.0), (-8.4, 0.3, 0.6)), cell=0.03):
    """A project whose only vehicle is a small box far from the camera's footprint, one camera, a polygon ROI."""
    from ..core.project import Project, Camera, Variant
    p = Project()
    p.vehicle.mesh = box_mesh(*body)
    p.lenses[lens.id] = lens
    c = Camera(tag='P', name='Plate', module_id='sturdecam31', lens_id=lens.id, **cam)
    v = Variant(name='Plate', cameras=[c])
    p.variants = [v]
    p.active_uid = v.uid
    p.roi.mode = 'polygon'
    p.roi.polygon = [list(q) for q in polygon]
    p.analysis.cell_m = cell
    return p, c


def _plate_eval(p):
    from ..core.analysis import Engine
    res = Engine().analyse(p, terrain=False)
    g = res.grid
    X, Y = np.meshgrid(g.xs, g.ys)
    return res, g, X, Y, res.cams[0]


def _footprint(pose, f_mm, m):
    """The quadrilateral (4, 2) the image corners of a rectilinear camera cast on the plate z = 0."""
    A = _optical_axes(pose['yaw'], pose['pitch'], pose['roll'])
    C = np.array([pose['x'], pose['y'], pose['z']])
    fpx = f_mm / (m.pixel_um * 1e-3)
    quad = []
    for u, v in ((0, 0), (m.width_px, 0), (m.width_px, m.height_px), (0, m.height_px)):
        d = A @ np.array([(u - m.width_px / 2) / fpx, (v - m.height_px / 2) / fpx, 1.0])
        if d[2] >= 0:
            raise ValueError('an image corner looks above the horizon')
        quad.append((C + d * (-C[2] / d[2]))[:2])
    return np.array(quad)


def _in_convex(P, poly):
    """Points (N, 2) inside (or on) a convex polygon (M, 2) of either winding."""
    poly = np.asarray(poly, float)
    sign = np.sign(np.dot(poly[:, 0], np.roll(poly[:, 1], -1)) - np.dot(poly[:, 1], np.roll(poly[:, 0], -1)))
    inside = np.ones(len(P), bool)
    for k in range(len(poly)):
        a, b = poly[k], poly[(k + 1) % len(poly)]
        inside &= sign * ((b[0] - a[0]) * (P[:, 1] - a[1]) - (b[1] - a[1]) * (P[:, 0] - a[0])) >= 0
    return inside


def _seg_dist(P, poly):
    """Distance of points (N, 2) to a polygon's outline."""
    d = np.full(len(P), np.inf)
    for k in range(len(poly)):
        a, b = np.asarray(poly[k]), np.asarray(poly[(k + 1) % len(poly)])
        ab = b - a
        t = np.clip(((P - a) @ ab) / (ab @ ab), 0, 1)
        d = np.minimum(d, np.linalg.norm(P - (a + t[:, None] * ab), axis=1))
    return d


def _shoelace(q):
    return 0.5 * abs(np.dot(q[:, 0], np.roll(q[:, 1], -1)) - np.dot(q[:, 1], np.roll(q[:, 0], -1)))


def _rect_density(f_mm, pixel_mm, C, X, Y, axis):
    """f / (p cos^1.5 θ) / R, px/cm, for a rectilinear lens: radial sampling f/(p cos²θ), tangential f/(p cos θ)."""
    D = np.stack([X - C[0], Y - C[1], -np.full_like(X, C[2])], -1)
    R = np.linalg.norm(D, axis=-1)
    cos = np.maximum((D @ axis) / R, 1e-9)
    return (f_mm / pixel_mm) / cos ** 1.5 / (100 * R)


def _footprint_outcome(p, c, quad, pose, f, extra=None):
    res, g, X, Y, ce = _plate_eval(p)
    m = p.module_for(c)
    P = np.c_[X.ravel(), Y.ravel()]
    ref = _in_convex(P, quad).reshape(X.shape) & g.roi
    vis = ce.vis & g.roi
    diff = vis != ref
    far = diff.ravel() & (_seg_dist(P, quad) > 1.5 * g.cell)
    share = float(diff.sum()) / max(int(ref.sum()), 1)
    axis = _optical_axes(pose['yaw'], pose['pitch'], pose['roll'])[:, 2]
    dens_ref = _rect_density(f, m.pixel_um * 1e-3, np.array([pose['x'], pose['y'], pose['z']]), X, Y, axis)
    de = float(np.max(rel_err(ce.dens[vis], dens_ref[vis])))
    detail = {'cells_seen': int(vis.sum()), 'cells_in_the_footprint': int(ref.sum()),
              'cells_differing': int(diff.sum()), 'cells_differing_away_from_the_edges': int(far.sum()),
              'cell_share_differing': share, 'density_rel_error': de, 'cell_m': g.cell,
              'area_m2': float(vis.sum()) * g.cell ** 2, 'footprint_area_m2': _shoelace(quad),
              'corners_on_the_plate_m': quad.round(4).tolist(), 'pose': pose}
    detail.update(extra or {})
    return Outcome(max(share, de), detail, ok=int(far.sum()) == 0)


@check('coverage.nadir_pyramid', 'coverage', 'A rectilinear camera looking straight down at a plate',
       'share of footprint cells the analysis marks differently from the rectangle the image casts on the plate '
       '(4h²·tan α_h·tan α_v); the largest relative difference of a seen cell\'s pixel density from '
       'f / (p·cos^1.5 θ) / R; cells differing away from the rectangle\'s edges must be none',
       'the pyramid of a pinhole camera cut by a plane, written out in the check', 1e-3, 'relative',
       refs=('hartley2004',))
def _coverage_nadir(ctx):
    from ..core.lens import Lens, builtin_module
    h, f = 3.0, 4.0
    m = builtin_module('sturdecam31')
    pose = dict(x=4.0, y=0.0, z=h, yaw=17.0, pitch=90.0, roll=0.0)
    lens = Lens(id='plate_rect', name='Rectilinear 4 mm', focal_mm=f, projection='rectilinear', focus_near_m=0.0)
    quad = _footprint(pose, f, m)
    lo, hi = quad.min(axis=0) - 0.4, quad.max(axis=0) + 0.4
    p, c = _plate_project(lens, pose, [(lo[0], lo[1]), (hi[0], lo[1]), (hi[0], hi[1]), (lo[0], hi[1])], cell=0.02)
    a = (m.width_px * m.pixel_um * 1e-3 / 2) / f
    b = (m.height_px * m.pixel_um * 1e-3 / 2) / f
    closed = 4 * h * h * a * b
    out = _footprint_outcome(p, c, quad, pose, f, {'closed_form_area_m2': closed})
    out.ok = out.ok and abs(_shoelace(quad) - closed) < 1e-9 * closed      # the corners agree with the closed form
    return out


@check('coverage.oblique_footprint', 'coverage', 'An aimed and rolled camera\'s footprint on a plate',
       'share of footprint cells the analysis marks differently from the quadrilateral the image corners of a '
       'rectilinear camera at yaw 30°, pitch 50° down and roll 8° cast on the plate (rays built from the documented '
       'ISO 8855 Z-Y-X convention with scipy, not the app\'s rotations); the largest relative density difference as '
       'for the downward camera; cells differing away from the edges must be none',
       'the image corners\' rays intersected with the plane, written out in the check', 1e-3, 'relative',
       refs=('iso8855', 'hartley2004'))
def _coverage_oblique(ctx):
    from ..core.lens import Lens, builtin_module
    f = 4.0
    m = builtin_module('sturdecam31')
    pose = dict(x=2.0, y=0.5, z=2.5, yaw=30.0, pitch=50.0, roll=8.0)
    lens = Lens(id='plate_rect', name='Rectilinear 4 mm', focal_mm=f, projection='rectilinear', focus_near_m=0.0)
    quad = _footprint(pose, f, m)
    lo, hi = quad.min(axis=0) - 0.4, quad.max(axis=0) + 0.4
    p, c = _plate_project(lens, pose, [(lo[0], lo[1]), (hi[0], lo[1]), (hi[0], hi[1]), (lo[0], hi[1])],
                          body=((-14.0, -0.3, 0.0), (-13.4, 0.3, 0.6)), cell=0.02)
    return _footprint_outcome(p, c, quad, pose, f)


@check('coverage.fisheye_cone', 'coverage', 'A fisheye\'s image circle on a plate is a disc',
       'share of cells the analysis marks differently from the disc of radius h·tan θ_max a downward equidistant '
       'lens\'s image circle (inside the sensor) casts; the largest relative difference of a seen cell\'s density '
       'from (f/p)·√(θ / sin θ) / R',
       'the cone of the image circle cut by the plane, written out in the check', 1e-3, 'relative',
       refs=('miyamoto1964',))
def _coverage_cone(ctx):
    from ..core.lens import Lens
    h, f, top = 3.0, 1.5, 60.0
    lens = Lens(id='plate_fish', name='Equidistant 1.5 mm', focal_mm=f, projection='equidistant', max_half_deg=top,
                focus_near_m=0.0)
    r = h * math.tan(math.radians(top))
    pose = dict(x=4.0, y=0.0, z=h, yaw=0.0, pitch=90.0, roll=0.0)
    p, c = _plate_project(lens, pose, [(4 - r - 0.4, -r - 0.4), (4 + r + 0.4, -r - 0.4), (4 + r + 0.4, r + 0.4),
                                       (4 - r - 0.4, r + 0.4)], body=((-12.0, -0.3, 0.0), (-11.4, 0.3, 0.6)),
                          cell=0.03)
    m = p.module_for(c)
    if f * math.radians(top) >= m.height_px * m.pixel_um * 1e-3 / 2:
        raise ValueError('the image circle does not lie inside the sensor')
    res, g, X, Y, ce = _plate_eval(p)
    rho = np.hypot(X - 4.0, Y)
    ref = (rho <= r) & g.roi
    vis = ce.vis & g.roi
    diff = vis != ref
    far = diff & (np.abs(rho - r) > 1.5 * g.cell)
    share = float(diff.sum()) / max(int(ref.sum()), 1)
    R = np.sqrt(rho ** 2 + h * h)
    th = np.arctan2(rho, h)
    k = np.sqrt(np.where(th > 1e-6, th / np.where(th > 1e-6, np.sin(th), 1.0), 1.0))
    dens_ref = (f / (m.pixel_um * 1e-3)) * k / (100 * R)
    de = float(np.max(rel_err(ce.dens[vis], dens_ref[vis])))
    return Outcome(max(share, de), {'cells_seen': int(vis.sum()), 'cells_in_the_disc': int(ref.sum()),
                                    'cells_differing': int(diff.sum()),
                                    'cells_differing_away_from_the_edge': int(far.sum()),
                                    'area_m2': float(vis.sum()) * g.cell ** 2, 'disc_area_m2': math.pi * r * r,
                                    'radius_m': r, 'density_rel_error': de, 'cell_m': g.cell},
                   ok=int(far.sum()) == 0)


# ============================================================================ occlusion
SIL_DEG = 1.0                                         # how far a cube map's silhouette may lie from the exact one
SIL_STEPS = (0.1, 0.2, 0.3, 0.45, 0.6, 0.8, 1.0, 1.5, 2.0)


def _ring(D, deg, n=16):
    """n unit directions deg off each direction of D (K, 3): (K, n, 3)."""
    a = np.where(np.abs(D[:, 2:3]) < 0.9, np.array([[0.0, 0.0, 1.0]]), np.array([[1.0, 0.0, 0.0]]))
    e1 = np.cross(D, a)
    e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
    e2 = np.cross(D, e1)
    t = math.radians(deg)
    ph = np.linspace(0, 2 * math.pi, n, endpoint=False)
    R = (math.cos(t) * D[:, None] + math.sin(t) * (np.cos(ph)[None, :, None] * e1[:, None]
                                                    + np.sin(ph)[None, :, None] * e2[:, None]))
    return R / np.linalg.norm(R, axis=2, keepdims=True)


def _silhouette_deg(rc, pos, D, hit, t_max=None):
    """Per direction (K, 3) whose exact status is hit (K,): the angle to the nearest direction whose exact status
    differs, to the steps of SIL_STEPS (inf beyond the last)."""
    out = np.full(len(D), np.inf)
    todo = np.arange(len(D))
    for deg in SIL_STEPS:
        if not len(todo):
            break
        ring = _ring(D[todo], deg)
        L = None if t_max is None else np.repeat(np.asarray(t_max, float)[todo], ring.shape[1])
        tr = rc.first_hit(pos, ring.reshape(-1, 3), t_max=L).reshape(len(todo), -1)
        changed = (np.isfinite(tr) != hit[todo, None]).any(axis=1)
        out[todo[changed]] = deg
        todo = todo[~changed]
    return out


def _occ_setup(ctx):
    """(project, cameras, the static body + rig geometry, an exact ray caster on its triangles): the project given
    when it has a vehicle and cameras, else the sample."""
    p = ctx.project
    vg = ctx.engine.vehicle_geom(p)
    cams = [c for c in p.variant.cameras if c.visible] or [c for v in p.variants for c in v.cameras][:6]
    if vg is None or not cams:
        p = ctx.sample()
        vg = ctx.engine.vehicle_geom(p)
        cams = [c for c in p.variant.cameras if c.visible]
    static = vg.static
    return p, cams, static, RayCaster(static.occ.tri)


def _sil_figure(dist, what):
    steps = [s for s in SIL_STEPS]
    counts = [int(np.sum(np.isclose(dist, s))) for s in steps] + [int(np.sum(~np.isfinite(dist)))]
    return figure(f'How far from the exact silhouette the {what} that disagree lie', 'angle to the exact silhouette',
                  f'{what} that disagree', [series('disagreements', np.arange(len(counts)), counts, kind='bar')],
                  kind='bars', categories=[f'≤ {s:g}°' for s in steps] + ['> 2°'])


@check('occlusion.cube_vs_rays', 'occlusion', 'Cube maps against exact ray casting',
       'share of random directions from each camera of the project in which the occlusion cube map (body and rig, '
       '256 × 256 texels per face) and an exact ray cast against the same triangles disagree on whether the body is '
       'hit; every disagreement must lie within 1° of the exact silhouette',
       'Möller–Trumbore ray–triangle intersection against every occluder triangle, written for the check',
       5e-3, 'share of directions', refs=('greene1986', 'moller1997'), cost='study')
def _occ_cube(ctx):
    p, cams, static, rc = _occ_setup(ctx)
    rng = ctx.rng('occlusion.cube_vs_rays')
    n = 1500 if ctx.quick else 4000
    tot = bad = 0
    depth_err, rows, dists = [], {}, []
    for cam in cams:
        pos = cam.pos()
        cube = ctx.engine.cubes.get(static.occ, pos)
        D = rng.normal(size=(n, 3))
        D /= np.linalg.norm(D, axis=1, keepdims=True)
        dc, _ = cube.lookup(D)
        t = rc.first_hit(pos, D)
        hc, he = np.isfinite(dc), np.isfinite(t)
        dis = np.flatnonzero(hc != he)
        sd = _silhouette_deg(rc, pos, D[dis], he[dis]) if len(dis) else np.zeros(0)
        dists.append(sd)
        both = hc & he
        depth_err.append(np.abs(dc[both] - t[both]) / t[both])
        rows[cam.tag] = {'directions': n, 'disagree': int(len(dis)), 'hit_share': round(float(he.mean()), 4),
                         'farthest_from_silhouette_deg': float(sd.max()) if len(sd) else 0.0}
        tot += n
        bad += len(dis)
    de = np.concatenate(depth_err) if depth_err else np.zeros(1)
    sd = np.concatenate(dists) if dists else np.zeros(0)
    worst = float(sd.max()) if len(sd) else 0.0
    return Outcome(bad / max(tot, 1), {'project': p.name, 'cameras': rows, 'directions': tot, 'disagree': bad,
                                       'farthest_from_silhouette_deg': worst, 'silhouette_band_deg': SIL_DEG,
                                       'depth_rel_error_median': float(np.median(de)),
                                       'depth_rel_error_p95': float(np.percentile(de, 95)),
                                       'triangles': int(len(static.occ.tri))},
                   _sil_figure(sd, 'directions'), ok=worst <= SIL_DEG,
                   note='A cube map texel stores the exact distance along its centre ray to the plane of the '
                        'triangle drawn there (painted far to near by each triangle\'s nearest point): the depth '
                        'error left is the texel\'s size (a random direction lies up to half a texel off the centre '
                        'ray; most at grazing angles and silhouettes), see the median and 95th percentile here. Until '
                        '0.2.0 it stored the triangle centroid\'s distance: 15 % off in the median, over 3× at the '
                        '95th percentile. What the analysis decides from it is checked in occlusion.ground_visibility.')


DEPTH_EXACT = 1e-3            # a texel's stored depth this close (relative) to the exact hit through its centre is exact
DEPTH_NEAR = 0.05             # … else this close to an exact hit within one texel of it: the texel's size, not an error


def _texel_ray(f, y, x, n):
    """Unit directions through the points (row y, column x: fractional texel coordinates, K) of cube face f (K)."""
    from ..core.occlusion import FACES
    F = np.array([FACES[k] for k in range(6)])                         # (6, 3 vectors, 3)
    u = (np.asarray(x, float) + 0.5) / n * 2 - 1
    v = 1 - (np.asarray(y, float) + 0.5) / n * 2
    d = F[f, 0] + u[:, None] * F[f, 1] + v[:, None] * F[f, 2]
    return d / np.linalg.norm(d, axis=1, keepdims=True)


@check('occlusion.cube_depth', 'occlusion', 'Cube-map depth against exact ray casting',
       'share of the texels where a camera\'s cube map (body and rig, 256 × 256 texels per face) is hit whose stored '
       'depth is more than 0.1 % off the exact first hit along the texel\'s centre ray and more than 5 % off every '
       'exact hit within one texel of it (rays through the neighbouring texels\' centres and the texel\'s edges): the '
       'depth must be exact but for the texel\'s own size',
       'Möller–Trumbore ray–triangle intersection against every occluder triangle, written for the check',
       5e-3, 'share of texels', refs=('greene1986', 'moller1997'), cost='study')
def _occ_depth(ctx):
    p, cams, static, rc = _occ_setup(ctx)
    rng = ctx.rng('occlusion.cube_depth')
    k = 1000 if ctx.quick else 3000
    offs = np.array([(dy, dx) for dy in (-1, -0.5, 0, 0.5, 1) for dx in (-1, -0.5, 0, 0.5, 1) if dx or dy], float)
    rows, errs = {}, []
    tot = off = bad = 0
    for cam in cams:
        pos = cam.pos()
        cube = ctx.engine.cubes.get(static.occ, pos)
        n = cube.depth.shape[1]
        idx = np.argwhere(np.isfinite(cube.depth))
        if not len(idx):
            continue
        sel = idx[np.sort(rng.choice(len(idx), size=min(k, len(idx)), replace=False))]
        f, y, x = sel.T
        dc = cube.depth[f, y, x].astype(float)
        t = rc.first_hit(pos, _texel_ray(f, y, x, n))
        with np.errstate(invalid='ignore', divide='ignore'):
            e = np.abs(dc - t) / t
        e = np.where(np.isfinite(e), e, np.inf)                     # the exact ray misses: a silhouette texel
        errs.append(e)
        o = np.flatnonzero(e > DEPTH_EXACT)
        unexplained = 0
        if len(o):
            m = len(offs)
            tn = rc.first_hit(pos, _texel_ray(np.repeat(f[o], m), np.repeat(y[o], m) + np.tile(offs[:, 0], len(o)),
                                              np.repeat(x[o], m) + np.tile(offs[:, 1], len(o)), n)).reshape(len(o), m)
            with np.errstate(invalid='ignore'):
                near = (np.abs(tn - dc[o, None]) <= DEPTH_NEAR * tn).any(axis=1)
            unexplained = int((~near).sum())
        rows[cam.tag] = {'texels': int(len(sel)), 'off_by_more_than_0_1pct': int(len(o)),
                         'no_surface_within_one_texel': unexplained}
        tot += len(sel)
        off += len(o)
        bad += unexplained
    e = np.concatenate(errs) if errs else np.zeros(0)
    fin = e[np.isfinite(e)]
    edges = (1e-6, 1e-3, 1e-2, 5e-2)
    counts = [int(np.sum(fin <= edges[0]))] + [int(np.sum((fin > a) & (fin <= b))) for a, b in zip(edges, edges[1:])] \
        + [int(np.sum(fin > edges[-1])), int(np.sum(~np.isfinite(e)))]
    cats = ['≤ 1e-6', '≤ 0.1 %', '≤ 1 %', '≤ 5 %', '> 5 %', 'exact ray misses']
    fig = figure('Stored depth against the exact hit along the texel\'s centre ray', 'relative difference',
                 'texels', [series('texels', np.arange(len(counts)), counts, kind='bar')], kind='bars', categories=cats)
    return Outcome(bad / max(tot, 1), {'project': p.name, 'cameras': rows, 'texels': tot,
                                       'off_by_more_than_0_1pct': off, 'no_surface_within_one_texel': bad,
                                       'centre_rel_error_median': float(np.median(fin)) if len(fin) else None,
                                       'centre_rel_error_p99': float(np.percentile(fin, 99)) if len(fin) else None,
                                       'triangles': int(len(static.occ.tri))},
                   fig, ok=tot > 0,
                   note='The texels off by more than 0.1 % lie at a silhouette or a crease: the triangle drawn there '
                        'covers part of the texel but not its centre (a polygon\'s one-pixel outline), so the depth '
                        'is of a surface within one texel of the centre ray.')


@check('occlusion.ground_visibility', 'occlusion', 'Which ground points a camera sees past the body',
       'share of ground points of the region of interest in each camera\'s image whose visibility '
       '(analysis.eval_points with the cube map) differs from an exact test of the straight line from the lens to the '
       'point against every occluder triangle (with the same 2 cm + 1 % distance allowance); every disagreement must '
       'lie within 1° of the exact silhouette',
       'Möller–Trumbore ray–triangle intersection along the line of sight, written for the check',
       5e-3, 'share of points', refs=('greene1986', 'moller1997'), cost='study')
def _occ_ground(ctx):
    from ..core.analysis import eval_points, camera_frames
    from ..core.lens import resolve
    p, cams, static, rc = _occ_setup(ctx)
    grid = ctx.engine.roi_grid(p)
    P_all = grid.points(0.0)[grid.roi]
    rng = ctx.rng('occlusion.ground_visibility')
    n = 1000 if ctx.quick else 4000
    tot = bad = 0
    rows, dists = {}, []
    for cam in cams:
        pos = cam.pos()
        intr = resolve(p.module_for(cam), p.lens_for(cam))
        cube = ctx.engine.cubes.get(static.occ, pos)
        _, W2O = camera_frames(cam)
        _, _, _, ok = intr.project_dirs((P_all - pos) @ W2O.T)
        idx = np.flatnonzero(ok)
        if not len(idx):
            continue
        idx = np.sort(rng.choice(idx, size=min(n, len(idx)), replace=False))
        P = P_all[idx]
        vis, _, _, _ = eval_points(P, cam, intr, cube)
        d = P - pos
        dist = np.linalg.norm(d, axis=1)
        D = d / dist[:, None]
        L = dist - (0.02 + 0.01 * dist)
        t = rc.first_hit(pos, D, t_max=L)
        hidden = np.isfinite(t)
        dis = np.flatnonzero(hidden == vis)                 # visible but hidden, or hidden but visible
        sd = _silhouette_deg(rc, pos, D[dis], hidden[dis], t_max=L[dis]) if len(dis) else np.zeros(0)
        dists.append(sd)
        rows[cam.tag] = {'points': int(len(idx)), 'hidden_share': round(float(hidden.mean()), 4),
                         'disagree': int(len(dis)),
                         'farthest_from_silhouette_deg': float(sd.max()) if len(sd) else 0.0}
        tot += len(idx)
        bad += len(dis)
    sd = np.concatenate(dists) if dists else np.zeros(0)
    worst = float(sd.max()) if len(sd) else 0.0
    return Outcome(bad / max(tot, 1), {'project': p.name, 'cameras': rows, 'points': tot, 'disagree': bad,
                                       'farthest_from_silhouette_deg': worst, 'silhouette_band_deg': SIL_DEG},
                   _sil_figure(sd, 'points'), ok=worst <= SIL_DEG and tot > 0)


# ============================================================================ perception range
def _pinhole_intr(f_mm=6.0):
    from ..core.lens import Intrinsics
    return Intrinsics(width=1920, height=1536, pixel_mm=0.003, f_mm=f_mm, projection='rectilinear', kb=(0, 0, 0, 0),
                      cx=960.0, cy=768.0, max_theta=math.radians(89.0), focus_near=0.1, focus_far=math.inf)


@check('perception.pinhole_pixels', 'perception', 'Object pixels through a pinhole lens',
       'largest relative difference of an object\'s pixel height and width (perception.object_pixels) from f·H/Z and '
       'f·W/Z for a level 6 mm pinhole camera, objects on and off the axis and a sign on a post',
       'the pinhole projection of a fronto-parallel segment, f·H/Z with Z the depth along the axis', 1e-9,
       'relative', refs=('hartley2004', 'johnson1958'))
def _pr_pinhole(ctx):
    from ..core import perception as PR
    from ..core.project import Camera
    cam = Camera(x=0.0, y=0.0, z=0.75, yaw=0.0, pitch=0.0, roll=0.0)
    it = _pinhole_intr()
    car = PR.ObjectClass('Car', 1.5, 1.8, 20.0)
    Z = np.array([5.0, 10.0, 40.0, 100.0])
    h, w, seen = PR.object_pixels(cam, it, np.c_[Z, np.zeros(4)], car)
    e = [rel_err(h, it.f_px * 1.5 / Z).max(), rel_err(w, it.f_px * 1.8 / Z).max()]
    h2, _, s2 = PR.object_pixels(cam, it, np.array([[20.0, 6.0]]), car)          # off the axis: the depth counts
    e.append(float(rel_err(h2, it.f_px * 1.5 / 20.0).max()))
    sign = PR.ObjectClass('Sign', 0.6, 0.6, 12.0, base=0.45)
    h3, _, s3 = PR.object_pixels(cam, it, np.array([[30.0, 0.0]]), sign)
    e.append(float(rel_err(h3, it.f_px * 0.6 / 30.0).max()))
    err = float(max(e))
    return Outcome(err, {'f_px': it.f_px, 'heights_px': h.round(4).tolist(), 'widths_px': w.round(4).tolist()},
                   ok=bool(seen.all() and s2.all() and s3.all()))


@check('perception.fisheye_pixels', 'perception', 'Object pixels through a fisheye follow its local sampling',
       'largest relative difference of an object\'s pixel height and width through the built-in H05R2 lens, 30 m '
       'out on the axis and 55° off it, from its size over the distance times the lens\'s local sampling (radial '
       'dr/dθ, tangential r/sin θ)',
       'the first-order local magnification of the lens at the object\'s direction', 0.02, 'relative',
       refs=('kannala2006',))
def _pr_fisheye(ctx):
    from ..core import perception as PR
    from ..core.lens import builtin_lenses, builtin_modules, resolve, dr_dtheta, r_of_theta
    from ..core.project import Camera
    it = resolve(builtin_modules()[0], builtin_lenses()[0])
    cam = Camera(x=0.0, y=0.0, z=0.75, yaw=0.0, pitch=0.0, roll=0.0)
    car = PR.ObjectClass('Car', 1.5, 1.8, 20.0)
    D, a = 30.0, math.radians(55.0)
    G = np.array([[D, 0.0], [D * math.cos(a), D * math.sin(a)]])
    h, w, seen = PR.object_pixels(cam, it, G, car)
    rad = lambda th: float(dr_dtheta(th, it.f_mm, it.projection, it.kb)) / it.pixel_mm            # noqa: E731
    tan = lambda th: float(r_of_theta(th, it.f_mm, it.projection, it.kb)) / it.pixel_mm / math.sin(th)  # noqa: E731
    want = [rad(1e-4) * 1.5 / D, rad(1e-4) * 1.8 / D, tan(a) * 1.5 / D, rad(a) * 1.8 / D]
    got = [h[0], w[0], h[1], w[1]]
    e = rel_err(got, want)
    return Outcome(float(e.max()), {'pixels': [round(float(x), 4) for x in got],
                                    'local_sampling_estimate': [round(x, 4) for x in want]}, ok=bool(seen.all()))


@check('perception.far_range', 'perception', 'Perception range of a pinhole camera',
       'relative difference of the far range ahead the study finds for a car with a level 6 mm rectilinear camera '
       'on the sample\'s front bumper from the range where the car is exactly min_px tall, f·H / min_px (measured '
       'from the vehicle centre), for min_px 20 and 33',
       'the pinhole relation f·H/Z = min_px solved for Z', 1e-3, 'relative', refs=('hartley2004', 'johnson1958'),
       cost='study')
def _pr_far(ctx):
    from ..core import perception as PR
    from ..core.lens import Lens, resolve
    p = ctx.sample()
    p.lenses['val_pin'] = Lens(id='val_pin', name='Pinhole 6 mm', focal_mm=6.0, projection='rectilinear')
    for c in p.variant.cameras:
        c.visible = c.tag == 'F'
    f = p.variant.by_tag('F')
    f.lens_id, f.yaw, f.pitch, f.roll = 'val_pin', 0.0, 0.0, 0.0
    it = resolve(p.module_for(f), p.lens_for(f))
    rows, worst = {}, 0.0
    for min_px in (20.0, 33.0):
        st = PR.PerceptionSettings.from_dict({'classes': [dict(PR.DEFAULT_CLASSES[0], min_px=min_px)]})
        out = PR.analyse(p, ctx.engine, None, st, weather='clear')
        k = int(np.argmin(np.abs(out.az - 0.0)))
        want = f.x - out.centre[0] + it.f_px * 1.5 / min_px
        got = float(out.far[0, k])
        rows[f'{min_px:g} px'] = {'far_m': got, 'closed_form_m': want}
        worst = max(worst, abs(got - want) / want)
    return Outcome(worst, rows)


# ============================================================================ critical targets
def _crit_project(cam_z=1.2, target=None, **st):
    """No vehicle: one 4 mm rectilinear camera at (0, 0, cam_z) looking along +x, one target."""
    from ..core import critical as CR
    from ..core.lens import Lens
    from ..core.project import Project, Camera, Variant
    p = Project()
    p.lenses['pin'] = Lens(id='pin', name='Pinhole', focal_mm=4.0, projection='rectilinear', max_half_deg=80.0,
                           focus_near_m=0.0)
    c = Camera(tag='P', name='Pinhole', x=0.0, y=0.0, z=cam_z, yaw=0.0, pitch=0.0, roll=0.0, module_id='sturdecam31',
               lens_id='pin')
    v = Variant(name='Pinhole', cameras=[c])
    p.variants = [v]
    p.active_uid = v.uid
    t = CR.clean_target(dict({'kind': 'pedestrian', 'name': 'Ped', 'x': 10.0, 'y': 0.0}, **(target or {})))
    p.analysis.critical = dict({'targets': [t], 'install_mm': 10.0, 'install_deg': 1.0}, **st)
    return p, c


@check('critical.pinhole_pixels', 'critical', 'Critical-target pixels through a pinhole lens',
       'largest relative difference of a 1.7 m × 0.5 m pedestrian\'s pixel height and width 10 m ahead of a level '
       '4 mm pinhole camera, and of its distance, from f·h/Z, f·w/Z and the straight-line distance',
       'the pinhole projection, written out in the check', 1e-6, 'relative', refs=('hartley2004',))
def _crit_pixels(ctx):
    from ..core import critical as CR
    from ..core.analysis import Engine
    from ..core.lens import resolve
    p, c = _crit_project()
    it = resolve(p.module_for(c), p.lens_for(c))
    lk = CR.run(p, Engine(), envelopes=False).targets[0].looks[0]
    want = [it.f_px * 1.7 / 10.0, it.f_px * 0.5 / 10.0, math.hypot(10.0, 0.85 - 1.2)]
    got = [lk.px, lk.px_width, lk.dist]
    e = rel_err(got, want)
    return Outcome(float(e.max()), {'got': [float(x) for x in got], 'closed_form': want, 'seen': lk.seen},
                   ok=abs(lk.seen - 1.0) < 1e-9 and bool(lk.met))


@check('critical.pinhole_envelope', 'critical', 'Mounting envelopes of a pinhole camera',
       'largest difference of the single-axis pitch and yaw limits the envelope search finds from where the target\'s '
       'top, feet or outer columns leave the image (half the field minus their angle), and of the backward x limit '
       'from where its height falls to min_px (in metres × 100, so 0.02 is 0.2 mm)',
       'the angles of the target\'s edges against the half field, and f·h / min_px, written out in the check', 0.02,
       'degrees', refs=('hartley2004',), cost='study')
def _crit_envelope(ctx):
    from ..core import critical as CR
    from ..core.analysis import Engine
    from ..core.lens import resolve
    D, zc, h, back = 10.0, 1.2, 1.7, 0.5
    p, c = _crit_project(target={'min_share': 1.0, 'min_px': 0.0}, search_mm=1000.0, search_deg=40.0)
    it = resolve(p.module_for(c), p.lens_for(c))
    f = it.f_mm / it.pixel_mm
    st = CR.settings_of(p)
    st.targets[0]['min_px'] = f * h / (D + back)                        # met exactly 0.5 m further back
    p.analysis.critical = st.to_dict()
    env = CR.run(p, Engine()).targets[0].looks[0].envelope
    half_v = math.degrees(math.atan(it.cy / f))
    half_h = math.degrees(math.atan((it.width / 2) / f))
    want = {'pitch down (top leaves)': (env.hi['pitch'], half_v - math.degrees(math.atan((h - zc) / D))),
            'pitch up (feet leave)': (env.lo['pitch'], half_v - math.degrees(math.atan(zc / D))),
            'yaw (outer column leaves)': (env.lo['yaw'], half_h - math.degrees(math.atan(0.2 / D))),
            'x back, ×100 m (pixels run out)': (env.lo['x'] * 100, back * 100)}
    err = max(abs(a - b) for a, b in want.values())
    return Outcome(err, {k: {'found': a, 'closed_form': b} for k, (a, b) in want.items()},
                   ok=env.lo_why['x'] == 'pixels' and env.hi_why['pitch'] == 'seen')
