"""Perception range: how far out the rig resolves the objects a driving dataset needs.

An object (car, pedestrian, cyclist, traffic sign; real dimensions) stands on the ground at a point; a camera resolves
it when the object is inside the image and the lens circle, its centre is not hidden by the body (the same occlusion
cube as the coverage analysis, so everything the engine treats as an occluder counts), and its height in the image is
at least the class's minimum pixel height. The pixel height is the distance between the projections of the object's
bottom and top through the camera's real lens model (exact, no small-angle approximation); the width likewise between
its left and right edges, square to the line of sight.

Directions are azimuths about the vehicle centre (the middle of the body's bounding box, on the ground), counter-
clockwise from +x (forward); ranges are measured from that centre, on the ground, in metres.
"""
from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field

import numpy as np

from .analysis import camera_frames
from .lens import dr_dtheta, r_of_theta, resolve

SCHEMA = 'vantage.perception_range/1'

# Typical detector minimums: set your own model's. Height and width in metres; min_px: the object's height in the
# image a detector needs; base: height of the object's bottom above the ground (a sign on a post).
DEFAULT_CLASSES = [
    {'name': 'Car', 'height': 1.5, 'width': 1.8, 'min_px': 20.0, 'base': 0.0},
    {'name': 'Pedestrian', 'height': 1.7, 'width': 0.5, 'min_px': 24.0, 'base': 0.0},
    {'name': 'Cyclist', 'height': 1.7, 'width': 0.6, 'min_px': 24.0, 'base': 0.0},
    {'name': 'Traffic sign', 'height': 0.6, 'width': 0.6, 'min_px': 12.0, 'base': 2.0},
]
SECTORS = [('front', 0.0), ('front-left', 45.0), ('left', 90.0), ('rear-left', 135.0), ('rear', 180.0),
           ('rear-right', -135.0), ('right', -90.0), ('front-right', -45.0)]
DEFAULT_REQUIRED = {'front': 80.0, 'front-left': 40.0, 'left': 25.0, 'rear-left': 30.0, 'rear': 40.0,
                    'rear-right': 30.0, 'right': 25.0, 'front-right': 40.0}
STEREO_AT = (10.0, 20.0, 50.0)       # ranges for the stereo depth resolution
DISPARITY_PX = 0.25                  # disparity error assumed for the stereo depth resolution
OVERLAP_AT = 20.0                    # camera pairs overlap where both see a car at this range
MIN_OVERLAP_DEG = 5.0                # a pair overlapping less than this is no stereo pair


@dataclass
class ObjectClass:
    name: str
    height: float
    width: float
    min_px: float
    base: float = 0.0


def clean_class(c) -> dict | None:
    """An object class as stored ({name, height, width, min_px, base}), or None when it is not a usable one."""
    try:
        o = {'name': str(c.get('name') or 'Object')[:40], 'height': float(c['height']), 'width': float(c['width']),
             'min_px': float(c['min_px']), 'base': float(c.get('base', 0.0) or 0.0)}
    except (KeyError, TypeError, ValueError, AttributeError):
        return None
    if all(math.isfinite(v) for k, v in o.items() if k != 'name') and o['height'] > 0 and o['width'] > 0 \
            and o['min_px'] > 0 and o['base'] >= 0:
        return o
    return None


@dataclass
class PerceptionSettings:
    classes: list = field(default_factory=lambda: [dict(c) for c in DEFAULT_CLASSES])
    required: dict = field(default_factory=lambda: dict(DEFAULT_REQUIRED))
    max_range: float = 250.0
    azimuths: int = 360
    samples: int = 120               # ranges per azimuth, log-spaced from 1 m to max_range

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d: dict | None) -> 'PerceptionSettings':
        s = PerceptionSettings()
        d = d or {}
        cls = [o for o in map(clean_class, d.get('classes') or []) if o is not None]
        seen = set()
        for o in cls:                    # names key the verdicts: keep them unique
            base, n = o['name'], 2
            while o['name'] in seen:
                o['name'] = f'{base} {n}'
                n += 1
            seen.add(o['name'])
        if cls:
            s.classes = cls
        req = dict(DEFAULT_REQUIRED)
        for k, v in (d.get('required') or {}).items():
            if k in req:
                try:
                    v = float(v)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(v) and v >= 0:
                    req[k] = v
        s.required = req
        for k, lo, hi in (('max_range', 10.0, 1000.0), ('azimuths', 36, 1440), ('samples', 20, 400)):
            if k in d:
                try:
                    v = float(d[k])
                except (TypeError, ValueError):
                    continue
                if math.isfinite(v):
                    setattr(s, k, type(getattr(s, k))(min(max(v, lo), hi)))
        return s

    def object_classes(self) -> list:
        return [ObjectClass(**c) for c in self.classes]


@dataclass
class PerceptionResult:
    settings: PerceptionSettings
    classes: list                   # [ObjectClass]
    cams: list                      # [(uid, tag, slot)]
    centre: tuple                   # (x, y) of the vehicle centre, m
    body: tuple | None              # ((x0, y0), (x1, y1)) footprint bounding box, None without a vehicle
    az: np.ndarray                  # (A,) azimuths, degrees
    ranges: np.ndarray              # (R,) ranges, m (log-spaced)
    far: np.ndarray                 # (K, A) farthest range any camera resolves the class at, 0 = never
    near: np.ndarray                # (K, A) nearest such range, inf = never
    best: np.ndarray                # (K, A) index into cams of the camera with the farthest range, -1 = none
    cam_far: np.ndarray             # (C, K, A) farthest range per camera
    count: np.ndarray               # (K, A, R) int8: cameras resolving the class at (azimuth, range)
    sectors: list                   # per sector: {'name', 'centre_deg', 'required_m', 'classes': {name: {...}}}
    pairs: list                     # overlapping camera pairs
    metrics: dict
    variant_uid: str = ''

    def tag(self, i):
        return self.cams[i][1] if 0 <= i < len(self.cams) else None

    def at(self, k, az_deg):
        """(far, near, camera tag) for class k in a direction."""
        a = int(round((az_deg % 360.0) / 360.0 * len(self.az))) % len(self.az)
        return float(self.far[k, a]), float(self.near[k, a]), self.tag(int(self.best[k, a]))


# =========================================================================== geometry
def _project(P, pos, W2O, intr):
    d = P - pos
    u, v, th, ok = intr.project_dirs(d @ W2O.T)
    return u, v, th, ok, d


def _angular_px(intr, theta):
    """Local angular resolution, px/rad: the geometric mean of radial and tangential sampling."""
    th = np.maximum(np.asarray(theta, float), 1e-4)
    rad = np.maximum(dr_dtheta(th, intr.f_mm, intr.projection, intr.kb), 1e-9) / intr.pixel_mm
    tan = np.maximum(r_of_theta(th, intr.f_mm, intr.projection, intr.kb), 1e-12) / intr.pixel_mm / np.sin(
        np.minimum(th, math.pi - 1e-3))
    return np.sqrt(rad * tan * intr.aspect)


def object_pixels(cam, intr, ground_xy, cls: ObjectClass, cube=None):
    """Height and width in pixels of the object standing at each ground point (N, 2), and whether the camera sees it
    (inside the image and the lens circle, centre not hidden by the body)."""
    G = np.asarray(ground_xy, float).reshape(-1, 2)
    n = len(G)
    pos = cam.pos()
    _, W2O = camera_frames(cam)
    zb, zt = cls.base, cls.base + cls.height
    zm = 0.5 * (zb + zt)
    bot = np.c_[G, np.full(n, zb)]
    top = np.c_[G, np.full(n, zt)]
    ub, vb, _, okb, _ = _project(bot, pos, W2O, intr)
    ut, vt, _, okt, _ = _project(top, pos, W2O, intr)
    h = np.hypot(ut - ub, vt - vb)
    los = G - pos[:2]
    L = np.maximum(np.linalg.norm(los, axis=1), 1e-9)
    perp = np.c_[-los[:, 1] / L, los[:, 0] / L] * (cls.width / 2)
    ul, vl, _, okl, _ = _project(np.c_[G + perp, np.full(n, zm)], pos, W2O, intr)
    ur, vr, _, okr, _ = _project(np.c_[G - perp, np.full(n, zm)], pos, W2O, intr)
    w = np.hypot(ur - ul, vr - vl)
    mid = np.c_[G, np.full(n, zm)]
    seen = okb & okt
    if cube is not None:
        d = mid - pos
        dist = np.linalg.norm(d, axis=1)
        depth, _ = cube.lookup(d)
        seen &= depth > dist - (0.02 + 0.01 * dist)
    return np.where(seen, h, 0.0), np.where(seen & okl & okr, w, 0.0), seen


# =========================================================================== the study
def _far_range(ok, visible, hpx, ranges, min_px, rows):
    """Per azimuth (rows of ok), the farthest range at which the object is resolved: the last resolved sample, and
    where the next one is in view but too small, the range between them at which its pixel height crosses min_px
    (1/height interpolated linearly in range: exact for a pinhole, within the sampling for any lens)."""
    R = len(ranges)
    has = ok.any(1)
    i = R - 1 - np.argmax(ok[:, ::-1], axis=1)
    j = np.minimum(i + 1, R - 1)
    far = np.where(has, ranges[i], 0.0)
    hi, hj = hpx[rows, i], hpx[rows, j]
    cross = has & (j > i) & visible[rows, j] & (hj > 0) & (hj < min_px) & (hi > hj)
    with np.errstate(divide='ignore', invalid='ignore'):
        t = (1.0 / min_px - 1.0 / hi) / (1.0 / hj - 1.0 / hi)
    t = np.clip(np.where(cross, t, 0.0), 0.0, 1.0)
    return far + t * (ranges[j] - ranges[i])


def _sector_of(az_deg):
    """Index into SECTORS of each azimuth (sectors are 45° wide, centred on their direction)."""
    a = (np.asarray(az_deg, float) + 22.5) % 360.0
    order = [0, 1, 2, 3, 4, 5, 6, 7]      # 0°, 45°, 90°, 135°, 180°, -135° (225°), -90° (270°), -45° (315°)
    return np.array(order)[(a // 45.0).astype(int) % 8]


def analyse(project, engine, res=None, settings: PerceptionSettings | None = None, progress=None, cancel=None):
    """The perception range of the project's active variant. Returns None when cancelled."""
    t0 = time.perf_counter()
    st = settings or PerceptionSettings.from_dict(getattr(project.analysis, 'perception', None))
    classes = st.object_classes()
    vg = engine.vehicle_geom(project)
    if vg is not None:
        lo, hi = vg.bbox
        centre = (float(lo[0] + hi[0]) / 2, float(lo[1] + hi[1]) / 2)
        body = ((float(lo[0]), float(lo[1])), (float(hi[0]), float(hi[1])))
    else:
        centre, body = (0.0, 0.0), None
    A, R = int(st.azimuths), int(st.samples)
    az = np.arange(A) * 360.0 / A
    ranges = np.geomspace(1.0, float(st.max_range), R)
    ca, sa = np.cos(np.radians(az)), np.sin(np.radians(az))
    gx = centre[0] + ca[:, None] * ranges[None, :]
    gy = centre[1] + sa[:, None] * ranges[None, :]
    G = np.stack([gx, gy], -1).reshape(-1, 2)
    inside = np.zeros(len(G), bool)
    if body is not None:
        (x0, y0), (x1, y1) = body
        inside = (G[:, 0] > x0 - 0.05) & (G[:, 0] < x1 + 0.05) & (G[:, 1] > y0 - 0.05) & (G[:, 1] < y1 + 0.05)
    cams = [c for c in project.variant.cameras if c.visible]
    K, C = len(classes), len(cams)
    resolved = np.zeros((C, K, A, R), bool)
    cam_far = np.zeros((C, K, A))
    rows = np.arange(A)
    seen20 = np.zeros((C, A), bool)
    theta20 = np.full((C, A), np.nan)
    i20 = int(np.argmin(np.abs(ranges - OVERLAP_AT)))
    intrs = []
    steps = max(1, C * K)
    for ci, c in enumerate(cams):
        intr = resolve(project.module_for(c), project.lens_for(c))
        intrs.append(intr)
        cube = engine.cube(c.pos()) if vg is not None else None
        for k, cls in enumerate(classes):
            if cancel is not None and cancel():
                return None
            hpx, _, seen = object_pixels(c, intr, G, cls, cube)
            ok = seen & ~inside & (hpx >= cls.min_px)
            resolved[ci, k] = ok.reshape(A, R)
            cam_far[ci, k] = _far_range(ok.reshape(A, R), (seen & ~inside).reshape(A, R), hpx.reshape(A, R),
                                        ranges, cls.min_px, rows)
            if k == 0:
                s = (seen & ~inside).reshape(A, R)[:, i20]
                seen20[ci] = s
                P = np.c_[G.reshape(A, R, 2)[:, i20], np.full(A, cls.base + cls.height / 2)]
                _, _, th, _, _ = _project(P, c.pos(), camera_frames(c)[1], intr)
                theta20[ci] = th
            if progress is not None:
                progress((ci * K + k + 1) / steps)
    count = resolved.sum(0).astype(np.int8) if C else np.zeros((K, A, R), np.int8)
    anyr = count > 0
    far = cam_far.max(0) if C else np.zeros((K, A))
    near = np.where(anyr.any(-1), ranges[np.argmax(anyr, axis=-1)], np.inf)
    best = np.where(far > 0, np.argmax(cam_far, axis=0), -1) if C else np.full((K, A), -1)
    # ---- sectors
    sec = _sector_of(az)
    sectors = []
    for si, (name, cdeg) in enumerate(SECTORS):
        m = sec == si
        req = float(st.required.get(name, 0.0))
        out = {}
        for k, cls in enumerate(classes):
            v = far[k, m]
            lo_ = float(v.min()) if v.size else 0.0
            out[cls.name] = {'min_m': lo_, 'median_m': float(np.median(v)) if v.size else 0.0,
                             'pass': bool(lo_ >= req), 'shortfall_m': max(0.0, req - lo_),
                             'worst_azimuth_deg': float(az[m][int(np.argmin(v))]) if v.size else None}
        sectors.append({'name': name, 'centre_deg': cdeg, 'required_m': req, 'classes': out})
    # ---- camera pairs
    pairs = []
    for i in range(C):
        for j in range(i + 1, C):
            both = seen20[i] & seen20[j]
            ov = float(both.sum() * 360.0 / A)
            if ov < MIN_OVERLAP_DEG:
                continue
            mid = math.degrees(math.atan2(sa[both].mean(), ca[both].mean()))
            a = int(np.argmin(np.abs(((az - mid) + 180) % 360 - 180)))
            if not both[a]:
                a = int(np.flatnonzero(both)[len(np.flatnonzero(both)) // 2])
            f = min(float(_angular_px(intrs[i], theta20[i, a])), float(_angular_px(intrs[j], theta20[j, a])))
            B = float(np.linalg.norm(cams[i].pos() - cams[j].pos()))
            pairs.append({'a': cams[i].uid, 'b': cams[j].uid, 'tags': (cams[i].tag, cams[j].tag), 'overlap_deg': ov,
                          'direction_deg': float(az[a]), 'baseline_m': B, 'f_px_per_rad': f,
                          'depth_error_m': {f'{z:g}': (z * z * DISPARITY_PX / (f * B) if B > 1e-6 else math.inf)
                                            for z in STEREO_AT}})
    pairs.sort(key=lambda p: -p['overlap_deg'])
    if progress is not None:
        progress(1.0)
    metrics = {'seconds': time.perf_counter() - t0, 'points': int(len(G) * C * K), 'cameras': C,
               'passed': sum(1 for s in sectors for v in s['classes'].values() if v['pass']),
               'checks': len(sectors) * K}
    return PerceptionResult(st, classes, [(c.uid, c.tag, c.slot) for c in cams], centre, body, az, ranges, far, near,
                            best, cam_far, count, sectors, pairs, metrics,
                            getattr(res, 'variant_uid', '') or project.variant.uid)


# =========================================================================== output
def summary(pr: PerceptionResult) -> dict:
    """Compact figures (the assistant, the batch report header)."""
    out = {}
    for cls in pr.classes:
        out[cls.name] = {s['name']: {'min_m': round(s['classes'][cls.name]['min_m'], 1),
                                     'required_m': s['required_m'],
                                     'pass': s['classes'][cls.name]['pass']} for s in pr.sectors}
    return {'per_class_sector': out,
            'stereo_pairs': [{'pair': list(p['tags']), 'overlap_deg': round(p['overlap_deg'], 1),
                              'baseline_m': round(p['baseline_m'], 3),
                              'depth_error_m': {k: round(v, 3) for k, v in p['depth_error_m'].items()}}
                             for p in pr.pairs[:4]],
            'passed': pr.metrics['passed'], 'checks': pr.metrics['checks']}


def to_doc(pr: PerceptionResult, project=None, L=None, A=None, origin=(0.0, 0.0)) -> dict:
    """The JSON export: per azimuth per class the farthest range and the camera, sector verdicts, camera pairs.
    L converts lengths to the export units, A angles; origin is the export frame's origin in the vehicle frame."""
    L = L or (lambda v: round(float(v), 4))
    A = A or (lambda v: round(float(v), 4))
    ox, oy = float(origin[0]), float(origin[1])
    rng = lambda v: None if not math.isfinite(float(v)) else L(v)
    doc = {
        'schema': SCHEMA,
        'project': getattr(project, 'name', ''), 'variant': project.variant.name if project is not None else '',
        'centre': [L(pr.centre[0] - ox), L(pr.centre[1] - oy)],
        'directions': 'azimuth about the vehicle centre, counter-clockwise from +x (forward); range from that centre '
                      'on the ground',
        'resolved_when': 'the object is inside the image and the lens circle, its centre is not hidden by the body, '
                         'and its height in the image is at least min_px',
        'classes': [{'name': c.name, 'height': L(c.height), 'width': L(c.width), 'min_px': c.min_px,
                     'above_ground': L(c.base)} for c in pr.classes],
        'required_ranges': {k: L(v) for k, v in pr.settings.required.items()},
        'azimuth': [A(a) for a in pr.az],
        'per_class': {c.name: {'far': [rng(v) for v in pr.far[k]],
                               'near': [rng(v) for v in pr.near[k]],
                               'camera': [pr.tag(int(i)) for i in pr.best[k]]} for k, c in enumerate(pr.classes)},
        'per_camera': {tag: {c.name: [rng(v) for v in pr.cam_far[ci, k]] for k, c in enumerate(pr.classes)}
                       for ci, (_, tag, _) in enumerate(pr.cams)},
        'sectors': [{'name': s['name'], 'centre': A(s['centre_deg']), 'required': L(s['required_m']),
                     'classes': {n: {'min': L(v['min_m']), 'median': L(v['median_m']), 'pass': v['pass'],
                                     'shortfall': L(v['shortfall_m'])} for n, v in s['classes'].items()}}
                    for s in pr.sectors],
        'stereo': {'disparity_error_px': DISPARITY_PX, 'overlap_at': L(OVERLAP_AT),
                   'pairs': [{'cameras': list(p['tags']), 'overlap': A(p['overlap_deg']),
                              'direction': A(p['direction_deg']), 'baseline': L(p['baseline_m']),
                              'focal_px_per_rad': round(p['f_px_per_rad'], 2),
                              'depth_error': {k: rng(v) for k, v in p['depth_error_m'].items()}} for p in pr.pairs]},
    }
    return doc


# per class, here and in the UI (no green or red: the required ranges are drawn green when met, red when short)
CLASS_COLORS = ['#3987e5', '#ec835a', '#9085e9', '#fab219', '#2fb5c7', '#d55181']
PNG_RINGS = (10, 25, 50, 100, 200, 400)


def _bgr(hex_):
    return int(hex_[5:7], 16), int(hex_[3:5], 16), int(hex_[1:3], 16)


def png(pr: PerceptionResult, size=900, k_sel=0) -> bytes:
    """The plan of the detection envelopes as PNG: forward up, left to the left, square-root range scale with rings;
    the required range per sector for class k_sel dashed (green met, red short); the body in the middle."""
    import cv2
    head = 56
    img = np.full((size + head, size, 3), (30, 29, 28), np.uint8)
    cx, cy = size / 2, head + size / 2
    rmax = float(pr.ranges[-1])
    scale = (size / 2 - 34) / math.sqrt(rmax)
    k_sel = min(max(int(k_sel or 0), 0), len(pr.classes) - 1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    aa = cv2.LINE_AA

    def at(r, a):
        s, t = math.sqrt(max(r, 0.0)) * scale, math.radians(a)
        return cx - math.sin(t) * s, cy - math.cos(t) * s

    def poly(pts):
        return np.round(np.array(pts) * 16).astype(np.int32)            # 4 fractional bits

    def text(t, x, y, col, s=0.45, th=1):
        cv2.putText(img, t, (int(x), int(y)), font, s, col, th, aa)
    for ring in PNG_RINGS:
        if ring <= rmax * 1.001:
            rr = int(round(math.sqrt(ring) * scale * 16))
            cv2.circle(img, (int(cx * 16), int(cy * 16)), rr, (62, 61, 58), 1, aa, 4)
            text(f'{ring} m', cx + 4, cy - rr / 16 + 14, (140, 140, 135), 0.4)
    for name, a in SECTORS:
        x, y = at(rmax, a - 22.5)
        cv2.line(img, (int(cx), int(cy)), (int(x), int(y)), (52, 51, 49), 1, aa)
    cl_sel = pr.classes[k_sel]
    for s in pr.sectors:
        ok = s['classes'][cl_sel.name]['pass']
        col = (12, 163, 12) if ok else (59, 59, 208)        # the theme's good and critical
        pts = [at(s['required_m'], s['centre_deg'] - 22.5 + t * 45.0 / 24) for t in range(25)]
        for j in range(0, 24, 2):                                         # dashed
            cv2.polylines(img, [poly(pts[j:j + 2])], False, col, 2, aa, 4)
    order = [k for k in range(len(pr.classes)) if k != k_sel] + [k_sel]
    for k in order:
        pts = poly([at(r, a) for r, a in zip(pr.far[k], pr.az)])
        col = _bgr(CLASS_COLORS[k % len(CLASS_COLORS)])
        if k == k_sel:
            over = img.copy()
            cv2.fillPoly(over, [pts], col, aa, 4)
            cv2.addWeighted(over, 0.16, img, 0.84, 0, img)
        cv2.polylines(img, [pts], True, col, 3 if k == k_sel else 2, aa, 4)
    if pr.body is not None:
        (x0, y0), (x1, y1) = pr.body
        edge = []
        for (ax, ay), (bx, by) in zip(((x0, y0), (x1, y0), (x1, y1), (x0, y1)), ((x1, y0), (x1, y1), (x0, y1), (x0, y0))):
            for t in np.linspace(0, 1, 12, endpoint=False):
                dx, dy = ax + (bx - ax) * t - pr.centre[0], ay + (by - ay) * t - pr.centre[1]
                edge.append(at(math.hypot(dx, dy), math.degrees(math.atan2(dy, dx))))
        cv2.fillPoly(img, [poly(edge)], (72, 70, 66), aa, 4)
        cv2.polylines(img, [poly(edge)], True, (150, 148, 142), 1, aa, 4)
    for name, a in SECTORS:
        x, y = at(rmax * 0.96, a)
        (w, h), _ = cv2.getTextSize(name, font, 0.45, 1)
        text(name, min(max(x - w / 2, 4), size - w - 4), min(max(y + h / 2, head + h + 4), size + head - 6),
             (175, 174, 168))
    m = pr.metrics
    text(f'Perception range  {m["passed"]} of {m["checks"]} sector checks pass', 16, 24, (236, 235, 230), 0.6)
    x = 16
    for k, cl in enumerate(pr.classes):
        col = _bgr(CLASS_COLORS[k % len(CLASS_COLORS)])
        label = f'{cl.name} {pr.at(k, 0.0)[0]:.0f} m ahead'
        cv2.rectangle(img, (x, 38), (x + 10, 48), col, -1)
        text(label, x + 15, 48, (220, 219, 214) if k == k_sel else (165, 164, 160), 0.45)
        x += cv2.getTextSize(label, font, 0.45, 1)[0][0] + 32
    text(f'dashed: required range for {cl_sel.name.lower()}', size - 260, size + head - 10, (140, 140, 135), 0.4)
    ok, buf = cv2.imencode('.png', img)
    return buf.tobytes()
