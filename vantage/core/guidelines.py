"""Parking guidelines and the swept path: what a surround-view display draws over the camera images while parking.

Vehicle model: kinematic bicycle about the rear axle. The road-wheel angle δ (degrees, positive = turning left,
counter-clockwise seen from above) turns the rear-axle centre on a circle of radius L / tan δ about a centre on the
rear-axle line; every point of the body turns about the same centre. Travel s is positive forward, negative in reverse.

Guidelines (ground polylines, world frame, metres):

* static: the vehicle's width lines straight back (or ahead) from the bumper and distance markers across them at the
  configured distances; they do not move with the steering.
* trajectory: the paths of the two bumper corners (body width, no mirrors, plus a margin) for the current steering
  angle over the path length, coloured by distance band (red, yellow, green), with distance ticks across them at the
  same travel distances.
* swept path: the ground area the whole body outline (with the mirrors) sweeps over that travel, as one polygon; an
  obstacle inside it would be hit (``hits``).

Projection into a camera honours the resolved lens (unit calibrations included), the image circle and body
occlusion, and densifies the lines first so they bend correctly on fisheye images. Pixel coordinates are SENSOR pixel
coordinates (pixel centre = i + 0.5), as in the remap tables.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict

import numpy as np

from .analysis import camera_frames
from .lens import resolve

SCHEMA = 'vantage.guidelines/1'
COLORS = {'red': '#e5463a', 'yellow': '#f2c230', 'green': '#3cc55a', 'static': '#8fd3f4', 'swept': '#ff8a3d'}
BANDS = ('red', 'yellow', 'green')
PIXEL_TEXT = ('SENSOR pixel coordinates: x right, y down, origin at the top-left corner of the top-left pixel, pixel '
              'centre = i + 0.5 (subtract 0.5 for OpenCV pixel coordinates); scale by image_width / sensor_width for '
              'other resolutions')
DIRECTIONS = ('reverse', 'forward')


# ====================================================================== settings
@dataclass
class GuidelineSettings:
    angle_deg: float = 0.0          # current road-wheel angle, + = turning left
    max_deg: float = 35.0           # road-wheel angle at full lock
    step_deg: float = 2.5           # angle step of the exported tables
    ratio: float = 15.0             # steering ratio (steering-wheel ° per road-wheel °); 0 = not shown
    direction: str = 'reverse'      # reverse | forward
    distances: list = field(default_factory=lambda: [0.5, 1.0, 2.0, 3.0])   # from the bumper, metres
    path_m: float = 3.0             # length of the trajectory lines
    margin_m: float = 0.0           # added to the body width on each side
    static: bool = True
    dynamic: bool = True
    swept: bool = True

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d) -> 'GuidelineSettings':
        s = GuidelineSettings()
        for k, v in (d or {}).items():
            if k not in GuidelineSettings.__dataclass_fields__:
                continue
            try:
                if k in ('static', 'dynamic', 'swept'):
                    setattr(s, k, bool(v))
                elif k == 'direction':
                    if v in DIRECTIONS:
                        s.direction = v
                elif k == 'distances':
                    ds = sorted({round(float(x), 4) for x in v if math.isfinite(float(x)) and 0 < float(x) <= 50})
                    if ds:
                        s.distances = ds
                else:
                    f = float(v)
                    if math.isfinite(f):
                        setattr(s, k, f)
            except (TypeError, ValueError):
                pass
        s.max_deg = min(max(s.max_deg, 1.0), 60.0)
        s.angle_deg = min(max(s.angle_deg, -s.max_deg), s.max_deg)
        s.step_deg = min(max(s.step_deg, 0.1), s.max_deg)
        s.ratio = max(s.ratio, 0.0)
        s.path_m = min(max(s.path_m, 0.2), 30.0)
        s.margin_m = min(max(s.margin_m, 0.0), 1.0)
        return s

    @property
    def length(self) -> float:
        """Trajectory length: at least as far as the last distance marker."""
        return max(self.path_m, max(self.distances))

    def band(self, d: float) -> str:
        """Colour band of a travel distance d (metres from the bumper): red up to the first marker, yellow up to the
        second, green beyond."""
        ds = self.distances
        for i, edge in enumerate(ds[:2]):
            if d <= edge + 1e-9:
                return BANDS[i]
        return 'green'


# ====================================================================== vehicle
@dataclass
class VehicleOutline:
    wheelbase: float
    track: float
    rear_axle_x: float              # x of the rear axle (0 with the default origin)
    front_x: float                  # front of the front bumper
    rear_x: float                   # back of the rear bumper
    half_width: float               # body, without the mirrors
    centre_y: float                 # body centreline
    hull: np.ndarray                # (K, 2) convex hull of everything (mirrors included), counter-clockwise
    body_hull: np.ndarray           # (K, 2) without the mirrors
    mirror_z: float | None = None   # lowest point of the mirrors (an obstacle lower than this passes under them)
    mirrors: list = field(default_factory=list)     # (K, 2) hull of each mirror

    @property
    def footprint(self) -> list:
        """Polygons whose union is the ground footprint: the body and each mirror on its own (one hull around both
        would fill in the space behind the mirrors)."""
        return [self.body_hull] + self.mirrors if self.mirrors else [self.hull]

    def corner(self, direction: str, side: int, margin=0.0) -> np.ndarray:
        """Bumper corner on the side (+1 left, -1 right) at the end facing the direction of travel."""
        x = self.rear_x if direction == 'reverse' else self.front_x
        return np.array([x, self.centre_y + side * (self.half_width + margin)])


def _hull(xy: np.ndarray) -> np.ndarray:
    xy = np.unique(np.round(np.asarray(xy, float), 3), axis=0)
    if len(xy) < 3:
        lo, hi = xy.min(axis=0), xy.max(axis=0)
        return np.array([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]])
    try:
        from scipy.spatial import ConvexHull
        h = ConvexHull(xy)
        return xy[h.vertices]               # counter-clockwise in 2D
    except Exception:
        lo, hi = xy.min(axis=0), xy.max(axis=0)
        return np.array([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]])


def vehicle_outline(project) -> VehicleOutline:
    veh = project.vehicle
    wc = veh.wheel_centres or {}
    m = veh.mesh
    spec = veh.spec or {}
    if m is not None and m.n_tris:
        keep = np.array([p.role != 'ignored' for p in m.parts], bool)[m.face_part]
        mirror = np.array(['mirror' in (p.name or '').lower() for p in m.parts], bool)[m.face_part]
        tri = m.tri[keep] if keep.any() else m.tri
        mir = mirror[keep] if keep.any() else mirror
        V = tri.reshape(-1, 3)
        lo, hi = V.min(axis=0), V.max(axis=0)
        body = tri[~mir].reshape(-1, 3) if (~mir).any() else V
        mirror_z = float(tri[mir][:, :, 2].min()) if mir.any() else None
        if not mir.any():
            # one-part meshes carry the mirrors in the body: take the width below the belt line
            low = body[body[:, 2] <= lo[2] + 0.55 * (hi[2] - lo[2])]
            if len(low) > 10:
                body = low
        blo, bhi = body.min(axis=0), body.max(axis=0)
        half_w = float(bhi[1] - blo[1]) / 2
        cy = float(bhi[1] + blo[1]) / 2
        front_x, rear_x = float(hi[0]), float(lo[0])
        step = max(1, len(V) // 40000)
        hull = _hull(V[::step, :2])
        bstep = max(1, len(body) // 40000)
        body_hull = _hull(body[::bstep, :2])
        mirrors = []
        if mir.any():
            mv = tri[mir].reshape(-1, 3)
            for side in (mv[:, 1] >= cy, mv[:, 1] < cy):
                if side.sum() >= 3:
                    mirrors.append(_hull(mv[side, :2]))
    else:
        L = float(spec.get('Length', 4.0) or 4.0)
        W = float(spec.get('Width', 1.8) or 1.8)
        rear_x, front_x = -0.8, -0.8 + L
        half_w, cy, mirror_z, mirrors = W / 2, 0.0, None, []
        hull = body_hull = np.array([[rear_x, -half_w], [front_x, -half_w], [front_x, half_w], [rear_x, half_w]])
    if 'FL' in wc and 'RL' in wc:
        wheelbase = abs(float(wc['FL'][0]) - float(wc['RL'][0]))
    else:
        wheelbase = float(spec.get('Wheelbase', 0) or 0) or 0.62 * (front_x - rear_x)
    rx = [float(wc[k][0]) for k in ('RL', 'RR') if k in wc]
    rear_axle_x = float(np.mean(rx)) if rx else 0.0
    tr = [abs(float(wc[a][1]) - float(wc[b][1])) for a, b in (('FL', 'FR'), ('RL', 'RR')) if a in wc and b in wc]
    track = float(np.mean(tr)) if tr else 2 * half_w - 0.25
    return VehicleOutline(max(wheelbase, 0.5), track, rear_axle_x, front_x, rear_x, half_w, cy, hull, body_hull, mirror_z,
                          mirrors)


# ====================================================================== kinematics
def turn_radius(outline: VehicleOutline, delta_deg: float) -> float:
    """Signed radius of the rear-axle centre's path (positive = centre on the left); inf when straight."""
    t = math.tan(math.radians(delta_deg))
    return outline.wheelbase / t if abs(t) > 1e-9 else math.inf


def poses(outline: VehicleOutline, delta_deg: float, s) -> tuple[np.ndarray, np.ndarray]:
    """Rotations (N, 2, 2) and translations (N, 2) taking body ground points at the start to travel s (N,)."""
    s = np.atleast_1d(np.asarray(s, float))
    R = turn_radius(outline, delta_deg)
    if not math.isfinite(R):
        rot = np.repeat(np.eye(2)[None], len(s), axis=0)
        return rot, np.stack([s, np.zeros_like(s)], axis=1)
    th = s / R
    c, si = np.cos(th), np.sin(th)
    rot = np.stack([np.stack([c, -si], 1), np.stack([si, c], 1)], 1)
    C = np.array([outline.rear_axle_x, R])
    t = C - rot @ C
    return rot, t


def trace(outline: VehicleOutline, delta_deg: float, p0, s) -> np.ndarray:
    """Ground path (N, 2) of the body point p0 over travel s."""
    rot, t = poses(outline, delta_deg, s)
    return rot @ np.asarray(p0, float) + t


def _travel(settings: GuidelineSettings, direction: str, length=None, step=0.02) -> np.ndarray:
    L = settings.length if length is None else length
    n = max(2, int(math.ceil(L / step)) + 1)
    s = np.linspace(0.0, L, n)
    return -s if direction == 'reverse' else s


# ====================================================================== guideline set
@dataclass
class GuideLine:
    id: str
    kind: str                       # static | trajectory | distance | swept
    color: str                      # band name (see COLORS)
    label: str
    pts: np.ndarray                 # (N, 2) ground, densified
    dash: bool = False
    closed: bool = False

    @property
    def hex(self) -> str:
        return COLORS.get(self.color, self.color)


@dataclass
class GuidelineSet:
    direction: str
    angle_deg: float
    radius: float                   # rear-axle centre path radius (inf when straight)
    lines: list                     # [GuideLine]
    swept: np.ndarray | None        # (K, 2) swept-path polygon or None
    hits: list                      # [(obstacle, travel m)] obstacles inside the swept path, nearest first
    settings: GuidelineSettings


def densify(pts, step=0.03) -> np.ndarray:
    """The polyline with extra points so no two neighbours are further apart than step."""
    pts = np.asarray(pts, float)
    if len(pts) < 2:
        return pts
    seg = np.diff(pts, axis=0)
    n = np.maximum(1, np.ceil(np.linalg.norm(seg, axis=1) / step).astype(int))
    k = np.repeat(np.arange(len(seg)), n)
    f = (np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n) + 1) / np.repeat(n, n)
    return np.vstack([pts[:1], pts[k] + seg[k] * f[:, None]])


def _split_bands(settings, pts, s, kind, ident, dash=False):
    """Cut a polyline sampled at travel distances |s| into band-coloured pieces."""
    d = np.abs(s)
    ds = settings.distances
    edges = [0.0] + [e for e in ds[:2] if e < d[-1]] + [d[-1]]
    out = []
    for k, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
        sel = (d >= a - 1e-9) & (d <= b + 1e-9)
        if sel.sum() >= 2:
            out.append(GuideLine(f'{ident}_{k}', kind, settings.band(b), '', pts[sel], dash))
    return out


def guideline_set(project_or_outline, settings: GuidelineSettings | None = None, direction=None, angle_deg=None,
                  obstacles=None, swept=None) -> GuidelineSet:
    """The guidelines for one direction and road-wheel angle (the settings' own when not given)."""
    o = project_or_outline if isinstance(project_or_outline, VehicleOutline) else vehicle_outline(project_or_outline)
    if obstacles is None and not isinstance(project_or_outline, VehicleOutline):
        obstacles = project_or_outline.obstacles
    st = settings or GuidelineSettings()
    direction = direction or st.direction
    angle = st.angle_deg if angle_deg is None else float(angle_deg)
    sgn = -1.0 if direction == 'reverse' else 1.0
    lines = []
    m = st.margin_m
    if st.static:
        for side, name in ((1, 'left'), (-1, 'right')):
            p0 = o.corner(direction, side, m)
            p1 = p0 + np.array([sgn * st.length, 0.0])
            lines.append(GuideLine(f'static_{name}', 'static', 'static', '', densify([p0, p1]), dash=True))
        for i, d in enumerate(st.distances):
            a, b = o.corner(direction, 1, m), o.corner(direction, -1, m)
            off = np.array([sgn * d, 0.0])
            lines.append(GuideLine(f'static_d{i}', 'static', 'static', f'{d:g} m', densify([a + off, b + off]), dash=True))
    if st.dynamic:
        s = _travel(st, direction)
        for side, name in ((1, 'left'), (-1, 'right')):
            pts = trace(o, angle, o.corner(direction, side, m), s)
            lines += _split_bands(st, pts, s, 'trajectory', f'trajectory_{name}')
        for i, d in enumerate(st.distances):
            if d > st.length + 1e-9:
                continue
            rot, t = poses(o, angle, [sgn * d])
            a = rot[0] @ o.corner(direction, 1, m) + t[0]
            b = rot[0] @ o.corner(direction, -1, m) + t[0]
            lines.append(GuideLine(f'distance_{i}', 'distance', st.band(d), f'{d:g} m', densify([a, b])))
    poly = None
    hits = []
    if (st.swept if swept is None else swept):
        poly = swept_path(o, angle, direction, st.length)
        lines.append(GuideLine('swept', 'swept', 'swept', '', densify(np.vstack([poly, poly[:1]])), closed=True))
        hits = obstacle_hits(o, angle, direction, st.length, obstacles or [])
    return GuidelineSet(direction, angle, turn_radius(o, angle), lines, poly, hits, st)


# ====================================================================== swept path
def swept_path(outline: VehicleOutline, delta_deg: float, direction: str, length: float, res=0.02) -> np.ndarray:
    """Outer boundary (K, 2) of the ground the body outline (with the mirrors) sweeps over the travel, counter-clockwise.
    Rasterised at res and traced, so it is exact to about one cell."""
    import cv2
    R = turn_radius(outline, delta_deg)
    turn = abs(length / R) if math.isfinite(R) else 0.0
    n = max(2, int(math.ceil(length / 0.04)), int(math.ceil(math.degrees(turn) / 0.75))) + 1
    s = np.linspace(0.0, length, n) * (-1.0 if direction == 'reverse' else 1.0)
    rot, t = poses(outline, delta_deg, s)
    polys = [np.einsum('nij,kj->nki', rot, part) + t[:, None, :] for part in outline.footprint]   # each (n, K, 2)
    allp = np.concatenate([q.reshape(-1, 2) for q in polys])
    lo = allp.min(axis=0) - 3 * res
    hi = allp.max(axis=0) + 3 * res
    W, H = (np.ceil((hi - lo) / res).astype(int) + 1)
    mask = np.zeros((H, W), np.uint8)
    shift = 4
    # poses 4 cm / 0.75° apart overlap, so the union of their outlines is the swept area (the scallops between them
    # are far below a cell)
    for q in polys:
        px = np.round((q - lo) / res * (1 << shift)).astype(np.int32)
        cv2.fillPoly(mask, list(px), 255, lineType=cv2.LINE_8, shift=shift)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cnts, key=cv2.contourArea)
    c = cv2.approxPolyDP(c, 0.5, True).reshape(-1, 2).astype(float)
    poly = lo + c * res                     # pixel i is centred on lo + i * res (as rasterised above)
    if _area(poly) < 0:
        poly = poly[::-1]
    return poly


def _area(poly) -> float:
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def point_in_polygon(pts, poly) -> np.ndarray:
    """Even-odd test for points (N, 2) in a simple polygon (K, 2)."""
    pts = np.atleast_2d(np.asarray(pts, float))
    x, y = pts[:, 0][:, None], pts[:, 1][:, None]
    a, b = poly, np.roll(poly, -1, axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        cross = ((a[:, 1] > y) != (b[:, 1] > y)) & (x < (b[:, 0] - a[:, 0]) * (y - a[:, 1]) / (b[:, 1] - a[:, 1]) + a[:, 0])
    return cross.sum(axis=1) % 2 == 1


def _poly_distance(pts, poly) -> np.ndarray:
    """Distance from points (N, 2) to the polygon outline (0 inside counts as inside: negative)."""
    a, b = poly, np.roll(poly, -1, axis=0)
    ab = b - a
    ap = pts[:, None, :] - a[None]
    t = np.clip((ap * ab).sum(-1) / np.maximum((ab * ab).sum(-1), 1e-12), 0, 1)
    d = np.linalg.norm(ap - t[..., None] * ab, axis=-1).min(axis=1)
    return np.where(point_in_polygon(pts, poly), -d, d)


def obstacle_hits(outline: VehicleOutline, delta_deg: float, direction: str, length: float, obstacles) -> list:
    """[(obstacle, travel metres)] for every obstacle the body would touch within the travel, nearest first. Obstacles
    lower than the mirrors only meet the body outline without them."""
    if not obstacles:
        return []
    s = np.linspace(0.0, length, max(2, int(math.ceil(length / 0.01)) + 1))
    rot, t = poses(outline, delta_deg, s * (-1.0 if direction == 'reverse' else 1.0))
    out = []
    for ob in obstacles:
        parts = outline.footprint if (outline.mirror_z is None or ob.height > outline.mirror_z) else [outline.body_hull]
        q = np.array([ob.x, ob.y], float)
        local = np.einsum('nji,nj->ni', rot, q - t)          # inverse rotation of (q - t)
        d = np.min([_poly_distance(local, part) for part in parts], axis=0)
        inside = np.nonzero(d <= ob.radius)[0]
        if len(inside):
            out.append((ob, float(s[inside[0]])))
    out.sort(key=lambda x: x[1])
    return out


def triangulate(poly) -> np.ndarray:
    """Ear clipping of a simple polygon (K, 2): triangle indices (K-2, 3)."""
    poly = np.asarray(poly, float)
    n = len(poly)
    idx = list(range(n)) if _area(poly) > 0 else list(range(n))[::-1]
    tris = []

    def convex(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]) > 1e-12

    guard = 0
    while len(idx) > 3 and guard < 4 * n * n:
        guard += 1
        m = len(idx)
        for k in range(m):
            i0, i1, i2 = idx[k - 1], idx[k], idx[(k + 1) % m]
            a, b, c = poly[i0], poly[i1], poly[i2]
            if not convex(a, b, c):
                continue
            others = [j for j in idx if j not in (i0, i1, i2)]
            if others:
                P = poly[others]
                d1 = (b[0] - a[0]) * (P[:, 1] - a[1]) - (b[1] - a[1]) * (P[:, 0] - a[0])
                d2 = (c[0] - b[0]) * (P[:, 1] - b[1]) - (c[1] - b[1]) * (P[:, 0] - b[0])
                d3 = (a[0] - c[0]) * (P[:, 1] - c[1]) - (a[1] - c[1]) * (P[:, 0] - c[0])
                if ((d1 >= 0) & (d2 >= 0) & (d3 >= 0)).any():
                    continue
            tris.append((i0, i1, i2))
            idx.pop(k)
            break
        else:                               # degenerate rest: fan it
            tris += [(idx[0], idx[j], idx[j + 1]) for j in range(1, len(idx) - 1)]
            idx = []
            break
    if len(idx) == 3:
        tris.append(tuple(idx))
    return np.array(tris, np.int64).reshape(-1, 3)


# ====================================================================== projection
@dataclass
class CamProjection:
    """Ground lines through one camera: per line, the visible pieces as sensor pixel polylines (and their ground
    points)."""
    width: int
    height: int
    pieces: list                    # [(GuideLine, [(pixels (M, 2), ground (M, 2))])]


def project_ground(project, engine, cam, lines, z=0.0, step=0.03) -> CamProjection:
    """Visible pieces of ground polylines (GuideLines or (N, 2) arrays) in this camera's image."""
    intr = resolve(project.module_for(cam), project.lens_for(cam))
    _, W2O = camera_frames(cam)
    pos = cam.pos()
    cube = None
    if engine is not None and engine.vehicle_geom(project) is not None:
        cube = engine.cube(pos)
    out = []
    jump = 0.25 * max(intr.width, intr.height)
    for ln in lines:
        pts = ln.pts if isinstance(ln, GuideLine) else np.asarray(ln, float)
        P2 = densify(pts, step)
        if len(P2) < 2:
            out.append((ln, []))
            continue
        P = np.c_[P2, np.full(len(P2), z)]
        dv = P - pos
        dist = np.linalg.norm(dv, axis=1)
        u, v, _, ok = intr.project_dirs(dv @ W2O.T)
        if cube is not None:
            bd, _ = cube.lookup(dv)
            ok &= bd > dist - 0.03
        pieces = []
        idx = np.nonzero(ok)[0]
        if len(idx) >= 2:
            # a new piece where the line was hidden or left the image, or jumps across it (behind the lens)
            brk = np.ones(len(idx), bool)
            brk[1:] = (np.diff(idx) != 1) | (np.abs(np.diff(u[idx])) + np.abs(np.diff(v[idx])) >= jump)
            starts = np.nonzero(brk)[0]
            for a, b in zip(starts, np.r_[starts[1:], len(idx)]):
                if b - a >= 2:
                    sl = idx[a:b]
                    pieces.append((np.c_[u[sl], v[sl]], P2[sl]))
        out.append((ln, pieces))
    return CamProjection(intr.width, intr.height, out)


def simplify(pts, tol=0.3) -> np.ndarray:
    """Douglas-Peucker on a polyline (N, 2)."""
    pts = np.asarray(pts, float)
    if len(pts) < 3:
        return pts
    keep = np.zeros(len(pts), bool)
    keep[[0, -1]] = True
    stack = [(0, len(pts) - 1)]
    while stack:
        a, b = stack.pop()
        if b - a < 2:
            continue
        p, q = pts[a], pts[b]
        seg = q - p
        L = np.hypot(*seg)
        r = pts[a + 1:b] - p
        d = np.abs(seg[0] * r[:, 1] - seg[1] * r[:, 0]) / L if L > 1e-12 else np.hypot(r[:, 0], r[:, 1])
        k = int(np.argmax(d))
        if d[k] > tol:
            keep[a + 1 + k] = True
            stack += [(a, a + 1 + k), (a + 1 + k, b)]
    return pts[keep]


# ====================================================================== tables for the ECU
def _angles(st: GuidelineSettings, angles=None) -> list:
    if angles is not None:
        return [float(a) for a in angles]
    n = int(math.floor(st.max_deg / st.step_deg + 1e-9))
    a = [k * st.step_deg for k in range(-n, n + 1)]
    if abs(a[-1] - st.max_deg) > 1e-9:
        a = [-st.max_deg] + a + [st.max_deg]
    return [round(x, 6) + 0.0 for x in a]


def _line_doc(ln: GuideLine, pieces, origin, scale, dp):
    ground = simplify(ln.pts, 0.005)

    def num(v):
        r = round(float(v) * scale, dp) + 0.0
        return int(r) if dp == 0 else r
    return {'id': ln.id, 'kind': ln.kind, 'color': ln.hex, 'band': ln.color, 'label': ln.label, 'dashed': ln.dash,
            'closed': ln.closed,
            'pixels': [[[round(float(x), 1), round(float(y), 1)] for x, y in simplify(px, 0.3)] for px, _ in pieces],
            'ground': [[num(x - origin[0]), num(y - origin[1])] for x, y in ground]}


def visible_directions(project, engine, cam, settings=None) -> list:
    """The directions whose straight-ahead guidelines this camera sees at all."""
    st = settings or GuidelineSettings.from_dict(getattr(project.analysis, 'guidelines', None))
    o = vehicle_outline(project)
    out = []
    for d in DIRECTIONS:
        gs = guideline_set(o, st, d, 0.0, obstacles=[], swept=False)
        pr = project_ground(project, engine, cam, gs.lines)
        if any(p for _, p in pr.pieces):
            out.append(d)
    return out


def guideline_table(project, engine, cam, angles=None, settings=None, directions=None, origin=(0.0, 0.0, 0.0),
                    scale=1.0, dp=3) -> dict:
    """Per direction and road-wheel angle, the guidelines in this camera's sensor pixels (visible pieces), with
    colours, labels and the ground polylines (shifted by origin and scaled, for the export frame). Static lines are
    listed once per direction."""
    st = settings or GuidelineSettings.from_dict(getattr(project.analysis, 'guidelines', None))
    o = vehicle_outline(project)
    dirs = list(directions) if directions is not None else visible_directions(project, engine, cam, st)
    angs = _angles(st, angles)
    doc = {'directions': {}, 'static': {}}
    dyn = GuidelineSettings.from_dict({**st.to_dict(), 'static': False})
    sta = GuidelineSettings.from_dict({**st.to_dict(), 'dynamic': False, 'swept': False})
    for d in dirs:
        gs = guideline_set(o, sta, d, 0.0, obstacles=[], swept=False)
        pr = project_ground(project, engine, cam, gs.lines)
        doc['static'][d] = [_line_doc(ln, pc, origin, scale, dp) for ln, pc in pr.pieces]
        rows = []
        for a in angs:
            gs = guideline_set(o, dyn, d, a, obstacles=[])
            pr = project_ground(project, engine, cam, gs.lines)
            R = gs.radius
            rows.append({'road_wheel_angle_deg': a,
                         'steering_wheel_deg': round(a * st.ratio, 2) + 0.0 if st.ratio > 0 else None,
                         'turn_radius_m': round(abs(R), 3) if math.isfinite(R) else None,
                         'lines': [_line_doc(ln, pc, origin, scale, dp) for ln, pc in pr.pieces]})
        doc['directions'][d] = rows
    return doc


# ====================================================================== preview image
def _bgr(hexs):
    h = hexs.lstrip('#')
    return int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16)


def _dashes(px, on=10.0, off=6.0):
    """Dash pieces of a pixel polyline."""
    seg = np.diff(px, axis=0)
    L = np.r_[0.0, np.cumsum(np.hypot(seg[:, 0], seg[:, 1]))]
    out = []
    t = 0.0
    while t < L[-1]:
        a, b = t, min(t + on, L[-1])
        s = np.linspace(a, b, max(2, int((b - a) / 2) + 2))
        out.append(np.c_[np.interp(s, L, px[:, 0]), np.interp(s, L, px[:, 1])])
        t += on + off
    return out


def preview_png(project, engine, cam, settings=None, directions=None, world=None, width=640) -> bytes:
    """PNG of the guidelines (current angle, the given directions) over the synthetic parking scene through this
    camera, as a display would show them."""
    import cv2
    from .synth import parking_world, render_capture
    st = settings or GuidelineSettings.from_dict(getattr(project.analysis, 'guidelines', None))
    intr = resolve(project.module_for(cam), project.lens_for(cam))
    w = int(width)
    h = int(round(w * intr.height / intr.width))
    img = render_capture(project, engine, cam, w, h, world=world or parking_world(project), seed=cam.slot, noise=1.0)
    img = cv2.cvtColor(np.ascontiguousarray(img, np.uint8), cv2.COLOR_RGB2BGR)
    sx, sy = w / intr.width, h / intr.height
    dirs = list(directions) if directions is not None else visible_directions(project, engine, cam, st)
    o = vehicle_outline(project)
    font = cv2.FONT_HERSHEY_SIMPLEX
    notes = []
    for d in dirs:
        gs = guideline_set(o, st, d, obstacles=project.obstacles)
        pr = project_ground(project, engine, cam, gs.lines)
        polys = [(ln, np.c_[px[:, 0] * sx, px[:, 1] * sy]) for ln, pcs in pr.pieces for px, _ in pcs]
        for ln, px in polys:                # a dark underlay so the colours read on any ground
            if ln.kind in ('trajectory', 'distance'):
                cv2.polylines(img, [np.round(px * 16).astype(np.int32)], False, (0, 0, 0), 5, cv2.LINE_AA, shift=4)
        for ln, px in polys:
            thick = {'static': 1, 'trajectory': 3, 'distance': 2, 'swept': 2}.get(ln.kind, 2)
            for q in (_dashes(px) if (ln.dash or ln.kind == 'swept') else [px]):
                cv2.polylines(img, [np.round(q * 16).astype(np.int32)], False, _bgr(ln.hex), thick, cv2.LINE_AA, shift=4)
        want = 'distance' if st.dynamic else 'static'
        for ln, pcs in pr.pieces:
            if ln.kind == want and ln.label and pcs:
                u, v = pcs[0][0][0]
                at = (int(min(u * sx + 6, w - 48)), int(max(v * sy - 6, 14)))
                cv2.putText(img, ln.label, at, font, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(img, ln.label, at, font, 0.45, _bgr(ln.hex), 1, cv2.LINE_AA)
        notes += [f'{ob.name} hit after {s:.2f} m' for ob, s in gs.hits]
    a = st.angle_deg
    R = turn_radius(o, a)
    head = (f'{cam.tag} | ' + (' + '.join(dirs) if dirs else 'no guideline in view') + f' | {a:+.1f} deg road wheel'
            + (f' | radius {abs(R):.2f} m' if math.isfinite(R) else ' | straight'))
    for i, text in enumerate([head] + sorted(set(notes))):
        y = 20 + 18 * i
        (tw, th), _ = cv2.getTextSize(text, font, 0.45, 1)
        cv2.rectangle(img, (6, y - th - 5), (14 + tw, y + 5), (20, 20, 20), -1)
        cv2.putText(img, text, (10, y), font, 0.45, (240, 240, 240) if i == 0 else _bgr(COLORS['red']), 1, cv2.LINE_AA)
    ok, buf = cv2.imencode('.png', img)
    if not ok:
        raise ValueError(f'Could not encode the guideline preview of {cam.tag}.')
    return buf.tobytes()
