"""Per-camera occlusion as a cube map of depth + shade: the body and rig rasterised with QPainter, the sensor and
camera housings composited in.

Each cube face stores, per direction from the camera centre, the distance to the nearest body or rig triangle and a
flat shade value. The triangles are painted far to near by the distance of their nearest point (painter's algorithm),
each in a colour that encodes its index; the depth of a texel is then the exact distance along the ray through the texel's
centre to the plane of the triangle drawn there, t = n·(v0 − pos) / (n·d) (metres, float32; at most the triangle's
farthest corner, for the texels a polygon's one-pixel outline adds at its edges). Only the texel's size limits it,
not the triangle's: the centroid's distance, stored until 0.2.0, was 15 % off in the median and over 3× in 5 % of
directions. Rotating a camera does not change its cube map, so tolerance studies and aim changes reuse it; only a
position change rebuilds it.

Housings (core.housings: every lidar, radar, GNSS antenna and IMU housing and every other camera's module body) are
not rasterised into that static map: a cube map asked for from an occluder that carries housings (HousedOccluder) is
the static body + rig map with each housing composited in, an exact ray test of the housing's solid at the centre of
every texel inside its angular footprint (depth in metres, float, and a shade lit by LIGHT). The static map, each
housing's footprint layer and the composite are cached separately, so moving one sensor recomputes one small layer
per camera and re-merges, and never re-rasterises a body map. The housing whose anchor (a camera's lens, a sensor's
optical centre) lies at the cube's position, within OWN_TOL_M, is that cube's own and left out of it.

Body in front. Where the static map leaves it in doubt (within the occluder's largest triangle radius of the housing's
hit: the triangle the painter's order put at a texel can be a farther one near a silhouette), the texel's ray is tested
exactly against the body and rig triangles near it: a housing texel is
kept only if no triangle lies in front of the hit or less than EMBED_TOL_M behind it (a radar behind a bumper fascia,
a camera module sunk into a panel do not show; the body is kept there). Elsewhere the housing is the nearer surface.
"""
from __future__ import annotations

import math
import threading
from collections import OrderedDict

import numpy as np

FACE_N = 256
NEAR = 0.004
# (forward, right, up) per face; right x up = forward is not required, lookup uses the same basis
FACES = [
    (np.array([1.0, 0, 0]), np.array([0, -1.0, 0]), np.array([0, 0, 1.0])),
    (np.array([-1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([0, 0, 1.0])),
    (np.array([0, 1.0, 0]), np.array([1.0, 0, 0]), np.array([0, 0, 1.0])),
    (np.array([0, -1.0, 0]), np.array([-1.0, 0, 0]), np.array([0, 0, 1.0])),
    (np.array([0, 0, 1.0]), np.array([1.0, 0, 0]), np.array([0, -1.0, 0])),
    (np.array([0, 0, -1.0]), np.array([1.0, 0, 0]), np.array([0, 1.0, 0])),
]
LIGHT = np.array([0.35, 0.45, 0.82])
LIGHT = LIGHT / np.linalg.norm(LIGHT)
EMBED_TOL_M = 0.03          # a housing face less than this in front of a body surface is set into the body
OWN_TOL_M = 0.15            # a housing anchored this close to a cube's position is that cube's own


class OccluderGeom:
    """Triangles used for line of sight, prepared once per vehicle."""

    def __init__(self, tri: np.ndarray, normals: np.ndarray, glass: np.ndarray | None = None, version: int = 0):
        self.tri = np.ascontiguousarray(tri, np.float64)
        self.normals = normals
        shade = 70 + 150 * np.clip(normals @ LIGHT, 0, 1) + 25 * np.clip(normals[:, 2], 0, 1)
        if glass is not None:
            shade = np.where(glass, shade * 0.45, shade)
        self.shade = np.clip(shade, 1, 255).astype(np.int32)
        self.centroid = self.tri.mean(axis=1)
        # each triangle's plane: unit normal (zero for a degenerate triangle) and its offset n·v0
        nrm = np.cross(self.tri[:, 1] - self.tri[:, 0], self.tri[:, 2] - self.tri[:, 0])
        ln = np.linalg.norm(nrm, axis=1, keepdims=True)
        self.plane_n = np.divide(nrm, ln, out=np.zeros_like(nrm), where=ln > 1e-15)
        self.plane_d = np.einsum('ij,ij->i', self.plane_n, self.tri[:, 0])
        self.version = version


class HousedOccluder:
    """A body + rig occluder with a variant's housings (a core.housings.HousingSet): cube maps asked for from it are
    the static maps with the housings composited in. Everything else (triangles, version) is the static occluder's,
    so caches of the body alone stay valid while only a housing moves."""

    def __init__(self, base: OccluderGeom, housings):
        self.base = base
        self.housings = housings

    def __getattr__(self, name):
        if name in ('base', 'housings'):            # not set yet (a copy being made): no recursion
            raise AttributeError(name)
        return getattr(self.base, name)


def texel_index(dirs: np.ndarray, n: int):
    """(face, row, column) of the texel each world direction (K, 3) falls in."""
    d = np.asarray(dirs, float)
    ad = np.abs(d)
    ax = np.argmax(ad, axis=1)
    sign = np.take_along_axis(d, ax[:, None], 1)[:, 0] >= 0
    face = ax * 2 + (~sign).astype(int)
    u = np.empty(len(d))
    v = np.empty(len(d))
    for f in range(6):
        m = face == f
        if not m.any():
            continue
        fw, rt, up = FACES[f]
        dd = d[m]
        z = dd @ fw
        u[m] = (dd @ rt) / z
        v[m] = (dd @ up) / z
    px = np.clip(((u + 1) * 0.5 * n).astype(int), 0, n - 1)
    py = np.clip(((1 - v) * 0.5 * n).astype(int), 0, n - 1)
    return face, py, px


def texel_dirs(f: int, y0: int, y1: int, x0: int, x1: int, n: int) -> np.ndarray:
    """Unit directions (rows y0..y1, columns x0..x1 inclusive, (h, w, 3)) through the texel centres of face f."""
    fw, rt, up = FACES[f]
    u = (np.arange(x0, x1 + 1) + 0.5) / n * 2 - 1
    v = 1 - (np.arange(y0, y1 + 1) + 0.5) / n * 2
    d = fw[None, None, :] + u[None, :, None] * rt[None, None, :] + v[:, None, None] * up[None, None, :]
    return d / np.linalg.norm(d, axis=2, keepdims=True)


_FACE_DIRS: dict = {}


def face_dirs(n: int) -> np.ndarray:
    """(6, n, n, 3) unit directions through every texel centre of a cube map with n × n texels per face (cached)."""
    d = _FACE_DIRS.get(n)
    if d is None:
        d = np.stack([texel_dirs(f, 0, n - 1, 0, n - 1, n) for f in range(6)])
        d.setflags(write=False)
        _FACE_DIRS[n] = d
    return d


class CubeDepth:
    def __init__(self, depth: np.ndarray, shade: np.ndarray, pos, hid: np.ndarray | None = None, housings=(),
                 base: 'CubeDepth | None' = None):
        self.depth = depth          # (6, N, N) float32 metres (inf = nothing)
        self.shade = shade          # (6, N, N) uint8 (0 = nothing)
        self.pos = np.asarray(pos, float)
        self.hid = hid              # (6, N, N) int8/16: index into housings of what is seen there, -1 body or nothing
        self.housings = list(housings)      # the housings composited in (core.housings.Housing), [] for a static map
        self.base = base            # the static body + rig map this one was composited from (None: it is static)

    @property
    def static(self) -> 'CubeDepth':
        return self.base if self.base is not None else self

    def lookup(self, dirs: np.ndarray):
        """World directions (K,3) -> (depth metres, shade)."""
        face, py, px = texel_index(dirs, self.depth.shape[1])
        return self.depth[face, py, px], self.shade[face, py, px]

    def lookup_ids(self, dirs: np.ndarray):
        """World directions (K,3) -> (depth metres, shade, housing index into self.housings or -1)."""
        face, py, px = texel_index(dirs, self.depth.shape[1])
        hid = self.hid[face, py, px] if self.hid is not None else np.full(len(face), -1, np.int16)
        return self.depth[face, py, px], self.shade[face, py, px], hid


def composite(static: CubeDepth, layers: list, housings: list) -> CubeDepth:
    """The static map with housing layers merged in. layers[k]: [(face, y0, x0, depth (h, w) float32, inf where the
    housing is missed or the body hides it, shade (h, w) uint8)] of housings[k] (core.housings.layer). The nearest
    housing wins; a housing texel replaces the body's (the layer already left out what the body hides)."""
    hd = np.full(static.depth.shape, np.inf, np.float32)
    hs = np.zeros(static.shade.shape, np.uint8)
    hid = np.full(static.depth.shape, -1, np.int8 if len(housings) < 127 else np.int16)
    for k, blocks in enumerate(layers):
        for f, y0, x0, d, s in blocks:
            h, w = d.shape
            sd = hd[f, y0:y0 + h, x0:x0 + w]
            win = d < sd
            sd[win] = d[win]
            hs[f, y0:y0 + h, x0:x0 + w][win] = s[win]
            hid[f, y0:y0 + h, x0:x0 + w][win] = k
    show = np.isfinite(hd)
    return CubeDepth(np.where(show, hd, static.depth).astype(np.float32), np.where(show, hs, static.shade),
                     static.pos, hid, housings, static)


def _clip_near(poly_cam):
    """Sutherland-Hodgman clip of a polygon (list of (x,y,z)) against z >= NEAR."""
    out = []
    n = len(poly_cam)
    for i in range(n):
        a = poly_cam[i]
        b = poly_cam[(i + 1) % n]
        ina, inb = a[2] >= NEAR, b[2] >= NEAR
        if ina:
            out.append(a)
        if ina != inb:
            t = (NEAR - a[2]) / (b[2] - a[2])
            out.append(a + (b - a) * t)
    return out


_paint_lock = threading.Lock()


def tri_distance(tri: np.ndarray, p) -> np.ndarray:
    """Distance from point p to each triangle (M, 3, 3): to its closest point, face, edge or corner (Ericson,
    Real-Time Collision Detection, 5.1.5, vectorised). A degenerate triangle gives its nearest corner's distance."""
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    p = np.asarray(p, float)
    ab, ac = b - a, c - a
    ap, bp, cp = p - a, p - b, p - c
    dot = lambda u, v: np.einsum('ij,ij->i', u, v)          # noqa: E731
    d1, d2, d3, d4, d5, d6 = dot(ab, ap), dot(ac, ap), dot(ab, bp), dot(ac, bp), dot(ab, cp), dot(ac, cp)
    vc, vb, va = d1 * d4 - d3 * d2, d5 * d2 - d1 * d6, d3 * d6 - d5 * d4
    conds = [(d1 <= 0) & (d2 <= 0), (d3 >= 0) & (d4 <= d3), (d6 >= 0) & (d5 <= d6),
             (vc <= 0) & (d1 >= 0) & (d3 <= 0), (vb <= 0) & (d2 >= 0) & (d6 <= 0),
             (va <= 0) & (d4 - d3 >= 0) & (d5 - d6 >= 0)]
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):    # (the regions not taken; degenerate ones)
        v_ab = d1 / (d1 - d3)
        w_ac = d2 / (d2 - d6)
        w_bc = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        den = 1.0 / (va + vb + vc)
        v_in, w_in = vb * den, vc * den
        choices = [a, b, c, a + v_ab[:, None] * ab, a + w_ac[:, None] * ac, b + w_bc[:, None] * (c - b)]
        q = np.select([m[:, None] for m in conds], choices, a + v_in[:, None] * ab + w_in[:, None] * ac)
        d = np.linalg.norm(q - p, axis=1)
    corner = np.linalg.norm(tri - p, axis=2).min(axis=1)
    return np.where(np.isfinite(d), np.minimum(d, corner), corner)


def build_cube(geom: OccluderGeom, pos, n: int = FACE_N) -> CubeDepth:
    """The cube map of geom's triangles seen from pos: per texel the exact distance (metres) along the texel's centre
    ray to the plane of the nearest triangle (painter's order) and its shade; inf and 0 where nothing is hit."""
    from PyQt6.QtGui import QImage, QPainter, QPolygonF, QColor, QPen
    from PyQt6.QtCore import QPointF

    pos = np.asarray(pos, float)
    rel = geom.tri - pos                     # (M,3,3)
    dist = np.linalg.norm(geom.centroid - pos, axis=1)
    # far to near (painter's algorithm) by each triangle's nearest point, ties by the centroid: a large triangle close
    # to the camera at a grazing angle is drawn over the ones behind it (by the centroid alone it often was not)
    order = np.lexsort((-dist, -tri_distance(geom.tri, pos)))
    rel = rel[order]
    shade = geom.shade[order]
    plane_n = geom.plane_n[order]
    plane_h = geom.plane_d[order] - plane_n @ pos            # n·(v0 − pos): the plane's offset from pos
    far = np.linalg.norm(rel, axis=2).max(axis=1)            # a hit on the triangle is no farther than its far corner
    cdist = np.minimum(dist[order], far)
    dirs = face_dirs(n)
    depth = np.full((6, n, n), np.inf, np.float32)
    shade_out = np.zeros((6, n, n), np.uint8)
    half = n / 2.0
    for f, (fw, rt, up) in enumerate(FACES):
        z = rel @ fw                           # (M,3)
        x = rel @ rt
        y = rel @ up
        front = (z >= NEAR).all(axis=1)
        partial = (z >= NEAR).any(axis=1) & ~front
        with np.errstate(divide='ignore', invalid='ignore'):
            px = x / z
            py = y / z
        # cull triangles fully outside the face frustum
        inside = front & ~((px > 1.02).all(1) | (px < -1.02).all(1) | (py > 1.02).all(1) | (py < -1.02).all(1))
        idx_full = np.nonzero(inside)[0]
        idx_part = np.nonzero(partial)[0]
        if len(idx_full) == 0 and len(idx_part) == 0:
            continue
        img = QImage(n, n, QImage.Format.Format_RGB32)
        img.fill(QColor(0, 0, 0))              # 0: nothing; a triangle is painted as its index + 1 (24 bits)
        sx = (px + 1) * half
        sy = (1 - py) * half
        items = []
        for i in idx_full:
            items.append((i, [(sx[i, 0], sy[i, 0]), (sx[i, 1], sy[i, 1]), (sx[i, 2], sy[i, 2])]))
        for i in idx_part:
            poly = _clip_near([np.array([x[i, k], y[i, k], z[i, k]]) for k in range(3)])
            if len(poly) >= 3:
                items.append((i, [((p[0] / p[2] + 1) * half, (1 - p[1] / p[2]) * half) for p in poly]))
        items.sort(key=lambda it: it[0])        # keep far-to-near order
        with _paint_lock:
            p = QPainter(img)
            p.setRenderHint(QPainter.RenderHint.Antialiasing, False)
            for i, pts in items:
                k = int(i) + 1
                col = QColor((k >> 16) & 255, (k >> 8) & 255, k & 255)
                p.setBrush(col)
                p.setPen(QPen(col, 1.0))
                p.drawPolygon(QPolygonF([QPointF(a, b) for a, b in pts]))
            p.end()
        ptr = img.constBits()
        ptr.setsize(img.sizeInBytes())
        arr = np.frombuffer(ptr, np.uint8).reshape(n, img.bytesPerLine() // 4, 4)[:, :n]
        ids = (arr[..., 2].astype(np.int64) << 16) | (arr[..., 1].astype(np.int64) << 8) | arr[..., 0]
        hit = ids > 0
        if not hit.any():
            continue
        tid = ids[hit] - 1
        den = np.einsum('ij,ij->i', plane_n[tid], dirs[f][hit])
        with np.errstate(divide='ignore', invalid='ignore'):
            t = plane_h[tid] / den
        # the plane behind the ray or parallel to it (an outline texel past a grazing triangle's edge): the
        # centroid's distance; never beyond the triangle's far corner
        t = np.where(np.isfinite(t) & (t > 0), t, cdist[tid])
        depth[f][hit] = np.minimum(t, far[tid])
        shade_out[f][hit] = shade[tid]
    return CubeDepth(depth, shade_out, pos)


class CubeCache:
    """LRU of cube maps keyed by (geometry version, position rounded to 1 mm).

    For a HousedOccluder the static body + rig map is kept as before and the housings are composited into it: each
    housing's footprint layer is cached on (static map, housing), each composite on (static map, the housings in it),
    so an unchanged camera with unchanged housings costs a dictionary lookup."""

    LAYERS = 384
    COMPOSITES = 32

    def __init__(self, size=64):
        self.size = size
        self._d = {}
        self._order = []
        self._lock = threading.Lock()
        self._layers = OrderedDict()        # (static key, housing key) -> blocks
        self._comp = OrderedDict()          # (static key, housing keys) -> CubeDepth

    @staticmethod
    def _key(geom, pos):
        return (geom.version, tuple(np.round(np.asarray(pos, float), 3)))

    def peek(self, geom, pos) -> CubeDepth | None:
        """The cube map for this position if its body map is already built (never rasterises one; housings are
        composited in, which is cheap)."""
        base = getattr(geom, 'base', geom)
        key = self._key(base, pos)
        with self._lock:
            cube = self._d.get(key)
        if cube is None:
            return None
        return self._housed(geom, key, cube)

    def get(self, geom, pos) -> CubeDepth:
        base = getattr(geom, 'base', geom)
        key = self._key(base, pos)
        cube = None
        with self._lock:
            if key in self._d:
                self._order.remove(key)
                self._order.append(key)
                cube = self._d[key]
        if cube is None:
            cube = build_cube(base, pos)
            with self._lock:
                if key in self._d:                  # another thread built it meanwhile: keep one entry
                    cube = self._d[key]
                else:
                    self._d[key] = cube
                    self._order.append(key)
                    while len(self._order) > self.size:
                        old = self._order.pop(0)
                        self._d.pop(old, None)
        return self._housed(geom, key, cube)

    def _housed(self, geom, key, cube: CubeDepth) -> CubeDepth:
        """The static map with the occluder's housings (all but the cube's own) composited in."""
        hs = getattr(geom, 'housings', None)
        if hs is None:
            return cube
        inc = hs.around(cube.pos)
        if not inc:
            return cube
        n = cube.depth.shape[1]
        sol = [hs.solids[i] for i in inc]
        ckey = (key, n, tuple(h.key for h in sol))
        with self._lock:
            hit = self._comp.get(ckey)
            if hit is not None:
                self._comp.move_to_end(ckey)
                return hit
        layers = []
        for h in sol:
            lkey = (key, n, h.key)
            with self._lock:
                blocks = self._layers.get(lkey)
                if blocks is not None:
                    self._layers.move_to_end(lkey)
            if blocks is None:
                blocks = hs.layer(h, cube, getattr(geom, 'base', geom))
                with self._lock:
                    self._layers[lkey] = blocks
                    while len(self._layers) > self.LAYERS:
                        self._layers.popitem(last=False)
            layers.append(blocks)
        comp = composite(cube, layers, sol)
        with self._lock:
            if ckey in self._comp:
                return self._comp[ckey]
            self._comp[ckey] = comp
            while len(self._comp) > self.COMPOSITES:
                self._comp.popitem(last=False)
        return comp
