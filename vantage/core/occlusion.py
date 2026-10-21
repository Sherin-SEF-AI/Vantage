"""Per-camera body occlusion as a cube map of depth + shade, rasterised with QPainter.

Each cube face stores, per direction from the camera centre, the distance to the nearest
body triangle (mm, 16 bit) and a flat shade value. Rotating a camera does not change its
cube map, so tolerance studies and aim changes reuse it; only a position change rebuilds it.
"""
from __future__ import annotations

import math
import threading

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
        self.version = version


class CubeDepth:
    def __init__(self, depth: np.ndarray, shade: np.ndarray, pos):
        self.depth = depth          # (6, N, N) float32 metres (inf = nothing)
        self.shade = shade          # (6, N, N) uint8 (0 = nothing)
        self.pos = np.asarray(pos, float)

    def lookup(self, dirs: np.ndarray):
        """World directions (K,3) -> (depth metres, shade)."""
        d = np.asarray(dirs, float)
        ad = np.abs(d)
        ax = np.argmax(ad, axis=1)
        sign = np.take_along_axis(d, ax[:, None], 1)[:, 0] >= 0
        face = ax * 2 + (~sign).astype(int)
        N = self.depth.shape[1]
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
        px = np.clip(((u + 1) * 0.5 * N).astype(int), 0, N - 1)
        py = np.clip(((1 - v) * 0.5 * N).astype(int), 0, N - 1)
        return self.depth[face, py, px], self.shade[face, py, px]


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


def build_cube(geom: OccluderGeom, pos, n: int = FACE_N) -> CubeDepth:
    from PyQt6.QtGui import QImage, QPainter, QPolygonF, QColor, QPen
    from PyQt6.QtCore import QPointF, Qt

    pos = np.asarray(pos, float)
    rel = geom.tri - pos                     # (M,3,3)
    dist = np.linalg.norm(geom.centroid - pos, axis=1)
    order = np.argsort(-dist)                # far to near (painter's algorithm)
    rel = rel[order]
    dist_mm = np.clip(dist[order] * 1000.0, 0, 65534).astype(np.int64)
    shade = geom.shade[order]
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
        img.fill(QColor(0, 255, 255))
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
                dm = int(dist_mm[i])
                col = QColor(int(shade[i]), dm >> 8, dm & 255)
                p.setBrush(col)
                p.setPen(QPen(col, 1.0))
                p.drawPolygon(QPolygonF([QPointF(a, b) for a, b in pts]))
            p.end()
        ptr = img.constBits()
        ptr.setsize(img.sizeInBytes())
        arr = np.frombuffer(ptr, np.uint8).reshape(n, img.bytesPerLine() // 4, 4)[:, :n]
        b, g, r = arr[..., 0].astype(np.int32), arr[..., 1].astype(np.int32), arr[..., 2]
        dmm = g * 256 + b
        hit = dmm < 65535
        depth[f] = np.where(hit, dmm / 1000.0, np.inf)
        shade_out[f] = np.where(hit, r, 0)
    return CubeDepth(depth, shade_out, pos)


class CubeCache:
    """LRU of cube maps keyed by (geometry version, position rounded to 1 mm)."""

    def __init__(self, size=64):
        self.size = size
        self._d = {}
        self._order = []
        self._lock = threading.Lock()

    def get(self, geom: OccluderGeom, pos) -> CubeDepth:
        key = (geom.version, tuple(np.round(np.asarray(pos, float), 3)))
        with self._lock:
            if key in self._d:
                self._order.remove(key)
                self._order.append(key)
                return self._d[key]
        cube = build_cube(geom, pos)
        with self._lock:
            self._d[key] = cube
            self._order.append(key)
            while len(self._order) > self.size:
                old = self._order.pop(0)
                self._d.pop(old, None)
        return cube
