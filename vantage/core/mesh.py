"""Triangle mesh with parts and mount zones, plus the geometric queries the planner needs."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

ROLE_MOUNTABLE = 'occluder · mountable'
ROLE_GLASS = 'occluder · not mountable'
ROLE_WHEELS = 'wheels'
ROLE_IGNORED = 'ignored'
ROLES = (ROLE_MOUNTABLE, ROLE_GLASS, ROLE_WHEELS, ROLE_IGNORED)


@dataclass
class Part:
    name: str
    material: str = ''
    alpha: float = 1.0
    color: tuple = (0.8, 0.8, 0.8)
    role: str = ROLE_MOUNTABLE

    def to_dict(self):
        return {'name': self.name, 'material': self.material, 'alpha': self.alpha,
                'color': list(self.color), 'role': self.role}

    @staticmethod
    def from_dict(d):
        return Part(d.get('name', 'Part'), d.get('material', ''), float(d.get('alpha', 1.0)),
                    tuple(d.get('color', (0.8, 0.8, 0.8))), d.get('role', ROLE_MOUNTABLE))


@dataclass
class Zone:
    name: str
    kind: str = 'allowed'          # 'allowed' | 'forbidden'
    auto: bool = False

    def to_dict(self):
        return {'name': self.name, 'kind': self.kind, 'auto': self.auto}

    @staticmethod
    def from_dict(d):
        return Zone(d.get('name', 'Zone'), d.get('kind', 'allowed'), bool(d.get('auto', False)))


class Mesh:
    def __init__(self, vertices, faces, face_part=None, parts=None, face_zone=None, zones=None):
        self.vertices = np.ascontiguousarray(vertices, dtype=np.float64)
        self.faces = np.ascontiguousarray(faces, dtype=np.int64).reshape(-1, 3)
        m = len(self.faces)
        self.face_part = np.zeros(m, np.int32) if face_part is None else np.asarray(face_part, np.int32)
        self.parts: list[Part] = parts if parts else [Part('Body')]
        self.face_zone = np.full(m, -1, np.int16) if face_zone is None else np.asarray(face_zone, np.int16)
        self.zones: list[Zone] = zones if zones is not None else []
        self._cache = {}

    # ------------------------------------------------------------ basics
    def invalidate(self):
        self._cache.clear()

    @property
    def n_tris(self):
        return len(self.faces)

    @property
    def tri(self) -> np.ndarray:
        if 'tri' not in self._cache:
            self._cache['tri'] = self.vertices[self.faces]
        return self._cache['tri']

    @property
    def normals(self) -> np.ndarray:
        if 'n' not in self._cache:
            t = self.tri
            n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
            a = np.linalg.norm(n, axis=1)
            self._cache['area'] = a / 2
            self._cache['n'] = n / np.where(a < 1e-20, 1.0, a)[:, None]
        return self._cache['n']

    @property
    def areas(self) -> np.ndarray:
        self.normals
        return self._cache['area']

    @property
    def centroids(self) -> np.ndarray:
        if 'c' not in self._cache:
            self._cache['c'] = self.tri.mean(axis=1)
        return self._cache['c']

    @property
    def bbox(self):
        if len(self.vertices) == 0:
            return np.zeros(3), np.zeros(3)
        if 'bb' not in self._cache:
            used = self.vertices[np.unique(self.faces)] if len(self.faces) else self.vertices
            self._cache['bb'] = (used.min(axis=0), used.max(axis=0))
        return self._cache['bb']

    def copy(self) -> 'Mesh':
        return Mesh(self.vertices.copy(), self.faces.copy(), self.face_part.copy(),
                    [Part(**p.__dict__) for p in self.parts], self.face_zone.copy(),
                    [Zone(**z.__dict__) for z in self.zones])

    def transformed(self, R=None, t=None, scale=1.0) -> 'Mesh':
        v = self.vertices * scale
        if R is not None:
            v = v @ np.asarray(R).T
        if t is not None:
            v = v + np.asarray(t)
        m = self.copy()
        m.vertices = v
        if R is not None and np.linalg.det(R) < 0:
            m.faces = m.faces[:, ::-1].copy()
        m.invalidate()
        return m

    def subset(self, face_mask) -> 'Mesh':
        idx = np.nonzero(face_mask)[0]
        f = self.faces[idx]
        used, inv = np.unique(f, return_inverse=True)
        return Mesh(self.vertices[used], inv.reshape(-1, 3), self.face_part[idx], self.parts,
                    self.face_zone[idx], self.zones)

    def part_mask(self, roles) -> np.ndarray:
        roles = set(roles)
        ok = np.array([p.role in roles for p in self.parts] or [True], bool)
        return ok[self.face_part]

    def occluder_faces(self) -> np.ndarray:
        return self.part_mask({ROLE_MOUNTABLE, ROLE_GLASS, ROLE_WHEELS})

    # ------------------------------------------------------------ decimation
    def decimate(self, target_tris: int) -> tuple['Mesh', float]:
        """Vertex-clustering simplification to roughly target_tris triangles.
        Returns (mesh, max deviation estimate in model units)."""
        if self.n_tris <= target_tris:
            return self.copy(), 0.0
        lo_b, hi_b = self.bbox
        diag = float(np.linalg.norm(hi_b - lo_b)) or 1.0
        lo, hi = diag * 1e-4, diag * 0.2
        best = None
        for _ in range(22):
            h = math.sqrt(lo * hi)
            m = self._cluster(h)
            if m.n_tris > target_tris:
                lo = h
            else:
                hi = h
                best = (m, h)
            if best and abs(best[0].n_tris - target_tris) < 0.03 * target_tris:
                break
        if best is None:
            best = (self._cluster(hi), hi)
        m, h = best
        return m, h * math.sqrt(3) / 2

    def _cluster(self, h: float) -> 'Mesh':
        lo = self.vertices.min(axis=0)
        key = np.floor((self.vertices - lo) / h).astype(np.int64)
        key = key[:, 0] * 73856093 ^ key[:, 1] * 19349663 ^ key[:, 2] * 83492791
        uniq, inv = np.unique(key, return_inverse=True)
        cnt = np.bincount(inv, minlength=len(uniq)).astype(float)
        nv = np.stack([np.bincount(inv, weights=self.vertices[:, k], minlength=len(uniq)) / cnt
                       for k in range(3)], axis=1)
        f = inv[self.faces]
        ok = (f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])
        f = f[ok]
        fp, fz = self.face_part[ok], self.face_zone[ok]
        # drop duplicates (same three vertices regardless of order)
        s = np.sort(f, axis=1)
        _, first = np.unique(s[:, 0] * 1_000_003 ** 2 + s[:, 1] * 1_000_003 + s[:, 2], return_index=True)
        first.sort()
        return Mesh(nv, f[first], fp[first], self.parts, fz[first], self.zones)

    # ------------------------------------------------------------ ray casting
    def _mask_key(self, faces_mask):
        """Cache key for a face mask by its contents (callers pass fresh arrays, so id() would leak and go stale)."""
        if faces_mask is None:
            return None
        import hashlib
        m = np.asarray(faces_mask)
        return hashlib.blake2b(np.packbits(m.astype(bool)).tobytes() if m.dtype == bool else m.tobytes(),
                               digest_size=16).digest()

    def _mt(self, faces_mask=None):
        key = ('mt', self._mask_key(faces_mask))
        if key not in self._cache:
            t = self.tri if faces_mask is None else self.tri[faces_mask]
            idx = np.arange(self.n_tris) if faces_mask is None else np.nonzero(faces_mask)[0]
            self._cache[key] = (t[:, 0].copy(), (t[:, 1] - t[:, 0]).copy(), (t[:, 2] - t[:, 0]).copy(), idx)
        return self._cache[key]

    def raycast(self, origins, dirs, faces_mask=None, max_t=np.inf):
        """Nearest hits for rays (K,3). Returns (t (K,), face index (K,), -1 for misses)."""
        O = np.atleast_2d(np.asarray(origins, float))
        D = np.atleast_2d(np.asarray(dirs, float))
        v0, e1, e2, idx = self._mt(faces_mask)
        M = len(v0)
        K = len(O)
        best_t = np.full(K, np.inf)
        best_f = np.full(K, -1, np.int64)
        if M == 0:
            return best_t, best_f
        chunk = max(1, int(3_000_000 // max(M, 1)))
        for s in range(0, K, chunk):
            o = O[s:s + chunk, None, :]
            d = D[s:s + chunk, None, :]
            p = np.cross(d, e2[None])
            det = np.einsum('kmj,mj->km', p, e1)
            inv = 1.0 / np.where(np.abs(det) < 1e-12, np.nan, det)
            tv = o - v0[None]
            u = np.einsum('kmj,kmj->km', tv, p) * inv
            q = np.cross(tv, e1[None])
            v = np.einsum('kmj,kmj->km', d, q) * inv
            t = np.einsum('mj,kmj->km', e2, q) * inv
            ok = (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9) & (t > 1e-6) & (t < max_t)
            t = np.where(ok, t, np.inf)
            j = np.argmin(t, axis=1)
            tt = t[np.arange(len(j)), j]
            hit = np.isfinite(tt)
            best_t[s:s + chunk] = tt
            best_f[s:s + chunk] = np.where(hit, idx[j], -1)
        return best_t, best_f

    # ------------------------------------------------------------ closest point
    def closest_point(self, p, faces_mask=None):
        """Closest point on the surface to p. Returns (point, face index, distance)."""
        p = np.asarray(p, float)
        tri = self.tri if faces_mask is None else self.tri[faces_mask]
        idx = np.arange(self.n_tris) if faces_mask is None else np.nonzero(faces_mask)[0]
        if len(tri) == 0:
            return p.copy(), -1, math.inf
        # cull far triangles first
        c = tri.mean(axis=1)
        r = np.linalg.norm(tri - c[:, None], axis=2).max(axis=1)
        lower = np.linalg.norm(c - p, axis=1) - r            # no point of triangle i is closer than this
        upper = np.linalg.norm(tri - p, axis=2).min()          # some vertex is at least this close
        keep = lower <= upper + 1e-12
        tri, idx = tri[keep], idx[keep]
        q = closest_on_triangles(p, tri)
        d = np.linalg.norm(q - p, axis=1)
        j = int(np.argmin(d))
        return q[j], int(idx[j]), float(d[j])

    # ------------------------------------------------------------ edges
    def crease_segments(self, angle_deg=35.0, faces_mask=None) -> np.ndarray:
        key = ('crease', angle_deg, self._mask_key(faces_mask))
        if key in self._cache:
            return self._cache[key]
        f = self.faces if faces_mask is None else self.faces[faces_mask]
        n = self.normals if faces_mask is None else self.normals[faces_mask]
        e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
        fi = np.tile(np.arange(len(f)), 3)
        es = np.sort(e, axis=1)
        k = es[:, 0] * (len(self.vertices) + 1) + es[:, 1]
        order = np.argsort(k, kind='stable')
        ks, fs, eo = k[order], fi[order], es[order]
        same = np.nonzero(ks[1:] == ks[:-1])[0]
        cosang = np.einsum('ij,ij->i', n[fs[same]], n[fs[same + 1]])
        sharp = same[cosang < math.cos(math.radians(angle_deg))]
        # boundary edges: appear once
        cnt_first = np.ones(len(ks), bool)
        cnt_first[same] = False
        cnt_first[same + 1] = False
        segs = np.concatenate([eo[sharp], eo[cnt_first]])
        out = self.vertices[segs] if len(segs) else np.zeros((0, 2, 3))
        self._cache[key] = out
        return out

    # ------------------------------------------------------------ rasterised plan masks
    def plan_mask(self, x0, y0, cell, nx, ny, faces_mask=None) -> np.ndarray:
        """Boolean (ny, nx) mask of grid cells covered by the mesh seen from above."""
        from PyQt6.QtGui import QImage, QPainter, QPolygonF, QColor, QPen
        from PyQt6.QtCore import QPointF, Qt
        tri = self.tri if faces_mask is None else self.tri[faces_mask]
        img = QImage(nx, ny, QImage.Format.Format_Grayscale8)
        img.fill(0)
        p = QPainter(img)
        # a 1 px pen marks partially covered cells as covered (ground under a body edge is not visible)
        p.setPen(QPen(QColor(255, 255, 255), 1.0))
        p.setBrush(QColor(255, 255, 255))
        px = (tri[:, :, 0] - x0) / cell
        py = (tri[:, :, 1] - y0) / cell
        for a, b in zip(px.tolist(), py.tolist()):
            p.drawPolygon(QPolygonF([QPointF(a[0], b[0]), QPointF(a[1], b[1]), QPointF(a[2], b[2])]))
        p.end()
        ptr = img.constBits()
        ptr.setsize(img.sizeInBytes())
        arr = np.frombuffer(ptr, np.uint8).reshape(ny, img.bytesPerLine())[:, :nx]
        return arr > 127


def closest_on_triangles(p, tri):
    """Vectorised closest point on each triangle (M,3,3) to point p (Ericson, RTCD 5.1.5)."""
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    ab, ac, ap = b - a, c - a, p - a
    d1 = np.einsum('ij,ij->i', ab, ap)
    d2 = np.einsum('ij,ij->i', ac, ap)
    bp = p - b
    d3 = np.einsum('ij,ij->i', ab, bp)
    d4 = np.einsum('ij,ij->i', ac, bp)
    cp = p - c
    d5 = np.einsum('ij,ij->i', ab, cp)
    d6 = np.einsum('ij,ij->i', ac, cp)
    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2
    out = np.empty_like(a)
    done = np.zeros(len(a), bool)

    def put(mask, val):
        nonlocal done
        m = mask & ~done
        out[m] = val[m] if np.ndim(val) == 2 else val
        done |= m

    put((d1 <= 0) & (d2 <= 0), a)
    put((d3 >= 0) & (d4 <= d3), b)
    with np.errstate(divide='ignore', invalid='ignore'):
        v = d1 / (d1 - d3)
        put((vc <= 0) & (d1 >= 0) & (d3 <= 0), a + v[:, None] * ab)
        put((d6 >= 0) & (d5 <= d6), c)
        w = d2 / (d2 - d6)
        put((vb <= 0) & (d2 >= 0) & (d6 <= 0), a + w[:, None] * ac)
        w2 = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        put((va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0), b + w2[:, None] * (c - b))
        denom = 1.0 / (va + vb + vc)
        vv = vb * denom
        ww = vc * denom
        put(np.ones(len(a), bool), a + ab * vv[:, None] + ac * ww[:, None])
    bad = ~np.isfinite(out).all(axis=1)
    out[bad] = a[bad]
    return out
