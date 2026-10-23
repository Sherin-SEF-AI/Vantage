"""Implicit solids and a watertight surface mesher for printable parts (core.brackets), in numpy.

A solid is a signed distance function f(P) over points P (N, 3): negative inside, zero on the surface, positive
outside, with an axis-aligned bounding box. The primitives are exact distances (boxes, rods, rounded rectangles) or
lower bounds of the distance (a convex hull's planes, the 45° countersink cone, the combinations), never more than
the true distance, so a ray can step by the value without passing through a surface (raytrace).

The mesher samples f on a regular grid (only near the surface: a coarse pass finds where it can be) and meshes it
cube by cube from a table derived from marching tetrahedra: every grid cube is split into the same six tetrahedra
around its main diagonal, so neighbouring cubes split their shared face along the same diagonal, and the linear
interpolation of f in the tetrahedra gives a surface that is a closed 2-manifold with no ambiguous cases. Merging
each cube's pieces into polygons over the cube's edges keeps every face's trace the same for both cubes on it, so
the result stays closed and manifold (every edge in exactly two triangles) at about a third of the triangles.
Vertices sit on the grid edges, each at least 2 % of the edge away from a grid point, so no triangle degenerates,
even after rounding to the float32 of an STL. Triangles are wound outward (from inside corners towards outside).
"""
from __future__ import annotations

import math
import struct

import numpy as np

PAD = 1.5                 # mm: a cut is evaluated only near its bounding box (enough for the mesher's band)
_T_CLAMP = 0.02           # interpolated vertices stay this share of an edge away from its grid points


# =========================================================================== solids
class Solid:
    """A signed distance function with its bounding box (lo, hi), mm."""
    __slots__ = ('f', 'lo', 'hi')

    def __init__(self, f, lo, hi):
        self.f = f
        self.lo = np.asarray(lo, float)
        self.hi = np.asarray(hi, float)

    def __call__(self, P):
        return self.f(np.asarray(P, float).reshape(-1, 3))


def _unit(v):
    v = np.asarray(v, float)
    return v / (np.linalg.norm(v) or 1.0)


def _box_bounds(c, A, half):
    ext = np.abs(np.asarray(A, float)) @ np.asarray(half, float)
    return c - ext, c + ext


def box(centre, axes, half, r=0.0) -> Solid:
    """Box with half sizes `half` along the columns of `axes` (orthonormal), every edge rounded by r (exact)."""
    c, A, h = np.asarray(centre, float), np.asarray(axes, float), np.asarray(half, float)
    r = float(min(max(r, 0.0), h.min() * 0.999))
    hh = h - r

    def f(P):
        q = np.abs((P - c) @ A) - hh
        return np.linalg.norm(np.maximum(q, 0.0), axis=1) + np.minimum(q.max(axis=1), 0.0) - r
    return Solid(f, *_box_bounds(c, A, h))


def round_rect(centre, axes, half, corner_r=0.0) -> Solid:
    """Box whose edges along its third axis are rounded by corner_r (a rounded rectangle extruded): flat top and
    bottom faces (a base plate that sits fully on a print bed). Exact."""
    c, A, h = np.asarray(centre, float), np.asarray(axes, float), np.asarray(half, float)
    r = float(min(max(corner_r, 0.0), h[:2].min() * 0.999))

    def f(P):
        q = (P - c) @ A
        d = np.abs(q[:, :2]) - (h[:2] - r)
        d2 = np.linalg.norm(np.maximum(d, 0.0), axis=1) + np.minimum(d.max(axis=1), 0.0) - r
        dz = np.abs(q[:, 2]) - h[2]
        w0, w1 = np.maximum(d2, 0.0), np.maximum(dz, 0.0)
        return np.hypot(w0, w1) + np.minimum(np.maximum(d2, dz), 0.0)
    return Solid(f, *_box_bounds(c, A, h))


def hex_prism(centre, axes, flats, half_h) -> Solid:
    """Regular hexagonal prism (a nut pocket): across-flats width `flats` in the plane of the first two axes, half
    height half_h along the third. Exact."""
    c, A = np.asarray(centre, float), np.asarray(axes, float)
    r = flats / 2
    kx, ky, kz = -0.8660254037844386, 0.5, 0.5773502691896258

    def f(P):
        q = (P - c) @ A
        x, y = np.abs(q[:, 0]), np.abs(q[:, 1])
        d = np.minimum(kx * x + ky * y, 0.0)
        x, y = x - 2 * d * kx, y - 2 * d * ky
        x = x - np.clip(x, -kz * r, kz * r)
        y = y - r
        d2 = np.hypot(x, y) * np.sign(y)
        dz = np.abs(q[:, 2]) - half_h
        return np.hypot(np.maximum(d2, 0.0), np.maximum(dz, 0.0)) + np.minimum(np.maximum(d2, dz), 0.0)
    R = r / 0.8660254037844386
    return Solid(f, *_box_bounds(c, A, np.array([R, R, half_h])))


def rod(p0, p1, r) -> Solid:
    """Cylinder of radius r from p0 to p1 with flat ends (exact)."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    L = float(np.linalg.norm(p1 - p0)) or 1e-9
    a = (p1 - p0) / L
    c = (p0 + p1) / 2

    def f(P):
        d = P - c
        t = d @ a
        rad = np.sqrt(np.maximum(np.einsum('ij,ij->i', d, d) - t * t, 0.0))
        q0, q1 = rad - r, np.abs(t) - L / 2
        return np.hypot(np.maximum(q0, 0.0), np.maximum(q1, 0.0)) + np.minimum(np.maximum(q0, q1), 0.0)
    ext = np.abs(a) * L / 2 + r * np.sqrt(np.maximum(1 - a * a, 0.0))
    return Solid(f, c - ext, c + ext)


def countersink(top, axis, r_top, above=0.5) -> Solid:
    """90° countersink cone: radius r_top where it meets the surface at `top`, narrowing by 1 mm per mm into the
    part (against `axis`, the surface's outward normal), reaching `above` mm out of the surface. Lower bound."""
    top, a = np.asarray(top, float), _unit(axis)
    apex = top - a * r_top
    L = r_top + above

    def f(P):
        d = P - apex
        t = d @ a
        rad = np.sqrt(np.maximum(np.einsum('ij,ij->i', d, d) - t * t, 0.0))
        return np.maximum(np.maximum((rad - t) / math.sqrt(2.0), t - L), -t)
    ext = np.full(3, r_top + above) + np.abs(a) * L / 2
    mid = apex + a * L / 2
    return Solid(f, mid - ext, mid + ext)


def halfspace(point, normal) -> Solid:
    """Everything on the side of the plane through point against the normal (exact)."""
    p, n = np.asarray(point, float), _unit(normal)
    return Solid(lambda P: (P - p) @ n, np.full(3, -np.inf), np.full(3, np.inf))


def hull(points) -> Solid:
    """Convex hull of the points: the largest of its face planes' distances (exact inside, a lower bound outside)."""
    from scipy.spatial import ConvexHull
    pts = np.asarray(points, float)
    eq = ConvexHull(pts, qhull_options='QJ').equations        # n · x + d <= 0 inside, n unit
    eq = np.unique(np.round(eq, 9), axis=0)                    # one plane per face (qhull gives triangles)
    N, d = eq[:, :3].copy(), eq[:, 3].copy()
    return Solid(lambda P: (P @ N.T + d).max(axis=1), pts.min(axis=0), pts.max(axis=0))


def union(*solids) -> Solid:
    solids = [s for s in solids if s is not None]

    def f(P):
        out = solids[0].f(P)
        for s in solids[1:]:
            out = np.minimum(out, s.f(P))
        return out
    return Solid(f, np.min([s.lo for s in solids], axis=0), np.max([s.hi for s in solids], axis=0))


def round_union(a: Solid, b: Solid, r: float) -> Solid:
    """Union with a fillet of radius r where the surfaces meet (circular for perpendicular faces)."""
    if r <= 0:
        return union(a, b)

    def f(P):
        fa, fb = a.f(P), b.f(P)
        u0, u1 = np.maximum(r - fa, 0.0), np.maximum(r - fb, 0.0)
        return np.maximum(r, np.minimum(fa, fb)) - np.hypot(u0, u1)
    return Solid(f, np.minimum(a.lo, b.lo) - r, np.maximum(a.hi, b.hi) + r)


def intersect(*solids) -> Solid:
    def f(P):
        out = solids[0].f(P)
        for s in solids[1:]:
            out = np.maximum(out, s.f(P))
        return out
    lo = np.max([s.lo for s in solids], axis=0)
    hi = np.min([s.hi for s in solids], axis=0)
    return Solid(f, lo, hi)


def subtract(a: Solid, *cuts) -> Solid:
    """a with the cuts removed. A cut is evaluated only within PAD mm of its bounding box: farther out it cannot
    change the sign, and the value stays a lower bound of the distance."""
    cuts = [c for c in cuts if c is not None]

    def f(P):
        out = a.f(P)
        for c in cuts:
            m = np.all((P >= c.lo - PAD) & (P <= c.hi + PAD), axis=1)
            if m.all():
                out = np.maximum(out, -c.f(P))
            elif m.any():
                out[m] = np.maximum(out[m], -c.f(P[m]))
        return out
    return Solid(f, a.lo, a.hi)


def translated(s: Solid, t) -> Solid:
    t = np.asarray(t, float)
    return Solid(lambda P: s.f(P - t), s.lo + t, s.hi + t)


# =========================================================================== mesher
# cube corners (dx, dy, dz) and the six tetrahedra around the diagonal 0-6 (Kuhn split: the same in every cube)
_CORNERS = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]])
_TETS = np.array([[0, 1, 2, 6], [0, 1, 5, 6], [0, 3, 2, 6], [0, 3, 7, 6], [0, 4, 5, 6], [0, 4, 7, 6]])


def _case_table():
    """For each 4-bit inside mask of a tetrahedron: its triangles as triples of edges (pairs of corner indices)."""
    table = {}
    for code in range(1, 15):
        ins = [i for i in range(4) if code >> i & 1]
        out = [i for i in range(4) if not code >> i & 1]
        if len(ins) == 1 or len(out) == 1:
            lone = ins[0] if len(ins) == 1 else out[0]
            others = [i for i in range(4) if i != lone]
            table[code] = [[(lone, others[0]), (lone, others[1]), (lone, others[2])]]
        else:
            a, b = ins
            c, d = out
            table[code] = [[(a, c), (a, d), (b, d)], [(a, c), (b, d), (b, c)]]
    return table


_CASES = _case_table()
_EDGE_PAIRS = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 3), (1, 2), (4, 7), (5, 6), (0, 4), (1, 5), (3, 7), (2, 6)]


def _cube_table():
    """Marching-cubes triangles derived from the tetrahedra: for each 8-bit inside mask of a cube, the surface the six
    tetrahedra give, with the vertices on the face and body diagonals dropped. On each cube face the surface's trace
    then becomes straight segments between edge vertices; the face's own tetrahedron diagonal decides an ambiguous
    face, and the neighbouring cube sees the same face, so the pieces still meet edge to edge. Each boundary cycle
    is closed as its own disc (where the tetrahedra joined two cycles through the cube's inside, the disc pair pinches
    that thin link, a choice marching cubes makes too): a fan from a corner of the cycle whose two faces the cycle
    crosses once (so no fan edge can lie in a face a neighbour also fans), else from an added centre vertex.
    Returns {mask: [(a, b, c)]} with vertices as cube edge indices (0-11) or ('c', (edge indices of its cycle))."""
    C = _CORNERS.astype(float)
    eid = {p: i for i, p in enumerate(_EDGE_PAIRS)}

    def faces_of(e):
        i, j = _EDGE_PAIRS[e]
        return {(k, int(C[i][k])) for k in range(3) if C[i][k] == C[j][k]}
    table = {}
    for code in range(1, 255):
        ins = [code >> i & 1 for i in range(8)]
        tris = []
        for tet in _TETS:
            tc = sum(ins[v] << k for k, v in enumerate(tet))
            if tc in (0, 15):
                continue
            pin = np.mean([C[v] for v in tet if ins[v]], axis=0)
            pout = np.mean([C[v] for v in tet if not ins[v]], axis=0)
            for tri in _CASES[tc]:
                labs = [tuple(sorted((int(tet[a]), int(tet[b])))) for a, b in tri]
                P = [(C[a] + C[b]) / 2 for a, b in labs]
                if np.cross(P[1] - P[0], P[2] - P[0]) @ (pout - pin) < 0:
                    labs = labs[::-1]
                tris.append(labs)
        count = {}
        for t in tris:
            for k in range(3):
                e = frozenset((t[k], t[(k + 1) % 3]))
                count[e] = count.get(e, 0) + 1
        nxt = {}
        for t in tris:
            for k in range(3):
                a, b = t[k], t[(k + 1) % 3]
                if count[frozenset((a, b))] == 1:
                    nxt[a] = b
        out, seen = [], set()
        for start in nxt:
            if start in seen:
                continue
            cyc, x = [], start
            while x not in seen:
                seen.add(x)
                cyc.append(x)
                x = nxt[x]
            cyc = [eid[v] for v in cyc if v in eid]                 # drop the diagonals' vertices
            m = len(cyc)
            if m == 3:
                out.append(tuple(cyc))
                continue
            seg = []
            for i in range(m):
                f = faces_of(cyc[i]) & faces_of(cyc[(i + 1) % m])
                seg.append(next(iter(f)))
            apex = next((i for i in range(m) if seg.count(seg[i - 1]) == 1 and seg.count(seg[i]) == 1), None)
            if apex is not None:
                cyc = cyc[apex:] + cyc[:apex]
                out += [(cyc[0], cyc[i], cyc[i + 1]) for i in range(1, m - 1)]
            else:
                c = ('c', tuple(cyc))
                out += [(c, cyc[i], cyc[(i + 1) % m]) for i in range(m)]
        table[code] = out
    return table


_CUBE = _cube_table()


def _eval_points(solid, xs, ys, zs, ii, jj, kk, out, progress=None, cancel=None, frac=(0.0, 1.0), chunk=1_000_000):
    """out[ii, jj, kk] = f at those grid points, in chunks. False when cancelled."""
    n = len(ii)
    for s in range(0, n, chunk):
        if cancel is not None and cancel():
            return False
        e = min(s + chunk, n)
        P = np.column_stack([xs[ii[s:e]], ys[jj[s:e]], zs[kk[s:e]]])
        out[ii[s:e], jj[s:e], kk[s:e]] = solid.f(P)
        if progress is not None:
            progress(frac[0] + (frac[1] - frac[0]) * e / max(n, 1))
    return True


def grid_eval(solid: Solid, lo, hi, h, progress=None, cancel=None, frac=(0.0, 0.6), coarse=4, lip=1.5):
    """f sampled on the grid lo + h·(i, j, k) covering [lo, hi]: (values float32 (nx, ny, nz), origin), or (None, lo)
    when cancelled. Only near the surface: f is first sampled every `coarse` points, and a block of the grid is
    sampled in full only where it can hold the surface (a corner's |f| within lip × (half the block's diagonal + 1.5
    steps), lip bounding how fast f changes: 1 for a distance, √2 for a fillet); elsewhere the block keeps its
    corners' sign (the values there never place a vertex)."""
    lo = np.asarray(lo, float)
    n = np.maximum(np.ceil((np.asarray(hi, float) - lo) / h).astype(int) + 1, 2)
    nx, ny, nz = (int(v) for v in n)
    xs, ys, zs = lo[0] + h * np.arange(nx), lo[1] + h * np.arange(ny), lo[2] + h * np.arange(nz)
    out = np.empty((nx, ny, nz), np.float32)
    k = max(int(coarse), 1)
    ci = [np.unique(np.r_[np.arange(0, m, k), m - 1]) for m in (nx, ny, nz)]
    if k == 1 or min(len(c) for c in ci) < 3:
        I, J, K = (g.ravel() for g in np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing='ij'))
        return (out if _eval_points(solid, xs, ys, zs, I, J, K, out, progress, cancel, frac) else None), lo
    I, J, K = (g.ravel() for g in np.meshgrid(*ci, indexing='ij'))
    mid = frac[0] + 0.15 * (frac[1] - frac[0])
    if not _eval_points(solid, xs, ys, zs, I, J, K, out, progress, cancel, (frac[0], mid)):
        return None, lo
    vc = out[np.ix_(*ci)]
    sx, sy, sz = (c.size - 1 for c in ci)
    corners = [vc[dx:dx + sx, dy:dy + sy, dz:dz + sz] for dx, dy, dz in _CORNERS]
    cmin = np.minimum.reduce(corners)
    cmax = np.maximum.reduce(corners)
    amin = np.minimum.reduce([np.abs(c) for c in corners])
    span = np.stack(np.meshgrid(*(np.diff(c) for c in ci), indexing='ij'), axis=-1) * h
    # a block is far when no point of it is within 1.5 grid steps of the surface: its filled values then never
    # take part in placing a vertex
    thr = lip * (0.5 * np.linalg.norm(span, axis=-1) + 1.5 * h)
    near = ((cmin < 0) & (cmax >= 0)) | (amin <= thr)
    sel = np.ix_(*(np.minimum(np.arange(m) // k, c.size - 2) for m, c in zip((nx, ny, nz), ci)))
    out[...] = np.where(cmax < 0, -thr, thr).astype(np.float32)[sel]
    mask = near[sel]
    fi, fj, fk = np.nonzero(mask)
    del mask
    if not _eval_points(solid, xs, ys, zs, fi, fj, fk, out, progress, cancel, (mid, frac[1])):
        return None, lo
    return out, lo


def march(vals, origin, h, progress=None, cancel=None, frac=(0.6, 1.0)):
    """The zero surface of the sampled grid as triangles (V (n, 3), F (m, 3) int64), wound outward, from the cube
    table (_cube_table). The solid must not touch the grid's border (its samples there positive) for it to close."""
    nx, ny, nz = vals.shape
    inside = vals < 0
    code = np.zeros((nx - 1, ny - 1, nz - 1), np.uint8)
    for k, (dx, dy, dz) in enumerate(_CORNERS):
        code |= inside[dx:nx - 1 + dx, dy:ny - 1 + dy, dz:nz - 1 + dz].astype(np.uint8) << k
    act = (code != 0) & (code != 255)
    ci, cj, ck = np.nonzero(act)
    codes = code[ci, cj, ck]
    del code, act
    base = (ci.astype(np.int64) * ny + cj) * nz + ck
    off = (_CORNERS[:, 0] * ny + _CORNERS[:, 1]) * nz + _CORNERS[:, 2]
    flat_v = vals.ravel()
    NP = np.int64(nx * ny * nz)
    CBASE = NP * NP                              # keys of the added centre vertices sit above every edge key

    def coords(ids):
        return np.stack([origin[0] + h * (ids // (ny * nz)), origin[1] + h * ((ids // nz) % ny),
                         origin[2] + h * (ids % nz)], axis=-1)

    def edge_keys(b, e):
        i, j = _EDGE_PAIRS[e]
        a_, c_ = b + off[i], b + off[j]
        return np.minimum(a_, c_) * NP + np.maximum(a_, c_)

    def edge_pos(keys):
        lo_, hi_ = keys // NP, keys % NP
        fa, fb = flat_v[lo_].astype(float), flat_v[hi_].astype(float)
        t = np.clip(fa / (fa - fb), _T_CLAMP, 1 - _T_CLAMP)
        pa, pb = coords(lo_), coords(hi_)
        return pa + t[:, None] * (pb - pa)
    tris, ckeys, cpos = [], [], []
    order = np.argsort(codes, kind='stable')
    bounds = np.searchsorted(codes[order], np.arange(257))
    cube_id = np.arange(len(base), dtype=np.int64)
    done = 0
    for cval in range(1, 255):
        sel = order[bounds[cval]:bounds[cval + 1]]
        if not len(sel):
            continue
        if cancel is not None and cancel():
            return None, None
        b = base[sel]
        centres = {}
        for tri in _CUBE[cval]:
            cols = []
            for lab in tri:
                if isinstance(lab, tuple):
                    if lab not in centres:
                        k = len(centres)
                        key = CBASE + cube_id[sel] * 8 + k
                        pos = np.mean([edge_pos(edge_keys(b, e)) for e in lab[1]], axis=0)
                        centres[lab] = key
                        ckeys.append(key)
                        cpos.append(pos)
                    cols.append(centres[lab])
                else:
                    cols.append(edge_keys(b, lab))
            tris.append(np.stack(cols, axis=1))
        done += len(sel)
        if progress is not None:
            progress(frac[0] + (frac[1] - frac[0]) * done / max(len(base), 1))
    if not tris:
        return np.zeros((0, 3)), np.zeros((0, 3), np.int64)
    K = np.concatenate(tris)
    uk, inv = np.unique(K.ravel(), return_inverse=True)
    V = np.empty((len(uk), 3))
    is_e = uk < CBASE
    V[is_e] = edge_pos(uk[is_e])
    if ckeys:
        ck_all, cp_all = np.concatenate(ckeys), np.concatenate(cpos)
        o = np.argsort(ck_all)
        V[~is_e] = cp_all[o][np.searchsorted(ck_all[o], uk[~is_e])]
    return V, inv.reshape(-1, 3).astype(np.int64)


def mesh(solid: Solid, h: float, lo=None, hi=None, progress=None, cancel=None):
    """The solid's surface as a closed triangle mesh (V mm, F), sampled every h mm. (None, None) when cancelled."""
    lo = np.asarray(solid.lo if lo is None else lo, float) - 3 * h
    hi = np.asarray(solid.hi if hi is None else hi, float) + 3 * h
    if not (np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))):
        raise ValueError('the solid has no finite bounding box')
    # coarse blocks no bigger than PAD: a block that holds part of a cut's surface then has its corners within the
    # cut's evaluated zone, so the band cannot miss a hole
    k = max(1, min(4, int((PAD - 0.05) / h)))
    vals, origin = grid_eval(solid, lo, hi, h, progress, cancel, coarse=k)
    if vals is None:
        return None, None
    # the border must be outside for the surface to close (the bounds hold the solid, but be certain)
    for sl in (np.s_[0], np.s_[-1]):
        vals[sl] = np.maximum(vals[sl], h)
        vals[:, sl] = np.maximum(vals[:, sl], h)
        vals[:, :, sl] = np.maximum(vals[:, :, sl], h)
    return march(vals, origin, h, progress, cancel)


# =========================================================================== mesh checks
def check(V, F) -> dict:
    """Is the mesh a closed, consistently wound 2-manifold? edges: every undirected edge in exactly two triangles
    (watertight, manifold edges), each directed edge once and its reverse once (consistent winding), no zero-area
    triangle, the enclosed volume positive (outward winding); shells (connected pieces) and the Euler genus."""
    V, F = np.asarray(V, float), np.asarray(F, np.int64)
    nv, nf = len(V), len(F)
    if nf == 0:
        return {'triangles': 0, 'vertices': nv, 'watertight': False, 'consistent': False, 'manifold_edges': False,
                'degenerate': 0, 'volume_mm3': 0.0, 'area_mm2': 0.0, 'shells': 0, 'genus': 0, 'ok': False}
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    kd = e[:, 0] * nv + e[:, 1]
    und = np.sort(e, axis=1)
    ku = und[:, 0] * nv + und[:, 1]
    _, cu = np.unique(ku, return_counts=True)
    _, cd = np.unique(kd, return_counts=True)
    manifold = bool(np.all(cu == 2))
    # every edge twice and no directed edge twice: its two uses run opposite ways
    consistent = bool(manifold and np.all(cd == 1))
    tri = V[F]
    cr = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = np.linalg.norm(cr, axis=1) / 2
    vol = float(np.einsum('ij,ij->i', tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() / 6.0)
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    g = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(nv, nv))
    used = np.zeros(nv, bool)
    used[F.ravel()] = True
    _, lab = connected_components(g, directed=False)
    shells = len(np.unique(lab[used]))
    chi = int(used.sum()) - len(cu) + nf
    genus = (2 * shells - chi) // 2
    degenerate = int((area < 1e-9).sum())
    return {'triangles': nf, 'vertices': int(used.sum()), 'watertight': manifold, 'manifold_edges': manifold,
            'consistent': consistent, 'degenerate': degenerate, 'volume_mm3': vol, 'area_mm2': float(area.sum()),
            'shells': shells, 'genus': int(genus),
            'ok': manifold and consistent and degenerate == 0 and vol > 0}


def shells(V, F) -> list:
    """Face index arrays of the mesh's connected pieces."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    nv = len(V)
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    g = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(nv, nv))
    _, lab = connected_components(g, directed=False)
    fl = lab[F[:, 0]]
    return [np.nonzero(fl == k)[0] for k in np.unique(fl)]


# =========================================================================== STL
_STL_DT = np.dtype([('n', '<f4', (3,)), ('v', '<f4', (3, 3)), ('attr', '<u2')])


def stl_bytes(V, F, header='') -> bytes:
    """Binary STL: an 80-byte header, the triangle count (uint32), then per triangle its unit normal and three
    vertices (float32 little-endian) and a zero attribute word."""
    V, F = np.asarray(V, float), np.asarray(F, np.int64)
    tri = V[F]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-30)
    rec = np.zeros(len(F), _STL_DT)
    rec['n'] = n
    rec['v'] = tri
    head = header.encode('ascii', 'replace')[:80]
    if head.lower().startswith(b'solid'):
        head = b'Vantage ' + head                     # an ASCII-STL reader would take 'solid…' for a text file
    head = head[:80].ljust(80, b' ')
    return head + struct.pack('<I', len(F)) + rec.tobytes()


def read_stl(data: bytes):
    """A binary STL back: (V, F, normals, header text); vertices merged where their coordinates are equal."""
    if len(data) < 84:
        raise ValueError('not a binary STL: shorter than its header')
    n = struct.unpack('<I', data[80:84])[0]
    if len(data) != 84 + 50 * n:
        raise ValueError(f'not a binary STL: {len(data)} bytes for {n} triangles')
    rec = np.frombuffer(data, _STL_DT, count=n, offset=84)
    tri = rec['v'].astype(np.float64).reshape(-1, 3)
    V, inv = np.unique(tri, axis=0, return_inverse=True)
    return V, inv.reshape(-1, 3), rec['n'].astype(np.float64), data[:80].decode('ascii', 'replace').rstrip()


# =========================================================================== queries
def raytrace(solid: Solid, O, D, t_max, eps=0.01, max_steps=400):
    """Sphere tracing: for rays O + t·D (D unit), whether each meets the solid before t_max, and where (t)."""
    O, D = np.asarray(O, float).reshape(-1, 3), np.asarray(D, float).reshape(-1, 3)
    n = len(O)
    t = np.zeros(n)
    hit = np.zeros(n, bool)
    alive = np.ones(n, bool)
    t_max = np.broadcast_to(np.asarray(t_max, float), (n,))
    for _ in range(max_steps):
        idx = np.nonzero(alive)[0]
        if not len(idx):
            break
        d = solid.f(O[idx] + D[idx] * t[idx, None])
        h = d < eps
        hit[idx[h]] = True
        alive[idx[h]] = False
        rest = idx[~h]
        t[rest] += np.maximum(d[~h], eps * 0.5)
        alive[rest[t[rest] > t_max[rest]]] = False
    return hit, t


def section(solid: Solid, origin, ex, ey, half, h=0.25):
    """The solid's cross-section in the plane through origin spanned by the unit vectors ex, ey, sampled every h mm
    over ±half: (area mm², centroid (u, v), smallest and largest principal second moment of area mm⁴)."""
    us = np.arange(-half[0], half[0] + h / 2, h)
    vs = np.arange(-half[1], half[1] + h / 2, h)
    U, W = np.meshgrid(us, vs, indexing='ij')
    P = np.asarray(origin, float) + U.ravel()[:, None] * ex + W.ravel()[:, None] * ey
    m = solid.f(P) < 0
    dA = h * h
    A = float(m.sum() * dA)
    if A <= 0:
        return 0.0, (0.0, 0.0), 0.0, 0.0
    u, v = U.ravel()[m], W.ravel()[m]
    cu, cv = u.mean(), v.mean()
    Iuu = float(((v - cv) ** 2).sum() * dA + len(u) * dA * dA / 12)
    Ivv = float(((u - cu) ** 2).sum() * dA + len(u) * dA * dA / 12)
    Iuv = float(((u - cu) * (v - cv)).sum() * dA)
    ev = np.linalg.eigvalsh(np.array([[Iuu, -Iuv], [-Iuv, Ivv]]))
    return A, (float(cu), float(cv)), float(ev[0]), float(ev[1])


# =========================================================================== drawing
def box_mesh(centre, axes, half):
    """A box as 8 vertices and 12 outward triangles (for drawing)."""
    s = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1], [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]])
    V = np.asarray(centre, float) + (s * np.asarray(half, float)) @ np.asarray(axes, float).T
    F = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
                  [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]])
    if np.linalg.det(np.asarray(axes, float)) < 0:
        F = F[:, ::-1]
    return V, F


def disc_mesh(centre, normal, r, seg=24):
    """A closed thin cylinder (lens front) as triangles, for drawing."""
    n = _unit(normal)
    t1 = np.cross(n, [0.0, 0.0, 1.0] if abs(n[2]) < 0.9 else [1.0, 0.0, 0.0])
    t1 /= np.linalg.norm(t1)
    t2 = np.cross(n, t1)
    a = np.linspace(0, 2 * math.pi, seg, endpoint=False)
    ring = np.cos(a)[:, None] * t1 + np.sin(a)[:, None] * t2
    c = np.asarray(centre, float)
    V = np.vstack([c + r * ring, c + r * ring + n * 0.6, [c], [c + n * 0.6]])
    F = []
    for i in range(seg):
        j = (i + 1) % seg
        F += [[i, j, seg + j], [i, seg + j, seg + i], [2 * seg, j, i], [2 * seg + 1, seg + i, seg + j]]
    return V, np.array(F)


def view_basis(yaw_deg, elev_deg):
    """Orthographic view axes (right, up, towards the viewer) for a view from azimuth yaw and elevation elev
    (degrees; elev 90 looks straight down -z)."""
    y, e = math.radians(yaw_deg), math.radians(elev_deg)
    back = np.array([math.cos(e) * math.cos(y), math.cos(e) * math.sin(y), math.sin(e)])
    right = np.cross([0.0, 0.0, 1.0], back)
    if np.linalg.norm(right) < 1e-6:
        right = np.array([-math.sin(y), math.cos(y), 0.0])
    right /= np.linalg.norm(right)
    up = np.cross(back, right)
    return np.stack([right, up, back])


def render(layers, basis, size, scale, centre, bg=(255, 255, 255), light=(0.35, 0.55, 0.75), ss=2,
           outline=True):
    """Shaded orthographic picture of meshes with a z-buffer, in numpy: an RGB uint8 image (h, w, 3).

    layers: [(V, F, rgb (0-255), alpha)], alpha < 1 draws that layer see-through over the others. basis: rows right,
    up, towards the viewer (view_basis). scale: pixels per unit; centre: the point drawn in the image's middle. ss:
    supersampling factor. outline darkens depth steps (silhouettes and creases)."""
    W, H = int(size[0]) * ss, int(size[1]) * ss
    B = np.asarray(basis, float)
    Lt = _unit(light)
    bgc = np.asarray(bg, float)
    base_rgb = np.empty((H, W, 3))
    base_rgb[:] = bgc
    zbuf = np.full(H * W, np.inf)
    out = base_rgb.reshape(-1, 3)
    for V, F, rgb, alpha in layers:
        V, F = np.asarray(V, float), np.asarray(F, np.int64)
        if not len(F):
            continue
        q = (V - centre) @ B.T
        xy = np.empty((len(V), 2))
        xy[:, 0] = W / 2 + q[:, 0] * scale * ss
        xy[:, 1] = H / 2 - q[:, 1] * scale * ss
        z = -q[:, 2]                                           # smaller is nearer
        tri = q[F]
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        nn = np.linalg.norm(n, axis=1)
        keep = (n[:, 2] > 0) & (nn > 0)                        # faces turned to the viewer (closed meshes)
        F2, n2 = F[keep], n[keep] / nn[keep, None]
        lam = 0.38 + 0.62 * np.clip(n2 @ Lt, 0, 1)
        col = np.clip(np.asarray(rgb, float)[None] * lam[:, None], 0, 255)
        pid, depth, idx = _raster(xy[F2], z[F2], W, H)
        if not len(pid):
            continue
        if alpha >= 1:
            closer = depth < zbuf[pid]
            pid, depth, idx = pid[closer], depth[closer], idx[closer]
            zbuf[pid] = depth
            out[pid] = col[idx]
        else:
            vis = depth < zbuf[pid]
            pid, idx = pid[vis], idx[vis]
            out[pid] = out[pid] * (1 - alpha) + col[idx] * alpha
    img = out.reshape(H, W, 3)
    if outline:
        zb = zbuf.reshape(H, W)
        fin = np.isfinite(zb)
        zc = np.where(fin, zb, 1e9)
        step = np.zeros((H, W), bool)
        thr = 1.2 / (scale * ss) * 2.5
        dx = np.abs(np.diff(zc, axis=1)) > thr
        dy = np.abs(np.diff(zc, axis=0)) > thr
        step[:, 1:] |= dx
        step[1:, :] |= dy
        img[step] *= 0.45
    import cv2
    small = cv2.resize(img.astype(np.float32), (W // ss, H // ss), interpolation=cv2.INTER_AREA)
    return np.clip(small, 0, 255).astype(np.uint8)


def _raster(xy, z, W, H, budget=3_000_000):
    """Pixels covered by the triangles xy (m, 3, 2): (pixel ids, depths, triangle index), nearest per pixel."""
    m = len(xy)
    x0 = np.clip(np.floor(xy[:, :, 0].min(axis=1)).astype(np.int64), 0, W - 1)
    x1 = np.clip(np.ceil(xy[:, :, 0].max(axis=1)).astype(np.int64), 0, W - 1)
    y0 = np.clip(np.floor(xy[:, :, 1].min(axis=1)).astype(np.int64), 0, H - 1)
    y1 = np.clip(np.ceil(xy[:, :, 1].max(axis=1)).astype(np.int64), 0, H - 1)
    bw, bh = x1 - x0 + 1, y1 - y0 + 1
    cnt = np.where((xy[:, :, 0].max(axis=1) >= 0) & (xy[:, :, 0].min(axis=1) < W) &
                   (xy[:, :, 1].max(axis=1) >= 0) & (xy[:, :, 1].min(axis=1) < H), bw * bh, 0)
    pids, deps, idxs = [], [], []
    order = np.arange(m)
    csum = np.cumsum(cnt)
    start = 0
    while start < m:
        lim = (csum[start - 1] if start else 0) + budget
        end = max(start + 1, int(np.searchsorted(csum, lim, side='right')))
        sel = order[start:end]
        c = cnt[sel]
        tot = int(c.sum())
        start = end
        if tot == 0:
            continue
        ti = np.repeat(sel, c)
        loc = np.arange(tot) - np.repeat(np.cumsum(c) - c, c)
        px = x0[ti] + loc % bw[ti]
        py = y0[ti] + loc // bw[ti]
        cx, cy = px + 0.5, py + 0.5
        a, b, cc = xy[ti, 0], xy[ti, 1], xy[ti, 2]
        area = (b[:, 0] - a[:, 0]) * (cc[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (cc[:, 0] - a[:, 0])
        ok = np.abs(area) > 1e-12
        sa = np.where(ok, area, 1.0)
        w0 = ((b[:, 0] - cx) * (cc[:, 1] - cy) - (b[:, 1] - cy) * (cc[:, 0] - cx)) / sa
        w1 = ((cc[:, 0] - cx) * (a[:, 1] - cy) - (cc[:, 1] - cy) * (a[:, 0] - cx)) / sa
        w2 = 1 - w0 - w1
        ins = ok & (w0 >= -1e-9) & (w1 >= -1e-9) & (w2 >= -1e-9)
        if not ins.any():
            continue
        ti, px, py = ti[ins], px[ins], py[ins]
        d = w0[ins] * z[ti, 0] + w1[ins] * z[ti, 1] + w2[ins] * z[ti, 2]
        pids.append(py * W + px)
        deps.append(d)
        idxs.append(ti)
    if not pids:
        return np.zeros(0, np.int64), np.zeros(0), np.zeros(0, np.int64)
    pid, dep, idx = np.concatenate(pids), np.concatenate(deps), np.concatenate(idxs)
    # nearest per pixel: one sort on (pixel, depth) packed into an int64 (the depth gets the bits the pixel leaves)
    nb = max(int(W * H - 1).bit_length(), 1)
    db = 62 - nb
    lo_, span = dep.min(), max(float(np.ptp(dep)), 1e-12)
    key = (pid.astype(np.int64) << db) | ((dep - lo_) / span * ((1 << db) - 1)).astype(np.int64)
    o = np.argsort(key)
    pid, dep, idx = pid[o], dep[o], idx[o]
    first = np.ones(len(pid), bool)
    first[1:] = pid[1:] != pid[:-1]
    return pid[first], dep[first], idx[first]
