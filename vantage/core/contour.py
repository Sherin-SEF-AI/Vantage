"""Vectorised marching squares: iso-line segments of a scalar grid."""
from __future__ import annotations

import numpy as np

# edges: 0 bottom (y0: a-b), 1 right (b-c), 2 top (c-d), 3 left (d-a); corners a(0,0) b(1,0) c(1,1) d(0,1)
_CASES = {
    1: [(3, 0)], 2: [(0, 1)], 3: [(3, 1)], 4: [(1, 2)], 5: [(3, 2), (0, 1)], 6: [(0, 2)], 7: [(3, 2)],
    8: [(2, 3)], 9: [(0, 2)], 10: [(0, 3), (1, 2)], 11: [(1, 2)], 12: [(1, 3)], 13: [(0, 1)], 14: [(0, 3)],
}


def iso_segments(field: np.ndarray, level: float, x0: float, y0: float, cell: float) -> np.ndarray:
    """Segments (S,2,2) in world coordinates where field crosses level.
    field[j, i] is sampled at (x0 + (i+0.5)*cell, y0 + (j+0.5)*cell)."""
    f = np.asarray(field, float) - level
    f = np.where(np.isfinite(f), f, -1.0)
    a, b = f[:-1, :-1], f[:-1, 1:]
    d, c = f[1:, :-1], f[1:, 1:]
    code = (a > 0) * 1 + (b > 0) * 2 + (c > 0) * 4 + (d > 0) * 8
    jj, ii = np.nonzero((code > 0) & (code < 15))
    if len(jj) == 0:
        return np.zeros((0, 2, 2))
    va, vb, vc, vd = a[jj, ii], b[jj, ii], c[jj, ii], d[jj, ii]

    def t(p, q):
        den = p - q
        return np.clip(np.where(np.abs(den) < 1e-12, 0.5, p / np.where(np.abs(den) < 1e-12, 1, den)), 0, 1)

    # edge points in cell-local coordinates
    e = np.zeros((4, len(jj), 2))
    e[0, :, 0], e[0, :, 1] = t(va, vb), 0.0
    e[1, :, 0], e[1, :, 1] = 1.0, t(vb, vc)
    e[2, :, 0], e[2, :, 1] = 1 - t(vc, vd), 1.0
    e[3, :, 0], e[3, :, 1] = 0.0, 1 - t(vd, va)
    cc = code[jj, ii]
    segs = []
    for k, pairs in _CASES.items():
        m = cc == k
        if not m.any():
            continue
        for e0, e1 in pairs:
            segs.append(np.stack([e[e0, m], e[e1, m]], axis=1) + np.stack([ii[m], jj[m]], 1)[:, None, :])
    s = np.concatenate(segs)
    return np.stack([x0 + (s[..., 0] + 0.5) * cell, y0 + (s[..., 1] + 0.5) * cell], axis=-1)


def chain(segs: np.ndarray, tol: float = 1e-6) -> list[np.ndarray]:
    """Join segments into polylines (for dashed drawing and smooth outlines)."""
    if len(segs) == 0:
        return []
    key = lambda p: (round(p[0] / tol), round(p[1] / tol))
    ends = {}
    for i, s in enumerate(segs):
        for e in (0, 1):
            ends.setdefault(key(s[e]), []).append((i, e))
    used = np.zeros(len(segs), bool)
    lines = []
    for i in range(len(segs)):
        if used[i]:
            continue
        used[i] = True
        pts = [segs[i][0], segs[i][1]]
        for direction in (1, 0):
            while True:
                tip = pts[-1] if direction else pts[0]
                nxt = None
                for j, e in ends.get(key(tip), []):
                    if not used[j]:
                        nxt = (j, e)
                        break
                if nxt is None:
                    break
                j, e = nxt
                used[j] = True
                other = segs[j][1 - e]
                if direction:
                    pts.append(other)
                else:
                    pts.insert(0, other)
        lines.append(np.array(pts))
    return lines
