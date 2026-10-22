"""Standing objects in the stitched bird's-eye view.

The bird's-eye view projects every camera pixel onto the ground, so an object standing at P is drawn by camera c as
a smear from its foot outward along the camera's rays: the object's point at height z lands on the ground at
P + (P - C_xy) * z / (C_z - z), and nowhere once z reaches the camera height. The stitched view shows c's smear only
where its output belongs to c (the seams and the blend of the remap tables, stitch.blend_weights) and only where c
sees the object. Near a seam both cameras' smears can leave their own sectors: the object vanishes from the display
although both raw images show it.

analyse() stands a pole (height, diameter) on every ROI cell and measures how much of its height the stitched view
shows, and why the rest is lost: at a seam (vanish) or off the view's edge / above the camera (cut); suggest_seams() looks for seam angles that make fewer objects vanish without leaving more ground below the
minimum density; smears() gives the drawn smears of one pole, for display.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, replace

import numpy as np

from .lens import resolve
from .mathutil import rot_zyx, BODY_TO_OPTICAL
from .stitch import blend_weights

SCHEMA = 'vantage.stitched_objects/1'
OBJECTS = [('pole', 'Pole', 0.5), ('child', 'Child', 0.9), ('adult', 'Adult', 1.8)]
DEFAULT_CELL = 0.05         # m (coarser when the analysis grid is finer: a planning map, kept interactive)
HEIGHTS = 8                 # samples up the object
ISSUE_M2 = 0.05             # vanish area that raises a warning


@dataclass
class Settings:
    height: float = 0.9         # m
    diameter: float = 0.3       # m
    threshold: float = 0.25     # shown fraction below which the object counts as vanished

    @staticmethod
    def from_dict(d: dict | None) -> 'Settings':
        s = Settings()
        for k, v in (d or {}).items():
            if k in Settings.__dataclass_fields__:
                try:
                    v = float(v)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(v):
                    setattr(s, k, v)
        s.height = min(max(s.height, 0.05), 5.0)
        s.diameter = min(max(s.diameter, 0.0), 2.0)
        s.threshold = min(max(s.threshold, 0.01), 0.99)
        return s

    def to_dict(self) -> dict:
        return {'height': self.height, 'diameter': self.diameter, 'threshold': self.threshold}

    @property
    def label(self) -> str:
        for _, name, h in OBJECTS:
            if abs(self.height - h) < 1e-6:
                return f'{h:g} m {name.lower()}'
        return f'{self.height:g} m object'


@dataclass
class Hotspot:
    x: float
    y: float
    area_m2: float
    min_shown: float
    seam: int                   # index into StitchedObjects.seams, -1 if none
    seen_by: list               # tags of the cameras that see the object there in their raw image


@dataclass
class StitchedObjects:
    x0: float                   # map grid (cell centres at x0 + (i + 0.5) * cell)
    y0: float
    cell: float
    nx: int
    ny: int
    settings: Settings
    considered: np.ndarray      # (ny, nx) bool: in the ROI, not under the body
    shown: np.ndarray           # (ny, nx) float32: fraction of the object's height the stitched view shows
    raw: np.ndarray             # (ny, nx) float32: the best single camera's fraction in its raw image
    by_cam: dict                # uid -> (ny, nx) float32: fraction shown by that camera
    vanish: np.ndarray          # (ny, nx) bool: a camera sees it, but the seams hand its smear to cameras that don't
    cut: np.ndarray             # (ny, nx) bool: a camera sees it, but its smear leaves the view or rises above the camera
    blind: np.ndarray           # (ny, nx) bool: no camera sees it
    metrics: dict
    seams: list                 # [{'a', 'b', 'tags', 'angle', 'vanish_m2'}]
    hotspots: list              # [Hotspot], largest first
    variant_uid: str = ''
    revision: int = 0
    ms: float = 0.0
    _geo: object = field(default=None, repr=False)

    @property
    def xs(self):
        return self.x0 + (np.arange(self.nx) + 0.5) * self.cell

    @property
    def ys(self):
        return self.y0 + (np.arange(self.ny) + 0.5) * self.cell

    def at(self, x, y):
        """(shown, raw, {uid: shown by that camera}) at a ground point, None outside the considered cells."""
        i, j = int((x - self.x0) // self.cell), int((y - self.y0) // self.cell)
        if not (0 <= i < self.nx and 0 <= j < self.ny) or not self.considered[j, i]:
            return None
        return (float(self.shown[j, i]), float(self.raw[j, i]),
                {u: float(a[j, i]) for u, a in self.by_cam.items() if a[j, i] > 0.005})


# ---------------------------------------------------------------------------------------------------- sampling
@dataclass
class _Cam:
    uid: str
    tag: str
    slot: int
    pos: np.ndarray
    W2O: np.ndarray
    intr: object
    cube: object


def _cams(project, engine, res):
    variant = project.variant_by_uid(res.variant_uid) or project.variant
    has_body = engine.vehicle_geom(project) is not None
    out = []
    for c in variant.cameras:
        if not c.visible or res.cam_index(c.uid) < 0:
            continue
        intr = resolve(project.module_for(c), project.lens_for(c))
        pos = c.pos()
        out.append(_Cam(c.uid, c.tag, c.slot, pos, BODY_TO_OPTICAL @ rot_zyx(c.yaw, c.pitch, c.roll).T, intr,
                        engine.cube(pos) if has_body else None))
    return variant, out


def _look(cam: _Cam, P):
    """What the remap tables use: the lens sees the point and the body does not hide it. (vis, density px/cm)."""
    d = P - cam.pos
    dist = np.linalg.norm(d, axis=1)
    u, v, th, ok = cam.intr.project_dirs(d @ cam.W2O.T)
    if cam.cube is not None:
        depth, _ = cam.cube.lookup(d)
        ok &= depth > dist - (0.02 + 0.01 * dist)
    return ok, np.where(ok, cam.intr.density_px_per_cm(th, dist), 0)


def _ground_lookup(g, G):
    """For ground points G (N, 2): inside the bird's-eye extent (the analysis grid) and under the body there."""
    gx = (G[:, 0] - g.x0) / g.cell
    gy = (G[:, 1] - g.y0) / g.cell
    inside = (gx >= 0) & (gx < g.nx) & (gy >= 0) & (gy < g.ny)
    gi = np.clip(gx.astype(int), 0, g.nx - 1)
    gj = np.clip(gy.astype(int), 0, g.ny - 1)
    return inside, g.body[gj, gi]


@dataclass
class _Draw:
    """Where camera k draws the object's height sample h: cells idx (into the considered cells), ground points G,
    and what every camera's remap table holds there (vis, dens) — everything except the seams."""
    k: int
    h: int
    idx: np.ndarray
    G: np.ndarray
    vis: dict
    dens: dict
    body: np.ndarray


@dataclass
class _Geo:
    cams: list
    uids: list
    centre: tuple
    blend_m: float
    M: int
    draws: list
    ground_vis: dict
    ground_dens: dict
    ground_xy: np.ndarray
    min_density: float
    cell_area: float
    seams: list
    seen_any: np.ndarray        # (K, M): some camera sees this height


def _geometry(project, engine, res, settings: Settings, cell, cancel=None, progress=None) -> tuple:
    g = res.grid
    s = max(1, int(round(cell / g.cell))) if cell else 1
    ii = np.arange(s // 2, g.nx, s)
    jj = np.arange(s // 2, g.ny, s)
    cell_m = g.cell * s
    xs, ys = g.xs[ii], g.ys[jj]
    considered = (g.roi & ~g.body)[np.ix_(jj, ii)]
    X, Y = np.meshgrid(xs, ys)
    cj, ci = np.nonzero(considered)
    P = np.stack([X[cj, ci], Y[cj, ci]], 1)
    M = len(P)
    variant, cams = _cams(project, engine, res)
    uids = [c.uid for c in cams]
    K = HEIGHTS
    zs = settings.height * (np.arange(K) + 0.5) / K
    r = settings.diameter / 2
    draws = []
    seen = np.zeros((len(cams), K, M), bool)
    for k, c in enumerate(cams):
        if cancel is not None and cancel.is_set():
            return None
        to = c.pos[:2] - P
        n = np.linalg.norm(to, axis=1)
        face = P + to / np.maximum(n, 1e-9)[:, None] * np.minimum(r, 0.95 * n)[:, None]    # the side facing c
        for h, z in enumerate(zs):
            Q = np.c_[face, np.full(M, z)]
            vis, _ = _look(c, Q)
            seen[k, h] = vis
            if z >= c.pos[2] - 1e-3:
                continue                                    # this ray never reaches the ground
            idx = np.flatnonzero(vis)
            if not len(idx):
                continue
            G = c.pos[:2] + (Q[idx, :2] - c.pos[:2]) * (c.pos[2] / (c.pos[2] - z))
            inside, body = _ground_lookup(g, G)
            idx, G, body = idx[inside], G[inside], body[inside]
            if not len(idx):
                continue
            Gw = np.c_[G, np.zeros(len(G))]
            gv, gd = {}, {}
            for c2 in cams:
                v2, d2 = _look(c2, Gw)
                gv[c2.uid] = v2 & ~body
                gd[c2.uid] = d2
            draws.append(_Draw(k, h, idx, G, gv, gd, body))
        if progress:
            progress(0.8 * (k + 1) / max(len(cams), 1))
    # the ground itself under the same remap tables, for the seam advice's density check
    Pw = np.c_[P, np.zeros(M)]
    ground_vis, ground_dens = {}, {}
    for c in cams:
        v, d = _look(c, Pw)
        ground_vis[c.uid], ground_dens[c.uid] = v, d
    geo = _Geo(cams, uids, tuple(res.centre), float(variant.blend_m), M, draws, ground_vis, ground_dens, P,
               float(project.analysis.min_density), cell_m * cell_m, list(res.seams), seen.any(0))
    grid = (float(xs[0] - cell_m / 2) if len(xs) else g.x0, float(ys[0] - cell_m / 2) if len(ys) else g.y0, cell_m,
            len(ii), len(jj), considered, (cj, ci))
    return geo, seen, grid


def _evaluate(geo: _Geo, seams, K=HEIGHTS):
    """For the given seam angles: the best weight any camera that sees a height draws it with (K, M), each camera's
    shown fraction (n, M), and where a camera's drawing lands on ground another camera owns (K, M)."""
    n = len(geo.cams)
    best = np.zeros((K, geo.M), np.float32)
    per = np.zeros((n, geo.M), np.float32)
    other = np.zeros((K, geo.M), bool)
    for d in geo.draws:
        W = blend_weights(d.G[:, 0], d.G[:, 1], geo.centre, seams, geo.uids, geo.blend_m, d.vis, d.dens, d.body)
        u = geo.uids[d.k]
        w = W[u]
        best[d.h, d.idx] = np.maximum(best[d.h, d.idx], w)          # one sample per cell and draw
        per[d.k, d.idx] += w / K
        others = sum((W[v] for v in geo.uids if v != u), np.zeros(len(w), np.float32))
        other[d.h, d.idx] |= others >= 0.5
    return best, per, other


def _vanish(geo: _Geo, best, other, blind, thr):
    """(shown fraction, vanishes at a seam, cut off) per cell. A height is lost where no camera that sees it draws it
    with weight 0.5; the loss is the seams' doing where some camera's drawing lands on ground another camera owns,
    otherwise the view's edge's (outside the view, above the camera, or a hole under the body)."""
    shown = best.mean(0)
    lost = (best < 0.5) & geo.seen_any
    seam = (lost & other).mean(0)
    edge = (lost & ~other).mean(0)
    gone = ~blind & (shown < thr)
    van = gone & (seam > 0) & (seam >= edge)
    return shown, van, gone & ~van


def _ground_low(geo: _Geo, seams) -> float:
    """Ground in the considered cells whose stitched density falls below the minimum, m²."""
    P = geo.ground_xy
    if not len(P):
        return 0.0
    body = np.zeros(len(P), bool)
    W = blend_weights(P[:, 0], P[:, 1], geo.centre, seams, geo.uids, geo.blend_m, geo.ground_vis, geo.ground_dens, body)
    dens = sum(W[u] * geo.ground_dens[u] for u in geo.uids) if geo.uids else np.zeros(len(P))
    covered = sum(W.values()) > 0 if W else np.zeros(len(P), bool)
    return float((covered & (dens < geo.min_density)).sum() * geo.cell_area)


# ---------------------------------------------------------------------------------------------------- analysis
def analyse(project, engine, res, height=None, diameter=None, threshold=None, cell=None, progress=None,
            cancel=None) -> StitchedObjects | None:
    """Stand an object on every considered cell and measure how much of it the stitched view shows.
    Settings default to project.analysis.stitched; None when cancelled or there is no result."""
    if res is None:
        return None
    t0 = time.perf_counter()
    st = Settings.from_dict(getattr(project.analysis, 'stitched', None))
    if height is not None:
        st.height = float(height)
    if diameter is not None:
        st.diameter = float(diameter)
    if threshold is not None:
        st.threshold = float(threshold)
    st = Settings.from_dict(st.to_dict())
    out = _geometry(project, engine, res, st, cell if cell is not None else max(DEFAULT_CELL, res.grid.cell),
                    cancel, progress)
    if out is None:
        return None
    geo, seen, (x0, y0, cell_m, nx, ny, considered, (cj, ci)) = out
    best, per_c, other = _evaluate(geo, geo.seams)
    raw_c = seen.mean(1).max(0) if len(geo.cams) else np.zeros(geo.M, np.float32)
    thr = st.threshold
    shown_c, van_c, cut_c = _vanish(geo, best, other, raw_c < thr, thr)

    def grid(v, fill=0.0):
        a = np.full((ny, nx), fill, np.float32)
        a[cj, ci] = v
        return a
    shown, raw = grid(shown_c), grid(raw_c)
    by_cam = {u: grid(per_c[k]) for k, u in enumerate(geo.uids)}
    blind = considered & (raw < thr)
    vanish = grid(van_c).astype(bool) & considered
    cut = grid(cut_c).astype(bool) & considered
    a = cell_m * cell_m
    seams_out = _per_seam(res, geo, vanish, x0, y0, cell_m, a)
    hotspots = _hotspots(vanish, shown, seen, geo, cj, ci, x0, y0, cell_m, a, seams_out, res.centre, thr)
    seen_any = considered & ~blind
    m = {'object': st.label, 'height_m': st.height, 'diameter_m': st.diameter, 'threshold': thr, 'cell_m': cell_m,
         'considered_m2': float(considered.sum() * a), 'vanish_m2': float(vanish.sum() * a),
         'cut_m2': float(cut.sum() * a), 'blind_m2': float(blind.sum() * a),
         'partial_m2': float((seen_any & (shown >= thr) & (shown < 0.5)).sum() * a),
         'shown_mean': float(shown[seen_any].mean()) if seen_any.any() else 0.0,
         'ground_low_m2': _ground_low(geo, geo.seams),
         'worst_seam': int(np.argmax([s['vanish_m2'] for s in seams_out])) if seams_out and
         max(s['vanish_m2'] for s in seams_out) > 0 else -1}
    if progress:
        progress(1.0)
    return StitchedObjects(x0, y0, cell_m, nx, ny, st, considered, shown, raw, by_cam, vanish, cut, blind, m, seams_out,
                           hotspots, res.variant_uid, int(getattr(res, 'revision', 0)),
                           (time.perf_counter() - t0) * 1000, geo)


def _nearest_seam(angles_deg, az):
    if not len(angles_deg):
        return np.full(np.shape(az), -1)
    d = np.abs((az[..., None] - np.radians(np.asarray(angles_deg))[None] + math.pi) % (2 * math.pi) - math.pi)
    return np.argmin(d, -1)


def _per_seam(res, geo, vanish, x0, y0, cell_m, a):
    tags = {c.uid: c.tag for c in geo.cams}
    seams = [s for s in geo.seams if s.a in tags and s.b in tags]
    out = [{'a': s.a, 'b': s.b, 'tags': (tags[s.a], tags[s.b]), 'angle': float(s.angle), 'vanish_m2': 0.0,
            'auto': bool(s.auto)} for s in seams]
    jj, ii = np.nonzero(vanish)
    if len(out) and len(jj):
        az = np.arctan2(y0 + (jj + 0.5) * cell_m - res.centre[1], x0 + (ii + 0.5) * cell_m - res.centre[0])
        near = _nearest_seam([s['angle'] for s in out], az)
        for k in range(len(out)):
            out[k]['vanish_m2'] = float((near == k).sum() * a)
    return out


def _hotspots(vanish, shown, seen, geo, cj, ci, x0, y0, cell_m, a, seams, centre, thr, limit=24):
    from scipy import ndimage
    lab, n = ndimage.label(vanish, structure=np.ones((3, 3)))
    if not n:
        return []
    cell_of = {(int(j), int(i)): q for q, (j, i) in enumerate(zip(cj, ci))}
    out = []
    for k in range(1, n + 1):
        jj, ii = np.nonzero(lab == k)
        x = float(x0 + (ii.mean() + 0.5) * cell_m)
        y = float(y0 + (jj.mean() + 0.5) * cell_m)
        qs = [cell_of[(int(j), int(i))] for j, i in zip(jj, ii)]
        frac = seen[:, :, qs].mean(axis=(1, 2)) if len(geo.cams) else np.zeros(0)
        by = [geo.cams[q].tag for q in np.argsort(-frac) if frac[q] >= thr]
        s = int(_nearest_seam([sm['angle'] for sm in seams], np.array([math.atan2(y - centre[1], x - centre[0])]))[0])
        out.append(Hotspot(x, y, float(len(jj) * a), float(shown[jj, ii].min()), s, by))
    out.sort(key=lambda h: -h.area_m2)
    return out[:limit]


# ---------------------------------------------------------------------------------------------------- seam advice
@dataclass
class SeamAdvice:
    a: str
    b: str
    tags: tuple
    angle: float                # degrees now
    best: float                 # degrees advised (== angle when nothing helps)
    vanish_before: float        # total vanish m² before this seam moved
    vanish_after: float         # ... after (with the seams advised before it)

    @property
    def key(self):
        return f'{self.a}|{self.b}'

    @property
    def moved(self):
        return abs(self.best - self.angle) > 0.05


@dataclass
class Advice:
    seams: list                 # [SeamAdvice] in the analysis' seam order
    vanish_now: float
    vanish_after: float
    ground_low_now: float
    ground_low_after: float

    @property
    def changes(self) -> dict:
        """'uidA|uidB' -> angle deg for the seams to move."""
        return {s.key: round(s.best, 1) for s in self.seams if s.moved}


def suggest_seams(sres: StitchedObjects, span_deg=15.0, step_deg=2.5, fine_deg=0.5, min_gap_deg=4.0,
                  min_gain_m2=None, cancel=None) -> Advice | None:
    """Coordinate descent over the seam angles (each within ±span_deg, keeping the seam order): moves a seam where
    that makes the chosen object vanish on less ground without more ground falling below the minimum density."""
    geo = sres._geo
    if geo is None or not geo.seams:
        return Advice([], sres.metrics['vanish_m2'], sres.metrics['vanish_m2'], sres.metrics['ground_low_m2'],
                      sres.metrics['ground_low_m2'])
    thr = sres.settings.threshold
    cj, ci = np.nonzero(sres.considered)
    blind = sres.blind[cj, ci]
    a = sres.cell * sres.cell
    tol_low = a                     # at most one more cell below the minimum density than now, in total
    gain_min = a if min_gain_m2 is None else min_gain_m2

    def vanish_of(seams):
        best, _, other = _evaluate(geo, seams)
        return float(_vanish(geo, best, other, blind, thr)[1].sum() * a)
    seams = [replace(s) for s in geo.seams]
    now = vanish_of(seams)
    low_now = _ground_low(geo, seams)
    cur, cur_low = now, low_now
    order = sorted(range(len(seams)), key=lambda k: -next((x['vanish_m2'] for x in sres.seams
                                                           if x['a'] == seams[k].a and x['b'] == seams[k].b), 0.0))
    advice = {}
    n = len(seams)
    for k in order:
        if cancel is not None and cancel.is_set():
            return None
        base = seams[k].angle
        if n > 1:
            gap_prev = (base - seams[(k - 1) % n].angle) % 360.0
            gap_next = (seams[(k + 1) % n].angle - base) % 360.0
        else:
            gap_prev = gap_next = 360.0
        lo = -min(span_deg, gap_prev - min_gap_deg)
        hi = min(span_deg, gap_next - min_gap_deg)
        before = cur
        best = (cur, 0.0, cur_low)

        def trial(delta):
            t = list(seams)
            t[k] = replace(seams[k], angle=_wrap(base + delta))
            return vanish_of(t), t

        if hi > lo:
            for delta in _steps(lo, hi, step_deg):
                v, t = trial(delta)
                if v < best[0] - 1e-9 or (abs(v - best[0]) < 1e-9 and abs(delta) < abs(best[1])):
                    low = _ground_low(geo, t)
                    if low <= low_now + tol_low:
                        best = (v, delta, low)
            c0 = best[1]
            for delta in np.arange(c0 - step_deg + fine_deg, c0 + step_deg - fine_deg / 2, fine_deg):
                if not (lo <= delta <= hi) or abs(delta - c0) < 1e-9:
                    continue
                v, t = trial(float(delta))
                if v < best[0] - 1e-9:
                    low = _ground_low(geo, t)
                    if low <= low_now + tol_low:
                        best = (v, float(delta), low)
        if best[1] != 0.0 and before - best[0] >= gain_min:
            seams[k] = replace(seams[k], angle=_wrap(base + best[1]))
            cur, cur_low = best[0], best[2]
            advice[k] = (base, seams[k].angle, before, cur)
        else:
            advice[k] = (base, base, before, before)
    tags = {c.uid: c.tag for c in geo.cams}
    out = [SeamAdvice(s.a, s.b, (tags.get(s.a, '?'), tags.get(s.b, '?')), advice[k][0], advice[k][1], advice[k][2],
                      advice[k][3]) for k, s in enumerate(seams)]
    return Advice(out, now, cur, low_now, cur_low)


def advise(project, engine, res, cell=0.1, **kw) -> Advice | None:
    """Seam advice from a coarser map (the descent evaluates many seam angles); the project's object settings."""
    sres = analyse(project, engine, res, cell=max(cell, res.grid.cell), cancel=kw.get('cancel'))
    return suggest_seams(sres, **kw) if sres is not None else None


def _wrap(deg):
    return (deg + 180.0) % 360.0 - 180.0


def _steps(lo, hi, step):
    k0, k1 = math.ceil(lo / step - 1e-9), math.floor(hi / step + 1e-9)
    return [k * step for k in range(k0, k1 + 1) if k != 0]


# ---------------------------------------------------------------------------------------------------- one pole
def smears(project, engine, res, x, y, height=None, diameter=None, n=12) -> list:
    """How the stitched view draws one pole at (x, y): per camera that sees it, the ground points its heights land
    on, whether the camera sees each height, and the camera's weight there in the stitched view.
    [{'uid', 'tag', 'slot', 'foot': (2,), 'points': (n, 2), 'seen': (n,), 'weight': (n,), 'shown': float}]."""
    if res is None:
        return []
    st = Settings.from_dict(getattr(project.analysis, 'stitched', None))
    h = float(height if height is not None else st.height)
    r = float(diameter if diameter is not None else st.diameter) / 2
    variant, cams = _cams(project, engine, res)
    uids = [c.uid for c in cams]
    P = np.array([x, y], float)
    zs = h * (np.arange(n) + 0.5) / n
    out = []
    for k, c in enumerate(cams):
        to = c.pos[:2] - P
        d = float(np.linalg.norm(to))
        face = P + (to / d * min(r, 0.95 * d) if d > 1e-9 else 0)
        Q = np.c_[np.tile(face, (n, 1)), zs]
        vis, _ = _look(c, Q)
        if not vis.any():
            continue
        below = zs < c.pos[2] - 1e-3
        G = np.where(below[:, None], c.pos[:2] + (Q[:, :2] - c.pos[:2]) * (c.pos[2] / np.maximum(c.pos[2] - zs, 1e-3))[:, None],
                     np.nan)
        w = np.zeros(n, np.float32)
        ok = below & vis
        if ok.any():
            Gk = G[ok]
            inside, body = _ground_lookup(res.grid, Gk)
            Gw = np.c_[Gk, np.zeros(len(Gk))]
            gv, gd = {}, {}
            for c2 in cams:
                v2, d2 = _look(c2, Gw)
                gv[c2.uid] = v2 & ~body
                gd[c2.uid] = d2
            W = blend_weights(Gk[:, 0], Gk[:, 1], tuple(res.centre), res.seams, uids, variant.blend_m, gv, gd, body)
            w[ok] = np.where(inside, W[c.uid], 0)
        out.append({'uid': c.uid, 'tag': c.tag, 'slot': c.slot, 'foot': face, 'points': G, 'seen': vis, 'weight': w,
                    'shown': float(w.mean())})
    return out


# ---------------------------------------------------------------------------------------------------- reporting
def issue(sres: StitchedObjects | None):
    """A warning for the Issues list when the chosen object vanishes on more than ISSUE_M2."""
    from .rules import Issue
    if sres is None or sres.metrics['vanish_m2'] <= ISSUE_M2:
        return None
    m = sres.metrics
    w = m['worst_seam']
    seam = sres.seams[w] if 0 <= w < len(sres.seams) else None
    where = (sres.hotspots[0].x, sres.hotspots[0].y) if sres.hotspots else None
    txt = (f'A {sres.settings.label} can vanish from the stitched bird\'s-eye view on {m["vanish_m2"]:.2f} m²'
           + (f', mostly at the {seam["tags"][0]} | {seam["tags"][1]} seam' if seam else '')
           + ', although a camera sees it. Camera views ▸ Stitch seams can suggest better seam angles.')
    return Issue('warning', txt, f'{seam["tags"][0]} | {seam["tags"][1]}' if seam else 'seams', [], 'stitched', where)


def to_doc(sres: StitchedObjects, advice: Advice | None = None, L=None, A=None, origin=(0.0, 0.0)) -> dict:
    """JSON-ready summary (schema vantage.stitched_objects/1). L and A format lengths (m) and angles (deg)."""
    L = L or (lambda v: round(float(v), 3))
    A = A or (lambda v: round(float(v), 2))
    ox, oy = origin[0], origin[1]
    m = sres.metrics
    doc = {'schema': SCHEMA, 'object': {'label': m['object'], 'height': L(m['height_m']), 'diameter': L(m['diameter_m'])},
           'threshold': m['threshold'], 'cell': L(m['cell_m']),
           'note': ('vanish: a camera sees the object, but the stitched view shows less than the threshold of its height '
                    'because the seams hand its smear to cameras that do not see it (moving a seam can help). cut: the '
                    'same, but because its smear leaves the bird\'s-eye view or the object is taller than the cameras '
                    'that see it (a larger view helps). blind: no camera sees it.'),
           'metrics': {'considered_m2': round(m['considered_m2'], 3), 'vanish_m2': round(m['vanish_m2'], 3),
                       'cut_m2': round(m['cut_m2'], 3), 'blind_m2': round(m['blind_m2'], 3), 'partial_m2': round(m['partial_m2'], 3),
                       'shown_mean': round(m['shown_mean'], 3), 'ground_below_min_density_m2': round(m['ground_low_m2'], 3)},
           'seams': [{'between': list(s['tags']), 'azimuth': A(s['angle']), 'vanish_m2': round(s['vanish_m2'], 3)}
                     for s in sres.seams],
           'hotspots': [{'x': L(h.x - ox), 'y': L(h.y - oy), 'area_m2': round(h.area_m2, 3),
                         'min_shown': round(h.min_shown, 3), 'seen_by': h.seen_by,
                         'seam': list(sres.seams[h.seam]['tags']) if 0 <= h.seam < len(sres.seams) else None}
                        for h in sres.hotspots]}
    if advice is not None:
        doc['seam_advice'] = {'vanish_m2_now': round(advice.vanish_now, 3), 'vanish_m2_after': round(advice.vanish_after, 3),
                              'ground_below_min_density_m2_now': round(advice.ground_low_now, 3),
                              'ground_below_min_density_m2_after': round(advice.ground_low_after, 3),
                              'seams': [{'between': list(s.tags), 'azimuth_now': A(s.angle), 'azimuth_advised': A(s.best),
                                         'move': bool(s.moved)} for s in advice.seams]}
    return doc


PALETTE = ['#cde2fb', '#b7d3f6', '#9ec5f4', '#86b6ef', '#6da7ec', '#5598e7', '#3987e5', '#2a78d6', '#256abf',
           '#1c5cab', '#184f95', '#104281', '#0d366b']       # none .. all of the object shown (the density palette)


def map_rgba(sres: StitchedObjects, k=4, palette=PALETTE, vanish_rgb=(236, 131, 90), mark_rgb=(250, 250, 250),
             dot_rgb=(24, 24, 22), blind_rgb=(208, 59, 59)) -> np.ndarray:
    """(ny*k, nx*k, 4) uint8, row 0 = max y: the shown fraction, pale (none) to dark blue (all of it); where the
    object vanishes at a seam orange and hatched, cut off at the view's edge dotted, blind hatched red."""
    pal = np.array([[int(h[i:i + 2], 16) for i in (1, 3, 5)] for h in palette], np.float32)
    idx = np.clip(np.round(sres.shown * (len(pal) - 1)), 0, len(pal) - 1).astype(int)
    img = np.zeros((sres.ny, sres.nx, 4), np.float32)
    img[..., :3] = pal[idx]
    img[..., 3] = (sres.considered & ~sres.blind) * 255

    def up(a):
        return np.repeat(np.repeat(a[::-1], k, 0), k, 1)
    img = up(img)
    H, W = img.shape[:2]
    jj, ii = np.mgrid[0:H, 0:W]
    stripes = ((ii + jj) % 6) < 2
    van = up(sres.vanish)
    img[van, :3] = vanish_rgb
    img[van & stripes, :3] = mark_rgb
    img[up(sres.cut) & (ii % 4 == 1) & (jj % 4 == 1), :3] = dot_rgb
    bl = up(sres.blind)
    img[bl, :3] = blind_rgb
    img[bl, 3] = np.where(stripes[bl], 220, 70)
    return img.astype(np.uint8)


def png(sres: StitchedObjects, k=4) -> bytes:
    import cv2
    rgba = map_rgba(sres, k)
    bgra = np.ascontiguousarray(rgba[..., [2, 1, 0, 3]])
    ok, buf = cv2.imencode('.png', bgra)
    return buf.tobytes()
