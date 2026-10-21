"""Calibration target planner: which cameras see each floor target, and where to put them."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .analysis import Engine, eval_points
from .lens import resolve
from .project import Project, Target, TargetSettings, new_uid


@dataclass
class TargetStatus:
    uid: str
    seen_by: list            # [(cam uid, tag, visible fraction, min density)]
    ok: bool


def board_points(t: Target, n=5):
    w, h = t.size_m
    u = np.linspace(-w / 2, w / 2, n)
    v = np.linspace(-h / 2, h / 2, n)
    U, V = np.meshgrid(u, v)
    c, s = math.cos(math.radians(t.yaw)), math.sin(math.radians(t.yaw))
    x = t.x + U * c - V * s
    y = t.y + U * s + V * c
    return np.stack([x.ravel(), y.ravel(), np.zeros(U.size)], 1)


def evaluate_targets(engine: Engine, project: Project, targets=None, variant=None) -> list[TargetStatus]:
    st: TargetSettings = project.target_settings
    variant = variant or project.variant
    cams = [c for c in variant.cameras if c.visible]
    out = []
    targets = project.targets if targets is None else targets
    for t in targets:
        P = board_points(t)
        seen = []
        for c in cams:
            intr = resolve(project.module_for(c), project.lens_for(c))
            vis, dens, _, _ = eval_points(P, c, intr, engine.cube(c.pos()))
            frac = float(vis.mean())
            md = float(dens[vis].min()) if vis.any() else 0.0
            if frac >= 0.9 and md >= st.min_density:
                seen.append((c.uid, c.tag, frac, md))
        out.append(TargetStatus(t.uid, seen, len(seen) >= st.cams_per_target))
    return out


def adjacent_pairs(result) -> list[tuple[str, str]]:
    return [(s.a, s.b) for s in (result.seams if result else [])]


def suggest(engine: Engine, project: Project, result, statuses=None, step=0.25):
    """Suggested moves for failing targets: put each one where an uncovered adjacent pair both see it."""
    statuses = statuses or evaluate_targets(engine, project)
    st = project.target_settings
    variant = project.variant
    pairs = adjacent_pairs(result)
    covered = set()
    for s in statuses:
        uids = {u for u, *_ in s.seen_by}
        for a, b in pairs:
            if a in uids and b in uids:
                covered.add((a, b))
    need = [p for p in pairs if p not in covered]
    failing = [t for t, s in zip(project.targets, statuses) if not s.ok]
    if not failing:
        return []
    grid = result.grid
    xs = np.arange(grid.x0 + 0.4, grid.x0 + grid.nx * grid.cell - 0.4, step)
    ys = np.arange(grid.y0 + 0.4, grid.y0 + grid.ny * grid.cell - 0.4, step)
    X, Y = np.meshgrid(xs, ys)
    cand = np.stack([X.ravel(), Y.ravel()], 1)
    keep = []
    for x, y in cand:
        g = grid.cell_of(x, y)
        if g and grid.roi[g[1], g[0]] and grid.dist[g[1], g[0]] > 0.35:
            keep.append((x, y))
    cand = np.array(keep) if keep else np.zeros((0, 2))
    cams = {c.uid: c for c in variant.cameras if c.visible}
    # score every candidate board for every camera
    proto = failing[0]
    seen_min = {}
    for uid, c in cams.items():
        intr = resolve(project.module_for(c), project.lens_for(c))
        cube = engine.cube(c.pos())
        P = np.concatenate([board_points(Target(x=x, y=y, rows=proto.rows, cols=proto.cols, square_mm=proto.square_mm), 3)
                            for x, y in cand]) if len(cand) else np.zeros((0, 3))
        vis, dens, _, _ = eval_points(P, c, intr, cube) if len(P) else (np.zeros(0, bool), np.zeros(0), None, None)
        vis = vis.reshape(len(cand), -1)
        dens = dens.reshape(len(cand), -1)
        ok = vis.mean(1) >= 0.9
        md = np.where(vis, dens, np.inf).min(1)
        seen_min[uid] = np.where(ok & (md >= st.min_density), md, 0.0)
    out = []
    taken = [(t.x, t.y) for t in project.targets if t not in failing]
    for t in failing:
        best = None
        for a, b in (need or pairs):
            if a not in seen_min or b not in seen_min:
                continue
            score = np.minimum(seen_min[a], seen_min[b])
            for k in np.argsort(-score):
                if score[k] <= 0:
                    break
                x, y = cand[k]
                if all(math.hypot(x - tx, y - ty) > 0.9 for tx, ty in taken):
                    d = math.hypot(x - t.x, y - t.y)
                    val = score[k] - 0.15 * d
                    if best is None or val > best[0]:
                        best = (val, x, y, (a, b))
                    break
        if best:
            _, x, y, (a, b) = best
            taken.append((x, y))
            if (a, b) in need:
                need.remove((a, b))
            out.append((t.uid, round(float(x), 3), round(float(y), 3), (cams[a].tag, cams[b].tag)))
    return out


def auto_place(engine: Engine, project: Project, result, count=None):
    """Fresh target layout: one target per adjacent camera pair, at the best shared spot."""
    st = project.target_settings
    pairs = adjacent_pairs(result)
    n = count or len(pairs)
    tmp = [Target(name=f'T{i + 1}', rows=st.rows, cols=st.cols, square_mm=st.square_mm, pattern=st.pattern, x=99, y=99)
           for i in range(n)]
    saved = project.targets
    project.targets = tmp
    try:
        sugg = suggest(engine, project, result, statuses=[type('S', (), {'ok': False, 'seen_by': []})() for _ in tmp])
    finally:
        project.targets = saved
    by = {uid: (x, y) for uid, x, y, _ in sugg}
    out = []
    for t in tmp:
        if t.uid in by:
            t.x, t.y = by[t.uid]
            out.append(t)
    return out
