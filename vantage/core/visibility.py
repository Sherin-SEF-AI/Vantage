"""Which cameras see a point: the lens model (field of view and the image rectangle) and occlusion by the body, the
rig and the sensor and camera housings (the engine's cube maps, core.housings composited in).

The same test the GPU runs per pixel for the 3D view's camera light (ui/render_gl.py), on the CPU: the software
renderer colours the body with it and the tests check the shader against it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .lens import Intrinsics, resolve
from .mathutil import BODY_TO_OPTICAL, rot_zyx


@dataclass
class CamView:
    uid: str
    W2O: np.ndarray              # (3,3) world -> OpenCV optical
    pos: np.ndarray              # (3,)
    intr: Intrinsics
    color: tuple                 # rgb 0..1
    cube: object = None          # occlusion.CubeDepth from this position, or None (no occlusion)


def cam_views(project, engine, cams, colors, cube='cached', cubes_hint=None) -> list[CamView]:
    """CamViews for cameras. cube: 'build' computes missing occlusion maps, 'cached' only uses ones the engine already
    has (falling back to cubes_hint[uid], e.g. the map from where the camera was a moment ago), None skips occlusion."""
    out = []
    vg = engine.vehicle_geom(project) if engine is not None else None
    for c, col in zip(cams, colors):
        intr = resolve(project.module_for(c), project.lens_for(c))
        R = rot_zyx(c.yaw, c.pitch, c.roll)
        cb = None
        if vg is not None and cube:
            if cube == 'build':
                cb = engine.cube(c.pos())
            else:
                cb = engine.cubes.peek(vg.occ, c.pos()) or (cubes_hint or {}).get(c.uid)
        out.append(CamView(c.uid, BODY_TO_OPTICAL @ R.T, c.pos(), intr, tuple(col), cb))
    return out


def seen_by(P: np.ndarray, views: list[CamView]) -> np.ndarray:
    """(K, len(views)) bool: which cameras see each world point."""
    P = np.asarray(P, float).reshape(-1, 3)
    out = np.zeros((len(P), len(views)), bool)
    for i, v in enumerate(views):
        d = P - v.pos
        u, w, th, ok = v.intr.project_dirs(d @ v.W2O.T)
        if v.cube is not None and ok.any():
            dist = np.linalg.norm(d, axis=1)
            depth, _ = v.cube.lookup(d)
            ok &= ~(np.isfinite(depth) & (depth < dist - (0.02 + 0.01 * dist)))
        out[:, i] = ok
    return out


def tint(base: np.ndarray, seen: np.ndarray, views: list[CamView], mix=0.6, points=None) -> np.ndarray:
    """Colours (K,3) tinted by the cameras that see them, as the GPU camera light does."""
    base = np.asarray(base, float)
    cols = np.array([v.color for v in views], float) if views else np.zeros((0, 3))
    n = seen.sum(axis=1)
    lum = base @ np.array([0.299, 0.587, 0.114])
    out = base * 0.4 + (lum[:, None] * 0.55) * 0.6                     # unseen: dimmed and desaturated
    hit = n > 0
    if hit.any():
        avg = (seen[hit].astype(float) @ cols) / n[hit, None]
        out[hit] = base[hit] * (1 - mix) + avg * (0.30 + 1.05 * lum[hit, None]) * mix
    if points is not None:
        over = n > 1
        dark = ((np.asarray(points, float).sum(axis=1) * 6.0) % 1.0) < 0.5     # the shader's hatching of overlaps
        out[over & dark] *= 0.90
    return np.clip(out, 0, 1)
