"""Faults written into a recording's frames for a dry run, so the watchdog's checks can be seen to fire.

Each fault is a dict with 'kind', 'camera' (a tag, or '*' for every camera), 'at_s' (seconds after the recording's
first frame) and its own fields:
- knock: the camera turned by yaw / pitch / roll degrees (about its own body axes, Vantage's convention: intrinsic
  Z-Y-X, pitch positive down) from at_s on. Each frame is re-rendered through the camera's lens model as the rotated
  camera would see it (every pixel's ray rotated and looked up in the original image; rays the original did not see are
  black); the lens circle stays where it is, because the lens did not move on the sensor. Exact for a rotation about the
  optical centre (a real knock also shifts the camera by millimetres: no parallax at road distances).
- freeze: the frame at at_s repeated, with the real stamps, for dur_s (the driver hands out the same buffer).
- drop: of the frames in [at_s, at_s + dur_s) only every keep_every-th is delivered.
- stall: no frames from at_s for dur_s (0: to the end).
- dark: the image's brightness times gain for dur_s (a covered lens, a failed auto-exposure).
"""
from __future__ import annotations

import math

import numpy as np

from .config import RadialLens

KINDS = ('knock', 'freeze', 'drop', 'stall', 'dark')
BODY_TO_OPTICAL = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])


def rot_zyx(yaw, pitch, roll):
    """Body -> world rotation, degrees, pitch positive down (as vantage.core.mathutil.rot_zyx)."""
    y, p, r = (math.radians(a) for a in (yaw, pitch, roll))
    Rz = np.array([[math.cos(y), -math.sin(y), 0], [math.sin(y), math.cos(y), 0], [0, 0, 1]])
    Ry = np.array([[math.cos(p), 0, math.sin(p)], [0, 1, 0], [-math.sin(p), 0, math.cos(p)]])
    Rx = np.array([[1, 0, 0], [0, math.cos(r), -math.sin(r)], [0, math.sin(r), math.cos(r)]])
    return Rz @ Ry @ Rx


def clean(faults) -> list:
    out = []
    for f in faults or []:
        if not isinstance(f, dict) or f.get('kind') not in KINDS:
            continue
        g = {'kind': f['kind'], 'camera': str(f.get('camera') or '*'), 'at_s': max(0.0, float(f.get('at_s', 0.0)))}
        if f['kind'] == 'knock':
            for k in ('yaw', 'pitch', 'roll'):
                g[k] = float(f.get(k, 0.0))
        else:
            g['dur_s'] = max(0.0, float(f.get('dur_s', 2.0)))
        if f['kind'] == 'drop':
            g['keep_every'] = max(2, int(f.get('keep_every', 3)))
        if f['kind'] == 'dark':
            g['gain'] = min(max(float(f.get('gain', 0.06)), 0.0), 4.0)
        out.append(g)
    return out


class Injector:
    """transform(tag, t, frame) -> [(t, frame)] for vantage.edge.sources.merged."""

    def __init__(self, faults, lenses: dict | None = None):
        self.faults = clean(faults)
        self.lenses = lenses or {}
        self.maps = {}
        self.held = {}
        self.t0 = {}
        self.k = {}

    def _knock_map(self, tag, f, w, h):
        key = (tag, w, h, f['yaw'], f['pitch'], f['roll'])
        m = self.maps.get(key)
        if m is None:
            ln = self.lenses.get(tag)
            if ln is None:
                raise ValueError(f'{tag}: a knock needs the camera\'s lens in the configuration')
            lens = ln if isinstance(ln, RadialLens) else RadialLens(ln)
            uu, vv = np.meshgrid(np.arange(w, dtype=float), np.arange(h, dtype=float))
            d_new, th = lens.unproject(uu.ravel(), vv.ravel(), w, h)
            M = BODY_TO_OPTICAL @ rot_zyx(f['yaw'], f['pitch'], f['roll']) @ BODY_TO_OPTICAL.T
            d_old = d_new @ M.T
            u, v, ok = lens.project(d_old, w, h)
            ok &= np.isfinite(th)
            mx = np.where(ok, u, -10.0).reshape(h, w).astype(np.float32)
            my = np.where(ok, v, -10.0).reshape(h, w).astype(np.float32)
            m = self.maps[key] = (mx, my)
        return m

    def __call__(self, tag, t, frame):
        import cv2
        t0 = self.t0.setdefault(tag, t)
        rel = t - t0
        out = [(t, frame)]
        for f in self.faults:
            if f['camera'] not in ('*', tag) or rel < f['at_s'] - 1e-9:
                continue
            end = f['at_s'] + f['dur_s'] if f['kind'] != 'knock' and f.get('dur_s') else math.inf
            inside = rel < end - 1e-9
            kind = f['kind']
            if kind == 'knock':
                mx, my = self._knock_map(tag, f, frame.shape[1], frame.shape[0])
                out = [(tt, cv2.remap(ff, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT))
                       for tt, ff in out]
            elif kind == 'freeze' and inside:
                key = (tag, id(f))
                if key not in self.held:
                    self.held[key] = out[0][1] if out else frame
                out = [(tt, self.held[key]) for tt, _ in out]
            elif kind == 'drop' and inside:
                key = (tag, id(f))
                self.k[key] = self.k.get(key, -1) + 1
                if self.k[key] % f['keep_every']:
                    out = []
            elif kind == 'stall' and inside:
                out = []
            elif kind == 'dark' and inside:
                out = [(tt, np.clip(ff.astype(np.float32) * f['gain'], 0, 255).astype(np.uint8)) for tt, ff in out]
        return out
