"""The project's own body, rig and housings as per-camera layers, cast exactly through each camera's lens, lit and
composited into CARLA's images: what core.ego_composite builds on, and what the live CARLA rig (vantage.sim.live)
composites every frame (LiveEgo).

It lives in vantage/sim because the live server runs in CARLA's own Python, where only vantage/sim and a few plain
files are shipped (packaging/release.BRIDGE_SOURCES): NumPy, the standard library, core.mathutil and
core.groundtruth only."""
from __future__ import annotations

import json
import math
import threading
import time
from dataclasses import dataclass, field

import numpy as np

from ..core.mathutil import BODY_TO_OPTICAL, rot_zyx


LAYERS_SCHEMA = 'vantage.ego_layers/1'


NEAR_M = 0.05                                 # core.synth.NEAR_CLIP: the lens protrudes from its module


GRID = 96                                     # cells per cube face side of the ray caster's angular grid


SHADOW_CELL_M = 0.04                          # the shadow caster's cell (m) across the sun's direction


PAIR_CHUNK = 1_200_000                        # ray-triangle pairs tested at once


HEADING_STEP = 5.0                            # degrees: the body's own shadow is cast again per this much heading


MID_GREY = 0.18


# CARLA 0.9.16 weather presets: (cloudiness, sun azimuth, sun altitude), LibCarla/source/carla/rpc/
# WeatherParameters.cpp at the 0.9.16 tag (read 2026-11-11). An azimuth of -1 leaves the level's own sun azimuth; the
# value CARLA then reports is used when the drive recorded it (run.json weather.params).
CARLA_PRESETS = {
    'ClearNoon': (5, -1, 45), 'CloudyNoon': (60, -1, 45), 'WetNoon': (5, -1, 45), 'WetCloudyNoon': (60, -1, 45),
    'MidRainyNoon': (60, -1, 45), 'HardRainNoon': (100, -1, 45), 'SoftRainNoon': (20, -1, 45),
    'ClearSunset': (5, -1, 15), 'CloudySunset': (60, -1, 15), 'WetSunset': (5, -1, 15),
    'WetCloudySunset': (60, -1, 15), 'MidRainSunset': (60, -1, 15), 'HardRainSunset': (100, -1, 15),
    'SoftRainSunset': (20, -1, 15), 'ClearNight': (5, -1, -90), 'CloudyNight': (60, -1, -90),
    'WetNight': (5, -1, -90), 'WetCloudyNight': (60, -1, -90), 'SoftRainNight': (60, -1, -90),
    'MidRainyNight': (80, -1, -90), 'HardRainNight': (100, -1, -90), 'DustStorm': (100, -1, 45)}


DEFAULT_AZIMUTH = 0.0                         # used when neither the drive nor the preset gives one (stated)


# Visible albedo (linear RGB) per material: typical values for the surface kinds, not measured on any car.
MATERIALS = (
    ('paint', None),                          # the settings' paint colour
    ('aluminium', (0.55, 0.56, 0.58)),
    ('chrome', (0.75, 0.76, 0.78)),
    ('carbon', (0.03, 0.03, 0.035)),
    ('trim', (0.035, 0.035, 0.035)),
    ('radome', (0.20, 0.20, 0.21)),
    ('lamp', (0.25, 0.27, 0.30)),
    ('tail_lamp', (0.45, 0.02, 0.02)),
    ('glass', (0.02, 0.025, 0.03)),
    ('tyre', (0.025, 0.025, 0.025)),
    ('rim', (0.45, 0.46, 0.48)),
    ('housing', (0.05, 0.05, 0.055)),
    ('lens', (0.01, 0.01, 0.012)),
)


MAT = {n: i for i, (n, _) in enumerate(MATERIALS)}


GLOSSY = frozenset((MAT['paint'], MAT['glass'], MAT['chrome'], MAT['lamp'], MAT['tail_lamp']))


# =========================================================================== colour
def srgb_to_linear(c):
    c = np.asarray(c, np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c):
    c = np.clip(np.asarray(c, np.float64), 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


def hex_linear(h: str) -> np.ndarray:
    h = h.lstrip('#')
    return srgb_to_linear(np.array([int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)]))


def albedo_table(paint: str) -> np.ndarray:
    """(len(MATERIALS), 3) linear albedo, the paint's from its sRGB colour."""
    return np.array([hex_linear(paint) if v is None else v for _, v in MATERIALS], np.float32)


def theta_of_r(x, projection: str, kb=(0.0, 0.0, 0.0, 0.0)):
    """theta for an image height x (in focal lengths): core.lens.theta_of_r, numpy only (a test checks they agree)."""
    x = np.asarray(x, float)
    if projection == 'rectilinear':
        return np.arctan(x)
    if projection == 'equidistant':
        return x.copy()
    if projection == 'equisolid':
        return 2 * np.arcsin(np.clip(x / 2, -1, 1))
    if projection == 'stereographic':
        return 2 * np.arctan(x / 2)
    if projection == 'orthographic':
        return np.arcsin(np.clip(x, -1, 1))
    if projection == 'kannala_brandt':
        k1, k2, k3, k4 = (list(kb) + [0.0] * 4)[:4]
        t = x.copy()
        for _ in range(16):
            t2 = t * t
            g = t * (1 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4)))) - x
            d = 1 + t2 * (3 * k1 + t2 * (5 * k2 + t2 * (7 * k3 + t2 * 9 * k4)))
            t = np.clip(t - g / np.where(np.abs(d) < 1e-9, 1e-9, d), 0, math.pi)
        return t
    raise ValueError(projection)


def lens_unproject(lens: dict, u, v):
    """Image pixel coordinates (pixel-edge convention: pixel (i, j) covers [i, i+1) × [j, j+1), as
    core.groundtruth.project) of a rig camera's lens block -> (optical unit rays (K, 3), theta, inside the field)."""
    u = np.asarray(u, float).ravel()
    v = np.asarray(v, float).ravel()
    if lens.get('model') == 'pinhole':
        K = np.asarray(lens['K'], float)
        x = (u - 0.5 - K[0, 2]) / K[0, 0]
        y = (v - 0.5 - K[1, 2]) / K[1, 1]
        d = np.stack([x, y, np.ones_like(x)], 1)
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        th = np.arccos(np.clip(d[:, 2], -1, 1))
        return d, th, th <= min(math.radians(float(lens['max_theta_deg'])), math.radians(89.0))
    W, H = lens['sensor_size']
    w, h = lens['image_size']
    us, vs = u * (W / w), v * (H / h)
    f = float(lens['f_px'])
    x = (us - float(lens['cx'])) / f
    y = (vs - float(lens['cy'])) / (f * float(lens.get('aspect', 1.0)))
    r = np.hypot(x, y)
    th = theta_of_r(r, lens['projection'], lens.get('kb', (0, 0, 0, 0)))
    s = np.where(r > 1e-12, np.sin(th) / np.where(r > 1e-12, r, 1.0), 0.0)
    d = np.stack([x * s, y * s, np.cos(th)], 1)
    ok = th <= math.radians(float(lens['max_theta_deg']))
    if lens['projection'] == 'rectilinear':
        ok &= th < math.radians(89.9)
    return d, th, ok


def lens_project(lens: dict, P):
    """Optical points or directions -> (u, v, theta, ok) in image pixel-edge coordinates (core.groundtruth.project)."""
    from ..core.groundtruth import project
    return project(lens, P)


def pixel_grid(w: int, h: int):
    """Pixel-centre image coordinates (u, v), each (h·w,), row by row."""
    U, V = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    return U.ravel(), V.ravel()


def scaled_lens(lens: dict, w: int, h: int) -> dict:
    """The lens block for an output image of w × h (the same sensor and field)."""
    out = dict(lens)
    if lens.get('model') == 'pinhole':
        w0, h0 = lens.get('image_size') or (w, h)
        K = np.asarray(lens['K'], float).copy()
        sx, sy = w / float(w0), h / float(h0)
        K[0] *= sx
        K[1] *= sy
        K[0, 2] = (K[0, 2] + 0.5) * sx - 0.5
        K[1, 2] = (K[1, 2] + 0.5) * sy - 0.5
        out['K'] = K.tolist()
    out['image_size'] = [int(w), int(h)]
    return out


def mount_rotation(pose: dict, deg=None) -> np.ndarray:
    """Body -> vehicle rotation of a camera at pose {'yaw', 'pitch', 'roll'} turned by deg [yaw, pitch, roll] about its
    own axes (R_actual = R_plan · R(deg), the bridge's convention)."""
    R = rot_zyx(float(pose['yaw']), float(pose['pitch']), float(pose['roll']))
    if deg is not None and any(abs(float(x)) > 0 for x in deg):
        R = R @ rot_zyx(*(float(x) for x in deg))
    return R


def mount_position(pose: dict, m=None) -> np.ndarray:
    p = np.array([float(pose['x']), float(pose['y']), float(pose['z'])])
    return p + (np.asarray(m, float) if m is not None else 0.0)


def W2O_of(R) -> np.ndarray:
    return BODY_TO_OPTICAL @ np.asarray(R, float).T


# =========================================================================== the exact ray caster
_FACES = (
    (np.array([1.0, 0, 0]), np.array([0, -1.0, 0]), np.array([0, 0, 1.0])),
    (np.array([-1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([0, 0, 1.0])),
    (np.array([0, 1.0, 0]), np.array([1.0, 0, 0]), np.array([0, 0, 1.0])),
    (np.array([0, -1.0, 0]), np.array([-1.0, 0, 0]), np.array([0, 0, 1.0])),
    (np.array([0, 0, 1.0]), np.array([1.0, 0, 0]), np.array([0, -1.0, 0])),
    (np.array([0, 0, -1.0]), np.array([1.0, 0, 0]), np.array([0, 1.0, 0])),
)


_EPS_Z = 1e-7


def _ray_cells(D, G):
    """Cube-face cell of each direction (K,)."""
    ad = np.abs(D)
    ax = np.argmax(ad, axis=1)
    neg = np.take_along_axis(D, ax[:, None], 1)[:, 0] < 0
    face = ax * 2 + neg.astype(np.int64)
    cell = np.zeros(len(D), np.int64)
    for f, (fw, rt, up) in enumerate(_FACES):
        m = face == f
        if not m.any():
            continue
        d = D[m]
        z = d @ fw
        x = (d @ rt) / z
        y = (d @ up) / z
        ix = np.clip(((x + 1) * 0.5 * G).astype(np.int64), 0, G - 1)
        iy = np.clip(((y + 1) * 0.5 * G).astype(np.int64), 0, G - 1)
        cell[m] = f * G * G + iy * G + ix
    return cell


def _tri_cells(rel, G):
    """(cells, triangle indices): every angular cell each triangle (relative to the origin) may cover."""
    cells, tris = [], []
    edges = ((0, 1), (1, 2), (2, 0))
    for f, (fw, rt, up) in enumerate(_FACES):
        z = rel @ fw                                  # (M, 3)
        x = rel @ rt
        y = rel @ up
        front = z > _EPS_Z
        cand = front.any(axis=1)
        if not cand.any():
            continue
        idx = np.flatnonzero(cand)
        zc, xc, yc, fc = z[idx], x[idx], y[idx], front[idx]
        # the triangle clipped to z > eps: its front corners and where its edges cross z = eps
        px, py, ok = [], [], []
        for k in range(3):
            zk = np.where(fc[:, k], zc[:, k], 1.0)
            px.append(xc[:, k] / zk)
            py.append(yc[:, k] / zk)
            ok.append(fc[:, k])
        for a, b in edges:
            cross = fc[:, a] != fc[:, b]
            den = zc[:, b] - zc[:, a]
            s = np.where(cross, (_EPS_Z - zc[:, a]) / np.where(np.abs(den) > 1e-30, den, 1.0), 0.0)
            xi = xc[:, a] + s * (xc[:, b] - xc[:, a])
            yi = yc[:, a] + s * (yc[:, b] - yc[:, a])
            px.append(xi / _EPS_Z)
            py.append(yi / _EPS_Z)
            ok.append(cross)
        PX, PY, OK = np.stack(px, 1), np.stack(py, 1), np.stack(ok, 1)
        with np.errstate(invalid='ignore'):
            x0 = np.where(OK, PX, np.inf).min(1)
            x1 = np.where(OK, PX, -np.inf).max(1)
            y0 = np.where(OK, PY, np.inf).min(1)
            y1 = np.where(OK, PY, -np.inf).max(1)
        keep = (x1 >= -1) & (x0 <= 1) & (y1 >= -1) & (y0 <= 1)
        if not keep.any():
            continue
        idx, x0, x1, y0, y1 = idx[keep], x0[keep], x1[keep], y0[keep], y1[keep]
        ix0 = np.clip(np.floor((np.maximum(x0, -1) + 1) * 0.5 * G), 0, G - 1).astype(np.int64)
        ix1 = np.clip(np.floor((np.minimum(x1, 1) + 1) * 0.5 * G), 0, G - 1).astype(np.int64)
        iy0 = np.clip(np.floor((np.maximum(y0, -1) + 1) * 0.5 * G), 0, G - 1).astype(np.int64)
        iy1 = np.clip(np.floor((np.minimum(y1, 1) + 1) * 0.5 * G), 0, G - 1).astype(np.int64)
        nx, ny = ix1 - ix0 + 1, iy1 - iy0 + 1
        n = nx * ny
        tot = int(n.sum())
        off = np.repeat(np.cumsum(n) - n, n)
        k = np.arange(tot) - off
        wx = np.repeat(nx, n)
        cx = np.repeat(ix0, n) + k % wx
        cy = np.repeat(iy0, n) + k // wx
        cells.append(f * G * G + cy * G + cx)
        tris.append(np.repeat(idx, n))
    if not cells:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    return np.concatenate(cells), np.concatenate(tris)


def _pairs_test(O, D, ray_cells, tri_cells, tri_ids, v0, e1, e2, near, t_best, i_best, chunk=PAIR_CHUNK):
    """Möller–Trumbore on every (ray, triangle) pair sharing a cell; keeps each ray's nearest hit beyond near.
    O: (3,) or (K, 3) ray origins; v0, e1, e2 the triangles' (corner, edges)."""
    if not len(tri_cells) or not len(ray_cells):
        return
    ordt = np.argsort(tri_cells, kind='stable')
    tc_s, tid_s = tri_cells[ordt], tri_ids[ordt]
    ordr = np.argsort(ray_cells, kind='stable')
    rc_s = ray_cells[ordr]
    uc, rstart, rcount = np.unique(rc_s, return_index=True, return_counts=True)
    ts = np.searchsorted(tc_s, uc, 'left')
    tn = np.searchsorted(tc_s, uc, 'right') - ts
    keep = tn > 0
    uc, rstart, rcount, ts, tn = uc[keep], rstart[keep], rcount[keep], ts[keep], tn[keep]
    npairs = rcount.astype(np.int64) * tn
    if not len(npairs):
        return
    per_origin = np.ndim(O) == 2
    i = 0
    cum = np.cumsum(npairs)
    while i < len(npairs):
        base = cum[i - 1] if i else 0
        j = int(np.searchsorted(cum, base + chunk, 'right'))
        j = max(j, i + 1)
        sel = slice(i, j)
        npc = npairs[sel]
        P = int(npc.sum())
        off = np.repeat(np.cumsum(npc) - npc, npc)
        k = np.arange(P, dtype=np.int64) - off
        tnr = np.repeat(tn[sel], npc)
        ray = ordr[np.repeat(rstart[sel], npc) + k // tnr]
        tri = tid_s[np.repeat(ts[sel], npc) + k % tnr]
        d = D[ray]
        a, b, c = v0[tri], e1[tri], e2[tri]
        o = O[ray] if per_origin else O
        p = np.cross(d, c)
        det = np.einsum('ij,ij->i', p, b)
        ok = np.abs(det) > 1e-14
        inv = 1.0 / np.where(ok, det, 1.0)
        tv = o - a
        u = np.einsum('ij,ij->i', tv, p) * inv
        q = np.cross(tv, b)
        v = np.einsum('ij,ij->i', d, q) * inv
        t = np.einsum('ij,ij->i', c, q) * inv
        ok &= (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9) & (t > near)
        if ok.any():
            ray, tri, t = ray[ok], tri[ok], t[ok]
            np.minimum.at(t_best, ray, t)
            win = t <= t_best[ray]
            i_best[ray[win]] = tri[win]
        i = j


class Caster:
    """Exact nearest hits of rays against a triangle set (vehicle frame)."""

    def __init__(self, tri):
        self.tri = np.ascontiguousarray(tri, np.float64).reshape(-1, 3, 3)
        self.v0 = self.tri[:, 0]
        self.e1 = self.tri[:, 1] - self.tri[:, 0]
        self.e2 = self.tri[:, 2] - self.tri[:, 0]

    def cast(self, O, D, near: float = 0.0, skip=None, grid: int = GRID):
        """Rays from one origin O (3,) along unit D (K, 3) -> (t (K,) inf where missed, triangle (K,) -1)."""
        D = np.asarray(D, np.float64).reshape(-1, 3)
        O = np.asarray(O, np.float64).reshape(3)
        K = len(D)
        t_best = np.full(K, np.inf)
        i_best = np.full(K, -1, np.int64)
        if not len(self.tri) or not K:
            return t_best, i_best
        ids = np.arange(len(self.tri)) if skip is None else np.flatnonzero(~np.asarray(skip, bool))
        if not len(ids):
            return t_best, i_best
        tcells, tloc = _tri_cells(self.tri[ids] - O, grid)
        _pairs_test(O, D, _ray_cells(D, grid), tcells, ids[tloc], self.v0, self.e1, self.e2, near, t_best, i_best)
        return t_best, i_best

    def blocked(self, P, s, near: float = 1e-3, cell: float = SHADOW_CELL_M, skip=None) -> np.ndarray:
        """Whether each point P (K, 3) sees along the unit direction s (3,) a triangle (parallel rays: the body's own
        shadow when s points at the sun)."""
        P = np.asarray(P, np.float64).reshape(-1, 3)
        K = len(P)
        out = np.zeros(K, bool)
        if not len(self.tri) or not K:
            return out
        s = np.asarray(s, float) / np.linalg.norm(s)
        a = np.cross(s, [0.0, 0.0, 1.0] if abs(s[2]) < 0.9 else [1.0, 0.0, 0.0])
        a /= np.linalg.norm(a)
        b = np.cross(s, a)
        ids = np.arange(len(self.tri)) if skip is None else np.flatnonzero(~np.asarray(skip, bool))
        T = self.tri[ids]
        tu, tv = T @ a, T @ b
        pu, pv = P @ a, P @ b
        lo_u = min(tu.min(), pu.min())
        lo_v = min(tv.min(), pv.min())
        nu = int((max(tu.max(), pu.max()) - lo_u) / cell) + 2
        iu0 = ((tu.min(1) - lo_u) / cell).astype(np.int64)
        iu1 = ((tu.max(1) - lo_u) / cell).astype(np.int64)
        iv0 = ((tv.min(1) - lo_v) / cell).astype(np.int64)
        iv1 = ((tv.max(1) - lo_v) / cell).astype(np.int64)
        # only triangles some part of which lies towards the sun of some point can shade it: all of them kept
        nx, ny = iu1 - iu0 + 1, iv1 - iv0 + 1
        n = nx * ny
        off = np.repeat(np.cumsum(n) - n, n)
        k = np.arange(int(n.sum())) - off
        cx = np.repeat(iu0, n) + k % np.repeat(nx, n)
        cy = np.repeat(iv0, n) + k // np.repeat(nx, n)
        tcell = cy * nu + cx
        pc = ((pv - lo_v) / cell).astype(np.int64) * nu + ((pu - lo_u) / cell).astype(np.int64)
        t_best = np.full(K, np.inf)
        i_best = np.full(K, -1, np.int64)
        D = np.broadcast_to(s, (K, 3))
        _pairs_test(P, np.ascontiguousarray(D), pc, tcell, ids[np.repeat(np.arange(len(ids)), n)], self.v0, self.e1,
                    self.e2, near, t_best, i_best)
        return np.isfinite(t_best)


# =========================================================================== the project's body as triangles
@dataclass
class EgoMesh:
    tri: np.ndarray                     # (M, 3, 3) float64, vehicle frame
    normal: np.ndarray                  # (M, 3) outward face normals
    corner_n: np.ndarray                # (M, 3, 3) smoothed corner normals
    mat: np.ndarray                     # (M,) uint8 index into MATERIALS
    owner: np.ndarray                   # (M,) int16: -1 body or rig part, else index into anchors (a housing)
    anchors: np.ndarray                 # (H, 3) the housings' anchors (a camera's lens, a sensor's optical centre)
    paint: str = '#b8bcc2'
    notes: list = field(default_factory=list)

    @property
    def albedo(self) -> np.ndarray:
        return albedo_table(self.paint)

    def own(self, pos, tol: float = 0.15) -> np.ndarray | None:
        """The triangles of the housing anchored at pos (left out of that camera's view), or None."""
        if not len(self.anchors):
            return None
        d = np.linalg.norm(self.anchors - np.asarray(pos, float), axis=1)
        k = int(np.argmin(d))                    # the nearest anchor only (core.housings.HousingSet.own)
        if d[k] > tol:
            return None
        return self.owner == k

    def arrays(self) -> dict:
        return {'tri': self.tri.astype(np.float32), 'normal': self.normal.astype(np.float32),
                'corner_n': self.corner_n.astype(np.float16), 'mat': self.mat.astype(np.uint8),
                'owner': self.owner.astype(np.int16), 'anchors': self.anchors.astype(np.float32)}

    @staticmethod
    def from_arrays(z, paint='#b8bcc2') -> 'EgoMesh':
        return EgoMesh(np.asarray(z['tri'], np.float64), np.asarray(z['normal'], np.float64),
                       np.asarray(z['corner_n'], np.float32), np.asarray(z['mat'], np.uint8),
                       np.asarray(z['owner'], np.int16), np.asarray(z['anchors'], np.float64).reshape(-1, 3), paint)


# =========================================================================== a camera's ego layer
@dataclass
class Layer:
    w: int
    h: int
    range: np.ndarray                   # (h, w) float32 metres along the pixel's ray; inf where the body is not
    tri: np.ndarray                     # (h, w) int32 triangle index, -1
    inside: np.ndarray                  # (h, w) bool: within the lens's field
    dirs: np.ndarray                    # (h·w, 3) float32 vehicle-frame unit rays
    pos: np.ndarray                     # (3,) the optical centre (vehicle frame)
    R: np.ndarray                       # (3, 3) body -> vehicle at this mount
    mount: tuple = (None, None)         # (deg, m) relative to the plan
    ms: float = 0.0                     # time the cast took

    @property
    def mask(self) -> np.ndarray:
        return np.isfinite(self.range)

    def points(self) -> np.ndarray:
        """(n, 3) the body's surface points of the masked pixels (vehicle frame), row by row."""
        m = self.mask.ravel()
        return self.pos + self.dirs[m].astype(np.float64) * self.range.ravel()[m, None]


def ego_layer(mesh: EgoMesh, caster: Caster | None, lens: dict, pose: dict, w: int, h: int, deg=None, m=None,
              grid: int = GRID) -> Layer:
    """The project's body as the camera sees it on an image of w × h at its pose turned by deg / moved by m."""
    t0 = time.perf_counter()
    caster = caster or Caster(mesh.tri)
    lens = scaled_lens(lens, w, h)
    u, v = pixel_grid(w, h)
    d, _, inside = lens_unproject(lens, u, v)
    R = mount_rotation(pose, deg)
    pos = mount_position(pose, m)
    dv = d @ W2O_of(R)
    t, ti = caster.cast(pos, dv, 1e-4, skip=mesh.own(mount_position(pose)), grid=grid)
    t = np.where(inside & (t > NEAR_M), t, np.inf)      # nearer: behind the protruding lens (core.synth's rule)
    ti = np.where(np.isfinite(t), ti, -1)
    return Layer(int(w), int(h), t.reshape(h, w).astype(np.float32), ti.reshape(h, w).astype(np.int32),
                 inside.reshape(h, w), dv.astype(np.float32), pos, R,
                 (None if deg is None else tuple(float(x) for x in deg),
                  None if m is None else tuple(float(x) for x in m)),
                 (time.perf_counter() - t0) * 1000.0)


def layer_normals(mesh: EgoMesh, layer: Layer) -> np.ndarray:
    """(n, 3) smoothed normals at the masked pixels (barycentric interpolation of the corner normals), turned to face
    the camera."""
    msk = layer.mask.ravel()
    ti = layer.tri.ravel()[msk]
    P = layer.points()
    T = mesh.tri[ti]
    v0, v1, v2 = T[:, 0], T[:, 1], T[:, 2]
    e1, e2, ep = v1 - v0, v2 - v0, P - v0
    d00 = np.einsum('ij,ij->i', e1, e1)
    d01 = np.einsum('ij,ij->i', e1, e2)
    d11 = np.einsum('ij,ij->i', e2, e2)
    d20 = np.einsum('ij,ij->i', ep, e1)
    d21 = np.einsum('ij,ij->i', ep, e2)
    den = d00 * d11 - d01 * d01
    den = np.where(np.abs(den) > 1e-18, den, 1.0)
    b1 = np.clip((d11 * d20 - d01 * d21) / den, 0, 1)
    b2 = np.clip((d00 * d21 - d01 * d20) / den, 0, 1)
    b0 = np.clip(1 - b1 - b2, 0, 1)
    cn = mesh.corner_n[ti].astype(np.float64)
    n = cn[:, 0] * b0[:, None] + cn[:, 1] * b1[:, None] + cn[:, 2] * b2[:, None]
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    dv = layer.dirs[msk].astype(np.float64)
    flip = np.einsum('ij,ij->i', n, dv) > 0
    n[flip] *= -1
    return n


# =========================================================================== light
def sun_angles(weather: dict | None) -> tuple[float, float, float]:
    """(altitude deg, azimuth deg, cloudiness 0-100) of a drive's CARLA weather: its parameters when recorded (the
    rig's mapped condition, run.json weather.params) or a preset's (CARLA_PRESETS)."""
    w = weather if isinstance(weather, dict) else {}
    p = w.get('params') if isinstance(w.get('params'), dict) else {}
    pre = CARLA_PRESETS.get(str(w.get('preset') or ''), (5, -1, 45))
    cloud = float(p.get('cloudiness', pre[0]))
    az = float(p.get('sun_azimuth_angle', pre[1]))
    alt = float(p.get('sun_altitude_angle', pre[2]))
    if az < 0 and az == -1:
        az = float(w.get('measured_azimuth', DEFAULT_AZIMUTH))
    return alt, az, cloud


def sun_world(alt_deg: float, az_deg: float) -> np.ndarray:
    """Unit vector towards the sun in a right-handed world (CARLA's mirrored: x, −y, z), from CARLA's sky light at
    yaw = azimuth, pitch = −altitude in Unreal's axes (see ASSUMPTIONS)."""
    a, z = math.radians(alt_deg), math.radians(az_deg)
    s_unreal = np.array([-math.cos(a) * math.cos(z), -math.cos(a) * math.sin(z), math.sin(a)])
    return np.array([s_unreal[0], -s_unreal[1], s_unreal[2]])


def to_vehicle(v_world, heading_rad: float) -> np.ndarray:
    """A world direction (right-handed, z up) in the vehicle frame of a vehicle heading heading_rad (from x, to y)."""
    c, s = math.cos(heading_rad), math.sin(heading_rad)
    x, y, z = v_world
    return np.array([c * x + s * y, -s * x + c * y, z])


def irradiance(alt_deg: float, cloud: float) -> tuple[float, float]:
    """(sun, sky) irradiance on a surface facing each (relative units): the sun's direct beam falls with cloud cover;
    the sky's diffuse light is about a fifth of the direct beam on a clear day (typical clear-sky diffuse fraction)
    and takes over under cloud. Night (sun below the horizon): a dim sky only."""
    c = min(max(cloud / 100.0, 0.0), 1.0)
    if alt_deg <= 0:
        return 0.0, 0.03
    sun = 1.0 * (1.0 - 0.85 * c)
    sky = 0.22 + 0.25 * c
    return sun, sky


def shade(mesh: EgoMesh, layer: Layer, sun_v: np.ndarray | None, alt_deg: float, cloud: float,
          shadow: np.ndarray | None = None, ground_albedo: float = 0.12) -> np.ndarray:
    """(n, 3) linear radiance (exposure 1) of the masked pixels of a layer."""
    msk = layer.mask.ravel()
    n = layer_normals(mesh, layer)
    mats = mesh.mat[layer.tri.ravel()[msk]]
    alb = mesh.albedo[mats].astype(np.float64)
    E_sun, E_sky = irradiance(alt_deg, cloud)
    up = n[:, 2]
    amb = E_sky * 0.5 * (1 + up) + (E_sun * max(math.sin(math.radians(alt_deg)), 0.0) + E_sky) * ground_albedo * \
        0.5 * (1 - up)
    direct = np.zeros(len(n))
    if sun_v is not None and E_sun > 0:
        direct = E_sun * np.clip(n @ sun_v, 0, None)
        if shadow is not None:
            direct = np.where(shadow, 0.0, direct)
    rad = alb * (amb + direct)[:, None]
    # glossy: the sky's Fresnel reflection (Schlick, F0 4 %; chrome 60 %), a brighter sky towards the zenith
    dv = layer.dirs[msk].astype(np.float64)
    cos_i = np.clip(-np.einsum('ij,ij->i', dv, n), 0, 1)
    f0 = np.where(mats == MAT['chrome'], 0.6, 0.04)
    F = f0 + (1 - f0) * (1 - cos_i) ** 5
    refl = dv + 2 * cos_i[:, None] * n
    sky_l = (E_sky + 0.3 * E_sun) * (0.35 + 0.65 * np.clip(refl[:, 2], 0, 1)) / math.pi * 1.6
    gloss = np.isin(mats, list(GLOSSY))
    rad += np.where(gloss, F * sky_l, 0.0)[:, None] * np.array([0.85, 0.92, 1.0])
    return rad


def exposure_gain(scene_linear_median: float, alt_deg: float, cloud: float) -> float:
    """Gain mapping the model's radiance to CARLA's image: middle grey lit as a horizontal surface comes out at the
    scene's median linear level."""
    E_sun, E_sky = irradiance(alt_deg, cloud)
    E_h = E_sun * max(math.sin(math.radians(alt_deg)), 0.0) + E_sky
    return float(scene_linear_median) / max(MID_GREY * E_h, 1e-6)


def scene_median(img_bgr: np.ndarray, exclude: np.ndarray) -> float | None:
    """Median linear luminance of an 8-bit BGR image outside exclude (and its black)."""
    g = img_bgr[..., 2] * 0.2126 + img_bgr[..., 1] * 0.7152 + img_bgr[..., 0] * 0.0722
    sel = ~exclude & (g > 2)
    if sel.sum() < 64:
        return None
    return float(np.median(srgb_to_linear(g[sel] / 255.0)))


TONE_KNEE = 0.6


def tone(x: np.ndarray, knee: float = TONE_KNEE) -> np.ndarray:
    """A luminance shoulder above the knee (the shader renderer's, ui.render_gl.tone_reference): a bright body keeps
    its shading and its edges instead of clipping to white."""
    x = np.maximum(np.asarray(x, float), 0)
    lum = np.array([0.2126, 0.7152, 0.0722])
    L = x @ lum
    s = np.ones_like(L)
    over = L > knee
    s[over] = (knee + (1 - knee) * (1 - np.exp(-(L[over] - knee) / (1 - knee)))) / L[over]
    x = x * s[:, None]
    m = x.max(axis=1)
    l2 = x @ lum
    t = np.where(m > 1, np.clip((m - 1) / np.maximum(m - l2, 1e-4), 0, 1), 0)
    return x + (l2[:, None] - x) * t[:, None]


def body_rgb8(rad: np.ndarray, gain: float) -> np.ndarray:
    """(n, 3) uint8 RGB from linear radiance and the exposure gain (with the tone shoulder)."""
    return np.round(linear_to_srgb(tone(rad * gain)) * 255.0).astype(np.uint8)


class ShadowCache:
    """The body's own shadow of a layer's pixels per heading bin (exact parallel rays, HEADING_STEP wide bins)."""

    def __init__(self, mesh: EgoMesh, caster: Caster, step: float = HEADING_STEP):
        self.mesh, self.caster, self.step = mesh, caster, float(step)
        self.cache = {}

    def get(self, layer: Layer, key, sun_world_v, heading_rad):
        b = int(round(math.degrees(heading_rad) / self.step))
        k = (key, b)
        hit = self.cache.get(k)
        if hit is None:
            sv = to_vehicle(sun_world_v, math.radians(b * self.step))
            P = layer.points()
            n = layer_normals(self.mesh, layer)
            hit = self.caster.blocked(P + n * 2e-3, sv, near=2e-3) if sv[2] > 0 else np.ones(len(P), bool)
            self.cache[k] = hit
            while len(self.cache) > 96:
                self.cache.pop(next(iter(self.cache)))
        return hit


# =========================================================================== one frame
def composite(img_bgr: np.ndarray, layer: Layer, body_bgr: np.ndarray, carla_car: np.ndarray,
              carla_range: np.ndarray | None = None, depth_test: bool = True):
    """(out, body pixels, holes): the image with the body painted where it is nearer than CARLA saw (CARLA's own car
    counts as nothing), and CARLA's car where the body does not cover it as holes (not filled here).
    body_bgr: (n, 3) uint8 BGR of the layer's masked pixels in row order."""
    h, w = layer.h, layer.w
    msk = layer.mask
    rng = layer.range
    body = msk.copy()
    if depth_test and carla_range is not None:
        cr = np.where((carla_range > 0) & ~carla_car, carla_range, np.inf)
        body &= rng <= cr + 0.02
    out = img_bgr.copy()
    full = np.zeros((h, w, 3), np.uint8)
    full[msk] = body_bgr
    out[body] = full[body]
    holes = carla_car & ~body & layer.inside
    return out, body, holes


def inpaint(img_bgr: np.ndarray, holes: np.ndarray, radius: float = 3.0) -> np.ndarray:
    import cv2
    if not holes.any():
        return img_bgr
    return cv2.inpaint(img_bgr, holes.astype(np.uint8) * 255, float(radius), cv2.INPAINT_TELEA)


def _png(a) -> bytes:
    import cv2
    ok, buf = cv2.imencode('.png', a)
    if not ok:
        raise RuntimeError('PNG encoding failed')
    return buf.tobytes()


class LiveEgo:
    """The live server's compositor (CARLA's Python: numpy and OpenCV only). apply() is called from the pool threads."""

    def __init__(self, z: dict, rig_doc: dict):
        self.meta = json.loads(bytes(np.asarray(z['meta'], np.uint8)).decode('utf-8'))
        if self.meta.get('schema') != LAYERS_SCHEMA:
            raise ValueError('not a Vantage ego layers file')
        self.mesh = EgoMesh.from_arrays({k[5:]: z[k] for k in z if k.startswith('mesh_')}, self.meta.get('paint',
                                                                                                         '#b8bcc2'))
        self.caster = Caster(self.mesh.tri)
        self.cams = {c['tag']: c for c in rig_doc.get('cameras', [])}
        self.lock = threading.RLock()
        self.layers = {}                      # (tag, mount) -> Layer
        self.planned = {}
        self.carla = {}                       # tag -> CARLA's own car at the planned mount (bool, render size)
        self.gain = {}
        self.turned = {}                      # (tag, mount) -> CARLA's car turned with the camera
        self.shadows = ShadowCache(self.mesh, self.caster, float(self.meta.get('heading_step_deg', HEADING_STEP)))
        self.inpaint_px = float(self.meta.get('inpaint_px', 3.0))
        self.mode = self.meta.get('carla_car', 'inpaint')
        self.use_shadows = bool(self.meta.get('shadows', True))
        self.counts = {}
        self.skipped = []
        for tag, c in self.cams.items():
            if f'{tag}__range' not in z:
                continue
            mc = self.meta['cameras'][tag]
            w, h = mc['size']
            if 'pose' in mc and (mc['pose'] != c['pose_vehicle'] or mc.get('lens') != c['lens'] or
                                 [w, h] != [int(c['image']['width']), int(c['image']['height'])]):
                self.skipped.append(tag)          # layers of another rig (a file left beside this one)
                continue
            lens = scaled_lens(c['lens'], w, h)
            u, v = pixel_grid(w, h)
            d, _, inside = lens_unproject(lens, u, v)
            R = mount_rotation(c['pose_vehicle'])
            L = Layer(w, h, np.asarray(z[f'{tag}__range'], np.float32), np.asarray(z[f'{tag}__tri'], np.int32),
                      inside.reshape(h, w), (d @ W2O_of(R)).astype(np.float32), mount_position(c['pose_vehicle']), R)
            self.planned[tag] = L
            self.layers[(tag, (None, None))] = L
            self.counts[tag] = {'frames': 0, 'ms': [], 'casts': 0, 'cast_ms': []}

    @staticmethod
    def load(path: str, rig_doc: dict) -> 'LiveEgo':
        with np.load(path, allow_pickle=False) as z:
            return LiveEgo({k: z[k] for k in z.files}, rig_doc)

    def body_png(self, tag) -> bytes:
        L = self.planned[tag]
        return _png(np.where(L.mask, 255, np.where(L.inside, 128, 0)).astype(np.uint8))

    def body_info(self, tag) -> dict:
        L = self.planned[tag]
        n = int(L.mask.sum())
        return {'pixels': n, 'share': round(n / max(int(L.inside.sum()), 1), 4), 'size': [L.w, L.h]}

    def set_carla(self, tag, mask):
        with self.lock:
            self.carla[tag] = np.asarray(mask, bool)
            self.turned = {k: v for k, v in self.turned.items() if k[0] != tag}

    def layer(self, tag, mount) -> Layer:
        key = (tag, _mount_key(mount))
        with self.lock:
            L = self.layers.get(key)
        if L is not None:
            return L
        c = self.cams[tag]
        P = self.planned[tag]
        deg, m = key[1]
        L = ego_layer(self.mesh, self.caster, c['lens'], c['pose_vehicle'], P.w, P.h, deg, m)
        with self.lock:
            self.layers[key] = L
            self.counts[tag]['casts'] += 1
            self.counts[tag]['cast_ms'].append(L.ms)
            for k in [k for k in self.layers if k[0] == tag and k[1] != (None, None) and k != key][:-3]:
                self.layers.pop(k)
        return L

    def carla_car(self, tag, mount, L: Layer):
        """CARLA's own car at this mount: the planned mask turned with the camera (a rotation's exact; see doc)."""
        base = self.carla.get(tag)
        if base is None:
            return np.zeros((L.h, L.w), bool)
        key = (tag, _mount_key(mount))
        if key[1][0] is None:
            return base
        with self.lock:
            hit = self.turned.get(key)
        if hit is not None:
            return hit
        P = self.planned[tag]
        lens = scaled_lens(self.cams[tag]['lens'], P.w, P.h)
        d_planned = L.dirs.astype(np.float64) @ W2O_of(P.R).T
        u, v, _, ok = lens_project(lens, d_planned)
        iu, iv = np.floor(u).astype(np.int64), np.floor(v).astype(np.int64)
        ok &= (iu >= 0) & (iu < P.w) & (iv >= 0) & (iv < P.h)
        out = np.zeros(P.w * P.h, bool)
        out[ok] = base[iv[ok], iu[ok]]
        out = out.reshape(P.h, P.w)
        with self.lock:
            self.turned[key] = out
        return out

    def apply(self, tag, img_bgr: np.ndarray, mount=None, ego_yaw_deg=None, weather: dict | None = None):
        """The frame (render size, BGR) with the project's body composited at the camera's mount."""
        if tag not in self.planned:
            return img_bgr
        t0 = time.perf_counter()
        L = self.layer(tag, mount)
        if img_bgr.shape[:2] != (L.h, L.w):
            return img_bgr
        car = self.carla_car(tag, mount, L)
        alt, az, cloud = sun_angles(weather)
        heading = -math.radians(float(ego_yaw_deg or 0.0))            # Unreal's yaw (left-handed) -> heading
        sunw = sun_world(alt, az)
        g = self.gain.get(tag)
        if g is None:
            md = scene_median(img_bgr, car | L.mask)
            g = exposure_gain(md if md is not None else 0.2, alt, cloud)
            with self.lock:
                self.gain[tag] = g
        shd = self.shadows.get(L, (tag, _mount_key(mount)), sunw, heading) if self.use_shadows and alt > 0 else None
        rad = shade(self.mesh, L, to_vehicle(sunw, heading) if alt > 0 else None, alt, cloud, shd)
        out, body, holes = composite(img_bgr, L, body_rgb8(rad, g)[:, ::-1], car, None, False)
        if holes.any():
            if self.mode == 'inpaint':
                out = inpaint(out, holes, self.inpaint_px)
            else:
                out[holes] = 0
        with self.lock:
            cn = self.counts[tag]
            cn['frames'] += 1
            cn['ms'].append((time.perf_counter() - t0) * 1000.0)
            del cn['ms'][:-500]
        return out

    def stats(self, tag) -> dict | None:
        cn = self.counts.get(tag)
        if cn is None:
            return None
        ms = np.array(cn['ms'] or [np.nan])
        return {'frames': cn['frames'], 'ms_median': None if not cn['ms'] else round(float(np.median(ms)), 2),
                'ms_p95': None if not cn['ms'] else round(float(np.percentile(ms, 95)), 2), 'casts': cn['casts'],
                'cast_ms': [round(x, 1) for x in cn['cast_ms'][-5:]], 'exposure_gain': self.gain.get(tag),
                'body': self.body_info(tag), 'carla_car_known': tag in self.carla, 'carla_car': self.mode}


def _mount_key(mount):
    if not mount:
        return (None, None)
    deg, m = mount
    return (None if deg is None else tuple(round(float(x), 6) for x in deg),
            None if m is None else tuple(round(float(x), 6) for x in m))
