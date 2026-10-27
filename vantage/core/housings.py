"""Sensor and camera housings as occluders: what every camera (and every lidar and radar) sees of the rig's own
hardware.

Solids. A lidar, radar, GNSS antenna or IMU housing is its library model's box or cylinder at its real pose
(core.sensors.body_box: size_mm, the optical centre optical_z_mm above the base; a cylinder for a spinning or
non-repetitive lidar and a GNSS antenna). A camera's housing is its module's body: a box of module.size_mm (width
across the image × height × length behind the lens) whose front face is centred on the camera's position, the lens's
entrance pupil, exactly as core.gnss places camera bodies for the sky check. margin_mm grows every solid on all sides
(brackets, cable glands, a sun hood). Hidden sensors and cameras, kinds switched off in the settings and poses that are
not finite do not block. HousingSettings.enabled = False restores the body + rig occluder alone, exactly.

Where they block. Every cube map the analysis engine hands out (core.occlusion) is the static body + rig map with the
housings composited in: an exact ray test at the centre of every texel inside each housing's angular footprint (its
bounding box clipped to the face's near plane and projected), the body and rig in front of a housing (or less than
EMBED_TOL_M behind its face) hiding it, tested exactly against the triangles where the map's depth leaves it in doubt.
The ground coverage, blind area, redundancy, density,
obstacles, the blind volume, perception range, synthetic captures, the sample dataset's images, the stitched and bowl
views, guidelines, the camera light, the installation check's captures and the rig health all look up those maps, so
they all see the housings; lidar and radar studies that use the engine's maps see them too. The housing anchored at a
map's own position (within occlusion.OWN_TOL_M: the camera's lens, the sensor's optical centre) is left out of it. A
camera inside another housing sees nothing (every ray starts inside the solid).

In view. The share of a camera's image a housing covers is taken on a grid of GRID_W pixel columns across the image
(rows in proportion), counting the pixels inside the image circle: a pixel counts for the nearest housing its ray
meets, unless a body or rig triangle lies in front of that hit or less than EMBED_TOL_M behind it (a housing set into
the body or behind a fascia does not show). It is a share of the image's pixels, not of a solid angle. Issues: a share
above WARN_SHARE (or any pixel in the middle ninth of the image) is a warning, above SERIOUS_SHARE serious; a camera
whose lens is inside another housing is critical.
"""
from __future__ import annotations

import copy
import csv
import io
import json
import math
import os
import threading
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass, field

import numpy as np

from . import sensors as S
from .mathutil import BODY_TO_OPTICAL, euler_zyx, rot_zyx
from .occlusion import EMBED_TOL_M, FACE_N, FACES, LIGHT, NEAR, OWN_TOL_M, texel_dirs

SCHEMA = 'vantage.housings/1'
FRAME = 'vehicle (ISO 8855, rear axle)'
KINDS = S.KINDS                      # lidar, radar, gnss, imu
WARN_SHARE = 0.02                    # of a camera's image
SERIOUS_SHARE = 0.10
GRID_W = 96                          # image columns sampled for the share in view
CAMERA_FALLBACK_MM = (25.0, 25.0, 44.0)
LIMITS = {'margin_mm': (0.0, 100.0)}
ASSUMPTIONS = (
    'Housings are the library models\' outer boxes and cylinders (sensors) and the camera modules\' boxes '
    '(width × height × length behind the lens, lens at the front face), grown by the margin: real housings have '
    'chamfers, windows, connectors and brackets that are not modelled. Cameras, lidars and radars see the housings '
    'through the same cube maps as the body (one texel ≈ 0.35° at the face centre, so shadow edges are quantised '
    'like the body\'s). A housing face less than 3 cm in front of a body or rig surface counts as set into it (behind '
    'a fascia, sunk into a panel) and does not show. The share of an image a housing covers is counted on a grid of '
    '96 pixel columns, inside the image circle, with the body and rig in front of a housing hiding it.')
LIMITS_TEXT = ('Only the vehicle, its rig and its own sensors and cameras block: cables, mounting hardware beyond the '
               'margin, other traffic and the environment are not modelled. The body\'s depth in the cube maps is '
               'each triangle\'s centroid distance; where that is in doubt next to a housing the texel is tested '
               'exactly, but a texel is one ray, so housing edges are as sharp as the 0.35° texels allow.')

# dark plastic and metal, per kind (synthetic images)
RGB = {'lidar': (0.16, 0.17, 0.18), 'radar': (0.09, 0.09, 0.10), 'gnss': (0.20, 0.20, 0.21), 'imu': (0.18, 0.19, 0.20),
       'camera': (0.11, 0.11, 0.12)}
GREY = {'lidar': 0.17, 'radar': 0.09, 'gnss': 0.20, 'imu': 0.19, 'camera': 0.11}


# =========================================================================== settings
def _flag(v, default):
    if isinstance(v, (bool, int)) and not isinstance(v, float):
        return bool(v)
    if isinstance(v, str):
        return v.strip().lower() in ('1', 'true', 'yes', 'on')
    return default


@dataclass
class HousingSettings:
    enabled: bool = True             # housings block cameras, lidars and radars (False: the body and rig only)
    cameras: bool = True             # camera modules block the other cameras
    kinds: tuple = KINDS             # sensor kinds whose housings block
    margin_mm: float = 0.0           # every solid grown by this on all sides

    def to_dict(self):
        d = asdict(self)
        d['kinds'] = list(self.kinds)
        return d

    @staticmethod
    def from_dict(d: dict | None) -> 'HousingSettings':
        s = HousingSettings()
        d = d if isinstance(d, dict) else {}
        s.enabled = _flag(d.get('enabled'), s.enabled)
        s.cameras = _flag(d.get('cameras'), s.cameras)
        k = d.get('kinds')
        if isinstance(k, str):
            k = [x.strip() for x in k.split(',')]
        if isinstance(k, (list, tuple)):
            want = {str(x).strip().lower() for x in k if isinstance(x, str)}
            s.kinds = tuple(x for x in KINDS if x in want)
        for name, (lo, hi) in LIMITS.items():
            if name in d:
                try:
                    v = float(d[name])
                except (TypeError, ValueError):
                    continue
                if math.isfinite(v):
                    setattr(s, name, min(max(v, lo), hi))
        return s

    def key(self) -> tuple:
        return (self.enabled, self.cameras, self.kinds, round(self.margin_mm, 6))


def settings_of(project) -> HousingSettings:
    return HousingSettings.from_dict(getattr(getattr(project, 'analysis', None), 'housings', None))


# =========================================================================== solids
@dataclass(eq=False)
class Housing:
    kind: str                        # 'sensor' | 'camera'
    uid: str
    tag: str
    name: str                        # 'lidar L1', 'camera F' (running text)
    what: str                        # lidar | radar | gnss | imu | camera
    centre: np.ndarray               # (3,) vehicle frame, metres
    half: np.ndarray                 # (3,) half sizes along the housing's axes; a cylinder: radius, radius, half height
    R: np.ndarray                    # (3, 3) housing -> vehicle
    round: bool                      # a cylinder about the housing's z axis
    anchor: np.ndarray               # (3,) the sensor's optical centre or the camera's lens
    slot: int = 0                    # a camera's colour slot
    key: tuple = ()

    @property
    def radius(self) -> float:
        """Radius of the bounding sphere about the centre."""
        return float(math.hypot(self.half[0], self.half[2]) if self.round else np.linalg.norm(self.half))

    @property
    def label(self) -> str:
        """'lidar L1's housing', 'camera F's module'."""
        return f"{self.name}'s module" if self.what == 'camera' else f"{self.name}'s housing"


def _finite(o, attrs=('x', 'y', 'z', 'yaw', 'pitch', 'roll')) -> bool:
    try:
        return all(math.isfinite(float(getattr(o, a))) for a in attrs)
    except (TypeError, ValueError):
        return False


def _tup(v) -> tuple:
    """A model's size as a key (a hand-edited library entry may hold anything)."""
    try:
        return tuple(v)
    except TypeError:
        return (repr(v),)


def _size(v, fallback) -> tuple:
    try:
        out = tuple(max(float(x), 1.0) / 1000.0 for x in v)
        if len(out) == 3 and all(math.isfinite(x) for x in out):
            return out
    except (TypeError, ValueError):
        pass
    return tuple(x / 1000.0 for x in fallback)


def _mk(kind, uid, tag, name, what, c, half, R, rnd, anchor, slot=0) -> Housing:
    c, half, R, anchor = (np.asarray(a, float) for a in (c, half, R, anchor))
    r6 = lambda a: tuple(np.round(np.asarray(a, float).ravel(), 6).tolist())       # noqa: E731
    return Housing(kind, uid, tag, name, what, c, half, R, bool(rnd), anchor, int(slot),
                   (kind, uid, tag, what, r6(c), r6(half), r6(R), bool(rnd), r6(anchor)))


def solids(project, variant=None, settings: HousingSettings | None = None) -> list[Housing]:
    """The housing solids of a variant (the active one by default): its visible sensors of the enabled kinds, then
    its visible cameras' modules (when cameras is on); [] when the settings switch housings off."""
    st = settings or settings_of(project)
    if not st.enabled:
        return []
    v = variant or project.variant
    g = st.margin_mm / 1000.0
    out = []
    for s in v.sensors:
        if not s.visible or s.kind not in st.kinds or not _finite(s):
            continue
        m = project.sensor_model_for(s)
        try:
            c, half, R, rnd = S.body_box(s, m)
        except (TypeError, ValueError):
            continue
        if not (np.isfinite(c).all() and np.isfinite(half).all()):
            continue
        out.append(_mk('sensor', s.uid, s.tag, f'{S.KIND_NOUN.get(s.kind, s.kind)} {s.tag}', s.kind, c, half + g, R,
                       rnd, s.pos()))
    if st.cameras:
        for cam in v.cameras:
            if not cam.visible or not _finite(cam):
                continue
            w, h, length = _size(project.module_for(cam).body_mm, CAMERA_FALLBACK_MM)   # typical where not given
            R = rot_zyx(cam.yaw, cam.pitch, cam.roll)
            c = cam.pos() - R[:, 0] * length / 2               # the module behind its lens (core.gnss._housings)
            out.append(_mk('camera', cam.uid, cam.tag, f'camera {cam.tag}', 'camera', c,
                           np.array([length / 2, w / 2, h / 2]) + g, R, False, cam.pos(), cam.slot))
    return out


def key(project, variant=None, settings: HousingSettings | None = None):
    """A cheap key of everything the solids depend on (settings, poses, model and module sizes), None when housings
    are off or there are none: the engine's vehicle geometry carries no housings then."""
    st = settings or settings_of(project)
    if not st.enabled:
        return None
    v = variant or project.variant
    items, rest = [], []
    for s in v.sensors:
        if s.visible and s.kind in st.kinds:
            m = project.sensor_model_for(s)
            items.append(('s', s.uid, s.tag, s.kind, s.x, s.y, s.z, s.yaw, s.pitch, s.roll, _tup(m.size_mm),
                          m.optical_z_mm, m.scan, m.kind))
        elif s.visible:
            rest.append((s.uid, s.x, s.y, s.z))
    for c in v.cameras:
        if c.visible and st.cameras:
            items.append(('c', c.uid, c.tag, c.slot, c.x, c.y, c.z, c.yaw, c.pitch, c.roll,
                          _tup(project.module_for(c).body_mm)))
        elif c.visible:
            rest.append((c.uid, c.x, c.y, c.z))
    return (st.key(), tuple(items), tuple(rest)) if items else None      # rest: anchors only (own housings)


def anchors(project, variant=None) -> list:
    """[(uid, position)] of every visible sensor (optical centre) and camera (lens) with a finite pose, whether or not
    its housing blocks: a cube map at one of them is that object's own."""
    v = variant or project.variant
    return [(o.uid, o.pos().astype(float)) for o in list(v.sensors) + list(v.cameras) if o.visible and _finite(o)]


class HousingSet:
    """The solids of one variant, as the occlusion cube cache uses them (core.occlusion.HousedOccluder)."""

    def __init__(self, solids_: list[Housing], key_, anchors_=None):
        self.solids = list(solids_)
        self.key = key_
        a = anchors_ if anchors_ is not None else [(h.uid, h.anchor) for h in self.solids]
        self.anchor_uids = [u for u, _ in a]
        self.anchors = np.array([q for _, q in a], float).reshape(-1, 3)
        self._index = {h.uid: i for i, h in enumerate(self.solids)}

    def own(self, pos) -> int:
        """Index of the housing of the object anchored at pos (the nearest sensor or camera within OWN_TOL_M, its
        housing blocking or not), -1 for none."""
        if not len(self.anchors):
            return -1
        d = np.linalg.norm(self.anchors - np.asarray(pos, float), axis=1)
        k = int(np.argmin(d))
        return self._index.get(self.anchor_uids[k], -1) if d[k] <= OWN_TOL_M else -1

    def around(self, pos) -> tuple:
        """Indices of the housings a cube map at pos sees: all but its own."""
        own = self.own(pos)
        return tuple(i for i in range(len(self.solids)) if i != own)

    def layer(self, h: Housing, cube, geom=None) -> list:
        """Housing h's layer for a cube map (core.occlusion.CubeDepth) of the body and rig geometry geom (an
        OccluderGeom; None: no body test)."""
        if getattr(geom, 'tri', None) is None:
            return layer(h, cube.pos, cube.depth.shape[1])
        return layer(h, cube.pos, cube.depth.shape[1], cube.depth, tri_info(geom))


def tri_info(geom) -> tuple:
    """(triangles, their lower and upper bounds, the largest centroid-to-corner distance) of an occluder, cached on
    it. The last bounds how far a cube map's depth can lie from the surface the texel's ray meets (the depth is exact
    on the triangle drawn there, but at a silhouette that triangle can be another one within a texel)."""
    info = getattr(geom, '_housing_tri_info', None)
    if info is None:
        tri = np.asarray(geom.tri, float)
        r = float(np.linalg.norm(tri - tri.mean(axis=1, keepdims=True), axis=2).max()) if len(tri) else 0.0
        info = (tri, tri.min(axis=1), tri.max(axis=1), r)
        try:
            geom._housing_tri_info = info
        except AttributeError:
            pass
    return info


_SETS = OrderedDict()
_lock = threading.Lock()


def housing_set(project, variant=None, settings: HousingSettings | None = None, key_=None) -> HousingSet | None:
    """The variant's HousingSet (cached on key()), None when housings are off or there are none."""
    st = settings or settings_of(project)
    k = key_ if key_ is not None else key(project, variant, st)
    if k is None:
        return None
    with _lock:
        hs = _SETS.get(k)
        if hs is not None:
            _SETS.move_to_end(k)
            return hs
    hs = HousingSet(solids(project, variant, st), k, anchors(project, variant))
    if not hs.solids:
        return None
    with _lock:
        _SETS[k] = hs
        while len(_SETS) > 16:
            _SETS.popitem(last=False)
    return hs


# =========================================================================== exact ray tests
def _slab(o, d, half):
    """Per-axis entry and exit distances of rays o + t d through |x_k| <= half_k: (lo (K, n), hi (K, n))."""
    with np.errstate(divide='ignore', invalid='ignore'):
        inv = 1.0 / d
        t1 = (-half - o) * inv
        t2 = (half - o) * inv
    lo = np.fmin(t1, t2)                   # nan only for a ray grazing the slab's face: inside it
    hi = np.fmax(t1, t2)
    return np.where(np.isnan(lo), -np.inf, lo), np.where(np.isnan(hi), np.inf, hi)


def ray(O, D, h: Housing, normals: bool = False):
    """Distance along rays O + t D (O (3,) or (K, 3); D (K, 3)) to housing h, inf for a miss, 0 for a ray starting
    inside (the same maths as core.gnss._ray_housing). normals: also the outward unit normal (K, 3) at the hit in the
    vehicle frame (for a ray starting inside: against the ray)."""
    D = np.atleast_2d(np.asarray(D, float))
    o = (np.asarray(O, float) - h.centre) @ h.R
    d = D @ h.R
    o = np.broadcast_to(o, d.shape)
    if h.round:
        lz, hz = _slab(o[:, 2:], d[:, 2:], h.half[2:])
        lz, hz = lz[:, 0], hz[:, 0]
        r = h.half[0]
        a = d[:, 0] ** 2 + d[:, 1] ** 2
        b = 2 * (o[:, 0] * d[:, 0] + o[:, 1] * d[:, 1])
        cc = o[:, 0] ** 2 + o[:, 1] ** 2 - r * r
        disc = b * b - 4 * a * cc
        flat = a < 1e-12
        sq = np.sqrt(np.maximum(disc, 0.0))
        with np.errstate(divide='ignore', invalid='ignore'):
            s_in = np.where(flat, np.where(cc <= 0, -np.inf, np.inf), (-b - sq) / (2 * a))
            s_out = np.where(flat, np.where(cc <= 0, np.inf, -np.inf), (-b + sq) / (2 * a))
        s_in = np.where(~flat & (disc < 0), np.inf, s_in)
        s_out = np.where(~flat & (disc < 0), -np.inf, s_out)
        tin, tout = np.maximum(lz, s_in), np.minimum(hz, s_out)
    else:
        lo, hi = _slab(o, d, h.half)
        tin, tout = lo.max(axis=1), hi.min(axis=1)
    hit = (tout >= np.maximum(tin, 0.0)) & (tout > 1e-9)
    t = np.where(hit, np.maximum(tin, 0.0), np.inf)
    if not normals:
        return t
    nh = np.zeros_like(d)
    rows = np.arange(len(d))
    if h.round:
        cap = lz >= s_in
        nh[:, 2] = np.where(cap, -np.sign(d[:, 2]), 0.0)
        tt = np.where(np.isfinite(t), t, 0.0)
        side = ~cap
        nh[side, 0] = (o[side, 0] + tt[side] * d[side, 0]) / r
        nh[side, 1] = (o[side, 1] + tt[side] * d[side, 1]) / r
    else:
        ax = np.argmax(lo, axis=1)
        nh[rows, ax] = -np.sign(d[rows, ax])
    inside = hit & (tin < 0)
    nh[inside] = -d[inside]
    nh /= np.maximum(np.linalg.norm(nh, axis=1, keepdims=True), 1e-12)
    return t, nh @ h.R.T


def contains(h: Housing, p, tol: float = 0.0) -> bool:
    """Whether point p lies inside housing h grown by tol (negative: shrunk)."""
    o = (np.asarray(p, float) - h.centre) @ h.R
    if h.round:
        return bool(abs(o[2]) <= h.half[2] + tol and math.hypot(o[0], o[1]) <= h.half[0] + tol)
    return bool((np.abs(o) <= h.half + tol).all())


def tone(shade) -> np.ndarray:
    """Brightness factor (0.55 .. 2.15) of a housing's base colour (GREY, RGB) from its cube-map shade."""
    return 0.55 + 1.6 * np.clip((np.asarray(shade, float) - 22.0) / 82.0, 0.0, 1.0)


def shade_of(normals: np.ndarray) -> np.ndarray:
    """Shade (uint8, as the cube maps store it) of dark plastic and metal lit by occlusion.LIGHT: 22..104, darker
    than the body's 70..245."""
    s = 22 + 70 * np.clip(normals @ LIGHT, 0, 1) + 12 * np.clip(normals[:, 2], 0, 1)
    return np.clip(s, 1, 255).astype(np.uint8)


_SIGNS = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)
_EDGES = [(i, j) for i in range(8) for j in range(i + 1, 8) if bin(i ^ j).count('1') == 1]


def corners(h: Housing) -> np.ndarray:
    """The 8 corners (vehicle frame) of the housing's box (a cylinder's bounding box)."""
    return h.centre + (_SIGNS * h.half) @ h.R.T


def layer(h: Housing, pos, n: int = FACE_N, static=None, tris: tuple | None = None) -> list:
    """The texels of a cube map at pos (n × n per face) whose centre ray meets housing h: [(face, row0, col0, depth
    (rows, cols) float32 metres, inf for a miss, shade (rows, cols) uint8)], one block per face it reaches. The block
    covers the housing's bounding box clipped to the face's near plane and projected, cropped to the texels hit.
    static (the body map's depth (6, n, n)) and tris (tri_info of its occluder): a texel whose body depth is within
    the largest triangle radius (a bound on the depth's error at silhouettes) of the hit, or nearer, is tested exactly
    against the triangles, and left out when the body lies in front of the hit or less than EMBED_TOL_M behind it."""
    pos = np.asarray(pos, float)
    P = corners(h) - pos
    inside = contains(h, pos)
    out = []
    for f, (fw, rt, up) in enumerate(FACES):
        z, x, y = P @ fw, P @ rt, P @ up
        if inside:
            x0, x1, y0, y1 = 0, n - 1, 0, n - 1
        else:
            keep = z >= NEAR
            if not keep.any():
                continue
            us, vs = list(x[keep] / z[keep]), list(y[keep] / z[keep])
            for i, j in _EDGES:                         # where the box's edges cross the near plane
                if keep[i] != keep[j]:
                    s = (NEAR - z[i]) / (z[j] - z[i])
                    us.append((x[i] + s * (x[j] - x[i])) / NEAR)
                    vs.append((y[i] + s * (y[j] - y[i])) / NEAR)
            u0, u1, v0, v1 = min(us), max(us), min(vs), max(vs)
            if u1 < -1 or u0 > 1 or v1 < -1 or v0 > 1:
                continue
            x0 = int(min(max(math.floor((max(u0, -1.0) + 1) * 0.5 * n), 0), n - 1))
            x1 = int(min(max(math.floor((min(u1, 1.0) + 1) * 0.5 * n), 0), n - 1))
            y0 = int(min(max(math.floor((1 - min(v1, 1.0)) * 0.5 * n), 0), n - 1))
            y1 = int(min(max(math.floor((1 - max(v0, -1.0)) * 0.5 * n), 0), n - 1))
        D = texel_dirs(f, y0, y1, x0, x1, n)
        t, nrm = ray(pos, D.reshape(-1, 3), h, normals=True)
        t = t.reshape(D.shape[:2])
        hit = np.isfinite(t)
        if hit.any() and static is not None and tris is not None and len(tris[0]):
            sd = static[f, y0:y1 + 1, x0:x1 + 1]
            with np.errstate(invalid='ignore'):
                doubt = hit & (t > 0) & (sd < t + EMBED_TOL_M + tris[3])
            if doubt.any():
                rr, cc = np.nonzero(doubt)
                front = _body_before(tris, pos, D[rr, cc], t[rr, cc] + EMBED_TOL_M, (rr // 16) * 64 + cc // 16)
                t[rr[front], cc[front]] = np.inf
                hit = np.isfinite(t)
        if not hit.any():
            continue
        rows, cols = np.flatnonzero(hit.any(axis=1)), np.flatnonzero(hit.any(axis=0))
        r0, r1, c0, c1 = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
        sh = np.where(t > 0, shade_of(nrm).reshape(t.shape), 12).astype(np.uint8)     # 12: the inside of a solid
        tt = t[r0:r1, c0:c1]
        out.append((f, y0 + int(r0), x0 + int(c0), tt.astype(np.float32),
                    np.where(np.isfinite(tt), sh[r0:r1, c0:c1], 0).astype(np.uint8)))
    return out


def nearest(project, pos, dirs, variant=None, settings=None, exclude_own=True):
    """The nearest housing along rays from pos (dirs (K, 3)): (t (K,) metres, housing per ray or None). The housing
    anchored at pos is left out (as a cube map at pos leaves it out) unless exclude_own is False. The body is not
    tested (the assistant asks this for a ray the cube map already found blocked)."""
    D = np.atleast_2d(np.asarray(dirs, float))
    D = D / np.maximum(np.linalg.norm(D, axis=1, keepdims=True), 1e-12)
    t = np.full(len(D), np.inf)
    who = [None] * len(D)
    hs = housing_set(project, variant, settings)
    if hs is None:
        return t, who
    skip = hs.own(pos) if exclude_own else -1
    for i, h in enumerate(hs.solids):
        if i == skip:
            continue
        th = ray(pos, D, h)
        for k in np.flatnonzero(th < t):
            t[k] = th[k]
            who[k] = h
    return t, who


# =========================================================================== body and rig in front of a housing
def _body_before(tris, pos, D, tmax, group=None) -> np.ndarray:
    """Whether rays pos + t D (D unit, (K, 3)) meet a triangle at t < tmax (K,) (tris: tri_info of the occluder):
    Möller–Trumbore with the origin fixed, per group of neighbouring rays (labels (K,), e.g. texel tiles) on the
    triangles whose bounds overlap that group's bounds."""
    K = len(D)
    out = np.zeros(K, bool)
    if not K or tris is None or not len(tris[0]):
        return out
    tri, tmin, tmaxb = tris[:3]
    if group is None:
        return _body_before_one(tri, tmin, tmaxb, pos, D, tmax)
    g = np.asarray(group)
    order = np.argsort(g, kind='stable')
    cut = np.flatnonzero(np.diff(g[order])) + 1
    for idx in np.split(order, cut):
        out[idx] = _body_before_one(tri, tmin, tmaxb, pos, D[idx], tmax[idx])
    return out


def _body_before_one(tri, tmin, tmaxb, pos, D, tmax) -> np.ndarray:
    K = len(D)
    out = np.zeros(K, bool)
    ends = pos + D * tmax[:, None]
    lo = np.minimum(ends.min(axis=0), pos) - 0.01
    hi = np.maximum(ends.max(axis=0), pos) + 0.01
    sel = ((tmaxb >= lo) & (tmin <= hi)).all(axis=1)
    q = tri[sel] - pos
    if not len(q):
        return out
    v0, e1, e2 = q[:, 0], q[:, 1] - q[:, 0], q[:, 2] - q[:, 0]
    N = np.cross(e2, e1)
    W = np.cross(e2, -v0)
    Q = np.cross(-v0, e1)
    c = np.einsum('ij,ij->i', e2, Q)
    rows = max(1, 1_000_000 // len(q))
    for s in range(0, K, rows):
        d = D[s:s + rows]
        det = d @ N.T
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            inv = 1.0 / det
            u = (d @ W.T) * inv
            v = (d @ Q.T) * inv
            t = c[None, :] * inv
            ok = (np.abs(det) > 1e-15) & (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9) & (t > 1e-6) & \
                 (t < tmax[s:s + rows, None])
        out[s:s + rows] = ok.any(axis=1)
    return out


# =========================================================================== what a camera sees of the housings
@dataclass
class InView:
    """The housings in one camera's image."""
    uid: str
    tag: str
    shares: list = field(default_factory=list)     # [{uid, tag, name, what, share, centre, where, nearest_m}]
    inside: list = field(default_factory=list)     # names of the housings the lens is inside
    pixels: int = 0                                # grid pixels inside the image circle
    mask: np.ndarray | None = None                 # (rows, cols) int16: index into shares, -1 none, -2 off the image

    @property
    def total(self) -> float:
        return float(sum(s['share'] for s in self.shares))


def _where(u, v) -> str:
    """Where in the image (u, v in 0..1 from the top left): 'upper left', 'top centre', 'centre' ..."""
    hz = 'left' if u < 1 / 3 else 'right' if u > 2 / 3 else ''
    vt = 'upper' if v < 1 / 3 else 'lower' if v > 2 / 3 else ''
    if vt and hz:
        return f'{vt} {hz}'
    if vt:
        return 'top centre' if vt == 'upper' else 'bottom centre'
    return f'{hz} side' if hz else 'centre'


_VIEWS = OrderedDict()


def _cam_key(project, cam) -> tuple:
    calib = cam.calib
    return (cam.uid, cam.tag, cam.x, cam.y, cam.z, cam.yaw, cam.pitch, cam.roll, cam.module_id, cam.lens_id,
            json.dumps(calib, sort_keys=True, default=str) if calib else '',
            json.dumps(project.module_for(cam).to_dict(), sort_keys=True, default=str),
            json.dumps(project.nominal_lens_for(cam).to_dict(), sort_keys=True, default=str))


def view_shares(project, cam, vg=None, settings: HousingSettings | None = None, variant=None) -> InView:
    """The housings in camera cam's image (cached on the camera, its lens, the housings and the body geometry).
    vg: the engine's vehicle geometry (its body and rig hide housings behind them); None: no body."""
    from .lens import resolve
    st = settings or settings_of(project)
    v = variant or project.variant
    hk = key(project, v, st)
    if hk is None or not _finite(cam):
        return InView(cam.uid, cam.tag)
    ck = (_cam_key(project, cam), hk, getattr(vg, 'version', None), GRID_W)
    with _lock:
        hit = _VIEWS.get(ck)
        if hit is not None:
            _VIEWS.move_to_end(ck)
            return hit
    hs = housing_set(project, v, st, hk)
    others = [h for h in (hs.solids if hs is not None else []) if h.uid != cam.uid]
    intr = resolve(project.module_for(cam), project.lens_for(cam))
    gw = GRID_W
    gh = max(8, int(round(GRID_W * intr.height / max(intr.width, 1))))
    us = (np.arange(gw) + 0.5) / gw * intr.width
    vs = (np.arange(gh) + 0.5) / gh * intr.height
    U, V = np.meshgrid(us, vs)
    d_opt, th = intr.unproject(U.ravel(), V.ravel())
    valid = np.isfinite(th) & (th <= intr.max_theta) & np.isfinite(d_opt).all(axis=1)
    R = rot_zyx(cam.yaw, cam.pitch, cam.roll)
    dw = d_opt @ (BODY_TO_OPTICAL @ R.T)                  # optical -> vehicle
    dw /= np.maximum(np.linalg.norm(dw, axis=1, keepdims=True), 1e-12)
    pos = cam.pos().astype(float)
    best = np.full(len(dw), np.inf)
    who = np.full(len(dw), -1, np.int32)
    for k, h in enumerate(others):
        w = h.centre - pos
        dist = float(np.linalg.norm(w))
        rho = h.radius
        if dist <= rho:
            cand = np.flatnonzero(valid)
        else:
            cosmin = math.cos(min(math.asin(min(rho / dist, 1.0)), math.pi))
            cand = np.flatnonzero(valid & (dw @ (w / dist) >= cosmin - 1e-9))
        if not len(cand):
            continue
        t = ray(pos, dw[cand], h)
        closer = t < best[cand]
        best[cand[closer]] = t[closer]
        who[cand[closer]] = k
    occ = getattr(vg, 'occ', None)
    hitk = np.flatnonzero(who >= 0)
    if len(hitk) and getattr(occ, 'tri', None) is not None:
        col, row = hitk % gw, hitk // gw                      # per housing, per 8 x 8 tile of the grid
        hidden = _body_before(tri_info(getattr(occ, 'base', occ)), pos, dw[hitk], best[hitk] + EMBED_TOL_M,
                              who[hitk] * 4096 + (row // 8) * 64 + col // 8)
        who[hitk[hidden]] = -1
    npx = int(valid.sum())
    rows = []
    mask = np.where(valid, -1, -2).astype(np.int16)
    uu, vv = U.ravel() / intr.width, V.ravel() / intr.height
    centre = (np.abs(uu - 0.5) <= 1 / 6) & (np.abs(vv - 0.5) <= 1 / 6)
    for k in np.unique(who[who >= 0]):
        m = who == k
        h = others[int(k)]
        rows.append({'uid': h.uid, 'tag': h.tag, 'name': h.name, 'what': h.what, 'label': h.label,
                     'share': float(m.sum()) / max(npx, 1), 'centre': bool((m & centre).any()),
                     'where': _where(float(uu[m].mean()), float(vv[m].mean())), 'nearest_m': float(best[m].min()),
                     '_k': int(k)})
    rows.sort(key=lambda r: (-r['share'], r['tag']))
    for i, r in enumerate(rows):
        mask[who == r.pop('_k')] = i
    out = InView(cam.uid, cam.tag, rows, [h.name for h in others if contains(h, pos, -0.001)], npx,
                 mask.reshape(gh, gw))
    with _lock:
        _VIEWS[ck] = out
        while len(_VIEWS) > 96:
            _VIEWS.popitem(last=False)
    return out


def in_view_text(iv: InView, limit: int = 4) -> tuple[str, str | None]:
    """('L1 6.1 %, RD2 0.4 %' or 'none' or 'inside RD1's housing', severity or None) for the Inspector."""
    if iv.inside:
        return f'inside {iv.inside[0]}: no view', 'critical'
    shown = [s for s in iv.shares if s['share'] >= 0.0005]
    if not shown:
        return 'none', None
    txt = ', '.join(f"{s['tag']} {s['share'] * 100:.1f} %" for s in shown[:limit])
    if len(shown) > limit:
        txt += f', +{len(shown) - limit}'
    sev = [_severity(s) for s in shown]
    return txt, 'serious' if 'serious' in sev else 'warning' if 'warning' in sev else None


def _severity(s) -> str | None:
    if s['share'] > SERIOUS_SHARE:
        return 'serious'
    if s['share'] > WARN_SHARE or s['centre']:
        return 'warning'
    return None


def _pct(x) -> str:
    p = x * 100
    return f'{p:.0f} %' if p >= 9.5 else f'{p:.1f} %'


def issues(project, vg=None) -> list:
    """Per camera of the active variant: a housing covering more than WARN_SHARE of its image or any of its centre
    (warning), more than SERIOUS_SHARE (serious); a camera whose lens is inside another housing (critical). Cached
    with view_shares (a few ms per camera when something moved)."""
    from .rules import Issue
    st = settings_of(project)
    v = project.variant
    if key(project, v, st) is None:
        return []
    out = []
    for cam in v.cameras:
        if not cam.visible or not _finite(cam):
            continue
        iv = view_shares(project, cam, vg, st, v)
        if iv.inside:
            out.append(Issue('critical', f'Camera {cam.tag} sits inside {iv.inside[0]}\'s housing: it sees nothing. '
                                         f'Move the camera or the housing apart.', cam.tag, [cam.uid],
                             where=(cam.x, cam.y)))
            continue
        for s in iv.shares:
            sev = _severity(s)
            if sev is None:
                continue
            extra = ', including the image centre' if s['centre'] and s['where'] != 'centre' else ''
            out.append(Issue(sev, f"Camera {cam.tag}: {s['label']} covers {_pct(s['share'])} of its image, "
                                  f"{s['where']}{extra}.", f"{cam.tag} · {s['tag']}", [cam.uid, s['uid']],
                             where=(cam.x, cam.y)))
    return out


def clear_cache():
    with _lock:
        _SETS.clear()
        _VIEWS.clear()


# =========================================================================== the report
@dataclass
class HousingsResult:
    variant_uid: str
    variant: str
    settings: HousingSettings
    solids: list                       # [Housing]
    cameras: list                      # [InView] per visible camera
    coverage: dict | None = None       # {'on': {...}, 'off': {...}} ground coverage with housings on and off
    ms: float = 0.0


def _project_for(project, variant):
    """The project with variant active (a shallow copy when it is not the active one)."""
    if variant is None or variant is project.variant:
        return project
    q = copy.copy(project)
    q.variants = list(project.variants) + ([] if any(v is variant for v in project.variants) else [variant])
    q.active_uid = variant.uid
    return q


def _with_settings(project, st: HousingSettings):
    q = copy.copy(project)
    q.analysis = copy.copy(project.analysis)
    q.analysis.housings = st.to_dict()
    return q


def _coverage_doc(res, v) -> dict:
    tags = {c.uid: c.tag for c in v.cameras}
    m = res.metrics
    return {'blind_m2': float(m['blind_m2']), 'covered_pct': float(m['covered_pct']), 'multi_pct': float(m['multi_pct']),
            'min_density_px_cm': float(m['min_density']), 'roi_m2': float(m['roi_area']),
            'per_camera_seen_m2': {tags.get(u, u): float(pc['seen_m2']) for u, pc in res.per_cam.items()}}


def run(project, engine=None, settings: HousingSettings | None = None, variant=None, progress=None, cancel=None,
        coverage: bool = True) -> HousingsResult | None:
    """The housings of a variant (the active one by default), what each visible camera sees of them, and (coverage)
    the ground coverage with the housings blocking and with the body and rig alone. Returns None when cancelled."""
    from .analysis import Engine
    t0 = time.perf_counter()
    st = settings or settings_of(project)
    on = HousingSettings.from_dict(dict(st.to_dict(), enabled=True))
    p = _project_for(project, variant)
    v = p.variant
    eng = engine or Engine()
    q_on = _with_settings(p, on)
    vg = eng.vehicle_geom(q_on)
    sol = solids(q_on, v, on)
    cams = [c for c in v.cameras if c.visible]
    views = []
    for i, c in enumerate(cams):
        if cancel is not None and cancel():
            return None
        views.append(view_shares(q_on, c, vg, on, v))
        if progress:
            progress(0.2 * (i + 1) / max(len(cams), 1))
    cov = None
    if coverage and vg is not None:
        cov = {}
        for k, (name, sset) in enumerate((('on', on), ('off', HousingSettings.from_dict(dict(on.to_dict(),
                                                                                              enabled=False))))):
            if cancel is not None and cancel():
                return None
            res = eng.analyse(_with_settings(p, sset))
            if res is None:
                cov = None
                break
            cov[name] = _coverage_doc(res, v)
            if progress:
                progress(0.2 + 0.4 * (k + 1))
    if progress:
        progress(1.0)
    return HousingsResult(v.uid, v.name, on, sol, views, cov, (time.perf_counter() - t0) * 1000.0)


def _r(x, nd=4):
    return None if x is None or not math.isfinite(x) else round(float(x), nd)


def summary(res: HousingsResult) -> dict:
    """Compact figures (the assistant, the batch log)."""
    out = {'variant': res.variant, 'housings': len(res.solids),
           'cameras': [{'tag': iv.tag, 'inside': iv.inside,
                        'in_view': [{'housing': s['tag'], 'share_pct': _r(s['share'] * 100, 2), 'where': s['where']}
                                    for s in iv.shares if s['share'] >= 0.0005]} for iv in res.cameras]}
    if res.coverage:
        on, off = res.coverage['on'], res.coverage['off']
        out['coverage'] = {'blind_m2': {'housings': _r(on['blind_m2']), 'body_only': _r(off['blind_m2'])},
                           'multi_pct': {'housings': _r(on['multi_pct'], 2), 'body_only': _r(off['multi_pct'], 2)},
                           'seen_m2': {t: {'housings': _r(on['per_camera_seen_m2'].get(t)),
                                           'body_only': _r(off['per_camera_seen_m2'].get(t))}
                                       for t in off['per_camera_seen_m2']}}
    return out


def to_doc(res: HousingsResult, project=None) -> dict:
    """The JSON export: every housing solid, what each camera sees of them, and the coverage with and without."""
    doc = {'schema': SCHEMA, 'project': getattr(project, 'name', ''), 'variant': res.variant, 'frame': FRAME,
           'units': 'metres, degrees (yaw, pitch positive down, roll: intrinsic Z-Y-X); shares of the image\'s '
                    'pixels inside the image circle',
           'settings': res.settings.to_dict(), 'assumptions': ASSUMPTIONS, 'limits': LIMITS_TEXT,
           'thresholds': {'warning_share': WARN_SHARE, 'serious_share': SERIOUS_SHARE, 'grid_columns': GRID_W},
           'housings': [{'tag': h.tag, 'name': h.name, 'kind': h.what, 'shape': 'cylinder' if h.round else 'box',
                         'centre': [_r(x, 5) for x in h.centre], 'half_size': [_r(x, 5) for x in h.half],
                         'rotation_ypr_deg': [_r(a, 4) for a in euler_zyx(h.R)],
                         'anchor': [_r(x, 5) for x in h.anchor]} for h in res.solids],
           'cameras': [{'tag': iv.tag, 'inside': iv.inside, 'pixels_sampled': iv.pixels,
                        'total_share': _r(iv.total, 5),
                        'in_view': [{'housing': s['tag'], 'kind': s['what'], 'share': _r(s['share'], 5),
                                     'where': s['where'], 'image_centre': s['centre'],
                                     'nearest_m': _r(s['nearest_m'], 4), 'severity': _severity(s)}
                                    for s in iv.shares]} for iv in res.cameras]}
    if res.coverage:
        on, off = res.coverage['on'], res.coverage['off']
        doc['coverage'] = {'housings_block': on, 'body_and_rig_only': off,
                           'change': {'blind_m2': _r(on['blind_m2'] - off['blind_m2']),
                                      'multi_pct': _r(on['multi_pct'] - off['multi_pct'], 3),
                                      'per_camera_seen_m2': {t: _r(on['per_camera_seen_m2'].get(t, 0.0) - a)
                                                             for t, a in off['per_camera_seen_m2'].items()}}}
    return doc


CSV_COLUMNS = ['camera', 'housing', 'kind', 'share_pct', 'where', 'image_centre', 'nearest_m', 'severity']


def to_csv(res: HousingsResult) -> str:
    """One row per camera and housing it sees (a camera that sees none: one row with no housing)."""
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator='\n')
    w.writerow(CSV_COLUMNS)
    for iv in res.cameras:
        if iv.inside:
            w.writerow([iv.tag, iv.inside[0], '', 100.0, 'inside', True, 0.0, 'critical'])
            continue
        if not iv.shares:
            w.writerow([iv.tag, '', '', 0.0, '', False, '', ''])
        for s in iv.shares:
            w.writerow([iv.tag, s['tag'], s['what'], _r(s['share'] * 100, 3), s['where'], s['centre'],
                        _r(s['nearest_m'], 4), _severity(s) or ''])
    return buf.getvalue()


def files(res: HousingsResult, project=None) -> list:
    """[(name, bytes)]: housings.json and housings.csv."""
    return [('housings.json', (json.dumps(to_doc(res, project), indent=1) + '\n').encode('utf-8')),
            ('housings.csv', to_csv(res).encode('utf-8'))]


def export(res: HousingsResult, project, out_dir) -> list[str]:
    """Write housings.json and housings.csv into out_dir; returns their paths."""
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for name, data in files(res, project):
        path = os.path.join(out_dir, name)
        with open(path, 'wb') as fh:
            fh.write(data)
        paths.append(path)
    return paths
