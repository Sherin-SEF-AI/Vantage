"""What the main 3D view shows of the rig beyond the body: sensor and camera models at their real sizes, a simulated
lidar sweep per lidar, radar fields as volumes, the sky each GNSS antenna sees and the generated mount brackets.

Models. Each housing is built from its model's size_mm at its real pose (core.sensors.body_box: the optical centre
optical_z_mm above the base). A spinning or non-repetitive lidar is a cylinder with a dark optical window band where
its beams leave the housing (from its beam elevations at the housing's radius) and a chamfered cap; a fixed-field
lidar a box with a dark window on its front face; a radar a box with a domed radome on its face; a GNSS antenna a dome
on a disc; an IMU or INS a box on a base plate with its x, y and z axes as red, green and blue arrows from its axis
centre. A camera is its module's box (width × height × length along the optical axis, lens included) behind a lens
barrel with a ring in the camera's colour, the optical centre at the lens front (as core.brackets and core.gnss take
it). The shapes are generic drawings; their sizes and poses are the project's.

Lidar points (View ▸ Overlays ▸ Lidar points). One sweep per visible lidar: every channel at its beam elevation, the
columns thinned to at most LIDAR_RAYS beams (the drawing stands for a sweep with more columns), traced against the
ground (z = 0), the project's obstacles (upright cylinders, as core.drivescene traces them), the body and the rig
(the engine's occlusion cube from the optical centre, the occluder of the cameras and the coverage study; hits within
OWN_MOUNT_M are the lidar's own mount) and the other sensors' and cameras' housings within HOUSING_REACH_M
(core.gnss's housings). A beam returns between range_min_m and min(range_m, POINTS_FAR); one that meets the vehicle,
the rig or a housing first is drawn where it lands in the warning colour, whatever its range: what the vehicle
blocks. Geometry only: one return per beam, no reflectivity, incidence or noise, a flat ground.

Radar fields. Each zone's sector (± azimuth × ± elevation about the radar's axis, out to its range on a car,
core.sensors.REF_RCS) as a translucent solid with an outline, drawn out to RADAR_FAR at most and cut at the ground.

GNSS sky (View ▸ Overlays ▸ GNSS sky). A hemisphere of SKY_R over each visible antenna in its sky frame, coloured
from core.gnss.sky_view (the cells, settings and cache of Analysis ▸ GNSS and IMU): each drawn cell shows the share
of its sky cells that something blocks; cells below the elevation mask are greyed.

Mount brackets. The brackets the Mount brackets page generated, in each camera's colour, while the camera still
stands where its bracket was made for.

The lidar sweeps and the sky views are worked out on a thread once an edit settles (OverlayController) and kept per
lidar and antenna on their own inputs, so an unrelated edit or a camera drag does not redo them.
"""
from __future__ import annotations

import math
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
from PyQt6.QtCore import QObject, QTimer, pyqtSignal
from PyQt6.QtGui import QColor

from ..core import sensors as S
from ..core.mathutil import rot_zyx
from .theme import THEME, STATUS
from .viewport3d import LIGHT, TriDraw, LineDraw

try:                                    # the point sprites of the renderer (ui.viewport3d); a stand-in until then
    from .viewport3d import PointDraw
except ImportError:                     # pragma: no cover - only before the renderer has them
    @dataclass
    class PointDraw:
        points: np.ndarray
        colors: np.ndarray
        size: float = 3.0
        depth: bool = True

LIDAR_RAYS = 48_000          # traced beams per lidar and sweep (columns thinned to this)
POINTS_FAR = 30.0            # m: returns drawn out to this range
OWN_MOUNT_M = 0.05           # hits this close to a lidar's optical centre are its own mount
HOUSING_REACH_M = 2.5        # other housings this close to a lidar (centre to centre, less their size) can block it
RADAR_FAR = 15.0             # m: radar volumes are cut here (as the lidar rings and radar edges)
SKY_R = 0.24                 # m: radius of the GNSS sky dome
SKY_AZ, SKY_EL = 72, 18      # drawn sky cells (azimuth × elevation, the elevation split at the mask)
DEBOUNCE_MS = 160
GROUND_LIFT = 0.004          # ground points sit this far above z = 0
SELF_PULL = 0.012            # vehicle hits are drawn this far towards the lidar (in front of the surface)
POINT_SIZE, SELF_SIZE = 2.5, 3.5

AXIS_RGB = (np.array([0xe0, 0x53, 0x3f]) / 255, np.array([0x3f, 0xae, 0x49]) / 255, np.array([0x3d, 0x86, 0xe6]) / 255)
WINDOW_RGB = np.array([0.055, 0.065, 0.075])        # lidar windows: dark smoked glass
RADOME_RGB = np.array([0.15, 0.15, 0.16])
GRAPHITE_RGB = np.array([0.21, 0.22, 0.24])         # camera housings
BARREL_RGB = np.array([0.09, 0.09, 0.10])
GLASS_RGB = np.array([0.04, 0.06, 0.10])
BASE_RGB = np.array([0.20, 0.20, 0.21])             # antenna discs, IMU base plates
# viridis (perceptually uniform, monotonic in lightness), 11 stops
VIRIDIS = np.array([[68, 1, 84], [72, 36, 117], [65, 68, 135], [53, 95, 141], [42, 120, 142], [33, 145, 140],
                    [34, 168, 132], [68, 191, 112], [122, 209, 81], [189, 223, 38], [253, 231, 37]], float) / 255.0


def sel_rgb() -> np.ndarray:
    return np.array(QColor(STATUS['warning']).getRgbF()[:3])


def has_outline() -> bool:
    """Whether the renderer draws TriDraw.outline (a screen-space rim for the selection)."""
    return 'outline' in getattr(TriDraw, '__dataclass_fields__', {})


def tri_draw(tri, colors, normals=None, depth=True, outline=None) -> TriDraw:
    kw = {'outline': tuple(float(v) for v in outline)} if outline is not None and has_outline() else {}
    return TriDraw(np.asarray(tri, np.float32), np.asarray(colors, np.float32), depth=depth,
                   normals=None if normals is None else np.asarray(normals, np.float32), **kw)


def ramp(t, dark: bool) -> np.ndarray:
    """Range colours (N, 3) for t in 0..1 (near..far): viridis running towards the background, so far points fade
    into it (dark theme: near bright yellow, far deep blue; light theme: near deep purple, far yellow-green)."""
    t = np.clip(np.asarray(t, float), 0.0, 1.0)
    u = 1.0 - (0.08 + 0.84 * t) if dark else 0.06 + 0.80 * t
    x = u * (len(VIRIDIS) - 1)
    i = np.clip(np.floor(x).astype(int), 0, len(VIRIDIS) - 2)
    f = (x - i)[:, None]
    return VIRIDIS[i] * (1 - f) + VIRIDIS[i + 1] * f


# ======================================================================= primitives (local frames, metres)
def _unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def _face_normals(T):
    return _unit(np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]))


def _flat(T, centre):
    """Corner normals (K, 3, 3) of flat faces, turned away from centre (convex parts) and the faces wound to match."""
    T = np.array(T, float)
    n = _face_normals(T)
    flip = np.einsum('ij,ij->i', n, T.mean(axis=1) - centre) < 0
    T[flip] = T[flip][:, ::-1]
    n[flip] *= -1
    return T, np.repeat(n[:, None], 3, axis=1)


def cylinder(r0, z0, z1, n, r1=None, cap0=False, cap1=False):
    """(triangles (K, 3, 3), corner normals (K, 3, 3)) of a cylinder or cone frustum about z from z0 (radius r0) to
    z1 (radius r1): smooth outward side normals, flat caps."""
    r1 = r0 if r1 is None else r1
    a = np.linspace(0.0, 2 * math.pi, n + 1)[:-1]
    c, s = np.cos(a), np.sin(a)
    j = np.r_[1:n, 0]
    bot = np.c_[r0 * c, r0 * s, np.full(n, z0)]
    top = np.c_[r1 * c, r1 * s, np.full(n, z1)]
    h = z1 - z0
    nr = _unit(np.c_[c * h, s * h, np.full(n, r0 - r1)])
    T = [np.stack([bot, bot[j], top[j]], 1), np.stack([bot, top[j], top], 1)]
    N = [np.stack([nr, nr[j], nr[j]], 1), np.stack([nr, nr[j], nr], 1)]
    for z, r, ring, up, on in ((z1, r1, top, 1.0, cap1), (z0, r0, bot, -1.0, cap0)):
        if not on or r <= 0:
            continue
        ctr = np.broadcast_to([0.0, 0.0, z], (n, 3))
        T.append(np.stack([ctr, ring, ring[j]], 1) if up > 0 else np.stack([ctr, ring[j], ring], 1))
        N.append(np.broadcast_to([0.0, 0.0, up], (n, 3, 3)))
    return np.concatenate(T), np.concatenate(N).copy()


def annulus(r_in, r_out, z, n, up=True):
    a = np.linspace(0.0, 2 * math.pi, n + 1)[:-1]
    c, s = np.cos(a), np.sin(a)
    j = np.r_[1:n, 0]
    i_ = np.c_[r_in * c, r_in * s, np.full(n, z)]
    o_ = np.c_[r_out * c, r_out * s, np.full(n, z)]
    T = np.concatenate([np.stack([i_, o_, o_[j]], 1), np.stack([i_, o_[j], i_[j]], 1)])
    if not up:
        T = T[:, ::-1]
    return T, np.broadcast_to([0.0, 0.0, 1.0 if up else -1.0], T.shape).copy()


def disc(r, z, n, up=True):
    a = np.linspace(0.0, 2 * math.pi, n + 1)[:-1]
    ring = np.c_[r * np.cos(a), r * np.sin(a), np.full(n, z)]
    j = np.r_[1:n, 0]
    T = np.stack([np.broadcast_to([0.0, 0.0, z], (n, 3)), ring, ring[j]], 1)
    return (T if up else T[:, ::-1]), np.broadcast_to([0.0, 0.0, 1.0 if up else -1.0], (n, 3, 3)).copy()


_BOX_F = [(0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7), (0, 1, 5), (0, 5, 4), (1, 2, 6), (1, 6, 5), (2, 3, 7), (2, 7, 6),
          (3, 0, 4), (3, 4, 7)]
_BOX_SIDE = ['-z', '-z', '+z', '+z', '-y', '-y', '+x', '+x', '+y', '+y', '-x', '-x']


def box(lo, hi, skip=()):
    """Flat-shaded box from corner lo to corner hi; skip: faces left out ('+x', '-y', ...)."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    q = np.array([[lo[0], lo[1], lo[2]], [hi[0], lo[1], lo[2]], [hi[0], hi[1], lo[2]], [lo[0], hi[1], lo[2]],
                  [lo[0], lo[1], hi[2]], [hi[0], lo[1], hi[2]], [hi[0], hi[1], hi[2]], [lo[0], hi[1], hi[2]]])
    F = [f for f, side in zip(_BOX_F, _BOX_SIDE) if side not in skip]
    return _flat(q[np.array(F)], (lo + hi) / 2)


def quad(p0, p1, p2, p3, normal):
    """Two triangles of a planar quad (corners in order) facing `normal`."""
    T = np.array([[p0, p1, p2], [p0, p2, p3]], float)
    if np.dot(_face_normals(T)[0], normal) < 0:
        T = T[:, ::-1]
    return T, np.broadcast_to(_unit(np.asarray(normal, float)), (2, 3, 3)).copy()


def face_with_window(x, y0, y1, z0, z1, wy0, wy1, wz0, wz1):
    """The +x face of a box (at x, spanning y0..y1, z0..z1) as a frame around a window (wy0..wy1, wz0..wz1): (frame,
    window) parts, coplanar but not overlapping (no depth fighting)."""
    nx = [1.0, 0.0, 0.0]
    P = lambda y, z: [x, y, z]               # noqa: E731
    frame = [quad(P(y0, z0), P(y1, z0), P(y1, wz0), P(y0, wz0), nx), quad(P(y0, wz1), P(y1, wz1), P(y1, z1), P(y0, z1), nx),
             quad(P(y0, wz0), P(wy0, wz0), P(wy0, wz1), P(y0, wz1), nx),
             quad(P(wy1, wz0), P(y1, wz0), P(y1, wz1), P(wy1, wz1), nx)]
    win = quad(P(wy0, wz0), P(wy1, wz0), P(wy1, wz1), P(wy0, wz1), nx)
    return (np.concatenate([f[0] for f in frame]), np.concatenate([f[1] for f in frame])), win


def ellipsoid_cap(a, c, z0, n_az, n_el, theta_max=math.pi / 2):
    """Upper cap of an ellipsoid (semi-axes a, a, c) centred at z0, from its top to polar angle theta_max."""
    th = np.linspace(0.0, theta_max, n_el + 1)
    ph = np.linspace(0.0, 2 * math.pi, n_az + 1)
    TH, PH = np.meshgrid(th, ph, indexing='ij')
    P = np.stack([a * np.sin(TH) * np.cos(PH), a * np.sin(TH) * np.sin(PH), z0 + c * np.cos(TH)], -1)
    N = _unit(np.stack([P[..., 0] / a ** 2, P[..., 1] / a ** 2, (P[..., 2] - z0) / c ** 2], -1))
    return _grid(P, N)


def _grid(P, N):
    """Triangles and corner normals of a (rows, cols) vertex grid."""
    a, b, c, d = P[:-1, :-1], P[:-1, 1:], P[1:, 1:], P[1:, :-1]
    na, nb, nc, nd = N[:-1, :-1], N[:-1, 1:], N[1:, 1:], N[1:, :-1]
    T = np.concatenate([np.stack([a, b, c], -2).reshape(-1, 3, 3), np.stack([a, c, d], -2).reshape(-1, 3, 3)])
    NN = np.concatenate([np.stack([na, nb, nc], -2).reshape(-1, 3, 3), np.stack([na, nc, nd], -2).reshape(-1, 3, 3)])
    area = np.linalg.norm(np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]), axis=1)
    keep = area > 1e-14
    return T[keep], NN[keep]


def pillow(x0, bulge, y0, y1, z0, z1, nu, nv):
    """A rounded (radome) surface over the rectangle y0..y1 × z0..z1 at x = x0, bulging forward by `bulge` at its
    middle and meeting x0 at its rim."""
    u = np.linspace(-1.0, 1.0, nu + 1)
    v = np.linspace(-1.0, 1.0, nv + 1)
    U, Vv = np.meshgrid(u, v, indexing='ij')
    fu, fv = (1 - U ** 2) ** 0.45, (1 - Vv ** 2) ** 0.45
    hy, hz = (y1 - y0) / 2, (z1 - z0) / 2
    P = np.stack([x0 + bulge * fu * fv, (y0 + y1) / 2 + U * hy, (z0 + z1) / 2 + Vv * hz], -1)
    with np.errstate(divide='ignore', invalid='ignore'):
        dfu = np.where(np.abs(U) < 1, -0.9 * U * (1 - U ** 2) ** -0.55, 0.0)
        dfv = np.where(np.abs(Vv) < 1, -0.9 * Vv * (1 - Vv ** 2) ** -0.55, 0.0)
    dxdy = np.clip(bulge * dfu * fv / hy, -8, 8)
    dxdz = np.clip(bulge * fu * dfv / hz, -8, 8)
    N = _unit(np.stack([np.ones_like(U), -dxdy, -dxdz], -1))
    return _grid(P, N)


def _to_axis(A, k):
    """Coordinates built about z turned onto axis k (0 x, 1 y, 2 z) by a cyclic permutation (a proper rotation)."""
    A = np.asarray(A, float)
    if k == 2:
        return A
    return A[..., [2, 0, 1]] if k == 0 else A[..., [1, 2, 0]]


def arrow(k, length, r_shaft, r_head, l_head, n):
    """An arrow from the origin along axis k: a shaft and a cone."""
    T1, N1 = cylinder(r_shaft, 0.0, length - l_head, n, cap0=False)
    T2, N2 = cylinder(r_head, length - l_head, length, n, r1=0.0, cap0=True)
    return _to_axis(np.concatenate([T1, T2]), k), _to_axis(np.concatenate([N1, N2]), k)


class Model:
    """Triangles, corner normals and colours of a drawn object in its own frame."""

    def __init__(self):
        self.T, self.N, self.C = [], [], []

    def add(self, part, rgb):
        T, N = part
        if len(T):
            self.T.append(np.asarray(T, float))
            self.N.append(np.asarray(N, float))
            self.C.append(np.broadcast_to(np.asarray(rgb, float), (len(T), 3)))
        return self

    def arrays(self):
        if not self.T:
            return np.zeros((0, 3, 3)), np.zeros((0, 3, 3)), np.zeros((0, 3))
        return np.concatenate(self.T), np.concatenate(self.N), np.concatenate(self.C)


# ======================================================================= sensor housings
def _segments(detail):
    return 36 if detail == 'high' else 12


def window_band(model, r) -> tuple[float, float]:
    """(lowest, highest) z of a round lidar's optical window about its optical centre: where its lowest and highest
    beams cross the housing's radius, a few millimetres more, kept inside the housing."""
    dx, _, h = (max(float(v), 1.0) / 1000.0 for v in model.size_mm)
    oz = model.optical_z_mm / 1000.0
    zb, zt = -oz, h - oz
    lo, hi = model.v_fov()
    w_lo = -(r * math.tan(math.radians(min(max(-lo, 0.0), 80.0))) + 0.003)
    w_hi = r * math.tan(math.radians(min(max(hi, 0.0), 80.0))) + 0.003
    w_lo = max(min(w_lo, -0.12 * h), zb + 0.12 * h)
    w_hi = min(max(w_hi, 0.12 * h), zt - 0.14 * h)
    if w_hi - w_lo < 0.15 * h:                        # the optical centre near an end: keep a visible band
        w_lo, w_hi = max(zb + 0.1 * h, -0.2 * h), min(zt - 0.1 * h, 0.2 * h)
    return w_lo, w_hi


def sensor_model(s, model, rgb, detail='high') -> Model:
    """The housing of a lidar, radar, GNSS antenna or IMU in its own frame (origin at its optical centre)."""
    dx, dy, h = (max(float(v), 1.0) / 1000.0 for v in model.size_mm)
    oz = model.optical_z_mm / 1000.0
    zb, zt = -oz, h - oz
    rgb = np.asarray(rgb, float)
    n = _segments(detail)
    M = Model()
    _, _, _, rnd = S.body_box(s, model)
    if model.kind == 'lidar' and rnd:
        r = dx / 2
        w0, w1 = window_band(model, r)
        cap = min(0.08 * h, 0.006)
        rw = r * 0.975                                  # the window sits a little inside the shell
        M.add(cylinder(r, zb, w0, n, cap0=True), rgb)
        M.add(annulus(rw, r, w0, n, up=False), rgb * 0.8)
        M.add(cylinder(rw, w0, w1, n), WINDOW_RGB)
        M.add(annulus(rw, r, w1, n, up=True), rgb * 0.8)
        M.add(cylinder(r, w1, zt - cap, n), rgb)
        M.add(cylinder(r, zt - cap, zt, n, r1=r - cap, cap1=True), rgb * 0.82)
        if detail == 'high':                            # a shallow recess on the cap
            M.add(annulus(r * 0.55, r * 0.6, zt + 0.0004, n), rgb * 0.62)
        return M
    if model.kind == 'lidar':                           # fixed-field: a box with a window on its +x face
        x0, x1, y0, y1 = -dx / 2, dx / 2, -dy / 2, dy / 2
        M.add(box([x0, y0, zb], [x1, y1, zt], skip=('+x',)), rgb)
        wz0 = max(zb + 0.14 * h, -0.36 * h)
        wz1 = min(zt - 0.14 * h, 0.36 * h)
        if wz1 - wz0 < 0.2 * h:
            wz0, wz1 = zb + 0.18 * h, zt - 0.18 * h
        frame, win = face_with_window(x1, y0, y1, zb, zt, -0.42 * dy, 0.42 * dy, wz0, wz1)
        M.add(frame, rgb * 0.85)
        M.add(win, WINDOW_RGB)
        return M
    if model.kind == 'radar':
        bulge = min(0.3 * dx, 0.012)
        xf = dx / 2 - bulge
        m_ = min(0.08 * min(dy, h), 0.006)
        M.add(box([-dx / 2, -dy / 2, zb], [xf, dy / 2, zt], skip=('+x',)), rgb)
        frame, _ = face_with_window(xf, -dy / 2, dy / 2, zb, zt, -dy / 2 + m_, dy / 2 - m_, zb + m_, zt - m_)
        M.add(frame, rgb * 0.8)
        nu, nv = (12, 8) if detail == 'high' else (4, 3)
        M.add(pillow(xf, bulge, -dy / 2 + m_, dy / 2 - m_, zb + m_, zt - m_, nu, nv), RADOME_RGB + rgb * 0.08)
        return M
    if model.kind == 'gnss':
        r = dx / 2
        hb = min(0.22 * h, 0.012)
        M.add(cylinder(r, zb, zb + hb, n, cap0=True), BASE_RGB)
        M.add(annulus(r * 0.93, r, zb + hb, n), BASE_RGB * 1.15)
        ne = 8 if detail == 'high' else 3
        M.add(ellipsoid_cap(r * 0.93, h - hb, zb + hb, n, ne), rgb)
        return M
    # IMU / INS: a box on a base plate, its axes as arrows from the axis centre
    hp = min(0.14 * h, 0.006)
    M.add(box([-dx / 2 - 0.08 * dx, -dy / 2, zb], [dx / 2 + 0.08 * dx, dy / 2, zb + hp]), BASE_RGB)
    M.add(box([-dx / 2, -dy / 2, zb + hp], [dx / 2, dy / 2, zt]), rgb)
    if detail == 'high':
        M.add(box([-dx / 2 * 0.8, -dy / 2 * 0.8, zt], [dx / 2 * 0.8, dy / 2 * 0.8, zt + min(0.002, 0.04 * h)]),
              rgb * 0.85)
    big = max(dx, dy, h)
    na = 16 if detail == 'high' else 6
    for k, half in enumerate((dx / 2, dy / 2, h / 2)):
        L = half + 0.55 * big
        M.add(arrow(k, L, 0.03 * big, 0.075 * big, 0.2 * big, na), AXIS_RGB[k])
    return M


# ======================================================================= camera modules
def camera_model(module, ring_rgb, detail='high') -> Model:
    """A camera module in the camera's body frame (x along the optical axis, origin at the optical centre at the lens
    front): its box (width × height across, the rest of its length behind the lens), a lens barrel and a ring in the
    camera's colour with the lens glass recessed in it."""
    try:
        w, hh, L = (max(float(v), 1.0) / 1000.0 for v in module.size_mm)
    except (TypeError, ValueError):
        w, hh, L = 0.025, 0.025, 0.0438
    m = min(w, hh)
    r_b, r_r, r_g = 0.33 * m, 0.38 * m, 0.27 * m
    x_box, x_ring = -0.38 * L, -0.08 * L
    n = 28 if detail == 'high' else 8
    M = Model()
    T, N = box([-L, -w / 2, -hh / 2], [x_box, w / 2, hh / 2])
    M.add((T, N), GRAPHITE_RGB)
    M.add(_xpart(cylinder(r_b, x_box, x_ring, n)), BARREL_RGB)
    M.add(_xpart(annulus(r_b, r_r, x_ring, n, up=False)), ring_rgb)
    M.add(_xpart(cylinder(r_r, x_ring, 0.0, n)), ring_rgb)
    M.add(_xpart(annulus(r_g, r_r, 0.0, n)), ring_rgb)
    if detail == 'high':
        rec = -0.02 * L
        M.add(_xpart(cylinder(r_g, rec, 0.0, n)), BARREL_RGB)
        M.add(_xpart(disc(r_g, rec, n)), GLASS_RGB)
    else:
        M.add(_xpart(disc(r_g, -0.001, n)), GLASS_RGB)
    return M


def _xpart(part):
    """A part built about z (from z0 to z1) along the camera's optical axis (+x)."""
    T, N = part
    return _to_axis(T, 0), _to_axis(N, 0)


# ======================================================================= posing and drawing
_POSED = OrderedDict()           # key -> (T, N, C) in the vehicle frame
_POSED_MAX = 96


def _posed(key, build, R, t):
    hit = _POSED.get(key)
    if hit is not None:
        _POSED.move_to_end(key)
        return hit
    T, N, C = build().arrays()
    out = (T @ R.T + t, N @ R.T, C)
    _POSED[key] = out
    while len(_POSED) > _POSED_MAX:
        _POSED.popitem(last=False)
    return out


_DRAWS = OrderedDict()           # (part ids, options) -> (parts kept alive, TriDraw)


def draw_parts(parts, gl, outline=None, tint=None, alpha=1.0, smooth=True):
    """One TriDraw of posed parts [(T, N, C)]: smooth corner normals for the shader renderer, the light baked into
    the colours for the software one; tint mixes the selection colour in (without an outline). The same parts give
    the same TriDraw object back, so the renderer keeps its shadow maps (it re-renders them when the triangles of
    the shadow casters change). smooth False: no corner normals, so the renderer neither smooths nor casts shadows
    with it (what is being dragged: its shadow maps would be redone every frame)."""
    parts = [q for q in parts if len(q[0])]
    if not parts:
        return None
    key = (tuple(id(q) for q in parts), gl, smooth, None if outline is None else tuple(outline), tint, alpha,
           THEME.name)
    hit = _DRAWS.get(key)
    if hit is not None:
        _DRAWS.move_to_end(key)
        return hit[1]
    T = np.concatenate([q[0] for q in parts])
    N = np.concatenate([q[1] for q in parts])
    C = np.concatenate([q[2] for q in parts])
    if tint is not None:
        C = C * (1 - tint) + sel_rgb() * tint
    if not gl:
        fn = _unit(N.mean(axis=1))
        C = C * (0.55 + 0.45 * np.clip(fn @ LIGHT, 0, 1))[:, None]
    rgba = np.c_[np.clip(C, 0, 1), np.full(len(T), alpha)]
    td = tri_draw(T, rgba, normals=N if gl and smooth else None, outline=outline)
    _DRAWS[key] = (parts, td)
    while len(_DRAWS) > 48:
        _DRAWS.popitem(last=False)
    return td


def sel_outline():
    return (*sel_rgb(), 1.0)


def finite_pose(o) -> bool:
    """Whether a camera's or sensor's pose is all numbers (a hand-edited file can hold anything)."""
    try:
        return all(math.isfinite(float(getattr(o, a))) for a in ('x', 'y', 'z', 'yaw', 'pitch', 'roll'))
    except (TypeError, ValueError):
        return False


def _pose_key(o):
    return tuple(round(float(getattr(o, a)), 5) for a in ('x', 'y', 'z', 'yaw', 'pitch', 'roll'))


def sensor_parts(project, s, rgb, gl):
    """(T, N, C) of a sensor's housing at its pose (cached)."""
    m = project.sensor_model_for(s)
    detail = 'high' if gl else 'low'
    key = ('sensor', s.kind, _pose_key(s), m.kind, m.scan, tuple(m.size_mm), m.optical_z_mm, tuple(m.v_fov()),
           tuple(np.round(rgb, 4)), detail)
    return _posed(key, lambda: sensor_model(s, m, rgb, detail), S.rotation(s), s.pos().astype(float))


def camera_parts(project, cam, gl):
    """(T, N, C) of a camera's module at its pose (cached)."""
    mod = project.module_for(cam)
    ring = np.array(THEME.cam(cam.slot).getRgbF()[:3])
    detail = 'high' if gl else 'low'
    key = ('camera', _pose_key(cam), tuple(mod.size_mm), tuple(np.round(ring, 4)), detail)
    return _posed(key, lambda: camera_model(mod, ring, detail), rot_zyx(cam.yaw, cam.pitch, cam.roll),
                  cam.pos().astype(float))


def camera_draws(project, cams, sel, gl, smooth=True) -> list:
    """The camera modules: one draw for the unselected ones, one with the selection outline for the selected.
    smooth False while something is dragged (see draw_parts)."""
    rest, chosen = [], []
    for c in cams:
        if finite_pose(c):
            (chosen if c.uid in sel else rest).append(camera_parts(project, c, gl))
    out = [draw_parts(rest, gl, smooth=smooth)]
    if chosen:
        out.append(draw_parts(chosen, gl, outline=sel_outline(), smooth=smooth) if has_outline() else
                   draw_parts(chosen, gl, tint=0.35, smooth=smooth))
    return [d for d in out if d is not None]


# ======================================================================= radar volumes
def radar_volume(s, m, far=RADAR_FAR, rcs=None, step_deg=4.0, rings=6) -> list:
    """Each zone's field as [(triangles (K, 3, 3), outward normals (K, 3), range share (K,) of each triangle's
    middle, outline polylines [rim, corner rays])]: the sector ± az × ± el about the radar's axis out to the zone's
    range for the target (cut at `far`), split into `rings` shells along its rays so it can fade with range, every
    vertex pulled back along its ray to the ground where the ray meets the ground first. The normals point out of
    the field (radially on its end, across the edge on its sides), so a view can leave out the faces turned away."""
    out = []
    pos = s.pos().astype(float)
    R = S.rotation(s)
    k = S.radar_scale(rcs)
    for rng, ah, eh in m.zones:
        r = min(float(rng) * k, far)
        ah, eh = min(float(ah), 180.0), min(float(eh), 89.5)
        if r <= 0 or ah <= 0 or eh <= 0:
            continue
        na = max(2, int(math.ceil(2 * ah / step_deg)) + 1)
        ne = max(2, int(math.ceil(2 * eh / step_deg)) + 1)
        az = np.radians(np.linspace(-ah, ah, na))
        el = np.radians(np.linspace(-eh, eh, ne))
        D = S.directions(np.degrees(el), np.degrees(az)) @ R.T                     # (ne, na, 3)
        with np.errstate(divide='ignore', invalid='ignore'):
            tg = np.where((D[..., 2] < -1e-9) & (pos[2] > 0), -pos[2] / D[..., 2], np.inf)

        def at(f):
            return pos + D * np.minimum(r * f, tg)[..., None]
        P = at(1.0)
        a, b, c, d = P[:-1, :-1], P[:-1, 1:], P[1:, 1:], P[1:, :-1]
        T = [np.stack([a, b, c], -2).reshape(-1, 3, 3), np.stack([a, c, d], -2).reshape(-1, 3, 3)]
        H = [T[0].mean(axis=1) - pos, T[1].mean(axis=1) - pos]                     # the end: radially out
        # the sides and the way out of the field across each: (edge selector, hint per edge vertex)
        ce, se = np.cos(el), np.sin(el)
        ca, sa = np.cos(az), np.sin(az)
        sides = [((0, slice(None)), np.stack([se[0] * ca, se[0] * sa, np.full(na, -ce[0])], -1)),
                 ((-1, slice(None)), np.stack([-se[-1] * ca, -se[-1] * sa, np.full(na, ce[-1])], -1))]
        if ah < 179.9:
            sides += [((slice(None), 0), np.tile([sa[0], -ca[0], 0.0], (ne, 1))),
                      ((slice(None), -1), np.tile([-sa[-1], ca[-1], 0.0], (ne, 1)))]
        prev = np.broadcast_to(pos, P.shape)
        for i in range(1, rings + 1):                   # the side walls as strips between shells
            cur = at(i / rings)
            for sl, hint in sides:
                e0, e1 = prev[sl], cur[sl]
                hw = (hint @ R.T)
                hm = 0.5 * (hw[:-1] + hw[1:])
                T.append(np.stack([e0[:-1], e1[:-1], e1[1:]], 1))
                T.append(np.stack([e0[:-1], e1[1:], e0[1:]], 1))
                H += [hm, hm]
            prev = cur
        T, H = np.concatenate(T), np.concatenate(H)
        n = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
        area = np.linalg.norm(n, axis=1)
        keep = area > 1e-12
        T, H, n = T[keep], H[keep], n[keep] / area[keep, None]
        flip = np.einsum('ij,ij->i', n, H) < 0
        T[flip] = T[flip][:, ::-1]
        n[flip] *= -1
        frac = np.linalg.norm(T.mean(axis=1) - pos, axis=1) / r
        rim = np.vstack([P[0, :], P[1:, -1], P[-1, ::-1][1:], P[::-1, 0][1:]])
        lines = [rim] + ([np.array([pos, P[i, j]]) for i, j in ((0, 0), (0, -1), (-1, -1), (-1, 0))]
                         if ah < 179.9 else [])
        out.append((T, n, np.clip(frac, 0.0, 1.0), lines))
    return out


def facing(tri, normals, vp) -> np.ndarray:
    """Which faces turn towards the viewport's eye (or, orthographic, against its view direction)."""
    Vi = np.linalg.inv(vp.cam.view())
    if vp.cam.ortho:
        return normals @ Vi[:3, 2] > 0
    return np.einsum('ij,ij->i', normals, Vi[:3, 3] - tri.mean(axis=1)) > 0


def radar_alpha(frac, selected: bool) -> np.ndarray:
    """Fill opacity of a radar volume, fading out towards the end of the drawn range."""
    a0 = 0.16 if selected else 0.075
    return a0 * (1.0 - np.asarray(frac)) ** 1.6 + (0.02 if selected else 0.006)


# ======================================================================= lidar sweeps
HIT_GROUND, HIT_OBSTACLE, HIT_VEHICLE, HIT_HOUSING = 0, 1, 2, 3


@dataclass
class Sweep:
    """One lidar's simulated sweep: the returns (ground, obstacles) and where the blocked beams land."""
    uid: str
    tag: str
    pos: np.ndarray
    points: np.ndarray               # (K, 3) float32 returns, vehicle frame
    rng: np.ndarray                  # (K,) float32 range from the optical centre, m
    hit: np.ndarray                  # (K,) int8 HIT_GROUND | HIT_OBSTACLE
    self_points: np.ndarray          # (M, 3) float32 where blocked beams meet the vehicle, the rig or a housing
    self_hit: np.ndarray             # (M,) int8 HIT_VEHICLE | HIT_HOUSING
    rays: int                        # beams traced (channels × columns)
    channels: int
    columns: int
    far: float
    blocked: float                   # share of the traced beams the vehicle, rig or a housing blocks
    ms: float = 0.0
    key: tuple = ()


def _housings(project, s):
    from ..core import gnss as G
    pos = s.pos().astype(float)
    return [h for h in G._housings(project, project.variant, s.uid)
            if np.linalg.norm(h[1] - pos) - np.linalg.norm(h[2]) <= HOUSING_REACH_M]


def lidar_sweep(project, engine, s, far=POINTS_FAR, max_rays=LIDAR_RAYS, vg=None) -> Sweep:
    """One sweep of lidar s (see the module docstring); engine None or no vehicle: ground, obstacles and housings."""
    from ..core import drivescene as DS
    from ..core import gnss as G
    t0 = time.perf_counter()
    m = project.sensor_model_for(s)
    E = m.elevations()
    C = len(E)
    half = m.az_half()
    width = 360.0 if half >= 180.0 else 2 * half
    J = int(max(1, min(m.columns(), max(32, max_rays // max(C, 1)))))
    az = S.scan_azimuths(m, width / J)
    pos = s.pos().astype(float)
    D = (S.directions(E, az) @ S.rotation(s).T).reshape(-1, 3)
    n = len(D)
    with np.errstate(divide='ignore', invalid='ignore'):
        t = np.where((D[:, 2] < -1e-9) & (pos[2] > 0), -pos[2] / D[:, 2], np.inf)
    what = np.where(np.isfinite(t), HIT_GROUND, -1).astype(np.int8)

    def nearer(tt, kind):
        closer = tt < t
        t[closer] = tt[closer]
        what[closer] = kind
    for ob in project.obstacles:
        if ob.height > 0 and ob.radius > 0:
            nearer(DS._ray_cyl(pos, D, ob.x, ob.y, ob.radius, 0.0, ob.height), HIT_OBSTACLE)
    if vg is None and engine is not None:
        vg = engine.vehicle_geom(project)
    if vg is not None and engine is not None:
        dv = engine.cube(pos, vg).lookup(D)[0].astype(float)
        nearer(np.where(dv > OWN_MOUNT_M, dv, np.inf), HIT_VEHICLE)
    for _, c, hb, Rh, rnd in _housings(project, s):
        th = G._ray_housing(pos, D, c, hb, Rh, rnd)
        nearer(np.where(th > OWN_MOUNT_M, th, np.inf), HIT_HOUSING)
    lim = min(float(m.range_m), far)
    ret = ((what == HIT_GROUND) | (what == HIT_OBSTACLE)) & (t >= m.range_min_m) & (t <= lim)
    blk = (what == HIT_VEHICLE) | (what == HIT_HOUSING)
    P = pos + D[ret] * t[ret, None]
    P[what[ret] == HIT_GROUND, 2] += GROUND_LIFT
    Q = pos + D[blk] * np.maximum(t[blk] - SELF_PULL, 0.0)[:, None]
    return Sweep(s.uid, s.tag, pos, P.astype(np.float32), t[ret].astype(np.float32), what[ret],
                 Q.astype(np.float32), what[blk], n, C, len(az), far, float(blk.sum()) / max(n, 1),
                 (time.perf_counter() - t0) * 1000.0)


def sweep_key(project, s, vg_version, far=POINTS_FAR) -> tuple:
    """Everything a lidar's sweep depends on."""
    m = project.sensor_model_for(s)
    houses = tuple((round(float(c[0]), 4), round(float(c[1]), 4), round(float(c[2]), 4), tuple(np.round(hb, 4)),
                    tuple(np.round(Rh, 4).ravel()), bool(rnd)) for _, c, hb, Rh, rnd in _housings(project, s))
    return (s.uid, _pose_key(s), m.kind, m.scan, tuple(m.elevations().round(4)), m.h_fov, m.h_res, m.range_m,
            m.range_min_m, vg_version, tuple((o.x, o.y, o.radius, o.height) for o in project.obstacles), houses, far)


def sweep_draws(sw: Sweep, dark: bool) -> list:
    """Point sprites of a sweep: the returns coloured by range, the blocked beams in the warning colour."""
    out = []
    if len(sw.points):
        rgb = ramp(sw.rng / max(sw.far, 1e-6), dark)
        out.append(PointDraw(sw.points, np.c_[rgb, np.full(len(rgb), 0.95)].astype(np.float32), POINT_SIZE, True))
    if len(sw.self_points):
        c = np.array(QColor(STATUS['critical']).getRgbF()[:3])
        col = np.tile(np.r_[c, 1.0], (len(sw.self_points), 1)).astype(np.float32)
        out.append(PointDraw(sw.self_points, col, SELF_SIZE, True))
    return out


# ======================================================================= GNSS sky domes
def _groups(n, k):
    """Split range(n) into at most k contiguous groups (index arrays)."""
    return [g for g in np.array_split(np.arange(n), min(max(k, 1), n)) if len(g)] if n else []


def sky_cells(sv, n_az=SKY_AZ, n_el=SKY_EL) -> list:
    """The drawn cells of an antenna's sky: [(az0, az1, el0, el1 degrees, blocked share, below the mask)], each a
    block of core.gnss's sky cells, the elevation split at the mask so no drawn cell straddles it."""
    A = len(sv.az)
    step = 360.0 / max(A, 1)
    edges = np.asarray(sv.el_edges, float)
    blocked = np.asarray(sv.who) >= 0
    below = np.flatnonzero(edges[1:] <= sv.mask_deg + 1e-9)
    above = np.flatnonzero(edges[1:] > sv.mask_deg + 1e-9)
    n_lo = max(1, int(round(n_el * (len(below) / max(len(edges) - 1, 1))))) if len(below) else 0
    rows = [(g, True) for g in _groups(len(below), n_lo)] + \
           [(g + len(below), False) for g in _groups(len(above), max(n_el - n_lo, 1))]
    cols = _groups(A, n_az)
    out = []
    for rg, low in rows:
        for cg in cols:
            share = float(blocked[np.ix_(rg, cg)].mean())
            out.append((float(sv.az[cg[0]] - step / 2), float(sv.az[cg[-1]] + step / 2), float(edges[rg[0]]),
                        float(edges[rg[-1] + 1]), share, low))
    return out


def sky_dome(sv, dark: bool, radius=SKY_R, n_az=SKY_AZ, n_el=SKY_EL):
    """(triangles (K, 3, 3), RGBA (K, 4), outline polylines) of the dome over an antenna."""
    cells = sky_cells(sv, n_az, n_el)
    if not cells:
        return np.zeros((0, 3, 3)), np.zeros((0, 4)), []
    c = np.array(cells)
    a0, a1, e0, e1, share, low = (c[:, i] for i in range(6))
    corners = [(a0, e0), (a1, e0), (a1, e1), (a0, e1)]
    F = np.asarray(sv.frame, float)
    pos = np.asarray(sv.pos, float)

    def pt(a, e):
        a, e = np.radians(a), np.radians(e)
        d = np.stack([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)], -1)
        return pos + (d @ F.T) * radius
    q = [pt(a, e) for a, e in corners]
    T = np.concatenate([np.stack([q[0], q[1], q[2]], 1), np.stack([q[0], q[2], q[3]], 1)])
    open_rgb = np.array([0.55, 0.78, 0.95]) if dark else np.array([0.30, 0.55, 0.85])
    grey = np.array([0.55, 0.55, 0.55])
    bad = np.array(QColor(STATUS['critical']).getRgbF()[:3])
    base = np.where(low[:, None] > 0.5, grey, open_rgb)
    rgb = base * (1 - share[:, None]) + bad * share[:, None]
    alpha = np.where(low > 0.5, 0.10, 0.13) + (0.72 - 0.13) * share
    rgba = np.c_[rgb, alpha]
    rgba = np.concatenate([rgba, rgba])
    area = np.linalg.norm(np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]), axis=1)
    keep = area > 1e-14
    ring = np.linspace(0.0, 360.0, 73)
    lines = [pt(ring, np.zeros_like(ring))]
    if sv.mask_deg > 0.5:
        lines.append(pt(ring, np.full_like(ring, sv.mask_deg)))
    return T[keep], rgba[keep], lines


# ======================================================================= mount brackets
_BRACKETS = OrderedDict()


def _bracket_fits(b, cam, tol_m=0.002, tol_deg=0.5) -> bool:
    """Whether the camera still stands where its bracket was made for (optical centre and axis)."""
    if b.oc is None or b.axis is None:
        return True
    oc = np.asarray(b.origin, float) + np.asarray(b.R, float) @ (np.asarray(b.oc, float) / 1000.0)
    ax = np.asarray(b.R, float) @ np.asarray(b.axis, float)
    fwd = rot_zyx(cam.yaw, cam.pitch, cam.roll)[:, 0]
    cosang = float(np.clip(_unit(ax) @ fwd, -1, 1))
    return float(np.linalg.norm(cam.pos() - oc)) <= tol_m and math.degrees(math.acos(cosang)) <= tol_deg


def bracket_parts(b, slot):
    """(T, N, C) of a generated bracket in the vehicle frame, smooth-shaded (cached per bracket)."""
    from ..core import brackets as BR
    rgb = np.array(THEME.cam(slot).getRgbF()[:3])
    key = (id(b), tuple(np.round(rgb, 4)))
    hit = _BRACKETS.get(key)
    if hit is not None and hit[0] is b:
        return hit[1]
    V, F = b.display_fast or b.display or BR.display_mesh(b, 6000)
    V = np.asarray(b.origin, float) + (np.asarray(V, float) / 1000.0) @ np.asarray(b.R, float).T
    F = np.asarray(F, np.int64)
    T = V[F]
    fn = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])          # area weighted
    vn = np.zeros_like(V)
    for k in range(3):
        np.add.at(vn, F[:, k], fn)
    N = _unit(vn)[F]
    out = (T, N, np.broadcast_to(rgb * 0.9, (len(T), 3)))
    _BRACKETS[key] = (b, out)
    while len(_BRACKETS) > 32:
        _BRACKETS.popitem(last=False)
    return out


def bracket_draws(ctx, gl) -> list:
    """The generated mount brackets of cameras still where their bracket fits, in each camera's colour."""
    ctl = getattr(ctx, 'brackets_ctl', None)
    res = getattr(ctl, 'result', None)
    if res is None or getattr(ctl, '_res_project', None) is not ctx.store.project:
        return []
    v = ctx.store.project.variant
    if res.variant_uid and res.variant_uid != v.uid:
        return []
    parts, sel = [], []
    chosen = set(ctx.store.sel_cams)
    for b in res.brackets:
        cam = v.cam(b.uid) if hasattr(v, 'cam') else next((c for c in v.cameras if c.uid == b.uid), None)
        if not b.generated or cam is None or not cam.visible or not _bracket_fits(b, cam):
            continue
        (sel if cam.uid in chosen else parts).append(bracket_parts(b, cam.slot))
    out = [draw_parts(parts, gl)]
    if sel:
        out.append(draw_parts(sel, gl, outline=sel_outline() if has_outline() else None))
    return [d for d in out if d is not None]


# ======================================================================= background work
class _Sig(QObject):
    done = pyqtSignal(int, object)


def controller(ctx) -> 'OverlayController':
    c = getattr(ctx, 'overlays3d', None)
    if c is None:
        c = ctx.overlays3d = OverlayController(ctx)
    return c


class OverlayController(QObject):
    """Lidar sweeps and GNSS sky views for the 3D overlays, worked out on a thread once an edit settles and kept per
    lidar and antenna on their inputs. Views ask with want(); ready fires when new data is in."""
    ready = pyqtSignal()

    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        self.sweeps: dict = {}           # lidar uid -> Sweep
        self.skies: dict = {}            # antenna uid -> (key, SkyView)
        self._keys = (None, {}, {})      # (cheap key, lidar uid -> key, antenna uid -> key) of the live project
        self._running = False
        self._pending = False
        self._gen = 0
        self._draws = {}                 # (uid, sweep id, dark) -> [PointDraw]
        self._domes = {}
        self.sig = _Sig()
        self.sig.done.connect(self._done)
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(DEBOUNCE_MS)
        self.timer.timeout.connect(self._start)
        self._need = (False, False)
        ctx.store.project_replaced.connect(self._reset)

    def _reset(self, *_):
        self._gen += 1
        self.sweeps, self.skies, self._draws, self._domes = {}, {}, {}, {}
        self._keys = (None, {}, {})
        self._running = False

    # ---- what the live project needs
    def _vg(self, project):
        try:
            return self.ctx.analysis.engine.vehicle_geom(project)
        except Exception:
            return None

    def keys(self):
        """(lidar uid -> sweep key, antenna uid -> sky key) for the live project (recomputed on a new revision)."""
        st = self.ctx.store
        p = st.project
        vg = self._vg(p)
        cheap = (getattr(st, 'revision', None), id(p), getattr(vg, 'version', None), self._need)
        if self._keys[0] == cheap and cheap[0] is not None:
            return self._keys[1], self._keys[2]
        lk, sk = {}, {}
        if self._need[0]:
            for s in p.variant.sensors:
                if s.visible and s.kind == 'lidar' and finite_pose(s):
                    lk[s.uid] = sweep_key(p, s, getattr(vg, 'version', None))
        if self._need[1]:
            from ..core import gnss as G
            gs = G.GnssSettings.from_dict(getattr(p.analysis, 'gnss', None))
            for s in p.variant.sensors:
                if s.visible and s.kind == 'gnss' and finite_pose(s):
                    sk[s.uid] = G._sky_key(p, vg, s, gs, p.variant)
        self._keys = (cheap, lk, sk)
        return lk, sk

    def want(self, lidar: bool, sky: bool):
        """Called by the views when they draw: schedules the work the shown overlays need."""
        self._need = (bool(lidar), bool(sky))
        lk, sk = self.keys()
        stale = any(self.sweeps.get(u) is None or self.sweeps[u].key != k for u, k in lk.items()) or \
            any(self.skies.get(u, (None,))[0] != k for u, k in sk.items())
        if stale and not self.timer.isActive() and not self._running:
            self.timer.start()
        elif stale and self._running:
            self._pending = True

    def busy(self) -> bool:
        lk, sk = self.keys()
        stale = any(self.sweeps.get(u) is None or self.sweeps[u].key != k for u, k in lk.items()) or \
            any(self.skies.get(u, (None,))[0] != k for u, k in sk.items())
        return self._running or self.timer.isActive() or stale

    def _start(self):
        st = self.ctx.store
        if getattr(st, 'dragging', False):           # a drag in progress: wait until it ends
            self.timer.start()
            return
        if self._running:
            self._pending = True
            return
        p = st.project
        lk, sk = self.keys()
        todo_l = {u: k for u, k in lk.items() if self.sweeps.get(u) is None or self.sweeps[u].key != k}
        todo_s = {u: k for u, k in sk.items() if self.skies.get(u, (None,))[0] != k}
        if not todo_l and not todo_s:
            return
        snap = p.snapshot()
        eng = self.ctx.analysis.engine.share()
        self._gen += 1
        gen = self._gen
        self._running = True

        def work():
            out = {'lidar': {}, 'sky': {}, 'project': p}
            try:
                vg = eng.vehicle_geom(snap)
                for s in snap.variant.sensors:
                    if s.uid in todo_l:
                        sw = lidar_sweep(snap, eng, s, vg=vg)
                        sw.key = todo_l[s.uid]
                        out['lidar'][s.uid] = sw
                if todo_s:
                    from ..core import gnss as G
                    gs = G.GnssSettings.from_dict(getattr(snap.analysis, 'gnss', None))
                    for s in snap.variant.sensors:
                        if s.uid in todo_s:
                            out['sky'][s.uid] = (todo_s[s.uid], G.sky_view(snap, vg, s, gs, snap.variant))
            except Exception as ex:                  # a broken input never takes the view down
                out['error'] = f'{type(ex).__name__}: {ex}'
            try:
                self.sig.done.emit(gen, out)
            except RuntimeError:
                pass
        threading.Thread(target=work, daemon=True).start()

    def _done(self, gen, out):
        if gen != self._gen:
            return
        self._running = False
        if out.get('project') is self.ctx.store.project:
            self.sweeps.update(out['lidar'])
            self.skies.update(out['sky'])
        if out.get('error'):
            print('3D overlays failed:', out['error'])
        if self._pending:
            self._pending = False
            self.timer.start()
        self.ready.emit()

    # ---- drawing
    def point_draws(self) -> list:
        """Point sprites of every visible lidar whose sweep is up to date (a moved lidar's come back once redone)."""
        lk, _ = self.keys()
        dark = THEME.name != 'light'
        out = []
        for u, k in lk.items():
            sw = self.sweeps.get(u)
            if sw is None or sw.key != k:
                continue
            dk = (u, id(sw), dark)
            if dk not in self._draws:
                self._draws = {kk: v for kk, v in self._draws.items() if kk[0] != u}
                self._draws[dk] = sweep_draws(sw, dark)
            out += self._draws[dk]
        return out

    def fresh_sweeps(self) -> list:
        lk, _ = self.keys()
        return [self.sweeps[u] for u, k in lk.items() if u in self.sweeps and self.sweeps[u].key == k]

    def fresh_skies(self) -> list:
        _, sk = self.keys()
        return [self.skies[u][1] for u, k in sk.items() if self.skies.get(u, (None,))[0] == k]

    def dome_draws(self, gl):
        """(translucent TriDraws, LineDraws) of the up-to-date sky domes."""
        _, sk = self.keys()
        dark = THEME.name != 'light'
        tris, lines = [], []
        for u, k in sk.items():
            e = self.skies.get(u)
            if e is None or e[0] != k:
                continue
            sv = e[1]
            dk = (u, id(sv), dark, gl)
            if dk not in self._domes:
                self._domes = {kk: v for kk, v in self._domes.items() if kk[0] != u}
                na, ne = (SKY_AZ, SKY_EL) if gl else (36, 9)
                T, rgba, ln = sky_dome(sv, dark, n_az=na, n_el=ne)
                from .sensorview import color as kind_color
                col = kind_color('gnss', 0.75)
                self._domes[dk] = (tri_draw(T, rgba) if len(T) else None,
                                   [LineDraw(q, col, 1.0) for q in ln])
            td, ld = self._domes[dk]
            if td is not None:
                tris.append(td)
            lines += ld
        return tris, lines
