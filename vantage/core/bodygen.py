"""Procedural vehicle bodies.

The Tata Tigor EV body is lofted from the side profile and front-view half widths
measured on 19 studio photos (wheel-centre calibration, 1.6 px RMS). Templates use
generic profiles with the same lofting code. All bodies are returned in ISO 8855
coordinates with the origin at the rear-axle centre on the ground.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .mesh import Mesh, Part, Zone, ROLE_MOUNTABLE, ROLE_GLASS, ROLE_WHEELS

# ------------------------------------------------------------------ measured Tigor data
TIGOR_U = [0.0, 0.025, 0.0501, 0.0751, 0.1002, 0.1252, 0.1503, 0.1753, 0.2004, 0.2254, 0.2504, 0.2755, 0.3005,
           0.3256, 0.3506, 0.3757, 0.4007, 0.4257, 0.4508, 0.4758, 0.5009, 0.5259, 0.551, 0.576, 0.6011, 0.6261,
           0.6511, 0.6762, 0.7012, 0.7263, 0.7513, 0.7764, 0.8014, 0.8264, 0.8515, 0.8765, 0.9016, 0.9266, 0.9517,
           0.9892, 1.0]
TIGOR_ZT = [0.62, 0.55, 0.709, 1.019, 1.161, 1.203, 1.249, 1.286, 1.32, 1.36, 1.413, 1.468, 1.496, 1.509, 1.52,
            1.531, 1.539, 1.544, 1.546, 1.546, 1.539, 1.533, 1.522, 1.501, 1.467, 1.42, 1.369, 1.31, 1.249, 1.188,
            1.126, 1.085, 1.054, 1.022, 0.985, 0.952, 0.918, 0.876, 0.821, 0.747, 0.62]
TIGOR_ZB = [0.4, 0.207, 0.191, 0.232] + [0.172] * 35 + [0.321, 0.36]
TIGOR_ZS = [0.18, 0.26, 0.34, 0.42, 0.5, 0.58, 0.66, 0.74, 0.82, 0.9, 0.98, 1.06, 1.14, 1.22, 1.3, 1.38, 1.46,
            1.54, 1.58]
TIGOR_HW = [0.838] * 12 + [0.822, 0.791, 0.709, 0.571, 0.518, 0.36, 0.0]
TIGOR_MIRROR_U = 0.6669
TIGOR_HUB_Y = 0.823
TIGOR_WHEEL_R = 0.29


@dataclass
class BodySpec:
    key: str
    name: str
    length: float
    width: float
    height: float
    wheelbase: float
    ground_clearance: float
    rear_overhang: float
    wheel_r: float
    track_half: float                 # hub lateral position (outer face)
    u: list                           # profile stations rear (0) -> front (1)
    top: list                         # top height at stations (m, unscaled for this spec)
    bottom: list
    hw_z: list                        # z levels for half width
    hw: list                          # half widths at hw_z
    mirror_u: float = 0.66
    mirror_z: float = 1.02
    belt_z: float = 0.95
    glass_u: tuple = (0.10, 0.70)     # cabin glass range along u
    plate_front_z: float = 0.55
    plate_rear_z: float = 0.86
    cameras: str = '6-cam'
    second_body: 'BodySpec | None' = None
    second_gap: float = 0.0
    spec: dict = field(default_factory=dict)   # datasheet dimensions for the spec check

    @property
    def front_overhang(self):
        return self.length - self.wheelbase - self.rear_overhang


def tigor_spec() -> BodySpec:
    return BodySpec(key='tigor', name='Tata Tigor EV', length=3.993, width=1.677, height=1.532, wheelbase=2.450,
                    ground_clearance=0.172, rear_overhang=0.778, wheel_r=TIGOR_WHEEL_R, track_half=TIGOR_HUB_Y,
                    u=TIGOR_U, top=TIGOR_ZT, bottom=TIGOR_ZB, hw_z=TIGOR_ZS, hw=TIGOR_HW,
                    mirror_u=TIGOR_MIRROR_U, mirror_z=1.02, belt_z=0.98, glass_u=(0.105, 0.72),
                    plate_front_z=0.546, plate_rear_z=0.858, cameras='6-cam',
                    spec={'Length': 3.993, 'Width': 1.677, 'Height': 1.532, 'Wheelbase': 2.450, 'Ground clearance': 0.172})


def _generic(key, name, L, W, H, wb, gc, ro, wr, kind, cams):
    """Generic profile families scaled to the given box."""
    if kind == 'sedan':
        u = [0, .03, .08, .16, .24, .30, .40, .55, .64, .72, .80, .90, .97, 1]
        t = [.42, .45, .62, .66, .80, .97, 1.0, 1.0, .94, .70, .64, .60, .52, .40]
        glass = (0.17, 0.71)
        belt = 0.62
    elif kind == 'hatch':
        u = [0, .03, .08, .14, .30, .45, .58, .66, .74, .84, .93, .98, 1]
        t = [.45, .55, .80, .97, 1.0, 1.0, .97, .78, .66, .62, .57, .48, .40]
        glass = (0.06, 0.70)
        belt = 0.62
    elif kind == 'suv':
        u = [0, .03, .07, .15, .40, .58, .66, .74, .84, .93, .98, 1]
        t = [.50, .60, .90, .99, 1.0, 1.0, .93, .74, .70, .66, .56, .45]
        glass = (0.05, 0.71)
        belt = 0.62
    elif kind == 'van':
        u = [0, .02, .05, .70, .78, .84, .92, .97, 1]
        t = [.55, .97, 1.0, 1.0, .96, .70, .62, .52, .42]
        glass = (0.70, 0.86)
        belt = 0.55
    else:
        raise ValueError(kind)
    bottom = [min(.35, x) for x in t]
    bottom = [gc / H + 0.08 if i in (0, len(u) - 1) else gc / H for i in range(len(u))]
    hz = [0.0, 0.35, 0.55, 0.62, 0.72, 0.82, 0.92, 0.98, 1.0]
    hwv = [0.49, 0.5, 0.5, 0.49, 0.46, 0.42, 0.37, 0.30, 0.0]
    return BodySpec(key=key, name=name, length=L, width=W, height=H, wheelbase=wb, ground_clearance=gc,
                    rear_overhang=ro, wheel_r=wr, track_half=W / 2 - 0.02,
                    u=u, top=[x * H for x in t], bottom=[x * H for x in bottom],
                    hw_z=[z * H for z in hz], hw=[h * W for h in hwv],
                    mirror_u=glass[1] - 0.04 if kind != 'van' else 0.86, mirror_z=belt * H + 0.08, belt_z=belt * H,
                    glass_u=glass, plate_front_z=0.5, plate_rear_z=0.8, cameras=cams,
                    spec={'Length': L, 'Width': W, 'Height': H, 'Wheelbase': wb, 'Ground clearance': gc})


def template_specs() -> dict[str, BodySpec]:
    truck = _generic('truck', 'Truck + trailer', 6.0, 2.45, 3.2, 3.8, 0.30, 1.2, 0.50, 'van', '8-cam')
    trailer = _generic('trailer', 'Trailer', 7.5, 2.50, 3.6, 5.0, 0.35, 1.3, 0.45, 'van', '8-cam')
    trailer.top = [3.6 * 0.98] * len(trailer.u)
    trailer.glass_u = (2.0, 2.0)
    truck.second_body = trailer
    truck.second_gap = 0.5
    return {
        'sedan': _generic('sedan', 'Sedan', 4.0, 1.7, 1.5, 2.55, 0.17, 0.80, 0.30, 'sedan', '6-cam'),
        'hatchback': _generic('hatchback', 'Hatchback', 3.8, 1.7, 1.5, 2.45, 0.17, 0.62, 0.29, 'hatch', '4-cam'),
        'suv': _generic('suv', 'SUV', 4.4, 1.8, 1.7, 2.70, 0.20, 0.90, 0.34, 'suv', '6-cam'),
        'van': _generic('van', 'Van', 5.3, 2.0, 2.3, 3.30, 0.18, 1.00, 0.33, 'van', '8-cam'),
        'truck': truck,
    }


# ------------------------------------------------------------------ lofting
def _interp(xs, ys, x):
    return float(np.interp(x, xs, ys))


def _box(x0, x1, y0, y1, z0, z1, taper=0.0):
    """Closed box (8 verts, 12 tris); taper narrows the top in y."""
    yc = (y0 + y1) / 2
    v = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                  [x0, y0 + taper * (yc - y0), z1], [x1, y0 + taper * (yc - y0), z1],
                  [x1, y1 - taper * (y1 - yc), z1], [x0, y1 - taper * (y1 - yc), z1]], float)
    f = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4],
                  [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]])
    return v, f


def _cylinder_y(cx, cz, y0, y1, r, seg=24, r_in=None):
    """Wheel: tyre tread + side walls as a closed cylinder along y."""
    a = np.linspace(0, 2 * math.pi, seg, endpoint=False)
    ring0 = np.stack([cx + r * np.cos(a), np.full(seg, y0), cz + r * np.sin(a)], 1)
    ring1 = np.stack([cx + r * np.cos(a), np.full(seg, y1), cz + r * np.sin(a)], 1)
    c0 = np.array([[cx, y0, cz]])
    c1 = np.array([[cx, y1, cz]])
    v = np.concatenate([ring0, ring1, c0, c1])
    f = []
    for i in range(seg):
        j = (i + 1) % seg
        f += [[i, seg + j, j], [i, seg + i, seg + j]]
        f += [[2 * seg, i, j], [2 * seg + 1, seg + j, seg + i]]
    return v, np.array(f)


def loft_body(spec: BodySpec, x_offset: float, nx=88, nz=16):
    """Body shell for one spec. x_offset = x of the rear tip in the output frame."""
    L, H = spec.length, spec.height
    us = np.linspace(0, 1, nx)
    xs = x_offset + us * L
    ref_h = max(spec.top)
    sz = H / ref_h if spec.key == 'tigor' else 1.0
    sy = spec.width / 1.677 if spec.key == 'tigor' else 1.0
    rear_axle_x = x_offset + spec.rear_overhang
    hubs = [rear_axle_x, rear_axle_x + spec.wheelbase]
    rc = 0.30 * L / 3.993 if spec.key == 'tigor' else 0.25
    arch_r = spec.wheel_r + 0.06
    rings = []
    for i, u in enumerate(us):
        x = xs[i]
        top = _interp(spec.u, spec.top, u) * sz
        bot_raw = _interp(spec.u, spec.bottom, u)
        bot = spec.ground_clearance if abs(bot_raw - spec.ground_clearance) < 2e-3 else max(spec.ground_clearance, bot_raw * sz)
        for hx in hubs:
            dx = abs(x - hx)
            if dx < arch_r:
                bot = max(bot, spec.wheel_r + math.sqrt(arch_r ** 2 - dx ** 2) * 0.92)
        bot = min(bot, top - 0.05)
        s = max(0.0, (abs(x - (x_offset + L / 2)) - (L / 2 - rc)) / rc)
        tap = 0.0 if s >= 1 else (1 - s ** 3) ** (1 / 3)
        tap = max(tap, 0.02)
        belt = spec.belt_z
        if top > belt + 0.10 and bot < belt - 0.10:
            # put a ring row exactly on the belt line so the glass edge is clean
            n_lo = nz // 2
            lo = bot + (belt - bot) * np.linspace(0, 1, n_lo, endpoint=False)
            hi = belt + (top - belt) * (1 - np.cos(np.linspace(0, math.pi / 2, nz - n_lo))) ** 0.8
            zk = np.concatenate([lo, hi])
        else:
            zk = bot + (top - bot) * (1 - np.cos(np.linspace(0, math.pi / 2, nz))) ** 0.8
        zk[0], zk[-1] = bot, top
        wk = np.array([_interp(spec.hw_z, spec.hw, z / sz) * sy for z in zk]) * tap
        wk = np.maximum(wk, 0.004)
        ring = [(x, -w, z) for w, z in zip(wk, zk)] + [(x, w, z) for w, z in zip(wk[::-1], zk[::-1])]
        rings.append(ring)
    R = np.array(rings)                     # (nx, 2nz, 3)
    n = R.shape[1]
    verts = R.reshape(-1, 3)
    faces = []
    for i in range(nx - 1):
        a, b = i * n, (i + 1) * n
        for j in range(n):
            k = (j + 1) % n
            faces += [[a + j, b + j, b + k], [a + j, b + k, a + k]]
    # end caps
    base = len(verts)
    cr = R[0].mean(axis=0)
    cf = R[-1].mean(axis=0)
    verts = np.concatenate([verts, [cr, cf]])
    for j in range(n):
        k = (j + 1) % n
        faces.append([base, j, k])
        faces.append([base + 1, (nx - 1) * n + k, (nx - 1) * n + j])
    return verts, np.array(faces, np.int64)


def build_body(spec: BodySpec) -> tuple[Mesh, dict]:
    """Build body + wheels + mirrors as a Mesh in ISO 8855 rear-axle coordinates.
    Returns (mesh, info) where info has landmark positions."""
    parts = [Part('Body', 'paint_matte', 1.0, (0.78, 0.78, 0.76), ROLE_MOUNTABLE),
             Part('Glass', 'glass_tint', 0.35, (0.18, 0.19, 0.2), ROLE_GLASS),
             Part('Wheels', 'tyre, rim', 1.0, (0.13, 0.13, 0.13), ROLE_WHEELS),
             Part('Mirrors', 'paint_matte', 1.0, (0.78, 0.78, 0.76), ROLE_MOUNTABLE)]
    V, F, FP = [], [], []

    def add(v, f, part):
        off = sum(len(x) for x in V)
        V.append(np.asarray(v, float))
        F.append(np.asarray(f, np.int64) + off)
        FP.append(np.full(len(f), part, np.int32) if np.ndim(part) == 0 else np.asarray(part, np.int32))

    x_rear = -spec.rear_overhang
    bodies = [(spec, x_rear)]
    if spec.second_body is not None:
        tb = spec.second_body
        bodies.append((tb, x_rear - spec.second_gap - tb.length))
    landmarks = {}
    for bi, (bs, xr) in enumerate(bodies):
        v, f = loft_body(bs, xr)
        # glass faces: above the belt line inside the cabin span, not the roof strip
        tri = v[f]
        c = tri.mean(axis=1)
        u = (c[:, 0] - xr) / bs.length
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
        sz = bs.height / max(bs.top) if bs.key == 'tigor' else 1.0
        top_c = np.interp(u, bs.u, bs.top) * sz
        glass = (u > bs.glass_u[0]) & (u < bs.glass_u[1]) & (tri[:, :, 2].min(axis=1) > bs.belt_z - 1e-6) & (tri[:, :, 2].max(axis=1) < top_c - 0.05) & (np.abs(n[:, 2]) < 0.93)
        add(v, f, np.where(glass, 1, 0))
        # wheels
        ra = xr + bs.rear_overhang
        hubs_x = [ra, ra + bs.wheelbase]
        if bs.key == 'trailer':
            hubs_x = [ra, ra + 1.3]
        for hx in hubs_x:
            for sgn in (1, -1):
                y_out = sgn * (bs.track_half + 0.015)
                y_in = sgn * (bs.track_half + 0.015 - 0.185 * (bs.wheel_r / 0.29))
                wv, wf = _cylinder_y(hx, bs.wheel_r, min(y_in, y_out), max(y_in, y_out), bs.wheel_r)
                add(wv, wf, 2)
                if bi == 0:
                    tag = ('R' if hx == hubs_x[0] else 'F') + ('L' if sgn > 0 else 'R')
                    landmarks[f'Wheel centre {tag}'] = (hx, sgn * (bs.track_half + 0.015), bs.wheel_r)
        if bi == 0 and bs.key != 'trailer':
            # mirrors: wedge boxes at the A-pillar base
            mx = xr + bs.mirror_u * bs.length
            w_at = np.interp(bs.mirror_z / sz, bs.hw_z, bs.hw) * (bs.width / 1.677 if bs.key == 'tigor' else 1.0)
            for sgn in (1, -1):
                y0, y1 = w_at - 0.04, w_at + 0.18
                bv, bf = _box(mx - 0.11, mx + 0.09, y0, y1, bs.mirror_z - 0.055, bs.mirror_z + 0.085, taper=0.15)
                if sgn < 0:
                    bv[:, 1] *= -1
                    bf = bf[:, ::-1]
                add(bv, bf, 3)
                landmarks['Mirror tip ' + ('L' if sgn > 0 else 'R')] = (mx, sgn * y1, bs.mirror_z)
    verts = np.concatenate(V)
    faces = np.concatenate(F)
    fpart = np.concatenate(FP)
    mesh = Mesh(verts, faces, fpart, parts)
    _auto_zones(mesh, spec)
    landmarks['Front bumper centre'] = (spec.length - spec.rear_overhang, 0.0, spec.plate_front_z)
    landmarks['Rear bumper centre'] = (-spec.rear_overhang, 0.0, spec.plate_rear_z)
    info = {'landmarks': landmarks, 'mirror_x': x_rear + spec.mirror_u * spec.length}
    return mesh, info


def _auto_zones(mesh: Mesh, spec: BodySpec):
    """Mount zones on the procedural body (same rules a user would paint)."""
    c, n = mesh.centroids, mesh.normals
    fp = mesh.face_part
    L, W = spec.length, spec.width
    xf, xr = L - spec.rear_overhang, -spec.rear_overhang
    zones = [Zone('Grille band above plate'), Zone('Rear bumper top edge'), Zone('Mirror housings (2)'),
             Zone('Rear quarters'), Zone('Glass (auto)', 'forbidden', auto=True), Zone('Door shut lines', 'forbidden')]
    fz = np.full(len(c), -1, np.int16)
    body = fp == 0
    pf = spec.plate_front_z
    fz[body & (c[:, 0] > xf - 0.14) & (c[:, 2] > pf - 0.02) & (c[:, 2] < pf + 0.16) & (np.abs(c[:, 1]) < 0.45 * W / 1.677 * 1.0) & (n[:, 0] > 0.2)] = 0
    pr = 0.56 if spec.key == 'tigor' else 0.55
    fz[body & (c[:, 0] < xr + 0.14) & (c[:, 2] > pr - 0.08) & (c[:, 2] < pr + 0.08) & (np.abs(c[:, 1]) < 0.5) & (n[:, 0] < -0.1)] = 1
    fz[fp == 3] = 2
    fz[body & (np.abs(c[:, 0]) < 0.35) & (c[:, 2] > 0.80) & (c[:, 2] < 1.0) & (np.abs(n[:, 1]) > 0.55)] = 3
    fz[fp == 1] = 4
    # door seams on the Tigor: A-post base, B-post, C-post (x in rear-axle frame)
    seams = [1.55, 0.72, 0.30] if spec.key == 'tigor' else [xr + 0.70 * L, xr + 0.50 * L, xr + 0.30 * L]
    for sx in seams:
        fz[body & (np.abs(c[:, 0] - sx) < 0.03) & (np.abs(n[:, 1]) > 0.6) & (c[:, 2] > 0.25) & (c[:, 2] < spec.belt_z + 0.02)] = 5
    mesh.zones = zones
    mesh.face_zone = fz
