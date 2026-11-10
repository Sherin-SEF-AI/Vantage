"""Procedural vehicle bodies.

The Tata Tigor EV body is lofted from the side profile and front-view half widths
measured on 19 studio photos (wheel-centre calibration, 1.6 px RMS). The vehicle library's
templates (core/vehicles.py) loft their family's profile, scaled to each vehicle's stated
dimensions, with the same code. All bodies are returned in ISO 8855
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
    drawn: list = field(default_factory=list)  # figures the maker does not state, drawn from the family's profile

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
    elif kind == 'van':                  # a one-piece van: the high roof runs over the cab to the windscreen
        u = [0, .02, .05, .76, .81, .88, .94, .98, 1]
        t = [.55, .97, 1.0, 1.0, .96, .62, .50, .44, .36]
        glass = (0.74, 0.90)
        belt = 0.50
    elif kind == 'box':                  # a cab with a taller box behind it (a tractor, a box truck)
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
                    mirror_u=glass[1] - 0.04 if kind not in ('van', 'box') else glass[1], mirror_z=belt * H + 0.08,
                    belt_z=belt * H,
                    glass_u=glass, plate_front_z=0.5, plate_rear_z=0.8, cameras=cams,
                    spec={'Length': L, 'Width': W, 'Height': H, 'Wheelbase': wb, 'Ground clearance': gc})


def template_specs() -> dict[str, BodySpec]:
    """The template bodies of the vehicle library (core/vehicles.py), by vehicle id: each family's profile scaled to
    the vehicle's stated dimensions."""
    from .vehicles import library, body_spec
    return {v.id: body_spec(v) for v in library()}


def template_spec(key: str) -> BodySpec | None:
    """The template body of a library vehicle id or an old template key ('sedan', 'suv', ...)."""
    from .vehicles import resolve, body_spec
    v = resolve(key)
    return body_spec(v) if v is not None else None


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



# ------------------------------------------------------------------ the library's bodies (shaped)
def _smooth(us, ys):
    """A shape-preserving cubic (PCHIP) through the profile's stations: no overshoot, no kinks between them."""
    from scipy.interpolate import PchipInterpolator
    return PchipInterpolator(np.asarray(us, float), np.asarray(ys, float), extrapolate=True)


def _plan_round(x, x0, x1, rf, rr):
    """The plan-view width factor at x of a body from x0 (rear) to x1 (front): 1 along the sides, closing over a
    rounded corner of radius rf at the nose and rr at the tail (a superellipse, so the ends stay full)."""
    if x > x1 - rf:
        s = (x - (x1 - rf)) / rf
    elif x < x0 + rr:
        s = ((x0 + rr) - x) / rr
    else:
        return 1.0
    s = min(max(s, 0.0), 1.0)
    return max((1.0 - s ** 2.6) ** (1 / 2.6), 0.035)


def _section(w, bot, top, belt, cabin, nz, tumble=0.22, tuck=0.05):
    """One cross-section's (y >= 0) half outline from the bottom centre over the side to the top centre, nz points:
    the sill tucked under, the side bulging slightly to the belt, the greenhouse leaning in by `tumble` where the
    station has one (cabin > 0), and the top rounded over to the centre (crown)."""
    h = max(top - bot, 0.02)
    r_top = min(0.16 if cabin else 0.07, 0.45 * h)
    r_bot = min(0.06, 0.3 * h)
    pts = []
    # bottom: the floor from the centre to the sill corner
    nb = 3
    for k in range(nb):
        t = k / nb
        pts.append((w * (1 - tuck) * (0.25 + 0.75 * t) - r_bot * t, bot))
    nside = nz - nb - 6
    zb = bot + r_bot
    z_belt = min(max(belt, zb + 0.02), top - r_top - 0.01)
    gh_top = top - r_top
    # a row exactly on the belt line (a clean window edge), the rest spread below and above it
    n_lo = max(2, int(round(nside * 0.55)))
    zs = list(zb + (z_belt - zb) * np.linspace(0, 1, n_lo))
    zs += list(z_belt + (gh_top - z_belt) * np.linspace(0, 1, nside - n_lo + 1)[1:])
    for z in zs:
        if z <= z_belt + 1e-9:
            f = (z - zb) / max(z_belt - zb, 1e-6)                  # 0 at the sill, 1 at the belt
            ww = w * (1 - tuck * (1 - f) ** 2.2) * (1 + 0.012 * math.sin(math.pi * f))
        else:
            g = (z - z_belt) / max(gh_top - z_belt, 1e-6)
            ww = w * (1 - (tumble if cabin else 0.04) * g ** 1.15)
        pts.append((ww, z))
    widest = max(x for x, _ in pts)                       # the section is exactly w at its widest
    if widest > 0:
        pts = [(x * w / widest, z) for x, z in pts]
    # round over the top: a quarter ellipse from the side to the centre, with a slight crown
    w_edge = pts[-1][0]
    for k in range(1, 7):
        a = k / 6 * math.pi / 2
        pts.append((w_edge * math.cos(a) ** 0.9 if k < 6 else 0.0, gh_top + r_top * math.sin(a)))
    return pts[:nz]


def loft_shaped(spec: BodySpec, x_offset: float, nx=160, nz=30):
    """A library vehicle's body shell (spec.key != 'tigor'): its family's side profile smoothed, the nose and tail
    rounded in plan, each section tucked at the sill and leaning in above the belt, wheel arches cut over the hubs.
    Returns (verts, faces, info) with the hood and deck heights for the lights."""
    L = spec.length
    top_f = _smooth(spec.u, spec.top)
    bot_f = _smooth(spec.u, spec.bottom)
    van = spec.key in ('trailer',) or spec.glass_u[0] >= 0.6 or spec.height > 2.2
    rf = min(0.10 * L, 0.42) if not van else min(0.05 * L, 0.22)
    rr = min(0.07 * L, 0.30) if not van else 0.12
    x0, x1 = x_offset, x_offset + L
    rear_axle = x_offset + spec.rear_overhang
    hubs = [rear_axle, rear_axle + spec.wheelbase]
    arch_r = spec.wheel_r + 0.065
    # stations: denser at the ends, where the shape turns
    t = np.linspace(0, 1, nx)
    us = 0.6 * (0.5 - 0.5 * np.cos(math.pi * t)) + 0.4 * t
    rings, lab_glass, lab_trim = [], [], []
    dtop = top_f.derivative()
    for u in us:
        x = x_offset + u * L
        top = float(top_f(u))
        bot = max(spec.ground_clearance, float(bot_f(u)))
        for hx in hubs:
            dx = abs(x - hx)
            if dx < arch_r:
                bot = max(bot, spec.wheel_r + math.sqrt(arch_r ** 2 - dx ** 2) * 0.93)
        bot = min(bot, top - 0.06)
        w = spec.width / 2 * _plan_round(x, x0, x1, rf, rr)
        cabin = spec.glass_u[0] - 0.02 < u < spec.glass_u[1] + 0.02 and top > spec.belt_z + 0.12
        half = _section(w, bot, top, spec.belt_z, cabin, nz)
        ring = [(x, -yy, zz) for yy, zz in half] + [(x, yy, zz) for yy, zz in half[::-1]]
        rings.append(ring)
        # each vertex's role, from where it sits in the loft (station, section row): faces whose three vertices
        # agree are glass or trim, so the edges follow the mesh lines instead of tearing across them
        slope = abs(float(dtop(u))) / L                   # the roof line's slope here (rise per metre)
        g_side, g_top, trim = [], [], []
        for k, (yy, zz) in enumerate(half):
            cap = k >= nz - 6
            above = zz > spec.belt_z + 0.015
            g_side.append(above and not cap and cabin and spec.glass_u[0] < u < spec.glass_u[1])
            g_top.append(above and slope > 0.35 and spec.glass_u[0] - 0.06 < u < spec.glass_u[1] + 0.05
                         and yy < 0.80 * w and (cap or not cabin))
            trim.append(zz < spec.ground_clearance + 0.075)       # the sills (the rows line up along the sides)
        lab_glass.append([a or b for a, b in zip(g_side, g_top)])
        lab_trim.append(trim)
    R = np.array(rings)
    nxs, n = R.shape[0], R.shape[1]
    verts = R.reshape(-1, 3)
    faces = []
    for i in range(nxs - 1):
        a, b = i * n, (i + 1) * n
        for j in range(n):
            k = (j + 1) % n
            faces += [[a + j, b + j, b + k], [a + j, b + k, a + k]]
    base = len(verts)
    verts = np.concatenate([verts, [R[0].mean(axis=0), R[-1].mean(axis=0)]])
    for j in range(n):
        k = (j + 1) % n
        faces.append([base, j, k])
        faces.append([base + 1, (nxs - 1) * n + k, (nxs - 1) * n + j])
    def per_vertex(lab):
        a = np.array([row + row[::-1] for row in lab], bool).reshape(-1)     # the ring mirrors its half
        return np.concatenate([a, [False, False]])                          # the two end-cap centres

    info = {'hood_z': float(top_f(0.97)), 'deck_z': float(top_f(0.03)), 'x0': x0, 'x1': x1,
            'v_glass': per_vertex(lab_glass), 'v_trim': per_vertex(lab_trim)}
    return verts, np.array(faces, np.int64), info


def _wheel(cx, cz, y_out, y_in, r, seg=40):
    """A tyre (tread with rounded shoulders and a sidewall down to the rim) and its rim (a dished disc with a hub),
    as two (verts, faces) pairs. y_out is the outer face."""
    sgn = 1.0 if y_out > y_in else -1.0
    wdt = abs(y_out - y_in)
    a = np.linspace(0, 2 * math.pi, seg, endpoint=False)
    ca, sa = np.cos(a), np.sin(a)
    r_rim = 0.66 * r
    # tyre profile from the inner rim edge round the tread to the outer rim edge: (radius, y)
    prof = [(r_rim, y_in), (0.93 * r, y_in), (0.985 * r, y_in + sgn * 0.12 * wdt), (r, y_in + sgn * 0.3 * wdt),
            (r, y_out - sgn * 0.3 * wdt), (0.985 * r, y_out - sgn * 0.12 * wdt), (0.93 * r, y_out),
            (r_rim, y_out)]
    tv = np.concatenate([np.stack([cx + rr * ca, np.full(seg, yy), cz + rr * sa], 1) for rr, yy in prof])
    tf = []
    for p in range(len(prof) - 1):
        for i in range(seg):
            j = (i + 1) % seg
            q0, q1 = p * seg, (p + 1) * seg
            tf += [[q0 + i, q1 + i, q1 + j], [q0 + i, q1 + j, q0 + j]]
    # rim: a disc recessed into the tyre with spokes suggested by a smaller hub
    y_face = y_out - sgn * 0.02
    rings = [(r_rim, y_out - sgn * 0.004), (0.92 * r_rim, y_face), (0.30 * r_rim, y_face - sgn * 0.01),
             (0.18 * r_rim, y_face + sgn * 0.006)]
    rv = np.concatenate([np.stack([cx + rr * ca, np.full(seg, yy), cz + rr * sa], 1) for rr, yy in rings]
                        + [np.array([[cx, y_face + sgn * 0.008, cz]])])
    rf = []
    for p in range(len(rings) - 1):
        for i in range(seg):
            j = (i + 1) % seg
            q0, q1 = p * seg, (p + 1) * seg
            rf += [[q0 + i, q1 + i, q1 + j], [q0 + i, q1 + j, q0 + j]]
    c = len(rings) * seg
    last = (len(rings) - 1) * seg
    for i in range(seg):
        rf.append([last + i, c, last + (i + 1) % seg])
    tf, rf = np.array(tf), np.array(rf)
    if sgn < 0:                                   # keep the outward winding on the right-hand wheels
        tf, rf = tf[:, ::-1], rf[:, ::-1]
    return (tv, tf), (rv, rf)


def _lens_decal(shell, x_from, d, yc, zc, ay, az, nu=20, nv=8, lift=0.006):
    """A lamp lens: an elliptical grid (half-axes ay, az about y=yc, z=zc) cast along x (direction d) onto the shell,
    lifted `lift` towards the viewer; cells whose rays miss are left out."""
    th = np.linspace(0, 2 * math.pi, nu, endpoint=False)
    rs = np.linspace(0, 1, nv)
    pts = [(yc, zc)] + [(yc + ay * r * math.cos(t), zc + az * r * math.sin(t)) for r in rs[1:] for t in th]
    O = np.array([[x_from, y, z] for y, z in pts])
    D = np.tile([d, 0.0, 0.0], (len(O), 1))
    t, fi = shell.raycast(O, D)
    ok = fi >= 0
    if ok.any():                   # a ray past the nose's edge hits the body's side far behind: not part of the lens
        ok &= t < float(np.min(t[ok])) + 0.30
    nrm_hit = shell.normals[np.maximum(fi, 0)]
    nrm_hit = np.where((nrm_hit @ np.array([-d, 0.0, 0.0]))[:, None] < 0, -nrm_hit, nrm_hit)   # towards the viewer
    V = O + D * np.where(ok, t, 0.0)[:, None] + nrm_hit * lift
    F = []
    for i in range(nu):                                   # the centre fan
        a, b = 1 + i, 1 + (i + 1) % nu
        F.append([0, a, b])
    for k in range(nv - 2):
        r0, r1 = 1 + k * nu, 1 + (k + 1) * nu
        for i in range(nu):
            j = (i + 1) % nu
            F += [[r0 + i, r1 + i, r1 + j], [r0 + i, r1 + j, r0 + j]]
    F = np.array(F, np.int64)
    F = F[ok[F].all(axis=1)]
    if d > 0:                                             # face the viewer at the tail
        F = F[:, ::-1]
    tri = V[F]
    nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    if len(F) and (nrm[:, 0] * -d).mean() < 0:
        F = F[:, ::-1]
    return V, F


def _ellipsoid(c, half, seg=14, ring=8):
    """A closed ellipsoid about c with half-axes half (x, y, z)."""
    v = [[c[0], c[1], c[2] - half[2]]]
    for k in range(1, ring):
        th = math.pi * k / ring
        for i in range(seg):
            ph = 2 * math.pi * i / seg
            v.append([c[0] + half[0] * math.sin(th) * math.cos(ph), c[1] + half[1] * math.sin(th) * math.sin(ph),
                      c[2] - half[2] * math.cos(th)])
    v.append([c[0], c[1], c[2] + half[2]])
    f = []
    for i in range(seg):
        f.append([0, 1 + (i + 1) % seg, 1 + i])
    for k in range(ring - 2):
        a, b = 1 + k * seg, 1 + (k + 1) * seg
        for i in range(seg):
            j = (i + 1) % seg
            f += [[a + i, a + j, b + j], [a + i, b + j, b + i]]
    top = len(v) - 1
    a = 1 + (ring - 2) * seg
    for i in range(seg):
        f.append([a + i, a + (i + 1) % seg, top])
    return np.array(v, float), np.array(f, np.int64)

GLASS_BELOW_ROOF = 0.05      # m: the side glass's upper edge under the roof line


def _glass_split(v, f, bs, xr, sz):
    """The lofted body's glass faces: inside the cabin span (glass_u), above the belt line, below the roof line by
    GLASS_BELOW_ROOF and not facing up or down. The span's front and rear edges cross each quad of the loft, and the
    upper edge crosses its rows where the roof slopes (the A- and C-pillars), so each triangle an edge crosses is cut
    along it, in the triangle's own plane: the surface stays exactly as it was (the analysis sees the same body) and
    the glass edges are straight lines, not the stairs of whole triangles they used to be. Returns (vertices, faces,
    glass mask)."""
    v = np.asarray(v, float)
    f = np.asarray(f, np.int64)
    L = bs.length
    fields = (lambda p: (p[..., 0] - xr) - bs.glass_u[0] * L,                       # behind the span's rear edge: < 0
              lambda p: bs.glass_u[1] * L - (p[..., 0] - xr),                       # past its front edge: < 0
              lambda p: np.interp((p[..., 0] - xr) / L, bs.u, bs.top) * sz - GLASS_BELOW_ROOF - p[..., 2])
    tri = v[f]
    nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
    # faces that can be glass at all (above the belt, which is a row of the loft, and not facing up or down)
    cand = (tri[:, :, 2].min(axis=1) > bs.belt_z - 1e-6) & (np.abs(nrm[:, 2]) < 0.93)
    for field in fields:
        v, f, cand = _cut(v, f, cand, field)
    tri = v[f]
    inside = np.all([(fl(tri) >= -1e-9).all(axis=1) for fl in fields], axis=0)
    return v, f, cand & inside


def _cut(v, f, cand, field):
    """Cut every candidate triangle that field's zero line crosses into three, in its own plane (new vertices on its
    edges where the field is zero): (vertices, faces, candidate mask)."""
    d = field(v[f])
    d = np.where(np.abs(d) < 5e-4, -5e-4, d)        # a vertex within 0.5 mm of the line counts as outside: no slivers
    pos = d > 0
    cross = np.flatnonzero(cand & pos.any(axis=1) & ~pos.all(axis=1))
    if not len(cross):
        return v, f, cand
    keep = np.ones(len(f), bool)
    keep[cross] = False
    nv, nf = [v], []
    base = len(v)
    for fi in cross:
        ids, dd = f[fi], d[fi]
        side = dd > 0
        k = int(np.flatnonzero(side)[0]) if side.sum() == 1 else int(np.flatnonzero(~side)[0])
        a, b, c = ids[k], ids[(k + 1) % 3], ids[(k + 2) % 3]
        da, db, dc = dd[k], dd[(k + 1) % 3], dd[(k + 2) % 3]
        pp = v[a] + (v[b] - v[a]) * (da / (da - db))
        qq = v[a] + (v[c] - v[a]) * (da / (da - dc))
        nv.append(np.array([pp, qq]))
        ip, iq = base, base + 1
        base += 2
        nf += [[a, ip, iq], [ip, b, c], [ip, c, iq]]
    f2 = np.concatenate([f[keep], np.asarray(nf, np.int64)])
    c2 = np.concatenate([cand[keep], np.ones(len(nf), bool)])
    return np.concatenate(nv), f2, c2


def build_body(spec: BodySpec) -> tuple[Mesh, dict]:
    """Build body + wheels + mirrors as a Mesh in ISO 8855 rear-axle coordinates.
    Returns (mesh, info) where info has landmark positions. The measured Tigor keeps its lofted rings; a library
    vehicle's body is shaped (loft_shaped) with tyres and rims, lamps, black trim and mirror pods."""
    shaped = spec.key != 'tigor'
    parts = [Part('Body', 'paint_matte', 1.0, (0.78, 0.78, 0.76), ROLE_MOUNTABLE),
             Part('Glass', 'glass_tint', 0.35, (0.18, 0.19, 0.2), ROLE_GLASS),
             Part('Wheels', 'tyre, rim', 1.0, (0.13, 0.13, 0.13), ROLE_WHEELS),
             Part('Mirrors', 'paint_matte', 1.0, (0.78, 0.78, 0.76), ROLE_MOUNTABLE)]
    if shaped:
        parts += [Part('Wheel rims', 'alloy', 1.0, (0.72, 0.73, 0.75), ROLE_WHEELS),
                  Part('Head lamps', 'lamp lens, smoked', 1.0, (0.30, 0.32, 0.36), ROLE_MOUNTABLE),
                  Part('Tail lamps', 'lamp lens, red', 1.0, (0.62, 0.05, 0.04), ROLE_MOUNTABLE),
                  Part('Black trim', 'plastic, black', 1.0, (0.06, 0.06, 0.06), ROLE_MOUNTABLE)]
    P_BODY, P_GLASS, P_TYRE, P_MIRROR, P_RIM, P_HEAD, P_TAIL, P_TRIM = range(8)
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
        if shaped:
            v, f, li = loft_shaped(bs, xr)
        else:
            v, f = loft_body(bs, xr)
        # glass faces: above the belt line inside the cabin span, not the roof strip
        tri = v[f]
        c = tri.mean(axis=1)
        u = (c[:, 0] - xr) / bs.length
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
        sz = bs.height / max(bs.top) if bs.key == 'tigor' else 1.0
        top_c = (_smooth(bs.u, bs.top)(u) if shaped else np.interp(u, bs.u, bs.top)) * sz
        if shaped:
            # the loft's vertex roles: a face is glass when its three vertices are (side windows above the belt,
            # windscreen and backlight where the roof line slopes, inside the pillars)
            glass = li['v_glass'][f].all(axis=1)
        else:
            v, f, glass = _glass_split(v, f, bs, xr, sz)
            tri = v[f]
            c = tri.mean(axis=1)
        part = np.where(glass, P_GLASS, P_BODY)
        if shaped and bs.key != 'trailer':
            W = bs.width
            xf_, xr_ = li['x1'], li['x0']
            ay = np.abs(c[:, 1])
            z = c[:, 2]
            # black trim: the sills and the lower bumpers (below the camera bands), not on the glass
            trim = li['v_trim'][f].all(axis=1) & (part == P_BODY)
            part = np.where(trim & (part == P_BODY), P_TRIM, part)
        add(v, f, part)
        if shaped and bs.key != 'trailer':
            # lamps: a lens surface projected onto the nose (tail) along x, 2 mm proud of the paint
            hz = li['hood_z']
            tz = min(li['deck_z'], bs.height * 0.75)
            shell = Mesh(v, f)
            for (yc, zc, ay_, az_), x_from, d, pidx in (((0.34 * bs.width, hz - 0.13, 0.13 * bs.width, 0.06),
                                                         li['x1'] + 1.0, -1.0, P_HEAD),
                                                        ((0.36 * bs.width, tz - 0.14, 0.12 * bs.width, 0.065),
                                                         li['x0'] - 1.0, 1.0, P_TAIL)):
                for sgn in (1, -1):
                    lv, lf = _lens_decal(shell, x_from, d, sgn * yc, zc, ay_, az_)
                    if len(lf):
                        add(lv, lf, pidx)
        # wheels
        ra = xr + bs.rear_overhang
        hubs_x = [ra, ra + bs.wheelbase]
        if bs.key == 'trailer':
            hubs_x = [ra, ra + 1.3]
        for hx in hubs_x:
            for sgn in (1, -1):
                y_out = sgn * (bs.track_half + 0.015)
                y_in = sgn * (bs.track_half + 0.015 - 0.185 * (bs.wheel_r / 0.29))
                if shaped:
                    (tv, tf), (rv, rf_) = _wheel(hx, bs.wheel_r, y_out, y_in, bs.wheel_r)
                    add(tv, tf, P_TYRE)
                    add(rv, rf_, P_RIM)
                else:
                    wv, wf = _cylinder_y(hx, bs.wheel_r, min(y_in, y_out), max(y_in, y_out), bs.wheel_r)
                    add(wv, wf, P_TYRE)
                if bi == 0:
                    tag = ('R' if hx == hubs_x[0] else 'F') + ('L' if sgn > 0 else 'R')
                    landmarks[f'Wheel centre {tag}'] = (hx, sgn * (bs.track_half + 0.015), bs.wheel_r)
        if bi == 0 and bs.key != 'trailer':
            # mirrors at the A-pillar base: a wedge (the Tigor), or a pod on a short arm (the library's bodies)
            mx = xr + bs.mirror_u * bs.length
            w_at = np.interp(bs.mirror_z / sz, bs.hw_z, bs.hw) * (bs.width / 1.677 if bs.key == 'tigor' else 1.0)
            if shaped:
                w_at = min(w_at, bs.width / 2 * 0.93)
            for sgn in (1, -1):
                y0, y1 = w_at - 0.04, w_at + 0.18
                if shaped:
                    pv, pf_ = _ellipsoid((mx - 0.01, (y0 + 0.06 + y1) / 2, bs.mirror_z + 0.015),
                                         (0.10, (y1 - y0 - 0.06) / 2, 0.07))
                    av, af = _box(mx - 0.05, mx + 0.04, y0, y0 + 0.08, bs.mirror_z - 0.03, bs.mirror_z + 0.02)
                    bv = np.concatenate([pv, av])
                    bf = np.concatenate([pf_, af + len(pv)])
                else:
                    bv, bf = _box(mx - 0.11, mx + 0.09, y0, y1, bs.mirror_z - 0.055, bs.mirror_z + 0.085, taper=0.15)
                if sgn < 0:
                    bv = bv.copy()
                    bv[:, 1] *= -1
                    bf = bf[:, ::-1]
                add(bv, bf, P_MIRROR)
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
    body = (fp == 0) | (fp == 7)                      # paint, and the library bodies' black trim
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
