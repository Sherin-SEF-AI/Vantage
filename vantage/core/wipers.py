"""Windscreen wipers: where the blades sweep on the windscreen, and whether the cameras behind it look through glass
the wipers clear.

The windscreen is the faces core.materials classes as laminated windscreen glass facing forward (a glass part, or
the band between the side windows and the bonnet of a body modelled as one part). Their plane goes through their
area-weighted centroid, square to their area-weighted mean normal. Positions on the glass are (w, h) in metres in
that plane: w across, positive to the vehicle's left, h up the glass from its bottom edge, with w = 0 at the middle
of the bottom edge. A wiper is a pivot (w, h; below the glass, in the cowl, is h < 0), an arm length to the blade's
centre, a blade length along the arm, its direction at rest (degrees from +w towards +h) and how far it turns from
there towards the top: it clears the ring sector between radii arm − blade/2 and arm + blade/2 over that turn. That
is the classic simplified wipe pattern (a real arm's blade is close to, not exactly on, the arm's line, and a
pantograph arm translates): enter your vehicle's figures.

Typical layouts sized from the windscreen (WiperSettings.layout): tandem (both arms park along the bottom pointing to
the passenger side and turn up towards the driver's side: the driver's pivot 22 % of the bottom width from the middle
towards the driver, reaching the top middle where windscreen cameras sit; the passenger's 12 % the other way),
opposed (pivots at the two bottom corners, turning up towards each other) and single (one arm at the middle). Left-
or right-hand drive mirrors them.

A camera behind the windscreen: its view's footprint on the glass (the image sampled on a grid, each ray met with the
windscreen's faces) mapped into the plane; the share of it inside some blade's sweep is what the wipers keep clear.
The glass outside the sweep collects rain and dirt as an exposed lens would: core.exposure's dirt model (wheel spray,
wake, frontal impact, attitude, height) scores it at those points.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np

LAYOUTS = ('tandem', 'opposed', 'single', 'custom')
LAYOUT_LABEL = {'tandem': 'Tandem', 'opposed': 'Opposed', 'single': 'Single arm', 'custom': 'Custom'}
DRIVES = ('lhd', 'rhd')
DRIVE_LABEL = {'lhd': 'Left-hand drive', 'rhd': 'Right-hand drive'}
CELL_M = 0.01                     # raster of the glass for its wiped share
PIVOT_H_M = -0.05                 # typical pivots sit below the glass's bottom edge, in the cowl
FOOT_GRID = (32, 24)              # camera rays for the footprint on the glass
FOOT_NEAR_M = 1.0
ARM_LIMITS = {'pivot_w': (-3.0, 3.0), 'pivot_h': (-1.0, 3.0), 'arm_m': (0.05, 2.0), 'blade_m': (0.05, 1.6),
              'park_deg': (-180.0, 360.0), 'sweep_deg': (5.0, 200.0)}
LIMITS = {'min_wiped': (0.0, 1.0)}
ASSUMPTIONS = ('Wipe pattern: the classic ring-sector model on a plane fitted to the windscreen (a blade along '
               'its arm, turning about the pivot); typical layouts are sized from the windscreen, not a vehicle\'s '
               'drawing. Dirt on the glass outside the sweep: core.exposure\'s empirical dirt score, not a measurement.')


@dataclass
class Arm:
    name: str = 'Wiper'
    pivot_w: float = 0.0          # m across the glass from the middle of its bottom edge, + to the vehicle's left
    pivot_h: float = PIVOT_H_M    # m up the glass from its bottom edge
    arm_m: float = 0.5            # pivot to the blade's centre
    blade_m: float = 0.5
    park_deg: float = 180.0       # the arm at rest, degrees from +w (left) towards +h (up)
    sweep_deg: float = 100.0      # how far it turns from rest, towards the top

    @property
    def r_in(self) -> float:
        return max(self.arm_m - self.blade_m / 2, 0.0)

    @property
    def r_out(self) -> float:
        return self.arm_m + self.blade_m / 2

    @property
    def interval(self) -> tuple[float, float]:
        """(start, end) of the swept angles, degrees, end > start (it turns from rest towards +h)."""
        p = self.park_deg % 360.0
        up = p > 90.0 and p < 270.0                  # parked pointing to +w's opposite side: turns clockwise
        return (p - self.sweep_deg, p) if up else (p, p + self.sweep_deg)

    def contains(self, w, h) -> np.ndarray:
        dw, dh = np.asarray(w, float) - self.pivot_w, np.asarray(h, float) - self.pivot_h
        r = np.hypot(dw, dh)
        a0, a1 = self.interval
        ang = (np.degrees(np.arctan2(dh, dw)) - a0) % 360.0
        return (r >= self.r_in - 1e-9) & (r <= self.r_out + 1e-9) & (ang <= a1 - a0 + 1e-9)

    def outline(self, n=48) -> np.ndarray:
        """The swept ring sector's outline (closed, (N, 2)) in glass coordinates."""
        a0, a1 = self.interval
        t = np.radians(np.linspace(a0, a1, n))
        outer = np.c_[self.pivot_w + self.r_out * np.cos(t), self.pivot_h + self.r_out * np.sin(t)]
        inner = np.c_[self.pivot_w + self.r_in * np.cos(t[::-1]), self.pivot_h + self.r_in * np.sin(t[::-1])]
        return np.vstack([outer, inner, outer[:1]])

    def blade_at(self, deg) -> np.ndarray:
        """The blade's two ends at arm angle deg, ((2, 2) glass coordinates)."""
        a = math.radians(deg)
        u = np.array([math.cos(a), math.sin(a)])
        p = np.array([self.pivot_w, self.pivot_h])
        return np.vstack([p + self.r_in * u, p + self.r_out * u])

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d) -> 'Arm | None':
        if not isinstance(d, dict):
            return None
        a = Arm(str(d.get('name') or 'Wiper')[:40])
        for k, (lo, hi) in ARM_LIMITS.items():
            try:
                v = float(d.get(k, getattr(a, k)))
            except (TypeError, ValueError):
                continue
            if math.isfinite(v):
                setattr(a, k, min(max(v, lo), hi))
        return a


@dataclass
class WiperSettings:
    enabled: bool = True
    layout: str = 'tandem'          # tandem | opposed | single | custom (the arms below)
    drive: str = 'lhd'              # lhd | rhd: the driver's side (left-hand drive: the driver sits on the left, +y)
    arms: list = field(default_factory=list)       # custom arms ([Arm.to_dict()]), used when layout is 'custom'
    min_wiped: float = 0.95         # a camera with less of its view on the glass wiped: a warning

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d: dict | None) -> 'WiperSettings':
        s = WiperSettings()
        d = d if isinstance(d, dict) else {}
        if d.get('layout') in LAYOUTS:
            s.layout = d['layout']
        if d.get('drive') in DRIVES:
            s.drive = d['drive']
        en = d.get('enabled', True)
        s.enabled = bool(en) if isinstance(en, (bool, int)) else str(en).strip().lower() in ('1', 'true', 'yes', 'on')
        arms = [Arm.from_dict(a) for a in (d.get('arms') or [])[:6]] if isinstance(d.get('arms'), list) else []
        s.arms = [a.to_dict() for a in arms if a is not None]
        if s.layout == 'custom' and not s.arms:
            s.layout = 'tandem'
        for k, (lo, hi) in LIMITS.items():
            try:
                v = float(d.get(k, getattr(s, k)))
            except (TypeError, ValueError):
                continue
            if math.isfinite(v):
                setattr(s, k, min(max(v, lo), hi))
        return s


# =========================================================================== the windscreen
@dataclass
class Windscreen:
    faces: np.ndarray            # face indices into the mesh
    source: str                  # 'glass part' | 'body region (automatic)'
    origin: np.ndarray           # the middle of the bottom edge, vehicle frame
    e_w: np.ndarray              # across, towards the vehicle's left
    e_h: np.ndarray              # up the glass
    normal: np.ndarray           # outward
    width_m: float               # across the bottom edge
    height_m: float              # up the glass
    area_m2: float               # of the faces (the curved glass)
    incline_deg: float           # from horizontal
    grid: tuple                  # (w0, h0, cell, nw, nh) of mask
    mask: np.ndarray             # (nh, nw) bool: the glass, in plane coordinates
    outline: list                # [(N, 2)] polygons of the glass in plane coordinates

    def to_plane(self, P) -> np.ndarray:
        d = np.atleast_2d(np.asarray(P, float)) - self.origin
        return np.c_[d @ self.e_w, d @ self.e_h]

    def to_world(self, Q) -> np.ndarray:
        Q = np.atleast_2d(np.asarray(Q, float))
        return self.origin + Q[:, :1] * self.e_w + Q[:, 1:2] * self.e_h


def windscreen(mesh, fm) -> Windscreen | None:
    """The windscreen's faces and their plane (module docstring); None when the model has none."""
    import cv2
    from .materials import CLASS_IDS
    k = CLASS_IDS.index('windscreen')
    sel = np.flatnonzero((fm.cls == k) & (fm.normals[:, 0] > 0.3) & (np.abs(fm.normals[:, 1]) < 0.6))
    if not len(sel):
        return None
    a = mesh.areas[sel]
    if a.sum() < 0.05:
        return None
    c = mesh.centroids[sel]
    cbar = (c * a[:, None]).sum(axis=0) / a.sum()
    n = (fm.normals[sel] * a[:, None]).sum(axis=0)          # the area-weighted mean normal: the plane's
    if np.linalg.norm(n) < 1e-9:
        return None
    n = n / np.linalg.norm(n)
    y = np.array([0.0, 1.0, 0.0])
    ew = y - (y @ n) * n
    if np.linalg.norm(ew) < 1e-6:
        return None
    ew /= np.linalg.norm(ew)
    eh = np.cross(n, ew)
    if eh[2] < 0:
        eh = -eh
    # raster the faces in the plane about their centroid, then put the origin at the middle of the bottom edge
    T2 = np.stack([(mesh.tri[sel] - cbar) @ ew, (mesh.tri[sel] - cbar) @ eh], axis=-1)        # (F, 3, 2)
    w0, h0 = float(T2[..., 0].min()) - CELL_M, float(T2[..., 1].min()) - CELL_M
    nw = int(math.ceil((float(T2[..., 0].max()) + CELL_M - w0) / CELL_M)) + 1
    nh = int(math.ceil((float(T2[..., 1].max()) + CELL_M - h0) / CELL_M)) + 1
    img = np.zeros((nh, nw), np.uint8)
    sh = 4
    pts = np.round((T2 - [w0, h0]) / CELL_M * (1 << sh)).astype(np.int32)
    cv2.fillPoly(img, list(pts), 1, cv2.LINE_8, sh)
    mask = img.astype(bool)
    rows = np.flatnonzero(mask.any(axis=1))
    if not len(rows):
        return None
    # the middle across from the lower quarter of the glass (one face's corner at the very bottom need not be it);
    # the width across its bottom 10 cm
    quarter = mask[rows[0]:rows[0] + max(1, int(round(0.25 * (rows[-1] - rows[0] + 1))))].any(axis=0)
    qc = np.flatnonzero(quarter)
    wc = w0 + (0.5 * (qc.min() + qc.max()) + 0.5) * CELL_M
    cols = np.flatnonzero(mask[rows[0]:rows[0] + max(1, int(round(0.10 / CELL_M)))].any(axis=0))
    hb = h0 + rows[0] * CELL_M
    origin = cbar + wc * ew + hb * eh
    w0, h0 = w0 - wc, h0 - hb
    cs, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    off = np.array([w0 + CELL_M / 2, h0 + CELL_M / 2])
    outline = [np.vstack([c_.reshape(-1, 2) * CELL_M + off, c_.reshape(-1, 2)[:1] * CELL_M + off])
               for c_ in cs if len(c_) >= 3]
    width = float((cols.max() - cols.min() + 1) * CELL_M)
    height = float((rows.max() - rows.min() + 1) * CELL_M)
    why = {p['index']: p['why'] for p in fm.parts}
    src = 'glass part' if any(why.get(int(i), 'body') != 'body' for i in np.unique(mesh.face_part[sel])) else \
        'body region between the side windows and the bonnet (automatic)'
    return Windscreen(sel, src, origin, ew, eh, n, width, height, float(a.sum()),
                      float(math.degrees(math.acos(min(abs(float(n[2])), 1.0)))),
                      (w0, h0, CELL_M, nw, nh), mask, outline)


def default_arms(ws: Windscreen, layout: str, drive: str) -> list[Arm]:
    """Typical arms for a layout, sized from the windscreen (module docstring)."""
    Wb, S = max(ws.width_m, 0.3), max(ws.height_m, 0.2)
    sgn = 1.0 if drive == 'lhd' else -1.0                # the driver's side along +w
    park = 180.0 if sgn > 0 else 0.0                     # pointing to the passenger's side
    if layout == 'opposed':
        r_out, r_in = min(S, 0.55 * Wb), 0.25 * S
        return [Arm('Left', 0.45 * Wb, PIVOT_H_M, (r_out + r_in) / 2, r_out - r_in, 180.0, 90.0),
                Arm('Right', -0.45 * Wb, PIVOT_H_M, (r_out + r_in) / 2, r_out - r_in, 0.0, 90.0)]
    if layout == 'single':
        r_out, r_in = 0.95 * S, 0.22 * S
        return [Arm('Single arm', 0.0, PIVOT_H_M, (r_out + r_in) / 2, r_out - r_in, park, 120.0)]
    d_out, d_in = min(math.hypot(0.22 * Wb, 0.92 * S - PIVOT_H_M), 1.15 * S), 0.30 * S     # reaches the top middle
    p_out, p_in = min(0.85 * S, 0.40 * Wb), 0.28 * S
    p_in = min(p_in, 0.8 * p_out)
    return [Arm('Driver', sgn * 0.22 * Wb, PIVOT_H_M, (d_out + d_in) / 2, d_out - d_in, park, 100.0),
            Arm('Passenger', -sgn * 0.12 * Wb, PIVOT_H_M, (p_out + p_in) / 2, p_out - p_in, park, 95.0)]


def arms_of(ws: Windscreen | None, st: WiperSettings) -> tuple[list, str]:
    """(arms, 'typical' | 'yours')."""
    if st.layout == 'custom' and st.arms:
        return [Arm.from_dict(a) for a in st.arms], 'yours'
    if ws is None:
        return [], 'typical'
    return default_arms(ws, st.layout, st.drive), 'typical'


# =========================================================================== the study
@dataclass
class WiperResult:
    settings: WiperSettings
    windscreen: Windscreen | None
    arms: list
    arms_source: str
    wiped: np.ndarray | None     # (nh, nw) bool on the windscreen's grid: glass some blade clears
    wiped_share: float           # of the glass's (plane) area
    cameras: list                # per camera behind the windscreen: dict
    findings: list


def wiped_mask(ws: Windscreen, arms) -> np.ndarray:
    w0, h0, cell, nw, nh = ws.grid
    W_, H_ = np.meshgrid(w0 + (np.arange(nw) + 0.5) * cell, h0 + (np.arange(nh) + 0.5) * cell)
    m = np.zeros((nh, nw), bool)
    for a in arms:
        m |= a.contains(W_, H_)
    return m & ws.mask


def in_sweep(arms, Q) -> np.ndarray:
    Q = np.atleast_2d(np.asarray(Q, float))
    m = np.zeros(len(Q), bool)
    for a in arms:
        m |= a.contains(Q[:, 0], Q[:, 1])
    return m


def footprint(project, cam, mesh, ws: Windscreen):
    """(plane points (N, 2), vehicle points (N, 3), face normals (N, 3)) where the camera's image rays cross the
    windscreen."""
    from .lens import resolve
    from .materials import _hits
    from .mathutil import BODY_TO_OPTICAL, rot_zyx
    intr = resolve(project.module_for(cam), project.lens_for(cam))
    gw, gh = FOOT_GRID
    U, V = np.meshgrid((np.arange(gw) + 0.5) / gw * intr.width, (np.arange(gh) + 0.5) / gh * intr.height)
    d_opt, th = intr.unproject(U.ravel(), V.ravel())
    ok = np.isfinite(th) & (th <= intr.max_theta) & np.isfinite(d_opt).all(axis=1)
    R = rot_zyx(cam.yaw, cam.pitch, cam.roll)
    D = d_opt[ok] @ (BODY_TO_OPTICAL @ R.T)
    D /= np.maximum(np.linalg.norm(D, axis=1, keepdims=True), 1e-12)
    pos = cam.pos().astype(float)
    T = _hits(pos, D, mesh.tri[ws.faces], FOOT_NEAR_M)
    j = np.argmin(T, axis=1) if T.shape[1] else np.zeros(len(D), int)
    t = T[np.arange(len(D)), j] if T.shape[1] else np.full(len(D), np.inf)
    hit = np.isfinite(t)
    P = pos + D[hit] * t[hit, None]
    return ws.to_plane(P), P, mesh.normals[ws.faces[j[hit]]]


def _where(dq) -> str:
    w, h = dq
    hz = 'left' if w > 0.02 else 'right' if w < -0.02 else ''
    vt = 'upper' if h > 0.02 else 'lower' if h < -0.02 else ''
    return ' '.join(x for x in (vt, hz) if x) or 'middle'


def analyse(project, mesh, fm, st: WiperSettings, glass_cams) -> WiperResult | None:
    """The wipe pattern on the windscreen and, per camera behind it (glass_cams: core.materials.glass_report dicts
    with class 'windscreen'), the share of its view the wipers clear and the dirt on the rest. None when wipers are
    switched off."""
    from . import exposure as E
    if not st.enabled:
        return None
    ws = windscreen(mesh, fm)
    arms, src = arms_of(ws, st)
    if ws is None:
        return WiperResult(st, None, arms, src, None, 0.0, [], [])
    wm = wiped_mask(ws, arms)
    share = float(wm.sum() / max(ws.mask.sum(), 1))
    v = project.variant
    wheels, _ = E.wheel_centres(project, project.vehicle.mesh)
    body = E._body_info(project.vehicle.mesh)
    cams = []
    findings = []
    for g in glass_cams:
        if g.get('class') != 'windscreen':
            continue
        cam = v.cam(g['uid'])
        if cam is None:
            continue
        Q, P, N = footprint(project, cam, mesh, ws)
        if not len(Q):
            continue
        inside = in_sweep(arms, Q)
        sw = float(inside.mean())
        rest = ~inside
        dirt = None
        where = ''
        if rest.any():
            f = E.dirt_factors(P[rest], [N[rest]], N[rest], wheels, body)
            sc = float(np.mean(E.dirt_score(f)))
            dirt = {'score': sc, 'class': E.dirt_class(sc)}
            where = _where(Q[rest].mean(axis=0) - Q.mean(axis=0))
        rec = {'uid': cam.uid, 'tag': cam.tag, 'slot': cam.slot, 'points': Q, 'wiped': inside, 'share_wiped': sw,
               'where_unwiped': where, 'dirt_unwiped': dirt,
               'centre': [float(x) for x in Q.mean(axis=0)]}
        cams.append(rec)
        if sw < st.min_wiped - 1e-9:
            sev = 'serious' if sw < 0.5 else 'warning'
            near = min(arms, key=lambda a: abs(np.hypot(*(Q[rest].mean(axis=0) - [a.pivot_w, a.pivot_h])) - a.r_out)) \
                if arms else None
            fix = f' or lengthen the {near.name.lower()} blade' if near is not None else ''
            dtxt = f' (dirt score {dirt["score"]:.2f}, {dirt["class"]}, exposure model)' if dirt else ''
            findings.append({'severity': sev, 'kind': 'wiper', 'tags': [cam.tag], 'uids': [cam.uid],
                             'message': f'Camera {cam.tag} looks through glass the wipers do not clear: '
                                        f'{(1 - sw) * 100:.0f} % of its view on the windscreen lies outside the '
                                        f'sweep ({where}); rain and dirt stay there{dtxt}. Move the camera into the '
                                        f'wiped area{fix}.'})
        else:
            findings.append({'severity': 'good', 'kind': 'wiper', 'tags': [cam.tag], 'uids': [cam.uid],
                             'message': f'Camera {cam.tag}: {sw * 100:.0f} % of its view on the windscreen is wiped.'})
    findings.append({'severity': 'good', 'kind': 'wiper_area', 'tags': [], 'uids': [],
                     'message': f'The wipers clear {share * 100:.0f} % of the windscreen ({ws.width_m:.2f} m wide, '
                                f'{ws.height_m:.2f} m high, {ws.incline_deg:.0f}° from horizontal; '
                                f'{"typical" if src == "typical" else "your"} '
                                f'{LAYOUT_LABEL.get(st.layout, "").lower()} '
                                f'arms).'})
    return WiperResult(st, ws, arms, src, wm, share, cams, findings)


def issues(project, vg=None) -> list:
    """Cameras behind the windscreen looking through glass the wipers do not clear (from core.materials' cached
    study; core.materials.issues includes these)."""
    from . import materials as M
    return [i for i in M.issues(project, vg) if i.message.startswith('Camera ') and 'wipers do not clear' in i.message]


# =========================================================================== drawing and output
def sweep_lines(res: WiperResult, mesh, lift_m=0.003) -> list:
    """For the 3D view: [(kind, index, (N, 3))] with kind 'sweep' (each arm's swept outline) or 'park' (its blade at
    rest), on the windscreen's surface (the plane points carried onto the faces along its normal), lifted lift_m off
    it."""
    from .materials import _hits
    ws = res.windscreen if res is not None else None
    if ws is None:
        return []
    tri = mesh.tri[ws.faces]
    out = []

    def onto(Q):
        P = ws.to_world(Q)
        t = np.full(len(P), np.inf)
        for i, o in enumerate(P + ws.normal * 0.25):          # down the normal onto the curved glass
            h = _hits(o, -ws.normal[None], tri, 0.5)[0]
            t[i] = h.min() if h.size else np.inf
        P = np.where(np.isfinite(t)[:, None], P + ws.normal * (0.25 - np.where(np.isfinite(t), t, 0.25))[:, None], P)
        return P + ws.normal * lift_m
    for i, a in enumerate(res.arms):
        out.append(('sweep', i, onto(a.outline(40))))
        out.append(('park', i, onto(a.blade_at(a.park_deg % 360.0))))
    return out


def summary(res: WiperResult) -> dict:
    ws = res.windscreen
    return {'windscreen': None if ws is None else {'width_m': round(ws.width_m, 3), 'height_m': round(ws.height_m, 3),
                                                     'incline_deg': round(ws.incline_deg, 1), 'source': ws.source},
            'layout': res.settings.layout, 'drive': res.settings.drive, 'arms': res.arms_source,
            'wiped_share': round(res.wiped_share, 4),
            'cameras': [{'tag': c['tag'], 'wiped_share': round(c['share_wiped'], 4),
                         'unwiped_dirt': c['dirt_unwiped']['class'] if c['dirt_unwiped'] else None}
                        for c in res.cameras]}


def to_doc(res: WiperResult) -> dict:
    ws = res.windscreen
    r = lambda x, nd=4: None if x is None else round(float(x), nd)          # noqa: E731
    return {'settings': res.settings.to_dict(), 'assumptions': ASSUMPTIONS,
            'glass_coordinates': 'w across (positive to the vehicle\'s left), h up the glass from the middle of its '
                                 'bottom edge, metres, in the plane fitted to the windscreen',
            'windscreen': None if ws is None else {
                'source': ws.source, 'origin': [r(x, 5) for x in ws.origin], 'w_axis': [r(x, 5) for x in ws.e_w],
                'h_axis': [r(x, 5) for x in ws.e_h], 'normal': [r(x, 5) for x in ws.normal],
                'width_m': r(ws.width_m), 'height_m': r(ws.height_m), 'area_m2': r(ws.area_m2),
                'incline_deg': r(ws.incline_deg, 2)},
            'arms': [dict(a.to_dict(), r_in_m=r(a.r_in), r_out_m=r(a.r_out),
                          swept_deg=[r(x, 2) for x in a.interval]) for a in res.arms],
            'arms_source': res.arms_source, 'wiped_share': r(res.wiped_share),
            'cameras': [{'tag': c['tag'], 'wiped_share': r(c['share_wiped']),
                         'unwiped_where': c['where_unwiped'] or None,
                         'unwiped_dirt': ({'score': r(c['dirt_unwiped']['score'], 3),
                                           'class': c['dirt_unwiped']['class']} if c['dirt_unwiped'] else None),
                         'footprint_centre_wh_m': [r(x) for x in c['centre']]} for c in res.cameras]}


def draw_png(res: WiperResult, img, rect, text):
    """The windscreen seen from inside, looking forward (the vehicle's left on the left), its wiped area, the arms
    at rest and the cameras' footprints (red: not wiped), into img (BGR) within rect (x, y, w, h)."""
    import cv2
    ws = res.windscreen
    x0, y0, W, H = rect
    w0, h0, cell, nw, nh = ws.grid
    text(f'Windscreen and wipers  {res.wiped_share * 100:.0f} % wiped  ({ws.source})', x0, y0 + 14, (236, 235, 230),
         0.45)
    s = min((W - 20) / max(nw * cell, 1e-6), (H - 40) / max(nh * cell, 1e-6))
    ox, oy = x0 + 10, y0 + 30

    def px(Q):
        Q = np.atleast_2d(Q)
        return np.c_[ox + (w0 + nw * cell - Q[:, 0]) * s, oy + (h0 + nh * cell - Q[:, 1]) * s].astype(np.int32)
    glass = np.zeros((nh, nw, 3), np.uint8)
    glass[ws.mask] = (70, 62, 55)
    glass[res.wiped] = (128, 108, 82)
    tile = cv2.resize(glass[::-1, ::-1], (max(int(nw * cell * s), 1), max(int(nh * cell * s), 1)),
                      interpolation=cv2.INTER_NEAREST)
    sub = img[oy:oy + tile.shape[0], ox:ox + tile.shape[1]]
    tile = tile[:sub.shape[0], :sub.shape[1]]
    sub[:] = np.where(tile.any(axis=2, keepdims=True), tile, sub)
    for a in res.arms:
        cv2.polylines(img, [px(a.outline(40))], True, (200, 199, 192), 1, cv2.LINE_AA)
        b = px(a.blade_at(a.park_deg % 360.0))
        cv2.line(img, tuple(b[0]), tuple(b[1]), (236, 235, 230), 2, cv2.LINE_AA)
        cv2.circle(img, tuple(px([[a.pivot_w, a.pivot_h]])[0]), 4, (236, 235, 230), -1, cv2.LINE_AA)
    palette = [(229, 135, 57), (38, 89, 217), (112, 158, 25), (0, 133, 201), (129, 81, 213), (0, 131, 0),
               (233, 133, 144), (103, 103, 230)]
    for c in res.cameras:
        col = palette[c['slot'] % len(palette)]
        P = px(c['points'])
        for (u, v), ok in zip(P, c['wiped']):
            cv2.circle(img, (int(u), int(v)), 2, col if ok else (47, 49, 176), -1, cv2.LINE_AA)
        cu, cv_ = px([c['centre']])[0]
        text(f'{c["tag"]} {c["share_wiped"] * 100:.0f} %', cu + 6, cv_ - 6, col, 0.4)
