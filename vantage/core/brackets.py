"""Mount brackets: a 3D-printable bracket per camera that holds the camera module at exactly its planned position and
angles on the surface it mounts on, so the build matches the plan without hand-made shims.

Geometry (millimetres, the bracket's own frame: origin at the base's contact point, z along the surface normal, so
the STL prints with z up):
- The mount surface is the nearest surface to the camera's optical centre: the body or a box rig part (a flat base)
  or a round tube (a bar or a mast of the rig: a split tube clamp of two halves bolted together).
- The housing is the module's size_mm read as width (across the image) × height × length along the optical axis,
  lens included, with the optical centre on its axis at the lens front (setback_mm moves it behind the front: a
  lens's entrance pupil sits a few millimetres inside the barrel; check your module's drawing).
- The camera plate lies flat against one housing face (back: square to the optical axis; bottom or top: square to
  the camera's up axis), carries the module's hole pattern (clearance holes with the screw heads' access bored
  through the support behind), and a solid support joins it to the base: the convex hull of the plate and its
  footprint on the base, with a gusset behind a steep plate. The plate's place follows from the planned pose, so
  the optical centre lands on the planned point and the optical axis on the planned direction.
- A flat base is a pad with two or four countersunk screw holes (each moved out until a screwdriver reaches it
  past the plate and camera) or a VHB tape pad sized by the 3M rule of thumb. It is fitted to the body under it:
  rays cast onto the body give the surface heights under the pad, the base plane is their best-fit plane lifted to
  rest on the highest point, and the gaps left (a curved panel) are reported.

Solids are signed distance functions (core.meshing) meshed by marching tetrahedra into a closed, consistently wound
mesh, checked edge by edge.

Checks per bracket: the share of the camera's image the bracket blocks (the camera's pixel rays traced against the
bracket's solid), the lever arm from the base to the housing, the thinnest wall and hole ligament, the gap under
the base, the housing or bracket running into the body, screw access, the connector's way out on a back mount, the
optical centre and axis measured back from the mesh (contact face and hole centres) against the plan, a beam
estimate of stiffness, and the print orientation with the least overhang. A camera whose housing would go into the
surface at the planned standoff gets the standoff a bracket needs instead (Move out on the page moves it); a camera
far from any surface gets no bracket (a custom mount or a rig part is needed).
"""
from __future__ import annotations

import json
import math
import os
import re
import time
from dataclasses import asdict, dataclass, field

import numpy as np

from . import meshing as M
from .mathutil import BODY_TO_OPTICAL, rot_zyx

SCHEMA = 'vantage.brackets/1'
FRAME_TEXT = ('bracket frame: millimetres, origin at the base\'s contact point on the mount surface, z along the '
              'surface normal (the STL prints z up); T_vehicle_bracket maps the bracket frame (in metres: divide the '
              'STL coordinates by 1000) into the vehicle frame (ISO 8855, rear axle, metres)')
ASSUMPTIONS = ('The housing is a box of the module\'s size (width × height × length along the optical axis, lens '
               'included) with the optical centre on its axis at the lens front, moved back by the setback; check '
               'your module\'s drawing. Hole patterns are presets or your figures, not read from a product. '
               'Material figures are typical for printed parts, the stiffness is a beam estimate and the VHB area '
               'follows 3M\'s rule of thumb (4 in² per lb of static load, times the design load): none is a test. '
               'The body under the base is the vehicle model; a real panel differs by its build tolerance.')
FACES = ('auto', 'back', 'bottom', 'top')
FACE_LABEL = {'auto': 'Auto', 'back': 'Back', 'bottom': 'Bottom', 'top': 'Top'}
FIXINGS = ('screws2', 'screws4', 'vhb')
FIXING_LABEL = {'screws2': '2 screws', 'screws4': '4 screws', 'vhb': 'VHB tape'}
# (nominal diameter, socket-head diameter, countersunk-head diameter) mm: ISO 4762 / ISO 10642 maxima, 1/4-20 UNC
NUTS = {'M3': (5.5, 2.4), 'M4': (7.0, 3.2), 'M5': (8.0, 4.7)}     # hex nut across flats, height (ISO 4032)
SCREWS = {'M2': (2.0, 3.8, 4.4), 'M2.5': (2.5, 4.5, 5.5), 'M3': (3.0, 5.5, 6.72), 'M4': (4.0, 7.0, 8.96),
          'M5': (5.0, 8.5, 11.2), '1/4-20': (6.35, 9.53, 12.7)}
BASE_SCREWS = ('M3', 'M4', 'M5')
PATTERNS = {'2xM2-16': {'label': '2 × M2 · 16 mm', 'screw': 'M2', 'count': 2, 'a': 16.0, 'b': 0.0},
            '4xM2-20x20': {'label': '4 × M2 · 20 × 20 mm', 'screw': 'M2', 'count': 4, 'a': 20.0, 'b': 20.0},
            '2xM3-20': {'label': '2 × M3 · 20 mm', 'screw': 'M3', 'count': 2, 'a': 20.0, 'b': 0.0},
            'tripod': {'label': '1/4-20 tripod', 'screw': '1/4-20', 'count': 1, 'a': 0.0, 'b': 0.0},
            'custom': {'label': 'Custom…', 'screw': '', 'count': 2, 'a': 16.0, 'b': 0.0}}
# typical printed-part figures (XY direction): modulus MPa, density g/cm³, heat deflection °C
MATERIALS = {
    'PETG': {'E': 1800.0, 'rho': 1.27, 'hdt': 70,
             'note': 'PETG prints easily but softens near 70 °C, which a dark body panel in the sun reaches: use it '
                     'for trials and shaded mounts.'},
    'ASA': {'E': 1800.0, 'rho': 1.07, 'hdt': 95,
            'note': 'ASA keeps its shape to about 95 °C and stands UV: the usual choice outside the car. Print it in '
                    'an enclosure.'},
    'PA-CF': {'E': 4500.0, 'rho': 1.20, 'hdt': 150,
              'note': 'Carbon-filled nylon is the stiffest and stands the most heat; dry the filament before '
                      'printing and use a hardened nozzle.'},
}
FACE_MIN = -0.02               # a plate may turn up to about 91° from facing the surface (an L-bracket at 90°)
DESIGN_G = 5.0                 # road shocks the VHB area and the deflection are sized for, in g
VHB_MM2_PER_G = 4 * 645.16 / 453.59          # 4 in² of tape per lb of static load
STATUS_ORDER = {'standoff': 0, 'floating': 1, 'inside': 2, 'face': 3, 'warning': 4, 'ok': 5, 'skipped': 6}
NOT_BUILT = ('standoff', 'floating', 'inside', 'face', 'skipped')
LIMITS = {'custom_d_mm': (1.0, 10.0), 'custom_a_mm': (0.0, 100.0), 'custom_b_mm': (0.0, 100.0),
          'custom_count': (1, 4), 'clearance_mm': (0.0, 1.5), 'setback_mm': (0.0, 40.0),
          'connector_mm': (0.0, 30.0), 'wall_mm': (2.0, 10.0), 'margin_mm': (1.0, 20.0), 'fillet_mm': (0.0, 5.0),
          'tape_mm': (0.0, 5.0), 'clamp_clear_mm': (0.0, 2.0), 'res_mm': (0.2, 2.0), 'overhang_mm': (10.0, 500.0),
          'reach_mm': (20.0, 500.0)}


@dataclass
class BracketSettings:
    face: str = 'auto'               # housing face the plate holds: auto | back | bottom | top
    pattern: str = '2xM2-16'         # PATTERNS key
    custom_count: int = 2            # custom pattern: 1, 2 or 4 holes
    custom_d_mm: float = 2.0         # screw diameter
    custom_a_mm: float = 16.0        # spacing along the face's first axis (width on the back, length below/above)
    custom_b_mm: float = 16.0        # spacing along the second axis (4 holes)
    clearance_mm: float = 0.4        # hole diameter over the screw's nominal
    setback_mm: float = 0.0          # optical centre behind the lens front
    connector_mm: float = 12.0       # back face: hole for the connector and cable (0 = none)
    wall_mm: float = 3.0
    margin_mm: float = 3.0           # plate beyond the housing face
    fillet_mm: float = 1.5
    material: str = 'ASA'
    fixing: str = 'screws2'          # screws2 | screws4 | vhb
    base_screw: str = 'M4'
    countersink: bool = True
    tape_mm: float = 1.1             # VHB foam thickness: the gap it fills
    clamp_clear_mm: float = 0.2      # tube clamp bore over the tube's radius
    clamp_bolt: str = 'M4'
    res_mm: float = 0.35             # mesher grid
    overhang_mm: float = 60.0        # lever arm warning
    reach_mm: float = 120.0          # farther from any surface: no printed bracket

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d: dict | None) -> 'BracketSettings':
        s = BracketSettings()
        d = d if isinstance(d, dict) else {}
        for k, opts in (('face', FACES), ('pattern', tuple(PATTERNS)), ('material', tuple(MATERIALS)),
                        ('fixing', FIXINGS), ('base_screw', BASE_SCREWS), ('clamp_bolt', BASE_SCREWS)):
            if d.get(k) in opts:
                setattr(s, k, d[k])
        v = d.get('countersink')
        if isinstance(v, (bool, int)):
            s.countersink = bool(v)
        elif isinstance(v, str):
            s.countersink = v.strip().lower() in ('1', 'true', 'yes', 'on')
        for k, (lo, hi) in LIMITS.items():
            if k in d:
                try:
                    v = float(d[k])
                except (TypeError, ValueError):
                    continue
                if math.isfinite(v):
                    setattr(s, k, min(max(v, lo), hi))
        c = int(round(s.custom_count))
        s.custom_count = 4 if c >= 3 else max(c, 1)
        return s

    def holes(self):
        """(screw label, nominal d, head d, count, spacing a, spacing b) of the camera's hole pattern."""
        if self.pattern == 'custom':
            d = self.custom_d_mm
            return (f'Ø{d:g}', d, 1.85 * d + 0.2, self.custom_count, self.custom_a_mm, self.custom_b_mm)
        p = PATTERNS[self.pattern]
        d, head, _ = SCREWS[p['screw']]
        return p['screw'], d, head, p['count'], p['a'], p['b']

    def pattern_label(self):
        if self.pattern == 'custom':
            _, d, _, n, a, b = self.holes()
            return f'{n} × Ø{d:g}' + ('' if n == 1 else f' · {a:g}' + (f' × {b:g}' if n == 4 else '') + ' mm')
        return PATTERNS[self.pattern]['label']


@dataclass
class Bracket:
    uid: str
    tag: str
    name: str = ''
    slot: int = 0
    status: str = 'skipped'          # ok | warning | standoff | floating | inside | face | skipped
    message: str = ''
    kind: str = ''                   # flat | clamp
    face: str = ''                   # back | bottom | top
    surface: str = ''
    gap_mm: float = float('nan')     # optical centre to the mount surface, measured on the model
    plan_standoff_mm: float = float('nan')
    required_mm: float = float('nan')     # the standoff a printed bracket needs (status standoff)
    normal: tuple | None = None      # mount surface normal at the contact point (vehicle), for moving the camera out
    contact: tuple | None = None     # contact point (vehicle, m)
    R: np.ndarray = field(default_factory=lambda: np.eye(3))      # bracket axes in the vehicle frame (columns)
    origin: np.ndarray = field(default_factory=lambda: np.zeros(3))   # bracket origin, vehicle frame, m
    V: np.ndarray | None = None      # vertices, mm, bracket frame
    F: np.ndarray | None = None
    params: dict = field(default_factory=dict)
    checks: list = field(default_factory=list)      # {'code', 'severity', 'text', 'value'}
    notes: list = field(default_factory=list)
    mesh: dict = field(default_factory=dict)
    housing: tuple | None = None     # (centre, axes, half) mm, bracket frame
    oc: np.ndarray | None = None     # optical centre, bracket frame, mm (the plan)
    axis: np.ndarray | None = None   # optical axis, bracket frame
    fov_mask: np.ndarray | None = None    # per sampled pixel: 0 outside the lens circle, 1 clear, 2 the bracket
    print_up: np.ndarray | None = None
    display: tuple | None = None     # (V, F) thinned for drawing, and a coarser one for dragging (made by the worker)
    display_fast: tuple | None = None
    seconds: float = 0.0
    solid: object = None             # the signed distance function (not exported)

    @property
    def generated(self) -> bool:
        return self.V is not None and self.F is not None and len(self.F) > 0

    def worst(self) -> str:
        sev = [c['severity'] for c in self.checks if c['severity'] in ('critical', 'serious', 'warning')]
        order = {'critical': 0, 'serious': 1, 'warning': 2}
        return min(sev, key=order.get) if sev else 'good'

    def check(self, code):
        return next((c for c in self.checks if c['code'] == code), None)


@dataclass
class BracketResult:
    settings: BracketSettings
    brackets: list
    variant: str = ''
    variant_uid: str = ''
    seconds: float = 0.0

    def by_uid(self, uid):
        return next((b for b in self.brackets if b.uid == uid), None)

    def by_tag(self, tag):
        return next((b for b in self.brackets if b.tag == tag), None)


# =========================================================================== small helpers
def _unit(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def _perp(v, n):
    """v without its component along the unit n."""
    v = np.asarray(v, float)
    return v - (v @ n) * n


def _check(out, code, severity, text, value=None):
    out.append({'code': code, 'severity': severity, 'text': text,
                'value': None if value is None else (round(float(value), 3) if np.isscalar(value) else value)})


def safe_tag(tag) -> str:
    """A camera tag as a file name."""
    t = re.sub(r'[^A-Za-z0-9_.-]+', '_', str(tag)).strip('._')
    return t or 'camera'


# =========================================================================== the mount surface
def _surface_name(mesh, fi):
    zi = int(mesh.face_zone[fi]) if len(mesh.face_zone) else -1
    if 0 <= zi < len(mesh.zones):
        return mesh.zones[zi].name, mesh.zones[zi].kind
    part = mesh.parts[int(mesh.face_part[fi])] if mesh.parts else None
    kind = ''
    if part is not None and part.material == 'rig' and 'not mountable' in part.role:
        kind = 'forbidden'
    return (part.name if part else 'Body'), kind


def find_mount(project, vg, cam, reach_mm=120.0) -> dict:
    """Where the camera mounts: {'kind': flat | clamp, 'q' contact point (m), 'n' outward normal, 'gap_mm',
    'surface', 'zone_kind', 'inside', 'floating', and for a clamp 'b' tube axis, 'r_mm' tube radius, 'part'}."""
    P = cam.pos()
    mesh = vg.mesh if vg.mesh.n_tris <= 60000 else vg.occ_mesh
    q, fi, d = mesh.closest_point(P, mesh.occluder_faces())
    out = {'kind': 'flat', 'q': q, 'n': np.array([0.0, 0.0, 1.0]), 'gap_mm': d * 1000.0, 'surface': 'Body',
           'zone_kind': '', 'inside': False, 'floating': False, 'fi': fi, 'mesh': mesh}
    if fi >= 0:
        out['n'] = mesh.normals[fi].copy()
        out['surface'], out['zone_kind'] = _surface_name(mesh, fi)
        out['inside'] = bool((P - q) @ out['n'] < -0.002)
    best = d if fi >= 0 else math.inf
    for part in getattr(project, 'rig', ()):
        if not part.visible or not part.occludes or part.kind not in ('bar', 'cylinder') or not part.radius > 0.002:
            continue
        c = np.array([part.x, part.y, part.z])
        yaw = math.radians(part.yaw)
        b = np.array([-math.sin(yaw), math.cos(yaw), 0.0]) if part.kind == 'bar' else np.array([0.0, 0.0, 1.0])
        hl = (part.length if part.kind == 'bar' else part.height) / 2
        t = float((P - c) @ b)
        radial = P - c - t * b
        rho = float(np.linalg.norm(radial))
        if abs(t) > hl - 0.004 or rho < 1e-6:
            continue
        dist = rho - part.radius
        if dist <= best + 0.003:                          # the tube's facets sit a little inside the true circle
            n = radial / rho
            out.update(kind='clamp', q=c + t * b + part.radius * n, n=n, gap_mm=dist * 1000.0, b=b,
                       r_mm=part.radius * 1000.0, part=part.name, surface=part.name, inside=dist < -0.002,
                       zone_kind='' if part.mountable else 'forbidden')
            best = dist
    out['floating'] = bool(out['gap_mm'] > reach_mm and not out['inside'])
    return out


# =========================================================================== design
class Design:
    """One bracket's parts in its own frame (mm): built by design(), cheap (no mesh)."""
    pass


def _face(face, P, A, size, s):
    """(u outward plate normal, face centre, e1, e2, face size along e1, along e2) of a housing face."""
    a, ey, ez = A[:, 0], A[:, 1], A[:, 2]
    w, h, L = size
    if face == 'back':
        return -a, P + a * (s - L), ey, ez, w, h
    if face == 'bottom':
        return -ez, P + a * (s - L / 2) - ez * h / 2, a, ey, L, w
    return ez, P + a * (s - L / 2) + ez * h / 2, a, ey, L, w


def face_scores(R_cam, n):
    """How squarely each housing face looks at the surface (−u·n, 1 = facing it)."""
    a, ez = R_cam[:, 0], R_cam[:, 2]
    return {'back': float(a @ n), 'bottom': float(ez @ n), 'top': float(-ez @ n)}


def choose_face(st: BracketSettings, R_cam, n):
    """(face, score, best face): the face the plate holds; it must face the surface (score > FACE_MIN). Auto prefers the
    bottom or top face when one faces the surface well enough (within 75°): a back plate would trap the connector
    and cable behind the camera, pointing into the support; without a connector hole the squarest face wins."""
    sc = face_scores(R_cam, n)
    best = max(sc, key=lambda k: sc[k] + (0.03 if k == 'back' else 0.0))
    if st.connector_mm > 0:
        side = max(('bottom', 'top'), key=lambda k: sc[k])
        if sc[side] >= 0.25:
            best = side
    face = best if st.face == 'auto' else st.face
    return face, sc[face], best


def _frame(n, u, A, b=None):
    """Bracket axes (columns x, y, z) in the vehicle frame: z the surface normal, x along the plate's lean (so a side
    view along y shows the plate's angle), or along the tube for a clamp."""
    z = _unit(n)
    ut = _perp(u, z)
    if np.linalg.norm(ut) > 1e-3:
        x = -_unit(ut)
    else:
        x = _perp(A[:, 0], z)
        x = _unit(x) if np.linalg.norm(x) > 1e-3 else _unit(_perp(A[:, 1], z))
    if b is not None:
        bb = _unit(_perp(b, z))
        x = bb if bb @ x >= 0 else -bb
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=1)


def design(st: BracketSettings, P, A, size, face, mount, access=True, mass_g=60.0) -> Design:
    """The bracket for a camera with optical centre P and axes A (columns: optical axis, left, up) in the bracket frame
    (mm), housing size (width, height, length), on mount ({'kind': flat} with the surface at z = 0, or {'kind':
    clamp, 'r_mm'} with the tube along x under z = 0). access: place screws where a screwdriver reaches them.
    mass_g: the camera and bracket (the VHB pad is sized for it)."""
    D = Design()
    D.flags = {}
    D.mass_g = mass_g
    D.kind, D.face = mount['kind'], face
    s, t, mrg, fil = st.setback_mm, st.wall_mm, st.margin_mm, st.fillet_mm
    w, h, L = size
    D.P, D.A = np.asarray(P, float), np.asarray(A, float)
    a = D.A[:, 0]
    D.housing = (D.P + a * (s - L / 2), D.A.copy(), np.array([L / 2, w / 2, h / 2]))
    u, Fc, e1, e2, d1, d2 = _face(face, D.P, D.A, size, s)
    D.u, D.Fc, D.e1, D.e2 = u, Fc, e1, e2
    # ---- hole pattern on the face
    _, dn, head, count, pa, pb = st.holes()
    off = {1: [(0.0, 0.0)], 2: [(-pa / 2, 0.0), (pa / 2, 0.0)],
           4: [(-pa / 2, -pb / 2), (pa / 2, -pb / 2), (-pa / 2, pb / 2), (pa / 2, pb / 2)]}[count]
    off = np.array(off)
    hr = head / 2
    D.hole_d = dn + st.clearance_mm
    # ---- the plate
    need1 = np.abs(off[:, 0]).max() + hr + 1.5
    need2 = np.abs(off[:, 1]).max() + hr + 1.5
    h2 = max(d2 / 2 + mrg, need2)
    if face == 'back':
        lo1, hi1 = -max(d1 / 2 + mrg, need1), max(d1 / 2 + mrg, need1)
    else:
        lo1 = min(-(L / 2 + mrg), off[:, 0].min() - hr - 1.5)
        front = L / 2 - s + min(s, 0.0) - 1.0          # 1 mm short of the lens front (and the optical centre)
        hi1 = front
        if off[:, 0].max() + hr + 1.5 > front:
            hi1 = off[:, 0].max() + hr + 1.5
            D.flags['plate_long'] = True
    D.flags['pattern_wide'] = bool(need1 > d1 / 2 + mrg + 1e-6 and face == 'back' or need2 > d2 / 2 + mrg + 1e-6)
    c1 = (lo1 + hi1) / 2
    pax = np.stack([e1, e2, u], axis=1)
    phalf = np.array([(hi1 - lo1) / 2, h2, t / 2])
    rp = max(min(fil, t / 2 - 0.1), 0.0)
    D.plate = (Fc + e1 * c1 + u * t / 2, pax, phalf)
    plate = M.box(*D.plate, r=rp)
    corners = np.array([Fc + u * t + e1 * x1 + e2 * x2 for x1 in (lo1 + rp, hi1 - rp) for x2 in (-h2 + rp, h2 - rp)])
    inner = np.array([Fc + e1 * x1 + e2 * x2 for x1 in (lo1, hi1) for x2 in (-h2, h2)])
    D.plate_corners = np.vstack([corners, inner])
    D.plate_extent = (lo1, hi1, -h2, h2)
    D.holes_cam = [Fc + e1 * o1 + e2 * o2 for o1, o2 in off]
    D.hole_off = off
    D.head = head
    # ---- the base level
    if D.kind == 'clamp':
        r = mount['r_mm']
        rb = r + st.clamp_clear_mm
        tc = max(t, 3.0)
        z_base = (rb - r) + tc
        D.clamp = {'r': r, 'rb': rb, 'tc': tc, 'Ro': rb + tc, 'z_axis': -r,
                   'gap': max(1.0, 2.5 * st.res_mm)}
    else:
        bs = SCREWS[st.base_screw]
        csk_depth = (bs[2] - (bs[0] + st.clearance_mm)) / 2
        tb = t
        if st.fixing != 'vhb' and st.countersink:
            tb = max(t, csk_depth + 1.2)
        z_base = tb
        D.tb = tb
    D.z_base = z_base
    # ---- the support: hull of the plate's outer face and its footprint (and a gusset behind a steep plate)
    top = corners
    H = float(top[:, 2].max() - z_base)
    D.support = None
    foot = []
    if H > 0.8:
        pts = [top]
        proj = top.copy()
        proj[:, 2] = np.minimum(top[:, 2], z_base - 0.2)
        pts.append(proj)
        foot.append(proj[:, :2])
        ut = np.array([u[0], u[1], 0.0])
        lean = float(np.linalg.norm(ut))
        g = H * max(0.0, lean - 0.5)
        if g > 0.5:
            ext = proj + _unit(ut) * g
            pts.append(ext)
            foot.append(ext[:, :2])
        D.support = M.hull(np.vstack(pts))
    if not foot:
        foot.append(D.plate_corners[:, :2])
    D.footprint = np.vstack(foot)
    D.support_H = H
    # ---- obstacles a screwdriver must pass: the support, the plate and the housing
    obst = M.union(*[x for x in (D.support, plate, M.box(*D.housing)) if x is not None])
    top_z = float(max(D.plate_corners[:, 2].max(), (D.housing[0] + np.abs(D.housing[1]) @ D.housing[2])[2])) + 5

    def clear(xy, z0, rad):
        if not access:
            return True
        zs = np.arange(z0, top_z, 1.0)
        if not len(zs):
            return True
        Pl = np.column_stack([np.full(len(zs), xy[0]), np.full(len(zs), xy[1]), zs])
        return bool(obst(Pl).min() > rad)
    cuts = [M.box(*D.housing)]
    # camera screws: clearance holes and the heads' access through the support
    for p in D.holes_cam:
        cuts.append(M.rod(p - u * 1.0, p + u * (t + 0.5), D.hole_d / 2))
        cuts.append(M.rod(p + u * t, p + u * 400.0, hr + 0.5))
    D.connector = None
    if face == 'back' and st.connector_mm > 0:
        D.connector = st.connector_mm
        cuts.append(M.rod(Fc - u * 1.0, Fc + u * 400.0, st.connector_mm / 2))
        lig = min((np.hypot(*o) - D.hole_d / 2 - st.connector_mm / 2 for o in off), default=99.0)
        D.flags['connector_lig'] = lig
    fx = D.footprint
    xl, yl = fx.min(axis=0)
    xh, yh = fx.max(axis=0)
    mb = max(2.0, t)
    D.screws, D.access_blocked = [], 0
    if D.kind == 'flat':
        base_part, base_cuts = _flat_base(D, st, xl, xh, yl, yh, mb, clear)
    else:
        base_part, base_cuts = _clamp(D, st, xl, xh, yl, yh, mb, clear)
    cuts += base_cuts
    body = base_part
    if D.support is not None:
        body = M.round_union(body, D.support, fil)
    body = M.round_union(body, plate, fil)
    D.parts = {'base': base_part, 'support': D.support, 'plate': plate}
    D.cuts = cuts
    solid = M.subtract(body, *cuts)
    if D.kind == 'clamp':
        cap = M.subtract(D.cap, *D.clamp_cuts)
        solid = M.union(solid, cap)
    D.solid = solid
    D.lo, D.hi = solid.lo, solid.hi
    return D


def _flat_base(D, st, xl, xh, yl, yh, mb, clear):
    bs = SCREWS[st.base_screw]
    rcs = bs[2] / 2 + 0.2 if st.countersink else bs[1] / 2 + 0.3
    edge = max(2.5, st.wall_mm)
    tb = D.tb
    xc, yc = (xl + xh) / 2, (yl + yh) / 2
    rects = [(xl - mb, xh + mb, yl - mb, yh + mb)]
    D.screws = []
    if st.fixing in ('screws2', 'screws4'):
        e = rcs + 1.0
        layouts = []
        if st.fixing == 'screws2':
            layouts.append([((xl - e, yc), (-1, 0)), ((xh + e, yc), (1, 0))])
            layouts.append([((xc, yl - e), (0, -1)), ((xc, yh + e), (0, 1))])
        else:
            layouts.append([((xl - e, yl), (-1, 0)), ((xl - e, yh), (-1, 0)), ((xh + e, yl), (1, 0)),
                            ((xh + e, yh), (1, 0))])
            layouts.append([((xl, yl - e), (0, -1)), ((xh, yl - e), (0, -1)), ((xl, yh + e), (0, 1)),
                            ((xh, yh + e), (0, 1))])
        best = None
        for lay in layouts:
            pos, push, blocked = [], 0.0, 0
            for (x, y), dvec in lay:
                p = np.array([x, y], float)
                k = 0
                while not clear(p, tb + 0.3, rcs + 0.3) and k < 20:
                    p = p + np.array(dvec, float) * 2.0
                    k += 1
                if k >= 20:
                    blocked += 1
                push += k * 2.0
                pos.append(p)
            span = np.vstack(pos + [np.array([[xl, yl], [xh, yh]])])
            area = float(np.ptp(span[:, 0]) * np.ptp(span[:, 1]))
            key = (blocked, push, area)
            if best is None or key < best[0]:
                best = (key, pos, blocked)
        _, pos, blocked = best
        D.access_blocked = blocked
        for p in pos:
            D.screws.append(np.array([p[0], p[1], tb]))
            rects.append((p[0] - rcs - edge, p[0] + rcs + edge, p[1] - rcs - edge, p[1] + rcs + edge))
    x0 = min(r[0] for r in rects)
    x1 = max(r[1] for r in rects)
    y0 = min(r[2] for r in rects)
    y1 = max(r[3] for r in rects)
    D.tape_need = 0.0
    if st.fixing == 'vhb':
        pad = max(mb, 4.0)
        x0, x1, y0, y1 = xl - pad, xh + pad, yl - pad, yh + pad
        need = VHB_MM2_PER_G * D.mass_g * DESIGN_G
        D.tape_need = need
        A = (x1 - x0) * (y1 - y0)
        if A < need:
            lx, ly = x1 - x0, y1 - y0
            k = (-(lx + ly) + math.sqrt((lx + ly) ** 2 + 4 * (need - A))) / 4
            k = min(k, 30.0)
            x0, x1, y0, y1 = x0 - k, x1 + k, y0 - k, y1 + k
    D.base_rect = (x0, x1, y0, y1)
    D.base_area = (x1 - x0) * (y1 - y0)
    hx, hy = (x1 - x0) / 2, (y1 - y0) / 2
    base = M.round_rect([(x0 + x1) / 2, (y0 + y1) / 2, tb / 2], np.eye(3), [hx, hy, tb / 2], min(5.0, hx, hy))
    bcuts = []
    for p in D.screws:
        bcuts.append(M.rod([p[0], p[1], -1.0], [p[0], p[1], tb + 1.0], (bs[0] + st.clearance_mm) / 2))
        if st.countersink:
            bcuts.append(M.countersink([p[0], p[1], tb], [0, 0, 1], bs[2] / 2 + 0.2))
    D.base_screw = (st.base_screw, bs[0] + st.clearance_mm, rcs)
    return base, bcuts


def _clamp(D, st, xl, xh, yl, yh, mb, clear):
    """Split tube clamp: a saddle (carrying the support) over the tube's top half and a cap under it, closed by a bolt
    through the ears on each side, put in from below (the head under the cap) into a nut held in a hex pocket on the
    saddle ear, so nothing above the clamp needs a tool."""
    c = D.clamp
    rb, Ro, za, gap = c['rb'], c['Ro'], c['z_axis'], c['gap']
    bs = SCREWS[st.clamp_bolt]
    nut_s, nut_m = NUTS[st.clamp_bolt]
    bd = bs[0] + st.clearance_mm
    wall = max(st.wall_mm, 2.5)
    ew = max(bd, nut_s / 0.866 + 0.4) + 2 * wall
    et = max(st.wall_mm, 4.0)                        # cap ear
    ets = max(st.wall_mm, nut_m + 0.3 + 2.0)         # saddle ear: the nut pocket and 2 mm under it
    Wmin = max(14.0, ew)
    x0, x1 = min(xl - mb, -Wmin / 2), max(xh + mb, Wmin / 2)
    y0, y1 = min(-Ro, yl - mb), max(Ro, yh + mb)        # from the tube out to under the support
    zt = D.z_base
    bolt_x = [(x0 + x1) / 2] if x1 - x0 <= 30 else [x0 + ew / 2, x1 - ew / 2]
    yb = {-1: y0 - ew / 2, 1: y1 + ew / 2}
    I = np.eye(3)
    zs0 = za + gap / 2
    zc0 = za - gap / 2
    xm, hx = (x0 + x1) / 2, (x1 - x0) / 2
    ylo, yhi = y0 - ew, y1 + ew
    ym, hy = (ylo + yhi) / 2, (yhi - ylo) / 2
    saddle = M.union(M.box([xm, (y0 + y1) / 2, (zs0 + zt) / 2], I, [hx, (y1 - y0) / 2, (zt - zs0) / 2]),
                     M.box([xm, ym, zs0 + ets / 2], I, [hx, hy, ets / 2]))
    cap = M.intersect(M.union(M.rod([x0, 0, za], [x1, 0, za], Ro), M.box([xm, ym, zc0 - et / 2], I, [hx, hy, et / 2])),
                      M.halfspace([0, 0, zc0], [0, 0, 1]))
    cap.lo = np.array([x0, ylo, za - Ro])
    cap.hi = np.array([x1, yhi, zc0])
    bore = M.rod([x0 - 5, 0, za], [x1 + 5, 0, za], rb)
    holes, nuts = [], []
    for x in bolt_x:
        for sgn in (-1, 1):
            holes.append(M.rod([x, yb[sgn], zc0 - et - 1], [x, yb[sgn], zs0 + ets + 1], bd / 2))
            ztop = zs0 + ets
            nuts.append(M.hex_prism([x, yb[sgn], ztop - (nut_m + 0.3) / 2 + 0.25], I, nut_s + 0.3,
                                    (nut_m + 0.3) / 2 + 0.25))
    D.cap = cap
    D.clamp_cuts = [bore] + holes
    D.saddle = saddle
    D.bolts = [np.array([x, yb[sgn], zc0 - et]) for x in bolt_x for sgn in (-1, 1)]
    D.access_blocked = 0
    D.clamp.update(x0=x0, x1=x1, ylo=ylo, yhi=yhi, et=et, ets=ets, ew=ew, bolt=st.clamp_bolt, bolt_d=bd,
                   z_top=zt, nut=(nut_s, nut_m))
    D.base_rect = (x0, x1, ylo, yhi)
    D.base_area = (x1 - x0) * (yhi - ylo)
    return saddle, [bore] + holes + nuts


# =========================================================================== interference and standoff
def _housing_points(D, step=2.0):
    c, A, half = D.housing
    pts = []
    for k in range(3):
        i, j = [x for x in range(3) if x != k]
        ni, nj = max(2, int(2 * half[i] / step) + 1), max(2, int(2 * half[j] / step) + 1)
        U, W = np.meshgrid(np.linspace(-half[i], half[i], ni), np.linspace(-half[j], half[j], nj), indexing='ij')
        for sgn in (-1, 1):
            loc = np.zeros((U.size, 3))
            loc[:, i], loc[:, j], loc[:, k] = U.ravel(), W.ravel(), sgn * half[k]
            pts.append(c + loc @ A.T)
    return np.vstack(pts)


def penetration(D) -> float:
    """How far (mm) the housing runs into the base or the tube clamp, or the plate below the mount surface."""
    if D.kind == 'flat':
        hp = _housing_points(D, 4.0)
        return max(0.0, D.z_base + 0.5 - float(hp[:, 2].min()), -float(D.plate_corners[:, 2].min()))
    c = D.clamp
    hp = _housing_points(D, 1.5)
    tube = M.rod([-2000, 0, c['z_axis']], [2000, 0, c['z_axis']], c['r'] + 0.5)
    obst = M.union(D.saddle, D.cap, tube)
    return max(0.0, 0.5 - float(obst(hp).min()))


def required_lift(st, P, A, size, face, mount, n_local, limit=300.0) -> float:
    """The smallest move (mm) of the camera along the surface normal that clears the housing; 0 when it clears."""
    D = design(st, P, A, size, face, mount, access=False)
    if penetration(D) <= 1e-6:
        return 0.0
    if D.kind == 'flat':
        hp = _housing_points(D, 4.0)
        return max(D.z_base + 0.5 - float(hp[:, 2].min()), -float(D.plate_corners[:, 2].min()), 0.0)
    lo, hi = 0.0, limit
    if penetration(design(st, P + n_local * hi, A, size, face, mount, access=False)) > 0:
        return math.inf
    for _ in range(16):
        mid = (lo + hi) / 2
        if penetration(design(st, P + n_local * mid, A, size, face, mount, access=False)) > 0:
            lo = mid
        else:
            hi = mid
    return hi


# =========================================================================== the body under the base
def fit_base(mesh, R, o, rect, n_s=7, tilt=True):
    """Surface heights under the base rect (x0, x1, y0, y1) (mm, bracket frame R, o): rays cast down the frame's z
    from 20 mm above. Returns {'hits', 'share_missing', 'plane': (a, b, c) the supporting plane z = a x + b y + c
    (best-fit tilt, or none with tilt False, lifted onto the highest point), 'gap_max', 'gap_mean'} or None without
    hits."""
    x0, x1, y0, y1 = rect
    xs = np.linspace(x0 + 1, x1 - 1, n_s)
    ys = np.linspace(y0 + 1, y1 - 1, n_s)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    loc = np.column_stack([X.ravel(), Y.ravel(), np.full(X.size, 20.0)])
    O = o + (loc / 1000.0) @ R.T
    Dv = np.repeat(-R[:, 2][None], len(O), axis=0)
    t, f = mesh.raycast(O, Dv, faces_mask=mesh.occluder_faces(), max_t=0.06)
    hit = f >= 0
    share_missing = float(1 - hit.mean())
    if hit.sum() < 3:
        return None
    x, y, z = loc[hit, 0], loc[hit, 1], 20.0 - t[hit] * 1000.0
    a = b = 0.0
    if tilt and hit.sum() >= 6 and np.ptp(x) > 2 and np.ptp(y) > 2:
        Am = np.column_stack([x, y, np.ones_like(x)])
        sol, *_ = np.linalg.lstsq(Am, z, rcond=None)
        if math.degrees(math.atan(math.hypot(sol[0], sol[1]))) <= 12.0:
            a, b = float(sol[0]), float(sol[1])
    c = float((z - a * x - b * y).max())
    gaps = c + a * x + b * y - z
    return {'hits': int(hit.sum()), 'share_missing': share_missing, 'plane': (a, b, c),
            'gap_max': float(gaps.max()), 'gap_mean': float(gaps.mean())}


# =========================================================================== checks
def fov_blocked(solid, oc, R_lo, intr, n_w=72, t_max=None):
    """Share of the camera's image (pixels within the lens circle) whose ray from the optical centre oc (bracket
    frame, mm) meets the solid; R_lo turns optical-frame directions into the bracket frame. Returns (share, mask)."""
    from .synth import pixel_rays
    n_h = max(8, int(round(n_w * intr.height / intr.width)))
    d, th = pixel_rays(intr, n_w, n_h)
    ok = np.isfinite(th) & (th <= intr.max_theta) & np.all(np.isfinite(d), axis=1)
    mask = np.zeros(len(d), np.uint8)
    if not ok.any():
        return 0.0, mask.reshape(n_h, n_w)
    Dl = d[ok] @ np.asarray(R_lo).T
    if t_max is None:
        corners = np.array([[x, y, z] for x in (solid.lo[0], solid.hi[0]) for y in (solid.lo[1], solid.hi[1])
                            for z in (solid.lo[2], solid.hi[2])])
        t_max = float(np.linalg.norm(corners - oc, axis=1).max()) + 2.0
    hit, _ = M.raytrace(solid, np.repeat(np.asarray(oc, float)[None], len(Dl), axis=0), Dl, t_max, eps=0.02)
    m = np.ones(int(ok.sum()), np.uint8)
    m[hit] = 2
    mask[ok] = m
    return float(hit.mean()), mask.reshape(n_h, n_w)


def placement(B: Bracket, face_tol=3.0):
    """The optical centre and plate normal measured back from the mesh: the plate's contact face (a plane fitted to
    the triangles facing the camera there) and the camera holes' centres (circles fitted to their walls), with the
    housing's size: (optical centre (bracket frame, mm), plate normal towards the camera, hole diameters) or None."""
    if not B.generated or B.params.get('plate') is None:
        return None
    V, F = B.V.astype(float), B.F.astype(np.int64)
    pl = B.params['plate']
    u, Fc, e1, e2 = (np.array(pl[k]) for k in ('u', 'face_centre', 'e1', 'e2'))
    lo1, hi1, lo2, hi2 = pl['extent']
    tri = V[F]
    cen = tri.mean(axis=1)
    nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = np.linalg.norm(nrm, axis=1)
    nrm = nrm / np.maximum(area, 1e-12)[:, None]
    rel = cen - Fc
    s1, s2, sn = rel @ e1, rel @ e2, rel @ u
    inset = 1.0 + B.params.get('fillet_mm', 0.0)
    holes = [np.array(h) for h in pl['holes']]
    head = pl['hole_d'] / 2 + 0.6
    far = np.ones(len(F), bool)
    for hc in holes:
        hr = hc - Fc
        far &= np.hypot(s1 - hr @ e1, s2 - hr @ e2) > head
    sel = (nrm @ -u > 0.995) & (np.abs(sn) < face_tol) & (s1 > lo1 + inset) & (s1 < hi1 - inset) & \
          (s2 > lo2 + inset) & (s2 < hi2 - inset) & far
    if sel.sum() < 6:
        return None
    pts = tri[sel].reshape(-1, 3)
    c0 = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - c0, full_matrices=False)
    nf = vt[2] if vt[2] @ -u > 0 else -vt[2]
    # hole walls: triangles near each hole's axis whose normals lie in the face plane
    centres, diam = [], []
    t = B.params.get('wall_mm', 3.0)
    for hc in holes:
        hr = hc - Fc
        dd = np.hypot(s1 - hr @ e1, s2 - hr @ e2)
        wsel = (dd < pl['hole_d'] / 2 + 0.6) & (np.abs(nrm @ nf) < 0.3) & (sn > 0.3) & (sn < t - 0.3)
        if wsel.sum() < 6:
            return None
        q = tri[wsel].reshape(-1, 3)
        q = q - np.outer((q - c0) @ nf, nf)               # onto the contact plane
        b1 = _unit(_perp(e1, nf))
        b2 = np.cross(nf, b1)
        x, y = (q - c0) @ b1, (q - c0) @ b2
        Am = np.column_stack([x, y, np.ones_like(x)])
        sol, *_ = np.linalg.lstsq(Am, x * x + y * y, rcond=None)
        cx, cy = sol[0] / 2, sol[1] / 2
        diam.append(2 * math.sqrt(max(sol[2] + cx * cx + cy * cy, 0.0)))
        centres.append(c0 + b1 * cx + b2 * cy)
    C = np.mean(centres, axis=0)
    size = B.params['housing_mm']
    h, L = size['height'], size['length']
    s = B.params.get('setback_mm', 0.0)
    if B.face == 'back':
        oc = C + nf * (L - s)
    else:
        ax = B.axis / np.linalg.norm(B.axis)
        oc = C + nf * (h / 2) + _unit(_perp(ax, nf)) * (L / 2 - s)
    return oc, nf, diam


def print_orientation(V, F, kind):
    """For each piece of the mesh: the up direction (bracket frame) with the least overhang (faces turned more than
    45° down that are not on the bed) and that area (mm²)."""
    tri = V[F]
    cr = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    ar = np.linalg.norm(cr, axis=1) / 2
    n = cr / np.maximum(2 * ar, 1e-12)[:, None]
    cands = {'with the base on the bed': np.array([0, 0, 1.0]), 'upside down': np.array([0, 0, -1.0]),
             'on its side (+x down)': np.array([-1.0, 0, 0]), 'on its side (−x down)': np.array([1.0, 0, 0]),
             'on its side (+y down)': np.array([0, -1.0, 0]), 'on its side (−y down)': np.array([0, 1.0, 0])}
    out = []
    for faces in M.shells(V, F):
        best = None
        for name, up in cands.items():
            h = tri[faces].reshape(-1, 3) @ up
            hmin = h.min()
            fh = (tri[faces] @ up).max(axis=1)
            nu = n[faces] @ up
            on_bed = (nu < -0.99) & (fh < hmin + 0.3)
            over = (nu < -0.7071) & ~on_bed
            a = float(ar[faces][over].sum())
            key = (round(a, 0), 0 if name == 'with the base on the bed' else 1)
            if best is None or key < best[0]:
                best = (key, name, up, a)
        out.append({'faces': int(len(faces)), 'orientation': best[1], 'up': best[2].tolist(),
                    'overhang_mm2': best[3]})
    out.sort(key=lambda d: -d['faces'])
    return out


def _wall(D, st):
    """The thinnest designed wall: (mm, what)."""
    items = [(st.wall_mm, 'plate')]
    if D.kind == 'flat':
        items.append((D.tb, 'base'))
        x0, x1, y0, y1 = D.base_rect
        for p in D.screws:
            lig = min(p[0] - x0, x1 - p[0], p[1] - y0, y1 - p[1]) - D.base_screw[2]
            items.append((lig, 'base edge beside a screw'))
    else:
        c = D.clamp
        items += [(c['tc'], 'clamp ring'), (c['et'], 'clamp ear'),
                  (c['ets'] - c['nut'][1] - 0.3, 'saddle ear under the nut')]
        items.append(((c['ew'] - c['bolt_d']) / 2, 'clamp ear beside the bolt'))
    lo1, hi1, lo2, hi2 = D.plate_extent
    for o1, o2 in D.hole_off:
        items.append((min(o1 - lo1, hi1 - o1, o2 - lo2, hi2 - o2) - D.hole_d / 2, 'plate edge beside a hole'))
    if len(D.hole_off) > 1:
        dmin = min(np.hypot(*(a - b)) for i, a in enumerate(D.hole_off) for b in D.hole_off[i + 1:])
        items.append((dmin - D.hole_d, 'between the camera holes'))
    if D.connector:
        items.append((D.flags.get('connector_lig', 99.0), 'between the connector hole and a screw hole'))
    return min(items, key=lambda x: x[0])


def _stiffness(D, st, module, solid):
    """Beam estimate: the support as a cantilever (its cross-section half way up to the plate's centre, its weaker
    principal axis) carrying the camera at the housing centre's height. (first mode Hz, tilt ° under DESIGN_G
    sideways, I mm⁴) or None when the plate sits on the base."""
    z0 = D.z_base
    zp = float(D.plate[0][2])
    Lc = float(D.housing[0][2] - z0)
    if D.support is None or zp - z0 < 2.0 or Lc < 2.0:
        return None
    zm = z0 + 0.5 * (zp - z0)
    lo, hi = solid.lo, solid.hi
    ctr = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, zm])
    A, _, Imin, _ = M.section(solid, ctr, np.array([1.0, 0, 0]), np.array([0, 1.0, 0]),
                              ((hi[0] - lo[0]) / 2 + 1, (hi[1] - lo[1]) / 2 + 1), h=0.3)
    if A <= 0 or Imin <= 0:
        return None
    E = MATERIALS[st.material]['E']
    k = 3 * E * Imin / Lc ** 3                       # N/mm
    m = _weight_g(module) / 1000.0
    f1 = math.sqrt(k * 1000.0 / m) / (2 * math.pi)
    defl = m * 9.81 * DESIGN_G / k
    tilt = math.degrees(1.5 * defl / Lc)
    return f1, tilt, Imin


def _body_intrusion(vg, B, D):
    """The deepest point (mm) where the housing or the bracket runs into the vehicle body (beyond the base's own
    contact), or 0."""
    mesh = vg.occ_mesh
    pts = [_housing_points(D, 3.0)]
    if B.V is not None and len(B.V):
        V = B.V.astype(float)
        keep = V[:, 2] > D.z_base + 0.8
        sub = V[keep]
        if len(sub) > 600:
            sub = sub[np.random.default_rng(0).choice(len(sub), 600, replace=False)]
        pts.append(sub)
    Pl = np.vstack(pts)
    Pv = B.origin + (Pl / 1000.0) @ B.R.T
    c = Pv.mean(axis=0)
    rad = float(np.linalg.norm(Pv - c, axis=1).max()) + 0.05
    tri = mesh.tri
    near = np.linalg.norm(mesh.centroids - c, axis=1) < rad + np.linalg.norm(tri - tri.mean(axis=1)[:, None],
                                                                              axis=2).max(axis=1)
    near &= mesh.occluder_faces()
    if not near.any():
        return 0.0
    idx = np.nonzero(near)[0]
    T = tri[idx]
    worst = 0.0
    from .mesh import closest_on_triangles
    from .rules import _inside
    cand = []
    for p in Pv:
        q = closest_on_triangles(p, T)
        d = np.linalg.norm(q - p, axis=1)
        j = int(np.argmin(d))
        if (p - q[j]) @ mesh.normals[idx[j]] < -0.001:
            cand.append((d[j], p))
    cand.sort(key=lambda x: -x[0])
    for d, p in cand[:6]:
        if _inside(mesh, p):
            worst = max(worst, d * 1000.0)
            break
    return worst


# =========================================================================== one camera
def _weight_g(module) -> float:
    try:
        w = float(module.weight_g)
    except (TypeError, ValueError):
        return 45.0
    return w if math.isfinite(w) and w > 0 else 45.0


def _camera_geometry(project, cam):
    m = project.module_for(cam)
    sz = tuple(float(v) for v in (list(m.size_mm) + [25.0, 25.0, 43.8])[:3])
    size = tuple(v if math.isfinite(v) and v > 1.0 else d for v, d in zip(sz, (25.0, 25.0, 43.8)))
    return m, size, rot_zyx(cam.yaw, cam.pitch, cam.roll)


def _local(R, o, P, Rc):
    """The camera's optical centre (mm) and axes in the bracket frame (R, o)."""
    return R.T @ (P - o) * 1000.0, R.T @ Rc


def _mount_local(mount):
    return {'kind': mount['kind'], 'r_mm': mount.get('r_mm', 0.0)}


def build(project, vg, cam, st: BracketSettings, intr=None, progress=None, cancel=None, mesh_it=True) -> Bracket:
    """One camera's bracket (mesh_it False: the design and checks that need no mesh: status, standoff)."""
    t0 = time.perf_counter()
    B = Bracket(cam.uid, cam.tag, cam.name, cam.slot, plan_standoff_mm=float(cam.standoff_mm))
    if str(cam.bracket).strip().lower() == 'none':
        B.status, B.message = 'skipped', 'Bracket set to None (Inspector ▸ Mount): flush mounted, no printed bracket.'
        return B
    if not all(math.isfinite(v) for v in (cam.x, cam.y, cam.z, cam.yaw, cam.pitch, cam.roll)):
        B.status, B.message = 'skipped', 'The camera\'s position or angles are not numbers.'
        return B
    mount = find_mount(project, vg, cam, st.reach_mm)
    B.kind, B.surface, B.gap_mm = mount['kind'], mount['surface'], float(mount['gap_mm'])
    B.normal, B.contact = tuple(float(v) for v in mount['n']), tuple(float(v) for v in mount['q'])
    if mount['inside']:
        B.status = 'inside'
        B.message = (f'The optical centre is {abs(B.gap_mm):.0f} mm inside {B.surface}: a flush or recessed mount, '
                     'not a surface bracket.')
        return B
    if mount['floating']:
        B.status = 'floating'
        B.message = (f'{B.gap_mm:.0f} mm from the nearest surface ({B.surface}), farther than the '
                     f'{st.reach_mm:.0f} mm a printed bracket reaches without flexing: needs a custom mount, or add a '
                     'rig part to carry it (Vehicle ▸ Add rig part ▸ Crossbar).')
        return B
    module, size, Rc = _camera_geometry(project, cam)
    n = np.asarray(mount['n'], float)
    face, score, best = choose_face(st, Rc, n)
    B.face = face
    if score < FACE_MIN:
        B.status = 'face'
        off = math.degrees(math.acos(np.clip(score, -1, 1)))
        B.message = (f'The housing\'s {face} face turns away from {B.surface} ({off:.0f}° from facing it): the plate '
                     f'cannot reach the surface. Choose the {best} face (or Auto).')
        return B
    u_v = {'back': -Rc[:, 0], 'bottom': -Rc[:, 2], 'top': Rc[:, 2]}[face]
    R = _frame(n, u_v, Rc, mount.get('b'))
    o = np.asarray(mount['q'], float)
    P = cam.pos()
    ml = _mount_local(mount)
    mass = _weight_g(module) + 15.0                      # the camera and a typical bracket
    fit = None
    if mount['kind'] == 'flat':
        # 1: on the tangent plane at the contact point; fit the base plane to the body under that base
        D = design(st, *_local(R, o, P, Rc), size, face, ml, access=False, mass_g=mass)
        fit = fit_base(mount['mesh'], R, o, D.base_rect)
        if fit is not None:
            a, b, c = fit['plane']
            o = o + R[:, 2] * c / 1000.0
            R = _frame(R @ np.array([-a, -b, 1.0]), u_v, Rc)
        # 2: on the fitted plane: centre the frame on the base and let the base rest on the body under it
        D = design(st, *_local(R, o, P, Rc), size, face, ml, access=True, mass_g=mass)
        x0, x1, y0, y1 = D.base_rect
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        o = o + R @ np.array([cx, cy, 0.0]) / 1000.0
        f2 = fit_base(mount['mesh'], R, o, (x0 - cx, x1 - cx, y0 - cy, y1 - cy), tilt=False)
        if f2 is not None:
            o = o + R[:, 2] * f2['plane'][2] / 1000.0
            fit = f2
    D = design(st, *_local(R, o, P, Rc), size, face, ml, access=True, mass_g=mass)
    B.R, B.origin = R, o
    Pl, Al = _local(R, o, P, Rc)
    B.oc, B.axis = Pl, Al[:, 0]
    B.housing = D.housing
    nl = R.T @ n
    lift = required_lift(st, Pl, Al, size, face, ml, nl)
    if lift > 0.05:
        B.status = 'standoff'
        B.required_mm = math.inf if not math.isfinite(lift) else math.ceil(B.gap_mm + lift + 1.0)
        where = f'At {B.gap_mm:.0f} mm from {B.surface} the housing runs {lift:.0f} mm into the ' + \
            ('base' if D.kind == 'flat' else 'clamp')
        if not math.isfinite(B.required_mm):
            B.message = where + ': no standoff clears it; choose another face.'
        elif B.required_mm > st.reach_mm:
            B.message = (where + f': a printed bracket would need {B.required_mm:.0f} mm standoff, beyond its '
                         f'{st.reach_mm:.0f} mm reach. Mount it flush through a cut-out, or on a custom mount.')
        else:
            B.message = where + (f': a printed bracket needs {B.required_mm:.0f} mm standoff (Move out) or a '
                                 'cut-out in the panel.')
        B.params = _params(B, D, st, size, mount, fit)
        B.seconds = time.perf_counter() - t0
        return B
    B.params = _params(B, D, st, size, mount, fit)
    B.solid = D.solid
    if not mesh_it:
        B.status = 'ok'
        B.message = 'Ready to generate.'
        return B
    V, F = M.mesh(D.solid, st.res_mm, progress=progress, cancel=cancel)
    if V is None:
        return None
    B.V, B.F = V.astype(np.float32), F.astype(np.int32)
    chk = M.check(V, F)
    rho = MATERIALS[st.material]['rho']
    chk['mass_g'] = chk['volume_mm3'] / 1000.0 * rho
    B.mesh = chk
    _checks(B, D, st, project, vg, cam, module, intr, mount, fit)
    B.display = display_mesh(B, 30000)
    B.display_fast = display_mesh(B, 6000)
    sev = B.worst()
    B.status = 'ok' if sev == 'good' else 'warning'
    worst = [c for c in B.checks if c['severity'] == sev]
    B.message = ('Generated: every check passes.' if sev == 'good' else worst[0]['text'])
    B.seconds = time.perf_counter() - t0
    return B


def _params(B, D, st, size, mount, fit):
    x0, x1, y0, y1 = D.base_rect
    p = {'housing_mm': {'width': size[0], 'height': size[1], 'length': size[2]},
         'setback_mm': st.setback_mm, 'wall_mm': st.wall_mm, 'margin_mm': st.margin_mm, 'fillet_mm': st.fillet_mm,
         'material': st.material, 'face': D.face,
         'plate_angle_deg': float(math.degrees(math.acos(np.clip(-D.u[2], -1, 1)))),
         'plate': {'u': D.u.tolist(), 'face_centre': D.Fc.tolist(), 'e1': D.e1.tolist(), 'e2': D.e2.tolist(),
                   'extent': list(D.plate_extent), 'thickness_mm': st.wall_mm,
                   'holes': [h.tolist() for h in D.holes_cam], 'hole_d': D.hole_d,
                   'pattern': st.pattern_label(), 'head_d': D.head},
         'height_mm': float(max(D.plate_corners[:, 2].max(), D.z_base)),
         'oc_height_mm': float(D.P[2]),
         'base_top_mm': D.z_base,
         'lever_mm': float(np.linalg.norm(D.housing[0] - np.array([(x0 + x1) / 2, (y0 + y1) / 2, D.z_base]))),
         'support_height_mm': max(D.support_H, 0.0)}
    if D.kind == 'flat':
        p['base'] = {'length_mm': x1 - x0, 'width_mm': y1 - y0, 'thickness_mm': D.tb, 'rect': [x0, x1, y0, y1],
                     'fixing': st.fixing, 'area_mm2': D.base_area}
        if st.fixing != 'vhb':
            p['base']['screws'] = {'size': D.base_screw[0], 'hole_d': D.base_screw[1], 'countersunk': st.countersink,
                                   'positions': [s[:2].tolist() for s in D.screws]}
        else:
            p['base']['tape_mm'] = st.tape_mm
            p['base']['tape_needed_mm2'] = D.tape_need
    else:
        c = D.clamp
        p['clamp'] = {'tube_radius_mm': c['r'], 'bore_mm': 2 * c['rb'], 'ring_wall_mm': c['tc'],
                      'length_mm': c['x1'] - c['x0'], 'width_mm': c['yhi'] - c['ylo'], 'split_gap_mm': c['gap'],
                      'bolt': c['bolt'], 'bolt_hole_d': c['bolt_d'], 'bolts': len(D.bolts),
                      'bolt_positions': [b[:2].tolist() for b in D.bolts], 'saddle_top_mm': c['z_top']}
        p['base'] = {'length_mm': x1 - x0, 'width_mm': y1 - y0, 'thickness_mm': c['z_top'] - c['z_axis'],
                     'rect': [x0, x1, y0, y1], 'fixing': 'tube clamp', 'area_mm2': D.base_area}
    if fit is not None:
        p['surface_fit'] = {'gap_max_mm': fit['gap_max'], 'gap_mean_mm': fit['gap_mean'],
                            'share_off_surface': fit['share_missing']}
    return p


def _checks(B, D, st, project, vg, cam, module, intr, mount, fit):
    out = B.checks
    chk = B.mesh
    if not chk.get('ok'):
        _check(out, 'mesh', 'critical', f'The mesh is not a closed solid ({chk["triangles"]} triangles, '
                                        f'watertight {chk["watertight"]}, consistent {chk["consistent"]}): '
                                        'do not print it; report the camera and settings.')
    else:
        _check(out, 'mesh', 'good', f'Closed, consistently wound mesh: {chk["triangles"]:,} triangles, '
                                    f'{chk["volume_mm3"] / 1000:.1f} cm³ ({chk["mass_g"]:.0f} g of {st.material} '
                                    'solid).', chk['volume_mm3'])
    # placement measured back from the mesh
    pm = placement(B)
    if pm is not None:
        oc, nf, diam = pm
        d_oc = float(np.linalg.norm(oc - B.oc))
        want = -D.u
        ang = math.degrees(math.acos(np.clip(nf @ want, -1, 1)))
        sev = 'good' if d_oc <= 0.5 and ang <= 0.5 else 'serious'
        _check(out, 'placement', sev, f'Measured on the mesh: the optical centre lands {d_oc:.2f} mm and the plate '
                                      f'{ang:.2f}° from the plan.', d_oc)
        B.params['measured'] = {'oc_error_mm': d_oc, 'plate_angle_error_deg': ang, 'hole_d_mm': diam}
    # field of view
    if intr is None:
        _check(out, 'fov', 'warning', 'The lens model does not resolve: the image was not checked for the bracket.')
    else:
        R_lo = B.R.T @ rot_zyx(cam.yaw, cam.pitch, cam.roll) @ BODY_TO_OPTICAL.T
        share, mask = fov_blocked(D.solid, B.oc, R_lo, intr)
        B.fov_mask = mask
        B.params['fov_blocked_pct'] = share * 100
        if share > 0:
            _check(out, 'fov', 'serious' if share > 0.05 else 'warning',
                   f'The bracket shows in {share * 100:.1f} % of the image. Move the camera out, choose another '
                   'face or a smaller margin.', share * 100)
        else:
            _check(out, 'fov', 'good', 'The bracket stays out of the image.', 0.0)
    # lever arm
    lever = B.params['lever_mm']
    if lever > st.overhang_mm:
        _check(out, 'overhang', 'warning', f'The camera sits {lever:.0f} mm from the base, beyond '
                                           f'{st.overhang_mm:.0f} mm: it will shake; add a strut or a rig part.', lever)
    else:
        _check(out, 'overhang', 'good', f'Lever arm {lever:.0f} mm from the base.', lever)
    # walls
    wv, what = _wall(D, st)
    wv = round(wv, 2)
    if wv < 1.2:
        _check(out, 'wall', 'serious', f'Thinnest wall {wv:.1f} mm ({what}): too thin to print soundly. Raise the '
                                       'wall or the margin.', wv)
    elif wv < 2.0:
        _check(out, 'wall', 'warning', f'Thinnest wall {wv:.1f} mm ({what}), under 2 mm.', wv)
    else:
        _check(out, 'wall', 'good', f'Thinnest wall {wv:.1f} mm ({what}).', wv)
    if st.res_mm > D.hole_d / 4:
        _check(out, 'resolution', 'warning', f'The mesh grid ({st.res_mm:g} mm) is coarse for the '
                                             f'{D.hole_d:.1f} mm holes: lower the resolution or drill them out.',
               st.res_mm)
    if D.flags.get('pattern_wide'):
        _check(out, 'pattern', 'warning', f'The hole pattern ({st.pattern_label()}) reaches past the {D.face} face: '
                                          'the plate is made larger to carry it. Check the pattern.')
    if D.flags.get('plate_long'):
        _check(out, 'pattern', 'warning', 'The hole pattern reaches the lens end of the housing: the plate runs to '
                                          'the lens front.')
    if D.access_blocked:
        _check(out, 'access', 'warning', 'A screwdriver cannot reach every base screw past the camera: fit them '
                                         'before the camera.')
    # the base on the body
    if fit is not None:
        g = fit['gap_max']
        lim = st.tape_mm if st.fixing == 'vhb' else 1.0
        if fit['share_missing'] > 0.1:
            _check(out, 'edge', 'warning', f'{fit["share_missing"] * 100:.0f} % of the base hangs over the edge of '
                                           f'{B.surface}.', fit['share_missing'] * 100)
        if g > lim:
            how = 'more than the tape fills: shim it or bed it in sealant' if st.fixing == 'vhb' else \
                'it rocks on the curved panel: bed it in sealant or add a shim'
            _check(out, 'base_gap', 'warning', f'Gaps up to {g:.1f} mm under the base, {how}.', g)
        else:
            _check(out, 'base_gap', 'good', f'The base sits on {B.surface} with gaps under {max(g, 0.05):.1f} mm.', g)
    if st.fixing == 'vhb' and D.kind == 'flat' and D.tape_need > D.base_area + 1:
        _check(out, 'tape', 'warning', f'The tape pad has {D.base_area / 100:.0f} cm², under the '
                                       f'{D.tape_need / 100:.0f} cm² the 3M rule asks for at {DESIGN_G:g} g: use '
                                       'screws.', D.base_area)
    if mount.get('zone_kind') == 'forbidden':
        _check(out, 'forbidden', 'serious', f'{B.surface} is marked as a surface not to mount on.')
    # connector
    if D.connector:
        if D.flags.get('connector_lig', 99) < 0.8:
            _check(out, 'connector', 'warning', f'The {D.connector:g} mm connector hole runs into the screw holes: '
                                                'use the bottom or top face, or a wider pattern.')
        u = D.u
        if u[2] < -1e-6:
            t0 = -D.Fc[2] / u[2]
            p0 = D.Fc + u * t0
            x0, x1, y0, y1 = D.base_rect
            if x0 <= p0[0] <= x1 and y0 <= p0[1] <= y1:
                _check(out, 'connector', 'warning', 'The connector and cable leave through the base into the panel: '
                                                    'drill the panel there, or hold the camera by its bottom or top '
                                                    'face.')
    # into the body
    pen = _body_intrusion(vg, B, D)
    if pen > 1.0:
        _check(out, 'body', 'serious', f'The housing or the bracket runs {pen:.0f} mm into the body: move the camera '
                                       'out or turn it.', pen)
    # stiffness
    stf = _stiffness(D, st, module, D.solid)
    if stf is not None:
        f1, tilt, I = stf
        B.params['stiffness'] = {'first_mode_hz': f1, 'tilt_deg_at_g': tilt, 'design_g': DESIGN_G,
                                 'section_I_mm4': I, 'E_mpa': MATERIALS[st.material]['E']}
        sev = 'warning' if f1 < 60 else 'good'
        _check(out, 'stiffness', sev, f'Beam estimate: first mode about {f1:.0f} Hz, {tilt:.2f}° tilt at '
                                      f'{DESIGN_G:g} g in {st.material}' + (' (road vibration reaches it: stiffen '
                                                                             'it).' if f1 < 60 else '.'), f1)
    # print
    po = print_orientation(B.V.astype(float), B.F.astype(np.int64), D.kind)
    B.params['print'] = po
    if po:
        B.print_up = np.array(po[0]['up'])
        if D.kind == 'clamp' and len(po) > 1:
            text = (f'Print the saddle {po[0]["orientation"]} ({po[0]["overhang_mm2"] / 100:.1f} cm² overhang) and '
                    f'the cap {po[1]["orientation"]}.')
        else:
            text = f'Print {po[0]["orientation"]}: {po[0]["overhang_mm2"] / 100:.1f} cm² overhang needs support.'
        B.notes.append(text)
    B.notes.append(MATERIALS[st.material]['note'])
    if D.kind == 'clamp':
        c = D.clamp
        B.notes.append(f'Tube clamp for the {2 * c["r"]:.0f} mm tube of {mount.get("part", "the rig")}: press '
                       f'{len(D.bolts)} {c["bolt"]} nuts into the saddle\'s hex pockets, put the bolts in from '
                       f'under the cap; the halves close on the {c["gap"]:.1f} mm split gap.')
    else:
        if st.fixing == 'vhb':
            B.notes.append(f'Fix the base with {st.tape_mm:g} mm VHB tape on a cleaned, primed panel; let it cure '
                           '72 h before driving.')
        else:
            n_s = len(D.screws)
            B.notes.append(f'Fix the base with {n_s} × {D.base_screw[0]} ' + ('countersunk ' if st.countersink else '')
                           + 'screws (rivet nuts in the panel).')
    B.notes.append('Screw the camera to the plate before fixing the bracket: the screw heads are reached through '
                   'the support.')
    if abs(B.gap_mm - B.plan_standoff_mm) > 2.0:
        B.notes.append(f'The camera is {B.gap_mm:.0f} mm from {B.surface}; its planned standoff says '
                       f'{B.plan_standoff_mm:.0f} mm. The bracket follows the camera\'s position.')


# =========================================================================== the study
def _cameras(variant, cams):
    vis = [c for c in variant.cameras if c.visible]
    if cams is None:
        return vis
    want = {getattr(c, 'uid', c) for c in cams} | {getattr(c, 'tag', c) for c in cams}
    return [c for c in vis if c.uid in want or c.tag in want]


def run(project, engine=None, settings=None, cams=None, variant=None, progress=None, cancel=None) -> BracketResult:
    """A bracket for each visible camera of the variant (cams: uids, tags or cameras to limit it to). None when
    cancelled."""
    t0 = time.perf_counter()
    st = settings if isinstance(settings, BracketSettings) else \
        BracketSettings.from_dict(settings if isinstance(settings, dict) else project.analysis.brackets)
    v = variant or project.variant
    if engine is None:
        from .analysis import Engine
        engine = Engine()
    vg = engine.vehicle_geom(project)
    todo = _cameras(v, cams)
    out = []
    from .lens import resolve
    for i, cam in enumerate(todo):
        if cancel is not None and cancel():
            return None
        if progress is not None:
            progress(i / max(len(todo), 1), cam.tag)
        if vg is None:
            out.append(Bracket(cam.uid, cam.tag, cam.name, cam.slot, status='skipped',
                               message='No vehicle model to mount on.'))
            continue
        try:
            intr = resolve(project.module_for(cam), project.lens_for(cam))
        except Exception:                            # a broken lens model: no image check, the bracket still made
            intr = None

        def sub(f, i=i):
            if progress is not None:
                progress((i + 0.9 * f) / max(len(todo), 1), cam.tag)
        b = build(project, vg, cam, st, intr, sub, cancel)
        if b is None:
            return None
        out.append(b)
    if progress is not None:
        progress(1.0, '')
    return BracketResult(st, out, v.name, v.uid, time.perf_counter() - t0)


def preflight(project, engine=None, settings=None, variant=None) -> list:
    """Each visible camera's bracket status without meshing (cheap, for the page's list): Brackets without meshes."""
    st = settings if isinstance(settings, BracketSettings) else BracketSettings.from_dict(project.analysis.brackets)
    v = variant or project.variant
    if engine is None:
        from .analysis import Engine
        engine = Engine()
    vg = engine.vehicle_geom(project)
    out = []
    for cam in _cameras(v, None):
        if vg is None:
            out.append(Bracket(cam.uid, cam.tag, cam.name, cam.slot, status='skipped',
                               message='No vehicle model to mount on.'))
            continue
        out.append(build(project, vg, cam, st, None, mesh_it=False))
    return out


def plan(project, engine=None, settings=None, variant=None) -> BracketResult:
    """preflight() as a result (no meshes): what summary() and the assistant read without the cost of meshing."""
    t0 = time.perf_counter()
    st = settings if isinstance(settings, BracketSettings) else \
        BracketSettings.from_dict(settings if isinstance(settings, dict) else project.analysis.brackets)
    v = variant or project.variant
    return BracketResult(st, preflight(project, engine, st, v), v.name, v.uid, time.perf_counter() - t0)


# =========================================================================== geometry out
def vehicle_mesh(b: Bracket):
    """The bracket's mesh in the vehicle frame: (V metres, F), for drawing in the 3D view."""
    if not b.generated:
        return np.zeros((0, 3)), np.zeros((0, 3), np.int64)
    return b.origin + (b.V.astype(float) / 1000.0) @ b.R.T, b.F.astype(np.int64)


def transform(b: Bracket) -> np.ndarray:
    """T_vehicle_bracket (4 × 4): bracket frame in metres → vehicle frame, metres."""
    T = np.eye(4)
    T[:3, :3] = b.R
    T[:3, 3] = b.origin
    return T


def stl(b: Bracket) -> bytes:
    hdr = f'Vantage bracket {b.tag}: mm, origin base contact, z = surface normal'
    return M.stl_bytes(b.V.astype(float), b.F.astype(np.int64), hdr)


def housing_mesh(b: Bracket):
    """The camera housing and its lens front (bracket frame, mm) as triangles, for drawing."""
    if b.housing is None:
        return np.zeros((0, 3)), np.zeros((0, 3), np.int64)
    c, A, half = b.housing
    V1, F1 = M.box_mesh(c, A, half)
    front = c + A[:, 0] * half[0]
    r = 0.36 * min(half[1], half[2]) * 2
    V2, F2 = M.disc_mesh(front, A[:, 0], r)
    return np.vstack([V1, V2]), np.vstack([F1, F2 + len(V1)])


# =========================================================================== output
def _r(v, d=3):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return round(v, d) if math.isfinite(v) else None


def summary(res: BracketResult) -> dict:
    """Compact figures (the assistant, the batch log). Geometry only for cameras that can take a bracket."""
    count = {}
    for b in res.brackets:
        count[b.status] = count.get(b.status, 0) + 1
    rows = []
    for b in res.brackets:
        row = {'tag': b.tag, 'status': b.status, 'message': b.message, 'mount': b.kind, 'face': b.face,
               'surface': b.surface, 'standoff_mm': _r(b.gap_mm, 1), 'required_standoff_mm': _r(b.required_mm, 1)}
        if b.status not in NOT_BUILT and 'base' in b.params:
            p = b.params
            row.update(base_mm=[_r(p['base']['length_mm'], 1), _r(p['base']['width_mm'], 1),
                                _r(p['base']['thickness_mm'], 1)],
                       height_mm=_r(p.get('height_mm'), 1), plate_angle_deg=_r(p.get('plate_angle_deg'), 1),
                       lever_mm=_r(p.get('lever_mm'), 1), fov_blocked_pct=_r(p.get('fov_blocked_pct'), 2),
                       triangles=b.mesh.get('triangles'), generated=b.generated)
        row['warnings'] = [c['text'] for c in b.checks if c['severity'] != 'good']
        rows.append(row)
    return {'cameras': len(res.brackets), 'generated': sum(b.generated for b in res.brackets),
            'by_status': count, 'material': res.settings.material, 'pattern': res.settings.pattern_label(),
            'resolution_mm': res.settings.res_mm, 'per_camera': rows, 'assumptions': ASSUMPTIONS}


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, (np.floating, float)):
        return _r(o, 4)
    if isinstance(o, np.integer):
        return int(o)
    return o


def _bracket_doc(b: Bracket) -> dict:
    d = {'tag': b.tag, 'uid': b.uid, 'name': b.name, 'status': b.status, 'message': b.message, 'mount': b.kind,
         'face': b.face, 'surface': b.surface, 'standoff_measured_mm': b.gap_mm,
         'standoff_planned_mm': b.plan_standoff_mm, 'required_standoff_mm': b.required_mm,
         'surface_normal_vehicle': b.normal, 'contact_point_vehicle_m': b.contact}
    if b.oc is not None and b.status not in NOT_BUILT:
        d['T_vehicle_bracket'] = transform(b)
        d['optical_centre_mm'] = b.oc
        d['optical_axis'] = b.axis
        d['optical_centre_vehicle_m'] = b.origin + b.R @ (b.oc / 1000.0)
        d['parameters'] = b.params
    d['checks'] = b.checks
    d['notes'] = b.notes
    if b.generated:
        d['mesh'] = b.mesh
        d['stl'] = f'brackets/{safe_tag(b.tag)}.stl'
    return _clean(d)


def to_doc(res: BracketResult, project=None) -> dict:
    """The JSON export: every bracket's parameters, angles, frame and checks."""
    return {'schema': SCHEMA, 'project': getattr(project, 'name', ''), 'variant': res.variant,
            'units': 'millimetres and degrees in the bracket frame; metres in the vehicle frame',
            'frame': FRAME_TEXT, 'vehicle_frame': 'vehicle (ISO 8855, rear axle)',
            'settings': res.settings.to_dict(), 'pattern': res.settings.pattern_label(),
            'material': dict(MATERIALS[res.settings.material], name=res.settings.material),
            'assumptions': ASSUMPTIONS, 'brackets': [_bracket_doc(b) for b in res.brackets]}


def files(res: BracketResult, project=None, tags=None) -> list:
    """[(name, bytes)]: brackets/<TAG>.stl per generated bracket, brackets/brackets.json, brackets/brackets_sheet.png
    (tags: only these cameras' STLs, and no JSON or sheet)."""
    out = []
    for b in res.brackets:
        if b.generated and (tags is None or b.tag in tags):
            out.append((f'brackets/{safe_tag(b.tag)}.stl', stl(b)))
    if tags is None:
        out.append(('brackets/brackets.json', (json.dumps(to_doc(res, project), indent=1) + '\n').encode('utf-8')))
        out.append(('brackets/brackets_sheet.png', sheet_png(res, project)))
    return out


def export(res: BracketResult, project, out_dir, tags=None) -> list[str]:
    """Write files() under out_dir; returns the paths."""
    paths = []
    for name, data in files(res, project, tags):
        path = os.path.join(out_dir, *name.split('/'))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as fh:
            fh.write(data)
        paths.append(path)
    return paths


# =========================================================================== drawing
BRACKET_RGB = (196, 198, 202)
HOUSING_RGB = (58, 135, 229)


def display_mesh(b: Bracket, target=40000):
    """The bracket's mesh thinned for drawing (vertex clustering; not for printing)."""
    from .mesh import Mesh
    V, F = b.V.astype(float), b.F.astype(np.int64)
    if len(F) <= target:
        return V, F
    m, _ = Mesh(V, F).decimate(target)
    return m.vertices, m.faces


def views(b: Bracket):
    """The sheet's three views: (name, basis) top, side (along the plate's edge: its true angle), three-quarter."""
    return [('Top', M.view_basis(-90, 90)), ('Side', M.view_basis(-90, 0)), ('Three-quarter', M.view_basis(-55, 28))]


def picture(b: Bracket, basis, size, scale=None, centre=None, bg=(255, 255, 255), housing_rgb=HOUSING_RGB,
            mesh=None):
    """RGB image of the bracket with the housing see-through, and (scale px/mm, centre mm) used."""
    V, F = mesh if mesh is not None else (b.display or display_mesh(b))
    HV, HF = housing_mesh(b)
    allv = np.vstack([V, HV]) if len(HV) else V
    if centre is None:
        centre = (allv.min(axis=0) + allv.max(axis=0)) / 2
    if scale is None:
        q = (allv - centre) @ np.asarray(basis).T
        ext = np.abs(q[:, :2]).max(axis=0) * 2 + 1e-6
        scale = 0.84 * min(size[0] / ext[0], size[1] / ext[1])
    layers = [(V, F, BRACKET_RGB, 1.0), (HV, HF, housing_rgb, 0.45)]
    img = M.render(layers, basis, size, scale, centre, bg=bg)
    return img, scale, centre


def _arrow(img, p0, p1, col, text=None, cv2=None):
    p0 = tuple(int(round(v)) for v in p0)
    p1 = tuple(int(round(v)) for v in p1)
    cv2.arrowedLine(img, p0, p1, col, 1, cv2.LINE_AA, tipLength=0.08)
    cv2.arrowedLine(img, p1, p0, col, 1, cv2.LINE_AA, tipLength=0.08)
    if text:
        text = _ascii(text)
        mx, my = (p0[0] + p1[0]) // 2, (p0[1] + p1[1]) // 2
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        cv2.rectangle(img, (mx - tw // 2 - 2, my - th // 2 - 3), (mx + tw // 2 + 2, my + th // 2 + 3),
                      (255, 255, 255), -1)
        cv2.putText(img, text, (mx - tw // 2, my + th // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1, cv2.LINE_AA)


def _ascii(text) -> str:
    """Text for OpenCV's Hershey fonts (ASCII only)."""
    rep = {'×': 'x', '·': '-', '°': ' deg', '–': '-', '—': '-', '…': '...', '²': '2', '³': '3', 'Ø': 'D', '≈': '~',
           '▸': '>', '’': "'", '‘': "'", '“': '"', '”': '"', 'µ': 'u', '−': '-'}
    return ''.join(rep.get(ch, ch) for ch in str(text)).encode('ascii', 'replace').decode('ascii')


def _put(img, text, org, scale, col, cv2):
    cv2.putText(img, _ascii(text), (int(org[0]), int(org[1])), cv2.FONT_HERSHEY_SIMPLEX, scale, col, 1, cv2.LINE_AA)


def sheet_png(res: BracketResult, project=None) -> bytes:
    """A PNG sheet: per bracket a row of three views (top, side, three-quarter) with the base size, the height and
    the plate's angle drawn as dimensions, and its figures; cameras without a bracket listed with the reason."""
    import cv2
    VW, VH, TW, HEAD = 330, 270, 310, 34
    W = 3 * VW + TW + 20
    rows = [b for b in res.brackets if b.generated]
    other = [b for b in res.brackets if not b.generated]
    H = 64 + len(rows) * (VH + HEAD + 10) + 22 * len(other) + 30
    img = np.full((H, W, 3), 255, np.uint8)
    ink, mute, dim = (20, 20, 20), (120, 118, 112), (40, 40, 200)
    _put(img, f'Mount brackets - {getattr(project, "name", "")} - {res.variant}', (12, 28), 0.7, ink, cv2)
    _put(img, f'{res.settings.material} - {res.settings.pattern_label()} - millimetres - grid '
              f'{res.settings.res_mm:g} mm', (12, 50), 0.45, mute, cv2)
    y = 60
    for b in rows:
        p = b.params
        what = 'tube clamp' if b.kind == 'clamp' else FIXING_LABEL.get(p['base'].get('fixing'), '')
        _put(img, f'{b.tag}  {b.name} - {b.surface} - {b.face} face - {what}', (12, y + 22), 0.5, ink, cv2)
        y += HEAD
        mesh = b.display or display_mesh(b, 30000)
        V = mesh[0]
        HV, _ = housing_mesh(b)
        allv = np.vstack([V, HV])
        centre = (allv.min(axis=0) + allv.max(axis=0)) / 2
        vs = views(b)

        def fit(basis):
            ext = np.abs((allv - centre) @ basis.T)[:, :2].max(axis=0) * 2 + 1e-6
            return 0.76 * min(VW / ext[0], VH / ext[1])
        flat_scale = min(fit(vs[0][1]), fit(vs[1][1]))           # top and side at one scale
        for k, (name, basis) in enumerate(vs):
            sc = flat_scale if k < 2 else fit(basis)
            pic, _, _ = picture(b, basis, (VW, VH), sc, centre, mesh=mesh)
            x0 = 10 + k * VW
            img[y:y + VH, x0:x0 + VW] = cv2.cvtColor(pic, cv2.COLOR_RGB2BGR)
            cv2.rectangle(img, (x0, y), (x0 + VW - 1, y + VH - 1), (225, 224, 217), 1)
            _put(img, name, (x0 + 6, y + 16), 0.42, mute, cv2)

            def px(P, basis=basis, sc=sc, x0=x0):
                q = (np.asarray(P, float) - centre) @ basis.T
                return x0 + VW / 2 + q[0] * sc, y + VH / 2 - q[1] * sc
            bx0, bx1, by0, by1 = p['base']['rect']
            if name == 'Top':
                a, c = px([bx0, by0, 0]), px([bx1, by0, 0])
                _arrow(img, (a[0], a[1] + 10), (c[0], c[1] + 10), dim, f'{bx1 - bx0:.1f}', cv2)
                a, c = px([bx1, by0, 0]), px([bx1, by1, 0])
                _arrow(img, (a[0] + 10, a[1]), (c[0] + 10, c[1]), dim, f'{by1 - by0:.1f}', cv2)
            elif name == 'Side':
                zt = p['height_mm']
                zb = -2 * b.params['clamp']['tube_radius_mm'] if b.kind == 'clamp' else 0.0
                xr = float(max(V[:, 0].max(), bx1))
                a, c = px([xr, 0, zb]), px([xr, 0, zt])
                _arrow(img, (a[0] + 10, a[1]), (c[0] + 10, c[1]), dim, f'{zt - zb:.1f}', cv2)
                g0, g1 = px([bx0, 0, 0]), px([bx1, 0, 0])
                cv2.line(img, (int(g0[0]) - 12, int(g0[1])), (int(g1[0]) + 12, int(g1[1])), mute, 1, cv2.LINE_AA)
                q = px(np.array(p['plate']['face_centre']))
                _put(img, f'plate {p["plate_angle_deg"]:.1f} deg', (q[0] - 40, q[1] - 12), 0.42, dim, cv2)
        tx = 10 + 3 * VW + 12
        lines = [f'Base {p["base"]["length_mm"]:.1f} x {p["base"]["width_mm"]:.1f} x {p["base"]["thickness_mm"]:.1f}',
                 f'Height {p["height_mm"]:.1f} (optical centre {p["oc_height_mm"]:.1f})',
                 f'Plate {p["plate_angle_deg"]:.1f} deg to the base',
                 f'Standoff {b.gap_mm:.1f}',
                 f'Lever arm {p["lever_mm"]:.0f}',
                 f'Image blocked {p.get("fov_blocked_pct", 0.0):.1f} %',
                 f'{b.mesh.get("volume_mm3", 0) / 1000:.1f} cm3, {b.mesh.get("mass_g", 0):.0f} g solid',
                 f'{b.mesh.get("triangles", 0):,} triangles']
        if 'measured' in p:
            lines.append(f'Measured: OC {p["measured"]["oc_error_mm"]:.2f}, plate '
                         f'{p["measured"]["plate_angle_error_deg"]:.2f} deg')
        for i, t in enumerate(lines):
            _put(img, t, (tx, y + 18 + 19 * i), 0.42, ink, cv2)
        yy = y + 24 + 19 * len(lines)
        for c in [c for c in b.checks if c['severity'] != 'good'][:3]:
            words, line, out = _ascii(c['text']).split(), '! ', []
            for wd in words:
                if len(line) + len(wd) > 46:
                    out.append(line)
                    line = '  '
                line += wd + ' '
            out.append(line)
            for t in out[:3]:
                if yy < y + VH - 4:
                    _put(img, t, (tx, yy), 0.38, (30, 110, 210), cv2)
                yy += 15
            yy += 3
        y += VH + 10
    for b in other:
        _put(img, f'{b.tag}: {b.message}'[:160], (12, y + 16), 0.42, mute, cv2)
        y += 22
    _put(img, 'Millimetres, bracket frame: z is the surface normal; the side view looks along the plate\'s edge.',
         (12, H - 10), 0.38, mute, cv2)
    ok, buf = cv2.imencode('.png', img)
    return buf.tobytes() if ok else b''
