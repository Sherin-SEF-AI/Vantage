"""Extrusion framing for the roof rig: T-slot aluminium members, mount plates, fixings and feet synthesised from the
rig parts (core.rig) and the sensors standing on them, with a cut list, a BOM, CNC plate drawings (DXF), a watertight
STL per part and a STEP AP214 assembly.

Frame and units. The vehicle frame (ISO 8855: x forward, y left, z up, origin at the rear-axle centre on the ground).
Positions in metres in the model, millimetres in every file a workshop reads (cut list, BOM, DXF, STL, STEP).

What is synthesised (from the visible rig parts on the roof and the variant's devices above the roof)
- Crossbars: one member per rig crossbar (kind 'bar'), of the main profile, its length and position kept, its
  underside on the bar's underside. With no crossbar on the roof but parts or devices to carry, two are placed as
  core.rig's roof-rack preset places them.
- Feet: two per crossbar, FEET_INSET_M in from its ends (as core.thermo_vib takes a rack's feet), each a CNC foot plate
  on a foot block as tall as the gap to the roof under it (the roof's height probed on the body mesh there) and a
  roof-rail clamp or fix-point foot bought for the vehicle.
- Rails: with two crossbars or more, two members along x stacked on the crossbars, at the edges of a rack platform
  (a thin box on the bars, which the rails replace) or just inboard of the feet. They make the ladder rigid.
- Posts: a sensor mast (kind 'cylinder') becomes a vertical member of the post profile standing on what is under it
  (a rail, a crossbar or a new cross member between the rails); a mast head (a box resting on the mast) becomes a
  top plate of its footprint.
- Device mounts: every camera, lidar, radar, GNSS antenna and IMU whose base is above the roof gets a mount plate on
  the member under it; a device higher than that member gets a standoff post; a device with no member under it gets
  a cross member between the rails (or an arm off the nearest crossbar in front of or behind the rails). Enclosures
  (boxes on the rack) are carried the same way. A device the frame cannot reach is listed (mount it on the body).
- Fixings: 'brackets' joins every butt and stacked joint with corner brackets (two per tee, two per stacked
  crossing, four at a post's foot), each with two bolts and two T-nuts, every member cut square; 'anchors' uses
  anchor fasteners (one per tee end, a counterbore in the butting member and an access hole in the through member)
  and end-tapped posts. Every plate is held by two bolts and T-nuts per member it crosses.

Profiles (PROFILES): real catalogue profiles, every figure cited. The FE model (core.fea) uses their area and
second moments; the drawn solids (STL, STEP) are the profile's outline with simplified T-slots (opening and undercut
as the catalogue drawing, no centre bore or inner cavities), so a solid's volume is not the profile's area × length.

Files: the cut list groups identical cuts (profile, length to 0.5 mm, end machining); the stock plan packs the cuts
into the profile's stock bars, first-fit decreasing with the saw kerf. STEP: written directly as AP214
(AUTOMOTIVE_DESIGN) with one MANIFOLD_SOLID_BREP per part (planar faces only: the outline extruded, holes as
24-gons) in an assembly; when gmsh (with OpenCASCADE) is installed it reads the file back and its volumes are checked
against ours (verify_step). DXF: R12 ASCII, millimetres, one file per plate and a sheet of all plates (outline on
layer OUTLINE, holes on HOLES, the label on TEXT).

Out of scope: welded frames, bending of the plates under their bolts, the corner brackets' own stiffness (core.fea
ties the members rigidly), slot covers and cable routing, the roof-rail clamps' design (bought for the vehicle), and
the sensors' own hole patterns (typical patterns: check the sensor's drawing). Typical figures are typical, not this
rig's.
"""
from __future__ import annotations

import csv
import io
import json
import math
import os
from dataclasses import dataclass, field

import numpy as np

SCHEMA = 'vantage.framing/1'
FRAME = 'vehicle (ISO 8855, rear axle): x forward, y left, z up'
SEVERITIES = ('critical', 'serious', 'warning', 'note', 'good')
IN = 0.0254
LB = 0.45359237
RHO_AL = 2700.0                 # kg/m³, wrought aluminium (ASM, 6061-T6 and 6063-T6)
FEET_INSET_M = 0.06             # as core.thermo_vib: a rack's feet sit this far in from the bar's ends
SEAT_M = 0.30                   # a bar this close above the roof is a roof crossbar (core.rig)
PLATFORM_MAX_H = 0.06           # a box this thin resting on the crossbars is a rack platform (the rails replace it)
CARRY_GAP_M = 0.02              # a device base within this of a member's top (plus the plate) sits on a plate on it
MIN_FOOT_M = 0.02               # the shortest foot block
CAM_BASE_M = 0.025              # a camera's plate this far under its optical centre (typical small module)
STEP_HOLE_SEGMENTS = 24
CAM_COLOUR = (0.35, 0.36, 0.38)
FRAME_PREFIX = 'Frame · '      # rig parts added from a frame (to_rig_parts): never framed again

# --------------------------------------------------------------------------- catalogue profiles
# b: the narrow side, d: the deep side (m); i_d: second moment for bending in the deep direction (about the axis
# along b), i_b: in the narrow direction (m⁴); area m²; kg_m; E Pa. A profile is drawn with its deep side vertical.
_SRC_8020 = ('80/20 Inc., "T-slot Profiles" catalogue p. 2-16, 1.00" x 1.00" T-slotted profile (part 1010): '
             'IX = IY = 0.0442 in⁴, area 0.435 in², 0.0424 lb/in, E 10,200,000 psi, 6005A-T61, slot .256", stock '
             '120"/242" (datasheet distributed by RS Components, read 2026-11-11)')
_SRC_IPS = ('Industrial Profile Systems (IPS), Inch Profiles catalogue (2025), 1" series p. 59-60 and 1.5" series '
            'p. 53/56: Ix, Iy (cm⁴), section (cm²), weight (kg/m); stock 6 m (read 2026-11-11). 80/20-compatible '
            'fractional profiles. p. 60 prints 3.16 cm² for 12-1020 where p. 59 prints 5.16 cm²: 5.16 is taken '
            '(1.41 kg/m of aluminium is 5.2 cm²)')
_SRC_FT = ('Framing Technology Inc., metric profile pages (framingtech.com, read 2026-11-11): weight kg/m, surface '
           '(section) area cm², I(x), I(y) cm⁴, 6063-T6, stock length')
_SRC_E = 'ASM Aerospace Specification Metals, aluminium 6061-T6 / 6063-T6 data sheets: E 68.9 GPa, ν 0.33, 2.70 g/cm³'


@dataclass(frozen=True)
class Profile:
    id: str
    label: str
    maker: str
    part: str
    b: float
    d: float
    area: float
    i_d: float
    i_b: float
    kg_m: float
    E: float
    alloy: str
    slot: float
    bolt: str
    clearance: float          # m: a plate's hole for the bolt
    stock_m: tuple
    source: str
    nu: float = 0.33
    bracket_kg: float = 0.05  # one corner bracket with its two bolts and T-nuts (typical)

    @property
    def rho(self) -> float:
        """Density the catalogue's mass and area imply (kg/m³)."""
        return self.kg_m / self.area


def _p(id_, label, maker, part, b, d, area, i_d, i_b, kg_m, E, alloy, slot, bolt, clr, stock, src, bk):
    return Profile(id_, label, maker, part, b, d, area, i_d, i_b, kg_m, E, alloy, slot, bolt, clr, stock, src,
                   bracket_kg=bk)


_CM2, _CM4 = 1e-4, 1e-8
PROFILES = {p.id: p for p in (
    _p('in-1010', '1010 · 1 × 1 in', '80/20', '1010', IN, IN, 0.435 * IN ** 2, 0.0442 * IN ** 4, 0.0442 * IN ** 4,
       0.0424 * LB / IN, 10.2e6 * 6894.757, '6005A-T61', 0.256 * IN, '1/4-20', 0.0071, (120 * IN, 242 * IN),
       _SRC_8020, 0.03),
    _p('in-1020', '1020 · 1 × 2 in', 'IPS', '12-1020', IN, 2 * IN, 5.16 * _CM2, 13.03 * _CM4, 3.62 * _CM4, 1.41,
       68.9e9, '6063-T6 (typical)', 0.26 * IN, '1/4-20', 0.0071, (6.0,), _SRC_IPS + '; ' + _SRC_E, 0.03),
    _p('in-2020', '2020 · 2 × 2 in', 'IPS', '12-2020', 2 * IN, 2 * IN, 8.13 * _CM2, 23.68 * _CM4, 23.68 * _CM4, 2.21,
       68.9e9, '6063-T6 (typical)', 0.26 * IN, '1/4-20', 0.0071, (6.0,), _SRC_IPS + '; ' + _SRC_E, 0.05),
    _p('in-1515', '1515 · 1.5 × 1.5 in', 'IPS', '10-1515', 1.5 * IN, 1.5 * IN, 6.19 * _CM2, 8.49 * _CM4, 8.49 * _CM4,
       1.70, 68.9e9, '6063-T6 (typical)', 0.32 * IN, '5/16-18', 0.0087, (6.0,), _SRC_IPS + '; ' + _SRC_E, 0.08),
    _p('in-1530', '1530 · 1.5 × 3 in', 'IPS', '10-1530', 1.5 * IN, 3 * IN, 10.38 * _CM2, 61.85 * _CM4, 15.15 * _CM4,
       2.81, 68.9e9, '6063-T6 (typical)', 0.32 * IN, '5/16-18', 0.0087, (6.0,), _SRC_IPS + '; ' + _SRC_E, 0.08),
    _p('mm-2020', '20 × 20 mm · slot 5', 'Framing Technology', '20X20-05', 0.020, 0.020, 1.82 * _CM2, 0.74 * _CM4,
       0.74 * _CM4, 0.491, 68.9e9, '6063-T6', 0.005, 'M4', 0.0045, (3.0,), _SRC_FT + '; ' + _SRC_E, 0.015),
    _p('mm-4040', '40 × 40 mm · slot 8', 'Framing Technology', '40X40-08', 0.040, 0.040, 6.49 * _CM2, 9.18 * _CM4,
       9.18 * _CM4, 1.75, 68.9e9, '6063-T6', 0.008, 'M8', 0.009, (6.0,), _SRC_FT + '; ' + _SRC_E, 0.09),
    _p('mm-4080', '40 × 80 mm · slot 8', 'Framing Technology', '40X80-08', 0.040, 0.080, 11.33 * _CM2, 70.16 * _CM4,
       16.94 * _CM4, 3.06, 68.9e9, '6063-T6', 0.008, 'M8', 0.009, (6.0,), _SRC_FT + '; ' + _SRC_E, 0.09),
)}
PLATE_MATERIAL = {'label': 'Aluminium plate 6061-T6', 'rho': RHO_AL, 'E': 68.9e9, 'nu': 0.33, 'source': _SRC_E}
DEVICE_SCREW = {'lidar': 'M8 (typical: check the sensor drawing)', 'radar': 'M6 (typical)', 'gnss': '5/8-11 '
                'antenna stud or M6 (typical)', 'imu': 'M4 (typical)', 'camera': 'M4 (typical)',
                'payload': 'M6 (typical)'}
DEVICE_HOLE_M = {'lidar': 0.009, 'radar': 0.0066, 'gnss': 0.0066, 'imu': 0.0045, 'camera': 0.0045, 'payload': 0.0066}
FOOT_CLAMP_KG = 0.25            # a roof-rail clamp or fix-point foot (typical)
FASTENER_KG = 0.008             # one bolt with its T-nut (typical)

CHOICES = {'main_profile': tuple(PROFILES), 'post_profile': tuple(PROFILES), 'joint': ('brackets', 'anchors'),
           'feet': ('rails', 'fixpoints')}
LIMITS = {'plate_mm': (3.0, 20.0), 'plate_margin_mm': (0.0, 50.0), 'kerf_mm': (0.0, 10.0), 'stock_m': (0.0, 12.0),
          'rail_inset_mm': (0.0, 400.0), 'min_plate_mm': (40.0, 300.0)}
JOINT_LABEL = {'brackets': 'Corner brackets', 'anchors': 'Anchor fasteners'}
FEET_LABEL = {'rails': 'Roof-rail clamps', 'fixpoints': 'Fix-point feet'}
ASSUMPTIONS = ('Members are catalogue T-slot profiles (figures cited per profile); plates 6061-T6; corner brackets, '
               'fasteners and feet typical. The frame follows the rig parts and device positions as designed; the '
               'roof\'s height under each foot is probed on the body mesh. Drawn solids are the profile outline with '
               'simplified T-slots. Sensor hole patterns are typical: check each sensor\'s drawing.')


def _num(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


@dataclass
class FramingSettings:
    main_profile: str = 'mm-4040'       # crossbars, rails, cross members, arms
    post_profile: str = 'mm-4040'       # posts and standoffs
    joint: str = 'brackets'             # brackets | anchors
    feet: str = 'rails'                 # rails (roof-rail clamps) | fixpoints
    plate_mm: float = 6.0               # mount plate thickness
    plate_margin_mm: float = 10.0       # past the device's footprint
    min_plate_mm: float = 80.0          # the smallest plate side (two bolts into the slot)
    kerf_mm: float = 3.0
    stock_m: float = 0.0                # 0 = the profile's own stock length
    rail_inset_mm: float = 0.0          # rails this far in from the platform's edges (or from the feet)
    frame_cameras: bool = True          # cameras above the roof get plates too
    replaced: list = field(default_factory=list)    # rig parts the frame stands for once added to the rig (they
                                                    # are hidden then, and still framed)

    def to_dict(self):
        return {k: getattr(self, k) for k in self.__dataclass_fields__}

    @staticmethod
    def from_dict(d) -> 'FramingSettings':
        """Tolerant: unknown keys dropped, numbers clamped to LIMITS, bad values fall back to the defaults."""
        s = FramingSettings()
        d = d if isinstance(d, dict) else {}
        for k, (lo, hi) in LIMITS.items():
            v = _num(d.get(k))
            if v is not None:
                setattr(s, k, min(max(v, lo), hi))
        for k, opts in CHOICES.items():
            v = d.get(k)
            if isinstance(v, str) and v.strip() in opts:
                setattr(s, k, v.strip())
        v = d.get('frame_cameras')
        if isinstance(v, bool):
            s.frame_cameras = v
        v = d.get('replaced')
        if isinstance(v, (list, tuple)):
            s.replaced = [str(x) for x in v if isinstance(x, str) and x][:500]
        return s


def settings_of(project) -> FramingSettings:
    return FramingSettings.from_dict(getattr(project.analysis, 'framing', None))


# =========================================================================== the frame's parts
@dataclass
class Member:
    id: str
    name: str
    role: str                 # crossbar | rail | cross member | arm | post | standoff
    profile: str
    p0: np.ndarray            # centreline ends (m)
    p1: np.ndarray
    up: np.ndarray            # the profile's deep direction
    from_uid: str = ''        # the rig part it replaces
    end0: str = 'square cut'
    end1: str = 'square cut'
    access_holes: int = 0     # anchors: holes in this through member

    @property
    def length(self) -> float:
        return float(np.linalg.norm(self.p1 - self.p0))

    @property
    def axis(self) -> np.ndarray:
        return (self.p1 - self.p0) / max(self.length, 1e-12)

    @property
    def prof(self) -> Profile:
        return PROFILES[self.profile]

    def frame(self):
        """(origin = the start of the centreline, ex = the narrow side, ey = the deep side, ez = the axis)."""
        ez = self.axis
        ey = self.up - (self.up @ ez) * ez
        ey /= max(np.linalg.norm(ey), 1e-12)
        return self.p0, np.cross(ey, ez), ey, ez

    @property
    def top(self) -> float:
        _, ex, ey, _ = self.frame()
        pr = self.prof
        return float(max(self.p0[2], self.p1[2]) + abs(ex[2]) * pr.b / 2 + abs(ey[2]) * pr.d / 2)

    @property
    def mass(self) -> float:
        return self.prof.kg_m * self.length

    def local(self, P):
        """(along the axis from p0, across along ex, across along ey) of the point P."""
        o, ex, ey, ez = self.frame()
        d = np.asarray(P, float) - o
        return float(d @ ez), float(d @ ex), float(d @ ey)


@dataclass
class Plate:
    id: str
    name: str
    role: str                 # sensor mount | camera mount | payload mount | top plate | foot plate
    centre: np.ndarray        # the centre of its underside (m)
    lx: float                 # along its own x (m)
    ly: float
    t: float
    yaw: float                # degrees about z
    holes: list = field(default_factory=list)   # [(x, y, diameter, purpose)] in the plate's frame, m
    device_uid: str = ''
    device_tag: str = ''
    on: list = field(default_factory=list)      # member ids under it

    @property
    def mass(self) -> float:
        area = self.lx * self.ly - sum(math.pi * h[2] ** 2 / 4 for h in self.holes)
        return max(area, 0.0) * self.t * PLATE_MATERIAL['rho']

    def frame(self):
        a = math.radians(self.yaw)
        ex = np.array([math.cos(a), math.sin(a), 0.0])
        ey = np.array([-math.sin(a), math.cos(a), 0.0])
        return self.centre, ex, ey, np.array([0.0, 0.0, 1.0])


@dataclass
class Foot:
    id: str
    name: str
    member: str               # the crossbar it carries
    centre: np.ndarray        # the centre of the foot block's underside, on the roof
    height: float             # block height up to the crossbar's underside
    lx: float = 0.08
    ly: float = 0.06
    yaw: float = 0.0
    roof_z: float = 0.0


@dataclass
class FramingResult:
    settings: FramingSettings
    variant: str
    variant_uid: str
    roof: dict
    members: list
    plates: list
    feet: list
    joints: list              # [{'a', 'b', 'kind', 'point'}]
    mounts: list              # per device: {'uid', 'tag', 'kind', 'status', 'plate', 'member', 'standoff', 'gap_mm'}
    unframed: list
    cut_list: list
    stock: list
    bom: list
    mass: dict
    findings: list = field(default_factory=list)
    notes: list = field(default_factory=list)

    def member(self, mid):
        return next((m for m in self.members if m.id == mid), None)

    def worst(self) -> str:
        sev = [f['severity'] for f in self.findings]
        return next((s for s in SEVERITIES if s in sev), 'good')


# =========================================================================== synthesis helpers
def _yaw_dir(yaw):
    a = math.radians(yaw)
    return np.array([math.cos(a), math.sin(a), 0.0])


def _roof_z(mesh, x, y, top, r=0.06) -> float:
    """Height of the roof at (x, y): the highest upward face of the body within r of it (else the roof's top)."""
    if mesh is None or getattr(mesh, 'n_tris', 0) == 0:
        return top
    c, n = mesh.centroids, mesh.normals
    near = (np.abs(c[:, 0] - x) < r) & (np.abs(c[:, 1] - y) < r) & (n[:, 2] > 0.3) & (c[:, 2] > top - 0.4)
    if not near.any():
        return top
    return float(min(c[near, 2].max(), top))


def _bottom(p) -> float:
    return p.z - (p.radius if p.kind == 'bar' else p.height / 2)


def _top(p) -> float:
    return p.z + (p.radius if p.kind == 'bar' else p.height / 2)


def _xy_overlap(a, b) -> bool:
    from .rig import footprint
    fa, fb = footprint(a), footprint(b)
    return bool(np.all(fa.min(axis=0) <= fb.max(axis=0) + 1e-3) and np.all(fb.min(axis=0) <= fa.max(axis=0) + 1e-3))


def _devices(project, v, rf, st) -> list[dict]:
    """The variant's devices whose base is above the roof: {'uid', 'tag', 'kind', 'P', 'base', 'foot' (lx, ly),
    'round', 'mass', 'obj'}; enclosures (thick boxes on the rack) as payloads come from the caller."""
    from . import sensors as SE
    from . import suspension as SU
    comps = {c['uid']: c for c in SU.rig_components(project, SU.settings_for(project), v)}
    top = rf['top']
    out = []
    for s in v.sensors:
        if not s.visible or not all(math.isfinite(x) for x in (s.x, s.y, s.z, s.yaw, s.pitch, s.roll)):
            continue
        m = project.sensor_model_for(s)
        c, half, R, rnd = SE.body_box(s, m)
        base = s.pos() - R @ np.array([0.0, 0.0, m.optical_z_mm / 1000.0])
        out.append({'uid': s.uid, 'tag': s.tag, 'kind': s.kind, 'P': s.pos(), 'base': base,
                    'foot': (2 * half[0], 2 * half[1]), 'round': bool(rnd), 'yaw': float(SE.heading(s)),
                    'mass': float(comps.get(s.uid, {}).get('mass', 1.0)), 'centre': c, 'half': half, 'R': R,
                    'obj': s})
    if st.frame_cameras:
        from .mathutil import rot_zyx
        for cam in v.cameras:
            if not cam.visible or not all(math.isfinite(x) for x in (cam.x, cam.y, cam.z)):
                continue
            mod = project.module_for(cam)
            try:
                w, h, ln = (max(float(x), 10.0) / 1000.0 for x in mod.body_mm())
            except Exception:                       # a module without a usable size: a typical small camera
                w, h, ln = 0.03, 0.03, 0.04
            P = cam.pos()
            base = P - np.array([0.0, 0.0, CAM_BASE_M + max(cam.standoff_mm, 0.0) / 1000.0])
            out.append({'uid': cam.uid, 'tag': cam.tag, 'kind': 'camera', 'P': P, 'base': base,
                        'foot': (max(ln, w), max(ln, w)), 'round': False, 'yaw': float(cam.yaw),
                        'mass': float(comps.get(cam.uid, {}).get('mass', 0.08)), 'centre': P.copy(),
                        'half': np.array([ln, w, h]) / 2, 'R': rot_zyx(cam.yaw, cam.pitch, cam.roll), 'obj': cam})
    return [d for d in out if d['base'][2] > top - 0.005 and
            rf['x_rear'] - 0.6 <= d['base'][0] <= rf['x_front'] + 0.6 and abs(d['base'][1]) <= rf['half_width'] + 0.6]


class _Builder:
    def __init__(self, st: FramingSettings):
        self.st = st
        self.members: list[Member] = []
        self.plates: list[Plate] = []
        self.feet: list[Foot] = []
        self.joints: list[dict] = []

    def add_member(self, name, role, profile, p0, p1, up, from_uid='') -> Member:
        m = Member(f'M{len(self.members) + 1}', name, role, profile, np.asarray(p0, float), np.asarray(p1, float),
                   np.asarray(up, float), from_uid)
        self.members.append(m)
        return m

    def add_plate(self, **kw) -> Plate:
        p = Plate(f'P{len(self.plates) + 1}', **kw)
        self.plates.append(p)
        return p

    def joint(self, a, b, kind, point):
        self.joints.append({'a': a, 'b': b, 'kind': kind, 'point': [float(x) for x in point]})

    def horizontals(self):
        return [m for m in self.members if abs(m.axis[2]) < 0.3]

    def support_under(self, B, reach=0.0):
        """The highest horizontal member whose top is at or below B and whose footprint reaches under B (laterally
        within half its width plus reach), or a post's top plate under B: (member, top z) or (None, None)."""
        best, bz = None, -math.inf
        for m in self.members:
            pr = m.prof
            if abs(m.axis[2]) < 0.3:
                s, u, _ = m.local(B)
                if not (-0.002 <= s <= m.length + 0.002 and abs(u) <= pr.b / 2 + reach):
                    continue
                t = m.top
            else:                                     # a post: only straight above its top
                q = m.p1 if m.p1[2] >= m.p0[2] else m.p0
                if math.hypot(B[0] - q[0], B[1] - q[1]) > max(pr.d, pr.b) / 2 + reach:
                    continue
                t = float(q[2])
            if t <= B[2] + 0.01 and t > bz:
                best, bz = m, t
        return best, (bz if best is not None else None)


def _member_bolts(pr: Profile, length_along: float):
    """Two plate holes into a member's slot, along its axis (offsets m)."""
    off = max(min(length_along / 2 - 0.015, 0.06), 0.012)
    return [(-off, pr.clearance), (off, pr.clearance)]


def _plate_for(bld: _Builder, st: FramingSettings, dev: dict, member: Member, z_top: float, role: str) -> Plate:
    """A mount plate under a device, its x along the member's axis, holes for the device and into the member."""
    pr = member.prof
    yaw = math.degrees(math.atan2(member.axis[1], member.axis[0])) if abs(member.axis[2]) < 0.3 else dev.get('yaw', 0.0)
    m = st.plate_margin_mm / 1000.0
    fx, fy = dev['foot']
    # the footprint turned into the plate's frame: its bounding box
    a = math.radians(dev.get('yaw', 0.0) - yaw)
    ex = abs(math.cos(a)) * fx + abs(math.sin(a)) * fy
    ey = abs(math.sin(a)) * fx + abs(math.cos(a)) * fy
    if dev.get('round'):
        ex = ey = max(fx, fy)
    lo = st.min_plate_mm / 1000.0
    lx, ly = max(ex + 2 * m, lo), max(ey + 2 * m, lo, pr.b + 0.02)
    holes = []
    kind = dev['kind']
    dh = DEVICE_HOLE_M.get(kind, 0.0066)
    if dev.get('round'):
        r = 0.35 * max(fx, fy)
        holes += [(r * math.cos(math.radians(45 + 90 * k)), r * math.sin(math.radians(45 + 90 * k)), dh, 'device')
                  for k in range(4)]
    else:
        ix, iy = max(ex / 2 - 0.008, dh), max(ey / 2 - 0.008, dh)
        holes += [(sx * ix, sy * iy, dh, 'device') for sx in (-1, 1) for sy in (-1, 1)]
    # into the member's slot along its axis (a post: into its top end, the centre bore tapped)
    if abs(member.axis[2]) < 0.3:
        holes += [(off, 0.0, d, 'member') for off, d in _member_bolts(pr, lx)]
    else:
        holes.append((0.0, 0.0, pr.clearance, 'member'))
    # drop device holes that cut into a member hole
    keep = []
    for h in holes:
        if h[3] == 'device' and any(math.hypot(h[0] - g[0], h[1] - g[1]) < (h[2] + g[2]) / 2 + 0.003
                                    for g in holes if g[3] == 'member'):
            continue
        keep.append(h)
    c = np.array([dev['base'][0], dev['base'][1], z_top])
    return bld.add_plate(name=f'{dev["tag"]} plate', role=role, centre=c, lx=lx, ly=ly, t=st.plate_mm / 1000.0,
                         yaw=yaw, holes=keep, device_uid=dev['uid'], device_tag=dev['tag'], on=[member.id])


# =========================================================================== synthesis
def synthesise(project, settings: FramingSettings | None = None, variant=None) -> FramingResult:
    """The frame for the project's rig and the variant's devices above the roof (cheap: no geometry beyond a few
    probes of the body mesh)."""
    from . import thermo_vib as TV
    from .rig import preset
    st = settings if isinstance(settings, FramingSettings) else (
        FramingSettings.from_dict(settings) if isinstance(settings, dict) else settings_of(project))
    v = variant or project.variant
    rf = TV.roof_box(project)
    top = rf['top']
    mesh = project.vehicle.mesh
    main, post = PROFILES[st.main_profile], PROFILES[st.post_profile]
    bld = _Builder(st)
    notes, findings = [], []
    parts = [p for p in getattr(project, 'rig', []) or [] if (p.visible or p.uid in st.replaced)
             and not p.name.startswith(FRAME_PREFIX)]
    on_roof = [p for p in parts if _bottom(p) < top + 1.5 and rf['x_rear'] - 0.6 <= p.x <= rf['x_front'] + 0.6
               and abs(p.y) <= rf['half_width'] + 0.6]
    devs = _devices(project, v, rf, st)
    bars = [p for p in on_roof if p.kind == 'bar' and _bottom(p) < top + SEAT_M]
    used = set()
    # ---------------------------------------------------------------- crossbars and feet
    lifted = [d for d in devs if d['base'][2] > top + MIN_FOOT_M]       # a device standing on the roof itself
    if not bars and (on_roof or lifted):                                  # (a magnet mount) needs no frame
        mk = preset('roof_rack', mesh)
        bars = [p for p in mk if p.kind == 'bar']
        for p in bars:
            p.uid = ''
        notes.append('No crossbar on the roof: two placed as the roof-rack preset places them.')
    crossbars = []
    for i, p in enumerate(sorted(bars, key=lambda q: -q.x)):
        ax = np.array([-math.sin(math.radians(p.yaw)), math.cos(math.radians(p.yaw)), 0.0])
        zb = max(_bottom(p), top + MIN_FOOT_M)
        c = np.array([p.x, p.y, zb + main.d / 2])
        m = bld.add_member(f'Crossbar {i + 1}', 'crossbar', main.id, c - ax * p.length / 2, c + ax * p.length / 2,
                           (0, 0, 1), p.uid)
        crossbars.append(m)
        used.add(p.uid)
        for sgn in (-1, 1):
            q = c + ax * sgn * (p.length / 2 - FEET_INSET_M)
            rz = _roof_z(mesh, q[0], q[1], top)
            h = max(zb - rz, MIN_FOOT_M)
            f = Foot(f'F{len(bld.feet) + 1}', f'{m.name} · {"left" if (ax * sgn)[1] > 0 else "right"} foot', m.id,
                     np.array([q[0], q[1], zb - h]), h, 0.08, max(main.b + 0.02, 0.06), p.yaw, rz)
            bld.feet.append(f)
            bld.joint(m.id, f.id, 'foot', q)
    # ---------------------------------------------------------------- rails
    rails = []
    if len(crossbars) >= 2:
        plat = [p for p in on_roof if p.kind == 'box' and p.height <= PLATFORM_MAX_H and any(
            abs(_bottom(p) - _top(b)) < 0.03 and _xy_overlap(p, b) for b in bars if b.uid)]
        xs = [float((m.p0[0] + m.p1[0]) / 2) for m in crossbars]
        cb_top = max(m.top for m in crossbars)
        inset = st.rail_inset_mm / 1000.0
        if plat:
            pl = plat[0]
            used.add(pl.uid)
            hy = pl.width / 2 - main.b / 2 - inset
            x0, x1 = pl.x - pl.length / 2, pl.x + pl.length / 2
            fu = pl.uid
        else:
            span = min(m.length for m in crossbars) / 2 - FEET_INSET_M - main.b
            hy = span - inset
            x0, x1 = min(xs) - main.b, max(xs) + main.b
            fu = ''
        hy = max(hy, main.b)
        z = cb_top + main.d / 2
        for sgn, side in ((1, 'left'), (-1, 'right')):
            r = bld.add_member(f'Rail {side}', 'rail', main.id, (x0, sgn * hy, z), (x1, sgn * hy, z), (0, 0, 1), fu)
            rails.append(r)
            for cb, xc in zip(crossbars, xs):
                bld.joint(r.id, cb.id, 'stacked', (xc, sgn * hy, cb_top))
    rail_y = (min(r.p0[1] for r in rails), max(r.p0[1] for r in rails)) if rails else None
    rail_x = (min(r.p0[0] for r in rails), max(r.p1[0] for r in rails)) if rails else None

    def cross_member(x, why):
        """A member along y between the rails at x, butting into them (or the existing one near x)."""
        for m in bld.members:
            if m.role == 'cross member' and abs(m.p0[0] - x) < 0.05:
                return m
        r = rails[0]
        z = float(r.p0[2])
        y0, y1 = rail_y[0] + main.b / 2, rail_y[1] - main.b / 2
        m = bld.add_member(f'Cross member at x {x:.2f} m', 'cross member', main.id, (x, y0, z), (x, y1, z),
                           (0, 0, 1))
        for rr in rails:
            bld.joint(m.id, rr.id, 'tee', (x, float(rr.p0[1]), z))
        notes.append(f'{m.name}: carries {why}.')
        return m

    def arm_to(B, why):
        """A member along x from the nearest crossbar's side to under B (a device in front of or behind the rails)."""
        if not crossbars:
            return None
        cb = min(crossbars, key=lambda m: abs(m.p0[0] - B[0]))
        s, u, _ = cb.local(B)
        if not (0.0 <= s <= cb.length):
            return None
        x_cb = float(cb.p0[0])
        sgn = 1.0 if B[0] > x_cb else -1.0
        z = float((cb.p0[2] + cb.p1[2]) / 2)
        start = np.array([x_cb + sgn * main.b / 2, B[1], z])
        end = np.array([B[0] + sgn * max(st.min_plate_mm / 2000.0, 0.03), B[1], z])
        if abs(end[0] - start[0]) < 0.02:
            return None
        m = bld.add_member(f'Arm to {why}', 'arm', main.id, start if sgn > 0 else end, end if sgn > 0 else start,
                           (0, 0, 1))
        bld.joint(m.id, cb.id, 'tee', start)
        return m

    # ---------------------------------------------------------------- masts → posts
    masts = [p for p in on_roof if p.kind == 'cylinder']
    payloads = []
    for p in masts:
        used.add(p.uid)
        B = np.array([p.x, p.y, _bottom(p)])
        sup, zt = bld.support_under(B + [0, 0, 0.02], 0.0)
        if sup is None and rails and rail_x[0] <= p.x <= rail_x[1] and rail_y[0] < p.y < rail_y[1]:
            sup = cross_member(p.x, p.name)
            zt = sup.top
        if sup is None:
            findings.append(_finding('serious', 'unframed', f'{p.name} stands where the frame has no member under '
                                     'it: it is left out of the frame.', [p.uid]))
            continue
        heads = [q for q in on_roof if q.kind == 'box' and abs(_bottom(q) - _top(p)) < 0.02 and _xy_overlap(q, p)]
        z1 = _top(heads[0]) if heads else _top(p)
        z1 -= st.plate_mm / 1000.0
        if z1 - zt < 0.05:
            findings.append(_finding('warning', 'short_post', f'{p.name} is shorter than 50 mm above the frame: no '
                                     'post, its devices sit on plates on the frame.', [p.uid]))
            continue
        m = bld.add_member(p.name if not p.name.startswith('Sensor mast ·') else 'Mast post', 'post', post.id,
                           (p.x, p.y, zt), (p.x, p.y, z1), (1, 0, 0), p.uid)
        m.name = f'Post ({p.name})'
        bld.joint(m.id, sup.id, 'post', (p.x, p.y, zt))
        if heads:
            h = heads[0]
            used.add(h.uid)
            payloads.append({'uid': h.uid, 'tag': h.name, 'kind': 'top', 'base': np.array([h.x, h.y, z1]),
                             'foot': (h.length, h.width), 'round': False, 'yaw': h.yaw, 'mass': 0.0, 'post': m})
    # ---------------------------------------------------------------- enclosures and other boxes on the rack
    for p in on_roof:
        if p.uid in used or p.kind != 'box' or p.height <= PLATFORM_MAX_H:
            continue
        from .suspension import rig_part_mass
        payloads.append({'uid': p.uid, 'tag': p.name, 'kind': 'payload',
                         'base': np.array([p.x, p.y, _bottom(p)]), 'foot': (p.length, p.width), 'round': False,
                         'yaw': p.yaw, 'mass': rig_part_mass(p), 'centre': np.array([p.x, p.y, p.z]),
                         'half': np.array([p.length, p.width, p.height]) / 2})
        used.add(p.uid)
    for p in on_roof:
        if p.uid not in used and p.kind != 'bar':
            notes.append(f'{p.name} is not part of the frame (it is neither a crossbar, a platform, a mast nor a box '
                         'to carry).')
    # ---------------------------------------------------------------- top plates on posts
    top_plates = {}
    for pl in payloads:
        if pl['kind'] != 'top':
            continue
        m = pl['post']
        P = _plate_for(bld, st, dict(pl, kind='payload'), m, pl['base'][2], 'top plate')
        P.name = f'{pl["tag"]} (top plate)'
        P.device_uid, P.device_tag = '', ''
        bld.joint(P.id, m.id, 'plate', pl['base'])
        top_plates[m.id] = P
    # ---------------------------------------------------------------- devices and payloads
    mounts, unframed = [], []
    items = [dict(d) for d in devs] + [d for d in payloads if d['kind'] == 'payload']
    t = st.plate_mm / 1000.0
    for d in sorted(items, key=lambda d: d['base'][2]):
        B = d['base']
        rec = {'uid': d['uid'], 'tag': d['tag'], 'kind': d['kind'], 'status': '', 'plate': '', 'member': '',
               'standoff': '', 'gap_mm': None}
        # on a post's top plate?
        tp = next((P for mid, P in top_plates.items()
                   if abs(B[2] - (P.centre[2] + P.t)) < 0.03 and math.hypot(B[0] - P.centre[0], B[1] - P.centre[1])
                   <= max(P.lx, P.ly) / 2), None)
        if tp is not None:
            rec.update(status='top plate', plate=tp.id, member=tp.on[0],
                       gap_mm=round(float(B[2] - tp.centre[2] - tp.t) * 1000, 1))
            tp.device_uid = tp.device_uid or d['uid']
            tp.device_tag = (tp.device_tag + ' ' if tp.device_tag else '') + d['tag']
            mounts.append(rec)
            continue
        sup, zt = bld.support_under(B - [0, 0, t], 0.03)
        if sup is None:
            # anything at all under it, too high? (a device sunk into the frame)
            hi, hz = None, -math.inf
            for m in bld.horizontals():
                s, u, _ = m.local(B)
                if -0.002 <= s <= m.length + 0.002 and abs(u) <= m.prof.b / 2 + 0.03 and m.top > hz:
                    hi, hz = m, m.top
            if hi is not None and hz > B[2] - t:
                rec.update(status='below frame', member=hi.id, gap_mm=round(float(B[2] - t - hz) * 1000, 1))
                findings.append(_finding('serious', 'conflict', f'{d["tag"]}\'s base is {abs(B[2] - t - hz) * 1000:.0f} '
                                         f'mm below the top of {hi.name} (with its {st.plate_mm:g} mm plate): raise it '
                                         'onto the frame or move it.', [d['uid']]))
                unframed.append(rec)
                mounts.append(rec)
                continue
            if rails and rail_x[0] - 0.01 <= B[0] <= rail_x[1] + 0.01 and rail_y[0] < B[1] < rail_y[1]:
                sup = cross_member(float(B[0]), d['tag'])
            else:
                sup = arm_to(B, d['tag'])
            if sup is None:
                rec['status'] = 'off frame'
                unframed.append(rec)
                mounts.append(rec)
                findings.append(_finding('warning', 'unframed', f'{d["tag"]} is not over the frame: mount it on the '
                                         'body (or extend the frame to it).', [d['uid']]))
                continue
            zt = sup.top
        gap = B[2] - t - zt
        if gap < -0.002:
            rec.update(status='below frame', member=sup.id, gap_mm=round(float(gap) * 1000, 1))
            findings.append(_finding('serious', 'conflict', f'{d["tag"]}\'s base is {abs(gap) * 1000:.0f} mm below the '
                                     f'top of its plate on {sup.name}: raise it by that much.', [d['uid']]))
            unframed.append(rec)
            mounts.append(rec)
            continue
        on = sup
        if gap > CARRY_GAP_M:
            so = bld.add_member(f'Standoff for {d["tag"]}', 'standoff', post.id, (B[0], B[1], zt),
                                (B[0], B[1], B[2] - t), (1, 0, 0))
            bld.joint(so.id, sup.id, 'post', (B[0], B[1], zt))
            rec['standoff'] = so.id
            on = so
            zt = B[2] - t
        role = {'camera': 'camera mount', 'payload': 'payload mount'}.get(d['kind'], 'sensor mount')
        P = _plate_for(bld, st, d, on, zt, role)
        bld.joint(P.id, on.id, 'plate', (B[0], B[1], zt))
        rec.update(status='plate', plate=P.id, member=on.id, gap_mm=round(float(B[2] - t - zt) * 1000, 1))
        mounts.append(rec)
    # ---------------------------------------------------------------- end machining
    _machining(bld, st)
    res = FramingResult(st, v.name, v.uid, {k: (val.tolist() if isinstance(val, np.ndarray) else val)
                                            for k, val in rf.items()},
                        bld.members, bld.plates, bld.feet, bld.joints, mounts, unframed, [], [], [], {}, findings,
                        notes)
    res.cut_list = cut_list(res)
    res.stock = stock_plan(res)
    res.bom = bom(res)
    res.mass = mass_summary(res)
    res.findings = _findings(res) + findings
    return res


def _finding(sev, kind, message, uids=()):
    return {'severity': sev, 'kind': kind, 'message': message, 'uids': list(uids)}


def _machining(bld: _Builder, st: FramingSettings):
    """Each member's end machining from its joints (anchors: counterbores, access holes and end taps)."""
    if st.joint != 'anchors':
        return
    by = {m.id: m for m in bld.members}
    for j in bld.joints:
        a, b = by.get(j['a']), by.get(j['b'])
        if a is None or b is None:
            continue
        P = np.asarray(j['point'])
        end = 0 if np.linalg.norm(P - a.p0) <= np.linalg.norm(P - a.p1) else 1
        if j['kind'] == 'tee':
            txt = f'anchor counterbore ({a.prof.bolt})'
            b.access_holes += 1
        elif j['kind'] == 'post':
            txt = f'tap {a.prof.bolt} (end)'
            b.access_holes += 1
        else:
            continue
        if end == 0:
            a.end0 = txt
        else:
            a.end1 = txt
    for m in bld.members:
        if m.role == 'post' or m.role == 'standoff':
            if m.end1 == 'square cut':
                m.end1 = f'tap {m.prof.bolt} (end)'


# =========================================================================== cut list, stock, BOM, mass
def _ends(m: Member) -> str:
    e = sorted([m.end0, m.end1])
    if e[0] == e[1]:
        return f'{e[0]} both ends'
    return f'{e[0]} / {e[1]}'


def cut_list(res: FramingResult) -> list[dict]:
    """Identical cuts grouped: [{'item', 'profile', 'label', 'length_mm', 'ends', 'access_holes', 'quantity',
    'members', 'kg_each', 'total_mm'}], longest first."""
    groups = {}
    for m in res.members:
        L = round(m.length * 2000.0) / 2.0          # 0.5 mm
        key = (m.profile, L, _ends(m), m.access_holes)
        groups.setdefault(key, []).append(m)
    rows = []
    for (prof, L, ends, ah), ms in sorted(groups.items(), key=lambda kv: (kv[0][0], -kv[0][1])):
        pr = PROFILES[prof]
        rows.append({'profile': prof, 'label': pr.label, 'part': f'{pr.maker} {pr.part}', 'length_mm': L,
                     'ends': ends, 'access_holes': ah, 'quantity': len(ms), 'members': [m.id for m in ms],
                     'names': [m.name for m in ms], 'kg_each': round(pr.kg_m * L / 1000.0, 4),
                     'total_mm': L * len(ms)})
    for i, r in enumerate(rows):
        r['item'] = _letters(i)
    return rows


def _letters(i: int) -> str:
    s = ''
    i += 1
    while i:
        i, r = divmod(i - 1, 26)
        s = chr(65 + r) + s
    return s


def stock_plan(res: FramingResult) -> list[dict]:
    """Per profile: the cuts packed into stock bars (first-fit decreasing, the saw kerf between cuts):
    [{'profile', 'stock_mm', 'bars', 'cuts_mm' per bar, 'offcut_mm' per bar, 'used_mm', 'waste_pct'}]."""
    st = res.settings
    kerf = st.kerf_mm
    out = []
    for prof in sorted({m.profile for m in res.members}):
        pr = PROFILES[prof]
        stock = st.stock_m * 1000.0 if st.stock_m > 0 else max(pr.stock_m) * 1000.0
        cuts = sorted((round(m.length * 2000.0) / 2.0 for m in res.members if m.profile == prof), reverse=True)
        bars = []
        too_long = [c for c in cuts if c > stock]
        for c in cuts:
            if c > stock:
                continue
            for b in bars:
                if sum(b) + kerf * len(b) + c <= stock + 1e-9:
                    b.append(c)
                    break
            else:
                bars.append([c])
        used = sum(sum(b) for b in bars)
        out.append({'profile': prof, 'label': pr.label, 'stock_mm': stock, 'bars': len(bars),
                    'cuts_mm': bars, 'offcut_mm': [round(stock - sum(b) - kerf * len(b), 1) for b in bars],
                    'used_mm': used, 'waste_pct': round(100.0 * (1 - used / (stock * len(bars))), 1) if bars else 0.0,
                    'too_long_mm': too_long})
    return out


def fixings(res: FramingResult) -> dict:
    """Counts: {'brackets': {profile: n}, 'bolts': {size: n}, 'tnuts': {size: n}, 'anchors': {size: n},
    'device_screws': {kind: n}, 'feet': n}."""
    st = res.settings
    by = {m.id: m for m in res.members}
    out = {'brackets': {}, 'bolts': {}, 'tnuts': {}, 'anchors': {}, 'device_screws': {}, 'feet': len(res.feet)}

    def add(d, k, n):
        d[k] = d.get(k, 0) + n
    for j in res.joints:
        a = by.get(j['a'])
        b = by.get(j['b'])
        if j['kind'] in ('tee', 'stacked', 'post') and a is not None:
            pr = a.prof if a.prof.b <= (b.prof.b if b is not None else 1) else b.prof
            if st.joint == 'anchors' and j['kind'] in ('tee', 'post'):
                add(out['anchors'], a.prof.bolt, 1)
                continue
            n = 4 if j['kind'] == 'post' else 2
            add(out['brackets'], pr.id, n)
            add(out['bolts'], pr.bolt, 2 * n)
            add(out['tnuts'], pr.bolt, 2 * n)
        elif j['kind'] == 'foot':
            pr = by[j['a']].prof
            add(out['bolts'], pr.bolt, 2)
            add(out['tnuts'], pr.bolt, 2)
    for P in res.plates:
        under = [by[m] for m in P.on if m in by]
        for m in under:
            n = sum(1 for h in P.holes if h[3] == 'member')
            add(out['bolts'], m.prof.bolt, n)
            if abs(m.axis[2]) < 0.3:
                add(out['tnuts'], m.prof.bolt, n)
        kind = next((mt['kind'] for mt in res.mounts if mt['plate'] == P.id), 'payload')
        add(out['device_screws'], kind, sum(1 for h in P.holes if h[3] == 'device'))
    return out


def bom(res: FramingResult) -> list[dict]:
    """The bill of materials: [{'category', 'item', 'spec', 'quantity', 'unit', 'kg_each', 'kg', 'source'}]."""
    rows = []
    for s in res.stock:
        pr = PROFILES[s['profile']]
        L = s['used_mm'] / 1000.0
        rows.append({'category': 'Profile', 'item': f'{pr.maker} {pr.part}', 'spec': f'{pr.label}, {pr.alloy}, '
                     f'{s["bars"]} × {s["stock_mm"] / 1000:g} m stock bar{"s" * (s["bars"] != 1)} '
                     f'({s["waste_pct"]:g} % offcut)', 'quantity': round(L, 3), 'unit': 'm',
                     'kg_each': pr.kg_m, 'kg': round(pr.kg_m * L, 3), 'source': pr.source})
    for P in res.plates:
        nh = len(P.holes)
        rows.append({'category': 'Plate (CNC)', 'item': f'{P.id} {P.name}', 'spec': f'{P.lx * 1000:.0f} × '
                     f'{P.ly * 1000:.0f} × {P.t * 1000:g} mm, {nh} hole{"s" * (nh != 1)} '
                     f'(plate_{P.id}.dxf)', 'quantity': 1, 'unit': 'pc', 'kg_each': round(P.mass, 4),
                     'kg': round(P.mass, 4), 'source': PLATE_MATERIAL['label']})
    for F in res.feet:
        m = 0.08 * 0.06 * 0.008 * RHO_AL
        rows.append({'category': 'Plate (CNC)', 'item': f'{F.id} {F.name} plate', 'spec': '80 × 60 × 8 mm, 2 member '
                     f'holes + 2 clamp holes; block {F.height * 1000:.0f} mm tall to the roof', 'quantity': 1,
                     'unit': 'pc', 'kg_each': round(m, 4), 'kg': round(m, 4), 'source': PLATE_MATERIAL['label']})
    fx = fixings(res)
    for prof, n in sorted(fx['brackets'].items()):
        pr = PROFILES[prof]
        rows.append({'category': 'Bracket', 'item': f'Corner bracket, {pr.label} series', 'spec': f'with 2 × '
                     f'{pr.bolt} bolts and T-nuts each (counted below)', 'quantity': n, 'unit': 'pc',
                     'kg_each': pr.bracket_kg, 'kg': round(n * pr.bracket_kg, 3), 'source': 'typical'})
    for size, n in sorted(fx['anchors'].items()):
        rows.append({'category': 'Fastener', 'item': f'Anchor fastener {size}', 'spec': 'counterbore in the butting '
                     'member, access hole in the through member', 'quantity': n, 'unit': 'pc',
                     'kg_each': FASTENER_KG, 'kg': round(n * FASTENER_KG, 3), 'source': 'typical'})
    for size, n in sorted(fx['bolts'].items()):
        rows.append({'category': 'Fastener', 'item': f'Button-head bolt {size}', 'spec': 'length to suit the '
                     'bracket or plate + T-nut', 'quantity': n, 'unit': 'pc', 'kg_each': FASTENER_KG / 2,
                     'kg': round(n * FASTENER_KG / 2, 3), 'source': 'typical'})
    for size, n in sorted(fx['tnuts'].items()):
        rows.append({'category': 'Fastener', 'item': f'Drop-in T-nut {size}', 'spec': 'for the profile\'s slot',
                     'quantity': n, 'unit': 'pc', 'kg_each': FASTENER_KG / 2, 'kg': round(n * FASTENER_KG / 2, 3),
                     'source': 'typical'})
    for kind, n in sorted(fx['device_screws'].items()):
        if n:
            rows.append({'category': 'Fastener', 'item': f'Device screw, {kind}', 'spec': DEVICE_SCREW.get(kind, ''),
                         'quantity': n, 'unit': 'pc', 'kg_each': FASTENER_KG / 2,
                         'kg': round(n * FASTENER_KG / 2, 3), 'source': 'typical'})
    if fx['feet']:
        rows.append({'category': 'Foot', 'item': FEET_LABEL[res.settings.feet], 'spec': 'bought for the vehicle\'s '
                     'roof rails or fix points; rated for the frame\'s mass and the roof load limit',
                     'quantity': fx['feet'], 'unit': 'pc', 'kg_each': FOOT_CLAMP_KG,
                     'kg': round(fx['feet'] * FOOT_CLAMP_KG, 3), 'source': 'typical'})
    return rows


def mass_summary(res: FramingResult) -> dict:
    """{'members_kg', 'plates_kg', 'fixings_kg', 'feet_kg', 'frame_kg', 'devices_kg', 'total_kg', 'length_m'}."""
    mem = sum(m.mass for m in res.members)
    pl = sum(r['kg'] for r in res.bom if r['category'] == 'Plate (CNC)')
    fxk = sum(r['kg'] for r in res.bom if r['category'] in ('Bracket', 'Fastener'))
    ft = sum(r['kg'] for r in res.bom if r['category'] == 'Foot')
    return {'members_kg': round(mem, 3), 'plates_kg': round(pl, 3), 'fixings_kg': round(fxk, 3),
            'feet_kg': round(ft, 3), 'frame_kg': round(mem + pl + fxk + ft, 3),
            'length_m': round(sum(m.length for m in res.members), 4)}


def _findings(res: FramingResult) -> list:
    out = []
    if not res.members:
        out.append(_finding('note', 'empty', 'Nothing on the roof to frame: add a roof rack, a mast or a sensor above '
                            'the roof.'))
        return out
    m = res.mass
    n_mount = sum(1 for x in res.mounts if x['status'] in ('plate', 'top plate'))
    out.append(_finding('good', 'summary', f'{len(res.members)} members ({m["length_m"]:.2f} m of profile), '
                        f'{len(res.plates)} plates, {len(res.feet)} feet: {m["frame_kg"]:.1f} kg; {n_mount} '
                        f'device{"s" * (n_mount != 1)} on the frame.'))
    for a in (x for x in res.members if x.role == 'arm' and x.length > 0.4):
        out.append(_finding('warning', 'cantilever', f'{a.name} overhangs {a.length * 1000:.0f} mm from its crossbar: '
                            'brace it or check its modes (Analysis ▸ Frame modes).'))
    for p in (x for x in res.members if x.role in ('post', 'standoff')):
        pr = p.prof
        if p.length / pr.b > 20:
            out.append(_finding('warning', 'slender', f'{p.name} is {p.length * 1000:.0f} mm tall on a '
                                f'{pr.b * 1000:.0f} mm profile (slenderness {p.length / pr.b:.0f}): a larger profile '
                                'or a brace stiffens it.'))
    for s in res.stock:
        if s['too_long_mm']:
            out.append(_finding('serious', 'stock', f'{len(s["too_long_mm"])} cut{"s" * (len(s["too_long_mm"]) != 1)} '
                                f'of {s["label"]} are longer than its {s["stock_mm"] / 1000:g} m stock bar.'))
    return out


# =========================================================================== geometry: outlines, meshes
def tslot_outline(b, d, slot, unit=None) -> np.ndarray:
    """The profile's outline (n, 2), counter-clockwise, centred, x across b, y across d: a rectangle with simplified
    T-slots (the opening `slot` through a lip of 9 % of the unit, an undercut twice the opening wide (narrower where
    it would meet the next face's) and 16 % of the unit deep), one slot per unit of each face."""
    u = unit or min(b, d)
    lip, und = 0.09 * u, 0.16 * u
    w2 = min(2.0 * slot, u - 2 * (lip + und) - 0.04 * u)      # undercuts of neighbouring faces never meet
    corners = np.array([[-b / 2, -d / 2], [b / 2, -d / 2], [b / 2, d / 2], [-b / 2, d / 2]])
    pts = []
    for k in range(4):
        A, Bc = corners[k], corners[(k + 1) % 4]
        L = float(np.linalg.norm(Bc - A))
        t = (Bc - A) / L
        n = np.array([-t[1], t[0]])                         # inward (the polygon runs counter-clockwise)
        pts.append(A)
        ns = max(int(round(L / u)), 1)
        for i in range(ns):
            c = (i + 0.5) * L / ns
            for s_, dn in ((c - slot / 2, 0.0), (c - slot / 2, lip), (c - w2 / 2, lip), (c - w2 / 2, lip + und),
                           (c + w2 / 2, lip + und), (c + w2 / 2, lip), (c + slot / 2, lip), (c + slot / 2, 0.0)):
                pts.append(A + t * s_ + n * dn)
    return np.array(pts)


def rect_outline(lx, ly) -> np.ndarray:
    return np.array([[-lx / 2, -ly / 2], [lx / 2, -ly / 2], [lx / 2, ly / 2], [-lx / 2, ly / 2]])


def circle_loop(x, y, d, n=STEP_HOLE_SEGMENTS) -> np.ndarray:
    """A hole as an n-gon (clockwise, as a hole in a counter-clockwise outline), inscribed in the circle."""
    a = -np.linspace(0, 2 * math.pi, n, endpoint=False)
    return np.c_[x + d / 2 * np.cos(a), y + d / 2 * np.sin(a)]


def poly_area(P) -> float:
    P = np.asarray(P, float)
    return 0.5 * float(np.sum(P[:, 0] * np.roll(P[:, 1], -1) - np.roll(P[:, 0], -1) * P[:, 1]))


def _ear_clip(P) -> list:
    """Triangles (i, j, k) of a simple counter-clockwise polygon, by ear clipping."""
    idx = list(range(len(P)))
    tris = []

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    guard = 0
    while len(idx) > 3 and guard < 10 * len(P) ** 2:
        guard += 1
        n = len(idx)
        for k in range(n):
            i, j, l_ = idx[(k - 1) % n], idx[k], idx[(k + 1) % n]
            a, b, c = P[i], P[j], P[l_]
            if cross(a, b, c) <= 1e-15:
                continue
            ok = True
            for m in idx:
                if m in (i, j, l_):
                    continue
                p = P[m]
                if cross(a, b, p) > -1e-14 and cross(b, c, p) > -1e-14 and cross(c, a, p) > -1e-14 and not (
                        np.allclose(p, a) or np.allclose(p, b) or np.allclose(p, c)):
                    ok = False
                    break
            if ok:
                tris.append((i, j, l_))
                idx.pop(k)
                break
        else:
            break
    if len(idx) == 3:
        tris.append(tuple(idx))
    return tris


def _cap_triangles(loops) -> tuple[np.ndarray, np.ndarray]:
    """(2D points (n, 2), triangles (m, 3) counter-clockwise) covering the outline loops[0] minus the holes
    loops[1:]. One loop: ear clipping. With holes (a convex outline): Delaunay of every loop vertex, the triangles
    inside a hole dropped (each hole's edges are Delaunay edges when nothing else is near it)."""
    pts = np.vstack(loops)
    if len(loops) == 1:
        return pts, np.array(_ear_clip(pts), np.int64)
    from scipy.spatial import Delaunay
    tri = Delaunay(pts).simplices
    cen = pts[tri].mean(axis=1)
    keep = np.ones(len(tri), bool)
    for h in loops[1:]:
        from matplotlib.path import Path as _P
        keep &= ~_P(h).contains_points(cen)
    tri = tri[keep]
    a = pts[tri]
    cr = (a[:, 1, 0] - a[:, 0, 0]) * (a[:, 2, 1] - a[:, 0, 1]) - (a[:, 1, 1] - a[:, 0, 1]) * (a[:, 2, 0] - a[:, 0, 0])
    tri[cr < 0] = tri[cr < 0][:, ::-1]
    return pts, tri


@dataclass
class Solid:
    """An extrusion: loops (outline counter-clockwise, holes clockwise) in the (ex, ey) plane at origin, extruded
    along ez by length. Vehicle frame, metres."""
    id: str
    name: str
    kind: str                 # member | plate | foot
    loops: list
    origin: np.ndarray
    ex: np.ndarray
    ey: np.ndarray
    ez: np.ndarray
    length: float
    colour: tuple = (0.62, 0.64, 0.67)

    @property
    def volume(self) -> float:
        return sum(poly_area(lp) for lp in self.loops) * self.length

    def mesh(self) -> tuple[np.ndarray, np.ndarray]:
        """(V, F): a closed, consistently wound triangle mesh with outward faces."""
        pts2, tri = _cap_triangles(self.loops)
        n = len(pts2)
        Vb = self.origin + pts2[:, :1] * self.ex + pts2[:, 1:2] * self.ey
        V = np.vstack([Vb, Vb + self.ez * self.length])
        F = [tri[:, ::-1], tri + n]
        off = 0
        for lp in self.loops:
            k = len(lp)
            i = off + np.arange(k)
            j = off + (np.arange(k) + 1) % k
            F.append(np.c_[i, j, j + n])
            F.append(np.c_[i, j + n, i + n])
            off += k
        return V, np.vstack(F).astype(np.int64)


def solids(res: FramingResult) -> list[Solid]:
    """Every part of the frame as an extrusion: members (the T-slot outline), plates (with their holes) and the feet
    (foot block and plate as one block)."""
    out = []
    for m in res.members:
        pr = m.prof
        o, ex, ey, ez = m.frame()
        out.append(Solid(m.id, m.name, 'member', [tslot_outline(pr.b, pr.d, pr.slot, min(pr.b, pr.d))], o, ex, ey, ez,
                         m.length, (0.66, 0.68, 0.71)))
    for P in res.plates:
        c, ex, ey, ez = P.frame()
        loops = [rect_outline(P.lx, P.ly)] + [circle_loop(h[0], h[1], h[2]) for h in P.holes]
        out.append(Solid(P.id, P.name, 'plate', loops, c, ex, ey, ez, P.t, (0.80, 0.81, 0.83)))
    for F in res.feet:
        a = math.radians(F.yaw)
        ex = np.array([math.cos(a), math.sin(a), 0.0])
        ey = np.array([-math.sin(a), math.cos(a), 0.0])
        out.append(Solid(F.id, F.name, 'foot', [rect_outline(F.lx, F.ly)], F.centre, ex, ey, np.array([0, 0, 1.0]),
                         F.height, (0.25, 0.26, 0.28)))
    return out


def stl_files(res: FramingResult) -> list:
    """[(name, binary STL)] one watertight STL per part, millimetres."""
    from .meshing import stl_bytes
    out = []
    for s in solids(res):
        V, F = s.mesh()
        out.append((f'framing_{s.id}.stl', stl_bytes(V * 1000.0, F, f'Vantage frame {s.id} {s.name} mm')))
    return out


def check_meshes(res: FramingResult) -> list[dict]:
    """core.meshing.check of every part's mesh (mm): watertight, consistent, positive volume; and the volume against
    the extrusion's own."""
    from .meshing import check
    out = []
    for s in solids(res):
        V, F = s.mesh()
        c = check(V * 1000.0, F)
        c.update(id=s.id, name=s.name, volume_expected_mm3=s.volume * 1e9)
        out.append(c)
    return out


# =========================================================================== STEP AP214
def _sstr(s) -> str:
    from .exporters import _step_str
    return _step_str(s)


def _sreal(v) -> str:
    s = f'{float(v):.6f}'.rstrip('0')
    return s + '0' if s.endswith('.') else s


def step_text(res: FramingResult, project=None, name='frame') -> str:
    """The frame as a STEP AP214 (AUTOMOTIVE_DESIGN) assembly in millimetres: a root product 'frame' and one part
    per member, plate and foot, each a MANIFOLD_SOLID_BREP of planar faces (its outline extruded), placed with an
    identity transform (the geometry is in the vehicle frame)."""
    rows = []

    def add(t):
        rows.append(t)
        return f'#{len(rows)}'

    def pt(v):
        return add(f"CARTESIAN_POINT('',({','.join(_sreal(x * 1000.0) for x in v)}))")

    def dr(v):
        v = np.asarray(v, float)
        v = v / max(np.linalg.norm(v), 1e-15)
        return add(f"DIRECTION('',({','.join(_sreal(x) for x in v)}))")

    app = add("APPLICATION_CONTEXT('core data for automotive mechanical design processes')")
    add(f"APPLICATION_PROTOCOL_DEFINITION('international standard','automotive_design',2000,{app})")
    pctx = add(f"PRODUCT_CONTEXT('',{app},'mechanical')")
    pdc = add(f"PRODUCT_DEFINITION_CONTEXT('part definition',{app},'design')")
    lu = add('(LENGTH_UNIT()NAMED_UNIT(*)SI_UNIT(.MILLI.,.METRE.))')
    au = add('(NAMED_UNIT(*)PLANE_ANGLE_UNIT()SI_UNIT($,.RADIAN.))')
    su = add('(NAMED_UNIT(*)SI_UNIT($,.STERADIAN.)SOLID_ANGLE_UNIT())')
    unc = add(f"UNCERTAINTY_MEASURE_WITH_UNIT(LENGTH_MEASURE(1.E-03),{lu},'distance_accuracy_value',"
              "'confusion accuracy')")
    gctx = add(f"(GEOMETRIC_REPRESENTATION_CONTEXT(3)GLOBAL_UNCERTAINTY_ASSIGNED_CONTEXT(({unc}))"
               f"GLOBAL_UNIT_ASSIGNED_CONTEXT(({lu},{au},{su}))REPRESENTATION_CONTEXT('Context #1',"
               "'3D Context with UNIT and UNCERTAINTY'))")

    def product(pid, label):
        prod = add(f"PRODUCT({_sstr(pid)},{_sstr(label)},'',({pctx}))")
        add(f"PRODUCT_RELATED_PRODUCT_CATEGORY('part',$,({prod}))")
        pdf = add(f"PRODUCT_DEFINITION_FORMATION('','',{prod})")
        pd = add(f"PRODUCT_DEFINITION('design','',{pdf},{pdc})")
        pds = add(f"PRODUCT_DEFINITION_SHAPE('','',{pd})")
        return pd, pds

    def origin_axis():
        return add(f"AXIS2_PLACEMENT_3D('',{pt((0, 0, 0))},{dr((0, 0, 1))},{dr((1, 0, 0))})")

    root_pd, root_pds = product('frame', f'Roof frame{" · " + project.name if project is not None else ""}')
    root_ax = origin_axis()
    root_rep = add(f"SHAPE_REPRESENTATION('frame',({root_ax}),{gctx})")
    add(f"SHAPE_DEFINITION_REPRESENTATION({root_pds},{root_rep})")
    for k, s in enumerate(solids(res)):
        brep = _brep(add, pt, dr, s)
        ax = origin_axis()
        rep = add(f"ADVANCED_BREP_SHAPE_REPRESENTATION({_sstr(s.id)},({brep},{ax}),{gctx})")
        pd, pds = product(s.id, s.name)
        add(f"SHAPE_DEFINITION_REPRESENTATION({pds},{rep})")
        nauo = add(f"NEXT_ASSEMBLY_USAGE_OCCURRENCE({_sstr(str(k + 1))},{_sstr(s.name)},'',{root_pd},{pd},$)")
        npds = add(f"PRODUCT_DEFINITION_SHAPE('','',{nauo})")
        tr = add(f"ITEM_DEFINED_TRANSFORMATION('','',{ax},{root_ax})")
        rr = add(f"(REPRESENTATION_RELATIONSHIP('','',{rep},{root_rep})"
                 f"REPRESENTATION_RELATIONSHIP_WITH_TRANSFORMATION({tr})SHAPE_REPRESENTATION_RELATIONSHIP())")
        add(f"CONTEXT_DEPENDENT_SHAPE_REPRESENTATION({rr},{npds})")
    stamp = ''
    try:
        from .provenance import utc_now
        stamp = utc_now()
    except Exception:                               # provenance's clock helper missing: no time stamp
        stamp = ''
    head = ['ISO-10303-21;', 'HEADER;',
            f"FILE_DESCRIPTION(({_sstr('Vantage roof frame assembly (core.framing)')},"
            f"{_sstr('vehicle frame ISO 8855 rear axle, millimetres; planar B-rep solids')}),'2;1');",
            f"FILE_NAME({_sstr(name + '.step')},{_sstr(stamp)},('Vantage'),('Vantage'),'Vantage','Vantage','');",
            "FILE_SCHEMA(('AUTOMOTIVE_DESIGN { 1 0 10303 214 1 1 1 1 }'));", 'ENDSEC;', 'DATA;']
    body = [f'#{i + 1}={r};' for i, r in enumerate(rows)]
    return '\n'.join(head + body + ['ENDSEC;', 'END-ISO-10303-21;']) + '\n'


def _brep(add, pt, dr, s: Solid) -> str:
    """A MANIFOLD_SOLID_BREP of the extrusion: caps and side faces, every edge shared by two faces with opposite
    orientations."""
    o, ex, ey, ez, L = s.origin, s.ex, s.ey, s.ez, s.length

    def at(p2, z):
        return o + p2[0] * ex + p2[1] * ey + z * ez
    vb, vt, pb, pt_ = [], [], [], []
    for lp in s.loops:
        b_, t_, q1, q2 = [], [], [], []
        for p2 in lp:
            P0, P1 = at(p2, 0.0), at(p2, L)
            q1.append(P0)
            q2.append(P1)
            b_.append(add(f"VERTEX_POINT('',{pt(P0)})"))
            t_.append(add(f"VERTEX_POINT('',{pt(P1)})"))
        vb.append(b_)
        vt.append(t_)
        pb.append(q1)
        pt_.append(q2)
    edges = {}

    def edge(v0, v1, P0, P1):
        """The EDGE_CURVE from v0 to v1 (made once), and the orientation to use it from v0 to v1."""
        if (v1, v0) in edges:
            return edges[(v1, v0)], '.F.'
        if (v0, v1) not in edges:
            d = np.asarray(P1) - np.asarray(P0)
            ln = add(f"LINE('',{pt(P0)},{add(f'VECTOR({chr(39)}{chr(39)},{dr(d)},{_sreal(np.linalg.norm(d) * 1000)})')})")
            edges[(v0, v1)] = add(f"EDGE_CURVE('',{v0},{v1},{ln},.T.)")
        return edges[(v0, v1)], '.T.'

    def loop(vs, Ps):
        oes = []
        n = len(vs)
        for i in range(n):
            e, ori = edge(vs[i], vs[(i + 1) % n], Ps[i], Ps[(i + 1) % n])
            oes.append(add(f"ORIENTED_EDGE('',*,*,{e},{ori})"))
        return add(f"EDGE_LOOP('',({','.join(oes)}))")

    def face(loops_vs, loops_P, normal, pnt):
        bounds = []
        for k, (vs, Ps) in enumerate(zip(loops_vs, loops_P)):
            lp = loop(vs, Ps)
            bounds.append(add(f"{'FACE_OUTER_BOUND' if k == 0 else 'FACE_BOUND'}('',{lp},.T.)"))
        ref = np.cross(normal, [0.0, 0.0, 1.0]) if abs(normal[2]) < 0.9 else np.cross(normal, [1.0, 0.0, 0.0])
        pl = add(f"PLANE('',{add(f'AXIS2_PLACEMENT_3D({chr(39)}{chr(39)},{pt(pnt)},{dr(normal)},{dr(ref)})')})")
        return add(f"ADVANCED_FACE('',({','.join(bounds)}),{pl},.T.)")
    faces = []
    # bottom cap: every loop reversed (outward normal −ez); top cap as given (+ez)
    faces.append(face([v[::-1] for v in vb], [p[::-1] for p in pb], -ez, pb[0][0]))
    faces.append(face(vt, pt_, ez, pt_[0][0]))
    for vs_b, vs_t, Pb, Pt in zip(vb, vt, pb, pt_):
        n = len(vs_b)
        for i in range(n):
            j = (i + 1) % n
            d = Pb[j] - Pb[i]
            nrm = np.cross(d, ez)
            nrm /= max(np.linalg.norm(nrm), 1e-15)
            faces.append(face([[vs_b[i], vs_b[j], vs_t[j], vs_t[i]]], [[Pb[i], Pb[j], Pt[j], Pt[i]]], nrm, Pb[i]))
    shell = add(f"CLOSED_SHELL('',({','.join(faces)}))")
    return add(f"MANIFOLD_SOLID_BREP({_sstr(s.name)},{shell})")


def verify_step(text: str, res: FramingResult | None = None) -> dict:
    """Read the STEP back with gmsh's OpenCASCADE kernel when gmsh is installed: {'available', 'volumes',
    'volume_mm3' per solid, 'expected_mm3', 'max_rel_error'}; {'available': False, 'reason'} without it."""
    try:
        import gmsh
    except Exception as e:                           # optional: the app writes STEP without it
        return {'available': False, 'reason': f'needs gmsh (with OpenCASCADE): {type(e).__name__}'}
    import tempfile
    path = None
    try:
        with tempfile.NamedTemporaryFile('w', suffix='.step', delete=False) as fh:
            fh.write(text)
            path = fh.name
        gmsh.initialize(readConfigFiles=False)
        gmsh.option.setNumber('General.Terminal', 0)
        gmsh.option.setString('Geometry.OCCTargetUnit', 'MM')
        ents = gmsh.model.occ.importShapes(path)
        gmsh.model.occ.synchronize()
        vols = [gmsh.model.occ.getMass(3, t) for d, t in ents if d == 3]
        out = {'available': True, 'volumes': len(vols), 'volume_mm3': sorted(vols)}
    except Exception as e:                           # a reader failure is the check's result
        return {'available': True, 'volumes': 0, 'error': f'{type(e).__name__}: {e}'}
    finally:
        try:
            gmsh.finalize()
        except Exception:
            pass
        if path:
            os.unlink(path)
    if res is not None:
        exp = sorted(s.volume * 1e9 for s in solids(res))
        out['expected_mm3'] = exp
        if len(exp) == len(vols) and vols:
            out['max_rel_error'] = max(abs(a - b) / max(b, 1e-9) for a, b in zip(sorted(vols), exp))
    return out


# =========================================================================== DXF (R12, mm)
def _dxf(entities_fn) -> str:
    out = []

    def g(code, val):
        out.append(f'{code:>3}\n{val}\n')
    g(0, 'SECTION'); g(2, 'HEADER')                                # noqa: E702
    g(9, '$ACADVER'); g(1, 'AC1009')                               # noqa: E702
    g(9, '$INSUNITS'); g(70, 4)                                    # noqa: E702
    g(0, 'ENDSEC')
    g(0, 'SECTION'); g(2, 'TABLES')                                # noqa: E702
    g(0, 'TABLE'); g(2, 'LAYER'); g(70, 3)                         # noqa: E702
    for name, col in (('OUTLINE', 7), ('HOLES', 1), ('TEXT', 3)):
        g(0, 'LAYER'); g(2, name); g(70, 0); g(62, col); g(6, 'CONTINUOUS')     # noqa: E702
    g(0, 'ENDTAB'); g(0, 'ENDSEC')                                 # noqa: E702
    g(0, 'SECTION'); g(2, 'ENTITIES')                              # noqa: E702
    entities_fn(g)
    g(0, 'ENDSEC'); g(0, 'EOF')                                    # noqa: E702
    return ''.join(out)


def _n(v) -> str:
    return f'{float(v):.4f}'


def _plate_entities(g, P: Plate, dx=0.0, dy=0.0):
    """The plate's outline (closed polyline), its holes (circles) and its label, mm, its centre at (dx, dy)."""
    lx, ly = P.lx * 1000, P.ly * 1000
    g(0, 'POLYLINE'); g(8, 'OUTLINE'); g(66, 1); g(10, '0.0'); g(20, '0.0'); g(30, '0.0'); g(70, 1)  # noqa: E702
    for x, y in ((-lx / 2, -ly / 2), (lx / 2, -ly / 2), (lx / 2, ly / 2), (-lx / 2, ly / 2)):
        g(0, 'VERTEX'); g(8, 'OUTLINE'); g(10, _n(x + dx)); g(20, _n(y + dy)); g(30, '0.0')      # noqa: E702
    g(0, 'SEQEND'); g(8, 'OUTLINE')                                 # noqa: E702
    for x, y, d, _ in P.holes:
        g(0, 'CIRCLE'); g(8, 'HOLES'); g(10, _n(x * 1000 + dx)); g(20, _n(y * 1000 + dy)); g(30, '0.0')  # noqa: E702
        g(40, _n(d * 500))
    g(0, 'TEXT'); g(8, 'TEXT'); g(10, _n(dx - lx / 2)); g(20, _n(dy - ly / 2 - 8)); g(30, '0.0')   # noqa: E702
    g(40, '4.0'); g(1, f'{P.id} {P.name} t={P.t * 1000:g} mm'.replace('\n', ' '))


def dxf_plate(P: Plate) -> str:
    return _dxf(lambda g: _plate_entities(g, P))


def dxf_sheet(res: FramingResult) -> str:
    """Every plate on one sheet, side by side 20 mm apart."""
    def ents(g):
        x = 0.0
        for P in res.plates:
            _plate_entities(g, P, x + P.lx * 500, P.ly * 500)
            x += P.lx * 1000 + 20
    return _dxf(ents)


def read_dxf(text: str) -> list[tuple]:
    """(code, value) pairs of a DXF; raises ValueError on an odd number of lines or a non-integer group code."""
    lines = text.splitlines()
    if len(lines) % 2:
        raise ValueError('odd number of lines')
    out = []
    for i in range(0, len(lines), 2):
        out.append((int(lines[i].strip()), lines[i + 1]))
    return out


# =========================================================================== rig parts, the picture
def to_rig_parts(res: FramingResult):
    """The frame as core.project RigParts (boxes), so it occludes and carries like any rig part: members, plates
    and feet."""
    from .project import RigPart
    out = []
    for s in solids(res):
        if s.kind == 'member':
            m = res.member(s.id)
            pr = m.prof
            c = (m.p0 + m.p1) / 2
            if abs(m.axis[2]) > 0.7:
                out.append(RigPart(name=f'{FRAME_PREFIX}{m.name}', kind='box', x=float(c[0]), y=float(c[1]), z=float(c[2]),
                                   length=pr.d, width=pr.b, height=m.length, radius=0.01))
            else:
                yaw = math.degrees(math.atan2(m.axis[1], m.axis[0]))
                out.append(RigPart(name=f'{FRAME_PREFIX}{m.name}', kind='box', x=float(c[0]), y=float(c[1]),
                                   z=float(c[2]), length=m.length, width=pr.b, height=pr.d, radius=0.01,
                                   yaw=round(yaw, 4)))
        elif s.kind == 'plate':
            P = next(p for p in res.plates if p.id == s.id)
            out.append(RigPart(name=f'{FRAME_PREFIX}{P.name}', kind='box', x=float(P.centre[0]), y=float(P.centre[1]),
                               z=float(P.centre[2] + P.t / 2), length=P.lx, width=P.ly, height=P.t, radius=0.01,
                               yaw=round(P.yaw, 4)))
        else:
            F = next(f for f in res.feet if f.id == s.id)
            out.append(RigPart(name=f'{FRAME_PREFIX}{F.name}', kind='box', x=float(F.centre[0]), y=float(F.centre[1]),
                               z=float(F.centre[2] + F.height / 2), length=F.lx, width=F.ly, height=F.height,
                               radius=0.01, yaw=round(F.yaw, 4)))
    for p in out:
        for k in ('x', 'y', 'z', 'length', 'width', 'height'):
            setattr(p, k, round(getattr(p, k), 4))
    return out


def replaced_uids(res: FramingResult, project) -> list[str]:
    """The rig parts the frame stands for: those its members replace, the platform and mast heads it absorbed."""
    out = {m.from_uid for m in res.members if m.from_uid}
    for P in res.plates:
        if P.role == 'top plate':
            for p in project.rig:
                if P.name.startswith(p.name):
                    out.add(p.uid)
    return sorted(out)


def apply_to_rig(project, res: FramingResult):
    """Add the frame to the project's rig as parts (one edit for the caller to wrap): earlier frame parts removed,
    the parts the frame stands for hidden and listed in the framing settings so it is still synthesised from them."""
    rep = replaced_uids(res, project)
    project.rig = [p for p in project.rig if not p.name.startswith(FRAME_PREFIX)]
    for p in project.rig:
        if p.uid in rep:
            p.visible = False
    new = to_rig_parts(res)
    project.rig.extend(new)
    st = FramingSettings.from_dict(project.analysis.framing)
    st.replaced = sorted(set(st.replaced) | set(rep))
    project.analysis.framing = st.to_dict()
    return new


def remove_from_rig(project):
    """Undo apply_to_rig as an edit: the frame parts removed, the parts it stood for shown again."""
    st = FramingSettings.from_dict(project.analysis.framing)
    project.rig = [p for p in project.rig if not p.name.startswith(FRAME_PREFIX)]
    for p in project.rig:
        if p.uid in st.replaced:
            p.visible = True
    st.replaced = []
    project.analysis.framing = st.to_dict()


def device_boxes(project, res: FramingResult, variant=None) -> list:
    """[(V, F, uid)] of the devices' housings near the frame, for drawing."""
    from . import thermo_vib as TV
    from .meshing import box_mesh
    v = variant or project.variant
    rf = TV.roof_box(project)
    out = []
    for d in _devices(project, v, rf, res.settings):
        V, F = box_mesh(d['centre'], d['R'], d['half'])
        out.append((V, F, d['uid']))
    return out


def png(res: FramingResult, project=None, size=(900, 620), yaw=-130.0, elev=24.0) -> bytes:
    """A shaded picture of the frame (and the devices) from the front left."""
    import cv2
    from .meshing import render, view_basis
    layers = []
    for s in solids(res):
        V, F = s.mesh()
        layers.append((V, F, tuple(int(255 * c) for c in s.colour), 1.0))
    if project is not None:
        for V, F, _ in device_boxes(project, res):
            layers.append((V, F, (40, 150, 170), 1.0))
    W, H = size
    if not layers:
        img = np.full((H, W, 3), 245, np.uint8)
    else:
        allv = np.vstack([L[0] for L in layers])
        c = (allv.min(axis=0) + allv.max(axis=0)) / 2
        rad = float(np.linalg.norm(allv - c, axis=1).max()) or 1.0
        img = render(layers, view_basis(yaw, elev), size, 0.45 * min(W, H) / rad, c, bg=(245, 245, 243))
    ok, buf = cv2.imencode('.png', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return buf.tobytes() if ok else b''


# =========================================================================== the study interface
def run(project, engine=None, settings=None, variant=None, progress=None, cancel=None) -> FramingResult | None:
    """The frame (cheap). engine is not needed (kept for the common signature)."""
    if cancel is not None and cancel():
        return None
    res = synthesise(project, settings, variant)
    if progress:
        progress(1.0)
    return res


def inputs_key(project, variant=None) -> str:
    v = variant or project.variant
    m = project.vehicle.mesh
    return json.dumps({'mesh': [id(m), getattr(m, 'n_tris', 0)], 'variant': v.to_dict(),
                       'rig': [r.to_dict() for r in project.rig], 'framing': project.analysis.framing,
                       'library': project.section('library')}, sort_keys=True, default=str)


_ISSUES: dict = {}


def issues(project, vg=None) -> list:
    """Devices sunk into the frame or off it, cuts longer than stock (cached on the inputs; only when the project has
    rig parts)."""
    from . import rules
    if not any(p.visible for p in getattr(project, 'rig', []) or []):
        return []
    key = inputs_key(project)
    hit = _ISSUES.get(key)
    if hit is not None:
        return list(hit)
    res = synthesise(project)
    out = [rules.Issue(f['severity'], f['message'], f'{res.variant} · frame', list(f.get('uids', [])), 'framing')
           for f in res.findings if f['severity'] in ('critical', 'serious', 'warning')]
    if len(_ISSUES) > 8:
        _ISSUES.pop(next(iter(_ISSUES)))
    _ISSUES[key] = out
    return list(out)


def _r(v, d=4):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return round(v, d) if math.isfinite(v) else None


def _v(a, d=4):
    return [_r(x * 1000.0, 1) for x in np.asarray(a, float)] if a is not None else None


def summary(res: FramingResult) -> dict:
    return {'variant': res.variant, 'members': len(res.members), 'plates': len(res.plates), 'feet': len(res.feet),
            'profile_m': res.mass.get('length_m'), 'frame_kg': res.mass.get('frame_kg'),
            'cuts': sum(r['quantity'] for r in res.cut_list), 'cut_items': len(res.cut_list),
            'stock_bars': {s['profile']: s['bars'] for s in res.stock},
            'mounted': [m['tag'] for m in res.mounts if m['status'] in ('plate', 'top plate')],
            'unframed': [m['tag'] for m in res.unframed], 'worst': res.worst(),
            'findings': [f['message'] for f in res.findings if f['severity'] in ('critical', 'serious', 'warning')]}


def to_doc(res: FramingResult, project=None) -> dict:
    """The JSON export: settings, profiles used (with sources), members, plates, feet, joints, mounts, the cut list,
    the stock plan, the BOM, masses and findings. Positions mm in the vehicle frame."""
    used = sorted({m.profile for m in res.members})
    return {
        'schema': SCHEMA, 'project': getattr(project, 'name', ''), 'variant': res.variant, 'frame': FRAME,
        'units': {'position': 'mm (vehicle frame)', 'length': 'mm', 'mass': 'kg', 'angle': 'deg'},
        'assumptions': ASSUMPTIONS, 'model': __doc__.split('Out of scope:')[0].strip(),
        'limits': 'Out of scope: ' + __doc__.split('Out of scope:')[1].strip(),
        'settings': res.settings.to_dict(),
        'profiles': {k: {'label': PROFILES[k].label, 'maker': PROFILES[k].maker, 'part': PROFILES[k].part,
                         'b_mm': PROFILES[k].b * 1000, 'd_mm': PROFILES[k].d * 1000,
                         'area_mm2': _r(PROFILES[k].area * 1e6, 2), 'i_deep_mm4': _r(PROFILES[k].i_d * 1e12, 0),
                         'i_narrow_mm4': _r(PROFILES[k].i_b * 1e12, 0), 'kg_m': PROFILES[k].kg_m,
                         'E_gpa': _r(PROFILES[k].E / 1e9, 2), 'alloy': PROFILES[k].alloy,
                         'source': PROFILES[k].source} for k in used},
        'roof': {'top_mm': _r(res.roof['top'] * 1000, 1), 'x_front_mm': _r(res.roof['x_front'] * 1000, 1),
                 'x_rear_mm': _r(res.roof['x_rear'] * 1000, 1)},
        'members': [{'id': m.id, 'name': m.name, 'role': m.role, 'profile': m.profile, 'start_mm': _v(m.p0),
                     'end_mm': _v(m.p1), 'up': [_r(x) for x in m.up], 'length_mm': _r(m.length * 1000, 1),
                     'ends': [m.end0, m.end1], 'access_holes': m.access_holes, 'kg': _r(m.mass, 3),
                     'replaces': m.from_uid} for m in res.members],
        'plates': [{'id': P.id, 'name': P.name, 'role': P.role, 'centre_mm': _v(P.centre),
                    'size_mm': [_r(P.lx * 1000, 1), _r(P.ly * 1000, 1), _r(P.t * 1000, 1)], 'yaw_deg': _r(P.yaw, 3),
                    'holes_mm': [{'x': _r(h[0] * 1000, 2), 'y': _r(h[1] * 1000, 2), 'd': _r(h[2] * 1000, 2),
                                  'for': h[3]} for h in P.holes], 'device': P.device_tag, 'on': P.on,
                    'kg': _r(P.mass, 4), 'dxf': f'plate_{P.id}.dxf'} for P in res.plates],
        'feet': [{'id': F.id, 'name': F.name, 'crossbar': F.member, 'base_mm': _v(F.centre),
                  'height_mm': _r(F.height * 1000, 1), 'roof_z_mm': _r(F.roof_z * 1000, 1)} for F in res.feet],
        'joints': [dict(j, point=[_r(x * 1000, 1) for x in j['point']]) for j in res.joints],
        'mounts': res.mounts, 'unframed': [m['tag'] for m in res.unframed],
        'cut_list': res.cut_list, 'stock': res.stock, 'bom': res.bom, 'fixings': fixings(res), 'mass': res.mass,
        'step': {'writer': 'direct AP214 (planar B-rep per part; outlines extruded, holes as '
                           f'{STEP_HOLE_SEGMENTS}-gons)', 'file': 'framing.step'},
        'findings': [{'severity': f['severity'], 'kind': f['kind'], 'message': f['message']} for f in res.findings],
        'notes': list(res.notes),
    }


CUT_COLUMNS = ('item', 'profile', 'part', 'length_mm', 'ends', 'access_holes', 'quantity', 'kg_each', 'total_mm',
               'members')
BOM_COLUMNS = ('category', 'item', 'spec', 'quantity', 'unit', 'kg_each', 'kg', 'source')


def cut_csv(res: FramingResult) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator='\n')
    w.writerow(CUT_COLUMNS)
    for r in res.cut_list:
        w.writerow([r['item'], r['label'], r['part'], f'{r["length_mm"]:.1f}', r['ends'], r['access_holes'],
                    r['quantity'], f'{r["kg_each"]:.3f}', f'{r["total_mm"]:.1f}', ' '.join(r['members'])])
    return buf.getvalue()


def bom_csv(res: FramingResult) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator='\n')
    w.writerow(BOM_COLUMNS)
    for r in res.bom:
        w.writerow([r[k] for k in BOM_COLUMNS])
    return buf.getvalue()


def bom_pdf(res: FramingResult, project=None) -> bytes | None:
    """An A4 sheet of the cut list, the stock plan and the BOM (needs a running QGuiApplication; None without)."""
    from PyQt6.QtGui import QGuiApplication
    if QGuiApplication.instance() is None:
        return None
    from .exporters import _PdfOut, _Sheet
    out = _PdfOut(False, f'Frame cut list and BOM · {getattr(project, "name", "")}')
    sh = _Sheet(out)
    W, H, M = 210.0, 297.0, 12.0
    y = M

    def page_break(need):
        nonlocal y
        if y + need > H - M:
            out.new_page()
            y = M

    sh.text(M, y, W - 2 * M, 8, 'Roof frame: cut list and bill of materials', sh.font(13, 'semibold'))
    y += 9
    sh.text(M, y, W - 2 * M, 5, f'{getattr(project, "name", "")} · {res.variant} · {res.mass["frame_kg"]:.1f} kg · '
            f'{res.mass["length_m"]:.2f} m of profile · {JOINT_LABEL[res.settings.joint]}', sh.font(8))
    y += 8

    def table(title, cols, rows):
        nonlocal y
        page_break(14)
        sh.text(M, y, W - 2 * M, 6, title, sh.font(10, 'semibold'))
        y += 7
        x = M
        for name, w, al in cols:
            sh.text(x, y, w, 4, name, sh.font(6.5, 'semibold', caps=True), align=al)
            x += w
        y += 4.5
        sh.line(M, y, W - M, y)
        y += 1
        for r in rows:
            page_break(5)
            x = M
            for (name, w, al), val in zip(cols, r):
                sh.text(x, y, w - 1, 4, str(val), sh.font(7), align=al, elide=True)
                x += w
            y += 4.2
        y += 4
    table('Cut list', [('Item', 10, 'l'), ('Profile', 34, 'l'), ('Length mm', 20, 'r'), ('Ends', 62, 'l'),
                       ('Qty', 10, 'r'), ('kg each', 16, 'r'), ('Members', 34, 'l')],
          [(r['item'], r['label'], f'{r["length_mm"]:.1f}', r['ends'] + (f'; {r["access_holes"]} access holes'
                                                                          if r['access_holes'] else ''),
            r['quantity'], f'{r["kg_each"]:.2f}', ' '.join(r['members'])) for r in res.cut_list])
    table('Stock bars', [('Profile', 40, 'l'), ('Stock mm', 22, 'r'), ('Bars', 12, 'r'), ('Offcut %', 18, 'r'),
                         ('Cuts per bar (mm)', 94, 'l')],
          [(s['label'], f'{s["stock_mm"]:.0f}', s['bars'], f'{s["waste_pct"]:g}',
            ' | '.join(' + '.join(f'{c:g}' for c in b) for b in s['cuts_mm'])) for s in res.stock])
    table('Bill of materials', [('Category', 22, 'l'), ('Item', 46, 'l'), ('Spec', 78, 'l'), ('Qty', 14, 'r'),
                                ('kg', 16, 'r')],
          [(r['category'], r['item'], r['spec'], f'{r["quantity"]:g} {r["unit"]}', f'{r["kg"]:.3f}')
           for r in res.bom])
    return out.finish()


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return str(o)


def files(res: FramingResult, project=None) -> list:
    """[(name, bytes)]: framing.json, the cut list and BOM (CSV, and PDF when a Qt application runs), framing.step,
    plates.dxf with one plate_<id>.dxf each, one framing_<id>.stl per part, framing.png."""
    out = [('framing.json', (json.dumps(to_doc(res, project), indent=1, default=_json_default) + '\n').encode()),
           ('framing_cutlist.csv', cut_csv(res).encode('utf-8')),
           ('framing_bom.csv', bom_csv(res).encode('utf-8'))]
    try:
        pdf = bom_pdf(res, project)
    except Exception:                                 # no fonts or painter here: the CSVs carry the same tables
        pdf = None
    if pdf:
        out.append(('framing_bom.pdf', pdf))
    if res.members or res.plates:
        out.append(('framing.step', step_text(res, project, 'framing').encode('utf-8')))
    if res.plates:
        out.append(('plates.dxf', dxf_sheet(res).encode('ascii', 'replace')))
        out += [(f'plate_{P.id}.dxf', dxf_plate(P).encode('ascii', 'replace')) for P in res.plates]
    out += stl_files(res)
    out.append(('framing.png', png(res, project)))
    return out


def export(res: FramingResult, project, out_dir) -> list[str]:
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for name, data in files(res, project):
        path = os.path.join(out_dir, name)
        with open(path, 'wb') as fh:
            fh.write(data)
        paths.append(path)
    return paths


# =========================================================================== validation
def validation_checks() -> list[dict]:
    """Spec-compatible checks for vantage.validation (the lead registers them)."""
    def mass_vs_area():
        worst, det = 0.0, {}
        for p in PROFILES.values():
            e = abs(p.kg_m / (p.area * RHO_AL) - 1.0)
            det[p.id] = round(e * 100, 2)
            worst = max(worst, e)
        return {'error': worst * 100, 'detail': det}

    def cut_sums():
        from . import samples
        from .rig import apply_fit, fit_plan
        p = samples.tigor_sample_project()
        p.active_uid = next(v for v in p.variants if v.sensors).uid
        for k in ('roof_rack', 'sensor_mast'):
            apply_fit(p, fit_plan(p, k))
        r = synthesise(p)
        a = sum(m.length for m in r.members) * 1000
        b = sum(c['total_mm'] for c in r.cut_list)
        return {'error': abs(a - b), 'detail': {'members_mm': a, 'cut_list_mm': b}}

    return [
        {'id': 'framing.profile_mass', 'model': 'framing', 'title': 'Catalogue mass per metre against area × density',
         'quantity': 'worst |kg/m ÷ (area × 2700 kg/m³) − 1| over the profile library, %',
         'reference': 'the makers\' own section areas and 2700 kg/m³ (ASM, wrought aluminium)', 'tolerance': 3.0,
         'unit': '%', 'refs': (), 'fn': mass_vs_area, 'cost': 'cheap',
         'why': 'two independent catalogue figures must agree: a transcription error in either shows'},
        {'id': 'framing.cut_list_sum', 'model': 'framing', 'title': 'Cut list total against the members',
         'quantity': '|Σ member lengths − Σ cut list (length × quantity)|, mm',
         'reference': 'the frame\'s own member list', 'tolerance': 0.5 * 20, 'unit': 'mm', 'refs': (),
         'fn': cut_sums, 'cost': 'cheap', 'why': 'grouping rounds each cut to 0.5 mm; nothing may be lost'},
    ]
