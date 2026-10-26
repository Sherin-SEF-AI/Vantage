"""Critical targets and inverse mounting tolerances: what the rig must resolve at known places, and how far each
camera's mount may be off before it stops resolving them.

Targets. A critical target is an object at a place the rig must resolve: a child 0.9 m tall standing 2 m in front of
the bumper, a pedestrian, a car, a traffic light at a junction, a sign, a kerb beside the car, a lane marking on a
curve. Presets give typical sizes and needs (TYPICAL says where each figure comes from); every figure is editable and
the targets are saved with the project (analysis.critical). Positions are in the vehicle frame (ISO 8855, rear axle,
metres): a standing target at the point it stands on, a kerb from its start along its heading, a lane marking from its
start along its heading, straight or curving with a signed radius (positive to the left).

Geometry. A standing target (child, pedestrian, car, traffic light, sign, any box) is a vertical rectangle of its
height and width, its bottom `base` above the ground, square to the camera's line of sight (as the perception range
models objects). A kerb is a vertical face `height` high along a straight line; a lane marking a painted line `width`
wide on the ground. Each is sampled as stations (columns across a standing target, points along a kerb or marking)
with nodes along the span its requirement measures: a standing target's or a kerb's height, a marking's width.

Seen. A node is in view when it projects inside the image and the lens circle through the camera's lens model (its
unit calibration when it has one), and seen when, in addition, nothing blocks the straight line to it: the body and
the rig (the occluder triangles the analysis engine builds its cube maps from) and the variant's housings (their
solids; the camera's own left out). Line of sight is tested exactly, ray against triangle and solid, not through the
cube maps' 0.35° texels, because the envelopes move the camera and a cube map belongs to one position; the two agree
except within a texel of a silhouette.

Pixels. Along each station the span is projected node by node; the pixel length of its seen part is the sum of the
projected segments (a segment seen at one end and leaving the image at the other counts the share inside,
interpolated on the distance to the image border; one hidden at the other end counts half). A standing target's
pixels are its best station's (its height in the image, of the part seen); a kerb's or a marking's the pixel count
that at least min_share of its length reaches (a station out of focus counts none).

Met. A camera meets a standing target when at least min_share of its face is seen, its pixels reach min_px and (focus
on) its distance lies in the lens's focus range; a kerb or a marking when min_share of its length reaches min_px. A
target is resolved when some camera meets it and, with min_points > 0, the lidars together put at least min_points on
it in one frame (counted as the lidar and radar coverage counts them). Radars report whether they detect it (rcs > 0).

Envelopes. For every camera and each target it meets, the camera's pose is moved along one axis at a time, both ways:
yaw, pitch and roll as increments of its Z-Y-X angles (pitch positive down), x, y and z in the vehicle frame. Angles
are scanned outwards in 0.25° steps to search_deg, positions in steps growing from 0.5 mm to search_mm, and the first
step at which the target is no longer met is bisected to 0.01° / 0.1 mm: how far that axis alone may be off. A move
that would put the lens inside the body, the rig or a housing stops there (limit 'surface': not a loss of the target);
no loss within the search range is 'open'. Joint: all six axes at once on the grid of (−, 0, +) offsets per axis (729
poses): the largest share of the single-axis limits at which every pose still meets the target, and whether the
target holds on that grid at ± the install tolerance. A camera's envelope is the tightest over the targets it meets;
its binding target the one that leaves the least margin over the install tolerance.
"""
from __future__ import annotations

import copy
import csv
import io
import itertools
import json
import math
import os
import threading
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass, field

import numpy as np

from . import housings as H
from .lens import resolve
from .mathutil import BODY_TO_OPTICAL, rot_zyx

SCHEMA = 'vantage.critical_targets/1'
FRAME = 'vehicle (ISO 8855, rear axle)'
KINDS = ('child', 'pedestrian', 'car', 'traffic_light', 'sign', 'object', 'kerb', 'lane')
LINEAR = ('kerb', 'lane')
KIND_LABEL = {'child': 'Child', 'pedestrian': 'Pedestrian', 'car': 'Car', 'traffic_light': 'Traffic light',
              'sign': 'Sign', 'object': 'Object', 'kerb': 'Kerb', 'lane': 'Lane marking'}
AXES = ('yaw', 'pitch', 'roll', 'x', 'y', 'z')
ANGLE_AXES = AXES[:3]
AXIS_UNIT = {'yaw': '°', 'pitch': '°', 'roll': '°', 'x': 'mm', 'y': 'mm', 'z': 'mm'}
MAX_TARGETS = 32
ANG_STEP = 0.25                      # degrees, the angle scan
POS_STEPS_MM = (0.5, 1, 2, 3, 5, 7.5, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100, 125, 150, 200, 250, 300, 400, 500,
                750, 1000, 1500, 2000)
ANG_TOL = 0.01                       # degrees: bisection to this
POS_TOL = 0.0001                     # metres
JOINT_STEPS = 6                      # bisection steps of the joint share (1/64)
SURFACE_KEEP = 0.95                  # the joint grid stays this share of the way to a surface

# Typical figures (TYPICAL says where they come from); every one is editable per target.
#   height, width, base (bottom above the ground), length (kerb, marking), radius (marking; + left, 0 straight);
#   min_px: pixels the camera must put across the target's height (a marking: across its width); min_share: share of
#   the face (a kerb or marking: of its length) that must be seen; min_points: lidar points per frame the target also
#   needs (0, the default: lidars are counted but not relied on; see TYPICAL); rcs: radar cross-section, m² (0:
#   radars are not relied on).
PRESETS = {
    'child': {'height': 0.9, 'width': 0.3, 'base': 0.0, 'min_px': 24.0, 'min_share': 0.5, 'min_points': 0.0,
              'rcs': 0.3},
    'pedestrian': {'height': 1.7, 'width': 0.5, 'base': 0.0, 'min_px': 24.0, 'min_share': 0.5, 'min_points': 0.0,
                   'rcs': 0.5},
    'car': {'height': 1.5, 'width': 1.8, 'base': 0.0, 'min_px': 20.0, 'min_share': 0.5, 'min_points': 0.0,
            'rcs': 10.0},
    'traffic_light': {'height': 1.0, 'width': 0.35, 'base': 5.5, 'min_px': 15.0, 'min_share': 0.8, 'min_points': 0.0,
                      'rcs': 0.0},
    'sign': {'height': 0.6, 'width': 0.6, 'base': 2.0, 'min_px': 12.0, 'min_share': 0.8, 'min_points': 0.0,
             'rcs': 0.0},
    'object': {'height': 1.0, 'width': 0.5, 'base': 0.0, 'min_px': 20.0, 'min_share': 0.5, 'min_points': 0.0,
               'rcs': 0.0},
    'kerb': {'height': 0.15, 'width': 0.15, 'base': 0.0, 'length': 6.0, 'min_px': 5.0, 'min_share': 0.8,
             'min_points': 0.0, 'rcs': 0.0},
    'lane': {'height': 0.0, 'width': 0.15, 'base': 0.0, 'length': 30.0, 'radius': 50.0, 'min_px': 3.0,
             'min_share': 0.8, 'min_points': 0.0, 'rcs': 0.0},
}
TYPICAL = ('Typical figures, not a standard: set your detector\'s. Minimum pixel heights are the perception range\'s '
           '(car 20 px, pedestrian 24 px, sign 12 px; a child is held to the pedestrian\'s); a traffic light head '
           '(1.0 × 0.35 m, a three-aspect head) 15 px; a kerb 5 px across its 15 cm face and a lane marking 3 px '
           'across its 15 cm width are typical segmentation needs. Lidar points are counted but not required unless a '
           'target asks for them (the lidar and radar coverage needs 20 on a car, 8 on a pedestrian); radar cross-'
           'sections typical (car 10 m², adult 0.5 m², child 0.3 m²). A child 0.9 m tall is about a three-year-old; '
           'an overhead signal head\'s bottom 5.5 m up is a common clearance, a post-mounted one 2.5 to 3.5 m: check '
           'your local standard.')
ASSUMPTIONS = (
    'Standing targets are flat rectangles square to each camera\'s line of sight, kerbs vertical faces, lane markings '
    'flat strips on level ground. Line of sight is tested exactly against the simplified body and rig (the '
    'occluder the coverage analysis uses, decimated within its stated deviation) and the housings\' boxes and '
    'cylinders. Pixels are measured through the camera\'s lens model (its unit calibration when it has one) along '
    'the projected target, not with a pinhole approximation. Install errors are rigid moves of one camera; the '
    'target and the rest of the rig stay where planned. Lidar points are counted per channel as the lidar and radar '
    'coverage counts them; radar detection is its zone and range scaled by (RCS / 10 m²)^¼.')
LIMITS_TEXT = (
    'Not modelled: motion blur, exposure, glare, dirt and weather (see the exposure study), lens shading and '
    'resolution loss towards the image edge beyond the lens model\'s sampling, a target partly hidden by other '
    'traffic or the road, body motion (see the load cases). Targets stand on flat ground (z = 0): a road from the '
    'road geometry study is not applied to them. The angle scan steps 0.25° and the position scan steps '
    'grow to 500 mm: a loss narrower than a step that the target recovers from beyond it is missed. The joint check '
    'tests the 729 corner, edge and centre poses of the grid, not every pose inside it.')
# accepted range of every number (values outside are clamped)
LIMITS = {'x': (-500.0, 500.0), 'y': (-500.0, 500.0), 'height': (0.0, 50.0),
          'width': (0.01, 50.0), 'base': (0.0, 100.0), 'length': (0.1, 500.0), 'radius': (-1e5, 1e5),
          'min_px': (0.1, 10000.0), 'min_share': (0.01, 1.0), 'min_points': (0.0, 1e6), 'rcs': (0.0, 1e4)}
SETTING_LIMITS = {'install_mm': (0.0, 1000.0), 'install_deg': (0.0, 45.0), 'search_deg': (1.0, 90.0),
                  'search_mm': (5.0, 2000.0)}
MIN_RADIUS = 2.0                     # a marking's |radius| below this (and not 0) is clamped to it


# =========================================================================== targets and settings
def _new_uid() -> str:
    from .project import new_uid
    return new_uid()


def _flag(v, default: bool) -> bool:
    if isinstance(v, bool) or (isinstance(v, int) and not isinstance(v, float)):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ('1', 'true', 'yes', 'on'):
            return True
        if s in ('0', 'false', 'no', 'off'):
            return False
    return default


def _num(v, default: float, lim=None) -> float:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(v):
        return default
    if lim is not None:
        v = min(max(v, lim[0]), lim[1])
    return v


def clean_target(d) -> dict | None:
    """A target as stored, from anything (tolerant: unknown kinds become 'object', junk numbers the kind's typical
    value, numbers out of range clamped); None when d is not a mapping."""
    if not isinstance(d, dict):
        return None
    kind = d.get('kind') if d.get('kind') in KINDS else 'object'
    p = PRESETS[kind]
    t = {'uid': str(d.get('uid') or _new_uid())[:40], 'name': str(d.get('name') or KIND_LABEL[kind])[:60],
         'kind': kind, 'enabled': _flag(d.get('enabled'), True), 'focus': _flag(d.get('focus'), True)}
    for k in ('x', 'y'):
        t[k] = _num(d.get(k), 0.0, LIMITS[k])
    t['yaw'] = (_num(d.get('yaw'), 0.0) + 180.0) % 360.0 - 180.0          # any heading, wrapped to [-180, 180)
    for k in ('height', 'width', 'base', 'min_px', 'min_share', 'min_points', 'rcs'):
        t[k] = _num(d.get(k), p[k], LIMITS[k])
    t['length'] = _num(d.get('length'), p.get('length', 6.0), LIMITS['length'])
    r = _num(d.get('radius'), p.get('radius', 0.0), LIMITS['radius'])
    t['radius'] = 0.0 if abs(r) < 1e-9 else math.copysign(max(abs(r), MIN_RADIUS), r)
    if kind != 'lane' and t['height'] < 0.01:
        t['height'] = p['height'] or 0.15
    return t


@dataclass
class CriticalSettings:
    targets: list = field(default_factory=list)
    install_mm: float = 0.0            # 0: the project's install tolerance (the fleet's, as the installation check)
    install_deg: float = 0.0
    search_deg: float = 30.0
    search_mm: float = 300.0
    joint: bool = True                 # the joint (six axes at once) check
    lidars: bool = True                # count lidar points on the targets
    radars: bool = True

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d: dict | None) -> 'CriticalSettings':
        s = CriticalSettings()
        d = d if isinstance(d, dict) else {}
        out, seen, uids = [], set(), set()
        raws = d.get('targets') if isinstance(d.get('targets'), list) else []
        for raw in raws[:MAX_TARGETS]:
            t = clean_target(raw)
            if t is None:
                continue
            while t['uid'] in uids:
                t['uid'] = _new_uid()
            uids.add(t['uid'])
            base, n = t['name'], 2
            while t['name'] in seen:                  # names key the export rows: keep them unique
                t['name'] = f'{base} {n}'
                n += 1
            seen.add(t['name'])
            out.append(t)
        s.targets = out
        for k, lim in SETTING_LIMITS.items():
            if k in d:
                setattr(s, k, _num(d[k], getattr(s, k), lim))
        for k in ('joint', 'lidars', 'radars'):
            if k in d:
                setattr(s, k, _flag(d[k], getattr(s, k)))
        return s

    def target(self, uid):
        return next((t for t in self.targets if t['uid'] == uid), None)


def settings_of(project) -> CriticalSettings:
    return CriticalSettings.from_dict(getattr(project.analysis, 'critical', None))


def install_tolerance(project, st: CriticalSettings | None = None) -> dict:
    """{'mm', 'deg', 'source'}: the settings' own figures, or (0) the project's install tolerance (the fleet's,
    which the installation check also uses: 10 mm per axis and 1° by default)."""
    st = st or settings_of(project)
    from . import fleet
    fs = fleet.settings_of(project)
    mm = st.install_mm if st.install_mm > 0 else fs.pos_tol_mm
    deg = st.install_deg if st.install_deg > 0 else fs.ang_tol_deg
    src = 'critical targets' if st.install_mm > 0 and st.install_deg > 0 else \
        'project (fleet)' if st.install_mm <= 0 and st.install_deg <= 0 else 'mixed'
    return {'mm': float(mm), 'deg': float(deg), 'source': src}


# =========================================================================== placing
def vehicle_box(project) -> dict:
    """front_x, rear_x, half width and mid x of the body (the occluder bounding box's when there is no mesh: a 4 m
    car)."""
    meas = project.vehicle.measured() if project.vehicle.mesh is not None else {}
    fx, rx = meas.get('front_x', 3.2), meas.get('rear_x', -0.8)
    hw = meas.get('Width', 1.7) / 2
    return {'front_x': float(fx), 'rear_x': float(rx), 'half_w': float(hw), 'mid_x': float((fx + rx) / 2)}


def describe_place(project, t: dict, box: dict | None = None) -> str:
    """Where a target is relative to the body, in words: '2.0 m ahead of the front bumper', '0.3 m right of the
    body'."""
    b = box or vehicle_box(project)
    x, y = t['x'], t['y']
    if t['kind'] == 'kerb':
        a = math.radians(t['yaw'])
        x, y = x + math.cos(a) * t['length'] / 2, y + math.sin(a) * t['length'] / 2
    if x > b['front_x']:
        s = f'{x - b["front_x"]:.1f} m ahead of the front bumper'
    elif x < b['rear_x']:
        s = f'{b["rear_x"] - x:.1f} m behind the rear bumper'
    elif abs(y) > b['half_w']:
        return f'{abs(y) - b["half_w"]:.1f} m {"left" if y > 0 else "right"} of the body'
    else:
        return 'under the body'
    if abs(y) > 0.05:
        s += f', {abs(y):.1f} m {"left" if y > 0 else "right"}'
    return s


# (label, kind, where, distance m, lateral m, extra)
PRESET_PLACES = (
    ('Child 2 m ahead', 'child', 'front', 2.0, 0.0, {}),
    ('Child 1 m behind', 'child', 'rear', 1.0, 0.0, {}),
    ('Child beside the front door', 'child', 'left', 0.5, 1.0, {}),
    ('Pedestrian 10 m ahead', 'pedestrian', 'front', 10.0, 0.0, {}),
    ('Car 30 m ahead', 'car', 'front', 30.0, 0.0, {}),
    ('Overhead traffic light 15 m ahead', 'traffic_light', 'front', 15.0, 0.0, {}),
    ('Traffic light on a post at the stop line', 'traffic_light', 'front', 3.0, 3.0,
     {'base': 2.5, 'name': 'Traffic light at the stop line'}),
    ('Sign 15 m ahead, 3 m left', 'sign', 'front', 15.0, 3.0, {}),
    ('Kerb by the right front wheel', 'kerb', 'right', 0.3, 0.0, {}),
    ('Lane marking on a 50 m curve', 'lane', 'front', 2.0, 1.75, {}),
)


def preset(project, label_or_kind: str, existing=()) -> dict:
    """A target from PRESET_PLACES (by its label) or a bare kind, placed relative to the body; its name made unique
    among existing targets."""
    row = next((r for r in PRESET_PLACES if r[0] == label_or_kind), None)
    if row is None:
        kind = label_or_kind if label_or_kind in KINDS else 'object'
        row = (KIND_LABEL[kind], kind, 'front', 3.0, 0.0, {})
    name, kind, where, dist, lat, extra = row
    b = vehicle_box(project)
    t = dict(PRESETS[kind])
    t.update({'kind': kind, 'name': extra.get('name', name), 'yaw': 0.0})
    t.update({k: v for k, v in extra.items() if k != 'name'})
    if kind == 'kerb':                          # 3 m centred on the front wheel: parking against a kerb
        side = -1.0 if where == 'right' else 1.0
        wc = project.vehicle.wheel_centres.get('FR' if side < 0 else 'FL')
        xw = float(wc[0]) if wc is not None else b['front_x'] - 0.8
        t.update(x=xw - 1.5, y=side * (b['half_w'] + dist), length=3.0, yaw=0.0)
    elif kind == 'lane':
        t.update(x=b['front_x'] + dist, y=lat)
    elif where == 'front':
        t.update(x=b['front_x'] + dist, y=lat)
    elif where == 'rear':
        t.update(x=b['rear_x'] - dist, y=lat)
    else:
        side = -1.0 if where == 'right' else 1.0
        t.update(x=b['mid_x'] + lat, y=side * (b['half_w'] + dist))
    for k in ('x', 'y'):
        t[k] = round(float(t[k]), 3)
    names = {e.get('name') for e in existing}
    base, n = t['name'], 2
    while t['name'] in names:
        t['name'] = f'{base} {n}'
        n += 1
    return clean_target(t)


# =========================================================================== sampling
@dataclass
class Shape:
    """A target sampled for one camera: stations × nodes along the measured span."""
    nodes: np.ndarray                # (S, N, 3)
    centre: np.ndarray               # (3,)
    linear: bool
    span_m: float                    # length of the measured span (height; a marking's width)
    length_m: float                  # a kerb's or a marking's length; a standing target's width
    edges: np.ndarray | None = None  # standing: (2, 3) left and right edges at mid-height
    centreline: np.ndarray | None = None   # kerb, marking: (S, 3) station centres (on the ground)

    @property
    def flat(self) -> np.ndarray:
        return self.nodes.reshape(-1, 3)


def _arc(t: dict, s: np.ndarray):
    """Centreline points (n, 2) and unit normals to the left (n, 2) at arc lengths s from the start."""
    a0 = math.radians(t['yaw'])
    R = t.get('radius', 0.0) if t['kind'] == 'lane' else 0.0
    if abs(R) < 1e-9:
        d = np.array([math.cos(a0), math.sin(a0)])
        P = np.array([t['x'], t['y']]) + s[:, None] * d
        n = np.broadcast_to(np.array([-d[1], d[0]]), P.shape).copy()
        return P, n
    nl = np.array([-math.sin(a0), math.cos(a0)])
    C = np.array([t['x'], t['y']]) + R * nl
    a = a0 + s / R
    n = np.stack([-np.sin(a), np.cos(a)], 1)
    return C - R * n, n


def shape(t: dict, cam_pos, fine: bool = True) -> Shape:
    """Target t sampled for a camera at cam_pos (a standing target faces it). fine=False: fewer nodes (the
    optimiser)."""
    if t['kind'] in LINEAR:
        L = float(t['length'])
        S = int(min(max(math.ceil(L / (0.25 if t['kind'] == 'kerb' else 0.5)), 8), 60 if fine else 20))
        N = 9 if fine else 5
        s = (np.arange(S) + 0.5) * L / S
        P, n = _arc(t, s)
        if t['kind'] == 'kerb':
            z = np.linspace(0.0, t['height'], N)
            nodes = np.concatenate([np.repeat(P[:, None, :], N, 1), np.broadcast_to(z[None, :, None], (S, N, 1))], 2)
            span = t['height']
        else:
            w = np.linspace(-0.5, 0.5, N) * t['width']
            xy = P[:, None, :] + w[None, :, None] * n[:, None, :]
            nodes = np.concatenate([xy, np.full((S, N, 1), 0.002)], 2)
            span = t['width']
        cl = np.c_[P, np.full(S, 0.002)]
        mid = cl[S // 2] + (np.array([0, 0, t['height'] / 2]) if t['kind'] == 'kerb' else 0.0)
        return Shape(nodes.astype(float), mid, True, float(span), L, None, cl)
    nc, N = (5, 33) if fine else (3, 17)
    G = np.array([t['x'], t['y']], float)
    los = G - np.asarray(cam_pos, float)[:2]
    L = float(np.linalg.norm(los))
    if L > 1e-6:
        perp = np.array([-los[1], los[0]]) / L
    else:                                                   # straight above: face along the target's own yaw
        a = math.radians(t['yaw'] + 90.0)
        perp = np.array([math.cos(a), math.sin(a)])
    u = ((np.arange(nc) + 0.5) / nc - 0.5) * t['width']
    z = t['base'] + np.linspace(0.0, t['height'], N)
    xy = G[None, :] + u[:, None] * perp[None, :]
    nodes = np.concatenate([np.repeat(xy[:, None, :], N, 1), np.broadcast_to(z[None, :, None], (nc, N, 1))], 2)
    zm = t['base'] + t['height'] / 2
    edges = np.array([[*(G + perp * t['width'] / 2), zm], [*(G - perp * t['width'] / 2), zm]])
    return Shape(nodes.astype(float), np.array([G[0], G[1], zm]), False, float(t['height']), float(t['width']),
                 edges)


# =========================================================================== line of sight
class Occluders:
    """What blocks a camera's line of sight: the body and rig triangles (the engine's occluder for the cube maps) and
    the variant's housings, tested exactly."""

    def __init__(self, vg, project=None, variant=None):
        occ = getattr(vg, 'occ', None)
        base = getattr(occ, 'base', occ)
        tri = getattr(base, 'tri', None)
        self.tris = H.tri_info(base) if tri is not None and len(tri) else None
        hs = H.housing_set(project, variant) if project is not None else None
        self.solids = list(hs.solids) if hs is not None else []
        self.labels = ['body or rig'] + [h.label for h in self.solids]

    def los(self, pos, P, exclude_uid: str = '', group=None):
        """(blocked (K,) bool, blocker (K,) int: -1 clear, 0 body or rig, 1 + k housing k) for the straight lines from
        pos to points P (K, 3). A point within 0.02 m + 1 % of its distance of an occluder counts as clear (as the
        coverage analysis treats a cube map's depth)."""
        P = np.asarray(P, float).reshape(-1, 3)
        pos = np.asarray(pos, float)
        D = P - pos
        dist = np.linalg.norm(D, axis=1)
        Du = D / np.maximum(dist, 1e-9)[:, None]
        tmax = dist - (0.02 + 0.01 * dist)
        th = np.full(len(P), np.inf)
        hid = np.full(len(P), -1)
        for k, h in enumerate(self.solids):
            if h.uid == exclude_uid:
                continue
            w = h.centre - pos
            dc = float(np.linalg.norm(w))
            if dc > h.radius:
                cand = np.flatnonzero(Du @ (w / dc) >= math.cos(math.asin(min(h.radius / dc, 1.0))) - 1e-9)
            else:
                cand = np.arange(len(P))
            if not len(cand):
                continue
            t = H.ray(pos, Du[cand], h)
            closer = t < th[cand]
            th[cand[closer]] = t[closer]
            hid[cand[closer]] = k
        hb = th < tmax
        lim = np.where(hb, th, tmax)
        body = np.zeros(len(P), bool)
        if self.tris is not None and len(P):
            ok = lim > 1e-6
            if ok.any():
                idx = np.flatnonzero(ok)
                body[idx] = H._body_before(self.tris, pos, Du[idx], lim[idx],
                                           None if group is None else np.asarray(group)[idx])
        who = np.where(body, 0, np.where(hb, hid + 1, -1))
        return body | hb, who

    def first_hit(self, pos, d, exclude_uid: str = '', tmax: float = 10.0) -> float:
        """Distance from pos along unit direction d to the first occluder surface (inf: none within tmax)."""
        pos, d = np.asarray(pos, float), np.asarray(d, float)
        best = math.inf
        for h in self.solids:
            if h.uid != exclude_uid:
                best = min(best, float(H.ray(pos, d[None], h)[0]))
        if self.tris is not None:
            tri, lo, hi = self.tris[:3]
            end = pos + d * min(tmax, best)
            sel = ((hi >= np.minimum(pos, end) - 0.01) & (lo <= np.maximum(pos, end) + 0.01)).all(axis=1)
            q = tri[sel] - pos
            if len(q):
                e1, e2 = q[:, 1] - q[:, 0], q[:, 2] - q[:, 0]
                p = np.cross(d, e2)
                det = np.einsum('ij,ij->i', e1, p)
                with np.errstate(divide='ignore', invalid='ignore'):
                    inv = 1.0 / det
                    s = -q[:, 0]
                    u = np.einsum('ij,ij->i', s, p) * inv
                    qq = np.cross(s, e1)
                    v = (qq @ d) * inv
                    t = np.einsum('ij,ij->i', e2, qq) * inv
                    ok = (np.abs(det) > 1e-15) & (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9) & (t > 1e-7)
                if ok.any():
                    best = min(best, float(t[ok].min()))
        return best if best <= tmax else math.inf


# =========================================================================== measuring
def _rot_batch(yaw, pitch, roll) -> np.ndarray:
    """rot_zyx for arrays of angles (degrees): (M, 3, 3)."""
    y, p, r = (np.radians(np.atleast_1d(np.asarray(a, float))) for a in (yaw, pitch, roll))
    y, p, r = np.broadcast_arrays(y, p, r)
    cy, sy, cp, sp, cr, sr = np.cos(y), np.sin(y), np.cos(p), np.sin(p), np.cos(r), np.sin(r)
    R = np.empty(y.shape + (3, 3))
    R[..., 0, 0] = cy * cp
    R[..., 0, 1] = cy * sp * sr - sy * cr
    R[..., 0, 2] = cy * sp * cr + sy * sr
    R[..., 1, 0] = sy * cp
    R[..., 1, 1] = sy * sp * sr + cy * cr
    R[..., 1, 2] = sy * sp * cr - cy * sr
    R[..., 2, 0] = -sp
    R[..., 2, 1] = cp * sr
    R[..., 2, 2] = cp * cr
    return R


@dataclass
class Need:
    min_px: float
    min_share: float
    focus: bool
    linear: bool


def need_of(t: dict) -> Need:
    return Need(float(t['min_px']), float(t['min_share']), bool(t['focus']), t['kind'] in LINEAR)


def measure(intr, R, pos, sh: Shape, clear, need: Need):
    """The target's figures for M poses of one camera at once: R (M, 3, 3) body->vehicle rotations, pos (M, 3),
    clear (M, K) or (K,) line of sight per node. Returns a dict of (M,) arrays: px, seen, in_view, px_width, area,
    dist, focus, met, fail ('' or the first failed criterion: 'seen', 'pixels', 'focus')."""
    R = np.asarray(R, float).reshape(-1, 3, 3)
    pos = np.asarray(pos, float).reshape(-1, 3)
    M = max(len(R), len(pos))
    R = np.broadcast_to(R, (M, 3, 3))
    pos = np.broadcast_to(pos, (M, 3))
    S, N = sh.nodes.shape[:2]
    W2O = np.einsum('ij,mkj->mik', BODY_TO_OPTICAL, R)              # BODY_TO_OPTICAL @ R.T
    d = sh.flat[None, :, :] - pos[:, None, :]                        # (M, K, 3)
    opt = np.einsum('mij,mkj->mki', W2O, d)
    u, v, th, ok = intr.project_dirs(opt)
    with np.errstate(invalid='ignore', divide='ignore'):
        m = intr.margin(u, v, th)
    rect = intr.projection == 'rectilinear'
    front = opt[..., 2] > 1e-6
    if rect:
        m = np.where(front, m, -1.0)
    clear = np.broadcast_to(np.asarray(clear, bool).reshape(-1, S * N), (M, S * N))
    inside = ok
    vis = inside & clear
    sh3 = (M, S, N)
    u, v, m, inside, vis, front = (a.reshape(sh3) for a in (u, v, m, inside, vis, front))
    L = np.hypot(np.diff(u, axis=2), np.diff(v, axis=2))
    good = np.isfinite(L)
    if rect:
        good &= front[..., :-1] & front[..., 1:]
    L = np.where(good, L, 0.0)
    ma, mb = m[..., :-1], m[..., 1:]
    with np.errstate(invalid='ignore', divide='ignore'):
        ta = np.clip(ma / (ma - mb), 0.0, 1.0)                     # a inside, b outside: the share inside
        tb = np.clip(mb / (mb - ma), 0.0, 1.0)
    ta, tb = np.nan_to_num(ta), np.nan_to_num(tb)
    ia, ib, va, vb = inside[..., :-1], inside[..., 1:], vis[..., :-1], vis[..., 1:]
    f = np.where(va & vb, 1.0, 0.0) + np.where(va & ~vb, np.where(~ib, ta, 0.5), 0.0) + \
        np.where(vb & ~va, np.where(~ia, tb, 0.5), 0.0)
    fi = np.where(ia & ib, 1.0, 0.0) + np.where(ia & ~ib, ta, 0.0) + np.where(ib & ~ia, tb, 0.0)
    px_st = (f * L).sum(-1)                                          # (M, S)
    seen_st = f.mean(-1)
    view_st = fi.mean(-1)
    out = {'seen': seen_st.mean(-1), 'in_view': view_st.mean(-1)}
    if need.linear:
        cl = sh.centreline
        dist_st = np.linalg.norm(cl[None, :, :] - pos[:, None, :], axis=2)
        foc = (dist_st >= intr.focus_near) & (dist_st <= intr.focus_far) if need.focus else np.ones_like(dist_st, bool)
        pe = np.where(foc, px_st, 0.0)
        k = max(1, int(math.ceil(need.min_share * S - 1e-9)))
        px = -np.sort(-pe, axis=1)[:, k - 1]
        px_any = -np.sort(-px_st, axis=1)[:, k - 1]                # the same ignoring focus
        resolved = (pe >= need.min_px).mean(-1)
        step = np.hypot(*np.diff(np.stack([u[..., N // 2], v[..., N // 2]], -1), axis=1).transpose(2, 0, 1))
        step = np.concatenate([step, step[:, -1:]], 1) if S > 1 else np.zeros((M, 1))
        out.update(px=px, px_width=np.zeros(M), area=(px_st * np.nan_to_num(step)).sum(-1), resolved=resolved,
                   dist=np.linalg.norm(sh.centre[None] - pos, axis=1), focus=foc.mean(-1) >= need.min_share - 1e-9)
        met = px >= need.min_px
        fail = np.where(met, '', np.where(out['seen'] < need.min_share - 1e-9, 'seen',
                                          np.where(px_any >= need.min_px, 'focus', 'pixels')))
    else:
        px = px_st.max(-1)
        dist = np.linalg.norm(sh.centre[None] - pos, axis=1)
        foc = (dist >= intr.focus_near) & (dist <= intr.focus_far) if need.focus else np.ones(M, bool)
        e = sh.edges
        de = e[None, :, :] - pos[:, None, :]
        oe = np.einsum('mij,mkj->mki', W2O, de)
        ue, ve, _, oke = intr.project_dirs(oe)
        pw = np.where(oke.all(-1), np.hypot(ue[:, 1] - ue[:, 0], ve[:, 1] - ve[:, 0]), 0.0)
        out.update(px=px, px_width=np.nan_to_num(pw), area=px_st.mean(-1) * np.nan_to_num(pw), dist=dist, focus=foc,
                   resolved=(px_st >= need.min_px).mean(-1))
        ok_seen = out['seen'] >= need.min_share - 1e-9
        ok_px = px >= need.min_px
        met = ok_seen & ok_px & foc
        fail = np.where(met, '', np.where(~ok_seen, 'seen', np.where(~ok_px, 'pixels', 'focus')))
    out['met'] = met
    out['fail'] = fail
    return out


# =========================================================================== results
@dataclass
class Envelope:
    """How far one camera may be off on each axis (both ways) before a target is lost: degrees for yaw, pitch and
    roll, metres for x, y and z (magnitudes, the minus side in lo)."""
    lo: dict
    hi: dict
    lo_why: dict                     # axis -> 'pixels' | 'seen' | 'focus' | 'surface' | 'open'
    hi_why: dict
    joint: float | None = None       # share of the single-axis limits that holds with all six axes off at once
    tol_ok: bool | None = None       # the target holds on the joint grid at ± the install tolerance
    tol_worst: dict | None = None    # {'px', 'seen', 'pose': (dyaw, dpitch, droll, dx, dy, dz)} on that grid

    def limit(self, axis, side) -> float:
        return (self.lo if side < 0 else self.hi)[axis]

    def why(self, axis, side) -> str:
        return (self.lo_why if side < 0 else self.hi_why)[axis]


@dataclass
class Look:
    """One camera's view of one target at its planned pose."""
    uid: str
    tag: str
    slot: int
    in_view: float                   # share of the target inside the image and the lens circle
    seen: float                      # share inside and not hidden
    px: float                        # pixels across its height (a marking: across its width), the part seen
    px_width: float                  # standing: pixels across its width at mid-height (0 when an edge is out of view)
    area: float                      # ≈ pixels on it
    resolved: float                  # share of the stations reaching min_px
    dist: float
    focus: bool
    met: bool
    fail: str
    hidden: dict                     # what hides it: label -> share of its in-view nodes
    envelope: Envelope | None = None

    @property
    def unoccluded(self) -> float:
        return self.seen / self.in_view if self.in_view > 1e-9 else 0.0


@dataclass
class TargetResult:
    target: dict
    looks: list                      # [Look] per visible camera
    lidar: dict | None = None        # {'points', 'per': {tag: points}, 'needed', 'met'}
    radar: dict | None = None        # {'by': [tags], 'rcs'}
    place: str = ''

    @property
    def name(self):
        return self.target['name']

    @property
    def met_by(self) -> list:
        return [lk.tag for lk in self.looks if lk.met]

    @property
    def cameras_ok(self) -> bool:
        return any(lk.met for lk in self.looks)

    @property
    def lidar_ok(self) -> bool:
        return self.lidar is None or self.lidar['met']

    @property
    def met(self) -> bool:
        return self.cameras_ok and self.lidar_ok

    def best(self) -> Look | None:
        """The camera that comes closest: a meeting one with the most pixels, else the most pixels × seen."""
        if not self.looks:
            return None
        return max(self.looks, key=lambda lk: (lk.met, lk.px if lk.met else lk.px * min(lk.seen / max(
            self.target['min_share'], 1e-9), 1.0)))


@dataclass
class CameraEnvelope:
    """A camera's envelope: per axis and side the tightest limit over the targets it meets, and which target."""
    uid: str
    tag: str
    slot: int
    targets: list                    # names of the targets it meets
    lo: dict                         # axis -> (limit, target name, why)
    hi: dict
    binding: str = ''                # the target leaving the least margin over the install tolerance ('' none)
    margin: float | None = None      # that margin: tightest limit / install tolerance (1 = exactly the tolerance)
    tight: list = field(default_factory=list)   # [(axis, side, limit, target, sole)] tighter than the tolerance
    joint: float | None = None       # the smallest joint share over its targets
    tol_ok: bool | None = None       # every target it meets holds jointly at ± the install tolerance
    sole: list = field(default_factory=list)    # the targets it meets that no other camera meets


@dataclass
class CriticalResult:
    settings: CriticalSettings
    install: dict
    targets: list                    # [TargetResult]
    cameras: list                    # [CameraEnvelope] per visible camera
    findings: list                   # [{'severity', 'kind', 'message', 'target', 'uids'}], worst first
    metrics: dict
    variant: str = ''
    variant_uid: str = ''
    envelopes: bool = True
    labels: list = field(default_factory=list)  # occluder labels (blocker index -> name)

    def target(self, key) -> TargetResult | None:
        return next((r for r in self.targets if r.target['uid'] == key or r.name == key), None)

    def camera(self, key) -> CameraEnvelope | None:
        return next((c for c in self.cameras if c.uid == key or c.tag == key), None)

    def worst(self) -> str:
        return self.findings[0]['severity'] if self.findings else 'good'


# =========================================================================== evaluation
class _Cam:
    """One camera's nominal pose and lens."""

    def __init__(self, project, cam):
        self.cam = cam
        self.uid, self.tag, self.slot = cam.uid, cam.tag, cam.slot
        self.intr = resolve(project.module_for(cam), project.lens_for(cam))
        self.pos = cam.pos().astype(float)
        self.ang = np.array([cam.yaw, cam.pitch, cam.roll], float)
        self.R = rot_zyx(*self.ang)


def _finite_cam(c) -> bool:
    return all(math.isfinite(float(getattr(c, k))) for k in ('x', 'y', 'z', 'yaw', 'pitch', 'roll'))


def _hidden(who, inview, labels) -> dict:
    n = int(inview.sum())
    if not n:
        return {}
    out = {}
    for k in np.unique(who[inview & (who >= 0)]):
        out[labels[int(k)] if int(k) < len(labels) else 'body or rig'] = float(((who == k) & inview).sum()) / n
    return out


def _look(cm: _Cam, occ: Occluders, t: dict):
    """(Look without envelope, Shape, clear mask) of camera cm on target t at the planned pose."""
    sh = shape(t, cm.pos)
    nd = need_of(t)
    group = (np.arange(sh.nodes.shape[0])[:, None] // 8 * np.ones(sh.nodes.shape[1], int)[None]).ravel()
    blocked, who = occ.los(cm.pos, sh.flat, cm.uid, group)          # the same grouping as the envelope search
    r = measure(cm.intr, cm.R[None], cm.pos[None], sh, ~blocked, nd)
    d = sh.flat - cm.pos
    _, _, _, inview = cm.intr.project_dirs(d @ (BODY_TO_OPTICAL @ cm.R.T).T)
    lk = Look(cm.uid, cm.tag, cm.slot, float(r['in_view'][0]), float(r['seen'][0]), float(r['px'][0]),
              float(r['px_width'][0]), float(r['area'][0]), float(r['resolved'][0]), float(r['dist'][0]),
              bool(r['focus'][0]), bool(r['met'][0]), str(r['fail'][0]), _hidden(who, inview, occ.labels))
    return lk, sh, ~blocked


def _pos_steps(search_m: float) -> np.ndarray:
    s = [v / 1000.0 for v in POS_STEPS_MM if v / 1000.0 < search_m - 1e-9]
    return np.array(s + [search_m])


class _Envelopes:
    """The envelope search of one camera over the targets it meets."""

    def __init__(self, cm: _Cam, occ: Occluders, st: CriticalSettings, tol: dict):
        self.cm, self.occ, self.st, self.tol = cm, occ, st, tol
        self.ang_steps = np.arange(1, int(round(st.search_deg / ANG_STEP)) + 1) * ANG_STEP
        if not len(self.ang_steps) or self.ang_steps[-1] < st.search_deg - 1e-9:
            self.ang_steps = np.r_[self.ang_steps, st.search_deg]
        self.pos_steps = _pos_steps(st.search_mm / 1000.0)
        self._surf = {}

    def pose(self, delta):
        """(R (M, 3, 3), pos (M, 3)) for offsets delta (M, 6): degrees, metres."""
        delta = np.atleast_2d(delta)
        a = self.cm.ang[None] + delta[:, :3]
        return _rot_batch(a[:, 0], a[:, 1], a[:, 2]), self.cm.pos[None] + delta[:, 3:]

    def clear_at(self, pos, sh: Shape):
        return self.clear_many(pos, [sh])[0]

    def clear_many(self, pos, shapes):
        """Line of sight from pos to every node of several targets, in one test (rays grouped per target and per
        eight stations, so each group's triangles are found in a tight box)."""
        P = np.concatenate([sh.flat for sh in shapes])
        group = np.concatenate([(q * 4096 + np.arange(sh.nodes.shape[0])[:, None] // 8 *
                                 np.ones(sh.nodes.shape[1], int)[None]).ravel() for q, sh in enumerate(shapes)])
        clear = ~self.occ.los(pos, P, self.cm.uid, group)[0]
        cut = np.cumsum([len(sh.flat) for sh in shapes])[:-1]
        return np.split(clear, cut)

    def surface(self, k, s) -> float:
        """How far the lens may move along axis k (3..5), side s, before it is inside an occluder (inf: free)."""
        key = (k, s)
        if key not in self._surf:
            d = np.zeros(3)
            d[k - 3] = s
            self._surf[key] = self.occ.first_hit(self.cm.pos, d, self.cm.uid, tmax=self.st.search_mm / 1000.0 + 0.01)
        return self._surf[key]

    def met_at(self, delta, sh, nd, clear=None):
        R, pos = self.pose(delta)
        if clear is None:
            clear = np.stack([self.clear_at(p, sh) for p in pos]) if np.any(np.abs(np.atleast_2d(delta)[:, 3:]) >
                                                                              0) else self.clear0[id(sh)]
        return measure(self.cm.intr, R, pos, sh, clear, nd)

    def run(self, items, cancel=None):
        """items: [(t, Shape, clear0)] of the targets the camera meets. Fills an Envelope per item; returns them."""
        self.clear0 = {id(sh): c for _, sh, c in items}
        lo = [dict() for _ in items]
        hi = [dict() for _ in items]
        lo_w = [dict() for _ in items]
        hi_w = [dict() for _ in items]
        for k, axis in enumerate(AXES):
            for s in (-1, 1):
                if cancel is not None and cancel():
                    return None
                res = self._axis(items, k, s)
                for i, (lim, why) in enumerate(res):
                    (lo if s < 0 else hi)[i][axis] = lim
                    (lo_w if s < 0 else hi_w)[i][axis] = why
        envs = [Envelope(lo[i], hi[i], lo_w[i], hi_w[i]) for i in range(len(items))]
        if self.st.joint:
            for env, (t, sh, _) in zip(envs, items):
                if cancel is not None and cancel():
                    return None
                self._joint(env, sh, need_of(t))
        return envs

    def _axis(self, items, k, s):
        """[(limit, why)] per item along axis k, side s."""
        rot = k < 3
        steps = self.ang_steps if rot else self.pos_steps
        surf = math.inf if rot else self.surface(k, s)
        if not rot:
            keep = steps < surf - 1e-4
            steps = steps[keep]
        n = len(steps)
        delta = np.zeros((n, 6))
        delta[:, k] = s * steps
        out = []
        if rot:
            R, pos = self.pose(delta)
            fails = []
            for t, sh, c0 in items:
                r = measure(self.cm.intr, R, pos, sh, c0, need_of(t))
                fails.append(r)
        else:
            R, pos = self.pose(delta)
            fails = [None] * len(items)
            per = [[] for _ in items]
            alive = list(range(len(items)))
            for j in range(n):                      # outwards: a target lost at a step is not tested further
                if not alive:
                    break
                clears = self.clear_many(pos[j], [items[i][1] for i in alive])
                for i, clear in zip(list(alive), clears):
                    t, sh, _ = items[i]
                    r = measure(self.cm.intr, R[j:j + 1], pos[j:j + 1], sh, clear[None], need_of(t))
                    per[i].append((bool(r['met'][0]), str(r['fail'][0])))
                    if not r['met'][0]:
                        alive.remove(i)
            fails = per
        for i, (t, sh, c0) in enumerate(items):
            nd = need_of(t)
            if rot:
                met = fails[i]['met']
                bad = np.flatnonzero(~met)
                why_at = fails[i]['fail']
            else:
                met = np.array([m for m, _ in fails[i]], bool)
                bad = np.flatnonzero(~met)
                why_at = np.array([w for _, w in fails[i]] + [''] * (n - len(fails[i])))
            if not len(bad):
                if not rot and math.isfinite(surf) and surf < self.st.search_mm / 1000.0:
                    out.append((float(surf), 'surface'))
                else:
                    out.append((float(self.st.search_deg if rot else self.st.search_mm / 1000.0), 'open'))
                continue
            j = int(bad[0])
            a = 0.0 if j == 0 else float(steps[j - 1])
            b = float(steps[j])
            why = str(why_at[j]) or 'pixels'
            tol = ANG_TOL if rot else POS_TOL
            while b - a > tol:
                mid = 0.5 * (a + b)
                dd = np.zeros((1, 6))
                dd[0, k] = s * mid
                r = self.met_at(dd, sh, nd, None if not rot else c0)
                if r['met'][0]:
                    a = mid
                else:
                    b = mid
                    why = str(r['fail'][0]) or why
            out.append((a, why))
        return out

    def _bounds(self, env: Envelope):
        """Per axis (minus, plus) magnitudes the joint grid may use (a surface limit kept short of the surface)."""
        lo, hi = [], []
        for ax in AXES:
            for side, src, w in ((-1, env.lo, env.lo_why), (1, env.hi, env.hi_why)):
                v = src[ax] * (SURFACE_KEEP if w[ax] == 'surface' else 1.0)
                (lo if side < 0 else hi).append(v)
        return np.array(lo), np.array(hi)

    def _grid(self, lo, hi, sh, nd):
        """The 729 poses with every axis at −lo, 0 or +hi: measure() figures for all of them."""
        levels = np.array(list(itertools.product((-1, 0, 1), repeat=6)), float)      # (729, 6)
        delta = np.where(levels < 0, -lo[None], np.where(levels > 0, hi[None], 0.0))
        out = {'met': np.zeros(len(delta), bool), 'px': np.zeros(len(delta)), 'seen': np.zeros(len(delta))}
        keys = np.round(delta[:, 3:] * 1e7).astype(np.int64)
        _, inv = np.unique(keys, axis=0, return_inverse=True)
        inv = inv.ravel()
        for g in np.unique(inv):
            rows = np.flatnonzero(inv == g)
            p = self.cm.pos + delta[rows[0], 3:]
            clear = self.clear_at(p, sh) if np.any(delta[rows[0], 3:] != 0) else self.clear0[id(sh)]
            R, pos = self.pose(delta[rows])
            r = measure(self.cm.intr, R, pos, sh, clear, nd)
            for key in out:
                out[key][rows] = r[key]
        return delta, out

    def _joint(self, env: Envelope, sh: Shape, nd: Need):
        lo, hi = self._bounds(env)
        a, b = 0.0, 1.0
        _, r = self._grid(lo, hi, sh, nd)
        if r['met'].all():
            a = 1.0
        else:
            for _ in range(JOINT_STEPS):
                mid = 0.5 * (a + b)
                _, r = self._grid(lo * mid, hi * mid, sh, nd)
                if r['met'].all():
                    a = mid
                else:
                    b = mid
        env.joint = a
        tol = np.array([self.tol['deg']] * 3 + [self.tol['mm'] / 1000.0] * 3)
        lo_s, hi_s = self._surface_caps()
        delta, r = self._grid(np.minimum(tol, lo_s), np.minimum(tol, hi_s), sh, nd)
        env.tol_ok = bool(r['met'].all())
        i = int(np.argmin(np.where(r['met'], 1e9, 0.0) + r['px'] / max(nd.min_px, 1e-9) + r['seen']))
        env.tol_worst = {'px': float(r['px'][i]), 'seen': float(r['seen'][i]),
                         'pose': [round(float(v), 6) for v in delta[i]]}

    def _surface_caps(self):
        lo, hi = [math.inf] * 3, [math.inf] * 3
        for k in range(3, 6):
            for s in (-1, 1):
                v = self.surface(k, s) * SURFACE_KEEP
                (lo if s < 0 else hi).append(v)
        return np.array(lo), np.array(hi)


def _lidar_radar(project, engine, vg, variant, targets, st: CriticalSettings, cancel=None):
    """Lidar points and radar detections per target: ({uid: lidar dict}, {uid: radar dict})."""
    from . import sensor_coverage as SC
    lid, rad = {}, {}
    sens = [s for s in variant.sensors if s.visible and _finite_cam(s)]
    lidars = [s for s in sens if s.kind == 'lidar'] if st.lidars else []
    radars = [s for s in sens if s.kind == 'radar'] if st.radars else []

    def cube(pos):
        if vg is None:
            return None
        return engine.cube(pos, vg) if engine is not None else _CUBES.get(vg.occ, pos)
    per = {t['uid']: {} for t in targets}
    for s in lidars:
        if cancel is not None and cancel():
            return None
        L = SC.lidar_view(project, s, cube(s.pos()))
        ground = None
        for t in targets:
            if t['kind'] == 'lane':
                if ground is None:
                    P, ok = L.ground()
                    ground = P[ok]
                n = _points_on_marking(t, ground) * L.weight
            elif t['kind'] == 'kerb':
                sh = shape(t, s.pos(), fine=False)
                G = sh.centreline[:, :2]
                n = float(L.points(G, 0.0, t['height'], t['length'] / len(G)).sum())
            else:
                n = float(L.points(np.array([[t['x'], t['y']]]), t['base'], t['height'], t['width'])[0])
            per[t['uid']][s.tag] = float(n)
    for t in targets:
        if lidars:
            tot = float(sum(per[t['uid']].values()))
            need = float(t['min_points'])
            lid[t['uid']] = {'points': tot, 'per': per[t['uid']], 'needed': need,
                             'met': need <= 0 or tot >= need - 1e-9, 'relied_on': need > 0}
        elif t['min_points'] > 0:
            lid[t['uid']] = {'points': 0.0, 'per': {}, 'needed': float(t['min_points']), 'met': False,
                             'relied_on': True, 'no_lidar': True}
    if radars:
        objs = [(t, np.array([[t['x'], t['y'], t['base'] + t['height'] / 2]])) for t in targets
                if t['kind'] not in LINEAR and t['rcs'] > 0]
        by = {t['uid']: [] for t, _ in objs}
        for s in radars:
            Rd = SC.radar_view(project, s, cube(s.pos()), vg)       # derated by what it looks through
            for t, P in objs:
                if Rd.sees(P, t['rcs'])[0]:
                    by[t['uid']].append(s.tag)
        for t, _ in objs:
            rad[t['uid']] = {'by': by[t['uid']], 'rcs': float(t['rcs'])}
    return lid, rad


def _points_on_marking(t, ground: np.ndarray) -> float:
    """Ground points (n, 3) within the marking's strip: the count (one traced beam each)."""
    if not len(ground):
        return 0.0
    L = float(t['length'])
    s = np.linspace(0.0, L, max(int(L / 0.05), 2) + 1)
    C, _ = _arc(t, s)
    lo, hi = C.min(0) - t['width'], C.max(0) + t['width']
    g = ground[:, :2]
    g = g[((g >= lo) & (g <= hi)).all(1)]
    if not len(g):
        return 0.0
    A, B = C[:-1], C[1:]
    AB = B - A
    den = np.maximum((AB * AB).sum(1), 1e-12)
    n = 0
    for i0 in range(0, len(g), 2048):
        q = g[i0:i0 + 2048]
        w = ((q[:, None, :] - A[None]) * AB[None]).sum(-1) / den[None]
        w = np.clip(w, 0.0, 1.0)
        dmin = np.linalg.norm(q[:, None, :] - (A[None] + w[..., None] * AB[None]), axis=-1).min(1)
        n += int((dmin <= t['width'] / 2).sum())
    return float(n)


def _project_for(project, variant):
    if variant is None or variant is project.variant:
        return project
    q = copy.copy(project)
    q.variants = list(project.variants) + ([] if any(v is variant for v in project.variants) else [variant])
    q.active_uid = variant.uid
    return q


def run(project, engine=None, settings: CriticalSettings | None = None, variant=None, progress=None, cancel=None,
        envelopes: bool = True) -> CriticalResult | None:
    """The critical targets of a variant (the active one by default): per target and visible camera what it sees,
    the lidars' points and the radars' detections, and (envelopes) every camera's mounting envelope. Returns None
    when cancelled."""
    from .analysis import Engine
    t0 = time.perf_counter()
    st = settings or settings_of(project)
    p = _project_for(project, variant)
    v = p.variant
    eng = engine or Engine()
    vg = eng.vehicle_geom(p, v)
    occ = Occluders(vg, p, v)
    tol = install_tolerance(p, st)
    targets = [t for t in st.targets if t['enabled']]
    cams = [_Cam(p, c) for c in v.cameras if c.visible and _finite_cam(c)]
    box = vehicle_box(p)

    def prog(f):
        if progress is not None:
            progress(min(max(f, 0.0), 1.0))
    results, shapes = [], {}
    for i, t in enumerate(targets):
        if cancel is not None and cancel():
            return None
        looks = []
        for cm in cams:
            lk, sh, c0 = _look(cm, occ, t)
            looks.append(lk)
            shapes[(t['uid'], cm.uid)] = (sh, c0)
        results.append(TargetResult(t, looks, place=describe_place(p, t, box)))
        prog(0.1 * (i + 1) / max(len(targets), 1))
    lr = _lidar_radar(p, eng, vg, v, targets, st, cancel)
    if lr is None:
        return None
    for r in results:
        r.lidar = lr[0].get(r.target['uid'])
        r.radar = lr[1].get(r.target['uid'])
    prog(0.2)
    cam_env = []
    for ci, cm in enumerate(cams):
        if cancel is not None and cancel():
            return None
        mine = [(r, next(lk for lk in r.looks if lk.uid == cm.uid)) for r in results]
        mine = [(r, lk) for r, lk in mine if lk.met]
        if envelopes and mine:
            items = [(r.target,) + shapes[(r.target['uid'], cm.uid)] for r, _ in mine]
            envs = _Envelopes(cm, occ, st, tol).run(items, cancel)
            if envs is None:
                return None
            for (r, lk), env in zip(mine, envs):
                lk.envelope = env
        sole = {r.name for r, _ in mine if r.met_by == [cm.tag]}
        cam_env.append(_camera_envelope(cm, mine, tol, sole))
        prog(0.2 + 0.8 * (ci + 1) / max(len(cams), 1))
    res = CriticalResult(st, tol, results, cam_env, [], {}, v.name, v.uid, envelopes, list(occ.labels))
    res.findings = _findings(res)
    res.metrics = {'seconds': time.perf_counter() - t0, 'targets': len(results),
                   'resolved': sum(r.met for r in results), 'cameras': len(cams),
                   'tight': sum(len(c.tight) > 0 for c in cam_env)}
    prog(1.0)
    with _lock:
        _RESULTS[_key(p, v, st, vg)] = res
        while len(_RESULTS) > 4:
            _RESULTS.popitem(last=False)
    return res


def _camera_envelope(cm: _Cam, mine, tol, sole=()) -> CameraEnvelope:
    """mine: [(TargetResult, Look)] of the targets the camera meets; sole: names of the targets no other camera
    meets."""
    lo, hi, tight = {}, {}, []
    best = (math.inf, '')
    for ax in AXES:
        t_unit = tol['deg'] if ax in ANGLE_AXES else tol['mm'] / 1000.0
        for side, out in ((-1, lo), (1, hi)):
            cands = [(lk.envelope.limit(ax, side), r.name, lk.envelope.why(ax, side)) for r, lk in mine
                     if lk.envelope is not None]
            if not cands:
                continue
            req = [c for c in cands if c[2] not in ('surface', 'open')]      # a surface or open limit loses nothing
            pick = min(req or cands, key=lambda c: c[0])
            if not req:
                pick = (pick[0], '', pick[2])                                 # no target sets it
            out[ax] = pick
            if req:
                ratio = pick[0] / max(t_unit, 1e-12)
                if ratio < best[0]:
                    best = (ratio, pick[1])
            for v, name, _ in req:
                if v < t_unit - 1e-12:
                    tight.append((ax, side, v, name, name in sole))
    tight.sort(key=lambda x: (AXES.index(x[0]), x[1], x[2]))
    joints = [lk.envelope.joint for _, lk in mine if lk.envelope is not None and lk.envelope.joint is not None]
    tols = [lk.envelope.tol_ok for _, lk in mine if lk.envelope is not None and lk.envelope.tol_ok is not None]
    return CameraEnvelope(cm.uid, cm.tag, cm.slot, [r.name for r, _ in mine], lo, hi, best[1],
                          None if not best[1] else float(best[0]), tight, min(joints) if joints else None,
                          all(tols) if tols else None, [r.name for r, _ in mine if r.name in sole])


def _fmt_limit(ax, v) -> str:
    return f'{v:.2f}°' if ax in ANGLE_AXES else f'{v * 1000:.1f} mm'


def _findings(res: CriticalResult) -> list:
    out = []
    tol = res.install
    for r in res.targets:
        t = r.target
        if not r.cameras_ok:
            b = r.best()
            if b is None:
                msg = f'No camera can resolve the critical target "{t["name"]}" ({r.place}): the variant has no camera.'
            else:
                why = {'seen': f'only {100 * b.seen:.0f} % of it is seen (needs {100 * t["min_share"]:.0f} %)',
                       'pixels': f'{b.px:.0f} px across it (needs {t["min_px"]:g} px)',
                       'focus': 'it is outside the lens\'s focus range'}.get(b.fail, 'it is not in view')
                if b.in_view <= 1e-9:
                    msg = (f'No camera resolves the critical target "{t["name"]}" ({r.place}): it is in no camera\'s '
                           f'image.')
                else:
                    msg = (f'No camera resolves the critical target "{t["name"]}" ({r.place}): the closest, {b.tag}, '
                           f'sees it with {why}.') if b.fail == 'pixels' else \
                        (f'No camera resolves the critical target "{t["name"]}" ({r.place}): for the closest, {b.tag}, '
                         f'{why}.')
            out.append({'severity': 'serious', 'kind': 'unresolved', 'message': msg, 'target': t['uid'],
                        'uids': [b.uid] if b is not None else [], 'where': (t['x'], t['y'])})
        elif r.lidar_ok:
            out.append({'severity': 'good', 'kind': 'resolved', 'target': t['uid'], 'uids': [],
                        'message': f'"{t["name"]}" is resolved by {", ".join(r.met_by)}' + (
                            f' and {r.lidar["points"]:.0f} lidar points' if r.lidar and r.lidar.get('relied_on')
                            else '') + '.', 'where': (t['x'], t['y'])})
        if r.lidar is not None and r.lidar.get('relied_on') and not r.lidar['met']:
            msg = (f'No lidar on the variant for "{t["name"]}", which needs {t["min_points"]:g} lidar points.'
                   if r.lidar.get('no_lidar') else
                   f'The lidars put {r.lidar["points"]:.1f} points per frame on "{t["name"]}", fewer than the '
                   f'{t["min_points"]:g} it needs.')
            out.append({'severity': 'warning', 'kind': 'lidar', 'message': msg, 'target': t['uid'], 'uids': [],
                        'where': (t['x'], t['y'])})
    by_name = {r.name: r for r in res.targets}
    tol_txt = f'the install tolerance (±{tol["mm"]:g} mm, ±{tol["deg"]:g}°)'
    for c in res.cameras:
        for sole in (True, False):
            rows = [x for x in c.tight if x[4] == sole]
            if not rows:
                continue
            parts = [f'{ax} {"−" if side < 0 else "+"}{_fmt_limit(ax, lim)} ({name}' +
                     ('' if sole else ', also ' + ', '.join(t for t in by_name[name].met_by if t != c.tag)) + ')'
                     for ax, side, lim, name, _ in rows]
            if sole:
                names = sorted({x[3] for x in rows})
                quoted = ', '.join('"' + n + '"' for n in names)
                msg = (f'{c.tag} alone resolves {quoted}, and its mount may be off by less than {tol_txt} before '
                       f'losing {"it" if len(names) == 1 else "them"}: ' + ', '.join(parts) + '.')
                out.append({'severity': 'warning', 'kind': 'tight', 'target': by_name[names[0]].target['uid'],
                            'uids': [c.uid], 'message': msg})
            else:
                out.append({'severity': 'warning', 'kind': 'tight_shared', 'target': '', 'uids': [c.uid],
                            'message': f'{c.tag}\'s envelope is tighter than {tol_txt} for targets other cameras '
                                       f'also resolve: ' + ', '.join(parts) + '.'})
    order = {'critical': 0, 'serious': 1, 'warning': 2, 'good': 3}
    out.sort(key=lambda f: order.get(f['severity'], 9))
    return out


# =========================================================================== the Issues list (cached)
_lock = threading.Lock()
_NOMINAL = OrderedDict()
_RESULTS = OrderedDict()
_LIDAR = OrderedDict()
_LOOKS = OrderedDict()


class _LazyCubes:
    def __init__(self):
        self._c = None

    def get(self, geom, pos):
        from .occlusion import CubeCache
        if self._c is None:
            self._c = CubeCache(8)
        return self._c.get(geom, pos)


_CUBES = _LazyCubes()


def _key(project, variant, st: CriticalSettings, vg) -> str:
    """Everything the study reads, as text."""
    cams = [H._cam_key(project, c) + (c.visible,) for c in variant.cameras]
    sens = [(s.uid, s.tag, s.kind, s.model_id, s.x, s.y, s.z, s.yaw, s.pitch, s.roll, s.visible) for s in
            variant.sensors]
    return json.dumps({'cams': cams, 'sensors': sens, 'settings': st.to_dict(),
                       'housings': repr(H.key(project, variant)), 'vg': getattr(vg, 'version', None),
                       'tol': install_tolerance(project, st)}, sort_keys=True, default=str)


def cached(project, variant=None, vg=None) -> CriticalResult | None:
    """The last full run's result for exactly this design and these settings, if any."""
    v = variant or project.variant
    with _lock:
        return _RESULTS.get(_key(project, v, settings_of(project), vg))


def latest(project, vg=None) -> CriticalResult | None:
    """The newest result for exactly this design: the last full run's, else the Issues list's check (no envelopes);
    None when neither is cached (the main views draw the targets without verdicts then)."""
    v = project.variant
    key = _key(project, v, settings_of(project), vg)
    with _lock:
        return _RESULTS.get(key) or _NOMINAL.get(key)


def issues(project, vg=None) -> list:
    """For the Issues list: a critical target no camera resolves (serious), a lidar requirement not met (warning),
    and (from the last full run of this design, when there is one) a camera whose mounting envelope is tighter than
    the install tolerance (warning). The camera check is cached on the design (a few ms per camera and target)."""
    from .rules import Issue
    st = settings_of(project)
    targets = [t for t in st.targets if t['enabled']]
    if not targets:
        return []
    v = project.variant
    key = _key(project, v, st, vg)
    with _lock:
        res = _RESULTS.get(key) or _NOMINAL.get(key)
    if res is None:
        occ = None
        cams = [c for c in v.cameras if c.visible and _finite_cam(c)]
        box = vehicle_box(project)
        hk = repr(H.key(project, v))
        results = []
        for t in targets:
            looks = []
            tk = json.dumps(t, sort_keys=True)
            for c in cams:                        # one camera moved: only its looks are measured again
                lkey = (H._cam_key(project, c), tk, hk, getattr(vg, 'version', None))
                with _lock:
                    lk = _LOOKS.get(lkey)
                if lk is None:
                    occ = occ or Occluders(vg, project, v)
                    lk = _look(_Cam(project, c), occ, t)[0]
                    with _lock:
                        _LOOKS[lkey] = lk
                        while len(_LOOKS) > 1024:
                            _LOOKS.popitem(last=False)
                looks.append(copy.copy(lk))
            results.append(TargetResult(t, looks, place=describe_place(project, t, box)))
        if any(t['min_points'] > 0 for t in targets):
            lkey = json.dumps({'sensors': [(s.uid, s.kind, s.model_id, s.x, s.y, s.z, s.yaw, s.pitch, s.roll,
                                            s.visible) for s in v.sensors],
                               'targets': [{k: t[k] for k in ('uid', 'kind', 'x', 'y', 'yaw', 'height', 'width',
                                                              'base', 'length', 'radius', 'min_points')}
                                           for t in targets],
                               'vg': getattr(vg, 'version', None), 'lidars': st.lidars}, sort_keys=True, default=str)
            with _lock:
                lid = _LIDAR.get(lkey)
            if lid is None:
                lid = _lidar_radar(project, None, vg, v, targets, CriticalSettings.from_dict(
                    dict(st.to_dict(), radars=False)))[0]
                with _lock:
                    _LIDAR[lkey] = lid
                    while len(_LIDAR) > 8:
                        _LIDAR.popitem(last=False)
            for r in results:
                r.lidar = lid.get(r.target['uid'])
        res = CriticalResult(st, install_tolerance(project, st), results, [], [], {}, v.name, v.uid, False)
        res.findings = _findings(res)
        with _lock:
            _NOMINAL[key] = res
            while len(_NOMINAL) > 16:
                _NOMINAL.popitem(last=False)
    out = []
    for f in res.findings:
        if f['severity'] == 'good' or f['kind'] == 'tight_shared':      # another camera covers those targets
            continue
        t = st.target(f['target']) if f['target'] else None
        obj = f'{t["name"]} · critical target' if t else ', '.join(
            c.tag for c in v.cameras if c.uid in f['uids']) or 'critical targets'
        out.append(Issue(f['severity'], f['message'], obj, list(f['uids']), '', f.get('where')))
    return out


def clear_cache():
    with _lock:
        _NOMINAL.clear()
        _RESULTS.clear()
        _LIDAR.clear()
        _LOOKS.clear()


# =========================================================================== the optimiser
class TargetCheck:
    """Whether a set of camera poses meets every critical target: for the placement optimiser. Line of sight from a
    camera's position is computed once per position (cached); rotations and the tolerance samples reuse it (a few
    millimetres of position error do not change what the body hides, as the optimiser's coverage score assumes)."""

    def __init__(self, project, vg, variant=None, settings: CriticalSettings | None = None):
        st = settings or settings_of(project)
        self.project = project
        self.variant = variant or project.variant
        self.targets = [t for t in st.targets if t['enabled']]
        self.occ = Occluders(vg, project, self.variant)
        self._intr = {}
        self._clear = OrderedDict()

    def __bool__(self):
        return bool(self.targets)

    def _intrinsics(self, cam):
        key = (cam.uid, cam.module_id, cam.lens_id)
        if key not in self._intr:
            self._intr[key] = resolve(self.project.module_for(cam), self.project.lens_for(cam))
        return self._intr[key]

    def _shapes(self, cam):
        key = (cam.uid, round(cam.x, 4), round(cam.y, 4), round(cam.z, 4))
        hit = self._clear.get(key)
        if hit is None:
            pos = cam.pos().astype(float)
            hit = []
            for t in self.targets:
                sh = shape(t, pos, fine=False)
                hit.append((sh, ~self.occ.los(pos, sh.flat, cam.uid)[0]))
            self._clear[key] = hit
            while len(self._clear) > 512:
                self._clear.popitem(last=False)
        return hit

    def met(self, cams, poses=None) -> np.ndarray:
        """(T,) bool: each target met by some camera. poses: {uid: (R, pos)} overriding a camera's planned pose
        (line of sight stays the planned position's)."""
        out = np.zeros(len(self.targets), bool)
        for c in cams:
            if not c.visible:
                continue
            intr = self._intrinsics(c)
            R, pos = (poses or {}).get(c.uid, (None, None))
            R = rot_zyx(c.yaw, c.pitch, c.roll) if R is None else R
            pos = c.pos().astype(float) if pos is None else np.asarray(pos, float)
            for i, (t, (sh, clear)) in enumerate(zip(self.targets, self._shapes(c))):
                if not out[i]:
                    out[i] = bool(measure(intr, R[None], pos[None], sh, clear, need_of(t))['met'][0])
            if out.all():
                break
        return out

    def names(self, mask) -> list:
        return [t['name'] for t, m in zip(self.targets, mask) if not m]


# =========================================================================== output
def _r(v, nd=4):
    return None if v is None or not math.isfinite(float(v)) else round(float(v), nd)


def _lim_doc(ax, v):
    return _r(v, 3) if ax in ANGLE_AXES else _r(v * 1000.0, 2)


def _env_doc(env: Envelope | None) -> dict | None:
    if env is None:
        return None
    return {'minus': {ax: _lim_doc(ax, env.lo[ax]) for ax in AXES},
            'plus': {ax: _lim_doc(ax, env.hi[ax]) for ax in AXES},
            'minus_limited_by': dict(env.lo_why), 'plus_limited_by': dict(env.hi_why),
            'joint_share': _r(env.joint, 4) if env.joint is not None else None,
            'holds_at_install_tolerance': env.tol_ok,
            'worst_at_install_tolerance': None if env.tol_worst is None else {
                'px': _r(env.tol_worst['px'], 2), 'seen_pct': _r(100 * env.tol_worst['seen'], 2),
                'pose': dict(zip(('yaw_deg', 'pitch_deg', 'roll_deg', 'x_mm', 'y_mm', 'z_mm'),
                                 [_r(v, 4) for v in env.tol_worst['pose'][:3]] +
                                 [_r(v * 1000.0, 3) for v in env.tol_worst['pose'][3:]]))}}


def summary(res: CriticalResult) -> dict:
    """Compact figures (the assistant, the batch report header)."""
    return {'variant': res.variant, 'targets': res.metrics.get('targets', len(res.targets)),
            'resolved': sum(r.met for r in res.targets),
            'install_tolerance': {'mm': res.install['mm'], 'deg': res.install['deg']},
            'per_target': [{'name': r.name, 'kind': r.target['kind'], 'where': r.place, 'resolved': r.met,
                            'cameras': r.met_by, 'best_px': _r(r.best().px, 1) if r.best() else None,
                            'needed_px': r.target['min_px'],
                            'lidar_points': _r(r.lidar['points'], 1) if r.lidar else None,
                            'radars': r.radar['by'] if r.radar else None} for r in res.targets],
            'per_camera': [{'camera': c.tag, 'binding_target': c.binding or None,
                            'margin_over_tolerance': _r(c.margin, 2) if c.margin is not None else None,
                            'tighter_than_tolerance': [f'{ax} {"-" if s < 0 else "+"}{_fmt_limit(ax, v)} ({n}'
                                                       f'{"" if sole else ", other cameras also resolve it"})'
                                                       for ax, s, v, n, sole in c.tight],
                            'sole_camera_for': c.sole,
                            'joint_share': _r(c.joint, 3) if c.joint is not None else None,
                            'holds_at_tolerance': c.tol_ok} for c in res.cameras if c.targets],
            'findings': [{'severity': f['severity'], 'message': f['message']} for f in res.findings
                         if f['severity'] != 'good'],
            'assumptions': ASSUMPTIONS}


def to_doc(res: CriticalResult, project=None) -> dict:
    """The JSON export: every target with what each camera sees of it, the lidars' points and the radars, every
    camera's envelope per target and its tightest envelope, and the findings."""
    tgt = []
    for r in res.targets:
        t = r.target
        tgt.append({
            'name': t['name'], 'kind': t['kind'], 'enabled': t['enabled'], 'where': r.place,
            'x': _r(t['x']), 'y': _r(t['y']), 'heading_deg': _r(t['yaw'], 3), 'height': _r(t['height']),
            'width': _r(t['width']), 'above_ground': _r(t['base']),
            'length': _r(t['length']) if t['kind'] in LINEAR else None,
            'radius': _r(t['radius']) if t['kind'] == 'lane' else None,
            'requirement': {'min_px': t['min_px'], 'measured_across': 'width' if t['kind'] == 'lane' else 'height',
                            'min_share_pct': _r(100 * t['min_share'], 2), 'in_focus': t['focus'],
                            'lidar_points': t['min_points'], 'radar_rcs_m2': t['rcs']},
            'resolved': r.met, 'cameras_meeting': r.met_by,
            'cameras': [{'camera': lk.tag, 'in_view_pct': _r(100 * lk.in_view, 2), 'seen_pct': _r(100 * lk.seen, 2),
                         'unoccluded_pct': _r(100 * lk.unoccluded, 2), 'hidden_by': {k: _r(100 * v, 2) for k, v in
                                                                                   lk.hidden.items()},
                         'px': _r(lk.px, 2), 'px_width': _r(lk.px_width, 2), 'pixels_on_target': _r(lk.area, 0),
                         'resolved_length_pct': _r(100 * lk.resolved, 2) if t['kind'] in LINEAR else None,
                         'distance_m': _r(lk.dist, 3), 'in_focus': lk.focus, 'met': lk.met,
                         'fails_on': lk.fail or None, 'envelope': _env_doc(lk.envelope)} for lk in r.looks],
            'lidar': None if r.lidar is None else {'points_per_frame': _r(r.lidar['points'], 2),
                                                   'per_lidar': {k: _r(v, 2) for k, v in r.lidar['per'].items()},
                                                   'needed': r.lidar['needed'], 'met': r.lidar['met']},
            'radar': None if r.radar is None else {'detected_by': r.radar['by'], 'rcs_m2': r.radar['rcs']}})
    cams = []
    for c in res.cameras:
        cams.append({'camera': c.tag, 'targets_met': c.targets, 'binding_target': c.binding or None,
                     'margin_over_tolerance': _r(c.margin, 3) if c.margin is not None else None,
                     'envelope': {'minus': {ax: {'limit': _lim_doc(ax, v[0]), 'target': v[1], 'limited_by': v[2]}
                                            for ax, v in c.lo.items()},
                                  'plus': {ax: {'limit': _lim_doc(ax, v[0]), 'target': v[1], 'limited_by': v[2]}
                                           for ax, v in c.hi.items()}},
                     'tighter_than_tolerance': [{'axis': ax, 'side': '-' if s < 0 else '+',
                                                 'limit': _lim_doc(ax, v), 'target': n, 'only_this_camera': sole}
                                                for ax, s, v, n, sole in c.tight],
                     'only_camera_for': c.sole,
                     'joint_share': _r(c.joint, 4) if c.joint is not None else None,
                     'holds_at_install_tolerance': c.tol_ok})
    st = res.settings
    return {'schema': SCHEMA, 'project': getattr(project, 'name', ''), 'variant': res.variant, 'frame': FRAME,
            'units': {'lengths': 'm', 'envelope_angles': 'deg (yaw, pitch, roll: Z-Y-X, pitch positive down)',
                      'envelope_positions': 'mm (x forward, y left, z up)', 'shares': 'percent'},
            'install_tolerance': {'mm': res.install['mm'], 'deg': res.install['deg'], 'source': res.install['source']},
            'search': {'deg': st.search_deg, 'mm': st.search_mm, 'joint': st.joint},
            'envelopes': res.envelopes, 'assumptions': ASSUMPTIONS, 'limits': LIMITS_TEXT, 'typical': TYPICAL,
            'limited_by': {'pixels': 'fewer pixels across the target than needed', 'seen': 'less of it seen than '
                           'needed', 'focus': 'outside the focus range', 'surface': 'the lens would be inside the '
                           'body, the rig or a housing (no loss of the target)', 'open': 'no loss within the search '
                           'range'},
            'targets': tgt, 'cameras': cams,
            'findings': [{'severity': f['severity'], 'kind': f['kind'], 'message': f['message']} for f in
                         res.findings]}


CSV_COLUMNS = ['target', 'kind', 'x_m', 'y_m', 'camera', 'in_view_pct', 'seen_pct', 'px', 'needed_px', 'distance_m',
               'in_focus', 'met'] + [f'{ax}_{s}_{"deg" if ax in ANGLE_AXES else "mm"}' for ax in AXES
                                     for s in ('minus', 'plus')] + ['joint_share', 'holds_at_install_tolerance']
SHEET_COLUMNS = ('camera', 'axis', 'unit', 'minus', 'minus_target', 'minus_limited_by', 'plus', 'plus_target',
                 'plus_limited_by', 'install_tolerance', 'verdict')


def to_csv(res: CriticalResult) -> str:
    """One row per target and camera: what it sees, whether it meets the target and its envelope for it."""
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator='\n')
    w.writerow(CSV_COLUMNS)
    b = lambda v: '' if v is None else v                              # noqa: E731
    for r in res.targets:
        for lk in r.looks:
            env = lk.envelope
            lims = []
            for ax in AXES:
                for side in (-1, 1):
                    lims.append(b(_lim_doc(ax, env.limit(ax, side))) if env else '')
            w.writerow([r.name, r.target['kind'], _r(r.target['x'], 3), _r(r.target['y'], 3), lk.tag,
                        _r(100 * lk.in_view, 1), _r(100 * lk.seen, 1), _r(lk.px, 1), r.target['min_px'],
                        _r(lk.dist, 2), lk.focus, lk.met] + lims +
                       [b(_r(env.joint, 3)) if env else '', b(env.tol_ok) if env else ''])
    return buf.getvalue()


def sheet_csv(res: CriticalResult) -> str:
    """The mechanical sheet: per camera and axis how far the mount may be off each way (the tightest over the targets
    it meets), which target limits it, and the verdict against the install tolerance."""
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator='\n')
    w.writerow(SHEET_COLUMNS)
    for c in res.cameras:
        if not c.targets:
            continue
        for ax in AXES:
            unit = 'deg' if ax in ANGLE_AXES else 'mm'
            tol = res.install['deg'] if ax in ANGLE_AXES else res.install['mm']
            lo, hi = c.lo.get(ax), c.hi.get(ax)
            if lo is None or hi is None:
                continue
            tight = any(a == ax for a, *_ in c.tight)
            w.writerow([c.tag, ax, unit, _lim_doc(ax, lo[0]), lo[1], lo[2], _lim_doc(ax, hi[0]), hi[1], hi[2], tol,
                        'tighter than the tolerance' if tight else 'ok'])
    return buf.getvalue()


def _bgr(h):
    return int(h[5:7], 16), int(h[3:5], 16), int(h[1:3], 16)


GOOD, WARN, BAD = '#0ca30c', '#fab219', '#d03b3b'          # the theme's status colours
CAM_COLORS = ['#3987e5', '#d95926', '#199e70', '#c98500', '#d55181', '#008300', '#9085e9', '#e66767']   # dark theme


_ASCII = {'·': '-', '±': '+/-', '°': ' deg', '−': '-', '–': '-', '≥': '>=', '×': 'x', '²': '2', '’': "'"}


def _ascii(t: str) -> str:
    """OpenCV's Hershey fonts draw ASCII only."""
    for k, v in _ASCII.items():
        t = t.replace(k, v)
    return t.encode('ascii', 'replace').decode('ascii')


def envelope_png(res: CriticalResult, width=1280) -> bytes:
    """The mechanical sheet as an image: per camera that resolves a target, a bar per axis from its minus to its plus
    limit (each row on a linear scale of ± 5 × the install tolerance; a longer bar is clipped with an arrow), the
    install tolerance dashed; green at least the tolerance, red tighter, grey where the body's surface stops the move;
    the limits and the target that sets the tighter side on the right."""
    import cv2
    cams = [c for c in res.cameras if c.targets]
    head, row, gap, pad = 64, 26, 34, 18
    H_ = head + max(len(cams), 1) * (row * 6 + gap + 22) + 20
    img = np.full((H_, width, 3), (30, 29, 28), np.uint8)
    font, aa = cv2.FONT_HERSHEY_SIMPLEX, cv2.LINE_AA

    def text(t, x, y, col=(220, 219, 214), s=0.45, th=1):
        cv2.putText(img, _ascii(t), (int(x), int(y)), font, s, col, th, aa)
    tol = res.install
    text(f'Mounting envelopes  {res.variant}', pad, 26, (236, 235, 230), 0.65)
    text(f'Install tolerance +/-{tol["mm"]:g} mm, +/-{tol["deg"]:g} deg (dashed; each row spans 5 x it). Bars: how '
         f'far each axis alone may be off before a critical target is lost.', pad, 48, (165, 164, 160), 0.42)
    x0, x1 = 90, width - 540
    cx = (x0 + x1) / 2
    half = (x1 - x0) / 2 - 10
    y = head
    if not cams:
        text('No camera resolves a critical target.', pad, y + 20, (165, 164, 160))
    for c in cams:
        text(f'{c.tag}' + (f'  binding: {c.binding}' if c.binding else '') +
             (f'  joint {100 * c.joint:.0f} %' if c.joint is not None else '') +
             ('' if c.tol_ok is None else ('  holds at the tolerance' if c.tol_ok else '  fails at the tolerance')) +
             (f'  only camera for: {", ".join(c.sole)}' if c.sole else ''), pad, y + 14, (236, 235, 230), 0.5)
        y += 22
        for ax in AXES:
            lo, hi = c.lo.get(ax), c.hi.get(ax)
            t_unit = tol['deg'] if ax in ANGLE_AXES else tol['mm'] / 1000.0
            span = 5.0 * max(t_unit, 1e-9)
            k = half / span
            text(ax, pad, y + row / 2 + 5, (195, 194, 183))
            cv2.line(img, (int(cx - half), y + row // 2), (int(cx + half), y + row // 2), (62, 61, 58), 1, aa)
            if lo is not None and hi is not None:
                for sgn, (v, name, why) in ((-1, lo), (1, hi)):
                    col = _bgr('#8a8984' if why == 'surface' else GOOD if why == 'open' or v >= t_unit - 1e-12
                               else BAD)
                    L = min(v, span) * k
                    xa, xb = (cx - L, cx) if sgn < 0 else (cx, cx + L)
                    cv2.rectangle(img, (int(xa), y + 6), (int(max(xb, xa + 1)), y + row - 6), col, -1)
                    if v > span:
                        tip = int(cx + sgn * (half + 9))
                        cv2.fillPoly(img, [np.array([[tip, y + row // 2], [int(cx + sgn * half), y + 5],
                                                     [int(cx + sgn * half), y + row - 5]], np.int32)], col, aa)
                lost = [q for q in (lo, hi) if q[2] not in ('surface', 'open')]
                who = (min(lost, key=lambda q: q[0])[1] + ' (' + min(lost, key=lambda q: q[0])[2] + ')') if lost \
                    else ('body surface' if 'surface' in (lo[2], hi[2]) else 'no loss within the search')
                text(f'-{_fmt_limit(ax, lo[0])} / +{_fmt_limit(ax, hi[0])}', x1 + 24, y + row / 2 + 5,
                     (220, 219, 214), 0.42)
                text(who, x1 + 250, y + row / 2 + 5, (150, 148, 142), 0.4)
            cv2.line(img, (int(cx), y + 3), (int(cx), y + row - 3), (200, 199, 194), 1, aa)
            for sgn in (-1, 1):
                xt = int(cx + sgn * t_unit * k)
                for yy in range(y + 2, y + row - 2, 6):
                    cv2.line(img, (xt, yy), (xt, min(yy + 3, y + row - 2)), (150, 148, 142), 1, aa)
            y += row
        y += gap
    ok, buf = cv2.imencode('.png', img)
    return buf.tobytes()


def plan_png(res: CriticalResult, project, size=900) -> bytes:
    """The targets on a plan of the vehicle: forward up, left to the left, scaled to hold every target; each target
    green when resolved, red when not, with a line to each camera that meets it."""
    import cv2
    var = next((v for v in project.variants if v.uid == res.variant_uid), project.variant)
    pts = [(c.x, c.y) for c in var.cameras if c.visible and _finite_cam(c)]
    b = vehicle_box(project)
    pts += [(b['front_x'], b['half_w']), (b['rear_x'], -b['half_w'])]
    shapes = []
    for r in res.targets:
        t = r.target
        if t['kind'] in LINEAR:
            sh = shape(t, np.zeros(3), fine=False)
            line = sh.centreline[:, :2]
        else:
            line = np.array([[t['x'], t['y']]])
        shapes.append(line)
        pts += [tuple(p) for p in line]
    P = np.array(pts, float)
    lo, hi = P.min(0) - 1.5, P.max(0) + 1.5
    head = 40
    s = (size - 40) / max(hi[0] - lo[0], hi[1] - lo[1])
    img = np.full((size + head, size, 3), (30, 29, 28), np.uint8)
    cxm, cym = (lo + hi) / 2

    def at(x, y):
        return int(size / 2 - (y - cym) * s), int(head + size / 2 - (x - cxm) * s)
    font, aa = cv2.FONT_HERSHEY_SIMPLEX, cv2.LINE_AA
    cv2.rectangle(img, at(b['front_x'], b['half_w']), at(b['rear_x'], -b['half_w']), (72, 70, 66), -1)
    cv2.rectangle(img, at(b['front_x'], b['half_w']), at(b['rear_x'], -b['half_w']), (150, 148, 142), 1)
    cams = {c.uid: c for c in var.cameras}
    for r, line in zip(res.targets, shapes):
        for lk in r.looks:
            c = cams.get(lk.uid)
            if lk.met and c is not None:
                mid = line[len(line) // 2]
                cv2.line(img, at(c.x, c.y), at(*mid), _bgr(CAM_COLORS[c.slot % len(CAM_COLORS)]), 1, aa)
    placed = []
    for r, line in zip(res.targets, shapes):
        col = _bgr(GOOD if r.met else BAD)
        if len(line) > 1:
            cv2.polylines(img, [np.array([at(*p) for p in line], np.int32)], False, col, 3, aa)
        else:
            cv2.circle(img, at(*line[0]), 6, col, -1, aa)
        q = at(*line[len(line) // 2])
        name = _ascii(r.name)
        (w, h), _ = cv2.getTextSize(name, font, 0.42, 1)
        for dx, dy in ((9, -6), (9, 6 + h), (-9 - w, -6), (-9 - w, 6 + h), (9, -10 - h), (9, 10 + 2 * h)):
            box = (q[0] + dx, q[1] + dy - h, q[0] + dx + w, q[1] + dy + 3)      # the first place free of labels
            if not any(box[0] < o[2] and o[0] < box[2] and box[1] < o[3] and o[1] < box[3] for o in placed):
                break
        placed.append(box)
        cv2.putText(img, name, (box[0], box[3] - 3), font, 0.42, (220, 219, 214), 1, aa)
    for c in var.cameras:
        if c.visible and _finite_cam(c):
            cv2.circle(img, at(c.x, c.y), 4, (236, 235, 230), -1, aa)
            cv2.putText(img, _ascii(c.tag), (at(c.x, c.y)[0] + 5, at(c.x, c.y)[1] + 4), font, 0.38, (236, 235, 230), 1,
                        aa)
    n_ok = sum(r.met for r in res.targets)
    cv2.putText(img, _ascii(f'Critical targets  {n_ok} of {len(res.targets)} resolved  ({res.variant})'), (16, 26),
                font, 0.6, (236, 235, 230), 1, aa)
    ok, buf = cv2.imencode('.png', img)
    return buf.tobytes()


def files(res: CriticalResult, project=None) -> list:
    """[(name, bytes)]: critical_targets.json, critical_targets.csv, mounting_envelopes.csv, mounting_envelopes.png
    and critical_targets.png."""
    out = [('critical_targets.json', (json.dumps(to_doc(res, project), indent=1) + '\n').encode('utf-8')),
           ('critical_targets.csv', to_csv(res).encode('utf-8')),
           ('mounting_envelopes.csv', sheet_csv(res).encode('utf-8')),
           ('mounting_envelopes.png', envelope_png(res))]
    if project is not None:
        out.append(('critical_targets.png', plan_png(res, project)))
    return out


def export(res: CriticalResult, project, out_dir) -> list[str]:
    """Write files() into out_dir; returns their paths."""
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for name, data in files(res, project):
        path = os.path.join(out_dir, name)
        with open(path, 'wb') as fh:
            fh.write(data)
        paths.append(path)
    return paths
