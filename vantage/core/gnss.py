"""GNSS antennas and the IMU / GNSS-INS: the sky each antenna sees, the dual-antenna heading baseline, the lever arms
the INS needs, and how well every sensor's pose will be known (the pose error budget).

Sky view. From each antenna's phase centre, rays over the sky (cells of az_step × el_step from the antenna's horizon to
the zenith, the elevation mask on a cell edge) are traced against the body and the rig parts (the vehicle mesh with the
rig merged in, its occluding parts; the simplified occluder for a mesh over FULL_MESH_MAX triangles) and against the
housings of the other sensors (a lidar, radar, IMU or antenna as its library box or cylinder, core.sensors.body_box; a
camera as its module's box, width × height × length, behind its optical centre). The sky is taken in the vehicle frame
(azimuth counter-clockwise from +x, elevation above the xy plane) unless the antenna is tilted more than 5°, then in its
own frame. Every blocked ray is attributed to the nearest thing it meets: a body part, a rig part, a sensor or a camera.
The visible share is solid-angle weighted over the sky above the mask: a cell weighs Δaz · (sin el₁ − sin el₀), the
cos(el) weighting integrated over the cell. The horizon profile is, per azimuth, the highest blocked elevation (the top
edge of the highest blocked cell, the mask not applied); the skyline, what the plots draw, is the top of the blocked run
that rises from the antenna's horizon (a part overhead, such as a mast head, is left out of it). Expected satellites =
n_sats (above the mask, a setting) × the visible share: a uniform sky is assumed, an indicator rather than a prediction
(real constellations are not uniform). Only the vehicle, its rig and its sensors block: buildings, trees and bridges are
not modelled.

Near field. A structure within near_m of the phase centre that rises more than NEAR_MIN_EL_DEG above the antenna's
horizon reflects signals into it (multipath); the gentle crown of the roof the antenna stands on does not count. An
antenna whose base does not sit on a surface reaching GROUND_PLANE_M around it (rays straight down from a ring that far
out must meet a surface within GROUND_PLANE_TOL_M of the base's height) lacks a ground plane.

Heading. An INS with two antenna inputs (model.antennas == 2) takes its heading from the baseline between the primary
and the secondary antenna (the first two by tag, or chosen): heading σ = model.heading_sigma(the baseline's horizontal
length); the INS must be told the baseline's direction in its own frame (azimuth from its +x, vertical angle).

Lever arms and the pose error budget. The lever arm from the reference IMU's axis centre to every antenna, camera,
lidar, radar and other IMU and to the vehicle frame origin (the nuScenes ego frame), in the IMU frame and the vehicle
frame. A sensor's world position error per vehicle axis is the INS position σ ⊕ the attitude error acting on the lever
arm (δp = δθ × L, roll, pitch and heading errors independent) ⊕ the lever-arm measurement error (the antenna's lever arm
entered in the INS and the sensor's own, each lever_arm_sigma_mm per axis); its angle error is the INS attitude σ. All
1σ, independent, small angles. The sensors' own extrinsic calibration errors are not included.
"""
from __future__ import annotations

import datetime as _dt
import json
import math
import os
import re
import threading
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass

import numpy as np

from . import rules
from . import sensors as S
from .mathutil import rot_zyx

SCHEMA = 'vantage.gnss_imu/1'
FRAME = 'vehicle (ISO 8855, rear axle)'
SEVERITIES = ('critical', 'serious', 'warning', 'note', 'good')
ISSUE_SEVERITIES = ('critical', 'serious', 'warning')
TILT_OWN_DEG = 5.0              # an antenna tilted more than this sees the sky about its own axis
GROUND_PLANE_M = 0.10           # an antenna needs a surface this far round its base (a 20 cm plate)
NEAR_MIN_EL_DEG = 3.0           # a near structure must rise this far above the antenna's horizon to reflect into it
GROUND_PLANE_TOL_M = 0.04       # ... within this of its base's height (a curved roof passes)
GROUND_PLANE_LIFT_M = 0.06      # the ground-plane rays start this far above the base
NO_SKY = 0.02                   # visible share below this: the antenna has no sky (inside or under something)
BLOCKED_WARN, BLOCKED_SERIOUS = 0.10, 0.25
BASELINE_WARN_M, BASELINE_SERIOUS_M = 1.0, 0.5
BASELINE_TILT_DEG = 15.0        # a baseline rising more than this gets a note
IMU_LIDAR_NOTE_M = 1.5          # an IMU farther than this from the nearest lidar gets a note
FULL_MESH_MAX = 60000           # trace the full mesh up to this many triangles, else the simplified occluder
WEDGE_DEG = 10.0                # azimuth wedges: each traces only the triangles that can reach it
CHUNK = 1_000_000               # rays × triangles per batch
EPS = 1e-6

ASSUMPTIONS = (
    'Sky view: only the vehicle body, the rig parts and the sensors\' housings block the sky; buildings, trees, '
    'bridges and other traffic are not modelled. Expected satellites = satellites above the mask (a setting, '
    'typical for a multi-constellation receiver) × the visible share of the sky, assuming satellites spread evenly '
    'over the sky: an indicator, not a prediction. Multipath is flagged from geometry alone (a structure within the'
    ' near-field distance that rises above the antenna\'s horizon). Heading σ scales the INS datasheet figure by its'
    ' rated baseline over the baseline\'s horizontal length. Pose error budget: 1σ, independent errors, small '
    'angles; the INS horizontal position σ is applied to each horizontal axis (conservative when the datasheet '
    'gives a 2D figure) and the vertical σ is it × the vertical ratio; the lever arm terms are δθ × L with '
    'independent roll, pitch and heading errors; the antenna\'s lever arm in the INS and each sensor\'s own lever arm'
    ' are measured to the lever-arm σ per axis. The sensors\' own extrinsic calibration errors, time-sync errors and'
    ' GNSS outages are not included. Library INS figures are typical, not a product\'s datasheet.')

LIMITS = {'mask_deg': (0.0, 60.0), 'az_step_deg': (0.5, 10.0), 'el_step_deg': (0.25, 5.0), 'n_sats': (1.0, 200.0),
          'near_m': (0.05, 5.0), 'pos_sigma_cm': (0.0, 1000.0), 'vertical_ratio': (1.0, 5.0),
          'attitude_sigma_deg': (0.0, 10.0), 'heading_sigma_deg': (0.0, 10.0), 'lever_arm_sigma_mm': (0.0, 1000.0),
          'label_range_m': (1.0, 1000.0)}
TAG_KEYS = ('imu', 'primary', 'secondary')

# blocker colours: one hue family per kind (lidar, radar, GNSS and IMU as the sensors are drawn in the 3D and plan
# views), cameras their colour slot (the dark theme's, for the PNG)
BLOCKER_COLORS = {'body': ['#8b8a84', '#b3b1aa', '#6d6c67', '#c9c7bf'],
                  'rig': ['#b0558f', '#7d5cc9', '#d27fb4', '#8a3f70', '#a58be0'],
                  'lidar': ['#0aa5b8', '#0a6f8f', '#3fc4d3'], 'radar': ['#a0743c', '#7a5220', '#d4a05a'],
                  'gnss': ['#7f9a1c', '#5c7212', '#a6c23a'], 'imu': ['#6e7f96', '#4f5e73', '#94a3b8']}
CAM_COLORS = ['#3987e5', '#d95926', '#199e70', '#c98500', '#d55181', '#008300', '#9085e9', '#e66767']
KIND_WORD = {'body': 'body', 'rig': 'rig part', 'lidar': 'lidar', 'radar': 'radar', 'gnss': 'GNSS antenna',
             'imu': 'IMU', 'camera': 'camera', 'origin': 'vehicle origin'}


# =========================================================================== settings
def _flag(v, default):
    if isinstance(v, (bool, int)) and not isinstance(v, float):
        return bool(v)
    if isinstance(v, str):
        return v.strip().lower() in ('1', 'true', 'yes', 'on')
    return default


@dataclass
class GnssSettings:
    mask_override: bool = False      # one elevation mask for every antenna instead of each model's
    mask_deg: float = 10.0
    az_step_deg: float = 2.0         # sky cells
    el_step_deg: float = 1.0
    n_sats: float = 32.0             # satellites above the mask, a multi-constellation receiver (an assumption)
    near_m: float = 0.5              # multipath: structures this close that rise above the antenna's horizon
    imu: str = ''                    # tag of the reference IMU ('' = the first GNSS/INS by tag, else the first IMU)
    primary: str = ''                # heading antennas ('' = the first two by tag)
    secondary: str = ''
    pos_sigma_cm: float = 0.0        # INS horizontal position σ (0 = the model's)
    vertical_ratio: float = 2.0      # vertical σ = horizontal σ × this (RTK, typical)
    attitude_sigma_deg: float = 0.0  # INS roll and pitch σ (0 = the model's)
    heading_sigma_deg: float = 0.0   # INS heading σ (0 = from the baseline)
    lever_arm_sigma_mm: float = 5.0  # how well each lever arm is measured, per axis
    label_range_m: float = 50.0      # the budget also gives the error of a point this far from each sensor

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d: dict | None) -> 'GnssSettings':
        s = GnssSettings()
        d = d if isinstance(d, dict) else {}
        s.mask_override = _flag(d.get('mask_override'), s.mask_override)
        for k in TAG_KEYS:
            v = d.get(k)
            if isinstance(v, str):
                setattr(s, k, v.strip()[:32])
        for k, (lo, hi) in LIMITS.items():
            if k in d:
                try:
                    v = float(d[k])
                except (TypeError, ValueError):
                    continue
                if math.isfinite(v):
                    setattr(s, k, min(max(v, lo), hi))
        return s


# =========================================================================== results
@dataclass
class SkyView:
    """One antenna's sky. Blocker identities: {'key', 'name', 'kind' (body | rig | lidar | radar | gnss | imu |
    camera), 'uid', 'slot' (a camera's colour slot), 'color', 'share' (of the sky above the mask), 'share_all' (of the
    sky above the horizon), 'max_el', 'nearest_m', 'az_from', 'az_to' (the arc it covers, degrees), 'near_el' (highest
    elevation within near_m, None when farther)}."""
    uid: str
    tag: str
    model: str
    pos: np.ndarray                 # phase centre, vehicle frame
    frame: np.ndarray               # (3, 3) sky frame -> vehicle frame (columns x, y, up)
    own_frame: bool                 # tilted more than TILT_OWN_DEG: the sky about its own axis
    tilt_deg: float
    mask_deg: float
    az: np.ndarray                  # (A,) cell centres, degrees counter-clockwise from the frame's +x
    el_edges: np.ndarray            # (E + 1,) degrees, 0 … 90
    who: np.ndarray                 # (E, A) int16: index into blockers, -1 open sky
    dist: np.ndarray                # (E, A) float32: distance to the blocker, m (inf: open)
    blockers: list
    visible: float                  # share of the sky above the mask that is open (solid-angle weighted)
    visible_all: float              # the same above the antenna's horizon (no mask)
    horizon: np.ndarray             # (A,) highest blocked elevation per azimuth, nan where nothing blocks
    sectors: list                   # blocked azimuth runs above the mask, largest first
    near: list                      # multipath structures [{'name', 'kind', 'el_deg', 'dist_m'}]
    ground: dict                    # {'ok', 'surface', 'kind', 'missing', 'points'}
    expected_sats: float
    n_sats: float
    ms: float = 0.0
    skyline: np.ndarray | None = None   # (A,) top of the blocked run rising from the horizon, nan where it is open

    @property
    def el(self) -> np.ndarray:
        return 0.5 * (self.el_edges[:-1] + self.el_edges[1:])

    @property
    def no_sky(self) -> bool:
        return self.visible < NO_SKY

    @property
    def blocked(self) -> float:
        return 1.0 - self.visible

    def worst_sector(self):
        return self.sectors[0] if self.sectors else None

    def at(self, az_deg, el_deg):
        """(blocker identity or None, distance) of the cell at a sky direction."""
        A = len(self.az)
        a = int(round((az_deg % 360.0) / 360.0 * A)) % A
        e = int(np.clip(np.searchsorted(self.el_edges, el_deg, 'right') - 1, 0, len(self.el_edges) - 2))
        j = int(self.who[e, a])
        return (self.blockers[j] if j >= 0 else None), float(self.dist[e, a])


@dataclass
class GnssResult:
    settings: GnssSettings
    antennas: list                  # [SkyView], by tag
    imu: dict | None                # the reference IMU: {'uid', 'tag', 'model', 'model_id', 'pos', 'R', 'yaw', ...}
    imus: list                      # every visible IMU [{'uid', 'tag', 'model', 'antennas'}]
    baseline: dict | None
    lever_arms: list                # [{'tag', 'kind', 'uid', 'role', 'pos', 'vehicle_mm', 'imu_mm', 'length_mm'}]
    budget: dict | None             # {'inputs': {...}, 'rows': [...]}, None without an IMU
    findings: list                  # [{'severity', 'kind', 'message', 'tags', 'uids'}], worst first
    variant: str = ''
    variant_uid: str = ''
    seconds: float = 0.0

    def antenna(self, tag):
        return next((a for a in self.antennas if a.tag == tag), None)

    def lever(self, tag):
        return next((r for r in self.lever_arms if r['tag'] == tag), None)

    def budget_row(self, tag):
        return next((r for r in (self.budget or {}).get('rows', []) if r['tag'] == tag), None)

    def worst(self) -> str:
        return self.findings[0]['severity'] if self.findings else 'good'


# =========================================================================== small helpers
def tag_key(tag: str):
    """Natural order of tags: GN2 before GN10."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', str(tag))]


def _finite_pose(s) -> bool:
    return all(math.isfinite(float(getattr(s, k))) for k in ('x', 'y', 'z', 'yaw', 'pitch', 'roll'))


def _num(v, default=0.0) -> float:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) else default


def _project_for(project, variant):
    """The project with variant active (a shallow copy when it is not the active one)."""
    if variant is None or variant is project.variant:
        return project
    import copy
    q = copy.copy(project)
    q.variants = list(project.variants) + ([] if any(v is variant for v in project.variants) else [variant])
    q.active_uid = variant.uid
    return q


def visible_of(v, kind) -> list:
    """The variant's visible sensors of a kind with a usable pose, by tag."""
    return sorted((s for s in v.sensors if s.visible and s.kind == kind and _finite_pose(s)),
                  key=lambda s: tag_key(s.tag))


def _el_edges(mask, step) -> np.ndarray:
    """Elevation cell edges from 0° to 90°, about step apart, with the mask on an edge."""
    lo = np.linspace(0.0, mask, max(int(math.ceil(mask / step - 1e-9)), 1) + 1) if mask > 1e-9 else np.array([0.0])
    hi = np.linspace(mask, 90.0, max(int(math.ceil((90.0 - mask) / step - 1e-9)), 1) + 1)
    return np.concatenate([lo[:-1], hi])


# =========================================================================== ray casting
def _tri_pre(q):
    """Per-triangle vectors for rays from the origin (Möller–Trumbore with the origin fixed): det = d·N, u = d·W / det,
    v = d·Q / det, t = c / det."""
    v0, e1, e2 = q[:, 0], q[:, 1] - q[:, 0], q[:, 2] - q[:, 0]
    N = np.cross(e2, e1)
    W = np.cross(e2, -v0)
    Q = np.cross(-v0, e1)
    return N, W, Q, np.einsum('ij,ij->i', e2, Q)


def _cast(D, pre, sel=None):
    """Nearest hit of rays from the origin with directions D (K, 3) on the triangles (pre from _tri_pre, optionally a
    subset sel): (t (K,), triangle index (K,), -1 for a miss)."""
    N, W, Q, c = pre if sel is None else (pre[0][sel], pre[1][sel], pre[2][sel], pre[3][sel])
    K, M = len(D), len(N)
    best_t = np.full(K, np.inf)
    best_j = np.full(K, -1, np.int64)
    if not K or not M:
        return best_t, best_j
    rows = max(1, CHUNK // M)
    for s in range(0, K, rows):
        d = D[s:s + rows]
        det = d @ N.T
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            inv = 1.0 / det
            u = (d @ W.T) * inv
            v = (d @ Q.T) * inv
            t = c[None, :] * inv
            ok = (np.abs(det) > 1e-15) & (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9) & (t > 1e-6)
        t = np.where(ok, t, np.inf)
        j = np.argmin(t, axis=1)
        tt = t[np.arange(len(j)), j]
        hit = np.isfinite(tt)
        best_t[s:s + rows] = tt
        best_j[s:s + rows] = np.where(hit, j if sel is None else np.asarray(sel)[j], -1)
    return best_t, best_j


def _trace_mesh(D, az, step, q):
    """Rays D (E, A, 3) from the origin of the sky frame against triangles q (M, 3, 3) in that frame, wedge by wedge
    of azimuth: a wedge only traces the triangles not wholly behind one of its bounding half-planes (or below the
    horizon). Returns (t (E, A), triangle index (E, A))."""
    E, A = D.shape[:2]
    t = np.full((E, A), np.inf)
    j = np.full((E, A), -1, np.int64)
    if not len(q):
        return t, j
    up = np.flatnonzero((q[..., 2] > -EPS).any(1))
    if not len(up):
        return t, j
    qu = q[up]
    pre = _tri_pre(qu)
    W = max(1, int(WEDGE_DEG / step + 1e-9))
    for a0 in range(0, A, W):
        a1 = min(a0 + W, A)
        lo, hi = math.radians(az[a0] - step / 2), math.radians(az[a1 - 1] + step / 2)
        out = ((qu @ np.array([-math.sin(lo), math.cos(lo), 0.0])) < -EPS).all(1) | \
              ((qu @ np.array([math.sin(hi), -math.cos(hi), 0.0])) < -EPS).all(1)
        sel = np.flatnonzero(~out)
        if not len(sel):
            continue
        tt, jj = _cast(D[:, a0:a1].reshape(-1, 3), pre, sel)
        t[:, a0:a1] = tt.reshape(E, a1 - a0)
        j[:, a0:a1] = np.where(jj >= 0, up[np.maximum(jj, 0)], -1).reshape(E, a1 - a0)
    return t, j


def _slab(o, d, half):
    """Entry and exit distances of rays o + t d through |x_k| ≤ half_k (per ray, over the axes given)."""
    with np.errstate(divide='ignore', invalid='ignore'):
        inv = 1.0 / d
        t1 = (-half - o) * inv
        t2 = (half - o) * inv
    lo = np.fmin(t1, t2)                   # nan only for a ray grazing the slab's face: inside it
    hi = np.fmax(t1, t2)
    return np.where(np.isnan(lo), -np.inf, lo).max(axis=1), np.where(np.isnan(hi), np.inf, hi).min(axis=1)


def _ray_housing(O, D, c, half, R, rnd):
    """Distance along rays O + t D (O (3,) or (K, 3), D (K, 3)) to a housing: a box (centre c, half sizes, rotation R
    housing -> frame) or, rnd, a cylinder about its z axis (radius half[0]); inf for a miss. A ray starting inside hits
    at 0."""
    o = (np.asarray(O, float) - c) @ R
    d = D @ R
    o = np.broadcast_to(o, d.shape)
    if not rnd:
        tin, tout = _slab(o, d, half)
    else:
        tin, tout = _slab(o[:, 2:], d[:, 2:], half[2:])
        r = half[0]
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
        tin, tout = np.maximum(tin, s_in), np.minimum(tout, s_out)
    hit = (tout >= np.maximum(tin, 0.0)) & (tout > 1e-9)
    return np.where(hit, np.maximum(tin, 0.0), np.inf)


# =========================================================================== what blocks
def _ident(key, name, kind, uid='', slot=0, order=0) -> dict:
    fam = BLOCKER_COLORS.get(kind)
    color = CAM_COLORS[int(slot) % 8] if kind == 'camera' else fam[int(order) % len(fam)] if fam else '#8b8a84'
    return {'key': key, 'name': name, 'kind': kind, 'uid': uid, 'slot': int(slot), 'color': color}


def _part_idents(project, mesh) -> list:
    """Identity of each part of the merged mesh: the vehicle's parts first, then one per visible rig part."""
    nb = len(project.vehicle.mesh.parts) if project.vehicle.mesh is not None else 0
    rig = [r for r in getattr(project, 'rig', ()) if r.visible]
    out = []
    for k, part in enumerate(mesh.parts):
        if k >= nb and k - nb < len(rig):
            r = rig[k - nb]
            out.append(_ident(f'rig:{r.uid}', r.name, 'rig', r.uid, order=k - nb))
        else:
            out.append(_ident(f'part:{part.name}', part.name, 'body', order=k))
    return out


_OCC = OrderedDict()            # occluder key -> (triangles (M, 3, 3), part index (M,), identities)
_SKY = OrderedDict()            # sky view key -> SkyView
_lock = threading.Lock()


def _occ_key(project, vg):
    mesh = vg.mesh if vg.mesh.n_tris <= FULL_MESH_MAX else vg.occ_mesh
    return (vg.version, id(mesh), mesh.n_tris, tuple(p.name for p in mesh.parts), tuple(p.role for p in mesh.parts),
            tuple((r.uid, r.name) for r in getattr(project, 'rig', ()) if r.visible))


def _occluder(project, vg):
    """(triangles (M, 3, 3), part index (M,), part identities) of the body and rig faces that block the sky."""
    key = _occ_key(project, vg)
    with _lock:
        if key in _OCC:
            _OCC.move_to_end(key)
            return _OCC[key]
    mesh = vg.mesh if vg.mesh.n_tris <= FULL_MESH_MAX else vg.occ_mesh
    faces = mesh.occluder_faces()
    out = (np.ascontiguousarray(mesh.tri[faces]), np.asarray(mesh.face_part[faces]), _part_idents(project, mesh))
    with _lock:
        _OCC[key] = out
        while len(_OCC) > 3:
            _OCC.popitem(last=False)
    return out


def _housings(project, v, exclude_uid) -> list:
    """[(identity, centre, half sizes, rotation, round)] of every visible sensor (but one) and camera."""
    out = []
    order = {}
    for s in v.sensors:
        if not s.visible or not _finite_pose(s):
            continue
        k = order[s.kind] = order.get(s.kind, -1) + 1
        if s.uid == exclude_uid:
            continue
        m = project.sensor_model_for(s)
        c, half, R, rnd = S.body_box(s, m)
        out.append((_ident(f'sensor:{s.uid}', s.tag, s.kind, s.uid, order=k), c, half, R, bool(rnd)))
    for cam in v.cameras:
        if not cam.visible or not all(math.isfinite(float(getattr(cam, a))) for a in ('x', 'y', 'z', 'yaw', 'pitch',
                                                                                          'roll')):
            continue
        mod = project.module_for(cam)
        try:
            w, h, length = (max(float(x), 1.0) / 1000.0 for x in mod.size_mm)
        except (TypeError, ValueError):
            w, h, length = 0.025, 0.025, 0.044
        R = rot_zyx(cam.yaw, cam.pitch, cam.roll)
        c = cam.pos() - R[:, 0] * length / 2               # the module behind its optical centre
        out.append((_ident(f'camera:{cam.uid}', cam.tag, 'camera', cam.uid, cam.slot), c,
                    np.array([length / 2, w / 2, h / 2]), R, False))
    return out


def _sky_key(project, vg, s, st, v):
    m = project.sensor_model_for(s)
    r6 = lambda x: round(float(x), 6)          # noqa: E731
    sens = tuple((o.uid, o.tag, o.kind, o.model_id,
                  *(r6(getattr(o, a)) for a in ('x', 'y', 'z', 'yaw', 'pitch', 'roll')),
                  tuple(project.sensor_model_for(o).size_mm), project.sensor_model_for(o).optical_z_mm,
                  project.sensor_model_for(o).scan)
                 for o in v.sensors if o.visible)
    cams = tuple((c.uid, c.tag, c.slot, *(r6(getattr(c, a)) for a in ('x', 'y', 'z', 'yaw', 'pitch', 'roll')),
                  tuple(project.module_for(c).size_mm)) for c in v.cameras if c.visible)
    mask = st.mask_deg if st.mask_override else m.mask_deg
    return (_occ_key(project, vg) if vg is not None else None, s.uid, m.name, r6(mask), r6(m.optical_z_mm),
            r6(st.az_step_deg), r6(st.el_step_deg), r6(st.near_m), r6(st.n_sats), sens, cams)


# =========================================================================== the sky view
def sky_view(project, vg, sensor, settings: GnssSettings | None = None, variant=None) -> SkyView:
    """The sky an antenna sees (cached on its pose, the mask, the steps, the geometry and the other sensors' poses:
    the Issues list asks for it on every edit)."""
    st = settings or GnssSettings.from_dict(getattr(project.analysis, 'gnss', None))
    v = variant or project.variant
    key = _sky_key(project, vg, sensor, st, v)
    with _lock:
        hit = _SKY.get(key)
        if hit is not None:
            _SKY.move_to_end(key)
            return hit
    sv = _compute_sky(project, vg, sensor, st, v)
    with _lock:
        _SKY[key] = sv
        while len(_SKY) > 48:
            _SKY.popitem(last=False)
    return sv


def clear_cache():
    with _lock:
        _SKY.clear()
        _OCC.clear()


def _arc(cols, A, step):
    """(from, to) degrees of the smallest arc covering the occupied azimuth columns (cols sorted), counter-clockwise."""
    if len(cols) == A:
        return 0.0, 360.0
    gaps = np.diff(np.r_[cols, cols[0] + A])
    g = int(np.argmax(gaps))
    first, last = cols[(g + 1) % len(cols)], cols[g]
    return float((first * step - step / 2) % 360.0), float((last * step + step / 2) % 360.0)


def _runs(mask):
    """Index arrays of the runs of True in a closed (wrapping) 1-D mask."""
    n = len(mask)
    if mask.all():
        return [np.arange(n)]
    out, start = [], None
    for i in range(n):
        if mask[i] and start is None:
            start = i
        elif not mask[i] and start is not None:
            out.append(np.arange(start, i))
            start = None
    if start is not None:
        out.append(np.arange(start, n))
    if len(out) > 1 and mask[0] and mask[-1]:
        out[0] = np.r_[out.pop(), out[0]]
    return out


def _compute_sky(project, vg, s, st: GnssSettings, v) -> SkyView:
    t0 = time.perf_counter()
    m = project.sensor_model_for(s)
    mask = float(st.mask_deg if st.mask_override else min(max(_num(m.mask_deg, 10.0), 0.0), 60.0))
    pos = s.pos().astype(float)
    Ra = S.rotation(s)
    tilt = math.degrees(math.acos(min(max(float(Ra[2, 2]), -1.0), 1.0)))
    own = tilt > TILT_OWN_DEG
    F = Ra if own else np.eye(3)
    A = max(int(round(360.0 / st.az_step_deg)), 8)
    step = 360.0 / A
    az = np.arange(A) * step
    edges = _el_edges(mask, st.el_step_deg)
    el = 0.5 * (edges[:-1] + edges[1:])
    D = S.directions(el, az)                                   # (E, A, 3) in the sky frame
    E = len(el)
    t = np.full((E, A), np.inf)
    who = np.full((E, A), -1, np.int32)
    idents, index = [], {}

    def ident_ix(idn):
        if idn['key'] not in index:
            index[idn['key']] = len(idents)
            idents.append(dict(idn))
        return index[idn['key']]
    tri = part = pids = None
    if vg is not None:
        tri, part, pids = _occluder(project, vg)
        tm, jm = _trace_mesh(D, az, step, (tri - pos) @ F)
        hit = jm >= 0
        if hit.any():
            parts_hit = part[jm[hit]]
            lut = np.zeros(int(parts_hit.max()) + 1, np.int32)
            for k in np.unique(parts_hit):
                lut[k] = ident_ix(pids[int(k)])
            who[hit] = lut[parts_hit]
            t[hit] = tm[hit]
    houses = _housings(project, v, s.uid)
    Df = D.reshape(-1, 3)
    for idn, c, half, R, rnd in houses:
        c_loc = (c - pos) @ F
        if c_loc[2] + float(np.linalg.norm(half)) < -EPS:      # wholly below the antenna's horizon
            continue
        th = _ray_housing(np.zeros(3), Df, c_loc, half, F.T @ R, rnd).reshape(E, A)
        closer = th < t
        if closer.any():
            t[closer] = th[closer]
            who[closer] = ident_ix(idn)
    blocked = who >= 0
    # ---- shares: each cell's solid angle, the cos(el) weighting integrated over it
    w_row = math.radians(step) * (np.sin(np.radians(edges[1:])) - np.sin(np.radians(edges[:-1])))
    above = edges[:-1] >= mask - 1e-9
    total = float(w_row[above].sum() * A)
    total_all = float(w_row.sum() * A)
    W = np.broadcast_to(w_row[:, None], (E, A))
    blk_above = blocked & above[:, None]
    visible = 1.0 - float(W[blk_above].sum()) / total if total > 0 else 1.0
    visible_all = 1.0 - float(W[blocked].sum()) / total_all
    top = E - 1 - np.argmax(blocked[::-1], axis=0)
    horizon = np.where(blocked.any(0), edges[top + 1], np.nan)
    gap = ~blocked
    first_open = np.where(gap.any(0), np.argmax(gap, axis=0), E)
    skyline = np.where(blocked[0], edges[first_open], np.nan)
    # ---- each blocker
    near_m = st.near_m
    for i, idn in enumerate(idents):
        mine = who == i
        mine_above = mine & above[:, None]
        cols = np.flatnonzero(mine.any(0))
        rows = np.flatnonzero(mine.any(1))
        a_from, a_to = _arc(cols, A, step) if len(cols) else (0.0, 0.0)
        close = mine & (t <= near_m)
        crow = np.flatnonzero(close.any(1))
        idn.update(share=float(W[mine_above].sum()) / total if total > 0 else 0.0,
                   share_all=float(W[mine].sum()) / total_all,
                   max_el=float(edges[rows.max() + 1]) if len(rows) else 0.0,
                   nearest_m=float(t[mine].min()), az_from=a_from, az_to=a_to,
                   near_el=float(edges[crow.max() + 1]) if len(crow) else None)
    order = sorted(range(len(idents)), key=lambda i: (-idents[i]['share'], -idents[i]['share_all'], idents[i]['name']))
    remap = np.full(len(idents) + 1, -1, np.int32)
    for new, old in enumerate(order):
        remap[old] = new
    idents = [idents[i] for i in order]
    who = np.where(who >= 0, remap[who], -1).astype(np.int16)
    # ---- blocked sectors above the mask
    sectors = []
    colblk = (who >= 0)[above].any(0) if above.any() else np.zeros(A, bool)
    for run in (_runs(colblk) if colblk.any() else []):
        sub = who[above][:, run]
        Wr = np.broadcast_to(w_row[above][:, None], sub.shape)
        names = {}
        for b in np.unique(sub[sub >= 0]):
            names[idents[int(b)]['name']] = float(Wr[sub == b].sum()) / total
        a_from, a_to = _arc(np.sort(run), A, step) if len(run) < A else (0.0, 360.0)
        sectors.append({'az_from': a_from, 'az_to': a_to, 'width_deg': len(run) * step,
                        'max_el': float(np.nanmax(horizon[run])), 'share': float(sum(names.values())),
                        'blockers': [n for n, _ in sorted(names.items(), key=lambda kv: -kv[1])]})
    sectors.sort(key=lambda x: -x['share'])
    near = [{'name': b['name'], 'kind': b['kind'], 'uid': b['uid'], 'el_deg': b['near_el'],
             'dist_m': b['nearest_m']} for b in idents
            if b['near_el'] is not None and b['near_el'] > NEAR_MIN_EL_DEG + 1e-9]
    ground = _ground(project, s, m, Ra, tri, part, pids, houses)
    return SkyView(s.uid, s.tag, f'{m.vendor} {m.name}'.strip(), pos, F, own, tilt, mask, az, edges, who,
                   t.astype(np.float32), idents, visible, visible_all, horizon, sectors, near, ground,
                   st.n_sats * visible, st.n_sats, (time.perf_counter() - t0) * 1000.0, skyline)


def _ground(project, s, m, Ra, tri, part, pids, houses) -> dict:
    """Whether the antenna's base sits on a surface reaching GROUND_PLANE_M round it: rays straight down (along the
    antenna's own -z) from 8 points on a ring that far out must meet a surface within GROUND_PLANE_TOL_M of the base's
    height (a curved roof passes, a bar, a lidar's top or a small bracket does not). The ray from the base centre names
    what it stands on."""
    up = Ra[:, 2]
    base = s.pos() - up * (_num(m.optical_z_mm, 0.0) / 1000.0)
    lift = GROUND_PLANE_LIFT_M
    ang = np.radians(np.arange(8) * 45.0)
    O = np.vstack([base, base + GROUND_PLANE_M * (np.cos(ang)[:, None] * Ra[:, 0] + np.sin(ang)[:, None] * Ra[:, 1])])
    O = O + up * lift
    down = -up
    t = np.full(len(O), np.inf)
    who = [None] * len(O)
    if tri is not None and len(tri):
        lo, hi = O.min(0) - 0.15, O.max(0) + 0.15
        near = np.flatnonzero(((tri.max(1) >= lo) & (tri.min(1) <= hi)).all(1))
        if len(near):
            for k, o in enumerate(O):
                tt, jj = _cast(down[None, :], _tri_pre(tri[near] - o))
                if jj[0] >= 0 and tt[0] < t[k]:
                    t[k] = tt[0]
                    who[k] = pids[int(part[near[jj[0]]])]
    for idn, c, half, R, rnd in houses:
        if idn['kind'] == 'camera':
            continue
        th = _ray_housing(O, np.broadcast_to(down, O.shape), c, half, R, rnd)
        for k in np.flatnonzero(th < t):
            t[k] = th[k]
            who[k] = idn
    ring = np.abs(t[1:] - lift) <= GROUND_PLANE_TOL_M + 1e-9
    under = who[0] if abs(t[0] - lift) <= GROUND_PLANE_TOL_M + 0.01 else None
    return {'ok': bool(ring.all()), 'surface': under['name'] if under else '', 'kind': under['kind'] if under else '',
            'missing': int((~ring).sum()), 'points': int(len(ring))}


# =========================================================================== the IMU, the baseline, lever arms
def reference_imu(project, v, st: GnssSettings):
    """The IMU the lever arms and the budget are taken from: the chosen one, else the first GNSS/INS (with antenna
    inputs) by tag, else the first IMU."""
    imus = visible_of(v, 'imu')
    if not imus:
        return None
    chosen = next((s for s in imus if st.imu and s.tag == st.imu), None)
    if chosen is None:
        chosen = next((s for s in imus if project.sensor_model_for(s).antennas > 0), imus[0])
    return chosen


def heading_pair(v, st: GnssSettings):
    """(primary, secondary) antennas of the heading baseline: the chosen ones, else the first two by tag."""
    ants = visible_of(v, 'gnss')
    if len(ants) < 2:
        return None, None
    prim = next((a for a in ants if st.primary and a.tag == st.primary), ants[0])
    sec = next((a for a in ants if st.secondary and a.tag == st.secondary and a is not prim), None)
    if sec is None:
        sec = next(a for a in ants if a is not prim)
    return prim, sec


def _imu_info(project, s) -> dict:
    m = project.sensor_model_for(s)
    R = S.rotation(s)
    return {'uid': s.uid, 'tag': s.tag, 'model': f'{m.vendor} {m.name}'.strip(), 'model_id': m.id,
            'antennas': int(m.antennas), 'pos': s.pos().astype(float), 'R': R, 'yaw': float(s.yaw),
            'pitch': float(s.pitch), 'roll': float(s.roll), 'spec': m}


def baseline(project, ins, prim, sec) -> dict | None:
    """The heading baseline from the primary to the secondary antenna: its vector, length, horizontal length,
    azimuth (counter-clockwise from +x) and vertical angle in the vehicle frame and in the IMU frame, and the heading
    σ the INS model gives for its horizontal length."""
    if ins is None or prim is None or sec is None:
        return None
    m = ins['spec']
    b = sec.pos().astype(float) - prim.pos().astype(float)
    L = float(np.linalg.norm(b))
    Lh = float(math.hypot(b[0], b[1]))
    bi = ins['R'].T @ b
    return {'primary': prim.tag, 'secondary': sec.tag, 'primary_uid': prim.uid, 'secondary_uid': sec.uid,
            'ins': ins['tag'], 'vector_m': b, 'length_m': L, 'horizontal_m': Lh,
            'azimuth_deg': math.degrees(math.atan2(b[1], b[0])) if Lh > 1e-9 else 0.0,
            'vertical_deg': math.degrees(math.atan2(b[2], Lh)) if L > 1e-9 else 0.0,
            'imu_vector_m': bi, 'imu_azimuth_deg': math.degrees(math.atan2(bi[1], bi[0])) if L > 1e-9 else 0.0,
            'imu_vertical_deg': math.degrees(math.atan2(bi[2], math.hypot(bi[0], bi[1]))) if L > 1e-9 else 0.0,
            'heading_sigma_deg': m.heading_sigma(Lh), 'rated_sigma_deg': float(m.heading_sigma_deg),
            'rated_baseline_m': float(m.heading_baseline_m)}


KIND_ORDER = {'origin': 0, 'gnss': 1, 'camera': 2, 'lidar': 3, 'radar': 4, 'imu': 5}


def lever_arms(project, v, ins, prim=None, sec=None) -> list:
    """Lever arms from the IMU's axis centre to the vehicle origin and every visible antenna, camera, lidar, radar and
    other IMU, in mm, in the vehicle frame and in the IMU frame."""
    if ins is None:
        return []
    p0, R = ins['pos'], ins['R']
    rows = [('Vehicle origin', 'origin', '', np.zeros(3))]
    rows += [(s.tag, 'gnss', s.uid, s.pos()) for s in visible_of(v, 'gnss')]
    rows += [(c.tag, 'camera', c.uid, c.pos()) for c in sorted((c for c in v.cameras if c.visible),
                                                               key=lambda c: tag_key(c.tag))]
    for kind in ('lidar', 'radar', 'imu'):
        rows += [(s.tag, kind, s.uid, s.pos()) for s in visible_of(v, kind) if s.uid != ins['uid']]
    out = []
    for tag, kind, uid, pos in rows:
        pos = np.asarray(pos, float)
        if not np.isfinite(pos).all():
            continue
        Lv = (pos - p0) * 1000.0
        role = ''
        if prim is not None and uid == prim.uid:
            role = 'primary'
        elif sec is not None and uid == sec.uid:
            role = 'secondary'
        out.append({'tag': tag, 'kind': kind, 'uid': uid, 'role': role, 'pos': pos, 'vehicle_mm': Lv,
                    'imu_mm': R.T @ Lv, 'length_mm': float(np.linalg.norm(Lv))})
    return out


def budget(ins, levers, base, st: GnssSettings) -> dict | None:
    """The pose error budget of the vehicle origin and every camera, lidar and radar (see the module docstring)."""
    if ins is None:
        return None
    m = ins['spec']
    pos_cm = st.pos_sigma_cm or _num(m.pos_sigma_cm)
    att = st.attitude_sigma_deg or _num(m.attitude_sigma_deg)
    if st.heading_sigma_deg > 0:
        head, head_src = st.heading_sigma_deg, 'setting'
    elif base is not None and math.isfinite(base['heading_sigma_deg']):
        head, head_src = base['heading_sigma_deg'], 'baseline'
    else:
        head, head_src = float('nan'), 'none'
    h = pos_cm * 10.0
    s_ins = np.array([h, h, h * st.vertical_ratio])
    sr = sp = math.radians(att)
    sy = math.radians(head) if math.isfinite(head) else 0.0
    n_meas = 2 if m.antennas > 0 else 1                 # the antenna's lever arm in the INS and the sensor's own
    s_meas = st.lever_arm_sigma_mm * math.sqrt(n_meas)
    rows = []
    for r in levers:
        if r['kind'] not in ('origin', 'camera', 'lidar', 'radar'):
            continue
        Lx, Ly, Lz = (r['vehicle_mm'] / 1000.0).tolist()
        rot = 1000.0 * np.array([math.hypot(sp * Lz, sy * Ly), math.hypot(sy * Lx, sr * Lz),
                                 math.hypot(sr * Ly, sp * Lx)])
        tot = np.sqrt(s_ins ** 2 + rot ** 2 + s_meas ** 2)
        hz = float(math.hypot(tot[0], tot[1]))
        rows.append({'tag': r['tag'], 'kind': r['kind'], 'uid': r['uid'], 'lever_m': r['length_mm'] / 1000.0,
                     'ins_mm': s_ins, 'rotation_mm': rot, 'measure_mm': np.full(3, s_meas), 'sigma_mm': tot,
                     'horizontal_mm': hz, 'vertical_mm': float(tot[2]), 'total_mm': float(np.linalg.norm(tot)),
                     'far_mm': float(math.hypot(hz, st.label_range_m * sy * 1000.0)),
                     'roll_deg': att, 'pitch_deg': att, 'heading_deg': head})
    return {'inputs': {'pos_sigma_cm': pos_cm, 'pos_source': 'setting' if st.pos_sigma_cm else 'model',
                       'vertical_ratio': st.vertical_ratio, 'attitude_sigma_deg': att,
                       'attitude_source': 'setting' if st.attitude_sigma_deg else 'model',
                       'heading_sigma_deg': head, 'heading_source': head_src,
                       'lever_arm_sigma_mm': st.lever_arm_sigma_mm, 'lever_arms_measured': n_meas,
                       'label_range_m': st.label_range_m},
            'rows': rows}


# =========================================================================== the study
_ENGINE = [None]


def _engine():
    if _ENGINE[0] is None:
        from .analysis import Engine
        _ENGINE[0] = Engine()
    return _ENGINE[0]


def has_inputs(project, variant=None) -> bool:
    v = variant or project.variant
    return any(s.visible and s.kind in ('gnss', 'imu') for s in v.sensors)


def run(project, engine=None, settings: GnssSettings | None = None, variant=None, progress=None,
        cancel=None) -> GnssResult | None:
    """The GNSS and IMU study of a variant (the active one by default). Returns None when cancelled."""
    p = _project_for(project, variant)
    eng = engine if engine is not None else _engine()
    return _analyse(p, eng.vehicle_geom(p), settings, progress, cancel)


def _analyse(p, vg, settings, progress=None, cancel=None):
    t0 = time.perf_counter()
    v = p.variant
    st = settings or GnssSettings.from_dict(getattr(p.analysis, 'gnss', None))

    def prog(f, text=''):
        if progress is not None:
            progress(min(max(float(f), 0.0), 1.0), text)
    ants = visible_of(v, 'gnss')
    sky = []
    for i, a in enumerate(ants):
        if cancel is not None and cancel():
            return None
        prog(i / (len(ants) + 1), f'Sky view of {a.tag}')
        sky.append(sky_view(p, vg, a, st, v))
    if cancel is not None and cancel():
        return None
    ref = reference_imu(p, v, st)
    ins = _imu_info(p, ref) if ref is not None else None
    prim, sec = heading_pair(v, st)
    base = baseline(p, ins, prim, sec) if ins is not None and ins['antennas'] >= 2 else None
    levers = lever_arms(p, v, ins, prim if base else None, sec if base else None)
    bud = budget(ins, levers, base, st)
    imus = [{'uid': s.uid, 'tag': s.tag, 'model': f'{m.vendor} {m.name}'.strip(), 'antennas': int(m.antennas)}
            for s in visible_of(v, 'imu') for m in (p.sensor_model_for(s),)]
    res = GnssResult(st, sky, ins, imus, base, levers, bud, [], v.name, v.uid)
    res.findings = _findings(p, v, res)
    res.seconds = time.perf_counter() - t0
    prog(1.0, 'Done')
    return res


# =========================================================================== findings
def _pct(x, nd=0):
    return f'{x * 100:.{nd}f} %'


def _f(out, sev, kind, message, objs=()):
    out.append({'severity': sev, 'kind': kind, 'message': message, 'tags': [o[0] for o in objs],
                'uids': [o[1] for o in objs if o[1]]})


def _findings(p, v, res: GnssResult) -> list:
    out = []
    st = res.settings
    bad = [s for s in v.sensors if s.visible and s.kind in ('gnss', 'imu') and not _finite_pose(s)]
    for s in bad:
        _f(out, 'critical', 'pose', f'{s.tag}: its position or angles are not numbers; it is left out.',
           [(s.tag, s.uid)])
    # ---- each antenna's sky
    for a in res.antennas:
        obj = [(a.tag, a.uid)]
        top = [b for b in a.blockers if b['share'] > 0.0005][:3]
        by = ', '.join(f'{b["name"]} {_pct(b["share"], 1)}' for b in top)
        if a.no_sky:
            what = top[0]['name'] if top else (a.blockers[0]['name'] if a.blockers else 'the vehicle')
            _f(out, 'critical', 'no_sky',
               f'{a.tag} has no sky: {_pct(a.blocked)} of the sky above its {a.mask_deg:g}° mask is blocked, by '
               f'{what}. It sits inside or under the body, the rig or a sensor: move it onto the roof or the top of '
               'the rig.', obj)
        elif a.blocked > BLOCKED_SERIOUS:
            _f(out, 'serious', 'sky',
               f'{a.tag}: {_pct(a.blocked)} of the sky above its {a.mask_deg:g}° mask is blocked ({by}): about '
               f'{a.expected_sats:.0f} of {a.n_sats:.0f} satellites in view. Move it clear of them or above them.', obj)
        elif a.blocked > BLOCKED_WARN:
            _f(out, 'warning', 'sky',
               f'{a.tag}: {_pct(a.blocked)} of the sky above its {a.mask_deg:g}° mask is blocked ({by}): about '
               f'{a.expected_sats:.0f} of {a.n_sats:.0f} satellites in view.', obj)
        else:
            _f(out, 'good', 'sky',
               f'{a.tag} sees {_pct(a.visible, 1)} of the sky above its {a.mask_deg:g}° mask: about '
               f'{a.expected_sats:.0f} of {a.n_sats:.0f} satellites' + (f' ({by} blocked).' if by else '.'), obj)
        if a.near and not a.no_sky:
            parts = '; '.join(f'{n["name"]} {n["dist_m"]:.2f} m away rises to {n["el_deg"]:.0f}°' for n in a.near[:3])
            it = 'it' if len(a.near) == 1 else 'them'
            _f(out, 'warning', 'multipath',
               f'{a.tag}: {parts} above the antenna\'s horizon within {st.near_m:g} m. Signals reflected off {it} '
               f'(multipath) degrade the position and heading: move the antenna farther from {it} or above {it}.',
               obj + [(n['name'], n['uid']) for n in a.near[:3] if n['uid']])
        g = a.ground
        if not a.no_sky and not g['ok']:
            if not g['surface']:
                _f(out, 'note', 'ground_plane',
                   f'{a.tag} has nothing under its base: mount it on a flat metal surface (the roof, or a plate at '
                   f'least {GROUND_PLANE_M * 200:.0f} cm across) as a ground plane against multipath.', obj)
            else:
                _f(out, 'note', 'ground_plane',
                   f'{a.tag} stands on {g["surface"]} ({KIND_WORD.get(g["kind"], g["kind"])}), which does not reach '
                   f'{GROUND_PLANE_M * 100:.0f} cm round its base at its height: give it a ground plane (a metal plate '
                   f'at least {GROUND_PLANE_M * 200:.0f} cm across) unless the antenna has its own.', obj)
        if a.own_frame:
            _f(out, 'note', 'tilt', f'{a.tag} is tilted {a.tilt_deg:.0f}°: its sky and mask are taken about its own '
                                    'axis.', obj)
    # ---- receivers, antenna counts and the heading baseline
    ants = res.antennas
    ins = res.imu
    receivers = [i for i in res.imus if i['antennas'] > 0]
    if ants and not receivers:
        _f(out, 'warning', 'no_receiver',
           f'{", ".join(a.tag for a in ants)} need{"s" if len(ants) == 1 else ""} a GNSS receiver: the rig has no '
           'GNSS/INS with antenna inputs. Add one (Vehicle ▸ Add IMU, a dual-antenna GNSS/INS).',
           [(a.tag, a.uid) for a in ants])
    if ins is not None and ins['antennas'] > 0:
        n_in = ins['antennas']
        if not ants:
            _f(out, 'warning', 'no_antennas',
               f'{ins["tag"]} takes {n_in} GNSS antenna{"s" if n_in > 1 else ""} but the rig has none: no position '
               'fix and no heading, only dead reckoning that drifts. Add an antenna (Vehicle ▸ Add GNSS antenna).',
               [(ins['tag'], ins['uid'])])
        elif n_in >= 2 and len(ants) < 2:
            _f(out, 'warning', 'no_heading',
               f'{ins["tag"]} has two antenna inputs but the rig has one antenna ({ants[0].tag}): no dual-antenna '
               'heading. The INS needs motion to find its heading (none while standing at the start of a drive). Add '
               'a second antenna.', [(ins['tag'], ins['uid']), (ants[0].tag, ants[0].uid)])
        if len(ants) > n_in:
            used = {res.baseline['primary'], res.baseline['secondary']} if res.baseline else {ants[0].tag}
            spare = [a for a in ants if a.tag not in used]
            _f(out, 'note', 'antenna_count',
               f'{len(ants)} antennas, {ins["tag"]} takes {n_in}: {", ".join(a.tag for a in spare)} '
               f'{"is" if len(spare) == 1 else "are"} not used by it (fine for another receiver).',
               [(a.tag, a.uid) for a in spare])
    b = res.baseline
    if b is not None:
        objs = [(b['primary'], b['primary_uid']), (b['secondary'], b['secondary_uid'])]
        hs = b['heading_sigma_deg']
        sig = f'heading σ {hs:.3f}°' if math.isfinite(hs) else 'no heading σ in the INS model'
        what = (f'Heading baseline {b["primary"]} → {b["secondary"]}: {b["horizontal_m"]:.2f} m '
                f'({sig}; {b["rated_sigma_deg"]:g}° at {b["rated_baseline_m"]:g} m rated)')
        if b['horizontal_m'] < BASELINE_SERIOUS_M:
            _f(out, 'serious', 'baseline', f'{what}: shorter than {BASELINE_SERIOUS_M:g} m. Move the antennas apart '
                                           '(front and rear of the roof).', objs)
        elif b['horizontal_m'] < BASELINE_WARN_M:
            _f(out, 'warning', 'baseline', f'{what}: shorter than {BASELINE_WARN_M:g} m; a longer baseline gives a '
                                           'better heading.', objs)
        else:
            _f(out, 'good', 'baseline', f'{what}.', objs)
        if abs(b['vertical_deg']) > BASELINE_TILT_DEG:
            _f(out, 'note', 'baseline_vertical',
               f'The baseline {b["primary"]} → {b["secondary"]} rises {b["vertical_deg"]:.0f}°: tell the INS its '
               f'vertical angle ({b["imu_vertical_deg"]:.1f}° in its frame); the heading follows the '
               f'{b["horizontal_m"]:.2f} m horizontal length.', objs)
    # ---- the IMU's placement and the budget
    if ins is not None:
        io = [(ins['tag'], ins['uid'])]
        others = [i for i in res.imus if i['uid'] != ins['uid']]
        if others:
            _f(out, 'note', 'imu_extra',
               f'The lever arms and the budget are from {ins["tag"]}; {", ".join(i["tag"] for i in others)} '
               f'{"is" if len(others) == 1 else "are"} listed as sensors (choose the reference under Lever arms).',
               io)
        d0 = float(np.linalg.norm(ins['pos']))
        _f(out, 'note', 'imu_origin',
           f'{ins["tag"]} is {d0:.2f} m from the vehicle origin: the ego pose (the nuScenes ego frame, rear axle on '
           'the ground) is its pose moved by that lever arm (in the export).', io)
        lids = visible_of(v, 'lidar')
        if lids:
            dl = [(float(np.linalg.norm(s.pos() - ins['pos'])), s) for s in lids]
            d, near = min(dl, key=lambda x: x[0])
            if d > IMU_LIDAR_NOTE_M:
                _f(out, 'note', 'imu_lidar',
                   f'{ins["tag"]} is {d:.2f} m from the nearest lidar, {near.tag}: a rigid mount near the lidars (on '
                   'the same rig plate) keeps body flex out of their poses.', io + [(near.tag, near.uid)])
            else:
                _f(out, 'good', 'imu_lidar', f'{ins["tag"]} is {d:.2f} m from the nearest lidar, {near.tag}.',
                   io + [(near.tag, near.uid)])
        if max(abs(ins['yaw']), abs(ins['pitch']), abs(ins['roll'])) > 0.01:
            _f(out, 'note', 'imu_axes',
               f'{ins["tag"]}\'s axes are turned from the vehicle\'s (yaw {ins["yaw"]:.2f}°, pitch '
               f'{ins["pitch"]:.2f}°, roll {ins["roll"]:.2f}°, pitch positive down): enter this rotation in the INS '
               '(ins_lever_arms.txt has it).', io)
        bud = res.budget
        inp = bud['inputs']
        missing = [w for w, ok in (('position σ', inp['pos_sigma_cm'] > 0),
                                   ('roll and pitch σ', inp['attitude_sigma_deg'] > 0),
                                   ('heading σ', math.isfinite(inp['heading_sigma_deg'])))
                   if not ok]
        if missing:
            _f(out, 'note', 'model_figures',
               f'{ins["tag"]} ({ins["model"]}) gives no {", ".join(missing)}: the pose error budget leaves '
               f'{"it" if len(missing) == 1 else "them"} out. Enter your unit\'s figures under Error budget.', io)
        rows = [r for r in bud['rows'] if r['kind'] != 'origin']
        if rows and not missing:
            w = max(rows, key=lambda r: r['horizontal_mm'])
            _f(out, 'good', 'budget',
               f'Every sensor\'s position is known to {w["horizontal_mm"]:.0f} mm horizontally or better (1σ; the '
               f'worst, {w["tag"]}, is {w["lever_m"]:.2f} m from {ins["tag"]}); a point {st.label_range_m:g} m from '
               f'it to {w["far_mm"]:.0f} mm.', [(w['tag'], w['uid'])])
    order = {s: i for i, s in enumerate(SEVERITIES)}
    out.sort(key=lambda f: order.get(f['severity'], 9))
    return out


# =========================================================================== the Issues list
def issues(project, vg=None) -> list:
    """The study's problems for the Issues list: an antenna without sky (critical), sky blocked above the mask (> 10 %
    warning, > 25 % serious), near-field multipath, the heading baseline and the antenna / receiver counts. An antenna
    inside a closed body or rig part is left to rules.sensor_issues (one issue, not two); the sky check catches the
    rest (under a roof box, inside another sensor's housing). Cheap after the first call: each antenna's sky is cached
    on its inputs."""
    v = project.variant
    if not has_inputs(project, v):
        return []
    if vg is None and project.vehicle.mesh is not None and project.vehicle.mesh.n_tris:
        vg = _engine().vehicle_geom(project)
    res = _analyse(project, vg, None)
    inside = getattr(rules, '_inside', None)
    out = []
    for f in res.findings:
        if f['severity'] not in ISSUE_SEVERITIES:
            continue
        if f['kind'] == 'no_sky' and vg is not None and inside is not None:
            a = next((a for a in res.antennas if a.uid in f['uids']), None)
            if a is not None and (a.pos[2] - _num(project.sensor_model_for(v.sensor(a.uid)).optical_z_mm) / 1000.0
                                  < -0.005 or inside(vg.occ_mesh, a.pos)):
                continue                        # below the ground or inside: rules.sensor_issues says so
        out.append(rules.Issue(f['severity'], f['message'], ', '.join(f['tags']) or f'{res.variant} · GNSS',
                               list(f['uids'])))
    return out


# =========================================================================== output
def _r(v, d=3):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return round(v, d) + 0.0 if math.isfinite(v) else None


def _vec(a, d=1):
    return [_r(x, d) for x in np.asarray(a, float)]


def sector_text(sec) -> str:
    return f'{sec["az_from"]:.0f}–{sec["az_to"]:.0f}° up to {sec["max_el"]:.0f}°'


def summary(res: GnssResult) -> dict:
    """Compact figures (the assistant, the batch report header)."""
    b = res.baseline
    return {
        'variant': res.variant,
        'antennas': [{'tag': a.tag, 'sky_visible_pct': _r(a.visible * 100, 1), 'mask_deg': a.mask_deg,
                      'expected_satellites': _r(a.expected_sats, 1),
                      'worst_sector': sector_text(a.sectors[0]) if a.sectors else None,
                      'blockers': [{'name': x['name'], 'kind': x['kind'], 'pct': _r(x['share'] * 100, 1)}
                                   for x in a.blockers if x['share'] > 0.0005][:5],
                      'multipath': [f'{n["name"]} {n["dist_m"]:.2f} m, {n["el_deg"]:.0f}°' for n in a.near],
                      'ground_plane': a.ground['ok']} for a in res.antennas],
        'imu': res.imu['tag'] if res.imu else None,
        'baseline': None if b is None else {
            'primary': b['primary'], 'secondary': b['secondary'], 'length_m': _r(b['length_m']),
            'horizontal_m': _r(b['horizontal_m']), 'azimuth_deg': _r(b['azimuth_deg'], 2),
            'vertical_deg': _r(b['vertical_deg'], 2), 'heading_sigma_deg': _r(b['heading_sigma_deg'], 4)},
        'antenna_lever_arms_imu_mm': {r['tag']: _vec(r['imu_mm']) for r in res.lever_arms if r['kind'] == 'gnss'},
        'pose_error_mm': [{'tag': r['tag'], 'horizontal': _r(r['horizontal_mm'], 1),
                           'vertical': _r(r['vertical_mm'], 1),
                           f'at_{res.settings.label_range_m:g}_m': _r(r['far_mm'], 1)}
                          for r in (res.budget or {}).get('rows', [])],
        'findings': [{'severity': f['severity'], 'message': f['message']} for f in res.findings
                     if f['severity'] not in ('good', 'note')],
        'assumptions': ASSUMPTIONS,
    }


def _sky_doc(a: SkyView) -> dict:
    return {
        'tag': a.tag, 'model': a.model, 'phase_centre_m': _vec(a.pos, 4),
        'sky_frame': 'antenna (tilted more than 5°)' if a.own_frame else 'vehicle', 'tilt_deg': _r(a.tilt_deg, 2),
        'mask_deg': a.mask_deg, 'sky_visible_pct': _r(a.visible * 100, 2),
        'sky_visible_above_horizon_pct': _r(a.visible_all * 100, 2), 'expected_satellites': _r(a.expected_sats, 1),
        'satellites_above_mask': a.n_sats,
        'blockers': [{'name': x['name'], 'kind': x['kind'], 'share_above_mask_pct': _r(x['share'] * 100, 2),
                      'share_above_horizon_pct': _r(x['share_all'] * 100, 2), 'max_elevation_deg': _r(x['max_el'], 1),
                      'azimuth_from_deg': _r(x['az_from'], 1), 'azimuth_to_deg': _r(x['az_to'], 1),
                      'nearest_m': _r(x['nearest_m']), 'near_field': x['near_el'] is not None}
                     for x in a.blockers],
        'blocked_sectors': [{'azimuth_from_deg': _r(s['az_from'], 1), 'azimuth_to_deg': _r(s['az_to'], 1),
                             'max_elevation_deg': _r(s['max_el'], 1), 'share_pct': _r(s['share'] * 100, 2),
                             'blockers': s['blockers']} for s in a.sectors],
        'horizon_profile': {'azimuth_deg': _vec(a.az, 2),
                            'max_blocked_elevation_deg': [_r(x, 2) for x in a.horizon],
                            'skyline_deg': [_r(x, 2) for x in a.skyline]},
        'multipath': [{'name': n['name'], 'kind': n['kind'], 'distance_m': _r(n['dist_m']),
                       'elevation_deg': _r(n['el_deg'], 1)} for n in a.near],
        'ground_plane': {'ok': a.ground['ok'], 'stands_on': a.ground['surface'] or None,
                         'ring_radius_m': GROUND_PLANE_M, 'ring_points_without_surface': a.ground['missing']},
    }


def to_doc(res: GnssResult, project=None) -> dict:
    """The JSON export: every antenna's sky view (blockers, sectors, horizon profile), the heading baseline, the lever
    arms in both frames, the pose error budget, the findings and the assumptions."""
    b = res.baseline
    ins = res.imu
    bud = res.budget
    return {
        'schema': SCHEMA,
        'project': getattr(project, 'name', ''), 'variant': res.variant,
        'created': _dt.datetime.now().astimezone().replace(microsecond=0).isoformat(),
        'frame': FRAME,
        'units': 'positions m, lever arms and errors mm, angles degrees (azimuth counter-clockwise from +x, elevation '
                 'above the horizontal); all errors 1 sigma',
        'settings': res.settings.to_dict(),
        'assumptions': ASSUMPTIONS,
        'antennas': [_sky_doc(a) for a in res.antennas],
        'imu': None if ins is None else {
            'tag': ins['tag'], 'model': ins['model'], 'antenna_inputs': ins['antennas'],
            'position_m': _vec(ins['pos'], 4), 'yaw_deg': _r(ins['yaw'], 4),
            'pitch_deg_positive_down': _r(ins['pitch'], 4), 'roll_deg': _r(ins['roll'], 4),
            'rotation_imu_to_vehicle': [[_r(x, 6) for x in row] for row in ins['R']]},
        'baseline': None if b is None else {
            'primary': b['primary'], 'secondary': b['secondary'], 'ins': b['ins'],
            'vector_vehicle_m': _vec(b['vector_m'], 4), 'vector_imu_m': _vec(b['imu_vector_m'], 4),
            'length_m': _r(b['length_m'], 4), 'horizontal_m': _r(b['horizontal_m'], 4),
            'azimuth_vehicle_deg': _r(b['azimuth_deg'], 3), 'vertical_vehicle_deg': _r(b['vertical_deg'], 3),
            'azimuth_imu_deg': _r(b['imu_azimuth_deg'], 3), 'vertical_imu_deg': _r(b['imu_vertical_deg'], 3),
            'heading_sigma_deg': _r(b['heading_sigma_deg'], 5), 'rated_heading_sigma_deg': b['rated_sigma_deg'],
            'rated_baseline_m': b['rated_baseline_m']},
        'lever_arms': [{'target': r['tag'], 'kind': r['kind'], 'role': r['role'] or None,
                        'vehicle_frame_mm': _vec(r['vehicle_mm']), 'imu_frame_mm': _vec(r['imu_mm']),
                        'length_mm': _r(r['length_mm'], 1)} for r in res.lever_arms],
        'pose_error_budget': None if bud is None else {
            'inputs': {k: (_r(v, 5) if isinstance(v, float) else v) for k, v in bud['inputs'].items()},
            'rows': [{'target': r['tag'], 'kind': r['kind'], 'lever_arm_m': _r(r['lever_m']),
                      'ins_position_mm': _vec(r['ins_mm']), 'attitude_on_lever_arm_mm': _vec(r['rotation_mm']),
                      'lever_arm_measurement_mm': _vec(r['measure_mm']), 'position_sigma_xyz_mm': _vec(r['sigma_mm']),
                      'horizontal_mm': _r(r['horizontal_mm'], 1), 'vertical_mm': _r(r['vertical_mm'], 1),
                      'total_mm': _r(r['total_mm'], 1),
                      'at_label_range_horizontal_mm': _r(r['far_mm'], 1),
                      'roll_sigma_deg': _r(r['roll_deg'], 5), 'pitch_sigma_deg': _r(r['pitch_deg'], 5),
                      'heading_sigma_deg': _r(r['heading_deg'], 5)} for r in bud['rows']]},
        'findings': [{'severity': f['severity'], 'kind': f['kind'], 'message': f['message'], 'objects': f['tags']}
                     for f in res.findings],
    }


def ins_text(res: GnssResult, project=None) -> str:
    """ins_lever_arms.txt: the antenna lever arms in the IMU frame (FLU and FRD), the IMU-to-vehicle rotation, the
    heading baseline and the lever arm to the vehicle origin: neutral numbers to type into the INS's own tool."""
    ins = res.imu
    L = []
    w = L.append
    w('# Vantage · INS lever arms')
    w(f'# Project: {getattr(project, "name", "")} · variant: {res.variant}')
    w(f'# Written {_dt.datetime.now().astimezone().replace(microsecond=0).isoformat()}. Neutral values: enter them in '
      'your INS\'s own configuration tool.')
    w('#')
    if ins is None:
        w('# No IMU in this variant: nothing to configure.')
        return '\n'.join(L) + '\n'
    p = ins['pos']
    w(f'# IMU: {ins["tag"]} ({ins["model"]}), centre of its axes at x {p[0]:.4f} y {p[1]:.4f} z {p[2]:.4f} m in the '
      'vehicle frame.')
    w('# Frames:')
    w('#   vehicle  ISO 8855: x forward, y left, z up; origin at the rear-axle centre on the ground')
    w('#   IMU FLU  the IMU\'s own axes as printed on the unit: x forward, y left, z up')
    w('#   IMU FRD  the same axes as x forward, y right, z down (the convention of many INS)')
    w('#')
    w('# Rotation IMU -> vehicle (v = R · u, IMU FLU to vehicle):')
    for row in ins['R']:
        w('#   [' + '  '.join(f'{x: .6f}' for x in row) + ']')
    y, pt, r = ins['yaw'], ins['pitch'], ins['roll']
    w(f'#   intrinsic Z-Y-X, z up: yaw {y:.4f}°, pitch {pt:.4f}° (positive nose down), roll {r:.4f}°')
    w(f'#   the same in FRD (z down, pitch positive nose up): yaw {-y + 0.0:.4f}°, pitch {-pt + 0.0:.4f}°, roll '
      f'{r:.4f}°')
    w('#')
    ants = [x for x in res.lever_arms if x['kind'] == 'gnss']
    w('# Lever arms from the IMU\'s axis centre to each GNSS antenna phase centre, metres')
    w(f'# {"antenna":<10}{"role":<11}{"x_FLU":>9}{"y_FLU":>9}{"z_FLU":>9}{"x_FRD":>10}{"y_FRD":>9}{"z_FRD":>9}')
    if not ants:
        w('# (no GNSS antenna in this variant)')
    for a in ants:
        u = a['imu_mm'] / 1000.0
        w(f'{a["tag"]:<12}{(a["role"] or "-"):<11}{u[0]:>9.4f}{u[1]:>9.4f}{u[2]:>9.4f}{u[0]:>10.4f}{-u[1] + 0.0:>9.4f}'
          f'{-u[2] + 0.0:>9.4f}')
    b = res.baseline
    w('#')
    if b is not None:
        bi = b['imu_vector_m']
        w(f'# Heading baseline {b["primary"]} (primary) -> {b["secondary"]} (secondary): length {b["length_m"]:.4f} m, '
          f'horizontal {b["horizontal_m"]:.4f} m')
        w(f'#   in the IMU frame (FLU): x {bi[0]:.4f} y {bi[1]:.4f} z {bi[2]:.4f} m, azimuth '
          f'{b["imu_azimuth_deg"]:.3f}° from +x counter-clockwise, vertical angle {b["imu_vertical_deg"]:.3f}°')
        w(f'#   in FRD: azimuth {-b["imu_azimuth_deg"] + 0.0:.3f}° clockwise from +x, vertical angle '
          f'{b["imu_vertical_deg"]:.3f}° (up)')
        hs = b['heading_sigma_deg']
        w('#   expected heading σ: ' + (f'{hs:.4f}°' if math.isfinite(hs) else 'not given by the INS model'))
    else:
        w('# No dual-antenna heading baseline in this variant.')
    w('#')
    o = next((x for x in res.lever_arms if x['kind'] == 'origin'), None)
    if o is not None:
        u = o['imu_mm'] / 1000.0
        w('# Output point: lever arm from the IMU to the vehicle frame origin (the nuScenes ego frame), metres')
        w(f'{"origin":<23}{u[0]:>9.4f}{u[1]:>9.4f}{u[2]:>9.4f}{u[0]:>10.4f}{-u[1] + 0.0:>9.4f}{-u[2] + 0.0:>9.4f}')
    return '\n'.join(L) + '\n'


# =========================================================================== the sky plots as PNG
def _bgr(hex_):
    return int(hex_[5:7], 16), int(hex_[3:5], 16), int(hex_[1:3], 16)


_ASCII = {'·': '-', '°': ' deg', '–': '-', '—': '-', '→': '->', '×': 'x', 'σ': 'sigma', '≥': '>=', '≤': '<='}


def _ascii(text) -> str:
    """Text the PNG's Hershey font can draw."""
    t = ''.join(_ASCII.get(ch, ch) for ch in str(text))
    return t.encode('ascii', 'replace').decode('ascii')


def png(res: GnssResult, panel=440) -> bytes:
    """One polar sky plot per antenna (forward up, left to the left, the zenith in the middle, rings every 15° of
    elevation): blocked cells coloured by their blocker (dimmer below the mask), the mask ring dashed, the skyline,
    and a legend of the blockers with their share of the sky above the mask."""
    import cv2
    ants = res.antennas
    n = max(len(ants), 1)
    head, title, legend = 56, 26, 150
    W, H = panel * n, head + title + panel + legend
    img = np.full((H, W, 3), (30, 29, 28), np.uint8)
    font, aa = cv2.FONT_HERSHEY_SIMPLEX, cv2.LINE_AA
    ink, ink2, muted, grid = (236, 235, 230), (183, 194, 195), (129, 135, 137), (70, 69, 66)

    def text(t, x, y, col, s=0.45, th=1):
        cv2.putText(img, _ascii(t), (int(x), int(y)), font, s, col, th, aa)
    text(f'GNSS sky view · {res.variant}', 16, 26, ink, 0.6)
    text('Forward up, left to the left; rings every 15 deg of elevation, the mask dashed', 16, 46, muted, 0.42)
    if not ants:
        text('No GNSS antenna in this variant.', 16, head + 30, ink2, 0.5)
    for k, a in enumerate(ants):
        x0 = k * panel
        y0 = head + title
        R = panel / 2 - 34
        cx, cy = x0 + panel / 2, y0 + panel / 2
        # blocked cells: every pixel of the disc looks up its cell
        yy, xx = np.mgrid[0:panel, 0:panel].astype(float)
        dx, dy = xx - panel / 2 + 0.5, yy - panel / 2 + 0.5
        r = np.hypot(dx, dy) / R
        inside = r <= 1.0
        el = 90.0 * (1.0 - r)
        azp = np.degrees(np.arctan2(-dx, -dy)) % 360.0
        A = len(a.az)
        ai = np.rint(azp / (360.0 / A)).astype(int) % A
        ei = np.clip(np.searchsorted(a.el_edges, el, 'right') - 1, 0, len(a.el_edges) - 2)
        who = np.where(inside, a.who[ei, ai], -1)
        lut = np.array([_bgr(b['color']) for b in a.blockers] + [(0, 0, 0)], np.float32)
        sub = img[y0:y0 + panel, x0:x0 + panel].astype(np.float32)
        below = inside & (el < a.mask_deg)
        sub[below] = sub[below] * 0.75 + np.array([60, 58, 55]) * 0.25
        blk = who >= 0
        alpha = np.where(el < a.mask_deg, 0.45, 0.9)[..., None]
        col = lut[np.where(blk, who, -1)]
        sub = np.where(blk[..., None], sub * (1 - alpha) + col * alpha, sub)
        img[y0:y0 + panel, x0:x0 + panel] = sub.astype(np.uint8)

        def pt(az_deg, el_deg):
            rr = R * (90.0 - el_deg) / 90.0
            t = math.radians(az_deg)
            return int(round((cx - math.sin(t) * rr) * 16)), int(round((cy - math.cos(t) * rr) * 16))
        for e in range(0, 90, 15):
            cv2.circle(img, (int(cx * 16), int(cy * 16)), int(R * (90 - e) / 90 * 16), grid, 1, aa, 4)
            if e:
                text(f'{e}', cx + 3, cy - R * (90 - e) / 90 + 12, muted, 0.32)
        for ang in range(0, 360, 45):
            cv2.line(img, pt(ang, 0), pt(ang, 90), grid, 1, aa, 4)
        rm = R * (90 - a.mask_deg) / 90
        for i in range(0, 72, 2):                      # the mask ring, dashed
            t0, t1 = math.radians(i * 5), math.radians(i * 5 + 5)
            cv2.line(img, (int((cx + math.cos(t0) * rm) * 16), int((cy + math.sin(t0) * rm) * 16)),
                     (int((cx + math.cos(t1) * rm) * 16), int((cy + math.sin(t1) * rm) * 16)), ink2, 1, aa, 4)
        hp = np.where(np.isfinite(a.skyline), a.skyline, 0.0)
        poly = np.array([pt(az, max(h, 0.0)) for az, h in zip(np.r_[a.az, a.az[:1]], np.r_[hp, hp[:1]])], np.int32)
        cv2.polylines(img, [poly], False, ink, 1, aa, 4)
        for lab, ang in (('F', 0), ('L', 90), ('B', 180), ('R', 270)):
            x, y = pt(ang, -9)
            (tw, th), _ = cv2.getTextSize(lab, font, 0.42, 1)
            text(lab, x / 16 - tw / 2, y / 16 + th / 2, ink2, 0.42)
        text(f'{a.tag}  {a.visible * 100:.1f} % open above {a.mask_deg:g} deg  ~{a.expected_sats:.0f} of '
             f'{a.n_sats:.0f} satellites', x0 + 12, head + 16, ink, 0.45)
        y = y0 + panel + 8
        shown = [b for b in a.blockers if b['share_all'] > 0][:6]
        if not shown:
            text('Open sky: nothing of the vehicle blocks it.', x0 + 16, y + 14, ink2, 0.42)
        for b in shown:
            y += 20
            cv2.rectangle(img, (x0 + 16, y - 10), (x0 + 28, y + 2), _bgr(b['color']), -1)
            text(f'{b["name"]} ({KIND_WORD.get(b["kind"], b["kind"])})  {b["share"] * 100:.1f} % above the mask, '
                 f'to {b["max_el"]:.0f} deg', x0 + 36, y, ink2, 0.4)
        if k:
            cv2.line(img, (x0, head), (x0, H - 8), grid, 1)
    ok, buf = cv2.imencode('.png', img)
    return buf.tobytes()


def files(res: GnssResult, project=None) -> list:
    """[(name, bytes)]: gnss_imu.json, ins_lever_arms.txt and gnss_sky.png (which says so when there is no antenna)."""
    return [('gnss_imu.json', (json.dumps(to_doc(res, project), indent=1, ensure_ascii=False) + '\n').encode('utf-8')),
            ('ins_lever_arms.txt', ins_text(res, project).encode('utf-8')),
            ('gnss_sky.png', png(res))]


def export(res: GnssResult, project, out_dir) -> list[str]:
    """Write gnss_imu.json, ins_lever_arms.txt and gnss_sky.png into out_dir; returns their paths."""
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for name, data in files(res, project):
        path = os.path.join(out_dir, name)
        with open(path, 'wb') as fh:
            fh.write(data)
        paths.append(path)
    return paths
