"""Cable harness: every camera's and sensor's cable routed along the vehicle, its length against what its link allows,
and the harness sheet for the people who build it.

The route. A cable leaves its device (the camera's or sensor's position), drops straight to the nearest vertex of the
routing surface, follows the body's outer surface to an entry point (a grommet: the roof, a door seal, the tailgate,
the bulkhead at the base of the windscreen, the rear panel; editable, the defaults placed from the body's shape) and
then runs inside to the compute box (project.ecu) along the trim: down or up a pillar, along the sill or the headliner
and across, an axis-aligned path whose Manhattan length is multiplied by the ECU's routing slack (route_factor, as the
estimate the rest of the app used). A device inside the body (an IMU on the cabin floor, a camera behind the
windscreen, a radar behind the bumper fascia) starts inside; a compute box outside the body (on a roof rack) is reached
along the surface, through an entry point only for devices inside. A GNSS antenna's coax goes to the variant's
GNSS/INS unit with antenna inputs when it has one, else to the receiver beside the compute box. Each lidar's, radar's
and IMU's power cable runs with its data cable from the power distribution beside the compute box (core.power).

The surface. The routing surface is the body without wheels and ignored parts (simplified to about ROUTE_TRIS
triangles when larger, coincident vertices welded) with the visible rig parts, its long triangles split until no edge
is longer than MAX_EDGE_M. Its graph joins the vertices along the triangles' edges and across every pair of nearly
flat neighbouring triangles (within FLAT_DEG, the straight line crossing their shared edge). Separate shells (a mirror,
each rig part) are joined across the smallest gap to the body and across any gap under BRIDGE_M: free air. An edge
costs its length times the worst of its triangles (and an edge between two vertices of glass or a moving part counts
as one): glass_cost on glass, moving_cost on a moving part or shut line (a door, the bonnet, the tailgate, a boot lid,
a hinge, a flap: by part or mount-zone name), free_cost in free air; so a route keeps off the glass and crosses a door
gap only where nothing else reaches, and a door's own harness goes through its door seal. The cheapest path is
Dijkstra's (scipy) from every entry point. A mesh path zig-zags, so it is then pulled straight wherever the straight
line between two of its vertices lies on the surface (flat within PULL_DEG, within ON_SURFACE_M of the surface, no
deeper than CUT_M inside it, off glass and moving parts), and its bends are slid along the mesh while that shortens it
(where it crosses a crease or rounds an obstacle): on a plane the result is the straight line, across a box's edge
within about 1.5 % of the unfolded shortest path (tests/test_harness.py). What is reported is this true length, not
the cost; what a route crosses (glass, moving parts, free air) is measured on the path as found.

The length of a run: (the drop from the device + the surface path + the drop to an outside compute box) × slack (clips,
bends, drip loops) + the inside path + the ECU's service loop + the camera's route extra (route_extra_m; for a sensor
the settings' extra).

Links (typical limits; every cable type says where its figure comes from). A camera's coax against its link's reach
(GMSL2, FPD-Link) and against the module's cable options (the fitted cable_m and the shortest option that reaches). A
lidar's, radar's or IMU's data cable against its interface (100/1000BASE-T1 15 m, BASE-T 100 m, CAN FD about 40 m,
RS-422 about 100 m at 1 Mbit/s); a dual interface ('CAN-FD / 100BASE-T1') is cabled as its Ethernet. A GNSS antenna's
coax: its loss at L1 (dB/m of the cable type × length + the connectors) against the antenna's LNA gain less what the
receiver needs at its input. A power cable: the voltage drop at the device (core.power.cable_check with the power
budget's bus, gauge and limits).

Limits of the model: the inside path is the Manhattan route, not the interior (the mesh has none); the surface is the
outer skin (under the bonnet or behind a bumper fascia the real cable runs just inside it, of about the same length);
bend radii, connector sizes against grommet holes, inline connectors at the grommets, bundles sharing a channel and
EMC separation are not modelled; coax insertion loss is modelled for GNSS only (a camera link's reach is a length).
"""
from __future__ import annotations

import csv
import io
import json
import math
import os
import re
import threading
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass, field

import numpy as np

SCHEMA = 'vantage.harness/1'
FRAME = 'vehicle (ISO 8855, rear axle)'
MAX_EDGE_M = 0.10                # triangles are split until no edge of the routing surface is longer than this
ROUTE_TRIS = 12000               # a larger body is simplified to about this many triangles first
MAX_ROUTE_TRIS = 120000          # the splitting stops at this many triangles
FLAT_DEG = 25.0                  # neighbouring triangles this close to flat are joined across their shared edge
BRIDGE_M = 0.12                  # separate shells this close are joined (free air)
FREE_NOTE_M = 0.30               # a free-air span or drop longer than this is flagged
SNAP_NOTE_M = 0.10               # an entry point this far from the body is moved onto it (noted)
NEAR_CLEAN_M = 0.03              # a device attaches to a vertex off glass and moving parts within this of the nearest
PULL_DEG = 15.0                  # a path is pulled straight over surface whose normals stay within this
ON_SURFACE_M = 0.065             # ... and whose every point lies this close to a vertex (on the surface, not a gap)
CUT_M = 0.015                    # ... and no deeper than this inside it (a chord over a gentle curve, not a corner)
RF_WARN_DB = 3.0                 # a GNSS cable with less RF margin than this is flagged
MOVING_NAME = re.compile(r'door|shut|bonnet|hood|tailgate|liftgate|hatch|boot[\s_-]*lid|trunk[\s_-]*lid|sunroof|'
                         r'hinge|sliding|fuel|charg|flap|wiper', re.I)
GLASS_NAME = re.compile(r'glass|window|windscreen|windshield', re.I)
GLASS, MOVING, FREE = 1, 2, 4                         # edge and face kinds (bits)
SEVERITIES = ('critical', 'serious', 'warning', 'note', 'good')
SEV_RANK = {s: i for i, s in enumerate(SEVERITIES)}
ISSUE_KINDS = ('limit', 'rf', 'glass', 'moving', 'free', 'capacity', 'entry')
ENTRY_KINDS = ('roof', 'door', 'tailgate', 'bulkhead', 'floor', 'other')
ENTRY_LABEL = {'roof': 'Roof grommet', 'door': 'Door seal', 'tailgate': 'Tailgate seal', 'bulkhead': 'Bulkhead grommet',
               'floor': 'Floor grommet', 'other': 'Entry point'}
COMPUTE = ('auto', 'inside', 'outside')
MAX_ENTRIES = 24
COORD_M = (-50.0, 50.0)
EXTRA_M = (0.0, 100.0)
CAPACITY = (0, 99)
LIMITS = {'slack': (1.0, 2.0), 'glass_cost': (1.0, 1000.0), 'moving_cost': (1.0, 1000.0), 'free_cost': (1.0, 100.0),
          'gnss_lna_db': (0.0, 60.0), 'gnss_min_db': (0.0, 60.0), 'connector_db': (0.0, 3.0),
          'warn_share': (0.5, 1.0)}


@dataclass(frozen=True)
class Cable:
    id: str
    label: str
    family: str                 # coax | ethernet | bus | serial | usb | rf | power | other
    limit_m: float | None       # the longest run the link is specified for (typical); None: no length limit here
    ends: tuple                 # typical connectors (device end, far end)
    basis: str                  # where the limit comes from
    db_per_m: float = 0.0       # RF cable: loss at GNSS L1 (1.575 GHz), typical


CABLES = {c.id: c for c in (
    Cable('gmsl2', 'GMSL2 coax', 'coax', 15.0, ('FAKRA', 'FAKRA'),
          'ADI (Maxim) GMSL2 serializers and deserializers: links up to 15 m of coax (typical); the module\'s own '
          'cable options apply first'),
    Cable('fpdlink3', 'FPD-Link III coax', 'coax', 15.0, ('FAKRA', 'FAKRA'),
          'TI FPD-Link III: up to 15 m of coax (typical); the module\'s cable options apply first'),
    Cable('fpdlink4', 'FPD-Link IV coax', 'coax', 15.0, ('FAKRA', 'FAKRA'),
          'TI FPD-Link IV: up to 15 m of coax (typical); the module\'s cable options apply first'),
    Cable('coax', 'Camera coax', 'coax', None, ('FAKRA', 'FAKRA'),
          'an interface without a known reach: the module\'s cable options only'),
    Cable('usb3', 'USB 3 cable', 'usb', 3.0, ('USB-C', 'USB-A'), 'USB 3.x: about 3 m passive (typical)'),
    Cable('usb2', 'USB 2.0 cable', 'usb', 5.0, ('USB', 'USB-A'), 'USB 2.0: 5 m'),
    Cable('mipi', 'MIPI CSI-2 flex', 'coax', 0.3, ('FPC', 'FPC'), 'MIPI CSI-2 on a flex cable: about 0.3 m (typical)'),
    Cable('mgbase_t1', 'Multi-gig BASE-T1', 'ethernet', 15.0, ('H-MTD', 'H-MTD'),
          'IEEE 802.3ch (2.5/5/10GBASE-T1): 15 m link segment'),
    Cable('1000base_t1', '1000BASE-T1', 'ethernet', 15.0, ('H-MTD', 'H-MTD'),
          'IEEE 802.3bp (1000BASE-T1): 15 m link segment (type A)'),
    Cable('100base_t1', '100BASE-T1', 'ethernet', 15.0, ('MATEnet', 'MATEnet'),
          'IEEE 802.3bw (100BASE-T1): 15 m link segment'),
    Cable('mgbase_t', 'Multi-gig BASE-T', 'ethernet', 100.0, ('M12 X-coded', 'RJ45'),
          'IEEE 802.3bz / 802.3an: 100 m channel'),
    Cable('1000base_t', '1000BASE-T', 'ethernet', 100.0, ('M12 X-coded', 'RJ45'),
          'IEEE 802.3ab (1000BASE-T): 100 m channel'),
    Cable('100base_tx', '100BASE-TX', 'ethernet', 100.0, ('M12 D-coded', 'RJ45'),
          'IEEE 802.3u (100BASE-TX): 100 m channel'),
    Cable('can_fd', 'CAN FD', 'bus', 40.0, ('M12 A-coded', 'DB9'),
          'CAN FD at 500 kbit/s arbitration and 2 Mbit/s data: about 40 m of bus (typical; the transceivers\' loop '
          'delay and the sample point set the real figure)'),
    Cable('rs422', 'RS-422', 'serial', 100.0, ('M12 A-coded', 'DB9'),
          'TIA-422 at 1 Mbit/s: about 100 m (rate × length about 10⁸ bit·m/s, typical)'),
    Cable('rf_rg174', 'RG-174 coax', 'rf', None, ('SMA', 'SMA'),
          'thin coax: about 1.1 dB/m at 1.575 GHz (typical, rounded from public catalogue figures)', 1.1),
    Cable('rf_rg58', 'RG-58 coax', 'rf', None, ('TNC', 'SMA'),
          'RG-58: about 0.7 dB/m at 1.575 GHz (typical, rounded from public catalogue figures)', 0.7),
    Cable('rf_lmr240', 'LMR-240 coax', 'rf', None, ('TNC', 'SMA'),
          'low-loss 240-size coax: about 0.34 dB/m at 1.575 GHz (typical, rounded from public catalogue figures)',
          0.34),
    Cable('rf_lmr400', 'LMR-400 coax', 'rf', None, ('TNC', 'N'),
          'low-loss 400-size coax: about 0.17 dB/m at 1.575 GHz (typical, rounded from public catalogue figures)',
          0.17),
    Cable('power', 'Power pair', 'power', None, ('DT 2-pin', 'distribution'),
          'the voltage drop at the device (the power budget\'s bus, gauge and limits)'),
    Cable('other', 'Cable', 'other', None, ('—', '—'), 'no limit known for this interface'),
)}
RF_CABLES = tuple(k for k, c in CABLES.items() if c.family == 'rf')
DATA_CABLES = tuple(k for k, c in CABLES.items() if c.family not in ('rf', 'power'))
ASSUMPTIONS = ('Link limits are typical figures for each link type (their basis is given per cable type), not a '
               'product\'s data sheet; the module\'s cable options are e-con\'s stated lengths. GNSS cable losses are '
               'typical catalogue figures at 1.575 GHz rounded, the LNA gain and the receiver\'s minimum typical: '
               'enter your antenna\'s and receiver\'s. Entry points are placed from the body\'s shape: move them to '
               'the grommets your vehicle has. The inside route is the Manhattan path × the routing slack, not the '
               'interior (the model has none).')
LIMITS_TEXT = ('The route follows the outer skin and the axis-aligned inside path; bend radii, connector sizes against '
               'grommet holes, inline connectors at the grommets, bundles sharing a channel and EMC separation are '
               'not modelled; coax insertion loss is modelled for GNSS only (a camera link\'s reach is a length).')


# =========================================================================== settings
def _bool(v, default=False) -> bool:
    if isinstance(v, (bool, int)) and not isinstance(v, float):
        return bool(v)
    if isinstance(v, str):
        return v.strip().lower() in ('1', 'true', 'yes', 'on')
    return default


def _float(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _clamp(v, lo_hi):
    return min(max(v, lo_hi[0]), lo_hi[1])


def clean_entry(d, i=0) -> dict | None:
    """An entry point as stored ({uid, name, kind, x, y, z, enabled, capacity}), or None when it is not usable."""
    if not isinstance(d, dict):
        return None
    xyz = [_float(d.get(k)) for k in ('x', 'y', 'z')]
    if any(v is None for v in xyz):
        return None
    kind = d.get('kind') if d.get('kind') in ENTRY_KINDS else 'other'
    uid = str(d.get('uid') or f'e{i + 1}')[:24]
    name = str(d.get('name') or ENTRY_LABEL[kind]).strip()[:48] or ENTRY_LABEL[kind]
    cap = _float(d.get('capacity'))
    return {'uid': uid, 'name': name, 'kind': kind, 'x': _clamp(xyz[0], COORD_M), 'y': _clamp(xyz[1], COORD_M),
            'z': _clamp(xyz[2], COORD_M), 'enabled': _bool(d.get('enabled', True), True),
            'capacity': int(_clamp(round(cap), CAPACITY)) if cap is not None else 0}


@dataclass
class HarnessSettings:
    enabled: bool = False              # the routed lengths replace the Manhattan estimate everywhere (Issues, power)
    slack: float = 1.05                # the surface path × this: clips, bends, drip loops
    glass_cost: float = 25.0           # an edge on glass costs this many times its length
    moving_cost: float = 10.0          # on a moving part or shut line
    free_cost: float = 2.0             # across a gap between shells (free air)
    compute: str = 'auto'              # the compute box: auto (from the body) | inside | outside
    gnss_cable: str = 'rf_rg58'        # a GNSS antenna's coax unless set per run
    gnss_lna_db: float = 28.0          # the antenna's LNA gain (typical active antenna)
    gnss_min_db: float = 15.0          # what the receiver needs at its input: LNA gain less the cable's loss (typical)
    connector_db: float = 0.1          # loss of each RF connector (typical)
    warn_share: float = 0.9            # a run longer than this share of its limit is flagged
    entries: list = field(default_factory=list)    # entry points; [] = the defaults from the body
    assign: dict = field(default_factory=dict)     # device uid -> entry uid (absent: the shortest)
    cables: dict = field(default_factory=dict)     # device uid -> cable id (absent: from its interface)
    extra: dict = field(default_factory=dict)      # sensor uid -> extra metres (a camera has route_extra_m)

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d: dict | None) -> 'HarnessSettings':
        s = HarnessSettings()
        d = d if isinstance(d, dict) else {}
        s.enabled = _bool(d.get('enabled', False))
        for k, lo_hi in LIMITS.items():
            v = _float(d.get(k)) if k in d else None
            if v is not None:
                setattr(s, k, _clamp(v, lo_hi))
        if d.get('compute') in COMPUTE:
            s.compute = d['compute']
        if d.get('gnss_cable') in RF_CABLES:
            s.gnss_cable = d['gnss_cable']
        ents, seen = [], set()
        for i, e in enumerate(d['entries'] if isinstance(d.get('entries'), list) else []):
            e = clean_entry(e, i)
            if e is None:
                continue
            while e['uid'] in seen:
                e['uid'] += '_'
            seen.add(e['uid'])
            ents.append(e)
            if len(ents) >= MAX_ENTRIES:
                break
        s.entries = ents
        for k, ok in (('assign', lambda v: isinstance(v, str) and v), ('cables', lambda v: v in CABLES)):
            src = d.get(k)
            setattr(s, k, {str(u)[:40]: v for u, v in (src.items() if isinstance(src, dict) else ()) if ok(v)})
        ex = d.get('extra')
        for u, v in (ex.items() if isinstance(ex, dict) else ()):
            v = _float(v)
            if v is not None:
                s.extra[str(u)[:40]] = _clamp(v, EXTRA_M)
        return s


def settings_of(project) -> HarnessSettings:
    return HarnessSettings.from_dict(getattr(project.analysis, 'harness', None))


# =========================================================================== link types
def camera_cable(module) -> str:
    """The cable type of a camera module's link, from its interface."""
    t = str(getattr(module, 'interface', '') or '')
    if re.search(r'GMSL', t, re.I):
        return 'gmsl2'
    if re.search(r'FPD', t, re.I):
        return 'fpdlink4' if re.search(r'\bIV\b|4', t) else 'fpdlink3'
    if re.search(r'USB', t, re.I):
        return 'usb3' if re.search(r'3', t) else 'usb2'
    if re.search(r'MIPI|CSI', t, re.I):
        return 'mipi'
    return 'coax'


def sensor_cable(model, gnss_cable='rf_rg58') -> str:
    """The cable type of a sensor's data link from its interface; of 'A / B' the Ethernet one, then a bus or serial
    link. A GNSS antenna's (coax) is the settings' RF cable."""
    if getattr(model, 'kind', '') == 'gnss':
        return gnss_cable if gnss_cable in RF_CABLES else 'rf_rg58'
    parts = [q.strip() for q in re.split(r'[/,|+]|\bor\b', str(getattr(model, 'interface', '') or '')) if q.strip()]
    order = (('mgbase_t1', r'(2\.5|5|10)\s*G\s*BASE-?T1'), ('1000base_t1', r'1000\s*BASE-?T1'),
             ('100base_t1', r'100\s*BASE-?T1'), ('mgbase_t', r'(2\.5|5|10)\s*G\s*BASE-?T'),
             ('1000base_t', r'1000\s*BASE-?T'), ('100base_tx', r'100\s*BASE-?T'), ('can_fd', r'\bCAN'),
             ('rs422', r'RS-?4[28]5|RS-?422'), ('rs422', r'RS-?232'))
    for cid, pat in order:
        if any(re.search(pat, q, re.I) for q in parts):
            return cid
    if any(re.search(r'coax|SMA|TNC', q, re.I) for q in parts):
        return gnss_cable if gnss_cable in RF_CABLES else 'rf_rg58'
    return 'other'


# =========================================================================== the routing surface
def _weld(V, F, tol=1e-4):
    """Coincident vertices merged (within tol), unused vertices dropped; degenerate faces flagged."""
    q = np.round(V / tol).astype(np.int64)
    _, idx, inv = np.unique(q, axis=0, return_index=True, return_inverse=True)
    inv = inv.reshape(-1)
    F2 = inv[F]
    ok = (F2[:, 0] != F2[:, 1]) & (F2[:, 1] != F2[:, 2]) & (F2[:, 0] != F2[:, 2])
    V2 = V[idx]
    used, remap = np.unique(F2[ok], return_inverse=True)
    F3 = np.full(F2.shape, -1, np.int64)
    F3[ok] = remap.reshape(-1, 3)
    return V2[used], F3, ok


def _subdivide(V, F, attrs, max_edge, max_tris):
    """Triangles bisected at the midpoint of their longest edge until no edge is longer than max_edge (or there would
    be more than max_tris): a long thin triangle becomes a row of short ones rather than a fan of slivers. A midpoint
    is shared by every triangle that later splits the same edge. attrs: per-face arrays carried to both halves."""
    mids: dict = {}
    for _ in range(40):
        e0, e1, e2 = (np.linalg.norm(V[F[:, j]] - V[F[:, (j + 1) % 3]], axis=1) for j in range(3))
        L = np.stack([e0, e1, e2], 1)
        k = np.argmax(L, axis=1)
        split = L[np.arange(len(F)), k] > max_edge * (1 + 1e-6)
        ns = int(split.sum())
        if not ns or len(F) + ns > max_tris:
            break
        Fs, ks = F[split], k[split]
        rot = np.stack([Fs[np.arange(ns), ks], Fs[np.arange(ns), (ks + 1) % 3], Fs[np.arange(ns), (ks + 2) % 3]], 1)
        a, b, c = rot[:, 0], rot[:, 1], rot[:, 2]
        keys = (np.minimum(a, b) << 32) + np.maximum(a, b)
        uk, inv = np.unique(keys, return_inverse=True)
        idx = np.empty(len(uk), np.int64)
        new = []
        n = len(V)
        for j, kk in enumerate(uk.tolist()):
            v = mids.get(kk)
            if v is None:
                v = mids[kk] = n + len(new)
                new.append(kk)
            idx[j] = v
        if new:
            nk = np.array(new, np.int64)
            V = np.vstack([V, (V[nk >> 32] + V[nk & ((1 << 32) - 1)]) / 2])
        m = idx[inv.reshape(-1)]
        F = np.vstack([F[~split], np.c_[a, m, c], np.c_[m, b, c]])
        attrs = [np.concatenate([x[~split], x[split], x[split]]) for x in attrs]
    return V, F, attrs


def _face_kinds(mesh) -> np.ndarray:
    """Per face: GLASS and/or MOVING bits from its part (role, name, material; rig parts never) and its mount zone."""
    from .mesh import ROLE_GLASS
    pk = np.zeros(max(len(mesh.parts), 1), np.uint8)
    for i, p in enumerate(mesh.parts):
        if p.material == 'rig':
            continue
        if p.role == ROLE_GLASS or any(GLASS_NAME.search(t or '') for t in (p.name, p.material,
                                                                             getattr(p, 'surface', ''))):
            pk[i] |= GLASS
        if MOVING_NAME.search(p.name or ''):
            pk[i] |= MOVING
    k = pk[np.clip(mesh.face_part, 0, len(pk) - 1)]
    if mesh.zones and len(mesh.face_zone) == len(k):
        zk = np.zeros(len(mesh.zones), np.uint8)
        for i, z in enumerate(mesh.zones):
            if GLASS_NAME.search(z.name or ''):
                zk[i] |= GLASS
            if MOVING_NAME.search(z.name or ''):
                zk[i] |= MOVING
        fz = mesh.face_zone.astype(np.int64)
        ok = (fz >= 0) & (fz < len(zk))
        k = k.copy()
        k[ok] |= zk[fz[ok]]
    return k


def _seg_cross(P, Q, A, B):
    """Whether segments PQ and AB pass each other within both (closest points' parameters in [0, 1])."""
    d1, d2, r = Q - P, B - A, P - A
    a = np.einsum('ij,ij->i', d1, d1)
    e = np.einsum('ij,ij->i', d2, d2)
    b = np.einsum('ij,ij->i', d1, d2)
    c = np.einsum('ij,ij->i', d1, r)
    f = np.einsum('ij,ij->i', d2, r)
    den = a * e - b * b
    with np.errstate(divide='ignore', invalid='ignore'):
        s = (b * f - c * e) / den
        t = (a * f - b * c) / den
    return (den > 1e-12 * a * e) & (s > -1e-6) & (s < 1 + 1e-6) & (t > -1e-6) & (t < 1 + 1e-6)


class RouteGraph:
    """The routing surface of one body (and rig) for given costs: its vertices, the cost graph, which vertices are
    clean (off glass and moving parts), the special edges (glass, moving, free air) and the closed body for the inside
    tests. Dijkstra trees are computed on demand and kept per source vertex."""

    def __init__(self, key, V, F, fk, solid, costs, stats):
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import connected_components
        from scipy.spatial import cKDTree
        self.key = key
        self.V, self.F, self.fk = V, F, fk
        self.solid = solid                       # core.mesh.Mesh: the body (wheels in) for inside tests, or None
        self.costs = costs                       # (glass, moving, free)
        n = len(V)
        g_cost, m_cost, f_cost = costs
        # ---- triangle edges, kinds from their faces
        m = len(F)
        E = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
        opp = np.concatenate([F[:, 2], F[:, 0], F[:, 1]])
        fi = np.tile(np.arange(m), 3)
        Es = np.sort(E, axis=1)
        key_ = Es[:, 0] * n + Es[:, 1]
        order = np.argsort(key_, kind='stable')
        ks, fio, oppo, Eso = key_[order], fi[order], opp[order], Es[order]
        uk, start = np.unique(ks, return_index=True)
        kind_e = np.bitwise_or.reduceat(fk[fio], start) if len(uk) else np.zeros(0, np.uint8)
        rows, cols, kinds = [uk // n], [uk % n], [kind_e]
        # ---- across nearly flat neighbours (the straight line over their shared edge)
        T = V[F]
        N = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
        N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-20)
        same = np.nonzero(ks[1:] == ks[:-1])[0]
        if len(same):
            f1, f2 = fio[same], fio[same + 1]
            o1, o2 = oppo[same], oppo[same + 1]
            flat = np.einsum('ij,ij->i', N[f1], N[f2]) > math.cos(math.radians(FLAT_DEG))
            cross = _seg_cross(V[o1], V[o2], V[Eso[same, 0]], V[Eso[same, 1]])
            keep = flat & cross & (o1 != o2)
            rows.append(np.minimum(o1[keep], o2[keep]))
            cols.append(np.maximum(o1[keep], o2[keep]))
            kinds.append(fk[f1[keep]] | fk[f2[keep]])
        r, c, kd = np.concatenate(rows), np.concatenate(cols), np.concatenate(kinds).astype(np.uint8)
        # an edge between two vertices of glass or a moving part counts as one (a chord along a one-triangle band
        # would otherwise slip through it)
        vflag = np.zeros(n, np.uint8)
        for j in range(3):
            np.bitwise_or.at(vflag, F[:, j], fk)
        kd |= vflag[r] & vflag[c]
        # ---- separate shells joined across their smallest gaps (free air)
        A0 = csr_matrix((np.ones(len(r)), (r, c)), shape=(n, n))
        ncomp, lab = connected_components(A0, directed=False)
        self.components = int(ncomp)
        tree = cKDTree(V)
        if ncomp > 1:
            br, bc = [], []
            sizes = np.bincount(lab)
            main = int(np.argmax(sizes))
            main_idx = np.nonzero(lab == main)[0]
            main_tree = cKDTree(V[main_idx])
            for comp in range(ncomp):
                if comp == main:
                    continue
                idx = np.nonzero(lab == comp)[0]
                d, j = main_tree.query(V[idx])
                k = int(np.argmin(d))
                br.append(idx[k])
                bc.append(main_idx[j[k]])
                near = np.nonzero(d <= BRIDGE_M)[0]
                if len(near):                                  # the shortest few more where the shells touch
                    near = near[np.argsort(d[near])[:24]]
                    br += idx[near].tolist()
                    bc += main_idx[j[near]].tolist()
                dd, jj = tree.query(V[idx], k=min(8, n), distance_upper_bound=BRIDGE_M)   # other small shells
                dd, jj = np.atleast_2d(dd), np.atleast_2d(jj)
                ok = np.isfinite(dd) & (jj < n)
                ok[ok] &= (lab[jj[ok]] != comp) & (lab[jj[ok]] != main)
                ii, kk = np.nonzero(ok)
                br += idx[ii].tolist()
                bc += jj[ii, kk].tolist()
            br, bc = np.array(br, np.int64), np.array(bc, np.int64)
            r = np.concatenate([r, np.minimum(br, bc)])
            c = np.concatenate([c, np.maximum(br, bc)])
            kd = np.concatenate([kd, np.full(len(br), FREE, np.uint8)])
        L = np.linalg.norm(V[r] - V[c], axis=1)
        mult = np.ones(len(L))
        mult = np.where(kd & GLASS, np.maximum(mult, g_cost), mult)
        mult = np.where(kd & MOVING, np.maximum(mult, m_cost), mult)
        mult = np.where(kd & FREE, np.maximum(mult, f_cost), mult)
        cost = L * mult + 1e-9
        # one edge per pair: the cheapest
        ek = r * n + c
        o = np.lexsort((cost, ek))
        ek, cost, kd, r, c = ek[o], cost[o], kd[o], r[o], c[o]
        first = np.r_[True, ek[1:] != ek[:-1]]
        r, c, cost, kd, ek = r[first], c[first], cost[first], kd[first], ek[first]
        ok = r != c
        r, c, cost, kd, ek = r[ok], c[ok], cost[ok], kd[ok], ek[ok]
        self.csr = csr_matrix((cost, (r, c)), shape=(n, n))
        sp = kd != 0
        self.special = dict(zip(ek[sp].tolist(), kd[sp].tolist()))
        vn = np.zeros((n, 3))
        an = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])               # area-weighted face normals
        for j in range(3):
            np.add.at(vn, F[:, j], an)
        self.vn = vn / np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-20)
        self.vflag = vflag
        self.clean = vflag == 0
        self.tree = tree
        ci = np.nonzero(self.clean)[0]
        self.clean_idx = ci
        self.clean_tree = cKDTree(V[ci]) if len(ci) else None
        self.n_edges = len(r)
        self.stats = dict(stats, vertices=n, triangles=m, edges=int(self.n_edges), shells=int(ncomp))
        self._trees: OrderedDict = OrderedDict()
        self._tlock = threading.Lock()
        self._adj = None
        self.cache = {}                          # what depends on this surface alone (the default entry points)

    # ---- queries
    def nearest(self, p, clean=False) -> tuple[int, float]:
        """(vertex, distance) nearest to p; clean: off glass and moving parts (any vertex when none is clean)."""
        p = np.asarray(p, float)
        if clean and self.clean_tree is not None:
            d, j = self.clean_tree.query(p)
            return int(self.clean_idx[j]), float(d)
        d, j = self.tree.query(p)
        return int(j), float(d)

    def attach(self, p) -> tuple[int, float]:
        """Where a device's cable reaches the surface: the nearest vertex, or a clean one hardly farther."""
        v, d = self.nearest(p)
        if not self.clean[v]:
            vc, dc = self.nearest(p, clean=True)
            if dc <= d + NEAR_CLEAN_M:
                return vc, dc
        return v, d

    def dijkstra(self, src: int):
        """(cost (n,), predecessor (n,)) from vertex src, cached."""
        with self._tlock:
            hit = self._trees.get(src)
            if hit is not None:
                self._trees.move_to_end(src)
                return hit
        from scipy.sparse.csgraph import dijkstra
        dist, pred = dijkstra(self.csr, directed=False, indices=src, return_predecessors=True)
        with self._tlock:
            self._trees[src] = (dist, pred)
            while len(self._trees) > 48:
                self._trees.popitem(last=False)
        return dist, pred

    def path(self, src: int, dst: int) -> list | None:
        """The cheapest vertex path src → dst (inclusive), or None when dst cannot be reached."""
        dist, pred = self.dijkstra(src)
        if not math.isfinite(dist[dst]):
            return None
        out = [dst]
        v = dst
        while v != src:
            v = int(pred[v])
            if v < 0:
                return None
            out.append(v)
        return out[::-1]

    def straight(self, a: int, b: int) -> bool:
        """Whether the straight line between vertices a and b lies on the surface: every point of it within
        ON_SURFACE_M of a vertex, none near glass or a moving part, and the surface under it flat (the normals away
        from its ends within PULL_DEG of their mean), so it neither cuts a corner nor spans a gap."""
        A, B = self.V[a], self.V[b]
        L = float(np.linalg.norm(B - A))
        if L < 1e-12:
            return True
        m = max(2, int(L / 0.02))
        t = np.arange(1, m) / m
        S = A + (B - A) * t[:, None]
        dk, ik = self.tree.query(S, k=4)
        d, idx = dk[:, 0], ik[:, 0]
        if d.max() > ON_SURFACE_M or self.vflag[idx].any():
            return False
        # inside the body: below the tangent plane of every nearby vertex (a corner cut, not a gentle curve)
        sd = np.einsum('kij,kij->ki', S[:, None, :] - self.V[ik], self.vn[ik]).max(axis=1)
        if sd.min() < -CUT_M:
            return False
        inner = (t * L > 0.07) & ((1 - t) * L > 0.07)
        if inner.sum() >= 2:
            Nn = self.vn[idx[inner]]
            mean = Nn.sum(axis=0)
            mean /= max(float(np.linalg.norm(mean)), 1e-12)
            if float((Nn @ mean).min()) < math.cos(math.radians(PULL_DEG)):
                return False
        return True

    def neighbours(self, v: int) -> np.ndarray:
        if self._adj is None:
            self._adj = (self.csr + self.csr.T).tocsr()
        return self._adj.indices[self._adj.indptr[v]:self._adj.indptr[v + 1]]

    def pull(self, path) -> list:
        """The vertex path pulled straight wherever the line between its vertices lies on the surface (straight()):
        the zig-zag of a mesh path removed. Then each bend is slid to the neighbouring vertex that shortens the path
        while both lines to it stay on the surface (where it crosses a crease or rounds an obstacle). Stretches
        over glass, moving parts or free air are kept as found."""
        if path is None or len(path) < 3:
            return path
        P = np.asarray(path, np.int64)
        n = len(P)
        nv = len(self.V)
        a_, b_ = np.minimum(P[1:], P[:-1]), np.maximum(P[1:], P[:-1])
        special = np.array([self.special.get(k, 0) for k in (a_ * nv + b_).tolist()], np.uint8) != 0
        bad = np.concatenate([[0], np.cumsum(special)])          # special edges before each point

        def ok(i, j):
            return j <= i + 1 or (bad[j] == bad[i] and self.straight(int(P[i]), int(P[j])))
        out, i = [0], 0
        while i < n - 1:
            good, step = i + 1, 1
            while good < n - 1:                                      # double while it holds, then bisect
                j = min(i + 2 * step, n - 1)
                if not ok(i, j):
                    lo, hi = good, j
                    while hi - lo > 1:
                        mid = (lo + hi) // 2
                        if ok(i, mid):
                            lo = mid
                        else:
                            hi = mid
                    good = lo
                    break
                good, step = j, 2 * step
            out.append(good)
            i = good
        Q = [int(P[k]) for k in out]
        # a bend next to a special stretch stays where it is
        fixed = [True] + [bool(special[out[k - 1]:out[k]].any() or special[out[k]:out[k + 1]].any())
                          for k in range(1, len(out) - 1)] + [True]
        V = self.V
        for _ in range(40):                                          # slide the bends while it shortens the path
            moved = False
            k = 1
            while k < len(Q) - 1:
                if fixed[k]:
                    k += 1
                    continue
                a, v, b = Q[k - 1], Q[k], Q[k + 1]
                if self.straight(a, b):                              # no bend needed any more
                    del Q[k], fixed[k]
                    moved = True
                    continue
                best = float(np.linalg.norm(V[a] - V[v]) + np.linalg.norm(V[v] - V[b]))
                choice = None
                for u in self.neighbours(v).tolist():
                    if u in (a, b) or self.vflag[u]:
                        continue
                    L = float(np.linalg.norm(V[a] - V[u]) + np.linalg.norm(V[u] - V[b]))
                    if L < best - 1e-9 and self.straight(a, u) and self.straight(u, b):
                        best, choice = L, u
                if choice is not None:
                    Q[k] = choice
                    moved = True
                k += 1
            if not moved:
                break
        return Q

    def measure(self, path) -> dict:
        """Length of a vertex path, and how much of it runs on glass, moving parts and free air."""
        if path is None or len(path) < 2:
            return {'length': 0.0, 'glass': 0.0, 'moving': 0.0, 'free': 0.0}
        P = np.asarray(path, np.int64)
        seg = np.linalg.norm(self.V[P[1:]] - self.V[P[:-1]], axis=1)
        a, b = np.minimum(P[1:], P[:-1]), np.maximum(P[1:], P[:-1])
        n = len(self.V)
        kinds = np.array([self.special.get(k, 0) for k in (a * n + b).tolist()], np.uint8)
        return {'length': float(seg.sum()), 'glass': float(seg[(kinds & GLASS) > 0].sum()),
                'moving': float(seg[(kinds & MOVING) > 0].sum()), 'free': float(seg[(kinds & FREE) > 0].sum())}

    def inside(self, p) -> bool:
        """Whether p is inside the closed body: rays along the six axes leave it through a face (its normal along the
        ray) for at least five of them (a body without a floor still counts)."""
        if self.solid is None or not self.solid.n_tris:
            return False
        d = np.array([[0, 0, 1], [0, 0, -1], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]], float)
        _, f = self.solid.raycast(np.repeat(np.asarray(p, float)[None], 6, axis=0), d)
        hit = f >= 0
        if hit.sum() < 5:
            return False
        return int((np.einsum('ij,ij->i', self.solid.normals[f[hit]], d[hit]) > 0).sum()) >= 5


_lock = threading.RLock()
_GRAPHS: OrderedDict = OrderedDict()
_ROUTES: OrderedDict = OrderedDict()


def graph_key(project, st: HarnessSettings) -> tuple | None:
    m = project.vehicle.mesh
    if m is None or not m.n_tris:
        return None
    from .rig import rig_key
    rig = [r for r in getattr(project, 'rig', ()) if r.visible]
    return (id(m), m.n_tris, id(m.face_zone), id(m.face_part),
            tuple((p.name, p.role, p.material, getattr(p, 'surface', '')) for p in m.parts),
            tuple((z.name, z.kind) for z in m.zones), rig_key(rig), round(st.glass_cost, 6),
            round(st.moving_cost, 6), round(st.free_cost, 6))


def route_graph(project, st: HarnessSettings | None = None) -> RouteGraph | None:
    """The routing surface of the project's body and visible rig parts for the settings' costs (cached)."""
    st = st or settings_of(project)
    key = graph_key(project, st)
    if key is None:
        return None
    with _lock:
        g = _GRAPHS.get(key)
        if g is not None:
            _GRAPHS.move_to_end(key)
            return g
    t0 = time.perf_counter()
    from .mesh import ROLE_GLASS, ROLE_MOUNTABLE, ROLE_WHEELS
    from .rig import merge, rig_mesh
    mesh = project.vehicle.mesh
    occ = mesh.part_mask({ROLE_MOUNTABLE, ROLE_GLASS, ROLE_WHEELS})
    base = mesh.subset(occ) if not occ.all() else mesh
    simplified = base.n_tris > ROUTE_TRIS
    if simplified:
        base, _ = base.decimate(ROUTE_TRIS)
    solid = base                                         # the closed body (wheels in) for inside tests
    rig = [r for r in getattr(project, 'rig', ()) if r.visible]
    surf = merge(base, rig_mesh(rig))
    roles = np.array([p.role for p in surf.parts] or [ROLE_MOUNTABLE])
    keep = np.isin(roles[np.clip(surf.face_part, 0, len(roles) - 1)], [ROLE_WHEELS], invert=True)
    if not keep.any():
        return None
    fk = _face_kinds(surf)[keep]
    V, F, ok = _weld(surf.vertices, surf.faces[keep])
    fk = fk[ok]
    F = F[ok]
    V, F, (fk,) = _subdivide(V, F, [fk], MAX_EDGE_M, MAX_ROUTE_TRIS)
    g = RouteGraph(key, V, F, fk, solid, (st.glass_cost, st.moving_cost, st.free_cost),
                   {'simplified': bool(simplified), 'seconds': 0.0})
    g.stats['seconds'] = time.perf_counter() - t0
    with _lock:
        _GRAPHS[key] = g
        while len(_GRAPHS) > 3:
            _GRAPHS.popitem(last=False)
    return g


def clear_cache():
    with _lock:
        _GRAPHS.clear()
        _ROUTES.clear()
        _ONE.clear()


# =========================================================================== entry points
def _body_dims(project):
    """(lo, hi, front axle x, rear axle x) of the body, without the exterior mirrors."""
    from .mesh import ROLE_GLASS, ROLE_MOUNTABLE, is_mirror
    m = project.vehicle.mesh
    sel = m.part_mask({ROLE_MOUNTABLE, ROLE_GLASS})
    mirror = np.array([is_mirror(p.name) for p in m.parts] or [False])
    mirror = mirror[np.clip(m.face_part, 0, len(mirror) - 1)]
    if (sel & ~mirror).any():
        sel = sel & ~mirror
    v = m.tri[sel].reshape(-1, 3) if sel.any() else m.vertices
    lo, hi = v.min(axis=0), v.max(axis=0)
    wc = project.vehicle.wheel_centres or {}
    L = hi[0] - lo[0]
    xf = float(wc['FL'][0]) if 'FL' in wc else float(lo[0] + 0.8 * L)
    xr = float(wc['RL'][0]) if 'RL' in wc else float(lo[0] + 0.2 * L)
    return lo, hi, xf, xr


def _top_z(g, x, y, default):
    """Height of the body's top surface at (x, y) (a ray straight down onto the closed body), else default."""
    if g is None or g.solid is None or not g.solid.n_tris:
        return default
    z0 = float(g.solid.bbox[1][2]) + 1.0
    t, f = g.solid.raycast(np.array([[x, y, z0]]), np.array([[0.0, 0.0, -1.0]]))
    return z0 - float(t[0]) if f[0] >= 0 else default


def default_entries(project, g: RouteGraph | None = None) -> list:
    """Entry points placed from the body's shape, each on the nearest vertex off glass and moving parts: a roof
    grommet near the roof's rear edge; a bulkhead grommet at the base of the windscreen (the cowl); a door seal on each
    side at the front door's leading edge (the A-pillar) and one at the rear door's trailing edge (from the shut lines
    when the body has them); a tailgate seal at the top of the rear opening; a rear-panel grommet behind the rear
    bumper."""
    m = project.vehicle.mesh
    if m is None or not m.n_tris:
        return []
    g = g or route_graph(project)
    ck = ('entries', json.dumps(project.vehicle.wheel_centres, sort_keys=True, default=str))
    if g is not None and ck in g.cache:
        return [dict(e) for e in g.cache[ck]]
    from .rig import roof
    lo, hi, xf, xr = _body_dims(project)
    rxf, rxr, hw, top = roof(m)
    H, W = hi[2] - lo[2], hi[1] - lo[1]
    kinds = _face_kinds(m)
    c, n = m.centroids, m.normals
    # the windscreen's base: the front-most glass facing forward near the middle; else behind the front axle
    ws = ((kinds & GLASS) > 0) & (c[:, 0] > (lo[0] + hi[0]) / 2) & (np.abs(c[:, 1]) < 0.3 * W) & (n[:, 0] > 0.25)
    bx = float(c[ws, 0].max()) + 0.08 if ws.any() else xf - 0.25
    bz = _top_z(g, bx, 0.0, lo[2] + 0.6 * H)
    # the doors' leading and trailing edges: the front-most and rear-most shut lines on the sides, else the A- and
    # C-pillar regions from the axles
    mv = ((kinds & MOVING) > 0) & (np.abs(c[:, 1]) > 0.3 * W) & (c[:, 2] > lo[2] + 0.25 * H) & (c[:, 0] < xf) & \
        (c[:, 0] > xr - 0.2)
    dxf = float(c[mv, 0].max()) + 0.04 if mv.any() else xf - 0.55
    dxr = float(c[mv, 0].min()) - 0.04 if mv.any() else xr + 0.35
    dz = lo[2] + 0.55 * H
    targets = [('roof', 'roof', 'Roof grommet', (rxr + 0.12 * (rxf - rxr), 0.0, top)),
               ('bulkhead', 'bulkhead', 'Bulkhead grommet', (bx, 0.0, bz)),
               ('door_fl', 'door', 'Door seal front left', (dxf, hi[1], dz)),
               ('door_fr', 'door', 'Door seal front right', (dxf, lo[1], dz)),
               ('door_rl', 'door', 'Door seal rear left', (dxr, hi[1], dz)),
               ('door_rr', 'door', 'Door seal rear right', (dxr, lo[1], dz)),
               ('tailgate', 'tailgate', 'Tailgate seal', (lo[0], 0.3 * hw, top - 0.12)),
               ('rear_panel', 'bulkhead', 'Rear panel grommet', (lo[0], 0.25 * hw, lo[2] + 0.4 * H))]
    out = []
    for uid, kind, name, p in targets:
        pos = np.array(p, float)
        if g is not None:
            v, _ = g.nearest(pos, clean=True)
            pos = g.V[v]
        out.append({'uid': uid, 'name': name, 'kind': kind, 'x': round(float(pos[0]), 4), 'y': round(float(pos[1]), 4),
                    'z': round(float(pos[2]), 4), 'enabled': True, 'capacity': 0})
    if g is not None:
        g.cache[ck] = [dict(e) for e in out]
    return out


def entries_of(project, st: HarnessSettings | None = None, g: RouteGraph | None = None) -> list:
    """The entry points in use: the settings' own, or the defaults from the body."""
    st = st or settings_of(project)
    return [dict(e) for e in st.entries] if st.entries else default_entries(project, g)


# =========================================================================== routing
@dataclass
class HarnessResult:
    settings: HarnessSettings
    runs: list                  # per cable: a dict (see run())
    entries: list               # {uid, name, kind, pos (snapped), asked, snap_m, enabled, capacity, cables}
    compute: dict               # the compute box: {name, location, pos, inside}
    findings: list              # [{'severity', 'kind', 'message', 'tags', 'uids', 'where'}], worst first
    totals: dict
    routed: bool                # False: no body, every run is the Manhattan estimate
    variant: str = ''
    variant_uid: str = ''
    metrics: dict = field(default_factory=dict)

    def run(self, rid):
        return next((r for r in self.runs if r['id'] == rid), None)

    def runs_of(self, uid):
        return [r for r in self.runs if r['uid'] == uid]

    def data_run(self, uid):
        return next((r for r in self.runs if r['uid'] == uid and r['what'] != 'power'), None)

    def entry(self, uid):
        return next((e for e in self.entries if e['uid'] == uid), None)

    def worst(self) -> str:
        return self.findings[0]['severity'] if self.findings else 'good'


def _variant(project, variant):
    var = project.variant_by_uid(variant) if isinstance(variant, str) else variant
    return var or project.variant


def _finite(*vals) -> bool:
    return all(isinstance(v, (int, float)) and math.isfinite(v) for v in vals)


def _num(v, default=0.0) -> float:
    v = _float(v)
    return v if v is not None and v >= 0 else default


def inside_path(a, b) -> np.ndarray:
    """The axis-aligned inside path from a to b: vertical first (a pillar), then along x (the sill or headliner),
    then across."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    pts = [a, np.array([a[0], a[1], b[2]]), np.array([b[0], a[1], b[2]]), b]
    out = [pts[0]]
    for q in pts[1:]:
        if np.linalg.norm(q - out[-1]) > 1e-9:
            out.append(q)
    return np.array(out)


def simplify(P, tol=0.004) -> np.ndarray:
    """The polyline P (n, 3) with the points that lie within tol of the line through their neighbours dropped
    (Ramer-Douglas-Peucker): the drawn and exported path; lengths are measured on the full path."""
    P = np.asarray(P, float)
    if len(P) < 3:
        return P
    keep = np.zeros(len(P), bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(P) - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        a, b = P[i], P[j]
        ab = b - a
        L2 = float(ab @ ab)
        Q = P[i + 1:j] - a
        if L2 < 1e-18:
            d = np.linalg.norm(Q, axis=1)
        else:
            t = np.clip(Q @ ab / L2, 0.0, 1.0)
            d = np.linalg.norm(Q - t[:, None] * ab, axis=1)
        k = int(np.argmax(d))
        if d[k] > tol:
            m = i + 1 + k
            keep[m] = True
            stack += [(i, m), (m, j)]
    return P[keep]


def manhattan(a, b) -> float:
    return float(np.abs(np.asarray(a, float) - np.asarray(b, float)).sum())


def _devices(project, var, st):
    """Visible cameras and sensors with finite positions: (uid, tag, kind, slot, pos, model, extra, cable id)."""
    out, skipped = [], []
    for c in var.cameras:
        if not c.visible:
            continue
        if not _finite(c.x, c.y, c.z):
            skipped.append(c.tag)
            continue
        mod = project.module_for(c)
        cid = st.cables.get(c.uid) if st.cables.get(c.uid) in DATA_CABLES else camera_cable(mod)
        out.append({'uid': c.uid, 'tag': c.tag, 'kind': 'camera', 'slot': c.slot, 'pos': c.pos().astype(float),
                    'model': mod.name, 'extra': _num(c.route_extra_m), 'cable': cid, 'obj': c, 'module': mod})
    count = {}
    for s in var.sensors:
        if not s.visible:
            continue
        if not _finite(s.x, s.y, s.z):
            skipped.append(s.tag)
            continue
        m = project.sensor_model_for(s)
        own = st.cables.get(s.uid)
        if m.kind == 'gnss':
            cid = own if own in RF_CABLES else sensor_cable(m, st.gnss_cable)
        else:
            cid = own if own in DATA_CABLES else sensor_cable(m, st.gnss_cable)
        out.append({'uid': s.uid, 'tag': s.tag, 'kind': m.kind, 'slot': count.get(m.kind, 0),
                    'pos': s.pos().astype(float),
                    'model': f'{m.vendor} {m.name}'.strip(), 'extra': st.extra.get(s.uid, 0.0), 'cable': cid, 'obj': s,
                    'smodel': m})
        count[m.kind] = count.get(m.kind, 0) + 1
    return out, skipped


def _endpoint(g: RouteGraph | None, pos, inside=None) -> dict:
    """Where a cable end meets the routing: inside the body, or its attach vertex and the drop to it."""
    pos = np.asarray(pos, float)
    if g is None:
        return {'pos': pos, 'inside': True if inside is None else inside, 'v': -1, 'drop': 0.0}
    ins = g.inside(pos) if inside is None else inside
    if ins:
        return {'pos': pos, 'inside': True, 'v': -1, 'drop': 0.0}
    v, d = g.attach(pos)
    return {'pos': pos, 'inside': False, 'v': v, 'drop': d}


def _route(g, a, b, entries, rf, forced=None):
    """The cheapest route from end a to end b (dicts from _endpoint), through an entry point when one end is inside
    and the other outside. Returns {'mode', 'surface' (vertex path in the direction of travel, or None), 'entry',
    'inside' ((p, q) of the inside part, or None), 'fallback' ('' or why the Manhattan estimate is used)}. Modes:
    'out_in' (along the surface, in at the entry), 'in_out' (inside, out at the entry, along the surface),
    'out_out' (along the surface only), 'in_in' (inside only; also every fallback)."""
    def manhattan_only(why):
        return {'mode': 'in_in', 'surface': None, 'entry': None, 'inside': (a['pos'], b['pos']), 'fallback': why}
    if g is None:
        return manhattan_only('nomesh')
    if a['inside'] and b['inside']:
        return manhattan_only('')
    if not a['inside'] and not b['inside']:
        path = g.path(a['v'], b['v'])
        if path is None:
            return manhattan_only('unreachable')
        return {'mode': 'out_out', 'surface': path, 'entry': None, 'inside': None, 'fallback': ''}
    out_end, in_end = (a, b) if not a['inside'] else (b, a)
    cand = [e for e in entries if e['enabled'] and e.get('v', -1) >= 0]
    if forced is not None:
        cand = [e for e in cand if e['uid'] == forced] or cand
    best = None
    for e in cand:
        dist, _ = g.dijkstra(e['v'])
        c = float(dist[out_end['v']])
        if not math.isfinite(c):
            continue
        c += manhattan(e['pos'], in_end['pos']) * rf
        if best is None or c < best[0] - 1e-12:
            best = (c, e)
    if best is None:
        return manhattan_only('noentry' if not cand else 'unreachable')
    e = best[1]
    path = g.path(e['v'], out_end['v'])                 # the entry → the outside end
    if out_end is a:
        return {'mode': 'out_in', 'surface': path[::-1], 'entry': e, 'inside': (e['pos'], b['pos']), 'fallback': ''}
    return {'mode': 'in_out', 'surface': path, 'entry': e, 'inside': (a['pos'], e['pos']), 'fallback': ''}


def _sev_max(*s):
    s = [x for x in s if x]
    return min(s, key=lambda x: SEV_RANK.get(x, 9)) if s else 'good'


def run(project, engine=None, settings: HarnessSettings | None = None, variant=None, progress=None, cancel=None,
        routed: bool = True) -> HarnessResult | None:
    """The cable harness of a variant (the active one by default; a Variant or its uid): one run per cable (each
    visible camera's link, each sensor's data or RF cable, each lidar's, radar's and IMU's power pair) with
    its route, length and link check. routed=False uses the Manhattan estimate for every run (no geometry; what the
    Issues list checks until routing is switched on). engine is accepted for the module contract and not needed (the
    routing surface is cached here). Returns None when cancelled."""
    t0 = time.perf_counter()
    st = settings or settings_of(project)
    var = _variant(project, variant)

    def prog(f):
        if progress is not None:
            progress(f)

    def stop():
        return cancel is not None and cancel()

    g = route_graph(project, st) if routed else None
    prog(0.3)
    if stop():
        return None
    e = project.ecu
    rf = _num(e.route_factor, 1.0) or 1.0
    loop = _num(e.service_loop_m)
    findings = []
    # ---- entry points (snapped onto the surface)
    entries = []
    for x in (entries_of(project, st, g) if g is not None else []):
        asked = np.array([x['x'], x['y'], x['z']], float)
        ent = dict(x, asked=asked, pos=asked.copy(), snap_m=0.0, v=-1, cables=[])
        if g is not None:
            v, d = g.nearest(asked, clean=True)
            ent.update(v=v, pos=g.V[v].copy(), snap_m=d)
            if d > SNAP_NOTE_M:
                findings.append(_finding('note', 'entry_snap', f'{x["name"]} is {d:.2f} m off the body: it is taken '
                                         f'on the surface at ({g.V[v][0]:.2f}, {g.V[v][1]:.2f}, {g.V[v][2]:.2f}).'))
        entries.append(ent)
    # ---- the compute box and the GNSS/INS units
    ecu_pos = np.array([e.x, e.y, e.z], float)
    if not _finite(*ecu_pos):
        ecu_pos = np.array([1.0, 0.0, 0.35])
    ecu_in = None if st.compute == 'auto' else st.compute == 'inside'
    ecu_end = _endpoint(g, ecu_pos, ecu_in)
    compute = {'name': e.name, 'location': e.location, 'pos': ecu_pos, 'inside': ecu_end['inside'],
               'how': 'from the body' if st.compute == 'auto' else 'set'}
    devices, skipped = _devices(project, var, st)
    ins_units = [d for d in devices if d['kind'] == 'imu' and getattr(d.get('smodel'), 'antennas', 0) >= 1]
    ends = {}
    runs = []
    for i, dev in enumerate(devices):
        if stop():
            return None
        prog(0.3 + 0.6 * i / max(len(devices), 1))
        a = ends.get(dev['uid'])
        if a is None:
            a = ends[dev['uid']] = _endpoint(g, dev['pos'])
        to_ins = dev['kind'] == 'gnss' and ins_units
        if to_ins:
            unit = min(ins_units, key=lambda u: float(np.linalg.norm(u['pos'] - dev['pos'])))
            b = ends.get(unit['uid'])
            if b is None:
                b = ends[unit['uid']] = _endpoint(g, unit['pos'])
            dest = {'name': f'{unit["tag"]} (GNSS/INS)', 'uid': unit['uid'], 'pos': unit['pos']}
        else:
            b = ecu_end
            dest = {'name': (f'{e.name}' + (' · GNSS receiver' if dev['kind'] == 'gnss' else '')), 'uid': '',
                    'pos': ecu_pos}
        rt = _route(g, a, b, entries, rf, st.assign.get(dev['uid']))
        base = _assemble(g, dev, a, b, rt, st, rf, loop, dest)
        runs.append(_check_data(project, dev, base, st))
        if dev['kind'] in ('lidar', 'radar', 'imu'):
            runs.append(_check_power(project, dev, base, st))
    prog(0.92)
    # ---- entry use and capacity
    for r in runs:
        if r['entry']:
            ent = next(x for x in entries if x['uid'] == r['entry'])
            ent['cables'].append(r['id'])
    for ent in entries:
        n = len(ent['cables'])
        if ent['capacity'] and n > ent['capacity']:
            tags = sorted({r['tag'] for r in runs if r['entry'] == ent['uid']})
            uids = list(dict.fromkeys(r['uid'] for r in runs if r['entry'] == ent['uid']))
            findings.append(_finding('warning', 'capacity', f'{ent["name"]} carries {n} cables, more than the '
                                     f'{ent["capacity"]} it takes: fit a larger grommet or route some through '
                                     f'another entry point.', tags, uids,
                                     (float(ent['pos'][0]), float(ent['pos'][1]))))
    for t in skipped:
        findings.append(_finding('warning', 'skipped', f'{t}: its position is not a number, so it has no route.', [t]))
    findings += _run_findings(runs, st, g is not None)
    if g is None and routed and project.vehicle.mesh is None:
        findings.append(_finding('note', 'nomesh', 'No vehicle body: every run is the Manhattan estimate '
                                 '(× the routing slack + the service loop).'))
    findings.sort(key=lambda f: SEV_RANK.get(f['severity'], 9))
    totals = _totals(runs, entries)
    res = HarnessResult(st, runs, [_entry_out(x) for x in entries], compute, findings, totals, g is not None,
                        var.name, var.uid)
    res.metrics = {'seconds': time.perf_counter() - t0, 'graph': dict(g.stats) if g is not None else None}
    prog(1.0)
    return res


def _entry_out(x) -> dict:
    return {'uid': x['uid'], 'name': x['name'], 'kind': x['kind'], 'enabled': x['enabled'],
            'capacity': x['capacity'], 'asked': x['asked'], 'pos': x['pos'], 'snap_m': x['snap_m'],
            'cables': list(x['cables'])}


def _assemble(g, dev, a, b, rt, st, rf, loop, dest) -> dict:
    """The geometry and the length of a route: the polyline from the device to the destination, which stretch of it
    is inside (first and last point index), and each part of the length."""
    mode = rt['mode']
    surf = {'length': 0.0, 'glass': 0.0, 'moving': 0.0, 'free': 0.0}
    if rt['surface'] is not None:
        surf = g.measure(rt['surface'])                          # what it crosses: on the path as found
        rt = dict(rt, surface=g.pull(rt['surface']))
        S = g.V[rt['surface']]
        surf['length'] = float(np.linalg.norm(np.diff(S, axis=0), axis=1).sum())     # its length: pulled straight
    drop = (a['drop'] if mode in ('out_in', 'out_out') else 0.0) + (b['drop'] if mode in ('in_out', 'out_out') else 0.0)
    inside_m = manhattan(*rt['inside']) * rf if rt['inside'] is not None else 0.0
    if mode == 'out_in':
        S = g.V[rt['surface']]
        ip = inside_path(S[-1], b['pos'])
        P = np.vstack([a['pos'][None], S, ip[1:]])
        span = (1 + len(S) - 1, len(P) - 1)
    elif mode == 'in_out':
        ip = inside_path(a['pos'], g.V[rt['surface'][0]])
        P = np.vstack([ip, g.V[rt['surface'][1:]], b['pos'][None]])
        span = (0, len(ip) - 1)
    elif mode == 'out_out':
        P = np.vstack([a['pos'][None], g.V[rt['surface']], b['pos'][None]])
        span = None
    else:
        P = inside_path(a['pos'], b['pos'])
        span = (0, len(P) - 1) if len(P) > 1 else None
    if span is not None:                                # simplified apart, so the inside part keeps its indices
        A, B, C = simplify(P[:span[0] + 1]), P[span[0]:span[1] + 1], simplify(P[span[1]:])
        P = np.vstack([A, B[1:], C[1:]])
        span = (len(A) - 1, len(A) - 1 + len(B) - 1)
    else:
        P = simplify(P)
    outside_m = (drop + surf['length']) * st.slack
    extra = float(dev['extra'])
    total = outside_m + inside_m + loop + extra
    # the old estimate, for comparison: the Manhattan route × the slack + the service loop (+ a camera's extra)
    man = manhattan(dev['pos'], dest['pos']) * rf + loop + (extra if dev['kind'] == 'camera' else 0.0)
    return {'uid': dev['uid'], 'tag': dev['tag'], 'kind': dev['kind'], 'slot': dev['slot'], 'model': dev['model'],
            'to': dest['name'], 'to_uid': dest['uid'], 'entry': rt['entry']['uid'] if rt['entry'] else None,
            'entry_name': rt['entry']['name'] if rt['entry'] else None, 'device_inside': bool(a['inside']),
            'mode': mode, 'fallback': rt['fallback'],
            'parts': {'drop': drop * st.slack, 'outside': surf['length'] * st.slack, 'inside': inside_m,
                      'service_loop': loop, 'extra': extra},
            'length_m': total, 'manhattan_m': man, 'glass_m': surf['glass'], 'moving_m': surf['moving'],
            'free_m': surf['free'], 'drop_m': drop, 'path': np.asarray(P, float), 'inside_span': span}


def _check_data(project, dev, base, st) -> dict:
    """The data (or RF) run of a device with its link check."""
    r = dict(base)
    cab = CABLES[dev['cable']]
    r.update(id=f'{dev["uid"]}:{"rf" if cab.family == "rf" else "data"}', what='rf' if cab.family == 'rf' else 'data',
             cable=cab.id, cable_label=cab.label, connectors=list(cab.ends), limit_m=cab.limit_m,
             limit_basis=cab.basis, share=None, fitted_m=None, option_m=None, options=None, loss_db=None,
             margin_db=None, drop_pct=None, peak_drop_pct=None, awg=None, problems=[])
    L = r['length_m']
    if cab.limit_m:
        r['share'] = L / cab.limit_m
        if L > cab.limit_m + 1e-9:
            r['problems'].append(('serious', 'limit', f'{dev["tag"]}: {L:.1f} m of {cab.label} is longer than the '
                                  f'{cab.limit_m:g} m the link allows ({cab.basis.split(":")[0]}).'))
        elif r['share'] > st.warn_share + 1e-9:
            r['problems'].append(('warning', 'limit', f'{dev["tag"]}: {L:.1f} m of {cab.label} uses '
                                  f'{r["share"] * 100:.0f} % of the {cab.limit_m:g} m the link allows.'))
    if dev['kind'] == 'camera':
        cam, mod = dev['obj'], dev['module']
        opts = sorted(v for v in (getattr(mod, 'cables_m', ()) or ()) if _finite(v) and v > 0)
        r['options'] = opts
        r['fitted_m'] = float(cam.cable_m) if _finite(cam.cable_m) else None
        r['option_m'] = next((o for o in opts if o >= L - 1e-9), None)
        if opts and r['option_m'] is None:
            r['problems'].append(('serious', 'option', f'{dev["tag"]}: {L:.1f} m routed, longer than the longest '
                                  f'cable of the {mod.name} ({opts[-1]:g} m): move the compute box closer or use a '
                                  f'longer cable if the link allows it.'))
        elif r['fitted_m'] is not None and L > r['fitted_m'] + 1e-6:
            alt = f'use the {r["option_m"]:g} m cable' if r['option_m'] else 'move the compute box closer'
            r['problems'].append(('serious', 'cable', f'{dev["tag"]}: {L:.2f} m routed, longer than its '
                                  f'{r["fitted_m"]:g} m cable: {alt}.'))
    if cab.family == 'rf':
        n_conn = 2
        loss = L * cab.db_per_m + n_conn * st.connector_db
        r['loss_db'] = loss
        r['margin_db'] = st.gnss_lna_db - loss - st.gnss_min_db
        if r['margin_db'] < 0:
            r['problems'].append(('serious', 'rf', f'{dev["tag"]}: {L:.1f} m of {cab.label} loses {loss:.1f} dB at L1; '
                                  f'the {st.gnss_lna_db:g} dB LNA leaves {st.gnss_lna_db - loss:.1f} dB, under the '
                                  f'{st.gnss_min_db:g} dB the receiver needs: use a lower-loss cable or an inline '
                                  f'amplifier.'))
        elif r['margin_db'] < RF_WARN_DB:
            r['problems'].append(('warning', 'rf', f'{dev["tag"]}: only {r["margin_db"]:.1f} dB of RF margin over '
                                  f'{L:.1f} m of {cab.label} ({loss:.1f} dB lost).'))
    if cab.id == 'other':
        r['problems'].append(('note', 'unknown', f'{dev["tag"]}: the length of its "{_iface(dev)}" link is not '
                              'checked (no limit known).'))
    # the share of what bounds it: a camera's fitted cable (or its link, whichever binds), the RF budget, the link
    uses = []
    if r['fitted_m']:
        uses.append((L / r['fitted_m'], f'its {r["fitted_m"]:g} m cable'))
    if r['share'] is not None:
        uses.append((r['share'], f'the {cab.limit_m:g} m {cab.label} allows'))
    if r['loss_db'] is not None:
        budget = st.gnss_lna_db - st.gnss_min_db
        uses.append((r['loss_db'] / budget if budget > 0 else math.inf, f'the {budget:g} dB RF budget'))
    r['use'], r['use_of'] = max(uses, key=lambda u: u[0]) if uses else (None, '')
    _route_problems(r)
    return r


def _iface(dev):
    if dev['kind'] == 'camera':
        return getattr(dev['module'], 'interface', '')
    return getattr(dev.get('smodel'), 'interface', '')


def _check_power(project, dev, base, st) -> dict:
    """The power pair of a lidar, radar or IMU, along its data cable from the distribution beside the compute box."""
    from . import power as PW
    r = dict(base)
    cab = CABLES['power']
    pst = PW.PowerSettings.from_dict(getattr(project.analysis, 'power', None))
    m = dev['smodel']
    w = _num(m.power_w)
    peak = pst.peak_of(m.kind, w)
    own = pst.cables.get(dev['uid'], {})
    awg = own.get('awg', pst.cable_awg)
    r.update(id=f'{dev["uid"]}:power', what='power', cable='power', cable_label=f'Power pair · {awg} AWG',
             connectors=list(cab.ends), limit_m=None, limit_basis=cab.basis, share=None, fitted_m=None,
             option_m=None, options=None, loss_db=None, margin_db=None, awg=awg, to='Power distribution',
             to_uid='', problems=[], use=None, use_of='')
    chk = PW.cable_check(pst, w, max(peak, w), r['length_m'], awg)
    r['drop_pct'], r['peak_drop_pct'], r['v_peak'] = chk['drop_pct'], chk['peak_drop_pct'], chk['v_peak']
    if 'm' in own:
        r['problems'].append(('note', 'power_set', f'{dev["tag"]}: the power budget uses the {own["m"]:g} m set '
                              f'there, not the {r["length_m"]:.1f} m routed.'))
    if chk['cable_severity'] in ('serious', 'warning'):
        what = {'collapse': f'cannot deliver {max(peak, w):.0f} W at {pst.bus_v:g} V',
                'drop': f'drops {chk["drop_pct"]:.1f} % (limit {pst.max_drop_pct:g} %)' if chk['drop_pct'] is not None
                else 'drops too much',
                'min_input': f'leaves {chk["v_peak"]:.1f} V at switch-on, under the {pst.min_input_v:g} V it needs'
                if chk['v_peak'] is not None else 'collapses at switch-on',
                'peak_drop': f'drops {chk["peak_drop_pct"]:.1f} % at switch-on'
                if chk['peak_drop_pct'] is not None else 'collapses at switch-on'}[chk['cable_problem']]
        fix = PW.fix_awg(pst, w, max(peak, w), r['length_m'])
        r['problems'].append((chk['cable_severity'], 'power', f'{dev["tag"]}: its power pair ({r["length_m"]:.1f} m of '
                              f'{awg} AWG) {what}' + (f': use {fix} AWG.' if fix else '.')))
    r['severity'] = _sev_max(*[p[0] for p in r['problems'] if p[0] != 'note'])
    return r


def _route_problems(r):
    """Problems of the route itself (glass, moving parts, free air, no entry point) and the run's severity."""
    tag = r['tag']
    if r['glass_m'] > 0.02:
        r['problems'].append(('warning', 'glass', f'{tag}: the route crosses {r["glass_m"]:.2f} m of glass (nothing '
                              'else reaches): add an entry point nearer, or move the device.'))
    if r['moving_m'] > 0.02:
        r['problems'].append(('warning', 'moving', f'{tag}: the route crosses a moving part or shut line for '
                              f'{r["moving_m"]:.2f} m: route it through a door seal or a grommet instead.'))
    free = max(r['free_m'], r['drop_m'])
    if free > FREE_NOTE_M:
        r['problems'].append(('warning', 'free', f'{tag}: {free:.2f} m of the cable hangs in free air between the '
                              'device and the body: add its mount as a rig part so the cable is routed along it.'))
    if r['fallback'] == 'noentry':
        r['problems'].append(('serious', 'entry', f'{tag}: no entry point is switched on, so the cable cannot get '
                              'inside: the Manhattan estimate is used.'))
    elif r['fallback'] == 'unreachable':
        r['problems'].append(('warning', 'entry', f'{tag}: no route along the surface reaches it: the Manhattan '
                              'estimate is used.'))
    r['severity'] = _sev_max(*[p[0] for p in r['problems'] if p[0] != 'note'])


def _finding(sev, kind, message, tags=(), uids=(), where=None) -> dict:
    return {'severity': sev, 'kind': kind, 'message': message, 'tags': list(tags), 'uids': list(uids),
            'where': where}


def _run_findings(runs, st, routed) -> list:
    out = []
    seen = set()
    for r in runs:
        for sev, kind, msg in r['problems']:
            if (r['uid'], kind) in seen and kind in ('glass', 'moving', 'free', 'entry'):
                continue                                  # the power pair repeats its data cable's route
            seen.add((r['uid'], kind))
            if not routed:
                msg = msg.rstrip('.') + ' (Manhattan estimate).'
            where = (float(r['path'][0][0]), float(r['path'][0][1]))
            out.append(_finding(sev, kind, msg, [r['tag']], [r['uid']], where))
    limited = [r for r in runs if r['share'] is not None]
    if limited and not any(p[1] == 'limit' for r in limited for p in r['problems']):
        b = max(limited, key=lambda r: r['share'])
        out.append(_finding('good', 'limit', f'Every link is within its limit: the nearest, {b["tag"]}, uses '
                            f'{b["share"] * 100:.0f} % ({b["length_m"]:.1f} of {b["limit_m"]:g} m of '
                            f'{b["cable_label"]}).'))
    rfs = [r for r in runs if r['margin_db'] is not None]
    if rfs and all(r['margin_db'] >= RF_WARN_DB for r in rfs):
        b = min(rfs, key=lambda r: r['margin_db'])
        out.append(_finding('good', 'rf', f'Every GNSS cable keeps its RF margin: the least, {b["tag"]}, '
                            f'{b["margin_db"]:.1f} dB over {b["length_m"]:.1f} m of {b["cable_label"]}.'))
    cams = [r for r in runs if r['kind'] == 'camera']
    if cams and not any(p[1] in ('cable', 'option') for r in cams for p in r['problems']):
        b = max(cams, key=lambda r: r['length_m'] / r['fitted_m'] if r['fitted_m'] else 0.0)
        if b['fitted_m']:
            out.append(_finding('good', 'cable', f'Every camera reaches the compute box on its cable: the longest '
                                f'route, {b["tag"]}, is {b["length_m"]:.2f} m of its {b["fitted_m"]:g} m.'))
    return out


def _totals(runs, entries) -> dict:
    by = {}
    conn = {}
    for r in runs:
        d = by.setdefault(r['cable'], {'label': CABLES[r['cable']].label, 'runs': 0, 'length_m': 0.0, 'longest_m': 0.0})
        d['runs'] += 1
        d['length_m'] += r['length_m']
        d['longest_m'] = max(d['longest_m'], r['length_m'])
        for c in r['connectors']:
            if c != '—':
                conn[c] = conn.get(c, 0) + 1
    data = [r for r in runs if r['what'] != 'power']
    longest = max(data, key=lambda r: r['length_m'], default=None)
    limited = [r for r in runs if r['share'] is not None]
    worst = max(limited, key=lambda r: r['share'], default=None)
    used = [r for r in runs if r.get('use') is not None]
    tight = max(used, key=lambda r: r['use'], default=None)
    return {'runs': len(runs), 'length_m': sum(r['length_m'] for r in runs),
            'tightest': None if tight is None else {'id': tight['id'], 'tag': tight['tag'], 'use': tight['use'],
                                                    'of': tight['use_of'], 'severity': tight['severity']},
            'data_length_m': sum(r['length_m'] for r in data),
            'manhattan_m': sum(r['manhattan_m'] for r in data), 'by_cable': by, 'connectors': conn,
            'longest': None if longest is None else {'id': longest['id'], 'tag': longest['tag'],
                                                     'length_m': longest['length_m'], 'cable': longest['cable_label']},
            'worst_share': None if worst is None else worst['share'],
            'worst_tag': None if worst is None else worst['tag'],
            'worst_id': None if worst is None else worst['id'],
            'entries_used': [e['name'] for e in entries if e['cables']]}


# =========================================================================== cached use by the rest of the app
def _inputs_key(project, var, st, routed) -> str:
    g = graph_key(project, st) if routed else None
    devs = [[c.uid, c.tag, c.x, c.y, c.z, c.visible, c.module_id, c.cable_m, c.route_extra_m] for c in var.cameras]
    devs += [var.name, project.vehicle.wheel_centres]
    devs += [[s.uid, s.tag, s.kind, s.model_id, s.x, s.y, s.z, s.visible] for s in var.sensors]
    lib = {'modules': {c.module_id: [getattr(project.module_for(c), 'interface', ''), project.module_for(c).name,
                                     list(getattr(project.module_for(c), 'cables_m', ()) or ())]
                       for c in var.cameras},
           'sensors': {s.model_id: [project.sensor_model_for(s).interface, project.sensor_model_for(s).kind,
                                    project.sensor_model_for(s).power_w, project.sensor_model_for(s).antennas]
                       for s in var.sensors}}
    return json.dumps([str(g), var.uid, devs, lib, project.ecu.to_dict(), st.to_dict(), routed,
                       getattr(project.analysis, 'power', None)], sort_keys=True, default=str)


def cached(project, variant=None, settings: HarnessSettings | None = None, routed: bool = True) -> HarnessResult:
    """run() kept for the same inputs (the Issues list, rules.route_length and the power budget ask often)."""
    st = settings or settings_of(project)
    var = _variant(project, variant)
    routed = routed and project.vehicle.mesh is not None
    key = _inputs_key(project, var, st, routed)
    with _lock:
        hit = _ROUTES.get(key)
        if hit is not None:
            _ROUTES.move_to_end(key)
            return hit
    res = run(project, settings=st, variant=var, routed=routed)
    with _lock:
        _ROUTES[key] = res
        while len(_ROUTES) > 8:
            _ROUTES.popitem(last=False)
    return res


_ONE: OrderedDict = OrderedDict()


def _entries_snapped(project, st, g) -> list:
    """The entry points snapped onto the routing surface (kept per surface and entry list)."""
    key = ('snapped', json.dumps(st.entries, sort_keys=True, default=str),
           json.dumps(project.vehicle.wheel_centres, sort_keys=True, default=str))
    hit = g.cache.get(key)
    if hit is None:
        hit = []
        for x in entries_of(project, st, g):
            v, d = g.nearest([x['x'], x['y'], x['z']], clean=True)
            hit.append(dict(x, v=v, pos=g.V[v].copy(), snap_m=d))
        g.cache[key] = hit
    return hit


def routed_length(project, obj, variant=None) -> float | None:
    """A camera's or sensor's routed cable length (metres, service loop and extra included) when routing is switched
    on (Analysis ▸ Cable harness ▸ Use the routed lengths) and the project has a body; None otherwise, so the caller
    keeps its Manhattan estimate. For rules.route_length and core.power.route_length, which the UI calls while it
    paints: only this device is routed (the surface, its entry points and the Dijkstra trees are kept), and the result
    is kept per device pose and settings, so a repeated call is a dictionary lookup and a moved device costs a few ms.
    The first call for a body builds its routing surface (about 0.3 s for the sample's)."""
    one = _route_one(project, obj, variant)
    return None if one is None else one[0]


def route_text(project, cam) -> str | None:
    """The routed path in words ('F → Bulkhead grommet → Jetson AGX Orin') when routing is on; None otherwise."""
    one = _route_one(project, cam, None)
    if one is None:
        return None
    return ' → '.join([cam.tag] + ([one[1]] if one[1] else []) + [one[2]])


def _route_one(project, obj, variant):
    """(length m, entry point name or None, destination name) of one device's data or RF run, or None (routing off,
    no body, a hidden device or one without a position)."""
    d = getattr(project.analysis, 'harness', None)
    if not (isinstance(d, dict) and _bool(d.get('enabled', False))) or project.vehicle.mesh is None:
        return None
    var = _variant(project, variant) if variant is not None else _variant_of(project, obj)
    st = settings_of(project)
    uid = getattr(obj, 'uid', None)
    devs, _ = _devices(project, var, st)
    dev = next((x for x in devs if x['uid'] == uid), None)
    if dev is None:
        return None
    e = project.ecu
    ins = [x for x in devs if x['kind'] == 'imu' and getattr(x.get('smodel'), 'antennas', 0) >= 1] \
        if dev['kind'] == 'gnss' else []
    key = json.dumps([str(graph_key(project, st)), st.to_dict(), [e.x, e.y, e.z, e.route_factor, e.service_loop_m],
                      dev['pos'].tolist(), dev['kind'], dev['extra'], [x['pos'].tolist() for x in ins],
                      project.vehicle.wheel_centres], sort_keys=True, default=str)
    with _lock:
        hit = _ONE.get(key)
        if hit is not None:
            _ONE.move_to_end(key)
            return hit
    g = route_graph(project, st)
    if g is None:
        return None
    rf, loop = _num(e.route_factor, 1.0) or 1.0, _num(e.service_loop_m)
    ecu_pos = np.array([e.x, e.y, e.z], float)
    if not _finite(*ecu_pos):
        ecu_pos = np.array([1.0, 0.0, 0.35])
    if ins:
        unit = min(ins, key=lambda u: float(np.linalg.norm(u['pos'] - dev['pos'])))
        b, dest = _endpoint(g, unit['pos']), {'name': f'{unit["tag"]} (GNSS/INS)', 'uid': unit['uid'],
                                              'pos': unit['pos']}
    else:
        b = _endpoint(g, ecu_pos, None if st.compute == 'auto' else st.compute == 'inside')
        dest = {'name': e.name + (' · GNSS receiver' if dev['kind'] == 'gnss' else ''), 'uid': '', 'pos': ecu_pos}
    a = _endpoint(g, dev['pos'])
    rt = _route(g, a, b, [dict(x, enabled=x['enabled']) for x in _entries_snapped(project, st, g)], rf,
                st.assign.get(uid))
    r = _assemble(g, dev, a, b, rt, st, rf, loop, dest)
    one = (float(r['length_m']), r['entry_name'], r['to'])
    with _lock:
        _ONE[key] = one
        while len(_ONE) > 512:
            _ONE.popitem(last=False)
    return one


def _variant_of(project, obj):
    """The variant holding obj (the very object first: duplicated variants share uids), else the active one."""
    vs = [project.variant] + [v for v in project.variants if v is not project.variant]
    for v in vs:
        if any(o is obj for o in list(v.cameras) + list(v.sensors)):
            return v
    uid = getattr(obj, 'uid', None)
    return next((v for v in vs if any(o.uid == uid for o in list(v.cameras) + list(v.sensors))), project.variant)


def issues(project, vg=None) -> list:
    """The harness's problems for the Issues list: a run longer than its link allows (or near it), a GNSS cable
    without RF margin, a route over glass or a moving part, a cable hanging in free air, an entry point over its
    capacity, no entry point. Routed when routing is switched on (cached), else from the Manhattan estimate (cheap);
    a camera's cable against its fitted length stays rules' (rules.route_length), a power pair's drop the power
    budget's."""
    from .rules import Issue
    st = settings_of(project)
    res = cached(project, settings=st, routed=st.enabled)
    out = []
    for f in res.findings:
        if f['severity'] not in ('critical', 'serious', 'warning') or f['kind'] not in ISSUE_KINDS:
            continue
        out.append(Issue(f['severity'], f['message'], ', '.join(f['tags']) or f'{res.variant} · harness',
                         list(f['uids']), where=f['where']))
    return out


# =========================================================================== output
def _r(v, d=3):
    return None if v is None or not _finite(float(v)) else round(float(v), d) + 0.0


def summary(res: HarnessResult) -> dict:
    """Compact figures (the assistant, the batch report header)."""
    t = res.totals
    return {'routed': res.routed, 'used_everywhere': res.settings.enabled, 'variant': res.variant,
            'runs': [{'run': r['id'].split(':')[1], 'tag': r['tag'], 'cable': r['cable_label'], 'to': r['to'],
                      'entry': r['entry_name'], 'length_m': _r(r['length_m'], 2),
                      'manhattan_m': _r(r['manhattan_m'], 2), 'limit_m': r['limit_m'],
                      'share_pct': _r(r['share'] * 100, 1) if r['share'] is not None else None,
                      'margin_db': _r(r['margin_db'], 1), 'drop_pct': _r(r['drop_pct'], 1),
                      'status': r['severity']} for r in res.runs],
            'total_m': _r(t['length_m'], 1), 'data_total_m': _r(t['data_length_m'], 1),
            'manhattan_total_m': _r(t['manhattan_m'], 1),
            'by_cable': {v['label']: {'runs': v['runs'], 'metres': _r(v['length_m'], 1)}
                         for v in t['by_cable'].values()},
            'longest': None if t['longest'] is None else {'tag': t['longest']['tag'],
                                                          'length_m': _r(t['longest']['length_m'], 2)},
            'worst_share_pct': _r(t['worst_share'] * 100, 1) if t['worst_share'] is not None else None,
            'tightest': None if t['tightest'] is None else {'tag': t['tightest']['tag'],
                                                            'use_pct': _r(t['tightest']['use'] * 100, 1),
                                                            'of': t['tightest']['of']},
            'entries_used': t['entries_used'],
            'findings': [{'severity': f['severity'], 'message': f['message']} for f in res.findings
                         if f['severity'] != 'good'],
            'assumptions': ASSUMPTIONS}


def _pts(P, d=4):
    return [[round(float(v), d) + 0.0 for v in q] for q in np.asarray(P, float)]


def _run_doc(r) -> dict:
    return {'id': r['id'], 'run': r['what'], 'device': r['tag'], 'kind': r['kind'], 'model': r['model'],
            'from': r['tag'], 'to': r['to'], 'entry': r['entry_name'], 'cable': r['cable'],
            'cable_label': r['cable_label'], 'connectors': list(r['connectors']),
            'length_m': _r(r['length_m']), 'manhattan_m': _r(r['manhattan_m']),
            'parts_m': {k: _r(v) for k, v in r['parts'].items()},
            'limit_m': r['limit_m'], 'limit_basis': r['limit_basis'],
            'share_of_limit_pct': _r(r['share'] * 100, 2) if r['share'] is not None else None,
            'use_pct': _r(r['use'] * 100, 2) if r.get('use') is not None else None, 'use_of': r.get('use_of') or None,
            'fitted_cable_m': r['fitted_m'], 'cable_options_m': r['options'], 'shortest_option_m': r['option_m'],
            'rf_loss_db': _r(r['loss_db'], 2), 'rf_margin_db': _r(r['margin_db'], 2),
            'awg': r['awg'], 'drop_pct': _r(r['drop_pct'], 2), 'peak_drop_pct': _r(r['peak_drop_pct'], 2),
            'on_glass_m': _r(r['glass_m']), 'on_moving_parts_m': _r(r['moving_m']), 'free_air_m': _r(max(r['free_m'],
                                                                                                        r['drop_m'])),
            'device_inside': r['device_inside'], 'status': r['severity'],
            'problems': [m for s, k, m in r['problems']],
            'path': _pts(r['path']), 'inside_path_points': list(r['inside_span']) if r['inside_span'] else None}


def to_doc(res: HarnessResult, project=None) -> dict:
    """The JSON export: the settings, every run with its path, the entry points, the compute box, totals, findings."""
    t, st = res.totals, res.settings
    return {
        'schema': SCHEMA, 'project': getattr(project, 'name', ''), 'variant': res.variant,
        'frame': FRAME, 'units': {'length': 'm', 'loss': 'dB', 'drop': '% of the bus voltage'},
        'routed': res.routed,
        'conventions': {'length': 'drop and surface path × slack + inside path (Manhattan × route_factor) + service '
                                  'loop + extra',
                        'path': 'polyline from the device to the destination; inside_path_points: first and last index '
                                'of the inside part'},
        'settings': {k: v for k, v in st.to_dict().items() if k not in ('entries',)},
        'assumptions': ASSUMPTIONS, 'limits': LIMITS_TEXT,
        'cable_types': {k: {'label': c.label, 'limit_m': c.limit_m, 'basis': c.basis,
                            'rf_db_per_m': c.db_per_m or None, 'connectors': list(c.ends)}
                        for k, c in CABLES.items() if k in t['by_cable']},
        'compute_box': {'name': res.compute['name'], 'location': res.compute['location'],
                        'position': _pts([res.compute['pos']])[0], 'inside': res.compute['inside']},
        'entry_points': [{'uid': e['uid'], 'name': e['name'], 'kind': e['kind'], 'enabled': e['enabled'],
                          'position': _pts([e['pos']])[0], 'asked': _pts([e['asked']])[0], 'snap_m': _r(e['snap_m']),
                          'capacity': e['capacity'] or None, 'cables': len(e['cables'])} for e in res.entries],
        'runs': [_run_doc(r) for r in res.runs],
        'totals': {'runs': t['runs'], 'length_m': _r(t['length_m']), 'data_length_m': _r(t['data_length_m']),
                   'manhattan_estimate_m': _r(t['manhattan_m']),
                   'by_cable': {k: {'label': v['label'], 'runs': v['runs'], 'length_m': _r(v['length_m']),
                                    'longest_m': _r(v['longest_m'])} for k, v in t['by_cable'].items()},
                   'connectors': dict(t['connectors'])},
        'findings': [{'severity': f['severity'], 'kind': f['kind'], 'message': f['message'], 'devices': f['tags']}
                     for f in res.findings],
    }


CSV_COLUMNS = ('run', 'device', 'kind', 'what', 'from', 'to', 'entry', 'cable', 'connectors', 'length_m',
               'manhattan_m', 'limit_m', 'margin_m', 'rf_loss_db', 'rf_margin_db', 'drop_pct', 'status', 'problems',
               'path')


def to_csv(res: HarnessResult) -> str:
    """One row per run (cable): its ends, the cable and connectors, the length against the limit and the path as
    'x y z' points separated by ';' (metres, vehicle frame)."""
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator='\n')
    w.writerow(CSV_COLUMNS)
    blank = lambda v: '' if v is None else v          # noqa: E731
    for r in res.runs:
        margin = None if r['limit_m'] is None else r['limit_m'] - r['length_m']
        w.writerow([r['id'], r['tag'], r['kind'], r['what'], r['tag'], r['to'], blank(r['entry_name']),
                    r['cable_label'], ' / '.join(r['connectors']), _r(r['length_m']), _r(r['manhattan_m']),
                    blank(r['limit_m']), blank(_r(margin)), blank(_r(r['loss_db'], 2)), blank(_r(r['margin_db'], 2)),
                    blank(_r(r['drop_pct'], 2)), r['severity'], ' | '.join(m for s, k, m in r['problems']),
                    '; '.join(' '.join(f'{v:.4f}' for v in q) for q in r['path'])])
    t = res.totals
    w.writerow(['total', '', '', '', '', '', '', f'{t["runs"]} cables', '', _r(t['length_m']), _r(t['manhattan_m']),
                '', '', '', '', '', '', '', ''])
    return buf.getvalue()


# camera colours as the app's dark theme (ui.theme.CAMERA['dark']), sensors as ui.sensorview, for the PNG without Qt
CAM_COLORS = ['#3987e5', '#d95926', '#199e70', '#c98500', '#d55181', '#008300', '#9085e9', '#e66767']
KIND_COLORS = {'lidar': '#0aa5b8', 'radar': '#a0743c', 'gnss': '#7f9a1c', 'imu': '#6e7f96'}


def run_color(r) -> str:
    return CAM_COLORS[r['slot'] % len(CAM_COLORS)] if r['kind'] == 'camera' else KIND_COLORS.get(r['kind'], '#888888')


def _bgr(hex_):
    return int(hex_[5:7], 16), int(hex_[3:5], 16), int(hex_[1:3], 16)


def png(res: HarnessResult, project, width=1400) -> bytes:
    """The harness seen from above and from the left: the body, every data and RF run in its device's colour (the
    inside part dashed), the entry points and the compute box."""
    import cv2
    m = project.vehicle.mesh
    tri = None
    if m is not None and m.n_tris:
        g = route_graph(project, res.settings) if res.routed else None
        src = g.solid if g is not None and g.solid is not None else m
        from .mesh import ROLE_IGNORED
        keep = np.array([p.role != ROLE_IGNORED for p in src.parts] or [True])[src.face_part]
        tri = src.tri[keep]
        gl = (_face_kinds(src)[keep] & GLASS) > 0
    P = [r['path'] for r in res.runs if r['what'] != 'power'] + [np.array([res.compute['pos']])]
    P += [np.array([e['pos']]) for e in res.entries]
    allp = np.vstack([np.asarray(q, float) for q in P if len(q)] + ([tri.reshape(-1, 3)] if tri is not None else []))
    lo, hi = allp.min(axis=0) - 0.3, allp.max(axis=0) + 0.3
    pad, head, gap = 30, 64, 40
    sx = (width - 2 * pad) / max(hi[0] - lo[0], 0.1)
    top_h = int((hi[1] - lo[1]) * sx)
    side_h = int((hi[2] - lo[2]) * sx)
    H = head + top_h + gap + side_h + pad + 30
    img = np.full((H, width, 3), 248, np.uint8)
    ink, muted = (40, 40, 40), (120, 120, 120)

    def to_top(Q):
        Q = np.asarray(Q, float)
        return np.stack([pad + (Q[..., 0] - lo[0]) * sx, head + (hi[1] - Q[..., 1]) * sx], -1)

    def to_side(Q):
        Q = np.asarray(Q, float)
        return np.stack([pad + (Q[..., 0] - lo[0]) * sx, head + top_h + gap + (hi[2] - Q[..., 2]) * sx], -1)

    def text(t, x, y, col=ink, s=0.45, th=1):
        from .pngtext import put_text       # OpenCV's Hershey fonts are ASCII only: · × → would print as ??
        put_text(img, t, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, s, col, th, cv2.LINE_AA)

    text(f'Cable harness · {getattr(project, "name", "")} · {res.variant}', pad, 26, ink, 0.6, 1)
    t = res.totals
    text(f'{t["runs"]} cables, {t["length_m"]:.1f} m in all (data and RF {t["data_length_m"]:.1f} m; the Manhattan '
         f'estimate {t["manhattan_m"]:.1f} m) · ' + ('routed on the body' if res.routed else 'Manhattan estimate'),
         pad, 48, muted, 0.42)
    text('Top view (forward to the right)', pad, head - 6, muted, 0.4)
    text('Left side', pad, head + top_h + gap - 6, muted, 0.4)
    if tri is not None:
        for view, axes in ((to_top, (0, 1)), (to_side, (0, 2))):
            order = np.argsort(tri[:, :, 2].mean(axis=1) if axes == (0, 1) else -tri[:, :, 1].mean(axis=1))
            pts = view(tri[order]).astype(np.int32)
            gls = gl[order]
            for q, isg in zip(pts, gls):
                cv2.fillConvexPoly(img, q, (214, 206, 196) if isg else (222, 222, 222), cv2.LINE_AA)
    for r in res.runs:
        if r['what'] == 'power':
            continue
        col = _bgr(run_color(r))
        span = r['inside_span']
        for view in (to_top, to_side):
            Q = view(r['path'])
            for i in range(len(Q) - 1):
                inside = span is not None and span[0] <= i < span[1]
                a, b = Q[i], Q[i + 1]
                if inside:                               # dashed
                    L = float(np.linalg.norm(b - a))
                    n = max(int(L / 8), 1)
                    for k in range(0, n, 2):
                        p0 = a + (b - a) * k / n
                        p1 = a + (b - a) * min(k + 1, n) / n
                        cv2.line(img, tuple(int(v) for v in p0), tuple(int(v) for v in p1), col, 2, cv2.LINE_AA)
                else:
                    cv2.line(img, tuple(int(v) for v in a), tuple(int(v) for v in b), col, 2, cv2.LINE_AA)
            q0 = Q[0]
            cv2.circle(img, (int(q0[0]), int(q0[1])), 4, col, -1, cv2.LINE_AA)
        q = to_top(r['path'][0])
        text(r['tag'], q[0] + 6, q[1] - 6, col, 0.42, 1)
    for e in res.entries:
        col = (60, 60, 60) if e['enabled'] else (170, 170, 170)
        for view in (to_top, to_side):
            q = view(e['pos'])
            cv2.circle(img, (int(q[0]), int(q[1])), 6, col, 2, cv2.LINE_AA)
        q = to_top(e['pos'])
        text(e['name'], q[0] + 8, q[1] + 14, col, 0.38)
    for view in (to_top, to_side):
        q = view(res.compute['pos'])
        cv2.rectangle(img, (int(q[0]) - 7, int(q[1]) - 7), (int(q[0]) + 7, int(q[1]) + 7), ink, 2, cv2.LINE_AA)
    q = to_top(res.compute['pos'])
    text(res.compute['name'], q[0] + 10, q[1] - 8, ink, 0.4)
    text('Solid: along the body · dashed: inside (Manhattan path) · circles: entry points · square: compute box',
         pad, H - 12, muted, 0.4)
    ok, buf = cv2.imencode('.png', img)
    return buf.tobytes()


def files(res: HarnessResult, project=None) -> list:
    """[(name, bytes)]: harness.json, harness.csv and harness.png."""
    out = [('harness.json', (json.dumps(to_doc(res, project), indent=1, ensure_ascii=False) + '\n').encode('utf-8')),
           ('harness.csv', to_csv(res).encode('utf-8'))]
    if project is not None:
        out.append(('harness.png', png(res, project)))
    return out


def export(res: HarnessResult, project, out_dir) -> list[str]:
    """Write harness.json, harness.csv and harness.png into out_dir; returns their paths."""
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for name, data in files(res, project):
        path = os.path.join(out_dir, name)
        with open(path, 'wb') as fh:
            fh.write(data)
        paths.append(path)
    return paths
