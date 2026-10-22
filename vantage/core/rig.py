"""The data-collection rig on the vehicle: roof racks, sensor masts, crossbars and enclosures.

Rig parts are simple solids in the vehicle frame (core.project.RigPart). They are triangulated here and merged into the
vehicle's geometry by core.analysis.prepare_vehicle, so they block cameras exactly like the body (cube depth maps, the
ground analysis, blind volume, camera images, synthetic captures) and cameras can be placed on the mountable ones.
"""
from __future__ import annotations

import math

import numpy as np

from .mesh import Mesh, Part, ROLE_GLASS, ROLE_IGNORED, ROLE_MOUNTABLE
from .project import RigPart

COLOR = (0.40, 0.42, 0.45)          # anodised aluminium
SEGMENTS = 20


def role(p: RigPart) -> str:
    """Occludes and mountable: a surface like the body; occludes only: like glass; neither: ignored."""
    if not p.occludes:
        return ROLE_IGNORED
    return ROLE_MOUNTABLE if p.mountable else ROLE_GLASS


def _rot(yaw):
    c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _prism(ring, h):
    """A closed prism over the closed polygon ring (n, 2), from z = -h/2 to h/2: vertices and outward faces."""
    n = len(ring)
    V = np.vstack([np.c_[ring, np.full(n, -h / 2)], np.c_[ring, np.full(n, h / 2)],
                   [[0.0, 0.0, -h / 2], [0.0, 0.0, h / 2]]])
    F = []
    for i in range(n):
        j = (i + 1) % n
        F += [[i, j, n + j], [i, n + j, n + i], [2 * n, j, i], [2 * n + 1, n + i, n + j]]
    return V, np.array(F)


def part_geometry(p: RigPart) -> tuple[np.ndarray, np.ndarray]:
    """(vertices (V, 3), faces (F, 3)) of one part in the vehicle frame, faces wound outward."""
    if p.kind == 'box':
        a, b = max(p.length, 1e-3) / 2, max(p.width, 1e-3) / 2
        V, F = _prism(np.array([[-a, -b], [a, -b], [a, b], [-a, b]]), max(p.height, 1e-3))
    else:
        t = np.linspace(0, 2 * math.pi, SEGMENTS, endpoint=False)
        r = max(p.radius, 1e-3)
        V, F = _prism(np.c_[np.cos(t), np.sin(t)] * r, max(p.height if p.kind == 'cylinder' else p.length, 1e-3))
        if p.kind == 'bar':                          # the axis along y
            V = V[:, [0, 2, 1]]
    V = V @ _rot(p.yaw).T + np.array([p.x, p.y, p.z])
    # wind every face outward (the solids are convex: the face normal points away from the centre)
    tri = V[F]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    out = np.einsum('ij,ij->i', n, tri.mean(axis=1) - np.array([p.x, p.y, p.z])) < 0
    F[out] = F[out][:, ::-1]
    return V, F


def rig_mesh(parts) -> Mesh | None:
    """The visible parts as one mesh, one mesh part per rig part (with its role), or None."""
    parts = [p for p in parts if p.visible]
    if not parts:
        return None
    Vs, Fs, FP, out = [], [], [], []
    n = 0
    for i, p in enumerate(parts):
        V, F = part_geometry(p)
        Vs.append(V)
        Fs.append(F + n)
        FP.append(np.full(len(F), i, np.int32))
        n += len(V)
        out.append(Part(p.name, 'rig', 1.0, COLOR, role(p)))
    return Mesh(np.vstack(Vs), np.vstack(Fs), np.concatenate(FP), out)


def merge(a: Mesh, b: Mesh | None) -> Mesh:
    """a with b's triangles appended as more parts (b's faces in no mount zone)."""
    if b is None or b.n_tris == 0:
        return a
    return Mesh(np.vstack([a.vertices, b.vertices]), np.vstack([a.faces, b.faces + len(a.vertices)]),
                np.concatenate([a.face_part, b.face_part + len(a.parts)]), list(a.parts) + list(b.parts),
                np.concatenate([a.face_zone, np.full(b.n_tris, -1, np.int16)]), a.zones)


def rig_key(parts) -> tuple:
    """What the geometry depends on, for the analysis caches."""
    return tuple((p.uid, p.kind, round(p.x, 5), round(p.y, 5), round(p.z, 5), round(p.length, 5), round(p.width, 5),
                  round(p.height, 5), round(p.radius, 5), round(p.yaw, 4), p.occludes, p.mountable)
                 for p in parts if p.visible)


def footprint(p: RigPart) -> np.ndarray:
    """The part's outline seen from above (closed polygon (n, 2))."""
    if p.kind == 'cylinder':
        t = np.linspace(0, 2 * math.pi, 24, endpoint=False)
        return np.c_[p.x + p.radius * np.cos(t), p.y + p.radius * np.sin(t)]
    a, b = (p.length / 2, p.width / 2) if p.kind == 'box' else (p.radius, p.length / 2)
    R = _rot(p.yaw)[:2, :2]
    return np.array([[-a, -b], [a, -b], [a, b], [-a, b]]) @ R.T + np.array([p.x, p.y])


def contains(parts, P) -> int:
    """Index of the first visible part whose solid contains the point P, else -1."""
    for i, p in enumerate(parts):
        if not p.visible:
            continue
        d = np.asarray(P, float) - np.array([p.x, p.y, p.z])
        u = _rot(-p.yaw) @ d
        if p.kind == 'box':
            if abs(u[0]) <= p.length / 2 and abs(u[1]) <= p.width / 2 and abs(u[2]) <= p.height / 2:
                return i
        elif p.kind == 'cylinder':
            if math.hypot(u[0], u[1]) <= p.radius and abs(u[2]) <= p.height / 2:
                return i
        elif math.hypot(u[0], u[2]) <= p.radius and abs(u[1]) <= p.length / 2:
            return i
    return -1


# ------------------------------------------------------------------ presets sized from the vehicle
def roof(mesh: Mesh | None):
    """(x_front, x_rear, half_width, top_z) of the roof: the upward faces near the top of the body."""
    if mesh is None or mesh.n_tris == 0:
        return 1.9, 0.6, 0.6, 1.5
    lo, hi = mesh.bbox
    c, n = mesh.centroids, mesh.normals
    up = (c[:, 2] > hi[2] - 0.08) & (n[:, 2] > 0.7)
    if up.sum() < 3:
        return float(lo[0] + 0.6 * (hi[0] - lo[0])), float(lo[0] + 0.35 * (hi[0] - lo[0])), float(hi[1] * 0.7), float(hi[2])
    xs, ys = c[up, 0], np.abs(c[up, 1])
    return float(xs.max()), float(xs.min()), float(max(ys.max(), 0.3)), float(hi[2])


def preset(kind: str, mesh: Mesh | None) -> list[RigPart]:
    """'roof_rack', 'sensor_mast', 'enclosure' or 'bar', sized and placed on the vehicle's roof."""
    xf, xr, hw, top = roof(mesh)
    xc, span = (xf + xr) / 2, max(xf - xr, 0.3)
    if kind == 'roof_rack':
        bz = top + 0.07
        out = [RigPart(name=f'Roof rack · bar {i + 1}', kind='bar', x=round(x, 3), y=0.0, z=round(bz, 3),
                       length=round(2 * hw + 0.12, 3), radius=0.016) for i, x in enumerate((xc + span * 0.32,
                                                                                            xc - span * 0.32))]
        out.append(RigPart(name='Roof rack · platform', kind='box', x=round(xc, 3), y=0.0, z=round(bz + 0.031, 3),
                           length=round(span * 0.64 + 0.12, 3), width=round(2 * hw * 0.9, 3), height=0.03))
        return out
    if kind == 'sensor_mast':
        h = 0.5
        return [RigPart(name='Sensor mast', kind='cylinder', x=round(xc, 3), y=0.0, z=round(top + h / 2, 3),
                        height=h, radius=0.03),
                RigPart(name='Sensor mast · head', kind='box', x=round(xc, 3), y=0.0, z=round(top + h + 0.06, 3),
                        length=0.18, width=0.18, height=0.12)]
    if kind == 'enclosure':
        return [RigPart(name='Sensor enclosure', kind='box', x=round(xc, 3), y=0.0, z=round(top + 0.1, 3),
                        length=0.6, width=round(min(0.5, 2 * hw * 0.8), 3), height=0.18)]
    if kind == 'bar':
        return [RigPart(name='Crossbar', kind='bar', x=round(xc, 3), y=0.0, z=round(top + 0.07, 3),
                        length=round(2 * hw + 0.12, 3), radius=0.016)]
    raise ValueError(f'unknown rig preset {kind!r}')


PRESETS = [('roof_rack', 'Roof rack'), ('sensor_mast', 'Sensor mast'), ('enclosure', 'Sensor enclosure'),
           ('bar', 'Crossbar')]
