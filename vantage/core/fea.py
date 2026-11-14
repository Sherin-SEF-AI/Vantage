"""Tetrahedral modal analysis of the roof frame (core.framing): quadratic tetrahedra, the sensors as rigid masses on
their plates, the feet fixed or on springs, the lowest modes by shift-invert Lanczos, and the road's ISO 8608 PSD
taken through the modes to each sensor's angular jitter, beside core.thermo_vib's one-mode beam estimate.

Frame and units. The vehicle frame (ISO 8855: x forward, y left, z up, origin at the rear-axle centre on the ground),
SI units inside (m, kg, N, Pa, Hz); reports in mm, degrees and Hz.

Mesh (written here, no mesher needed)
- Every part of the frame is a structured block meshed as hexahedra split into six tetrahedra each about the main
  diagonal (Kuhn), which conforms across neighbouring blocks' faces, then made quadratic (10-node tetrahedra, T10:
  a node on every edge's middle). Straight-sided T10: the stiffness integrand is quadratic, integrated exactly by the
  4-point rule; the consistent mass matrix is integrated exactly from the barycentric monomial formula
  ∫ L1^a L2^b L3^c L4^d dV = 6V·a!b!c!d!/(a+b+c+d+3)!.
- A T-slot member is an equivalent rectangular tube: outer sides and wall solved so that its area and both second
  moments equal the catalogue's (core.framing.PROFILES), its density the catalogue's mass over that area, its modulus
  the maker's. The slots and inner cavities are not meshed: axial, bending and mass match the catalogue; torsion is
  the equivalent tube's own (the catalogues give no torsion constant). Plates are solid blocks, two elements thick.
- Joints are tied (Abaqus' *TIE): every node of the resting or butting part within the gap of the other part's surface
  is held to the point of the other part's surface nearest to it, through that point's T10 shape functions, by a
  penalty (PENALTY × the stiffness matrix's mean diagonal). Bolted joints with corner brackets are softer than a tie:
  the frequencies are an upper bound on the joints' account.
- A sensor (or a payload) is a rigid body: a 6-DOF node at its housing's centre carrying its mass and the inertia of
  its housing as a uniform solid, tied rigidly (penalty) to the plate's top nodes under its footprint.
- Feet: 'fixed' holds every node of a crossbar's underside over a foot's pad; 'spring' puts the set roof stiffness
  (N/mm per foot, vertical and lateral) spread over those nodes. The default roof stiffness is an assumption, not a
  measurement: replace it with a measured value.

Solve: K φ = ω² M φ for the lowest N modes with scipy.sparse.linalg.eigsh in shift-invert mode about a small negative
shift (so a free structure's rigid-body modes are found, not a singular factorisation). Modes are mass-normalised.

Output per mode: frequency, effective modal mass in x, y and z (Γ² with Γ = φᵀ M r, r the rigid translation; as a
share of the moving mass), the part that moves most and how, every part's largest motion per axis, and every
sensor's translation (mm) and rotation (deg), with the mode scaled to 1 mm at its largest structural displacement.

Road jitter: the frame's feet move with the body as core.thermo_vib's half-car moves it on the ISO 8608 road at the
set speed (heave and pitch at the feet's centre: a rigid base motion with influence vectors for x, z and pitch). Each
mode, damped by ζ, responds to −Γ·a_base; the sensors' rotation relative to the body (flex) and in total are
integrated over the road PSD as core.thermo_vib does, and set beside its one-mode beam estimate of the same devices
(which takes the rig parts as drawn: a round mast tube, a crossbar, a typical bracket).

Out of scope: bolt slip and the brackets' compliance, contact (a tie never opens), the roof's own flexibility beyond
the foot springs, prestress, large displacements, damping other than modal ζ, the sensors' own internal modes.
"""
from __future__ import annotations

import csv
import io
import json
import math
import os
import time
from dataclasses import dataclass, field

import numpy as np

SCHEMA = 'vantage.fea/1'
FRAME = 'vehicle (ISO 8855, rear axle): x forward, y left, z up'
PENALTY = 1e3                   # tie stiffness / mean diagonal of K
SHIFT_HZ = 1.0                  # the shift-invert shift: −(2π·SHIFT_HZ)² (a smaller one loses free modes)
GAUSS4 = (0.5854101966249685, 0.1381966011250105)      # degree-2 rule, weights 1/4 (exact for a T10's stiffness)
LIMITS = {'mesh_mm': (5.0, 200.0), 'modes': (1.0, 40.0), 'k_foot_z': (1.0, 1e6), 'k_foot_xy': (1.0, 1e6),
          'zeta': (0.001, 0.3), 'plate_cells_mm': (5.0, 100.0)}
CHOICES = {'bc': ('fixed', 'spring')}
ASSUMPTIONS = ('Quadratic tetrahedra on structured blocks; members as equivalent tubes with the catalogue\'s area and '
               'second moments; plates solid; joints tied (rigid); sensors rigid bodies on their plates; feet fixed '
               'or on the set roof stiffness (an assumption); linear, small displacements; modal damping ζ.')


def _num(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


@dataclass
class FEASettings:
    mesh_mm: float = 25.0           # element size along members and across plates
    modes: int = 12
    bc: str = 'fixed'               # fixed | spring
    k_foot_z: float = 300.0         # N/mm per foot, vertical (spring BC: an assumption, replace with a measurement)
    k_foot_xy: float = 150.0        # N/mm per foot, in the roof's plane
    zeta: float = 0.02              # modal damping (bolted aluminium, as core.thermo_vib's mounts)
    plate_cells_mm: float = 20.0

    def to_dict(self):
        return {k: getattr(self, k) for k in self.__dataclass_fields__}

    @staticmethod
    def from_dict(d) -> 'FEASettings':
        s = FEASettings()
        d = d if isinstance(d, dict) else {}
        for k, (lo, hi) in LIMITS.items():
            v = _num(d.get(k))
            if v is not None:
                setattr(s, k, min(max(v, lo), hi))
        s.modes = int(round(s.modes))
        v = d.get('bc')
        if isinstance(v, str) and v.strip() in CHOICES['bc']:
            s.bc = v.strip()
        return s


def settings_of(project) -> FEASettings:
    return FEASettings.from_dict(getattr(project.analysis, 'fea', None))


# =========================================================================== T10 element
_EDGES = ((0, 1), (1, 2), (0, 2), (0, 3), (1, 3), (2, 3))
_KUHN = [(0, 1, 3, 7), (0, 1, 5, 7), (0, 2, 3, 7), (0, 2, 6, 7), (0, 4, 5, 7), (0, 4, 6, 7)]   # hex corner bits x+2y+4z


def _mono_mass() -> np.ndarray:
    """∫ N_a N_b dV / V for the T10 shape functions, exact."""
    def poly(a):
        if a < 4:                                      # L_a (2 L_a − 1)
            e2 = [0, 0, 0, 0]
            e2[a] = 2
            e1 = [0, 0, 0, 0]
            e1[a] = 1
            return {tuple(e2): 2.0, tuple(e1): -1.0}
        i, j = _EDGES[a - 4]
        e = [0, 0, 0, 0]
        e[i] += 1
        e[j] += 1
        return {tuple(e): 4.0}
    P = [poly(a) for a in range(10)]
    f = math.factorial
    M = np.zeros((10, 10))
    for a in range(10):
        for b in range(10):
            s = 0.0
            for ea, ca in P[a].items():
                for eb, cb in P[b].items():
                    e = [x + y for x, y in zip(ea, eb)]
                    s += ca * cb * 6.0 * f(e[0]) * f(e[1]) * f(e[2]) * f(e[3]) / f(sum(e) + 3)
            M[a, b] = s
    return M


M_REF = _mono_mass()


def shape10(L) -> np.ndarray:
    """T10 shape functions (…, 10) at barycentric coordinates L (…, 4)."""
    L = np.asarray(L, float)
    out = [L[..., i] * (2 * L[..., i] - 1) for i in range(4)]
    out += [4 * L[..., i] * L[..., j] for i, j in _EDGES]
    return np.stack(out, axis=-1)


def _grads(X4):
    """(grad L (E, 4, 3), volume (E,)) of straight tetrahedra with corners X4 (E, 4, 3)."""
    J = np.stack([X4[:, 1] - X4[:, 0], X4[:, 2] - X4[:, 0], X4[:, 3] - X4[:, 0]], axis=2)     # columns
    det = np.linalg.det(J)
    Ji = np.linalg.inv(J)                                   # rows: grad of L1, L2, L3
    g = np.concatenate([-Ji.sum(axis=1, keepdims=True), Ji], axis=1)
    return g, det / 6.0


def element_matrices(X4, E, nu, rho):
    """(Ke (n, 30, 30), Me (n, 30, 30)) of straight T10s with corners X4 (n, 4, 3); E, nu, rho scalars or (n,)."""
    g, V = _grads(X4)
    if np.any(V <= 0):
        raise ValueError('inverted tetrahedron in the mesh')
    E = np.broadcast_to(np.asarray(E, float), V.shape)
    nu = np.broadcast_to(np.asarray(nu, float), V.shape)
    rho = np.broadcast_to(np.asarray(rho, float), V.shape)
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    a, b = GAUSS4
    n = len(V)
    K = np.zeros((n, 10, 3, 10, 3))
    for q in range(4):
        Lq = np.full(4, b)
        Lq[q] = a
        G = np.empty((n, 10, 3))
        for i in range(4):
            G[:, i] = (4 * Lq[i] - 1) * g[:, i]
        for k, (i, j) in enumerate(_EDGES):
            G[:, 4 + k] = 4 * (Lq[i] * g[:, j] + Lq[j] * g[:, i])
        w = 0.25 * V
        K += (w * lam)[:, None, None, None, None] * np.einsum('nai,nbj->naibj', G, G)
        GG = np.einsum('nak,nbk->nab', G, G)
        K += (w * mu)[:, None, None, None, None] * (GG[:, :, None, :, None] * np.eye(3)[None, None, :, None, :])
        K += (w * mu)[:, None, None, None, None] * np.einsum('naj,nbi->naibj', G, G)
    Me = (rho * V)[:, None, None, None, None] * (M_REF[None, :, None, :, None] * np.eye(3)[None, None, :, None, :])
    return K.reshape(n, 30, 30), Me.reshape(n, 30, 30)


# =========================================================================== structured blocks
@dataclass
class Block:
    """One part's mesh block in its own frame (origin, R with columns ex, ey, ez): grid lines and the kept cells."""
    id: str
    name: str
    kind: str                         # member | plate | test
    origin: np.ndarray
    R: np.ndarray
    lines: tuple                      # (xl, yl, zl) local coordinates
    keep: np.ndarray                  # (nx, ny, nz) bool, cells meshed
    E: float
    nu: float
    rho: float
    gap: float = 0.0                  # how far the real part's surface is outside this block's (equivalent tube)
    node0: int = 0
    tet0: int = 0
    cell_ids: np.ndarray | None = None    # (nx, ny, nz) → hex index within the block, −1 where not meshed
    node_ids: np.ndarray | None = None    # (nx+1, ny+1, nz+1) → global corner node id, −1 where unused
    nodes: np.ndarray | None = None       # global node ids of every node of this block (corners and mid-edges)

    @property
    def lo(self):
        return np.array([self.lines[0][0], self.lines[1][0], self.lines[2][0]])

    @property
    def hi(self):
        return np.array([self.lines[0][-1], self.lines[1][-1], self.lines[2][-1]])

    def to_local(self, P):
        return (np.asarray(P, float) - self.origin) @ self.R

    def to_world(self, q):
        return self.origin + np.asarray(q, float) @ self.R.T


def _lines(a, b, n):
    return np.linspace(a, b, max(int(n), 1) + 1)


def equivalent_tube(prof) -> tuple[float, float, float]:
    """(outer b, outer d, wall t) m of a rectangular tube with the profile's area and both second moments."""
    from scipy.optimize import least_squares
    A, Id, Ib = prof.area, prof.i_d, prof.i_b

    def props(x):
        bo, do, t = np.exp(x)
        bi, di = bo - 2 * t, do - 2 * t
        return (bo * do - bi * di, (bo * do ** 3 - bi * di ** 3) / 12, (do * bo ** 3 - di * bi ** 3) / 12)

    def res(x):
        a, i1, i2 = props(x)
        bo, do, t = np.exp(x)
        pen = max(0.0, 2.2 * t - min(bo, do)) * 1e3
        return [a / A - 1, i1 / Id - 1, i2 / Ib - 1, pen]
    best = None
    for f in (0.15, 0.08, 0.25):
        x0 = np.log([prof.b * 0.85, prof.d * 0.85, prof.b * f])
        r = least_squares(res, x0, xtol=1e-14, ftol=1e-14)
        if best is None or r.cost < best.cost:
            best = r
    bo, do, t = np.exp(best.x)
    return float(bo), float(do), float(t)


def member_block(m, mesh_mm) -> Block:
    """A framing member as its equivalent tube, the ring of a 4 × 4 cross-section grid without its middle 2 × 2."""
    pr = m.prof
    bo, do, t = equivalent_tube(pr)
    o, ex, ey, ez = m.frame()
    xl = np.array([-bo / 2, -bo / 2 + t, 0.0, bo / 2 - t, bo / 2])
    yl = np.array([-do / 2, -do / 2 + t, 0.0, do / 2 - t, do / 2])
    zl = _lines(0.0, m.length, math.ceil(m.length / (mesh_mm / 1000.0)))
    keep = np.ones((4, 4, len(zl) - 1), bool)
    keep[1:3, 1:3, :] = False
    gap = max(pr.b - bo, pr.d - do) / 2
    return Block(m.id, m.name, 'member', o, np.stack([ex, ey, ez], axis=1), (xl, yl, zl), keep, pr.E, pr.nu,
                 pr.kg_m / (bo * do - (bo - 2 * t) * (do - 2 * t)), gap)


def _merge_lines(base, extra, lo, hi, min_gap=0.0015):
    """base grid lines with the extra lines inside (lo, hi) added, lines closer than min_gap to an added one
    dropped (the added lines win: they are where the part underneath has its nodes)."""
    ex = sorted({round(float(v), 9) for v in extra if lo + min_gap < v < hi - min_gap})
    out = [v for v in base if all(abs(v - e) >= min_gap for e in ex)] + ex
    return np.array(sorted(out))


def plate_block(P, cell_mm, under=()) -> Block:
    """A plate as a solid block two elements thick; under: the member blocks it rests on, whose node lines are added
    to the plate's grid where their sides run along the plate's (so the plate's underside nodes sit over their
    walls: a tie then has no offset in the plane)."""
    from .framing import PLATE_MATERIAL as PM
    c, ex, ey, ez = P.frame()
    h = cell_mm / 1000.0
    xl = _lines(-P.lx / 2, P.lx / 2, math.ceil(P.lx / h))
    yl = _lines(-P.ly / 2, P.ly / 2, math.ceil(P.ly / h))
    xa, ya = [], []
    for b in under:
        o = (b.origin - c)
        for k in (0, 1):
            ax = b.R[:, k]
            for axis, acc in ((ex, xa), (ey, ya)):
                dp = float(ax @ axis)
                if abs(dp) > 0.999:
                    acc += [float(o @ axis) + dp * v for v in b.lines[k]]
    xl = _merge_lines(xl, xa, -P.lx / 2, P.lx / 2)
    yl = _merge_lines(yl, ya, -P.ly / 2, P.ly / 2)
    zl = _lines(0.0, P.t, 2)
    keep = np.ones((len(xl) - 1, len(yl) - 1, 2), bool)
    return Block(P.id, P.name, 'plate', c, np.stack([ex, ey, ez], axis=1), (xl, yl, zl), keep, PM['E'], PM['nu'],
                 PM['rho'], 0.0)


@dataclass
class Body:
    """A rigid body: 6 DOFs (translation, rotation) at its centre of mass."""
    id: str
    name: str
    uid: str
    tag: str
    kind: str
    centre: np.ndarray
    mass: float
    inertia: np.ndarray               # (3, 3) about the centre, vehicle axes
    slaves: np.ndarray | None = None  # node ids
    dof0: int = 0


class Mesh:
    """Blocks → corner nodes, T4s → T10 nodes and elements."""

    def __init__(self, blocks):
        self.blocks = blocks
        coords, tets, owner = [], [], []
        n0 = 0
        for bi, b in enumerate(blocks):
            xl, yl, zl = b.lines
            nx, ny, nz = len(xl) - 1, len(yl) - 1, len(zl) - 1
            used = np.zeros((nx + 1, ny + 1, nz + 1), bool)
            ii, jj, kk = np.nonzero(b.keep)
            for dx in (0, 1):
                for dy in (0, 1):
                    for dz in (0, 1):
                        used[ii + dx, jj + dy, kk + dz] = True
            ids = -np.ones(used.shape, np.int64)
            ids[used] = n0 + np.arange(int(used.sum()))
            ui, uj, uk = np.nonzero(used)
            q = np.c_[xl[ui], yl[uj], zl[uk]]
            coords.append(b.to_world(q))
            b.node_ids = ids
            b.node0 = n0
            n0 += int(used.sum())
            cells = -np.ones(b.keep.shape, np.int64)
            cells[b.keep] = np.arange(len(ii))
            b.cell_ids = cells
            corner = np.stack([ids[ii + (c & 1), jj + ((c >> 1) & 1), kk + ((c >> 2) & 1)] for c in range(8)], axis=1)
            t4 = np.concatenate([corner[:, list(k)][:, None, :] for k in _KUHN], axis=1).reshape(-1, 4)
            b.tet0 = sum(len(t) for t in tets)
            tets.append(t4)
            owner.append(np.full(len(t4), bi, np.int32))
        X = np.vstack(coords) if coords else np.zeros((0, 3))
        T4 = np.vstack(tets) if tets else np.zeros((0, 4), np.int64)
        self.owner = np.concatenate(owner) if owner else np.zeros(0, np.int32)
        # orient every tetrahedron positively
        if len(T4):
            g = X[T4]
            det = np.einsum('ij,ij->i', np.cross(g[:, 1] - g[:, 0], g[:, 2] - g[:, 0]), g[:, 3] - g[:, 0])
            flip = det < 0
            T4[flip] = T4[flip][:, [0, 2, 1, 3]]
        self.n_corner = len(X)
        # mid-edge nodes
        e = np.concatenate([T4[:, [i, j]] for i, j in _EDGES])
        es = np.sort(e, axis=1)
        key = es[:, 0] * max(len(X), 1) + es[:, 1]
        uk, inv = np.unique(key, return_inverse=True)
        a, b_ = uk // max(len(X), 1), uk % max(len(X), 1)
        mid = (X[a] + X[b_]) / 2 if len(uk) else np.zeros((0, 3))
        self.X = np.vstack([X, mid])
        mids = len(X) + inv.reshape(6, -1).T
        self.T = np.hstack([T4, mids]) if len(T4) else np.zeros((0, 10), np.int64)
        for bi, b in enumerate(blocks):
            sel = self.T[self.owner == bi]
            b.nodes = np.unique(sel)

    @property
    def n_nodes(self):
        return len(self.X)

    def surface(self) -> np.ndarray:
        """Boundary triangles (corner nodes) of the T4 mesh, wound outward."""
        T4 = self.T[:, :4]
        faces = np.concatenate([T4[:, [1, 2, 3]], T4[:, [0, 3, 2]], T4[:, [0, 1, 3]], T4[:, [0, 2, 1]]])
        key = np.sort(faces, axis=1)
        _, idx, cnt = np.unique(key, axis=0, return_index=True, return_counts=True)
        return faces[idx[cnt == 1]]

    def locate(self, b: Block, P):
        """(tet index, barycentric L (4,)) of the point P on or in block b (pushed onto its meshed cells)."""
        q = b.to_local(P)
        lo, hi = b.lo, b.hi
        q = np.clip(q, lo, hi)
        idx = [int(np.clip(np.searchsorted(ln, v, side='right') - 1, 0, len(ln) - 2)) for ln, v in zip(b.lines, q)]
        c = b.cell_ids[idx[0], idx[1], idx[2]]
        if c < 0:                                       # inside a tube's hollow: onto the nearest outer face
            d = np.r_[q - lo, hi - q]
            k = int(np.argmin(d))
            q[k % 3] = lo[k % 3] if k < 3 else hi[k % 3]
            idx = [int(np.clip(np.searchsorted(ln, v, side='right') - 1, 0, len(ln) - 2)) for ln, v in zip(b.lines, q)]
            c = b.cell_ids[idx[0], idx[1], idx[2]]
            if c < 0:
                return None
        P = b.to_world(q)
        best = None
        for t in range(6):
            ti = b.tet0 + 6 * int(c) + t
            X4 = self.X[self.T[ti, :4]]
            J = np.stack([X4[1] - X4[0], X4[2] - X4[0], X4[3] - X4[0]], axis=1)
            lam = np.linalg.solve(J, P - X4[0])
            L = np.r_[1 - lam.sum(), lam]
            m = L.min()
            if best is None or m > best[2]:
                best = (ti, L, m)
            if m >= -1e-9:
                break
        ti, L, _ = best
        L = np.clip(L, 0.0, None)
        return ti, L / L.sum()


# =========================================================================== assembly
def assemble(mesh: Mesh, bodies=(), chunk=4000):
    """(K, M) sparse CSR over 3 DOFs per node then 6 per body."""
    from scipy.sparse import coo_matrix
    nd = 3 * mesh.n_nodes + 6 * len(bodies)
    rows, cols, kv, mv = [], [], [], []
    Eb = np.array([b.E for b in mesh.blocks])
    nub = np.array([b.nu for b in mesh.blocks])
    rhob = np.array([b.rho for b in mesh.blocks])
    loc = np.arange(30)
    for s in range(0, len(mesh.T), chunk):
        T = mesh.T[s:s + chunk]
        own = mesh.owner[s:s + chunk]
        Ke, Me = element_matrices(mesh.X[T[:, :4]], Eb[own], nub[own], rhob[own])
        dof = (3 * T[:, :, None] + np.arange(3)).reshape(len(T), 30)
        rows.append(np.repeat(dof[:, loc], 30, axis=1).ravel())
        cols.append(np.tile(dof, (1, 30)).ravel())
        kv.append(Ke.ravel())
        mv.append(Me.ravel())
    r = np.concatenate(rows) if rows else np.zeros(0, np.int64)
    c = np.concatenate(cols) if cols else np.zeros(0, np.int64)
    K = coo_matrix((np.concatenate(kv) if kv else np.zeros(0), (r, c)), shape=(nd, nd)).tocsr()
    M = coo_matrix((np.concatenate(mv) if mv else np.zeros(0), (r, c)), shape=(nd, nd)).tolil()
    for b in bodies:
        d = b.dof0
        for i in range(3):
            M[d + i, d + i] = b.mass
        M[d + 3:d + 6, d + 3:d + 6] = b.inertia
    return K, M.tocsr()


def tie_rows(mesh: Mesh, a: Block, b: Block, tol: float, select=None):
    """Constraint rows (data, rows, cols) holding a's nodes within tol of b's surface to b; select: an extra mask
    function on a's node coordinates. Returns (C pieces, number of tied nodes)."""
    nodes = a.nodes
    P = mesh.X[nodes]
    q = (P - b.origin) @ b.R
    d = np.linalg.norm(q - np.clip(q, b.lo, b.hi), axis=1)
    inside = np.all((q > b.lo + 1e-6) & (q < b.hi - 1e-6), axis=1)
    m = (d <= tol) & ~inside
    if select is not None:
        m &= select(P)
    if m.any():                                   # the contact face only: nodes deeper in a are not tied (tying
        m &= d <= d[m].min() + 1e-4               # several layers to one surface point would lock its rotation)
    data, rr, cc = [], [], []
    n = 0
    for nid in nodes[m]:
        hit = mesh.locate(b, mesh.X[nid])
        if hit is None:
            continue
        ti, L = hit
        N = shape10(L)
        for i in range(3):
            rr += [0] * 11
            cc += [3 * nid + i] + list(3 * mesh.T[ti] + i)
            data += [1.0] + list(-N)
        n += 1
        data.append(None)                         # row separator
        rr.append(None)
        cc.append(None)
    return _pack(data, rr, cc), n


def _pack(data, rr, cc):
    """Turn the per-node row lists (three rows of 11 entries each, then a None) into (data, row, col) arrays."""
    D, R, C = [], [], []
    row = 0
    buf = []
    for d, c in zip(data, cc):
        if d is None:
            for k in range(3):
                for dd, ccc in buf[11 * k:11 * (k + 1)]:
                    D.append(dd)
                    R.append(row)
                    C.append(ccc)
                row += 1
            buf = []
        else:
            buf.append((d, c))
    return np.array(D), np.array(R, np.int64), np.array(C, np.int64), row


def rigid_rows(mesh: Mesh, body: Body):
    """Rows holding every slave node to the body: u_s − u_c + [r]× θ = 0."""
    D, R, C = [], [], []
    row = 0
    for nid in body.slaves:
        r = mesh.X[nid] - body.centre
        rx = np.array([[0, -r[2], r[1]], [r[2], 0, -r[0]], [-r[1], r[0], 0]])
        for i in range(3):
            D += [1.0, -1.0] + list(rx[i])
            C += [3 * nid + i, body.dof0 + i] + [body.dof0 + 3 + j for j in range(3)]
            R += [row] * 5
            row += 1
    return np.array(D), np.array(R, np.int64), np.array(C, np.int64), row


def penalty_matrix(pieces, nd, alpha):
    from scipy.sparse import coo_matrix
    D, R, C = [], [], []
    off = 0
    for d, r, c, n in pieces:
        if n == 0:
            continue
        D.append(d)
        R.append(r + off)
        C.append(c)
        off += n
    if not D:
        from scipy.sparse import csr_matrix
        return csr_matrix((nd, nd)), 0
    Cm = coo_matrix((np.concatenate(D), (np.concatenate(R), np.concatenate(C))), shape=(off, nd)).tocsr()
    return (alpha * (Cm.T @ Cm)).tocsr(), off


def eigen(K, M, n, free=None, sigma=None):
    """(f Hz (n,), mass-normalised modes (nd, n)) of the lowest n modes; free: the DOFs kept (the others fixed)."""
    from scipy.sparse.linalg import eigsh
    nd = K.shape[0]
    idx = np.arange(nd) if free is None else np.asarray(free)
    Kf, Mf = K[idx][:, idx], M[idx][:, idx]
    n = int(min(n, len(idx) - 2))
    sig = -(2 * math.pi * SHIFT_HZ) ** 2 if sigma is None else sigma
    w2, V = eigsh(Kf, k=n, M=Mf, sigma=sig, which='LM')
    o = np.argsort(w2)
    w2, V = w2[o], V[:, o]
    V /= np.sqrt(np.einsum('ij,ij->j', V, Mf @ V))[None, :]
    out = np.zeros((nd, n))
    out[idx] = V
    return np.sqrt(np.maximum(w2, 0.0)) / (2 * math.pi), out


# =========================================================================== the frame model
@dataclass
class FEModel:
    mesh: Mesh
    bodies: list
    K: object
    M: object
    free: np.ndarray
    feet: list                        # [{'id', 'member', 'nodes'}]
    ties: list                        # [{'a', 'b', 'kind', 'nodes'}]
    notes: list
    frame: object = None
    springs: object = None

    @property
    def n_dof(self):
        return self.K.shape[0]


def _box_inertia(m, half, R, rnd=False):
    a, b, c = 2 * np.asarray(half, float)
    if rnd:
        r = half[0]
        I = np.diag([m * (3 * r * r + c * c) / 12, m * (3 * r * r + c * c) / 12, m * r * r / 2])
    else:
        I = np.diag([m * (b * b + c * c) / 12, m * (a * a + c * c) / 12, m * (a * a + b * b) / 12])
    return R @ I @ R.T


def build_model(project, frame=None, settings: FEASettings | None = None, variant=None, progress=None,
                cancel=None) -> FEModel | None:
    """The frame's FE model: blocks, ties, rigid sensors, feet."""
    from . import framing as FR
    st = settings or settings_of(project)
    prog = progress or (lambda f: None)
    stop = cancel or (lambda: False)
    fr = frame if frame is not None else FR.synthesise(project, variant=variant)
    v = variant or project.variant
    notes = []
    blocks = [member_block(m, st.mesh_mm) for m in fr.members]
    mb = {b.id: b for b in blocks}
    blocks += [plate_block(P, st.plate_cells_mm, [mb[i] for i in P.on if i in mb]) for P in fr.plates]
    by = {b.id: b for b in blocks}
    mesh = Mesh(blocks)
    prog(0.1)
    if stop():
        return None
    # rigid bodies: devices and payloads on plates
    from . import thermo_vib as TV
    rf = TV.roof_box(project)
    devs = {d['uid']: d for d in FR._devices(project, v, rf, fr.settings)}
    from .suspension import rig_part_mass
    parts = {p.uid: p for p in getattr(project, 'rig', []) or []}
    bodies = []
    nd0 = 3 * mesh.n_nodes
    for mt in fr.mounts:
        if mt['status'] not in ('plate', 'top plate') or not mt['plate']:
            continue
        P = next(p for p in fr.plates if p.id == mt['plate'])
        d = devs.get(mt['uid'])
        if d is not None:
            centre, half, R, rnd, mass = d['centre'], d['half'], d['R'], d['round'], d['mass']
        elif mt['uid'] in parts:
            p = parts[mt['uid']]
            centre = np.array([p.x, p.y, p.z])
            half = np.array([p.length, p.width, p.height]) / 2
            R, rnd, mass = np.eye(3), False, rig_part_mass(p)
        else:
            continue
        b = by[P.id]
        q = (mesh.X[b.nodes] - b.origin) @ b.R
        top = np.abs(q[:, 2] - P.t) < 1e-9
        loc = (centre - b.origin) @ b.R
        rr = max(float(np.max(half[:2])), 0.01)
        under = top & (np.abs(q[:, 0] - loc[0]) <= rr + 1e-3) & (np.abs(q[:, 1] - loc[1]) <= rr + 1e-3)
        if under.sum() < 3:
            under = top
        bodies.append(Body(f'B{len(bodies) + 1}', mt['tag'], mt['uid'], mt['tag'], mt['kind'], np.asarray(centre),
                           float(mass), _box_inertia(mass, half, R, rnd), b.nodes[under], nd0 + 6 * len(bodies)))
    K, M = assemble(mesh, bodies)
    prog(0.45)
    if stop():
        return None
    nd = K.shape[0]
    alpha = PENALTY * float(K.diagonal()[:nd0].mean()) if nd0 else 1.0
    pieces, ties = [], []
    for j in fr.joints:
        if j['kind'] == 'foot':
            continue
        a, b = by.get(j['a']), by.get(j['b'])
        if a is None or b is None:
            continue
        tol = a.gap + b.gap + 0.003
        pc, n = tie_rows(mesh, a, b, tol)
        if n == 0:
            notes.append(f'No nodes of {a.name} near {b.name}: the joint is not tied.')
            continue
        pieces.append(pc)
        ties.append({'a': a.id, 'b': b.id, 'kind': j['kind'], 'nodes': n})
    for bd in bodies:
        pieces.append(rigid_rows(mesh, bd))
    Kp, _ = penalty_matrix(pieces, nd, alpha)
    K = (K + Kp).tocsr()
    prog(0.55)
    # feet
    feet, fixed = [], []
    from scipy.sparse import diags
    spring = np.zeros(nd)
    for F in fr.feet:
        b = by.get(F.member)
        if b is None:
            continue
        m = fr.member(F.member)
        q = (mesh.X[b.nodes] - b.origin) @ b.R
        s_f, _, _ = m.local(F.centre + np.array([0, 0, F.height]))
        pad = (np.abs(q[:, 1] - b.lo[1]) < 1e-9) & (np.abs(q[:, 2] - s_f) <= F.lx / 2 + 1e-9)
        ids = b.nodes[pad]
        feet.append({'id': F.id, 'member': F.member, 'nodes': int(len(ids))})
        if st.bc == 'fixed':
            fixed += [3 * i + k for i in ids for k in range(3)]
        elif len(ids):
            for i in ids:
                spring[3 * i] += st.k_foot_xy * 1000.0 / len(ids)
                spring[3 * i + 1] += st.k_foot_xy * 1000.0 / len(ids)
                spring[3 * i + 2] += st.k_foot_z * 1000.0 / len(ids)
    if st.bc == 'spring':
        K = (K + diags(spring)).tocsr()
    free = np.setdiff1d(np.arange(nd), np.unique(np.asarray(fixed, np.int64)))
    if not feet:
        notes.append('The frame has no feet: it is solved free (six rigid-body modes at 0 Hz).')
    return FEModel(mesh, bodies, K, M, free, feet, ties, notes, fr, spring if st.bc == 'spring' else None)


# =========================================================================== the study
@dataclass
class FEAResult:
    settings: FEASettings
    variant: str
    variant_uid: str
    stats: dict
    modes: list               # per mode: f_hz, eff_mass_pct {x, y, z}, part, motion, devices {tag: {...}}
    devices: list             # per device on the frame: road jitter (FEA) beside the beam estimate
    beams: list               # closed-form cantilever of each post vs its FEA mode
    road: dict
    shapes: dict              # {'X': corner node coords (n, 3), 'faces': (m, 3), 'modes': (n_modes, n, 3) mm}
    findings: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    wind: dict | None = None
    convergence: list | None = None
    seconds: float = 0.0


def _influence(model: FEModel, P0):
    """(nd, 3) rigid base-motion fields: unit x, unit z translation, unit pitch (rotation about +y through P0)."""
    nd = model.n_dof
    n = model.mesh.n_nodes
    r = np.zeros((nd, 3))
    r[0:3 * n:3, 0] = 1.0
    r[2:3 * n:3, 1] = 1.0
    d = model.mesh.X - P0
    r[0:3 * n:3, 2] = d[:, 2]
    r[2:3 * n:3, 2] = -d[:, 0]
    for b in model.bodies:
        r[b.dof0, 0] = 1.0
        r[b.dof0 + 2, 1] = 1.0
        dc = b.centre - P0
        r[b.dof0, 2] = dc[2]
        r[b.dof0 + 2, 2] = -dc[0]
        r[b.dof0 + 4, 2] = 1.0
    return r


def _mode_motion(model: FEModel, phi, fr):
    """(the part moving most, its main direction word) of one mode."""
    X = model.mesh.X
    n = model.mesh.n_nodes
    u = phi[:3 * n].reshape(-1, 3)
    best, bv = None, -1.0
    for b in model.mesh.blocks:
        a = float(np.abs(u[b.nodes]).max()) if len(b.nodes) else 0.0
        if a > bv:
            best, bv = b, a
    if best is None:
        return '', ''
    uu = u[best.nodes]
    k = int(np.argmax(np.abs(uu).max(axis=0)))
    word = {0: 'fore-aft', 1: 'side to side', 2: 'up and down'}[k]
    if best.kind == 'member':                        # twisting: turning about its own axis more than swaying
        ax = best.R[:, 2]
        c = X[best.nodes].mean(axis=0)
        r = X[best.nodes] - c
        r_perp = r - (r @ ax)[:, None] * ax
        spin = np.cross(r_perp, uu) @ ax / np.maximum((r_perp ** 2).sum(axis=1), 1e-12)
        sway = np.abs(uu - uu.mean(axis=0)).max()
        if np.abs(uu.mean(axis=0)).max() < 0.3 * np.abs(uu).max() and np.abs(spin).mean() * \
                np.sqrt((r_perp ** 2).sum(axis=1)).max() > 0.5 * sway:
            word = 'twisting'
    return best.name, word


def run(project, engine=None, settings: FEASettings | dict | None = None, variant=None, progress=None,
        cancel=None, frame=None, convergence=False, wind=None) -> FEAResult | None:
    """Build the frame's model, solve its modes, take the road through them, compare with the beam estimate.
    wind: an aero result (core.aero) or {device uid / member id: (Fx, Fy, Fz) N} for a static wind case."""
    from . import framing as FR
    from . import thermo_vib as TV
    t0 = time.perf_counter()
    st = settings if isinstance(settings, FEASettings) else (
        FEASettings.from_dict(settings) if isinstance(settings, dict) else settings_of(project))
    prog = progress or (lambda f: None)
    stop = cancel or (lambda: False)
    v = variant or project.variant
    fr = frame if frame is not None else FR.synthesise(project, variant=v)
    if not fr.members:
        return FEAResult(st, v.name, v.uid, {'nodes': 0, 'elements': 0, 'dofs': 0}, [], [], [], {}, {},
                         [{'severity': 'note', 'kind': 'empty', 'message': 'Nothing on the roof to analyse: '
                           'the frame (Analysis ▸ Frame) is empty.'}], [], seconds=time.perf_counter() - t0)
    model = build_model(project, fr, st, v, lambda f: prog(0.5 * f), stop)
    if model is None:
        return None
    if stop():
        return None
    f, phi = eigen(model.K, model.M, st.modes, model.free)
    prog(0.75)
    if stop():
        return None
    n = model.mesh.n_nodes
    M = model.M
    mtot = {}
    feet_c = np.array([F.centre for F in fr.feet]).mean(axis=0) if fr.feet else model.mesh.X.mean(axis=0)
    r = _influence(model, feet_c)
    r_xyz = np.zeros((model.n_dof, 3))
    r_xyz[0:3 * n:3, 0] = 1
    r_xyz[1:3 * n:3, 1] = 1
    r_xyz[2:3 * n:3, 2] = 1
    for b in model.bodies:
        for i in range(3):
            r_xyz[b.dof0 + i, i] = 1
    free_mask = np.zeros(model.n_dof, bool)
    free_mask[model.free] = True
    rf_ = r_xyz * free_mask[:, None]
    for i, k in enumerate('xyz'):
        mtot[k] = float(rf_[:, i] @ (M @ rf_[:, i]))
    Gam = phi.T @ (M @ rf_)                          # (modes, 3)
    modes = []
    shapes = []
    for i in range(len(f)):
        ph = phi[:, i]
        u = ph[:3 * n].reshape(-1, 3)
        peak = float(np.linalg.norm(u, axis=1).max()) or 1.0
        s = 0.001 / peak                              # 1 mm at the largest structural displacement
        part, word = _mode_motion(model, ph, fr)
        devs = {}
        for b in model.bodies:
            t = ph[b.dof0:b.dof0 + 3] * s * 1000.0
            a = np.degrees(ph[b.dof0 + 3:b.dof0 + 6] * s)
            devs[b.tag] = {'move_mm': [float(x) for x in t], 'turn_deg': [float(x) for x in a],
                           'move_abs_mm': float(np.linalg.norm(t)), 'turn_abs_deg': float(np.linalg.norm(a))}
        parts = {}
        for b in model.mesh.blocks:                   # each part's largest motion per axis (mm, signed)
            ub = u[b.nodes] * s * 1000.0
            k_ = np.argmax(np.abs(ub), axis=0)
            parts[b.name] = [float(ub[k_[j], j]) for j in range(3)]
        eff = {k: float(100.0 * Gam[i, j] ** 2 / mtot[k]) if mtot[k] > 0 else 0.0 for j, k in enumerate('xyz')}
        modes.append({'mode': i + 1, 'f_hz': float(f[i]), 'eff_mass_pct': eff,
                      'participation': [float(x) for x in Gam[i]], 'part': part, 'motion': word,
                      'devices': devs, 'parts': parts})
        shapes.append(u[:model.mesh.n_corner] * s * 1000.0)
    cum = {k: float(sum(m['eff_mass_pct'][k] for m in modes)) for k in 'xyz'}
    # ---------------------------------------------------------------- road through the modes
    tv_st = TV.settings_of(project)
    hc = TV.half_car(project, tv_st, v)
    V = tv_st.speed_kmh / 3.6
    fgrid = TV.freq_grid([float(x) for x in f if x > 0])
    w = 2 * math.pi * fgrid
    G = TV.road_psd_temporal(fgrid, tv_st.road_class, V)
    Xa = -(w * w)[:, None] * hc.response(fgrid, V)       # q̈ per unit road height
    aB = TV.base_accel(hc, Xa, feet_c)                  # (F, 3)
    base = np.stack([aB[:, 0], aB[:, 2], Xa[:, 1]], axis=1)    # x, z, pitch
    Gr = phi.T @ (M @ (r * free_mask[:, None]))         # (modes, 3)
    wn = 2 * math.pi * f
    H = 1.0 / (wn[None, :] ** 2 - w[:, None] ** 2 + 2j * st.zeta * wn[None, :] * w[:, None])    # (F, modes)
    eta = -(base @ Gr.T) * H                           # (F, modes)
    tvres = None
    try:
        tvres = TV.run(project, variant=v, events=False)
    except Exception as e:                              # the beam estimate is a comparison, not needed for the FEA
        model.notes.append(f'core.thermo_vib could not run for the comparison: {type(e).__name__}: {e}')
    tvd = {d['uid']: d for d in (tvres.devices if tvres is not None else [])}
    devices = []
    body_ang = np.zeros((len(fgrid), 3), complex)
    body_ang[:, 1] = -Xa[:, 1] / np.where(w > 0, w * w, 1.0)
    for b in model.bodies:
        rot = eta @ phi[b.dof0 + 3:b.dof0 + 6].T        # (F, 3) rad per unit road height
        lin = eta @ phi[b.dof0:b.dof0 + 3].T
        rate = 1j * w[:, None] * (rot + body_ang)
        flex_deg = math.degrees(TV._rms(rot, G, fgrid))
        tot_deg = math.degrees(TV._rms(rot + body_ang, G, fgrid))
        pa = (np.abs(rot) ** 2).sum(axis=1) * G
        rec = {'uid': b.uid, 'tag': b.tag, 'kind': b.kind, 'mass_kg': b.mass, 'flex_deg_rms': flex_deg,
               'flex_mm_rms': TV._rms(lin, G, fgrid) * 1000.0, 'angle_deg_rms': tot_deg,
               'rate_deg_s_rms': math.degrees(TV._rms(rate, G, fgrid)),
               'peak_hz': float(fgrid[int(np.argmax(pa))]) if pa.max() > 0 else None}
        t = tvd.get(b.uid)
        if t is not None:
            rec['beam'] = {'flex_deg_rms': t['vib']['flex_deg_rms'], 'flex_mm_rms': t['vib']['flex_mm_rms'],
                           'angle_deg_rms': t['vib']['angle_deg_rms'], 'f1_hz': t['f1_hz'],
                           'chain': [m['name'] + f' {m["f1_hz"]:.1f} Hz' for m in t['modes']]}
            rec['ratio_flex'] = (flex_deg / t['vib']['flex_deg_rms']) if t['vib']['flex_deg_rms'] > 0 else None
        devices.append(rec)
    prog(0.85)
    # ---------------------------------------------------------------- closed-form check per post
    beams = []
    for m in fr.members:
        if m.role not in ('post', 'standoff'):
            continue
        pr = m.prof
        top_plate = [P for P in fr.plates if m.id in P.on]
        m_tip = sum(P.mass for P in top_plate) + sum(b.mass for b in model.bodies if any(
            b.uid == mt['uid'] for mt in fr.mounts if mt['member'] == m.id))
        L = m.length
        f_b = TV.cantilever_f1(pr.E, pr.i_b, L, m_tip, pr.kg_m * L)
        cand = [md for md in modes if md['part'] in (m.name,) + tuple(P.name for P in top_plate)
                or any(abs(d['move_abs_mm']) > 0.5 for d in md['devices'].values())]
        near = min(cand or modes, key=lambda md: abs(md['f_hz'] - f_b)) if modes else None
        beams.append({'member': m.id, 'name': m.name, 'length_mm': L * 1000, 'tip_kg': m_tip,
                      'beam_f1_hz': f_b, 'fea_hz': near['f_hz'] if near else None,
                      'fea_mode': near['mode'] if near else None,
                      'note': 'cantilever on a rigid base, tip mass + 0.24 of its own (Rayleigh), weak axis'})
    # ---------------------------------------------------------------- wind
    wind_out = None
    if wind is not None:
        wind_out = static_wind(model, wind)
    conv = None
    if convergence:
        conv = []
        for k in (2.0, 1.4, 1.0):
            s2 = FEASettings.from_dict(dict(st.to_dict(), mesh_mm=st.mesh_mm * k, plate_cells_mm=st.plate_cells_mm * k))
            if k == 1.0:
                ff = f
                mm = model
            else:
                mm = build_model(project, fr, s2, v)
                ff, _ = eigen(mm.K, mm.M, min(st.modes, 6), mm.free)
            conv.append({'mesh_mm': s2.mesh_mm, 'dofs': int(len(mm.free)), 'f_hz': [float(x) for x in ff[:6]]})
            if stop():
                return None
    surf = model.mesh.surface()
    res = FEAResult(st, v.name, v.uid,
                    {'nodes': int(n), 'elements': int(len(model.mesh.T)), 'dofs': int(len(model.free)),
                     'bodies': len(model.bodies), 'ties': model.ties, 'feet': model.feet,
                     'moving_mass_kg': mtot, 'cum_eff_mass_pct': cum, 'bc': st.bc,
                     'frame_kg': fr.mass.get('frame_kg')},
                    modes, devices, beams,
                    {'speed_kmh': tv_st.speed_kmh, 'road_class': tv_st.road_class, 'zeta': st.zeta,
                     'feet_centre_m': [float(x) for x in feet_c]},
                    {'X': model.mesh.X[:model.mesh.n_corner], 'faces': surf, 'modes': np.array(shapes),
                     'owner_faces': None},
                    [], list(fr.notes) + model.notes, wind_out, conv)
    res.findings = _findings(res, fr)
    res.seconds = time.perf_counter() - t0
    prog(1.0)
    _CACHE['last'] = (inputs_key(project, v), res)
    return res


def static_wind(model: FEModel, wind) -> dict:
    """Static deflection under steady wind forces: {device tag: {'move_mm', 'turn_deg'}} and the forces used.
    wind: a core.aero result (its element forces at the highway case) or {member id | plate id | device uid:
    (Fx, Fy, Fz) N}."""
    from scipy.sparse.linalg import splu
    forces = wind
    if hasattr(wind, 'loads'):
        forces = wind.loads
    nd = model.n_dof
    F = np.zeros(nd)
    blocks = {b.id: b for b in model.mesh.blocks}
    bodies = {b.uid: b for b in model.bodies}
    applied = 0.0
    for key, vec in (forces or {}).items():
        vec = np.asarray(vec, float)
        if key in bodies:
            F[bodies[key].dof0:bodies[key].dof0 + 3] += vec
        elif key in blocks:
            ids = blocks[key].nodes
            for i in range(3):
                F[3 * ids + i] += vec[i] / len(ids)
        else:
            continue
        applied += float(np.linalg.norm(vec))
    idx = model.free
    lu = splu(model.K[idx][:, idx].tocsc())
    u = np.zeros(nd)
    u[idx] = lu.solve(F[idx])
    out = {}
    for b in model.bodies:
        out[b.tag] = {'move_mm': [float(x * 1000) for x in u[b.dof0:b.dof0 + 3]],
                      'turn_deg': [float(math.degrees(x)) for x in u[b.dof0 + 3:b.dof0 + 6]],
                      'move_abs_mm': float(np.linalg.norm(u[b.dof0:b.dof0 + 3]) * 1000),
                      'turn_abs_deg': float(math.degrees(np.linalg.norm(u[b.dof0 + 3:b.dof0 + 6])))}
    n = model.mesh.n_nodes
    return {'devices': out, 'force_n': applied,
            'peak_mm': float(np.linalg.norm(u[:3 * n].reshape(-1, 3), axis=1).max() * 1000) if n else 0.0}


def _finding(sev, kind, message, uids=()):
    return {'severity': sev, 'kind': kind, 'message': message, 'uids': list(uids)}


def _findings(res: FEAResult, fr) -> list:
    from . import thermo_vib as TV
    out = []
    if not res.modes:
        return out
    f1 = res.modes[0]['f_hz']
    out.append(_finding('good', 'summary', f'First mode {f1:.1f} Hz ({res.modes[0]["part"]}, '
                        f'{res.modes[0]["motion"]}); {len(res.modes)} modes, {res.stats["dofs"]:,} DOFs.'))
    for m in res.modes[:6]:
        for band, name in ((TV.BODY_BAND, 'the body\'s bounce and pitch'), (TV.HOP_BAND, 'wheel hop')):
            lo, hi = band
            if lo * (1 - TV.NEAR_PEAK) <= m['f_hz'] <= hi * (1 + TV.NEAR_PEAK):
                out.append(_finding('serious', 'resonance', f'Mode {m["mode"]} ({m["f_hz"]:.1f} Hz, {m["part"]} '
                                    f'{m["motion"]}) lies in {name} ({lo:g}-{hi:g} Hz): stiffen or brace it.'))
    for d in res.devices:
        b = d.get('beam')
        if b and b['flex_deg_rms'] > 0 and d['flex_deg_rms'] > 2 * b['flex_deg_rms']:
            out.append(_finding('warning', 'beam_low', f'{d["tag"]}: the frame flexes {d["flex_deg_rms"]:.4f}° RMS on '
                                f'the road, {d["flex_deg_rms"] / b["flex_deg_rms"]:.1f}× the beam estimate: '
                                'the one-mode estimate is optimistic for this frame.', [d['uid']]))
    return out


_CACHE: dict = {}


def cached(project, variant=None):
    """The last result for these inputs (core.aero uses it for the lock-in check), or None."""
    hit = _CACHE.get('last')
    if hit and hit[0] == inputs_key(project, variant):
        return hit[1]
    return None


def inputs_key(project, variant=None) -> str:
    from . import framing as FR
    a = project.analysis
    return FR.inputs_key(project, variant) + json.dumps({'fea': a.fea, 'thermo_vib': a.thermo_vib,
                                                        'suspension': a.suspension}, sort_keys=True, default=str)


# =========================================================================== outputs
def _r(v, d=4):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return round(v, d) if math.isfinite(v) else None


def _clean(o, d=5):
    if isinstance(o, dict):
        return {k: _clean(v, d) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v, d) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist(), d)
    if isinstance(o, (float, np.floating)):
        return _r(o, d)
    if isinstance(o, np.integer):
        return int(o)
    return o


def summary(res: FEAResult) -> dict:
    return {'variant': res.variant, 'dofs': res.stats.get('dofs'), 'bc': res.settings.bc,
            'modes_hz': [_r(m['f_hz'], 2) for m in res.modes],
            'first_mode': ({'f_hz': _r(res.modes[0]['f_hz'], 2), 'part': res.modes[0]['part'],
                            'motion': res.modes[0]['motion']} if res.modes else None),
            'jitter': {d['tag']: {'fea_flex_deg_rms': _r(d['flex_deg_rms'], 5),
                                  'beam_flex_deg_rms': _r((d.get('beam') or {}).get('flex_deg_rms'), 5)}
                       for d in res.devices},
            'findings': [f['message'] for f in res.findings if f['severity'] in ('critical', 'serious', 'warning')],
            'seconds': _r(res.seconds, 2)}


def to_doc(res: FEAResult, project=None) -> dict:
    return {'schema': SCHEMA, 'project': getattr(project, 'name', ''), 'variant': res.variant, 'frame': FRAME,
            'units': {'frequency': 'Hz', 'move': 'mm', 'turn': 'deg', 'mass': 'kg',
                      'mode_scale': 'each mode scaled to 1 mm at its largest structural displacement',
                      'jitter': 'RMS over the ISO 8608 road PSD at the set speed'},
            'assumptions': ASSUMPTIONS, 'model': __doc__.split('Out of scope:')[0].strip(),
            'limits': 'Out of scope: ' + __doc__.split('Out of scope:')[1].strip(),
            'settings': res.settings.to_dict(), 'stats': _clean(res.stats), 'modes': _clean(res.modes),
            'devices': _clean(res.devices), 'beams': _clean(res.beams), 'road': _clean(res.road),
            'wind': _clean(res.wind), 'convergence': _clean(res.convergence),
            'findings': [{'severity': f['severity'], 'kind': f['kind'], 'message': f['message']} for f in res.findings],
            'notes': list(res.notes), 'seconds': _r(res.seconds, 3)}


def modes_csv(res: FEAResult) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator='\n')
    w.writerow(['mode', 'f_hz', 'eff_mass_x_pct', 'eff_mass_y_pct', 'eff_mass_z_pct', 'part', 'motion'] +
               [f'{t}_{q}' for t in (res.modes[0]['devices'] if res.modes else {}) for q in ('move_mm', 'turn_deg')])
    for m in res.modes:
        row = [m['mode'], f'{m["f_hz"]:.3f}'] + [f'{m["eff_mass_pct"][k]:.2f}' for k in 'xyz'] + [m['part'],
                                                                                                 m['motion']]
        for t, d in m['devices'].items():
            row += [f'{d["move_abs_mm"]:.4f}', f'{d["turn_abs_deg"]:.5f}']
        w.writerow(row)
    return buf.getvalue()


def jitter_csv(res: FEAResult) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator='\n')
    w.writerow(['device', 'kind', 'fea_flex_deg_rms', 'fea_flex_mm_rms', 'fea_angle_deg_rms', 'beam_flex_deg_rms',
                'beam_flex_mm_rms', 'beam_angle_deg_rms', 'beam_f1_hz'])
    for d in res.devices:
        b = d.get('beam') or {}
        w.writerow([d['tag'], d['kind'], f'{d["flex_deg_rms"]:.6f}', f'{d["flex_mm_rms"]:.5f}',
                    f'{d["angle_deg_rms"]:.6f}', _r(b.get('flex_deg_rms'), 6), _r(b.get('flex_mm_rms'), 5),
                    _r(b.get('angle_deg_rms'), 6), _r(b.get('f1_hz'), 3)])
    return buf.getvalue()


def png(res: FEAResult, mode=0, size=(900, 620), yaw=-130.0, elev=24.0, phase=1.0, scale_mm=None) -> bytes:
    """The deformed frame of one mode (grey: at rest, colour: deformed), as PNG."""
    import cv2
    img = mode_image(res, mode, size, yaw, elev, phase, scale_mm)
    ok, buf = cv2.imencode('.png', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return buf.tobytes() if ok else b''


def mode_image(res: FEAResult, mode=0, size=(900, 620), yaw=-130.0, elev=24.0, phase=1.0, scale_mm=None, zoom=1.0,
               bg=(245, 245, 243), body=(150, 152, 156), hot=(214, 96, 40)) -> np.ndarray:
    """RGB image of mode `mode` at `phase` (−1…1): the undeformed frame see-through and the deformed one shaded."""
    from .meshing import render, view_basis
    X = res.shapes.get('X')
    Fc = res.shapes.get('faces')
    W, H = size
    if X is None or not len(X) or Fc is None or not len(res.modes):
        return np.full((H, W, 3), bg, np.uint8)
    U = res.shapes['modes'][min(mode, len(res.modes) - 1)] / 1000.0
    c = (X.min(axis=0) + X.max(axis=0)) / 2
    rad = float(np.linalg.norm(X - c, axis=1).max()) or 1.0
    amp = (scale_mm / 1000.0) if scale_mm else 0.08 * rad
    Xd = X + U * (amp / 0.001) * phase
    layers = [(X, Fc, body, 0.25), (Xd, Fc, hot, 1.0)]
    return render(layers, view_basis(yaw, elev), size, 0.45 * min(W, H) / rad * zoom, c, bg=bg, ss=1)


def files(res: FEAResult, project=None) -> list:
    out = [('fea.json', (json.dumps(to_doc(res, project), indent=1) + '\n').encode()),
           ('fea_modes.csv', modes_csv(res).encode()),
           ('fea_jitter.csv', jitter_csv(res).encode())]
    for i in range(min(len(res.modes), 3)):
        out.append((f'fea_mode{i + 1}.png', png(res, i)))
    return out


def export(res: FEAResult, project, out_dir) -> list[str]:
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for name, data in files(res, project):
        path = os.path.join(out_dir, name)
        with open(path, 'wb') as fh:
            fh.write(data)
        paths.append(path)
    return paths


# =========================================================================== validation: beams and a plate
def bar_model(L, b, h, E, nu, rho, n_len, n_sec=2, bc='cantilever'):
    """A solid rectangular bar along z (b along x, h along y) meshed with T10s: (K, M, free DOFs)."""
    xl = _lines(-b / 2, b / 2, n_sec)
    yl = _lines(-h / 2, h / 2, n_sec)
    zl = _lines(0.0, L, n_len)
    blk = Block('bar', 'bar', 'test', np.zeros(3), np.eye(3), (xl, yl, zl), np.ones((n_sec, n_sec, n_len), bool),
                E, nu, rho)
    mesh = Mesh([blk])
    K, M = assemble(mesh)
    nd = K.shape[0]
    if bc == 'cantilever':
        fixed = np.nonzero(np.abs(mesh.X[:, 2]) < 1e-12)[0]
        free = np.setdiff1d(np.arange(nd), (3 * fixed[:, None] + np.arange(3)).ravel())
    else:
        free = np.arange(nd)
    return K, M, free


def plate_model(a, h, E, nu, rho, n, layers=2):
    """A square plate a × a × h, clamped along x = 0 (a cantilever plate): (K, M, free DOFs)."""
    xl = _lines(0.0, a, n)
    yl = _lines(0.0, a, n)
    zl = _lines(0.0, h, layers)
    blk = Block('plate', 'plate', 'test', np.zeros(3), np.eye(3), (xl, yl, zl), np.ones((n, n, layers), bool),
                E, nu, rho)
    mesh = Mesh([blk])
    K, M = assemble(mesh)
    fixed = np.nonzero(np.abs(mesh.X[:, 0]) < 1e-12)[0]
    free = np.setdiff1d(np.arange(K.shape[0]), (3 * fixed[:, None] + np.arange(3)).ravel())
    return K, M, free


EB_BETA_CANTILEVER = (1.875104069, 4.694091133, 7.854757438)       # roots of cos βL cosh βL = −1
EB_BETA_FREE = (4.730040745, 7.853204624, 10.99560784)              # roots of cos βL cosh βL = 1
LEISSA_CFFF = (3.4917, 8.5246, 21.429)     # λ = ω a² √(ρh/D), square cantilever plate, ν = 0.3 (Leissa 1969)


def euler_bernoulli(beta, L, E, I, rho, A):
    return beta ** 2 / (2 * math.pi) * math.sqrt(E * I / (rho * A * L ** 4))


def validate_beams(L=0.4, b=0.02, E=70e9, nu=0.33, rho=2700.0, n_len=20, n_sec=2) -> dict:
    """Cantilever and free-free bars against Euler–Bernoulli: {'cantilever': [(fea, closed, err %)...],
    'free': [...]}."""
    I, A = b ** 4 / 12, b * b
    out = {}
    K, M, free = bar_model(L, b, b, E, nu, rho, n_len, n_sec, 'cantilever')
    f, _ = eigen(K, M, 6, free)
    ref = [euler_bernoulli(bt, L, E, I, rho, A) for bt in EB_BETA_CANTILEVER[:2]]
    fea = [f[0], f[2]]                                   # each bending frequency twice (square bar)
    out['cantilever'] = [(float(a), float(r), float(100 * (a / r - 1))) for a, r in zip(fea, ref)]
    K, M, free = bar_model(L, b, b, E, nu, rho, n_len, n_sec, 'free')
    f, _ = eigen(K, M, 12, free)
    el = [x for x in f if x > 1.0]
    ref = [euler_bernoulli(bt, L, E, I, rho, A) for bt in EB_BETA_FREE[:1]]
    out['free'] = [(float(el[0]), float(ref[0]), float(100 * (el[0] / ref[0] - 1)))]
    out['rigid_modes'] = int(sum(x < 1.0 for x in f))
    return out


def validate_plate(a=0.2, h=0.01, E=70e9, nu=0.3, rho=2700.0, n=12, layers=2) -> list:
    """The square cantilever plate's first modes against Leissa's λ (ν = 0.3): [(fea Hz, Leissa Hz, err %)]."""
    D = E * h ** 3 / (12 * (1 - nu ** 2))
    K, M, free = plate_model(a, h, E, nu, rho, n, layers)
    f, _ = eigen(K, M, 3, free)
    ref = [lam / (2 * math.pi * a * a) * math.sqrt(D / (rho * h)) for lam in LEISSA_CFFF[:2]]
    return [(float(x), float(r), float(100 * (x / r - 1))) for x, r in zip(f[:2], ref)]


def convergence_beam(levels=(5, 10, 20, 40), L=0.4, b=0.02) -> list:
    """The cantilever's first frequency as the mesh refines: [(segments, Hz, error %)]."""
    E, nu, rho = 70e9, 0.33, 2700.0
    ref = euler_bernoulli(EB_BETA_CANTILEVER[0], L, E, b ** 4 / 12, rho, b * b)
    out = []
    for n in levels:
        K, M, free = bar_model(L, b, b, E, nu, rho, n, 2, 'cantilever')
        f, _ = eigen(K, M, 1, free)
        out.append((n, float(f[0]), float(100 * (f[0] / ref - 1))))
    return out


def validation_checks() -> list[dict]:
    def cant():
        r = validate_beams()
        e = max(abs(x[2]) for x in r['cantilever'])
        return {'error': e, 'detail': r}

    def free():
        r = validate_beams()
        return {'error': abs(r['free'][0][2]), 'detail': r, 'ok': r['rigid_modes'] == 6}

    def plate():
        r = validate_plate()
        return {'error': max(abs(x[2]) for x in r), 'detail': {'modes': r}}
    return [
        {'id': 'fea.cantilever_eb', 'model': 'fea', 'title': 'Cantilever bar against Euler–Bernoulli',
         'quantity': 'worst |f_FEA / f_EB − 1| of the first two bending modes, %', 'reference':
         'Euler–Bernoulli closed form, βL = 1.8751, 4.6941 (20 × 20 × 400 mm aluminium bar)', 'tolerance': 2.0,
         'unit': '%', 'refs': (), 'fn': cant, 'cost': 'study',
         'why': 'L/h = 20: shear and rotary inertia lower a real bar by under 1 % (Timoshenko), the rest is mesh'},
        {'id': 'fea.free_free_eb', 'model': 'fea', 'title': 'Free-free bar against Euler–Bernoulli',
         'quantity': '|f_FEA / f_EB − 1| of the first elastic mode, % (and six rigid-body modes)',
         'reference': 'Euler–Bernoulli closed form, βL = 4.7300', 'tolerance': 2.0, 'unit': '%', 'refs': (),
         'fn': free, 'cost': 'study', 'why': 'as the cantilever; a free solid must show exactly six zero modes'},
        {'id': 'fea.plate_leissa', 'model': 'fea', 'title': 'Square cantilever plate against Leissa',
         'quantity': 'worst |f_FEA / f_Leissa − 1| of the first two modes, %',
         'reference': 'Leissa (1969) Vibration of Plates, NASA SP-160: C-F-F-F square, λ = 3.492, 8.525 (ν = 0.3)',
         'tolerance': 3.0, 'unit': '%', 'refs': (), 'fn': plate, 'cost': 'study',
         'why': 'a/h = 20: Mindlin effects lower the solid plate by about 1 %; Leissa\'s values are Kirchhoff'},
    ]
