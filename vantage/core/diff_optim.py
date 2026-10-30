"""Gradient refinement of a rig: the cameras' six pose variables (and, optionally, the stitch seams' angles) tuned
continuously on a smooth surrogate of the coverage objective, after the evolutionary placement search
(core.optimiser) has made the discrete choices.

Surrogate. The ground of the ROI is sampled on a grid sample_cell wide (the analysis's ROI grid at that cell), refined
to the analysis's own cells around the ground the exact analysis finds blind or below the minimum density at the start
of each stage, every point weighted by the ground it stands for (m²). For each camera c and point i:
  * in view: exp(Σ log σ(margin / τ)) over the margin to the lens's θ_max (rad) and to the four image borders (px),
    the point projected through the real lens model (core.lens: r(θ) of the projection, the unit calibration when the
    camera has one, the principal point and the pixel aspect);
  * clear: σ((occluder depth − (range − tol)) / τ) per texel of the engine's cube depth map of the camera (body, rig and
    housings; core.occlusion), tol = 0.02 m + 1 % of the range as the analysis has it, interpolated bilinearly between
    the four texels around the ray, and averaged over a small stencil of directions τ_dir wide so it varies smoothly
    with the direction to the point (the sigmoid is interpolated, not the depth: a depth interpolated across a
    silhouette would put a step where the near and far texels meet);
  * p_ci = in view × clear; dense enough: q_ci = p_ci × σ(ln(density / minimum) / τ), the density in px/cm as the
    analysis defines it (lens.Intrinsics.density_px_per_cm: √(radial × tangential) sampling over the range).
Coverage = 1 − Π_c (1 − p_ci). Soft blind area B = Σ a_i Π(1 − p_ci); soft density shortfall L = Σ a_i (Π(1 − q_ci) −
Π(1 − p_ci)), the seen ground below the minimum density.
Critical targets (core.critical, enabled ones, the cameras' requirement only, as the optimiser counts them): each
camera's nodes of the target's shape (critical.shape, the coarse sampling the optimiser uses) are seen softly as above;
a station's pixel span is Σ ½(seen_k + seen_k+1) × |segment in the image|; a standing target is met when its seen share
reaches min_share and its best station (a soft maximum) min_px, in focus (soft); a kerb or a marking when the share of
stations reaching min_px does. U = Σ_t Π_c (1 − met_ct) unresolved targets.
Seams: every standing critical target and every obstacle (Analysis ▸ Obstacles) costs up to 1 when a stitch seam passes
near it: σ(along / 0.1 m) × exp(−½ (across / w)²), w = half the variant's blend width + half the object's width,
combined over the seams as 1 − Π(1 − g). The seam angles are the analysis's: an automatic seam is its soft argmax over
the same 41 candidate angles and score (core.analysis.compute_seams: share of the ray both cameras see, their density
there, a small pull to the middle) with soft visibility, so moving a camera moves the seam; a manual seam stays; with
`seams` on, every seam angle is a variable (±seam_span_deg around its start) and a seam pays (1 − share of its ray both
cameras see) × 0.5 w_seam so it stays where both cameras see. S = Σ objects near a seam.
Objective = w_blind·B + w_density·L + w_critical·U + w_seam·S (+ the seam overlap term), plus two penalties: the
distance off the surface (below) and a small pull to the start pose (1e-4 per 10 mm or 1° moved) so moves the
objective does not ask for are not made.

Variables. Per camera x, y, z (vehicle frame, ISO 8855, mm from the start) and yaw, pitch, roll (deg, intrinsic Z-Y-X,
pitch positive down). Bounds: ± max_move_mm per axis, inside the bounding box of the camera's mount zone (the zone of
the surface it sits on, grown by its distance off the surface + 10 mm); ± max_turn_deg in yaw and pitch, ± max_roll_deg
in roll. A camera stays on its surface by a PENALTY during a stage (1 per 5 mm that its distance to the closest
surface point differs from its distance at the start, squared; the gradient along the closest face's normal) and is
RE-PROJECTED onto the mesh after every stage (the closest surface point plus the start distance along its normal);
a re-projection that lands outside the camera's mount zone (or on a forbidden zone) is pulled back toward the start
by halves until it is inside. A locked camera does not move; a mirrored pair (linked L/R that are mirror images now)
moves as one: the second camera is the mirror image of the first (x, −y, z, −yaw, pitch, −roll); a linked pair that is
not a mirror image now moves apart (said in the notes); a hidden camera stays as it is and does not count.

Solver. scipy L-BFGS-B with the gradient from torch autograd (CPU, float64), τ annealed from soft to sharp over
`stages` stages (every τ times 1, …, sharp_end, geometric); the cube maps, the refined ground sample, the target shapes
and the seam candidates are rebuilt at the start of every stage from the poses then. A stage stops after `iterations`
iterations or when no variable moved more than 0.01° / 0.1 mm in one iteration: the solver's STEP tolerance, not the
accuracy of the result in the world (the occluder mesh, the 0.35° cube texels and the analysis's 5 cm cells are far
coarser than that).

Hybrid search. Each start (the current variant, or the evolutionary optimiser's Pareto candidates) is refined, then
every layout the refinement passes through at a stage boundary is checked with the exact engine at the project's
resolution (Engine.analyse; critical targets with critical.TargetCheck, exact line of sight), and the best exact layout
is kept (the start itself when nothing beats it). The surrogate's figures at the sharpest τ are reported beside the
exact ones, before and after.

Limits (also in to_doc): the cube maps belong to the stage's start positions, so within a stage the parallax of
occluders as a camera moves is ignored (a few cm at most; a stage rebuilds them); the ground is flat at the analysis's
obstacle height (a road, analysis.terrain, is only in the exact check); critical targets are seen through the cube
maps in the surrogate, exactly in the check; lidar points on the targets are not counted; the method is local: it
finds a nearby optimum of the surrogate and leaves the discrete choices (which cameras, which mount) to the
evolutionary search. The τ values are heuristics chosen on the sample, not derived.
"""
from __future__ import annotations

import copy
import csv
import io
import json
import math
import time
from dataclasses import dataclass, field

import numpy as np

from .lens import resolve
from .mathutil import BODY_TO_OPTICAL

SCHEMA = 'vantage.diff_optim/1'
FRAME = 'vehicle (ISO 8855, rear axle)'
STARTS = ('variant', 'pareto')

# base softness at the first stage (each is multiplied by the stage's sharpness s, 1 … sharp_end); heuristics
TAU_THETA = math.radians(2.0)       # rad, margin to the lens's θ_max
TAU_PX = 30.0                       # px, margin to the image border
TAU_OCC = 0.10                      # m, occluder depth against the point's range
TAU_DIR = 1.0                       # deg, the direction stencil of the occlusion lookup
TAU_DENS = 0.15                     # ln(density / minimum)
TAU_SHARE = 0.05                    # share of a target seen / of its stations resolved
TAU_TPX = 3.0                       # px, a target's pixel span against min_px
TAU_FOCUS = 0.10                    # m, the focus range
TAU_SEAM = 0.03                     # score units, the soft argmax of an automatic seam
DEPTH_FAR = 1e3                     # m, a texel where nothing is hit
SURFACE_MM = 5.0                    # the surface penalty is 1 per this distance off, squared
MOVE_PULL = 1e-4                    # pull to the start pose per (10 mm)² or (1°)²
STEP_MM = 0.1                       # the solver's step tolerance (mm, deg)
STEP_DEG = 0.01
MAX_STARTS = 5

LIMITS = {'candidates': (1, MAX_STARTS), 'w_blind': (0.0, 100.0), 'w_density': (0.0, 100.0),
          'w_critical': (0.0, 100.0), 'w_seam': (0.0, 100.0), 'max_move_mm': (0.0, 300.0),
          'max_turn_deg': (0.0, 30.0), 'max_roll_deg': (0.0, 30.0), 'seam_span_deg': (0.0, 45.0),
          'sample_cell': (0.05, 0.40), 'stages': (1, 8), 'sharp_end': (0.01, 1.0), 'iterations': (1, 500)}
INTS = ('candidates', 'stages', 'iterations')


@dataclass
class DiffSettings:
    start: str = 'variant'          # variant: the current variant | pareto: the optimiser's Pareto candidates
    candidates: int = 3             # Pareto candidates refined (the first of the front)
    w_blind: float = 1.0            # per m² of blind ground
    w_density: float = 0.25         # per m² of seen ground below the minimum density
    w_critical: float = 0.5         # per unresolved critical target
    w_seam: float = 0.1             # per object or standing target near a stitch seam
    max_move_mm: float = 50.0       # per axis, from the start pose
    max_turn_deg: float = 8.0       # yaw and pitch, from the start pose
    max_roll_deg: float = 3.0
    move: bool = True               # positions free (off: aim only)
    seams: bool = False             # the seam angles as variables (written as manual seams)
    seam_span_deg: float = 15.0
    sample_cell: float = 0.10       # m, the coarse ground sample
    stages: int = 4                 # annealing stages
    sharp_end: float = 0.1          # the last stage's τ as a share of the first's
    iterations: int = 40            # L-BFGS-B iterations per stage at most

    def to_dict(self):
        return {k: getattr(self, k) for k in self.__dataclass_fields__}

    @staticmethod
    def from_dict(d) -> 'DiffSettings':
        """Tolerant: unknown keys dropped, numbers clamped to LIMITS, junk falls back to the default."""
        s = DiffSettings()
        d = d if isinstance(d, dict) else {}
        if d.get('start') in STARTS:
            s.start = d['start']
        for k, (lo, hi) in LIMITS.items():
            if k not in d or isinstance(d[k], bool):
                continue
            try:
                v = float(d[k])
            except (TypeError, ValueError):
                continue
            if not math.isfinite(v):
                continue
            v = min(max(v, lo), hi)
            setattr(s, k, int(round(v)) if k in INTS else v)
        for k in ('move', 'seams'):
            v = d.get(k)
            if isinstance(v, bool) or (isinstance(v, int) and not isinstance(v, float)):
                setattr(s, k, bool(v))
            elif isinstance(v, str) and v.strip().lower() in ('1', 'true', 'yes', 'on', '0', 'false', 'no', 'off'):
                setattr(s, k, v.strip().lower() in ('1', 'true', 'yes', 'on'))
        return s

    def sharpness(self) -> list:
        """The τ multiplier of every stage, 1 … sharp_end (geometric)."""
        n = max(1, int(self.stages))
        if n == 1:
            return [float(self.sharp_end)]
        return [float(self.sharp_end ** (k / (n - 1))) for k in range(n)]


def settings_of(project) -> DiffSettings:
    return DiffSettings.from_dict(getattr(project.analysis, 'diff_optim', None))


# ============================================================================================ poses and units
POSE = ('x', 'y', 'z', 'yaw', 'pitch', 'roll')


def pose_of(cam) -> tuple:
    return (float(cam.x), float(cam.y), float(cam.z), float(cam.yaw), float(cam.pitch), float(cam.roll))


def mirror(p) -> tuple:
    """The mirror image of a pose in the vehicle's centre plane (y = 0)."""
    return (p[0], -p[1], p[2], -p[3], p[4], -p[5])


def _is_mirror(a, b) -> bool:
    m = mirror(pose_of(a))
    q = pose_of(b)
    return all(abs(m[i] - q[i]) <= 1e-3 for i in range(3)) and all(abs(m[i] - q[i]) <= 0.01 for i in range(3, 6))


def _set_pose(cam, p):
    cam.x, cam.y, cam.z, cam.yaw, cam.pitch, cam.roll = (float(v) for v in p)


@dataclass
class Unit:
    lead: str
    follow: str = ''                # the mirror partner, moved as the lead's mirror image
    fixed: bool = False             # locked (or the partner of a locked camera)
    why: str = ''


def units_of(cams) -> tuple[list, list]:
    """The moving units of the visible cameras (a mirrored pair together, a locked camera and its mirror partner
    fixed) and notes about pairs that move apart."""
    vis = [c for c in cams if c.visible]
    by = {c.uid: c for c in vis}
    out, notes, seen = [], [], set()
    for c in vis:
        if c.uid in seen:
            continue
        seen.add(c.uid)
        p = by.get(c.link_uid) if c.link_uid else None
        if p is not None and p.uid not in seen:
            if _is_mirror(c, p):
                seen.add(p.uid)
                fixed = c.locked or p.locked
                out.append(Unit(c.uid, p.uid, fixed, 'locked' if fixed else ''))
                continue
            notes.append(f'{c.tag} and {p.tag} are linked but not mirror images now: they move apart.')
        out.append(Unit(c.uid, '', bool(c.locked), 'locked' if c.locked else ''))
    return out, notes


# ============================================================================================ torch helpers
def _torch():
    import torch
    return torch


def torch_missing() -> str | None:
    """Why the refinement cannot run here (the downloads leave torch out), or None."""
    from .splat_raster import TORCH_MISSING, torch_ok
    return None if torch_ok() else TORCH_MISSING.format(what='Gradient refinement', build='the CPU build is enough')


class _one_thread:
    """torch on one intra-op thread while refining (restored after): the surrogate's tensors are a few thousand
    elements, where a thread pool only adds overhead, and its spin-waiting threads starve a loaded machine."""

    def __enter__(self):
        T = _torch()
        self.n = T.get_num_threads()
        T.set_num_threads(1)
        return self

    def __exit__(self, *exc):
        _torch().set_num_threads(self.n)
        return False


def _rot_t(T, yaw, pitch, roll):
    """rot_zyx for torch scalars in degrees (body -> vehicle)."""
    y, p, r = (a * (math.pi / 180.0) for a in (yaw, pitch, roll))
    cy, sy, cp, sp, cr, sr = T.cos(y), T.sin(y), T.cos(p), T.sin(p), T.cos(r), T.sin(r)
    return T.stack([T.stack([cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr]),
                    T.stack([sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr]),
                    T.stack([-sp, cp * sr, cp * cr])])


def _r_t(T, t, f, proj, kb):
    """lens.r_of_theta in torch (mm on the sensor)."""
    if proj == 'rectilinear':
        return f * T.tan(T.clamp(t, max=math.radians(89.9)))
    if proj == 'equidistant':
        return f * t
    if proj == 'equisolid':
        return 2 * f * T.sin(t / 2)
    if proj == 'stereographic':
        return 2 * f * T.tan(T.clamp(t, max=math.radians(179)) / 2)
    if proj == 'orthographic':
        return f * T.sin(T.clamp(t, max=math.pi / 2))
    if proj == 'kannala_brandt':
        k1, k2, k3, k4 = kb
        t2 = t * t
        return f * t * (1 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4))))
    raise ValueError(proj)


def _dr_t(T, t, f, proj, kb):
    """lens.dr_dtheta in torch."""
    if proj == 'rectilinear':
        c = T.cos(T.clamp(t, max=math.radians(89.9)))
        return f / (c * c)
    if proj == 'equidistant':
        return f * T.ones_like(t)
    if proj == 'equisolid':
        return f * T.cos(t / 2)
    if proj == 'stereographic':
        c = T.cos(t / 2)
        return f / (c * c)
    if proj == 'orthographic':
        return f * T.cos(T.clamp(t, max=math.pi / 2))
    if proj == 'kannala_brandt':
        k1, k2, k3, k4 = kb
        t2 = t * t
        return f * (1 + t2 * (3 * k1 + t2 * (5 * k2 + t2 * (7 * k3 + t2 * 9 * k4))))
    raise ValueError(proj)


# ============================================================================================ the exact check
def _variant_with(variant, cams, seams=None):
    """A copy of variant with these cameras (and manual seams added)."""
    v = copy.copy(variant)
    v.cameras = [copy.deepcopy(c) for c in cams]
    v.seams = dict(variant.seams)
    if seams:
        v.seams.update(seams)
    return v


def _objects(project, crit_targets) -> list:
    """(name, x, y, half width) of every standing critical target and obstacle: what must not sit on a seam."""
    from .critical import LINEAR
    out = [(t['name'], float(t['x']), float(t['y']), float(t['width']) / 2) for t in crit_targets
           if t['kind'] not in LINEAR]
    out += [(o.name, float(o.x), float(o.y), float(o.radius)) for o in project.obstacles]
    return out


def _near_seams(objects, centre, angles_deg, blend_m) -> list:
    """Names of the objects within w of a seam line (on its side of the centre)."""
    out = []
    for name, x, y, hw in objects:
        rx, ry = x - centre[0], y - centre[1]
        w = blend_m / 2 + hw
        for a in angles_deg:
            c, s = math.cos(math.radians(a)), math.sin(math.radians(a))
            if rx * c + ry * s > 0 and abs(-rx * s + ry * c) < w:
                out.append(name)
                break
    return out


def exact_check(project, engine, variant, cams, st: DiffSettings, crit_targets, objects, seams=None) -> dict:
    """The exact figures of a layout at the project's resolution: blind and below-minimum ground (m²), unresolved
    critical targets, objects near a seam and the weighted objective (as the surrogate weighs its own)."""
    v = _variant_with(variant, cams, seams)
    res = engine.analyse(project, variant=v)
    g = res.grid
    a = g.cell ** 2
    low = float((g.roi & (res.count > 0) & (res.best_dens < project.analysis.min_density)).sum() * a)
    blind = float(res.metrics['blind_m2'])
    unresolved, missing = 0, []
    if crit_targets:
        from .critical import TargetCheck, CriticalSettings
        tc = TargetCheck(project, engine.vehicle_geom(project, v), v, CriticalSettings(targets=list(crit_targets)))
        met = tc.met(v.cameras)
        unresolved = int((~met).sum())
        missing = tc.names(met)
    near = _near_seams(objects, res.centre, [s.angle for s in res.seams], v.blend_m)
    obj = st.w_blind * blind + st.w_density * low + st.w_critical * unresolved + st.w_seam * len(near)
    return {'blind_m2': blind, 'low_m2': low, 'unresolved': unresolved, 'missing': missing, 'seam_near': len(near),
            'near': near, 'objective': float(obj), 'min_density': float(res.metrics['min_density']),
            'covered_pct': float(res.metrics['covered_pct']), 'min_ok_pct': float(res.metrics['min_ok_pct']),
            'seams': [{'a': s.a, 'b': s.b, 'angle': float(s.angle), 'auto': bool(s.auto)} for s in res.seams],
            'result': res}


# ============================================================================================ the surrogate
class Surrogate:
    """The smooth objective of one stage: built from the layout at the stage's start (its cube maps, its exact
    analysis for the refined ground sample and the seams), evaluated for any variable vector x (offsets from the
    refinement's start pose: mm and degrees per unit, then the seam offsets in degrees)."""

    def __init__(self, project, engine, variant, orig: dict, units: list, x0, sharp: float, st: DiffSettings,
                 exact: dict, crit_targets, objects, seam_keys: list, seam_start: dict, mesh_info: dict):
        T = _torch()
        self.T = T
        self.project, self.engine, self.st, self.s = project, engine, st, float(sharp)
        self.orig, self.units = orig, units
        self.seam_keys = list(seam_keys)
        self.seam_start = seam_start
        self.mesh_info = mesh_info
        self.n_unit = 6 * len(units)
        x0 = np.asarray(x0, float)
        poses = poses_of(orig, units, x0)
        cams = [c for c in variant.cameras if c.visible]
        self.cams = cams
        v = _variant_with(variant, cams)
        for c in v.cameras:
            _set_pose(c, poses[c.uid])
        self.vg = engine.vehicle_geom(project, v)
        res = exact['result']
        self.res = res
        self.dt = T.float64
        self.B = T.tensor(BODY_TO_OPTICAL, dtype=self.dt)
        z = {'ground': 0.0, 'pole05': 0.5, 'pole10': 1.0, 'child': 0.9}.get(project.analysis.obstacle_mode, 0.0)
        self.z = z
        # ---- cameras: lens, cube map at the stage's start position
        self.cdat = []
        for c in v.cameras:
            intr = resolve(project.module_for(c), project.lens_for(c))
            cube = engine.cube(c.pos(), self.vg)
            dep = np.where(np.isfinite(cube.depth), cube.depth, DEPTH_FAR).astype(np.float64)
            self.cdat.append({'uid': c.uid, 'tag': c.tag, 'intr': intr, 'n': cube.depth.shape[1],
                              'depth': T.tensor(dep.reshape(-1), dtype=self.dt)})
        self.index = {d['uid']: i for i, d in enumerate(self.cdat)}
        # ---- ground: coarse cells, the analysis's own cells around blind and below-minimum ground
        self.G, self.Ga = _ground_sample(res, project, st.sample_cell, z)
        self.Gt = T.tensor(self.G, dtype=self.dt)
        self.Gat = T.tensor(self.Ga, dtype=self.dt)
        self.min_d = float(project.analysis.min_density)
        # ---- critical targets: each camera's shape of each target, from its start position
        from .critical import shape, need_of, LINEAR
        self.targets = []
        for t in crit_targets:
            per = []
            for c in v.cameras:
                sh = shape(t, c.pos().astype(float), fine=False)
                per.append({'nodes': T.tensor(sh.flat, dtype=self.dt), 'S': sh.nodes.shape[0], 'N': sh.nodes.shape[1],
                            'centre': T.tensor(sh.centre, dtype=self.dt),
                            'centreline': None if sh.centreline is None else T.tensor(sh.centreline, dtype=self.dt)})
            self.targets.append({'t': t, 'need': need_of(t), 'linear': t['kind'] in LINEAR, 'per': per})
        # ---- seams and the objects that must not sit on them
        self.objects = objects
        self.blend = float(variant.blend_m)
        self.centre = (float(res.centre[0]), float(res.centre[1]))
        self.seams = _seam_plan(res, v, st.sample_cell, z, self.seam_keys, self.seam_start, self.index) \
            if objects else []
        if objects:
            self.obj_xy = T.tensor([[o[1] - self.centre[0], o[2] - self.centre[1]] for o in objects], dtype=self.dt)
            self.obj_w = T.tensor([self.blend / 2 + o[3] for o in objects], dtype=self.dt)
        # the stencil of directions of the occlusion lookup (in texels, per camera's face size)
        self.n_eval = 0

    # ------------------------------------------------------------------ one camera
    def _look(self, k, R, pos, P):
        """Soft seen (K,), density px/cm (K,), u, v (px) of points P (K, 3) for camera k at pose (R, pos)."""
        T, s = self.T, self.s
        cd = self.cdat[k]
        intr = cd['intr']
        d = P - pos
        dist = T.sqrt((d * d).sum(1) + 1e-12)
        opt = d @ (R @ self.B.T)
        x, y, zc = opt[:, 0], opt[:, 1], opt[:, 2]
        rho = T.sqrt(x * x + y * y + 1e-18)
        th = T.atan2(rho, zc)
        r = _r_t(T, th, intr.f_mm, intr.projection, intr.kb) / intr.pixel_mm
        u = intr.cx + x * r / rho
        v = intr.cy + y * r / rho * intr.aspect
        ls = T.nn.functional.logsigmoid
        tp = TAU_PX * s
        lv = ls((intr.max_theta - th) / (TAU_THETA * s)) + ls(u / tp) + ls((intr.width - u) / tp) + \
            ls(v / tp) + ls((intr.height - v) / tp)
        if intr.projection == 'rectilinear':
            lv = lv + ls(zc / dist / (0.02 * s))
        view = T.exp(lv)
        clear = self._clear(k, d / dist[:, None], dist)
        thc = T.clamp(th, min=1e-4)
        radial = T.clamp(_dr_t(T, thc, intr.f_mm, intr.projection, intr.kb), min=1e-6) / intr.pixel_mm
        tang = T.clamp(_r_t(T, thc, intr.f_mm, intr.projection, intr.kb), min=1e-9) / intr.pixel_mm / \
            T.sin(T.clamp(thc, max=math.pi - 1e-3))
        dens = T.sqrt(radial * tang * intr.aspect) / (T.clamp(dist, min=1e-3) * 100.0)
        return view * clear, dens, u, v, dist

    def _clear(self, k, dirs, dist):
        """Soft line of sight: the per-texel sigmoid of (depth − (range − tol)) interpolated bilinearly at the ray's
        direction, averaged over a stencil τ_dir wide."""
        T, s = self.T, self.s
        cd = self.cdat[k]
        n = cd['n']
        with T.no_grad():
            ad = dirs.abs()
            ax = ad.argmax(1)
            neg = dirs.gather(1, ax[:, None])[:, 0] < 0
            face = ax * 2 + neg.long()
        FW, RT, UP = _face_basis(T, self.dt)
        a = (dirs * FW[face]).sum(1)
        uf = (dirs * RT[face]).sum(1) / a
        vf = (dirs * UP[face]).sum(1) / a
        px = (uf + 1) * 0.5 * n - 0.5
        py = (1 - vf) * 0.5 * n - 0.5
        thr = dist - (0.02 + 0.01 * dist)
        tau = TAU_OCC * s
        dep = cd['depth']
        delta = TAU_DIR * s / (90.0 / n)
        offs = ((0.0, 0.0),) if delta < 0.25 else ((0.0, 0.0), (delta, 0.0), (-delta, 0.0), (0.0, delta),
                                                     (0.0, -delta))
        base = face * (n * n)
        acc = 0.0
        for ox, oy in offs:
            qx = T.clamp(px + ox, 0.0, n - 1.0)
            qy = T.clamp(py + oy, 0.0, n - 1.0)
            with T.no_grad():
                i0 = T.clamp(T.floor(qx), max=n - 2).long()
                j0 = T.clamp(T.floor(qy), max=n - 2).long()
            fx = qx - i0
            fy = qy - j0
            idx = base + j0 * n + i0
            c00 = T.sigmoid((dep[idx] - thr) / tau)
            c01 = T.sigmoid((dep[idx + 1] - thr) / tau)
            c10 = T.sigmoid((dep[idx + n] - thr) / tau)
            c11 = T.sigmoid((dep[idx + n + 1] - thr) / tau)
            acc = acc + (1 - fx) * (1 - fy) * c00 + fx * (1 - fy) * c01 + (1 - fx) * fy * c10 + fx * fy * c11
        return acc / len(offs)

    # ------------------------------------------------------------------ the objective
    def _poses_t(self, xt):
        """{uid: (R (3,3), pos (3,))} for the variables xt (torch)."""
        T = self.T
        out = {}
        for ui, u in enumerate(self.units):
            o = self.orig[u.lead]
            dx = xt[6 * ui:6 * ui + 6]
            pos = T.stack([o[0] + dx[0] / 1000, o[1] + dx[1] / 1000, o[2] + dx[2] / 1000])
            yaw, pitch, roll = o[3] + dx[3], o[4] + dx[4], o[5] + dx[5]
            out[u.lead] = (_rot_t(T, yaw, pitch, roll), pos)
            if u.follow:
                mpos = T.stack([pos[0], -pos[1], pos[2]])
                out[u.follow] = (_rot_t(T, -yaw, pitch, -roll), mpos)
        return out

    def terms(self, xt):
        """The objective's terms (torch scalars): blind, low, unresolved, seam_near, overlap."""
        T, s = self.T, self.s
        poses = self._poses_t(xt)
        C = len(self.cdat)
        Kg = len(self.G)
        ps, qs = [], []
        tgt_seen = [[None] * C for _ in self.targets]
        seam_look = {}
        sv = xt[self.n_unit:]
        seam_pts = []
        for j, sm in enumerate(self.seams):
            if sm['mode'] == 'auto':
                seam_pts.append(sm['P'])
            elif sm['mode'] == 'var':
                phi = sm['phi0'] + sv[sm['var']] * (math.pi / 180.0)
                r = sm['radii']
                seam_pts.append(T.stack([self.centre[0] + T.cos(phi) * r, self.centre[1] + T.sin(phi) * r,
                                         T.full_like(r, self.z)], 1))
            else:
                seam_pts.append(None)
        for k, cd in enumerate(self.cdat):
            R, pos = poses[cd['uid']]
            parts = [self.Gt]
            sizes = [Kg]
            for t in self.targets:
                parts.append(t['per'][k]['nodes'])
                sizes.append(len(t['per'][k]['nodes']))
            mine = [j for j, sm in enumerate(self.seams) if seam_pts[j] is not None and k in (sm['ia'], sm['ib'])]
            for j in mine:
                parts.append(seam_pts[j])
                sizes.append(len(seam_pts[j]))
            P = T.cat(parts, 0)
            seen, dens, u, v, dist = self._look(k, R, pos, P)
            chunks = list(T.split(seen, sizes))
            dchunks = list(T.split(dens, sizes))
            uch, vch = T.split(u, sizes), T.split(v, sizes)
            p = chunks[0]
            ps.append(p)
            qs.append(p * T.sigmoid(T.log(T.clamp(dchunks[0], min=1e-9) / self.min_d) / (TAU_DENS * s)))
            for ti in range(len(self.targets)):
                tgt_seen[ti][k] = (chunks[1 + ti], uch[1 + ti], vch[1 + ti], pos)
            for n_, j in enumerate(mine):
                seam_look[(j, k)] = (chunks[1 + len(self.targets) + n_], dchunks[1 + len(self.targets) + n_])
        Pm = T.stack(ps)
        Qm = T.stack(qs)
        miss_p = T.prod(1 - Pm, 0)
        miss_q = T.prod(1 - Qm, 0)
        blind = (self.Gat * miss_p).sum()
        low = (self.Gat * (miss_q - miss_p)).sum()
        unresolved = T.zeros((), dtype=self.dt)
        for ti, t in enumerate(self.targets):
            miss = T.ones((), dtype=self.dt)
            for k in range(C):
                miss = miss * (1 - self._met(t, k, *tgt_seen[ti][k]))
            unresolved = unresolved + miss
        seam_near = T.zeros((), dtype=self.dt)
        overlap = T.zeros((), dtype=self.dt)
        if self.seams:
            phis = []
            for j, sm in enumerate(self.seams):
                if sm['mode'] == 'manual':
                    phis.append(T.tensor(sm['phi0'], dtype=self.dt))
                    continue
                pa, da = seam_look[(j, sm['ia'])]
                pb, db = seam_look[(j, sm['ib'])]
                both = pa * pb
                if sm['mode'] == 'var':
                    phis.append(sm['phi0'] + sv[sm['var']] * (math.pi / 180.0))
                    overlap = overlap + (1 - both.mean())
                    continue
                J, Rn = sm['J'], sm['R']
                both = both.reshape(J, Rn)
                dmin = T.clamp(T.minimum(da, db), 0.0, 10.0).reshape(J, Rn)
                m = sm['inroi']
                nin = T.clamp(m.sum(1), min=1.0)
                frac = (both * m).sum(1) / nin
                dsc = (both * dmin * m).sum(1) / nin / 10
                score = frac + 0.25 * dsc - sm['pull']
                w = T.softmax(score / (TAU_SEAM * s), 0)
                phis.append((w * sm['cand']).sum())
            ph = T.stack(phis)
            e = T.stack([T.cos(ph), T.sin(ph)], 1)                       # (J, 2)
            along = self.obj_xy @ e.T                                      # (O, J)
            across = -self.obj_xy[:, :1] * e[:, 1][None] + self.obj_xy[:, 1:] * e[:, 0][None]
            g = T.sigmoid(along / 0.1) * T.exp(-0.5 * (across / self.obj_w[:, None]) ** 2)
            seam_near = (1 - T.prod(1 - g, 1)).sum()
        return {'blind': blind, 'low': low, 'unresolved': unresolved, 'seam_near': seam_near, 'overlap': overlap}

    def _met(self, t, k, seen, u, v, pos):
        """Soft: camera k meets target t (measure() with soft visibility)."""
        T, s = self.T, self.s
        per = t['per'][k]
        need = t['need']
        S, N = per['S'], per['N']
        seen = seen.reshape(S, N)
        u, v = u.reshape(S, N), v.reshape(S, N)
        L = T.sqrt(T.diff(u, dim=1) ** 2 + T.diff(v, dim=1) ** 2 + 1e-9)
        f = (seen[:, :-1] + seen[:, 1:]) / 2
        px_st = (f * L).sum(1)
        seen_st = f.mean(1)
        intr = self.cdat[k]['intr']
        ts = TAU_SHARE * s
        tpx = max(TAU_TPX * s, 0.25)
        tf = TAU_FOCUS * s

        def focus(dist):
            out = T.sigmoid((dist - intr.focus_near) / tf)
            if math.isfinite(intr.focus_far):
                out = out * T.sigmoid((intr.focus_far - dist) / tf)
            return out
        if t['linear']:
            dist_st = T.sqrt(((per['centreline'] - pos) ** 2).sum(1))
            pe = px_st * (focus(dist_st) if need.focus else 1.0)
            frac = T.sigmoid((pe - need.min_px) / tpx).mean()
            kk = max(1, int(math.ceil(need.min_share * S - 1e-9)))
            return T.sigmoid((frac - (kk - 0.5) / S) / ts)
        w = T.softmax(px_st / tpx, 0)
        px = (w * px_st).sum()
        met = T.sigmoid((seen_st.mean() - need.min_share) / ts) * T.sigmoid((px - need.min_px) / tpx)
        if need.focus:
            met = met * focus(T.sqrt(((per['centre'] - pos) ** 2).sum()))
        return met

    def objective_t(self, xt):
        st = self.st
        tm = self.terms(xt)
        f = st.w_blind * tm['blind'] + st.w_density * tm['low'] + st.w_critical * tm['unresolved'] + \
            st.w_seam * (tm['seam_near'] + 0.5 * tm['overlap'])
        return f, tm

    def value_and_grad(self, x):
        """(objective, gradient) at x (numpy float64), the surface penalty and the pull to the start included."""
        T = self.T
        xt = T.tensor(np.asarray(x, float), dtype=self.dt, requires_grad=True)
        f, _ = self.objective_t(xt)
        f.backward()
        g = xt.grad.detach().numpy().copy()
        val = float(f.detach())
        pv, pg = self.penalty(x)
        self.n_eval += 1
        return val + pv, g + pg

    def value(self, x) -> float:
        T = self.T
        with T.no_grad():
            f, _ = self.objective_t(T.tensor(np.asarray(x, float), dtype=self.dt))
        return float(f) + self.penalty(x)[0]

    def parts(self, x) -> dict:
        """The surrogate's figures at x (no penalties): blind_m2, low_m2, unresolved, seam_near, objective."""
        T = self.T
        with T.no_grad():
            f, tm = self.objective_t(T.tensor(np.asarray(x, float), dtype=self.dt))
        return {'blind_m2': float(tm['blind']), 'low_m2': float(tm['low']), 'unresolved': float(tm['unresolved']),
                'seam_near': float(tm['seam_near']), 'objective': float(f)}

    def penalty(self, x):
        """The surface penalty and the pull to the start pose, with their analytic gradient (numpy)."""
        x = np.asarray(x, float)
        g = np.zeros_like(x)
        val = 0.0
        X = x[:self.n_unit].reshape(-1, 6)
        sc = np.array([10.0, 10.0, 10.0, 1.0, 1.0, 1.0])
        val += MOVE_PULL * float(((X / sc) ** 2).sum())
        g[:self.n_unit] += (MOVE_PULL * 2 * X / sc ** 2).reshape(-1)
        mi = self.mesh_info
        if mi is None or not self.st.move:
            return val, g
        for ui, u in enumerate(self.units):
            if u.fixed:
                continue
            o = self.orig[u.lead]
            p = np.array(o[:3]) + X[ui, :3] / 1000.0
            gap, n = _surface_gap(mi, p)
            if gap is None:
                continue
            e = (gap - mi['gap'][u.lead]) * 1000.0 / SURFACE_MM
            val += e * e
            g[6 * ui:6 * ui + 3] += 2 * e * (1000.0 / SURFACE_MM) * n / 1000.0
        return val, g


_BASIS = {}


def _face_basis(T, dt):
    key = str(dt)
    if key not in _BASIS:
        from .occlusion import FACES
        _BASIS[key] = tuple(T.tensor(np.array([f[i] for f in FACES]), dtype=dt) for i in range(3))
    return _BASIS[key]


def _ground_sample(res, project, cell, z):
    """Ground points (K, 3) and the area each stands for (K,): the ROI on cells `cell` wide, the analysis's own
    cells around ground that is blind or below the minimum density (two coarse cells around it)."""
    from scipy import ndimage
    g = res.grid
    a = g.cell ** 2
    r = max(1, int(round(cell / g.cell)))
    roi = g.roi
    bad = roi & ((res.count == 0) | (res.best_dens < project.analysis.min_density))
    if r > 1 and bad.any():
        bad = ndimage.binary_dilation(bad, iterations=2 * r)
    xs, ys = g.xs, g.ys
    P, W = [], []
    for J in range(0, g.ny, r):
        for I in range(0, g.nx, r):
            blk = roi[J:J + r, I:I + r]
            if not blk.any():
                continue
            jj, ii = np.nonzero(blk)
            if r == 1 or bad[J:J + r, I:I + r].any():
                for j, i in zip(jj, ii):
                    P.append((xs[I + i], ys[J + j]))
                    W.append(a)
            else:
                P.append((float(xs[I + ii].mean()), float(ys[J + jj].mean())))
                W.append(a * len(jj))
    P = np.array(P, float).reshape(-1, 2)
    return np.c_[P, np.full(len(P), z)], np.array(W, float)


def _seam_plan(res, variant, cell, z, seam_keys, seam_start, index):
    """The seams of the exact analysis for the surrogate: per seam its cameras (indices into the surrogate's), its
    mode (auto: the soft argmax over compute_seams' candidates; manual: fixed; var: a variable) and its sample."""
    T = _torch()
    dt = T.float64
    g = res.grid
    n = len(res.cams)
    if n < 2 or not res.seams:
        return []
    from .analysis import _azimuth
    az_cells, rad = _azimuth(g, res.centre)
    cover = res.count > 0
    azs = []
    for i, e in enumerate(res.cams):
        m = g.roi & cover & (res.best == i)
        if m.sum() > 4:
            w = e.dens[m]
            azs.append(math.atan2(float((np.sin(az_cells[m]) * w).sum()), float((np.cos(az_cells[m]) * w).sum())))
        else:
            cam = variant.cam(e.uid)
            azs.append(math.atan2(cam.y - res.centre[1], cam.x - res.centre[0]) if cam else 0.0)
    order = list(np.argsort(azs))
    rmax = float(rad[g.roi].max()) if g.roi.any() else 5.0
    radii = np.arange(0.3, rmax + cell, cell)
    out = []
    for k, sm in enumerate(res.seams):
        ia, ib = order[k], order[(k + 1) % n]
        a_uid, b_uid = res.cams[ia].uid, res.cams[ib].uid
        if a_uid != sm.a or b_uid != sm.b or a_uid not in index or b_uid not in index:
            continue
        key = f'{sm.a}|{sm.b}'
        d = {'ia': index[a_uid], 'ib': index[b_uid], 'key': key, 'phi0': math.radians(sm.angle)}
        if key in seam_keys:
            d['mode'] = 'var'
            d['var'] = seam_keys.index(key)
            d['phi0'] = math.radians(seam_start[key])
            X = res.centre[0] + math.cos(d['phi0']) * radii
            Y = res.centre[1] + math.sin(d['phi0']) * radii
            ii = np.clip(((X - g.x0) / g.cell).astype(int), 0, g.nx - 1)
            jj = np.clip(((Y - g.y0) / g.cell).astype(int), 0, g.ny - 1)
            ok = g.roi[jj, ii]
            if not ok.any():
                continue
            d['radii'] = T.tensor(radii[ok], dtype=dt)
        elif not sm.auto:
            d['mode'] = 'manual'
        else:
            d['mode'] = 'auto'
            a0 = azs[ia]
            span = (azs[ib] - a0) % (2 * math.pi)
            if span < 1e-3:
                span = 2 * math.pi / n
            cand = a0 + span * np.linspace(0.08, 0.92, 41)
            X = res.centre[0] + np.cos(cand)[:, None] * radii[None]
            Y = res.centre[1] + np.sin(cand)[:, None] * radii[None]
            ii = ((X - g.x0) / g.cell).astype(int)
            jj = ((Y - g.y0) / g.cell).astype(int)
            ok = (ii >= 0) & (ii < g.nx) & (jj >= 0) & (jj < g.ny)
            inroi = ok & g.roi[np.clip(jj, 0, g.ny - 1), np.clip(ii, 0, g.nx - 1)]
            d.update(J=len(cand), R=len(radii), cand=T.tensor(cand, dtype=dt),
                     inroi=T.tensor(inroi.astype(float), dtype=dt),
                     pull=T.tensor(0.02 * np.abs(np.linspace(-1, 1, len(cand))), dtype=dt),
                     P=T.tensor(np.stack([X.ravel(), Y.ravel(), np.full(X.size, z)], 1), dtype=dt))
        out.append(d)
    return out


# ============================================================================================ the surface
def _mesh_info(vg, cams_by_uid, units) -> dict | None:
    """The surface the cameras sit on (the mesh the mount check uses), each moving camera's start distance off it,
    the outward side and its mount zone."""
    if vg is None:
        return None
    m = vg.mesh if vg.mesh.n_tris <= 60000 else vg.occ_mesh
    mask = m.occluder_faces()
    info = {'mesh': m, 'mask': mask, 'gap': {}, 'side': {}, 'zone': {}, 'box': {}}
    for u in units:
        c = cams_by_uid[u.lead]
        p = c.pos().astype(float)
        q, fi, d = m.closest_point(p, mask)
        if fi < 0:
            continue
        n = m.normals[fi]
        side = 1.0 if (p - q) @ n >= 0 else -1.0
        info['side'][u.lead] = side
        info['gap'][u.lead] = float(d)
        zi = int(m.face_zone[fi]) if len(m.face_zone) else -1
        kind = m.zones[zi].kind if 0 <= zi < len(m.zones) else ''
        info['zone'][u.lead] = (zi, kind)
        if zi >= 0 and kind == 'allowed':
            V = m.tri[m.face_zone == zi].reshape(-1, 3)
            grow = d + 0.01
            info['box'][u.lead] = (V.min(0) - grow, V.max(0) + grow)
    return info


def _surface_gap(mi, p):
    """(distance off the surface, the gradient of that distance (unit vector)) at p; (None, None) off the mesh."""
    m = mi['mesh']
    q, fi, d = m.closest_point(p, mi['mask'])
    if fi < 0:
        return None, None
    if d > 1e-9:
        return d, (p - q) / d
    return d, m.normals[fi]


def _snap(mi, uid, p):
    """p re-projected onto the surface at the camera's start distance; (point, zone index, zone kind)."""
    m = mi['mesh']
    q, fi, d = m.closest_point(np.asarray(p, float), mi['mask'])
    if fi < 0:
        return np.asarray(p, float), -1, ''
    n = m.normals[fi] * mi['side'].get(uid, 1.0)
    zi = int(m.face_zone[fi]) if len(m.face_zone) else -1
    kind = m.zones[zi].kind if 0 <= zi < len(m.zones) else ''
    return q + n * mi['gap'][uid], zi, kind


def _zone_ok(mi, uid, zi, kind) -> bool:
    z0, k0 = mi['zone'].get(uid, (-1, ''))
    if z0 >= 0 and k0 == 'allowed':
        return zi == z0
    return kind != 'forbidden'


def poses_of(orig, units, x) -> dict:
    """{uid: pose} for the variables x (numpy): each unit's lead at its start + offsets (mm, deg), its mirror
    partner the mirror image; every other camera at its start."""
    out = dict(orig)
    x = np.asarray(x, float)
    for ui, u in enumerate(units):
        o = orig[u.lead]
        dx = x[6 * ui:6 * ui + 6]
        p = (o[0] + dx[0] / 1000, o[1] + dx[1] / 1000, o[2] + dx[2] / 1000, o[3] + dx[3], o[4] + dx[4], o[5] + dx[5])
        out[u.lead] = p
        if u.follow:
            out[u.follow] = mirror(p)
    return out


def _bounds(st: DiffSettings, units, orig, mi, n_seam) -> list:
    out = []
    for u in units:
        if u.fixed:
            out += [(0.0, 0.0)] * 6
            continue
        o = np.array(orig[u.lead][:3])
        M = st.max_move_mm if st.move else 0.0
        box = mi['box'].get(u.lead) if mi else None
        for i in range(3):
            lo, hi = -M, M
            if box is not None and M > 0:
                lo = max(lo, (box[0][i] - o[i]) * 1000)
                hi = min(hi, (box[1][i] - o[i]) * 1000)
            out.append((min(lo, 0.0), max(hi, 0.0)))
        out += [(-st.max_turn_deg, st.max_turn_deg)] * 2 + [(-st.max_roll_deg, st.max_roll_deg)]
    out += [(-st.seam_span_deg, st.seam_span_deg)] * n_seam
    return out


def _project_x(st, units, orig, mi, x, bounds):
    """x with every moving camera re-projected onto its surface (inside its mount zone, pulled back toward the start
    by halves when the projection leaves it), clipped to the bounds; and the cameras pulled back."""
    x = np.array(x, float)
    pulled = []
    if mi is None or not st.move:
        return x, pulled
    for ui, u in enumerate(units):
        if u.fixed or u.lead not in mi['gap']:
            continue
        o = np.array(orig[u.lead][:3])
        dx = x[6 * ui:6 * ui + 3].copy()
        for k in range(10):
            p, zi, kind = _snap(mi, u.lead, o + dx / 1000)
            if _zone_ok(mi, u.lead, zi, kind):
                break
            dx = dx * 0.5 if k < 9 else dx * 0.0
            if k == 0:
                pulled.append(u.lead)
        else:
            p = o
        new = (p - o) * 1000
        for i in range(3):
            lo, hi = bounds[6 * ui + i]
            new[i] = min(max(new[i], lo), hi)
        x[6 * ui:6 * ui + 3] = new
    return x, pulled


# ============================================================================================ results
@dataclass
class Refinement:
    label: str
    cams_before: list
    cams_after: list
    units: list
    exact_before: dict
    exact_after: dict
    surrogate_before: dict
    surrogate_after: dict
    seams_after: dict = field(default_factory=dict)       # 'uidA|uidB' -> deg (with seam variables)
    history: list = field(default_factory=list)           # {'iteration', 'stage', 'sharpness', 'objective'}
    checks: list = field(default_factory=list)            # the exact check at every stage boundary
    kept: int = -1                                        # index into checks of the layout kept
    stages: list = field(default_factory=list)            # per stage: iterations, evaluations, stop reason
    notes: list = field(default_factory=list)
    seconds: float = 0.0

    @property
    def improved(self) -> bool:
        return self.exact_after['objective'] < self.exact_before['objective'] - 1e-12

    def deltas(self) -> list:
        """Per camera: tag, before and after pose, the change (mm, deg), how it moves and the limits it reached."""
        after = {c.uid: c for c in self.cams_after}
        role = {}
        for u in self.units:
            role[u.lead] = 'locked' if u.fixed else 'moves'
            if u.follow:
                role[u.follow] = 'locked' if u.fixed else 'mirrors'
        out = []
        for c in self.cams_before:
            a = after.get(c.uid)
            if a is None:
                continue
            b0, a0 = pose_of(c), pose_of(a)
            d = [(a0[i] - b0[i]) * 1000 for i in range(3)] + [a0[i] - b0[i] for i in range(3, 6)]
            out.append({'uid': c.uid, 'tag': c.tag, 'before': b0, 'after': a0, 'delta': tuple(d),
                        'role': role.get(c.uid, 'hidden' if not c.visible else 'moves')})
        return out


@dataclass
class DiffResult:
    settings: DiffSettings
    variant: str
    variant_uid: str
    refinements: list
    findings: list
    seconds: float
    cancelled: bool = False
    notes: list = field(default_factory=list)


def starts_from_optimiser(opt, cands) -> list:
    """(label, cameras) of the evolutionary optimiser's candidates (its own camera_for: slid on the surface,
    switched-off cameras left out)."""
    out = []
    for c in cands:
        cams = []
        for uid, gene in c.genes.items():
            cam = opt.camera_for(uid, gene)
            if cam is not None:
                cams.append(copy.deepcopy(cam))
        out.append((c.label or 'candidate', cams))
    return out


def _cancelled(cancel) -> bool:
    if cancel is None:
        return False
    if hasattr(cancel, 'is_set'):
        return cancel.is_set()
    return bool(cancel())


def refine(project, engine, variant, cams, st: DiffSettings, label='current', progress=None, cancel=None,
           live=None) -> Refinement:
    """One start refined (see the module docstring). Cancelled, the layouts reached so far are checked and the best
    exact one kept (the start when there is none)."""
    from .critical import settings_of as crit_settings
    from scipy.optimize import minimize
    t0 = time.perf_counter()
    crit_targets = [t for t in crit_settings(project).targets if t['enabled']]
    objects = _objects(project, crit_targets)
    cams = [copy.deepcopy(c) for c in cams]
    units, notes = units_of(cams)
    by = {c.uid: c for c in cams}
    orig = {c.uid: pose_of(c) for c in cams}
    base_v = _variant_with(variant, cams)
    vg = engine.vehicle_geom(project, base_v)
    mi = _mesh_info(vg, by, units)
    ex0 = exact_check(project, engine, variant, cams, st, crit_targets, objects)
    seam_keys, seam_start = [], {}
    if st.seams:
        if objects:
            seam_keys = [f'{s["a"]}|{s["b"]}' for s in ex0['seams']]
            seam_start = {f'{s["a"]}|{s["b"]}': s['angle'] for s in ex0['seams']}
        else:
            notes.append('No standing critical target or obstacle: the seams stay automatic.')
    n_unit = 6 * len(units)
    x = np.zeros(n_unit + len(seam_keys))
    bounds = _bounds(st, units, orig, mi, len(seam_keys))
    sharp = st.sharpness()

    def layout(xv):
        ps = poses_of(orig, units, xv)
        out = []
        for c in cams:
            cc = copy.deepcopy(c)
            _set_pose(cc, ps[c.uid])
            out.append(cc)
        seams = {k: _wrap(seam_start[k] + float(xv[n_unit + i])) for i, k in enumerate(seam_keys)}
        return out, seams

    hist, checks, stages = [], [], []
    it = [0]
    ex = ex0
    for k, s in enumerate(sharp):
        if _cancelled(cancel):
            break
        lay, seams = layout(x)
        if k > 0:
            ex = exact_check(project, engine, variant, lay, st, crit_targets, objects, seams or None)
        checks.append(_check_row(it[0], k, ex, x))
        if live:
            live(label, hist, checks)
        sur = Surrogate(project, engine, base_v, orig, units, x, s, st, ex, crit_targets, objects, seam_keys,
                        seam_start, mi)
        prev = [x.copy()]
        small = [False]
        stop = ['iterations']

        def cb(intermediate_result):
            xk = np.asarray(intermediate_result.x, float)
            it[0] += 1
            hist.append({'iteration': it[0], 'stage': k, 'sharpness': s, 'objective': float(intermediate_result.fun)})
            d = np.abs(xk - prev[0])
            prev[0] = xk.copy()
            if live and it[0] % 2 == 0:
                live(label, hist, checks)
            if progress:
                progress((k + min(1.0, (it[0] - stages_it0) / max(st.iterations, 1))) / len(sharp))
            if _cancelled(cancel):
                stop[0] = 'cancelled'
                raise StopIteration
            ok = True
            for ui in range(len(units)):
                dd = d[6 * ui:6 * ui + 6]
                if (dd[:3] > STEP_MM).any() or (dd[3:] > STEP_DEG).any():
                    ok = False
                    break
            ok = ok and bool((d[n_unit:] <= STEP_DEG).all())
            # two steps in a row within the tolerance, not the first (L-BFGS-B's first step after a restart is
            # scaled down and small whatever the slope)
            if ok and small[0] and it[0] - stages_it0 >= 3:
                stop[0] = 'step tolerance'
                raise StopIteration
            small[0] = ok
        stages_it0 = it[0]
        f0, _ = sur.value_and_grad(x)
        hist.append({'iteration': it[0], 'stage': k, 'sharpness': s, 'objective': float(f0), 'start': True})
        r = minimize(sur.value_and_grad, x, jac=True, method='L-BFGS-B', bounds=bounds, callback=cb,
                     options={'maxiter': int(st.iterations), 'ftol': 1e-15, 'gtol': 1e-12, 'maxcor': 10})
        if stop[0] == 'iterations' and r.status == 0:
            stop[0] = 'converged'
        x, pulled = _project_x(st, units, orig, mi, r.x, bounds)
        for uid in pulled:
            n_ = f'{by[uid].tag}: a re-projection onto the surface left its mount zone; pulled back toward the start.'
            if n_ not in notes:
                notes.append(n_)
        stages.append({'stage': k, 'sharpness': s, 'iterations': it[0] - stages_it0, 'evaluations': sur.n_eval,
                       'stop': stop[0], 'objective': float(r.fun)})
        if stop[0] == 'cancelled':
            break
    lay, seams = layout(x)
    ex = exact_check(project, engine, variant, lay, st, crit_targets, objects, seams or None)
    checks.append(_check_row(it[0], len(stages), ex, x))
    if live:
        live(label, hist, checks)
    best = min(range(len(checks)), key=lambda i: (checks[i]['objective'], -i))
    xb = checks[best]['x']
    lay_b, seams_b = layout(xb)
    exb = checks[best]['exact']
    sharpest = sharp[-1]
    sur0 = Surrogate(project, engine, base_v, orig, units, np.zeros_like(x), sharpest, st, ex0, crit_targets,
                     objects, seam_keys, seam_start, mi).parts(np.zeros_like(x))
    sur1 = Surrogate(project, engine, base_v, orig, units, xb, sharpest, st, exb, crit_targets, objects, seam_keys,
                     seam_start, mi).parts(xb) if best else dict(sur0)
    if 0 < best < len(checks) - 1:
        notes.append(f'The exact check was best after stage {checks[best]["stage"]}; that layout is kept.')
    ref = Refinement(label, cams, lay_b, units, _strip(ex0), _strip(exb), sur0, sur1, seams_b, hist,
                     [_strip_check(c) for c in checks], best, stages, notes, time.perf_counter() - t0)
    return ref


def _wrap(a):
    return (a + 180.0) % 360.0 - 180.0


def _strip(ex) -> dict:
    return {k: v for k, v in ex.items() if k != 'result'}


def _check_row(iteration, stage, ex, x) -> dict:
    return {'iteration': iteration, 'stage': stage, 'objective': ex['objective'], 'blind_m2': ex['blind_m2'],
            'low_m2': ex['low_m2'], 'unresolved': ex['unresolved'], 'seam_near': ex['seam_near'],
            'x': np.array(x, float), 'exact': ex}


def _strip_check(c) -> dict:
    return {k: v for k, v in c.items() if k not in ('x', 'exact')}


def run(project, engine=None, settings: DiffSettings | None = None, variant=None, progress=None, cancel=None,
        starts=None, live=None) -> DiffResult:
    """Refine each start (default: the variant's cameras; starts: [(label, cameras)], e.g. starts_from_optimiser)
    and check every result with the exact engine. live(label, history, checks) is called as the solver goes."""
    from .analysis import Engine
    t0 = time.perf_counter()
    st = settings or settings_of(project)
    engine = engine or Engine()
    variant = variant or project.variant
    if getattr(project.vehicle.mesh, 'n_tris', 0) == 0 or not any(c.visible for c in variant.cameras):
        raise ValueError(f'Nothing to refine in {variant.name}: it needs a vehicle body and a visible camera.')
    if torch_missing():
        raise ValueError(torch_missing())
    starts = list(starts) if starts else [('current', list(variant.cameras))]
    starts = starts[:MAX_STARTS]
    refs, cancelled = [], False
    with _one_thread():
        refs, cancelled = _run_starts(project, engine, st, variant, starts, progress, cancel, live)
    res = DiffResult(st, variant.name, variant.uid, refs, [], time.perf_counter() - t0, cancelled)
    res.findings = _findings(res, project)
    if progress:
        progress(1.0)
    return res


def _run_starts(project, engine, st, variant, starts, progress, cancel, live):
    refs, cancelled = [], False
    for i, (label, cams) in enumerate(starts):
        if _cancelled(cancel):
            cancelled = True
            break

        def prog(f, i=i):
            if progress:
                progress((i + f) / len(starts))
        ref = refine(project, engine, variant, cams, st, label, prog, cancel, live)
        if ref is not None:
            refs.append(ref)
        if any(s['stop'] == 'cancelled' for s in (ref.stages if ref else [])):
            cancelled = True
            break
    return refs, cancelled


# ============================================================================================ findings
def _f(sev, kind, message, uids=()):
    return {'severity': sev, 'kind': kind, 'message': message, 'uids': list(uids)}


def _findings(res: DiffResult, project) -> list:
    out = []
    mind = project.analysis.min_density
    st = res.settings
    if res.cancelled:
        out.append(_f('note', 'cancelled', 'The refinement was cancelled; the layouts below are as far as it got.'))
    for r in res.refinements:
        b, a = r.exact_before, r.exact_after
        moved = [d for d in r.deltas() if any(abs(v) > 0.05 for v in d['delta'])]
        uids = [d['uid'] for d in moved]
        if r.improved:
            bits = [f'blind ground {b["blind_m2"]:.3f} → {a["blind_m2"]:.3f} m²',
                    f'ground below {mind:g} px/cm {b["low_m2"]:.3f} → {a["low_m2"]:.3f} m²']
            if b['unresolved'] or a['unresolved']:
                bits.append(f'unresolved critical targets {b["unresolved"]} → {a["unresolved"]}')
            if b['seam_near'] or a['seam_near']:
                bits.append(f'objects on a seam {b["seam_near"]} → {a["seam_near"]}')
            out.append(_f('good', 'improved', f'{r.label}: the exact check improves: ' + ', '.join(bits) +
                          f' (objective {b["objective"]:.4f} → {a["objective"]:.4f}).', uids))
        else:
            out.append(_f('note', 'kept', f'{r.label}: no refined layout beats the start in the exact check '
                                          f'(objective {b["objective"]:.4f}); the start is kept.'))
        sb, sa = r.surrogate_before, r.surrogate_after
        gap_b = sb['blind_m2'] - b['blind_m2']
        gap_a = sa['blind_m2'] - a['blind_m2']
        sev = 'warning' if max(abs(gap_b), abs(gap_a)) > max(0.1, 0.5 * max(b['blind_m2'], a['blind_m2'])) else 'note'
        out.append(_f(sev, 'surrogate', f'{r.label}: the surrogate puts the blind ground at {sb["blind_m2"]:.3f} m² '
                                        f'before and {sa["blind_m2"]:.3f} m² after, the exact engine at '
                                        f'{b["blind_m2"]:.3f} and {a["blind_m2"]:.3f} m²; below the minimum density '
                                        f'{sb["low_m2"]:.3f} / {sa["low_m2"]:.3f} against {b["low_m2"]:.3f} / '
                                        f'{a["low_m2"]:.3f} m².'))
        lim = []
        for d in moved:
            if d['role'] != 'moves':
                continue
            dd = d['delta']
            hit = [n for n, v, m in (('yaw', dd[3], st.max_turn_deg), ('pitch', dd[4], st.max_turn_deg),
                                     ('roll', dd[5], st.max_roll_deg)) if m > 0 and abs(v) >= m - 0.05]
            hit += [n for n, v in zip('xyz', dd[:3]) if st.max_move_mm > 0 and abs(v) >= st.max_move_mm - 0.5]
            if hit:
                lim.append(f'{d["tag"]} ({", ".join(hit)})')
        if lim and r.improved:
            out.append(_f('note', 'limit', f'{r.label}: at a move limit: {"; ".join(lim)}. A wider limit may '
                                           f'help; check the mount first.'))
        for n in r.notes:
            out.append(_f('note', 'note', f'{r.label}: {n}'))
    return out


# ============================================================================================ variants
def to_variant(ref: Refinement, variant, name: str):
    """A new variant: a copy of variant (sensors, blend, seams) with the refined cameras (new uids; links and manual
    seams remapped)."""
    from .project import new_uid
    v = copy.deepcopy(variant)
    v.uid = new_uid()
    v.name = name
    a = ref.exact_after
    v.note = (f'Gradient refinement of {ref.label}: exact blind {ref.exact_before["blind_m2"]:.3f} → '
              f'{a["blind_m2"]:.3f} m², below minimum {ref.exact_before["low_m2"]:.3f} → {a["low_m2"]:.3f} m²')
    cams = [copy.deepcopy(c) for c in ref.cams_after]
    for c in variant.cameras:                         # hidden cameras of the variant the start left out
        if not c.visible and all(x.uid != c.uid for x in cams):
            cams.append(copy.deepcopy(c))
    m = {}
    for c in cams:
        m[c.uid] = new_uid()
    seams = dict(variant.seams)
    seams.update(ref.seams_after)
    v.seams = {}
    for k, ang in seams.items():
        a_, _, b_ = k.partition('|')
        if a_ in m and b_ in m:
            v.seams[f'{m[a_]}|{m[b_]}'] = round(float(ang), 2)
    for c in cams:
        c.uid = m[c.uid]
        c.link_uid = m.get(c.link_uid, '')
    v.cameras = cams
    return v


# ============================================================================================ export
LIMITS_TEXT = [
    'The surrogate is smooth by construction (sigmoids of the margins, τ annealed); its figures at the sharpest τ '
    'are reported beside the exact engine\'s and the exact ones decide what is kept.',
    'Cube maps belong to the stage\'s start positions: within a stage the parallax of occluders as a camera moves '
    'is ignored; every stage rebuilds them.',
    'Flat ground at the analysis\'s obstacle height in the surrogate; a road (analysis.terrain) only in the exact '
    'check.',
    'Critical targets are seen through the cube maps (0.35° texels) in the surrogate and by exact rays in the check; '
    'lidar points on them are not counted (the cameras\' requirement only, as the evolutionary optimiser).',
    'A camera stays on its surface by a penalty during a stage and is re-projected onto the mesh after it; the '
    'mount zone bounds the move.',
    'The step tolerance 0.01° / 0.1 mm is when the solver stops, not the accuracy of the pose in the world.',
    'Local: a nearby optimum of the surrogate. Which cameras and mounts are used is the evolutionary search\'s job.',
    'The τ values and weights are heuristics, chosen on the sample.']


def _r(v, nd=4):
    return None if v is None else round(float(v), nd)


def to_doc(res: DiffResult, project=None) -> dict:
    st = res.settings
    refs = []
    for r in res.refinements:
        refs.append({
            'label': r.label, 'improved': r.improved, 'seconds': _r(r.seconds, 2),
            'exact': {'before': _ex_doc(r.exact_before), 'after': _ex_doc(r.exact_after)},
            'surrogate': {'before': {k: _r(v) for k, v in r.surrogate_before.items()},
                          'after': {k: _r(v) for k, v in r.surrogate_after.items()}},
            'cameras': [{'uid': d['uid'], 'tag': d['tag'], 'role': d['role'],
                         'before': dict(zip(POSE, (_r(v, 5) for v in d['before']))),
                         'after': dict(zip(POSE, (_r(v, 5) for v in d['after']))),
                         'delta': dict(zip(('x_mm', 'y_mm', 'z_mm', 'yaw_deg', 'pitch_deg', 'roll_deg'),
                                           (_r(v, 3) for v in d['delta'])))} for d in r.deltas()],
            'seams_after': {k: _r(v, 2) for k, v in r.seams_after.items()},
            'stages': [{k: (_r(v, 6) if isinstance(v, float) else v) for k, v in s.items()} for s in r.stages],
            'history': [{k: (_r(v, 6) if isinstance(v, float) else v) for k, v in h.items()} for h in r.history],
            'checks': [{k: (_r(v, 6) if isinstance(v, float) else v) for k, v in c.items()} for c in r.checks],
            'kept_check': r.kept, 'notes': list(r.notes)})
    return {
        'schema': SCHEMA, 'project': getattr(project, 'name', ''), 'variant': res.variant,
        'variant_uid': res.variant_uid, 'frame': FRAME,
        'units': {'position': 'm (deltas in mm)', 'angle': 'deg (yaw, pitch, roll intrinsic Z-Y-X, pitch positive '
                  'down)', 'area': 'm²', 'density': 'px/cm', 'objective': 'weighted sum (the weights\' units)'},
        'settings': st.to_dict(), 'sharpness': st.sharpness(),
        'softness': {'theta_rad': TAU_THETA, 'image_px': TAU_PX, 'occluder_m': TAU_OCC, 'direction_deg': TAU_DIR,
                     'density_ln': TAU_DENS, 'share': TAU_SHARE, 'target_px': TAU_TPX, 'focus_m': TAU_FOCUS,
                     'seam_score': TAU_SEAM, 'note': 'at the first stage; every stage multiplies them by its '
                                                     'sharpness. Heuristics.'},
        'solver': {'method': 'L-BFGS-B (scipy)', 'gradient': 'torch autograd, CPU, float64',
                   'step_tolerance': {'mm': STEP_MM, 'deg': STEP_DEG},
                   'note': 'the solver\'s step tolerance, not the accuracy of the pose in the world'},
        'surface': 'penalised during a stage (1 per 5 mm off the start distance from the surface, squared), '
                   're-projected onto the mesh after every stage, kept inside the mount zone',
        'limits': LIMITS_TEXT, 'cancelled': res.cancelled, 'seconds': _r(res.seconds, 2),
        'refinements': refs,
        'findings': [{'severity': f['severity'], 'kind': f['kind'], 'message': f['message']} for f in res.findings]}


def _ex_doc(e) -> dict:
    out = {k: (_r(v, 5) if isinstance(v, float) else v) for k, v in e.items() if k not in ('seams', 'near', 'missing')}
    out['seams_deg'] = [_r(s['angle'], 2) for s in e.get('seams', [])]
    out['near_seam'] = list(e.get('near', []))
    out['unresolved_targets'] = list(e.get('missing', []))
    return out


CAMERA_COLUMNS = ('start', 'camera', 'role', 'x', 'y', 'z', 'yaw', 'pitch', 'roll', 'dx_mm', 'dy_mm', 'dz_mm',
                  'dyaw_deg', 'dpitch_deg', 'droll_deg')


def cameras_csv(res: DiffResult) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator='\n')
    w.writerow(CAMERA_COLUMNS)
    for r in res.refinements:
        for d in r.deltas():
            w.writerow([r.label, d['tag'], d['role']] + [_r(v, 5) for v in d['after']] + [_r(v, 3) for v in d['delta']])
    return buf.getvalue()


def history_csv(res: DiffResult) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator='\n')
    w.writerow(('start', 'iteration', 'stage', 'sharpness', 'surrogate_objective', 'exact_objective',
                'exact_blind_m2', 'exact_low_m2'))
    for r in res.refinements:
        for h in r.history:
            w.writerow([r.label, h['iteration'], h['stage'], _r(h['sharpness'], 4), _r(h['objective'], 6), '', '', ''])
        for c in r.checks:
            w.writerow([r.label, c['iteration'], c['stage'], '', '', _r(c['objective'], 6), _r(c['blind_m2'], 4),
                        _r(c['low_m2'], 4)])
    return buf.getvalue()


def files(res: DiffResult, project=None) -> list:
    doc = to_doc(res, project)
    return [('diff_optim.json', (json.dumps(doc, indent=1, ensure_ascii=False) + '\n').encode('utf-8')),
            ('diff_optim_cameras.csv', cameras_csv(res).encode('utf-8')),
            ('diff_optim_history.csv', history_csv(res).encode('utf-8'))]


# ============================================================================================ batch
def batch(project, engine=None, from_variant: str | None = None, settings: DiffSettings | None = None):
    """For --refine-rig [--refine-from VARIANT]: (doc, exit code). 4 when the variant is not in the project."""
    variant = project.variant
    if from_variant:
        variant = next((v for v in project.variants if v.name == from_variant or v.uid == from_variant), None)
        if variant is None:
            return {'schema': SCHEMA, 'error': f'no variant named {from_variant!r}'}, 4
    if torch_missing():
        return {'schema': SCHEMA, 'error': torch_missing()}, 4
    res = run(project, engine, settings, variant)
    return to_doc(res, project), 0


# ============================================================================================ gradient check
def surrogate_at(project, engine=None, variant=None, settings: DiffSettings | None = None, sharp=1.0, x=None):
    """(surrogate, x) of the variant's current layout at sharpness `sharp` (for checks and tests)."""
    from .analysis import Engine
    from .critical import settings_of as crit_settings
    engine = engine or Engine()
    st = settings or settings_of(project)
    variant = variant or project.variant
    cams = [copy.deepcopy(c) for c in variant.cameras]
    units, _ = units_of(cams)
    orig = {c.uid: pose_of(c) for c in cams}
    crit_targets = [t for t in crit_settings(project).targets if t['enabled']]
    objects = _objects(project, crit_targets)
    ex = exact_check(project, engine, variant, cams, st, crit_targets, objects)
    seam_keys, seam_start = [], {}
    if st.seams and objects:
        seam_keys = [f'{s["a"]}|{s["b"]}' for s in ex['seams']]
        seam_start = {f'{s["a"]}|{s["b"]}': s['angle'] for s in ex['seams']}
    mi = _mesh_info(engine.vehicle_geom(project, _variant_with(variant, cams)), {c.uid: c for c in cams}, units)
    x = np.zeros(6 * len(units) + len(seam_keys)) if x is None else np.asarray(x, float)
    sur = Surrogate(project, engine, _variant_with(variant, cams), orig, units, x, sharp, st, ex, crit_targets,
                    objects, seam_keys, seam_start, mi)
    return sur, x
