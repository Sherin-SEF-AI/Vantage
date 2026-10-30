"""Placement optimiser: evolutionary search over mount offsets and aim, scored under tolerance.

Critical targets (core.critical, Analysis ▸ Critical targets) can join the search: 'require' keeps every candidate
that leaves a target unresolved off the Pareto front (as a cable over its limit is), 'weigh' adds critical_weight per
unresolved target to the candidate's objective (in the objective's own unit: m² of blind area, cameras, px/cm). Under
tolerance a target counts as resolved only when it stays resolved in every tolerance sample (the line of sight of the
planned position is kept for the samples, as the coverage score keeps the planned position's occlusion map). The
lidars do not move in the search, so only the cameras' requirement is checked.

Targets as constraints (need_density, need_blind; the assistant's optimise_rig states them): a candidate that misses
one ranks behind every candidate that meets them and stays off the Pareto front while any candidate meets them; the
blind limit is checked against the worst blind area under tolerance when the search scores under tolerance, the
density against the nominal pose. With units on, cameras are switched on and off as units (a mirrored pair together),
never a kept camera, and a hidden camera stays off; fewer cameras are tried whenever count allows them, whatever the
objective. With tol_seed every candidate is scored under the same tolerance draws (common random numbers), so two
candidates differ by their poses and not by their luck. Every one of these is off by default, so the Optimise page
searches exactly as before."""
from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass, field

import numpy as np

from .analysis import Engine, eval_points, slide_on_surface
from .lens import resolve
from .mathutil import rot_zyx
from .project import Project, Variant, Camera, new_uid
from .rules import route_length


@dataclass
class FreeVar:
    uid: str
    tag: str
    axis: str = 'x'          # tangent move axis: 'x' or 'y' ('' = fixed position)
    move: float = 0.10       # ± metres along the surface
    yaw: tuple = (0.0, 0.0)  # absolute range (deg); equal values = fixed
    pitch: tuple = (30.0, 50.0)
    linked_to: str = ''      # mirrors this camera
    removable: bool = False
    keep: bool = False       # never switched off by a units search (a locked camera, one the search must leave)
    step: float = 0.0        # slide quantum (m); 0 = 1 cm. Each slide position costs an occlusion map: coarser is faster

    def describe(self):
        if self.linked_to:
            return 'linked'
        parts = []
        if self.axis and self.move:
            parts.append(f'±{self.move:.2f}')
        if self.yaw[0] != self.yaw[1]:
            parts.append(f'{self.yaw[0]:.0f}–{self.yaw[1]:.0f}°')
        if self.pitch[0] != self.pitch[1]:
            parts.append(f'{self.pitch[0]:.0f}–{self.pitch[1]:.0f}°')
        return ' · '.join(parts) or 'fixed'


@dataclass
class OptSettings:
    objective: str = 'blind'          # blind | cameras | density
    under_tolerance: bool = True
    tol_deg: float = 5.0
    tol_mm: float = 10.0
    tol_samples: int = 24
    zones: str = 'allowed'            # allowed | any
    height: tuple = (0.45, 1.05)
    cable_max: float = 3.0
    count: tuple = (4, 6)
    module: str = ''                  # '' = keep
    generations: int = 60
    population: int = 60
    cell: float = 0.10
    seed: int = 7
    vars: list = field(default_factory=list)
    critical: str = 'off'             # off | require (every critical target resolved) | weigh
    critical_weight: float = 1.0      # 'weigh': added to the objective per unresolved target
    need_density: float = 0.0         # px/cm every seen ground cell must keep (0 = no density target)
    need_blind: float = -1.0          # m² of blind ground at most (worst under tolerance when scored so; < 0 = none)
    units: bool = False               # switch cameras as units: mirrored pairs together, kept ones never, hidden off
    tol_seed: int = -1                # >= 0: every candidate scored with the same tolerance draws (from this seed)


CRITICAL_MODES = ('off', 'require', 'weigh')


@dataclass
class Candidate:
    genes: dict                      # uid -> [s, yaw, pitch, enabled]
    blind: float = math.inf
    worst: float = math.inf
    min_density: float = 0.0
    cams: int = 0
    cable_max: float = 0.0
    cable_ok: bool = True
    positions: dict = field(default_factory=dict)   # uid -> (x, y, z)
    rank: int = 0
    label: str = ''
    crit_unmet: int = 0              # critical targets no camera resolves at the candidate's pose
    crit_tol: int = 0                # ... in its worst tolerance sample (= crit_unmet without the tolerance score)
    crit_missing: list = field(default_factory=list)   # their names (at the candidate's pose)


def default_vars(project: Project, variant: Variant) -> list[FreeVar]:
    out = []
    seen = set()
    for c in variant.cameras:
        if c.uid in seen:
            continue
        axis = 'y' if abs(c.yaw) < 30 or abs(abs(c.yaw) - 180) < 30 else 'x'
        if c.tag in ('F', 'R'):
            fv = FreeVar(c.uid, c.tag, axis='y', move=0.10, yaw=(c.yaw, c.yaw), pitch=(30, 50))
        else:
            fv = FreeVar(c.uid, c.tag, axis=axis, move=0.30 if c.tag.startswith('Q') else 0.0,
                         yaw=(abs(c.yaw) - 15, abs(c.yaw) + 15) if c.yaw >= 0 else (-abs(c.yaw) - 15, -abs(c.yaw) + 15),
                         pitch=(max(25, c.pitch - 10), min(70, c.pitch + 10)))
            if c.tag.startswith('Q'):
                fv.yaw = (80, 120) if c.yaw > 0 else (-120, -80)
        out.append(fv)
        if c.link_uid:
            p = variant.cam(c.link_uid)
            if p:
                out.append(FreeVar(p.uid, p.tag, linked_to=c.uid))
                seen.add(p.uid)
        seen.add(c.uid)
    return out


class Optimiser:
    def __init__(self, engine: Engine, project: Project, st: OptSettings, variant: Variant | None = None):
        self.engine = engine
        self.project = project
        self.st = st
        self.variant = variant or project.variant
        self.base = {c.uid: copy.deepcopy(c) for c in self.variant.cameras}
        self.vars = {v.uid: v for v in (st.vars or default_vars(project, self.variant))}
        self.vg = engine.vehicle_geom(project)
        self.grid = engine.roi_grid(project, st.cell)
        self.P = self.grid.points(0.0)[self.grid.roi]
        self.a = self.grid.cell ** 2
        self.rng = np.random.default_rng(st.seed)
        self.evaluated: list[Candidate] = []
        self.front: list[Candidate] = []
        self.generation = 0
        self.evals = 0
        self.t0 = time.perf_counter()
        self._surface_cache = {}
        self._units, self._leaders, self._toggle = [], set(), False
        if st.units:
            for uid, fv in self.vars.items():
                if fv.linked_to or fv.keep or uid not in self.base or not self.base[uid].visible:
                    continue
                members = (uid,) + tuple(u for u, f in self.vars.items() if f.linked_to == uid and u in self.base)
                if not any(self.vars[m].keep for m in members):
                    self._units.append(members)
                    self._leaders.add(uid)
            on = sum(1 for c in self.base.values() if c.visible)
            self._toggle = bool(self._units) and st.count[0] < on
        self.crit = None
        if st.critical in ('require', 'weigh'):
            from .critical import TargetCheck
            tc = TargetCheck(project, self.vg, self.variant)
            self.crit = tc if tc else None

    # ---------------------------------------------------------- genes
    def nominal(self) -> Candidate:
        g = {}
        for uid, c in self.base.items():
            g[uid] = [0.0, c.yaw, c.pitch, 1.0 if c.visible or not self.st.units else 0.0]
        return Candidate(g, label='current')

    def mutate(self, cand: Candidate, scale=1.0) -> Candidate:
        g = {u: list(v) for u, v in cand.genes.items()}
        for uid, fv in self.vars.items():
            if fv.linked_to or uid not in g:
                continue
            s, yaw, pitch, en = g[uid]
            if fv.axis and fv.move:
                if fv.step <= 0:
                    s = round(float(np.clip(s + self.rng.normal(0, 0.35 * fv.move * scale), -fv.move, fv.move)), 2)
                else:                                   # on the slide quantum (same draw as above)
                    q = round(float(s + self.rng.normal(0, 0.35 * fv.move * scale)) / fv.step) * fv.step
                    s = float(np.clip(round(q, 4), -fv.move, fv.move))
            if fv.yaw[1] > fv.yaw[0]:
                yaw = float(np.clip(yaw + self.rng.normal(0, 4 * scale), fv.yaw[0], fv.yaw[1]))
            if fv.pitch[1] > fv.pitch[0]:
                pitch = float(np.clip(pitch + self.rng.normal(0, 3 * scale), fv.pitch[0], fv.pitch[1]))
            if (self.st.objective == 'cameras' or self._toggle) and self.rng.random() < 0.08 * scale \
                    and (not self.st.units or uid in self._leaders):
                en = 1.0 - en
            g[uid] = [s, yaw, pitch, en]
        self._apply_links(g)
        self._enforce_count(g)
        return Candidate(g)

    def _apply_links(self, g):
        for uid, fv in self.vars.items():
            if fv.linked_to and fv.linked_to in g:
                s, yaw, pitch, en = g[fv.linked_to]
                g[uid] = [s, -yaw, pitch, en]

    def _enforce_count(self, g):
        if self.st.units:
            return self._enforce_units(g)
        lo, hi = self.st.count
        on = [u for u, v in g.items() if v[3] > 0.5]
        while len(on) < lo:
            off = [u for u, v in g.items() if v[3] <= 0.5]
            if not off:
                break
            g[off[0]][3] = 1.0
            on.append(off[0])
        while len(on) > hi:
            g[on.pop()][3] = 0.0

    def _enforce_units(self, g):
        """The camera count within st.count by switching whole units (a mirrored pair together; see units): one that
        lands inside the range when there is one, else any, picked at random so no unit is always the one dropped."""
        lo, hi = self.st.count
        for _ in range(len(self._units) + 1):
            n = sum(1 for v in g.values() if v[3] > 0.5)
            if lo <= n <= hi:
                return
            add = n < lo
            pool = [u for u in self._units if (g[u[0]][3] <= 0.5) == add]
            fit = [u for u in pool if lo <= n + (len(u) if add else -len(u)) <= hi] or pool
            if not fit:
                return
            for uid in fit[int(self.rng.integers(len(fit)))]:
                g[uid][3] = 1.0 if add else 0.0

    # ---------------------------------------------------------- geometry
    def camera_for(self, uid, gene) -> Camera | None:
        base = self.base[uid]
        s, yaw, pitch, en = gene
        if en <= 0.5:
            return None
        c = copy.deepcopy(base)
        c.yaw, c.pitch = yaw, pitch
        fv = self.vars.get(uid)
        if fv and fv.linked_to:
            fv = self.vars.get(fv.linked_to)
        if fv and fv.axis and abs(s) > 1e-6:
            key = (uid, round(s, 3))
            if key not in self._surface_cache:
                self._surface_cache[key] = self._slide(base, fv.axis, s)
            c.x, c.y, c.z = self._surface_cache[key]
        return c

    def _slide(self, cam: Camera, axis, s):
        step = np.array([1.0, 0, 0]) if axis == 'x' else np.array([0, 1.0, 0])
        if cam.y < 0 and axis == 'y':
            step = -step
        return tuple(np.round(slide_on_surface(self.vg.occ_mesh, cam, step, s), 4))

    # ---------------------------------------------------------- scoring
    def evaluate(self, cand: Candidate):
        cams = []
        for uid, gene in cand.genes.items():
            c = self.camera_for(uid, gene)
            if c is not None:
                cams.append(c)
        cand.cams = len(cams)
        cand.positions = {c.uid: (c.x, c.y, c.z) for c in cams}
        if not cams:
            cand.blind = cand.worst = 1e9
            if self.crit is not None:                   # nothing resolves any target
                cand.crit_unmet = cand.crit_tol = len(self.crit.targets)
                cand.crit_missing = [t['name'] for t in self.crit.targets]
            return cand
        data = []
        for c in cams:
            intr = resolve(self.project.module_for(c), self.project.lens_for(c))
            cube = self.engine.cube(c.pos())
            data.append((c, intr, cube))
        nominal = [eval_points(self.P, c, intr, cube)[:2] for c, intr, cube in data]
        V = np.stack([v for v, _ in nominal])
        D = np.stack([d for _, d in nominal])
        blind = float((V.sum(0) == 0).sum() * self.a)
        cov = V.any(0)
        cand.blind = blind
        cand.min_density = float(D.max(0)[cov].min()) if cov.any() else 0.0
        worst = blind
        if self.crit is not None:
            met = self.crit.met(cams)
            cand.crit_unmet = cand.crit_tol = int((~met).sum())
            cand.crit_missing = self.crit.names(met)
        if self.st.under_tolerance and self.st.tol_samples > 0:
            common = self._common_draws()
            for k in range(self.st.tol_samples):
                Vs, Ds = [], []
                poses = {}
                for (c, intr, cube), (v0, d0) in zip(data, nominal):
                    dv = self.rng.uniform(-1, 1, 5) if common is None else common[k, self._index[c.uid]]
                    R = rot_zyx(c.yaw + dv[0] * self.st.tol_deg, c.pitch + dv[1] * self.st.tol_deg, c.roll)
                    pos = c.pos() + np.r_[dv[2:5]] * self.st.tol_mm / 1000
                    v, d, _, _ = eval_points(self.P, c, intr, cube, R_override=R, pos_override=pos)
                    Vs.append(v)
                    poses[c.uid] = (R, pos)
                b = float((np.stack(Vs).sum(0) == 0).sum() * self.a)
                worst = max(worst, b)
                if self.crit is not None and cand.crit_tol < len(self.crit.targets):
                    cand.crit_tol = max(cand.crit_tol, int((~self.crit.met(cams, poses)).sum()))
        cand.worst = worst
        cables = [route_length(self.project, c, routed=False) for c in cams]     # the estimate: fast in the loop
        cand.cable_max = max(cables)
        cand.cable_ok = all(r <= max(c.cable_m, self.st.cable_max) + 1e-6 for r, c in zip(cables, cams))
        z_ok = all(self.st.height[0] - 1e-6 <= c.z <= self.st.height[1] + 1e-6 for c in cams)
        if not z_ok:
            cand.worst += 10.0
        self.evals += 1
        return cand

    def _common_draws(self):
        """With tol_seed: one mounting error per tolerance sample and camera, the same for every candidate (a camera
        keeps its own draws when others are switched off); None: each candidate draws afresh."""
        if self.st.tol_seed < 0:
            return None
        key = (self.st.tol_seed, self.st.tol_samples)
        if getattr(self, '_draws_key', None) != key:
            self._index = {u: i for i, u in enumerate(self.base)}
            self._draws = np.random.default_rng(self.st.tol_seed).uniform(-1, 1, (self.st.tol_samples,
                                                                                   len(self.base), 5))
            self._draws_key = key
        return self._draws

    def crit_penalty(self, c: Candidate) -> int:
        """Critical targets the candidate leaves unresolved (in its worst tolerance sample when scored under
        tolerance); 0 without critical targets in the search."""
        if self.crit is None:
            return 0
        return c.crit_tol if self.st.under_tolerance else c.crit_unmet

    def has_targets(self) -> bool:
        return self.st.need_density > 0 or self.st.need_blind >= 0

    def shortfall(self, c: Candidate) -> float:
        """How far the candidate misses the stated targets (0: it meets them, or there are none): the density it lacks
        as a fraction of the target plus the blind area over the limit in m²."""
        s = 0.0
        if self.st.need_density > 0:
            s += max(0.0, self.st.need_density - c.min_density) / self.st.need_density
        if self.st.need_blind >= 0:
            s += max(0.0, c.worst - self.st.need_blind)
        return round(s, 4)

    def _blind_ok(self, c: Candidate, default: float) -> bool:
        """Blind enough for the cameras / density objectives: within the stated limit, else under their default."""
        return c.worst <= self.st.need_blind + 1e-9 if self.st.need_blind >= 0 else c.worst < default

    def objectives(self, c: Candidate):
        pen = self.crit_penalty(c)
        w = self.st.critical_weight * pen if self.st.critical == 'weigh' else 0.0
        if self.st.objective == 'cameras':
            out = (c.cams + (0 if self._blind_ok(c, 0.01) else 10) + w, c.worst, -c.min_density)
        elif self.st.objective == 'density':
            out = (-c.min_density + (0 if self._blind_ok(c, 0.05) else 10) + w, c.worst, c.cams)
        else:
            out = (c.worst + w, -c.min_density, c.cams)
        if self.has_targets():
            out = (self.shortfall(c),) + out
        return (pen,) + out if self.st.critical == 'require' else out

    def feasible(self, c: Candidate) -> bool:
        """Within the cable limit, with critical targets required resolving all of them, and meeting the stated
        targets."""
        return c.cable_ok and not (self.st.critical == 'require' and self.crit_penalty(c) > 0) \
            and not (self.has_targets() and self.shortfall(c) > 0)

    def _front(self, cands):
        """The Pareto front of the feasible candidates; without any, of those closest to feasible (cable within
        its limit where possible, then the fewest unresolved targets)."""
        ok = [c for c in cands if self.feasible(c)]
        if not ok and cands:
            pool = [c for c in cands if c.cable_ok] or list(cands)
            least = min(self.crit_penalty(c) for c in pool)
            ok = [c for c in pool if self.crit_penalty(c) == least]
            if self.has_targets():                      # then the ones closest to the targets
                near = min(self.shortfall(c) for c in ok)
                ok = [c for c in ok if self.shortfall(c) <= near + 1e-9]
        key = None
        if self.st.critical == 'weigh' and self.crit is not None:
            w = self.st.critical_weight
            key = lambda c: (c.worst + w * self.crit_penalty(c), -c.min_density, c.cams)   # noqa: E731
        return pareto(ok, key)

    # ---------------------------------------------------------- loop
    def run(self, progress=None, cancel=None):
        st = self.st
        nom = self.evaluate(self.nominal())
        nom.label = 'current'
        self.nominal_cand = nom
        pop = [nom] + [self.evaluate(self.mutate(nom, 1.5)) for _ in range(st.population - 1)]
        self.evaluated += pop
        self.front = self._front(pop)
        for gen in range(st.generations):
            if cancel and cancel():
                break
            self.generation = gen + 1
            pop.sort(key=self.objectives)
            parents = pop[:max(4, st.population // 3)]
            kids = []
            for _ in range(st.population):
                a = parents[int(self.rng.integers(len(parents)))]
                b = parents[int(self.rng.integers(len(parents)))]
                g = {u: list(a.genes[u]) if self.rng.random() < 0.5 else list(b.genes[u]) for u in a.genes}
                child = self.mutate(Candidate(g), scale=max(0.3, 1.0 - gen / max(st.generations, 1)))
                kids.append(self.evaluate(child))
                if cancel and cancel():
                    break
            self.evaluated += kids
            pop = sorted(pop + kids, key=self.objectives)[:st.population]
            self.front = self._front(self.front + kids)
            if progress:
                progress(self)
        if not self.front:
            self.front = self._front(self.evaluated)
        return self.front

    def candidates(self, k=5):
        front = sorted(self.front, key=self.objectives)
        uniq, seen = [], set()
        nom = getattr(self, 'nominal_cand', None)
        for c in front:
            if nom is not None and c is nom:
                continue
            key = (round(c.worst, 3), round(c.min_density, 1), c.cams, c.crit_tol, c.crit_unmet)
            if key in seen:
                continue
            seen.add(key)
            uniq.append(c)
            if len(uniq) >= k:
                break
        for i, c in enumerate(uniq):
            c.label = f'C{i + 1}'
        return uniq

    def changes(self, cand: Candidate) -> str:
        out = []
        removed = [self.base[u].tag for u, g in cand.genes.items() if g[3] <= 0.5]
        if removed:
            out.append('remove ' + ', '.join(removed))
        done = set()
        for uid, g in cand.genes.items():
            if uid in done or g[3] <= 0.5:
                continue
            b = self.base[uid]
            fv = self.vars.get(uid)
            pair = b.link_uid if b.link_uid in cand.genes else ''
            tag = f'{b.tag}/{self.base[pair].tag}' if pair else b.tag
            bits = []
            if abs(g[1] - b.yaw) > 0.5:
                bits.append(f'ψ {abs(b.yaw) if pair else b.yaw:.1f}→{abs(g[1]) if pair else g[1]:.1f}')
            if abs(g[2] - b.pitch) > 0.5:
                bits.append(f'θ {b.pitch:.1f}→{g[2]:.1f}')
            if abs(g[0]) > 0.005 and fv and fv.axis:
                bits.append(f'{fv.axis} {g[0]:+.3f}')
            if bits:
                out.append(f'{tag} ' + ' '.join(bits))
            done.add(uid)
            if pair:
                done.add(pair)
        if not cand.cable_ok:
            out.append('cable over limit')
        if cand.crit_missing:
            out.append('misses ' + ', '.join(cand.crit_missing))
        elif self.crit is not None and self.crit_penalty(cand):
            out.append(f'loses {self.crit_penalty(cand)} critical target{"s" * (self.crit_penalty(cand) > 1)} '
                       f'under tolerance')
        return ' · '.join(out) or 'no change'

    def targets_text(self, cand: Candidate) -> str:
        """'7/8' critical targets resolved at the candidate's pose ('' without critical targets in the search)."""
        if self.crit is None:
            return ''
        n = len(self.crit.targets)
        return f'{n - cand.crit_unmet}/{n}'

    def to_variant(self, cand: Candidate, name: str) -> Variant:
        note = f'Optimiser {cand.label}: worst {cand.worst:.3f} m² under tolerance'
        if self.crit is not None:
            note += f'; critical targets resolved {self.targets_text(cand)}'
        v = Variant(name=name, note=note)
        uid_map = {}
        for uid, gene in cand.genes.items():
            c = self.camera_for(uid, gene)
            if c is None:
                continue
            c = copy.deepcopy(c)
            new = new_uid()
            uid_map[uid] = new
            c.uid = new
            v.cameras.append(c)
        for c in v.cameras:
            c.link_uid = uid_map.get(c.link_uid, '')
        v.sensors = copy.deepcopy(self.variant.sensors)   # the lidars and radars stay as they were
        return v


def pareto(cands: list[Candidate], key=None) -> list[Candidate]:
    """The candidates no other beats on every objective (worst blind area, minimum density, cameras; key(c) gives
    other objective tuples, smaller is better)."""
    if not cands:
        return []
    key = key or (lambda c: (c.worst, -c.min_density, c.cams))
    P = np.array([key(c) for c in cands], float)
    keep = np.ones(len(P), bool)
    for i in range(len(P)):
        if not keep[i]:
            continue
        le = (P <= P[i]).all(1) & (P < P[i]).any(1)
        if le.any():
            keep[i] = False
    return [c for c, k in zip(cands, keep) if k]
