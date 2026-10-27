"""Mounting tolerance: Monte Carlo study, one-at-a-time sensitivity and worst single-camera error.

Rotations reuse each camera's cube map exactly; position errors (millimetres) move the ray
origin but keep the nominal occlusion map, which is accurate for small offsets.
"""
from __future__ import annotations

import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np

from .analysis import Engine, eval_points, describe_location, surface_frame, slide_on_surface
from .lens import resolve
from .mathutil import rot_zyx
from .project import Project, Camera


@dataclass
class TolSettings:
    yaw: float = 5.0
    pitch: float = 5.0
    roll: float = 5.0
    pos_mm: float = 10.0
    distribution: str = 'uniform'      # uniform | normal (± = 2 sigma)
    samples: int = 2000
    seed: int = 1729
    cell: float = 0.05
    cameras: list = field(default_factory=list)   # uids to perturb ([] = all)
    correlate_pairs: bool = False
    metric_blind: bool = True
    metric_density: bool = True
    metric_seam: bool = True


@dataclass
class Case:
    index: int
    blind: float
    min_density: float
    seam_shift: float
    perturb: dict          # uid -> (dyaw, dpitch, droll, dx, dy, dz) degrees / metres
    culprit: str
    where: str
    centroid: tuple | None


@dataclass
class TolResult:
    blind: np.ndarray
    min_density: np.ndarray
    seam_shift: np.ndarray
    worst: list
    sensitivity: list      # [(label, uid, param, delta_minus, delta_plus)]
    seconds: float
    threads: int
    settings: TolSettings
    nominal_blind: float
    finished_at: float = 0.0
    cancelled: bool = False    # stopped early: the arrays hold only the samples that ran

    @property
    def zero_pct(self):
        return 100.0 * float((self.blind <= 1e-9).mean()) if len(self.blind) else 0.0

    def pct(self, q):
        return float(np.percentile(self.blind, q)) if len(self.blind) else 0.0


class _Ctx:
    """Nominal per-camera data on the ROI cells."""

    def __init__(self, engine: Engine, project: Project, cell: float, variant=None):
        self.project = project
        variant = variant or project.variant
        self.grid = engine.roi_grid(project, cell)
        self.engine = engine
        g = self.grid
        self.P = g.points(0.0)[g.roi]
        self.xy = self.P[:, :2]
        self.a = g.cell ** 2
        self.cams = [c for c in variant.cameras if c.visible]
        self.intr = {c.uid: resolve(project.module_for(c), project.lens_for(c)) for c in self.cams}
        self.cube = {c.uid: engine.cube(c.pos()) for c in self.cams}
        self.nom = {}
        for c in self.cams:
            vis, dens, _, _ = eval_points(self.P, c, self.intr[c.uid], self.cube[c.uid])
            self.nom[c.uid] = (vis, dens)
        self.centre = ((g.xs[g.body.any(axis=0)].min() + g.xs[g.body.any(axis=0)].max()) / 2, 0.0) if g.body.any() else (0, 0)
        ang = np.arctan2(self.xy[:, 1] - self.centre[1], self.xy[:, 0] - self.centre[0])
        self.ang = ang
        self.nominal_seams = self.seam_angles({u: v for u, v in self.nom.items()})

    def eval_cam(self, cam: Camera, d):
        dyaw, dp, dr, dx, dy, dz = d
        R = rot_zyx(cam.yaw + dyaw, cam.pitch + dp, cam.roll + dr)
        pos = cam.pos() + np.array([dx, dy, dz])
        vis, dens, _, _ = eval_points(self.P, cam, self.intr[cam.uid], self.cube[cam.uid], R_override=R, pos_override=pos)
        return vis, dens

    def combine(self, per):
        if not self.cams:
            n = len(self.P)
            return float(n * self.a), 0.0, np.ones(n, bool)
        V = np.stack([per[c.uid][0] for c in self.cams])
        D = np.stack([per[c.uid][1] for c in self.cams])
        count = V.sum(0)
        blind_mask = count == 0
        best = D.max(0)
        mind = float(best[~blind_mask].min()) if (~blind_mask).any() else 0.0
        return float(blind_mask.sum() * self.a), mind, blind_mask

    def seam_angles(self, per):
        """Azimuth where best-camera ownership changes, per adjacent pair (cheap estimate)."""
        if not self.cams:
            return {}
        D = np.stack([per[c.uid][1] for c in self.cams])
        best = np.argmax(D, 0)
        cov = D.max(0) > 0
        bins = np.linspace(-math.pi, math.pi, 181)
        idx = np.digitize(self.ang, bins) - 1
        owner = np.full(180, -1)
        for b in range(180):
            m = (idx == b) & cov
            if m.any():
                owner[b] = np.bincount(best[m], minlength=len(self.cams)).argmax()
        out = {}
        for b in range(180):
            a, c = owner[b], owner[(b + 1) % 180]
            if a >= 0 and c >= 0 and a != c:
                out[(min(a, c), max(a, c))] = math.degrees(bins[b + 1])
        return out


def _sample_perturbations(rng, st: TolSettings, cams, n):
    out = []
    pick = set(st.cameras) if st.cameras else {c.uid for c in cams}
    order = {c.uid: i for i, c in enumerate(cams)}      # mirrored pairs by the cameras' order, never their random uids
    for _ in range(n):
        d = {}
        for c in cams:
            if c.uid not in pick:
                continue
            if st.distribution == 'normal':
                v = rng.normal(0, 0.5, 6)
                v = np.clip(v, -1, 1)
            else:
                v = rng.uniform(-1, 1, 6)
            d[c.uid] = (v[0] * st.yaw, v[1] * st.pitch, v[2] * st.roll,
                        v[3] * st.pos_mm / 1000, v[4] * st.pos_mm / 1000, v[5] * st.pos_mm / 1000)
        if st.correlate_pairs:
            for c in cams:
                if c.link_uid and c.uid in d and c.link_uid in d and order[c.uid] < order[c.link_uid]:
                    a = d[c.uid]
                    d[c.link_uid] = (-a[0], a[1], -a[2], a[3], -a[4], a[5])
        out.append(d)
    return out


def run_monte_carlo(engine: Engine, project: Project, st: TolSettings, progress=None, cancel=None, threads=None) -> TolResult:
    t0 = time.perf_counter()
    ctx = _Ctx(engine, project, st.cell)
    rng = np.random.default_rng(st.seed)
    perts = _sample_perturbations(rng, st, ctx.cams, st.samples)
    nb, nmd, _ = ctx.combine(ctx.nom)
    threads = threads or max(1, min(8, (os.cpu_count() or 2)))
    blind = np.zeros(len(perts))
    mind = np.zeros(len(perts))
    seam = np.zeros(len(perts))
    ran = np.zeros(len(perts), bool)
    masks = {}
    done = [0]
    cams_by_uid = {c.uid: c for c in ctx.cams}

    def work(i):
        if cancel and cancel():
            return
        per = dict(ctx.nom)
        for uid, d in perts[i].items():
            per[uid] = ctx.eval_cam(cams_by_uid[uid], d)
        b, md, bm = ctx.combine(per)
        blind[i], mind[i] = b, md
        if st.metric_seam:
            sa = ctx.seam_angles(per)
            shifts = [abs((sa[k] - v + 180) % 360 - 180) for k, v in ctx.nominal_seams.items() if k in sa]
            seam[i] = max(shifts) if shifts else 0.0
        if b > 0:
            masks[i] = bm
        ran[i] = True
        done[0] += 1
        if progress and done[0] % 20 == 0:
            progress(done[0] / len(perts))

    with ThreadPoolExecutor(threads) as ex:
        list(ex.map(work, range(len(perts))))
    # a cancelled study keeps only the samples that were evaluated (the others are not 'zero blind')
    idx = np.nonzero(ran)[0]
    cancelled = len(idx) < len(perts)
    if not len(idx):
        raise RuntimeError('Cancelled before any sample was evaluated.')
    order = idx[np.argsort(-blind[idx] - 1e-6 * (10 - mind[idx]))]
    worst = []
    for i in order[:10]:
        i = int(i)
        bm = masks.get(i)
        culprit, cen, where = '', None, '—'
        if bm is not None and bm.any():
            # the camera that nominally covered most of the new blind cells
            score = {uid: float((ctx.nom[uid][0] & bm).sum()) for uid in perts[i]}
            culprit = max(score, key=score.get) if score else ''
            xy = ctx.xy[bm]
            cen = (float(xy[:, 0].mean()), float(xy[:, 1].mean()))
            where = describe_location(cen[0], cen[1], project)
        else:
            culprit = max(perts[i], key=lambda u: sum(abs(x) for x in perts[i][u][:3])) if perts[i] else ''
        worst.append(Case(i, float(blind[i]), float(mind[i]), float(seam[i]), perts[i], culprit, where, cen))
    sens = sensitivity(ctx, st) if not cancelled else []
    if progress:
        progress(1.0)
    return TolResult(blind[idx], mind[idx], seam[idx], worst, sens, time.perf_counter() - t0, threads, st, nb, time.time(),
                     cancelled)


def sensitivity(ctx: _Ctx, st: TolSettings):
    nb, _, _ = ctx.combine(ctx.nom)
    rows = []
    pick = set(st.cameras) if st.cameras else {c.uid for c in ctx.cams}
    params = [('ψ', 0, st.yaw), ('θ', 1, st.pitch), ('φ', 2, st.roll), ('x', 3, st.pos_mm / 1000),
              ('y', 4, st.pos_mm / 1000), ('z', 5, st.pos_mm / 1000)]
    for c in ctx.cams:
        if c.uid not in pick:
            continue
        for sym, k, amp in params:
            if amp == 0:
                continue
            res = []
            for s in (-1, 1):
                d = [0.0] * 6
                d[k] = s * amp
                per = dict(ctx.nom)
                per[c.uid] = ctx.eval_cam(c, d)
                b, _, _ = ctx.combine(per)
                res.append(b - nb)
            rows.append((f'{c.tag} {sym}', c.uid, sym, res[0], res[1]))
    rows.sort(key=lambda r: -max(abs(r[3]), abs(r[4])))
    return rows


def worst_single_camera(engine: Engine, project: Project, tol_deg=5.0, tol_m=0.10, cell=0.05):
    """Worst blind area when any ONE camera is off by tol_deg in yaw/pitch, or slid tol_m along the
    body surface (horizontal tangent). Returns (blind m2, camera uid, perturbation)."""
    ctx = _Ctx(engine, project, cell)
    worst = (0.0, '', None)
    mesh = engine.vg.occ_mesh
    for c in ctx.cams:
        trials = []
        for dy in (-tol_deg, 0, tol_deg):
            for dp in (-tol_deg, 0, tol_deg):
                if dy or dp:
                    trials.append((dy, dp, 0, 0, 0, 0))
        _, _, t1, _, _ = surface_frame(mesh, c.pos())
        for tdir in (t1,):          # along the body: sideways on the bumpers, fore-aft on the sides
            for s in (-1, 1):
                p = slide_on_surface(mesh, c, tdir, s * tol_m)
                trials.append((0.0, 0.0, 0.0) + tuple(p - c.pos()))
        for d in trials:
            per = dict(ctx.nom)
            if any(abs(x) > 0 for x in d[3:]):
                # a 10 cm shift is big enough to rebuild the occlusion map
                pos = c.pos() + np.array(d[3:])
                cube = engine.cube(pos)
                R = rot_zyx(c.yaw, c.pitch, c.roll)
                vis, dens, _, _ = eval_points(ctx.P, c, ctx.intr[c.uid], cube, R_override=R, pos_override=pos)
                per[c.uid] = (vis, dens)
            else:
                per[c.uid] = ctx.eval_cam(c, d)
            b, _, _ = ctx.combine(per)
            if b > worst[0]:
                worst = (b, c.uid, d)
    return worst
