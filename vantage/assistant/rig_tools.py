"""Whole-rig design for the assistant: optimise_rig and add_camera_options.

optimise_camera (tools.py) searches one camera's pose. These tools work on the rig as a whole:

- optimise_rig runs the placement optimiser of the Optimise page (core.optimiser: an evolutionary search over every
  free camera's slide along the body, yaw and pitch, cameras switched off when the count allows it, a Pareto front of
  worst blind area under tolerance, minimum density and camera count) on a snapshot, within a time budget. Stated
  targets (a minimum density, a blind-area limit, a camera count, critical targets) are constraints of the search.
  Each option on the front comes back as apply_changes-style change dicts, re-checked at the project's ground
  resolution through tools._simulate (the path simulate_changes, propose_changes and optimise_camera use), so
  propose_changes can offer an option unchanged and shows the same figures.
- add_camera_options finds where one more camera (or a mirrored pair) would close the largest blind region or density
  gap: candidate mounts on the mountable body surface (inside the allowed zones when the vehicle has any), each aimed
  at the gap and refined, evaluated against the cameras already there, the best re-checked at the project's
  resolution with the camera added to a copy.

Nothing changes in the user's design: both tools work on copies, and an option only reaches the design when the user
accepts the proposal the model makes from it. Locked cameras never move or go; a mirrored pair moves as a mirror image
and is dropped as a pair.

Limits (stated in every result): the search scores the ground on a grid of at least 10 cm (5 cm for the add search) at
the planned occlusion (the housings stay where they are planned, as the Optimise page scores them); the re-check at
the project's resolution includes everything. The lidars and radars do not move. Positions only slide along the body
surface within the ranges; a new camera's mount is a surface point, not a bracket design."""
from __future__ import annotations

import copy
import math
import time

import numpy as np

BUDGET_MAX = 90.0                      # seconds, either tool
OBJECTIVES = ('blind', 'cameras', 'density')
ADD_OBJECTIVES = ('blind', 'density')
RIG_VARY = ('position', 'yaw', 'pitch')
WHERE = ('any', 'front', 'rear', 'left', 'right', 'roof')
CRITICAL = ('off', 'require', 'weigh')
POP = 20                               # optimiser population per generation
TOL_SAMPLES = 8                        # tolerance samples per candidate (the Optimise page uses 24)
SEARCH_CELL = 0.10                     # m, the optimiser's ground grid (at least)
LIMITS = ('The search scores the ground on a {cell:g} m grid with the housings where they are planned; every option '
          'is re-checked at the project\'s resolution with everything included. The lidars and radars do not move; '
          'cameras slide along the body surface and turn within the ranges.')


def _T():
    from . import tools
    return tools


def _num(v, name, lo=None, hi=None, default=None):
    """A finite float from v (None gives default), clamped to [lo, hi]; ValueError naming the argument."""
    if v is None:
        return default
    if isinstance(v, bool):
        raise ValueError(f'{name} must be a number')
    try:
        x = float(v)
    except (TypeError, ValueError):
        raise ValueError(f'{name} must be a number') from None
    if not math.isfinite(x):
        raise ValueError(f'{name} must be finite')
    if lo is not None:
        x = max(lo, x)
    if hi is not None:
        x = min(hi, x)
    return x


def _wrap(a):
    return (float(a) + 180.0) % 360.0 - 180.0


def _mirrored(a, b) -> bool:
    """Whether cameras a and b are mirror images across the centre plane (as mirror_props would make them)."""
    return (abs(a.x - b.x) < 0.01 and abs(a.y + b.y) < 0.01 and abs(a.z - b.z) < 0.01
            and abs(_wrap(a.yaw + b.yaw)) < 0.5 and abs(a.pitch - b.pitch) < 0.5)


def _deadline(tc, t_end):
    return lambda: tc.cancelled() or time.perf_counter() >= t_end


# =========================================================================== optimise_rig
def _rig_vars(p, vary, ranges, free_uids):
    """FreeVars for the visible cameras: the Optimise page's defaults (core.optimiser.default_vars) where the camera is
    free, narrowed to what vary names and widened or narrowed by ranges; fixed and kept for the others (locked, not
    named, or the partner of a locked camera). A mirrored pair stays linked only when it is a mirror image now."""
    from ..core.optimiser import FreeVar, default_vars
    v = p.variant
    defaults = {fv.uid: fv for fv in default_vars(p, v)}
    vis = {c.uid: c for c in v.cameras if c.visible}
    out, notes, seen = [], [], set()
    for c in v.cameras:
        if c.uid not in vis or c.uid in seen:
            continue
        partner = vis.get(c.link_uid) if c.link_uid else None
        group = [c] + ([partner] if partner is not None else [])
        seen.update(x.uid for x in group)
        linked = partner is not None and _mirrored(c, partner)
        # a mirrored pair is free only as a whole; partners that are not mirror images now are searched apart
        frees = [all(y.uid in free_uids and not y.locked for y in group)] * len(group) if linked else \
            [x.uid in free_uids and not x.locked for x in group]
        if partner is not None and not linked:
            notes.append(f'{c.tag} and {partner.tag} are linked but not mirror images now: searched independently')
        for i, x in enumerate(group):
            if not frees[i]:
                out.append(FreeVar(x.uid, x.tag, axis='', move=0.0, yaw=(x.yaw, x.yaw), pitch=(x.pitch, x.pitch),
                                   keep=True))
                continue
            if i == 1 and linked:
                out.append(FreeVar(x.uid, x.tag, linked_to=c.uid))
                continue
            d = defaults.get(x.uid)
            if d is not None and d.linked_to:           # a partner searched on its own: not the link's defaults
                d = None
            axis = d.axis if d is not None and d.axis else \
                ('y' if abs(x.yaw) < 30 or abs(abs(x.yaw) - 180) < 30 else 'x')
            move = d.move if d is not None else 0.10
            yaw = tuple(d.yaw) if d is not None else (x.yaw - 15, x.yaw + 15)
            pitch = tuple(d.pitch) if d is not None else (x.pitch - 10, x.pitch + 10)
            if ranges.get('position_m') is not None:
                move = ranges['position_m']
            if ranges.get('yaw_deg') is not None:
                yaw = (x.yaw - ranges['yaw_deg'], x.yaw + ranges['yaw_deg'])
            if ranges.get('pitch_deg') is not None:
                pitch = (x.pitch - ranges['pitch_deg'], x.pitch + ranges['pitch_deg'])
            # the current pose always lies inside its range, so the search starts from it
            yaw = (min(yaw[0], x.yaw), max(yaw[1], x.yaw))
            pitch = (max(-85.0, min(pitch[0], x.pitch)), min(85.0, max(pitch[1], x.pitch)))
            if 'position' not in vary:
                move = 0.0
            if 'yaw' not in vary:
                yaw = (x.yaw, x.yaw)
            if 'pitch' not in vary:
                pitch = (x.pitch, x.pitch)
            out.append(FreeVar(x.uid, x.tag, axis=axis if move > 0 else '', move=round(float(move), 3),
                               yaw=(float(yaw[0]), float(yaw[1])), pitch=(float(pitch[0]), float(pitch[1])),
                               step=round(max(0.02, move / 6), 3)))
    return out, notes


def _pose_props(base, new) -> dict:
    """The absolute pose values that differ between base and new (mm, 0.01°)."""
    out = {}
    for k in ('x', 'y', 'z'):
        if abs(getattr(new, k) - getattr(base, k)) > 5e-4:
            out[k] = round(float(getattr(new, k)), 4)
    if abs(_wrap(new.yaw - base.yaw)) > 0.05:
        out['yaw'] = round(_wrap(new.yaw), 2)
    if abs(new.pitch - base.pitch) > 0.05:
        out['pitch'] = round(float(new.pitch), 2)
    return out


def _same_pose(a, b) -> bool:
    return (all(abs(getattr(a, k) - getattr(b, k)) <= 3e-3 for k in ('x', 'y', 'z'))
            and abs(_wrap(a.yaw - b.yaw)) <= 0.05 and abs(a.pitch - b.pitch) <= 0.05)


def candidate_changes(opt, cand):
    """A candidate of the optimiser as apply_changes change dicts, and the tags it switches off. A mirrored pair is
    one change with mirror_partner when mirroring the first camera's new pose gives the second's (within 3 mm and
    0.05°), else a change for each; a removed camera is {'camera': tag, 'visible': False}."""
    from ..core.project import mirror_props
    out, removed, done = [], [], set()
    for uid, gene in cand.genes.items():
        b = opt.base[uid]
        fv = opt.vars.get(uid)
        if uid in done or not b.visible or fv is None or fv.linked_to:
            continue
        group = [uid] + [u for u, f in opt.vars.items() if f.linked_to == uid and u in opt.base]
        done.update(group)
        mirror = {'mirror_partner': False} if b.link_uid else {}
        if gene[3] <= 0.5:
            for u in group:
                out.append({'camera': opt.base[u].tag, 'visible': False,
                            **({'mirror_partner': False} if opt.base[u].link_uid else {})})
                removed.append(opt.base[u].tag)
            continue
        new = opt.camera_for(uid, gene)
        props = _pose_props(b, new)
        if len(group) == 1:
            if props:
                out.append({'camera': b.tag, **props, **mirror})
            continue
        f = group[1]
        fb, fnew = opt.base[f], opt.camera_for(f, cand.genes[f])
        src, pred = copy.deepcopy(b), copy.deepcopy(fb)
        for k, val in props.items():
            setattr(src, k, val)
        mirror_props(src, pred, list(props))
        if _same_pose(pred, fnew):
            if props:
                out.append({'camera': b.tag, **props, 'mirror_partner': True})
            continue
        if props:
            out.append({'camera': b.tag, **props, 'mirror_partner': False})
        fprops = _pose_props(fb, fnew)
        if fprops:
            out.append({'camera': fb.tag, **fprops, 'mirror_partner': False})
    return out, removed


def _polish(opt, cand, ev, t_end):
    """The candidate with every camera (or mirrored pair) whose own change does not pay put back where it is now:
    reverted one at a time when the rig scores at least as well on the objective without it, or no worse on the search
    grid (blind and worst blind no larger, targets and critical targets no further off, minimum density at most
    0.01 px/cm lower), with the cables no worse.
    The search drifts the cameras that do not matter for the objective; the 10 cm grid cannot tell, the project's
    finer grid can, and the user gets fewer changes to make."""
    from ..core.optimiser import Candidate
    ref = ev(cand)
    for uid, fv in opt.vars.items():
        if time.perf_counter() > t_end:
            break
        g, b = ref.genes.get(uid), opt.base.get(uid)
        if fv.linked_to or fv.keep or g is None or g[3] <= 0.5:
            continue
        if abs(g[0]) < 1e-9 and abs(g[1] - b.yaw) < 1e-9 and abs(g[2] - b.pitch) < 1e-9:
            continue
        genes = {u: list(x) for u, x in ref.genes.items()}
        genes[uid][:3] = [0.0, b.yaw, b.pitch]
        opt._apply_links(genes)
        t = ev(Candidate(genes, label=cand.label))
        no_worse = (opt.shortfall(t) <= opt.shortfall(ref) + 1e-9 and opt.crit_penalty(t) <= opt.crit_penalty(ref)
                    and t.worst <= ref.worst + 1e-9 and t.blind <= ref.blind + 1e-9
                    and t.min_density >= ref.min_density - 0.01)
        if (t.cable_ok or not ref.cable_ok) and (no_worse or opt.objectives(t) <= opt.objectives(ref)):
            ref = t
    return ref


def _worth(m, cur, objective, under_tol) -> bool:
    """Whether an option improves on the current rig on what was asked, by more than the noise of the grid (0.002 m²
    of blind ground, 0.005 m² of the worst case under tolerance, 0.05 px/cm), as optimise_camera judges."""
    db = cur['blind_m2'] - m['blind_m2']
    dw = (cur['worst_blind_m2'] - m['worst_blind_m2']) if under_tol else 0.0
    dd = m['min_density'] - cur['min_density']
    dc = cur['cameras'] - m['cameras']
    if objective == 'cameras':
        return dc > 0 or (db >= 0.002 or dw >= 0.005) and dd > -0.05
    if objective == 'density':
        return dd >= 0.05 and db > -0.005
    return (db >= 0.002 or (dw >= 0.005 and db > -0.002)) and dd > -0.05


def _fewer(opt, n, lo):
    """For each camera count below n that the search tried (down to lo): its rig closest to the targets, on the search
    grid, with what it misses. Tells the user why no option drops a camera."""
    out = []
    for k in range(n - 1, lo - 1, -1):
        pool = [c for c in opt.evaluated if c.cams == k]
        if not pool:
            continue
        c = min(pool, key=lambda c: (opt.shortfall(c), opt.crit_penalty(c), c.worst, -c.min_density))
        miss = []
        if opt.st.need_density > 0 and c.min_density < opt.st.need_density:
            miss.append(f'minimum density {c.min_density:.2f} < {opt.st.need_density:g} px/cm')
        if opt.st.need_blind >= 0 and c.worst > opt.st.need_blind:
            miss.append(f'blind {c.worst:.3f} > {opt.st.need_blind:g} m² (worst case)')
        if opt.crit is not None and opt.crit_penalty(c):
            miss.append(f'{opt.crit_penalty(c)} critical target(s) unresolved')
        if not c.cable_ok:
            miss.append('a cable over its limit')
        out.append({'cameras': k, 'removes': [opt.base[u].tag for u, g in c.genes.items()
                                              if g[3] <= 0.5 and opt.base[u].visible],
                    'on_search_grid': {'worst_blind_m2': _r(c.worst, 3), 'min_density_px_cm': _r(c.min_density, 2)},
                    'misses': miss or ['nothing on the search grid; it did not improve on the current rig at the '
                                       'project\'s resolution'],
                    'rigs_tried': len(pool)})
    return out


def _dominated(rows, i) -> bool:
    a = rows[i]
    for j, b in enumerate(rows):
        if j != i and all(x <= y for x, y in zip(b, a)) and any(x < y for x, y in zip(b, a)):
            return True
    return False


def _key(m):
    """Dominance values of an option (smaller is better): blind and minimum density at the project's resolution,
    cameras, worst blind under tolerance on the search grid."""
    return (round(m['blind_m2'], 3), -round(m['min_density'], 2), m['cameras'], round(m['worst_blind_m2'], 3))


def optimise_rig(tc, cameras=None, vary=None, ranges=None, objective='blind', min_density=None, max_blind_m2=None,
                 count=None, under_tolerance=True, tolerance_deg=None, tolerance_mm=None, critical='off', options=4,
                 budget_s=30, seed=7, generations=400, population=POP, **_):
    """The placement optimiser over the whole rig on a snapshot, as up to `options` complete rig options."""
    T = _T()
    from ..core.optimiser import Candidate, Optimiser, OptSettings
    t0 = time.perf_counter()
    p = tc.project.snapshot()
    vis = [c for c in p.variant.cameras if c.visible]
    if not vis:
        return {'error': 'the active variant has no visible camera to optimise'}
    if objective not in OBJECTIVES:
        return {'error': f"objective must be one of {', '.join(OBJECTIVES)}"}
    if critical not in CRITICAL:
        return {'error': f"critical must be one of {', '.join(CRITICAL)}"}
    if not isinstance(under_tolerance, bool):
        return {'error': 'under_tolerance must be true or false'}
    if vary is not None and (not isinstance(vary, list) or not vary or any(k not in RIG_VARY for k in vary)):
        return {'error': f'vary must name some of {", ".join(RIG_VARY)}'}
    vary = list(vary or RIG_VARY)
    if ranges is not None and not isinstance(ranges, dict):
        return {'error': 'ranges must be an object: {"position_m", "yaw_deg", "pitch_deg"}'}
    unknown = [k for k in (ranges or {}) if k not in ('position_m', 'yaw_deg', 'pitch_deg')]
    if unknown:
        return {'error': f'unknown range {unknown[0]!r}; ranges: position_m, yaw_deg, pitch_deg'}
    try:
        rg = {'position_m': _num((ranges or {}).get('position_m'), 'ranges.position_m', 0.0, 0.5),
              'yaw_deg': _num((ranges or {}).get('yaw_deg'), 'ranges.yaw_deg', 0.0, 45.0),
              'pitch_deg': _num((ranges or {}).get('pitch_deg'), 'ranges.pitch_deg', 0.0, 30.0)}
        need_d = _num(min_density, 'min_density', 0.0, 80.0)
        need_b = _num(max_blind_m2, 'max_blind_m2', 0.0, 1e4)
        tol_deg = _num(tolerance_deg, 'tolerance_deg', 0.0, 15.0, 5.0)
        tol_mm = _num(tolerance_mm, 'tolerance_mm', 0.0, 100.0, 10.0)
        budget = _num(budget_s, 'budget_s', 5.0, BUDGET_MAX, 30.0)
        k_opt = int(_num(options, 'options', 1, 5, 4))
        seed = int(_num(seed, 'seed', 0, 2 ** 31, 7))
        gens = int(_num(generations, 'generations', 1, 2000, 400))
        pop = int(_num(population, 'population', 6, 200, POP))
    except ValueError as e:
        return {'error': str(e)}
    want = None
    if cameras is not None:
        if not isinstance(cameras, list) or not cameras:
            return {'error': 'cameras must be a non-empty list of camera tags (or left out for every camera)'}
        want = set()
        for t in cameras:
            c = T._by_tag(p, t)
            if c is None:
                return {'error': f'no camera {t!r}; cameras: {", ".join(x.tag for x in p.variant.cameras)}'}
            if not c.visible:
                return {'error': f'{c.tag} is switched off in the analysis; switch it on to optimise it'}
            want.add(c.uid)
            if c.link_uid:                               # a mirrored partner comes along
                want.add(c.link_uid)
    free_uids = {c.uid for c in vis if not c.locked and (want is None or c.uid in want)}
    if not free_uids:
        return {'error': 'every camera named is locked; the user has to unlock one first'}
    n = len(vis)
    if count is not None:
        if (not isinstance(count, list) or len(count) != 2
                or not all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in count)):
            return {'error': 'count must be [min, max] cameras'}
        lo, hi = sorted(int(round(x)) for x in count)
        if hi < 1:
            return {'error': 'count must allow at least one camera'}
    else:
        lo, hi = (1, n) if objective == 'cameras' else (n, n)
    notes = []
    if hi > n:
        notes.append(f'the rig has {n} cameras: optimise_rig moves and drops cameras, it does not add them '
                     f'(add_camera_options finds where one more would help)')
        hi = n
    if lo > n:
        return {'error': f'the rig has {n} visible cameras, fewer than the {lo} asked for; add_camera_options finds '
                         f'where more would help'}
    lo = max(1, lo)
    fvars, vnotes = _rig_vars(p, vary, rg, free_uids)
    free_uids = {fv.uid for fv in fvars if not fv.keep}
    if not free_uids:
        return {'error': 'every camera the search could move is locked or mirrored with a locked one; the user has to '
                         'unlock one first'}
    notes += vnotes
    base = T._base(tc)
    if base is None:
        return {'error': 'The analysis cannot run (no vehicle or ROI).'}
    t_an = max(0.4, time.perf_counter() - t0)                   # ~ one analysis (less when _base was cached)
    cell = max(p.analysis.cell_m, SEARCH_CELL)
    st = OptSettings(objective=objective, under_tolerance=bool(under_tolerance), tol_deg=tol_deg, tol_mm=tol_mm,
                     tol_samples=TOL_SAMPLES if under_tolerance else 0, height=(0.05, 5.0), tol_seed=seed + 1,
                     count=(lo, hi), generations=gens, population=pop, cell=cell, seed=seed, vars=fvars,
                     critical=critical, need_density=need_d or 0.0,
                     need_blind=need_b if need_b is not None else -1.0, units=True)
    tc.status('Optimising the rig: preparing the search')
    opt = Optimiser(tc.engine, p, st)
    if opt.grid is None:
        return {'error': 'The analysis cannot run (no vehicle or ROI).'}
    implied = {}
    if need_b is None and objective in ('cameras', 'density'):
        # "fewest cameras" / "most density" never at the price of more blind ground than the current rig has
        nom = opt.evaluate(opt.nominal())                       # under the search's draws (common to every rig)
        opt.st.need_blind = round(nom.worst + 1e-3, 4)
        implied['max_worst_blind_m2'] = _r(opt.st.need_blind, 3)
        opt.evals = 0
    n_check = min(k_opt + 3, 8)
    reserve = n_check * (2.2 * t_an + 0.25) + 0.5               # the re-checks at the project's resolution
    t_end = t0 + max(budget * 0.35, budget - reserve)
    stop = _deadline(tc, t_end)

    def prog(o):
        best = min(o.front, key=o.objectives) if o.front else None
        tc.status(f'Optimising the rig: generation {o.generation} · {o.evals} rigs scored'
                  + (f' · best worst-case blind {best.worst:.3f} m², {best.cams} cameras' if best else '')
                  + f' · {max(0.0, t_end - time.perf_counter()):.0f} s left')
    opt.run(progress=prog, cancel=stop)
    search = {'rigs_scored': opt.evals, 'generations': opt.generation, 'seconds': _r(time.perf_counter() - t0, 1),
              'stopped': 'cancelled' if tc.cancelled() else 'budget' if time.perf_counter() >= t_end else 'finished',
              'grid_cell_m': cell, 'seed': seed,
              'tolerance': (f'worst of {TOL_SAMPLES} random mounting errors up to ±{tol_deg:g}° and ±{tol_mm:g} mm '
                            f'per camera, the same draws for every rig; the worst case reported is under '
                            f'{TOL_SAMPLES} other draws' if under_tolerance else 'off: the nominal poses only')}
    if tc.cancelled():
        return {'error': 'cancelled', 'search': search}
    # The search scores every rig under the same tolerance draws (seed + 1), so rigs differ by their poses, not their
    # luck; the polish uses them too. The worst case reported comes from fresh draws (seed + 2), the same for the
    # current rig and every option: an out-of-sample figure, not the one the search tuned for.
    def ev(c, tol_seed=seed + 1):
        keep, opt.st.tol_seed = opt.st.tol_seed, tol_seed
        try:
            return opt.evaluate(Candidate({u: list(g) for u, g in c.genes.items()}, label=c.label))
        finally:
            opt.st.tol_seed = keep
    nom = ev(opt.nominal(), seed + 2)
    cands, seen_genes = [], set()
    for c in opt.candidates(n_check):
        c = _polish(opt, c, ev, t0 + budget - 2.2 * t_an - 0.25)
        key = tuple((u, tuple(round(x, 3) for x in g)) for u, g in sorted(c.genes.items()))
        if key not in seen_genes:
            seen_genes.add(key)
            cands.append(c)
    # the current design as the yardstick: an option it dominates is no option
    tc.status('Optimising the rig: re-checking the options at the project\'s resolution')
    em0 = base.metrics
    cur = {'blind_m2': float(em0['blind_m2']), 'min_density': float(em0['min_density']), 'cameras': n,
           'worst_blind_m2': float(nom.worst)}
    opts = []
    for c in list(cands):
        # a rig that drops cameras also as "drop only": the other cameras' moves may only pay on the search grid
        if any(g[3] <= 0.5 for u, g in c.genes.items() if opt.base[u].visible) and \
                any(abs(g[0]) > 1e-9 or abs(g[1] - opt.base[u].yaw) > 1e-9 or abs(g[2] - opt.base[u].pitch) > 1e-9
                    for u, g in c.genes.items() if g[3] > 0.5):
            genes = {u: [0.0, opt.base[u].yaw, opt.base[u].pitch, g[3]] for u, g in c.genes.items()}
            opt._apply_links(genes)
            cands.append(ev(Candidate(genes, label=c.label + ' drop only')))
    for cand in cands:
        if tc.cancelled() or (opts and time.perf_counter() > t0 + budget - 2.2 * t_an - 0.25):
            break
        changes, removed = candidate_changes(opt, cand)
        if not changes:
            continue
        sim, rep = T._simulate(tc, changes)
        if sim is None:
            notes.append(f'{cand.label}: {rep.get("error")}')
            continue
        eff = rep['effect']
        after = {'blind_m2': float(eff['blind_m2']['after'] or 0.0), 'min_density': float(eff['min_density']['after'] or 0.0),
                 'cameras': n - len(removed), 'worst_blind_m2': float(ev(cand, seed + 2).worst)}
        opts.append({'cand': cand, 'changes': changes, 'removed': removed, 'rep': rep, 'm': after})
    if tc.cancelled():
        return {'error': 'cancelled', 'search': search}

    def meets(o):
        m, c = o['m'], o['cand']
        out = {}
        if need_d is not None:
            out['min_density'] = m['min_density'] >= need_d - 1e-9
        if need_b is not None:
            out['max_blind_m2'] = m['blind_m2'] <= need_b + 1e-9
        if count is not None:
            out['count'] = lo <= m['cameras'] <= hi
        if opt.crit is not None:
            out['critical'] = opt.crit_penalty(c) == 0
        if implied:                                   # the implied limit, at the project's resolution (± one cell)
            out['no_more_blind_than_now'] = m['blind_m2'] <= cur['blind_m2'] + max(0.002, p.analysis.cell_m ** 2)
        out['cable'] = bool(c.cable_ok)
        return out
    for o in opts:
        o['meets'] = meets(o)
        o['ok'] = all(o['meets'].values())
    dropped = 0
    if any(o['ok'] for o in opts):
        dropped = sum(1 for o in opts if not o['ok'])
        opts = [o for o in opts if o['ok']]
    rows = [_key(o['m']) for o in opts]
    keep = [o for i, o in enumerate(opts) if not _dominated(rows, i) and not _dominated([_key(cur), rows[i]], 1)
            and _worth(o['m'], cur, objective, under_tolerance)]
    worse = len(opts) - len(keep)
    rank = {'blind': lambda o: (o['m']['blind_m2'], o['m']['worst_blind_m2'], -o['m']['min_density'], o['m']['cameras']),
            'cameras': lambda o: (o['m']['cameras'], o['m']['blind_m2'], o['m']['worst_blind_m2'], -o['m']['min_density']),
            'density': lambda o: (-o['m']['min_density'], o['m']['blind_m2'], o['m']['worst_blind_m2'],
                                  o['m']['cameras'])}[objective]
    keep.sort(key=rank)
    seen, uniq = set(), []
    for o in keep:
        k = _key(o['m'])
        if k not in seen:
            seen.add(k)
            uniq.append(o)
    keep = uniq[:k_opt]
    tags_vis = [c.tag for c in vis]
    out_opts = []
    for i, o in enumerate(keep):
        label = chr(ord('A') + i)
        m, rep = o['m'], o['rep']
        eff = rep['effect']
        used = [t for t in tags_vis if t not in o['removed']]
        out_opts.append({
            'option': label,
            'changes': o['changes'],
            'cameras': {'count': len(used), 'used': used, 'removed': o['removed']},
            'metrics': {k: [eff[k]['before'], eff[k]['after']] for k in eff},
            'worst_blind_under_tolerance_m2': ([_r(nom.worst, 3), _r(m['worst_blind_m2'], 3)]
                                               if under_tolerance else None),
            'critical_targets_resolved': opt.targets_text(o['cand']) or None,
            'cable_ok': bool(o['cand'].cable_ok),
            'issues': {'before': rep['issues']['before'], 'after': rep['issues']['after'],
                       'new': rep['issues']['new'], 'resolved': rep['issues']['resolved']},
            'meets_targets': o['meets'], 'meets_all_targets': o['ok'],
            'summary': opt.changes(o['cand']),
            'trade_off': ''})
    for i, row in enumerate(out_opts):
        row['trade_off'] = _trade_off(row, out_opts, i, objective)
        if not row['meets_all_targets']:
            row['trade_off'] += ' · misses ' + ', '.join(k.replace('_', ' ') for k, v in row['meets_targets'].items()
                                                          if not v)
        row['metrics'] = {k: [_r(a, 3), _r(b, 3)] for k, (a, b) in row['metrics'].items()}
    targets = {k: v for k, v in (('min_density_px_cm', need_d), ('max_blind_m2', need_b),
                                 ('count', [lo, hi] if count is not None else None),
                                 ('critical', critical if critical != 'off' else None)) if v is not None}
    locked = [c.tag for c in vis if c.uid not in free_uids]
    res = {'objective': objective, 'targets': targets, **({'implied_targets': implied} if implied else {}),
           'fixed_cameras': locked, 'varied': vary,
           'current': {'blind_m2': _r(cur['blind_m2'], 3), 'min_density_px_cm': _r(cur['min_density'], 2),
                       'cameras': n, **({'worst_blind_under_tolerance_m2': _r(nom.worst, 3)} if under_tolerance else {}),
                       **({'critical_targets_resolved': opt.targets_text(nom)} if opt.crit is not None else {})},
           'options': out_opts, 'search': {**search, 'seconds': _r(time.perf_counter() - t0, 1)},
           'limits': LIMITS.format(cell=cell)}
    if dropped:
        notes.append(f'{dropped} option{"s" * (dropped > 1)} that missed the targets left out')
    if worse:
        notes.append(f'{worse} option{"s" * (worse > 1)} no better than another (or than the current rig) left out')
    if lo < n and not any(o['cameras']['count'] < n for o in out_opts):
        res['fewer_cameras'] = _fewer(opt, n, lo)
        if res['fewer_cameras']:
            notes.append('no option drops a camera: fewer_cameras shows the closest rig the search found per count '
                         'and what it misses')
    if not out_opts:
        res['result'] = ('no rig found that improves on the current one' + (' and meets the targets' if targets else '')
                         + ' within the ranges and the budget' +
                         ('; a longer budget_s or wider ranges may find one' if search['stopped'] == 'budget' else ''))
    elif not any(o['meets_all_targets'] for o in out_opts):
        res['result'] = 'no option meets every target; the closest are shown with what each misses'
    res['next'] = ('explain the options and their trade-offs; offer the one the user prefers with '
                   'propose_changes(changes=option.changes): nothing changes until they accept it') if out_opts else \
        'tell the user; try wider ranges, another objective, or add_camera_options'
    if notes:
        res['notes'] = notes
    return res


def _trade_off(row, rows, i, objective) -> str:
    """One line: what the option gives and costs, and what it is best at among the options."""
    m = row['metrics']
    b0, b1 = m['blind_m2']
    d0, d1 = m['min_density']
    cams = row['cameras']
    s = (f"{cams['count']} camera{'s' * (cams['count'] != 1)}"
         + (f" (drops {', '.join(cams['removed'])})" if cams['removed'] else '')
         + f": blind {b0:.3f}→{b1:.3f} m², minimum density {d0:.2f}→{d1:.2f} px/cm")
    w = row.get('worst_blind_under_tolerance_m2')
    if w:
        s += f', worst case under tolerance {w[0]:.3f}→{w[1]:.3f} m²'
    best = []
    if len(rows) > 1:
        if b1 <= min(r['metrics']['blind_m2'][1] for r in rows) + 1e-9:
            best.append('least blind ground')
        if d1 >= max(r['metrics']['min_density'][1] for r in rows) - 1e-9:
            best.append('highest minimum density')
        if cams['count'] <= min(r['cameras']['count'] for r in rows) and \
                any(r['cameras']['count'] > cams['count'] for r in rows):
            best.append('fewest cameras')
    if best:
        s += ' · best for ' + ' and '.join(best)
    return s


def _r(v, n=3):
    return _T()._r(v, n)


# =========================================================================== add_camera_options
def _mount_faces(vg, zones):
    """Face indices of the occluder mesh a camera may be mounted on: a mountable part outside every forbidden zone
    (inside an allowed zone when zones='allowed' and the vehicle has allowed zones), not facing the ground, above
    0.25 m. Returns (faces, centres, normals, areas, zone names)."""
    from ..core.mesh import ROLE_MOUNTABLE
    m = vg.occ_mesh
    fi = np.nonzero(m.part_mask({ROLE_MOUNTABLE}))[0]
    tri = m.vertices[m.faces[fi]]
    cen = tri.mean(1)
    nrm = m.normals[fi]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    zi = m.face_zone[fi].astype(int) if len(m.face_zone) else np.full(len(fi), -1)
    kinds = np.array([m.zones[z].kind if 0 <= z < len(m.zones) else '' for z in zi])
    names = np.array([m.zones[z].name if 0 <= z < len(m.zones) else '' for z in zi])
    ok = (kinds != 'forbidden') & (cen[:, 2] > 0.25) & (nrm[:, 2] > -0.95)
    allowed = [z for z in m.zones if z.kind == 'allowed']
    if zones == 'allowed' and allowed:
        ok &= kinds == 'allowed'
    return fi[ok], cen[ok], nrm[ok], area[ok], names[ok]


def _spread(points, weights, k, min_gap):
    """Up to k points at least min_gap apart, heaviest first (a greedy farthest-point pick)."""
    order = np.argsort(-weights)
    out = []
    for i in order:
        if all(np.linalg.norm(points[i] - points[j]) >= min_gap for j in out):
            out.append(i)
            if len(out) >= k:
                break
    return out


def _aim_at(pos, target):
    from ..core.mathutil import yaw_pitch_from_dir
    d = np.asarray(target, float) - np.asarray(pos, float)
    yaw, pitch = yaw_pitch_from_dir(d)
    return float(yaw), float(np.clip(pitch, 10.0, 80.0))


def _resolve_unit(p, module, lens):
    """(module id, lens id) for a new camera: the named ones, else the module and lens most of the rig uses."""
    T = _T()
    cams = p.variant.cameras
    mod = None
    if module is not None:
        t = str(module).strip().lower()
        hits = [m for m in p.modules.values() if m.id.lower() == t or m.name.lower() == t] or \
            [m for m in p.modules.values() if t and (t in m.id.lower() or t in m.name.lower())]
        if len(hits) != 1:
            raise ValueError(f'no single camera module matches {module!r}; modules: ' + ', '.join(sorted(p.modules)))
        mod = hits[0].id
    if lens is not None:
        lz = T._lens(p, lens, mod or '')
        if lz is None:
            raise ValueError(f'no single lens matches {lens!r}; lens ids: ' + ', '.join(sorted(p.lenses)))
        if mod is None:
            mod = lz.module_id if lz.module_id in p.modules else None
        lens_id = lz.id
    else:
        lens_id = None
    if mod is None:
        ids = [c.module_id for c in cams if c.module_id in p.modules]
        mod = max(set(ids), key=ids.count) if ids else next(iter(p.modules))
    if lens_id is None:
        ids = [c.lens_id for c in cams if c.module_id == mod and c.lens_id in p.lenses] or \
            [l.id for l in p.lenses.values() if l.module_id == mod] or list(p.lenses)
        lens_id = max(set(ids), key=ids.count)
    return mod, lens_id


def _free_tag(v, base):
    used = v.tags()
    if base not in used:
        return base
    return next((f'{base}{i}' for i in range(2, 100) if f'{base}{i}' not in used), base + '+')


def new_camera(p, change):
    """The Camera an add change describes (tag, name, location, slot and port free in p's active variant), or raise
    ValueError. change: {'camera': tag, 'add': True, 'x', 'y', 'z', 'yaw', 'pitch', 'roll', 'module', 'lens'}."""
    from ..core.project import Camera
    from ..ui import ops
    v = p.variant
    try:
        x, y, z = (float(change[k]) for k in ('x', 'y', 'z'))
        yaw, pitch = float(change.get('yaw', 0.0)), float(change.get('pitch', 45.0))
        roll = float(change.get('roll', 0.0) or 0.0)
    except (KeyError, TypeError, ValueError):
        raise ValueError('an added camera needs x, y, z (m) and yaw, pitch (deg) as numbers') from None
    if not all(math.isfinite(t) for t in (x, y, z, yaw, pitch, roll)):
        raise ValueError('an added camera\'s pose must be finite')
    if not (0.05 <= z <= 5.0 and abs(x) <= 20 and abs(y) <= 20 and -90 <= pitch <= 90):
        raise ValueError('an added camera must sit within 20 m of the origin at 0.05–5 m height, pitch within ±90°')
    mod, lens = _resolve_unit(p, change.get('module'), change.get('lens'))
    tag = str(change.get('camera') or '').strip()[:8] or 'C'
    tag = _free_tag(v, tag)
    c = Camera(tag=tag, name=str(change.get('name') or f'Camera {tag}')[:40],
               location=str(change.get('location') or '')[:80], slot=v.free_slot(),
               x=round(x, 4), y=round(y, 4), z=round(z, 4), yaw=round(_wrap(yaw), 2), pitch=round(pitch, 2),
               roll=round(_wrap(roll), 2), module_id=mod, lens_id=lens, port=ops._free_port(v))
    return c


def apply_add(p, change):
    """Add the camera (and with mirror_partner its mirrored partner) an add change describes to p's active variant in
    place: for tools.apply_changes. Returns (rows, added cameras, errors) with rows in apply_changes' row format."""
    from ..core.project import MAX_CAMERAS
    from ..ui import ops
    v = p.variant
    pair = bool(change.get('mirror_partner', False))
    if len(v.cameras) + (2 if pair else 1) > MAX_CAMERAS:
        return [], [], [f'up to {MAX_CAMERAS} cameras (the recorder\'s GMSL ports): the rig has {len(v.cameras)}']
    try:
        c = new_camera(p, change)
    except ValueError as e:
        return [], [], [str(e)]
    v.cameras.append(c)
    made = [c]
    if pair:
        m = ops.mirrored_copy(v, c)
        v.cameras.append(m)
        made.append(m)
    rows = [{'kind': 'add', 'camera': x.tag, 'slot': x.slot, 'prop': 'added', 'before': '',
             'after': f'({x.x:.3f}, {x.y:.3f}, {x.z:.3f}) m, yaw {x.yaw:.1f}°, pitch {x.pitch:.1f}°, '
                      f'{p.lenses[x.lens_id].name if x.lens_id in p.lenses else x.lens_id}',
             'unit': '', 'mirrored_from': c.tag if x is not c else None} for x in made]
    return rows, made, []


def _gap_masks(cams_data, P, need):
    """(blind mask, below-need mask, best density) of the existing cameras on the points P."""
    from ..core.analysis import eval_points
    if cams_data:
        VD = [eval_points(P, c, intr, cube)[:2] for c, intr, cube in cams_data]
        V = np.stack([v for v, _ in VD]).any(0)
        D = np.stack([d for _, d in VD]).max(0)
    else:
        V, D = np.zeros(len(P), bool), np.zeros(len(P), np.float32)
    return ~V, D < need, D


def _check_add(tc, changes):
    """The analysis of the snapshot with the added camera(s) next to the snapshot's: the effect rows of
    tools._simulate (same metrics and issue counts)."""
    T = _T()
    t0 = time.perf_counter()
    base, trial = tc.project.snapshot(), tc.project.snapshot()
    rows, errors = [], []
    for ch in changes:
        r, _, e = apply_add(trial, ch)
        rows += r
        errors += e
    if errors:
        return None, {'error': '; '.join(errors)}
    r0, i0 = T._evaluate(tc, base)
    r1, i1 = T._evaluate(tc, trial)
    if r0 is None or r1 is None:
        return None, {'error': 'The analysis cannot run (no vehicle or ROI).'}
    sev = lambda lst: {s: sum(1 for i in lst if i.severity == s) for s in ('critical', 'serious', 'warning')}  # noqa
    eff = {key: {'before': _r(float(r0.metrics.get(key, 0.0)), 3), 'after': _r(float(r1.metrics.get(key, 0.0)), 3)}
           for key, _, _, _ in T.METRICS}
    return rows, {'effect': eff, 'issues': {'before': sev(i0), 'after': sev(i1),
                                            'new': [i.message for i in i1 if i.message not in {j.message for j in i0}][:6]},
                  'seconds': _r(time.perf_counter() - t0, 2)}


def add_camera_options(tc, where=None, module=None, lens=None, objective=None, min_density=None, pair=None,
                       zones='allowed', options=3, budget_s=30, **_):
    """Where one more camera (or a mirrored pair) would close the largest blind region or density gap."""
    T = _T()
    from ..core.analysis import eval_points
    from ..core.lens import resolve
    from ..core.project import MAX_CAMERAS
    from ..ui import ops
    t0 = time.perf_counter()
    p = tc.project.snapshot()
    v = p.variant
    if where is not None and where not in WHERE:
        return {'error': f'where must be one of {", ".join(WHERE)}'}
    if objective is not None and objective not in ADD_OBJECTIVES:
        return {'error': f'objective must be one of {", ".join(ADD_OBJECTIVES)}'}
    if zones not in ('allowed', 'any'):
        return {'error': "zones must be 'allowed' or 'any'"}
    if pair is not None and not isinstance(pair, bool):
        return {'error': 'pair must be true or false'}
    try:
        need = _num(min_density, 'min_density', 0.5, 80.0, float(p.analysis.min_density))
        budget = _num(budget_s, 'budget_s', 5.0, BUDGET_MAX, 30.0)
        k_opt = int(_num(options, 'options', 1, 5, 3))
        mod, lens_id = _resolve_unit(p, module, lens)
    except ValueError as e:
        return {'error': str(e)}
    room = MAX_CAMERAS - len(v.cameras)
    if room < 1:
        return {'error': f'the rig already has {len(v.cameras)} cameras, the most the recorder takes '
                         f'({MAX_CAMERAS} GMSL ports); optimise_rig can drop one'}
    base = T._base(tc)
    vg = tc.engine.vehicle_geom(p)
    if base is None or vg is None:
        return {'error': 'The analysis cannot run (no vehicle or ROI).'}
    t_an = max(0.4, time.perf_counter() - t0)
    cell = max(p.analysis.cell_m, 0.05)
    grid = tc.engine.roi_grid(p, cell)
    P = grid.points(0.0)[grid.roi]
    a = grid.cell ** 2
    cams = [c for c in v.cameras if c.visible]
    data = [(c, resolve(p.module_for(c), p.lens_for(c)), tc.engine.cube(c.pos())) for c in cams]
    blind, low, D0 = _gap_masks(data, P, need)
    blind_m2, low_m2 = float(blind.sum() * a), float(low.sum() * a)
    if objective is None:
        objective = 'blind' if blind_m2 >= 0.02 else 'density'
    gap = blind if objective == 'blind' else low
    if not gap.any():
        return {'result': ('no gap to close: every ground cell is seen' if objective == 'blind' else
                           f'no gap to close: every ground cell is seen at {need:g} px/cm or more'),
                'current': {'blind_m2': _r(blind_m2, 3), f'below_{need:g}_px_cm_m2': _r(low_m2, 3)},
                'next': 'an extra camera would only add redundancy; say so'}
    G = P[gap]
    fi, cen, nrm, area, znames = _mount_faces(vg, zones)
    mount_ok = set(int(f) for f in fi)                  # a mirrored partner must land on one of these too
    meas = p.vehicle.measured()
    xf, xr = meas.get('front_x', 3.0), meas.get('rear_x', -0.8)
    sel = np.ones(len(fi), bool)
    if where == 'front':
        sel = cen[:, 0] > xf - 0.6
    elif where == 'rear':
        sel = cen[:, 0] < xr + 0.6
    elif where == 'left':
        sel = cen[:, 1] > 0.25
    elif where == 'right':
        sel = cen[:, 1] < -0.25
    elif where == 'roof':
        sel = (nrm[:, 2] > 0.6) & (cen[:, 2] > cen[:, 2].max() - 0.3) if len(cen) else sel
    if cams:                                    # not on top of a camera already there
        C = np.array([c.pos() for c in cams])
        sel &= np.min(np.linalg.norm(cen[:, None, :] - C[None], axis=2), axis=1) > 0.15
    fi, cen, nrm, area, znames = fi[sel], cen[sel], nrm[sel], area[sel], znames[sel]
    if not len(fi):
        msg = 'no mountable surface' + (f' at the {where}' if where and where != 'any' else '')
        if zones == 'allowed':
            msg += ' inside the allowed zones (zones="any" also tries the rest of the mountable body)'
        return {'error': msg}
    # weigh each face by the gap it faces (gap cells within 4 m in front of its surface)
    Gs = G[::max(1, len(G) // 3000)]
    w = np.zeros(len(fi))
    for i in range(len(fi)):
        d = Gs - cen[i]
        dist = np.linalg.norm(d, axis=1)
        near = ((d @ nrm[i]) > 0.0) & (dist < 4.0)
        w[i] = float(np.sum(near / np.maximum(dist, 0.5)))
    if not (w > 0).any():
        return {'error': 'no mountable surface faces the gap' +
                         (' inside the allowed zones (zones="any" tries the rest of the body)' if zones == 'allowed' else '')}
    picks = _spread(cen, w * np.sqrt(area + 1e-6), 40, 0.12)
    picks = [i for i in picks if w[i] > 0]
    intr = resolve(p.modules[mod], p.lenses[lens_id])
    std = 0.012                                   # the default standoff (m)
    t_end = t0 + budget - (k_opt * 2) * (2.2 * t_an + 0.25) - 0.5
    t_end = max(t_end, t0 + budget * 0.4)

    def make(pos, yaw, pitch):
        from ..core.project import Camera
        return Camera(x=float(pos[0]), y=float(pos[1]), z=float(pos[2]), yaw=float(yaw), pitch=float(pitch),
                      module_id=mod, lens_id=lens_id)

    def score_cams(news):
        """(gap left on the objective's mask, the other mask's) with the new cameras added."""
        Vn, Dn = ~blind, D0.copy()
        for c, cube in news:
            vv, dd = eval_points(P, c, intr, cube)[:2]
            Vn = Vn | vv
            Dn = np.maximum(Dn, dd)
        bl, lo_ = float((~Vn).sum() * a), float((Dn < need).sum() * a)
        return (bl, lo_) if objective == 'blind' else (lo_, bl)

    scored = []
    for n_done, i in enumerate(picks):
        if tc.cancelled():
            return {'error': 'cancelled'}
        if time.perf_counter() > t_end:
            break
        pos = cen[i] + nrm[i] * std
        cube = tc.engine.cube(pos)
        near = G[(np.linalg.norm(G - pos, axis=1) < 4.0) & ((G - pos) @ nrm[i] > 0)]
        aims = []
        if len(near):
            aims.append(_aim_at(pos, near.mean(0)))
        aims.append(ops.default_aim(p, pos, nrm[i]))
        best = None
        for yaw, pitch in aims:
            s = score_cams([(make(pos, yaw, pitch), cube)])
            if best is None or s < best[0]:
                best = (s, yaw, pitch)
        # refine the aim: a small pattern search on yaw and pitch
        s, yaw, pitch = best
        step = [12.0, 8.0]
        while step[0] >= 1.5 and time.perf_counter() < t_end:
            moved = False
            for dy, dp in ((step[0], 0), (-step[0], 0), (0, step[1]), (0, -step[1])):
                cand = (yaw + dy, float(np.clip(pitch + dp, 10.0, 80.0)))
                sc = score_cams([(make(pos, *cand), cube)])
                if sc < s:
                    s, (yaw, pitch), moved = sc, cand, True
                    break
            if not moved:
                step = [step[0] / 2, step[1] / 2]
        scored.append({'i': i, 'pos': pos, 'yaw': _wrap(yaw), 'pitch': pitch, 's': s, 'pair': False})
        tc.status(f'Looking for a mount: {n_done + 1} of {len(picks)} spots tried · '
                  f'best leaves {min(x["s"][0] for x in scored):.3f} m² of the gap')
    if not scored:
        return {'error': 'the budget ran out before a mount was tried; give a larger budget_s'}
    scored.sort(key=lambda r: r['s'])
    # a mirrored pair for the best side spots, when the rig has room
    want_pair = (room >= 2) if pair is None else (pair and room >= 2)
    if want_pair:
        for r in list(scored[:6]):
            if abs(r['pos'][1]) < 0.25 or time.perf_counter() > t_end + 2 * t_an:
                continue
            mpos = r['pos'] * np.array([1.0, -1.0, 1.0])
            q, f, d = vg.occ_mesh.closest_point(mpos, vg.occ_mesh.occluder_faces())
            if f < 0 or d > 0.03 or int(f) not in mount_ok:   # not symmetric there, or not mountable on that side
                continue
            s = score_cams([(make(r['pos'], r['yaw'], r['pitch']), tc.engine.cube(r['pos'])),
                            (make(mpos, -r['yaw'], r['pitch']), tc.engine.cube(mpos))])
            scored.append({**r, 's': s, 'pair': True})
        scored.sort(key=lambda r: r['s'])
    gap0 = (blind_m2, low_m2) if objective == 'blind' else (low_m2, blind_m2)
    picked, out = [], []
    for r in scored:
        if len(out) >= k_opt or tc.cancelled():
            break
        if r['s'][0] >= gap0[0] - 1e-9:
            continue
        if any(np.linalg.norm(r['pos'] - q['pos']) < 0.3 and r['pair'] == q['pair'] for q in picked):
            continue
        if out and time.perf_counter() > t0 + budget - 2.2 * t_an - 0.25:
            break
        tag, name, loc = ops.location_tag(p, r['pos'], nrm[r['i']])
        tag = _free_tag(v, tag)                                     # never an existing camera's or sensor's tag
        ch = {'camera': tag, 'add': True, 'x': round(float(r['pos'][0]), 4), 'y': round(float(r['pos'][1]), 4),
              'z': round(float(r['pos'][2]), 4), 'yaw': round(r['yaw'], 1), 'pitch': round(r['pitch'], 1),
              'roll': 0.0, 'module': mod, 'lens': lens_id, 'mirror_partner': r['pair'], 'name': name,
              'location': loc}
        rows, rep = _check_add(tc, [ch])
        if rows is None:
            continue
        picked.append(r)
        eff = rep['effect']
        out.append({'option': chr(ord('A') + len(out)), 'change': ch,
                    'adds': [row['camera'] for row in rows],
                    'mount': {'surface': str(znames[r['i']]) or 'body', 'location': loc,
                              'normal': [_r(t, 2) for t in nrm[r['i']]]},
                    'on_search_grid': {'gap_m2': [_r(gap0[0], 3), _r(r['s'][0], 3)],
                                       'other_gap_m2': [_r(gap0[1], 3), _r(r['s'][1], 3)]},
                    'metrics': {k: [x['before'], x['after']] for k, x in eff.items()},
                    'issues': rep['issues'],
                    'trade_off': ''})
    for row in out:
        m = row['metrics']
        n_add = len(row['adds'])
        row['trade_off'] = (f"+{n_add} camera{'s' * (n_add > 1)} ({', '.join(row['adds'])}, {row['mount']['location']}): "
                            f"blind {m['blind_m2'][0]:.3f}→{m['blind_m2'][1]:.3f} m², above minimum density "
                            f"{m['min_ok_pct'][0]:.1f}→{m['min_ok_pct'][1]:.1f} %, minimum density "
                            f"{m['min_density'][0]:.2f}→{m['min_density'][1]:.2f} px/cm")
    gap_name = 'blind ground' if objective == 'blind' else f'ground below {need:g} px/cm (blind included)'
    try:
        supported = not _T().apply_changes(tc.project.snapshot(), [dict(out[0]['change'])])[3] if out else False
    except Exception:
        supported = False
    res = {'objective': objective, 'gap': {'what': gap_name, 'm2': _r(gap0[0], 3)},
           'unit': {'module': p.modules[mod].name, 'lens': p.lenses[lens_id].name},
           'zones': zones if any(z.kind == 'allowed' for z in vg.occ_mesh.zones) else 'any (the vehicle has no '
                                                                                        'allowed zones)',
           'options': out,
           'search': {'spots_tried': len([s for s in scored if not s['pair']]), 'pairs_tried': len([s for s in scored if s['pair']]),
                      'grid_cell_m': cell, 'seconds': _r(time.perf_counter() - t0, 1)},
           'limits': ('A spot is a surface point with the default 12 mm standoff, aimed at the gap; the search scores '
                      'the ground without the new camera\'s housing, the re-check at the project\'s resolution '
                      'includes it. Cable routes and ports are checked by the issues after adding.')}
    if not out:
        res['result'] = f'no mount closes any of the {gap_name}' + (f' at the {where}' if where else '')
    if out:
        res['next'] = ('offer the chosen option with propose_changes(changes=[option.change]); nothing changes until '
                       'the user accepts it' if supported else
                       'describe the option (where, aim, unit, effect): the proposal card cannot add cameras yet, so '
                       'the user places it with the Place camera tool at the given position and aim')
    return res


# =========================================================================== registry
_RANGES = {'type': 'object', 'properties': {
    'position_m': {'type': 'number', 'description': 'how far a camera may slide along the body (±m, at most 0.5)'},
    'yaw_deg': {'type': 'number', 'description': '± degrees of yaw around the current aim (at most 45)'},
    'pitch_deg': {'type': 'number', 'description': '± degrees of pitch around the current aim (at most 30)'}}}


def _tools():
    from .tools import Tool
    return [
        Tool('optimise_rig', 'Optimise the whole rig',
             'Design the camera layout as a whole: an evolutionary search over every free camera\'s slide along the '
             'body, yaw and pitch (and, when count allows fewer cameras, which to drop) against an objective and '
             'stated targets, scored under mounting tolerance. Returns up to 5 complete rig options on the Pareto '
             'front (blind area, minimum density, camera count), each as changes ready for propose_changes, '
             're-checked at the project\'s resolution, with whether it meets the targets and its trade-off. Use it '
             'for layout, camera count or rig-wide targets ("4-6 cameras, at least 3 px/cm everywhere, least blind '
             'ground"); for one camera use optimise_camera. Locked cameras stay; mirrored pairs stay mirrored. Takes '
             'up to budget_s seconds. Nothing changes until the user accepts a proposal.',
             {'cameras': {'type': 'array', 'items': {'type': 'string'},
                          'description': 'tags the search may move or drop (default every unlocked visible camera; '
                                         'a mirrored partner comes along); the others stay as they are'},
              'vary': {'type': 'array', 'items': {'type': 'string', 'enum': list(RIG_VARY)},
                       'description': 'what may change (default all three)'},
              'ranges': _RANGES,
              'objective': {'type': 'string', 'enum': list(OBJECTIVES),
                            'description': 'blind: least blind ground (default); cameras: fewest cameras; density: '
                                           'highest minimum density'},
              'min_density': {'type': 'number', 'description': 'target: px/cm every seen ground cell keeps'},
              'max_blind_m2': {'type': 'number', 'description': 'target: m² of blind ground at most'},
              'count': {'type': 'array', 'items': {'type': 'integer'},
                        'description': '[min, max] cameras (default: all of them; objective cameras: 1 to all)'},
              'under_tolerance': {'type': 'boolean', 'description': 'score each rig at its worst under random '
                                                                    'mounting errors (default true)'},
              'tolerance_deg': {'type': 'number', 'description': 'mounting error ± degrees (default 5)'},
              'tolerance_mm': {'type': 'number', 'description': 'mounting error ± mm (default 10)'},
              'critical': {'type': 'string', 'enum': list(CRITICAL),
                           'description': 'critical targets: off, require every one resolved, or weigh them in'},
              'options': {'type': 'integer', 'description': 'how many options to return, 1-5 (default 4)'},
              'budget_s': {'type': 'number', 'description': 'seconds, 5-90 (default 30)'}},
             optimise_rig, 'compute'),
        Tool('add_camera_options', 'Find where to add a camera',
             'Where one more camera (or a mirrored pair) would close the largest blind region (objective blind) or '
             'the ground below the minimum density (objective density): mountable spots on the body (inside the '
             'allowed zones unless zones is any), each aimed at the gap, scored against the cameras already there, '
             'the best re-checked at the project\'s resolution with the camera added to a copy. Returns up to 5 '
             'options with the add change, mount, aim, unit and effect. Use it when moving the cameras cannot close '
             'a gap (optimise_rig) or the user asks where to add one. Nothing changes until the user accepts.',
             {'where': {'type': 'string', 'enum': list(WHERE), 'description': 'part of the body to look at '
                                                                              '(default anywhere)'},
              'module': {'type': 'string', 'description': 'camera module id or name (default the rig\'s usual one)'},
              'lens': {'type': 'string', 'description': 'lens id or name (default the rig\'s usual one)'},
              'objective': {'type': 'string', 'enum': list(ADD_OBJECTIVES),
                            'description': 'default blind when there is blind ground, else density'},
              'min_density': {'type': 'number', 'description': 'px/cm for the density gap (default the project\'s)'},
              'pair': {'type': 'boolean', 'description': 'also try mirrored pairs (default when two ports are free)'},
              'zones': {'type': 'string', 'enum': ['allowed', 'any']},
              'options': {'type': 'integer', 'description': '1-5 (default 3)'},
              'budget_s': {'type': 'number', 'description': 'seconds, 5-90 (default 30)'}},
             add_camera_options, 'compute'),
    ]


def register(T=None):
    """Add TOOLS_RIG to tools.TOOLS and tools.BY_NAME once (idempotent): for when this module is imported before
    tools has finished loading its own registry."""
    T = T or _T()
    have = {t.name for t in T.TOOLS}
    for t in TOOLS_RIG:
        if t.name not in have:
            T.TOOLS.append(t)
        T.BY_NAME[t.name] = T.BY_NAME.get(t.name, t)


TOOLS_RIG = _tools()

# tools.py imports this module at its end (TOOLS += TOOLS_RIG). Imported first instead, this module's
# `from .tools import Tool` runs tools.py to its end, where that import finds this module half loaded: register then.
import sys as _sys                                                     # noqa: E402
_tm = _sys.modules.get(__package__ + '.tools')
if (_tm is not None and not getattr(getattr(_tm, '__spec__', None), '_initializing', False)
        and hasattr(_tm, 'BY_NAME') and 'optimise_rig' not in _tm.BY_NAME):
    register(_tm)
del _sys, _tm
