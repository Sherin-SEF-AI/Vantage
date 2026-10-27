"""Checks of the vehicle models: the suspension's static solve and load transfer against hand calculations, the road
geometry and the horizon."""
from __future__ import annotations

import math

import numpy as np

from .suite import Outcome, check

L_WB, T_TR = 2.5, 1.5                                       # the hand calculations' wheelbase and track, m
XY = {'FL': (L_WB, T_TR / 2), 'FR': (L_WB, -T_TR / 2), 'RL': (0.0, T_TR / 2), 'RR': (0.0, -T_TR / 2)}


def _series(a, b):
    """Two springs in series, N/mm."""
    return a * b / (a + b)


def _square(k_w=25.0, k_t=200.0, arb=0.0, kerb=300.0):
    """Four identical linear corners (wheel rate k_w, tyre k_t, N/mm), no bump stops."""
    from ..core import suspension as SU
    st = SU.SuspensionSettings()
    for c in SU.CORNERS:
        st.corners[c] = SU.Corner(spring_rate=k_w, motion_ratio=1.0, tyre_rate=k_t, kerb_load=kerb, bump_gap=70.0,
                                  bump_rate=0.0, bump_curve=[])
    st.arb_front = st.arb_rear = arb
    return st


# ============================================================================ suspension
@check('suspension.static_hand', 'suspension', 'Static loads against hand calculations',
       'largest relative difference of the corner drops, the tyre forces, the pitch and the roll the static solve '
       'gives from hand calculations: a load over the front axle (each front corner W/2 on its spring and tyre in '
       'series, pitch atan(drop / L)); a load to the left, with and without an anti-roll bar (heave W/4 on the series '
       'rate, ±W·y/(2T) per wheel on the wheel rate plus the bar, roll atan(2a / T)); a bump stop beyond its gap '
       '(gap + (F − k·gap)/(k + k_bump))',
       'springs in series, force and moment balance, written out in the check', 1e-9, 'relative',
       refs=('gillespie1992', 'milliken1995'))
def _static_hand(ctx):
    from ..core import suspension as SU
    G = SU.G
    e = {}
    st = _square(k_w=22.0, k_t=180.0)
    m = 150.0
    s = SU.solve(st, XY, [('front', m, L_WB, 0.0, 0.5)])
    drop = m * G / 2 / _series(22.0, 180.0)
    e['front load: front drop'] = abs(s.corners['FL'].drop_mm - drop) / drop
    e['front load: rear drop (mm/drop)'] = abs(s.corners['RR'].drop_mm) / drop
    want = math.degrees(math.atan(drop / 1000 / L_WB))
    e['front load: pitch'] = abs(s.pitch_deg - want) / want
    for arb in (0.0, 15.0):
        st = _square(25.0, 200.0, arb=arb)
        m, y0 = 90.0, 0.45
        s = SU.solve(st, XY, [('left', m, L_WB / 2, y0, 0.5)])
        W = m * G
        heave = W / 4 / _series(25.0, 200.0)
        dF = W * y0 / (2 * T_TR)
        a = dF / _series(25.0 + arb, 200.0)
        for c, sign in (('FL', 1), ('RL', 1), ('FR', -1), ('RR', -1)):
            e[f'left load, bar {arb:g}: {c} drop'] = abs(s.corners[c].drop_mm - (heave + sign * a)) / (heave + a)
            e[f'left load, bar {arb:g}: {c} force'] = abs(s.corners[c].added_n - (W / 4 + sign * dF)) / (W / 4 + dF)
        roll = -math.degrees(math.atan(2 * a / T_TR / 1000))
        e[f'left load, bar {arb:g}: roll'] = abs(s.roll_deg - roll) / abs(roll)
    k_w, k_b, gap = 25.0, 400.0, 20.0
    st = _square(k_w, k_t=1e4 / 2)
    for c in SU.CORNERS:
        st.corners[c].bump_gap, st.corners[c].bump_rate, st.corners[c].bump_curve = gap, k_b, []
    F = 400.0 * G / 4
    s = SU.solve(st, XY, [('l', 400.0, L_WB / 2, 0, 0.5)])
    want = gap + (F - k_w * gap) / (k_w + k_b)
    e['bump stop: travel'] = abs(s.corners['FL'].travel_mm - want) / want
    return Outcome(max(e.values()), e, ok=s.converged)


@check('suspension.balance', 'suspension', 'The static solve balances force and moments',
       'largest relative residual of the vertical force and of the moments about x and y (Σ tyre forces against Σ '
       'weights) and of the four corners\' drops from one plane, for 12 random sets of loads on typical corners '
       'with a measured load–travel curve at one corner',
       'static equilibrium of a rigid body on four supports', 1e-6, 'relative', refs=('gillespie1992',))
def _balance(ctx):
    from ..core import suspension as SU
    G = SU.G
    st = SU.typical()
    st.corners['FL'].curve = [[2500.0, -30.0], [3600.0, 0.0], [5200.0, 30.0], [8000.0, 50.0]]
    st.corners['FL'].kerb_load = 3600.0 / G
    rng = ctx.rng('suspension.balance')
    xy = {'FL': (2.6, 0.74), 'FR': (2.6, -0.74), 'RL': (0.0, 0.72), 'RR': (0.0, -0.72)}
    X = np.array([xy[c][0] for c in SU.CORNERS])
    Y = np.array([xy[c][1] for c in SU.CORNERS])
    worst, conv = 0.0, True
    for _ in range(12):
        loads = [(f'm{i}', float(rng.uniform(5, 160)), float(rng.uniform(-0.8, 3.0)), float(rng.uniform(-0.6, 0.6)),
                  0.5) for i in range(int(rng.integers(1, 6)))]
        s = SU.solve(st, xy, loads)
        conv &= s.converged
        W = np.array([m * G for _, m, *_ in loads])
        F = np.array([s.corners[c].added_n for c in SU.CORNERS])
        Wt = W.sum()
        r = [abs(F.sum() - Wt) / Wt,
             abs((F * X).sum() - sum(w * l[2] for w, l in zip(W, loads))) / (Wt * L_WB),
             abs((F * Y).sum() - sum(w * l[3] for w, l in zip(W, loads))) / (Wt * T_TR)]
        d = np.array([s.corners[c].drop_mm for c in SU.CORNERS])
        A = np.c_[np.ones(4), X, Y]
        res = d - A @ np.linalg.lstsq(A, d, rcond=None)[0]
        r.append(float(np.abs(res).max()) / max(float(np.abs(d).max()), 1e-9))
        worst = max(worst, max(r))
    return Outcome(worst, {'cases': 12}, ok=bool(conv))


@check('suspension.load_transfer', 'suspension', 'Brake dive and cornering roll against hand calculations',
       'largest relative difference of the corner drops and the pitch under 0.8 g braking (with 0, 30/10 and 100 % '
       'anti-dive/anti-lift: the linkage takes its share of ΔF = M·a·h/L off the springs) and of the drops and the '
       'roll at 0.5 g lateral (roll centres 0 and 0.2 m, anti-roll bar 0 and 12 N/mm: the geometric share '
       'M/2·a·rc/T through the links, the elastic share on the springs and bar), unsprung masses zero',
       'quasi-static load transfer with anti-dive and roll centres (Gillespie; Milliken), written out in the check',
       1e-6, 'relative', refs=('gillespie1992', 'milliken1995'))
def _transfer(ctx):
    from ..core import suspension as SU
    G = SU.G
    e = {}
    for ad, al in ((0.0, 0.0), (30.0, 10.0), (100.0, 100.0)):
        st = _square()
        dyn = SU.Dynamics(cg_height=0.55, anti_dive=ad, anti_lift=al, unsprung_kg=0.0)
        sv, tr = SU.dynamic_solve(st, XY, [], -0.8, 0.0, dyn)
        M = st.kerb_mass
        dF = M * 0.8 * G * 0.55 / L_WB
        front = (1 - ad / 100) * dF / 2 / 25.0 + dF / 2 / 200.0
        rear = -((1 - al / 100) * dF / 2 / 25.0 + dF / 2 / 200.0)
        e[f'braking {ad:g}/{al:g} %: front drop'] = abs(sv.corners['FL'].drop_mm - front) / abs(front)
        e[f'braking {ad:g}/{al:g} %: rear rise'] = abs(sv.corners['RR'].drop_mm - rear) / abs(rear)
        p = math.degrees(math.atan((front - rear) / 1000 / L_WB))
        e[f'braking {ad:g}/{al:g} %: pitch'] = abs(sv.pitch_deg - p) / abs(p)
    for arb, rc in ((0.0, 0.0), (12.0, 0.0), (0.0, 0.2)):
        st = _square(arb=arb)
        h = 0.55
        dyn = SU.Dynamics(cg_height=h, rc_front=rc, rc_rear=rc, unsprung_kg=0.0)
        sv, tr = SU.dynamic_solve(st, XY, [], 0.0, 0.5, dyn)
        M = st.kerb_mass
        a = 0.5 * G
        geo = M / 2 * a * rc / T_TR
        ela = M * a * (h - rc) / 2 / T_TR
        drop = ela / (25.0 + arb) + (geo + ela) / 200.0
        e[f'cornering bar {arb:g}, rc {rc:g}: drop'] = abs(sv.corners['FR'].drop_mm - drop) / drop
        roll = math.degrees(math.atan(2 * drop / 1000 / T_TR))
        e[f'cornering bar {arb:g}, rc {rc:g}: roll'] = abs(sv.roll_deg - roll) / roll
    return Outcome(max(e.values()), e)


# ============================================================================ road geometry
REF = {'front_x': 3.0, 'rear_x': -1.0, 'left_y': 0.9, 'right_y': -0.9, 'width': 1.8}
FEATS = [{'kind': 'ramp', 'grade': 7.0, 'start': 2.0, 'where': 'ahead'},
         {'kind': 'crest', 'radius': 150.0, 'grade': -9.0, 'start': 1.0, 'where': 'behind'},
         {'kind': 'crest', 'radius': 80.0, 'grade': 5.0, 'start': 4.0, 'where': 'ahead'},
         {'kind': 'garage', 'grade': -16.0, 'transition': 2.0, 'start': 1.5, 'where': 'ahead'},
         {'kind': 'bump', 'height': 0.09, 'length': 0.7, 'start': 1.2, 'where': 'behind'},
         {'kind': 'kerb', 'height': 0.13, 'offset': 0.6, 'side': 'left'},
         {'kind': 'kerb', 'height': 0.10, 'offset': 0.4, 'side': 'right'},
         {'kind': 'kerb', 'height': 0.15, 'offset': 3.0, 'side': 'ahead'},
         {'kind': 'crown', 'crown': 2.5, 'offset': 1.2}]


def _closed_form(f, x, y, face_m):
    """Each road primitive's height from its definition, written out again."""
    k = f['kind']
    if k == 'crown':
        return -f['crown'] / 100 * abs(y - f['offset'])
    if k == 'kerb' and f['side'] in ('left', 'right'):
        s = (y - (REF['left_y'] + f['offset'])) if f['side'] == 'left' else ((REF['right_y'] - f['offset']) - y)
        return f['height'] * min(max(s / face_m, 0.0), 1.0)
    side = f.get('where', f.get('side'))
    dist = f.get('start', f.get('offset'))
    s = x - (REF['front_x'] + dist) if side == 'ahead' else (REF['rear_x'] - dist) - x
    if s <= 0:
        return 0.0
    if k == 'ramp':
        return f['grade'] / 100 * s
    if k == 'kerb':
        return f['height'] * min(s / face_m, 1.0)
    if k == 'bump':
        return f['height'] * math.sin(math.pi * s / f['length']) ** 2 if s < f['length'] else 0.0
    if k == 'garage':
        g, L = f['grade'] / 100, f['transition']
        return g / 2 * s if s < L else g / 2 * L + g * (s - L)
    g, R = f['grade'] / 100, f['radius']                    # crest: the integral of a grade changing as s / R
    Lc = R * abs(g)
    return math.copysign(1, g) * s * s / (2 * R) if s < Lc else math.copysign(1, g) * Lc * Lc / (2 * R) + g * (s - Lc)


@check('terrain.primitives', 'terrain', 'Road primitives against their definitions',
       'largest difference of the road height (ramp, crest and sag vertical curves, garage ramp, speed bump, kerbs, '
       'crowned road) from its definition written out again, and of its gradient from central differences away from '
       'the kinks, at 400 random points per primitive',
       'each primitive\'s closed form (a vertical curve is the parabola of a grade changing linearly with distance) '
       'and a central difference with step 10⁻⁵ m', 1e-6, 'm (height) and m/m (gradient)', refs=('aashto2018',))
def _primitives(ctx):
    from ..core import terrain as TR
    rng = ctx.rng('terrain.primitives')
    worst, rows = 0.0, {}
    for i, f in enumerate(FEATS):
        ter = TR.Terrain([f], REF)
        x, y = rng.uniform(-25, 30, 400), rng.uniform(-6, 6, 400)
        h = ter.height(x, y)
        eh = float(np.abs(h - [_closed_form(f, a, b, TR.FACE_M) for a, b in zip(x, y)]).max())
        e = 1e-5
        gx, gy = ter.gradient(x, y)
        nx = (ter.height(x + e, y) - ter.height(x - e, y)) / (2 * e)
        ny = (ter.height(x, y + e) - ter.height(x, y - e)) / (2 * e)
        q = TR.Feature(f, REF)
        s = q.s(x, y)
        smooth = np.all([np.abs(s - b) > 1e-3 for b in q.breaks()], axis=0) & (np.abs(q.q(x, y)) > 1e-3)
        eg = float(max(np.abs(gx[smooth] - nx[smooth]).max(), np.abs(gy[smooth] - ny[smooth]).max()))
        rows[f"{f['kind']} {i}"] = {'height_m': eh, 'gradient': eg, 'smooth_points': int(smooth.sum())}
        worst = max(worst, eh, eg)
    x = np.linspace(-20, 20, 97)
    yy = np.linspace(-5, 5, 97)
    add = float(np.abs(TR.Terrain(FEATS, REF).height(x, yy) - sum(TR.Terrain([f], REF).height(x, yy)
                                                                  for f in FEATS)).max())
    return Outcome(max(worst, add), {'primitives': rows, 'superposition_m': add})


def _brute_clear(ter, C, P, n=4000):
    """A dense march along every line of sight: clear when no sample dips under the road (by more than the
    allowance) and the point faces the camera."""
    t = np.linspace(0, 1, n + 1)[1:-1]
    out = np.zeros(len(P), bool)
    for a in range(0, len(P), 100):
        d = P[a:a + 100] - C
        L = np.linalg.norm(d, axis=1)
        X, Y, Z = (C[k] + t[None] * d[:, k:k + 1] for k in range(3))
        far = (1 - t[None]) * L[:, None] > 0.012
        out[a:a + 100] = ((Z < ter.height(X, Y) - 0.002 - 0.001 * (1 - t[None]) * L[:, None]) & far).any(1)
    gx, gy = ter.gradient(P[:, 0], P[:, 1])
    v = C - P
    return ~out & (-gx * v[:, 0] - gy * v[:, 1] + v[:, 2] > 0)


@check('terrain.line_of_sight', 'terrain', 'Seeing the road past its own crests and kerbs',
       'share of road points (3000 per camera position, four positions, the primitives together) whose visibility '
       'the ray march (terrain.clear) decides differently from a dense brute-force march of 4000 samples per line of '
       'sight',
       'a brute-force march along every line of sight, written for the check', 1e-3, 'share of points',
       refs=('aashto2018',), cost='study')
def _line_of_sight(ctx):
    from ..core import terrain as TR
    ter = TR.Terrain(FEATS[:1] + FEATS[3:], REF)
    bad = tot = 0
    rows = {}
    for cam in ((3.2, 0.0, 0.62), (-0.9, 0.3, 0.55), (1.9, 0.95, 0.95), (1.0, -1.0, 0.3)):
        rng = ctx.rng('terrain.line_of_sight', abs(int(cam[0] * 100)))
        n = 1000 if ctx.quick else 3000
        P = np.c_[rng.uniform(-12, 20, n), rng.uniform(-5, 5, n), np.zeros(n)]
        P[:, 2] = ter.height(P[:, 0], P[:, 1])
        C = np.array(cam, float)
        C[2] += float(ter.height(C[0], C[1]))
        got = ter.clear(C, P)
        ref = _brute_clear(ter, C, P)
        rows[str(cam)] = {'hidden': int((~ref).sum()), 'differ': int((got != ref).sum())}
        bad += int((got != ref).sum())
        tot += n
    return Outcome(bad / tot, {'cameras': rows, 'points': tot},
                   note='A ray that grazes the road within the allowance may land either side of it.')


@check('terrain.crest_tangent', 'terrain', 'The road hidden beyond a crest',
       'difference of where the sample\'s front camera loses sight of the road beyond a crest (radius 250 m, falling '
       'to −10 %, 6 m ahead of the bumper; terrain.profile) from where the line from the lens tangent to the crest\'s '
       'parabola meets it (s² + 2as − 2Rz = 0)',
       'the tangent from the camera to the vertical curve, written out in the check', 0.3, 'm',
       refs=('aashto2018',), cost='study')
def _crest(ctx):
    from ..core import terrain as TR
    R, start = 250.0, 6.0
    q = ctx.sample()
    q.analysis.terrain = {'features': [{'kind': 'crest', 'radius': R, 'grade': -10.0, 'start': start}]}
    ter = TR.build(q)
    views = TR.Views(q, ctx.engine)
    views.cams = [v for v in views.cams if v[0].tag == 'F']
    if not views.cams:
        raise ValueError('the sample has no front camera F')
    pr = TR.profile(ter, views, ter.ref, 'ahead', 60.0)
    c = views.cams[0][0]
    a = ter.ref['front_x'] + start - c.x
    s_t = -a + math.sqrt(a * a + 2 * R * c.z)
    want = start + s_t
    return Outcome(abs(pr['seen_to_m'] - want), {'seen_to_m': pr['seen_to_m'], 'tangent_m': want,
                                                 'camera_height_m': c.z, 'profile_step_m': 0.25},
                   ok=bool(pr['hidden']), note='The profile samples the road every 0.25 m beyond 10 m.')


@check('terrain.horizon', 'terrain', 'The horizon in a rectilinear camera',
       'largest difference of the horizon row loadcases.horizon_at gives from c_y − f·tan(pitch) for camera and body '
       'pitches between −2° and 35°, and of its tilt from the roll (rolls ±2–3°)',
       'the vanishing line of the ground plane in a pinhole camera, written out in the check', 1e-6, 'px',
       refs=('hartley2004',))
def _horizon(ctx):
    from ..core import loadcases as LC
    from ..core.lens import Intrinsics
    from ..core.mathutil import BODY_TO_OPTICAL, rot_zyx
    f_mm, pix = 4.0, 0.003
    intr = Intrinsics(width=1920, height=1280, pixel_mm=pix, f_mm=f_mm, projection='rectilinear', kb=(),
                      cx=960.0, cy=640.0, max_theta=math.radians(80), focus_near=0.0, focus_far=math.inf)
    e = []
    for cam_pitch in (0.0, 10.0, 25.0):
        for body_pitch in (0.0, 1.5, -2.0):
            R = rot_zyx(0.0, body_pitch, 0.0) @ rot_zyx(0.0, cam_pitch, 0.0)
            h = LC.horizon_at(intr, BODY_TO_OPTICAL @ R.T)
            el = math.radians(cam_pitch + body_pitch)
            e.append(abs(h['v'] - (640.0 - f_mm / pix * math.tan(el))))
            e.append(abs(h['tilt_deg']))
    for roll in (2.0, -3.0):
        h = LC.horizon_at(intr, BODY_TO_OPTICAL @ rot_zyx(0.0, 0.0, roll).T)
        e.append(abs(abs(h['tilt_deg']) - abs(roll)))
    return Outcome(max(e), {'cases': len(e)})
