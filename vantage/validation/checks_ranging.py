"""Checks of the lidar and radar models (beams, rings, points on objects, the radar equation, the bumper fascia's
layer physics) and of cameras behind glass."""
from __future__ import annotations

import cmath
import math

import numpy as np

from .suite import Outcome, check, figure, rel_err, series


def _empty(*sensors):
    """A project without a vehicle: nothing blocks, the centre is the origin."""
    from ..core.project import Project
    p = Project()
    p.variant.sensors.extend(sensors)
    return p


def _lidar(model_id='ouster_os1_128', tag='L1', **pose):
    from ..core.project import Sensor
    return Sensor(kind='lidar', model_id=model_id, tag=tag, **pose)


# ============================================================================ lidar
@check('lidar.ring_radii', 'lidar', 'Ground rings of a level lidar',
       'largest relative difference of each channel\'s ground ring radius (sensors.ground_rings) from h / tan|ε| for '
       'every downward channel of the OS1-128 2 m above the ground; the front of a ring of a lidar pitched 5° down '
       'must lie at h / tan(|ε| + 5°) (to the 1° azimuth sampling)',
       'the cone of a beam at elevation ε cut by the ground plane, written out in the check', 1e-9, 'relative',
       refs=('rep103',))
def _ring_radii(ctx):
    from ..core import sensors as S
    from ..core.project import Project
    h = 2.0
    m = Project().sensor_models['ouster_os1_128']
    E = m.elevations()
    rings = dict(S.ground_rings(_lidar(x=0.0, y=0.0, z=h), m, 1.0))
    worst, n = 0.0, 0
    radii, want = [], []
    for ch, el in enumerate(E):
        if ch not in rings:
            continue
        r = np.hypot(rings[ch][:, 0], rings[ch][:, 1])
        ref = h / math.tan(math.radians(-el))
        worst = max(worst, float(rel_err(r, ref).max()))
        radii.append(float(r.mean()))
        want.append(ref)
        n += 1
    # tilted 5° down: the ring of the -10°-ish channel comes nearer in front
    k = int(np.argmin(np.abs(E + 10.0)))
    tilted = dict(S.ground_rings(_lidar(x=0.0, y=0.0, z=h, pitch=5.0), m, 1.0))
    ring = tilted[k]
    front = ring[np.argmin(np.abs(np.arctan2(ring[:, 1], ring[:, 0])))]
    tilt_ref = h / math.tan(math.radians(-E[k] + 5.0))
    tilt_err = abs(front[0] - tilt_ref) / tilt_ref
    fig = figure('Ground ring radius per channel, OS1-128 at 2 m', 'beam elevation, °', 'ring radius, m',
                 [series('h / tan|ε|', [E[c] for c in sorted(rings)], want),
                  series('sensors.ground_rings', [E[c] for c in sorted(rings)], radii, kind='points')])
    return Outcome(worst, {'channels_on_the_ground': n, 'height_m': h, 'tilted_front_m': float(front[0]),
                           'tilted_closed_form_m': tilt_ref, 'tilted_rel_error': tilt_err},
                   fig, ok=n > 10 and tilt_err < 2e-3)


@check('lidar.ground_points', 'lidar', 'Points a level lidar puts on the ground',
       'relative difference of the points per frame summed over the ground density map (lidar and radar coverage) '
       'from every downward channel within range times the columns per revolution, a level OS1-128 2 m up with '
       'nothing in the way and a map that holds every ring; the nearest ground point in every direction must be '
       'the lowest ring',
       'the beam count: channels reaching the ground within range × columns per revolution', 1e-6, 'relative',
       refs=('rep103',))
def _ground_points(ctx):
    from ..core import sensor_coverage as SC
    h = 2.0
    p = _empty(_lidar(x=0.0, y=0.0, z=h))
    st = SC.CoverageSettings.from_dict({'classes': [SC.DEFAULT_CLASSES[0]], 'grid_extent': 200.0, 'grid_cell': 0.25})
    res = SC.run(p, ctx.engine, settings=st)
    m = p.sensor_models['ouster_os1_128']
    E = m.elevations()
    x0, y0, cell = res.grid
    half = st.grid_extent / 2
    radii, inside = [], 0
    for e in E:
        if e >= 0:
            continue
        r = h / math.tan(math.radians(-e))
        slant = h / math.sin(math.radians(-e))
        if slant > m.range_m or slant < m.range_min_m:
            continue
        if r >= half - cell:
            raise ValueError('a ring in range does not fit the map: the reference would not be exact')
        radii.append(r)
    want = len(radii) * m.columns()
    got = float(res.density.sum()) * cell * cell
    li = res.lidars[0]
    near_err = float(np.max(rel_err(res.near, min(radii))))
    return Outcome(abs(got - want) / want, {'points_per_frame_on_the_ground': got, 'beam_count': want,
                                            'rings_in_range': len(radii), 'columns': m.columns(),
                                            'nearest_ground_rel_error': near_err, 'lowest_ring_m': min(radii)},
                   ok=near_err < 2e-3 and li.blocked == 0.0)


PHASES = 16          # azimuth phases of the scan the cast count is averaged over


def _points_cases(ctx, pitch):
    """The study's count against the phase-averaged cast for a car at 8–60 m, ahead and 35° to the side, of an
    OS1-128 2 m up pitched `pitch` degrees down: (worst relative difference, rows, channel counts agree)."""
    from ..core import sensor_coverage as SC, sensors as S
    car = SC.CoverageClass('Car', 1.5, 1.8, 0.0, 20.0, 10.0)
    worst, rows, chan_ok = 0.0, [], True
    s = _lidar(x=0.0, y=0.0, z=2.0, pitch=pitch)
    p = _empty(s)
    m = p.sensor_model_for(s)
    azs = S.scan_azimuths(m)
    step = float(azs[1] - azs[0])
    Rs = S.rotation(s)
    o = s.pos()
    for rng_m in (8.0, 20.0, 35.0, 60.0):
        for az in (0.0, 35.0):
            g = np.array([rng_m * math.cos(math.radians(az)), rng_m * math.sin(math.radians(az))])
            got = float(SC.object_points(p, ctx.engine, s, car, g[None])[0])
            n = np.r_[g - o[:2], 0.0]
            n /= np.linalg.norm(n)
            side = np.cross([0.0, 0.0, 1.0], n)
            casts, chans = [], set()
            for k in range(PHASES):
                D = S.directions(m.elevations(), azs + step * k / PHASES) @ Rs.T
                den = D @ n
                with np.errstate(divide='ignore', invalid='ignore'):
                    t = np.where(den > 1e-9, float(np.r_[g, 0.0] @ n - o @ n) / den, -1.0)
                X = o + D * t[..., None]
                hit = (t > 0) & (np.abs((X - np.r_[g, 0.0]) @ side) <= car.width / 2) & (X[..., 2] >= 0) \
                    & (X[..., 2] <= car.height) & (t <= m.range_m)
                casts.append(int(hit.sum()))
                chans.add(int(hit.any(axis=1).sum()))
            cast = float(np.mean(casts))
            ch_model = int(round(got / (math.degrees(2 * math.atan(car.width / 2 / rng_m)) / m.h_res)))
            chan_ok &= min(abs(ch_model - c) for c in chans) <= 1
            e = abs(got - cast) / cast
            rows.append({'range_m': rng_m, 'azimuth_deg': az, 'counted': round(got, 2), 'cast_mean': round(cast, 2),
                         'cast_range': [min(casts), max(casts)], 'channels_counted': ch_model,
                         'channels_cast': sorted(chans), 'rel_difference': round(e, 4)})
            worst = max(worst, e)
    return worst, rows, chan_ok


_POINTS_Q = ('relative difference between the points per frame the coverage study counts on a car (1.5 m × 1.8 m) '
             '8–60 m from an OS1-128 2 m up, ahead and 35° to the side, and the beams that meet the car\'s face '
             'when every beam of a revolution is cast, averaged over 16 phases of the scan\'s azimuths (one phase '
             'differs by up to a point per channel); the channels crossing the car must agree within one')
_POINTS_R = ('every beam (channel elevation × column azimuth, through the lidar\'s rotation) intersected with a '
             'vertical rectangle square to the line of sight, written for the check')


@check('lidar.points_level', 'lidar', 'Lidar points on an object: a level lidar', 'largest ' + _POINTS_Q, _POINTS_R,
       0.02, 'relative', refs=('rep103',))
def _points_level(ctx):
    worst, rows, ok = _points_cases(ctx, 0.0)
    return Outcome(worst, {'cases': rows, 'phases': PHASES}, ok=ok)


@check('lidar.points_tilted', 'lidar', 'Lidar points on an object: a lidar pitched 5° down',
       'largest ' + _POINTS_Q.split('; the channels')[0] + '; the lidar pitched 5° down', _POINTS_R, 0.10, 'relative',
       refs=('rep103',))
def _points_tilted(ctx):
    worst, rows, _ = _points_cases(ctx, 5.0)
    return Outcome(worst, {'cases': rows, 'phases': PHASES},
                   note='The study gives every channel that crosses the object its full angular width / h_res. A '
                        'pitched lidar\'s channel that crosses the top or bottom edge off its axis meets the face '
                        'over part of the width only (the edge is tilted in the lidar\'s frame), so the count runs '
                        'high there (and the channels it counts can differ by two from those cast): up to about 8 % '
                        'in these cases, against 1 % for a level lidar.')


# ============================================================================ radar
@check('radar.cross_section', 'radar', 'Radar range against the cross-section',
       'largest relative difference of the pedestrian\'s and cyclist\'s radar range from the car\'s scaled by '
       '(σ / σ_car)^¼, and of the car\'s from the zone\'s 100 m (slant to its centre), a corner radar with nothing '
       'in the way',
       'the radar equation: received power ∝ σ / R⁴, so the range at a fixed threshold ∝ σ^¼', 1e-4, 'relative',
       refs=('skolnik2001',))
def _radar_rcs(ctx):
    from ..core import sensor_coverage as SC
    from ..core.project import Sensor
    s = Sensor(kind='radar', model_id='radar_corner', tag='RD1', x=0.0, y=0.0, z=0.5)
    p = _empty(s)
    res = SC.run(p, ctx.engine)
    names = [c.name for c in res.classes]
    car = names.index('Car')
    zc = 0.75 - 0.5
    errs = {'car': abs(res.radar[car, 0] - math.sqrt(100.0 ** 2 - zc ** 2)) / 100.0}
    for name in ('Pedestrian', 'Cyclist'):
        k = names.index(name)
        rcs = res.classes[k].rcs
        errs[name.lower()] = abs(res.radar[k, 0] / res.radar[car, 0] - (rcs / res.classes[car].rcs) ** 0.25)
    return Outcome(max(errs.values()), {'errors': errs, 'ranges_m': {n: float(res.radar[i, 0])
                                                                     for i, n in enumerate(names)}})


def _airy(eps, tan, d, f_ghz, theta_deg, te=True):
    """|r|², |t|² of one slab in air from the interface Fresnel coefficients and the sum of its internal bounces."""
    from ..core import materials as M
    N = cmath.sqrt(eps * (1 - 1j * tan))
    th = math.radians(theta_deg)
    ci = math.cos(th)
    ct = cmath.sqrt(1 - (math.sin(th) / N) ** 2)
    if te:
        r01 = (ci - N * ct) / (ci + N * ct)
        t01, t10 = 2 * ci / (ci + N * ct), 2 * N * ct / (N * ct + ci)
    else:
        r01 = (N * ci - ct) / (N * ci + ct)
        t01, t10 = 2 * ci / (N * ci + ct), 2 * N * ct / (ct + N * ci)
    r10 = -r01
    ph = cmath.exp(-1j * 2 * math.pi / M.wavelength_m(f_ghz) * N * d * ct)
    t = sum(t01 * t10 * r10 ** (2 * k) * ph ** (2 * k + 1) for k in range(400))
    r = r01 + sum(t01 * t10 * r10 ** (2 * k + 1) * ph ** (2 * k + 2) for k in range(400))
    return abs(r) ** 2, abs(t) ** 2


@check('radar.slab', 'radar', 'One fascia layer against the sum of its reflections',
       'largest difference of the power reflectance and transmittance of a single dielectric slab (materials.stack_rt, '
       'the characteristic-matrix method) from the sum of its internal bounces, TE and TM, lossless and lossy, normal '
       'to 75° incidence at 76.5 GHz',
       'the Airy summation of the slab\'s multiple reflections with the Fresnel interface coefficients, written for '
       'the check', 1e-9, 'power ratio', refs=('macleod2010', 'bornwolf1999'))
def _radar_slab(ctx):
    from ..core import materials as M
    F = 76.5
    worst, rows = 0.0, []
    for eps, tan, d_mm, theta in ((2.75, 0.005, 3.0, 0.0), (2.75, 0.005, 3.0, 35.0), (6.31, 0.045, 2.1, 60.0),
                                  (3.2, 0.02, 1.7, 75.0), (2.3, 0.0, 2.5, 20.0)):
        r_te, t_te, r_tm, t_tm = M.stack_rt([(eps, tan, d_mm / 1000)], F, np.array([theta]))
        for te, r, t in ((True, r_te, t_te), (False, r_tm, t_tm)):
            R, T = _airy(eps, tan, d_mm / 1000, F, theta, te)
            e = max(abs(abs(r[0]) ** 2 - R), abs(abs(t[0]) ** 2 - T))
            rows.append({'eps': eps, 'tan_delta': tan, 'mm': d_mm, 'incidence_deg': theta, 'pol': 'TE' if te else 'TM',
                         'R': round(R, 6), 'T': round(T, 6), 'difference': e})
            worst = max(worst, e)
    return Outcome(worst, {'cases': rows})


@check('radar.thin_film', 'radar', 'Layer identities: half wave, quarter wave, Brewster, energy',
       'largest deviation from the closed forms: a lossless half-wave layer transmits everything; a quarter-wave '
       'layer reflects ((n²−1)/(n²+1))²; TM at Brewster\'s angle tan θ_B = n is not reflected; a lossless four-layer '
       'stack (substrate, primer, basecoat, clear coat) keeps R + T = 1 from 0° to 85°',
       'thin-film closed forms (Macleod; Born & Wolf)', 1e-9, 'power ratio', refs=('macleod2010', 'bornwolf1999'))
def _radar_thin_film(ctx):
    from ..core import materials as M
    F = 76.5
    lam = M.wavelength_m(F)
    n = math.sqrt(2.75)
    e = {}
    T, R, _, _ = M.transmission([(2.75, 0.0, lam / (2 * n))], F, np.array([0.0]))
    e['half wave: 1 − T'] = abs(1 - T[0])
    T, R, _, _ = M.transmission([(2.75, 0.0, lam / (4 * n))], F, np.array([0.0]))
    e['quarter wave: R'] = abs(R[0] - ((n * n - 1) / (n * n + 1)) ** 2)
    brew = math.degrees(math.atan(n))
    for d in (0.7e-3, 2.3e-3, 5.1e-3):
        _, _, r_tm, _ = M.stack_rt([(2.75, 0.0, d)], F, np.array([brew]))
        e[f'Brewster TM |r|, {d * 1000:g} mm'] = abs(r_tm[0])
    stack = [(2.75, 0.0, 3e-3), (2.8, 0.0, 5e-6), (43.0, 0.0, 15e-6), (3.5, 0.0, 40e-6)]
    th = np.linspace(0.0, 85.0, 18)
    r_te, t_te, r_tm, t_tm = M.stack_rt(stack, F, th)
    e['lossless stack: |R + T − 1|'] = float(max(np.abs(np.abs(r_te) ** 2 + np.abs(t_te) ** 2 - 1).max(),
                                                 np.abs(np.abs(r_tm) ** 2 + np.abs(t_tm) ** 2 - 1).max()))
    return Outcome(max(e.values()), {k: float(v) for k, v in e.items()})


EMILSSON = {'s21_db': (-2.34, -1.98), 's11_db': (-5.75, -4.86)}     # sample 11, the two measured values


@check('radar.painted_plate', 'radar', 'A painted bumper plate against a measurement',
       'distance of the modelled 77 GHz one-way transmission |S21| of a 3.416 mm talc-filled PP plate with the '
       'library\'s primer, a metallic basecoat (ε = 48.94, 15 µm) and a 40 µm clear coat, for substrate ε 2.9–3.1, '
       'from the range of Emilsson\'s two measurements of the plate (−2.34 and −1.98 dB); |S11| must lie within '
       '0.9 dB of its measured range (−5.75 to −4.86 dB)',
       'Emilsson\'s measured S21 and S11 of the painted plate (sample 11)', 0.4, 'dB', refs=('emilsson2017',))
def _radar_plate(ctx):
    from ..core import materials as M
    s21, s11 = [], []
    for eps in (2.9, 3.0, 3.1):
        stack = [(eps, 0.01, 3.416e-3), (2.8, 0.011, 5e-6), (48.94, 0.005 / 48.94, 15e-6), (3.5, 0.02, 40e-6)]
        r_te, t_te, _, _ = M.stack_rt(stack, 77.0, np.array([0.0]))
        s21.append(20 * math.log10(abs(t_te[0])))
        s11.append(20 * math.log10(abs(r_te[0])))

    def dist(vals, band):
        lo, hi = band
        return max(max(lo - v, v - hi, 0.0) for v in vals)
    d21, d11 = dist(s21, EMILSSON['s21_db']), dist(s11, EMILSSON['s11_db'])
    bare = M.stack_rt([(3.0, 0.01, 3.416e-3)], 77.0, np.array([0.0]))[1]
    return Outcome(d21, {'model_s21_db': [round(v, 3) for v in s21], 'model_s11_db': [round(v, 3) for v in s11],
                         'measured_s21_db': EMILSSON['s21_db'], 'measured_s11_db': EMILSSON['s11_db'],
                         's11_distance_db': d11, 'unpainted_s21_db': 20 * math.log10(abs(bare[0]))},
                   ok=d11 <= 0.9)


@check('radar.fascia_range', 'radar', 'A radar behind the bumper fascia loses range by the radar equation',
       'relative difference of the long-range radar RD1\'s range on a car straight ahead (from the vehicle centre), '
       'moved 8 cm behind the sample\'s front bumper fascia, from where the car\'s centre lies 250 m × 10^(−L/40) '
       'from the radar, L the two-way loss the materials study gives for that zone',
       'the radar equation (range ∝ (two-way power transmission)^¼) applied to the materials study\'s loss, the '
       'geometry written out in the check', 1e-3, 'relative', refs=('skolnik2001', 'macleod2010'), cost='study')
def _radar_fascia(ctx):
    from ..core import materials as M, sensor_coverage as SC
    p = ctx.sample()
    p.active_uid = next(v for v in p.variants if v.sensors).uid
    p.variant.sensors = [s for s in p.variant.sensors if s.tag == 'RD1']
    rd = p.variant.sensors[0]
    st = SC.CoverageSettings.from_dict({'classes': [SC.DEFAULT_CLASSES[0]], 'max_range': 400.0})
    rd.x = float(p.vehicle.mesh.bbox[1][0]) - 0.08
    rep = M.cached_run(p, ctx.engine.vehicle_geom(p)).radar('RD1')
    zone = rep['zones'][0]
    loss = zone['loss_db']
    res = SC.run(p, ctx.engine, settings=st)
    got = float(res.radar[0, 0])
    cls = res.classes[0]
    reach = zone['range_m'] * 10 ** (-loss / 40) * (cls.rcs / 10.0) ** 0.25
    cx, cy = res.centre
    dz = cls.base + cls.height / 2 - rd.z
    want = rd.x - cx + math.sqrt(reach ** 2 - (cy - rd.y) ** 2 - dz ** 2)
    return Outcome(abs(got - want) / want, {'zone_range_m': zone['range_m'], 'two_way_loss_db': loss,
                                            'factor': zone['factor'], 'range_from_centre_m': got,
                                            'radar_equation_m': want, 'mount': rep['mount']},
                   ok=rep['mount'] == 'behind' and 0 < loss < 10)


@check('radar.corner_reflector', 'radar', 'Cross-section of a trihedral corner reflector',
       'relative difference of observability.reflector_rcs from 4πa⁴ / 3λ² at 77 GHz for edges of 50–300 mm',
       'the peak RCS of a triangular trihedral (Knott et al.)', 1e-9, 'relative', refs=('knott2004',))
def _radar_reflector(ctx):
    from ..core import observability as OB
    lam = 299792458.0 / 77e9
    a = np.array([50.0, 100.0, 150.0, 300.0])
    got = np.array([OB.reflector_rcs(x) for x in a])
    want = 4 * math.pi * (a / 1000) ** 4 / (3 * lam ** 2)
    return Outcome(float(rel_err(got, want).max()), {'edge_mm': a.tolist(), 'rcs_m2': got.round(4).tolist()})


# ============================================================================ glass
@check('glass.plate_shift', 'glass', 'A ray through a glass plate shifts by the plate formula',
       'largest relative difference of the sideways shift materials.glass_optics gives for 3.5 mm side glass '
       '(n 1.52) from d·sin θ·(1 − cos θ / √(n² − sin²θ)), 0–75° incidence, and of the windscreen\'s slide along its '
       'surface from Σ d_j (tan θ − sin θ_j / cos θ_j)',
       'the textbook lateral displacement of a plane-parallel plate (Hecht)', 1e-9, 'relative',
       refs=('hecht2017',))
def _glass_shift(ctx):
    from ..core import materials as M
    th = np.array([5.0, 30.0, 60.0, 75.0])
    o = M.glass_optics(M.CLASSES['side_glass'], th)
    d, n = 3.5e-3, 1.52
    t = np.radians(th)
    want = d * np.sin(t) * (1 - np.cos(t) / np.sqrt(n * n - np.sin(t) ** 2))
    e1 = float(rel_err(o['shift_m'], want).max())
    lam = M.glass_optics(M.CLASSES['windscreen'], np.array([60.0]))
    exp = sum(dm / 1000 * (math.tan(math.radians(60)) - (math.sin(math.radians(60)) / nn)
                           / math.sqrt(1 - (math.sin(math.radians(60)) / nn) ** 2))
              for nn, dm in M.CLASSES['windscreen'].optical)
    e2 = abs(lam['slide_m'][0] - exp) / exp
    return Outcome(max(e1, e2), {'side_glass_shift_mm': (o['shift_m'] * 1000).round(5).tolist(),
                                 'windscreen_slide_mm_at_60deg': float(lam['slide_m'][0] * 1000)})


@check('glass.energy', 'glass', 'Glass transmission and reflection conserve energy',
       'largest |R + T − 1| of a non-absorbing glass (n 1.5) from 0° to 85° (incoherent slab, Fresnel at both faces), '
       'and the difference of a class\'s transmission at normal incidence from its stated visible transmission',
       'energy conservation, and the class\'s own t_vis', 1e-9, 'power ratio', refs=('bornwolf1999', 'hecht2017'))
def _glass_energy(ctx):
    from ..core import materials as M
    R0 = ((1.5 - 1) / 2.5) ** 2
    clear = M.MaterialClass('t', 't', 'glass', optical=((1.5, 3.0),), t_vis=(1 - R0) / (1 + R0))
    o = M.glass_optics(clear, np.linspace(0, 85, 18))
    e1 = float(np.abs(o['T'] + o['R'] - 1).max())
    e2 = abs(float(M.glass_optics(M.CLASSES['side_glass'], np.array([0.0]))['T'][0]) - M.CLASSES['side_glass'].t_vis)
    return Outcome(max(e1, e2), {'energy_error': e1, 'normal_incidence_error': e2})
