"""Checks of the rig models: the lidar sweep against the cameras, triggers and motion blur (sync and timing), and the
GNSS antennas' sky view and the pose error budget."""
from __future__ import annotations

import math

import numpy as np

from .suite import Outcome, check, rel_err


def _timing_rig(cams, fps=10.0, lidar=None, model='ouster_os1_128'):
    """A bare project: level 6 mm pinhole cameras (tag, x, y, z, yaw) on a 1920 × 1536, 3 µm module at fps, and a
    spinning lidar at the origin, 1.8 m up (no vehicle)."""
    from ..core.lens import Lens, Module
    from ..core.project import Camera, Project, Sensor
    p = Project()
    p.modules['m'] = Module(id='m', name='Test module', width_px=1920, height_px=1536, pixel_um=3.0, fps=fps)
    p.lenses['pin'] = Lens(id='pin', name='Pinhole 6 mm', focal_mm=6.0, projection='rectilinear')
    v = p.variant
    for i, (tag, x, y, z, yaw) in enumerate(cams):
        v.cameras.append(Camera(tag=tag, slot=i, x=x, y=y, z=z, yaw=yaw, pitch=0.0, roll=0.0, module_id='m',
                                lens_id='pin'))
    v.sensors.append(Sensor(kind='lidar', model_id=model, tag='L1', x=0.0, y=0.0, z=1.8, **(lidar or {})))
    return p


SIDES = [('F', 0.0, 0.0, 1.0, 0.0), ('L', 0.0, 0.5, 1.0, 90.0), ('R', 0.0, -0.5, 1.0, -90.0)]


def _apart(a, b, T=100.0):
    return abs((a - b + T / 2) % T - T / 2)


# ============================================================================ timing
@check('timing.sweep', 'timing', 'When the lidar sweep crosses each camera',
       'largest difference, in milliseconds, of the sweep\'s crossing, entry and duration over the image from the '
       'geometry of a 10 Hz lidar (clockwise and counter-clockwise, phase 0° and 90°, turned round), level 6 mm '
       'pinhole cameras ahead, left and right, and one 2 m ahead looking left (parallax: it is crossed when the lidar '
       'faces atan(20 m / 2 m))',
       'the head\'s azimuth phase ∓ 360°·f·t written out in the check; the image\'s half field atan(2.88 mm / 6 mm)',
       1e-6, 'ms', refs=('ieee1588',))
def _sweep(ctx):
    import copy
    from ..core import timing as TM
    e = {}
    c = {x['tag']: x for x in TM.plan(_timing_rig(SIDES)).cams}
    e['cw F crossing'] = _apart(c['F']['cross_ms'], 0.0)
    e['cw L crossing'] = abs(c['L']['cross_ms'] - 75.0)
    e['cw R crossing'] = abs(c['R']['cross_ms'] - 25.0)
    half = math.degrees(math.atan(2.88 / 6.0))
    e['image sweep duration'] = abs(c['F']['sweep_ms'] - 2 * half / 3.6)
    e['image entry'] = abs(c['F']['enter_ms'] - (360.0 - half) / 3.6)
    p = _timing_rig(SIDES)
    m = copy.deepcopy(p.sensor_models['ouster_os1_128'])
    m.id, m.rotation, m.builtin = 'ccw_lidar', 'ccw', False
    p.sensor_models[m.id] = m
    p.variant.sensors[0].model_id = m.id
    c = {x['tag']: x for x in TM.plan(p).cams}
    e['ccw L crossing'] = abs(c['L']['cross_ms'] - 25.0)
    e['ccw R crossing'] = abs(c['R']['cross_ms'] - 75.0)
    c = {x['tag']: x for x in TM.plan(_timing_rig(SIDES, lidar={'phase_deg': 90.0})).cams}
    e['phase 90°: L crossing'] = _apart(c['L']['cross_ms'], 0.0)
    e['phase 90°: F crossing'] = abs(c['F']['cross_ms'] - 25.0)
    s = TM.plan(_timing_rig([('S', 2.0, 0.0, 1.0, 90.0)])).cams[0]
    az = math.degrees(math.atan2(20.0, 2.0))
    e['parallax crossing'] = abs(s['cross_ms'] - (360.0 - az) / 3.6)
    c = {x['tag']: x for x in TM.plan(_timing_rig(SIDES, lidar={'yaw': 180.0})).cams}
    e['turned round: F crossing'] = abs(c['F']['cross_ms'] - 50.0)
    return Outcome(max(e.values()), e)


@check('timing.trigger', 'timing', 'Triggers centre the middle row\'s exposure on the sweep',
       'largest difference of each camera\'s ideal trigger from its crossing − readout/2 − exposure/2 (modulo the '
       'period) and of its misalignment from zero, cameras alone on their lines, three readout and exposure times',
       'a rolling shutter\'s middle row is exposed around trigger + readout/2 + exposure/2 (Liang et al.)', 1e-9,
       'ms', refs=('liang2008',))
def _trigger(ctx):
    from ..core import timing as TM
    worst = 0.0
    for ro, ex in ((20.0, 2.0), (0.0, 1.0), (30.0, 5.0)):
        p = _timing_rig(SIDES)
        for i, cam in enumerate(p.variant.cameras):
            cam.sync = 'ABC'[i]
        c = {x['tag']: x for x in TM.plan(p, settings=TM.TimingSettings(readout_ms=ro, exposure_ms=ex)).cams}
        for tag in 'FLR':
            want = (c[tag]['cross_ms'] - ro / 2 - ex / 2) % 100.0
            worst = max(worst, _apart(c[tag]['ideal_ms'], want), _apart(c[tag]['trigger_ms'], want),
                        abs(c[tag]['misalign_ms']))
    return Outcome(worst, {'cases': 9})


@check('timing.motion_blur', 'timing', 'Motion blur and rolling-shutter skew through the lens',
       'largest relative difference of the image speed of a static object 5 m to the side at 36 km/h (level 6 mm '
       'pinhole) from f·v/d at the image centre and f·v/(d·cos α) in the corner, of the blur over a 2 ms exposure and '
       'the skew over a 20 ms readout from speed × time; and of a fisheye pitched down from v·sin(pitch)/d × its '
       'centre sampling (to 1 %)',
       'the angular rate v·sin β / d of a static point seen from a moving camera, through the lens, written out in '
       'the check', 1e-4, 'relative', refs=('liang2008',))
def _blur(ctx):
    from ..core import timing as TM
    from ..core import sensors
    from ..core.lens import resolve
    from ..core.project import Sensor
    st = TM.TimingSettings(speed_kmh=36.0, exposure_ms=2.0, readout_ms=20.0, object_m=5.0, blur_limit_px=1.0)
    p = _timing_rig([('L', 0.0, 0.5, 1.0, 90.0), ('F', 0.0, 0.0, 1.0, 0.0)])
    c = {x['tag']: x for x in TM.plan(p, settings=st).cams}
    side = c['L']
    e = {'centre rate': abs(side['rate_centre_px_s'] - 4000.0) / 4000.0,
         'centre blur': abs(side['blur_centre_px'] - 8.0) / 8.0,
         'centre skew': abs(side['skew_centre_px'] - 80.0) / 80.0}
    r_mm = math.hypot(959.5, 767.5) * 0.003
    worst = 2000.0 * 10.0 / (5.0 * math.cos(math.atan(r_mm / 6.0)))
    e['corner rate'] = abs(side['rate_worst_px_s'] - worst) / worst
    e['corner skew'] = abs(side['skew_px'] - worst * 0.02) / (worst * 0.02)
    q = ctx.sample()
    m = q.sensor_models['ouster_os1_128']
    q.variant.sensors = [Sensor(kind='lidar', model_id='ouster_os1_128', tag='L1',
                                **sensors.default_pose('lidar', m, q.vehicle.mesh, q.rig))]
    f = next(x for x in TM.plan(q).cams if x['tag'] == 'F')
    cam = q.variant.by_tag('F')
    intr = resolve(q.module_for(cam), q.lens_for(cam))
    want = 50 / 3.6 * math.sin(math.radians(cam.pitch)) / 5.0 * intr.f_px
    fish = abs(f['rate_centre_px_s'] - want) / want
    return Outcome(max(e.values()), dict(e, fisheye_centre_rate=fish), ok=fish < 0.01)


# ============================================================================ GNSS
TOP, OZ = 1.5, 0.05


def _gnss_car():
    from ..core.project import Project
    from .checks_optics import box_mesh
    p = Project()
    p.vehicle.mesh = box_mesh((-0.5, -0.9, 0.0), (3.5, 0.9, TOP))
    return p


def _antenna(tag='GN1', x=1.0, y=0.0, z=TOP + OZ, model='gnss_survey'):
    from ..core.project import Sensor
    return Sensor(kind='gnss', model_id=model, tag=tag, x=x, y=y, z=z)


def _wall(d, w, h, x0=1.0):
    from ..core.project import RigPart
    ztop = TOP + OZ + h
    return RigPart(name='Wall', kind='box', x=x0 + d + 0.005, y=0.0, z=(TOP + ztop) / 2, length=0.01, width=w,
                   height=ztop - TOP)


def _wall_share(d, w, h, mask):
    """The share of the sky above the mask a wall's front face hides: for each azimuth within ±atan(w/2d) its top
    edge stands at atan(h·cos az / d); solid angle ∝ ∫ (sin el_top − sin mask) d az."""
    from scipy.integrate import quad
    a = math.atan(w / 2 / d)
    f = lambda az: max(math.sin(math.atan(h * math.cos(az) / d)) - math.sin(math.radians(mask)), 0.0)  # noqa: E731
    return quad(f, -a, a, limit=200)[0] / (2 * math.pi * (1 - math.sin(math.radians(mask))))


@check('gnss.sky_solid_angle', 'gnss', 'Sky blocked by a wall against its solid angle',
       'largest relative difference of the share of the sky above the 10° mask a wall in front of a roof antenna '
       'hides (sky view on 0.5° × 0.5° cells) from the wall face\'s solid angle, four walls; the cells of an open '
       'roof must add up to the hemisphere and see all of it',
       'the solid angle of a rectangle seen from a point, integrated with scipy.integrate.quad', 0.02, 'relative',
       refs=('groves2013',), cost='study')
def _sky(ctx):
    from ..core import gnss as G
    from ..core.analysis import Engine
    G.clear_cache()
    p = _gnss_car()
    p.variant.sensors.append(_antenna())
    a = G.run(p, Engine()).antennas[0]
    A = len(a.az)
    w = math.radians(360.0 / A) * (np.sin(np.radians(a.el_edges[1:])) - np.sin(np.radians(a.el_edges[:-1])))
    hemi = abs(w.sum() * A - 2 * math.pi) / (2 * math.pi)
    open_ok = abs(a.visible - 1.0) < 1e-9 and hemi < 1e-12
    rows, worst = {}, 0.0
    walls = ((1.0, 1.2, 0.8), (0.6, 1.2, 0.8), (1.0, 0.6, 0.6), (0.5, 1.5, 1.0))
    st = G.GnssSettings.from_dict({'az_step_deg': 0.5, 'el_step_deg': 0.5})
    for d, wd, h in (walls[:2] if ctx.quick else walls):
        q = _gnss_car()
        q.variant.sensors.append(_antenna())
        q.rig.append(_wall(d, wd, h))
        G.clear_cache()
        got = G.run(q, Engine(), st).antennas[0].blocked
        want = _wall_share(d, wd, h, 10.0)
        rows[f'd {d:g} m, w {wd:g} m, h {h:g} m'] = {'blocked': got, 'solid_angle_share': want}
        worst = max(worst, abs(got - want) / want)
    G.clear_cache()
    return Outcome(worst, {'walls': rows, 'hemisphere_rel_error': hemi}, ok=open_ok)


@check('gnss.pose_budget', 'gnss', 'The pose error budget against a hand calculation and Monte Carlo',
       'largest relative difference of a camera\'s position σ (3 m, 1 m and 1 m lever arm from the IMU) from the hand '
       'calculation INS σ ⊕ (δθ × L) ⊕ lever-arm σ, and of its attitude part from the spread of 400 000 sampled '
       'attitude errors crossed with the lever arm',
       'small-angle error propagation δp = δθ × L (Groves), by hand and by Monte Carlo', 0.01, 'relative',
       refs=('groves2013',), cost='study')
def _budget(ctx):
    from ..core import gnss as G
    from ..core.analysis import Engine
    from ..core.project import Camera, Sensor
    G.clear_cache()
    p = _gnss_car()
    p.variant.sensors += [Sensor(kind='imu', model_id='ins_dual', tag='IMU1', x=0.5, y=0.0, z=0.4),
                          _antenna('GN1', x=0.5), _antenna('GN2', x=2.5)]
    p.variant.cameras.append(Camera(tag='C1', x=3.5, y=1.0, z=1.4))
    st = G.GnssSettings.from_dict(dict(pos_sigma_cm=2.0, vertical_ratio=2.0, attitude_sigma_deg=0.05,
                                       heading_sigma_deg=0.2, lever_arm_sigma_mm=5.0, label_range_m=50.0))
    r = G.run(p, Engine(), st).budget_row('C1')
    G.clear_cache()
    sr = sp = math.radians(0.05)
    sy = math.radians(0.2)
    Lx, Ly, Lz = 3.0, 1.0, 1.0
    rot = np.array([1000 * math.sqrt((sp * Lz) ** 2 + (sy * Ly) ** 2),
                    1000 * math.sqrt((sy * Lx) ** 2 + (sr * Lz) ** 2),
                    1000 * math.sqrt((sr * Ly) ** 2 + (sp * Lx) ** 2)])
    meas = 5.0 * math.sqrt(2)
    tot = np.array([math.sqrt(20 ** 2 + rot[0] ** 2 + meas ** 2), math.sqrt(20 ** 2 + rot[1] ** 2 + meas ** 2),
                    math.sqrt(40 ** 2 + rot[2] ** 2 + meas ** 2)])
    n = 100_000 if ctx.quick else 400_000
    dth = ctx.rng('gnss.pose_budget').normal(size=(n, 3)) * np.array([sr, sp, sy])
    mc = np.cross(dth, [Lx, Ly, Lz]).std(axis=0) * 1000
    e = {'rotation vs hand': float(rel_err(r['rotation_mm'], rot).max()),
         'total vs hand': float(rel_err(r['sigma_mm'], tot).max()),
         'rotation vs Monte Carlo': float(rel_err(r['rotation_mm'], mc).max())}
    return Outcome(max(e.values()), dict(e, rotation_mm=[float(x) for x in r['rotation_mm']],
                                         sigma_mm=[float(x) for x in r['sigma_mm']], samples=n))
