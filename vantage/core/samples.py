"""Sample and template projects."""
from __future__ import annotations

import math
import os

import numpy as np

from .bodygen import build_body, tigor_spec, template_specs, BodySpec
from .mathutil import look_rotation, euler_zyx
from .project import (Project, Vehicle, Variant, Camera, Target, Obstacle, PhotoSet, PhotoFrame, Sensor, new_uid)

TAGS_6 = [('F', 'Front'), ('R', 'Rear'), ('ML', 'Mirror left'), ('MR', 'Mirror right'),
          ('QL', 'Quarter left'), ('QR', 'Quarter right')]

# Wheel-centre detections on the 19-frame studio spin of the Tigor EV (930 x 620 px),
# labelled by the turntable fit. (frame, wheel, u, v)
TIGOR_HUB_OBS = [(1, 'FR', 410.18, 394.37), (1, 'RR', 312.83, 359.18), (2, 'FR', 488.14, 394.67), (2, 'RR', 279.67, 364.74),
                 (3, 'FR', 560.5, 391.33), (3, 'RR', 264.66, 372.76), (4, 'FR', 594.75, 387.56), (4, 'RR', 269.46, 378.53),
                 (5, 'RR', 279.33, 382.15), (6, 'RR', 346.5, 392.3), (6, 'FR', 637.53, 372.48), (7, 'RR', 436.78, 395.64),
                 (7, 'FR', 617.97, 362.86), (8, 'RR', 513.0, 395.3), (10, 'RL', 391.22, 394.24), (11, 'RL', 477.12, 395.11),
                 (11, 'FL', 287.97, 363.72), (12, 'RL', 553.45, 392.55), (12, 'FL', 270.93, 371.47), (13, 'RL', 606.7, 386.26),
                 (13, 'FL', 277.98, 379.55), (14, 'RL', 626.54, 381.53), (15, 'FL', 313.5, 387.62), (15, 'RL', 637.95, 378.4),
                 (16, 'FL', 368.88, 392.69), (16, 'RL', 640.82, 370.1), (17, 'FL', 444.28, 393.91), (17, 'RL', 620.27, 361.65)]
TIGOR_PHOTO = {'D': 9.0102, 'hc': 1.3648, 'beta': 0.08373, 'f': 1131.23, 'u0': 453.94, 'v0': 331.03, 'w': 930, 'h': 620,
               'axle_mid_x': 1.225, 'hub_y': 0.82337, 'hub_r': 0.29806,
               'az': [-0.02709, -0.39176, -0.74999, -1.11277, -1.32926, -1.49386, -2.02515, -2.47465, -2.82375, -3.14272,
                      -3.44159, -3.83641, -4.21345, -4.57082, -4.76665, -4.95603, -5.28625, -5.6455, -5.9808]}


def vehicle_from_spec(spec: BodySpec) -> Vehicle:
    mesh, info = build_body(spec)
    wc = {}
    for k, v in info['landmarks'].items():
        if k.startswith('Wheel centre '):
            wc[k[-2:]] = list(v)
    veh = Vehicle(name=spec.name, spec_name=f'{spec.name}' + (' sedan' if spec.key == 'tigor' else ' (template body)'),
                  source=f'template:{spec.key}', template=spec.key, mesh=mesh, wheel_centres=wc, spec=dict(spec.spec),
                  landmarks={k: tuple(v) for k, v in info['landmarks'].items()}, imported_tris=mesh.n_tris,
                  occluder_target=min(8000, mesh.n_tris))
    return veh


def snap_point(mesh, hint, direction, standoff_m):
    """Ray from outside along -direction onto the body; returns (camera position, surface normal)."""
    d = np.asarray(direction, float)
    d = d / np.linalg.norm(d)
    o = np.asarray(hint, float) + d * 0.6
    t, f = mesh.raycast(o[None], -d[None], faces_mask=mesh.occluder_faces())
    if f[0] < 0:
        return np.asarray(hint, float), d
    p = o - d * t[0]
    n = mesh.normals[f[0]]
    if n @ d < 0:
        n = -n
    return p + n * standoff_m, n


def make_camera(mesh, tag, name, location, hint, direction, yaw, pitch, slot, port, standoff_mm=12.0, snap=True):
    if snap:
        p, n = snap_point(mesh, hint, direction, standoff_mm / 1000)
    else:
        p = np.asarray(hint, float)
        q, fi, dist = mesh.closest_point(p)
        standoff_mm = round(dist * 1000)
    return Camera(tag=tag, name=name, location=location, slot=slot, x=round(float(p[0]), 3), y=round(float(p[1]), 3),
                  z=round(float(p[2]), 3), yaw=yaw, pitch=pitch, port=port, standoff_mm=standoff_mm)


def link_pairs(variant: Variant):
    pairs = [('ML', 'MR'), ('QL', 'QR'), ('FL', 'FR'), ('RL', 'RR'), ('SL', 'SR'), ('TL', 'TR')]
    for a, b in pairs:
        ca, cb = variant.by_tag(a), variant.by_tag(b)
        if ca and cb:
            ca.link_uid, cb.link_uid = cb.uid, ca.uid


def tigor_variants(mesh) -> list[Variant]:
    """Mount points measured on the photo-fitted Tigor (ISO 8855, rear-axle origin)."""
    def cam(tag, name, loc, x, y, z, yaw, pitch, slot, port):
        q, fi, d = mesh.closest_point(np.array([x, y, z]))
        return Camera(tag=tag, name=name, location=loc, slot=slot, x=x, y=y, z=z, yaw=yaw, pitch=pitch, port=port,
                      standoff_mm=float(round(d * 1000)))

    v3 = Variant(name='6-cam · v3', note='Recommended layout: no blind ground within 2 m.')
    v3.cameras = [cam('F', 'Front', 'Front, grille above plate', 3.230, 0.000, 0.620, 0.0, 40.0, 0, 1),
                  cam('R', 'Rear', 'Rear, bumper top edge', -0.800, 0.000, 0.560, 180.0, 40.0, 1, 2),
                  cam('ML', 'Mirror left', 'Under left mirror', 1.880, 0.940, 0.950, 65.0, 45.0, 2, 3),
                  cam('MR', 'Mirror right', 'Under right mirror', 1.880, -0.940, 0.950, -65.0, 45.0, 3, 4),
                  cam('QL', 'Quarter left', 'Left rear quarter', 0.000, 0.880, 0.950, 90.0, 50.0, 4, 5),
                  cam('QR', 'Quarter right', 'Right rear quarter', 0.000, -0.880, 0.950, -90.0, 50.0, 5, 6)]
    link_pairs(v3)
    four = Variant(name='4-cam', note='Best aim found for four cameras.')
    four.cameras = [cam('F', 'Front', 'Front, grille above plate', 3.230, 0.000, 0.620, 0.0, 40.0, 0, 1),
                    cam('R', 'Rear', 'Rear, bumper top edge', -0.800, 0.000, 0.560, 180.0, 60.0, 1, 2),
                    cam('ML', 'Mirror left', 'Under left mirror', 1.880, 0.940, 0.950, 105.0, 55.0, 2, 3),
                    cam('MR', 'Mirror right', 'Under right mirror', 1.880, -0.940, 0.950, -105.0, 55.0, 3, 4)]
    link_pairs(four)
    off = Variant(name='4-cam, mirror cameras 5° off', note='Tolerance case: mirror units 5° off and 10 cm back.')
    off.cameras = [cam('F', 'Front', 'Front, grille above plate', 3.230, 0.000, 0.620, 0.0, 40.0, 0, 1),
                   cam('R', 'Rear', 'Rear, bumper top edge', -0.800, 0.000, 0.560, 180.0, 60.0, 1, 2),
                   cam('ML', 'Mirror left', 'Under left mirror', 1.780, 0.940, 0.950, 110.0, 50.0, 2, 3),
                   cam('MR', 'Mirror right', 'Under right mirror', 1.780, -0.940, 0.950, -110.0, 50.0, 3, 4)]
    link_pairs(off)
    fh = Variant(name='6-cam front-heavy', note='Extra pair on the front bumper corners.')
    fh.cameras = [cam('F', 'Front', 'Front, grille above plate', 3.230, 0.000, 0.620, 0.0, 35.0, 0, 1),
                  cam('R', 'Rear', 'Rear, bumper top edge', -0.800, 0.000, 0.560, 180.0, 40.0, 1, 2),
                  cam('ML', 'Mirror left', 'Under left mirror', 1.880, 0.940, 0.950, 90.0, 55.0, 2, 3),
                  cam('MR', 'Mirror right', 'Under right mirror', 1.880, -0.940, 0.950, -90.0, 55.0, 3, 4),
                  cam('FL', 'Front left', 'Front bumper, left corner', 3.235, 0.718, 0.620, 45.0, 40.0, 4, 5),
                  cam('FR', 'Front right', 'Front bumper, right corner', 3.235, -0.718, 0.620, -45.0, 40.0, 5, 6)]
    link_pairs(fh)
    rig = Variant(name='Data rig · 6-cam + lidar + radars',
                  note='Collection rig: the v3 cameras, a 128-channel lidar on the roof and five radars.')
    rig.cameras = [Camera.from_dict(dict(c.to_dict(), uid=new_uid(), link_uid='')) for c in v3.cameras]
    link_pairs(rig)
    rig.sensors = data_rig_sensors(mesh)
    return [v3, four, off, fh, rig]


def data_rig_sensors(mesh) -> list[Sensor]:
    """A roof lidar (Ouster OS1-128), a long-range radar in the front bumper and a corner radar at each bumper corner,
    placed on the body's surface (the radars just behind the fascia)."""
    from .sensors import builtin_sensor_models, default_pose
    models = {m.id: m for m in builtin_sensor_models()}
    out = [Sensor(tag='L1', name='Roof lidar', kind='lidar', model_id='ouster_os1_128',
                  **default_pose('lidar', models['ouster_os1_128'], mesh))]
    front = default_pose('radar', models['radar_long'], mesh)
    out.append(Sensor(tag='RD1', name='Front radar', kind='radar', model_id='radar_long', **front))
    lo, hi = mesh.bbox
    for i, (name, x, side, yaw) in enumerate((('Front left corner', hi[0] - 0.25, 1, 45.0),
                                               ('Front right corner', hi[0] - 0.25, -1, -45.0),
                                               ('Rear left corner', lo[0] + 0.25, 1, 135.0),
                                               ('Rear right corner', lo[0] + 0.25, -1, -135.0))):
        d = np.array([math.cos(math.radians(yaw)), math.sin(math.radians(yaw)), 0.0])
        p, _ = snap_point(mesh, np.array([x, side * (hi[1] - 0.2), 0.5]), d, -0.01)
        out.append(Sensor(tag=f'RD{i + 2}', name=f'{name} radar', kind='radar', model_id='radar_corner',
                          x=round(float(p[0]), 3), y=round(float(p[1]), 3), z=0.5, yaw=yaw))
    return out


def tigor_photos(mesh) -> PhotoSet:
    P = TIGOR_PHOTO
    # the studio photos are not distributed with Vantage: point VANTAGE_TIGOR_PHOTOS at the zip, or put it in
    # data/ inside the project or in the folder that contains the project
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    src = ''
    for cand in (os.environ.get('VANTAGE_TIGOR_PHOTOS', ''), os.path.join(root, 'data', 'Tata-Tigor-EV-360.zip'),
                 os.path.join(os.path.dirname(root), 'Tata-Tigor-EV-360.zip')):
        if cand and os.path.exists(cand):
            src = cand
            break
    ps = PhotoSet(source=src, width=P['w'], height=P['h'], f_px=P['f'], cx=P['u0'], cy=P['v0'], k1=0.0,
                  method='Turntable fit to wheel centres', solved_at='')
    hy, hr, ax = P['hub_y'], P['hub_r'], P['axle_mid_x']
    ps.landmarks = {'Wheel centre FL': [ax + 1.225, hy, hr], 'Wheel centre FR': [ax + 1.225, -hy, hr],
                    'Wheel centre RL': [ax - 1.225, hy, hr], 'Wheel centre RR': [ax - 1.225, -hy, hr]}
    b = P['beta']
    for k, al in enumerate(P['az']):
        ca, sa = math.cos(al), math.sin(al)
        fwd = np.array([-ca * math.cos(b), -sa * math.cos(b), -math.sin(b)])
        right = np.array([-sa, ca, 0.0])
        down = np.cross(fwd, right)
        C = np.array([P['D'] * ca + ax, P['D'] * sa, P['hc']])
        R = np.stack([right, down, fwd])
        t = -R @ C
        M = np.eye(4)
        M[:3, :3], M[:3, 3] = R, t
        pts = [(f'Wheel centre {w}', u, v) for (fr, w, u, v) in TIGOR_HUB_OBS if fr == k]
        ps.frames.append(PhotoFrame(name=f'Tata-Tigor-EV-360/images/img_0_0_{k}.jpg', pose=M.round(6).tolist(), points=pts))
    ps.current = 7
    from .photos import refresh_residuals
    refresh_residuals(ps)
    return ps


def tigor_targets() -> list[Target]:
    pos = [(4.55, 0.0), (3.87, 1.13), (3.87, -1.13), (1.25, 2.35), (1.25, -2.35), (-1.88, 1.63), (-1.88, -1.63), (-2.10, 0.0)]
    return [Target(name=f'T{i + 1}', x=x, y=y) for i, (x, y) in enumerate(pos)]


def tigor_sample_project() -> Project:
    p = Project()
    p.name = 'tigor-ev-svm'
    p.vehicle = vehicle_from_spec(tigor_spec())
    p.variants = tigor_variants(p.vehicle.mesh)
    p.active_uid = p.variants[0].uid
    p.targets = tigor_targets()
    p.obstacles = [Obstacle(name='Pole 1.0 m', x=4.0, y=1.2, height=1.0), Obstacle(name='Pole 0.5 m', x=-1.6, y=-1.4, height=0.5)]
    p.photos = tigor_photos(p.vehicle.mesh)
    p.add_log('import', 'Tata Tigor EV body from the 19-frame photo fit · '
              f'{p.vehicle.mesh.n_tris} tris · units m · up +Z · forward +X')
    return p


def template_layout(mesh, spec: BodySpec, n: int) -> list[Camera]:
    L, W = spec.length, spec.width
    xr = -spec.rear_overhang
    xf = xr + L
    mx = xr + spec.mirror_u * L
    hw = W / 2
    cams = [
        make_camera(mesh, 'F', 'Front', 'Front, grille above plate', (xf, 0, spec.plate_front_z + 0.08), (1, 0, 0), 0.0, 40.0, 0, 1),
        make_camera(mesh, 'R', 'Rear', 'Rear, bumper top edge', (xr, 0, 0.56 if spec.height < 2 else 0.9), (-1, 0, 0), 180.0, 40.0, 1, 2),
        make_camera(mesh, 'ML', 'Mirror left', 'Under left mirror', (mx - 0.01, hw + 0.08, spec.mirror_z - 0.07), (0, 0, -1), 65.0, 45.0, 2, 3),
        make_camera(mesh, 'MR', 'Mirror right', 'Under right mirror', (mx - 0.01, -hw - 0.08, spec.mirror_z - 0.07), (0, 0, -1), -65.0, 45.0, 3, 4),
    ]
    if n == 4:
        cams[2].yaw, cams[2].pitch, cams[3].yaw, cams[3].pitch = 105.0, 55.0, -105.0, 55.0
        cams[1].pitch = 60.0
    if n >= 6:
        zq = min(spec.belt_z, spec.height * 0.62)
        cams += [make_camera(mesh, 'QL', 'Quarter left', 'Left rear quarter', (0.0, hw, zq), (0, 1, 0), 90.0, 50.0, 4, 5),
                 make_camera(mesh, 'QR', 'Quarter right', 'Right rear quarter', (0.0, -hw, zq), (0, -1, 0), -90.0, 50.0, 5, 6)]
    if n >= 8:
        zc = spec.plate_front_z + 0.1
        cams += [make_camera(mesh, 'FL', 'Front left', 'Front left corner', (xf - 0.15, hw * 0.8, zc), (0.7, 0.7, 0), 45.0, 40.0, 6, 7),
                 make_camera(mesh, 'FR', 'Front right', 'Front right corner', (xf - 0.15, -hw * 0.8, zc), (0.7, -0.7, 0), -45.0, 40.0, 7, 8)]
    return cams


def template_project(key: str) -> Project:
    specs = template_specs()
    if key == 'tigor':
        return tigor_sample_project()
    spec = specs[key]
    p = Project()
    p.name = f'{key}-template'
    p.vehicle = vehicle_from_spec(spec)
    n = int(spec.cameras.split('-')[0])
    v = Variant(name=f'{n}-cam · v1')
    v.cameras = template_layout(p.vehicle.mesh, spec, n)
    link_pairs(v)
    p.variants = [v]
    p.active_uid = v.uid
    p.add_log('import', f'{spec.name} template body · {p.vehicle.mesh.n_tris} tris')
    return p


def blank_project() -> Project:
    p = Project()
    p.name = 'untitled'
    p.variants = [Variant(name='Variant 1')]
    p.active_uid = p.variants[0].uid
    return p
