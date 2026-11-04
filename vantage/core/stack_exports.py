"""Exports of a variant's sensor rig into the AV research stack: the open-source driving stack (Autoware Universe), the
end-to-end benchmarks (NAVSIM, HUGSIM), the training frameworks (MMDetection3D, OpenPCDet, LeRobot).

Every file follows the target project's own format, read from its repository on 2026-11-04 (the commit read is given
with each exporter). Where a convention the target does not document had to be chosen, the exporter says so in the
file it writes and in its README; nothing here is a format of Vantage's own making except the sidecar files named
``vantage_*`` (which the target tools ignore).

Frames. Vantage's vehicle frame is ISO 8855 at the rear-axle centre on the ground (x forward, y left, z up, metres),
the frame nuScenes calls the ego frame. A camera's optical frame is OpenCV's (x right, y down, z forward); a lidar's
or radar's frame is REP 103 (x forward, y left, z up) at its optical centre. The poses written are the variant's own
(a calibrated or as-built variant carries the measured poses), from core.exporters' frames:
``T_vehicle_camera`` maps optical-frame points into the vehicle frame, ``T_vehicle_sensor`` sensor-frame points.

openpilot is not a target: it runs on one comma device whose cameras are fixed in its housing, and calibrates that
device's roll, pitch and yaw to the car at run time (selfdrive/locationd/calibrationd.py publishes
``liveCalibration.rpyCalib``); it has no documented multi-sensor rig format to export into.

Sources (read 2026-11-04):

- Autoware: autowarefoundation/sample_sensor_kit_launch @ 005bf85 (2025-03-28),
  sample_sensor_kit_description/config/{sensors,sensor_kit}_calibration.yaml and urdf/sensor_kit.xacro;
  tier4/sensor_component_description @ 03ba094, camera_description/urdf/monocular_camera.xacro (``camera_link`` and its
  ``camera_optical_link`` at rpy (-pi/2, 0, -pi/2)); autoware-documentation @ 6ffe771,
  docs/contributing/coding-guidelines/ros-nodes/coordinate-system.md ("base_link ... the center of the rear axle") and
  docs/tutorials/integrating-autoware/creating-vehicle-and-sensor-model/creating-individual-params/index.md.
- NAVSIM: autonomousvision/navsim @ 0a380a9 (2025-10-27), navsim/common/dataclasses.py (``Camera``, ``Cameras``,
  ``SensorConfig``); OpenScene v1.1 process_data/create_nuplan_data_with_vis.py (huggingface.co/datasets/
  OpenDriveLab/OpenScene, the ``cams`` and ``lidar2ego_*`` keys of NAVSIM's log pickles).
- HUGSIM: hyzhou404/HUGSIM @ 62c690d (2025-11-08), configs/sim/nuscenes_camera.yaml and sim/utils/sim_utils.py
  (``rt2pose``: scipy ``from_euler('XYZ', ..., degrees=True)``; ``v2c`` maps vehicle points into the camera),
  sim/hugsim_env/envs/hug_sim.py (every camera rendered relative to ``CAM_FRONT``).
- MMDetection3D: open-mmlab/mmdetection3d @ fe25f7a (2024-01-08), docs/en/advanced_guides/datasets/nuscenes.md (v2 info
  keys ``images[CAM].cam2img / cam2ego / lidar2cam``, ``lidar_points.lidar2ego``) and configs/_base_/datasets/
  nus-mono3d.py; the BEV multi-view pipeline of that page.
- OpenPCDet: open-mmlab/OpenPCDet @ 233f849 (2025-10-08), tools/cfgs/dataset_configs/custom_dataset.yaml and
  docs/CUSTOM_DATASET_TUTORIAL.md (unified frame x forward, y left, z up; z range / voxel = 40, x and y range / voxel a
  multiple of 16).
- LeRobot: huggingface/lerobot @ ba83e4c (2026-11-03), src/lerobot/datasets/utils.py (``DatasetInfo``, the
  ``meta/info.json`` fields), dataset_metadata.py (``CODEBASE_VERSION = "v3.0"``), utils/constants.py
  (``DEFAULT_FEATURES``); a v3.0 dataset's info.json (lerobot/aloha_static_coffee) for the image feature layout.
"""
from __future__ import annotations

import json
import math
import os
import time

import numpy as np

from . import exporters as EX
from .mathutil import quat_from_matrix, rpy_from_matrix

TARGETS = ('autoware', 'navsim', 'hugsim', 'mmdet3d', 'openpcdet', 'lerobot')
READ_ON = '2026-11-04'

# the nominal yaw (deg, left positive) of each target's camera slots: a Vantage camera takes the slot nearest its yaw
NUSCENES_SLOTS = {'CAM_FRONT': 0.0, 'CAM_FRONT_LEFT': 55.0, 'CAM_FRONT_RIGHT': -55.0, 'CAM_BACK': 180.0,
                  'CAM_BACK_LEFT': 110.0, 'CAM_BACK_RIGHT': -110.0}
NAVSIM_SLOTS = {'CAM_F0': 0.0, 'CAM_L0': 45.0, 'CAM_L1': 90.0, 'CAM_L2': 135.0, 'CAM_B0': 180.0, 'CAM_R0': -45.0,
                'CAM_R1': -90.0, 'CAM_R2': -135.0}
NUSCENES_CLASSES = ['car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle', 'motorcycle', 'pedestrian',
                    'traffic_cone', 'barrier']


# ---------------------------------------------------------------- shared
def _ctx(project, result=None):
    return EX._context(project, result, EX.ExportOptions(frame='rear_axle'))


def _write(out_dir, rel, text_or_bytes) -> str:
    path = os.path.join(out_dir, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = text_or_bytes.encode('utf-8') if isinstance(text_or_bytes, str) else text_or_bytes
    with open(path, 'wb') as f:
        f.write(data)
    return path


def _r(v, n=9):
    return EX._r(v, n)


def _mat(M, n=9):
    return [[_r(v, n) for v in row] for row in np.asarray(M, float)]


def _cam_yaw(cm) -> float:
    a = cm.R[:, 0]                                  # the camera body's x axis: where it looks, in the vehicle frame
    return math.degrees(math.atan2(a[1], a[0]))


def assign_slots(models, slots: dict) -> dict:
    """{slot name: CameraModel} giving each slot the camera nearest its nominal yaw (one camera per slot, the total
    angular difference smallest); cameras left over have no slot."""
    from scipy.optimize import linear_sum_assignment
    names = list(slots)
    if not models:
        return {}
    C = np.array([[abs(((_cam_yaw(cm) - slots[s]) + 180.0) % 360.0 - 180.0) for s in names] for cm in models])
    r, c = linear_sum_assignment(C)
    return {names[j]: models[i] for i, j in zip(r, c)}


def _ref_lidar(ctx):
    """(safe tag, sensor, model, T_vehicle_lidar) of the variant's first visible lidar, or None: the 'lidar' frame the
    lidar-centred formats (NAVSIM, MMDetection3D, HUGSIM) give the cameras relative to."""
    first = EX._first(ctx, 'lidar')
    if first is None:
        return None
    t, s = first
    return t, s, ctx.project.sensor_model_for(s), EX._T_vehicle_sensor(ctx, s)


def _lens_sidecar(cm) -> dict:
    """The lens as the pinhole formats cannot carry it: Kannala-Brandt k1..k4 (cv2.fisheye) or none (rectilinear)."""
    pin = EX._pinhole(cm)
    return {'model': 'pinhole (no distortion)' if pin else 'kannala_brandt (cv2.fisheye k1..k4)',
            'coefficients': [] if pin else [EX._sig(k) for k in cm.k],
            'image_size': [int(cm.intr.width), int(cm.intr.height)], 'camera': cm.cam.tag,
            'source': 'unit_calibration' if cm.cam.calib else 'library_lens'}


def _stamp():
    return time.strftime('%Y-%m-%dT%H:%M:%S')


# ---------------------------------------------------------------- Autoware Universe
def autoware_names(ctx) -> dict:
    """{camera or sensor uid: the Autoware frame name}: cameras camera0..N/camera_link in their slot order (as the
    sample sensor kit numbers them), lidars <tag>_base_link, GNSS antennas gnss_link (gnss<i>_link with two), IMUs
    <tag>/imu_link, radars <tag>/radar_link. The lidar, IMU and GNSS names follow the sample kit; Autoware has no
    radar in its sample kit, so the radar name is this exporter's choice."""
    out = {}
    for i, cm in enumerate(sorted(ctx.models, key=lambda m: (m.cam.slot, m.cam.tag))):
        out[cm.cam.uid] = f'camera{i}/camera_link'
    gn = [s for _, s, _ in EX._sensors(ctx) if s.kind == 'gnss']
    for t, s, m in EX._sensors(ctx):
        n = EX._ros_name(t)
        if s.kind == 'lidar':
            out[s.uid] = f'{n}_base_link'
        elif s.kind == 'radar':
            out[s.uid] = f'{n}/radar_link'
        elif s.kind == 'imu':
            out[s.uid] = f'{n}/imu_link'
        elif s.kind == 'gnss':
            out[s.uid] = 'gnss_link' if len(gn) == 1 else f'gnss{gn.index(s)}_link'
    return out


def _xyzrpy(T) -> dict:
    roll, pitch, yaw = rpy_from_matrix(T[:3, :3])
    return {'x': _r(T[0, 3], 6), 'y': _r(T[1, 3], 6), 'z': _r(T[2, 3], 6),
            'roll': _r(roll, 7), 'pitch': _r(pitch, 7), 'yaw': _r(yaw, 7)}


def _yaml_calib(parent: str, children: dict) -> str:
    """The sample kit's layout: the parent frame, then each child with x y z roll pitch yaw on its own lines."""
    lines = [f'{parent}:']
    for name, d in children.items():
        lines.append(f'  {name}:')
        for k in ('x', 'y', 'z', 'roll', 'pitch', 'yaw'):
            lines.append(f'    {k}: {d[k]!r}')
    return '\n'.join(lines) + '\n'


def autoware_poses(project, result=None, kit_origin='lidar'):
    """(T_base_kit, {frame name: T_kit_frame}, names) in Autoware's frames: base_link is Vantage's rear-axle frame (see
    export_autoware for its height); sensor_kit_base_link sits at the first lidar's base (kit_origin='lidar', axes as
    base_link) or at base_link ('base_link'). A camera's camera_link has the camera body's axes (x along the optical
    axis, y left, z up); a lidar's <tag>_base_link is the bottom of its housing (the optical centre less the model's
    optical height along the lidar's z), as Autoware's lidar descriptions put the driver frame above the base."""
    ctx = _ctx(project, result)
    names = autoware_names(ctx)
    frames = {}
    for cm in ctx.models:
        T = np.eye(4)
        T[:3, :3] = cm.R                              # camera body (x forward along the axis, y left, z up)
        T[:3, 3] = cm.cam.pos() - ctx.origin
        frames[names[cm.cam.uid]] = T
    for t, s, m in EX._sensors(ctx):
        T = EX._T_vehicle_sensor(ctx, s)
        if s.kind == 'lidar':
            T = T.copy()
            T[:3, 3] = T[:3, 3] - T[:3, :3] @ np.array([0.0, 0.0, float(m.optical_z_mm) / 1000.0])
        frames[names[s.uid]] = T
    T_base_kit = np.eye(4)
    if kit_origin == 'lidar':
        lid = [s for _, s, _ in EX._sensors(ctx) if s.kind == 'lidar']
        if lid:
            T_base_kit[:3, 3] = frames[names[lid[0].uid]][:3, 3]
    inv = EX._T_inv(T_base_kit)
    return T_base_kit, {n: inv @ T for n, T in frames.items()}, names


def export_autoware(project, engine, out_dir, result=None, kit='vantage', kit_origin='lidar') -> list:
    """Autoware Universe sensor-kit calibration (sample_sensor_kit_launch @ 005bf85):
    ``<kit>_sensor_kit_description/config/sensors_calibration.yaml`` (base_link -> sensor_kit_base_link) and
    ``sensor_kit_calibration.yaml`` (sensor_kit_base_link -> every sensor), x y z in metres and roll pitch yaw in
    radians as URDF fixed axes (sensor_kit.xacro passes them to ``<origin rpy=... xyz=...>``), plus a ROS camera_info
    per camera (core.exporters' format) for the camera drivers' camera_info_url.

    Uncertain: Autoware's documentation puts base_link at "the center of the rear axle" without its height; this
    writes it on the ground below the rear-axle centre (Vantage's frame, and the sample kit's 2.0 m kit height reads so).
    If your vehicle description puts base_link at axle height, subtract the wheel radius from every z of
    sensors_calibration.yaml (the README gives it)."""
    ctx = _ctx(project, result)
    T_bk, frames, names = autoware_poses(project, result, kit_origin)
    d = f'autoware/{kit}_sensor_kit_description/config'
    out = [_write(out_dir, f'{d}/sensors_calibration.yaml',
                  _yaml_calib('base_link', {'sensor_kit_base_link': _xyzrpy(T_bk)})),
           _write(out_dir, f'{d}/sensor_kit_calibration.yaml', _yaml_calib('sensor_kit_base_link',
                                                                           {n: _xyzrpy(T) for n, T in frames.items()}))]
    for cm in ctx.models:
        n = names[cm.cam.uid].split('/')[0]
        out.append(_write(out_dir, f'autoware/camera_info/{n}.yaml', EX._camera_info(ctx, cm)))
    wc = project.vehicle.wheel_centres or {}
    wr = float(wc['RL'][2]) if 'RL' in wc else None
    rows = '\n'.join(f'| `{n}` | {tag} |' for tag, n in
                     [(cm.cam.tag, names[cm.cam.uid]) for cm in ctx.models] +
                     [(f'{s.tag} ({s.kind})', names[s.uid]) for _, s, _ in EX._sensors(ctx)])
    readme = f"""# Autoware sensor kit calibration · Vantage {project.name} / {ctx.variant.name}, exported {_stamp()}

Copy `{kit}_sensor_kit_description/config/*.yaml` into your sensor kit description package's `config/` folder
(Autoware reads its calibration only there; the `individual_params` package is archived). The layout is the sample
sensor kit's (autowarefoundation/sample_sensor_kit_launch @ 005bf85): `sensors_calibration.yaml` places
`sensor_kit_base_link` relative to `base_link`, `sensor_kit_calibration.yaml` every sensor relative to the kit; x y z
in metres, roll pitch yaw in radians as URDF fixed axes (R = Rz(yaw) Ry(pitch) Rx(roll)).

| Autoware frame | Vantage |
|---|---|
{rows}

- **base_link**: Vantage's rear-axle frame, on the ground below the rear-axle centre (x forward, y left, z up).
  Autoware documents base_link as the centre of the rear axle but not its height: if your vehicle description puts it
  at axle height, subtract {f'{wr:.3f} m (the rear wheel radius here)' if wr else 'the wheel radius'} from the z of
  every entry in `sensors_calibration.yaml`.
- **sensor_kit_base_link**: {'at the base of the first lidar, with base_link\'s axes' if kit_origin == 'lidar' else 'at base_link'}.
- **cameras**: `cameraN/camera_link` has the camera body's axes (x along the optical axis, y left, z up); Autoware's
  monocular_camera.xacro adds `camera_optical_link` at rpy (-pi/2, 0, -pi/2), the OpenCV optical frame the images and
  `camera_info/` are in. The camera numbering is the cameras' slot order; rename to your launch files' names.
- **lidars**: `<tag>_base_link` is the bottom of the housing (the optical centre less the model's optical height); your
  lidar's description xacro adds its driver frame above it. If your driver's frame axes differ from REP 103 (an
  Ouster's x points to its connector), add the driver's fixed rotation there.
- **radars** (`<tag>/radar_link`): the sample kit has none; this name is Vantage's choice.
- **camera_info/**: ROS camera_info YAMLs (equidistant = Kannala-Brandt for the fisheye lenses) for each camera
  driver's `camera_info_url`.
"""
    out.append(_write(out_dir, 'autoware/README.md', readme))
    return out


# ---------------------------------------------------------------- NAVSIM
def export_navsim(project, engine, out_dir, result=None) -> list:
    """NAVSIM camera calibration (autonomousvision/navsim @ 0a380a9): the ``cams`` block of a NAVSIM / OpenScene log
    frame for the variant's rig, the eight slots CAM_F0, CAM_L0..L2, CAM_R0..R2, CAM_B0 that ``Cameras.from_camera_dict``
    reads (``data_path``, ``sensor2lidar_rotation`` 3x3, ``sensor2lidar_translation``, ``cam_intrinsic`` 3x3,
    ``distortion``), with ``lidar2ego_translation`` and ``lidar2ego_rotation`` (w, x, y, z), as OpenScene's
    create_nuplan_data_with_vis.py writes them, and a ``SensorConfig`` for the slots the rig fills.

    Limits: NAVSIM evaluates on nuPlan's recorded rig (OpenScene data); this file describes the Vantage rig in its
    conventions for a dataset of your own or a synthetic one. ``distortion`` is nuPlan's 5-coefficient plumb bob
    (k1, k2, p1, p2, k3, the uncertain part): a fisheye lens cannot be written in it, so it is zeros and the
    Kannala-Brandt terms are in ``vantage_lens`` (undistort the images first). The slots are filled by nearest nominal
    yaw (evenly spaced, an assumption; nuPlan's own camera angles differ)."""
    ctx = _ctx(project, result)
    ref = _ref_lidar(ctx)
    T_vl = ref[3] if ref else np.eye(4)
    inv_l = EX._T_inv(T_vl)
    slots = assign_slots(ctx.models, NAVSIM_SLOTS)
    cams = {}
    filled = set()
    for name in NAVSIM_SLOTS:
        cm = slots.get(name)
        if cm is None:
            # NAVSIM's Cameras.from_camera_dict builds its eight cameras from the dictionary's keys and then reads all
            # eight: an unfilled slot must still be a key (its SensorConfig entry is False, so it is loaded empty)
            cams[name] = {'data_path': None, 'sensor2lidar_rotation': None, 'sensor2lidar_translation': None,
                          'cam_intrinsic': None, 'distortion': None, 'vantage_lens': None}
            continue
        filled.add(name)
        T = inv_l @ EX._T_vehicle_camera(ctx, cm)                    # camera (optical) -> lidar
        pin = EX._pinhole(cm)
        cams[name] = {'data_path': f'{name}/<frame>.jpg', 'sensor2lidar_rotation': _mat(T[:3, :3], 10),
                      'sensor2lidar_translation': [_r(v, 6) for v in T[:3, 3]], 'cam_intrinsic': _mat(cm.K, 6),
                      'distortion': [0.0] * 5, 'vantage_lens': dict(_lens_sidecar(cm), exact_in_distortion=pin)}
    x, y, z, w = quat_from_matrix(T_vl[:3, :3])
    doc = {'lidar2ego_translation': [_r(v, 6) for v in T_vl[:3, 3]],
           'lidar2ego_rotation': [_r(w, 10), _r(x, 10), _r(y, 10), _r(z, 10)], 'cams': cams,
           'vantage': {'project': project.name, 'variant': ctx.variant.name, 'exported': _stamp(),
                       'lidar': ref[1].tag if ref else None,
                       'slots': {n: slots[n].cam.tag for n in slots},
                       'unassigned_cameras': [cm.cam.tag for cm in ctx.models if cm not in slots.values()],
                       'note': 'sensor2lidar maps camera optical-frame points (x right, y down, z forward) into the '
                               'lidar frame (REP 103 at its optical centre; the vehicle frame when the rig has no '
                               'lidar); lidar2ego into the ego frame (ISO 8855 at the rear axle on the ground).'}}
    sc = ', '.join(f'{n.lower()}={n in filled}' for n in NAVSIM_SLOTS) + f', lidar_pc={ref is not None}'
    py = (f'"""NAVSIM SensorConfig for the Vantage rig {project.name} / {ctx.variant.name}: the slots it fills '
          f'(exported {_stamp()})."""\nfrom navsim.common.dataclasses import SensorConfig\n\n'
          f'sensor_config = SensorConfig({sc})\n')
    return [_write(out_dir, 'navsim/vantage_cams.json', EX._json(doc)),
            _write(out_dir, 'navsim/vantage_sensor_config.py', py)]


# ---------------------------------------------------------------- HUGSIM
def _euler_xyz_deg(R) -> list:
    from scipy.spatial.transform import Rotation
    return [float(v) for v in Rotation.from_matrix(np.asarray(R, float)).as_euler('XYZ', degrees=True)]


# nuScenes' LIDAR_TOP axes (x right, y forward, z up) in a REP 103 lidar frame; HUGSIM's code reads its lidar so
R_REP_NUSCLIDAR = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


def export_hugsim(project, engine, out_dir, result=None) -> list:
    """HUGSIM camera configuration (hyzhou404/HUGSIM @ 62c690d, configs/sim/nuscenes_camera.yaml): ``cams`` with each
    camera's ``extrinsics`` (``v2c_rot``/``v2c_trans``: vehicle -> camera, ``l2c_rot``/``l2c_trans``: lidar -> camera;
    rotations as scipy intrinsic 'XYZ' Euler angles in degrees, rt2pose in sim/utils/sim_utils.py) and ``intrinsics``
    (H, W, cx, cy, fovx, fovy in degrees: a pinhole), ``cam_rect`` (identity) and ``cam_align`` (the mosaic layout).

    The vehicle frame is nuScenes' ego frame (checked: HUGSIM's CAM_BACK v2c puts that camera at (0.055, 0.011,
    1.568) m, nuScenes' figure), Vantage's rear-axle frame. Uncertain: the lidar frame is taken as nuScenes' LIDAR_TOP
    axes (x right, y forward, z up; HUGSIM's traj2control says so) at the first lidar. HUGSIM renders every camera
    relative to CAM_FRONT, so the camera nearest straight ahead is CAM_FRONT; the others take nuScenes' names by
    nearest yaw (CAM_<TAG> past six). Its 3DGS rasteriser is a pinhole: a fisheye camera is written as the pinhole
    of the same focal length and principal point (its field is then 2·atan(W / 2f), not the lens's)."""
    import yaml
    ctx = _ctx(project, result)
    ref = _ref_lidar(ctx)
    T_vl = ref[3].copy() if ref else np.eye(4)
    T_vl[:3, :3] = T_vl[:3, :3] @ R_REP_NUSCLIDAR
    slots = assign_slots(ctx.models, NUSCENES_SLOTS)
    named = {cm.cam.uid: n for n, cm in slots.items()}
    cams = {}
    for cm in ctx.models:
        name = named.get(cm.cam.uid, f'CAM_{cm.tag.upper()}')
        T_vc = EX._T_vehicle_camera(ctx, cm)
        v2c = EX._T_inv(T_vc)
        l2c = v2c @ T_vl
        W, H = int(cm.intr.width), int(cm.intr.height)
        cams[name] = {'extrinsics': {'l2c_rot': _euler_xyz_deg(l2c[:3, :3]), 'l2c_trans': [float(v) for v in l2c[:3, 3]],
                                     'v2c_rot': _euler_xyz_deg(v2c[:3, :3]), 'v2c_trans': [float(v) for v in v2c[:3, 3]]},
                      'intrinsics': {'H': H, 'W': W, 'cx': float(cm.cx), 'cy': float(cm.cy),
                                     'fovx': math.degrees(2 * math.atan(W / (2 * cm.fx))),
                                     'fovy': math.degrees(2 * math.atan(H / (2 * cm.fy)))}}
    front = [n for n in ('CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT') if n in cams]
    back = [n for n in ('CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT') if n in cams]
    rest = [n for n in cams if n not in front + back]
    doc = {'cams': cams, 'cam_rect': {'rot': [0.0, 0.0, 0.0], 'trans': [0.0, 0.0, 0.0]},
           'cam_align': [r for r in (front, back, rest) if r],
           # which Vantage camera fills each slot (HUGSIM reads only cams, cam_rect and cam_align; this is for people)
           'vantage': {'project': project.name, 'variant': ctx.variant.name,
                       'slots': {n: slots[n].cam.tag for n in cams if n in slots}}}
    head = (f'# HUGSIM camera configuration for the Vantage rig {project.name} / {ctx.variant.name}, exported '
            f'{_stamp()}.\n# v2c: vehicle (nuScenes ego: rear axle, ground, x forward) -> camera (OpenCV optical); l2c: '
            f'lidar ({ref[1].tag if ref else "none: the vehicle frame"}, nuScenes LIDAR_TOP axes) -> camera.\n'
            f'# Rotations: scipy intrinsic XYZ Euler, degrees. Pinhole intrinsics: a fisheye lens is its same-focal '
            f'pinhole.\n')
    if 'CAM_FRONT' not in cams:
        head += '# WARNING: no camera looks ahead: HUGSIM renders relative to CAM_FRONT and needs one.\n'
    return [_write(out_dir, 'hugsim/vantage_camera.yaml', head + yaml.safe_dump(doc, sort_keys=False))]


# ---------------------------------------------------------------- MMDetection3D
def ego_mask(project, engine, cm, scale=4):
    """(mask (H, W) bool: the own vehicle or rig, out (H, W) bool: outside the lens's image circle) at the camera's full
    resolution, from core.asbuilt.sim_mask (each pixel's ray looked up in the engine's occlusion cube at the camera)
    on a grid `scale` times coarser, each cell repeated over its pixels."""
    from .asbuilt import sim_mask
    W, H = int(cm.intr.width), int(cm.intr.height)
    ha, wa = int(math.ceil(H / scale)), int(math.ceil(W / scale))
    body, circle, _ = sim_mask(project, cm.cam, cm.intr, (ha, wa), engine)
    up = lambda a: np.repeat(np.repeat(a, scale, 0), scale, 1)[:H, :W]      # noqa: E731
    return up(body), ~up(circle)


def largest_free_rect(bad: np.ndarray) -> tuple:
    """(x0, y0, x1, y1), end exclusive, of the largest axis-aligned rectangle with no True in `bad` (the histogram-stack
    method, O(H·W)); (0, 0, 0, 0) when every pixel is bad."""
    H, W = bad.shape
    h = np.zeros(W, int)
    best, rect = 0, (0, 0, 0, 0)
    for y in range(H):
        h = np.where(bad[y], 0, h + 1)
        st = []
        for x in range(W + 1):
            cur = h[x] if x < W else 0
            start = x
            while st and st[-1][1] >= cur:
                s, hh = st.pop()
                area = hh * (x - s)
                if area > best:
                    best, rect = area, (s, y - hh + 1, x, y + 1)
                start = s
            st.append((start, cur))
    return rect


def ego_crop(mask: np.ndarray, out: np.ndarray, scale=4) -> dict:
    """The crops a training pipeline needs to keep the own vehicle out: the largest rectangle free of it (and inside the
    image circle) and the rows above the highest vehicle pixel in the lower half (a hood or bumper crop)."""
    H, W = mask.shape
    bad = mask | out
    ha, wa = int(math.ceil(H / scale)), int(math.ceil(W / scale))
    pad = np.ones((ha * scale, wa * scale), bool)
    pad[:H, :W] = bad
    coarse = pad.reshape(ha, scale, wa, scale).any(axis=(1, 3))          # a cell is bad if any of its pixels is
    x0, y0, x1, y1 = largest_free_rect(coarse)
    rect = [int(x0 * scale), int(y0 * scale), int(min(x1 * scale, W)), int(min(y1 * scale, H))]
    lower = np.nonzero(mask[H // 2:].any(axis=1))[0]
    keep_rows = int(H // 2 + lower[0]) if len(lower) else H
    return {'rect_xyxy': rect, 'keep_rows_above': keep_rows, 'ego_share': float(mask.mean())}


def _png(mask: np.ndarray) -> bytes:
    import cv2
    ok, buf = cv2.imencode('.png', mask.astype(np.uint8) * 255)
    if not ok:
        raise OSError('could not encode the mask PNG')
    return buf.tobytes()


def export_mmdet3d(project, engine, out_dir, result=None, data_root='data/vantage/') -> list:
    """MMDetection3D (open-mmlab/mmdetection3d @ fe25f7a): ``vantage_rig_infos.json`` with the rig in the v2 info keys
    (``images[CAM].cam2img`` 3x3, ``cam2ego`` 4x4, ``lidar2cam`` 4x4; ``lidar_points.lidar2ego`` 4x4) for your info
    converter to copy into every sample, a dataset config ``vantage_nus_multiview.py`` (``NuScenesDataset`` with the BEV
    multi-view pipeline of the nuScenes guide and ``data_prefix`` per camera), and per camera an ego mask PNG
    (255 = own vehicle) with its crops.

    Uncertain/limits: ``NuScenesDataset`` knows nuScenes' six camera names, so the cameras take them by nearest
    nominal yaw (the others are left out of the config, listed in the infos); MMDetection3D has no transform that
    crops each view at its own rectangle, so the crops are given (``vantage_ego_crops``) for a transform of your own;
    the lidar frame is the first lidar's (REP 103: x forward, y left, z up, MMDetection3D's LiDAR axes); the cameras
    are fisheye: undistort with the Kannala-Brandt terms in the infos before using cam2img as a pinhole."""
    ctx = _ctx(project, result)
    ref = _ref_lidar(ctx)
    T_vl = ref[3] if ref else np.eye(4)
    slots = assign_slots(ctx.models, NUSCENES_SLOTS)
    named = {cm.cam.uid: n for n, cm in slots.items()}
    images, crops, out = {}, {}, []
    for cm in ctx.models:
        name = named.get(cm.cam.uid, f'CAM_{cm.tag.upper()}')
        T_ec = EX._T_vehicle_camera(ctx, cm)
        images[name] = {'cam2img': _mat(cm.K, 6), 'cam2ego': _mat(T_ec, 9),
                        'lidar2cam': _mat(EX._T_inv(T_ec) @ T_vl, 9), 'vantage_lens': _lens_sidecar(cm),
                        'in_config': name in slots}
        if engine is not None:
            mask, outside = ego_mask(project, engine, cm)
            crops[name] = ego_crop(mask, outside)
            out.append(_write(out_dir, f'mmdet3d/ego_masks/{name}.png', _png(mask)))
    infos = {'metainfo': {'classes': NUSCENES_CLASSES, 'vantage_project': project.name,
                          'vantage_variant': ctx.variant.name, 'exported': _stamp()},
             'images': images,
             'lidar_points': {'lidar2ego': _mat(T_vl, 9), 'num_pts_feats': 4,
                              'vantage_lidar': ref[1].tag if ref else None},
             'vantage_ego_crops': crops}
    out.append(_write(out_dir, 'mmdet3d/vantage_rig_infos.json', EX._json(infos)))
    names = [n for n in NUSCENES_SLOTS if n in slots]
    prefix = ',\n    '.join(f"{n}='samples/{n}'" for n in names)
    crops_py = {n: crops[n]['rect_xyxy'] for n in names if n in crops}
    cfg = f'''# MMDetection3D dataset config for the Vantage rig {project.name} / {ctx.variant.name} (exported {_stamp()}).
# Layout and keys follow configs/_base_/datasets/nus-mono3d.py and the BEV multi-view pipeline of
# docs/en/advanced_guides/datasets/nuscenes.md (mmdetection3d @ fe25f7a). Build your ann_file pickles with your info
# converter, copying cam2img / cam2ego / lidar2cam / lidar2ego from vantage_rig_infos.json into every sample.
dataset_type = 'NuScenesDataset'
data_root = {data_root!r}
class_names = {NUSCENES_CLASSES!r}
metainfo = dict(classes=class_names)
input_modality = dict(use_lidar={ref is not None}, use_camera=True)
backend_args = None
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
data_prefix = dict(
    pts={'samples/LIDAR' if ref else ''!r},
    {prefix})
train_transforms = [
    dict(type='PhotoMetricDistortion3D'),
]
train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True, num_views={len(names)}),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
    dict(type='MultiViewWrapper', transforms=train_transforms),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='Pack3DDetInputs', keys=['img', 'gt_bboxes_3d', 'gt_labels_3d'])
]
train_dataloader = dict(
    batch_size=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='vantage_infos_train.pkl',
        data_prefix=data_prefix,
        metainfo=metainfo,
        modality=input_modality,
        pipeline=train_pipeline,
        test_mode=False,
        backend_args=backend_args))
# Vantage: per camera the largest rectangle (x0, y0, x1, y1, pixels, end exclusive) free of the own vehicle and inside
# the image circle; MMDetection3D has no per-view fixed crop, so apply these in a transform of your own (the masks are
# ego_masks/<CAM>.png, 255 = own vehicle).
vantage_ego_crops = {crops_py!r}
vantage_unassigned_cameras = {[cm.cam.tag for cm in ctx.models if cm.cam.uid not in named]!r}
'''
    out.append(_write(out_dir, 'mmdet3d/vantage_nus_multiview.py', cfg))
    return out


# ---------------------------------------------------------------- OpenPCDet
def pcdet_ranges(height_m: float, range_m: float, vxy=0.1, vz=0.15) -> list:
    """POINT_CLOUD_RANGE in the lidar's own frame obeying the custom-dataset tutorial: x and y range / voxel a multiple
    of 16 (half range a multiple of 8 voxels: 0.8 m at 0.1 m), z range / voxel = 40 (6.0 m at 0.15 m), from 0.3 m below
    the ground (the lidar `height_m` above it)."""
    step = 8 * vxy
    r = max(step, math.floor(min(range_m, 75.2) / step + 1e-9) * step)
    z0 = round(-(height_m + 0.3), 2)
    return [_r(-r, 4), _r(-r, 4), _r(z0, 4), _r(r, 4), _r(r, 4), _r(z0 + 40 * vz, 4)]


def export_openpcdet(project, engine, out_dir, result=None) -> list:
    """OpenPCDet custom-dataset config (open-mmlab/OpenPCDet @ 233f849, tools/cfgs/dataset_configs/custom_dataset.yaml):
    the template's keys (DATASET, DATA_PATH, POINT_CLOUD_RANGE, MAP_CLASS_TO_KITTI, DATA_SPLIT, INFO_PATH,
    POINT_FEATURE_ENCODING, DATA_AUGMENTOR, DATA_PROCESSOR with VOXEL_SIZE) with the range and voxels from the first
    lidar: points in the lidar's own frame (REP 103: OpenPCDet's unified x forward, y left, z up), the range the lidar's
    stated range (at most the template's 75.2 m) rounded so x and y range / voxel is a multiple of 16, z from 0.3 m
    below the ground to 6 m above that (z range / voxel = 40). The lidar's pose in the vehicle frame is in the comments
    and in vantage_lidar.json (OpenPCDet's custom dataset has no key for it)."""
    import yaml
    ctx = _ctx(project, result)
    ref = _ref_lidar(ctx)
    if ref is None:
        raise ValueError('The variant has no lidar: OpenPCDet needs one.')
    tag, s, m, T = ref
    rng = float(m.range_max_m or m.range_m or 75.2)
    pcr = pcdet_ranges(float(T[2, 3]), rng)
    doc = {'DATASET': 'CustomDataset', 'DATA_PATH': '../data/custom', 'POINT_CLOUD_RANGE': pcr,
           'MAP_CLASS_TO_KITTI': {'Vehicle': 'Car', 'Pedestrian': 'Pedestrian', 'Cyclist': 'Cyclist'},
           'DATA_SPLIT': {'train': 'train', 'test': 'val'},
           'INFO_PATH': {'train': ['custom_infos_train.pkl'], 'test': ['custom_infos_val.pkl']},
           'POINT_FEATURE_ENCODING': {'encoding_type': 'absolute_coordinates_encoding',
                                      'used_feature_list': ['x', 'y', 'z', 'intensity'],
                                      'src_feature_list': ['x', 'y', 'z', 'intensity']},
           'DATA_AUGMENTOR': {'DISABLE_AUG_LIST': ['placeholder'], 'AUG_CONFIG_LIST': [
               {'NAME': 'random_world_flip', 'ALONG_AXIS_LIST': ['x', 'y']},
               {'NAME': 'random_world_rotation', 'WORLD_ROT_ANGLE': [-0.78539816, 0.78539816]},
               {'NAME': 'random_world_scaling', 'WORLD_SCALE_RANGE': [0.95, 1.05]}]},
           'DATA_PROCESSOR': [
               {'NAME': 'mask_points_and_boxes_outside_range', 'REMOVE_OUTSIDE_BOXES': True},
               {'NAME': 'shuffle_points', 'SHUFFLE_ENABLED': {'train': True, 'test': False}},
               {'NAME': 'transform_points_to_voxels', 'VOXEL_SIZE': [0.1, 0.1, 0.15], 'MAX_POINTS_PER_VOXEL': 5,
                'MAX_NUMBER_OF_VOXELS': {'train': 150000, 'test': 150000}}]}
    roll, pitch, yaw = rpy_from_matrix(T[:3, :3])
    head = (f'# OpenPCDet custom dataset config for lidar {s.tag} ({m.vendor} {m.name}) of the Vantage rig '
            f'{project.name} / {ctx.variant.name}, exported {_stamp()}.\n'
            f'# Points in the lidar frame (x forward, y left, z up at its optical centre, {T[2, 3]:.3f} m above the '
            f'ground).\n# Lidar in the vehicle frame (rear axle, ground): xyz {T[0, 3]:.4f} {T[1, 3]:.4f} {T[2, 3]:.4f} m, '
            f'rpy {roll:.6f} {pitch:.6f} {yaw:.6f} rad.\n# The gt_sampling augmentation of the template needs your '
            f'dbinfos; add it back once you have them.\n')
    side = {'lidar': s.tag, 'model': f'{m.vendor} {m.name}'.strip(), 'T_vehicle_lidar': _mat(T, 9),
            'stated_range_m': rng, 'frame': 'REP 103 at the optical centre; vehicle: ISO 8855 rear axle on the ground'}
    return [_write(out_dir, 'openpcdet/vantage_custom_dataset.yaml', head + yaml.safe_dump(doc, sort_keys=False)),
            _write(out_dir, 'openpcdet/vantage_lidar.json', EX._json(side))]


# ---------------------------------------------------------------- LeRobot
LEROBOT_DEFAULT_FEATURES = {
    'timestamp': {'dtype': 'float32', 'shape': [1], 'names': None},
    'frame_index': {'dtype': 'int64', 'shape': [1], 'names': None},
    'episode_index': {'dtype': 'int64', 'shape': [1], 'names': None},
    'index': {'dtype': 'int64', 'shape': [1], 'names': None},
    'task_index': {'dtype': 'int64', 'shape': [1], 'names': None},
}


def export_lerobot(project, engine, out_dir, result=None, fps=None, use_videos=True) -> list:
    """LeRobot dataset metadata (huggingface/lerobot @ ba83e4c, codebase v3.0): ``meta/info.json`` as
    ``create_empty_dataset_info`` makes it (``DatasetInfo``: codebase_version, fps, features, zero counters, chunk and
    file-size defaults, the data and video path templates, robot_type, splits) with one ``observation.images.<camera>``
    feature per camera (dtype video or image, shape [height, width, 3], names [height, width, channel]) and
    LeRobot's DEFAULT_FEATURES, and ``meta/vantage_calibration.json``: every camera's intrinsics, lens and pose.

    LeRobot stores no camera calibration (its info.json has no field for it): the sidecar is for your own loader, and
    LeRobot ignores it. The video_info LeRobot adds when it encodes videos is left for it to write."""
    ctx = _ctx(project, result)
    fps = int(fps or min((int(round(cm.module.fps)) for cm in ctx.models if cm.module.fps), default=30))
    feats = {}
    cal = {}
    for cm in ctx.models:
        key = f'observation.images.{EX._ros_name(cm.tag)}'
        feats[key] = {'dtype': 'video' if use_videos else 'image',
                      'shape': [int(cm.intr.height), int(cm.intr.width), 3], 'names': ['height', 'width', 'channel']}
        cal[key] = {'camera': cm.cam.tag, 'K': _mat(cm.K, 6), 'lens': _lens_sidecar(cm),
                    'T_vehicle_camera': _mat(EX._T_vehicle_camera(ctx, cm), 9),
                    'frame': 'optical (x right, y down, z forward) into ISO 8855 rear axle on the ground'}
    feats.update(LEROBOT_DEFAULT_FEATURES)
    info = {'codebase_version': 'v3.0', 'robot_type': 'vantage_sensor_rig', 'total_episodes': 0, 'total_frames': 0,
            'total_tasks': 0, 'chunks_size': 1000, 'data_files_size_in_mb': 100, 'video_files_size_in_mb': 200,
            'fps': fps, 'splits': {},
            'data_path': 'data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet',
            'video_path': 'videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4' if use_videos else None,
            'features': feats}
    side = {'project': project.name, 'variant': ctx.variant.name, 'exported': _stamp(), 'cameras': cal,
            'note': 'LeRobot does not read this file; it is the rig calibration for your own loader.'}
    return [_write(out_dir, 'lerobot/meta/info.json', json.dumps(info, indent=4)),
            _write(out_dir, 'lerobot/meta/vantage_calibration.json', EX._json(side))]


# ---------------------------------------------------------------- all of them
EXPORTERS = {'autoware': export_autoware, 'navsim': export_navsim, 'hugsim': export_hugsim,
             'mmdet3d': export_mmdet3d, 'openpcdet': export_openpcdet, 'lerobot': export_lerobot}


def export_stack(project, engine, out_dir, targets=TARGETS, result=None) -> dict:
    """{target: [paths]} for each target asked (a target that cannot be written, such as OpenPCDet without a lidar,
    gets {'error': message} instead of paths)."""
    out = {}
    for t in targets:
        fn = EXPORTERS[t]
        try:
            out[t] = fn(project, engine, out_dir, result=result)
        except ValueError as e:
            out[t] = {'error': str(e)}
    return out
