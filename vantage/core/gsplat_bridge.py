"""3D Gaussian Splatting bridge: a Vantage rig as the camera and lidar configuration of Nerfstudio, NVIDIA 3DGRUT /
3DGUT and the neural AD simulators, for replaying a logged drive through a rig that was never built (novel-rig log
replay), and the rig's timing (rolling shutter, exposure, lidar sweep) those tools need.

Formats, read from their sources (2026-11-04):

- **Nerfstudio** ``transforms.json``
  (https://github.com/nerfstudio-project/nerfstudio/blob/main/docs/quickstart/data_conventions.md,
  nerfstudio/data/dataparsers/nerfstudio_dataparser.py, nerfstudio/cameras/cameras.py): ``camera_model`` is global
  (OPENCV, OPENCV_FISHEYE, EQUIRECTANGULAR, FISHEYE624; per-frame camera_model is not supported), intrinsics
  ``fl_x fl_y cx cy w h k1 k2 k3 k4`` at the top or per frame (a per-frame field must be given for every frame);
  ``transform_matrix`` is camera-to-world with OpenGL camera axes (+X right, +Y up, +Z back: the camera looks down
  -Z), world +Z up; pixel (0, 0)'s ray goes through its centre (get_image_coords adds 0.5): cx is in Vantage's own
  convention (pixel centres at i + 0.5), not OpenCV's (centres at i).
- **3DGRUT / 3DGUT** (https://github.com/nv-tlabs/3dgrut: README, threedgrut/datasets/dataset_colmap.py,
  datasetNcore.py): trains from COLMAP-style folders (``sparse/0/cameras.txt, images.txt, points3D.txt`` next to the
  images; multiple sensors supported; camera models PINHOLE, SIMPLE_PINHOLE, SIMPLE_RADIAL, RADIAL, OPENCV,
  FULL_OPENCV, OPENCV_FISHEYE) with ``python train.py --config-name apps/colmap_3dgut.yaml path=<folder>``; its
  COLMAP loader treats every image as a global shutter. Rolling shutter and spinning lidars come in through NVIDIA's
  NCore v4 sequences (https://github.com/NVIDIA/ncore: ncore/impl/data/types.py ``OpenCVFisheyeCameraModelParameters``
  {resolution, shutter_type, principal_point, focal_length, radial_coeffs, max_angle}, ``ShutterType``
  {ROLLING_TOP_TO_BOTTOM, ROLLING_LEFT_TO_RIGHT, ROLLING_BOTTOM_TO_TOP, ROLLING_RIGHT_TO_LEFT, GLOBAL};
  ncore/impl/sensors/lidar.py ``RowOffsetStructuredSpinningLidarModel`` {row_elevations_rad, column_azimuths_rad,
  row_azimuth_offsets_rad, spinning_frequency_hz, spinning_direction 'cw'/'ccw'}). An NCore v4 sequence is a
  component store written with NCore's own tools; Vantage writes the sensor parameters they take
  (``ncore_sensors.json``), not a sequence.
- **COLMAP** text model (https://github.com/colmap/colmap/blob/main/doc/format.rst, doc/cameras.rst): images.txt
  holds the world-to-camera pose (QW QX QY QZ TX TY TZ, OpenCV camera axes: x right, y down, z forward); pixel
  coordinates are corner-based (the top-left pixel's centre is at (0.5, 0.5)): Vantage's convention.
- **SplatAD / NeuRAD** (https://github.com/georghess/neurad-studio: nerfstudio/data/dataparsers/ad_dataparser.py,
  nerfstudio/cameras/lidars.py, nerfstudio/data/utils/lidar_elevation_mappings.py): cameras carry
  ``rolling_shutter_time`` and ``time_to_center_pixel`` (seconds); lidars are known types with a per-channel elevation
  mapping {channel: degrees} and an azimuth resolution (degrees). rig_timing.json gives both for every camera and
  lidar of the rig; a new lidar type is added to neurad-studio's tables by hand.
- **HUGSIM** (https://github.com/hyzhou404/HUGSIM: configs/sim/nuscenes_camera.yaml, sim/utils/sim_utils.py):
  pinhole cameras given by fovx/fovy/cx/cy/H/W and Euler 'XYZ' extrinsics; its vehicle frame is the dataset's.
  Not written: its cameras are pinhole only and its vehicle frame is not documented beyond the datasets it ships.

Lens: Vantage's Kannala–Brandt model is OpenCV's fisheye model (r = f·θ(1 + k1θ² + k2θ⁴ + k3θ⁶ + k4θ⁸)); other
projections are fitted to it (core.exporters.kb_coefficients; the fit's RMS in pixels is written with each camera).

Frames: the vehicle frame is ISO 8855 (x forward, y left, z up, rear-axle centre on the ground); a camera's optical
frame is OpenCV's (x right, y down, z forward), Nerfstudio's camera frame is OpenGL's (x right, y up, z back):
c2w_gl = [R_vehicle_optical · diag(1, -1, -1) | C]. With a trajectory the world is the first ego pose's vehicle
frame (z up), each frame's sensor pose the ego pose at that sensor's own time (trigger, plus the middle row's
mid-exposure for a rolling shutter).
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

from .mathutil import quat_from_matrix, rot_zyx, BODY_TO_OPTICAL

SCHEMA = 'vantage.gsplat_bridge/1'
GL_FROM_CV = np.diag([1.0, -1.0, -1.0])           # OpenCV camera axes -> OpenGL camera axes (and back)


# ---------------------------------------------------------------- cameras
def camera_params(project, cam) -> dict:
    """OpenCV-fisheye parameters of a camera in the corner-based pixel convention (COLMAP, Nerfstudio, NCore):
    fx, fy, cx, cy, k1..k4, w, h, the fit's RMS (0 for a native Kannala–Brandt lens), the image circle (rad) and
    whether the lens is rectilinear."""
    from . import exporters as EX
    cm = EX.camera_model(project, cam)
    k = [float(x) for x in cm.k]
    return {'fx': float(cm.fx), 'fy': float(cm.fy), 'cx': float(cm.cx) + 0.5, 'cy': float(cm.cy) + 0.5,
            'k1': k[0], 'k2': k[1], 'k3': k[2], 'k4': k[3], 'w': int(cm.intr.width), 'h': int(cm.intr.height),
            'fit_rms_px': float(cm.fit_rms_px), 'max_angle_rad': float(cm.intr.max_theta),
            'projection': cm.intr.projection, 'rectilinear': cm.intr.projection == 'rectilinear'}


def project_opencv_fisheye(par, d_cv):
    """Pixel (u, v) (corner-based) of OpenCV-frame directions (K, 3) through the OpenCV fisheye model with par."""
    d = np.asarray(d_cv, float).reshape(-1, 3)
    a, b = d[:, 0] / d[:, 2], d[:, 1] / d[:, 2]
    r = np.hypot(a, b)
    th = np.arctan(r)
    th2 = th * th
    thd = th * (1 + par['k1'] * th2 + par['k2'] * th2 ** 2 + par['k3'] * th2 ** 3 + par['k4'] * th2 ** 4)
    s = np.where(r > 1e-12, thd / np.where(r > 1e-12, r, 1.0), 1.0)
    return par['fx'] * a * s + par['cx'], par['fy'] * b * s + par['cy']


def optical_to_vehicle(o) -> np.ndarray:
    """R_vehicle_optical of a camera (its OpenCV optical axes in the vehicle frame)."""
    R = rot_zyx(o.yaw, o.pitch, o.roll)
    return (BODY_TO_OPTICAL @ R.T).T


def c2w_opengl(R_v_opt, C, T_world_vehicle=None) -> np.ndarray:
    """Nerfstudio's 4×4 camera-to-world (OpenGL camera axes) of a camera with optical axes R_v_opt at C (vehicle
    frame), in the world of T_world_vehicle (identity: the vehicle frame)."""
    T = np.eye(4)
    T[:3, :3] = np.asarray(R_v_opt, float) @ GL_FROM_CV
    T[:3, 3] = np.asarray(C, float)
    return (T_world_vehicle if T_world_vehicle is not None else np.eye(4)) @ T


def w2c_opencv(R_v_opt, C, T_world_vehicle=None):
    """COLMAP's world-to-camera (R (3, 3), t (3,)) in OpenCV camera axes."""
    T = np.eye(4)
    T[:3, :3] = np.asarray(R_v_opt, float)
    T[:3, 3] = np.asarray(C, float)
    Tw = (T_world_vehicle if T_world_vehicle is not None else np.eye(4)) @ T
    R = Tw[:3, :3].T
    return R, -R @ Tw[:3, 3]


# ---------------------------------------------------------------- trajectories
def constant_velocity_trajectory(speed_mps=10.0, yaw_rate_dps=0.0, duration_s=2.0, rate_hz=10.0) -> list:
    """[(t_s, T_world_ego 4×4)]: a template drive at constant speed and yaw rate (core.motion_footprint.ego_pose),
    the world being the vehicle frame at t = 0."""
    from .motion_footprint import ego_pose
    t = np.arange(max(int(round(duration_s * rate_hz)), 1) + 1) / rate_hz
    psi, x, y = ego_pose(t, speed_mps, yaw_rate_dps)
    out = []
    for ti, p, xi, yi in zip(t, psi, x, y):
        T = np.eye(4)
        T[:3, :3] = rot_zyx(math.degrees(p), 0.0, 0.0)
        T[:2, 3] = (xi, yi)
        out.append((float(ti), T))
    return out


def trajectory_from_nuscenes(dataroot, version=None) -> list:
    """[(t_s, T_world_ego)] from a nuScenes folder's ego_pose.json (e.g. one core.dataset wrote), time from its first
    pose, the world moved to that first pose."""
    if version is None:
        cands = [d for d in sorted(os.listdir(dataroot)) if os.path.isfile(os.path.join(dataroot, d, 'ego_pose.json'))]
        if not cands:
            raise FileNotFoundError(f'no */ego_pose.json under {dataroot}')
        version = cands[0]
    rows = json.load(open(os.path.join(dataroot, version, 'ego_pose.json'), encoding='utf-8'))
    rows = sorted(rows, key=lambda r: r['timestamp'])
    out, T0inv, t0 = [], None, None
    for r in rows:
        w, x, y, z = r['rotation']
        T = np.eye(4)
        T[:3, :3] = _quat_matrix(w, x, y, z)
        T[:3, 3] = r['translation']
        if T0inv is None:
            T0inv, t0 = np.linalg.inv(T), r['timestamp']
        out.append(((r['timestamp'] - t0) / 1e6, T0inv @ T))
    return out


def _quat_matrix(w, x, y, z):
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def interp_pose(traj, t_s) -> np.ndarray:
    """The ego pose at t_s: translation linear, rotation by slerp between the two poses around it (clamped)."""
    from scipy.spatial.transform import Rotation, Slerp
    ts = np.array([t for t, _ in traj])
    if len(traj) == 1 or t_s <= ts[0]:
        return traj[0][1].copy()
    if t_s >= ts[-1]:
        return traj[-1][1].copy()
    i = int(np.searchsorted(ts, t_s)) - 1
    (ta, A), (tb, B) = traj[i], traj[i + 1]
    f = (t_s - ta) / max(tb - ta, 1e-12)
    T = np.eye(4)
    T[:3, :3] = Slerp([0, 1], Rotation.from_matrix([A[:3, :3], B[:3, :3]]))(f).as_matrix()
    T[:3, 3] = A[:3, 3] * (1 - f) + B[:3, 3] * f
    return T


# ---------------------------------------------------------------- timing
def rig_timing(project) -> dict:
    """The rig's timing: per camera its trigger (ms after the sync pulse, from the sync plan), readout and exposure,
    shutter (NCore ShutterType name), and neurad-studio's rolling_shutter_time and time_to_center_pixel (s, relative to
    the image timestamp = the trigger); per spinning lidar its rate, rotation, phase, beam elevations and azimuth
    resolution (neurad-studio's elevation mapping and NCore's row/column angles). Channels are numbered from the lowest
    beam up: the firmware's own channel order may differ."""
    from . import timing as TM
    from . import sensors as SN
    from .motion_footprint import camera_trigger
    st = TM.TimingSettings.from_dict(getattr(project.analysis, 'timing', None))
    try:
        pl = TM.plan(project, settings=st)
    except Exception:
        pl = None
    cams = []
    for c in project.variant.cameras:
        if not c.visible:
            continue
        e = pl.cam(c.uid) if pl is not None else None
        trig, free = camera_trigger(e, 0.0) if e is not None else (0.0, True)
        ro, ex = float(st.readout_ms), float(st.exposure_ms)
        cams.append({'tag': c.tag, 'uid': c.uid, 'trigger_ms': float(trig), 'free_running': bool(free),
                     'frame_ms': float((e or {}).get('frame_ms') or 0.0), 'readout_ms': ro, 'exposure_ms': ex,
                     'shutter_type': 'ROLLING_TOP_TO_BOTTOM' if ro > 0 else 'GLOBAL',
                     'neurad': {'rolling_shutter_time': ro / 1000.0,
                                'time_to_center_pixel': (ro / 2.0 + ex / 2.0) / 1000.0}})
    lids = []
    for s in project.variant.sensors:
        if not s.visible or s.kind != 'lidar':
            continue
        m = project.sensor_model_for(s)
        el = np.asarray(m.elevations(), float)
        h_res = float(getattr(m, 'h_res', 0.0) or 0.0)
        az = np.arange(0.0, float(m.h_fov or 360.0), h_res) if h_res > 0 else np.zeros(0)
        sw = TM.Sweep.of(s, m) if getattr(m, 'scan', '') == 'spinning' else None
        R = SN.rotation(s)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = s.pos()
        lids.append({'tag': s.tag, 'uid': s.uid, 'model': f'{m.vendor} {m.name}'.strip(), 'scan': m.scan,
                     'rate_hz': float(m.rate_hz), 'period_ms': 1000.0 / float(m.rate_hz) if m.rate_hz else 0.0,
                     'rotation': m.rotation, 'phase_deg': float(s.phase_deg or 0.0),
                     'channels': int(len(el)), 'elevations_deg': [round(float(x), 6) for x in el],
                     'h_res_deg': h_res, 'h_fov_deg': float(m.h_fov or 360.0),
                     'T_vehicle_lidar': np.round(T, 9).tolist(),
                     'neurad': {'elevation_mapping': {str(i): round(float(x), 6) for i, x in enumerate(el)},
                                'azimuth_resolution_deg': h_res},
                     'ncore': {'model': 'RowOffsetStructuredSpinningLidarModel',
                               'row_elevations_rad': [round(math.radians(float(x)), 9) for x in el],
                               'column_azimuths_rad': [round(math.radians(float(x)), 9) for x in az],
                               'row_azimuth_offsets_rad': [0.0] * len(el),
                               'spinning_frequency_hz': float(m.rate_hz), 'spinning_direction': m.rotation}
                     if sw is not None else None})
    return {'schema': SCHEMA, 'project': project.name, 'variant': project.variant.name,
            'reference': 'times in ms after the sync pulse of the sync plan (Analysis ▸ Sync and timing); readout '
                         'and exposure from its settings (one figure for every camera)',
            'cameras': cams, 'lidars': lids}


# ---------------------------------------------------------------- the frames
def _frames(project, trajectory, timing):
    """[(sample k, t_s of the sample, camera, its params, R_v_opt, C, T_world_ego at the camera's time, cam_t_s)]."""
    tim = {c['uid']: c for c in timing['cameras']}
    traj = trajectory or [(0.0, np.eye(4))]
    out = []
    for k, (t, _) in enumerate(traj):
        for c in project.variant.cameras:
            if not c.visible:
                continue
            ti = tim.get(c.uid, {'trigger_ms': 0.0, 'readout_ms': 0.0, 'exposure_ms': 0.0})
            tc = t + (ti['trigger_ms'] + ti['readout_ms'] / 2.0 + ti['exposure_ms'] / 2.0) / 1000.0
            Tw = interp_pose(traj, tc) if trajectory else np.eye(4)
            out.append((k, t, c, optical_to_vehicle(c), np.asarray(c.pos(), float), Tw, tc))
    return out


def export_nerfstudio(project, out_dir, trajectory=None, timing=None) -> list[str]:
    """nerfstudio/transforms.json (+ README.md) in out_dir: one frame per camera and trajectory sample, per-frame
    intrinsics (the rig's cameras differ), the global camera model OPENCV_FISHEYE (OPENCV when every lens is
    rectilinear), file paths images/<TAG>/<sample>.jpg for the user's images."""
    timing = timing or rig_timing(project)
    d = os.path.join(out_dir, 'nerfstudio')
    os.makedirs(d, exist_ok=True)
    pars = {c.uid: camera_params(project, c) for c in project.variant.cameras if c.visible}
    model = 'OPENCV' if pars and all(p['rectilinear'] for p in pars.values()) else 'OPENCV_FISHEYE'
    frames = []
    for k, t, c, Rvo, C, Tw, tc in _frames(project, trajectory, timing):
        p = pars[c.uid]
        f = {'file_path': f'images/{c.tag}/{k:05d}.jpg', 'transform_matrix': np.round(c2w_opengl(Rvo, C, Tw), 9).tolist(),
             'fl_x': p['fx'], 'fl_y': p['fy'], 'cx': p['cx'], 'cy': p['cy'], 'w': p['w'], 'h': p['h'],
             'k1': p['k1'] if model == 'OPENCV_FISHEYE' else 0.0, 'k2': p['k2'] if model == 'OPENCV_FISHEYE' else 0.0,
             'k3': p['k3'] if model == 'OPENCV_FISHEYE' else 0.0, 'k4': p['k4'] if model == 'OPENCV_FISHEYE' else 0.0,
             'time': round(tc, 6), 'vantage_camera': c.tag, 'vantage_sample': k}
        if model == 'OPENCV':
            f.update({'p1': 0.0, 'p2': 0.0})
        frames.append(f)
    doc = {'camera_model': model, 'orientation_override': 'none', 'frames': frames,
           'vantage': {'schema': SCHEMA, 'project': project.name, 'variant': project.variant.name,
                       'world': 'the vehicle frame (ISO 8855) of the first ego pose, z up',
                       'fit_rms_px': {c.tag: pars[c.uid]['fit_rms_px'] for c in project.variant.cameras
                                      if c.uid in pars}}}
    p1 = os.path.join(d, 'transforms.json')
    with open(p1, 'w', encoding='utf-8') as fh:
        json.dump(doc, fh, indent=1)
    p2 = os.path.join(d, 'README.md')
    with open(p2, 'w', encoding='utf-8') as fh:
        fh.write(f'# Nerfstudio transforms for {project.name} / {project.variant.name}\n\n'
                 f'`transforms.json` holds {len(frames)} frames: every camera at every sample of the trajectory, '
                 f'with per-frame intrinsics ({model}, corner-based pixels) and camera-to-world matrices (OpenGL '
                 'camera axes, world z up, metres).\n\nPut each camera\'s images at `images/<TAG>/<sample>.jpg` '
                 '(5-digit sample numbers), or edit `file_path`. Each frame\'s pose is the ego pose at that camera\'s '
                 'middle-row mid-exposure (`time`, seconds); Nerfstudio itself renders a global shutter.\n\n'
                 'Train: `ns-train splatfacto --data .` (`orientation_override` "none" keeps the vehicle frame).\n')
    return [os.path.abspath(p1), os.path.abspath(p2)]


def _ground_points(trajectory, half=25.0, step=1.0):
    """A sparse ground grid (z = 0, grey) round the trajectory: only an initialisation for a splat trainer that needs
    a point cloud; replace it with the drive's lidar points."""
    xs = [T[0, 3] for _, T in trajectory] or [0.0]
    ys = [T[1, 3] for _, T in trajectory] or [0.0]
    gx = np.arange(min(xs) - half, max(xs) + half + 1e-9, step)
    gy = np.arange(min(ys) - half, max(ys) + half + 1e-9, step)
    X, Y = np.meshgrid(gx, gy)
    return np.c_[X.ravel(), Y.ravel(), np.zeros(X.size)]


def export_3dgut(project, out_dir, trajectory=None, timing=None) -> list[str]:
    """3dgut/: a COLMAP text model 3DGRUT's COLMAP loader reads (sparse/0/cameras.txt with one OPENCV_FISHEYE camera
    per rig camera, images.txt with one image per camera and sample, points3D.txt a ground grid to initialise from),
    ncore_sensors.json (the parameters NCore v4's camera and spinning-lidar models take, with rolling shutter) and a
    README."""
    timing = timing or rig_timing(project)
    d = os.path.join(out_dir, '3dgut')
    sp = os.path.join(d, 'sparse', '0')
    os.makedirs(sp, exist_ok=True)
    cams = [c for c in project.variant.cameras if c.visible]
    pars = {c.uid: camera_params(project, c) for c in cams}
    ids = {c.uid: i + 1 for i, c in enumerate(cams)}
    L = ['# Camera list with one line of data per camera:', '#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]',
         f'# Number of cameras: {len(cams)}  (Vantage {project.name} / {project.variant.name})']
    for c in cams:
        p = pars[c.uid]
        L.append(f'{ids[c.uid]} OPENCV_FISHEYE {p["w"]} {p["h"]} ' + ' '.join(
            f'{p[k]:.12g}' for k in ('fx', 'fy', 'cx', 'cy', 'k1', 'k2', 'k3', 'k4')))
    with open(os.path.join(sp, 'cameras.txt'), 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(L) + '\n')
    fr = _frames(project, trajectory, timing)
    L = ['# Image list with two lines of data per image:', '#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME',
         '#   POINTS2D[] as (X, Y, POINT3D_ID)', f'# Number of images: {len(fr)}']
    for n, (k, t, c, Rvo, C, Tw, tc) in enumerate(fr, start=1):
        R, tv = w2c_opencv(Rvo, C, Tw)
        x, y, z, w = quat_from_matrix(R)
        if w < 0:
            x, y, z, w = -x, -y, -z, -w
        L.append(f'{n} {w:.12g} {x:.12g} {y:.12g} {z:.12g} {tv[0]:.12g} {tv[1]:.12g} {tv[2]:.12g} {ids[c.uid]} '
                 f'{c.tag}/{k:05d}.jpg')
        L.append('')
    with open(os.path.join(sp, 'images.txt'), 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(L) + '\n')
    pts = _ground_points(trajectory or [(0.0, np.eye(4))])
    L = ['# 3D point list with one line of data per point:',
         '#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)',
         f'# Number of points: {len(pts)}  (a ground grid to start from: replace it with the drive\'s lidar points)']
    L += [f'{i + 1} {x:.4f} {y:.4f} {z:.4f} 128 128 128 1.0' for i, (x, y, z) in enumerate(pts)]
    with open(os.path.join(sp, 'points3D.txt'), 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(L) + '\n')
    tim = {c['uid']: c for c in timing['cameras']}
    ncore = {'schema': SCHEMA, 'note': 'Parameters for NCore v4 sensors (ncore.impl.data.types, ncore.impl.sensors); '
                                       'an NCore sequence is written with NCore\'s own tools. Pixels corner-based.',
             'cameras': [], 'lidars': []}
    for c in cams:
        p, ti = pars[c.uid], tim.get(c.uid, {})
        T = np.eye(4)
        T[:3, :3] = optical_to_vehicle(c)
        T[:3, 3] = c.pos()
        ncore['cameras'].append({
            'tag': c.tag, 'model': 'OpenCVFisheyeCameraModelParameters', 'resolution': [p['w'], p['h']],
            'shutter_type': ti.get('shutter_type', 'GLOBAL'), 'principal_point': [p['cx'], p['cy']],
            'focal_length': [p['fx'], p['fy']], 'radial_coeffs': [p['k1'], p['k2'], p['k3'], p['k4']],
            'max_angle': p['max_angle_rad'], 'T_vehicle_camera': np.round(T, 9).tolist(),
            'camera_axes': 'OpenCV (x right, y down, z forward)', 'readout_ms': ti.get('readout_ms', 0.0),
            'exposure_ms': ti.get('exposure_ms', 0.0), 'trigger_ms': ti.get('trigger_ms', 0.0),
            'fit_rms_px': p['fit_rms_px']})
    for lid in timing['lidars']:
        if lid['ncore'] is not None:
            ncore['lidars'].append({'tag': lid['tag'], **lid['ncore'], 'T_vehicle_lidar': lid['T_vehicle_lidar'],
                                    'lidar_axes': 'x forward, y left, z up'})
    pn = os.path.join(d, 'ncore_sensors.json')
    with open(pn, 'w', encoding='utf-8') as fh:
        json.dump(ncore, fh, indent=1)
    pr = os.path.join(d, 'README.md')
    with open(pr, 'w', encoding='utf-8') as fh:
        fh.write(f'# 3DGRUT / 3DGUT input for {project.name} / {project.variant.name}\n\n'
                 '`sparse/0/` is a COLMAP text model (one OPENCV_FISHEYE camera per rig camera, one image per camera '
                 'and trajectory sample, world-to-camera poses in OpenCV axes). Put the images at '
                 '`images/<TAG>/<sample>.jpg` next to `sparse/`, replace `points3D.txt` (a ground grid) with the '
                 'drive\'s lidar points, then:\n\n'
                 '    python train.py --config-name apps/colmap_3dgut.yaml path=<this folder> out_dir=runs '
                 'experiment_name=rig\n\n'
                 '3DGRUT\'s COLMAP loader treats every image as a global shutter. For rolling shutter and the '
                 'spinning lidar, build an NCore v4 sequence with NCore\'s tools from `ncore_sensors.json`.\n')
    return [os.path.abspath(os.path.join(sp, n)) for n in ('cameras.txt', 'images.txt', 'points3D.txt')] + \
        [os.path.abspath(pn), os.path.abspath(pr)]


def export_gsplat_bridge(project, engine=None, out_dir='.', trajectory=None, speed_mps=10.0, yaw_rate_dps=0.0,
                         duration_s=2.0, rate_hz=10.0) -> list[str]:
    """Everything: rig_timing.json, nerfstudio/ and 3dgut/ for the active variant; without a trajectory, a
    constant-velocity template drive (speed_mps, yaw_rate_dps, duration_s at rate_hz)."""
    os.makedirs(out_dir, exist_ok=True)
    traj = trajectory if trajectory is not None else constant_velocity_trajectory(speed_mps, yaw_rate_dps,
                                                                                  duration_s, rate_hz)
    timing = rig_timing(project)
    pt = os.path.join(out_dir, 'rig_timing.json')
    with open(pt, 'w', encoding='utf-8') as fh:
        json.dump(timing, fh, indent=1)
    return [os.path.abspath(pt)] + export_nerfstudio(project, out_dir, traj, timing) + \
        export_3dgut(project, out_dir, traj, timing)
