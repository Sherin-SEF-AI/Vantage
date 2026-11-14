"""The user's own vehicle in CARLA: the project's body and rig composited into CARLA's camera images, lidar sweeps and
ground truth, live and on recorded drives, so the checks that compare a camera with the plan's body (the as-built
check, rig health's body-edge drift indicator) can be run on simulated drives.

Why. CARLA renders its own car (the nearest blueprint by size: an Audi A2 for the sample's Tata Tigor), not the
project's. Its hood, mirrors and roof are where the project's are not, so the as-built check (core.asbuilt) and the
drift indicator (core.rig_health.alignment) measure the two bodies' difference instead of the camera, and a front
camera sees too little of CARLA's car to be checked at all.

What is rendered (the ego layer). Every pixel of a camera's output image is unprojected through the camera's own
lens model (the rig's lens block: the same model core.lens.resolve gives, with the rig's image scale; numpy only, so
it runs in CARLA's Python too) and its ray, from the camera's mount (the planned one, or the actual one of a turned or
moved camera: R_actual = R_plan · R(deg), the bridge's convention), is cast exactly against the triangles of the
project's body, its rig parts (core.rig) and every other sensor's and camera's housing (core.housings' boxes and
cylinders, triangulated; the camera's own housing left out): Möller–Trumbore on every ray–triangle pair the ray's
cell of an angular grid about the camera (6 cube faces × GRID² cells; a triangle in every cell its projected bounding
box touches, clipped at the camera's plane) can meet, so each pixel's depth is the exact range along its own ray, not
a texel's. Surface closer than NEAR_M is the lens protruding from its module (as core.synth.NEAR_CLIP) and is not
seen, nor is the body behind it on that ray (the rule of core.synth and the occlusion cube). Per pixel: range (m),
triangle, smoothed normal (corner normals averaged across edges flatter than CREASE_DEG,
as the shader renderer does, interpolated barycentrically), material.

Materials and shading. Each face's material class comes from core.materials (painted steel, fascia, trim, glass,
tyre, aluminium rig parts …; lamps by part name, as the 3D view colours them), with a typical visible albedo per class
(MATERIALS: typical values, not measured on any car) and the paint's colour from the settings. A pixel is lit by
CARLA's sun (its weather's sun altitude and azimuth; the presets' values read from CARLA 0.9.16's
LibCarla/source/carla/rpc/WeatherParameters.cpp), turned into the vehicle frame by the ego's heading at the frame's
time, with the body's own shadow (a parallel ray from the pixel's surface point towards the sun, exact, cached per
HEADING_STEP of heading), a sky hemisphere and the ground's bounce; glass and paint add the sky's Fresnel reflection
(Schlick). The sun's share of the light falls with cloudiness (CARLA's cloudiness 0–100). Exposure: the light is
scaled so the scene around the body, taken as middle grey (18 %, the reflectance a reflected-light meter assumes,
ISO 2720), comes out at the level CARLA's image shows there: one gain per camera (CARLA's exposure is manual), from
the median over the frames; then a luminance shoulder above 0.6 (the shader renderer's), so a bright body keeps its
shading. It is a plausible picture of the body under that light, not a physical render: no
reflections of the scene, no headlights, no dirt.

Compositing a frame. The body is painted wherever it is nearer than what CARLA saw there: CARLA's range image
(ground truth, recorded drives) or, without one (live), wherever the body is (nothing in CARLA's world is nearer to a
camera than the vehicle it rides on, except CARLA's own car, which is removed). CARLA's own car (its ego pixels:
semantic class 29 on a recorded drive; the instance-segmentation body mask live) where the project's body does not
cover it is a hole:
* recorded drives: filled from the neighbouring frames of the same camera (NEIGHBOURS each side): every pixel of
  theirs that is not CARLA's car, not a moving object (vehicles, walkers, riders: their motion is not the ego's) and
  has a range is moved into the world by the drive's ego poses and the camera's actual mount and projected into the
  frame (nearest surface wins; a static world assumed); gaps of a pixel or two closed by a 3 × 3 nearest fill; what
  is left is inpainted (OpenCV's Telea, from the hole's edge) or marked invalid (settings.carla_car = 'invalid':
  black, and left out of the valid mask);
* live: inpainted (Telea, radius 3 px, at the render size) or marked invalid: a camera rides on the car, so its
  neighbouring frames do not see behind CARLA's car either.
The lidar (recorded): every point whose ray from the lidar meets the project's body before it is blocked: removed
when the drive drops the lidar's own hits (output.drop_self_hits, the default), else moved onto the body (labelled
vehicle.ego in lidarseg); points CARLA's own car returned (nuScenes-lidarseg vehicle.ego) are removed (the scene
behind them is not known) and counted. The ground truth: semantic class ego (29) on the body's pixels, instance
EGO_INSTANCE there (the ego as an instance of its own), the body's range in the range image; filled holes take the
moved pixels' class and range (instance 0), unfilled ones class 0 (unlabelled) and range 0; every object's pixel count,
tight box and occlusion are recounted (objects the body now hides completely are dropped) and the COCO files rewritten.

Live (vantage.sim.live). The layers of every camera at the planned mount, the triangles, normals and materials are
written beside the live rig (sim_ego.npz, LiveEgo). The live server composites each frame at its render size before
the noise and the resize, at the camera's current mount: a mount fault casts the layer again (in a pool thread, cached
per mount); CARLA's own car at a turned mount is its planned body mask turned with the camera (exact for a rotation
about the optical centre; a translation fault's parallax on CARLA's car is not modelled). /<TAG>.body.png then serves
the project's body at the planned mount, so the drift indicator measures the camera against the user's body;
/<TAG>.carla_body.png keeps CARLA's car. A setting (analysis.ego_composite.live, the live server's --no-ego, POST
/ego) turns it off.

Frames: vehicle (ISO 8855, rear axle on the ground), metres and degrees; a camera's optical frame is OpenCV's.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import threading
import time
from dataclasses import asdict, dataclass, field

import numpy as np

from .mathutil import rot_zyx
# the layers and the live compositor live beside the live CARLA server, which runs them in CARLA's own Python
from ..sim.ego_layers import (  # noqa: F401  (re-exported: this module's names as before)
    _EPS_Z, _FACES, _mount_key, _pairs_test, _png, _ray_cells, _tri_cells, albedo_table, body_rgb8, CARLA_PRESETS,
    Caster, composite, DEFAULT_AZIMUTH, ego_layer, EgoMesh, exposure_gain, GLOSSY, GRID, HEADING_STEP, hex_linear,
    inpaint, irradiance, Layer, layer_normals, LAYERS_SCHEMA, lens_project, lens_unproject, linear_to_srgb, LiveEgo,
    MAT, MATERIALS, MID_GREY, mount_position, mount_rotation, NEAR_M, PAIR_CHUNK, pixel_grid, scaled_lens,
    scene_median, shade, SHADOW_CELL_M, ShadowCache, srgb_to_linear, sun_angles, sun_world, theta_of_r, to_vehicle,
    tone, TONE_KNEE, W2O_of)

SCHEMA = 'vantage.ego_composite/1'
REPORT = 'ego_composite.json'                 # the composited drive's own report (beside run.json)
EGO_DIR = 'ego_composite'                     # per camera masks of a composited drive
EGO_CLASS = 29                                # core.groundtruth.EGO_CLASS
EGO_INSTANCE = 65535                          # the ego's instance id in a composited drive's instance images
NUSC_EGO = 31                                 # nuScenes-lidarseg 'vehicle.ego'
CREASE_DEG = 40.0
MOVING_TAGS = frozenset((12, 13, 14, 15, 16, 17, 18, 19))   # walkers, riders and vehicles (CARLA 0.9.16 tags)
FRAME = 'vehicle (ISO 8855, rear axle on the ground): x forward, y left, z up, metres'
CARLA_CAR_MODES = ('inpaint', 'invalid')


CLASS_MAT = {'steel': 'paint', 'fascia': 'paint', 'pc_abs': 'paint', 'aluminium': 'aluminium', 'chrome': 'chrome',
             'cfrp': 'carbon', 'trim': 'trim', 'radome': 'radome', 'lamp': 'lamp', 'windscreen': 'glass',
             'side_glass': 'glass', 'tyre': 'tyre'}
LIMITS = {'neighbours': (0, 20), 'heading_step_deg': (1.0, 45.0), 'jpeg_quality': (0, 100),
          'inpaint_px': (1.0, 15.0)}
INTS = ('neighbours', 'jpeg_quality')
BOOLS = ('live', 'shadows', 'depth_test', 'lidar', 'ground_truth', 'keep_rosbag')
ASSUMPTIONS = (
    'The body is the project\'s mesh (with its rig parts and every other sensor\'s and camera\'s housing as boxes and '
    'cylinders) cast exactly through each camera\'s lens model; its look is a plausible shading under CARLA\'s sun and '
    'sky with typical albedos (not a physical render: no scene reflections, no lights, no dirt). The body is mounted '
    'where CARLA\'s car is (the rear-axle centres on the ground coincide): it moves with CARLA\'s physics, and other '
    'traffic collides with CARLA\'s car, not with it. CARLA\'s own car where the project\'s body does not cover it is '
    'filled from neighbouring frames assuming a static world (moving objects are not used), then inpainted or marked '
    'invalid; live, it is inpainted or marked invalid. CARLA\'s sun azimuth convention is taken as its sky\'s '
    'directional light (yaw = azimuth, pitch = −altitude, Unreal\'s axes); CARLA does not document it. A translation '
    'fault moves the project\'s body exactly, CARLA\'s car (live) only by the turn.')


def _flag(v, default):
    if isinstance(v, (bool, int, np.bool_)):
        return bool(v)
    if isinstance(v, str):
        return v.strip().lower() in ('1', 'true', 'yes', 'on')
    return default


# =========================================================================== settings
@dataclass
class EgoSettings:
    live: bool = True                   # composite the live streams (vantage.sim.live) with the project's body
    paint: str = '#b8bcc2'              # sRGB colour of the paint (a typical silver)
    carla_car: str = 'inpaint'          # CARLA's own car where the project's body does not cover it: inpaint | invalid
    neighbours: int = 6                 # recorded drives: frames on each side moved into those holes
    inpaint_px: float = 3.0             # Telea's radius
    shadows: bool = True                # the body's own shadow
    heading_step_deg: float = HEADING_STEP
    depth_test: bool = True             # recorded drives: the body only where it is nearer than CARLA's range image
    lidar: bool = True                  # recorded drives: the lidar sweeps too
    ground_truth: bool = True           # recorded drives: the label images, objects and COCO files too
    keep_rosbag: bool = False           # hard-link the drive's ROS 2 bag (its images are CARLA's, not composited)
    jpeg_quality: int = 0               # 0: the drive's own
    drive: str = ''                     # the recorded drive last composited
    out_dir: str = ''                   # its composited copy ('' = next to it, <drive>-ego)

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d) -> 'EgoSettings':
        s = EgoSettings()
        d = d if isinstance(d, dict) else {}
        for k in BOOLS:
            if k in d:
                setattr(s, k, _flag(d[k], getattr(s, k)))
        for k, (lo, hi) in LIMITS.items():
            if k in d:
                try:
                    v = float(d[k])
                except (TypeError, ValueError):
                    continue
                if math.isfinite(v):
                    v = min(max(v, lo), hi)
                    setattr(s, k, int(round(v)) if k in INTS else v)
        if d.get('carla_car') in CARLA_CAR_MODES:
            s.carla_car = d['carla_car']
        p = d.get('paint')
        if isinstance(p, str) and _hex_ok(p):
            s.paint = p.lower()
        for k in ('drive', 'out_dir'):
            if isinstance(d.get(k), str):
                setattr(s, k, d[k][:4096])
        return s


def _hex_ok(s: str) -> bool:
    s = s.strip()
    return len(s) == 7 and s[0] == '#' and all(c in '0123456789abcdefABCDEF' for c in s[1:])


def settings_of(project) -> EgoSettings:
    return EgoSettings.from_dict(getattr(project.analysis, 'ego_composite', None))


# =========================================================================== lenses (numpy only: CARLA's Python too)
def _r_of_theta(t, projection, kb):
    from .groundtruth import r_of_theta
    return r_of_theta(t, 1.0, projection, kb)


def corner_normals(tri: np.ndarray, face_n: np.ndarray, crease_deg: float = CREASE_DEG) -> np.ndarray:
    """Per-corner normals (M, 3, 3): the area-weighted average of the faces sharing a vertex position, except across
    edges sharper than the crease angle (as ui.render_gl.vertex_normals)."""
    M = len(tri)
    if M == 0:
        return np.zeros((0, 3, 3), np.float32)
    P = tri.reshape(-1, 3)
    fn = face_n / np.maximum(np.linalg.norm(face_n, axis=1, keepdims=True), 1e-12)
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    _, inv = np.unique(np.round(P / 1e-5).astype(np.int64), axis=0, return_inverse=True)
    inv = inv.ravel()
    acc = np.zeros((inv.max() + 1, 3))
    np.add.at(acc, inv, np.repeat(fn * area[:, None], 3, axis=0))
    vn = acc[inv]
    vn /= np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)
    cf = np.repeat(fn, 3, axis=0)
    keep = np.sum(vn * cf, axis=1) >= math.cos(math.radians(crease_deg))
    return np.where(keep[:, None], vn, cf).reshape(M, 3, 3).astype(np.float32)


def _box_tris(c, half, R):
    s = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float)
    V = (s * half) @ np.asarray(R, float).T + c
    F = [[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4],
         [1, 5, 7], [1, 7, 3]]
    return V[np.array(F)]


def _cyl_tris(c, half, R, n=24):
    a = np.linspace(0, 2 * math.pi, n, endpoint=False)
    r, hz = float(half[0]), float(half[2])
    ring = np.c_[r * np.cos(a), r * np.sin(a)]
    top = np.c_[ring, np.full(n, hz)]
    bot = np.c_[ring, np.full(n, -hz)]
    T = []
    for i in range(n):
        j = (i + 1) % n
        T += [[bot[i], bot[j], top[j]], [bot[i], top[j], top[i]], [[0, 0, hz], top[i], top[j]],
              [[0, 0, -hz], bot[j], bot[i]]]
    T = np.array(T, float)
    return T @ np.asarray(R, float).T + c


def _part_material(part, cls_id: str) -> str:
    import re
    name = (getattr(part, 'name', '') or '').lower()
    if re.search(r'\brims?\b', name) or 'alloy' in name:
        return 'rim'
    if 'tail lamp' in name or 'brake' in name:
        return 'tail_lamp'
    if 'head lamp' in name or 'headlight' in name:
        return 'lamp'
    if 'black trim' in name:
        return 'trim'
    return CLASS_MAT.get(cls_id, 'paint')


def ego_mesh(project, variant=None, settings: EgoSettings | None = None, engine=None) -> EgoMesh | None:
    """The project's body, rig parts and housings as triangles with materials (None without a vehicle mesh)."""
    from . import materials as MT
    from .mesh import ROLE_IGNORED
    st = settings or settings_of(project)
    var = variant or project.variant
    vg = engine.vehicle_geom(project, var) if engine is not None else None
    mesh, parts = MT._geometry(project, vg)
    if mesh is None:
        return None
    fm = MT.face_materials(mesh, parts, MT.settings_of(project))
    fp = np.clip(mesh.face_part, 0, max(len(parts) - 1, 0))
    keep = np.array([getattr(p, 'role', '') != ROLE_IGNORED for p in parts] or [True], bool)[fp]
    mats = np.zeros(mesh.n_tris, np.uint8)
    for i, p in enumerate(parts):
        m = fp == i
        if not m.any():
            continue
        cls = fm.cls[m]
        for c in np.unique(cls):
            cid = MT.CLASS_IDS[int(c)] if c >= 0 else 'steel'
            mats[np.flatnonzero(m)[cls == c]] = MAT[_part_material(p, cid)]
    tri = np.asarray(mesh.tri, np.float64)[keep]
    nrm = np.asarray(fm.normals if hasattr(fm, 'normals') else fm[2], np.float64)[keep]
    mats = mats[keep]
    owner = np.full(len(tri), -1, np.int16)
    anchors = []
    notes = []
    try:
        from . import housings as HS
        sols = HS.solids(project, var)
    except Exception as e:                       # noqa: BLE001 (housings are optional detail)
        sols = []
        notes.append(f'No housings ({type(e).__name__}).')
    extra_t, extra_m, extra_o = [], [], []
    for k, h in enumerate(sols):
        T = _cyl_tris(h.centre, h.half, h.R) if h.round else _box_tris(h.centre, h.half, h.R)
        extra_t.append(T)
        extra_m.append(np.full(len(T), MAT['housing'], np.uint8))
        extra_o.append(np.full(len(T), k, np.int16))
        anchors.append(np.asarray(h.anchor, float))
    if extra_t:
        T = np.concatenate(extra_t)
        n = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
        tri = np.concatenate([tri, T])
        nrm = np.concatenate([nrm, n])
        mats = np.concatenate([mats] + extra_m)
        owner = np.concatenate([owner] + extra_o)
    return EgoMesh(tri, nrm, corner_normals(tri, nrm), mats, owner,
                   np.array(anchors, float).reshape(-1, 3), st.paint, notes)


def heading_of_quat(q) -> float:
    """Heading (rad) of a nuScenes rotation quaternion [w, x, y, z]."""
    w, x, y, z = (float(a) for a in q)
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


# =========================================================================== recorded drives
def _read_json(path):
    with open(path, encoding='utf-8') as fh:
        return json.load(fh)


def _write_json(path, doc, indent=None):
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as fh:
        json.dump(doc, fh, indent=indent)
    os.replace(tmp, path)


def drive_rig(drive_dir: str) -> tuple[dict, dict]:
    """(sim_rig.json, run.json) of a recorded drive."""
    rig = None
    for rel in (('rig', 'sim_rig.json'), ('sim_rig.json',)):
        p = os.path.join(drive_dir, *rel)
        if os.path.isfile(p):
            rig = _read_json(p)
            break
    if rig is None:
        raise ValueError(f'{drive_dir}: no rig/sim_rig.json beside the drive (the CARLA page and --carla-run write it '
                         'there; for a bridge run of your own, copy the sim_rig.json and sim_remap.npz it used into '
                         'rig/).')
    run = _read_json(os.path.join(drive_dir, 'run.json'))
    return rig, run


def actual_mounts(rig: dict, run: dict) -> dict:
    """{camera tag: (pose_vehicle, deg or None, m or None)}: each camera's actual mount as the drive recorded it."""
    out = {c['tag']: (c['pose_vehicle'], None, None) for c in rig.get('cameras', [])}
    for f in run.get('faults_injected') or []:
        t = f.get('tag')
        if t not in out or f.get('sensor_kind', 'camera') != 'camera':
            continue
        pose, deg, m = out[t]
        if f.get('kind') == 'rotation':
            deg = [float(x) for x in f.get('deg') or f.get('pose_delta_deg') or (0, 0, 0)]
        elif f.get('kind') == 'translation':
            m = [float(x) for x in f.get('m') or f.get('position_delta_m') or (0, 0, 0)]
        out[t] = (pose, deg, m)
    return out


def _nus_tables(drive_dir):
    root = os.path.join(drive_dir, 'nuscenes')
    vers = sorted(d for d in os.listdir(root) if d.startswith('v1.0') and os.path.isdir(os.path.join(root, d))) \
        if os.path.isdir(root) else []
    if not vers:
        return None
    t = {}
    for name in ('sample_data', 'ego_pose', 'calibrated_sensor', 'sensor'):
        p = os.path.join(root, vers[0], name + '.json')
        t[name] = _read_json(p) if os.path.isfile(p) else []
    t['version'] = vers[0]
    return t


def ego_poses(drive_dir) -> dict:
    """{nuScenes file name (relative to nuscenes/): (R 3×3, t (3,)) vehicle -> global} of every sample_data."""
    t = _nus_tables(drive_dir)
    if t is None:
        return {}
    eps = {e['token']: e for e in t['ego_pose']}
    out = {}
    for sd in t['sample_data']:
        e = eps.get(sd.get('ego_pose_token'))
        if e is None:
            continue
        w, x, y, z = e['rotation']
        R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                      [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                      [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
        out[sd['filename']] = (R, np.asarray(e['translation'], float))
    return out


def _link_tree(src, dst, skip_dirs=()):
    """Hard-link (copy across file systems) every file of src into dst, but the folders in skip_dirs."""
    for root, dirs, files in os.walk(src):
        rel = os.path.relpath(root, src)
        if any(rel == s or rel.startswith(s + os.sep) for s in skip_dirs):
            dirs[:] = []
            continue
        os.makedirs(os.path.join(dst, rel), exist_ok=True)
        for f in files:
            s, d = os.path.join(root, f), os.path.join(dst, rel, f)
            if os.path.lexists(d):
                continue
            try:
                os.link(s, d)
            except OSError:
                shutil.copy2(s, d)


def _replace(path, data: bytes):
    """Write a file of the copy anew (never through a hard link into the original drive)."""
    if os.path.lexists(path):
        os.unlink(path)
    with open(path, 'wb') as fh:
        fh.write(data)


@dataclass
class _Frame:
    rel: str                            # the image's path relative to the drive
    nus: str                            # relative to nuscenes/
    gt: dict | None
    pose: tuple | None                  # (R, t) vehicle -> global


def _cam_frames(drive_dir, gt_doc, tag, poses) -> list:
    cam = gt_doc['cameras'][tag]
    out = []
    for fr in cam['frames']:
        rel = fr['image']
        nus = rel[len('nuscenes/'):] if rel.startswith('nuscenes/') else rel
        out.append(_Frame(rel, nus, fr, poses.get(nus)))
    return out


def _load_frame(drive_dir, gt_doc, fr: _Frame):
    """(BGR image, semantic, instance, range m) of a frame."""
    import cv2
    from . import groundtruth as GT
    img = cv2.imread(os.path.join(drive_dir, fr.rel), cv2.IMREAD_COLOR)
    sem = GT.load_semantic(gt_doc, fr.gt)
    inst = GT.load_instance(gt_doc, fr.gt)
    rng = GT.load_range(gt_doc, fr.gt)
    return img, sem, inst, rng


class _Reprojector:
    """Moves neighbouring frames' static pixels into a frame's holes (see the module doc)."""

    def __init__(self, lens: dict, w: int, h: int, R_cam: np.ndarray, pos: np.ndarray):
        self.lens = scaled_lens(lens, w, h)
        u, v = pixel_grid(w, h)
        d, _, ok = lens_unproject(self.lens, u, v)
        self.d_opt = d
        self.ok = ok
        self.w, self.h = w, h
        self.W2O = W2O_of(R_cam)
        self.pos = pos

    def world_points(self, rng, sem, carla_car, pose):
        """(points in the global frame (n, 3), their flat pixel indices) of a source frame's usable pixels."""
        r = rng.ravel()
        use = self.ok & (r > 0) & ~carla_car.ravel() & ~np.isin(sem.ravel(), list(MOVING_TAGS))
        idx = np.flatnonzero(use)
        Pv = self.pos + (self.d_opt[idx] * r[idx, None]) @ self.W2O
        R, t = pose
        return Pv @ R.T + t, idx

    def fill(self, holes, target_pose, sources):
        """sources: [(global points, pixel indices, image BGR, semantic, range)] -> (colours (h, w, 3), sem, range,
        filled mask) for the hole pixels."""
        h, w = self.h, self.w
        zbuf = np.full(h * w, np.inf)
        col = np.zeros((h * w, 3), np.uint8)
        sem_o = np.zeros(h * w, np.uint8)
        hole = holes.ravel()
        R, t = target_pose
        for Pg, idx, img, sem, rng in sources:
            Pv = (Pg - t) @ R
            Po = (Pv - self.pos) @ self.W2O.T
            u, v, _, ok = lens_project(self.lens, Po)
            iu = np.floor(u).astype(np.int64)
            iv = np.floor(v).astype(np.int64)
            ok &= (iu >= 0) & (iu < w) & (iv >= 0) & (iv < h)
            tgt = iv * w + iu
            ok[ok] &= hole[tgt[ok]]
            if not ok.any():
                continue
            tgt, src, z = tgt[ok], idx[ok], np.linalg.norm(Po[ok], axis=1)
            order = np.argsort(-z)                     # far first: the nearest written last wins
            tgt, src, z = tgt[order], src[order], z[order]
            better = z < zbuf[tgt]
            tgt, src, z = tgt[better], src[better], z[better]
            zbuf[tgt] = z
            col[tgt] = img.reshape(-1, 3)[src]
            sem_o[tgt] = sem.ravel()[src]
        filled = np.isfinite(zbuf) & hole
        # close the splat's pinholes: a hole pixel with filled neighbours takes the nearest one's
        if filled.any():
            import cv2
            f2 = filled.reshape(h, w).astype(np.uint8)
            near = cv2.dilate(f2, np.ones((3, 3), np.uint8))
            gap = (near > 0) & ~filled.reshape(h, w) & holes
            if gap.any():
                zz = np.where(filled, zbuf, 1e9).reshape(h, w).astype(np.float32)
                mins = -cv2.dilate(-zz, np.ones((3, 3), np.uint8))
                ys, xs = np.nonzero(gap)
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        yy = np.clip(ys + dy, 0, h - 1)
                        xx = np.clip(xs + dx, 0, w - 1)
                        take = (zz[yy, xx] == mins[ys, xs]) & (zz[yy, xx] < 1e9) & ~np.isfinite(zbuf[ys * w + xs])
                        k = ys[take] * w + xs[take]
                        s = yy[take] * w + xx[take]
                        zbuf[k] = zbuf[s]
                        col[k] = col[s]
                        sem_o[k] = sem_o[s]
                filled = np.isfinite(zbuf) & hole
        return col.reshape(h, w, 3), sem_o.reshape(h, w), np.where(filled, zbuf, 0.0).reshape(h, w), \
            filled.reshape(h, w)


def _jpeg(img, q) -> bytes:
    import cv2
    ok, buf = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), int(q)])
    if not ok:
        raise RuntimeError('JPEG encoding failed')
    return buf.tobytes()


def _objects_after(objs: list, inst: np.ndarray) -> list:
    """The frame's objects recounted on the composited instance image: pixels, tight box; occlusion grown by the
    share of pixels the body took; objects left with no pixel dropped."""
    out = []
    ids = inst.ravel()
    cnt = np.bincount(ids, minlength=1)
    for o in objs:
        oid = int(o['id'])
        n = int(cnt[oid]) if oid < len(cnt) else 0
        if n == 0:
            continue
        o = dict(o)
        if n != int(o.get('pixels', n)):
            ys, xs = np.nonzero(inst == oid)
            o['bbox'] = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
            before = max(int(o.get('pixels', n)), 1)
            occ = float(o.get('occluded') or 0.0)
            vis = (1 - occ) * n / before
            o['occluded'] = round(1 - vis, 4)
            o['pixels'] = n
        out.append(o)
    return out


def composite_drive(project, drive_dir: str, out_dir: str | None = None, settings: EgoSettings | None = None,
                    variant=None, cameras=None, progress=None, cancel=None, engine=None) -> dict:
    """Composite the project's body into a recorded CARLA drive with ground truth (semantic, instance and range
    images): a copy in out_dir (default <drive>-ego) with the camera images, label images, objects, COCO files and
    lidar sweeps rewritten and everything else hard-linked. Returns the report (also written as ego_composite.json)."""
    from . import groundtruth as GT
    st = settings or settings_of(project)
    prog = progress or (lambda f, text='': None)
    drive_dir = os.path.abspath(drive_dir)
    out_dir = os.path.abspath(out_dir or drive_dir.rstrip(os.sep) + '-ego')
    if out_dir == drive_dir:
        raise ValueError('The composited copy must go to another folder.')
    rig, run = drive_rig(drive_dir)
    gt_doc = GT.read(drive_dir)
    if gt_doc is None:
        raise ValueError(f'{drive_dir}: no ground truth (ground_truth/index.json): CARLA\'s own car cannot be told '
                         'apart. Record the drive with ground truth (semantic, instance, range).')
    t_start = time.perf_counter()
    prog(0.0, 'Reading the drive')
    mesh = ego_mesh(project, variant, st, engine)
    if mesh is None:
        raise ValueError('The project has no vehicle mesh.')
    caster = Caster(mesh.tri)
    poses = ego_poses(drive_dir)
    mounts = actual_mounts(rig, run)
    alt, az, cloud = sun_angles(run.get('weather') or rig.get('weather'))
    sunw = sun_world(alt, az)
    q = int(st.jpeg_quality or (rig.get('output') or {}).get('jpeg_quality', 90))
    skip = ['rosbag'] if not st.keep_rosbag else []
    if os.path.isdir(out_dir) and os.listdir(out_dir):
        if not os.path.isfile(os.path.join(out_dir, REPORT)) and not os.path.isfile(os.path.join(out_dir,
                                                                                                   'run.json')):
            raise ValueError(f'{out_dir} exists and is not a composited drive: choose an empty folder.')
        shutil.rmtree(out_dir)
    _link_tree(drive_dir, out_dir, skip)
    tags = [t for t in gt_doc['cameras'] if t in mounts and (not cameras or t in cameras)]
    cams_rig = {c['tag']: c for c in rig['cameras']}
    rep = {'schema': SCHEMA, 'drive': os.path.basename(drive_dir), 'project': project.name,
           'variant': (variant or project.variant).name, 'settings': st.to_dict(), 'frame': FRAME,
           'sun': {'altitude_deg': alt, 'azimuth_deg': az, 'cloudiness': cloud,
                   'vehicle_frame_note': 'turned by each frame\'s ego heading'},
           'mesh': {'triangles': int(len(mesh.tri)), 'housing_triangles': int((mesh.owner >= 0).sum())},
           'cameras': {}, 'lidars': {}, 'notes': list(mesh.notes), 'assumptions': ASSUMPTIONS}
    gt_out = json.loads(json.dumps(gt_doc))
    gt_out.pop('_dir', None)
    os.makedirs(os.path.join(out_dir, EGO_DIR), exist_ok=True)
    total = sum(len(gt_doc['cameras'][t]['frames']) for t in tags) or 1
    done = 0
    shadows = ShadowCache(mesh, caster, st.heading_step_deg)
    coco_updates = {}
    for tag in tags:
        if cancel is not None and cancel():
            raise InterruptedError('cancelled')
        cam = gt_doc['cameras'][tag]
        w, h = int(cam['width']), int(cam['height'])
        pose, deg, m = mounts[tag]
        lens = cams_rig[tag]['lens']
        layer = ego_layer(mesh, caster, lens, pose, w, h, deg, m)
        frames = _cam_frames(drive_dir, gt_doc, tag, poses)
        rp = _Reprojector(lens, w, h, layer.R, layer.pos)
        cache = {}

        def get(i):
            if i not in cache:
                img, sem, inst, rng = _load_frame(drive_dir, gt_doc, frames[i])
                car = sem == EGO_CLASS
                cache[i] = (img, sem, inst, rng, car)
                for k in [k for k in cache if abs(k - i) > st.neighbours + 1]:
                    cache.pop(k)
            return cache[i]
        # one exposure gain for the camera: the median scene level over up to 8 frames
        meds = []
        for i in np.linspace(0, len(frames) - 1, min(8, len(frames))).astype(int):
            img, sem, inst, rng, car = get(int(i))
            md = scene_median(img, car | layer.mask)
            if md is not None:
                meds.append(md)
        cache.clear()
        gain = exposure_gain(float(np.median(meds)) if meds else 0.2, alt, cloud)
        stats = {'frames': 0, 'body_px': 0, 'hole_px': 0, 'filled_px': 0, 'inpainted_px': 0, 'invalid_px': 0,
                 'carla_car_px': 0, 'objects_dropped': 0, 'ms': []}
        body_share = float(layer.mask.sum()) / max(int(layer.inside.sum()), 1)
        for i, fr in enumerate(frames):
            if cancel is not None and cancel():
                raise InterruptedError('cancelled')
            t0 = time.perf_counter()
            img, sem, inst, rng, car = get(i)
            heading = heading_of_quat_R(fr.pose[0]) if fr.pose is not None else 0.0
            sv = to_vehicle(sunw, heading)
            shd = shadows.get(layer, (tag, deg and tuple(deg), m and tuple(m)), sunw, heading) if st.shadows and \
                alt > 0 else None
            rad = shade(mesh, layer, sv if alt > 0 else None, alt, cloud, shd)
            rgb = body_rgb8(rad, gain)
            out, body, holes = composite(img, layer, rgb[:, ::-1], car, rng if st.depth_test else None,
                                         st.depth_test)
            sem2, inst2, rng2 = sem.copy(), inst.copy(), rng.copy()
            sem2[body] = EGO_CLASS
            inst2[body] = EGO_INSTANCE
            rng2[body] = layer.range[body]
            filled = np.zeros_like(holes)
            if holes.any() and fr.pose is not None and st.neighbours > 0:
                srcs = []
                for j in range(max(0, i - st.neighbours), min(len(frames), i + st.neighbours + 1)):
                    if j == i or frames[j].pose is None:
                        continue
                    jimg, jsem, _, jrng, jcar = get(j)
                    Pg, idx = rp.world_points(jrng, jsem, jcar | layer.mask, frames[j].pose)
                    srcs.append((Pg, idx, jimg, jsem, jrng))
                col, fsem, frng, filled = rp.fill(holes, fr.pose, srcs)
                out[filled] = col[filled]
                sem2[filled] = fsem[filled]
                inst2[filled] = 0
                rng2[filled] = frng[filled]
            rest = holes & ~filled
            if rest.any():
                if st.carla_car == 'inpaint':
                    out = inpaint(out, rest, st.inpaint_px)
                else:
                    out[rest] = 0
                sem2[rest] = 0
                inst2[rest] = 0
                rng2[rest] = 0
            car_left = car & ~body & ~holes          # CARLA's car outside the lens field (not seen)
            sem2[car_left] = 0
            _replace(os.path.join(out_dir, fr.rel), _jpeg(out, q))
            gfr = fr.gt
            if st.ground_truth:
                _replace(os.path.join(out_dir, gfr['semantic']), _png(sem2))
                _replace(os.path.join(out_dir, gfr['instance']), _png(inst2.astype(np.uint16)))
                _replace(os.path.join(out_dir, gfr['range']), _png(GT.range_png(rng2)))
            new_objs = _objects_after(gfr.get('objects') or [], inst2)
            stats['objects_dropped'] += len(gfr.get('objects') or []) - len(new_objs)
            gfi = next(f for f in gt_out['cameras'][tag]['frames'] if f['image'] == fr.rel)
            gfi['objects'] = new_objs
            coco_updates.setdefault(tag, {})[fr.rel] = inst2 if st.ground_truth else None
            # samples/ keyframes are the same image as their sweep in some writers: rewrite a twin of the same name
            twin = fr.rel.replace('/sweeps/', '/samples/')
            if twin != fr.rel and os.path.isfile(os.path.join(drive_dir, twin)):
                _replace(os.path.join(out_dir, twin), _jpeg(out, q))
            if i == 0:
                valid = layer.inside & ~(rest if st.carla_car == 'invalid' else np.zeros_like(rest))
                _replace(os.path.join(out_dir, EGO_DIR, f'CAM_{tag}.body.png'),
                         _png(np.where(layer.mask, 255, np.where(layer.inside, 128, 0)).astype(np.uint8)))
                _replace(os.path.join(out_dir, EGO_DIR, f'CAM_{tag}.range.png'),
                         _png(GT.range_png(np.where(layer.mask, layer.range, 0.0))))
            stats['frames'] += 1
            stats['body_px'] += int(body.sum())
            stats['hole_px'] += int(holes.sum())
            stats['filled_px'] += int(filled.sum())
            stats['inpainted_px' if st.carla_car == 'inpaint' else 'invalid_px'] += int(rest.sum())
            stats['carla_car_px'] += int(car.sum())
            stats['ms'].append((time.perf_counter() - t0) * 1000.0)
            done += 1
            prog(0.02 + 0.85 * done / total, f'{tag}: frame {i + 1} of {len(frames)}')
        ms = np.array(stats.pop('ms') or [0.0])
        n = max(stats['frames'], 1)
        rep['cameras'][tag] = dict(stats, body_share=round(body_share, 4), exposure_gain=round(gain, 4),
                                   layer_ms=round(layer.ms, 1), frame_ms_median=round(float(np.median(ms)), 1),
                                   mount={'pose': pose, 'deg': deg, 'm': m},
                                   holes_filled_share=round(stats['filled_px'] / max(stats['hole_px'], 1), 4),
                                   body_px_per_frame=round(stats['body_px'] / n),
                                   carla_car_px_per_frame=round(stats['carla_car_px'] / n))
    # COCO files of the cameras
    if st.ground_truth:
        _rewrite_coco(drive_dir, out_dir, coco_updates)
    # lidar sweeps
    if st.lidar:
        prog(0.9, 'Lidar sweeps')
        rep['lidars'] = _composite_lidars(drive_dir, out_dir, rig, mesh, caster, run, cancel)
    gt_out.setdefault('notes', []).append(
        'Composited with the project\'s own body (core.ego_composite): the ego\'s pixels are the project\'s body '
        f'(class ego, instance {EGO_INSTANCE}, its range); CARLA\'s own car where the body does not cover it is '
        'filled from neighbouring frames, else ' + ('inpainted' if st.carla_car == 'inpaint' else 'unlabelled') + '.')
    gt_out.setdefault('conventions', {})['ego_instance'] = EGO_INSTANCE
    _replace(os.path.join(out_dir, GT.INDEX), json.dumps(gt_out, indent=1).encode('utf-8'))
    run2 = dict(run)
    run2['ego_composite'] = {'schema': SCHEMA, 'report': REPORT, 'body': 'the project\'s (core.ego_composite)',
                             'source_drive': os.path.basename(drive_dir)}
    run2['out_dir'] = out_dir
    _replace(os.path.join(out_dir, 'run.json'), json.dumps(run2, indent=1).encode('utf-8'))
    rep['seconds'] = round(time.perf_counter() - t_start, 2)
    rep['out_dir'] = out_dir
    _replace(os.path.join(out_dir, REPORT), json.dumps(rep, indent=1, default=_jsonable).encode('utf-8'))
    readme = os.path.join(out_dir, 'README.md')
    if os.path.isfile(readme):
        with open(readme, encoding='utf-8') as fh:
            txt = fh.read()
        txt += ('\n\n## Composited with the project\'s body\n\nThis copy has the project\'s own body and rig '
                'composited into the camera images, label images and lidar sweeps (Vantage core.ego_composite; see '
                f'{REPORT}). The ROS 2 bag is ' + ('the original (CARLA\'s car).' if st.keep_rosbag else
                                                   'left out (its images are not composited).') + '\n')
        _replace(readme, txt.encode('utf-8'))
    prog(1.0, 'Done')
    return rep


def heading_of_quat_R(R) -> float:
    return math.atan2(float(R[1, 0]), float(R[0, 0]))


def _jsonable(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, tuple):
        return list(o)
    return str(o)


def _rewrite_coco(drive_dir, out_dir, updates: dict):
    from . import groundtruth as GT
    for tag, frames in updates.items():
        p = os.path.join(drive_dir, 'ground_truth', 'coco', f'CAM_{tag}.json')
        if not os.path.isfile(p):
            continue
        doc = _read_json(p)
        by_img = {im['file_name']: im['id'] for im in doc.get('images', [])}
        inst_of = {by_img[k]: v for k, v in frames.items() if k in by_img and v is not None}
        anns = []
        for a in doc.get('annotations', []):
            inst = inst_of.get(a['image_id'])
            if inst is None:
                anns.append(a)
                continue
            m = inst == int(a.get('object_id', -1))
            if not m.any():
                continue
            ys, xs = np.nonzero(m)
            a = dict(a, bbox=[int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)],
                     area=int(m.sum()), segmentation=GT.rle_encode(m))
            anns.append(a)
        doc['annotations'] = anns
        doc.setdefault('info', {})['ego_composite'] = 'the project\'s body composited in (core.ego_composite)'
        _replace(os.path.join(out_dir, 'ground_truth', 'coco', f'CAM_{tag}.json'), json.dumps(doc).encode('utf-8'))


def _quat_R(q):
    w, x, y, z = (float(a) for a in q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def composite_points(P_sensor, R_sensor, pos, caster: Caster, drop: bool, skip=None):
    """A lidar sweep's points (sensor frame) against the project's body: (keep mask, new points in the sensor frame,
    moved mask): a point whose ray meets the body first is removed (drop) or moved onto the body."""
    P = np.asarray(P_sensor, float)
    r = np.linalg.norm(P, axis=1)
    ok = r > 1e-6
    D = np.zeros_like(P)
    D[ok] = P[ok] / r[ok, None]
    Dv = D @ np.asarray(R_sensor, float).T
    t, _ = caster.cast(pos, Dv, 1e-3, skip=skip)
    blocked = ok & (t < r - 0.02)
    keep = ~blocked if drop else np.ones(len(P), bool)
    newP = P.copy()
    if not drop:
        newP[blocked] = D[blocked] * t[blocked, None]
    return keep, newP, (blocked if not drop else np.zeros(len(P), bool))


def _composite_lidars(drive_dir, out_dir, rig, mesh: EgoMesh, caster: Caster, run, cancel) -> dict:
    t = _nus_tables(drive_dir)
    if t is None:
        return {}
    sens = {s['token']: s for s in t['sensor']}
    cal = {c['token']: c for c in t['calibrated_sensor']}
    drop = bool((rig.get('output') or {}).get('drop_self_hits', True))
    root = os.path.join(drive_dir, 'nuscenes')
    ver = t['version']
    out = {}
    lidarseg = {}
    p = os.path.join(root, ver, 'lidarseg.json')
    if os.path.isfile(p):
        lidarseg = {r['sample_data_token']: r['filename'] for r in _read_json(p)}
    panoptic = {}
    p = os.path.join(root, ver, 'panoptic.json')
    if os.path.isfile(p):
        panoptic = {r['sample_data_token']: r['filename'] for r in _read_json(p)}
    for sd in t['sample_data']:
        c = cal.get(sd.get('calibrated_sensor_token'))
        s = sens.get(c['sensor_token']) if c else None
        if s is None or s.get('modality') != 'lidar' or not sd['filename'].endswith('.pcd.bin'):
            continue
        if cancel is not None and cancel():
            raise InterruptedError('cancelled')
        ch = s['channel']
        f = os.path.join(root, sd['filename'])
        if not os.path.isfile(f):
            continue
        a = np.fromfile(f, np.float32).reshape(-1, 5)
        R = _quat_R(c['rotation'])
        pos = np.asarray(c['translation'], float)
        skip = mesh.own(pos)
        keep, newP, moved = composite_points(a[:, :3], R, pos, caster, drop, skip)
        a2 = a.copy()
        a2[:, :3] = newP.astype(np.float32)
        lab_p = lidarseg.get(sd['token'])
        carla_ego = np.zeros(len(a), bool)
        lab = None
        if lab_p and os.path.isfile(os.path.join(root, lab_p)):
            lab = np.fromfile(os.path.join(root, lab_p), np.uint8)
            if len(lab) == len(a):
                carla_ego = lab == NUSC_EGO
                lab = lab.copy()
                lab[moved] = NUSC_EGO
            else:
                lab = None
        keep &= ~carla_ego | moved
        _replace(os.path.join(out_dir, 'nuscenes', sd['filename']), a2[keep].tobytes())
        if lab is not None:
            _replace(os.path.join(out_dir, 'nuscenes', lab_p), lab[keep].tobytes())
        pan_p = panoptic.get(sd['token'])
        if pan_p and os.path.isfile(os.path.join(root, pan_p)):
            with np.load(os.path.join(root, pan_p)) as z:
                pan = z['data']
            if len(pan) == len(a):
                pan = pan.copy()
                pan[moved] = NUSC_EGO * 1000
                import io
                buf = io.BytesIO()
                np.savez_compressed(buf, data=pan[keep])
                _replace(os.path.join(out_dir, 'nuscenes', pan_p), buf.getvalue())
        o = out.setdefault(ch, {'sweeps': 0, 'points_in': 0, 'points_out': 0, 'blocked_by_body': 0,
                                'moved_onto_body': 0, 'carla_car_removed': 0, 'drop_self_hits': drop})
        o['sweeps'] += 1
        o['points_in'] += int(len(a))
        o['points_out'] += int(keep.sum())
        o['blocked_by_body'] += int((~keep & ~carla_ego).sum() + moved.sum())
        o['moved_onto_body'] += int(moved.sum())
        o['carla_car_removed'] += int((carla_ego & ~moved).sum())
    return out


# =========================================================================== a turned camera without CARLA
def turn_drive(drive_dir: str, out_dir: str, tag: str, deg, progress=None) -> dict:
    """A copy of a recorded drive with camera `tag` turned by deg [yaw, pitch, roll] about its optical centre (the
    bridge's convention): its images and label images resampled exactly through the lens model (a rotation about the
    optical centre has no parallax), the rim the planned camera did not see black (class 0, range 0), run.json's
    faults_injected recording the turn. A stand-in for a CARLA drive recorded with the fault, for measuring the
    checks' answers on many turns; the CARLA-recorded ones are the reference."""
    import cv2
    from . import groundtruth as GT
    drive_dir, out_dir = os.path.abspath(drive_dir), os.path.abspath(out_dir)
    rig, run = drive_rig(drive_dir)
    gt_doc = GT.read(drive_dir)
    if gt_doc is None or tag not in gt_doc['cameras']:
        raise ValueError(f'{drive_dir}: no ground truth of camera {tag}.')
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    _link_tree(drive_dir, out_dir, ['rosbag'])
    cam_rig = next(c for c in rig['cameras'] if c['tag'] == tag)
    cam = gt_doc['cameras'][tag]
    w, h = int(cam['width']), int(cam['height'])
    lens = scaled_lens(cam_rig['lens'], w, h)
    pose = cam_rig['pose_vehicle']
    prev = actual_mounts(rig, run)[tag]
    R0 = mount_rotation(pose, prev[1])
    R1 = R0 @ rot_zyx(*(float(x) for x in deg))
    u, v = pixel_grid(w, h)
    d1, _, ok1 = lens_unproject(lens, u, v)
    d0 = d1 @ (W2O_of(R1) @ W2O_of(R0).T)         # the new camera's rays in the old camera's optical frame
    us, vs, _, ok0 = lens_project(lens, d0)
    ok = ok1 & ok0 & (us >= 0) & (us < w) & (vs >= 0) & (vs < h)
    mx = np.where(ok, us - 0.5, -1).reshape(h, w).astype(np.float32)
    my = np.where(ok, vs - 0.5, -1).reshape(h, w).astype(np.float32)
    q = int((rig.get('output') or {}).get('jpeg_quality', 90))
    for k, fr in enumerate(cam['frames']):
        img = cv2.imread(os.path.join(drive_dir, fr['image']), cv2.IMREAD_COLOR)
        img2 = cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        _replace(os.path.join(out_dir, fr['image']), _jpeg(img2, q))
        twin = fr['image'].replace('/sweeps/', '/samples/')
        if twin != fr['image'] and os.path.isfile(os.path.join(drive_dir, twin)):
            _replace(os.path.join(out_dir, twin), _jpeg(img2, q))
        for key, loader, enc in (('semantic', GT.load_semantic, lambda a: a.astype(np.uint8)),
                                 ('instance', GT.load_instance, lambda a: a.astype(np.uint16))):
            a = loader(gt_doc, fr)
            a2 = cv2.remap(a, mx, my, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            _replace(os.path.join(out_dir, fr[key]), _png(enc(a2)))
        r = GT.load_range(gt_doc, fr).astype(np.float32)
        r2 = cv2.remap(r, mx, my, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        _replace(os.path.join(out_dir, fr['range']), _png(GT.range_png(r2)))
        if progress:
            progress((k + 1) / len(cam['frames']), f'{tag}: frame {k + 1}')
    run2 = json.loads(json.dumps(run))
    tot = [float(x) for x in deg]
    if prev[1] is not None:
        from .mathutil import euler_zyx
        tot = list(euler_zyx(rot_zyx(*prev[1]) @ rot_zyx(*tot)))
    faults = [f for f in run2.get('faults_injected') or [] if not (f.get('tag') == tag and f.get('kind') == 'rotation')]
    faults.append({'tag': tag, 'kind': 'rotation', 'deg': tot, 'sensor_kind': 'camera', 'plan_pose': pose,
                   'pose_delta_deg': tot, 'how': 'synthetic: the recorded images turned exactly about the optical '
                                                 'centre (core.ego_composite.turn_drive)'})
    run2['faults_injected'] = faults
    _replace(os.path.join(out_dir, 'run.json'), json.dumps(run2, indent=1).encode('utf-8'))
    return {'tag': tag, 'deg': tot, 'rim_share': round(float(1 - ok[ok1].mean()), 4) if ok1.any() else None,
            'out_dir': out_dir}


# =========================================================================== live (vantage.sim.live)
def live_npz(project, rig_doc: dict, variant=None, settings: EgoSettings | None = None, engine=None,
             cancel=None) -> bytes:
    """sim_ego.npz for the live server: the body's triangles, normals, materials and housings, and every camera's
    layer at the planned mount at its render size (the rig's image size)."""
    import io
    st = settings or settings_of(project)
    mesh = ego_mesh(project, variant, st, engine)
    if mesh is None:
        raise ValueError('The project has no vehicle mesh.')
    caster = Caster(mesh.tri)
    arrs = {'mesh_' + k: v for k, v in mesh.arrays().items()}
    meta = {'schema': LAYERS_SCHEMA, 'project': project.name, 'paint': st.paint, 'carla_car': st.carla_car,
            'inpaint_px': st.inpaint_px, 'shadows': st.shadows, 'heading_step_deg': st.heading_step_deg,
            'frame': FRAME, 'cameras': {}}
    for c in rig_doc.get('cameras', []):
        if cancel is not None and cancel():
            raise InterruptedError('cancelled')
        w, h = int(c['image']['width']), int(c['image']['height'])
        L = ego_layer(mesh, caster, c['lens'], c['pose_vehicle'], w, h)
        arrs[f'{c["tag"]}__range'] = L.range
        arrs[f'{c["tag"]}__tri'] = L.tri
        meta['cameras'][c['tag']] = {'size': [w, h], 'layer_ms': round(L.ms, 1), 'pose': c['pose_vehicle'],
                                     'lens': c['lens'],
                                     'body_share': round(float(L.mask.sum()) / max(int(L.inside.sum()), 1), 4)}
    arrs['meta'] = np.frombuffer(json.dumps(meta).encode('utf-8'), np.uint8)
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrs)
    return buf.getvalue()


# =========================================================================== the checks on a (composited) drive
def rotation_error_deg(plan: dict, delta_added, true_deg) -> float:
    """Angle (deg) between a fitted pose (the plan's yaw, pitch, roll plus delta_added, as core.asbuilt adds it) and the
    true one (R_plan · R(true_deg), the bridge's fault)."""
    Rf = rot_zyx(*(float(plan[k]) + float(d) for k, d in zip(('yaw', 'pitch', 'roll'), delta_added)))
    Rt = mount_rotation(plan, true_deg)
    c = (np.trace(Rf.T @ Rt) - 1.0) / 2.0
    return math.degrees(math.acos(min(1.0, max(-1.0, c))))


def euler_added(plan: dict, deg) -> list:
    """A fault R_plan · R(deg) as core.asbuilt's parameters: the yaw, pitch, roll added to the plan's."""
    from .mathutil import euler_zyx
    y, p, r = euler_zyx(mount_rotation(plan, deg))
    d = [y - float(plan['yaw']), p - float(plan['pitch']), r - float(plan['roll'])]
    return [((v + 180.0) % 360.0) - 180.0 for v in d]


def drift_series(project, engine, drive_dir: str, tag: str, frames: int = 12) -> list:
    """rig_health's body-edge drift indicator (alignment, sensor px) of a drive's camera on `frames` frames spread
    over it, against the project's planned body."""
    import cv2
    from . import groundtruth as GT
    from . import rig_health as RH
    gt = GT.read(drive_dir)
    if gt is None or tag not in gt['cameras']:
        return []
    cam = project.variant.by_tag(tag)
    ev = RH.expected_view(project, engine, cam)
    fr = gt['cameras'][tag]['frames']
    out = []
    for i in np.linspace(0, len(fr) - 1, min(frames, len(fr))).astype(int):
        img = cv2.imread(os.path.join(drive_dir, fr[int(i)]['image']), cv2.IMREAD_COLOR)
        if img is None:
            continue
        a = RH.alignment(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), ev)
        out.append(None if a is None else float(a))
    return out


def check_drive(project, drive_dir: str, engine=None, asbuilt_settings=None, drift_frames: int = 12,
                progress=None, exact: bool | None = None) -> dict:
    """The as-built check (core.asbuilt, on the drive's nuScenes folder) and the drift indicator on a drive, against
    the faults it records (run.json faults_injected): per camera the injected turn, the as-built fit, its error
    (the angle between the fitted and the true mount), whether a fault was flagged, and the drift indicator's median."""
    from . import asbuilt as AB
    if engine is None:
        from .analysis import Engine
        engine = Engine()
    rig, run = drive_rig(drive_dir)
    mounts = actual_mounts(rig, run)
    st = asbuilt_settings or AB.AsBuiltSettings()
    st = AB.AsBuiltSettings.from_dict(dict(st.to_dict(), lidar_check=False,
                                           **({} if exact is None else {'exact_body': bool(exact)})))
    res = AB.run(project, engine, st, path=os.path.join(drive_dir, 'nuscenes'), progress=progress)
    rows = {}
    doc = AB.to_doc(res, project)
    for c in doc.get('cameras', []):
        tag = c['camera']
        pose, deg, m = mounts.get(tag, (None, None, None))
        if pose is None:
            continue
        injected = [float(x) for x in deg] if deg else [0.0, 0.0, 0.0]
        fit = (c.get('fit') or {}).get('ypr_deg')
        corr = c.get('correction_ypr_deg')
        err = rotation_error_deg(pose, fit, injected) if fit else None
        mag = rotation_error_deg(pose, [0, 0, 0], injected)
        flagged = c['verdict'] in ('holds_with_corrections', 'does_not_hold')
        drift = drift_series(project, engine, drive_dir, tag, drift_frames) if drift_frames else []
        dv = [x for x in drift if x is not None]
        rows[tag] = {'injected_deg': injected, 'injected_angle_deg': round(mag, 4),
                     'injected_as_added_deg': [round(x, 4) for x in euler_added(pose, injected)],
                     'verdict': c['verdict'], 'check': c['check'], 'reason': c.get('reason'),
                     'fit_deg': fit, 'stderr_deg': (c.get('fit') or {}).get('stderr_ypr_deg'),
                     'constrained': (c.get('fit') or {}).get('constrained'), 'correction_deg': corr,
                     'iou_plan': c.get('iou_plan'), 'iou_fitted': c.get('iou_fitted'),
                     'error_deg': None if err is None else round(err, 4), 'flagged': flagged,
                     'drift_px_median': round(float(np.median(dv)), 3) if dv else None,
                     'drift_frames': len(dv), 'drift_none': len(drift) - len(dv)}
    return {'schema': SCHEMA + '#check', 'drive': os.path.basename(os.path.abspath(drive_dir)),
            'composited': os.path.isfile(os.path.join(drive_dir, REPORT)), 'cameras': rows,
            'exact_body': bool(st.exact_body),
            'asbuilt_verdict': doc.get('verdict'), 'asbuilt_seconds': round(float(getattr(res, 'seconds', 0.0)), 2)}


# =========================================================================== the exact simulated mask (core.asbuilt)
_EXACT = {}


def exact_caster(project, engine, variant=None):
    """(EgoMesh, Caster) of the project's body, rig and housings, cached on the engine's geometry and the housings."""
    var = variant or project.variant
    vg = engine.vehicle_geom(project, var) if engine is not None else None
    hs = getattr(vg, 'housings', None)
    key = (id(project.vehicle.mesh), getattr(vg, 'version', None), getattr(hs, 'key', None),
           tuple((c.uid, c.x, c.y, c.z) for c in var.cameras))
    hit = _EXACT.get(key)
    if hit is None:
        mesh = ego_mesh(project, var, None, engine)
        hit = (mesh, Caster(mesh.tri)) if mesh is not None else (None, None)
        _EXACT.clear()
        _EXACT[key] = hit
    return hit


def exact_mask(project, cam, intr, rays_opt, theta, W2O, shift=(0.0, 0.0, 0.0), engine=None):
    """(body, inside the lens circle, body range m) of optical rays (K, 3) for a camera turned to W2O and moved by
    shift: core.asbuilt.sim_mask's answer by exact ray casting instead of the occlusion cube's 0.35° texels."""
    mesh, caster = exact_caster(project, engine)
    circle = theta <= intr.max_theta
    if mesh is None:
        return np.zeros(len(theta), bool), circle, np.full(len(theta), np.inf, np.float32)
    pos = cam.pos() + np.asarray(shift, float)
    t, _ = caster.cast(pos, rays_opt @ W2O, 1e-4, skip=mesh.own(cam.pos()))
    body = circle & np.isfinite(t) & (t > NEAR_M)
    return body, circle, np.where(body, t, np.inf).astype(np.float32)


def fault_study(project, drive_dir: str, work_dir: str, cases: list, engine=None, settings: EgoSettings | None = None,
                exact_modes=(False, True), drift_frames: int = 8, keep: bool = False, progress=None,
                cancel=None) -> dict:
    """The checks' answers on turned cameras of a recorded drive: per case (tag, deg) the drive turned exactly
    (turn_drive), composited with the project's body (composite_drive) and checked (check_drive, with the occlusion
    cube's and the exact simulated mask); the drive as recorded (no turn) is case 0. Rows: injected, fitted, error,
    flagged, drift."""
    if engine is None:
        from .analysis import Engine
        engine = Engine()
    st = settings or EgoSettings(lidar=False)
    os.makedirs(work_dir, exist_ok=True)
    rows = []
    allc = [(None, None)] + [(c[0], [float(x) for x in c[1]]) for c in cases]
    for k, (tag, deg) in enumerate(allc):
        if cancel is not None and cancel():
            break
        src = drive_dir
        tmp = []
        if tag is not None:
            src = os.path.join(work_dir, f'turned_{k}')
            turn_drive(drive_dir, src, tag, deg)
            tmp.append(src)
        out = os.path.join(work_dir, f'ego_{k}')
        composite_drive(project, src, out, st, engine=engine, cancel=cancel)
        tmp.append(out)
        for ex in exact_modes:
            r = check_drive(project, out, engine, drift_frames=drift_frames if ex == exact_modes[0] else 0,
                            exact=ex)
            for t, c in r['cameras'].items():
                if tag is not None and t != tag:
                    continue
                rows.append(dict(c, case=k, tag=t, exact_body=bool(ex), synthetic=tag is not None))
        if not keep:
            for d in tmp:
                shutil.rmtree(d, ignore_errors=True)
        if progress is not None:
            progress((k + 1) / len(allc), f'case {k + 1} of {len(allc)}')
    return {'schema': SCHEMA + '#fault_study', 'drive': os.path.basename(os.path.abspath(drive_dir)), 'rows': rows,
            'tolerance_deg': 0.3}


# =========================================================================== validation
def _validation_setup():
    from . import samples, simrig as SR
    from .analysis import Engine
    p = samples.tigor_sample_project()
    p.active_uid = next(v for v in p.variants if v.sensors).uid
    e = Engine()
    rig = SR.build(p, SR.SimSettings.from_dict({'image_scale': 0.25}))
    return p, e, rig


def check_layer_against_cube() -> dict:
    """The exact layer of every camera of the sample's data rig against the analysis's occlusion cube (core.asbuilt's
    simulated mask): the same body within the cube's 0.35° texels (IoU ≥ 0.95), the same depth (median within
    5 mm). Also the numpy lens against core.lens (rays within 1e-4°)."""
    from . import asbuilt as AB
    from .lens import resolve
    p, e, rig = _validation_setup()
    mesh = ego_mesh(p, engine=e)
    c = Caster(mesh.tri)
    rows = []
    for cam in rig.doc['cameras']:
        w, h = cam['image']['width'], cam['image']['height']
        L = ego_layer(mesh, c, cam['lens'], cam['pose_vehicle'], w, h)
        c0 = p.variant.by_tag(cam['tag'])
        intr = resolve(p.module_for(c0), p.lens_for(c0))
        body, _, bd = AB.sim_mask(p, c0, intr, (h, w), e)
        both = body & L.mask
        u, v = pixel_grid(w, h)
        d, _, _ = lens_unproject(cam['lens'], u, v)
        d2, _ = intr.unproject(u * intr.width / w, v * intr.height / h)
        rows.append({'tag': cam['tag'], 'iou': float((body & L.mask).sum() / max((body | L.mask).sum(), 1)),
                     'depth_median_m': float(np.median(np.abs(bd[both] - L.range[both]))) if both.any() else None,
                     'ray_max_deg': float(np.degrees(np.arccos(np.clip((d * d2).sum(1), -1, 1))).max())})
    ok = all(r['iou'] >= 0.95 and (r['depth_median_m'] is None or r['depth_median_m'] < 0.005) and
             r['ray_max_deg'] < 1e-4 for r in rows)
    return {'ok': ok, 'rows': rows}


def _gl_worker():
    """(a subprocess with PYOPENGL_PLATFORM=egl) Rasterise triangles into a float depth buffer through a pinhole:
    stdin JSON {'npz': path with tri (M,3,3) in the camera's optical frame, 'K', 'w', 'h', 'near', 'far'}; writes
    depth.npy (planar depth, inf where nothing) beside the npz; prints a JSON line."""
    import ctypes
    import sys
    req = json.loads(sys.stdin.read())
    try:
        from OpenGL import EGL, GL
        dpy = EGL.eglGetDisplay(EGL.EGL_DEFAULT_DISPLAY)
        if not dpy or not EGL.eglInitialize(dpy, None, None):
            raise RuntimeError('no EGL display')
        attrs = (EGL.EGLint * 13)(EGL.EGL_SURFACE_TYPE, EGL.EGL_PBUFFER_BIT, EGL.EGL_RED_SIZE, 8, EGL.EGL_GREEN_SIZE,
                                 8, EGL.EGL_BLUE_SIZE, 8, EGL.EGL_DEPTH_SIZE, 24, EGL.EGL_RENDERABLE_TYPE,
                                 EGL.EGL_OPENGL_BIT, EGL.EGL_NONE)
        cfg, n = EGL.EGLConfig(), EGL.EGLint()
        if not EGL.eglChooseConfig(dpy, attrs, ctypes.pointer(cfg), 1, ctypes.pointer(n)) or n.value < 1:
            raise RuntimeError('no EGL config')
        surf = EGL.eglCreatePbufferSurface(dpy, cfg, (EGL.EGLint * 5)(EGL.EGL_WIDTH, 16, EGL.EGL_HEIGHT, 16,
                                                                      EGL.EGL_NONE))
        EGL.eglBindAPI(EGL.EGL_OPENGL_API)
        ctx = EGL.eglCreateContext(dpy, cfg, EGL.EGL_NO_CONTEXT, None)
        if not ctx or not EGL.eglMakeCurrent(dpy, surf, surf, ctx):
            raise RuntimeError('no EGL OpenGL context')
    except Exception as e:                         # noqa: BLE001
        print(json.dumps({'skip': f'{type(e).__name__}: {e}'}))
        return
    with np.load(req['npz']) as z:
        tri = np.asarray(z['tri'], np.float64)
    K = np.asarray(req['K'], float)
    w, h, n_, f_ = int(req['w']), int(req['h']), float(req['near']), float(req['far'])
    fbo = GL.glGenFramebuffers(1)
    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fbo)
    cb, db = GL.glGenRenderbuffers(2)
    GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, cb)
    GL.glRenderbufferStorage(GL.GL_RENDERBUFFER, GL.GL_RGBA8, w, h)
    GL.glFramebufferRenderbuffer(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_RENDERBUFFER, cb)
    GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, db)
    GL.glRenderbufferStorage(GL.GL_RENDERBUFFER, GL.GL_DEPTH_COMPONENT32F, w, h)
    GL.glFramebufferRenderbuffer(GL.GL_FRAMEBUFFER, GL.GL_DEPTH_ATTACHMENT, GL.GL_RENDERBUFFER, db)
    if GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER) != GL.GL_FRAMEBUFFER_COMPLETE:
        print(json.dumps({'skip': 'framebuffer incomplete'}))
        return
    GL.glViewport(0, 0, w, h)
    GL.glClearDepth(1.0)
    GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
    GL.glEnable(GL.GL_DEPTH_TEST)
    GL.glDepthFunc(GL.GL_LESS)
    GL.glDisable(GL.GL_CULL_FACE)
    # optical (x right, y down, z forward) -> GL eye (x right, y up, -z forward); K's principal point in pixel-edge
    # coordinates (u = fx X/Z + cx, pixel i over [i, i+1))
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    P = np.array([[2 * fx / w, 0, 1 - 2 * cx / w, 0], [0, 2 * fy / h, 2 * cy / h - 1, 0],
                  [0, 0, -(f_ + n_) / (f_ - n_), -2 * f_ * n_ / (f_ - n_)], [0, 0, -1, 0]])
    GL.glMatrixMode(GL.GL_PROJECTION)
    GL.glLoadMatrixd(P.T.ravel())
    GL.glMatrixMode(GL.GL_MODELVIEW)
    GL.glLoadIdentity()
    eye = tri.reshape(-1, 3) * np.array([1.0, -1.0, -1.0])
    arr = np.ascontiguousarray(eye, np.float32)
    GL.glEnableClientState(GL.GL_VERTEX_ARRAY)
    GL.glVertexPointer(3, GL.GL_FLOAT, 0, arr)
    GL.glDrawArrays(GL.GL_TRIANGLES, 0, len(arr))
    GL.glFinish()
    GL.glPixelStorei(GL.GL_PACK_ALIGNMENT, 1)
    zb = GL.glReadPixels(0, 0, w, h, GL.GL_DEPTH_COMPONENT, GL.GL_FLOAT)
    zw = np.frombuffer(zb, np.float32).reshape(h, w)[::-1].astype(np.float64)
    zn = 2 * zw - 1
    Z = 2 * f_ * n_ / ((f_ + n_) - zn * (f_ - n_))
    Z = np.where(zw < 1.0, Z, np.inf)
    np.save(os.path.join(os.path.dirname(req['npz']), 'depth.npy'), Z)
    print(json.dumps({'ok': True, 'renderer': GL.glGetString(GL.GL_RENDERER).decode(errors='replace')}))


def gl_depth(tri_optical: np.ndarray, K, w: int, h: int, near: float = 0.05, far: float = 50.0,
             work_dir: str | None = None, timeout: float = 120.0):
    """(planar depth (h, w), renderer) of triangles in a pinhole camera's optical frame rasterised by OpenGL (EGL,
    headless) in a subprocess; (None, why) when no EGL / OpenGL is available."""
    import subprocess
    import sys
    import tempfile
    tmp = work_dir or tempfile.mkdtemp(prefix='ego_gl_')
    npz = os.path.join(tmp, 'tri.npz')
    np.savez(npz, tri=np.asarray(tri_optical, np.float64))
    env = dict(os.environ, PYOPENGL_PLATFORM='egl')
    src = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    env['PYTHONPATH'] = src + (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    r = subprocess.run([sys.executable, '-c', 'from vantage.core.ego_composite import _gl_worker; _gl_worker()'],
                       input=json.dumps({'npz': npz, 'K': np.asarray(K, float).tolist(), 'w': w, 'h': h,
                                         'near': near, 'far': far}),
                       capture_output=True, text=True, timeout=timeout, env=env)
    line = (r.stdout.strip().splitlines() or ['{}'])[-1]
    try:
        doc = json.loads(line)
    except ValueError:
        return None, (r.stderr or 'no answer')[-300:]
    if not doc.get('ok'):
        return None, doc.get('skip') or 'failed'
    return np.load(os.path.join(tmp, 'depth.npy')), doc.get('renderer', '')


def check_layer_against_gl(work_dir: str | None = None) -> dict:
    """The exact layer against OpenGL's depth buffer (EGL): the sample's body (with its rig and housings) seen by two
    virtual pinhole cameras 640 × 480 (front quarter, above): the same pixels covered (all but silhouette ties) and
    the same range (OpenGL's planar depth along each pixel's ray). Skipped (ok None) without EGL."""
    from .mathutil import rot_zyx as R_
    p, e, _ = _validation_setup()
    mesh = ego_mesh(p, engine=e)
    c = Caster(mesh.tri)
    w, h = 640, 480
    K = np.array([[520.0, 0, 322.3], [0, 520.0, 238.9], [0, 0, 1]])
    lens = {'model': 'pinhole', 'K': (K - np.array([[0, 0, 0.5], [0, 0, 0.5], [0, 0, 0]])).tolist(),
            'max_theta_deg': 60.0, 'image_size': [w, h]}
    rows = []
    for name, pose in (('front_quarter', {'x': 7.5, 'y': -3.5, 'z': 1.8, 'yaw': 150.0, 'pitch': 12.0, 'roll': 0.0}),
                       ('above', {'x': 1.2, 'y': 0.3, 'z': 6.5, 'yaw': 0.0, 'pitch': 88.0, 'roll': 2.0})):
        L = ego_layer(mesh, c, lens, pose, w, h)
        R = R_(pose['yaw'], pose['pitch'], pose['roll'])
        pos = np.array([pose['x'], pose['y'], pose['z']])
        tri_o = (mesh.tri - pos) @ W2O_of(R).T
        Z, renderer = gl_depth(tri_o, K, w, h, 0.05, 50.0, work_dir)
        if Z is None:
            return {'ok': None, 'skip': renderer}
        u, v = pixel_grid(w, h)
        d, _, _ = lens_unproject(lens, u, v)
        rng_gl = (Z.ravel() / d[:, 2]).reshape(h, w)
        a, b = L.mask, np.isfinite(rng_gl)
        both = a & b
        err = np.abs(L.range[both] - rng_gl[both])
        rows.append({'view': name, 'pixels': int(a.sum()), 'coverage_disagree': int((a ^ b).sum()),
                     'coverage_disagree_share': float((a ^ b).sum() / max(a.sum(), 1)),
                     'range_median_mm': float(np.median(err) * 1000),
                     'range_p99_mm': float(np.percentile(err, 99) * 1000),
                     'renderer': renderer})
    ok = all(r['coverage_disagree_share'] < 0.01 and r['range_median_mm'] < 1.0 for r in rows)
    return {'ok': ok, 'rows': rows}


def validation_checks() -> list:
    """Spec-compatible checks for vantage/validation (the lead registers them)."""
    return [
        {'id': 'ego_layer_vs_gl_depth', 'title': 'The composited body\'s depth equals OpenGL\'s depth buffer',
         'model': 'core.ego_composite', 'run': 'vantage.core.ego_composite:check_layer_against_gl',
         'reference': 'Independent: the same triangles rasterised by OpenGL (EGL, headless) into a 32-bit float '
                      'depth buffer through the same pinhole, planar depth turned into range along each pixel\'s ray',
         'tolerance': 'covered pixels differ on under 1 % of the body\'s pixels (silhouette ties of the two '
                      'samplers); range median under 1 mm (float depth precision at 1–10 m)',
         'why': 'The composited body, its mask and its range are what the checks and the ground truth stand on.'},
        {'id': 'ego_layer_vs_occlusion_cube', 'title': 'The composited body is the analysis\'s body',
         'model': 'core.ego_composite', 'run': 'vantage.core.ego_composite:check_layer_against_cube',
         'reference': 'The analysis\'s occlusion cube (core.occlusion, as core.asbuilt and core.rig_health look it '
                      'up) and core.lens\'s own unprojection, on every camera of the sample\'s data rig',
         'tolerance': 'IoU ≥ 0.95 (the cube\'s 0.35° texels along the outline), depth median within 5 mm, rays '
                      'within 1e-4°',
         'why': 'The as-built check and the drift indicator compare the frames with this body.'}]


# =========================================================================== module contract
@dataclass
class EgoResult:
    report: dict                         # composite_drive's report
    check: dict | None = None            # check_drive's answer on the composited copy (None: not asked)
    project: str = ''
    variant: str = ''
    seconds: float = 0.0


def run(project, engine=None, settings: EgoSettings | None = None, variant=None, progress=None, cancel=None,
        check: bool = True) -> EgoResult | None:
    """Composite the project's body into the recorded drive of the settings (settings.drive, into settings.out_dir or
    <drive>-ego) and, with check, run the as-built check (exact body) and the drift indicator on the copy. None when
    cancelled; ValueError (for the user) when no drive is set or it has no ground truth."""
    st = settings or settings_of(project)
    if not st.drive:
        raise ValueError('Choose a recorded CARLA drive with ground truth (a folder with run.json and '
                         'ground_truth/index.json) to composite the project\'s body into.')
    t0 = time.perf_counter()
    prog = progress or (lambda f, text='': None)
    try:
        rep = composite_drive(project, st.drive, st.out_dir or None, st, variant,
                              progress=lambda f, t='': prog(0.8 * f if check else f, t), cancel=cancel,
                              engine=engine)
    except InterruptedError:
        return None
    chk = None
    if check:
        prog(0.82, 'As-built check and drift on the composited drive')
        chk = check_drive(project, rep['out_dir'], engine, exact=True,
                          progress=lambda f, t='': prog(0.82 + 0.18 * f, t))
    v = variant or project.variant
    return EgoResult(rep, chk, project.name, v.name, round(time.perf_counter() - t0, 2))


def summary(res: EgoResult) -> dict:
    cams = res.report.get('cameras', {})
    out = {'schema': SCHEMA, 'project': res.project, 'variant': res.variant, 'out_dir': res.report.get('out_dir'),
           'cameras': {t: {k: c.get(k) for k in ('frames', 'body_share', 'hole_px', 'filled_px', 'frame_ms_median')}
                       for t, c in cams.items()},
           'lidars': {t: {k: L.get(k) for k in ('points_in', 'points_out', 'blocked_by_body')}
                      for t, L in (res.report.get('lidars') or {}).items()}, 'seconds': res.seconds}
    if res.check:
        out['check'] = {'asbuilt_verdict': res.check.get('asbuilt_verdict'),
                        'cameras': {t: {k: c.get(k) for k in ('verdict', 'injected_deg', 'fit_deg', 'error_deg',
                                                              'flagged', 'drift_px_median')}
                                    for t, c in res.check['cameras'].items()}}
    return out


def to_doc(res: EgoResult, project=None) -> dict:
    return {'schema': SCHEMA, 'project': res.project, 'variant': res.variant, 'frame': FRAME,
            'composite': res.report, 'check': res.check, 'assumptions': ASSUMPTIONS, 'seconds': res.seconds,
            'units': {'range': 'm', 'angles': 'degrees (yaw, pitch, roll intrinsic Z-Y-X, pitch positive down)',
                      'drift': 'sensor pixels (core.rig_health.alignment)'}}


def check_csv(res: EgoResult) -> str:
    import csv
    import io
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(['camera', 'verdict', 'injected_yaw', 'injected_pitch', 'injected_roll', 'fit_yaw', 'fit_pitch',
                'fit_roll', 'error_deg', 'flagged', 'drift_px_median', 'body_share'])
    for t, c in ((res.check or {}).get('cameras') or {}).items():
        inj = c.get('injected_as_added_deg') or [None] * 3
        fit = c.get('fit_deg') or [None] * 3
        w.writerow([t, c.get('verdict'), *inj, *fit, c.get('error_deg'), c.get('flagged'), c.get('drift_px_median'),
                    (res.report.get('cameras', {}).get(t) or {}).get('body_share')])
    return buf.getvalue()


def files(res: EgoResult, project=None) -> list:
    out = [('ego_composite.json', json.dumps(to_doc(res, project), indent=1, default=_jsonable).encode('utf-8'))]
    if res.check:
        out.append(('ego_composite_check.csv', check_csv(res).encode('utf-8')))
    return out


def export(res: EgoResult, project, out_dir) -> list[str]:
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for name, data in files(res, project):
        p = os.path.join(out_dir, name)
        with open(p, 'wb') as fh:
            fh.write(data)
        paths.append(p)
    return paths
