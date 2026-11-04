"""Plücker rays and 3D positional embeddings per pixel, from each camera's exact lens and the vehicle that hides part of
its view: what multi-camera transformers (PETR, CAPE, StreamPETR, cross-rig foundation models) consume instead of a
pinhole matrix, for fisheye lenses a pinhole matrix cannot describe.

For every camera and every pixel of a grid (stride 1: every pixel; stride 8 / 16 / 32: the centre of each backbone
feature cell), with pixel centres at i + ½ (Vantage's convention; OpenCV's are at integers):

- the ray's origin o (the camera's optical centre) and unit direction d, unprojected through the real lens
  (core.lens.Intrinsics.unproject: Kannala–Brandt, equidistant, equisolid, stereographic or rectilinear);
- its Plücker coordinates: direction d and moment m = o × d (so m · d = 0 and m is the same for every point o + t d of
  the line). The 6-channel field is [d, m] (order 'dm', default) or [m, d] (order 'md');
- where the ray meets the flat ground z = 0: the range along the ray and the incidence angle to the ground's normal
  (0° straight down, 90° grazing); NaN for a ray that never meets it;
- an ego mask: the pixels that see the vehicle, its rig or a housing (the engine's occlusion cube map at the camera, as
  the coverage analysis uses), with the distance to it along the ray;
- a validity mask: inside the lens's image circle (θ ≤ max θ, in front of a rectilinear lens) and not on the vehicle.

Frames: 'vehicle' (default) is the project's ISO 8855 vehicle frame, metres: x forward, y left, z up, origin at the
rear-axle centre on the ground. nuScenes' ego frame has the same axes at the rear axle (its z origin is the ground
plane too in practice; if your ego frame's origin is at the rear axle's height, subtract that height from o_z, which
changes m but not d). 'camera' is the camera's optical frame (OpenCV: x right, y down, z forward); there o = 0 and the
moment is zero, so 'camera' gives the directions only (as a ray-direction embedding).

Pure functions; export_plucker writes one .npz per camera and a manifest.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass

import numpy as np

from .analysis import camera_frames
from .lens import resolve

SCHEMA = 'vantage.plucker/1'
CONVENTION = ('Plücker line coordinates of each pixel ray: d = unit direction, m = o × d (moment about the frame '
              'origin, metres); channels [d_x, d_y, d_z, m_x, m_y, m_z] for order "dm", [m, d] for "md". Pixel '
              '(row j, column i) of a stride-s grid samples image point (u, v) = ((i + ½)·s, (j + ½)·s), pixel '
              'centres at integer + ½; u right, v down.')


@dataclass
class RayField:
    """One camera's per-pixel rays on a stride grid. Arrays are (rows, cols[, k]); row j is v, column i is u."""
    tag: str
    uid: str
    frame: str
    stride: int
    width: int                  # sensor pixels
    height: int
    u: np.ndarray               # (cols,) image u of each column (px, centres at i + ½)
    v: np.ndarray               # (rows,)
    origin: np.ndarray          # (3,) o in the frame
    rotation: np.ndarray        # (3, 3) optical → frame: d_frame = rotation @ d_optical
    d: np.ndarray               # (rows, cols, 3) unit directions
    m: np.ndarray               # (rows, cols, 3) moments o × d
    theta: np.ndarray           # (rows, cols) off-axis angle (rad)
    in_lens: np.ndarray         # (rows, cols) inside the image circle
    ego: np.ndarray             # (rows, cols) the pixel sees the vehicle / rig / a housing
    body_depth: np.ndarray      # (rows, cols) distance to it along the ray (m), NaN elsewhere
    ground_range: np.ndarray    # (rows, cols) range to the ground z = 0 along the ray (m), NaN if it never meets it
    incidence_deg: np.ndarray   # (rows, cols) angle to the ground normal where it meets the ground, NaN elsewhere

    @property
    def valid(self) -> np.ndarray:
        return self.in_lens & ~self.ego

    def plucker(self, order='dm') -> np.ndarray:
        """(rows, cols, 6): [d, m] ('dm') or [m, d] ('md')."""
        if order not in ('dm', 'md'):
            raise ValueError(f'order must be "dm" or "md", not {order!r}')
        return np.concatenate([self.d, self.m] if order == 'dm' else [self.m, self.d], axis=-1)


def _grid(n, stride):
    k = int(math.ceil(n / stride))
    return np.minimum((np.arange(k) + 0.5) * stride, n - 1e-6)


def ray_field(project, engine, cam, stride=16, frame='vehicle', result=None, far=200.0) -> RayField:
    """The camera's rays on a stride-pixel grid (RayField), from its exact lens, in the vehicle frame (ISO 8855, rear
    axle on the ground) or its own optical frame. The ego mask and body depth come from the engine's occlusion cube at
    the camera (the vehicle, the rig and the other housings; the camera's own housing left out). far: a ray that meets
    nothing nearer counts as open sky / ground beyond it."""
    if frame not in ('vehicle', 'camera'):
        raise ValueError(f'frame must be "vehicle" or "camera", not {frame!r}')
    stride = max(int(stride), 1)
    intr = resolve(project.module_for(cam), project.lens_for(cam))
    W, H = int(intr.width), int(intr.height)
    us, vs = _grid(W, stride), _grid(H, stride)
    rows, cols = len(vs), len(us)
    _, W2O = camera_frames(cam)                      # optical = W2O @ world
    O2W = W2O.T
    pos = np.asarray(cam.pos(), float)
    cube = None
    if result is not None and cam.uid in getattr(result, 'cubes', {}):
        cube = result.cubes[cam.uid]
    elif project.vehicle.mesh is not None:
        cube = engine.cube(pos, engine.vehicle_geom(project))
    out = {k: np.empty((rows, cols, 3), np.float32) for k in ('d', 'm')}
    th_a = np.empty((rows, cols), np.float32)
    lens_a = np.empty((rows, cols), bool)
    ego_a = np.zeros((rows, cols), bool)
    bd_a = np.full((rows, cols), np.nan, np.float32)
    gr_a = np.full((rows, cols), np.nan, np.float32)
    inc_a = np.full((rows, cols), np.nan, np.float32)
    o = pos if frame == 'vehicle' else np.zeros(3)
    chunk = max(1, 400_000 // max(cols, 1))          # rows per pass: bounded memory at full resolution
    for r0 in range(0, rows, chunk):
        r1 = min(rows, r0 + chunk)
        U, V = np.meshgrid(us, vs[r0:r1])
        d_opt, theta = intr.unproject(U, V)
        dw = d_opt.reshape(-1, 3) @ O2W.T             # world directions
        dw /= np.maximum(np.linalg.norm(dw, axis=1, keepdims=True), 1e-12)
        lens = theta.reshape(-1) <= intr.max_theta + 1e-9
        if intr.projection == 'rectilinear':
            lens &= d_opt.reshape(-1, 3)[:, 2] > 1e-6
        with np.errstate(divide='ignore', invalid='ignore'):
            tg = np.where(dw[:, 2] < -1e-9, -pos[2] / dw[:, 2], np.inf)
        if cube is not None:
            depth, _ = cube.lookup(dw)
            depth = np.asarray(depth, float)
        else:
            depth = np.full(len(dw), np.inf)
        ego = lens & np.isfinite(depth) & (depth < np.minimum(tg, far))
        d_f = dw if frame == 'vehicle' else d_opt.reshape(-1, 3)
        m_f = np.cross(np.broadcast_to(o, d_f.shape), d_f)
        sl = slice(r0, r1)
        out['d'][sl] = d_f.reshape(r1 - r0, cols, 3)
        out['m'][sl] = m_f.reshape(r1 - r0, cols, 3)
        th_a[sl] = theta.reshape(r1 - r0, cols)
        lens_a[sl] = lens.reshape(r1 - r0, cols)
        ego_a[sl] = ego.reshape(r1 - r0, cols)
        bd_a[sl] = np.where(ego, depth, np.nan).reshape(r1 - r0, cols)
        hit = lens & ~ego & np.isfinite(tg)
        gr_a[sl] = np.where(hit, tg, np.nan).reshape(r1 - r0, cols)
        inc = np.degrees(np.arccos(np.clip(-dw[:, 2], -1.0, 1.0)))
        inc_a[sl] = np.where(hit, inc, np.nan).reshape(r1 - r0, cols)
    rot = O2W if frame == 'vehicle' else np.eye(3)
    return RayField(cam.tag, cam.uid, frame, stride, W, H, us, vs, o.astype(float), rot, out['d'], out['m'], th_a,
                    lens_a, ego_a, bd_a, gr_a, inc_a)


def _intr_summary(intr) -> dict:
    return {'width': int(intr.width), 'height': int(intr.height), 'projection': intr.projection,
            'f_px': float(intr.f_px), 'cx': float(intr.cx), 'cy': float(intr.cy), 'aspect': float(intr.aspect),
            'kb': [float(k) for k in intr.kb], 'max_theta_deg': math.degrees(intr.max_theta),
            'pixel_um': float(intr.pixel_mm * 1000.0)}


def export_plucker(project, engine, out_dir, stride=16, dtype='float16', frame='vehicle', order='dm', result=None,
                   cameras=None) -> list[str]:
    """Write plucker_<tag>.npz per visible camera (one at a time: a full-resolution 1920 × 1536 field is about 35 MB
    in float16) and plucker_manifest.json. Each .npz holds: plucker (rows, cols, 6), theta_rad and incidence_deg in
    `dtype`; valid, in_lens, ego (bool); ground_range_m and body_depth_m (float32, NaN where not defined); u, v (float32
    pixel positions of the columns and rows), origin (3,), rotation (3, 3: optical → frame). Returns the paths."""
    if dtype not in ('float16', 'float32'):
        raise ValueError('dtype must be float16 or float32')
    os.makedirs(out_dir, exist_ok=True)
    cams = cameras if cameras is not None else [c for c in project.variant.cameras if c.visible]
    paths, rows = [], []
    for cam in cams:
        f = ray_field(project, engine, cam, stride=stride, frame=frame, result=result)
        tag = ''.join(ch if ch.isalnum() or ch in '-_' else '_' for ch in cam.tag) or cam.uid
        path = os.path.join(out_dir, f'plucker_{tag}.npz')
        np.savez_compressed(path, plucker=f.plucker(order).astype(dtype), valid=f.valid, in_lens=f.in_lens,
                            ego=f.ego, theta_rad=f.theta.astype(dtype),
                            ground_range_m=f.ground_range.astype(np.float32),     # grazing rays exceed float16's range
                            incidence_deg=f.incidence_deg.astype(dtype), body_depth_m=f.body_depth.astype(np.float32),
                            u=f.u.astype(np.float32), v=f.v.astype(np.float32), origin=f.origin,
                            rotation=f.rotation)
        intr = resolve(project.module_for(cam), project.lens_for(cam))
        rows.append({'camera': cam.tag, 'uid': cam.uid, 'file': os.path.basename(path),
                     'shape': [int(f.d.shape[0]), int(f.d.shape[1]), 6], 'origin_m': [float(x) for x in f.origin],
                     'rotation_optical_to_frame': [[float(x) for x in r] for r in f.rotation],
                     'valid_share': float(f.valid.mean()), 'ego_share': float(f.ego.mean()),
                     'ground_share': float(np.isfinite(f.ground_range).mean()),
                     'intrinsics': _intr_summary(intr)})
        paths.append(path)
        del f
    man = {'schema': SCHEMA, 'project': project.name, 'variant': project.variant.name, 'frame': frame,
           'frame_text': ('ISO 8855 vehicle frame: x forward, y left, z up, metres, origin at the rear-axle centre on '
                          'the ground' if frame == 'vehicle' else 'camera optical frame (OpenCV): x right, y down, '
                          'z forward; origin at the optical centre, so the moments are zero'),
           'stride': int(stride), 'order': order, 'dtype': dtype, 'convention': CONVENTION,
           'units': {'d': 'unitless', 'm': 'metres', 'theta_rad': 'rad', 'ground_range_m': 'm', 'incidence_deg': 'deg',
                     'body_depth_m': 'm'},
           'cameras': rows}
    mp = os.path.join(out_dir, 'plucker_manifest.json')
    with open(mp, 'w', encoding='utf-8') as fh:
        json.dump(man, fh, indent=1)
    return paths + [mp]
