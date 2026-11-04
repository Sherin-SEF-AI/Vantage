"""Virtual cameras for physical fisheye cameras: rectify each one to a pinhole or cylindrical image, or every camera of
one or more rigs to one shared canonical camera, so datasets from different rigs and fleets can be normalised to a
single camera model before training.

Each virtual camera is a remap of the physical image through the project's real lens model (core.lens.Intrinsics):
for every virtual pixel the ray it stands for is turned into the physical camera's optical frame and projected
through the physical lens; cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR) then produces the virtual image. Pixel
conventions: Vantage puts pixel centres at i + ½; the maps and the written K follow OpenCV (centres at integers).

- virtual_pinhole: f is the physical lens's px/rad at the image centre (dr/dθ at θ = 0, so no upsampling there)
  unless given; the image is the largest rectangle (the physical aspect, or a given one) about the axis whose every
  border ray lands inside the physical image and its image circle (and under max_half_deg, a pinhole cannot reach
  90°).
- virtual_cylindrical: x is the azimuth about the virtual camera's vertical axis (u = cx + f·φ), y the tangent of the
  elevation (v = cy + f·tan ψ): wide horizontal fields without the pinhole's stretch.
- canonical: one K and image size for several cameras (also from different projects or variants): the half-angles
  every camera can fill (the smallest of their own inscribed rectangles) and the lowest of their centre
  resolutions, so every camera fills the whole canonical image (the minimum coverage over cameras is maximised:
  it is 100 %) and none is upsampled.
- level: None keeps the physical orientation (R_virtual_from_physical = I); 'roll' removes the camera's roll about its
  axis relative to the vehicle's horizon; 'horizon' also its pitch (the virtual axis horizontal, the same yaw).
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass

import numpy as np

from .analysis import camera_frames
from .lens import dr_dtheta, resolve
from .mathutil import BODY_TO_OPTICAL, rot_zyx


@dataclass
class VirtualCamera:
    tag: str
    mode: str                      # pinhole | cylindrical
    width: int
    height: int
    fx: float
    fy: float
    cx: float                      # OpenCV convention (pixel centres at integers)
    cy: float
    R_vp: np.ndarray               # virtual optical -> physical optical rotation (R_virtual_from_physical is R_vp.T)
    map_x: np.ndarray              # (H, W) float32, physical OpenCV pixel x of each virtual pixel
    map_y: np.ndarray
    valid: np.ndarray              # (H, W) bool: inside the physical image and its image circle
    half_deg: tuple                # (horizontal, vertical) half-angle covered (deg; vertical: elevation for cylindrical)
    level: str | None = None
    note: str = ''

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]])

    @property
    def R_virtual_from_physical(self) -> np.ndarray:
        return self.R_vp.T

    @property
    def coverage(self) -> float:
        return float(self.valid.mean())


def _intr(project, cam):
    return resolve(project.module_for(cam), project.lens_for(cam))


def centre_px_per_rad(intr) -> float:
    """The physical lens's radial sampling at the image centre, px/rad (dr/dθ at θ → 0)."""
    return float(dr_dtheta(1e-6, intr.f_mm, intr.projection, intr.kb)) / intr.pixel_mm


def level_rotation(cam, level=None) -> np.ndarray:
    """R_vp: virtual optical -> physical optical. None: identity; 'roll': the camera without its roll; 'horizon':
    without its roll and pitch (the same yaw)."""
    if not level:
        return np.eye(3)
    _, W2O_p = camera_frames(cam)
    R_body_v = rot_zyx(cam.yaw, 0.0 if level == 'horizon' else cam.pitch, 0.0)
    W2O_v = BODY_TO_OPTICAL @ R_body_v.T
    return W2O_p @ W2O_v.T                      # virtual optical -> vehicle -> physical optical


def _physical(intr, d_phys):
    """Physical OpenCV pixel coordinates and validity of physical-optical directions (..., 3)."""
    u, v, th, ok = intr.project_dirs(d_phys.reshape(-1, 3))
    ok &= th <= intr.max_theta
    shp = d_phys.shape[:-1]
    return (u - 0.5).reshape(shp), (v - 0.5).reshape(shp), ok.reshape(shp)


def _pinhole_dirs(tx, ty):
    d = np.stack([tx, ty, np.ones_like(tx)], -1)
    return d / np.linalg.norm(d, axis=-1, keepdims=True)


def _cyl_dirs(phi, ty):
    d = np.stack([np.sin(phi), ty, np.cos(phi)], -1)
    return d / np.linalg.norm(d, axis=-1, keepdims=True)


def _border(a, b, n=96):
    """Points on the border of the rectangle [-a, a] × [-b, b] (and its axes)."""
    t = np.linspace(-1, 1, n)
    xs = np.concatenate([t * a, t * a, np.full(n, a), np.full(n, -a), t * a, np.zeros(n)])
    ys = np.concatenate([np.full(n, b), np.full(n, -b), t * b, t * b, np.zeros(n), t * b])
    return xs, ys


def inscribed(intr, R_vp, mode='pinhole', aspect=None, max_half_deg=75.0, iters=40):
    """The largest (horizontal, vertical) extent about the virtual axis, at the given aspect (horizontal / vertical
    extent: the physical image's width / height by default), whose border rays all land in the physical image:
    (a, b) as tangents (pinhole) or (azimuth rad, elevation tangent) (cylindrical)."""
    asp = float(aspect) if aspect else intr.width / intr.height
    cap = math.radians(max_half_deg)
    lo, hi = 0.0, (math.tan(cap) if mode == 'pinhole' else cap)

    def fits(a):
        # pinhole: tangents at the aspect; cylindrical: the elevation angle is the azimuth / aspect, as a tangent
        b = a / asp if mode == 'pinhole' else math.tan(min(a / asp, cap))
        xs, ys = _border(a, b)
        d = _pinhole_dirs(xs, ys) if mode == 'pinhole' else _cyl_dirs(xs, ys)
        _, _, ok = _physical(intr, d @ R_vp.T)
        return bool(ok.all()), b
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if fits(mid)[0]:
            lo = mid
        else:
            hi = mid
    return lo, fits(lo)[1]


MAX_SIDE = 4096          # the longest side of a virtual image (px): a wide pinhole otherwise grows without bound


def _make(tag, mode, intr, R_vp, f, a, b, level, note='', max_side=MAX_SIDE):
    if 2 * f * max(a, b) > max_side:               # keep the image within max_side: lower f (the centre then loses px)
        f = max_side / (2 * max(a, b))
        note = (note + '; ' if note else '') + f'f lowered to {f:.1f} px to keep the image within {max_side} px'
    W = max(2, int(2 * round(f * a)))              # a, b: half-extents in the image plane's units (f · a pixels)
    H = max(2, int(2 * round(f * b)))
    cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
    if mode == 'pinhole':
        half = (math.degrees(math.atan(a)), math.degrees(math.atan(b)))
    else:
        half = (math.degrees(a), math.degrees(math.atan(b)))
    mx = np.empty((H, W), np.float32)
    my = np.empty((H, W), np.float32)
    ok = np.empty((H, W), bool)
    xs = (np.arange(W, dtype=np.float64) - cx) / f
    step = max(1, 262144 // W)                     # rows per chunk: a quarter million pixels at a time
    for j0 in range(0, H, step):
        j1 = min(H, j0 + step)
        ty = (np.arange(j0, j1, dtype=np.float64) - cy) / f
        X, Y = np.meshgrid(xs, ty)
        d = _pinhole_dirs(X, Y) if mode == 'pinhole' else _cyl_dirs(X, Y)
        u, v, k = _physical(intr, d @ R_vp.T)
        mx[j0:j1], my[j0:j1], ok[j0:j1] = u, v, k
    return VirtualCamera(tag, mode, W, H, float(f), float(f), float(cx), float(cy), R_vp, mx, my, ok, half, level, note)


def virtual_pinhole(project, cam, f_px=None, level=None, aspect=None, max_half_deg=75.0, intr=None,
                    max_side=MAX_SIDE) -> VirtualCamera:
    """The physical camera as the largest valid pinhole image (see the module docstring)."""
    intr = intr or _intr(project, cam)
    R_vp = level_rotation(cam, level) if cam is not None else np.eye(3)
    f = float(f_px) if f_px else centre_px_per_rad(intr)
    a, b = inscribed(intr, R_vp, 'pinhole', aspect, max_half_deg)
    return _make(getattr(cam, 'tag', 'cam'), 'pinhole', intr, R_vp, f, a, b, level,
                 'f = the physical centre px/rad' if not f_px else 'f given', max_side)


def virtual_cylindrical(project, cam, f_px=None, level='roll', aspect=None, max_half_deg=85.0,
                        intr=None, max_side=MAX_SIDE) -> VirtualCamera:
    """The physical camera as the largest valid cylindrical image (azimuth linear, elevation as a tangent)."""
    intr = intr or _intr(project, cam)
    R_vp = level_rotation(cam, level) if cam is not None else np.eye(3)
    f = float(f_px) if f_px else centre_px_per_rad(intr)
    a, b = inscribed(intr, R_vp, 'cylindrical', aspect, max_half_deg)
    return _make(getattr(cam, 'tag', 'cam'), 'cylindrical', intr, R_vp, f, a, b, level, '', max_side)


def canonical(project, cams, mode='pinhole', f_px=None, level=None, aspect=None, max_half_deg=75.0,
              projects=None, max_side=MAX_SIDE) -> list:
    """One shared virtual camera (the same K and size) for every camera given; projects: the project of each camera
    when they come from different projects (else `project` for all)."""
    projs = projects or [project] * len(cams)
    intrs = [_intr(pj, c) for pj, c in zip(projs, cams)]
    Rs = [level_rotation(c, level) for c in cams]
    asp = float(aspect) if aspect else min(i.width / i.height for i in intrs)
    ext = [inscribed(i, R, mode, asp, max_half_deg) for i, R in zip(intrs, Rs)]
    a = min(e[0] for e in ext)
    b = min(e[1] for e in ext)
    f = float(f_px) if f_px else min(centre_px_per_rad(i) for i in intrs)
    if 2 * f * max(a, b) > max_side:
        f = max_side / (2 * max(a, b))             # one f for all: lowered here, not per camera
    note = f'canonical over {", ".join(c.tag for c in cams)}: the smallest field and centre resolution among them'
    return [_make(c.tag, mode, i, R, f, a, b, level, note, max_side) for c, i, R in zip(cams, intrs, Rs)]


# ---------------------------------------------------------------- files
def _yaml(vc: VirtualCamera, source: str) -> str:
    def mat(name, M):
        M = np.asarray(M, float)
        r, c = (M.shape + (1,))[:2] if M.ndim == 2 else (M.size, 1)
        vals = ', '.join(f'{x:.10g}' for x in M.ravel())
        return f'{name}: !!opencv-matrix\n   rows: {r}\n   cols: {c}\n   dt: d\n   data: [ {vals} ]\n'
    return ('%YAML:1.0\n---\n'
            f'camera_name: "{vc.tag}_virtual"\nmodel: "{vc.mode}"\n'
            f'image_width: {vc.width}\nimage_height: {vc.height}\n'
            + mat('camera_matrix', vc.K)
            + mat('R_virtual_from_physical', vc.R_virtual_from_physical)
            + f'half_fov_deg: [ {vc.half_deg[0]:.4f}, {vc.half_deg[1]:.4f} ]\n'
            + f'level: "{vc.level or "none"}"\ncoverage: {vc.coverage:.6f}\n'
            + f'source: "{source}"\n'
            + ('note: "cylindrical: u = cx + fx * azimuth, v = cy + fy * tan(elevation), about the virtual y axis"\n'
               if vc.mode == 'cylindrical' else
               'note: "pinhole: OpenCV pixel centres at integers; remap with cv2.remap(img, map_x, map_y, '
               'cv2.INTER_LINEAR)"\n'))


def export_rectify(project, out_dir, mode='pinhole', cams=None, level=None, f_px=None, aspect=None) -> list:
    """Per camera <tag>_virtual.yaml (OpenCV FileStorage: K, size, R_virtual_from_physical, model) and <tag>_maps.npz
    (map_x, map_y float32, valid uint8) for mode 'pinhole', 'cylindrical' or 'canonical' (a shared pinhole)."""
    cams = cams if cams is not None else [c for c in project.variant.cameras if c.visible]
    os.makedirs(out_dir, exist_ok=True)
    if mode == 'canonical':
        vcs = canonical(project, cams, 'pinhole', f_px, level, aspect)
    elif mode == 'cylindrical':
        vcs = [virtual_cylindrical(project, c, f_px, level or 'roll', aspect) for c in cams]
    elif mode == 'pinhole':
        vcs = [virtual_pinhole(project, c, f_px, level, aspect) for c in cams]
    else:
        raise ValueError(mode)
    paths = []
    src = f'Vantage {project.name} / {project.variant.name}'
    for vc in vcs:
        py = os.path.join(out_dir, f'{vc.tag}_virtual.yaml')
        with open(py, 'w', encoding='utf-8') as fh:
            fh.write(_yaml(vc, src))
        pn = os.path.join(out_dir, f'{vc.tag}_maps.npz')
        np.savez_compressed(pn, map_x=vc.map_x, map_y=vc.map_y, valid=vc.valid.astype(np.uint8))
        paths += [py, pn]
    return paths
