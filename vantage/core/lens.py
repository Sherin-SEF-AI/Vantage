"""Camera modules, lenses and projection models.

r(θ) is the image height in millimetres for a ray θ radians off the optical axis.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict

import numpy as np

PROJECTIONS = ('rectilinear', 'equidistant', 'equisolid', 'stereographic', 'orthographic', 'kannala_brandt')
PROJECTION_LABEL = {
    'rectilinear': 'Rectilinear', 'equidistant': 'Equidistant', 'equisolid': 'Equisolid',
    'stereographic': 'Stereographic', 'orthographic': 'Orthographic', 'kannala_brandt': 'Measured',
}
COC_MM = 0.005          # circle of confusion used for hyperfocal distance


@dataclass
class Module:
    """Sensor + housing (one physical camera product)."""
    id: str
    name: str = 'Camera'
    vendor: str = ''
    sensor: str = ''
    width_px: int = 1920
    height_px: int = 1536
    pixel_um: float = 3.0
    optical_format: str = ''
    interface: str = 'GMSL2'
    fps: float = 30.0
    hw_sync: bool = True
    size_mm: tuple = (25.0, 25.0, 43.8)
    weight_g: float = 45.0
    cables_m: tuple = (3.0, 15.0)
    builtin: bool = False

    @property
    def sensor_w_mm(self):
        return self.width_px * self.pixel_um * 1e-3

    @property
    def sensor_h_mm(self):
        return self.height_px * self.pixel_um * 1e-3

    def to_dict(self):
        d = asdict(self)
        d['size_mm'] = list(self.size_mm)
        d['cables_m'] = list(self.cables_m)
        return d

    @staticmethod
    def from_dict(d):
        d = dict(d)
        d['size_mm'] = tuple(d.get('size_mm', (25, 25, 43.8)))
        d['cables_m'] = tuple(d.get('cables_m', (3.0, 15.0)))
        known = Module.__dataclass_fields__.keys()
        return Module(**{k: v for k, v in d.items() if k in known})


@dataclass
class Lens:
    id: str
    name: str = 'Lens'
    label: str = ''                    # 'wide' / 'narrow'
    module_id: str = ''                # module it was characterised on ('' = any)
    focal_mm: float = 2.62
    f_number: float = 2.0
    focus_near_m: float = 0.35
    focus_far_m: float = math.inf
    projection: str = 'equisolid'
    kb: tuple = (0.0, 0.0, 0.0, 0.0)   # Kannala-Brandt k1..k4 (used when projection == 'kannala_brandt')
    sheet_fov: tuple = (0.0, 0.0, 0.0)  # datasheet H, V, D degrees (0 = unknown)
    max_half_deg: float = 0.0          # image-circle limit (half angle); 0 = derive from sheet D
    pp_offset_px: tuple = (0.0, 0.0)   # principal point offset from the sensor centre
    aspect: float = 1.0                # fy / fx (pixel aspect of a calibrated unit; 1 for a nominal lens)
    measured: list = field(default_factory=list)   # [(theta_deg, r_mm), ...] calibration samples
    fit_rms_px: float = 0.0
    builtin: bool = False

    def to_dict(self):
        d = asdict(self)
        d['kb'] = list(self.kb)
        d['sheet_fov'] = list(self.sheet_fov)
        d['pp_offset_px'] = list(self.pp_offset_px)
        d['focus_far_m'] = None if math.isinf(self.focus_far_m) else self.focus_far_m
        d['measured'] = [list(p) for p in self.measured]
        return d

    @staticmethod
    def from_dict(d):
        d = dict(d)
        for k in ('kb', 'sheet_fov', 'pp_offset_px'):
            if k in d:
                d[k] = tuple(d[k])
        if d.get('focus_far_m') is None:
            d['focus_far_m'] = math.inf
        d['measured'] = [tuple(p) for p in d.get('measured', [])]
        known = Lens.__dataclass_fields__.keys()
        return Lens(**{k: v for k, v in d.items() if k in known})


# ---------------------------------------------------------------- projection maths
def r_of_theta(theta, f, projection, kb=(0, 0, 0, 0)):
    t = np.asarray(theta, float)
    if projection == 'rectilinear':
        return f * np.tan(np.minimum(t, math.radians(89.9)))
    if projection == 'equidistant':
        return f * t
    if projection == 'equisolid':
        return 2 * f * np.sin(t / 2)
    if projection == 'stereographic':
        return 2 * f * np.tan(np.minimum(t, math.radians(179)) / 2)
    if projection == 'orthographic':
        return f * np.sin(np.minimum(t, math.pi / 2))
    if projection == 'kannala_brandt':
        k1, k2, k3, k4 = kb
        t2 = t * t
        return f * t * (1 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4))))
    raise ValueError(projection)


def dr_dtheta(theta, f, projection, kb=(0, 0, 0, 0)):
    t = np.asarray(theta, float)
    if projection == 'rectilinear':
        c = np.cos(np.minimum(t, math.radians(89.9)))
        return f / (c * c)
    if projection == 'equidistant':
        return f * np.ones_like(t)
    if projection == 'equisolid':
        return f * np.cos(t / 2)
    if projection == 'stereographic':
        c = np.cos(t / 2)
        return f / (c * c)
    if projection == 'orthographic':
        return f * np.cos(np.minimum(t, math.pi / 2))
    if projection == 'kannala_brandt':
        k1, k2, k3, k4 = kb
        t2 = t * t
        return f * (1 + t2 * (3 * k1 + t2 * (5 * k2 + t2 * (7 * k3 + t2 * 9 * k4))))
    raise ValueError(projection)


def theta_of_r(r, f, projection, kb=(0, 0, 0, 0)):
    x = np.asarray(r, float) / f
    if projection == 'rectilinear':
        return np.arctan(x)
    if projection == 'equidistant':
        return x
    if projection == 'equisolid':
        return 2 * np.arcsin(np.clip(x / 2, -1, 1))
    if projection == 'stereographic':
        return 2 * np.arctan(x / 2)
    if projection == 'orthographic':
        return np.arcsin(np.clip(x, -1, 1))
    if projection == 'kannala_brandt':
        t = np.array(x, float, copy=True)
        for _ in range(12):
            g = r_of_theta(t, 1.0, projection, kb) - x
            d = dr_dtheta(t, 1.0, projection, kb)
            t = t - g / np.where(np.abs(d) < 1e-9, 1e-9, d)
            t = np.clip(t, 0, math.pi)
        return t
    raise ValueError(projection)


def fit_kb_to_fov(f_mm, sheet_fov_deg, half_sizes_mm, base='equisolid', ridge=1e-3):
    """Kannala-Brandt k1..k4 so r(θ_sheet) matches the sensor half sizes.

    The distortion polynomial is fitted to the datasheet FOV points plus the base projection
    sampled densely (so the curve stays well behaved), weighting the sheet points strongly.
    """
    th_s, r_s = [], []
    for fov, half in zip(sheet_fov_deg, half_sizes_mm):
        if fov and fov > 0:
            th_s.append(math.radians(fov / 2))
            r_s.append(half)
    th_b = np.linspace(0.02, max(th_s) if th_s else 1.2, 24)
    r_b = r_of_theta(th_b, f_mm, base)
    th = np.concatenate([np.array(th_s), th_b])
    r = np.concatenate([np.array(r_s), r_b])
    w = np.concatenate([np.full(len(th_s), 8.0), np.full(len(th_b), 1.0)])
    # r/f - θ = k1 θ^3 + k2 θ^5 + k3 θ^7 + k4 θ^9
    A = np.stack([th ** 3, th ** 5, th ** 7, th ** 9], axis=1) * w[:, None]
    b = (r / f_mm - th) * w
    k = np.linalg.solve(A.T @ A + ridge * np.eye(4), A.T @ b)
    return tuple(float(v) for v in k)


@dataclass
class Intrinsics:
    """Everything the analysis needs to map a ray to a pixel, resolved for one module + lens."""
    width: int
    height: int
    pixel_mm: float
    f_mm: float
    projection: str
    kb: tuple
    cx: float
    cy: float
    max_theta: float     # radians, image-circle limit
    focus_near: float
    focus_far: float
    aspect: float = 1.0  # fy / fx

    @property
    def f_px(self):
        return self.f_mm / self.pixel_mm

    def project_dirs(self, d_opt: np.ndarray):
        """Optical-frame directions (K,3) -> (u, v, theta, valid_in_image)."""
        x, y, z = d_opt[..., 0], d_opt[..., 1], d_opt[..., 2]
        rho = np.hypot(x, y)
        theta = np.arctan2(rho, z)
        r = r_of_theta(theta, self.f_mm, self.projection, self.kb) / self.pixel_mm
        s = np.where(rho > 1e-12, r / np.where(rho > 1e-12, rho, 1.0), 0.0)
        u = self.cx + x * s
        v = self.cy + y * s * self.aspect
        ok = (theta <= self.max_theta) & (u >= 0) & (u <= self.width) & (v >= 0) & (v <= self.height)
        if self.projection == 'rectilinear':
            ok &= z > 1e-6
        return u, v, theta, ok

    def unproject(self, u, v):
        """Pixel coordinates -> optical-frame unit directions and theta."""
        x = (np.asarray(u, float) - self.cx) * self.pixel_mm
        y = (np.asarray(v, float) - self.cy) / self.aspect * self.pixel_mm
        r = np.hypot(x, y)
        theta = theta_of_r(r, self.f_mm, self.projection, self.kb)
        s = np.where(r > 1e-12, np.sin(theta) / np.where(r > 1e-12, r, 1.0), 0.0)
        d = np.stack([x * s, y * s, np.cos(theta)], axis=-1)
        return d, theta

    def margin(self, u, v, theta):
        """Signed distance-like margin to the image border, positive inside (unitless, ~[-1, 1])."""
        mu = np.minimum(u, self.width - u) / (self.width / 2)
        mv = np.minimum(v, self.height - v) / (self.height / 2)
        mt = (self.max_theta - theta) / max(self.max_theta, 1e-6)
        return np.minimum(np.minimum(mu, mv), mt * 2)

    def density_px_per_cm(self, theta, rng_m):
        """Pixel density on a target facing the camera at range rng_m, in px/cm:
        sqrt(radial x tangential) angular sampling, i.e. sqrt(pixels per cm^2)."""
        th = np.maximum(theta, 1e-4)
        radial = np.maximum(dr_dtheta(th, self.f_mm, self.projection, self.kb), 1e-6) / self.pixel_mm
        tang = np.maximum(r_of_theta(th, self.f_mm, self.projection, self.kb), 1e-9) / self.pixel_mm / np.sin(np.minimum(th, math.pi - 1e-3))
        return np.sqrt(radial * tang) / (np.maximum(rng_m, 1e-3) * 100.0)


def resolve(module: Module, lens: Lens) -> Intrinsics:
    pm = module.pixel_um * 1e-3
    proj = lens.projection
    kb = tuple(lens.kb)
    max_half = lens.max_half_deg
    if not max_half:
        d = lens.sheet_fov[2] if len(lens.sheet_fov) > 2 and lens.sheet_fov[2] else 0
        if d:
            max_half = min(d / 2 + 2.0, 95.0)
        else:
            hd = math.hypot(module.sensor_w_mm, module.sensor_h_mm) / 2
            # the same 95° cap as the datasheet path: a short lens would otherwise reach past 180° (behind itself)
            max_half = min(math.degrees(float(theta_of_r(hd, lens.focal_mm, proj, kb))) + 1.0, 95.0)
    if proj == 'rectilinear':
        max_half = min(max_half, 89.0)
    return Intrinsics(width=module.width_px, height=module.height_px, pixel_mm=pm, f_mm=lens.focal_mm,
                      projection=proj, kb=kb,
                      cx=module.width_px / 2 + lens.pp_offset_px[0], cy=module.height_px / 2 + lens.pp_offset_px[1],
                      max_theta=math.radians(max_half), focus_near=lens.focus_near_m, focus_far=lens.focus_far_m,
                      aspect=float(lens.aspect or 1.0))


def model_fov(module: Module, lens: Lens, projection: str | None = None, kb=None):
    """Horizontal, vertical and diagonal FOV (degrees) for the chosen projection."""
    proj = projection or lens.projection
    kb = lens.kb if kb is None else kb
    hw, hh = module.sensor_w_mm / 2, module.sensor_h_mm / 2
    hd = math.hypot(hw, hh)
    out = []
    for half in (hw, hh, hd):
        out.append(2 * math.degrees(float(theta_of_r(half, lens.focal_mm, proj, kb))))
    return tuple(out)


def derived(module: Module, lens: Lens) -> dict:
    intr = resolve(module, lens)
    pm = intr.pixel_mm
    fpx = lens.focal_mm / pm
    fh, fv, fd = model_fov(module, lens)
    th_h = math.radians(fh / 2)
    px_deg_c = float(dr_dtheta(1e-4, lens.focal_mm, lens.projection, lens.kb)) / pm * math.pi / 180
    px_deg_e = float(dr_dtheta(th_h, lens.focal_mm, lens.projection, lens.kb)) / pm * math.pi / 180
    hyper = lens.focal_mm ** 2 / (lens.f_number * COC_MM) / 1000 + lens.focal_mm / 1000
    return {
        'fov': (fh, fv, fd), 'ifov_deg': math.degrees(pm / lens.focal_mm), 'px_per_deg_centre': px_deg_c,
        'px_per_deg_edge': px_deg_e, 'h_edge_deg': fh / 2, 'f_px': fpx, 'cx': intr.cx, 'cy': intr.cy,
        'hyperfocal_m': hyper, 'image_circle_r_mm': float(r_of_theta(intr.max_theta, lens.focal_mm, lens.projection, lens.kb)),
        'max_half_deg': math.degrees(intr.max_theta), 'half_diag_mm': math.hypot(module.sensor_w_mm, module.sensor_h_mm) / 2,
    }


# ---------------------------------------------------------------- built-in library
def builtin_modules() -> list[Module]:
    return [Module(id='sturdecam31', name='STURDeCAM31', vendor='e-con Systems', sensor='Sony ISX031',
                   width_px=1920, height_px=1536, pixel_um=3.0, optical_format='1/2.4"', interface='GMSL2',
                   fps=30.0, hw_sync=True, size_mm=(25.0, 25.0, 43.8), weight_g=45.0, cables_m=(3.0, 15.0),
                   builtin=True)]


def builtin_lenses() -> list[Lens]:
    m = builtin_modules()[0]
    hs = (m.sensor_w_mm / 2, m.sensor_h_mm / 2, math.hypot(m.sensor_w_mm, m.sensor_h_mm) / 2)
    wide = Lens(id='h05r2', name='H05R2', label='wide', module_id='sturdecam31', focal_mm=2.62, f_number=2.0,
                focus_near_m=0.35, projection='kannala_brandt', sheet_fov=(128.7, 98.7, 170.1), builtin=True)
    wide.kb = fit_kb_to_fov(2.62, wide.sheet_fov, hs, base='equisolid')
    narrow = Lens(id='h03r2', name='H03R2', label='narrow', module_id='sturdecam31', focal_mm=5.5, f_number=2.0,
                  focus_near_m=1.28, projection='kannala_brandt', sheet_fov=(54.04, 42.86, 70.17), builtin=True)
    narrow.kb = fit_kb_to_fov(5.5, narrow.sheet_fov, hs, base='equisolid')
    return [wide, narrow]
