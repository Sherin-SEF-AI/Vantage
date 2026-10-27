"""Camera modules, lenses and projection models, and the built-in library of e-con Systems camera modules.

r(θ) is the image height in millimetres for a ray θ radians off the optical axis.

The built-in library holds e-con Systems' automotive / GMSL2 / FPD-Link camera modules whose figures could be read on
e-con's own public product pages (see LIBRARY_NOTE): each module and lens carries a `note` naming the page and the date
it was read, and a figure e-con does not state is left as not given (0 or ''), never filled with a guess. e-con's
public pages state each lens's field of view but not its focal length: a built-in lens's focal length is then derived
from the stated FOV (the base projection that fits them best), and Kannala-Brandt terms are fitted so the model
reproduces the stated FOV (fit_kb_to_fov). The FOV is a measurement e-con publishes; the focal length and the
distortion curve between the stated angles are a model.
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
SHUTTERS = ('', 'rolling', 'global')
# a module whose size leaves a dimension out (0) is drawn and checked with these (width, height, length) mm: the
# STURDeCAM31 library figures, the same fallback core/brackets.py uses
TYPICAL_BODY_MM = (25.0, 25.0, 43.8)


def _f(v, default=0.0) -> float:
    """A library figure as a finite float (a hand-edited project file may hold anything)."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) else default


def _nums(v, n=None, default=()) -> tuple:
    """A tuple of finite floats from a list/tuple (n: exactly that many), else the default."""
    if not isinstance(v, (list, tuple)):
        return tuple(default)
    out = []
    for x in v:
        try:
            x = float(x)
        except (TypeError, ValueError):
            return tuple(default)
        if not math.isfinite(x):
            return tuple(default)
        out.append(x)
    if n is not None and len(out) != n:
        return tuple(default)
    return tuple(out)


@dataclass
class Module:
    """Sensor + housing (one physical camera product). A figure of 0 or '' is not given by the source."""
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
    size_mm: tuple = (25.0, 25.0, 43.8)   # width, height (across the optical axis), length along it; 0 = not given
    weight_g: float = 45.0
    cables_m: tuple = (3.0, 15.0)
    power_w: float = 0.0               # 0 = not given (the power budget then uses its typical camera figure)
    builtin: bool = False
    shutter: str = ''                  # 'rolling' | 'global' | '' (not given)
    dynamic_range_db: float = 0.0      # the sensor's stated dynamic range (HDR figure); 0 = not given
    ip_rating: str = ''                # e.g. 'IP69K'; '' = not given or no enclosure
    temp_c: tuple = ()                 # operating temperature (min, max) °C; () = not given
    chroma: str = ''                   # 'colour' | 'monochrome' | 'RGB-IR' | ''
    url: str = ''                      # where the figures come from
    note: str = ''                     # the source and any caveat, shown in the library and the module editor

    @property
    def sensor_w_mm(self):
        return self.width_px * self.pixel_um * 1e-3

    @property
    def sensor_h_mm(self):
        return self.height_px * self.pixel_um * 1e-3

    @property
    def body_mm(self) -> tuple:
        """(width, height, length) mm for drawing and clearance checks: size_mm with a dimension that is not given
        (0, missing, junk) taken from TYPICAL_BODY_MM."""
        sz = list(_nums(self.size_mm)) + [0.0, 0.0, 0.0]
        return tuple(v if v > 1.0 else d for v, d in zip(sz[:3], TYPICAL_BODY_MM))

    def summary(self) -> str:
        """One line for tooltips and the assistant: what the module is, figures not given left out."""
        parts = [f'{self.vendor} {self.name}'.strip(), self.sensor, f'{self.width_px} × {self.height_px}',
                 f'{self.pixel_um:g} µm', self.optical_format, self.interface, f'{self.fps:g} fps',
                 f'{self.shutter} shutter' if self.shutter else '', self.chroma if self.chroma != 'colour' else '',
                 f'{self.dynamic_range_db:g} dB' if _f(self.dynamic_range_db) > 0 else '', self.ip_rating,
                 f'{self.temp_c[0]:g} to {self.temp_c[1]:g} °C' if len(self.temp_c) == 2 else '',
                 f'{self.power_w:g} W' if _f(self.power_w) > 0 else '']
        return ' · '.join(p for p in parts if p)

    def missing(self) -> list[str]:
        """The figures the source does not give (shown so a user knows what is unconfirmed)."""
        out = []
        sz = (list(_nums(self.size_mm)) + [0.0, 0.0, 0.0])[:3]
        if not all(v > 0 for v in sz):
            out.append('length (depth)' if sz[0] > 0 and sz[1] > 0 else 'size')
        for k, text in (('weight_g', 'weight'), ('power_w', 'power')):
            if _f(getattr(self, k)) <= 0:
                out.append(text)
        if not self.shutter:
            out.append('shutter')
        if _f(self.dynamic_range_db) <= 0:
            out.append('dynamic range')
        if not self.ip_rating:
            out.append('IP rating')
        if len(self.temp_c) != 2:
            out.append('operating temperature')
        return out

    def to_dict(self):
        d = asdict(self)
        d['size_mm'] = list(self.size_mm)
        d['cables_m'] = list(self.cables_m)
        d['temp_c'] = list(self.temp_c)
        return d

    @staticmethod
    def from_dict(d):
        """Tolerant: older projects lack the newer fields (they take the defaults) and junk in them is dropped."""
        d = dict(d)
        d['size_mm'] = _nums(d.get('size_mm', (25, 25, 43.8)), 3, (25.0, 25.0, 43.8))
        d['cables_m'] = _nums(d.get('cables_m', (3.0, 15.0)), None, (3.0, 15.0)) or (3.0, 15.0)
        d['temp_c'] = _nums(d.get('temp_c', ()), 2, ())
        d['dynamic_range_db'] = max(0.0, _f(d.get('dynamic_range_db', 0.0)))
        for k in ('shutter', 'ip_rating', 'chroma', 'url', 'note'):
            d[k] = d.get(k) if isinstance(d.get(k), str) else ''
        if d['shutter'] not in SHUTTERS:
            d['shutter'] = ''
        known = Module.__dataclass_fields__.keys()
        return Module(**{k: v for k, v in d.items() if k in known})


@dataclass
class Lens:
    id: str
    name: str = 'Lens'
    label: str = ''                    # 'wide' / 'narrow' / 'fisheye'
    module_id: str = ''                # module it was characterised on ('' = any)
    focal_mm: float = 2.62
    f_number: float = 2.0              # 0 = not given
    focus_near_m: float = 0.35         # 0 = not given (no near focus limit is applied)
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
    note: str = ''                     # the source and how the model was made from it

    def summary(self) -> str:
        """One line for tooltips and the assistant: the stated FOV, focal length, aperture and focus range."""
        fov = ' × '.join(f'{v:g}° {a}' for v, a in zip(tuple(self.sheet_fov)[:3], 'HVD') if _f(v) > 0)
        far = '∞' if math.isinf(_f(self.focus_far_m, math.inf)) else f'{self.focus_far_m:g} m'
        parts = [f'{self.name} ({self.label})' if self.label else self.name, fov, f'{self.focal_mm:.3g} mm',
                 f'F{self.f_number:g}' if _f(self.f_number) > 0 else 'aperture not given',
                 f'focus {self.focus_near_m:g} m – {far}' if _f(self.focus_near_m) > 0 else 'focus not given']
        return ' · '.join(p for p in parts if p)

    def missing(self) -> list[str]:
        out = []
        if _f(self.f_number) <= 0:
            out.append('aperture')
        if _f(self.focus_near_m) <= 0:
            out.append('focus distance')
        if not any(_f(v) > 0 for v in tuple(self.sheet_fov)[:3]):
            out.append('field of view')
        return out

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
        d['note'] = d.get('note') if isinstance(d.get('note'), str) else ''
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


def fit_kb_to_fov(f_mm, sheet_fov_deg, half_sizes_mm, base='equisolid', ridge=1e-3, weight=8.0):
    """Kannala-Brandt k1..k4 so r(θ_sheet) matches the sensor half sizes.

    The distortion polynomial is fitted to the datasheet FOV points plus the base projection
    sampled densely (so the curve stays well behaved), weighting the sheet points strongly (weight against 1 for each
    base sample; SHEET_WEIGHT reproduces the sheet within ~0.3° for the built-in lenses).
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
    w = np.concatenate([np.full(len(th_s), float(weight)), np.full(len(th_b), 1.0)])
    # r/f - θ = k1 θ^3 + k2 θ^5 + k3 θ^7 + k4 θ^9
    A = np.stack([th ** 3, th ** 5, th ** 7, th ** 9], axis=1) * w[:, None]
    b = (r / f_mm - th) * w
    k = np.linalg.solve(A.T @ A + ridge * np.eye(4), A.T @ b)
    return tuple(float(v) for v in k)


SHEET_WEIGHT = 50.0      # fit_kb_to_fov weight for the built-in e-con lenses and the module editor's fit
FIT_BASES = ('rectilinear', 'stereographic', 'equidistant', 'equisolid')


def _sheet_points(sheet_fov_deg, half_sizes_mm):
    th = [math.radians(f / 2) for f in tuple(sheet_fov_deg)[:3] if f and f > 0]
    r = [h for f, h in zip(tuple(sheet_fov_deg)[:3], half_sizes_mm) if f and f > 0]
    return np.array(th, float), np.array(r, float)


def focal_from_fov(sheet_fov_deg, half_sizes_mm, base) -> float:
    """The focal length (mm) of a `base` projection lens that best reproduces the stated FOV points: least squares of
    r = f·g(θ) over the stated H/V/D points, relative to each half size (so the diagonal does not dominate)."""
    th, r = _sheet_points(sheet_fov_deg, half_sizes_mm)
    if not len(th):
        raise ValueError('no field of view to fit')
    g = np.asarray(r_of_theta(th, 1.0, base), float) / r
    return float(g.sum() / (g @ g))


def best_base(sheet_fov_deg, half_sizes_mm, bases=FIT_BASES) -> tuple[str, float, float]:
    """(base projection, focal mm, relative RMS) of the base that alone best reproduces the stated FOV points: the
    honest starting curve for the distortion fit. Rectilinear is only a candidate below 160° (a wider lens cannot be
    one)."""
    th, r = _sheet_points(sheet_fov_deg, half_sizes_mm)
    best = None
    for b in bases:
        if b == 'rectilinear' and th.max() >= math.radians(80):
            continue
        f = focal_from_fov(sheet_fov_deg, half_sizes_mm, b)
        rel = float(np.sqrt(np.mean((np.asarray(r_of_theta(th, f, b), float) / r - 1.0) ** 2)))
        if best is None or rel < best[2] - 1e-12:
            best = (b, f, rel)
    return best


def monotonic(f_mm, kb, max_theta_rad, n=720) -> bool:
    """True when the Kannala-Brandt curve r(θ) rises all the way to max_theta (a lens model that folds back would map
    two rays to one pixel)."""
    t = np.linspace(1e-4, max(float(max_theta_rad), 1e-3), n)
    return bool(np.all(dr_dtheta(t, f_mm, 'kannala_brandt', kb) > 0))


def fit_sheet(f_mm, sheet_fov_deg, half_sizes_mm, base='equisolid', reach_deg=0.0):
    """fit_kb_to_fov at SHEET_WEIGHT, falling back to the gentler default weight if the strong fit would fold back
    before reach_deg (default: the widest stated half angle + 2°)."""
    reach = math.radians(reach_deg or max(f / 2 for f in tuple(sheet_fov_deg)[:3] if f) + 2.0)
    for w in (SHEET_WEIGHT, 20.0, 8.0):
        kb = fit_kb_to_fov(f_mm, sheet_fov_deg, half_sizes_mm, base=base, weight=w)
        if monotonic(f_mm, kb, reach):
            return kb
    return kb


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
        sqrt(radial x tangential) angular sampling, i.e. sqrt(pixels per cm^2).  fy = aspect * fx stretches every
        pixel area by the aspect, so the density by its square root."""
        th = np.maximum(theta, 1e-4)
        radial = np.maximum(dr_dtheta(th, self.f_mm, self.projection, self.kb), 1e-6) / self.pixel_mm
        tang = np.maximum(r_of_theta(th, self.f_mm, self.projection, self.kb), 1e-9) / self.pixel_mm / np.sin(np.minimum(th, math.pi - 1e-3))
        return np.sqrt(radial * tang * self.aspect) / (np.maximum(rng_m, 1e-3) * 100.0)


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


def model_fov(module: Module, lens: Lens, projection: str | None = None, kb=None, circle: bool = False):
    """Horizontal, vertical and diagonal FOV (degrees) for the chosen projection.

    circle=True gives the field the image actually shows: each limited by the lens's image circle (resolve's
    max_theta), e.g. a fisheye whose circle cuts the sensor corners."""
    proj = projection or lens.projection
    kb = lens.kb if kb is None else kb
    hw, hh = module.sensor_w_mm / 2, module.sensor_h_mm / 2
    hd = math.hypot(hw, hh)
    lim = math.degrees(resolve(module, lens).max_theta) if circle else math.inf
    out = []
    for half in (hw, hh, hd):
        out.append(2 * min(math.degrees(float(theta_of_r(half, lens.focal_mm, proj, kb))), lim))
    return tuple(out)


def derived(module: Module, lens: Lens) -> dict:
    """Figures the module editor shows. hyperfocal_m is None when the aperture is not given."""
    intr = resolve(module, lens)
    pm = intr.pixel_mm
    fpx = lens.focal_mm / pm
    fh, fv, fd = model_fov(module, lens)
    th_h = math.radians(fh / 2)
    px_deg_c = float(dr_dtheta(1e-4, lens.focal_mm, lens.projection, lens.kb)) / pm * math.pi / 180
    px_deg_e = float(dr_dtheta(th_h, lens.focal_mm, lens.projection, lens.kb)) / pm * math.pi / 180
    fno = _f(lens.f_number)
    hyper = lens.focal_mm ** 2 / (fno * COC_MM) / 1000 + lens.focal_mm / 1000 if fno > 0 else None
    return {
        'fov': (fh, fv, fd), 'ifov_deg': math.degrees(pm / lens.focal_mm), 'px_per_deg_centre': px_deg_c,
        'px_per_deg_edge': px_deg_e, 'h_edge_deg': fh / 2, 'f_px': fpx, 'cx': intr.cx, 'cy': intr.cy,
        'hyperfocal_m': hyper, 'image_circle_r_mm': float(r_of_theta(intr.max_theta, lens.focal_mm, lens.projection, lens.kb)),
        'max_half_deg': math.degrees(intr.max_theta), 'half_diag_mm': math.hypot(module.sensor_w_mm, module.sensor_h_mm) / 2,
    }


# ---------------------------------------------------------------- built-in library
CHECKED = '2026-10-24'             # when e-con's pages were read for the figures below
_E = 'https://www.e-consystems.com'
LIBRARY_NOTE = (
    'Built-in modules: e-con Systems camera modules, with the figures e-con states on its public product pages '
    f'(read {CHECKED}; each entry names its page). A figure e-con does not state is left as not given (0 or blank) '
    'and the app then uses its typical figure where it needs one. e-con states each lens\'s field of view but not '
    'its focal length or aperture: the focal length is derived from the stated FOV with the base projection that fits '
    'it best, and the distortion terms are fitted so the model reproduces the stated FOV within 0.5°. The resolution '
    'between the stated angles, the centre resolution and the image circle are therefore modelled, not measured; a '
    'unit calibration (Calibrate…) replaces them with your camera\'s own.')


def _module(mid, name, sensor, res, pixel_um, fmt, fps, *, shutter, dr_db, temp, size, urls, extra='', weight=0.0,
            cables=(15.0,), power=0.0, sync=False, ip='', chroma='colour', interface='GMSL2') -> Module:
    pages = ' and '.join(urls)
    note = f'Figures from e-con Systems\' page{"s" if len(urls) > 1 else ""} {pages} (read {CHECKED}).'
    return Module(id=mid, name=name, vendor='e-con Systems', sensor=sensor, width_px=res[0], height_px=res[1],
                  pixel_um=pixel_um, optical_format=fmt, interface=interface, fps=fps, hw_sync=sync,
                  size_mm=tuple(float(v) for v in size), weight_g=weight, cables_m=tuple(float(c) for c in cables),
                  power_w=power, builtin=True, shutter=shutter, dynamic_range_db=dr_db, ip_rating=ip,
                  temp_c=tuple(float(t) for t in temp), chroma=chroma, url=urls[0],
                  note=(note + ' ' + extra).strip())


def builtin_modules() -> list[Module]:
    """e-con Systems' GMSL2 / FPD-Link camera modules whose figures e-con's public pages state (see LIBRARY_NOTE).
    The first is the STURDeCAM31 every new camera starts with."""
    k = _E + '/nvidia-cameras/jetson-agx-orin-cameras/'
    return [
        # ---- STURDeCAM: enclosed, automotive grade
        _module('sturdecam31', 'STURDeCAM31', 'Sony ISX031', (1920, 1536), 3.0, '1/2.42"', 30.0,
                shutter='rolling', dr_db=120.0, temp=(-40, 85), size=(25.0, 25.0, 43.8), weight=40.0,
                cables=(3.0, 15.0), sync=True, ip='IP69K', power=1.324,
                urls=(_E + '/camera-modules/sony-isx031-ip69k-gmsl2-hdr-camera.asp',
                      k + 'sony-isx031-ip69k-hdr-gmsl2-camera.asp'),
                extra='1920 × 1536 at 30 fps with up to 8 cameras on one Orin (60 fps with up to 4). Weight 40 g with '
                      'the lens. e-con gives the enclosure as 25 × 25 mm; the 43.8 mm length is the earlier library '
                      'figure, not on e-con\'s current pages. 3 m cable by default, 15 m on request.'),
        _module('sturdecam31fsh', 'STURDeCAM31FSH', 'Sony ISX031', (1920, 1536), 3.0, '1/2.42"', 30.0,
                shutter='rolling', dr_db=120.0, temp=(-40, 85), size=(35.0, 35.0, 36.8), cables=(3.0, 15.0),
                power=4.5, ip='IP69K',
                urls=(_E + '/camera-modules/automatic-defogging-de-icing-lens/3mp-sony-isx031-gmsl2-camera.asp',),
                extra='Heated lens for de-icing and de-fogging: the heater draws 2–4 W while enabled; 4.5 W is '
                      'e-con\'s typical figure with the heater on. Hardware frame sync is not stated.'),
        _module('sturdecam34', 'STURDeCAM34', 'onsemi AR0341AT', (1920, 1536), 2.1, '1/3.6"', 30.0,
                shutter='rolling', dr_db=140.0, temp=(-40, 85), size=(25.0, 25.0, 0.0), weight=46.3,
                cables=(3.0, 15.0), power=2.192, sync=True, ip='IP69K',
                urls=(_E + '/automotive-cameras/3mp-ar0341at-ip69k-gmsl2-140db-hdr-camera.asp',
                      k + '3mp-ar0341at-ip69k-gmsl2-140db-hdr-camera.asp'),
                extra='140 dB HDR is e-con\'s headline figure; its specification list says up to 150 dB. Weight '
                      '46.3 g with lens and enclosure. Enclosure 25 × 25 mm (length not given).'),
        _module('sturdecam39', 'STURDeCAM39', 'OmniVision OX03J10', (1920, 1536), 3.0, '1/2.4"', 30.0,
                shutter='rolling', dr_db=140.0, temp=(-40, 85), size=(25.0, 25.0, 48.53), power=1.332, sync=True,
                ip='IP69K',
                urls=(_E + '/gmsl-cameras/3mp-ox03j10-140db-hdr-surround-view-gmsl2-camera.asp',
                      k + '3mp-ox03j10-140db-hdr-surround-view-gmsl2-camera.asp'),
                extra='Surround-view camera. 25 × 25 × 48.53 mm with the lens (38.77 mm without). Cable up to 15 m. '
                      'Frame sync per e-con\'s trigger-mode application note.'),
        _module('sturdecam25', 'STURDeCAM25', 'onsemi AR0234CS', (1920, 1200), 3.0, '1/2.6"', 60.0,
                shutter='global', dr_db=71.4, temp=(-30, 70), size=(42.0, 42.0, 45.22), weight=77.67,
                power=1.5, sync=True, ip='IP67',
                urls=(_E + '/camera-modules/gmsl2-ar0234-global-shutter-camera.asp',
                      k + 'fhd-ar0234-IP67-global-shutter-gmsl2-multi-camera.asp'),
                extra='Global shutter, 1920 × 1200 at 60 fps (“2.3 MP @ 60 fps”; Full HD 65 fps, HD 120 fps). '
                      'Linear sensor (71.4 dB), not HDR.'),
        _module('sturdecam29', 'STURDeCAM29', 'OmniVision OV2311', (1600, 1300), 3.0, '1/2.9"', 60.0,
                shutter='global', dr_db=68.0, temp=(-40, 85), size=(59.02, 52.52, 42.75), weight=131.0,
                power=1.7, ip='IP65', chroma='monochrome',
                urls=(_E + '/gmsl-cameras/2mp-ov2311-monochrome-global-shutter-gmsl2-camera.asp',),
                extra='Monochrome global shutter with 940 nm IR LEDs. Size 52.52 (L) × 59.02 (W) × 42.75 (D) mm, '
                      'taken as 59.02 wide, 52.52 high and 42.75 long. Cable up to 15 m. Hardware frame sync is not '
                      'stated.'),
        _module('sturdecam57', 'STURDeCAM57', 'OmniVision OX05B1S', (2592, 1944), 2.2, '1/2.5"', 50.0,
                shutter='global', dr_db=71.0, temp=(-40, 85), size=(70.0, 35.0, 30.0), weight=100.0,
                cables=(3.0, 15.0), power=3.42, ip='IP67', chroma='RGB-IR',
                urls=(_E + '/gmsl-cameras/5mp-ox05b1s-rgb-ir-global-shutter-gmsl2-camera.asp',
                      k + '5mp-ox05b1s-rgb-ir-global-shutter-gmsl2-camera.asp'),
                extra='In-cabin (driver and occupant monitoring) camera: 50 fps of interleaved RGB and IR frames '
                      '(25 + 25), 940 nm IR LED for 0.5–2 m. Size 70 × 35 × 30 mm, taken as 70 wide, 35 high, 30 '
                      'long. Hardware frame sync is not stated.'),
        _module('sturdecam81', 'STURDeCAM81', 'onsemi AR0821', (3840, 2160), 2.1, '1/1.7"', 30.0,
                shutter='rolling', dr_db=100.0, temp=(-30, 70), size=(42.0, 42.0, 0.0), power=2.496, sync=True,
                ip='IP67', urls=(k + '4k-ar0821-IP67-hdr-gmsl2-multi-camera.asp',),
                extra='8 MP at 30 fps in HDR and linear modes (3840 × 2160 used; active area 3848 × 2168). '
                      'Enclosure 42 × 42 mm (length not given). Cable 15 m. Frame sync and trigger controls.'),
        _module('sturdecam88', 'STURDeCAM88', 'OmniVision OX08B40', (3840, 2160), 2.1, '1/1.73"', 30.0,
                shutter='rolling', dr_db=140.0, temp=(-40, 80), size=(42.0, 42.0, 0.0), weight=80.0,
                cables=(3.0, 15.0), power=2.5, sync=True, ip='IP67',
                urls=(_E + '/gmsl-cameras/8mp-ox08b40-ip67-gmsl2-140db-hdr-camera.asp',
                      k + '4k-omnivision-ox08b40-140db-hdr-gmsl2-camera.asp'),
                extra='4K front-view camera. 2.5 W typical (Orin kit page). Weight 80 g with the narrow lens. '
                      'Enclosure 42 × 42 mm (length not given). Sync of up to 8 cameras.'),
        _module('sturdecam20', 'STURDeCAM20', 'onsemi AR0230AT', (1920, 1080), 3.0, '1/2.7"', 30.0,
                shutter='rolling', dr_db=96.0, temp=(-40, 85), size=(42.0, 42.0, 0.0), power=2.196, sync=True,
                ip='IP67',
                urls=(_E + '/camera-modules/gmsl2-surround-view-camera.asp',
                      k + 'ar0230-ip67-gmsl2-hdr-camera-for-agx-xavier.asp'),
                extra='Older model (its page is live but not in e-con\'s current GMSL menu). Full HD at 30 fps; the '
                      'stated FOV is taken as the Full HD image\'s. Power 0.78–2.196 W (the maximum is used). '
                      'Aluminium enclosure 42 × 42 mm (length not given).'),
        _module('sturdecam21', 'STURDeCAM21', 'onsemi AR0233AT', (1920, 1080), 3.0, '1/2.5"', 30.0,
                shutter='rolling', dr_db=120.0, temp=(-40, 85), size=(42.0, 42.0, 0.0), power=2.196, sync=True,
                ip='IP67',
                urls=(_E + '/camera-modules/ip67-ar0233-gmsl2-hdr-camera.asp',
                      k + 'ar0233-ip67-full-hd-gmsl2-hdr-camera.asp'),
                extra='Older model (its page is live but not in e-con\'s current GMSL menu). Full HD at 30 fps from '
                      'the 2048 × 1280 sensor; the stated FOV is taken as the Full HD image\'s. Up to 120 dB (module '
                      'page; the Orin kit page says 140 dB). Power 0.78–2.196 W (the maximum is used). Enclosure '
                      '42 × 42 mm (length not given).'),
        # ---- NileCAM: board cameras (no enclosure)
        _module('nilecam81', 'NileCAM81', 'onsemi AR0821', (3840, 2160), 2.1, '1/1.7"', 16.0,
                shutter='rolling', dr_db=100.0, temp=(-30, 70), size=(30.0, 30.0, 0.0), cables=(3.0, 15.0),
                power=2.496,
                urls=(_E + '/camera-modules/ar0821-4k-hdr-gmsl2-camera-module.asp',
                      k + 'ar0821-4k-hdr-gmsl2-camera.asp'),
                extra='Board camera, 30 × 30 mm, no enclosure (IP67 enclosure optional). 8 MP at 16 fps (3840 × 2160 '
                      'used); Full HD at 60 fps.'),
        _module('nilecam85', 'NileCAM85', 'Sony IMX715', (3840, 2160), 1.45, '1/2.8"', 30.0,
                shutter='rolling', dr_db=0.0, temp=(-30, 70), size=(30.0, 30.0, 0.0), cables=(3.0, 15.0),
                urls=(_E + '/camera-modules/sony-starvis-imx715-low-light-gmsl2-camera-module.asp',),
                extra='Board camera, 30 × 30 mm, no enclosure. Low-light 4K at 30 fps, RAW 10/12-bit.'),
        _module('nilecam130', 'NileCAM130', 'onsemi AR1335', (4208, 3120), 1.1, '1/3.2"', 10.0,
                shutter='rolling', dr_db=69.0, temp=(-30, 70), size=(30.0, 30.0, 0.0), power=1.1,
                urls=(_E + '/camera-modules/ar1335-13mp-gmsl2-camera-module.asp',),
                extra='Board camera, 30 × 30 mm, no enclosure. 13 MP at 10 fps (4K at 15, HD at 72).'),
        _module('nilecam210', 'NileCAM210', 'Sony IMX462', (1920, 1080), 2.9, '1/2.8"', 60.0,
                shutter='rolling', dr_db=0.0, temp=(-30, 70), size=(30.0, 30.0, 0.0), cables=(3.0, 15.0),
                urls=(_E + '/camera-modules/sony-starvis-imx462-low-light-gmsl2-camera-module.asp',),
                extra='Board camera, 30 × 30 mm, no enclosure. Low-light Full HD at 60 fps, RAW 12-bit. e-con states '
                      '7.488 W without saying for how many cameras: not used (power not given).'),
        # ---- FPD-Link
        _module('neducam25', 'NeduCAM25', 'onsemi AR0234CS', (1920, 1200), 3.0, '1/2.6"', 60.0,
                shutter='global', dr_db=71.4, temp=(-20, 70), size=(30.0, 30.0, 0.0), sync=True,
                interface='FPD-Link III',
                urls=(_E + '/fpd-link-iii-camera-modules/full-hd-ar0234-global-shutter-camera.asp',
                      k + 'ar0234-global-shutter-fpd-link-iii-synchronized-multi-camera.asp'),
                extra='FPD-Link III board camera, 30 × 30 mm (IP67 enclosure optional). 2 MP (1920 × 1200) at 60 '
                      'fps. The link budget checks it against the same per-link figure as a GMSL2 camera.'),
        _module('nedu4cam57', 'Nedu4CAM57', 'OmniVision OX05B1S', (2592, 1944), 2.2, '1/2.53"', 50.0,
                shutter='global', dr_db=71.0, temp=(-40, 85), size=(70.0, 35.0, 30.0), weight=100.0, cables=(10.0,),
                power=3.48, ip='IP67', chroma='RGB-IR', interface='FPD-Link IV',
                urls=(_E + '/fpd-link-iv-camera-modules/5mp-ox05b1s-global-shutter-rgb-ir-camera.asp',),
                extra='FPD-Link IV twin of STURDeCAM57 (in-cabin RGB-IR, 25 + 25 fps), cable up to 10 m. Size 70 × 35 '
                      '× 30 mm, taken as 70 wide, 35 high, 30 long. Hardware frame sync is not stated.'),
    ]


def half_sizes(module: Module) -> tuple:
    """Sensor half width, half height and half diagonal, mm."""
    return module.sensor_w_mm / 2, module.sensor_h_mm / 2, math.hypot(module.sensor_w_mm, module.sensor_h_mm) / 2


def _fov_label(h):
    return 'fisheye' if h >= 180 else 'wide' if h >= 100 else 'standard' if h >= 70 else 'narrow'


def _lens(lid, name, m: Module, sheet, *, where='', focus=0.0, focus_far=math.inf, circle='', extra='',
          base=None, d_at_least=False) -> Lens:
    """A built-in lens from e-con's stated FOV (H, V, D; 0 = not stated). circle: why the stated D is the image
    circle's field rather than the sensor diagonal's ('' = it is the diagonal's): it then limits the image
    (max_half_deg) instead of being fitted, and the sensor corners beyond it are dark. d_at_least: e-con gives D as
    'at least'."""
    hs = half_sizes(m)
    fit = (sheet[0], sheet[1], 0.0) if circle else tuple(sheet)
    b, f, _ = best_base(fit, hs) if base is None else (base, focal_from_fov(fit, hs, base), 0.0)
    f = round(f, 3)                                   # µm: well inside what the FOV figures pin down
    kb = fit_sheet(f, fit, hs, base=b)
    shown = ' × '.join(f'{"≥ " if a == "D" and d_at_least else ""}{v:g}° {a}' for v, a in zip(sheet, 'HVD') if v)
    note = (f'FOV {shown} as e-con states it{" (" + where + ")" if where else ""} on {m.name}\'s page (read '
            f'{CHECKED}). e-con does not state the focal length: {f:.3g} mm is derived from that FOV with '
            f'{_an(b)} {b} base projection, and the distortion terms are fitted to reproduce it. Aperture not stated.')
    return Lens(id=lid, name=name, label=_fov_label(sheet[0]), module_id=m.id, focal_mm=f, f_number=0.0,
                focus_near_m=focus, focus_far_m=focus_far, projection='kannala_brandt', kb=kb,
                sheet_fov=tuple(float(v) for v in sheet), max_half_deg=sheet[2] / 2 if circle else 0.0,
                builtin=True, note=' '.join(t for t in (note, circle, extra) if t))


def _an(word):
    return 'an' if word[:1] in 'aeiou' else 'a'


def _circle_reason(m: Module, h_deg, d_deg) -> str:
    """Why a stated D cannot be the sensor diagonal's field for a lens h_deg wide on this sensor."""
    hw, _, hd = half_sizes(m)
    out, deg = 100 * (hd / hw - 1), (d_deg - h_deg) / 2
    return (f'e-con\'s {d_deg:g}° D cannot be the sensor diagonal\'s field for a lens {h_deg:g}° wide (the corners '
            f'lie {out:.0f} % further out than the side edges but only {deg:g}° further off axis), so it is taken '
            f'as the image circle: the corners beyond {d_deg / 2:g}° off axis are dark.')


def builtin_lenses() -> list[Lens]:
    """The lenses e-con states a field of view for, each on its module (h05r2 first: the default lens)."""
    M = {m.id: m for m in builtin_modules()}
    hs = half_sizes(M['sturdecam31'])
    sheets = ('as e-con\'s STURDeCAM31 Lens Datasheet (Rev 1.2) and Module Datasheet (Rev 1.4) state them; neither '
              'gives a distortion table: the distortion terms are fitted to reproduce the stated FOV')
    wide = Lens(id='h05r2', name='H05R2', label='wide', module_id='sturdecam31', focal_mm=2.62, f_number=2.0,
                focus_near_m=0.35, projection='kannala_brandt', sheet_fov=(128.7, 98.7, 170.1), builtin=True,
                note=f'FOV 128.7° H × 98.7° V × 170.1° D as e-con states it for the wide lens (on request) on '
                     f'STURDeCAM31\'s pages (read {CHECKED}). Focal length 2.62 mm, aperture F2.0 and focus 0.35 m '
                     f'to infinity {sheets}.')
    wide.kb = fit_kb_to_fov(2.62, wide.sheet_fov, hs, base='equisolid')
    narrow = Lens(id='h03r2', name='H03R2', label='narrow', module_id='sturdecam31', focal_mm=5.5, f_number=2.0,
                  focus_near_m=1.28, projection='kannala_brandt', sheet_fov=(54.04, 42.86, 70.17), builtin=True,
                  note=f'e-con\'s default STURDeCAM31 lens (70° DFOV): FOV 54.04° H × 42.86° V × 70.17° D as e-con '
                       f'states it (read {CHECKED}). Focal length 5.5 mm, aperture F2.0 and focus 1.28 m to infinity '
                       f'{sheets} (refitted in 2026-10: within 0.5°).')
    narrow.kb = fit_sheet(5.5, narrow.sheet_fov, hs, base='equisolid')
    return [
        wide, narrow,
        _lens('sturdecam31fsh_heated', 'Heated fisheye', M['sturdecam31fsh'], (196.0, 155.0, 210.0),
              where='default heated lens', focus=0.5, circle=_circle_reason(M['sturdecam31fsh'], 196.0, 210.0)),
        _lens('sturdecam34_h01r1', 'H01R1', M['sturdecam34'], (64.62, 51.28, 83.5), where='kit lens H01R1',
              focus=1.7),
        _lens('sturdecam39_option1', 'Lens option 1', M['sturdecam39'], (196.7, 155.1, 200.0), where='default lens',
              focus=0.5, d_at_least=True,
              circle='The model limits the image to a 200° circle, so the corners may see a little more than it '
                     'counts.'),
        _lens('sturdecam39_option2', 'Lens option 2', M['sturdecam39'], (121.0, 96.7, 164.5), where='lens option 2'),
        _lens('sturdecam25_std', '133° DFOV', M['sturdecam25'], (111.1, 68.3, 132.8), where='lens provided by e-con',
              focus=1.0),
        _lens('sturdecam29_std', '131° DFOV', M['sturdecam29'], (93.0, 74.0, 131.0), where='standard lens'),
        _lens('sturdecam57_std', '148° DFOV', M['sturdecam57'], (114.76, 85.92, 148.32), where='kit lens',
              focus=0.5, focus_far=2.0, extra='Focus 0.5 m to 2 m (driver and occupant monitoring).'),
        _lens('sturdecam81_std', '123° DFOV', M['sturdecam81'], (106.3, 57.1, 122.6), where='lens provided by e-con'),
        _lens('sturdecam88_h02r1', 'H02R1', M['sturdecam88'], (68.23, 38.83, 80.22), where='narrow kit lens H02R1'),
        _lens('sturdecam88_h04r1', 'H04R1', M['sturdecam88'], (120.12, 67.04, 137.36), where='wide kit lens H04R1'),
        _lens('sturdecam20_std', '117° DFOV', M['sturdecam20'], (98.5, 53.5, 116.5), where='lens provided by e-con',
              focus=1.0),
        _lens('sturdecam21_std', '114° DFOV', M['sturdecam21'], (98.22, 50.34, 113.83),
              where='lens provided by e-con', focus=1.0),
        _lens('nilecam81_std', '120° DFOV', M['nilecam81'], (104.8, 58.7, 120.0), where='lens provided by e-con'),
        _lens('nilecam85_std', '131° DFOV', M['nilecam85'], (113.0, 61.0, 131.0), where='lens provided by e-con'),
        _lens('nilecam130_std', '110° DFOV', M['nilecam130'], (86.0, 64.0, 110.0), where='lens provided by e-con'),
        _lens('nilecam210_std', '132° DFOV', M['nilecam210'], (113.0, 61.0, 132.0), where='lens provided by e-con'),
        _lens('neducam25_std', '128° DFOV', M['neducam25'], (104.6, 61.6, 128.2), where='lens provided by e-con'),
        _lens('nedu4cam57_std', '148° DFOV', M['nedu4cam57'], (114.76, 85.92, 148.32), where='standard lens',
              focus=0.5, focus_far=2.0, extra='Focus 0.5 m to 2 m (driver and occupant monitoring).'),
    ]


def builtin_module(mid: str) -> Module | None:
    return next((m for m in builtin_modules() if m.id == mid), None)


def builtin_lens(lid: str) -> Lens | None:
    return next((l for l in builtin_lenses() if l.id == lid), None)


def merge_builtins(modules: dict, lenses: dict) -> tuple[dict, dict]:
    """A project's library with every built-in entry it lacks added after its own (a saved entry always wins: the
    user may have edited it). Lets projects saved before a library entry existed offer it."""
    mods = dict(modules)
    for m in builtin_modules():
        mods.setdefault(m.id, m)
    lns = dict(lenses)
    for l in builtin_lenses():
        lns.setdefault(l.id, l)
    return mods, lns
