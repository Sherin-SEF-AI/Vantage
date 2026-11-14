"""Relighting re-rendered frames to conditions that were never driven: a wet road with glare, night under the
vehicle's own headlamps, snow (Simulate ▸ Log replay ▸ Neural replay ▸ Relight). The re-shot frames of the log
replay (core.splat_reproject, core.redrive, core.neural_replay) are relit before they are written or scored, so a
proposed rig can be judged under them; every relit frame and figure says relit, not recorded.

Two methods
-----------
* 'physical' (always available, deterministic): from each re-rendered pixel's colour, range and its dominant
  Gaussian's normal (the scene's geometry), in linear light (the sRGB transfer function, IEC 61966-2-1):
  - wet: horizontal surfaces (normal within 30° of up) darkened to wet_darkening of their reflectance (a water film
    darkens a rough surface by total internal reflection: Lekner and Dorf, "Why some things are darker when wet",
    Applied Optics 27(7), 1988; 0.6 is a typical asphalt figure, not measured here), vertical ones to 0.85 of it, plus a
    water film's mirror reflection: Schlick's Fresnel term (n = 1.333, R₀ = 0.020) times the sky the reflected ray
    sees (the scene's sky texture) and a specular lobe of the sun (σ 3° of a rippled film) at sun_az_deg, sun_el_deg
    (relative to the vehicle's heading);
  - night: the scene lit by the ambient night_ambient_lux and the vehicle's two headlamps, core.weather's UN R112
    class B shaped low beam (headlamp_cd, its lamp positions; core.thermal_ir validates the same pattern against the
    regulation's test points), each surface's illuminance E = Σ I·cos / d² (no shadows), its reflectance from the
    daylight render; the camera's night exposure maps a surface lit by exposure_lux to its daylight level; the sky
    dark (0.01 of its daylight level); emissive Gaussians (traffic lights' lamps) keep their light; read noise added
    (seeded, deterministic per camera and instant);
  - snow: surfaces facing up (normal within 35° of up) covered towards a fresh snow's reflectance 0.85 (Warren, "Optical
    properties of snow", Rev. Geophys. 20(1), 1982: visible albedo 0.8–0.9), falling snow as extinction from
    core.weather's ITU-R P.1814 snow coefficients at snow_rate_mm_h (Koschmieder's airlight blend over each pixel's
    range), and flakes (seeded) drawn in front of the scene where nearer than its surface.
* 'diffusion' (opt-in download, needs the diffusers package and a GPU to be practical): InstructPix2Pix (Brooks,
  Holynski and Efros, "InstructPix2Pix: Learning to Follow Image Editing Instructions", CVPR 2023, arXiv:2211.09800;
  weights timbrooks/instruct-pix2pix, MIT licence per the model card, read 2026-11-11) edits each frame with an
  instruction per condition. It has no notion of the scene's geometry or the headlamps' photometry: it changes the
  look, not the light physically; seeded per frame. Without diffusers or the files, the physical method is used and
  the result says so.

Limits: no shadows from the headlamps or the sun, no reflections of the scene in the water (the sky only), no rain
streaks or drops on the lens, no snow on the moving objects' sides; the night exposure and noise are typical, not a
sensor's. A relit frame is a what-if, not a recording of that weather.
"""
from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass

import numpy as np

SCHEMA = 'vantage.relight/1'
CONDITIONS = ('none', 'wet', 'night', 'snow')
METHODS = ('physical', 'diffusion')
LIMITS = {'strength': (0.0, 1.0), 'sun_az_deg': (-180.0, 180.0), 'sun_el_deg': (0.0, 90.0),
          'wet_darkening': (0.2, 1.0), 'night_ambient_lux': (0.0, 100.0), 'exposure_lux': (1.0, 1000.0),
          'snow_rate_mm_h': (0.0, 20.0), 'read_noise': (0.0, 0.1), 'seed': (0, 2 ** 31 - 1),
          'diffusion_steps': (1, 100), 'image_guidance': (0.5, 5.0), 'text_guidance': (1.0, 20.0),
          'diffusion_px': (256, 1024), 'lamp_height_m': (0.3, 1.5)}
INTS = ('seed', 'diffusion_steps', 'diffusion_px')
PROMPTS = {'wet': 'make the road wet after heavy rain, with reflections on the asphalt',
           'night': 'make it night, lit only by street lights and car headlights',
           'snow': 'make it a snowy winter day with snow on the ground'}
R0_WATER = ((1.333 - 1) / (1.333 + 1)) ** 2
SNOW_ALBEDO = 0.85
SKY_NIGHT = 0.01


def _f(v, default):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) else default


@dataclass
class RelightSettings:
    condition: str = 'none'
    method: str = 'physical'
    strength: float = 1.0
    sun_az_deg: float = 30.0           # the sun relative to the vehicle's heading (left positive) and its elevation
    sun_el_deg: float = 25.0
    wet_darkening: float = 0.6
    night_ambient_lux: float = 1.0     # moonlit street with distant lighting (0.1: an unlit road under the moon)
    exposure_lux: float = 30.0         # the night exposure renders a surface lit by this as it looked by day
    lamp_height_m: float = 0.7
    snow_rate_mm_h: float = 2.0
    read_noise: float = 0.01           # night: read noise, share of full scale
    seed: int = 0
    diffusion_steps: int = 20
    image_guidance: float = 1.5
    text_guidance: float = 7.5
    diffusion_px: int = 512

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d) -> 'RelightSettings':
        s = RelightSettings()
        d = d if isinstance(d, dict) else {}
        if d.get('condition') in CONDITIONS:
            s.condition = d['condition']
        if d.get('method') in METHODS:
            s.method = d['method']
        for k, (lo, hi) in LIMITS.items():
            if k not in d or isinstance(d[k], bool):
                continue
            v = _f(d[k], None)
            if v is None:
                continue
            v = min(max(v, lo), hi)
            setattr(s, k, int(round(v)) if k in INTS else float(v))
        return s

    def label(self) -> str:
        if self.condition == 'none':
            return 'as recorded'
        return f'relit: {self.condition} ({self.method})'


# =========================================================================== colour
def to_linear(c):
    c = np.asarray(c, float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def to_srgb(c):
    c = np.clip(np.asarray(c, float), 0.0, None)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * np.power(c, 1 / 2.4) - 0.055)


def schlick(cos_i, r0=R0_WATER):
    return r0 + (1 - r0) * (1 - np.clip(cos_i, 0, 1)) ** 5


def _sun_dir(st: RelightSettings) -> np.ndarray:
    a, e = math.radians(st.sun_az_deg), math.radians(st.sun_el_deg)
    return np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])


def snow_extinction(rate_mm_h: float, wavelength_nm: float = 550.0) -> float:
    """σ (1/m) of falling dry snow: ITU-R P.1814 Table 3, a·S^b dB/km with a = 5.42e-5·λ + 5.50, b = 1.38 (λ nm,
    S mm/h), as core.weather uses it."""
    if rate_mm_h <= 0:
        return 0.0
    db_km = (5.42e-5 * wavelength_nm + 5.50) * rate_mm_h ** 1.38
    return db_km / (10 * math.log10(math.e)) / 1000.0


# =========================================================================== the physical method
def relight_physical(rgb, depth, normals, dirs, st: RelightSettings, sky_reflect=None, emissive=None,
                     points=None, rng_key=0) -> np.ndarray:
    """One frame relit (module docstring). rgb (h, w, 3) uint8 or float [0, 1]; depth (h, w) m along the ray (0 or
    NaN: no surface, the sky); normals (h, w, 3) unit, vehicle frame (NaN where none); dirs (h, w, 3) unit view rays,
    vehicle frame (from the camera); sky_reflect(dirs (k, 3)) -> linear RGB (k, 3) of the sky (wet); emissive (h, w)
    bool (pixels that emit: kept at night); points (h, w, 3) vehicle-frame surface points (night: the headlamps).
    Returns uint8 RGB. Deterministic for a given rng_key."""
    img = np.asarray(rgb)
    f = img.astype(float) / 255.0 if img.dtype == np.uint8 else img.astype(float)
    lin = to_linear(f)
    h, w = lin.shape[:2]
    d = np.nan_to_num(np.asarray(depth, float), nan=0.0)
    surf = d > 0.05
    n = np.asarray(normals, float) if normals is not None else np.full((h, w, 3), np.nan)
    has_n = surf & np.isfinite(n).all(-1)
    nz = np.where(has_n, n[..., 2], 0.0)
    D = np.asarray(dirs, float)
    em = np.zeros((h, w), bool) if emissive is None else np.asarray(emissive, bool)
    k = float(st.strength)
    out = lin.copy()
    rng = np.random.default_rng([int(st.seed), int(rng_key) & 0x7FFFFFFF])
    if st.condition == 'wet':
        horiz = has_n & (nz > math.cos(math.radians(30.0)))
        vert = has_n & ~horiz
        dark = np.where(horiz, st.wet_darkening, np.where(vert, 0.85, 1.0))
        out = lin * (1 - k + k * dark[..., None])
        if horiz.any():
            nn = n[horiz]
            v = D[horiz]
            cos_i = np.abs((v * nn).sum(1))
            refl = v - 2 * (v * nn).sum(1, keepdims=True) * nn
            Fr = schlick(cos_i)[:, None]
            sky = sky_reflect(refl) if sky_reflect is not None else np.full((len(refl), 3), 0.6)
            sun = _sun_dir(st)
            ang = np.degrees(np.arccos(np.clip(refl @ sun, -1, 1)))
            lobe = 12.0 * np.exp(-0.5 * (ang / 3.0) ** 2)[:, None]
            out[horiz] = out[horiz] + k * Fr * (np.asarray(sky, float) + lobe)
    elif st.condition == 'night':
        E = np.full((h, w), float(st.night_ambient_lux))
        if points is not None:
            from . import weather as WX
            P = np.asarray(points, float)
            ws = WX.WeatherSettings()
            lamps = WX.lamp_positions(None, ws)
            lamps[:, 2] = st.lamp_height_m
            for L in lamps:
                dv = P - L
                r = np.maximum(np.linalg.norm(dv, axis=-1), 1e-3)
                hh = np.degrees(np.arctan2(dv[..., 1], dv[..., 0]))
                vv = np.degrees(np.arcsin(np.clip(dv[..., 2] / r, -1, 1)))
                I = WX.headlamp_cd(hh, vv, 'low', ws.traffic, ws.low_beam_cd, ws.high_beam_cd)
                cos = np.where(has_n, np.maximum(-np.nan_to_num((n * dv).sum(-1)) / r, 0.0), 1.0)
                E = E + np.where(surf, I * cos / (r * r), 0.0)
        scale = np.where(surf, E / st.exposure_lux, SKY_NIGHT)
        night = lin * scale[..., None]
        night = night + rng.normal(0.0, st.read_noise, night.shape)
        night[em] = lin[em]
        out = (1 - k) * lin + k * night
    elif st.condition == 'snow':
        up = has_n & (nz > math.cos(math.radians(35.0)))
        cover = np.clip((nz - math.cos(math.radians(35.0))) / (1 - math.cos(math.radians(35.0))), 0, 1)
        cover = np.where(up, 0.4 + 0.6 * cover, 0.0) * k
        out = lin * (1 - cover[..., None]) + to_linear(SNOW_ALBEDO) * cover[..., None]
        sig = snow_extinction(st.snow_rate_mm_h)
        air = to_linear(0.78)
        dd = np.where(surf, d, 300.0)
        tr = np.exp(-sig * dd)[..., None]
        out = out * (1 - k + k * tr) + k * air * (1 - tr)
        nfl = int(st.snow_rate_mm_h * 120 * k * (h * w) / (640 * 480))
        if nfl > 0:
            fx = rng.uniform(0, w, nfl)
            fy = rng.uniform(0, h, nfl)
            fd = rng.uniform(0.8, 15.0, nfl)
            fr = np.clip(3.0 / fd * (w / 640), 0.6, 4.0)
            for x, y, dist, r in zip(fx, fy, fd, fr):
                xi, yi = int(x), int(y)
                if dd[yi, xi] <= dist:
                    continue
                x0, x1 = max(int(x - 2 * r), 0), min(int(x + 2 * r) + 1, w)
                y0, y1 = max(int(y - 2 * r), 0), min(int(y + 2 * r) + 1, h)
                yy, xx = np.mgrid[y0:y1, x0:x1]
                a = 0.85 * np.exp(-0.5 * (((xx + 0.5 - x) ** 2 + (yy + 0.5 - y) ** 2) / (r * r)))
                out[y0:y1, x0:x1] = out[y0:y1, x0:x1] * (1 - a[..., None]) + to_linear(0.95) * a[..., None]
    return (np.clip(to_srgb(out), 0, 1) * 255 + 0.5).astype(np.uint8)


# =========================================================================== the diffusion method
def _ip2p_spec():
    from . import foundation_models as FM
    big = (('text_encoder/model.fp16.safetensors', 246144864,
            '77795e2023adcf39bc29a884661950380bd093cf0750a966d473d1718dc9ef4e'),
           ('unet/diffusion_pytorch_model.fp16.safetensors', 1719148344,
            '0d6bbc0a95dd125196d327a660b43d24c56f433eb30d2776f1327fb86bd38f78'),
           ('vae/diffusion_pytorch_model.fp16.safetensors', 167335342,
            '4fbcf0ebe55a0984f5a5e00d8c4521d52359af7229bb4d81890039d2aa16dd7c'))
    small = (('model_index.json', 616, '4ccaf9b31f00937caf1c61578c497b53d63131a830fc08f812b2b55d86c9855e'),
             ('scheduler/scheduler_config.json', 569,
              '81fe3f35ef35556fed140f94a61fbefcb3982cbd771686014c9be00435c3bbcc'),
             ('text_encoder/config.json', 617, 'c90121fab17b226c98ae813c1778add7edbdcbdcf7ceaff7aa197fdf8c8a6b3a'),
             ('tokenizer/merges.txt', 524619, '9fd691f7c8039210e0fced15865466c65820d09b63988b0174bfe25de299051a'),
             ('tokenizer/special_tokens_map.json', 472,
              'c4864a9376a8401918425bed71fc14fc0e81f9b59ec45c1cf96cccb2df508eac'),
             ('tokenizer/tokenizer_config.json', 806,
              '00439066fcba73de57644cf41e4e3b9f2dbb09d7f3fc2005898ba52399045882'),
             ('tokenizer/vocab.json', 1059962, 'e089ad92ba36837a0d31433e555c8f45fe601ab5c221d4f607ded32d9f7a4349'),
             ('unet/config.json', 1021, 'f6c4ea486da4d813c389de1fc405fcb64139ca18f0729733dc177dad17854347'),
             ('vae/config.json', 553, '46621c584ebe6666853057a58a4a60047f18f8b83a16cf4603e8359a91e485f6'))
    return FM.FoundationSpec(
        'instruct_pix2pix', 'InstructPix2Pix (fp16)', 'relight', 'instruct_pix2pix', 'timbrooks/instruct-pix2pix',
        '31519b5cb02a7fd89b906d88731cd4d6a7bbf88d', FM._f(*(small + big)), 'MIT',
        'https://huggingface.co/timbrooks/instruct-pix2pix', True,
        'MIT (the model card\'s licence field, read 2026-11-11; fine-tuned from Stable Diffusion 1.5)',
        'Brooks, Holynski and Efros: InstructPix2Pix: Learning to Follow Image Editing Instructions, CVPR 2023, '
        'arXiv:2211.09800', 1066.0, note='image editing by instruction; the fp16 weights, about 2.1 GB')


def diffusion_missing() -> str | None:
    try:
        import diffusers  # noqa: F401
        import torch  # noqa: F401
    except ImportError:
        return ('Diffusion relighting needs PyTorch and Hugging Face diffusers ("pip install torch diffusers"), which '
                'the downloads do not include; the physical relighting is used instead.')
    return None


def diffusion_status(folder=None, verify=True) -> dict:
    from . import foundation_models as FM
    return FM.status(_ip2p_spec(), folder, verify=verify)


def download_diffusion(progress=None, cancel=None, folder=None, fetch=None):
    """Download InstructPix2Pix's fp16 files from the pinned revision (sizes and SHA-256 checked; MIT)."""
    from . import foundation_models as FM
    sp = _ip2p_spec()
    d = FM.model_dir(sp, folder)
    for f in sp.files:
        os.makedirs(os.path.dirname(os.path.join(d, f.name)), exist_ok=True)
    return FM.download(sp, progress, cancel, folder, fetch=fetch)


class DiffusionRelighter:
    """InstructPix2Pix through diffusers (fp16 on CUDA, float32 on the CPU), one instruction per condition."""

    def __init__(self, folder=None, device='auto', mem_gb=6.0):
        why = diffusion_missing()
        if why:
            raise ValueError(why)
        from . import foundation_models as FM
        path = FM._load_check(_ip2p_spec(), folder)
        import torch
        from diffusers import StableDiffusionInstructPix2PixPipeline
        self.dev, _ = FM._torch_device(device, mem_gb)
        dt = torch.float16 if self.dev.type == 'cuda' else torch.float32
        self.pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            path, torch_dtype=dt, variant='fp16', safety_checker=None, feature_extractor=None,
            requires_safety_checker=False).to(self.dev)
        self.pipe.set_progress_bar_config(disable=True)
        self.torch = torch

    def __call__(self, rgb: np.ndarray, st: RelightSettings, key: int = 0) -> np.ndarray:
        import cv2
        from PIL import Image
        h, w = rgb.shape[:2]
        s = st.diffusion_px / max(h, w)
        W, H = max(64, int(round(w * s / 8)) * 8), max(64, int(round(h * s / 8)) * 8)
        im = Image.fromarray(cv2.resize(rgb, (W, H), interpolation=cv2.INTER_AREA))
        g = self.torch.Generator(device=self.dev).manual_seed(int(st.seed) * 1000003 + int(key) % 1000003)
        out = self.pipe(PROMPTS[st.condition], image=im, num_inference_steps=st.diffusion_steps,
                        image_guidance_scale=st.image_guidance, guidance_scale=st.text_guidance, generator=g).images[0]
        res = cv2.resize(np.asarray(out), (w, h), interpolation=cv2.INTER_CUBIC)
        if st.strength < 1.0:
            res = (st.strength * res.astype(float) + (1 - st.strength) * rgb.astype(float) + 0.5).astype(np.uint8)
        return res


_DIFF = {}


def _diffusion(device='auto', mem_gb=6.0):
    k = (device, mem_gb)
    if k not in _DIFF:
        _DIFF.clear()
        _DIFF[k] = DiffusionRelighter(device=device, mem_gb=mem_gb)
    return _DIFF[k]


# =========================================================================== the hook for re-rendered frames
def depth_normals(P, depth, smooth: int = 5, max_jump: float = 0.05):
    """Unit surface normals (h, w, 3) from per-pixel surface points P (h, w, 3) and their range (h, w): the cross
    product of the point grid's two image-axis differences, averaged over smooth × smooth pixels, facing the camera
    (the origin of P's rays); NaN where there is no surface or at an edge, where the range's second difference along
    a row or column exceeds max_jump of the range."""
    import cv2
    P = np.asarray(P, float)
    d = np.nan_to_num(np.asarray(depth, float), nan=0.0)
    surf = d > 0.05
    gx = np.zeros_like(P)
    gy = np.zeros_like(P)
    gx[:, 1:-1] = 0.5 * (P[:, 2:] - P[:, :-2])
    gy[1:-1] = 0.5 * (P[2:] - P[:-2])
    # an edge: the range is not locally linear (its second difference, a share of the range, over max_jump); a
    # plane seen at a grazing angle changes its range fast but smoothly, and keeps its normals
    jump = np.zeros(d.shape, bool)
    lim = max_jump * np.maximum(d, 1e-3)
    jump[:, 1:-1] |= (np.abs(d[:, 2:] - 2 * d[:, 1:-1] + d[:, :-2]) > lim[:, 1:-1]) | ~surf[:, 2:] | ~surf[:, :-2]
    jump[1:-1] |= (np.abs(d[2:] - 2 * d[1:-1] + d[:-2]) > lim[1:-1]) | ~surf[2:] | ~surf[:-2]
    n = np.cross(gx, gy)
    ok = surf & ~jump & (np.linalg.norm(n, axis=-1) > 1e-12)
    n = np.where(ok[..., None], n / np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-12), 0.0)
    if smooth > 1:
        w = cv2.boxFilter(ok.astype(np.float32), -1, (smooth, smooth), normalize=False)
        n = np.stack([cv2.boxFilter(n[..., k].astype(np.float32), -1, (smooth, smooth), normalize=False)
                      for k in range(3)], -1) / np.maximum(w, 1e-6)[..., None]
        ok = ok | ((w > 0) & surf)
    nn = np.linalg.norm(n, axis=-1, keepdims=True)
    n = n / np.maximum(nn, 1e-12)
    flip = (n * P).sum(-1) > 0
    n[flip] *= -1
    n[~ok | (nn[..., 0] < 1e-6)] = np.nan
    return n


def frame_geometry(ts, cam, t: float, idx, rng):
    """For a rendered camera image: (view rays (h, w, 3), normals (h, w, 3) NaN where none, surface points (h, w, 3)),
    all in the vehicle frame at t. The normals come from the rendered range image (depth_normals: smooth over a
    surface, unlike the dominant Gaussian's axis, which a near-isotropic Gaussian does not define)."""
    m = cam.model
    D, _ = _rays(m)
    Dv = D @ np.asarray(cam.R, float).T
    P = np.asarray(cam.t, float) + Dv * np.asarray(rng, float)[..., None]
    rel = P - np.asarray(cam.t, float)
    N = depth_normals(rel, rng)
    return Dv, N, P


_RAYS = {}


def _rays(model):
    key = (model.width, model.height, round(model.fx, 5), round(model.cx, 5), model.projection, model.kb)
    if key not in _RAYS:
        if len(_RAYS) > 32:
            _RAYS.clear()
        _RAYS[key] = model.rays()
    return _RAYS[key]


def relighter(settings: dict, ts, emissive=None):
    """A callable (rgb uint8, range (h, w), dominant Gaussian (h, w), CamSpec, t) -> rgb uint8 relighting core.
    splat_reproject's re-rendered camera images (its _render_rig_frame hook), or None for 'none'. emissive: (N,) bool
    of the Gaussians that emit (kept at night)."""
    st = RelightSettings.from_dict(settings)
    if st.condition == 'none':
        return None
    em = None if emissive is None else np.asarray(emissive, bool)
    diff = None
    if st.method == 'diffusion' and diffusion_missing() is None and diffusion_status(verify=False)['verified']:
        diff = 'pending'

    def fn(rgb, rng, idx, cam, t):
        nonlocal diff
        key = (hash(cam.tag) & 0xFFFF) * 100003 + int(round(t * 1000))
        if diff == 'pending':
            try:
                diff = _diffusion(getattr(ts.dev, 'type', 'auto'), 5.0)      # the process stays under 6 GB
            except (ValueError, ImportError, OSError, RuntimeError):
                diff = None
        if diff is not None:
            return diff(rgb, st, key)
        import torch
        Dv, N, P = frame_geometry(ts, cam, t, idx, rng)
        Rw, _ = ts.drive.traj.pose(t) if ts.drive is not None else (np.eye(3), None)

        def sky(dirs_v):
            with torch.no_grad():
                Dw = torch.tensor(np.asarray(dirs_v, float) @ Rw.T, dtype=torch.float32, device=ts.dev)
                return to_linear(ts.sky_at(Dw).clamp(0, 1).cpu().numpy())
        emit = None
        if em is not None and len(em) and em.any():
            emit = np.zeros(idx.shape, bool)
            ok = idx >= 0
            emit[ok] = em[idx[ok]]
        return relight_physical(rgb, rng, N, Dv, st, sky, emit, P, key)
    fn.settings = st
    if diff is None and st.method == 'diffusion':
        fn.label = (f'relit: {st.condition} (physical; the diffusion model is not available: '
                    f'{diffusion_missing() or "not downloaded"})')
    else:
        fn.label = st.label() if diff is None else f'relit: {st.condition} (diffusion)'
    return fn


def validation_checks() -> list:
    """Checks for vantage/validation as dicts with suite.Spec's fields; fn(ctx) returns a suite.Outcome. Proposed
    reference key: 'schlick1994' (C. Schlick, "An inexpensive BRDF model for physically-based rendering", Computer
    Graphics Forum 13(3), 1994). The lead registers them."""
    def outcome(c, err):
        try:
            from ..validation.suite import Outcome
            return Outcome(float(err), c, ok=bool(c['passed']))
        except ImportError:                               # pragma: no cover
            return {'error': float(err), 'detail': c}
    return [
        {'id': 'relight.schlick_water', 'model': 'relight',
         'title': 'The wet road\'s Fresnel term matches water\'s exact reflectance',
         'quantity': 'largest |R_Schlick − R_Fresnel| of water (n 1.333, unpolarised) from 0° to 60°',
         'reference': 'the Fresnel equations', 'tolerance': 0.01, 'unit': 'reflectance',
         'refs': ('schlick1994', 'hecht2017'),
         'fn': lambda ctx=None: outcome(check_schlick(), check_schlick()['value']),
         'cost': 'cheap'},
        {'id': 'relight.snow_extinction', 'model': 'relight',
         'title': 'Falling snow\'s extinction is ITU-R P.1814\'s',
         'quantity': 'relative difference of σ at 2 mm/h, 550 nm from the recommendation\'s coefficients',
         'reference': 'ITU-R P.1814 Table 3', 'tolerance': 1e-6, 'unit': 'relative', 'refs': ('itu_p1814',),
         'fn': lambda ctx=None: outcome(check_snow(), abs(check_snow()['value'] - check_snow()['reference']) /
                                        check_snow()['reference']), 'cost': 'cheap'},
    ]


def check_schlick() -> dict:
    n = 1.333
    worst = 0.0
    for a in np.radians(np.arange(0, 61, 5)):
        ci = math.cos(a)
        st = math.sin(a) / n
        ct = math.sqrt(1 - st * st)
        rs = ((ci - n * ct) / (ci + n * ct)) ** 2
        rp = ((ct - n * ci) / (ct + n * ci)) ** 2
        worst = max(worst, abs(0.5 * (rs + rp) - float(schlick(ci))))
    return {'passed': worst <= 0.01, 'value': round(worst, 5), 'tolerance': 0.01}


def check_snow() -> dict:
    from . import weather as WX
    ref = None
    for name in ('snow_extinction', 'snow_sigma'):
        f = getattr(WX, name, None)
        if f is not None:
            try:
                ref = float(f(2.0))
            except TypeError:
                ref = None
    mine = snow_extinction(2.0)
    if ref is None:
        db = (5.42e-5 * 550.0 + 5.50) * 2.0 ** 1.38
        ref = db * math.log(10) / 10 / 1000.0
    return {'passed': abs(mine - ref) <= 1e-6 * max(ref, 1e-12), 'value': mine, 'reference': ref}
