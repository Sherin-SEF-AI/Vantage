"""Colour scales for the 2D maps and charts, selected per theme and checked by a small validator.

Three scales, each with one job:
  * pixel density (continuous magnitude): one blue hue, lightness monotone in OKLab, interpolated in OKLab between the
    palette's blue steps. Its domain starts at the project's minimum density (log scale): ground below the minimum is
    not on the ramp at all but neutral grey and hatched, so "not enough pixels" never reads as "a bit fewer pixels".
    The end that means "just enough" sits nearest the view's surface (pale on the light theme, dark on the dark
    theme), so more pixels always means more contrast against the background.
  * redundancy (ordered classes 1, 2, 3+ cameras): three steps of the same blue, far apart in lightness; 0 cameras
    (blind) is not a step of the ramp but the reserved critical red, hatched.
  * change in density (diverging): orange (worse) and blue (better) arms around a neutral grey midpoint, each arm
    monotone in lightness away from the midpoint.

The validator (`validate`) computes rather than eyeballs: monotone OKLab lightness, adjacent steps far enough apart
(normal vision and simulated protan/deutan, Machado et al. 2009 at full severity), contrast against the surface the
maps are drawn on, and the grey for "below minimum" kept apart from the start of the ramp."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# view surfaces the maps sit on (ui.theme TOKENS['viewport'])
SURFACE = {'dark': '#1a1a19', 'light': '#fcfcfb'}

# the palette's blue steps (100 … 700); every scale below is drawn from these
BLUE = ['#cde2fb', '#b7d3f6', '#9ec5f4', '#86b6ef', '#6da7ec', '#5598e7', '#3987e5', '#2a78d6', '#256abf',
        '#1c5cab', '#184f95', '#104281', '#0d366b']

# pixel density, from "just meets the minimum" to "far above it"
DENSITY = {'light': BLUE, 'dark': BLUE[1:11][::-1]}     # dark: stops short of white, so the body stays apart
# ground that does not meet the minimum density: neutral grey + hatch (the hatch is the secondary channel)
BELOW_MIN = {'light': '#c3c2b7', 'dark': '#383835'}
HATCH = {'light': '#52514e', 'dark': '#c3c2b7'}
# ground seen by 1, 2, 3+ cameras (0 = blind: STATUS critical, hatched)
REDUNDANCY = {'light': ['#86b6ef', '#2a78d6', '#104281'], 'dark': ['#184f95', '#3987e5', '#9ec5f4']}
# change of best-camera density, worse (red) … same (grey) … better (blue)
# (the orange arm is matched in OKLab lightness to the blue arm: blue/orange is the pair that survives protan/deutan)
DIVERGING = {'light': ['#9c390c', '#d45e2f', '#e89d83', '#e1e0d9', '#86b6ef', '#3987e5', '#1c5cab'],
             'dark': ['#e89d83', '#d45e2f', '#883008', '#383835', '#184f95', '#3987e5', '#86b6ef']}

DENSITY_HI = 16.0          # top of the density scale (px/cm) unless the thresholds need more room
TICKS = (1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10, 12, 16, 20, 25, 30, 40, 50, 60, 80, 100)


def _theme(theme):
    if theme in ('dark', 'light'):
        return theme
    from .theme import THEME
    return 'light' if THEME.name == 'light' else 'dark'


# ------------------------------------------------------------------ colour maths
def hex_rgb(h) -> np.ndarray:
    """'#rrggbb' (or a list of them) -> float sRGB 0..1, shape (3,) or (n, 3)."""
    if isinstance(h, str):
        h = h.lstrip('#')
        return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], float) / 255.0
    return np.array([hex_rgb(x) for x in h])


def rgb_hex(rgb) -> str:
    r, g, b = (int(round(min(max(float(v), 0.0), 1.0) * 255)) for v in rgb)
    return f'#{r:02x}{g:02x}{b:02x}'


def to_linear(c):
    c = np.asarray(c, float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def to_srgb(c):
    c = np.clip(np.asarray(c, float), 0.0, 1.0)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055)


_M1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929],
                [0.2119034982, 0.6806995451, 0.1073969566],
                [0.0883024619, 0.2817188376, 0.6299787005]])
_M2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468],
                [1.9779984951, -2.4285922050, 0.4505937099],
                [0.0259040371, 0.7827717662, -0.8086757660]])


def oklab(rgb):
    """sRGB 0..1 (..., 3) -> OKLab (..., 3) (Ottosson 2020)."""
    lms = to_linear(rgb) @ _M1.T
    return np.cbrt(lms) @ _M2.T


def oklab_to_rgb(lab):
    lms = (np.asarray(lab, float) @ np.linalg.inv(_M2).T) ** 3
    return to_srgb(lms @ np.linalg.inv(_M1).T)


def luminance(rgb):
    lin = to_linear(rgb)
    return lin[..., 0] * 0.2126 + lin[..., 1] * 0.7152 + lin[..., 2] * 0.0722


def contrast(a, b) -> float:
    """WCAG contrast ratio of two colours (hex or sRGB)."""
    la = float(luminance(hex_rgb(a) if isinstance(a, str) else a))
    lb = float(luminance(hex_rgb(b) if isinstance(b, str) else b))
    hi, lo = max(la, lb), min(la, lb)
    return (hi + 0.05) / (lo + 0.05)


# Machado, Oliveira & Fernandes (2009), severity 1.0, applied in linear RGB
_CVD = {'protan': np.array([[0.152286, 1.052583, -0.204868], [0.114503, 0.786281, 0.099216],
                            [-0.003882, -0.048116, 1.051998]]),
        'deutan': np.array([[0.367322, 0.860646, -0.227968], [0.280085, 0.672501, 0.047413],
                            [-0.011820, 0.042940, 0.968881]])}


def simulate(rgb, kind):
    return to_srgb(np.clip(to_linear(rgb) @ _CVD[kind].T, 0, 1))


def delta_e(a, b, kind=None) -> float:
    """Euclidean OKLab distance x100, optionally under simulated colour-vision deficiency."""
    a = hex_rgb(a) if isinstance(a, str) else np.asarray(a, float)
    b = hex_rgb(b) if isinstance(b, str) else np.asarray(b, float)
    if kind:
        a, b = simulate(a, kind), simulate(b, kind)
    return float(np.linalg.norm(oklab(a) - oklab(b)) * 100)


def lightness(h) -> float:
    return float(oklab(hex_rgb(h))[0])


def ramp(hexes, n=256) -> np.ndarray:
    """(n, 3) uint8 colours interpolated in OKLab through the anchors (evenly spaced)."""
    lab = oklab(hex_rgb(hexes))
    x = np.linspace(0, len(hexes) - 1, n)
    i = np.minimum(x.astype(int), len(hexes) - 2)
    f = (x - i)[:, None]
    out = lab[i] * (1 - f) + lab[i + 1] * f
    return np.round(oklab_to_rgb(out) * 255).astype(np.uint8)


# ------------------------------------------------------------------ the scales
@dataclass
class DensityScale:
    """Best-camera pixel density (px/cm) -> colour. Log scale from the minimum density (lo) to hi; values below the
    minimum are flagged (grey, hatched) rather than coloured."""
    lo: float
    hi: float
    target: float
    theme: str
    lut: np.ndarray            # (256, 3) uint8

    def t(self, v):
        """Position on the colour bar, 0 at the minimum, 1 at hi (clipped); NaN where v <= 0."""
        v = np.asarray(v, float)
        with np.errstate(divide='ignore', invalid='ignore'):
            t = (np.log(np.maximum(v, 1e-9)) - math.log(self.lo)) / (math.log(self.hi) - math.log(self.lo))
        return np.where(v > 0, np.clip(t, 0.0, 1.0), np.nan)

    def rgb(self, v) -> np.ndarray:
        """uint8 (..., 3); below-minimum values get the neutral grey."""
        v = np.asarray(v, float)
        t = np.nan_to_num(self.t(v), nan=0.0)
        out = self.lut[np.clip(np.rint(t * (len(self.lut) - 1)), 0, len(self.lut) - 1).astype(np.intp)]
        below = v < self.lo
        if below.any():
            out = np.where(below[..., None], np.round(hex_rgb(BELOW_MIN[self.theme]) * 255).astype(np.uint8), out)
        return out

    def hex(self, v) -> str:
        return rgb_hex(self.rgb(float(v)) / 255.0)

    def qcolor(self, v, alpha=None):
        from PyQt6.QtGui import QColor
        r, g, b = (int(c) for c in self.rgb(float(v)))
        c = QColor(r, g, b)
        if alpha is not None:
            c.setAlphaF(alpha)
        return c

    @property
    def below_hex(self):
        return BELOW_MIN[self.theme]

    @property
    def hatch_hex(self):
        return HATCH[self.theme]

    def ticks(self):
        """Round values inside (lo, hi] for the colour bar, plus the ends."""
        inner = [v for v in TICKS if self.lo * 1.08 < v < self.hi * 0.97]
        return [self.lo] + inner + [self.hi]


_LUTS = {}


def density_scale(minv=3.0, target=5.0, theme=None) -> DensityScale:
    th = _theme(theme)
    lo = float(minv) if minv and minv > 0 and math.isfinite(minv) else 3.0
    tg = float(target) if target and math.isfinite(target) else lo
    hi = max(DENSITY_HI, lo * 5.0, tg * 1.25)
    hi = next((v for v in TICKS if v >= hi - 1e-9), hi)      # a round top for the colour bar
    if th not in _LUTS:
        _LUTS[th] = ramp(DENSITY[th])
    return DensityScale(lo, hi, tg, th, _LUTS[th])


def redundancy(theme=None) -> list[str]:
    """Hex colours for ground seen by 1, 2 and 3 or more cameras."""
    return list(REDUNDANCY[_theme(theme)])


def diverging(theme=None, n=257) -> np.ndarray:
    """(n, 3) uint8: index 0 = most worse, n//2 = no change, n-1 = most better."""
    return ramp(DIVERGING[_theme(theme)], n)


# ------------------------------------------------------------------ validator
def _monotone(Ls):
    d = np.diff(Ls)
    return bool((d > 0).all() or (d < 0).all())


def check_sequential(hexes, surface, min_span=0.4, min_step=8.0, levels=6):
    """A continuous ramp: lightness strictly monotone (also under protan/deutan), spanning at least min_span in OKLab
    L, running away from the surface (its first anchor nearest the surface in lightness), and `levels` evenly sampled
    steps at least min_step apart (OKLab x100) for every reader, i.e. that many levels can be told apart. Returns a
    list of problems (empty = passes)."""
    out = []
    Ls = [lightness(h) for h in hexes]
    if not _monotone(Ls):
        out.append(f'lightness not monotone: {np.round(Ls, 3).tolist()}')
    if abs(Ls[-1] - Ls[0]) < min_span:
        out.append(f'lightness span {abs(Ls[-1] - Ls[0]):.3f} < {min_span}')
    Ls_ = lightness(surface)
    if abs(Ls[0] - Ls_) > abs(Ls[-1] - Ls_):
        out.append('the ramp starts at the end farthest from the surface')
    lut = ramp(hexes, levels) / 255.0
    for kind in (None, 'protan', 'deutan'):
        sim = lut if kind is None else simulate(lut, kind)
        Lk = oklab(sim)[:, 0]
        if not _monotone(Lk):
            out.append(f'lightness not monotone under {kind}')
        de = min(delta_e(sim[i], sim[i + 1]) for i in range(len(sim) - 1))
        if de < min_step:
            out.append(f'sampled steps {de:.1f} apart ({kind or "normal"}) < {min_step}')
    return out


def check_steps(hexes, surface, min_dl=0.06, min_de=15.0, min_cvd=8.0, min_contrast=2.0):
    """Ordered classes: monotone lightness with visible gaps (dL >= min_dl), adjacent classes >= min_de apart for
    normal vision and >= min_cvd under protan/deutan, every class >= min_contrast against the surface."""
    out = []
    Ls = [lightness(h) for h in hexes]
    if not _monotone(Ls):
        out.append(f'lightness not monotone: {np.round(Ls, 3).tolist()}')
    for a, b in zip(hexes, hexes[1:]):
        if abs(lightness(a) - lightness(b)) < min_dl:
            out.append(f'{a}/{b}: lightness gap < {min_dl}')
        if delta_e(a, b) < min_de:
            out.append(f'{a}/{b}: dE {delta_e(a, b):.1f} < {min_de}')
        for kind in ('protan', 'deutan'):
            if delta_e(a, b, kind) < min_cvd:
                out.append(f'{a}/{b}: dE {delta_e(a, b, kind):.1f} under {kind} < {min_cvd}')
    for h in hexes:
        if contrast(h, surface) < min_contrast:
            out.append(f'{h}: contrast {contrast(h, surface):.2f} vs surface < {min_contrast}')
    return out


def check_diverging(hexes, surface, min_dl=0.06):
    """Two arms around a neutral midpoint: the midpoint grey (chroma < 0.02), each arm monotone in lightness going
    away from it with visible steps, the arms' hues opposite (warm vs cool) and their ends distinct for all readers."""
    out = []
    n = len(hexes)
    if n % 2 == 0:
        return ['diverging scale needs an odd number of steps']
    m = n // 2
    lab = oklab(hex_rgb(hexes))
    if math.hypot(lab[m, 1], lab[m, 2]) > 0.02:
        out.append('midpoint is not neutral')
    for arm in (hexes[m::-1], hexes[m:]):
        Ls = [lightness(h) for h in arm]
        if not _monotone(Ls):
            out.append(f'arm {arm} not monotone in lightness')
        if min(abs(a - b) for a, b in zip(Ls, Ls[1:])) < min_dl:
            out.append(f'arm {arm}: lightness gap < {min_dl}')
    if lab[0, 1] <= 0 or lab[0, 2] <= 0 or lab[-1, 2] >= 0:
        out.append('arms are not warm (worse) and blue (better)')
    for kind in (None, 'protan', 'deutan'):
        if delta_e(hexes[0], hexes[-1], kind) < 15:
            out.append(f'arm ends too close ({kind or "normal"})')
    return out


def validate(theme=None) -> dict:
    """Every scale of a theme against its surface -> {scale: [problems]} (all empty = the theme passes)."""
    th = _theme(theme)
    surf = SURFACE[th]
    rep = {'density': check_sequential(DENSITY[th], surf), 'redundancy': check_steps(REDUNDANCY[th], surf),
           'diverging': check_diverging(DIVERGING[th], surf)}
    below = []
    lo = DENSITY[th][0]
    if delta_e(BELOW_MIN[th], lo) < 8:
        below.append(f'below-minimum grey {BELOW_MIN[th]} too close to the ramp start {lo}')
    if contrast(BELOW_MIN[th], surf) < 1.4:
        below.append(f'below-minimum grey {BELOW_MIN[th]} vanishes into the surface')
    if contrast(HATCH[th], BELOW_MIN[th]) < 2.5:
        below.append('hatch lines too faint on the below-minimum grey')
    rep['below_min'] = below
    return rep
