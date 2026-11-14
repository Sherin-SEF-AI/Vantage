"""Tests for relighting re-rendered frames (core.relight).

References: Fresnel's exact reflectance of water against Schlick's approximation (0–60°), ITU-R P.1814's snow
coefficients as core.weather uses them, Koschmieder's law over a known range, the inverse-square law of core.weather's
headlamp pattern, and determinism of the seeded parts. The diffusion method runs only where diffusers and the model
files are present (test_real_*).
Run with:  QT_QPA_PLATFORM=offscreen python3 -m pytest tests/test_relight.py -q
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vantage.core import relight as RL  # noqa: E402


def _scene(h=48, w=64):
    """A camera 1.5 m up looking along +x, its upper half sky, its lower half a flat road (vehicle frame)."""
    v, u = np.mgrid[0:h, 0:w] + 0.5
    az = np.radians((w / 2 - u) / w * 90.0)
    el = np.radians((h / 2 - v) / h * 60.0)
    D = np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], -1)
    depth = np.where(D[..., 2] < -1e-3, 1.5 / np.maximum(-D[..., 2], 1e-3), 0.0)
    depth = np.minimum(depth, 200.0) * (D[..., 2] < -1e-3)
    N = np.full((h, w, 3), np.nan)
    N[depth > 0] = [0.0, 0.0, 1.0]
    P = np.array([1.0, 0.0, 1.5]) + D * depth[..., None]
    rgb = np.full((h, w, 3), 120, np.uint8)
    rgb[depth == 0] = (150, 180, 220)
    return rgb, depth, N, D, P


def test_settings_are_tolerant():
    s = RL.RelightSettings.from_dict({'condition': 'night', 'method': 'magic', 'strength': 7, 'seed': 'x',
                                      'snow_rate_mm_h': -3, 'diffusion_px': 99999})
    assert s.condition == 'night' and s.method == 'physical' and s.strength == 1.0 and s.seed == 0
    assert s.snow_rate_mm_h == 0.0 and s.diffusion_px == 1024
    assert RL.RelightSettings.from_dict(None) == RL.RelightSettings()
    assert RL.RelightSettings.from_dict(s.to_dict()) == s
    assert 'relit' in s.label() and RL.RelightSettings().label() == 'as recorded'


def test_fresnel_and_snow_against_their_references():
    assert RL.check_schlick()['passed']
    assert RL.schlick(1.0) == pytest.approx(RL.R0_WATER) and RL.R0_WATER == pytest.approx(0.0204, abs=1e-4)
    c = RL.check_snow()
    assert c['passed']
    # ITU-R P.1814 Table 3, dry snow at 550 nm, 2 mm/h, by hand: (5.42e-5·550 + 5.50)·2^1.38 dB/km
    db = (5.42e-5 * 550 + 5.50) * 2 ** 1.38
    assert RL.snow_extinction(2.0) == pytest.approx(db / 4.342944819 / 1000, rel=1e-9)
    assert RL.snow_extinction(0.0) == 0.0


def test_physical_relight_is_deterministic():
    rgb, depth, N, D, P = _scene()
    for cond in ('wet', 'night', 'snow'):
        st = RL.RelightSettings(condition=cond)
        a = RL.relight_physical(rgb, depth, N, D, st, points=P, rng_key=7)
        b = RL.relight_physical(rgb, depth, N, D, st, points=P, rng_key=7)
        assert a.dtype == np.uint8 and a.shape == rgb.shape and np.array_equal(a, b)
    st = RL.RelightSettings(condition='snow', snow_rate_mm_h=8.0)
    a = RL.relight_physical(rgb, depth, N, D, st, rng_key=1)
    c = RL.relight_physical(rgb, depth, N, D, st, rng_key=2)
    assert not np.array_equal(a, c)                 # the flakes depend on the camera and instant
    none = RL.relight_physical(rgb, depth, N, D, RL.RelightSettings(condition='none'))
    assert np.abs(none.astype(int) - rgb.astype(int)).max() <= 1   # the sRGB round trip only


def test_wet_darkens_the_road_and_shows_the_sun_glare():
    rgb, depth, N, D, P = _scene()
    road = depth > 0
    st = RL.RelightSettings(condition='wet', sun_az_deg=0.0, sun_el_deg=20.0)
    out = RL.relight_physical(rgb, depth, N, D, st, sky_reflect=lambda d: np.full((len(d), 3), 0.02))
    lin_in = RL.to_linear(rgb / 255.0)
    lin_out = RL.to_linear(out / 255.0)
    # away from the glare a road pixel is the dry one × 0.6 plus the dim sky's Fresnel reflection
    refl = D - 2 * D[..., 2:3] * np.array([0.0, 0.0, 1.0]) * 1.0
    ang = np.degrees(np.arccos(np.clip(refl @ np.array([math.cos(math.radians(20)), 0, math.sin(math.radians(20))]),
                                       -1, 1)))
    far_from_sun = road & (ang > 20)
    cos_i = np.abs(D[..., 2])
    expect = lin_in[..., 0] * 0.6 + RL.schlick(cos_i) * 0.02
    assert np.allclose(lin_out[..., 0][far_from_sun], expect[far_from_sun], atol=0.004)
    near_sun = road & (ang < 2)
    if near_sun.any():
        assert lin_out[near_sun].mean() > lin_in[near_sun].mean() * 1.5     # the glare
    assert np.array_equal(out[~road], RL.relight_physical(rgb, depth, N, D, st)[~road])   # the sky untouched


def test_night_follows_the_headlamps_and_keeps_emitters():
    from vantage.core import weather as WX
    rgb, depth, N, D, P = _scene()
    road = depth > 0
    em = np.zeros(depth.shape, bool)
    em[5, 5] = True
    st = RL.RelightSettings(condition='night', night_ambient_lux=0.0, read_noise=0.0)
    out = RL.relight_physical(rgb, depth, N, D, st, emissive=em, points=P)
    assert np.array_equal(out[5, 5], rgb[5, 5])                       # an emitter keeps its light
    sky = (depth == 0) & ~em
    assert out[sky].mean() < 0.2 * rgb[sky].mean()                    # the sky goes dark
    # a road pixel's linear value is its daylight one × E / exposure_lux, E from the two lamps (inverse square)
    ws = WX.WeatherSettings()
    lamps = WX.lamp_positions(None, ws)
    lamps[:, 2] = st.lamp_height_m
    v, u = np.nonzero(road)
    k = np.argsort(depth[v, u])[len(v) // 2]
    X = P[v[k], u[k]]
    E = 0.0
    for L in lamps:
        dv = X - L
        r = np.linalg.norm(dv)
        I = WX.headlamp_cd(math.degrees(math.atan2(dv[1], dv[0])), math.degrees(math.asin(dv[2] / r)), 'low',
                           ws.traffic, ws.low_beam_cd, ws.high_beam_cd)
        E += float(I) * max(-dv[2] / r, 0.0) / (r * r)
    lin = RL.to_linear(rgb[v[k], u[k]] / 255.0) * E / st.exposure_lux
    assert RL.to_linear(out[v[k], u[k]] / 255.0) == pytest.approx(lin, abs=0.003)


def test_snow_whitens_the_ground_and_follows_koschmieder():
    rgb, depth, N, D, P = _scene()
    st = RL.RelightSettings(condition='snow', snow_rate_mm_h=0.0)          # cover only, no falling snow
    out = RL.relight_physical(rgb, depth, N, D, st)
    road = depth > 0
    assert out[road].mean() > rgb[road].mean() + 40
    st2 = RL.RelightSettings(condition='snow', snow_rate_mm_h=5.0, strength=1.0)
    flat = np.full_like(N, np.nan)                                         # no cover: the extinction alone
    o2 = RL.relight_physical(rgb, depth, flat, D, st2, rng_key=0)
    sig = RL.snow_extinction(5.0)
    t = np.exp(-sig * depth[road])[:, None]
    exp_lin = RL.to_linear(rgb[road] / 255.0) * t + RL.to_linear(0.78) * (1 - t)
    err = np.abs(RL.to_linear(o2[road] / 255.0) - exp_lin).max(1)
    assert np.median(err) < 0.004                       # Koschmieder over each pixel's range
    assert np.mean(err > 0.02) < 0.05                   # a few pixels under flakes
    sky = depth == 0                                     # the sky behind the snowfall: 300 m of it
    t_sky = math.exp(-sig * 300.0)
    assert RL.to_linear(o2[sky] / 255.0).mean() == pytest.approx(
        (RL.to_linear(rgb[sky] / 255.0) * t_sky + RL.to_linear(0.78) * (1 - t_sky)).mean(), abs=0.01)


def test_normals_from_the_range_image():
    """A flat road seen from 1.5 m up gives up-facing normals everywhere on it (no blotches), and an object's edge
    (a range jump) gets none."""
    rgb, depth, N, D, P = _scene(96, 128)
    n = RL.depth_normals(D * depth[..., None], depth)
    road = (depth > 0) & np.isfinite(n).all(-1)
    assert road.sum() > 0.8 * (depth > 0).sum()
    assert np.abs(n[road] - np.array([0.0, 0.0, 1.0])).max() < 1e-3
    d2 = depth.copy()
    d2[60:, 64:] *= 0.5                                  # a nearer object over part of the road
    n2 = RL.depth_normals(D * d2[..., None], d2, smooth=1)
    assert not np.isfinite(n2[70, 63]).all() and np.isfinite(n2[70, 30]).all()


def test_relighter_none_and_fallback_label(monkeypatch):
    assert RL.relighter({'condition': 'none'}, None) is None
    monkeypatch.setattr(RL, 'diffusion_missing', lambda: 'needs diffusers')
    fn = RL.relighter({'condition': 'wet', 'method': 'diffusion'}, None)
    assert callable(fn) and fn.label.startswith('relit: wet (physical') and 'needs diffusers' in fn.label


def test_diffusion_download_refuses_a_wrong_file(tmp_path):
    """The downloader keeps nothing whose size or SHA-256 is not the published file's."""
    class R:
        def __init__(self, data):
            self.data, self.k = data, 0

        def read(self, n):
            b = self.data[self.k:self.k + n]
            self.k += n
            return b

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False
    sp = RL._ip2p_spec()
    first = sp.files[0]
    with pytest.raises(ValueError):
        RL.download_diffusion(folder=str(tmp_path), fetch=lambda url, timeout: R(b'x' * first.size))
    assert not os.path.exists(os.path.join(str(tmp_path), sp.id, first.name))
    assert not RL.diffusion_status(folder=str(tmp_path))['verified']


def test_validation_checks_are_specs():
    from vantage.validation.suite import Spec
    for c in RL.validation_checks():
        sp = Spec(**c)
        out = sp.fn(None)
        assert out.ok and out.error <= sp.tolerance


def test_no_unseeded_randomness():
    src = open(RL.__file__, encoding='utf-8').read()
    assert 'np.random.rand' not in src and 'np.random.normal' not in src and 'uuid4' not in src


@pytest.mark.skipif(RL.diffusion_missing() is not None or not RL.diffusion_status(verify=False)['verified'],
                    reason='needs diffusers and the InstructPix2Pix files')
def test_real_diffusion_relights_a_frame():
    rgb, depth, N, D, P = _scene(128, 160)
    d = RL.DiffusionRelighter(mem_gb=5.0)
    st = RL.RelightSettings(condition='snow', method='diffusion', diffusion_steps=4, diffusion_px=256)
    a = d(rgb, st, key=3)
    b = d(rgb, st, key=3)
    assert a.shape == rgb.shape and np.array_equal(a, b)
