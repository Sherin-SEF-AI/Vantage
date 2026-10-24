"""e-con Systems camera modules in the built-in library (core/lens.py): every lens reproduces the field of view e-con
states (checked by projecting rays at the stated half angles onto the pixel grid, independently of model_fov), the
library hangs together (ids, modules, notes, the figures typed again from e-con's pages), projects round-trip and old
files load, nothing assumes a single built-in module (analysis, exports, the library panel, the inspector, placing a
camera), and the module editor lists the library and switches cameras to a preset (offscreen)."""
import copy
import json
import math
import os
import zipfile

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np
import pytest
from PyQt6.QtWidgets import QMenu

from vantage.core import lens as L
from vantage.core import samples
from vantage.core.analysis import Engine
from vantage.core.lens import Lens, Module, builtin_lenses, builtin_modules, model_fov, resolve, r_of_theta, theta_of_r
from vantage.core.project import Project

from test_gui import app, win, pump, settle   # noqa: F401 (fixtures)

# The figures as e-con's pages state them (read 2026-10-24), typed again here from the pages rather than copied from
# lens.py: (sensor, width, height, pixel µm, optical format, fps, shutter, dynamic range dB, temperature °C, power W,
# weight g, IP rating, interface). 0 / '' = not stated.
E = 'https://www.e-consystems.com/'
SHEET = {
    'sturdecam31': ('Sony ISX031', 1920, 1536, 3.0, '1/2.42"', 30, 'rolling', 120, (-40, 85), 1.324, 40, 'IP69K', 'GMSL2'),
    'sturdecam31fsh': ('Sony ISX031', 1920, 1536, 3.0, '1/2.42"', 30, 'rolling', 120, (-40, 85), 4.5, 0, 'IP69K',
                       'GMSL2'),
    'sturdecam34': ('onsemi AR0341AT', 1920, 1536, 2.1, '1/3.6"', 30, 'rolling', 140, (-40, 85), 2.192, 46.3, 'IP69K',
                    'GMSL2'),
    'sturdecam39': ('OmniVision OX03J10', 1920, 1536, 3.0, '1/2.4"', 30, 'rolling', 140, (-40, 85), 1.332, 0, 'IP69K',
                    'GMSL2'),
    'sturdecam25': ('onsemi AR0234CS', 1920, 1200, 3.0, '1/2.6"', 60, 'global', 71.4, (-30, 70), 1.5, 77.67, 'IP67',
                    'GMSL2'),
    'sturdecam29': ('OmniVision OV2311', 1600, 1300, 3.0, '1/2.9"', 60, 'global', 68, (-40, 85), 1.7, 131, 'IP65',
                    'GMSL2'),
    'sturdecam57': ('OmniVision OX05B1S', 2592, 1944, 2.2, '1/2.5"', 50, 'global', 71, (-40, 85), 3.42, 100, 'IP67',
                    'GMSL2'),
    'sturdecam81': ('onsemi AR0821', 3840, 2160, 2.1, '1/1.7"', 30, 'rolling', 100, (-30, 70), 2.496, 0, 'IP67',
                    'GMSL2'),
    'sturdecam88': ('OmniVision OX08B40', 3840, 2160, 2.1, '1/1.73"', 30, 'rolling', 140, (-40, 80), 2.5, 80, 'IP67',
                    'GMSL2'),
    'sturdecam20': ('onsemi AR0230AT', 1920, 1080, 3.0, '1/2.7"', 30, 'rolling', 96, (-40, 85), 2.196, 0, 'IP67',
                    'GMSL2'),
    'sturdecam21': ('onsemi AR0233AT', 1920, 1080, 3.0, '1/2.5"', 30, 'rolling', 120, (-40, 85), 2.196, 0, 'IP67',
                    'GMSL2'),
    'nilecam81': ('onsemi AR0821', 3840, 2160, 2.1, '1/1.7"', 16, 'rolling', 100, (-30, 70), 2.496, 0, '', 'GMSL2'),
    'nilecam85': ('Sony IMX715', 3840, 2160, 1.45, '1/2.8"', 30, 'rolling', 0, (-30, 70), 0.0, 0, '', 'GMSL2'),
    'nilecam130': ('onsemi AR1335', 4208, 3120, 1.1, '1/3.2"', 10, 'rolling', 69, (-30, 70), 1.1, 0, '', 'GMSL2'),
    'nilecam210': ('Sony IMX462', 1920, 1080, 2.9, '1/2.8"', 60, 'rolling', 0, (-30, 70), 0.0, 0, '', 'GMSL2'),
    'neducam25': ('onsemi AR0234CS', 1920, 1200, 3.0, '1/2.6"', 60, 'global', 71.4, (-20, 70), 0.0, 0, '',
                  'FPD-Link III'),
    'nedu4cam57': ('OmniVision OX05B1S', 2592, 1944, 2.2, '1/2.53"', 50, 'global', 71, (-40, 85), 3.48, 100, 'IP67',
                   'FPD-Link IV'),
}
# lens: (module, H, V, D, near focus m) as e-con states them; D of a 'circle' lens is the image circle's field
LENSES = {
    'h05r2': ('sturdecam31', 128.7, 98.7, 170.1, 0.35),
    'h03r2': ('sturdecam31', 54.04, 42.86, 70.17, 1.28),
    'sturdecam31fsh_heated': ('sturdecam31fsh', 196.0, 155.0, 210.0, 0.5),
    'sturdecam34_h01r1': ('sturdecam34', 64.62, 51.28, 83.5, 1.7),
    'sturdecam39_option1': ('sturdecam39', 196.7, 155.1, 200.0, 0.5),
    'sturdecam39_option2': ('sturdecam39', 121.0, 96.7, 164.5, 0.0),
    'sturdecam25_std': ('sturdecam25', 111.1, 68.3, 132.8, 1.0),
    'sturdecam29_std': ('sturdecam29', 93.0, 74.0, 131.0, 0.0),
    'sturdecam57_std': ('sturdecam57', 114.76, 85.92, 148.32, 0.5),
    'sturdecam81_std': ('sturdecam81', 106.3, 57.1, 122.6, 0.0),
    'sturdecam88_h02r1': ('sturdecam88', 68.23, 38.83, 80.22, 0.0),
    'sturdecam88_h04r1': ('sturdecam88', 120.12, 67.04, 137.36, 0.0),
    'sturdecam20_std': ('sturdecam20', 98.5, 53.5, 116.5, 1.0),
    'sturdecam21_std': ('sturdecam21', 98.22, 50.34, 113.83, 1.0),
    'nilecam81_std': ('nilecam81', 104.8, 58.7, 120.0, 0.0),
    'nilecam85_std': ('nilecam85', 113.0, 61.0, 131.0, 0.0),
    'nilecam130_std': ('nilecam130', 86.0, 64.0, 110.0, 0.0),
    'nilecam210_std': ('nilecam210', 113.0, 61.0, 132.0, 0.0),
    'neducam25_std': ('neducam25', 104.6, 61.6, 128.2, 0.0),
    'nedu4cam57_std': ('nedu4cam57', 114.76, 85.92, 148.32, 0.5),
}
CIRCLE = {'sturdecam31fsh_heated', 'sturdecam39_option1'}     # the stated D is the image circle (corners dark)
MODS = {m.id: m for m in builtin_modules()}


def _lenses():
    return {l.id: l for l in builtin_lenses()}


def _ray(theta, phi):
    """Optical-frame unit ray theta off the axis at azimuth phi (x right, y down, z forward)."""
    return np.array([[math.sin(theta) * math.cos(phi), math.sin(theta) * math.sin(phi), math.cos(theta)]])


# ----------------------------------------------------------------------------- the stated FOV, on the pixel grid
@pytest.mark.parametrize('lid', list(LENSES))
def test_every_lens_reproduces_the_stated_fov_on_the_pixel_grid(lid):
    """A ray 0.5° inside the stated half angle lands inside the image and one 0.5° outside lands beyond its edge
    (forward projection through r_of_theta: independent of model_fov's inverse)."""
    lens = _lenses()[lid]
    mid, H, V, D, _ = LENSES[lid]
    m = MODS[mid]
    assert lens.module_id == mid and tuple(lens.sheet_fov) == (H, V, D)
    it = resolve(m, lens)
    W, Hh = it.width, it.height
    half = math.radians(0.5)
    diag = math.atan2(Hh, W)
    for fov, phi, edge in ((H, 0.0, lambda u, v: u - W), (V, math.pi / 2, lambda u, v: v - Hh)):
        for sgn in (-1, 1):
            u, v, th, ok = it.project_dirs(_ray(math.radians(fov / 2) + sgn * half, phi))
            assert (edge(u[0], v[0]) < 0) == (sgn < 0), (lid, fov, sgn, u, v)
    if lid in CIRCLE:
        # the corners are beyond the image circle, which is D / 2 off axis
        assert math.degrees(it.max_theta) == pytest.approx(D / 2, abs=1e-9)
        u, v, th, ok = it.project_dirs(_ray(it.max_theta, diag))
        assert u[0] < W and v[0] < Hh                      # the circle ends inside the corner
        assert it.project_dirs(_ray(it.max_theta - 1e-9, diag))[3][0]
        assert not it.project_dirs(_ray(it.max_theta + half, diag))[3][0]
    else:
        for sgn in (-1, 1):
            u, v, th, ok = it.project_dirs(_ray(math.radians(D / 2) + sgn * half, diag))
            assert (u[0] < W and v[0] < Hh) == (sgn < 0), (lid, D, sgn)
        assert math.degrees(it.max_theta) >= D / 2          # the corners are imaged
    seen = model_fov(m, lens, circle=True)
    assert all(abs(a - b) < 0.5 for a, b in zip(seen, (H, V, D))), (lid, seen)


@pytest.mark.parametrize('lid', list(LENSES))
def test_lens_curves_rise_over_the_whole_sensor_and_invert(lid):
    """r(θ) rises to the sensor corner (or the image circle, whichever is further), so every pixel has one ray, and
    theta_of_r inverts r_of_theta there; unproject(project(ray)) is the ray."""
    lens = _lenses()[lid]
    m = MODS[lens.module_id]
    it = resolve(m, lens)
    hd = math.hypot(m.sensor_w_mm, m.sensor_h_mm) / 2
    corner = float(theta_of_r(hd, lens.focal_mm, lens.projection, lens.kb))
    reach = max(corner, it.max_theta)
    assert L.monotonic(lens.focal_mm, lens.kb, reach)
    assert float(r_of_theta(corner, lens.focal_mm, lens.projection, lens.kb)) == pytest.approx(hd, rel=1e-6)
    th = np.linspace(0.01, reach, 60)
    r = r_of_theta(th, lens.focal_mm, lens.projection, lens.kb)
    assert np.allclose(theta_of_r(r, lens.focal_mm, lens.projection, lens.kb), th, atol=1e-6)
    us = np.array([3.0, it.width * 0.3, it.width - 2.0])
    vs = np.array([it.height - 3.0, it.height * 0.6, 2.0])
    d, t = it.unproject(us, vs)
    u2, v2, t2, ok = it.project_dirs(d)
    keep = t <= it.max_theta
    assert np.allclose(u2[keep], us[keep], atol=1e-3) and np.allclose(v2[keep], vs[keep], atol=1e-3)


def test_the_focal_length_comes_from_the_stated_fov_with_the_best_base():
    """A built-in e-con lens's focal length is not stated: it is the least-squares focal length of the base projection
    that fits the stated FOV best (recomputed here by brute force over focal lengths), and the note says so."""
    for lid, lens in _lenses().items():
        if lid in ('h05r2', 'h03r2'):
            continue
        m = MODS[lens.module_id]
        hs = L.half_sizes(m)
        pts = [(math.radians(f / 2), h) for f, h in zip(lens.sheet_fov, hs) if f]
        if lid in CIRCLE:
            pts = pts[:2]
        best = None
        for base in ('rectilinear', 'stereographic', 'equidistant', 'equisolid'):
            if base == 'rectilinear' and max(t for t, _ in pts) >= math.radians(80):
                continue
            fs = np.linspace(0.5, 10.0, 19001)
            err = sum(((r_of_theta(t, fs, base) / h) - 1) ** 2 for t, h in pts)
            k = int(np.argmin(err))
            if best is None or err[k] < best[0]:
                best = (err[k], base, fs[k])
        assert lens.focal_mm == pytest.approx(best[2], abs=1.1e-3), lid
        assert f'{best[1]} base projection' in lens.note and 'does not state the focal length' in lens.note
        assert lens.f_number == 0.0 and 'aperture' in lens.missing()


# ----------------------------------------------------------------------------- the library hangs together
def test_library_ids_modules_and_order():
    mods, lenses = builtin_modules(), builtin_lenses()
    assert len({m.id for m in mods}) == len(mods) == len(SHEET)
    assert len({l.id for l in lenses}) == len(lenses) == len(LENSES)
    assert not {m.id for m in mods} & {l.id for l in lenses}
    # the ids old projects and the tests use, first in the library (new cameras start with them)
    assert mods[0].id == 'sturdecam31' and [l.id for l in lenses[:2]] == ['h05r2', 'h03r2']
    for l in lenses:
        assert l.module_id in MODS and l.builtin and l.projection == 'kannala_brandt'
    for m in mods:
        assert m.builtin and m.vendor == 'e-con Systems'
        assert any(l.module_id == m.id for l in lenses), m.id          # every module has a lens to place
    assert L.builtin_module('sturdecam88').name == 'STURDeCAM88' and L.builtin_lens('nope') is None


def test_every_figure_matches_the_pages():
    for mid, (sensor, w, h, pix, fmt, fps, shutter, dr, temp, power, weight, ip, iface) in SHEET.items():
        m = MODS[mid]
        assert (m.sensor, m.width_px, m.height_px, m.pixel_um, m.optical_format) == (sensor, w, h, pix, fmt), mid
        assert (m.fps, m.shutter, m.dynamic_range_db, m.temp_c) == (fps, shutter, dr, tuple(float(t) for t in temp))
        assert (m.power_w, m.weight_g, m.ip_rating, m.interface) == (power, weight, ip, iface), mid
        assert m.url.startswith(E) and m.url in m.note and L.CHECKED in m.note
    # enclosure / board size (width, height, length; 0 = not stated), cable lengths and stated hardware frame sync
    size = {'sturdecam31': (25, 25, 43.8), 'sturdecam31fsh': (35, 35, 36.8), 'sturdecam34': (25, 25, 0),
            'sturdecam39': (25, 25, 48.53), 'sturdecam25': (42, 42, 45.22), 'sturdecam29': (59.02, 52.52, 42.75),
            'sturdecam57': (70, 35, 30), 'sturdecam81': (42, 42, 0), 'sturdecam88': (42, 42, 0),
            'sturdecam20': (42, 42, 0), 'sturdecam21': (42, 42, 0), 'nilecam81': (30, 30, 0), 'nilecam85': (30, 30, 0),
            'nilecam130': (30, 30, 0), 'nilecam210': (30, 30, 0), 'neducam25': (30, 30, 0), 'nedu4cam57': (70, 35, 30)}
    short = {'sturdecam31', 'sturdecam31fsh', 'sturdecam34', 'sturdecam57', 'sturdecam88', 'nilecam81', 'nilecam85',
             'nilecam210'}                                       # 3 m kit cable as well as 15 m
    sync = {'sturdecam31', 'sturdecam34', 'sturdecam39', 'sturdecam25', 'sturdecam81', 'sturdecam88', 'sturdecam20',
            'sturdecam21', 'neducam25'}
    for mid, m in MODS.items():
        assert m.size_mm == tuple(float(v) for v in size[mid]) and m.hw_sync == (mid in sync), mid
        assert m.cables_m == ((10.0,) if mid == 'nedu4cam57' else (3.0, 15.0) if mid in short else (15.0,)), mid
    for lid, (mid, H, V, D, near) in LENSES.items():
        l = _lenses()[lid]
        assert l.focus_near_m == near and L.CHECKED in l.note, lid
    assert _lenses()['sturdecam57_std'].focus_far_m == 2.0 and math.isinf(_lenses()['h05r2'].focus_far_m)


def test_the_optical_format_matches_the_pixel_array():
    """The stated optical format (type 1/x": a diagonal of about 2/3 × 25.4 / x mm) agrees with resolution × pixel
    pitch within 15 %: catches a pitch or a resolution typed wrong."""
    for m in builtin_modules():
        x = float(m.optical_format.strip('"').split('/')[1])
        nominal = 25.4 / x * 2 / 3
        diag = math.hypot(m.sensor_w_mm, m.sensor_h_mm)
        assert abs(diag / nominal - 1) < 0.15, (m.id, diag, nominal)


def test_not_given_is_zero_or_blank_and_listed():
    m34 = MODS['sturdecam34']
    assert m34.size_mm == (25.0, 25.0, 0.0) and m34.missing() == ['length (depth)']
    assert m34.body_mm == (25.0, 25.0, 43.8)                       # drawn with the typical length
    fsh = MODS['sturdecam31fsh']
    assert fsh.weight_g == 0.0 and 'weight' in fsh.missing() and not fsh.hw_sync
    n85 = MODS['nilecam85']
    assert {'power', 'dynamic range', 'IP rating', 'weight'} <= set(n85.missing())
    assert MODS['sturdecam31'].missing() == []                      # e-con states every figure the module has
    assert _lenses()['sturdecam29_std'].missing() == ['aperture', 'focus distance']
    d = L.derived(MODS['sturdecam29'], _lenses()['sturdecam29_std'])
    assert d['hyperfocal_m'] is None and d['f_px'] > 0


def test_sturdecam31_corrections_and_the_sample_lens_unchanged():
    m = MODS['sturdecam31']
    # corrected from the current pages: 1/2.42" (was 1/2.4"), 40 g with lens (was 45 g)
    assert m.optical_format == '1/2.42"' and m.weight_g == 40.0
    assert (m.width_px, m.height_px, m.pixel_um, m.fps, m.size_mm, m.cables_m, m.hw_sync) == \
        (1920, 1536, 3.0, 30.0, (25.0, 25.0, 43.8), (3.0, 15.0), True)
    hs = L.half_sizes(m)
    wide = _lenses()['h05r2']
    # every sample camera uses H05R2: its model is exactly the earlier one (default fit weight)
    assert wide.kb == L.fit_kb_to_fov(2.62, (128.7, 98.7, 170.1), hs, base='equisolid')
    assert (wide.focal_mm, wide.f_number, wide.focus_near_m) == (2.62, 2.0, 0.35)
    # H03R2 was refitted: the earlier fit missed the stated vertical FOV by more than 1°
    narrow = _lenses()['h03r2']
    before = Lens(id='x', focal_mm=5.5, projection='kannala_brandt', sheet_fov=narrow.sheet_fov,
                  kb=L.fit_kb_to_fov(5.5, narrow.sheet_fov, hs, base='equisolid'))
    assert abs(model_fov(m, before)[1] - 42.86) > 1.0
    assert all(abs(a - b) < 0.5 for a, b in zip(model_fov(m, narrow), narrow.sheet_fov))


def test_the_sheet_fit_falls_back_to_a_gentler_weight_rather_than_fold():
    """Inconsistent figures (a 123° D barely wider than a 120° H on 16:9) would fold the curve back at the strong
    weight: fit_sheet then takes the strongest weight whose curve still rises."""
    m = Module(id='x', width_px=1920, height_px=1080, pixel_um=3.0)
    hs = L.half_sizes(m)
    sheet = (120.0, 60.0, 123.0)
    b, f, _ = L.best_base(sheet, hs)
    reach = math.radians(123.0 / 2 + 2)
    assert not L.monotonic(f, L.fit_kb_to_fov(f, sheet, hs, base=b, weight=L.SHEET_WEIGHT), reach)
    kb = L.fit_sheet(f, sheet, hs, base=b)
    assert kb == L.fit_kb_to_fov(f, sheet, hs, base=b, weight=20.0) and L.monotonic(f, kb, reach)


def test_summaries_name_the_stated_figures_only():
    s88 = MODS['sturdecam88'].summary()
    assert s88.startswith('e-con Systems STURDeCAM88 · OmniVision OX08B40 · 3840 × 2160 · 2.1 µm')
    assert 'rolling shutter' in s88 and 'IP67' in s88 and '-40 to 80 °C' in s88 and '2.5 W' in s88
    assert ' W' not in MODS['nilecam85'].summary() and 'dB' not in MODS['nilecam85'].summary()
    l57 = _lenses()['sturdecam57_std'].summary()
    assert '114.76° H × 85.92° V × 148.32° D' in l57 and 'aperture not given' in l57 and 'focus 0.5 m – 2 m' in l57
    assert 'F2' in _lenses()['h05r2'].summary() and 'focus not given' in _lenses()['sturdecam88_h02r1'].summary()


def test_circle_reasons_are_stated():
    fsh = _lenses()['sturdecam31fsh_heated']
    assert 'taken as the image circle' in fsh.note and '28 %' in fsh.note and fsh.max_half_deg == 105.0
    o1 = _lenses()['sturdecam39_option1']
    assert '≥ 200° D' in o1.note and '200° circle' in o1.note and o1.max_half_deg == 100.0
    raw = model_fov(MODS['sturdecam39'], o1)
    assert raw[2] > 200.0 and model_fov(MODS['sturdecam39'], o1, circle=True)[2] == pytest.approx(200.0)


# ----------------------------------------------------------------------------- files: round trip, old projects
def _mixed(p):
    """Every built-in lens on a camera of the active variant (cycling over its cameras)."""
    cams = p.variant.cameras
    out = {}
    for i, l in enumerate(builtin_lenses()):
        c = cams[i % len(cams)]
        c.module_id, c.lens_id = l.module_id, l.id
        out[c.uid] = (l.module_id, l.id)
    return out


def test_project_round_trip_keeps_the_library_and_assignments(tmp_path):
    p = samples.tigor_sample_project()
    want = _mixed(p)
    p.modules['sturdecam34'].weight_g = 47.0                      # an edited built-in survives as edited
    path = str(tmp_path / 'mixed.vantage')
    p.save(path)
    q = Project.load(path)
    assert {k: m.to_dict() for k, m in q.modules.items()} == {k: m.to_dict() for k, m in p.modules.items()}
    assert {k: l.to_dict() for k, l in q.lenses.items()} == {k: l.to_dict() for k, l in p.lenses.items()}
    assert q.modules['sturdecam34'].weight_g == 47.0 and q.modules['sturdecam31fsh'].temp_c == (-40.0, 85.0)
    for c in q.variant.cameras:
        if c.uid in want:
            assert (c.module_id, c.lens_id) == want[c.uid]
            a = resolve(q.module_for(c), q.lens_for(c))
            b = resolve(p.module_for(p.variant.cam(c.uid)), p.lens_for(p.variant.cam(c.uid)))
            assert (a.f_px, a.kb, a.max_theta, a.width, a.height) == (b.f_px, b.kb, b.max_theta, b.width, b.height)


OLD_MODULE = {'id': 'sturdecam31', 'name': 'STURDeCAM31', 'vendor': 'e-con Systems', 'sensor': 'Sony ISX031',
              'width_px': 1920, 'height_px': 1536, 'pixel_um': 3.0, 'optical_format': '1/2.4"', 'interface': 'GMSL2',
              'fps': 30.0, 'hw_sync': True, 'size_mm': [25.0, 25.0, 43.8], 'weight_g': 45.0, 'cables_m': [3.0, 15.0],
              'power_w': 0.0, 'builtin': True}


def _old_lens(lid, name, focal, sheet, near, kb):
    return {'id': lid, 'name': name, 'label': 'wide', 'module_id': 'sturdecam31', 'focal_mm': focal, 'f_number': 2.0,
            'focus_near_m': near, 'focus_far_m': None, 'projection': 'kannala_brandt', 'kb': list(kb),
            'sheet_fov': list(sheet), 'max_half_deg': 0.0, 'pp_offset_px': [0.0, 0.0], 'aspect': 1.0, 'measured': [],
            'fit_rms_px': 0.0, 'builtin': True}


def test_an_old_project_file_without_the_new_fields_loads(tmp_path):
    """A project saved before this library (one module, two lenses, no note/shutter/...): it loads, keeps its own
    copies as saved, and merge_builtins offers the new modules beside them."""
    p = samples.tigor_sample_project()
    path = str(tmp_path / 'old.vantage')
    p.save(path)
    with zipfile.ZipFile(path) as z:
        data = json.loads(z.read('project.json'))
        mesh = z.read('mesh.npz')
    old_kb = (0.0855, -1.5636, -0.2386, 0.1331)
    data['modules'] = {'sturdecam31': OLD_MODULE}
    data['lenses'] = {'h05r2': _old_lens('h05r2', 'H05R2', 2.62, (128.7, 98.7, 170.1), 0.35, _lenses()['h05r2'].kb),
                      'h03r2': _old_lens('h03r2', 'H03R2', 5.5, (54.04, 42.86, 70.17), 1.28, old_kb)}
    old = str(tmp_path / 'old2.vantage')
    with zipfile.ZipFile(old, 'w') as z:
        z.writestr('project.json', json.dumps(data))
        z.writestr('mesh.npz', mesh)
    q = Project.load(old)
    m = q.modules['sturdecam31']
    assert (m.optical_format, m.weight_g, m.note, m.shutter, m.temp_c, m.dynamic_range_db) == ('1/2.4"', 45.0, '', '',
                                                                                              (), 0.0)
    assert q.lenses['h03r2'].kb == old_kb and q.lenses['h03r2'].note == ''
    assert Engine().analyse(q) is not None
    mods, lenses = L.merge_builtins(q.modules, q.lenses)
    assert list(mods)[0] == 'sturdecam31' and mods['sturdecam31'] is q.modules['sturdecam31']     # saved wins
    assert set(mods) == {m.id for m in builtin_modules()} and set(lenses) == set(LENSES)
    assert lenses['h03r2'].kb == old_kb and lenses['sturdecam88_h04r1'].module_id == 'sturdecam88'
    # a hand-edited file with junk in the newer fields still loads
    junk = dict(OLD_MODULE, temp_c='hot', dynamic_range_db='lots', note=None, shutter='slow', size_mm=None,
                cables_m='3m', url=7)
    jm = Module.from_dict(junk)
    assert (jm.temp_c, jm.dynamic_range_db, jm.note, jm.shutter, jm.size_mm, jm.cables_m, jm.url) == \
        ((), 0.0, '', '', (25.0, 25.0, 43.8), (3.0, 15.0), '')
    assert Lens.from_dict(dict(_old_lens('a', 'A', 2.0, (0, 0, 0), 0.3, (0, 0, 0, 0)), note=5)).note == ''


def _loading_merges():
    import inspect
    return 'merge_builtins' in inspect.getsource(Project.load)


@pytest.mark.skipif(not _loading_merges(), reason='Project.load does not merge the built-in library yet (core/project.py '
                                                  'change proposed in the round-5 econcams report)')
def test_loading_an_old_project_offers_every_builtin(tmp_path):
    p = samples.tigor_sample_project()
    path = str(tmp_path / 'p.vantage')
    p.save(path)
    with zipfile.ZipFile(path) as z:
        data = json.loads(z.read('project.json'))
        mesh = z.read('mesh.npz')
    data['modules'] = {'sturdecam31': OLD_MODULE}
    data['lenses'] = {'h05r2': _old_lens('h05r2', 'H05R2', 2.62, (128.7, 98.7, 170.1), 0.35, (0.1, 0, 0, 0))}
    with zipfile.ZipFile(path, 'w') as z:
        z.writestr('project.json', json.dumps(data))
        z.writestr('mesh.npz', mesh)
    q = Project.load(path)
    assert list(q.modules)[0] == 'sturdecam31' and set(q.modules) == set(SHEET) and set(q.lenses) == set(LENSES)
    assert q.modules['sturdecam31'].weight_g == 45.0 and q.lenses['h05r2'].kb == (0.1, 0, 0, 0)     # saved wins
    assert q.lenses['h03r2'].to_dict() == _lenses()['h03r2'].to_dict()


# ----------------------------------------------------------------------------- nothing assumes one built-in module
@pytest.fixture(scope='module')
def engine_sample():
    p = samples.tigor_sample_project()
    return p, Engine()


@pytest.mark.parametrize('lid', list(LENSES))
def test_an_analysis_runs_with_each_module_and_lens(engine_sample, lid):
    p0, e = engine_sample
    p = copy.deepcopy(p0)
    lens = _lenses()[lid]
    for c in p.variant.cameras:
        c.module_id, c.lens_id = lens.module_id, lid
    res = e.analyse(p)
    met = res.metrics
    assert 0.0 < met['covered_pct'] <= 100.0 and all(math.isfinite(float(v)) for v in met.values())
    # the camera ahead sees the ground point its own model says it sees (the analysis uses this module + lens)
    f = p.variant.by_tag('F')
    assert np.isfinite(res.best_dens).all() and res.per_cam[f.uid]['seen_m2'] > 0
    intr = resolve(p.module_for(f), p.lens_for(f))
    assert (intr.width, intr.height) == (MODS[lens.module_id].width_px, MODS[lens.module_id].height_px)
    if lens.focus_far_m < math.inf:                                 # an in-cabin lens: little of the ground in focus
        assert met['in_focus_pct'] < met['covered_pct']


def test_exports_read_each_modules_figures(engine_sample, tmp_path):
    from vantage.core import exporters as ex
    p0, e = engine_sample
    p = copy.deepcopy(p0)
    want = _mixed(p)
    res = e.analyse(p)
    ids = ['opencv_yaml', 'ros_camera_info', 'kalibr_camchain', 'kitti_calib', 'nuscenes_calib', 'jetson_json',
           'extrinsics_json']
    paths = ex.export(ids, p, res, ex.ExportOptions(out_dir=str(tmp_path)))
    assert paths and all(os.path.getsize(x) > 0 for x in paths)
    jet = json.load(open(next(x for x in paths if x.endswith('.json') and 'jetson' in os.path.basename(x))))
    cams = {c['tag']: c for c in jet['cameras']}
    for c in p.variant.cameras:
        if c.uid in want and c.visible:
            m = p.module_for(c)
            got = cams[c.tag]
            assert (got['module'], got['sensor'], got['interface']) == (m.name, m.sensor, m.interface)
            assert got['resolution'] == {'width': m.width_px, 'height': m.height_px}
            cm = ex.camera_model(p, c)
            assert cm.k == tuple(p.lens_for(c).kb) and cm.fit_rms_px == 0.0


# ----------------------------------------------------------------------------- the module editor and the library
def _editor(win, tag='F'):
    from vantage.ui.dialogs.module_editor import ModuleEditor
    c = win.store.variant.by_tag(tag)
    return ModuleEditor(win.ctx, c.module_id, c.lens_id, win)


def test_module_editor_lists_the_library_and_switches_cameras_to_a_preset(app, win):
    st = win.store
    p = st.project
    dlg = _editor(win)
    keys = dlg.preset.keys()
    assert keys == [l.id for l in builtin_lenses()] + ['manual'] and dlg.preset.value() == 'h05r2'
    # each module is a heading (not selectable) above its own lenses
    heads = [dlg.preset.itemText(i) for i in range(dlg.preset.count()) if dlg.preset.itemData(i) is None]
    assert heads == [f'{m.name} · {m.sensor}' for m in builtin_modules()]
    i88 = dlg.preset.findData('sturdecam88_h04r1')
    assert not dlg.preset.model().item(i88 - 2).isEnabled() and dlg.preset.itemText(i88 - 2).startswith('STURDeCAM88')
    assert 'e-consystems.com' in dlg.source.text() and 'Not given' not in dlg.source.text()
    users = [c.uid for v in p.variants for c in v.cameras if (c.module_id, c.lens_id) == ('sturdecam31', 'h05r2')]
    n = len(dlg.users)
    dlg.preset.setCurrentIndex(i88)
    dlg.preset.activated.emit(i88)
    assert (dlg.mod.id, dlg.lens.id) == ('sturdecam88', 'sturdecam88_h04r1')
    assert dlg.f['m_sensor'].text() == 'OmniVision OX08B40' and dlg.f['m_res'].text() == '3840 × 2160'
    assert dlg.f['m_size'].text() == '42 × 42 × —' and dlg.f['m_ip'].text() == 'IP67'
    assert dlg.f['m_temp'].text() == '−40 … 80' and dlg.f['m_shutter'].currentData() == 'rolling'
    assert dlg.ok.text() == f'Switch {n} cameras to STURDeCAM88 + H04R1'
    assert 'Not given: length (depth)' in dlg.source.text() and 'H04R1' in dlg.source.text()
    assert all(abs(float(dlg.cmp.item(i, 3).text().replace('−', '-'))) < 0.5 for i in range(3))
    assert dlg.der_v['Hyperfocal at F'][1].text() == '— (no aperture)'
    lib = {k: l.to_dict() for k, l in p.lenses.items()}
    dlg.ok.click()
    settle(app, win)
    p = st.project
    moved = [c for v in p.variants for c in v.cameras if c.uid in users]
    assert moved and all((c.module_id, c.lens_id) == ('sturdecam88', 'sturdecam88_h04r1') for c in moved)
    assert {k: l.to_dict() for k, l in p.lenses.items()} == lib            # the library entries are untouched
    assert p.lens_for(moved[0]).name == 'H04R1'
    st.undo.undo()
    p = st.project
    assert all((c.module_id, c.lens_id) == ('sturdecam31', 'h05r2')
               for v in p.variants for c in v.cameras if c.uid in users)
    st.undo.redo()
    assert st.project.variant.by_tag('F').lens_id == 'sturdecam88_h04r1'
    st.undo.undo()


def test_module_editor_not_given_fields_and_an_edit_of_a_builtin(app, win):
    st = win.store
    f = st.variant.by_tag('F')
    dlg = _editor(win)
    assert dlg.f['m_size'].text() == '25 × 25 × 43.8' and dlg.f['m_temp'].text() == '−40 … 85'
    dlg.f['m_size'].setText('25 × 25 × —')
    dlg.f['m_temp'].setText('')
    dlg.f['m_ip'].setText('IP69K')
    dlg.f['m_power'].set_value(1.324)
    dlg.f['m_power'].committed.emit(1.324)
    dlg.f['l_fno'].set_value(0.0)
    dlg.f['l_fno'].committed.emit(0.0)
    dlg.f['l_far'].setText('40')
    dlg.read_fields()
    assert dlg.mod.size_mm == (25.0, 25.0, 0.0) and dlg.mod.temp_c == () and dlg.mod.power_w == 1.324
    assert dlg.lens.f_number == 0.0 and dlg.lens.focus_far_m == 40.0 and dlg.preset.value() == 'manual'
    assert dlg.der_v['Hyperfocal at F'][1].text() == '— (no aperture)'
    assert 'length (depth)' in dlg.source.text() and 'operating temperature' in dlg.source.text()
    assert dlg.ok.text().startswith('Apply to')
    dlg.ok.click()
    settle(app, win)
    m = st.project.module_for(st.variant.by_tag('F'))
    assert (m.id, m.size_mm, m.power_w, m.temp_c) == ('sturdecam31', (25.0, 25.0, 0.0), 1.324, ())
    assert st.project.lenses[f.lens_id].f_number == 0.0
    assert m.body_mm == (25.0, 25.0, 43.8)
    st.undo.undo()
    assert st.project.modules['sturdecam31'].size_mm == (25.0, 25.0, 43.8)


def test_module_editor_fisheye_with_an_image_circle_and_an_import(app, win, tmp_path, monkeypatch):
    from PyQt6.QtWidgets import QFileDialog
    dlg = _editor(win)
    i = dlg.preset.findData('sturdecam31fsh_heated')
    dlg.preset.activated.emit(i)
    # the stated 210° D is the image circle: the comparison is good
    assert dlg.warn.text.text().startswith('The model matches the datasheet FOV')
    assert all(abs(float(dlg.cmp.item(k, 3).text().replace('−', '-'))) < 0.5 for k in range(3))
    assert dlg.der_v['FOV H · V · D'][0].text().endswith('circle')
    dlg.fit_sheet()                                  # refitting keeps the stated FOV (H and V; D is the circle)
    assert all(abs(float(dlg.cmp.item(k, 3).text().replace('−', '-'))) < 0.5 for k in range(3))
    assert dlg.lens.max_half_deg == 105.0 and dlg.der_v['FOV H · V · D'][0].text().endswith('circle')
    dlg.reject()
    # importing another camera's JSON does not carry e-con's source over to it
    path = str(tmp_path / 'other.json')
    with open(path, 'w') as fh:
        json.dump({'module': {'name': 'Other cam', 'width_px': 1280, 'height_px': 960, 'pixel_um': 3.75,
                              'temp_c': [-20, 60], 'shutter': 'global'},
                   'lens': {'name': 'Other lens', 'focal_mm': 2.1}}, fh)
    monkeypatch.setattr(QFileDialog, 'getOpenFileName', staticmethod(lambda *a, **k: (path, '')))
    from vantage.ui.dialogs.module_editor import ModuleEditor
    st = win.store
    n = len(st.project.modules)
    d2 = ModuleEditor(win.ctx, 'sturdecam31', 'h05r2', win, start_import=True)
    assert (d2.mod.name, d2.mod.vendor, d2.mod.note, d2.mod.url, d2.mod.ip_rating) == ('Other cam', '', '', '', '')
    assert (d2.mod.temp_c, d2.mod.shutter, d2.lens.note) == ((-20.0, 60.0), 'global', '')
    d2.apply()
    settle(app, win)
    assert len(st.project.modules) == n + 1
    new = next(m for m in st.project.modules.values() if m.name == 'Other cam')
    assert not new.builtin and new.note == '' and st.project.modules['sturdecam31'].note.startswith('Figures from')
    st.undo.undo()
    assert len(st.project.modules) == n


def test_library_panel_inspector_and_placing_use_every_module(app, win):
    from vantage.ui import ops
    from vantage.ui.outliner import ROLE_ID, ROLE_KIND
    st = win.store
    p = st.project
    lib = win.library
    lib.kind.setCurrentIndex(0)
    lib.rebuild()
    t = lib.tree
    groups = []
    for i in range(t.topLevelItemCount()):
        g = t.topLevelItem(i)
        groups.append([g.child(k).data(0, ROLE_ID) for k in range(g.childCount())
                       if g.child(k).data(0, ROLE_KIND) == 'lens'])
    # one group per module holding exactly that module's lenses, in library order
    for m in builtin_modules():
        assert [(m.id, l.id) for l in builtin_lenses() if l.module_id == m.id] in groups, m.id
    # the inspector offers every module and only the chosen module's lenses
    f = st.variant.by_tag('F')
    st.select_cams([f.uid])
    win.inspector.refresh()
    cam = win.inspector.pages['camera']
    assert [cam.module.itemData(i) for i in range(cam.module.count())] == list(p.modules)
    cam._module(cam.module.findData('sturdecam34'))
    settle(app, win)
    f = st.variant.by_tag('F')
    assert (f.module_id, f.lens_id) == ('sturdecam34', 'sturdecam34_h01r1')
    win.inspector.refresh()
    assert [cam.lens.itemData(i) for i in range(cam.lens.count())] == ['sturdecam34_h01r1']
    # a new camera placed with another module's lens (the library's "use for new cameras")
    win.ctx.place_lens = ('sturdecam88', 'sturdecam88_h02r1')
    n = len(st.variant.cameras)
    ops.add_camera(win.ctx, np.array([1.0, 0.0, 1.6]), np.array([0.0, 0.0, 1.0]))
    settle(app, win)
    assert len(st.variant.cameras) == n + 1
    c = st.variant.cameras[-1]
    assert (c.module_id, c.lens_id) == ('sturdecam88', 'sturdecam88_h02r1')
    assert win.ctx.analysis.result is not None and c.uid in win.ctx.analysis.result.per_cam


# ----------------------------------------------------------------------------- the library across the app
def test_library_search_and_folding(app, win):
    from vantage.ui.outliner import ROLE_ID
    lib = win.library
    lib.kind.setCurrentIndex(0)
    lib.rebuild()
    t = lib.tree
    top = [t.topLevelItem(i) for i in range(t.topLevelItemCount())]
    assert top[0].text(0) == 'STURDeCAM31 · ISX031' and top[0].isExpanded() and not top[1].isExpanded()
    assert 'IP69K' in top[0].toolTip(0) and 'e-consystems.com' in top[0].toolTip(0)
    assert 'as e-con states it' in top[0].child(0).toolTip(0) and 'F2' in top[0].child(0).toolTip(0)
    lib.search.setText('ox08')
    top = [t.topLevelItem(i) for i in range(t.topLevelItemCount())]
    assert [x.text(0) for x in top][:1] == ['STURDeCAM88 · OmniVision OX08B40'] and top[0].isExpanded()
    assert [top[0].child(k).data(0, ROLE_ID)[1] for k in range(top[0].childCount())] == ['sturdecam88_h02r1', 'sturdecam88_h04r1']
    lib.search.setText('')


def test_lens_menu_groups_by_module(app, win):
    st = win.store
    f = st.variant.by_tag('F')
    st.select_cams([f.uid])
    m = QMenu()
    win._fill_lens_menu(m, False)
    subs = [a.menu() for a in m.actions() if a.menu()]
    assert [s.title() for s in subs] == [f'{x.name} · {x.sensor}' for x in builtin_modules()]
    s88 = subs[[x.id for x in builtin_modules()].index('sturdecam88')]
    acts = s88.actions()
    assert [a.text() for a in acts] == ['H02R1 · narrow', 'H04R1 · wide']
    assert subs[0].actions()[0].isChecked()
    acts[1].trigger()
    settle(app, win)
    f = st.variant.by_tag('F')
    assert (f.module_id, f.lens_id) == ('sturdecam88', 'sturdecam88_h04r1')
    st.undo.undo()


def test_library_tool_and_lens_changes():
    from vantage.assistant import tools as T
    p = samples.tigor_sample_project()
    tc = T.ToolContext(project=p.snapshot(), result=None, engine=Engine())
    out, ok = T.run(tc, 'get_camera_library', {})
    d = json.loads(out)
    assert ok and len(d['modules']) == 17 and d['modules'][0]['id'] == 'sturdecam31'
    m88 = next(m for m in d['modules'] if m['id'] == 'sturdecam88')
    assert [l['id'] for l in m88['lenses']] == ['sturdecam88_h02r1', 'sturdecam88_h04r1'] and m88['not_given'] == ['length (depth)']
    out, ok = T.run(tc, 'get_camera_library', {'module': 'zzz'})
    assert not ok
    q = copy.deepcopy(p)
    rows, targets, before, errors = T.apply_changes(q, [{'camera': 'F', 'lens': 'sturdecam88_h04r1', 'mirror_partner': False}])
    f = q.variant.by_tag('F')
    assert not errors and (f.module_id, f.lens_id) == ('sturdecam88', 'sturdecam88_h04r1')
    f.module_id, f.lens_id = 'nilecam85', 'nilecam85_std'
    rows, targets, before, errors = T.apply_changes(q, [{'camera': 'F', 'lens': '131° DFOV', 'mirror_partner': False}])
    assert not errors and (f.module_id, f.lens_id) == ('nilecam85', 'nilecam85_std')


def test_body_mm_in_the_3d_model():
    from vantage.ui import overlays3d as O3
    import numpy as np
    p = samples.tigor_sample_project()
    cam = p.variant.cameras[0]
    cam.module_id, cam.lens_id = 'sturdecam34', 'sturdecam34_h01r1'
    cam.yaw = cam.pitch = cam.roll = 0.0
    T_, N_, C_ = O3.camera_parts(p, cam, True)
    q = T_.reshape(-1, 3) - cam.pos()
    assert q[:, 0].min() == __import__('pytest').approx(-0.0438)
