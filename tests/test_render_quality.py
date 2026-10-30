"""Image quality of the 3D view: the shader renderer's tone curve, data colours, sky occlusion, point sprites, outlines,
glass and frame time (run headless through EGL in a subprocess, skipped without EGL OpenGL), and the software
(QPainter) renderer's points, outlines and studio backdrop."""
import json
import os
import subprocess
import sys
import tempfile

os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ.setdefault('XDG_CONFIG_HOME', tempfile.mkdtemp(prefix='vantage-test-cfg-'))

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------- shader renderer (EGL subprocess)
@pytest.fixture(scope='module')
def gl(tmp_path_factory):
    out = tmp_path_factory.mktemp('glq')
    r = subprocess.run([sys.executable, os.path.join(HERE, 'gl_harness.py'), str(out), '--quality'],
                       capture_output=True, text=True, timeout=600)
    lines = [ln for ln in r.stdout.splitlines() if ln.startswith('{')]
    if not lines:
        pytest.fail(f'harness failed: {r.stderr[-3000:]}')
    doc = json.loads(lines[-1])
    if 'skip' in doc:
        pytest.skip(f'no headless OpenGL: {doc["skip"]}')
    return doc


def test_every_shader_compiles_and_links(gl):
    assert gl['init'], gl['error']
    assert {'mesh', 'ground', 'flat', 'bowl', 'depth', 'point', 'outline', 'probe', 'contact'} <= set(gl['programs'])
    assert gl['gl_error_end'] == 0
    assert gl['pixel_store'] == [4, 4]


def test_tone_curve_keeps_mid_grey_and_matches_the_reference(gl):
    assert gl['tone_mid_grey'] in (127, 128, 129)            # sRGB 0.5 in, 0.5 out
    assert gl['tone_identity_err'] <= 1                      # nothing changes below the knee
    assert gl['tone_max_err'] <= 1                           # GPU == tone_reference (to 1/255)
    assert gl['tone_monotonic'] and gl['tone_8x'] == 255     # highlights roll off, never fold back
    assert gl['tone_hue_kept']                               # a bright orange stays orange


def test_floor_grid_and_axes_from_the_ground_shader(gl):
    """The grid lies on the 0.5 m and 1 m marks counted from the display origin (major lines brighter), the floor in
    between is untouched, the axes are red (x) and green (y), and with the grid off nothing changes."""
    assert gl['grid_major'] > 40 and gl['grid_minor'] > 15 and gl['grid_major'] > gl['grid_minor']
    assert gl['grid_between'] <= 2
    assert gl['grid_off_same'] == 0
    assert gl['axis_x_red'] and gl['axis_y_green']


def test_tone_reference_shoulder():
    from vantage.ui.render_gl import tone_reference, to_srgb, to_linear, TONE_KNEE
    grey = to_linear([0.5, 0.5, 0.5])
    assert np.allclose(tone_reference(grey), grey)
    hi = tone_reference(np.array([[1.0, 1.0, 1.0], [3.0, 3.0, 3.0], [50.0, 50.0, 50.0]]))[:, 0]
    assert TONE_KNEE < hi[0] < hi[1] < hi[2] <= 1.0
    assert abs(float(to_srgb(to_linear(0.3))) - 0.3) < 1e-9


def test_data_colours_on_the_ground_stay_exact(gl):
    """Coverage-like ground texture and the plane colour are not tone-mapped or tinted (shadows and sky occlusion on,
    the car in the scene)."""
    assert gl['data_colour_err'] <= 2
    assert all(abs(v - 128) <= 1 for v in gl['plane_mid_grey'])


def test_camera_light_colours_are_the_legend_formula(gl):
    assert gl['camlight_colour_n'] > 100
    assert gl['camlight_colour_err'] <= 3


def test_sky_occlusion_darkens_a_crevice_not_an_open_panel(gl):
    assert gl['ao_crevice_ratio'] < 0.93                     # slab next to the wall darker than the open slab
    assert gl['noao_crevice_ratio'] > 0.97                   # ... only because of the occlusion
    assert gl['ao_open_kept'] > 0.95                         # the open slab keeps its brightness


def test_point_sprites_are_depth_tested_round_and_soft(gl):
    assert gl['point_front_green']
    assert gl['point_behind_hidden']                         # behind the body: not drawn
    assert gl['point_through_magenta']                       # depth=False: drawn over it
    assert 7 <= gl['point_size_px'] <= 11                    # size 9 px (device pixel ratio 1)
    assert gl['point_soft_edge'] > 0                         # anti-aliased rim


def test_point_sprites_off_screen_are_premultiplied(gl):
    """Stills and clips composite the off-screen layer over the page: soft rims must not overflow into odd colours."""
    assert gl['point_premultiplied_excess'] <= 1
    assert gl['point_light_bg_hue_kept']


def test_outline_around_a_selected_tri_draw(gl):
    for tag in ('stencil', 'nostencil'):
        assert gl[f'outline_{tag}_rim'], tag
        assert gl[f'outline_{tag}_inside_clear'], tag
        assert gl[f'outline_{tag}_outside_clear'], tag
    assert gl['stencil_left_clear']


def test_fixed_function_path_draws_points_and_outlines(gl):
    if 'legacy_outline' not in gl:
        pytest.skip('no QOpenGLWidget')
    assert gl['legacy_outline'] and gl['legacy_point']


def test_glass_is_dark_and_a_little_see_through(gl):
    assert gl['glass_red_through'] > 8                       # the red behind shows through the glass
    assert gl['paint_red_through'] < 4                       # not through paint
    assert gl['glass_dark'] < 110


def test_sky_maps_of_huge_meshes_build_over_frames(gl):
    assert gl['sky_progressive_first_pending']
    assert gl['sky_progressive_frames'] <= 26


def test_frame_time_at_1600x900(gl):
    print(f"\nframe {gl['frame_ms']} ms, first (maps built) {gl['frame_first_ms']} ms on {gl['renderer']}")
    assert gl['frame_ms'] < 120 and gl['frame_first_ms'] < 2000


# ---------------------------------------------------------------------------- software renderer (QPainter)
@pytest.fixture(scope='module')
def qapp():
    from PyQt6.QtWidgets import QApplication
    a = QApplication.instance() or QApplication([])
    from vantage.ui.theme import THEME
    THEME.apply(a, 'dark')
    return a


def _box(lo, hi):
    sys.path.insert(0, HERE)
    from gl_harness import box
    return box(lo, hi)


class _Provider:
    """A minimal scene: a grey box, optional points and an outlined box on the ground."""

    def __init__(self, points=(), tris=(), style=None, bg_image=None):
        from PyQt6.QtCore import QObject, pyqtSignal

        class _S(QObject):
            changed = pyqtSignal()
        self._sig = _S()
        self.changed = self._sig.changed
        from vantage.ui.viewport3d import MeshDraw
        T, N = _box([0, -0.5, 0], [1, 0.5, 1])
        self._mesh = MeshDraw(T, N, np.tile([0.7, 0.7, 0.7], (12, 1)), np.ones(12))
        self._points, self._tris, self._style, self._bg = list(points), list(tris), style, bg_image

    def background_color(self):
        from PyQt6.QtGui import QColor
        return QColor('#1a1a19')

    def background_image(self, vp):
        return self._bg

    def mesh(self):
        return self._mesh

    def ground(self):
        return None

    def lines(self, vp):
        return []

    def tris_extra(self, vp):
        return self._tris

    def points(self, vp):
        return self._points

    def render_style(self, vp):
        return self._style

    def overlay(self, p, vp):
        pass

    def hud_label(self, vp):
        return ''


def _soft(provider, az=0.0, el=12.0, target=(0.5, 0.0, 0.5), dist=6.0, ortho=False, h=3.0):
    from vantage.ui.viewport3d import SoftViewport
    vp = SoftViewport(provider)
    vp.resize(480, 320)
    vp.cam.target, vp.cam.az, vp.cam.el, vp.cam.dist = np.array(target, float), az, el, dist
    vp.cam.ortho, vp.cam.ortho_h = ortho, h
    return vp


def _grab(vp):
    from PyQt6.QtGui import QImage
    img = vp.grab().toImage().convertToFormat(QImage.Format.Format_RGBA8888)
    ptr = img.constBits()
    ptr.setsize(img.sizeInBytes())
    return np.frombuffer(ptr, np.uint8).reshape(img.height(), img.bytesPerLine())[:, :img.width() * 4] \
        .reshape(img.height(), img.width(), 4).copy()


def _at(img, vp, X, rad=1):
    (x, y), = vp.project([X])[0]
    x, y = int(round(x)), int(round(y))
    return img[y - rad:y + rad + 1, x - rad:x + rad + 1, :3].reshape(-1, 3).mean(axis=0)


def test_software_points_drawn_and_hidden_behind_the_body(qapp):
    from vantage.ui.viewport3d import PointDraw
    front, behind, through = [1.6, 0.0, 0.5], [-0.6, 0.0, 0.5], [-0.6, 0.25, 0.65]
    pts = [PointDraw(np.array([front, behind], np.float32), np.array([[0.1, 0.9, 0.2, 1]] * 2, np.float32), size=8),
           PointDraw(np.array([through], np.float32), np.array([[0.9, 0.1, 0.9, 1]], np.float32), size=8, depth=False)]
    vp = _soft(_Provider(points=pts))
    img = _grab(vp)
    f, b, t = _at(img, vp, front), _at(img, vp, behind), _at(img, vp, through)
    assert f[1] - f[0] > 80                                  # drawn where projected
    assert b[1] - b[0] < 20                                  # behind the box: painted over
    assert t[0] - t[1] > 80 and t[2] - t[1] > 80             # depth=False: on top


def test_software_points_are_capped(qapp):
    from vantage.ui import viewport3d as V3
    n = V3.SOFT_POINT_CAP * 3
    rng = np.random.default_rng(0)
    P = np.c_[rng.uniform(-4, 4, n), rng.uniform(-4, 4, n), rng.uniform(0, 2, n)].astype(np.float32)
    C = np.tile([0.2, 0.6, 0.9, 1.0], (n, 1)).astype(np.float32)
    vp = _soft(_Provider(points=[V3.PointDraw(P, C, 2.0)]))
    slabs = vp._point_slabs([V3.PointDraw(P, C, 2.0)])
    assert sum(len(g[2]) for _, groups in slabs for g in groups) <= V3.SOFT_POINT_CAP
    assert [d for d, _ in slabs] == sorted([d for d, _ in slabs], reverse=True)     # far to near
    _grab(vp)                                                # and it paints


def test_software_outline_around_the_silhouette(qapp):
    from vantage.ui.viewport3d import TriDraw
    T, N = _box([2.0, -0.5, 0.0], [3.0, 0.5, 1.0])
    sel = TriDraw(T, np.tile([0.45, 0.45, 0.48, 1.0], (12, 1)).astype(np.float32), outline=(1.0, 0.85, 0.1, 1.0))
    vp = _soft(_Provider(tris=[sel]), az=-90.0, el=89.99, target=(2.5, 0.0, 0.0), ortho=True, h=3.0)
    img = _grab(vp)
    (ex, ey), = vp.project([[3.0, 0.0, 1.0]])[0]
    ex, ey = int(round(ex)), int(round(ey))
    yel = lambda v: v[0] > 180 and v[1] > 140 and v[2] < 100
    assert any(yel(img[ey, ex + k, :3].astype(int)) for k in (-1, 0, 1))
    assert not yel(img[ey, ex - 8, :3].astype(int)) and not yel(img[ey, ex + 8, :3].astype(int))


def test_backdrop_horizon_gradient_and_dither():
    from vantage.ui.viewport3d import backdrop_image, backdrop_colors
    z, hz, fl = backdrop_colors((0.10, 0.10, 0.10), (0.18, 0.18, 0.17))
    assert hz.mean() > fl.mean() and hz.mean() > z.mean()    # a bright haze at the horizon
    img = backdrop_image(400, 150, z, hz, fl)
    ptr = img.constBits()
    ptr.setsize(img.sizeInBytes())
    a = np.frombuffer(ptr, np.uint8).reshape(400, 8, 4)[..., :3].astype(float)
    rows = a.mean(axis=(1, 2)) / 255
    assert abs(rows[399] - fl.mean()) < 0.01                 # the floor below
    assert abs(rows[150] - hz.mean()) < 0.01                 # the haze at the horizon row
    blocks = rows[:144].reshape(-1, 8).mean(axis=1)          # one dither period per block
    assert np.all(np.diff(blocks) >= -1e-3)                  # brightening down to the horizon, no steps back
    assert len(np.unique(a[100:110, :, 0])) >= 2             # dithered: no flat 8-bit bands
    flat = backdrop_image(50, None, z, hz, fl)               # straight down: all floor
    ptr = flat.constBits()
    ptr.setsize(flat.sizeInBytes())
    f = np.frombuffer(ptr, np.uint8).reshape(50, 8, 4)[..., :3].astype(float) / 255
    assert np.abs(f.mean(axis=(0, 1)) - fl).max() < 0.01


def test_backdrop_only_behind_perspective_views_with_a_ground_plane(qapp):
    from PyQt6.QtGui import QImage
    from vantage.ui.render_gl import Style
    vp = _soft(_Provider(style=Style()), az=30.0, el=8.0)
    assert vp._backdrop_brush() is not None
    hy = vp.horizon_y()
    assert 0 < hy < vp.height()                              # a low view sees the horizon
    img = _grab(vp)
    assert img[5, 5, :3].astype(int).sum() != img[vp.height() - 5, 5, :3].astype(int).sum()
    vp.cam.el, vp.cam.ortho = 89.99, True                    # top view: flat
    vp.cam.az = -90.0
    assert vp.horizon_y() is None or vp._backdrop_brush() is not None
    img = _grab(vp)
    assert np.abs(img[5, 5, :3].astype(int) - img[vp.height() - 5, 5, :3].astype(int)).max() <= 2
    assert _soft(_Provider(style=Style(ground_plane=False)))._backdrop_brush() is None     # photo / lighting off
    photo = QImage(16, 16, QImage.Format.Format_RGB32)
    photo.fill(0xff0000)
    from PyQt6.QtCore import QRectF
    pv = _soft(_Provider(style=Style(), bg_image=(photo, QRectF(0, 0, 480, 320))))
    corner = _grab(pv)[5, 5, :3].astype(int)
    assert corner[0] > 200 and corner[1] < 40                # the reference photo, not the backdrop
