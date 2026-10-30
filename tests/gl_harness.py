"""Run the shader renderer in a headless EGL context and report what it drew (used by test_render_gl.py and
test_render_quality.py in a subprocess, because PyOpenGL picks its platform once per process).

    python3 tests/gl_harness.py OUT_DIR              prints one JSON line; 'skip' when no EGL/OpenGL is available
    python3 tests/gl_harness.py OUT_DIR --quality    the image-quality checks (tone curve, data colours, sky
                                                     occlusion, points, outlines, glass, frame time)
"""
import ctypes
import json
import os
import sys

os.environ['PYOPENGL_PLATFORM'] = 'egl'
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))


def egl_context():
    from OpenGL import EGL
    dpy = EGL.eglGetDisplay(EGL.EGL_DEFAULT_DISPLAY)
    if not dpy or not EGL.eglInitialize(dpy, None, None):
        raise RuntimeError('no EGL display')
    attrs = (EGL.EGLint * 13)(EGL.EGL_SURFACE_TYPE, EGL.EGL_PBUFFER_BIT, EGL.EGL_RED_SIZE, 8, EGL.EGL_GREEN_SIZE, 8,
                             EGL.EGL_BLUE_SIZE, 8, EGL.EGL_DEPTH_SIZE, 24, EGL.EGL_RENDERABLE_TYPE, EGL.EGL_OPENGL_BIT,
                             EGL.EGL_NONE)
    cfg, n = EGL.EGLConfig(), EGL.EGLint()
    if not EGL.eglChooseConfig(dpy, attrs, ctypes.pointer(cfg), 1, ctypes.pointer(n)) or n.value < 1:
        raise RuntimeError('no EGL config')
    surf = EGL.eglCreatePbufferSurface(dpy, cfg, (EGL.EGLint * 5)(EGL.EGL_WIDTH, 16, EGL.EGL_HEIGHT, 16, EGL.EGL_NONE))
    EGL.eglBindAPI(EGL.EGL_OPENGL_API)
    ctx = EGL.eglCreateContext(dpy, cfg, EGL.EGL_NO_CONTEXT, None)
    if not ctx or not EGL.eglMakeCurrent(dpy, surf, surf, ctx):
        raise RuntimeError('no EGL OpenGL context')


def main(out_dir):
    try:
        egl_context()
    except Exception as e:
        print(json.dumps({'skip': str(e)}))
        return
    import numpy as np
    from OpenGL import GL
    from PyQt6.QtGui import QGuiApplication
    app = QGuiApplication([])
    from vantage.core import samples, bowl as bowl_core
    from vantage.core.analysis import Engine
    from vantage.core.visibility import cam_views, seen_by
    from vantage.ui.render_gl import GLRenderer, Frame, Style, BowlDraw
    from vantage.ui.scene import mesh_colors
    from vantage.ui.viewport3d import OrbitCam, MeshDraw
    out = {'renderer': GL.glGetString(GL.GL_RENDERER).decode(errors='replace')}
    p = samples.tigor_sample_project()
    e = Engine()
    res = e.analyse(p)
    m = p.vehicle.mesh
    base, alpha, keep = mesh_colors(m, 'solid')
    md = MeshDraw(m.tri[keep].astype(np.float32), m.normals[keep], base[keep], alpha[keep], version=1)
    r = GLRenderer()
    out['init'] = r.init()
    out['error'] = r.error
    if not r.ok:
        print(json.dumps(out))
        return
    W, H = 640, 400
    bg = (0.12, 0.12, 0.13, 1.0)

    def frame(cam, **kw):
        return Frame(cam.view(), cam.proj(W / H), W, H, target=cam.target.copy(), clear=bg, **kw)
    cam = OrbitCam()
    cam.target, cam.dist = np.array([1.2, 0, 0.4]), 9.0
    lit = r.render_offscreen(frame(cam, mesh=md))
    flat = r.render_offscreen(frame(cam, mesh=md, style=Style(shadows=False, ao=False)))
    np.save(os.path.join(out_dir, 'lit.npy'), lit)
    out['gl_error'] = int(GL.glGetError())
    # the car is drawn, and the ground under and beside it is darker with shadows and contact occlusion on
    top = OrbitCam()
    top.target, top.ortho, top.az, top.el, top.ortho_h = np.array([1.2, 0, 0]), True, -90.0, 89.99, 9.0
    gt_lit = r.render_offscreen(frame(top, mesh=None, style=Style()))      # shadow maps kept from the mesh above
    mid = lit[H // 3:2 * H // 3, W // 3:2 * W // 3, :3].astype(int)
    out['car_pixels'] = float((mid.mean(axis=2) > 150).mean())             # the light grey body
    out['shadow_darker'] = float(flat[..., :3].mean() - lit[..., :3].mean())
    # camera light on the ground matches the CPU visibility test (lens model + occlusion)
    c = p.variant.by_tag('ML')
    views = cam_views(p, e, [c], [(1.0, 0.0, 0.0)], cube='build')
    no_light = r.render_offscreen(frame(top, style=Style(shadows=False, ao=False)))
    light = r.render_offscreen(frame(top, cams=views, style=Style(shadows=False, ao=False)))
    np.save(os.path.join(out_dir, 'camlight.npy'), light)
    rng = np.random.default_rng(0)
    pts = np.c_[rng.uniform(-3.5, 6.0, 3000), rng.uniform(-4.5, 4.5, 3000), np.zeros(3000)]
    V, P = top.view(), top.proj(W / H)
    q = np.c_[pts, np.ones(len(pts))] @ (P @ V).T
    sx = ((q[:, 0] / q[:, 3] + 1) * 0.5 * W).astype(int)
    sy = ((1 - q[:, 1] / q[:, 3]) * 0.5 * H).astype(int)
    ok = (sx >= 2) & (sx < W - 2) & (sy >= 2) & (sy < H - 2)
    cpu = seen_by(pts, views)[:, 0]
    a, b = light[sy[ok], sx[ok], :3].astype(int), no_light[sy[ok], sx[ok], :3].astype(int)
    gpu = (a[:, 0] - a[:, 1]) > (b[:, 0] - b[:, 1]) + 12                  # tinted red
    out['camlight_agree'] = float((gpu == cpu[ok]).mean())
    out['camlight_seen'] = int(cpu[ok].sum())
    # the textured bowl
    shape = bowl_core.BowlShape()
    mesh = bowl_core.bowl_mesh(res, shape)
    tex = np.zeros((64, 256, 4), np.uint8)
    tex[..., 1] = 200
    tex[..., 3] = 255
    bd = BowlDraw(mesh.vertices, mesh.uv, mesh.faces, tex)
    bw = r.render_offscreen(frame(cam, bowl=bd, style=Style(ground_plane=False, shadows=False, ao=False)))
    np.save(os.path.join(out_dir, 'bowl.npy'), bw)
    out['bowl_green'] = float((bw[..., 1].astype(int) - bw[..., 0].astype(int) > 100).mean())
    out['pixel_store'] = [int(GL.glGetIntegerv(GL.GL_UNPACK_ALIGNMENT)), int(GL.glGetIntegerv(GL.GL_PACK_ALIGNMENT))]
    # translucent flat triangles with per-vertex normals (the blind volume): a red ground quad, lit smoothly
    from vantage.ui.viewport3d import TriDraw
    quad = np.array([[[-1, -1, 0.5], [3, -1, 0.5], [3, 1, 0.5]], [[-1, -1, 0.5], [3, 1, 0.5], [-1, 1, 0.5]]], np.float32)
    nrm = np.tile(np.array([0, 0, 1], np.float32), (2, 3, 1))
    col = np.array([[0.9, 0.1, 0.1, 0.6]] * 2, np.float32)
    sm = r.render_offscreen(frame(top, tris=[TriDraw(quad, col, normals=nrm)], style=Style(shadows=False, ao=False)))
    np.save(os.path.join(out_dir, 'smooth.npy'), sm)
    mid = sm[H // 2 - 20:H // 2 + 20, W // 2 - 20:W // 2 + 20, :3].astype(int)
    out['smooth_red'] = float((mid[..., 0] - mid[..., 1] > 60).mean())
    out['smooth_flat'] = float(mid[..., 0].std())             # one normal: no faceting across the two triangles
    out['gl_error_end'] = int(GL.glGetError())
    r.release()
    print(json.dumps(out))


# ---------------------------------------------------------------------------- quality checks
def box(lo, hi):
    """(12,3,3) triangles and (12,3) outward normals of an axis-aligned box."""
    import numpy as np
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    c = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    F = [[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4],
         [1, 5, 7], [1, 7, 3]]
    T = c[F]
    n = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    out = (T.mean(axis=1) - (lo + hi) / 2)
    n *= np.sign(np.sum(n * out, axis=1))[:, None]
    return T.astype(np.float32), n.astype(np.float32)


def project(cam, W, H, X):
    import numpy as np
    X = np.atleast_2d(np.asarray(X, float))
    q = np.c_[X, np.ones(len(X))] @ (cam.proj(W / H) @ cam.view()).T
    return np.stack([(q[:, 0] / q[:, 3] + 1) * 0.5 * W, (1 - q[:, 1] / q[:, 3]) * 0.5 * H], 1)


def quality(out_dir):
    try:
        egl_context()
    except Exception as e:
        print(json.dumps({'skip': str(e)}))
        return
    import time
    import numpy as np
    from OpenGL import GL
    from PyQt6.QtGui import QGuiApplication, QImage
    app = QGuiApplication([])
    from vantage.core import samples
    from vantage.core.analysis import Engine
    from vantage.core.visibility import cam_views, seen_by
    from vantage.ui import render_gl as RG
    from vantage.ui.render_gl import GLRenderer, Frame, Style
    from vantage.ui.scene import mesh_colors
    from vantage.ui.viewport3d import OrbitCam, MeshDraw, TriDraw, PointDraw, GroundDraw, mesh_materials
    out = {'renderer': GL.glGetString(GL.GL_RENDERER).decode(errors='replace')}
    r = GLRenderer()
    out['init'] = r.init()
    out['error'] = r.error
    out['programs'] = sorted(r.prog)
    if not r.ok:
        print(json.dumps(out))
        return
    W, H = 480, 320
    bg = (0.1, 0.1, 0.1, 1.0)

    def cam_at(target, dist, az, el, ortho=False, h=4.0):
        c = OrbitCam()
        c.target, c.dist, c.az, c.el, c.ortho, c.ortho_h = np.asarray(target, float), dist, az, el, ortho, h
        return c

    def frame(cam, w=W, h=H, **kw):
        return Frame(cam.view(), cam.proj(w / h), w, h, target=cam.target.copy(), clear=bg, ortho=cam.ortho, **kw)

    def px(img, cam, X, rad=1):
        (x, y), = project(cam, W, H, X)
        x, y = int(round(x)), int(round(y))
        return img[y - rad:y + rad + 1, x - rad:x + rad + 1, :3].reshape(-1, 3).mean(axis=0)

    # 1. the tone curve on the GPU: a mid grey stays a mid grey, below the knee nothing changes, highlights roll off
    lin = np.r_[np.linspace(0, 1.0, 40), [1.5, 2.0, 4.0, 8.0]]
    vals = np.c_[lin, lin, lin]
    rng = np.random.default_rng(1)
    cols = rng.uniform(0, 1.6, (20, 3))
    gpu = r.tone_probe(np.r_[vals, cols])
    ref = np.round(RG.to_srgb(RG.tone_reference(np.r_[vals, cols])) * 255)
    out['tone_max_err'] = float(np.abs(gpu.astype(float) - ref).max())
    out['tone_mid_grey'] = int(r.tone_probe(RG.to_linear([[0.5, 0.5, 0.5]]))[0, 0])
    below = lin <= RG.TONE_KNEE
    out['tone_identity_err'] = float(np.abs(gpu[:len(lin)][below, 0] - np.round(RG.to_srgb(lin[below]) * 255)).max())
    g = gpu[:len(lin), 0].astype(int)
    out['tone_monotonic'] = bool(np.all(np.diff(g) >= 0))
    out['tone_8x'] = int(g[-1])
    sat = r.tone_probe([[3.0, 1.2, 0.3]])[0].astype(int)                 # a bright orange stays orange
    out['tone_hue_kept'] = bool(sat[0] > sat[1] > sat[2])

    # 2. data colours on the ground are not tone-mapped: a coverage-like texture and the plane colour come out exact,
    #    with shadows, contact shadow and sky occlusion on and the car in the scene
    p = samples.tigor_sample_project()
    m = p.vehicle.mesh
    base, alpha, keep = mesh_colors(m, 'solid')
    car = MeshDraw(m.tri[keep].astype(np.float32), m.normals[keep], base[keep], alpha[keep], version=1,
                   materials=mesh_materials(m)[keep])
    patches = np.array([[220, 40, 40], [40, 200, 60], [50, 90, 230], [240, 200, 30], [128, 128, 128], [20, 20, 20]],
                       np.uint8)
    img = np.zeros((1, len(patches), 4), np.uint8)
    img[0, :, :3] = patches
    img[0, :, 3] = 255
    qimg = QImage(img.tobytes(), len(patches), 1, len(patches) * 4, QImage.Format.Format_RGBA8888).copy()
    gd = GroundDraw(qimg, (-3.0, 6.0, 3.0, 8.0), 0.002, nearest=True, version=7)
    top = cam_at([0.0, 7.0, 0.0], 9.0, -90.0, 89.99, ortho=True, h=6.0)
    mid = (0.5, 0.5, 0.5)
    st = Style(ground_rgb=mid)
    shot = r.render_offscreen(frame(top, mesh=car, ground=gd, style=st))
    xs = -3.0 + (np.arange(len(patches)) + 0.5) * 1.0
    got = np.array([px(shot, top, [x, 7.0, 0.0]) for x in xs])
    out['data_colour_err'] = float(np.abs(got - patches).max())
    out['plane_mid_grey'] = px(shot, top, [0.0, 9.3, 0.0]).round(1).tolist()

    # 3. camera-light colours on the ground: exactly the legend formula (mix toward the camera colour)
    e = Engine()
    c = p.variant.by_tag('ML')
    views = cam_views(p, e, [c], [(1.0, 0.0, 0.0)], cube='build')
    topc = cam_at([1.2, 0.0, 0.0], 9.0, -90.0, 89.99, ortho=True, h=9.0)
    gr = np.array(Style().ground_rgb)
    light = r.render_offscreen(frame(topc, cams=views, style=Style(shadows=False, ao=False)))
    pts = np.c_[rng.uniform(-3.5, 6.0, 4000), rng.uniform(-4.5, 4.5, 4000), np.zeros(4000)]
    seen = seen_by(pts, views)[:, 0]
    lum = float(gr @ [0.299, 0.587, 0.114])
    expect = (gr + (np.array([1.0, 0, 0]) * (0.30 + 1.05 * lum) - gr) * 0.6) * 255
    errs = []
    for q, sflag in zip(pts, seen):
        if not sflag:
            continue
        (x, y), = project(topc, W, H, q)
        x, y = int(x), int(y)
        if 3 <= x < W - 3 and 3 <= y < H - 3:
            nb = seen_by(np.array([q + [dx, dy, 0] for dx in (-0.06, 0.06) for dy in (-0.06, 0.06)]), views)[:, 0]
            if nb.all():                                    # well inside the seen area
                errs.append(np.abs(light[y, x, :3] - expect).max())
    out['camlight_colour_err'] = float(np.median(errs)) if errs else 99.0
    out['camlight_colour_n'] = len(errs)

    # 4. sky occlusion: a crevice (a slab meeting a wall) is darker than the open slab; without AO it is not
    s1, n1 = box([-1.5, -1.5, 0.4], [1.5, 1.5, 0.5])
    s2, n2 = box([-0.05, -1.5, 0.5], [0.05, 1.5, 1.1])
    T = np.r_[s1, s2]
    N = np.r_[n1, n2]
    grey = np.tile([0.6, 0.6, 0.6], (len(T), 1))
    slab = MeshDraw(T, N, grey, np.ones(len(T)), version=2)
    vc = cam_at([0.6, 0.0, 0.5], 4.0, 0.0, 55.0)
    crev, opn = [0.09, 0.0, 0.5], [1.25, 0.0, 0.5]
    a_on = r.render_offscreen(frame(vc, mesh=slab, style=Style(shadows=False, ao=True)))
    a_off = r.render_offscreen(frame(vc, mesh=slab, style=Style(shadows=False, ao=False)))
    np.save(os.path.join(out_dir, 'ao_on.npy'), a_on)
    lum3 = lambda v: float(np.mean(v))
    out['ao_crevice_ratio'] = lum3(px(a_on, vc, crev, 2)) / max(lum3(px(a_on, vc, opn, 2)), 1)
    out['noao_crevice_ratio'] = lum3(px(a_off, vc, crev, 2)) / max(lum3(px(a_off, vc, opn, 2)), 1)
    out['ao_open_kept'] = lum3(px(a_on, vc, opn, 2)) / max(lum3(px(a_off, vc, opn, 2)), 1)

    # 5. point sprites: drawn where projected, hidden behind the body, shown through it with depth=False
    cube_t, cube_n = box([0, -0.5, 0], [1, 0.5, 1])
    cube = MeshDraw(cube_t, cube_n, np.tile([0.7, 0.7, 0.7], (12, 1)), np.ones(12), version=3)
    pc = cam_at([0.5, 0.0, 0.5], 6.0, 0.0, 12.0)
    front, behind, through = [1.6, 0.0, 0.5], [-0.6, 0.0, 0.5], [-0.6, 0.25, 0.65]
    green = PointDraw(np.array([front, behind], np.float32), np.array([[0.1, 0.9, 0.2, 1]] * 2, np.float32), size=9.0)
    magenta = PointDraw(np.array([through], np.float32), np.array([[0.9, 0.1, 0.9, 1]], np.float32), size=9.0,
                        depth=False)
    pimg = r.render_offscreen(frame(pc, mesh=cube, points=[green, magenta], style=Style(shadows=False)))
    np.save(os.path.join(out_dir, 'points.npy'), pimg)
    f_, b_, t_ = px(pimg, pc, front), px(pimg, pc, behind), px(pimg, pc, through)
    out['point_front_green'] = bool(f_[1] - f_[0] > 80)
    out['point_behind_hidden'] = bool(b_[1] - b_[0] < 20)
    out['point_through_magenta'] = bool(t_[0] - t_[1] > 80 and t_[2] - t_[1] > 80)
    (x0, y0), = project(pc, W, H, front)
    blob = (pimg[..., 1].astype(int) - pimg[..., 0] > 80)
    ys, xs_ = np.nonzero(blob[int(y0) - 12:int(y0) + 13, int(x0) - 12:int(x0) + 13])
    out['point_size_px'] = float(max(xs_.max() - xs_.min() + 1, ys.max() - ys.min() + 1)) if len(xs_) else 0.0
    edge = pimg[int(y0) - 12:int(y0) + 13, int(x0) - 12:int(x0) + 13, 1].astype(int)
    out['point_soft_edge'] = int(((edge > 40) & (edge < 200)).sum())    # anti-aliased: partial pixels on the rim
    # off-screen on a transparent background (stills, Render image, clips): premultiplied output, so red and yellow
    # points composited over a light page keep their colour (no rgb > alpha overflow on the soft rims)
    many = np.array([[1.6, y, z] for y in np.linspace(-0.8, 0.8, 9) for z in np.linspace(0.2, 1.0, 5)], np.float32)
    rgb = np.where((np.arange(len(many)) % 2 == 0)[:, None], [[1.0, 0.1, 0.05, 1.0]], [[1.0, 0.9, 0.1, 1.0]])
    clear = r.render_offscreen(Frame(pc.view(), pc.proj(W / H), W, H, target=pc.target.copy(), clear=(0, 0, 0, 0),
                                     points=[PointDraw(many, rgb.astype(np.float32), size=9.0)],
                                     style=Style(shadows=False, ao=False, ground_plane=False)))
    a = clear[..., 3:4].astype(int)
    out['point_premultiplied_excess'] = int((clear[..., :3].astype(int) - a).max())     # <= 0: valid premultiplied
    over = clear[..., :3].astype(float) + 245.0 * (1 - a / 255.0)       # composite over a light background
    xy = project(pc, W, H, many)
    hues = []
    for (x, y), c in zip(xy, rgb):
        v = over[int(round(y)) - 4:int(round(y)) + 5, int(round(x)) - 4:int(round(x)) + 5].reshape(-1, 3)
        hues.append(bool((v[:, 0] >= v[:, 2] - 2).all() and (v[:, 0] >= v[:, 1] - 2).all()))   # red dominant: no cyan/magenta rims
    out['point_light_bg_hue_kept'] = all(hues)

    # 6. outlines: a selected box gets a rim of the outline colour just outside its silhouette, with the stencil
    #    buffer and without it
    oc = cam_at([0.5, 0.0, 0.0], 9.0, -90.0, 89.99, ortho=True, h=3.0)
    sel = TriDraw(cube_t, np.tile([0.45, 0.45, 0.48, 1.0], (12, 1)).astype(np.float32),
                  normals=np.repeat(cube_n[:, None], 3, axis=1), outline=(1.0, 0.85, 0.1, 1.0))
    for tag, nost in (('stencil', False), ('nostencil', True)):
        r.no_stencil = nost
        oimg = r.render_offscreen(frame(oc, tris=[sel], style=Style(shadows=False, ao=False, ground_plane=False)))
        np.save(os.path.join(out_dir, f'outline_{tag}.npy'), oimg)
        (ex, ey), = project(oc, W, H, [1.0, 0.0, 1.0])               # the box's +x edge, mid-way, from above
        ex, ey = int(round(ex)), int(round(ey))
        yel = lambda v: bool(v[0] > 180 and v[1] > 150 and v[2] < 90)
        rim = [oimg[ey, ex + k, :3].astype(int) for k in (1, 2)]
        out[f'outline_{tag}_rim'] = any(yel(v) for v in rim)
        out[f'outline_{tag}_inside_clear'] = not yel(oimg[ey, ex - 6, :3].astype(int))
        out[f'outline_{tag}_outside_clear'] = not yel(oimg[ey, ex + 8, :3].astype(int))
    r.no_stencil = False
    out['stencil_left_clear'] = int(GL.glGetIntegerv(GL.GL_STENCIL_TEST)) == 0

    # 6b. the fixed-function path (no shaders) draws points and outlines too
    from vantage.ui import viewport3d as V3
    if V3.GLViewport is not None:
        class Fake:
            pass
        fake = Fake()
        fake.GL = GL
        fake.devicePixelRatioF = lambda: 1.0
        ms_fbo, res_fbo, fbos, rbs = r._offscreen_fbos(W, H, 0)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, res_fbo)
        GL.glViewport(0, 0, W, H)
        GL.glClearColor(0.1, 0.1, 0.1, 1.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glLoadMatrixd(oc.proj(W / H).T.astype(np.float64).ravel())
        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glLoadMatrixd(oc.view().T.astype(np.float64).ravel())
        V3.GLViewport._draw_outline(fake, sel)
        V3.GLViewport._draw_tris(fake, sel.tri, sel.colors)
        V3.GLViewport._draw_points(fake, PointDraw(np.array([[2.0, 0.0, 0.5]], np.float32),
                                                   np.array([[0.1, 0.9, 0.2, 1.0]], np.float32), size=9.0))
        data = GL.glReadPixels(0, 0, W, H, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE)
        limg = np.frombuffer(data, np.uint8).reshape(H, W, 4)[::-1]
        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glLoadIdentity()
        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glLoadIdentity()
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
        GL.glDeleteFramebuffers(len(fbos), fbos)
        GL.glDeleteRenderbuffers(len(rbs), rbs)
        (ex, ey), = project(oc, W, H, [1.0, 0.0, 1.0])
        ex, ey = int(round(ex)), int(round(ey))
        yel = lambda v: bool(v[0] > 180 and v[1] > 150 and v[2] < 90)
        out['legacy_outline'] = any(yel(limg[ey, ex + k, :3].astype(int)) for k in (0, 1, 2))
        pv = px(limg, oc, [2.0, 0.0, 0.5])
        out['legacy_point'] = bool(pv[1] - pv[0] > 80)
        GL.glPixelStorei(GL.GL_PACK_ALIGNMENT, 4)          # PyOpenGL leaves 1 after a read

    # 7. glass is dark and a little see-through; paint is not
    def sheet(x0, x1, z):                                   # one upward-facing quad: car glass is a single sheet
        q = np.array([[[x0, -1, z], [x1, -1, z], [x1, 1, z]], [[x0, -1, z], [x1, 1, z], [x0, 1, z]]], np.float32)
        return q, np.tile([0, 0, 1.0], (2, 1)).astype(np.float32)
    red_t, red_n = sheet(-1, 1, 0.0)
    gl_t, gl_n = sheet(-1, 0, 0.5)
    pa_t, pa_n = sheet(0, 1, 0.5)
    T = np.r_[red_t, gl_t, pa_t]
    N = np.r_[red_n, gl_n, pa_n]
    colr = np.r_[np.tile([0.85, 0.1, 0.1], (2, 1)), np.tile([0.23, 0.24, 0.23], (2, 1)), np.tile([0.23, 0.24, 0.23], (2, 1))]
    mats = np.r_[np.zeros(2), np.ones(2), np.zeros(2)].astype(np.int8)
    gm = MeshDraw(T, N, colr, np.ones(6), version=4, materials=mats)
    gc = cam_at([0.0, 0.0, 0.3], 9.0, -90.0, 89.99, ortho=True, h=3.0)
    gimg = r.render_offscreen(frame(gc, mesh=gm, style=Style(shadows=False, ao=False, ground_plane=False)))
    gv, pv = px(gimg, gc, [-0.5, 0.0, 0.52], 3), px(gimg, gc, [0.5, 0.0, 0.52], 3)
    out['glass_red_through'] = float(gv[0] - gv[1])
    out['paint_red_through'] = float(pv[0] - pv[1])
    out['glass_dark'] = float(np.mean(gv))

    # 8. very large meshes refresh their sky maps over several frames (pending until done)
    RG.SKY_BUDGET, keep_budget = 1, RG.SKY_BUDGET
    r2 = GLRenderer()
    r2.init()
    ms_fbo, res_fbo, fbos, rbs = r2._offscreen_fbos(64, 64, 0)
    small = cam_at([1.2, 0, 0.4], 9.0, 58.0, 34.0)
    steps = []
    for i in range(30):
        r2.render(Frame(small.view(), small.proj(1.0), 64, 64, target=small.target.copy(), mesh=car, clear=bg), res_fbo)
        steps.append(r2.pending)
        if not r2.pending:
            break
    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
    GL.glDeleteFramebuffers(len(fbos), fbos)
    GL.glDeleteRenderbuffers(len(rbs), rbs)
    r2.release()
    RG.SKY_BUDGET = keep_budget
    out['sky_progressive_frames'] = len(steps)
    out['sky_progressive_first_pending'] = bool(steps[0])

    # 8b. the floor grid and axes drawn by the ground shader: lines on the 0.5 m / 1 m marks from the display
    # origin, plain floor between them, the x axis red and the y axis green, nothing with the grid off
    gcam = cam_at([2.0, 2.0, 0.0], 9.0, -90.0, 89.99, ortho=True, h=6.0)
    gst = dict(shadows=False, ao=False, grid_minor=(0.9, 0.9, 0.9, 0.3), grid_major=(0.9, 0.9, 0.9, 0.6))
    g_off = r.render_offscreen(frame(gcam, style=Style(**gst)))
    g_on = r.render_offscreen(frame(gcam, style=Style(grid=True, axes=True, origin=(0.3, -0.2), **gst)))
    lum = lambda img, X: float(px(img, gcam, X, rad=0).mean())            # noqa: E731
    out['grid_major'] = lum(g_on, [2.3, 1.13, 0.0]) - lum(g_off, [2.3, 1.13, 0.0])      # x = origin + 2 m
    out['grid_minor'] = lum(g_on, [1.8, 1.13, 0.0]) - lum(g_off, [1.8, 1.13, 0.0])      # x = origin + 1.5 m
    out['grid_between'] = abs(lum(g_on, [2.05, 1.13, 0.0]) - lum(g_off, [2.05, 1.13, 0.0]))
    out['grid_off_same'] = float(np.abs(g_off.astype(int) - r.render_offscreen(frame(gcam, style=Style(**gst))).astype(int)).max())
    ax = px(g_on, gcam, [1.4, -0.2, 0.0], rad=0)
    ay = px(g_on, gcam, [0.3, 0.9, 0.0], rad=0)
    out['axis_x_red'] = bool(ax[0] > ax[1] + 60 and ax[0] > ax[2] + 60)
    out['axis_y_green'] = bool(ay[1] > ay[0] + 60 and ay[1] > ay[2] + 40)

    # 9. frame time at 1600 x 900 (4x MSAA): the car with shadows, sky occlusion, contact shadow and camera light
    BW, BH = 1600, 900
    ms_fbo, res_fbo, fbos, rbs = r._offscreen_fbos(BW, BH, 4)
    target = ms_fbo if ms_fbo is not None else res_fbo
    hero = cam_at([1.2, 0, 0.4], 9.0, -42.0, 24.0)
    fr = Frame(hero.view(), hero.proj(BW / BH), BW, BH, target=hero.target.copy(), mesh=car, cams=views, clear=bg)
    car2 = MeshDraw(car.tri, car.normals, car.base, car.alpha, version=9, materials=car.materials)
    fr.mesh = car2                                          # a new mesh: the first frame builds every cached map
    GL.glFinish()
    t0 = time.perf_counter()
    r.render(fr, target, complete=True)
    GL.glFinish()
    out['frame_first_ms'] = round((time.perf_counter() - t0) * 1000, 2)
    times = []
    for i in range(20):
        hero.az = -42.0 + i * 2.0
        fr = Frame(hero.view(), hero.proj(BW / BH), BW, BH, target=hero.target.copy(), mesh=car2, cams=views, clear=bg)
        GL.glFinish()
        t0 = time.perf_counter()
        r.render(fr, target)
        GL.glFinish()
        times.append((time.perf_counter() - t0) * 1000)
    out['frame_ms'] = round(float(np.median(times)), 2)
    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
    GL.glDeleteFramebuffers(len(fbos), fbos)
    GL.glDeleteRenderbuffers(len(rbs), rbs)
    out['pixel_store'] = [int(GL.glGetIntegerv(GL.GL_UNPACK_ALIGNMENT)), int(GL.glGetIntegerv(GL.GL_PACK_ALIGNMENT))]
    out['gl_error_end'] = int(GL.glGetError())
    r.release()
    print(json.dumps(out))


if __name__ == '__main__':
    if '--quality' in sys.argv:
        quality(sys.argv[1])
    else:
        main(sys.argv[1])
