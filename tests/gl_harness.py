"""Run the shader renderer in a headless EGL context and report what it drew (used by test_render_gl.py in a
subprocess, because PyOpenGL picks its platform once per process).

    python3 tests/gl_harness.py OUT_DIR      prints one JSON line; 'skip' when no EGL/OpenGL is available
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
    out['gl_error_end'] = int(GL.glGetError())
    r.release()
    print(json.dumps(out))


if __name__ == '__main__':
    main(sys.argv[1])
