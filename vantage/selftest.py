"""`vantage --self-test`: checks an installation without showing a window and prints one line per check.

    analysis     a template project analysed end to end
    opencv       calibration and installation checks need it
    fonts        the bundled IBM Plex fonts load
    opengl       the shader renderer draws a frame off-screen (a warning only: the 3D view falls back to software)
    main window  the main window is built (and shown and painted on the offscreen platform)

The exit code is 0 when everything the app needs works, 1 otherwise.
"""
from __future__ import annotations

import os
import sys
import time
import traceback

OK, WARN, FAIL = 'ok', 'warning', 'FAILED'


def _egl_context():
    """A headless EGL OpenGL context (Linux, for the offscreen Qt platform, which has no OpenGL)."""
    import ctypes
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
    return dpy, surf, ctx


def _check_analysis():
    from .core import samples
    from .core.analysis import Engine
    p = samples.template_project('sedan')
    res = Engine().analyse(p)
    cov = float(res.metrics['covered_pct'])
    if not 50 < cov <= 100:
        return FAIL, f'sedan template covers {cov:.1f} % (expected most of the ROI)'
    return OK, f'sedan template, {len(p.variant.cameras)} cameras, {cov:.1f} % covered, {res.ms:.0f} ms'


def _check_opencv():
    import cv2
    import numpy as np
    img = np.zeros((60, 80), np.uint8)
    cv2.findChessboardCornersSB(img, (4, 3))
    return OK, f'OpenCV {cv2.__version__}'


def _check_fonts():
    from PyQt6.QtGui import QFontDatabase
    fams = set(QFontDatabase.families())
    missing = [f for f in ('IBM Plex Sans', 'IBM Plex Mono') if f not in fams]
    if missing:
        return FAIL, 'missing ' + ', '.join(missing)
    return OK, 'IBM Plex Sans, IBM Plex Mono'


def _check_opengl():
    import numpy as np
    from PyQt6.QtGui import QGuiApplication
    plat = QGuiApplication.platformName()
    keep = []
    if plat in ('offscreen', 'minimal'):
        if not sys.platform.startswith('linux'):
            return WARN, f'not checked: the {plat} platform has no OpenGL'
        try:
            keep.append(_egl_context())
        except Exception as e:
            return WARN, f'no headless OpenGL ({e}); the 3D view would use the software renderer'
    else:
        from PyQt6.QtGui import QOffscreenSurface, QOpenGLContext, QSurfaceFormat
        surf = QOffscreenSurface()
        surf.setFormat(QSurfaceFormat.defaultFormat())
        surf.create()
        ctx = QOpenGLContext()
        ctx.setFormat(QSurfaceFormat.defaultFormat())
        if not ctx.create() or not ctx.makeCurrent(surf):
            return WARN, 'no OpenGL context; the 3D view will use the software renderer'
        keep += [surf, ctx]
    from OpenGL import GL
    from .core import samples
    from .ui.render_gl import Frame, GLRenderer
    from .ui.scene import mesh_colors
    from .ui.viewport3d import MeshDraw, OrbitCam
    name = GL.glGetString(GL.GL_RENDERER)
    name = name.decode(errors='replace') if name else '?'
    r = GLRenderer()
    if not r.init():
        return WARN, f'{name}: shaders failed ({(r.error or "").splitlines()[0][:120]}); the 3D view uses fixed-function GL'
    m = samples.template_project('sedan').vehicle.mesh
    base, alpha, keep_t = mesh_colors(m, 'solid')
    md = MeshDraw(m.tri[keep_t].astype(np.float32), m.normals[keep_t], base[keep_t], alpha[keep_t], version=1)
    cam = OrbitCam()
    W, H = 320, 200
    img = r.render_offscreen(Frame(cam.view(), cam.proj(W / H), W, H, target=cam.target.copy(), mesh=md,
                                   clear=(0.1, 0.1, 0.1, 1.0)))
    r.release()
    if len(keep) == 2:
        keep[1].doneCurrent()
    lit = float((img[..., :3].mean(axis=2) > 120).mean())
    if lit < 0.02:
        return FAIL, f'{name}: the shader renderer drew an empty frame'
    return OK, f'shader renderer on {name}'


def _check_window():
    from PyQt6.QtWidgets import QApplication
    from .core import samples
    from .ui.mainwindow import MainWindow
    from .ui.viewport3d import gl_available
    errors = []
    hook = sys.excepthook
    sys.excepthook = lambda t, v, tb: errors.append(''.join(traceback.format_exception(t, v, tb)))
    try:
        w = MainWindow(samples.template_project('sedan'), soft_3d=not gl_available())
        shown = QApplication.platformName() in ('offscreen', 'minimal')
        if shown:                       # never flash a window on a real desktop
            w.resize(1400, 900)
            w.show()
        t = time.time()
        while time.time() - t < 1.5:
            QApplication.processEvents()
            time.sleep(0.02)
        title = w.windowTitle()
        w.cancel_jobs()
        w.hide()                        # not close(): closing saves the window layout to the user's settings
        w.deleteLater()
        QApplication.processEvents()
    finally:
        sys.excepthook = hook
    if errors:
        return FAIL, errors[0].strip().splitlines()[-1]
    return OK, f'{title!r}' + (' shown and painted' if shown else ' built (not shown)')


def run() -> int:
    from . import __version__
    plat = os.environ.get('QT_QPA_PLATFORM', '')
    if sys.platform.startswith('linux') and plat.startswith(('offscreen', 'minimal')):
        os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')      # before anything imports OpenGL
    print(f'Vantage {__version__} self-test · Python {sys.version.split()[0]} · {sys.platform}'
          + (' · bundled' if getattr(sys, 'frozen', False) else ''))
    from .app import make_app
    app, _ = make_app()
    failed = False
    for name, fn in (('analysis', _check_analysis), ('opencv', _check_opencv), ('fonts', _check_fonts),
                     ('opengl', _check_opengl), ('main window', _check_window)):
        try:
            status, msg = fn()
        except Exception as e:
            status, msg = FAIL, f'{type(e).__name__}: {e}'
        failed |= status == FAIL
        print(f'  {name:<12} {status:<8} {msg}')
    print('self-test ' + ('FAILED' if failed else 'passed'))
    sys.stdout.flush()
    del app
    return 1 if failed else 0
