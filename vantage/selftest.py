"""`vantage --self-test`: checks an installation without showing a window and prints one line per check, in the order
a new user meets them.

    analysis     the Tata Tigor EV sample opened and its Data rig (cameras, lidar, radars) analysed end to end
    opencv       the chessboard detector (calibration, installation check) and the PNG and JPEG codecs
    fonts        the bundled IBM Plex fonts load
    qt plugins   the platform plugins this system's desktop needs load with their libraries (X11 and Wayland with
                 their OpenGL integrations on Linux, Windows, Cocoa), and the image formats
    time zones   named time zones resolve (sun glare studies; without them a fixed UTC offset is used)
    https        the assistant's certificate authorities load (no connection is made)
    keyring      which system keyring would keep the assistant's API key (information; no key is read)
    ros 2 bags   a tiny bag written and read back: MCAP uncompressed, with zstd and lz4 chunks when those modules
                 are present, and a SQLite (.db3) bag
    opengl       the shader renderer draws a frame off-screen (a warning only: the 3D view falls back to QPainter)
    main window  the main window is built with the sample and its analysis finishes (shown and painted on the
                 offscreen platform, never on a desktop)
    pages        every page of the main window built and shown
    software 3d  the QPainter 3D view, the fallback without OpenGL, draws the sample
    exports      every export format written into a temporary folder, checked and removed

Nothing is written outside that temporary folder (the window is hidden, never closed, so its layout is not saved), no
network connection is made and the window's assistant is kept away from the system keyring. The exit code is 0 when
everything the app needs works, 1 otherwise; a warning says what does not work here and what the app does instead.
"""
from __future__ import annotations

import ctypes
import json
import os
import shutil
import sys
import tempfile
import time
import traceback

OK, WARN, FAIL = 'ok', 'warning', 'FAILED'
# EGL platform enums (EGL_EXT_platform_device, EGL_MESA_platform_surfaceless); PyOpenGL has no constants for them
_EGL_PLATFORM_DEVICE = 0x313F
_EGL_PLATFORM_SURFACELESS = 0x31DD


# ------------------------------------------------------------------------------------------------------- OpenGL (EGL)
def _egl_try(get):
    """An EGL display from get() initialised with a pbuffer-capable OpenGL config, a context made current on it:
    (dpy, surf, ctx), or raises."""
    from OpenGL import EGL
    dpy = get()
    if not dpy or not EGL.eglInitialize(dpy, None, None):
        raise RuntimeError('no display')
    try:
        attrs = (EGL.EGLint * 13)(EGL.EGL_SURFACE_TYPE, EGL.EGL_PBUFFER_BIT, EGL.EGL_RED_SIZE, 8, EGL.EGL_GREEN_SIZE, 8,
                                 EGL.EGL_BLUE_SIZE, 8, EGL.EGL_DEPTH_SIZE, 24, EGL.EGL_RENDERABLE_TYPE,
                                 EGL.EGL_OPENGL_BIT, EGL.EGL_NONE)
        cfg, n = EGL.EGLConfig(), EGL.EGLint()
        if not EGL.eglChooseConfig(dpy, attrs, ctypes.pointer(cfg), 1, ctypes.pointer(n)) or n.value < 1:
            raise RuntimeError('no OpenGL config')
        surf = EGL.eglCreatePbufferSurface(dpy, cfg, (EGL.EGLint * 5)(EGL.EGL_WIDTH, 16, EGL.EGL_HEIGHT, 16,
                                                                       EGL.EGL_NONE))
        EGL.eglBindAPI(EGL.EGL_OPENGL_API)
        ctx = EGL.eglCreateContext(dpy, cfg, EGL.EGL_NO_CONTEXT, None)
        if not ctx or not EGL.eglMakeCurrent(dpy, surf, surf, ctx):
            raise RuntimeError('no OpenGL context')
        return dpy, surf, ctx
    except BaseException:
        EGL.eglTerminate(dpy)
        raise


def _egl_context():
    """A headless EGL OpenGL context (Linux, for the offscreen Qt platform, which has no OpenGL of its own). Tried in
    turn: the default display (a desktop session), Mesa's surfaceless platform (a GPU or, without one, the llvmpipe
    software renderer, as on a CI machine), then each EGL device (GPUs without a display server, NVIDIA's driver).
    Returns ((dpy, surf, ctx), how) or raises with what each attempt said."""
    from OpenGL import EGL
    tries = [('default display', lambda: EGL.eglGetDisplay(EGL.EGL_DEFAULT_DISPLAY)),
             ('surfaceless', lambda: EGL.eglGetPlatformDisplay(_EGL_PLATFORM_SURFACELESS, EGL.EGL_DEFAULT_DISPLAY,
                                                               None))]
    try:
        from OpenGL.EGL.EXT.device_enumeration import eglQueryDevicesEXT
        from OpenGL.EGL.EXT.platform_base import eglGetPlatformDisplayEXT
        devs, n = (EGL.EGLDeviceEXT * 8)(), EGL.EGLint()
        if eglQueryDevicesEXT(8, devs, ctypes.pointer(n)):
            tries += [(f'EGL device {i}', lambda d=devs[i]: eglGetPlatformDisplayEXT(_EGL_PLATFORM_DEVICE, d, None))
                      for i in range(n.value)]
    except Exception:                   # no device extension in this libEGL
        pass
    said = []
    for how, get in tries:
        try:
            return _egl_try(get), how
        except Exception as e:
            said.append(f'{how}: {_first_line(e)}')
    raise RuntimeError('; '.join(said) or 'no EGL')


def _egl_release(handles):
    from OpenGL import EGL
    dpy, surf, ctx = handles
    EGL.eglMakeCurrent(dpy, EGL.EGL_NO_SURFACE, EGL.EGL_NO_SURFACE, EGL.EGL_NO_CONTEXT)
    EGL.eglDestroyContext(dpy, ctx)
    EGL.eglDestroySurface(dpy, surf)
    EGL.eglTerminate(dpy)


def _first_line(e) -> str:
    text = str(e).strip()
    if type(e).__name__ == 'EGLError':          # PyOpenGL's EGLError prints a multi-line repr: keep the error name
        import re
        m = re.search(r'err\s*=\s*(\w+)', text)
        return f'EGLError {m.group(1)}' if m else 'EGLError'
    return (text.splitlines() or [type(e).__name__])[0][:160]


# ------------------------------------------------------------------------------------------------------------ checks
def _check_analysis(S):
    from .core import samples
    from .core.analysis import Engine
    p = samples.tigor_sample_project()
    p.photos.source = ''                # the studio photos are not ours to read: the self-test never opens them
    rig = next((v for v in p.variants if v.sensors), None)
    if rig is None:
        return FAIL, 'the sample has no variant with lidars or radars'
    p.active_uid = rig.uid
    eng = Engine()
    t = time.perf_counter()
    res = eng.analyse(p)
    dt = time.perf_counter() - t
    cov = float(res.metrics['covered_pct'])
    S.update(project=p, engine=eng, result=res)
    kinds = [s.kind for s in rig.sensors]
    what = ', '.join(f'{kinds.count(k)} {k}{"s" if kinds.count(k) > 1 else ""}' for k in ('lidar', 'radar', 'gnss', 'imu')
                     if k in kinds)
    if not 50 < cov <= 100:
        return FAIL, f'{rig.name}: {cov:.1f} % of the ground covered (expected most of it)'
    return OK, f"Tigor EV sample's data rig: {len(rig.cameras)} cameras, {what}; {cov:.1f} % covered in {dt:.1f} s"


def _check_opencv(S):
    import cv2
    import numpy as np
    img = np.zeros((60, 80), np.uint8)
    cv2.setRNGSeed(0)                               # the detector draws on OpenCV's generator
    cv2.findChessboardCornersSB(img, (4, 3))
    pic = (np.arange(48 * 64 * 3) % 251).astype(np.uint8).reshape(48, 64, 3)
    for ext in ('.png', '.jpg'):
        ok, buf = cv2.imencode(ext, pic)
        back = cv2.imdecode(buf, cv2.IMREAD_COLOR) if ok else None
        if back is None or back.shape != pic.shape or (ext == '.png' and not np.array_equal(back, pic)):
            return FAIL, f'OpenCV {cv2.__version__}: the {ext[1:].upper()} codec does not round-trip'
    return OK, f'OpenCV {cv2.__version__}: chessboard detector, PNG and JPEG codecs'


def _check_cameras(S):
    """How live cameras open: V4L2 devices, GStreamer pipelines (through OpenCV, or through the system's
    gst-launch-1.0 when OpenCV has no GStreamer, as in the bundles; a test pipeline is run), RTSP/HTTP streams and
    video files through FFmpeg. No camera is opened."""
    import cv2
    from .core import capture as C
    info = cv2.getBuildInformation().splitlines()

    def has(key):
        line = next((l for l in info if l.strip().lower().startswith(key + ':')), '')
        return 'YES' in line.split(':', 1)[-1].upper()
    linux = sys.platform.startswith('linux')
    parts = ['V4L2 devices' if has('v4l/v4l2') else 'no V4L2'] if linux else []
    if C.opencv_has_gstreamer():
        parts.append('GStreamer pipelines through OpenCV')
    elif shutil.which('gst-launch-1.0'):
        try:
            g = C.GstPipe('videotestsrc num-buffers=2 ! video/x-raw,width=64,height=48', timeout=10)
        except RuntimeError as e:
            if 'videotestsrc' not in str(e):
                return WARN, f"the system's gst-launch-1.0 cannot run a test pipeline: {_first_line(e)}"
            parts.append("GStreamer pipelines through the system's gst-launch-1.0 (not tried: no videotestsrc)")
        else:
            ok, f = g.read()
            g.release()
            if not ok or f.shape != (48, 64, 3):
                return WARN, "the system's gst-launch-1.0 ran a test pipeline but no frame came back"
            parts.append("GStreamer pipelines through the system's gst-launch-1.0 (a test pipeline ran)")
    elif linux:
        parts.append('GStreamer pipelines need gst-launch-1.0 (gstreamer1.0-tools)')
    if not has('ffmpeg'):
        return WARN, '; '.join(parts + ['no FFmpeg in OpenCV: RTSP/HTTP streams and video files cannot be opened'])
    return OK, '; '.join(parts + ['RTSP/HTTP streams and video files through FFmpeg'])


def _check_fonts(S):
    from PyQt6.QtGui import QFontDatabase
    fams = set(QFontDatabase.families())
    missing = [f for f in ('IBM Plex Sans', 'IBM Plex Mono') if f not in fams]
    if missing:
        return FAIL, 'missing ' + ', '.join(missing)
    return OK, 'IBM Plex Sans, IBM Plex Mono'


def _plugin(kind, prefix):
    """(path, error) of the first Qt plugin file in <plugins>/<kind>/ whose name starts with prefix, loaded."""
    from PyQt6.QtCore import QCoreApplication, QPluginLoader
    for d in QCoreApplication.libraryPaths():
        folder = os.path.join(d, kind)
        if not os.path.isdir(folder):
            continue
        for f in sorted(os.listdir(folder)):
            if f.startswith(prefix) and f.endswith(('.so', '.dll', '.dylib')):
                path = os.path.join(folder, f)
                loader = QPluginLoader(path)
                if loader.load():
                    return path, ''
                err = loader.errorString()
                if path in err:             # "Cannot load library <path>: (<why>)": the why is what matters
                    err = err.split(path, 1)[1].lstrip(': ').strip('()')
                return path, err
    return None, f'no {prefix}* in {kind}/'


def _check_qt_plugins(S):
    """Loading a plugin pulls in its libraries, so a missing one (libxcb-cursor on X11, say) shows here although the
    self-test itself runs on the offscreen platform. It fails when no desktop platform can load; a session type that
    cannot (X11 without libxcb-cursor0 on a Wayland desktop, say) is a warning."""
    from PyQt6.QtGui import QImageReader
    if sys.platform.startswith('linux'):
        need = [('platforms', 'libqxcb', 'X11', True), ('xcbglintegrations', 'libqxcb-glx', 'X11 OpenGL (GLX)', False),
                ('xcbglintegrations', 'libqxcb-egl', 'X11 OpenGL (EGL)', False),
                ('platforms', 'libqwayland', 'Wayland', True),
                ('wayland-graphics-integration-client', 'libqt-plugin-wayland-egl', 'Wayland OpenGL', False)]
    elif sys.platform == 'win32':
        need = [('platforms', 'qwindows', 'Windows', True)]
    elif sys.platform == 'darwin':
        need = [('platforms', 'libqcocoa', 'Cocoa', True)]
    else:
        need = []
    loaded, why = [], {}
    platforms = 0
    for kind, prefix, label, platform in need:
        _, err = _plugin(kind, prefix)
        if err:
            why.setdefault(err, []).append(label)
        else:
            loaded.append(label)
            platforms += platform
    fmts = {bytes(f).decode() for f in QImageReader.supportedImageFormats()}
    no_fmt = [f for f in ('png', 'jpeg', 'svg') if f not in fmts]
    text = (', '.join(loaded) + ' load' if loaded else 'no platform plugin loads')
    text += '; images ' + ', '.join(f for f in ('png', 'jpeg', 'svg') if f in fmts)
    for err, labels in why.items():
        hint = ' (install libxcb-cursor0)' if 'libxcb-cursor' in err else ''
        text += f'; {", ".join(labels)}: {err}{hint}'
    if no_fmt:
        text += '; missing image formats ' + ', '.join(no_fmt)
    if need and not platforms:
        return FAIL, text + ' (the window cannot open)'
    return (WARN if why or no_fmt else OK), text


def _check_time_zones(S):
    import importlib.util
    try:
        from zoneinfo import ZoneInfo, TZPATH
        for name in ('Europe/Berlin', 'Asia/Kolkata', 'America/Los_Angeles'):
            ZoneInfo(name)
    except Exception as e:
        return WARN, (f'named time zones unavailable ({_first_line(e)}): sun glare studies use the fixed UTC offset '
                      '(pip install tzdata)')
    system = any(os.path.isfile(os.path.join(d, 'Europe', 'Berlin')) for d in TZPATH)
    src = ' and '.join(s for s, have in (('the system database', system),
                                         ('the tzdata package', importlib.util.find_spec('tzdata') is not None)) if have)
    return OK, f'Europe/Berlin, Asia/Kolkata, America/Los_Angeles from {src or "?"}'


def _check_https(S):
    import ssl
    from .assistant.client import _ssl_context
    total = _ssl_context().cert_store_stats().get('x509_ca', 0)
    try:
        import certifi
        own = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        own.load_verify_locations(certifi.where())
        n = own.cert_store_stats().get('x509_ca', 0)
    except Exception as e:
        if not total:
            return FAIL, f'no certificate authorities ({_first_line(e)}): the assistant cannot check its connection'
        return WARN, f'certifi unusable ({_first_line(e)}); {total} system certificate authorities instead'
    if not n:
        return FAIL, 'the certifi bundle holds no certificate authorities'
    return OK, f'{n} certificate authorities from certifi {certifi.__version__} ({total} in all); no connection made'


def _check_keyring(S):
    if S['no_keyring']:
        return OK, 'turned off by VANTAGE_NO_KEYRING: the assistant keeps its key in a private file'
    try:
        import keyring
        from keyring.backends import fail
    except ImportError:
        return OK, 'keyring not installed: the assistant keeps its key in a private file in the settings folder'
    try:
        kr = keyring.get_keyring()
    except Exception as e:
        return WARN, f'keyring {keyring.__version__ if hasattr(keyring, "__version__") else ""} fails: {_first_line(e)}'
    name = f'{type(kr).__module__.rsplit(".", 1)[-1]}.{type(kr).__name__}'
    if isinstance(kr, fail.Keyring):
        if sys.platform in ('win32', 'darwin'):     # these systems always have one: the backends are missing
            return WARN, 'no keyring backend found (the OS keyring is there, keyring cannot reach it)'
        return OK, 'no keyring service on this system: the assistant keeps its key in a private file'
    return OK, f'{name} (the key itself is not read)'


def _check_bags(S):
    import importlib.util
    import sqlite3
    import cv2
    import numpy as np
    from .core import rosbag as rb
    comps = ['none'] + [c for c, mod in (('zstd', 'zstandard'), ('lz4', 'lz4')) if importlib.util.find_spec(mod)]
    t0 = 1_700_000_000_000_000_000
    rng = np.random.default_rng(7)
    pts = np.zeros(64, dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('intensity', '<f4')])
    for k in pts.dtype.names:
        pts[k] = rng.normal(size=64)
    pic = (np.arange(48 * 64 * 3) % 251).astype(np.uint8).reshape(48, 64, 3)
    png = cv2.imencode('.png', np.ascontiguousarray(pic[..., ::-1]))[1].tobytes()   # BGR in: read back as pic
    T = np.eye(4)
    T[:3, 3] = (1.2, 0.0, 1.9)
    order = [7, 2, 9, 0, 4, 1, 8, 3, 6, 5]                        # written out of time order on purpose

    def imu(k):
        return {'header': rb.header(t0 + k * 10_000_000, 'imu'), 'linear_acceleration': {'x': float(k), 'z': 9.81}}

    def check(bag, what):
        if bag.counts.get('/imu') != 10 or bag.message_count != 14:
            raise ValueError(f'{what}: {bag.message_count} messages read, 14 written')
        got = [(t, m) for _, t, m in bag.messages('/imu')]
        if [t for t, _ in got] != sorted(t for t, _ in got) or \
                [m['linear_acceleration']['x'] for _, m in got] != [float(k) for k in range(10)]:
            raise ValueError(f'{what}: the IMU messages came back wrong or out of time order')
        if 'lidar' not in what:
            return
        for topic, _, m in bag.messages(['/lidar/points', '/cam/F/image', '/cam/F/compressed', '/tf_static']):
            if topic == '/lidar/points' and not np.array_equal(rb.pointcloud2_array(m), pts):
                raise ValueError(f'{what}: the point cloud changed')
            if topic in ('/cam/F/image', '/cam/F/compressed') and not np.array_equal(rb.image_array(m, True), pic):
                raise ValueError(f'{what}: the image on {topic} changed')
            if topic == '/tf_static' and not np.allclose(rb.transform_matrix(m['transforms'][0]), T, atol=1e-9):
                raise ValueError(f'{what}: the static transform changed')

    tmp = S['tmp']
    for comp in comps:
        path = os.path.join(tmp, f'bag-{comp}')
        with rb.BagWriter(path, compression=comp, chunk_size=2048) as bag:
            for topic, typ in (('/imu', 'sensor_msgs/msg/Imu'), ('/lidar/points', 'sensor_msgs/msg/PointCloud2'),
                               ('/cam/F/image', 'sensor_msgs/msg/Image'),
                               ('/cam/F/compressed', 'sensor_msgs/msg/CompressedImage'),
                               ('/tf_static', 'tf2_msgs/msg/TFMessage')):
                bag.add_topic(topic, typ)
            bag.write('/tf_static', t0, {'transforms': [rb.transform_stamped(T, t0, 'base_link', 'lidar_L1')]})
            for k in order:
                bag.write('/imu', t0 + k * 10_000_000, imu(k))
            bag.write('/lidar/points', t0 + 5_000_000, rb.pointcloud2(pts, t0 + 5_000_000, 'lidar_L1'))
            bag.write('/cam/F/image', t0 + 6_000_000, rb.image_msg(pic, t0 + 6_000_000, 'cam_F', 'rgb8'))
            bag.write('/cam/F/compressed', t0 + 6_000_000, rb.compressed_image(png, t0 + 6_000_000, 'cam_F', 'png'))
        with rb.BagReader(path) as bag:
            check(bag, f'MCAP ({comp}) lidar')
    # a rosbag2 SQLite bag, as ROS 2 Humble records by default
    db3 = os.path.join(tmp, 'bag-sqlite', 'bag-sqlite_0.db3')
    os.makedirs(os.path.dirname(db3))
    con = sqlite3.connect(db3)
    try:
        con.execute('CREATE TABLE topics(id INTEGER PRIMARY KEY, name TEXT NOT NULL, type TEXT NOT NULL, '
                    'serialization_format TEXT NOT NULL, offered_qos_profiles TEXT NOT NULL)')
        con.execute('CREATE TABLE messages(id INTEGER PRIMARY KEY, topic_id INTEGER NOT NULL, '
                    'timestamp INTEGER NOT NULL, data BLOB NOT NULL)')
        con.executemany('INSERT INTO topics VALUES (?, ?, ?, ?, ?)',
                        [(1, '/imu', 'sensor_msgs/msg/Imu', 'cdr', ''), (2, '/odom', 'nav_msgs/msg/Odometry', 'cdr', '')])
        rows = [(1, t0 + k * 10_000_000, rb.encode('sensor_msgs/msg/Imu', imu(k))) for k in order]
        rows += [(2, t0 + k * 25_000_000, rb.encode('nav_msgs/msg/Odometry', {'header': rb.header(t0, 'odom')}))
                 for k in range(4)]
        con.executemany('INSERT INTO messages(topic_id, timestamp, data) VALUES (?, ?, ?)', rows)
        con.commit()
    finally:
        con.close()
    with rb.BagReader(db3) as bag:
        check(bag, 'SQLite')
    missing = [mod for c, mod in (('zstd', 'zstandard'), ('lz4', 'lz4')) if c not in comps]
    text = f'MCAP written and read back ({", ".join(comps)} chunks), SQLite .db3 read'
    if missing:
        return WARN, text + f'; compressed bags need {" and ".join(missing)} (pip install {" ".join(missing)})'
    return OK, text


def _check_opengl(S):
    import numpy as np
    from PyQt6.QtGui import QGuiApplication
    plat = QGuiApplication.platformName()
    egl, how, qt = None, '', []
    if plat in ('offscreen', 'minimal'):
        if not sys.platform.startswith('linux'):
            return WARN, f'not checked: the {plat} platform has no OpenGL'
        try:
            egl, how = _egl_context()
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
        qt, how = [surf, ctx], 'Qt'
    try:
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
    finally:
        if egl is not None:
            _egl_release(egl)
        if qt:
            qt[1].doneCurrent()
    lit = float((img[..., :3].mean(axis=2) > 120).mean())
    if lit < 0.02:
        return FAIL, f'{name}: the shader renderer drew an empty frame'
    return OK, f'shader renderer on {name} ({how})'


def _pump(seconds, until=None):
    from PyQt6.QtWidgets import QApplication
    t = time.time()
    while time.time() - t < seconds:
        QApplication.processEvents()
        if until is not None and until():
            return True
        time.sleep(0.01)
    return until() if until is not None else True


def _check_window(S):
    from PyQt6.QtWidgets import QApplication
    from .core import samples
    from .ui.mainwindow import MainWindow
    from .ui.viewport3d import gl_available
    p = samples.tigor_sample_project()     # a copy of its own: the window's edits never reach the exports' project
    p.photos.source = ''
    rig = next((v for v in p.variants if v.sensors), None)
    if rig is not None:
        p.active_uid = rig.uid
    shown = QApplication.platformName() in ('offscreen', 'minimal')
    t = time.perf_counter()
    w = MainWindow(p, soft_3d=not gl_available())
    S['window'] = w
    if shown:                           # never flash a window on a real desktop
        w.resize(1400, 900)
        w.show()
    done = _pump(30.0, lambda: w.ctx.analysis.result is not None)
    _pump(0.3)
    if not done:
        return FAIL, 'the analysis did not finish within 30 s'
    return OK, (f'{w.windowTitle()!r} ' + ('shown and painted' if shown else 'built (not shown)')
                + f', analysed in {time.perf_counter() - t:.1f} s')


def _check_pages(S):
    from .ui.mainwindow import PAGE_CLASSES
    w = S.get('window')
    if w is None:
        return FAIL, 'no main window'
    n0 = len(S['errors'])
    broken = []
    for name in PAGE_CLASSES:
        before = len(S['errors'])
        try:
            w.set_page(name)
            _pump(0.08)
        except Exception as e:
            broken.append(f'{name}: {type(e).__name__}: {_first_line(e)}')
            continue
        if len(S['errors']) > before:
            broken.append(f'{name}: {S["errors"][-1].strip().splitlines()[-1]}')
    w.set_page('viewports')
    _pump(0.1)
    w.cancel_jobs()                     # a page's study may have started: it is not needed
    if broken:
        del S['errors'][n0:]            # reported here, not again by the checks that follow
        return FAIL, f'{len(broken)} of {len(PAGE_CLASSES)} pages failed: ' + '; '.join(broken)
    return OK, f'{len(PAGE_CLASSES)} pages built and shown'


def _qimage_array(img):
    import numpy as np
    from PyQt6.QtGui import QImage
    img = img.convertToFormat(QImage.Format.Format_RGBA8888)
    ptr = img.constBits()
    ptr.setsize(img.sizeInBytes())
    return np.frombuffer(ptr, np.uint8).reshape(img.height(), img.bytesPerLine())[:, :img.width() * 4] \
        .reshape(img.height(), img.width(), 4).copy()


def _png(img) -> bytes:
    from PyQt6.QtCore import QBuffer, QIODevice
    buf = QBuffer()
    buf.open(QIODevice.OpenModeFlag.WriteOnly)
    img.save(buf, 'PNG')
    return bytes(buf.data())


def _check_software_3d(S):
    """The images the exports take from the window, drawn as the export dialog draws them; the 3D view with QPainter
    is checked for actually showing the vehicle."""
    import numpy as np
    from .ui.camview import render_camera
    from .ui.planview import PlanView
    from .ui.scene import MainScene
    from .ui.viewport3d import SoftViewport
    w = S.get('window')
    if w is None:
        return FAIL, 'no main window'
    with w.ctx.analysis._lock:          # the window's own analysis thread is done with its engine
        vp = SoftViewport(MainScene(w.ctx))
        vp.resize(800, 500)
        vp.frame_box(*w.viewarea._bbox(), margin=0.82)
        img = vp.grab().toImage()
        vp.deleteLater()
        images = {'viewport': _png(img)}
        for key in ('bev', 'coverage'):
            pv = PlanView(w.ctx, key, interactive=False)
            pv.show_tooltip = False
            pv.resize(800, 550)
            pv.fit()
            images[key] = _png(pv.grab().toImage())
            pv.deleteLater()
    p, eng, res = S['project'], S['engine'], S['result']
    for c in p.variant.cameras:
        if c.visible:
            m = p.module_for(c)
            cimg, _ = render_camera(p, eng, c, 480, int(480 * m.height_px / m.width_px), res)
            images[f'cam:{c.uid}'] = _png(cimg)
    S['images'] = images
    a = _qimage_array(img)[..., :3].astype(int)
    if a.size == 0:
        return FAIL, 'the QPainter 3D view drew nothing'
    bg = np.median(a[:8, :8].reshape(-1, 3), axis=0)
    drawn = float((np.abs(a - bg).max(axis=2) > 24).mean())
    if drawn < 0.03:
        return FAIL, f'the QPainter 3D view drew an empty frame ({100 * drawn:.1f} % of pixels)'
    return OK, f'QPainter 3D view {img.width()} × {img.height()} px, {100 * drawn:.0f} % drawn; bird\'s-eye, ' \
               f'coverage and {len(images) - 3} camera views'


def _check_file(path):
    """'' or what is wrong with an exported file."""
    size = os.path.getsize(path)
    if not size:
        return 'empty'
    ext = os.path.splitext(path)[1].lower()
    with open(path, 'rb') as fh:
        head = fh.read(8)
    if ext == '.json':
        with open(path, encoding='utf-8') as fh:
            json.load(fh)
    elif ext == '.png' and head != b'\x89PNG\r\n\x1a\n':
        return 'not a PNG'
    elif ext == '.pdf' and not head.startswith(b'%PDF'):
        return 'not a PDF'
    elif ext == '.npz':
        import numpy as np
        with np.load(path) as z:
            if not z.files:
                return 'no arrays'
    return ''


def _check_exports(S):
    from .core import exporters as ex
    p, eng, res = S.get('project'), S.get('engine'), S.get('result')
    if res is None:
        return FAIL, 'no analysis to export'
    images = S.get('images') or {}
    out = os.path.join(S['tmp'], 'exports')
    failed, files, size = [], 0, 0
    for fmt in ex.FORMATS:
        if fmt.group == 'Images' and not images:
            failed.append(f'{fmt.id}: no images (the window failed)')
            continue
        try:
            paths = ex.export([fmt.id], p, res, ex.ExportOptions(out_dir=out, images=images, engine=eng))
            want = fmt.filename(p, p.variant)
            want = [want] if isinstance(want, str) else list(want)
            got = {os.path.relpath(x, out).replace(os.sep, '/') for x in paths}
            # the bracket STLs are what may be written: a camera the generator turns down gets none
            lost = [n for n in want if n not in got and not (fmt.id == 'brackets' and n.endswith('.stl'))]
            if not paths and want:                  # (a format with nothing to write here, e.g. no experiment set up)
                raise ValueError('nothing written')
            if lost:
                raise ValueError(f'{", ".join(lost[:3])} not written')
            for x in paths:
                bad = _check_file(x)
                if bad:
                    raise ValueError(f'{os.path.relpath(x, out)}: {bad}')
            files += len(paths)
            size += sum(os.path.getsize(x) for x in paths)
        except Exception as e:
            failed.append(f'{fmt.id}: {type(e).__name__}: {_first_line(e)}')
    if failed:
        return FAIL, f'{len(failed)} of {len(ex.FORMATS)} formats failed: ' + '; '.join(failed[:4])
    return OK, f'{len(ex.FORMATS)} formats, {files} files ({size / 1e6:.0f} MB) written, checked and removed'


CHECKS = (('analysis', _check_analysis), ('opencv', _check_opencv), ('cameras', _check_cameras),
          ('fonts', _check_fonts), ('qt plugins', _check_qt_plugins), ('time zones', _check_time_zones),
          ('https', _check_https), ('keyring', _check_keyring), ('ros 2 bags', _check_bags), ('opengl', _check_opengl),
          ('main window', _check_window), ('pages', _check_pages), ('software 3d', _check_software_3d),
          ('exports', _check_exports))
NAMES = tuple(n for n, _ in CHECKS)
# checks whose warning is fine in a bundle smoke test: CI machines have no GPU (and Windows/macOS no headless GL)
BUNDLE_MAY_WARN = ('opengl',)


def run() -> int:
    from . import __version__
    for s in (sys.stdout, sys.stderr):
        try:                            # a console that cannot show a character gets '?', not a crash
            s.reconfigure(errors='replace')
        except (AttributeError, ValueError):
            pass
    plat = os.environ.get('QT_QPA_PLATFORM', '')
    if sys.platform.startswith('linux') and plat.startswith(('offscreen', 'minimal')):
        os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')      # before anything imports OpenGL
        os.environ.setdefault('EGL_LOG_LEVEL', 'fatal')        # Mesa's notes about the displays it tries
    t0 = time.perf_counter()
    print(f'Vantage {__version__} self-test · Python {sys.version.split()[0]} · {sys.platform}'
          + (' · bundled' if getattr(sys, 'frozen', False) else ''), flush=True)
    S = {'no_keyring': bool(os.environ.get('VANTAGE_NO_KEYRING')), 'errors': [],
         'tmp': tempfile.mkdtemp(prefix='vantage-self-test-')}
    os.environ['VANTAGE_NO_KEYRING'] = '1'      # the hidden window's assistant: no keyring prompt during a check
    from .app import make_app
    app, _ = make_app()
    hook = sys.excepthook
    sys.excepthook = lambda t, v, tb: S['errors'].append(''.join(traceback.format_exception(t, v, tb)))
    failed = False
    try:
        for name, fn in CHECKS:
            n0 = len(S['errors'])
            try:
                status, msg = fn(S)
            except Exception as e:
                status, msg = FAIL, f'{type(e).__name__}: {_first_line(e)}'
            if len(S['errors']) > n0 and status != FAIL:        # an exception in a Qt slot during the check
                status, msg = FAIL, S['errors'][-1].strip().splitlines()[-1]
            failed |= status == FAIL
            print(f'  {name:<12} {status:<8} {msg}', flush=True)
    finally:
        sys.excepthook = hook
        w = S.pop('window', None)
        if w is not None:
            w.cancel_jobs()
            w.hide()                    # not close(): closing saves the window layout to the user's settings
            w.deleteLater()
            app.processEvents()
        shutil.rmtree(S['tmp'], ignore_errors=True)
        if S['no_keyring']:
            os.environ['VANTAGE_NO_KEYRING'] = '1'
        else:
            os.environ.pop('VANTAGE_NO_KEYRING', None)
    print(f'self-test {"FAILED" if failed else "passed"} in {time.perf_counter() - t0:.0f} s')
    sys.stdout.flush()
    del app
    return 1 if failed else 0
