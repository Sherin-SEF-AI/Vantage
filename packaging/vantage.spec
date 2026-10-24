# PyInstaller spec for Vantage: one folder with two launchers that share one copy of Python, Qt, NumPy and OpenCV.
#
#   Vantage(.exe)        the window; built without a console, so double-clicking it on Windows opens no console
#   vantage-cli(.exe)    the same program with a console: batch mode, --self-test, --version, error output
#
# A windowed Windows executable has no stdout and its exit code is not waited for by the shell, so batch mode needs
# the console one. Two launchers in one folder cost a few MB; two separate bundles would cost ~400 MB.
#
#   pyinstaller packaging/vantage.spec        (or: python packaging/build.py)
#
# Build from packaging/requirements-bundle.txt: it adds what the bundles carry beyond the app's requirements
# (keyring, zstandard, lz4, tzdata) and takes OpenCV without its own GUI (opencv-python-headless).
import importlib.util
import os
import sys

from PyInstaller.utils.hooks import collect_submodules

ROOT = os.path.dirname(SPECPATH)            # noqa: F821  (SPECPATH is set by PyInstaller)
ICONS = os.path.join(ROOT, 'packaging', 'icons')
sys.path.insert(0, os.path.join(ROOT, 'packaging'))
import release                              # noqa: E402  (packaging/release.py)

VERSION = release.version(ROOT)

# every module of the app, whether imported statically or by name: the pages (mainwindow.PAGE_CLASSES through
# importlib), the dialogs and studies imported inside functions, the assistant, the exporters, the batch mode
hiddenimports = collect_submodules('vantage')
# PyOpenGL picks its platform module (EGL, GLX, WGL, CGL) and array handlers by name at run time
for mod in ('OpenGL.platform', 'OpenGL.arrays', 'OpenGL.raw.EGL', 'OpenGL.raw.GLX', 'OpenGL.raw.WGL'):
    hiddenimports += collect_submodules(mod, on_error='ignore')
hiddenimports += ['OpenGL.EGL', 'OpenGL.GLX', 'OpenGL.WGL']
# optional modules the bundles carry when they are installed: compressed ROS 2 bags, the OS keyring for the
# assistant's key, IANA time zones where the system has none (Windows); their PyInstaller hooks collect the rest
OPTIONAL = {'zstandard': 'zstd-compressed ROS 2 bags', 'lz4': 'lz4-compressed ROS 2 bags',
            'keyring': "the assistant's key in the OS keyring", 'tzdata': 'named time zones on Windows',
            'certifi': "the assistant's HTTPS certificates"}
have = [m for m in OPTIONAL if importlib.util.find_spec(m) is not None]
hiddenimports += have + (['lz4.frame'] if 'lz4' in have else [])
print('vantage.spec: optional modules bundled: ' + (', '.join(have) or 'none')
      + ''.join(f'; MISSING {m} ({why})' for m, why in OPTIONAL.items() if m not in have))

excludes = ['tkinter', 'matplotlib', 'IPython', 'pytest', 'PyQt6.QtWebEngineCore', 'PyQt6.QtWebEngineWidgets',
            'PyQt6.QtQml', 'PyQt6.QtQuick', 'PyQt6.QtMultimedia', 'PyQt6.QtBluetooth', 'PyQt6.QtSql',
            'PyQt6.QtDesigner', 'PyQt6.QtPdf', 'PyQt6.QtPdfWidgets', 'PyQt6.QtTest', 'PyQt6.Qt3DCore',
            'PyQt6.QtCharts', 'PyQt6.QtDataVisualization', 'PyQt6.QtPositioning', 'PyQt6.QtRemoteObjects',
            'PyQt6.QtSensors', 'PyQt6.QtSerialPort', 'PyQt6.QtSpatialAudio', 'PyQt6.QtTextToSpeech',
            'PyQt6.QtWebChannel', 'PyQt6.QtWebSockets', 'PyQt6.QtNfc', 'PyQt6.QtHelp',
            # other Qt bindings (PyInstaller refuses to bundle two) and big packages that optional imports elsewhere
            # (SciPy's array-API layer, keyring's libsecret backend) would pull in from a well-stocked build machine
            'PyQt5', 'PySide2', 'PySide6', 'torch', 'tensorflow', 'jax', 'jaxlib', 'cupy', 'dask', 'pandas',
            'sparse', 'numba', 'llvmlite', 'sklearn', 'sympy', 'onnxruntime', 'gi', 'yt_dlp', 'notebook',
            'jupyter_client',
            # zstandard's cffi backend (11 MB) is for PyPy; CPython uses its C backend
            'zstandard.backend_cffi', 'zstandard._cffi']

a = Analysis(                               # noqa: F821
    [os.path.join(ROOT, 'run_vantage.py')],
    pathex=[ROOT],
    datas=release.data_files(ROOT),
    hiddenimports=hiddenimports,
    excludes=excludes,
    noarchive=False,
)

# the pages are imported by name, so check that every one made it in rather than finding out on a user's machine
_collected = {name for name, _, _ in a.pure}
_missing = [m for m in release.page_modules(ROOT) if m not in _collected]
if _missing:
    raise SystemExit('vantage.spec: pages missing from the bundle: ' + ', '.join(_missing))

# Qt's translations of its own dialogs: Vantage's interface is English and loads none
a.datas = [d for d in a.datas if not d[0].replace('\\', '/').startswith('PyQt6/Qt6/translations/')]

# the licences of everything bundled, and a notice listing it (licenses/THIRD-PARTY.txt)
from PyInstaller.config import CONF                     # noqa: E402
_top = {name.split('.')[0] for name, _, _ in a.pure}
_top |= {b[0].replace('\\', '/').split('/')[0].split('.')[0] for b in a.binaries if b[2] == 'EXTENSION'}
a.datas += [(os.path.join(dest, os.path.basename(src)), src, 'DATA')
            for src, dest in release.license_files(_top, os.path.join(CONF['workpath'], 'THIRD-PARTY.txt'))]
a.datas += [('LICENSE', os.path.join(ROOT, 'LICENSE'), 'DATA')]      # Vantage's own: GPL-3.0-or-later


def _dest(entry):
    return entry[0].replace('\\', '/')


def drop_unused_libraries(binaries, plugins):
    """Leave out the Qt plugins given (dest path suffixes) and the libraries only they need. Linux only: the GTK 3
    platform theme pulls in ~35 libraries (GTK, GLib/GIO, cairo, pango, SELinux, mount, ...) from the build machine,
    and a bundled GTK loads the user's own GTK modules and themes, a known source of crashes on other distributions
    (Vantage styles itself; file dialogs still go through the desktop portal); the PDF image-format plugin pulls in
    QtPdf (Vantage writes PDFs with QtGui and never reads one as an image)."""
    from PyInstaller.depend.bindepend import get_imports
    gone = [b for b in binaries if _dest(b).endswith(plugins)]
    if not gone:
        return binaries
    by_name = {os.path.basename(b[0]): b for b in binaries}

    def reach(entries):
        seen, todo = set(), list(entries)
        while todo:
            e = todo.pop()
            if e[0] in seen:
                continue
            seen.add(e[0])
            todo += [by_name[n] for n, _ in get_imports(e[1]) if n in by_name]
        return seen
    # everything that is loaded on its own (Python extensions, Qt plugins: entries in subfolders, but not Qt's own
    # libraries, which are only loaded as something's dependency) except those
    roots = [b for b in binaries if '/' in _dest(b) and not _dest(b).startswith('PyQt6/Qt6/lib/') and b not in gone]
    drop = (reach(gone) - reach(roots)) | {g[0] for g in gone}
    print(f'vantage.spec: leaving out {", ".join(os.path.basename(_dest(g)) for g in gone)} and '
          f'{len(drop) - len(gone)} libraries only they need')
    return [b for b in binaries if b[0] not in drop]


if sys.platform.startswith('linux'):
    a.binaries = drop_unused_libraries(a.binaries, ('platformthemes/libqgtk3.so', 'imageformats/libqpdf.so'))
pyz = PYZ(a.pure)                           # noqa: F821

common = dict(exclude_binaries=True, debug=False, strip=False, upx=False, disable_windowed_traceback=False)
if sys.platform == 'win32':
    common['icon'] = os.path.join(ICONS, 'vantage.ico')
elif sys.platform == 'darwin':
    common['icon'] = os.path.join(ICONS, 'vantage.icns')
# Python's UTF-8 mode: output to a pipe or a file (batch mode, the self-test on CI) is UTF-8 on every system, not
# Windows' ANSI code page, which cannot encode the arrows and symbols the reports print; files Vantage reads and writes
# without naming an encoding are UTF-8 everywhere too
options = [('X utf8', None, 'OPTION')]
gui = EXE(pyz, a.scripts, options, name='Vantage', console=False, **common)          # noqa: F821
cli = EXE(pyz, a.scripts, options, name='vantage-cli', console=True, **common)       # noqa: F821
coll = COLLECT(gui, cli, a.binaries, a.datas, strip=False, upx=False, name='Vantage')   # noqa: F821

if sys.platform == 'darwin':
    UTI = 'io.github.sherin-sef-ai.vantage.project'
    app = BUNDLE(                           # noqa: F821
        coll,
        name='Vantage.app',
        icon=os.path.join(ICONS, 'vantage.icns'),
        bundle_identifier='io.github.sherin-sef-ai.vantage',
        version=VERSION,
        info_plist={
            'CFBundleShortVersionString': VERSION,
            'CFBundleVersion': VERSION,
            # COLLECT takes the console flag of its last launcher (vantage-cli), which would make PyInstaller mark
            # the app background-only: no Dock icon, no menu bar
            'LSBackgroundOnly': False,
            'NSHighResolutionCapable': True,
            # NumPy's and SciPy's wheels for macOS 14 on Apple silicon (what the macos-14 runner installs) need 14
            'LSMinimumSystemVersion': '14.0',
            'UTExportedTypeDeclarations': [{
                'UTTypeIdentifier': UTI, 'UTTypeDescription': 'Vantage project',
                'UTTypeConformsTo': ['public.data'],
                'UTTypeIconFile': 'vantage.icns',
                'UTTypeTagSpecification': {'public.filename-extension': ['vantage'],
                                           'public.mime-type': ['application/x-vantage-project']}}],
            'CFBundleDocumentTypes': [{'CFBundleTypeName': 'Vantage project', 'CFBundleTypeRole': 'Editor',
                                       'LSHandlerRank': 'Owner', 'LSItemContentTypes': [UTI],
                                       'CFBundleTypeIconFile': 'vantage.icns'}],
        },
    )
