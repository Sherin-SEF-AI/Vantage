# PyInstaller spec for Vantage: one folder with two launchers that share one copy of Python, Qt, NumPy and OpenCV.
#
#   Vantage(.exe)        the window; built without a console, so double-clicking it on Windows opens no console
#   vantage-cli(.exe)    the same program with a console: batch mode, --self-test, --version, error output
#
# A windowed Windows executable has no stdout and its exit code is not waited for by the shell, so batch mode needs
# the console one. Two launchers in one folder cost a few MB; two separate bundles would cost ~400 MB.
#
#   pyinstaller packaging/vantage.spec        (or: python packaging/build.py)
import os
import sys

from PyInstaller.utils.hooks import collect_submodules

ROOT = os.path.dirname(SPECPATH)            # noqa: F821  (SPECPATH is set by PyInstaller)
PKG = os.path.join(ROOT, 'vantage')
ICONS = os.path.join(ROOT, 'packaging', 'icons')

# every non-Python file in the package (fonts, icon, the C++ stitcher), and the runtime's Python stitcher as a file
# too, because exports copy it next to the remap tables
datas = []
for d, subdirs, files in os.walk(PKG):
    subdirs[:] = [s for s in subdirs if s != '__pycache__']
    rel = os.path.relpath(d, ROOT)
    for f in files:
        if not f.endswith(('.py', '.pyc')) or (os.path.basename(d) == 'runtime' and f.endswith('.py')):
            datas.append((os.path.join(d, f), rel))

hiddenimports = collect_submodules('vantage')
for mod in ('OpenGL.platform', 'OpenGL.arrays', 'OpenGL.raw.EGL', 'OpenGL.raw.GLX', 'OpenGL.raw.WGL'):
    hiddenimports += collect_submodules(mod, on_error='ignore')       # PyOpenGL picks these by name at run time
hiddenimports += ['OpenGL.EGL', 'OpenGL.GLX', 'OpenGL.WGL']

excludes = ['tkinter', 'matplotlib', 'IPython', 'pytest', 'PyQt6.QtWebEngineCore', 'PyQt6.QtWebEngineWidgets',
            'PyQt6.QtQml', 'PyQt6.QtQuick', 'PyQt6.QtMultimedia', 'PyQt6.QtBluetooth', 'PyQt6.QtSql',
            'PyQt6.QtDesigner', 'PyQt6.QtPdf']

a = Analysis(                               # noqa: F821
    [os.path.join(ROOT, 'run_vantage.py')],
    pathex=[ROOT],
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=excludes,
    noarchive=False,
)


def drop_gtk_theme(binaries):
    """Leave out Qt's GTK 3 platform theme and the ~35 libraries only it needs (GTK, GLib/GIO, cairo, pango, SELinux,
    mount, ...). They come from the build machine, and a bundled GTK loads the user's own GTK modules and themes, a
    known source of crashes on other distributions. Vantage styles itself; file dialogs still go through the desktop
    portal theme."""
    from PyInstaller.depend.bindepend import get_imports
    theme = [b for b in binaries if b[0].replace('\\', '/').endswith('platformthemes/libqgtk3.so')]
    if not theme:
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
    # everything that is loaded on its own (Python extensions, Qt plugins: entries in subfolders) except the theme
    roots = [b for b in binaries if '/' in b[0].replace('\\', '/') and b not in theme]
    drop = (reach(theme) - reach(roots)) | {t[0] for t in theme}
    print(f'vantage.spec: leaving out the GTK 3 theme and {len(drop) - 1} libraries only it needs')
    return [b for b in binaries if b[0] not in drop]


if sys.platform.startswith('linux'):
    a.binaries = drop_gtk_theme(a.binaries)
pyz = PYZ(a.pure)                           # noqa: F821

common = dict(exclude_binaries=True, debug=False, strip=False, upx=False, disable_windowed_traceback=False,
              icon=os.path.join(ICONS, 'vantage.ico' if sys.platform == 'win32' else 'vantage.icns'))
gui = EXE(pyz, a.scripts, [], name='Vantage', console=False, **common)          # noqa: F821
cli = EXE(pyz, a.scripts, [], name='vantage-cli', console=True, **common)       # noqa: F821
coll = COLLECT(gui, cli, a.binaries, a.datas, strip=False, upx=False, name='Vantage')   # noqa: F821

if sys.platform == 'darwin':
    sys.path.insert(0, ROOT)
    from vantage import __version__
    app = BUNDLE(                           # noqa: F821
        coll,
        name='Vantage.app',
        icon=os.path.join(ICONS, 'vantage.icns'),
        bundle_identifier='io.github.sherin-sef-ai.vantage',
        info_plist={
            'CFBundleShortVersionString': __version__,
            'NSHighResolutionCapable': True,
            'CFBundleDocumentTypes': [{'CFBundleTypeName': 'Vantage project', 'CFBundleTypeRole': 'Editor',
                                       'LSHandlerRank': 'Owner', 'CFBundleTypeExtensions': ['vantage']}],
        },
    )
