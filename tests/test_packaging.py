"""Packaging: the version flag, the package data (every resource reaches the wheel), the entry points and the self-test
that the bundles run after building."""
import glob
import importlib
import os
import struct
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

try:
    import tomllib
except ImportError:             # Python < 3.11
    tomllib = None


def _pyproject():
    if tomllib is None:
        pytest.skip('tomllib needs Python 3.11')
    with open(os.path.join(ROOT, 'pyproject.toml'), 'rb') as fh:
        return tomllib.load(fh)


def _run(*args, timeout=60):
    env = dict(os.environ, QT_QPA_PLATFORM='offscreen')
    return subprocess.run([sys.executable, os.path.join(ROOT, 'run_vantage.py'), *args], capture_output=True,
                          text=True, timeout=timeout, env=env, cwd=ROOT)


def test_version_flag_needs_no_qt():
    from vantage import __version__
    code = ('import sys; from vantage import app\n'
            'try:\n    app.main(["--version"])\n'
            'except SystemExit as e:\n    print("exit", e.code, "qt", "PyQt6" in sys.modules)\n')
    r = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=60, cwd=ROOT)
    assert r.stdout.splitlines() == [f'vantage {__version__}', 'exit 0 qt False'], r.stderr


def test_every_resource_is_package_data():
    """A file under vantage/ that is not Python must match a package-data glob, or pip installs leave it out."""
    globs = _pyproject()['tool']['setuptools']['package-data']['vantage']
    pkg = os.path.join(ROOT, 'vantage')
    covered = {os.path.normpath(p) for g in globs for p in glob.glob(os.path.join(pkg, g), recursive=True)}
    files = []
    for d, subdirs, names in os.walk(pkg):
        subdirs[:] = [s for s in subdirs if s != '__pycache__']
        files += [os.path.normpath(os.path.join(d, n)) for n in names if not n.endswith(('.py', '.pyc'))]
    assert files and not [os.path.relpath(f, ROOT) for f in files if f not in covered]


def test_version_is_read_from_the_package():
    doc = _pyproject()
    assert 'version' in doc['project']['dynamic']
    assert doc['tool']['setuptools']['dynamic']['version'] == {'attr': 'vantage.__version__'}


def test_entry_points_resolve():
    doc = _pyproject()['project']
    eps = {**doc['scripts'], **doc['gui-scripts']}
    assert set(eps) == {'vantage', 'vantage-gui'}
    for target in eps.values():
        mod, fn = target.split(':')
        assert callable(getattr(importlib.import_module(mod), fn))


def test_requirements_match_the_dependencies():
    deps = {d.split('>')[0].split('=')[0].strip().lower() for d in _pyproject()['project']['dependencies']}
    with open(os.path.join(ROOT, 'requirements.txt')) as fh:
        reqs = {ln.split('#')[0].split('>')[0].split('=')[0].strip().lower() for ln in fh if ln.split('#')[0].strip()}
    assert deps == reqs


def test_icons_are_valid():
    from vantage.app import APP_ICON
    with open(APP_ICON, 'rb') as fh:
        assert fh.read(8) == b'\x89PNG\r\n\x1a\n'
    with open(os.path.join(ROOT, 'packaging', 'icons', 'vantage.ico'), 'rb') as fh:
        reserved, kind, count = struct.unpack('<HHH', fh.read(6))
        assert (reserved, kind) == (0, 1) and count >= 4
    with open(os.path.join(ROOT, 'packaging', 'icons', 'vantage.icns'), 'rb') as fh:
        head = fh.read(8)
        assert head[:4] == b'icns' and struct.unpack('>I', head[4:])[0] == os.path.getsize(fh.name)


def test_self_test_passes():
    r = _run('--self-test', timeout=180)
    assert r.returncode == 0, r.stdout + r.stderr
    lines = r.stdout.splitlines()
    assert lines[-1] == 'self-test passed'
    for check in ('analysis', 'opencv', 'fonts', 'opengl', 'main window'):
        assert any(ln.split()[:1] == [check.split()[0]] for ln in lines), check


def test_desktop_entry_and_icon_install_and_uninstall(tmp_path, monkeypatch):
    """vantage --install-desktop writes vantage.desktop (the id the window announces), the icon at every theme size
    and the file type into the user's data folder, and --uninstall-desktop removes exactly that."""
    import configparser
    from PyQt6.QtGui import QImage
    from vantage import app, desktop
    monkeypatch.setattr(desktop, '_refresh', lambda base: None)          # leave the real desktop databases alone
    monkeypatch.setattr(desktop, 'desktop_dir', lambda: str(tmp_path / 'Desktop'))    # and the real desktop
    monkeypatch.setenv('XDG_DATA_HOME', str(tmp_path))
    assert app.main(['--install-desktop']) == 0
    p = desktop.paths()
    cp = configparser.ConfigParser(interpolation=None)
    cp.optionxform = str
    cp.read(p['desktop'])
    e = cp['Desktop Entry']
    assert e['Icon'] == 'vantage' == desktop.APP_ID and e['Type'] == 'Application' and e['Exec'].endswith(' %f')
    assert 'run_vantage.py' in e['Exec'] and 'application/x-vantage-project' in e['MimeType']
    for path, n in zip(p['icons'], desktop.SIZES):
        img = QImage(path)
        assert not img.isNull() and img.width() == img.height() == n
    assert '*.vantage' in open(p['mime']).read()
    assert app.main(['--install-desktop']) == 0                         # again: replaces its own files
    removed = desktop.uninstall()
    assert set(removed) == {p['desktop'], p['mime'], *p['icons']}
    assert not any(os.path.exists(x) for x in removed)
    # an entry someone else wrote (the bundle's install.sh) is left alone
    os.makedirs(os.path.dirname(p['desktop']), exist_ok=True)
    with open(p['desktop'], 'w') as fh:
        fh.write('[Desktop Entry]\nName=Vantage\n')
    assert app.main(['--install-desktop']) == 1 and app.main(['--uninstall-desktop']) == 1
    assert open(p['desktop']).read() == '[Desktop Entry]\nName=Vantage\n'


def test_exec_quoting():
    from vantage.desktop import _q
    assert _q('/usr/bin/python3') == '/usr/bin/python3'
    assert _q('/home/a b/run.py') == '"/home/a b/run.py"' and _q('/x/$HOME') == '"/x/\\$HOME"'


def test_desktop_shortcut(tmp_path, monkeypatch):
    """--desktop-shortcut installs the menu entry and puts an executable launcher on the desktop; uninstall removes
    it, and leaves a launcher it did not write."""
    from vantage import app, desktop
    monkeypatch.setattr(desktop, '_refresh', lambda base: None)
    monkeypatch.setattr(desktop, 'desktop_dir', lambda: str(tmp_path / 'Desktop'))
    monkeypatch.setenv('XDG_DATA_HOME', str(tmp_path / 'share'))
    assert app.main(['--desktop-shortcut']) == 0
    sc = tmp_path / 'Desktop' / 'vantage.desktop'
    assert sc.is_file() and os.access(sc, os.X_OK) and desktop.MARK in sc.read_text()
    assert os.path.isfile(desktop.paths()['desktop'])
    assert str(sc) in desktop.uninstall() and not sc.exists()
    sc.write_text('[Desktop Entry]\nName=Mine\n')
    assert app.main(['--uninstall-desktop']) == 0 and sc.read_text() == '[Desktop Entry]\nName=Mine\n'
