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
