"""Packaging: the version flag, the package data (every resource reaches the wheel), the entry points, the self-test
that the bundles run after building, the PyInstaller spec and what it collects, the release workflow (bundles first,
checksums, CHANGELOG notes), the Linux installer and the Windows installer script."""
import glob
import importlib
import importlib.util
import os
import shutil
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


def _packaging_module(name):
    """packaging/<name>.py as a module (packaging/ is not a package)."""
    path = os.path.join(ROOT, 'packaging', f'{name}.py')
    sys.path.insert(0, os.path.dirname(path))
    try:
        spec = importlib.util.spec_from_file_location(f'_vantage_packaging_{name}', path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        sys.path.remove(os.path.dirname(path))
    return mod


def _text(*parts):
    with open(os.path.join(ROOT, *parts), encoding='utf-8') as fh:
        return fh.read()


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


def _req_name(line):
    """'name' of a requirement line such as 'PyQt6>=6.5' or 'tzdata; sys_platform == "win32"'."""
    import re
    return re.split(r'[;<>=!~\[ ]', line.split('#')[0].strip(), 1)[0].lower()


def test_requirements_match_the_dependencies():
    deps = {_req_name(d) for d in _pyproject()['project']['dependencies']}
    with open(os.path.join(ROOT, 'requirements.txt')) as fh:
        reqs = {_req_name(ln) for ln in fh if ln.split('#')[0].strip()}
    assert deps == reqs and 'tzdata' in deps


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
    """The self-test from source: every check runs and none fails (warnings are allowed here: a developer machine may
    lack a GPU, libxcb-cursor0 or zstandard); the exports, pages and bags checks pass outright."""
    from vantage.selftest import NAMES
    r = _run('--self-test', timeout=300)
    assert r.returncode == 0, r.stdout + r.stderr
    lines = r.stdout.splitlines()
    assert lines[-1].startswith('self-test passed'), lines[-1]
    status = {}
    for ln in lines[1:-1]:
        name = next(n for n in NAMES if ln[2:].startswith(n + ' '))
        status[name] = ln[2 + 12:].split()[0]
    assert set(status) == set(NAMES), r.stdout
    assert 'FAILED' not in status.values(), r.stdout
    for check in ('analysis', 'fonts', 'main window', 'pages', 'software 3d', 'exports', 'https'):
        assert status[check] == 'ok', r.stdout
    # what the bundles' smoke test makes of it: nothing beyond the warnings this machine explains
    build = _packaging_module('build')
    bad = build.self_test_problems(r.stdout)
    assert all(any(b.startswith(n + ' ') and status[n] == 'warning' for n in NAMES) for b in bad), bad


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
    assert _q('/home/a b/run.py') == '"/home/a b/run.py"' and _q('/x/$HOME') == '"/x/\\\\$HOME"' and _q('/x/50% off') == '"/x/50%% off"'


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


# ---------------------------------------------------------------------------------------------------- the release
def test_version_tops_the_changelog():
    """The version (0.1.1: the first release plus Linux arm64) is the CHANGELOG's top entry, which the release job
    publishes as the notes, with the sections a user needs before downloading."""
    from vantage import __version__
    release = _packaging_module('release')
    assert __version__ == '0.1.1' == release.version()
    entries = release.changelog()
    assert entries and entries[0][0] == __version__
    assert len({v for v, _, _ in entries}) == len(entries)                  # one entry per version
    notes = release.notes(__version__)
    for heading in ('**What it does**', '**Downloads**', '**Unsigned builds**', '**Known limits**'):
        assert heading in notes, heading
    for name in (f'Vantage-{__version__}-windows-x64-setup.exe', f'Vantage-{__version__}-macos-arm64.zip',
                 f'Vantage-{__version__}-linux-x86_64.tar.gz', f'Vantage-{__version__}-linux-arm64.tar.gz',
                 'JetPack 6', 'SHA256SUMS.txt', 'SmartScreen', 'Open Anyway'):
        assert name in notes, name
    with pytest.raises(ValueError):
        release.notes('9.9.9')


def test_release_notes_command():
    """What the workflow runs: the notes for the matching tag on stdout (UTF-8), a mismatched tag fails."""
    from vantage import __version__
    release = _packaging_module('release')
    cmd = [sys.executable, os.path.join(ROOT, 'packaging', 'release.py')]
    r = subprocess.run(cmd + ['notes', '--tag', f'v{__version__}'], capture_output=True, timeout=60)
    assert r.returncode == 0 and r.stdout.decode('utf-8') == release.notes(__version__), r.stderr
    r = subprocess.run(cmd + ['notes', '--tag', 'v99.0.0'], capture_output=True, text=True, timeout=60)
    assert r.returncode == 1 and 'does not match' in r.stderr
    r = subprocess.run(cmd + ['version'], capture_output=True, text=True, timeout=60)
    assert r.stdout.strip() == __version__


def test_spec_collects_every_page():
    """The pages are imported by name (mainwindow.PAGE_CLASSES through importlib), so no import statement leads
    PyInstaller to them: the spec collects every vantage module and checks each page made it into the bundle."""
    from vantage.ui.mainwindow import PAGE_CLASSES
    release = _packaging_module('release')
    want = [f'vantage.ui.workspaces.{mod}' for mod, _ in PAGE_CLASSES.values()]
    assert release.page_modules() == want
    for mod, cls in PAGE_CLASSES.values():
        m = importlib.import_module(f'vantage.ui.workspaces.{mod}')
        assert isinstance(getattr(m, cls), type)
    spec = _text('packaging', 'vantage.spec')
    assert "collect_submodules('vantage')" in spec and 'release.page_modules(ROOT)' in spec
    if importlib.util.find_spec('PyInstaller') is None:
        return                                  # the CI's Linux job has PyInstaller: the real collection is checked there
    from PyInstaller.utils.hooks import collect_submodules
    got = set(collect_submodules('vantage'))
    assert not [m for m in want if m not in got]
    assert {'vantage.assistant.client', 'vantage.core.rosbag', 'vantage.selftest', 'vantage.batch'} <= got


def test_spec_data_files():
    """Every non-Python file of the package goes into the bundle (fonts, icon, the C++ stitcher), and the Python
    stitcher too, as a file the exports copy; nothing from __pycache__."""
    release = _packaging_module('release')
    files = release.data_files()
    src = {os.path.relpath(f, ROOT).replace(os.sep, '/') for f, _ in files}
    assert all(dest == os.path.dirname(os.path.relpath(f, ROOT)) for f, dest in files)
    assert not [f for f in src if '__pycache__' in f or f.endswith('.pyc')]
    for need in ('vantage/resources/icons/vantage.png', 'vantage/resources/fonts/IBMPlexSans-Regular.ttf',
                 'vantage/resources/fonts/OFL.txt', 'vantage/runtime/bev_stitch.cpp', 'vantage/runtime/bev_stitch.py',
                 'vantage/runtime/CMakeLists.txt'):
        assert need in src, need
    for d, subdirs, names in os.walk(os.path.join(ROOT, 'vantage')):
        subdirs[:] = [x for x in subdirs if x != '__pycache__']
        for n in names:
            if not n.endswith(('.py', '.pyc')):
                assert os.path.relpath(os.path.join(d, n), ROOT).replace(os.sep, '/') in src, n


def test_spec_platform_details():
    """The pitfalls the spec guards against, in its text (PyInstaller is not needed to read it)."""
    spec = _text('packaging', 'vantage.spec')
    # UTF-8 mode in both launchers: batch output piped on Windows is not limited to the ANSI code page
    assert "('X utf8', None, 'OPTION')" in spec and spec.count('options, name=') == 2
    assert "name='Vantage', console=False" in spec and "name='vantage-cli', console=True" in spec
    # COLLECT inherits vantage-cli's console flag, which PyInstaller would turn into LSBackgroundOnly on macOS
    assert "'LSBackgroundOnly': False" in spec
    assert "'CFBundleShortVersionString': VERSION" in spec and "bundle_identifier='io.github.sherin-sef-ai.vantage'" in spec
    assert "'LSItemContentTypes': [UTI]" in spec and "'public.filename-extension': ['vantage']" in spec
    for gone in ("'tkinter'", "'PyQt6.QtWebEngineCore'", "'PyQt5'", "'torch'", "'zstandard.backend_cffi'"):
        assert gone in spec, gone
    assert "'PyQt6/Qt6/translations/'" in spec


def test_bundle_requirements():
    """packaging/requirements-bundle.txt pins every requirement of the app (OpenCV as its headless build), the
    optional modules the bundles carry and PyInstaller, each to one version."""
    deps = {_req_name(d) for d in _pyproject()['project']['dependencies']}
    pins, marked = {}, {}
    for ln in _text('packaging', 'requirements-bundle.txt').splitlines():
        ln = ln.split('#')[0].strip()
        if ln:
            spec, _, marker = ln.partition(';')
            name, _, ver = spec.partition('==')
            ver = ver.strip()
            assert ver and all(c.isdigit() or c == '.' for c in ver), ln
            pins[name.strip().lower()] = ver
            if marker.strip():
                marked.setdefault(name.strip().lower(), set()).add((marker.strip(), ver))
    # PyQt6 is pinned per machine: Linux arm64 (Jetson, JetPack 6, glibc 2.35) takes the last release with arm64
    # wheels for glibc 2.28; the markers cover every machine exactly once
    for name in ('pyqt6', 'pyqt6-qt6'):
        m = dict(marked[name])
        assert set(m) == {'platform_machine != "aarch64"', 'platform_machine == "aarch64"'}, m
        assert m['platform_machine == "aarch64"'].startswith('6.7.')
    want = {('opencv-python-headless' if d == 'opencv-python' else d) for d in deps}
    want |= {'keyring', 'zstandard', 'lz4', 'tzdata', 'pyinstaller', 'pyinstaller-hooks-contrib', 'pyqt6-qt6'}
    assert want <= set(pins), want - set(pins)
    assert 'opencv-python' not in pins                     # two OpenCVs in one environment overwrite each other
    wf = _text('.github', 'workflows', 'build.yml')
    assert 'pip install -r packaging/requirements-bundle.txt' in wf


def _workflow():
    import yaml
    doc = yaml.safe_load(_text('.github', 'workflows', 'build.yml'))
    return doc, doc.get('on', doc.get(True))       # YAML 1.1 reads the key `on` as True


def test_workflow_releases_only_after_every_bundle():
    doc, on = _workflow()
    assert 'workflow_dispatch' in on and on['push']['tags'] == ['v*']
    jobs = doc['jobs']
    b, rel = jobs['bundle'], jobs['release']
    assert set(b['strategy']['matrix']['os']) == {'ubuntu-22.04', 'ubuntu-22.04-arm', 'windows-latest', 'macos-14'}
    assert b['strategy']['fail-fast'] is False
    assert doc['permissions'] == {'contents': 'read'} and rel['permissions'] == {'contents': 'write'}
    # the release waits for the whole matrix and runs only for a version tag (never by hand)
    needs = rel['needs'] if isinstance(rel['needs'], list) else [rel['needs']]
    assert needs == ['bundle']
    assert "startsWith(github.ref, 'refs/tags/v')" in rel['if'] and 'success()' in rel['if']
    assert not any(x in rel['if'] for x in ('always()', 'failure()', 'cancelled()'))
    steps = {s.get('name', s.get('uses', '')): s for s in b['steps']}
    assert steps['Build, smoke-test and pack']['run'].strip() == 'python packaging/build.py --smoke'
    assert 'release.py notes --tag' in steps['Check the tag, the version and the CHANGELOG agree']['run']
    assert 'pytest' in steps['Packaging tests (Linux)']['run'] and 'tests/test_packaging.py' in \
        steps['Packaging tests (Linux)']['run']
    apt = steps['Qt and OpenGL libraries (Linux)']['run']
    for lib in ('libxcb-cursor0', 'libxcb-keysyms1', 'libxcb-shape0', 'libxcb-icccm4', 'libgl1-mesa-dri', 'libegl1',
                'gstreamer1.0-tools', 'gstreamer1.0-plugins-base'):
        assert lib in apt, lib
    upload = steps['actions/upload-artifact@v4']['with']
    assert upload['if-no-files-found'] == 'error' and 'runner.arch' in upload['name']    # two Linux jobs
    rsteps = {s.get('name', s.get('uses', '')): s for s in rel['steps']}
    sums = rsteps['Checksums and release notes']['run']
    assert 'sha256sum' in sums and 'SHA256SUMS.txt' in sums and 'release.py notes --tag' in sums
    pub = rsteps['Publish the release']['run']
    assert '--notes-file notes.md' in pub and 'dist/*' in pub and '--verify-tag' in pub
    for s in b['steps'] + rel['steps']:
        if 'uses' in s:
            assert '@v' in s['uses'], s['uses']             # actions pinned to a major version


def test_self_test_problems():
    """build.py's reading of a bundle's self-test: a failure, a warning outside BUNDLE_MAY_WARN, a check that did not
    run or a missing verdict fails the smoke test; an OpenGL warning does not."""
    from vantage.selftest import BUNDLE_MAY_WARN, NAMES
    build = _packaging_module('build')

    def out(status=None, drop=(), end='self-test passed in 20 s'):
        status = status or {}
        lines = ['Vantage 0.1.0 self-test · Python 3.12 · linux · bundled']
        lines += [f'  {n:<12} {status.get(n, "ok"):<8} fine' for n in NAMES if n not in drop]
        return '\n'.join(lines + [end]) + '\n'
    assert build.self_test_problems(out()) == []
    assert BUNDLE_MAY_WARN == ('opengl',)
    assert build.self_test_problems(out({'opengl': 'warning'})) == []
    bad = build.self_test_problems(out({'ros 2 bags': 'warning'}))
    assert len(bad) == 1 and bad[0].startswith('ros 2 bags')
    assert build.self_test_problems(out({'opengl': 'FAILED'}))
    assert build.self_test_problems(out(drop=('exports',))) == ['exports: did not run']
    assert build.self_test_problems(out(end='self-test FAILED in 3 s')) == ['no "self-test passed" line']


def test_archive_names_match_the_installers():
    """The file names build.py writes are the ones the Inno Setup script, the README and the CHANGELOG use."""
    build = _packaging_module('build')
    iss = _text('packaging', 'windows', 'vantage.iss')
    assert 'OutputBaseFilename=Vantage-{#AppVersion}-windows-x64-setup' in iss
    assert "f'{tag}-windows-x64-setup.exe'" in _text('packaging', 'build.py')
    assert build.VERSION == _packaging_module('release').version()


def test_inno_setup_script():
    """Per-user install without administrator rights, the .vantage association, and a self-test shortcut whose
    command line survives cmd.exe's quote stripping wherever the program is installed."""
    iss = _text('packaging', 'windows', 'vantage.iss')
    for line in ('PrivilegesRequired=lowest', 'DefaultDirName={autopf}\\Vantage', 'ChangesAssociations=yes',
                 'ArchitecturesInstallIn64BitMode=x64compatible', 'SetupIconFile=..\\icons\\vantage.ico',
                 'VersionInfoVersion={#AppVersion}'):
        assert line in iss, line
    assert 'Software\\Classes\\.vantage\\OpenWithProgids' in iss
    assert 'ValueData: """{app}\\Vantage.exe"" ""%1"""' in iss
    shortcut = next(ln for ln in iss.splitlines() if ln.startswith('Name: "{group}\\Vantage self-test"'))
    quoted = shortcut.split('Parameters: "', 1)[1].rsplit('"; IconFilename', 1)[0]
    params = quoted.replace('""', '"')                  # Inno Setup: "" inside a quoted value is one quote
    for app in (r'C:\Users\ana\AppData\Local\Programs\Vantage', r'C:\Program Files\Vantage',
                r'C:\Users\Jo (work)\AppData\Local\Programs\Vantage'):
        line = params.replace('{app}', app)
        assert line.startswith('/k ')
        cmdline = line[3:]
        # cmd /k with more than two quotes: the first and the last quote of the command line are removed
        assert cmdline.count('"') > 2
        i, j = cmdline.index('"'), cmdline.rindex('"')
        ran = cmdline[:i] + cmdline[i + 1:j] + cmdline[j + 1:]
        assert ran == f'"{app}\\vantage-cli.exe" --self-test', ran


def _exec_argv(value):
    """A desktop entry's Exec value as the Desktop Entry spec reads it: first the string escapes (\\s \\n \\t \\r
    \\\\), then the quoting (inside "...", a backslash escapes " ` $ and itself)."""
    s, i = '', 0
    while i < len(value):
        if value[i] == '\\':
            nxt = value[i + 1]
            assert nxt in 'sntr\\', f'invalid string escape \\{nxt} in {value!r}'   # GLib rejects the whole entry
            s += {'s': ' ', 'n': '\n', 't': '\t', 'r': '\r', '\\': '\\'}[nxt]
            i += 2
        else:
            s += value[i]
            i += 1
    args, cur, quoted, i = [], None, False, 0
    while i < len(s):
        c = s[i]
        if quoted:
            if c == '\\':
                assert s[i + 1] in '"`$\\', s
                cur += s[i + 1]
                i += 1
            elif c == '"':
                quoted = False
            else:
                cur += c
        elif c == '"':
            quoted, cur = True, cur or ''
        elif c == ' ':
            if cur is not None:
                args.append(cur)
            cur = None
        else:
            cur = (cur or '') + c
        i += 1
    if cur is not None:
        args.append(cur)
    return args


def _glib_exec_argv(path):
    """GLib's reading of the Exec line (GKeyFile, then g_shell_parse_argv), in a separate Python; None without
    PyGObject."""
    code = ('import sys\nfrom gi.repository import GLib\nkf = GLib.KeyFile()\n'
            'kf.load_from_file(sys.argv[1], GLib.KeyFileFlags.NONE)\n'
            'print(repr(GLib.shell_parse_argv(kf.get_string("Desktop Entry", "Exec"))[1]))\n')
    for py in ('/usr/bin/python3', sys.executable):
        if not os.path.exists(py):
            continue
        r = subprocess.run([py, '-I', '-c', code, str(path)], capture_output=True, text=True, timeout=60)
        if 'No module named' in r.stderr:
            continue
        assert r.returncode == 0, r.stderr           # GLib refused the entry
        import ast
        return ast.literal_eval(r.stdout.strip())
    return None


def test_exec_line_escaping(tmp_path):
    """install.sh's escaping of the program path for the Exec line, checked against the spec's reading and GLib's
    for every character that needs care."""
    if not shutil.which('sh') or not shutil.which('sed'):
        pytest.skip('needs sh and sed')
    script = _text('packaging', 'linux', 'install.sh')
    sed = next(ln for ln in script.splitlines() if ln.startswith('exe=$(')).split('|', 1)[1].rsplit(')', 1)[0]
    assert sed in _text('packaging', 'linux', 'uninstall.sh')        # uninstall.sh recognises what install.sh wrote
    for path in ('/opt/vantage/Vantage', '/home/a b/.local/opt/vantage/Vantage', '/x/$HOME/V', '/x/q"t/V',
                 '/x/back\\slash/V', '/x/50% `tick`/V'):
        r = subprocess.run(['sh', '-c', f"printf '%s' \"$1\" | {sed}", 'sh', path], capture_output=True, text=True)
        line = f'Exec="{r.stdout}" %f'
        assert _exec_argv(line[5:]) == [path.replace('%', '%%'), '%f'], (path, line)
        entry = tmp_path / 'v.desktop'
        entry.write_text(f'[Desktop Entry]\nType=Application\nName=V\n{line}\n')
        glib = _glib_exec_argv(entry)
        assert glib is None or glib == [path.replace('%', '%%'), '%f'], (path, line, glib)


@pytest.mark.skipif(not sys.platform.startswith('linux') or not shutil.which('sh'), reason='Linux installer')
def test_install_and_uninstall_sh(tmp_path):
    """install.sh from an extracted bundle into a home with a space and a $ in its path, with XDG_DATA_HOME
    elsewhere: the program, the command, the menu entry (Exec quoted per the Desktop Entry spec), the icon and the
    file type; uninstall.sh removes exactly that and leaves an entry `vantage --install-desktop` wrote."""
    import configparser
    from vantage import desktop
    pkg = os.path.join(ROOT, 'packaging', 'linux')
    bundle = tmp_path / 'Vantage-0.1.0-linux-x86_64'
    (bundle / '_internal' / 'vantage' / 'resources' / 'icons').mkdir(parents=True)
    shutil.copy(os.path.join(ROOT, 'vantage', 'resources', 'icons', 'vantage.png'),
                bundle / '_internal' / 'vantage' / 'resources' / 'icons' / 'vantage.png')
    for exe in ('Vantage', 'vantage-cli'):
        p = bundle / exe
        p.write_text('#!/bin/sh\necho "vantage 0.1.0 $0"\n')
        p.chmod(0o755)
    for f in ('install.sh', 'uninstall.sh', 'vantage.desktop', 'vantage-mime.xml'):
        shutil.copy2(os.path.join(pkg, f), bundle / f)
    home = tmp_path / 'my home $USER'
    data = tmp_path / 'data home'
    home.mkdir()
    env = {'HOME': str(home), 'XDG_DATA_HOME': str(data), 'PATH': '/usr/bin:/bin'}
    r = subprocess.run(['sh', str(bundle / 'install.sh')], env=env, capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    dest = home / '.local' / 'opt' / 'vantage'
    assert (dest / 'Vantage').is_file() and (dest / '.vantage-install').is_file()
    link = home / '.local' / 'bin' / 'vantage'
    assert link.is_symlink() and os.readlink(link) == str(dest / 'vantage-cli')
    assert subprocess.run([str(link)], capture_output=True, text=True).stdout.startswith('vantage 0.1.0')
    entry = data / 'applications' / 'vantage.desktop'
    cp = configparser.ConfigParser(interpolation=None)
    cp.optionxform = str
    cp.read(entry)
    e = cp['Desktop Entry']
    assert e['Icon'] == desktop.APP_ID and 'application/x-vantage-project' in e['MimeType']
    assert _exec_argv(e['Exec']) == [str(dest / 'Vantage'), '%f'], e['Exec']
    glib = _glib_exec_argv(entry)                       # what GNOME itself makes of the entry, where GLib is around
    assert glib is None or glib == [str(dest / 'Vantage'), '%f'], glib
    if shutil.which('desktop-file-validate'):
        v = subprocess.run(['desktop-file-validate', str(entry)], capture_output=True, text=True)
        assert v.returncode == 0, v.stdout + v.stderr
    assert (data / 'icons' / 'hicolor' / '512x512' / 'apps' / 'vantage.png').is_file()
    assert '*.vantage' in (data / 'mime' / 'packages' / 'vantage.xml').read_text()
    assert not (home / '.local' / 'share').exists()           # XDG_DATA_HOME was honoured
    # a second install replaces its own; a folder it did not make is left alone
    assert subprocess.run(['sh', str(bundle / 'install.sh')], env=env, capture_output=True).returncode == 0
    r = subprocess.run(['sh', str(dest / 'uninstall.sh')], env=env, capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    assert not dest.exists() and not link.exists() and not entry.exists()
    assert not (data / 'mime' / 'packages' / 'vantage.xml').exists()
    assert not (data / 'icons' / 'hicolor' / '512x512' / 'apps' / 'vantage.png').exists()
    dest.mkdir(parents=True)
    r = subprocess.run(['sh', str(bundle / 'install.sh')], env=env, capture_output=True, text=True)
    assert r.returncode == 1 and 'not made by this installer' in r.stderr
    dest.rmdir()
    # an entry written by `vantage --install-desktop` survives uninstall.sh
    assert subprocess.run(['sh', str(bundle / 'install.sh')], env=env, capture_output=True).returncode == 0
    entry.write_text(desktop.entry('/opt/other/vantage'))
    assert subprocess.run(['sh', str(dest / 'uninstall.sh')], env=env, capture_output=True).returncode == 0
    assert desktop.MARK in entry.read_text() and not dest.exists()


def test_license_notice(tmp_path):
    """The bundles carry the licences of what they collected and a notice naming each component, its version and
    licence (a distribution packaged without its licence file is named as such)."""
    import importlib.metadata as md
    release = _packaging_module('release')
    notice = tmp_path / 'THIRD-PARTY.txt'
    mods = {'yaml', 'PyQt6', 'numpy', 'no_such_module_here'}
    files = release.license_files(mods, str(notice))
    by_mod = md.packages_distributions()
    dists = {d for m in mods for d in by_mod.get(m, ())}
    assert dists, 'no distribution found for PyYAML / PyQt6 / numpy'
    assert (str(notice), 'licenses') in files
    assert all(os.path.isfile(src) for src, _ in files)
    assert all(dest == 'licenses' or dest.split('/')[1] in dists | {'Python'} for _, dest in files), files
    text = notice.read_text(encoding='utf-8')
    assert text.splitlines()[3].startswith(f'Python {sys.version.split()[0]} — ')
    for d in dists:
        assert f'{d} {md.version(d)} — ' in text, d
        if not any(dest.split('/')[1:2] == [d] for _, dest in files):
            assert f'{d} {md.version(d)} — ' in text and 'no licence file' in text
    assert "release.license_files(_top" in _text('packaging', 'vantage.spec')
