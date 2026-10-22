"""Build the Vantage bundle for this platform, smoke-test it and pack it for download.

    python packaging/build.py [--dist DIR] [--work DIR] [--smoke] [--no-archive]

Needs PyInstaller (pip install pyinstaller) and the app's requirements in the same Python. Writes into DIR (default
dist/):
    Linux    Vantage/ and Vantage-<version>-linux-<arch>.tar.gz (the folder plus install.sh / uninstall.sh)
    Windows  Vantage/, Vantage-<version>-windows-<arch>.zip and, when Inno Setup's iscc is on PATH or in its default
             folder, Vantage-<version>-windows-x64-setup.exe
    macOS    Vantage.app and Vantage-<version>-macos-<arch>.zip
--smoke runs the bundled app: --version, a batch analysis of the sedan template and --self-test (QT_QPA_PLATFORM is
set to offscreen for them, so no window opens).
"""
import argparse
import os
import platform
import shutil
import subprocess
import sys
import tarfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from vantage import __version__  # noqa: E402


def arch():
    m = platform.machine().lower()
    return {'amd64': 'x86_64', 'x64': 'x86_64', 'aarch64': 'arm64'}.get(m, m)


def run(cmd, **kw):
    print('+', ' '.join(str(c) for c in cmd), flush=True)
    return subprocess.run(cmd, check=True, **kw)


def build(dist, work):
    run([sys.executable, '-m', 'PyInstaller', '--noconfirm', '--clean', '--distpath', dist, '--workpath', work,
         os.path.join(HERE, 'vantage.spec')])


def cli_path(dist):
    if sys.platform == 'darwin':
        return os.path.join(dist, 'Vantage.app', 'Contents', 'MacOS', 'vantage-cli')
    return os.path.join(dist, 'Vantage', 'vantage-cli' + ('.exe' if os.name == 'nt' else ''))


def smoke(dist, work):
    cli = cli_path(dist)
    env = dict(os.environ, QT_QPA_PLATFORM='offscreen')
    out = run([cli, '--version'], env=env, capture_output=True, text=True).stdout.strip()
    if out != f'vantage {__version__}':
        raise SystemExit(f'--version printed {out!r}')
    report = os.path.join(work, 'smoke-analysis.json')
    r = subprocess.run([cli, '--template', 'sedan', '--analyse', report], env=env)
    if r.returncode not in (0, 2) or not os.path.getsize(report):     # 2: the plan has critical issues, still a run
        raise SystemExit(f'batch analysis failed (exit {r.returncode})')
    run([cli, '--self-test'], env=env)
    print('smoke test passed', flush=True)


def archive(dist):
    tag = f'Vantage-{__version__}'
    if sys.platform.startswith('linux'):
        folder = os.path.join(dist, 'Vantage')
        for f in ('install.sh', 'uninstall.sh', 'vantage.desktop', 'vantage-mime.xml'):
            shutil.copy2(os.path.join(HERE, 'linux', f), folder)
        out = os.path.join(dist, f'{tag}-linux-{arch()}.tar.gz')
        with tarfile.open(out, 'w:gz') as tf:
            tf.add(folder, arcname=f'{tag}-linux-{arch()}')
        return [out]
    if os.name == 'nt':
        outs = [shutil.make_archive(os.path.join(dist, f'{tag}-windows-{arch()}'), 'zip', dist, 'Vantage')]
        iscc = shutil.which('iscc') or next((p for p in (r'C:\Program Files (x86)\Inno Setup 6\ISCC.exe',
                                                          r'C:\Program Files\Inno Setup 6\ISCC.exe')
                                             if os.path.exists(p)), None)
        if iscc:
            run([iscc, f'/DAppVersion={__version__}', f'/DBundleDir={os.path.join(dist, "Vantage")}',
                 f'/O{dist}', os.path.join(HERE, 'windows', 'vantage.iss')])
            outs.append(os.path.join(dist, f'{tag}-windows-x64-setup.exe'))
        else:
            print('Inno Setup (iscc) not found: no installer, only the zip', flush=True)
        return outs
    if sys.platform == 'darwin':
        out = os.path.join(dist, f'{tag}-macos-{arch()}.zip')
        run(['ditto', '-c', '-k', '--keepParent', os.path.join(dist, 'Vantage.app'), out])
        return [out]
    return []


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--dist', default=os.path.join(ROOT, 'dist'))
    ap.add_argument('--work', default=os.path.join(ROOT, 'build'))
    ap.add_argument('--smoke', action='store_true', help='run the bundled app after building')
    ap.add_argument('--no-archive', action='store_true')
    args = ap.parse_args()
    dist, work = os.path.abspath(args.dist), os.path.abspath(args.work)
    build(dist, work)
    if args.smoke:
        smoke(dist, work)
    if not args.no_archive:
        for p in archive(dist):
            print(f'{p}  {os.path.getsize(p) / 1e6:.0f} MB', flush=True)


if __name__ == '__main__':
    main()
