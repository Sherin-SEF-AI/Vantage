"""Build the Vantage bundle for this platform, smoke-test it and pack it for download.

    python packaging/build.py [--dist DIR] [--work DIR] [--smoke] [--no-archive]

Needs PyInstaller and the bundle's requirements in the same Python (pip install -r packaging/requirements-bundle.txt).
Writes into DIR (default dist/):
    Linux    Vantage/ and Vantage-<version>-linux-<arch>.tar.gz (the folder plus install.sh / uninstall.sh)
    Windows  Vantage/, Vantage-<version>-windows-x64.zip and, when Inno Setup's iscc is on PATH or in its default
             folder, Vantage-<version>-windows-x64-setup.exe
    macOS    Vantage.app and Vantage-<version>-macos-<arch>.zip
and prints each archive's size and SHA-256.
--smoke runs the bundled app: --version, a batch analysis of the sedan template and --self-test (QT_QPA_PLATFORM is
set to offscreen for them, so no window opens). In a bundle every self-test check must pass without a warning, except
those in selftest.BUNDLE_MAY_WARN (OpenGL: build machines have no GPU), so a library or module that did not make it
into the bundle fails the build.
"""
import argparse
import hashlib
import os
import platform
import shutil
import subprocess
import sys
import tarfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import release  # noqa: E402

VERSION = release.version(ROOT)


def arch():
    m = platform.machine().lower()
    if os.name == 'nt':
        return {'amd64': 'x64', 'x86_64': 'x64', 'arm64': 'arm64'}.get(m, m)
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


def self_test_problems(text: str) -> list[str]:
    """What a bundle's self-test output says is wrong: a FAILED or (outside BUNDLE_MAY_WARN) warning line, a check
    that did not run, or no final verdict."""
    sys.path.insert(0, ROOT)
    from vantage.selftest import BUNDLE_MAY_WARN, FAIL, NAMES, OK
    lines = [ln.rstrip() for ln in text.splitlines()]
    seen, bad = set(), []
    for ln in lines:
        if not ln.startswith('  '):
            continue
        for name in NAMES:
            rest = ln[2:]
            if rest.startswith(name + ' ') and rest[len(name):].strip():
                status = rest[12:].split()[0] if len(rest) > 12 else ''
                seen.add(name)
                if status == FAIL or (status != OK and name not in BUNDLE_MAY_WARN):
                    bad.append(ln.strip())
                break
    bad += [f'{n}: did not run' for n in NAMES if n not in seen]
    if not any(ln.startswith('self-test passed') for ln in lines):
        bad.append('no "self-test passed" line')
    return bad


def smoke(dist, work):
    cli = cli_path(dist)
    env = dict(os.environ, QT_QPA_PLATFORM='offscreen')
    out = run([cli, '--version'], env=env, capture_output=True, text=True).stdout.strip()
    if out != f'vantage {VERSION}':
        raise SystemExit(f'--version printed {out!r}, not vantage {VERSION}')
    report = os.path.join(work, 'smoke-analysis.json')
    r = subprocess.run([cli, '--template', 'sedan', '--analyse', report], env=env)
    if r.returncode not in (0, 2) or not os.path.getsize(report):     # 2: the plan has critical issues, still a run
        raise SystemExit(f'batch analysis failed (exit {r.returncode})')
    print('+', cli, '--self-test', flush=True)
    r = subprocess.run([cli, '--self-test'], env=env, capture_output=True, text=True, encoding='utf-8',
                       errors='replace', timeout=900)
    sys.stdout.write(r.stdout)
    if r.stderr.strip():
        print('stderr:\n' + r.stderr.strip()[-4000:], flush=True)
    bad = self_test_problems(r.stdout)
    if r.returncode or bad:
        raise SystemExit(f'self-test exit {r.returncode}; in a bundle every check must pass:\n  ' + '\n  '.join(bad))
    print('smoke test passed', flush=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for block in iter(lambda: fh.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def archive(dist):
    tag = f'Vantage-{VERSION}'
    if os.path.isdir(os.path.join(dist, 'Vantage')):                 # Vantage's own licence at the top of the folder
        shutil.copy2(os.path.join(ROOT, 'LICENSE'), os.path.join(dist, 'Vantage', 'LICENSE.txt'))
    if sys.platform.startswith('linux'):
        folder = os.path.join(dist, 'Vantage')
        for f in ('install.sh', 'uninstall.sh', 'vantage.desktop', 'vantage-mime.xml'):
            shutil.copy2(os.path.join(HERE, 'linux', f), folder)
        for f in ('install.sh', 'uninstall.sh'):
            os.chmod(os.path.join(folder, f), 0o755)
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
            run([iscc, f'/DAppVersion={VERSION}', f'/DBundleDir={os.path.join(dist, "Vantage")}',
                 f'/O{dist}', os.path.join(HERE, 'windows', 'vantage.iss')])
            outs.append(os.path.join(dist, f'{tag}-windows-x64-setup.exe'))
        else:
            print('Inno Setup (iscc) not found: no installer, only the zip', flush=True)
        return outs
    if sys.platform == 'darwin':
        out = os.path.join(dist, f'{tag}-macos-{arch()}.zip')
        # ditto keeps the app's symlinks, permissions and code signature (zip -r does not)
        run(['ditto', '-c', '-k', '--sequesterRsrc', '--keepParent', os.path.join(dist, 'Vantage.app'), out])
        return [out]
    return []


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--dist', default=os.path.join(ROOT, 'dist'))
    ap.add_argument('--work', default=os.path.join(ROOT, 'build'))
    ap.add_argument('--smoke', action='store_true', help='run the bundled app after building')
    ap.add_argument('--no-archive', action='store_true')
    ap.add_argument('--no-build', action='store_true', help='smoke-test and pack the bundle already in --dist')
    args = ap.parse_args()
    dist, work = os.path.abspath(args.dist), os.path.abspath(args.work)
    os.makedirs(work, exist_ok=True)
    if not args.no_build:
        build(dist, work)
    if args.smoke:
        smoke(dist, work)
    if not args.no_archive:
        for p in archive(dist):
            print(f'{os.path.basename(p)}  {os.path.getsize(p) / 1e6:.0f} MB  sha256 {sha256(p)}', flush=True)


if __name__ == '__main__':
    main()
