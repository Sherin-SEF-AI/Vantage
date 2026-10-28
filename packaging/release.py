"""What a Vantage release is made of, in one place for the PyInstaller spec, packaging/build.py, the GitHub workflow and
tests/test_packaging.py: the package's data files, the pages the main window imports by name, the version and its
CHANGELOG entry.

    python packaging/release.py version                  prints the version (vantage/__init__.py)
    python packaging/release.py notes [--tag vX.Y.Z]     prints the CHANGELOG entry of that version (release notes);
                                                         fails when the tag is not v<version> or the entry is missing

Reads the source files only (no import of the app), so it runs before any requirement is installed.
"""
from __future__ import annotations

import argparse
import ast
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = os.path.join(ROOT, 'vantage')


def version(root: str = ROOT) -> str:
    """__version__ from vantage/__init__.py."""
    with open(os.path.join(root, 'vantage', '__init__.py'), encoding='utf-8') as fh:
        m = re.search(r'''^__version__\s*=\s*['"]([^'"]+)['"]''', fh.read(), re.M)
    if not m:
        raise ValueError('no __version__ in vantage/__init__.py')
    return m.group(1)


# the CARLA bridge runs in CARLA's own Python with the bundle's folder on PYTHONPATH (vantage.sim.server.vantage_source):
# these sources must be there as plain files, not only compiled inside the bundle
BRIDGE_SOURCES = ('vantage/__init__.py', 'vantage/core/__init__.py', 'vantage/core/rosbag.py', 'vantage/core/mathutil.py',
                  'vantage/core/groundtruth.py')


def data_files(root: str = ROOT) -> list[tuple[str, str]]:
    """(source file, destination folder) for every file of the package that is not Python (fonts, icons, the C++
    stitcher and its CMake file), plus as plain files too: the runtime's Python stitcher (exports copy it next to the
    remap tables, so it must exist on disk, not only as compiled code inside the bundle) and the CARLA bridge's sources
    (vantage/sim and BRIDGE_SOURCES, run by CARLA's Python)."""
    out = []
    pkg = os.path.join(root, 'vantage')
    for d, subdirs, files in os.walk(pkg):
        subdirs[:] = sorted(s for s in subdirs if s != '__pycache__')
        rel = os.path.relpath(d, root)
        for f in sorted(files):
            src = os.path.relpath(os.path.join(d, f), root).replace(os.sep, '/')
            if not f.endswith(('.py', '.pyc')) or (f.endswith('.py') and (
                    os.path.basename(d) in ('runtime', 'sim') and rel.replace(os.sep, '/').count('/') == 1
                    or src in BRIDGE_SOURCES)):
                out.append((os.path.join(d, f), rel))
    return out


def page_modules(root: str = ROOT) -> list[str]:
    """The modules mainwindow.PAGE_CLASSES names; the window imports them with importlib when a page is first shown,
    so nothing in the code imports them statically. Read from the source (ast), not by importing the UI."""
    path = os.path.join(root, 'vantage', 'ui', 'mainwindow.py')
    with open(path, encoding='utf-8') as fh:
        tree = ast.parse(fh.read(), path)
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'PAGE_CLASSES'
                                                for t in node.targets):
            table = ast.literal_eval(node.value)
            return [f'vantage.ui.workspaces.{mod}' for mod, _ in table.values()]
    raise ValueError('no PAGE_CLASSES table in vantage/ui/mainwindow.py')


_LICENCE = re.compile(r'(LICEN[CS]E|COPYING|NOTICE)', re.I)


def license_files(modules, notice_path: str) -> list[tuple[str, str]]:
    """(source file, destination folder) for the licence files of every installed distribution that provides one of
    the top-level modules given (what the bundle collected), under licenses/<distribution>/, plus Python's own and a
    notice listing each component with its version and licence, written to notice_path. The downloads carry
    third-party code under GPL, LGPL, BSD, MIT, Apache and other licences, and most of them ask for their notice to
    travel with it."""
    import importlib.metadata as md
    by_mod = md.packages_distributions()
    names = sorted({d for m in modules for d in by_mod.get(m, ())}, key=str.lower)
    out, lines = [], []
    for name in names:
        try:
            dist = md.distribution(name)
        except md.PackageNotFoundError:
            continue
        meta = dist.metadata
        lic = meta.get('License-Expression') or (meta.get('License') or '').strip().splitlines()[0:1]
        lic = lic if isinstance(lic, str) else (lic[0] if lic else '')
        if not lic or len(lic) > 60:
            classifiers = [c.split('::')[-1].strip() for c in meta.get_all('Classifier') or [] if c.startswith('License ::')]
            lic = ', '.join(classifiers) or (lic[:60] + '…' if lic else 'see the files')
        found = 0
        for f in dist.files or []:
            rel = str(f).replace('\\', '/')
            if not (_LICENCE.search(f.name) or '/licenses/' in rel) or rel.endswith(('.py', '.pyc')):
                continue
            src = str(f.locate())
            if not os.path.isfile(src):
                continue
            sub = re.sub(r'^[^/]+\.dist-info/(licenses/)?', '', rel)
            out.append((src, '/'.join(['licenses', name] + ([os.path.dirname(sub)] if os.path.dirname(sub) else []))))
            found += 1
        lines.append(f'{name} {dist.version} — {lic}' + ('' if found else ' (no licence file in its package)'))
    for cand in (os.path.join(os.path.dirname(os.__file__), 'LICENSE.txt'), os.path.join(sys.base_prefix, 'LICENSE.txt'),
                 os.path.join(sys.base_prefix, 'share', 'doc', 'python3', 'copyright')):
        if os.path.isfile(cand):
            out.append((cand, 'licenses/Python'))
            break
    lines.insert(0, f'Python {sys.version.split()[0]} — PSF-2.0')
    with open(notice_path, 'w', encoding='utf-8') as fh:
        fh.write(f'Vantage {version()} bundles these open-source components. Each one\'s licence is in the folder of '
                 'the same name next to this file;\nVantage\'s own source is at https://github.com/Sherin-SEF-AI/Vantage\n\n'
                 + '\n'.join(lines) + '\n\nThe system libraries PyInstaller copied from the build machine (OpenSSL, SQLite, '
                 'libxcb, GLib, fontconfig and the like on Linux) are under their own licences, as their '
                 'distributions publish them.\n')
    out.append((notice_path, 'licenses'))
    return out


_HEAD = re.compile(r'^## +\[?v?(\d+\.\d+\.\d+[^\]\s]*)\]?(.*)$')


def changelog(root: str = ROOT) -> list[tuple[str, str, str]]:
    """[(version, rest of the heading line, body)] from CHANGELOG.md, newest first."""
    with open(os.path.join(root, 'CHANGELOG.md'), encoding='utf-8') as fh:
        lines = fh.read().splitlines()
    out, cur = [], None
    for ln in lines:
        m = _HEAD.match(ln)
        if m:
            cur = [m.group(1), m.group(2).strip(), []]
            out.append(cur)
        elif cur is not None:
            cur[2].append(ln)
    return [(v, head, '\n'.join(body).strip() + '\n') for v, head, body in out]


def notes(ver: str, root: str = ROOT) -> str:
    """The CHANGELOG entry of version ver (its body, without the heading)."""
    for v, _, body in changelog(root):
        if v == ver:
            return body
    raise ValueError(f'CHANGELOG.md has no entry for {ver}')


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('version')
    n = sub.add_parser('notes')
    n.add_argument('--tag', default='', help='the git tag being released; must be v<version>')
    args = ap.parse_args(argv)
    ver = version()
    if args.cmd == 'version':
        print(ver)
        return 0
    if args.tag and args.tag != f'v{ver}':
        print(f'release.py: tag {args.tag} does not match the version {ver} in vantage/__init__.py (tag v{ver})',
              file=sys.stderr)
        return 1
    entries = changelog()
    if not entries or entries[0][0] != ver:
        print(f'release.py: the top entry of CHANGELOG.md is {entries[0][0] if entries else "missing"}, not {ver}',
              file=sys.stderr)
        return 1
    sys.stdout.buffer.write(notes(ver).encode('utf-8'))     # UTF-8 whatever the console's code page
    sys.stdout.flush()
    return 0


if __name__ == '__main__':
    sys.exit(main())
