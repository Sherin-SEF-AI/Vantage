"""The menu entry and icon for Vantage run from a source checkout or a pip install on Linux.

A Linux desktop (GNOME, KDE, …) shows an application's icon in its dock, app grid and window switcher from an installed
.desktop file whose name matches the id the window announces: QGuiApplication.setDesktopFileName('vantage') in app.py,
so vantage.desktop. The packaged bundle's install.sh writes one; `vantage --install-desktop` writes the same for this
installation into the user's data folder ($XDG_DATA_HOME, default ~/.local/share), with the icon at the theme's sizes
and the .vantage file type; `vantage --desktop-shortcut` also puts a launcher on the desktop (marked trusted, so GNOME's
desktop icons start it on a double-click), and `vantage --uninstall-desktop` removes what they wrote (and nothing
else).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys

from .sysenv import system_env

APP_ID = 'vantage'
MARK = 'X-Vantage-Installed-By=vantage --install-desktop'
SIZES = (16, 24, 32, 48, 64, 128, 256, 512)
MIME_XML = '''<?xml version="1.0" encoding="UTF-8"?>
<mime-info xmlns="http://www.freedesktop.org/standards/shared-mime-info">
  <mime-type type="application/x-vantage-project">
    <comment>Vantage project</comment>
    <sub-class-of type="application/zip"/>
    <glob pattern="*.vantage"/>
  </mime-type>
</mime-info>
'''
MIME_MARK = '<!-- written by vantage --install-desktop -->\n'


def data_home() -> str:
    return os.environ.get('XDG_DATA_HOME') or os.path.join(os.path.expanduser('~'), '.local', 'share')


def command() -> str:
    """The Exec line that starts this Vantage: run_vantage.py in a checkout, else the installed `vantage` command,
    else `python -m vantage` with this interpreter."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    run = os.path.join(root, 'run_vantage.py')
    if os.path.isfile(run):
        return f'{_q(sys.executable)} {_q(run)}'
    exe = shutil.which('vantage')
    if exe:
        return _q(exe)
    return f'{_q(sys.executable)} -m vantage'


def _q(s: str) -> str:
    """Quote an Exec argument as the Desktop Entry spec wants it: inside double quotes " ` $ \\ take a backslash, and
    that backslash is itself escaped because the value is unescaped as a string first (GLib rejects \\$ or \\" as
    string escapes); % is doubled (field codes)."""
    s = s.replace('%', '%%')
    if not any(c in s for c in ' \t"\'\\$`<>~|&;*?#()'):
        return s
    return '"' + s.replace('\\', '\\\\\\\\').replace('"', '\\\\"').replace('$', '\\\\$').replace('`', '\\\\`') + '"'


def _read(path: str) -> str:
    with open(path, encoding='utf-8', errors='replace') as fh:
        return fh.read()


def entry(exec_cmd: str) -> str:
    return '\n'.join([
        '[Desktop Entry]',
        'Type=Application',
        'Name=Vantage',
        'GenericName=Sensor rig planner',
        'Comment=Design, simulate and check the sensor rigs of autonomous-driving data-collection vehicles',
        f'Exec={exec_cmd} %f',
        f'Icon={APP_ID}',
        'Terminal=false',
        'Categories=Science;Engineering;',
        'MimeType=application/x-vantage-project;',
        'Keywords=sensor rig;camera;lidar;radar;GNSS;calibration;data collection;autonomous driving;',
        'StartupWMClass=Vantage',
        'StartupNotify=true',
        MARK,
        ''])


def paths(base: str | None = None) -> dict:
    base = base or data_home()
    return {'desktop': os.path.join(base, 'applications', f'{APP_ID}.desktop'),
            'icons': [os.path.join(base, 'icons', 'hicolor', f'{n}x{n}', 'apps', f'{APP_ID}.png') for n in SIZES],
            'mime': os.path.join(base, 'mime', 'packages', f'{APP_ID}.xml')}


def install(base: str | None = None, icon_png: str | None = None) -> list[str]:
    """Write the menu entry, the icon at every theme size and the file type; returns the paths written. Leaves a
    vantage.desktop alone that this command did not write (the bundle's installer, a package)."""
    from .app import APP_ICON
    p = paths(base)
    if os.path.exists(p['desktop']) and MARK not in _read(p['desktop']):
        raise FileExistsError(f'{p["desktop"]} was not written by vantage --install-desktop (an installed bundle or '
                              f'package): left as it is.')
    written = []
    os.makedirs(os.path.dirname(p['desktop']), exist_ok=True)
    with open(p['desktop'], 'w', encoding='utf-8') as fh:
        fh.write(entry(command()))
    written.append(p['desktop'])
    for path, n in zip(p['icons'], SIZES):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        _icon(icon_png or APP_ICON, n, path)
        written.append(path)
    os.makedirs(os.path.dirname(p['mime']), exist_ok=True)
    with open(p['mime'], 'w', encoding='utf-8') as fh:
        fh.write(MIME_XML.replace('?>\n', '?>\n' + MIME_MARK, 1))
    written.append(p['mime'])
    _refresh(base or data_home())
    return written


def uninstall(base: str | None = None, desktop: str | None = None) -> list[str]:
    """Remove what install() and shortcut() wrote; returns the paths removed."""
    p = paths(base)
    removed = []
    if os.path.exists(p['desktop']):
        if MARK not in _read(p['desktop']):
            raise FileExistsError(f'{p["desktop"]} was not written by vantage --install-desktop: left as it is.')
        os.remove(p['desktop'])
        removed.append(p['desktop'])
        for path in p['icons']:
            if os.path.exists(path):
                os.remove(path)
                removed.append(path)
    if os.path.exists(p['mime']) and MIME_MARK in _read(p['mime']):
        os.remove(p['mime'])
        removed.append(p['mime'])
    sc = os.path.join(desktop_dir() if desktop is None else desktop, f'{APP_ID}.desktop')
    if os.path.exists(sc) and MARK in _read(sc):
        os.remove(sc)
        removed.append(sc)
    if removed:
        _refresh(base or data_home())
    return removed


def desktop_dir() -> str:
    """The user's desktop folder (xdg-user-dir DESKTOP, else ~/Desktop)."""
    if shutil.which('xdg-user-dir'):
        try:
            d = subprocess.run(['xdg-user-dir', 'DESKTOP'], capture_output=True, text=True, timeout=10,
                               env=system_env()).stdout.strip()
            if d:
                return d
        except (OSError, subprocess.SubprocessError):
            pass
    return os.path.join(os.path.expanduser('~'), 'Desktop')


def shortcut(folder: str | None = None, base: str | None = None) -> list[str]:
    """A launcher on the desktop (and the menu entry and icons it needs, through install()); returns the paths
    written."""
    written = install(base)
    folder = folder or desktop_dir()
    path = os.path.join(folder, f'{APP_ID}.desktop')
    if os.path.exists(path) and MARK not in _read(path):
        raise FileExistsError(f'{path} was not written by vantage: left as it is.')
    os.makedirs(folder, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write(entry(command()))
    os.chmod(path, 0o755)                                     # a launcher must be executable to start
    if shutil.which('gio'):                                    # and trusted, for GNOME's desktop icons
        subprocess.run(['gio', 'set', path, 'metadata::trusted', 'true'], env=system_env(), stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, check=False, timeout=10)
    return written + [path]


def _icon(src: str, n: int, dst: str):
    """The 512 px icon scaled to n px (smooth), as PNG."""
    from PyQt6.QtCore import Qt
    from PyQt6.QtGui import QGuiApplication, QImage
    QGuiApplication.instance() or QGuiApplication(sys.argv[:1])
    img = QImage(src)
    if img.isNull():
        raise OSError(f'cannot read the icon {src}')
    if img.width() != n:
        img = img.scaled(n, n, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
    if not img.save(dst, 'PNG'):
        raise OSError(f'cannot write {dst}')


def _refresh(base: str):
    """Ask the desktop to read the new entry, icons and file type (each tool is optional)."""
    for cmd in (['update-desktop-database', os.path.join(base, 'applications')],
                ['gtk-update-icon-cache', '-f', '-t', os.path.join(base, 'icons', 'hicolor')],
                ['update-mime-database', os.path.join(base, 'mime')]):
        if shutil.which(cmd[0]):
            subprocess.run(cmd, env=system_env(), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
                           timeout=30)
