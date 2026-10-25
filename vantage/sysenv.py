"""The environment for a program of the system (gst-launch-1.0, ffmpeg, xdg-user-dir, gio …) started from Vantage.

Inside the downloadable bundle PyInstaller points LD_LIBRARY_PATH (and a few plug-in paths) at the bundle's own
libraries, and a child process inherits that: a system program would then load the bundle's GLib, zlib or OpenSSL in
place of the system's and fail or crash. system_env() gives the environment the user's session had. From a source
checkout or a pip install it is simply os.environ.
"""
from __future__ import annotations

import os
import sys


def frozen() -> bool:
    return bool(getattr(sys, 'frozen', False))


def system_env(base: dict | None = None) -> dict:
    env = dict(os.environ if base is None else base)
    if not frozen():
        return env
    bundle = os.path.normcase(os.path.abspath(getattr(sys, '_MEIPASS', os.path.dirname(sys.executable))))
    for key in ('LD_LIBRARY_PATH', 'DYLD_LIBRARY_PATH'):
        orig = env.pop(key + '_ORIG', None)        # PyInstaller keeps the session's value here
        if orig:
            env[key] = orig
        else:
            env.pop(key, None)
    for key, val in list(env.items()):              # plug-in paths a runtime hook pointed into the bundle
        if key.endswith(('_PATH', '_DIR', '_PREFIX')) and val and any(
                os.path.normcase(os.path.abspath(p)).startswith(bundle) for p in val.split(os.pathsep) if p):
            del env[key]
    return env
