"""The working folder: where Vantage writes what it makes without asking each time — sample datasets (the page and the
assistant), exports of a project that has not been saved yet. A preference (Preferences ▸ General) kept in the
settings; ~/Documents/Vantage until it is changed. Inside it every project gets a folder named after it:

    <working folder>/<project>/datasets/<variant>-<YYYYMMDD-HHMMSS>/     one sample dataset per run, never overwritten
    <working folder>/<project>/exports/<variant>/                        exports of an unsaved project
"""
from __future__ import annotations

import datetime as _dt
import os
import re

KEY = 'paths/working'


def default() -> str:
    return os.path.join(os.path.expanduser('~'), 'Documents', 'Vantage')


def get(settings=None) -> str:
    """The saved working folder (absolute), or the default."""
    v = settings.value(KEY, '') if settings is not None else ''
    v = str(v or '').strip()
    return os.path.abspath(os.path.expanduser(v)) if v else default()


def put(settings, path: str) -> str:
    """Save path as the working folder (an empty path goes back to the default); returns what is now in use."""
    path = str(path or '').strip()
    settings.setValue(KEY, os.path.abspath(os.path.expanduser(path)) if path else '')
    return get(settings)


def slug(s: str) -> str:
    t = re.sub(r'[^a-z0-9]+', '-', (s or '').lower()).strip('-')
    return t or 'untitled'


def project_dir(project, base: str | None = None, *parts) -> str:
    """<working folder>/<project slug>/<parts…>; base overrides the working folder."""
    return os.path.join(base or default(), slug(project.name), *parts)


def dataset_dir(project, variant=None, base: str | None = None, when: _dt.datetime | None = None) -> str:
    """A new folder for one sample dataset: <working folder>/<project>/datasets/<variant>-<YYYYMMDD-HHMMSS> (a second
    run within the same second gets -2, -3 … so nothing is overwritten)."""
    v = variant or project.variant
    stamp = (when or _dt.datetime.now()).strftime('%Y%m%d-%H%M%S')
    root = project_dir(project, base, 'datasets')
    out = os.path.join(root, f'{slug(v.name)}-{stamp}')
    k = 2
    while os.path.exists(out):
        out = os.path.join(root, f'{slug(v.name)}-{stamp}-{k}')
        k += 1
    return out


def display(path: str, base: str | None = None) -> str:
    """path relative to the working folder when it lies inside it (what the assistant may be told), else its name."""
    base = os.path.abspath(base or default())
    p = os.path.abspath(path)
    if os.path.commonpath([p, base]) == base:
        return os.path.relpath(p, base)
    return os.path.basename(p)
