"""Passwords in stream URLs stay out of project files.

A camera stream such as rtsp://admin:secret@10.0.0.5:554/stream1 is saved in the project as
rtsp://admin@10.0.0.5:554/stream1; the password goes to a private file in the user's settings folder (readable only by
the user), keyed by user@host:port, and is put back when the stream is opened. A project shared with someone else
therefore carries no camera passwords: they enter their own once."""
from __future__ import annotations

import json
import os
import re
import sys

FILE = 'stream-credentials.json'
_URL = re.compile(r'^(?P<scheme>[a-z][a-z0-9+.-]*://)(?P<user>[^/@:\s]+)(?::(?P<pw>[^@/\s]*))?@(?P<host>[^/\s]+)(?P<rest>.*)$',
                  re.IGNORECASE)
_IN_TEXT = re.compile(r'([a-z][a-z0-9+.-]*://[^/@:\s]+):[^@/\s]*@', re.IGNORECASE)


def _dir() -> str:
    if sys.platform.startswith('win'):
        root = os.environ.get('APPDATA') or os.path.expanduser('~')
    elif sys.platform == 'darwin':
        root = os.path.expanduser('~/Library/Application Support')
    else:
        root = os.environ.get('XDG_CONFIG_HOME') or os.path.expanduser('~/.config')
    return os.path.join(root, 'Vantage')


def _path() -> str:
    return os.path.join(_dir(), FILE)


def _load() -> dict:
    try:
        with open(_path(), encoding='utf-8') as f:
            d = json.load(f)
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def _save(d: dict):
    os.makedirs(_dir(), exist_ok=True)
    tmp = _path() + '.tmp'
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, 'w', encoding='utf-8') as f:
        json.dump(d, f, indent=1, sort_keys=True)
    os.replace(tmp, _path())
    try:
        os.chmod(_path(), 0o600)
    except OSError:
        pass


def split(url: str):
    """(url without the password, 'user@host' key or None, password or None)."""
    m = _URL.match((url or '').strip())
    if not m:
        return url, None, None
    key = f'{m["user"]}@{m["host"]}'.lower()
    clean = f'{m["scheme"]}{m["user"]}@{m["host"]}{m["rest"]}'
    return clean, key, m['pw'] if m['pw'] else None


def remember(url: str) -> str:
    """The URL to save in a project: any password moves to the private store."""
    clean, key, pw = split(url)
    if key is not None and pw is not None:
        d = _load()
        if d.get(key) != pw:
            d[key] = pw
            _save(d)
    return clean


def resolve(url: str) -> str:
    """The URL to open: a stored password is put back when the URL names a user without one."""
    m = _URL.match((url or '').strip())
    if not m or m['pw']:
        return url
    pw = _load().get(f'{m["user"]}@{m["host"]}'.lower())
    if not pw:
        return url
    return f'{m["scheme"]}{m["user"]}:{pw}@{m["host"]}{m["rest"]}'


def mask(text: str) -> str:
    """Text with the password of every URL in it shown as ***: for labels, messages, reports and the assistant."""
    return _IN_TEXT.sub(r'\1:***@', text or '')


def forget(url: str):
    _, key, _ = split(url)
    d = _load()
    if key in d:
        del d[key]
        _save(d)
