"""Where the API key lives: the GROQ_API_KEY environment variable if set, else the system keyring (when the optional
keyring package and a keyring service are available), else a private file in the user's settings folder that only the
user can read. Never in project files or the settings file."""
from __future__ import annotations

import os
import sys

ENV = 'GROQ_API_KEY'
SERVICE, ACCOUNT = 'Vantage', 'groq-api-key'
FILE = 'assistant-key'

SOURCES = {'environment': f'from the {ENV} environment variable', 'keyring': 'in the system keyring',
           'file': 'in a private file in your settings folder (readable only by you)'}


def config_dir() -> str:
    if sys.platform.startswith('win'):
        root = os.environ.get('APPDATA') or os.path.expanduser('~')
    elif sys.platform == 'darwin':
        root = os.path.expanduser('~/Library/Application Support')
    else:
        root = os.environ.get('XDG_CONFIG_HOME') or os.path.expanduser('~/.config')
    return os.path.join(root, 'Vantage')


def _path() -> str:
    return os.path.join(config_dir(), FILE)


def _keyring():
    if os.environ.get('VANTAGE_NO_KEYRING'):
        return None
    try:
        import keyring
        from keyring.backends import fail
        if isinstance(keyring.get_keyring(), fail.Keyring):
            return None
        return keyring
    except Exception:
        return None


def load() -> tuple[str | None, str | None]:
    """(key, source) — source is 'environment', 'keyring', 'file' or None."""
    k = os.environ.get(ENV, '').strip()
    if k:
        return k, 'environment'
    kr = _keyring()
    if kr is not None:
        try:
            k = (kr.get_password(SERVICE, ACCOUNT) or '').strip()
            if k:
                return k, 'keyring'
        except Exception:
            pass
    try:
        with open(_path(), encoding='utf-8') as f:
            k = f.read().strip()
        if k:
            return k, 'file'
    except OSError:
        pass
    return None, None


def save(key: str) -> str:
    """Store the key; returns where it went."""
    key = key.strip()
    kr = _keyring()
    if kr is not None:
        try:
            kr.set_password(SERVICE, ACCOUNT, key)
            _remove_file()
            return 'keyring'
        except Exception:
            pass
    os.makedirs(config_dir(), exist_ok=True)
    path = _path()
    tmp = path + '.tmp'
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, 'w', encoding='utf-8') as f:
        f.write(key)
    os.replace(tmp, path)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    return 'file'


def delete():
    kr = _keyring()
    if kr is not None:
        try:
            kr.delete_password(SERVICE, ACCOUNT)
        except Exception:
            pass
    _remove_file()


def _remove_file():
    try:
        os.remove(_path())
    except OSError:
        pass


def mask(key: str | None) -> str:
    if not key:
        return ''
    return f'{key[:4]}…{key[-4:]}' if len(key) > 12 else '…' + key[-2:]
