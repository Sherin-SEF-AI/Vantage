"""Where the API keys live, one per provider (Groq, Claude): the provider's environment variable if set
(GROQ_API_KEY, ANTHROPIC_API_KEY), else the system keyring (when the optional keyring package and a keyring service are
available), else a private file in the user's settings folder that only the user can read. Never in project files or
the settings file."""
from __future__ import annotations

import os
import sys

PROVIDERS = ('groq', 'claude')
# provider -> (environment variable, keyring account, file name in the settings folder)
WHERE = {'groq': ('GROQ_API_KEY', 'groq-api-key', 'assistant-key'),
         'claude': ('ANTHROPIC_API_KEY', 'anthropic-api-key', 'assistant-key-anthropic')}
ENV = WHERE['groq'][0]
SERVICE, ACCOUNT = 'Vantage', WHERE['groq'][1]
FILE = WHERE['groq'][2]


def sources(provider='groq') -> dict:
    env = WHERE[_p(provider)][0]
    return {'environment': f'from the {env} environment variable', 'keyring': 'in the system keyring',
            'file': 'in a private file in your settings folder (readable only by you)'}


def _p(provider) -> str:
    if provider not in WHERE:
        raise KeyError(f'unknown provider {provider!r}')
    return provider


SOURCES = sources('groq')


def env_name(provider='groq') -> str:
    return WHERE[_p(provider)][0]


def config_dir() -> str:
    if sys.platform.startswith('win'):
        root = os.environ.get('APPDATA') or os.path.expanduser('~')
    elif sys.platform == 'darwin':
        root = os.path.expanduser('~/Library/Application Support')
    else:
        root = os.environ.get('XDG_CONFIG_HOME') or os.path.expanduser('~/.config')
    return os.path.join(root, 'Vantage')


def _path(provider='groq') -> str:
    return os.path.join(config_dir(), WHERE[_p(provider)][2])


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


def load(provider='groq') -> tuple[str | None, str | None]:
    """(key, source) — source is 'environment', 'keyring', 'file' or None."""
    env, account, _ = WHERE[_p(provider)]
    k = os.environ.get(env, '').strip()
    if k:
        return k, 'environment'
    kr = _keyring()
    if kr is not None:
        try:
            k = (kr.get_password(SERVICE, account) or '').strip()
            if k:
                return k, 'keyring'
        except Exception:
            pass
    try:
        with open(_path(provider), encoding='utf-8') as f:
            k = f.read().strip()
        if k:
            return k, 'file'
    except OSError:
        pass
    return None, None


def save(key: str, provider='groq') -> str:
    """Store the key; returns where it went."""
    key = key.strip()
    _, account, _ = WHERE[_p(provider)]
    kr = _keyring()
    if kr is not None:
        try:
            kr.set_password(SERVICE, account, key)
            _remove_file(provider)
            return 'keyring'
        except Exception:
            pass
    os.makedirs(config_dir(), exist_ok=True)
    path = _path(provider)
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


def delete(provider='groq'):
    _, account, _ = WHERE[_p(provider)]
    kr = _keyring()
    if kr is not None:
        try:
            kr.delete_password(SERVICE, account)
        except Exception:
            pass
    _remove_file(provider)


def _remove_file(provider='groq'):
    try:
        os.remove(_path(provider))
    except OSError:
        pass


def mask(key: str | None) -> str:
    if not key:
        return ''
    return f'{key[:4]}…{key[-4:]}' if len(key) > 12 else '…' + key[-2:]
