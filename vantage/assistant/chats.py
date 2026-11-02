"""The assistant's past conversations, kept on this computer: one JSON file per conversation in <config>/chats
(readable by the user alone), saved after every answer. A conversation keeps its transcript as the panel shows it
(Save conversation's markdown: masked, no keys), its title (the first question), the project it was about, the
provider and model and when it started and last changed. Nothing here is ever sent: continuing a conversation
starts a new one with a hand-over summary of it, as a change of provider does."""
from __future__ import annotations

import json
import os
import re
import time
import uuid

from .secrets import config_dir

MAX_CHATS = 400                 # the oldest beyond this are removed
TITLE_MAX = 90


def chat_dir() -> str:
    return os.path.join(config_dir(), 'chats')


def new_id() -> str:
    return time.strftime('%Y%m%d-%H%M%S-') + uuid.uuid4().hex[:6]


def _path(cid: str) -> str:
    if not re.fullmatch(r'[0-9A-Za-z_-]{6,64}', cid or ''):
        raise ValueError(f'not a chat id: {cid!r}')
    return os.path.join(chat_dir(), f'{cid}.json')


def title_of(question: str) -> str:
    t = ' '.join(str(question or '').split())
    return t if len(t) <= TITLE_MAX else t[:TITLE_MAX - 1] + '…'


def save(cid: str, *, title: str, markdown: str, project: str = '', project_path: str = '', provider: str = '',
         model: str = '', started: float | None = None) -> dict | None:
    """Write or update a conversation; None when there is nothing to keep."""
    if not markdown.strip():
        return None
    d = chat_dir()
    os.makedirs(d, mode=0o700, exist_ok=True)
    old = load(cid) or {}
    rec = {'id': cid, 'title': title or old.get('title') or 'Conversation', 'project': project,
           'project_path': project_path, 'provider': provider, 'model': model,
           'started': started or old.get('started') or time.time(), 'updated': time.time(), 'markdown': markdown}
    tmp = _path(cid) + '.tmp'
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, 'w', encoding='utf-8') as f:
        json.dump(rec, f, ensure_ascii=False)
    os.replace(tmp, _path(cid))
    _prune()
    return rec


def load(cid: str) -> dict | None:
    try:
        with open(_path(cid), encoding='utf-8') as f:
            d = json.load(f)
        return d if isinstance(d, dict) and d.get('id') == cid else None
    except (OSError, ValueError):
        return None


def listing(project_path: str = '', project: str = '', everything=False) -> list:
    """The conversations, newest first (without their transcripts): this project's, or all of them."""
    out = []
    try:
        names = os.listdir(chat_dir())
    except OSError:
        return out
    for n in names:
        if not n.endswith('.json'):
            continue
        rec = load(n[:-5])
        if rec is None:
            continue
        mine = (project_path and rec.get('project_path') == project_path) or \
               (not project_path and not rec.get('project_path') and rec.get('project') == project)
        if everything or mine:
            out.append({k: v for k, v in rec.items() if k != 'markdown'})
    return sorted(out, key=lambda r: -float(r.get('updated', 0)))


def delete(cid: str) -> bool:
    try:
        os.remove(_path(cid))
        return True
    except (OSError, ValueError):
        return False


def _prune():
    try:
        files = sorted((os.path.getmtime(os.path.join(chat_dir(), n)), n) for n in os.listdir(chat_dir())
                       if n.endswith('.json'))
    except OSError:
        return
    for _, n in files[:max(0, len(files) - MAX_CHATS)]:
        try:
            os.remove(os.path.join(chat_dir(), n))
        except OSError:
            pass
