"""The lab client: publish, pull, sync and compare projects on a lab server, the pooled fleet twin, and the pooled
perception records, for the app (the Lab page) and batch mode. Standard library HTTP (urllib), HTTPS through the
system's certificate store (or a given CA file).

What is sent: publish() sends exactly core.project_dir.publishable(files_of(project)) (the project folder's files
without the per-user view file), and nothing else; files_to_publish() returns that set so the page can show it before
anything leaves. Stream passwords are not in project files (core.credentials keeps them in the user's settings),
API keys never are (assistant.secrets), and the lab token is kept in the system keyring or, without one, a private
file in the settings folder (lab-tokens.json, readable by the user alone); tests/test_lab_server.py checks that none
of them reaches the server. Perception records leave anonymised (core.perf_model.anonymise) and only for a project
whose owner switched share_records on.

Sync state (which lab project and version a local copy was last published or pulled as) is kept per user in the
settings folder (lab-sync.json), keyed by the project's file or folder, never in the project."""
from __future__ import annotations

import hashlib
import json
import os
import ssl
import urllib.error
import urllib.parse
import urllib.request

from . import API_VERSION
from .store import tree_digest

TOKENS_FILE = 'lab-tokens.json'
SYNC_FILE = 'lab-sync.json'
KEYRING_SERVICE = 'Vantage lab'
TIMEOUT = 120.0


class LabClientError(Exception):
    """A refused or failed request: status (HTTP, or 0 when the server could not be reached), message, extra."""

    def __init__(self, status: int, message: str, extra: dict | None = None):
        super().__init__(f'{message} (HTTP {status})' if status else message)
        self.status, self.message, self.extra = int(status), message, dict(extra or {})


# =========================================================================== the token, kept like the API keys
def _config_dir() -> str:
    from ..assistant.secrets import config_dir
    return config_dir()


def _private_json(name: str) -> dict:
    try:
        with open(os.path.join(_config_dir(), name), encoding='utf-8') as fh:
            d = json.load(fh)
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def _write_private_json(name: str, d: dict):
    os.makedirs(_config_dir(), exist_ok=True)
    path = os.path.join(_config_dir(), name)
    tmp = path + '.tmp'
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, 'w', encoding='utf-8') as fh:
        json.dump(d, fh, indent=1, sort_keys=True)
    os.replace(tmp, path)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def norm_url(url: str) -> str:
    u = str(url or '').strip().rstrip('/')
    if u and '://' not in u:
        u = 'http://' + u
    return u


def save_token(url: str, token: str) -> str:
    """Keep the token for this server: the system keyring when there is one, else a private file. Returns where."""
    from ..assistant.secrets import _keyring
    url, token = norm_url(url), token.strip()
    kr = _keyring()
    if kr is not None:
        try:
            kr.set_password(KEYRING_SERVICE, url, token)
            d = _private_json(TOKENS_FILE)
            if d.pop(url, None) is not None:
                _write_private_json(TOKENS_FILE, d)
            return 'keyring'
        except Exception:                               # noqa: BLE001  a keyring that refuses: the file
            pass
    d = _private_json(TOKENS_FILE)
    d[url] = token
    _write_private_json(TOKENS_FILE, d)
    return 'file'


def load_token(url: str) -> str | None:
    from ..assistant.secrets import _keyring
    url = norm_url(url)
    env = os.environ.get('VANTAGE_LAB_TOKEN', '').strip()
    if env:
        return env
    kr = _keyring()
    if kr is not None:
        try:
            t = kr.get_password(KEYRING_SERVICE, url)
            if t:
                return t
        except Exception:                               # noqa: BLE001
            pass
    return _private_json(TOKENS_FILE).get(url) or None


def forget_token(url: str):
    from ..assistant.secrets import _keyring
    url = norm_url(url)
    kr = _keyring()
    if kr is not None:
        try:
            kr.delete_password(KEYRING_SERVICE, url)
        except Exception:                               # noqa: BLE001
            pass
    d = _private_json(TOKENS_FILE)
    if d.pop(url, None) is not None:
        _write_private_json(TOKENS_FILE, d)


# =========================================================================== sync state (per user, per local copy)
def _key(path_or_project) -> str:
    p = getattr(path_or_project, 'path', path_or_project) or getattr(path_or_project, 'name', '')
    p = str(p)
    return os.path.realpath(p) if os.path.exists(p) else p


def sync_state(path_or_project) -> dict | None:
    """{'url', 'project', 'version', 'digest'} the local copy was last published or pulled as, or None."""
    d = _private_json(SYNC_FILE).get(_key(path_or_project))
    return d if isinstance(d, dict) else None


def set_sync_state(path_or_project, url: str, pid: str, version: int, digest: str):
    d = _private_json(SYNC_FILE)
    d[_key(path_or_project)] = {'url': norm_url(url), 'project': pid, 'version': int(version), 'digest': digest}
    _write_private_json(SYNC_FILE, d)


# =========================================================================== what is sent
def files_to_publish(project, thumb_png: bytes | None = None) -> dict:
    """{path: bytes}: exactly what publish() sends for this project."""
    from ..core import project_dir as PD
    return PD.publishable(PD.files_of(project, thumb_png, text_mesh=_text_mesh_of(project)))


def _text_mesh_of(project) -> bool:
    path = getattr(project, 'path', '') or ''
    return bool(path) and os.path.isfile(os.path.join(path, 'mesh.obj')) and \
        not os.path.isfile(os.path.join(path, 'mesh.npz'))


def manifest_of(files: dict) -> dict:
    return {p: hashlib.sha256(b).hexdigest() for p, b in files.items()}


def digest_of(files: dict) -> str:
    return tree_digest(manifest_of(files))


VOLATILE = ('thumb.png',)          # re-rendered on every save: never a reason to sync, never a conflict


def core_digest(manifest: dict) -> str:
    """The tree digest of a manifest ({path: sha or [sha, size]}) without the thumbnail: what sync compares."""
    return tree_digest({p: v for p, v in manifest.items() if p not in VOLATILE})


# =========================================================================== the client
class LabClient:
    def __init__(self, url: str, token: str | None = None, timeout: float = TIMEOUT, cafile: str | None = None):
        self.url = norm_url(url)
        if not self.url:
            raise LabClientError(0, 'no lab server address')
        self.token = token if token is not None else load_token(self.url)
        self.timeout = timeout
        self._ctx = ssl.create_default_context(cafile=cafile) if self.url.startswith('https') else None

    # ---------------------------------------------------------------- HTTP
    def request(self, method: str, path: str, body=None, query: dict | None = None, raw: bool = False,
                auth: bool = True):
        url = f'{self.url}/api/v{API_VERSION}/{path.lstrip("/")}'
        if query:
            url += '?' + urllib.parse.urlencode({k: v for k, v in query.items() if v is not None})
        headers = {'Accept': 'application/json', 'User-Agent': 'vantage-lab-client'}
        data = None
        if isinstance(body, (bytes, bytearray)):
            data = bytes(body)
            headers['Content-Type'] = 'application/octet-stream'
        elif body is not None:
            data = json.dumps(body, ensure_ascii=False, allow_nan=False).encode('utf-8')
            headers['Content-Type'] = 'application/json'
        if auth:
            if not self.token:
                raise LabClientError(401, 'no token for this lab: enter the token your lab admin gave you')
            headers['Authorization'] = f'Bearer {self.token}'
        req = urllib.request.Request(url, data=data, method=method, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout, context=self._ctx) as r:
                payload = r.read()
        except urllib.error.HTTPError as e:
            try:
                doc = json.loads(e.read().decode('utf-8'))
            except Exception:                           # noqa: BLE001
                doc = {}
            msg = doc.pop('error', None) or e.reason or 'refused'
            doc.pop('status', None)
            raise LabClientError(e.code, str(msg), doc) from None
        except (urllib.error.URLError, OSError) as e:
            raise LabClientError(0, f'cannot reach the lab at {self.url}: {getattr(e, "reason", e)}') from None
        if raw:
            return payload
        return json.loads(payload.decode('utf-8')) if payload else {}

    # ---------------------------------------------------------------- simple calls
    def health(self) -> dict:
        return self.request('GET', 'health', auth=False)

    def whoami(self) -> dict:
        return self.request('GET', 'whoami')

    def projects(self) -> list:
        return self.request('GET', 'projects')['projects']

    def register(self, name: str, readers=(), share_fleet=False, share_records=False) -> dict:
        return self.request('POST', 'projects', {'name': name, 'readers': list(readers), 'share_fleet': share_fleet,
                                                 'share_records': share_records})

    def project(self, pid: str) -> dict:
        return self.request('GET', f'projects/{pid}')

    def patch(self, pid: str, **kw) -> dict:
        return self.request('PATCH', f'projects/{pid}', kw)

    def versions(self, pid: str) -> list:
        return self.request('GET', f'projects/{pid}/versions')['versions']

    def version(self, pid: str, ref='head') -> dict:
        return self.request('GET', f'projects/{pid}/versions/{ref}')

    def compare(self, pid: str, a, b) -> dict:
        """{'diff': vantage.project_diff/1, 'text'} of version b against version a, computed by the server."""
        return self.request('GET', f'projects/{pid}/compare', query={'a': a, 'b': b})

    # ---------------------------------------------------------------- publish and pull
    def upload(self, files: dict, progress=None) -> dict:
        """Upload the files the lab does not have yet: {'uploaded', 'bytes', 'skipped'}."""
        man = manifest_of(files)
        missing = set(self.request('POST', 'blobs/missing', {'sha256': sorted(set(man.values()))})['missing'])
        by_sha = {}
        for p, s in man.items():
            by_sha.setdefault(s, p)
        sent = nbytes = 0
        todo = [s for s in sorted(missing) if s in by_sha]
        for i, s in enumerate(todo):
            b = files[by_sha[s]]
            self.request('PUT', f'blobs/{s}', b)
            sent += 1
            nbytes += len(b)
            if progress is not None:
                progress((i + 1) / max(len(todo), 1), by_sha[s])
        return {'uploaded': sent, 'bytes': nbytes, 'skipped': len(set(man.values())) - sent}

    def publish_files(self, pid: str, files: dict, message: str = '', parent='auto', content_sha256=None,
                      progress=None) -> dict:
        """Publish {path: bytes} as a version of pid. parent: the version it builds on ('auto': the head now, so it
        always lands; an int: refused with 409 when the project moved on; None: no check)."""
        up = self.upload(files, progress)
        if parent == 'auto':
            pr = self.project(pid)
            parent = pr['head']['number'] if pr.get('head') else 0
        doc = self.request('POST', f'projects/{pid}/versions', {'manifest': manifest_of(files), 'message': message,
                                                               'parent': parent, 'content_sha256': content_sha256})
        doc['upload'] = up
        return doc

    def publish(self, project, pid: str, message: str = '', parent='auto', thumb_png: bytes | None = None,
                progress=None, remember: bool = True) -> dict:
        """Publish a Project (files_to_publish(project): nothing else is sent)."""
        from ..core.provenance import project_digest
        files = files_to_publish(project, thumb_png)
        doc = self.publish_files(pid, files, message, parent, project_digest(project)['sha256'], progress)
        if remember and getattr(project, 'path', ''):
            set_sync_state(project, self.url, pid, doc['number'], core_digest(manifest_of(files)))
        return doc

    def files_at(self, pid: str, ref='head') -> tuple:
        """(version doc, {path: bytes}) with every file checked against its SHA-256."""
        v = self.version(pid, ref)
        out = {}
        for path, (sha, _) in v['manifest'].items():
            b = self.request('GET', f'blobs/{sha}', raw=True)
            if hashlib.sha256(b).hexdigest() != sha:
                raise LabClientError(0, f'{path} arrived damaged (its SHA-256 differs): try again')
            out[path] = b
        return v, out

    def pull(self, pid: str, ref='head', dest: str | None = None):
        """A version as a Project; with dest, saved there too (a folder path or name.vantage.d: the project folder;
        else a .vantage file) and remembered as synced."""
        from ..core import project_dir as PD
        v, files = self.files_at(pid, ref)
        p = PD.project_from_files(files, what=f'{pid} version {v["number"]}')
        if dest:                                        # the lab's name stays: it is part of the content
            if not PD.wants_dir(dest) and not dest.lower().endswith('.vantage'):
                dest += '.vantage'
            p.save(dest)
            set_sync_state(p, self.url, pid, v['number'], core_digest(v['manifest']))
        p.lab_version = v
        return p

    def sync(self, project, pid: str, message: str = '', thumb_png: bytes | None = None) -> dict:
        """Bring a local copy and the lab together. {'status': 'up to date' | 'published' | 'pulled' | 'merged' |
        'conflicts', 'project' (the project to use: a pulled or merged one, else the one given), 'version',
        'conflicts': [...]}. Both changed since the last sync: a three-way merge (core.project_dir.merge_bytes) of
        the last synced version, the local copy and the head, published when clean."""
        from ..core import project_dir as PD
        st = sync_state(project)
        if st and (st.get('url') != self.url or st.get('project') != pid):
            st = None
        local = files_to_publish(project, thumb_png)
        ld = core_digest(manifest_of(local))
        try:
            head = self.version(pid, 'head')
        except LabClientError as e:
            if e.status != 404:
                raise
            head = None
        if head and core_digest(head['manifest']) == ld:
            if getattr(project, 'path', ''):
                set_sync_state(project, self.url, pid, head['number'], ld)
            return {'status': 'up to date', 'project': project, 'version': head['number'], 'conflicts': []}
        if not head:
            doc = self.publish(project, pid, message or 'first version', parent=0, thumb_png=thumb_png)
            return {'status': 'published', 'project': project, 'version': doc['number'], 'conflicts': []}
        base_n = st['version'] if st else None
        local_changed = st is None or ld != st.get('digest')
        remote_changed = base_n is None or head['number'] != base_n
        if not remote_changed:
            doc = self.publish(project, pid, message, parent=base_n, thumb_png=thumb_png)
            return {'status': 'published', 'project': project, 'version': doc['number'], 'conflicts': []}
        hv, theirs = self.files_at(pid, 'head')
        if not local_changed:
            p = PD.project_from_files(theirs, path=getattr(project, 'path', ''))
            if getattr(project, 'path', ''):
                set_sync_state(project, self.url, pid, hv['number'], core_digest(hv['manifest']))
            return {'status': 'pulled', 'project': p, 'version': hv['number'], 'conflicts': []}
        if base_n is None:
            raise LabClientError(409, f'this copy was never synced with {pid} and differs from it: pull the lab '
                                      'version into a new folder, or publish as a new project')
        _, base = self.files_at(pid, base_n)
        merged, conflicts = merge_files(base, local, theirs)
        p = PD.project_from_files(merged, path=getattr(project, 'path', ''))
        if conflicts:
            return {'status': 'conflicts', 'project': p, 'version': hv['number'], 'conflicts': conflicts}
        doc = self.publish(p, pid, message or f'merged with version {hv["number"]}', parent=hv['number'],
                           thumb_png=thumb_png)
        return {'status': 'merged', 'project': p, 'version': doc['number'], 'conflicts': []}

    # ---------------------------------------------------------------- fleets and records
    def pool(self, projects=None) -> dict:
        """{'sources': [core.fleet_twin.pool_source docs], 'skipped': [...]} the team may pool."""
        q = {'projects': ','.join(projects)} if projects else None
        return self.request('GET', 'fleet/pool', query=q)

    def pooled_twin(self, base_project, projects=None, settings=None, engine=None, progress=None, cancel=None):
        """The fleet twin of the lab's shared fleets (and this project's own), on base_project's plan: (result,
        pool doc)."""
        from ..core import fleet_twin as FT
        doc = self.pool(projects)
        res = FT.run_pooled(base_project, doc['sources'], engine=engine, settings=settings, progress=progress,
                            cancel=cancel)
        return res, doc

    def push_records(self, pid: str, docs) -> dict:
        """Send perception records for pid, anonymised here first (the lab refuses any that are not)."""
        from ..core import perf_model as PM
        out = [PM.anonymise(d) for d in docs]
        total = {'added': 0, 'duplicates': 0, 'rejected': 0, 'errors': []}
        for i in range(0, len(out), 500):
            r = self.request('POST', 'records', {'project': pid, 'records': out[i:i + 500]})
            for k in ('added', 'duplicates', 'rejected'):
                total[k] += r.get(k, 0)
            total['errors'] += r.get('errors', [])
        return total

    def pull_records(self, store=None, after: int = 0) -> dict:
        """Every record of the lab into the local store (core.perf_model): {'received', 'added', 'duplicates',
        'rejected', 'next'}."""
        from ..core import perf_model as PM
        st = PM._as_store(store)
        got = added = dup = rej = 0
        while True:
            r = self.request('GET', 'records', query={'after': after, 'limit': 1000})
            docs = r['records']
            got += len(docs)
            res = st.add_many(docs)
            added += res['added']
            dup += res['duplicates']
            rej += res['rejected']
            after = r['next']
            if not r.get('more'):
                break
        return {'received': got, 'added': added, 'duplicates': dup, 'rejected': rej, 'next': after}


def project_records(project, store=None) -> list:
    """The perception records of this project in the local store: the project's own store (when it keeps one),
    else the user's store's records whose rig is one of the project's variants or whose provenance names it."""
    from ..core import perf_model as PM
    own = PM.project_store_path(project)
    if own and store is None:
        return PM.records(store=PM.store_for(project))
    hashes = set()
    for v in project.variants:
        try:
            hashes.add(PM.rig_hash(PM.rig_block(project, v)))
        except Exception:                                # noqa: BLE001  a variant the block cannot describe
            continue

    def mine(d):
        pv = d.get('provenance') if isinstance(d.get('provenance'), dict) else {}
        pj = pv.get('project') if isinstance(pv.get('project'), dict) else {}
        return d.get('rig_hash') in hashes or (pj.get('name') and pj.get('name') == project.name)
    return PM.records(mine, store)


def merge_files(base: dict, ours: dict, theirs: dict) -> tuple:
    """Three project folders' files ({path: bytes}) merged file by file (core.project_dir.merge_bytes; a binary
    file from the side that changed it): (merged files, conflicts [{'file', 'path', ...}])."""
    from ..core import project_dir as PD
    out, conflicts = {}, []
    for f in sorted(set(base) | set(ours) | set(theirs)):
        b, o, t = base.get(f), ours.get(f), theirs.get(f)
        if o == t:
            if o is not None:
                out[f] = o
            continue
        if o == b:
            if t is not None:
                out[f] = t
            continue
        if t == b:
            if o is not None:
                out[f] = o
            continue
        if f in VOLATILE:
            if o is not None or t is not None:
                out[f] = o if o is not None else t
            continue
        if o is None or t is None:                      # deleted on one side, changed on the other
            conflicts.append({'file': f, 'path': '.', 'ours': 'deleted' if o is None else 'changed',
                              'theirs': 'deleted' if t is None else 'changed'})
            if o is not None:
                out[f] = o
            continue
        if f.endswith(('.json', '.jsonl')):
            m, c = PD.merge_bytes(b or b'', o, t, f)
            out[f] = m
            conflicts += [dict(x, file=f) for x in c]
        else:
            out[f] = o
            conflicts.append({'file': f, 'path': '.', 'ours': 'binary', 'theirs': 'binary'})
    return out, conflicts
