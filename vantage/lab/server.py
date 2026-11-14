"""The lab server's REST API, on Python's standard library HTTP server (no dependency beyond Vantage itself: the server
reads published projects with core.project_dir to compare versions and pool fleets).

Run it:  vantage --lab-server ROOT [--host 127.0.0.1] [--port 8765]   (or python -m vantage.lab serve --root ROOT)
First:   python -m vantage.lab init --root ROOT --team my-team         prints the first admin token, once
Tokens:  python -m vantage.lab token --root ROOT --team other-team --role member --name alice
Backup:  python -m vantage.lab backup --root ROOT DEST                  (consistent while serving)

Auth: every request but /api/v1/health carries "Authorization: Bearer <token>". 401: no or a bad token; 403: the
token's team or role may not do this; 404: no such thing; 409: a conflict (a version published since, a name taken);
413: too large. Bodies and answers are JSON (blobs: raw bytes).

TLS: the server speaks plain HTTP and listens on 127.0.0.1 by default. To reach it from other machines put a reverse
proxy in front that terminates HTTPS, e.g. Caddy (automatic certificates):

    lab.example.org {
        reverse_proxy 127.0.0.1:8765
    }

or nginx (with a certificate): `location / { proxy_pass http://127.0.0.1:8765; client_max_body_size 1g;
proxy_request_buffering off; }` inside a `listen 443 ssl` server block. Tokens travel in a header: never expose the
plain HTTP port beyond the machine (--host 0.0.0.0 prints a warning).

Endpoints (all under /api/v1):
    GET    health                                   {service, api, version} (no token)
    GET    whoami                                   the token's team and role
    GET    projects                                 projects the team can read
    POST   projects                                 {name, readers?, share_fleet?, share_records?} register (member)
    GET    projects/TEAM/NAME                       one project, its head
    PATCH  projects/TEAM/NAME                       {readers?, share_fleet?, share_records?, name?} (its team)
    GET    projects/TEAM/NAME/versions              the history
    POST   projects/TEAM/NAME/versions              {manifest: {path: sha256}, message, parent, content_sha256}
    GET    projects/TEAM/NAME/versions/REF          a version and its manifest (REF: number, head, digest)
    GET    projects/TEAM/NAME/compare?a=REF&b=REF   the semantic diff (core.project_dir.diff) and its text
    POST   blobs/missing                            {sha256: [...]} -> {missing: [...]}
    PUT    blobs/SHA256                             upload one file (member); its SHA-256 must be its name
    GET    blobs/SHA256                             a file of a version the team can read
    GET    fleet/pool?projects=A,B                  pooled fleet inputs (core.fleet_twin.pool_source) of the
                                                    projects the team can read whose fleets are shared
    POST   records                                  {project, records: [...]} anonymised perception records
    GET    records?after=SEQ&limit=N                every record of the lab, oldest first
    GET    teams, POST teams {name}, GET tokens, POST tokens {team, role, name}, DELETE tokens/ID, GET stats (admin)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlparse

from . import API_VERSION, DEFAULT_PORT, ROLES, SERVICE
from .store import LabError, LabStore, Principal

MAX_JSON = 64 << 20
MAX_BLOB = 1 << 30
PROJECT = r'projects/(?P<pid>[a-z0-9][a-z0-9._-]{0,62}/[a-z0-9][a-z0-9._-]{0,62})'


def _version() -> str:
    try:
        from .. import __version__
        return __version__
    except Exception:                                 # noqa: BLE001
        return ''


class _ProjectCache:
    """Projects loaded from versions (by digest), for compare and the pooled fleet: the last few."""

    def __init__(self, n=8):
        self.n, self.d, self.lock = n, {}, threading.Lock()

    def get(self, store: LabStore, doc: dict):
        with self.lock:
            p = self.d.get(doc['digest'])
            if p is not None:
                return p
        from ..core import project_dir as PD
        p = PD.project_from_files(store.version_files(doc), what=f'{doc["project"]} version {doc["number"]}')
        with self.lock:
            self.d[doc['digest']] = p
            while len(self.d) > self.n:
                self.d.pop(next(iter(self.d)))
        return p


def validate_record(doc) -> list:
    """A perception record the lab accepts: valid (core.perf_model.validate_record) and anonymised."""
    from ..core import perf_model as PM
    errs = PM.validate_record(doc)
    if not errs and not doc.get('anonymised'):
        errs = ['not anonymised (core.perf_model.anonymise): the lab takes no names or paths']
    return errs


class LabServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, addr, store: LabStore, max_blob: int = MAX_BLOB, quiet: bool = False):
        super().__init__(addr, LabHandler)
        self.store = store
        self.max_blob = int(max_blob)
        self.quiet = quiet
        self.cache = _ProjectCache()

    @property
    def url(self) -> str:
        host, port = self.server_address[:2]
        return f'http://{host}:{port}'


class LabHandler(BaseHTTPRequestHandler):
    server: LabServer
    protocol_version = 'HTTP/1.1'
    server_version = f'{SERVICE}/{API_VERSION}'

    ROUTES = [
        ('GET', r'health', 'health', False),
        ('GET', r'whoami', 'whoami', True),
        ('GET', r'projects', 'list_projects', True),
        ('POST', r'projects', 'register', True),
        ('GET', PROJECT, 'get_project', True),
        ('PATCH', PROJECT, 'patch_project', True),
        ('GET', PROJECT + r'/versions', 'list_versions', True),
        ('POST', PROJECT + r'/versions', 'publish', True),
        ('GET', PROJECT + r'/versions/(?P<ref>[A-Za-z0-9]{1,64})', 'get_version', True),
        ('GET', PROJECT + r'/compare', 'compare', True),
        ('POST', r'blobs/missing', 'missing', True),
        ('PUT', r'blobs/(?P<sha>[0-9a-f]{64})', 'put_blob', True),
        ('GET', r'blobs/(?P<sha>[0-9a-f]{64})', 'get_blob', True),
        ('GET', r'fleet/pool', 'fleet_pool', True),
        ('POST', r'records', 'post_records', True),
        ('GET', r'records', 'get_records', True),
        ('GET', r'teams', 'list_teams', True),
        ('POST', r'teams', 'add_team', True),
        ('GET', r'tokens', 'list_tokens', True),
        ('POST', r'tokens', 'add_token', True),
        ('DELETE', r'tokens/(?P<tid>\d+)', 'revoke_token', True),
        ('GET', r'stats', 'stats', True),
    ]
    _COMPILED = [(m, re.compile(rf'^/api/v{API_VERSION}/{p}/?$'), h, a) for m, p, h, a in ROUTES]

    # ---------------------------------------------------------------- plumbing
    def log_message(self, fmt, *args):
        if not self.server.quiet:
            sys.stderr.write(f'{self.log_date_time_string()} {self.address_string()} {fmt % args}\n')

    def _send(self, status: int, body: bytes, ctype='application/json', headers=None):
        self.send_response(status)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != 'HEAD':
            self.wfile.write(body)

    def _json(self, status: int, doc):
        self._send(status, json.dumps(doc, ensure_ascii=False, allow_nan=False, default=str).encode('utf-8'))

    def _error(self, status: int, message: str, **extra):
        headers = {'WWW-Authenticate': 'Bearer realm="vantage-lab"'} if status == 401 else None
        body = json.dumps({'error': message, 'status': status, **extra}, default=str).encode('utf-8')
        self._send(status, body, headers=headers)

    def _body_json(self):
        n = self._length()
        if n > MAX_JSON:
            raise LabError(413, 'the request is too large')
        raw = self.rfile.read(n) if n else b''
        if not raw:
            return {}
        try:
            d = json.loads(raw.decode('utf-8'))
        except (ValueError, UnicodeDecodeError):
            raise LabError(400, 'the body is not JSON') from None
        if not isinstance(d, dict):
            raise LabError(400, 'the body is not a JSON object')
        return d

    def _length(self) -> int:
        if self.headers.get('Transfer-Encoding', '').lower() == 'chunked':
            raise LabError(411, 'send Content-Length (chunked uploads are not taken)')
        try:
            n = int(self.headers.get('Content-Length') or 0)
        except ValueError:
            raise LabError(400, 'bad Content-Length') from None
        if n < 0:
            raise LabError(400, 'bad Content-Length')
        return n

    def _who(self) -> Principal:
        auth = self.headers.get('Authorization', '')
        tok = auth[7:].strip() if auth.lower().startswith('bearer ') else None
        return self.server.store.authenticate(tok)

    def _dispatch(self):
        url = urlparse(self.path)
        path = unquote(url.path)
        self.query = {k: v[-1] for k, v in parse_qs(url.query).items()}
        allowed = []
        for m, rx, handler, auth in self._COMPILED:
            g = rx.match(path)
            if not g:
                continue
            if m != self.command:
                allowed.append(m)
                continue
            try:
                who = self._who() if auth else None
                return getattr(self, handler)(who, **g.groupdict())
            except LabError as e:
                if e.status in (400, 411, 413) and self.command in ('POST', 'PUT', 'PATCH'):
                    self.close_connection = True          # the body may not have been read
                return self._error(e.status, e.message, **e.extra)
            except (BrokenPipeError, ConnectionResetError):
                return None
            except Exception as e:                        # noqa: BLE001  never a traceback to the client
                if not self.server.quiet:
                    traceback.print_exc()
                return self._error(500, f'the server failed: {type(e).__name__}')
        if allowed:
            self.close_connection = True
            return self._error(405, f'{self.command} is not allowed here', allow=sorted(set(allowed)))
        self.close_connection = True
        return self._error(404, 'no such endpoint')

    do_GET = do_POST = do_PUT = do_PATCH = do_DELETE = _dispatch

    def _admin(self, who: Principal):
        if not who.admin:
            raise LabError(403, 'only a lab admin can do this')

    # ---------------------------------------------------------------- endpoints
    def health(self, who):
        self._json(200, {'service': SERVICE, 'api': API_VERSION, 'version': _version()})

    def whoami(self, who):
        self._json(200, who.to_dict())

    def list_projects(self, who):
        self._json(200, {'projects': self.server.store.projects(who)})

    def register(self, who):
        b = self._body_json()
        doc = self.server.store.register(who, str(b.get('name') or ''), b.get('readers') or (),
                                         bool(b.get('share_fleet')), bool(b.get('share_records')))
        self._json(201, doc)

    def get_project(self, who, pid):
        self._json(200, self.server.store.project(who, pid))

    def patch_project(self, who, pid):
        b = self._body_json()
        kw = {k: b[k] for k in ('readers', 'share_fleet', 'share_records', 'name') if k in b}
        self._json(200, self.server.store.patch(who, pid, **kw))

    def list_versions(self, who, pid):
        self._json(200, {'project': pid, 'versions': self.server.store.versions(who, pid)})

    def publish(self, who, pid):
        b = self._body_json()
        parent = b.get('parent', None)
        if parent is not None and (isinstance(parent, bool) or not isinstance(parent, int) or parent < 0):
            raise LabError(400, 'parent is a version number (0: none yet) or null')
        doc = self.server.store.publish(who, pid, b.get('manifest') or {}, str(b.get('message') or ''), parent,
                                        b.get('content_sha256') if isinstance(b.get('content_sha256'), str) else None)
        self._json(200 if doc.get('existing') else 201, doc)

    def get_version(self, who, pid, ref):
        self._json(200, self.server.store.version(who, pid, ref))

    def compare(self, who, pid):
        from ..core import project_dir as PD
        st = self.server.store
        a = st.version(who, pid, self.query.get('a') or 'head')
        b = st.version(who, pid, self.query.get('b') or 'head')
        pa, pb = self.server.cache.get(st, a), self.server.cache.get(st, b)
        doc = PD.diff(pa, pb, f'{pid}@{a["number"]}', f'{pid}@{b["number"]}')
        self._json(200, {'diff': doc, 'text': PD.diff_text(doc)})

    def missing(self, who):
        b = self._body_json()
        shas = b.get('sha256') or []
        if not isinstance(shas, list) or len(shas) > 20000:
            raise LabError(400, 'sha256 is a list (at most 20000)')
        self._json(200, {'missing': self.server.store.missing(shas)})

    def put_blob(self, who, sha):
        n = self._length()
        doc = self.server.store.put_blob(who, sha, self.rfile, n, self.server.max_blob)
        self._json(201, doc)

    def get_blob(self, who, sha):
        st = self.server.store
        if not st.has_blob(sha) or not st.blob_readable(who, sha):
            raise LabError(404, 'no such file in a project you can read')
        self._send(200, st.read_blob(sha), 'application/octet-stream', {'X-Content-SHA256': sha})

    def fleet_pool(self, who):
        from ..core import fleet_twin as FT
        st = self.server.store
        want = {x.strip() for x in (self.query.get('projects') or '').split(',') if x.strip()}
        sources, skipped = [], []
        for pdoc in st.projects(who):
            if want and pdoc['id'] not in want:
                continue
            if pdoc['team'] != who.team and not pdoc['share_fleet'] and not who.admin:
                skipped.append({'project': pdoc['id'], 'why': 'its team does not share its fleet'})
                continue
            if not pdoc['head']:
                continue
            v = st.version(who, pdoc['id'], 'head')
            p = self.server.cache.get(st, v)
            src = FT.pool_source(p, f'{pdoc["id"]}@{v["number"]}')
            if not src['vehicles'] and not src['observations']:
                continue
            src.update({'project': pdoc['id'], 'version': v['number'], 'digest': v['digest']})
            sources.append(src)
        for w in sorted(want - {s['project'] for s in sources} - {s['project'] for s in skipped}):
            skipped.append({'project': w, 'why': 'not readable, no version, or no fleet'})
        self._json(200, {'schema': FT.POOL_SCHEMA, 'sources': sources, 'skipped': skipped})

    def post_records(self, who):
        b = self._body_json()
        docs = b.get('records')
        if not isinstance(docs, list) or len(docs) > 10000:
            raise LabError(400, 'records is a list (at most 10000 per request)')
        self._json(200, self.server.store.add_records(who, str(b.get('project') or ''), docs, validate_record))

    def get_records(self, who):
        try:
            after, limit = int(self.query.get('after', 0)), int(self.query.get('limit', 500))
        except ValueError:
            raise LabError(400, 'after and limit are numbers') from None
        self._json(200, self.server.store.records(who, after, limit))

    def list_teams(self, who):
        self._admin(who)
        self._json(200, {'teams': self.server.store.teams()})

    def add_team(self, who):
        self._admin(who)
        self._json(201, {'team': self.server.store.add_team(str(self._body_json().get('name') or ''), who)})

    def list_tokens(self, who):
        self._admin(who)
        self._json(200, {'tokens': self.server.store.tokens()})

    def add_token(self, who):
        self._admin(who)
        b = self._body_json()
        self._json(201, self.server.store.add_token(str(b.get('team') or ''), str(b.get('role') or 'member'),
                                                    str(b.get('name') or ''), who))

    def revoke_token(self, who, tid):
        self._admin(who)
        if not self.server.store.revoke(int(tid), who):
            raise LabError(404, f'no token {tid}')
        self._json(200, {'revoked': int(tid)})

    def stats(self, who):
        self._admin(who)
        self._json(200, self.server.store.stats())


# =========================================================================== running it
def make_server(root: str, host: str = '127.0.0.1', port: int = DEFAULT_PORT, max_blob: int = MAX_BLOB,
                quiet: bool = False, create: bool = True) -> LabServer:
    """The server on (host, port) over the lab folder root (port 0: a free port the system picks)."""
    return LabServer((host, int(port)), LabStore(root, create=create), max_blob=max_blob, quiet=quiet)


def serve_in_thread(server: LabServer) -> threading.Thread:
    th = threading.Thread(target=server.serve_forever, kwargs={'poll_interval': 0.1}, daemon=True)
    th.start()
    return th


def main(argv=None) -> int:
    """python -m vantage.lab {serve,init,token,revoke,teams,tokens,backup,gc,stats} … ; `vantage --lab-server ROOT`
    is `serve --root ROOT`."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if '--lab-server' in argv:
        i = argv.index('--lab-server')
        rest = argv[i + 1:]
        root = rest.pop(0) if rest and not rest[0].startswith('-') else None
        argv = ['serve'] + (['--root', root] if root else []) + rest
    ap = argparse.ArgumentParser(prog='vantage-lab', description='The Vantage lab server')
    sub = ap.add_subparsers(dest='cmd', required=True)

    def with_root(name, help_):
        p = sub.add_parser(name, help=help_)
        p.add_argument('--root', required=True, help='the lab folder (made if missing)')
        return p
    s = with_root('serve', 'run the server')
    s.add_argument('--host', default='127.0.0.1')
    s.add_argument('--port', type=int, default=DEFAULT_PORT)
    s.add_argument('--max-blob-mb', type=float, default=MAX_BLOB / 2 ** 20)
    s.add_argument('--quiet', action='store_true')
    i = with_root('init', 'make the lab and its first admin token')
    i.add_argument('--team', required=True)
    t = with_root('token', 'a new token (printed once)')
    t.add_argument('--team', required=True)
    t.add_argument('--role', default='member', choices=ROLES)
    t.add_argument('--name', default='')
    r = with_root('revoke', 'revoke a token by its id')
    r.add_argument('id', type=int)
    with_root('teams', 'list the teams')
    with_root('tokens', 'list the tokens (never the tokens themselves)')
    with_root('stats', 'counts and bytes')
    with_root('gc', 'remove uploads no version refers to (older than a day)')
    b = with_root('backup', 'a consistent copy of the lab folder')
    b.add_argument('dest')
    a = ap.parse_args(argv)
    try:
        if a.cmd == 'serve':
            srv = make_server(a.root, a.host, a.port, int(a.max_blob_mb * 2 ** 20), a.quiet)
            if a.host not in ('127.0.0.1', 'localhost', '::1'):
                print('warning: the lab speaks plain HTTP: put a reverse proxy with HTTPS in front of it '
                      '(see vantage/lab/server.py)', file=sys.stderr)
            print(f'Vantage lab on {srv.url} (folder {os.path.abspath(a.root)}); Ctrl+C stops it', file=sys.stderr)
            try:
                srv.serve_forever()
            except KeyboardInterrupt:
                pass
            finally:
                srv.server_close()
            return 0
        st = LabStore(a.root, create=a.cmd == 'init')
        if a.cmd == 'init':
            tok = st.add_token(a.team, 'admin', 'first admin')
            print(json.dumps({'team': tok['team'], 'role': 'admin', 'token': tok['token'],
                              'note': 'shown once: keep it in a password manager'}, indent=1))
        elif a.cmd == 'token':
            print(json.dumps(st.add_token(a.team, a.role, a.name), indent=1))
        elif a.cmd == 'revoke':
            return 0 if st.revoke(a.id) else 1
        elif a.cmd == 'teams':
            print('\n'.join(st.teams()))
        elif a.cmd == 'tokens':
            print(json.dumps(st.tokens(), indent=1))
        elif a.cmd == 'stats':
            print(json.dumps(st.stats(), indent=1))
        elif a.cmd == 'gc':
            print(json.dumps(st.gc(), indent=1))
        elif a.cmd == 'backup':
            print(json.dumps(st.backup(a.dest), indent=1))
        return 0
    except LabError as e:
        print(f'vantage lab: {e.message}', file=sys.stderr)
        return 2
    except OSError as e:
        print(f'vantage lab: {e}', file=sys.stderr)
        return 2
