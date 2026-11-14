"""The lab's storage: files on disk plus an SQLite index, in one folder (a backup is a copy of the folder, made safely
with backup()).

    ROOT/lab.sqlite3            teams, tokens (SHA-256 of each token, never the token), projects, versions, records,
                                the audit trail
    ROOT/objects/ab/cdef…       every file of every published version, once, named by its SHA-256 (content-addressed:
                                the same mesh in a hundred versions is stored once)
    ROOT/tmp/                   uploads in progress

A version is immutable: its manifest ({path: [sha256, bytes]}) and its digest (SHA-256 of the lines
"<sha256>  <path>" sorted by path, as sha256sum prints them) never change; the same files published again are the same
version. Versions of a project are numbered 1, 2, … in the order they were accepted; publishing names the version it
was based on (parent) and is refused (409) when the project has moved on since, so two people publishing at once
never silently overwrite each other: the second pulls, merges and publishes again (vantage.lab.client.sync).

Access: a token belongs to a team and has a role: admin (the lab: teams and tokens, and everything), member (register
projects in its team, publish to them, change their sharing, send perception records) or reader. A project is read by
its own team, the teams in its readers list ('*': every team) and admins; written by its team's members and admins.
Its fleet joins another team's pooled fleet twin only when share_fleet is on; its perception records are accepted
only when share_records is on (opt-in per project). Perception records are anonymised (core.perf_model.anonymise) and,
once accepted, every member of the lab can read them (the point: everyone's runs train the model)."""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import re
import secrets
import shutil
import sqlite3

from . import ROLES

SCHEMA_VERSION = 1
DB = 'lab.sqlite3'
SHA_RE = re.compile(r'^[0-9a-f]{64}$')
NAME_RE = re.compile(r'^[a-z0-9][a-z0-9._-]{0,62}$')
TOKEN_PREFIX = 'vlab_'
MAX_PATH = 200
MAX_FILES = 5000

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS teams(name TEXT PRIMARY KEY, created TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS tokens(id INTEGER PRIMARY KEY AUTOINCREMENT, hash TEXT UNIQUE NOT NULL,
    team TEXT NOT NULL REFERENCES teams(name), role TEXT NOT NULL, name TEXT NOT NULL DEFAULT '',
    created TEXT NOT NULL, revoked INTEGER NOT NULL DEFAULT 0);
CREATE TABLE IF NOT EXISTS projects(id TEXT PRIMARY KEY, team TEXT NOT NULL REFERENCES teams(name),
    name TEXT NOT NULL, created TEXT NOT NULL, readers TEXT NOT NULL DEFAULT '[]',
    share_fleet INTEGER NOT NULL DEFAULT 0, share_records INTEGER NOT NULL DEFAULT 0);
CREATE TABLE IF NOT EXISTS versions(project TEXT NOT NULL REFERENCES projects(id), number INTEGER NOT NULL,
    digest TEXT NOT NULL, parent INTEGER, message TEXT NOT NULL DEFAULT '', author TEXT NOT NULL DEFAULT '',
    created TEXT NOT NULL, content_sha256 TEXT, manifest TEXT NOT NULL, bytes INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY(project, number), UNIQUE(project, digest));
CREATE TABLE IF NOT EXISTS blobs(sha TEXT PRIMARY KEY, size INTEGER NOT NULL, created TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS version_blobs(project TEXT NOT NULL, number INTEGER NOT NULL, sha TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS version_blobs_sha ON version_blobs(sha);
CREATE TABLE IF NOT EXISTS records(seq INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT UNIQUE NOT NULL,
    origin_id TEXT, project TEXT NOT NULL, team TEXT NOT NULL, created TEXT NOT NULL, doc TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS audit(seq INTEGER PRIMARY KEY AUTOINCREMENT, time TEXT NOT NULL, team TEXT,
    token INTEGER, action TEXT NOT NULL, target TEXT NOT NULL DEFAULT '');
"""


class LabError(Exception):
    """A request the lab refuses: status is the HTTP status (400, 401, 403, 404, 409, 413), message why."""

    def __init__(self, status: int, message: str, **extra):
        super().__init__(message)
        self.status = int(status)
        self.message = message
        self.extra = extra


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z')


def token_hash(token: str) -> str:
    return hashlib.sha256(token.encode('utf-8')).hexdigest()


def new_token() -> str:
    """A bearer token: 256 bits from the operating system's secure generator (secrets), never stored in clear."""
    return TOKEN_PREFIX + secrets.token_urlsafe(32)


def tree_digest(manifest: dict) -> str:
    """The digest of a version: SHA-256 of '<sha256>  <path>\\n' lines sorted by path ({path: sha or [sha, size]})."""
    lines = []
    for path in sorted(manifest):
        v = manifest[path]
        sha = v[0] if isinstance(v, (list, tuple)) else v
        lines.append(f'{sha}  {path}\n')
    return hashlib.sha256(''.join(lines).encode('utf-8')).hexdigest()


def check_path(path: str) -> str:
    """A manifest path: relative, '/'-separated, no '..', no absolute or drive paths, printable."""
    p = str(path)
    if (not p or len(p) > MAX_PATH or p.startswith(('/', '\\')) or '\\' in p or ':' in p
            or any(part in ('', '.', '..') for part in p.split('/')) or any(ord(ch) < 32 for ch in p)):
        raise LabError(400, f'bad file path in the manifest: {p[:80]!r}')
    return p


def slug(name: str) -> str:
    s = re.sub(r'[^a-z0-9._-]+', '-', str(name).lower()).strip('-._')[:63]
    return s


class Principal:
    """Who a request is from: the token's team, role, id and name."""

    def __init__(self, team: str, role: str, token_id: int = 0, name: str = ''):
        self.team, self.role, self.token_id, self.name = team, role, token_id, name

    @property
    def admin(self) -> bool:
        return self.role == 'admin'

    def to_dict(self):
        return {'team': self.team, 'role': self.role, 'token': self.token_id, 'name': self.name}


class LabStore:
    """The storage folder. Every method opens its own SQLite connection, so threads can share one LabStore."""

    def __init__(self, root: str, create: bool = False):
        self.root = os.path.abspath(root)
        if create:
            os.makedirs(self.root, exist_ok=True)
        if not os.path.isdir(self.root):
            raise LabError(500, f'{self.root} is not a lab folder')
        os.makedirs(self.objects, exist_ok=True)
        os.makedirs(self.tmp, exist_ok=True)
        with self.db() as c:
            c.execute('PRAGMA journal_mode=WAL')
            c.executescript(SCHEMA_SQL)
            c.execute("INSERT OR IGNORE INTO meta VALUES('schema', ?)", (str(SCHEMA_VERSION),))

    # ---------------------------------------------------------------- plumbing
    @property
    def objects(self) -> str:
        return os.path.join(self.root, 'objects')

    @property
    def tmp(self) -> str:
        return os.path.join(self.root, 'tmp')

    def db(self) -> sqlite3.Connection:
        c = sqlite3.connect(os.path.join(self.root, DB), timeout=60, isolation_level=None)
        c.row_factory = sqlite3.Row
        c.execute('PRAGMA foreign_keys=ON')
        c.execute('PRAGMA busy_timeout=60000')
        return _Conn(c)

    def audit(self, c, who: Principal | None, action: str, target: str = ''):
        c.execute('INSERT INTO audit(time, team, token, action, target) VALUES(?,?,?,?,?)',
                  (utc_now(), who.team if who else None, who.token_id if who else None, action, target[:300]))

    # ---------------------------------------------------------------- teams and tokens
    def add_team(self, name: str, who: Principal | None = None) -> str:
        n = slug(name)
        if not NAME_RE.match(n):
            raise LabError(400, 'a team name needs letters or digits (a-z, 0-9, . _ -)')
        with self.db() as c:
            c.execute('INSERT OR IGNORE INTO teams VALUES(?, ?)', (n, utc_now()))
            self.audit(c, who, 'team.add', n)
        return n

    def teams(self) -> list:
        with self.db() as c:
            return [r['name'] for r in c.execute('SELECT name FROM teams ORDER BY name')]

    def add_token(self, team: str, role: str = 'member', name: str = '', who: Principal | None = None) -> dict:
        """A new token for team (made if missing) with role; {'token' (shown once), 'id', 'team', 'role'}."""
        if role not in ROLES:
            raise LabError(400, f'role is one of {", ".join(ROLES)}')
        team = self.add_team(team, who)
        tok = new_token()
        with self.db() as c:
            cur = c.execute('INSERT INTO tokens(hash, team, role, name, created) VALUES(?,?,?,?,?)',
                            (token_hash(tok), team, role, str(name)[:80], utc_now()))
            self.audit(c, who, 'token.add', f'{team}:{role}:{cur.lastrowid}')
            return {'token': tok, 'id': cur.lastrowid, 'team': team, 'role': role, 'name': str(name)[:80]}

    def tokens(self) -> list:
        with self.db() as c:
            return [dict(r) for r in c.execute('SELECT id, team, role, name, created, revoked FROM tokens ORDER BY id')]

    def revoke(self, token_id: int, who: Principal | None = None) -> bool:
        with self.db() as c:
            n = c.execute('UPDATE tokens SET revoked=1 WHERE id=?', (int(token_id),)).rowcount
            self.audit(c, who, 'token.revoke', str(token_id))
        return n > 0

    def authenticate(self, token: str | None) -> Principal:
        """The principal of a bearer token; 401 when it is missing, unknown or revoked."""
        if not token:
            raise LabError(401, 'no token: send "Authorization: Bearer <token>"')
        with self.db() as c:
            r = c.execute('SELECT id, team, role, name, revoked FROM tokens WHERE hash=?',
                          (token_hash(token.strip()),)).fetchone()
        if r is None or r['revoked']:
            raise LabError(401, 'the token is not valid (unknown or revoked)')
        return Principal(r['team'], r['role'], r['id'], r['name'])

    # ---------------------------------------------------------------- projects
    def _project_row(self, c, pid: str):
        r = c.execute('SELECT * FROM projects WHERE id=?', (pid,)).fetchone()
        if r is None:
            raise LabError(404, f'no project {pid}')
        return r

    def can_read(self, who: Principal, row) -> bool:
        readers = json.loads(row['readers'] or '[]')
        return who.admin or row['team'] == who.team or '*' in readers or who.team in readers

    def can_write(self, who: Principal, row) -> bool:
        return who.admin or (row['team'] == who.team and who.role == 'member')

    def project_doc(self, c, row) -> dict:
        head = c.execute('SELECT number, digest, created FROM versions WHERE project=? ORDER BY number DESC LIMIT 1',
                         (row['id'],)).fetchone()
        n = c.execute('SELECT count(*) FROM versions WHERE project=?', (row['id'],)).fetchone()[0]
        return {'id': row['id'], 'team': row['team'], 'name': row['name'], 'created': row['created'],
                'readers': json.loads(row['readers'] or '[]'), 'share_fleet': bool(row['share_fleet']),
                'share_records': bool(row['share_records']), 'versions': n,
                'head': dict(head) if head else None}

    def register(self, who: Principal, name: str, readers=(), share_fleet=False, share_records=False) -> dict:
        if who.role not in ('member', 'admin'):
            raise LabError(403, 'readers cannot register projects')
        s = slug(name)
        if not NAME_RE.match(s):
            raise LabError(400, 'a project name needs letters or digits')
        pid = f'{who.team}/{s}'
        rd = self._readers(readers)
        with self.db() as c:
            c.execute('BEGIN IMMEDIATE')
            if c.execute('SELECT 1 FROM projects WHERE id=?', (pid,)).fetchone():
                c.execute('ROLLBACK')
                raise LabError(409, f'{pid} is already registered', project=pid)
            c.execute('INSERT INTO projects VALUES(?,?,?,?,?,?,?)',
                      (pid, who.team, str(name)[:120], utc_now(), json.dumps(rd), int(bool(share_fleet)),
                       int(bool(share_records))))
            self.audit(c, who, 'project.register', pid)
            c.execute('COMMIT')
            return self.project_doc(c, self._project_row(c, pid))

    def _readers(self, readers) -> list:
        if isinstance(readers, str):
            readers = [x.strip() for x in readers.split(',')]
        out = []
        for r in readers or ():
            r = str(r).strip()
            if r == '*':
                out.append('*')
            elif r:
                s = slug(r)
                if not NAME_RE.match(s):
                    raise LabError(400, f'bad team name in readers: {r[:40]!r}')
                out.append(s)
        return sorted(set(out))

    def patch(self, who: Principal, pid: str, **kw) -> dict:
        with self.db() as c:
            row = self._project_row(c, pid)
            if not self.can_write(who, row):
                raise LabError(403, f'{who.team} cannot change {pid}')
            if 'readers' in kw:
                c.execute('UPDATE projects SET readers=? WHERE id=?', (json.dumps(self._readers(kw['readers'])), pid))
            for k in ('share_fleet', 'share_records'):
                if k in kw:
                    c.execute(f'UPDATE projects SET {k}=? WHERE id=?', (int(bool(kw[k])), pid))
            if 'name' in kw and str(kw['name']).strip():
                c.execute('UPDATE projects SET name=? WHERE id=?', (str(kw['name'])[:120], pid))
            self.audit(c, who, 'project.patch', f'{pid} {sorted(kw)}')
            return self.project_doc(c, self._project_row(c, pid))

    def projects(self, who: Principal) -> list:
        with self.db() as c:
            return [self.project_doc(c, r) for r in c.execute('SELECT * FROM projects ORDER BY id')
                    if self.can_read(who, r)]

    def project(self, who: Principal, pid: str) -> dict:
        with self.db() as c:
            row = self._project_row(c, pid)
            if not self.can_read(who, row):
                raise LabError(403, f'{who.team} cannot read {pid}')
            return self.project_doc(c, row)

    # ---------------------------------------------------------------- blobs
    def blob_path(self, sha: str) -> str:
        return os.path.join(self.objects, sha[:2], sha[2:])

    def has_blob(self, sha: str) -> bool:
        return os.path.isfile(self.blob_path(sha))

    def missing(self, shas) -> list:
        out = []
        for s in shas:
            s = str(s)
            if not SHA_RE.match(s):
                raise LabError(400, f'not a SHA-256: {s[:70]!r}')
            if not self.has_blob(s):
                out.append(s)
        return sorted(set(out))

    def put_blob(self, who: Principal, sha: str, stream, length: int, max_bytes: int) -> dict:
        """Store an upload (read from stream, length bytes) when its SHA-256 is sha; idempotent."""
        if who.role not in ('member', 'admin'):
            raise LabError(403, 'readers cannot upload')
        if not SHA_RE.match(sha):
            raise LabError(400, 'the blob name is not a SHA-256')
        if length > max_bytes:
            raise LabError(413, f'the file is larger than this lab accepts ({max_bytes} bytes)')
        h = hashlib.sha256()
        fd, tmp = _mkstemp(self.tmp)
        try:
            left = length
            with os.fdopen(fd, 'wb') as fh:
                while left > 0:
                    chunk = stream.read(min(left, 1 << 20))
                    if not chunk:
                        raise LabError(400, 'the upload ended early')
                    h.update(chunk)
                    fh.write(chunk)
                    left -= len(chunk)
            if h.hexdigest() != sha:
                raise LabError(400, 'the upload\'s SHA-256 is not its name: refused')
            dst = self.blob_path(sha)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if not os.path.exists(dst):
                os.replace(tmp, dst)
            with self.db() as c:
                c.execute('INSERT OR IGNORE INTO blobs VALUES(?,?,?)', (sha, length, utc_now()))
            return {'sha256': sha, 'bytes': length}
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    def blob_readable(self, who: Principal, sha: str) -> bool:
        """A blob is read through a version of a project the caller can read."""
        with self.db() as c:
            rows = c.execute('SELECT DISTINCT p.* FROM version_blobs vb JOIN projects p ON p.id = vb.project '
                             'WHERE vb.sha=?', (sha,)).fetchall()
            return any(self.can_read(who, r) for r in rows)

    def read_blob(self, sha: str) -> bytes:
        with open(self.blob_path(sha), 'rb') as fh:
            return fh.read()

    # ---------------------------------------------------------------- versions
    def publish(self, who: Principal, pid: str, manifest: dict, message: str = '', parent=None,
                content_sha256: str | None = None) -> dict:
        """Add a version: manifest {path: sha256} of blobs already uploaded. parent: the version it was based on
        (0: none yet); refused with 409 when the project has moved past it. The same files as an existing version:
        that version (existing: true)."""
        if not isinstance(manifest, dict) or not manifest:
            raise LabError(400, 'the manifest is empty')
        if len(manifest) > MAX_FILES:
            raise LabError(400, f'more than {MAX_FILES} files')
        man = {}
        for path, sha in manifest.items():
            p = check_path(path)
            s = sha[0] if isinstance(sha, (list, tuple)) else sha
            if not isinstance(s, str) or not SHA_RE.match(s):
                raise LabError(400, f'{p}: not a SHA-256')
            man[p] = s
        if 'project.json' not in man:
            raise LabError(400, 'a version needs project.json (a Vantage project folder)')
        miss = self.missing(man.values())
        if miss:
            raise LabError(400, f'{len(miss)} file(s) were not uploaded first', missing=miss)
        sizes = {p: os.path.getsize(self.blob_path(s)) for p, s in man.items()}
        digest = tree_digest(man)
        with self.db() as c:
            row = self._project_row(c, pid)
            if not self.can_write(who, row):
                raise LabError(403, f'{who.team} cannot publish to {pid}')
            c.execute('BEGIN IMMEDIATE')                       # one publish at a time per lab: numbers stay dense
            try:
                ex = c.execute('SELECT number FROM versions WHERE project=? AND digest=?', (pid, digest)).fetchone()
                head = c.execute('SELECT max(number) FROM versions WHERE project=?', (pid,)).fetchone()[0] or 0
                if ex is not None:
                    c.execute('COMMIT')
                    return dict(self._version_doc(c, pid, ex['number']), existing=True)
                if parent is not None and int(parent) != head:
                    c.execute('ROLLBACK')
                    raise LabError(409, f'{pid} is at version {head}, not {int(parent)}: pull and merge first',
                                   head=head)
                n = head + 1
                full = {p: [s, sizes[p]] for p, s in man.items()}
                c.execute('INSERT INTO versions VALUES(?,?,?,?,?,?,?,?,?,?)',
                          (pid, n, digest, int(parent) if parent is not None else None, str(message)[:2000],
                           f'{who.team}/{who.name or who.token_id}', utc_now(), content_sha256,
                           json.dumps(full, sort_keys=True), sum(sizes.values())))
                c.executemany('INSERT INTO version_blobs VALUES(?,?,?)', [(pid, n, s) for s in sorted(set(man.values()))])
                self.audit(c, who, 'version.publish', f'{pid}@{n} {digest[:12]}')
                c.execute('COMMIT')
            except LabError:
                raise
            except Exception:
                c.execute('ROLLBACK')
                raise
            return dict(self._version_doc(c, pid, n), existing=False)

    def _version_doc(self, c, pid, number, manifest=True) -> dict:
        r = c.execute('SELECT * FROM versions WHERE project=? AND number=?', (pid, int(number))).fetchone()
        if r is None:
            raise LabError(404, f'{pid} has no version {number}')
        d = {'project': pid, 'number': r['number'], 'digest': r['digest'], 'parent': r['parent'],
             'message': r['message'], 'author': r['author'], 'created': r['created'],
             'content_sha256': r['content_sha256'], 'bytes': r['bytes']}
        if manifest:
            d['manifest'] = json.loads(r['manifest'])
        return d

    def versions(self, who: Principal, pid: str) -> list:
        with self.db() as c:
            row = self._project_row(c, pid)
            if not self.can_read(who, row):
                raise LabError(403, f'{who.team} cannot read {pid}')
            return [self._version_doc(c, pid, r['number'], manifest=False)
                    for r in c.execute('SELECT number FROM versions WHERE project=? ORDER BY number', (pid,))]

    def version(self, who: Principal, pid: str, ref) -> dict:
        """ref: a number, 'head', or a digest (or its first 12+ hex digits)."""
        with self.db() as c:
            row = self._project_row(c, pid)
            if not self.can_read(who, row):
                raise LabError(403, f'{who.team} cannot read {pid}')
            ref = str(ref)
            if ref == 'head':
                r = c.execute('SELECT max(number) FROM versions WHERE project=?', (pid,)).fetchone()[0]
                if r is None:
                    raise LabError(404, f'{pid} has no version yet')
                return self._version_doc(c, pid, r)
            if ref.isdigit():
                return self._version_doc(c, pid, int(ref))
            if re.fullmatch(r'[0-9a-f]{12,64}', ref):
                rows = c.execute('SELECT number FROM versions WHERE project=? AND digest LIKE ?',
                                 (pid, ref + '%')).fetchall()
                if len(rows) == 1:
                    return self._version_doc(c, pid, rows[0]['number'])
                if len(rows) > 1:
                    raise LabError(400, f'{ref} names more than one version')
            raise LabError(404, f'{pid} has no version {ref}')

    def version_files(self, doc: dict) -> dict:
        """{path: bytes} of a version (each file checked against its SHA-256)."""
        out = {}
        for path, (sha, _) in doc['manifest'].items():
            b = self.read_blob(sha)
            if hashlib.sha256(b).hexdigest() != sha:
                raise LabError(500, f'{path} is damaged in the store (its SHA-256 differs): restore from a backup')
            out[path] = b
        return out

    # ---------------------------------------------------------------- records
    def add_records(self, who: Principal, pid: str, docs: list, validate) -> dict:
        if who.role not in ('member', 'admin'):
            raise LabError(403, 'readers cannot send records')
        with self.db() as c:
            row = self._project_row(c, pid)
            if not self.can_write(who, row):
                raise LabError(403, f'{who.team} cannot send records for {pid}')
            if not row['share_records']:
                raise LabError(403, f'{pid} does not share its perception records (share_records is off: opt in '
                                    'first)')
            added = dup = 0
            errors = []
            c.execute('BEGIN IMMEDIATE')
            for d in docs:
                errs = validate(d)
                if errs:
                    errors.append({'id': d.get('id') if isinstance(d, dict) else None, 'errors': errs[:5]})
                    continue
                cur = c.execute('INSERT OR IGNORE INTO records(id, origin_id, project, team, created, doc) '
                                'VALUES(?,?,?,?,?,?)', (d['id'], d.get('origin_id'), pid, who.team, utc_now(),
                                                       json.dumps(d, sort_keys=True, separators=(',', ':'))))
                if cur.rowcount:
                    added += 1
                else:
                    dup += 1
            self.audit(c, who, 'records.add', f'{pid} +{added}')
            c.execute('COMMIT')
        return {'added': added, 'duplicates': dup, 'rejected': len(errors), 'errors': errors[:20]}

    def records(self, who: Principal, after: int = 0, limit: int = 500) -> dict:
        limit = max(1, min(int(limit), 5000))
        with self.db() as c:
            rows = c.execute('SELECT seq, doc FROM records WHERE seq > ? ORDER BY seq LIMIT ?',
                             (int(after), limit)).fetchall()
            total = c.execute('SELECT count(*) FROM records').fetchone()[0]
        return {'records': [json.loads(r['doc']) for r in rows], 'next': rows[-1]['seq'] if rows else int(after),
                'more': len(rows) == limit, 'total': total}

    # ---------------------------------------------------------------- upkeep
    def gc(self) -> dict:
        """Remove uploaded files no version refers to (an upload whose publish never came), older than a day."""
        cut = (_dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(days=1)).isoformat()
        removed = 0
        with self.db() as c:
            rows = c.execute('SELECT sha, created FROM blobs WHERE sha NOT IN (SELECT sha FROM version_blobs)').fetchall()
            for r in rows:
                if r['created'].replace('Z', '+00:00') < cut:
                    try:
                        os.remove(self.blob_path(r['sha']))
                    except OSError:
                        pass
                    c.execute('DELETE FROM blobs WHERE sha=?', (r['sha'],))
                    removed += 1
        return {'removed': removed}

    def backup(self, dest: str) -> dict:
        """A consistent copy of the lab in dest (SQLite's online backup, then the objects): safe while it serves."""
        dest = os.path.abspath(dest)
        if os.path.exists(dest) and os.listdir(dest):
            raise LabError(400, f'{dest} is not empty')
        os.makedirs(dest, exist_ok=True)
        src = sqlite3.connect(os.path.join(self.root, DB))
        dst = sqlite3.connect(os.path.join(dest, DB))
        with dst:
            src.backup(dst)
        src.close()
        dst.close()
        n = 0
        for d in os.listdir(self.objects):
            sd = os.path.join(self.objects, d)
            if os.path.isdir(sd):
                os.makedirs(os.path.join(dest, 'objects', d), exist_ok=True)
                for f in os.listdir(sd):
                    shutil.copy2(os.path.join(sd, f), os.path.join(dest, 'objects', d, f))
                    n += 1
        return {'dest': dest, 'objects': n}

    def stats(self) -> dict:
        with self.db() as c:
            q = lambda s: c.execute(s).fetchone()[0]          # noqa: E731
            return {'teams': q('SELECT count(*) FROM teams'), 'projects': q('SELECT count(*) FROM projects'),
                    'versions': q('SELECT count(*) FROM versions'), 'blobs': q('SELECT count(*) FROM blobs'),
                    'bytes': q('SELECT coalesce(sum(size), 0) FROM blobs'), 'records': q('SELECT count(*) FROM records')}


class _Conn:
    """An SQLite connection that closes at the end of a with block (sqlite3's own only commits)."""

    def __init__(self, c):
        self._c = c

    def __enter__(self):
        return self._c

    def __exit__(self, *exc):
        try:
            if self._c.in_transaction:
                self._c.execute('ROLLBACK')
        finally:
            self._c.close()
        return False


def _mkstemp(folder):
    import tempfile
    return tempfile.mkstemp(dir=folder, prefix='up-', suffix='.part')
