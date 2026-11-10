"""Where Claude's tool calls go: a Vantage window that is open (LiveLink, through the window's bridge) or a project file
(FileSession, here in the MCP process).

A window that lets Claude in (Preferences ▸ General ▸ Claude, on by default) listens on 127.0.0.1 only and writes
<config>/mcp/<pid>.json (readable by the user alone) with its port and a random token; every request carries the
token. Without a window, FileSession opens the .vantage file, analyses it as the app does, applies each change as an
undoable step and saves it, after copying the file as it was to <name>.vantage.bak the first time."""
from __future__ import annotations

import copy
import json
import os
import shutil
import socket
import itertools
import threading
import sys
import time

from ..assistant.secrets import config_dir


def bridge_dir() -> str:
    return os.path.join(config_dir(), 'mcp')


def _alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if sys.platform.startswith('win'):
        return True                                  # a dead window's port refuses the connection instead
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def windows() -> list:
    """The open Vantage windows that let Claude in, newest first."""
    out = []
    try:
        names = os.listdir(bridge_dir())
    except OSError:
        return out
    for n in names:
        if not n.endswith('.json'):
            continue
        f = os.path.join(bridge_dir(), n)
        try:
            with open(f, encoding='utf-8') as fh:
                d = json.load(fh)
            if not _alive(int(d.get('pid', 0))):
                os.remove(f)
                continue
            out.append(d)
        except (OSError, ValueError, TypeError):
            continue
    return sorted(out, key=lambda d: -float(d.get('started', 0)))


def same_file(a: str, b: str) -> bool:
    try:
        return bool(a) and bool(b) and os.path.samefile(a, b)
    except OSError:
        return os.path.abspath(a) == os.path.abspath(b)


# =========================================================================== the window
class LinkError(Exception):
    pass


_CALLS = itertools.count(1)


class LiveLink:
    """Requests to one window's bridge: one JSON line each way, on a small pool of connections so that calls can run
    side by side and a cancel reaches the window while a call is still running."""
    mode = 'live'

    def __init__(self, info: dict, timeout=900.0):
        self.info = info
        self.timeout = timeout
        self._free = []                      # idle connections: [socket, buffer]
        self._lock = threading.Lock()
        self._n = 0

    @property
    def pid(self) -> int:
        return int(self.info.get('pid', 0))

    def _take(self):
        with self._lock:
            if self._free:
                return self._free.pop()
        try:
            sock = socket.create_connection(('127.0.0.1', int(self.info['port'])), timeout=10)
        except OSError as e:
            raise LinkError(f'the Vantage window (pid {self.pid}) does not answer: {e}') from None
        sock.settimeout(self.timeout)
        return [sock, b'']

    def _give(self, conn):
        with self._lock:
            if len(self._free) < 4:
                self._free.append(conn)
                return
        try:
            conn[0].close()
        except OSError:
            pass

    def request(self, method: str, **params) -> dict:
        conn = self._take()
        with self._lock:
            self._n += 1
            n = self._n
        msg = {'token': self.info['token'], 'id': n, 'method': method, **params}
        sock = conn[0]
        try:
            sock.sendall((json.dumps(msg) + '\n').encode('utf-8'))
            while b'\n' not in conn[1]:
                chunk = sock.recv(1 << 20)
                if not chunk:
                    raise OSError('the window closed the connection')
                conn[1] += chunk
        except OSError as e:
            try:
                sock.close()
            except OSError:
                pass
            raise LinkError(f'lost the Vantage window (pid {self.pid}): {e}') from None
        line, conn[1] = conn[1].split(b'\n', 1)
        self._give(conn)
        doc = json.loads(line.decode('utf-8'))
        if 'error' in doc:
            raise LinkError(str(doc['error']))
        return doc.get('result') or {}

    def call(self, name, args, cancel=None) -> dict:
        """A tool call in the window; cancel (a threading.Event) set while it runs is passed on to the window."""
        cid = f'{os.getpid()}-{next(_CALLS)}'          # unique at the window: this process and its count
        if cancel is None:
            return self.request('call', name=name, args=args or {}, call_id=cid)
        stop = threading.Event()

        def watch():
            while not stop.is_set():
                if cancel.wait(0.2):
                    try:
                        self.request('cancel', call_id=cid)
                    except LinkError:
                        pass
                    return
        th = threading.Thread(target=watch, daemon=True)
        th.start()
        try:
            return self.request('call', name=name, args=args or {}, call_id=cid)
        finally:
            stop.set()

    def revision(self):
        """(project path, a number that changes with every change to the project) for resource subscriptions."""
        r = self.request('ping')
        return r.get('path') or r.get('project'), (int(r.get('revision', 0)), int(r.get('undo_index', 0)))

    def close(self):
        with self._lock:
            conns, self._free = self._free, []
        for c in conns:
            try:
                c[0].close()
            except OSError:
                pass


# =========================================================================== a file
class FileSession:
    """A project file worked on without the window: the same tools, edits as undoable steps, saved as they are made."""
    mode = 'file'

    def __init__(self):
        self.p = None
        self.undo_stack, self.redo_stack = [], []
        self.rev = 0
        self._an = None
        self._mtime = 0.0
        self._backed_up = set()
        self._dirty = False
        self._lock = threading.RLock()        # one call at a time on the file (the analysis and the undo stack)

    # ---------------------------------------------------------------- opening
    def open(self, path=None, sample=False, template=None) -> dict:
        from ..core import samples
        from ..core.project import Project
        if path:
            path = os.path.abspath(os.path.expanduser(path))
            if not path.lower().endswith('.vantage') or not os.path.isfile(path):
                return {'error': f'{path} is not a .vantage project file'}
            open_in = [w for w in windows() if same_file(w.get('path', ''), path)]
            if open_in:
                return {'error': f'{os.path.basename(path)} is open in a Vantage window (pid {open_in[0]["pid"]}): Claude '
                                 'works on it there'}
            try:
                p = Project.load(path)
            except (OSError, ValueError, KeyError) as e:
                return {'error': f'cannot open {path}: {e}'}
            self._mtime = os.path.getmtime(path)
        elif sample:
            p = samples.tigor_sample_project()
        elif template:
            try:
                p = samples.template_project(template)
            except (KeyError, ValueError) as e:
                return {'error': f'template: {e}'}
        else:
            return {'error': 'give path (a .vantage file), sample: true or template'}
        self.p = p
        self.undo_stack, self.redo_stack, self.rev, self._an, self._dirty = [], [], 0, None, False
        return {'opened': p.path or f'{p.name} (not saved yet: save_project with a path keeps it)',
                'variant': p.variant.name, 'variants': [v.name for v in p.variants]}

    def _need(self):
        if self.p is None:
            raise RuntimeError('no project: open the Vantage app (Claude then works in its window) or call '
                               'open_project with a path, sample: true or a template')

    # ---------------------------------------------------------------- the session interface
    def project(self):
        self._need()
        return self.p.snapshot()

    def where(self) -> str:
        return f'project file {self.p.path}' if self.p is not None and self.p.path else 'an unsaved project in memory'

    def unsaved(self) -> bool:
        return self._dirty

    def context(self):
        from ..assistant import tools as T
        from ..assistant.evals import _analysed
        from .. import workdir
        self._need()
        if self._an is None or self._an[0] != self.rev:
            self._an = (self.rev, _analysed(self.p))
        a = self._an[1]
        return T.ToolContext(self.p.snapshot(), a['result'], a['engine'].share(), list(a['issues']), dict(a['mounts']),
                             a['worst'], revision=self.rev, brief=self.p.analysis.assistant.get('brief', ''),
                             working_dir=workdir.default(), provider='claude', model='mcp')

    def edit(self, keys, fn, text):
        self._need()
        self._check_disk()
        p = self.p
        keys = [k if k != 'variant' else f'variant:{p.active_uid}' for k in keys]     # as the window's store does
        before = {k: _state(p, k) for k in keys}
        try:
            fn(p)
        except Exception:
            for k, st in before.items():
                p.restore(k, _copy(k, st))
            raise
        after = {k: _state(p, k) for k in keys}
        self.undo_stack.append((text, before, after))
        self.redo_stack.clear()
        p.add_log('edit', text)
        self._changed()

    def _changed(self):
        self.rev += 1
        self._dirty = True
        if self.p.path:
            self._write(self.p.path)

    def _check_disk(self):
        path = self.p.path
        if path and os.path.exists(path) and self._mtime and abs(os.path.getmtime(path) - self._mtime) > 1e-3:
            raise RuntimeError(f'{os.path.basename(path)} changed on disk since Claude opened it: open_project it again '
                               'before changing it (nothing was changed)')
        if path and any(same_file(w.get('path', ''), path) for w in windows()):
            raise RuntimeError(f'{os.path.basename(path)} is now open in a Vantage window: Claude works on it there '
                               '(nothing was changed here)')

    def _write(self, path):
        if os.path.exists(path):
            if path not in self._backed_up:
                shutil.copy2(path, path + '.bak')
                self._backed_up.add(path)
        self.p.save(path)
        self._mtime = os.path.getmtime(path)
        self._dirty = False

    def undo(self) -> str:
        self._need()
        if not self.undo_stack:
            return 'nothing to undo'
        self._check_disk()
        text, before, after = self.undo_stack.pop()
        for k, st in before.items():
            self.p.restore(k, _copy(k, st))
        self.redo_stack.append((text, before, after))
        self._changed()
        return f'undid: {text}'

    def redo(self) -> str:
        self._need()
        if not self.redo_stack:
            return 'nothing to redo'
        self._check_disk()
        text, before, after = self.redo_stack.pop()
        for k, st in after.items():
            self.p.restore(k, _copy(k, st))
        self.undo_stack.append((text, before, after))
        self._changed()
        return f'redid: {text}'

    def save(self, path=None) -> str:
        self._need()
        if path:
            path = os.path.abspath(os.path.expanduser(path))
            if not path.lower().endswith('.vantage'):
                path += '.vantage'
            if any(same_file(w.get('path', ''), path) for w in windows()):
                return f'{os.path.basename(path)} is open in a Vantage window: not overwritten'
            self.p.path = path
            self.p.name = os.path.splitext(os.path.basename(path))[0]
            self._mtime = os.path.getmtime(path) if os.path.exists(path) else 0.0
        if not self.p.path:
            return 'not saved: the project has no file yet; give path'
        self._write(self.p.path)
        return f'saved {self.p.path}'

    def activity(self, limit=20) -> dict:
        self._need()
        steps = [{'text': t, 'by': 'claude' if t.startswith('Claude ·') else 'user', 'state': 'done'}
                 for t, _, _ in self.undo_stack[-limit:]]
        log = [{'kind': e.kind, 'text': e.text, 'time': time.strftime('%Y-%m-%d %H:%M', time.localtime(e.t))}
               for e in self.p.log[-limit:]]
        return {'where': self.where(), 'steps_this_session': steps, 'project_log': log,
                'unsaved_changes': self._dirty}

    def show(self, req):
        raise _needs()

    proposal = plan = log = view = render = turntable = show

    def revision(self):
        """(project path or name, a number that changes with every change) for resource subscriptions."""
        if self.p is None:
            return None, (-1, 0)
        return self.p.path or self.p.name, (self.rev, id(self.p))

    def call(self, name, args, cancel=None) -> dict:
        from . import service as SV
        with self._lock:
            return SV.call(self, name, args, cancel=cancel)


def _state(p, k):
    """A section's state for undo. The vehicle keeps its mesh by reference (a mesh is never changed in place: an import
    or an edit replaces it), as the window's store keeps it; everything else is copied."""
    if k == 'vehicle':
        return {'meta': copy.deepcopy(p.vehicle.meta()), 'vehicle': copy.copy(p.vehicle)}
    return copy.deepcopy(p.section(k))


def _copy(k, st):
    return st if k == 'vehicle' else copy.deepcopy(st)


def _needs():
    from .service import NeedsWindow
    return NeedsWindow('this shows in the Vantage window, and none is open')
