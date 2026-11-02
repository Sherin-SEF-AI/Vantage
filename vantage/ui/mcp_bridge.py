"""The window's side of Claude over MCP: the Vantage MCP server (vantage.mcp.server, started by Claude Code or Claude
Desktop) sends each tool call here, and it runs on this window's project.

The bridge listens on 127.0.0.1 only, on a free port, and writes <config>/mcp/<pid>.json (the port and a random token;
readable by the user alone) for the server to find. Every request carries the token. A tool runs on a worker thread
(the window stays responsive); whatever touches the project or the window is handed to the UI thread. A change is
one undo step "Claude · …" in Edit ▸ Undo like the user's own, and the reply waits for the analysis of it.
Preferences ▸ General ▸ Claude turns it off (setting mcp/bridge, on by default)."""
from __future__ import annotations

import hmac
import json
import os
import secrets
import threading
import time

from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtNetwork import QHostAddress, QTcpServer

from ..mcp import service as SV
from ..mcp.session import bridge_dir

SETTING = 'mcp/bridge'
CLAUDE = 'Claude ·'


def enabled(settings) -> bool:
    return settings.value(SETTING, True, type=bool)


class _Invoker(QObject):
    """Runs a function on the UI thread for a worker thread and hands back its result."""
    job = pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self.job.connect(self._run)

    def _run(self, item):
        fn, box, done = item
        try:
            box['value'] = fn()
        except BaseException as e:               # noqa: BLE001 — re-raised in the worker
            box['error'] = e
        finally:
            done.set()

    def call(self, fn, timeout=120.0):
        if threading.current_thread() is threading.main_thread():
            return fn()
        box, done = {}, threading.Event()
        self.job.emit((fn, box, done))
        if not done.wait(timeout):
            raise TimeoutError('the Vantage window did not answer in time')
        if 'error' in box:
            raise box['error']
        return box.get('value')


class LiveSession:
    """The session interface of vantage.mcp.service on this window's project (called from a worker thread)."""
    mode = 'live'

    def __init__(self, ctx, invoker: _Invoker):
        self.ctx = ctx
        self.ui = invoker.call

    # ---------------------------------------------------------------- reading
    def project(self):
        return self.ui(lambda: self.ctx.store.project.snapshot())

    def where(self) -> str:
        p = self.ctx.store.project
        return f'the Vantage window (pid {os.getpid()}): {p.path or "a project not saved yet"}'

    def unsaved(self) -> bool:
        return self.ui(lambda: not self.ctx.store.undo.isClean())

    def _fresh(self, timeout=180.0) -> bool:
        """Wait until the analysis covers the project as it is now (and its issues are in)."""
        st, an = self.ctx.store, self.ctx.analysis
        end = time.time() + timeout
        asked = False
        while time.time() < end:
            ok, busy = self.ui(lambda: (an.result is not None and getattr(an.result, 'revision', -1) == st.revision,
                                        an._busy or an.timer.isActive()))
            if ok and not busy:
                return True
            if not busy and not ok and not asked:
                self.ui(lambda: an.request(force=True))
                asked = True
            elif not busy:
                asked = False
            time.sleep(0.05)
        return False

    def context(self):
        self._fresh()
        return self.ui(self.ctx.assistant._context)

    # ---------------------------------------------------------------- changing
    def edit(self, keys, fn, text):
        def run():
            st = self.ctx.store
            p = st.project
            import copy
            saved = {k: copy.deepcopy(p.section(k)) for k in keys}
            try:
                st.edit([k if not k.startswith(f'variant:{p.active_uid}') else 'variant' for k in keys], fn, text)
            except Exception:
                for k, state in saved.items():          # the trial passed, so this is a race: leave it as it was
                    p.restore(k, state)
                raise
            w = self.ctx.window
            if w is not None:
                if 'variants' in keys and hasattr(w, '_fill_variant_combo'):
                    w._fill_variant_combo()
                w.msg(text, 5000)
        self.ui(run)
        self._fresh()

    def _step(self, redo: bool) -> str:
        u = self.ctx.store.undo
        i = u.index()
        j = i if redo else i - 1
        if not 0 <= j < u.count():
            return 'nothing to ' + ('redo' if redo else 'undo')
        text = u.command(j).text()
        if not text.startswith(CLAUDE):
            return (f'the {"next" if redo else "last"} step is the user\'s ({text}): left as it is')
        (u.redo if redo else u.undo)()
        if self.ctx.window is not None:
            self.ctx.window.msg(f'{"Redid" if redo else "Undid"}: {text}', 5000)
        return f'{"redid" if redo else "undid"}: {text}'

    def undo(self) -> str:
        r = self.ui(lambda: self._step(False))
        self._fresh()
        return r

    def redo(self) -> str:
        r = self.ui(lambda: self._step(True))
        self._fresh()
        return r

    def save(self, path=None) -> str:
        def run():
            w, p = self.ctx.window, self.ctx.store.project
            if path:
                q = os.path.abspath(os.path.expanduser(path))
                if not q.lower().endswith('.vantage'):
                    q += '.vantage'
                p.path = q
                p.name = os.path.splitext(os.path.basename(q))[0]
            if not p.path:
                return 'not saved: this project has no file yet; give path (or the user saves it in the window)'
            return f'saved {p.path}' if w is not None and w.save() else 'not saved (the window reported a problem)'
        return self.ui(run)

    def activity(self, limit=20) -> dict:
        def run():
            st, w = self.ctx.store, self.ctx.window
            u = st.undo
            i = u.index()
            steps = []
            for j in range(max(0, u.count() - limit), u.count()):
                t = u.command(j).text()
                steps.append({'text': t, 'by': 'claude' if t.startswith(CLAUDE) else 'user',
                              'state': 'done' if j < i else 'undone'})
            return {'where': self.where(), 'steps': steps, 'page': w.page_name() if w is not None else '',
                    'selection': [c.tag for c in st.selected_cams()], 'variant': st.project.variant.name,
                    'unsaved_changes': not u.isClean()}
        return self.ui(run)

    def open(self, **_):
        return {'error': 'Claude works on the project open in the Vantage window: open another one there (File ▸ Open)'}

    # ---------------------------------------------------------------- showing in the window
    def _panel(self):
        w = self.ctx.window
        if w is None:
            return None
        side = getattr(w, 'side_panel', None)
        if side is not None and side.mode() == 'claude' and w.docks['assistant'].isVisible():
            w.msg('Claude put a card in the assistant side of the panel: switch to it to apply it', 8000)
        else:                                          # (never switch away from the terminal Claude talks in)
            w.show_assistant()
        return w.assistant_panel

    def show(self, req):
        ctl = self.ctx.assistant
        self.ui(lambda: ctl._do_ui((ctl._tid, req)))

    def proposal(self, prop):
        def run():
            ctl = self.ctx.assistant
            ctl.proposals[prop.id] = {'p': prop, 'state': 'pending', 'undo_index': -1}
            pn = self._panel()
            if pn is not None:
                pn.show_from_claude('proposal', prop)
        self.ui(run)

    def plan(self, plan):
        def run():
            ctl = self.ctx.assistant
            ctl._add_plan(plan)
            pn = self._panel()
            if pn is not None:
                pn.show_from_claude('copilot', plan)
        self.ui(run)

    def log(self, entry):
        def run():
            ctl = self.ctx.assistant
            ctl.log_proposals[entry['id']] = {'entry': entry, 'state': 'pending'}
            pn = self._panel()
            if pn is not None:
                pn.show_from_claude('log', entry)
        self.ui(run)


class Bridge(QObject):
    """The local server the MCP process talks to."""
    _reply = pyqtSignal(object, object)          # (socket, reply dict): written on the UI thread
    calls = pyqtSignal(str)                      # a tool Claude called (for the status bar)

    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        self.server = None
        self.token = ''
        self.file = ''
        self.invoker = _Invoker()
        self.session = LiveSession(ctx, self.invoker)
        self._buf = {}
        self._reply.connect(self._write)

    @property
    def running(self) -> bool:
        return self.server is not None and self.server.isListening()

    def start(self) -> bool:
        if self.running:
            return True
        self.server = QTcpServer(self)
        if not self.server.listen(QHostAddress(QHostAddress.SpecialAddress.LocalHost), 0):
            self.server = None
            return False
        self.server.newConnection.connect(self._accept)
        self.token = secrets.token_hex(24)
        self._publish()
        st = self.ctx.store
        st.project_replaced.connect(self._publish)
        return True

    def stop(self):
        if self.server is not None:
            self.server.close()
            self.server.deleteLater()
            self.server = None
        try:
            self.ctx.store.project_replaced.disconnect(self._publish)
        except (TypeError, RuntimeError):
            pass
        if self.file:
            try:
                os.remove(self.file)
            except OSError:
                pass
            self.file = ''

    def _publish(self, *_):
        if not self.running:
            return
        d = bridge_dir()
        os.makedirs(d, mode=0o700, exist_ok=True)
        self.file = os.path.join(d, f'{os.getpid()}.json')
        p = self.ctx.store.project
        doc = {'pid': os.getpid(), 'port': int(self.server.serverPort()), 'token': self.token,
               'project': p.name, 'path': p.path, 'started': time.time()}
        tmp = self.file + '.tmp'
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(doc, f)
        os.replace(tmp, self.file)

    # ---------------------------------------------------------------- connections
    def _accept(self):
        while self.server is not None and self.server.hasPendingConnections():
            sock = self.server.nextPendingConnection()
            self._buf[id(sock)] = b''
            sock.readyRead.connect(lambda s=sock: self._read(s))
            sock.disconnected.connect(lambda s=sock: (self._buf.pop(id(s), None), s.deleteLater()))

    def _read(self, sock):
        buf = self._buf.get(id(sock), b'') + bytes(sock.readAll())
        if len(buf) > 32 << 20:
            sock.abort()
            return
        while b'\n' in buf:
            line, buf = buf.split(b'\n', 1)
            self._handle(sock, line)
        self._buf[id(sock)] = buf

    def _handle(self, sock, line):
        try:
            req = json.loads(line.decode('utf-8'))
        except ValueError:
            return self._write(sock, {'id': None, 'error': 'not JSON'})
        rid = req.get('id')
        if not hmac.compare_digest(str(req.get('token', '')), self.token):
            self._write(sock, {'id': rid, 'error': 'refused: wrong token'})
            sock.disconnectFromHost()
            return
        if req.get('method') == 'ping':
            p = self.ctx.store.project
            return self._write(sock, {'id': rid, 'result': {'project': p.name, 'path': p.path}})
        if req.get('method') != 'call':
            return self._write(sock, {'id': rid, 'error': f'no method {req.get("method")!r}'})
        name, args = str(req.get('name', '')), req.get('args') or {}
        t = SV.BY_NAME.get(name)
        self.calls.emit(name)
        w = self.ctx.window
        if w is not None and t is not None:
            w.msg(f'Claude: {t.title}…', 4000)
        threading.Thread(target=self._work, args=(sock, rid, name, args), daemon=True).start()

    def _work(self, sock, rid, name, args):
        out = SV.call(self.session, name, args)
        try:
            self._reply.emit(sock, {'id': rid, 'result': out})
        except RuntimeError:                       # the window closed meanwhile
            pass

    def _write(self, sock, doc):
        try:
            if sock.state() == sock.SocketState.ConnectedState:
                sock.write((json.dumps(doc, ensure_ascii=False, default=str) + '\n').encode('utf-8'))
        except RuntimeError:                       # the socket is gone
            pass
