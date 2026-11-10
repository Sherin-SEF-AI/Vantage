"""The Vantage MCP server: Claude (Claude Code, Claude Desktop or any MCP client) reads and changes rig designs.

    vantage --mcp [--mcp-project FILE.vantage] [--mcp-mode auto|live|file]
    python -m vantage.mcp ...

It speaks MCP over stdio (JSON-RPC 2.0, one message per line) with the Python standard library only. Each tool call
goes to the Vantage window that is open (live: the change shows there as an undo step "Claude · …"), or, with no
window, to the project file (file mode: analysed here, saved after each change, the original kept as .bak).

What it offers (MCP 2024-11-05 to 2025-11-25):

- tools: about a hundred (vantage.mcp.service), annotated (read-only, destructive, idempotent), with structured
  content for every JSON result and pictures where a tool draws one;
- background jobs: start_job runs any tool in the background, get_job / cancel_job / list_jobs follow it;
- cancellation (notifications/cancelled) of any call: a batch run is stopped at once, a study at its next check;
- progress (notifications/progress, every PROGRESS_S seconds while a call runs, when the client gives a token);
- resources and resource templates (vantage.mcp.catalog), with subscriptions: a subscribed client is told when the
  project changes, in the window or on the file;
- prompts for the common workflows, and argument completion for prompts and resource templates;
- logging (notifications/message, once the client sets a level with logging/setLevel).

Calls run on worker threads in the order they arrive: reads run side by side, a call that changes something waits
for every call before it and holds back the ones after it, so a client that sends an edit and then a read gets the
read after the edit.

Register it with Claude Code once:

    claude mcp add --scope user vantage -- vantage --mcp
    (from source: claude mcp add --scope user vantage -- python3 /path/to/run_vantage.py --mcp)"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import traceback

from .. import __version__
from . import catalog as CT
from . import service as SV
from .jobs import Jobs
from .session import FileSession, LinkError, LiveLink, same_file, windows

VERSIONS = ('2025-11-25', '2025-06-18', '2025-03-26', '2024-11-05')
LEVELS = ('debug', 'info', 'notice', 'warning', 'error', 'critical', 'alert', 'emergency')
PROGRESS_S = 2.0                 # a progress notification this often while a call with a progress token runs
WATCH_S = 1.5                    # how often subscriptions look for a change of the project
JOB_TOOLS = ('start_job', 'get_job', 'cancel_job', 'list_jobs')


class Router:
    """Which session a call goes to: the open window (newest, or the one with --mcp-project open), else the file."""

    def __init__(self, project=None, mode='auto'):
        self.mode = mode
        self.project = project
        self.file = FileSession()
        self.link = None
        self.sticky_file = mode == 'file'
        self._tried = False
        self._lock = threading.RLock()
        self.jobs = None                          # set by the Server

    def session(self):
        with self._lock:
            if not self.sticky_file:
                ws = windows()
                if ws:
                    here = os.environ.get('VANTAGE_MCP_PID', '')  # started in a window's Terminal panel: that window
                    pick = next((w for w in ws if here and str(w['pid']) == here), None) or \
                        next((w for w in ws if self.project and same_file(w.get('path', ''), self.project)), None) or \
                        next((w for w in ws if self.link is not None and int(w['pid']) == self.link.pid), None) or ws[0]
                    if self.link is None or self.link.pid != int(pick['pid']):
                        if self.link is not None:
                            self.link.close()
                        self.link = LiveLink(pick)
                    else:
                        self.link.info = pick
                    return self.link
                if self.mode == 'live':
                    raise LinkError('no Vantage window is open with Claude allowed in (Preferences ▸ General ▸ Claude)')
            if self.file.p is None and self.project and not self._tried:
                self._tried = True
                r = self.file.open(path=self.project)
                if 'error' in r:
                    print(f'vantage mcp: {r["error"]}', file=sys.stderr)
            return self.file

    def open(self, args) -> dict:
        with self._lock, self.file._lock:
            if not any(args.get(k) for k in ('path', 'sample', 'template')):
                self.sticky_file = self.mode == 'file'
                ws = windows()
                return {'result': 'working in the Vantage window again' if ws and not self.sticky_file else
                        'no Vantage window is open: give path, sample or template to work on a file'}
            r = self.file.open(path=args.get('path'), sample=bool(args.get('sample')), template=args.get('template'))
            if 'error' not in r:
                self.sticky_file = True
                r['note'] = ('Claude works on this file now, even with the Vantage window open; open_project without '
                             'arguments goes back to the window')
            return r

    def call(self, name, args, cancel=None) -> dict:
        args = args if isinstance(args, dict) else {}
        if name == 'open_project':
            out = self.open(args)
            return SV.text(out, 'error' in out)
        if name in JOB_TOOLS:
            return self._job_tool(name, args)
        try:
            s = self.session()
        except LinkError as e:
            return SV.text({'error': str(e)}, True)
        if isinstance(s, LiveLink):
            try:
                return s.call(name, args, cancel)
            except LinkError as e:
                with self._lock:
                    self.link = None
                return SV.text({'error': f'{e}. Call the tool again to use whatever is open now.'}, True)
        return s.call(name, args, cancel)

    # ---------------------------------------------------------------- background jobs
    def _job_tool(self, name, args) -> dict:
        jobs = self.jobs
        if jobs is None:
            return SV.text({'error': 'background jobs run only in the MCP server'}, True)
        if name == 'start_job':
            tool = str(args.get('tool') or '')
            t = SV.BY_NAME.get(tool)
            if t is None and tool != 'open_project':
                return SV.text({'error': f'no tool {tool!r}'}, True)
            if tool in JOB_TOOLS:
                return SV.text({'error': 'a job runs a tool, not another job tool'}, True)
            a = args.get('arguments') or {}
            if not isinstance(a, dict):
                return SV.text({'error': 'arguments: an object (the tool\'s arguments)'}, True)
            return SV.text(jobs.start(tool, a, str(args.get('label') or '')))
        if name == 'list_jobs':
            return SV.text({'jobs': jobs.list()})
        if name == 'cancel_job':
            out = jobs.cancel(args.get('job_id'))
            return SV.text(out, 'error' in out)
        job = jobs.get(args.get('job_id'), args.get('wait_s') or 0)
        if job is None:
            return SV.text({'error': f'no job {args.get("job_id")!r} (list_jobs lists them)'}, True)
        return job_result(job)

    # ---------------------------------------------------------------- for subscriptions and completion
    def revision(self):
        try:
            s = self.session()
            return s.revision()
        except Exception:                          # noqa: BLE001 — no window, no file: nothing to watch
            return None

    def lookup(self, kind) -> list:
        if kind == 'jobs':
            return self.jobs.ids() if self.jobs is not None else []
        try:
            r = self.call('vantage_status' if kind in ('cameras', 'variants') else 'get_issues', {})
        except Exception:                          # noqa: BLE001
            return []
        st = r.get('structuredContent') or {}
        if kind == 'cameras':
            return list(st.get('cameras', [])) + list(st.get('sensors', []))
        if kind == 'variants':
            return list(st.get('variants', []))
        if kind == 'issues':
            items = st.get('issues', st.get('items', []))
            return [i.get('message', str(i)) if isinstance(i, dict) else str(i) for i in items]
        return []


def job_result(job) -> dict:
    """get_job's result: the job's state, and when it has ended, the tool's own result (its pictures too)."""
    summ = job.summary()
    content = [{'type': 'text', 'text': json.dumps(summ, ensure_ascii=False)}]
    out = {'content': content, 'isError': False, 'structuredContent': {'job': summ}}
    if job.result is not None:
        content.extend(job.result.get('content', []))
        if 'structuredContent' in job.result:
            out['structuredContent']['result'] = job.result['structuredContent']
    return out


class _Order:
    """Arrival order for calls: a read starts once every earlier write has finished; a write starts once every
    earlier call has."""

    def __init__(self):
        self._c = threading.Condition()
        self._n = 0
        self._open = {}                            # ticket -> 'r' | 'w' (not finished)

    def enter(self, write: bool) -> int:
        with self._c:
            t = self._n
            self._n += 1
            self._open[t] = 'w' if write else 'r'
            while any(k < t and (write or v == 'w') for k, v in self._open.items()):
                self._c.wait()
            return t

    def leave(self, t):
        with self._c:
            self._open.pop(t, None)
            self._c.notify_all()


class Server:
    def __init__(self, router: Router, out):
        self.router = router
        self.out = out
        self._wlock = threading.Lock()
        self.level = None                          # no log messages until the client sets a level
        self.inflight = {}                         # request id -> cancel event
        self.threads = []
        self.order = _Order()
        self.subs = set()
        self._subs_lock = threading.Lock()
        self._watch = None
        self._stop = threading.Event()
        self.client = {}
        self.jobs = Jobs(lambda tool, args, cancel: self._run_tool(tool, args, cancel), on_done=self._job_done)
        router.jobs = self.jobs

    # ---------------------------------------------------------------- output
    def send(self, doc):
        line = json.dumps(doc, ensure_ascii=False, default=str) + '\n'
        with self._wlock:
            try:
                self.out.write(line)
                self.out.flush()
            except (OSError, ValueError):          # the client went away
                pass

    def notify(self, method, params):
        self.send({'jsonrpc': '2.0', 'method': method, 'params': params})

    def log(self, level, data):
        if self.level is not None and LEVELS.index(level) >= LEVELS.index(self.level):
            self.notify('notifications/message', {'level': level, 'logger': 'vantage', 'data': data})

    # ---------------------------------------------------------------- running tools
    def _run_tool(self, name, args, cancel):
        t = SV.BY_NAME.get(name)
        write = t is not None and not t.read_only and name not in ('get_job', 'list_jobs', 'cancel_job', 'start_job')
        ticket = self.order.enter(write) if name not in JOB_TOOLS else None
        t0 = time.time()
        try:
            self.log('debug', {'tool': name, 'event': 'start'})
            res = self.router.call(name, args, cancel)
        finally:
            if ticket is not None:
                self.order.leave(ticket)
        dt = time.time() - t0
        if res.get('isError'):
            body = next((c.get('text', '') for c in res.get('content', []) if c.get('type') == 'text'), '')
            self.log('warning', {'tool': name, 'event': 'error', 'seconds': round(dt, 2), 'error': body[:500]})
        else:
            self.log('info', {'tool': name, 'event': 'done', 'seconds': round(dt, 2)})
        return res

    def _job_done(self, job):
        self.log('notice', {'job': job.id, 'tool': job.tool, 'state': job.state})
        uri = f'vantage://job/{job.id}'
        with self._subs_lock:
            if uri in self.subs:
                self.notify('notifications/resources/updated', {'uri': uri})

    def _heartbeat(self, token, name, done, t0):
        n = 0
        while not done.wait(PROGRESS_S):
            n += 1
            el = time.time() - t0
            self.notify('notifications/progress', {'progressToken': token, 'progress': round(el, 1),
                                                   'message': f'{name}: running for {el:.0f} s'})

    def _threaded(self, mid, fn, progress=None, label=''):
        ev = threading.Event()
        self.inflight[mid] = ev

        def run():
            done = threading.Event()
            if progress is not None:
                threading.Thread(target=self._heartbeat, args=(progress, label, done, time.time()),
                                 daemon=True).start()
            try:
                result = fn(ev)
                reply = {'jsonrpc': '2.0', 'id': mid, 'result': result}
            except CT.NotFound as e:
                reply = {'jsonrpc': '2.0', 'id': mid, 'error': {'code': -32002, 'message': str(e)}}
            except (ValueError, TypeError) as e:
                reply = {'jsonrpc': '2.0', 'id': mid, 'error': {'code': -32602, 'message': str(e)}}
            except Exception as e:                 # noqa: BLE001 — the server keeps running
                traceback.print_exc(file=sys.stderr)
                reply = {'jsonrpc': '2.0', 'id': mid, 'error': {'code': -32603, 'message': f'{type(e).__name__}: {e}'}}
            finally:
                done.set()
                self.inflight.pop(mid, None)
            if not ev.is_set():                    # a cancelled request gets no answer
                self.send(reply)
        th = threading.Thread(target=run, daemon=True)
        self.threads.append(th)
        self.threads = [t for t in self.threads if t.is_alive() or t is th]
        th.start()

    # ---------------------------------------------------------------- resources
    def _read(self, uri, cancel):
        kind, *rest = CT.resolve(uri)
        if kind == 'doc':
            return {'contents': CT.doc_contents(uri, rest[0])}
        if kind == 'job':
            job = self.jobs.get(rest[0])
            if job is None:
                raise CT.NotFound(f'no job {rest[0]!r}')
            r = job_result(job)
            return {'contents': CT.contents_from_result(uri, r)}
        name, args = rest
        res = self._run_tool(name, args, cancel)
        if res.get('isError'):
            body = next((c.get('text', '') for c in res.get('content', []) if c.get('type') == 'text'), '')
            raise CT.NotFound(f'{uri}: {body[:400]}')
        return {'contents': CT.contents_from_result(uri, res)}

    def _subscribe(self, uri, on):
        CT.resolve(uri)                            # NotFound for an unknown URI
        with self._subs_lock:
            (self.subs.add if on else self.subs.discard)(uri)
            if self.subs and (self._watch is None or not self._watch.is_alive()):
                self._watch = threading.Thread(target=self._watcher, daemon=True)
                self._watch.start()
        return {}

    def _watcher(self):
        last = self.router.revision()
        while not self._stop.wait(WATCH_S):
            with self._subs_lock:
                subs = [u for u in self.subs if not u.startswith(('vantage://docs/', 'vantage://job/',
                                                                  'vantage://camera-library',
                                                                  'vantage://validation-checks',
                                                                  'vantage://export-formats',
                                                                  'vantage://batch-options'))]
                if not self.subs:
                    return
            now = self.router.revision()
            if now != last and now is not None:
                last = now
                for u in subs:
                    self.notify('notifications/resources/updated', {'uri': u})

    # ---------------------------------------------------------------- the protocol
    def handle(self, msg):
        if not isinstance(msg, dict) or msg.get('jsonrpc') != '2.0':
            return {'jsonrpc': '2.0', 'id': None, 'error': {'code': -32600, 'message': 'invalid request'}}
        mid, method, params = msg.get('id'), msg.get('method'), msg.get('params') or {}
        if 'method' not in msg:                   # a response to something we sent (we send no requests): ignore
            return None
        if mid is None:                           # a notification: nothing to answer
            if method == 'notifications/cancelled':
                ev = self.inflight.get(params.get('requestId'))
                if ev is not None:
                    ev.set()
            return None
        meta = params.get('_meta') or {}
        token = meta.get('progressToken') if isinstance(meta, dict) else None
        try:
            if method == 'initialize':
                want = params.get('protocolVersion')
                self.client = params.get('capabilities') or {}
                result = {'protocolVersion': want if want in VERSIONS else VERSIONS[0],
                          'capabilities': {'tools': {'listChanged': False},
                                           'resources': {'subscribe': True, 'listChanged': False},
                                           'prompts': {'listChanged': False},
                                           'completions': {}, 'logging': {}},
                          'serverInfo': {'name': 'vantage', 'title': 'Vantage', 'version': __version__,
                                         'websiteUrl': 'https://sherin-sef-ai.github.io/Vantage/'},
                          'instructions': SV.PROTOCOL_NOTE + INSTRUCTIONS}
            elif method == 'ping':
                result = {}
            elif method == 'tools/list':
                result = {'tools': SV.listing()}
            elif method == 'tools/call':
                name, args = params.get('name', ''), params.get('arguments') or {}
                self._threaded(mid, lambda ev: self._run_tool(name, args, ev), token, name)
                return None
            elif method == 'resources/list':
                result = {'resources': CT.resource_list()}
            elif method == 'resources/templates/list':
                result = {'resourceTemplates': CT.TEMPLATES}
            elif method == 'resources/read':
                uri = str(params.get('uri', ''))
                self._threaded(mid, lambda ev: self._read(uri, ev), token, uri)
                return None
            elif method in ('resources/subscribe', 'resources/unsubscribe'):
                result = self._subscribe(str(params.get('uri', '')), method.endswith('/subscribe'))
            elif method == 'prompts/list':
                result = {'prompts': CT.PROMPTS}
            elif method == 'prompts/get':
                result = CT.prompt_messages(params.get('name', ''), params.get('arguments') or {})
            elif method == 'completion/complete':
                ref, arg = params.get('ref') or {}, params.get('argument') or {}
                self._threaded(mid, lambda ev: {'completion': self._complete(ref, arg)})
                return None
            elif method == 'logging/setLevel':
                lv = params.get('level')
                if lv not in LEVELS:
                    return {'jsonrpc': '2.0', 'id': mid, 'error': {'code': -32602,
                                                                   'message': f'level: one of {", ".join(LEVELS)}'}}
                self.level = lv
                result = {}
            else:
                return {'jsonrpc': '2.0', 'id': mid, 'error': {'code': -32601, 'message': f'no method {method}'}}
        except CT.NotFound as e:
            return {'jsonrpc': '2.0', 'id': mid, 'error': {'code': -32002, 'message': str(e)}}
        except ValueError as e:
            return {'jsonrpc': '2.0', 'id': mid, 'error': {'code': -32602, 'message': str(e)}}
        except Exception as e:                    # noqa: BLE001 — the server keeps running
            traceback.print_exc(file=sys.stderr)
            return {'jsonrpc': '2.0', 'id': mid, 'error': {'code': -32603, 'message': f'{type(e).__name__}: {e}'}}
        return {'jsonrpc': '2.0', 'id': mid, 'result': result}

    def _complete(self, ref, arg):
        vals = CT.complete(ref, arg, self.router.lookup)
        return {'values': vals, 'total': len(vals), 'hasMore': False}

    def serve(self, inp):
        for raw in inp:
            line = raw.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except ValueError:
                self.send({'jsonrpc': '2.0', 'id': None, 'error': {'code': -32700, 'message': 'parse error'}})
                continue
            for m in (msg if isinstance(msg, list) else [msg]):
                reply = self.handle(m)
                if reply is not None:
                    self.send(reply)
        for th in list(self.threads):              # the input ended: answer what is still running first
            th.join()
        self._stop.set()


INSTRUCTIONS = (' Long work (run_batch, design_rig, optimise_rig, run_validation …) can run in the background: '
                'start_job, then get_job. Resources give the project by URI (vantage://overview, vantage://camera/F, '
                'vantage://study/power_budget …) and can be subscribed to; prompts give the common workflows. '
                'CARLA runs, downloads and the automation command never start from here: a plan the user approves '
                'in the window runs them (propose_plan).')


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog='vantage --mcp', description='Vantage MCP server (stdio)')
    ap.add_argument('--mcp', action='store_true', help=argparse.SUPPRESS)
    ap.add_argument('--mcp-project', '--project', dest='project', help='a .vantage file for when no window is open')
    ap.add_argument('--mcp-mode', '--mode', dest='mode', choices=('auto', 'live', 'file'), default='auto')
    args, _ = ap.parse_known_args(argv)
    out = sys.stdout
    sys.stdout = sys.stderr                           # only protocol messages may reach stdout
    try:
        out.reconfigure(encoding='utf-8')
        sys.stdin.reconfigure(encoding='utf-8')
    except (AttributeError, ValueError):
        pass
    Server(Router(args.project, args.mode), out).serve(sys.stdin)
    return 0
