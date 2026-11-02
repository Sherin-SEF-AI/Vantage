"""The Vantage MCP server: Claude (Claude Code, Claude Desktop or any MCP client) reads and changes rig designs.

    vantage --mcp [--mcp-project FILE.vantage] [--mcp-mode auto|live|file]
    python -m vantage.mcp ...

It speaks MCP over stdio (JSON-RPC 2.0, one message per line) with the Python standard library only. Each tool call
goes to the Vantage window that is open (live: the change shows there as an undo step "Claude · …"), or, with no
window, to the project file (file mode: analysed here, saved after each change, the original kept as .bak).

Register it with Claude Code once:

    claude mcp add --scope user vantage -- vantage --mcp
    (from source: claude mcp add --scope user vantage -- python3 /path/to/run_vantage.py --mcp)"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback

from .. import __version__
from . import service as SV
from .session import FileSession, LinkError, LiveLink, same_file, windows

VERSIONS = ('2025-11-25', '2025-06-18', '2025-03-26', '2024-11-05')


class Router:
    """Which session a call goes to: the open window (newest, or the one with --mcp-project open), else the file."""

    def __init__(self, project=None, mode='auto'):
        self.mode = mode
        self.project = project
        self.file = FileSession()
        self.link = None
        self.sticky_file = mode == 'file'
        self._tried = False

    def session(self):
        if not self.sticky_file:
            ws = windows()
            if ws:
                here = os.environ.get('VANTAGE_MCP_PID', '')      # started in a window's Terminal panel: that window
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

    def call(self, name, args) -> dict:
        if name == 'open_project':
            out = self.open(args if isinstance(args, dict) else {})
            return SV.text(out, 'error' in out)
        try:
            s = self.session()
        except LinkError as e:
            return SV.text({'error': str(e)}, True)
        if isinstance(s, LiveLink):
            try:
                return s.request('call', name=name, args=args or {})
            except LinkError as e:
                self.link = None
                return SV.text({'error': f'{e}. Call the tool again to use whatever is open now.'}, True)
        return SV.call(s, name, args)


class Server:
    def __init__(self, router: Router, out):
        self.router = router
        self.out = out

    def send(self, doc):
        self.out.write(json.dumps(doc, ensure_ascii=False, default=str) + '\n')
        self.out.flush()

    def handle(self, msg):
        if not isinstance(msg, dict) or msg.get('jsonrpc') != '2.0':
            return {'jsonrpc': '2.0', 'id': None, 'error': {'code': -32600, 'message': 'invalid request'}}
        mid, method, params = msg.get('id'), msg.get('method'), msg.get('params') or {}
        if mid is None:                               # a notification (initialized, cancelled …): nothing to answer
            return None
        try:
            if method == 'initialize':
                want = params.get('protocolVersion')
                result = {'protocolVersion': want if want in VERSIONS else VERSIONS[0],
                          'capabilities': {'tools': {'listChanged': False}},
                          'serverInfo': {'name': 'vantage', 'title': 'Vantage', 'version': __version__},
                          'instructions': SV.PROTOCOL_NOTE}
            elif method == 'ping':
                result = {}
            elif method == 'tools/list':
                result = {'tools': SV.listing()}
            elif method == 'tools/call':
                result = self.router.call(params.get('name', ''), params.get('arguments') or {})
            elif method in ('resources/list', 'prompts/list'):
                result = {method.split('/')[0]: []}
            else:
                return {'jsonrpc': '2.0', 'id': mid, 'error': {'code': -32601, 'message': f'no method {method}'}}
        except Exception as e:                        # noqa: BLE001 — the server keeps running
            traceback.print_exc(file=sys.stderr)
            return {'jsonrpc': '2.0', 'id': mid, 'error': {'code': -32603, 'message': f'{type(e).__name__}: {e}'}}
        return {'jsonrpc': '2.0', 'id': mid, 'result': result}

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
            if isinstance(msg, list):
                replies = [r for r in (self.handle(m) for m in msg) if r is not None]
                if replies:
                    self.send(replies)
                continue
            reply = self.handle(msg)
            if reply is not None:
                self.send(reply)


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
