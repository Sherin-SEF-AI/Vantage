"""A local stand-in for Anthropic's Messages API, for the Claude provider's tests: POST /v1/messages streamed as the
SDK reads it (server-sent events message_start, content_block_start / content_block_delta / content_block_stop,
message_delta, message_stop) with thinking blocks (thinking_delta, signature_delta), text, tool_use blocks
(input_json_delta), a refusal with stop_details, max_tokens, and errors with Anthropic's error body (a 429 with
retry-after, 401, 400, 529); GET /v1/models. Every request is recorded, raw bytes and headers included, so a test can
scan what left the app.

    srv = FakeAnthropic(key='sk-ant-test').start()     # srv.url -> http://127.0.0.1:PORT
    srv.script.append(srv.reply(tools=[('get_coverage', {})], thinking='Coverage first.'))
    srv.script.append(srv.reply(text='The design is fine.'))
"""
from __future__ import annotations

import itertools
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODELS = ['claude-opus-5-5', 'claude-sonnet-5-5']
_IDS = itertools.count(1)


class FakeAnthropic:
    def __init__(self, key='sk-ant-test-key', models=None):
        self.key = key
        self.models = list(models or MODELS)
        self.script = []            # responses, consumed in order; when empty: auto (a tool call first, then text)
        self.requests = []          # parsed JSON bodies of POST /v1/messages
        self.raw = []               # (headers dict, raw body bytes) of every request, GET included
        self.sent = []              # the content blocks each streamed response carried (as the SDK received them)
        self.delay = 0.0            # seconds between streamed events
        self._srv = None

    # ---------------------------------------------------------------- scripted responses
    @staticmethod
    def reply(text='', thinking='', tools=(), stop=None, chunk=16, usage=None, stop_details=None,
              signature=None, fallback=None):
        """One assistant message: optional thinking, text, then tool_use blocks [(name, input[, id])]; stop_reason
        defaults to tool_use when there are tools, else end_turn."""
        blocks = []
        if thinking:
            blocks.append({'type': 'thinking', 'thinking': thinking,
                           'signature': signature or f'sig-{abs(hash(thinking)) % 10 ** 12:012d}'})
        if fallback:
            blocks.append({'type': 'fallback', 'from': {'model': fallback[0]}, 'to': {'model': fallback[1]}})
        if text:
            blocks.append({'type': 'text', 'text': text})
        for t in tools:
            name, inp = t[0], t[1]
            tid = t[2] if len(t) > 2 else f'toolu_{next(_IDS):04d}'
            blocks.append({'type': 'tool_use', 'id': tid, 'name': name, 'input': inp})
        return {'blocks': blocks, 'stop': stop or ('tool_use' if tools else 'end_turn'), 'chunk': chunk,
                'usage': usage or {}, 'stop_details': stop_details}

    @staticmethod
    def refusal(category='cyber', explanation='This request was declined by a safety classifier.', text=''):
        return FakeAnthropic.reply(text=text, stop='refusal',
                                   stop_details={'type': 'refusal', 'category': category, 'explanation': explanation})

    @staticmethod
    def error(status, message='', etype=None, retry_after=None):
        etype = etype or {400: 'invalid_request_error', 401: 'authentication_error', 403: 'permission_error',
                          404: 'not_found_error', 429: 'rate_limit_error', 500: 'api_error',
                          529: 'overloaded_error'}.get(status, 'api_error')
        return {'status': status, 'message': message or etype, 'etype': etype, 'retry_after': retry_after}

    def _auto(self, body):
        last = body['messages'][-1]
        content = last.get('content')
        if last['role'] == 'user' and not (isinstance(content, list) and any(
                isinstance(b, dict) and b.get('type') == 'tool_result' for b in content)):
            return self.reply(tools=[('get_coverage', {})], thinking='I should read the coverage figures first.')
        return self.reply(text='**Coverage is complete.** The tools report the figures; this is the fake server.')

    # ---------------------------------------------------------------- server
    @property
    def url(self):
        return f'http://127.0.0.1:{self._srv.server_address[1]}'

    def start(self):
        fake = self

        class H(BaseHTTPRequestHandler):
            protocol_version = 'HTTP/1.1'

            def log_message(self, *a):
                pass

            def _auth(self):
                auth = self.headers.get('Authorization') or ''
                key = self.headers.get('x-api-key') or (auth[7:] if auth.startswith('Bearer ') else '')
                if key != fake.key:                       # the SDK sends the key as a bearer token (2.x)
                    self._json(401, {'type': 'error', 'error': {'type': 'authentication_error',
                                                                 'message': 'invalid x-api-key'},
                                     'request_id': 'req_fake'})
                    return False
                return True

            def _json(self, status, doc, headers=None):
                body = json.dumps(doc).encode()
                self.send_response(status)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(body)))
                self.send_header('request-id', 'req_fake')
                for k, v in (headers or {}).items():
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(body)
                self.wfile.flush()

            def do_GET(self):
                fake.raw.append((dict(self.headers.items()), b''))
                if not self._auth():
                    return
                if self.path.split('?')[0].endswith('/v1/models'):
                    self._json(200, {'data': [{'id': m, 'type': 'model', 'display_name': m,
                                               'created_at': '2026-10-01T00:00:00Z'} for m in fake.models],
                                     'has_more': False, 'first_id': fake.models[0], 'last_id': fake.models[-1]})
                else:
                    self._json(404, {'type': 'error', 'error': {'type': 'not_found_error', 'message': 'not found'}})

            def do_POST(self):
                n = int(self.headers.get('Content-Length') or 0)
                raw = self.rfile.read(n) if n else b''
                fake.raw.append((dict(self.headers.items()), raw))
                if not self._auth():
                    return
                try:
                    body = json.loads(raw or b'{}')
                except ValueError:
                    self._json(400, {'type': 'error', 'error': {'type': 'invalid_request_error',
                                                                 'message': 'not JSON'}})
                    return
                fake.requests.append(body)
                if body.get('model') not in fake.models:
                    self._json(404, {'type': 'error', 'error': {'type': 'not_found_error',
                                                                 'message': f'model: {body.get("model")}'}})
                    return
                resp = fake.script.pop(0) if fake.script else fake._auto(body)
                if 'status' in resp:
                    hdr = {'retry-after': str(resp['retry_after'])} if resp.get('retry_after') is not None else None
                    self._json(resp['status'], {'type': 'error', 'error': {'type': resp['etype'],
                                                                           'message': resp['message']},
                                                'request_id': 'req_fake'}, hdr)
                    return
                self._stream(body, resp)

            def _event(self, name, doc):
                data = f'event: {name}\ndata: {json.dumps(doc)}\n\n'.encode()
                self.wfile.write(f'{len(data):x}\r\n'.encode() + data + b'\r\n')
                self.wfile.flush()
                if fake.delay:
                    time.sleep(fake.delay)

            def _stream(self, body, resp):
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.send_header('Cache-Control', 'no-cache')
                self.send_header('Transfer-Encoding', 'chunked')
                self.send_header('request-id', 'req_fake')
                self.end_headers()
                u = {'input_tokens': 1500, 'output_tokens': 120, 'cache_read_input_tokens': 0,
                     'cache_creation_input_tokens': 2000}
                if len(fake.requests) > 1:
                    u.update(cache_read_input_tokens=2000, cache_creation_input_tokens=300)
                u.update(resp['usage'])
                msg = {'id': f'msg_{next(_IDS):04d}', 'type': 'message', 'role': 'assistant',
                       'model': body['model'], 'content': [], 'stop_reason': None, 'stop_sequence': None,
                       'usage': {k: u[k] for k in ('input_tokens', 'cache_read_input_tokens',
                                                   'cache_creation_input_tokens')} | {'output_tokens': 1}}
                self._event('message_start', {'type': 'message_start', 'message': msg})
                ch = max(1, int(resp['chunk']))
                for i, b in enumerate(resp['blocks']):
                    t = b['type']
                    if t == 'thinking':
                        self._event('content_block_start', {'type': 'content_block_start', 'index': i,
                                                            'content_block': {'type': 'thinking', 'thinking': '',
                                                                              'signature': ''}})
                        for j in range(0, len(b['thinking']), ch):
                            self._event('content_block_delta', {'type': 'content_block_delta', 'index': i, 'delta': {
                                'type': 'thinking_delta', 'thinking': b['thinking'][j:j + ch]}})
                        self._event('content_block_delta', {'type': 'content_block_delta', 'index': i, 'delta': {
                            'type': 'signature_delta', 'signature': b['signature']}})
                    elif t == 'text':
                        self._event('content_block_start', {'type': 'content_block_start', 'index': i,
                                                            'content_block': {'type': 'text', 'text': ''}})
                        for j in range(0, len(b['text']), ch):
                            self._event('content_block_delta', {'type': 'content_block_delta', 'index': i, 'delta': {
                                'type': 'text_delta', 'text': b['text'][j:j + ch]}})
                    elif t == 'tool_use':
                        self._event('content_block_start', {'type': 'content_block_start', 'index': i,
                                                            'content_block': {'type': 'tool_use', 'id': b['id'],
                                                                              'name': b['name'], 'input': {}}})
                        arg = b['input'] if isinstance(b['input'], str) else json.dumps(b['input'])
                        for j in range(0, len(arg), ch):          # the arguments arrive in pieces
                            self._event('content_block_delta', {'type': 'content_block_delta', 'index': i, 'delta': {
                                'type': 'input_json_delta', 'partial_json': arg[j:j + ch]}})
                    else:                                         # a fallback marker or another block, whole
                        self._event('content_block_start', {'type': 'content_block_start', 'index': i,
                                                            'content_block': b})
                    self._event('content_block_stop', {'type': 'content_block_stop', 'index': i})
                delta = {'stop_reason': resp['stop'], 'stop_sequence': None}
                if resp.get('stop_details') is not None:
                    delta['stop_details'] = resp['stop_details']
                self._event('message_delta', {'type': 'message_delta', 'delta': delta,
                                              'usage': {'output_tokens': u['output_tokens']}})
                self._event('message_stop', {'type': 'message_stop'})
                self.wfile.write(b'0\r\n\r\n')
                self.wfile.flush()
                fake.sent.append([dict(b) for b in resp['blocks']])

        self._srv = ThreadingHTTPServer(('127.0.0.1', 0), H)
        self._srv.daemon_threads = True
        threading.Thread(target=self._srv.serve_forever, daemon=True).start()
        return self

    def stop(self):
        if self._srv is not None:
            self._srv.shutdown()
            self._srv.server_close()
