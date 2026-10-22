"""A local stand-in for Groq's OpenAI-compatible API, for the assistant tests: /models and streamed
/chat/completions with reasoning, content and tool calls, scripted per request; every request is recorded.

    srv = FakeGroq(key='test-key').start()       # srv.url -> http://127.0.0.1:PORT/openai/v1
    srv.script.append(srv.tool_call('get_coverage', {}))
    srv.script.append(srv.answer('The design is fine.', reasoning='Looked at coverage.'))
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODELS = ['openai/gpt-oss-120b', 'openai/gpt-oss-20b', 'qwen/qwen3.8-27b', 'llama-3.1-8b-instant', 'whisper-large-v3',
          'meta-llama/llama-prompt-guard-2-86m']


class FakeGroq:
    def __init__(self, key='test-key', models=None):
        self.key = key
        self.models = list(models or MODELS)
        self.script = []            # responses, consumed in order; when empty: auto (tool call first, then an answer)
        self.requests = []          # payloads of /chat/completions
        self.delay = 0.0            # seconds between streamed chunks
        self._srv = None

    # ---------------------------------------------------------------- scripted responses
    @staticmethod
    def answer(text, reasoning='', chunk=12):
        chunks = []
        for i in range(0, len(reasoning), chunk):
            chunks.append({'reasoning': reasoning[i:i + chunk]})
        for i in range(0, len(text), chunk):
            chunks.append({'content': text[i:i + chunk]})
        return {'chunks': chunks, 'finish': 'stop'}

    @staticmethod
    def tool_call(name, args, call_id=None, reasoning=''):
        arg = json.dumps(args)
        cid = call_id or f'call_{name}'
        chunks = [{'reasoning': reasoning}] if reasoning else []
        chunks.append({'tool_calls': [{'index': 0, 'id': cid, 'type': 'function',
                                       'function': {'name': name, 'arguments': ''}}]})
        for i in range(0, len(arg), 10):            # arguments arrive in pieces
            chunks.append({'tool_calls': [{'index': 0, 'function': {'arguments': arg[i:i + 10]}}]})
        return {'chunks': chunks, 'finish': 'tool_calls'}

    @staticmethod
    def error(status, message='', code='', retry_after=None, failed_generation=None):
        return {'status': status, 'message': message, 'code': code, 'retry_after': retry_after,
                'failed_generation': failed_generation}

    def _auto(self, payload):
        last = payload['messages'][-1]
        if last['role'] == 'user':
            return self.tool_call('get_coverage', {}, reasoning='I should read the coverage figures first.')
        return self.answer('**Coverage is complete.** The tools report the figures; this is the fake server.')

    # ---------------------------------------------------------------- server
    @property
    def url(self):
        return f'http://127.0.0.1:{self._srv.server_address[1]}/openai/v1'

    def start(self):
        fake = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _auth(self):
                if self.headers.get('Authorization') != f'Bearer {fake.key}':
                    self._json(401, {'error': {'message': 'Invalid API Key', 'type': 'invalid_request_error',
                                               'code': 'invalid_api_key'}})
                    return False
                return True

            def _json(self, status, doc, headers=None):
                body = json.dumps(doc).encode()
                self.send_response(status)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(body)))
                for k, v in (headers or {}).items():
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if not self._auth():
                    return
                if self.path.endswith('/models'):
                    self._json(200, {'object': 'list', 'data': [{'id': m, 'object': 'model', 'active': True}
                                                                for m in fake.models]})
                else:
                    self._json(404, {'error': {'message': 'not found'}})

            def do_POST(self):
                n = int(self.headers.get('Content-Length') or 0)
                payload = json.loads(self.rfile.read(n) or b'{}')
                if not self._auth():
                    return
                fake.requests.append(payload)
                if payload.get('model') not in fake.models:
                    self._json(404, {'error': {'message': f'The model `{payload.get("model")}` does not exist',
                                               'code': 'model_not_found'}})
                    return
                resp = fake.script.pop(0) if fake.script else fake._auto(payload)
                if 'status' in resp:
                    err = {'message': resp['message'] or 'error', 'type': 'invalid_request_error',
                           'code': resp['code'] or None}
                    if resp.get('failed_generation'):
                        err['failed_generation'] = resp['failed_generation']
                    hdr = {'retry-after': str(resp['retry_after'])} if resp.get('retry_after') else None
                    self._json(resp['status'], {'error': err}, hdr)
                    return
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.end_headers()
                import time
                for d in resp['chunks']:
                    self._sse({'id': 'x', 'object': 'chat.completion.chunk', 'model': payload['model'],
                               'choices': [{'index': 0, 'delta': d, 'finish_reason': None}]})
                    if fake.delay:
                        time.sleep(fake.delay)
                self._sse({'id': 'x', 'object': 'chat.completion.chunk', 'model': payload['model'],
                           'choices': [{'index': 0, 'delta': {}, 'finish_reason': resp['finish']}],
                           'x_groq': {'usage': {'prompt_tokens': 1200, 'completion_tokens': 85,
                                                'completion_tokens_details': {'reasoning_tokens': 20},
                                                'total_time': 0.21}}})
                self.wfile.write(b'data: [DONE]\n\n')
                self.wfile.flush()

            def _sse(self, doc):
                self.wfile.write(b'data: ' + json.dumps(doc).encode() + b'\n\n')
                self.wfile.flush()

        self._srv = ThreadingHTTPServer(('127.0.0.1', 0), H)
        threading.Thread(target=self._srv.serve_forever, daemon=True).start()
        return self

    def stop(self):
        if self._srv is not None:
            self._srv.shutdown()
            self._srv.server_close()
