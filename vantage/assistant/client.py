"""A small client for an OpenAI-compatible chat API (Groq by default): model list, streamed chat completions with tool
calls and reasoning, and errors a user can act on. Standard library only, so it works the same in the bundles."""
from __future__ import annotations

import json
import os
import ssl
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass, field

from .. import __version__

GROQ_URL = 'https://api.groq.com/openai/v1'
KEYS_URL = 'https://console.groq.com/keys'
DEFAULT_MODEL = 'openai/gpt-oss-120b'


def base_url() -> str:
    """The API root; VANTAGE_ASSISTANT_URL points it elsewhere (a test server, a proxy)."""
    return (os.environ.get('VANTAGE_ASSISTANT_URL') or GROQ_URL).rstrip('/')


# ------------------------------------------------------------------ models
@dataclass(frozen=True)
class ModelInfo:
    id: str
    label: str
    family: str                   # 'gpt-oss' | 'qwen' | 'minimax' | 'muse' | 'llama' | ''
    efforts: tuple = ()           # reasoning_effort values the model takes
    default_effort: str = ''
    rank: int = 50                # lower first in the picker
    note: str = ''


# What Groq's documentation says about each model (October 2026). Unknown models still work: they get no reasoning
# parameters, and the picker lists them after these.
KNOWN = {
    'openai/gpt-oss-120b': ModelInfo('openai/gpt-oss-120b', 'GPT-OSS 120B', 'gpt-oss', ('low', 'medium', 'high'),
                                     'medium', 0, 'Most capable · reasoning · production'),
    'qwen/qwen3.8-27b': ModelInfo('qwen/qwen3.8-27b', 'Qwen 3.8 27B', 'qwen', ('none', 'low', 'medium', 'high'),
                                  'medium', 10, 'Reasoning · preview'),
    'minimaxai/minimax-m2.7': ModelInfo('minimaxai/minimax-m2.7', 'MiniMax M2.7', 'minimax', (), '', 12,
                                        'Reasoning · preview'),
    'meta-models/muse-spark-open': ModelInfo('meta-models/muse-spark-open', 'Muse Spark Open', 'muse',
                                             ('low', 'medium', 'high', 'xhigh'), 'medium', 14, 'Reasoning · preview'),
    'openai/gpt-oss-20b': ModelInfo('openai/gpt-oss-20b', 'GPT-OSS 20B', 'gpt-oss', ('low', 'medium', 'high'),
                                    'medium', 20, 'Faster · reasoning · production'),
    'llama-3.3-70b-versatile': ModelInfo('llama-3.3-70b-versatile', 'Llama 3.3 70B', 'llama', (), '', 30,
                                         'No reasoning · production'),
    'llama-3.1-8b-instant': ModelInfo('llama-3.1-8b-instant', 'Llama 3.1 8B', 'llama', (), '', 40,
                                      'Fastest · no reasoning · production'),
}
_NOT_CHAT = ('whisper', 'orpheus', 'guard', 'tts', 'playai', 'distil-whisper', 'embed')


def model_info(model_id: str) -> ModelInfo:
    if model_id in KNOWN:
        return KNOWN[model_id]
    fam = next((f for f, keys in (('gpt-oss', ('gpt-oss',)), ('qwen', ('qwen',)), ('llama', ('llama',)))
                if any(k in model_id for k in keys)), '')
    name = model_id.split('/')[-1]
    return ModelInfo(model_id, name, fam if fam == 'llama' else '', (), '', 60)


def is_chat_model(model_id: str) -> bool:
    return not any(k in model_id.lower() for k in _NOT_CHAT)


def reasoning_params(model_id: str, effort: str, show: bool) -> dict:
    """Request parameters for a model's reasoning, as Groq documents them per family."""
    info = model_info(model_id)
    out = {}
    if info.family == 'gpt-oss':
        out['reasoning_effort'] = effort if effort in info.efforts else info.default_effort
        out['include_reasoning'] = bool(show)
    elif info.family in ('qwen', 'minimax', 'muse'):
        out['reasoning_format'] = 'parsed' if show else 'hidden'     # 'raw' is refused together with tools
        if info.efforts:
            out['reasoning_effort'] = effort if effort in info.efforts else info.default_effort
    return out


# ------------------------------------------------------------------ errors
class ApiError(Exception):
    """kind: auth | permission | rate | model | bad_request | tool_use_failed | server | network | cancelled"""

    def __init__(self, kind: str, message: str, status: int = 0, retry_after: float | None = None, detail: str = ''):
        super().__init__(message)
        self.kind, self.message, self.status, self.retry_after, self.detail = kind, message, status, retry_after, detail


def _error_from(status: int, body: bytes, headers) -> ApiError:
    msg, code, detail = '', '', ''
    try:
        doc = json.loads(body.decode('utf-8', 'replace'))
        err = doc.get('error', doc) if isinstance(doc, dict) else {}
        msg = str(err.get('message') or '')
        code = str(err.get('code') or err.get('type') or '')
        detail = str(err.get('failed_generation') or '')
    except Exception:
        msg = body.decode('utf-8', 'replace')[:300]
    retry = None
    try:
        ra = headers.get('retry-after') if headers is not None else None
        retry = float(ra) if ra else None
    except (TypeError, ValueError):
        retry = None
    if status == 401:
        return ApiError('auth', 'The API key was not accepted. Check it in Preferences ▸ Assistant.', status)
    if status == 403:
        return ApiError('permission', msg or 'This key may not use that model.', status)
    if status == 429:
        return ApiError('rate', 'Groq is rate-limiting this key' + (f'; try again in {retry:.0f} s.' if retry else '.'),
                        status, retry)
    if status == 404 or code == 'model_not_found' or 'model' in code and 'not' in code:
        return ApiError('model', msg or 'That model is not available to this key.', status)
    if code == 'tool_use_failed':
        return ApiError('tool_use_failed', msg or 'The model wrote an invalid tool call.', status, detail=detail)
    if 400 <= status < 500:
        return ApiError('bad_request', msg or f'The request was refused ({status}).', status, detail=code)
    return ApiError('server', msg or f'Groq returned an error ({status}). Try again in a moment.', status)


# ------------------------------------------------------------------ transport
def _ssl_context():
    ctx = ssl.create_default_context()
    try:                            # bundles do not always find the system certificates; certifi does
        import certifi
        ctx.load_verify_locations(certifi.where())
    except Exception:
        pass
    return ctx


@dataclass
class Usage:
    prompt: int = 0
    completion: int = 0
    reasoning: int = 0
    seconds: float = 0.0

    def add(self, u: dict):
        if not u:
            return
        self.prompt += int(u.get('prompt_tokens') or 0)
        self.completion += int(u.get('completion_tokens') or 0)
        det = u.get('completion_tokens_details') or {}
        self.reasoning += int(det.get('reasoning_tokens') or 0)
        self.seconds += float(u.get('total_time') or 0.0)


@dataclass
class Delta:
    """One piece of a streamed answer."""
    content: str = ''
    reasoning: str = ''
    tool_calls: list = field(default_factory=list)     # raw OpenAI-style tool_call deltas (index, id, function...)
    finish: str | None = None
    usage: dict | None = None


class Client:
    def __init__(self, api_key: str, url: str | None = None, timeout: float = 90.0):
        self.key = (api_key or '').strip()
        self.url = (url or base_url()).rstrip('/')
        self.timeout = timeout
        self._ssl = _ssl_context() if self.url.startswith('https') else None

    def _request(self, path, payload=None, stream=False):
        data = json.dumps(payload).encode('utf-8') if payload is not None else None
        req = urllib.request.Request(self.url + path, data=data, method='POST' if data is not None else 'GET')
        req.add_header('Authorization', f'Bearer {self.key}')
        req.add_header('User-Agent', f'Vantage/{__version__}')
        req.add_header('Accept', 'text/event-stream' if stream else 'application/json')
        if data is not None:
            req.add_header('Content-Type', 'application/json')
        try:
            return urllib.request.urlopen(req, timeout=self.timeout, context=self._ssl)
        except urllib.error.HTTPError as e:
            raise _error_from(e.code, e.read() or b'', e.headers) from None
        except urllib.error.URLError as e:
            reason = getattr(e, 'reason', e)
            raise ApiError('network', f'Could not reach {self.url.split("/")[2]}: {reason}. Check the connection.') from None
        except (TimeoutError, OSError) as e:
            raise ApiError('network', f'The connection failed: {e}.') from None

    def models(self) -> list[str]:
        """Chat models this key may use, most capable first."""
        with self._request('/models') as r:
            doc = json.loads(r.read().decode('utf-8'))
        ids = [m.get('id', '') for m in doc.get('data', []) if m.get('id') and m.get('active', True) is not False]
        ids = [i for i in ids if is_chat_model(i)]
        return sorted(ids, key=lambda i: (model_info(i).rank, i))

    def stream(self, payload: dict, cancel: threading.Event | None = None):
        """Yield Delta objects for a streamed chat completion (payload without 'stream')."""
        body = dict(payload, stream=True)
        resp = self._request('/chat/completions', body, stream=True)
        try:
            for raw in resp:
                if cancel is not None and cancel.is_set():
                    raise ApiError('cancelled', 'Stopped.')
                line = raw.decode('utf-8', 'replace').strip()
                if not line or line.startswith(':') or not line.startswith('data:'):
                    continue
                data = line[5:].strip()
                if data == '[DONE]':
                    break
                try:
                    chunk = json.loads(data)
                except ValueError:
                    continue
                if 'error' in chunk:
                    err = chunk['error'] if isinstance(chunk['error'], dict) else {'message': str(chunk['error'])}
                    raise _error_from(int(err.get('status') or 400), json.dumps({'error': err}).encode(), None)
                usage = chunk.get('usage') or (chunk.get('x_groq') or {}).get('usage')
                for ch in chunk.get('choices') or [{}]:
                    d = ch.get('delta') or {}
                    yield Delta(content=d.get('content') or '', reasoning=d.get('reasoning') or '',
                                tool_calls=d.get('tool_calls') or [], finish=ch.get('finish_reason'), usage=usage)
                    usage = None
                if not chunk.get('choices') and usage:
                    yield Delta(usage=usage)
        except (TimeoutError, OSError) as e:
            raise ApiError('network', f'The connection dropped: {e}.') from None
        finally:
            resp.close()
