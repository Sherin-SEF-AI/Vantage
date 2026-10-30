"""Claude as the assistant's model, through Anthropic's official Python SDK (the optional `anthropic` package; without
it Preferences says how to install it and the Groq provider works as before).

How the requests are made, as Anthropic's documentation for these models says:
- Models: claude-opus-5-5 (default) and claude-sonnet-5-5, exact IDs. Thinking is adaptive (Claude Opus 5.5 cannot
  turn it off); output_config.effort is set explicitly, high on a Copilot (planning) turn and medium for a plain
  question. budget_tokens, temperature, top_p and top_k are never sent (they are 400s on these models).
- Streaming with client.beta.messages.stream(), every tool declared with eager_input_streaming, so the SDK's tolerant
  parser can hand over a partial or silently truncated input: the loop validates every parsed input against its
  schema before running it (agent.BaseSession), and checks the stop_reason first: max_tokens with a tool_use runs no
  tool, a refusal runs nothing and shows stop_details plainly.
- Forced tool_choice is a 400 on these models: tool_choice auto, with strict: true on every tool whose schema is
  closed (every object lists its properties; additionalProperties: false is added). Parallel tool calls come back as
  all their tool_result blocks in one user message, failures with is_error.
- Prompt caching: one frozen system prompt (nothing volatile: the context of a turn goes in its user message), the
  tools in a fixed order (sorted by name), a cache_control breakpoint on the system prompt and automatic caching of
  the conversation's tail; the usage line shows cache_read_input_tokens.
- Append-only history: the assistant's response.content goes back unchanged, thinking blocks and their signatures
  included (preserved thinking: editing an earlier turn would invalidate them); nothing earlier is ever shortened.
  Long conversations use the server's compaction (beta compact-2026-01-12, a compact_20260112 edit), whose compaction
  block is passed back like any other. A failed turn is removed from the end, which leaves the earlier prefix as it
  was.
- Refusals: server-side fallback, fallbacks: "default" under the beta server-side-fallback-2026-07-01 (a setting, on
  by default): a declined request is re-run on Anthropic's recommended model for that refusal category; a response
  that fell back mid-output is echoed back without the thinking and tool_use blocks before its last fallback block,
  as documented.
- Pictures: a tool that draws (look) answers with its text and the picture as an image block (base64 PNG or JPEG,
  at most 1280 px on the long edge and about 1.2 megapixels) inside the same tool_result. The inspector shows the
  request with the image data shortened.
- The design log (assistant/memory.py) goes in as a mid-conversation system message after the user's message, only
  when it changed since the last one sent: the frozen system prompt and the cached history stay as they were.
- Errors: the SDK's typed exceptions, most specific first: rate limit (with retry-after), authentication, permission,
  not found, bad request, overloaded (529), other status errors, connection. The SDK's own retries (max_retries 2)
  stay at their default.
"""
from __future__ import annotations

import copy
import json
import os

from . import providers as PR
from . import tools as T
from .agent import WRAP_UP, BaseSession, Call, Reply, context_note, strip_handover
from .client import ApiError

try:
    import anthropic
except ImportError:                     # optional: Groq still works
    anthropic = None

BETA_COMPACT = 'compact-2026-01-12'
BETA_FALLBACK = 'server-side-fallback-2026-07-01'
COMPACT_EDIT = {'type': 'compact_20260112'}
MAX_TOKENS = 32000                      # thinking counts towards it, so it is sized for thinking and the reply
EFFORT_PLANNING, EFFORT_QUESTION = 'high', 'medium'
JSON_RETRIES = 2
EMPTY_LOG = 'DESIGN LOG\nThe design log is now empty: earlier entries no longer apply.'


def base_url() -> str | None:
    """VANTAGE_CLAUDE_URL points the SDK elsewhere (a test server); else the SDK's own default."""
    return (os.environ.get('VANTAGE_CLAUDE_URL') or '').rstrip('/') or None


class ClaudeUnavailable(ApiError):
    def __init__(self):
        super().__init__('unavailable', PR.INSTALL)


# ------------------------------------------------------------------ tools as Claude takes them
def _closed(schema) -> bool:
    """Every object in the schema lists its properties (strict tool use needs additionalProperties: false)."""
    if not isinstance(schema, dict):
        return True
    t = schema.get('type')
    if t == 'object':
        props = schema.get('properties')
        if not isinstance(props, dict):          # a free-form object (e.g. optimise_camera's ranges)
            return False
        return all(_closed(v) for v in props.values())
    if t == 'array':
        return _closed(schema.get('items', {}))
    return True


def _strict_schema(schema):
    s = copy.deepcopy(schema)

    def walk(o):
        if isinstance(o, dict):
            if o.get('type') == 'object':
                o.setdefault('properties', {})
                o['additionalProperties'] = False
                for v in o['properties'].values():
                    walk(v)
            elif o.get('type') == 'array' and isinstance(o.get('items'), dict):
                walk(o['items'])
    walk(s)
    return s


def tool_defs() -> list:
    """The tools in a fixed order (sorted by name), each with eager_input_streaming; strict where the schema is
    closed."""
    out = []
    for t in sorted(T.TOOLS, key=lambda x: x.name):
        schema = T.input_schema(t)
        closed = _closed(schema)
        d = {'name': t.name, 'description': t.description,
             'input_schema': _strict_schema(schema) if closed else schema, 'eager_input_streaming': True}
        if closed:
            d['strict'] = True
        out.append(d)
    return out


def _plain(o):
    """The SDK's content blocks as plain JSON (for the inspector and the tests)."""
    if hasattr(o, 'to_dict'):
        return o.to_dict()
    if isinstance(o, dict):
        return {k: _plain(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_plain(v) for v in o]
    return o


def elide_images(o):
    """A copy of a request with every image's base64 data replaced by its size (for the inspector)."""
    if isinstance(o, dict):
        if o.get('type') == 'base64' and isinstance(o.get('data'), str) and len(o['data']) > 200:
            return dict(o, data=f'<{o.get("media_type", "image")}, {len(o["data"]) * 3 // 4 / 1024:.0f} kB, '
                                f'base64, shortened here>')
        return {k: elide_images(v) for k, v in o.items()}
    if isinstance(o, list):
        return [elide_images(v) for v in o]
    return o


def image_block(pic) -> dict:
    return {'type': 'image', 'source': {'type': 'base64', 'media_type': pic.media_type, 'data': pic.b64()}}


def echo_content(content) -> list:
    """The assistant content to pass back: unchanged, except after a mid-output fallback, where the thinking,
    redacted_thinking and tool_use blocks before the last fallback block are left out (Anthropic's echo rule)."""
    blocks = list(content or [])
    idx = [i for i, b in enumerate(blocks) if getattr(b, 'type', None) == 'fallback']
    if not idx:
        return blocks
    last = idx[-1]
    return [b for i, b in enumerate(blocks) if i >= last or getattr(b, 'type', None) not in
            ('thinking', 'redacted_thinking', 'tool_use', 'server_tool_use')]


def map_error(e) -> ApiError:
    """The SDK's exception as an error the panel shows, most specific class first."""
    A = anthropic
    msg = getattr(e, 'message', '') or str(e)
    if isinstance(e, A.RateLimitError):
        retry = None
        try:
            ra = e.response.headers.get('retry-after')
            retry = float(ra) if ra else None
        except (TypeError, ValueError, AttributeError):
            retry = None
        return ApiError('rate', 'Anthropic is rate-limiting this key' + (f'; try again in {retry:.0f} s.' if retry
                                                                          else '.'), 429, retry)
    if isinstance(e, A.AuthenticationError):
        return ApiError('auth', 'Anthropic did not accept the API key. Check it in Preferences ▸ Assistant.', 401)
    if isinstance(e, A.PermissionDeniedError):
        return ApiError('permission', msg or 'This key may not do that.', 403)
    if isinstance(e, A.NotFoundError):
        return ApiError('model', msg or 'That model is not available to this key.', 404)
    if isinstance(e, A.BadRequestError):
        return ApiError('bad_request', msg or 'The request was refused (400).', 400, detail=str(getattr(e, 'type', '')))
    if isinstance(e, A.OverloadedError):
        return ApiError('overloaded', 'Anthropic\'s API is overloaded at the moment; try again shortly.', 529)
    if isinstance(e, A.APIStatusError):
        st = int(getattr(e, 'status_code', 0) or 0)
        if st == 402:
            return ApiError('billing', 'Anthropic reports a billing problem with this key\'s account.', 402)
        if st == 413:
            return ApiError('bad_request', 'The request was too large for the API.', 413)
        if st >= 500:
            return ApiError('server', msg or f'Anthropic returned an error ({st}). Try again in a moment.', st)
        return ApiError('bad_request', msg or f'The request was refused ({st}).', st)
    if isinstance(e, A.APIConnectionError):
        return ApiError('network', 'Could not reach Anthropic\'s API: check the connection.')
    return ApiError('server', f'{type(e).__name__}: {msg}')


class ClaudeSession(BaseSession):
    """A conversation with Claude. self.messages is append-only (see the module doc)."""
    provider = PR.CLAUDE

    def __init__(self, key: str, settings, url: str | None = None):
        if anthropic is None:
            raise ClaudeUnavailable()
        if not (key or '').strip():             # never let the SDK look for a key of its own (environment, profile)
            raise ApiError('auth', 'No Anthropic API key: add one in Preferences ▸ Assistant.')
        super().__init__(settings)
        self.client = anthropic.Anthropic(api_key=key.strip(), base_url=url or base_url(),
                                          timeout=anthropic.Timeout(300.0, connect=15.0))
        from .agent import SYSTEM
        self.system = [{'type': 'text', 'text': SYSTEM, 'cache_control': {'type': 'ephemeral'}}]
        self.tools = tool_defs()
        self.messages: list = []

    def reset(self):
        self.messages = []
        self.usage = PR.Usage(provider=self.provider, model=self.settings.model)

    def _mark(self):
        return len(self.messages)

    def _rollback(self, mark):
        del self.messages[mark:]            # the tail only: every earlier message stays byte for byte

    def _begin(self, text, tc):
        from . import memory as M
        self.messages.append({'role': 'user', 'content': [{'type': 'text',
                                                           'text': f'{text}\n\n{context_note(tc, agent_text=True)}'}]})
        log = M.prompt_block(tc.project)
        last = self.design_log_sent()
        if (log or last) and (log or EMPTY_LOG) != last:
            self.messages.append({'role': 'system', 'content': log or EMPTY_LOG})

    def design_log_sent(self) -> str:
        """The design log as the model last received it ('' when never sent)."""
        return next((m['content'] for m in reversed(self.messages) if m.get('role') == 'system'), '')

    def request_kwargs(self, wrap=False) -> dict:
        st = self.settings
        betas = [BETA_COMPACT] + ([BETA_FALLBACK] if st.fallbacks else [])
        kw = {'model': st.model, 'max_tokens': MAX_TOKENS, 'system': self.system, 'tools': self.tools,
              'messages': self.messages, 'tool_choice': {'type': 'none' if wrap else 'auto'},
              'thinking': {'type': 'adaptive', 'display': 'summarized'} if st.show_reasoning else {'type': 'adaptive'},
              'output_config': {'effort': EFFORT_PLANNING if st.planning else EFFORT_QUESTION},
              'cache_control': {'type': 'ephemeral'}, 'context_management': {'edits': [dict(COMPACT_EDIT)]},
              'betas': betas}
        if st.fallbacks:
            kw['fallbacks'] = 'default'
        return kw

    def _request(self, wrap, emit, cancel) -> Reply:
        if wrap and self.messages and self.messages[-1]['role'] == 'user' and \
                not any(b.get('text') == WRAP_UP for b in self.messages[-1]['content'] if isinstance(b, dict)):
            # appended once, after the tool results of a message not sent yet: nothing sent before changes
            self.messages[-1]['content'].append({'type': 'text', 'text': WRAP_UP})
        kw = self.request_kwargs(wrap)
        self.last_payload = elide_images(_plain({k: v for k, v in kw.items()}))
        try:
            with self.client.beta.messages.stream(**kw) as stream:
                for ev in stream:
                    if cancel.is_set():
                        raise ApiError('cancelled', 'Stopped.')
                    t = getattr(ev, 'type', '')
                    if t == 'thinking' and ev.thinking:
                        emit_event(emit, 'reasoning', ev.thinking)
                    elif t == 'text' and ev.text:
                        emit_event(emit, 'content', ev.text)
                msg = stream.get_final_message()
        except ApiError:
            raise
        except ValueError as e:             # tool input JSON the SDK could not parse at all: no tool_use id to answer
            raise ApiError('invalid_json', f'The model wrote a tool input that is not JSON ({e}).') from None
        except anthropic.AnthropicError as e:
            raise map_error(e) from None
        u = msg.usage
        self.usage.add_claude(getattr(msg, 'model', None) or self.settings.model, u.input_tokens, u.output_tokens,
                              getattr(u, 'cache_creation_input_tokens', 0) or 0,
                              getattr(u, 'cache_read_input_tokens', 0) or 0)
        emit_event(emit, 'usage', self.usage)
        content = echo_content(msg.content)
        text = ''.join(b.text for b in content if getattr(b, 'type', None) == 'text')
        calls = []
        for b in content:
            if getattr(b, 'type', None) == 'tool_use':
                inp = b.input
                calls.append(Call(b.id, b.name, inp if isinstance(inp, dict) else None,
                                  json.dumps(inp, default=str) if not isinstance(inp, str) else inp))
        refusal = None
        if msg.stop_reason == 'refusal':
            sd = getattr(msg, 'stop_details', None)
            refusal = {'category': getattr(sd, 'category', None), 'explanation': getattr(sd, 'explanation', None)}
        return Reply(text, calls, msg.stop_reason or '', refusal, content)

    def _append_assistant(self, reply, wrap):
        self.messages.append({'role': 'assistant', 'content': list(reply.native or [])})
        if wrap:                            # told not to call tools: any call it made still needs its result
            pending = [b for b in reply.native or [] if getattr(b, 'type', None) == 'tool_use']
            if pending:
                self.messages.append({'role': 'user', 'content': [
                    {'type': 'tool_result', 'tool_use_id': b.id, 'is_error': True,
                     'content': 'Not run: the budget for this request is used up.'} for b in pending]})

    def _append_results(self, results, wrap):
        blocks = []
        for call, result, ok, *rest in results:
            pics = rest[0] if rest else []
            b = {'type': 'tool_result', 'tool_use_id': call.id,
                 'content': [{'type': 'text', 'text': result}] + [image_block(p) for p in pics] if pics else result}
            if not ok:
                b['is_error'] = True
            blocks.append(b)
        self.messages.append({'role': 'user', 'content': blocks})

    def _retry(self, e, emit, cancel, retries) -> bool:
        if e.kind == 'invalid_json' and retries.get('json', 0) < JSON_RETRIES:
            retries['json'] = retries.get('json', 0) + 1
            emit_event(emit, 'status', 'The model wrote a tool input that was not JSON; asking again…')
            return True
        return super()._retry(e, emit, cancel, retries)

    def history(self) -> list:
        """The conversation as plain JSON (what the next request sends)."""
        return _plain(self.messages)

    def transcript_markdown(self) -> str:
        lines = []
        for m in self.messages:
            blocks = m['content'] if isinstance(m['content'], list) else []
            if m['role'] == 'user':
                texts = [b['text'] for b in blocks if isinstance(b, dict) and b.get('type') == 'text'
                         and b.get('text') != WRAP_UP]
                if texts:
                    lines += ['### You', '', strip_handover(texts[0].rsplit('\n\n[Vantage context:', 1)[0]), '']
            else:
                texts = [b.text for b in blocks if getattr(b, 'type', None) == 'text']
                if texts:
                    lines += ['### Assistant', '', ''.join(texts), '']
        return '\n'.join(lines)


def emit_event(emit, kind, value):
    from .agent import Event
    if kind == 'usage':
        emit(Event('usage', {'usage': value}))
    elif kind == 'status':
        emit(Event('status', {'text': value}))
    else:
        emit(Event(kind, {'text': value}))
