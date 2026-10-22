"""One conversation with the model: streams the answer, runs the tools it calls, and reports everything as events
(called on the worker thread that runs the turn; the UI forwards them to the Qt thread)."""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field

from . import tools as T
from .client import ApiError, Client, Usage, reasoning_params

MAX_ROUNDS = 12                 # tool rounds per question
AGENT_ROUNDS = 30               # ... when working towards a goal
AGENT_SECONDS = 240.0           # time budget for a goal; then the model writes up what it has
MAX_HISTORY_CHARS = 240_000     # older tool results are shortened past this

SYSTEM = """You are the design assistant built into Vantage, a desktop app for planning surround-view (360°) camera \
rigs on vehicles. You help an automotive camera engineer understand and improve the design that is open in the app.

How to work:
- Get every number from the tools. Never guess or invent values. If a tool says something has not been computed, \
say so and say where in Vantage to run it.
- Call the tools you need before answering; a few targeted calls are better than many. Explain *why* with the \
per-camera reasons the tools give (outside the lens field of view, outside the image frame, hidden by the body, low \
pixel density).
- To suggest a change, check it with simulate_changes, then offer it with propose_changes. You cannot change the \
design yourself: the user sees the proposal with its predicted effect and applies it or not. Prefer the smallest \
change that solves the problem, and keep left/right pairs symmetric unless asked otherwise.
- Units: metres, degrees, m², px/cm. Frame: ISO 8855, x forward, y left, z up, origin at the rear-axle centre on the \
ground. Yaw is positive to the left, pitch positive pointing down. Pixel density is sqrt(radial × tangential \
sampling) / range.
- Name cameras by their tags (F, R, ML, ...).
- run_study, compare_variants and optimise_camera compute for a few seconds: use them when the question needs \
them. show_in_app points the user at what you describe (select cameras, frame a point, switch a heatmap); use it when \
it helps them see the answer, once or twice per answer, not routinely.
- If the user has written a project brief (requirements, constraints), check your answers and proposals against it.

How to answer:
- Lead with the answer in one or two sentences, then the evidence. Be specific and brief: no filler, no restating \
the question, no closing offers.
- Use Markdown: short paragraphs, bullet lists, and small tables for comparisons. Round sensibly (areas to 0.01 m², \
small moves in mm, angles to 0.1°).
- Say plainly when an answer depends on data Vantage does not have (real suspension data, a real unit's calibration, \
real captures)."""


AGENT = """

AGENT MODE. The user has given you a goal to work towards on your own:
1. First call update_plan with 3-7 concrete steps.
2. Work through them, calling update_plan as each step starts (in_progress) and ends (done or skipped).
3. Measure before you change anything: read the figures, run the studies the goal needs, and compare options with \
simulate_changes and optimise_camera. Prefer changes that keep left/right pairs symmetric.
4. Finish with propose_changes for the best option you found (or say plainly that none is needed or none works), then \
a short report: what you tried, the measured effect of each option, the trade-offs, and what is left open.
You have about {rounds} rounds of tool calls and {seconds:.0f} seconds; the user can stop you at any time."""


@dataclass
class Settings:
    model: str
    effort: str = 'medium'
    show_reasoning: bool = True
    max_tokens: int = 8192
    agent: bool = False


@dataclass
class Event:
    kind: str                   # turn | reasoning | content | tool_start | tool_done | proposal | status | usage | error | done
    data: dict = field(default_factory=dict)


def context_note(tc: T.ToolContext) -> str:
    p = tc.project
    bits = [f'project "{p.name}"', f'variant "{p.variant.name}"']
    if tc.page:
        bits.append(f'open page: {tc.page}')
    if tc.selection:
        bits.append('selected camera' + ('s' if len(tc.selection) > 1 else '') + ': ' + ', '.join(tc.selection))
    out = '[Vantage context: ' + '; '.join(bits) + ']'
    if tc.brief.strip():
        out += '\n[Project brief from the user:\n' + tc.brief.strip()[:3000] + '\n]'
    if tc.agent:
        out += '\n[Agent mode: work towards this goal on your own, as instructed.]'
    return out


class Session:
    def __init__(self, key: str, settings: Settings, url: str | None = None):
        self.client = Client(key, url)
        self.settings = settings
        self.messages: list = [{'role': 'system', 'content': SYSTEM}]
        self.usage = Usage()
        self._plain = False                 # the model refused the reasoning parameters once: stop sending them
        self.lock = threading.Lock()

    def reset(self):
        self.messages = [{'role': 'system', 'content': SYSTEM}]
        self.usage = Usage()

    # ---------------------------------------------------------------- one question
    def run_turn(self, text: str, tc: T.ToolContext, emit, cancel: threading.Event):
        with self.lock:
            mark = len(self.messages)
            self.messages.append({'role': 'user', 'content': f'{text}\n\n{context_note(tc)}'})
            tc.on_proposal = lambda prop: emit(Event('proposal', {'proposal': prop}))
            tc.on_plan = lambda steps: emit(Event('plan', {'steps': steps}))
            tc.on_status = lambda text: emit(Event('tool_status', {'text': text}))
            tc.cancel = cancel
            ok = False
            try:
                ok = self._loop(tc, emit, cancel)
            except ApiError as e:
                emit(Event('error', {'kind': e.kind, 'message': e.message, 'detail': e.detail}))
            except Exception as e:                  # never leave the panel waiting
                emit(Event('error', {'kind': 'internal', 'message': f'{type(e).__name__}: {e}'}))
            if not ok:
                del self.messages[mark:]            # a failed or stopped turn leaves no half answer behind
            emit(Event('done', {'ok': ok, 'usage': self.usage}))

    def _system(self):
        st = self.settings
        if not st.agent:
            return SYSTEM
        return SYSTEM + AGENT.format(rounds=AGENT_ROUNDS - 2, seconds=AGENT_SECONDS)

    def _payload(self, wrap_up=False):
        st = self.settings
        msgs = [{'role': 'system', 'content': self._system()}] + self._history()[1:]
        if wrap_up:
            msgs.append({'role': 'user', 'content': '[Vantage: the budget for this request is used up. Do not call '
                                                    'tools any more; answer now with what you have found.]'})
        out = {'model': st.model, 'messages': msgs, 'tools': T.schemas(), 'tool_choice': 'none' if wrap_up else 'auto',
               'max_completion_tokens': int(st.max_tokens)}
        if not self._plain:
            out.update(reasoning_params(st.model, st.effort, st.show_reasoning))
        return out

    def _loop(self, tc, emit, cancel) -> bool:
        rounds, retried_tool, retried_rate = 0, False, False
        limit = AGENT_ROUNDS if self.settings.agent else MAX_ROUNDS
        t0 = time.perf_counter()
        wrapped = False
        while True:
            if cancel.is_set():
                raise ApiError('cancelled', 'Stopped.')
            elapsed = time.perf_counter() - t0
            wrap = rounds >= limit - 1 or (self.settings.agent and elapsed > AGENT_SECONDS)
            if wrap and wrapped:
                emit(Event('error', {'kind': 'rounds', 'message': f'Stopped after {rounds} rounds of tool calls. '
                                                                  'Ask a narrower question.'}))
                return False
            emit(Event('turn', {'round': rounds}))
            emit(Event('progress', {'round': rounds, 'max': limit, 'elapsed': elapsed, 'agent': self.settings.agent}))
            try:
                content, calls = self._stream(self._payload(wrap_up=wrap), emit, cancel)
                if wrap:
                    wrapped = True
                    if calls and not content.strip():       # it would not stop calling tools
                        emit(Event('error', {'kind': 'rounds', 'message': f'Stopped after {rounds} rounds of tool '
                                                                          'calls. Ask a narrower question.'}))
                        return False
                    calls = []                              # told not to; keep what it said and stop
            except ApiError as e:
                if e.kind == 'tool_use_failed' and not retried_tool:
                    retried_tool = True
                    emit(Event('status', {'text': 'The model wrote an invalid tool call; asking again…'}))
                    continue
                if e.kind == 'bad_request' and not self._plain and 'reason' in (e.message + e.detail).lower():
                    self._plain = True
                    emit(Event('status', {'text': 'This model does not take reasoning settings; continuing without.'}))
                    continue
                if e.kind == 'rate' and not retried_rate and (e.retry_after or 0) <= 20:
                    retried_rate = True
                    wait = e.retry_after or 3.0
                    emit(Event('status', {'text': f'Rate limited by Groq; retrying in {wait:.0f} s…'}))
                    if cancel.wait(wait):
                        raise ApiError('cancelled', 'Stopped.')
                    continue
                raise
            msg = {'role': 'assistant', 'content': content or None}
            if calls:
                msg['tool_calls'] = calls
            self.messages.append(msg)
            if not calls:
                return True
            for call in calls:
                if cancel.is_set():
                    raise ApiError('cancelled', 'Stopped.')
                name = call['function']['name']
                try:
                    args = json.loads(call['function'].get('arguments') or '{}')
                except ValueError:
                    args = None
                tool = T.BY_NAME.get(name)
                emit(Event('tool_start', {'id': call['id'], 'name': name, 'label': tool.label if tool else name,
                                          'kind': tool.kind if tool else 'read', 'args': args or {}}))
                t0 = time.perf_counter()
                if args is None:
                    result, ok = T.dump({'error': 'the arguments were not valid JSON'}), False
                else:
                    result, ok = T.run(tc, name, args)
                emit(Event('tool_done', {'id': call['id'], 'name': name, 'result': result, 'ok': ok,
                                         'ms': (time.perf_counter() - t0) * 1000}))
                self.messages.append({'role': 'tool', 'tool_call_id': call['id'], 'name': name, 'content': result})
            rounds += 1

    def _stream(self, payload, emit, cancel):
        content, parts = [], {}
        for d in self.client.stream(payload, cancel):
            if d.reasoning:
                emit(Event('reasoning', {'text': d.reasoning}))
            if d.content:
                content.append(d.content)
                emit(Event('content', {'text': d.content}))
            for tc in d.tool_calls:
                i = int(tc.get('index', len(parts)))
                cur = parts.setdefault(i, {'id': '', 'type': 'function', 'function': {'name': '', 'arguments': ''}})
                if tc.get('id'):
                    cur['id'] = tc['id']
                fn = tc.get('function') or {}
                if fn.get('name'):
                    cur['function']['name'] += fn['name'] if not cur['function']['name'] else ''
                if fn.get('arguments'):
                    cur['function']['arguments'] += fn['arguments']
            if d.usage:
                self.usage.add(d.usage)
                emit(Event('usage', {'usage': self.usage}))
        calls = []
        for i in sorted(parts):
            c = parts[i]
            if not c['function']['name']:
                continue
            c['id'] = c['id'] or f'call_{len(self.messages)}_{i}'
            calls.append(c)
        return ''.join(content), calls

    def _history(self):
        """The messages, with old tool results shortened once the conversation gets long."""
        total = sum(len(str(m.get('content') or '')) for m in self.messages)
        if total <= MAX_HISTORY_CHARS:
            return self.messages
        out = []
        last_user = max(i for i, m in enumerate(self.messages) if m['role'] == 'user')
        for i, m in enumerate(self.messages):
            if m['role'] == 'tool' and i < last_user and len(m['content']) > 400:
                m = dict(m, content=m['content'][:400] + '…(shortened: ask again for the full figures)')
            out.append(m)
        return out

    def transcript_markdown(self) -> str:
        lines = []
        for m in self.messages:
            if m['role'] == 'user':
                text = m['content'].rsplit('\n\n[Vantage context:', 1)[0]
                lines += ['### You', '', text, '']
            elif m['role'] == 'assistant' and m.get('content'):
                lines += ['### Assistant', '', m['content'], '']
        return '\n'.join(lines)
