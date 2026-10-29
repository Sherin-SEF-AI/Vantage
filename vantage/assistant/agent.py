"""One conversation with the model: streams the answer, runs the tools it calls, and reports everything as events
(called on the worker thread that runs the turn; the UI forwards them to the Qt thread).

The loop (rounds, the budget, cancelling, the events, a failed turn leaving nothing behind) is the same for both
providers; how a request is made and how the history is kept is the provider's own: Session (Groq, an OpenAI-style
message list whose older tool results are shortened when the conversation gets long) and claude.ClaudeSession
(Anthropic's content blocks, append-only: the assistant's content goes back unchanged, thinking blocks included, and
the server's compaction keeps long conversations within the context)."""
from __future__ import annotations

import json
import os
import re
import threading
import time
from dataclasses import dataclass, field

from . import providers as PR
from . import tools as T
from .client import ApiError, Client, reasoning_params

MAX_ROUNDS = 12                 # tool rounds per question
AGENT_ROUNDS = 30               # ... when working towards a goal
AGENT_SECONDS = 240.0           # time budget for a goal; then the model writes up what it has
MAX_HISTORY_CHARS = 240_000     # Groq: older tool results are shortened past this

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
real captures).

THE COPILOT. When the user wants something run or written (a CARLA drive, a simulation campaign, detection quality \
on recordings, critical targets in CARLA, the CARLA live rig, a designed experiment, a sample dataset, a study, an \
export), you plan it; you never run it:
1. Find what you need with the read tools: get_catalogue (towns, weathers and Vantage conditions, scenarios and \
their parameters, fault kinds and ranges, actor kinds, the detector model, the recordings found, past runs, export \
formats, studies, plan kinds), get_plan_settings (a kind's settings: key, current value, range or choices), \
estimate_plan, validate_plan and explain_limit.
2. Call propose_plan with every step: its kind, a short label and the values you set, each as {key, value (JSON \
text: "\\"Town10HD\\"", "30", "true", "[...]"), source ("request" when the user asked for it, "copilot" when you \
chose it), reason (one line)}. Values you do not set keep the page's default. Nested settings take dotted keys \
(scenario.name, scenario.params.distance_m, ground_truth.semantic); lists are set whole (faults, actors, factors, \
recordings, formats, seeds).
3. Read what propose_plan returns. Errors make the plan invalid and it cannot be approved: fix them with \
propose_plan(revise=true), or say plainly why they cannot be fixed. Warnings and the estimate are shown to the user.
4. A follow-up ("more rain", "only F", "halve it") revises the plan on screen: propose_plan with revise=true and only \
the values that change, steps in the same order. Never start over unless asked.
5. Then answer in two or three sentences: what the plan does, its cost and the warnings that matter. The user \
approves, edits or rejects it on the plan card; nothing runs until they approve. Never say a run started or finished \
unless a tool result says so. After a run, read its results with the read tools; a next run is a new plan.

NEXT STEPS. When a next step is genuinely useful (a check the answer calls for, the obvious follow-up), end the answer \
by calling suggest_next with at most 3 short items {label, prompt, kind: ask | goal | plan}. They are shown as buttons \
under your answer and run nothing unless the user clicks; a plan still needs approval. Skip it when nothing is worth \
suggesting, and never suggest what you just did."""


AGENT = """

AGENT MODE. The user has given you a goal to work towards on your own:
1. First call update_plan with 3-7 concrete steps.
2. Work through them, calling update_plan as each step starts (in_progress) and ends (done or skipped).
3. Measure before you change anything: read the figures, run the studies the goal needs, and compare options with \
simulate_changes and optimise_camera. Prefer changes that keep left/right pairs symmetric.
4. Finish with propose_changes for the best option you found (or say plainly that none is needed or none works), then \
a short report: what you tried, the measured effect of each option, the trade-offs, and what is left open.
You have about {rounds} rounds of tool calls and {seconds:.0f} seconds; the user can stop you at any time."""

WRAP_UP = ('[Vantage: the budget for this request is used up. Do not call tools any more; answer now with what you '
           'have found.]')


@dataclass
class Settings:
    model: str
    effort: str = 'medium'
    show_reasoning: bool = True
    max_tokens: int = 8192
    agent: bool = False
    planning: bool = False           # a Copilot turn (Claude: effort high; plain questions medium)
    fallbacks: bool = True           # Claude: server-side fallback on a refusal (fallbacks: "default")


@dataclass
class Event:
    kind: str                   # turn | reasoning | content | tool_start | tool_done | proposal | next_steps | status | usage | error | done
    data: dict = field(default_factory=dict)


@dataclass
class Call:
    """A tool call as the loop runs it: id, name, the parsed input (None when it was not valid JSON) and the raw text."""
    id: str
    name: str
    args: dict | None
    raw: str = ''


@dataclass
class Reply:
    text: str = ''
    calls: list = field(default_factory=list)
    stop: str = ''                  # end_turn | tool_use | max_tokens | refusal | pause_turn | stop | tool_calls
    refusal: dict | None = None     # {'category', 'explanation'} on a refusal
    native: object = None           # the provider's own message (Claude: the content blocks, passed back unchanged)


# ------------------------------------------------------------------ privacy
_HOME = os.path.expanduser('~')
_ABS = re.compile(r'(?<![\w:/.\\~-])(?:/(?:[^\s"\'<>|*?\\/,;]+/)+)([^\s"\'<>|*?\\/,;]*)')


def mask(text: str) -> str:
    """What leaves the machine never names this machine's folders: the home folder becomes ~, any other absolute path
    keeps only its last part; a URL's password is removed (core.credentials.mask)."""
    if not isinstance(text, str) or not text:
        return text
    if _HOME and _HOME not in ('/', ''):
        text = text.replace(_HOME, '~')
    text = _ABS.sub(lambda m: '…/' + m.group(1), text)
    try:
        from ..core import credentials as CR
        text = CR.mask(text)
    except Exception:                       # noqa: BLE001
        pass
    return text


def context_note(tc: T.ToolContext, agent_text=False) -> str:
    p = tc.project
    bits = [f'project "{p.name}"', f'variant "{p.variant.name}"']
    if tc.page:
        bits.append(f'open page: {tc.page}')
    if tc.selection:
        bits.append('selected camera' + ('s' if len(tc.selection) > 1 else '') + ': ' + ', '.join(tc.selection))
    out = '[Vantage context: ' + '; '.join(bits) + ']'
    if tc.brief.strip():
        out += '\n[Project brief from the user:\n' + tc.brief.strip()[:3000] + '\n]'
    if getattr(tc, 'plan', None) is not None:
        pl = tc.plan
        out += (f'\n[Copilot: the plan on screen is {pl.id} v{pl.version} "{pl.title}" ({pl.status}, '
                f'{len(pl.steps)} step' + ('s' if len(pl.steps) != 1 else '') + '); a follow-up revises it with '
                'propose_plan(revise=true).]')
    if tc.agent:
        out += '\n[Agent mode: work towards this goal on your own, as instructed.]'
        if agent_text:
            out += AGENT.format(rounds=AGENT_ROUNDS - 2, seconds=AGENT_SECONDS)
    return mask(out)


class BaseSession:
    """The loop both providers share. A subclass makes the requests and keeps the history."""
    provider = PR.GROQ

    def __init__(self, settings: Settings):
        self.settings = settings
        self.usage = PR.Usage(provider=self.provider, model=settings.model)
        self.lock = threading.Lock()
        self.last_payload = None            # the last request as it was sent (the "What was sent" inspector)

    # ---------------------------------------------------------------- one question
    def run_turn(self, text: str, tc: T.ToolContext, emit, cancel: threading.Event):
        with self.lock:
            self.usage.model = self.settings.model
            mark = self._mark()
            tc.request = tc.request or text          # recorded with a plan proposed this turn
            self._begin(mask(text), tc)
            tc.on_proposal = lambda prop: emit(Event('proposal', {'proposal': prop}))
            tc.on_plan = lambda steps: emit(Event('plan', {'steps': steps}))
            tc.on_status = lambda text: emit(Event('tool_status', {'text': text}))
            tc.on_next = lambda items: emit(Event('next_steps', {'items': items}))
            prev = tc.on_copilot

            def on_copilot(plan, prev=prev):
                if prev is not None:
                    prev(plan)
                emit(Event('copilot', {'plan': plan}))
            tc.on_copilot = on_copilot
            tc.cancel = cancel
            ok = False
            try:
                ok = self._loop(tc, emit, cancel)
            except ApiError as e:
                emit(Event('error', {'kind': e.kind, 'message': e.message, 'detail': e.detail}))
            except Exception as e:                  # never leave the panel waiting
                emit(Event('error', {'kind': 'internal', 'message': f'{type(e).__name__}: {e}'}))
            if not ok:
                self._rollback(mark)                # a failed or stopped turn leaves no half answer behind
            emit(Event('done', {'ok': ok, 'usage': self.usage}))

    def _loop(self, tc, emit, cancel) -> bool:
        rounds, retries = 0, {}
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
                reply = self._request(wrap, emit, cancel)
            except ApiError as e:
                if self._retry(e, emit, cancel, retries):
                    continue
                raise
            if wrap:
                wrapped = True
                if reply.calls and not reply.text.strip():          # it would not stop calling tools
                    emit(Event('error', {'kind': 'rounds', 'message': f'Stopped after {rounds} rounds of tool '
                                                                      'calls. Ask a narrower question.'}))
                    return False
                reply.calls = []                                    # told not to; keep what it said and stop
            if reply.stop == 'refusal':
                rf = reply.refusal or {}
                cat = rf.get('category')
                raise ApiError('refusal', 'The model declined this request' + (f' ({cat})' if cat else '') +
                               (f': {rf["explanation"]}' if rf.get('explanation') else '.') +
                               ' Nothing was run. Rephrase it, or ask about the design itself.',
                               detail=json.dumps(rf, default=str))
            if reply.stop == 'max_tokens' and reply.calls:
                raise ApiError('truncated', 'The answer was cut off at its token limit while writing a tool call, so '
                                            'the call was not run. Ask again, more narrowly.')
            self._append_assistant(reply, wrap)
            if not reply.calls:
                return True
            results = []
            for call in reply.calls:
                if cancel.is_set():
                    raise ApiError('cancelled', 'Stopped.')
                tool = T.BY_NAME.get(call.name)
                emit(Event('tool_start', {'id': call.id, 'name': call.name, 'label': tool.label if tool else call.name,
                                          'kind': tool.kind if tool else 'read', 'args': call.args or {}}))
                t1 = time.perf_counter()
                if call.args is None:
                    result, ok = T.dump({'error': 'the arguments were not valid JSON',
                                         'INVALID_JSON': call.raw[:2000]}), False
                else:
                    bad = T.validate_args(call.name, call.args)
                    if bad:
                        result, ok = T.dump({'error': 'the input does not match the tool\'s schema: ' + '; '.join(bad)}), \
                            False
                    else:
                        result, ok = T.run(tc, call.name, call.args)
                result = mask(result)
                emit(Event('tool_done', {'id': call.id, 'name': call.name, 'result': result, 'ok': ok,
                                         'ms': (time.perf_counter() - t1) * 1000}))
                results.append((call, result, ok))
            self._append_results(results, wrap=False)
            rounds += 1

    # ---------------------------------------------------------------- the provider's part
    def _mark(self):
        raise NotImplementedError

    def _rollback(self, mark):
        raise NotImplementedError

    def _begin(self, text, tc):
        raise NotImplementedError

    def _request(self, wrap, emit, cancel) -> Reply:
        raise NotImplementedError

    def _append_assistant(self, reply: Reply, wrap: bool):
        raise NotImplementedError

    def _append_results(self, results, wrap: bool):
        raise NotImplementedError

    def _retry(self, e: ApiError, emit, cancel, retries: dict) -> bool:
        if e.kind in ('rate', 'overloaded') and not retries.get(e.kind) and (e.retry_after or 0) <= 20:
            retries[e.kind] = True
            wait = e.retry_after or 3.0
            who = PR.LABEL[self.provider]
            emit(Event('status', {'text': (f'Rate limited by {who}' if e.kind == 'rate' else f'{who} is overloaded')
                                  + f'; retrying in {wait:.0f} s…'}))
            if cancel.wait(wait):
                raise ApiError('cancelled', 'Stopped.')
            return True
        return False

    def reset(self):
        raise NotImplementedError

    def transcript_markdown(self) -> str:
        raise NotImplementedError


class Session(BaseSession):
    """A conversation on Groq (OpenAI-compatible chat completions)."""
    provider = PR.GROQ

    def __init__(self, key: str, settings: Settings, url: str | None = None):
        super().__init__(settings)
        self.client = Client(key, url)
        self.messages: list = [{'role': 'system', 'content': SYSTEM}]
        self._plain = False                 # the model refused the reasoning parameters once: stop sending them

    def reset(self):
        self.messages = [{'role': 'system', 'content': SYSTEM}]
        self.usage = PR.Usage(provider=self.provider, model=self.settings.model)

    def _mark(self):
        return len(self.messages)

    def _rollback(self, mark):
        del self.messages[mark:]

    def _begin(self, text, tc):
        self.messages.append({'role': 'user', 'content': f'{text}\n\n{context_note(tc)}'})

    def _system(self):
        st = self.settings
        if not st.agent:
            return SYSTEM
        return SYSTEM + AGENT.format(rounds=AGENT_ROUNDS - 2, seconds=AGENT_SECONDS)

    def _payload(self, wrap_up=False):
        st = self.settings
        msgs = [{'role': 'system', 'content': self._system()}] + self._history()[1:]
        if wrap_up:
            msgs.append({'role': 'user', 'content': WRAP_UP})
        out = {'model': st.model, 'messages': msgs, 'tools': T.schemas(), 'tool_choice': 'none' if wrap_up else 'auto',
               'max_completion_tokens': int(st.max_tokens)}
        if not self._plain:
            out.update(reasoning_params(st.model, st.effort, st.show_reasoning))
        return out

    def _retry(self, e, emit, cancel, retries) -> bool:
        if e.kind == 'tool_use_failed' and not retries.get('tool'):
            retries['tool'] = True
            emit(Event('status', {'text': 'The model wrote an invalid tool call; asking again…'}))
            return True
        if e.kind == 'bad_request' and not self._plain and 'reason' in (e.message + e.detail).lower():
            self._plain = True
            emit(Event('status', {'text': 'This model does not take reasoning settings; continuing without.'}))
            return True
        return super()._retry(e, emit, cancel, retries)

    def _request(self, wrap, emit, cancel) -> Reply:
        payload = self._payload(wrap_up=wrap)
        self.last_payload = payload
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
        native = []
        for i in sorted(parts):
            c = parts[i]
            if not c['function']['name']:
                continue
            c['id'] = c['id'] or f'call_{len(self.messages)}_{i}'
            native.append(c)
        calls = []
        for c in native:
            try:
                args = json.loads(c['function'].get('arguments') or '{}')
                args = args if isinstance(args, dict) else None
            except ValueError:
                args = None
            calls.append(Call(c['id'], c['function']['name'], args, c['function'].get('arguments') or ''))
        return Reply(''.join(content), calls, 'tool_calls' if calls else 'stop', native=native)

    def _append_assistant(self, reply, wrap):
        msg = {'role': 'assistant', 'content': reply.text or None}
        if reply.calls:
            msg['tool_calls'] = reply.native
        self.messages.append(msg)

    def _append_results(self, results, wrap):
        for call, result, ok in results:
            self.messages.append({'role': 'tool', 'tool_call_id': call.id, 'name': call.name, 'content': result})

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


def new_session(provider: str, key: str, settings: Settings, url: str | None = None) -> BaseSession:
    """A conversation with the chosen provider (Claude needs the anthropic SDK: claude.ClaudeUnavailable otherwise)."""
    if provider == PR.CLAUDE:
        from .claude import ClaudeSession
        return ClaudeSession(key, settings, url)
    return Session(key, settings, url)
