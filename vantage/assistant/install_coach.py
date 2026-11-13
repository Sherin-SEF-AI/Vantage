"""Claude as the install guide's coach (core.install_guide): plain-language feedback to the technician fitting the
rig, on events, from the guide's own measurements.

What Claude does and does not do
- The guide measures (poses, covariances, coverage, the tests); Claude never computes a calibration. It reads the
  guide's state, can look at a camera's current frame (to spot what the numbers cannot: glare, a dirty or covered lens,
  a cable or a hand in view, a bent or shaded board, a board laid in the wrong place) and can ask what turning a
  camera would do to the coverage (core.install_guide.coverage_whatif). It answers in short instructions.
- Every number in its answer must be one of the guide's: number_guard() checks each figure against the numbers of
  the state and the tool results it was given (the figure as written, to the decimals it shows) and removes any
  sentence with a figure that matches none, saying so. Counts up to 12, camera tags and board names are allowed.
- When it is asked: on events (should_ask): a stage done or newly failing, a camera that went off, the installer
  stuck on a stage for STUCK_S seconds; never more often than every MIN_GAP_S seconds; and whenever the technician asks.
  Not per frame: a reply takes seconds, and the guide's own overlays do the real-time part.

Request: Anthropic's SDK (the optional anthropic package), the assistant's Claude key (Preferences ▸ Assistant;
assistant.secrets), claude-opus-5-5 by default, adaptive thinking, effort low for an event and medium for a question,
streaming, tool_choice auto with strict closed schemas, every input checked before a tool runs, server-side fallback on
a refusal, a frozen system prompt with a cache breakpoint (the volatile state goes in the user message). The frame sent
by look() is a JPEG at most 1024 px wide: nothing else of the project leaves the machine."""
from __future__ import annotations

import base64
import json
import math
import re
import threading
import time

from . import providers as PR

MAX_TOKENS = 16000
MAX_ROUNDS = 4
MIN_GAP_S = 20.0
STUCK_S = 90.0
TIMEOUT_S = 90.0
SECTIONS = ('all', 'connect', 'identify', 'lens', 'aim', 'boards', 'solve', 'coverage', 'verify', 'stands', 'events',
            'settings')

SYSTEM = """You coach a technician who is fitting a camera rig to a car, through Vantage's install guide. The guide \
streams every camera live and measures everything itself: which stream is which camera, each camera's aim from the \
car's own outline, the boards on the floor (and on stands) in every image, every camera's pose with its 1σ \
uncertainty from one joint least-squares fit, the ground coverage graded robust/uncertain/blind from that \
uncertainty, and independent checks. Stages, in order: connect, identify, lens, aim, boards, solve, coverage, verify.

Your job: tell the technician what to do next and why, in plain words, from the guide's state and, when it helps, \
a look at a camera's image.

Rules
- Every number you write must be copied from the state or a tool result (you may round it to fewer decimals). Never \
estimate, convert or invent a figure: a sentence with a figure the guide did not give is removed before the \
technician sees it.
- Short imperative sentences, at most six lines, the most important first. Name cameras by their tags (F, R, ML, MR, \
FN, RN …) and boards by their names (T1, S1 …).
- Directions are the technician's as they stand behind the camera looking where it looks: "turn it right", "tilt it \
down", "roll its left side down". The guide's aim instructions already say this and give the shim; repeat them.
- When the numbers cannot explain a problem (no board found where the plan puts it, a camera that cannot be aimed, \
residuals larger than the noise), look at that camera's image and say what you see: glare, a dirty or covered lens, \
a cable or hand in view, a board that is bent, shaded, cut off by the image edge or laid in the wrong place.
- You never decide that a stage is done: the guide's tests do. If they pass, say so and name the next stage.
- No preamble, no summary of the rules, no markdown headings."""

TOOLS = [
    {'name': 'get_state',
     'description': 'The install guide\'s current state: one section, or all of it. Sections: connect (stream rates), '
                    'identify (which stream is which camera), lens (calibrations), aim (each camera\'s turn from the '
                    'plan, with instructions), boards (boards expected and found per camera), solve (poses, 1σ, '
                    'ratios to the requirement, χ² of the fit), coverage (robust/uncertain/blind ground), verify (the '
                    'independent tests), stands, events, settings.',
     'input_schema': {'type': 'object', 'properties': {'section': {'type': 'string', 'enum': list(SECTIONS)}},
                      'required': ['section'], 'additionalProperties': False}},
    {'name': 'look',
     'description': 'The current image of one camera (a JPEG at most 1024 px wide) with the boards the plan expects '
                    'outlined in yellow. Use it to see what the numbers cannot tell: glare, a covered lens, a board '
                    'in the wrong place.',
     'input_schema': {'type': 'object', 'properties': {'camera': {'type': 'string', 'description': 'Camera tag'}},
                      'required': ['camera'], 'additionalProperties': False}},
    {'name': 'coverage_whatif',
     'description': 'The planned coverage of the ground region with one camera turned by the given degrees (Vantage '
                    'angles: yaw positive turns left, pitch positive tilts down, roll positive drops its right side), '
                    'against the plan as it is.',
     'input_schema': {'type': 'object', 'properties': {
         'camera': {'type': 'string'}, 'd_yaw_deg': {'type': 'number'}, 'd_pitch_deg': {'type': 'number'},
         'd_roll_deg': {'type': 'number'}},
         'required': ['camera', 'd_yaw_deg', 'd_pitch_deg', 'd_roll_deg'], 'additionalProperties': False}},
]
for _t in TOOLS:
    _t['strict'] = True
    _t['eager_input_streaming'] = True


# =========================================================================== the number guard
_TAG = re.compile(r'\b[A-Za-z]{1,3}\d+[A-Za-z]?\b')          # T12, S1, FN2, sm_87: names, not figures
_NUM = re.compile(r'(?<![\w.])[-+−]?\d+(?:[.,]\d+)?')
_SENT = re.compile(r'(?<=[.!?])\s+|\n+')


def numbers_in(o, out=None) -> list[float]:
    """Every number in a JSON-like value, strings included (the guide's instructions carry figures in text)."""
    out = [] if out is None else out
    if isinstance(o, bool) or o is None:
        return out
    if isinstance(o, (int, float)):
        if math.isfinite(o):
            out.append(float(o))
        return out
    if isinstance(o, str):
        for m in _NUM.finditer(_TAG.sub(' ', o)):
            try:
                out.append(float(m.group().replace('−', '-').replace(',', '.')))
            except ValueError:
                pass
        return out
    if isinstance(o, dict):
        for v in o.values():
            numbers_in(v, out)
        return out
    if isinstance(o, (list, tuple)):
        for v in o:
            numbers_in(v, out)
    return out


def _matches(text_num: str, allowed: list[float]) -> bool:
    s = text_num.replace('−', '-').replace(',', '.').lstrip('+')
    try:
        a = float(s)
    except ValueError:
        return True
    if a.is_integer() and 0 <= abs(a) <= 12 and '.' not in s:
        return True                                     # counts, steps, a camera's position in a list
    dec = len(s.split('.')[1]) if '.' in s else 0
    for b in allowed:
        for x in (b, -b, b * 100.0, b / 100.0):        # a sign the sentence turned into a word; a fraction as %
            if round(x, dec) == round(a, dec) or (dec == 0 and abs(x - a) < 0.5 + 1e-9):
                return True
    return False


def number_guard(text: str, allowed: list[float]) -> tuple[str, list[str]]:
    """The answer with every sentence that carries a figure not among the allowed numbers removed; and the removed
    sentences."""
    keep, removed = [], []
    for sent in _SENT.split(text or ''):
        if not sent.strip():
            continue
        bad = [m.group() for m in _NUM.finditer(_TAG.sub(' ', sent)) if not _matches(m.group(), allowed)]
        (removed if bad else keep).append(sent.strip())
    return '\n'.join(keep), removed


# =========================================================================== when to ask
def signature(state: dict) -> tuple:
    stages = tuple((r['stage'], bool(r['done'])) for r in state.get('stages', []))
    off = tuple(sorted(c['camera'] for c in (state.get('connect') or {}).get('cameras', []) if not c.get('ok')))
    why = (state.get('solve') or {}).get('why', '')
    return stages, off, why, state.get('stage')


class Coach:
    """provide_state() -> the guide's state (GuideSession.state()); provide_frame(tag) -> the camera's latest frame
    (numpy) or None; overlay(tag, frame) -> the frame with the expected boards drawn (optional); whatif(tag, dy, dp,
    dr) -> coverage_whatif's dict (optional)."""

    def __init__(self, provide_state, provide_frame, overlay=None, whatif=None, key=None, model=None, url=None):
        self.provide_state = provide_state
        self.provide_frame = provide_frame
        self.overlay = overlay
        self.whatif = whatif
        self._key = key
        self.model = model or PR.default_model('claude')
        self.url = url
        self.last_sig = None
        self.last_t = 0.0
        self.stage_since = time.time()
        self.last_stage = None
        self.busy = threading.Lock()
        self.log: list = []                  # every exchange: {'t', 'reason', 'question', 'text', 'removed', ...}
        self.usage = {'input_tokens': 0, 'output_tokens': 0, 'cache_read_input_tokens': 0}

    # ------------------------------------------------------------------ availability
    def key(self):
        if self._key:
            return self._key
        from . import secrets
        k, _ = secrets.load('claude')
        return k

    def available(self) -> tuple[bool, str]:
        from . import claude as CL
        if CL.anthropic is None:
            return False, 'the Claude coach needs Anthropic\'s Python SDK (pip install anthropic)'
        if not self.key():
            return False, 'no Claude key: add one in Preferences ▸ Assistant'
        return True, ''

    # ------------------------------------------------------------------ policy
    def should_ask(self, state: dict, now=None) -> str | None:
        """The reason to ask Claude now, or None."""
        now = time.time() if now is None else now
        if state.get('stage') != self.last_stage:
            self.last_stage = state.get('stage')
            self.stage_since = now
        if now - self.last_t < MIN_GAP_S:
            return None
        sig = signature(state)
        if self.last_sig is None:
            return 'the guide started'
        if sig != self.last_sig:
            old = dict(self.last_sig[0])
            new = dict(sig[0])
            done = [s for s, d in new.items() if d and not old.get(s)]
            lost = [s for s, d in new.items() if not d and old.get(s)]
            if done:
                return 'stage done: ' + ', '.join(done)
            if lost:
                return 'stage no longer passes: ' + ', '.join(lost)
            if sig[1] != self.last_sig[1] and sig[1]:
                return 'cameras not delivering: ' + ', '.join(sig[1])
            if sig[2] != self.last_sig[2] and sig[2]:
                return 'the solve changed: ' + sig[2]
            if sig[3] != self.last_sig[3]:
                return f'now on the {sig[3]} stage'
        if now - self.stage_since > STUCK_S and now - self.last_t > STUCK_S:
            return f'the technician has been on the {state.get("stage")} stage for {int(now - self.stage_since)} s'
        return None

    # ------------------------------------------------------------------ tools
    def _tool(self, name, inp, state, allowed, images):
        if name == 'get_state':
            sec = inp.get('section')
            if sec not in SECTIONS:
                return {'error': f'unknown section {sec!r}'}, True
            val = state if sec == 'all' else state.get(sec)
            numbers_in(val, allowed)
            return val, False
        if name == 'look':
            tag = str(inp.get('camera', ''))
            f = self.provide_frame(tag)
            if f is None:
                return {'error': f'no image from {tag}'}, True
            if self.overlay is not None:
                try:
                    f = self.overlay(tag, f)
                except Exception:
                    pass
            images.append(_jpeg_block(f))
            return {'camera': tag, 'image': 'attached'}, False
        if name == 'coverage_whatif':
            if self.whatif is None:
                return {'error': 'not available'}, True
            try:
                args = [float(inp[k]) for k in ('d_yaw_deg', 'd_pitch_deg', 'd_roll_deg')]
            except (KeyError, TypeError, ValueError):
                return {'error': 'd_yaw_deg, d_pitch_deg and d_roll_deg must be numbers'}, True
            r = self.whatif(str(inp.get('camera', '')), *args)
            numbers_in(r, allowed)
            allowed.extend(args)
            return r, False
        return {'error': f'unknown tool {name}'}, True

    # ------------------------------------------------------------------ the request
    def ask(self, reason='', question=None, cancel=None, effort=None) -> dict:
        """One exchange; returns {'text', 'removed', 'reason', 'question', 'model', 'seconds', 'error'}."""
        from . import claude as CL
        out = {'t': time.time(), 'reason': reason, 'question': question, 'text': '', 'removed': [], 'error': '',
               'model': self.model, 'seconds': 0.0}
        ok, why = self.available()
        if not ok:
            out['error'] = why
            return out
        if not self.busy.acquire(blocking=False):
            out['error'] = 'busy'
            return out
        t0 = time.time()
        try:
            state = self.provide_state()
            self.last_sig = signature(state)
            self.last_t = time.time()
            allowed = numbers_in(state)
            user = ([f'Why you are asked now: {reason}.'] if reason else []) + \
                   ([f'The technician asks: {question}'] if question else []) + \
                   ['The guide\'s state (JSON):', json.dumps(state, separators=(',', ':'), default=str)]
            messages = [{'role': 'user', 'content': [{'type': 'text', 'text': '\n'.join(user)}]}]
            cl = CL.anthropic.Anthropic(api_key=self.key().strip(), base_url=self.url or CL.base_url(),
                                        timeout=CL.anthropic.Timeout(TIMEOUT_S, connect=15.0))
            text = ''
            for rnd in range(MAX_ROUNDS):
                kw = {'model': self.model, 'max_tokens': MAX_TOKENS,
                      'system': [{'type': 'text', 'text': SYSTEM, 'cache_control': {'type': 'ephemeral'}}],
                      'messages': messages, 'tools': TOOLS,
                      'tool_choice': {'type': 'auto' if rnd < MAX_ROUNDS - 1 else 'none'},
                      'thinking': {'type': 'adaptive'},
                      'output_config': {'effort': effort or ('medium' if question else 'low')},
                      'betas': [CL.BETA_FALLBACK], 'fallbacks': 'default'}
                try:
                    with cl.beta.messages.stream(**kw) as stream:
                        for _ in stream:
                            if cancel is not None and cancel.is_set():
                                out['error'] = 'stopped'
                                return out
                        msg = stream.get_final_message()
                except CL.anthropic.AnthropicError as e:
                    out['error'] = CL.map_error(e).message
                    return out
                u = msg.usage
                for k in self.usage:
                    self.usage[k] += int(getattr(u, k, 0) or 0)
                if msg.stop_reason == 'refusal':
                    out['error'] = 'Claude declined to answer (a refusal).'
                    return out
                content = CL.echo_content(msg.content)
                text = ''.join(getattr(b, 'text', '') for b in content if getattr(b, 'type', None) == 'text')
                calls = [b for b in content if getattr(b, 'type', None) == 'tool_use']
                if msg.stop_reason != 'tool_use' or not calls:
                    break
                messages.append({'role': 'assistant', 'content': list(content)})
                results = []
                for b in calls:
                    inp = b.input if isinstance(b.input, dict) else {}
                    images = []
                    val, err = self._tool(b.name, inp, state, allowed, images)
                    body = [{'type': 'text', 'text': json.dumps(val, separators=(',', ':'), default=str)}] + images
                    results.append({'type': 'tool_result', 'tool_use_id': b.id, 'content': body, 'is_error': err})
                messages.append({'role': 'user', 'content': results})
            clean, removed = number_guard(text, allowed)
            out.update(text=clean, removed=removed)
        finally:
            out['seconds'] = time.time() - t0
            self.busy.release()
        self.log.append(out)
        del self.log[:-50]
        return out


def _jpeg_block(frame, max_w=1024) -> dict:
    import cv2
    import numpy as np
    f = np.asarray(frame)
    h, w = f.shape[:2]
    if w > max_w:
        f = cv2.resize(f, (max_w, max(1, round(h * max_w / w))), interpolation=cv2.INTER_AREA)
    if f.ndim == 3:
        f = cv2.cvtColor(f, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode('.jpg', f, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return {'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/jpeg',
                                       'data': base64.standard_b64encode(buf.tobytes()).decode('ascii')}}
