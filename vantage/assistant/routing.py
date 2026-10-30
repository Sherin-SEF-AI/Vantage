"""Automatic model choice: which provider and model answers a request, decided on this machine, deterministically, from
what the request is (the mode, a plan, a whole-rig optimisation, a question that needs a picture) and which keys
there are. Nothing is sent to decide it.

The table (route()):
- agent goal                           -> deep  ("goal mode")
- a Copilot plan (a run to plan, or a follow-up to the draft plan on screen) -> deep ("Copilot plan")
- whole-rig optimisation               -> deep  ("rig optimisation")
- a question that needs a picture (spatial: where, which side, the shape of a gap, what a camera sees; or an
  explicit look/picture/map) -> deep ("image question"), as only Claude takes images
- anything else                        -> fast  ("quick question")
deep = Claude Opus 5.5; fast = Groq GPT-OSS 120B, or Claude Sonnet 5.5 when there is no Groq key. When the tier's
model cannot be used (no Claude key, or the anthropic SDK not installed) the best available model answers and the
route says why (fallback), so the panel can say it once.

One conversation keeps one provider's history (Groq's message list and Claude's content blocks are never mixed). A
request routed to the other provider starts a new conversation with a hand-over: a short summary of the conversation
so far, built here from its transcript, sent in front of the first question (handover_text)."""
from __future__ import annotations

import re
from dataclasses import dataclass

from . import providers as PR

FAST, DEEP = 'fast', 'deep'
DEEP_MODEL = (PR.CLAUDE, 'claude-opus-5-5')
FAST_GROQ = (PR.GROQ, 'openai/gpt-oss-120b')
FAST_CLAUDE = (PR.CLAUDE, 'claude-sonnet-5-5')
REASONS = {'goal': 'goal mode', 'plan': 'Copilot plan', 'rig': 'rig optimisation', 'image': 'image question',
           'quick': 'quick question', 'reask': 're-asked', 'manual': 'chosen in Preferences'}

# a whole-rig optimisation: optimise/improve/tune the rig, all/every camera(s), several cameras together
_RIG = re.compile(r'\b(optimi[sz]e|optimi[sz]ation|improve|tune|re-?place|re-?arrange|re-?position)\b[^.?!]{0,60}'
                  r'\b(rig|whole|all (the )?cameras|every camera|all sensors|the cameras|camera layout|layout)\b'
                  r'|\boptimise_rig\b|\brig[- ]wide\b|\bwhole[- ]rig\b', re.IGNORECASE)
# needs a picture: asked to look, or a spatial question about the ground, the gaps, what a camera sees
_LOOK = re.compile(r'\b(look at|show me|picture|image|draw|map|bird\'?s[- ]eye|top view|heat ?map|what does \w+ see|'
                   r'what (?:can|could) \w+ see|view (?:of|from|through))\b', re.IGNORECASE)
_SPATIAL = re.compile(r'\b(where|which side|what side|shape|outline|extent|gap|gaps|patch|patches|layout|left or '
                      r'right|front or rear|corner)\b', re.IGNORECASE)


@dataclass
class Route:
    provider: str
    model: str
    tier: str                       # fast | deep
    reason: str                     # a key of REASONS
    fallback: str = ''              # why the tier's own model is not used ('' when it is)
    note: str = ''                  # anything else worth saying (e.g. pictures need Claude)

    @property
    def label(self) -> str:
        """'Claude Opus 5.5 · goal mode': the model and why it was chosen, shown on the answer."""
        return f'{PR.model_info(self.provider, self.model).label} · {REASONS.get(self.reason, self.reason)}'

    def to_dict(self) -> dict:
        return {'provider': self.provider, 'model': self.model, 'tier': self.tier, 'reason': self.reason,
                'label': self.label, 'fallback': self.fallback, 'note': self.note}


def needs_picture(text: str) -> bool:
    t = str(text or '')
    return bool(_LOOK.search(t) or _SPATIAL.search(t))


def is_rig_optimisation(text: str) -> bool:
    return bool(_RIG.search(str(text or '')))


def classify(text: str, agent=False, planning=False) -> tuple[str, str]:
    """(tier, reason) for a request: the routing table above, first match wins."""
    if agent:
        return DEEP, 'goal'
    if planning:
        return DEEP, 'plan'
    if is_rig_optimisation(text):
        return DEEP, 'rig'
    if needs_picture(text):
        return DEEP, 'image'
    return FAST, 'quick'


def usable(keys: dict, claude_sdk: bool) -> dict:
    """{provider: bool}: a key, and for Claude the SDK."""
    return {PR.GROQ: bool(keys.get(PR.GROQ)), PR.CLAUDE: bool(keys.get(PR.CLAUDE)) and bool(claude_sdk)}


def _why_no_claude(keys, claude_sdk) -> str:
    if not keys.get(PR.CLAUDE):
        return 'there is no Claude key'
    if not claude_sdk:
        return 'the anthropic SDK is not installed'
    return ''


def route(text: str, agent=False, planning=False, keys: dict | None = None, claude_sdk=True) -> Route | None:
    """The provider and model for one request; None when no provider can be used."""
    keys = keys or {}
    ok = usable(keys, claude_sdk)
    tier, reason = classify(text, agent, planning)
    if tier == DEEP:
        if ok[PR.CLAUDE]:
            return Route(*DEEP_MODEL, tier, reason)
        if ok[PR.GROQ]:
            note = 'Pictures need Claude: this answer works from the figures alone.' if reason == 'image' else ''
            return Route(*FAST_GROQ, tier, reason, fallback=f'Claude Opus 5.5 would answer this, but '
                                                           f'{_why_no_claude(keys, claude_sdk)}: GPT-OSS 120B on Groq '
                                                           f'answers instead.', note=note)
        return None
    if ok[PR.GROQ]:
        return Route(*FAST_GROQ, tier, reason)
    if ok[PR.CLAUDE]:
        return Route(*FAST_CLAUDE, tier, reason)
    return None


def alternative(r: Route, keys: dict | None = None, claude_sdk=True) -> Route | None:
    """The other tier's model for a "re-ask with …" chip (None when there is no other model to ask)."""
    ok = usable(keys or {}, claude_sdk)
    if r.tier == FAST or (r.provider, r.model) != DEEP_MODEL:
        if ok[PR.CLAUDE] and (r.provider, r.model) != DEEP_MODEL:
            return Route(*DEEP_MODEL, DEEP, 'reask')
        return None
    if ok[PR.GROQ]:
        return Route(*FAST_GROQ, FAST, 'reask')
    if ok[PR.CLAUDE]:
        return Route(*FAST_CLAUDE, FAST, 'reask')
    return None


def default_auto(fresh_install: bool, keys: dict) -> bool:
    """Automatic is off for existing users; on for a new installation when both keys exist."""
    return bool(fresh_install and keys.get(PR.GROQ) and keys.get(PR.CLAUDE))


HANDOVER_CHARS = 2400


def handover_text(transcript_md: str, from_provider: str, extra: str = '', cap=HANDOVER_CHARS) -> str:
    """A short local summary of the conversation so far, from its Markdown transcript: the latest exchanges first kept
    (each question and the start of its answer), capped. Sent once in front of the first question to the other
    provider; never the other provider's raw history."""
    turns, cur = [], None
    for line in (transcript_md or '').splitlines():
        if line.startswith('### You'):
            cur = {'q': [], 'a': []}
            turns.append(cur)
            mode = 'q'
        elif line.startswith('### Assistant'):
            if cur is None:
                cur = {'q': [], 'a': []}
                turns.append(cur)
            mode = 'a'
        elif cur is not None and line.strip():
            cur[mode].append(line.strip())
    bits = []
    used = 0
    for t in reversed(turns):
        q = ' '.join(t['q'])[:300]
        a = ' '.join(t['a'])
        a = a[:420] + ('…' if len(a) > 420 else '')
        item = f'- You asked: {q}' + (f'\n  Answer: {a}' if a else '')
        if used + len(item) > cap:
            break
        bits.append(item)
        used += len(item)
    bits.reverse()
    skipped = len(turns) - len(bits)
    who = PR.LABEL.get(from_provider, from_provider)
    head = (f'[Earlier in this conversation (with {who}; a summary made by Vantage, the full figures are in the tools'
            + (f'; {skipped} earlier exchange{"s" if skipped != 1 else ""} left out' if skipped else '') + '):')
    body = '\n'.join(bits) if bits else '- (nothing yet)'
    if extra:
        body += '\n' + extra
    return f'{head}\n{body}\n]'
