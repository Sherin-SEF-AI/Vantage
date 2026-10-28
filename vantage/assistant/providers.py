"""The two providers the assistant can use, chosen in Preferences ▸ Assistant: Groq (an OpenAI-compatible API,
client.py) and Claude (Anthropic's Messages API through the official anthropic SDK, claude.py). Each has its own key
(secrets.py), its own model list and its own usage and cost.

Prices are constants with their source and date: Claude's from Anthropic's model documentation; Groq's are not
bundled (no source was available to this build), so a Groq conversation shows its tokens and a cost only when the
user enters Groq's prices in Preferences."""
from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field

from . import client as C

GROQ, CLAUDE = 'groq', 'claude'
PROVIDERS = (GROQ, CLAUDE)
LABEL = {GROQ: 'Groq', CLAUDE: 'Claude'}
KEYS_URL = {GROQ: C.KEYS_URL, CLAUDE: 'https://platform.claude.com/'}
KEY_HINT = {GROQ: 'gsk_…  paste your Groq API key', CLAUDE: 'sk-ant-…  paste your Anthropic API key'}
INSTALL = ('The Claude provider needs Anthropic\'s Python SDK, which this installation does not have. From source: '
           'pip install --user "anthropic==2.3.0" (or pip install ".[claude]" in the source folder); the downloadable '
           'bundles include it.')

CLAUDE_DEFAULT = 'claude-opus-5-5'
CLAUDE_MODELS = {
    'claude-opus-5-5': C.ModelInfo('claude-opus-5-5', 'Claude Opus 5.5', 'claude', ('medium', 'high'), 'high', 0,
                                   'Most capable · adaptive thinking'),
    'claude-sonnet-5-5': C.ModelInfo('claude-sonnet-5-5', 'Claude Sonnet 5.5', 'claude', ('medium', 'high'), 'high',
                                     10, 'Faster, lower cost · adaptive thinking'),
}
# USD per million tokens: input, output, 5-minute cache write, cache read. Source: Anthropic's model documentation
# (the model-migration guide's "Migrating to Claude Opus 5.5" and "Migrating to Claude Sonnet 5.5"), read 2026-10-28.
PRICES = {
    'claude-opus-5-5': {'input': 4.00, 'output': 20.00, 'cache_write': 5.00, 'cache_read': 0.20},
    'claude-sonnet-5-5': {'input': 2.00, 'output': 10.00, 'cache_write': 2.50, 'cache_read': 0.20},
}
PRICE_SOURCE = {CLAUDE: 'Anthropic\'s model documentation, read 2026-10-28 (USD per million tokens)',
                GROQ: 'your entry in Preferences ▸ Assistant (Groq\'s prices are not bundled)'}


def claude_available() -> bool:
    """The anthropic SDK can be imported (an optional dependency)."""
    try:
        return importlib.util.find_spec('anthropic') is not None
    except (ImportError, ValueError):
        return False


def models(provider) -> list:
    return list(CLAUDE_MODELS) if provider == CLAUDE else list(C.KNOWN)


def model_info(provider, model_id) -> C.ModelInfo:
    if provider == CLAUDE:
        return CLAUDE_MODELS.get(model_id) or C.ModelInfo(model_id, model_id, 'claude', ('medium', 'high'), 'high', 60)
    return C.model_info(model_id)


def default_model(provider) -> str:
    return CLAUDE_DEFAULT if provider == CLAUDE else C.DEFAULT_MODEL


@dataclass
class Usage:
    """Tokens of one conversation with one provider (Groq: prompt/completion/reasoning; Claude: input, output and
    the prompt cache's reads and writes), and the cost when the prices are known."""
    prompt: int = 0
    completion: int = 0
    reasoning: int = 0
    seconds: float = 0.0
    cache_read: int = 0
    cache_write: int = 0
    requests: int = 0
    provider: str = GROQ
    model: str = ''
    by_model: dict = field(default_factory=dict)        # model -> [input, output, cache_write, cache_read]

    def add(self, u: dict):
        """A Groq usage dict (OpenAI field names)."""
        if not u:
            return
        self.prompt += int(u.get('prompt_tokens') or 0)
        self.completion += int(u.get('completion_tokens') or 0)
        det = u.get('completion_tokens_details') or {}
        self.reasoning += int(det.get('reasoning_tokens') or 0)
        self.seconds += float(u.get('total_time') or 0.0)
        self.requests += 1
        row = self.by_model.setdefault(self.model, [0, 0, 0, 0])
        row[0] += int(u.get('prompt_tokens') or 0)
        row[1] += int(u.get('completion_tokens') or 0)

    def add_claude(self, model, input_tokens=0, output_tokens=0, cache_write=0, cache_read=0):
        self.prompt += int(input_tokens or 0) + int(cache_write or 0) + int(cache_read or 0)
        self.completion += int(output_tokens or 0)
        self.cache_read += int(cache_read or 0)
        self.cache_write += int(cache_write or 0)
        self.requests += 1
        row = self.by_model.setdefault(model or self.model, [0, 0, 0, 0])
        row[0] += int(input_tokens or 0)
        row[1] += int(output_tokens or 0)
        row[2] += int(cache_write or 0)
        row[3] += int(cache_read or 0)

    @property
    def tokens(self) -> int:
        return self.prompt + self.completion

    def cost(self, groq_prices: dict | None = None) -> float | None:
        """USD, or None when a price is unknown."""
        total = 0.0
        for m, (i, o, w, r) in self.by_model.items():
            if self.provider == CLAUDE:
                pr = PRICES.get(m)
            else:
                pr = groq_prices if groq_prices and groq_prices.get('input') is not None else None
            if pr is None:
                return None
            total += (i * pr['input'] + o * pr['output'] + w * pr.get('cache_write', pr['input']) +
                      r * pr.get('cache_read', pr['input'])) / 1e6
        return total

    def line(self, groq_prices: dict | None = None) -> str:
        """'12.3k tokens in (9.8k from the cache), 1.2k out · ≈ $0.04': the panel's usage line."""
        if not self.requests:
            return ''
        bits = f'{self.prompt / 1000:.1f}k tokens in'
        if self.provider == CLAUDE:
            bits += f' ({self.cache_read / 1000:.1f}k cache_read_input_tokens)'
        bits += f', {self.completion / 1000:.1f}k out'
        c = self.cost(groq_prices)
        if c is not None:
            bits += f' · ≈ ${c:.3f}' if c < 1 else f' · ≈ ${c:.2f}'
        elif self.provider == GROQ:
            bits += ' · cost not estimated (enter Groq\'s prices in Preferences)'
        return bits
