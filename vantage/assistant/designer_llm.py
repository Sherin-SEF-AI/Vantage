"""Requirements read out of a brief and documents: the design agent's first step.

The brief the user types and the documents they add (core.docs_ingest: sentences with ids and page/paragraph
locations) become three lists, each row citing the exact sentence it came from:

- requirements in core.requirements item form (metric, params, op, target, priority, text), each validated by
  requirements.validate_item against the project;
- constraints that no study measures (core.requirements.CONSTRAINT_KINDS: a budget, sensor counts, a mounting
  tolerance to survive);
- statements Vantage cannot check (an operating temperature, a regulation's paperwork, a speed limit of the ODD …),
  with their quote and why, never silently dropped.

Two ways to read them:

- the LLM: Groq or Claude through the providers the assistant uses (the user's key from Preferences ▸ Assistant; the
  text of the brief and documents is sent to that provider, nothing else of the project). It answers through one tool,
  report_requirements, whose input schema is strict JSON (params as key/value pairs so every object is closed);
- deterministic, with no LLM: requirements.parse_brief and requirements.parse_constraints on every sentence.

Numbers are never the LLM's: every target and constraint value must appear in the sentence it quotes (in the
metric's unit, after a unit conversion the code makes: km to m, h to min, $6k to 6000 …); the quote must be the
cited sentence's own words. A row that fails either check is rejected with the reason and shown; an unknown metric
makes the row "not checkable by Vantage". Sentences that read like requirements (a modal verb, a figure with a unit)
and that nothing cites are added to the not-checkable list by the code, so nothing in the documents is lost. The only
targets without a figure in the quote are the app's own conventions for "no …" statements (0 issues, 0 gaps; "no
blind ground" is ≤ 5 cm², the Issues list's threshold), and only when the quote says "no", "none", "zero" or "without".

No Qt here. The network is used only by extract() with a provider and a key; tests point it at the fake servers."""
from __future__ import annotations

import copy
import json
import math
import re
import threading
import time
from dataclasses import dataclass, field

from ..core import requirements as RQ

SCHEMA_ID = 'vantage.designer_extraction/1'
TOOL = 'report_requirements'
MAX_TOKENS = 16000
TIMEOUT_S = 120.0
METHODS = ('deterministic', 'groq', 'claude')
_NEG = re.compile(r'\b(no|none|zero|without|never|nothing|free\s+of|every|all|each)\b', re.I)
_MODAL = re.compile(r'\b(shall|must|should|required?|requires|needs?\s+to|has\s+to|have\s+to|is\s+to|are\s+to|'
                    r'at\s+least|at\s+most|no\s+more\s+than|minimum|maximum|not\s+exceed|within|ideally|'
                    r'target)\b', re.I)

# The tool's input: strict JSON (every object closed and every field required), as Anthropic's strict tool use and
# Groq's tool calls both take it.
_SRC = {'source_id': {'type': 'string', 'description': 'The id of the sentence it comes from, e.g. "D1-p2-3.1"'},
        'quote': {'type': 'string', 'description': 'The words of that sentence it rests on, copied exactly'}}
SCHEMA = {
    'type': 'object',
    'properties': {
        'requirements': {'type': 'array', 'items': {'type': 'object', 'properties': {
            **_SRC,
            'metric': {'type': 'string', 'description': 'A metric id from the catalogue'},
            'params': {'type': 'array', 'items': {'type': 'object', 'properties': {
                'key': {'type': 'string'}, 'value': {'type': 'string'}}, 'required': ['key', 'value'],
                'additionalProperties': False}},
            'op': {'type': 'string', 'enum': list(RQ.OPS)},
            'target': {'type': 'number', 'description': 'The figure, in the metric\'s unit; it must appear in the quote'},
            'priority': {'type': 'string', 'enum': list(RQ.PRIORITIES)},
            'text': {'type': 'string', 'description': 'The requirement in a few words'}},
            'required': ['source_id', 'quote', 'metric', 'params', 'op', 'target', 'priority', 'text'],
            'additionalProperties': False}},
        'constraints': {'type': 'array', 'items': {'type': 'object', 'properties': {
            **_SRC,
            'kind': {'type': 'string', 'enum': list(RQ.CONSTRAINT_KINDS)},
            'op': {'type': 'string', 'enum': list(RQ.OPS)},
            'value': {'type': 'number'},
            'unit': {'type': 'string', 'description': 'USD/EUR/GBP for a budget, mm or ° for a tolerance, "" for counts'},
            'priority': {'type': 'string', 'enum': list(RQ.PRIORITIES)}},
            'required': ['source_id', 'quote', 'kind', 'op', 'value', 'unit', 'priority'],
            'additionalProperties': False}},
        'not_checkable': {'type': 'array', 'items': {'type': 'object', 'properties': {
            **_SRC, 'why': {'type': 'string'}}, 'required': ['source_id', 'quote', 'why'],
            'additionalProperties': False}}},
    'required': ['requirements', 'constraints', 'not_checkable'],
    'additionalProperties': False}

SYSTEM = """You read requirements documents for an autonomous-driving data-collection sensor rig and report what they
require, through the report_requirements tool only. Rules:
- Every row cites one sentence by its id (the [id] before it) and quotes the words it rests on exactly as written.
- requirements: only what one metric of the catalogue measures, in that metric's unit, with its parameters from the
  listed choices ("" means every sector / every class / the worst sensor). The target is a figure written in the quote
  (convert the unit when needed: km to m, hours to minutes), never one you assume or round.
- constraints: a budget (kind budget, the amount and currency as written), sensor counts (max_cameras, min_cameras,
  max_lidars, max_radars, max_sensors) and a mounting tolerance the rig must survive (tolerance_mm, tolerance_deg).
- not_checkable: every other statement that reads like a requirement (an operating temperature, a regulation, a
  speed, a process step), with why the catalogue cannot check it.
- "shall"/"must" is priority must; "should"/"ideally"/"preferably" is should.
- Report nothing that is not in the text."""


def _catalogue_text(project) -> str:
    """The metrics of this build with their units and parameter choices, for the model's instructions."""
    lines = []
    for mid, m in sorted(RQ.METRICS.items()):
        if not RQ.available(m.study):
            continue
        prm = []
        for p_ in m.params:
            try:
                ch = [str(k) for k, _ in p_.options(project)]
            except Exception:
                ch = []
            prm.append(f'{p_.key} in {ch[:12]}')
        lines.append(f'- {mid}: {m.label} [{m.unit or "count"}], natural op {m.better}'
                     + (f'; params: {"; ".join(prm)}' if prm else '') + (f'. {m.help}' if m.help else ''))
    return '\n'.join(lines)


def user_prompt(docs, project) -> str:
    out = ['METRIC CATALOGUE', _catalogue_text(project), '', 'SENTENCES']
    for d in docs:
        out.append(f'# {d.name} ({d.kind})')
        out += [f'[{s.id}] {s.text}' for s in d.sentences]
    out.append('')
    out.append(f'Report every requirement, constraint and uncheckable statement with {TOOL}.')
    return '\n'.join(out)


# =========================================================================== results
@dataclass
class Extraction:
    requirements: list = field(default_factory=list)     # [{'item', 'source', 'metric_label', 'origin', 'accepted'}]
    constraints: list = field(default_factory=list)      # [{'kind', 'op', 'value', 'unit', 'priority', 'text', 'source',
    #                                                       'origin', 'accepted'}]
    not_checkable: list = field(default_factory=list)    # [{'source', 'why', 'origin'}]
    rejected: list = field(default_factory=list)         # [{'row', 'source', 'why'}]
    method: str = 'deterministic'
    model: str = ''
    usage: dict = field(default_factory=dict)
    seconds: float = 0.0
    documents: list = field(default_factory=list)        # docs_ingest summaries
    notes: list = field(default_factory=list)

    def to_dict(self):
        return {'schema': SCHEMA_ID, 'method': self.method, 'model': self.model, 'usage': dict(self.usage),
                'seconds': round(self.seconds, 2), 'requirements': copy.deepcopy(self.requirements),
                'constraints': copy.deepcopy(self.constraints), 'not_checkable': copy.deepcopy(self.not_checkable),
                'rejected': copy.deepcopy(self.rejected), 'documents': copy.deepcopy(self.documents),
                'notes': list(self.notes)}

    @staticmethod
    def from_dict(d) -> 'Extraction':
        d = d if isinstance(d, dict) else {}
        ex = Extraction()
        for k in ('requirements', 'constraints', 'not_checkable', 'rejected', 'documents', 'notes'):
            v = d.get(k)
            setattr(ex, k, [x for x in v if isinstance(x, (dict, str))] if isinstance(v, list) else [])
        ex.method = d.get('method') if d.get('method') in METHODS else 'deterministic'
        ex.model = str(d.get('model') or '')[:80]
        ex.usage = d.get('usage') if isinstance(d.get('usage'), dict) else {}
        try:
            ex.seconds = float(d.get('seconds') or 0.0)
        except (TypeError, ValueError):
            ex.seconds = 0.0
        return ex

    def summary(self) -> dict:
        return {'method': self.method, 'model': self.model, 'requirements': len(self.requirements),
                'constraints': len(self.constraints), 'not_checkable': len(self.not_checkable),
                'rejected': len(self.rejected)}


def _src(doc, s, quote=None) -> dict:
    return {'id': s.id, 'doc': doc.name, 'location': s.location(), 'quote': quote if quote is not None else s.text,
            'sentence': s.text, 'sha1': s.sha1}


# =========================================================================== the figures a quote holds
def _norm(t: str) -> str:
    t = (t or '').replace('’', "'").replace('‘', "'").replace('“', '"').replace('”', '"')
    t = t.replace('–', '-').replace('—', '-').replace('−', '-').replace(' ', ' ')
    return re.sub(r'\s+', ' ', t).strip().strip('.;:,').lower()


def quote_in(quote: str, sentence: str) -> bool:
    q = _norm(quote)
    return bool(q) and q in _norm(sentence)


_UNIT_OF = {'m': (('m', 1.0), ('mm', 1e-3)), 'mm': (('mm', 1.0), ('m', 1e3)), 'm²': (('m2', 1.0),),
            'm³': (('m3', 1.0),), 'ms': (('ms', 1.0),), 's': (('ms', 1e-3),), 'min': (('min', 1.0), ('h', 60.0)),
            'h': (('h', 1.0), ('min', 1.0 / 60.0)), '%': (('pct', 1.0),), 'px': (('px', 1.0),),
            'px/cm': (('pxcm', 1.0),), 'W': (('w', 1.0),)}
_DEG = re.compile(r'(\d+(?:[.,]\d+)?)\s*(?:°|deg\b|degrees?\b)', re.I)
_RAW = re.compile(r'(?<![\w.])(\d+(?:[.,]\d+)?)(?![\d])')


def _close(a, b) -> bool:
    return abs(a - b) <= 1e-9 + 1e-6 * max(abs(a), abs(b))


def figures(quote: str, unit: str) -> list:
    """The values the quote states in a metric's unit (after the code's unit conversions)."""
    out = []
    if unit == '°':
        out += [float(m.group(1).replace(',', '.')) for m in _DEG.finditer(quote)]
    elif unit in _UNIT_OF:
        for v, u, _, _ in RQ._quantities(quote):
            for uu, f in _UNIT_OF[unit]:
                if u == uu:
                    out.append(round(v * f, 9))
    else:                                          # counts, ×, Hz, GB/h …: the numbers as written
        out += [float(m.group(1).replace(',', '.')) for m in _RAW.finditer(quote)]
        out += [float(v) for w, v in RQ._WORDNUM.items() if re.search(r'\b' + w + r'\b', quote, re.I) and v > 0]
    return out


def implicit_ok(metric: str, target: float, quote: str) -> bool:
    """A target with no figure in the quote: only the app's conventions for a "no …" statement."""
    if not _NEG.search(quote or ''):
        return False
    if _close(target, 0.0):
        return True
    return metric == 'coverage.blind_m2' and _close(target, 0.0005) and bool(re.search(r'\bblind\b', quote, re.I))


def target_ok(metric: str, target, quote: str) -> tuple[bool, str]:
    m = RQ.METRICS.get(metric)
    t = RQ._num(target)
    if m is None or t is None:
        return False, 'no target'
    if any(_close(t, v) for v in figures(quote, m.unit)):
        return True, ''
    if implicit_ok(metric, t, quote):
        return True, ''
    stated = figures(quote, m.unit)
    return False, (f'the figure {t:g} {m.unit} is not in the quoted source'
                   + (f' (it states {", ".join(f"{v:g}" for v in stated)} {m.unit})' if stated else ''))


def constraint_ok(kind: str, value, unit: str, quote: str) -> tuple[bool, str]:
    v = RQ._num(value)
    if v is None:
        return False, 'no value'
    if kind == 'budget':
        vals = [RQ.money_value(mt) for mt in RQ._MONEY.finditer(quote)]
        if any(_close(v, a) for a, _ in vals):
            return True, ''
        return False, f'the amount {v:g} is not in the quoted source' + (
            f' (it states {", ".join(f"{a:g} {c}" for a, c in vals)})' if vals else '')
    if kind == 'tolerance_mm':
        ok = any(_close(v, x) for x in figures(quote, 'mm'))
        return ok, '' if ok else f'the tolerance {v:g} mm is not in the quoted source'
    if kind == 'tolerance_deg':
        ok = any(_close(v, x) for x in figures(quote, '°'))
        return ok, '' if ok else f'the tolerance {v:g}° is not in the quoted source'
    vals = figures(quote, '')
    ok = any(_close(v, x) or _close(v, x - 1) for x in vals) or (_close(v, 0.0) and bool(_NEG.search(quote)))
    return ok, '' if ok else f'the count {v:g} is not in the quoted source'


# =========================================================================== validating what the model returned
def _clean_constraint(c, source, origin) -> dict:
    return {'kind': c['kind'], 'op': c['op'] if c.get('op') in RQ.OPS else '<=', 'value': float(c['value']),
            'unit': str(c.get('unit') or '')[:8], 'priority': c.get('priority') if c.get('priority') in RQ.PRIORITIES
            else 'must', 'text': str(c.get('text') or source.get('quote') or '')[:200], 'source': source,
            'origin': origin, 'accepted': True}


def validate(raw, docs, project, method='groq', model='') -> Extraction:
    """The model's report, checked row by row (see the module doc). Unknown fields are ignored; a malformed row is
    rejected with the reason, never dropped."""
    from ..core import docs_ingest as DI
    ex = Extraction(method=method, model=model, documents=DI.summary(docs)['documents'])
    raw = raw if isinstance(raw, dict) else {}
    taken = set()

    def locate(row):
        sid = str(row.get('source_id') or '')
        doc, s = DI.find(docs, sid)
        quote = str(row.get('quote') or '')
        if s is None:
            return None, f'no sentence {sid!r} in the documents'
        if not quote_in(quote, s.text):
            return None, f'the quote is not the words of {sid} ({doc.name}, {s.location()})'
        return _src(doc, s, quote.strip()), ''
    for row in raw.get('requirements') or [] if isinstance(raw.get('requirements'), list) else []:
        if not isinstance(row, dict):
            ex.rejected.append({'row': row, 'source': None, 'why': 'not an object'})
            continue
        src, why = locate(row)
        if src is None:
            ex.rejected.append({'row': row, 'source': None, 'why': why})
            continue
        params = row.get('params')
        if isinstance(params, list):
            params = {str(p.get('key')): str(p.get('value') if p.get('value') is not None else '')
                      for p in params if isinstance(p, dict) and p.get('key')}
        elif not isinstance(params, dict):
            params = {}
        mid = str(row.get('metric') or '')
        if mid not in RQ.METRICS or not RQ.metric_available(mid):
            ex.not_checkable.append({'source': src, 'why': f'no Vantage metric for it (the model named {mid!r})',
                                     'origin': method})
            continue
        item = {'metric': mid, 'params': params, 'op': row.get('op'), 'target': row.get('target'),
                'priority': row.get('priority'), 'text': row.get('text') or src['quote'],
                'note': f'Read from {src["doc"]} ({src["location"]}).'}
        ok, msg = RQ.validate_item(item, project)
        if not ok:
            ex.rejected.append({'row': row, 'source': src, 'why': msg})
            continue
        ok, msg = target_ok(mid, item['target'], src['quote'])
        if not ok:
            ex.rejected.append({'row': row, 'source': src, 'why': msg})
            continue
        it = RQ.clean_item(item, project, taken)
        taken.add(it['id'])
        ex.requirements.append({'item': it, 'source': src, 'metric_label': RQ.METRICS[mid].label, 'origin': method,
                                'accepted': True})
    for row in raw.get('constraints') or [] if isinstance(raw.get('constraints'), list) else []:
        if not isinstance(row, dict):
            ex.rejected.append({'row': row, 'source': None, 'why': 'not an object'})
            continue
        src, why = locate(row)
        if src is None:
            ex.rejected.append({'row': row, 'source': None, 'why': why})
            continue
        if row.get('kind') not in RQ.CONSTRAINT_KINDS or row.get('op') not in RQ.OPS:
            ex.rejected.append({'row': row, 'source': src, 'why': f'unknown constraint {row.get("kind")!r} or '
                                                                  f'operator {row.get("op")!r}'})
            continue
        ok, msg = constraint_ok(row['kind'], row.get('value'), row.get('unit') or '', src['quote'])
        if not ok:
            ex.rejected.append({'row': row, 'source': src, 'why': msg})
            continue
        ex.constraints.append(_clean_constraint(row, src, method))
    for row in raw.get('not_checkable') or [] if isinstance(raw.get('not_checkable'), list) else []:
        if not isinstance(row, dict):
            continue
        src, why = locate(row)
        if src is None:                          # still shown: the model's own words, flagged
            ex.rejected.append({'row': row, 'source': None, 'why': why})
            continue
        ex.not_checkable.append({'source': src, 'why': str(row.get('why') or 'not checkable')[:300], 'origin': method})
    _dedupe(ex)
    cover_unmapped(ex, docs)
    return ex


def _dedupe(ex: Extraction):
    """One row per requirement or constraint; a second source saying the same is kept as the row's 'also'."""
    def merge(rows, key):
        seen, keep = {}, []
        for r in rows:
            k = key(r)
            if k in seen:
                seen[k].setdefault('also', []).append(r['source'])
            else:
                seen[k] = r
                keep.append(r)
        return keep
    ex.requirements = merge(ex.requirements, lambda r: (r['item']['metric'], json.dumps(r['item']['params'],
                                                                                         sort_keys=True),
                                                        r['item']['op'], r['item']['target']))
    ex.constraints = merge(ex.constraints, lambda c: (c['kind'], c['op'], c['value']))


def _quote_from(source: str, sentence: str) -> str:
    """The parser's phrase as the sentence's own words: the phrase itself when the sentence holds it; a phrase it
    joined from fragments ('Robotaxi pedestrians at 80 m', 'to 40 m in the front-left … front-right sectors') as the
    span of the sentence from its first words to its last; else its longest tail in the sentence; else the sentence."""
    if quote_in(source, sentence):
        return source.strip()
    words = (source or '').split()
    low = sentence.lower()

    def find(ws, last=False):
        pat = r'\s+'.join(re.escape(w.lower()) for w in ws)
        hits = list(re.finditer(pat, low))
        return (hits[-1] if last else hits[0]) if hits else None
    head = next((m for k in range(len(words), 0, -1) if (m := find(words[:k])) is not None), None)
    tail = next((m for k in range(len(words), 0, -1) if (m := find(words[-k:], last=True)) is not None), None)
    if head is not None and tail is not None and tail.end() > head.start() and \
            (head.end() - head.start()) + (tail.end() - tail.start()) >= min(len(source), 12):
        return sentence[head.start():tail.end()].strip()
    for i in range(len(words)):
        q = ' '.join(words[i:])
        if len(q) >= 3 and quote_in(q, sentence):
            return q
    return sentence


def requirement_like(text: str) -> bool:
    return bool(_MODAL.search(text) or RQ._quantities(text) or RQ._MONEY.search(text))


def cover_unmapped(ex: Extraction, docs):
    """Every sentence that reads like a requirement and that no row cites goes to not_checkable (origin 'code')."""
    cited = {r['source']['id'] for r in ex.requirements + ex.constraints + ex.not_checkable if r.get('source')}
    cited |= {a['id'] for r in ex.requirements + ex.constraints for a in r.get('also') or [] if a}
    cited |= {r['source']['id'] for r in ex.rejected if r.get('source')}
    for d in docs:
        for s in d.sentences:
            if s.id not in cited and requirement_like(s.text):
                ex.not_checkable.append({'source': _src(d, s), 'origin': 'code',
                                         'why': 'reads like a requirement but no Vantage metric or constraint matched '
                                                'it'})


# =========================================================================== deterministic (no LLM)
def deterministic(docs, project) -> Extraction:
    """requirements.parse_brief and parse_constraints on every sentence: each proposal cites its sentence; its figures
    are the parser's own reading of the words (checked like the model's)."""
    from ..core import docs_ingest as DI
    t0 = time.perf_counter()
    ex = Extraction(method='deterministic', documents=DI.summary(docs)['documents'])
    taken = set()
    for d in docs:
        for s in d.sentences:
            for o in RQ.parse_brief(s.text, project):
                it = dict(o['item'])
                src = _src(d, s, _quote_from(o['source'], s.text))
                q = src['quote']
                it['text'] = (q[0].upper() + q[1:])[:300] if q else it['text']
                ok, why = target_ok(it['metric'], it['target'], q)
                if not ok:                       # the parser misread: shown, never used
                    ex.rejected.append({'row': it, 'source': src, 'why': why})
                    continue
                ok, why = RQ.validate_item(it, project)
                if not ok:
                    ex.rejected.append({'row': it, 'source': src, 'why': why})
                    continue
                it['note'] = f'Read from {d.name} ({s.location()}) by the brief parser.'
                if it['id'] in taken:
                    it['id'] = RQ._new_id(taken)
                taken.add(it['id'])
                ex.requirements.append({'item': it, 'source': src, 'metric_label': o['metric_label'],
                                        'origin': 'parser', 'accepted': True})
            for c in RQ.parse_constraints(s.text, project):
                src = _src(d, s, _quote_from(c['source'], s.text))
                ok, why = constraint_ok(c['kind'], c['value'], c['unit'], src['quote'])
                if not ok:
                    ex.rejected.append({'row': c, 'source': src, 'why': why})
                    continue
                ex.constraints.append(_clean_constraint(c, src, 'parser'))
    _dedupe(ex)
    cover_unmapped(ex, docs)
    ex.seconds = time.perf_counter() - t0
    return ex


# =========================================================================== the LLM call
class ExtractionError(Exception):
    pass


def _parse_json(text: str):
    t = (text or '').strip()
    mt = re.search(r'```(?:json)?\s*(.*?)```', t, re.S)
    if mt:
        t = mt.group(1)
    i, j = t.find('{'), t.rfind('}')
    if i < 0 or j <= i:
        raise ExtractionError('The model answered without the report (no JSON).')
    try:
        return json.loads(t[i:j + 1])
    except ValueError as e:
        raise ExtractionError(f'The model\'s report is not valid JSON: {e}') from None


def _call_groq(key, model, system, user, url=None, cancel=None):
    from . import client as C
    cl = C.Client(key, url, timeout=TIMEOUT_S)
    payload = {'model': model, 'messages': [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}],
               'tools': [{'type': 'function', 'function': {'name': TOOL, 'description': 'Report what the documents '
                                                                                       'require.',
                                                           'parameters': SCHEMA}}],
               'tool_choice': {'type': 'function', 'function': {'name': TOOL}}, 'max_completion_tokens': MAX_TOKENS}
    payload.update(C.reasoning_params(model, 'medium', False))
    args, content, usage = {}, [], {}
    for dl in cl.stream(payload, cancel):
        content.append(dl.content or '')
        for tc in dl.tool_calls or []:
            k = tc.get('index', 0)
            fn = tc.get('function') or {}
            a = args.setdefault(k, {'name': '', 'arguments': ''})
            a['name'] += fn.get('name') or ''
            a['arguments'] += fn.get('arguments') or ''
        if dl.usage:
            usage = {'input_tokens': dl.usage.get('prompt_tokens'), 'output_tokens': dl.usage.get('completion_tokens')}
    call = next((a for a in args.values() if a['name'] == TOOL), None)
    return _parse_json(call['arguments'] if call else ''.join(content)), usage


def _call_claude(key, model, system, user, url=None, cancel=None):
    from . import claude as CL
    from .client import ApiError
    if CL.anthropic is None:
        raise ExtractionError('The Claude provider needs Anthropic\'s Python SDK (anthropic), which this installation '
                              'does not have.')
    cl = CL.anthropic.Anthropic(api_key=key.strip(), base_url=url or CL.base_url(),
                                timeout=CL.anthropic.Timeout(TIMEOUT_S, connect=15.0))
    tool = {'name': TOOL, 'description': 'Report what the documents require.', 'input_schema': SCHEMA, 'strict': True}
    kw = {'model': model, 'max_tokens': MAX_TOKENS, 'system': [{'type': 'text', 'text': system}],
          'messages': [{'role': 'user', 'content': [{'type': 'text', 'text': user}]}], 'tools': [tool],
          'tool_choice': {'type': 'auto'}, 'thinking': {'type': 'adaptive'}, 'output_config': {'effort': 'medium'}}
    try:
        with cl.beta.messages.stream(**kw) as stream:
            for _ in stream:
                if cancel is not None and cancel.is_set():
                    raise ApiError('cancelled', 'Stopped.')
            msg = stream.get_final_message()
    except CL.anthropic.AnthropicError as e:
        raise CL.map_error(e) from None
    if msg.stop_reason == 'refusal':
        raise ExtractionError('Claude declined to read the documents (a refusal).')
    use = next((b for b in msg.content if getattr(b, 'type', None) == 'tool_use' and b.name == TOOL), None)
    u = msg.usage
    usage = {'input_tokens': getattr(u, 'input_tokens', None), 'output_tokens': getattr(u, 'output_tokens', None)}
    if use is not None:
        inp = use.input
        return (inp if isinstance(inp, dict) else _parse_json(str(inp))), usage
    return _parse_json(''.join(getattr(b, 'text', '') for b in msg.content)), usage


def extract(docs, project, provider=None, key=None, model=None, url=None, cancel=None) -> Extraction:
    """The requirements of the documents: by the LLM when a provider and a key are given (raises ExtractionError or
    client.ApiError when it fails: the caller says so and may fall back to deterministic), else deterministic."""
    if not provider or not key:
        return deterministic(docs, project)
    if provider not in ('groq', 'claude'):
        raise ExtractionError(f'Unknown provider {provider!r}.')
    from . import providers as PR
    model = model or PR.default_model(provider)
    if isinstance(cancel, threading.Event):
        ev = cancel
    else:
        ev = threading.Event()
        if callable(cancel) and cancel():
            ev.set()
    t0 = time.perf_counter()
    user = user_prompt(docs, project)
    raw, usage = (_call_groq if provider == 'groq' else _call_claude)(key, model, SYSTEM, user, url, ev)
    ex = validate(raw, docs, project, provider, model)
    ex.usage = usage
    ex.seconds = time.perf_counter() - t0
    if ex.rejected:
        ex.notes.append(f'{len(ex.rejected)} row{"s" * (len(ex.rejected) != 1)} of the model\'s report rejected '
                        '(listed with the reason).')
    return ex
