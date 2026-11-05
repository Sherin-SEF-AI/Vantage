"""What every page of the app is for, what it needs, how far the user has got with it and what to do there next.

The assistant's recommendations (recommend.page_rules) and the model's context (the ToolContext's page block and the
get_page_context tool) read this module, so the assistant knows the page the user is on and suggests what to do there.
No Qt, no network, nothing computed: a page's state is read off what the app already holds (the page's last result and
its core module's summary(), its settings via settings_of(project), whether it runs, whether its result is older than
the design), never by running a study here.

GUIDES has one entry per page id of mainwindow.PAGE_CLASSES and the built-in views ('viewports': the 3D and top views):
- purpose: one plain sentence, what the page is for;
- needs: its prerequisites, each a small check on the project and the window (a variant with cameras, a lidar, a
  recording with ground truth, CARLA set up, a downloaded model, torch, accepted requirements, fleet records …) and the
  page or button that provides what is missing;
- where its last result lives (an attribute of the page, or a controller on the app context), the core module whose
  summary() reads it, the page's own stale and busy flags, its run button (the widget attribute, the text on it) and
  whether it runs by itself after edits while it shows;
- long: the work is long or needs CARLA, the GPU or training: the recommendation is a plan the user approves (the
  Copilot's approval path), never a run;
- before / after: the pages that usually come before and after it (the workflow), with the button that hands over
  where there is one.

page_state(page, project, window_like) -> dict: compact and JSON-able (no paths, no keys), < 30 ms a page: the
summary is computed once per result object and cached.
page_recommendations(page, state, context) -> [Recommendation]: deterministic; what is missing first (open the page or
button that provides it), else Run (open, pointing at the page's own button; a plan for long work), else the result's
top findings as questions with the figures read off the result, the next page of the workflow and the export; and
"What can I do here?" on every page. Nothing runs by itself: every item is ask, plan or open (see recommend.py)."""
from __future__ import annotations

import importlib
import importlib.util
import math
import os
import re
import time
from dataclasses import dataclass, field

SOURCE = 'Page guide'
PREFIX = 'page:'                 # recommend.PAGE_PREFIX: every item's id starts with it
BUDGET_MS = 30.0                 # page_state's budget per page (the tests hold it to this)
STATUSES = ('needs', 'running', 'never_run', 'stale', 'done', 'ready')


@dataclass(frozen=True)
class Need:
    """A prerequisite: check(project, window_like) is True when it is met, False when it is missing, None when it
    cannot be told from here (no window): only a False is reported."""
    id: str
    text: str                    # what is missing, as it reads after "needs": "a lidar on the variant"
    check: object
    how: str = ''                # where to get it: "add one from the Library (Sensors)"
    fix: dict | None = None      # the open target that provides it: {'page', 'focus'}


@dataclass(frozen=True)
class Next:
    """A workflow neighbour: the page after this one, why, and the button that hands over (on this page or there)."""
    page: str
    why: str
    focus: str = ''              # a button attribute on the page opened (target page, or this page when here=True)
    here: bool = False           # the button is on this page (e.g. "Use as mounting tolerance")


@dataclass(frozen=True)
class Guide:
    page: str
    label: str
    purpose: str                 # one plain sentence
    needs: tuple = ()
    result: str = ''             # the last result: an attribute of the page ('result'), or of the app context
                                 # ('ctx:perception.result', 'ctx:analysis.result'); '' = a tool page with no result
    module: str = ''             # the core module whose summary(result) reads it
    page_summary: bool = False   # the page's own summary() reads its result (no module summary)
    settings: str = ''           # the core module with settings_of(project)
    run: str = ''                # the page's run button (widget attribute) …
    run_text: str = ''           # … and the text on it
    auto: bool = False           # works itself out after edits while the page shows (no Run needed)
    busy: str = 'busy'           # the page's "running" flag
    stale: str = ''              # the page's "result is older than the design" flag ('' = none kept)
    long: str = ''               # why a run is long work (CARLA, the GPU, training): a plan, never a run
    plan_kind: str = ''          # the Copilot's step kind for it (copilot.KINDS)
    estimate: str = ''           # a label attribute on the page that shows the estimate before a run
    export: str = ''             # the export button (widget attribute)
    before: tuple = ()           # page ids that usually come before
    after: tuple = ()            # Next(...) items
    findings: object = None      # callable(summary, result, project) -> [finding] (else the summary's 'findings')
    ask: str = ''                # a question worth asking once there is a result (a prompt; figures are added)
    result_fn: object = None     # callable(page, window_like, project) -> the result, where no one attribute holds it
    stale_fn: object = None      # callable(page, window_like, project, result) -> bool, where no flag is kept
    rules: bool = False          # recommend.py's own rules already ask about this page's findings: not repeated


# ====================================================================================================== helpers
def _num(v, default=None):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) else default


def _short(text, n=72):
    text = ' '.join(str(text).split())
    if len(text) <= n:
        return text
    cut = text[:n - 1]
    if ' ' in cut[n // 2:]:
        cut = cut[:cut.rindex(' ')]
    return cut.rstrip(' ,;:-–') + '…'


_SECRET = re.compile(r'(key|token|secret|password|passwd|credential|auth)', re.IGNORECASE)
_PATHISH = re.compile(r'(path|file|folder|dir|root|url|uri|command|cmd)s?$', re.IGNORECASE)
_DROP = ('schema', 'assumptions', 'limits', 'references', 'provenance', 'timing', 'notes', 'vantage')


def _looks_like_path(s: str) -> bool:
    s = s.strip()
    return bool(s) and (s.startswith(('/', '~', '\\\\')) or bool(re.match(r'^[A-Za-z]:[\\/]', s)) or
                        '://' in s or (os.sep in s and len(s) > 3 and ' ' not in s))


_GONE = object()                 # compact(): a value left out (a path, a URL)


def compact(o, depth=0, items=4, text=160):
    """A small JSON-able copy (see _compact); None when o itself is a path."""
    out = _compact(o, depth, items, text)
    return None if out is _GONE else out


def _compact(o, depth=0, items=4, text=160):
    """A small JSON-able copy: lists cut to `items` (with the count), strings to `text`, three levels deep; secrets,
    paths and URLs left out; numbers rounded."""
    if isinstance(o, dict):
        if depth >= 3:
            return f'{{{len(o)} entries}}'
        out = {}
        for k, v in o.items():
            k = str(k)
            if k in _DROP or _SECRET.search(k) or _PATHISH.search(k):
                continue
            c = _compact(v, depth + 1, items, text)
            if c is not _GONE:
                out[k] = c
        return out
    if isinstance(o, (list, tuple)):
        if depth >= 3:
            return f'[{len(o)} items]'
        vals = [_compact(v, depth + 1, items, text) for v in list(o)[:items]]
        vals = [v for v in vals if v is not _GONE]
        return vals + ([f'… {len(o) - items} more'] if len(o) > items else [])
    if isinstance(o, bool) or o is None:
        return o
    if isinstance(o, int):
        return o
    if isinstance(o, float):
        return round(o, 4) if math.isfinite(o) else None
    try:
        import numpy as np
        if isinstance(o, np.generic):
            return _compact(o.item(), depth, items, text)
        if isinstance(o, np.ndarray):
            return f'array {tuple(o.shape)}'
    except ImportError:                                  # pragma: no cover
        pass
    if isinstance(o, str):
        if _looks_like_path(o):
            return _GONE
        return _short(o, text)
    return f'<{type(o).__name__}>'          # an object (a result, an array holder): never its repr


def _scrub(o):
    try:
        from ..core import provenance as PV
        return PV.scrub(o)
    except Exception:                                    # noqa: BLE001 — compact() drops paths anyway
        return o


def _variant(project):
    return getattr(project, 'variant', None)


def _cams(project):
    v = _variant(project)
    return [c for c in getattr(v, 'cameras', []) or [] if getattr(c, 'visible', True)]


def _sensors(project, kind=None):
    v = _variant(project)
    out = [s for s in getattr(v, 'sensors', []) or [] if getattr(s, 'visible', True)]
    return [s for s in out if kind is None or s.kind == kind]


def _ctx(win):
    return getattr(win, 'ctx', None)


def _qsetting(win, key, default=''):
    st = getattr(_ctx(win), 'settings', None)
    if st is None:
        return None
    try:
        return st.value(key, default)
    except Exception:                                    # noqa: BLE001
        return None


def _settings_of(module, project):
    try:
        return importlib.import_module(f'vantage.core.{module}').settings_of(project)
    except Exception:                                    # noqa: BLE001 — junk settings: the page's defaults
        return None


_SPEC: dict = {}


def _have(mod: str) -> bool:
    """An optional library is installed (found, not imported: importing torch takes seconds)."""
    if mod not in _SPEC:
        try:
            _SPEC[mod] = importlib.util.find_spec(mod) is not None
        except (ImportError, ValueError):
            _SPEC[mod] = False
    return _SPEC[mod]


def _get(obj, dotted: str):
    for part in dotted.split('.'):
        if obj is None:
            return None
        obj = getattr(obj, part, None)
    return obj


def _holder(g: Guide, pg, win):
    """(the object that holds the result, the attribute path on it)."""
    if g.result.startswith('ctx:'):
        return _ctx(win), g.result[4:]
    return pg, g.result


def _flag(spec: str, pg, win) -> bool:
    """A yes/no the app keeps: 'busy' (an attribute of the page), 'ctx:perception.busy' (of a controller on the app
    context), 'job:tolerance' (a main-window job of that name is running); every one of them '|'-joined is an or."""
    for part in (spec or '').split('|'):
        if not part:
            continue
        if part.startswith('job:'):
            v = part[4:] in (getattr(win, '_jobs', None) or {})
        elif part.startswith('ctx:'):
            v = _get(_ctx(win), part[4:]) is True
        else:
            v = pg is not None and _get(pg, part)
            v = v is True or (isinstance(v, str) and bool(v))     # a page's busy can name the job ('build')
        if v:
            return True
    return False


def _result_of(g: Guide, pg, win, project=None):
    if g.result_fn is not None:
        try:
            return g.result_fn(pg, win, project)
        except Exception:                                # noqa: BLE001
            return None
    if not g.result:
        return None
    obj, attr = _holder(g, pg, win)
    return _get(obj, attr) if obj is not None else None


_CACHE: dict = {}                # page -> (the result object, its compact summary, its findings)


def _summary(g: Guide, res, pg, project) -> tuple[dict, list]:
    """The result's compact summary and its findings, computed once per result object (a new run is a new object)."""
    hit = _CACHE.get(g.page)
    if hit is not None and hit[0] is res:
        return hit[1], hit[2]
    full = None
    try:
        if g.page_summary and pg is not None and hasattr(pg, 'summary'):
            full = pg.summary()
        elif g.module:
            full = importlib.import_module(f'vantage.core.{g.module}').summary(res)
        elif isinstance(res, dict):
            full = res
    except Exception as e:                               # noqa: BLE001 — an odd result never breaks the panel
        full = {'error': f'{type(e).__name__}: {e}'}
    full = full if isinstance(full, dict) else {}
    try:
        found = (g.findings(full, res, project) if g.findings is not None else None)
    except Exception:                                    # noqa: BLE001
        found = None
    if found is None:
        found = generic_findings(full)
    out = compact(_scrub(full))
    found = [compact(_scrub(f), items=6, text=300) for f in found[:3]]
    _CACHE[g.page] = (res, out, found)
    return out, found


SEV_ORDER = {'critical': 0, 'fault': 0, 'fail': 0, 'serious': 1, 'warning': 2, 'info': 3, 'good': 4}


def generic_findings(summary: dict) -> list:
    """The summary's own findings, the most severe first: [{'severity', 'message', 'key'}]. Strings count as
    warnings (the studies that list plain sentences only list what is worth saying)."""
    raw = summary.get('findings') if isinstance(summary, dict) else None
    out = []
    for i, f in enumerate(raw if isinstance(raw, list) else []):
        if isinstance(f, dict):
            msg = str(f.get('message') or f.get('text') or '').strip()
            sev = str(f.get('severity') or 'warning')
        else:
            msg, sev = str(f).strip(), 'warning'
        if msg and SEV_ORDER.get(sev, 3) <= 2:
            out.append((SEV_ORDER.get(sev, 3), i, {'severity': sev, 'message': msg}))
    out.sort(key=lambda t: (t[0], t[1]))
    return [f for _, _, f in out]


def finding(message, severity='warning', title='', prompt='', **figures) -> dict:
    """A finding for an ask item: the message (figures read off the result), and optionally the item's title and the
    prompt sent (else they are made from the message)."""
    d = {'severity': severity, 'message': str(message)}
    if title:
        d['title'] = str(title)
    if prompt:
        d['prompt'] = str(prompt)
    if figures:
        d['figures'] = figures
    return d


def _check(n: Need, project, win):
    try:
        v = n.check(project, win)
    except Exception:                                    # noqa: BLE001 — a check that cannot tell is not a block
        return None
    return None if v is None else bool(v)


def _settings_doc(g: Guide, project) -> dict:
    """The page's settings in brief: plain values only (numbers, switches, short words), no paths, at most 12."""
    if not g.settings:
        return {}
    s = _settings_of(g.settings, project)
    try:
        d = s.to_dict() if s is not None else {}
    except Exception:                                    # noqa: BLE001
        d = {}
    out = {}
    for k, v in d.items():
        if len(out) >= 12:
            break
        if _SECRET.search(str(k)) or _PATHISH.search(str(k)):
            continue
        if isinstance(v, bool) or v is None or isinstance(v, int):
            out[k] = v
        elif isinstance(v, float):
            out[k] = round(v, 4) if math.isfinite(v) else None
        elif isinstance(v, str) and len(v) <= 40 and not _looks_like_path(v):
            out[k] = v
        elif isinstance(v, (list, tuple)) and len(v) <= 6 and all(isinstance(x, (int, float, str)) and
                                                                  not (isinstance(x, str) and _looks_like_path(x))
                                                                  for x in v):
            out[k] = list(v)
        elif isinstance(v, (list, tuple, dict)):
            out[k] = f'{len(v)} item' + ('' if len(v) == 1 else 's')
    return out


def _text_of(w) -> str:
    try:
        return ' '.join(str(w.text()).split())
    except Exception:                                    # noqa: BLE001
        return ''


# ====================================================================================================== the state
def page_state(page: str, project, window_like=None) -> dict:
    """What the app knows about the page now, read off what it already holds (never computed here): {'page', 'label',
    'purpose', 'opened', 'status' (needs | running | never_run | stale | done | ready), 'missing': [{'id', 'text',
    'how', 'fix'}], 'running', 'stale', 'has_result', 'auto', 'long', 'run': {'button', 'text'}, 'estimate',
    'summary' (the result's summary(), compact), 'findings' (its top three), 'settings' (plain values), 'before',
    'after', 'ms'}. JSON-able; no paths, no keys."""
    t0 = time.perf_counter()
    g = GUIDES.get(page) or Guide(page, str(page), '')
    pages = getattr(window_like, '_page_widgets', None) or {}
    pg = pages.get(page)
    missing = []
    for n in g.needs:
        if _check(n, project, window_like) is False:
            missing.append({'id': n.id, 'text': n.text, 'how': n.how, 'fix': dict(n.fix) if n.fix else None})
    res = _result_of(g, pg, window_like, project)
    running = _flag(g.busy, pg, window_like)
    stale = bool(res is not None and _flag(g.stale, pg, window_like))
    if res is not None and not stale and g.stale_fn is not None:
        try:
            stale = bool(g.stale_fn(pg, window_like, project, res))
        except Exception:                                # noqa: BLE001
            stale = False
    summary, found = ({}, []) if res is None else _summary(g, res, pg, project)
    if running:
        status = 'running'
    elif missing and res is None:
        status = 'needs'
    elif not g.result and g.result_fn is None:
        status = 'needs' if missing else 'ready'
    elif res is None:
        status = 'never_run'
    elif stale:
        status = 'stale'
    else:
        status = 'done'
    est = _text_of(getattr(pg, g.estimate, None)) if (pg is not None and g.estimate) else ''
    out = {'page': page, 'label': g.label, 'purpose': g.purpose, 'opened': pg is not None, 'status': status,
           'missing': missing, 'running': running, 'stale': stale, 'has_result': res is not None, 'auto': g.auto,
           'long': g.long, 'run': {'button': g.run, 'text': g.run_text} if g.run else None,
           'estimate': _short(est, 140) if est else '', 'summary': summary, 'findings': found,
           'settings': _settings_doc(g, project),
           'before': [{'page': b, 'label': _label(b)} for b in g.before],
           'after': [{'page': a.page, 'label': _label(a.page), 'why': a.why} for a in g.after]}
    out['headline'] = headline(out)
    out['ms'] = round((time.perf_counter() - t0) * 1000, 2)
    return out


def _label(page) -> str:
    g = GUIDES.get(page)
    return g.label if g is not None else str(page)


def headline(st: dict) -> str:
    """One line: where the user is with the page."""
    s = st.get('status')
    if s == 'running':
        return 'Running now.'
    if s == 'needs':
        return 'Cannot run yet: it needs ' + '; '.join(m['text'] for m in st.get('missing', [])[:3]) + '.'
    if s == 'never_run':
        return ('Works itself out while the page shows; nothing yet.' if st.get('auto') else
                'Not run yet on this design.')
    if s == 'stale':
        return 'Has a result, older than the design (run it again to update it).'
    if s == 'done':
        f = st.get('findings') or []
        if f:
            return f'Has a result; top finding: {_short(f[0].get("message", ""), 120)}'
        return 'Has a result with nothing serious in it.'
    return 'A tool page: nothing to run.' if not st.get('missing') else 'Needs ' + st['missing'][0]['text'] + '.'


# ====================================================================================================== recommendations
# the items' base priorities (recommend.SEVERITY['info'] × these; the open page adds recommend.PAGE_BOOST): above the
# starters, below the evidence rules about the page (a warning or worse), and below every critical item (recommend)
VALUE = {'need': 1.3, 'run': 1.2, 'finding': 1.15, 'next': 0.9, 'export': 0.7, 'help': 0.62}


def _plan_label(kind: str) -> str:
    try:
        from . import copilot as CP
        return CP.KIND_LABEL.get(kind, kind)
    except Exception:                                    # noqa: BLE001
        return kind


def page_recommendations(page: str, state: dict, context=None) -> list:
    """The page's own items for the recommendation list and the live strip, from its page_state; deterministic."""
    from . import recommend as RC
    g = GUIDES.get(page)
    if g is None or not isinstance(state, dict):
        return []
    label = state.get('label') or g.label
    about = (('page', page),)
    out = []

    def rec(what, title, why, kind, value, **kw):
        out.append(RC._rec(f'{PREFIX}{page}:{what}', title, why, kind, 'info', value, source=SOURCE, about=about,
                           **kw))

    status = state.get('status')
    missing = state.get('missing') or []
    # what is missing first: the page or button that provides it
    for i, m in enumerate(missing[:2]):
        fix = m.get('fix') or None
        how = m.get('how') or ''
        rec(f'need:{m["id"]}', _short(f'{label} needs {m["text"]}' + (f': {how}' if how else ''), 90),
            f'{label} cannot run without {m["text"]}' + (f'; {how}.' if how else '.'),
            'open' if fix else 'ask', VALUE['need'] - 0.02 * i,
            target=dict(fix) if fix else None,
            prompt='' if fix else f'The {label} page needs {m["text"]}. How do I get it, and what should I set up '
                                  f'first?', evidence={'missing': m['id']})
    est = state.get('estimate') or ''
    if status in ('never_run', 'stale') and not missing:
        why = ('Not run on this design yet.' if status == 'never_run' else
               'Its result is older than the design: run it again to update the figures.')
        if est:
            why += f' {est}'
        if g.long:
            kind = g.plan_kind
            what = f'a {_plan_label(kind)} step ({kind})' if kind else f'the {label} run'
            rec('plan', _short(f'Plan the {label} run for your approval', 80),
                f'{why} It needs {g.long}: the Copilot writes the plan with its estimate, and nothing runs until you '
                f'approve it.', 'plan', VALUE['run'],
                prompt=f'Plan this for my approval with propose_plan: {what} with the {label} page\'s settings as '
                       f'they are now. Say what it needs, how long it takes and what it will tell me.',
                evidence={'status': status})
        elif g.run and not (g.auto and status == 'never_run'):     # an auto page works its first result out itself
            rec('run', _short(f'Run {label}: press {g.run_text}' if g.run_text else f'Run {label}', 80), why,
                'open', VALUE['run'], target={'page': page, 'focus': g.run}, evidence={'status': status})
        elif status == 'stale' and g.auto:
            pass                                            # it updates itself while the page shows
    # the result's top findings, as questions with the figures read off it
    if state.get('has_result') and status != 'running' and not g.rules:
        for i, f in enumerate((state.get('findings') or [])[:3]):
            msg = str(f.get('message', ''))
            if not msg:
                continue
            key = re.sub(r'[\d.,]+', '#', msg)[:60]
            title = f.get('title') or f'Ask about: {_short(msg, 60)}'
            prompt = f.get('prompt') or (f'On the {label} page: "{msg}". What does it mean for the rig and the data, '
                                         f'and what should I change?')
            sev = str(f.get('severity', 'warning'))
            rec(f'finding:{key}', _short(title, 80), f'{label}: {_short(msg, 200)}', 'ask',
                VALUE['finding'] - 0.03 * i - (0.02 if sev not in ('critical', 'serious', 'fault', 'fail') else 0),
                prompt=prompt, evidence={'message': msg})
        if g.ask and not (state.get('findings') or []):
            rec('question', _short(g.ask.split('\n')[0], 70), f'{label} has a result to read.', 'ask',
                VALUE['finding'] - 0.1, prompt=g.ask, evidence={'status': status})
    # the next page of the workflow (once there is a result, or on a tool page)
    if status in ('done', 'stale', 'ready') and g.after:
        for i, a in enumerate(g.after[:1]):
            tgt = {'page': page if a.here else a.page}
            if a.focus:
                tgt['focus'] = a.focus
            rec(f'next:{a.page}', _short(f'Next: {_label(a.page)}', 70), _short(a.why, 200), 'open',
                VALUE['next'] - 0.02 * i, target=tgt, evidence={'status': status})
    if state.get('has_result') and g.export and status in ('done', 'stale'):
        rec('export', f'Export the {label} result', 'Writes the result as the page\'s export (JSON and its other '
                                                   'files) for a report or another tool.', 'open', VALUE['export'],
            target={'page': page, 'focus': g.export}, evidence={'status': status})
    rec('help', 'What can I do here?', _short(f'{g.purpose}', 200), 'ask', VALUE['help'],
        prompt=f'I am on the {label} page. What is it for, what can I do here with my design as it is now, and what '
               f'should I do first? (get_page_context has the page\'s state.)')
    return out


def context_line(pc: dict, limit: int = 1500) -> str:
    """The open page for the model's per-question context: id, label, purpose and the page's state in brief."""
    import json
    if not isinstance(pc, dict) or not pc.get('page'):
        return ''
    keep = {k: pc.get(k) for k in ('status', 'headline', 'missing', 'findings', 'estimate', 'settings')
            if pc.get(k)}
    if pc.get('summary'):
        keep['summary'] = pc['summary']
    rec = [r.get('title') for r in (pc.get('recommended') or [])][:4]
    if rec:
        keep['recommended'] = rec
    text = json.dumps(keep, ensure_ascii=False, separators=(',', ':'), default=str)
    if len(text) > limit:
        keep.pop('summary', None)
        keep.pop('settings', None)
        text = json.dumps(keep, ensure_ascii=False, separators=(',', ':'), default=str)[:limit]
    return (f'[Open page: {pc.get("label")} (id {pc["page"]}): {pc.get("purpose", "")} Page state: {text}. '
            f'get_page_context has the details.]')


def guide_of(page_or_label: str) -> Guide | None:
    """The guide of a page id or of its label ('Recording budget')."""
    if page_or_label in GUIDES:
        return GUIDES[page_or_label]
    return next((g for g in GUIDES.values() if g.label == page_or_label), None)


# ====================================================================================================== prerequisites
def _has_cameras(p, w):
    return bool(_cams(p))


def _has(kind):
    return lambda p, w: bool(_sensors(p, kind))


def _has_lidar_or_radar(p, w):
    return bool(_sensors(p, 'lidar') or _sensors(p, 'radar'))


def _has_gnss_or_imu(p, w):
    return bool(_sensors(p, 'gnss') or _sensors(p, 'imu'))


def _has_vehicle_mesh(p, w):
    v = getattr(p, 'vehicle', None)
    return None if v is None else getattr(v, 'mesh', None) is not None


def _analysis_done(p, w):
    an = getattr(_ctx(w), 'analysis', None)
    return None if an is None else an.result is not None


def _two_variants(p, w):
    return len(getattr(p, 'variants', []) or []) >= 2


def _requirements(p, w):
    return bool(((getattr(p.analysis, 'requirements', None) or {}).get('items')) or [])


def _fleet_vehicles(p, w):
    f = getattr(p, 'fleet', None)
    return bool(isinstance(f, dict) and f.get('vehicles'))


def _targets(p, w):
    return bool(getattr(p, 'targets', None))


def _carla_root(p, w):
    """CARLA is set up on this machine (Simulate ▸ CARLA ▸ its folder); unknown without the window."""
    root = _qsetting(w, 'sim/carla_root', '')
    sim = (getattr(w, '_page_widgets', None) or {}).get('sim')
    if root is None and sim is None:
        return None
    from ..sim import server as SV
    return SV.is_carla_root(str(root or '')) or (sim is not None and SV.is_carla_root(getattr(sim, 'carla_root', '')))


def _yolox(p, w):
    """The detector (YOLOX) is in the models folder with its published size (the SHA-256 is checked when it loads)."""
    from ..core import detection as DT
    sp = DT.spec_of(DT.DEFAULT_MODEL)
    path = DT.model_path(sp)
    return os.path.isfile(path) and os.path.getsize(path) == sp.size


def _torch(p, w):
    return _have('torch')


def _torch_transformers(p, w):
    return _have('torch') and _have('transformers')


def _settings_list(module, attr):
    def check(p, w):
        s = _settings_of(module, p)
        return None if s is None else bool(getattr(s, attr, None))
    return check


_ADD_SENSOR = 'Vehicle ▸ Add lidar / Add radar (or the Library\'s Sensors)'
NEED_CAMERAS = Need('cameras', 'a camera on the variant', _has_cameras,
                    'place one from the Library on the 3D and top views', {'page': 'viewports'})
NEED_LIDAR = Need('lidar', 'a lidar on the variant', _has('lidar'), 'Vehicle ▸ Add lidar', {'page': 'viewports'})
NEED_RANGE_SENSOR = Need('lidar_or_radar', 'a lidar or a radar on the variant', _has_lidar_or_radar, _ADD_SENSOR,
                         {'page': 'viewports'})
NEED_ANALYSIS = Need('analysis', 'the coverage analysis', _analysis_done, 'press F5 (Analysis ▸ Run now)',
                     {'page': 'viewports', 'point': 'analysis'})
NEED_MESH = Need('vehicle', 'a vehicle model', _has_vehicle_mesh, 'File ▸ Import vehicle model',
                 {'page': 'viewports'})
NEED_CARLA = Need('carla', 'CARLA set up on this computer', _carla_root,
                  'choose the CARLA folder on the CARLA page', {'page': 'sim'})
NEED_YOLOX = Need('detector', 'the YOLOX detector downloaded', _yolox,
                  'press Download on the Detection quality page', {'page': 'detection', 'focus': 'b_download'})
NEED_TORCH = Need('torch', 'PyTorch installed in Vantage\'s Python', _torch,
                  'install torch (the downloads leave it out)', None)
NEED_REQUIREMENTS = Need('requirements', 'written requirements', _requirements,
                         'write them on the Requirements scorecard', {'page': 'requirements', 'focus': 'b_new'})
NEED_FLEET = Need('fleet', 'vehicles with measured sensor poses', _fleet_vehicles,
                  'import them on the Fleet page', {'page': 'fleet', 'focus': 'b_import'})
NEED_TARGETS = Need('targets', 'calibration boards placed', _targets,
                    'place them on Calibrate ▸ Targets', {'page': 'targets'})


# ====================================================================================================== findings
def _f_timing(s, res, p):
    out = []
    w = _num(s.get('worst_misalignment_ms'))
    if w is not None and s.get('lidar'):
        sug = _num(s.get('suggested_worst_misalignment_ms'))
        lid = (s.get('lidar') or {}).get('tag', 'the lidar')
        more = f'; the suggested trigger groups bring it to {sug:.2f} ms' if sug is not None and sug < w else ''
        out.append(finding(f'The worst camera fires {w:.2f} ms away from {lid}\'s sweep '
                           f'({_num(s.get("worst_travel_cm"), 0):.1f} cm of travel at speed){more}.',
                           'warning' if w > 1.0 else 'info',
                           title=f'Ask how to cut the {w:.2f} ms trigger misalignment',
                           prompt=f'The sync plan\'s worst camera fires {w:.2f} ms away from {lid}\'s sweep{more}. '
                                  f'Which cameras, why, and how should I group the trigger lines?',
                           worst_ms=round(w, 2)))
    elif not s.get('lidar'):
        out.append(finding('No spinning lidar: the plan shows motion blur and rolling-shutter skew only.', 'info'))
    return out


def _f_perception(s, res, p):
    out = []
    for cls, secs in (s.get('per_class_sector') or {}).items():
        for sec, v in (secs or {}).items():
            req = _num(v.get('required_m'), 0.0)
            if not v.get('pass', True) and req:
                got = _num(v.get('min_m'), 0.0)
                out.append((got / req, finding(
                    f'{str(cls).capitalize()} in the {sec} sector: detected from {got:.1f} m, {req:g} m required.',
                    'serious', title=f'Ask why {str(cls).lower()} range is short {sec}: {got:.1f} of {req:g} m',
                    prompt=f'Perception range: a {str(cls).lower()} in the {sec} sector is detected from {got:.1f} m '
                           f'at its worst azimuth, short of the {req:g} m required. Which camera limits it and what '
                           f'change reaches the required range?', min_m=round(got, 1))))
    out.sort(key=lambda t: t[0])
    return [f for _, f in out]


def _f_coverage(s, res, p):
    out = []
    for cls, secs in (s.get('sectors') or {}).items():
        for sec, v in (secs or {}).items():
            if v.get('verdict') == 'gap':
                out.append(finding(f'{str(cls).capitalize()} {sec}: no sensor reaches the required '
                                   f'{_num(v.get("required_m"), 0):g} m (camera {v.get("camera_m")}, lidar '
                                   f'{v.get("lidar_m")}, radar {v.get("radar_m")} m).', 'serious',
                                   title=f'Ask how to close the {str(cls).lower()} gap {sec}'))
    return out


def _f_volume(s, res, p):
    m = getattr(res, 'metrics', {}) or {}
    out = []
    for name, o in (m.get('objects') or {}).items():
        a = _num((o or {}).get('undetected_m2'), 0.0)
        if a > 0.01:
            out.append(finding(f'A {name} ({_num(o.get("height"), 0):g} m) can stand unseen on {a:.2f} m² around the '
                               f'car.', 'serious', title=f'Ask where a {name} can stand unseen ({a:.2f} m²)'))
    return out


def _f_tolerance(s, res, p):
    b = getattr(res, 'blind', None)
    nominal = _num(getattr(res, 'nominal_blind', None))
    try:
        import numpy as np
        b = np.asarray(b, float)
        p95 = float(np.percentile(b, 95)) if b.size else None
    except Exception:                                    # noqa: BLE001
        p95 = None
    if p95 is None:
        return []
    return [finding(f'Under the mounting tolerance the blind ground reaches {p95:.3f} m² in 95 % of builds '
                    f'(nominal {nominal or 0:.3f} m²).', 'warning' if p95 > (nominal or 0) + 0.01 else 'info',
                    title=f'Ask which camera\'s mounting error costs most ({p95:.3f} m² at p95)',
                    prompt=f'The tolerance study gives {p95:.3f} m² blind ground at the 95th percentile against '
                           f'{nominal or 0:.3f} m² nominal. Which camera and direction drive it, and how do I make '
                           f'the rig robust to it?', p95_m2=round(p95, 3))]


def _f_compare(s, res, p):
    rows = sorted((v['blind_m2'], name) for name, v in (res or {}).items() if v.get('blind_m2') is not None)
    if len(rows) < 2:
        return []
    (b0, n0), (b1, n1) = rows[0], rows[-1]
    return [finding(f'{n0} has the least blind ground ({b0:.3f} m²), {n1} the most ({b1:.3f} m²).', 'info',
                    title=f'Ask which variant to keep: {_short(n0, 24)} or another',
                    prompt=f'Compare my variants: {n0} has {b0:.3f} m² blind ground, {n1} {b1:.3f} m². Which should I '
                           f'keep and why (coverage, density, robustness to the mounting tolerance)?')]


def _f_camviews(s, res, p):
    seams = sorted(getattr(res, 'seams', None) or [], key=lambda x: -float(x.get('vanish_m2', 0.0)))
    if not seams or float(seams[0].get('vanish_m2', 0.0)) <= 0.005:
        return []
    a, b = seams[0].get('tags', ('?', '?'))
    v = float(seams[0]['vanish_m2'])
    return [finding(f'A standing object vanishes on {v:.3f} m² at the {a}–{b} seam of the stitched view.', 'serious',
                    title=f'Ask where to move the {a}–{b} seam ({v:.3f} m² vanish)')]


def _f_optimiser(s, res, p):
    front = getattr(res, 'front', None) or []
    n = getattr(res, 'evals', None) or len(getattr(res, 'evaluated', []) or [])
    if not front:
        return []
    return [finding(f'The search evaluated {n} rigs; {len(front)} are on the Pareto front.', 'info',
                    title='Ask which Pareto candidate to apply and why',
                    prompt=f'The optimiser evaluated {n} rigs and has {len(front)} on its Pareto front. Which one '
                           f'should I apply as a new variant, and what does it trade off?')]


def _f_power(s, res, p):
    out = generic_findings(s)
    if not out:
        load = _num(s.get('dcdc_peak_load_pct'))
        rt = _num(s.get('engine_off_runtime_min'))
        if load is not None:
            out.append(finding(f'The DC-DC converter runs at {load:.0f} % at peak; engine-off runtime '
                               f'{rt or 0:.0f} min.', 'info'))
    return out


def _f_requirements(s, res, p):
    out = []
    items = sorted((it for it in (s.get('items') or []) if it.get('status') == 'fail'),
                   key=lambda it: it.get('priority') != 'must')
    for it in items[:3]:
        out.append(finding(f'Requirement {it.get("id")} ({it.get("priority")}) fails: {it.get("value")} against '
                           f'{it.get("target")}' + (' (from an earlier run)' if it.get('stale') else '') + '.',
                           'serious' if it.get('priority') == 'must' else 'warning',
                           title=f'Ask how to meet {it.get("id")}: {_short(it.get("text", ""), 40)}',
                           prompt=f'Requirement {it.get("id")} "{it.get("text", "")}" fails ({it.get("value")} against '
                                  f'{it.get("target")}). Why, and what is the smallest change that meets it? Check it '
                                  f'with simulate_changes.'))
    if not out and s.get('not_evaluated'):
        out.append(finding(f'{s["not_evaluated"]} requirement(s) not evaluated yet: Run evaluates their studies.',
                           'warning'))
    return out


def _f_detection(s, res, p):
    out = []
    for rec in s.get('recordings') or []:
        for c in rec.get('classes') or []:
            r = _num(c.get('recall'))
            if r is not None and (c.get('objects') or 0) >= 5:
                out.append((r, finding(f'{c["class"]} recall {r:.2f} on {c["objects"]} objects in '
                                       f'{_short(rec.get("recording", ""), 40)} (detection range '
                                       f'{c.get("detection_range") or "—"}).', 'serious' if r < 0.5 else 'warning',
                                       title=f'Ask why {c["class"]} recall is {r:.2f}',
                                       prompt=f'Detection quality: {c["class"]} recall is {r:.2f} on {c["objects"]} '
                                              f'objects ({rec.get("recording")}). Where does the detector miss them '
                                              f'(range, camera, height) and what would raise it?', recall=round(r, 2))))
    out.sort(key=lambda t: t[0])
    return [f for _, f in out[:2]]


def _region_name(grid, rid):
    try:
        sr = grid.parse_id(rid)
        return grid.region_name(*sr) if sr else rid
    except Exception:                                    # noqa: BLE001
        return rid


def _f_rig_perception(s, res, p):
    """The weakest region of each class for the reference rig (enough objects), then the significant comparisons."""
    out = []
    try:
        ref = res.rigs[0]['name']
        min_gt = int(res.settings.min_gt)
        cells = []
        for cls in res.classes:
            for rid, c in ((res.metrics.get(ref) or {}).get(cls) or {}).items():
                r = _num((c or {}).get('recall'))
                if rid == 'all' or r is None or (c.get('n_gt') or 0) < min_gt:
                    continue
                cells.append((r, cls, rid, int(c.get('n_gt') or 0)))
        cells.sort()
        seen = set()
        for r, cls, rid, n in cells:
            if cls in seen or r >= 0.9:
                continue
            seen.add(cls)
            name = _region_name(res.grid, rid)
            out.append(finding(f'{ref}: {cls} recall {r:.2f} at {name} ({n} object-instants).',
                               'serious' if r < 0.5 else 'warning', title=f'Ask why {cls} recall drops at {name}',
                               prompt=f'Rig perception: {ref}\'s {cls} recall is {r:.2f} at {name} on {n} '
                                      f'object-instants. Which cameras cover that region, why is recall low there, '
                                      f'and what change to the rig would raise it?', recall=round(r, 2), region=rid))
            if len(out) >= 2:
                break
    except Exception:                                    # noqa: BLE001 — a result of another shape: its findings
        pass
    for t in (s.get('findings') or [])[:3 - len(out)]:
        out.append(finding(str(t), 'warning', title=f'Ask about: {_short(t, 60)}'))
    return out


def _f_foundation(s, res, p):
    out = []
    rows = [r for r in s.get('ranges') or [] if _num(r.get('r50_over_geometric')) is not None]
    rows.sort(key=lambda r: r['r50_over_geometric'])
    for r in rows[:2]:
        out.append(finding(f'{r["model"]} finds a {r["class"]} with {r["camera"]} to {r["r50"]} (50 %), '
                           f'{r["r50_over_geometric"]:.2f}× the geometric {r.get("geometric_m")} m.', 'warning',
                           title=f'Ask why {r["camera"]} sees {r["class"]} only to {_short(r["r50"], 16)}',
                           prompt=f'Foundation models: {r["model"]} detects a {r["class"]} with camera {r["camera"]} '
                                  f'to {r["r50"]} at 50 %, {r["r50_over_geometric"]:.2f}× the geometric pixels-on-'
                                  f'target range ({r.get("geometric_m")} m). Why the gap, and what would close it?'))
    for tag, d in sorted((s.get('depth') or {}).get('cameras', {}).items(),
                         key=lambda kv: _num(kv[1].get('usable_m'), 1e9))[:1]:
        out.append(finding(f'Monocular depth stays usable to {d.get("usable_m")} m with {tag}.', 'info'))
    return out


def _f_robust(s, res, p):
    out = []
    for rr in s.get('recordings') or []:
        for c in ('pedestrian', 'vehicle'):
            d = rr.get(f'{c}_recall') or {}
            nom, lo = _num(d.get('nominal')), _num(d.get('p05'))
            if nom is not None and lo is not None and nom - lo > 0.02:
                out.append((lo - nom, finding(
                    f'{c} recall {nom:.2f} nominal falls to {lo:.2f} in 5 % of builds '
                    f'({_short(rr.get("recording", ""), 30)}).', 'serious' if nom - lo > 0.1 else 'warning',
                    title=f'Ask which tolerance costs most {c} recall ({nom:.2f} → {lo:.2f})',
                    prompt=f'Robust perception: {c} recall is {nom:.2f} nominal but {lo:.2f} in the worst 5 % of '
                           f'builds. Which tolerance source and which camera drive it, and what would tighten it?')))
    out.sort(key=lambda t: t[0])
    return [f for _, f in out[:2]]


def _f_risk(s, res, p):
    out = []
    for cls, d in sorted((s.get('classes') or {}).items(), key=lambda kv: _num(kv[1].get('weighted'), 1.0)):
        w, a = _num(d.get('weighted')), _num(d.get('area'))
        if w is None:
            continue
        out.append(finding(f'{cls}: {w * 100:.0f} % of the pre-crash exposure covered ({(a or 0) * 100:.0f} % of the '
                           f'ground); the largest gap is {d.get("largest_gap")}.',
                           'serious' if w < 0.85 else 'warning' if w < 0.95 else 'info',
                           title=f'Ask how to cover the {cls} gap at {_short(d.get("largest_gap") or "", 24)}',
                           prompt=f'Risk-weighted coverage: the rig covers {w * 100:.0f} % of the {cls} pre-crash '
                                  f'exposure, the largest gap at {d.get("largest_gap")}. Which camera change closes it '
                                  f'best? Check with simulate_changes.', weighted=round(w, 3)))
        if len(out) >= 2:
            break
    return out


def _f_perf_model(s, res, p):
    if not s.get('trained'):
        return [finding(f'Not trained: {s.get("reason") or "too few records"} ({s.get("records", 0)} records).',
                        'warning', title='Ask how many more records the model needs')]
    out = []
    rig = s.get('rig') or {}
    for c, d in sorted(rig.items(), key=lambda kv: -(_num(kv[1].get('std'), 0.0)))[:1]:
        out.append(finding(f'Predicted {c} recall for this variant {_num(d.get("mean"), 0):.2f} ± '
                           f'{_num(d.get("std"), 0):.2f}.', 'info',
                           title=f'Ask how far to trust the {c} prediction (± {_num(d.get("std"), 0):.2f})'))
    for pr in (s.get('proposals') or [])[:1]:
        out.append(finding(f'Evaluate {pr.get("name")} next: {pr.get("why")}', 'info',
                           title=f'Ask why {_short(pr.get("name"), 30)} is the rig to simulate next'))
    return out


def _f_designer(s, res, p):
    opts = s.get('options') or []
    if not opts:
        return []
    o = opts[0]
    return [finding(f'Option {o.get("label")}: {o.get("verdict")}, cost {o.get("cost")} {o.get("currency") or ""}, '
                    f'{o.get("power_w")} W.', 'info', title=f'Ask what option {o.get("label")} trades off',
                    prompt=f'The design agent\'s option {o.get("label")} ({o.get("verdict")}, cost {o.get("cost")}, '
                           f'{o.get("power_w")} W). What does it change, what does its critique say, and should I '
                           f'create it as a variant?')]


def _f_asbuilt(s, res, p):
    out = []
    v = s.get('verdict')
    if v and v != 'holds':
        out.append(finding(f'As built, the plan {str(s.get("verdict_text") or v)}', 'serious' if v == 'does_not_hold'
                           else 'warning', title='Ask what the as-built check found'))
    return out + generic_findings(s)[:2]


def _f_lidar_calib(s, res, p):
    out = []
    if s.get('verdict') not in (None, 'good'):
        out.append(finding(f'{s.get("lidar")}–{s.get("camera")}: {s.get("verdict")}: {s.get("message") or ""}',
                           'serious' if s.get('verdict') in ('serious', 'failed', 'unobservable') else 'warning'))
    return out + generic_findings(s)[:2]


def _f_online_calib(s, res, p):
    out = [finding(f'Knock: {k}', 'serious', title=f'Ask about the knock: {_short(k, 40)}') for k in
           (s.get('knocks') or [])[:2]]
    out += [finding(f'Excursion: {k}', 'warning') for k in (s.get('excursions') or [])[:1]]
    return out


def _f_campaign(s, res, p):
    out = []
    if s.get('headline'):
        out.append(finding(str(s['headline']), 'info', title='Ask what the campaign shows'))
    for f in (s.get('false_alarms') or [])[:1]:
        if f.get('flagged'):
            out.append(finding(f'{f.get("check")} flags {f.get("sensor")} on {f.get("flagged")} of '
                               f'{f.get("clean_drives")} clean drives.', 'warning'))
    return out


def _f_critical_carla(s, res, p):
    a = _num(s.get('agreement_pct'))
    if a is None:
        return []
    return [finding(f'CARLA agrees with the critical-targets study on {a:.0f} % of the camera views '
                    f'({s.get("camera_views_compared")} compared).', 'warning' if a < 90 else 'info',
                    title=f'Ask why CARLA and the study disagree ({100 - a:.0f} % of views)')]


def _f_experiments(s, res, p):
    out = []
    for r in (s.get('responses') or [])[:1]:
        d = (r.get('drivers') or [{}])[0]
        if d.get('factor'):
            out.append(finding(f'{r.get("label")}: driven most by {d.get("factor")} (score {d.get("score")}).', 'info',
                               title=f'Ask how {_short(d.get("factor"), 24)} drives {_short(r.get("label"), 24)}'))
    if s.get('best'):
        out.append(finding(f'The best design point meets the targets: {bool(s["best"].get("meets_targets"))}.',
                           'info', title='Ask whether to add the best design as a variant'))
    return out


def _f_sim(s, res, p):
    if not isinstance(res, dict):
        return []
    st = res.get('status')
    if st not in (None, 'ok'):
        return [finding(f'The last drive ended {st}: {_short(res.get("error") or "see its report", 160)}', 'serious',
                        title='Ask why the last CARLA drive failed')]
    return []


def _f_dataset(s, res, p):
    return [finding(f'Written: {s.get("samples")} samples, {s.get("size_mb")} MB over {s.get("duration_s")} s.',
                    'info', title='Ask how to load the sample dataset in my training stack')]


def _f_automation(s, res, p):
    st = s.get('status')
    if st in (None, 'ok', 'done'):
        return []
    return [finding(f'The last automation run: {s.get("status_text") or st}', 'warning',
                    title='Ask why the last automation run did not pass')]


# ====================================================================================================== results
def _r_targets(pg, w, p):
    t = _get(_ctx(w), 'analysis.targets')
    return t or None


def _r_compare(pg, w, p):
    """Per variant (by name) the figures the page compares, read off its analysis results."""
    r = getattr(pg, 'results', None)
    if not r:
        return None
    names = {v.uid: v.name for v in getattr(p, 'variants', [])}
    out = {}
    for uid, res in dict(r).items():
        m = getattr(res, 'metrics', None) or {}
        out[names.get(uid, uid)] = {k: _num(m.get(k)) for k in ('blind_m2', 'multi_pct', 'min_density')}
    return out


def _r_install(pg, w, p):
    r = getattr(pg, 'results', None)
    return {'cameras': len(r)} if r else None


def _r_rig(pg, w, p):
    if pg is None or not getattr(pg, 'running', False):
        return None
    return pg.health()


def _r_photo(pg, w, p):
    ps = getattr(p, 'photos', None)
    return {'rms': getattr(ps, 'rms', None)} if ps is not None and getattr(ps, 'solved_at', None) else None


def _r_automation(pg, w, p):
    a = getattr(_ctx(w), 'automation', None)
    if a is None:
        return None
    runs = a.runs() if hasattr(a, 'runs') else []
    return runs[0] if runs else None


def _r_optimiser(pg, w, p):
    o = getattr(pg, 'opt', None)
    return o if o is not None and getattr(o, 'front', None) else None


def _s_requirements(pg, w, p, res):
    shown = getattr(pg, 'res', None)
    return any(getattr(r, 'stale', False) for r in getattr(shown, 'items', []) or [])


def _cheap(module, attr):
    """A settings field that names the input (a recording, a drive): set or not."""
    def check(p, w):
        try:
            M = importlib.import_module(f'vantage.core.{module}')
            if module == 'drivelog':
                s = M.DriveLogSettings.from_dict(getattr(p.analysis, 'drivelog', None) or {})
            else:
                s = M.settings_of(p)
        except Exception:                                # noqa: BLE001
            return None
        return bool(getattr(s, attr, None))
    return check


def _critical_targets(p, w):
    s = _settings_of('critical', p)
    return None if s is None else any(t.get('enabled') for t in (s.targets or []))


def _foundation_input(p, w):
    s = _settings_of('foundation_range', p)
    return None if s is None else (s.source == 'sweep' or bool(s.recordings))


def _designer_reqs(p, w):
    s = _settings_of('rig_designer', p)
    return None if s is None else any(r.get('accepted', True) for r in (s.requirements or []))


def _saved(p, w):
    return bool(getattr(p, 'path', ''))


def _photos(p, w):
    ps = getattr(p, 'photos', None)
    return bool(getattr(ps, 'frames', None))


def _sensing(p, w):
    return bool(_cams(p) or _sensors(p, 'lidar') or _sensors(p, 'radar'))


def _need_recordings(module, page, what='a CARLA drive recorded with ground truth'):
    return Need('recordings', what, _cheap(module, 'recordings'), f'add one on the {_LABELS[page]} page (Add…)',
                {'page': page})


_LABELS = {'detection': 'Detection quality', 'rig_perception': 'Rig perception', 'robust_perception':
           'Robust perception', 'foundation': 'Foundation models'}
NEED_DRIVELOG = Need('recording', 'a recorded drive (a ROS 2 bag or a nuScenes folder)', _cheap('drivelog', 'path'),
                     'press Open bag… or Open folder…', {'page': 'drivelog', 'focus': 'b_bag'})
NEED_ASBUILT = Need('recording', 'a recording of the real rig', _cheap('asbuilt', 'path'),
                    'press Open bag… or Open folder…', {'page': 'asbuilt', 'focus': 'b_bag'})
NEED_ONLINE = Need('recording', 'a recorded drive (a ROS 2 bag or a nuScenes folder)', _cheap('online_calib', 'path'),
                   'open one with Open bag… or Open folder… on the page', {'page': 'online_calib'})
NEED_SPLAT = Need('drive', 'a recorded drive (a nuScenes folder or a ROS 2 bag)', _cheap('splat_reproject', 'drive'),
                  'choose one with Folder… or Bag… on the page', {'page': 'splat'})
NEED_CRITICAL = Need('critical_targets', 'critical targets', _critical_targets,
                     'press Add on Analysis ▸ Critical targets', {'page': 'critical', 'focus': 'b_add'})
NEED_GNSS = Need('gnss_or_imu', 'a GNSS antenna or an IMU', _has_gnss_or_imu, 'press Add GNSS antenna or Add IMU',
                 {'page': 'gnss', 'focus': 'b_add_gnss'})
NEED_TWO_VARIANTS = Need('variants', 'a second variant to compare', _two_variants,
                         'Layout ▸ Duplicate variant, then change it', {'page': 'viewports'})
NEED_SAVED = Need('saved', 'the project saved to a file', _saved, 'File ▸ Save (the export folder is next to it)',
                  None)
NEED_PHOTOS = Need('photos', 'a photo spin set', _photos, 'Load spin set… in the Inspector', {'page': 'photo'})
NEED_DESIGNER = Need('requirements', 'accepted requirements', _designer_reqs,
                     'press Read requirements (from the brief or documents)', {'page': 'designer', 'focus': 'b_read'})
NEED_FOUNDATION = Need('recordings', 'a recording with ground truth (or the range sweep)', _foundation_input,
                       'add one with Add… or choose the sweep', {'page': 'foundation', 'focus': 'b_add_rec'})
NEED_FM_TORCH = Need('torch', 'torch and transformers in Vantage\'s Python', _torch_transformers,
                     'run Vantage from source with "pip install torch transformers"', None)
NEED_SENSING = Need('sensors', 'a camera, lidar or radar to record', _sensing,
                    'place one on the 3D and top views', {'page': 'viewports'})


# ====================================================================================================== the pages
_G = [
    Guide('viewports', '3D and top views',
          'Place and aim the cameras and sensors on the vehicle and see the coverage analysis (blind ground, pixel '
          'density, redundancy) and the Issues list update as you edit.',
          needs=(NEED_CAMERAS,), result='ctx:analysis.result', busy='ctx:analysis._busy', stale='ctx:analysis.stale',
          auto=True, after=(Next('camviews', 'The stitched bird\'s-eye and bowl views, and where standing objects '
                                              'vanish at the seams.'),
                            Next('tolerance', 'How mounting errors move the coverage.'))),
    Guide('camviews', 'Camera views',
          'See every camera\'s image, the stitched bird\'s-eye view and the bowl view, and find seam angles where a '
          'standing object stays visible.', needs=(NEED_CAMERAS, NEED_ANALYSIS), result='ctx:stitched.result',
          busy='ctx:stitched._busy|ctx:stitched.advising', auto=True, rules=True, findings=_f_camviews,
          before=('viewports',), after=(Next('camviews', 'Search seam angles that keep a standing object visible.',
                                             'b_suggest', here=True),)),
    Guide('tolerance', 'Tolerance study',
          'A Monte Carlo of mounting errors (the analysis tolerance) on every camera: the blind ground and density '
          'you can expect over many builds, the sensitivities and the pinned worst cases.', needs=(NEED_CAMERAS,),
          result='result', busy='job:tolerance', run='b_run', run_text='Run study', export='b_csv',
          findings=_f_tolerance, before=('viewports', 'fleet_twin'),
          after=(Next('robust_perception', 'Carry the tolerances to detection with error bars.'),)),
    Guide('optimiser', 'Optimiser',
          'An evolutionary search over the camera poses for a Pareto set of rigs (blind ground, worst case, '
          'density), then apply the one you choose as a new variant.', needs=(NEED_CAMERAS,), result_fn=_r_optimiser,
          busy='job:optimiser', run='b_run', run_text='Start', findings=_f_optimiser, before=('viewports',),
          after=(Next('compare', 'Compare the applied candidate with your other variants.'),)),
    Guide('compare', 'Compare variants',
          'Every variant\'s coverage side by side against a reference: blind ground, redundancy, minimum density and '
          'the worst single-camera error.', needs=(NEED_TWO_VARIANTS,), result_fn=_r_compare, auto=True,
          findings=_f_compare, after=(Next('requirements', 'Check the chosen variant against the requirements.'),)),
    Guide('photo', 'Photo matching',
          'Match the vehicle model to a spin set of reference photos (landmark correspondences, residuals) so the 3D '
          'body matches the real car.', needs=(NEED_PHOTOS,), result_fn=_r_photo,
          after=(Next('viewports', 'Check the matched model on the 3D and top views.'),)),
    Guide('targets', 'Targets',
          'Plan the calibration boards on the floor around the car so every camera, and every neighbouring pair, '
          'sees enough of them.', needs=(NEED_CAMERAS,), result_fn=_r_targets,
          after=(Next('observability', 'How firmly the boards pin every camera\'s pose.'),
                 Next('install', 'Check the installed cameras against the boards.'))),
    Guide('install', 'Installation check',
          'Capture each camera, find the floor boards, solve the installed pose, compare it with the plan and stitch '
          'the bird\'s-eye view with the planned and measured poses.', needs=(NEED_CAMERAS, NEED_TARGETS),
          result_fn=_r_install, run='b_run', run_text='Run check', export='b_report', before=('targets', 'rig'),
          after=(Next('fleet', 'Record the measured poses per vehicle (Import… takes the installation report).',
                      'b_import'),)),
    Guide('volume', 'Blind volume',
          'Where around the vehicle something can stand unseen by every camera, in 3D, and how tall an object must be '
          'to be seen.', needs=(NEED_ANALYSIS, NEED_MESH), result='ctx:volume.result', busy='ctx:volume.busy',
          stale='ctx:volume.stale', run='b_run', run_text='Run', auto=True, export='b_export', findings=_f_volume,
          before=('viewports',), after=(Next('critical', 'Check the objects the rig must resolve at known places.'),)),
    Guide('perception', 'Perception range',
          'How far out the rig resolves cars, pedestrians, cyclists and signs in each direction against the ranges '
          'the dataset needs, the stereo overlaps, and every weather condition.', needs=(NEED_ANALYSIS,),
          result='ctx:perception.result', module='perception', busy='ctx:perception.busy',
          stale='ctx:perception.stale', run='b_run', run_text='Run', auto=True, export='b_export',
          rules=True, findings=_f_perception, before=('viewports',),
          after=(Next('coverage', 'Where lidars and radars reach in the same directions ("From perception" takes '
                                  'these classes).', 'b_from'),)),
    Guide('rig', 'Live rig',
          'Stream every camera at once with its health (frame rate, drops, exposure, sharpness, sync), check each '
          'still looks where the plan says, and coach a board calibration live.', needs=(NEED_CAMERAS,),
          rules=True, result_fn=_r_rig, busy='running', run='b_start', run_text='Start', export='b_save',
          after=(Next('install', 'Check the installation from the live frames.', 'b_check', here=True),
                 Next('fleet_twin', 'Follow each sensor\'s drift over time.'))),
    Guide('coverage', 'Lidar and radar coverage',
          'Where the lidars and radars reach and what the vehicle blocks, how far cameras, lidars and radars detect '
          'each object per direction, and whether lidar points project into each camera.',
          needs=(NEED_RANGE_SENSOR,), result='ctx:sensor_coverage.result', module='sensor_coverage',
          busy='ctx:sensor_coverage.busy', stale='ctx:sensor_coverage.stale', run='b_run', run_text='Run', auto=True,
          export='b_export', findings=_f_coverage, before=('perception',),
          after=(Next('lidar_calib', 'Plan the lidar–camera calibration.'),)),
    Guide('timing', 'Sync and timing',
          'When each camera fires against the spinning lidar\'s sweep, which cameras share a trigger line, and the '
          'motion blur and rolling-shutter skew at driving speed.', needs=(NEED_CAMERAS,), result='res',
          module='timing', auto=True, export='b_export', findings=_f_timing,
          after=(Next('timing', 'Group the cameras on trigger lines that fire with the sweep.', 'b_suggest',
                      here=True), Next('clock_tree', 'The clock distribution behind the triggers.'))),
    Guide('clock_tree', 'Clock tree and time sync',
          'The rig\'s clock distribution from GNSS through the grandmaster and switches to every sensor, a fault '
          'scenario run as a Monte Carlo, and the projection error the timestamp errors make.',
          needs=(NEED_CAMERAS,), result='res', module='clock_tree', settings='clock_tree', run='b_run',
          run_text='Run', auto=True, export='b_export', before=('timing',),
          after=(Next('timing', 'The trigger plan these clocks drive.'),)),
    Guide('isp', 'Exposure metering and HDR seams',
          'Each camera\'s auto-exposure metering mask and the AE rectangles to load into the ISP, and how far the '
          'exposures drift apart across the stitched view\'s seams.', needs=(NEED_CAMERAS,), result='res',
          settings='isp_photometrics', stale='stale', run='b_run', run_text='Run', auto=True, export='b_export',
          before=('exposure',), after=(Next('camviews', 'The stitched view whose seams the exposures meet at.'),)),
    Guide('recording', 'Recording budget',
          'How much data every camera, lidar and radar records, whether the links, the network and the disks keep '
          'up, and how many hours fit on one disk set.', needs=(NEED_SENSING,), result='result', module='recording',
          auto=True, export='b_export', after=(Next('power', 'What the recorder and the sensors draw.'),)),
    Guide('power', 'Power budget',
          'What every device draws, whether the DC-DC converter, fuses and cables carry it, the alternator margin, '
          'the engine-off runtime and the heat the enclosure must shed.', result='result', module='power',
          busy='', auto=True, export='b_export', findings=_f_power, before=('recording',),
          after=(Next('harness', 'The cables that carry the power and the data.'),)),
    Guide('gnss', 'GNSS and IMU',
          'The sky each GNSS antenna sees, the dual-antenna heading baseline, the lever arms from the IMU, and how '
          'well each sensor\'s world pose will be known.', needs=(NEED_GNSS,), result='result', module='gnss',
          auto=True, export='b_export', after=(Next('timing', 'The sensors\' sync against the GNSS time.'),)),
    Guide('exposure', 'Sun glare and dirt',
          'Which cameras get the sun in their image over the campaign (where, when, at which headings), the flare '
          'risk, the lidars\' solar background, and how exposed each lens and window is to dirt and spray.',
          result='ctx:exposure.result', module='exposure', busy='ctx:exposure.busy', stale='ctx:exposure.stale',
          run='b_run', run_text='Run', auto=True, export='b_export',
          after=(Next('isp', 'Metering masks and HDR seams for the same cameras.'),)),
    Guide('fleet', 'Fleet',
          'The same rig built on many vehicles: every vehicle\'s measured sensor poses against the plan, how each '
          'sensor moves over months, and which vehicle differs from the rest.', result='result', module='fleet',
          settings='fleet', auto=True, export='b_export', before=('install', 'asbuilt'),
          after=(Next('fleet_twin', 'Drift per vehicle and sensor over time, forecast.'),)),
    Guide('brackets', 'Mount brackets',
          'A 3D-printable bracket per camera that holds it at its planned position and angles on the surface under '
          'it, with the checks and the STL files.', needs=(NEED_CAMERAS,), result='ctx:brackets_ctl.result',
          module='brackets', busy='ctx:brackets_ctl.busy', stale='ctx:brackets_ctl.stale', run='b_gen',
          run_text='Generate', export='b_all', before=('harness',),
          after=(Next('install', 'Check the fitted cameras against the plan.'),)),
    Guide('dataset', 'Sample dataset',
          'Write a short synthetic drive recorded by exactly the planned rig as a nuScenes folder and a ROS 2 bag, '
          'to test ingestion, labelling and training loaders before any hardware exists.', needs=(NEED_SENSING,),
          result='result', module='dataset', run='b_generate', run_text='Generate', estimate='g_status',
          findings=_f_dataset, after=(Next('drivelog', 'Check the written drive as you would a real one.'),)),
    Guide('drivelog', 'Drive-log check',
          'Check a recorded drive (a ROS 2 bag or a nuScenes folder) against the plan: stream health, data rate, the '
          'measured sync, each camera\'s rotation against the lidar, the GNSS and IMU.', needs=(NEED_DRIVELOG,),
          rules=True, result='result', module='drivelog', stale='stale', run='b_run', run_text='Run', export='b_export',
          before=('sim', 'dataset'), after=(Next('asbuilt', 'The as-built check of the same recording.'),
                                            Next('online_calib', 'The extrinsics over the whole drive.'))),
    Guide('requirements', 'Requirements scorecard',
          'The programme\'s written requirements as pass/fail checks on the app\'s own studies, one verdict, the '
          'evidence behind each value, and the design-review pack.', result='full', module='requirements',
          settings='requirements', run='b_run', run_text='Run', export='b_export', stale_fn=_s_requirements,
          rules=True, findings=_f_requirements,
          needs=(Need('items', 'written requirements', _requirements, 'press New requirement, Add from the plan, '
                      'Templates or Suggest from the brief', {'page': 'requirements', 'focus': 'b_new'}),),
          after=(Next('designer', 'A Pareto set of complete rigs that meets these requirements.'),)),
    Guide('lidar_calib', 'Lidar–camera calibration',
          'Where a lidar sits relative to a camera, measured from a checkerboard held at several poses both see: '
          'plan the poses, capture them (synthetic or real), solve, check the overlay, apply.',
          needs=(NEED_CAMERAS, NEED_LIDAR), result='result', module='lidar_calib', stale='stale',
          run='b_run', run_text='Capture and solve', estimate='plan_note', export='b_export', findings=_f_lidar_calib,
          before=('coverage',), after=(Next('observability', 'How firmly the calibration pins every pose.'),)),
    Guide('asbuilt', 'As-built check',
          'A recording of the real rig against the plan: each camera\'s view of the body against the simulated one, '
          'the pose correction, the coverage as built and a verdict; an as-built variant from it.',
          needs=(NEED_ASBUILT,), result='result', module='asbuilt', settings='asbuilt', stale='stale', run='b_run',
          run_text='Run', export='b_export', findings=_f_asbuilt, before=('drivelog',),
          after=(Next('asbuilt', 'Make the confident corrections an as-built variant.', 'b_add', here=True),
                 Next('fleet', 'Record the as-built poses per vehicle.'))),
    Guide('online_calib', 'Online self-calibration',
          'Every camera\'s extrinsics over a whole recorded drive from the drive\'s own content, with knocks told '
          'from reversible flex, and the final segment as an as-built variant.', needs=(NEED_ONLINE,),
          result='result', module='online_calib', settings='online_calib', stale='stale', run='b_run',
          run_text='Run', export='b_export', findings=_f_online_calib, before=('drivelog',),
          after=(Next('fleet_twin', 'The drift over time (Import calibration… takes this page\'s export).',
                      'b_import'),)),
    Guide('critical', 'Critical targets',
          'The objects the rig must resolve at known places (a child in front of the bumper, a traffic light, a '
          'kerb), which cameras resolve them, and how far each mount may be off before they stop.',
          needs=(NEED_CRITICAL,), result='ctx:critical.result', module='critical', settings='critical',
          busy='ctx:critical.busy', stale='ctx:critical.stale', run='b_run', run_text='Run', auto=True,
          export='b_export', after=(Next('critical_carla', 'The same targets stood in CARLA and compared.', 'b_carla',
                                         here=True),)),
    Guide('materials', 'Materials, glass and wipers',
          'The body\'s materials, what every radar looks through (loss, range, multipath), the cameras behind glass '
          'and the wipers\' sweep.', needs=(NEED_MESH,), result='ctx:materials.result', module='materials',
          settings='materials', busy='ctx:materials.busy', auto=True, export='b_export',
          after=(Next('coverage', 'The radars\' reach with the fascia\'s loss.'),)),
    Guide('automation', 'Automation',
          'Regenerate what downstream needs when the design changes (the exports, the requirements check) and run '
          'your own command, on save or after changes.', needs=(NEED_SAVED,), result_fn=_r_automation,
          module='automation', settings='automation', busy='ctx:automation.busy', run='b_run', run_text='Run now',
          export='b_export', findings=_f_automation,
          after=(Next('requirements', 'The requirements it checks on every run.'),)),
    Guide('terrain', 'Road and body motion',
          'The road under the car (ramps, crests, speed bumps, kerbs) against flat ground, and how braking, '
          'accelerating and cornering move the cameras.', needs=(NEED_MESH,), result='ctx:terrain_ctl.result',
          module='terrain', settings='terrain', busy='ctx:terrain_ctl.busy', stale='ctx:terrain_ctl.stale',
          run='b_run', run_text='Run', auto=True, export='b_export',
          after=(Next('thermo_vib', 'How heat and vibration move the cameras too.'),)),
    Guide('thermo_vib', 'Thermal and vibration drift',
          'How the midday sun warps the rig and how the road shakes it: each device\'s drift over the day, its first '
          'mode, the blur and the pairs\' drift in pixels.', result='result', module='thermo_vib',
          settings='thermo_vib', stale='stale', run='b_run', run_text='Run', auto=True, export='b_export',
          before=('terrain',), after=(Next('tolerance', 'The drift as part of the mounting tolerance study.'),)),
    Guide('harness', 'Cable harness',
          'Every camera\'s and sensor\'s cable routed along the body to the compute box: each run\'s length against '
          'its link, the GNSS cables\' RF margin and the power pairs\' drop.', result='ctx:harness_ctl.result',
          module='harness', settings='harness', busy='ctx:harness_ctl.busy', stale='ctx:harness_ctl.stale',
          auto=True, export='b_export', before=('power',),
          after=(Next('brackets', 'The mounts the cables run to.'),)),
    Guide('observability', 'Calibration observability',
          'How firmly the planned calibration pins every camera\'s pose, each pair, the lidar–camera pairs and the '
          'radars; why a direction stays weak and what moving or adding a board gains.', needs=(NEED_TARGETS,),
          result='result', module='observability', settings='observability', stale='stale', run='b_run',
          run_text='Run', auto=True, export='b_export', before=('targets', 'lidar_calib'),
          after=(Next('observability', 'Check the predictions with a Monte Carlo of the real solves.', 'b_verify',
                      here=True),)),
    Guide('experiments', 'Experiments',
          'Many designs of the rig evaluated automatically: factors to vary, responses to compute, a design (full '
          'factorial, Latin hypercube, Sobol …) and the sensitivities, Pareto front and best design.',
          result='result', module='experiments', settings='experiments', stale='stale', run='b_run', run_text='Run',
          long='many design points (minutes)', plan_kind='experiment', estimate='cost', export='b_export',
          findings=_f_experiments, after=(Next('experiments', 'Add the best design as a variant.', 'b_variant',
                                               here=True),)),
    Guide('sim', 'CARLA simulator',
          'Drive the planned rig in the CARLA simulator (every camera, lidar, radar, GNSS and IMU as planned) and get '
          'the drive back as a nuScenes folder and a ROS 2 bag with ground truth.', needs=(NEED_CARLA,),
          result='result', settings='simrig', run='b_run', run_text='Run', long='CARLA (a server on the GPU)',
          plan_kind='carla_run', export='b_export', findings=_f_sim,
          after=(Next('drivelog', 'Check the drive\'s streams, drops and sync.', 'b_drivelog', here=True),
                 Next('detection', 'Score detection on the drive\'s ground truth.', 'b_detect', here=True))),
    Guide('detection', 'Detection quality',
          'Run an off-the-shelf detector (YOLOX) on CARLA drives with ground truth: recall, precision and AP per '
          'camera, class and range, the rig as a whole, and rigs compared.',
          needs=(_need_recordings('detection_eval', 'detection'), NEED_YOLOX), result='result',
          module='detection_eval', settings='detection_eval', stale='stale', run='b_run', run_text='Run',
          long='detector inference on every frame (minutes)', plan_kind='detection', export='b_export',
          findings=_f_detection, before=('sim', 'campaign'),
          after=(Next('rig_perception', '3D detection per region around the car, rig against rig.'),
                 Next('robust_perception', 'The same detection under the rig\'s tolerances.'))),
    Guide('campaign', 'Simulation campaigns',
          'Many CARLA drives, each varied one known way (a fault, the weather, a rig variant, a seed), checked by '
          'Vantage\'s own checks: detection curves, false alarms, the smallest fault each check finds.',
          needs=(NEED_CARLA,), result='result', module='simcampaign', settings='simcampaign', stale='stale',
          run='b_run', run_text='Run', long='many CARLA drives', plan_kind='campaign', estimate='cost',
          export='b_export', findings=_f_campaign, before=('sim',),
          after=(Next('detection', 'Detection quality on the campaign\'s drives.'),)),
    Guide('critical_carla', 'Critical targets in CARLA',
          'The critical targets stood in CARLA around the still car, recorded with ground truth and compared camera '
          'by camera with the critical-targets study.', needs=(NEED_CRITICAL, NEED_CARLA), result='result',
          module='critical_carla', settings='critical_carla', stale='stale', run='b_run', run_text='Run',
          long='CARLA drives', plan_kind='critical_carla', estimate='plan_lbl', export='b_export',
          findings=_f_critical_carla, before=('critical',),
          after=(Next('critical', 'The study it checks: adjust the targets or the rig there.'),)),
    Guide('splat', 'Log replay',
          'Rebuild a recorded drive\'s street as 3D Gaussians, render it through the recorded rig and a proposed '
          'variant, compare detection between them, and re-drive the log through the proposed rig.',
          needs=(NEED_SPLAT, NEED_TORCH), result='result', module='splat_reproject', settings='splat_reproject',
          stale='stale', run='b_run', run_text='Run A/B', long='PyTorch, best on a GPU (the scene is optimised)',
          export='b_export', before=('drivelog',),
          after=(Next('rig_perception', 'Rig perception on the re-shot drive.'),)),
    Guide('rig_perception', 'Rig perception',
          'Bird\'s-eye 3D detection of each class in each region around the car, for each rig on the same scenes, '
          'with intervals; rig against rig, paired.',
          needs=(_need_recordings('rig_perception', 'rig_perception', 'CARLA drives with ground truth'), NEED_YOLOX),
          result='result', module='rig_perception', settings='rig_perception', stale='stale', run='b_run',
          run_text='Run', long='the detector on every frame, and CARLA for a scene set', plan_kind='rig_perception',
          estimate='set_text', export='b_export', findings=_f_rig_perception, before=('detection', 'splat'),
          after=(Next('robust_perception', 'The same detection under the rig\'s tolerances, with error bars.'),
                 Next('perf_model', 'Its records train the learned model.'))),
    Guide('foundation', 'Foundation models',
          'How far modern detectors really find each class with each camera (50 % and 90 % ranges) and how far '
          'monocular depth stays usable, measured on rendered frames with ground truth.',
          needs=(NEED_FOUNDATION, NEED_FM_TORCH), result='result', module='foundation_range',
          settings='foundation_range', stale='stale', run='b_run', run_text='Run',
          long='torch and the models (a GPU helps; CARLA for the sweep)', plan_kind='foundation_range',
          export='b_export', findings=_f_foundation, before=('detection',),
          after=(Next('perf_model', 'Send the measured records to the learned model.', 'b_records', here=True),)),
    Guide('robust_perception', 'Robust perception',
          'The rig\'s tolerances (mounting, thermal, vibration, calibration) carried to what its perception sees, '
          'with error bars, measured on recordings with ground truth.',
          needs=(_need_recordings('robust_perception', 'robust_perception'), NEED_YOLOX), result='result',
          module='robust_perception', settings='robust_perception', run='b_run', run_text='Run',
          long='many builds through the detector (minutes of CPU)', plan_kind='robust_perception', export='b_export',
          findings=_f_robust, before=('tolerance', 'fleet_twin', 'rig_perception'),
          after=(Next('perf_model', 'Add its records to the learned model.', 'b_store', here=True),)),
    Guide('risk_coverage', 'Risk-weighted coverage',
          'The rig\'s coverage weighted by where hazards appear around the car in the seconds before a crash (NHTSA '
          'pre-crash statistics, Euro NCAP scenarios).', result='result', module='risk_coverage',
          settings='risk_coverage', stale='stale', run='b_run', run_text='Run', auto=True, export='b_export',
          findings=_f_risk, before=('viewports',),
          after=(Next('optimiser', 'Search for a rig that closes the weighted gaps.'),)),
    Guide('perf_model', 'Learned detection model',
          'A model trained on the perception records measured so far that predicts any rig\'s detection per class '
          'and region with an uncertainty, and proposes which rigs to simulate next.', result='result',
          module='perf_model', settings='perf_model', stale='_stale', run='b_train', run_text='Train',
          long='training the model', plan_kind='perf_model_train', export='b_export', findings=_f_perf_model,
          before=('rig_perception', 'foundation', 'robust_perception'),
          after=(Next('designer', 'The design agent scores rigs with the trained model.'),)),
    Guide('designer', 'Design agent',
          'Requirements in (typed, or read from documents), a Pareto set of complete rigs out, each explained and '
          'critiqued; nothing changes until you accept one.', needs=(NEED_DESIGNER,), result='res',
          module='rig_designer', settings='rig_designer', run='b_run', run_text='Run', export='b_export',
          findings=_f_designer, before=('requirements', 'perf_model'),
          after=(Next('compare', 'The chosen option against your other variants.', 'b_compare', here=True),)),
    Guide('fleet_twin', 'Fleet twin',
          'Every vehicle\'s sensors followed over time from what the fleet produces: each axis\'s drift with its '
          'uncertainty, when each sensor will leave its limit, knocks flagged, the drift handed back to the design.',
          needs=(NEED_FLEET,), result='result', module='fleet_twin', settings='fleet_twin', stale='stale',
          run='b_run', run_text='Run', auto=True, export='b_export', before=('fleet', 'online_calib'),
          after=(Next('tolerance', 'Use the measured drift as the mounting tolerance.', 'b_tol', here=True),
                 Next('robust_perception', 'Carry the measured spread to detection.'))),
]
GUIDES: dict = {g.page: g for g in _G}
