"""The project's design log: what was decided, rejected, required and noted about the design, kept IN the project file
(project.analysis.assistant_log, a list of plain dicts; old files without it load with an empty log) so the assistant
remembers between sessions and does not suggest again what the user turned down.

An entry: {id, kind: decision | rejected | constraint | note, text, about: [camera or sensor uids], date: YYYY-MM-DD,
source: user | assistant (proposed by the assistant and accepted by the user), changes: [{camera, prop, delta | after}]
(optional: the concrete change a rejected or decided entry is about, so a later suggestion of the same change can be
recognised)}.

Who writes it:
- the user, in the panel's Design log view (add, edit, delete), each edit one undo step (store.edit('analysis', …));
- the assistant only proposes (add_design_log): the entry is written when the user accepts it, like a proposal;
- discarding a proposed change writes a 'rejected' entry by itself ("rejected: pitch F −4° (blind +0.06 m²)").

What reads it: the model gets the log in its instructions (prompt_text, compact and capped); proposals that repeat a
rejected change say so in the tool result (matches_rejected); the local recommendations that would suggest a rejected
change are left out (suppress). No Qt here, no network; deterministic."""
from __future__ import annotations

import math
import re
import time
import uuid

KINDS = ('decision', 'rejected', 'constraint', 'note')
SOURCES = ('user', 'assistant')
KIND_LABEL = {'decision': 'Decision', 'rejected': 'Rejected', 'constraint': 'Constraint', 'note': 'Note'}
SOURCE_LABEL = {'user': 'you', 'assistant': 'proposed by the assistant, accepted by you'}
TEXT_MAX = 400
PROMPT_CAP = 2400               # characters of the log in the model's instructions
MAX_ENTRIES = 500
PROPS = ('x', 'y', 'z', 'yaw', 'pitch', 'roll', 'lens', 'module', 'visible')
MATCH_FRACTION = 0.5            # a suggested move at least half as large as a rejected one, same direction, matches


# ====================================================================================================== entries
def clean(d, tags_to_uid=None) -> dict | None:
    """One entry, made safe: known kind and source, text trimmed, about a list of strings, a date; None when junk."""
    if not isinstance(d, dict):
        return None
    kind = d.get('kind') if d.get('kind') in KINDS else 'note'
    text = ' '.join(str(d.get('text') or '').split())[:TEXT_MAX]
    if not text:
        return None
    about = d.get('about') if isinstance(d.get('about'), (list, tuple)) else []
    about = [str(a) for a in about if str(a).strip()][:24]
    if tags_to_uid:
        about = [tags_to_uid.get(a.lower(), a) for a in about]
    date = str(d.get('date') or '')[:10]
    if not re.fullmatch(r'\d{4}-\d{2}-\d{2}', date):
        date = time.strftime('%Y-%m-%d')
    src = d.get('source') if d.get('source') in SOURCES else 'user'
    out = {'id': str(d.get('id') or '')[:40] or uuid.uuid4().hex[:12], 'kind': kind, 'text': text, 'about': about,
           'date': date, 'source': src}
    ch = []
    for c in d.get('changes') or [] if isinstance(d.get('changes'), list) else []:
        if isinstance(c, dict) and c.get('camera') and c.get('prop') in PROPS:
            e = {'camera': str(c['camera']), 'prop': c['prop']}
            for k in ('delta', 'after'):
                if k in c and c[k] is not None:
                    v = c[k]
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        if math.isfinite(float(v)):
                            e[k] = float(v)
                    else:
                        e[k] = str(v)[:80]
            ch.append(e)
    if ch:
        out['changes'] = ch[:12]
    return out


def entries(project) -> list:
    """The log, cleaned (a damaged or hand-edited file never breaks the panel), oldest first."""
    raw = getattr(getattr(project, 'analysis', None), 'assistant_log', None)
    out, seen = [], set()
    for d in raw if isinstance(raw, list) else []:
        e = clean(d)
        if e is not None and e['id'] not in seen:
            seen.add(e['id'])
            out.append(e)
    return out


def new_entry(kind, text, about=(), source='user', changes=None, date=None) -> dict | None:
    return clean({'kind': kind, 'text': text, 'about': list(about or []), 'source': source,
                  'changes': changes or [], 'date': date or time.strftime('%Y-%m-%d')})


# the mutators run inside store.edit('analysis', fn, text): one undo step each
def add(project, entry: dict) -> dict | None:
    e = clean(entry)
    if e is None:
        return None
    log = entries(project)
    if any(x['id'] == e['id'] for x in log):
        e['id'] = uuid.uuid4().hex[:12]
    log.append(e)
    project.analysis.assistant_log = log[-MAX_ENTRIES:]
    return e


def update(project, eid, **fields) -> bool:
    log = entries(project)
    for i, e in enumerate(log):
        if e['id'] == eid:
            new = clean(dict(e, **{k: v for k, v in fields.items() if k in ('kind', 'text', 'about', 'changes',
                                                                             'source', 'date')}))
            if new is None:
                return False
            log[i] = new
            project.analysis.assistant_log = log
            return True
    return False


def remove(project, eid) -> bool:
    log = entries(project)
    keep = [e for e in log if e['id'] != eid]
    if len(keep) == len(log):
        return False
    project.analysis.assistant_log = keep
    return True


# ====================================================================================================== names
def _uid_tags(project) -> dict:
    """uid -> tag for every camera and sensor of every variant (an entry may concern a camera of another variant)."""
    out = {}
    for v in getattr(project, 'variants', []) or []:
        for c in getattr(v, 'cameras', []):
            out.setdefault(c.uid, c.tag)
        for s in getattr(v, 'sensors', []) or []:
            out.setdefault(s.uid, s.tag)
    return out


def tag_to_uid(project) -> dict:
    """tag (lower case) -> uid in the active variant (cameras and sensors)."""
    out = {}
    try:
        v = project.variant
    except AttributeError:
        return out
    for c in list(v.cameras) + list(getattr(v, 'sensors', []) or []):
        out.setdefault(c.tag.lower(), c.uid)
        out.setdefault(c.uid.lower(), c.uid)
    return out


def about_tags(project, e) -> list:
    names = _uid_tags(project)
    return [names.get(a, a) for a in e.get('about', [])]


def for_model(project, e) -> dict:
    """An entry as a tool returns it (tags instead of uids)."""
    out = {'id': e['id'], 'kind': e['kind'], 'text': e['text'], 'about': about_tags(project, e), 'date': e['date'],
           'source': 'user' if e['source'] == 'user' else 'assistant (accepted by the user)'}
    if e.get('changes'):
        out['changes'] = e['changes']
    return out


ORDER = {'constraint': 0, 'rejected': 1, 'decision': 2, 'note': 3}


def prompt_text(project, cap=PROMPT_CAP) -> str:
    """The log for the model's instructions: constraints, rejected ideas, decisions, notes (newest first within a
    kind), one line each, capped; '' for an empty log."""
    log = entries(project)
    if not log:
        return ''
    order = sorted(range(len(log)), key=lambda i: (ORDER[log[i]['kind']], -i))
    lines, used, left = [], 0, 0
    for n, i in enumerate(order):
        e = log[i]
        tags = about_tags(project, e)
        line = f'- {e["kind"]} ({e["date"]}): {e["text"]}' + (f' [about {", ".join(tags)}]' if tags else '')
        if used + len(line) + 1 > cap:
            left = len(order) - n
            break
        lines.append(line)
        used += len(line) + 1
    out = '\n'.join(lines)
    if left:
        out += f'\n- … {left} older entr{"y" if left == 1 else "ies"} not shown (get_design_log lists them all)'
    return out


INSTRUCTIONS = ('[The project\'s design log, kept with the project. Respect every constraint. Do not propose a change '
                'that was rejected (or a larger move in the same direction) unless the user asks for it again; if it '
                'is the only fix, say so and ask. Decisions stand unless the user reopens them.]')


def prompt_block(project, cap=PROMPT_CAP) -> str:
    body = prompt_text(project, cap)
    return f'DESIGN LOG\n{INSTRUCTIONS}\n{body}' if body else ''


# ====================================================================================================== rejected changes
def _fmt(v, unit):
    if unit == 'm':
        return f'{v * 1000:+.0f} mm'
    if unit == '°':
        return f'{v:+.1f}°'.replace('-', '−')
    return f'{v:+g}'


def changes_of_rows(rows) -> list:
    """A Proposal's rows as changes: camera rows with numeric before/after give a delta; lens/module/visible the new
    value. Mirrored partner rows are kept (they are part of the same idea)."""
    out = []
    for r in rows or []:
        if r.get('kind', 'camera') != 'camera' or not r.get('camera'):
            continue
        prop = r.get('prop')
        if prop not in PROPS:
            continue
        a, b = r.get('before'), r.get('after')
        if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
            d = float(b) - float(a)
            if prop in ('yaw', 'roll'):
                d = (d + 180.0) % 360.0 - 180.0
            if abs(d) > 1e-9:
                out.append({'camera': r['camera'], 'prop': prop, 'delta': round(d, 4)})
        elif b is not None:
            out.append({'camera': r['camera'], 'prop': prop, 'after': b if isinstance(b, (bool, int, float))
                        else str(b)})
    return out


def rejected_entry(prop, source='user') -> dict | None:
    """The 'rejected' entry for a discarded Proposal: "rejected: pitch F −4.0° (blind +0.060 m²)"."""
    rows = list(getattr(prop, 'rows', []) or [])
    if not rows:
        return None
    bits = []
    for r in rows:
        if r.get('mirrored_from'):
            continue
        a, b, unit = r.get('before'), r.get('after'), r.get('unit', '')
        what = r.get('camera') or r.get('label') or ''
        if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
            bits.append(f'{r["prop"]} {what} {_fmt(float(b) - float(a), unit)}')
        else:
            bits.append(f'{r["prop"]} {what} → {b}')
    eff = []
    for ef in getattr(prop, 'effects', []) or []:
        if ef.get('key') in ('blind_m2', 'min_density') and abs(float(ef['after']) - float(ef['before'])) > 1e-6:
            d = float(ef['after']) - float(ef['before'])
            eff.append(f'blind {d:+.3f} m²' if ef['key'] == 'blind_m2' else f'min density {d:+.2f} px/cm')
    text = 'rejected: ' + ', '.join(bits[:6]) + (f' ({", ".join(eff)})' if eff else '')
    title = str(getattr(prop, 'title', '') or '').strip()
    if title:
        text += f' — "{title[:80]}"'
    about = list(dict.fromkeys(list(getattr(prop, 'targets', {}) or {})))
    return new_entry('rejected', text, about, source, changes_of_rows(rows))


_NUM = r'([+\-−]?\d+(?:\.\d+)?)\s*(°|deg|degrees?|mm|cm|m)?'


def parse_changes(text, tags) -> list:
    """Changes named in plain words: "pitch F −4°", "F pitched +2°", "lower ML 20 mm" (z), "raise ML 20 mm"…; tags
    are the camera tags to look for. Deterministic and narrow: what it does not recognise it leaves out."""
    out = []
    if not text:
        return out
    t = str(text).replace('−', '-')
    tagset = sorted({str(x) for x in tags if x}, key=len, reverse=True)
    if not tagset:
        return out
    T = '|'.join(re.escape(x) for x in tagset)
    props = 'pitch|yaw|roll|x|y|z'

    def unit_val(v, u, prop):
        v = float(v)
        if u == 'mm':
            v /= 1000.0
        elif u == 'cm':
            v /= 100.0
        return v
    for m in re.finditer(rf'\b({props})(?:ed)?\s+(?:of\s+|camera\s+)?({T})\b\s*(?:by\s+)?{_NUM}', t):
        out.append({'camera': m.group(2), 'prop': m.group(1), 'delta': unit_val(m.group(3), m.group(4), m.group(1))})
    for m in re.finditer(rf'\b({T})\s+(?:camera\s+)?({props})(?:ed)?\s*(?:by\s+)?{_NUM}', t):
        out.append({'camera': m.group(1), 'prop': m.group(2), 'delta': unit_val(m.group(3), m.group(4), m.group(2))})
    for m in re.finditer(rf'\b(lower|raise)\s+(?:camera\s+)?({T})\b\s*(?:by\s+)?(\d+(?:\.\d+)?)\s*(mm|cm|m)\b', t,
                         re.IGNORECASE):
        v = unit_val(m.group(3), m.group(4), 'z')
        out.append({'camera': m.group(2), 'prop': 'z', 'delta': -v if m.group(1).lower() == 'lower' else v})
    seen, uniq = set(), []
    for c in out:
        k = (c['camera'], c['prop'], round(c['delta'], 6))
        if k not in seen:
            seen.add(k)
            uniq.append(c)
    return uniq


def _same(a, b) -> bool:
    """Does change a repeat the rejected change b (same camera and property; for a move, the same direction and at
    least MATCH_FRACTION of its size; for a value, the same value)?"""
    if str(a.get('camera', '')).lower() != str(b.get('camera', '')).lower() or a.get('prop') != b.get('prop'):
        return False
    if 'delta' in a and 'delta' in b:
        da, db = float(a['delta']), float(b['delta'])
        return da * db > 0 and abs(da) >= MATCH_FRACTION * abs(db) - 1e-9
    if 'after' in a and 'after' in b:
        return str(a['after']).lower() == str(b['after']).lower()
    return False


def _entry_changes(project, e) -> list:
    """An entry's changes with camera tags (stored ones; else parsed from its text)."""
    names = _uid_tags(project)
    ch = [dict(c, camera=names.get(c['camera'], c['camera'])) for c in e.get('changes', [])]
    if not ch:
        ch = parse_changes(e['text'], [c.tag for c in getattr(project.variant, 'cameras', [])])
    return ch


def matches_rejected(project, changes) -> list:
    """The rejected entries that the changes repeat."""
    out = []
    for e in entries(project):
        if e['kind'] != 'rejected':
            continue
        rej = _entry_changes(project, e)
        if any(_same(c, r) for c in changes or [] for r in rej):
            out.append(e)
    return out


def rec_changes(project, rec) -> list:
    """The concrete change a local recommendation suggests: the simulated pitch hint (evidence delta), or one named in
    its title or prompt."""
    out = []
    ev = getattr(rec, 'evidence', {}) or {}
    cam = next((a[1] for a in getattr(rec, 'about', ()) or () if len(a) >= 2 and a[0] == 'camera'), None)
    if cam and isinstance(ev.get('delta'), (int, float)) and str(getattr(rec, 'id', '')).startswith('density:'):
        out.append({'camera': cam, 'prop': 'pitch', 'delta': float(ev['delta'])})
    tags = [c.tag for c in getattr(project.variant, 'cameras', [])]
    out += parse_changes(f'{getattr(rec, "title", "")} . {getattr(rec, "prompt", "")}', tags)
    return out


def suppress(project, recs) -> tuple[list, list]:
    """(kept, suppressed): recommendations that would suggest a rejected change are left out."""
    if not any(e['kind'] == 'rejected' for e in entries(project)):
        return list(recs), []
    kept, gone = [], []
    for r in recs:
        (gone if matches_rejected(project, rec_changes(project, r)) else kept).append(r)
    return kept, gone


def summary(project) -> dict:
    log = entries(project)
    return {'count': len(log), **{k: sum(1 for e in log if e['kind'] == k) for k in KINDS}}


def export_doc(project) -> dict:
    """The log for an export (the design document's JSON): entries with tags, oldest first."""
    return {'schema': 'vantage.design_log/1', 'entries': [for_model(project, e) for e in entries(project)]}


def to_markdown(project) -> str:
    log = entries(project)
    if not log:
        return ''
    lines = ['## Design log', '']
    for e in log:
        tags = about_tags(project, e)
        lines.append(f'- **{KIND_LABEL[e["kind"]]}** ({e["date"]}, {SOURCE_LABEL[e["source"]]}): {e["text"]}'
                     + (f' — about {", ".join(tags)}' if tags else ''))
    return '\n'.join(lines) + '\n'
