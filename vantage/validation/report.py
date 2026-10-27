"""The validation report: JSON (every number), CSV (one row per check), a standalone HTML page and the figures as
SVG. The page needs nothing else: its figures are inline SVG and it follows the reader's light or dark setting."""
from __future__ import annotations

import csv
import html
import io
import json
import math
import os

import numpy as np

from .references import REFERENCES, cite

# chart tokens (light, dark): categorical slots in fixed order, surfaces, ink, grid
SERIES = (('#2a78d6', '#3987e5'), ('#eb6834', '#d95926'), ('#1baf7a', '#199e70'), ('#eda100', '#c98500'),
          ('#e87ba4', '#d55181'), ('#008300', '#008300'), ('#6250d6', '#9085e9'), ('#e34948', '#e66767'))
TOKENS = {'surface': ('#fcfcfb', '#1a1a19'), 'plane': ('#f9f9f7', '#0d0d0d'), 'ink': ('#0b0b0b', '#f0efec'),
          'ink2': ('#52514e', '#c3c2b7'), 'muted': ('#898781', '#898781'), 'grid': ('#e1e0d9', '#2c2c2a'),
          'axis': ('#c3c2b7', '#383835'), 'good': ('#0ca30c', '#0ca30c'), 'bad': ('#d03b3b', '#d03b3b'),
          'hair': ('rgba(11,11,11,0.10)', 'rgba(255,255,255,0.10)')}


def _vars(k):
    return ''.join(f'--{n}:{v[k]};' for n, v in TOKENS.items()) + \
        ''.join(f'--s{i + 1}:{c[k]};' for i, c in enumerate(SERIES))


CSS_VARS = (f':root{{color-scheme:light;{_vars(0)}}}'
            f'@media (prefers-color-scheme: dark){{:root:not([data-theme="light"]){{color-scheme:dark;{_vars(1)}}}}}'
            f':root[data-theme="dark"]{{color-scheme:dark;{_vars(1)}}}')


# ============================================================================ JSON
def clean(o):
    """JSON-safe: numpy scalars and arrays to Python, tuples to lists, non-finite numbers to None."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        f = float(o)
        return f if math.isfinite(f) else None
    if o is None or isinstance(o, (int, str)):
        return o
    return str(o)


def check_doc(c) -> dict:
    return {'id': c.id, 'model': c.model, 'title': c.title, 'quantity': c.quantity, 'reference': c.reference,
            'references': list(c.refs), 'error': c.error, 'tolerance': c.tolerance, 'unit': c.unit,
            'passed': c.passed, 'seconds': round(c.seconds, 3), 'note': c.note, 'exception': c.exception,
            'detail': c.detail, 'figure': c.figure}


# ============================================================================ CSV
CSV_COLUMNS = ('id', 'model', 'title', 'error', 'tolerance', 'unit', 'passed', 'seconds', 'quantity', 'reference',
               'exception')


def to_csv(result) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator='\n')
    w.writerow(CSV_COLUMNS)
    for c in result.checks:
        w.writerow([c.id, c.model, c.title, f'{c.error:.6g}' if math.isfinite(c.error) else '', f'{c.tolerance:.6g}',
                    c.unit, 'yes' if c.passed else 'no', f'{c.seconds:.3f}', c.quantity, c.reference, c.exception])
    return buf.getvalue()


# ============================================================================ SVG figures
def _nice(lo, hi, n=5):
    """Round tick values covering [lo, hi]."""
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return [0.0, 1.0]
    if hi - lo < 1e-12 * max(abs(hi), 1.0):
        lo, hi = lo - 0.5 * (abs(lo) or 1.0), hi + 0.5 * (abs(hi) or 1.0)
    raw = (hi - lo) / n
    mag = 10 ** math.floor(math.log10(raw))
    step = next(s * mag for s in (1, 2, 2.5, 5, 10) if s * mag >= raw)
    a = math.floor(lo / step) * step
    out = []
    v = a
    while v <= hi + step * 1e-9:
        out.append(round(v, 12))
        v += step
    if out[-1] < hi:
        out.append(out[-1] + step)
    return out


def _fmt(v):
    if v == 0:
        return '0'
    a = abs(v)
    if a >= 1e4 or a < 1e-3:
        return f'{v:.0e}'.replace('e+0', 'e').replace('e-0', 'e-')
    return f'{v:.6g}'


def svg_figure(fig: dict, width=640, height=300, standalone=False) -> str:
    """An inline SVG of a figure spec (suite.figure). standalone: with its own style block and namespace, for an
    .svg file."""
    if not fig:
        return ''
    ml, mr, mt, mb = 64, 18, 34, 46
    W, H = width - ml - mr, height - mt - mb
    title = html.escape(fig.get('title', ''))
    data = fig.get('series') or []
    out = []
    if fig.get('kind') == 'bars':
        vals = [v if v is not None else 0.0 for v in (data[0]['y'] if data else [])]
        cats = fig.get('categories') or [str(i) for i in range(len(vals))]
        top = max(vals + [fig.get('hline') or 0.0, 1e-12])
        ticks = _nice(0.0, top)
        ymax = ticks[-1]
        n = max(len(vals), 1)
        band = W / n
        bw = min(24.0, band * 0.6)
        for t in ticks:
            y = mt + H - t / ymax * H
            out.append(f'<line x1="{ml}" x2="{ml + W}" y1="{y:.1f}" y2="{y:.1f}" class="grid"/>'
                       f'<text x="{ml - 6}" y="{y + 4:.1f}" class="tick" text-anchor="end">{_fmt(t)}</text>')
        for i, (v, c) in enumerate(zip(vals, cats)):
            x = ml + band * (i + 0.5)
            h = max(v / ymax * H, 0.0)
            y0 = mt + H
            r = min(4.0, h / 2, bw / 2)
            path = (f'M{x - bw / 2:.1f},{y0:.1f} V{y0 - h + r:.1f} Q{x - bw / 2:.1f},{y0 - h:.1f} {x - bw / 2 + r:.1f},'
                    f'{y0 - h:.1f} H{x + bw / 2 - r:.1f} Q{x + bw / 2:.1f},{y0 - h:.1f} {x + bw / 2:.1f},'
                    f'{y0 - h + r:.1f} V{y0:.1f} Z')
            out.append(f'<path d="{path}" class="bar s1"><title>{html.escape(str(c))}: {_fmt(v)}</title></path>')
            if n <= 24:
                lx, ly = x, mt + H + 12
                out.append(f'<text x="{lx:.1f}" y="{ly:.1f}" class="tick cat" text-anchor="end" '
                           f'transform="rotate(-35 {lx:.1f} {ly:.1f})">{html.escape(str(c))[:18]}</text>')
        if fig.get('hline') is not None:
            y = mt + H - fig['hline'] / ymax * H
            out.append(f'<line x1="{ml}" x2="{ml + W}" y1="{y:.1f}" y2="{y:.1f}" class="ref"/>'
                       f'<text x="{ml + W}" y="{y - 5:.1f}" class="tick" text-anchor="end">'
                       f'{html.escape(fig.get("hline_label", ""))}</text>')
        out.append(f'<line x1="{ml}" x2="{ml + W}" y1="{mt + H}" y2="{mt + H}" class="axis"/>')
        height = height + 40                                      # room for the slanted category labels
    else:
        xs = [v for s in data for v in s['x'] if v is not None]
        ys = [v for s in data for v in s['y'] if v is not None]
        logy = bool(fig.get('logy')) and ys and min(ys) > 0
        if fig.get('hline') is not None:
            ys.append(fig['hline'])
        if fig.get('diag') and xs:
            ys += [min(xs), max(xs)]
            xs += ys[-2:]
        if not xs or not ys:
            return ''
        fy = (lambda v: math.log10(v)) if logy else (lambda v: v)          # noqa: E731
        xt = _nice(min(xs), max(xs))
        yt = _nice(min(fy(v) for v in ys), max(fy(v) for v in ys))
        X = lambda v: ml + (v - xt[0]) / (xt[-1] - xt[0]) * W               # noqa: E731
        Y = lambda v: mt + H - (fy(v) - yt[0]) / (yt[-1] - yt[0]) * H        # noqa: E731
        for t in yt:
            y = mt + H - (t - yt[0]) / (yt[-1] - yt[0]) * H
            lab = _fmt(10 ** t) if logy else _fmt(t)
            out.append(f'<line x1="{ml}" x2="{ml + W}" y1="{y:.1f}" y2="{y:.1f}" class="grid"/>'
                       f'<text x="{ml - 6}" y="{y + 4:.1f}" class="tick" text-anchor="end">{lab}</text>')
        for t in xt:
            x = X(t)
            out.append(f'<text x="{x:.1f}" y="{mt + H + 16}" class="tick" text-anchor="middle">{_fmt(t)}</text>')
        out.append(f'<line x1="{ml}" x2="{ml + W}" y1="{mt + H}" y2="{mt + H}" class="axis"/>')
        if fig.get('diag'):
            a, b = max(xt[0], 10 ** yt[0] if logy else yt[0]), min(xt[-1], 10 ** yt[-1] if logy else yt[-1])
            out.append(f'<line x1="{X(a):.1f}" y1="{Y(a):.1f}" x2="{X(b):.1f}" y2="{Y(b):.1f}" class="ref"/>')
        if fig.get('hline') is not None:
            y = Y(fig['hline'])
            out.append(f'<line x1="{ml}" x2="{ml + W}" y1="{y:.1f}" y2="{y:.1f}" class="ref"/>'
                       f'<text x="{ml + W}" y="{y - 5:.1f}" class="tick" text-anchor="end">'
                       f'{html.escape(fig.get("hline_label", ""))}</text>')
        for k, s in enumerate(data[:8]):
            cls = f's{k + 1}'
            pts = [(X(x), Y(y), x, y) for x, y in zip(s['x'], s['y']) if x is not None and y is not None
                   and (not logy or y > 0)]
            if s.get('kind') == 'points':
                for px, py, x, y in pts:
                    out.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4" class="dot {cls}">'
                               f'<title>{html.escape(s["label"])}: {_fmt(x)}, {_fmt(y)}</title></circle>')
            elif pts:
                d = 'M' + ' L'.join(f'{px:.1f},{py:.1f}' for px, py, _, _ in pts)
                out.append(f'<path d="{d}" class="line {cls}"><title>{html.escape(s["label"])}</title></path>')
                if len(pts) <= 60:
                    for px, py, x, y in pts:
                        out.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="7" class="hit">'
                                   f'<title>{html.escape(s["label"])}: {_fmt(x)}, {_fmt(y)}</title></circle>')
    out.append(f'<text x="{ml}" y="18" class="ttl">{title}</text>'
               f'<text x="{ml + W / 2:.1f}" y="{height - 8 if fig.get("kind") != "bars" else height - 6}" '
               f'class="tick" text-anchor="middle">{html.escape(fig.get("xlabel", ""))}</text>'
               f'<text x="14" y="{mt + H / 2:.1f}" class="tick" text-anchor="middle" '
               f'transform="rotate(-90 14 {mt + H / 2:.1f})">{html.escape(fig.get("ylabel", ""))}</text>')
    style = ('.grid{stroke:var(--grid);stroke-width:1}.axis{stroke:var(--axis);stroke-width:1}'
             '.ref{stroke:var(--muted);stroke-width:1.5;stroke-dasharray:5 4}'
             '.tick{fill:var(--muted);font:11px system-ui,-apple-system,"Segoe UI",sans-serif;'
             'font-variant-numeric:tabular-nums}.ttl{fill:var(--ink);font:500 13px system-ui,-apple-system,'
             '"Segoe UI",sans-serif}.line{fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round}'
             '.dot{stroke:var(--surface);stroke-width:2}.hit{fill:transparent}'
             + ''.join(f'.line.s{i + 1}{{stroke:var(--s{i + 1})}}.dot.s{i + 1},.bar.s{i + 1}{{fill:var(--s{i + 1})}}'
                       for i in range(len(SERIES))))
    body = ''.join(out)
    legend = ''
    if fig.get('kind') != 'bars' and len(data) >= 2:
        items = ''.join(f'<span class="key"><i class="sw s{k + 1}"></i>{html.escape(s["label"])}</span>'
                        for k, s in enumerate(data[:8]))
        legend = f'<div class="legend">{items}</div>'
    if standalone:
        keys = ''
        if fig.get('kind') != 'bars' and len(data) >= 2:
            keys = ''.join(f'<rect x="{ml + 4 + 150 * (k % 4)}" y="{height + 6 + 18 * (k // 4)}" width="10" '
                           f'height="10" rx="2" class="bar s{k + 1}"/><text x="{ml + 18 + 150 * (k % 4)}" '
                           f'y="{height + 15 + 18 * (k // 4)}" class="tick">{html.escape(s["label"])[:24]}</text>'
                           for k, s in enumerate(data[:8]))
        hh = height + (26 + 18 * ((len(data[:8]) - 1) // 4) if keys else 0)
        return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {hh}" width="{width}" height="{hh}" '
                f'role="img" aria-label="{title}"><style>{CSS_VARS}svg{{background:var(--surface)}}{style}</style>'
                f'<rect width="100%" height="100%" fill="var(--surface)"/>{body}{keys}</svg>')
    return (f'<figure class="fig"><svg viewBox="0 0 {width} {height}" role="img" aria-label="{title}">'
            f'<style>{style}</style>{body}</svg>{legend}</figure>')


# ============================================================================ HTML
def _num(v, unit=''):
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return '—'
    s = f'{v:.3g}' if isinstance(v, float) else str(v)
    return f'{s} {unit}'.strip()


def _flat(d, prefix='', depth=0, out=None):
    """Detail rows (key, value) flattened two levels deep; long lists shortened."""
    out = [] if out is None else out
    for k, v in (d or {}).items():
        key = f'{prefix}{k}'
        if isinstance(v, dict) and depth < 1:
            _flat(v, key + ' · ', depth + 1, out)
        else:
            if isinstance(v, float):
                txt = _num(v)
            elif isinstance(v, (list, tuple)):
                vv = [(_num(x) if isinstance(x, float) else json.dumps(clean(x), ensure_ascii=False)) for x in v[:8]]
                txt = ', '.join(vv) + (f' … ({len(v)})' if len(v) > 8 else '')
            elif isinstance(v, dict):
                txt = json.dumps(clean(v), ensure_ascii=False)
                txt = txt if len(txt) < 240 else txt[:237] + '…'
            else:
                txt = str(v)
            out.append((key, txt))
    return out


PAGE_CSS = CSS_VARS + '''
*{box-sizing:border-box}
body{margin:0;background:var(--plane);color:var(--ink);font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1080px;margin:0 auto;padding:32px 16px 72px}
h1{font-size:30px;letter-spacing:-.02em;margin:0 0 6px}h2{font-size:22px;margin:48px 0 6px;scroll-margin-top:16px}
h3{font-size:16px;margin:0 0 6px}p{color:var(--ink2);margin:6px 0}a{color:var(--s1)}
.meta{color:var(--muted);font-size:13px}code{font:12.5px ui-monospace,"IBM Plex Mono",monospace}
table{border-collapse:collapse;width:100%;font-size:13.5px;margin:10px 0;font-variant-numeric:tabular-nums}
th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--grid);vertical-align:top;color:var(--ink2)}
th{color:var(--ink);font-weight:600;background:var(--surface)}
.wrap{overflow-x:auto}
.card{background:var(--surface);border:1px solid var(--hair);border-radius:10px;padding:16px 18px;margin:14px 0;
scroll-margin-top:16px}
.badge{display:inline-flex;align-items:center;gap:6px;font-weight:600;font-size:12.5px;border-radius:999px;
padding:2px 10px;border:1px solid currentColor}.pass{color:var(--good)}.fail{color:var(--bad)}
.tiles{display:flex;flex-wrap:wrap;gap:12px;margin:18px 0}.tile{background:var(--surface);border:1px solid var(--hair);
border-radius:10px;padding:12px 16px;min-width:150px}.tile b{display:block;font-size:26px;font-weight:600}
.tile span{color:var(--muted);font-size:13px}
.kv{display:grid;grid-template-columns:minmax(120px,190px) 1fr;gap:4px 14px;font-size:13.5px;margin:10px 0}
.kv dt{color:var(--muted)}.kv dd{margin:0;color:var(--ink2)}
details{margin-top:8px}summary{cursor:pointer;color:var(--muted);font-size:13px}
.fig{margin:12px 0 0}.fig svg{width:100%;height:auto;max-width:720px;display:block}
.legend{display:flex;flex-wrap:wrap;gap:6px 16px;font-size:12.5px;color:var(--ink2);margin-top:4px}
.key{display:inline-flex;align-items:center;gap:6px}.sw{display:inline-block;width:10px;height:10px;border-radius:2px}
''' + ''.join(f'.sw.s{i + 1}{{background:var(--s{i + 1})}}' for i in range(len(SERIES))) + '''
.err{color:var(--bad);font-size:13px}.refs li{margin:4px 0;color:var(--ink2);font-size:13.5px}
@media (max-width:640px){.kv{grid-template-columns:1fr}h1{font-size:24px}}
'''


def _badge(ok):
    return ('<span class="badge pass">✓ pass</span>' if ok else '<span class="badge fail">✕ fail</span>')


def to_html(result, project=None) -> str:
    """The standalone report page."""
    env = result.environment
    n, ok = len(result.checks), result.passed
    groups = result.by_model()
    e = html.escape
    parts = [f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
             f'<meta name="viewport" content="width=device-width, initial-scale=1">'
             f'<title>Validation report</title><style>{PAGE_CSS}</style></head><body><main>'
             f'<h1>Vantage validation report</h1>'
             f'<p class="meta">Vantage {e(str(env.get("vantage")))} · {e(result.started_utc)} UTC · '
             f'{result.seconds:.1f} s · project: {e(result.project)} · variant: {e(result.variant)} · seed '
             f'{result.settings.seed}{" · quick run" if result.settings.quick else ""}'
             f'{" · cancelled" if result.cancelled else ""}</p>'
             f'<p>Every check compares a quantity one of Vantage\'s models computes with an independent reference '
             f'(a closed form written out again, another library, a published table, worked example or measurement, '
             f'brute force or Monte Carlo) and passes when the error is at most the tolerance. The methods behind '
             f'each model are described in the technical reference (methods.html on the Vantage website).</p>'
             f'<div class="tiles"><div class="tile"><b>{ok} / {n}</b><span>checks passed</span></div>'
             f'<div class="tile"><b>{len(groups)}</b><span>models checked</span></div>'
             f'<div class="tile"><b>{n - ok}</b><span>failed</span></div></div>']
    parts.append('<div class="wrap"><table><tr><th>Model</th><th>Checks</th><th>Passed</th><th>Worst error / '
                 'tolerance</th></tr>')
    for mid, title, mods, anchor, cs in groups:
        worst = max((c.error / c.tolerance if c.tolerance > 0 and math.isfinite(c.error) else
                     (0.0 if c.error == 0 else math.inf)) for c in cs)
        parts.append(f'<tr><td><a href="#{mid}">{e(title)}</a></td><td>{len(cs)}</td>'
                     f'<td>{sum(c.passed for c in cs)}</td><td>{_num(worst)}</td></tr>')
    parts.append('</table></div>')
    used = []
    for mid, title, mods, anchor, cs in groups:
        parts.append(f'<h2 id="{mid}">{e(title)}</h2><p class="meta">Validates <code>{e(mods)}</code></p>')
        for c in cs:
            used += [r for r in c.refs if r not in used]
            refs = ', '.join(f'<a href="#ref-{r}">[{r}]</a>' for r in c.refs)
            rows = ''.join(f'<tr><td>{e(k)}</td><td>{e(v)}</td></tr>' for k, v in _flat(c.detail))
            parts.append(
                f'<div class="card" id="{e(c.id)}"><h3>{e(c.title)} {_badge(c.passed)}</h3>'
                f'<p class="meta"><code>{e(c.id)}</code> · {c.seconds:.2f} s</p>'
                f'<dl class="kv"><dt>Quantity</dt><dd>{e(c.quantity)}</dd>'
                f'<dt>Reference</dt><dd>{e(c.reference)} {refs}</dd>'
                f'<dt>Measured error</dt><dd><b>{_num(c.error, c.unit)}</b></dd>'
                f'<dt>Tolerance</dt><dd>{_num(c.tolerance, c.unit)}</dd></dl>'
                + (f'<p>{e(c.note)}</p>' if c.note else '')
                + (f'<p class="err">The check raised {e(c.exception)}</p>' if c.exception else '')
                + svg_figure(c.figure)
                + (f'<details><summary>Numbers behind it</summary><div class="wrap"><table>{rows}</table></div>'
                   f'</details>' if rows else '')
                + '</div>')
    parts.append('<h2 id="references">References</h2><ol class="refs">')
    parts += [f'<li id="ref-{k}"><b>[{k}]</b> {e(t)}</li>' for k, t in cite(used)]
    parts.append('</ol><h2 id="environment">Environment</h2><div class="wrap"><table>')
    parts += [f'<tr><td>{e(k)}</td><td>{e(str(v))}</td></tr>' for k, v in env.items()]
    parts.append('</table></div></main></body></html>\n')
    return ''.join(parts)


# ============================================================================ the technical reference's tables
REGION = '<!-- validation:{0} -->'
REGION_END = '<!-- /validation:{0} -->'


def _cell(v, unit, tol=None):
    """A table figure: two significant digits below 10⁻³, three above, four when it is within 1 % below its
    tolerance (so 0.9998 of 1 does not read as 1)."""
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return '—'
    if tol and 0.99 * tol <= v < tol:
        return f'{v:.4g} {unit}'.strip()
    return f'{v:.2g} {unit}'.strip() if abs(v) < 1e-3 and v != 0 else f'{v:.3g} {unit}'.strip()


def methods_fragments(result, report='validation/validation.html') -> dict:
    """HTML for site/methods.html's marked regions: 'summary' (a row per model), one table per model (its checks:
    title, reference, error, tolerance, verdict, each linked to the report) and 'references' (the list)."""
    e = html.escape
    env = result.environment
    run = (f'Vantage {e(str(env.get("vantage")))}, run {e(result.started_utc[:10])}, seed {result.settings.seed}'
           + (', quick' if result.settings.quick else ''))
    out = {}
    rows = []
    for mid, title, mods, anchor, cs in result.by_model():
        ok = sum(c.passed for c in cs)
        rows.append(f'<tr><td><a href="#{anchor}">{e(title)}</a></td><td>{len(cs)}</td><td>{ok}</td></tr>\n')
        body = ''.join(
            f'<tr><td><a href="{report}#{e(c.id)}">{e(c.title)}</a><br><code>{e(c.id)}</code></td>'
            f'<td>{e(c.reference)}</td><td>{_cell(c.error, e(c.unit), c.tolerance)}</td>'
            f'<td>{_cell(c.tolerance, e(c.unit))}</td>'
            f'<td class="{"ok" if c.passed else "no"}">{"✓ pass" if c.passed else "✕ fail"}</td></tr>\n' for c in cs)
        out[mid] = (f'<table class="val"><caption>{ok} of {len(cs)} checks passed ({run}); figures and the numbers '
                    f'behind each in the <a href="{report}#{mid}">validation report</a>.</caption>'
                    f'<tr><th>Check</th><th>Reference</th><th>Error</th><th>Tolerance</th><th>Result</th></tr>\n'
                    f'{body}</table>')
    n = len(result.checks)
    out['summary'] = (f'<table class="val"><caption>{result.passed} of {n} checks passed ({run}); the full '
                      f'<a href="{report}">validation report</a>.</caption><tr><th>Model</th><th>Checks</th>'
                      f'<th>Passed</th></tr>{"".join(rows)}</table>')
    out['references'] = '\n'.join(f'<li id="ref-{k}"><b>[{k}]</b> {e(t)}</li>' for k, t in REFERENCES.items())
    return out


def update_methods(path, result, report='validation/validation.html') -> list:
    """Replace the marked regions of the technical reference (site/methods.html) with this run's tables; returns the
    regions replaced. A region the page does not mark is left out; the rest of the page is not touched."""
    with open(path, encoding='utf-8') as fh:
        page = fh.read()
    done = []
    for name, frag in methods_fragments(result, report).items():
        a, b = REGION.format(name), REGION_END.format(name)
        i, j = page.find(a), page.find(b)
        if i < 0 or j < i:
            continue
        page = page[:i + len(a)] + '\n' + frag + '\n' + page[j:]
        done.append(name)
    tmp = path + '.part'
    with open(tmp, 'w', encoding='utf-8') as fh:
        fh.write(page)
    os.replace(tmp, path)
    return done
