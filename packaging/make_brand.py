"""Write the Vantage logo kit (the Surround mark: the car from above inside four coverage arcs) as SVG files in
branding/: the mark at three optical sizes, the wordmark and lockups in black and white, and the app icon tile.
packaging/make_icon.py renders the application icons and the website's images from these files.

    python3 packaging/make_brand.py ARCHIVO_VARIABLE_TTF

The wordmark is Archivo (SIL Open Font License 1.1) at weight 600 and width 125, tracked 0.30 em, converted to
outlines here, so the SVGs need no font. The font file itself is not stored in the repository.
"""
from __future__ import annotations

import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'branding')
INK, PAPER, TILE = '#EDEDE8', '#0A0B0B', '#0A0B0B'

# the mark on a 100-unit grid, per optical size: ring radius, ring stroke, gap (along the ring), body rectangle
SIZES = {
    'master': dict(r=38, w=7, gap=14, body=(41, 33, 18, 34, 5)),
    '32': dict(r=37, w=10, gap=16, body=(39, 31, 22, 38, 5)),
    '16': dict(r=35, w=14, gap=26, body=(38, 31, 24, 38, 4)),
}


def arc_paths(r, gap, cx=50.0, cy=50.0):
    """Four arcs of radius r, the gaps centred on the car's axes (front, rear, left, right)."""
    half = math.degrees(gap / 2 / r)
    out = []
    for k in range(4):
        a0 = math.radians(k * 90 + half)
        a1 = math.radians((k + 1) * 90 - half)
        x0, y0 = cx + r * math.cos(a0), cy + r * math.sin(a0)
        x1, y1 = cx + r * math.cos(a1), cy + r * math.sin(a1)
        out.append(f'M{x0:.3f} {y0:.3f}A{r} {r} 0 0 1 {x1:.3f} {y1:.3f}')
    return ''.join(out)


def mark_body(size='master', color='currentColor'):
    s = SIZES[size]
    x, y, w, h, rx = s['body']
    return (f'<path d="{arc_paths(s["r"], s["gap"])}" fill="none" stroke="{color}" stroke-width="{s["w"]}" '
            f'stroke-linecap="butt"/>'
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{color}"/>')


def svg(view, body, title):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{view}" role="img" aria-label="{title}">'
            f'<title>{title}</title>{body}</svg>\n')


def wordmark(font_path):
    """'VANTAGE' as outlines: (path d, width, cap height), in font units scaled to a cap height of 100."""
    from fontTools.pens.svgPathPen import SVGPathPen
    from fontTools.pens.transformPen import TransformPen
    from fontTools.ttLib import TTFont
    from fontTools.varLib.instancer import instantiateVariableFont
    f = instantiateVariableFont(TTFont(font_path), {'wght': 600, 'wdth': 125})
    upm = f['head'].unitsPerEm
    cap = f['OS/2'].sCapHeight or 0.7 * upm
    k = 100.0 / cap
    gs = f.getGlyphSet()
    cmap = f.getBestCmap()
    track = 0.30 * upm
    pen = SVGPathPen(gs)
    x = 0.0
    text = 'VANTAGE'
    for i, ch in enumerate(text):
        g = cmap[ord(ch)]
        tp = TransformPen(pen, (k, 0, 0, -k, x * k, 100.0))     # flip y, baseline at y = 100 (cap height 100)
        gs[g].draw(tp)
        x += gs[g].width + (track if i < len(text) - 1 else 0)
    # the last glyph's right side bearing stays, as type does; the width ends at its advance
    return pen.getCommands(), x * k


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    os.makedirs(OUT, exist_ok=True)
    d_word, w_word = wordmark(sys.argv[1])
    files = {}
    for size in SIZES:
        sfx = '' if size == 'master' else f'-{size}'
        for name, col in (('black', PAPER), ('white', INK)):
            files[f'mark{sfx}-{name}.svg'] = svg('0 0 100 100', mark_body(size, col), 'Vantage')
    for name, col in (('black', PAPER), ('white', INK)):
        files[f'wordmark-{name}.svg'] = svg(f'0 0 {w_word:.1f} 100', f'<path d="{d_word}" fill="{col}"/>', 'Vantage')
        # horizontal lockup: the mark 100 high, the wordmark's capitals 40 high, centred on the mark, 40 apart
        s = 0.40
        tx = 100 + 40
        body = (f'{mark_body("master", col)}'
                f'<g transform="translate({tx} {50 - 50 * s}) scale({s})"><path d="{d_word}" fill="{col}"/></g>')
        files[f'lockup-horizontal-{name}.svg'] = svg(f'0 0 {tx + w_word * s:.1f} 100', body, 'Vantage')
        # stacked lockup: the mark over the wordmark, capitals 22 high, centred
        s2 = 0.22
        W = max(100, w_word * s2)
        body = (f'<g transform="translate({(W - 100) / 2:.1f} 0)">{mark_body("master", col)}</g>'
                f'<g transform="translate({(W - w_word * s2) / 2:.1f} 124) scale({s2})"><path d="{d_word}" '
                f'fill="{col}"/></g>')
        files[f'lockup-stacked-{name}.svg'] = svg(f'0 0 {W:.1f} 146', body, 'Vantage')
    # app icon tiles: the mark at 60 % of a rounded square, per optical size
    for size in SIZES:
        sfx = '' if size == 'master' else f'-{size}'
        for name, bg, fg in (('dark', TILE, INK), ('light', '#F2F1EC', PAPER)):
            k = {'master': 0.6, '32': 0.7, '16': 0.86}[size]          # small tiles: the mark fills more of it
            o = 50 - 50 * k
            body = (f'<rect width="100" height="100" rx="{22 if size == "master" else 16}" fill="{bg}"/>'
                    f'<g transform="translate({o:g} {o:g}) scale({k})">{mark_body(size, fg)}</g>')
            files[f'app-icon{sfx}-{name}.svg'] = svg('0 0 100 100', body, 'Vantage')
    for fn, text in files.items():
        with open(os.path.join(OUT, fn), 'w', encoding='utf-8') as fh:
            fh.write(text)
    print(f'wrote {len(files)} files to {OUT}')


if __name__ == '__main__':
    main()
