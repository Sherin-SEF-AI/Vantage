"""The website's validation figure: every check of the published report (site/validation/validation.json) as its
measured error divided by its tolerance, one row per model, log scale; plus the cube-map depth check as it was before
0.3.0 (the integrator's measurement with HEAD's occlusion.py: 0.715 share of texels against 0.005).

    python3 demo/validation_figure.py site/validation/validation.json site/media/validation.jpg

Re-run it after publishing the report again (python -m vantage.validation site/validation --methods site/methods.html):
the figure prints the report's version and date.
"""
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import numpy as np

FONTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'site', 'assets', 'fonts', '')
for f in ('IBMPlexSans-Regular.ttf', 'IBMPlexSans-Medium.ttf', 'IBMPlexSans-SemiBold.ttf', 'IBMPlexMono-Regular.ttf'):
    fm.fontManager.addfont(FONTS + f)
SANS, MONO = 'IBM Plex Sans', 'IBM Plex Mono'

DARK = dict(bg='#0f0f0e', ink='#f0efec', ink2='#c3c2b7', muted='#8f8d86', hair='#2a2a28', strong='#3a3a37',
            dot='#3987e5', fail='#e0524f', ghost='#8f8d86')
OLD_CUBE_DEPTH = 0.715            # the share of texels off before 0.3.0 (integrator7.md), tolerance 0.005
ZERO_X = 3e-17                    # where an exact zero is drawn (left of the axis break)
X0, X1 = 1e-15, 1e3


def main(src, out):
    d = json.load(open(src))
    C = DARK
    plt.rcParams.update({'mathtext.fontset': 'custom', 'mathtext.rm': SANS, 'mathtext.it': SANS, 'mathtext.bf': SANS,
                         'font.family': SANS, 'text.color': C['ink'], 'axes.edgecolor': C['hair'],
                         'xtick.color': C['muted'], 'ytick.color': C['ink2']})
    fig = plt.figure(figsize=(16, 9), dpi=100, facecolor=C['bg'])
    ax = fig.add_axes([0.235, 0.10, 0.72, 0.70], facecolor=C['bg'])
    models = d['models']
    n = len(models)
    ys = np.arange(n)[::-1]
    ratios = []
    for y, m in zip(ys, models):
        for c in m['checks']:
            e, t = float(c['error']), float(c['tolerance'])
            r = 0.0 if e == 0 else e / t
            ratios.append((y, r, c, m))
    # the fail side and the tolerance line
    ax.axvspan(1.0, X1, color=C['fail'], alpha=0.07, lw=0)
    ax.axvline(1.0, color=C['fail'], lw=1.5, alpha=0.9)
    ax.text(1.0, n - 0.35, ' tolerance', color=C['ink2'], fontsize=13, va='bottom', ha='left')
    ax.text(30, -0.95, 'fail', color=C['muted'], fontsize=13, ha='center', va='center')
    # row hairlines
    for y in ys:
        ax.axhline(y, color=C['hair'], lw=1, zorder=0)
    # an axis break between the exact zeros and the log axis
    ax.axvspan(1.2e-16, 6e-16, color=C['bg'], lw=0, zorder=1)
    # the dots
    xs = [ZERO_X if r == 0 else r for _, r, _, _ in ratios]
    yy = [y for y, _, _, _ in ratios]
    ax.scatter(xs, yy, s=110, color=C['dot'], edgecolors=C['bg'], linewidths=2, zorder=4)
    # the cube-map depth before 0.3.0
    ym = next(y for y, m in zip(ys, models) if m['id'] == 'occlusion')
    now = next(r for y, r, c, m in ratios if c['id'] == 'occlusion.cube_depth')
    old = OLD_CUBE_DEPTH / 0.005
    ax.scatter([old], [ym], s=110, facecolors='none', edgecolors=C['ghost'], linewidths=2, zorder=4)
    ax.annotate('', xy=(now * 1.6, ym + 0.02), xytext=(old / 1.6, ym + 0.02),
                arrowprops=dict(arrowstyle='-|>', color=C['ghost'], lw=1.5, shrinkA=0, shrinkB=0,
                                connectionstyle='arc3,rad=0.45', mutation_scale=14), zorder=3)
    ax.text(0.045, ym, f'cube-map depth: {old:.0f}× before 0.3.0, {now:.2f}× now', color=C['ink2'],
            fontsize=12.5, va='center', ha='right', zorder=5,
            bbox=dict(boxstyle='square,pad=0.25', facecolor=C['bg'], edgecolor='none'))
    # the closest call
    y0, r0, c0, _ = max(ratios, key=lambda t: t[1])
    ax.text(r0 / 1.5, y0 + 0.42, 'P.838-3 table: 0.9998 (the table is the equations rounded)', color=C['ink2'],
            fontsize=12, ha='right', va='center')
    ax.set_xscale('log')
    ax.set_xlim(1e-17, X1)
    ax.set_ylim(-1.3, n - 0.3)
    ticks = [ZERO_X, 1e-12, 1e-9, 1e-6, 1e-3, 1, 1e3]
    ax.set_xticks(ticks)
    ax.set_xticklabels(['0', '$10^{-12}$', '$10^{-9}$', '$10^{-6}$', '$10^{-3}$', '1', '1000'], fontsize=14)
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_yticks(ys)
    ax.set_yticklabels([f'{m["title"]}' for m in models], fontsize=13.5)
    for lab, m in zip(ax.get_yticklabels(), models):
        lab.set_color(C['ink2'])
    for y, m in zip(ys, models):
        ax.text(1.4e3, y, f' {m["passed"]}/{len(m["checks"])}', color=C['muted'], fontsize=12, family=MONO,
                va='center', ha='left')
    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)
    ax.spines['bottom'].set_color(C['strong'])
    ax.tick_params(axis='y', length=0, pad=10)
    ax.tick_params(axis='x', length=4, color=C['strong'])
    ax.set_xlabel('measured error ÷ tolerance  (log scale; a dot left of the line passes)', color=C['muted'],
                  fontsize=13, labelpad=10)
    s = d['summary']
    fig.text(0.035, 0.925, f'{s["checks"]} checks of {n} models against independent references: '
                           f'{s["passed"]} pass', fontsize=26, weight='semibold', color=C['ink'])
    env = d['environment']
    fig.text(0.035, 0.875, f'Each dot is one check: the error it measured against a closed form, OpenCV, a published table '
                           f'or example, exact ray casting or Monte Carlo, divided by its tolerance.',
             fontsize=14, color=C['ink2'])
    fig.text(0.035, 0.845, f'Vantage {env["vantage"]}, the built-in sample, seed {d["settings"]["seed"]}, '
                           f'{d["created_utc"][:10]}: the published validation report.', fontsize=14, color=C['muted'])
    fig.savefig(out, dpi=100, facecolor=C['bg'], pil_kwargs={'quality': 90})
    print(out, len(ratios), 'checks; max ratio', max(r for _, r, _, _ in ratios))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
