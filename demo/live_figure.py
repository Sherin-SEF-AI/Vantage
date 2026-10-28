"""The website's CARLA live rig figure, from real live sessions (vantage.sim.live on a CARLA 0.9.16 server): one camera's
frames as CARLA served them, as planned, turned 5° in pitch and cleared, with rig health's body-edge drift against
CARLA's own car and against the project's body; and the frame rates CARLA reached.

    python3 live_figure.py SESSION OUT.jpg --rates RATES.json [--camera F] [--fonts site/assets/fonts]

SESSION holds <camera>-plan.jpg, -pitch+5.jpg, -cleared.jpg and summary.json (drift medians per phase). RATES.json:
[[label, frames per second], ...] as measured. Reads only; draws with matplotlib.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager as fm  # noqa: E402

DARK = dict(bg='#0f0f0e', ink='#f0efec', ink2='#c3c2b7', muted='#8f8d86', hair='#2a2a28', bar='#3987e5',
            warn='#e8a33d')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('session')
    ap.add_argument('out')
    ap.add_argument('--rates', required=True)
    ap.add_argument('--camera', default='F')
    ap.add_argument('--fonts', default='site/assets/fonts')
    a = ap.parse_args()
    sans = 'DejaVu Sans'
    for f in ('IBMPlexSans-Regular.ttf', 'IBMPlexSans-Medium.ttf', 'IBMPlexSans-SemiBold.ttf'):
        p = os.path.join(a.fonts, f)
        if os.path.isfile(p):
            fm.fontManager.addfont(p)
            sans = 'IBM Plex Sans'
    C = DARK
    plt.rcParams.update({'font.family': sans, 'text.color': C['ink'], 'axes.edgecolor': C['hair'],
                         'xtick.color': C['muted'], 'ytick.color': C['ink2']})
    s = json.load(open(os.path.join(a.session, 'summary.json')))
    drift = s['drift'][a.camera]
    st = s['status']['cameras'][a.camera]
    fig = plt.figure(figsize=(16, 9), dpi=100, facecolor=C['bg'])
    fig.text(0.03, 0.948, 'CARLA as a live rig: the stream rig health watches', fontsize=22, weight='semibold')
    fig.text(0.03, 0.912, f'CARLA {s["status"].get("server_version", "")} · {s["status"].get("town", "")} · '
             f'{s["status"].get("traffic", 0)} cars · camera {a.camera} served at {st["size"][0]} × {st["size"][1]} '
             f'from {st["render_size"][0]} × {st["render_size"][1]} views · real live frames', fontsize=12.5,
             color=C['ink2'])
    phases = (('plan', 'As planned'), ('pitch +5', 'Turned 5° in pitch, live'), ('cleared', 'Fault cleared'))
    files = {'plan': 'plan', 'pitch +5': 'pitch+5', 'cleared': 'cleared'}
    w, h, x0, y0 = 0.30, 0.40, 0.03, 0.43
    for i, (k, title) in enumerate(phases):
        img = plt.imread(os.path.join(a.session, f'{a.camera}-{files[k]}.jpg'))
        ax = fig.add_axes([x0 + i * (w + 0.0125), y0, w, h])
        ax.imshow(img)
        ax.set_axis_off()
        ax.set_title(title, fontsize=13, color=C['ink2'], loc='left', pad=6)
        dc, dp = drift['carla'][k]['median'], drift['project'][k]['median']
        ax.text(0.0, -0.06, f'drift against CARLA\'s car {dc:g} px', transform=ax.transAxes, fontsize=12,
                color=C['warn'] if dc >= 12 else C['ink'], va='top')
        ax.text(0.0, -0.13, f'against the project\'s body {dp:g} px', transform=ax.transAxes, fontsize=11,
                color=C['muted'], va='top')
    # frame rates CARLA reached
    rates = json.load(open(a.rates))
    ax = fig.add_axes([0.24, 0.07, 0.70, 0.20], facecolor=C['bg'])
    labels = [r[0] for r in rates][::-1]
    vals = [r[1] for r in rates][::-1]
    ax.barh(range(len(vals)), vals, color=C['bar'], height=0.62)
    for i, v in enumerate(vals):
        ax.text(v + 0.4, i, f'{v:g} fps', va='center', fontsize=11.5, color=C['ink'])
    ax.axvline(30, color=C['muted'], lw=1.2, ls='--')
    ax.text(30, len(vals) - 0.35, ' real time (30 fps)', fontsize=10.5, color=C['muted'], va='bottom')
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(labels, fontsize=11.5)
    ax.tick_params(length=0)
    ax.set_xlim(0, 36)
    ax.set_xticks([])
    for sp in ('top', 'right', 'bottom'):
        ax.spines[sp].set_visible(False)
    fig.text(0.03, 0.29, 'Frames a second each camera reached (RTX 5080)', fontsize=12, color=C['ink2'])
    fig.text(0.03, 0.33, "Rig health's body-edge drift warns at 12 px. CARLA renders its own car, not the project's "
             "body, so the drift is measured against CARLA's car as each camera sees it at its planned mount.",
             fontsize=11, color=C['muted'])
    fig.savefig(a.out, facecolor=C['bg'], dpi=100, pil_kwargs={'quality': 88})
    print('ok')


if __name__ == '__main__':
    main()
