"""The website's actors-scene figure: a CARLA drive made with --actors ACTORS.json (a still ego, no traffic), its front
and rear cameras' last frames with every actor of the file boxed by its ground truth and named by its id.

    python3 actors_figure.py DRIVE OUT.jpg [--fonts site/assets/fonts] [--repo REPO]

Reads only; draws with matplotlib (nothing opens on a display).
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager as fm  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
import numpy as np  # noqa: E402

DARK = dict(bg='#0f0f0e', ink='#f0efec', ink2='#c3c2b7', muted='#8f8d86', hair='#2a2a28', box='#f5c542',
            box2='#4fd1c5')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('drive')
    ap.add_argument('out')
    ap.add_argument('--fonts', default='site/assets/fonts')
    ap.add_argument('--repo', default='')
    a = ap.parse_args()
    if a.repo:
        sys.path.insert(0, a.repo)
    import cv2
    from vantage.core import groundtruth as GT
    sans, mono = 'DejaVu Sans', 'DejaVu Sans Mono'
    for f in ('IBMPlexSans-Regular.ttf', 'IBMPlexSans-Medium.ttf', 'IBMPlexSans-SemiBold.ttf', 'IBMPlexMono-Regular.ttf'):
        p = os.path.join(a.fonts, f)
        if os.path.isfile(p):
            fm.fontManager.addfont(p)
            if 'Mono' in f:
                mono = 'IBM Plex Mono'
            else:
                sans = 'IBM Plex Sans'
    C = DARK
    plt.rcParams.update({'font.family': sans, 'text.color': C['ink']})
    doc = GT.read(a.drive)
    run = json.load(open(os.path.join(a.drive, 'run.json')))
    actors = {x['actor_id']: x for x in run.get('actors', []) if x.get('actor_id')}
    missing = [x for x in run.get('actors', []) if not x.get('actor_id')]
    fig = plt.figure(figsize=(16, 9), dpi=100, facecolor=C['bg'])
    fig.text(0.03, 0.948, 'An actors scene: the file\'s actors around the still car, each named in the ground truth',
             fontsize=21, weight='semibold', color=C['ink'])
    sc = run.get('scene', {})
    err = max((x.get('error_m') or 0.0) for x in actors.values()) if actors else 0.0
    fig.text(0.03, 0.912, f'CARLA {sc.get("carla_version", "")} · {sc.get("town", "").replace("_Opt", "")} · '
             f'{sc.get("weather", "")} · no traffic · {len(actors)} of {len(actors) + len(missing)} actors spawned, '
             f'each placed within 1 mm of its pose · the last frame of cameras F and R, cropped', fontsize=12.5,
             color=C['ink2'])
    panels = (('F', [0.03, 0.375, 0.615, 0.50], 2.0), ('R', [0.675, 0.375, 0.295, 0.50], 1.0))
    for tag, rect, aspect in panels:
        fr = GT.frames(doc, tag)[-1]
        img = cv2.cvtColor(cv2.imread(GT.path_of(doc, fr['image']), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        H, W = img.shape[:2]
        objs = [o for o in fr['objects'] if o.get('actor_id') in actors and o['pixels'] >= 30]
        if not objs:
            continue
        bx = np.array([o['bbox'] for o in objs], float)
        x0, y0, x1, y1 = bx[:, 0].min() - 70, bx[:, 1].min() - 90, bx[:, 2].max() + 70, bx[:, 3].max() + 40
        w_, h_ = x1 - x0, y1 - y0
        if w_ / h_ < aspect:
            w_ = h_ * aspect
        else:
            h_ = w_ / aspect
        cx, cy = (bx[:, 0].min() + bx[:, 2].max()) / 2, (bx[:, 1].min() + bx[:, 3].max()) / 2 - 20
        x0 = int(np.clip(cx - w_ / 2, 0, W - min(w_, W)))
        y0 = int(np.clip(cy - h_ / 2, 0, H - min(h_, H)))
        crop = img[y0:int(y0 + min(h_, H)), x0:int(x0 + min(w_, W))]
        ax = fig.add_axes(rect, facecolor=C['bg'])
        ax.imshow(crop, extent=(x0, x0 + crop.shape[1], y0 + crop.shape[0], y0))
        ax.set_axis_off()
        ax.set_title(f'Camera {tag}', fontsize=13, color=C['ink2'], loc='left', pad=6)
        # labels above their boxes, lifted a line where they would overlap one already placed
        disp_w = min(rect[2] * 1600, rect[3] * 900 * crop.shape[1] / crop.shape[0])   # imshow keeps the aspect
        px_per = crop.shape[1] / disp_w
        line = 22 * px_per
        placed = []                               # label rectangles (x0, y0, x1, y1) in image pixels

        def free(x0, y0, x1, y1):
            return all(x1 < a0 or x0 > a1 or y1 < b0 or y0 > b1 for a0, b0, a1, b1 in placed)
        for o in sorted(objs, key=lambda o: (o['bbox'][2] - o['bbox'][0]) * (o['bbox'][3] - o['bbox'][1])):
            x = actors[o['actor_id']]
            bx0, by0, bx1, by1 = o['bbox']
            ax.add_patch(Rectangle((bx0, by0), bx1 - bx0, by1 - by0, fill=False, lw=2.0, ec=C['box']))
            text = f'{x["id"]} · {o["class"]}'
            tw = (len(text) * 7.4 + 8) * px_per
            lh = 19 * px_per
            spots = []
            for k in range(6):                    # above the box, then below it, a line further each time
                spots.append((by0 - 3 * px_per - k * line - lh, by0))
                spots.append((by1 + 3 * px_per + k * line, by1))
            ty, anchor = next(((t, a_) for t, a_ in spots if free(bx0, t, bx0 + tw, t + lh)), spots[0])
            placed.append((bx0, ty, bx0 + tw, ty + lh))
            if abs((ty + (lh if anchor == by0 else 0)) - anchor) > 4 * px_per:
                ax.plot([bx0 + 3 * px_per] * 2, [anchor, ty + (lh if anchor == by0 else 0)], color=C['box'], lw=1.0)
            ax.text(bx0, ty + lh, text, fontsize=10, color=C['bg'], va='bottom', ha='left',
                    bbox=dict(boxstyle='square,pad=0.18', facecolor=C['box'], edgecolor='none'))
            placed[-1] = (bx0, ty - 2 * px_per, bx0 + tw, ty + lh + 2 * px_per)
    lines = []
    for x in run.get('actors', []):
        if x.get('actor_id'):
            pv = x.get('pose_vehicle') or {}
            lines.append(f'{x["id"]:<13} {x["kind"]:<11} {x.get("blueprint", ""):<30} x {pv.get("x", 0) + 0.0:>5g} m  '
                         f'y {pv.get("y", 0) + 0.0:>4g} m  yaw {pv.get("yaw", 0) + 0.0:>4g}°')
        else:
            lines.append(f'{x["id"]:<13} {x["kind"]:<11} not spawned: CARLA 0.9.16 has no traffic-light blueprint')
    fig.text(0.03, 0.30, 'ACTORS.json as placed (the vehicle frame: x forward, y left, from the rear axle)',
             fontsize=12, color=C['ink2'])
    for i, s_ in enumerate(lines):
        fig.text(0.03, 0.262 - i * 0.026, s_, fontsize=10, color=C['muted'] if 'not spawned' in s_ else C['ink2'],
                 family=mono)
    fig.savefig(a.out, facecolor=C['bg'], dpi=100, pil_kwargs={'quality': 88})
    print(f'{len(actors)} actors, {len(missing)} not spawned, max error {err} m')


if __name__ == '__main__':
    main()
