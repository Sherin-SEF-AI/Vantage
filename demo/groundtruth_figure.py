"""The website's ground-truth figure: one camera frame of a CARLA drive recorded with ground truth, beside its
semantic, instance and range images (vantage.ground_truth/1), and the lidar sweep nearest it from above, every point
coloured by its nuScenes-lidarseg class.

    python3 groundtruth_figure.py DRIVE OUT.jpg [--camera F] [--fonts site/assets/fonts]

DRIVE is a Simulate ▸ CARLA run with ground truth on (ground_truth/index.json). The frame is the camera's with the most
objects of at least 150 visible pixels. Reads only; draws with matplotlib (nothing opens on a display).
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

DARK = dict(bg='#0f0f0e', ink='#f0efec', ink2='#c3c2b7', muted='#8f8d86', hair='#2a2a28', panel='#171716')
THINGS = ('car', 'truck', 'bus', 'pedestrian', 'rider', 'bicycle', 'motorcycle', 'traffic_light', 'traffic_sign')


def quat_matrix(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('drive')
    ap.add_argument('out')
    ap.add_argument('--camera', default='F')
    ap.add_argument('--frame', type=int, default=-1, help='the frame index (default: most object pixels)')
    ap.add_argument('--fonts', default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                    'site', 'assets', 'fonts'))
    ap.add_argument('--repo', default='')
    a = ap.parse_args()
    if a.repo:
        sys.path.insert(0, a.repo)
    import cv2
    from vantage.core import groundtruth as GT
    sans = 'DejaVu Sans'
    for f in ('IBMPlexSans-Regular.ttf', 'IBMPlexSans-Medium.ttf', 'IBMPlexSans-SemiBold.ttf'):
        p = os.path.join(a.fonts, f)
        if os.path.isfile(p):
            fm.fontManager.addfont(p)
            sans = 'IBM Plex Sans'
    C = DARK
    plt.rcParams.update({'font.family': sans, 'text.color': C['ink'], 'axes.edgecolor': C['hair'],
                         'xtick.color': C['muted'], 'ytick.color': C['muted']})

    doc = GT.read(a.drive)
    if doc is None:
        raise SystemExit(f'no ground truth in {a.drive}')
    run = json.load(open(os.path.join(a.drive, 'run.json')))
    frs = GT.frames(doc, a.camera)
    score = [sum(min(o['pixels'], 20000) for o in f['objects'] if o['class'] in THINGS) for f in frs]
    k = int(np.argmax(score)) if a.frame < 0 else a.frame
    fr = frs[k]
    t0 = frs[0]['timestamp']
    img = cv2.cvtColor(cv2.imread(GT.path_of(doc, fr['image']), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    sem = GT.load_semantic(doc, fr)
    inst = GT.load_instance(doc, fr)
    rng = GT.load_range(doc, fr)
    pal = GT.palette()
    names = {c['id']: c['name'] for c in doc['classes']}
    cols = {c['name']: np.array(c['color']) / 255.0 for c in doc['classes']}
    h, w = sem.shape

    # instance ids: a distinct, stable colour per object id (golden-ratio hues), nothing dark grey
    ids = np.unique(inst)
    ids = ids[ids > 0]
    lut = np.zeros((int(inst.max()) + 1, 3), np.float32) + np.array([0.09, 0.09, 0.086])
    import colorsys
    for i, oid in enumerate(ids):
        lut[oid] = colorsys.hsv_to_rgb((0.13 + 0.618034 * i) % 1.0, 0.62, 0.95)
    inst_rgb = lut[inst]
    # range: log scale, 0 = nothing within range
    valid = rng > 0
    lr = np.where(valid, np.log10(np.clip(rng, 1.0, 200.0)), np.nan)

    fig = plt.figure(figsize=(16, 9), dpi=100, facecolor=C['bg'])
    fig.text(0.03, 0.948, 'Ground truth from CARLA, through the fisheye lens model', fontsize=22, weight='semibold',
             color=C['ink'])
    sc = run.get('scene', {})
    tsec = (fr['timestamp'] - t0) / 1e6
    fig.text(0.03, 0.912, f'CARLA {run.get("server_version", "0.9.16")} · {sc.get("town", "")} · '
             f'{sc.get("weather", "")} · {sc.get("traffic", 0)} cars · seed {sc.get("seed", "")} · camera '
             f'{a.camera} at {tsec:.1f} s ({w} × {h}, five pinhole views remapped) · the roof lidar\'s sweep '
             'nearest it', fontsize=12.5, color=C['ink2'])

    pw, ph = 0.30, 0.385
    xs, ys = (0.03, 0.335), (0.465, 0.045)
    panels = [('Camera image, objects\' tight boxes', xs[0], ys[0]), ('Semantic labels', xs[1], ys[0]),
              ('Instances (one colour per object)', xs[0], ys[1]), ('Range along each pixel\'s ray', xs[1], ys[1])]
    axes = []
    for title, x, y in panels:
        ax = fig.add_axes([x, y, pw, ph], facecolor=C['bg'])
        ax.set_axis_off()
        ax.set_title(title, fontsize=13, color=C['ink2'], loc='left', pad=6)
        axes.append(ax)
    axes[0].imshow(img)
    n_box = 0
    for o in fr['objects']:
        if o['pixels'] < 40 or o['class'] not in THINGS:
            continue
        x0, y0, x1, y1 = o['bbox']
        axes[0].add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, lw=1.6,
                                    ec=np.clip(cols.get(o['class'], np.ones(3)) * 1.0 + 0.25, 0, 1)))
        n_box += 1
    axes[1].imshow(pal[sem])
    axes[2].imshow(inst_rgb)
    im = axes[3].imshow(lr, cmap='magma_r', vmin=0.0, vmax=np.log10(200.0))
    axes[3].set_facecolor('#000000')
    cax = fig.add_axes([xs[1] + pw - 0.13, ys[1] - 0.012, 0.11, 0.010])
    cb = fig.colorbar(im, cax=cax, orientation='horizontal', ticks=[0, 1, 2])
    cb.ax.set_xticklabels(['1 m', '10 m', '100 m'], fontsize=10, color=C['muted'])
    cb.outline.set_edgecolor(C['hair'])
    # the semantic legend: the classes in view
    present = [i for i in np.unique(sem) if (sem == i).mean() > 0.002]
    present.sort(key=lambda i: -(sem == i).mean())
    lx = 0.665
    fig.text(lx, 0.255, 'Semantic labels in view', fontsize=11, color=C['muted'])
    for j, i in enumerate(present[:8]):
        yy = 0.215 - j * 0.026
        fig.patches.append(Rectangle((lx, yy), 0.010, 0.016, transform=fig.transFigure, color=pal[i] / 255.0))
        fig.text(lx + 0.015, yy + 0.001, names.get(int(i), str(i)).replace('_', ' '), fontsize=10, color=C['ink2'])

    # the lidar sweep nearest the frame, from above, coloured by lidarseg class
    sweeps = doc['lidars'][next(iter(doc['lidars']))]['sweeps']
    sw = min(sweeps, key=lambda s: abs(s['timestamp'] - fr['timestamp']))
    V = os.path.join(a.drive, 'nuscenes', 'v1.0-vantage')
    sd = {r['token']: r for r in json.load(open(os.path.join(V, 'sample_data.json')))}
    cs = {r['token']: r for r in json.load(open(os.path.join(V, 'calibrated_sensor.json')))}
    rec = sd[sw['sample_data_token']]
    pts = np.fromfile(os.path.join(a.drive, 'nuscenes', rec['filename']), np.float32).reshape(-1, 5)[:, :3]
    cal = cs[rec['calibrated_sensor_token']]
    P = pts @ quat_matrix(cal['rotation']).T + np.array(cal['translation'])
    lab = GT.load_lidarseg(doc, sw)
    # nuScenes index -> the colour of the first Vantage class mapped to it
    ncol = np.full((32, 3), 0.5)
    nname = {}
    for c in reversed(doc['classes']):
        ncol[c['nuscenes_index']] = np.array(c['color']) / 255.0
        nname[c['nuscenes_index']] = c['nuscenes']
    ncol[24] = np.array([0.42, 0.40, 0.46])         # driveable surface: lighter than the road's own colour on black
    lum = ncol @ np.array([0.2126, 0.7152, 0.0722])
    dark = lum < 0.33
    ncol[dark] = ncol[dark] + (1.0 - ncol[dark]) * 0.45   # dark classes (cars are navy) lifted to show on black
    R = 42.0
    keep = (np.abs(P[:, 0]) < R) & (np.abs(P[:, 1]) < R)
    lx0 = 0.665
    side = 0.305
    ax = fig.add_axes([lx0, 0.315, side, side * 16 / 9 * 0.98], facecolor=C['panel'])
    order = np.argsort(np.isin(lab, (17, 23, 15, 16, 2, 3, 14, 21)))      # things drawn last, on top
    o = order[keep[order]]
    ax.scatter(P[o, 1] * -1, P[o, 0], s=0.35, c=ncol[lab[o]], linewidths=0, rasterized=True)
    ax.add_patch(Rectangle((-0.9, -0.9), 1.8, 4.2, fill=True, color=C['ink'], alpha=0.9))
    ax.set_xlim(-R, R)
    ax.set_ylim(-R, R)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    for s_ in ax.spines.values():
        s_.set_color(C['hair'])
    ax.set_title(f'The roof lidar: {len(lab):,} points by nuScenes-lidarseg class', fontsize=13, color=C['ink2'],
                 loc='left', pad=6)
    fig.text(lx0, 0.288, f'{2 * R:.0f} m across, from above; the car in white, heading up', fontsize=10,
             color=C['muted'])
    # its legend: the classes with the most points
    u, n = np.unique(lab[keep], return_counts=True)
    top = [i for i in u[np.argsort(-n)] if i != 0][:8]
    lx1 = lx0 + 0.13
    fig.text(lx1, 0.255, 'Lidar classes (nuScenes)', fontsize=11, color=C['muted'])
    for j, i in enumerate(top):
        yy = 0.215 - j * 0.026
        fig.patches.append(Rectangle((lx1, yy), 0.010, 0.016, transform=fig.transFigure, color=ncol[i]))
        fig.text(lx1 + 0.015, yy + 0.001, nname.get(int(i), str(i)), fontsize=10, color=C['ink2'])
    fig.savefig(a.out, facecolor=C['bg'], dpi=100, pil_kwargs={'quality': 88})
    print(f'frame {k} of {len(frs)} ({tsec:.1f} s), {len(fr["objects"])} objects, {n_box} boxes drawn, '
          f'{len(ids)} instances in view; sweep {sw["sample_data_token"][:8]}, {len(lab)} points; '
          f'{len(present)} classes in view')


if __name__ == '__main__':
    main()
