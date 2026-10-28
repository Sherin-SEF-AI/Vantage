"""51 The rig in CARLA (episode gt040, chapter 1): the drive gt_drive_walkers (in $VANTAGE_GT_DRIVES, read only) at
real speed: the six fisheyes with their objects' tight boxes; how a fisheye's labels are made (its pinhole views, the
instance and depth images beside them, read through the lens's remap at the nearest view pixel); the right mirror's
image beside its semantic labels, and both magnified on a car's edge, pixel for pixel.

Figures narrated, each read from the drive's own files while recording (r.expect): CARLA 0.9.16 and Town10HD
(run.json scene: carla_version, town Town10HD_Opt), six fisheyes (ground_truth/index.json cameras), the 128-beam roof
lidar (run.json notes: "L1: 128 evenly spaced beams"); the label cameras "same size, field, pose and step", read "at
the nearest view pixel ... never interpolated" (index.json notes). Five pinhole views per fisheye: the reference
drive's rig (simcore report: "6 fisheye cameras at 960 × 768 (5 cube views each)"; docs: "a fisheye is five").
"""

META = {
    'id': 'gt-rig',
    'part': 1,
    'title': 'The rig in CARLA',
    'promise': 'Six fisheyes and a roof lidar, and how their labels are made.',
    'menu': 'Simulate ▸ CARLA ▸ Ground truth',
}

LINES = [
    # 0
    ("The sample's data rig in CARLA 0.9.16: six fisheye cameras and a 128-beam roof lidar, in Town10HD traffic.",
     "The sample's data rig in CARLA zero point nine point sixteen: six fisheye cameras, and a hundred and "
     "twenty-eight beam roof lidar, in Town ten H D traffic."),
    # 1
    ("Each fisheye is five pinhole views joined through its lens's remap. Beside each view, CARLA renders an "
     "instance image and a depth image, from the same pose on the same step.",
     "Each fisheye is five pinhole views, joined through its lens's remap. Beside each view, CARLA renders an "
     "instance image, and a depth image, from the same pose on the same step."),
    # 2
    ("Vantage reads them through the same remap at the nearest view pixel: magnified, every label pixel is one "
     "class, never a blend.",
     "Vantage reads them through the same remap, at the nearest view pixel: magnified, every label pixel is one "
     "class, never a blend."),
]


DRIVE = 'gt_drive_walkers'
PAIR = ('MR', 4.0, 5.0)        # the right mirror, played from 4.0 s to 5.0 s, then held for the magnified edge
LOUPE_OBJ = 1015               # the car whose edge is magnified (a parked car of the map: id 1000 + its index)


def edge_point(d, tag, k, oid):
    """A point of an object's outline (instance image) where the colour image changes most across it: there the
    image's anti-aliased edge and the labels' hard one differ most."""
    import cv2
    import numpy as np
    inst = d.raw(tag, k, 'instance')
    img = d.raw(tag, k, 'image')
    m = (inst == oid).astype(np.uint8)
    edge = (m > 0) & (cv2.erode(m, np.ones((3, 3), np.uint8)) == 0)
    grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    g = np.hypot(cv2.Sobel(grey, cv2.CV_32F, 1, 0), cv2.Sobel(grey, cv2.CV_32F, 0, 1))
    g = cv2.blur(g * edge, (15, 15))
    g[:20], g[-20:], g[:, :24], g[:, -24:] = 0, 0, 0, 0
    y, x = np.unravel_index(int(np.argmax(g)), g.shape)
    return float(x), float(y)


def scene(stage, r):
    import gtfootage as G

    v = G.start(stage, r)
    d = G.Drive(DRIVE)
    run, idx = d.run, d.doc
    sc = run['scene']
    print('scene:', sc, flush=True)
    r.expect(sc['carla_version'], '0.9.16', 'CARLA version')
    r.expect(sc['town'].startswith('Town10HD'), True, 'the town (Town10HD_Opt)')
    r.expect(len(d.tags), 6, 'six fisheyes')
    r.expect(any(n.startswith('L1: 128 evenly spaced beams') for n in run.get('notes', [])), True, '128-beam lidar')
    notes = ' '.join(idx.get('notes', []))
    r.expect('same size, field, pose and step' in notes, True, 'label cameras beside every view')
    r.expect('read at the nearest view pixel' in notes and 'never interpolated' in notes, True, 'nearest view pixel')
    r.expect(d.size('MR'), (960, 768), 'the fisheye size')

    # ---- 0: the six fisheyes at real speed
    six = v.use(G.SixGrid(d, title=f'Real CARLA {sc["carla_version"]} drive · Town10HD · {sc["traffic"]} vehicles, '
                                   f'{run["walkers"]} pedestrians · seed {sc["seed"]}'))
    v.clock.play(0.0)
    r.lower_third()
    r.hold(0.3)
    r.say(0)
    r.until_word(0, 'six fisheye')
    r.callout(lambda: six.rect('ML').united(six.rect('QR')), '', dur=2.0, pad=10, dim=0.25)
    r.finish_speech(0.1)

    # ---- 1: five pinhole views per fisheye; the label cameras beside them
    r.say(1)
    r.zoom(six.rect('F'), 1.0, margin=40)
    r.until_word(1, 'Beside each view')
    tag, t_a, t_b = PAIR
    pair = G.Pair(d, tag, title=f'{tag} · {G.CAM_NAMES[tag]} camera · its image and its labels')
    v.use(pair)
    v.clock.play(t_a)
    r.zoom(None, 0)
    r.crossfade(0.5)
    r.hold(0.6)
    r.callout(pair.rect('semantic'), '', dur=max(2.0, r.speech_left() - 0.6), pad=6, dim=0.3)
    G.play_until(r, v, t_b)
    r.finish_speech(0.1)

    # ---- 2: read at the nearest view pixel: magnified on a car's edge
    k = d.index_at(tag, t_b)
    car = next(o for o in d.frame(tag, k)['objects'] if o['id'] == LOUPE_OBJ)
    x0, y0, x1, y1 = car['bbox']
    print('car', car['class'], car['id'], car['bbox'], car['pixels'], flush=True)
    cx, cy = edge_point(d, tag, k, LOUPE_OBJ)                       # on its outline, where the image's edge is sharpest
    print('loupe at', cx, cy, flush=True)
    pair.loupe = (round(cx - 20), round(cy - 16), round(cx + 20), round(cy + 16))
    f_l = r.f
    pair.loupe_k = (lambda: min(1.0, (r.f - f_l) / (0.5 * r.fps)))
    r.say(2)
    r.hold(0.6)
    r.until_word(2, 'magnified')
    lr = pair.loupe_rects()
    r.zoom(lr['semantic'][1].united(lr['semantic'][0]), 1.0, margin=60)
    r.hold(max(1.5, r.speech_left() - 0.6))
    r.zoom(None, 0.8)
    r.finish_speech(1.0)
    d.release()
