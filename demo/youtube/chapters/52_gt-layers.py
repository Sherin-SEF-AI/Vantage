"""52 Four layers on one camera (episode gt040, chapter 2): the right mirror (MR) of the drive gt_drive_walkers (in
$VANTAGE_GT_DRIVES, read only) with its four layers and the roof lidar from above, played at real speed and held
(marked "paused" in the header) while a layer is explained: the tight boxes, the frame's objects and their full boxes,
the semantic classes and the ego, one pedestrian's instance across the drive, the range.

Figures narrated: the frame's 16 objects (ground_truth/index.json, MR frame 27, captured at 2.625 s, every object
listed whatever its size; the boxes drawn are those of 40 px or more), CARLA's 29 classes plus the ego (index.json
classes: ids 0-28 with a CARLA label, 29 'ego'), pedestrian 56 in 193 frames of four cameras (index.json: its id in F,
MR, QR and R), read while recording and checked with r.expect; the range against the lidar, a median of -0.9 mm on the
ground, is the measurement on the same drive without walkers (gt_drive) in the simcore report ("Ground: 258,276
points, median -0.9 mm"; CHANGELOG 0.4.0: "ranges agree with the lidar to -0.9 mm median"), said so on screen.
"""

META = {
    'id': 'gt-layers',
    'part': 1,
    'title': 'Four layers on one camera',
    'promise': 'Boxes, classes, instances and range, frame by frame.',
    'menu': 'ground_truth/CAM_MR · semantic, instance and range PNGs',
}

LINES = [
    # 0
    ("One camera, four layers, at real speed. Boxes: each object's tight box comes from its own pixels.",
     "One camera, four layers, at real speed. Boxes: each object's tight box comes from its own pixels."),
    # 1
    ("This frame lists 16 objects, each with its class, range, occlusion and truncation. Dashed: its 3D box through "
     "the lens.",
     "This frame lists sixteen objects, each with its class, range, occlusion, and truncation. Dashed: its three D "
     "box through the lens."),
    # 2
    ("Semantic: CARLA's 29 classes, plus one Vantage adds: the ego, in cyan.",
     "Semantic: CARLA's twenty-nine classes, plus one Vantage adds: the ego, in cyan."),
    # 3
    ("Instances: one colour per object, all drive long. This pedestrian is object 56 in all its 193 frames, in four "
     "cameras.",
     "Instances: one colour per object, all drive long. This pedestrian is object fifty-six in all its hundred and "
     "ninety-three frames, in four cameras."),
    # 4
    ("Range, along each pixel's ray: on the same drive without walkers, it differs from the lidar on the ground by a "
     "median of −0.9 mm.",
     "Range, along each pixel's ray: on the same drive without walkers, it differs from the lidar on the ground by a "
     "median of minus zero point nine millimetres."),
]


DRIVE = 'gt_drive_walkers'
TAG = 'MR'
HOLD1 = 2.65         # the frame paused on for its objects (MR frame 27, captured at 2.625 s)
HOLD2 = 5.0          # ... for the pedestrian's instance
PED = 56
HOLD3 = 6.4          # ... for the range


def scene(stage, r):
    import numpy as np
    from PyQt6.QtCore import QRectF
    import gtfootage as G

    v = G.start(stage, r)
    d = G.Drive(DRIVE)
    idx = d.doc
    quad = v.use(G.Quad(d, TAG))
    size = d.size(TAG)

    # what is narrated, from the drive's own files
    k1 = d.index_at(TAG, HOLD1)
    n_obj = len(d.frame(TAG, k1)['objects'])
    n_boxed = len(d.objects(TAG, k1))
    print(f'MR frame {k1 + 1}: {n_obj} objects, {n_boxed} of 40 px or more', flush=True)
    r.expect(n_obj, 16, 'objects listed in the frame')
    carla = [c for c in idx['classes'] if c.get('carla_label')]
    r.expect((len(carla), [c['name'] for c in idx['classes'] if c['id'] == 29]), (29, ['ego']), 'the classes')
    views = {}
    for tag, cam in idx['cameras'].items():
        for f in cam['frames']:
            if any(o['id'] == PED for o in f['objects']):
                views[tag] = views.get(tag, 0) + 1
    print('pedestrian', PED, views, flush=True)
    r.expect((sum(views.values()), len(views)), (193, 4), f'pedestrian {PED}: frames and cameras')
    k2 = d.index_at(TAG, HOLD2)
    ped = next(o for o in d.frame(TAG, k2)['objects'] if o['id'] == PED)
    r.expect(ped['class'], 'pedestrian', 'object 56 is a pedestrian')

    def zone(name, pad=40):
        rr = quad.rect(name)
        return QRectF(rr.x() - 8, rr.y() - 34, rr.width() + 16, rr.height() + 42)

    # ---- 0: four layers at real speed; boxes
    v.clock.play(0.0)
    r.lower_third()
    r.hold(0.3)
    r.say(0)
    G.play_until(r, v, HOLD1)
    r.until_word(0, 'Boxes')
    r.zoom(zone('boxes'), 1.0)
    r.until_word(0, 'its own pixels')
    r.callout(quad.obj_rect('boxes', 51), '', dur=max(1.6, r.speech_left() + 0.2), pad=5, dim=0.3)
    r.finish_speech(0.1)

    # ---- 1: the frame's objects; the full boxes
    r.say(1)
    r.figure('Objects in this frame', n_obj, '', 0, dur=max(3.0, r.at_word(1, 'Dashed') - r.now() + 0.4),
             src=quad.rect('boxes'), note=f'boxed here: the {n_boxed} of 40 px or more', pos='auto')
    r.until_word(1, 'Dashed')
    quad.full = True
    v.update()
    r.hold(max(1.5, r.speech_left() + 0.2))
    r.finish_speech(0.1)
    quad.full = False

    # ---- 2: semantic; the ego
    r.say(2)
    r.zoom(zone('semantic'), 0.9)
    sem = d.raw(TAG, k1, 'semantic')
    ys, xs = np.nonzero(sem == 29)
    ego = G.map_box(quad.rect('semantic'), size, (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
    G.play_until(r, v, HOLD2)
    r.until_word(2, 'the ego')
    r.callout(ego, '', dur=max(1.6, r.speech_left() + 0.4), pad=4, dim=0.3)
    r.finish_speech(0.1)

    # ---- 3: instances; one pedestrian across the drive
    r.say(3)
    r.zoom(zone('instance'), 0.9)
    r.until_word(3, 'This pedestrian')
    r.figure(f'Frames of object {PED}', sum(views.values()), '', 0, dur=max(2.6, r.speech_left() + 0.2),
             src=quad.obj_rect('instance', PED), note=f'in {", ".join(t for t in G.GRID[0] + G.GRID[1] if t in views)}',
             pos='auto')
    r.finish_speech(0.1)

    # ---- 4: range
    r.say(4)
    r.zoom(zone('range'), 0.9)
    G.play_until(r, v, HOLD3)
    r.until_word(4, 'on the same drive')
    rg = quad.rect('range')
    ground = QRectF(rg.x() + rg.width() * 0.08, rg.y() + rg.height() * 0.35, rg.width() * 0.84, rg.height() * 0.4)
    r.figure('Range against the lidar', -0.9, 'mm', 1, dur=max(2.6, r.speech_left() + 0.3), src=ground,
             note='median on the ground · same drive, no walkers', pos='auto')
    r.finish_speech(0.2)
    r.zoom(None, 0.8)
    r.finish_speech(1.0)
    d.release()
