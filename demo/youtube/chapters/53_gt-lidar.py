"""53 Every lidar point labelled (episode gt040, chapter 3): the roof lidar of the drive gt_drive_walkers (in
$VANTAGE_GT_DRIVES, read only) from above, each point in its nuScenes-lidarseg class with the objects' 3D boxes as
footprints, beside the front camera; played at real speed, held while the figures are read.

Figures: the points of the sweep held at 7.5 s (on screen, not narrated: ground_truth/index.json's sweep 'points',
its .pcd.bin's and its lidarseg file's lengths, equal); the semantic lidar matched in order on 99.75 % of the drive's
800 steps, range RMS 1.5 cm (run.json ground_truth.lidar_labels.L1: slices 800, matched_in_order_share 0.9975,
range_rms_diff_m 0.01499), read while recording, checked with r.expect and quoted on screen; every lidar point inside
an annotated vehicle's box labelled vehicle, all 20,513 (simcore report, measured on the same drive without walkers,
gt_drive: "20,513 points in 100 boxes at the 20 keyframes, 100.0 %"), said so on screen.
"""

META = {
    'id': 'gt-lidar',
    'part': 1,
    'title': 'Every lidar point labelled',
    'promise': 'nuScenes-lidarseg and panoptic, from the same rays.',
    'menu': 'nuscenes/lidarseg · nuscenes/panoptic',
}

LINES = [
    # 0
    ("A semantic lidar runs beside the roof lidar on the same rays, so every point of every sweep gets its "
     "nuScenes-lidarseg class and its panoptic instance.",
     "A semantic lidar runs beside the roof lidar, on the same rays, so every point of every sweep gets its new "
     "scenes lidar seg class, and its panoptic instance."),
    # 1
    ("Checked on all of the drive's 800 steps: the two matched in order on 99.75 per cent.",
     "Checked on all of the drive's eight hundred steps: the two matched in order on ninety-nine point seven five "
     "per cent."),
    # 2
    ("On the same drive without walkers, every lidar point inside an annotated vehicle's box is labelled vehicle: "
     "all 20,513.",
     "On the same drive without walkers, every lidar point inside an annotated vehicle's box is labelled vehicle: "
     "all twenty thousand, five hundred and thirteen."),
]


DRIVE = 'gt_drive_walkers'
HOLD = 7.5           # the drive time held for the sweep's figures
HI_CLASSES = ('car', 'truck', 'bus')


def scene(stage, r):
    import numpy as np
    from PyQt6.QtCore import QRectF
    import gtfootage as G
    from vantage.core import groundtruth as GT

    v = G.start(stage, r)
    d = G.Drive(DRIVE)
    ll = d.run['ground_truth']['lidar_labels']['L1']
    print('lidar labels:', ll, flush=True)
    r.expect((ll['slices'], G.fmt(100 * ll['matched_in_order_share'], 2), G.fmt(100 * ll['range_rms_diff_m'], 1)),
             (800, '99.75', '1.5'), 'run.json: the semantic lidar against the recorded one')
    lay = v.use(G.LidarBig(d, 'F', title='Roof lidar, every point labelled · real CARLA 0.9.16 drive'))
    lay.facts = ['run.json · ground_truth.lidar_labels.L1',
                 f'matched in order on {100 * ll["matched_in_order_share"]:.2f} % of {ll["slices"]} steps',
                 f'range RMS {100 * ll["range_rms_diff_m"]:.2f} cm: the lidar\'s own noise']
    sw, P, L = d.sweep(d.time_of('F', d.index_at('F', HOLD)))
    print('sweep', sw['sample_data_token'], sw['points'], len(P), len(L), flush=True)
    r.expect((len(L), len(P)), (sw['points'], sw['points']), 'the sweep\'s points, all labelled')

    # ---- 0: the semantic lidar beside the roof lidar; every point labelled
    v.clock.pause(2.0)
    r.lower_third()
    r.hold(0.3)
    r.say(0)
    v.clock.play(max(0.0, HOLD - (r.at_word(0, 'so every point') - r.now())))     # held there from those words
    bev = lay.rect('lidar')
    r.until_word(0, 'on the same rays')
    r.zoom(QRectF(bev.center().x() - 420, bev.center().y() - 236, 840, 472), 1.2, wait=False)
    r.until_word(0, 'so every point')
    G.play_until(r, v, HOLD)
    r.zoom(None, 0.8)
    r.figure('Points in this sweep', len(L), '', 0, dur=max(3.0, r.speech_left() + 0.6),
             src=lambda: lay.rect('points'), note='each with its lidarseg class and panoptic id', pos='auto')
    r.finish_speech(0.1)

    # ---- 1: checked on every step
    r.say(1)
    r.callout(lambda: lay.rect('fact1'), '', dur=max(2.0, r.speech_left() + 0.3), pad=6, dim=0.3)
    r.until_word(1, 'ninety-nine')
    r.figure('Matched in order', 100 * ll['matched_in_order_share'], '%', 2, dur=max(2.4, r.speech_left() + 0.4),
             src=lambda: lay.rect('fact1'), start=90.0, pos='auto')
    r.finish_speech(0.1)

    # ---- 2: points inside the vehicles' boxes; the footprints
    tt = d.time_of('F', d.index_at('F', HOLD))
    objs = [o for o in d.objects_all(tt, 40, HI_CLASSES) if o.get('box3d')]
    near = sorted(objs, key=lambda o: np.hypot(*o['box3d']['center'][:2]))
    car = near[0]
    inside = int(GT.in_box(P, car['box3d']['center'], car['box3d']['size'],
                           GT.box_matrix(car['box3d'].get('yaw_deg', 0.0))).sum())
    print('nearest vehicle', car['class'], car['id'], car['box3d'], 'points inside', inside, flush=True)
    lay.hi = car['id']
    s_ = bev.width() / (2 * lay.half)
    cxy = car['box3d']['center']
    c = (bev.center().x() - cxy[1] * s_, bev.center().y() - cxy[0] * s_)
    fp = GT.box_footprint(car['box3d'])
    xs_ = [bev.center().x() - y * s_ for x, y in fp]
    ys_ = [bev.center().y() - x * s_ for x, y in fp]
    fp_rect = QRectF(min(xs_) - 4, min(ys_) - 4, max(xs_) - min(xs_) + 8, max(ys_) - min(ys_) + 8)
    r.say(2)
    v.update()
    r.zoom(QRectF(c[0] - 260, c[1] - 150, 520, 300), 1.2)
    r.until_word(2, 'all twenty')
    r.figure('In a vehicle\'s box: vehicle', 100, '%', 0, dur=max(2.6, r.speech_left() + 0.4),
             src=fp_rect, note='20,513 points · same drive, no walkers', start=0.0)
    r.finish_speech(0.2)
    r.zoom(None, 0.8)
    r.finish_speech(1.0)
    d.release()
