"""33 As-built check: Calibrate ▸ As-built check on a recording of the sample's data rig as built. The drive is
demo/clip_asbuilt.make_drive's (not filmed, about 30 s): a 2.5 s synthetic drive written as a ROS 2 bag by the sample
dataset with the left mirror camera ML pitched 2 deg down and rolled 1 deg and the rear camera R turned 1.5 deg in
yaw; the plan keeps the nominal poses. It goes to ~/drives/rig_test_drive in the recording's throw-away home.

Figures read from the page while recording (r.expect), the same as core.asbuilt.run gives in a light run: verdict
'Holds' with corrections to 2 cameras; 6 of 6 cameras checked (4 from their outline, R and QR through the drive-log
estimate against the lidar); largest correction 1.99 deg (ML); ML pitch +1.99 +-0.02 deg, roll +1.09 +-0.10 deg; R
not checkable from its outline, the lidar estimate yaw +1.44 deg (high confidence); blind ground 0.24 m² as planned
(+0.00).
"""

META = {
    'id': 'asbuilt',
    'part': 4,
    'title': 'As-built check',
    'promise': 'Does each real camera see the vehicle where the plan says, and does the coverage still hold?',
    'menu': 'Calibrate ▸ As-built check',
}

LINES = [
    ("Before trusting the plan's coverage figures, settle one question: does each real camera see the vehicle where "
     "the simulation says it does? That is the as-built check.",
     "Before trusting the plan's coverage figures, settle one question: does each real camera see the vehicle where "
     "the simulation says it does? That is the as-built check."),
    ("It reads a short drive recorded by the real rig: a ROS 2 bag, a nuScenes folder, or images per camera. This one "
     "is synthetic: the left mirror camera pitched 2 degrees and rolled 1, the rear camera turned 1.5.",
     "It reads a short drive recorded by the real rig: a ross two bag, a new scenes folder, or images per camera. This "
     "one is synthetic: the left mirror camera pitched two degrees and rolled one, the rear camera turned one and a "
     "half."),
    ("In frames taken while the vehicle moves, what stays the same is the vehicle itself. Road and sky stay the same "
     "too; they are told apart by static edges, and what cannot be decided is left out, not guessed.",
     "In frames taken while the vehicle moves, what stays the same is the vehicle itself. Road and sky stay the same "
     "too; they are told apart by static edges, and what cannot be decided is left out, not guessed."),
    ("The verdict: the plan's coverage holds, with corrections to two cameras. All six were checked, and the largest "
     "correction is 1.99 degrees.",
     "The verdict: the plan's coverage holds, with corrections to two cameras. All six were checked, and the largest "
     "correction is one point nine nine degrees."),
    ("For the left mirror camera: the real outline in blue, and in orange the outline simulated from the plan's pose, "
     "through the lens model. They do not meet.",
     "For the left mirror camera: the real outline in blue, and in orange, the outline simulated from the plan's "
     "pose, through the lens model. They do not meet."),
    ("The fitted pose lays one on the other: pitch +1.99 degrees, give or take 0.02, and roll +1.09. The drive was "
     "made with 2 and 1.",
     "The fitted pose lays one on the other: pitch plus one point nine nine degrees, give or take zero point zero two, "
     "and roll plus one point zero nine. The drive was made with two and one."),
    ("The rear camera's bumper is the road's grey in this scene, so its outline cannot tell. The drive-log estimate "
     "against the lidar stands in: yaw +1.44 degrees, with high confidence.",
     "The rear camera's bumper is the road's grey in this scene, so its outline cannot tell. The drive log estimate "
     "against the lidar stands in: yaw plus one point four four degrees, with high confidence."),
    ("As built: a copy of the variant with every confident correction is analysed against the plan. Blind ground, "
     "the two-camera share and each camera's ground all hold.",
     "As built: a copy of the variant, with every confident correction, is analysed against the plan. Blind ground, "
     "the two camera share, and each camera's ground, all hold."),
    ("Add as-built variant keeps that copy, in one undo step, to calibrate with. Export writes the check as JSON, a CSV "
     "per camera, and an overlay image per camera.",
     "Add as-built variant keeps that copy, in one undo step, to calibrate with. Export writes the check as Jason, a C "
     "S V per camera, and an overlay image per camera."),
]


def scene(stage, r):
    import os
    from PyQt6.QtCore import QRectF
    import clip_asbuilt as CA
    import helpers_tourc as H

    bag = os.path.join(stage.home, 'drives', CA.NAME)
    CA.make_drive(bag)
    w = stage._rig_window()
    r.win = w
    st = w.store
    stage.tilde(w.msg_label)
    vp = stage.vp

    # 0 - open the page
    r.lower_third()
    r.say(0)
    r.cursor_on = False
    stage.orbit(r, vp, 22, 3.6, d_el=4)
    r.cursor_on = True
    r.until_word(0, 'That is the as-built')
    r.menu('Calibrate', 'As-built check')
    pg = w.page('asbuilt')
    stage.tilde_path_label(pg)
    stage.settle(2)
    r.crossfade(0.4)
    r.finish_speech(0.2)

    # 1 - open the recording
    r.say(1)
    r.click(pg.b_dir, real=False, dur=0.7)            # the folder dialog cannot be filmed: open the bag directly
    pg.open_path(bag)
    stage.settle(2)
    r.expect(pg.path_label.text(), '~/drives/rig_test_drive', 'the recording shown')
    pl = r.wrect(pg.path_label)
    r.zoom(QRectF(pl.x() - 20, pl.y() - 60, 520, 300), 0.9)
    r.callout(pg.path_label, dur=2.6, pad=4)
    r.until_word(1, 'This one is synthetic')
    tt = r.wrect(pg.topics)
    r.zoom(QRectF(tt.x() - 10, tt.y() - 30, tt.width() + 20, tt.height() + 40), 0.9)
    r.finish_speech(0.2)
    r.zoom(None, 0.8)

    # 2 - Run: a time-lapse
    r.say(2)
    r.click(pg.b_run, real=False)
    pg.b_run.click()
    r.move(r.wpos(pg.tabs), 0.8)
    r.cursor_on = False
    r.callout(w.state_btn, 'Progress', dur=3.2, pad=4, side='below')      # the job runs in the background
    H.wait_job(r, lambda: pg.result is not None and not pg.busy, timeout=900, fast=15.0)
    stage.settle(1)
    r.zoom(None, 0.6, wait=False)
    print('status:', pg.status.text(), flush=True)
    r.finish_speech(0.2)

    # 3 - the verdict tiles
    T = pg.tiles
    print('tiles:', {k: (t.value, t.unit, t.note) for k, t in T.items()}, flush=True)
    r.expect((T['verdict'].value, T['verdict'].note), ('Holds', 'with corrections to 2 cameras'), 'verdict tile')
    r.expect((T['cameras'].value, T['cameras'].unit), ('6', 'of 6'), 'cameras tile')
    r.expect((T['worst'].value, T['worst'].note.split(' ')[0]), ('1.99', 'ML'), 'largest correction tile')
    r.say(3)
    t0, t4 = r.wrect(T['verdict']), r.wrect(T['blind'])
    r.zoom(QRectF(t0.x() - 10, t0.y() - 20, t4.right() - t0.x() + 20, t0.height() + 40), 1.0)
    r.callout(T['verdict'], dur=2.6, pad=3, dim=0.15)
    r.until_word(3, 'All six were checked')
    r.callout(T['cameras'], dur=1.8, pad=3, dim=0.15)
    r.until_word(3, 'the largest')
    r.callout(T['worst'], dur=max(1.8, r.speech_left() + 0.3), pad=3, dim=0.15)
    r.finish_speech(0.3)
    r.zoom(None, 0.8)

    # 4 - ML: real outline against the plan's simulated one
    r.say(4)
    i = pg.cam_pick.findData('ML')
    if i >= 0:
        pg.cam_pick.setCurrentIndex(i)
    stage.settle(1)
    ir = pg.view.image_rect()
    tl = r.wpos(pg.view, ir.topLeft())
    r.zoom(QRectF(tl.x() + ir.width() * 0.40, tl.y(), ir.width() * 0.60, ir.height()), 1.0, margin=6)
    r.hold(max(1.5, r.speech_left() - 0.2))
    r.finish_speech(0.2)

    # 5 - the fitted pose
    ml = st.variant.by_tag('ML')
    t = pg.cams
    mi = next(k for k in range(t.rowCount()) if t.item(k, 0).text() == 'ML')
    row = [t.item(mi, j).text() if t.item(mi, j) else '' for j in range(t.columnCount())]
    print('ML row:', row, flush=True)
    r.expect((row[7], row[8]), ('+1.99', '+1.09'), "ML's pitch and roll corrections")
    r.say(5)
    r.cursor_on = True
    r.click(pg.mode.buttons['fitted'], dur=0.6)
    r.hold(1.8)
    r.cursor_on = False
    g = r.wrect(t)
    last = t.columnViewportPosition(10) + t.columnWidth(10)
    rows = QRectF(g.x(), g.y(), last + 8, t.horizontalHeader().height() + t.rowCount() * t.rowHeight(0) + 4)
    r.zoom(rows, 1.0, margin=10)
    r.callout(H.cell(r, t, mi), dur=max(2.0, r.speech_left() + 0.2), pad=2, dim=0.2)
    r.finish_speech(0.2)

    # 6 - R: the lidar estimate stands in
    ri = next(k for k in range(t.rowCount()) if t.item(k, 0).text() == 'R')
    rrow = [t.item(ri, j).text() if t.item(ri, j) else '' for j in range(t.columnCount())]
    print('R row:', rrow, flush=True)
    r.expect(rrow[10].startswith('+1.44'), True, "R's lidar estimate")
    r.say(6)
    r.callout(H.cell(r, t, ri), dur=max(2.6, r.speech_left() - 3.0), pad=2, dim=0.2)
    r.until_word(6, 'stands in')
    r.callout(H.cell(r, t, ri, 10), dur=max(2.0, r.speech_left() + 0.3), pad=3, dim=0.2)
    r.finish_speech(0.2)
    r.zoom(None, 0.8)

    # 7 - the coverage as built
    bar = pg.tabs.tabBar()
    r.say(7)
    r.cursor_on = True
    r.click(bar, bar.tabRect(1).center(), dur=0.6)
    stage.settle(1)
    print('coverage line:', pg.cov_line.text(), flush=True)
    cov = r.wrect(pg.cov_line).united(r.wrect(pg.vis))
    r.zoom(cov, 1.0, margin=12)
    r.cursor_on = False
    r.hold(max(1.6, r.speech_left() - 0.2))
    r.finish_speech(0.2)
    r.zoom(None, 0.8)

    # 8 - add the as-built variant; export
    r.say(8)
    r.click(bar, bar.tabRect(0).center(), dur=0.5)
    stage.settle(1)
    r.cursor_on = True
    n0 = len(st.project.variants)
    r.click(pg.b_add, real=False)
    pg.b_add.click()
    stage.settle(3)
    r.expect(len(st.project.variants), n0 + 1, 'the as-built variant added')
    print('variant now:', st.variant.name, flush=True)
    r.callout(w.variant_combo, dur=2.6, pad=4, side='below')
    r.until_word(8, 'Export writes')
    r.callout(pg.b_export, dur=max(1.8, r.speech_left() + 0.3), pad=4)
    r.cursor_on = False
    r.finish_speech(0.8)
