"""41 Critical targets and mounting envelopes (episode v020, chapter 1): Analysis ▸ Critical targets on the sample's
data rig. Six presets added from the Add menu (a child 2 m ahead, a pedestrian 10 m ahead, an overhead traffic light
15 m ahead, a sign 15 m ahead and 3 m left, the kerb by the right front wheel, a lane marking on a 50 m curve); the
pedestrian's cameras and F's mounting envelope (each axis alone, the hover read-out, the grey surface limits, the joint
grid), the camera's tightest envelope over all its targets, ML's limit tighter than the tolerance; the overhead light in
no camera's image, renamed and typed 40 m ahead where F resolves it; the child dragged in the plan to the front corner
where it is lost, renamed; the overlay and the Issues list in the main window; the placement optimiser with every
target required (8 generations of 40, a marked time-lapse) and Add the best three; the export (the mechanical sheet
shown as the export wrote it), a real batch run with --critical-targets, the limits.

Figures read from the app while recording (r.expect), the same as core.critical gives on the sample: 5 of 6 targets
resolved; the pedestrian: F only, 100 % in view and seen, 153 px of 24, 10.0 m, in focus, 146 lidar points per frame,
radars RD1, RD2, RD3; F's envelope on it: pitch +8.25°, x −17.0 mm and z −74.0 mm at the body's surface, joint 62 %,
holds at the tolerance (±1°, ±10 mm); F over all its targets: binding the sign, pitch +3.28°; ML: z +6.1 mm on the
sign; the overhead light at 40 m: F, 22 px of 15, pitch +1.48°; the child at 0.3 m ahead and 0.8 m right: F sees 47 %
(needs 50 %), unresolved; optimiser (seed 7, 8 x 40, every target required): the current layout 5/6, C1 to C3 6/6,
C1 turns ML/MR from 65.0° to 53.6° with a worst case under tolerance of 0.22 m² against 0.24 m²; the batch run exits
with 2.
"""

META = {
    'id': 'new-critical',
    'part': 1,
    'title': 'Critical targets and mounting envelopes',
    'promise': 'What the rig must never miss, and how far each mount may be off.',
    'menu': 'Analysis ▸ Critical targets…',
}

LINES = [
    # 0
    ("A coverage map shows how much ground the cameras see. A data-collection rig is also judged on particular "
     "objects at particular places: a child just in front of the bumper, a traffic light at a junction, the kerb you "
     "park against.",
     "A coverage map shows how much ground the cameras see. A data collection rig is also judged on particular "
     "objects, at particular places: a child just in front of the bumper, a traffic light at a junction, the kerb you "
     "park against."),
    # 1
    ("Analysis ▸ Critical targets checks exactly those. Add places presets relative to the body, with typical sizes "
     "and needs: a child 2 m ahead, a pedestrian 10 m ahead, an overhead traffic light, a sign, the kerb by the right "
     "front wheel and a lane marking on a curve.",
     "Analysis, critical targets, checks exactly those. Add places presets relative to the body, with typical sizes "
     "and needs: a child two metres ahead, a pedestrian ten metres ahead, an overhead traffic light, a sign, the kerb "
     "by the right front wheel, and a lane marking on a curve."),
    # 2
    ("The study runs by itself after every edit, in a few seconds. The table lists each target, the pixels it needs, "
     "and the cameras that resolve it: five of the six here.",
     "The study runs by itself after every edit, in a few seconds. The table lists each target, the pixels it needs, "
     "and the cameras that resolve it: five of the six, here."),
    # 3
    ("The plan draws them in place: green resolved, red not, with a dashed line from every camera that resolves one.",
     "The plan draws them in place: green resolved, red not, with a dashed line from every camera that resolves one."),
    # 4
    ("Pick the pedestrian. For each camera: how much of it is in the image, how much is seen, with the body, the rig "
     "and the housings tested ray by ray, and the pixels across its height through the camera's real lens model.",
     "Pick the pedestrian. For each camera: how much of it is in the image, how much is seen, with the body, the rig, "
     "and the housings tested ray by ray, and the pixels across its height, through the camera's real lens model."),
    # 5
    ("Only F resolves it: 153 pixels across its 1.7 metres, where the detector needs 24, ten metres away and in "
     "focus. The roof lidar puts 146 points a frame on it, and three radars detect it.",
     "Only F resolves it: a hundred and fifty-three pixels across its one point seven metres, where the detector "
     "needs twenty-four, ten metres away, and in focus. The roof lidar puts a hundred and forty-six points a frame on "
     "it, and three radars detect it."),
    # 6
    ("Now the mechanical question: how far may F's mount be off before it loses the pedestrian? Each axis is moved "
     "alone, both ways, until the target is lost, and the limit is found to a hundredth of a degree and a tenth of a "
     "millimetre.",
     "Now the mechanical question: how far may F's mount be off, before it loses the pedestrian? Each axis is moved "
     "alone, both ways, until the target is lost, and the limit is found to a hundredth of a degree, and a tenth of a "
     "millimetre."),
    # 7
    ("Each bar spans five times the install tolerance, which is dashed: ±1° and ±10 mm. Pitched 8.25° further down, F "
     "loses the pedestrian; yaw and roll hold to the end of the 30° search.",
     "Each bar spans five times the install tolerance, which is dashed: plus or minus one degree, and ten "
     "millimetres. Pitched eight point two five degrees further down, F loses the pedestrian. Yaw and roll hold to "
     "the end of the thirty degree search."),
    # 8
    ("Grey is the body's surface, not a loss: moved back 17 mm, or down 74, the lens would be inside the body.",
     "Grey is the body's surface, not a loss: moved back seventeen millimetres, or down seventy-four, the lens would "
     "be inside the body."),
    # 9
    ("Then all six axes at once, on a grid of 729 poses: the pedestrian holds with every axis at 62 per cent of its "
     "own limit, and it holds at the install tolerance.",
     "Then all six axes at once, on a grid of seven hundred and twenty-nine poses: the pedestrian holds with every "
     "axis at sixty-two per cent of its own limit, and it holds at the install tolerance."),
    # 10
    ("For the bracket, All its targets shows the tightest limit over every target the camera resolves, and the "
     "target that sets it. For F, that is the sign: 3.28° of pitch.",
     "For the bracket, all its targets shows the tightest limit over every target the camera resolves, and the "
     "target that sets it. For F, that is the sign: three point two eight degrees of pitch."),
    # 11
    ("ML is tighter than the tolerance: raised by 6.1 mm, it no longer sees enough of the sign. F resolves the sign "
     "too, so this stays a finding on the page; a camera that alone resolves a target raises a warning in the Issues "
     "list.",
     "M L is tighter than the tolerance: raised by six point one millimetres, it no longer sees enough of the sign. "
     "F resolves the sign too, so this stays a finding on the page. A camera that alone resolves a target, raises a "
     "warning in the issues list."),
    # 12
    ("The overhead traffic light is in no camera's image: a serious finding. The fisheyes look down at the road, and "
     "the signal head hangs 5.5 metres up, 15 metres ahead.",
     "The overhead traffic light is in no camera's image: a serious finding. The fisheyes look down at the road, and "
     "the signal head hangs five and a half metres up, fifteen metres ahead."),
    # 13
    ("Rename it, and type where it stands: 40 metres ahead, it comes into F's image, 22 pixels tall for the 15 it "
     "needs. This rig sees the signal from 40 metres back, and loses it as it draws up to the stop line.",
     "Rename it, and type where it stands: forty metres ahead, it comes into F's image, twenty-two pixels tall, for "
     "the fifteen it needs. This rig sees the signal from forty metres back, and loses it as it draws up to the stop "
     "line."),
    # 14
    ("And only just: pitched 1.48° further down, F loses it. The light now binds F's bracket, with half a degree to "
     "spare over the one-degree tolerance.",
     "And only just: pitched one point four eight degrees further down, F loses it. The light now binds F's bracket, "
     "with half a degree to spare over the one degree tolerance."),
    # 15
    ("Targets can be dragged in the plan. Take the child to the front corner, 0.3 metres ahead and 0.8 to the right: "
     "F sees only 47 per cent of it, where half is needed, MR even less, and the child is lost.",
     "Targets can be dragged in the plan. Take the child to the front corner, zero point three metres ahead, and zero "
     "point eight to the right: F sees only forty-seven per cent of it, where half is needed, M R even less, and the "
     "child is lost."),
    # 16
    ("Every figure of a target can be typed: its name, kind, place, size and height above the ground, the pixels it "
     "needs, the share that must be seen, lidar points it also needs, and its radar cross-section. Every edit can be "
     "undone.",
     "Every figure of a target can be typed: its name, kind, place, size, and height above the ground, the pixels it "
     "needs, the share that must be seen, lidar points it also needs, and its radar cross section. Every edit can be "
     "undone."),
    # 17
    ("The page's view turns to 3D: each target stands at its size, with a dashed sight line from every camera that "
     "resolves it.",
     "The page's view turns to 3D: each target stands at its size, with a dashed sight line from every camera that "
     "resolves it."),
    # 18
    ("In the main window, View ▸ Overlays ▸ Critical targets draws them by verdict, and the Issues list carries the "
     "lost child as a serious issue.",
     "In the main window, view, overlays, critical targets, draws them by verdict, and the issues list carries the "
     "lost child as a serious issue."),
    # 19
    ("The placement optimiser takes them as a constraint. Set Critical targets to Every target resolved: a layout "
     "that loses one never reaches the Pareto front. Eight generations of forty layouts.",
     "The placement optimiser takes them as a constraint. Set critical targets to, every target resolved: a layout "
     "that loses one never reaches the Pareto front. Eight generations of forty layouts."),
    # 20
    ("The current layout resolves five of the six. Every candidate on the front resolves all six: the first turns ML "
     "and MR eleven degrees further forward, onto the corner, and leaves less ground blind under tolerance.",
     "The current layout resolves five of the six. Every candidate on the front resolves all six: the first turns M L "
     "and M R eleven degrees further forward, onto the corner, and leaves less ground blind under tolerance."),
    # 21
    ("Add the best three makes them variants in one step, to compare side by side.",
     "Add the best three makes them variants, in one step, to compare side by side."),
    # 22
    ("Export writes every target and envelope as JSON and CSV, the plan, and the mechanical sheet: each camera's "
     "limits per axis, the target that sets them, and the verdict against the tolerance, for the team designing the "
     "brackets.",
     "Export writes every target and envelope as Jason and C S V, the plan, and the mechanical sheet: each camera's "
     "limits per axis, the target that sets them, and the verdict against the tolerance, for the team designing the "
     "brackets."),
    # 23
    ("In batch mode, --critical-targets writes the same files and prints a line per target. The child at the corner "
     "is not resolved, so it exits with 2: a CI job fails until the rig resolves every target. The assistant has a "
     "tool that runs the study too.",
     "In batch mode, critical targets writes the same files, and prints a line per target. The child at the corner "
     "is not resolved, so it exits with two: a C I job fails until the rig resolves every target. The assistant has "
     "a tool that runs the study, too."),
    # 24
    ("The limits, as the page states them: targets are flat and square to each camera, on level ground; blur, glare, "
     "weather and body motion are left out; and the sizes and pixel needs are typical. Set your detector's.",
     "The limits, as the page states them: targets are flat, and square to each camera, on level ground; blur, glare, "
     "weather, and body motion are left out; and the sizes and pixel needs are typical. Set your detector's."),
    # 25 (said while the optimiser runs, between 19 and 20)
    ("Each layout is checked against all six targets at its own pose, and again in each of its 24 tolerance "
     "samples.",
     "Each layout is checked against all six targets at its own pose, and again in each of its twenty-four "
     "tolerance samples."),
]


TARGETS = ('Child 2 m ahead', 'Pedestrian 10 m ahead', 'Overhead traffic light 15 m ahead',
           'Sign 15 m ahead, 3 m left', 'Kerb by the right front wheel', 'Lane marking on a 50 m curve')


def scene(stage, r):
    import os
    import site
    import sys

    import numpy as np
    from PyQt6.QtCore import QPointF, QRectF
    from PyQt6.QtGui import QImage
    from PyQt6.QtWidgets import QScrollArea

    import clips
    import helpers_tourc as H
    import studio
    from clip_critical import pick_from_menu, select_row
    from helpers_tour2a import light_wait_for
    from vantage.core import critical as CR
    from vantage.ui.workspaces import critical_ws as CWS

    # The page re-runs the study by itself after an edit while its last run took under 8 s (about 3.5 s here for six
    # targets). The final recording composes 2x frames on the same machine and can slow a run past that, which would
    # stop the re-runs on camera; the recording raises the limit, and waits for the study with light_wait_for, which
    # leaves the computer to the study, so the run time the page shows is the study's own.
    CWS.AUTO_MAX_S = 60.0

    w = stage._rig_window()
    r.win = w
    app = stage.app
    st = w.store
    p = st.project

    def pump(n=4):
        for _ in range(n):
            app.processEvents()

    orig_msg = w.ctx.msg                       # status-bar messages show paths in the throw-away home from ~
    w.ctx.msg = lambda text, *a, **k: orig_msg(clips._from_home(text), *a, **k)

    # ---- 0: the problem, over the data rig
    vp = stage.vp
    vp.cam.az, vp.cam.el = -30.0, 20.0
    vp.update()
    stage.settle(2)
    r.cursor_on = False
    r.cur = np.array([1180.0, 640.0])
    r.lower_third()
    r.say(0)
    stage.orbit(r, vp, 28, max(4.0, r.line(0) - 0.8), d_el=4)
    r.finish_speech(0.2)

    # ---- 1: Analysis ▸ Critical targets; the presets
    r.say(1)
    r.cursor_on = True
    r.menu('Analysis', 'Critical targets')
    pg = w.page('critical')
    ctl = pg.ctl
    r.crossfade(0.4)
    stage.settle(1)
    r.hold(0.3)
    r.until_word(1, 'Add places')
    for text in TARGETS:
        pick_from_menu(r, pg.b_add, pg.add_menu, text)
        r.hold(0.25)

    def done(n=len(TARGETS)):
        res = ctl.result
        return (res is not None and not ctl.busy and not ctl.stale and len(res.targets) == n
                and not ctl.timer.isActive())

    r.say(2)                                   # the study runs by itself while the line starts
    light_wait_for(r, done, timeout=180)
    pump(6)

    # ---- 2: the table and the status
    names = [t['name'] for t in ctl.settings.targets]
    r.expect(names, list(TARGETS), 'the targets added')
    print('status:', pg.status.text(), flush=True)
    r.expect(pg.status.text().startswith('5 of 6 targets resolved'), True, 'targets resolved')
    secs = float(pg.status.text().split('·')[1].split()[0])
    r.expect(secs < 8.0, True, f'a run of a few seconds (shown: {secs} s)')
    stat = [pg.targets.item(i, 3).text() for i in range(pg.targets.rowCount())]
    print('status column:', stat, flush=True)
    r.expect(stat, ['F', 'F', 'unresolved', 'F, ML', 'MR', 'F, ML'], 'the status column')
    tr = r.wrect(pg.targets)
    r.zoom(QRectF(tr.x() - 6, tr.y() - 70, tr.width() + 12, tr.height() + 90), 1.0)
    r.cursor_on = False
    r.until_word(2, 'the pixels it needs')
    r.callout(lambda: H.union(H.cell(r, pg.targets, 0, 2)(), H.cell(r, pg.targets, 5, 2)()), '', dur=1.8, pad=2,
              dim=0.25)
    r.until_word(2, 'the cameras that resolve')
    r.callout(lambda: H.union(H.cell(r, pg.targets, 0, 3)(), H.cell(r, pg.targets, 5, 3)()), '',
              dur=max(1.8, r.speech_left() - 1.2), pad=2, dim=0.25)
    r.until_word(2, 'five of the six')
    r.callout(pg.status, '', dur=max(1.4, r.speech_left() + 0.3), pad=4, dim=0.25)
    r.finish_speech(0.2)

    # ---- 3: the plan
    r.say(3)
    r.cursor_on = True
    r.click(pg.b_fit, None, 0.6)
    pump()
    r.cursor_on = False
    plr = r.wrect(pg.plan)
    r.zoom(QRectF(plr.x(), plr.center().y() - plr.width() * 0.3, plr.width(), plr.width() * 0.6), 1.0, margin=4)
    r.hold(max(1.5, r.speech_left() + 0.2))
    r.finish_speech(0.2)
    r.zoom(None, 0.8)

    # ---- 4: the pedestrian's cameras
    r.say(4)
    r.cursor_on = True
    select_row(r, pg.targets, 1)
    pump(4)
    res = ctl.result
    ped = res.target(ctl.settings.targets[1]['uid'])
    lk = r.wrect(pg.looks)
    r.zoom(QRectF(lk.x() - 6, lk.y() - 30, lk.width() + 12, lk.height() + 56), 1.0)
    r.cursor_on = False
    hh = pg.looks.horizontalHeader()

    def col(c):
        def f():
            x = hh.sectionViewportPosition(c)
            tl = r.wpos(hh, QPointF(x, 0))
            return QRectF(tl.x(), tl.y(), hh.sectionSize(c), hh.height() + pg.looks.rowHeight(0) * pg.looks.rowCount())
        return f
    r.until_word(4, 'in the image')
    r.callout(col(1), '', dur=1.6, pad=2, dim=0.25)
    r.until_word(4, 'how much is seen')
    r.callout(col(2), '', dur=2.4, pad=2, dim=0.25)
    r.until_word(4, 'the pixels across')
    r.callout(col(3), '', dur=max(1.6, r.speech_left()), pad=2, dim=0.25)
    r.finish_speech(0.2)

    # ---- 5: F alone; lidar and radar
    rowF = next(i for i in range(pg.looks.rowCount()) if pg.looks.item(i, 0).text() == 'F')
    rowtxt = [pg.looks.item(rowF, c).text() for c in range(7)]
    print('F on the pedestrian:', rowtxt, '|', pg.sensors_lbl.text(), flush=True)
    r.expect(rowtxt, ['F', '100', '100', '153', '10.0', 'yes', 'meets'], 'F on the pedestrian')
    r.expect(ped.met_by, ['F'], 'only F resolves the pedestrian')
    r.expect(pg.sensors_lbl.text(), 'Lidar points per frame: 146.3 (L1 146), not required · Radars detecting it '
                                    '(RCS 0.5 m²): RD1, RD2, RD3', 'lidar and radar on the pedestrian')
    r.say(5)
    r.figure('Pixels across it', 153, 'px', 0, dur=max(3.0, r.at_word(5, 'The roof lidar') - r.now()),
             src=H.cell(r, pg.looks, rowF, 3), pos='bl', note='needs 24 px')
    r.until_word(5, 'The roof lidar')
    r.zoom(QRectF(lk.x() - 6, lk.y() - 30, lk.width() + 12, r.wrect(pg.sensors_lbl).bottom() - lk.y() + 50), 0.7)
    r.callout(pg.sensors_lbl, '', dur=max(2.0, r.speech_left() + 0.2), pad=4, dim=0.25)
    r.finish_speech(0.2)

    # ---- 6: the envelope chart
    rows, title, note = pg.envelope_rows()
    print('envelope:', title, note, [(a, lo[:3:2], hi[:3:2]) for a, lo, hi in rows], flush=True)
    r.expect(title, 'F · Pedestrian 10 m ahead', 'the envelope shown')
    lim = {a: (lo, hi) for a, lo, hi in rows}
    r.expect(f'{lim["pitch"][1][0]:.2f} {lim["pitch"][1][2]}', '8.25 seen', 'F pitch limit on the pedestrian')
    r.expect(f'{lim["x"][0][0] * 1000:.1f} {lim["x"][0][2]} {lim["z"][0][0] * 1000:.1f} {lim["z"][0][2]}',
             '17.0 surface 74.0 surface', 'the surface limits')
    r.expect(all(lim[a][0][2] == 'open' and lim[a][1][2] == 'open' and abs(lim[a][1][0] - 30) < 1e-9
                 for a in ('yaw', 'roll')), True, 'yaw and roll open to the search')
    r.expect(note, 'joint 62 % · holds at the tolerance', 'the joint check')
    r.expect((res.install['deg'], res.install['mm']), (1.0, 10.0), 'the install tolerance')
    r.say(6)
    br = r.wrect(pg.bars)
    env_zone = QRectF(br.x() - 6, br.y() - 44, br.width() + 12, br.height() + 70)
    r.zoom(env_zone, 1.1)
    r.hold(max(1.5, r.speech_left() + 0.2))
    r.finish_speech(0.2)

    # ---- 7: reading the bars; the hover read-out
    r.say(7)
    r.until_word(7, 'which is dashed')

    def tol_marks():
        x0, x1 = pg.bars.LEFT, max(pg.bars.LEFT + 80, pg.bars.width() - pg.bars.RIGHT)
        cx, half = (x0 + x1) / 2, (x1 - x0) / 2
        a = r.wpos(pg.bars, QPointF(cx - half / 5 - 6, pg.bars.TOP))
        return QRectF(a.x(), a.y(), 2 * half / 5 + 12, 6 * pg.bars.ROW)
    r.callout(tol_marks, '', dur=2.4, pad=2, dim=0.2)

    def bar_rows(top, bottom=None):
        """The chart's rows from hit rectangle `top` to `bottom` (inclusive), full width, in window coordinates."""
        bottom = bottom or top
        return QRectF(r.wpos(pg.bars, QPointF(0, top.top())), r.wpos(pg.bars, QPointF(pg.bars.width(),
                                                                                      bottom.bottom()))).normalized()
    r.until_word(7, 'Pitched')
    pg.bars.update()
    pump()
    hit = next(hr for hr, h in pg.bars._hits if h[0] == 'pitch' and h[1] == 1)
    r.cursor_on = True
    r.hover(pg.bars, QPointF(hit.center().x() + hit.width() * 0.3, hit.center().y()), 0.8)
    r.hold(max(1.6, r.at_word(7, 'Yaw and roll') - r.now()))
    r.leave(pg.bars)
    r.cursor_on = False
    hy = next(hr for hr, h in pg.bars._hits if h[0] == 'yaw' and h[1] == 1)
    hro = next(hr for hr, h in pg.bars._hits if h[0] == 'roll' and h[1] == 1)
    r.callout(lambda: bar_rows(hy, hro), '', dur=max(1.6, r.speech_left() + 0.2), pad=2, dim=0.2)
    r.finish_speech(0.2)

    # ---- 8: grey = the surface
    r.say(8)
    hx = next(hr for hr, h in pg.bars._hits if h[0] == 'x' and h[1] == -1)
    hz = next(hr for hr, h in pg.bars._hits if h[0] == 'z' and h[1] == -1)
    r.callout(lambda: bar_rows(hx), '', dur=r.at_word(8, 'or down') - r.now() + 0.3, pad=2, dim=0.2)
    r.until_word(8, 'or down')
    r.callout(lambda: bar_rows(hz), '', dur=max(1.6, r.speech_left() + 0.2), pad=2, dim=0.2)
    r.finish_speech(0.2)

    # ---- 9: the joint grid
    r.say(9)

    def title_rect():
        tl = r.wpos(pg.bars, QPointF(0, 0))
        return QRectF(tl.x(), tl.y(), pg.bars.width(), pg.bars.TOP)
    r.until_word(9, 'sixty-two')
    r.callout(title_rect, '', dur=max(2.2, r.speech_left() + 0.2), pad=2, dim=0.25)
    r.finish_speech(0.2)

    # ---- 10: the camera over all its targets
    r.say(10)
    r.cursor_on = True
    r.click(pg.scope_ctl.buttons['all'], None, 0.6)
    pump(4)
    rows, title, note = pg.envelope_rows()
    lim = {a: (lo, hi) for a, lo, hi in rows}
    print('F all:', title, note, flush=True)
    r.expect(title, 'F · every target it resolves', 'F over all its targets')
    r.expect(f'{lim["pitch"][1][0]:.2f} {lim["pitch"][1][1]}', '3.28 Sign 15 m ahead, 3 m left', 'F binding target')
    r.expect(note.startswith('binding: Sign 15 m ahead, 3 m left'), True, 'the binding note')
    r.cursor_on = False
    hp = next(hr for hr, h in pg.bars._hits if h[0] == 'pitch' and h[1] == 1)
    r.until_word(10, 'For F')
    r.callout(lambda: bar_rows(hp), '', dur=max(2.0, r.speech_left() + 0.2), pad=2, dim=0.2)
    r.finish_speech(0.2)

    # ---- 11: ML, tighter than the tolerance
    r.say(11)
    r.cursor_on = True
    i_ml = pg.cam_pick.findData(p.variant.by_tag('ML').uid)
    r.choose(pg.cam_pick, i_ml)
    pump(4)
    rows, title, note = pg.envelope_rows()
    lim = {a: (lo, hi) for a, lo, hi in rows}
    print('ML all:', title, note, pg.env_note.text(), flush=True)
    r.expect(f'{lim["z"][1][0] * 1000:.1f} {lim["z"][1][1]} {lim["z"][1][2]}', '6.1 Sign 15 m ahead, 3 m left seen',
             'ML z limit')
    r.expect(pg.env_note.text(), 'ML: 1 limit tighter than the install tolerance.', 'ML note')
    r.cursor_on = False
    hzz = next(hr for hr, h in pg.bars._hits if h[0] == 'z' and h[1] == 1)
    r.callout(lambda: bar_rows(hzz), '', dur=r.at_word(11, 'F resolves the sign') - r.now(), pad=2, dim=0.2)
    r.until_word(11, 'F resolves the sign')
    fh = r.wrect(pg.find_host.parentWidget())
    r.zoom(QRectF(fh.x() - 6, fh.y() - 34, fh.width() + 12, 200), 0.8)
    tight = next(x for x in pg.findings if 'tighter than the install tolerance' in x.text.text())
    r.callout(tight, '', dur=max(2.0, r.speech_left() + 0.2), pad=4, dim=0.25)
    r.finish_speech(0.2)
    r.zoom(None, 0.8)

    # ---- 12: the overhead traffic light
    r.say(12)
    r.cursor_on = True
    select_row(r, pg.targets, 2)
    pump(4)
    tl_res = ctl.result.target(ctl.settings.targets[2]['uid'])
    r.expect((tl_res.met, max(lk_.in_view for lk_ in tl_res.looks)), (False, 0.0), 'the light in no image')
    tlt = ctl.settings.targets[2]
    r.expect(f"{tlt['base']:.1f} {tlt['x'] - CR.vehicle_box(p)['front_x']:.1f}", '5.5 15.0',
             'the light 5.5 m up, 15 m ahead')
    ser = next(x for x in pg.findings if 'Overhead traffic light' in x.text.text())
    r.expect(ser.text.text(), 'No camera resolves the critical target "Overhead traffic light 15 m ahead" (15.0 m '
                              'ahead of the front bumper): it is in no camera\'s image.', 'the serious finding')
    r.cursor_on = False
    r.zoom(QRectF(fh.x() - 6, fh.y() - 34, fh.width() + 12, 200), 0.8)
    r.callout(ser, '', dur=max(2.4, r.speech_left() + 0.2), pad=4, dim=0.25)
    r.finish_speech(0.2)
    r.zoom(None, 0.7)

    # ---- 13: renamed, typed 40 m ahead
    r.say(13)
    r.cursor_on = True
    eb = r.wrect(pg.edit_box)
    r.zoom(QRectF(eb.x() - 6, eb.y() - 8, eb.width() + 12, eb.height() + 16), 0.8, wait=False)
    r.enter(pg.name, 'Traffic light 40 m ahead', cps=14)
    pump()
    fx = pg.fields['x'][0]
    r.until_word(13, 'type where it stands')
    r.enter(fx, '43.2')
    pump()
    if abs(ctl.settings.targets[2]['x'] - 43.2) > 1e-6:      # the field did not commit on Enter offscreen
        fx.committed.emit(43.2)
        pump()
    r.expect(ctl.settings.targets[2]['name'], 'Traffic light 40 m ahead', 'renamed')
    r.expect(pg.place_lbl.text(), '40.0 m ahead of the front bumper.', 'where it stands now')
    light_wait_for(r, lambda: done() and ctl.result.target(ctl.settings.targets[2]['uid']).met, timeout=180)
    pump(4)
    r.zoom(QRectF(lk.x() - 6, lk.y() - 30, lk.width() + 12, lk.height() + 56), 0.8)
    rowF = next(i for i in range(pg.looks.rowCount()) if pg.looks.item(i, 0).text() == 'F')
    pxF = pg.looks.item(rowF, 3).text()
    print('F on the light at 40 m:', [pg.looks.item(rowF, c).text() for c in range(7)], flush=True)
    r.expect((pxF, pg.looks.item(rowF, 6).text()), ('22', 'meets'), 'F on the light at 40 m')
    r.expect(ctl.settings.targets[2]['min_px'], 15.0, 'the light needs 15 px')
    r.figure('F, pixels on the light', 22, 'px', 0, dur=max(2.6, r.speech_left() - 0.5), src=H.cell(r, pg.looks, rowF, 3),
             note='needs 15 px · 40 m ahead')
    r.finish_speech(0.2)

    # ---- 14: only just
    r.say(14)
    r.zoom(env_zone, 0.8, wait=False)
    r.cursor_on = True
    r.choose(pg.cam_pick, pg.cam_pick.findData(p.variant.by_tag('F').uid))
    r.click(pg.scope_ctl.buttons['target'], None, 0.5)
    pump(4)
    r.cursor_on = False
    rows, title, note = pg.envelope_rows()
    lim = {a: (lo, hi) for a, lo, hi in rows}
    print('F on the light:', title, note, flush=True)
    r.expect(title, 'F · Traffic light 40 m ahead', 'F envelope on the light')
    r.expect(f'{lim["pitch"][1][0]:.2f}', '1.48', 'F pitch limit on the light')
    r.zoom(env_zone, 0.8)
    hp = next(hr for hr, h in pg.bars._hits if h[0] == 'pitch' and h[1] == 1)
    r.callout(lambda: bar_rows(hp), '', dur=max(2.0, r.at_word(14, 'The light now') - r.now()), pad=2, dim=0.2)
    r.until_word(14, 'The light now')
    r.cursor_on = True
    r.click(pg.scope_ctl.buttons['all'], None, 0.6)
    pump(4)
    rows, title, note = pg.envelope_rows()
    print('F all now:', title, note, flush=True)
    r.expect(note.startswith('binding: Traffic light 40 m ahead'), True, 'the light binds F')
    r.cursor_on = False
    r.callout(title_rect, '', dur=max(1.8, r.speech_left() + 0.2), pad=2, dim=0.25)
    r.finish_speech(0.2)
    r.zoom(None, 0.8)

    # ---- 15: the child dragged to the front corner
    r.say(15)
    b = CR.vehicle_box(p)
    pg.plan.fit((b['front_x'] - 4.6, -3.2, b['front_x'] + 3.4, 3.2))      # the view: around the front of the car
    pump(2)
    r.crossfade(0.4)
    r.tick(2)
    r.zoom(pg.plan, 0.8, margin=4)
    child = ctl.settings.targets[0]
    a = pg.plan.w2s(child['x'], child['y'])
    dest = (round(b['front_x'] + 0.3, 3), -0.8)
    bb = pg.plan.w2s(*dest)
    r.cursor_on = True
    r.until_word(15, 'Take the child')
    H.drag(r, pg.plan, QPointF(a), QPointF(bb), dur=2.0)
    pump()
    child = ctl.settings.targets[0]
    print('child now at', child['x'], child['y'], flush=True)
    r.expect((round(child['x'] - b['front_x'], 3), child['y']), (0.3, -0.8), 'the child at the corner')
    light_wait_for(r, lambda: done() and not ctl.result.target(child['uid']).met, timeout=180)
    pump(4)
    r.cursor_on = False
    cres = ctl.result.target(child['uid'])
    seen = {lk_.tag: f'{100 * lk_.seen:.0f}' for lk_ in cres.looks if lk_.in_view > 0}
    print('child at the corner:', seen, cres.met_by, flush=True)
    r.expect(seen['F'], '47', 'F sees 47 % of the child')
    r.expect(float(seen['MR']) < 47, True, 'MR sees less')
    r.expect(ctl.settings.targets[0]['min_share'], 0.5, 'half must be seen')
    r.until_word(15, 'F sees only')
    r.zoom(QRectF(lk.x() - 6, lk.y() - 30, lk.width() + 12, lk.height() + 56), 0.8)
    rowF = next(i for i in range(pg.looks.rowCount()) if pg.looks.item(i, 0).text() == 'F')
    r.callout(H.cell(r, pg.looks, rowF), '', dur=max(2.0, r.speech_left() - 0.6), pad=2, dim=0.25)
    r.finish_speech(0.2)
    r.zoom(None, 0.7)

    # ---- 16: every figure editable; the child renamed
    r.say(16)
    r.zoom(QRectF(eb.x() - 6, eb.y() - 8, eb.width() + 12, eb.height() + 16), 0.8)
    r.cursor_on = True
    r.enter(pg.name, 'Child at the front corner', cps=14)
    pump()
    r.expect(ctl.settings.targets[0]['name'], 'Child at the front corner', 'the child renamed')
    r.cursor_on = False
    for word, keys in (('kind', ('kind',)), ('place', ('x', 'y')), ('size', ('height', 'width', 'base')),
                       ('the pixels it', ('min_px',)), ('the share', ('min_share',)), ('lidar points', ('min_points',)),
                       ('radar cross', ('rcs',))):
        r.until_word(16, word)
        tgt = [pg.kind] if keys == ('kind',) else [pg.fields[k][0] for k in keys] + [pg.field_labels[k] for k in keys]
        r.callout(lambda tgt=tgt: H.union(*[r.wrect(x) for x in tgt]), '', dur=1.3, pad=3, dim=0.2)
    r.finish_speech(0.2)
    r.zoom(None, 0.8)

    # ---- 17: the page in 3D
    r.say(17)
    r.cursor_on = True
    r.click(pg.view_mode.buttons['3d'], None, 0.6)
    pump(4)
    stage.settle(1)
    v3 = pg.view3d
    v3.cam.az, v3.cam.el = 205.0, 26.0
    pg._fit()
    v3.cam.target = np.array([b['front_x'] + 1.0, 0.0, 0.5])     # the near targets, not the light 40 m out
    v3.cam.dist = 14.0
    v3.update()
    pump(2)
    r.crossfade(0.4)
    r.cursor_on = False
    r.zoom(pg.stack, 0.8, margin=4)
    stage.orbit(r, v3, -20, max(3.0, r.speech_left() + 0.2), d_el=4)
    r.finish_speech(0.2)
    r.zoom(None, 0.6)

    # ---- 18: the main window: overlay and the Issues list
    r.say(18)
    r.cursor_on = True
    r.menu('View', 'Back to viewports')
    stage.settle(2)
    r.crossfade(0.4)
    r.menu('View', 'Overlays', 'Critical targets')
    pump(4)
    r.expect(bool(w.ctx.overlays.get('critical')), True, 'the overlay is on')
    stage.settle(3)
    iss = w.bottom.issues
    tabs = w.bottom.tabs
    r.click(tabs, QPointF(tabs.tabRect(w.bottom.keys.index('issues')).center()), 0.6, real=False)
    w.bottom.show_tab('issues')
    pump(4)
    iss.refresh()
    rows_ = [iss.t.item(i, 1).text() for i in range(iss.t.rowCount())]
    print('issues:', rows_, flush=True)
    k_child = next(i for i, t in enumerate(rows_) if t.startswith('No camera resolves the critical target "Child at '
                                                                  'the front corner"'))
    r.cursor_on = False
    vp = stage.vp
    stage.fly(r, vp, az=-20.0, el=34.0, dist=vp.cam.dist * 0.62, target=(3.6, -0.3, 0.3), dur=1.6)
    r.until_word(18, 'and the issues list')
    bbx = r.wrect(w.docks['bottom'])

    def irow():
        a_ = iss.t.visualItemRect(iss.t.item(k_child, 0))
        c_ = iss.t.visualItemRect(iss.t.item(k_child, 2))
        tl_ = r.wpos(iss.t.viewport(), QPointF(a_.topLeft()))
        return QRectF(tl_.x(), tl_.y(), c_.right() - a_.left(), a_.height())
    r.zoom(QRectF(bbx.x(), bbx.y() - 10, min(bbx.width() * 0.62, 1180), bbx.height() + 10), 0.8)
    r.callout(irow, '', dur=max(2.0, r.speech_left() + 0.3), pad=2, dim=0.25)
    r.finish_speech(0.2)
    r.zoom(None, 0.7)

    # ---- 19: the optimiser, every target required
    r.say(19)
    r.cursor_on = True
    r.menu('Layout', 'Optimise placement')
    op = w.page('optimiser')
    stage.settle(2)
    r.crossfade(0.4)
    pump(4)
    cz = r.wrect(op.crit).united(r.wrect(op.crit_note))
    r.zoom(QRectF(cz.x() - 30, cz.y() - 60, max(cz.width() + 60, 420), cz.height() + 90), 0.8)
    r.until_word(19, 'Set critical targets')
    r.choose(op.crit, 1)
    pump()
    r.expect(op.crit.currentData(), 'require', 'every target required')
    print('crit note:', op.crit_note.text(), flush=True)
    r.callout(op.crit_note, '', dur=1.8, pad=3, dim=0.2)
    r.until_word(19, 'Eight generations')
    r.zoom(QRectF(r.wrect(op.gens).x() - 40, r.wrect(op.gens).y() - 120, 420, 240), 0.6)
    r.enter(op.gens, '8')
    r.enter(op.pop, '40')
    if int(op.gens.value()) != 8:
        op.gens.set_value(8)
    if int(op.pop.value()) != 40:
        op.pop.set_value(40)
    r.zoom(None, 0.6, wait=False)
    r.say(25)
    r.click(op.b_run, None, 0.7)
    r.hold(0.3)
    sc = r.wrect(op.scatter)
    r.zoom(QRectF(sc.x(), sc.y() - 50, sc.width(), sc.height() + 50), 0.8, wait=False)
    r.wait_for(lambda: 'optimiser' not in w._jobs, timeout=1500, speed=10)
    stage.settle(2)
    pump(6)
    r.finish_speech(0.2)

    # ---- 20: the candidates
    o = op.opt
    print('summary:', op.summary.text(), flush=True)
    rowsC = [[op.table.item(i, c).text() for c in range(8)] for i in range(op.table.rowCount())]
    for rr in rowsC:
        print('  ', rr, flush=True)
    r.expect(o.targets_text(o.nominal_cand), '5/6', 'the current layout')
    r.expect([x[0] for x in rowsC][:3], ['C1', 'C2', 'C3'], 'three candidates')
    r.expect(all(x[6].strip() == '6/6' for x in rowsC), True, 'every candidate resolves all six')
    r.expect('ML/MR ψ 65.0→53.6' in rowsC[0][7], True, 'C1 turns ML/MR forward')
    r.expect(f'{o.nominal_cand.worst:.2f}/{op.cands[0].worst:.2f}', '0.24/0.22', 'worst under tolerance')
    r.say(20)
    tb = r.wrect(op.table)
    r.zoom(QRectF(tb.x(), tb.y() - 34, tb.width(), 34 + 5 * 24), 1.0)
    hht = op.table.horizontalHeader()

    def tcol(c):
        def f():
            x = hht.sectionViewportPosition(c)
            tl_ = r.wpos(hht, QPointF(x, 0))
            return QRectF(tl_.x(), tl_.y(), hht.sectionSize(c), hht.height() + op.table.rowHeight(0) * min(
                3, op.table.rowCount()))
        return f
    r.until_word(20, 'Every candidate')
    r.callout(tcol(6), '', dur=2.2, pad=2, dim=0.25)
    r.until_word(20, 'the first turns')
    r.callout(H.cell(r, op.table, 0), '', dur=max(2.2, r.at_word(20, 'and leaves less') - r.now()), pad=2, dim=0.25)
    r.until_word(20, 'and leaves less')
    r.callout(tcol(3), '', dur=max(1.8, r.speech_left() + 0.2), pad=2, dim=0.25)
    r.finish_speech(0.2)
    r.zoom(None, 0.8)

    # ---- 21: add the best three
    r.say(21)
    n0 = len(p.variants)
    r.cursor_on = True
    r.click(op.b_three, None, 0.7)
    pump(4)
    r.expect(len(st.project.variants), n0 + 3, 'three variants added')
    r.cursor_on = False
    r.callout(w.variant_combo, '', dur=max(2.0, r.speech_left() + 0.3), pad=4, dim=0.25)
    r.finish_speech(0.2)

    # ---- 22: export; the mechanical sheet
    r.say(22)
    r.cursor_on = True
    r.menu('Analysis', 'Critical targets')
    stage.settle(1)
    r.crossfade(0.4)
    pg.view_mode.buttons['plan'].click()
    pump(2)
    folder = os.path.join(stage.home, 'critical')
    r.click(pg.b_export, None, 0.7, real=False)
    paths = pg.export(folder)
    pump()
    print('exported:', [os.path.basename(x) for x in paths], flush=True)
    r.expect(sorted(os.path.basename(x) for x in paths), sorted(['critical_targets.json', 'critical_targets.csv',
                                                                 'mounting_envelopes.csv', 'mounting_envelopes.png',
                                                                 'critical_targets.png']), 'the exported files')
    r.hold(0.8)
    sheet = QImage(os.path.join(folder, 'mounting_envelopes.png'))
    r.crossfade(0.5)
    H.pages_card(r, [sheet], 'mounting_envelopes.png, as the export wrote it')
    r.cursor_on = False
    r.hold(max(3.0, r.speech_left() + 0.4))
    r.finish_speech(0.2)

    # ---- 23: batch mode (a real run, made now and not filmed)
    home = stage.home
    bin_ = os.path.join(home, 'bin')
    work = os.path.join(home, 'projects')
    os.makedirs(bin_, exist_ok=True)
    os.makedirs(work, exist_ok=True)
    with open(os.path.join(bin_, 'vantage'), 'w') as fh_:
        fh_.write(f'#!/bin/sh\nexec {sys.executable} {os.path.join(studio.ROOT, "run_vantage.py")} "$@"\n')
    os.chmod(os.path.join(bin_, 'vantage'), 0o755)
    env = {k: v for k, v in os.environ.items() if k not in ('QT_SCALE_FACTOR',)}
    env.update(PATH=bin_ + os.pathsep + env.get('PATH', ''), HOME=home, PYTHONUSERBASE=site.USER_BASE,
               QT_QPA_PLATFORM='offscreen')
    snap = st.project.snapshot()
    snap.active_uid = next(v.uid for v in snap.variants if v.name == 'Data rig · 6-cam + lidar + radars')
    snap.save(os.path.join(work, 'data-rig.vantage'), b'')
    code, lines_, secs = H.run_timed('vantage data-rig.vantage --critical-targets envelopes/', work, env, home)
    print(f'batch: exit {code} in {secs:.1f} s', flush=True)
    for _, ln in lines_:
        print('   ', ln, flush=True)
    r.expect(code, 2, 'the batch exit code')
    r.expect(any('Child at the front corner' in ln and 'NOT RESOLVED' in ln for _, ln in lines_), True,
             'the batch names the lost child')
    noise = (lambda ln: not ln.startswith(('This plugin does not', '[ WARN', 'qt.')))
    script = ([('cmd', 'vantage data-rig.vantage --critical-targets envelopes/')] +
              H.timed_output(lines_, keep=noise) +
              [('wait', 0.4), ('cmd', 'echo $?'), ('out', str(code), 'warn')])
    r.say(23)
    span = H.terminal(r, script, title='~/projects')
    print(f'terminal span {span:.1f} s, line {r.line(23):.1f} s', flush=True)
    r.crossfade(0.5)
    r.hold(max(span + 0.8, r.speech_left() + 0.3))
    r.finish_speech(0.2)

    # ---- 24: the limits, as the page states them
    r.say(24)
    r.card = None
    sa = pg.find_host.parentWidget()
    while sa is not None and not isinstance(sa, QScrollArea):
        sa = sa.parentWidget()
    sa.verticalScrollBar().setValue(sa.verticalScrollBar().maximum())      # the notes: assumptions and limits
    pump(2)
    r.crossfade(0.6)
    pump(2)
    vr = r.wrect(sa.viewport())
    r.zoom(QRectF(vr.x(), vr.y(), vr.width(), vr.height()), 1.0, margin=6)
    r.hold(max(2.0, r.speech_left() - 1.4))
    r.zoom(None, 1.0)
    r.cursor_on = False
    r.finish_speech(1.0)
