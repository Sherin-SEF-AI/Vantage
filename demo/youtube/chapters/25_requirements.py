"""25 Requirements scorecard: the data rig with two GNSS antennas and a dual-antenna INS (stage._nav_window) signed off
against written requirements. Analysis ▸ Requirements scorecard; Add from the plan; a template; Run (each study a
requirement needs, once, in the background); the verdict and the tiles; the failing camera–lidar misalignment row and
its evidence; the sync plan opened from the evidence, its suggested trigger lines applied; back on the scorecard the
row passes at once and the study values are marked stale; Export and the batch mode's exit code.

Figures read from the app while recording (r.expect): 24 requirements from the plan, 25 with the template; after Run:
verdict Fail, 11 of 15 must requirements pass; R17 (camera–lidar misalignment ≤ 1 ms) 12.41 ms, the suggested lines
0.94 ms; after Apply: 0.94 ms, Pass; 16 values stale.
"""

META = {
    'id': 'requirements',
    'part': 3,
    'title': 'Requirements scorecard',
    'promise': 'The rig signed off against the programme’s written requirements.',
    'menu': 'Analysis ▸ Requirements scorecard',
}

LINES = [
    ("A programme writes its requirements down: so many hours per disk, pedestrians seen to 40 m, cameras within a "
     "millisecond of the lidar. The scorecard turns them into checks on the app's own studies.",
     "A programme writes its requirements down: so many hours per disk, pedestrians seen to forty metres, cameras "
     "within a millisecond of the lidar. The scorecard turns them into checks, on the app's own studies."),
    ("Add from the plan turns what the plan already sets into requirements: the required range per sector, the "
     "recording shift, the blur limit, the power runtime and more. Twenty-four here.",
     "Add from the plan turns what the plan already sets into requirements: the required range per sector, the "
     "recording shift, the blur limit, the power runtime, and more. Twenty-four here."),
    ("Templates add typical data-rig requirements: here, pedestrians resolved to 40 m in every sector. A project "
     "brief's own sentences can be suggested too, for you to tick.",
     "Templates add typical data rig requirements: here, pedestrians resolved to forty metres in every sector. A "
     "project brief's own sentences can be suggested too, for you to tick."),
    ("Cheap figures are checked on every edit. Run computes each study the requirements need, once, in the "
     "background.",
     "Cheap figures are checked on every edit. Run computes each study the requirements need, once, in the "
     "background."),
    ("One verdict: fail. Eleven of fifteen must requirements pass. Each row shows its value, its margin, where "
     "positive is good, and its status.",
     "One verdict: fail. Eleven of fifteen must requirements pass. Each row shows its value, its margin, where "
     "positive is good, and its status."),
    ("Select a failing row for its evidence: on one trigger line the cameras are 12.4 ms off the lidar sweep, and the "
     "sync plan's suggested lines would bring that to 0.94 ms.",
     "Select a failing row for its evidence: on one trigger line, the cameras are twelve point four milliseconds off "
     "the lidar sweep, and the sync plan's suggested lines would bring that to zero point nine four."),
    ("Open the study from the evidence, and apply the suggested trigger lines.",
     "Open the study from the evidence, and apply the suggested trigger lines."),
    ("Back on the scorecard the row passes at once, with 0.94 ms. The values from heavier studies are marked stale "
     "until the next run.",
     "Back on the scorecard, the row passes at once, with zero point nine four milliseconds. The values from heavier "
     "studies are marked stale, until the next run."),
    ("Export writes the scorecard and a design-review pack as PDF, with every study's assumptions and limits. In "
     "batch mode, a failing must requirement exits with code 5, for CI.",
     "Export writes the scorecard, and a design review pack as P D F, with every study's assumptions and limits. In "
     "batch mode, a failing must requirement exits with code five, for C I."),
]


def scene(stage, r):
    import numpy as np
    from PyQt6.QtCore import QPoint, QPointF, QRectF
    import clip_requirements as CR

    w = stage._nav_window()
    r.win = w
    r.cursor_on = False
    vp = stage.vp
    vp.cam.az, vp.cam.el = -36.0, 24.0
    w.viewarea.frame_all()
    stage.settle(2)
    r.cur = np.array([1150.0, 560.0])

    # ---- 0: the opener; the page from its menu
    r.lower_third()
    r.say(0)
    stage.orbit(r, vp, 34, 3.4, d_el=3)
    r.cursor_on = True
    r.menu('Analysis', 'Requirements scorecard')
    pg = w.page('requirements')
    stage.settle(1)
    r.crossfade(0.4)
    r.finish_speech(0.1)
    t = pg.table

    def rows_rect(r0, c0, r1, c1):
        return lambda: QRectF(r.local_rect(t.viewport(), QRectF(t.visualItemRect(t.item(r0, c0))))()).united(
            r.local_rect(t.viewport(), QRectF(t.visualItemRect(t.item(r1, c1))))())

    tw = r.wrect(t)
    tiles = r.wrect(pg.tiles['verdict']).united(r.wrect(pg.tiles['open']))

    # ---- 1: from the plan
    r.say(1)
    r.click(pg.b_plan, None, 0.8)
    stage.settle(1)
    r.expect(t.rowCount(), 24, 'requirements from the plan')
    r.zoom(QRectF(tw.x(), tw.y() - 30, tw.width() * 0.62, tw.height() * 0.62), 1.0)
    r.until_word(1, 'Twenty-four')
    r.callout(rows_rect(0, 0, min(10, t.rowCount() - 1), 2), '', dur=max(1.8, r.speech_left() + 0.4), pad=3, dim=0.2)
    r.finish_speech(0.2)

    # ---- 2: a template; the brief
    r.say(2)
    r.zoom(None, 0.8, wait=False)
    CR._template(r, pg, 'Pedestrians resolved to 40 m')
    stage.settle(1)
    r.expect(t.rowCount(), 25, 'requirements with the template')
    t.scrollToBottom()
    r.tick(2)
    r.callout(rows_rect(t.rowCount() - 1, 0, t.rowCount() - 1, 2), '', dur=2.4, pad=3, dim=0.2)
    r.until_word(2, "A project brief's")
    r.zoom(QRectF(r.wrect(pg.b_plan).x() - 20, r.wrect(pg.b_new).y() - 60, 400, 220), 0.9)
    r.callout(pg.b_brief, '', dur=max(1.8, r.speech_left() + 0.3), pad=4, dim=0.2)
    r.finish_speech(0.2)

    # ---- 3: Run (a marked time-lapse when it is long)
    r.say(3)
    r.zoom(None, 0.8, wait=False)
    CR._run(r, pg, speed=2)
    stage.settle(1)
    r.finish_speech(0.2)

    # ---- 4: the verdict and the rows
    r.say(4)
    r.expect(pg.tiles['verdict'].value, 'Fail', 'verdict')
    r.expect(pg.tiles['must'].value, '11/15', 'must requirements passed')
    r.zoom(QRectF(tiles.x(), tiles.y() - 10, tiles.width(), tiles.height() + 30), 1.0)
    r.callout(pg.tiles['verdict'], '', dur=1.6, pad=3, dim=0.2)
    r.until_word(4, 'Eleven of fifteen')
    r.callout(pg.tiles['must'], '', dur=1.8, pad=3, dim=0.2)
    r.until_word(4, 'Each row')
    t.scrollToTop()
    r.tick(2)
    r.zoom(QRectF(tw.x() + tw.width() * 0.3, tw.y() - 30, tw.width() * 0.7, tw.height() * 0.6), 1.0)
    r.callout(rows_rect(0, 4, 9, 6), '', dur=max(2.0, r.speech_left() + 0.4), pad=3, dim=0.2)
    r.finish_speech(0.2)

    # ---- 5: a failing row's evidence
    r.say(5)
    k = CR._row_of(pg, 'timing.misalignment_ms')
    r.expect(t.item(k, 1).text(), 'Camera–lidar misalignment at most 1 ms', 'the timing requirement')
    r.expect(t.item(k, 4).text(), '12.41 ms', 'its value')
    CR._click_row(r, pg, k)
    stage.settle(1)
    ev = {pg.ev_table.item(i, 0).text(): pg.ev_table.item(i, 1).text() for i in range(pg.ev_table.rowCount())}
    print('evidence:', ev, flush=True)
    r.expect(ev.get('Worst misalignment, suggested lines'), '0.94 ms', 'suggested lines')
    r.zoom(CR._row_rect(r, pg, k).united(CR._evidence_rect(r, pg)), 1.0, margin=10)
    r.callout(lambda: CR._row_rect(r, pg, k), '', dur=2.4, pad=3, dim=0.2)
    r.until_word(5, "the sync plan's suggested")
    ei = next(i for i in range(pg.ev_table.rowCount()) if pg.ev_table.item(i, 0).text().startswith('Worst misalignment, sug'))
    r.callout(r.local_rect(pg.ev_table.viewport(), QRectF(pg.ev_table.visualItemRect(pg.ev_table.item(ei, 0)).united(
        pg.ev_table.visualItemRect(pg.ev_table.item(ei, 1))))), '', dur=max(1.8, r.speech_left() + 0.4), pad=3, dim=0.2)
    r.finish_speech(0.2)

    # ---- 6: open the study; apply the suggested lines
    r.say(6)
    r.zoom(None, 0.8, wait=False)
    r.click(pg.b_open, dur=0.7)
    tm = w.page('timing')
    stage.settle(1)
    r.crossfade(0.4)
    r.hold(0.3)
    r.click(tm.b_suggest, dur=0.6)
    stage.settle(1)
    r.hold(0.5)
    r.click(tm.b_apply, dur=0.6)
    stage.settle(1)
    r.hold(0.4)
    r.finish_speech(0.2)

    # ---- 7: back on the scorecard: passes at once; stale values
    r.say(7)
    r.menu('Analysis', 'Requirements scorecard')
    stage.settle(1)
    r.crossfade(0.4)
    r.expect(t.item(k, 4).text(), '0.94 ms', 'the row after Apply')
    r.expect(t.item(k, 6).text(), 'Pass', 'its status after Apply')
    r.zoom(CR._row_rect(r, pg, k).united(CR._evidence_rect(r, pg)), 1.0, margin=10)
    r.callout(lambda: CR._row_rect(r, pg, k), '', dur=2.4, pad=3, dim=0.2)
    r.until_word(7, 'marked stale')
    r.zoom(None, 0.8)
    print('verdict note:', pg.tiles['verdict'].note, '| status:', pg.status.text(), flush=True)
    r.expect('16 stale' in pg.tiles['verdict'].note, True, 'stale values')
    r.callout(pg.tiles['verdict'], '', dur=max(1.8, r.speech_left() + 0.4), pad=3, dim=0.2)
    r.finish_speech(0.2)

    # ---- 8: export; the batch exit code
    r.say(8)
    er = r.wrect(pg.b_export)
    r.zoom(QRectF(er.x() - 160, er.y() - 160, 460, 220), 0.9)
    r.callout(pg.b_export, '', dur=max(2.4, r.at_word(8, 'In batch mode') - r.now()), pad=4, dim=0.2)
    r.until_word(8, 'In batch mode')
    r.zoom(None, 0.8)
    r.callout(pg.tiles['verdict'], '', dur=max(1.8, r.speech_left() + 0.3), pad=3, dim=0.2)
    r.finish_speech(0.6)
    r.cursor_on = False
    r.hold(0.6)
