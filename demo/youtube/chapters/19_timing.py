"""19 Sync and timing: Analysis ▸ Sync and timing on the sample's data rig (six cameras on trigger line A, the roof
lidar L1 at 10 Hz). The dial of one rotation, the camera table, Suggest groups and Apply, the rolling-shutter readout
and exposure, motion blur at driving speed, the exposure doubled.

Figures read from the app while recording (r.expect): worst misalignment 12.4 ms (17 cm at 50 km/h, ML) on one
line; suggested 4 lines: 12.4 -> 0.9 ms (17 -> 1 cm); applied: 0.9 ms; motion blur 8.5 px (R) for an object 5 m away
at 50 km/h with a 2 ms exposure, exposures up to 0.24 ms within 1 px; at 4 ms: 16.9 px.
"""

META = {
    'id': 'timing',
    'part': 3,
    'title': 'Sync and timing',
    'promise': 'Fire every camera as the lidar sweeps past it.',
    'menu': 'Analysis ▸ Sync and timing',
}

LINES = [
    ("Datasets such as nuScenes pair each camera image with a lidar sweep, so each camera should expose just as the "
     "spinning lidar sweeps across its view. Sync and timing plans when.",
     "Datasets such as new scenes pair each camera image with a lidar sweep, so each camera should expose just as the "
     "spinning lidar sweeps across its view. Sync and timing plans when."),
    ("The dial is one rotation of the roof lidar at 10 Hz: each camera's field of view, the moment the sweep crosses "
     "it, and the moment its trigger fires.",
     "The dial is one rotation of the roof lidar, at ten hertz: each camera's field of view, the moment the sweep "
     "crosses it, and the moment its trigger fires."),
    ("All six cameras share trigger line A, so they fire together, and the table shows the cost: up to 12.4 ms off "
     "the sweep, 17 cm of travel at 50 km/h.",
     "All six cameras share trigger line A, so they fire together, and the table shows the cost: up to twelve point "
     "four milliseconds off the sweep. Seventeen centimetres of travel, at fifty kilometres an hour."),
    ("Suggest groups splits the cameras over the four trigger lines, trying every grouping: the worst misalignment "
     "drops to 0.9 ms.",
     "Suggest groups splits the cameras over the four trigger lines, trying every grouping: the worst misalignment "
     "drops to zero point nine milliseconds."),
    ("Apply sets the cameras' trigger lines as one undo step, and the dial shows the new triggers.",
     "Apply sets the cameras' trigger lines as one undo step, and the dial shows the new triggers."),
    ("A rolling shutter reads its rows over the readout time, so each trigger aims the middle row's exposure at the "
     "sweep. Enter your sensor's readout and exposure here.",
     "A rolling shutter reads its rows over the readout time, so each trigger aims the middle row's exposure at the "
     "sweep. Enter your sensor's readout, and exposure, here."),
    ("Motion blur needs no lidar. At 50 km/h, a 2 ms exposure smears an object 5 m away by up to 8.5 pixels; to stay "
     "within one pixel, expose for at most 0.24 ms.",
     "Motion blur needs no lidar. At fifty kilometres an hour, a two millisecond exposure smears an object five metres "
     "away by up to eight point five pixels. To stay within one pixel, expose for at most zero point two four "
     "milliseconds."),
    ("Double the exposure to 4 ms and the worst blur doubles too, to 16.9 pixels.",
     "Double the exposure to four milliseconds, and the worst blur doubles too, to sixteen point nine pixels."),
    ("These times are planned, not measured: the live rig page measures the real sync skew. The plan exports as JSON "
     "with the dial, for the batch mode and the assistant too.",
     "These times are planned, not measured: the live rig page measures the real sync skew. The plan exports as "
     "Jason, with the dial, for the batch mode, and the assistant, too."),
]


def scene(stage, r):
    import numpy as np
    from PyQt6.QtCore import QPointF, QRectF

    w = stage._rig_window()
    r.win = w
    r.cursor_on = False
    vp = stage.vp
    vp.cam.az, vp.cam.el = -150.0, 26.0
    w.viewarea.frame_all()
    stage.settle(2)
    r.cur = np.array([1150.0, 560.0])

    # ---- 0: the opener; the page from its menu
    r.lower_third()
    r.say(0)
    stage.orbit(r, vp, 40, 3.4, d_el=4)
    r.cursor_on = True
    r.until_word(0, 'Sync and timing plans')
    r.menu('Analysis', 'Sync and timing')
    pg = w.page('timing')
    stage.settle(2)
    r.crossfade(0.4)
    r.finish_speech(0.1)

    res = pg.res
    m = res.metrics
    r.expect(f'{m["worst_ms"]:.1f}', '12.4', 'worst misalignment on one line')
    r.expect(f'{m["worst_travel_cm"]:.0f}', '17', 'travel during the worst misalignment')
    r.expect(res.lidar.rate_hz, 10.0, 'lidar rate')

    # ---- 1: the dial
    r.say(1)
    dial = r.wrect(pg.dial)
    r.zoom(dial, 1.0, margin=10)
    r.move_to(pg.dial, QPointF(pg.dial.width() * 0.62, pg.dial.height() * 0.3), 1.0)
    r.finish_speech(0.3)

    # ---- 2: the table: one line, 12.4 ms
    r.say(2)
    t = pg.table
    tr = r.wrect(t)
    r.zoom(QRectF(tr.x() - 8, tr.y() - 34, tr.width() + 16, 34 + 23 * 7 + 10), 1.0)
    rows = range(t.rowCount())

    def col(c):
        return lambda: QRectF(r.local_rect(t.viewport(), QRectF(t.visualItemRect(t.item(0, c))))()).united(
            r.local_rect(t.viewport(), QRectF(t.visualItemRect(t.item(t.rowCount() - 1, c))))())
    r.callout(col(1), '', dur=2.2, pad=3, dim=0.2)
    r.until_word(2, 'up to twelve')
    i_ml = next(i for i in rows if t.item(i, 0).text() == 'ML')
    r.figure('Worst misalignment', m['worst_ms'], 'ms', 1, dur=max(3.2, r.speech_left() + 0.8),
             src=r.local_rect(t.viewport(), QRectF(t.visualItemRect(t.item(i_ml, 5)))),
             note=f'{m["worst_travel_cm"]:.0f} cm at 50 km/h')
    r.finish_speech(0.2)

    # ---- 3: Suggest groups
    r.say(3)
    r.click(pg.b_suggest, None, 0.7)
    stage.settle(1)
    r.hold(0.3)
    after = res.suggestion['worst_ms']
    r.expect(f'{after:.1f}', '0.9', 'suggested worst misalignment')
    r.expect(res.suggestion['lines'], 4, 'suggested lines')
    r.callout(col(1), '', dur=2.6, pad=3, dim=0.2)
    r.until_word(3, 'drops to')
    pr = r.wrect(pg.proposal)
    r.zoom(QRectF(pr.x() - 8, pr.y() - 60, pr.width() + 16, pr.height() + 90), 0.8)
    r.callout(pg.proposal, '', dur=max(2.2, r.speech_left() + 0.6), pad=4, dim=0.25)
    r.finish_speech(0.2)

    # ---- 4: Apply; the dial
    r.say(4)
    r.click(pg.b_apply, None, 0.7)
    stage.settle(1)
    r.expect(f'{pg.res.metrics["worst_ms"]:.1f}', '0.9', 'worst misalignment after Apply')
    r.zoom(dial, 0.8, margin=10)
    r.figure('Worst misalignment', pg.res.metrics['worst_ms'], 'ms', 1, dur=max(2.6, r.speech_left() + 0.6),
             src=pg.head, start=12.4, note='four trigger lines')
    r.finish_speech(0.2)

    # ---- 5: readout and exposure
    r.say(5)
    f = pg.fields
    fr = r.wrect(f['speed_kmh']).united(r.wrect(f['trigger_lines']))
    r.zoom(QRectF(fr.x() - 110, fr.y() - 40, 420, fr.height() + 160), 1.0)
    r.until_word(5, 'Enter your')
    r.callout(lambda: r.wrect(f['exposure_ms']).united(r.wrect(f['readout_ms'])).adjusted(-100, 0, 0, 0), '',
              dur=max(2.2, r.speech_left() + 0.4), pad=4, dim=0.2)
    r.finish_speech(0.2)

    # ---- 6: motion blur
    r.say(6)
    mm = pg.res.metrics
    r.expect(f'{mm["worst_blur_px"]:.1f}', '8.5', 'worst blur at 2 ms')
    r.expect(f'{mm["max_exposure_ms"]:.2f}', '0.24', 'longest exposure within 1 px')
    r.callout(pg.figures, '', dur=2.4, pad=4, dim=0.2)
    r.until_word(6, 'eight point five')
    r.figure('Worst motion blur', mm['worst_blur_px'], 'px', 1, dur=max(3.2, r.speech_left() + 0.4), src=pg.figures,
             note='2 ms exposure · 50 km/h · object at 5 m')
    r.finish_speech(0.2)

    # ---- 7: the exposure doubled
    r.say(7)
    r.enter(f['exposure_ms'], '4')
    if abs(pg.settings.exposure_ms - 4.0) > 1e-6:
        f['exposure_ms'].committed.emit(4.0)
    stage.settle(1)
    r.hold(0.3)
    b4 = pg.res.metrics['worst_blur_px']
    r.expect(f'{b4:.1f}', '16.9', 'worst blur at 4 ms')
    r.figure('Worst motion blur', b4, 'px', 1, dur=max(3.0, r.speech_left() + 0.6), src=pg.figures, start=8.5,
             note='4 ms exposure')
    r.finish_speech(0.3)

    # ---- 8: planned, not measured; export
    r.say(8)
    r.zoom(None, 0.9)
    r.until_word(8, 'The plan exports')
    r.zoom(QRectF(r.wrect(pg.b_export).x() - 300, r.wrect(pg.b_export).y() - 140, 460, 200), 0.8)
    r.callout(pg.b_export, '', dur=max(1.8, r.speech_left()), pad=4, dim=0.2)
    r.finish_speech(0.3)
    r.zoom(None, 0.8)
    r.cursor_on = False
    r.hold(0.8)
