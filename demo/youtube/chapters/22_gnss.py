"""22 GNSS and IMU: the sample's data rig with its lidar raised on a sensor mast (as demo/clip_gnss.py sets it up);
two survey GNSS antennas and a dual-antenna GNSS/INS added from the Vehicle menu; Analysis ▸ GNSS and IMU: the sky
plots, the heading baseline, the lever arms and the pose error budget; the primary antenna moved to the foot of the
mast: its sky plot, the findings.

Figures read from the app while recording (r.expect): both antennas 98.3 % open above the 10° mask (about 31 of 32
satellites; the mast and its head take the rest); the baseline 1.15 m, heading σ 0.173° (typical INS); IMU1 0.45 m
from the vehicle origin; every sensor's position to 32 mm horizontally, 154 mm at 50 m. GN1 beside the mast: 89.8 %
open, about 29 of 32 satellites, multipath flagged, baseline 0.74 m.
"""

META = {
    'id': 'gnss',
    'part': 3,
    'title': 'GNSS and IMU',
    'promise': 'Open sky for the antennas, and the lever arms the INS needs.',
    'menu': 'Vehicle ▸ Add GNSS antenna / Add IMU  ·  Analysis ▸ GNSS and IMU',
}

LINES = [
    ("A data rig logs where it is, too. Here the lidar sits on a sensor mast; add two survey GNSS antennas from the "
     "Vehicle menu, each placed on the roof clear of the mast.",
     "A data rig logs where it is, too. Here the lidar sits on a sensor mast. Add two survey G N S S antennas from the "
     "vehicle menu, each placed on the roof, clear of the mast."),
    ("Then a dual-antenna GNSS/INS, which lands above the rear axle. Its figures are typical ones from the library.",
     "Then a dual antenna G N S S, I N S, which lands above the rear axle. Its figures are typical ones, from the "
     "library."),
    ("Analysis ▸ GNSS and IMU traces rays from each antenna's phase centre over the whole sky, against the body, the "
     "rig and every sensor's housing.",
     "G N S S and I M U, from the analysis menu, traces rays from each antenna's phase centre over the whole sky, "
     "against the body, the rig, and every sensor's housing."),
    ("The sky plots: forward is up, a ring every 15 degrees, the elevation mask dashed. The mast and its head take "
     "1.7 per cent of each antenna's sky: about 31 of 32 satellites stay in view.",
     "The sky plots: forward is up, a ring every fifteen degrees, the elevation mask dashed. The mast and its head "
     "take one point seven per cent of each antenna's sky: about thirty-one of thirty-two satellites stay in view."),
    ("The two antennas form a 1.15 m heading baseline: with this INS, a heading to 0.17 degrees. Here is the baseline "
     "in the INS's own frame, as it must be told.",
     "The two antennas form a one point one five metre heading baseline: with this I N S, a heading to zero point one "
     "seven degrees. Here is the baseline in the I N S's own frame, as it must be told."),
    ("Lever arms run from the IMU to every antenna, camera, lidar and radar, and to the vehicle origin, the nuScenes "
     "ego frame, in millimetres.",
     "Lever arms run from the I M U to every antenna, camera, lidar, and radar, and to the vehicle origin, the new "
     "scenes ego frame, in millimetres."),
    ("The pose error budget adds the INS's own errors to each lever arm: every sensor's position is known to 32 mm, "
     "and a point 50 m away to 154 mm.",
     "The pose error budget adds the I N S's own errors to each lever arm: every sensor's position is known to "
     "thirty-two millimetres, and a point fifty metres away, to a hundred and fifty-four."),
    ("Now move the primary antenna to the foot of the mast. Its sky plot shows the wedge the mast takes: 89.8 per "
     "cent open, about 29 satellites.",
     "Now move the primary antenna to the foot of the mast. Its sky plot shows the wedge the mast takes: eighty-nine "
     "point eight per cent open, about twenty-nine satellites."),
    ("And the findings flag what else went wrong: multipath off the mast, eight centimetres away, and a baseline cut "
     "to 0.74 m. Export writes the lever arms for the INS.",
     "And the findings flag what else went wrong: multipath off the mast, eight centimetres away, and a baseline cut "
     "to zero point seven four metres. Export writes the lever arms for the I N S."),
]


def scene(stage, r):
    import numpy as np
    from PyQt6.QtCore import QPointF, QRectF
    import clip_gnss
    from vantage.core import sensors
    from vantage.ui.sensorview import set_props

    w = stage._rig_window(clip_gnss._mast_rig)
    r.win = w
    st = w.store
    r.cursor_on = False
    vp = stage.vp
    vp.cam.az, vp.cam.el = -50.0, 34.0
    w.viewarea.frame_all()
    vp.cam.dist *= 0.82
    stage.settle(2)
    r.cur = np.array([1150.0, 560.0])

    # ---- 0: the antennas from the Vehicle menu
    r.lower_third()
    r.say(0)
    stage.orbit(r, vp, 18, 2.4, d_el=2)
    r.cursor_on = True
    r.until_word(0, 'Add two')
    r.menu('Vehicle', 'Add GNSS antenna', 'Survey GNSS antenna')
    stage.settle(2)
    r.hold(0.6)
    r.menu('Vehicle', 'Add GNSS antenna', 'Survey GNSS antenna')
    stage.settle(2)
    r.cursor_on = False
    stage.orbit(r, vp, 16, max(1.6, r.speech_left() + 0.2), d_el=2)
    r.finish_speech(0.1)

    # ---- 1: the INS
    r.say(1)
    r.cursor_on = True
    r.menu('Vehicle', 'Add IMU', 'Dual-antenna GNSS/INS')
    stage.settle(2)
    kinds = [s.kind for s in st.variant.sensors]
    r.expect((kinds.count('gnss'), kinds.count('imu')), (2, 1), 'antennas and IMU added')
    r.cursor_on = False
    stage.orbit(r, vp, 22, max(1.8, r.speech_left() + 0.3), d_el=3)
    r.finish_speech(0.1)

    # ---- 2: the study from its menu
    r.say(2)
    r.cursor_on = True
    r.menu('Analysis', 'GNSS and IMU')
    pg = w.page('gnss')
    r.crossfade(0.4)
    r.wait_for(lambda: pg.result is not None and not pg.busy, timeout=120)
    stage.settle(1)
    r.finish_speech(0.2)
    res = pg.result
    a1, a2 = res.antennas[0], res.antennas[1]
    r.expect((f'{a1.visible * 100:.1f}', f'{a2.visible * 100:.1f}'), ('98.3', '98.3'), 'open sky')
    r.expect((f'{a1.expected_sats:.0f}', f'{a1.n_sats:.0f}'), ('31', '32'), 'satellites in view')

    # ---- 3: the sky plots
    r.say(3)
    plots = r.wrect(pg.plots[0]).united(r.wrect(pg.plots[-1]))
    r.zoom(plots, 1.0, margin=12)
    r.move_to(pg.plots[0], QPointF(pg.plots[0].width() * 0.8, pg.plots[0].height() * 0.85), 0.8)
    r.until_word(3, 'The mast and its head')
    r.zoom(None, 0.8)
    r.figure('Open sky · each antenna', a1.visible * 100, '%', 1, dur=max(3.0, r.speech_left() + 0.6),
             src=pg.tiles['sky'], note='about 31 of 32 satellites')
    r.finish_speech(0.2)

    # ---- 4: the heading baseline
    r.say(4)
    b = res.baseline
    r.expect((f'{b["horizontal_m"]:.2f}', f'{b["heading_sigma_deg"]:.3f}'), ('1.15', '0.173'), 'baseline and heading')
    r.figure('Heading baseline', b['horizontal_m'], 'm', 2, dur=3.0, src=pg.tiles['heading'],
             note='heading σ 0.173° · typical INS')
    r.hold(3.0)
    br = r.wrect(pg.base_text)
    r.zoom(QRectF(br.x() - 10, br.y() - 30, br.width() + 20, br.height() + 50), 1.0)
    r.callout(pg.base_text, '', dur=max(2.0, r.speech_left() + 0.4), pad=4, dim=0.2)
    r.finish_speech(0.2)

    # ---- 5: the lever arms
    r.say(5)
    lr = r.wrect(pg.levers)
    r.zoom(QRectF(lr.x() - 10, lr.y() - 30, lr.width() + 20, lr.height() + 40), 1.0)
    r.move_to(pg.levers.viewport(), QPointF(pg.levers.viewport().width() * 0.6, 12), 0.8)
    r.until_word(5, 'vehicle origin')
    t = pg.levers
    r.callout(r.local_rect(t.viewport(), QRectF(t.visualItemRect(t.item(0, 0)).united(
        t.visualItemRect(t.item(0, t.columnCount() - 1))))), '', dur=max(2.0, r.speech_left() + 0.4), pad=3, dim=0.2)
    r.finish_speech(0.2)

    # ---- 6: the pose error budget
    r.say(6)
    rows = [x for x in res.budget['rows'] if x['kind'] != 'origin']
    worst = max(rows, key=lambda x: x['horizontal_mm'])
    r.expect((f'{worst["horizontal_mm"]:.0f}', f'{worst["far_mm"]:.0f}'), ('32', '154'), 'pose error budget')
    bud = r.wrect(pg.budget)
    nr = r.wrect(pg.budget_note)
    r.zoom(QRectF(bud.x() - 10, nr.y() - 30, bud.width() + 20, bud.bottom() - nr.y() + 40), 1.0)
    r.until_word(6, 'thirty-two')
    r.zoom(None, 0.8)
    r.figure('Sensor position σ', worst['horizontal_mm'], 'mm', 0, dur=max(3.0, r.speech_left() + 0.6),
             src=pg.tiles['pose'], note='154 mm at 50 m')
    r.finish_speech(0.2)

    # ---- 7: the primary antenna at the foot of the mast
    r.say(7)
    gn1 = next(s for s in st.variant.sensors if s.kind == 'gnss')
    mast = min(st.project.rig, key=lambda part: part.z)
    x = round(mast.x - 0.11, 3)
    m = st.project.sensor_model_for(gn1)
    z = round(sensors.surface_top(x, 0.0, st.project.vehicle.mesh, st.project.rig) + m.optical_z_mm / 1000.0, 3)
    r.click(pg.plots[0], None, 0.7)
    before = pg.result
    set_props(w.ctx, gn1.uid, f'Moved {gn1.tag} beside the mast', x=x, y=0.0, z=z)
    r.wait_for(lambda: pg.result is not before and not pg.busy, timeout=60)
    stage.settle(1)
    res2 = pg.result
    g1 = next(a for a in res2.antennas if a.tag == gn1.tag)
    r.expect((f'{g1.visible * 100:.1f}', f'{g1.expected_sats:.0f}'), ('89.8', '29'), 'GN1 beside the mast')
    r.cursor_on = False
    r.zoom(r.wrect(pg.plots[0]), 1.0, margin=14)
    r.until_word(7, 'eighty-nine')
    r.zoom(None, 0.8)
    r.figure('Open sky · GN1', g1.visible * 100, '%', 1, dur=max(2.8, r.speech_left() + 0.6), src=pg.tiles['sky'],
             start=98.3, note='beside the mast · about 29 satellites')
    r.finish_speech(0.2)

    # ---- 8: the findings; export
    r.say(8)
    r.expect(f'{res2.baseline["horizontal_m"]:.2f}', '0.74', 'baseline after the move')
    texts = [f.text.text() for f in pg.findings]
    print('findings:', texts, flush=True)
    r.expect(any('multipath' in x and '0.08 m' in x for x in texts), True, 'multipath finding')
    first, last = pg.findings[0], pg.findings[min(2, len(pg.findings) - 1)]
    fa, fb = r.wrect(first), r.wrect(last)
    r.zoom(QRectF(fa.x() - 10, fa.y() - 30, fa.width() + 20, fb.bottom() - fa.y() + 40), 1.0)
    r.callout(lambda: r.wrect(pg.findings[0]).united(r.wrect(pg.findings[min(2, len(pg.findings) - 1)])), '',
              dur=max(2.4, r.at_word(8, 'Export writes') - r.now()), pad=3, dim=0.2)
    r.until_word(8, 'Export writes')
    r.zoom(QRectF(r.wrect(pg.b_export).x() - 280, r.wrect(pg.b_export).y() - 120, 420, 180), 0.8)
    r.callout(pg.b_export, '', dur=max(1.6, r.speech_left()), pad=4, dim=0.2)
    r.finish_speech(0.3)
    r.zoom(None, 0.8)
    r.hold(0.8)
