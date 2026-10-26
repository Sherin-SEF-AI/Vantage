"""09 Live coverage analysis: the ground grid, the heat maps and their legends, the hover read-out, the Coverage panel,
a variant with fewer cameras, the region of interest widened, and the finer grid.

Figures (the sample, 6-cam · v3, ROI 2.0 m, 5 cm cells), read from the app while recording (r.expect): blind in ROI
0.010 m², seen by at least two cameras 53.0 %, minimum density 3.1 px/cm, worst single-camera error 0.040 m²,
R alone 4.038 m²; the 4-cam variant: minimum density 2.1 px/cm.
"""

META = {
    'id': 'coverage',
    'part': 2,
    'title': 'Live coverage analysis',
    'promise': 'Blind ground, pixel density and overlap, recomputed with every edit.',
    'menu': 'View ▸ Overlays ▸ Heatmap  ·  Window ▸ Coverage',
}

LINES = [
    ("Everything Vantage proves starts on the ground around the vehicle: a grid of five-centimetre cells inside a "
     "region of interest, here two metres around the body.",
     "Everything Vantage proves starts on the ground around the vehicle: a grid of five centimetre cells, inside a "
     "region of interest, here two metres around the body."),
    ("Every cell is tested against every camera, through its real lens model, with the body, the rig and the other "
     "sensors' housings blocking the view.",
     "Every cell is tested against every camera, through its real lens model, with the body, the rig, and the other "
     "sensors' housings blocking the view."),
    ("From the top, the coverage map colours each patch of ground by the camera that sees it best.",
     "From the top, the coverage map colours each patch of ground by the camera that sees it best."),
    ("Density is how many pixels the best camera puts on each centimetre of ground. The blue scale starts at the "
     "project's minimum of three pixels per centimetre, with lines at the minimum and at the target of five.",
     "Density is how many pixels the best camera puts on each centimetre of ground. The blue scale starts at the "
     "project's minimum of three pixels per centimetre, with lines at the minimum, and at the target of five."),
    ("Hover anywhere for a read-out: the value under the cursor, how it compares with the thresholds, and every "
     "camera that sees that point, with its density, range and angle off its axis.",
     "Hover anywhere for a read-out: the value under the cursor, how it compares with the thresholds, and every "
     "camera that sees that point, with its density, range, and angle off its axis."),
    ("Redundancy counts the cameras on each patch. On this layout, 53 per cent of the region is seen by at least "
     "two.",
     "Redundancy counts the cameras on each patch. On this layout, fifty-three per cent of the region is seen by at "
     "least two."),
    ("The Coverage panel sums it up: the blind area inside the region, the share covered, the minimum density, and "
     "the worst blind area if any one camera ends up five degrees or ten centimetres off.",
     "The coverage panel sums it up: the blind area inside the region, the share covered, the minimum density, and "
     "the worst blind area if any one camera ends up five degrees, or ten centimetres, off."),
    ("Hover a camera's bar to light up the ground only that camera sees: the rear camera alone covers about four "
     "square metres.",
     "Hover a camera's bar to light up the ground only that camera sees: the rear camera alone covers about four "
     "square metres."),
    ("Switch to the four-camera variant and everything follows: the minimum density drops to 2.1 pixels per "
     "centimetre, and ground below the minimum turns grey and hatched.",
     "Switch to the four camera variant, and everything follows: the minimum density drops to two point one pixels "
     "per centimetre, and ground below the minimum turns grey and hatched."),
    ("Back on six cameras, widen the region of interest to three metres: the maps and the figures follow as soon as "
     "the analysis finishes, and the toolbar shows how long that took.",
     "Back on six cameras, widen the region of interest to three metres: the maps and the figures follow as soon as "
     "the analysis finishes, and the toolbar shows how long that took."),
    ("For finer detail, the grid goes down to two- or one-centimetre cells, from the Analysis menu.",
     "For finer detail, the grid goes down to two, or one centimetre cells, from the analysis menu."),
]


def scene(stage, r):
    from PyQt6.QtCore import QPointF, QRectF

    w = stage.window()
    r.win = w
    st = w.store
    rows = w.bottom.coverage.rows
    bars = w.bottom.coverage.bars

    def shown(k):
        return rows[k].v.text()
    r.expect(shown('Blind in ROI'), '0.010', 'blind in ROI')
    r.expect(shown('Seen by ≥ 2 cameras'), '53.0', 'seen by two or more')
    r.expect(shown('Minimum density'), '3.1', 'minimum density')
    stage.settle(10)
    vp = stage.vp

    # 0 - the grid inside the ROI, in 3D
    r.lower_third()
    r.say(0)
    r.cursor_on = False
    stage.orbit(r, vp, -28, 3.2, d_el=6)
    r.callout(r.tree_item(w.outliner, 'ROI'), 'Region of interest', dur=3.2, side='right')
    r.hold(0.6)
    stage.orbit(r, vp, -14, 2.4, d_el=2)
    # 1 - every cell against every camera: the best-camera map on the ground
    r.say(1)
    r.cursor_on = True
    r.menu('View', 'Overlays', 'Heatmap', 'Coverage')
    stage.settle(5)
    r.cursor_on = False
    stage.orbit(r, vp, 40, max(3.0, r.speech_left() - 0.2), d_el=6)
    r.finish_speech(0.2)

    # 2 - top view: coverage by best camera
    r.cursor_on = True
    r.say(2)
    r.toolbar('view_top')
    stage.settle(5)
    plan = w.viewarea.panes[0].plan
    r.crossfade(0.35)
    r.hold(0.4)

    def legend():
        lr = plan.legend_rect
        return None if lr is None else QRectF(r.wpos(plan, lr.topLeft()), lr.size())
    plan_rect = r.wrect(plan)
    r.move(QPointF(plan_rect.center().x() + 120, plan_rect.center().y() + 60), 0.9)
    r.hold(1.2)
    r.callout(legend, 'Best camera', dur=3.0, side='left')
    r.finish_speech(0.1)

    # 3 - density, its scale, the minimum and target lines
    r.say(3)
    r.menu('View', 'Overlays', 'Heatmap', 'Density')
    stage.settle(5)
    r.hold(0.4)
    r.zoom(legend, 1.0, margin=60)
    r.hold(max(1.5, r.speech_left() - 4.5))
    r.zoom(None, 0.9)
    r.finish_speech(0.2)

    # 4 - the hover read-out
    lo, hi = w.ctx.analysis.engine.vehicle_geom(st.project).bbox
    a = plan.w2s(hi[0] + 1.6, -0.9)
    b = plan.w2s(hi[0] + 0.7, 0.4)
    c = plan.w2s((lo[0] + hi[0]) / 2 + 0.3, hi[1] + 0.55)
    r.say(4)
    r.glide(plan, a, b, 1.6)
    r.hold(1.0)
    zone = QRectF(r.wpos(plan, QPointF(min(b.x(), c.x()) - 80, c.y() - 120)), r.wpos(plan, QPointF(b.x() + 360, b.y() + 160)))
    r.zoom(zone.normalized(), 1.0, wait=False)
    r.glide(plan, b, c, 2.4)
    r.hold(max(1.0, r.speech_left() - 0.6))
    r.zoom(None, 0.8, wait=False)
    r.leave(plan)
    r.finish_speech(0.1)

    # 5 - redundancy
    r.say(5)
    r.menu('View', 'Overlays', 'Heatmap', 'Redundancy')
    stage.settle(5)
    r.hold(0.6)
    r.figure('Seen by ≥ 2 cameras', float(shown('Seen by ≥ 2 cameras')), '%', 1, dur=4.2,
             src=rows['Seen by ≥ 2 cameras'])
    r.finish_speech(0.6)

    # 6 - the Coverage panel
    panel_rect = r.wrect(w.bottom.coverage)
    left = QRectF(panel_rect.x(), panel_rect.y() - 30, 900, panel_rect.height() + 30)
    r.say(6)
    r.zoom(left, 1.0)
    r.hold(0.4)
    r.callout(rows['Blind in ROI'], dur=2.0, pad=3)
    r.until_word(6, 'the share covered')
    r.callout(rows['Covered'], dur=1.6, pad=3)
    r.until_word(6, 'the minimum density')
    r.callout(rows['Minimum density'], dur=1.8, pad=3)
    r.until_word(6, 'worst blind area')
    r.expect(shown('Worst single-camera error'), '0.040', 'worst single-camera error')
    r.callout(rows['Worst single-camera error'], dur=max(2.0, r.speech_left() + 0.4), pad=3)
    r.finish_speech(0.3)
    r.zoom(None, 0.9)

    # 7 - hover the rear camera's bar: the ground only it sees
    r.say(7)
    R = next(i for i, row in enumerate(bars.rows) if row[2] == 'R')
    r.expect(f'{bars.rows[R][3]:.3f}', '4.038', "R's own ground")
    top, rh = bars._geom()
    pt = QPointF(bars.width() * 0.45, top + R * rh + rh / 2)
    r.hover(bars, pt, 1.0)
    for _ in range(3):
        r.send_move(bars, pt)
        r.tick()
    r.callout(r.local_rect(bars, QRectF(0, top + R * rh, bars.width(), rh)), dur=2.4, pad=2, dim=0.0)
    r.hold(1.2)
    r.figure("Seen only by R", bars.rows[R][3], 'm²', 3, dur=max(3.0, r.speech_left() + 0.2),
             src=r.local_rect(bars, QRectF(bars.width() - 80, top + R * rh, 76, rh)))
    r.finish_speech(0.0)
    r.leave(bars)

    # 8 - the four-camera variant
    v4 = next(v for v in st.project.variants if v.name == '4-cam')
    v6 = st.project.variant
    r.say(8)
    r.menu('View', 'Overlays', 'Heatmap', 'Density')
    stage.settle(4)
    r.choose(w.variant_combo, w.variant_combo.findData(v4.uid))
    stage.wait_analysis(r)
    r.hold(0.4)
    r.expect(shown('Minimum density'), '2.1', 'minimum density with four cameras')
    r.figure('Minimum density', float(shown('Minimum density')), 'px/cm', 1, dur=3.0, src=rows['Minimum density'],
             note='4-cam variant')
    r.hold(3.2)                                       # the tile and its source go before the zoom
    r.zoom(QRectF(plan_rect.x() + plan_rect.width() * 0.08, plan_rect.y(), plan_rect.width() * 0.84,
                  plan_rect.height()), 1.0, wait=False)
    r.finish_speech(0.6)
    r.zoom(None, 0.8, wait=False)

    # 9 - back to six cameras; the ROI widened to 3 m
    r.say(9)
    r.choose(w.variant_combo, w.variant_combo.findData(v6.uid))
    stage.wait_analysis(r)
    r.menu('Analysis', 'Region of interest')
    stage.settle(3)
    roi = w.inspector.pages['roi']
    r.zoom(QRectF(r.wrect(w.docks['inspector']).x() - 40, r.wrect(w.docks['inspector']).y(),
                  r.wrect(w.docks['inspector']).width() + 40, 520), 0.9)
    area0 = float(shown('ROI area'))
    r.enter(roi.dist, '3')
    if abs(st.project.roi.distance - 3.0) > 1e-6:                        # Enter did not commit: commit it
        roi.dist.committed.emit(3.0)
    r.zoom(None, 0.8, wait=False)
    stage.wait_analysis(r)
    r.hold(0.8)
    r.callout(w.state_btn, 'Analysis time', dur=2.8, style='box', pad=4, side='below')
    r.figure('ROI area', float(shown('ROI area')), 'm²', 3, dur=3.6, src=rows['ROI area'], start=area0,
             note='ROI 2 m \u2192 3 m')
    r.finish_speech(0.2)

    # 10 - the finer grid
    r.say(10)
    r.menu('Analysis', 'Ground resolution', '2 cm')
    stage.wait_analysis(r, timeout=120)
    r.hold(0.6)
    r.zoom(QRectF(plan_rect.x(), plan_rect.y(), plan_rect.width() * 0.55, plan_rect.height() * 0.5), 1.0)
    r.hold(max(1.2, r.speech_left()))
    r.zoom(None, 0.8)
    r.cursor_on = False
    r.finish_speech(1.2)
