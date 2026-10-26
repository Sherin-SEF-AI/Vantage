"""06 Rig parts: Vehicle ▸ Add rig part (roof rack: two crossbars and a platform; sensor mast: a tube with a head 0.5 m
above the roof), the parts in the Outliner and the Inspector, a camera placed on the mast head with Place on surface,
its view with the rig blocking it, the platform's Occludes switch, camera light, and the parts in the ROS 2 URDF.

Figures read from the app while recording (r.expect): the rack is three parts and the mast two; the mast head's
centre is 2.092 m up (the roof is 1.532 m); the roof camera is tagged F2, on "Sensor mast · head"; with it the ground
seen by at least two cameras goes from 53.0 to 59.4 %; "Body in view" for F2 44.2 % with the platform occluding,
42.9 % without; the URDF preview has a fixed link per part.
"""

META = {
    'id': 'rig',
    'part': 1,
    'title': 'Rig parts',
    'promise': 'Racks and masts that carry sensors, and block cameras like the body.',
    'menu': 'Vehicle ▸ Add rig part  ·  Inspector ▸ Rig part',
}

LINES = [
    ("The rig is part of the design. Vehicle ▸ Add rig part puts a roof rack on the roof, two crossbars and a "
     "platform, sized from the roof, in one undo step.",
     "The rig is part of the design. Vehicle, add rig part, puts a roof rack on the roof: two crossbars and a "
     "platform, sized from the roof, in one undo step."),
    ("A sensor mast is an upright tube with a head half a metre above the roof; a sensor enclosure and a crossbar "
     "are added the same way.",
     "A sensor mast is an upright tube, with a head half a metre above the roof. A sensor enclosure and a crossbar "
     "are added the same way."),
    ("Parts are boxes, upright cylinders or bars across the vehicle. Each is listed in the Outliner, picked with a "
     "click, and edited in the Inspector: centre, size and yaw.",
     "Parts are boxes, upright cylinders, or bars across the vehicle. Each is listed in the outliner, picked with a "
     "click, and edited in the inspector: centre, size, and yaw."),
    ("And two switches: whether the part blocks the cameras, and whether it can carry them.",
     "And two switches: whether the part blocks the cameras, and whether it can carry them."),
    ("The mast head is mountable, so Place on surface puts a camera on its front, two metres up. The ground seen by "
     "at least two cameras rises from 53 to 59 per cent.",
     "The mast head is mountable, so place on surface puts a camera on its front, two metres up. The ground seen by "
     "at least two cameras rises from fifty-three, to fifty-nine per cent."),
    ("Through its lens, everything that blocks it is tinted in the camera's colour: the bonnet, the roof, and the "
     "rack's platform below it.",
     "Through its lens, everything that blocks it is tinted in the camera's colour: the bonnet, the roof, and the "
     "rack's platform below it."),
    ("Rig parts block cameras exactly like the body: in the ground coverage, the blind volume, the camera images and "
     "the installation check. Switch the platform's Occludes off, and the blocked share of the roof camera's image "
     "drops from 44.2 to 42.9 per cent.",
     "Rig parts block cameras exactly like the body: in the ground coverage, the blind volume, the camera images, "
     "and the installation check. Switch the platform's occludes off, and the blocked share of the roof camera's "
     "image drops from forty-four point two, to forty-two point nine per cent."),
    ("Camera light shows it from outside: what the roof camera lights on the bonnet and the ground, and the rack's "
     "shadow.",
     "Camera light shows it from outside: what the roof camera lights on the bonnet and the ground, and the rack's "
     "shadow."),
    ("Rig parts are saved with the project and exported: each a fixed link with its geometry in the ROS 2 URDF, a "
     "rig list in the extrinsics, and a table in the report.",
     "Rig parts are saved with the project, and exported: each a fixed link with its geometry, in the ROS two U R D F, "
     "a rig list in the extrinsics, and a table in the report."),
]


def scene(stage, r):
    import numpy as np
    from PyQt6.QtCore import QPoint, QPointF, QRectF, Qt

    from helpers_tour_a import click_sent, project_pt
    from vantage.ui.dialogs.export_dialog import ExportDialog

    w = stage.window()
    r.win = w
    st = w.store
    vp = stage.vp
    app = stage.app
    rows = w.bottom.coverage.rows
    w.ctx.set_overlay('camlight', False)
    stage.settle(3)
    stage.mark_result()

    def shown(k):
        return rows[k].v.text()

    def pump(n=4):
        for _ in range(n):
            app.processEvents()
    r.expect(shown('Seen by ≥ 2 cameras'), '53.0', 'seen by two or more before')

    # ---- 0: a roof rack
    r.cursor_on = True
    r.lower_third()
    r.say(0)
    stage.fly(r, vp, az=-50.0, el=26.0, dist=vp.cam.dist * 0.85, target=(1.0, 0.0, 1.0), dur=1.4)
    r.menu('Vehicle', 'Add rig part', 'Roof rack')
    stage.wait_analysis(r)
    r.expect(len(st.project.rig), 3, 'parts of the roof rack')
    r.crossfade(0.3)
    r.cursor_on = False
    stage.orbit(r, vp, 18, max(2.0, r.speech_left() + 0.2), d_el=3)
    r.finish_speech(0.1)

    # ---- 1: a sensor mast
    r.say(1)
    r.cursor_on = True
    r.menu('Vehicle', 'Add rig part', 'Sensor mast')
    stage.wait_analysis(r)
    r.expect(len(st.project.rig), 5, 'parts with the mast')
    head = max(st.project.rig, key=lambda q: q.z)
    r.expect(f'{head.z:.3f}', '2.092', "the mast head's centre height")
    r.crossfade(0.3)
    r.until_word(1, 'A sensor enclosure')
    r.peek_menu('Vehicle', 'Add rig part', 'Sensor enclosure', hold=max(0.8, r.speech_left() - 1.2))
    r.finish_speech(0.1)

    # ---- 2: Outliner, pick, Inspector
    r.say(2)
    r.callout(r.tree_item(w.outliner, 'Rig'), '', dur=2.0, side='right')
    r.hold(1.2)
    q = project_pt(r, vp, (head.x, head.y, head.z + 0.03))
    click_sent(r, vp, q, 0.7)
    stage.settle(2)
    rp = w.inspector.pages['rig']
    insp = r.wrect(w.docks['inspector'])
    r.until_word(2, 'edited in the inspector')
    r.zoom(QRectF(insp.x() - 60, insp.y(), insp.width() + 60, 340), 0.9)
    r.finish_speech(0.1)
    # ---- 3: occludes / mountable
    r.say(3)
    r.callout(lambda: r.wrect(rp.occ).united(r.wrect(rp.mnt)).adjusted(-120, 0, 60, 0), '', dur=max(2.4,
              r.speech_left() + 0.2), pad=3, dim=0.25)
    r.finish_speech(0.2)
    r.zoom(None, 0.8)

    # ---- 4: a camera on the mast head
    r.say(4)
    st.select_cams([])
    r.toolbar('tool_place')
    stage.mark_result()
    front = (head.x + head.length / 2, 0.0, head.z)
    q = project_pt(r, vp, front)
    r.hover(vp, q, 0.7)
    r.hold(0.5)
    click_sent(r, vp, q, 0.2)
    pump()
    cam = st.variant.cameras[-1]
    print('roof camera:', cam.tag, cam.pos(), cam.yaw, cam.pitch, flush=True)
    r.expect(cam.tag, 'F2', 'the roof camera')
    stage.wait_analysis(r)
    r.toolbar('tool_select')
    st.select_cams([cam.uid])
    stage.settle(2)
    page = w.inspector.pages['camera']
    r.expect(page.surface.currentText().startswith('Sensor mast · head'), True, 'mounted on the mast head')
    r.expect(shown('Seen by ≥ 2 cameras'), '59.4', 'seen by two or more with the roof camera')
    r.cursor_on = False
    r.figure('Seen by ≥ 2 cameras', float(shown('Seen by ≥ 2 cameras')), '%', 1, dur=max(3.0, r.speech_left() + 0.6),
             src=rows['Seen by ≥ 2 cameras'], start=53.0, note='camera on the mast')
    r.finish_speech(0.4)

    # ---- 5: through its lens: blockers in blue
    r.say(5)
    w.set_split('side')
    stage.settle(3)
    left, right = w.viewarea.panes[0], w.viewarea.panes[1]
    right.set_kind('through')
    right.cam.uid = cam.uid
    vl = left.vp3d
    vl.cam.target = np.array([1.4, 0.0, 1.3])
    vl.cam.dist, vl.cam.az, vl.cam.el = 4.6, -40.0, 24.0
    vl.update()
    stage.settle(3)
    pump()
    r.crossfade(0.4)
    r.hold(0.8)
    r.zoom(right, 1.0, margin=6)
    r.finish_speech(0.2)

    # ---- 6: Occludes off on the platform
    body_on = w.camdock.vals['Body in view'].text()
    print('F2 body in view, platform occluding:', body_on, flush=True)
    r.expect(body_on, '44.2 %', "F2's blocked share with the platform")
    r.say(6)
    r.zoom(None, 0.8)
    r.cursor_on = True
    plat = next(p for p in st.project.rig if p.name.endswith('platform'))
    st.select_obj(('rig', plat.uid))
    stage.settle(2)
    r.hold(max(0.5, r.at_word(6, 'Switch the platform') - r.now()))
    stage.mark_result()
    r.click(rp.occ, None, 0.7)
    pump()
    if plat.uid and next(p for p in st.project.rig if p.uid == plat.uid).occludes:
        def off(pr):
            for q_ in pr.rig:
                if q_.uid == plat.uid:
                    q_.occludes = False
        st.edit('rig', off, 'Platform no longer blocks cameras')
    stage.wait_analysis(r)
    st.select_cams([cam.uid])
    stage.settle(2)
    body_off = w.camdock.vals['Body in view'].text()
    print('F2 body in view, platform not occluding:', body_off, flush=True)
    r.expect(body_off, '42.9 %', "F2's blocked share without the platform")
    r.figure('Body in view · F2', 42.9, '%', 1, dur=max(3.0, r.speech_left() + 0.6),
             src=w.camdock.vals['Body in view'], start=44.2, note='platform does not occlude')
    r.finish_speech(0.2)
    r.zoom(None, 0.7)

    # ---- 7: camera light from outside
    r.say(7)
    st.undo.undo()                                          # the platform blocks again
    stage.wait_analysis(r)
    w.set_split('single')
    stage.settle(2)
    vp = w.viewarea.panes[0].vp3d
    vp.cam.target = np.array([1.3, 0.0, 0.8])
    vp.cam.dist, vp.cam.az, vp.cam.el = vp.cam.dist, -40.0, 28.0
    w.ctx.set_overlay('camlight', True)
    w.ctx.set_overlay('frustums', False)
    stage.settle(3)
    r.crossfade(0.5)
    r.cursor_on = False
    stage.orbit(r, vp, 70, max(4.0, r.speech_left() + 0.6), d_el=6)
    r.finish_speech(0.2)

    # ---- 8: exported: the URDF
    r.say(8)
    r.cursor_on = True
    r.key_combo('Ctrl+E', dur=1.2)
    d = ExportDialog(w.ctx, w, preselect='ros2')
    stage.extra.append(d)
    stage.tilde(d.out)
    d.show()
    pump(10)
    d.move(w.mapToGlobal(QPoint((w.width() - d.width()) // 2, (w.height() - d.height()) // 2)))
    pump()
    r.crossfade(0.4)
    r.wait_for(lambda: '<!-- rig:' in d.preview.toPlainText(), timeout=60)
    txt = d.preview.toPlainText()
    links = [ln for ln in txt.splitlines() if '<link name="rig_' in ln]
    print('URDF rig links:', links, flush=True)
    r.expect(len(links), 5, 'rig links in the URDF')
    first = next(i for i, ln in enumerate(txt.splitlines()) if '<!-- rig:' in ln)
    sb = d.preview.verticalScrollBar()
    r.hold(0.6)
    n = round(1.0 * stage.fps)
    target = min(sb.maximum(), first - 1)
    v0 = sb.value()
    for k in range(1, n + 1):
        t = k / n
        sb.setValue(round(v0 + (target - v0) * t * t * (3 - 2 * t)))
        r.tick()
    pv = r.wrect(d.preview)
    r.zoom(QRectF(pv.x(), pv.y(), pv.width() * 0.8, pv.height() * 0.6), 0.9)
    r.hold(max(1.6, r.speech_left() - 1.0))
    d.hide()
    r.zoom(None, 0.0)
    r.crossfade(0.4)
    r.cursor_on = False
    r.finish_speech(1.0)
