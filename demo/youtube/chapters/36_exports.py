"""36 Exports and dataset calibration: Export ▸ Export (Ctrl+E) on the sample (6-cam · v3). The dialog is shown with
show() (the app opens it with exec(), which would block the recorder), its paths from ~. The formats by group with the
live preview: the extrinsics JSON in another frame and in millimetres, the ROS 2 URDF, then the Datasets group
(nuScenes, KITTI, ROS camera_info, Kalibr) ticked and previewed, the other groups, and a real export into the
project's folder in the throw-away home ('Open the folder when done' switched off first, so nothing opens).

Figures: the number of files the Export button names, read from the dialog (r.expect against core.exporters'
file names); the accuracy of the dataset formats is what the README and docs state from the repository's tests
(nuScenes devkit to 0.0001 px, the KITTI devkit's projection within 0.001 px, Kalibr's model error stated per camera
in the file), narrated as checked by the project's tests.
"""

META = {
    'id': 'exports',
    'part': 5,
    'title': 'Exports and dataset calibration',
    'promise': 'The rig’s calibration in every format the pipeline reads, previewed before it is written.',
    'menu': 'Export ▸ Export  ·  Ctrl+E',
}

LINES = [
    ("Everything the design decides goes out as files. Export lists every format by group, with a live preview of "
     "each file before it is written.",
     "Everything the design decides goes out as files. Export lists every format by group, with a live preview of "
     "each file before it is written."),
    ("Documents first: the mount sheet for the workshop, the report as HTML or PDF, and each study's own results.",
     "Documents first: the mount sheet for the workshop, the report as H T M L or P D F, and each study's own "
     "results."),
    ("Extrinsics of every camera, lidar, radar, antenna and IMU, as JSON, YAML or CSV, in the frame, units and angles "
     "set here. Change them, and the preview follows.",
     "Extrinsics of every camera, lidar, radar, antenna, and I M U, as Jason, yamel, or C S V, in the frame, units, "
     "and angles set here. Change them, and the preview follows."),
    ("For calibration and robotics: an OpenCV fisheye file per camera, a ROS 2 URDF with a static TF launch file, and "
     "the Jetson's camera configuration.",
     "For calibration and robotics: an open C V fisheye file per camera, a ross two U R D F with a static T F launch "
     "file, and the Jetson's camera configuration."),
    ("The Datasets group writes the calibration the way training datasets lay it out: nuScenes, KITTI, ROS "
     "camera_info and Kalibr.",
     "The datasets group writes the calibration the way training datasets lay it out: new scenes, kitty, ross camera "
     "info, and calibre."),
    ("Each format is checked by the project's tests against the tool that reads it: the nuScenes devkit, the KITTI "
     "devkit's projection, ROS's own parsers, and Kalibr, which gets the camera model that reproduces each lens.",
     "Each format is checked by the project's tests against the tool that reads it: the new scenes dev kit, the kitty "
     "dev kit's projection, ross's own parsers, and calibre, which gets the camera model that reproduces each lens."),
    ("Then surround view, with the bird's-eye remap tables, a reference stitcher, the bowl and the guidelines; "
     "images of the views; and the mount points for CAD, as DXF and STEP.",
     "Then surround view, with the bird's eye remap tables, a reference stitcher, the bowl, and the guidelines; "
     "images of the views; and the mount points for CAD, as D X F and step."),
    ("Export writes every ticked format into the project's folder, and says where.",
     "Export writes every ticked format into the project's folder, and says where."),
]


def scene(stage, r):
    import os
    from PyQt6.QtCore import QRectF
    from vantage.core import exporters as ex
    from vantage.ui.dialogs.export_dialog import ExportDialog
    import helpers_tourc as H

    w = stage.window()
    r.win = w
    st = w.store
    stage.tilde(w.msg_label)
    vp = stage.vp

    # 0 - open the dialog
    r.lower_third()
    r.say(0)
    r.cursor_on = False
    stage.orbit(r, vp, 18, 3.0, d_el=4)
    r.cursor_on = True
    r.until_word(0, 'Export lists')
    r.peek_menu('Export', 'Export', hold=0.6)
    r.key_combo('Ctrl+E', dur=1.4)
    d = ExportDialog(w.ctx, w)
    stage.extra.append(d)
    stage.tilde(d.out, d.path_lbl)                # shown from ~ ...
    _opts = d.opts

    def opts(with_images=False):                  # ... and written to the real folder in the throw-away home
        o = _opts(with_images)
        o.out_dir = os.path.expanduser(o.out_dir)
        return o
    d.opts = opts
    d.open_after.setChecked(False)               # nothing opens on the desktop when it exports
    d.show()
    stage.settle(2)
    stage.film_centred(r, d)
    r.crossfade(0.5)
    r.finish_speech(0.2)

    def row(fid):
        return d.rows[fid]

    def group_rect(first, last):
        return lambda: H.union(r.wrect(row(first)), r.wrect(row(last)))

    def scroll_to(fid):
        d.format_scroll.ensureWidgetVisible(row(fid), 0, 40)
        r.tick(2)

    # 1 - documents
    r.say(1)
    lst = r.wrect(d.format_scroll)
    r.zoom(QRectF(lst.x() - 10, lst.y() - 10, lst.width() + 520, lst.height() + 20), 1.0)
    docs = [f.id for f in ex.FORMATS if f.group == 'Documents']
    r.callout(group_rect(docs[0], docs[2]), dur=2.6, pad=3)
    r.until_word(1, "each study's own")
    r.callout(group_rect(docs[3], docs[-1]), dur=max(2.0, r.speech_left()), pad=3)
    r.finish_speech(0.2)
    r.zoom(None, 0.8)

    # 2 - extrinsics: frame and units change the preview
    r.say(2)
    r.click(row('extrinsics_json'), None, 0.6)
    d.select('extrinsics_json')
    stage.settle(1)
    pv = r.wrect(d.preview)
    r.zoom(QRectF(pv.x() - 10, pv.y() - 40, pv.width() + 20, pv.height() * 0.6), 1.0)
    r.until_word(2, 'Change them')
    r.zoom(None, 0.6)
    r.choose(d.frame, d.frame.findData('front_bumper'))
    stage.settle(1)
    r.click(d.units.buttons['mm'], dur=0.5)
    stage.settle(1)
    r.zoom(QRectF(pv.x() - 10, pv.y() - 40, pv.width() + 20, pv.height() * 0.6), 0.7)
    r.finish_speech(0.2)

    # 3 - calibration and robotics: the URDF
    r.say(3)
    r.zoom(None, 0.6)
    r.choose(d.frame, d.frame.findData('rear_axle'))
    r.click(d.units.buttons['m'], dur=0.4)
    cal = [f.id for f in ex.FORMATS if f.group == 'Calibration and robotics']
    r.callout(group_rect(cal[0], cal[-1]), dur=2.0, pad=3)
    r.click(row('ros2'), None, 0.6)
    d.select('ros2')
    stage.settle(1)
    r.zoom(QRectF(pv.x() - 10, pv.y() - 40, pv.width() + 20, pv.height() * 0.6), 0.9)
    r.finish_speech(0.3)

    # 4 - the datasets group: tick and preview each
    r.say(4)
    scroll_to('kalibr_camchain')
    r.zoom(QRectF(lst.x() - 10, lst.y() - 10, lst.width() + 520, lst.height() + 20), 0.9)
    dsets = ('nuscenes_calib', 'kitti_calib', 'ros_camera_info', 'kalibr_camchain')
    for fid in dsets:
        r.click(row(fid).cb, None, 0.5)
        d.select(fid)
        stage.settle(1)
        r.hold(0.7)
    r.finish_speech(0.2)

    # 5 - checked against the tools; Kalibr's model per lens
    r.say(5)
    r.zoom(QRectF(pv.x() - 10, pv.y() - 40, pv.width() + 20, pv.height() * 0.6), 1.0)
    r.hold(max(1.5, r.speech_left() - 3.2))
    d.select('nuscenes_calib')
    stage.settle(1)
    r.finish_speech(0.2)
    r.zoom(None, 0.8)

    # 6 - surround view, images, mount points
    r.say(6)
    scroll_to('step')
    r.zoom(QRectF(lst.x() - 10, lst.y() - 10, lst.width() + 520, lst.height() + 20), 1.0)
    sv = [f.id for f in ex.FORMATS if f.group == 'Surround view']
    r.callout(group_rect(sv[0], sv[-1]), dur=2.6, pad=3)
    r.until_word(6, 'and the mount points')
    r.callout(group_rect('dxf', 'step'), dur=max(2.0, r.speech_left()), pad=3)
    r.finish_speech(0.2)
    r.zoom(None, 0.8)

    # 7 - export
    ids = d.checked()
    n = sum(len(ex._names(ex.format_by_id(fid), st.project, st.variant)) for fid in ids)
    r.expect(d.go.text(), f'Export {n} files', 'the export button')
    print('exporting', ids, n, flush=True)
    r.say(7)
    r.cursor_on = True
    r.click(d.go, real=False)
    d.go.click()
    stage.settle(3)
    r.expect(d.isVisible(), False, 'the dialog closed after the export')
    stage.film_window(r, w)
    r.crossfade(0.5)
    print('status bar:', w.msg_label.text(), flush=True)
    r.expect(w.msg_label.text().startswith('Exported ') and '~/' in w.msg_label.text(), True, 'the export message')
    mr = r.wrect(w.msg_label)
    r.zoom(QRectF(mr.x() - 40, mr.y() - 200, max(760, mr.width() + 80), 260), 0.9)
    r.callout(w.msg_label, dur=max(2.0, r.speech_left() + 0.4), pad=4, dim=0.2, side='above')
    r.cursor_on = False
    r.finish_speech(0.6)
    r.zoom(None, 0.8)
    r.hold(0.4)
