"""Website clip: Calibrate ▸ Lidar–camera calibration on the sample's data rig (front camera F, roof lidar L1).

The pose plan, a mounting error typed in for the synthetic capture, Capture and solve, the lidar's points on the
camera image through the plan and through the solve, and the figures."""
from __future__ import annotations

from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtTest import QTest


def _type(r, field, text):
    """Click a number field, select what is there and type text, a key at a time."""
    le = field.lineEdit()
    r.click(le, None, 0.5)
    QTest.keyClick(le, Qt.Key.Key_A, Qt.KeyboardModifier.ControlModifier)
    for ch in text:
        QTest.keyClicks(le, ch)
        r.tick(2)
    QTest.keyClick(le, Qt.Key.Key_Return)
    r.hold(0.2)


def clip(stage, r):
    def setup(p):
        p.analysis.lidar_calib = {'camera': 'F', 'lidar': 'L1', 'seed': 4}
    w = stage._rig_window(setup)
    r.win = w
    r.chapter('Calibration', 'Where the lidar really sits, from a board both sensors see')
    r.menu('Calibrate', 'Lidar–camera calibration')
    pg = w.page('lidar_calib')
    r.wait_for(lambda: pg.plan is not None and bool(pg.plan.get('poses')), timeout=60)
    r.hold(1.3)
    # a mounting error for the synthetic capture: 15 mm forward, 8 mm up, 0.6° of yaw
    grid = r.wrect(pg.fields['err_x_mm'])
    r.zoom(QRectF(0, grid.y() - 80, 560, 300), 0.7)
    for key, text in (('err_x_mm', '15'), ('err_z_mm', '8'), ('err_yaw_deg', '0.6')):
        _type(r, pg.fields[key], text)
    r.hold(0.4)
    r.zoom(None, 0.6)
    r.click(pg.b_run)
    r.wait_for(lambda: pg.result is not None and not pg.busy, timeout=240, speed=4)
    r.hold(0.6)
    # the visual proof, close up on the board: the lidar's points through the plan, then through the solve
    ov = r.wrect(pg.overlay)
    side = min(ov.width() - 12, ov.height() - 48)          # the square the image is drawn in
    top = ov.y() - 34                                       # the Plan / Solved buttons and the caption stay in view
    h = 34 + 22 + side * 0.86
    r.zoom(QRectF(ov.center().x() - h * 8 / 9, top, h * 16 / 9, h), 0.8, margin=0)
    r.hold(0.8)
    r.click(pg.mode.buttons['plan'])
    r.hold(1.8)
    r.click(pg.mode.buttons['solved'])
    r.hold(2.6)
    r.zoom(None, 0.6)
    # the figures: Δ against the plan, the uncertainty, the verdict
    t0, t1 = r.wrect(pg.tiles['verdict']), r.wrect(pg.tiles['sigma'])
    r.zoom(QRectF(t0.x(), t0.y(), t1.right() - t0.x(), t1.bottom() - t0.y() + 150), 0.8)
    r.hold(2.1)
    r.zoom(None, 0.6)
    r.hold(0.3)
