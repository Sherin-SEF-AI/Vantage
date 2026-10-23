"""Website clip 'dataset': Export ▸ Sample dataset on the data rig with GNSS and an INS. The preview of the synthetic
drive at two instants (the six camera images, the lidar top view with the ground-truth boxes), the drive shortened
to 2 s on the form, then Generate into a temp folder: the job runs on camera and the summary with the devkit snippet
appears. The written dataset is removed afterwards."""
from __future__ import annotations

import shutil
import tempfile

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QStyle, QStyleOptionSlider


def _slider_x(sl, v):
    """x (widget coordinates) of a QSlider's handle centre at value v."""
    opt = QStyleOptionSlider()
    sl.initStyleOption(opt)
    st = sl.style()
    groove = st.subControlRect(QStyle.ComplexControl.CC_Slider, opt, QStyle.SubControl.SC_SliderGroove, sl)
    handle = st.subControlRect(QStyle.ComplexControl.CC_Slider, opt, QStyle.SubControl.SC_SliderHandle, sl)
    span = groove.width() - handle.width()
    return groove.x() + handle.width() / 2 + QStyle.sliderPositionFromValue(sl.minimum(), sl.maximum(), v, span)


def _scrub_to(r, pg, value):
    """Drag the preview's time slider to value (tenths of a second); the preview renders that instant."""
    sl = pg.slider
    y = sl.height() / 2
    r.drag(sl, QPointF(_slider_x(sl, sl.value()), y), QPointF(_slider_x(sl, value), y), 1.0)
    if sl.value() != value:                    # a style that would not follow the drag exactly
        sl.setValue(value)
        pg._slider_released()
    r.tick(2)
    r.wait_for(lambda: pg.preview_data is not None and not pg._pbusy and not pg.timer.isActive()
               and abs(pg.preview_data['t'] - value / 10) < 1e-6, timeout=60)


def clip(stage, r):
    def setup(p):
        p.analysis.dataset = {'image_scale': 0.25}      # small images, so the generation finishes on camera
    w = stage._nav_window(setup)
    r.win = w
    out = tempfile.mkdtemp(prefix='vantage-dataset-')
    try:
        r.chapter('Sample dataset', 'A synthetic drive recorded by exactly this rig')
        r.menu('Export', 'Sample dataset')
        pg = w.page('dataset')
        r.wait_for(lambda: pg.preview_data is not None and not pg._pbusy, timeout=60)
        r.hold(0.6)
        _scrub_to(r, pg, 30)
        r.hold(0.3)
        top = r.wrect(pg.top)
        r.zoom(QRectF(top.x(), top.y(), top.width(), top.height() * 0.95), 0.8, margin=20)
        r.hold(1.6)
        r.zoom(None, 0.6)
        _scrub_to(r, pg, 65)                   # into the bend
        r.hold(0.6)
        # a shorter drive on the form: the estimate follows
        f, _ = pg.fields['duration_s']
        r.click(f, dur=0.6)
        f.setFocus()
        f.lineEdit().selectAll()
        r.tick(3)
        QTest.keyClicks(f.lineEdit(), '2')
        r.tick(6)
        QTest.keyClick(f.lineEdit(), Qt.Key.Key_Return)
        r.hold(0.6)
        # generate into the temp folder (the folder dialog is not filmed): progress, then the summary
        r.click(pg.b_generate, real=False)
        pg.generate(out)
        r.move(r.wpos(pg.top), 0.6, wait=False)          # the pointer off the progress line
        r.wait_for(lambda: not pg.busy, timeout=240, speed=2)
        r.hold(0.4)
        box = r.wrect(pg.out)
        r.zoom(QRectF(box.x(), box.y() - 26, 860, 150), 0.8, margin=12)
        r.hold(2.4)
        r.zoom(None, 0.5)
        r.hold(0.3)
    finally:
        shutil.rmtree(out, ignore_errors=True)
