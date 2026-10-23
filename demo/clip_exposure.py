"""Website clip: Analysis ▸ Sun glare and dirt on the sample's data rig. The study runs; the table of sun-in-image
shares and dirt scores; an east-west route typed as headings raises the front, rear and quarter cameras' glare; a
quarter camera's month × hour heatmap and its sky; the dirt map of the body."""
from __future__ import annotations

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtTest import QTest


def _type(r, edit, text, per_key=2):
    """Click into a line edit, select what is there and type text visibly, then Return."""
    r.click(edit, QPointF(edit.width() * 0.6, edit.height() / 2))
    QTest.keyClick(edit, Qt.Key.Key_A, Qt.KeyboardModifier.ControlModifier)
    r.tick(2)
    for ch in text:
        QTest.keyClicks(edit, ch)
        r.tick(per_key)
    QTest.keyClick(edit, Qt.Key.Key_Return)
    r.tick(2)


def clip(stage, r):
    w = stage._rig_window()
    r.win = w
    r.chapter('Sun and dirt', 'When the sun shines into each camera, and which lenses get dirty')
    r.menu('Analysis', 'Sun glare and dirt')
    pg = w.page('exposure')
    ctl = pg.ctl
    r.wait_for(lambda: ctl.result is not None and not ctl.busy, timeout=60)
    r.hold(1.2)
    # the table: sun in the image, near it, low ahead, when, on which headings; the dirt score and advice
    t = r.wrect(pg.table)
    r.zoom(QRectF(t.x(), t.y() - 22, t.width() * 0.62, t.height() + 22), 0.9, margin=12)
    r.hold(2.0)
    r.zoom(None, 0.6)
    # an east-west motorway, typed as headings: the study runs again by itself
    before = ctl.result
    _type(r, pg.head_text, '90:1, 270:1')
    r.wait_for(lambda: ctl.result is not before and not ctl.busy and not ctl.stale, timeout=60)
    r.hold(0.5)
    left = r.wrect(pg.head_text)
    t = r.wrect(pg.table)
    r.zoom(QRectF(left.x(), t.y() - 22, t.x() + t.width() * 0.45 - left.x(), left.bottom() - t.y() + 40), 0.9,
           margin=10)
    r.hold(2.2)
    r.zoom(None, 0.6)
    # the left quarter camera: its month x hour heatmap and its sky (field, body, where the sun is)
    row = next(k for k, i in enumerate(ctl.result.items) if i.tag == 'QL')
    r.click(pg.table.viewport(), QPointF(pg.table.visualItemRect(pg.table.item(row, 0)).center()))
    r.move(r.wpos(pg.sky, QPointF(pg.sky.width() - 14, pg.sky.height() * 0.3)), 0.5)     # off the figures
    r.hold(0.2)
    h, s = r.wrect(pg.heat), r.wrect(pg.sky)
    r.zoom(QRectF(h.x(), h.y() - 4, s.right() - h.x(), s.bottom() - h.y() + 4), 0.9, margin=8)
    r.hold(2.6)
    r.zoom(None, 0.6)
    # the dirt map: every face of the body, the sensors in their class colour
    r.zoom(r.wrect(pg.dirt), 0.9, margin=8)
    r.hold(2.2)
    r.zoom(None, 0.6)
    r.hold(0.4)
