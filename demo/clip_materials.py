"""Website clip 'materials': what the sample data rig's radars look through, and a camera behind the windscreen.

Analysis ▸ Materials, glass and wipers…: the radar table (RD1 in front of the bumper loses nothing; the corner
radars RD2–RD5 sit 12 mm behind the painted fascia: 2.67 dB there and back, a car at 86 m instead of 100 m), RD2's
loss over its field, the paint switched to metallic (3.9 dB, 80 m) and to a repaired metallic bumper (8.3 dB,
62 m) and back; then the camera FW, added behind the windscreen for this clip (the sample has none): the light it
loses, its apparent centre shift and the dashboard reflected into its image, the glare shield ticked, and the
windscreen seen from inside with the typical wipers' sweep and the camera's footprint. The project is never saved.
"""
from __future__ import annotations

from PyQt6.QtCore import QPointF, QRectF


def click_tab(r, tabs, title):
    bar = tabs.tabBar()
    i = next(k for k in range(tabs.count()) if tabs.tabText(k) == title)
    r.click(bar, QPointF(bar.tabRect(i).center()), 0.5, real=False)
    tabs.setCurrentIndex(i)
    r.hold(0.2)


def windscreen_camera(p):
    """A camera behind the windscreen (the sample has none): a STURDeCAM21 looking ahead from below the roof line, as
    the materials tests place it."""
    from vantage.core.project import Camera
    p.variant.cameras.append(Camera(tag='FW', name='Windscreen', x=1.80, y=0.0, z=1.36, yaw=0.0, pitch=5.0, slot=6,
                                    port=7, module_id='sturdecam21', lens_id='sturdecam21_std'))


def clip(stage, r):
    w = stage._rig_window(windscreen_camera)
    r.win = w
    r.chapter('Materials, glass and wipers', 'What the radars look through, and the cameras behind glass')
    r.menu('Analysis', 'Materials, glass and wipers')
    pg = w.page('materials')
    ctl = pg.ctl

    def ready(paint):
        res = ctl.result
        return res is not None and not ctl.busy and not ctl.stale and res.settings.paint == paint

    r.wait_for(lambda: ready('solid'), timeout=90)
    if not ready('solid'):
        raise RuntimeError('the materials study never finished')
    r.hold(0.5)
    t = r.wrect(pg.radars)
    r.zoom(QRectF(t.x(), t.y() - 26, t.width(), min(t.height(), 230) + 26), 0.8, margin=10)
    r.hold(2.0)
    r.zoom(None, 0.6)
    # RD2: the loss over its field
    row = next(i for i in range(pg.radars.rowCount()) if pg.radars.item(i, 0).text().startswith('RD2'))
    vp = pg.radars.viewport()
    rect = pg.radars.visualRect(pg.radars.model().index(row, 0))
    r.click(vp, QPointF(rect.center()), 0.6, real=False)
    pg.radars.selectRow(row)
    stage.settle(1)
    r.hold(0.8)
    # metallic paint, then a repaired (repainted) metallic bumper
    for key in ('metallic', 'repaired'):
        r.click(pg.paint.buttons[key])
        r.wait_for(lambda k=key: ready(k), timeout=60)
        r.hold(0.3)
        r.zoom(QRectF(t.x(), t.y() - 26, t.width(), min(t.height(), 230) + 26), 0.6, margin=10)
        r.hold(1.3)
        r.zoom(None, 0.5)
    r.click(pg.paint.buttons['solid'])
    r.wait_for(lambda: ready('solid'), timeout=60)
    # the camera behind the windscreen: light, centre shift, dashboard glare; the wipers' sweep over its view
    click_tab(r, pg.tabs, 'Glass and wipers')
    stage.settle(1)
    r.hold(0.3)
    c = r.wrect(pg.cams)
    r.zoom(QRectF(c.x(), c.y() - 26, c.width(), min(c.height(), 80) + 26), 0.7, margin=10)
    r.hold(1.8)
    r.zoom(r.wrect(pg.glass), 0.7, margin=10)
    r.hold(1.6)
    r.zoom(None, 0.6)
    r.cursor_on = False
    r.hold(0.4)
