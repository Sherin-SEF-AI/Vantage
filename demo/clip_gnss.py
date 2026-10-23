"""Website clip 'gnss': GNSS antennas and a dual-antenna INS added to the data rig from the Vehicle menu, Analysis ▸
GNSS and IMU with their open sky, the heading baseline and the lever arms, then one antenna moved beside the lidar's
mast: its sky plot shows the wedge the mast and its head take, and the findings flag the lost sky, multipath and the
shorter baseline. Called by demo/clips.py (--clips gnss)."""
from __future__ import annotations

from PyQt6.QtCore import QRectF


def _mast_rig(p):
    """The data rig with its lidar raised on a sensor mast (as Vehicle ▸ Add rig part ▸ Sensor mast and Add lidar put
    them): the roof stays free for the antennas."""
    from vantage.core import rig as rigm
    from vantage.core import sensors
    p.rig = rigm.preset('sensor_mast', p.vehicle.mesh)
    lid = next(s for s in p.variant.sensors if s.kind == 'lidar')
    pose = sensors.default_pose('lidar', p.sensor_models[lid.model_id], p.vehicle.mesh, p.rig, [])
    for k, v in pose.items():
        setattr(lid, k, v)


def clip(stage, r):
    from vantage.core import sensors
    from vantage.ui.sensorview import add_sensor, set_props
    w = stage._rig_window(_mast_rig)
    r.win = w
    st = w.store
    r.chapter('GNSS and IMU', 'Open sky, the heading baseline and the INS lever arms')
    r.hold(0.4)
    r.menu('Vehicle', 'Add GNSS antenna', 'Survey GNSS antenna')       # near the rear of the roof, clear of the mast
    stage.settle(2)
    r.hold(0.3)
    add_sensor(w.ctx, 'gnss')                                           # the second near the front edge
    stage.settle(2)
    r.hold(0.4)
    add_sensor(w.ctx, 'imu')                                            # a dual-antenna GNSS/INS above the rear axle
    stage.settle(2)
    r.hold(0.4)
    r.menu('Analysis', 'GNSS and IMU')
    pg = w.page('gnss')
    r.wait_for(lambda: pg.result is not None and not pg.busy, timeout=60)
    r.hold(1.6)
    # the baseline and the lever arms the INS needs
    right = r.wrect(pg.base_text)
    lev = r.wrect(pg.levers)
    r.zoom(QRectF(right.x(), right.y() - 24, max(right.width(), lev.width()), lev.bottom() - right.y() + 24), 0.9)
    r.hold(2.0)
    r.zoom(None, 0.6)
    # GN1 moved onto the mast's foot, under the lidar: the sky plot and the findings change at once
    gn1 = next(s for s in st.variant.sensors if s.kind == 'gnss')
    r.click(pg.plots[0])
    r.hold(0.5)
    mast = min(st.project.rig, key=lambda part: part.z)
    x = round(mast.x - 0.11, 3)
    m = st.project.sensor_model_for(gn1)
    z = round(sensors.surface_top(x, 0.0, st.project.vehicle.mesh, st.project.rig) + m.optical_z_mm / 1000.0, 3)
    before = pg.result
    set_props(w.ctx, gn1.uid, f'Moved {gn1.tag} beside the mast', x=x, y=0.0, z=z)
    r.wait_for(lambda: pg.result is not before and not pg.busy, timeout=30)
    r.hold(1.2)
    r.cursor_on = False
    r.zoom(r.wrect(pg.plots[0]), 0.9, margin=16)
    r.hold(2.4)
    first, last = pg.findings[0], pg.findings[min(3, len(pg.findings) - 1)]
    a, b = r.wrect(first), r.wrect(last)
    r.zoom(QRectF(a.x() - 10, a.y() - 30, a.width() + 20, b.bottom() - a.y() + 40), 0.9)
    r.hold(2.4)
    r.zoom(None, 0.6)
    r.hold(0.5)
