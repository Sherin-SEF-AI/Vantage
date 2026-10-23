"""Website clip `fleet`: the data rig built on five vehicles and measured four times over six months.

The fleet is prepared before the window opens, the way tests/test_fleet.py builds it: each vehicle's sensors sit a few
millimetres and a tenth of a degree off the plan (its own build offsets) with a little measurement noise; on V2 the
front camera pitches down 0.2° a month, and on V4 the left mirror camera is mounted 2° yawed. The records are
simulated, and labelled so ('simulated check N'). The clip opens Calibrate ▸ Fleet, shows the matrix, clicks the
outlier, shows the findings, then clicks the drifting camera for its drift chart; the poster frame (72 %) falls on the
drift chart.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
from PyQt6.QtCore import QPointF

START = dt.date(2026, 3, 2)
DAYS = (0, 60, 120, 180)


def build_fleet(p, seed=7):
    """Five vehicles × four simulated checks of the active variant (the data rig)."""
    from vantage.core import fleet as FL
    rng = np.random.default_rng(seed)
    plan = FL.plan_sensors(p, p.variant)
    fleet = {}
    for i in range(5):
        v = FL.add_vehicle(fleet, f'V{i + 1}', vin=f'DEMO-{i + 1:04d}', note='simulated for the demo')
        off = {ps['tag']: np.r_[rng.uniform(-3, 3, 3), rng.uniform(-0.15, 0.15, 3)] for ps in plan}
        for j, day in enumerate(DAYS):
            poses = {}
            for ps in plan:
                d = off[ps['tag']] + np.r_[rng.normal(0, 0.3, 3), rng.normal(0, 0.01, 3)]
                if (v['name'], ps['tag']) == ('V2', 'F'):
                    d[4] += 0.2 / 30.0 * day                  # pitches down 0.2° a month
                if (v['name'], ps['tag']) == ('V4', 'ML'):
                    d[3] += 2.0                               # mounted 2° yawed
                q = ps['pose']
                poses[ps['tag']] = {'x': q['x'] + d[0] / 1000, 'y': q['y'] + d[1] / 1000, 'z': q['z'] + d[2] / 1000,
                                    'yaw': q['yaw'] + d[3], 'pitch': q['pitch'] + d[4], 'roll': q['roll'] + d[5]}
            FL.add_record(fleet, v['id'], {'kind': 'manual', 'date': (START + dt.timedelta(days=day)).isoformat(),
                                           'source': f'simulated check {j + 1}', 'poses': poses})
    p.fleet = fleet


def clip(stage, r):
    w = stage._rig_window(build_fleet)
    r.win = w
    r.chapter('Fleet', 'The same rig on every vehicle, measured over months')
    r.menu('Calibrate', 'Fleet')
    pg = w.page('fleet')
    stage.settle(2)
    r.wait_for(lambda: pg.result is not None and not pg.busy, timeout=30)
    r.hold(1.5)
    res = pg.result
    vid = {v['name']: v['id'] for v in res.vehicles}
    # the matrix: every vehicle × sensor, coloured by its latest offset against the tolerance
    r.zoom(r.wrect(pg.matrix_scroll), 0.9, margin=40)
    r.hold(2.0)
    # the outlier vehicle: V4's left mirror camera, 2° off the rest of the fleet
    cell = pg.matrix.cell_rect(vid['V4'], 'ML')
    r.click(pg.matrix, cell.center(), 0.7)
    r.move(r.wpos(pg.matrix, QPointF(cell.right() + cell.width() * 0.6, cell.bottom() + 14)), 0.4)   # off the figure
    r.hold(1.0)
    r.zoom(None, 0.6)
    # the findings
    finds = pg.find_host.parentWidget().parentWidget().parentWidget()
    r.zoom(r.wrect(finds), 0.9, margin=12)
    r.hold(2.6)
    r.zoom(None, 0.6)
    # the drifting camera: its pitch climbs through the tolerance band, the trend carried on over the drift window
    r.click(pg.matrix, pg.matrix.cell_rect(vid['V2'], 'F').center(), 0.7)
    r.move(r.wpos(pg.chart, QPointF(pg.chart.width() * 0.62, pg.chart.height() * 0.9)), 0.5)   # down to its chart
    r.hold(0.2)
    r.zoom(r.wrect(pg.chart).united(r.wrect(pg.detail)), 0.9, margin=16)
    r.hold(3.4)
    r.zoom(None, 0.7)
    r.hold(2.8)
