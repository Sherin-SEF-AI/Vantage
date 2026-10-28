"""Website clip 'critcarla': Simulate ▸ Critical targets in CARLA… on drives recorded in CARLA 0.9.16.

The sample's critical targets (its ten presets and a cyclist 6 m ahead, the overhead traffic light as a stand-in prop)
on the data rig, each stood in CARLA as the actor nearest it: three child walkers, an adult, a Mini, a bicycle with
its rider, a street-sign board. Nine real drives (three scenes at three spawn points, ground truth on, half-size
images), recorded before with the same rig and targets, are opened with Open drives… and compared without a server:
30 of 30 camera views agree with the study, the pixel figure is a median 2.6 % above the prediction at the actor's
size, and the crops show each camera's view with the actor outlined. Traffic lights, the kerb and the lane marking are
listed as not spawnable.

The drives are a working copy whose sensor files link to the recording; the clip links that folder into the
throw-away home as ~/carla-drives/critical-targets (the folder VANTAGE_CLIP_CRITICAL names) and removes the link at
the end. No CARLA server is started or asked: the CARLA settings name port 2300. The project is never saved.
"""
from __future__ import annotations

import os

from PyQt6.QtCore import QPointF, QRectF



def setup(p):
    """The targets, ids and settings the drives were recorded with (their run.json names actors by target id)."""
    from vantage.core import critical as CR, simrig as SR, critical_carla as CC
    v = next(v for v in p.variants if v.sensors)
    p.active_uid = v.uid
    targets = [CR.preset(p, r[0]) for r in CR.PRESET_PLACES]
    box = CR.vehicle_box(p)
    targets.append(CR.clean_target({'kind': 'object', 'name': 'Cyclist 6 m ahead, 2.5 m right',
                                    'x': round(box['front_x'] + 6.0, 3), 'y': -2.5, 'height': 1.7, 'width': 0.6,
                                    'min_px': 24.0, 'min_share': 0.5}))
    for i, t in enumerate(targets):
        t['uid'] = f'tgt{i:02d}'
    p.analysis.critical = CR.CriticalSettings.from_dict({'targets': targets}).to_dict()
    for c in v.cameras:
        c.uid = f'cam_{c.tag}'
    for s in v.sensors:
        s.uid = f'sen_{s.tag}'
    v.uid = 'data_rig'
    p.active_uid = v.uid
    tl = next(t for t in targets if t['name'].startswith('Overhead traffic light'))
    p.analysis.sim = SR.SimSettings.from_dict({'port': 2300, 'tm_port': 8400, 'town': 'Town10HD'}).to_dict()
    p.analysis.critical_carla = CC.CriticalCarlaSettings.from_dict(
        {'image_scale': 0.5, 'frames': 3, 'actors': {tl['uid']: 'standin'}}).to_dict()


def need(var, what) -> str:
    """A folder the clip needs, named by an environment variable (the recording machine's, never in the repo)."""
    v = os.environ.get(var, '')
    if not v or not os.path.isdir(v):
        raise RuntimeError(f'set {var} to {what}')
    return v


def link_drives(home) -> str:
    src = need('VANTAGE_CLIP_CRITICAL', 'a folder of the critical targets\' CARLA drives (each a drive folder)')
    os.makedirs(os.path.join(home, 'carla-drives'), exist_ok=True)
    dst = os.path.join(home, 'carla-drives', 'critical-targets')
    os.symlink(src, dst)
    return dst


def click_tab(r, tabs, title):
    bar = tabs.tabBar()
    i = next(k for k in range(tabs.count()) if tabs.tabText(k) == title)
    r.click(bar, QPointF(bar.tabRect(i).center()), 0.5, real=False)
    tabs.setCurrentIndex(i)
    r.hold(0.2)


def pick_combo(r, combo, data, dur=0.5):
    r.click(combo, None, dur, real=False)
    combo.showPopup()
    r.hold(0.35)
    view = combo.view()
    i = combo.findData(data)
    rect = view.visualRect(view.model().index(i, 0))
    r.move_to(view.viewport(), QPointF(rect.center().x() - rect.width() * 0.2, rect.center().y()), 0.4)
    view.setCurrentIndex(view.model().index(i, 0))
    r.hold(0.2)
    r.ripple()
    combo.setCurrentIndex(i)
    combo.hidePopup()
    r.hold(0.2)


def clip(stage, r):
    from vantage.ui.workspaces import critical_carla_ws as CW
    link = link_drives(stage.home)
    orig = CW.QFileDialog
    try:
        _clip(stage, r, link)
    finally:
        CW.QFileDialog = orig
        os.unlink(link)


def _clip(stage, r, link):
    from vantage.ui.workspaces import critical_carla_ws as CW
    w = stage.window(setup)
    r.win = w
    r.chapter('Critical targets in CARLA', 'The study checked in another renderer')
    r.menu('Simulate', 'Critical targets in CARLA')
    pg = w.page('critical_carla')
    stage.settle(2)
    r.hold(0.3)
    # the targets as CARLA actors: child walkers, an adult, a Mini, a bicycle with its rider, a board for the sign
    t = r.wrect(pg.tg_table)
    r.zoom(QRectF(t.x(), t.y() - 26, t.width(), (t.height() + 60) * 1.45), 0.7, margin=10)
    r.hold(2.2)
    r.zoom(None, 0.5)

    class Pick:                                   # the folder dialog answers with the drives' folder in ~
        @staticmethod
        def getExistingDirectory(*_a, **_k):
            return link
    CW.QFileDialog = Pick
    r.click(pg.b_open, dur=0.6)
    if not pg.busy and pg.result is None:
        raise RuntimeError(f'Open drives did not start: {pg.status_text}')
    r.wait_for(lambda: not pg.busy and pg.result is not None, timeout=600, speed=3)
    res = pg.result
    if res is None:
        raise RuntimeError(f'no result: {pg.status_text}')
    m = res.metrics
    print(f'  agree {m["agree"]}/{m["pairs"]}, nominal {m["agree_nominal"]}/{m["pairs_nominal"]}, targets '
          f'{m["targets_agree"]}/{m["targets_scenes"]}, px {m["px_error_rel_clear"]}, drives {len(res.drives)}',
          flush=True)
    r.hold(0.4)
    # the tiles and the agreement table
    a, b = r.wrect(pg.tiles['agree']), r.wrect(pg.rows)
    r.zoom(QRectF(a.x(), a.y(), b.right() - a.x(), b.y() + b.height() * 0.55 - a.y()), 0.7, margin=8)
    r.hold(2.3)
    # the crops: the child 2 m ahead in F, the child beside the door in ML
    click_tab(r, pg.tabs, 'Crops')
    for uid, hold in (('tgt00', 1.8), ('tgt02', 1.6)):
        if pg.crop_target.findData(uid) >= 0:
            pick_combo(r, pg.crop_target, uid)
            r.hold(hold)
    r.zoom(None, 0.5)
    # the chart: CARLA's pixels against the prediction at the actor's size
    click_tab(r, pg.tabs, 'Chart')
    c = r.wrect(pg.chart)
    r.zoom(c, 0.6, margin=6)
    r.hold(1.8)
    r.zoom(None, 0.5)
    r.cursor_on = False
    r.hold(0.4)
