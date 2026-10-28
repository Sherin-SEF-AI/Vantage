"""Website clip 'campaign': Simulate ▸ Simulation campaigns… on the real fault-detectability campaign.

49 drives recorded in CARLA 0.9.16 with the sample's data rig (an hour of CARLA): F and ML turned 0.1–1° about each
axis, R's clock 1–10 ms late, MR's frames dropped, QR frozen for 1 s, three seeds and a clean drive each. Every
drive and every check is already on disk, so Run drives nothing and checks nothing: it reuses all 49 drives and
their cached checks and analyses them (about 40 s, shown as a time-lapse). The tiles (the False alarms tile counts
the checks that can judge on CARLA: 0 on the clean drives, the as-built check's 6 said beside it as not applicable),
the drive-log check's detection probability against the rotation's size with its Wilson intervals and the 90 % line,
the thresholds (F: every size from 0.5°, fit 0.35°; ML: from 1°, fit 0.52°), then the False alarms tab on the clean
drives: the last of the checks that judge, none flagging, then the as-built check's rows, greyed and marked not
applicable (it cannot judge a mount on CARLA's car, another body).

The campaign folder is a working copy whose sensor files link to the recording, linked into the throw-away home
(the folder VANTAGE_CLIP_CAMPAIGN names) and shown from ~; the link is removed at the end. The campaign names its
own CARLA port (2400, where nothing runs): no drive is due, so no server is asked or started. The project is never
saved.
"""
from __future__ import annotations

import json
import os

from PyQt6.QtCore import QPointF, QRectF



def need(var, what) -> str:
    """A folder the clip needs, named by an environment variable (the recording machine's, never in the repo)."""
    v = os.environ.get(var, '')
    if not v or not os.path.isdir(v):
        raise RuntimeError(f'set {var} to {what}')
    return v


def link_campaign(home) -> str:
    src = need('VANTAGE_CLIP_CAMPAIGN', 'a working copy of the campaign\'s folder (campaign_plan.json, drives/, rigs/)')
    if not os.path.isfile(os.path.join(src, 'campaign_plan.json')):
        raise RuntimeError(f'no campaign at {src}')
    parent = os.path.join(home, 'Documents', 'Vantage', 'campaigns')
    os.makedirs(parent, exist_ok=True)
    dst = os.path.join(parent, 'fault-detectability')
    os.symlink(src, dst)
    return dst


def clip(stage, r):
    link = link_campaign(stage.home)
    try:
        _clip(stage, r, link)
    finally:
        os.unlink(link)


def _clip(stage, r, link):
    from vantage.core import simcampaign as SC, simrig as SR
    with open(os.path.join(link, 'campaign_plan.json'), encoding='utf-8') as fh:
        stored = json.load(fh)['settings']

    def setup(p):
        p.analysis.sim = SR.SimSettings.from_dict({'port': 2300, 'tm_port': 8400}).to_dict()
        p.analysis.campaign = SC.CampaignSettings.from_dict(dict(stored, out_dir=link)).to_dict()
    w = stage._rig_window(setup)
    r.win = w
    pg = w.page('campaign')                     # built before it shows: its folder field shown from ~
    stage.tilde(pg.out_edit)
    r.chapter('Simulation campaigns', 'How small a fault each check finds, how often it cries wolf')
    r.menu('Simulate', 'Simulation campaigns')
    stage.settle(2)
    r.hold(0.3)
    # the design: the fault sets, the seeds, clean drives, the cost
    f = r.wrect(pg.faults)
    r.zoom(QRectF(f.x(), f.y() - 34, f.width(), f.height() + 250), 0.7, margin=10)
    r.hold(2.0)
    r.zoom(None, 0.5)
    r.click(pg.b_run, dur=0.6)
    if not pg.busy:
        raise RuntimeError(f'the run did not start: {pg.status.text()}')
    r.wait_for(lambda: not pg.busy and pg.result is not None, timeout=900, speed=8)
    res = pg.result
    if res is None:
        raise RuntimeError(f'no result: {pg.status.text()}')
    states = {s: sum(x.status == s for x in res.states.values()) for s in ('reused', 'ok', 'pending', 'failed')}
    print(f'  {pg.status.text()} · {states}', flush=True)
    if states['ok'] or states['pending'] or states['failed']:
        raise RuntimeError(f'not every drive was reused: {states}')
    for line in SC.headline(res)[:6]:
        print('   ', line, flush=True)
    r.hold(0.3)
    # the tiles and the detectability chart of the drive-log check's camera rotation
    i = pg.check_pick.findData('drivelog.extrinsics|rotation')
    if i >= 0:
        pg.check_pick.setCurrentIndex(i)
    a = r.wrect(pg.t_drives)
    ch = r.wrect(pg.chart)
    r.zoom(QRectF(a.x(), a.y(), ch.right() - a.x(), ch.bottom() - a.y()), 0.7, margin=8)
    r.hold(3.0)
    # the thresholds: the smallest rotation found 90 % of the time, empirical and fitted
    t = r.wrect(pg.table_thresholds)
    r.zoom(QRectF(t.x(), t.y(), t.width(), t.height() * 0.5), 0.6, margin=8)
    r.hold(2.4)
    r.zoom(None, 0.5)
    # false alarms on the clean drives: none of the checks that can judge on CARLA flags one (the tile's 0); the
    # as-built check cannot judge a mount on CARLA's car (another body), so its flags are listed after them, greyed,
    # marked not applicable, and not counted
    bar = pg.tabs.tabBar()
    i = next(k for k in range(pg.tabs.count()) if pg.tabs.tabText(k) == 'False alarms')
    r.click(bar, QPointF(bar.tabRect(i).center()), 0.5, real=False)
    pg.tabs.setCurrentIndex(i)
    t = pg.table_fa
    na = [k for k in range(t.rowCount()) if t.item(k, 2).text() == 'clean drives' and
          t.item(k, 7).text() == 'not applicable']
    if not na:
        raise RuntimeError('no check marked not applicable on the clean drives')
    t.verticalScrollBar().setValue(max(0, na[0] - 6))            # the last checks that judge, then those that cannot
    print(f'  false alarms: tile {pg.t_fa.value} {pg.t_fa.unit} · {pg.t_fa.note}; rows not applicable '
          f'{[(t.item(k, 1).text(), t.item(k, 3).text() + "/" + t.item(k, 4).text()) for k in na]}', flush=True)
    r.hold(0.2)
    fa = r.wrect(t)
    r.zoom(QRectF(fa.x(), fa.y(), fa.width(), min(fa.height(), 14 * 22 + 30)), 0.6, margin=8)
    r.hold(2.6)
    r.zoom(None, 0.5)
    r.cursor_on = False
    r.hold(0.4)
