"""The research and AV-stack bridges in the export registry (exporters._BRIDGES): every format writes its files through
exporters.export like any other, under its own folder, with the exported variant active; the Export menu's Research,
AV research stack and Stream to Rerun / Foxglove submenus; Analysis ▸ Compare rigs."""
import json
import os
import time

import pytest

from vantage.core import exporters as ex
from vantage.core import samples
from test_gui import app, win, pump, settle  # noqa: F401  (fixtures)


def _rig():
    p = samples.tigor_sample_project()
    p.active_uid = next(v.uid for v in p.variants if v.sensors)
    return p


RESEARCH = [f.id for f in ex.FORMATS if f.group in ('Research', 'AV stack')]
NOT_BRIDGES = {'perception_records'}          # Research formats written by exporters itself, not through a bridge
BRIDGED = [f for f in RESEARCH if f not in NOT_BRIDGES]


def test_every_bridge_is_registered_off_by_default():
    assert set(BRIDGED) == set(ex._BRIDGES) and NOT_BRIDGES <= set(RESEARCH)
    assert {'plucker_rays', 'occupancy_density', 'rectify_maps', 'rectify_canonical', 'rig_gap', 'stress_test',
            'gsplat_bridge', 'foxglove_mcap', 'stack_autoware', 'stack_navsim', 'stack_hugsim', 'stack_mmdet3d',
            'stack_openpcdet', 'stack_lerobot'} <= set(BRIDGED)
    assert not any(ex.format_by_id(f).default for f in BRIDGED)          # batch --export without names: unchanged
    assert {'Research', 'AV stack'} <= set(ex.GROUPS)


@pytest.mark.parametrize('fid', ['plucker_rays', 'occupancy_density', 'rectify_maps', 'rig_gap', 'stress_test',
                                 'gsplat_bridge', 'foxglove_mcap', 'stack_autoware', 'stack_navsim', 'stack_hugsim',
                                 'stack_openpcdet', 'stack_lerobot'])
def test_a_bridge_writes_under_its_folder(fid, tmp_path):
    p = _rig()
    paths = ex.export([fid], p, None, ex.ExportOptions(out_dir=str(tmp_path)))
    sub = ex._BRIDGES[fid][2]
    assert paths and all(os.path.relpath(x, tmp_path).split(os.sep)[0] == sub for x in paths), paths
    assert all(os.path.getsize(x) > 0 for x in paths)
    for x in paths:
        if x.endswith('.json'):
            json.load(open(x, encoding='utf-8'))


def test_a_bridge_that_does_not_apply_says_so_and_the_rest_export(tmp_path):
    """OpenPCDet on a variant without a lidar: a note instead of its files, and the other formats still written."""
    p = samples.tigor_sample_project()                           # 6-cam · v3: cameras only
    paths = ex.export(['stack_openpcdet', 'stack_navsim'], p, None, ex.ExportOptions(out_dir=str(tmp_path)))
    note = next(x for x in paths if x.endswith('openpcdet/NOT_WRITTEN.txt'))
    assert 'lidar' in open(note).read().lower() and any(x.endswith('vantage_cams.json') for x in paths)


def test_the_exported_variant_is_the_one_bridged(tmp_path):
    """The bridge exports the variant asked for (made active on a copy), not whichever is active in the project."""
    p = _rig()
    four = next(v for v in p.variants if v.name.startswith('4-cam'))
    res = type('R', (), {'variant_uid': four.uid})()
    paths = ex.export(['stack_navsim'], p, res, ex.ExportOptions(out_dir=str(tmp_path)))
    cams = json.load(open(next(x for x in paths if x.endswith('vantage_cams.json'))))['cams']
    assert len([c for c in cams.values() if c['data_path']]) == len(four.cameras) and p.active_uid != four.uid


def test_menus_streaming_and_rig_comparison(app, win, tmp_path):  # noqa: F811
    menus = {a.text().replace('&', ''): a.menu() for a in win.menuBar().actions()}
    subs = {a.text(): a.menu() for a in menus['Export'].actions() if a.menu()}
    assert len(subs['Research'].actions()) == len([f for f in ex.FORMATS if f.group == 'Research'])
    assert len(subs['AV research stack'].actions()) == len([f for f in ex.FORMATS if f.group == 'AV stack'])
    assert [a.text() for a in subs['Stream to Rerun / Foxglove'].actions()][0].startswith('Save the scene for Foxglove')
    win.store.set_project(_rig())
    settle(app, win)
    out = win.stream_foxglove(str(tmp_path / 'scene.mcap'))
    assert os.path.getsize(out) > 10_000
    from vantage.core import viz_bridge as VB
    if not VB.rerun_available():
        assert 'pip install rerun-sdk' in win.stream_rerun()
    from vantage.ui.dialogs.rig_gap_dialog import RigGapDialog
    dlg = RigGapDialog(win.ctx, win)
    dlg.show()
    dlg.run()
    t = time.time()
    while dlg.result is None and time.time() - t < 60:
        pump(app, 0.05)
    assert dlg.result is not None and 0.0 <= dlg.result.total <= 1.0
    assert 'heuristic' in dlg.headline.text() and dlg.table.item(6, 0).text() == 'Total'
    dlg.close()
