"""The command line and exports after the 0.10.0 hardening pass: one list of export frames, modifiers refused alone
(not the window opening), every format the default export leaves out documented, the KITTI README and the blind
volume saying what they hold, and --tolerance patterns that match nothing said."""
import os

import pytest

from vantage import batch
from vantage.core import exporters as ex
from vantage.core import provenance as PV
from vantage.core import samples
from vantage.core.project import FRAME_PRESETS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_one_list_of_export_frames():
    ap = batch.parser()
    assert ap.parse_args(['--sample', '--export', 'x', '--frame', 'centre']).frame == 'centre'
    with pytest.raises(SystemExit):
        ap.parse_args(['--sample', '--export', 'x', '--frame', 'ground_centre'])     # refused before any analysis
    from vantage.assistant import copilot as CP
    from vantage.assistant import copilot_tools as CT
    out, bad = CP._clean_export({'formats': ['extrinsics_json'], 'frame': 'centre'})
    assert out['frame'] == 'centre' and not bad                   # (it became rear_axle: wrong coordinates)
    out, bad = CP._clean_export({'formats': ['extrinsics_json'], 'frame': 'ground_centre'})
    assert any('ground_centre' in b for b in bad)
    assert CT._choices('export', 'frame') == list(FRAME_PRESETS)
    for path in ('vantage/assistant/copilot.py', 'vantage/assistant/copilot_tools.py', 'vantage/batch.py',
                 'site/docs.html'):
        assert 'ground_centre' not in open(os.path.join(ROOT, path), encoding='utf-8').read(), path


@pytest.mark.parametrize('argv', [['--formats', 'extrinsics_json'], ['--tolerance', '5%'], ['--refine-from', '2'],
                                  ['--live-cameras', 'F'], ['--risk-weight', 'fatal'], ['--verify', '3'],
                                  ['--carla-python', 'py'],
                                  ['--analyse', '-', '--formats', 'extrinsics_json']])
def test_a_modifier_alone_is_refused_not_the_window_opened(argv):
    args = batch.parser().parse_args(['--sample'] + argv)
    assert batch.wanted(args)                                     # batch mode, not the window
    with pytest.raises(SystemExit) as e:
        batch.run(args)                                           # refused before any analysis
    assert 'goes with' in str(e.value) or 'needs' in str(e.value) or 'checks the predictions' in str(e.value)


def test_the_docs_list_every_format_the_default_export_leaves_out():
    text = open(os.path.join(ROOT, 'docs', 'command-line.md'), encoding='utf-8').read()
    missing = [f.id for f in ex.FORMATS if not f.default and f'`{f.id}`' not in text]
    assert not missing, missing


def test_kitti_readme_and_blind_volume_say_what_they_hold(tmp_path):
    from vantage.core.analysis import Engine
    from vantage.core import volume as VOL
    p = samples.tigor_sample_project()
    assert not any(s.kind == 'lidar' for s in p.variant.sensors)
    e = Engine()
    res = e.analyse(p)
    paths = ex.export(['kitti_calib'], p, res, ex.ExportOptions(out_dir=str(tmp_path), engine=e))
    readme = next(x for x in paths if x.endswith('README.txt'))
    assert 'kitti/object/' not in open(readme, encoding='utf-8').read()   # no lidar: no object files to point to
    vr = VOL.analyse_volume(p, e, res, 0.6, 0.1, 0.1)
    head = VOL.to_obj(vr).splitlines()[:2]
    assert 'rear-axle' in ' '.join(head) and 'whatever the export' in ' '.join(head)
    assert 'rear-axle centre' in VOL.to_doc(vr, p)['note']


def test_a_tolerance_pattern_that_matches_nothing_is_said():
    doc = {'metrics': {'blind_m2': 0.010, 'covered_pct': 99.0}}
    tol = PV.parse_tolerances(['blnd_m2=1', 'covered_pct=0.5'])
    rep = PV.report('r.json', doc, {'metrics': {'blind_m2': 0.012, 'covered_pct': 99.2}}, tol)
    assert rep['unused_tolerances'] == ['blnd_m2']
    assert any("'blnd_m2' matched no number" in r for r in rep['reasons'])
