"""The multi-sensor tools wired into the rest of the app: the sample's data-rig variant, batch mode, the exports and the
assistant (the studies themselves are tested in test_sensor_coverage, test_timing and test_recording)."""
import json

import pytest

from vantage.core import samples


def _rig_project():
    p = samples.tigor_sample_project()
    v = next(v for v in p.variants if v.sensors)
    p.active_uid = v.uid
    return p, v


def test_the_sample_has_a_data_rig_variant():
    p, v = _rig_project()
    kinds = [s.kind for s in v.sensors]
    assert kinds.count('lidar') == 1 and kinds.count('radar') == 5
    assert v.tags() >= {'F', 'L1', 'RD1', 'RD5'} and len(v.tags()) == len(v.cameras) + len(v.sensors)
    assert not p.variants[0].sensors                # the default layout stays cameras only
    from vantage.core.analysis import Engine
    from vantage.core.rules import sensor_issues
    assert sensor_issues(p, Engine().vehicle_geom(p)) == []


def test_batch_recording_budget(tmp_path):
    from vantage import batch
    from vantage.core import recording as RB
    from test_advanced import _args
    p, v = _rig_project()
    src = tmp_path / 'p.vantage'
    p.save(str(src))
    out = tmp_path / 'budget.json'
    a = _args(recording_budget=str(out))
    a.sample, a.path = False, str(src)
    assert batch.run(a) == 0
    doc = json.loads(out.read_text())
    assert doc['schema'] == RB.SCHEMA and doc['variant'] == v.name
    tags = {r['tag'] for r in doc['sensors']}
    assert {'L1', 'RD1', 'F'} <= tags
    assert batch.wanted(_args(recording_budget='-'))
    # a recorder too slow for the data fails the run with exit code 2
    p.analysis.recording = {'write_mb_s': 50}
    p.save(str(src))
    assert batch.run(a) == 2


def test_recording_budget_export_and_assistant(tmp_path):
    from vantage.assistant import tools as T
    from vantage.core import exporters as ex
    p, v = _rig_project()
    paths = ex.export(['recording_budget'], p, None, ex.ExportOptions(out_dir=str(tmp_path)))
    assert sorted(x.rsplit('/', 1)[-1] for x in paths) == ['recording_budget.csv', 'recording_budget.json']
    from vantage.core.analysis import Engine
    tc = T.ToolContext(p.snapshot(), None, Engine())
    out = json.loads(T.run(tc, 'get_recording_budget', {})[0])
    assert 'meaning' in out and 'error' not in out
    ov = json.loads(T.run(tc, 'get_overview', {})[0])
    assert [s['tag'] for s in ov['sensors']] == [s.tag for s in v.sensors]
    lid = ov['sensors'][0]
    assert lid['channels'] == 128 and lid['rotation'] == 'cw' and 'port' not in lid


def test_sync_plan_batch_export_and_assistant(tmp_path):
    from vantage import batch
    from vantage.assistant import tools as T
    from vantage.core import exporters as ex
    from vantage.core import timing as TM
    from vantage.core.analysis import Engine
    from test_advanced import _args
    p, v = _rig_project()
    src = tmp_path / 'p.vantage'
    p.save(str(src))
    out = tmp_path / 'sync.json'
    a = _args(sync_plan=str(out), units='mm')
    a.sample, a.path = False, str(src)
    assert batch.run(a) == 0
    doc = json.loads(out.read_text())
    assert doc['schema'] == TM.SCHEMA and doc['variant'] == v.name and doc['units'] == 'mm'
    assert batch.wanted(_args(sync_plan='-'))
    paths = ex.export(['sync_plan'], p, None, ex.ExportOptions(out_dir=str(tmp_path / 'x')))
    assert sorted(x.rsplit('/', 1)[-1] for x in paths) == ['sync_plan.json', 'sync_plan.png']
    tc = T.ToolContext(p.snapshot(), None, Engine())
    s = json.loads(T.run(tc, 'get_sync_plan', {})[0])
    assert s['lidar']['tag'] == 'L1' and set(s['suggested_groups']) == {c.tag for c in v.cameras}
    assert s['suggested_worst_misalignment_ms'] <= s['worst_misalignment_ms']


def test_trigger_lines_are_not_mirrored_and_dash_runs_free():
    from vantage.core.project import mirror_props
    from vantage.core import timing as TM
    p, v = _rig_project()
    ml, mr = v.by_tag('ML'), v.by_tag('MR')
    ml.sync = 'C'
    mirror_props(ml, mr, ['sync', 'pitch'])
    assert mr.sync == 'A' and mr.pitch == ml.pitch
    mr.sync = '-'
    res = TM.plan(p)
    assert next(c for c in res.cams if c['tag'] == 'MR')['sync'] == ''


def test_sensor_coverage_batch_export_preview_and_assistant(tmp_path):
    from vantage import batch
    from vantage.assistant import tools as T
    from vantage.core import exporters as ex
    from vantage.core import sensor_coverage as SC
    from vantage.core.analysis import Engine
    from test_advanced import _args
    p, v = _rig_project()
    src = tmp_path / 'p.vantage'
    p.save(str(src))
    out = tmp_path / 'coverage.json'
    a = _args(sensor_coverage=str(out), units='mm', frame='front_bumper')
    a.sample, a.path = False, str(src)
    assert batch.run(a) == 0
    doc = json.loads(out.read_text())
    assert doc['schema'] == SC.SCHEMA and doc['variant'] == v.name and doc['units'] == 'mm'
    assert doc['frame'] == 'front_bumper' and len(doc['lidars']) == 1 and len(doc['radars']) == 5
    assert batch.wanted(_args(sensor_coverage='-'))
    opts = ex.ExportOptions(out_dir=str(tmp_path / 'x'))
    rel, text, size = ex.preview('sensor_coverage', p, None, opts)
    assert rel.endswith('sensor_coverage.json') and 'ground_density' not in text and size > 1000
    paths = ex.export(['sensor_coverage'], p, None, opts)
    assert sorted(x.rsplit('/', 1)[-1] for x in paths) == ['sensor_coverage.json', 'sensor_coverage.png']
    tc = T.ToolContext(p.snapshot(), None, Engine())
    s = json.loads(T.run(tc, 'run_study', {'study': 'sensor_coverage'})[0])
    assert s['study'] == 'sensor_coverage' and s['lidars'][0]['lidar'] == 'L1' and 'meaning' in s
    assert 0 < s['lidars'][0]['blocked_pct'] < 100
    p.active_uid = p.variants[0].uid                 # cameras only: the study says what is missing
    tc = T.ToolContext(p.snapshot(), None, Engine())
    assert 'error' in json.loads(T.run(tc, 'run_study', {'study': 'sensor_coverage'})[0])
