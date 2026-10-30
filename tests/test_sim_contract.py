"""The simulator bridge contract (docs/simulator-bridge.md): the rig file as `vantage --carla-rig` writes it validates
against docs/sim_rig.schema.json, the run reports the CARLA bridge writes (against the fake carla module; with SUMO
traffic against tests/fake_sumo too) against docs/sim_run.schema.json, every field the document's tables name is in those files, and every field the schemas
describe is in the document's tables."""
import json
import os
import re
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DOCS = os.path.join(ROOT, 'docs')
DOC = os.path.join(DOCS, 'simulator-bridge.md')
RIG_SCHEMA = os.path.join(DOCS, 'sim_rig.schema.json')
RUN_SCHEMA = os.path.join(DOCS, 'sim_run.schema.json')
DATA_RIG = 'Data rig · 6-cam + lidar + radars'


def _validate(doc, schema):
    """The draft 2020-12 validator project.schema.json is checked with (tests/test_provenance.py), and jsonschema too
    when it is installed."""
    from test_provenance import _validate as validate
    errs = validate(doc, schema)
    try:
        import jsonschema
    except ImportError:
        jsonschema = None
    if jsonschema is not None:
        v = jsonschema.Draft202012Validator(schema)
        errs += [f'jsonschema: {"/".join(map(str, e.absolute_path))}: {e.message}' for e in v.iter_errors(doc)]
    return errs


def _load(path):
    with open(path, encoding='utf-8') as fh:
        return json.load(fh)


# ============================================================================ the documents
@pytest.fixture(scope='module')
def rigs(tmp_path_factory):
    """Rig files of the sample's data rig: as `vantage --sample --variant <data rig> --carla-rig OUT` writes them (one
    plain, one with every fault kind, a scenario, an actors scene and walkers), and one with rectified images built
    by core.simrig directly."""
    from vantage import batch
    from vantage.core import samples
    from vantage.core import simrig as SR
    base = tmp_path_factory.mktemp('rigs')
    out = {}
    a = batch.parser().parse_args(['--sample', '--variant', DATA_RIG, '--carla-rig', str(base / 'plain')])
    assert batch.run(a) == 0
    out['plain'] = _load(base / 'plain' / 'sim_rig.json')
    faults = [{'tag': 'F', 'kind': 'rotation', 'deg': [0.0, 1.0, 0.0]},
              {'tag': 'RD1', 'kind': 'translation', 'm': [0.1, 0.0, 0.0]},
              {'tag': 'R', 'kind': 'clock_offset', 'ms': 5},
              {'tag': 'L1', 'kind': 'drop', 'rate': 0.1}, {'tag': 'ML', 'kind': 'drop', 'frames': [1, 2]},
              {'tag': 'MR', 'kind': 'freeze', 'from_s': 1.0, 'to_s': 2.0},
              {'tag': 'L1', 'kind': 'range_bias', 'm': 0.2}]
    actors = [{'id': 'kid', 'kind': 'child', 'pose': {'x': 7.0, 'y': 0.8, 'yaw': 90.0}},
              {'id': 'a2', 'kind': 'blueprint:vehicle.audi.a2', 'pose': {'x': -12.0, 'y': 0.0}}]
    a = batch.parser().parse_args(['--sample', '--variant', DATA_RIG, '--carla-rig', str(base / 'faults'),
                                   '--carla-faults', json.dumps(faults), '--carla-scenario', 'cut_in',
                                   '--carla-scenario-params', '{"cut_at_s": 1.5}', '--carla-actors',
                                   json.dumps(actors), '--carla-spawn-point', '3', '--carla-walkers', '4',
                                   '--carla-ground-truth', 'range'])
    assert batch.run(a) == 0
    out['faults'] = _load(base / 'faults' / 'sim_rig.json')
    assert {(f['tag'], f['kind']) for f in out['faults']['faults']} == {(f['tag'], f['kind']) for f in faults}
    assert _load(base / 'faults' / 'faults.json') == out['faults']['faults']
    p = samples.tigor_sample_project()
    p.active_uid = next(v.uid for v in p.variants if v.name == DATA_RIG)
    rig = SR.build(p, SR.SimSettings.from_dict({'undistort': True, 'image_scale': 0.1, 'weather': 'ClearNoon',
                                                'channel_names': 'nuscenes', 'camera_rate': 'module'}), tables=False)
    out['rectified'] = json.loads(dict(SR.files(rig, p))['sim_rig.json'])      # as export() writes it
    rig = SR.build(p, SR.SimSettings.from_dict({'image_scale': 0.1, 'traffic_source': 'sumo', 'town': 'Town04',
                                                'sumo': {'demand': 'period', 'period_s': 1.5, 'tls': 'sumo'}}),
                   tables=False)
    out['sumo'] = json.loads(dict(SR.files(rig, p))['sim_rig.json'])          # SUMO traffic: its traffic_source
    return out


@pytest.fixture(scope='module')
def runs(tmp_path_factory):
    """Run reports (and the drives) of the CARLA bridge against the fake carla: a drive with an IMU, two GNSS antennas,
    a radar and every ground truth; one with every fault kind, a scenario and walkers; an actors scene; a run the
    bridge fails (its command line); and one whose bridge ended without a report (Vantage writes that one)."""
    from test_sim import FAKE_DIR, _mini_run
    base = tmp_path_factory.mktemp('runs')
    out = {}
    r = _mini_run(base / 'plain', {'duration_s': 1.0}, cams=('F', 'MR'), radars=('RD1',), extra=True)
    out['plain'] = r
    faults = [{'tag': 'F', 'kind': 'rotation', 'deg': [0.0, 2.0, 0.0]}, {'tag': 'F', 'kind': 'clock_offset', 'ms': 5},
              {'tag': 'F', 'kind': 'freeze', 'from_s': 0.2, 'to_s': 0.4},
              {'tag': 'L1', 'kind': 'drop', 'frames': [2, 5]}, {'tag': 'L1', 'kind': 'range_bias', 'm': 0.3},
              {'tag': 'RD1', 'kind': 'freeze', 'from_s': 0.3, 'to_s': 0.7},
              {'tag': 'RD1', 'kind': 'translation', 'm': [0.1, 0.0, 0.0]}, {'tag': 'IMU1', 'kind': 'drop', 'rate': 0.5}]
    out['faults'] = _mini_run(base / 'faults', {'faults': faults, 'duration_s': 1.0, 'walkers': 2,
                                                'scenario': {'name': 'lead_vehicle_brake',
                                                             'params': {'brake_at_s': 0.4, 'gap_m': 12.0}}},
                              radars=('RD1',), extra=True)
    actors = [{'id': 'kid', 'kind': 'child', 'pose': {'x': 7.0, 'y': 0.8, 'yaw': 90.0}},
              {'id': 'tl', 'kind': 'traffic_light', 'pose': {'x': 15.0, 'y': 4.0}},
              {'id': 'cone', 'kind': 'cone', 'pose': {'x': 5.0, 'y': -2.0}}]
    out['actors'] = _mini_run(base / 'actors', {'actors': actors, 'spawn_point': 3, 'traffic': 2,
                                                'duration_s': 0.5})
    # the bridge's own command line, failing: no such spawn point
    rig_json = os.path.join(out['plain']['out'], 'rig', 'sim_rig.json')
    env = dict(os.environ, PYTHONPATH=FAKE_DIR + os.pathsep + ROOT)
    fail = base / 'failed'
    res = subprocess.run([sys.executable, '-m', 'vantage.sim.carla_bridge', rig_json, str(fail), '--weather',
                          'ClearNoon', '--spawn-point', '99999', '--duration', '0.3'], env=env,
                         capture_output=True, text=True, timeout=600, cwd=str(base))
    assert res.returncode == 2, res.stderr[-2000:]
    out['failed'] = {'out': str(fail), 'report': _load(fail / 'run.json')}
    # a bridge that ends without a report: Vantage's critical-targets drive writes run.json itself
    from vantage.core import critical_carla as CC
    from vantage.core import samples
    from vantage.core import simrig as SR
    p = samples.tigor_sample_project()
    v = next(v for v in p.variants if v.name == DATA_RIG)
    gone = base / 'gone'
    with pytest.raises(RuntimeError):
        CC.drive(p, v, SR.SimSettings.from_dict({'image_scale': 0.05, 'duration_s': 0.5}), str(gone),
                 runner=lambda rig_json, out_dir, progress, cancel: None)
    out['gone'] = {'out': str(gone), 'report': _load(gone / 'run.json')}
    # SUMO traffic (tests/fake_sumo): on a net generated from the map, SUMO running the lights; and the user's .sumocfg
    from test_sumo import FAKE_ROOT, fake_sumo, sumo_drive
    with fake_sumo():
        out['sumo'] = sumo_drive(base / 'sumo', {'sumo': {'tls': 'sumo'}}, cams=())
        net = base / 'sumo' / 'sumo' / 'run.sumocfg'
        cfg = base / 'mine.sumocfg'
        (base / 'mine.rou.xml').write_text('<routes><vType id="vehicle.audi.a2" vClass="passenger" length="3.7" '
                                           'width="1.8" height="1.5"/><trip id="mine0" depart="0" from="east" '
                                           'to="east" type="vehicle.audi.a2"/></routes>')
        import xml.etree.ElementTree as ET
        net_file = ET.parse(net).getroot().find('.//net-file').get('value')
        cfg.write_text(f'<configuration><input><net-file value="{net_file}"/><route-files value="mine.rou.xml"/>'
                       '</input></configuration>')
        out['sumo_cfg'] = sumo_drive(base / 'sumo_cfg', {'sumo': {'net': str(cfg)}}, cams=(), carla_root=FAKE_ROOT)
    return out


# ============================================================================ the schemas
def test_the_rig_schema_describes_every_rig_file(rigs, runs):
    schema = _load(RIG_SCHEMA)
    docs = dict(rigs)
    for name in ('plain', 'faults', 'actors', 'sumo'):         # rigs with an IMU and GNSS antennas; SUMO traffic
        docs['run_' + name] = _load(os.path.join(runs[name]['out'], 'rig', 'sim_rig.json'))
    for name, doc in docs.items():
        assert _validate(doc, schema) == [], name
    assert rigs['plain']['provenance']['schema'] == 'vantage.provenance/1'     # stamped as --carla-rig stamps it
    assert rigs['rectified']['cameras'][0]['lens']['model'] == 'pinhole'
    assert docs['run_plain']['imus'] and docs['run_plain']['gnss']
    # and it rejects what is wrong
    import copy
    for mutate in (lambda d: d.update(schema='vantage.sim_rig/2'),
                   lambda d: d['cameras'][0]['pose_vehicle'].pop('pitch'),
                   lambda d: d['cameras'][0]['lens'].update(projection='fisheye'),
                   lambda d: d['lidars'][0].update(rotation='cw'),
                   lambda d: d['output'].update(channel_names='topics'),
                   lambda d: d['faults'].append({'tag': 'F', 'kind': 'tilt'}),
                   lambda d: d['simulation'].update(tick_hz='80'),
                   lambda d: d.update(extra=1)):
        d = copy.deepcopy(rigs['faults'])
        mutate(d)
        assert _validate(d, schema), d.get('schema')


def test_the_run_schema_describes_every_run_report(runs):
    schema = _load(RUN_SCHEMA)
    for name, r in runs.items():
        assert _validate(r['report'], schema) == [], name
    assert runs['plain']['report']['status'] == 'ok' and runs['faults']['report']['status'] == 'ok'
    assert runs['failed']['report']['status'] == 'failed' and 'Spawn point' in runs['failed']['report']['error']
    assert runs['gone']['report']['status'] == 'failed' and 'exit_code' in runs['gone']['report']
    assert runs['sumo']['report']['traffic_source']['net']['kind'] == 'generated'
    assert runs['sumo_cfg']['report']['traffic_source']['demand']['mode'] == 'user'
    assert 'traffic_source' not in runs['plain']['report']
    # an 'ok' report must carry what the checks read; a failed one need not
    import copy
    ok = runs['faults']['report']
    for mutate in (lambda d: d.pop('scene'),
                   lambda d: d['performance'].pop('simulated_s'),
                   lambda d: d.pop('faults_injected'),
                   lambda d: d['faults_injected'][0].update(kind='tilt'),
                   lambda d: d.update(status='done'),
                   lambda d: d['channels'][0].update(frames=-1),
                   lambda d: d.update(extra=1)):
        d = copy.deepcopy(ok)
        mutate(d)
        assert _validate(d, schema)
    for mutate in (lambda d: d['traffic_source'].update(source='carla'),
                   lambda d: d['traffic_source']['net'].update(kind='downloaded'),
                   lambda d: d['traffic_source'].pop('actors'),
                   lambda d: d['traffic_source']['actors'][0].update(colour='red')):
        d = copy.deepcopy(runs['sumo']['report'])
        mutate(d)
        assert _validate(d, schema)
    assert _validate({'schema': 'vantage.sim_run/1', 'status': 'cancelled', 'notes': []}, schema) == []
    assert _validate({'schema': 'vantage.sim_run/1', 'status': 'ok'}, schema)


# ============================================================================ the document's tables
MARKERS = (('`sim_rig.json`', 'rig'), ('`run.json`', 'run'), ('`dataset.json`', 'dataset'),
           ('`ground_truth/index.json`', 'gt'), ('`nuscenes/v1.0-vantage/`', 'nus'))


def _doc_paths() -> dict:
    """{target: {path}}: the key paths in the first column of the 'key' tables under a heading naming a file (an H2's
    file holds for its H3s unless they name their own). A bare name after a path is a sibling of its last key."""
    out = {t: set() for _, t in MARKERS}
    h2 = cur = None
    in_table = key_table = False
    with open(DOC, encoding='utf-8') as fh:
        for line in fh:
            m = re.match(r'^(#{2,4}) (.*)', line)
            if m:
                found = next((t for mk, t in MARKERS if mk in m.group(2)), None)
                if len(m.group(1)) == 2:
                    h2 = cur = found
                else:
                    cur = found or h2
                continue
            if not line.startswith('|'):
                in_table = False
                continue
            if not in_table:                                # a table's header row: only 'key' tables name keys
                in_table, key_table = True, line.split('|')[1].strip() == 'key'
                continue
            if cur is None or not key_table or not line.startswith('| `'):
                continue
            cell = line.split('|')[1]
            parent = None
            for item in re.findall(r'`([^`]+)`', cell):
                if not re.fullmatch(r'[A-Za-z_][\w.\[\]{}]*', item):
                    continue
                if parent is not None and not re.search(r'[.\[{]', item):
                    path = parent + '.' + item if parent else item
                else:
                    path = item
                parent = path.rsplit('.', 1)[0] if '.' in path else ''
                out[cur].add(re.sub(r'\{[^}]*\}', '{}', path))
    return out


def _segments(path):
    return re.findall(r'[^.\[\]]+|\[\]', path)


def _has(doc, path) -> bool:
    """Whether some value in doc lies at path ('a.b', 'a[].b', 'a.{}.b')."""
    nodes = [doc]
    for seg in _segments(path):
        nxt = []
        for n in nodes:
            if seg == '[]' and isinstance(n, list):
                nxt += n
            elif seg == '{}' and isinstance(n, dict):
                nxt += list(n.values())
            elif isinstance(n, dict) and seg in n:
                nxt.append(n[seg])
        nodes = nxt
        if not nodes:
            return False
    return True


ATOMIC = {'pose', 'carla_pose', 'vec3', 'mat3', 'mat4', 'quat', 'size2'}


def _schema_paths(schema) -> set:
    """Every key path the schema describes (not inside its small shared types, nor the provenance block)."""
    out = set()

    def resolve(node):
        name = None
        while isinstance(node, dict) and '$ref' in node:
            name = node['$ref'].rsplit('/', 1)[-1]
            n = schema
            for part in node['$ref'].lstrip('#/').split('/'):
                n = n[part]
            node = n
        return node, name

    def walk(node, path):
        node, name = resolve(node)
        if name in ATOMIC or not isinstance(node, dict):
            return
        for br in node.get('anyOf', ()):
            walk(br, path)
        for k, v in node.get('properties', {}).items():
            if k == 'provenance':
                continue
            p = f'{path}.{k}' if path else k
            out.add(p)
            walk(v, p)
        if isinstance(node.get('items'), dict):
            walk(node['items'], path + '[]')
        if isinstance(node.get('additionalProperties'), dict) and path:
            walk(node['additionalProperties'], path + '.{}')
    walk(schema, '')
    return out


def _with_prefixes(paths) -> set:
    """The paths and every path leading to them (a documented 'a[].b' documents 'a' and 'a[]')."""
    out = set()
    for p in paths:
        cur = ''
        for seg in _segments(p):
            cur = cur + '[]' if seg == '[]' else (cur + '.' + seg if cur else seg)
            out.add(cur)
    return out


def test_every_field_the_tables_name_is_written(rigs, runs):
    named = _doc_paths()
    assert all(named[t] for t in ('rig', 'run', 'dataset', 'gt', 'nus')), {t: len(v) for t, v in named.items()}
    rig_docs = list(rigs.values()) + [_load(os.path.join(runs[n]['out'], 'rig', 'sim_rig.json'))
                                      for n in ('plain', 'faults', 'actors', 'sumo')]
    run_docs = [r['report'] for r in runs.values()]
    drives = [runs[n]['out'] for n in ('plain', 'faults', 'actors')]
    ds_docs = [_load(os.path.join(d, 'dataset.json')) for d in drives]
    gt_docs = [_load(os.path.join(d, 'ground_truth', 'index.json')) for d in drives]
    nus_docs = []
    for d in drives:
        V = os.path.join(d, 'nuscenes', 'v1.0-vantage')
        nus_docs.append({f[:-5]: _load(os.path.join(V, f)) for f in os.listdir(V) if f.endswith('.json')})
    for target, docs in (('rig', rig_docs), ('run', run_docs), ('dataset', ds_docs), ('gt', gt_docs),
                         ('nus', nus_docs)):
        missing = sorted(p for p in named[target] if not any(_has(d, p) for d in docs))
        assert not missing, (target, missing)
    # the nuScenes tables the document lists are every table written
    assert {_segments(p)[0] for p in named['nus']} == set(nus_docs[0])


def test_every_field_the_schemas_describe_is_in_the_tables():
    named = _doc_paths()
    for target, path in (('rig', RIG_SCHEMA), ('run', RUN_SCHEMA)):
        described = _schema_paths(_load(path))
        documented = _with_prefixes(named[target])
        assert not sorted(described - documented), (target, sorted(described - documented))
        assert not sorted(named[target] - described), (target, sorted(named[target] - described))


def test_the_schemas_name_their_documents():
    from vantage.core import simrig as SR
    from vantage.sim import carla_bridge as CB
    rig, run = _load(RIG_SCHEMA), _load(RUN_SCHEMA)
    assert rig['properties']['schema']['const'] == SR.SCHEMA == CB.RIG_SCHEMA
    assert run['properties']['schema']['const'] == SR.RUN_SCHEMA == CB.RUN_SCHEMA
    assert rig['$schema'] == run['$schema'] == _load(os.path.join(DOCS, 'project.schema.json'))['$schema']
    assert set(rig['$defs']['fault']['properties']['kind']['enum']) == set(CB.FAULT_KINDS)
    assert set(rig['$defs']['scenario']['properties']['name']['enum']) == set(CB.SCENARIOS)
    assert set(rig['$defs']['weather']['properties']['params']['properties']) == set(SR.WEATHER_KEYS)
    text = open(DOC, encoding='utf-8').read()
    for name in ('sim_rig.schema.json', 'sim_run.schema.json', 'tests/test_sim_contract.py', 'tests/fake_carla',
                 'tests/fake_sumo', 'tests/test_sumo.py'):
        assert name in text
