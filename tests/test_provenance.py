"""Provenance: the block on every export and batch output (no paths of this machine), the project's content hash and a
result's hash, `vantage --reproduce`, the determinism audit (no unseeded random numbers; studies bit for bit the same
in two processes), the project format's JSON Schema and the citation file."""
import ast
import copy
import datetime as dt
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import zipfile

import numpy as np
import pytest
import yaml

from vantage import __version__, app
from vantage.core import exporters as ex
from vantage.core import provenance as PV
from vantage.core import samples
from vantage.core.analysis import Engine
from vantage.core.project import Project

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = os.path.join(ROOT, 'vantage')
FIXED_NOW = dt.datetime(2026, 10, 27, 9, 30, 0, tzinfo=dt.timezone(dt.timedelta(hours=5, minutes=30)))


def _rig():
    p = samples.tigor_sample_project()
    p.active_uid = next(v for v in p.variants if v.sensors).uid          # the Data rig: cameras, a lidar, radars
    return p


@pytest.fixture(scope='module')
def proj(tmp_path_factory):
    """The sample's Data rig saved and read back (fixed uids, float32 vertices), in a deep folder."""
    d = tmp_path_factory.mktemp('prov') / 'projects' / 'rig'
    d.mkdir(parents=True)
    p = _rig()
    p.analysis.dataset = {'duration_s': 0.5, 'image_scale': 0.1}           # a short sample dataset
    path = str(d / 'rig.vantage')
    p.save(path)
    return path


def _main(*argv):
    return app.main([str(a) for a in argv])


def _load(path):
    with open(path, encoding='utf-8') as fh:
        return json.load(fh)


def _json_files(folder):
    for root, _, files in os.walk(folder):
        for f in sorted(files):
            if f.endswith('.json'):
                yield os.path.join(root, f)


# ============================================================================ canonical JSON, hashes
def test_canonical_json_and_the_result_hash():
    # the canonical form, checked against a hand-written string
    assert PV.canonical({'b': 2.0, 'a': [1, float('nan'), -0.0, 0.25], 'c': {'z': True, 'y': None}}) == \
        b'{"a":[1,NaN,0,0.25],"b":2,"c":{"y":null,"z":true}}'
    assert PV.canonical({'t': (1, 2)}) == PV.canonical({'t': [1.0, 2.0]})
    assert PV.canonical({'n': np.float32(0.5), 'a': np.arange(3)}) == b'{"a":[0,1,2],"n":0.5}'
    big = 2 ** 60 + 1                                                     # nanosecond stamps stay exact integers
    assert PV.canonical({'t': big}) != PV.canonical({'t': big + 1})
    doc = {'schema': 'vantage.x/1', 'created': '2026-10-27T09:30:00+05:30', 'metrics': {'blind_m2': 0.01, 'seconds': 1.7},
           'note': 'run at 2026-10-27 09:30:00'}
    h = PV.result_digest(doc)
    # independent: SHA-256 of the canonical text written out by hand
    want = hashlib.sha256(b'{"created":"<time>","metrics":{"blind_m2":0.01,"seconds":null},'
                          b'"note":"run at <time>","schema":"vantage.x/1"}').hexdigest()
    assert h == want
    later = dict(doc, created='2027-01-01T00:00:00Z', metrics={'blind_m2': 0.01, 'seconds': 9.9})
    assert PV.result_digest(later) == h                                    # stamps and computation times do not count
    assert PV.result_digest(dict(doc, provenance={'x': 1})) == h
    assert PV.result_digest(dict(doc, metrics={'blind_m2': 0.0100001, 'seconds': 1.7})) != h
    p = _rig()
    stamped = PV.stamp(doc, p)
    assert list(stamped)[-1] == 'provenance' and 'provenance' not in doc
    assert PV.verify(stamped) == {'ok': True, 'recorded': h, 'actual': h}
    stamped['metrics'] = {'blind_m2': 0.02}
    assert PV.verify(stamped)['ok'] is False
    assert PV.verify({'a': 1})['ok'] is None


def test_the_project_content_hash():
    a, b = _rig(), _rig()
    assert a.variant.uid != b.variant.uid                                 # fresh random uids every time ...
    da, db = PV.project_digest(a), PV.project_digest(b)
    assert da == db                                                       # ... yet the same content
    assert set(da['sections']) == {'name', 'vehicle', 'mesh', 'library', 'variants', 'roi', 'analysis', 'targets',
                                   'obstacles', 'rig', 'ecu', 'photos', 'frame', 'streams', 'fleet'}
    # what does not count: view toggles, the log, times, the active variant, where the photos and body came from
    c = _rig()
    c.analysis.heatmap, c.analysis.live = 'density', False
    c.add_log('edit', 'moved F')
    c.saved_at, c.created = 123.0, 456.0
    c.active_uid = c.variants[0].uid
    c.photos.source, c.photos.current = '/somewhere/else/photos.zip', 3
    c.vehicle.source = '/another/computer/body.obj'
    c.analysis.assistant = {'brief': 'a data rig for highway driving'}
    for v in c.variants:
        v.created += 1000
    assert PV.project_digest(c) == da
    # an int and the same float are the same number (a reader may turn 2 into 2.0)
    c.roi.distance = 2
    assert PV.project_digest(c)['sha256'] == da['sha256']
    # what counts, and in which section
    d = _rig()
    d.variants[1].cameras[0].x += 0.001
    dd = PV.project_digest(d)
    assert dd['sha256'] != da['sha256'] and [k for k in da['sections'] if da['sections'][k] != dd['sections'][k]] == \
        ['variants']
    assert PV.variant_digest(d.variants[1]) != PV.variant_digest(a.variants[1])
    assert [PV.variant_digest(x) for i, x in enumerate(d.variants) if i != 1] == \
        [PV.variant_digest(x) for i, x in enumerate(a.variants) if i != 1]           # the other variants keep theirs
    e = _rig()
    e.analysis.cell_m = 0.04
    e.lenses['h05r2'].focal_mm += 0.01
    de = PV.project_digest(e)
    assert {k for k in da['sections'] if da['sections'][k] != de['sections'][k]} == {'analysis', 'library'}
    # a variant's hash does not depend on its place in the project
    f = _rig()
    f.variants.reverse()
    assert PV.variant_digest(f.variants[0]) == PV.variant_digest(a.variants[-1])
    assert PV.project_digest(f)['sha256'] != da['sha256']                 # the order of the variants is content
    # a password in a stream URL never counts (nor appears)
    g, h = _rig(), _rig()
    g.streams = {'F': 'rtsp://cam:secret1@10.0.0.5/stream'}
    h.streams = {'F': 'rtsp://cam:other2@10.0.0.5/stream'}
    assert PV.project_digest(g) == PV.project_digest(h) != da


def test_saving_and_reading_keeps_the_hash_but_for_the_32_bit_vertices(tmp_path):
    a = _rig()
    da = PV.project_digest(a)
    a.save(str(tmp_path / 'a.vantage'))
    b = Project.load(str(tmp_path / 'a.vantage'))
    db = PV.project_digest(b)
    assert [k for k in da['sections'] if da['sections'][k] != db['sections'][k]] == ['mesh']
    assert b.vehicle.mesh.vertices.dtype == np.float64 and np.array_equal(
        b.vehicle.mesh.vertices, a.vehicle.mesh.vertices.astype(np.float32).astype(np.float64))
    b.vehicle.mesh.vertices = a.vehicle.mesh.vertices.copy()
    assert PV.project_digest(b) == da                                      # the rest of the file reads back exactly
    b2 = Project.load(str(tmp_path / 'a.vantage'))
    b2.save(str(tmp_path / 'b.vantage'))
    assert PV.project_digest(Project.load(str(tmp_path / 'b.vantage'))) == db   # from the first save on: stable
    # the mesh hash, written out independently
    m = b2.vehicle.mesh
    h = hashlib.sha256()
    for name in ('vertices', 'faces', 'face_part', 'face_zone'):
        arr = np.ascontiguousarray(getattr(m, name))
        h.update(f'{name}:{arr.dtype.str}:{arr.shape}\n'.encode())
        h.update(arr.tobytes())
    assert PV.mesh_digest(m) == h.hexdigest()


# ============================================================================ the block
def test_the_block_names_versions_settings_and_seeds():
    p = _rig()
    b = PV.block(p, {'schema': 'vantage.observability/1', 'x': 1}, when=FIXED_NOW)
    assert b['schema'] == PV.SCHEMA and b['study'] == 'observability' and b['interface'] == 'app'
    assert b['created_utc'] == '2026-10-27T04:00:00Z'
    v = b['vantage']
    assert v['version'] == __version__ and re.fullmatch(r'[0-9a-f]{64}', v['code_sha256'])
    if os.path.exists(os.path.join(ROOT, '.git')) and shutil.which('git'):
        head = subprocess.run(['git', '-C', ROOT, 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
        assert v['git']['commit'] == head and isinstance(v['git']['dirty'], bool)
    r = b['runtime']
    import cv2
    import scipy
    assert r['numpy'] == np.__version__ and r['scipy'] == scipy.__version__ and r['opencv'] == cv2.__version__
    assert r['python'] == '.'.join(map(str, sys.version_info[:3])) and r['qt']
    assert b['variant'] == {'name': p.variant.name, 'index': p.variants.index(p.variant) + 1, 'uid': p.variant.uid,
                            'sha256': PV.variant_digest(p.variant)}
    assert b['project']['sha256'] == PV.project_digest(p)['sha256'] and b['project']['file'] is None
    s = b['settings']
    assert s['analysis']['cell_m'] == p.analysis.cell_m and s['roi']['distance'] == p.roi.distance
    from vantage.core import observability as OB
    assert s['observability'] == json.loads(json.dumps(OB.settings_of(p).to_dict()))   # every default filled in
    assert b['seeds'] == {'observability.seed': OB.settings_of(p).seed}
    p.analysis.lidar_calib = {'seed': 42}
    assert PV.block(p, {'schema': 'vantage.lidar_camera_calibration/1'})['seeds'] == {'lidar_calib.seed': 42}
    assert PV.block(p, {'schema': 'vantage.dataset/1'})['seeds'] == {'dataset.seed': 1}
    # every study's settings resolve through its settings class (not the stored dict)
    for key in PV._SETTINGS:
        assert PV.effective_settings(p, key), key
    assert set(PV.study_settings(p, 'perception_range')) == {'analysis', 'roi', 'perception', 'weather'}
    # a result of no project (a validation run): the environment, the time and the result's hash
    doc = PV.stamp({'schema': 'vantage.validation/1', 'checks': [{'error': 1e-9}]}, None, when=FIXED_NOW)
    pv = doc['provenance']
    assert pv['project'] is None and pv['variant'] is None and pv['settings'] == {} and PV.verify(doc)['ok']
    assert pv['study'] == 'validation' and pv['vantage']['version'] == __version__
    # a project without variants gets none made for it
    q = samples.blank_project()
    q.variants = []
    assert PV.block(q, {'schema': 'vantage.t/1'})['variant'] is None and q.variants == []


def test_scrub_takes_out_every_path_of_this_machine(tmp_path):
    home = os.path.expanduser('~')
    o = PV.scrub({'path': str(tmp_path / 'logs' / 'drive.mcap'), 'source': 'template:tigor',
                  'note': f'read {home}/data/a.csv and https://www.e-consystems.com/a/b.asp at 3 m/s, 1/2.42"',
                  'win': r'C:\Users\someone\rig\rig.vantage', 'cwd': os.getcwd(), 'h': home,
                  'list': [str(tmp_path / 'x.json'), 'plain'], 'units': 'px/cm'})
    assert o['path'] == 'drive.mcap' and o['source'] == 'template:tigor'
    assert o['note'] == 'read a.csv and https://www.e-consystems.com/a/b.asp at 3 m/s, 1/2.42"'
    assert o['win'] == 'rig.vantage' and o['list'] == ['x.json', 'plain'] and o['units'] == 'px/cm'
    assert home not in json.dumps(o) and os.getcwd() not in json.dumps(o) and str(tmp_path) not in json.dumps(o)
    assert PV.reduce_path('/a/b/c.json') == 'c.json' and PV.reduce_path('C:\\a\\b\\') == 'b'
    assert PV.reduce_path('-') == '-' and PV.reduce_path('name') == 'name'


@pytest.mark.parametrize('text', [
    '{\n "schema": "vantage.t/1",\n "a": [\n  1,\n  2.5\n ],\n "b": {\n  "c": null\n }\n}\n',
    '{\n  "schema": "vantage.t/1",\n  "a": [1, 2.5],\n  "b": {"c": null}\n}',
    '{\r\n\t"schema": "vantage.t/1",\r\n\t"a": 1\r\n}\r\n',
    '{"schema": "vantage.t/1", "a": [1, 2.5]}',
    '{"schema":"vantage.t/1","a":[1,2.5]}',
    '{}\n',
])
def test_stamping_a_json_file_keeps_every_other_byte(text):
    p = _rig()
    out = PV.stamp_bytes(text.encode(), p, when=FIXED_NOW).decode()
    body = text.rstrip()[:-1].rstrip()
    assert out.startswith(body) and out.endswith(text[len(text.rstrip()):])
    doc, back = json.loads(text), json.loads(out)
    assert list(back)[-1] == 'provenance' and {k: v for k, v in back.items() if k != 'provenance'} == doc
    assert PV.verify(back)['ok'] is True
    again = PV.stamp_bytes(out.encode(), p, when=FIXED_NOW).decode()        # an existing block is replaced
    assert json.loads(again) == back
    assert PV.stamp_bytes(b'[{"a": 1}]', p) == b'[{"a": 1}]'                # a list (a nuScenes table) stays
    assert PV.stamp_bytes(b'{not json', p) == b'{not json'


# ============================================================================ exports
EXPORT_IDS = ['extrinsics_json', 'jetson_json', 'nuscenes_calib', 'kitti_calib', 'recording_budget', 'power_budget',
              'sync_plan', 'sensor_coverage', 'guidelines', 'gnss_imu']


def test_every_exported_json_object_carries_its_provenance(proj, tmp_path, monkeypatch):
    monkeypatch.setattr(ex, '_now', lambda: FIXED_NOW)
    p = Project.load(proj)
    eng = Engine()
    res = eng.analyse(p)
    opts = ex.ExportOptions(out_dir=str(tmp_path / 'x'), engine=eng, units='mm', frame='front_axle')
    paths = ex.export(EXPORT_IDS, p, res, opts)
    ctx = ex._context(p, res, ex.ExportOptions(engine=eng, units='mm', frame='front_axle'))
    raw = {rel: data for fid in EXPORT_IDS for rel, data in ex._render_files(fid, ctx)}
    def untimed(data):                       # a module's own wall-clock stamp (gnss) may tick between the renders
        try:
            return PV._TS.sub('<t>', data.decode('utf-8'))
        except UnicodeDecodeError:
            return data
    seen = 0
    for path in paths:
        rel = os.path.relpath(path, tmp_path / 'x').replace(os.sep, '/')
        data = open(path, 'rb').read()
        if not rel.endswith('.json'):
            assert untimed(data) == untimed(raw[rel]), rel                  # other files untouched
            continue
        doc = json.loads(data)
        if not isinstance(doc, dict):
            assert data == raw[rel], rel                                    # nuScenes tables are lists: untouched
            continue
        seen += 1
        text, before = untimed(data), untimed(raw[rel])
        assert text.startswith(before.rstrip()[:-1].rstrip()), rel          # only the block added at the end
        pv = doc['provenance']
        assert list(doc)[-1] == 'provenance' and PV.verify(doc)['ok'], rel
        assert pv['interface'] == 'app' and pv['command'] is None and pv['created_utc'] == '2026-10-27T04:00:00Z'
        fid = pv['output']['format']
        assert fid in EXPORT_IDS and pv['output']['name'] == rel and pv['study'] == fid
        assert pv['settings']['export'] == {'format': fid, 'frame': 'front_axle', 'units': 'mm', 'angles': 'deg'}
        assert pv['project']['file'] == 'rig.vantage' and pv['variant']['name'] == p.variant.name
        assert pv['project']['sha256'] == PV.project_digest(p)['sha256']
        blob = json.dumps(pv)
        assert os.path.dirname(proj) not in blob and str(tmp_path) not in blob
    assert seen >= 9                                       # extrinsics, jetson, 2 budgets, sync, coverage, guidelines …
    # the preview shows exactly what is written
    rel, text, size = ex.preview('extrinsics_json', p, res, opts)
    assert text == open(os.path.join(tmp_path, 'x', 'extrinsics.json'), encoding='utf-8').read()


def test_stamped_files_still_read_back(proj, tmp_path):
    """Files Vantage writes and reads (suspension data, extrinsics into the fleet, unit calibrations) read the same
    with their provenance block."""
    from vantage.core import calib
    from vantage.core import fleet as FL
    from vantage.core import suspension as SU
    p = Project.load(proj)
    eng = Engine()
    ex.export(['suspension_load_cases', 'extrinsics_json'], p, eng.analyse(p),
              ex.ExportOptions(out_dir=str(tmp_path), engine=eng))
    sd = tmp_path / 'suspension' / 'suspension_data.json'
    assert 'provenance' in _load(sd)
    st, what, warns = SU.load_file(str(sd), SU.settings_for(p))
    assert what == 'suspension file' and not warns
    assert {k: v for k, v in st.to_dict().items() if k not in ('source', 'note')} == \
        {k: v for k, v in SU.settings_for(p).to_dict().items() if k not in ('source', 'note')}
    doc = FL.import_files(p, 'V9', [str(tmp_path / 'extrinsics.json')])
    assert doc['imported'] and not doc['errors']
    cam = p.variant.cameras[0]
    cam.calib = calib.clean_record(calib.simulated_unit(p.module_for(cam), p.nominal_lens_for(cam), seed=3))
    f = tmp_path / 'calib.json'
    f.write_text(calib.to_json(p, [cam]))
    assert PV.stamp_path(str(f), p) and 'provenance' in _load(f)
    recs = calib.read_file(str(f))
    assert len(recs) == 1 and recs[0]['fx'] == cam.calib['fx'] and recs[0]['hint'] == cam.tag
    # a module's own export (a page's Export button) stamped afterwards
    from vantage.core import recording as RB
    paths = PV.stamp_paths(RB.export(RB.budget(p), p, str(tmp_path / 'rb')), p, str(tmp_path / 'rb'))
    doc = _load(next(x for x in paths if x.endswith('.json')))
    assert doc['provenance']['output'] == {'name': 'recording_budget.json', 'in_folder': True}
    assert doc['provenance']['study'] == 'recording_budget' and PV.verify(doc)['ok']
    assert all(not x.endswith('.csv') or 'provenance' not in open(x).read() for x in paths)


# ============================================================================ batch outputs
def test_batch_outputs_carry_the_command_with_file_names_only(proj, tmp_path):
    home = os.path.expanduser('~')
    work = tmp_path / 'deep' / 'run'
    p = Project.load(proj)
    p.analysis.drivelog = {'path': str(tmp_path / 'logs' / 'drive.mcap')}           # settings that hold paths
    p.analysis.asbuilt = {'path': str(tmp_path / 'logs' / 'asbuilt')}
    p.streams = {'F': 'rtsp://cam:topsecret@10.0.0.5/stream'}
    src = work / 'in' / 'rig with spaces.vantage'
    src.parent.mkdir(parents=True)
    p.save(str(src))
    susp = work / 'in' / 'suspension.json'
    from vantage.core import suspension as SU
    susp.write_text(SU.dumps(SU.settings_for(p)))
    out = work / 'out'
    code = _main(src, '--variant', p.variant.name, '--perception-range', out / 'p.json', '--observability',
                 out / 'obs', '--export', out / 'x', '--formats', 'extrinsics_json,nuscenes_calib', '--suspension-data',
                 susp, '--suspension', out / 'susp.json', '--units', 'mm')
    assert code in (0, 2)
    files = sorted(_json_files(out))
    assert len(files) == 6                  # and the two nuScenes tables (lists)
    needles = [str(tmp_path), str(work), home, os.getcwd(), ROOT, 'topsecret']
    stamped = 0
    for f in files:
        doc = _load(f)
        if not isinstance(doc, dict):
            continue
        stamped += 1
        pv = doc['provenance']
        blob = json.dumps(pv, ensure_ascii=False)
        for n in needles:
            assert n not in blob, (f, n)
        assert pv['interface'] == 'batch' and PV.verify(doc)['ok']
        cmd = pv['command']
        assert cmd[:2] == ['vantage', 'rig with spaces.vantage']
        assert cmd[cmd.index('--variant') + 1] == p.variant.name                   # a name, kept whole
        assert cmd[cmd.index('--perception-range') + 1] == 'p.json' and cmd[cmd.index('--observability') + 1] == 'obs'
        assert cmd[cmd.index('--suspension-data') + 1] == 'suspension.json' and '--units' in cmd
        assert not any(os.sep in c or '/' in c for c in cmd)
        assert pv['inputs'] == [{'name': 'suspension.json', 'kind': 'file', 'bytes': susp.stat().st_size, 'files': 1,
                                 'sha256': hashlib.sha256(susp.read_bytes()).hexdigest(), 'hashed': 'full',
                                 'option': '--suspension-data'}]
        assert pv['project']['file'] == 'rig with spaces.vantage'
    assert stamped == 4                     # p.json, susp.json, obs/observability.json, x/extrinsics.json
    assert _load(out / 'p.json')['provenance']['output'] == {'option': '--perception-range', 'name': 'p.json',
                                                              'in_folder': False}
    obs = _load(out / 'obs' / 'observability.json')['provenance']
    assert obs['output'] == {'option': '--observability', 'name': 'observability.json', 'in_folder': True}
    assert obs['seeds'] == {'observability.seed': obs['settings']['observability']['seed']}
    x = _load(out / 'x' / 'extrinsics.json')['provenance']
    assert x['output'] == {'option': '--export', 'format': 'extrinsics_json', 'name': 'extrinsics.json',
                           'in_folder': True} and x['settings']['export']['units'] == 'mm'
    s = _load(out / 'susp.json')['provenance']
    # the suspension data changed the project before the study: both hashes are named
    assert s['project']['input_sha256'] == PV.project_digest(p)['sha256'] != s['project']['sha256']
    assert isinstance(_load(out / 'x' / 'nuscenes' / 'sensor.json'), list)


def test_the_on_change_command_is_never_recorded(proj, tmp_path):
    from vantage import batch
    from test_advanced import _args
    src = tmp_path / 'w' / 'rig.vantage'
    src.parent.mkdir()
    shutil.copy(proj, src)
    marker = 'MARKER-7731'
    a = _args(watch=str(src), on_change=f'exit 0 # {marker}', watch_runs=1, formats='extrinsics_json')
    assert batch.run(a) == 0
    f = src.parent / 'exports' / 'automation' / 'extrinsics.json'
    text = f.read_text('utf-8')
    pv = json.loads(text)['provenance']
    assert marker not in text and '--on-change' not in pv['command'] and '--watch' in pv['command']
    assert pv['command'][pv['command'].index('--watch') + 1] == 'rig.vantage' and PV.verify(json.loads(text))['ok']


# ============================================================================ --reproduce
def _reproduce(*argv):
    return _main(*argv)


def test_reproduce_a_batch_output_exactly(proj, tmp_path):
    out = tmp_path / 'r'
    assert _main(proj, '--perception-range', out / 'p.json', '--sensor-coverage', out / 's.json') == 0
    rep = out / 'rep.json'
    assert _reproduce(proj, '--reproduce', out / 'p.json', rep) == 0
    r = _load(rep)
    assert r['schema'] == PV.REPRODUCE_SCHEMA and r['verdict'] == 'reproduces' and r['same']['result_sha256']
    c = r['counts']
    assert c['numbers'] > 5000 and c['exact'] == c['numbers'] and c['differs'] == c['within'] == 0
    assert len(r['numbers']) == c['numbers'] and all(row[3] == 'exact' for row in r['numbers'])
    assert r['integrity']['intact'] and r['project_given']['same_as_recorded'] and r['reasons'] == []
    assert r['command'] == ['vantage', 'rig.vantage', '--perception-range', 'p.json']
    assert r['provenance']['output']['option'] == '--reproduce' and PV.verify(r)['ok']
    # moved elsewhere with the project beside it: found by its name
    other = tmp_path / 'elsewhere'
    other.mkdir()
    shutil.copy(out / 's.json', other / 's.json')
    shutil.copy(proj, other / 'rig.vantage')
    assert _reproduce('--reproduce', other / 's.json', other / 'rep.json') == 0
    assert _load(other / 'rep.json')['verdict'] == 'reproduces'
    # without it: cannot be tried
    alone = tmp_path / 'alone'
    alone.mkdir()
    shutil.copy(out / 's.json', alone / 's.json')
    old = os.getcwd()
    os.chdir(alone)
    try:
        assert _reproduce('--reproduce', alone / 's.json', alone / 'rep.json') == 4
    finally:
        os.chdir(old)
    r = _load(alone / 'rep.json')
    assert r['verdict'] == 'cannot_reproduce' and 'rig.vantage' in r['message']


def test_reproduce_reports_every_number_that_differs_and_why(proj, tmp_path):
    out = tmp_path / 'r'
    assert _main(proj, '--recording-budget', out / 'b.json') == 0
    doc = _load(out / 'b.json')
    target = next(k for k, v in PV._leaves(PV.neutral(doc)) if PV._isnum(v) and abs(v) > 1)
    # change one number by a little: the file no longer holds what Vantage wrote, and the number differs
    def edit(d, path, f):
        parts = re.findall(r'[^.\[\]]+|\[\d+\]', path)
        o = d
        for k in parts[:-1]:
            o = o[int(k[1:-1])] if k.startswith('[') else o[k]
        last = parts[-1]
        if last.startswith('['):
            o[int(last[1:-1])] = f(o[int(last[1:-1])])
        else:
            o[last] = f(o[last])
    tampered = copy.deepcopy(doc)
    edit(tampered, target, lambda v: v * 1.001 + 1e-6)
    (out / 't.json').write_text(json.dumps(tampered, indent=1))
    assert _reproduce(proj, '--reproduce', out / 't.json', out / 'rep.json') == 6
    r = _load(out / 'rep.json')
    assert r['verdict'] == 'does_not_reproduce' and r['counts']['differs'] == 1
    row = next(x for x in r['numbers'] if x[3] != 'exact')
    assert row[0] == target and row[3] == 'differs' and row[4] == pytest.approx(abs(row[1] - row[2]))
    assert r['integrity']['intact'] is False and any('changed after Vantage wrote it' in x for x in r['reasons'])
    assert target in r['message']
    # within a tolerance on that quantity (relative, then by its key's name), it reproduces
    key = re.sub(r'\[\d+\]$', '', target.rsplit('.', 1)[-1])
    assert _reproduce(proj, '--reproduce', out / 't.json', out / 'rep2.json', '--tolerance', '0.2%') == 0
    assert _load(out / 'rep2.json')['verdict'] == 'reproduces_within_tolerance'
    assert _reproduce(proj, '--reproduce', out / 't.json', out / 'rep3.json', '--tolerance', f'{key}=0.2%',
                      'nothing_like_it=1') == 0
    assert _reproduce(proj, '--reproduce', out / 't.json', out / 'rep4.json', '--tolerance', 'nothing_like_it=1') == 6
    # an edited version in the provenance: the numbers still agree, and the report says what differs
    other = copy.deepcopy(doc)
    other['provenance']['vantage']['version'] = '0.1.0'
    other['provenance']['runtime']['numpy'] = '1.26.0'
    (out / 'v.json').write_text(json.dumps(other, indent=1))
    assert _reproduce(proj, '--reproduce', out / 'v.json', out / 'rep5.json') == 0
    r = _load(out / 'rep5.json')
    assert r['verdict'] == 'reproduces' and r['integrity']['intact'] is True
    assert any('a different Vantage version: 0.1.0 then' in x for x in r['reasons'])
    assert any('numpy 1.26.0 →' in x for x in r['reasons'])
    # another project: a camera moved
    q = Project.load(proj)
    q.variant.cameras[0].z += 0.1
    q.save(str(tmp_path / 'moved.vantage'))
    assert _main(tmp_path / 'moved.vantage', '--sensor-coverage', out / 'm.json') == 0
    assert _reproduce(proj, '--reproduce', out / 'm.json', out / 'rep6.json') == 6
    r = _load(out / 'rep6.json')
    assert r['project_given']['same_as_recorded'] is False and r['counts']['differs'] > 0
    assert any(x == 'the project differs: variants' for x in r['reasons'])
    assert any('the variant differs' in x for x in r['reasons'])
    # a bad tolerance, no provenance, a live measurement
    with pytest.raises(SystemExit):
        _reproduce(proj, '--reproduce', out / 'b.json', '--tolerance', 'x=abc')
    (out / 'plain.json').write_text('{"schema": "vantage.recording_budget/1"}')
    assert _reproduce(proj, '--reproduce', out / 'plain.json') == 4
    live = copy.deepcopy(doc)
    live['provenance']['output']['option'] = '--rig-health'
    (out / 'live.json').write_text(json.dumps(live))
    assert _reproduce(proj, '--reproduce', out / 'live.json', out / 'rep7.json') == 4
    assert 'live streams' in _load(out / 'rep7.json')['message']


def test_reproduce_folder_outputs_exports_and_inputs(proj, tmp_path):
    out = tmp_path / 'r'
    p = Project.load(proj)
    from vantage.core import suspension as SU
    susp = out / 'data' / 'suspension.json'
    susp.parent.mkdir(parents=True)
    st = SU.file_doc(SU.settings_for(p))
    susp.write_text(json.dumps(st))
    assert _main(proj, '--critical-targets', out / 'crit', '--export', out / 'x', '--formats',
                 'power_budget,extrinsics_json', '--angles', 'rad') in (0, 2)
    assert _main(proj, '--suspension-data', susp, '--suspension', out / 'data' / 'susp.json') in (0, 2)
    for f in (out / 'crit' / 'critical_targets.json', out / 'x' / 'power_budget' / 'power_budget.json',
              out / 'x' / 'extrinsics.json'):
        assert _reproduce(proj, '--reproduce', f, out / 'rep.json') == 0, f
        r = _load(out / 'rep.json')
        assert r['verdict'] == 'reproduces' and r['same']['result_sha256'], f
    assert _load(out / 'rep.json')['command'][-6:] == ['--export', 'out', '--formats', 'extrinsics_json', '--angles',
                                                       'rad']
    # the input sits next to the result: found and hashed again
    assert _reproduce(proj, '--reproduce', out / 'data' / 'susp.json', out / 'rep.json') == 0
    r = _load(out / 'rep.json')
    assert r['verdict'] == 'reproduces' and r['same']['inputs']
    # a different input: the numbers move and the report names it
    st2 = copy.deepcopy(st)
    flat = [k for k, v in PV._leaves(st2) if PV._isnum(v) and 'spring_rate' in k]
    assert flat
    obj = st2
    parts = re.findall(r'[^.\[\]]+|\[\d+\]', flat[0])
    for k in parts[:-1]:
        obj = obj[int(k[1:-1])] if k.startswith('[') else obj[k]
    k = parts[-1]
    if k.startswith('['):
        obj[int(k[1:-1])] *= 1.5
    else:
        obj[k] *= 1.5
    susp.write_text(json.dumps(st2))
    assert _reproduce(proj, '--reproduce', out / 'data' / 'susp.json', out / 'rep.json') == 6
    r = _load(out / 'rep.json')
    assert any(x == 'different input files: suspension.json' for x in r['reasons'])
    # and when it is gone, the reproduction cannot be tried
    susp.unlink()
    assert _reproduce(proj, '--reproduce', out / 'data' / 'susp.json', out / 'rep.json') == 4
    assert 'suspension.json' in _load(out / 'rep.json')['message']


def test_reproduce_an_export_made_in_the_app(proj, tmp_path, monkeypatch):
    p = Project.load(proj)
    p.active_uid = p.variants[1].uid                                        # not the variant the file opens with
    eng = Engine()
    res = eng.analyse(p)
    ex.export(['sync_plan', 'recording_budget'], p, res, ex.ExportOptions(out_dir=str(tmp_path / 'x'), engine=eng,
                                                                          frame='front_bumper'))
    f = tmp_path / 'x' / 'sync_plan' / 'sync_plan.json'
    pv = _load(f)['provenance']
    assert pv['interface'] == 'app' and pv['variant']['index'] == 2
    assert _reproduce(proj, '--reproduce', f, tmp_path / 'rep.json') == 0
    r = _load(tmp_path / 'rep.json')
    assert r['verdict'] == 'reproduces' and r['command'] == [
        'vantage', 'rig.vantage', '--export', 'out', '--formats', 'sync_plan', '--variant', '2', '--frame',
        'front_bumper']
    # the app's project had unsaved edits: the report says the project given is not the one exported
    p.variants[1].cameras[0].yaw += 3
    ex.export(['extrinsics_json'], p, eng.analyse(p), ex.ExportOptions(out_dir=str(tmp_path / 'y'), engine=eng))
    code = _reproduce(proj, '--reproduce', tmp_path / 'y' / 'extrinsics.json', tmp_path / 'rep.json')
    r = _load(tmp_path / 'rep.json')
    assert r['project_given']['same_as_recorded'] is False
    assert code == 6 and any(x.startswith('the project differs: variants') for x in r['reasons'])


def test_compare_and_tolerances_on_known_documents():
    a = {'schema': 'vantage.t/1', 'created': '2026-01-01T00:00:00Z', 'metrics': {'blind_m2': 1.0, 'cov_pct': 50.0,
                                                                                    'seconds': 2.0},
         'cams': [{'tag': 'F', 'id': '1a2b3c4d5e:data', 'd': 3.0}, {'tag': 'R', 'id': 'ffffeeee00:data', 'd': 4.0}],
         'flag': True, 'nan': float('nan')}
    b = copy.deepcopy(a)
    b['created'] = '2027-05-05T05:05:05Z'                                   # a time stamp: not compared
    b['metrics']['seconds'] = 9.0                                           # a computation time: not compared
    b['cams'][0]['id'], b['cams'][1]['id'] = '0000000000:data', '9999999999:data'   # other uids, same order
    r = PV.compare(a, b)
    assert r['verdict'] == 'reproduces' and r['counts']['numbers'] == 5 and r['counts']['texts_changed'] == 0
    b['metrics']['blind_m2'] = 1.004
    b['metrics']['cov_pct'] = 50.1                                          # 0.2 % off; blind_m2 0.4 %
    r = PV.compare(a, b)
    assert r['verdict'] == 'does_not_reproduce' and r['counts']['differs'] == 2
    assert [x[0] for x in r['worst']][:2] == ['metrics.blind_m2', 'metrics.cov_pct']          # by relative error
    tol = PV.parse_tolerances(['blind_m2=0.005', '*_pct=0.5%'])
    assert tol == [('blind_m2', 0.005, False), ('*_pct', 0.005, True)]
    r = PV.compare(a, b, tol)
    assert r['verdict'] == 'reproduces_within_tolerance' and r['counts']['within'] == 2
    assert PV.compare(a, b, PV.parse_tolerances(['blind_m2=0.003', '1%']))['counts']['differs'] == 1
    assert PV.tolerance_for('cams[1].d', PV.parse_tolerances(['cams[*].d=0.1']))[0] == 0.1
    assert PV.tolerance_for('x.y', PV.parse_tolerances(['metrics.*=1', '2']))[0] == 2
    b2 = copy.deepcopy(a)
    b2['cams'][1]['tag'] = 'RR'
    del b2['flag']
    b2['extra'] = 1
    r = PV.compare(a, b2)
    assert r['texts'] == [['cams[1].tag', 'R', 'RR']] and r['missing'] == ['flag'] and r['extra'] == ['extra']
    b3 = copy.deepcopy(a)
    b3['cams'][0]['id'] = 'ffffeeee00:data'                                 # two uids swapped: not the same result
    assert PV.compare(a, b3)['counts']['texts_changed'] == 1
    for bad in ('x=abc', '-1', '=', 'inf'):
        with pytest.raises(ValueError):
            PV.parse_tolerances([bad])


# ============================================================================ determinism
RNG_OK = {'default_rng', 'Generator', 'SeedSequence', 'PCG64', 'PCG64DXSM', 'Philox', 'SFC64', 'MT19937'}
SEEDED_CLASSES = {'SeedSequence', 'PCG64', 'PCG64DXSM', 'Philox', 'SFC64', 'MT19937', 'default_rng'}
CV_USES_RNG = {'kmeans', 'randu', 'randn', 'randShuffle', 'findChessboardCornersSB', 'findChessboardCornersSBWithMeta',
               'solvePnPRansac', 'findHomography', 'estimateAffine2D', 'estimateAffinePartial2D', 'estimateAffine3D',
               'findFundamentalMat', 'findEssentialMat', 'recoverPose'}
SCIPY_SEEDED = {'differential_evolution', 'dual_annealing', 'basinhopping', 'Sobol', 'LatinHypercube', 'Halton',
                'PoissonDisk', 'MultivariateNormalQMC', 'bootstrap', 'permutation_test', 'monte_carlo_test', 'rvs',
                'shgo'}
IDENTITIES = {('core/project.py', 'uuid4'), ('core/automation.py', 'token_hex')}   # identities, never numbers
# calls whose result is thrown away may be listed here; none are now (the SB detector's warm-up and the self-test's
# check seed OpenCV's generator like every real call)
CV_DISCARDED: set = set()


def _dotted(node):
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return '.'.join(reversed(parts))


def _audit(src: str, rel: str) -> list:
    """Every use of a random number generator in a module's source that is not seeded."""
    tree = ast.parse(src)
    out = []
    funcs = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    scipy_names = {'scipy'}                                  # names bound to scipy modules or functions
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or '').startswith('scipy'):
            scipy_names |= {a.asname or a.name for a in node.names}
        elif isinstance(node, ast.Import):
            scipy_names |= {(a.asname or a.name).split('.')[0] for a in node.names if a.name.startswith('scipy')}

    def seeds_cv_first(call):
        for f in funcs:
            if f.lineno <= call.lineno <= (f.end_lineno or f.lineno):
                if (rel, f.name) in CV_DISCARDED:
                    return True
                if any(isinstance(n, ast.Call) and _dotted(n.func) == 'cv2.setRNGSeed' and n.lineno < call.lineno
                       for n in ast.walk(f)):
                    return True
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import) and any(a.name == 'random' for a in node.names):
            out.append(f'{rel}:{node.lineno} imports the random module (a global generator)')
        if isinstance(node, ast.ImportFrom) and node.module == 'random':
            out.append(f'{rel}:{node.lineno} imports from the random module')
        if not isinstance(node, ast.Call):
            continue
        name = _dotted(node.func)
        last = name.rsplit('.', 1)[-1]
        m = re.fullmatch(r'(?:np|numpy)\.random\.(\w+)', name)
        if m and m.group(1) not in RNG_OK:
            out.append(f'{rel}:{node.lineno} {name}: numpy\'s global generator')
        if (m or name in SEEDED_CLASSES) and last in SEEDED_CLASSES:
            if not node.args and not node.keywords or (node.args and isinstance(node.args[0], ast.Constant)
                                                       and node.args[0].value is None):
                out.append(f'{rel}:{node.lineno} {name}() without a seed')
        if name.startswith('cv2.') and last in CV_USES_RNG and not seeds_cv_first(node):
            out.append(f'{rel}:{node.lineno} {name} uses OpenCV\'s generator without cv2.setRNGSeed first')
        if last in SCIPY_SEEDED and name.split('.')[0] in scipy_names and \
                not any(k.arg in ('seed', 'rng', 'random_state') for k in node.keywords):
            out.append(f'{rel}:{node.lineno} {name} without seed=, rng= or random_state=')
        if last in ('uuid1', 'uuid4', 'token_hex', 'token_bytes', 'urandom', 'randbits') and (rel, last) not in IDENTITIES:
            out.append(f'{rel}:{node.lineno} {name}: random bytes')
    return out


def test_the_audit_finds_unseeded_generators():
    bad = ('import random\nimport numpy as np, cv2\nfrom scipy.stats import qmc\n'
           'a = np.random.rand(3)\nb = np.random.default_rng()\nc = np.random.default_rng(None)\n'
           'def f(x):\n    return cv2.kmeans(x, 2, None, None, 3, 0)\n'
           'def g(x):\n    cv2.setRNGSeed(1)\n    return cv2.findChessboardCornersSB(x, (9, 6))\n'
           's = qmc.Sobol(3)\nt = qmc.Sobol(3, seed=4)\nimport os\nk = os.urandom(4)\nr = np.random.default_rng(7)\n')
    found = _audit(bad, 'x.py')
    assert len(found) == 7, found
    assert any('rand: ' in x for x in found) and sum('without a seed' in x for x in found) == 2
    assert any('kmeans' in x for x in found) and not any('findChessboardCornersSB' in x for x in found)
    assert any('qmc.Sobol without' in x for x in found) and any('urandom' in x for x in found)


def test_no_unseeded_random_numbers_in_vantage():
    problems, seeded = [], 0
    for root, dirs, files in os.walk(PKG):
        dirs[:] = [d for d in dirs if d != '__pycache__']
        for f in files:
            if f.endswith('.py'):
                path = os.path.join(root, f)
                rel = os.path.relpath(path, PKG).replace(os.sep, '/')
                src = open(path, encoding='utf-8').read()
                problems += _audit(src, rel)
                seeded += len(re.findall(r'default_rng\(', src))
    assert not problems, '\n'.join(problems)
    assert seeded >= 25                                     # the audit read the code that draws random numbers


def test_seeded_studies_are_bit_for_bit_the_same(proj, tmp_path):
    from vantage.core import asbuilt as ABL
    from vantage.core import calib
    from vantage.core import dataset as DSET
    from vantage.core.synth import render_capture
    from vantage.core.tolerance import TolSettings, run_monte_carlo
    p = Project.load(proj)
    eng = Engine()
    cam = p.variant.cameras[0]
    assert np.array_equal(render_capture(p, eng, cam, 320, 256, seed=3), render_capture(p, eng, cam, 320, 256, seed=3))
    m, lens = p.module_for(cam), p.nominal_lens_for(cam)
    u1, u2 = calib.simulated_unit(m, lens, seed=5), calib.simulated_unit(m, lens, seed=5)
    assert u1 == u2
    v1, v2 = calib.board_views(m, lens, u1, n=3, seed=2, scale=0.25), calib.board_views(m, lens, u2, n=3, seed=2,
                                                                                         scale=0.25)
    assert all(a[0] == b[0] and np.array_equal(a[1], b[1]) for a, b in zip(v1, v2))
    rng = np.random.default_rng(0)
    t = rng.uniform(0, 2 * np.pi, 80)
    pts = np.c_[100 + 40 * np.cos(t), 80 + 40 * np.sin(t)] + rng.normal(0, 0.5, (80, 2))
    assert ABL.fit_circle(pts, seed=1)['r'] == ABL.fit_circle(pts, seed=1)['r']
    st = TolSettings(samples=40, correlate_pairs=True, cell=0.1)
    r1 = run_monte_carlo(eng, p, st, threads=4)
    r2 = run_monte_carlo(eng, p, st, threads=2)                             # the threads do not change a sample
    assert np.array_equal(r1.blind, r2.blind) and np.array_equal(r1.min_density, r2.min_density)
    assert np.array_equal(r1.seam_shift, r2.seam_shift)
    # the sample dataset: camera noise, lidar noise and drop-outs, radar clutter, IMU and GNSS errors, rendered on a
    # thread pool: every data file the same, byte for byte
    dirs = []
    for k in range(2):
        d = tmp_path / f'ds{k}'
        DSET.generate(p, eng, DSET.DatasetSettings.from_dict(p.analysis.dataset), str(d))
        dirs.append(d)
    files = sorted(os.path.relpath(os.path.join(r, f), dirs[0]) for r, _, fs in os.walk(dirs[0]) for f in fs)
    assert len(files) > 20
    for f in files:
        a, b = (dirs[0] / f).read_bytes(), (dirs[1] / f).read_bytes()
        if f in ('dataset.json', 'README.md'):                              # the manifest names its folder and time
            a, b = (x.replace(str(d).encode(), b'<out>') for x, d in ((a, dirs[0]), (b, dirs[1])))
            if f.endswith('.json'):
                assert PV.neutral(json.loads(a)) == PV.neutral(json.loads(b))
            continue
        assert a == b, f


def _renamed(order):
    """The sample with uids that sort with (order 1) or against (order -1) the cameras' and boards' order: the same
    rig, other identities (a sample or template gets fresh random uids every time it is built)."""
    p = samples.tigor_sample_project()
    for i, t in enumerate(p.targets):
        t.uid = f'{(i if order > 0 else 99 - i):02d}aaaaaaaa'
    v = p.variant
    new = {c.uid: f'{(i if order > 0 else 99 - i):02d}bbbbbbbb' for i, c in enumerate(v.cameras)}
    for c in v.cameras:
        c.uid, c.link_uid = new[c.uid], new.get(c.link_uid, c.link_uid)
    v.seams = {}
    return p


def test_the_tolerance_study_does_not_depend_on_the_uids():
    from vantage.core.tolerance import TolSettings, run_monte_carlo
    eng = Engine()
    a, b = (run_monte_carlo(eng, _renamed(o), TolSettings(samples=60, correlate_pairs=True, metric_seam=False),
                            threads=2) for o in (1, -1))
    assert np.array_equal(a.blind, b.blind) and np.array_equal(a.min_density, b.min_density)


def test_the_joint_solve_does_not_depend_on_the_uids():
    from vantage.core.install import InstallSettings, check_all
    from vantage.core.synth import render_capture
    eng = Engine()
    poses = []
    for order in (1, -1):
        p = _renamed(order)
        cams = p.variant.cameras
        imgs = {c.uid: render_capture(p, eng, c, 960, 768, seed=c.slot) for c in cams}
        res, jr = check_all(p, cams, imgs, InstallSettings(joint=True))
        assert jr is not None and jr.ok
        poses.append(np.array([np.r_[res[c.uid].pos, res[c.uid].yaw, res[c.uid].pitch, res[c.uid].roll]
                               for c in cams]))
    assert np.array_equal(poses[0], poses[1])


def test_two_processes_compute_the_same_numbers(proj, tmp_path):
    """The same project in two fresh processes with different string-hash seeds: every result identical (the result
    hashes agree, and so does every number and text but the time stamps)."""
    env = dict(os.environ, QT_QPA_PLATFORM='offscreen')
    outs = []
    for k, seed in enumerate(('1', '4242')):
        out = tmp_path / f'run{k}'
        env['PYTHONHASHSEED'] = seed
        r = subprocess.run([sys.executable, '-m', 'vantage', proj, '--analyse', out / 'a.json', '--perception-range',
                            out / 'p.json', '--sensor-coverage', out / 's.json', '--critical-targets', out / 'c.json',
                            '--weather', out / 'w.json', '--observability', out / 'o.json', '--dataset', out / 'ds'],
                           env=env, cwd=ROOT, capture_output=True, text=True, timeout=600)
        assert r.returncode in (0, 2), r.stderr[-2000:]
        outs.append(out)
    for name in ('a.json', 'p.json', 's.json', 'c.json', 'w.json', 'o.json'):
        a, b = _load(outs[0] / name), _load(outs[1] / name)
        assert a['provenance']['result_sha256'] == b['provenance']['result_sha256'], name
        assert PV.compare(a, b)['verdict'] == 'reproduces', name
        assert PV.neutral(a) == PV.neutral(b), name
    for root, _, fs in os.walk(outs[0] / 'ds'):
        for f in fs:
            if f in ('dataset.json', 'README.md'):
                continue
            q = os.path.join(root, f)
            assert open(q, 'rb').read() == open(q.replace(str(outs[0]), str(outs[1])), 'rb').read(), q


# ============================================================================ the project format
SCHEMA_PATH = os.path.join(ROOT, 'docs', 'project.schema.json')


def _validate(inst, schema, root=None, path='$') -> list:
    """A JSON Schema (draft 2020-12) validator for the keywords docs/project.schema.json uses, written for this test
    so it runs without the jsonschema package (which, when installed, checks the schema too)."""
    root = root or schema
    errs = []
    if '$ref' in schema:
        node = root
        for part in schema['$ref'].lstrip('#/').split('/'):
            node = node[part]
        errs += _validate(inst, node, root, path)
    types = schema.get('type')
    if types is not None:
        types = [types] if isinstance(types, str) else types
        def is_(t):
            return {'object': isinstance(inst, dict), 'array': isinstance(inst, list),
                    'string': isinstance(inst, str), 'boolean': isinstance(inst, bool), 'null': inst is None,
                    'number': isinstance(inst, (int, float)) and not isinstance(inst, bool),
                    'integer': (isinstance(inst, int) and not isinstance(inst, bool)) or
                               (isinstance(inst, float) and inst.is_integer())}[t]
        if not any(is_(t) for t in types):
            return errs + [f'{path}: {inst!r} is not {types}']
    if 'const' in schema and inst != schema['const']:
        errs.append(f'{path}: {inst!r} is not {schema["const"]!r}')
    if 'enum' in schema and inst not in schema['enum']:
        errs.append(f'{path}: {inst!r} is not one of {schema["enum"]}')
    if isinstance(inst, (int, float)) and not isinstance(inst, bool):
        if 'minimum' in schema and inst < schema['minimum']:
            errs.append(f'{path}: {inst} < {schema["minimum"]}')
        if 'maximum' in schema and inst > schema['maximum']:
            errs.append(f'{path}: {inst} > {schema["maximum"]}')
        if 'exclusiveMinimum' in schema and inst <= schema['exclusiveMinimum']:
            errs.append(f'{path}: {inst} <= {schema["exclusiveMinimum"]}')
        if 'exclusiveMaximum' in schema and inst >= schema['exclusiveMaximum']:
            errs.append(f'{path}: {inst} >= {schema["exclusiveMaximum"]}')
    if isinstance(inst, list):
        if 'minItems' in schema and len(inst) < schema['minItems']:
            errs.append(f'{path}: fewer than {schema["minItems"]} items')
        if 'maxItems' in schema and len(inst) > schema['maxItems']:
            errs.append(f'{path}: more than {schema["maxItems"]} items')
        if 'items' in schema:
            for i, x in enumerate(inst):
                errs += _validate(x, schema['items'], root, f'{path}[{i}]')
    if isinstance(inst, dict):
        for k in schema.get('required', ()):
            if k not in inst:
                errs.append(f'{path}: {k} is missing')
        if 'maxProperties' in schema and len(inst) > schema['maxProperties']:
            errs.append(f'{path}: more than {schema["maxProperties"]} keys')
        props = schema.get('properties', {})
        for k, v in inst.items():
            if k in props:
                errs += _validate(v, props[k], root, f'{path}.{k}')
            elif 'additionalProperties' in schema:
                ap = schema['additionalProperties']
                if ap is False:
                    errs.append(f'{path}: unknown key {k!r}')
                elif isinstance(ap, dict):
                    errs += _validate(v, ap, root, f'{path}.{k}')
    if 'anyOf' in schema and all(_validate(inst, s, root, path) for s in schema['anyOf']):
        errs.append(f'{path}: matches none of anyOf')
    return errs


def _project_json(p, tmp_path, name) -> dict:
    path = str(tmp_path / f'{name}.vantage')
    p.save(path)
    with zipfile.ZipFile(path) as z:
        return json.loads(z.read('project.json'))


def _full_project():
    """The sample with every optional part filled in: sensors of all kinds, a rig part, a unit calibration, a stream,
    a fleet record, a polygon ROI, load cases and a study's settings."""
    from vantage.core import calib
    from vantage.core.project import RigPart, Sensor
    p = _rig()
    v = p.variant
    v.sensors += [Sensor(tag='G1', name='GNSS', kind='gnss', model_id=next(m.id for m in p.sensor_models.values()
                                                                            if m.kind == 'gnss'), x=1.0, z=1.5),
                  Sensor(tag='IMU1', name='IMU', kind='imu', model_id=next(m.id for m in p.sensor_models.values()
                                                                           if m.kind == 'imu'), x=1.2, z=0.4)]
    p.rig.append(RigPart(name='Roof rack', kind='bar', z=1.6, length=1.2))
    cam = v.cameras[0]
    cam.calib = calib.clean_record(calib.simulated_unit(p.module_for(cam), p.nominal_lens_for(cam), seed=1,
                                                        serial='SN-1'))
    p.streams = {cam.tag: 'rtsp://10.0.0.5/stream'}
    p.fleet = {'vehicles': [{'id': 'v1', 'name': 'V1', 'vin': '', 'note': '', 'records': [{'id': 'r1',
                                                                                            'date': '2026-10-01'}]}]}
    p.roi.mode, p.roi.polygon = 'polygon', [(-2.0, -2.0), (6.0, -2.0), (6.0, 2.0)]
    p.analysis.load_cases = [['Full', 30.0, 45.0]]
    p.analysis.perception = {'max_range': 200.0}
    p.add_log('edit', 'filled in')
    return p


def test_the_schema_describes_every_project_vantage_writes(tmp_path):
    schema = json.load(open(SCHEMA_PATH, encoding='utf-8'))
    docs = {'sample': _project_json(samples.tigor_sample_project(), tmp_path, 'sample'),
            'blank': _project_json(samples.blank_project(), tmp_path, 'blank'),
            'full': _project_json(_full_project(), tmp_path, 'full')}
    for key in ('tigor', 'sedan', 'hatchback', 'suv', 'van', 'truck'):
        docs[key] = _project_json(samples.template_project(key), tmp_path, key)
    for name, doc in docs.items():
        assert _validate(doc, schema) == [], name
    full = docs['full']
    assert full['variants'][-1]['sensors'][-1]['kind'] == 'imu' and full['rig'] and full['fleet']['vehicles']
    try:
        import jsonschema
    except ImportError:
        jsonschema = None
    if jsonschema is not None:                              # an independent validator, when installed
        jsonschema.Draft202012Validator.check_schema(schema)
        for name, doc in docs.items():
            jsonschema.validate(doc, schema, cls=jsonschema.Draft202012Validator)
    # and it rejects what is wrong
    bad = []
    for mutate in (lambda d: d.update(format='other'),
                   lambda d: d.update(version='1'),
                   lambda d: d['variants'][0]['cameras'][0].update(x='a'),
                   lambda d: d['variants'][0]['cameras'][0].update(colour=1),
                   lambda d: d['analysis'].update(new_study={}),
                   lambda d: d['roi'].update(mode='circle'),
                   lambda d: d['variants'][-1]['sensors'][0].update(kind='sonar'),
                   lambda d: d['variants'][0]['cameras'][0].update(calib={'fx': 1000.0}),
                   lambda d: d['lenses']['h05r2'].update(kb=[0.1, 0.2]),
                   lambda d: d.pop('format')):
        d = copy.deepcopy(docs['full'])
        mutate(d)
        errs = _validate(d, schema)
        bad.append(bool(errs))
        if jsonschema is not None:
            with pytest.raises(jsonschema.ValidationError):
                jsonschema.validate(d, schema, cls=jsonschema.Draft202012Validator)
    assert all(bad), bad


def test_the_format_page_names_every_field():
    import dataclasses
    from vantage.core import lens, sensors
    from vantage.core import project as P
    text = open(os.path.join(ROOT, 'docs', 'project-format.md'), encoding='utf-8').read()
    named = set(re.findall(r'`([A-Za-z_][\w]*)`', text))
    for cls in (P.Camera, P.Sensor, P.ROI, P.Target, P.RigPart, P.Obstacle, P.ECU, P.TargetSettings, P.PhotoSet,
                P.AnalysisSettings, lens.Module, lens.Lens, sensors.SensorModel):
        missing = [f.name for f in dataclasses.fields(cls) if f.name not in named]
        assert not missing, (cls.__name__, missing)
    schema = json.load(open(SCHEMA_PATH, encoding='utf-8'))
    assert set(schema['properties']) == set(Project().to_json()) | {'format', 'version'}
    assert set(schema['$defs']['analysis']['properties']) == set(P.AnalysisSettings().to_dict())


def test_the_citation_file():
    cff = yaml.safe_load(open(os.path.join(ROOT, 'CITATION.cff'), encoding='utf-8'))
    assert cff['cff-version'] == '1.2.0' and cff['type'] == 'software'
    assert cff['version'] == __version__
    assert cff['license'] == 'GPL-3.0-or-later'
    assert 'GPL-3.0-or-later' in open(os.path.join(ROOT, 'pyproject.toml'), encoding='utf-8').read()
    assert cff['repository-code'] == 'https://github.com/Sherin-SEF-AI/Vantage'
    assert cff['authors'] and all(a.get('name') or a.get('family-names') for a in cff['authors'])
    changelog = open(os.path.join(ROOT, 'CHANGELOG.md'), encoding='utf-8').read()
    m = re.search(rf'^## {re.escape(__version__)} — (\d{{4}}-\d{{2}}-\d{{2}})', changelog, re.M)
    assert m and str(cff['date-released']) == m.group(1)
