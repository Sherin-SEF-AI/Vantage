"""Simulation campaigns (core.simcampaign): the design (templates, multiplexing, seeds, drop frames), the statistics
against independent references (Wilson intervals: known values and their exact coverage; the detection threshold on
synthetic curves with a known answer; Immerkær's noise estimate on an image of known noise), a small fault campaign end
to end through the bridge against the fake carla (every fault kind found where it should be, the estimates against
the injected truth, false alarms on clean drives, exports), resume after a cancel, a weather sweep with a stand-in
detector whose range shrinks in fog, and the page."""
import csv
import importlib.util
import io
import json
import math
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
FAKE_DIR = os.path.join(HERE, 'fake_carla')


def _fake():
    name = 'fake_carla_for_tests'
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, os.path.join(FAKE_DIR, 'carla', '__init__.py'))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _project(cams=('F', 'ML', 'R')):
    """The sample's data rig cut to a few cameras and the lidar (on a mast, so the fake ego box does not swallow
    its points)."""
    from vantage.core import samples
    p = samples.tigor_sample_project()
    v = next(v for v in p.variants if v.sensors)
    p.active_uid = v.uid
    v.cameras[:] = [c for c in v.cameras if c.tag in cams]
    v.sensors[:] = [s for s in v.sensors if s.kind == 'lidar']
    v.sensors[0].z = 2.0
    return p


class Runner:
    """The bridge in-process against the fake carla (what the campaign's subprocess runner does with a real server);
    counts the drives it ran."""

    def __init__(self, cancel_after=None):
        self.runs = []
        self.cancel_after = cancel_after

    def __call__(self, rig_json, out, args, progress, cancel):
        from vantage.sim import carla_bridge as CB
        fc = _fake()
        fc.reset()
        self.runs.append(os.path.basename(out))
        doc, remaps = CB.load_rig(rig_json)
        a = CB.resolve_args(CB.parser().parse_args([rig_json, out] + list(args)), doc)
        b = CB.Bridge(fc, doc, remaps, out, a, None, cancel)
        try:
            b.run()
        except CB.Cancelled:                        # as the bridge's main(): what was written goes, run.json says so
            if b.rec is not None:
                b.rec.cleanup()
            b.report.update(status='cancelled')
            with open(os.path.join(out, 'run.json'), 'w') as fh:
                json.dump(b.report, fh, default=str)
            return 130
        return 0


BASE = {'image_scale': 0.1, 'duration_s': 1.2, 'traffic': 8, 'warmup_s': 0.25, 'tick_hz': 80.0}


# ======================================================================= settings, templates and the design
def test_settings_tolerant_and_templates_fit_the_rig():
    from vantage.core import simcampaign as SC
    p = _project(cams=('F', 'R', 'ML', 'MR', 'QL', 'QR'))
    st = SC.CampaignSettings.from_dict({'seeds': [3, 'x', 3, 2.0, -5], 'fault_sets': [
        {'kind': 'rotation', 'tags': ['F', 'F', ''], 'axes': ['pitch', 'spin'], 'magnitudes': [0.5, 99, 'a', 0.5, -1]},
        {'kind': 'warp', 'tags': ['F'], 'magnitudes': [1]}, {'kind': 'drop', 'tags': [], 'magnitudes': [1]}],
        'evaluators': ['drivelog', 'nonsense'], 'target': 7, 'server': 'cloud', 'weathers': ['ClearNoon', 'Mars',
                                                                                            'vantage:fog'],
        'scenarios': ['cut_in', {'name': 'nope'}], 'asbuilt_on': 'some', 'clean': 'yes'})
    assert st.seeds == [3, 2, 0]
    assert st.fault_sets == [{'kind': 'rotation', 'tags': ['F'], 'magnitudes': [0.001, 0.5, 30.0], 'axes': ['pitch'],
                              'signs': 'alternate'}]
    assert st.evaluators == ['drivelog'] and st.target == 0.999 and st.server == 'running'
    assert st.weathers == ['ClearNoon', 'vantage:fog'] and [s['name'] for s in st.scenarios] == ['cut_in']
    assert st.asbuilt_on == 'clean' and st.clean is True
    assert SC.CampaignSettings.from_dict(st.to_dict()) == st
    assert SC.CampaignSettings.from_dict('junk') == SC.CampaignSettings()
    # the fault template on the sample's six cameras: rotations on the front and the first side camera, the clock on
    # the rear one, a drop and a freeze on others, one left as a control
    t = SC.template('fault_detectability', p)
    fs = {f['kind']: f for f in t.fault_sets}
    assert fs['rotation']['tags'] == ['F', 'ML'] and fs['rotation']['magnitudes'] == [0.1, 0.2, 0.3, 0.5, 1.0]
    assert fs['clock_offset']['tags'] == ['R'] and fs['clock_offset']['magnitudes'] == [1.0, 2.0, 5.0, 10.0]
    assert fs['drop']['tags'] == ['MR'] and fs['drop']['magnitudes'] == [1.0, 5.0, 10.0]
    assert fs['freeze']['tags'] == ['QR'] and t.seeds == [1, 2, 3]
    assert t.base['tick_hz'] == 320.0                     # the sync plan rounds to ~1 ms (80 Hz: 5.2 ms)
    w = SC.template('weather_sweep', p)
    assert len(w.weathers) == 6 and 'detection' in w.evaluators and not w.fault_sets
    assert SC.settings_of(p).template == 'fault_detectability'           # no campaign saved: the template


def test_plan_multiplexes_sensors_and_repeats_by_seed():
    from vantage.core import simcampaign as SC
    p = _project(cams=('F', 'R', 'ML', 'MR', 'QL', 'QR'))
    st = SC.template('fault_detectability', p)
    pl = SC.plan(p, st)
    ref = [d for d in pl.drives if d['role'] == 'reference']
    drives = [d for d in pl.drives if d['role'] == 'drive']
    assert len(ref) == 1 and ref[0]['gt'] == 'instance' and ref[0]['scenario']['name'] == 'parked'
    assert pl.drives[0] is ref[0]                                  # the reference first
    assert len(drives) == 3 * (1 + 15) and all(d['gt'] == 'none' for d in drives)
    for seed in (1, 2, 3):
        ds = [d for d in drives if d['seed'] == seed]
        assert sum(d['clean'] for d in ds) == 1
        # every rotation condition of F and of ML exactly once per seed; never the same schedule for both
        for tag in ('F', 'ML'):
            got = sorted((c['axis'], c['magnitude']) for d in ds for c in d['conditions'] if c['tag'] == tag)
            assert got == sorted((a, m) for a in ('yaw', 'pitch', 'roll') for m in (0.1, 0.2, 0.3, 0.5, 1.0))
        same = [d for d in ds if d['conditions'] and
                [(c['axis'], c['magnitude']) for c in d['conditions'] if c['tag'] == 'F'] ==
                [(c['axis'], c['magnitude']) for c in d['conditions'] if c['tag'] == 'ML']]
        assert len(same) < 5
        # signs flip from one seed to the next
        signs = {c['sign'] for d in ds for c in d['conditions'] if c['kind'] == 'rotation'}
        assert signs == {1 if seed % 2 else -1}
        # drops: exactly that share of the frames, never at the ends; the freeze centred
        for d in ds:
            for f, c in zip(d['faults'], d['conditions']):
                if c['kind'] == 'drop':
                    n = round(c['magnitude'] / 100 * 100)
                    assert len(f['frames']) == max(1, n) and min(f['frames']) >= 2 and max(f['frames']) < 98
                if c['kind'] == 'freeze':
                    assert (f['from_s'], f['to_s']) == (4.5, 5.5)
                if c['kind'] == 'rotation':
                    k = ('yaw', 'pitch', 'roll').index(c['axis'])
                    assert f['deg'][k] == c['sign'] * c['magnitude'] and sum(map(abs, f['deg'])) == c['magnitude']
    assert [d['faults'] for d in SC.plan(p, st).drives] == [d['faults'] for d in pl.drives]      # deterministic
    st2 = SC.CampaignSettings.from_dict(dict(st.to_dict(), multiplex=False, evaluators=['drivelog']))
    pl2 = SC.plan(p, st2)
    assert len(pl2.drives) == 3 * (1 + 30 + 4 + 3 + 1) and all(len(d['conditions']) <= 1 for d in pl2.drives)
    bad = SC.CampaignSettings.from_dict({'fault_sets': [{'kind': 'range_bias', 'tags': ['F'], 'magnitudes': [5]},
                                                        {'kind': 'drop', 'tags': ['ZZ'], 'magnitudes': [5]}]})
    pb = SC.plan(p, bad)
    assert len(pb.problems) == 2 and all(d['clean'] for d in pb.drives)
    est = SC.estimate(p, st, pl, out_dir=str(os.devnull) + '-none')
    assert est['drives'] == 49 and est['done'] == 0 and est['drive_s'] > 49 * 30 and 'drives' in est['text']


# ======================================================================= statistics against references
def _binom_pmf(k, n, p):
    return math.comb(n, k) * p ** k * (1 - p) ** (n - k)


def test_wilson_interval_known_values_and_exact_coverage():
    from vantage.core import simcampaign as SC
    lo, hi = SC.wilson(0, 10)
    assert lo == 0.0 and abs(hi - 0.27753) < 1e-4                # Brown, Cai & DasGupta's table: 0 of 10
    lo, hi = SC.wilson(5, 10)
    assert abs(lo - 0.23659) < 1e-4 and abs(hi - 0.76341) < 1e-4
    lo, hi = SC.wilson(9, 9)
    assert abs(lo - 0.70085) < 1e-4 and hi == 1.0
    assert SC.wilson(0, 0) is None
    # exact coverage (the binomial sums, no simulation), averaged over p: near the nominal level
    for n in (10, 30):
        cov = []
        for p in np.linspace(0.05, 0.95, 91):
            c = 0.0
            for k in range(n + 1):
                a, b = SC.wilson(k, n)
                if a <= p <= b:
                    c += _binom_pmf(k, n, p)
            cov.append(c)
        assert 0.93 <= float(np.mean(cov)) <= 0.97, (n, np.mean(cov))
        assert min(cov) > 0.80


def test_detection_threshold_on_synthetic_curves():
    from vantage.core import simcampaign as SC
    rng = np.random.default_rng(7)
    a, b, floor = 4.0, 6.0, 0.05                     # p(m) = f + (1 − f)·σ(a + b log10 m): m90 ≈ 0.49
    q = (0.9 - floor) / (1 - floor)
    truth = 10 ** ((math.log(q / (1 - q)) - a) / b)
    mags = [0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 4.0]
    n = [400] * len(mags)
    p = [floor + (1 - floor) / (1 + math.exp(-(a + b * math.log10(m)))) for m in mags]
    k = [int(rng.binomial(nn, pp)) for nn, pp in zip(n, p)]
    th = SC.threshold(mags, k, n, 0.9, floor, boot=200, seed=3)
    assert abs(th['fit'] / truth - 1) < 0.08, (th['fit'], truth)
    assert th['fit_ci'][0] <= truth <= th['fit_ci'][1]
    assert th['b'] > 0 and not th['extrapolated']
    # the empirical figure: the smallest tested magnitude from which every larger one reaches the target
    assert SC.empirical_threshold([0.1, 0.2, 0.5, 1.0], [0, 9, 8, 10], [10, 10, 10, 10], 0.9) == 1.0
    assert SC.empirical_threshold([0.1, 0.2, 0.5, 1.0], [0, 9, 10, 10], [10, 10, 10, 10], 0.9) == 0.2
    assert SC.empirical_threshold([0.1, 1.0], [0, 5], [10, 10], 0.9) is None
    # perfectly separated data: finite, between the last miss and the first hit
    th = SC.threshold([0.1, 0.2, 0.5, 1.0], [0, 0, 3, 3], [3, 3, 3, 3], 0.9, 0.0, boot=50)
    assert th['separated'] and th['empirical'] == 0.5 and 0.2 < th['fit'] <= 0.55
    # never reached
    th = SC.threshold([0.1, 1.0], [0, 1], [10, 10], 0.9, 0.0, boot=0)
    assert th['empirical'] is None and (th['fit'] is None or th['fit'] > 1.0)
    # a floor at the target leaves no threshold
    assert SC.fit_threshold({'a': 0.0, 'b': 1.0}, 0.9, 0.95) is None


def test_noise_estimate_and_image_statistics():
    from vantage.core import simcampaign as SC
    rng = np.random.default_rng(1)
    yy, xx = np.mgrid[0:400, 0:500]
    base = 60 + 0.2 * xx + 0.1 * yy
    for sigma in (2.0, 5.0):
        img = np.clip(base + rng.normal(0, sigma, base.shape), 0, 255).astype(np.uint8)
        est = SC.noise_sigma(img)
        assert abs(est / sigma - 1) < 0.12, (est, sigma)
    flat = np.full((200, 200), 128, np.uint8)
    s = SC.image_stats(flat)
    assert s['mean'] == 128 and s['contrast'] == 0 and s['noise'] < 1e-6 and s['clip_bright'] == 0
    valid = np.zeros((200, 200), bool)
    valid[50:150, 50:150] = True
    img = np.zeros((200, 200), np.uint8)
    img[valid] = 200
    assert SC.image_stats(img, valid)['mean'] == 200            # only the lens circle counts


# ======================================================================= rig health offline on a written drive
def _write_drive(root, streams):
    """A drive folder as the bridge writes one, reduced to what rig health reads: nuScenes sample_data, calibrated_sensor
    and sensor tables, the camera frames (JPEG) and run.json. streams: {channel: [(timestamp µs, image)]}."""
    import cv2
    V = os.path.join(root, 'nuscenes', 'v1.0-vantage')
    os.makedirs(V, exist_ok=True)
    sd, cs, sn = [], [], []
    for i, (ch, frames) in enumerate(streams.items()):
        sn.append({'token': f's{i}', 'channel': ch, 'modality': 'camera'})
        cs.append({'token': f'c{i}', 'sensor_token': f's{i}'})
        os.makedirs(os.path.join(root, 'nuscenes', 'sweeps', ch), exist_ok=True)
        for t, img in frames:
            name = f'sweeps/{ch}/{ch}__{t}.jpg'
            cv2.imwrite(os.path.join(root, 'nuscenes', name), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
            sd.append({'token': f'{ch}{t}', 'calibrated_sensor_token': f'c{i}', 'timestamp': t, 'filename': name})
    for n, rows in (('sample_data', sd), ('calibrated_sensor', cs), ('sensor', sn)):
        with open(os.path.join(V, n + '.json'), 'w') as fh:
            json.dump(rows, fh)
    with open(os.path.join(root, 'run.json'), 'w') as fh:
        json.dump({'status': 'ok', 'performance': {'simulated_s': 3.0}}, fh)


def test_rig_health_offline_counts_drops_freezes_and_offsets(tmp_path):
    """Frames written with known gaps, a known frozen run and a known offset between two cameras: rig health's
    statistics read them back exactly from the stamps and the images."""
    from vantage.core import simcampaign as SC
    p = _project(cams=('F', 'R'))
    var = p.variant
    rng = np.random.default_rng(2)
    t0 = 1_767_268_800_000_000
    imgs = [rng.integers(0, 255, (96, 120, 3), dtype=np.uint8) for _ in range(30)]
    for k in range(10, 17):                          # frames 10–16 repeat frame 9: 7 identical frames in a row
        imgs[k] = imgs[9]
    f_frames = [(t0 + k * 100_000, imgs[k]) for k in range(30) if k not in (4, 22, 25)]      # three dropped
    r_frames = [(t0 + k * 100_000 + 30_000, rng.integers(0, 255, (96, 120, 3), dtype=np.uint8)) for k in range(30)]
    root = str(tmp_path / 'drive')
    _write_drive(root, {'CAM_F': f_frames, 'CAM_R': r_frames})
    rig = {'cameras': [{'tag': 'F', 'capture': {'recorded_hz': 10.0}}, {'tag': 'R', 'capture': {'recorded_hz': 10.0}}]}
    st = SC.CampaignSettings.from_dict({'health': {'freeze_frames': 5}})
    ev = SC._eval_rig_health(p, var, root, rig, {}, st, None)
    F, R = ev['cameras']['F'], ev['cameras']['R']
    assert F['frames'] == 27 and F['dropped'] == 3 and F['max_frozen'] == 7
    assert R['frames'] == 30 and R['dropped'] == 0 and R['max_frozen'] == 0
    assert abs(R['offset_ms'] - 30.0) < 1e-3 and F['offset_ms'] == 0.0 and ev['reference_camera'] == 'F'
    assert R['sync_flag'] and not F['sync_flag']
    assert F['verdict'] == 'fault' and F['drift_px'] is None            # no reference drive: no drift
    oc = SC._outcomes({'rig_health': ev}, 'F', 'camera', {})
    assert oc['rig_health.drops'][1] and oc['rig_health.freeze'][1] and not oc['rig_health.drift'][0]
    assert abs(oc['rig_health.drops'][2] - 100 * 3 / 29) < 1e-6        # 3 missing over 26 intervals + 3
    assert SC._outcomes({'rig_health': ev}, 'R', 'camera', {})['rig_health.sync'][1]


# ======================================================================= a fault campaign end to end (fake carla)
@pytest.fixture(scope='module')
def fault_run(tmp_path_factory):
    from vantage.core import simcampaign as SC
    p = _project()
    out = str(tmp_path_factory.mktemp('campaign') / 'faults')
    st = SC.CampaignSettings.from_dict({
        'name': 'Fake faults', 'template': 'custom', 'seeds': [1, 5], 'out_dir': out,
        'fault_sets': [{'kind': 'rotation', 'tags': ['F'], 'axes': ['pitch'], 'magnitudes': [3.0], 'signs': 'positive'},
                       {'kind': 'clock_offset', 'tags': ['R'], 'magnitudes': [15.0]},
                       {'kind': 'drop', 'tags': ['ML'], 'magnitudes': [25.0]}],
        'base': dict(BASE, traffic=20), 'drivelog': {'tolerance_deg': 1.5, 'frames_per_camera': 6,
                                                     'sync_tol_ms': 6.0},
        'health': {'freeze_frames': 4}, 'evaluators': ['drivelog', 'rig_health', 'image_stats', 'returns']})
    runner = Runner()
    res = SC.run(p, settings=st, runner=runner)
    return {'p': p, 'st': st, 'res': res, 'runner': runner, 'out': out}


def test_fault_campaign_finds_each_fault_and_measures_it(fault_run):
    from vantage.core import simcampaign as SC
    res, out = fault_run['res'], fault_run['out']
    assert res is not None and res.complete
    assert fault_run['runner'].runs == ['ref-data-rig-6-cam-lidar-radars', 'd0001', 'd0002', 'd0003', 'd0004']
    for did in ('d0002', 'd0004'):                                   # the faulty drives say so
        run = json.load(open(os.path.join(out, 'drives', did, 'run.json')))
        assert {(f['tag'], f['kind']) for f in run['faults_injected']} == {('F', 'rotation'), ('R', 'clock_offset'),
                                                                          ('ML', 'drop')}
    obs = res.observations

    def flagged(check, tag, clean):
        return [r['flagged'] for r in obs if r['check'] == check and r['tag'] == tag and r['clean_drive'] == clean]
    # drops: both checks find them; nothing on the clean drives; the measured share is the injected one exactly
    assert flagged('drivelog.streams', 'ML', False) == [True, True] and flagged('drivelog.streams', 'ML', True) == \
        [False, False]
    assert flagged('rig_health.drops', 'ML', False) == [True, True]
    acc = {(a['check'], a['tag'], a['kind']): a for a in res.detectability['accuracy']}
    assert acc[('drivelog.streams', 'ML', 'drop')]['absolute']['max_abs'] < 1e-6
    assert acc[('rig_health.drops', 'ML', 'drop')]['absolute']['max_abs'] < 3.0
    # the clock offset: the sync check finds it and measures it (its plan rounding of 80 Hz steps, ±4.2 ms here, taken
    # out) within 0.1 ms
    assert flagged('drivelog.sync', 'R', False) == [True, True] and flagged('drivelog.sync', 'R', True) == \
        [False, False]
    a = acc[('drivelog.sync', 'R', 'clock_offset')]
    assert a['absolute']['n'] == 2 and a['absolute']['max_abs'] < 0.1 and a['paired']['max_abs'] < 0.1
    # the rotation: the drive-log check finds a 3° pitch of F (tolerance 1.5°) and measures it within a pixel (1.3° at
    # this image scale); rig health's drift against CARLA's own car (the reference drive) rises; the clean drives pass
    assert flagged('drivelog.extrinsics', 'F', False) == [True, True]
    assert flagged('drivelog.extrinsics', 'F', True) == [False, False]
    a = acc[('drivelog.extrinsics', 'F', 'rotation')]
    assert a['absolute']['n'] == 2 and a['absolute']['max_abs'] < 1.0, a
    drift = {r['clean_drive']: r['stat'] for r in obs if r['check'] == 'rig_health.drift' and r['tag'] == 'F'}
    assert drift[False] > drift[True]
    assert os.path.isfile(os.path.join(out, 'drives', 'ref-data-rig-6-cam-lidar-radars', 'campaign_body', 'F.png'))
    # the curves and thresholds: one magnitude each, found every time from it
    th = {(t['check'], t['tag']): t for t in res.detectability['thresholds'] if t['axis'] in ('', 'any')}
    assert th[('drivelog.streams', 'ML')]['empirical'] == 25.0 and th[('drivelog.sync', 'R')]['empirical'] == 15.0
    assert th[('drivelog.extrinsics', 'F')]['empirical'] == 3.0
    fa = {(f['check'], f['tag']): f for f in res.detectability['false_alarms']}
    assert fa[('drivelog.extrinsics', 'F')]['n'] == 2 and fa[('drivelog.extrinsics', 'F')]['k'] == 0
    # image statistics and returns on every drive
    for did in ('d0001', 'd0002'):
        ev = res.evaluations[did]
        assert set(ev['image_stats']['cameras']) == {'F', 'ML', 'R'}
        assert ev['returns']['sensors']['L1']['points_mean'] > 100
    s = SC.summary(res)
    assert s['drives'] == 5 and s['driven'] == 5 and s['failed'] == 0 and s['headline']
    json.dumps(s)


def test_fault_campaign_exports(fault_run, tmp_path):
    from vantage.core import simcampaign as SC
    res, p = fault_run['res'], fault_run['p']
    paths = SC.export(res, p, str(tmp_path / 'x'))
    names = {os.path.basename(x) for x in paths}
    assert {'campaign.json', 'campaign_observations.csv', 'campaign_detection_curves.csv', 'campaign_thresholds.csv',
            'campaign_false_alarms.csv', 'campaign_accuracy.csv', 'campaign_drives.csv',
            'campaign_detection_curves.png', 'campaign_report.html'} <= names
    doc = json.load(open(tmp_path / 'x' / 'campaign.json'))
    assert doc['schema'] == SC.SCHEMA and doc['frame'].startswith('vehicle') and len(doc['drives']) == 5
    assert doc['drives'][0]['role'] == 'reference' and doc['drives'][2]['conditions']
    rows = list(csv.DictReader(io.StringIO((tmp_path / 'x' / 'campaign_detection_curves.csv').read_text())))
    r = next(r for r in rows if r['check'] == 'drivelog.sync' and r['sensor'] == 'R')
    assert r['trials'] == '2' and r['detected'] == '2' and float(r['ci_lo']) < 0.5
    html = (tmp_path / 'x' / 'campaign_report.html').read_text()
    assert 'Smallest fault detected' in html and 'data:image/png;base64,' in html and 'Wilson' in html
    import cv2
    img = cv2.imread(str(tmp_path / 'x' / 'campaign_detection_curves.png'))
    assert img is not None and img.shape[1] >= 1000



def _with_asbuilt(fault_run):
    """The fault campaign's result with as-built evaluations added (written here, as core.asbuilt reports a camera):
    on both clean drives F 'does not hold', ML 'holds', R 'holds with corrections'; on the faulty drives (F turned)
    F 'does not hold'. Analysed by the module itself from the evaluations on."""
    import copy
    import dataclasses
    from vantage.core import simcampaign as SC
    res, p, st, out = fault_run['res'], fault_run['p'], fault_run['st'], fault_run['out']
    evals = copy.deepcopy(res.evaluations)
    clean = {d['id'] for d in res.plan.drives if d['role'] == 'drive' and d['clean']}
    for did in evals:
        cams = {'F': 'does_not_hold', 'ML': 'holds', 'R': 'holds_with_corrections'} if did in clean else \
            {'F': 'does_not_hold', 'ML': 'holds', 'R': 'holds'}
        evals[did]['asbuilt'] = {'verdict': 'does_not_hold', 'cameras': {t: {'camera': t, 'verdict': v, 'iou': 0.6}
                                                                         for t, v in cams.items()}}
    obs = SC.observations(p, st, res.plan, evals, out)
    return dataclasses.replace(res, evaluations=evals, observations=obs, detectability=SC.detectability(obs, st))


def test_checks_that_cannot_judge_on_carla_are_listed_apart(fault_run, tmp_path):
    """The as-built check cannot judge a mount on a CARLA drive (CARLA's car is another body): its flags on clean
    drives stay visible, marked not applicable with the reason, but are no false alarms (the summary's, the headline's,
    the export's), it gets no threshold, and its curve is marked; every other check is unchanged."""
    from vantage.core import simcampaign as SC
    res = _with_asbuilt(fault_run)
    det = res.detectability
    assert not SC.applicable('asbuilt.verdict') and SC.applicable('drivelog.extrinsics')
    fa = {(f['check'], f['tag']): f for f in det['false_alarms']}
    for tag, k in (('F', 2), ('ML', 0), ('R', 2)):
        f = fa[('asbuilt.verdict', tag)]
        assert f['n'] == 2 and f['k'] == k and f['applicable'] is False and \
            f['not_applicable'] == SC.NOT_APPLICABLE['asbuilt.verdict']
    assert all(f['applicable'] and f['not_applicable'] is None for f in det['false_alarms']
               if f['check'] != 'asbuilt.verdict')
    assert fa[('drivelog.extrinsics', 'F')]['k'] == 0                                  # unchanged
    assert all(not c['applicable'] for c in det['crosstalk'] if c['check'] == 'asbuilt.verdict')
    curves = [c for c in det['curves'] if c['check'] == 'asbuilt.verdict']
    assert curves and not any(c['applicable'] for c in curves)
    assert any(c['kind'] == 'rotation' and c['relevant'] for c in curves)           # turned F flagged: no detection
    assert not any(t['check'] == 'asbuilt.verdict' for t in det['thresholds'])
    assert any(t['check'] == 'drivelog.extrinsics' and t['empirical'] == 3.0 for t in det['thresholds'])
    assert all(r['applicable'] == (r['check'] != 'asbuilt.verdict') for r in res.observations)
    s = SC.summary(res)
    assert not any(f['check'] == 'asbuilt.verdict' for f in s['false_alarms'])
    assert [(f['sensor'], f['flagged'], f['clean_drives']) for f in s['not_applicable_on_carla']] == \
        [('F', 2, 2), ('R', 2, 2)]
    assert s['not_applicable_why'] == {'asbuilt.verdict': SC.NOT_APPLICABLE['asbuilt.verdict']}
    hl = [h for h in SC.headline(res) if 'not applicable on CARLA' in h]
    assert len(hl) == 1 and 'F 2 of 2' in hl[0] and 'R 2 of 2' in hl[0] and 'ML' not in hl[0]
    paths = SC.export(res, fault_run['p'], str(tmp_path / 'x'))
    doc = json.load(open(tmp_path / 'x' / 'campaign.json'))
    assert doc['not_applicable_on_carla'] == SC.NOT_APPLICABLE
    rows = list(csv.DictReader(io.StringIO((tmp_path / 'x' / 'campaign_false_alarms.csv').read_text())))
    ab = [r for r in rows if r['check'] == 'asbuilt.verdict']
    assert ab and all(r['applicable_on_carla'] == 'False' and r['note'].startswith('Not applicable on CARLA')
                      for r in ab)
    assert all(r['applicable_on_carla'] == 'True' and not r['note'] for r in rows if r['check'] != 'asbuilt.verdict')
    obs_rows = list(csv.DictReader(io.StringIO((tmp_path / 'x' / 'campaign_observations.csv').read_text())))
    assert {r['applicable_on_carla'] for r in obs_rows if r['check'] == 'asbuilt.verdict'} == {'False'}
    html = (tmp_path / 'x' / 'campaign_report.html').read_text()
    assert 'Not applicable on CARLA: listed apart, not false alarms' in html and len(paths) >= 8

def test_rerun_reuses_drives_and_evaluations(fault_run):
    from vantage.core import simcampaign as SC
    runner = Runner()
    res = SC.run(fault_run['p'], settings=fault_run['st'], runner=runner)
    assert runner.runs == [] and res.complete
    assert all(s.status == 'reused' for s in res.states.values())
    assert res.observations == fault_run['res'].observations
    est = SC.estimate(fault_run['p'], fault_run['st'])
    assert est['done'] == 5 and est['remaining_s'] == 0 and est['measured']


def test_cancel_then_resume(tmp_path):
    from vantage.core import simcampaign as SC
    p = _project(cams=('F',))
    st = SC.CampaignSettings.from_dict({
        'name': 'Resume', 'template': 'custom', 'seeds': [4], 'out_dir': str(tmp_path / 'c'),
        'fault_sets': [{'kind': 'clock_offset', 'tags': ['F'], 'magnitudes': [5.0, 10.0]}],
        'base': dict(BASE, duration_s=1.0, traffic=2), 'evaluators': ['returns']})
    runner = Runner()

    def cancel():                                   # asked during the second drive
        return len(runner.runs) >= 2
    res = SC.run(p, settings=st, runner=runner, cancel=cancel)
    assert res is None and runner.runs == ['d0001', 'd0002']
    pl = SC.plan(p, st)
    done = [d['id'] for d in pl.drives if SC._drive_done(SC.campaign_dir(p, st), d) is not None]
    assert done == ['d0001']                         # the cancelled drive left nothing that counts as done
    runner2 = Runner()
    res = SC.run(p, settings=st, runner=runner2)
    assert runner2.runs == ['d0002', 'd0003'] and res.complete
    assert res.states['d0001'].status == 'reused' and res.states['d0003'].status == 'ok'
    # a changed rig invalidates the drives: driven again
    st2 = SC.CampaignSettings.from_dict(dict(st.to_dict(), base=dict(st.base, image_scale=0.12)))
    runner3 = Runner()
    SC.run(p, settings=st2, runner=runner3, phases=('drive',))
    assert runner3.runs == ['d0001', 'd0002', 'd0003']


def test_a_rig_reworded_or_keyed_by_an_older_hash_keeps_its_drives(tmp_path):
    """A rig file a newer Vantage words differently (its prose, the render and memory budget it reports) or a key made
    by an older rig_hash is still the same recording: Run keeps the drives, the rig file and its key (the real fault
    campaign's rig gained two budget figures after it was driven, and Run would have dropped its 49 drives). A change
    that reaches the recording still drives again."""
    from vantage.core import simcampaign as SC
    p = _project(cams=('F',))
    st = SC.CampaignSettings.from_dict({
        'name': 'Keys', 'template': 'custom', 'seeds': [4], 'out_dir': str(tmp_path / 'k'),
        'base': dict(BASE, duration_s=1.0, traffic=2), 'evaluators': ['returns']})
    runner = Runner()
    assert SC.run(p, settings=st, runner=runner).complete and runner.runs == ['d0001']
    out, pl = SC.campaign_dir(p, st), SC.plan(p, st)
    rd = os.path.join(out, 'rigs', pl.drives[0]['rig'])
    doc = SC._read_json(os.path.join(rd, 'sim_rig.json'))
    doc.update(assumptions='as an older Vantage worded them', notes=['an older note'])
    for k in ('bridge_memory_gb', 'render_mpx_per_frame'):
        doc['simulation'].pop(k, None)
    SC._write_json(os.path.join(rd, 'sim_rig.json'), doc)
    legacy = 'f' * 64                                # a key from another hash function
    SC._write_json(os.path.join(rd, 'rig_key.json'), {'hash': legacy})
    for d in pl.drives:
        f = os.path.join(out, 'drives', d['id'], 'campaign_drive.json')
        SC._write_json(f, dict(SC._read_json(f), key=SC.drive_key(d, legacy)))
    runner2 = Runner()
    res = SC.run(p, settings=st, runner=runner2)
    assert runner2.runs == [] and res.complete and all(s.status == 'reused' for s in res.states.values())
    assert SC._read_json(os.path.join(rd, 'rig_key.json'))['hash'] == legacy
    assert SC._read_json(os.path.join(rd, 'sim_rig.json'))['assumptions'] == 'as an older Vantage worded them'
    st2 = SC.CampaignSettings.from_dict(dict(st.to_dict(), base=dict(st.base, image_scale=0.12)))
    runner3 = Runner()
    SC.run(p, settings=st2, runner=runner3, phases=('drive',))
    assert runner3.runs == ['d0001']


def test_no_server_and_no_python_are_said(tmp_path, monkeypatch):
    from vantage.core import simcampaign as SC
    p = _project(cams=('F',))
    st = SC.CampaignSettings.from_dict({'seeds': [1], 'out_dir': str(tmp_path / 'n'), 'evaluators': ['returns'],
                                        'base': dict(BASE, port=2459)})
    monkeypatch.setenv('VANTAGE_CARLA_PYTHON', '')
    with pytest.raises(RuntimeError, match='No CARLA server answers on port 2459'):
        SC.run(p, settings=st, python=sys.executable)
    with pytest.raises(RuntimeError, match='Python'):
        SC.run(p, settings=st, python='/nonexistent/python')


def test_campaign_starts_and_stops_its_own_server(tmp_path, monkeypatch):
    """server 'own': with nothing on the port the campaign starts a capped server (vantage.sim.server.Server), waits
    for it, drives, and stops it, also after a failure; a server that never comes up is said and stopped."""
    from vantage.core import simcampaign as SC
    from vantage.sim import server as SV
    p = _project(cams=('F',))
    st = SC.CampaignSettings.from_dict({'name': 'Own', 'seeds': [4], 'out_dir': str(tmp_path / 'o'), 'server': 'own',
                                        'evaluators': ['returns'], 'base': dict(BASE, traffic=2, port=2461)})
    log = []
    up = {'on': False}

    class FakeServer:
        def __init__(self, root, port, memory_gb, quality, no_render, log_path):
            log.append(('new', root, port, memory_gb, no_render))
            self.ok = root != 'dead'

        def start(self):
            log.append('start')
            up['on'] = self.ok

        @property
        def running(self):
            return self.ok

        def ready(self):
            return up['on']

        def stop(self):
            log.append('stop')
            up['on'] = False
    monkeypatch.setattr(SV, 'Server', FakeServer)
    monkeypatch.setattr(SV, 'port_open', lambda host='127.0.0.1', port=2000, timeout=0.5: up['on'])
    monkeypatch.setattr(SC, '_carla_root', lambda: '/opt/carla-test')
    runner = Runner()
    monkeypatch.setattr(SC, '_bridge_subprocess', lambda py, t: runner)
    monkeypatch.setattr(SC.time, 'sleep', lambda s: None)
    res = SC.run(p, settings=st, python=sys.executable)
    assert res.complete and runner.runs == ['d0001']
    assert log == [('new', '/opt/carla-test', 2461, 12.0, False), 'start', 'stop'] and not up['on']
    log.clear()
    monkeypatch.setattr(SC, '_carla_root', lambda: 'dead')
    st2 = SC.CampaignSettings.from_dict(dict(st.to_dict(), seeds=[5]))
    with pytest.raises(RuntimeError, match='did not come up'):
        SC.run(p, settings=st2, python=sys.executable)
    assert log[-1] == 'stop'


def test_rig_comparison_pairs_two_variants(tmp_path):
    """Rig A vs B: two variants driven through the same scenario and seed, compared by core.detection_eval (paired)."""
    from vantage.core import simcampaign as SC
    from vantage.core.project import Variant, new_uid
    p = _project(cams=('F',))
    a = p.variant
    b = Variant.from_dict(a.to_dict())
    b.uid, b.name = new_uid(), 'F pitched up'
    for c in b.cameras:
        c.uid = new_uid()
        c.pitch -= 15.0
    for x in b.sensors:
        x.uid = new_uid()
    p.variants.append(b)
    st = SC.CampaignSettings.from_dict({
        'name': 'AB', 'template': 'rig_ab', 'variants': [a.uid, b.uid], 'seeds': [4], 'out_dir': str(tmp_path / 'ab'),
        'base': dict(BASE, traffic=10, duration_s=1.0), 'evaluators': ['detection', 'returns'],
        'detection': {'bootstrap': 50, 'cache': False, 'min_px': 4.0, 'min_pixels': 10}})
    runner = Runner()
    SC.run(p, settings=st, runner=runner, phases=('drive',))
    assert runner.runs == ['d0001', 'd0002']
    out = SC.campaign_dir(p, st)
    oracle = FogOracle([(os.path.join(out, 'drives', 'd0001'), 1e9), (os.path.join(out, 'drives', 'd0002'), 1e9)])
    res = SC.run(p, settings=st, runner=runner, detector=oracle)
    rg = res.rigs
    assert rg is not None and [v['variant'] for v in rg['variants']] == [a.name, 'F pitched up']
    cmp = rg['comparison']
    assert cmp and len(cmp['recordings']) == 2 and cmp['comparisons']
    assert all(c['paired'] for c in cmp['comparisons'])
    assert SC.summary(res)['drives'] == 2


# ======================================================================= a weather sweep with a stand-in detector
class FogOracle:
    """Finds every labelled object of a drive's frames within a range that depends on the drive: all of them in
    clear weather, those nearer than 12 m in fog (a detector whose range fog shortens). The fake carla renders no
    weather, so both drives' images are the same: the drives are told apart by the order the campaign evaluates them
    in (drive by drive, every frame once)."""

    def __init__(self, drives):
        import cv2
        from vantage.core import groundtruth as GT
        self.by, self.cutoffs, self.calls = {}, [], 0
        for path, cutoff in drives:
            doc = GT.read(path)
            n = 0
            for c in doc['cameras'].values():
                for f in c['frames']:
                    im = cv2.imread(GT.path_of(doc, f['image']), cv2.IMREAD_COLOR)
                    self.by[im.tobytes()] = f['objects']
                    n += 1
            self.cutoffs.append((n, cutoff))

    def __call__(self, img):
        from vantage.core import detection as DT
        k, i = self.calls, 0
        while i < len(self.cutoffs) - 1 and k >= self.cutoffs[i][0]:
            k -= self.cutoffs[i][0]
            i += 1
        self.calls += 1
        objs, cutoff = self.by.get(img.tobytes(), []), self.cutoffs[i][1]
        objs = [o for o in objs if DT.GT_TO_CLASS.get(o['class']) and o['range_m'] < cutoff]
        boxes = np.array([o['bbox'] for o in objs], np.float32).reshape(-1, 4)
        cls = np.array([DT.CLASSES.index(DT.GT_TO_CLASS[o['class']]) for o in objs], np.int8)
        return {'boxes': boxes, 'scores': np.full(len(objs), 0.9, np.float32), 'cls': cls,
                'coco': np.zeros(len(objs), np.int16)}


def test_weather_sweep_with_a_detector_that_fog_shortens(tmp_path):
    from vantage.core import simcampaign as SC
    p = _project(cams=('F',))
    st = SC.CampaignSettings.from_dict({
        'name': 'Weather', 'template': 'custom', 'seeds': [4], 'out_dir': str(tmp_path / 'w'),
        'weathers': ['vantage:clear', 'vantage:fog'], 'base': dict(BASE, traffic=10, duration_s=1.0),
        'evaluators': ['detection', 'image_stats', 'returns'],
        'detection': {'bootstrap': 0, 'cache': False, 'range_edges': [6.0, 12.0, 24.0], 'min_px': 4.0,
                      'min_pixels': 10}})
    runner = Runner()
    res = SC.run(p, settings=st, runner=runner, phases=('drive',))
    assert runner.runs == ['d0001', 'd0002']
    out = SC.campaign_dir(p, st)
    for did in ('d0001', 'd0002'):
        run = json.load(open(os.path.join(out, 'drives', did, 'run.json')))
        assert run['ground_truth']['settings']['instance'] and run['ground_truth']['settings']['range']
    oracle = FogOracle([(os.path.join(out, 'drives', 'd0001'), 1e9), (os.path.join(out, 'drives', 'd0002'), 12.0)])
    res = SC.run(p, settings=st, runner=runner, detector=oracle)
    wx = res.weather
    assert wx is not None and [r['weather'] for r in wx['rows']] == ['vantage:clear', 'vantage:fog']
    clear, fog = wx['rows']
    cls = next(c for c in ('car', 'vehicle') if c in clear['classes'] and clear['classes'][c]['n'])
    assert clear['classes'][cls]['recall'] == 1.0
    near = [b for b in fog['classes'][cls]['bins'] if b['bin'] in ('0–6 m', '6–12 m') and b['n']]
    far = [b for b in fog['classes'][cls]['bins'] if b['bin'] in ('12–24 m', '24+ m') and b['n']]
    assert near and all(b['recall'] == 1.0 for b in near)
    assert far and all(b['recall'] == 0.0 for b in far)
    assert fog['classes'][cls]['recall'] < 1.0 and fog['classes'][cls]['recall_ratio'] < 1.0
    # Vantage's own model of the same named condition, beside it (for this fisheye the pixel height limits a car's
    # range well inside the fog's 150 m visibility, so Vantage predicts little or no loss: the two models disagree)
    vf = fog['classes'][cls]['vantage_factor']
    assert vf is not None and 0.0 < vf <= 1.0 and clear['classes'][cls]['vantage_factor'] == 1.0
    # CARLA's lidar ignores the weather: the same seed gives the same returns
    assert fog['returns']['L1']['points_ratio'] == pytest.approx(1.0, abs=0.02)
    assert res.weather['notes'] and SC.summary(res)['weather']
    paths = SC.export(res, p, str(tmp_path / 'ex'))
    assert any(x.endswith('campaign_weather.csv') for x in paths) and any(x.endswith('campaign_weather.png')
                                                                          for x in paths)


# ======================================================================= the page
from test_gui import app, win, pump, _wait   # noqa: E402,F401 (fixtures)


def _open(win, name, cls):
    if name not in win._page_widgets:
        pg = cls(win.ctx)
        win.pages.addWidget(pg)
        win._page_widgets[name] = pg
    win.set_page(name)
    return win._page_widgets[name]


def test_page_builds_runs_cancels_resumes_and_exports(app, win, tmp_path):
    from vantage.core import simcampaign as SC
    from vantage.ui.workspaces.campaign_ws import CampaignPage
    p = win.store.project
    v = next(v for v in p.variants if v.sensors)
    win.store.project.active_uid = v.uid
    v.cameras[:] = [c for c in v.cameras if c.tag in ('F',)]
    v.sensors[:] = [s for s in v.sensors if s.kind == 'lidar']
    v.sensors[0].z = 2.0
    pg = _open(win, 'campaign', CampaignPage)
    pump(app, 0.2)
    # a template: the settings go into the project in one undoable edit
    assert pg.apply_template('fault_detectability')
    pump(app, 0.1)
    st = SC.settings_of(win.store.project)
    assert st.template == 'fault_detectability' and st.fault_sets
    assert 'drives' in pg.cost.text()
    win.store.undo.undo()
    pump(app, 0.1)
    assert not win.store.project.analysis.campaign
    win.store.undo.redo()
    pump(app, 0.1)
    # edit it down to a small campaign through the page's fields
    pg.name_edit.setText('Page test')
    pg.name_edit.editingFinished.emit()
    pg.seeds_edit.setText('4')
    pg.seeds_edit.editingFinished.emit()
    assert pg.set_fault_sets([{'kind': 'clock_offset', 'tags': ['F'], 'magnitudes': [5.0, 10.0]}])
    pg.set_base({'image_scale': 0.1, 'duration_s': 1.0, 'traffic': 2, 'warmup_s': 0.25, 'tick_hz': 80.0})
    pg.set_out_dir(str(tmp_path / 'page'))
    for e, cb in pg.eval_box.items():
        cb.setChecked(e in ('drivelog', 'returns'))
    pump(app, 0.1)
    st = SC.settings_of(win.store.project)
    assert st.name == 'Page test' and st.seeds == [4] and st.evaluators == ['drivelog', 'returns']
    assert pg.table_drives.rowCount() == 3                 # clean + two faulty
    assert '3 drives' in pg.cost.text()
    # run with a cancel during the second drive, then resume
    runner = Runner()
    pg.runner = runner
    pg.cancel_when = lambda: len(runner.runs) >= 2
    assert pg.run()
    _wait(app, lambda: not pg.busy, 300)
    assert pg.result is None and 'Cancelled' in pg.status.text()
    pg.cancel_when = None
    assert pg.run()
    _wait(app, lambda: not pg.busy, 300)
    assert pg.result is not None and pg.result.complete
    assert runner.runs == ['d0001', 'd0002', 'd0002', 'd0003']
    assert pg.tabs.count() >= 4 and pg.table_thresholds.rowCount() >= 1
    assert pg.chart.curves                                    # the detection curves drawn
    assert any(pg.table_drives.item(r, 5).text() in ('ok', 'reused') for r in range(pg.table_drives.rowCount()))
    paths = pg.export(str(tmp_path / 'ex'))
    assert any(x.endswith('campaign_report.html') for x in paths)
    assert pg.summary()['drives'] == 3
    # a project replaced while the page is open: the result goes
    win.store.set_project(SC_sample())
    pump(app, 0.2)
    assert pg.result is None


def SC_sample():
    from vantage.core import samples
    return samples.tigor_sample_project()


def test_page_false_alarm_tile_counts_only_checks_that_judge(app, win, fault_run):
    """The False alarms tile counts the checks that can judge on CARLA; the as-built check's flags (not applicable)
    are said beside it and listed in the False alarms tab after the others, greyed, marked with the reason; the
    detectability picker marks its curve."""
    from vantage.core import simcampaign as SC
    from vantage.ui.workspaces.campaign_ws import CampaignPage
    res = _with_asbuilt(fault_run)
    det = res.detectability
    pg = _open(win, 'campaign', CampaignPage)
    pg.result = res
    pg.refresh()
    pump(app, 0.2)
    n_judge = sum(1 for f in det['false_alarms'] if f['k'] and f['applicable'])
    assert pg.t_fa.value == str(n_judge) and pg.t_fa.unit == 'checks'
    assert pg.t_fa.note == 'on clean drives · 2 not applicable on CARLA'
    assert pg.t_fa.sev == ('warning' if n_judge else 'good')
    t = pg.table_fa
    rows = [(t.item(r, 2).text(), t.item(r, 7).text(), t.item(r, 0).text()) for r in range(t.rowCount())]
    clean = [x for x in rows if x[0] == 'clean drives']
    assert rows[:len(clean)] == clean                                                 # clean drives first
    assert any(x[1] == 'not applicable' for x in clean)
    for part in (clean, rows[len(clean):]):                                           # then cross-talk
        on_carla = [x[1] for x in part]
        k = on_carla.index('not applicable') if 'not applicable' in on_carla else len(on_carla)
        assert set(on_carla[:k]) == {'judges'} and set(on_carla[k:]) <= {'not applicable'}   # apart, after the others
        assert {x[2] for x in part[k:]} <= {SC.CHECKS['asbuilt.verdict'][0]}
    k = [x[1] for x in rows].index('not applicable')
    assert t.item(k, 7).toolTip() == SC.NOT_APPLICABLE['asbuilt.verdict']
    flagged = [r for r in range(k, len(clean)) if t.item(r, 3).text() != '0']
    assert flagged and t.item(flagged[0], 3).foreground().color() != QColor_warning()
    i = pg.check_pick.findData('asbuilt.verdict|rotation')
    assert i >= 0 and pg.check_pick.itemText(i).endswith('(not applicable on CARLA)')
    pg.result = None
    pg.refresh()


def QColor_warning():
    from PyQt6.QtGui import QColor
    from vantage.ui.theme import STATUS
    return QColor(STATUS['warning'])
