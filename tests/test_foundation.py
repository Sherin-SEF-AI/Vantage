"""Foundation models on rendered frames (core.foundation_models, core.foundation_range and their page): the model specs
and the pinned, SHA-checked download against a fake server (a wrong byte is never kept, a cancel leaves nothing, a
non-commercial licence needs the opt-in); the "needs torch and transformers" path with torch blocked; the logistic
range fit recovering a known R50 and R90 from simulated detections with intervals that cover them; depth metrics
against a hand-made depth pair and the ground-plane scale recovering a known plane; the sweep placing objects where it
says (and the fake carla putting them there); the study on a fake-carla recording with detectors that know the ground
truth (every object, or only objects at least H px tall: the fitted H50 is H); the perception records; and the page
driven like a user. test_real_* use the downloaded models (VANTAGE_FOUNDATION_DIR) and skip when they are absent."""
import builtins
import hashlib
import importlib.util
import io
import json
import math
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
FAKE_DIR = os.path.join(HERE, 'fake_carla')
REAL_DIR = os.environ.get('VANTAGE_FOUNDATION_DIR', '')
REAL_REC = os.environ.get('VANTAGE_FOUNDATION_RECORDING') or os.path.join(
    os.environ.get('VANTAGE_CARLA_DRIVES', '/nonexistent'), 'data-rig-6-cam-lidar-radars')   # a CARLA drive with GT


def _real(mid):
    from vantage.core import foundation_models as FM
    ok = bool(REAL_DIR) and os.path.isdir(os.path.join(REAL_DIR, mid)) and FM.torch_missing() is None
    return pytest.mark.skipif(not ok, reason=f'the real model {mid} is not present (VANTAGE_FOUNDATION_DIR)')


@pytest.fixture(autouse=True)
def _folders(tmp_path, monkeypatch):
    """Never the user's data folder: every test gets its own models folders (test_real_* point at the real one)."""
    from vantage.core import detection as DT
    from vantage.core import foundation_models as FM
    monkeypatch.setenv(DT.ENV_DIR, str(tmp_path / 'models'))
    monkeypatch.setenv(FM.ENV_DIR, str(tmp_path / 'foundation'))
    monkeypatch.delenv(FM.ENV_GPU_LOCK, raising=False)


# ======================================================================= specs and downloads
def test_every_spec_is_pinned_and_licensed():
    from vantage.core import foundation_models as FM
    assert set(FM.DETECTORS) == {'owlv2_b16_ens', 'gdino_tiny'}
    assert set(FM.DEPTH_MODELS) == {'da2_small', 'da2_metric_outdoor_small', 'da2_base'}
    for sp in FM.SPECS.values():
        assert len(sp.revision) == 40 and all(ch in '0123456789abcdef' for ch in sp.revision)
        assert sp.files and any(f.name.endswith('.safetensors') for f in sp.files)
        for f in sp.files:
            assert len(f.sha256) == 64 and f.size > 0
        assert sp.licence and sp.licence_note and sp.paper and 'arXiv' in sp.paper
        assert sp.size == sum(f.size for f in sp.files)
    # what the model cards say (checked 2026-11-05): Apache-2.0 for the detectors and DA2 Small; the Base model is
    # CC-BY-NC-4.0 and the metric outdoor model is trained on non-commercial data: opt-in only
    assert [m for m, s in FM.SPECS.items() if s.commercial] == ['owlv2_b16_ens', 'gdino_tiny', 'da2_small']
    assert FM.SPECS['da2_base'].licence == 'CC-BY-NC-4.0'
    assert 'CC BY-NC-SA 3.0' in FM.SPECS['da2_metric_outdoor_small'].licence_note
    assert FM.SPECS['da2_metric_outdoor_small'].depth_kind == 'metric'
    assert FM.allowed('da2_small') and not FM.allowed('da2_base') and FM.allowed('da2_base', opt_in=True)
    cat = FM.catalogue()
    assert cat[0]['id'] == 'yolox_s' and {c['id'] for c in cat} >= set(FM.SPECS)
    assert not next(c for c in cat if c['id'] == 'da2_base')['allowed']


class _Server:
    """A fake Hugging Face: serves files by URL, counts requests, can corrupt one."""

    def __init__(self, files):
        self.files = files
        self.urls = []
        self.corrupt = None

    def __call__(self, url, timeout):
        self.urls.append(url)
        name = url.rsplit('/', 1)[1]
        data = self.files[name]
        if name == self.corrupt:
            data = data[:-1] + bytes([data[-1] ^ 1])
        return io.BytesIO(data)


def _fake_spec(monkeypatch, commercial=True):
    from vantage.core import foundation_models as FM
    files = {'config.json': b'{"a": 1}', 'model.safetensors': bytes(range(256)) * 4000}
    sp = FM.FoundationSpec('fake', 'Fake model', 'detect', 'owlv2', 'org/fake', 'a' * 40,
                           tuple(FM.FileSpec(n, len(d), hashlib.sha256(d).hexdigest()) for n, d in files.items()),
                           'Apache-2.0' if commercial else 'CC-BY-NC-4.0', 'https://example.org', commercial,
                           'a note', 'Someone: a paper, arXiv:0000.00000', 1.0)
    monkeypatch.setitem(FM.SPECS, 'fake', sp)
    return sp, files


def test_download_checks_every_file_and_keeps_nothing_wrong(monkeypatch, tmp_path):
    from vantage.core import foundation_models as FM
    sp, files = _fake_spec(monkeypatch)
    srv = _Server(files)
    assert FM.status('fake')['present'] is False
    # a corrupted file: refused, nothing of it kept, the good one stays
    srv.corrupt = 'model.safetensors'
    with pytest.raises(ValueError, match='SHA-256'):
        FM.download('fake', fetch=srv)
    d = FM.model_dir('fake')
    assert sorted(os.listdir(d)) == ['config.json']
    assert srv.urls[0] == 'https://huggingface.co/org/fake/resolve/' + 'a' * 40 + '/config.json'
    s = FM.status('fake')
    assert s['present'] and not s['verified'] and 'download it again' in s['text']
    # cancelled half way: no .part left behind
    srv.corrupt = None
    n = len(srv.urls)
    assert FM.download('fake', fetch=srv, cancel=lambda: len(srv.urls) > n) is None
    assert not [f for f in os.listdir(d) if f.endswith('.part')]
    # done: the good file already there is not fetched again
    seen = []
    path = FM.download('fake', fetch=srv, progress=lambda f, t: seen.append(f))
    assert path == d and FM.status('fake')['verified'] and seen[-1] == 1.0
    assert sum(u.endswith('config.json') for u in srv.urls) == 1
    # a file changed on disk is caught (size and content)
    with open(os.path.join(d, 'config.json'), 'wb') as fh:
        fh.write(b'{"a": 2}')
    assert not FM.status('fake')['verified']
    assert FM.remove('fake') and not os.path.isdir(d) and not FM.remove('fake')


def test_non_commercial_needs_the_opt_in(monkeypatch):
    from vantage.core import foundation_models as FM
    _, files = _fake_spec(monkeypatch, commercial=False)
    with pytest.raises(PermissionError, match='a note'):
        FM.download('fake', fetch=_Server(files))
    assert FM.download('fake', fetch=_Server(files), opt_in=True)
    from vantage.core import foundation_range as FR
    st = FR.FoundationSettings.from_dict({'depth_model': ''})
    st.detectors = ['fake']
    with pytest.raises(ValueError, match='a note'):
        FR._load_models(st)


def test_settings_are_tolerant_and_clamped():
    from vantage.core import foundation_range as FR
    st = FR.FoundationSettings.from_dict({'iou': 7, 'bootstrap': -3, 'detectors': ['yolox_s', 'nope', 'gdino_tiny'],
                                          'depth_model': 'nope', 'views': 'x', 'ranges_m': '50, 5, x, 400',
                                          'classes': [{'name': 'kangaroo', 'height': 'tall'}, {'name': ''}, 3],
                                          'scores': {'owlv2': 5, 'other': 0.4}, 'seeds': [3, 3, 'a'],
                                          'sweep_classes': ['kangaroo', 'unicorn'], 'source': 'sweep'})
    assert st.iou == 0.95 and st.bootstrap == 0 and st.detectors == ['yolox_s', 'gdino_tiny']
    assert st.depth_model == 'da2_small' and st.views == 'both' and st.ranges_m == [5.0, 50.0, 300.0]
    assert [c['name'] for c in st.classes] == ['kangaroo'] and st.classes[0]['query'] == 'a kangaroo'
    assert st.classes[0]['height'] == 1.0 and st.scores['owlv2'] == 0.99 and 'other' not in st.scores
    assert st.seeds == [3] and st.sweep_classes == ['kangaroo'] and st.source == 'sweep'
    assert FR.FoundationSettings.from_dict(st.to_dict()).to_dict() == st.to_dict()
    assert FR.FoundationSettings.from_dict(None).to_dict() == FR.FoundationSettings().to_dict()


def test_classes_from_ground_truth():
    from vantage.core import foundation_range as FR
    cl = FR.FoundationSettings().classes
    k = lambda o: FR.class_of_object(o, cl)       # noqa: E731
    name = lambda i: cl[i]['name'] if i >= 0 else None   # noqa: E731
    assert name(k({'class': 'pedestrian', 'category': 'human.pedestrian.adult'})) == 'pedestrian'
    assert name(k({'class': 'pedestrian', 'category': 'human.pedestrian.child'})) == 'child'
    assert name(k({'class': 'car', 'category': 'vehicle.truck'})) == 'truck'
    assert name(k({'class': 'traffic_sign', 'category': 'static.manmade'})) is None
    assert name(k({'class': 'static', 'category': 'movable_object.trafficcone'})) == 'traffic cone'
    i = FR.class_index(cl, 'child')
    assert [cl[j]['name'] for j in FR.accepted(cl, i)] == ['child', 'pedestrian']


# ======================================================================= no torch
@pytest.fixture
def no_torch(monkeypatch):
    real = builtins.__import__

    def fake(name, *a, **k):
        if name in ('torch', 'transformers') or name.startswith(('torch.', 'transformers.')):
            raise ImportError(f'No module named {name}')
        return real(name, *a, **k)
    monkeypatch.setattr(builtins, '__import__', fake)
    for m in [m for m in sys.modules if m in ('torch', 'transformers') or m.startswith(('torch.', 'transformers.'))]:
        monkeypatch.delitem(sys.modules, m)


def test_without_torch_it_says_so(no_torch):
    from vantage.core import foundation_models as FM
    from vantage.core import foundation_range as FR
    why = FM.torch_missing()
    assert why and 'needs torch and transformers' in why and 'pip install torch transformers' in why
    assert FM.device_info()['device'] == 'none'
    st = FR.FoundationSettings.from_dict({'detectors': ['owlv2_b16_ens'], 'recordings': ['/nowhere']})
    with pytest.raises(ValueError, match='needs torch and transformers'):
        FR.run(None, settings=st)
    with pytest.raises(ImportError):
        FM.OpenVocabDetector('owlv2_b16_ens', ['a car'])
    # YOLOX alone (OpenCV) and depth off need no torch: the check passes to the model files
    st = FR.FoundationSettings.from_dict({'detectors': ['yolox_s'], 'depth_model': ''})
    with pytest.raises(FileNotFoundError, match='YOLOX-S is not downloaded'):
        FR._check_models(st)
    assert len(FM.catalogue()) == len(FM.SPECS) + 4


# ======================================================================= the range fit
def _simulate(r50, slope, ranges, n, seed=0):
    """Trials at each range, P(detect) = σ(slope·log10(r50 / r)): R50 = r50, R90 = r50 / 10^(ln 9 / slope)."""
    rng = np.random.default_rng(seed)
    r = np.repeat(ranges, n)
    p = 1.0 / (1.0 + np.exp(-slope * np.log10(r50 / r)))
    return r, rng.random(len(r)) < p


def test_logistic_fit_recovers_a_known_r50_and_r90():
    from vantage.core import foundation_range as FR
    ranges = np.geomspace(5, 120, 14)
    hit_50 = hit_90 = 0
    for seed in range(20):
        r, d = _simulate(40.0, 6.0, ranges, 30, seed)
        f = FR.detect_fit(r, d, boot=200, seed=seed)
        r90 = 40.0 / 10 ** (math.log(9) / 6.0)
        assert f['x'][0] == pytest.approx(40.0, rel=0.2) and f['x'][1] == pytest.approx(r90, rel=0.3)
        assert f['ci'][0][0] < f['x'][0] < f['ci'][0][1] and f['ci'][1][0] < f['x'][1] < f['ci'][1][1]
        hit_50 += f['ci'][0][0] <= 40.0 <= f['ci'][0][1]
        hit_90 += f['ci'][1][0] <= r90 <= f['ci'][1][1]
        assert not f['beyond'][0] and not f['below'][0] and f['b'] > 0
    assert hit_50 >= 16 and hit_90 >= 16                    # 95 % intervals: about 19 of 20 cover the truth
    # the fitted curve is the simulated one
    assert FR.curve(f, [40.0])[0] == pytest.approx(0.5, abs=0.08)
    # every trial detected: beyond the farthest range; none: nearer than the nearest
    f = FR.detect_fit(ranges, np.ones(len(ranges), bool), boot=50)
    assert f['beyond'] == [True, True] and f['x'] == [None, None]
    f = FR.detect_fit(ranges, np.zeros(len(ranges), bool), boot=50)
    assert f['below'] == [True, True]
    assert FR.range_text({'range': f}) == '< 5 m'
    # near objects missed and far ones found (a fisheye cutting near cars off): no fall-off, no crossing claimed
    f = FR.detect_fit(ranges, ranges > 30, boot=50)
    assert f['flat'] and f['x'] == [None, None] and FR.range_text({'range': f}) == 'no fall-off'
    # three objects seen in many frames: no bootstrap interval (it would only repeat the same objects)
    r, d = _simulate(40.0, 6.0, ranges, 30, 1)
    f = FR.detect_fit(r, d, np.arange(len(r)) % 3, boot=50)
    assert f['clusters'] == 3 and f['ci'] == [None, None] and 'no interval' in FR.range_text({'range': f})
    # heights: P rises with them; a step at 30 px
    h = np.geomspace(5, 200, 300)
    f = FR.detect_fit(h, h >= 30, increasing=True, ratio=1.1, boot=50)
    assert f['x'][0] == pytest.approx(30, rel=0.12)


def test_cluster_bootstrap_widens_for_repeated_frames():
    """The same objects seen in ten frames each are not ten times the information: resampling the objects gives a
    wider interval than treating every frame as its own trial."""
    from vantage.core import foundation_range as FR
    r, d = _simulate(30.0, 5.0, np.geomspace(5, 90, 12), 4, seed=3)
    rr, dd = np.repeat(r, 10), np.repeat(d, 10)
    cl = np.repeat(np.arange(len(r)), 10)
    f_obj = FR.detect_fit(rr, dd, cl, boot=300, seed=1)
    f_frame = FR.detect_fit(rr, dd, None, boot=300, seed=1)
    w = lambda f: f['ci'][0][1] - f['ci'][0][0]      # noqa: E731
    assert f_obj['x'][0] == pytest.approx(f_frame['x'][0]) and w(f_obj) > 1.8 * w(f_frame)


# ======================================================================= depth
def test_depth_metrics_on_a_hand_made_pair():
    from vantage.core import foundation_range as FR
    gt = np.array([[2.0, 4.0, 8.0, 16.0], [32.0, 64.0, np.nan, 10.0]])
    pred = np.array([[2.2, 4.0, 6.0, 20.0], [32.0, 90.0, 5.0, -1.0]])
    ok, rel, d1 = FR.depth_scores(pred, gt, np.ones(gt.shape, bool))
    # the NaN truth and the negative prediction are left out
    assert ok.sum() == 6
    assert rel == pytest.approx([0.1, 0.0, 0.25, 0.25, 0.0, 26 / 64])
    assert list(d1) == [True, True, False, False, True, False]      # 6/8 → 1.333, 20/16 = 1.25 is not below 1.25
    n, s, dd = FR.depth_bins(pred, gt, np.ones(gt.shape, bool), [5.0, 20.0])
    assert list(n) == [2, 2, 2] and s == pytest.approx([0.1, 0.5, 26 / 64]) and list(dd) == [2, 0, 1]
    # usable: the farthest bin under the threshold (with enough pixels), contiguous from the nearest
    edges = [5.0, 10.0, 20.0, 40.0]
    nn = np.array([100, 100, 100, 5, 100])
    ss = np.array([5.0, 20.0, 8.0, 0.0, 50.0])                      # AbsRel 0.05, 0.2, 0.08, (too few), 0.5
    assert FR.usable_range(edges, nn, ss, 0.1, 50) == (20.0, 5.0)
    assert FR.usable_range(edges, nn, ss * 0 + 1, 0.1, 50) == (math.inf, math.inf)
    assert FR.usable_range(edges, nn, ss * 100, 0.1, 50) == (0.0, 0.0)


def test_ground_plane_scale_recovers_a_known_plane():
    """A pinhole view 1.4 m above a flat ground, pitched 12° down: relative inverse depth = 3 / z + 0.2 on the
    ground plus outliers (a box standing on it); the fit gives back α = 3, β = 0.2 and the ground's depth."""
    from vantage.core import foundation_range as FR
    W, H, f = 160, 120, 90.0
    gy, gx = np.mgrid[0:H, 0:W].astype(float)
    d = np.stack([(gx - W / 2) / f, (gy - H / 2) / f, np.ones_like(gx)], -1)
    p = math.radians(12)
    # view optical (x right, y down, z forward) -> vehicle (x forward, y left, z up), pitched down
    R = np.array([[0, -math.sin(p), math.cos(p)], [-1, 0, 0], [0, -math.cos(p), -math.sin(p)]])
    dv = d @ R.T
    z_true = np.where(dv[..., 2] < -1e-6, -1.4 / dv[..., 2], np.nan)
    disp = np.where(np.isfinite(z_true), 3.0 / z_true + 0.2, 0.05)
    disp[85:100, 60:80] = 3.0 / 8.0 + 0.2                          # a box at 8 m standing on nearer ground
    depth, info = FR.ground_scale(disp, dv, 1.4, np.ones((H, W), bool))
    assert info['alpha'] == pytest.approx(3.0, rel=1e-3) and info['beta'] == pytest.approx(0.2, abs=1e-3)
    ok = np.isfinite(z_true) & (z_true < 60)
    ok[85:100, 60:80] = False
    assert np.nanmax(np.abs(depth[ok] - z_true[ok]) / z_true[ok]) < 1e-3
    assert depth[90, 70] == pytest.approx(8.0, rel=1e-3) and 0.5 < info['inliers'] < 1.0
    assert FR.ground_scale(disp, dv, 1.4, np.zeros((H, W), bool))[0] is None


# ======================================================================= the sweep
def _fake():
    name = 'fake_carla_for_tests'
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, os.path.join(FAKE_DIR, 'carla', '__init__.py'))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _runner(calls):
    from vantage.sim import carla_bridge as CB
    fc = _fake()

    def runner(rig_json, out, progress, cancel):
        calls.append(out)
        fc.reset()
        doc, remaps = CB.load_rig(rig_json)
        a = CB.resolve_args(CB.parser().parse_args([rig_json, out]), doc)
        CB.Bridge(fc, doc, remaps, out, a, None, cancel).run()
    return runner


def _project():
    from vantage.core import samples
    p = samples.tigor_sample_project()
    v = next(v for v in p.variants if v.sensors)
    p.active_uid = v.uid
    p.analysis.sim = {'image_scale': 0.08, 'warmup_s': 0.25, 'min_tick_hz': 10.0, 'max_tick_hz': 30.0}
    return p


SWEEP = {'source': 'sweep', 'sweep_cameras': ['F'], 'sweep_classes': ['pedestrian', 'car'], 'ranges_m': [7, 16],
         'offsets_deg': [0, 25], 'road_lateral_m': [], 'seeds': [1], 'image_scale': 0.1, 'bootstrap': 30,
         'depth_model': '', 'views': 'both', 'view_width': 256}


def test_sweep_places_objects_where_it_says():
    from vantage.core import critical as CR
    from vantage.core import foundation_range as FR
    p = _project()
    st = FR.FoundationSettings.from_dict(dict(SWEEP, sweep_cameras=['F', 'ML'], road_lateral_m=[3.5]))
    pls, notes = FR.sweep_placements(p, st)
    cams = {c.tag: c for c in p.variant.cameras}
    box = CR.vehicle_box(p)
    by = {q.id: q for q in pls}
    for tag in ('F', 'ML'):
        c = cams[tag]
        for off in (0.0, 25.0):
            for r in (7.0, 16.0):
                q = next((q for q in pls if q.cls == 'pedestrian' and q.line == f'{tag} {off:+g}°' and
                          q.planned_m == r), None)
                if q is None:                             # too near the body (MIN_SEP_M): the rule says so
                    b = math.radians(c.yaw + off)
                    x, y = c.x + r * math.cos(b), c.y + r * math.sin(b)
                    assert box['rear_x'] - 2 < x < box['front_x'] + 2 and abs(y) < box['half_w'] + 2
                    continue
                b = math.radians(c.yaw + off)
                assert (q.x, q.y) == pytest.approx((c.x + r * math.cos(b), c.y + r * math.sin(b)), abs=1e-3)
                assert q.actor['family'] == 'walker' and q.actor['pose']['x'] == pytest.approx(q.x, abs=1e-3)
                # walkers face the vehicle
                to_v = math.degrees(math.atan2(-q.y, box['mid_x'] - q.x))
                assert abs((q.actor['pose']['yaw'] - to_v + 180) % 360 - 180) < 0.01
    road = [q for q in pls if q.line == 'road +3.5 m ahead' and q.cls == 'car']
    assert road and all(q.y == pytest.approx(3.5) for q in road)
    assert all(q.actor['family'] == 'vehicle' for q in pls if q.cls == 'car')
    # nothing on or next to the vehicle
    for q in pls:
        assert not (box['rear_x'] - 1 < q.x < box['front_x'] + 1 and abs(q.y) < box['half_w'] + 0.5)
    # scenes: no two placements of a scene overlap in the evaluated cameras or stand together
    pl = FR.sweep_plan(p, st)
    cams_ev = [c for c in p.variant.cameras if c.tag in ('F', 'ML')]
    cls = {c['name']: c for c in st.classes}
    for g in pl['groups']:
        qs = [by[i] for i in g]
        for i in range(len(qs)):
            for j in range(i + 1, len(qs)):
                a, b = qs[i], qs[j]
                ba, bb = (FR._image_boxes(p, cams_ev, x, cls[x.cls]) for x in (a, b))
                assert not any(u in bb and FR._overlap(ba[u], bb[u]) for u in ba)
    assert sorted(i for g in pl['groups'] for i in g) == sorted(by)
    assert len(pl['drives']) == len(pl['groups']) and 'drives' in pl['notes'][-1]
    # a class CARLA cannot stand in is said so
    st2 = FR.FoundationSettings.from_dict(dict(SWEEP, sweep_classes=['stroller']))
    assert any('stroller' in n for n in FR.sweep_placements(p, st2)[1])


@pytest.fixture(scope='module')
def sweep(tmp_path_factory):
    """One sweep through the fake carla (F only, pedestrians and cars at three ranges, two lines)."""
    from vantage.core import foundation_range as FR
    p = _project()
    st = FR.FoundationSettings.from_dict(SWEEP)
    out = str(tmp_path_factory.mktemp('sweep'))
    calls = []
    drives, plan = FR.run_sweep(p, st, out_dir=out, runner=_runner(calls))
    return p, st, drives, plan, calls


class GTDetector:
    """A detector that knows the ground truth: every labelled object of the study's classes in the image it is shown
    (a fisheye frame or a pinhole view, found by the image's hash) whose box is at least min_h px tall, with its own
    box and class, score 0.9; plus one false box per image (a corner nobody stands in)."""

    def __init__(self, recs, st, min_h=0.0, views=True):
        import cv2
        from vantage.core import detection_eval as DE
        from vantage.core import foundation_range as FR
        from vantage.core import groundtruth as GT
        self.table = {}
        self.min_h = min_h
        vst = st.view_settings()
        for path in recs:
            rec = DE.open_recording(path)
            for tag, cam in rec.cameras.items():
                views_ = DE.make_views(cam, st.view_settings(cam)) if views else []
                for fr in cam.frames:
                    img = cv2.imread(GT.path_of(rec.gt, fr['image']))
                    g = FR._objects(fr, st.classes, vst, None)
                    self.table[hashlib.sha1(img.tobytes()).hexdigest()] = g
                    inst = GT.load_instance(rec.gt, fr)
                    for v in views_:
                        vg = DE.view_gt(g, inst, v, vst)
                        self.table[hashlib.sha1(DE.remap(img, v).tobytes()).hexdigest()] = vg
        self.calls = 0

    def __call__(self, img):
        self.calls += 1
        g = self.table.get(hashlib.sha1(np.ascontiguousarray(img).tobytes()).hexdigest())
        if g is None:
            raise AssertionError('an image the detector was not shown before')
        keep = (g['scls'] >= 0) & (g['height'] >= self.min_h)
        boxes = np.vstack([g['box'][keep], [[0, 0, 3, 3]]]).astype(np.float32)
        return {'boxes': boxes, 'scores': np.full(len(boxes), 0.9, np.float32),
                'labels': np.r_[g['scls'][keep], 0].astype(int)}


def test_the_fake_carla_puts_them_there(sweep):
    from vantage.core import detection_eval as DE
    p, st, drives, plan, calls = sweep
    assert len(calls) == len(plan['drives']) >= 2
    by = {q.id: q for q in plan['placements']}
    seen = 0
    for d in drives:
        rec = DE.open_recording(d['dir'])
        rows = rec.run['actors']
        assert {r['id'] for r in rows} == set(plan['groups'][d['group']])
        for r in rows:
            q = by[r['id']]
            assert r['actor_id'] is not None and r['blueprint'] == q.actor['blueprint']
            assert (r['pose_vehicle']['x'], r['pose_vehicle']['y']) == pytest.approx((q.x, q.y), abs=0.05)
        # the ground truth's range from the camera is the planned one (the box centre: half its height up)
        cam = rec.cameras['F']
        for o in cam.frames[-1]['objects']:
            hit = next((r for r in rows if r['actor_id'] == o.get('actor_id')), None)
            if hit is None:
                continue
            q = by[hit['id']]
            h = hit['bbox']['size_lwh'][2]
            exp = math.dist(cam.t, (q.x, q.y, h / 2))
            assert o['range_m'] == pytest.approx(exp, rel=0.03, abs=0.15)
            seen += 1
    assert seen >= 3
    # a second run reuses the drives
    from vantage.core import foundation_range as FR
    again = []
    FR.run_sweep(p, st, out_dir=os.path.dirname(drives[0]['dir']), runner=_runner(again))
    assert again == []


def test_study_on_the_sweep_with_detectors_that_know_the_truth(sweep):
    from vantage.core import foundation_range as FR
    p, st, drives, plan, calls = sweep
    every = GTDetector([d['dir'] for d in drives], st)
    tall = GTDetector([d['dir'] for d in drives], st, min_h=30.0)
    st2 = FR.FoundationSettings.from_dict(dict(st.to_dict(), detectors=['every', 'tall']))
    st2.detectors = ['every', 'tall']
    res = FR.run(p, None, st2, detectors={'every': every, 'tall': tall}, drives=drives)
    T = res.trials
    assert len(T['hit']) and every.calls == tall.calls > 0
    # a trial per placement and camera frame: the clusters are the placements of each drive
    pids = {c.split(':', 1)[1] for c in T['cluster']}
    assert pids <= {q.id for q in plan['placements']}
    e = T['model'] == 'every'
    seen = ~T['absent']
    assert T['hit'][e & seen].all() and not T['hit'][T['absent']].any()     # every object with a pixel found
    t = (T['model'] == 'tall') & seen
    assert (T['hit'][t] == (T['height'][t] >= 30.0)).all()
    # one false detection per image (the corner box), none counted on the objects
    fd = {(f['model'], f['mode']): f for f in res.false_det}
    assert fd[('every', 'fisheye')]['per_image'] == 1.0 and fd[('every', 'pinhole')]['per_image'] == 1.0
    # fits exist per camera and class, and the geometric comparison is filled in
    f = res.fit('every', 'fisheye', 'F', 'pedestrian')
    assert f is not None and f['n'] == int((e & (T['mode'] == 'fisheye') & (T['cls'] == 0)).sum())
    assert f['geometric_m'] and f['geometric_m'] > 0
    assert f['range']['beyond'][0] or f['range']['x'][0] > 0
    # summary, doc, csv, files: JSON-able and complete
    s = FR.summary(res)
    json.dumps(s)
    doc = FR.to_doc(res, p)
    json.dumps(doc, default=str)
    assert doc['schema'] == FR.SCHEMA and doc['plan']['placements'] and doc['frame'].startswith('vehicle')
    names = [n for n, _ in FR.files(res, p)]
    assert 'foundation_range.json' in names and any(n.startswith('foundation_pdetect_f_pedestrian') for n in names)
    assert any(n.startswith('examples/') for n in names) and any(n.startswith('records/') for n in names)
    assert FR.to_csv(res).splitlines()[0].split(',')[:3] == ['model', 'mode', 'camera']


def test_study_on_a_recording_and_the_perception_records(sweep, tmp_path):
    """source 'recordings' on a fake-carla drive: every labelled object of a class is a trial; the height detector's
    H50 is its threshold; records are per detector and image kind, rig-level (an object once), in the grid's
    regions, and carry no paths or images."""
    from vantage.core import foundation_range as FR
    from vantage.core.regions import default_grid
    p, st, drives, plan, calls = sweep
    path = drives[0]['dir']
    st2 = FR.FoundationSettings.from_dict(dict(st.to_dict(), source='recordings', recordings=[path], frame_step=1,
                                               views='fisheye', bootstrap=20))
    st2.detectors = ['tall']
    tall = GTDetector([path], st2, min_h=30.0, views=False)
    res = FR.run(p, None, st2, detectors={'tall': tall})
    T = res.trials
    assert len(T['hit']) and set(T['mode']) == {'fisheye'}
    assert not T['absent'].any() and (T['hit'] == (T['height'] >= 30.0)).all()
    for f in res.fits:
        hf = f['height']
        if hf['x'][0] is not None and hf['tested'][0] < 30 < hf['tested'][1]:
            assert 20 < hf['x'][0] < 45
    assert res.records
    grid = default_grid(p)
    for r in res.records:
        assert r['schema'] == 'vantage.perception_record/1' and r['source'] == 'carla'
        assert r['detector']['id'] == 'tall' and r['detector']['backend'] == 'external:foundation-2d-fisheye'
        assert r['grid'] == grid.to_dict() and r['rig']['cameras'] and r['rig_hash'] == FR._sha(r['rig'])
        body = {k: v for k, v in r.items() if k not in ('id', 'created', 'provenance')}
        assert r['id'] == FR._sha(body) == FR.record_id(r)
        try:
            from vantage.core import perf_model as PM
            assert r['id'] == PM.record_id(r)
        except ImportError:
            pass
        for cl, regs in r['metrics'].items():
            for rid, m in regs.items():
                assert grid.parse_id(rid) is not None
                assert m['tp'] + m['fn'] == m['n_gt'] and set(m) >= {'recall', 'precision', 'ap', 'ate_m', 'fp'}
        text = json.dumps(r)
        assert path not in text and str(tmp_path) not in text and os.path.expanduser('~') not in text
    # rig level: an object counts once whatever the cameras
    n_obj = sum(m['n_gt'] for r in res.records for regs in r['metrics'].values() for m in regs.values())
    assert n_obj <= len({(c, s) for c, s in zip(T['cluster'], T['slot'])})
    # a re-run gives the same ids (provenance and time are left out of the id)
    again = FR.run(p, None, st2, detectors={'tall': tall})
    assert sorted(r['id'] for r in again.records) == sorted(r['id'] for r in res.records)
    out = FR.store_records(res, str(tmp_path / 'rec'), store=str(tmp_path / 'records.jsonl'))
    assert not out['errors'], out['errors']
    assert out['stored'] + len(out['written']) == len(res.records)
    if out['how'].startswith('core.perf_model'):                 # the store knows them: a second hand-over adds none
        again2 = FR.store_records(res, None, store=str(tmp_path / 'records.jsonl'))
        assert again2['stored'] == 0 and again2['duplicates'] == len(res.records)


def test_cancel_returns_none(sweep):
    from vantage.core import foundation_range as FR
    p, st, drives, plan, calls = sweep
    st2 = FR.FoundationSettings.from_dict(dict(st.to_dict(), detectors=['every']))
    st2.detectors = ['every']
    n = []

    def det(img):
        n.append(1)
        return {'boxes': np.zeros((0, 4)), 'scores': np.zeros(0), 'labels': np.zeros(0, int)}
    assert FR.run(p, None, st2, detectors={'every': det}, drives=drives, cancel=lambda: len(n) >= 2) is None


# ======================================================================= the page
from test_gui import app, win, pump, settle, _wait   # noqa: E402,F401  (fixtures)


def _open(win, name, cls):
    if name not in win._page_widgets:
        pg = cls(win.ctx)
        win.pages.addWidget(pg)
        win._page_widgets[name] = pg
    win.set_page(name)
    return win._page_widgets[name]


def test_page_manages_models_edits_classes_runs_and_exports(app, win, tmp_path, monkeypatch, sweep):
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest
    from vantage.core import foundation_models as FM
    from vantage.core import foundation_range as FR
    from vantage.ui.workspaces.foundation_ws import FoundationPage
    _, files = _fake_spec(monkeypatch)
    pg = _open(win, 'foundation', FoundationPage)
    pump(app, 0.3)
    # the model manager lists every model with its licence; the non-commercial ones cannot be ticked
    names = [pg.model_table.item(r, 1).text() for r in range(pg.model_table.rowCount())]
    assert 'OWLv2 B/16 ensemble' in names and 'YOLOX-S' in names and 'Fake model' in names
    r_base = names.index('Depth Anything V2 Base (relative)')
    assert 'opt-in' in pg.model_table.item(r_base, 3).text()
    assert not (pg.model_table.item(r_base, 0).flags() & Qt.ItemFlag.ItemIsEnabled)
    pg.opt_in.click()
    pump(app, 0.2)
    assert FR.settings_of(win.store.project).opt_in
    assert pg.model_table.item(r_base, 0).flags() & Qt.ItemFlag.ItemIsEnabled
    win.store.undo.undo()
    pump(app, 0.2)
    assert not FR.settings_of(win.store.project).opt_in and not pg.opt_in.isChecked()
    # download the (fake) model as a user does: select its row, press Download
    pg.model_table.selectRow(names.index('Fake model'))
    srv = _Server(files)
    pg.download(fetch=srv)
    _wait(app, lambda: not pg.dl_busy, 60)
    assert FM.status('fake')['verified'] and 'Downloaded' in pg.status_text
    r_fake = [pg.model_table.item(r, 1).text() for r in range(pg.model_table.rowCount())].index('Fake model')
    assert pg.model_table.item(r_fake, 5).text() == 'ready'
    pg.model_table.selectRow(r_fake)
    assert pg.remove_model() and not FM.status('fake')['present']
    # tick a detector in the table: stored, undoable
    r_g = names.index('Grounding DINO tiny')
    pg.model_table.item(r_g, 0).setCheckState(Qt.CheckState.Checked)
    pump(app, 0.2)
    assert 'gdino_tiny' in FR.settings_of(win.store.project).detectors
    win.store.undo.undo()
    pump(app, 0.2)
    assert 'gdino_tiny' not in FR.settings_of(win.store.project).detectors
    # the class list: add a class, type its query into the table, undo
    n0 = pg.class_table.rowCount()
    pg.add_class('kangaroo')
    pump(app, 0.2)
    assert pg.class_table.rowCount() == n0 + 1
    pg.class_table.item(n0, 1).setText('a kangaroo crossing')
    pump(app, 0.2)
    assert FR.settings_of(win.store.project).classes[-1]['query'] == 'a kangaroo crossing'
    win.store.undo.undo()
    win.store.undo.undo()
    pump(app, 0.2)
    assert pg.class_table.rowCount() == n0
    # a recording; typed fields; run with injected detectors as a user presses Run
    path = sweep[2][0]['dir']
    assert pg.add_recording(path)
    pump(app, 0.2)
    assert pg.recs.count() == 1
    f = pg.fields['frame_step']
    f.lineEdit().selectAll()
    QTest.keyClicks(f.lineEdit(), '1')
    QTest.keyClick(f.lineEdit(), Qt.Key.Key_Return)
    pump(app, 0.2)
    assert FR.settings_of(win.store.project).frame_step == 1
    st = FR.settings_of(win.store.project)

    def store(p):
        p.analysis.foundation = dict(st.to_dict(), detectors=['yolox_s'], depth_model='', views='fisheye',
                                     bootstrap=20)
    win.store.edit('analysis', store, 'test: yolox only')
    pump(app, 0.2)
    st = FR.settings_of(win.store.project)
    pg.detectors = {'yolox_s': GTDetector([path], st, min_h=20.0, views=False)}
    QTest.mouseClick(pg.b_run, Qt.MouseButton.LeftButton)
    pump(app, 0.2)
    _wait(app, lambda: not pg.busy, 300)
    res = pg.result
    assert res is not None, pg.status.text()
    assert pg.range_table.rowCount() == len([f for f in res.fits if f['subset'] == 'all']) >= 1
    assert pg.tiles['trials'].value == str(len(res.trials['hit']))
    pg.tabs.setCurrentIndex(1)
    pump(app, 0.2)
    assert pg.cam_pick.count() >= 1 and pg.plot.pixmap() is not None and not pg.plot.pixmap().isNull()
    pg.tabs.setCurrentIndex(3)
    pump(app, 0.2)
    assert pg.example_pick.count() >= 1 and not pg.example.pixmap().isNull()
    # the records to the rig model's store (a test store), once
    out = pg.send_records(str(tmp_path / 'store.jsonl'))
    assert out and not out['errors'] and (out['stored'] >= 1 or not out['how'].startswith('core.perf_model'))
    # stale after an edit, export writes the files
    pg._set('iou 0.6', iou=0.6)
    pump(app, 0.2)
    assert pg.stale and 'Run again' in pg.status.text()
    paths = pg.export(str(tmp_path / 'exp'))
    assert any(p_.endswith('foundation_range.json') for p_ in paths)
    with open(next(p_ for p_ in paths if p_.endswith('foundation_range.json'))) as fh:
        assert json.load(fh)['provenance']
    # the sweep shows its plan
    pg._set('frames from sweep', source='sweep')
    pump(app, 0.2)
    assert pg.sweep_grid.isVisibleTo(pg) and not pg.recs.isVisibleTo(pg)
    assert 'drives' in pg.sweep_note.text() or 'not a visible camera' in pg.sweep_note.text()


# ======================================================================= the real models
@_real('owlv2_b16_ens')
def test_real_owlv2_finds_the_pedestrian_on_a_carla_frame(monkeypatch):
    import cv2
    from vantage.core import detection_eval as DE
    from vantage.core import foundation_models as FM
    from vantage.core import groundtruth as GT
    if not os.path.isdir(REAL_REC):
        pytest.skip('the CARLA recording is not here')
    monkeypatch.setenv(FM.ENV_DIR, REAL_DIR)
    assert FM.status('owlv2_b16_ens')['verified']
    rec = DE.open_recording(REAL_REC)
    fr = rec.cameras['F'].frames[10]
    img = cv2.imread(GT.path_of(rec.gt, fr['image']))
    det = FM.OpenVocabDetector('owlv2_b16_ens', ['a pedestrian', 'a car'])
    out = det(img)
    det.close()
    ped = next(o for o in fr['objects'] if o['class'] == 'pedestrian' and o['range_m'] < 15)
    sel = (out['labels'] == 0) & (out['scores'] >= 0.2)
    assert DE.iou_matrix(out['boxes'][sel], [ped['bbox']]).max() >= 0.5


@_real('da2_small')
def test_real_depth_anything_scaled_from_the_ground(monkeypatch):
    import cv2
    from vantage.core import detection_eval as DE
    from vantage.core import foundation_models as FM
    from vantage.core import foundation_range as FR
    from vantage.core import groundtruth as GT
    if not os.path.isdir(REAL_REC):
        pytest.skip('the CARLA recording is not here')
    monkeypatch.setenv(FM.ENV_DIR, REAL_DIR)
    rec = DE.open_recording(REAL_REC)
    cam = rec.cameras['F']
    fr = cam.frames[10]
    st = FR.FoundationSettings()
    v = DE.make_views(cam, st.view_settings(cam))[1]
    assert v.w == 864 and abs(v.K[0, 0] - cam.K[0, 0]) < 0.02 * cam.K[0, 0]     # the camera's own focal length
    img = DE.remap(cv2.imread(GT.path_of(rec.gt, fr['image'])), v)
    rng = DE.remap(GT.load_range(rec.gt, fr).astype(np.float32), v, nearest=True)
    dv, nrm = FR._view_rays(v, cam)
    gt = np.where(rng > 0, rng / nrm, np.nan)
    dm = FM.DepthModel('da2_small')
    disp = dm(img)
    dm.close()
    depth, info = FR.ground_scale(disp, dv, float(cam.t[2]), np.isfinite(gt))
    ok, rel, d1 = FR.depth_scores(depth, gt, np.isfinite(gt) & (gt < 10))
    assert info['inliers'] > 0.5 and float(np.mean(rel)) < 0.15 and float(np.mean(d1)) > 0.85
