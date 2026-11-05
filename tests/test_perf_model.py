"""The learned rig-performance model (core.perf_model and its page): the record store (round trip, de-duplication,
filters, per-project opt-in), the record validator, the anonymised export, the physics features against hand-computed
pixel densities for a pinhole and a fisheye at known directions, the Gaussian process's marginal-likelihood gradient
against finite differences, recovery of a known synthetic recall(px, occlusion) with calibrated intervals, "not
trained" below the minimum, cross-validation against the baselines, active learning (propose) on a 1-D toy problem
with a known answer, a campaign with a fake evaluator that improves the model, the interim evaluator on a synthetic
recording with known hits and misses, and the page driven like a user."""
import json
import math
import os

import numpy as np
import pytest

from vantage.core import perf_model as PM
from vantage.core import regions as RG


@pytest.fixture(autouse=True)
def _data_dir(tmp_path, monkeypatch):
    """Never the user's data folder."""
    monkeypatch.setenv('VANTAGE_PERF_DIR', str(tmp_path / 'perf'))
    PM._STORES.clear()


# ======================================================================= synthetic records
GRID1 = RG.RegionGrid(sectors=4, range_edges=(10.0, 30.0))
NF = len(PM.FEATURE_NAMES)
NR = GRID1.n_regions
PX = PM.FEATURE_NAMES.index('log2_px_obj')
OCC = PM.FEATURE_NAMES.index('body_share')


def truth(lpx, occ):
    """The analytic recall the synthetic records are drawn from."""
    return PM.sigmoid(-4.0 + 1.1 * lpx - 3.0 * occ)


def synth_feats(lpx, occ, grid=GRID1, cls='pedestrian', jitter=None):
    """Features of a rig whose every region has the given log2 pixels and body share (arrays per region)."""
    n = grid.n_regions
    X = np.zeros((n, NF))
    X[:, PX] = lpx
    X[:, OCC] = occ
    if jitter is not None:
        X[:, 1] = X[:, PX] - 0.3 + jitter
    return {'names': list(PM.FEATURE_NAMES), 'regions': [r for _, _, r, _ in grid.regions()], 'classes': [cls],
            'X': {cls: X}, 'grid': grid.to_dict(), 'rig_hash': ''}


def rig_of(i):
    return {'vehicle': {'name': 'Test car', 'length_m': 4.0, 'width_m': 1.8, 'height_m': 1.5},
            'cameras': [{'tag': f'C{i}', 'module': '', 'lens': '', 'width': 640, 'height': 480, 'x': 1.0 + 0.01 * i,
                         'y': 0.0, 'z': 1.2, 'yaw': 0.0, 'pitch': 10.0, 'roll': 0.0, 'hfov_deg': 90.0,
                         'vfov_deg': 70.0, 'max_theta_deg': 60.0, 'projection': 'rectilinear'}], 'sensors': []}


def synth_record(i, lpx, occ, rng, scene='S1', grid=GRID1, n=(20, 200), cls='pedestrian'):
    ns = rng.integers(n[0], n[1], grid.n_regions)
    p = truth(lpx, occ)
    k = rng.binomial(ns, p)
    m = {cls: {rid: PM.metrics_cell(int(nn), int(kk), 0) for (_, _, rid, _), nn, kk in zip(grid.regions(), ns, k)}}
    return PM.make_record(source='synthetic-test', scene={'kind': 'synthetic', 'town': scene, 'seed': 1},
                          detector={'id': 'analytic', 'backend': 'external:test'}, rig=rig_of(i), grid=grid,
                          metrics=m)


def synth_set(n_rigs, seed, scenes=('S1', 'S2', 'S3')):
    rng = np.random.default_rng(seed)
    docs, feats = [], {}
    for i in range(n_rigs):
        lpx = rng.uniform(1.0, 8.0, GRID1.n_regions)
        occ = rng.uniform(0.0, 1.0, GRID1.n_regions)
        d = synth_record(seed * 1000 + i, lpx, occ, rng, scene=scenes[i % len(scenes)])
        docs.append(d)
        feats[d['rig_hash']] = synth_feats(lpx, occ)
    return docs, feats


# ======================================================================= the store and the validator
def test_store_round_trip_dedup_and_filters(tmp_path):
    docs, _ = synth_set(4, 1)
    st = PM.Store(tmp_path / 'r.jsonl')
    for d in docs:
        assert PM.add_record(d, st)['added']
    again = PM.add_record(docs[0], st)
    assert not again['added'] and again.get('duplicate')
    back = PM.records(store=st)
    assert [d['id'] for d in back] == [d['id'] for d in docs]
    assert json.loads(json.dumps(back[1])) == docs[1]
    assert len(PM.records({'town': 'S1'}, st)) == 2
    assert len(PM.records({'rig_hash': docs[2]['rig_hash']}, st)) == 1
    assert len(PM.records(lambda d: d['scene']['town'] == 'S3', st)) == 1
    assert st.remove([docs[0]['id']]) == 1 and len(st.records()) == 3
    # a record without id/created gets them; its id is the canonical hash
    d = {k: v for k, v in docs[0].items() if k not in ('id', 'created')}
    r = PM.add_record(d, st)
    assert r['added'] and r['id'] == docs[0]['id']
    # the default store is under VANTAGE_PERF_DIR, the project's when it opts in
    assert PM.default_store_path().startswith(str(tmp_path / 'perf'))
    from vantage.core import samples
    p = samples.tigor_sample_project()
    assert PM.project_store_path(p) is None
    p.analysis.perf_model = {'store_in_project': True}
    p.path = str(tmp_path / 'proj' / 'car.vantage')
    assert PM.project_store_path(p) == str(tmp_path / 'proj' / 'car.perception_records.jsonl')
    assert PM.store_for(p).path == PM.project_store_path(p)


def test_validation_rejects_bad_records():
    d, = synth_set(1, 2)[0]
    assert PM.validate_record(d) == []

    def bad(fn, word):
        x = json.loads(json.dumps(d))
        fn(x)
        if word != 'id does not match':
            x['rig_hash'] = PM.rig_hash(x['rig']) if isinstance(x.get('rig'), dict) else x.get('rig_hash')
            x['id'] = PM.record_id(x)
        errs = PM.validate_record(x)
        assert any(word in e for e in errs), (word, errs)
    bad(lambda x: x.update(schema='other/1'), 'schema')
    bad(lambda x: x['metrics']['pedestrian']['S0R0'].update(tp=5, fn=1, n_gt=7), 'tp + fn')
    bad(lambda x: x['metrics']['pedestrian']['S0R0'].update(recall=0.123), 'recall')
    bad(lambda x: x['metrics'].update(dog={}), 'unknown class')
    bad(lambda x: x['metrics']['pedestrian'].update(S9R9=PM.metrics_cell(1, 1, 0)), 'not a region')
    bad(lambda x: x['metrics']['pedestrian']['S0R0'].update(precision=1.5), 'precision')
    bad(lambda x: x['metrics']['pedestrian']['S0R0'].update(n_gt=-1), 'counts')
    bad(lambda x: x['rig']['cameras'][0].update(x='a'), 'not a number')
    bad(lambda x: x['rig']['cameras'][0].update(note='/home/someone/drive/F.png'), 'user path')
    bad(lambda x: x['rig']['cameras'][0].update(img='iVBORw0KGgoAAAANSUhEUgAA'), 'image data')
    bad(lambda x: x['rig']['cameras'][0].update(blob='x' * 5000), 'string of')
    bad(lambda x: x.update(source='guess'), 'source')
    bad(lambda x: x['metrics']['pedestrian']['S0R0'].update(tp=0, fn=0, n_gt=0), 'recall without')
    x = json.loads(json.dumps(d))
    x['metrics']['pedestrian']['S0R1']['tp'] += 0                  # unchanged: still valid
    x['rig']['cameras'][0]['z'] = 1.3                              # the rig changed after the hash
    assert any('rig_hash' in e for e in PM.validate_record(x))
    bad(lambda x: x.update(created='yesterday'), 'ISO-8601')
    assert PM.validate_record([1, 2]) == ['not a JSON object']
    st = PM.Store()
    assert not st.add({'schema': PM.RECORD_SCHEMA})['added']


def test_anonymised_export_has_no_names_or_paths_and_reimports(tmp_path):
    docs, _ = synth_set(3, 3)
    from vantage.core import provenance as PV, samples
    p = samples.tigor_sample_project()
    p.path = str(tmp_path / 'secret-project' / 'mine.vantage')
    prov = PV.block(p, None, study=PM.RECORD_SCHEMA)
    d = dict(docs[0], provenance=prov, rig_label='secret rig A')
    d['rig']['vehicle']['name'] = 'Secret Vehicle X'
    d['rig_hash'] = PM.rig_hash(d['rig'])
    d['id'] = PM.record_id(d)
    st = PM.Store(tmp_path / 'a.jsonl')
    assert st.add(d)['added']
    st.add_many(docs[1:])
    out = tmp_path / 'pool.jsonl'
    assert PM.export_records(out, anonymise=True, store=st) == 3
    text = out.read_text()
    for word in ('Secret Vehicle X', p.name, 'secret-project', str(tmp_path), 'mine.vantage', 'Test car'):
        assert word not in text, word
    rows = [json.loads(line) for line in text.splitlines()]
    assert all(PM.validate_record(r) == [] for r in rows)
    assert rows[0]['origin_id'] == d['id'] and rows[0]['anonymised']
    assert rows[0]['provenance']['project']['sha256'] == prov['project']['sha256']
    assert rows[0]['metrics'] == d['metrics'] and rows[0]['rig']['cameras'] == d['rig']['cameras']
    # back into the same store: recognised as the same records
    r = PM.import_records(out, st)
    assert r['added'] == 0 and r['duplicates'] == 3
    # into a fresh store: added; a second time: duplicates
    st2 = PM.Store(tmp_path / 'b.jsonl')
    assert PM.import_records(out, st2)['added'] == 3
    assert PM.import_records(out, st2)['duplicates'] == 3
    bad = tmp_path / 'bad.jsonl'
    bad.write_text('{"schema": "vantage.perception_record/1"}\nnot json\n')
    assert PM.import_records(bad, st2)['rejected'] == 2


# ======================================================================= features
def _cam_block(proj, hfov, yaw=0.0, pitch=0.0, w=1000, h=800, maxth=None):
    return {'tag': 'C', 'module': '', 'lens': '', 'width': w, 'height': h, 'x': 0.0, 'y': 0.0, 'z': 1.0, 'yaw': yaw,
            'pitch': pitch, 'roll': 0.0, 'hfov_deg': hfov, 'vfov_deg': 0.0, 'max_theta_deg': maxth,
            'projection': proj}


def test_features_match_hand_computed_pixel_density():
    # pinhole: f = (w/2) / tan(hfov/2); at angle θ off the axis and distance d: px/m = f / (d cos^1.5 θ)
    c = _cam_block('rectilinear', 90.0)
    cam = PM._rig_cams_from_block({'cameras': [c]})[0][0]
    f_px = 500.0 / math.tan(math.radians(45.0))
    for th_deg, d in ((0.0, 10.0), (20.0, 15.0), (35.0, 7.0)):
        th = math.radians(th_deg)
        P = np.array([[d * math.cos(th), d * math.sin(th), 1.0]])          # level, th to the left
        u, v, t, ok = cam.intr.project_dirs((P - cam.C) @ cam.R)
        assert ok[0] and abs(t[0] - th) < 1e-9
        got = cam.intr.density_px_per_cm(t, d)[0] * 100
        assert abs(got - f_px / (d * math.cos(th) ** 1.5)) < 1e-6 * got
        assert abs(u[0] - (500.0 - f_px * math.tan(th))) < 1e-6           # left of the axis: smaller u
    # equidistant fisheye: f = (w/2) / (hfov/2 in rad); px/m = f sqrt(θ / sin θ) / d
    c = _cam_block('equidistant', 180.0, maxth=95.0)
    cam = PM._rig_cams_from_block({'cameras': [c]})[0][0]
    f_px = 500.0 / (math.pi / 2)
    for th_deg, d in ((30.0, 5.0), (60.0, 12.0), (85.0, 3.0)):
        th = math.radians(th_deg)
        P = np.array([[d * math.cos(th), -d * math.sin(th), 1.0]])         # to the right
        u, v, t, ok = cam.intr.project_dirs((P - cam.C) @ cam.R)
        assert ok[0] and abs(t[0] - th) < 1e-9
        got = cam.intr.density_px_per_cm(t, d)[0] * 100
        assert abs(got - f_px * math.sqrt(th / math.sin(th)) / d) < 1e-6 * got
        assert abs(u[0] - (500.0 + f_px * th)) < 1e-6
    # describe_rig: pixels on a pedestrian ahead = height × px/m of the best camera, median over the region's points
    grid = RG.RegionGrid(sectors=4, range_edges=(10.0, 20.0))
    rig = {'vehicle': {}, 'cameras': [_cam_block('rectilinear', 90.0)], 'sensors': []}
    f = PM.describe_rig(rig, grid=grid, classes=('pedestrian',))
    G, _ = PM._region_points(grid)
    cam = PM._rig_cams_from_block(rig)[0][0]
    h = PM.CLASS_SIZE['pedestrian'][0]
    for k, rid in enumerate(f['regions']):
        P = np.c_[G[k], np.full(len(G[k]), h / 2)]
        d = P - cam.C
        dist = np.linalg.norm(d, axis=1)
        u, v, t, ok = cam.intr.project_dirs(d @ cam.R)
        pxm = np.where(ok, cam.intr.density_px_per_cm(t, dist) * 100, 0)
        want = np.median(np.log2(np.maximum(h * pxm, PM.PX_FLOOR)))
        assert abs(f['X']['pedestrian'][k, PX] - want) < 1e-9, rid
    ahead = f['regions'].index('S0R1')
    behind = f['regions'].index('S2R1')
    assert f['X']['pedestrian'][ahead, PM.FEATURE_NAMES.index('seen_share')] == 1.0
    assert f['X']['pedestrian'][behind, PM.FEATURE_NAMES.index('seen_share')] == 0.0
    assert f['X']['pedestrian'][behind, PX] == math.log2(PM.PX_FLOOR)


def test_features_of_a_variant_and_its_record_block_agree_and_are_fast():
    from vantage.core import samples
    p = samples.tigor_sample_project()
    v = next(v for v in p.variants if v.sensors)
    f1 = PM.describe_rig(p, v)
    rb = PM.rig_block(p, v)
    f2 = PM.describe_rig(rb, project=p)
    for c in PM.CLASSES:
        assert np.allclose(f1['X'][c], f2['X'][c])
    assert f1['occlusion'] and f1['lidars'] == 1 and f1['radars'] == 5 and f1['cameras'] == 6
    import time
    t = time.perf_counter()
    PM.describe_rig(p, v)
    assert time.perf_counter() - t < 0.2                    # cube maps cached: features in milliseconds
    # the body hides part of some view; the lidar puts points on near objects; a quarter-resolution record halves px/m
    X = f1['X']['pedestrian']
    assert X[:, PM.FEATURE_NAMES.index('log2_lidar_pts')].max() > 3
    # pixels as the detector sees them: the 1920 × 1536 modules run at 1280 × 1024 (DETECTOR_INPUT_PX), so a record
    # at 2/3 scale has the same features and one at 1/3 scale (640 × 512) half the px/m
    assert PM.DETECTOR_INPUT_PX == 1280 and f1['input_px'] == 1280
    seen = f1['X']['car'][:, PX] > 2
    for div, drop in ((1.5, 0.0), (3.0, 1.0)):
        small = json.loads(json.dumps(rb))
        for c in small['cameras']:
            c['width'], c['height'] = int(round(c['width'] / div)), int(round(c['height'] / div))
        f3 = PM.describe_rig(small, project=p)
        assert np.allclose(f3['X']['car'][seen, PX], f1['X']['car'][seen, PX] - drop, atol=0.05)
    # without the cap, the full-size module has 1.5 × the px/m of the detector's view
    f4 = PM.describe_rig(p, v, input_px=0)
    assert np.allclose(f4['X']['car'][seen, PX], f1['X']['car'][seen, PX] + math.log2(1.5), atol=0.05)


# ======================================================================= the Gaussian process
def test_marginal_likelihood_gradient_matches_finite_differences():
    rng = np.random.default_rng(4)
    d, n = 3, 40
    X, r, s2 = rng.normal(size=(n, d)), rng.normal(size=n), rng.uniform(0.05, 0.5, n)
    th = np.r_[np.log([0.7, 1.3, 2.0]), math.log(0.8), math.log(0.2), math.log(1.7)]
    f, g = PM._nll_and_grad(th, X, r, s2, d)
    num = np.array([(PM._nll_and_grad(th + e, X, r, s2, d)[0] - PM._nll_and_grad(th - e, X, r, s2, d)[0]) / 2e-6
                    for e in np.eye(len(th)) * 1e-6])
    assert np.abs(num - g).max() < 1e-5 * max(1.0, np.abs(g).max())


def _synthetic_rows(m, seed):
    r = np.random.default_rng(seed)
    lpx, occ, n = r.uniform(1, 8, m), r.uniform(0, 1, m), r.integers(5, 200, m)
    X = np.zeros((m, NF))
    X[:, PX], X[:, OCC] = lpx, occ
    X[:, 1:5] = r.normal(size=(m, 4)) * 0.1
    p = truth(lpx, occ)
    return X, r.binomial(n, p), n, p


def test_gp_recovers_a_known_function_with_calibrated_intervals():
    X, k, n, _ = _synthetic_rows(400, 2)
    Xt, kt, nt, pt = _synthetic_rows(300, 3)
    cm = PM.ClassModel('pedestrian').fit(X, k, n, PM.GPSettings(restarts=2))
    mu, var = cm.predict(Xt)
    y, s2 = PM.empirical_logit(kt, nt)
    a = np.abs(y - mu) / np.sqrt(var + cm.noise_var(s2))
    cov95, cov68 = np.mean(a <= PM.Z95), np.mean(a <= PM.Z68)
    assert 0.88 <= cov95 <= 0.99, cov95
    assert 0.58 <= cov68 <= 0.78, cov68
    pm, _ = PM.logistic_moments(mu, var)
    assert np.sqrt(np.mean((pm - pt) ** 2)) < 0.03
    assert cm.b > 0.5 and cm.phi < 2.0                       # the monotone prior; no dispersion in binomial data
    # the analytic mean gradient against finite differences, and the FITC path agreeing with the exact one
    G = cm.grad(Xt[:2])
    e = 1e-5
    num = np.array([[(cm.predict(Xt[i:i + 1] + e * np.eye(NF)[j])[0][0] -
                      cm.predict(Xt[i:i + 1] - e * np.eye(NF)[j])[0][0]) / (2 * e) for j in range(NF)]
                    for i in range(2)])
    assert np.abs(num - G).max() < 1e-5
    sp = PM.ClassModel('pedestrian').fit(X, k, n, PM.GPSettings(restarts=1, sparse_n=100, inducing=150))
    assert sp.kind == 'fitc'
    mu2, var2 = sp.predict(Xt)
    assert np.sqrt(np.mean((PM.logistic_moments(mu2, var2)[0] - pt) ** 2)) < 0.04
    # a saved and reloaded class model predicts the same
    back = PM.ClassModel.from_dict(json.loads(json.dumps(cm.to_dict())))
    assert np.allclose(back.predict(Xt)[0], mu) and np.allclose(back.predict(Xt)[1], var)


def test_sparse_posterior_keeps_the_group_offsets_exactly():
    """FITC with every training row as an inducing input, the scene, town and region offsets as low-rank terms, must
    be the exact posterior: for a new town, a known town and a known scene set, mean and full covariance."""
    rng = np.random.default_rng(0)
    n = 300
    X = rng.normal(size=(n, NF))
    X[:, PX] = rng.uniform(0, 8, n)
    sc = rng.integers(0, 6, n)
    tw = sc // 2
    rg = rng.integers(0, 5, n)
    N = rng.integers(5, 80, n)
    lg = -2 + 0.5 * X[:, PX] + rng.normal(0, 0.5, 6)[sc] + rng.normal(0, 0.8, 3)[tw] + rng.normal(0, 0.6, (3, 5))[tw, rg]
    k = rng.binomial(N, PM.sigmoid(lg))
    cm = PM.ClassModel('pedestrian').fit(X, k, N, PM.GPSettings(restarts=1, max_iter=40), scenes=sc, regions=rg,
                                         towns=tw)
    assert cm.kind == 'exact' and cm.levels == list(PM.GROUP_LEVELS) and cm.gv['town_region'] > 0.05
    f = PM.ClassModel.__new__(PM.ClassModel)
    f.__dict__.update(cm.__dict__)
    f._posterior_fitc_at(cm.Ztrain, cm.Ztrain, cm.rtrain, cm.s2train)
    Xt = rng.normal(size=(7, NF))
    Xt[:, PX] = rng.uniform(0, 8, 7)
    for kw in ({}, {'scenes': ['9'] * 7, 'regions': list(range(7)), 'towns': ['2'] * 7},
               {'scenes': ['1'] * 7, 'regions': [0, 1, 2, 3, 4, 0, 1], 'towns': ['0'] * 7}):
        mu_e, cov_e = cm.predict(Xt, full_cov=True, **kw)
        mu_f, cov_f = f.predict(Xt, full_cov=True, **kw)
        assert np.abs(mu_e - mu_f).max() < 1e-5 and np.abs(cov_e - cov_f).max() < 1e-6
        assert np.allclose(np.diag(cov_f), f.predict(Xt, **kw)[1])
    # a known town narrows the interval against a new one; the saved model predicts the same
    v_new = cm.predict(Xt)[1]
    v_known = cm.predict(Xt, scenes=['9'] * 7, regions=list(range(7)), towns=['2'] * 7)[1]
    assert np.all(v_known < v_new)
    back = PM.ClassModel.from_dict(json.loads(json.dumps(cm.to_dict())))
    assert np.allclose(back.predict(Xt)[0], cm.predict(Xt)[0]) and back.levels == cm.levels


def test_not_trained_below_the_minimum_and_predicts_nothing():
    docs, feats = synth_set(3, 5)
    m = PM.Model(PM.PerfModelSettings(cv=False)).fit(docs, feats_of=lambda d: feats[d['rig_hash']])
    assert not m.trained and m.reason() == 'not trained (3 records, need 8)'
    p = m.predict(synth_feats(np.full(NR, 5.0), np.zeros(NR)))
    assert p['classes'] == {} and not p['trained']
    with pytest.raises(RuntimeError, match='not trained'):
        PM.propose([synth_feats(np.full(NR, 5.0), np.zeros(NR))], 1, model=m)
    assert PM.load_model() is None


def test_model_trains_cross_validates_and_beats_the_baselines(tmp_path):
    docs, feats = synth_set(24, 6)
    st = PM.PerfModelSettings(classes=['pedestrian'], restarts=2, objective={'pedestrian': 1.0})
    m = PM.Model(st).fit(docs, feats_of=lambda d: feats[d['rig_hash']])
    assert m.trained and m.n_records == 24 and m.n_scenes == 3
    cv = m.cv[('pedestrian', 'recall')]
    for by in ('scene', 'rig'):
        c = cv[by]
        assert c['ok'] and c['gp']['n'] == 24 * GRID1.n_regions
        assert 0.85 <= c['gp']['coverage95'] <= 1.0
        assert c['gp']['rmse'] < c['logistic']['rmse'] < c['constant']['rmse']
        assert c['gp']['nlpd'] < c['constant']['nlpd']
    assert cv['scene']['folds'] == 3 and cv['rig']['folds'] == 10
    rows = PM.cv_table(m)
    assert {r['method'] for r in rows} == {'gp', 'logistic', 'constant'}
    # predict: means and std per region, intervals ordered, rig summary
    f = synth_feats(np.linspace(1, 8, NR), np.zeros(NR))
    p = m.predict(f)
    d = p['classes']['pedestrian']
    assert np.all(d['lo95'] <= d['lo68']) and np.all(d['lo68'] <= d['mean'] + 1e-9)
    assert np.all(d['mean'] <= d['hi68'] + 1e-9) and np.all(d['hi68'] <= d['hi95'])
    assert np.all(np.diff(d['mean']) > 0)                            # monotone in pixels
    assert np.abs(d['mean'] - truth(np.linspace(1, 8, NR), 0)).max() < 0.06
    assert 0 < p['rig']['pedestrian']['std'] < 0.1
    # saved, reloaded: the same predictions; the reason travels
    path = PM.save_model(m)
    back = PM.load_model()
    assert path == PM.default_model_path() and back.trained
    assert np.allclose(back.predict(f)['classes']['pedestrian']['mean'], d['mean'])
    # explanations: more pixels raised the objective; the body's share lowered it
    ref = synth_feats(np.full(NR, 3.0), np.zeros(NR))
    ex = m.explain(synth_feats(np.full(NR, 5.0), np.zeros(NR)), ref)
    assert ex[0]['feature'] == 'log2_px_obj' and ex[0]['delta'] > 0
    dm = m.objective(synth_feats(np.full(NR, 5.0), np.zeros(NR)))[0] - m.objective(ref)[0]
    assert abs(sum(e['delta'] for e in ex) - dm) < 0.2 * abs(dm)
    ex = m.explain(synth_feats(np.full(NR, 5.0), np.full(NR, 0.8)), synth_feats(np.full(NR, 5.0), np.zeros(NR)))
    assert ex[0]['feature'] == 'body_share' and ex[0]['delta'] < 0 and 'lowered' in ex[0]['text']
    # the files of the study
    res = PM.PerfModelResult(st, m, '', PM.store_summary(docs), 'v', f, p)
    names = [n for n, _ in PM.files(res)]
    assert names[:3] == ['perf_model.json', 'perf_model_calibration.csv', 'perf_model_predictions.csv']
    doc = json.loads(dict(PM.files(res))['perf_model.json'])
    assert doc['schema'] == PM.SCHEMA and doc['calibration'] and doc['regions'][0]['pedestrian']['recall'] is not None


# ======================================================================= towns: the honest check (round 15)
def town_set(towns=('A', 'B', 'C', 'D'), seeds=(1, 2), rigs=4, seed=3, offset=0.6):
    """Records over towns, each with scene sets per seed; each town shifts the logit by its own offset (a town is
    harder or easier as a whole), so a town the model has not seen is a real extrapolation."""
    rng = np.random.default_rng(seed)
    off = {t: rng.normal(0, offset) for t in towns}
    docs, feats = [], {}
    i = 0
    for t in towns:
        for sd in seeds:
            for _ in range(rigs):
                lpx = rng.uniform(1.0, 8.0, NR)
                occ = rng.uniform(0.0, 1.0, NR)
                ns = rng.integers(20, 200, NR)
                k = rng.binomial(ns, PM.sigmoid(PM.logit(truth(lpx, occ)) + off[t]))
                m = {'pedestrian': {rid: PM.metrics_cell(int(nn), int(kk), 0)
                                    for (_, _, rid, _), nn, kk in zip(GRID1.regions(), ns, k)}}
                d = PM.make_record(source='synthetic-test', scene={'kind': 'synthetic', 'town': t, 'seed': sd},
                                   detector={'id': 'analytic', 'backend': 'external:test'}, rig=rig_of(i),
                                   grid=GRID1, metrics=m)
                docs.append(d)
                feats[d['rig_hash']] = synth_feats(lpx, occ)
                i += 1
    return docs, feats


def test_leave_one_town_out_and_the_learning_curve_over_towns_and_records():
    docs, feats = town_set()
    st = PM.PerfModelSettings(classes=['pedestrian'], restarts=1, objective={'pedestrian': 1.0})
    m = PM.Model(st).fit(docs, feats_of=lambda d: feats[d['rig_hash']])
    assert m.trained and m.n_scenes == 8
    cv = m.cv[('pedestrian', 'recall')]
    assert set(cv) == set(PM.CV_KINDS)
    t = cv['town']
    assert t['ok'] and t['folds'] == 4 and t['groups'] == 4 and t['gp']['n'] == 32 * NR
    assert sorted(t['per_group']) == ['A', 'B', 'C', 'D']                 # one fold per town, every scene set in it
    assert all(g['gp']['n'] == 8 * NR for g in t['per_group'].values())
    assert cv['scene']['folds'] == 8
    for by in PM.CV_KINDS:
        g = cv[by]['gp']
        assert 0.0 < g['rmse_floor'] < g['rmse'] and g['n_big'] > 0 and g['rmse_big'] is not None
    assert t['gp']['rmse'] < t['logistic']['rmse'] < t['constant']['rmse'] and t['gp']['nlpd'] < t['constant']['nlpd']
    assert 0.85 <= t['gp']['coverage95'] <= 1.0                         # a new town's offset widens the intervals
    cm = m.models[('pedestrian', 'recall')]
    assert cm.levels == list(PM.GROUP_LEVELS) and cm.gv['town'] > 0.1     # the towns' spread is found
    # round 14's model (no town offsets): a held-out scene set gains nothing from its town's other scene sets, and a
    # new town's intervals are too narrow
    st14 = PM.PerfModelSettings(classes=['pedestrian'], restarts=1, objective={'pedestrian': 1.0}, town_effects=False)
    m14 = PM.Model(st14).fit(docs, feats_of=lambda d: feats[d['rig_hash']])
    cv14 = m14.cv[('pedestrian', 'recall')]
    assert m14.models[('pedestrian', 'recall')].levels == list(PM.SCENE_LEVELS)
    assert cv['scene']['gp']['rmse'] < 0.5 * cv14['scene']['gp']['rmse']
    assert cv14['town']['gp']['coverage95'] < t['gp']['coverage95']
    rows = PM.cv_table(m)
    assert {r['cv'] for r in rows} == set(PM.CV_KINDS) and all('rmse_floor' in r for r in rows if r['method'])
    assert 'rmse_floor' in PM.to_csv(PM.PerfModelResult(st, m, '', {}, 'v')).splitlines()[0]
    assert len(PM.calibration_png(m)) > 1000
    back = PM.Model.from_dict(json.loads(json.dumps(m.to_dict())))
    assert set(back.cv[('pedestrian', 'recall')]) == set(PM.CV_KINDS)
    # the learning curve: 1, 2, 3 training towns (distinct draws), and fractions of the records at 3 towns
    rows = PM.training_rows(docs, lambda d: feats[d['rig_hash']], 'pedestrian')
    assert sorted(set(rows['town'].tolist())) == ['A', 'B', 'C', 'D']
    lc = PM.learning_curve(rows, 'pedestrian', st=st, repeats=3, record_fracs=(0.25, 1.0), seed=1)
    assert lc['ok'] and [r['k'] for r in lc['by_towns']] == [1, 2, 3]
    by_k = {r['k']: r for r in lc['by_towns']}
    assert by_k[1]['splits'] == 3 and by_k[3]['splits'] == 3            # C(4, 3) = 4 sets, 3 drawn
    assert by_k[1]['train_records'] == 8 and by_k[3]['train_records'] == 24
    assert by_k[1]['gp']['n'] == 3 * 24 * NR                            # each set scored on the 3 other towns
    assert by_k[3]['gp']['rmse'] < by_k[1]['gp']['rmse']                # more towns, less error on new towns
    fr = {r['frac']: r for r in lc['by_records']}
    assert fr[0.25]['train_records'] == 6 and fr[1.0]['train_records'] == 24 and fr[1.0]['splits'] == 4
    assert fr[1.0]['gp']['rmse'] <= fr[0.25]['gp']['rmse'] + 0.01
    assert PM.learning_curve({**rows, 'town': np.full(len(rows['town']), 'A')}, 'pedestrian', st=st)['ok'] is False


def test_a_campaign_drive_finds_its_rig_file(tmp_path):
    """core.simcampaign keeps a campaign's rig files beside its drives (rigs/<id>/sim_rig.json); a drive's own
    rig/ folder wins; nothing outside the campaign is read (a rig id with a path in it)."""
    import json as _j
    from vantage.core import rig_perception as RP
    drive = tmp_path / 'camp' / 'drives' / 'd0001'
    drive.mkdir(parents=True)
    rigf = tmp_path / 'camp' / 'rigs' / 'bank--clearnoon' / 'sim_rig.json'
    rigf.parent.mkdir(parents=True)
    rigf.write_text(_j.dumps({'variant': 'bank'}))
    assert RP.rig_file(str(drive)) is None
    (drive / 'campaign_drive.json').write_text(_j.dumps({'drive': {'rig': 'bank--clearnoon'}}))
    assert RP.rig_file(str(drive)) == str(rigf)
    (drive / 'campaign_drive.json').write_text(_j.dumps({'drive': {'rig': '../../rigs/bank--clearnoon'}}))
    assert RP.rig_file(str(drive)) is None
    own = drive / 'rig' / 'sim_rig.json'
    own.parent.mkdir()
    own.write_text('{}')
    assert RP.rig_file(str(drive)) == str(own)


def test_the_campaign_scene_set_and_its_towns():
    from vantage.core import perception_scenes as PS, samples
    p = samples.tigor_sample_project()
    v = next(v for v in p.variants if v.sensors)
    st = PS.SceneSetSettings.from_dict({'set_id': 'campaign', 'variants': [v.uid], 'seeds': [1, 2], 'town': 'Town05',
                                        'weather': 'WetNoon'})
    pl = PS.plan(p, st)
    drives = [d for d in pl.drives if d['role'] == 'drive']
    assert [d['scenario']['name'] for d in drives] == ['free_drive', 'pedestrian_crossing'] * 2
    assert [d['seed'] for d in drives] == [1, 1, 2, 2] and {d['weather'] for d in drives} == {'WetNoon'}
    cs = PS.campaign_settings(p, st)
    assert cs.base['town'] == 'Town05' and cs.base['walkers'] == 60 and cs.base['traffic'] == 40
    assert set(PS.TOWNS) == {'Town01', 'Town02', 'Town03', 'Town04', 'Town05', 'Town10HD'}


# ======================================================================= active learning
def _toy_model(train_lpx, seed=7):
    """A 1-D toy: one region, recall = truth(log2 px, 0); trained on rigs at train_lpx."""
    g = RG.RegionGrid(sectors=1, range_edges=(1000.0,))
    rng = np.random.default_rng(seed)
    docs, feats = [], {}
    for i, l in enumerate(train_lpx):
        lp = np.array([l, 99.0])                                         # the open bin is never observed
        ns = np.array([80, 0])
        k = rng.binomial(ns, truth(lp, 0.0))
        m = {'pedestrian': {'S0R0': PM.metrics_cell(int(ns[0]), int(k[0]), 0)}}
        d = PM.make_record(source='synthetic-test', scene={'kind': 'synthetic', 'town': f'T{i % 2}', 'seed': 1},
                           detector={'id': 'analytic', 'backend': 'external:test'}, rig=rig_of(i), grid=g, metrics=m)
        docs.append(d)
        feats[d['rig_hash']] = synth_feats(lp, np.zeros(2), grid=g)
    st = PM.PerfModelSettings(classes=['pedestrian'], objective={'pedestrian': 1.0}, max_range_m=500.0, cv=False,
                              min_records=4, min_rows=4, restarts=2)
    return PM.Model(st).fit(docs, feats_of=lambda d: feats[d['rig_hash']]), g


def test_propose_prefers_uncertain_and_improving_candidates_on_a_toy():
    # trained where recall is low (log2 px 0.5–2); 'far' extrapolates into the steep middle of the curve, 'low' below
    # the data where recall is near 0 either way (its latent std is large, its recall's is not)
    m, g = _toy_model([0.5, 0.8, 1.1, 1.4, 1.7, 2.0])
    assert m.trained
    cand = {'near': 1.2, 'edge': 2.1, 'far': 4.0, 'low': -1.0}
    C = [(name, synth_feats(np.array([v, 99.0]), np.zeros(2), grid=g)) for name, v in cand.items()]
    stats = {name: m.objective(f, epistemic=True) for name, f in C}
    # the known answer: far from the data the std is largest; the monotone prior makes 'far' the most promising too
    cm = m.models[('pedestrian', 'recall')]
    lat = {n: float(cm.predict(f['X']['pedestrian'][:1])[1][0]) for n, f in C}
    assert lat['far'] > 3 * lat['near'] and lat['low'] > lat['near']     # the prior's slope is uncertain too
    assert max(stats, key=lambda n: stats[n][1]) == 'far'
    p = PM.propose(C, 1, model=m, mode='std')
    assert p[0]['name'] == 'far' and 'least certain' in p[0]['why']
    p = PM.propose(C, 4, model=m, mode='ei')
    assert p[0]['name'] == 'far'
    best = PM.incumbent(m)
    want = PM.expected_improvement([stats[n][0] for n in cand], [stats[n][1] for n in cand], best)
    assert abs(p[0]['ei'] - max(want)) < 1e-9
    assert {x['name'] for x in p} == set(cand)
    # the kriging believer: a second copy of the chosen rig loses its uncertainty once the first is picked
    C2 = C + [('far again', synth_feats(np.array([4.0, 99.0]), np.zeros(2), grid=g))]
    p = PM.propose(C2, 2, model=m, mode='std')
    assert p[0]['name'] in ('far', 'far again') and p[1]['name'] not in ('far', 'far again')


def dip(lp):
    """A truth the logistic prior cannot foresee: recall dips by half around 6.5 px (log2)."""
    return truth(lp, 0.0) * (1.0 - 0.5 * np.exp(-0.5 * ((np.asarray(lp, float) - 6.5) / 0.8) ** 2))


def test_campaign_with_a_fake_evaluator_improves_the_model(tmp_path):
    g = RG.RegionGrid(sectors=1, range_edges=(1000.0,))
    pool = np.linspace(0.5, 9.0, 18)
    rng = np.random.default_rng(9)
    feats = {}

    def make(i, l):
        lp = np.array([l, 99.0])
        ns = np.array([100, 0])
        k = rng.binomial(ns, dip(lp))
        m = {'pedestrian': {'S0R0': PM.metrics_cell(int(ns[0]), int(k[0]), 0)}}
        d = PM.make_record(source='synthetic-test', scene={'kind': 'synthetic', 'town': f'T{i % 2}', 'seed': 1},
                           detector={'id': 'analytic', 'backend': 'external:test'}, rig=rig_of(100 + i), grid=g,
                           metrics=m)
        feats[d['rig_hash']] = synth_feats(lp, np.zeros(2), grid=g)
        return d
    store = PM.Store(tmp_path / 'c.jsonl')
    for i, l in enumerate((1.0, 1.4, 1.8, 2.2)):                    # the start: only low-resolution rigs
        store.add(make(i, l))
    cands = [PM.Candidate(f'rig {l:.2f}', synth_feats(np.array([l, 99.0]), np.zeros(2), grid=g)) for l in pool]
    calls = []

    def evaluator(c, progress=None, cancel=None):
        calls.append(c.name)
        return [make(10 + len(calls), float(c.features['X']['pedestrian'][0, PX]))]
    st = PM.PerfModelSettings(classes=['pedestrian'], objective={'pedestrian': 1.0}, max_range_m=500.0, cv=False,
                              min_records=4, min_rows=4, restarts=2, mode='std')
    orig = PM.FeatureCache.get
    PM.FeatureCache.get = lambda self, d, *a, **k: feats[d['rig_hash']]
    try:
        def err(model):
            mu, var = model.models[('pedestrian', 'recall')].predict(np.vstack([c.features['X']['pedestrian'][:1]
                                                                                 for c in cands]))
            return float(np.sqrt(np.mean((PM.logistic_moments(mu, var)[0] - dip(pool)) ** 2)))
        before = PM.Model(st).fit(store.records())
        out = PM.campaign(None, cands, rounds=4, k=3, evaluator=evaluator, store=store, settings=st, save=True)
    finally:
        PM.FeatureCache.get = orig
    assert len(calls) == 12 and len(set(calls)) == 12
    assert len(store.records()) == 16
    after = out['model']
    assert err(after) < 0.5 * err(before), (err(before), err(after))
    pts = [c for c in out['curve'] if c.get('prospective')]
    assert len(pts) == 4 and all(c['prospective']['n'] == 3 for c in pts)
    assert out['curve'][-1]['final'] and out['curve'][-1]['records'] == 16
    assert PM.load_model() is not None


# ======================================================================= the interim evaluator
def test_region_eval_counts_rigs_of_a_recording(tmp_path):
    from test_detection import ColourDetector, make_recording, scene_a
    base = scene_a()

    def fn(cam, k):
        objs = base(cam, k)
        if cam == 'G':                                   # G misses car A and sees car B
            for o in objs:
                if o['id'] == 1:
                    o['colour'] = None
                if o['id'] == 3:
                    o['colour'] = (255, 0, 0)
            objs = [o for o in objs if o['id'] != 0]
        return objs
    root = make_recording(str(tmp_path / 'drive'), fn, cams=('F', 'G'))
    os.makedirs(os.path.join(root, 'rig'), exist_ok=True)
    lens = {'model': 'lens', 'projection': 'kannala_brandt', 'f_px': 100.0, 'kb': [0.01, -0.002, 0, 0], 'cx': 160,
            'cy': 120, 'aspect': 1.0, 'max_theta_deg': 90.0, 'sensor_size': [320, 240], 'image_size': [320, 240]}
    with open(os.path.join(root, 'rig', 'sim_rig.json'), 'w') as fh:
        json.dump({'cameras': [{'tag': t, 'pose_vehicle': {'x': 2.0, 'y': 0.0, 'z': 1.2, 'yaw': 0, 'pitch': 0,
                                                           'roll': 0}, 'lens': lens} for t in ('F', 'G')]}, fh)
    grid = RG.RegionGrid(sectors=8)
    res = PM.region_eval(root, settings={'cache': False}, detector=ColourDetector(), grid=grid)
    car_f = PM.rig_metrics(res, ['F'])['car']
    car_g = PM.rig_metrics(res, ['G'])['car']
    car_fg = PM.rig_metrics(res, ['F', 'G'])['car']
    # car A at 8 m (S0R0): F finds it in all 20 frames, G never; car B at 25 m (S0R2): only G; truck at 45 m (S0R3)
    assert (car_f['S0R0']['n_gt'], car_f['S0R0']['tp']) == (20, 20)
    assert (car_g['S0R0']['n_gt'], car_g['S0R0']['tp']) == (20, 0)
    assert (car_f['S0R2']['tp'], car_g['S0R2']['tp'], car_fg['S0R2']['tp']) == (0, 20, 20)
    assert car_fg['S0R0']['recall'] == 1.0 and car_fg['S0R0']['n_tracks'] == 1
    tr = PM.rig_metrics(res, ['F'])['truck']['S0R3']
    assert (tr['n_gt'], tr['tp']) == (20, 10)
    ped = PM.rig_metrics(res, ['F'])['pedestrian']
    assert sum(c['n_gt'] for c in ped.values()) == 20                 # the 6-px pedestrian is an ignore region
    # F's decoy car (frames 0-4) is a false detection placed by its range; G has none
    assert sum(c['fp'] for c in PM.rig_metrics(res, ['F'])['car'].values()) == 5
    assert sum(c['fp'] for c in car_g.values()) == 0
    assert car_f['S0R0']['ap'] == pytest.approx(1.0)
    docs = PM.library_records(res, {'F only': ['F'], 'both': ['F', 'G']}, cam_meta={})
    assert len(docs) == 2 and all(PM.validate_record(d) == [] for d in docs)
    assert docs[0]['detector']['evaluator'] == PM.INTERIM and docs[0]['rig']['cameras'][0]['width'] == 320
    assert docs[0]['scene']['scene_hash'] == docs[1]['scene']['scene_hash']
    assert docs[0]['rig_hash'] != docs[1]['rig_hash']


# ======================================================================= the page
from test_gui import app, win, pump, settle, _wait   # noqa: E402,F401 (fixtures)


def _open(win, name, cls):
    if name not in win._page_widgets:
        pg = cls(win.ctx)
        win.pages.addWidget(pg)
        win._page_widgets[name] = pg
    win.set_page(name)
    return win._page_widgets[name]


def test_page_store_train_predict_propose_campaign_and_undo(app, win, tmp_path):
    from vantage.ui.workspaces.perf_model_ws import PerfModelPage
    p = win.ctx.store.project
    grid = RG.default_grid(p)
    # real-geometry records: the sample's variants as rigs, recall drawn from the analytic truth of their features
    rng = np.random.default_rng(3)
    docs = []
    for i, v in enumerate(p.variants * 2):
        f = PM.describe_rig(p, v, grid, classes=('pedestrian',))
        X = f['X']['pedestrian']
        ns = rng.integers(10, 120, len(f['regions']))
        k = rng.binomial(ns, truth(X[:, PX], X[:, OCC]))
        m = {'pedestrian': {rid: PM.metrics_cell(int(a), int(b), 0) for rid, a, b in zip(f['regions'], ns, k)}}
        docs.append(PM.make_record(source='synthetic-test', scene={'kind': 'synthetic', 'town': f'T{i % 3}',
                                                                   'seed': i}, detector={'id': 'analytic',
                                                                                         'backend': 'external:test'},
                                   rig=PM.rig_block(p, v), grid=grid, metrics=m))
    src = tmp_path / 'in.jsonl'
    PM.export_records(src, anonymise=False, docs=docs)
    pg = _open(win, 'perf_model', PerfModelPage)
    pg.activate()
    assert 'No perception records' in pg.store_text.text()
    r = pg.import_file(str(src))
    assert r['added'] == 10 and pg.store_table.rowCount() == 10
    assert '10 records' in pg.store_text.text()
    # settings are undoable edits of the project
    pg.w_ped.setValue(2.0)
    pg.w_ped.committed.emit(2.0)
    assert PM.settings_of(p).objective['pedestrian'] == 2.0
    win.ctx.store.undo.undo()
    assert PM.settings_of(win.ctx.store.project).objective.get('pedestrian') == 1.0
    pg._set('test', objective={'pedestrian': 1.0}, classes=['pedestrian'], restarts=1, max_iter=60)
    assert pg.train()
    _wait(app, lambda: not pg.busy, 300)
    assert pg.model is not None and pg.model.trained, pg.status_text
    assert pg.result.prediction['trained'] and pg.map_mean.values
    assert pg.cv_table.rowCount() > 0 and pg.cal_scene.curves
    assert pg.t_obj.value != '—'
    # hover a region of the map: its name and interval in the tooltip
    from PyQt6.QtCore import QPointF as P
    from PyQt6.QtGui import QMouseEvent
    from PyQt6.QtCore import QEvent, Qt as Q
    from PyQt6.QtWidgets import QApplication
    pg.tabs.setCurrentIndex(0)
    pump(app, 0.2)
    mp = pg.map_mean
    mp.repaint()
    path, rid = mp._cells[1]
    c = path.boundingRect().center()
    QApplication.sendEvent(mp, QMouseEvent(QEvent.Type.MouseMove, P(c), P(c), Q.MouseButton.NoButton,
                                           Q.MouseButton.NoButton, Q.KeyboardModifier.NoModifier))
    assert mp.hover is not None and '·' in mp.tip_text()
    # another variant: the prediction follows, and what moved it is said in words against the reference
    last = p.variants[-1].uid
    win.ctx.store.edit('variants', lambda pj: setattr(pj, 'active_uid', last), 'Switch variant')
    pg.predict(sync=True)
    pg.refresh()
    assert pg.result.variant == win.ctx.store.project.variant.name
    if pg.result.reference:
        assert pg.why.text().startswith('Against')
    # active learning
    picks = pg.propose()
    assert picks and pg.al_table.rowCount() == len(picks)

    def fake_eval(c, progress=None, cancel=None):
        f = PM.describe_rig(p, c.variant, grid, classes=('pedestrian',))
        X = f['X']['pedestrian']
        ns = np.full(len(f['regions']), 50)
        k = rng.binomial(ns, truth(X[:, PX], X[:, OCC]))
        m = {'pedestrian': {rid: PM.metrics_cell(int(a), int(b), 0) for rid, a, b in zip(f['regions'], ns, k)}}
        return [PM.make_record(source='synthetic-test', scene={'kind': 'synthetic', 'town': 'T9', 'seed': 9},
                               detector={'id': 'analytic', 'backend': 'external:test'},
                               rig=PM.rig_block(p, c.variant), grid=grid, metrics=m)]
    pg.evaluator = fake_eval
    assert pg.run_campaign(rounds=1, k=2)
    _wait(app, lambda: not pg.busy, 300)
    assert pg.campaign_result is not None and 'Learning curve' in pg.al_text.text()
    assert pg.store_table.rowCount() == 12
    # export the records (anonymised) and the report
    out = tmp_path / 'out.jsonl'
    assert pg.export_records(str(out), anonymise=True) == 12
    assert p.vehicle.name not in out.read_text()
    paths = pg.export(str(tmp_path / 'report'))
    assert any(x.endswith('perf_model.json') for x in paths)
    # the store with the project: an undoable switch
    pg.in_project.setChecked(True)
    assert PM.settings_of(win.ctx.store.project).store_in_project
    win.ctx.store.undo.undo()
    assert not PM.settings_of(win.ctx.store.project).store_in_project
