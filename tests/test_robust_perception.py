"""Robust perception (core.robust_perception): the rotation remap is exact, the BEV error of a pitch error matches the
closed form, zero tolerance gives zero spread, the Sobol estimators recover known indices, the error model's modes,
the perception record, and the page driven like a user. Synthetic recordings are rendered here (two cameras, a
pedestrian and a car as boxes on a flat ground, the CARLA bridge's ground-truth layout); a colour detector stands in
for YOLOX. The real-data check (test_real_*) runs on the CARLA drives when they and the model are present."""
import json
import math
import os

import cv2
import numpy as np
import pytest

from vantage.core import detection as DT
from vantage.core import detection_eval as DE
from vantage.core import groundtruth as GT
from vantage.core import robust_perception as RP
from vantage.core.mathutil import quat_from_matrix, rot_zyx

B2O = RP.BODY_TO_OPTICAL
W, H, F = 192, 144, 70.0
CAMS = (('F', (3.0, 0.0, 1.5), (0.0, 10.0, 0.0)), ('L', (1.6, 0.9, 1.5), (35.0, 10.0, 0.0)))
OBJECTS = ((1, 'pedestrian', (13.0, -0.6, 0.85), (0.5, 0.5, 1.7), 0.0, (0, 0, 255)),
           (2, 'car', (21.0, 2.5, 0.75), (4.5, 1.8, 1.5), 0.0, (255, 0, 0)))


def _cam(tag, pos, ypr, D=(0.0, 0.0, 0.0, 0.0), w=W, h=H, f=F):
    c = DE.Camera(tag, f'CAM_{tag}', w, h, [])
    c.K = np.array([[f, 0, (w - 1) / 2], [0, f, (h - 1) / 2], [0, 0, 1.0]])
    c.D = np.asarray(D, float)
    c.lens = 'kb'
    c.R = rot_zyx(*ypr) @ B2O.T
    c.t = np.asarray(pos, float)
    return c


def make_recording(root, frames=3, move=0.6):
    """A drive of `frames` instants: the ego stands, the pedestrian walks left, the car drives away."""
    os.makedirs(os.path.join(root, 'nuscenes', 'v1.0-vantage'), exist_ok=True)
    sens, cal, cams_doc = [], [], {}
    vv, uu = np.mgrid[0:H, 0:W]
    for ci, (tag, pos, ypr) in enumerate(CAMS):
        c = _cam(tag, pos, ypr)
        d, ok = DE.kb_unproject(c, uu.ravel().astype(float), vv.ravel().astype(float))
        dv = d @ c.R.T
        q = quat_from_matrix(c.R)
        sens.append({'token': f's{ci}', 'channel': c.channel, 'modality': 'camera'})
        cal.append({'token': f'c{ci}', 'sensor_token': f's{ci}', 'translation': list(pos),
                    'rotation': [q[3], q[0], q[1], q[2]], 'camera_intrinsic': c.K.tolist(),
                    'vantage_distortion': {'model': 'kannala_brandt', 'coefficients': [0, 0, 0, 0]}})
        frs = []
        os.makedirs(os.path.join(root, 'ground_truth', c.channel), exist_ok=True)
        for k in range(frames):
            t_us = 1_000_000 + 100_000 * k
            o = c.t
            with np.errstate(divide='ignore', invalid='ignore'):
                tg = np.where(dv[:, 2] < -1e-9, -o[2] / dv[:, 2], np.inf)
            rng = np.where(ok, tg, np.inf)
            inst = np.zeros(len(rng), np.uint16)
            colour = np.zeros((len(rng), 3), np.uint8)
            P = o[None, :] + np.where(np.isfinite(rng), rng, 0)[:, None] * dv
            chk = ((np.floor(P[:, 0]) + np.floor(P[:, 1])) % 2).astype(int)
            colour[:] = np.where(chk[:, None] == 1, 110, 150)
            colour[~np.isfinite(rng)] = (200, 180, 160)
            objs = []
            for oid, cls, cen, size, yaw, col in OBJECTS:
                cen = np.array(cen, float) + (np.array([0, move * k, 0]) if cls == 'pedestrian' else
                                              np.array([move * k, 0, 0]))
                R = GT.box_matrix(yaw)
                tin, _ = GT.ray_box(o, dv, cen, size, R)
                hit = ok & (tin < rng)
                rng = np.where(hit, tin, rng)
                inst[hit] = oid
                colour[hit] = col
                objs.append((oid, cls, cen, size, yaw))
            inst_img = inst.reshape(H, W)
            ids, boxes, counts = DE.boxes_of_ids(inst_img)
            name = f'{c.channel}__{t_us}'
            img_rel = f'ground_truth/{c.channel}/{name}.png'
            cv2.imwrite(os.path.join(root, img_rel), colour.reshape(H, W, 3))
            cv2.imwrite(os.path.join(root, f'ground_truth/{c.channel}/{name}.instance.png'), inst_img)
            r_img = np.where(np.isfinite(rng), rng, 0).reshape(H, W)
            cv2.imwrite(os.path.join(root, f'ground_truth/{c.channel}/{name}.range.png'), GT.range_png(r_img))
            ob_doc = []
            for oid, cls, cen, size, yaw in objs:
                if oid not in ids:
                    continue
                j = list(ids).index(oid)
                b = boxes[j]
                edge = b[0] <= 0 or b[1] <= 0 or b[2] >= W or b[3] >= H
                ob_doc.append({'id': oid, 'actor_id': 100 + oid, 'instance_token': f'tok{oid}', 'class': cls,
                               'category': 'human.pedestrian.adult' if cls == 'pedestrian' else 'vehicle.car',
                               'bbox': [int(v) for v in b], 'pixels': int(counts[j]),
                               'truncated': 0.3 if edge else 0.0, 'occluded': 0.0,
                               'range_m': float(np.linalg.norm(cen - o)),
                               'box3d': {'center': [float(v) for v in cen], 'size': list(size), 'yaw_deg': yaw}})
            frs.append({'sample_data_token': f'{tag}{k}', 'timestamp': t_us, 'image': img_rel,
                        'instance': f'ground_truth/{c.channel}/{name}.instance.png',
                        'range': f'ground_truth/{c.channel}/{name}.range.png', 'objects': ob_doc})
        cams_doc[tag] = {'channel': c.channel, 'width': W, 'height': H, 'frames': frs}
    with open(os.path.join(root, 'nuscenes', 'v1.0-vantage', 'sensor.json'), 'w') as fh:
        json.dump(sens, fh)
    with open(os.path.join(root, 'nuscenes', 'v1.0-vantage', 'calibrated_sensor.json'), 'w') as fh:
        json.dump(cal, fh)
    with open(os.path.join(root, 'ground_truth', 'index.json'), 'w') as fh:
        json.dump({'schema': GT.SCHEMA, 'cameras': cams_doc, 'lidars': {}}, fh)
    return root


def colour_detector(img):
    """Pure red blobs are people, pure blue ones cars (the synthetic renders' colours)."""
    boxes, scores, coco, cls = [], [], [], []
    for col, cname, ci in (((0, 0, 255), 'person', 'pedestrian'), ((255, 0, 0), 'car', 'car')):
        m = np.all(np.abs(img.astype(int) - np.array(col)) < 40, axis=2).astype(np.uint8)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
        for k in range(1, n):
            x, y, w, h, a = stats[k]
            if a < 4:
                continue
            boxes.append([x, y, x + w, y + h])
            scores.append(0.9)
            coco.append(DT.COCO_CLASSES.index(cname))
            cls.append(DT.CLASSES.index(ci))
    return {'boxes': np.asarray(boxes, float).reshape(-1, 4), 'scores': np.asarray(scores, float),
            'coco': np.asarray(coco, int), 'cls': np.asarray(cls, int), 'ms': 0.0}


def zero_settings(rec, **kw):
    d = dict(recordings=[rec], mount_from='custom', mount_deg=0.0, mount_mm=0.0, thermal_from='custom',
             thermal_deg=0.0, thermal_mm=0.0, vib_from='custom', vib_body_deg=0.0, vib_heave_mm=0.0, vib_flex_deg=0.0,
             vib_flex_mm=0.0, calib_from='custom', calib_deg=0.0, calib_mm=0.0, samples=3, sobol_n=8, oat_samples=2,
             frame_step=1, bev_frame_step=1, boot=20)
    d.update(kw)
    return RP.RobustSettings.from_dict(d)


@pytest.fixture(scope='module')
def rec(tmp_path_factory):
    return make_recording(str(tmp_path_factory.mktemp('robust_rec')))


# =========================================================================== the image remap
def test_rotation_remap_round_trip_exact():
    c = _cam('F', (0, 0, 1), (0, 20, 0), D=(0.01, -0.005, 0.002, -0.0005), w=320, h=240, f=110)
    R = rot_zyx(2.0, -1.5, 1.0)                         # a small turn of the camera (in its own frame)
    rays = RP.grid_rays(c)
    v1, miss1 = RP.rotation_view(c, R, rays)
    v2, miss2 = RP.rotation_view(c, R.T, rays)
    # composing the maps: true pixel p -> recorded pixel q (v2) -> back (v1) must land on p
    mx = cv2.remap(v1.map_x, v2.map_x, v2.map_y, cv2.INTER_LINEAR, borderValue=-10)
    my = cv2.remap(v1.map_y, v2.map_x, v2.map_y, cv2.INTER_LINEAR, borderValue=-10)
    gy, gx = np.mgrid[0:c.height, 0:c.width]
    both = ~miss2
    # every pixel of the bilinear neighbourhood in the first map valid (its fill must not be sampled)
    valid1 = cv2.erode((~miss1).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(np.float32)
    both &= cv2.remap(valid1, v2.map_x, v2.map_y, cv2.INTER_NEAREST) > 0.5
    # keep pixels away from the field's edge (the bilinear sampling of the map mixes in the -10 fill there)
    both &= cv2.erode(both.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    assert both.mean() > 0.6
    err = np.hypot(mx - gx, my - gy)[both]
    assert err.max() < 0.05 and np.median(err) < 0.02      # bilinear sampling of a smooth map
    # a smooth image remapped by R then by R^T returns inside the field
    img = (127 + 100 * np.sin(gx / 25.0) * np.cos(gy / 31.0)).astype(np.float32)   # smooth: two bilinear passes
    back = RP.remap_image(RP.remap_image(img, v1), v2)     # img(map1(map2(p))), the composition above
    assert np.abs(back - img)[both].max() < 1.0
    # identity: no missing pixels inside the lens's field, the map is the pixel grid
    v0, miss0 = RP.rotation_view(c, np.eye(3), rays)
    inside = rays[1]
    assert not miss0[inside & (np.hypot(gx - c.K[0, 2], gy - c.K[1, 2]) < 0.95 * min(c.width, c.height) / 2)].any()
    ok = ~miss0
    assert np.abs(v0.map_x[ok] - gx[ok]).max() < 1e-3 and np.abs(v0.map_y[ok] - gy[ok]).max() < 1e-3


def test_rotation_marks_missing_never_invents():
    c = _cam('F', (0, 0, 1), (0, 0, 0), w=200, h=150, f=60)
    R = rot_zyx(0.0, 0.0, 0.0)
    rays = RP.grid_rays(c)
    _, miss0 = RP.rotation_view(c, R, rays)
    # a 25° turn about the optical y axis (yaw, in the optical frame) exposes one side of the image
    a = math.radians(25)
    Ry = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
    view, miss = RP.rotation_view(c, Ry, rays)
    assert miss.mean() > miss0.mean() + 0.05
    img = np.full((150, 200, 3), 200, np.uint8)
    out = RP.remap_image(img, view)
    assert (out[miss] == 0).all()                       # missing content is black, never filled
    assert (out[~miss] == 200).all()


def test_translation_bound():
    c = _cam('F', (0, 0, 1), (0, 0, 0), w=200, h=150, f=60)
    lim = DE._theta_limit(c)
    assert RP.max_px_per_rad(c) == pytest.approx(60 * max(1.0, lim / math.sin(lim)), rel=2e-3)


# =========================================================================== the BEV error
@pytest.mark.parametrize('h, D, dp', [(1.5, 40.0, 0.5), (1.5, 20.0, 0.25), (2.0, 30.0, -0.3), (0.6, 10.0, 1.0)])
def test_bev_error_matches_pitch_formula(h, D, dp):
    c = _cam('F', (0.0, 0.0, h), (0.0, 3.0, 0.0), w=640, h=480, f=300)
    z = np.zeros(3)
    true, used = RP.build_poses(c, RP.design_euler(c), (np.array([0.0, dp, 0.0]), z), (z, z), (0.0, 0.0), (0, 0))
    X = np.array([[D, 0.0, 0.0]])
    q = (X - true.t) @ true.R
    u, v, ok = DE.kb_project(c, q)
    assert ok[0]
    G, okl = RP.lift(c, used, np.column_stack([u, v]))
    expect = RP.pitch_error_analytic(h, D, dp)
    assert okl[0]
    assert G[0, 0] - D == pytest.approx(expect, rel=1e-5, abs=1e-6)
    assert G[0, 1] == pytest.approx(0.0, abs=1e-6)
    # the closed form itself: 0.5° at 1.5 m and 40 m moves the point by metres
    assert RP.pitch_error_analytic(1.5, 40.0, 0.5) > 10.0


def test_body_pitch_rotates_and_moves_about_cg():
    c = _cam('F', (3.0, 0.0, 0.6), (0.0, 40.0, 0.0))
    z = np.zeros(3)
    true, _ = RP.build_poses(c, RP.design_euler(c), (z, z), (z, z), (1.0, 0.0), (1.5, 0.5))
    # nose down by 1°: a camera 1.5 m ahead of the centre of gravity drops by 1.5 sin 1°
    a = math.radians(1.0)
    assert true.t[2] == pytest.approx(0.5 - 1.5 * math.sin(a) + 0.1 * math.cos(a), abs=1e-9)
    assert true.t[0] == pytest.approx(1.5 + 1.5 * math.cos(a) + 0.1 * math.sin(a), abs=1e-9)
    yaw, pitch, roll = RP.design_euler(RP.DE.Camera('x', 'x', 1, 1, [], R=true.R))
    assert pitch == pytest.approx(41.0, abs=1e-6)


# =========================================================================== Sobol
def test_sobol_ishigami():
    S1, S2, S3, T1, T2, T3 = RP.ishigami_indices()
    assert (S1, S2) == pytest.approx((0.3139, 0.4424), abs=5e-4)
    so = RP.sobol_indices(lambda U: RP.ishigami(2 * np.pi * U - np.pi), 3, {'x1': [0], 'x2': [1], 'x3': [2]},
                          n=40000, seed=3, boot=0)
    g = so['groups']
    assert g['x1']['S'][0] == pytest.approx(S1, abs=0.03)
    assert g['x2']['S'][0] == pytest.approx(S2, abs=0.03)
    assert g['x3']['S'][0] == pytest.approx(0.0, abs=0.03)
    assert g['x3']['ST'][0] == pytest.approx(T3, abs=0.03)
    assert g['x1']['ST'][0] == pytest.approx(T1, abs=0.03)


def test_sobol_linear_groups_and_intervals():
    # Y = 3 U0 + U1 (+ U2 ignored): S0 = 9/10, S1 = 1/10; a group of both is 1
    f = lambda U: np.column_stack([3 * U[:, 0] + U[:, 1], U[:, 2]])
    so = RP.sobol_indices(f, 3, {'a': [0], 'b': [1], 'ab': [0, 1]}, n=20000, seed=5, boot=50)
    assert so['groups']['a']['S'][0] == pytest.approx(0.9, abs=0.02)
    assert so['groups']['b']['S'][0] == pytest.approx(0.1, abs=0.02)
    assert so['groups']['ab']['S'][0] == pytest.approx(1.0, abs=0.02)
    lo, hi = so['groups']['a']['S_ci'][0]
    assert lo <= 0.9 <= hi + 0.02


# =========================================================================== the error model
def _sources(tags, st):
    return RP.resolve_sources(None, st, tags)


def test_error_model_modes():
    tags = ['F', 'L']
    st = RP.RobustSettings(mount_from='custom', mount_deg=1.0, mount_mm=10.0, thermal_from='custom', thermal_deg=0.0,
                           thermal_mm=0.0, vib_from='custom', vib_body_deg=0.0, vib_heave_mm=0.0, vib_flex_deg=0.0,
                           vib_flex_mm=0.0, calib_from='custom', calib_deg=0.0, calib_mm=0.0)
    u = np.random.default_rng(0).random(RP.ErrorModel(_sources(tags, st), tags, st).D)
    em = RP.ErrorModel(_sources(tags, st), tags, st)
    e = em.errors(u)
    for t in tags:
        da, dp = e[t]['true']
        assert np.abs(da).max() <= 1.0 + 1e-12 and np.abs(dp).max() <= 0.010 + 1e-12 and np.abs(da).max() > 0
        assert np.allclose(e[t]['used'][0], da) and np.allclose(e[t]['used'][1], dp)    # the calibration measures it
    st.mode = 'design'
    e = RP.ErrorModel(_sources(tags, st), tags, st).errors(u)
    for t in tags:
        assert np.allclose(e[t]['used'][0], 0) and np.allclose(e[t]['used'][1], 0)
    assert 'calibration' not in st.active()
    # one source at a time
    st.mode = 'per_vehicle'
    st.calib_deg = 0.5
    em = RP.ErrorModel(_sources(tags, st), tags, st)
    e = em.errors(u, only=['calibration'])
    assert np.allclose(e['F']['true'][0], 0) and np.abs(e['F']['used'][0]).max() > 0
    assert set(em.cols('mounting')).isdisjoint(em.cols('calibration'))
    assert len(em.cols()) == em.D


def test_settings_tolerant_and_clamped():
    s = RP.RobustSettings.from_dict({'samples': 'x', 'mount_deg': 1e9, 'mode': 'bogus', 'detect': 'no', 'seed': -5,
                                     'recordings': ['a', 3, None, ''], 'sobol_n': float('nan')})
    assert s.samples == RP.RobustSettings().samples and s.mount_deg == RP.LIMITS['mount_deg'][1]
    assert s.mode == 'per_vehicle' and s.detect is False and s.seed == 0 and s.recordings == ['a']
    assert RP.RobustSettings.from_dict(s.to_dict()) == s
    assert RP.RobustSettings.from_dict(None) == RP.RobustSettings()


# =========================================================================== the study on a synthetic drive
def test_zero_tolerance_zero_spread(rec):
    res = RP.run(None, settings=zero_settings(rec), detector=colour_detector)
    rr = res.recordings[0]
    assert rr.info['objects'] > 0 and rr.info['multi_view_objects'] > 0
    for cls, row in rr.bev.items():
        for rid, d in row.items():
            assert d['err_m']['mean'] == pytest.approx(0.0, abs=1e-9)
            assert d['err_p95_m'] == pytest.approx(0.0, abs=1e-9)
    ped = rr.det['pedestrian']['all']
    assert ped['nominal'] == pytest.approx(1.0)                 # the colour detector finds every visible person
    assert ped['recall']['p05'] == ped['recall']['p95'] == ped['nominal']
    g = rr.ghost['all']['all']
    assert g['rate']['mean'] == pytest.approx(g['nominal'])
    for s in rr.missing.values():
        assert s['mean'] == pytest.approx(0.0, abs=1e-12)
    # the records: one per recording, metrics per class and region
    assert len(res.records) == 1
    r = res.records[0]
    assert r['schema'] == 'vantage.perception_record/1' and r['id'] == RP._record_id(r)
    assert 'pedestrian' in r['metrics'] and all(k in next(iter(r['metrics']['pedestrian'].values()))
                                                for k in ('n_gt', 'tp', 'fp', 'fn', 'recall', 'precision', 'ap',
                                                          'ate_m'))
    assert rec not in json.dumps(r)                             # no user paths in a record


def test_tolerance_spreads_and_exports(rec, tmp_path):
    st = zero_settings(rec, mode='design', mount_deg=1.5, mount_mm=10.0, samples=4, sobol_n=64)
    res = RP.run(None, settings=st, detector=colour_detector)
    rr = res.recordings[0]
    errs = [d['err_m']['mean'] for row in rr.bev.values() for d in row.values()]
    assert max(errs) > 0.05
    src = rr.sobol['by_source']
    assert 'mounting' in src['groups'] and src['n'] == 64
    j = src['metrics'].index('err:all')
    # the only source: A_B^mounting is B itself, so S = (E[yB²] − E[yA yB]) / V and ST = E[(yA − yB)²] / 2V, both 1
    # in expectation
    assert src['groups']['mounting']['S'][j] == pytest.approx(1.0, abs=0.45)
    assert src['groups']['mounting']['ST'][j] == pytest.approx(1.0, abs=0.45)
    assert rr.worst and rr.worst['sample'] in range(4)
    img, text = RP.build_image(res, 0, 'worst', 'F', 1)
    assert img.shape == (H, W, 3) and 'worst build' in text
    paths = RP.export(res, None, str(tmp_path))
    names = {os.path.basename(p) for p in paths}
    assert {'robust_perception.json', 'robust_perception_regions.csv', 'robust_perception_sensitivity.csv',
            'perception_records.json'} <= names
    doc = json.load(open(tmp_path / 'robust_perception.json'))
    assert doc['schema'] == RP.SCHEMA and doc['recordings'][0]['bev'] and doc['frame'].startswith('vehicle')
    assert rec not in json.dumps(doc)                           # folders by name only
    s = RP.summary(res)
    assert s['recordings'][0]['objects'] == rr.info['objects']


def test_cancel_returns_none(rec):
    calls = []

    def cancel():
        calls.append(1)
        return len(calls) > 3
    assert RP.run(None, settings=zero_settings(rec), detector=colour_detector, cancel=cancel) is None


def test_no_recordings_raises():
    with pytest.raises(ValueError):
        RP.run(None, settings=RP.RobustSettings())


# =========================================================================== real data (skipped without it)
REAL = os.path.join(os.environ.get('VANTAGE_CARLA_DRIVES', '/nonexistent'), 'data-rig-6-cam-lidar-radars')


@pytest.mark.skipif(not os.path.isdir(REAL), reason='needs the CARLA drive with ground truth')
def test_real_recording_bev_only():
    st = RP.RobustSettings(recordings=[REAL], detect=False, sobol_n=8, bev_frame_step=20, boot=10)
    from vantage.core import samples
    p = samples.tigor_sample_project()
    v = next(v for v in p.variants if v.sensors)
    res = RP.run(p, settings=st, variant=v)
    rr = res.recordings[0]
    assert rr.info['objects'] > 20
    assert rr.nominal['lift_ok'] > 0.5
    assert rr.sobol['by_source']['groups']


# =========================================================================== the page
from test_gui import app, win, pump, settle, _wait   # noqa: E402,F401 (fixtures)


def _open(win, name, cls):
    if name not in win._page_widgets:
        pg = cls(win.ctx)
        win.pages.addWidget(pg)
        win._page_widgets[name] = pg
    win.set_page(name)
    return win._page_widgets[name]


def test_page_runs_and_undoes(app, win, rec, tmp_path):
    from vantage.ui.workspaces.robust_ws import RobustPage
    pg = _open(win, 'robust_perception', RobustPage)
    pg.activate()
    pump(app, 0.2)
    assert not pg.run()                                         # no recording yet: says so
    assert 'recording' in pg.status.text().lower()
    assert pg.add_recording(rec)
    assert pg.rec_list.count() == 1
    for k, v in (('mount_from', 'custom'), ('thermal_from', 'custom'), ('vib_from', 'custom'),
                 ('calib_from', 'custom'), ('samples', 2), ('sobol_n', 8), ('oat_samples', 1), ('frame_step', 2)):
        pg.set_option(k, v)
    pg.fields['mount_deg'].setValue(1.0)
    pg.fields['mount_deg'].committed.emit(1.0)
    pump(app, 0.1)
    assert win.store.project.analysis.robust_perception['mount_deg'] == 1.0
    win.store.undo.undo()
    pump(app, 0.1)
    assert win.store.project.analysis.robust_perception.get('mount_deg', 1.0) != 1.0 or \
        RP.settings_of(win.store.project).mount_deg == RP.RobustSettings().mount_deg
    win.store.undo.redo()
    pump(app, 0.1)
    pg.detector = colour_detector
    assert pg.run()
    _wait(app, lambda: pg.result is not None and not pg.busy, timeout=240)
    assert pg.c_rec.count() == 1 and pg.map.cells
    for fig in ('bev', 'oov', 'ghost', 'recall'):
        pg.fig.set_value(fig)
        pg.refresh_views()
        pump(app, 0.05)
    pg.tabs.setCurrentIndex(2)
    pg.refresh_image()
    pump(app, 0.1)
    assert pg.img.pixmap() is not None and not pg.img.pixmap().isNull()
    paths = pg.export(str(tmp_path))
    assert any(p.endswith('robust_perception.json') for p in paths)
    # project replaced while the page is open: the result goes
    from vantage.core import samples
    win.store.set_project(samples.tigor_sample_project())
    pump(app, 0.2)
    assert pg.result is None and pg.rec_list.count() == 0
