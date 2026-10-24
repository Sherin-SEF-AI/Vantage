"""As-built check: a synthetic "real" recording with KNOWN mounting errors, written by the sample dataset generator
(core.dataset: a ROS 2 bag and a nuScenes folder of the same drive) from a copy of the sample data rig in which ML is
pitched 2° down and rolled 1° and R is yawed 1.5°, while the plan keeps the nominal poses. Two cameras are added to the
rig: FE, a circular fisheye on the roof whose image circle lies inside the sensor, and NB, a narrow camera on the front
bumper looking up that sees none of the vehicle.

Checked: the static masks match the simulated masks of the perturbed copy; the fitted corrections recover the
injected errors within their stated errors (and within 0.3° where the outline constrains them); the cameras mounted as
planned hold; the rear camera R, whose bumper is the shade of the plain road in this scene, is not checkable from its
outline and the drive-log estimate against the lidar stands in (yaw 1.5°); NB is not checkable; the as-built coverage
equals an analysis of the perturbed copy; the fisheye's image circle matches the lens model; the nuScenes folder and a
plain image folder work as input; mapping overrides persist and undo; cancel mid-run; a recording with no images gives
a clear message; the maths on its own (settings, circle fit, outline refinement, a synthetic static mask with a known
outline); and the page."""
import json
import math
import os
import time

import numpy as np
import pytest

from vantage.core import asbuilt as AB
from vantage.core import dataset as DS
from vantage.core import samples
from vantage.core.analysis import Engine
from vantage.core.lens import Lens
from vantage.core.mathutil import rot_zyx
from vantage.core.project import AnalysisSettings, Camera

ERR = {'ML': (0.0, 2.0, 1.0), 'R': (1.5, 0.0, 0.0)}       # yaw, pitch, roll (deg) of the real rig against the plan
SCALE, DUR = 0.35, 2.5


def make_project(err=None):
    """The sample data rig with FE (circular fisheye on the roof) and NB (sees no body); err moves cameras."""
    p = samples.tigor_sample_project()
    v = next(v for v in p.variants if v.sensors)
    p.active_uid = v.uid
    p.lenses['circ'] = Lens(id='circ', name='Circular fisheye', module_id='sturdecam31', focal_mm=1.25,
                            projection='equidistant', max_half_deg=95.0)
    v.cameras.append(Camera(tag='FE', name='Roof fisheye', slot=6, x=-0.25, y=0.0, z=1.58, yaw=180.0, pitch=60.0,
                            lens_id='circ'))
    v.cameras.append(Camera(tag='NB', name='Up camera', slot=7, x=3.26, y=0.35, z=0.70, yaw=0.0, pitch=-12.0,
                            lens_id='h03r2'))
    for c in v.cameras:
        d = (err or {}).get(c.tag)
        if d:
            c.yaw, c.pitch, c.roll = c.yaw + d[0], c.pitch + d[1], c.roll + d[2]
    return p, v


@pytest.fixture(scope='module')
def drive(tmp_path_factory):
    """The recording (generated once) and the check of it against the plan (drive-log estimate on)."""
    out = str(tmp_path_factory.mktemp('asbuilt') / 'drive')
    eng = Engine()
    q, w = make_project(ERR)
    st = DS.DatasetSettings.from_dict({'duration_s': DUR, 'image_scale': SCALE, 'nuscenes': True, 'rosbag': True,
                                       'keyframe_hz': 2})
    gen = DS.generate(q, eng, st, out)
    p, v = make_project()
    t0 = time.time()
    res = AB.run(p, eng, AB.AsBuiltSettings(path=gen.bag_dir))
    return {'bag': gen.bag_dir, 'nus': gen.nuscenes_dir, 'out': out, 'eng': eng, 'true': (q, w), 'plan': (p, v),
            'res': res, 'seconds': time.time() - t0}


# ============================================================================ the maths on its own
def test_settings_round_trip_clamp_and_junk():
    st = AB.AsBuiltSettings.from_dict({'frames': 1e9, 'search_deg': float('nan'), 'static_sigma': 'x',
                                       'lidar_check': 'no', 'fit_translation': 1, 'tol_rotation_deg': -3,
                                       'mappings': {'/a': {'/f/image': 'F', 3: 'x'}, 'b': 'junk'}, 'path': 5,
                                       'unknown': 1})
    assert st.frames == AB.LIMITS['frames'][1] and st.search_deg == AB.AsBuiltSettings().search_deg
    assert st.static_sigma == AB.AsBuiltSettings().static_sigma and st.lidar_check is False
    assert st.fit_translation is True and st.tol_rotation_deg == AB.LIMITS['tol_rotation_deg'][0]
    assert st.mappings == {'/a': {'/f/image': 'F'}} and st.path == ''
    assert AB.AsBuiltSettings.from_dict(st.to_dict()) == st
    assert AB.AsBuiltSettings.from_dict(None) == AB.AsBuiltSettings()
    # an old project file without the slot: defaults
    a = AnalysisSettings.from_dict({'cell_m': 0.05})
    assert AB.AsBuiltSettings.from_dict(a.asbuilt) == AB.AsBuiltSettings()


def test_circle_fit_recovers_a_known_circle_through_outliers():
    rng = np.random.default_rng(4)
    a = np.linspace(0.3, 5.0, 400)                     # an arc of 270°
    P = np.c_[312.4 + 201.7 * np.cos(a), 250.9 + 201.7 * np.sin(a)] + rng.normal(0, 0.3, (400, 2))
    P = np.vstack([P, rng.uniform(0, 600, (60, 2))])    # a sixth of the points are junk
    f = AB.fit_circle(P)
    assert f['cx'] == pytest.approx(312.4, abs=0.1) and f['cy'] == pytest.approx(250.9, abs=0.1)
    assert f['r'] == pytest.approx(201.7, abs=0.1) and f['rms_px'] < 0.5 and f['arc_deg'] >= 260


def test_a_static_disc_in_a_moving_texture_gives_its_outline_to_a_tenth_of_a_pixel():
    """Frames of moving random texture with a static disc (the 'vehicle') sampled at pixel centres, noise and JPEG:
    the static mask is the disc, and the refined outline points lie on the true circle (the midpoint between pixel
    centres is unbiased, the median image's edge puts back pixels the moving world leaked into)."""
    import cv2
    h, w, cx, cy, R = 200, 260, 131.3, 97.8, 61.6
    yy, xx = np.mgrid[0:h, 0:w]
    disc = (xx - cx) ** 2 + (yy - cy) ** 2 <= R * R
    rng = np.random.default_rng(1)
    frames = []
    big = rng.uniform(40, 220, (h // 8 + 20, w // 8 + 30)).astype(np.float32)
    tex = cv2.GaussianBlur(cv2.resize(big, ((w // 8 + 30) * 8, (h // 8 + 20) * 8), interpolation=cv2.INTER_CUBIC),
                           (0, 0), 2.0)
    for k in range(20):
        dx = 7 * k
        img = tex[40:40 + h, dx % 150: dx % 150 + w].copy()
        img[disc] = 150.0 + 30.0 * np.sin(xx[disc] / 9.0)            # the body: static shading
        img += rng.normal(0, 2.0, img.shape)
        ok, buf = cv2.imencode('.jpg', np.clip(img, 0, 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 90])
        frames.append(cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE))
    M, D, sigma = AB.frame_stats(frames)
    assert 0.5 <= sigma < 4.0
    st = AB.AsBuiltSettings()
    cl = AB.classify(M, D, sigma, disc, np.ones((h, w), bool), 12.0, st)
    veh = cl['labels'] == AB.VEH
    iou = (veh & disc).sum() / (veh | disc).sum()
    assert iou > 0.97
    pts, nrm = AB._pairs(veh, cl['labels'] == AB.WORLD)
    nrm = AB._smooth_normals(veh, pts, nrm)
    ref, n = AB.refine_outline(M, pts, nrm, st.edge_contrast)
    err = np.hypot(ref[:, 0] - cx, ref[:, 1] - cy) - R
    assert n > 0.5 * len(pts)
    assert abs(float(np.mean(err))) < 0.1 and float(np.mean(np.abs(err))) < 0.4
    # a mask a pixel short all round (blur, a leak the threshold missed): the median image's edge puts it back
    small = cv2.erode(veh.astype(np.uint8), np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)) > 0
    pts, nrm = AB._pairs(small, ~small)
    nrm = AB._smooth_normals(small, pts, nrm)
    raw = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) - R
    ref, n = AB.refine_outline(M, pts, nrm, st.edge_contrast)
    err = np.hypot(ref[:, 0] - cx, ref[:, 1] - cy) - R
    assert float(np.mean(raw)) < -0.8 and n > 0.5 * len(pts)
    assert abs(float(np.median(err))) < 0.25


def test_pairs_put_the_outline_between_pixel_centres():
    A = np.zeros((6, 8), bool)
    A[2:4, 3:6] = True
    pts, nrm = AB._pairs(A, ~A)
    assert len(pts) == 10                                  # 3 + 3 along the rows, 2 + 2 down the columns
    assert {tuple(p) for p in pts if p[0] == 2.5} == {(2.5, 2.0), (2.5, 3.0)}
    assert all(np.linalg.norm(n) == 1.0 for n in nrm)
    left = pts[:, 0] == 2.5
    assert np.all(nrm[left] == [-1.0, 0.0])               # out of the vehicle


def test_predicted_circle_follows_the_lens_model():
    p, v = make_project()
    fe = v.by_tag('FE')
    intr = AB._intr(p, fe)
    pc = AB.predicted_circle(intr, (538, 672))
    # equidistant: r = f θ / pixel; the sensor's 1920 px map to 672
    r_sensor = 1.25 * math.radians(95.0) / 0.003
    assert pc['r'] == pytest.approx(r_sensor * 672 / 1920, rel=1e-6) and pc['visible']
    assert pc['cx'] == pytest.approx(960 * 672 / 1920 - 0.5)
    f = v.by_tag('F')
    assert not AB.predicted_circle(AB._intr(p, f), (538, 672))['visible']          # the H05R2 circle covers it


def test_image_folders_are_found_by_subfolder_or_prefix(tmp_path):
    import cv2
    img = np.full((24, 32, 3), 90, np.uint8)
    for d in ('F', 'ML'):
        os.makedirs(tmp_path / 'a' / d)
        for k in range(3):
            cv2.imwrite(str(tmp_path / 'a' / d / f'{k:04d}.png'), img)
    os.makedirs(tmp_path / 'b')
    for t in ('F', 'QR'):
        for k in range(2):
            cv2.imwrite(str(tmp_path / 'b' / f'{t}_{1767268800123 + k * 100}.jpg'), img)
    assert {k: len(v) for k, v in AB.image_streams(tmp_path / 'a').items()} == {'F': 3, 'ML': 3}
    assert sorted(AB.image_streams(tmp_path / 'b')) == ['F', 'QR']
    assert AB.recording_kind(tmp_path / 'a') == 'images'
    info = AB.open_recording(tmp_path / 'b')
    assert info.source == 'images' and info.counts == {'F': 2, 'QR': 2}
    assert AB._name_time(str(tmp_path / 'b' / 'F_1767268800223.jpg')) == pytest.approx(1767268800.223)
    os.makedirs(tmp_path / 'c')
    with pytest.raises(ValueError, match='holds no recording'):
        AB.recording_kind(tmp_path / 'c')
    with pytest.raises(FileNotFoundError):
        AB.recording_kind(tmp_path / 'nothing')


# ============================================================================ the recording
def test_streams_are_mapped_and_read_while_moving(drive):
    res = drive['res']
    assert set(res.mapping.values()) == {c.tag for c in drive['plan'][1].cameras}
    assert res.recording['motion'] == 'odometry' and res.recording['duration_s'] == pytest.approx(DUR, abs=0.15)
    for r in res.cameras:
        assert r['frames'] >= 12, r['tag']
        assert r['size'] == (672, 538) and r['ashape'] == (538, 672)


def test_static_masks_match_the_simulation_of_the_rig_as_built(drive):
    """Over the pixels the frames decide, the static mask is the perturbed copy's simulated mask."""
    res, eng = drive['res'], drive['eng']
    q, w = drive['true']
    for tag in ('F', 'ML', 'MR', 'QL', 'FE'):
        r = res.camera(tag)
        L = r['_labels']
        cam = w.by_tag(tag)
        body, _, _ = AB.sim_mask(q, cam, AB._intr(q, cam), r['ashape'], eng)
        decided = (L == AB.VEH) | (L == AB.WORLD)
        veh = L == AB.VEH
        iou = ((veh & body & decided).sum()) / ((veh | body) & decided).sum()
        assert iou >= 0.9, (tag, iou)
        assert r['iou_fit'] >= 0.9, tag
    ml = res.camera('ML')
    assert ml['iou_fit'] > ml['iou_plan'] + 0.05            # the fitted pose explains the mask


def test_the_fitted_corrections_recover_the_injected_errors(drive):
    res = drive['res']
    ml = res.camera('ML')
    fit = ml['fit']
    assert ml['check'] == 'fitted' and ml['apply'] and ml['verdict'] == 'holds_with_corrections'
    for k, true in enumerate(ERR['ML']):
        assert fit['constrained'][k], AXES_ERR(ml)
        err = abs(fit['delta_deg'][k] - true)
        assert err <= 3.0 * fit['stderr_deg'][k] + 0.02, (k, fit['delta_deg'], fit['stderr_deg'])
        assert err <= 0.3, (k, fit['delta_deg'])
    assert ml['correction_deg'][1] == pytest.approx(2.0, abs=0.3) and ml['correction_deg'][2] == pytest.approx(1.0,
                                                                                                                  abs=0.3)
    # the cameras mounted as planned hold, within their stated errors of zero
    for tag in ('F', 'MR', 'QL'):
        r = res.camera(tag)
        assert r['verdict'] == 'holds' and not r['apply'], (tag, r['fit'])
        f = r['fit']
        for k in range(3):
            if f['constrained'][k]:
                assert abs(f['delta_deg'][k]) <= 3.0 * f['stderr_deg'][k] + 0.05, (tag, f)


def AXES_ERR(r):
    return f'{r["tag"]}: {r.get("fit")}'


def test_a_camera_the_outline_cannot_check_takes_the_lidar_estimate_or_says_so(drive):
    res = drive['res']
    rr = res.camera('R')                               # its bumper is the shade of the plain road: no outline
    assert rr['check'] == 'not_checkable' and rr['reason']
    assert rr['source'] == 'lidar' and rr['apply'] and rr['verdict'] == 'holds_with_corrections'
    assert rr['correction_deg'][0] == pytest.approx(1.5, abs=0.3)
    assert abs(rr['correction_deg'][1]) < 0.3 and abs(rr['correction_deg'][2]) < 0.3
    nb = res.camera('NB')                              # sees none of the vehicle: its outline cannot tell …
    assert nb['check'] == 'not_checkable' and nb['fit'] is None
    assert 'too little of the vehicle' in nb['reason'] and nb['outline_px'] < AB.MIN_OUTLINE_PX
    assert any(f['message'].startswith('NB: not checkable from its outline') for f in res.findings)
    # … and only the drive-log estimate decides it (without that, the page test sees it 'Not checkable')
    assert (nb['verdict'], nb.get('source')) in (('holds', 'lidar'), ('not_checkable', None))
    # the two estimates of ML agree within their errors
    ml = res.camera('ML')
    assert ml['lidar'] is not None and ml['lidar']['delta_deg'][1] == pytest.approx(2.0, abs=0.3)
    assert ml['agree'] is True


def test_the_fisheye_image_circle_matches_the_lens_model(drive):
    fe = drive['res'].camera('FE')
    c = fe['circle']
    assert c['visible_predicted'] and c['visible'] and c['measured'] is not None
    assert c['centre_error_px'] < 1.0 and abs(c['radius_error_px']) < 1.0, c
    assert c['measured']['arc_deg'] >= 300 and c['measured']['rms_px'] < 0.8
    assert fe['check'] == 'fitted' and fe['verdict'] == 'holds', fe.get('fit')
    assert not any('image circle' in f['message'] for f in drive['res'].findings)


def test_the_as_built_coverage_equals_an_analysis_of_the_rig_as_built(drive):
    res, eng = drive['res'], drive['eng']
    q, w = drive['true']
    p, v = drive['plan']
    assert res.verdict == 'holds_with_corrections'
    assert res.asbuilt is not None and sorted(res.metrics['corrected']) == ['ML', 'R']
    true = eng.analyse(q, variant=w)
    cov = res.coverage
    for row in cov['rows']:
        assert row['as_built'] == pytest.approx(true.metrics[row['key']], abs=0.3 if row['unit'] == '%' else 0.03), row
    for tag, d in cov['cameras'].items():
        seen = true.per_cam[w.by_tag(tag).uid]['seen_m2']
        assert d['as_built_m2'] == pytest.approx(seen, rel=0.02, abs=0.02), tag
    # and the as-built cameras are the rig as built
    for c in res.asbuilt.cameras:
        t = w.by_tag(c.tag)
        R1, R2 = rot_zyx(c.yaw, c.pitch, c.roll), rot_zyx(t.yaw, t.pitch, t.roll)
        ang = math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(R1 @ R2.T) - 1) / 2))))
        assert ang < 0.35, (c.tag, ang)
    assert cov['holds'] and cov['requirements'] is not None and cov['requirements']['items']
    assert res.asbuilt.name.startswith(f'{v.name} · as built ')


def test_summary_doc_csv_files_and_export(drive, tmp_path):
    res = drive['res']
    p, v = drive['plan']
    s = AB.summary(res)
    json.dumps(s)
    assert s['verdict'] == 'holds_with_corrections' and s['corrected'] == res.metrics['corrected']
    assert len(s['per_camera']) == 8 and s['coverage']
    doc = AB.to_doc(res, p)
    text = json.dumps(doc)
    assert doc['schema'] == AB.SCHEMA and doc['frame'] == 'vehicle (ISO 8855, rear axle)' and 'degrees' in doc['units']
    assert 'NaN' not in text and 'Infinity' not in text
    ml = next(c for c in doc['cameras'] if c['camera'] == 'ML')
    assert ml['applied'] and ml['fit']['constrained'] == [True, True, True] and ml['estimates_agree'] is True
    assert doc['as_built_variant']['name'] == res.asbuilt.name
    csv_text = AB.to_csv(res)
    assert csv_text.splitlines()[0].startswith('camera,stream,verdict') and len(csv_text.splitlines()) == 9
    names = [n for n, _ in AB.files(res, p)]
    assert names[:3] == ['asbuilt_check.json', 'asbuilt_cameras.csv', 'asbuilt_coverage.png']
    assert 'asbuilt_overlay_ML.png' in names and 'asbuilt_overlay_FE.png' in names
    paths = AB.export(res, p, str(tmp_path / 'exp'))
    assert all(os.path.getsize(x) > 0 for x in paths)
    import cv2
    img = cv2.imread(next(x for x in paths if x.endswith('overlay_ML.png')))
    assert img is not None and img.shape[1] > 2 * 640


def test_the_as_built_variant_and_the_issues_list(drive):
    res = drive['res']
    p, v = drive['plan']
    q = p.snapshot()
    a = AB.asbuilt_variant(q, res, '2026-10-24')
    assert a.name == f'{v.name} · as built 2026-10-24' and 'ML' in a.note and 'R ' in a.note
    ml, ml0 = a.by_tag('ML'), v.by_tag('ML')
    assert ml.pitch == pytest.approx(ml0.pitch + 2.0, abs=0.3) and ml.link_uid == ''
    assert {c.uid for c in a.cameras}.isdisjoint({c.uid for c in v.cameras})
    assert a.by_tag('F').yaw == v.by_tag('F').yaw
    # the Issues list: while the plan is as checked
    q.analysis.asbuilt = AB.AsBuiltSettings(path=res.path).to_dict()
    AB.remember(q, res)
    iss = AB.issues(q)
    assert {i.objects for i in iss} == {'ML', 'R'} and all(i.severity == 'serious' for i in iss)
    q.variant.by_tag('ML').pitch += 0.5                    # the plan moved on: the check no longer applies
    assert AB.issues(q) == []


def test_mapping_overrides_take_a_stream_away_and_back(drive):
    p, v = drive['plan']
    info = AB.open_recording(drive['bag'])
    m = AB.mapping_for(v, info, {})
    ml_topic = next(t for t, g in m.items() if g == 'ML')
    assert ml_topic.startswith('/ml/')
    m2 = AB.mapping_for(v, info, {ml_topic: ''})
    assert 'ML' not in m2.values()
    mr_topic = next(t for t, g in m.items() if g == 'MR')
    m3 = AB.mapping_for(v, info, {mr_topic: 'ML'})           # a stream given to another camera takes it over
    assert m3[mr_topic] == 'ML' and ml_topic not in m3
    assert set(AB.compatible_tags(v, 'sensor_msgs/msg/CompressedImage')) == {c.tag for c in v.cameras}
    assert AB.compatible_tags(v, 'sensor_msgs/msg/PointCloud2') == []


# ============================================================================ other inputs
def _only(p, v, tags):
    for c in v.cameras:
        c.visible = c.tag in tags
    return p


def test_a_nuscenes_folder_is_read(drive):
    p, v = make_project()
    _only(p, v, ('ML', 'F'))
    res = AB.run(p, drive['eng'], AB.AsBuiltSettings(path=drive['nus'], lidar_check=False))
    assert res.source == 'nuscenes' and res.recording['motion'] == 'odometry'
    ml = res.camera('ML')
    assert ml['apply'] and ml['correction_deg'][1] == pytest.approx(2.0, abs=0.3), ml['fit']
    assert res.camera('F')['verdict'] == 'holds'
    assert [c['tag'] for c in res.cameras] == ['F', 'ML']                   # hidden cameras are not checked


def test_a_plain_image_folder_works(drive, tmp_path):
    """The bag's ML and F frames written as PNG files in a folder per camera: no motion data, so frames are kept by
    how much the picture changed."""
    import cv2
    from vantage.core import rosbag as RB
    root = tmp_path / 'imgs'
    k = {}
    with RB.BagReader(drive['bag']) as rd:
        for topic, t, msg in rd.messages([t for t in rd.topics if t.startswith(('/ml/image', '/f/image'))]):
            tag = 'ML' if topic.startswith('/ml/') else 'F'
            os.makedirs(root / tag, exist_ok=True)
            img = RB.image_array(msg, to_rgb=True)
            cv2.imwrite(str(root / tag / f'{k.get(tag, 0):05d}.png'), img[..., ::-1])
            k[tag] = k.get(tag, 0) + 1
    p, v = make_project()
    _only(p, v, ('ML', 'F'))
    res = AB.run(p, drive['eng'], AB.AsBuiltSettings(path=str(root)))
    assert res.source == 'images' and res.recording['motion'] == 'frames'
    assert any('An image folder has no lidar' in n for n in res.notes)
    ml = res.camera('ML')
    assert ml['frames'] >= 10 and ml['apply'], ml
    assert ml['correction_deg'][1] == pytest.approx(2.0, abs=0.3) and ml['correction_deg'][2] == pytest.approx(1.0,
                                                                                                                  abs=0.3)
    assert res.camera('F')['verdict'] == 'holds'


def test_a_recording_without_images_says_so(tmp_path):
    from vantage.core import rosbag as RB
    path = str(tmp_path / 'imu_only')
    with RB.BagWriter(path) as bw:
        bw.add_topic('/imu', 'sensor_msgs/msg/Imu')
        for k in range(20):
            bw.write('/imu', 1_700_000_000_000_000_000 + k * 10_000_000,
                     {'header': RB.header(1_700_000_000_000_000_000 + k * 10_000_000, 'imu'),
                      'linear_acceleration': {'z': 9.81}})
    p, v = make_project()
    with pytest.raises(ValueError, match='holds no images'):
        AB.run(p, Engine(), AB.AsBuiltSettings(path=path))
    with pytest.raises(ValueError, match='No recording'):
        AB.run(p, Engine(), AB.AsBuiltSettings())
    _only(p, v, ())
    with pytest.raises(ValueError, match='no visible camera'):
        AB.run(p, Engine(), AB.AsBuiltSettings(path=path))


def test_cancel_mid_run_and_a_scan_used_again(drive):
    p, v = make_project()
    _only(p, v, ('ML',))
    seen = []

    def prog(f, text=''):
        seen.append(f)
    out = AB.run(p, drive['eng'], AB.AsBuiltSettings(path=drive['bag'], lidar_check=False), progress=prog,
                 cancel=lambda: len(seen) > 3)
    assert out is None and seen
    res = AB.run(p, drive['eng'], AB.AsBuiltSettings(path=drive['bag'], lidar_check=False))
    t0 = time.time()
    again = AB.run(p, drive['eng'], AB.AsBuiltSettings(path=drive['bag'], lidar_check=False, tol_rotation_deg=0.5),
                   scan=res.scan)
    assert again.scan is res.scan and time.time() - t0 < 30
    assert again.camera('ML')['apply']
    other = AB.run(p, drive['eng'], AB.AsBuiltSettings(path=drive['bag'], lidar_check=False, frames=12),
                   scan=res.scan)
    assert other.scan is not res.scan and other.camera('ML')['frames'] <= 12


def test_a_shift_is_fitted_when_asked_and_held_where_the_outline_cannot_tell(drive):
    p, v = make_project()
    _only(p, v, ('ML',))
    res = AB.run(p, drive['eng'], AB.AsBuiltSettings(path=drive['bag'], lidar_check=False, fit_translation=True))
    ml = res.camera('ML')
    fit = ml['fit']
    assert fit['shift_m'] is not None and len(fit['constrained_t']) == 3
    for k in range(3):
        if fit['constrained_t'][k]:
            assert abs(fit['shift_m'][k]) <= 3 * fit['stderr_m'][k] + 0.01, fit
        else:
            assert fit['shift_m'][k] == 0.0
    assert ml['correction_deg'][1] == pytest.approx(2.0, abs=0.4), fit


# ============================================================================ the page
from test_gui import app, win, pump, settle, _wait   # noqa: E402,F401 (fixtures)


def _open(win, name, cls):
    if name not in win._page_widgets:
        pg = cls(win.ctx)
        win.pages.addWidget(pg)
        win._page_widgets[name] = pg
    win.set_page(name)
    return win._page_widgets[name]


def test_page_opens_runs_shows_adds_the_variant_and_undoes(app, win, drive, tmp_path):
    from vantage.ui.workspaces.asbuilt_ws import AsBuiltPage
    p, v = make_project()
    _only(p, v, ('F', 'ML', 'NB'))
    win.store.set_project(p)
    pump(app, 0.3)
    pg = _open(win, 'asbuilt', AsBuiltPage)
    pump(app, 0.2)
    assert not pg.b_run.isEnabled() and pg.topics.rowCount() == 0
    assert pg.open_path(drive['bag'])
    pump(app, 0.2)
    assert pg.b_run.isEnabled() and pg.topics.rowCount() == 8          # the image streams only
    row = next(r for r in range(pg.topics.rowCount()) if pg.topics.item(r, 0).text().startswith('/ml/'))
    assert pg.topics.cellWidget(row, 3).currentData() == 'ML'
    assert pg.recent.count() >= 2 and pg.recent.itemData(1) == AB.norm_path(drive['bag'])
    # a setting through its field, undoable
    pg.fields['tol_rotation_deg'].set_value(0.5)
    pg.fields['tol_rotation_deg'].committed.emit(0.5)
    assert pg.settings.tol_rotation_deg == 0.5
    win.store.undo.undo()
    pump(app, 0.1)
    assert pg.settings.tol_rotation_deg == 0.3 and pg.settings.path == AB.norm_path(drive['bag'])
    # a mapping edit through the drop-down: the F stream to no camera, persisted and undone
    rf = next(r for r in range(pg.topics.rowCount()) if pg.topics.item(r, 0).text().startswith('/f/'))
    cb = pg.topics.cellWidget(rf, 3)
    cb.setCurrentIndex(0)
    cb.activated.emit(0)
    pump(app, 0.1)
    f_topic = pg.topics.item(rf, 0).text()
    assert pg.settings.overrides(drive['bag']) == {f_topic: ''} and pg.b_reset.isEnabled()
    assert json.loads(json.dumps(win.store.project.analysis.asbuilt))['mappings']    # saved with the project
    win.store.undo.undo()
    pump(app, 0.1)
    assert pg.settings.overrides(drive['bag']) == {}
    # the drive-log estimate off keeps the run short
    pg.toggles['lidar_check'].click()
    pump(app, 0.1)
    assert pg.settings.lidar_check is False
    assert pg.run()
    _wait(app, lambda: not pg.busy and pg.result is not None, 240)
    res = pg.result
    assert 'Failed' not in pg.status.text() and res.verdict == 'holds_with_corrections'
    assert pg.cams.rowCount() == 3 and pg.tiles['verdict'].value == 'Holds'
    assert pg.tiles['verdict'].note.startswith('with corrections')
    assert pg.cov.rowCount() == len(AB.COV_KEYS) and len(pg.findings) >= len(res.findings)
    # pick ML in the table: the overlay shows its frame with both outlines; fitted mode moves the simulated one
    r = next(i for i in range(pg.cams.rowCount()) if pg.cams.item(i, 0).text() == 'ML')
    pg.cams.selectRow(r)
    pump(app, 0.1)
    assert pg.cam_pick.currentData() == 'ML' and pg.view.img is not None
    plan_sim = [ln.copy() for ln in pg.view.layers['sim']]
    assert pg.view.layers['real'] and plan_sim
    pg.mode.buttons['fitted'].click()
    pump(app, 0.1)
    assert 'fitted' in pg.view.caption and not all(
        a.shape == b.shape and np.allclose(a, b) for a, b in zip(plan_sim, pg.view.layers['sim']))
    pg.backdrop.buttons['median'].click()
    pump(app, 0.1)
    assert pg.view.img is not None and pg.view.img.width() == res.camera('ML')['ashape'][1]
    nb = next(i for i in range(pg.cams.rowCount()) if pg.cams.item(i, 0).text() == 'NB')
    assert pg.cams.item(nb, 13).text() == 'Not checkable'
    # export
    paths = pg.export(str(tmp_path / 'exp'))
    assert any(x.endswith('asbuilt_check.json') for x in paths)
    # add the as-built variant: one undo step
    n = len(win.store.project.variants)
    assert pg.b_add.isEnabled() and '(1)' in pg.b_add.text()
    pg.b_add.click()
    pump(app, 0.2)
    pj = win.store.project
    assert len(pj.variants) == n + 1 and ' · as built ' in pj.variant.name
    assert pj.variant.by_tag('ML').pitch == pytest.approx(v.by_tag('ML').pitch + 2.0, abs=0.3)
    win.store.undo.undo()
    pump(app, 0.2)
    assert len(win.store.project.variants) == n
    assert pg.summary()['verdict'] == 'holds_with_corrections'
    # a camera moved in the plan makes the result stale
    def turn(pj):
        pj.variant.by_tag('F').yaw += 1.0
    win.store.edit('variant', turn, 'turn F')
    pump(app, 0.1)
    assert pg.stale and 'stale' in pg.status.text()
    # a new project drops the result
    win.store.set_project(samples.tigor_sample_project())
    pump(app, 0.2)
    assert pg.result is None and pg.cams.rowCount() == 0 and pg.view.img is None


def test_page_cancel(app, win, drive):
    from vantage.ui.workspaces.asbuilt_ws import AsBuiltPage
    p, v = make_project()
    _only(p, v, ('ML', 'F', 'MR', 'QL'))
    p.analysis.asbuilt = AB.AsBuiltSettings(path=drive['bag'], lidar_check=False).to_dict()
    win.store.set_project(p)
    pump(app, 0.2)
    pg = _open(win, 'asbuilt', AsBuiltPage)
    pump(app, 0.1)
    assert pg.run()
    pump(app, 0.05)
    pg.cancel()
    _wait(app, lambda: not pg.busy, 120)
    assert pg.result is None and 'Cancelled' in pg.status.text()
