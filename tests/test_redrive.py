"""Tests for re-driving a recorded drive through a proposed rig (core.redrive) and its part of the Log replay page.

References: the novel-view angle against the closed form of a camera moved sideways in front of a wall
(atan(d / D)), the extrapolation distance against the move itself, the additive isotonic fit against a known monotone
answer and pool-adjacent-violators against a hand-worked case, the re-shot export against the variant it was made
for (core.lens intrinsics, the camera poses, the drive's ego poses and boxes) and against the readers that must open
it (core.splat_reproject, core.detection_eval, core.drivelog, a ROS 2 bag). A 1 s synthetic drive of the sample's data
rig (core.dataset, images at 15 %) runs everything on the CPU.
Run with:  QT_QPA_PLATFORM=offscreen python3 -m pytest tests/test_redrive.py -q
"""
from __future__ import annotations

import json
import math
import os
import sys

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np  # noqa: E402
import pytest  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

torch = pytest.importorskip('torch')
cv2 = pytest.importorskip('cv2')

from vantage.core import dataset as DSET  # noqa: E402
from vantage.core import redrive as RD  # noqa: E402
from vantage.core import splat_raster as SR  # noqa: E402
from vantage.core import splat_reproject as SP  # noqa: E402
from vantage.core.analysis import Engine  # noqa: E402
from vantage.core.lens import resolve  # noqa: E402
from vantage.core.mathutil import BODY_TO_OPTICAL, rot_zyx  # noqa: E402
from vantage.core.project import Variant  # noqa: E402
from vantage.core.samples import tigor_sample_project  # noqa: E402

from test_gui import app, win, pump, _wait  # noqa: E402,F401  (fixtures)

CPU = torch.device('cpu')


# ----------------------------------------------------------------------------- settings and the fit
def test_settings_are_tolerant():
    s = RD.RedriveSettings.from_dict({'error_budget': 1e6, 'conf_scale': 'x', 'schedule': 'nonsense',
                                      'sources': ['loo', 'junk', 'holdout'], 'rosbag': 'yes', 'reuse_share': 0.3,
                                      'collect_share': 0.9, 't_start': 5, 't_end': 2, 'calib_views': 2.6})
    assert s.error_budget == 128.0 and s.conf_scale == 0.25 and s.schedule == 'recorded'
    assert s.sources == ['holdout', 'loo'] and s.rosbag is False and s.collect_share == 0.3 and s.t_end == 0.0
    assert s.calib_views == 4
    assert RD.RedriveSettings.from_dict(s.to_dict()) == s
    assert RD.RedriveSettings.from_dict(None) == RD.RedriveSettings()
    p = tigor_sample_project()
    p.analysis.redrive = {'error_budget': 7}
    assert RD.settings_of(p).error_budget == 7.0


def test_pava_against_a_hand_worked_case():
    # 1, 3, 2, 4 with weights 1, 1, 3, 1: the violating pair (3, 2) pools to (3 + 6) / 4 = 2.25
    assert np.allclose(RD.pava([1, 3, 2, 4], [1, 1, 3, 1]), [1, 2.25, 2.25, 4])
    assert np.allclose(RD.pava([4, 2, 3, 1], [1, 1, 1, 1], increasing=False), [4, 2.5, 2.5, 1])
    assert np.allclose(RD.pava([5, 5, 5], [1, 2, 3]), [5, 5, 5])


def test_additive_isotonic_fit_recovers_a_known_monotone_answer():
    """error = 2 + 0.2·angle (steps of the angle) + 6·unseen + noise, the other features noise: the fit is monotone
    in the angle as the feature's direction says, flat-ish in the noise features, and calibrated in-sample."""
    rng = np.random.default_rng(1)
    n = 20000
    X = np.zeros((n, len(RD.FEATURE_NAMES)))
    ang = rng.uniform(0, 40, n)
    uns = (rng.uniform(size=n) < 0.2).astype(float)
    X[:, RD.FEATURE_NAMES.index('angle_deg')] = ang
    X[:, RD.FEATURE_NAMES.index('unseen')] = uns
    X[:, RD.FEATURE_NAMES.index('spread')] = rng.uniform(0, 1, n)
    y = 2 + 0.2 * ang + 6 * uns + rng.normal(0, 0.5, n)
    m = RD.fit_error_model(X, np.maximum(y, 0), bins=16)
    k = RD.FEATURE_NAMES.index('angle_deg')
    assert np.all(np.diff(m.steps[k]) >= -1e-9) and m.steps[k][-1] - m.steps[k][0] > 6.0
    ku = RD.FEATURE_NAMES.index('unseen')
    assert m.steps[ku][-1] - m.steps[ku][0] == pytest.approx(6.0, abs=0.3)
    ks = RD.FEATURE_NAMES.index('spread')
    assert m.steps[ks][-1] - m.steps[ks][0] < 0.5
    e = m.expected(X)
    assert abs(e.mean() - y.mean()) < 0.1 and np.mean(np.abs(e - y)) < 1.0
    # P(error ≤ budget) falls as the prediction rises, and matches the share measured
    p = m.p_within(X, 6.0)
    assert abs(p.mean() - np.mean(y <= 6.0)) < 0.02
    lo, hi = X.copy()[:1], X.copy()[:1]
    lo[0, k], hi[0, k] = 1.0, 39.0
    assert m.expected(hi)[0] > m.expected(lo)[0] and m.p_within(hi, 6.0)[0] < m.p_within(lo, 6.0)[0]
    back = RD.ErrorModel.from_dict(json.loads(json.dumps(m.to_dict())))
    assert np.allclose(back.expected(X[:500]), e[:500])
    # reasons: an unseen pixel is 'unseen', a far-angled one 'viewpoint'
    r = m.reasons(np.vstack([np.where(np.arange(X.shape[1]) == ku, 1.0, 0.0), hi[0]]))
    assert [RD.REASONS[i] for i in r] == ['unseen', 'viewpoint']
    rel = RD.reliability(e, y, 6.0, p)
    assert rel['spearman'] > 0.9 and rel['calibration_error'] < 0.3 and len(rel['deciles']) == 10


# ----------------------------------------------------------------------------- a synthetic scene with a known answer
def _wall(D=10.0, half=8.0, step=0.1, seed=0):
    g = np.arange(-half, half + 1e-9, step)
    yy, zz = np.meshgrid(g, g)
    P = np.stack([np.full(yy.size, D), yy.ravel(), zz.ravel()], 1).astype(np.float32)
    n = len(P)
    R = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    q = SR.rot_to_quat_np(np.repeat(R[None], n, 0)).astype(np.float32)
    sc = np.tile([step * 0.6, step * 0.6, 0.004], (n, 1)).astype(np.float32)
    col = np.random.default_rng(seed).uniform(0.1, 0.9, (n, 3)).astype(np.float32)
    return SP.Scene(P, sc, q, np.full(n, 0.95, np.float32), col, np.full(n, -1, np.int32), np.zeros(n, np.int32), [],
                    np.full((16, 64, 3), 0.6, np.float32), np.zeros(3))


def _cam_model(w=64, h=48):
    # the principal point on the centre of pixel (w // 2, h // 2): that pixel's ray is the optical axis
    return SP.CamModel(w, h, 40.0, 40.0, w // 2 + 0.5, h // 2 + 0.5, 'equidistant', (0, 0, 0, 0), math.radians(80))


def test_confidence_is_monotone_in_novelty_with_the_known_angle():
    """A wall 10 m ahead recorded by one camera at the origin; a camera moved d to the side sees the point its centre
    ray hits (10, d, 0) at atan(d / 10) from the recorded ray, d from the recorded centre. The expected error of a
    model fitted on errors that grow with the angle grows with d."""
    D = 10.0
    scene = _wall(D)
    traj = SP.Trajectory([0.0, 2.0], np.zeros((2, 3)), np.tile([1.0, 0, 0, 0], (2, 1)))
    m = _cam_model()
    R = BODY_TO_OPTICAL.T                                       # optical -> vehicle, looking along +x
    cam = SP.DriveCamera('F', 'CAM_F', m, R, np.zeros(3), [(0.0, '', None), (0.5, '', None), (1.0, '', None)])
    drive = SP.Drive('synthetic', '.', 'nuscenes', 0, np.zeros(3), traj, {'F': cam}, {}, {})
    sst = SP.SplatSettings(holdout_every=0, device='cpu')
    ev = RD.build_evidence(scene, drive, sst, dev=CPU)
    assert ev.n == 3 and np.allclose(ev.lidar, 0.0)
    ts = SP.TorchScene(scene, drive, CPU)
    ang, ext = [], []
    for d in (0.0, 1.0, 2.0, 4.0):
        fe = RD.render_features(ts, scene, ev, drive, m, R, np.array([0.0, d, 0.0]), 0.5, m.fx)
        cy, cx = m.height // 2, m.width // 2
        assert fe['views_log'][cy, cx] == pytest.approx(math.log1p(3))
        assert fe['angle_deg'][cy, cx] == pytest.approx(math.degrees(math.atan2(d, D)), abs=0.1)
        assert fe['extrap_m'][cy, cx] == pytest.approx(d, abs=0.02)
        assert fe['X_v'][cy, cx] == pytest.approx([D, d, 0.0], abs=0.02)
        ang.append(float(np.nanmedian(fe['angle_deg'][fe['valid']])))
        ext.append(fe)
    assert ang == sorted(ang) and ang[-1] > ang[0] + 15
    # unseen: a camera turned to look where the recorded one never did
    side = rot_zyx(90.0, 0.0, 0.0) @ BODY_TO_OPTICAL.T
    fe = RD.render_features(ts, scene, ev, drive, m, side, np.zeros(3), 0.5, m.fx)
    assert np.nanmean(fe['unseen'][fe['valid']]) > 0.5
    # the error model: errors that grow with the angle -> the expected error grows with the move
    rng = np.random.default_rng(0)
    X = np.zeros((4000, len(RD.FEATURE_NAMES)))
    X[:, 0] = rng.uniform(0, 30, 4000)
    model = RD.fit_error_model(X, 1.0 + X[:, 0] + rng.normal(0, 0.3, 4000), bins=12)
    cal = RD.Calibration(model, 6.0, {}, {}, {}, {}, {}, 1.0)
    E = []
    for d in (0.0, 1.0, 2.0, 4.0):
        spec = SP.CamSpec('F', 'F', m, R, np.array([0.0, d, 0.0]))
        fv = RD.conf_view(ts, scene, ev, drive, spec, 0.5, m.fx, cal, 6.0)
        E.append(float(np.nanmean(fv['expected'][fv['valid']])))
    assert E == sorted(E) and E[-1] > E[0] + 5


# ----------------------------------------------------------------------------- a synthetic drive
@pytest.fixture(scope='module')
def drive(tmp_path_factory):
    p = tigor_sample_project()
    v = next(v for v in p.variants if v.sensors)
    p.active_uid = v.uid
    b = Variant.from_dict(v.to_dict())
    b.uid, b.name = 'proposed-b', 'Data rig · cameras 20° up'
    for c in b.cameras:
        c.pitch -= 20.0
    b.by_tag('F').z += 0.2
    p.variants.append(b)
    engine = Engine()
    out = str(tmp_path_factory.mktemp('drive'))
    assert DSET.generate(p, engine, DSET.DatasetSettings.from_dict({'duration_s': 1.0, 'image_scale': 0.15}),
                         out) is not None
    cache = str(tmp_path_factory.mktemp('cache'))
    p.analysis.splat = {'drive': out, 'device': 'cpu', 'max_gaussians': 6000, 'voxel_m': 0.1, 'optimise': False,
                        'train_scale': 0.25, 'holdout_every': 4, 'eval_instants': 1, 'render_scale': 0.12,
                        'colour_step': 3, 'val_frames': 6, 'val_sweeps': 1, 'variant_b': b.uid}
    p.analysis.redrive = {'variant': b.uid, 'calib_views': 12, 'calib_pixels': 600, 'report_instants': 2,
                          'sources': ['holdout'], 'ground_truth': 'always', 'conf_scale': 0.5}
    res = RD.run(p, engine, cache_root=cache)
    assert res is not None
    return {'p': p, 'engine': engine, 'out': out, 'cache': cache, 'b': b, 'res': res,
            'root': str(tmp_path_factory.mktemp('exports'))}


def test_report_shares_sum_and_verdicts(drive):
    res = drive['res']
    rep = res.report
    assert {c['camera'] for c in rep['cameras']} == {c.tag for c in drive['b'].cameras if c.visible}
    for c in rep['cameras'] + [rep['rig']]:
        assert sum(c['reasons'].values()) == pytest.approx(1.0, abs=2e-3)
        assert c['reasons']['ok'] == pytest.approx(c['within_share'], abs=2e-3)
        assert c['verdict'] in ('reuse', 'partial', 're-collect')
    for r in rep['regions']:
        assert sum(r['reasons'].values()) == pytest.approx(1.0, abs=2e-3) and 0 <= r['within_share'] <= 1
        assert r['region'][0] == 'S'
    tot = sum(c['pixels'] for c in rep['cameras'])
    assert rep['rig']['pixels'] == tot
    assert rep['rig']['within_share'] == pytest.approx(
        sum(c['within_share'] * c['pixels'] for c in rep['cameras']) / tot, abs=2e-3)
    inv = rep['inverse']
    assert inv['old_pixels'] > 0 and 0 <= inv['lost_share'] <= 1
    # a verdict says what it is from the shares and the settings
    st = RD.RedriveSettings(error_budget=6, reuse_share=0.8, collect_share=0.5)
    assert RD.verdict(0.84, 'unseen', st) == ('reuse', 'reuse: 84 % of pixels within 6 grey levels')
    assert RD.verdict(0.3, 'unseen', st)[0] == 're-collect' and 'never seen' in RD.verdict(0.3, 'unseen', st)[1]
    # calibration stored with its fit, out-of-fold reliability measured
    cal = RD.Calibration.from_dict(res.calibration)
    assert cal is not None and cal.sources['holdout']['views'] >= 4
    assert res.calibration['reliability_out_of_fold']['pixels'] > 100
    doc = RD.to_doc(res, drive['p'])
    json.dumps(doc)
    assert doc['schema'] == RD.SCHEMA and '_overlays' not in doc['report']
    names = [n for n, _ in RD.files(res, drive['p'])]
    assert names[:3] == ['redrive.json', 'redrive_cameras.csv', 'redrive_regions.csv'] and 'redrive_polar.png' in names
    assert RD.summary(res)['rig_verdict'] == rep['rig']['verdict']


def test_export_round_trip(drive):
    from vantage.core import detection_eval as DE
    from vantage.core import drivelog as DL
    p, b, res = drive['p'], drive['b'], drive['res']
    out = os.path.join(drive['root'], 'reshot')
    st = RD.RedriveSettings.from_dict(dict(RD.settings_of(p).to_dict(), rosbag=True, rolling_shutter='off'))
    doc = RD.export_dataset(p, out, st, b, prepared=res._prepared)
    assert doc is not None and doc['complete'] and RD.is_complete(out) and not os.path.exists(out + '.part')
    assert doc['ground_truth'] and doc['counts']['annotations'] > 0
    # the readers: splat_reproject (the calibration and poses exactly the variant's), the bag, detection_eval, drivelog
    d = SP.open_drive(out)
    src = SP.open_drive(drive['out'])
    scale = src.image_scale
    for c in b.cameras:
        dc = d.cameras[c.tag]
        intr = resolve(p.module_for(c), p.lens_for(c))
        w, h = max(16, round(intr.width * scale)), max(16, round(intr.height * scale))
        ref = SP.CamModel.from_intrinsics(intr, w, h)
        assert (dc.model.width, dc.model.height) == (w, h)
        assert dc.model.fx == pytest.approx(ref.fx, rel=1e-9) and dc.model.cx == pytest.approx(ref.cx, abs=1e-9)
        assert dc.model.kb == pytest.approx(ref.kb) and dc.model.projection == ref.projection
        assert np.allclose(dc.R, rot_zyx(c.yaw, c.pitch, c.roll) @ BODY_TO_OPTICAL.T, atol=1e-8)
        assert np.allclose(dc.t, c.pos(), atol=1e-6)
        assert len(dc.frames) == len(src.cameras[c.tag].frames)
    # ego poses: the drive's
    for t in (0.1, 0.5, 0.9):
        assert np.allclose(d.traj.pose(t)[1] + d.origin, src.traj.pose(t)[1] + src.origin, atol=2e-3)
    # boxes: carried over in the world frame
    assert d.box_source == 'ground truth'
    ver = os.path.join(out, 'nuscenes', 'v1.0-vantage')
    smp = {r['token']: r['timestamp'] for r in json.load(open(os.path.join(ver, 'sample.json')))}
    anns = json.load(open(os.path.join(ver, 'sample_annotation.json')))
    for a in anns[:20]:
        t = (smp[a['sample_token']] - src.t0_us) * 1e-6
        tr = next(tr for i, tr in src.tracks.items()
                  if a['instance_token'] == RD._token('vantage', 'instance', doc['run_key'], i))
        Rb, cb = tr.pose(t)
        assert np.allclose(a['translation'], cb + src.origin, atol=1e-5)
    # the confidence sidecars, one per image
    cidx = json.load(open(os.path.join(out, 'confidence', 'index.json')))
    assert cidx['schema'] == RD.CONF_SCHEMA and sum(len(v) for v in cidx['cameras'].values()) == \
        sum(len(c.frames) for c in d.cameras.values())
    f0 = cidx['cameras']['F'][0]
    E = cv2.imread(os.path.join(out, f0['confidence']), cv2.IMREAD_UNCHANGED)
    assert E.dtype == np.uint16 and (E > 0).any()
    bag = os.path.join(out, doc['rosbag'])
    db = SP.open_drive(bag, os.path.join(drive['root'], 'bagcache'))
    assert set(db.cameras) == set(d.cameras) and db.box_source == 'bag boxes'
    for tag, c in d.cameras.items():
        assert np.allclose(c.R, db.cameras[tag].R, atol=1e-6) and np.allclose(c.t, db.cameras[tag].t, atol=1e-6)
    rec = DE.open_recording(out)
    assert set(rec.cameras) == set(d.cameras) and rec.info()['objects'] > 0
    assert rec.cameras['F'].D is not None and np.allclose(rec.cameras['F'].t, b.by_tag('F').pos(), atol=1e-6)
    nus = os.path.join(out, 'nuscenes')                            # drivelog takes the dataroot, as for a dataset
    assert DL.is_nuscenes(nus)
    info = DL.open_log(nus)
    assert {f'CAM_{c.tag}' for c in b.cameras} <= set(info.topics) and 'LIDAR_L1' in info.topics
    assert DL.scan_log(nus, n_frames=2, sync_sweeps=4) is not None
    # bev's rig perception reads it as a nuScenes folder with ground truth (when that module is present)
    try:
        from vantage.core import rig_perception as RP
    except ImportError:                                            # pragma: no cover — another agent's module
        RP = None
    if RP is not None:
        info = RP.recording_info(out)
        assert info['ok'] and info['instances'] > 0, info
        assert RP.PerceptionRecording(out).cam_gt
        rst = RP.RigPerceptionSettings.from_dict({'recordings': [{'path': out, 'rig': 're-shot'}], 'bootstrap': 20,
                                                  'cache': 'off', 'min_gt': 1})
        rr = RP.run(p, settings=rst, detector=lambda img: {'boxes': np.zeros((0, 4)), 'scores': np.zeros(0),
                                                           'cls': np.zeros(0, int), 'coco': np.zeros(0, int)})
        assert rr is not None and [r['name'] for r in rr.rigs] == ['re-shot']
    # scoring a re-shot drive against a recording of it (here: the source drive itself, a different rig)
    sc = RD.score_reshoot(out, drive['out'])
    assert 'cameras' in sc


def test_cancel_mid_export_leaves_nothing_complete(drive):
    p, b, res = drive['p'], drive['b'], drive['res']
    out = os.path.join(drive['root'], 'cancelled')
    n = {'k': 0}

    def cancel():
        n['k'] += 1
        return n['k'] > 25
    doc = RD.export_dataset(p, out, RD.settings_of(p), b, prepared=res._prepared, cancel=cancel)
    assert doc is None and n['k'] > 25
    assert not os.path.exists(out) and not os.path.exists(out + '.part') and not RD.is_complete(out)
    # a folder of someone else's files is refused
    other = os.path.join(drive['root'], 'other')
    os.makedirs(other)
    open(os.path.join(other, 'notes.txt'), 'w').write('mine')
    with pytest.raises(ValueError):
        RD.export_dataset(p, other, RD.settings_of(p), b, prepared=res._prepared)
    assert os.listdir(other) == ['notes.txt']


def test_nudge_renders_and_reports(drive):
    p, b, res = drive['p'], drive['b'], drive['res']
    c = b.by_tag('F')
    a = RD.nudge_view(p, b, c, res._prepared, 0.5, RD.settings_of(p), 1.0, drive['engine'])
    c2 = RD.nudged_camera(c, dz=0.5, dpitch=-10.0)
    assert c2.z == pytest.approx(c.z + 0.5) and c2.pitch == pytest.approx(c.pitch - 10.0) and c.z != c2.z
    half = RD.nudge_view(p, b, c2, res._prepared, 0.5, RD.settings_of(p), 0.5, drive['engine'])
    assert half['rgb'].shape[1] < a['rgb'].shape[1] and 0.0 <= half['within_share'] <= 1.0
    assert np.isfinite(a['expected'][a['valid']]).all()


def test_no_unseeded_randomness():
    src = open(RD.__file__, encoding='utf-8').read()
    assert 'np.random.rand' not in src and 'np.random.seed' not in src and 'random.random(' not in src


# ----------------------------------------------------------------------------- the page
def test_page_redrive_nudge_apply_undo_and_reshoot(app, win, drive, tmp_path):
    from vantage.ui.workspaces.splat_ws import SplatPage
    p = win.store.project
    src = drive['p']
    v = next(v for v in p.variants if v.sensors)
    p.active_uid = v.uid
    b = Variant.from_dict(drive['b'].to_dict())
    p.variants.append(b)
    pg = SplatPage(win.ctx)
    pg.cache_root = drive['cache']
    pump(app, 0.1)
    assert not pg.b_rd_run.isEnabled()
    assert pg.open_drive(drive['out'])
    pump(app, 0.1)

    def fn(q):
        q.analysis.splat = dict(src.analysis.splat)
        q.analysis.redrive = dict(src.analysis.redrive, variant=b.uid)
    win.store.edit('analysis', fn, 'test settings')
    pump(app, 0.1)
    assert pg.b_rd_run.isEnabled() and pg.tabs.tabText(pg.tabs.count() - 1) == 'Re-drive'
    # the budget typed into its field: undoable
    pg.rd_budget.set_value(9.0)
    pg.rd_budget.committed.emit(9.0)
    pump(app, 0.1)
    assert pg.rd_settings.error_budget == 9.0
    win.store.undo.undo()
    pump(app, 0.1)
    assert pg.rd_settings.error_budget == RD.settings_of(src).error_budget
    assert pg.redrive()
    _wait(app, lambda: not pg.busy, 300)
    assert pg.rd_result is not None, pg.status.text()
    assert pg.rd_table.rowCount() == len([c for c in b.cameras if c.visible])
    assert pg.rd_polar.img is not None and pg.rd_overlay.img is not None
    _wait(app, lambda: pg.nudge_out is not None and not pg._nudge_running, 120)
    assert pg.nudge_view.img is not None and pg.nudge_cam.count() == pg.rd_table.rowCount()
    # a drag renders at a reduced scale, the release at full scale
    pg.nudge_cam.setCurrentIndex(pg.nudge_cam.findText('F'))
    pg.nudge_f['dz'].scrubbed.emit(0.3)
    _wait(app, lambda: pg.nudge_out is not None and pg.nudge_out.get('scale') == 0.5 and not pg._nudge_running, 120)
    pg.nudge_f['dz'].set_value(0.3)
    pg.nudge_f['dz'].committed.emit(0.3)
    pg.nudge_f['dpitch'].set_value(-5.0)
    pg.nudge_f['dpitch'].committed.emit(-5.0)
    _wait(app, lambda: pg.nudge_out is not None and pg.nudge_out.get('scale') == 1.0 and not pg._nudge_running
          and abs(pg.nudge_out['camera'].z - (b.by_tag('F').z + 0.3)) < 1e-9, 120)
    z0, pitch0 = b.by_tag('F').z, b.by_tag('F').pitch
    n_undo = win.store.undo.index()
    assert pg.apply_nudge()
    pump(app, 0.1)
    vb = p.variant_by_uid(b.uid)
    assert vb.by_tag('F').z == pytest.approx(z0 + 0.3) and vb.by_tag('F').pitch == pytest.approx(pitch0 - 5.0)
    assert win.store.undo.index() == n_undo + 1 and pg.nudge_values()['dz'] == 0.0      # one step
    win.store.undo.undo()
    pump(app, 0.1)
    vb = p.variant_by_uid(b.uid)
    assert vb.by_tag('F').z == pytest.approx(z0) and vb.by_tag('F').pitch == pytest.approx(pitch0)
    # the re-shot export from the page
    out = str(tmp_path / 'reshot')
    assert pg.reshoot(out)
    _wait(app, lambda: not pg.busy, 300)
    assert pg.rd_export is not None and RD.is_complete(out), pg.status.text()
    pg.cancel()
    pg.close()
