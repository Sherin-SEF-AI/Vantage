"""Detection quality per rig (core.detection, core.detection_eval, the page): the class mapping, YOLOX's decoding,
COCO's matching and average precision against hand-computed cases and a reference implementation (and pycocotools
when it is installed), the block bootstrap's coverage, the fisheye ↔ pinhole-view geometry against cv2.fisheye, a
synthetic recording to the ground-truth contract (vantage.ground_truth/1) evaluated end to end with a detector whose
hits and misses are known, two recordings compared (paired), the detection cache, the model download's checksum, a
variant driven through the bridge against the fake carla and evaluated, and the page.

Tests marked real_model run only when the published model file is present: VANTAGE_DETECT_MODELS=<folder holding
yolox_s.onnx> (they are skipped otherwise)."""
import http.server
import json
import math
import os
import threading

import numpy as np
import pytest

from vantage.core import detection as DT
from vantage.core import detection_eval as DE

REAL_DIR = os.environ.get('VANTAGE_DETECT_MODELS', '')
real_model = pytest.mark.skipif(not (REAL_DIR and os.path.isfile(os.path.join(REAL_DIR, 'yolox_s.onnx'))),
                                reason='the real model is not present (VANTAGE_DETECT_MODELS)')


@pytest.fixture(autouse=True)
def _models_dir(tmp_path, monkeypatch):
    """Never the user's data folder: every test gets its own models folder."""
    monkeypatch.setenv(DT.ENV_DIR, str(tmp_path / 'models'))


# ======================================================================= a synthetic recording to the GT contract
BG = (100, 100, 100)
HIDDEN = (92, 92, 92)               # an object drawn in this colour is never detected (a miss)
# colour -> (Vantage class, score) of the colour detector
PALETTE = {(0, 0, 255): ('pedestrian', 0.9), (255, 0, 0): ('car', 0.8), (0, 255, 0): ('truck', 0.7),
           (0, 255, 255): ('cyclist', 0.85), (200, 0, 0): ('car', 0.6)}
COCO_OF = {'pedestrian': 0, 'cyclist': 1, 'car': 2, 'motorcyclist': 3, 'bus': 5, 'truck': 7}


class ColourDetector:
    """A detector that finds blobs of the palette's colours (exact colours on PNG images; within 40 levels after a
    bilinear remap): each connected blob is one detection of its colour's class and score."""

    def __init__(self, tol=40):
        self.tol = tol
        self.calls = 0

    def __call__(self, img):
        import cv2
        self.calls += 1
        boxes, scores, cls, coco = [], [], [], []
        im = img.astype(np.int16)
        for col, (name, sc) in PALETTE.items():
            m = np.all(np.abs(im - np.array(col)) <= self.tol, axis=2).astype(np.uint8)
            if not m.any():
                continue
            n, _lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
            for k in range(1, n):
                x, y, w, h, area = stats[k]
                if area < 4:
                    continue
                boxes.append([x, y, x + w, y + h])
                scores.append(sc)
                cls.append(DT.CLASSES.index(name))
                coco.append(COCO_OF[name])
        return {'boxes': np.array(boxes, np.float32).reshape(-1, 4), 'scores': np.array(scores, np.float32),
                'cls': np.array(cls, np.int8), 'coco': np.array(coco, np.int16), 'ms': 1.0}


def _quat(R):
    R = np.asarray(R, float)
    w = math.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = math.copysign(math.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2, R[2, 1] - R[1, 2])
    y = math.copysign(math.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2, R[0, 2] - R[2, 0])
    z = math.copysign(math.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2, R[1, 0] - R[0, 1])
    return [w, x, y, z]


FRONT = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])     # optical -> vehicle, looking forward
K_FE = [[100.0, 0.0, 159.5], [0.0, 100.0, 119.5], [0.0, 0.0, 1.0]]
D_FE = [0.01, -0.002, 0.0, 0.0]


def make_recording(root, objects_fn, n_frames=20, rate_hz=10.0, cams=('F',), variant='Rig A', scenario='free_drive',
                   seed=1, size=(320, 240), t0_us=1_767_268_800_000_000):
    """A drive folder to the contract: ground_truth/index.json, PNG images and label images, nuScenes calibration
    (a Kannala-Brandt fisheye looking forward), run.json and dataset.json. objects_fn(cam, k) -> [dict(id, cls (GT
    class name), box [x0, y0, x1, y1], range, colour, xy, actor)] (colour None: drawn hidden; id 0: a decoy drawn but
    not labelled)."""
    import cv2
    W, H = size
    os.makedirs(os.path.join(root, 'nuscenes', 'v1.0-vantage'), exist_ok=True)
    doc = {'schema': 'vantage.ground_truth/1', 'classes': [], 'cameras': {}, 'lidars': {}, 'notes': []}
    sens, cal = [], []
    for tag in cams:
        ch = f'CAM_{tag}'
        frames = []
        d_img = os.path.join(root, 'nuscenes', 'samples', ch)
        d_gt = os.path.join(root, 'ground_truth', ch)
        os.makedirs(d_img, exist_ok=True)
        os.makedirs(d_gt, exist_ok=True)
        for k in range(n_frames):
            t = t0_us + int(round(k * 1e6 / rate_hz))
            img = np.full((H, W, 3), BG, np.uint8)
            inst = np.zeros((H, W), np.uint16)
            rng = np.full((H, W), 50.0 * 256, np.float64)
            objs = []
            for o in objects_fn(tag, k):
                x0, y0, x1, y1 = o['box']
                col = o.get('colour') or HIDDEN
                img[y0:y1, x0:x1] = col
                rng[y0:y1, x0:x1] = o['range'] * 256
                if not o.get('id'):
                    continue
                inst[y0:y1, x0:x1] = o['id']
                xy = o.get('xy', (o['range'], 0.0))
                objs.append({'id': o['id'], 'actor_id': o.get('actor', 1000 + o['id']), 'instance_token': None,
                             'category': 'vehicle.car', 'class': o['cls'], 'bbox': [x0, y0, x1, y1],
                             'pixels': (x1 - x0) * (y1 - y0), 'bbox_full': [x0, y0, x1, y1],
                             'truncated': o.get('trunc', 0.0), 'occluded': o.get('occ', 0.0), 'range_m': o['range'],
                             'box3d': {'center': [xy[0], xy[1], 0.8], 'size': [4.0, 1.8, 1.5], 'yaw_deg': 0.0}})
            name = f'drive__{ch}__{t}'
            rel_img = f'nuscenes/samples/{ch}/{name}.png'
            cv2.imwrite(os.path.join(root, rel_img), img)
            cv2.imwrite(os.path.join(d_gt, f'{name}.instance.png'), inst)
            cv2.imwrite(os.path.join(d_gt, f'{name}.range.png'), np.clip(np.rint(rng), 0, 65535).astype(np.uint16))
            cv2.imwrite(os.path.join(d_gt, f'{name}.semantic.png'), np.zeros((H, W), np.uint8))
            frames.append({'sample_data_token': f'{tag}{k}', 'timestamp': t, 'image': rel_img,
                           'semantic': f'ground_truth/{ch}/{name}.semantic.png',
                           'instance': f'ground_truth/{ch}/{name}.instance.png',
                           'range': f'ground_truth/{ch}/{name}.range.png', 'objects': objs})
        doc['cameras'][tag] = {'channel': ch, 'width': W, 'height': H, 'frames': frames}
        sens.append({'token': f's{tag}', 'channel': ch, 'modality': 'camera'})
        cal.append({'token': f'c{tag}', 'sensor_token': f's{tag}', 'translation': [2.0, 0.0, 1.2],
                    'rotation': _quat(FRONT), 'camera_intrinsic': K_FE,
                    'vantage_distortion': {'model': 'kannala_brandt', 'coefficients': D_FE, 'image_size': [W, H]}})
    with open(os.path.join(root, 'ground_truth', 'index.json'), 'w') as fh:
        json.dump(doc, fh)
    with open(os.path.join(root, 'nuscenes', 'v1.0-vantage', 'sensor.json'), 'w') as fh:
        json.dump(sens, fh)
    with open(os.path.join(root, 'nuscenes', 'v1.0-vantage', 'calibrated_sensor.json'), 'w') as fh:
        json.dump(cal, fh)
    run = {'schema': 'vantage.sim_run/1', 'status': 'ok', 'project': 'Test', 'variant': variant,
           'scenario': {'name': scenario}, 'scene': {'town': 'Town10HD', 'seed': seed, 'weather': 'ClearNoon',
                                                    'duration_s': n_frames / rate_hz}}
    with open(os.path.join(root, 'run.json'), 'w') as fh:
        json.dump(run, fh)
    with open(os.path.join(root, 'dataset.json'), 'w') as fh:
        json.dump({'schema': 'vantage.dataset/1', 'project': 'Test', 'variant': variant}, fh)
    return root


def scene_a(detect_far_car=False):
    """Per frame: car A at 8 m (found), pedestrian at 15 m (found), car B at 25 m (found only with detect_far_car),
    a truck at 45 m (found in even frames), a decoy car at 15 m in frames 0–4 (a false detection at score 0.6), a
    pedestrian 6 px tall at 30 m (ignored, and found: its detection is ignored too)."""
    def fn(cam, k):
        return [
            {'id': 1, 'cls': 'car', 'box': [20, 120, 80, 160], 'range': 8.0, 'colour': (255, 0, 0), 'xy': (8.0, 0.0),
             'actor': 11},
            {'id': 2, 'cls': 'pedestrian', 'box': [100, 90, 112, 120], 'range': 15.0, 'colour': (0, 0, 255),
             'xy': (15.0, 1.0), 'actor': 12},
            {'id': 3, 'cls': 'car', 'box': [150, 100, 170, 114], 'range': 25.0,
             'colour': (255, 0, 0) if detect_far_car else None, 'xy': (25.0, -2.0), 'actor': 13},
            {'id': 4, 'cls': 'truck', 'box': [200, 95, 230, 115], 'range': 45.0,
             'colour': (0, 255, 0) if k % 2 == 0 else None, 'xy': (45.0, 3.0), 'actor': 14},
            {'id': 5, 'cls': 'pedestrian', 'box': [250, 100, 254, 106], 'range': 30.0, 'colour': (0, 0, 255),
             'xy': (30.0, 0.0), 'actor': 15},
        ] + ([{'id': 0, 'cls': 'car', 'box': [260, 160, 290, 180], 'range': 15.0, 'colour': (200, 0, 0)}]
             if k < 5 else [])
    return fn


def _settings(paths, **kw):
    d = {'recordings': [str(p) for p in paths], 'bootstrap': 300, 'block_s': 0.5, 'cache': False}
    d.update(kw)
    return DE.DetectionSettings.from_dict(d)


# ======================================================================= classes and settings
def test_class_mapping_covers_coco_and_the_ground_truth():
    from vantage.core import groundtruth as GT
    assert set(DT.COCO_TO_CLASS.values()) == set(DT.CLASSES)
    assert all(n in DT.COCO_CLASSES for n in DT.COCO_TO_CLASS)
    assert DT.COCO_TO_CLASS['person'] == 'pedestrian' and DT.COCO_TO_CLASS['bicycle'] == 'cyclist'
    assert DT.COCO_TO_CLASS['stop sign'] == 'stop_sign' and DT.COCO_TO_CLASS['traffic light'] == 'traffic_light'
    for i, c in zip(DT.MAPPED_COCO, DT.MAPPED_CLASS):
        assert DT.COCO_TO_CLASS[DT.COCO_CLASSES[i]] == DT.CLASSES[c] and DT.CLASS_OF_COCO[i] == c
    assert (DT.CLASS_OF_COCO >= 0).sum() == len(DT.COCO_TO_CLASS)
    # every ground-truth class the evaluation uses is in core.groundtruth's class table
    names = {c[1] for c in GT.CLASSES}
    assert set(DT.GT_TO_CLASS) <= names
    assert DT.GT_TO_CLASS['rider'] == DT.GT_TO_CLASS['bicycle'] == 'cyclist'
    assert DT.GT_TO_CLASS['motorcycle'] == 'motorcyclist'
    assert DT.GROUPS['vehicle'] == ('car', 'truck', 'bus') and DT.EVAL_CLASSES[-1] == 'vehicle'
    assert len(DT.COCO_IDS) == len(DT.COCO_CLASSES) == 80 and DT.COCO_IDS[0] == 1 and DT.COCO_IDS[-1] == 90


def test_settings_are_tolerant_and_clamped():
    s = DE.DetectionSettings.from_dict({'score': 7, 'iou': 'x', 'model': 'nope', 'backend': 'cuda', 'input_px': 650,
                                        'range_edges': '20, 5, junk, 40', 'view_yaws': [0, 400],
                                        'classes': ['car', 'x'],
                                        'recordings': ['/a', '/a', 3, '/b'], 'bootstrap': -5, 'views': 'pinhole',
                                        'cache': 'no', 'fuse_riders': 0, 'min_pixels': 2.6, 'confidence': float('nan')})
    assert s.score == 0.99 and s.iou == 0.5 and s.model == 'yolox_s' and s.backend == 'cpu'
    assert s.input_px == 640 and s.range_edges == [5.0, 20.0, 40.0] and s.view_yaws == [0.0, 180.0]
    assert s.classes == ['car'] and s.recordings == ['/a', '/b'] and s.bootstrap == 0 and s.views == 'pinhole'
    assert s.cache is False and s.fuse_riders is False and s.min_pixels == 3 and s.confidence == 0.95
    assert DE.DetectionSettings.from_dict(None) == DE.DetectionSettings()
    assert DE.DetectionSettings.from_dict(DE.DetectionSettings().to_dict()) == DE.DetectionSettings()
    d = s.detector()
    assert isinstance(d, DT.DetectorSettings) and d.input_px == 640 and not d.fuse_riders
    assert DE.bin_labels([10, 20, 40]) == ['0–10 m', '10–20 m', '20–40 m', '40+ m']
    assert list(DE.bin_of([0, 9.99, 10, 39, 40, 1e3, float('nan')], [10, 20, 40])) == [0, 0, 1, 2, 3, 3, -1]


# ======================================================================= YOLOX decoding and post-processing
def test_letterbox_and_input_size():
    st = DT.DetectorSettings()
    assert DT.input_size(960, 768, st) == (960, 768)
    assert DT.input_size(1920, 1536, st) == (1280, 1024)
    assert DT.input_size(300, 200, st) == (320, 224)
    assert DT.input_size(300, 200, DT.DetectorSettings(input_px=640)) == (640, 640)
    img = np.zeros((100, 200, 3), np.uint8)
    img[:] = (10, 20, 30)
    blob, r = DT.letterbox(img, 320, 320)
    assert blob.shape == (1, 3, 320, 320) and r == 1.6
    assert blob[0, :, 0, 0].tolist() == [10, 20, 30] and blob[0, 0, 200, 0] == 114 and blob[0, 0, 159, 319] == 10


def test_decode_postprocess_and_nms():
    W, H = 64, 64                      # 8×8 + 4×4 + 2×2 anchors at strides 8, 16, 32
    n = 64 + 16 + 4
    raw = np.zeros((1, n, 85), np.float32)
    raw[..., 4] = 0.0
    # anchor (3, 2) of stride 8 (index 2·8 + 3 = 19): a car centred at (3 + 0.5, 2 + 0.25)·8, 16 × 8 px
    raw[0, 19, :4] = [0.5, 0.25, math.log(2.0), math.log(1.0)]
    raw[0, 19, 4] = 0.9
    raw[0, 19, 5 + 2] = 0.8                                        # COCO 'car'
    # anchor 20 (its neighbour): the same car slightly weaker -> suppressed by NMS
    raw[0, 20, :4] = [-0.5, 0.25, math.log(2.0), math.log(1.0)]
    raw[0, 20, 4] = 0.9
    raw[0, 20, 5 + 2] = 0.5
    # the same box also as a truck (another class: kept)
    raw[0, 21, :4] = [-1.5, 0.25, math.log(2.0), math.log(1.0)]
    raw[0, 21, 4] = 0.5
    raw[0, 21, 5 + 7] = 0.5
    # a dog (unmapped): never kept
    raw[0, 30, 4] = 0.99
    raw[0, 30, 5 + 16] = 0.99
    boxes, obj, probs = DT.decode(raw, W, H)
    assert np.allclose(boxes[19], [20.0, 14.0, 36.0, 22.0])
    st = DT.DetectorSettings(keep_score=0.1)
    d = DT.postprocess(boxes, obj, probs, 2.0, 100, 100, st)
    names = [DT.CLASSES[c] for c in d['cls']]
    assert names == ['car', 'truck'] and np.allclose(d['scores'], [0.72, 0.25])
    assert np.allclose(d['boxes'][0], [10.0, 7.0, 18.0, 11.0])        # divided by r = 2
    assert [DT.COCO_CLASSES[c] for c in d['coco']] == ['car', 'truck']
    with pytest.raises(ValueError):
        DT.decode(raw[:, :10], W, H)


def test_rider_fusion():
    d = {'boxes': np.array([[100, 100, 140, 200], [95, 150, 150, 210], [300, 100, 320, 160]], np.float32),
         'scores': np.array([0.9, 0.7, 0.8], np.float32), 'coco': np.array([0, 1, 0], np.int16),
         'cls': np.array([0, 1, 0], np.int8)}
    f = DT.fuse_riders(d)
    names = [DT.CLASSES[c] for c in f['cls']]
    assert sorted(names) == ['cyclist', 'pedestrian']
    cyc = f['boxes'][names.index('cyclist')]
    assert np.allclose(cyc, [95, 100, 150, 210]) and f['scores'][names.index('cyclist')] == pytest.approx(0.7)
    # a person beside the bike (its centre outside the bike's width) stays a pedestrian
    d2 = dict(d, boxes=np.array([[160, 100, 200, 200], [95, 150, 150, 210], [300, 100, 320, 160]], np.float32))
    assert sorted(DT.CLASSES[c] for c in DT.fuse_riders(d2)['cls']) == ['cyclist', 'pedestrian', 'pedestrian']


# ======================================================================= matching and average precision
def test_match_hand_computed():
    gt = [[0, 0, 10, 10], [20, 0, 30, 10], [40, 0, 44, 4]]
    ign = [False, False, True]
    dets = [[0, 0, 10, 10], [1, 0, 11, 10], [20, 0, 30, 10], [40, 0, 44, 4], [60, 0, 70, 10], [80, 0, 82, 2]]
    sc = [0.9, 0.95, 0.5, 0.6, 0.4, 0.3]
    small = [False, False, False, False, False, True]
    status, dg, gs = DE.match(gt, ign, 0.5, dets, sc, small)
    # det 1 (0.95, IoU 0.82 with gt 0) takes gt 0; det 0 (0.9) then has no unmatched object: false positive
    assert status.tolist() == [0, 1, 1, -1, 0, -1]
    assert dg.tolist() == [-1, 0, 1, 2, -1, -1] and gs.tolist() == [0.95, 0.5, 0.6]
    # an IoU just under the threshold misses; a per-object threshold (small objects) can lower it
    iou = DE.iou_matrix([[0, 0, 10, 10]], [[3, 0, 13, 10]])[0, 0]
    assert iou == pytest.approx(70 / 130)
    assert DE.match([[0, 0, 10, 10]], [False], 0.6, [[3, 0, 13, 10]], [1.0])[0].tolist() == [0]
    assert DE.match([[0, 0, 10, 10]], [False], [0.5], [[3, 0, 13, 10]], [1.0])[0].tolist() == [1]
    # an object to ignore is taken only when no other object is left at the threshold
    st_, dg_, _ = DE.match([[0, 0, 10, 10], [0, 0, 10, 11]], [True, False], 0.5, [[0, 0, 10, 10]], [1.0])
    assert st_.tolist() == [1] and dg_.tolist() == [1]
    # nothing to match
    assert DE.match([], [], 0.5, [[0, 0, 1, 1]], [0.5])[0].tolist() == [0]
    assert DE.match([[0, 0, 1, 1]], [False], 0.5, [], [])[2].tolist() == [-np.inf]


def test_average_precision_hand_computed_and_vectorised():
    # TP, FP, TP over 2 objects: precision 1, 1/2, 2/3, recall 1/2, 1/2, 1; envelope 1, 2/3, 2/3:
    # thresholds 0 … 0.50 (51) read 1, 0.51 … 1 (50) read 2/3
    ap = DE.ap_coco([0.9, 0.8, 0.7], [1, 0, 1], 2)
    assert ap == pytest.approx((51 * 1.0 + 50 * 2 / 3) / 101)
    assert DE._ap_weighted([1, 0, 1], np.ones((1, 3)), [2])[0] == pytest.approx(ap)
    assert DE.ap_coco([0.9], [0], 3) == 0.0 and DE.ap_coco([], [], 0) is None
    assert DE.ap_coco([0.9, 0.8], [1, 1], 2) == pytest.approx(1.0)
    rng = np.random.default_rng(5)
    for _ in range(200):
        n = int(rng.integers(1, 40))
        sc = rng.random(n)
        tp = (rng.random(n) < 0.6).astype(float)
        ngt = int(tp.sum() + rng.integers(0, 5)) or 1
        ref = DE.ap_coco(sc, tp, ngt)
        o = np.argsort(-sc, kind='mergesort')
        assert DE._ap_weighted(tp[o], np.ones((1, n)), [ngt])[0] == pytest.approx(ref, abs=1e-12)
        # a weight of 2 on every detection and every object is the same set twice over
        w = rng.integers(0, 3, n).astype(float)
        rep = np.repeat(np.arange(n), w.astype(int))
        if rep.size:
            ref2 = DE.ap_coco(np.repeat(sc[o], w[o].astype(int)), np.repeat(tp[o], w[o].astype(int)), 2 * ngt)
            assert DE._ap_weighted(tp[o], w[o][None], [2 * ngt])[0] == pytest.approx(ref2, abs=1e-12)
    # chunking gives the same rows
    W = rng.integers(0, 3, (50, 30)).astype(float)
    tp = (rng.random(30) < 0.5).astype(float)
    a = DE._ap_weighted(tp, W, W.sum(1) * 0.7 + 1)
    b = DE._ap_weighted(tp, W, W.sum(1) * 0.7 + 1, chunk_elems=90)
    assert np.allclose(a, b, equal_nan=True)


def test_average_precision_matches_pycocotools():
    pc = pytest.importorskip('pycocotools.cocoeval')
    from pycocotools.coco import COCO
    rng = np.random.default_rng(3)
    images, anns, dets = [], [], []
    aid = 1
    all_gt, all_dt = {}, {}
    for i in range(1, 31):
        images.append({'id': i, 'width': 640, 'height': 480})
        g = []
        for _ in range(int(rng.integers(0, 5))):
            x, y = rng.uniform(0, 500), rng.uniform(0, 380)
            w, h = rng.uniform(40, 120), rng.uniform(40, 100)
            anns.append({'id': aid, 'image_id': i, 'category_id': 3, 'bbox': [x, y, w, h], 'area': w * h,
                         'iscrowd': 0})
            aid += 1
            g.append([x, y, x + w, y + h])
        d = []
        for b in g:
            if rng.random() < 0.8:
                j = rng.normal(0, 8, 4)
                d.append(([b[0] + j[0], b[1] + j[1], b[2] + j[2], b[3] + j[3]], float(rng.random())))
        for _ in range(int(rng.integers(0, 3))):
            x, y = rng.uniform(0, 500), rng.uniform(0, 380)
            d.append(([x, y, x + 60, y + 50], float(rng.random() * 0.6)))
        for bb, s in d:
            dets.append({'image_id': i, 'category_id': 3, 'bbox': [bb[0], bb[1], bb[2] - bb[0], bb[3] - bb[1]],
                         'score': s})
        all_gt[i], all_dt[i] = g, d
    coco = COCO()
    coco.dataset = {'images': images, 'annotations': anns, 'categories': [{'id': 3, 'name': 'car'}]}
    coco.createIndex()
    cd = coco.loadRes(dets)
    E = pc.COCOeval(coco, cd, 'bbox')
    E.params.areaRng = [[0, 1e10]]
    E.params.areaRngLbl = ['all']
    E.params.maxDets = [100]
    E.evaluate()
    E.accumulate()
    ref50 = float(E.eval['precision'][0, :, 0, 0, 0].mean())
    ref = float(E.eval['precision'][:, :, 0, 0, 0].mean())
    for thr, want in ((0.5, ref50), (None, ref)):
        aps = []
        for t in ([thr] if thr else DE.AP_IOUS):
            sc, tp, n = [], [], 0
            for i in all_gt:
                g, d = all_gt[i], all_dt[i]
                st_, _, _ = DE.match(g, [False] * len(g), t, [b for b, _ in d], [s for _, s in d])
                sc += [s for _, s in d]
                tp += st_.tolist()
                n += len(g)
            aps.append(DE.ap_coco(sc, tp, n))
        assert np.mean(aps) == pytest.approx(want, abs=1e-6)


def test_block_bootstrap_coverage():
    """95 % percentile intervals of a share from independent blocks cover the true value about 95 % of the time."""
    rng = np.random.default_rng(11)
    p, nb, hit = 0.35, 60, 0
    reps = 300
    for k in range(reps):
        per = rng.binomial(8, p, nb).astype(float)
        W = DE.boot_weights(nb, 400, k)
        val, boot = DE._ratio(per, np.full(nb, 8.0), W)
        ci = DE._pct(boot, 0.95)
        hit += ci[0] <= p <= ci[1]
    assert 0.90 <= hit / reps <= 0.99
    W = DE.boot_weights(7, 50, 0)
    assert W.shape == (50, 7) and np.all(W.sum(1) == 7)


# ======================================================================= fisheye and pinhole views
def _cam(w=320, h=240):
    return DE.Camera('F', 'CAM_F', w, h, [], np.array(K_FE), np.array(D_FE), 'kb', FRONT.copy(),
                     np.array([2.0, 0.0, 1.2]))


def test_kb_projection_matches_cv2_fisheye_and_inverts():
    import cv2
    cam = _cam()
    rng = np.random.default_rng(2)
    th = rng.uniform(0, math.radians(80), 300)
    ph = rng.uniform(-math.pi, math.pi, 300)
    P = np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], 1)
    u, v, ok = DE.kb_project(cam, P)
    ref, _ = cv2.fisheye.projectPoints(P[:, None, :].astype(np.float64), np.zeros(3), np.zeros(3), np.array(K_FE),
                                       np.array(D_FE))
    assert ok.all() and np.allclose(np.c_[u, v], ref[:, 0], atol=1e-6)
    d, ok2 = DE.kb_unproject(cam, u, v)
    assert ok2.all() and np.allclose(d, P, atol=1e-7)


def test_pinhole_views_geometry():
    import cv2
    cam = _cam()
    st = DE.DetectionSettings.from_dict({'view_yaws': [0, 40], 'view_fov_deg': 90, 'view_pitch_deg': 0,
                                         'view_width': 200, 'view_aspect': 0.75})
    vs = DE.make_views(cam, st)
    assert [v.yaw for v in vs] == [0.0, 40.0] and vs[0].w == 200 and vs[0].h == 150
    # the centre of the straight-ahead view reads the fisheye's principal point
    v0 = vs[0]
    assert v0.map_x[75, 100] == pytest.approx(159.5 + 0.5 * 0, abs=0.6) and v0.map_y[75, 100] == pytest.approx(119.5,
                                                                                                             abs=0.6)
    # a vehicle-frame point lands where both models put it: through the view's map back into the fisheye
    for v in vs:
        for p_v in ([10.0, 2.0, 1.0], [6.0, 5.5, 0.3], [8.0, -1.0, 2.0]):
            d_cam = FRONT.T @ (np.array(p_v) - np.array([2.0, 0.0, 1.2]))
            u, w_, ok = DE.kb_project(cam, d_cam[None])
            d_view = v.R.T @ d_cam
            if d_view[2] <= 0:
                continue
            pu = v.K[0, 0] * d_view[0] / d_view[2] + v.K[0, 2]
            pv = v.K[1, 1] * d_view[1] / d_view[2] + v.K[1, 2]
            if not (0 <= pu < v.w - 1 and 0 <= pv < v.h - 1):
                continue
            mx = cv2.remap(v.map_x, np.array([[pu]], np.float32), np.array([[pv]], np.float32), cv2.INTER_LINEAR)
            my = cv2.remap(v.map_y, np.array([[pu]], np.float32), np.array([[pv]], np.float32), cv2.INTER_LINEAR)
            assert float(mx[0, 0]) == pytest.approx(float(u[0]), abs=0.05)
            assert float(my[0, 0]) == pytest.approx(float(w_[0]), abs=0.05)
    # the views are level: the straight-ahead view's horizon row is horizontal (pitch 0 -> its centre row)
    up = DE.kb_unproject(cam, v0.map_x[74, [20, 180]], v0.map_y[74, [20, 180]])[0] @ FRONT.T
    dn = DE.kb_unproject(cam, v0.map_x[75, [20, 180]], v0.map_y[75, [20, 180]])[0] @ FRONT.T
    assert np.allclose(up[:, 2] + dn[:, 2], 0.0, atol=2e-4) and np.allclose(up[:, 2], up[0, 2], atol=1e-4)
    # what lies inside a view, as fisheye pixels
    assert v0.inside[120, 160] and not v0.inside[0, 0]
    ids, boxes, counts = DE.boxes_of_ids(np.array([[0, 2, 2], [0, 0, 2], [5, 0, 0]], np.uint16))
    assert ids.tolist() == [2, 5] and boxes.tolist() == [[1, 0, 3, 2], [0, 2, 1, 3]] and counts.tolist() == [3, 1]


# ======================================================================= end to end on synthetic recordings
@pytest.fixture(scope='module')
def recs(tmp_path_factory):
    root = tmp_path_factory.mktemp('det_recs')
    a = make_recording(str(root / 'a'), scene_a(False), variant='Rig A')
    b = make_recording(str(root / 'b'), scene_a(True), variant='Rig B')
    c = make_recording(str(root / 'c'), scene_a(True), variant='Rig C', seed=2)
    return {'a': a, 'b': b, 'c': c}


def test_recording_reads_to_the_contract(recs):
    r = DE.open_recording(recs['a'])
    assert r.label == 'Rig A · free drive · seed 1' and r.scenario == 'free_drive' and r.seed == 1
    cam = r.cameras['F']
    assert cam.lens == 'kb' and np.allclose(cam.R, FRONT) and np.allclose(cam.D, D_FE) and len(cam.frames) == 20
    info = DE.recording_info(recs['a'])
    assert info['ok'] and info['frames'] == 20 and info['objects'] == 100
    bad = DE.recording_info(os.path.dirname(recs['a']))
    assert not bad['ok'] and 'no ground truth' in bad['text']
    assert DE.open_recording(recs['a']).same_scene(DE.open_recording(recs['b']))
    assert not DE.open_recording(recs['a']).same_scene(DE.open_recording(recs['c']))


def test_end_to_end_known_answers(recs):
    det = ColourDetector()
    res = DE.run(None, settings=_settings([recs['a']]), detector=det)
    assert det.calls == 20
    car = res.row(0, 'fisheye', 'all', 'car', 'all')
    assert car['n_gt'] == 40 and car['detected'] == 20 and car['recall'] == pytest.approx(0.5)
    assert car['tp'] == 20 and car['fp'] == 5 and car['precision'] == pytest.approx(0.8)
    assert car['ap50'] == pytest.approx(51 / 101)
    assert car['recall_ci'][0] <= 0.5 <= car['recall_ci'][1]
    ped = res.row(0, 'fisheye', 'F', 'pedestrian', 'all')
    assert ped['n_gt'] == 20 and ped['recall'] == 1.0 and ped['fp'] == 0 and ped['precision'] == 1.0
    truck = res.row(0, 'fisheye', 'all', 'truck', 'all')
    assert truck['n_gt'] == 20 and truck['recall'] == pytest.approx(0.5)
    veh = res.row(0, 'fisheye', 'all', 'vehicle', 'all')
    assert veh['n_gt'] == 60 and veh['detected'] == 30 and veh['precision'] == pytest.approx(30 / 35)
    # per range bin: the decoys are ranged by the range image (15 m)
    assert res.row(0, 'fisheye', 'all', 'car', '0–10 m')['recall'] == 1.0
    assert res.row(0, 'fisheye', 'all', 'car', '20–40 m')['recall'] == 0.0
    c1020 = res.row(0, 'fisheye', 'all', 'car', '10–20 m')
    assert c1020['n_gt'] == 0 and c1020['fp'] == 5 and c1020['precision'] == 0.0
    # detection range: the farthest bin with recall ≥ 0.5
    rg = {(r['camera'], r['class']): r for r in res.ranges}
    assert rg[('all', 'car')]['range_m'] == 10.0 and rg[('all', 'truck')]['beyond_last_edge']
    assert rg[('all', 'pedestrian')]['range_m'] == 20.0
    # recall against height: the tiny pedestrian is counted there (ignored for size, found)
    hp = next(h for h in res.heights if h['class'] == 'pedestrian')
    by = {b['bin']: b for b in hp['bins']}
    assert by['< 16 px']['n'] == 20 and by['16–32 px']['recall'] == 1.0
    hc = next(h for h in res.heights if h['class'] == 'car')
    hb = {b['bin']: b for b in hc['bins']}
    assert hb['< 16 px']['recall'] == 0.0 and hb['32–64 px']['recall'] == 1.0
    assert 16 <= hc['h50_px'] <= 64
    # the rig: one camera, so the rig's recall is the camera's, by distance from the vehicle
    rig = {(r['class'], r['distance_bin']): r for r in res.rig}
    assert rig[('car', 'all')]['n'] == 40 and rig[('car', 'all')]['recall'] == pytest.approx(0.5)
    assert rig[('truck', '40+ m')]['recall'] == pytest.approx(0.5)
    # COCO's AP over IoU 0.5:0.95 equals AP50 for exact boxes
    assert car['ap'] == pytest.approx(car['ap50'])
    # overlays: the frame with the most objects, boxes coloured by outcome
    k = DE.best_frame(res, 0, 'fisheye', 'F')
    img = DE.overlay_image(res, 0, 'fisheye', 'F', k)
    assert img.shape == (240, 320, 3) and np.any(np.all(img == DE._COL['miss'], axis=2))
    s = DE.summary(res)
    json.dumps(s)
    assert s['recordings'][0]['classes'][0]['class'] == 'pedestrian'


def test_comparison_paired_and_unpaired(recs, tmp_path):
    res = DE.run(None, settings=_settings([recs['a'], recs['b'], recs['c']]), detector=ColourDetector())
    comp = {(c['recording'], c['mode'], c['class'], c['range_bin']): c for c in res.comparisons}
    lb, lc = res.recordings[1]['label'], res.recordings[2]['label']
    c = comp[(lb, 'fisheye', 'car', 'all')]
    assert c['paired'] and c['recall_reference'] == pytest.approx(0.5) and c['recall'] == pytest.approx(1.0)
    assert c['recall_difference'] == pytest.approx(0.5) and c['recall_difference_ci'][0] > 0
    assert c['precision_difference'] == pytest.approx(40 / 45 - 0.8)
    r = comp[(lb, 'rig', 'car', 'all')]
    assert r['paired'] and r['n_gt'] == 40 and r['recall_difference'] == pytest.approx(0.5)
    assert not comp[(lc, 'fisheye', 'car', 'all')]['paired']
    assert comp[(lc, 'rig', 'car', 'all')]['recall_difference'] == pytest.approx(0.5)
    assert any('not paired' in n for n in res.notes)
    # differences of the same figures with themselves are exactly 0 with a 0-wide interval when paired
    import shutil
    twin = str(tmp_path / 'a2')
    shutil.copytree(recs['a'], twin)
    assert DE.DetectionSettings.from_dict({'recordings': [recs['a'], recs['a']]}).recordings == [recs['a']]
    same = DE.run(None, settings=_settings([recs['a'], twin]), detector=ColourDetector())
    c0 = next(c for c in same.comparisons if c['class'] == 'vehicle' and c['range_bin'] == 'all' and
              c['mode'] == 'fisheye')
    assert c0['recall_difference'] == 0 and c0['recall_difference_ci'] == [0.0, 0.0]
    assert same.recordings[0]['label'] != same.recordings[1]['label']


def test_rig_comparison_on_one_time_grid(tmp_path):
    """A rig whose cameras run three times as fast gets no extra tries: both rigs' objects on the coarser grid, each
    camera's frame nearest each slot, objects paired by place and time."""
    def still(cam, k):
        return [{'id': 1, 'cls': 'car', 'box': [20, 120, 80, 160], 'range': 8.0, 'colour': (255, 0, 0),
                 'xy': (8.0, 0.0), 'actor': 11},
                {'id': 2, 'cls': 'car', 'box': [150, 100, 170, 114], 'range': 25.0, 'colour': None, 'xy': (25.0, -2.0),
                 'actor': 13}]
    a = make_recording(str(tmp_path / 'a'), still, n_frames=20, rate_hz=10.0, variant='Slow')
    b = make_recording(str(tmp_path / 'b'), still, n_frames=60, rate_hz=30.0, variant='Fast')
    res = DE.run(None, settings=_settings([a, b]), detector=ColourDetector())
    rig = next(c for c in res.comparisons if c['mode'] == 'rig' and c['class'] == 'car' and c['range_bin'] == 'all')
    assert rig['paired'] and rig['n_gt'] == 40 and rig['recall_reference'] == 0.5 and rig['recall'] == 0.5
    assert rig['recall_difference'] == 0.0
    own = {r['rec']: r for r in res.rig if r['class'] == 'car' and r['distance_bin'] == 'all'}
    assert own[0]['n'] == 40 and own[1]['n'] == 120          # each rig on its own frames


def test_files_and_export(recs, tmp_path):
    res = DE.run(None, settings=_settings([recs['a'], recs['b']]), detector=ColourDetector())
    names = [n for n, _ in DE.files(res)]
    for n in ('detection_quality.json', 'detection_quality.csv', 'detection_ranges.csv', 'detection_rig.csv',
              'detection_comparison.csv', 'detection_quality.png'):
        assert n in names
    assert any(n.startswith('detections/') and n.endswith('CAM_F.json') for n in names)
    assert any(n.startswith('overlays/') for n in names)
    paths = DE.export(res, None, str(tmp_path / 'out'))
    doc = json.load(open(os.path.join(tmp_path, 'out', 'detection_quality.json')))
    assert doc['schema'] == DE.SCHEMA and doc['range_bins'][0] == '0–10 m' and doc['references']
    assert len(doc['per_camera']) == len(res.rows) and all(not k.startswith('_') for k in doc['per_camera'][0])
    coco = json.load(open(next(p for p in paths if p.endswith('CAM_F.json'))))
    assert coco and set(coco[0]) == {'image_id', 'category_id', 'bbox', 'score'}
    import csv
    rows = list(csv.DictReader(open(os.path.join(tmp_path, 'out', 'detection_quality.csv'))))
    car = next(r for r in rows if r['class'] == 'car' and r['camera'] == 'all' and r['range_bin'] == 'all' and
               r['recording'].startswith('Rig A'))
    assert float(car['recall']) == 0.5 and float(car['precision']) == 0.8


def test_views_end_to_end(recs):
    det = ColourDetector()
    st = _settings([recs['b']], views='both', view_yaws=[0], view_fov_deg=120, view_pitch_deg=0, view_width=320)
    res = DE.run(None, settings=st, detector=det)
    assert det.calls == 40 and res.modes() == ['fisheye', 'pinhole']
    pv = res.row(0, 'pinhole', 'all', 'pedestrian', 'all')
    assert pv['n_gt'] == 20 and pv['recall'] == 1.0
    v = next(r for r in res.views if r['class'] == 'car' and r['range_bin'] == 'all')
    assert v['n'] == 40 and v['recall_fisheye'] == 1.0 and v['recall_views'] == 1.0 and v['difference'] == 0
    assert v['in_a_view'] == 40
    # the view's ground-truth boxes are the instance image remapped: they hold the objects the detector boxes
    fr = res.frames[(0, 'pinhole', 'F')][0]
    gt = fr['gt'][fr['gt'][:, 5] == 0]
    det_ = fr['det'][fr['det'][:, 6] == 1]
    assert len(det_) >= 3 and np.all(DE.iou_matrix(det_[:, :4], gt[:, :4]).max(1) > 0.6)
    img = DE.overlay_image(res, 0, 'pinhole', 'F', 0)
    assert img.shape == (240, 320, 3)


def test_cache_reuses_and_invalidates(recs, tmp_path):
    import shutil
    src = str(tmp_path / 'a')
    shutil.copytree(recs['a'], src)
    det = ColourDetector()
    r1 = DE.run(None, settings=_settings([src], cache=True), detector=det)
    assert det.calls == 20 and os.path.isdir(os.path.join(src, DE.CACHE_DIR))
    r2 = DE.run(None, settings=_settings([src], cache=True), detector=det)
    assert det.calls == 20 and r2.row()['recall'] == r1.row()['recall']
    # another detector setting is another cache; a changed image is detected again
    DE.run(None, settings=_settings([src], cache=True, nms_iou=0.5), detector=det)
    assert det.calls == 40
    f = DE.open_recording(src).cameras['F'].frames[3]['image']
    os.utime(os.path.join(src, f), ns=(1, 1))
    DE.run(None, settings=_settings([src], cache=True), detector=det)
    assert det.calls == 41


def test_cancel_and_missing_inputs(recs, tmp_path):
    det = ColourDetector()
    assert DE.run(None, settings=_settings([recs['a']]), detector=det, cancel=lambda: det.calls >= 5) is None
    assert det.calls == 5
    with pytest.raises(ValueError, match='Choose a recording'):
        DE.run(None, settings=DE.DetectionSettings(), detector=det)
    with pytest.raises(ValueError, match='no ground truth'):
        DE.run(None, settings=_settings([str(tmp_path)]), detector=det)
    with pytest.raises(ValueError, match='No class'):
        DE.run(None, settings=_settings([recs['a']], classes=[]), detector=det)
    # without the model: the run says so (the page offers the download)
    with pytest.raises(FileNotFoundError, match='not downloaded'):
        DE.run(None, settings=_settings([recs['a']]))
    # a project without detection settings reads the defaults
    from vantage.core import samples
    assert DE.settings_of(samples.tigor_sample_project()) == DE.DetectionSettings()


def test_ignore_regions_and_signs():
    st = DE.DetectionSettings()
    fr = {'objects': [
        {'id': 1, 'class': 'car', 'bbox': [0, 0, 50, 40], 'pixels': 2000, 'occluded': 0.9, 'truncated': 0,
         'range_m': 5},
        {'id': 2, 'class': 'car', 'bbox': [0, 0, 50, 40], 'pixels': 2000, 'occluded': 0.1, 'truncated': 0.6,
         'range_m': 5},
        {'id': 3, 'class': 'car', 'bbox': [0, 0, 50, 8], 'pixels': 400, 'range_m': 5},
        {'id': 4, 'class': 'car', 'bbox': [0, 0, 50, 40], 'pixels': 10, 'range_m': 5},
        {'id': 5, 'class': 'traffic_sign', 'bbox': [0, 0, 50, 40], 'pixels': 2000, 'range_m': 5},
        {'id': 6, 'class': 'traffic_sign', 'bbox': [0, 0, 50, 40], 'pixels': 2000, 'range_m': 5,
         'type_id': 'traffic.stop'},
        {'id': 7, 'class': 'building', 'bbox': [0, 0, 50, 40], 'pixels': 2000, 'range_m': 5},
        {'id': 8, 'class': 'bicycle', 'actor_id': 9, 'bbox': [0, 0, 50, 40], 'pixels': 2000, 'range_m': 5,
         'box3d': {'center': [3.0, 4.0, 0.5]}}]}
    g = DE.gt_objects(fr, st)
    assert g['ignore'].tolist() == [DE.IGN_VIS, DE.IGN_VIS, DE.IGN_SIZE, DE.IGN_SIZE, DE.IGN_SIGN, 0, 0, 0]
    assert [DT.CLASSES[c] if c >= 0 else None for c in g['cls']] == ['car'] * 4 + ['stop_sign', 'stop_sign', None,
                                                                                   'cyclist']
    assert g['dist'][7] == 5.0 and g['key'][7] == 'actor:9'


# ======================================================================= the model file
class _Files(http.server.BaseHTTPRequestHandler):
    payload = b''

    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-Length', str(len(self.payload)))
        self.end_headers()
        self.wfile.write(self.payload)

    def log_message(self, *a):
        pass


@pytest.fixture
def http_file():
    srv = http.server.ThreadingHTTPServer(('127.0.0.1', 0), _Files)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    yield srv
    srv.shutdown()


def test_download_checks_size_and_sha256(http_file, tmp_path):
    import hashlib
    data = os.urandom(3_000_000)
    _Files.payload = data
    url = f'http://127.0.0.1:{http_file.server_port}/m.onnx'
    good = DT.ModelSpec('t', 'Test', 't.onnx', url, hashlib.sha256(data).hexdigest(), len(data), 64, 0, 0, 0)
    seen = []
    path = DT.download(good, progress=lambda f, t: seen.append(f))
    assert path == os.path.join(DT.models_dir(), 't.onnx') and open(path, 'rb').read() == data
    assert seen and seen[-1] == 1.0 and str(tmp_path) in path
    st = DT.status(good)
    assert st['present'] and st['verified'] and 'SHA-256 checked' in st['text']
    # a wrong checksum: refused, nothing left behind
    bad = DT.ModelSpec('u', 'Bad', 'u.onnx', url, '0' * 64, len(data), 64, 0, 0, 0)
    with pytest.raises(ValueError, match='SHA-256'):
        DT.download(bad)
    assert not os.path.exists(os.path.join(DT.models_dir(), 'u.onnx'))
    assert not os.path.exists(os.path.join(DT.models_dir(), 'u.onnx.part'))
    # a wrong size, a cancel
    short = DT.ModelSpec('v', 'Short', 'v.onnx', url, hashlib.sha256(data).hexdigest(), len(data) + 5, 64, 0, 0, 0)
    with pytest.raises(ValueError, match='bytes'):
        DT.download(short)
    w = DT.ModelSpec('w', 'W', 'w.onnx', url, good.sha256, good.size, 64, 0, 0, 0)
    assert DT.download(w, cancel=lambda: True) is None
    assert sorted(os.listdir(DT.models_dir())) == ['t.onnx']
    # a model that is not there / modified
    st = DT.status('yolox_s')
    assert not st['present'] and 'not downloaded' in st['text']
    with pytest.raises(FileNotFoundError):
        DT.load(DT.DetectorSettings())
    with open(DT.model_path('yolox_s'), 'wb') as fh:
        fh.write(b'x' * DT.MODELS['yolox_s'].size)
    st = DT.status('yolox_s')
    assert st['present'] and not st['verified'] and 'does not match' in st['text']
    with pytest.raises(ValueError):
        DT.load(DT.DetectorSettings())
    with pytest.raises(ValueError, match='none of the published'):
        DT.import_file(path)
    assert DT.remove('yolox_s') and not DT.remove('yolox_s')


def test_models_folder_follows_the_platform(monkeypatch, tmp_path):
    monkeypatch.delenv(DT.ENV_DIR, raising=False)
    monkeypatch.setenv('XDG_DATA_HOME', str(tmp_path / 'xdg'))
    if os.name == 'posix' and not os.sys.platform.startswith('darwin'):
        assert DT.models_dir() == str(tmp_path / 'xdg' / 'Vantage' / 'models')
    assert '.config' not in DT.models_dir()


def test_pinned_models_are_https_and_documented():
    for m in DT.MODELS.values():
        assert m.url.startswith('https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/')
        assert len(m.sha256) == 64 and int(m.sha256, 16) >= 0 and m.size > 1e6 and m.licence == 'Apache-2.0'
    assert DT.DEFAULT_MODEL == 'yolox_s' and 'Apache License 2.0' in DT.LICENCE_TEXT


@real_model
def test_real_model_checksum_and_forward(monkeypatch):
    monkeypatch.setenv(DT.ENV_DIR, REAL_DIR)
    st = DT.status('yolox_s')
    assert st['present'] and st['verified']
    det = DT.load(DT.DetectorSettings())
    img = np.full((480, 640, 3), 114, np.uint8)
    out = det.detect(img)
    assert out['boxes'].shape[1] == 4 and len(out['scores']) == len(out['cls']) and out['input'] == (640, 480)
    assert det.describe()['sha256'] == DT.MODELS['yolox_s'].sha256
    again = det.detect(img)
    assert np.array_equal(out['boxes'], again['boxes'])


# ======================================================================= evaluate_variant through the (fake) bridge
def test_evaluate_variant_runs_the_rig_and_evaluates(tmp_path):
    """A variant driven through the bridge against the fake carla (in-process), then evaluated; a second call with
    the same rig reuses the drive. Skipped while the fake carla writes no ground truth."""
    import importlib.util
    import sys
    from vantage.core import samples, simrig
    from vantage.sim import carla_bridge as CB
    here = os.path.dirname(os.path.abspath(__file__))
    name = 'fake_carla_for_tests'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, os.path.join(here, 'fake_carla', 'carla', '__init__.py'))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    fc = sys.modules[name]
    p = samples.tigor_sample_project()
    v = next(v for v in p.variants if v.sensors)
    p.active_uid = v.uid
    sim = simrig.SimSettings.from_dict({'image_scale': 0.1, 'duration_s': 0.6, 'traffic': 4, 'warmup_s': 0.25,
                                        'seed': 4})
    runs = []

    def runner(rig_json, out, progress, cancel):
        runs.append(out)
        fc.reset()
        doc, remaps = CB.load_rig(rig_json)
        a = CB.resolve_args(CB.parser().parse_args([rig_json, out]), doc)
        CB.Bridge(fc, doc, remaps, out, a, None, cancel).run()

    class Oracle:
        """Detects every object of the drive's ground truth exactly (the pipeline is under test, not a detector)."""
        def __init__(self):
            self.by = None

        def __call__(self, img):
            import cv2
            if self.by is None:
                from vantage.core import groundtruth as GT
                doc = GT.read(out_dir)
                self.by = {}
                for c in doc['cameras'].values():
                    for f in c['frames']:
                        im = cv2.imread(GT.path_of(doc, f['image']), cv2.IMREAD_COLOR)
                        self.by[im.tobytes()] = f['objects']
            objs = [o for o in self.by.get(img.tobytes(), []) if DT.GT_TO_CLASS.get(o['class'])]
            boxes = np.array([o['bbox'] for o in objs], np.float32).reshape(-1, 4)
            cls = np.array([DT.CLASSES.index(DT.GT_TO_CLASS[o['class']]) for o in objs], np.int8)
            return {'boxes': boxes, 'scores': np.full(len(objs), 0.9, np.float32), 'cls': cls,
                    'coco': np.zeros(len(objs), np.int16)}
    out_dir = str(tmp_path / 'point')
    try:
        res = DE.evaluate_variant(p, v, sim, DE.DetectionSettings.from_dict({'bootstrap': 50, 'cache': False}),
                                  out_dir, detector=Oracle(), runner=runner)
    except ValueError as e:
        if 'no ground truth' in str(e):
            pytest.skip('the fake carla writes no ground truth yet')
        raise
    assert runs == [out_dir] and res is not None and res.recordings[0]['path'] == out_dir
    found = [r for r in res.rows if r['camera'] == 'all' and r['range_bin'] == 'all' and r['n_gt']]
    assert found and all(r['recall'] == 1.0 and r['precision'] == 1.0 for r in found)
    res2 = DE.evaluate_variant(p, v, sim, DE.DetectionSettings.from_dict({'bootstrap': 50, 'cache': False}),
                               out_dir, detector=Oracle(), runner=runner)
    assert runs == [out_dir] and res2 is not None                 # the same rig: the drive is reused


# ======================================================================= the page
from test_gui import app, win, pump, _wait   # noqa: E402,F401 (fixtures)


def _open(win, name, cls):
    if name not in win._page_widgets:
        pg = cls(win.ctx)
        win.pages.addWidget(pg)
        win._page_widgets[name] = pg
    win.set_page(name)
    return win._page_widgets[name]


def test_page_runs_compares_shows_frames_and_undoes(app, win, recs, tmp_path, http_file):
    from vantage.ui.workspaces.detection_ws import DetectionPage
    pg = _open(win, 'detection', DetectionPage)
    pump(app, 0.2)
    assert pg.status.text().startswith('Download the detector') and not pg.b_run.isEnabled()
    assert 'not downloaded' in pg.model_status.text.text() and pg.b_download.isEnabled()
    # recordings: added (undoable), the reference changed, a folder without ground truth refused
    assert pg.add_recording(recs['a']) and pg.add_recording(recs['b'])
    assert not pg.add_recording(str(tmp_path))
    pump(app, 0.1)
    assert pg.settings.recordings == [recs['a'], recs['b']] and pg.recs.count() == 2
    assert pg.recs.item(0).text().startswith('★ Rig A')
    assert pg.make_reference(1) and pg.settings.recordings[0] == recs['b']
    win.store.undo.undo()
    pump(app, 0.1)
    assert pg.settings.recordings[0] == recs['a'] and pg.recs.item(0).text().startswith('★ Rig A')
    # a setting typed into its field, undone
    f = pg.fields['score']
    f.set_value(0.5)
    f.committed.emit(0.5)
    pump(app, 0.1)
    assert pg.settings.score == 0.5
    win.store.undo.undo()
    pump(app, 0.1)
    assert pg.settings.score == 0.3 and pg.fields['score'].value() == pytest.approx(0.3)
    pg.class_box['bus'].setChecked(False)
    pump(app, 0.1)
    assert 'bus' not in pg.settings.classes
    win.store.undo.undo()
    assert 'bus' in pg.settings.classes
    # without the model Run refuses and says so; with the test's detector it runs as a background job
    assert not pg.run() and 'Download' in pg.status.text()
    pg.detector = ColourDetector()
    pg.views.buttons['both'].click()
    pump(app, 0.1)
    assert pg.settings.views == 'both'
    pg._set('views', view_yaws=[0.0], view_fov_deg=120.0, view_pitch_deg=0.0, view_width=320, bootstrap=200,
            cache=False)
    assert pg.run()
    _wait(app, lambda: pg.result is not None and not pg.busy, 120)
    res = pg.result
    assert len(res.recordings) == 2 and res.modes() == ['fisheye', 'pinhole']
    assert pg.rec_pick.count() == 2 and pg.cls_pick.currentData() == 'vehicle'
    assert pg.tiles['objects'].value == '60' and pg.tiles['recall'].value == '50 %'
    assert pg.cam_table.rowCount() > 0 and pg.class_table.rowCount() > 0 and pg.cmp_table.rowCount() > 0
    assert pg.view_table.rowCount() > 0 and pg.cam_chart.series and pg.rig_chart.series
    # the picks: the other recording, a class, the pinhole views
    pg.rec_pick.setCurrentIndex(1)
    pump(app, 0.1)
    assert pg.tiles['recall'].value == '83 %'
    pg.cls_pick.setCurrentIndex(pg.cls_pick.findData('pedestrian'))
    pg.mode_pick.buttons['pinhole'].click()
    pump(app, 0.1)
    assert pg._mode() == 'pinhole' and pg.tiles['recall'].value == '100 %'
    # frames: the slider walks the frames, the overlay shows
    pg.tabs.setCurrentIndex(TABS_FRAMES)
    pg.mode_pick.buttons['fisheye'].click()
    pg._fill_cams()
    pump(app, 0.1)
    assert pg.cam_pick.count() == 1 and pg.slider.maximum() == 19
    pg.slider.setValue(7)
    pump(app, 0.1)
    assert pg.frame_view.img is not None and pg.frame_label.text().startswith('8/20')
    # an edit after the run marks the result stale
    pg._set('score', score=0.4)
    pump(app, 0.1)
    assert pg.stale and 'Run again' in pg.status.text()
    win.store.undo.undo()
    pump(app, 0.1)
    assert not pg.stale
    # export
    paths = pg.export(str(tmp_path / 'exp'))
    assert any(x.endswith('detection_quality.json') for x in paths)
    assert pg.summary()['recordings'][0]['recording'].startswith('Rig A')
    # the download: a test model served locally, the job bar, the status turning good
    import hashlib
    data = b'onnx' * 1000
    _Files.payload = data
    spec = DT.ModelSpec('yolox_s', 'YOLOX-S', 'yolox_s.onnx', f'http://127.0.0.1:{http_file.server_port}/m',
                        hashlib.sha256(data).hexdigest(), len(data), 640, 40.5, 9.0, 26.8)
    old = DT.MODELS['yolox_s']
    DT.MODELS['yolox_s'] = spec
    try:
        assert pg.download()
        _wait(app, lambda: not pg.dl_busy, 60)
        assert pg.model_status.text.text().startswith('YOLOX-S ready') and not pg.b_download.isEnabled()
    finally:
        DT.MODELS['yolox_s'] = old
    # a new project: the result goes
    from vantage.core import samples
    win.store.set_project(samples.tigor_sample_project())
    pump(app, 0.2)
    assert pg.result is None and pg.rec_pick.count() == 0


TABS_FRAMES = 5
