"""vantage.core.groundtruth: the class table against CARLA 0.9.16's labels and nuScenes-lidarseg's categories, the lens
projection against core.lens, boxes, COCO run-length encoding, the camera labels read through the remap tables (nearest
view pixel, range along the output ray), objects and their geometry on a scene ray-cast independently, the traffic
lights' and signs' boxes claiming their pixels, and the reader."""
import json
import math
import os

import numpy as np
import pytest

from vantage.core import groundtruth as GT

# carla.CityObjectLabel of the real carla 0.9.16 client module (printed 2026-10-27): name -> value
CARLA_LABELS = {'NONE': 0, 'Roads': 1, 'Sidewalks': 2, 'Buildings': 3, 'Walls': 4, 'Fences': 5, 'Poles': 6,
                'TrafficLight': 7, 'TrafficSigns': 8, 'Vegetation': 9, 'Terrain': 10, 'Sky': 11, 'Pedestrians': 12,
                'Rider': 13, 'Car': 14, 'Truck': 15, 'Bus': 16, 'Train': 17, 'Motorcycle': 18, 'Bicycle': 19,
                'Static': 20, 'Dynamic': 21, 'Other': 22, 'Water': 23, 'RoadLines': 24, 'Ground': 25, 'Bridge': 26,
                'RailTrack': 27, 'GuardRail': 28}
# ColorConverter.CityScapesPalette's colours read off a 0.9.16 server's semantic image (RGB)
MEASURED_PALETTE = {1: (128, 64, 128), 2: (244, 35, 232), 3: (70, 70, 70), 6: (153, 153, 153), 7: (250, 170, 30),
                    8: (220, 220, 0), 9: (107, 142, 35), 11: (70, 130, 180), 12: (220, 20, 60), 13: (255, 0, 0),
                    14: (0, 0, 142), 18: (0, 0, 230), 19: (119, 11, 32), 20: (110, 190, 160), 21: (170, 120, 50),
                    22: (55, 90, 80), 24: (157, 234, 50), 25: (81, 0, 81)}
CONTRACT = ('unlabelled', 'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic_light', 'traffic_sign',
            'vegetation', 'terrain', 'sky', 'pedestrian', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
            'bicycle', 'static', 'dynamic', 'other', 'water', 'road_line', 'ground', 'bridge', 'rail_track',
            'guard_rail')


# ======================================================================= classes
def test_class_table_is_carlas_labels_with_their_nuscenes_categories():
    doc = GT.classes_doc()
    names = [c['name'] for c in doc]
    assert set(CONTRACT) <= set(names) and names[:29] == list(CONTRACT) and doc[29]['name'] == 'ego'
    for c in doc[:29]:                                      # the class id is CARLA's own tag value
        assert CARLA_LABELS[c['carla_label']] == c['id'] == c['carla_tags'][0]
        assert c['nuscenes'] in GT.NUSC_INDEX and c['nuscenes_index'] == GT.NUSC_INDEX[c['nuscenes']]
        assert len(c['color']) == 3
    pal = GT.palette()
    for tag, rgb in MEASURED_PALETTE.items():
        assert tuple(pal[tag]) == rgb, tag
    assert np.array_equal(GT.CLASS_OF_TAG[:29], np.arange(29)) and GT.CLASS_OF_TAG[200] == 0
    # nuScenes-lidarseg: 32 categories, indices 0..31 in order, the devkit's names
    assert [i for i, _, _ in GT.NUSCENES_LIDARSEG] == list(range(32))
    assert GT.NUSC_NAME[0] == 'noise' and GT.NUSC_NAME[17] == 'vehicle.car' and GT.NUSC_NAME[31] == 'vehicle.ego'
    assert GT.NUSC_NAME[24] == 'flat.driveable_surface' and GT.NUSC_NAME[28] == 'static.manmade'
    assert GT.NUSC_OF_TAG[14] == 17 and GT.NUSC_OF_TAG[1] == GT.NUSC_OF_TAG[24] == 24 and GT.NUSC_OF_TAG[11] == 0
    rows = GT.nuscenes_categories(lambda n: 'tok:' + n, [('vehicle.car', 'x'), ('my.extra', 'y')])
    assert [r['index'] for r in rows] == list(range(33)) and rows[-1]['name'] == 'my.extra'
    assert len({r['name'] for r in rows}) == 33


def test_actor_categories():
    assert GT.category_of('walker.pedestrian.0009', {'age': 'child'}) == 'human.pedestrian.child'
    assert GT.category_of('walker.pedestrian.0001', {'age': 'adult'}) == 'human.pedestrian.adult'
    assert GT.category_of('vehicle.ford.ambulance', {'base_type': 'van'}) == 'vehicle.emergency.ambulance'
    assert GT.category_of('vehicle.diamondback.century', {'base_type': 'bicycle'}) == 'vehicle.bicycle'
    assert GT.category_of('vehicle.yamaha.yzf', {'base_type': 'motorcycle'}) == 'vehicle.motorcycle'
    assert GT.category_of('vehicle.carlamotors.carlacola', {'base_type': 'truck'}) == 'vehicle.truck'
    assert GT.category_of('vehicle.mitsubishi.fusorosa', {'base_type': 'Bus'}) == 'vehicle.bus.rigid'
    assert GT.category_of('static.prop.trafficcone01') == 'movable_object.trafficcone'
    assert GT.category_of('static.prop.streetbarrier') == 'movable_object.barrier'
    assert GT.category_of('vehicle.tesla.model3', {'base_type': 'car'}) == 'vehicle.car'


# ======================================================================= CARLA images
def test_depth_and_instance_decoding():
    rng = np.random.default_rng(1)
    d = rng.uniform(0.5, 900.0, (37, 51))
    img = GT.encode_depth(d)
    assert np.abs(GT.decode_depth(img) - d).max() < 1.2e-4         # a step (6e-5 m) and float32's rounding at 900 m
    raw = img.tobytes()
    assert np.allclose(GT.unpack_depth(GT.packed(raw, 51, 37)), GT.decode_depth(img), atol=1e-6)
    bgra = rng.integers(0, 256, (20, 30, 4), dtype=np.uint8)
    tag, ids = GT.decode_instance(bgra)
    assert np.array_equal(tag, bgra[..., 2]) and np.array_equal(ids, bgra[..., 1].astype(int) +
                                                                256 * bgra[..., 0].astype(int))
    t2, i2 = GT.unpack_instance(GT.packed(bgra.tobytes(), 30, 20))
    assert np.array_equal(t2, tag) and np.array_equal(i2, ids)
    s = np.zeros(3, GT.SEMANTIC_LIDAR)
    s['idx'] = [0, 7, 70000]
    assert list(GT.semantic_lidar(s.tobytes())['idx']) == [0, 7, 70000] and GT.SEMANTIC_LIDAR.itemsize == 24


# ======================================================================= lens
def test_projection_is_the_lens_model():
    from vantage.core import lens as L
    t = np.linspace(0.0, 1.6, 50)
    for proj in ('rectilinear', 'equidistant', 'equisolid', 'stereographic', 'orthographic', 'kannala_brandt'):
        kb = (0.01, -0.002, 0.0003, -0.00002)
        assert np.allclose(GT.r_of_theta(t, 2.1, proj, kb), L.r_of_theta(t, 2.1, proj, kb), atol=1e-12), proj
    from test_sim import _project
    from vantage.core import simrig
    p = _project(extra=False)
    rig = simrig.build(p, simrig.SimSettings.from_dict({'image_scale': 0.2}), tables=False)
    for doc in rig.doc['cameras'][:2]:
        cam = p.variant.by_tag(doc['tag'])
        intr = L.resolve(p.module_for(cam), p.lens_for(cam))
        lens = doc['lens']
        D = np.random.default_rng(4).normal(size=(500, 3))
        D[:, 2] = np.abs(D[:, 2])
        u, v, th, ok = intr.project_dirs(D)
        u2, v2, th2, ok2 = GT.project(lens, D * 3.7)                 # points along the rays: the same pixels
        w, h = lens['image_size']
        assert np.allclose(u2, u * w / intr.width) and np.allclose(v2, v * h / intr.height)
        assert np.array_equal(ok2, th <= intr.max_theta)
        # and the output rays (core.simrig) project back to their own pixel centres (pixel i covers [i, i+1))
        d, okr = simrig.output_rays(intr, w, h)
        uu, vv, _, okp = GT.project(lens, d[okr])
        ys, xs = np.nonzero(okr)
        assert np.abs(uu - (xs + 0.5)).max() < 1e-6 and np.abs(vv - (ys + 0.5)).max() < 1e-6
    K = np.array([[200.0, 0, 99.5], [0, 200.0, 79.5], [0, 0, 1]])
    u, v, _, ok = GT.project({'model': 'pinhole', 'K': K.tolist(), 'max_theta_deg': 80.0}, [[0.1, -0.2, 1.0],
                                                                                             [0, 0, -1.0]])
    assert u[0] == pytest.approx(200 * 0.1 + 99.5 + 0.5) and v[0] == pytest.approx(-40 + 79.5 + 0.5) and not ok[1]


# ======================================================================= boxes
def test_box_helpers_against_sampling():
    rng = np.random.default_rng(3)
    for _ in range(20):
        c = rng.uniform(-5, 5, 3)
        size = rng.uniform(0.3, 4.0, 3)
        R = GT.box_matrix(rng.uniform(-180, 180))
        E = GT.box_edges(c, size, R)
        F = GT.box_faces(c, size, R)
        assert len(E) == 12 * 17 and len(F) == 6 * 81
        q = np.abs((E - c) @ R) / (size / 2)
        assert np.allclose(np.sort(q, 1)[:, 1:], 1.0)              # every edge point: on two faces
        q = np.abs((F - c) @ R) / (size / 2)
        assert np.allclose(q.max(1), 1.0)
        assert GT.in_box(F, c, size, R, 1e-9).all() and not GT.in_box(c + R @ (size * 0.51 * [1, 0, 0]), c, size, R)
        o = c + rng.uniform(-12, 12, 3)
        D = rng.normal(size=(400, 3))
        D /= np.linalg.norm(D, axis=1, keepdims=True)
        tin, tout = GT.ray_box(o, D, c, size, R)
        s = np.linspace(0, 40, 8001)
        for k in range(0, 400, 7):                                 # brute force along the ray
            inside = GT.in_box(o + s[:, None] * D[k], c, size, R)
            if inside.any():
                assert abs(tin[k] - s[np.argmax(inside)]) < 0.01 and abs(tout[k] - s[len(s) - 1 - np.argmax(inside[::-1])]) < 0.01
            else:
                assert not np.isfinite(tin[k]) or tin[k] > 39.99


# ======================================================================= COCO RLE
def test_rle_round_trip_and_from_pixels():
    rng = np.random.default_rng(2)
    for _ in range(300):
        h, w = rng.integers(1, 30, 2)
        m = rng.random((h, w)) < rng.random()
        r = GT.rle_encode(m)
        assert r['size'] == [h, w] and np.array_equal(GT.rle_decode(r), m)
        assert GT.rle_of_pixels(np.flatnonzero(m.ravel()), h, w) == r
        # the counts, column-major runs from the zeros: an independent run count
        flat = m.ravel(order='F')
        runs, cur, n = [], False, 0
        for x in flat:
            if bool(x) != cur:
                runs.append(n)
                cur, n = bool(x), 0
            n += 1
        runs.append(n)
        assert GT.rle_counts(r) == runs
    # pycocotools' own string for a 3 x 4 mask (its documented algorithm: delta of counts two back, 5-bit groups)
    m = np.array([[0, 1, 1, 0], [0, 1, 0, 0], [1, 1, 0, 1]], bool)
    assert GT.rle_counts(GT.rle_encode(m)) == [2, 5, 4, 1]                 # 0 0 1 | 1 1 1 | 1 0 0 | 0 0 1


# ======================================================================= camera labels through the remap
@pytest.fixture(scope='module')
def cam():
    from test_sim import _project
    from vantage.core import simrig
    p = _project(extra=False)
    rig = simrig.build(p, simrig.SimSettings.from_dict({'image_scale': 0.15}))
    doc = next(c for c in rig.doc['cameras'] if c['tag'] == 'F')
    return doc, rig.remaps, GT.CameraGT(doc, rig.remaps)


def _view_rays(doc):
    """Each atlas pixel's ray in the camera's optical frame, from the views alone (as CARLA renders them)."""
    at = doc['atlas']
    out = np.zeros((at['height'], at['width'], 3))
    for k, v in enumerate(doc['views']):
        u, vv = np.meshgrid(np.arange(v['width'], dtype=float), np.arange(v['height'], dtype=float))
        d = np.stack([(u - (v['width'] / 2 - 0.5)) / v['f_px'], (vv - (v['height'] / 2 - 0.5)) / v['f_px'],
                      np.ones_like(u)], -1)
        d /= np.linalg.norm(d, axis=-1, keepdims=True)
        out[at['offsets'][k]:at['offsets'][k] + v['height'], :v['width']] = d @ np.asarray(v['optical']).T
    return out


def test_labels_read_the_nearest_view_pixel_and_the_range_runs_along_the_ray(cam):
    """Every output pixel reads the view pixel nearest its ray (within half a view pixel, never a mix), the blend
    band's pixels the face with the higher weight; a sphere of radius 7 m about the camera gives range 7 m at every
    pixel from the views' planar depths."""
    doc, remaps, cg = cam
    at = doc['atlas']
    vr = _view_rays(doc)
    idx = np.arange(at['height'] * at['width']).reshape(at['height'], at['width'])
    got = cg.gather(idx, fill=-1)
    seen = got >= 0
    assert seen.mean() > 0.5 and np.array_equal(seen, cg.ok)
    ray_view = vr.reshape(-1, 3)[got[seen]]
    ang = np.degrees(np.arccos(np.clip(np.sum(ray_view * cg.rays[seen], 1), -1, 1)))
    f_min = min(v['f_px'] for v in doc['views'])
    assert ang.max() < math.degrees(0.75 / f_min)                  # within ¾ of a view pixel's angle
    # in the seam band a pixel reads the face with the higher weight: the remap's own face has it in these tables
    # (the face whose axis is nearest the ray); a table giving another face more weight moves the read there
    b = doc['remap']['blend']
    idx_b, bx, by, bw = (remaps[b[k]] for k in ('idx', 'x', 'y', 'w'))
    assert np.all(np.argmax(bw, 1) == 0) and cg.blend_moved == 0
    bw2 = bw.copy()
    sel = np.flatnonzero(bw[:, 1] > 0)[::3]
    bw2[sel] = [0.2, 0.8, 0.0]
    cg2 = GT.CameraGT(doc, dict(remaps, **{b['w']: bw2}))
    assert cg2.blend_moved == len(sel)
    got2 = cg2.gather(idx, fill=-1).ravel()[idx_b[sel]]
    want = np.floor(by[sel, 0] + 0.5).astype(int) * at['width'] + np.floor(bx[sel, 0] + 0.5).astype(int)
    assert np.array_equal(got2, want)
    # range: planar depth of a sphere seen by each view
    z = np.zeros((at['height'], at['width']), np.float32)
    for k, v in enumerate(doc['views']):
        sl = slice(at['offsets'][k], at['offsets'][k] + v['height'])
        rv = vr[sl, :v['width']] @ np.asarray(v['optical'])        # back into the view's own frame: its z is depth
        z[sl, :v['width']] = 7.0 * rv[..., 2]
    r = cg.range_of(cg.gather(z))
    # nearest sampling: a pixel's ray and its view pixel's differ by up to ¾ of a view pixel; on a sphere the planar
    # depth changes by tan(θ) of that across it (θ ≤ 55° in a cube face)
    err = np.abs(r[cg.ok] - 7.0)
    kpx = np.searchsorted(np.asarray(at['offsets']), got[seen] // at['width'], side='right') - 1
    axes = np.array([np.asarray(v['optical'])[:, 2] for v in doc['views']])
    cos_px = np.sum(ray_view * axes[kpx], 1)                       # the view pixel's ray against its view's axis
    cos_out = np.sum(cg.rays[seen] * axes[kpx], 1)                 # the output pixel's own ray
    assert np.abs(err - 7.0 * np.abs(cos_px / cos_out - 1.0)).max() < 1e-5 and np.all(r[~cg.ok] == 0)
    assert err.max() <= 7.0 * 1.5 * 0.75 / f_min
    assert np.allclose(cg.range_m(z), r)
    # a plane facing a view (planar depth 5 m everywhere in it): the range along each output ray, exactly
    z2 = np.full_like(z, 5.0)
    r2 = cg.range_of(cg.gather(z2))
    for k, v in enumerate(doc['views']):
        on = cg.ok & ((got // at['width']) >= at['offsets'][k]) & ((got // at['width']) < at['offsets'][k] + v['height'])
        dz = (cg.rays[on] @ np.asarray(v['optical']))[:, 2]
        assert np.allclose(r2[on], 5.0 / dz, rtol=1e-5), v['name']
    png = GT.range_png(r)
    assert np.abs(png[cg.ok] / GT.RANGE_SCALE - r[cg.ok]).max() <= 0.5 / GT.RANGE_SCALE + 1e-6
    assert GT.range_png(np.array([300.0, 0.0, 10.0])).tolist() == [0, 0, 2560]


def test_range_interpolates_inverse_depth_within_a_surface_and_never_across_an_edge(cam):
    """The range image reads each pixel's planar depth at its ray's exact position: a tilted plane comes out exact at
    every pixel (inverse depth is linear in a pinhole view's pixels); at an edge between two surfaces (two instance
    ids) a pixel takes one of them, never a value between."""
    doc, _, cg = cam
    at = doc['atlas']
    vr = _view_rays(doc)
    n = np.array([0.3, -0.5, 0.81])
    n /= np.linalg.norm(n)
    c = 6.0                                                        # the plane n · X = c in the camera's optical frame
    z = np.zeros((at['height'], at['width']))
    for k, v in enumerate(doc['views']):
        sl = slice(at['offsets'][k], at['offsets'][k] + v['height'])
        d = vr[sl, :v['width']]
        t = c / np.maximum(d @ n, 1e-6)                            # the range along each view pixel's ray
        z[sl, :v['width']] = t * (d @ np.asarray(v['optical']))[..., 2]
    seg = np.zeros(z.shape, np.uint32)
    r = cg.range_interp(z, seg)
    want = c / np.maximum(cg.rays[cg.ok] @ n, 1e-6)
    front = (cg.rays[cg.ok] @ n) > 0.05                            # (the plane in front of the camera)
    assert np.allclose(r[cg.ok][front], want[front], rtol=1e-5)
    # an edge: one surface at 5 m, another at 9 m beyond a line through the image; each pixel one of the two
    z2 = np.where(vr[..., 0] > 0.1, 5.0, 9.0) * 1.0
    seg2 = (vr[..., 0] > 0.1).astype(np.uint32)
    for k, v in enumerate(doc['views']):                           # planar depths of two spheres: exact ranges 5 / 9
        sl = slice(at['offsets'][k], at['offsets'][k] + v['height'])
        z2[sl, :v['width']] *= (vr[sl, :v['width']] @ np.asarray(v['optical']))[..., 2]
    r2 = cg.range_interp(z2, seg2)[cg.ok]
    near5, near9 = np.abs(r2 - 5.0) < 0.05, np.abs(r2 - 9.0) < 0.09
    assert np.all(near5 | near9) and near5.sum() > 100 and near9.sum() > 100
    # without the instance image, depths within 5 % of each other count as one surface (where the plane is steep
    # across a pixel, beyond that, the nearest view pixel's depth)
    r3 = cg.range_interp(z, None)[cg.ok]
    nearest = cg.range_of(cg.gather(z))[cg.ok]
    exact = np.isclose(r3, want, rtol=1e-5)
    assert np.all(exact | np.isclose(r3, nearest, rtol=1e-6)) and exact[front].mean() > 0.9


def _render(cg, pose, boxes):
    """Instance ids, tags and range of boxes in the vehicle frame along the camera's output rays: an independent ray
    cast (core.groundtruth.ray_box)."""
    R, pos = pose['R'], pose['pos']
    D = cg.rays.reshape(-1, 3) @ (R @ GT.BODY_TO_OPTICAL.T).T
    best = np.full(len(D), np.inf)
    ids = np.zeros(len(D), np.uint16)
    tags = np.full(len(D), 11, np.uint8)                          # sky
    for rid, tag, c, size, Rb in boxes:
        tin, _ = GT.ray_box(pos, D, c, size, Rb)
        near = tin < best
        best[near] = tin[near]
        ids[near] = rid
        tags[near] = tag
    sh = cg.rays.shape[:2]
    ok = cg.ok.ravel()
    best[~ok] = np.inf
    return ids.reshape(sh), tags.reshape(sh), np.where(np.isfinite(best), best, 0.0).reshape(sh).astype(np.float32)


def test_objects_boxes_truncation_and_occlusion(cam):
    from vantage.sim.carla_bridge import pose_matrix
    doc, _, cg = cam
    pv = doc['pose_vehicle']
    pose = {'R': pose_matrix(pv), 'pos': np.array([pv['x'], pv['y'], pv['z']])}
    car = {'key': ('actor', 7), 'actor_id': 7, 'reg': 1, 'category': 'vehicle.car', 'center': np.array([12.0, 1.0, 0.75]),
           'size': [4.5, 1.9, 1.5], 'R': GT.box_matrix(20.0)}
    occ = (9, 20, np.array([8.0, 0.2, 0.75]), [0.3, 0.8, 1.6], np.eye(3))      # a static thing half in front of it
    ids, tags, rng = _render(cg, pose, [(7, 14, car['center'], car['size'], car['R'])])
    h, w = ids.shape
    found, sem = GT.objects_in_frame(tags, ids, rng, cg.rays, pose, doc['lens'], {7: car}, [], ego_id=1, assoc={})
    assert len(found) == 1 and found[0]['class'] == 14 and found[0]['pixels'] == int((ids == 7).sum())
    g = GT.object_geometry(car, pose, doc['lens'], w, h, cg.rays, rng, pix=found[0]['pix'])
    # a box object: its mask's box is its 3D box through the lens, within a pixel
    assert np.abs(np.array(found[0]['bbox']) - np.array(g['bbox_full'])).max() <= 1.01
    assert g['occluded'] == 0.0 and g['truncated'] == 0.0
    assert g['range_m'] == pytest.approx(np.linalg.norm(car['center'] - pose['pos']), abs=1e-3)
    ids2, tags2, rng2 = _render(cg, pose, [(7, 14, car['center'], car['size'], car['R']), occ])
    found2, _ = GT.objects_in_frame(tags2, ids2, rng2, cg.rays, pose, doc['lens'], {7: car}, [], 1, {})
    g2 = GT.object_geometry(car, pose, doc['lens'], w, h, cg.rays, rng2, pix=found2[0]['pix'])
    hidden = int(((ids2 == 9) & (ids == 7)).sum())
    assert 0.05 < g2['occluded'] < 0.95
    assert g2['occluded'] == pytest.approx(hidden / (hidden + found2[0]['pixels']), abs=0.01)
    # a box on the image's left border: about half of its surface outside the image
    edge = cg.rays[h // 2, np.flatnonzero(cg.ok[h // 2])[0]]
    centre = pose['pos'] + (pose['R'] @ GT.BODY_TO_OPTICAL.T) @ edge * 8.0
    side = dict(car, center=centre, size=[1.0, 1.0, 1.0], R=np.eye(3))
    g3 = GT.object_geometry(side, pose, doc['lens'], w, h)
    assert 0.2 < g3['truncated'] < 0.8 and g3['occluded'] is None and g3['bbox_full'][0] < 0 < g3['bbox_full'][2]
    # the ego's own pixels: class 'ego', never an object
    ids3 = ids.copy()
    ids3[ids == 7] = 1
    found3, sem3 = GT.objects_in_frame(tags, ids3, rng, cg.rays, pose, doc['lens'], {7: car}, [], 1, {})
    assert not found3 and np.all(sem3[ids == 7] == GT.EGO_CLASS)


def test_level_objects_by_their_boxes_and_lights_claim_their_heads(cam):
    """A parked car of the map (no actor id) is found by its level box; a traffic light's head box claims the static
    pixels inside it whatever CARLA tagged them (here 'pole' and 'other'), each head its own, a car passing in front
    keeps its own."""
    from vantage.sim.carla_bridge import pose_matrix
    doc, _, cg = cam
    pv = doc['pose_vehicle']
    pose = {'R': pose_matrix(pv), 'pos': np.array([pv['x'], pv['y'], pv['z']])}
    parked = {'key': ('level', 0), 'level_label': 'Car', 'actor_id': None, 'reg': 1000, 'category': 'vehicle.car',
              'center': np.array([10.0, -3.0, 0.75]), 'size': [4.5, 1.9, 1.5], 'R': np.eye(3)}
    head = {'key': ('level', 1), 'level_label': 'TrafficLight', 'actor_id': 31, 'reg': 1001, 'category': 'static.manmade',
            'part_class': 7, 'center': np.array([9.0, 3.0, 1.5]), 'size': [0.4, 0.6, 1.0], 'R': np.eye(3)}
    head2 = dict(head, key=('level', 2), reg=1002, center=np.array([15.0, 4.666, 1.6]))
    # rendered: the parked car with a level id; the head as 'pole' (top half) and 'other' (bottom half), its id 31
    ids, tags, rng = _render(cg, pose, [(45000, 14, parked['center'], parked['size'], parked['R']),
                                        (31, 6, head['center'] + [0, 0, 0.25], [0.4, 0.6, 0.5], np.eye(3)),
                                        (31, 22, head['center'] - [0, 0, 0.25], [0.4, 0.6, 0.5], np.eye(3)),
                                        (31, 22, head2['center'], head2['size'], np.eye(3))])
    assoc = {}
    found, sem = GT.objects_in_frame(tags, ids, rng, cg.rays, pose, doc['lens'], {}, [parked], 99, assoc,
                                     parts=[head, head2])
    by = {f['obj']['key']: f for f in found}
    assert assoc[45000] == ('level', 0) and by[('level', 0)]['class'] == 14
    hp = by[('level', 1)]
    other = by.get(('level', 2), {'pixels': 0})['pixels']
    assert hp['class'] == 7 and hp['pixels'] > 20 and hp['pixels'] + other == int((ids == 31).sum())
    assert np.all(sem.ravel()[hp['pix']] == 7) and set(np.unique(tags.ravel()[hp['pix']])) == {6, 22}
    assert hp['raw_tags'][6] > 0 and hp['raw_tags'][22] > 0 and hp['raw_tags'][7] == 0
    # a car in front of the head keeps its pixels
    car = {'key': ('actor', 5), 'actor_id': 5, 'reg': 3, 'category': 'vehicle.car', 'center': np.array([6.0, 2.0, 1.4]),
           'size': [0.6, 0.6, 0.6], 'R': np.eye(3)}
    ids2, tags2, rng2 = _render(cg, pose, [(31, 22, head['center'], head['size'], np.eye(3)),
                                           (5, 14, car['center'], car['size'], np.eye(3))])
    found2, sem2 = GT.objects_in_frame(tags2, ids2, rng2, cg.rays, pose, doc['lens'], {5: car}, [], 99, {},
                                       parts=[head])
    by2 = {f['obj']['key']: f for f in found2}
    assert by2[('actor', 5)]['pixels'] == int((ids2 == 5).sum()) and np.all(sem2[ids2 == 5] == 14)


# ======================================================================= the reader
def test_read_and_loaders(tmp_path):
    import cv2
    out = tmp_path
    (out / 'ground_truth' / 'CAM_F').mkdir(parents=True)
    sem = np.random.default_rng(0).integers(0, 29, (6, 8)).astype(np.uint8)
    inst = np.zeros((6, 8), np.uint16)
    inst[1:3, 2:5] = 1003
    rng = np.full((6, 8), 12.5, np.float32)
    for name, a in (('semantic', sem), ('instance', inst), ('range', GT.range_png(rng))):
        cv2.imwrite(str(out / 'ground_truth' / 'CAM_F' / f'x.{name}.png'), a)
    frame = {k: f'ground_truth/CAM_F/x.{k}.png' for k in ('semantic', 'instance', 'range')}
    frame.update(sample_data_token='t', timestamp=1, image=None, objects=[{'id': 1003, 'class': 'car', 'actor_id': 4}])
    (out / 'nuscenes' / 'lidarseg').mkdir(parents=True)
    np.array([17, 24], np.uint8).tofile(out / 'nuscenes' / 'lidarseg' / 'a_lidarseg.bin')
    np.savez_compressed(out / 'nuscenes' / 'lidarseg' / 'a_panoptic.npz', data=np.array([17003, 24000], np.uint16))
    sweep = {'lidarseg': 'nuscenes/lidarseg/a_lidarseg.bin', 'panoptic': 'nuscenes/lidarseg/a_panoptic.npz'}
    doc = {'schema': GT.SCHEMA, 'classes': GT.classes_doc(), 'cameras': {'F': {'channel': 'CAM_F', 'width': 8,
                                                                                'height': 6, 'frames': [frame]}},
           'lidars': {'L1': {'channel': 'LIDAR_L1', 'sweeps': [sweep]}}}
    (out / 'ground_truth' / 'index.json').write_text(json.dumps(doc))
    d = GT.read(str(out))
    assert d is not None and d['_root'] == os.path.abspath(str(out))
    f = GT.frames(d, 'F')[0]
    assert np.array_equal(GT.load_semantic(d, f), sem) and np.array_equal(GT.load_instance(d, f), inst)
    assert np.allclose(GT.load_range(d, f), 12.5)
    assert GT.load_lidarseg(d, sweep).tolist() == [17, 24] and GT.load_panoptic(d, sweep).tolist() == [17003, 24000]
    s = GT.summary(d)
    assert s['objects'] == 1 and s['actors'] == 1 and s['cameras'] == {'F': 1} and s['lidars'] == {'L1': 1}
    (out / 'ground_truth' / 'index.json').write_text(json.dumps(dict(doc, schema='other/1')))
    assert GT.read(str(out)) is None and GT.read(str(tmp_path / 'none')) is None


# ======================================================================= viewing (Simulate ▸ CARLA ▸ Ground truth)
def test_viewer_colours_and_the_range_scale():
    """The label images in colour: the semantic one is the classes table's palette pixel for pixel, the classes
    present counted exactly; one colour per object id, the same on every call, none for 0, ids next to each other
    apart; the range on a log scale: 1 at the near end, 0 at the far one, ½ at their geometric mean, NaN where there is
    no range; round limits and ticks."""
    sem = np.zeros((4, 6), np.uint8)
    sem[:, 2:] = GT.CLASS_ID['road']
    sem[0, 5] = GT.CLASS_ID['car']
    sem[1, 0] = GT.EGO_CLASS
    rgb = GT.colour_semantic(sem)
    assert rgb.shape == (4, 6, 3) and rgb.dtype == np.uint8
    assert tuple(rgb[2, 3]) == (128, 64, 128) and tuple(rgb[0, 5]) == (0, 0, 142)
    assert tuple(rgb[1, 0]) == (0, 255, 255)
    cl = GT.classes_in(sem)
    assert [c['name'] for c in cl] == ['road', 'unlabelled', 'car', 'ego']           # a tie: by class id
    assert [c['pixels'] for c in cl] == [15, 7, 1, 1] and math.isclose(sum(c['share'] for c in cl), 1.0)
    ids = np.arange(0, 301)
    cols = GT.object_colours(ids)
    assert (cols[0] == 0).all() and (cols[1:].max(axis=1) > 100).all()
    assert np.array_equal(cols, GT.object_colours(ids)) and np.array_equal(GT.object_colours([7])[0], cols[7])
    d = np.abs(cols[1:-1].astype(int) - cols[2:].astype(int)).sum(axis=1)
    assert d.min() >= 60                                          # neighbouring ids never look alike
    inst = np.array([[0, 5, 5], [1004, 0, 5]], np.uint16)
    ci = GT.colour_instance(inst)
    assert (ci[0, 0] == 0).all() and np.array_equal(ci[0, 1], GT.object_colours([5])[0]) and \
        np.array_equal(ci[1, 0], GT.object_colours([1004])[0])
    rng = np.array([[0.0, 1.0, 10.0, 100.0, 300.0, 0.5]], np.float32)
    t = GT.range_position(rng, 1.0, 100.0)
    assert np.isnan(t[0, 0]) and t[0, 1] == 1.0 and math.isclose(t[0, 2], 0.5, abs_tol=1e-6) and t[0, 3] == 0.0
    assert t[0, 4] == 0.0 and t[0, 5] == 1.0
    r = np.random.default_rng(3).uniform(2.3, 87.0, 10000).astype(np.float32)
    lo, hi = GT.range_limits(np.concatenate([r, np.zeros(500, np.float32)]))
    assert lo in (2.0, 1.0) and hi == 100.0 and GT.range_limits(np.zeros((3, 3))) == (1.0, 100.0)
    assert GT.scale_ticks(0.5, 200.0) == [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0]
    pal = GT.lidarseg_palette()
    assert tuple(pal[GT.NUSC_INDEX['vehicle.car']]) == (0, 0, 142)
    assert tuple(pal[GT.NUSC_INDEX['human.pedestrian.child']]) == (220, 20, 60)       # the pedestrian's
    assert tuple(pal[GT.NUSC_INDEX['vehicle.bicycle']]) == (119, 11, 32)              # the bicycle's, not the rider's
    assert tuple(pal[GT.NUSC_INDEX['noise']]) == GT.NOISE_RGB and tuple(pal[GT.NUSC_INDEX['vehicle.ego']]) == \
        (0, 255, 255)


def test_lidar_sweep_through_its_calibration_seen_from_above(tmp_path):
    """A sweep's points read from its nuScenes .pcd.bin through its calibrated_sensor (a lidar turned 90° to the left
    and raised: each point R·p + t, written independently here), the sweep nearest a time, a footprint of a turned box,
    and the top view: each point at its pixel, the higher point over a lower one, a dot of 2 × 2."""
    V = tmp_path / 'nuscenes' / 'v1.0-vantage'
    V.mkdir(parents=True)
    (tmp_path / 'nuscenes' / 'sweeps' / 'LIDAR_L1').mkdir(parents=True)
    (tmp_path / 'nuscenes' / 'lidarseg' / 'v1.0-vantage').mkdir(parents=True)
    pts = np.array([[10.0, 0.0, -1.5, 3, 0], [0.0, -4.0, 0.5, 5, 1], [0.0, -4.0, -1.0, 6, 2], [55.0, 0, 0, 1, 1]],
                   np.float32)
    fn = 'sweeps/LIDAR_L1/x__LIDAR_L1__2000.pcd.bin'
    pts.tofile(str(tmp_path / 'nuscenes' / fn))
    lab = np.array([24, 17, 30, 28], np.uint8)
    lab.tofile(str(tmp_path / 'nuscenes' / 'lidarseg' / 'v1.0-vantage' / 'sdB_lidarseg.bin'))
    c, s_ = math.cos(math.radians(45)), math.sin(math.radians(45))             # 90° about z: (w, x, y, z)
    (V / 'sample_data.json').write_text(json.dumps([
        {'token': 'sdA', 'filename': 'sweeps/LIDAR_L1/x__LIDAR_L1__1000.pcd.bin', 'calibrated_sensor_token': 'csA'},
        {'token': 'sdB', 'filename': fn, 'calibrated_sensor_token': 'csA'}]))
    (V / 'calibrated_sensor.json').write_text(json.dumps([{'token': 'csA', 'translation': [1.0, 0.5, 2.0],
                                                           'rotation': [c, 0.0, 0.0, s_]}]))
    doc = {'_root': str(tmp_path), 'lidars': {'L1': {'sweeps': [
        {'sample_data_token': 'sdA', 'timestamp': 1000, 'lidarseg': 'nuscenes/lidarseg/v1.0-vantage/sdA_lidarseg.bin'},
        {'sample_data_token': 'sdB', 'timestamp': 2000, 'lidarseg': 'nuscenes/lidarseg/v1.0-vantage/sdB_lidarseg.bin'}]}}}
    assert GT.sweep_near(doc, 'L1', 1400)['sample_data_token'] == 'sdA'
    sw = GT.sweep_near(doc, 'L1', 1600)
    assert sw['sample_data_token'] == 'sdB' and GT.sweep_near(doc, 'L9', 0) is None
    P = GT.lidar_points(doc, sw)
    R = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    assert np.allclose(P, pts[:, :3] @ R.T + [1.0, 0.5, 2.0], atol=1e-5)
    assert np.array_equal(GT.load_lidarseg(doc, sw), lab)
    # the top view: 0.1 m per pixel; x forward up, y left to the left; the origin in the middle
    img = GT.topdown(P, GT.lidarseg_palette()[lab], half_m=20.0, px=400)
    hit = np.argwhere(img[..., 3] > 0)
    assert len(hit) == 2                                   # two points share a pixel; one is beyond the edge
    # point 0: (1, 10.5, 0.5) -> row 200 - 10, col 200 - 105; points 1 and 2 meet at (5, 0.5): the higher (car) wins
    assert tuple(img[190, 95, :3]) == tuple(GT.lidarseg_palette()[24])
    assert tuple(img[150, 195, :3]) == tuple(GT.lidarseg_palette()[17])
    big = GT.topdown(P, GT.lidarseg_palette()[lab], half_m=20.0, px=400, dot=2)
    assert (big[..., 3] > 0).sum() == 8 and tuple(big[151, 196, :3]) == tuple(GT.lidarseg_palette()[17])
    fp = GT.box_footprint({'center': [5.0, 2.0, 0.7], 'size': [4.0, 2.0, 1.5], 'yaw_deg': 90.0})
    assert np.allclose(fp, [[4.0, 4.0], [4.0, 0.0], [6.0, 0.0], [6.0, 4.0]], atol=1e-9)
