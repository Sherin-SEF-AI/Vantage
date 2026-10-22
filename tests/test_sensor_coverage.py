"""Lidar and radar coverage: ground rings and density of a lidar at a known height, points on objects counted against
the beam geometry (tilt included), what the roof blocks, radar range against the cross-section and the bumper fascia,
sector verdicts against the perception range's required ranges, the camera-lidar overlap, settings, cancel, the
export, and the page (offscreen)."""
import json
import math
import os

import numpy as np
import pytest

from vantage.core import samples, sensor_coverage as SC, sensors
from vantage.core.analysis import Engine, camera_frames
from vantage.core.lens import resolve
from vantage.core.project import Camera, Project, Sensor


def _lidar(model_id='ouster_os1_128', tag='L1', **pose):
    return Sensor(kind='lidar', model_id=model_id, tag=tag, **pose)


def _radar(model_id='radar_long', tag='RD1', **pose):
    return Sensor(kind='radar', model_id=model_id, tag=tag, **pose)


def _rig(p):
    """One 128-channel lidar on the roof, a long-range radar in the front bumper, corner radars at the rear corners."""
    m = p.sensor_models['ouster_os1_128']
    p.variant.sensors.append(_lidar(**sensors.default_pose('lidar', m, p.vehicle.mesh, p.rig)))
    p.variant.sensors.append(_radar(**sensors.default_pose('radar', p.sensor_models['radar_long'], p.vehicle.mesh)))
    lo, hi = p.vehicle.mesh.bbox
    for tag, y, yaw in (('RD2', hi[1] - 0.2, 135.0), ('RD3', lo[1] + 0.2, -135.0)):
        p.variant.sensors.append(_radar('radar_corner', tag, x=float(lo[0]) - 0.02, y=float(y), z=0.5, yaw=yaw))
    return p


def _empty(*sens, cams=()):
    """A project without a vehicle: nothing blocks, the centre is the origin."""
    p = Project()
    p.variant.sensors.extend(sens)
    p.variant.cameras.extend(cams)
    return p


@pytest.fixture(scope='module')
def sample():
    p = _rig(samples.tigor_sample_project())
    e = Engine()
    return p, e, SC.run(p, e)


CAR = SC.CoverageClass('Car', 1.5, 1.8, 0.0, 20.0, 10.0)


# ----------------------------------------------------------------------------- lidar geometry
def test_ground_ring_and_density_of_a_level_lidar_at_a_known_height():
    h = 2.0
    p = _empty(_lidar(x=0.0, y=0.0, z=h))
    res = SC.run(p, Engine(), settings=SC.CoverageSettings.from_dict({'classes': [SC.DEFAULT_CLASSES[0]]}))
    m = p.sensor_models['ouster_os1_128']
    E = m.elevations()
    li = res.lidars[0]
    ring = h / math.tan(math.radians(abs(E[0])))
    assert li.ring_m == pytest.approx(ring, rel=1e-9)
    assert li.blocked == 0.0 and not li.blocked_channel.any()
    # nothing blocks: the nearest ground point in every direction is the lowest ring (no body: from the centre)
    np.testing.assert_allclose(res.near, ring, rtol=2e-3)
    np.testing.assert_allclose(res.blind, res.near)
    # every traced beam stands for step / h_res real points: the rings inside the map carry a frame's points each
    x0, y0, cell = res.grid
    half = res.settings.grid_extent / 2
    radii = np.array([h / math.tan(math.radians(-e)) for e in E if e < 0])
    per_ring = m.columns()
    assert per_ring == 1024
    total = float(res.density.sum()) * cell * cell
    full, part = int((radii < half - cell).sum()), int(((radii >= half - cell) & (radii < half * math.sqrt(2))).sum())
    assert full * per_ring * 0.999 <= total <= (full + part) * per_ring * 1.001
    assert li.points_per_frame == 128 * 1024


def test_points_on_an_object_follow_the_beams_and_the_tilt():
    """A car straight ahead of a level lidar 2 m up: the channels between its bottom and top elevation, times its
    angular width over h_res. Tilted 5° down, the same car sits 5° higher in the lidar's own frame."""
    h = 2.0
    e = Engine()
    m = Project().sensor_models['ouster_os1_128']
    E = m.elevations()
    for pitch in (0.0, 5.0):
        s = _lidar(x=0.0, y=0.0, z=h, pitch=pitch)
        p = _empty(s)
        rr = np.array([10.0, 30.0, 60.0])
        pts = SC.object_points(p, e, s, CAR, np.c_[rr, np.zeros(3)])
        for r, n in zip(rr, pts):
            lo = math.degrees(math.atan2(0.0 - h, r)) + pitch
            hi = math.degrees(math.atan2(CAR.height - h, r)) + pitch
            ch = int(((E >= lo - 1e-9) & (E <= hi + 1e-9)).sum())
            assert ch > 0
            assert n == pytest.approx(ch * math.degrees(2 * math.atan(CAR.width / 2 / r)) / m.h_res, rel=1e-6)
            assert n == pytest.approx(ch * math.degrees(CAR.width / r) / m.h_res, rel=5e-3)    # ≈ channels × w/r / h_res
    # out of range on a 10 % target: no points
    assert SC.object_points(p, e, s, CAR, np.array([[m.range_m + 5.0, 0.0]]))[0] == 0.0


def test_lidar_range_is_where_the_points_fall_below_the_minimum():
    s = _lidar(x=0.0, y=0.0, z=2.0)
    p = _empty(s)
    e = Engine()
    for need in (20.0, 60.0):
        st = SC.CoverageSettings.from_dict({'classes': [dict(SC.DEFAULT_CLASSES[0], min_points=need)]})
        res = SC.run(p, e, settings=st)
        far = res.lidar[0, 0]
        assert 10.0 < far < 90.0
        pts = SC.object_points(p, e, s, CAR, np.array([[far, 0.0], [far + 0.01, 0.0]]))
        assert pts[0] >= need > pts[1]                       # found between the samples, not at the last one
        assert res.lidars[0].far[0, 0] == pytest.approx(far)
        assert np.isnan(res.camera).all() and np.isnan(res.radar).all()   # no camera, no radar
    assert SC.run(p, e, settings=SC.CoverageSettings.from_dict(
        {'classes': [dict(SC.DEFAULT_CLASSES[0], min_points=20.0)]})).lidar[0, 0] > far    # fewer points: farther


def test_the_roof_blocks_the_downward_beams_of_a_roof_lidar(sample):
    p, e, res = sample
    li = res.lidars[0]
    assert li.sensor.z == pytest.approx(float(p.vehicle.mesh.bbox[1][2]) + 0.036, abs=0.03)    # just above the roof
    assert li.blocked_channel[0] == 1.0 and li.blocked_channel[-1] == 0.0
    assert 0.2 < li.blocked < 0.6
    assert li.blocked == pytest.approx(float(np.mean(li.blocked_channel)))
    assert all(v is not None and v > 0.2 for v in li.blocked_sectors.values())
    # its lowest ring would lie 3.8 m out, but the roof hides the near ground: the blind zone is far larger
    assert li.ring_m == pytest.approx(li.sensor.z / math.tan(math.radians(22.5)), rel=1e-6)
    assert np.nanmedian(li.blind) > 5.0 and np.isfinite(li.blind).all()
    # the same lidar 0.6 m higher (on a mast) loses fewer beams to the roof and sees the ground nearer the car
    q = p.snapshot()
    q.variant.sensors[0].z += 0.6
    up = SC.run(q, e).lidars[0]
    assert up.blocked < 0.5 * li.blocked
    assert np.median(up.blind) < 0.6 * np.median(li.blind)
    # points on a car ahead: about 1/r² (a quarter at twice the distance), none beyond the range
    c = li.car_points
    assert c[25.0] > 60 and c[50.0] == pytest.approx(c[25.0] / 4, rel=0.25) and c[100.0] == 0.0


# ----------------------------------------------------------------------------- radar
def test_radar_range_scales_with_the_fourth_root_of_the_cross_section():
    s = _radar('radar_corner', x=0.0, y=0.0, z=0.5)               # one zone: 100 m on a car, ±75°, ±10°
    p = _empty(s)
    res = SC.run(p, Engine())
    names = [c.name for c in res.classes]
    car, ped, cyc, sign = (names.index(n) for n in ('Car', 'Pedestrian', 'Cyclist', 'Traffic sign'))
    zc = 0.75 - 0.5                                                  # the car's centre above the radar
    assert res.radar[car, 0] == pytest.approx(math.sqrt(100.0 ** 2 - zc ** 2), abs=1e-3)
    for k, rcs in ((ped, 0.5), (cyc, 1.0)):
        assert res.radar[k, 0] / res.radar[car, 0] == pytest.approx((rcs / 10.0) ** 0.25, rel=1e-4)
    a60 = res.az_index(60.0)
    assert res.radar[car, a60] == pytest.approx(res.radar[car, 0], rel=1e-4)      # inside ±75°
    assert res.radar[car, res.az_index(90.0)] == 0.0 and res.radar[car, res.az_index(180.0)] == 0.0
    assert np.isnan(res.radar[sign]).all()                          # rcs 0: radars are not relied on
    assert res.sectors[0]['classes']['Traffic sign']['radar_m'] is None
    assert res.radars[0].far[car, 0] == pytest.approx(res.radar[car, 0])


def test_a_radar_behind_the_bumper_fascia_still_sees_but_one_inside_the_body_does_not(sample):
    p0, e, _ = sample
    p = p0.snapshot()
    p.variant.sensors = [s for s in p.variant.sensors if s.tag == 'RD1']
    rd = p.variant.sensors[0]
    st = SC.CoverageSettings.from_dict({'classes': [SC.DEFAULT_CLASSES[0]]})
    out = SC.run(p, e, settings=st)
    assert out.radar[0, 0] == pytest.approx(250.0) and out.metrics['radars'] == 1 and out.lidars == []
    front = float(p.vehicle.mesh.bbox[1][0])
    rd.x = front - 0.08                                             # 8 cm into the bumper: the fascia is ignored
    assert SC.run(p, e, settings=st).radar[0, 0] == pytest.approx(250.0)
    rd.x = front - 0.6                                              # in the engine bay: the bumper is in the way
    assert SC.run(p, e, settings=st).radar[0, 0] == 0.0


# ----------------------------------------------------------------------------- the study
def test_sectors_take_the_shortest_range_and_the_verdict_follows_the_required_ranges(sample):
    p, e, res = sample
    assert [s['name'] for s in res.sectors] == [n for n, _ in SC.PR.SECTORS]
    assert res.metrics['checks'] == 8 * len(res.classes) == sum(res.metrics[v] for v in SC.VERDICTS)
    sec = SC._sector_of(res.az)
    for si, s in enumerate(res.sectors):
        d = s['classes']['Car']
        assert d['lidar_m'] == pytest.approx(float(res.lidar[0, sec == si].min()))
        assert d['camera_m'] == pytest.approx(float(res.camera[0, sec == si].min()))
        assert d['radar_m'] == pytest.approx(float(res.radar[0, sec == si].min()))
        assert s['required_m'] == SC.PR.DEFAULT_REQUIRED[s['name']]
    # the sector whose three ranges lie farthest apart: require less than all, between the two longest, more than all
    spread = [sorted([s['classes']['Car'][f'{m}_m'] for m in SC.MODALITIES]) for s in res.sectors]
    si = int(np.argmax([v[2] - v[1] for v in spread]))
    lo, mid, hi = spread[si]
    assert hi - mid > 1.0
    name = res.sectors[si]['name']
    for req, verdict, n in ((lo - 1.0, 'redundant', 3), ((mid + hi) / 2, 'single', 1), (hi + 1.0, 'gap', 0)):
        q = p.snapshot()
        q.analysis.perception = {'required': {name: req}}
        d = SC.run(q, e).sectors[si]['classes']['Car']
        assert d['verdict'] == verdict and d['modalities'] == n and sum(d['meets'].values()) == n
    # the same requirement reaches every class: pedestrians in the same sector
    assert res.sectors[si]['classes']['Pedestrian']['verdict'] in SC.VERDICTS
    out = SC.summary(res)
    assert out['sectors']['Car'][name]['verdict'] == res.sectors[si]['classes']['Car']['verdict']


def test_results_shapes_and_speed(sample):
    p, e, res = sample
    K, A, R = len(res.classes), len(res.az), len(res.ranges)
    assert res.camera.shape == res.lidar.shape == res.radar.shape == (K, A) and res.points.shape == (K, A, R)
    assert res.metrics['lidars'] == 1 and res.metrics['radars'] == 3 and res.metrics['cameras'] == 6
    # the cameras' range is the perception range of the class of the same name
    from vantage.core import perception as PR
    pr = PR.analyse(p, e)
    np.testing.assert_allclose(res.camera[0], pr.far[0])
    assert res.at(0, 0.0)['camera'] == pytest.approx(pr.at(0, 0.0)[0])
    # lidar points are summed over the lidars and grow towards the vehicle
    a = res.az_index(0.0)
    i10, i50 = (int(np.argmin(np.abs(res.ranges - r))) for r in (10.0, 50.0))
    assert res.points[0, a, i10] > res.points[0, a, i50] > 0
    assert res.points_at(0, 0.0, 50.0) == pytest.approx(float(res.points[0, a, i50]))
    assert res.density.max() > 0 and res.blind.shape == (A,) and (res.outline > 0.5).all()
    again = SC.run(p, e)                                             # warm: the cube maps are cached
    assert again.metrics['seconds'] < 3.0
    np.testing.assert_allclose(again.lidar, res.lidar)


def test_a_variant_other_than_the_active_one():
    p = samples.tigor_sample_project()
    v2 = p.variants[1]
    v2.sensors.append(_lidar(x=1.0, y=0.0, z=2.2))
    active = p.active_uid
    res = SC.run(p, Engine(), variant=v2)
    assert p.active_uid == active and res.variant_uid == v2.uid
    assert res.metrics['lidars'] == 1 and res.metrics['cameras'] == sum(c.visible for c in v2.cameras)
    assert SC.run(p, Engine()).metrics['lidars'] == 0                  # the active variant has none
    assert SC.to_doc(res, p)['variant'] == v2.name != p.variant.name


def test_camera_lidar_overlap_of_a_lidar_next_to_the_camera():
    """A level narrow camera and a spinning lidar at the same place: the lidar covers the image except where it is
    taller than the lidar's ±22.5°; its rings across the centre column are the channels within the column's span. A
    fixed-field lidar facing backwards covers nothing of it."""
    cam = Camera(tag='F', x=0.0, y=0.0, z=2.0, yaw=0.0, pitch=0.0, roll=0.0, lens_id='h03r2')
    s = _lidar(x=0.0, y=0.0, z=2.0)
    p = _empty(s, cams=[cam])
    e = Engine()
    res = SC.run(p, e, settings=SC.CoverageSettings.from_dict({'classes': [SC.DEFAULT_CLASSES[0]]}))
    o = res.overlap[0]
    assert o['tag'] == 'F' and o['best'] == 'L1' and o['per_lidar'] == {'L1': pytest.approx(o['share']['20'])}
    assert 0.75 < o['share']['20'] <= 1.0 and o['share']['10'] == pytest.approx(o['share']['50'], abs=0.02)
    intr = resolve(p.module_for(cam), p.lens_for(cam))
    d, th = intr.unproject(np.array([intr.cx, intr.cx]), np.array([0.0, float(intr.height)]))
    el = np.degrees(np.arcsin((d @ camera_frames(cam)[1])[:, 2]))
    E = p.sensor_models['ouster_os1_128'].elevations()
    expect = int(((E >= max(el.min(), E[0])) & (E <= min(el.max(), E[-1]))).sum())
    assert abs(o['rings'] - expect) <= 1
    s.model_id, s.yaw = 'hesai_at128', 180.0
    o = SC.run(p, e).overlap[0]
    assert o['share'] == {'10': 0.0, '20': 0.0, '50': 0.0} and o['best'] is None and o['rings'] == 0


# ----------------------------------------------------------------------------- settings, cancel
def test_settings_round_trip_and_reject_bad_values():
    st = SC.CoverageSettings.from_dict({'classes': [dict(SC.DEFAULT_CLASSES[0], min_points=33, rcs=2.5)],
                                        'max_range': 120, 'azimuths': 180, 'grid_cell': 0.5})
    again = SC.CoverageSettings.from_dict(json.loads(json.dumps(st.to_dict())))
    assert again == st and st.classes[0]['min_points'] == 33 and st.azimuths == 180 and st.max_range == 120.0
    assert SC.CoverageSettings.from_dict({}) == SC.CoverageSettings() == SC.CoverageSettings.from_dict(None)
    assert SC.CoverageSettings.from_dict('nonsense') == SC.CoverageSettings()
    bad = [dict(SC.DEFAULT_CLASSES[0], min_points=0), dict(SC.DEFAULT_CLASSES[0], min_points=-3),
           dict(SC.DEFAULT_CLASSES[0], rcs=-1), dict(SC.DEFAULT_CLASSES[0], min_points=float('nan')),
           dict(SC.DEFAULT_CLASSES[0], height='tall'), {'name': 'x'}, None, 5]
    assert all(SC.clean_class(c) is None for c in bad)
    st = SC.CoverageSettings.from_dict({'classes': bad, 'max_range': 5, 'azimuths': 'many', 'samples': 1e9,
                                        'grid_extent': float('inf'), 'grid_cell': -1})
    assert [c['name'] for c in st.classes] == [c['name'] for c in SC.DEFAULT_CLASSES]
    assert st.max_range == 10.0 and st.azimuths == 360 and st.samples == 400
    assert st.grid_extent == 40.0 and st.grid_cell == 0.1
    st = SC.CoverageSettings.from_dict({'classes': [SC.DEFAULT_CLASSES[0], SC.DEFAULT_CLASSES[0]]})
    assert [c['name'] for c in st.classes] == ['Car', 'Car 2']
    # a perception class (no min_points, no rcs) is usable with typical figures; rcs 0 is allowed
    assert SC.clean_class({'name': 'Bus', 'height': 3.0, 'width': 2.5}) == \
        {'name': 'Bus', 'height': 3.0, 'width': 2.5, 'base': 0.0, 'min_points': 10.0, 'rcs': 1.0}
    assert SC.clean_class(dict(SC.DEFAULT_CLASSES[3]))['rcs'] == 0.0
    # the map never grows past 800 × 800 cells
    st = SC.CoverageSettings.from_dict({'grid_extent': 200, 'grid_cell': 0.1})
    assert st.grid_extent / st.grid_cell <= 800 + 1e-9


def test_cancel_and_progress(sample):
    p, e, _ = sample
    steps = []
    assert SC.run(p, e, progress=steps.append, cancel=lambda: len(steps) >= 2) is None
    steps = []
    assert SC.run(p, e, progress=steps.append) is not None
    assert steps == sorted(steps) and steps[-1] == 1.0 and len(steps) > 4
    # a cancel during the cameras' part stops the perception range too
    calls = []
    assert SC.run(p, e, progress=calls.append, cancel=lambda: 0.1 < (calls[-1] if calls else 0) < 0.4) is None


# ----------------------------------------------------------------------------- output
def test_summary_has_figures_but_no_paths_or_streams(sample):
    p0, e, res = sample
    p = p0.snapshot()
    p.path = '/home/someone/secret/rig.vantage'
    p.streams = {'F': 'rtsp://user:pw@10.0.0.2/stream'}
    out = SC.summary(SC.run(p, e))
    text = json.dumps(out)
    assert '://' not in text and 'secret' not in text and 'rtsp' not in text
    assert out['checks'] == 32 and set(out['verdicts']) == set(SC.VERDICTS)
    assert out['lidars'][0]['lidar'] == 'L1' and 0 < out['lidars'][0]['blocked_pct'] < 100
    assert out['lidars'][0]['ground_blind_from_body']['median_m'] > 0
    assert [o['camera'] for o in out['camera_lidar_overlap']] == [c.tag for c in p.variant.cameras if c.visible]
    assert {r['radar'] for r in out['radars']} == {'RD1', 'RD2', 'RD3'}


def test_export_json_and_png(sample, tmp_path):
    import cv2
    p, e, res = sample
    paths = SC.export(res, p, str(tmp_path), units='mm')
    assert [os.path.basename(x) for x in paths] == ['sensor_coverage.json', 'sensor_coverage.png']
    doc = json.loads((tmp_path / 'sensor_coverage.json').read_text())
    assert doc['schema'] == SC.SCHEMA == 'vantage.sensor_coverage/1'
    assert doc['units'] == 'mm' and doc['frame'] == p.frame and doc['variant'] == p.variant.name
    assert doc['required_ranges']['front'] == 80000 and len(doc['azimuth']) == 360
    assert [c['name'] for c in doc['classes']] == ['Car', 'Pedestrian', 'Cyclist', 'Traffic sign']
    assert doc['classes'][0]['height'] == 1500 and doc['classes'][0]['min_points'] == 20
    car = doc['per_class']['Car']
    assert len(car['camera']) == len(car['lidar']) == len(car['radar']) == 360 and max(car['radar']) == 250000
    assert doc['per_class']['Traffic sign']['radar'] == [None] * 360
    li = doc['lidars'][0]
    s = p.variant.sensors[0]
    o = p.origin_offset()
    assert li['tag'] == 'L1' and li['position'] == [round((s.x - o[0]) * 1000), round((s.y - o[1]) * 1000),
                                                    round((s.z - o[2]) * 1000)]
    assert len(li['blocked_per_channel']) == len(li['channel_elevation']) == 128
    assert li['nearest_ring'] == round(res.lidars[0].ring_m * 1000)
    assert {r['tag'] for r in doc['radars']} == {'RD1', 'RD2', 'RD3'}
    assert [c['camera'] for c in doc['camera_lidar']['cameras']] == [c.tag for c in p.variant.cameras if c.visible]
    g = doc['ground_density']
    assert len(g['values']) == len(g['values'][0]) == 160 and g['cell'] == 250
    assert doc['sectors'][0]['classes']['Car']['verdict'] in SC.VERDICTS
    img = cv2.imread(str(tmp_path / 'sensor_coverage.png'))
    assert img is not None and img.shape[1] == 900 and img.shape[0] > 900
    rings = SC.png(res, size=600, k_sel=1, layer='rings')
    assert cv2.imdecode(np.frombuffer(rings, np.uint8), cv2.IMREAD_COLOR).shape[1] == 600
    # in the default frame and metres, the vehicle centre is where the study put it
    doc = SC.to_doc(res, p)
    assert doc['centre'] == [round(res.centre[0], 4), round(res.centre[1], 4)]


# ----------------------------------------------------------------------------- the page, driven like a user
from test_gui import app, win, pump, settle, _wait     # noqa: E402,F401  (fixtures)


def _open(app, win):
    menus = {a.text().replace('&', ''): a.menu() for a in win.menuBar().actions()}
    act = win.actions_by_id['sensor_coverage']
    assert act in menus['Analysis'].actions()
    act.trigger()                                        # Analysis ▸ Lidar and radar coverage…
    pump(app, 0.2)
    pg = win.page('coverage')
    assert win.page_name() == 'coverage' and pg.isVisible()
    return pg


def _type_into_cell(app, table, row, col, text):
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QApplication, QLineEdit
    at = table.visualItemRect(table.item(row, col)).center()
    QTest.mouseClick(table.viewport(), Qt.MouseButton.LeftButton, pos=at)
    QTest.mouseDClick(table.viewport(), Qt.MouseButton.LeftButton, pos=at)
    pump(app, 0.1)
    ed = QApplication.focusWidget()
    assert isinstance(ed, QLineEdit)
    QTest.keyClick(ed, Qt.Key.Key_A, Qt.KeyboardModifier.ControlModifier)
    QTest.keyClicks(ed, text)
    QTest.keyClick(ed, Qt.Key.Key_Return)
    pump(app, 0.1)


def test_page_runs_edits_undoes_and_stops_when_left(app, win, tmp_path):
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest
    _rig(win.store.project)                              # the rig's sensors, as the Vehicle menu would add them
    pg = _open(app, win)
    ctl = win.ctx.sensor_coverage
    _wait(app, lambda: ctl.result is not None and not ctl.busy, 60)      # a quick study runs when the page shows
    # Run: a main-window job (progress bar, Cancel)
    before = ctl.result
    QTest.mouseClick(pg.b_run, Qt.MouseButton.LeftButton)
    assert 'sensor_coverage' in win._jobs and not pg.b_run.isEnabled()
    _wait(app, lambda: ctl.result is not before and not ctl.busy, 60)
    pump(app, 0.1)
    res = ctl.result
    assert 'sector checks' in pg.status.text() and pg.b_run.isEnabled() and pg.b_export.isEnabled()
    assert pg.map.res is res and pg.grid.res is res
    assert pg.sectors.rowCount() == 8 and pg.lidars.rowCount() == 1 and pg.radars.rowCount() == 3
    assert pg.radars.columnCount() == 1 + sum(c.rcs > 0 for c in res.classes)
    assert pg.radars.item(0, 1).text() == '≥250'                        # the long-range radar: to the study's end
    assert pg.overlap.rowCount() == sum(c.visible for c in win.store.variant.cameras)
    assert pg.sectors.item(0, 5).text().lower() == res.sectors[0]['classes']['Car']['verdict']
    assert pg.lidars.item(0, 1).text() == f'{100 * res.lidars[0].blocked:.1f}'
    pg.layer.buttons['rings'].click()
    pg.reach.buttons['20'].click()
    pg.pick.buttons['1'].click()
    pump(app, 0.1)
    assert pg.map.layer == 'rings' and pg.map.rmax() == 20.0 and pg.map.k == 1 and pg.grid.k == 1
    assert pg.sector_head.text().startswith('Pedestrian')
    pg.map.grab()                                        # paints with the result (no exception)
    # the car's minimum points, typed into the table: saved with the project, re-run by itself, undoable
    far0 = res.at(0, 0.0)['lidar']
    _type_into_cell(app, pg.classes, 0, 2, '60')
    assert win.store.project.analysis.sensor_coverage['classes'][0]['min_points'] == 60
    _wait(app, lambda: ctl.result.classes[0].min_points == 60 and not ctl.busy and not ctl.stale, 60)
    assert ctl.result.at(0, 0.0)['lidar'] < 0.85 * far0 and pg.classes.item(0, 2).text() == '60'
    _type_into_cell(app, pg.classes, 0, 2, '-3')         # nonsense is refused; the table shows what is stored
    assert win.store.project.analysis.sensor_coverage['classes'][0]['min_points'] == 60
    assert pg.classes.item(0, 2).text() == '60'
    _type_into_cell(app, pg.classes, 1, 3, '0')          # pedestrians: radars are not relied on
    assert win.store.project.analysis.sensor_coverage['classes'][1]['rcs'] == 0
    _wait(app, lambda: ctl.result.classes[1].rcs == 0 and not ctl.busy and not ctl.stale, 60)
    assert ctl.result.at(1, 0.0)['radar'] is None
    win.store.undo.undo()
    pump(app, 0.1)
    assert ctl.settings.classes[1]['rcs'] == 0.5 and pg.classes.item(1, 3).text() == '0.5'
    win.store.undo.undo()
    pump(app, 0.1)
    assert win.store.project.analysis.sensor_coverage == {}             # back to the defaults, as it started
    assert pg.classes.item(0, 2).text() == '20'
    _wait(app, lambda: ctl.result.classes[0].min_points == 20 and ctl.result.classes[1].rcs == 0.5
          and not ctl.busy and not ctl.stale, 60)
    assert ctl.result.at(0, 0.0)['lidar'] == pytest.approx(far0)
    paths = pg.export(str(tmp_path))
    assert [os.path.basename(x) for x in paths] == ['sensor_coverage.json', 'sensor_coverage.png']
    assert json.loads((tmp_path / 'sensor_coverage.json').read_text())['schema'] == SC.SCHEMA
    # another page: no more re-runs, the result goes stale
    win.actions_by_id['blind_volume'].trigger()
    pump(app, 0.2)
    assert not ctl.active and not pg.isVisible()
    shown = ctl.result
    win.store.edit('analysis', lambda q: setattr(q.analysis, 'sensor_coverage', {'max_range': 150}), 'test')
    settle(app, win)
    pump(app, 0.5)
    assert ctl.stale and ctl.result is shown


def test_page_without_sensors_says_how_to_add_one(app, win):
    from vantage.ui.coverageview import EMPTY
    assert not win.store.variant.sensors
    pg = _open(app, win)
    ctl = win.ctx.sensor_coverage
    pump(app, 0.6)
    assert ctl.result is None and not ctl.busy
    assert pg.map.empty == EMPTY and EMPTY in pg.status.text() and not pg.b_run.isEnabled()
    pg.map.grab()
    # a lidar added while the page shows: it runs by itself; undone: the empty state again
    m = win.store.project.sensor_models['ouster_os1_128']
    s = _lidar(**sensors.default_pose('lidar', m, win.store.project.vehicle.mesh, win.store.project.rig))
    win.store.edit('variant', lambda q: q.variant.sensors.append(s), 'Added lidar L1')
    _wait(app, lambda: ctl.result is not None and not ctl.busy, 60)
    pump(app, 0.1)
    assert pg.b_run.isEnabled() and pg.lidars.rowCount() == 1 and pg.map.res is ctl.result
    win.store.undo.undo()
    _wait(app, lambda: ctl.result is None and not ctl.busy, 60)
    pump(app, 0.1)
    assert EMPTY in pg.status.text() and pg.lidars.rowCount() == 0 and not pg.b_run.isEnabled()
