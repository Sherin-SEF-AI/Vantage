"""Sync and timing plan: the lidar sweep against each camera (known geometry, parallax, both rotation senses), triggers
and rolling shutter, trigger groups and the suggested split over the lines, other lidars' phases, motion blur through
the lens, settings, the export, and the page (offscreen)."""
import copy
import json
import math
import os

import pytest

from vantage.core import samples, sensors, timing as TM
from vantage.core.lens import Lens, Module
from vantage.core.project import Camera, Project, Sensor


def _rig(cams, fps=10.0, lidar=None, model='ouster_os1_128'):
    """A bare project: level 6 mm pinhole cameras (tag, x, y, z, yaw) on a 1920 × 1536, 3 µm module at fps, and a
    spinning lidar at the origin, 1.8 m up (no vehicle)."""
    p = Project()
    p.modules['m'] = Module(id='m', name='Test module', width_px=1920, height_px=1536, pixel_um=3.0, fps=fps)
    p.lenses['pin'] = Lens(id='pin', name='Pinhole 6 mm', focal_mm=6.0, projection='rectilinear')
    v = p.variant
    for i, (tag, x, y, z, yaw) in enumerate(cams):
        v.cameras.append(Camera(tag=tag, slot=i, x=x, y=y, z=z, yaw=yaw, pitch=0.0, roll=0.0, module_id='m',
                                lens_id='pin'))
    v.sensors.append(Sensor(kind='lidar', model_id=model, tag='L1', x=0.0, y=0.0, z=1.8, **(lidar or {})))
    return p


def _ccw(p):
    m = copy.deepcopy(p.sensor_models['ouster_os1_128'])
    m.id, m.rotation, m.builtin = 'ccw_lidar', 'ccw', False
    p.sensor_models[m.id] = m
    return m.id


SIDES = [('F', 0.0, 0.0, 1.0, 0.0), ('L', 0.0, 0.5, 1.0, 90.0), ('R', 0.0, -0.5, 1.0, -90.0)]


def _apart(a, b, T=100.0):
    """Circular distance of two times in one period, ms."""
    return abs((a - b + T / 2) % T - T / 2)


@pytest.fixture(scope='module')
def tigor():
    p = samples.tigor_sample_project()
    m = p.sensor_models['ouster_os1_128']
    p.variant.sensors = [Sensor(kind='lidar', model_id='ouster_os1_128', tag='L1',
                                **sensors.default_pose('lidar', m, p.vehicle.mesh, p.rig))]
    return p


# ----------------------------------------------------------------------------- the sweep
def test_sweep_crosses_the_cameras_at_known_times():
    """A lidar at the origin spinning clockwise at 10 Hz with phase 0 faces forward at the sync pulse, the right side
    a quarter turn later and the left side three quarters later; counter-clockwise mirrors it."""
    res = TM.plan(_rig(SIDES))
    c = {e['tag']: e for e in res.cams}
    assert res.lidar.rate_hz == 10.0 and res.lidar.rotation == 'cw' and res.period_ms == pytest.approx(100.0)
    assert _apart(c['F']['cross_ms'], 0.0) < 1e-9
    assert c['L']['cross_ms'] == pytest.approx(75.0) and c['L']['azimuth_deg'] == pytest.approx(90.0)
    assert c['R']['cross_ms'] == pytest.approx(25.0) and c['R']['azimuth_deg'] == pytest.approx(-90.0)
    # the field: a 6 mm pinhole across 5.76 mm is 2·atan(2.88 / 6) wide; clockwise the sweep enters at its left edge
    half = math.degrees(math.atan(2.88 / 6.0))
    f = c['F']
    assert f['field_deg'] == pytest.approx(2 * half, abs=0.01) and f['left_deg'] == pytest.approx(half, abs=0.01)
    assert f['sweep_ms'] == pytest.approx(2 * half / 3.6, abs=0.01)
    assert f['enter_ms'] == pytest.approx((360.0 - half) / 3.6, abs=0.01)
    assert f['hfov_deg'] == pytest.approx(2 * half, abs=0.01)
    # counter-clockwise: the left side first, entering at the right edge
    p = _rig(SIDES)
    p.variant.sensors[0].model_id = _ccw(p)
    c = {e['tag']: e for e in TM.plan(p).cams}
    assert c['L']['cross_ms'] == pytest.approx(25.0) and c['R']['cross_ms'] == pytest.approx(75.0)
    assert c['F']['enter_ms'] == pytest.approx((360.0 - half) / 3.6, abs=0.01)
    # the phase turns the whole plan: facing 90° (left) at the sync pulse, clockwise
    p = _rig(SIDES, lidar={'phase_deg': 90.0})
    c = {e['tag']: e for e in TM.plan(p).cams}
    assert _apart(c['L']['cross_ms'], 0.0) < 1e-9 and c['F']['cross_ms'] == pytest.approx(25.0)


def test_parallax_and_the_lidars_own_yaw():
    """A camera 2 m ahead of the lidar looking left: its view 20 m out lies at atan(20 / 2) from the lidar, not 90°.
    A lidar turned round (yaw 180°) sees forward at its own azimuth 180°."""
    res = TM.plan(_rig([('S', 2.0, 0.0, 1.0, 90.0)]))
    az = math.degrees(math.atan2(20.0, 2.0))
    s = res.cams[0]
    assert s['azimuth_deg'] == pytest.approx(az) and s['cross_ms'] == pytest.approx((360.0 - az) / 3.6)
    res = TM.plan(_rig(SIDES, lidar={'yaw': 180.0}))
    c = {e['tag']: e for e in res.cams}
    assert abs(c['F']['azimuth_deg']) == pytest.approx(180.0) and c['F']['cross_ms'] == pytest.approx(50.0)
    assert float(res.lidar.direction(c['F']['cross_ms'])) == pytest.approx(0.0, abs=1e-9)
    assert float(res.lidar.direction(c['L']['cross_ms'])) == pytest.approx(90.0)


def test_trigger_centres_the_middle_row_on_the_sweep():
    """Alone on its line a camera is triggered readout/2 + exposure/2 before the crossing, modulo the period."""
    for ro, ex in ((20.0, 2.0), (0.0, 1.0), (30.0, 5.0)):
        st = TM.TimingSettings(readout_ms=ro, exposure_ms=ex)
        p = _rig(SIDES)
        for i, c in enumerate(p.variant.cameras):
            c.sync = 'ABC'[i]
        c = {e['tag']: e for e in TM.plan(p, settings=st).cams}
        for tag in 'FLR':
            want = (c[tag]['cross_ms'] - ro / 2 - ex / 2) % 100.0
            assert c[tag]['ideal_ms'] == pytest.approx(want) and c[tag]['trigger_ms'] == pytest.approx(want)
            assert c[tag]['misalign_ms'] == pytest.approx(0.0, abs=1e-9) and c[tag]['travel_cm'] == pytest.approx(0.0)
        assert _apart(c['F']['trigger_ms'], 100.0 - ro / 2 - ex / 2) < 1e-9


def test_cameras_on_one_line_share_a_trigger():
    """Two cameras on one line (10 fps: one frame per rotation) are 50 ms apart: the common trigger is half way and
    each is half the spacing off; the vehicle travels v·25 ms meanwhile. With F between them the line follows F."""
    p = _rig(SIDES[1:])
    res = TM.plan(p)
    (g,) = res.groups
    assert g['letter'] == 'A' and set(g['tags']) == {'L', 'R'} and g['worst_ms'] == pytest.approx(25.0)
    c = {e['tag']: e for e in res.cams}
    assert abs(c['L']['misalign_ms']) == pytest.approx(25.0) and abs(c['R']['misalign_ms']) == pytest.approx(25.0)
    assert _apart(c['L']['trigger_ms'], c['R']['trigger_ms']) < 1e-9
    assert c['L']['travel_cm'] == pytest.approx(25e-3 * 50 / 3.6 * 100)
    assert res.metrics['worst_ms'] == pytest.approx(25.0) and res.metrics['lines_used'] == 1
    res = TM.plan(_rig(SIDES))
    c = {e['tag']: e for e in res.cams}
    assert res.groups[0]['worst_ms'] == pytest.approx(25.0) and c['F']['misalign_ms'] == pytest.approx(0.0, abs=1e-9)
    assert c['L']['misalign_ms'] == pytest.approx(25.0) and c['R']['misalign_ms'] == pytest.approx(-25.0)
    # at 30 fps each camera fires three times a rotation: only the spacing modulo 33.3 ms counts
    res = TM.plan(_rig(SIDES[1:], fps=30.0))
    c = {e['tag']: e for e in res.cams}
    assert c['L']['ratio'] == pytest.approx(3.0) and c['L']['cycle_ms'] == pytest.approx(100 / 3)
    assert res.groups[0]['worst_ms'] == pytest.approx(50.0 % (100 / 3) / 2)
    assert res.groups[0]['period_ms'] == pytest.approx(100 / 3)


def test_suggested_groups_split_the_lines():
    """Six cameras round the lidar, 60° apart (10 fps): one line leaves them up to 41.7 ms off; pairs of neighbours
    on three lines 8.3 ms, and a fourth line does not do better, so the suggestion uses three; six lines: none off."""
    cams = [(f'C{i}', 0.0, 0.0, 1.0, 60.0 * i) for i in range(6)]
    res = TM.plan(_rig(cams))
    assert res.metrics['worst_ms'] == pytest.approx(500 / 12)
    letters = res.suggestion['letters']
    assert set(letters) == {c['uid'] for c in res.cams} and len(set(letters.values())) == 3
    assert res.suggestion['lines'] == 3 and res.suggestion['worst_ms'] == pytest.approx(100 / 12)
    groups, worst = TM.evaluate_groups(res, letters)
    assert worst == pytest.approx(100 / 12) and all(len(g['uids']) == 2 for g in groups)
    for g in groups:                                     # neighbours: 60° apart, the sweep 16.7 ms apart
        a, b = (res.cam(u) for u in g['uids'])
        assert abs((a['cross_ms'] - b['cross_ms'] + 50) % 100 - 50) == pytest.approx(100 / 6)
    assert sorted(letters.values()) == ['A', 'A', 'B', 'B', 'C', 'C']
    assert TM.evaluate_groups(res, TM.suggest_groups(res, 6))[1] == pytest.approx(0.0, abs=1e-9)
    assert TM.evaluate_groups(res, TM.suggest_groups(res, 1))[1] == pytest.approx(500 / 12)
    assert len(set(TM.suggest_groups(res, 6).values())) == 6


def test_suggested_groups_on_the_sample(tigor):
    """The Tigor's six fisheyes at 30 fps against a 10 Hz OS1 on the roof, all on line A: four lines cut the worst
    misalignment; the letters cover every camera and use at most four lines."""
    res = TM.plan(tigor)
    before = res.metrics['worst_ms']
    assert res.metrics['lines_used'] == 1 and before > 5.0
    letters = TM.suggest_groups(res, 4)
    assert set(letters) == {c.uid for c in tigor.variant.cameras} and len(set(letters.values())) <= 4
    _, after = TM.evaluate_groups(res, letters)
    assert after < before / 3 and after == pytest.approx(res.suggestion['worst_ms'])
    _, two = TM.evaluate_groups(res, TM.suggest_groups(res, 2))
    assert after <= two <= before
    # applied, the plan agrees with the evaluation
    q = copy.deepcopy(tigor)
    for c in q.variant.cameras:
        c.sync = letters[c.uid]
    out = TM.plan(q)
    assert out.metrics['worst_ms'] == pytest.approx(after) and out.metrics['lines_used'] == len(set(letters.values()))
    assert TM.suggest_groups(out, 4) == letters
    # the travel during one rotation, for deskewing the point cloud
    assert res.metrics['rotation_travel_m'] == pytest.approx(50 / 3.6 * 0.1)


def test_phase_for_a_second_lidar():
    """A second lidar of the same model turned round (yaw 180°) must lead by half a turn to face where the reference
    faces: phase 180°; at phase 0 the two sweep a direction 50 ms apart."""
    p = _rig(SIDES)
    p.variant.sensors.append(Sensor(kind='lidar', model_id='ouster_os1_128', tag='L2', x=1.0, y=0.5, z=1.5, yaw=180.0))
    res = TM.plan(p)
    (d,) = res.lidars
    assert d['tag'] == 'L2' and d['same_sense'] and d['same_rate']
    assert d['suggested_phase_deg'] == pytest.approx(180.0)
    assert d['misalign_ms'] == pytest.approx(50.0) and d['after_ms'] == pytest.approx(0.0, abs=1e-9)
    assert TM.suggested_phases(res) == {d['uid']: pytest.approx(180.0)}
    p.variant.sensors[1].phase_deg = 180.0
    assert TM.plan(p).lidars[0]['misalign_ms'] == pytest.approx(0.0, abs=1e-9)
    # the reference at phase 30°, the second yawed 90°: it must face 30 - 90 in its own frame
    p.variant.sensors[0].phase_deg = 30.0
    p.variant.sensors[1].yaw = 90.0
    assert TM.plan(p).lidars[0]['suggested_phase_deg'] == pytest.approx(300.0)
    # one turning the other way meets the reference ahead and behind, half a turn apart at the sides
    p = _rig(SIDES)
    p.variant.sensors.append(Sensor(kind='lidar', model_id=_ccw(p), tag='L2', x=1.0, y=0.0, z=1.5))
    (d,) = TM.plan(p).lidars
    assert not d['same_sense'] and d['suggested_phase_deg'] == pytest.approx(0.0, abs=1e-9)
    assert d['after_ms'] == pytest.approx(50.0) and d['lag_ms'] == pytest.approx(0.0, abs=1e-9)
    assert 'other way' in d['note']
    # mounted upside down, a clockwise lidar turns counter-clockwise seen from above
    p = _rig(SIDES)
    p.variant.sensors.append(Sensor(kind='lidar', model_id='ouster_os1_128', tag='L2', x=1.0, z=1.5, roll=180.0))
    assert not TM.plan(p).lidars[0]['same_sense']
    # a non-spinning lidar gets no phase advice
    p = _rig(SIDES)
    p.variant.sensors.append(Sensor(kind='lidar', model_id='hesai_at128', tag='L2', x=3.0, z=0.8))
    assert TM.plan(p).lidars == []


def test_reference_lidar_choice_and_no_sweep():
    p = _rig(SIDES)
    p.variant.sensors.append(Sensor(kind='lidar', model_id='hesai_xt32', tag='L2', x=1.0, z=1.5, yaw=90.0))
    second = p.variant.sensors[1].uid
    res = TM.plan(p, settings=TM.TimingSettings(lidar_uid=second))
    assert res.lidar.tag == 'L2' and [d['tag'] for d in res.lidars] == ['L1']
    assert not any('reference lidar' in n for n in res.notes)
    res = TM.plan(p, settings=TM.TimingSettings(lidar_uid='gone'))
    assert res.lidar.tag == 'L1' and any('reference lidar' in n for n in res.notes)
    p.variant.sensors[0].visible = False
    assert TM.plan(p).lidar.tag == 'L2'
    # a fixed-field lidar only: no sweep plan, but the motion figures
    p = _rig(SIDES, model='hesai_at128')
    res = TM.plan(p)
    assert res.lidar is None and 'do not spin' in res.reason and res.groups == [] and res.period_ms is None
    assert all(c['blur_worst_px'] > 0 and 'cross_ms' not in c for c in res.cams)
    assert TM.suggest_groups(res) == {} and TM.evaluate_groups(res, {})[1] is None
    p.variant.sensors.clear()
    res = TM.plan(p)
    assert res.lidar is None and 'No spinning lidar' in res.reason and len(res.cams) == 3
    # another variant than the active one, by uid or as itself
    from vantage.core.project import Variant
    other = Variant(name='With lidar', cameras=[Camera(tag='B', yaw=180.0, pitch=0.0, module_id='m', lens_id='pin')],
                    sensors=[Sensor(kind='lidar', model_id='ouster_os1_128', tag='L9', z=1.8)])
    p.variants.append(other)
    for v in (other.uid, other):
        out = TM.plan(p, v)
        assert out.lidar.tag == 'L9' and [c['tag'] for c in out.cams] == ['B'] and out.variant_uid == other.uid
        assert out.cams[0]['cross_ms'] == pytest.approx(50.0)
    s = TM.summary(res)
    assert s['lidar'] is None and 'no_sweep_plan' in s and len(s['cameras']) == 3
    json.dumps(s)


def test_free_running_and_hidden_cameras():
    p = _rig(SIDES)
    p.variant.cameras[2].visible = False
    p.variant.cameras[1].sync = ''
    res = TM.plan(p)
    c = {e['tag']: e for e in res.cams}
    assert set(c) == {'F', 'L'} and not c['L']['triggered'] and c['L']['misalign_ms'] is None
    assert c['L']['ideal_ms'] == pytest.approx(64.0) and any('L: on no trigger line' in n for n in res.notes)
    assert set(res.suggestion['letters']) == {c['F']['uid']}
    p.modules['m'].hw_sync = False
    res = TM.plan(p)
    assert not res.cams[0]['triggered'] and res.groups == [] and res.metrics['worst_ms'] is None
    assert any('no hardware trigger input' in n for n in res.notes)


# ----------------------------------------------------------------------------- frame rates
def test_frame_rate_ratio():
    cases = {10.0: ('integer', 100.0), 30.0: ('integer', 100 / 3), 5.0: ('fraction', 100.0),
             29.97: ('drifts', 100.0), 12.0: ('drifts', 100.0)}
    for fps, (how, cycle) in cases.items():
        res = TM.plan(_rig(SIDES[:1], fps=fps))
        c = res.cams[0]
        assert c['ratio'] == pytest.approx(fps / 10.0) and c['alignment'] == how
        assert c['cycle_ms'] == pytest.approx(cycle)
        flagged = any('not an integer multiple' in n for n in res.notes)
        assert flagged == (how == 'drifts') and (res.metrics['drifting'] == ['F']) == (how == 'drifts')
    res = TM.plan(_rig(SIDES[:1], fps=29.97))
    assert res.cams[0]['beat_s'] == pytest.approx(1 / 0.03) and 'back in step every 33.3 s' in ' '.join(res.notes)


# ----------------------------------------------------------------------------- motion
def test_blur_of_a_side_camera_through_a_pinhole():
    """A level 6 mm pinhole looking left at 36 km/h (10 m/s), object 5 m out: at the image centre the object moves at
    v/d = 2 rad/s, f = 2000 px/rad: 4000 px/s, 8 px in 2 ms, 80 px of skew over a 20 ms readout. Off the axis a
    pinhole sees it move f·v/Z with Z = d·cos α the depth: fastest in the corners."""
    st = TM.TimingSettings(speed_kmh=36.0, exposure_ms=2.0, readout_ms=20.0, object_m=5.0, blur_limit_px=1.0)
    p = _rig([('L', 0.0, 0.5, 1.0, 90.0), ('F', 0.0, 0.0, 1.0, 0.0)])
    res = TM.plan(p, settings=st)
    c = {e['tag']: e for e in res.cams}
    side = c['L']
    assert side['rate_centre_px_s'] == pytest.approx(4000.0, rel=1e-6)
    assert side['blur_centre_px'] == pytest.approx(8.0, rel=1e-6)
    assert side['skew_centre_px'] == pytest.approx(80.0, rel=1e-6)
    r_mm = math.hypot(959.5, 767.5) * 0.003                        # the corner sample, 0.5 px in
    worst = 2000.0 * 10.0 / (5.0 * math.cos(math.atan(r_mm / 6.0)))
    assert side['rate_worst_px_s'] == pytest.approx(worst, rel=1e-4)
    assert side['max_exposure_ms'] == pytest.approx(1000.0 / worst, rel=1e-4)
    assert side['skew_px'] == pytest.approx(worst * 0.02, rel=1e-4)
    # looking ahead the centre of the image does not move (the focus of expansion)
    assert c['F']['blur_centre_px'] == pytest.approx(0.0, abs=1e-6) and c['F']['blur_worst_px'] > 0.5
    assert res.metrics['worst_blur_tag'] == 'L'
    assert res.metrics['max_exposure_ms'] == pytest.approx(1000.0 / worst, rel=1e-4)
    assert any(n.startswith('L: up to') for n in res.notes)
    # standing still: no blur, any exposure
    res = TM.plan(p, settings=TM.TimingSettings(speed_kmh=0.0))
    assert all(e['blur_worst_px'] == 0.0 and e['max_exposure_ms'] is None for e in res.cams)
    assert res.metrics['max_exposure_ms'] is None and res.metrics['rotation_travel_m'] == 0.0


def test_fisheye_blur_follows_the_lens_model(tigor):
    """The sample's wide fisheye pitched 40° down: the image centre looks 40° off the direction of travel, so the
    object moves v·sin 40° / d across it at the lens's centre sampling f/pixel."""
    from vantage.core.lens import resolve
    res = TM.plan(tigor)
    f = next(c for c in res.cams if c['tag'] == 'F')
    cam = tigor.variant.by_tag('F')
    intr = resolve(tigor.module_for(cam), tigor.lens_for(cam))
    want = 50 / 3.6 * math.sin(math.radians(cam.pitch)) / 5.0 * intr.f_px
    assert f['rate_centre_px_s'] == pytest.approx(want, rel=0.01)
    assert f['blur_worst_px'] > f['blur_centre_px']


# ----------------------------------------------------------------------------- settings and output
def test_settings_are_validated():
    st = TM.TimingSettings.from_dict({'speed_kmh': 'fast', 'exposure_ms': -1, 'readout_ms': 1e9, 'trigger_lines': 3.6,
                                      'object_m': float('nan'), 'blur_limit_px': None, 'lidar_uid': 5, 'nonsense': 1})
    assert st.speed_kmh == 50.0 and st.exposure_ms == 0.001 and st.readout_ms == 100.0
    assert st.trigger_lines == 4 and isinstance(st.trigger_lines, int)
    assert st.object_m == 5.0 and st.blur_limit_px == 1.0 and st.lidar_uid == ''
    assert TM.TimingSettings.from_dict({'trigger_lines': 20}).trigger_lines == 8
    assert TM.TimingSettings.from_dict({'trigger_lines': 0}).trigger_lines == 1
    assert TM.TimingSettings.from_dict(None) == TM.TimingSettings()
    st = TM.TimingSettings(speed_kmh=80.0, readout_ms=12.5, trigger_lines=2, lidar_uid='abc')
    assert TM.TimingSettings.from_dict(st.to_dict()) == st
    # the project's stored settings are used by default
    p = _rig(SIDES)
    p.analysis.timing = {'readout_ms': 0.0, 'exposure_ms': 4.0}
    c = {e['tag']: e for e in TM.plan(p).cams}
    assert c['L']['ideal_ms'] == pytest.approx(73.0)


def test_export_json_and_png(tigor, tmp_path):
    import cv2
    p = copy.deepcopy(tigor)
    p.units = 'mm'
    p.variant.sensors.append(Sensor(kind='lidar', model_id='hesai_xt32', tag='L2', x=3.0, y=0.5, z=1.2, yaw=180.0))
    res = TM.plan(p)
    paths = TM.export(res, p, str(tmp_path))
    assert [os.path.basename(x) for x in paths] == ['sync_plan.json', 'sync_plan.png']
    doc = json.loads((tmp_path / 'sync_plan.json').read_text())
    assert doc['schema'] == TM.SCHEMA == 'vantage.sync_plan/1'
    assert doc['units'] == 'mm' and doc['frame'] == p.frame and doc['variant'] == p.variant.name
    lid, o = p.variant.sensors[0], p.origin_offset()
    assert doc['lidar']['tag'] == 'L1'
    assert doc['lidar']['position'] == pytest.approx([(lid.x - o[0]) * 1000, (lid.y - o[1]) * 1000,
                                                      (lid.z - o[2]) * 1000], abs=0.1)
    assert doc['lidar']['travel_per_rotation'] == pytest.approx(50 / 3.6 * 100, abs=0.1)     # mm
    assert doc['settings']['object'] == 5000 and doc['settings']['readout_ms'] == 20.0
    assert [c['tag'] for c in doc['cameras']] == [c.tag for c in p.variant.cameras]
    f = doc['cameras'][0]
    assert _apart(f['crossing_ms'], 0.0) < 1e-3 and f['group'] == 'A'
    assert f['travel'] == pytest.approx(abs(f['misalignment_ms']) * 50 / 3.6, abs=0.2)      # mm
    assert set(f['motion']) == {'blur_centre_px', 'blur_worst_px', 'max_exposure_ms', 'skew_centre_px', 'skew_px'}
    assert len(doc['groups']) == 1 and set(doc['suggested_groups']['groups']) == {c.tag for c in p.variant.cameras}
    assert doc['other_lidars'][0]['tag'] == 'L2' and doc['other_lidars'][0]['suggested_phase'] == pytest.approx(180.0)
    img = cv2.imread(str(tmp_path / 'sync_plan.png'))
    assert img is not None and img.shape[1] == 900 and img.shape[0] > 900
    s = TM.summary(res)
    assert s['lidar']['tag'] == 'L1' and s['other_lidars'][0]['suggested_phase_deg'] == pytest.approx(180.0)
    assert set(s['suggested_groups']) == {c.tag for c in p.variant.cameras}
    json.dumps(s)
    # with no spinning lidar the export still carries the motion figures and says why there is no sweep plan
    p.variant.sensors.clear()
    paths = TM.export(TM.plan(p), p, str(tmp_path / 'none'))
    doc = json.loads((tmp_path / 'none' / 'sync_plan.json').read_text())
    assert doc['lidar'] is None and 'No spinning lidar' in doc['no_sweep_plan'] and doc['groups'] == []
    assert all(c['motion']['blur_worst_px'] > 0 and c['crossing_ms'] is None for c in doc['cameras'])
    assert cv2.imread(paths[1]) is not None


# ----------------------------------------------------------------------------- the page, driven like a user
from test_gui import app, win, pump, settle, _wait     # noqa: E402,F401  (fixtures)


def _lidar(win, tag='L1', **pose):
    """An OS1-128 at pose, or where Vehicle ▸ Add lidar puts one."""
    p = win.store.project
    m = p.sensor_models['ouster_os1_128']
    return Sensor(kind='lidar', model_id='ouster_os1_128', tag=tag,
                  **(pose or sensors.default_pose('lidar', m, p.vehicle.mesh, p.rig)))


def _set_sensors(win, *ss):
    """The active variant's lidars and radars become exactly these (one undo step)."""
    win.store.edit('variant', lambda q: setattr(q.variant, 'sensors', list(ss)), 'Lidars for the timing plan')
    return ss


def _open(app, win):
    menus = {a.text().replace('&', ''): a.menu() for a in win.menuBar().actions()}
    act = win.actions_by_id['sync_plan']
    assert act in menus['Analysis'].actions()
    act.trigger()                                        # Analysis ▸ Sync and timing…
    pump(app, 0.2)
    pg = win.page('timing')
    assert win.page_name() == 'timing' and pg.isVisible()
    return pg


def _type_into_field(app, f, text):
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest
    QTest.mouseClick(f.lineEdit(), Qt.MouseButton.LeftButton)
    QTest.keyClick(f.lineEdit(), Qt.Key.Key_A, Qt.KeyboardModifier.ControlModifier)
    QTest.keyClicks(f.lineEdit(), text)
    QTest.keyClick(f.lineEdit(), Qt.Key.Key_Return)
    pump(app, 0.1)


def test_page_suggests_groups_applies_and_undoes(app, win, tmp_path):
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest
    st = win.store
    (lidar,) = _set_sensors(win, _lidar(win))
    settle(app, win)
    pg = _open(app, win)
    res = pg.res
    assert res.lidar is not None and res.lidar.uid == lidar.uid and pg.dial.res is res
    assert pg.table.rowCount() == len(st.variant.cameras) == 6 and res.metrics['lines_used'] == 1
    assert pg.table.item(0, 1).text() == 'A' and 'Worst misalignment' in pg.head.text()
    assert pg.lidars.rowCount() == 1 and not pg.b_apply.isVisible()
    pg.dial.grab()                                       # paints without errors
    # Suggest groups: the proposal in the table and the worst misalignment before and after
    QTest.mouseClick(pg.b_suggest, Qt.MouseButton.LeftButton)
    pump(app, 0.1)
    sug = res.suggestion
    assert pg.proposing and pg.b_apply.isVisible() and pg.b_apply.isEnabled()
    assert f'{res.metrics["worst_ms"]:.1f} → {sug["worst_ms"]:.1f} ms' in pg.proposal.text()
    shown = {pg.table.item(r, 0).text(): pg.table.item(r, 1).text() for r in range(pg.table.rowCount())}
    tags = {c['uid']: c['tag'] for c in res.cams}
    assert all(shown[tags[u]] == ('A' if L == 'A' else f'A → {L}') for u, L in sug['letters'].items())
    # Apply: every camera's sync letter in one undo step
    n = st.undo.count()
    QTest.mouseClick(pg.b_apply, Qt.MouseButton.LeftButton)
    pump(app, 0.2)
    assert st.undo.count() == n + 1 and not pg.proposing
    assert {c.uid: c.sync for c in st.variant.cameras} == sug['letters']
    _wait(app, lambda: pg.res is not res and not pg._dirty, 10)
    assert pg.res.metrics['worst_ms'] == pytest.approx(sug['worst_ms'])
    assert pg.res.metrics['lines_used'] == sug['lines'] > 1
    st.undo.undo()
    pump(app, 0.2)
    assert all(c.sync == 'A' for c in st.variant.cameras)
    _wait(app, lambda: pg.res.metrics['lines_used'] == 1, 10)
    # a setting typed into its field: saved with the project, the plan follows, undoable
    travel = pg.res.metrics['worst_travel_cm']
    _type_into_field(app, pg.fields['speed_kmh'], '100')
    assert st.project.analysis.timing['speed_kmh'] == 100
    _wait(app, lambda: pg.res.settings.speed_kmh == 100, 10)
    assert pg.res.metrics['worst_travel_cm'] == pytest.approx(2 * travel)
    st.undo.undo()
    pump(app, 0.2)
    assert st.project.analysis.timing == {} and pg.fields['speed_kmh'].value() == 50
    # export what is shown
    paths = pg.export(str(tmp_path))
    assert [os.path.basename(x) for x in paths] == ['sync_plan.json', 'sync_plan.png']
    assert json.loads((tmp_path / 'sync_plan.json').read_text())['lidar']['tag'] == lidar.tag


def test_second_lidar_phases_and_leaving_the_page(app, win):
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest
    st = win.store
    _, second = _set_sensors(win, _lidar(win), _lidar(win, 'L2', x=2.6, y=0.0, z=1.5, yaw=180.0, pitch=0.0, roll=0.0))
    settle(app, win)
    pg = _open(app, win)
    (d,) = pg.res.lidars
    assert d['uid'] == second.uid and d['suggested_phase_deg'] == pytest.approx(180.0) and pg.lidars.rowCount() == 2
    assert pg.b_phases.isEnabled()
    n = st.undo.count()
    QTest.mouseClick(pg.b_phases, Qt.MouseButton.LeftButton)
    pump(app, 0.2)
    assert st.undo.count() == n + 1 and st.variant.sensor(second.uid).phase_deg == pytest.approx(180.0)
    _wait(app, lambda: pg.res.lidars[0]['misalign_ms'] < 1e-6, 10)
    assert not pg.b_phases.isEnabled()
    st.undo.undo()
    pump(app, 0.2)
    assert st.variant.sensor(second.uid).phase_deg == 0.0
    # another page: edits no longer work the plan out; back on the page they do
    win.actions_by_id['blind_volume'].trigger()
    pump(app, 0.2)
    assert not pg.isVisible()
    shown = pg.res
    c = st.variant.by_tag('F')
    st.set_cam(c.uid, yaw=c.yaw + 20.0)
    pump(app, 0.3)
    assert pg.res is shown and pg._dirty
    win.actions_by_id['sync_plan'].trigger()
    pump(app, 0.2)
    assert pg.res is not shown and not pg._dirty
    f = next(e for e in pg.res.cams if e['tag'] == 'F')
    assert f['azimuth_deg'] > 5.0                        # turned 20° left: its view lies left of the lidar's forward
