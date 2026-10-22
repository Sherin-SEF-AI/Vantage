"""Perception range: pixel heights through the real lens, occlusion by the body, sector verdicts, stereo pairs, the
export, batch mode and the page (offscreen)."""
import copy
import json
import math
import os

import numpy as np
import pytest

from vantage.core import perception as PR, samples
from vantage.core.analysis import Engine
from vantage.core.lens import Intrinsics, builtin_lenses, builtin_modules, dr_dtheta, r_of_theta, resolve
from vantage.core.project import Camera


@pytest.fixture(scope='module')
def sample():
    p = samples.tigor_sample_project()
    e = Engine()
    res = e.analyse(p)
    return p, e, res


@pytest.fixture(scope='module')
def pr(sample):
    p, e, res = sample
    return PR.analyse(p, e, res)


def _pinhole(f_mm=6.0, w=1920, h=1536, pixel_mm=0.003):
    return Intrinsics(width=w, height=h, pixel_mm=pixel_mm, f_mm=f_mm, projection='rectilinear', kb=(0, 0, 0, 0),
                      cx=w / 2, cy=h / 2, max_theta=math.radians(89.0), focus_near=0.1, focus_far=math.inf)


def _only(p, *tags):
    for c in p.variant.cameras:
        c.visible = c.tag in tags


# ----------------------------------------------------------------------------- pixels through the lens
def test_pinhole_height_is_f_H_over_Z():
    cam = Camera(x=0.0, y=0.0, z=0.75, yaw=0.0, pitch=0.0, roll=0.0)    # level, looking forward (+x)
    intr = _pinhole()
    car = PR.ObjectClass('Car', 1.5, 1.8, 20.0)
    Z = np.array([5.0, 10.0, 40.0, 100.0])
    hpx, wpx, seen = PR.object_pixels(cam, intr, np.c_[Z, np.zeros(4)], car)
    assert seen.all()
    np.testing.assert_allclose(hpx, intr.f_px * 1.5 / Z, rtol=1e-9)
    np.testing.assert_allclose(wpx, intr.f_px * 1.8 / Z, rtol=1e-9)
    # off the axis a pinhole still scales with the depth along the axis, not the distance
    hpx, _, seen = PR.object_pixels(cam, intr, np.array([[20.0, 6.0]]), car)
    assert seen[0] and hpx[0] == pytest.approx(intr.f_px * 1.5 / 20.0, rel=1e-9)
    # a sign on a post: only its own 0.6 m count
    sign = PR.ObjectClass('Sign', 0.6, 0.6, 12.0, base=0.45)                # centred on the camera height
    hpx, _, _ = PR.object_pixels(cam, intr, np.array([[30.0, 0.0]]), sign)
    assert hpx[0] == pytest.approx(intr.f_px * 0.6 / 30.0, rel=1e-9)
    # behind the camera and outside the image: not seen, no pixels
    hpx, wpx, seen = PR.object_pixels(cam, intr, np.array([[-10.0, 0.0], [5.0, 40.0]]), car)
    assert not seen.any() and not hpx.any() and not wpx.any()


def test_fisheye_resolves_less_across_and_more_along_the_edge_than_the_centre():
    """A level fisheye: an object 30 m out on the axis and 55° off it. Its height runs tangential to the image circle
    there and its width radial; both match the lens's local sampling (r/sinθ and dr/dθ), which a pinhole formula
    would not."""
    m, wide = builtin_modules()[0], builtin_lenses()[0]
    intr = resolve(m, wide)
    cam = Camera(x=0.0, y=0.0, z=0.75, yaw=0.0, pitch=0.0, roll=0.0)
    car = PR.ObjectClass('Car', 1.5, 1.8, 20.0)
    D, a = 30.0, math.radians(55.0)
    G = np.array([[D, 0.0], [D * math.cos(a), D * math.sin(a)]])
    hpx, wpx, seen = PR.object_pixels(cam, intr, G, car)
    assert seen.all()
    f = dict(f=intr.f_mm, projection=intr.projection, kb=intr.kb)
    radial = lambda th: float(dr_dtheta(th, **f)) / intr.pixel_mm
    tangential = lambda th: float(r_of_theta(th, **f)) / intr.pixel_mm / math.sin(th)
    centre = radial(1e-4)
    assert hpx[0] == pytest.approx(centre * 1.5 / D, rel=0.01)
    assert wpx[0] == pytest.approx(centre * 1.8 / D, rel=0.01)
    assert hpx[1] == pytest.approx(tangential(a) * 1.5 / D, rel=0.02)
    assert wpx[1] == pytest.approx(radial(a) * 1.8 / D, rel=0.02)
    assert wpx[1] < 0.93 * wpx[0] and hpx[1] > 1.07 * hpx[0]          # equisolid-like: cos(θ/2), 1/cos(θ/2)


# ----------------------------------------------------------------------------- the study
def test_far_range_of_a_pinhole_is_f_H_over_min_px(sample):
    """A level 6 mm rectilinear camera on the front bumper: the car is min_px tall at f·H/min_px ahead of the camera.
    The study finds that range between its samples, not just the last sample before it."""
    from vantage.core.lens import Lens
    p0, e, res = sample
    p = copy.deepcopy(p0)
    p.lenses['pin'] = Lens(id='pin', name='Pinhole 6 mm', focal_mm=6.0, projection='rectilinear')
    _only(p, 'F')
    f = p.variant.by_tag('F')
    f.lens_id, f.yaw, f.pitch, f.roll = 'pin', 0.0, 0.0, 0.0
    intr = resolve(p.module_for(f), p.lens_for(f))
    for min_px in (20.0, 33.0):
        st = PR.PerceptionSettings.from_dict({'classes': [dict(PR.DEFAULT_CLASSES[0], min_px=min_px)]})
        out = PR.analyse(p, e, res, st)
        k = int(np.argmin(np.abs(out.az - 0.0)))
        expect = f.x - out.centre[0] + intr.f_px * 1.5 / min_px     # range from the vehicle centre
        assert out.far[0, k] == pytest.approx(expect, rel=1e-3)
        assert out.at(0, 0.0)[2] == 'F'


def test_result_shapes_and_the_far_range_is_the_best_camera(pr):
    K, A, R = len(pr.classes), len(pr.az), len(pr.ranges)
    C = len(pr.cams)
    assert pr.far.shape == pr.near.shape == pr.best.shape == (K, A) and pr.cam_far.shape == (C, K, A)
    assert pr.count.shape == (K, A, R) and pr.count.max() <= C
    np.testing.assert_allclose(pr.far, pr.cam_far.max(0))
    k = 0
    far, near, tag = pr.at(k, 0.0)
    assert far == pr.far[k, 0] and tag == pr.tag(int(pr.best[k, 0])) and far > 20.0 and near < far
    assert [s['name'] for s in pr.sectors] == [n for n, _ in PR.SECTORS]
    assert pr.metrics['checks'] == len(PR.SECTORS) * K and 0 < pr.metrics['passed'] <= pr.metrics['checks']
    # a smaller, lower object is resolved less far than a car in every direction
    names = [c.name for c in pr.classes]
    assert (pr.far[names.index('Car')] >= pr.far[names.index('Pedestrian')] - 1e-9).all()


def test_the_body_hides_what_is_behind_it(sample):
    """The front camera turned round to look back over the car: the body hides a car behind the vehicle that the
    same camera would resolve with nothing in the way."""
    p0, e, res = sample
    p = copy.deepcopy(p0)
    _only(p, 'F')
    f = p.variant.by_tag('F')
    f.yaw, f.pitch = 180.0, 0.0
    st = PR.PerceptionSettings.from_dict({'classes': [PR.DEFAULT_CLASSES[0]]})
    out = PR.analyse(p, e, res, st)
    assert out.metrics['cameras'] == 1
    behind = [out.at(0, a)[0] for a in (170.0, 180.0, 190.0)]
    assert behind == [0.0, 0.0, 0.0]
    # without the body the same camera resolves that car 20 m behind the vehicle
    car = st.object_classes()[0]
    intr = resolve(p.module_for(f), p.lens_for(f))
    G = np.array([[out.centre[0] - 20.0, out.centre[1]]])
    h_free, _, seen_free = PR.object_pixels(f, intr, G, car, None)
    h_body, _, seen_body = PR.object_pixels(f, intr, G, car, e.cube(f.pos()))
    assert seen_free[0] and h_free[0] >= car.min_px and not seen_body[0] and h_body[0] == 0.0
    assert out.at(0, 0.0)[0] == 0.0                     # it looks backwards: nothing ahead either


def test_ranges_shrink_as_the_detector_needs_more_pixels(sample):
    p, e, res = sample
    base = dict(PR.DEFAULT_CLASSES[1])
    st = PR.PerceptionSettings.from_dict({'classes': [dict(base, name=f'{px}px', min_px=px) for px in (8, 16, 32, 64)]})
    out = PR.analyse(p, e, res, st)
    for k in range(3):
        assert (out.far[k] >= out.far[k + 1]).all()
        assert (out.count[k] >= out.count[k + 1]).all()
    assert out.far[0].mean() > 1.5 * out.far[2].mean()          # 4x the pixels: much closer
    # pixel height falls like 1/range, so doubling min_px about halves the far range in a direction
    a = int(np.argmax(out.far[1]))
    assert out.far[2, a] / out.far[1, a] == pytest.approx(0.5, abs=0.08)


def test_sector_verdicts_follow_the_required_ranges(sample, pr):
    p, e, res = sample
    s = pr.sectors[0]
    assert s['name'] == 'front'
    m = PR._sector_of(pr.az) == 0
    car = s['classes']['Car']
    assert car['min_m'] == pytest.approx(pr.far[0, m].min()) and car['median_m'] == pytest.approx(np.median(pr.far[0, m]))
    st = PR.PerceptionSettings.from_dict({'required': {'front': car['min_m'] - 1.0, 'rear': car['min_m'] + 500}})
    out = PR.analyse(p, e, res, st)
    front = out.sectors[0]['classes']['Car']
    assert front['pass'] and front['shortfall_m'] == 0.0
    st = PR.PerceptionSettings.from_dict({'required': {'front': car['min_m'] + 1.0}})
    front = PR.analyse(p, e, res, st).sectors[0]['classes']['Car']
    assert not front['pass'] and front['shortfall_m'] == pytest.approx(1.0)
    assert not out.sectors[4]['classes']['Car']['pass']                      # rear: 500 m beyond the range
    assert PR.summary(out)['per_class_sector']['Car']['front']['pass']
    # settings are validated: negative ranges and unusable classes fall back
    st = PR.PerceptionSettings.from_dict({'required': {'front': -5, 'nowhere': 3}, 'max_range': 5,
                                          'classes': [{'name': 'x', 'height': 0, 'width': 1, 'min_px': 1}]})
    assert st.required['front'] == PR.DEFAULT_REQUIRED['front'] and 'nowhere' not in st.required
    assert st.max_range == 10.0 and [c['name'] for c in st.classes] == [c['name'] for c in PR.DEFAULT_CLASSES]
    st = PR.PerceptionSettings.from_dict({'classes': [dict(PR.DEFAULT_CLASSES[0]), dict(PR.DEFAULT_CLASSES[0])]})
    assert [c['name'] for c in st.classes] == ['Car', 'Car 2']


def test_stereo_pair_overlap_and_baseline_for_a_known_rig(sample):
    """Two narrow cameras side by side on the front bumper, level and looking ahead, 0.5 m apart."""
    p0, e, res = sample
    p = copy.deepcopy(p0)
    _only(p, 'F', 'R')
    f, r = p.variant.by_tag('F'), p.variant.by_tag('R')
    f.lens_id = r.lens_id = 'h03r2'                     # narrow: 54° wide
    f.yaw = r.yaw = 0.0
    f.pitch = r.pitch = f.roll = r.roll = 0.0
    r.x, r.z = f.x, f.z
    f.y, r.y = 0.25, -0.25
    out = PR.analyse(p, e, res)
    assert len(out.pairs) == 1
    pp = out.pairs[0]
    assert set(pp['tags']) == {'F', 'R'}
    assert pp['baseline_m'] == pytest.approx(0.5, abs=1e-9)
    assert abs(pp['direction_deg']) <= 2.0
    # both see the car 20 m out across about the lens's width (a little less: the cameras sit ahead of the centre)
    assert 38.0 <= pp['overlap_deg'] <= 54.0
    intr = resolve(p.module_for(f), p.lens_for(f))
    assert pp['f_px_per_rad'] == pytest.approx(intr.f_px, rel=0.03)
    for z in PR.STEREO_AT:
        assert pp['depth_error_m'][f'{z:g}'] == pytest.approx(z * z * PR.DISPARITY_PX / (pp['f_px_per_rad'] * 0.5))
    # turned back to back they share nothing
    r.yaw = 180.0
    assert PR.analyse(p, e, res).pairs == []


def test_cancel_and_progress(sample):
    p, e, res = sample
    steps = []
    assert PR.analyse(p, e, res, progress=steps.append, cancel=lambda: len(steps) >= 2) is None
    steps = []
    assert PR.analyse(p, e, res, progress=steps.append) is not None
    assert steps == sorted(steps) and steps[-1] == 1.0


# ----------------------------------------------------------------------------- export and batch
def test_export_json_and_png(sample, tmp_path):
    import cv2
    from vantage.core import exporters as ex
    p, e, res = sample
    paths = ex.export(['perception_range'], p, res, ex.ExportOptions(out_dir=str(tmp_path), engine=e, units='mm'))
    assert [os.path.basename(x) for x in paths] == ['perception_range.json', 'perception_range.png']
    doc = json.loads((tmp_path / 'perception_range' / 'perception_range.json').read_text())
    assert doc['schema'] == PR.SCHEMA == 'vantage.perception_range/1'
    assert doc['units'] == 'mm' and doc['variant'] == p.variant.name and doc['frame'] == p.frame
    assert len(doc['azimuth']) == 360 and doc['required_ranges']['front'] == 80000
    names = [c['name'] for c in doc['classes']]
    assert names == ['Car', 'Pedestrian', 'Cyclist', 'Traffic sign'] and doc['classes'][0]['height'] == 1500
    car = doc['per_class']['Car']
    assert len(car['far']) == len(car['camera']) == 360 and max(car['far']) > 50000
    assert set(doc['per_camera']) == {c.tag for c in p.variant.cameras if c.visible}
    s = doc['sectors'][0]
    assert s['name'] == 'front' and s['required'] == 80000 and set(s['classes']) == set(names)
    assert doc['stereo']['pairs'] and all(q['overlap'] >= PR.MIN_OVERLAP_DEG for q in doc['stereo']['pairs'])
    img = cv2.imread(str(tmp_path / 'perception_range' / 'perception_range.png'))
    assert img is not None and img.shape[1] == 900 and img.shape[0] > 900
    path, text, size = ex.preview('perception_range', p, res, ex.ExportOptions(engine=e))
    assert path.endswith('perception_range/perception_range.json') and 'sector checks pass' in text and size > 0
    # no analysis result is no obstacle: the study needs only the cameras and the body
    paths = ex.export(['perception_range'], p, None, ex.ExportOptions(out_dir=str(tmp_path / 'nores'), engine=e))
    assert len(paths) == 2


def test_batch_perception_range_honours_the_variant_and_settings(tmp_path):
    from vantage import batch
    from test_advanced import _args
    p = samples.tigor_sample_project()
    assert len(p.variants) >= 2
    p.analysis.perception = {'classes': [{'name': 'Bus', 'height': 3.0, 'width': 2.5, 'min_px': 30, 'base': 0}],
                             'required': {'front': 10}}
    src = tmp_path / 'p.vantage'
    p.save(str(src))
    out = tmp_path / 'ranges.json'
    a = _args(perception_range=str(out), variant='2')
    a.sample, a.path = False, str(src)
    assert batch.run(a) == 0
    doc = json.loads(out.read_text())
    v2 = p.variants[1]
    assert doc['schema'] == PR.SCHEMA and doc['variant'] == v2.name
    assert set(doc['per_camera']) == {c.tag for c in v2.cameras if c.visible}
    assert [c['name'] for c in doc['classes']] == ['Bus'] and doc['required_ranges']['front'] == 10
    assert doc['sectors'][0]['classes']['Bus']['pass']
    assert batch.wanted(_args(perception_range='-'))


def test_assistant_runs_the_study(sample):
    from vantage.assistant import tools as T
    assert 'perception_range' in T.STUDIES
    p, e, res = sample
    tc = T.ToolContext(p.snapshot(), res, e.share())
    out = T.run_study(tc, study='perception_range')
    assert out['study'] == 'perception_range' and out['checks'] == 32 and 0 < out['passed'] <= 32
    assert set(out['per_class_sector']) == {'Car', 'Pedestrian', 'Cyclist', 'Traffic sign'}
    assert out['stereo_pairs'] and 'meaning' in out
    json.dumps(out)


# ----------------------------------------------------------------------------- the page, driven like a user
from test_gui import app, win, pump, settle, _wait     # noqa: E402,F401  (fixtures)


def _open(app, win):
    menus = {a.text().replace('&', ''): a.menu() for a in win.menuBar().actions()}
    act = win.actions_by_id['perception_range']
    assert act in menus['Analysis'].actions()
    act.trigger()                                        # Analysis ▸ Perception range…
    pump(app, 0.2)
    pg = win.page('perception')
    assert win.page_name() == 'perception' and pg.isVisible()
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


def _type_into_field(app, f, text):
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest
    QTest.mouseClick(f.lineEdit(), Qt.MouseButton.LeftButton)
    QTest.keyClick(f.lineEdit(), Qt.Key.Key_A, Qt.KeyboardModifier.ControlModifier)
    QTest.keyClicks(f.lineEdit(), text)
    QTest.keyClick(f.lineEdit(), Qt.Key.Key_Return)
    pump(app, 0.1)


def test_page_runs_edits_and_undoes(app, win, tmp_path):
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest
    pg = _open(app, win)
    ctl = win.ctx.perception
    _wait(app, lambda: ctl.result is not None and not ctl.busy, 60)      # a quick study runs when the page shows
    # Run: a main-window job (progress bar, Cancel)
    before = ctl.result
    QTest.mouseClick(pg.b_run, Qt.MouseButton.LeftButton)
    assert 'perception' in win._jobs and not pg.b_run.isEnabled()
    _wait(app, lambda: ctl.result is not before and not ctl.busy, 60)
    pump(app, 0.1)
    pr = ctl.result
    assert 'sector checks pass' in pg.status.text() and pg.b_run.isEnabled()
    assert pg.verdicts.rowCount() == len(pr.classes) == 4
    assert pg.verdicts.item(0, 1).text() == f'{pr.sectors[0]["classes"]["Car"]["min_m"]:.0f}'
    assert pg.pairs.rowCount() == len(pr.pairs) > 0 and pg.map.pr is pr and pg.roses.pr is pr
    pg.mode.buttons['cameras'].click()
    pg.pick.buttons['3'].click()
    pump(app, 0.1)
    assert pg.map.mode == 'cameras' and pg.map.k == 3
    # the car's minimum pixel height, typed into the table: saved with the project, re-run by itself, undoable
    far0 = pr.at(0, 0.0)[0]
    _type_into_cell(app, pg.classes, 0, 3, '40')
    assert win.store.project.analysis.perception['classes'][0]['min_px'] == 40
    _wait(app, lambda: ctl.result.classes[0].min_px == 40 and not ctl.busy and not ctl.stale, 60)
    assert ctl.result.at(0, 0.0)[0] < 0.75 * far0 and pg.classes.item(0, 3).text() == '40'
    # nonsense is refused; the table shows what is stored
    _type_into_cell(app, pg.classes, 0, 3, '-3')
    assert win.store.project.analysis.perception['classes'][0]['min_px'] == 40
    assert pg.classes.item(0, 3).text() == '40'
    _type_into_cell(app, pg.classes, 1, 1, '180 cm')                    # lengths take units
    assert win.store.project.analysis.perception['classes'][1]['height'] == pytest.approx(1.8)
    assert pg.classes.item(1, 1).text() == '1.80'
    # a required range, typed into its field
    f = pg.req['front']
    _type_into_field(app, f, '30')
    assert win.store.project.analysis.perception['required']['front'] == 30
    _wait(app, lambda: ctl.result.sectors[0]['required_m'] == 30 and not ctl.busy, 60)
    assert pg.verdicts.horizontalHeaderItem(1).text() == 'Front 30'
    win.store.undo.undo()
    pump(app, 0.1)
    assert win.store.project.analysis.perception['required']['front'] == 80 and f.value() == 80
    win.store.undo.undo()
    pump(app, 0.1)
    assert ctl.settings.classes[1]['height'] == 1.7 and pg.classes.item(1, 1).text() == '1.70'
    win.store.undo.undo()
    pump(app, 0.1)
    assert win.store.project.analysis.perception == {}                   # back to the defaults, as it started
    assert ctl.settings.classes[0]['min_px'] == 20 and pg.classes.item(0, 3).text() == '20'
    _wait(app, lambda: ctl.result.classes[0].min_px == 20 and not ctl.busy and not ctl.stale, 60)
    assert ctl.result.at(0, 0.0)[0] == pytest.approx(far0)
    # add an object class; every view follows; export what is shown
    QTest.mouseClick(pg.b_add, Qt.MouseButton.LeftButton)
    pump(app, 0.1)
    assert pg.classes.rowCount() == 5 and pg.classes.item(4, 0).text() == 'Object'
    _wait(app, lambda: len(ctl.result.classes) == 5 and not ctl.busy, 60)
    pump(app, 0.1)
    assert pg.verdicts.rowCount() == 5 and len(pg.pick.buttons) == 5
    paths = pg.export(str(tmp_path))
    assert [os.path.basename(x) for x in paths] == ['perception_range.json', 'perception_range.png']
    assert json.loads((tmp_path / 'perception_range.json').read_text())['classes'][4]['name'] == 'Object'


def test_a_camera_edit_reruns_and_leaving_the_page_stops_it(app, win):
    pg = _open(app, win)
    ctl = win.ctx.perception
    _wait(app, lambda: ctl.result is not None and not ctl.busy, 60)
    before = ctl.result
    c = win.store.variant.by_tag('F')
    win.store.set_cam(c.uid, pitch=c.pitch + 10.0)
    settle(app, win)
    _wait(app, lambda: ctl.result is not before and not ctl.busy and not ctl.stale, 60)
    win.actions_by_id['blind_volume'].trigger()                           # another page: no more re-runs
    pump(app, 0.2)
    assert not ctl.active and not pg.isVisible()
    shown = ctl.result
    win.store.set_cam(c.uid, pitch=c.pitch + 5.0)
    settle(app, win)
    pump(app, 0.5)
    assert ctl.stale and ctl.result is shown

