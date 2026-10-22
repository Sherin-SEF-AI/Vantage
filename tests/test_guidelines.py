"""Parking guidelines and the swept path: bicycle-model geometry, projection into the cameras (lens, image circle,
body occlusion), the ECU tables and their export, and the steering bar driven with real mouse events."""
import copy
import json
import math

import numpy as np
import pytest

from vantage.core import samples, exporters as ex, guidelines as G
from vantage.core.analysis import Engine, camera_frames
from vantage.core.lens import resolve
from vantage.core.project import Obstacle

from test_gui import app, win, pump      # noqa: F401  (fixtures)


@pytest.fixture(scope='module')
def sample():
    p = samples.tigor_sample_project()
    return p, Engine(), G.vehicle_outline(p)


# ----------------------------------------------------------------------------- vehicle and kinematics
def test_outline_from_the_mesh(sample):
    p, _, o = sample
    m = p.vehicle.measured()
    assert o.wheelbase == pytest.approx(m['Wheelbase'])
    assert o.front_x == pytest.approx(m['front_x']) and o.rear_x == pytest.approx(m['rear_x'])
    assert 2 * o.half_width == pytest.approx(m['Width'], abs=0.01)             # the body, without the mirrors
    assert o.hull[:, 1].max() > o.half_width + 0.1                             # the footprint has them
    assert o.mirror_z is not None and 0.5 < o.mirror_z < 1.3


def test_straight_wheels_give_straight_lines(sample):
    _, _, o = sample
    st = G.GuidelineSettings(angle_deg=0.0, margin_m=0.05)
    for direction, x0, sgn in (('reverse', o.rear_x, -1), ('forward', o.front_x, 1)):
        gs = G.guideline_set(o, st, direction, obstacles=[], swept=False)
        traj = [ln for ln in gs.lines if ln.kind == 'trajectory']
        assert {ln.color for ln in traj} == {'red', 'yellow', 'green'}
        for ln in traj:
            side = 1 if 'left' in ln.id else -1
            assert np.allclose(ln.pts[:, 1], side * (o.half_width + 0.05), atol=1e-9)
        xs = np.concatenate([ln.pts[:, 0] for ln in traj])
        assert (sgn * (xs - x0)).min() == pytest.approx(0, abs=1e-9)
        assert (sgn * (xs - x0)).max() == pytest.approx(st.length, abs=1e-9)
        for ln in (ln for ln in gs.lines if ln.kind == 'distance'):
            d = float(ln.label.split()[0])
            assert np.allclose(ln.pts[:, 0], x0 + sgn * d) and ln.color == st.band(d)
        static = [ln for ln in gs.lines if ln.kind == 'static' and ln.label]
        assert [ln.label for ln in static] == ['0.5 m', '1 m', '2 m', '3 m']


@pytest.mark.parametrize('delta', [12.0, -25.0, 35.0])
def test_turning_follows_the_bicycle_model(sample, delta):
    _, _, o = sample
    R = G.turn_radius(o, delta)
    assert R == pytest.approx(o.wheelbase / math.tan(math.radians(delta)))
    C = np.array([o.rear_axle_x, R])
    s = np.linspace(0, -4.0, 41)
    axle = G.trace(o, delta, [o.rear_axle_x, 0.0], s)
    assert np.allclose(np.linalg.norm(axle - C, axis=1), abs(R), atol=1e-9)      # the rear-axle centre on its circle
    rot, _ = G.poses(o, delta, s)
    heading = np.arctan2(rot[:, 1, 0], rot[:, 0, 0])
    assert np.allclose(heading, s / R, atol=1e-9)
    for corner in (o.corner('reverse', 1), o.corner('reverse', -1)):            # every body point turns about C
        tr = G.trace(o, delta, corner, s)
        assert np.allclose(np.linalg.norm(tr - C, axis=1), np.linalg.norm(corner - C), atol=1e-9)
    # turning left (+) while reversing takes the rear to the left
    assert np.sign(axle[-1, 1]) == np.sign(delta)


@pytest.mark.parametrize('direction,delta', [('reverse', 0.0), ('reverse', 22.0), ('forward', -30.0)])
def test_swept_path_holds_the_body_at_every_pose(sample, direction, delta):
    _, _, o = sample
    L = 3.0
    poly = G.swept_path(o, delta, direction, L)
    s = np.linspace(0, L, 61) * (-1 if direction == 'reverse' else 1)
    rot, t = G.poses(o, delta, s)
    body = np.concatenate(o.footprint)                                 # the body and both mirrors
    assert len(o.footprint) == 3
    P = (np.einsum('nij,kj->nki', rot, body) + t[:, None]).reshape(-1, 2)
    assert G._poly_distance(P, poly).max() < 0.03                      # inside, to the raster's ~2 cm
    straight = G.swept_path(o, 0.0, direction, L)
    if delta:
        w = lambda q: q[:, 1].max() - q[:, 1].min()
        assert w(poly) > w(straight) + 0.3                             # turning sweeps wider than the car
        assert abs(G._area(poly)) > abs(G._area(straight))


def test_obstacles_in_the_swept_path_are_hits(sample):
    p, _, o = sample
    st = G.GuidelineSettings(angle_deg=0.0, path_m=3.0)
    mirror_x = float(o.hull[np.argmax(o.hull[:, 1]), 0])
    y = (o.half_width + o.hull[:, 1].max()) / 2                       # beside the body, inside the mirror's sweep
    x = mirror_x - 2.0
    tall = Obstacle(name='Pillar', x=x, y=y, height=1.8, radius=0.02)
    low = Obstacle(name='Kerb stone', x=x, y=y, height=0.3, radius=0.02)
    far = Obstacle(name='Far', x=o.rear_x - 1.0, y=3.0, height=1.8, radius=0.2)
    gs = G.guideline_set(o, st, 'reverse', obstacles=[tall, low, far])
    hits = {ob.name: s for ob, s in gs.hits}
    assert set(hits) == {'Pillar'}                                     # the mirror passes over the low one
    assert hits['Pillar'] == pytest.approx(2.0 - 0.02, abs=0.05)


def test_settings_survive_bad_values():
    st = G.GuidelineSettings.from_dict({'angle_deg': 80, 'max_deg': 30, 'direction': 'sideways', 'distances': [3, 'x'],
                                        'path_m': float('nan'), 'ratio': -2, 'swept': 0, 'unknown': 1})
    assert st.max_deg == 30 and st.angle_deg == 30 and st.direction == 'reverse'
    assert st.distances == [0.5, 1.0, 2.0, 3.0] and st.path_m == 3.0 and st.ratio == 0 and st.swept is False
    st = G.GuidelineSettings.from_dict({'distances': [2, 0.5, 2, -1]})
    assert st.distances == [0.5, 2.0] and st.band(0.4) == 'red' and st.band(1.0) == 'yellow' and st.band(2.5) == 'green'


# ----------------------------------------------------------------------------- projection
def test_projected_pixels_land_back_on_the_ground_lines(sample):
    p, e, o = sample
    cam = p.variant.by_tag('R')
    gs = G.guideline_set(o, G.GuidelineSettings(angle_deg=18.0), 'reverse', obstacles=[])
    pr = G.project_ground(p, e, cam, gs.lines)
    intr = resolve(p.module_for(cam), p.lens_for(cam))
    _, W2O = camera_frames(cam)
    n = 0
    for ln, pieces in pr.pieces:
        for px, ground in pieces:
            assert (px[:, 0] >= 0).all() and (px[:, 0] <= intr.width).all()
            assert (px[:, 1] >= 0).all() and (px[:, 1] <= intr.height).all()
            d, _ = intr.unproject(px[:, 0], px[:, 1])
            dw = d @ W2O
            t = -cam.z / dw[:, 2]
            back = cam.pos()[:2] + dw[:, :2] * t[:, None]
            assert np.abs(back - ground).max() < 0.005
            n += len(px)
    assert n > 300                                                     # the rear camera sees its guidelines


def test_the_body_hides_and_splits_a_line(sample):
    p0, _, o = sample
    p = copy.deepcopy(p0)
    cam = p.variant.by_tag('F')
    cam.x, cam.y, cam.z, cam.yaw, cam.pitch, cam.roll = (o.front_x + o.rear_x) / 2, 0.0, 3.0, 0.0, 90.0, 0.0
    line = np.array([[o.rear_x - 2.5, 0.0], [o.front_x + 2.5, 0.0]])
    free = G.project_ground(p, None, cam, [line]).pieces[0][1]
    hidden = G.project_ground(p, Engine(), cam, [line]).pieces[0][1]
    assert len(free) == 1
    assert len(hidden) == 2                                           # in front of and behind the car
    behind, ahead = sorted(hidden, key=lambda pc: pc[1][:, 0].mean())
    assert behind[1][:, 0].max() < o.rear_x + 0.05 and ahead[1][:, 0].min() > o.front_x - 0.05


def test_directions_follow_the_cameras(sample):
    p, e, _ = sample
    assert G.visible_directions(p, e, p.variant.by_tag('R')) == ['reverse']
    assert G.visible_directions(p, e, p.variant.by_tag('F')) == ['forward']


# ----------------------------------------------------------------------------- tables and export
def test_table_for_the_ecu(sample):
    p, e, _ = sample
    st = G.GuidelineSettings(max_deg=30.0, step_deg=5.0)
    cam = p.variant.by_tag('R')
    doc = G.guideline_table(p, e, cam, settings=st)
    json.dumps(doc)
    rows = doc['directions']['reverse']
    assert [r['road_wheel_angle_deg'] for r in rows] == [-30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30]
    mid = rows[6]
    assert mid['turn_radius_m'] is None and mid['steering_wheel_deg'] == 0
    assert rows[-1]['turn_radius_m'] == pytest.approx(G.vehicle_outline(p).wheelbase / math.tan(math.radians(30)), abs=1e-3)
    assert rows[-1]['steering_wheel_deg'] == pytest.approx(30 * st.ratio)
    intr = resolve(p.module_for(cam), p.lens_for(cam))
    kinds = set()
    for r in rows:
        for ln in r['lines']:
            kinds.add(ln['kind'])
            assert ln['color'].startswith('#') and len(ln['color']) == 7
            for piece in ln['pixels']:
                a = np.array(piece)
                assert (a[:, 0] >= 0).all() and (a[:, 0] <= intr.width).all() and (a[:, 1] <= intr.height).all()
    assert kinds == {'trajectory', 'distance', 'swept'}
    assert {ln['kind'] for ln in doc['static']['reverse']} == {'static'}
    # a turn the other way mirrors the lines across the rear camera's image
    left, right = rows[-1], rows[0]
    lp = np.concatenate([np.array(pc) for ln in left['lines'] if ln['kind'] == 'trajectory' for pc in ln['pixels']])
    rp = np.concatenate([np.array(pc) for ln in right['lines'] if ln['kind'] == 'trajectory' for pc in ln['pixels']])
    assert lp[:, 0].mean() > intr.width / 2 > rp[:, 0].mean()        # the rear camera sees the car's left on its right


def test_export_and_preview(sample, tmp_path):
    p0, e, _ = sample
    p = copy.deepcopy(p0)
    p.analysis.guidelines = {'angle_deg': 15.0, 'step_deg': 5.0}
    p.obstacles = [Obstacle(name='Pillar', x=-2.4, y=1.3, height=1.8, radius=0.2)]
    res = e.analyse(p)
    opts = ex.ExportOptions(out_dir=str(tmp_path), engine=e, units='mm', frame='front_bumper')
    paths = ex.export(['guidelines'], p, res, opts)
    tags = [c.tag for c in p.variant.cameras if c.visible]
    assert sorted(x.split('/guidelines/')[-1] for x in paths) == sorted(f'{t}.{k}' for t in tags for k in ('json', 'png'))
    doc = json.loads((tmp_path / 'guidelines' / 'R.json').read_text('utf-8'))
    assert doc['schema'] == 'vantage.guidelines/1' and 'i + 0.5' in doc['pixel_coordinates']
    assert doc['units'] == 'mm' and 'front bumper' in doc['frame']
    assert doc['vehicle']['front_bumper_x'] == 0 and isinstance(doc['vehicle']['wheelbase'], int)
    pt = doc['static']['reverse'][0]['ground'][0]
    assert all(isinstance(v, int) for v in pt) and pt[0] < -3000           # mm from the front bumper
    assert list(doc['directions']) == ['reverse'] and len(doc['directions']['reverse']) == 15
    png = (tmp_path / 'guidelines' / 'R.png').read_bytes()
    assert png[:8] == b'\x89PNG\r\n\x1a\n'
    rel, text, size = ex.preview('guidelines', p, res, opts)
    assert rel.endswith('guidelines/F.json') and size > 1000 and 'vantage.guidelines/1' in text


# ----------------------------------------------------------------------------- GUI
def test_steering_bar_and_overlays(app, win):
    from PyQt6.QtCore import Qt, QPoint
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QStyle, QStyleOptionSlider
    from vantage.ui import guides
    st = win.store
    win.set_page('camviews')
    pump(app, 0.5)
    pg = win._page_widgets['camviews']
    bar = pg.guide_bar
    gc = guides.controller(win.ctx)
    assert not win.ctx.overlays.get('guidelines')
    QTest.mouseClick(bar.on, Qt.MouseButton.LeftButton)
    pump(app, 0.2)
    assert win.ctx.overlays['guidelines'] and win.actions_by_id['ov_guidelines'].isChecked()
    rear = st.variant.by_tag('R')
    assert any(pcs for _, pcs in gc.camera(rear).pieces)
    vp = win.viewarea.panes[0].vp3d
    assert any(abs(l.pts[..., 2].max() - guides.Z_LINES) < 1e-9 for l in vp.provider.lines(vp))
    # drag the slider handle to the left with the mouse: the views follow, the project only changes on release
    sl = bar.slider
    opt = QStyleOptionSlider()
    sl.initStyleOption(opt)
    h = sl.style().subControlRect(QStyle.ComplexControl.CC_Slider, opt, QStyle.SubControl.SC_SliderHandle, sl).center()
    n0 = st.undo.count()
    QTest.mousePress(sl, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, h)
    for dx in range(4, 61, 4):
        QTest.mouseMove(sl, QPoint(h.x() - dx, h.y()))
        pump(app, 0.02)
    assert gc.settings.angle_deg > 3.0                                  # previewed: left of centre is a left turn
    assert gc.stored().angle_deg == 0.0 and st.undo.count() == n0
    QTest.mouseRelease(sl, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(h.x() - 60, h.y()))
    pump(app, 0.2)
    a = gc.stored().angle_deg
    assert a > 3.0 and st.undo.count() == n0 + 1                        # one undo step for the whole drag
    assert st.project.analysis.guidelines['angle_deg'] == a
    assert 'road wheel' in bar.readout.text() and 'radius' in bar.readout.text()
    st.undo.undo()
    pump(app, 0.2)
    assert gc.stored().angle_deg == 0.0 and sl.value() == 0
    # direction and toggles are undoable settings too
    bar.dir.buttons['forward'].click()
    pump(app, 0.1)
    assert gc.stored().direction == 'forward'
    bar.checks['swept'].click()
    pump(app, 0.1)
    assert gc.stored().swept is False and gc.ground().swept is None
    st.undo.undo()
    st.undo.undo()
    pump(app, 0.1)
    assert gc.stored().direction == 'reverse' and gc.stored().swept is True
    win.actions_by_id['ov_guidelines'].trigger()
    pump(app, 0.1)
    assert not win.ctx.overlays['guidelines'] and not bar.on.isChecked()
