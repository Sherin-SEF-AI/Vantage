"""Installation check with board corners picked by hand: the fallback for a camera the detection fails on (glare, a
board half hidden). Four or more picks give the whole pose, two or three the rotation with the position as planned,
and the picked pose seeds the detection again; in the core, in batch mode and on the page."""
import argparse
import json
import os

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np
import pytest

from vantage.core import samples, capture
from vantage.core.analysis import Engine
from vantage.core.install import (check_all, check_camera, picks_from_rows, picks_to_rows, planned_corners,
                                  solve_from_picks)
from vantage.core.mathutil import rot_zyx
from vantage.core.synth import render_capture

from test_gui import app, win, pump, _wait   # noqa: F401  (fixtures)


@pytest.fixture(scope='module')
def sample(app):
    p = samples.tigor_sample_project()
    return p, Engine()


def _pose(cam, d):
    return rot_zyx(cam.yaw + d[0], cam.pitch + d[1], cam.roll + d[2]), cam.pos() + np.asarray(d[3:])


def _picks(p, cam, shape, pose, noise=0.0, seed=0, boards=2, corners=4):
    """Where the boards' corners really are in the image (through the true pose), as a person would pick them."""
    rng = np.random.default_rng(seed)
    out = {}
    for tu, _, uv, ok in planned_corners(p, cam, shape, pose=pose):
        ks = [k for k in range(4) if ok[k]][:corners]
        if len(ks) >= 2 and len(out) < boards:
            out[tu] = {k: tuple(uv[k] + rng.normal(0, noise, 2)) for k in ks}
    return out


D = {'F': (1.2, -0.9, 0.6, 0.012, -0.008, 0.01), 'QR': (-0.8, 0.7, -0.5, -0.01, 0.012, 0.006)}


@pytest.mark.parametrize('tag', list(D))
def test_picked_corners_solve_a_camera_whose_boards_cannot_be_detected(sample, tag):
    """Glare stand-in: the boards are not in the image at all, so the detection finds nothing; corners picked within
    a pixel still give the whole pose."""
    p, e = sample
    c = p.variant.by_tag(tag)
    d = np.array(D[tag])
    pose = _pose(c, d)
    img = render_capture(p, e, c, 960, 768, pose=pose, seed=4, targets=[])
    assert check_camera(p, c, img).pos is None
    r = solve_from_picks(p, c, img, _picks(p, c, img.shape, pose, noise=0.7, seed=1))
    assert r.method == 'picked' and r.pos is not None and r.picked >= 6
    assert np.abs(r.dang_deg - d[:3]).max() < 0.6
    assert np.abs(r.dpos_mm - d[3:] * 1000).max() < 30
    assert 'picked corners (less certain)' in r.message
    exact = solve_from_picks(p, c, img, _picks(p, c, img.shape, pose))
    assert np.abs(exact.dang_deg - d[:3]).max() < 0.02 and np.abs(exact.dpos_mm - d[3:] * 1000).max() < 1.0
    assert max(e for *_, e in exact.pick_err_px) < 0.05


def test_two_or_three_picks_give_the_rotation_with_the_position_as_planned(sample):
    p, e = sample
    c = p.variant.by_tag('F')
    d = np.array([1.5, -1.0, 0.8, 0, 0, 0])
    pose = _pose(c, d)
    img = render_capture(p, e, c, 960, 768, pose=pose, seed=4, targets=[])
    for n in (2, 3):
        r = solve_from_picks(p, c, img, _picks(p, c, img.shape, pose, boards=1, corners=n))
        assert r.method == 'picked-rotation' and r.picked == n
        assert np.allclose(r.pos, c.pos()) and np.abs(r.dpos_mm).max() == 0
        assert np.abs(r.dang_deg - d[:3]).max() < 0.05
        assert 'position as planned' in r.message
    one = solve_from_picks(p, c, img, _picks(p, c, img.shape, pose, boards=1, corners=1))
    assert one.pos is None and one.status == 'serious' and 'at least two' in one.message


def test_picks_seed_the_detection_which_then_replaces_them(sample):
    """With the boards in the image, rough picks only point the detection at them: the result is as precise as the
    automatic check's (the detected corners replace the picks)."""
    p, e = sample
    c = p.variant.by_tag('QR')
    d = np.array(D['QR'])
    img = render_capture(p, e, c, 960, 768, pose=_pose(c, d), seed=4)
    auto = check_camera(p, c, img)
    r = solve_from_picks(p, c, img, _picks(p, c, img.shape, _pose(c, d), noise=3.0, seed=2))
    assert r.method == 'picked+detected' and r.picked == 0 and len(r.boards) == len(auto.boards)
    assert r.rms_px < 1.0 and np.abs(r.dang_deg - d[:3]).max() < 0.3
    assert np.abs(r.dpos_mm - d[3:] * 1000).max() < 8
    assert 'picked corners found the boards' in r.message


def test_a_wrong_pick_is_named(sample):
    """A corner dropped on the wrong corner of the board is far off the solve, and the message says so."""
    p, e = sample
    c = p.variant.by_tag('F')
    pose = _pose(c, D['F'])
    img = render_capture(p, e, c, 960, 768, pose=pose, seed=4, targets=[])
    pk = _picks(p, c, img.shape, pose)
    tu = next(iter(pk))
    cs = pk[tu]
    k0, k1 = sorted(cs)[:2]
    cs[k0], cs[k1] = cs[k1], cs[k0]                    # two corners swapped
    r = solve_from_picks(p, c, img, pk)
    assert 'off the solve: check it is the right corner' in r.message


def test_check_all_routes_picked_cameras_and_keeps_rotation_only_out_of_the_joint_solve(sample):
    p, e = sample
    cams = [p.variant.by_tag(t) for t in ('F', 'ML', 'QR')]
    imgs, picks = {}, {}
    for c in cams:
        imgs[c.uid] = render_capture(p, e, c, 960, 768, seed=c.slot)
    f = cams[0]
    imgs[f.uid] = render_capture(p, e, f, 960, 768, seed=1, targets=[])         # F: the detection finds nothing
    picks[f.uid] = _picks(p, f, imgs[f.uid].shape, _pose(f, np.zeros(6)), boards=1, corners=2)
    res, jr = check_all(p, cams, imgs, picks=picks)
    assert res[f.uid].method == 'picked-rotation' and res[f.uid].pos is not None
    assert all(res[c.uid].method == 'detected' for c in cams[1:])
    assert jr is not None and jr.poses and f.uid not in jr.poses
    assert res[f.uid].status == 'good' and 'position as planned' in res[f.uid].message


def test_picks_rows_round_trip_and_bad_rows(sample):
    p, _ = sample
    t1, t2 = p.targets[0], p.targets[1]
    picks = {t1.uid: {0: (10.0, 20.5), 2: (30.25, 40.0)}, t2.uid: {3: (1.0, 2.0)}}
    rows = picks_to_rows(p, picks)
    assert rows[0] == {'board': t1.name, 'corner': 0, 'x': 10.0, 'y': 20.5} and len(rows) == 3
    back, bad = picks_from_rows(p, json.loads(json.dumps(rows)))
    assert back == picks and bad == []
    back, bad = picks_from_rows(p, [{'board': 'nope', 'corner': 0, 'x': 1, 'y': 2},
                                    {'board': t1.name, 'corner': 7, 'x': 1, 'y': 2},
                                    {'board': t1.name, 'corner': 1, 'x': 'nan', 'y': 2}, {'x': 1}])
    assert back == {} and len(bad) == 4


# ----------------------------------------------------------------------------- batch
def _args(**kw):
    from vantage import batch
    ap = argparse.ArgumentParser()
    ap.add_argument('path', nargs='?')
    ap.add_argument('--sample', action='store_true')
    ap.add_argument('--template')
    batch.add_arguments(ap)
    a = ap.parse_args([])
    a.sample = True
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def test_batch_install_picks(sample, tmp_path):
    from vantage import batch
    p, e = sample
    caps = tmp_path / 'caps'
    caps.mkdir()
    for c in p.variant.cameras:
        capture.save_image(str(caps / f'{c.tag}.png'), render_capture(p, e, c, 960, 768, seed=5,
                                                                      targets=[] if c.tag == 'F' else None))
    rep = tmp_path / 'install.json'
    assert batch.run(_args(install_check=str(caps), report=str(rep))) == 3            # F: no board found
    f = p.variant.by_tag('F')
    pk = _picks(p, f, (768, 960), _pose(f, np.zeros(6)))
    names = {t.uid: t.name for t in p.targets}
    picks_file = tmp_path / 'picks.json'
    picks_file.write_text(json.dumps({'f': [dict(r) for r in picks_to_rows(p, pk)],
                                      'NOPE': [{'board': 'T1', 'corner': 0, 'x': 1, 'y': 1}]}))
    code = batch.run(_args(install_check=str(caps), report=str(rep), install_picks=str(picks_file)))
    doc = json.loads(rep.read_text('utf-8'))
    row = next(r for r in doc['cameras'] if r['tag'] == 'F')
    assert code == 0, [(c['tag'], c['status'], c['message']) for c in doc['cameras']]
    assert row['method'] == 'picked' and row['status'] == 'good'
    assert {(r['board'], r['corner']) for r in row['picked_corners']} == {(names[u], k) for u, cs in pk.items() for k in cs}
    assert all(e['px'] < 0.1 for e in row['pick_error_px'])
    assert all(r['method'] == 'detected' for r in doc['cameras'] if r['tag'] != 'F')
    # the report itself can be fed back as the picks
    assert batch.run(_args(install_check=str(caps), report=str(tmp_path / 'again.json'), install_picks=str(rep))) == 0
    with pytest.raises(SystemExit, match='goes with --install-check'):
        batch.run(_args(analyse=str(tmp_path / 'a.json'), install_picks=str(picks_file)))


# ----------------------------------------------------------------------------- the page
def test_page_pick_corners_by_dragging(app, win, tmp_path):
    """On the installation page: Pick corners enlarges the camera, its board corners are dragged onto the image with
    the mouse (a right-click takes a pick back), and the check solves the camera from them."""
    from PyQt6.QtCore import Qt, QPointF
    from PyQt6.QtTest import QTest
    win.set_page('install')
    pg = win.page('install')
    p = win.store.project
    f = win.store.variant.by_tag('F')
    eng = Engine()
    d = np.array([0.9, -0.7, 0.5, 0, 0, 0])
    pose = _pose(f, d)
    img = render_capture(p, eng, f, 960, 768, pose=pose, seed=2, targets=[])       # no board to detect
    pg.select(f.uid)
    pg.b_pick.setChecked(True)                       # no capture yet: refused
    assert not pg.pick_mode and not pg.b_pick.isChecked()
    pg.set_image(f.uid, img)
    pg.b_pick.setChecked(True)
    pump(app, 0.3)
    tile = pg.tiles[f.uid]
    assert pg.pick_mode and pg.single and tile.picking() and tile.isVisible()
    handles = tile.handles()
    assert handles and not any(h[5] for h in handles)
    truth = {tu: uv for tu, _, uv, _ in planned_corners(p, f, img.shape, pose=pose)}
    rect, sc = tile._geometry()
    moved = 0
    for tu, name, k, at, guide, _ in handles:
        to = QPointF(rect.left() + truth[tu][k][0] * sc, rect.top() + truth[tu][k][1] * sc)
        if not tile.rect().contains(to.toPoint()):
            continue
        QTest.mousePress(tile, Qt.MouseButton.LeftButton, pos=at.toPoint())
        QTest.mouseMove(tile, pos=QPointF((at.x() + to.x()) / 2, (at.y() + to.y()) / 2).toPoint())
        assert tile._drag == (tu, k)                 # the magnifier shows while dragging
        tile.repaint()
        QTest.mouseMove(tile, pos=to.toPoint())
        QTest.mouseRelease(tile, Qt.MouseButton.LeftButton, pos=to.toPoint())
        moved += 1
        if moved == 6:
            break
    assert pg.n_picks(f.uid) == moved >= 4 and tile._drag is None
    assert 'picked' in pg.table.item([c.uid for c in pg._cams()].index(f.uid), 1).text()
    assert 'Run check' in pg.status.text()
    last = next(h for h in tile.handles() if h[5])
    QTest.mouseClick(tile, Qt.MouseButton.RightButton, pos=last[3].toPoint())
    assert pg.n_picks(f.uid) == moved - 1
    pg.run(check=True)
    _wait(app, lambda: not pg.busy and pg.results)
    r = pg.results[f.uid]
    assert r.method == 'picked' and r.pos is not None, r.message
    assert np.abs(r.dang_deg - d[:3]).max() < 0.5      # picks land on whole screen pixels of a scaled-down tile
    assert pg.images[f.uid] is img and pg.n_picks(f.uid) == moved - 1      # the capture and its picks are kept
    row = next(x for x in pg.report()['cameras'] if x['tag'] == 'F')
    assert row['method'] == 'picked' and len(row['picked_corners']) == moved - 1
    pg.clear_picks(f.uid)
    assert not pg.picks and not pg.b_clear_picks.isEnabled()
    pg.b_pick.setChecked(False)
    assert not pg.pick_mode
    win.set_page('viewports')
