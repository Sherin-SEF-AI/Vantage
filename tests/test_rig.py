"""The data-collection rig on the vehicle (roof racks, masts, crossbars, enclosures): it blocks cameras like the body,
cameras mount on it, it is saved, undone, exported and edited in the UI."""
import json
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from PyQt6.QtCore import Qt, QPoint
from PyQt6.QtTest import QTest

from vantage.assistant import tools as T
from vantage.core import exporters as ex, rig as R, rules, samples, volume as vol
from vantage.core.analysis import Engine
from vantage.core.mesh import Mesh
from vantage.core.project import Project, RigPart

from test_gui import app, win, pump, settle, vp_of, _wait     # noqa: F401  (fixtures)


@pytest.fixture(scope='module')
def base():
    p = samples.tigor_sample_project()
    e = Engine()
    return p, e, e.analyse(p)


def _plate(p):
    """A plate standing just in front of the front camera, across its view of the ground."""
    f = p.variant.by_tag('F')
    return RigPart(name='Plate', kind='box', x=f.x + 0.12, y=f.y, z=f.z - 0.05, length=0.02, width=0.5, height=0.3)


def _cam_vis(res, uid):
    return next(c for c in res.cams if c.uid == uid).vis


# ------------------------------------------------------------------ geometry
def test_parts_are_closed_solids_wound_outward():
    for kind in ('box', 'cylinder', 'bar'):
        p = RigPart(kind=kind, x=1.0, y=0.2, z=1.7, length=0.5, width=0.3, height=0.2, radius=0.05, yaw=30)
        V, F = R.part_geometry(p)
        T_ = V[F]
        n = np.cross(T_[:, 1] - T_[:, 0], T_[:, 2] - T_[:, 0])
        vol6 = np.einsum('ij,ij->i', T_[:, 0], n).sum()                   # divergence theorem: 6 × volume
        want = {'box': 0.5 * 0.3 * 0.2, 'cylinder': np.pi * 0.05 ** 2 * 0.2, 'bar': np.pi * 0.05 ** 2 * 0.5}[kind]
        assert vol6 / 6 == pytest.approx(want, rel=0.05)
        assert R.contains([p], (1.0, 0.2, 1.7)) == 0 and R.contains([p], (1.0, 0.2, 2.5)) == -1


def test_presets_are_sized_from_the_roof(base):
    p, e, _ = base
    lo, hi = p.vehicle.mesh.bbox
    xf, xr, hw, top = R.roof(p.vehicle.mesh)
    assert lo[0] < xr < xf < hi[0] and top == pytest.approx(hi[2]) and 0.3 < hw < hi[1]
    rack = R.preset('roof_rack', p.vehicle.mesh)
    bars = [q for q in rack if q.kind == 'bar']
    assert len(bars) == 2 and all(q.z > top and q.length > 2 * hw for q in bars)
    assert all(xr <= q.x <= xf for q in rack)
    mast, head = R.preset('sensor_mast', p.vehicle.mesh)
    assert mast.kind == 'cylinder' and mast.z - mast.height / 2 == pytest.approx(top, abs=1e-3)
    assert head.z - top == pytest.approx(0.5 + 0.06, abs=1e-3)                  # the head 0.5 m above the roof
    enc, = R.preset('enclosure', p.vehicle.mesh)
    assert enc.width < 2 * hw and enc.z - enc.height / 2 > top - 0.02
    # a smaller vehicle gets a smaller rack, lower down
    m = p.vehicle.mesh
    small = R.preset('bar', Mesh(m.vertices * 0.8, m.faces, m.face_part, m.parts))[0]
    assert small.length < bars[0].length and small.z < bars[0].z


# ------------------------------------------------------------------ analysis
def test_a_box_in_front_of_a_camera_blinds_the_ground_behind_it(base):
    p0, e, r0 = base
    p = p0.snapshot()
    p.rig = [_plate(p)]
    r1 = e.analyse(p)
    f = p.variant.by_tag('F')
    lost = _cam_vis(r0, f.uid) & ~_cam_vis(r1, f.uid) & r1.grid.roi
    assert lost.sum() * r1.grid.cell ** 2 > 0.5                               # F no longer sees past the plate
    assert r1.metrics['blind_m2'] > r0.metrics['blind_m2'] + 0.5
    assert r1.metrics['roi_area'] == pytest.approx(r0.metrics['roi_area'])     # the rig is not part of the footprint
    # the assistant explains why: hidden by the body, naming the rig part in the way
    jj, ii = np.nonzero(lost)
    k = len(jj) // 2
    x, y = float(r1.grid.xs[ii[k]]), float(r1.grid.ys[jj[k]])
    tc = T.ToolContext(p.snapshot(), r1, e.share())
    row = next(r for r in T.explain_point(tc, x=x, y=y)['cameras'] if r['camera'] == 'F')
    assert not row['seen'] and row['reason'] == 'hidden by the body' and 'Plate' in row['detail']
    # the blind volume sees it too
    v0, v1 = vol.analyse_volume(p0, e, r0, height=1.0, dz=0.25), vol.analyse_volume(p, e, r1, height=1.0, dz=0.25)
    assert v1.level_blind_m2.sum() > v0.level_blind_m2.sum()


def test_a_part_that_does_not_occlude_or_is_hidden_changes_nothing(base):
    p0, e, r0 = base
    p = p0.snapshot()
    p.rig = [_plate(p)]
    p.rig[0].occludes = False
    assert e.analyse(p).metrics['blind_m2'] == pytest.approx(r0.metrics['blind_m2'], abs=1e-9)
    p.rig[0].occludes, p.rig[0].visible = True, False
    assert e.analyse(p).metrics['blind_m2'] == pytest.approx(r0.metrics['blind_m2'], abs=1e-9)


def test_the_geometry_cache_follows_the_rig(base):
    p0, e, _ = base
    p = p0.snapshot()
    g0 = e.vehicle_geom(p)
    assert e.vehicle_geom(p) is g0                                             # no rig: the same geometry
    p.rig = [_plate(p)]
    n = len(R.part_geometry(p.rig[0])[1])
    g1 = e.vehicle_geom(p)
    assert g1 is not g0 and g1.occ_mesh.n_tris == g0.occ_mesh.n_tris + n
    assert g1.body is g0.body                                                  # the simplified body is reused
    p.rig[0].x += 0.3                                                          # a moved part: new occluder
    g2 = e.vehicle_geom(p)
    assert g2 is not g1 and g2.occ_mesh.tri[-n:, :, 0].max() == pytest.approx(p.rig[0].x + 0.01)
    p.rig[0].x -= 0.3                                                          # moved back: the cached one again
    assert e.vehicle_geom(p) is g1
    p.rig[0].occludes = False
    assert e.vehicle_geom(p).occ_mesh.n_tris == g0.occ_mesh.n_tris            # see-through: not in the occluder
    p.rig = []
    g = e.vehicle_geom(p)                     # (g0 itself may have left the small cache by now)
    assert g.occ_mesh is g0.body and g.mesh is p.vehicle.mesh


def test_a_camera_mounts_on_a_mast(base):
    p0, e, _ = base
    p = p0.snapshot()
    p.rig = R.preset('sensor_mast', p.vehicle.mesh)
    head = p.rig[1]
    c = p.variant.by_tag('F')
    c.x, c.y, c.z = head.x, head.y, head.z + head.height / 2 + c.standoff_mm / 1000
    vg = e.vehicle_geom(p)
    mi = rules.mount_info(p, c, vg)
    assert mi.surface == head.name and not mi.floating and mi.zone_kind == ''
    assert mi.normal[2] > 0.99                                                 # on top of the head
    p.rig[1].mountable = False
    mi = rules.mount_info(p, c, e.vehicle_geom(p))
    assert mi.surface == head.name and mi.zone_kind == 'forbidden'
    issues = rules.collect_issues(p, None, e.vehicle_geom(p), {c.uid: mi})
    assert any('forbidden' in i.message and c.uid in i.uids for i in issues)


# ------------------------------------------------------------------ project
def test_save_load_round_trip_and_old_projects(base, tmp_path):
    p = base[0].snapshot()
    p.rig = R.preset('roof_rack', p.vehicle.mesh)
    p.rig[0].yaw, p.rig[1].occludes, p.rig[2].mountable = 12.5, False, False
    path = str(tmp_path / 'rig.vantage')
    p.save(path)
    q = Project.load(path)
    assert [r.to_dict() for r in q.rig] == [r.to_dict() for r in p.rig]
    # an old project without a rig loads with none
    data = p.to_json()
    data.pop('rig')
    assert 'rig' not in data
    p.rig = []
    p.save(path)
    assert Project.load(path).rig == []


def test_exports_list_the_rig(base, tmp_path):
    p0, e, r0 = base
    p = p0.snapshot()
    p.rig = R.preset('roof_rack', p.vehicle.mesh) + R.preset('sensor_mast', p.vehicle.mesh)
    p.rig[0].visible = False
    ex.export(['extrinsics_json', 'ros2', 'report_html'], p, r0,
              ex.ExportOptions(out_dir=str(tmp_path), frame='front_bumper'))
    doc = json.loads(next(tmp_path.rglob('extrinsics.json')).read_text('utf-8'))
    names = [r['name'] for r in doc['rig']]
    assert names == [q.name for q in p.rig if q.visible]
    mast = next(r for r in doc['rig'] if r['kind'] == 'cylinder')
    fx = p.vehicle.measured()['front_x']
    assert mast['centre'][0] == pytest.approx(p.rig[3].x - fx, abs=1e-3) and mast['radius'] == pytest.approx(0.03)
    assert mast['occludes'] is True and mast['mountable'] is True
    root = ET.parse(next(tmp_path.rglob('vantage_cameras.urdf'))).getroot()
    links = {l.get('name'): l for l in root.findall('link')}
    joints = {j.find('child').get('link'): j for j in root.findall('joint')}
    for r in doc['rig']:
        assert r['frame'] in links and joints[r['frame']].get('type') == 'fixed'
        geo = links[r['frame']].find('visual/geometry')
        assert geo.find('box' if r['kind'] == 'box' else 'cylinder') is not None
        assert links[r['frame']].find('collision/geometry') is not None
    bar = next(r for r in doc['rig'] if r['kind'] == 'bar')
    assert links[bar['frame']].find('visual/origin').get('rpy').startswith('1.5708')   # the bar runs along y
    report = next(tmp_path.rglob('report.html')).read_text('utf-8')
    assert '<h3>Rig</h3>' in report and 'Sensor mast · head' in report and 'Roof rack · bar 1' not in report
    # no rig: the files are as before
    ex.export(['extrinsics_json'], p0, r0, ex.ExportOptions(out_dir=str(tmp_path / 'none')))
    assert 'rig' not in json.loads(next((tmp_path / 'none').rglob('extrinsics.json')).read_text('utf-8'))


def test_the_assistant_overview_lists_the_rig(base):
    p0, e, r0 = base
    p = p0.snapshot()
    p.rig = R.preset('sensor_mast', p.vehicle.mesh)
    p.rig[1].occludes = False
    ov = T.get_overview(T.ToolContext(p, r0, e.share()))
    assert [r['name'] for r in ov['rig']] == ['Sensor mast', 'Sensor mast · head']
    assert ov['rig'][0]['kind'] == 'cylinder' and ov['rig'][0]['size_m'] == {'radius': 0.03, 'height': 0.5}
    assert ov['rig'][1]['occludes'] is False and ov['rig'][1]['mountable'] is True


# ------------------------------------------------------------------ UI
def _menu_click(app, menu, text):
    act = next(a for a in menu.actions() if a.text() == text)
    r = menu.actionGeometry(act)
    QTest.mouseMove(menu, r.center())
    pump(app, 0.1)
    QTest.mouseClick(menu, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, r.center())
    pump(app, 0.2)
    return act


def _rig_group(out):
    """The outliner's Rig group (looked up again after every rebuild)."""
    items = [out.topLevelItem(i) for i in range(out.topLevelItemCount())]
    return next(it for it in items if it.text(0).startswith('Rig ('))


def test_add_a_preset_from_the_vehicle_menu_edit_it_and_undo(app, win):
    st = win.store
    blind0 = win.ctx.analysis.result.metrics['blind_m2']
    i0 = st.undo.index()
    mb = win.menuBar()
    ve = next(a for a in mb.actions() if a.text() == 'Vehicle')
    QTest.mouseClick(mb, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, mb.actionGeometry(ve).center())
    pump(app, 0.2)
    assert ve.menu().isVisible()
    sub = _menu_click(app, ve.menu(), 'Add rig part')
    _wait(app, lambda: sub.menu().isVisible(), 5)
    _menu_click(app, sub.menu(), 'Sensor mast')
    assert [q.name for q in st.project.rig] == ['Sensor mast', 'Sensor mast · head']
    assert st.undo.index() == i0 + 1                                           # one undo step for the preset
    mast = st.project.rig[0]
    assert st.sel_obj == ('rig', mast.uid)
    settle(app, win)
    # the inspector shows the part; the outliner lists the rig
    assert win.inspector.current == 'rig'
    page = win.inspector.pages['rig']
    assert page.s_part.isVisible() and page.name.text() == 'Sensor mast'
    assert page.pz.lineEdit().text() == f'{mast.z:.3f}'
    assert not page.sx.isEnabled() and page.sx.lineEdit().text() == '0.060'      # a cylinder: X shows its diameter
    out = win.outliner
    rig = _rig_group(out)
    assert rig.text(0) == 'Rig (2)'
    assert [rig.child(i).text(0) for i in range(rig.childCount())] == ['Sensor mast', 'Sensor mast · head']
    assert rig.child(0).isSelected()
    # type a new height into the inspector
    page.sz.window().activateWindow()
    page.sz.setFocus()
    pump(app, 0.05)
    page.sz.lineEdit().selectAll()
    QTest.keyClicks(page.sz.lineEdit(), '800 mm')
    QTest.keyClick(page.sz.lineEdit(), Qt.Key.Key_Return)
    settle(app, win)
    assert st.project.rig[0].height == pytest.approx(0.8)
    assert st.undo.index() == i0 + 2
    # select the head in the outliner and switch off its occlusion
    QTest.mouseClick(out.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
                     out.visualItemRect(_rig_group(out).child(1)).center())
    pump(app, 0.2)
    head = st.project.rig[1]
    assert st.sel_obj == ('rig', head.uid) and page.name.text() == 'Sensor mast · head'
    QTest.mouseClick(page.occ, Qt.MouseButton.LeftButton)
    settle(app, win)
    assert st.project.rig[1].occludes is False
    # undo walks it all back
    st.undo.undo()
    assert st.project.rig[1].occludes is True
    st.undo.undo()
    assert st.project.rig[0].height == pytest.approx(0.5)
    st.undo.undo()
    assert st.project.rig == []
    settle(app, win)
    assert win.ctx.analysis.result.metrics['blind_m2'] == pytest.approx(blind0, abs=1e-6)
    st.undo.redo()
    assert len(st.project.rig) == 2


def test_rig_is_drawn_picked_and_carries_a_placed_camera(app, win):
    st = win.store
    from vantage.ui.rigview import add_preset
    mast, head = add_preset(win.ctx, 'sensor_mast')
    st.select_obj(())
    settle(app, win)
    vp = vp_of(win)
    # drawn in 3D: the scene hands the renderer the rig's triangles
    tris = vp.provider.tris_extra(vp)
    assert any(len(t.tri) == len(R.part_geometry(head)[1]) + len(R.part_geometry(mast)[1]) for t in tris)
    s, _, ok = vp.project([[head.x, head.y, head.z]])
    assert ok[0]
    at = QPoint(int(round(s[0][0])), int(round(s[0][1])))
    # click selects the part
    QTest.mouseClick(vp, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, at)
    pump(app, 0.2)
    assert st.sel_obj == ('rig', head.uid)
    # the place tool puts a camera on the head
    n0 = len(st.variant.cameras)
    win.ctx.set_tool('place')
    QTest.mouseClick(vp, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, at)
    settle(app, win)
    win.ctx.set_tool('select')
    assert len(st.variant.cameras) == n0 + 1
    cam = st.variant.cameras[-1]
    mi = rules.mount_info(st.project, cam, win.ctx.analysis.engine.vehicle_geom(st.project))
    assert mi.surface == head.name and not mi.floating
    # Delete removes the selected part
    st.select_obj(('rig', mast.uid))
    win.actions_by_id['delete'].trigger()
    assert [q.uid for q in st.project.rig] == [head.uid]
