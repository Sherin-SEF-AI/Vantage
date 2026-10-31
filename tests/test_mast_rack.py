"""A sensor mast fitted together with a roof rack stands on the rack's platform, whichever is added first; the cameras
and sensors on the mast go up with it; a mast on a bare roof and saved projects stay as they were."""
import types

import numpy as np
import pytest

from vantage.core import rig as R
from vantage.core import samples
from vantage.core import sensors as S
from vantage.core.project import Camera, Project


@pytest.fixture(scope='module')
def sample():
    p = samples.tigor_sample_project()
    p.active_uid = next(v.uid for v in p.variants if v.sensors)
    return p


def _fit(p, *kinds):
    for k in kinds:
        R.apply_fit(p, R.fit_plan(p, k))
    return p


def _by(p, suffix):
    return next(q for q in p.rig if q.name.endswith(suffix))


def _lidar_on_head(p):
    L1 = next(s for s in p.variant.sensors if s.tag == 'L1')
    pose = S.default_pose('lidar', p.sensor_model_for(L1), p.vehicle.mesh, p.rig)
    for k, v in pose.items():
        setattr(L1, k, v)
    return L1


def _strictly_inside(p, V, eps=1e-6):
    """Vertices V (n, 3) strictly inside part p's solid (more than eps from its surface)."""
    u = (V - np.array([p.x, p.y, p.z])) @ R._rot(p.yaw)
    if p.kind == 'box':
        return np.all(np.abs(u) < np.array([p.length, p.width, p.height]) / 2 - eps, axis=1)
    if p.kind == 'cylinder':
        return (np.hypot(u[:, 0], u[:, 1]) < p.radius - eps) & (np.abs(u[:, 2]) < p.height / 2 - eps)
    return (np.hypot(u[:, 0], u[:, 2]) < p.radius - eps) & (np.abs(u[:, 1]) < p.length / 2 - eps)


def _intersect(a, b) -> bool:
    """Do the two parts' solids overlap: their mesh bounding boxes overlap in volume and a vertex of one (or the
    segment of the tube's axis) lies inside the other."""
    Va, Vb = R.part_geometry(a)[0], R.part_geometry(b)[0]
    lo, hi = np.maximum(Va.min(0), Vb.min(0)), np.minimum(Va.max(0), Vb.max(0))
    if np.any(hi - lo <= 1e-6):
        return False
    axis = np.c_[np.full(50, a.x), np.full(50, a.y), np.linspace(a.z - a.height / 2, a.z + a.height / 2, 50)]
    return bool(_strictly_inside(b, Va).any() or _strictly_inside(a, Vb).any() or _strictly_inside(b, axis).any())


# ------------------------------------------------------------------ geometry
def test_a_mast_on_a_bare_roof_is_where_it_was(sample):
    top = R.roof(sample.vehicle.mesh)[3]
    mast, head = R.preset('sensor_mast', sample.vehicle.mesh)
    assert (mast.z, head.z) == (round(top + 0.25, 3), round(top + 0.56, 3))         # 1.782, 2.092 on the sample
    again = R.preset('sensor_mast', sample.vehicle.mesh, [])
    assert [(q.z, q.height) for q in again] == [(mast.z, mast.height), (head.z, head.height)]
    p = _fit(sample.snapshot(), 'sensor_mast')
    assert [q.z for q in p.rig] == [mast.z, head.z]


@pytest.mark.parametrize('order', [('roof_rack', 'sensor_mast'), ('sensor_mast', 'roof_rack')])
def test_the_mast_stands_on_the_platform_in_either_order(sample, order):
    p = _fit(sample.snapshot(), *order)
    top = R.roof(p.vehicle.mesh)[3]
    mast, head, plat = _by(p, 'Sensor mast'), _by(p, 'head'), _by(p, 'platform')
    # the meshes: the tube's lowest vertex is the platform's highest, and no vertex of either is inside the other
    Vm, Vp, Vh = (R.part_geometry(q)[0] for q in (mast, plat, head))
    assert Vm[:, 2].min() == pytest.approx(Vp[:, 2].max(), abs=1e-6)
    assert Vh[:, 2].min() == pytest.approx(Vm[:, 2].max(), abs=1e-6)
    assert not _strictly_inside(plat, Vm).any() and not _strictly_inside(mast, Vp).any()
    assert not _intersect(mast, plat) and not _intersect(head, plat)
    for a in p.rig:                                   # nothing of the rig cuts through anything else
        for b in p.rig:
            if a is not b and a.kind != 'bar' and b.kind != 'bar':
                assert not _intersect(a, b), (a.name, b.name)
    # it stands on the platform (its foot inside the platform's outline), the tube as long as on a bare roof
    assert abs(mast.x - plat.x) < plat.length / 2 and abs(mast.y - plat.y) < plat.width / 2
    assert mast.height == 0.5 and R.bottom_of(mast) - top == pytest.approx(R.top_of(plat) - top, abs=1e-6)
    assert R.top_of(head) - top == pytest.approx(0.62 + R.top_of(plat) - top, abs=1e-6)
    # nothing of the mast is left in the gap between the roof and the platform
    assert R.contains(p.rig, (mast.x, mast.y, top + 0.04)) == -1


def test_both_orders_give_the_same_rig_and_the_old_sum_intersects(sample):
    a = _fit(sample.snapshot(), 'roof_rack', 'sensor_mast')
    b = _fit(sample.snapshot(), 'sensor_mast', 'roof_rack')
    key = lambda p: sorted((q.name, q.kind, round(q.x, 4), round(q.y, 4), round(q.z, 4), q.height) for q in p.rig)
    assert key(a) == key(b)
    # what both orders gave before: the tube from the roof up through the platform
    old = R.preset('roof_rack', sample.vehicle.mesh) + R.preset('sensor_mast', sample.vehicle.mesh)
    assert _intersect(_by(types.SimpleNamespace(rig=old), 'Sensor mast'), _by(types.SimpleNamespace(rig=old), 'platform'))


def test_other_parts_under_a_mast(sample):
    top = R.roof(sample.vehicle.mesh)[3]
    p = _fit(sample.snapshot(), 'enclosure', 'sensor_mast')             # a mast on an enclosure stands on its top
    assert R.bottom_of(_by(p, 'Sensor mast')) == pytest.approx(R.top_of(_by(p, 'enclosure')), abs=1e-6)
    p = _fit(sample.snapshot(), 'sensor_mast', 'bar')                   # a crossbar under the mast lifts it onto it
    assert R.bottom_of(_by(p, 'Sensor mast')) == pytest.approx(R.top_of(_by(p, 'Crossbar')), abs=1e-6)
    # a mast moved off the rack's outline stays on the roof when the rack comes
    p = _fit(sample.snapshot(), 'sensor_mast')
    for q in p.rig:
        q.x -= 3.0
    _fit(p, 'roof_rack')
    assert R.bottom_of(_by(p, 'Sensor mast')) == pytest.approx(top, abs=1e-6)
    # a hidden mast is not lifted (it is not fitted); masts are never stood on a mast
    p = _fit(sample.snapshot(), 'sensor_mast')
    p.rig[0].visible = False
    plan = R.fit_plan(p, 'roof_rack')
    assert plan[1] == {}


# ------------------------------------------------------------------ what rides on the mast
def test_the_devices_on_the_mast_go_up_with_it(sample):
    p = _fit(sample.snapshot(), 'sensor_mast')
    top = R.roof(p.vehicle.mesh)[3]
    head = _by(p, 'head')
    L1 = _lidar_on_head(p)
    z_l1 = L1.z
    assert z_l1 == pytest.approx(R.top_of(head) + p.sensor_model_for(L1).optical_z_mm / 1000, abs=1e-3)
    # a camera on the head's front face (as Place on surface puts it) and one in another variant on the same head
    cam = Camera(tag='RF', name='Roof front', x=head.x + head.length / 2 + 0.012, y=0.0, z=head.z, slot=6)
    p.variant.cameras.append(cam)
    other = next(v for v in p.variants if v is not p.variant)
    cam2 = Camera(tag='RF', name='Roof front', x=head.x + head.length / 2 + 0.012, y=0.0, z=head.z + 0.02, slot=6)
    other.cameras.append(cam2)
    # a GNSS antenna on the roof at the mast's foot and the bumper radars stay where they are
    gm = p.sensor_models[S.default_model_id('gnss')]
    from vantage.core.project import Sensor
    gn = Sensor(tag='GN1', kind='gnss', model_id=gm.id, x=head.x - 0.06, y=0.0,
                z=round(top + gm.optical_z_mm / 1000, 3))
    p.variant.sensors.append(gn)
    still = {s.uid: s.z for v in p.variants for s in v.sensors if s.uid not in (L1.uid,)}
    still.update({c.uid: c.z for v in p.variants for c in v.cameras if c.uid not in (cam.uid, cam2.uid)})
    parts, dz, moved = R.fit_plan(p, 'roof_rack')
    plat = next(q for q in parts if q.kind == 'box')
    lift = R.top_of(plat) - top
    assert lift == pytest.approx(0.116, abs=1e-6)
    assert set(dz) == {q.uid for q in p.rig} and all(v == pytest.approx(lift) for v in dz.values())
    assert {d.uid for _, d, _ in moved} == {L1.uid, cam.uid, cam2.uid}
    R.apply_fit(p, (parts, dz, moved))
    assert L1.z == pytest.approx(z_l1 + lift, abs=1e-4)
    assert cam.z == pytest.approx(head.z, abs=1e-4) and cam2.z == pytest.approx(head.z + 0.02, abs=1e-4)
    assert (L1.x, L1.y) == (round(head.x, 3), round(head.y, 3))
    after = {s.uid: s.z for v in p.variants for s in v.sensors}
    after.update({c.uid: c.z for v in p.variants for c in v.cameras})
    assert all(after[u] == z for u, z in still.items())
    # the same rig and lidar as rack first, then mast, then Add lidar
    q = _fit(sample.snapshot(), 'roof_rack', 'sensor_mast')
    assert _lidar_on_head(q).z == pytest.approx(L1.z, abs=1e-3)


def test_adding_the_rack_in_the_app_is_one_undo_step(sample):
    from vantage.ui.rigview import add_preset
    from vantage.ui.store import ProjectStore
    st = ProjectStore(sample.snapshot())
    ctx = types.SimpleNamespace(store=st)
    mast, head = add_preset(ctx, 'sensor_mast')
    L1 = _lidar_on_head(st.project)
    z0, zl = (mast.z, head.z), L1.z
    n = st.undo.count()
    parts = add_preset(ctx, 'roof_rack')
    assert st.undo.count() == n + 1 and len(st.project.rig) == 5
    assert 'Sensor mast lifted onto it by 116 mm with 1 device on it' in st.undo.text(n)
    L1 = next(s for s in st.project.variant.sensors if s.tag == 'L1')
    assert L1.z == pytest.approx(zl + 0.116, abs=1e-4)
    assert st.sel_obj == ('rig', parts[0].uid)
    st.undo.undo()
    m = {q.name: q.z for q in st.project.rig}
    assert len(st.project.rig) == 2 and (m['Sensor mast'], m['Sensor mast · head']) == z0
    assert next(s for s in st.project.variant.sensors if s.tag == 'L1').z == zl
    st.undo.redo()
    assert next(s for s in st.project.variant.sensors if s.tag == 'L1').z == pytest.approx(zl + 0.116, abs=1e-4)
    st.undo.clear()                 # (a stack left dirty signals its deleted store when it is torn down)


# ------------------------------------------------------------------ saved projects
def test_a_saved_project_loads_as_it_was_saved(sample, tmp_path):
    """No migration: a project saved with the old intersecting mast keeps its parts and sensors where they were saved
    (they may have been measured on the real vehicle); only a new fit stands a mast on a rack."""
    p = sample.snapshot()
    p.rig = R.preset('roof_rack', p.vehicle.mesh) + R.preset('sensor_mast', p.vehicle.mesh)
    L1 = _lidar_on_head(p)
    path = str(tmp_path / 'old.vantage')
    p.save(path)
    q = Project.load(path)
    assert [r.to_dict() for r in q.rig] == [r.to_dict() for r in p.rig]
    assert next(s for s in q.variant.sensors if s.tag == 'L1').z == L1.z
    assert R.bottom_of(_by(q, 'Sensor mast')) == pytest.approx(R.roof(q.vehicle.mesh)[3], abs=1e-6)
    # a project saved with the fitted rig round-trips too
    r = _fit(sample.snapshot(), 'roof_rack', 'sensor_mast')
    r.save(path)
    assert [x.to_dict() for x in Project.load(path).rig] == [x.to_dict() for x in r.rig]


# ------------------------------------------------------------------ studies that see the mast
def test_thermal_and_vibration_carry_the_mast_on_the_rack(sample):
    from vantage.core import thermo_vib as TV
    p = _fit(sample.snapshot(), 'roof_rack', 'sensor_mast')
    tree = TV.rig_tree(p)
    by = {t['name']: t for t in tree}
    assert [tree[i]['name'] for i in by['Sensor mast']['supports']] == ['Roof rack · platform']
    assert by['Sensor mast · head']['supports'] == [next(i for i, t in enumerate(tree) if t['name'] == 'Sensor mast')]
    assert by['Sensor mast']['base'][2] == pytest.approx(R.top_of(_by(p, 'platform')), abs=1e-6)
    # the cantilever is still the 0.5 m tube: same length, same first mode alone
    modes = TV.part_modes(tree, {t['uid']: (0.0, 1.0) for t in tree}, TV.ThermoVibSettings())
    mast = next(m for m in modes.values() if m.kind == 'mast')
    assert np.linalg.norm(mast.tip - mast.base) == pytest.approx(0.5)


def test_the_lidar_and_cameras_see_the_mast_above_the_platform(sample):
    from vantage.core.analysis import Engine
    p = _fit(sample.snapshot(), 'roof_rack', 'sensor_mast')
    g = Engine().vehicle_geom(p)
    mast = _by(p, 'Sensor mast')
    names = [pt.name for pt in g.mesh.parts]
    k = names.index('Sensor mast')
    V = g.mesh.vertices[np.unique(g.mesh.faces[g.mesh.face_part == k])]
    assert V[:, 2].min() == pytest.approx(R.bottom_of(mast), abs=1e-6)
    assert V[:, 2].min() == pytest.approx(R.top_of(_by(p, 'platform')), abs=1e-6)
