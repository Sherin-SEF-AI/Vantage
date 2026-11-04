"""The vehicle library (core/vehicles.py): real vehicles with their stated dimensions and sources, template bodies
that reproduce them, the old template keys, the 3D-model download (a fake server: no network), the COLLADA loader,
and the import wizard taking a library vehicle's model with its stated figures."""
import dataclasses
import io
import math
import os
import tarfile
import time
import zipfile

import numpy as np
import pytest

from vantage.core import loaders
from vantage.core import samples
from vantage.core import vehicles as VL
from vantage.core.analysis import Engine

LIB = VL.library()


def test_every_vehicle_is_real_and_sourced():
    assert len(LIB) >= 12 and len({v.id for v in LIB}) == len(LIB)
    for v in LIB:
        assert v.maker and v.name and v.generation and v.family in VL.FAMILY, v.id
        assert v.sources and all(u.startswith(('https://', 'http://')) and w for w, u in v.sources), v.id
        assert v.height > 0 and v.wheelbase > 0 and v.tyre, v.id
        if v.family != 'truck':
            assert v.length > v.wheelbase and v.width > 1.5, v.id
            assert 2.0 < v.wheelbase < 4.5 and 1.3 < v.height < 3.0, v.id
        if v.front_overhang and v.rear_overhang and v.length:
            assert v.front_overhang + v.wheelbase + v.rear_overhang == pytest.approx(v.length, abs=0.012), v.id
        if v.width_mirrors:
            assert v.width_mirrors > v.width, v.id
    # the public datasets' and fleets' vehicles are there
    used = ' '.join(v.used_by for v in LIB)
    assert 'nuScenes' in used and 'KITTI' in used and 'Waymo' in used and 'Apollo' in used


def test_tyre_sizes():
    assert VL.tyre_radius('215/55 R17') == pytest.approx((17 * 25.4 / 2 + 215 * 0.55) / 1000, abs=1e-4)
    assert VL.tyre_radius('P245/45R18') == pytest.approx((18 * 25.4 / 2 + 245 * 0.45) / 1000, abs=1e-4)
    assert VL.tyre_radius('LT245/75R16') > 0.38 and VL.tyre_radius('11R22.5') == 0.0
    assert VL.tyre_width('235/55 R19') == 0.235


@pytest.mark.parametrize('vid', [v.id for v in LIB])
def test_a_template_body_has_the_stated_dimensions(vid):
    v = VL.by_id()[vid]
    p = samples.template_project(vid)
    meas = p.vehicle.measured()
    spec = VL.body_spec(v)
    assert p.vehicle.template == vid and p.vehicle.name == v.title and p.vehicle.spec == v.stated()
    if v.trailer is None:
        for k in ('Length', 'Width', 'Height', 'Wheelbase'):
            assert meas[k] == pytest.approx(v.stated()[k], abs=0.006), (vid, k, meas[k])
    else:
        assert meas['Wheelbase'] == pytest.approx(v.wheelbase, abs=0.006)
        assert 'width' in spec.drawn and 'length' in spec.drawn
    # what was not stated is drawn, and said so; what was stated is never on that list
    for word, figure in (('overhangs', v.front_overhang or v.rear_overhang), ('ground clearance', v.ground_clearance),
                         ('track', v.front_track)):
        assert (word in spec.drawn) == (not figure) or v.family == 'truck', (vid, word)
    log = p.log[-1].text
    assert v.sources[0][0] in log and (not spec.drawn or 'drawn, not stated' in log)
    # the wheels stand on the ground, on the stated track
    wc = p.vehicle.wheel_centres
    r = VL.tyre_radius(v.tyre)
    if r and v.family != 'truck':
        assert wc['FL'][2] == pytest.approx(r, abs=1e-6)
        if v.front_track:
            hub = abs(wc['FL'][1]) - 0.0925 * r / 0.29
            assert hub == pytest.approx(v.front_track / 2, abs=0.006) or abs(wc['FL'][1]) <= v.width / 2


def test_the_old_template_keys_open_real_vehicles():
    names = {k: samples.template_project(k).vehicle.name for k in VL.ALIASES}
    assert names == {k: VL.by_id()[v].title for k, v in VL.ALIASES.items()}
    assert VL.resolve('sedan').id == 'lincoln_mkz_2017' and VL.resolve('nope') is None
    with pytest.raises(KeyError):
        samples.template_project('nope')
    r = Engine().analyse(samples.template_project('van'))
    assert r is not None


def test_only_freely_licensed_models_are_listed():
    ok = {'CC0-1.0', 'CC-BY-4.0', 'BSD-3-Clause', 'Apache-2.0', 'MIT'}
    have = [v for v in LIB if v.model3d]
    assert {v.id for v in have} >= {'lincoln_mkz_2017', 'toyota_prius_xw50', 'lexus_rx450h_2020'}
    for v in have:
        m = v.model3d
        assert m.licence in ok and m.attribution and m.page.startswith('https://') and m.url.startswith('https://')
        assert 0 < m.size_mb <= VL.MAX_DOWNLOAD_MB and m.units in ('m', 'cm', 'mm')
        assert m.attribution in VL.credit(v) and m.licence in VL.credit(v)


# ---------------------------------------------------------------- the download (a fake server)
class _Resp(io.BytesIO):
    def __init__(self, data):
        super().__init__(data)
        self.headers = {'Content-Length': str(len(data))}


def _zip(files):
    b = io.BytesIO()
    with zipfile.ZipFile(b, 'w') as z:
        for n, d in files.items():
            z.writestr(n, d)
    return b.getvalue()


def _tgz(files):
    b = io.BytesIO()
    with tarfile.open(fileobj=b, mode='w:gz') as t:
        for n, d in files.items():
            ti = tarfile.TarInfo(n)
            ti.size = len(d)
            t.addfile(ti, io.BytesIO(d))
    return b.getvalue()


OBJ = b'v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n'


def _fake(v, url_data, member):
    m = dataclasses.replace(v.model3d, url='https://example.invalid/model.bin', member=member)
    return dataclasses.replace(v, model3d=m), (lambda url: _Resp(url_data))


def test_download_unpacks_the_mesh_once_and_records_the_licence(tmp_path):
    v = VL.by_id()['toyota_prius_xw50']
    calls = []
    fv, op = _fake(v, _zip({'meshes/Hybrid.obj': OBJ, 'meshes/Hybrid.mtl': b'newmtl a\n',
                            'materials/textures/t.png': b'png'}), 'meshes/Hybrid.obj')

    def opener(url):
        calls.append(url)
        return op(url)

    got = []
    path = VL.download_model(fv, folder=str(tmp_path), opener=opener, progress=got.append)
    assert path.endswith(os.path.join('mesh', 'meshes', 'Hybrid.obj')) and os.path.exists(path)
    assert os.path.exists(os.path.join(os.path.dirname(path), 'Hybrid.mtl')) and got and got[-1] == 1.0
    lic = (tmp_path / v.id / 'LICENCE.txt').read_text()
    assert v.model3d.attribution in lic and 'CC0-1.0' in lic and 'SHA-256' in lic
    assert len(loaders.load_mesh(path).faces) == 1
    assert VL.download_model(fv, folder=str(tmp_path), opener=opener) == path and len(calls) == 1   # cached
    # a tar.gz with the mesh at its root
    w = VL.by_id()['lincoln_mkz_2017']
    fw, op2 = _fake(w, _tgz({'body.obj': OBJ}), 'body.obj')
    assert VL.download_model(fw, folder=str(tmp_path), opener=op2).endswith('body.obj')


def test_download_refuses_what_it_should(tmp_path):
    v = VL.by_id()['toyota_prius_xw50']
    # a member reaching out of the folder is never written; the mesh asked for is not there
    fv, op = _fake(v, _zip({'../evil.obj': OBJ, 'other/readme.txt': b'x'}), 'meshes/Hybrid.obj')
    with pytest.raises(VL.DownloadError):
        VL.download_model(fv, folder=str(tmp_path / 'a'), opener=op)
    assert not (tmp_path / 'evil.obj').exists() and not (tmp_path / 'a' / 'evil.obj').exists()
    # too large
    fv2, _ = _fake(v, b'', 'x.obj')
    big = lambda url: _Resp(b'0' * int(VL.MAX_DOWNLOAD_MB * 1e6 + 70000))          # noqa: E731
    with pytest.raises(VL.DownloadError, match='larger than'):
        VL.download_model(fv2, folder=str(tmp_path / 'b'), opener=big)
    assert not os.path.exists(tmp_path / 'b' / v.id / '.mesh')
    # the network failing: a short message, nothing left behind

    def down(url):
        raise OSError('no route to host')

    with pytest.raises(VL.DownloadError, match='could not download'):
        VL.download_model(fv2, folder=str(tmp_path / 'c'), opener=down)
    # a vehicle without a model
    with pytest.raises(VL.DownloadError):
        VL.download_model(VL.by_id()['tata_nexon_ev_2023'], folder=str(tmp_path / 'd'), opener=down)


# ---------------------------------------------------------------- COLLADA
def _dae(path, V, F, meter=1.0, up='Z_UP', matrix=None, poly=False):
    pos = ' '.join(f'{x:.6f}' for x in np.asarray(V, float).ravel())
    nrm = ' '.join('0 0 1' for _ in V)
    mat = ' '.join(f'{x:.6f}' for x in (np.eye(4) if matrix is None else matrix).ravel())
    if poly:                                         # one quad per two triangles' worth, as polylist with normals
        idx = ' '.join(f'{a} {a}' for f in F for a in f)
        prim = (f'<polylist count="{len(F)}" material="paint"><input semantic="VERTEX" source="#g-vertices" offset="0"/>'
                f'<input semantic="NORMAL" source="#g-normals" offset="1"/><vcount>{" ".join(str(len(f)) for f in F)}'
                f'</vcount><p>{idx}</p></polylist>')
    else:
        prim = (f'<triangles count="{len(F)}" material="paint"><input semantic="VERTEX" source="#g-vertices" '
                f'offset="0"/><p>{" ".join(str(a) for f in F for a in f)}</p></triangles>')
    xml = f'''<?xml version="1.0" encoding="utf-8"?>
<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema" version="1.4.1">
 <asset><unit name="u" meter="{meter}"/><up_axis>{up}</up_axis></asset>
 <library_effects><effect id="paint-fx"><profile_COMMON><technique sid="c"><phong><diffuse><color>0.8 0.1 0.1 1</color>
 </diffuse></phong></technique></profile_COMMON></effect></library_effects>
 <library_materials><material id="paint-mat" name="Paint"><instance_effect url="#paint-fx"/></material>
 </library_materials>
 <library_geometries><geometry id="g" name="Body"><mesh>
  <source id="g-positions"><float_array id="g-pa" count="{3 * len(V)}">{pos}</float_array>
   <technique_common><accessor source="#g-pa" count="{len(V)}" stride="3"><param name="X" type="float"/>
   <param name="Y" type="float"/><param name="Z" type="float"/></accessor></technique_common></source>
  <source id="g-normals"><float_array id="g-na" count="{3 * len(V)}">{nrm}</float_array>
   <technique_common><accessor source="#g-na" count="{len(V)}" stride="3"/></technique_common></source>
  <vertices id="g-vertices"><input semantic="POSITION" source="#g-positions"/></vertices>
  {prim}
 </mesh></geometry></library_geometries>
 <library_visual_scenes><visual_scene id="s"><node id="n"><matrix>{mat}</matrix>
  <instance_geometry url="#g"><bind_material><technique_common>
   <instance_material symbol="paint" target="#paint-mat"/></technique_common></bind_material></instance_geometry>
 </node></visual_scene></library_visual_scenes>
</COLLADA>'''
    with open(path, 'w') as f:
        f.write(xml)


def test_collada_triangles_polylists_units_and_node_transforms(tmp_path):
    V = np.array([[0, 0, 0], [100, 0, 0], [100, 50, 0], [0, 50, 0]], float)
    M = np.eye(4)
    M[:3, 3] = (10, 0, 5)
    _dae(tmp_path / 'a.dae', V, [(0, 1, 2), (0, 2, 3)], meter=0.01, matrix=M)
    m = loaders.load_mesh(str(tmp_path / 'a.dae'))
    assert m.fmt == 'dae' and m.unit_hint == 'm' and m.up_hint == '+Z' and len(m.faces) == 2
    np.testing.assert_allclose(m.vertices.min(0), [0.1, 0.0, 0.05])
    np.testing.assert_allclose(m.vertices.max(0), [1.1, 0.5, 0.05])
    assert m.parts[0].name == 'Paint' and m.parts[0].color == pytest.approx((0.8, 0.1, 0.1))
    # a polylist quad is two triangles; Y up is read
    _dae(tmp_path / 'b.dae', V, [(0, 1, 2, 3)], up='Y_UP', poly=True)
    b = loaders.load_mesh(str(tmp_path / 'b.dae'))
    assert len(b.faces) == 2 and b.up_hint == '+Y'
    # what is not COLLADA, or declares entities, is refused with a message
    (tmp_path / 'c.dae').write_text('<?xml version="1.0"?><!DOCTYPE x [<!ENTITY a "b">]><COLLADA/>')
    with pytest.raises(loaders.MeshLoadError, match='entities'):
        loaders.load_mesh(str(tmp_path / 'c.dae'))
    (tmp_path / 'd.dae').write_text('<?xml version="1.0"?><other/>')
    with pytest.raises(loaders.MeshLoadError):
        loaders.load_mesh(str(tmp_path / 'd.dae'))


# ---------------------------------------------------------------- the import wizard and the window
from test_gui import app, win, pump, settle  # noqa: E402,F401  (fixtures)


def test_a_downloaded_model_imports_with_the_vehicles_stated_figures(app, win, tmp_path, monkeypatch):  # noqa: F811
    from vantage.ui.dialogs.import_wizard import ImportWizard
    st = win.store
    m = st.project.vehicle.mesh
    lv = VL.by_id()['toyota_prius_xw50']
    # the sample's body written as the Prius model is published: centimetres, Z up
    obj = tmp_path / 'Hybrid.obj'
    with open(obj, 'w') as fh:
        for x, y, z in m.vertices:
            fh.write(f'v {x * 100:.3f} {y * 100:.3f} {z * 100:.3f}\n')
        for f in m.faces:
            fh.write(f'f {f[0] + 1} {f[1] + 1} {f[2] + 1}\n')
    # the window's action: download (faked), then the wizard
    seen = []
    monkeypatch.setattr(VL, 'download_model', lambda v, **k: str(obj))
    win.download_vehicle_model('toyota_prius_xw50', ask=False, done=seen.append)
    t = time.time()
    while not seen and time.time() - t < 10:
        pump(app, 0.05)
    assert seen == [str(obj)]
    wiz = ImportWizard(win.ctx, win, path=str(obj), library=lv)
    wiz.show()
    t = time.time()
    while wiz.lm is None and time.time() - t < 30:
        pump(app, 0.05)
    assert wiz.units == 'cm' and wiz.up == '+Z' and wiz.spec == lv.stated()
    assert 'Open Robotics' in wiz.f_note.text()
    wiz.go(4)
    wiz.finish()
    settle(app, win)
    veh = st.project.vehicle
    assert veh.name == lv.title and veh.template == lv.id and 'downloaded 3D model' in veh.spec_name
    assert veh.spec == lv.stated() and abs(veh.measured()['Length'] - 3.993) < 0.01
    assert any(lv.model3d.attribution in e.text for e in st.project.log)
    st.undo.undo()
    settle(app, win)
    assert st.project.vehicle.mesh is m


# ---------------------------------------------------------------- the shaped bodies and the range rings
@pytest.mark.parametrize('vid', ['tesla_model3_2024', 'renault_zoe_ph1', 'mercedes_sprinter_170_hr'])
def test_a_library_body_has_lamps_trim_rims_and_a_windscreen(vid):
    from vantage.ui import viewport3d as V3
    m = samples.template_project(vid).vehicle.mesh
    names = [p.name for p in m.parts]
    assert {'Head lamps', 'Tail lamps', 'Wheel rims', 'Black trim', 'Glass'} <= set(names)
    mats = V3.mesh_materials(m)
    want = {'Wheel rims': V3.MAT_METAL, 'Black trim': V3.MAT_TRIM, 'Head lamps': V3.MAT_LENS,
            'Tail lamps': V3.MAT_LENS, 'Glass': V3.MAT_GLASS, 'Wheels': V3.MAT_TYRE}
    for name, mat in want.items():
        assert (mats[m.face_part == names.index(name)] == mat).all(), name
    # the windscreen is glass (faces turned forward), not only the side windows
    g = m.face_part == names.index('Glass')
    assert (m.normals[g][:, 0] > 0.3).sum() > 50 and (np.abs(m.normals[g][:, 1]) > 0.45).sum() > 50
    # the lamps are the first thing seen from in front (behind): not under the paint
    for name, d in (('Head lamps', -1.0), ('Tail lamps', 1.0)):
        sel = m.face_part == names.index(name)
        c = m.centroids[sel]
        t, fi = m.raycast(c - 3.0 * np.array([d, 0, 0]), np.tile([d, 0.0, 0.0], (len(c), 1)))
        assert (m.face_part[fi[fi >= 0]] == names.index(name)).mean() > 0.95, name
    # the measured Tigor keeps its own body
    t = samples.tigor_sample_project().vehicle.mesh
    assert 'Head lamps' not in [p.name for p in t.parts]


def test_range_rings_overlay(app, win):  # noqa: F811
    from vantage.ui.context import OVERLAYS
    assert 'rings' in {k for k, *_ in OVERLAYS} and 'ov_rings' in win.actions_by_id
    win.store.set_project(samples.template_project('tesla_model3_2024'))
    settle(app, win)
    sc = win.viewarea.panes[0].scene
    rings = sc._range_rings()
    assert [d for d, _ in rings] == [1.0, 2.0, 5.0, 10.0]
    P = win.store.project.vehicle.mesh.vertices
    for d, pts in rings:
        assert np.allclose(pts[0], pts[-1]) and pts[:, 0].max() == pytest.approx(P[:, 0].max() + d, abs=1e-6)
        assert pts[:, 1].min() == pytest.approx(P[:, 1].min() - d, abs=1e-6)
    win.actions_by_id['ov_rings'].trigger()
    settle(app, win)
    try:
        assert win.ctx.overlays['rings']
        n = len(sc.lines(win.viewarea.panes[0].vp3d))
        assert len(sc._ring_labels) == 8 and n > 4
    finally:
        win.actions_by_id['ov_rings'].trigger()
        settle(app, win)
