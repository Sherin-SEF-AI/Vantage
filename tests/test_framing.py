"""core.framing: catalogue profiles, the frame synthesised from the rig, cut list and stock sums, watertight STL,
STEP topology (and OpenCASCADE when gmsh is installed), DXF, the BOM PDF, adding the frame to the rig, the page."""
import math
import os
import re

import numpy as np
import pytest

from vantage.core import framing as FR
from vantage.core import meshing as M


def scene():
    """The sample's Data rig with a roof rack and a 0.5 m mast, L1 moved onto the mast head."""
    from vantage.core import samples
    from vantage.core.rig import apply_fit, fit_plan
    p = samples.tigor_sample_project()
    v = next(v for v in p.variants if v.sensors)
    p.active_uid = v.uid
    for k in ('roof_rack', 'sensor_mast'):
        apply_fit(p, fit_plan(p, k))
    head = next(r for r in p.rig if r.name.endswith('head'))
    L1 = next(s for s in v.sensors if s.tag == 'L1')
    m = p.sensor_model_for(L1)
    L1.x, L1.y = head.x, head.y
    L1.z = round(head.z + head.height / 2 + m.optical_z_mm / 1000.0, 4)
    return p


@pytest.fixture(scope='module')
def proj():
    return scene()


@pytest.fixture(scope='module')
def res(proj):
    return FR.synthesise(proj)


# ------------------------------------------------------------------ the catalogue
def test_profiles_mass_agrees_with_area():
    """Two independent catalogue figures: kg/m and the section area × aluminium's density (2700 kg/m³)."""
    for p in FR.PROFILES.values():
        assert abs(p.kg_m / (p.area * FR.RHO_AL) - 1) < 0.03, p.id
        assert p.i_d >= p.i_b > 0 and p.d >= p.b > 0
        assert p.source


def test_8020_1010_from_the_catalogue():
    p = FR.PROFILES['in-1010']
    assert p.area == pytest.approx(0.435 * 0.0254 ** 2)
    assert p.i_d * 1e8 == pytest.approx(1.8397, rel=1e-3)       # 0.0442 in⁴ in cm⁴
    assert p.kg_m == pytest.approx(0.757, rel=2e-3)             # 0.0424 lb/in
    assert p.E == pytest.approx(70.33e9, rel=1e-3)              # 10,200,000 psi


def test_settings_tolerant():
    s = FR.FramingSettings.from_dict({'main_profile': 'nope', 'plate_mm': 'x', 'kerf_mm': 99, 'joint': 'anchors',
                                      'frame_cameras': 'yes', 'replaced': ['a', 3, None], 'junk': 1})
    assert s.main_profile == 'mm-4040' and s.plate_mm == 6.0 and s.kerf_mm == 10.0 and s.joint == 'anchors'
    assert s.frame_cameras is True and s.replaced == ['a']
    assert FR.FramingSettings.from_dict(None).to_dict() == FR.FramingSettings().to_dict()


# ------------------------------------------------------------------ synthesis
def test_frame_of_the_sample(res):
    roles = [m.role for m in res.members]
    assert roles.count('crossbar') == 2 and roles.count('rail') == 2 and roles.count('post') == 1
    assert len(res.feet) == 4
    post = next(m for m in res.members if m.role == 'post')
    assert post.length == pytest.approx(0.596, abs=0.002)
    # L1 sits on the post's top plate with no gap
    l1 = next(m for m in res.mounts if m['tag'] == 'L1')
    assert l1['status'] == 'top plate' and abs(l1['gap_mm']) < 0.5
    # the body-mounted cameras and radars are not framed
    assert all(m['tag'] not in ('F', 'RD1') for m in res.mounts)
    # every foot block reaches from the roof to its crossbar's underside
    for F in res.feet:
        cb = res.member(F.member)
        assert F.centre[2] + F.height == pytest.approx(cb.p0[2] - cb.prof.d / 2, abs=1e-6)
        assert F.height >= FR.MIN_FOOT_M


def test_cut_list_sums(res):
    total = sum(m.length for m in res.members) * 1000
    assert sum(r['total_mm'] for r in res.cut_list) == pytest.approx(total, abs=0.5 * len(res.members))
    assert sum(r['quantity'] for r in res.cut_list) == len(res.members)
    for s in res.stock:
        cuts = sorted(c for b in s['cuts_mm'] for c in b)
        mine = sorted(r['length_mm'] for r in res.cut_list if r['profile'] == s['profile'] for _ in range(r['quantity']))
        assert cuts == mine
        for b, off in zip(s['cuts_mm'], s['offcut_mm']):
            assert sum(b) + res.settings.kerf_mm * len(b) + off == pytest.approx(s['stock_mm'])
            assert off >= -1e-9
    # BOM profile length equals the cut list, masses add up
    prof_m = sum(r['quantity'] for r in res.bom if r['category'] == 'Profile')
    assert prof_m * 1000 == pytest.approx(total, abs=0.5 * len(res.members))
    assert res.mass['frame_kg'] == pytest.approx(sum(r['kg'] for r in res.bom), abs=0.01)


def test_stock_packing_first_fit():
    st = FR.FramingSettings(stock_m=1.0, kerf_mm=10)
    r = FR.FramingResult(st, 'v', 'u', {}, [FR.Member(f'M{i}', 'm', 'rail', 'mm-4040', np.zeros(3),
                                                       np.array([L, 0, 0]), np.array([0, 0, 1.0]))
                                            for i, L in enumerate((0.6, 0.5, 0.4, 0.3, 1.2))],
                         [], [], [], [], [], [], [], [], {})
    s = FR.stock_plan(r)[0]
    assert s['too_long_mm'] == [1200.0]
    assert s['cuts_mm'] == [[600.0, 300.0], [500.0, 400.0]]


def test_empty_rig_gives_empty_frame():
    from vantage.core import samples
    p = samples.tigor_sample_project()
    p.active_uid = next(v for v in p.variants if v.sensors).uid
    r = FR.synthesise(p)
    assert r.members == [] and r.findings[0]['kind'] == 'empty'
    names = [n for n, _ in FR.files(r, p)]
    assert 'framing.json' in names and 'framing.step' not in names


def test_device_sunk_into_frame_is_flagged():
    p = scene()
    v = p.variant
    # the sample's L1 stays on the roof under the rack's rails: put it back there
    L1 = next(s for s in v.sensors if s.tag == 'L1')
    L1.x, L1.y, L1.z = 0.966, 0.40, 1.57
    r = FR.synthesise(p)
    rec = next(m for m in r.mounts if m['tag'] == 'L1')
    assert rec['status'] == 'below frame' and rec['gap_mm'] < 0
    assert any(f['kind'] == 'conflict' and L1.uid in f['uids'] for f in r.findings)


def test_device_above_rails_gets_cross_member_and_standoff():
    p = scene()
    L1 = next(s for s in p.variant.sensors if s.tag == 'L1')
    L1.x, L1.y, L1.z = 1.2, 0.0, 1.85
    r = FR.synthesise(p)
    rec = next(m for m in r.mounts if m['tag'] == 'L1')
    assert rec['status'] == 'plate' and rec['standoff']
    so = r.member(rec['standoff'])
    assert so.role == 'standoff'
    under = next(j['b'] for j in r.joints if j['a'] == so.id)
    assert r.member(under).role == 'cross member'
    P = next(q for q in r.plates if q.id == rec['plate'])
    m = p.sensor_model_for(L1)
    assert P.centre[2] + P.t == pytest.approx(L1.z - m.optical_z_mm / 1000, abs=1e-6)


def test_device_in_front_gets_arm():
    p = scene()
    L1 = next(s for s in p.variant.sensors if s.tag == 'L1')
    L1.x, L1.y, L1.z = 1.6, 0.0, 1.80
    r = FR.synthesise(p)
    assert any(m.role == 'arm' for m in r.members)
    rec = next(m for m in r.mounts if m['tag'] == 'L1')
    assert rec['status'] == 'plate'


def test_anchors_machining(proj):
    r = FR.synthesise(proj, FR.FramingSettings(joint='anchors'))
    cm = next(m for m in r.members if m.role == 'cross member')
    assert 'anchor counterbore' in cm.end0 and 'anchor counterbore' in cm.end1
    assert sum(m.access_holes for m in r.members) >= 3
    fx = FR.fixings(r)
    stacked = sum(j['kind'] == 'stacked' for j in r.joints)
    assert sum(fx['anchors'].values()) == sum(j['kind'] in ('tee', 'post') for j in r.joints)
    assert fx['brackets'].get('mm-4040', 0) == 2 * stacked            # stacked crossings keep their brackets


# ------------------------------------------------------------------ geometry and files
def test_outline_simple_and_area():
    P = FR.tslot_outline(0.04, 0.08, 0.008, 0.04)
    assert FR.poly_area(P) > 0
    # no self-intersection: every pair of non-adjacent edges is disjoint
    n = len(P)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    for i in range(n):
        a, b = P[i], P[(i + 1) % n]
        for j in range(i + 2, n):
            if (j + 1) % n == i:
                continue
            c, d = P[j], P[(j + 1) % n]
            if cross(a, b, c) * cross(a, b, d) < -1e-18 and cross(c, d, a) * cross(c, d, b) < -1e-18:
                pytest.fail(f'edges {i} and {j} cross')
    tri = FR._ear_clip(P)
    assert len(tri) == n - 2
    assert sum(abs(FR.poly_area(P[list(t)])) for t in tri) == pytest.approx(FR.poly_area(P), rel=1e-9)


def test_stl_watertight(res):
    for c in FR.check_meshes(res):
        assert c['ok'] and c['shells'] == 1, c['id']
        assert c['volume_mm3'] == pytest.approx(c['volume_expected_mm3'], rel=1e-6)
    name, data = FR.stl_files(res)[0]
    V, F, N, head = M.read_stl(data)
    assert len(F) == (len(data) - 84) // 50 and 'frame' in head


def _step_entities(text):
    body = text.split('DATA;')[1].split('ENDSEC;')[0]
    ents = {}
    for line in body.strip().splitlines():
        m = re.match(r'#(\d+)=(.*);$', line)
        assert m, line
        ents[int(m.group(1))] = m.group(2)
    return ents


def test_step_well_formed(res, proj):
    t = FR.step_text(res, proj)
    assert t.startswith('ISO-10303-21;') and t.rstrip().endswith('END-ISO-10303-21;')
    assert "FILE_SCHEMA(('AUTOMOTIVE_DESIGN" in t
    ents = _step_entities(t)
    assert sorted(ents) == list(range(1, len(ents) + 1))
    for k, e in ents.items():
        for ref in re.findall(r'#(\d+)', e):
            assert int(ref) in ents and int(ref) != k
    sols = FR.solids(res)
    assert sum(e.startswith('MANIFOLD_SOLID_BREP') for e in ents.values()) == len(sols)
    assert sum(e.startswith('NEXT_ASSEMBLY_USAGE_OCCURRENCE') for e in ents.values()) == len(sols)
    # Euler–Poincaré for the whole file: Σ (V − E + 2F − L) = Σ 2 (1 − holes)
    V = sum(e.startswith('VERTEX_POINT') for e in ents.values())
    E = sum(e.startswith('EDGE_CURVE') for e in ents.values())
    Fc = sum(e.startswith('ADVANCED_FACE') for e in ents.values())
    L = sum(e.startswith('EDGE_LOOP') for e in ents.values())
    holes = sum(len(s.loops) - 1 for s in sols)
    assert V - E + 2 * Fc - L == 2 * len(sols) - 2 * holes
    # every edge is used exactly twice, once each way
    uses = {}
    for e in ents.values():
        m = re.match(r"ORIENTED_EDGE\('',\*,\*,#(\d+),\.(T|F)\.\)", e)
        if m:
            uses.setdefault(int(m.group(1)), []).append(m.group(2))
    assert len(uses) == E and all(sorted(u) == ['F', 'T'] for u in uses.values())


def test_real_step_reads_in_opencascade(res, proj):
    pytest.importorskip('gmsh')
    out = FR.verify_step(FR.step_text(res, proj), res)
    assert out['available'] and out['volumes'] == len(FR.solids(res))
    assert out['max_rel_error'] < 1e-6


def test_dxf_well_formed(res):
    for P in res.plates:
        pairs = FR.read_dxf(FR.dxf_plate(P))
        vals = [v for c, v in pairs if c == 0]
        assert vals[0] == 'SECTION' and vals[-1] == 'EOF' and 'ENTITIES' in [v for c, v in pairs if c == 2]
        assert vals.count('CIRCLE') == len(P.holes) and vals.count('VERTEX') == 4
        r = [float(v) for c, v in pairs if c == 40][:len(P.holes)]
        assert sorted(r) == pytest.approx(sorted(h[2] * 500 for h in P.holes), abs=1e-3)
    sheet = FR.read_dxf(FR.dxf_sheet(res))
    assert [v for c, v in sheet if c == 0].count('CIRCLE') == sum(len(P.holes) for P in res.plates)


def test_real_dxf_reads_in_ezdxf(res, tmp_path):
    ezdxf = pytest.importorskip('ezdxf')
    path = tmp_path / 'plates.dxf'
    path.write_text(FR.dxf_sheet(res))
    doc = ezdxf.readfile(str(path))
    aud = doc.audit()
    assert not aud.has_errors
    assert len(doc.modelspace().query('CIRCLE')) == sum(len(P.holes) for P in res.plates)


def test_bom_pdf_and_files(res, proj, tmp_path):
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])        # noqa: F841
    pdf = FR.bom_pdf(res, proj)
    assert pdf[:5] == b'%PDF-'
    paths = FR.export(res, proj, str(tmp_path))
    names = {os.path.basename(p) for p in paths}
    assert {'framing.json', 'framing_cutlist.csv', 'framing_bom.csv', 'framing_bom.pdf', 'framing.step',
            'plates.dxf', 'framing.png'} <= names
    assert sum(n.endswith('.stl') for n in names) == len(FR.solids(res))
    import json
    doc = json.load(open(tmp_path / 'framing.json'))
    assert doc['schema'] == FR.SCHEMA and doc['profiles']['mm-4040']['source']
    rows = open(tmp_path / 'framing_cutlist.csv').read().strip().splitlines()
    assert len(rows) == 1 + len(res.cut_list)


def test_apply_to_rig_round_trip():
    p = scene()
    r0 = FR.synthesise(p)
    n0 = len(p.rig)
    added = FR.apply_to_rig(p, r0)
    assert len(added) == len(FR.solids(r0))
    hidden = [q for q in p.rig if not q.visible]
    assert {q.uid for q in hidden} == set(FR.replaced_uids(r0, p))
    r1 = FR.synthesise(p)                                     # still from the parts it stands for
    assert [m.role for m in r1.members] == [m.role for m in r0.members]
    FR.remove_from_rig(p)
    assert len(p.rig) == n0 and all(q.visible for q in p.rig)


def test_validation_checks():
    for c in FR.validation_checks():
        out = c['fn']()
        assert out['error'] <= c['tolerance'], (c['id'], out)


# ------------------------------------------------------------------ the page
from test_gui import app, win, pump, settle, _wait   # noqa: E402,F401  (fixtures)


def _open(win, name, cls):
    if name not in win._page_widgets:
        pg = cls(win.ctx)
        win.pages.addWidget(pg)
        win._page_widgets[name] = pg
    win.set_page(name)
    return win._page_widgets[name]


def test_page(app, win, tmp_path):
    from vantage.core.rig import apply_fit, fit_plan
    from vantage.ui.workspaces.framing_ws import FramingPage
    s = win.ctx.store

    def fit(p):
        p.active_uid = next(v for v in p.variants if v.sensors).uid
        for k in ('roof_rack', 'sensor_mast'):
            apply_fit(p, fit_plan(p, k))
    s.edit(['rig', 'variant'], fit, 'fit a rack and a mast')
    pg = _open(win, 'framing_test', FramingPage)
    pg.activate()
    pump(app, 0.3)
    assert pg.result is not None and len(pg.result.members) >= 5
    assert pg.t_cut.rowCount() == len(pg.result.cut_list) and pg.t_bom.rowCount() == len(pg.result.bom)
    # a profile change is one undo step
    c = pg.combos['main_profile']
    i = c.findData('in-1010')
    c.setCurrentIndex(i)
    c.activated.emit(i)
    pump(app, 0.2)
    assert s.project.analysis.framing['main_profile'] == 'in-1010'
    assert pg.result.members[0].profile == 'in-1010'
    s.undo.undo()
    pump(app, 0.2)
    assert FR.settings_of(s.project).main_profile == 'mm-4040'
    # a typed field
    pg.fields['plate_mm'].committed.emit(8.0)
    pump(app, 0.2)
    assert FR.settings_of(s.project).plate_mm == 8.0
    # the view renders
    img = pg.view.render_image(320, 240)
    assert img is not None and not img.isNull()
    # add to the rig, undo
    n = len(s.project.rig)
    assert pg.apply_to_rig()
    pump(app, 0.2)
    assert any(q.name.startswith(FR.FRAME_PREFIX) for q in s.project.rig)
    s.undo.undo()
    pump(app, 0.2)
    assert len(s.project.rig) == n and not any(q.name.startswith(FR.FRAME_PREFIX) for q in s.project.rig)
    paths = pg.export(str(tmp_path))
    assert any(p.endswith('framing.step') for p in paths)
