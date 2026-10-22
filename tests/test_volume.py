"""3D blind volume: core analysis, surface, export and batch mode."""
import copy
import json
import math

import numpy as np
import pytest

from vantage.core import samples, volume as vol
from vantage.core.analysis import Engine


@pytest.fixture(scope='module')
def sample():
    p = samples.tigor_sample_project()
    e = Engine()
    res = e.analyse(p)
    return p, e, res


@pytest.fixture(scope='module')
def vr(sample):
    p, e, res = sample
    return vol.analyse_volume(p, e, res)


def _cell(vr, x, y):
    return int((y - vr.y0) // vr.cell), int((x - vr.x0) // vr.cell)


def test_a_camera_sees_the_space_in_front_and_not_behind_the_body(sample):
    p0, e, res = sample
    p = copy.deepcopy(p0)
    f = p.variant.by_tag('F')
    for c in p.variant.cameras:
        c.visible = c is f
    v = vol.analyse_volume(p, e, res, height=1.0, dz=0.1)
    assert v.metrics['cameras'] == 1
    j, i = _cell(v, f.x + 1.0, f.y)                      # 1 m ahead of the front camera
    k = v.level_index(0.3)
    assert v.roi[j, i] and v.count[k, j, i] == 1
    lo = v.xs[v.body.any(axis=0)].min()
    j, i = _cell(v, lo - 0.3, 0.0)                       # behind the car: the body is in the way
    assert v.roi[j, i] and (v.count[:, j, i] == 0).all() and np.isinf(v.detect_h[j, i])
    # one camera leaves far more blind than six
    full = vol.analyse_volume(p0, e, res, height=1.0, dz=0.1)
    assert v.metrics['blind_m3'] > full.metrics['blind_m3'] + 1.0
    assert set(v.per_cam_m3) == {f.uid} and v.per_cam_m3[f.uid] > 0


def test_ground_level_matches_the_ground_analysis(sample):
    p, e, res = sample
    v = vol.analyse_volume(p, e, res, height=0.02, dz=0.02, cell=res.cell)
    assert (v.nx, v.ny) == (res.grid.nx, res.grid.ny)
    ground_blind = res.grid.roi & (res.count == 0)
    agree = (v.blind[0] == ground_blind)[res.grid.roi].mean()
    assert agree > 0.995
    assert v.metrics['ground_blind_m2'] == pytest.approx(res.metrics['blind_m2'], abs=0.02)


def test_metrics_objects_and_rose_are_consistent(vr):
    m = vr.metrics
    a2, a3 = vr.cell ** 2, vr.cell ** 2 * vr.dz
    assert m['blind_m3'] == pytest.approx(vr.blind.sum() * a3)
    assert m['roi_m3'] == pytest.approx(vr.roi.sum() * a2 * len(vr.z) * vr.dz)
    assert np.allclose(vr.level_blind_m2, vr.blind.sum(axis=(1, 2)) * a2)
    assert 0 <= m['blind_pct'] <= 100 and m['covered_to_m'] <= vr.height
    # an object is detected when the lowest seen point of its column is no higher than the object
    for name, h in vr.objects:
        und = vr.roi & ~vr.object_seen[name]
        assert np.array_equal(und, vr.roi & ~(vr.detect_h <= h + 1e-9))
        assert m['objects'][name]['undetected_m2'] == pytest.approx(und.sum() * a2)
        assert vr.reach_object[name].max() == pytest.approx(vr.dist[und].max() if und.any() else 0.0)
    # taller objects are never harder to see
    seen = [vr.object_seen[n].sum() for n, _ in sorted(vr.objects, key=lambda o: o[1])]
    assert seen == sorted(seen)
    # the rose: farthest blind point per sector
    k = vr.level_index(1.0)
    b = vr.blind[k]
    assert vr.reach_level[k].max() == pytest.approx(vr.dist[b].max())
    X, Y = np.meshgrid(vr.xs, vr.ys)
    az = np.degrees(np.arctan2(Y - vr.centre[1], X - vr.centre[0]))
    s = 18                                                  # the sector around -90° (right)
    lo = vr.sector_deg[s] - 2.5
    sel = b & (az >= lo) & (az < lo + 5.0)
    assert vr.reach_level[k][s] == pytest.approx(vr.dist[sel].max() if sel.any() else 0.0)
    assert len(vr.sector_deg) == vol.SECTORS and vr.sector_deg[0] == pytest.approx(-177.5)


def test_surface_nets_close_a_sphere_with_outward_faces():
    n = 24
    Z, Y, X = np.mgrid[:n, :n, :n]
    ball = (X - 11.5) ** 2 + (Y - 12.0) ** 2 + (Z - 11.0) ** 2 <= 7.5 ** 2
    for kw in ({}, {'smooth': 0}, {'blur': 1.0}):
        v, q = vol.surface_nets(ball, **kw)
        F = np.concatenate([q[:, [0, 1, 2]], q[:, [0, 2, 3]]])
        e = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1)
        _, uses = np.unique(e, axis=0, return_counts=True)
        assert (uses == 2).all(), kw                         # watertight: every edge between two faces
        T = v[F][:, :, ::-1]                                  # (z, y, x) index order -> x, y, z
        signed = np.einsum('ij,ij->i', T[:, 0], np.cross(T[:, 1], T[:, 2])).sum() / 6.0
        # outward, and about the right size (blurring a ball shrinks it by about a third of a sample)
        assert signed == pytest.approx(ball.sum(), rel=0.2 if kw.get('blur') else 0.1), kw


def test_blind_surface_in_world_coordinates(vr):
    P, F = vol.blind_surface(vr)
    assert len(F) > 1000 and F.max() < len(P)
    assert P[:, 0].min() >= vr.x0 - 1e-6 and P[:, 0].max() <= vr.x0 + vr.nx * vr.cell + 1e-6
    assert P[:, 2].min() >= 0 and P[:, 2].max() <= vr.height + 1e-6
    k = vr.level_index(0.9) + 1
    P2, F2 = vol.blind_surface(vr, caps=False, top=k, body_walls=False, blur=1.0)
    assert len(F2) and P2[F2][:, :, 2].max() <= k * vr.dz + 1e-6
    assert (P2[F2][:, :, 2] > 1e-3).any(axis=1).all()        # no faces on the ground
    P3, F3 = vol.blind_surface(vr, pool=4)
    assert 0 < len(F3) < len(F)


def test_cancel_and_progress(sample):
    p, e, res = sample
    assert vol.analyse_volume(p, e, res, cancel=lambda: True) is None
    seen = []
    vol.analyse_volume(p, e, res, height=0.5, dz=0.1, progress=seen.append)
    assert seen and seen == sorted(seen) and seen[-1] == pytest.approx(1.0)


def test_settings_are_clamped():
    st = vol.VolumeSettings.from_dict({'height': -1, 'dz': 'x', 'cell': 10, 'other': 3})
    assert st.height == pytest.approx(0.1) and st.dz == pytest.approx(0.05)
    assert st.cell == pytest.approx(0.5)
    assert vol.VolumeSettings.from_dict({'height': float('nan')}).height == 2.0
    assert vol.VolumeSettings.from_dict(None).to_dict() == {'height': 2.0, 'dz': 0.05, 'cell': 0.0}


def test_export_format_and_doc(sample, tmp_path):
    from vantage.core import exporters as ex
    p, e, res = sample
    paths = ex.export(['blind_volume'], p, res, ex.ExportOptions(out_dir=str(tmp_path), engine=e))
    assert [x.rsplit('/', 1)[1] for x in paths] == ['blind_volume.json', 'blind_volume.obj']
    doc = json.loads(open(paths[0]).read())
    assert doc['schema'] == 'vantage.blind_volume/1' and doc['variant'] == p.variant.name
    assert len(doc['rose']['sector_deg']) == vol.SECTORS and len(doc['levels']) == 40
    assert {o['name'] for o in doc['objects']} == {n for n, _ in vol.DEFAULT_OBJECTS}
    assert len(doc['cameras']) == len([c for c in p.variant.cameras if c.visible])
    obj = open(paths[1]).read().splitlines()
    nv = sum(1 for ln in obj if ln.startswith('v '))
    faces = [ln for ln in obj if ln.startswith('f ')]
    assert nv > 100 and faces and max(int(t) for ln in faces for t in ln.split()[1:]) == nv


def test_batch_blind_volume(tmp_path):
    from vantage import batch
    from test_advanced import _args
    out = tmp_path / 'volume.json'
    assert batch.run(_args(blind_volume=str(out), variant='1')) == 0
    doc = json.loads(out.read_text())
    assert doc['schema'] == 'vantage.blind_volume/1' and doc['blind_m3'] > 0
    assert math.isfinite(doc['covered_to_m'])
