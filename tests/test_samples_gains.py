"""The sample projects are deterministic (the same uids on every call), and the bird's-eye exposure gains report the
cameras whose gains were clipped (core.stitch.estimate_gains and its Apache-2.0 twin in vantage/runtime)."""
import copy
import json
import types

import numpy as np
import pytest

from vantage.core import samples, stitch


# ----------------------------------------------------------------------------- deterministic samples
def _uids(p):
    out = {'active': p.active_uid}
    for i, v in enumerate(p.variants):
        out[f'v{i}'] = v.uid
        for c in v.cameras:
            out[f'v{i}/cam/{c.tag}'] = (c.uid, c.link_uid)
        for s in v.sensors:
            out[f'v{i}/sensor/{s.tag}'] = s.uid
    for kind, objs in (('target', p.targets), ('obstacle', p.obstacles), ('rig', p.rig)):
        for o in objs:
            out[f'{kind}/{o.name}'] = o.uid
    return out


def _all_uids(p):
    us = [v.uid for v in p.variants] + [o.uid for v in p.variants for o in list(v.cameras) + list(v.sensors)]
    return us + [o.uid for o in list(p.targets) + list(p.obstacles) + list(p.rig)]


@pytest.fixture(scope='module')
def tigor_twice():
    return samples.tigor_sample_project(), samples.tigor_sample_project()


def test_tigor_sample_has_the_same_uids_on_every_call(tigor_twice):
    a, b = tigor_twice
    ua, ub = _uids(a), _uids(b)
    assert ua == ub
    assert len(a.variants) == 5 and any(v.sensors for v in a.variants) and a.targets and a.obstacles
    assert a.active_uid == a.variants[0].uid
    # objects of one call are found by uid in the other
    for va in a.variants:
        vb = b.variant_by_uid(va.uid)
        assert vb is not None and vb.name == va.name
        for c in va.cameras:
            assert vb.cam(c.uid).tag == c.tag
        for s in va.sensors:
            assert vb.sensor(s.uid).tag == s.tag


def test_sample_uids_are_unique_and_links_hold(tigor_twice):
    a, _ = tigor_twice
    us = _all_uids(a)
    assert len(us) == len(set(us)) and all(len(u) == 10 for u in us)
    for v in a.variants:
        for c in v.cameras:
            if c.link_uid:
                partner = v.cam(c.link_uid)
                assert partner is not None and partner.link_uid == c.uid
    assert sum(bool(c.link_uid) for c in a.variants[0].cameras) == 4          # ML/MR and QL/QR


def test_sample_project_hash_is_stable(tigor_twice):
    from vantage.assistant.evals import project_hash
    a, b = tigor_twice
    # the creation times stay the real ones (a project made from the sample was created when it was made): with
    # those equal, the two projects hash the same
    b.created = a.created
    for va, vb in zip(a.variants, b.variants):
        vb.created = va.created
    assert project_hash(a) == project_hash(b)
    da, db = a.to_json(), b.to_json()
    for d in (da, db):
        d.pop('saved_at')
        d.pop('log')                     # its entries carry the time they were written
    assert json.dumps(da, sort_keys=True, default=str) == json.dumps(db, sort_keys=True, default=str)


def test_template_and_blank_projects_are_deterministic():
    for make in (lambda: samples.template_project('sedan'), samples.blank_project):
        a, b = make(), make()
        assert _uids(a) == _uids(b)
        us = _all_uids(a)
        assert len(us) == len(set(us))
    # different projects get different uids
    assert samples.template_project('sedan').variants[0].uid != samples.blank_project().variants[0].uid


def test_layout_builders_still_make_fresh_uids(tigor_twice):
    """apply_preset adds tigor_variants' cameras to a project that already holds the sample's: those must be new."""
    a, _ = tigor_twice
    v1 = samples.tigor_variants(a.vehicle.mesh)
    v2 = samples.tigor_variants(a.vehicle.mesh)
    assert {c.uid for c in v1[0].cameras}.isdisjoint(c.uid for c in v2[0].cameras)
    assert {c.uid for c in v1[0].cameras}.isdisjoint(c.uid for c in a.variants[0].cameras)


def test_stable_uids_keeps_seams_links_and_the_active_variant():
    from vantage.core.project import Project, Variant, Camera, Target
    p = Project()
    p.name = 'x'
    v1, v2 = Variant(name='a'), Variant(name='b')
    ca, cb = Camera(tag='ML'), Camera(tag='MR')
    v2.cameras = [ca, cb]
    samples.link_pairs(v2)
    v2.seams = {f'{ca.uid}|{cb.uid}': 30.0}
    p.variants, p.active_uid, p.targets = [v1, v2], v2.uid, [Target(name='T1')]
    samples.stable_uids(p)
    assert p.active_uid == v2.uid and ca.link_uid == cb.uid and cb.link_uid == ca.uid
    assert v2.seams == {f'{ca.uid}|{cb.uid}': 30.0}
    q = copy.deepcopy(p)
    samples.stable_uids(q)
    assert _uids(q) == _uids(p)


# ----------------------------------------------------------------------------- clipped exposure gains
def _blocks(levels, blk=16):
    """warped/valid grids where each camera pair (i, j) overlaps on its own uniform block of grey levels."""
    n = 1 + max(max(i, j) for i, j, _, _ in levels)
    uids = [f'c{i}' for i in range(n)]
    shape = (blk, blk * len(levels))
    warped = {u: np.zeros(shape + (3,), np.float32) for u in uids}
    valid = {u: np.zeros(shape, bool) for u in uids}
    for k, (i, j, Ia, Ib) in enumerate(levels):
        sl = (slice(None), slice(k * blk, (k + 1) * blk))
        warped[uids[i]][sl], warped[uids[j]][sl] = Ia, Ib
        valid[uids[i]][sl] = valid[uids[j]][sl] = True
    lut = types.SimpleNamespace(uids=uids, body=np.zeros(shape, bool))
    return lut, warped, valid


def test_unclipped_gains_report_nothing():
    lut, warped, valid = _blocks([(0, 1, 120.0, 100.0), (1, 2, 110.0, 130.0), (2, 0, 125.0, 118.0)])
    g = stitch.estimate_gains(lut, warped=(warped, valid))
    assert isinstance(g, dict) and set(g) == set(lut.uids) and g.clipped == {}
    assert all(stitch.GAIN_MIN < float(v.min()) and float(v.max()) < stitch.GAIN_MAX for v in g.values())
    assert stitch.clipped_note(g) == '' and stitch.clipped_note({'c0': np.ones(3)}) == ''


def test_clipped_gains_name_the_camera_and_the_solved_gain():
    lut, warped, valid = _blocks([(0, 1, 240.0, 10.0), (1, 2, 100.0, 104.0)])
    g = stitch.estimate_gains(lut, warped=(warped, valid))
    assert g.clipped, g
    for u, raw in g.clipped.items():
        assert raw.shape == (3,) and raw.dtype == np.float32
        assert np.any((raw < stitch.GAIN_MIN) | (raw > stitch.GAIN_MAX))
        assert np.allclose(g[u], np.clip(raw, stitch.GAIN_MIN, stitch.GAIN_MAX))
    for u in set(g) - set(g.clipped):
        assert np.all((g[u] >= stitch.GAIN_MIN) & (g[u] <= stitch.GAIN_MAX))
    tags = {'c0': 'F', 'c1': 'MR', 'c2': 'R'}
    note = stitch.clipped_note(g, tags)
    for u, raw in g.clipped.items():
        k = int(np.argmax(np.abs(np.log(raw / np.clip(raw, stitch.GAIN_MIN, stitch.GAIN_MAX)))))
        ap = min(max(float(raw[k]), stitch.GAIN_MIN), stitch.GAIN_MAX)
        assert f'{tags[u]} {raw[k]:.2f} → {ap:.2f}' in note
    # backward compatible: a dict of (3,) gains that copies, applies and serialises as before
    h = copy.deepcopy(g)
    assert h.clipped.keys() == g.clipped.keys() and dict(h).keys() == dict(g).keys()
    assert (g or {}) is g
    json.dumps({u: v.tolist() for u, v in g.items()})


def _runtime():
    import importlib.util
    import os
    path = os.path.join(os.path.dirname(stitch.__file__), '..', 'runtime', 'bev_stitch.py')
    spec = importlib.util.spec_from_file_location('bev_stitch_rt', path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _runtime_stitcher(rt, lut, warped, valid):
    st = object.__new__(rt.Stitcher)
    st.tags, st.valid, st.body, st.clipped = list(lut.uids), valid, lut.body, {}
    return st


@pytest.mark.parametrize('levels', [[(0, 1, 120.0, 100.0), (1, 2, 110.0, 130.0)],
                                    [(0, 1, 240.0, 10.0), (1, 2, 100.0, 104.0)]])
def test_runtime_stitcher_reports_the_same_clipping(levels):
    rt = _runtime()
    assert (rt.GAIN_MIN, rt.GAIN_MAX) == (stitch.GAIN_MIN, stitch.GAIN_MAX)
    lut, warped, valid = _blocks(levels)
    ref = stitch.estimate_gains(lut, warped=(warped, valid))
    st = _runtime_stitcher(rt, lut, warped, valid)
    g = st.estimate_gains(warped)
    assert type(g) is dict and all(np.allclose(g[u], ref[u], rtol=1e-5) for u in ref)
    assert st.clipped.keys() == ref.clipped.keys()
    assert all(np.allclose(st.clipped[u], ref.clipped[u], rtol=1e-5) for u in ref.clipped)
    assert st.clipped_note() == stitch.clipped_note(ref).replace('→', '->')
    assert bool(st.clipped_note()) == (levels[0][2] == 240.0)


def test_runtime_cli_warns_about_clipped_gains(tmp_path, capsys):
    import cv2
    rt = _runtime()
    w, h, sw, sh = 32, 16, 16, 16
    cams = [{'tag': t, 'sensor_width': sw, 'sensor_height': sh, 'map': f'{t}_map.bin'} for t in ('A', 'B')]
    (tmp_path / 'bev_lut.json').write_text(json.dumps({'width': w, 'height': h, 'cameras': cams}), 'utf-8')
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    m = np.dstack([xx * (sw - 1) / (w - 1), yy]).astype(np.float32)
    np.savez(tmp_path / 'bev_lut.npz', A_map=m, B_map=m, A_weight=np.full((h, w), 0.5, np.float32),
             B_weight=np.full((h, w), 0.5, np.float32), body_mask=np.zeros((h, w), np.uint8))
    frames = tmp_path / 'frames'
    frames.mkdir()
    for t, level in (('A', 120), ('B', 100)):
        cv2.imwrite(str(frames / f'{t}.png'), np.full((sh, sw, 3), level, np.uint8))
    assert rt.main(['--lut', str(tmp_path), '--frames', str(frames), '--balance']) == 0
    out = capsys.readouterr()
    assert out.out.count('gain ') == 2 and 'clipped' not in out.err
    cv2.imwrite(str(frames / 'A.png'), np.full((sh, sw, 3), 240, np.uint8))
    cv2.imwrite(str(frames / 'B.png'), np.full((sh, sw, 3), 10, np.uint8))
    assert rt.main(['--lut', str(tmp_path), '--frames', str(frames), '--balance']) == 0
    err = capsys.readouterr().err
    assert 'warning: gains clipped to 0.4-2.5:' in err and ('A ' in err or 'B ' in err)
