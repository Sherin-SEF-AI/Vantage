"""Engine tests: lens models, coverage numbers on the sample, studies, rules and persistence."""
import math
import os

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np
import pytest

from vantage.core import samples
from vantage.core.analysis import Engine
from vantage.core.lens import builtin_lenses, builtin_modules, model_fov, resolve, r_of_theta, theta_of_r, PROJECTIONS
from vantage.core.mathutil import rot_zyx, euler_zyx, parse_number, quat_from_matrix
from vantage.core.project import Project


@pytest.fixture(scope='module')
def qapp():
    from PyQt6.QtGui import QGuiApplication
    return QGuiApplication.instance() or QGuiApplication([])


@pytest.fixture(scope='module')
def sample(qapp):
    p = samples.tigor_sample_project()
    e = Engine()
    return p, e, e.analyse(p)


def test_rotation_conventions():
    R = rot_zyx(0, 30, 0)
    assert R[:, 0][2] < 0, 'positive pitch must tilt the optical axis down'
    R = rot_zyx(90, 0, 0)
    assert np.allclose(R[:, 0], [0, 1, 0]), 'positive yaw turns left'
    for y, p, r in ((65, 45, 0), (-120, 30, 5), (180, 40, -3)):
        yy, pp, rr = euler_zyx(rot_zyx(y, p, r))
        assert np.allclose(rot_zyx(yy, pp, rr), rot_zyx(y, p, r), atol=1e-9)
    q = quat_from_matrix(rot_zyx(30, 10, 5))
    assert abs(np.linalg.norm(q) - 1) < 1e-9


def test_parse_number():
    assert parse_number('1.88+0.05', 'length') == pytest.approx(1.93)
    assert parse_number('950 mm', 'length') == pytest.approx(0.95)
    assert parse_number('950', 'length', 'mm') == pytest.approx(950)
    assert parse_number('1.2 m', 'length', 'mm') == pytest.approx(1200)
    assert parse_number('65deg', 'angle') == pytest.approx(65)
    assert parse_number('−0.8', 'length') == pytest.approx(-0.8)
    with pytest.raises(ValueError):
        parse_number('import os', 'length')


def test_lens_fit_matches_datasheet():
    m = builtin_modules()[0]
    wide = next(l for l in builtin_lenses() if l.id == 'h05r2')
    h, v, d = model_fov(m, wide)
    assert abs(h - 128.7) < 1.0 and abs(v - 98.7) < 1.0 and abs(d - 170.1) < 1.0
    for proj in PROJECTIONS:
        th = np.radians([5, 20, 40, 60])
        r = r_of_theta(th, 2.62, proj, wide.kb)
        assert np.allclose(theta_of_r(r, 2.62, proj, wide.kb), th, atol=1e-4), proj


def test_intrinsics_roundtrip():
    m = builtin_modules()[0]
    intr = resolve(m, builtin_lenses()[0])
    u = np.array([100.0, 960.0, 1800.0])
    v = np.array([200.0, 768.0, 1400.0])
    d, th = intr.unproject(u, v)
    uu, vv, tt, ok = intr.project_dirs(d)
    assert np.allclose(uu, u, atol=1e-3) and np.allclose(vv, v, atol=1e-3) and ok.all()


def test_sample_coverage(sample):
    p, e, r = sample
    m = r.metrics
    assert m['blind_m2'] < 0.02
    assert m['covered_pct'] > 99.9
    assert m['min_density'] > 3.0
    assert 30 < m['roi_area'] < 40
    assert len(r.seams) == 6
    assert sum(pc['unique_m2'] for pc in r.per_cam.values()) > 10


def test_variant_ranking(sample):
    p, e, r6 = sample
    by = {v.name: e.analyse(p, v).metrics['blind_m2'] for v in p.variants}
    assert by['6-cam · v3'] < by['4-cam'] < by['4-cam, mirror cameras 5° off']
    assert by['6-cam front-heavy'] > 1.0


def test_tolerance_and_worst_single(sample):
    from vantage.core.tolerance import TolSettings, run_monte_carlo, worst_single_camera
    p, e, r = sample
    res = run_monte_carlo(e, p, TolSettings(samples=60, seed=3))
    assert len(res.blind) == 60 and len(res.worst) == 10 and res.sensitivity
    assert res.blind.max() >= res.nominal_blind - 1e-9
    w = worst_single_camera(e, p)
    assert 0 <= w[0] < 1.0


def test_targets(sample):
    from vantage.core.targets import evaluate_targets, suggest
    p, e, r = sample
    st = evaluate_targets(e, p)
    ok = [s.ok for s in st]
    assert sum(ok) == 6 and not ok[0] and not ok[-1]
    sugg = suggest(e, p, r, st)
    assert sugg, 'failing targets should get suggestions'


def test_rules(sample):
    from vantage.core.rules import collect_issues, mount_info, route_length
    p, e, r = sample
    mounts = {c.uid: mount_info(p, c, e.vg) for c in p.variant.cameras}
    assert all(not m.floating for m in mounts.values())
    assert all(m.zone_kind == 'allowed' for m in mounts.values())
    issues = collect_issues(p, r, e.vg, mounts)
    assert [i.severity for i in issues] == ['critical', 'warning']
    assert 'disk set holds' in issues[1].message          # six raw cameras fill 8 TB within the shift
    c = p.variant.by_tag('F')
    c.cable_m = 1.0
    issues = collect_issues(p, r, e.vg, mounts)
    assert any(i.severity == 'serious' and 'cable' in i.message.lower() for i in issues)
    c.cable_m = 3.0
    assert route_length(p, c) < 3.0


def test_optimiser_small(sample):
    from vantage.core.optimiser import Optimiser, OptSettings
    p, e, r = sample
    o = Optimiser(e, p, OptSettings(generations=2, population=8, tol_samples=2))
    o.run()
    assert o.evals >= 16 and o.front
    v = o.to_variant(o.front[0], 'test')
    assert len(v.cameras) >= 4


def test_project_roundtrip(tmp_path, sample):
    p, e, r = sample
    path = str(tmp_path / 'x.vantage')
    p.save(path, b'')
    q = Project.load(path)
    assert q.vehicle.mesh.n_tris == p.vehicle.mesh.n_tris
    assert [v.name for v in q.variants] == [v.name for v in p.variants]
    assert q.variant.by_tag('ML').yaw == 65.0
    assert len(q.photos.frames) == 19 and abs(q.photos.rms - p.photos.rms) < 1e-9
    assert len(q.vehicle.mesh.zones) == len(p.vehicle.mesh.zones)


def test_templates(qapp):
    for key in ('sedan', 'hatchback', 'suv', 'van', 'truck'):
        p = samples.template_project(key)
        r = Engine().analyse(p)
        assert r is not None and len(p.variant.cameras) >= 4, key
