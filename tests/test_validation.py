"""The validation suite (vantage.validation): the registry and its settings, a quick run of every check passing and
its report (JSON, CSV, HTML, SVG figures), injected model faults caught by the checks meant to catch them, a check
that raises reported without stopping the suite, seeded reproducibility, the project given left untouched, cancel,
and the command line."""
import json
import math
import os
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from vantage import validation as V
from vantage.validation import report as RP, suite as SU


@pytest.fixture(scope='module')
def quick():
    """One quick run of every check (the Monte Carlo checks with fewer trials, tolerances unchanged)."""
    return V.run(settings=V.ValidationSettings(quick=True))


def _only(*ids, **kw):
    """Settings for the models of the given check ids, and a filter to those checks."""
    models = sorted({i.split('.')[0] for i in ids})
    return V.ValidationSettings(models=tuple(models), **kw)


# ----------------------------------------------------------------------------- the registry
def test_every_check_is_described_and_every_model_has_checks():
    specs = SU.specs()
    ids = [s.id for s in specs]
    assert len(ids) == len(set(ids)) and len(ids) >= 60
    for s in specs:
        assert s.model in V.MODEL_IDS and s.id.startswith(s.model + '.')
        assert len(s.quantity) > 40 and len(s.reference) > 10 and s.unit and s.title
        assert math.isfinite(s.tolerance) and s.tolerance >= 0
        assert s.refs and all(r in V.REFERENCES for r in s.refs), s.id
        assert s.cost in ('cheap', 'study')
    assert {s.model for s in specs} == set(SU.available_models()) and 'experiments' in SU.available_models()
    cat = V.catalogue()
    assert [c['id'] for c in cat] == ids and all(c['references'] for c in cat)
    # only the models asked for, in the models' order
    sel = V.catalogue(V.ValidationSettings(models=('weather', 'lens')))
    assert {c['model'] for c in sel} == {'lens', 'weather'} and sel[0]['model'] == 'lens'


def test_settings_are_tolerant_and_round_trip():
    st = V.ValidationSettings.from_dict({'models': ['lens', 'nope', 'weather'], 'quick': 'yes', 'seed': -5})
    assert st.models == ('lens', 'weather') and st.quick is False and st.seed == 0
    assert V.ValidationSettings.from_dict({'models': 'formats, sun', 'seed': 1e30}).models == ('sun', 'formats')
    assert V.ValidationSettings.from_dict({'seed': 1e30}).seed == SU.LIMITS['seed'][1]
    for junk in (None, [], 'x', {'seed': 'abc', 'models': 7}):
        assert V.ValidationSettings.from_dict(junk) == V.ValidationSettings()
    st = V.ValidationSettings(models=('lens',), quick=True, seed=42)
    assert V.ValidationSettings.from_dict(st.to_dict()) == st


# ----------------------------------------------------------------------------- a quick run and its report
def test_a_quick_run_passes_every_check(quick):
    failed = [(c.id, c.error, c.tolerance, c.exception) for c in quick.failed]
    assert not failed, failed
    assert quick.ok and not quick.cancelled and len(quick.checks) == len(SU.specs())
    assert V.exit_code(quick) == 0
    for c in quick.checks:
        assert math.isfinite(c.error) and 0 <= c.error <= c.tolerance, c.id
    s = V.summary(quick)
    assert s['passed'] == s['checks'] == len(quick.checks) and s['failed'] == [] and s['ok'] and s['quick']
    assert set(s['models']) == set(SU.available_models())
    json.dumps(s, allow_nan=False)


def test_the_report_files(quick, tmp_path):
    paths = V.export(quick, None, str(tmp_path))
    names = {os.path.relpath(p, tmp_path).replace(os.sep, '/') for p in paths}
    assert {'validation.json', 'validation.csv', 'validation.html'} <= names
    figs = [c for c in quick.checks if c.figure]
    assert len(figs) >= 8 and {f'figures/{c.id}.svg' for c in figs} <= names
    doc = json.loads((tmp_path / 'validation.json').read_text('utf-8'))
    assert doc['schema'] == V.SCHEMA == 'vantage.validation/1' and doc['summary']['ok']
    assert doc['environment']['vantage'] and doc['settings']['quick'] is True
    flat = [c for m in doc['models'] for c in m['checks']]
    assert [c['id'] for c in flat] == [c.id for c in quick.checks]
    for c in flat:
        assert {'quantity', 'reference', 'error', 'tolerance', 'unit', 'passed', 'detail'} <= set(c)
        assert all(r in doc['references'] for r in c['references'])
    # CSV: one row per check
    rows = (tmp_path / 'validation.csv').read_text('utf-8').strip().splitlines()
    assert rows[0].split(',')[:3] == ['id', 'model', 'title'] and len(rows) == len(quick.checks) + 1
    # HTML: every check has its anchor, every model its section; the figures are well-formed SVG
    page = (tmp_path / 'validation.html').read_text('utf-8')
    assert all(f'id="{c.id}"' in page for c in quick.checks)
    assert all(f'id="{m}"' in page for m in SU.available_models()) and 'prefers-color-scheme: dark' in page
    for c in figs:
        root = ET.fromstring((tmp_path / 'figures' / f'{c.id}.svg').read_text('utf-8'))
        assert root.tag.endswith('svg')


def test_svg_figures_of_each_kind():
    xy = SU.figure('t', 'x', 'y', [SU.series('a', [0, 1, 2], [1, 10, 100]), SU.series('b', [0, 2], [2, 3], 'points')],
                   logy=True, hline=5.0, hline_label='limit')
    bars = SU.figure('b', 'c', 'v', [SU.series('v', [0, 1], [0.2, 0.0], 'bar')], kind='bars', categories=['p', 'q'],
                     hline=0.5)
    diag = SU.figure('d', 'p', 'm', [SU.series('axes', [1, 2, 3], [1.1, 1.9, 3.2], 'points')], diag=True)
    for f in (xy, bars, diag):
        ET.fromstring(RP.svg_figure(f, standalone=True))
        assert '<svg' in RP.svg_figure(f)
    assert 'class="legend"' in RP.svg_figure(xy) and 'class="legend"' not in RP.svg_figure(diag)
    assert RP.svg_figure(None) == '' and RP.svg_figure(SU.figure('e', 'x', 'y', [])) == ''
    flat = SU.figure('f', 'x', 'y', [SU.series('c', [0, 1], [3.0, 3.0])])        # a flat line still gets an axis
    ET.fromstring(RP.svg_figure(flat, standalone=True))


# ----------------------------------------------------------------------------- injected faults are caught
def _run_ids(ids, **kw):
    res = V.run(settings=_only(*ids, **kw))
    return {c.id: c for c in res.checks if c.id in ids}, res


def test_a_wrong_rotation_convention_fails_the_footprint_checks(monkeypatch):
    """Pitch taken the other way round in the engine's camera frames (positive up): the footprints, written from the
    documented convention with scipy, no longer match."""
    from vantage.core import analysis
    real = analysis.rot_zyx
    monkeypatch.setattr(analysis, 'rot_zyx', lambda y, p, r: real(y, -p, r))
    got, res = _run_ids(['coverage.oblique_footprint', 'coverage.nadir_pyramid'], quick=True)
    assert not got['coverage.oblique_footprint'].passed and not got['coverage.nadir_pyramid'].passed
    assert V.exit_code(res) == 6 and not res.ok
    assert {f['id'] for f in V.summary(res)['failed']} >= {'coverage.oblique_footprint', 'coverage.nadir_pyramid'}


def test_published_coefficients_changed_fail_their_checks(monkeypatch):
    from vantage.core import weather as W
    # Japan's rain coefficients (P.1814 Table 2) where France's are documented
    monkeypatch.setattr(W, 'rain_optical_db_km', lambda rate: 1.58 * np.power(np.maximum(rate, 0.0), 0.63))
    # one P.838-3 coefficient of the k_H fit off in its third digit
    terms, m, c = W._P838['kH']
    monkeypatch.setitem(W._P838, 'kH', (((terms[0][0] * 1.001,) + terms[0][1:],) + terms[1:], m, c))
    got, _ = _run_ids(['weather.p1814_coefficients', 'weather.p838_table'])
    assert not got['weather.p1814_coefficients'].passed and got['weather.p1814_coefficients'].error > 0.1
    assert not got['weather.p838_table'].passed and got['weather.p838_table'].error > 1.0


def test_a_density_formula_off_by_its_aspect_fails(monkeypatch):
    """density_px_per_cm without the pixel aspect: the projected patch area disagrees for the 1.004-aspect lens."""
    from vantage.core import lens as LN
    real = LN.Intrinsics.density_px_per_cm

    def no_aspect(self, theta, rng_m):
        return real(self, theta, rng_m) / math.sqrt(self.aspect)
    monkeypatch.setattr(LN.Intrinsics, 'density_px_per_cm', no_aspect)
    got, _ = _run_ids(['lens.density'])
    c = got['lens.density']
    assert not c.passed and c.error == pytest.approx(1 - 1 / math.sqrt(1.004), rel=0.05)


def test_a_wald_interval_in_place_of_wilsons_fails(monkeypatch):
    from scipy.stats import norm
    from vantage.core import experiments as EX

    def wald(k, n, conf=0.95):
        z, ph = float(norm.ppf(0.5 + conf / 2)), k / n
        h = z * math.sqrt(ph * (1 - ph) / n)
        return [max(0.0, ph - h), min(1.0, ph + h)]
    monkeypatch.setattr(EX, 'wilson', wald)
    got, _ = _run_ids(['experiments.wilson'])
    assert not got['experiments.wilson'].passed and got['experiments.wilson'].error > 0.01


def test_a_check_that_raises_fails_and_the_suite_goes_on(monkeypatch):
    from vantage.core import weather as W

    def broken(*a, **kw):
        raise RuntimeError('permittivity unavailable')
    monkeypatch.setattr(W, 'liquid_water_coefficient', broken)
    res = V.run(settings=V.ValidationSettings(models=('weather',)))
    c = res.check('weather.p840_water')
    assert not c.passed and 'RuntimeError: permittivity unavailable' in c.exception and math.isnan(c.error)
    assert all(x.passed for x in res.checks if x.id != 'weather.p840_water')     # the others still ran
    doc = V.to_doc(res)
    json.dumps(doc, allow_nan=False)                                 # the NaN error is written as null
    row = next(x for m in doc['models'] for x in m['checks'] if x['id'] == 'weather.p840_water')
    assert row['error'] is None and row['exception']
    assert 'The check raised RuntimeError' in RP.to_html(res)


# ----------------------------------------------------------------------------- reproducible, project untouched, cancel
def test_seeded_checks_reproduce_exactly():
    ids = ['suspension.balance', 'gnss.pose_budget', 'occlusion.ground_visibility']
    a, _ = _run_ids(ids, quick=True, seed=7)
    b, _ = _run_ids(ids, quick=True, seed=7)
    c, _ = _run_ids(['gnss.pose_budget'], quick=True, seed=8)
    for i in ids:
        assert a[i].error == b[i].error and a[i].detail == b[i].detail, i
    assert c['gnss.pose_budget'].error != a['gnss.pose_budget'].error
    # a check's draws do not depend on which other checks ran with it
    alone, _ = _run_ids(['occlusion.ground_visibility'], quick=True, seed=7)
    assert alone['occlusion.ground_visibility'].error == a['occlusion.ground_visibility'].error


def _state(p):
    d = p.to_json()
    d.pop('saved_at')                                  # the time of the call
    return json.dumps(d, sort_keys=True, default=str)


def test_the_project_given_is_looked_at_but_not_changed():
    from vantage.core import samples
    p = samples.tigor_sample_project()
    p.name = 'my rig'
    before = _state(p)
    res = V.run(p, settings=V.ValidationSettings(models=('occlusion', 'formats'), quick=True))
    assert _state(p) == before
    assert res.ok and res.project == 'my rig'
    assert res.check('occlusion.ground_visibility').detail['project'] == 'my rig'
    assert set(res.check('formats.kitti').detail) >= {'reference rig', 'project'}


def test_a_project_without_a_vehicle_falls_back_to_the_sample():
    from vantage.core.project import Project
    p = Project()
    p.name = 'empty'
    res = V.run(p, settings=V.ValidationSettings(models=('occlusion', 'formats'), quick=True))
    assert res.ok and res.project == 'empty'
    assert res.check('occlusion.cube_vs_rays').detail['project'] != 'empty'
    assert 'project' not in res.check('formats.kitti').detail          # nothing of its own to export


def test_cancel_returns_the_checks_done():
    calls = []

    def cancel():
        calls.append(1)
        return len(calls) > 2
    res = V.run(settings=V.ValidationSettings(models=('lens',)), cancel=cancel)
    assert res.cancelled and len(res.checks) == 2 and V.exit_code(res) == 6 and not res.ok
    seen = []
    V.run(settings=V.ValidationSettings(models=('sun',)), progress=seen.append)
    assert seen == [0.5, 1.0]


# ----------------------------------------------------------------------------- the command line
def test_command_line_lists_runs_and_writes(tmp_path, capsys):
    from vantage.validation.__main__ import main
    assert main(['--list', '--models', 'timing']) == 0
    out = capsys.readouterr().out
    assert 'timing.sweep' in out and 'lens.inverse' not in out
    assert main([str(tmp_path / 'r'), '--models', 'timing,sun', '--quick', '--seed', '3']) == 0
    out = capsys.readouterr().out
    assert 'pass  timing.sweep' in out and '5/5 checks passed' in out
    doc = json.loads((tmp_path / 'r' / 'validation.json').read_text('utf-8'))
    assert doc['settings'] == {'models': ['sun', 'timing'], 'quick': True, 'seed': 3}
    assert main([str(tmp_path / 'x'), '--models', 'nonsense']) == 4
    assert main([str(tmp_path / 'y'), '--project', str(tmp_path / 'missing.vantage'), '--models', 'sun']) == 4


def test_cube_depths_off_the_surface_fail_the_depth_check(monkeypatch):
    """Every cube-map depth 10 % long (as a centroid's distance often was): occlusion.cube_depth fails, the hit/miss
    checks do not notice."""
    from vantage.core import occlusion as OC
    real = OC.build_cube

    def long(geom, pos, n=OC.FACE_N):
        c = real(geom, pos, n)
        return OC.CubeDepth(c.depth * np.float32(1.1), c.shade, c.pos)
    monkeypatch.setattr(OC, 'build_cube', long)
    got, _ = _run_ids(['occlusion.cube_depth', 'occlusion.cube_vs_rays'], quick=True)
    assert not got['occlusion.cube_depth'].passed and got['occlusion.cube_depth'].error > 0.5
    assert got['occlusion.cube_vs_rays'].passed
