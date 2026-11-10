"""The fixes to existing features found while filming the new-features video: the learned model's Propose in the
background and its store line, the Foundation tile, the rig perception list, the body-edge drift indicator, the
image-size fields, the online calibration's units, the re-drive's second recording out of sample, the design agent
without a time limit, the critical-target drives that never delete what Vantage did not write, and the Tigor body's
glass edges."""
import json
import math
import os
import threading
import time

import numpy as np
import pytest

from vantage.core import samples


@pytest.fixture(scope='module')
def sample():
    return samples.tigor_sample_project()


@pytest.fixture(scope='module')
def qapp():
    from PyQt6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


# ======================================================================= the body-edge drift indicator
def _shifted(ev, shift, scale=1.0, seed=0):
    """A frame whose body (the plan's, flat, on a textured background with a few strong speckles) lies `shift`
    working pixels lower, rendered at `scale` of the sensor and scaled back up, as JPEG."""
    import cv2
    rng = np.random.default_rng(seed)
    W, H = int(ev.sensor[0]), int(ev.sensor[1])
    body = np.roll(ev.body, shift, axis=0).astype(np.uint8)
    img = cv2.resize(np.where(body > 0, 70, 150).astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
    img = img.astype(np.float32) + rng.normal(0, 6, (H, W))
    spots = rng.random((H, W)) < 0.002                                  # a speckled road: strong, scattered edges
    img[spots] = 250
    img = np.clip(img, 0, 255).astype(np.uint8)
    if scale < 1:
        img = cv2.resize(cv2.resize(img, (int(W * scale), int(H * scale)), interpolation=cv2.INTER_AREA), (W, H),
                         interpolation=cv2.INTER_LINEAR)
    ok, jpg = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return cv2.imdecode(jpg, cv2.IMREAD_GRAYSCALE)


def test_drift_follows_the_shift_of_the_body_outline_at_every_render_size(sample):
    """The indicator used to read a constant 1.5–3 px on a turned camera (each outline point took its nearest
    gradient peak, mostly texture); it now finds the outline's shift as a whole and measures it across the outline."""
    from vantage.core import rig_health as rh
    from vantage.core.analysis import Engine
    ev = rh.expected_view(sample, Engine(), sample.variant.by_tag('ML'))
    for scale in (1.0, 0.25, 0.1):
        still = rh.alignment(_shifted(ev, 0, scale), ev)
        moved = [rh.alignment(_shifted(ev, s, scale), ev) for s in (5, 10, 20)]
        assert still is not None and still < 4.0, (scale, still)
        assert all(m is not None for m in moved), (scale, moved)
        assert moved[0] < moved[1] < moved[2], (scale, moved)
        px = 20 * ev.sensor[0] / ev.w                                    # 20 working px in sensor px
        assert 0.35 * px < moved[2] < 1.15 * px, (scale, moved)       # across the outline: up to the full shift


def test_drift_is_none_without_a_body_edge(sample):
    import cv2
    from vantage.core import rig_health as rh
    from vantage.core.analysis import Engine
    ev = rh.expected_view(sample, Engine(), sample.variant.by_tag('ML'))
    flat = np.full((int(ev.sensor[1]), int(ev.sensor[0])), 128, np.uint8)
    assert rh.alignment(flat, ev) is None
    ev.edge, ev.normal = ev.edge[:10], ev.normal[:10]
    assert rh.alignment(cv2.GaussianBlur(flat, (3, 3), 0), ev) is None


# ======================================================================= the Tigor body's glass
def test_tigor_glass_edges_are_cut_in_plane_and_the_surface_is_unchanged(sample):
    from vantage.core import bodygen as BG
    m = sample.vehicle.mesh
    t = m.tri
    area = 0.5 * np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1)
    assert area.min() > 1e-8                                            # no slivers
    gi = [i for i, p in enumerate(m.parts) if p.name == 'Glass'][0]
    g = t[m.face_part == gi]
    spec = BG.tigor_spec()
    xr = -spec.rear_overhang
    sz = spec.height / max(spec.top)
    # every glass corner lies inside the glass span and under the glass's upper edge (to 0.5 mm): the edges are cut,
    # no whole triangle pokes over them as the stairs did
    u = (g[..., 0] - xr) / spec.length
    top = np.interp(u, spec.u, spec.top) * sz - BG.GLASS_BELOW_ROOF
    assert (g[..., 2] <= top + 5e-4).all()
    assert (u >= spec.glass_u[0] - 5e-4 / spec.length).all() and (u <= spec.glass_u[1] + 5e-4 / spec.length).all()
    assert sample.vehicle.occluder_target == m.n_tris                  # the occluder is the whole body, as before


def test_a_cut_keeps_the_triangle_s_plane():
    from vantage.core.bodygen import _cut
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], float) + [0, 0, 2.0]
    f = np.array([[0, 1, 2]])
    v2, f2, c2 = _cut(v, f, np.array([True]), lambda p: 0.4 - p[..., 0])
    assert len(f2) == 3 and c2.all() and np.allclose(v2[:, 2], 2.0)
    a = lambda tr: 0.5 * np.linalg.norm(np.cross(tr[:, 1] - tr[:, 0], tr[:, 2] - tr[:, 0]), axis=1)   # noqa: E731
    assert math.isclose(a(v2[f2]).sum(), 0.5, rel_tol=1e-12)


# ======================================================================= the design agent
def test_design_agent_has_no_time_limit_unless_one_is_set():
    from vantage.core import rig_designer as RD
    assert RD.DesignerSettings().budget_s == 0.0
    assert RD.DesignerSettings.from_dict({'budget_s': 0}).budget_s == 0.0
    assert RD.DesignerSettings.from_dict({'budget_s': 3}).budget_s == 10.0
    assert RD.DesignerSettings.from_dict({'budget_s': 120}).budget_s == 120.0
    assert RD.DesignerSettings.from_dict({'budget_s': 99999}).budget_s == RD.LIMITS['budget_s'][1]


# ======================================================================= critical-target drives
def test_a_drive_folder_with_files_vantage_did_not_write_is_left_alone(tmp_path, sample):
    from vantage.core import critical_carla as CC
    from vantage.core import simrig as SR
    out = tmp_path / 'drives'
    (out / 'nuscenes').mkdir(parents=True)
    (out / 'nuscenes' / 'precious.json').write_text('{"keep": true}')
    (out / 'README.md').write_text('mine')
    with pytest.raises(RuntimeError, match='no Vantage drive wrote'):
        CC.drive(sample, sample.variant, SR.settings_of(sample), str(out), runner=lambda *a: None)
    assert (out / 'nuscenes' / 'precious.json').read_text() == '{"keep": true}' and (out / 'README.md').exists()
    assert not (out / '.rig-new').exists() and not (out / 'rig').exists()
    # a folder a drive of ours wrote (its run report carries Vantage's schema) is replaced as before
    (out / 'run.json').write_text(json.dumps({'schema': SR.RUN_SCHEMA, 'status': 'failed'}))
    assert CC._vantage_drive(str(out))


# ======================================================================= re-drive: the second recording
def test_the_second_recording_is_tested_with_a_calibration_fitted_without_it(monkeypatch, sample):
    from vantage.core import redrive as RDV
    seen = {}

    class Cal:
        def to_dict(self):
            return {}

    class Ev:
        info, key = {}, 'k'

    class Drv:
        notes = []

        def info(self):
            return {}
    main, wo = Cal(), Cal()

    def prepare(*a, **k):
        return {'drive': Drv(), 'scene': None, 'evidence': Ev(), 'calibration': main, 'paired': object(), 'loo': None,
                'dev': None, 'timing': {}, 'engine': None, 'cache_root': None}

    def calibrate(project, drive, scene, ev, sst, st, *a, **k):
        seen['fit_sources'] = list(st.sources)
        return wo

    def paired_validation(project, drive, scene, ev, calib, paired, sst, st, *a, held_out=None, **k):
        seen['calib'], seen['held_out'] = calib, held_out
        return {'in_calibration': not held_out}
    monkeypatch.setattr(RDV, 'prepare', prepare)
    monkeypatch.setattr(RDV, 'calibrate', calibrate)
    monkeypatch.setattr(RDV, 'paired_validation', paired_validation)
    monkeypatch.setattr(RDV, 'report', lambda *a, **k: {})
    monkeypatch.setattr(RDV, '_findings', lambda res, st: [])
    monkeypatch.setattr(RDV, 'proposed_variant', lambda *a, **k: sample.variant)
    st = RDV.RedriveSettings.from_dict({'sources': ['holdout', 'paired']})
    RDV.run(sample, None, st)
    assert seen['fit_sources'] == ['holdout'] and seen['calib'] is wo and seen['held_out'] is True
    seen.clear()
    RDV.run(sample, None, RDV.RedriveSettings.from_dict({'sources': ['paired']}))   # nothing else to fit from
    assert 'fit_sources' not in seen and seen['calib'] is main and seen['held_out'] is False


# ======================================================================= the pages
def test_page_helpers(sample, monkeypatch, qapp):
    from vantage.core import perf_model as PM
    from vantage.ui.widgets import ScrubSpinBox
    from vantage.ui.workspaces import foundation_ws as FW
    from vantage.ui.workspaces import online_calib_ws as OW
    from vantage.ui.workspaces import perf_model_ws as PW
    from vantage.ui.workspaces import rig_perception_ws as RW
    # the store line says where the records really are
    monkeypatch.delenv('VANTAGE_PERF_DIR', raising=False)
    s = PM.Store('/x/y/records.jsonl')
    assert PW.store_place(sample, s) == 'in your data folder'
    monkeypatch.setenv('VANTAGE_PERF_DIR', '/x/y')
    assert PW.store_place(sample, s) == 'in /x/y (VANTAGE_PERF_DIR)'
    # the farthest R50 is a measured one; fits beyond the tested range only bound it
    fit = lambda x, beyond=False, sub='all': {'subset': sub, 'range': {'x': [x, None], 'beyond': [beyond, False],  # noqa: E731
                                                                       'below': [False, False], 'tested': [4, 100]}}
    best, beyond = FW.farthest_measured([fit(120.0, True), fit(48.0), fit(31.0), fit(90.0, sub='near')])
    assert best['range']['x'][0] == 48.0 and len(beyond) == 1
    assert FW.farthest_measured([fit(130.0, True)])[0] is None
    # the rig's name once
    assert RW.rec_text('Data rig', 'Data rig · cut in · seed 3') == 'Data rig · cut in · seed 3'
    assert RW.rec_text('A', 'B · seed 1') == 'A · B · seed 1'
    # online calibration: translations in mm, rotations in degrees, with units
    assert OW.axis_value('x', 0.0021, '±') == '±2.1 mm'
    assert OW.axis_value('pitch', -0.1234, '+') == '−0.123°'
    assert OW.axis_value('z', 0.0005) == '0.5 mm'
    assert OW._nice(0.734912) == '0.73' and OW._nice(4.56) == '4.6' and OW._nice(12.4) == '12'
    # the image size: 12.5 %, not 13 %
    f = ScrubSpinBox('', 'plain', 1, '%', step=5.0)
    f.trim = True
    f.set_value(12.5)
    assert f.lineEdit().text() == '12.5'
    f.set_value(50.0)
    assert f.lineEdit().text() == '50'
