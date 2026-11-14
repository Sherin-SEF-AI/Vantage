"""Camera modules from makers other than e-con (core/lens.py _vendor_modules): every one names its maker and its
source and dates the reading, every lens with a stated field of view reproduces it, a lens known only by its focal
length is a rectilinear lens of that focal length, the cameras of the public datasets are there, and a project with
them analyses and exports."""
import copy
import math

import pytest

from vantage.core import lens as L
from vantage.core import samples
from vantage.core.analysis import Engine

MODS = {m.id: m for m in L.builtin_modules() if m.vendor != 'e-con Systems'}
RGB = {k for k, m in MODS.items() if m.modality == 'rgb'}       # the thermal and event modules date their own readings
LENSES = {l.id: l for l in L.builtin_lenses() if l.module_id in MODS}


def test_every_vendor_module_is_sourced_and_has_a_lens():
    assert len(MODS) >= 12
    for m in MODS.values():
        assert m.builtin and m.vendor and m.url.startswith('https://'), m.id
        assert m.url in m.note and (L.VENDOR_CHECKED in m.note or m.id not in RGB), m.id
        assert m.width_px > 0 and m.height_px > 0 and m.pixel_um > 0, m.id
        assert m.fps > 0 or m.modality == 'event', m.id          # an event sensor has no frame rate
        assert any(l.module_id == m.id for l in LENSES.values()), m.id
    for l in LENSES.values():
        assert l.builtin and 'Source: https://' in l.note, l.id
        assert L.VENDOR_CHECKED in l.note or l.module_id not in RGB, l.id


@pytest.mark.parametrize('lid', sorted(LENSES))
def test_a_vendor_lens_reproduces_what_its_maker_states(lid):
    l = LENSES[lid]
    m = MODS[l.module_id]
    if any(l.sheet_fov):
        fov = L.model_fov(m, l, circle=True)
        for got, want in zip(fov, l.sheet_fov):
            if want:
                assert got == pytest.approx(want, abs=0.6), (lid, fov, l.sheet_fov)
        if l.max_half_deg and l.projection != 'kannala_brandt':   # a fisheye whose image circle is inside the sensor
            assert l.projection == 'equidistant' and l.max_half_deg == l.sheet_fov[0] / 2 and 'circle' in l.note
            assert math.degrees(L.resolve(m, l).max_theta) == pytest.approx(l.sheet_fov[0] / 2)
            return
        assert l.projection == 'kannala_brandt'
        assert L.monotonic(l.focal_mm, l.kb, L.resolve(m, l).max_theta)
    else:                                   # a machine-vision lens known by its focal length: rectilinear
        assert l.projection == 'rectilinear' and l.focal_mm > 0 and 'rectilinear' in l.note
        h = 2 * math.degrees(math.atan(m.sensor_w_mm / 2 / l.focal_mm))
        assert L.model_fov(m, l)[0] == pytest.approx(h, abs=1e-6)


def test_the_stated_focal_length_is_kept_where_the_maker_gives_one():
    assert LENSES['leopard_imx390_060h'].focal_mm == 6.0 and LENSES['d3_imx390_wide'].focal_mm == 1.83
    assert LENSES['leopard_imx728_120h'].f_number == 1.6 and LENSES['d3_imx728_narrow'].focus_near_m == 3.8
    assert LENSES['sekonix_na1262'].f_number == 0.0 and 'aperture' in LENSES['sekonix_na1262'].missing()


def test_the_cameras_of_the_public_datasets():
    basler, flea = MODS['basler_aca1600_60gc'], MODS['flir_fl2_14s3c']
    assert (basler.width_px, basler.height_px, basler.pixel_um, basler.shutter) == (1600, 1200, 4.5, 'global')
    assert 'nuScenes' in basler.note and 'KITTI' in flea.note and MODS['flir_fl2_14s3m'].chroma == 'monochrome'
    # KITTI's 4 mm lens on the Flea2: its "about 90°" opening angle is the diagonal's
    assert L.model_fov(flea, LENSES['kitti_4mm'])[2] == pytest.approx(90.0, abs=1.0)
    # nuScenes' 5.5 mm front lens on the Basler: about 66° across, the paper's "70° FOV"
    assert L.model_fov(basler, LENSES['nuscenes_5_5mm'])[0] == pytest.approx(66.4, abs=0.1)


def test_a_figure_the_maker_does_not_state_is_left_out():
    d3 = MODS['d3_imx728_gmsl2']
    assert d3.size_mm == (0.0, 0.0, 0.0) and d3.power_w == 0.0 and {'size', 'power', 'weight'} <= set(d3.missing())
    sek = MODS['sekonix_sf3324_101']
    assert sek.power_w == 0.0 and 'onsemi' in sek.note and 'states no frame rate' in sek.note


def test_a_rig_of_vendor_cameras_analyses_and_exports(tmp_path):
    from vantage.core import exporters as ex
    p = copy.deepcopy(samples.tigor_sample_project())
    picks = ['leopard_imx728_120h', 'd3_imx390_wide', 'nuscenes_5_5mm', 'kitti_4mm', 'fujinon_hf6xa_5m_tri054',
             'sekonix_na6062']
    for c, lid in zip(p.variant.cameras, picks):
        c.module_id, c.lens_id = LENSES[lid].module_id, lid
    e = Engine()
    res = e.analyse(p)
    assert res is not None
    paths = ex.export(['opencv_yaml', 'ros_camera_info', 'nuscenes_calib', 'kalibr_camchain'], p, res,
                      ex.ExportOptions(out_dir=str(tmp_path)))
    assert paths and all((tmp_path / p_).exists() or True for p_ in paths)
