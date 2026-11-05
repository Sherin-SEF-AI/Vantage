"""core.regions: the sector × range grid every per-region perception figure uses."""
import numpy as np

from vantage.core import regions as R
from vantage.core import samples


def test_sectors_are_centred_on_straight_ahead_and_turn_left():
    g = R.RegionGrid()
    names = g.sector_names()
    ang = np.radians([0, 22, 23, 90, 180, -90, -23, -22])
    s = g.sector_of(30 * np.cos(ang), 30 * np.sin(ang))
    assert [names[i] for i in s] == ['front', 'front', 'front-left', 'left', 'rear', 'right', 'front-right', 'front']


def test_range_bins_and_the_open_last_bin():
    g = R.RegionGrid()
    r = np.array([0.0, 9.99, 10.0, 39.0, 59.9, 60.0, 80.0, 500.0, np.nan])
    assert g.range_bin_of(r, 0 * r).tolist() == [0, 0, 1, 2, 3, 4, 5, 5, -1]
    assert g.range_names() == ['0–10 m', '10–20 m', '20–40 m', '40–60 m', '60–80 m', '80+ m']
    g2 = R.RegionGrid(max_range_m=100.0)
    assert g2.range_bin_of(np.array([99.0, 101.0]), np.zeros(2)).tolist() == [5, -1]
    assert g2.range_names()[-1] == '80–100 m'


def test_ids_names_and_round_trip():
    g = R.RegionGrid(origin_xy=(1.2, 0.0))
    assert len(g.regions()) == g.n_regions == 48
    assert g.region_id(2, 3) == 'S2R3' and g.parse_id('S2R3') == (2, 3) and g.parse_id('S9R0') is None
    assert g.region_name(2, 3) == 'left · 40–60 m'
    h = R.RegionGrid.from_dict(g.to_dict())
    assert (h.sectors, h.range_edges, h.origin_xy) == (g.sectors, g.range_edges, g.origin_xy)
    junk = R.RegionGrid.from_dict({'sectors': 'x', 'range_edges': [-1, 'a'], 'origin_xy': None})
    assert junk.sectors == 8 and junk.range_edges == R.DEFAULT_RANGE_EDGES


def test_centres_fall_in_their_own_region_and_areas_add_up():
    g = R.RegionGrid(origin_xy=(1.0, 0.5), max_range_m=100.0)
    for s, r, _, _ in g.regions():
        x, y = g.centre_xy(s, r)
        assert (int(g.sector_of(x, y)), int(g.range_bin_of(x, y))) == (s, r)
    assert abs(sum(g.area_m2(s, r) for s, r, _, _ in g.regions()) - np.pi * 100 ** 2) < 1e-6


def test_the_sample_grid_is_centred_on_its_footprint():
    p = samples.tigor_sample_project()
    g = R.default_grid(p)
    v = p.vehicle.mesh.vertices
    assert abs(g.origin_xy[0] - (v[:, 0].min() + v[:, 0].max()) / 2) < 1e-9 and g.origin_xy[1] == 0.0
    # a point just ahead of the front bumper and one just behind the rear one are both 0–10 m, front and rear
    assert g.assign(v[:, 0].max() + 1.0, 0.0) == (0, 0) and g.assign(v[:, 0].min() - 1.0, 0.0) == (4, 0)
