"""Sun glare and dirt exposure: the solar position against published values (Meeus' and the SPA paper's examples) and
hand-checked spherical-astronomy cases (solar noon, a polar day's midnight), the campaign's sampling, cameras that
must and must not see the sun, body occlusion, heading distributions, GPX/CSV tracks, the dirt heuristic, the
settings, the export and the page (offscreen)."""
import copy
import datetime as dt
import json
import math

import numpy as np
import pytest

from vantage.core import exposure as X, samples, sensors
from vantage.core.analysis import Engine
from vantage.core.project import Camera, Sensor, Variant


@pytest.fixture(scope='module')
def base():
    return samples.tigor_sample_project()


@pytest.fixture(scope='module')
def eng():
    return Engine()


def _solo(base, cams=(), sens=(), **settings):
    """A copy of the sample with one new active variant holding just these cameras and sensors."""
    p = copy.deepcopy(base)
    v = Variant(name='test', cameras=list(cams), sensors=list(sens))
    p.variants.append(v)
    p.active_uid = v.uid
    p.analysis.exposure = settings
    return p


def _rig(base):
    p = copy.deepcopy(base)
    p.active_uid = next(v for v in p.variants if v.sensors).uid
    return p


def _saemundsson(h):
    """Refraction (degrees) for a true altitude h (degrees): Meeus (16.4)."""
    return 1.02 / math.tan(math.radians(h + 10.3 / (h + 5.11))) / 60.0


def _angle(a, b):
    return abs((a - b + 180.0) % 360.0 - 180.0)


# ----------------------------------------------------------------------------- solar position
def test_ephemeris_matches_meeus_example_25a():
    # Meeus, Astronomical Algorithms, example 25.a: 1992 October 13.0 TD
    e = X.solar_ephemeris(X.julian_day('1992-10-13T00:00'))
    assert float(e['declination']) == pytest.approx(-7.78507, abs=0.001)
    assert float(e['right_ascension']) == pytest.approx(198.38083, abs=0.001)
    assert float(e['apparent_longitude']) == pytest.approx(199.90895, abs=0.001)
    assert float(e['distance_au']) == pytest.approx(0.99766, abs=1e-5)
    # example 28.b (low accuracy): the equation of time 13 min 42.6 s
    assert float(e['equation_of_time_min']) == pytest.approx(13 + 42.6 / 60, abs=0.02)
    # julian_day agrees for a datetime, an aware datetime and a string
    jd = 2448908.5
    assert X.julian_day(dt.datetime(1992, 10, 13)) == pytest.approx(jd)
    assert X.julian_day(dt.datetime(1992, 10, 13, 2, tzinfo=dt.timezone(dt.timedelta(hours=2)))) == pytest.approx(jd)


def test_sun_position_matches_the_spa_paper_example():
    # Reda & Andreas (2004), NREL SPA: Golden, Colorado, 2003-10-17 12:30:30 local (UTC−7), 1830 m:
    # topocentric zenith 50.11162°, azimuth 194.34024° (refraction for 820 mbar, 11 °C; parallax 0.002°)
    az, el = X.sun_at('2003-10-17T19:30:30', 39.742476, -105.1786)
    assert 90.0 - el == pytest.approx(50.11162, abs=0.05)
    assert az == pytest.approx(194.34024, abs=0.05)
    assert abs(90.0 - el - 50.11162) < 0.01 and abs(az - 194.34024) < 0.01       # in fact far closer


@pytest.mark.parametrize('lat, lon_of, geo, az_ref', [
    # solar noon (hour angle 0) at 00:00 UT on 1992-10-13: the longitude where the true solar time is 12:00, from the
    # published equation of time; the sun's elevation is then 90° − |lat − dec|, due north or south
    (-36.85, lambda E: (720 - E) / 4, 90 - abs(-36.85 + 7.78507), 0.0),          # southern hemisphere, sun north
    (52.0, lambda E: (720 - E) / 4, 90 - (52.0 + 7.78507), 180.0),               # northern hemisphere, sun south
    # polar day: at 85° S the October sun never sets; at local midnight (hour angle 180°) it stands at its lowest,
    # |lat| + |dec| − 90 above the horizon, due south (below the pole)
    (-85.0, lambda E: -E / 4, 85.0 + 7.78507 - 90.0, 180.0),
])
def test_sun_position_hand_checked_cases(lat, lon_of, geo, az_ref):
    E = 13 + 42.6 / 60                                   # minutes, Meeus example 28.b
    when = '1992-10-13T00:00'
    az, el = X.sun_at(when, lat, lon_of(E), refract=False)
    assert el == pytest.approx(geo, abs=0.01) and _angle(az, az_ref) < 0.05
    az, el = X.sun_at(when, lat, lon_of(E))                # with refraction: Saemundsson's formula on top
    assert el == pytest.approx(geo + _saemundsson(geo), abs=0.05)
    assert el > geo                                      # refraction lifts the sun


def test_polar_day_and_night_over_a_whole_day():
    jd = X._jd0(dt.date(2026, 6, 21)) + np.arange(0, 24, 0.25) / 24.0
    _, el = X.sun_position(jd, 85.0, 0.0, refract=False)
    # at 85° N on the June solstice (declination = obliquity, 23.44°) the sun circles between δ − 5° and δ + 5°
    assert el.min() == pytest.approx(23.44 - 5.0, abs=0.1) and el.max() == pytest.approx(23.44 + 5.0, abs=0.1)
    _, el = X.sun_position(jd, -80.0, 0.0)
    assert (el < 0).all()                                # polar night in the south
    # the vectorised result equals one instant at a time
    az, el = X.sun_position(jd[:5], 48.14, 11.58)
    for k in range(5):
        a1, e1 = X.sun_position(jd[k], 48.14, 11.58)
        assert float(a1) == pytest.approx(az[k]) and float(e1) == pytest.approx(el[k])


def test_vehicle_frame_directions():
    # the sun due east, 30° up; driving north it is on the right (−y), driving east straight ahead, driving south on
    # the left
    d = X.vehicle_dirs([90.0], [30.0], [0.0, 90.0, 180.0])[0]
    c, s = math.cos(math.radians(30)), math.sin(math.radians(30))
    assert d[0] == pytest.approx([0, -c, s], abs=1e-12)
    assert d[1] == pytest.approx([c, 0, s], abs=1e-12)
    assert d[2] == pytest.approx([0, c, s], abs=1e-12)


# ----------------------------------------------------------------------------- the campaign
def test_campaign_sampling_weights_and_time_zones():
    st = X.ExposureSettings.from_dict({})
    cp = X.campaign(st, 36)
    assert cp.total_days == 365 and cp.n_days == 53 and cp.day_step == 7
    assert cp.w.sum() == pytest.approx(365 * 10.0)                       # every day, 10 h of driving
    assert len(cp.jd) == 53 * 40 and set(np.unique(cp.hour)) == set(range(8, 18))
    assert cp.local_h.min() == pytest.approx(8.125) and cp.local_h.max() == pytest.approx(17.875)
    # the first sample: 2026-01-01 08:07:30 local = 07:07:30 UTC
    assert cp.jd[0] == pytest.approx(float(X.julian_day('2026-01-01T07:07:30')), abs=1e-7)
    # a window over midnight: 22:00 to 06:00 is 8 h a day; its morning hours fall on the next day's month
    st2 = X.ExposureSettings.from_dict({'hour_from': 22, 'hour_to': 6, 'start': '2026-01-31', 'end': '2026-01-31',
                                        'minutes': 60, 'days': 1})
    c2 = X.campaign(st2)
    assert c2.w.sum() == pytest.approx(8.0) and list(c2.hour) == [22, 23, 0, 1, 2, 3, 4, 5]
    assert list(c2.month) == [0, 0, 1, 1, 1, 1, 1, 1]
    # a time zone follows daylight saving time: in July Berlin is UTC+2, an hour earlier in UTC than the fixed +1
    summer = {'start': '2026-07-01', 'end': '2026-07-01', 'days': 1}
    fixed = X.campaign(X.ExposureSettings.from_dict(summer))
    try:
        zoned = X.campaign(X.ExposureSettings.from_dict(dict(summer, timezone='Europe/Berlin')))
    except Exception:                                    # pragma: no cover - no time zone data
        pytest.skip('no time zone data')
    if not zoned.note:
        assert fixed.jd[0] - zoned.jd[0] == pytest.approx(1 / 24, abs=1e-9)
    unknown = X.campaign(X.ExposureSettings.from_dict(dict(summer, timezone='Nowhere/Atlantis')))
    assert 'not known' in unknown.note and unknown.jd[0] == pytest.approx(fixed.jd[0])
    # far too many samples: the day step goes up, the note says so
    big = X.ExposureSettings.from_dict({'start': '2026-01-01', 'end': '2030-12-31', 'minutes': 1, 'days': 1,
                                        'hour_from': 0, 'hour_to': 24})
    cb = X.campaign(big, 36)
    assert len(cb.jd) * 36 <= X.MAX_SAMPLES and cb.day_step > 1 and 'raised' in cb.note
    assert cb.w.sum() == pytest.approx(cb.total_days * 24.0)


# ----------------------------------------------------------------------------- cameras and the sun
def test_a_camera_looking_up_at_the_equator_at_an_equinox_noon_sees_the_sun(base, eng):
    cam = Camera(tag='UP', x=1.0, y=0.0, z=1.65, yaw=0.0, pitch=-90.0)
    p = _solo(base, [cam], lat=0.0, lon=0.0, utc_offset_h=0.0, start='2026-03-20', end='2026-03-20', hour_from=11.75,
              hour_to=12.25, minutes=5)
    res = X.run(p, eng)
    s = res.item('UP').sun
    assert res.metrics['daylight_share'] == 1.0
    assert s['share'] == pytest.approx(1.0) and s['low'] == 0.0 and s['blocked'] == 0.0
    assert s['hours'] == pytest.approx(0.5)
    # the sun is near the image centre: its pixel track on the equinox passes close to the principal point at noon
    tr = res.item('UP').tracks['0'][0]
    k = int(np.argmin(np.abs(tr['hour'] - 12.0)))
    im = res.item('UP').image
    assert tr['ok'][k] and math.hypot(tr['u'][k] - im['cx'], tr['v'][k] - im['cy']) < 0.05 * im['w']


def test_a_north_facing_camera_never_sees_the_winter_sun(base, eng):
    north = Camera(tag='N', x=3.3, y=0.0, z=0.7, yaw=0.0, pitch=0.0)            # driving north, looking ahead
    south = Camera(tag='S', x=-0.85, y=0.0, z=0.7, yaw=180.0, pitch=0.0)        # looking back, at the sun
    p = _solo(base, [north, south], start='2026-12-01', end='2027-01-31', headings=[[0, 1]])
    res = X.run(p, eng)
    n, s = res.item('N').sun, res.item('S').sun
    assert n['share'] == 0.0 and n['near'] == 0.0 and n['low'] == 0.0
    assert np.nan_to_num(n['month_hour']).max() == 0.0 and n['worst_months'] == []
    assert s['share'] > 0.5 and s['low'] > 0.3                # the winter sun is low and to the south
    assert X.months_text(s['worst_months']) != '—'
    f = [f for f in res.findings if f['kind'] == 'sun']
    assert f[0]['severity'] == 'serious' and f[0]['tags'] == ['S'] and 'S has the sun' in f[0]['message']
    assert any(g['severity'] == 'good' and g['tags'] == ['N'] for g in f)


def test_body_occlusion_hides_the_sun_from_a_camera_looking_into_the_body(base, eng):
    cam = Camera(tag='IN', x=1.5, y=1.0, z=0.9, yaw=-90.0, pitch=0.0)          # beside the left door, facing it
    p = _solo(base, [cam])
    with_body = X.run(p, eng).item('IN').sun
    without = X.run(p, None).item('IN').sun                                     # no engine: nothing hides the sun
    assert without['share'] > 0.2 and without['blocked'] == 0.0
    assert with_body['share'] < 0.15 * without['share']
    # what the body hides is exactly the difference
    assert with_body['share'] + with_body['blocked'] == pytest.approx(without['share'], abs=1e-12)


def test_uniform_and_weighted_headings(base, eng):
    p = _rig(base)
    shares = {}
    for name, h in (('east', [[90, 1]]), ('uniform', 'uniform'), ('west', [[270, 1]]), ('mix', [[90, 3], [270, 1]])):
        p.analysis.exposure = {'hour_from': 7, 'hour_to': 10, 'headings': h}       # mornings: the sun in the east
        res = X.run(p, eng)
        shares[name] = {t: res.item(t).sun['share'] for t in ('F', 'R')}
        if name == 'uniform':
            s = res.item('F').sun
            assert s['headings'].mean() == pytest.approx(s['share'])           # every heading equally likely
            lo, hi, best = s['worst_span']
            assert 45 <= best <= 135                    # the front camera suffers driving towards the morning sun
    assert shares['east']['F'] > shares['uniform']['F'] > shares['west']['F'] == 0.0
    assert shares['west']['R'] > shares['uniform']['R'] > shares['east']['R'] == 0.0
    # a weighted list is the weighted mean of its headings
    assert shares['mix']['F'] == pytest.approx(0.75 * shares['east']['F'] + 0.25 * shares['west']['F'])
    assert shares['uniform']['F'] == pytest.approx(shares['uniform']['R'])       # symmetric by construction


def test_the_sample_data_rig(base, eng):
    p = _rig(base)
    res = X.run(p, eng)
    assert [i.tag for i in res.items] == ['F', 'R', 'ML', 'MR', 'QL', 'QR', 'L1', 'RD1', 'RD2', 'RD3', 'RD4', 'RD5']
    f = res.item('F').sun
    assert 0.0 < f['share'] < 0.2 and f['near'] > 0 and f['low'] <= f['share']
    # shares by month and by hour add back to the total
    assert np.nansum(f['month_hour'] * res.month_hours) / res.month_hours.sum() == pytest.approx(f['share'])
    assert res.item('ML').sun['blocked'] > 0                    # the mirror housing hides some of the sky
    assert f['pitch_down'][10.0] < f['pitch_down'][5.0] < f['share']
    assert res.item('L1').sun['share'] > f['share']             # a lidar's band is wide: an indicator
    assert all(i.sun is None for i in res.items if i.kind == 'radar')
    assert res.sky_density.sum() == pytest.approx(res.metrics['daylight_share'], rel=1e-6)
    assert res.metrics['seconds'] < 5.0 and res.metrics['samples'] == 53 * 40 * 36
    # sky chart: the field's outline and the hidden directions
    it = res.item('F')
    assert it.outline is not None and len(it.outline) > 10 and it.sky_blocked.shape == it.sky_margin.shape
    assert set(it.tracks) == {'worst'} | {str(h) for h in X.TRACK_HEADINGS}


def test_hidden_cameras_nan_poses_empty_variants_and_cancel(base, eng):
    a = Camera(tag='A', x=3.3, z=0.7)
    b = Camera(tag='B', x=3.3, z=0.7, visible=False)
    c = Camera(tag='C', x=float('nan'), z=0.7)
    res = X.run(_solo(base, [a, b, c]), eng)
    assert [i.tag for i in res.items] == ['A'] and res.metrics['skipped'] == ['C']
    assert any(f['kind'] == 'pose' and 'C' in f['message'] for f in res.findings)
    empty = X.run(_solo(base), eng)
    assert empty.items == [] and empty.findings[0]['kind'] == 'empty'
    json.dumps(X.to_doc(empty), allow_nan=False)
    assert X.png(empty)[:4] == b'\x89PNG'
    calls = []
    assert X.run(_rig(base), eng, progress=lambda f, t='': calls.append(f), cancel=lambda: len(calls) > 2) is None
    # no vehicle at all: nothing hides the sun, the wheels are assumed, the dirt map is empty
    from vantage.core.project import Project
    bare = Project()
    bare.variant.cameras.append(Camera(tag='A', x=3.3, z=0.7))
    nobody = X.run(bare, Engine())
    assert nobody.items[0].sun['blocked'] == 0.0 and 'assumed' in nobody.dirt_info['wheels_from']
    assert len(nobody.dirt_faces['tri']) == 0 and X.png(nobody)[:4] == b'\x89PNG'
    json.dumps(X.to_doc(nobody, bare), allow_nan=False)
    night = X.run(_solo(base, [a], lat=80.0, start='2026-12-01', end='2026-12-31'), eng)
    assert night.metrics['daylight_share'] == 0.0 and night.items[0].sun['share'] == 0.0
    assert any(f['kind'] == 'daylight' for f in night.findings)


def test_lidar_sun_indicator(base, eng):
    m = base.sensor_models['ouster_os1_128']
    li = Sensor(kind='lidar', model_id='ouster_os1_128', tag='L1',
                **sensors.default_pose('lidar', m, base.vehicle.mesh, base.rig))
    noon = X.run(_solo(base, [], [li], lat=0.0, lon=0.0, utc_offset_h=0.0, start='2026-03-20', end='2026-03-20',
                        hour_from=11.75, hour_to=12.25), eng)
    assert noon.items[0].sun['share'] == 0.0                    # the sun overhead, far above its ±22.5° band
    north = X.run(_solo(base, [], [li], lat=60.0, lon=10.0, start='2026-12-01', end='2026-12-31'), eng)
    assert north.items[0].sun['share'] == pytest.approx(north.metrics['daylight_share'])    # always low: inside
    assert any(f['kind'] == 'lidar_sun' for f in north.findings)


# ----------------------------------------------------------------------------- route headings
def _gpx(points, times=True):
    t0 = dt.datetime(2026, 5, 4, 8, 0, tzinfo=dt.timezone.utc)
    rows = []
    for k, (la, lo, s) in enumerate(points):
        tt = f'<time>{(t0 + dt.timedelta(seconds=s)).isoformat().replace("+00:00", "Z")}</time>' if times else ''
        rows.append(f'<trkpt lat="{la:.7f}" lon="{lo:.7f}"><ele>500</ele>{tt}</trkpt>')
    return ('<?xml version="1.0" encoding="UTF-8"?><gpx version="1.1" creator="test" '
            'xmlns="http://www.topografix.com/GPX/1/1"><trk><name>t</name><trkseg>' + ''.join(rows)
            + '</trkseg></trk></gpx>')


def test_gpx_and_csv_tracks_give_heading_weights(tmp_path):
    lat0, lon0 = 48.1, 11.5
    dlat = 1000.0 / 111195.0                           # 1 km north
    dlon = 1000.0 / (111195.0 * math.cos(math.radians(lat0 + dlat)))
    pts = [(lat0 + dlat * k / 10, lon0, 10.0 * k) for k in range(11)]                     # north 1 km in 100 s
    pts += [(lat0 + dlat, lon0 + dlon * 2 * k / 20, 100.0 + 10.0 * k) for k in range(1, 21)]  # east 2 km, 200 s
    pts += [(lat0 + dlat, lon0 + 2 * dlon, 300.0 + 10.0 * k) for k in range(1, 7)]       # parked a minute
    path = tmp_path / 'route.gpx'
    path.write_text(_gpx(pts))
    r = X.read_route(str(path))
    h = dict((round(a), w) for a, w in r['headings'])
    assert r['weighting'] == 'time' and set(h) == {0, 90}
    assert h[0] == pytest.approx(1 / 3, abs=0.01) and h[90] == pytest.approx(2 / 3, abs=0.01)
    assert r['km'] == pytest.approx(3.0, rel=0.01) and r['hours'] == pytest.approx(360 / 3600)
    assert r['lat'] == pytest.approx(lat0 + 0.75 * dlat, abs=0.002) and r['name'] == 'route.gpx'
    # the same track without times: weighted by distance, still 1 : 2
    r2 = X.route_headings(X.parse_gpx(_gpx(pts, times=False)))
    assert r2['weighting'] == 'distance' and dict((round(a), w) for a, w in r2['headings'])[90] == \
        pytest.approx(2 / 3, abs=0.01)
    # a CSV track with a header and ISO times; going south-west
    csv_path = tmp_path / 'track.csv'
    csv_path.write_text('lat,lon,time\n' + '\n'.join(f'{lat0 - 0.001 * k},{lon0 - 0.0015 * k},'
                                                     f'2026-05-04T08:00:{10 * k:02d}Z' for k in range(6)))
    r3 = X.read_route(str(csv_path))
    assert len(r3['headings']) == 1 and 200 <= r3['headings'][0][0] <= 250
    with pytest.raises(ValueError):                    # one point: no segment to take a heading from
        X.route_headings([(np.array([1.0]), np.array([2.0]), np.array([np.nan]))])
    bad = tmp_path / 'bad.gpx'
    bad.write_text('<gpx><trk>')
    with pytest.raises(ValueError):
        X.read_route(str(bad))
    # typed headings
    assert X.parse_heading_text('0:3, 90, 180:3') == [[0.0, 3.0], [90.0, 1.0], [180.0, 3.0]]
    assert X.parse_heading_text('north') is None and X.parse_heading_text('0:0') is None
    assert X.parse_heading_text(X.heading_text([[10.0, 0.25], [190.0, 0.75]])) == [[10.0, 0.25], [190.0, 0.75]]


# ----------------------------------------------------------------------------- dirt and spray
def test_dirt_a_low_camera_behind_a_wheel_scores_higher_than_a_roof_camera(base, eng):
    low = Camera(tag='LOW', x=-0.3, y=0.86, z=0.4, yaw=180.0, pitch=10.0)       # behind the left rear wheel
    roof = Camera(tag='ROOF', x=1.0, y=0.0, z=1.6, yaw=0.0, pitch=5.0)
    p = _solo(base, [low, roof])
    res = X.run(p, eng)
    a, b = res.item('LOW').dirt, res.item('ROOF').dirt
    assert a['score'] > b['score'] + 0.3 and a['class'] == 'high' and b['class'] == 'low'
    assert a['factors']['spray'] > 0.5 and b['factors']['spray'] < 0.01
    assert a['dominant'] in ('spray', 'wake') and 'washer' in a['advice']
    assert any(f['kind'] == 'dirt' and f['tags'] == ['LOW'] for f in res.findings)
    # recessed: 0.6 × (before the cap at 1, the factors are unchanged)
    p.analysis.exposure = {'recessed': [roof.uid]}
    b2 = X.run(p, eng).item('ROOF').dirt
    assert b2['score'] == pytest.approx(0.6 * b['score']) and b2['recessed'] and b2['factors'] == b['factors']
    # the Issues list gets the high one (cheap and cached)
    iss = X.issues(p)
    assert [i.uids for i in iss] == [[low.uid]] and 'high dirt' in iss[0].message and iss[0].severity == 'warning'
    assert X.issues(p) == iss


def test_dirt_factors_by_hand():
    wheels = [(0.0, 0.8, 0.3, 0.3)]
    body = {'height': 1.5, 'box_factor': 1.0}
    up, fwd, back = np.array([[0, 0, 1.0]]), np.array([[1.0, 0, 0]]), np.array([[-1.0, 0, 0]])
    # straight behind the tyre at the ground, on its track: full spray
    f = X.dirt_factors([[-0.3, 0.8, 0.1]], [back], back, wheels, body)
    assert f['spray'][0] == pytest.approx(0.75 + 0.25 * 0.25)               # 0.1 m outboard of the inner edge
    # 1.5 m further back: e⁻¹ of it; 0.8 m up (0.5 above 0.3): e^-(0.5/0.45)²
    g = X.dirt_factors([[-1.8, 0.8, 0.1], [-0.3, 0.8, 0.8]], [back], back, wheels, body)
    assert g['spray'][0] == pytest.approx(f['spray'][0] * math.exp(-1.0))
    assert g['spray'][1] == pytest.approx(f['spray'][0] * math.exp(-(0.5 / 0.45) ** 2))
    # the other side of the car is far inboard of this wheel
    assert X.dirt_factors([[-0.3, -0.8, 0.1]], [back], back, wheels, body)['spray'][0] < 0.02
    # rear wake: full for a backward window at the ground, 0.4 of it at the roof; none facing forward
    assert f['wake'][0] == pytest.approx(1.0 - 0.6 * 0.1 / 1.5)
    assert X.dirt_factors([[0, 0, 1.5]], [back], back, wheels, body)['wake'][0] == pytest.approx(0.4)
    h = X.dirt_factors([[3.5, 0, 0.5]], [fwd], fwd, wheels, body)
    assert h['wake'][0] == 0.0 and h['frontal'][0] == pytest.approx(1.0)      # bumper height, facing forwards
    # attitude: level 0.3, up 1, 25° down 0
    down = np.array([[math.cos(math.radians(25)), 0, -math.sin(math.radians(25))]])
    assert X.dirt_factors([[0, 0, 1]], [up], up, wheels, body)['attitude'][0] == 1.0
    assert X.dirt_factors([[0, 0, 1]], [fwd], fwd, wheels, body)['attitude'][0] == pytest.approx(0.3)
    assert X.dirt_factors([[0, 0, 1]], [down], down, wheels, body)['attitude'][0] == 0.0
    assert h['height'][0] == pytest.approx(math.exp(-0.5 / 0.7))
    sc = X.dirt_score(h)[0]
    assert sc == pytest.approx(min(1.0, sum(X.DIRT_WEIGHTS[k] * h[k][0] for k in X.DIRT_WEIGHTS)))
    assert [X.dirt_class(v) for v in (0.1, 0.3, 0.49, 0.5)] == ['low', 'moderate', 'moderate', 'high']


def test_dirt_map_of_the_body(base, eng):
    res = X.run(_rig(base), eng)
    sc, tri = res.dirt_faces['score'], res.dirt_faces['tri']
    c = tri.mean(axis=1)
    ok = np.isfinite(sc)
    assert len(sc) == base.vehicle.mesh.n_tris and (~ok).sum() > 0                 # the wheels are left out
    roof = ok & (c[:, 2] > 1.35)
    sill = ok & (c[:, 2] < 0.45) & (np.abs(c[:, 0] - (-0.4)) < 0.3)              # low, just behind the rear wheels
    assert sc[sill].mean() > sc[roof].mean() + 0.2
    # the four views hold every face, far to near
    for name, _, _ in X.DIRT_VIEWS:
        poly, s, glass, shade = X.view_faces(res, name)
        assert poly.shape == (len(sc), 3, 2) and np.nanmax(s) <= 1.0 and shade.min() >= 0.78 - 1e-6
    # lidar windows: eight facets, the worst one named
    li = res.item('L1').dirt
    assert len(li['facets']) == 8 and li['facet'] in X.FACET_NAMES
    assert li['facets']['front'] > li['facets']['left']
    # radar fascias are scored, with the note that it matters less
    assert all('77 GHz' in i.dirt['advice'] for i in res.items if i.kind == 'radar')
    assert not any(f['kind'] == 'dirt' and set(f['tags']) & {'RD1', 'RD2'} for f in res.findings)


# ----------------------------------------------------------------------------- settings and export
def test_settings_are_clamped_and_round_trip():
    st = X.ExposureSettings.from_dict({'lat': 200, 'lon': 'x', 'minutes': 0, 'days': 1e9, 'hour_from': -5,
                                       'start': 'nonsense', 'end': '2020-01-01', 'headings': [[10, -1]],
                                       'flare_margin_deg': float('nan'), 'timezone': 'bad zone!', 'recessed': [3, 'a'],
                                       'route': {'name': 'r', 'headings': 'x'}, 'low_sun_deg': '12'})
    assert st.lat == 90.0 and st.lon == 11.58 and st.minutes == 1.0 and st.days == 60.0 and st.hour_from == 0.0
    assert st.start == '2020-01-01' and (dt.date.fromisoformat(st.end) - dt.date(2020, 1, 1)).days == \
        X.MAX_SPAN_DAYS - 1                              # swapped, then cut to five years
    assert st.headings == 'uniform' and st.flare_margin_deg == 10.0 and st.timezone == ''
    assert st.recessed == ['a'] and st.route == {} and st.low_sun_deg == 12.0
    for junk in (None, [], 'x', {'headings': [[1, 2, 3], ['a', 1]]}):
        assert X.ExposureSettings.from_dict(junk).to_dict()['lat'] == 48.14
    st = X.ExposureSettings.from_dict({'headings': [[370, 2], [90, 2]], 'timezone': 'Asia/Kolkata',
                                       'route': {'name': 'drive.gpx', 'headings': [[0, 1]], 'km': 12.5}})
    assert st.headings == [[10.0, 2.0], [90.0, 2.0]] and st.heading_table()[1].tolist() == [0.5, 0.5]
    assert X.ExposureSettings.from_dict(json.loads(json.dumps(st.to_dict()))).to_dict() == st.to_dict()
    assert X.ExposureSettings.from_dict({'heading_step': 45}).heading_table()[0].tolist() == \
        [0, 45, 90, 135, 180, 225, 270, 315]


def test_json_export_round_trip_and_files(base, eng, tmp_path):
    p = _rig(base)
    res = X.run(p, eng)
    doc = X.to_doc(res, p)
    text = json.dumps(doc, allow_nan=False)                    # no NaN anywhere: valid JSON
    back = json.loads(text)
    assert back == json.loads(json.dumps(doc))
    assert back['schema'] == X.SCHEMA == 'vantage.exposure/1' and back['variant'] == res.variant
    assert 'ISO 8855' in back['frame'] and back['units']['length'] == 'm'
    assert [c['tag'] for c in back['cameras']] == ['F', 'R', 'ML', 'MR', 'QL', 'QR']
    assert [c['tag'] for c in back['lidars']] == ['L1'] and len(back['radars']) == 5
    f = back['cameras'][0]
    assert f['sun']['in_field_pct'] == pytest.approx(100 * res.item('F').sun['share'], abs=1e-3)
    assert len(f['sun']['month_hour_pct']) == 12 and len(f['sun']['month_hour_pct'][0]) == 24
    assert f['sun']['month_hour_pct'][0][3] is None                            # 03 h: nobody drives
    assert f['dirt']['class'] == res.item('F').dirt['class'] and f['sun_tracks']['days']
    assert back['settings'] == res.settings.to_dict() and back['assumptions'] == X.ASSUMPTIONS
    assert back['radars'][0]['sun'] is None and 'sunlight' in back['radars'][0]['note']
    s = X.summary(res)
    json.dumps(s, allow_nan=False)
    assert s['items'][0]['tag'] == 'F' and 'sun_in_image_pct' in s['items'][0]
    names = [n for n, _ in X.files(res, p)]
    assert names == ['exposure.json', 'exposure.png']
    paths = X.export(res, p, str(tmp_path / 'out'))
    import cv2
    img = cv2.imread(paths[1])
    assert img is not None and img.shape[1] == 1400 and img.shape[0] > 600
    assert json.loads(open(paths[0], encoding='utf-8').read())['schema'] == X.SCHEMA


# ----------------------------------------------------------------------------- the page, driven like a user
from test_gui import app, win, pump, settle, _wait     # noqa: E402,F401  (fixtures)


def _open(win, name, cls):
    if name not in win._page_widgets:
        pg = cls(win.ctx)
        win.pages.addWidget(pg)
        win._page_widgets[name] = pg
    win.set_page(name)
    return win._page_widgets[name]


def _type_into(app, w, text):
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest
    le = w.lineEdit() if hasattr(w, 'lineEdit') else w
    QTest.mouseClick(le, Qt.MouseButton.LeftButton)
    QTest.keyClick(le, Qt.Key.Key_A, Qt.KeyboardModifier.ControlModifier)
    QTest.keyClicks(le, text)
    QTest.keyClick(le, Qt.Key.Key_Return)
    pump(app, 0.1)


def test_page_runs_edits_undoes_selects_and_exports(app, win, tmp_path):
    from PyQt6.QtCore import QPoint, Qt
    from PyQt6.QtTest import QTest
    from vantage.ui.workspaces.exposure_ws import ExposurePage
    store = win.store
    rig = next(v for v in store.project.variants if v.sensors)                  # the data rig, as the variant box
    store.edit('variants', lambda pj: setattr(pj, 'active_uid', rig.uid), f'Switched to {rig.name}')
    pump(app, 0.2)
    pg = _open(win, 'exposure', ExposurePage)
    assert win.page_name() == 'exposure' and pg.isVisible()
    ctl = win.ctx.exposure
    _wait(app, lambda: ctl.result is not None and not ctl.busy, 60)         # a quick study runs when the page shows
    pump(app, 0.1)
    res = ctl.result
    assert pg.table.rowCount() == len(res.items) == 12
    assert pg.table.item(0, 0).text().startswith('F ·') and pg.table.item(0, 1).text() == \
        f'{100 * res.item("F").sun["share"]:.1f}'
    assert pg.table.item(7, 1).text() == '—'                                   # RD1: radars do not see sunlight
    assert len(pg.findings) == len(res.findings) and pg.heat.item.tag == 'F' and pg.sky.item.tag == 'F'
    assert 'samples' in pg.status.text()
    f_share = pg.table.item(0, 1).text()
    # Run: a main-window job with the job bar
    QTest.mouseClick(pg.b_run, Qt.MouseButton.LeftButton)
    assert 'exposure' in win._jobs and not pg.b_run.isEnabled()
    _wait(app, lambda: ctl.result is not res and not ctl.busy, 60)
    pump(app, 0.1)
    assert pg.b_run.isEnabled() and pg.table.item(0, 1).text() == f_share
    # Sydney instead of Munich, typed into the latitude field: saved, re-run by itself, the table changes; undo
    before = ctl.result
    _type_into(app, pg.fields['lat'], '-33.87')
    assert store.project.analysis.exposure['lat'] == pytest.approx(-33.87)
    _wait(app, lambda: ctl.result is not before and not ctl.busy and not ctl.stale, 60)
    pump(app, 0.1)
    assert ctl.result.settings.lat == pytest.approx(-33.87) and pg.table.item(0, 1).text() != f_share
    assert pg.fields['lat'].value() == pytest.approx(-33.87)
    store.undo.undo()
    _wait(app, lambda: ctl.result.settings.lat == 48.14 and not ctl.busy and not ctl.stale, 60)
    pump(app, 0.1)
    assert store.project.analysis.exposure == {} and pg.table.item(0, 1).text() == f_share
    assert pg.fields['lat'].value() == pytest.approx(48.14)
    # a date that is not one is refused; the field shows what is stored
    _type_into(app, pg.texts['start'], 'yesterday')
    assert store.project.analysis.exposure == {} and pg.texts['start'].text() == '2026-01-01'
    # a place from the presets: latitude, longitude, offset and time zone as one step
    pg.preset.activated.emit(next(k for k, row in enumerate(pg.preset.itemText(i) for i in range(pg.preset.count()))
                                  if row == 'Chennai'))
    pump(app, 0.05)
    ex = store.project.analysis.exposure
    assert (ex['lat'], ex['lon'], ex['utc_offset_h'], ex['timezone']) == (13.08, 80.27, 5.5, 'Asia/Kolkata')
    assert pg.texts['timezone'].text() == 'Asia/Kolkata' and pg.preset.currentIndex() == 0
    # typed headings: mostly north-south
    _type_into(app, pg.head_text, '0:3, 180:3, 90:1, 270:1')
    assert store.project.analysis.exposure['headings'] == [[0.0, 3.0], [180.0, 3.0], [90.0, 1.0], [270.0, 1.0]]
    assert pg.mode.value() == 'route' and not pg.step_f.isEnabled() and '4 typed headings' in pg.route_lbl.text()
    _type_into(app, pg.head_text, 'north please')                                   # refused, the list stays
    assert len(store.project.analysis.exposure['headings']) == 4 and pg.head_text.text().startswith('0:3')
    _wait(app, lambda: not ctl.busy and not ctl.stale and isinstance(ctl.result.settings.headings, list), 60)
    pg.b_defaults.click()                                 # back to the defaults: one step
    pump(app, 0.05)
    assert store.project.analysis.exposure == {} and pg.mode.value() == 'uniform'
    store.undo.undo()
    pump(app, 0.05)
    assert len(store.project.analysis.exposure['headings']) == 4
    store.undo.undo()
    store.undo.undo()
    pump(app, 0.05)
    assert store.project.analysis.exposure == {}
    _wait(app, lambda: not ctl.busy and not ctl.stale, 60)
    # pick the right mirror camera in the table: the charts follow
    r = next(k for k, i in enumerate(ctl.result.items) if i.tag == 'MR')
    QTest.mouseClick(pg.table.viewport(), Qt.MouseButton.LeftButton,
                     pos=pg.table.visualItemRect(pg.table.item(r, 0)).center())
    pump(app, 0.1)
    assert pg.sel_uid == ctl.result.item('MR').uid and pg.heat.item.tag == 'MR' and pg.tracks.item.tag == 'MR'
    assert pg.dirt.selected == pg.sel_uid
    pg.head.buttons['90'].click()
    pump(app, 0.05)
    assert pg.tracks.key == '90' and pg.tracks.tracks()[0]['heading'] == 90.0
    for w in (pg.heat, pg.sky, pg.tracks, pg.dirt, pg.rose):
        w.grab()                                        # paints with the result (no exception)
        for fx, fy in ((0.5, 0.5), (0.3, 0.6), (0.7, 0.3)):            # hover: the tooltips
            QTest.mouseMove(w, QPoint(int(w.width() * fx), int(w.height() * fy)))
            pump(app, 0.02)
    pg.dirt.picked.emit(ctl.result.item('L1').uid)        # a click on a marker in the dirt map
    pump(app, 0.05)
    assert pg.heat.item.tag == 'L1' and pg.table.currentRow() == 6 and pg.tracks.item.tag == 'L1'
    pg.tracks.grab()
    # recessed, ticked in the table: one undo step
    chk = pg.table.item(0, 9)
    chk.setCheckState(Qt.CheckState.Checked)
    pump(app, 0.05)
    uid_f = ctl.result.item('F').uid
    assert store.project.analysis.exposure['recessed'] == [uid_f]
    _wait(app, lambda: ctl.result.item('F').dirt['recessed'] and not ctl.busy and not ctl.stale, 60)
    store.undo.undo()
    pump(app, 0.1)
    assert store.project.analysis.exposure == {}
    # a route's headings from a GPX file; the location follows the track
    lat0, lon0 = 12.97, 77.59
    pts = [(lat0 + 0.0009 * k, lon0, 10.0 * k) for k in range(12)]
    path = tmp_path / 'drive.gpx'
    path.write_text(_gpx(pts))
    route = pg.import_route(str(path))
    assert route is not None and pg.mode.value() == 'route'
    ex = store.project.analysis.exposure
    assert ex['headings'] == [[0.0, 1.0]] and ex['lat'] == pytest.approx(lat0 + 0.0009 * 5.5, abs=1e-4)
    assert 'drive.gpx' in pg.route_lbl.text()
    pg.mode.buttons['uniform'].click()
    pump(app, 0.05)
    assert store.project.analysis.exposure['headings'] == 'uniform'
    pg.mode.buttons['route'].click()                      # back to the remembered route
    pump(app, 0.05)
    assert store.project.analysis.exposure['headings'] == [[0.0, 1.0]]
    _wait(app, lambda: not ctl.busy and not ctl.stale, 60)
    paths = pg.export(str(tmp_path / 'export'))
    assert [p.rsplit('/', 1)[-1] for p in paths] == ['exposure.json', 'exposure.png']
    doc = json.loads(open(paths[0], encoding='utf-8').read())
    assert doc['schema'] == X.SCHEMA and doc['settings']['headings'] == [[0.0, 1.0]]
    # leave the page: edits are not computed until it shows again
    win.set_page('viewports')
    pump(app, 0.2)
    shown = ctl.result
    store.edit('analysis', lambda q: setattr(q.analysis, 'exposure', {'lat': 10.0}), 'test')
    pump(app, 0.6)
    assert ctl.result is shown and ctl.stale and not ctl.busy
    win.set_page('exposure')
    _wait(app, lambda: ctl.result is not shown and not ctl.busy and not ctl.stale, 60)
    assert ctl.result.settings.lat == 10.0
    # another project opened while the page shows: its active variant (six cameras, no sensors) is studied
    store.set_project(samples.tigor_sample_project())
    pump(app, 0.05)
    _wait(app, lambda: ctl.result is not None and not ctl.busy and ctl.result.variant == store.variant.name, 60)
    pump(app, 0.1)
    assert pg.table.rowCount() == 6 and pg.heat.item.tag == 'F' and ctl.result.settings.lat == 48.14
