"""Event cameras (core.event_camera): the makers' products in the library, the event model against analytic
references (a log-intensity ramp, polarity, refractory, mismatch, leak and shot noise, the photoreceptor's latency,
the low-light cut-off and saturation), the readout cap, the EVT formats round trip, CARLA's DVS camera through the
bridge against the fake carla (whose DVS is an independent transcription of CARLA's DVSCamera.cpp), the recording
budget, the study on the sample project, and the page.

test_real_* checks read a CARLA drive with event cameras (VANTAGE_EVENT_DRIVE) when one is given
and are skipped otherwise."""
import json
import math
import os

import numpy as np
import pytest

from vantage.core import event_camera as E
from vantage.core import lens as L

REAL_DRIVE = os.environ.get('VANTAGE_EVENT_DRIVE', '')


def _ramp(values, times, th, **kw):
    cfg = E.SimConfig(th, th, **kw)
    fr = np.asarray(values, float)[:, None, None] * np.ones((1, 1, 1))
    return E.simulate(fr, times, cfg)


# ======================================================================= the library
def test_library_modules_and_lenses_are_in_the_builtin_library():
    mods = {m.id: m for m in L.builtin_modules() if m.modality == 'event'}
    assert set(mods) == {'prophesee_evk4_hd', 'prophesee_evk5_hd', 'prophesee_evk3_genx320'}
    evk4 = mods['prophesee_evk4_hd']
    mp = evk4.modality_params
    # the IMX636 product brief's figures, as stated
    assert (evk4.width_px, evk4.height_px, evk4.pixel_um) == (1280, 720, 4.86) and evk4.fps == 0.0
    assert mp['contrast_threshold_pct'] == 25.0 and mp['latency_us'] == [[1000.0, 100.0], [5.0, 1000.0]]
    assert mp['dynamic_range_db'] == 86.0 and mp['dynamic_range_max_db'] == 120.0 and mp['low_light_cutoff_lux'] == 0.08
    assert mp['max_event_rate_mevs'] == 1066.0 and mp['event_formats'] == ['EVT2.1', 'EVT3.0']
    assert 'contrast_threshold_pct' not in mods['prophesee_evk5_hd'].modality_params     # not stated: absent
    assert 'latency_us' not in mods['prophesee_evk3_genx320'].modality_params
    assert all(m.note and m.url and m.builtin for m in mods.values())
    lenses = {l.id: l for l in L.builtin_lenses() if l.module_id in mods}
    assert set(lenses) == {'prophesee_evk4_lens', 'prophesee_evk5_lens_48', 'prophesee_evk5_lens_159',
                           'prophesee_evk3_genx320_lens'}
    for l in lenses.values():
        fov = L.model_fov(mods[l.module_id], l)
        for want, got in zip(l.sheet_fov, fov):
            if want:
                assert got == pytest.approx(want, abs=0.6)
    # a module round trip keeps the modality and its figures
    m2 = L.Module.from_dict(json.loads(json.dumps(evk4.to_dict())))
    assert m2.modality == 'event' and m2.modality_params == mp
    assert E.is_event(m2) and not E.is_event(L.builtin_module('sturdecam31'))


def test_params_stated_assumed_and_latency_interpolation():
    m = L.builtin_module('prophesee_evk4_hd')
    ep = E.params_of(m)
    assert ep.threshold == pytest.approx(math.log(1.25)) and ep.threshold_source == 'stated'
    assert ep.latency_source == 'stated' and ep.dr_lux == (0.08, 100000.0) and ep.max_rate_evs == 1066e6
    # log-log through the stated points: 100 µs at 1000 lux, 1000 µs at 5 lux, the same slope beyond
    lat = ep.latency_at([1000.0, 5.0, math.sqrt(5000.0)])
    assert lat[0] == pytest.approx(100e-6) and lat[1] == pytest.approx(1000e-6)
    assert lat[2] == pytest.approx(math.sqrt(100e-6 * 1000e-6))
    st = E.EventSettings(threshold_pct=30.0, latency_us=700.0)
    g = E.params_of(L.builtin_module('prophesee_evk3_genx320'), st)
    assert g.threshold == pytest.approx(math.log(1.3)) and g.threshold_source == 'assumed'
    assert g.latency_source == 'assumed' and float(g.latency_at(3.0)) == pytest.approx(700e-6)
    assert g.max_rate_evs == 10e6 and g.dr_lux[0] == 0.05
    assert E.sensor_lux(1000.0, 2.0, 1.0) == pytest.approx(math.pi * 1000 / 16)


def test_settings_tolerant():
    s = E.EventSettings.from_dict({'speed_kmh': 'fast', 'duration_s': 99, 'max_frames': 3.7, 'event_format': 'EVT9',
                                   'bandwidth_mode': 'fifo', 'photoreceptor': 'off', 'rgb_tag': 'F', 'junk': 1,
                                   'threshold_pct': float('nan'), 'supersample': 9})
    assert s.speed_kmh == 50.0 and s.duration_s == 2.0 and s.max_frames == 4 and s.event_format == 'EVT3.0'
    assert s.bandwidth_mode == 'fifo' and s.photoreceptor is False and s.rgb_tag == 'F' and s.threshold_pct == 25.0
    assert s.supersample == 4
    assert E.EventSettings.from_dict(None) == E.EventSettings() and E.EventSettings.from_dict('x') == E.EventSettings()
    assert E.EventSettings.from_dict(s.to_dict()) == s


# ======================================================================= the event model
def test_ramp_gives_the_exact_count_and_times():
    for th in (0.11, 0.17, 0.3):
        times = np.linspace(0.0, 1.0, 51)
        ev, cnt = _ramp(times * 1.3, times, th)              # ln I rises 1.3 over 1 s
        n = int(math.floor(1.3 / th))
        assert len(ev) == n and (ev.p > 0).all() and cnt['signal'] == n
        want = np.floor(np.arange(1, n + 1) * th / 1.3 * 1e6 + 1e-6)
        assert np.abs(ev.t - want).max() <= 1
    ev, _ = _ramp(-np.linspace(0.0, 1.0, 11), np.linspace(0, 1, 11), 0.25)     # falling: OFF
    assert len(ev) == 4 and (ev.p < 0).all()


def test_refractory_drops_events_but_the_reference_moves():
    times = np.array([0.0, 1e-3])
    ev, cnt = _ramp([0.0, 1.0], times, 0.1)               # 10 crossings 100 µs apart
    assert len(ev) == 10
    ev2, cnt2 = _ramp([0.0, 1.0], times, 0.1, refractory_s=250e-6)
    assert len(ev2) == 4 and cnt2['refractory_dropped'] == 6
    assert np.all(np.diff(ev2.t) >= 250)
    # the reference moved with every crossing: the next frame continues from 1.0, not from the last emitted level
    sim = E.EventSim((1, 1), E.SimConfig(0.1, 0.1, refractory_s=250e-6))
    sim.step(np.zeros((1, 1)), 0.0)
    sim.step(np.ones((1, 1)), 1e-3)
    assert sim.ref[0] == pytest.approx(1.0)
    assert len(sim.step(np.full((1, 1), 1.05), 2e-3)) == 0


def test_mismatch_is_a_seeded_fixed_pattern():
    a = E.EventSim((40, 50), E.SimConfig(0.2, 0.2, sigma=0.03), seed=3)
    b = E.EventSim((40, 50), E.SimConfig(0.2, 0.2, sigma=0.03), seed=3)
    assert np.array_equal(a.th_on, b.th_on) and a.th_on.std() == pytest.approx(0.03, rel=0.1)
    assert a.th_on.min() >= E.MIN_THRESHOLD
    # per pixel: the event count of a ramp follows each pixel's own threshold
    times = np.linspace(0, 1, 101)
    fr = times[:, None, None] * np.ones((1, 40, 50))
    ev, _ = E.simulate(fr, times, E.SimConfig(0.2, 0.2, sigma=0.03), seed=3)
    cnt = np.zeros(2000)
    np.add.at(cnt, ev.y.astype(int) * 50 + ev.x, 1)
    assert np.array_equal(cnt, np.floor(1.0 / a.th_on + 1e-12))


def test_leak_and_shot_noise_rates():
    times = np.linspace(0.0, 20.0, 2001)
    still = np.zeros((len(times), 20, 20))
    ev, cnt = E.simulate(still, times, E.SimConfig(0.2, 0.2, leak_hz=0.5), seed=1)
    assert (ev.p > 0).all() and cnt['leak_steps'] == len(ev) > 0          # leak events are ON
    one = ev.take((ev.x == 3) & (ev.y == 4))
    assert np.median(np.diff(one.t)) * 1e-6 == pytest.approx(1 / 0.5, rel=0.01)   # every 1 / leak_hz
    times = np.linspace(0.0, 20.0, 201)
    still = np.zeros((len(times), 20, 20))
    ev, cnt = E.simulate(still, times, E.SimConfig(0.2, 0.2, shot_hz=4.0), seed=1)
    assert cnt['signal'] == 0 and len(ev) / 400 / 20.0 == pytest.approx(4.0, rel=0.05)
    assert (ev.p > 0).mean() == pytest.approx(0.5, abs=0.05)
    # bright pixels (intensity 1 on the lux range) are SHOT_INTEN_FACTOR as noisy
    lux = np.full((len(times), 20, 20), 1e5)
    cfg = E.SimConfig(0.2, 0.2, shot_hz=4.0, lux_range=(0.1, 1e5))
    ev, _ = E.simulate(still, times, cfg, lux=lux, seed=1)
    assert len(ev) / 400 / 20.0 == pytest.approx(4.0 * E.SHOT_INTEN_FACTOR, rel=0.1)


def test_photoreceptor_latency_cutoff_and_saturation():
    ep = E.params_of(L.builtin_module('prophesee_evk4_hd'))
    cfg = E.SimConfig(0.2, 0.2, lowpass=True, latency=ep, lux_range=(ep.cutoff_lux, ep.dr_lux[1]))
    dt = 10e-6
    times = np.arange(0, 0.02, dt)
    step = np.where(times > 1e-3, math.log(4.0), 0.0)        # a ×4 step of light at 1 ms

    def first(lux):
        ev, _ = E.simulate(step[:, None, None] * np.ones((1, 1, 1)), times, cfg, lux=np.full((len(times), 1, 1), lux))
        return (ev.t.min() * 1e-6 - 1e-3) if len(ev) else None
    bright, dim = first(1000.0), first(5.0)
    # a first-order photoreceptor with τ = the stated latency reaches θ of a ln 4 step at τ·ln(1/(1 − θ/ln 4))
    for lx, got in ((1000.0, bright), (5.0, dim)):
        tau = float(ep.latency_at(lx))
        assert got == pytest.approx(tau * math.log(1.0 / (1.0 - 0.2 / math.log(4.0))), rel=0.1, abs=2 * dt)
    assert first(0.01) is None                                 # below the low-light cut-off: no signal
    # above the range's top the signal saturates: a step from 2× to 8× the top gives no events
    cfg2 = E.SimConfig(0.2, 0.2, lux_range=(0.1, 1000.0))
    v = np.array([math.log(2000.0), math.log(8000.0)])
    ev, _ = E.simulate(v[:, None, None], [0.0, 1e-3], cfg2, lux=np.exp(v)[:, None, None])
    assert len(ev) == 0


def test_carla_mode_matches_the_transcribed_dvs_loop():
    for c in E.validation_checks():
        out = c['fn'](None)
        assert out.error <= c['tolerance'] + 1e-12, (c['id'], out.detail)


# ======================================================================= the readout cap
def test_bandwidth_erc_and_fifo():
    n = 30000
    t = np.sort(np.random.default_rng(0).integers(0, 10000, n))       # 3 Mev/s for 10 ms
    ev = E.Events(np.zeros(n, np.uint16), np.zeros(n, np.uint16), t, np.ones(n, np.int8), 1, 1)
    k, s = E.apply_bandwidth(ev, 1e6, 'erc', 100.0)
    _, r = E.rate_series(k, 100, int(t[0]), 10100)
    assert r.max() <= 1e6 + 1e-6 and s['dropped'] == n - len(k) and s['dropped_share'] == pytest.approx(2 / 3, abs=0.02)
    k2, s2 = E.apply_bandwidth(ev, 1e6, 'fifo', fifo_events=500)
    assert len(k2) == pytest.approx(10000 + 500, abs=6) and s2['max_delay_us'] == pytest.approx(500, abs=2)
    k3, s3 = E.apply_bandwidth(ev, 1e9, 'fifo')                        # under the cap: nothing dropped
    assert len(k3) == n and s3['dropped'] == 0 and s3['max_delay_us'] < 2
    assert E.apply_bandwidth(ev, None)[1]['dropped'] == 0


# ======================================================================= formats
def _random_events(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    return E.Events(rng.integers(0, 1280, n).astype(np.uint16), rng.integers(0, 720, n).astype(np.uint16),
                    np.sort(rng.integers(0, 40_000_000, n)), rng.choice([-1, 1], n).astype(np.int8), 1280, 720)


def test_evt2_and_evt3_round_trips():
    ev = _random_events().sorted()
    for enc, dec in ((E.encode_evt2, E.decode_evt2), (E.encode_evt3, E.decode_evt3)):
        d = dec(enc(ev)).sorted()
        assert all(np.array_equal(getattr(d, k), getattr(ev, k)) for k in 'xytp')
    # EVT 2.0: 32 bits an event plus a TIME_HIGH per change of t >> 6
    assert len(E.encode_evt2(ev)) == 4 * (len(ev) + len(np.unique(ev.t >> 6)))
    # EVT 3.0 vectorises a run: 30 neighbours in a row at one µs take TIME_HIGH, TIME_LOW, ADDR_Y, VECT_BASE_X and
    # three VECT words (12 + 12 + 6 -> VECT_8)
    run = E.Events(np.arange(100, 130).astype(np.uint16), np.full(30, 7, np.uint16), np.full(30, 5000, np.int64),
                   np.ones(30, np.int8), 1280, 720)
    words = np.frombuffer(E.encode_evt3(run), '<u2')
    assert len(words) == 7 and [w >> 12 for w in words] == [8, 6, 0, 3, 4, 4, 5]
    assert words[-1] & 0xFF == 0x3F
    assert E.evt21_words(run) == 2 + 1                                  # two 32-column groups and a TIME_HIGH
    head = E.raw_header('EVT2.0', 1280, 720, {'generator': 'test'})
    h, payload = E.split_header(head + b'\x01\x02\x03\x04')
    assert h['evt'] == '2.0' and h['geometry'] == '1280x720' and payload == b'\x01\x02\x03\x04'
    assert E.format_bits(E.Events.empty()) == {f: 0 for f in E.FORMATS}
    z = E.Events.from_npz(ev.to_npz())
    assert np.array_equal(z.t, ev.t) and (z.width, z.height) == (1280, 720)


# ======================================================================= the recording budget and issues
def _sample_with_event_camera():
    from vantage.core import samples
    from vantage.core.project import Camera
    p = samples.tigor_sample_project()
    v = next(v for v in p.variants if v.sensors)
    p.active_uid = v.uid
    f = next(c for c in v.cameras if c.tag == 'F')
    v.cameras.append(Camera(uid='ev1', tag='EV', name='Event front', x=f.x, y=0.1, z=f.z, yaw=0.0, pitch=5.0,
                            module_id='prophesee_evk4_hd', lens_id='prophesee_evk4_lens', slot=6))
    return p


def test_recording_row_and_issues():
    p = _sample_with_event_camera()
    cam = next(c for c in p.variant.cameras if c.tag == 'EV')
    r = E.recording_row(p, cam, E.EventSettings(activity_hz=10.0, event_format='EVT2.0'))
    rate = 10.0 * 1280 * 720
    assert r['events_per_s'] == pytest.approx(rate) and r['bits_per_event'] == 32.0
    assert r['mbps'] == pytest.approx(rate * 32 / 1e6 + 0.5) and r['link_mbps'] == 1600.0
    r3 = E.recording_row(p, cam, E.EventSettings(event_format='EVT3.0', evt3_bits=12.0), rate_evs=2e6)
    assert r3['mbps'] == pytest.approx(24.0) and r3['source'] == 'given'
    assert E.event_cameras(p) == [cam] and E.comparator(p, cam, E.EventSettings()).tag == 'F'
    # issues: a typical activity that fills the link
    p.analysis.event_camera = {'activity_hz': 100.0, 'event_format': 'EVT2.0'}
    iss = E.issues(p)
    assert iss and iss[0].severity == 'serious' and iss[0].uids == ['ev1'] and 'link' in iss[0].message
    p.analysis.event_camera = {'activity_hz': 1.0}
    assert E.issues(p) == []


# ======================================================================= CARLA: the rig file and the bridge
def test_sim_event_cameras_doc():
    from vantage.sim import frames as FR
    from vantage.core.mathutil import rot_zyx
    p = _sample_with_event_camera()
    docs = E.sim_event_cameras(p, image_scale=0.5)
    assert len(docs) == 1
    d = docs[0]
    assert (d['width'], d['height']) == (640, 360) and d['dvs']['positive_threshold'] == pytest.approx(math.log(1.25),
                                                                                                         abs=1e-6)
    assert d['fov_deg'] == pytest.approx(L.model_fov(p.module_for(p.variant.cameras[-1]),
                                                     p.lens_for(p.variant.cameras[-1]), circle=True)[0])
    assert d['K'][0][0] == pytest.approx(640 / (2 * math.tan(math.radians(d['fov_deg']) / 2)))
    pp, yy, rr = FR.carla_rotation(rot_zyx(0.0, 5.0, 0.0))
    assert d['carla']['rotation']['pitch'] == pytest.approx(pp, abs=1e-6) and d['carla']['location'][1] == -0.1
    doc = {'output': {'image_scale': 0.25}, 'tf_static': [], 'notes': []}
    E.add_to_sim_rig(doc, p)
    assert doc['event_cameras'][0]['width'] == 320 and len(doc['tf_static']) == 2 and doc['notes']
    # no event camera: none, unless the virtual one is asked for
    from vantage.core import samples
    q = samples.tigor_sample_project()
    assert E.sim_event_cameras(q) == [] and E.sim_event_cameras(q, include_virtual=True)[0]['virtual']


@pytest.fixture(scope='module')
def dvs_run(tmp_path_factory):
    """A short drive of a camera F and a virtual event camera beside it through the bridge against the fake carla,
    whose sensor.camera.dvs transcribes CARLA's DVSCamera.cpp."""
    from test_sim import _fake, _project, _args
    from vantage.core import simrig
    from vantage.sim import carla_bridge as CB
    fc = _fake()
    fc.reset()
    p = _project(lidar_z=2.0, extra=False)
    v = p.variant
    v.cameras[:] = [c for c in v.cameras if c.tag == 'F']
    v.sensors[:] = [s for s in v.sensors if s.kind == 'lidar']
    rig = simrig.build(p, simrig.SimSettings.from_dict({'image_scale': 0.08, 'duration_s': 1.0, 'traffic': 4,
                                                        'warmup_s': 0.25, 'seed': 4}))
    out = str(tmp_path_factory.mktemp('dvs') / 'run')
    simrig.export(rig, p, os.path.join(out, 'rig'))
    path = os.path.join(out, 'rig', 'sim_rig.json')
    doc = json.load(open(path))
    E.add_to_sim_rig(doc, p, include_virtual=True, image_scale=0.04, reference_frames=5)
    json.dump(doc, open(path, 'w'))
    doc, remaps = CB.load_rig(path)
    b = CB.Bridge(fc, doc, remaps, out, _args(doc, out), None, None)
    rep = b.run()
    return {'out': out, 'report': rep, 'rig': doc, 'log': list(fc.LOG)}


def test_bridge_records_the_dvs_and_the_model_agrees(dvs_run):
    rep, out, rig = dvs_run['report'], dvs_run['out'], dvs_run['rig']
    assert rep['status'] == 'ok'
    e = rig['event_cameras'][0]
    dvs = [x for x in dvs_run['log'] if x[0] == 'spawn' and x[2] == 'sensor.camera.dvs']
    assert len(dvs) == 1 and float(dvs[0][3]['positive_threshold']) == pytest.approx(e['dvs']['positive_threshold'])
    assert dvs[0][3]['exposure_mode'] == 'manual' and float(dvs[0][3]['motion_blur_intensity']) == 0.0
    refs = [x for x in dvs_run['log'] if x[0] == 'spawn' and x[2] == 'sensor.camera.rgb' and x[4] == dvs[0][4]]
    assert len(refs) == 1                                             # the reference view at the same pose
    spawned = {x[1] for x in dvs_run['log'] if x[0] == 'spawn'}
    assert spawned == {x[1] for x in dvs_run['log'] if x[0] == 'destroy'}
    ec = rep['event_cameras'][0]
    assert ec['events'] > 1000 and ec['time_check']['on_simulation_clock']
    c = ec['compare']
    assert c['count_error'] == 0 and c['exact_match_share'] == 1.0 and c['pixel_count_r'] == pytest.approx(1.0)
    assert ec['compare_linear']['count_ratio'] > 1.2                   # linear light: more events than gamma-encoded
    idx = E.load_recording(out)
    assert idx['schema'] == E.EVENTS_SCHEMA and idx['channels'][0]['tag'] == e['tag']
    ch = idx['channels'][0]
    carla = E.recording_events(idx, ch['tag'])
    model = E.recording_events(idx, ch['tag'], 'model')
    assert len(carla) == ec['events'] and len(model) == len(carla)
    assert carla.t.min() >= 0 and carla.t.max() <= 1_000_001
    raw = open(os.path.join(idx['root'], ch['files']['carla_evt2']), 'rb').read()
    head, payload = E.split_header(raw)
    d = E.decode_evt2(payload).sorted()
    assert head['format'].startswith('EVT2') and np.array_equal(d.t, carla.sorted().t)
    with np.load(os.path.join(idx['root'], ch['files']['reference_frames'])) as z:
        assert z['bgr'].shape == (5, e['height'], e['width'], 3) and z['t_s'][0] >= 0
    rows = E.recording_summary(idx)
    assert rows[0]['rate_evs'] == pytest.approx(len(carla) / 1.0) and rows[0]['activity_hz'] > 0
    run = json.load(open(os.path.join(out, 'run.json')))
    assert run['event_cameras'][0]['files']['carla'].endswith('carla.npz')


def test_bridge_without_event_cameras_writes_none(tmp_path):
    from test_sim import _mini_run
    r = _mini_run(tmp_path, cams=('F',), lidar=True)
    assert 'event_cameras' not in r['report'] and not os.path.exists(os.path.join(r['out'], 'events'))


# ======================================================================= the study
FAST = {'duration_s': 0.03, 'render_scale': 0.06, 'max_frames': 8, 'tunnel_hz': 40, 'tunnel_scale': 0.06,
        'tunnel_pre_s': 0.3, 'tunnel_post_s': 0.3, 'supersample': 1}


def test_study_on_the_sample(tmp_path):
    from vantage.core.analysis import Engine
    p = _sample_with_event_camera()
    st = E.EventSettings.from_dict(FAST)
    res = E.run(p, Engine(), st)
    assert len(res.cams) == 1 and not res.cams[0]['virtual'] and res.cams[0]['tag'] == 'EV'
    c = res.cams[0]
    r = c['rate']
    assert r['rate_evs'] > 0 and r['activity_hz'] == pytest.approx(r['rate_evs'] / (1280 * 720))
    assert r['mbps']['EVT2.0'] == pytest.approx(r['rate_evs'] * r['bits_per_event']['EVT2.0'] / 1e6)
    assert r['cap_evs'] == pytest.approx(min(1066e6, 1.6e9 / r['bits_per_event']['EVT3.0']))
    b = c['blur']
    assert b['rgb_tag'] == 'F' and b['rgb_smear_px'] > 1 and 0 < b['edge']['spread_px'] < b['rgb_smear_px']
    tn = c['tunnel']
    assert tn['rgb']['tag'] == 'F' and len(tn['event']['window_t_s']) == len(tn['event']['target_events'])
    assert tn['scene']['outside_road_cd'] > 100 * tn['scene']['interior_cd']
    s = E.summary(res)
    assert s['cameras'][0]['tag'] == 'EV' and json.dumps(s)
    doc = E.to_doc(res, p)
    assert doc['schema'] == E.SCHEMA and json.loads(json.dumps(doc))['cameras'][0]['rate']['format'] == 'EVT3.0'
    paths = E.export(res, p, str(tmp_path))
    names = {os.path.basename(x) for x in paths}
    assert {'event_cameras.json', 'event_cameras.csv', 'event_rate_EV.csv', 'tunnel_EV.csv', 'events_EV.png',
            'time_surface_EV.png'} <= names
    assert E.to_csv(res).splitlines()[0].split(',') == list(E.CSV_COLUMNS)
    # a variant without an event camera: a virtual one beside F; a cancel returns None
    from vantage.core import samples
    q = samples.tigor_sample_project()
    q.active_uid = next(v for v in q.variants if v.sensors).uid
    res2 = E.run(q, Engine(), st)
    assert res2.cams[0]['virtual'] and res2.cams[0]['tag'] == 'F-EV'
    assert any(f['kind'] == 'virtual' for f in res2.findings)
    assert E.run(q, Engine(), st, cancel=lambda: True) is None


def test_edge_spread_follows_the_mismatch():
    m = L.builtin_module('prophesee_evk4_hd')
    sa = E.EventSettings(sigma_ln=0.0, photoreceptor=False, edge_contrast=0.6)
    sb = E.EventSettings(sigma_ln=0.05, photoreceptor=False, edge_contrast=0.6)
    a = E.edge_events(E.params_of(m, sa), sa, 2000.0, 4000.0, 2.0)
    b = E.edge_events(E.params_of(m, sb), sb, 2000.0, 4000.0, 2.0)
    assert a['spread_px'] <= 0.01 and b['spread_px'] > 5 * max(a['spread_px'], 0.005)
    # ideal pixels: the first event when the log signal rises θ = ln 1.25 on a 1:2.5 edge, at the covered share
    # (1.25·Ld − Ld) / (Lb − Ld) of the pixel; the edge's centre crosses the pixel's centre at half
    cover = (1.25 - 1) / (2.5 - 1)
    assert a['lag_px'] == pytest.approx(cover - 0.5, abs=0.02)


# ======================================================================= the page
from test_gui import app, win, pump, _wait     # noqa: E402,F401  (fixtures)


def test_page_runs_edits_loads_a_drive_and_exports(app, win, tmp_path, dvs_run):
    from vantage.ui.workspaces.event_ws import EventPage
    store = win.store
    rig_v = next(v for v in store.project.variants if v.name.startswith('Data rig'))
    store.edit('variants', lambda pj: setattr(pj, 'active_uid', rig_v.uid), f'Switched to {rig_v.name}')
    store.edit('analysis', lambda pj: setattr(pj.analysis, 'event_camera', dict(FAST)), 'Fast event study')
    pump(app, 0.1)
    pg = EventPage(win.ctx)
    pg.resize(1500, 900)
    pg.show()
    _wait(app, lambda: pg.res is not None and not pg.busy, 180)
    pump(app, 0.1)
    res = pg.res
    assert pg.table.rowCount() == 1 and pg.table.item(0, 0).text() == 'F-EV *'
    assert pg.table.item(0, 2).text() == f'{res.cams[0]["rate"]["rate_evs"] / 1e6:.1f}'
    assert len(pg.findings) == len(res.findings) and pg.previews.items
    assert 'virtual' in pg.status.text() and 'F-EV' in pg.head.text()
    pg.view.buttons['rate'].click()
    pump(app, 0.1)
    assert pg.rate_chart.isVisible() and not pg.tunnel_chart.isVisible() and 'Mev/s' in pg.chart_note.text()
    # a setting: saved as one undo step, re-run by itself, undone
    pg.fmt.buttons['EVT2.0'].click()
    pump(app, 0.05)
    assert store.project.analysis.event_camera['event_format'] == 'EVT2.0'
    _wait(app, lambda: pg.res is not res and not pg.busy and not pg.stale, 180)
    assert pg.res.cams[0]['rate']['format'] == 'EVT2.0'
    store.undo.undo()
    assert store.project.analysis.event_camera.get('event_format', 'EVT3.0') == 'EVT3.0'
    assert pg.fmt.value() == 'EVT3.0'
    _wait(app, lambda: not pg.busy and not pg.stale, 180)
    pg.set_option('speed_kmh', 80.0)
    assert store.project.analysis.event_camera['speed_kmh'] == 80.0 and pg.fields['speed_kmh'][0].value() == 80.0
    store.undo.undo()
    _wait(app, lambda: not pg.busy and not pg.stale, 180)
    # cancel
    pg.run()
    assert pg.busy
    pg.cancel()
    assert not pg.busy and 'Cancelled' in pg.status.text()
    pump(app, 0.3)
    # a CARLA drive with event cameras
    assert not pg.load_drive(str(tmp_path))
    assert pg.load_drive(dvs_run['out'])
    assert 'matched exactly' in pg.drive_text.text() and '100.0 %' in pg.drive_text.text()
    paths = pg.export(str(tmp_path / 'exp'))
    assert any(x.endswith('event_cameras.json') for x in paths)
    pg.close()
    pg.deleteLater()
    pump(app, 0.1)


# ======================================================================= real CARLA (round 17)
@pytest.mark.skipif(not os.path.isfile(os.path.join(REAL_DRIVE, 'events', 'index.json')),
                    reason='needs the round-17 CARLA drive with an event camera')
def test_real_carla_dvs_against_the_model():
    idx = E.load_recording(REAL_DRIVE)
    ch = idx['channels'][0]
    assert ch['time_check']['on_simulation_clock']
    c = ch['compare']
    assert c['carla_events'] > 10000 and c['count_error'] < 0.02 and c['pixel_count_r'] > 0.98
    rows = E.recording_summary(idx)
    assert rows[0]['rate_evs'] > 0
