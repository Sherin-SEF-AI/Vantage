"""Recommended actions in the assistant (vantage/assistant/recommend.py and the panel): every rule on a state that
triggers it and one that does not, with the figures equal to the app's own results; the ranking by what the user is
doing now (the selection, the last edit with the analysis before and after it, the page, a job that finished), the
deduplication, stable ordering, dismissal coming back on a material change, the model's suggest_next items, the
speed; and the panel: the empty state's list, the header chip, the live strip following the selection and an edit,
Ask, Plan it (a plan card that is not approved), Open (a page, nothing run), the Next steps under an answer (Groq
here; Claude where the SDK is installed), and that computing the recommendations sends nothing."""
import copy
import json
import os
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from fake_groq import FakeGroq
from vantage.assistant import agent as A
from vantage.assistant import copilot as CP
from vantage.assistant import recommend as RC
from vantage.assistant import secrets as K
from vantage.assistant import tools as T
from vantage.core import rules, samples
from vantage.core.analysis import Engine
from vantage.core.tolerance import worst_single_camera


@pytest.fixture(autouse=True)
def _quiet_machine(monkeypatch):
    """No CARLA server answers anywhere and the machine has memory to spare (as test_copilot's)."""
    from vantage.sim import server as SV
    monkeypatch.setattr(SV, 'port_open', lambda *a, **k: False)
    monkeypatch.setattr(CP, 'mem_available_gb', lambda: 30.0)


@pytest.fixture(scope='module')
def sample():
    p = samples.tigor_sample_project()
    e = Engine()
    res = e.analyse(p)
    vg = e.vehicle_geom(p)
    mounts = {c.uid: rules.mount_info(p, c, vg) for c in p.variant.cameras}
    iss = rules.collect_issues(p, res, vg, mounts)
    worst = worst_single_camera(e, p, p.analysis.tol_deg, p.analysis.tol_mm / 1000.0, cell=max(p.analysis.cell_m, 0.05))
    return SimpleNamespace(p=p, e=e, res=res, issues=iss, worst=worst)


def _state(s, **kw):
    return RC.State(s.p, s.res, list(s.issues), s.worst, **kw)


def _ids(recs):
    return [r.id for r in recs]


def _by(recs, rid):
    return next((r for r in recs if r.id == rid), None)


def _tag(p, uid):
    return next(c.tag for c in p.variant.cameras if c.uid == uid)


# ============================================================================================ each rule
def test_blind_ground_names_the_largest_patch_and_its_figures_are_the_analysis(sample):
    from scipy import ndimage
    res, p = sample.res, sample.p
    r = _by(RC.ground_rules(_state(sample), RC.Context()), 'ground:blind')
    assert r is not None and r.kind == 'goal' and r.source.startswith('Analysis')
    # independently: the patches of the analysis's own blind cells (4-connected, then merged with diagonals)
    g = res.grid
    mask = g.roi & (res.count == 0)
    lab, n = ndimage.label(mask, structure=np.ones((3, 3)))
    sizes = np.bincount(lab.ravel())[1:] * g.cell ** 2
    assert f'{sizes.max():.3f} m²' in r.title and f'{res.metrics["blind_m2"]:.3f} m²' in r.why
    assert r.evidence == {'blind_m2': round(res.metrics['blind_m2'], 3), 'patch_m2': round(float(sizes.max()), 3)}
    k = int(np.argmax(sizes))
    jj, ii = np.nonzero(lab == k + 1)
    x, y = float(g.xs[ii].mean()), float(g.ys[jj].mean())
    near = min(p.variant.cameras, key=lambda c: np.hypot(c.x - x, c.y - y)).tag
    assert r.title.endswith(f'near {near}') and ('camera', near) in r.about
    assert f'({x:.2f}, {y:.2f})' in r.why
    # no blind ground: no recommendation
    clear = copy.copy(res)
    clear.metrics = dict(res.metrics, blind_m2=0.0)
    clear.count = np.ones_like(res.count)
    assert _by(RC.ground_rules(RC.State(p, clear), RC.Context()), 'ground:blind') is None


def test_minimum_density_on_the_worst_camera_and_on_the_selected_one(sample):
    res, p = sample.res, sample.p
    tags = {c.uid: c.tag for c in p.variant.cameras}
    worst = min(res.per_cam, key=lambda u: res.per_cam[u]['min_dens'])
    recs = RC.ground_rules(_state(sample), RC.Context())
    dens = [r for r in recs if r.id.startswith('density:')]
    assert _ids(dens) == [f'density:{tags[worst]}']                     # only the worst camera without a selection
    d = res.per_cam[worst]
    assert f'{d["min_dens"]:.2f} px/cm' in dens[0].title and f'{d["best_m2"]:.2f} m²' in dens[0].why
    qr = p.variant.by_tag('QR')
    recs = RC.ground_rules(_state(sample), RC.Context(selection=(('camera', 'QR', 'QR'),)))
    q = _by(recs, 'density:QR')
    assert q is not None and f'{res.per_cam[qr.uid]["min_dens"]:.2f} px/cm' in q.title
    # every camera above the target: nothing
    p2 = copy.deepcopy(p)
    p2.analysis.target_density = 1.0
    assert not [r for r in RC.ground_rules(RC.State(p2, res), RC.Context()) if r.id.startswith('density:')]
    # below the minimum is serious, below the target only a warning
    p3 = copy.deepcopy(p)
    p3.analysis.min_density = 10.0
    p3.analysis.target_density = 12.0
    r3 = _by(RC.ground_rules(RC.State(p3, res), RC.Context()), f'density:{tags[worst]}')
    assert r3.base > dens[0].base


def test_a_simulated_pitch_for_the_selected_camera_is_the_engines_own(sample):
    """pitch_hint analyses copies (off the UI thread in the app); the figure it reports is what the engine gives for
    that pitch, checked here by applying it independently; the rule then names it."""
    # the sample as it is: no pitch step helps QR, so the plain recommendation
    assert RC.pitch_hint(sample.p, sample.e, 'QR') is None
    r = _by(RC.ground_rules(_state(sample), RC.Context(selection=(('camera', 'QR', 'QR'),))), 'density:QR')
    assert 'try pitch' not in r.title and r.title.startswith("Raise QR's minimum density")
    # the quarter cameras pitched 10° down: pitching QR back up helps
    p = samples.tigor_sample_project()
    for t in ('QR', 'QL'):
        p.variant.by_tag(t).pitch += 10
    e = Engine()
    res = e.analyse(p)
    sample = SimpleNamespace(p=p, e=e, res=res, issues=[], worst=None)
    qr = p.variant.by_tag('QR')
    p0 = qr.pitch
    t0 = time.perf_counter()
    h = RC.pitch_hint(p, e, 'QR')
    ms = (time.perf_counter() - t0) * 1000
    assert p.variant.by_tag('QR').pitch == p0                           # the project is never changed
    assert h is not None and h['delta'] == -4.0
    assert h['delta'] in RC.PITCH_STEPS and h['min_now'] == pytest.approx(res.per_cam[qr.uid]['min_dens'])
    q = p.snapshot()
    for c in q.variant.cameras:                                          # by hand: QR and its mirrored partner
        if c.uid == qr.uid:
            c.pitch += h['delta']
    part = q.variant.cam(qr.link_uid) if qr.link_uid else None
    if part is not None:
        part.pitch = q.variant.cam(qr.uid).pitch
    r2 = Engine().analyse(q)
    assert r2.per_cam[qr.uid]['min_dens'] == pytest.approx(h['min_dens'], abs=1e-4)
    assert r2.metrics['blind_m2'] == pytest.approx(h['blind_m2'], abs=1e-9)
    assert h['blind_m2'] <= res.metrics['blind_m2'] + 1e-9 and h['min_dens'] > h['min_now']
    st = _state(sample, hints={'pitch:QR': h})
    r = _by(RC.ground_rules(st, RC.Context(selection=(('camera', 'QR', 'QR'),))), 'density:QR')
    assert r.title == (f"QR's minimum density is {h['min_now']:.2f} px/cm (target "
                       f"{p.analysis.target_density:g}): try pitch {h['delta']:+g}°")
    assert f'gives {h["min_dens"]:.2f} px/cm' in r.why and r.kind == 'ask' and 'simulate_changes' in r.prompt
    print(f'\npitch_hint for QR: {h["delta"]:+g}° in {ms:.0f} ms (off the UI thread)')


def test_the_image_scale_that_fits_is_built_and_checked():
    from vantage.core import simrig as SR
    from vantage.sim import carla_bridge as CB
    p = samples.tigor_sample_project()
    p.active_uid = next(v for v in p.variants if v.sensors).uid
    st = dict(SR.settings_of(p).to_dict(), image_scale=1.0,
              ground_truth={'semantic': True, 'instance': True, 'range': True})
    h = RC.scale_hint(p, st, CB.RENDER_BUDGET_MPX)
    assert h is not None and h['for'] == 1.0 and h['scale'] < 1.0 and h['mpx'] <= CB.RENDER_BUDGET_MPX
    up = CP._render_mpx(p, st, round(h['scale'] + 0.05, 2))[0]
    assert up > CB.RENDER_BUDGET_MPX                                      # the largest 5 % step that fits
    carla = {'installed': True, 'port': 2000, 'server_up': True, 'runs': 1, 'last_run': None,
             'render': {'mpx': 150.0, 'gt_mpx': 200.0, 'budget': CB.RENDER_BUDGET_MPX, 'scale': 1.0}}
    r = _by(RC.carla_rules(RC.State(p, carla=carla, hints={'scale': h}), RC.Context(page='sim')), 'carla:budget')
    assert r.title == f'Image scale 1 with ground truth exceeds the render budget: use {h["scale"]:g}'
    assert f'renders {h["mpx"]:.0f} Mpx' in r.why and f'fits at {h["scale"]:g}' in r.prompt


def test_a_seam_where_a_standing_object_vanishes(sample):
    so = SimpleNamespace(settings=SimpleNamespace(height=0.9),
                         seams=[{'tags': ('QR', 'MR'), 'vanish_m2': 0.002}, {'tags': ('ML', 'QL'), 'vanish_m2': 0.118}])
    r = _by(RC.ground_rules(_state(sample, stitched=so), RC.Context()), 'seam:ML|QL')
    assert r is not None and '0.9 m' in r.title and '0.118 m²' in r.title and r.topic == 'vanish'
    adv = SimpleNamespace(changes=[1], vanish_after=0.02)
    r = _by(RC.ground_rules(_state(sample, stitched=so, advice=adv), RC.Context()), 'seam:ML|QL')
    assert 'brings it to 0.020 m²' in r.why
    so.seams[1]['vanish_m2'] = 0.001
    assert not [x for x in RC.ground_rules(_state(sample, stitched=so), RC.Context()) if x.id.startswith('seam:')]


def test_issues_most_severe_first_by_name_and_the_blind_one_left_to_the_ground_rule(sample):
    I = rules.Issue
    iss = [I('warning', 'QL has no GMSL port assigned.', 'QL', [sample.p.variant.by_tag('QL').uid]),
           I('critical', '0.010 m² of ground in the ROI is blind, mostly right rear wheel arch.', 'v'),
           I('serious', 'Routed cable 6.20 m is longer than the 5.00 m module cable. Use the 8 m cable.', 'R', [])]
    recs = RC.issue_rules(RC.State(sample.p, sample.res, iss), RC.Context())
    assert [r.title for r in recs] == ['Fix: Routed cable 6.20 m is longer than the 5.00 m module cable',
                                       'Fix: QL has no GMSL port assigned']
    assert recs[0].base > recs[1].base and recs[0].kind == 'ask' and iss[2].message in recs[0].prompt
    assert ('camera', 'QL') in recs[1].about
    # the sample's own list: every issue but the blind ground, by its message
    recs = RC.issue_rules(_state(sample), RC.Context())
    want = [i.message for i in sample.issues if 'is blind' not in i.message][:4]
    assert [r.why.split(': ', 1)[1] for r in recs] == want[:len(recs)] and recs
    assert not RC.issue_rules(RC.State(sample.p, sample.res, []), RC.Context())


def test_a_camera_whose_mounting_error_moves_coverage(sample):
    p, res = sample.p, sample.res
    b, uid, pert = sample.worst
    r = RC.tolerance_rules(_state(sample), RC.Context())
    tag = _tag(p, uid)
    extra = b - res.metrics['blind_m2']
    assert extra > RC.TOL_EXTRA_M2
    assert [x.id for x in r] == [f'tolerance:{tag}'] and r[0].kind == 'goal'
    assert f'{extra:.3f} m²' in r[0].title and f'{res.metrics["blind_m2"]:.3f} to {b:.3f} m²' in r[0].why
    assert f'{p.analysis.tol_deg:g}° / {p.analysis.tol_mm:g} mm' in r[0].why
    small = RC.State(p, res, worst=(res.metrics['blind_m2'] + 0.001, uid, pert))
    assert not RC.tolerance_rules(small, RC.Context())


def test_requirements_failing_by_name_or_write_them_from_the_brief(sample):
    from vantage.core import requirements as RQ
    p = copy.deepcopy(sample.p)
    r = RC.requirement_rules(RC.State(p, sample.res), RC.Context())
    assert _ids(r) == ['requirements:write'] and r[0].title == 'Write requirements for this rig'
    r = RC.requirement_rules(RC.State(p, sample.res, brief='At least 4 px/cm near the bumpers.'), RC.Context())
    assert r[0].title == 'Write requirements from your brief' and 'brief' in r[0].prompt
    # written: the scorecard's own result, a failing must by its id
    item = {'id': 'R1', 'text': 'Blind ground at most 0.001 m²', 'metric': 'coverage.blind_m2', 'params': {},
            'op': '<=', 'target': 0.001, 'priority': 'must', 'note': ''}
    p.analysis.requirements = {'items': [item]}
    res = RQ.run(p, sample.e, issues_list=False)
    it = res.item('R1')
    assert it.status == 'fail'
    r = RC.requirement_rules(RC.State(p, sample.res, requirements=res), RC.Context())
    assert _ids(r) == ['requirement:R1'] and r[0].base == RC.SEVERITY['serious']
    assert RQ.fmt_value(it.value, it.metric) in r[0].why and RQ.fmt_value(0.001, it.metric) in r[0].why
    item['target'] = 1.0
    p.analysis.requirements = {'items': [item]}
    res = RQ.run(p, sample.e, issues_list=False)
    assert res.item('R1').status == 'pass' and not RC.requirement_rules(RC.State(p, sample.res, requirements=res),
                                                                         RC.Context())


def test_perception_sector_below_its_required_range(sample):
    pr = SimpleNamespace(sectors=[
        {'name': 'front', 'centre_deg': 0, 'required_m': 40.0,
         'classes': {'Pedestrian': {'min_m': 28.4, 'pass': False, 'shortfall_m': 11.6}}},
        {'name': 'rear', 'centre_deg': 180, 'required_m': 10.0,
         'classes': {'Pedestrian': {'min_m': 12.0, 'pass': True, 'shortfall_m': 0.0}}}])
    r = RC.perception_rules(RC.State(sample.p, perception=pr), RC.Context())
    assert _ids(r) == ['perception:front:Pedestrian'] and '28.4 of 40 m' in r[0].title and '11.6 m short' in r[0].why
    pr.sectors[0]['classes']['Pedestrian'].update(pass_=True)
    pr.sectors[0]['classes']['Pedestrian']['pass'] = True
    assert not RC.perception_rules(RC.State(sample.p, perception=pr), RC.Context())


def test_lidar_without_a_calibration_plan():
    p = samples.tigor_sample_project()
    assert not RC.lidar_rules(RC.State(p), RC.Context())                  # the active variant has no lidar
    p.active_uid = next(v for v in p.variants if v.sensors).uid
    r = RC.lidar_rules(RC.State(p), RC.Context())
    assert _ids(r) == ['lidar_calib:plan'] and r[0].kind == 'open' and r[0].target == {'page': 'lidar_calib'}
    assert ('sensor', 'L1') in r[0].about
    assert not RC.lidar_rules(RC.State(p, lidar_calib_done=True), RC.Context())
    p.analysis.lidar_calib = {'camera': 'F', 'lidar': 'L1'}
    assert not RC.lidar_rules(RC.State(p), RC.Context())


def test_stale_analysis(sample):
    r = RC.stale_rules(RC.State(sample.p, stale=True), RC.Context())
    assert _ids(r) == ['analysis:stale'] and r[0].kind == 'open' and r[0].title == 'Update the analysis'
    assert not RC.stale_rules(RC.State(sample.p, stale=False), RC.Context())


def test_carla_never_run_server_budget_unchecked_and_unscored(sample):
    p = sample.p
    base = {'installed': True, 'port': 2300, 'server_up': False, 'runs': 0, 'last_run': None, 'render': None}
    sim = RC.Context(page='sim')
    r = RC.carla_rules(RC.State(p, carla=base), sim)
    assert _ids(r) == ['carla:first', 'carla:server']
    assert r[0].kind == 'plan' and 'propose_plan' in r[0].prompt and 'ground truth' in r[0].prompt
    assert r[1].title == 'No server on port 2300: start it' and r[1].target == {'page': 'sim', 'focus': 'b_start'}
    assert _ids(RC.carla_rules(RC.State(p, carla=base), RC.Context(page='viewports'))) == ['carla:first']
    assert not RC.carla_rules(RC.State(p, carla=dict(base, installed=False)), sim)
    over = dict(base, runs=1, server_up=True, render={'mpx': 120.0, 'gt_mpx': 180.0, 'budget': 200.0, 'scale': 1.0})
    r = RC.carla_rules(RC.State(p, carla=over), sim)
    assert _ids(r) == ['carla:budget'] and r[0].kind == 'plan' and '300 Mpx' in r[0].why and '200' in r[0].why
    assert r[0].title.startswith('Image scale 1 with ground truth is over the render budget')
    fits = dict(over, render={'mpx': 60.0, 'gt_mpx': 90.0, 'budget': 200.0, 'scale': 0.5})
    assert not RC.carla_rules(RC.State(p, carla=fits), sim)
    last = {'path': '/runs/20261029-101500', 'name': '20261029-101500', 'status': 'ok', 'ground_truth': True,
            'checked': False, 'scored': False}
    r = RC.carla_rules(RC.State(p, carla=dict(fits, last_run=last)), sim)
    assert _ids(r) == ['carla:check:20261029-101500', 'carla:score:20261029-101500']
    assert r[0].kind == r[1].kind == 'open'
    assert r[0].target == {'page': 'sim', 'run': last['path'], 'focus': 'b_drivelog'}
    assert r[1].target['focus'] == 'b_detect'
    done = dict(last, checked=True, scored=True)
    assert not RC.carla_rules(RC.State(p, carla=dict(fits, last_run=done)), sim)
    assert not [x for x in RC.carla_rules(RC.State(p, carla=dict(fits, last_run=dict(last, ground_truth=False))),
                                          sim) if x.id.startswith('carla:score')]
    bad = dict(last, status='error', error='The bridge refused: 306 Mpx')
    r = RC.carla_rules(RC.State(p, carla=dict(fits, last_run=bad)), sim)
    assert _ids(r) == ['carla:failed:20261029-101500'] and '306 Mpx' in r[0].prompt


def test_drive_log_check_a_rotated_camera_and_serious_findings(sample):
    dl = SimpleNamespace(extrinsics=[{'tag': 'ML', 'verdict': 'rotated', 'delta_deg': [0.3, -0.2, 0.1],
                                      'confidence': 'high'},
                                     {'tag': 'F', 'verdict': 'ok', 'delta_deg': [0.01, 0.0, 0.0]}],
                         findings=[{'severity': 'serious', 'message': 'QR dropped 12 frames (1.2 %).'},
                                   {'severity': 'ok', 'message': 'All streams present.'}])
    r = RC.drivelog_rules(RC.State(sample.p, drivelog=dl), RC.Context())
    ang = float(np.linalg.norm([0.3, -0.2, 0.1]))
    assert _ids(r)[0] == 'drivelog:rotated:ML' and r[0].title == f'Ask why ML is off by {ang:.2f}°'
    assert 'yaw +0.30°, pitch -0.20°, roll +0.10°' in r[0].why and ('job', 'drivelog') in r[0].about
    assert len(r) == 2 and 'QR dropped 12 frames' in r[1].why
    dl.extrinsics[0]['verdict'] = 'ok'
    dl.findings = []
    assert not RC.drivelog_rules(RC.State(sample.p, drivelog=dl), RC.Context())


def _det(rows, min_gt=5):
    return SimpleNamespace(settings=SimpleNamespace(min_gt=min_gt), rows=rows,
                           recordings=[{'label': 'drive-1', 'weather': 'ClearSunset', 'town': 'Town10HD'}])


def test_detection_weak_cell_becomes_a_plan(sample):
    row = lambda cls, b, rec, n, cam='all': {'rec': 0, 'recording': 'drive-1', 'mode': 'fisheye', 'camera': cam,
                                             'class': cls, 'range_bin': b, 'recall': rec, 'n_gt': n}
    rows = [row('person', 'all', 0.7, 80), row('person', '20–40 m', 0.31, 24), row('person', '40+ m', 0.1, 3),
            row('car', '20–40 m', 0.45, 6), row('person', '20–40 m', 0.05, 30, 'F')]
    r = RC.detection_rules(RC.State(sample.p, detection=_det(rows)), RC.Context())
    assert _ids(r) == ['detection:weak:person:20–40 m'] and r[0].kind == 'plan'
    assert r[0].title == 'Generate data where recall is weak: person 20–40 m in ClearSunset'
    assert 'recall 0.31 on 24 person objects' in r[0].why and 'propose_plan' in r[0].prompt
    rows[1]['recall'] = 0.6
    rows[3]['recall'] = 0.55
    assert not RC.detection_rules(RC.State(sample.p, detection=_det(rows)), RC.Context())   # 40+ m: too few objects


def test_campaign_missing_the_users_target_and_false_alarms(sample):
    det = {'thresholds': [{'check': 'drivelog.extrinsics', 'tag': 'F', 'kind': 'rotation', 'axis': 'any', 'unit': '°',
                           'empirical': None, 'target': 0.9, 'tested': [0.1, 0.5, 1.0]},
                          {'check': 'drivelog.extrinsics', 'tag': 'ML', 'kind': 'rotation', 'axis': 'any',
                           'unit': '°', 'empirical': 0.5, 'target': 0.9, 'tested': [0.1, 0.5, 1.0]}],
           'false_alarms': [{'check': 'drivelog.sync', 'tag': 'QR', 'k': 2, 'n': 6, 'applicable': True},
                            {'check': 'asbuilt.pose', 'tag': 'F', 'k': 6, 'n': 6, 'applicable': False}]}
    r = RC.campaign_rules(RC.State(sample.p, campaign=SimpleNamespace(detectability=det)), RC.Context())
    assert _ids(r) == ['campaign:miss:drivelog.extrinsics:F:rotation', 'campaign:false:drivelog.sync:QR']
    assert 'up to 1 °' in r[0].title and 'your 90 % detection target' in r[0].why
    assert 'on 2 of 6 clean drives' in r[1].title
    det['thresholds'][0]['empirical'] = 0.8
    det['false_alarms'][0]['k'] = 0
    assert not RC.campaign_rules(RC.State(sample.p, campaign=SimpleNamespace(detectability=det)), RC.Context())


def test_an_unhealthy_live_stream(sample):
    h = {'cameras': {'F': {'status': 'ok', 'reasons': []},
                     'QR': {'status': 'fault', 'reasons': ['no frames for 3.2 s from rtsp://admin:pw@10.0.0.5/s']}}}
    r = RC.rig_rules(RC.State(sample.p, rig_health=h), RC.Context())
    assert _ids(r) == ['rig:QR'] and 'no frames for 3.2 s' in r[0].why
    assert 'rtsp://' not in r[0].why and 'pw@' not in r[0].prompt                 # a stream URL never shows
    h['cameras']['QR']['status'] = 'ok'
    assert not RC.rig_rules(RC.State(sample.p, rig_health=h), RC.Context())


def _draft(warnings=(), carla_s=900.0, scale=1.0, dur=20.0):
    st = CP.Step('carla_run', 'Dusk drive', {'duration_s': dur, 'image_scale': scale, 'weather': 'SoftRainSunset'},
                 warnings=list(warnings), estimate={'carla_s': carla_s, 'text': f'{dur:g} s drive'})
    return CP.Plan('P7', 2, 'Dusk drive in rain', steps=[st])


def test_copilot_draft_waiting_its_revisions_and_a_failed_run():
    p = samples.tigor_sample_project()
    pl = _draft(['MemAvailable is 6.0 GB; the run needs about 9.0 GB with the server.',
                 'CARLA\'s lidar and radar ignore rain and fog: they show in the camera images only.'])
    r = RC.plan_rules(RC.State(p, plans=[pl]), RC.Context())
    assert _ids(r) == ['plan:review:P7', 'plan:rev:P7:0:duration', 'plan:rev:P7:0:scale', 'plan:rev:P7:0:clear']
    assert r[0].kind == 'open' and r[0].target == {'plan': 'P7', 'version': 2}
    assert r[1].title == 'Halve the duration: 20 → 10 s' and r[1].kind == 'plan' and '10 s' in r[1].prompt
    assert r[2].title == 'Use half-size images: scale 1 → 0.5' and 'MemAvailable' in r[2].why
    quiet = _draft([], carla_s=60.0)
    assert _ids(RC.plan_rules(RC.State(p, plans=[quiet]), RC.Context())) == ['plan:review:P7']
    quiet.status = 'approved'
    assert not RC.plan_rules(RC.State(p, plans=[quiet]), RC.Context())
    rec = {'id': 'P3', 'version': 1, 'title': 'Short drive', 'runs': [
        {'status': 'failed', 'finished_utc': '2026-10-29T10:00:00Z',
         'steps': [{'status': 'failed', 'message': 'No CARLA server answers on port 2400.'}]}]}
    r = RC.plan_rules(RC.State(p, records=[rec]), RC.Context())
    assert _ids(r) == ['plan:failed:P3:1'] and r[0].title == 'See why plan P3 failed'
    assert 'step 1: No CARLA server answers on port 2400.' in r[0].why and r[0].kind == 'ask'
    rec['runs'][0]['status'] = 'done'
    assert not RC.plan_rules(RC.State(p, records=[rec]), RC.Context())


def _edit(before_blind, after_blind, idx=4, text='Changed F', before_min=3.2, after_min=3.2):
    return {'index': idx, 'text': text, 'before': {'blind_m2': before_blind, 'min_density': before_min,
                                                   'min_ok_pct': 99.0},
            'after': {'blind_m2': after_blind, 'min_density': after_min, 'min_ok_pct': 98.0}}


def test_the_last_edit_before_and_after(sample):
    cx = RC.Context(edits=[_edit(0.18, 0.24)], undo_index=4)
    r = RC.edit_rules(_state(sample), cx)
    assert _ids(r) == ['edit:4:undo', 'edit:4:why']
    assert r[0].title == 'Your last change added 0.060 m² blind ground: undo it' and r[0].kind == 'open'
    assert r[0].target['undo'] == 4 and '0.180 → 0.240 m²' in r[0].why
    assert r[1].kind == 'ask' and 'Changed F' in r[1].prompt and '0.180 → 0.240 m²' in r[1].prompt
    assert not RC.edit_rules(_state(sample), RC.Context(edits=[_edit(0.18, 0.24)], undo_index=3))   # undone
    assert not RC.edit_rules(_state(sample), RC.Context(edits=[_edit(0.18, 0.182)], undo_index=4))   # too small
    assert not RC.edit_rules(_state(sample), RC.Context(edits=[_edit(0.24, 0.18)], undo_index=4))    # better
    low = RC.Context(edits=[_edit(0.18, 0.18, before_min=3.2, after_min=2.6)], undo_index=4)
    r = RC.edit_rules(_state(sample), low)
    assert r[0].title == 'Your last change lowered the minimum density to 2.60 px/cm: undo it'


def test_generic_starters_only_when_little_is_wrong(sample):
    p = samples.tigor_sample_project()
    p.analysis.requirements = {'items': [{'id': 'R1', 'text': 'x', 'metric': 'coverage.blind_m2', 'params': {},
                                          'op': '<=', 'target': 1, 'priority': 'must', 'note': ''}]}
    recs = RC.recommend(RC.State(p), RC.Context())
    assert _ids(recs) == ['start:summary', 'start:child', 'start:review']
    assert recs[1].title == 'Where can a child go unseen?'
    assert not [r for r in RC.recommend(_state(sample), RC.Context()) if r.id.startswith('start:')]


# ============================================================================================ ranking
def test_the_current_thing_ranks_first(sample):
    st = _state(sample)
    plain = RC.recommend(st, RC.Context())
    assert plain[0].id == 'ground:blind'                                 # a critical finding leads without a context
    sel = RC.recommend(st, RC.Context(selection=(('camera', 'QR', 'QR'),)))
    assert set(_ids(sel[:2])) == {'density:QR', 'unique:QR'}
    cx = RC.Context(selection=(('camera', 'QR', 'QR'),))
    assert [r.id for r in RC.context_items(sel, cx)][:2] == _ids(sel[:2])
    assert RC.context_label(RC.Context(selection=(('camera', 'QR', 'QR'),), page_label='Coverage')) == \
        'For QR · Coverage'
    edit = RC.Context(edits=[_edit(0.18, 0.24)], undo_index=4)
    assert _ids(RC.recommend(st, edit)[:2]) == ['edit:4:undo', 'edit:4:why']
    assert RC.context_label(edit) == 'After: Changed F'
    carla = {'installed': True, 'port': 2000, 'server_up': False, 'runs': 3, 'last_run': None, 'render': None}
    page = RC.recommend(_state(sample, carla=carla), RC.Context(page='sim', page_label='CARLA simulator'))
    assert page[0].id == 'carla:server'
    dl = SimpleNamespace(extrinsics=[{'tag': 'ML', 'verdict': 'rotated', 'delta_deg': [0.4, 0, 0]}], findings=[])
    job = RC.Context(job={'kind': 'drivelog', 'ok': True, 't': time.time()}, now=time.time())
    assert RC.recommend(_state(sample, drivelog=dl), job)[0].id == 'drivelog:rotated:ML'
    old = RC.Context(job={'kind': 'drivelog', 'ok': True, 't': time.time() - 2 * RC.JOB_RECENT_S}, now=time.time())
    assert RC.recommend(_state(sample, drivelog=dl), old)[0].id == 'ground:blind'    # no longer current
    # the strip: only items about the current thing, at most three
    assert len(RC.context_items(sel, RC.Context(selection=(('camera', 'QR', 'QR'),)))) <= RC.STRIP
    assert not RC.context_items(plain, RC.Context(page='fleet'))


def test_dedupe_by_id_and_topic_and_stable_order(sample):
    st = _state(sample)
    p = sample.p
    worst = min(sample.res.per_cam, key=lambda u: sample.res.per_cam[u]['min_dens'])
    tag = _tag(p, worst)
    recs = RC.recommend(st, RC.Context(selection=(('camera', tag, tag),)))
    assert len(_ids(recs)) == len(set(_ids(recs))) and _ids(recs).count(f'density:{tag}') == 1
    so = SimpleNamespace(settings=SimpleNamespace(height=0.9), seams=[{'tags': ('ML', 'QL'), 'vanish_m2': 0.12}])
    iss = list(sample.issues) + [rules.Issue('warning', 'A 0.9 m child can vanish from the stitched bird\'s-eye view '
                                                        'on 0.120 m², mostly at the ML | QL seam.', 'ML | QL')]
    recs = RC.recommend(RC.State(p, sample.res, iss, sample.worst, stitched=so), RC.Context())
    vanish = [r for r in recs if r.topic == 'vanish']
    assert _ids(vanish) == ['seam:ML|QL']                                # the specific one kept, the issue dropped
    # stable: the order shown last time wins within a severity step; a priority change across a step reorders
    a = RC.Recommendation('a', 'A', '', 'ask', priority=55.0)
    b = RC.Recommendation('b', 'B', '', 'ask', priority=58.0)
    c = RC.Recommendation('c', 'C', '', 'ask', priority=95.0)
    assert _ids(RC.stable([a, b, c], None)) == ['c', 'b', 'a']
    assert _ids(RC.stable([a, b, c], ['c', 'a', 'b'])) == ['c', 'a', 'b']
    a.priority = 99.0
    assert _ids(RC.stable([a, b, c], ['c', 'a', 'b'])) == ['c', 'a', 'b']
    a.priority = 130.0
    assert _ids(RC.stable([a, b, c], ['c', 'b', 'a']))[0] == 'a'
    # deterministic: the same state gives the same list
    assert [r.to_dict() for r in RC.recommend(st, RC.Context())] == [r.to_dict() for r in RC.recommend(st,
                                                                                                         RC.Context())]


def test_dismissed_until_the_evidence_changes_materially(sample):
    st = _state(sample)
    d = RC.Dismissed()
    first = RC.recommend(st, RC.Context(), d)
    d.dismiss(first[0])
    again = RC.recommend(st, RC.Context(), d)
    assert first[0].id not in _ids(again) and len(again) == len(first) - 1
    d2 = RC.Dismissed(json.loads(json.dumps(d.data)))                    # as kept in QSettings
    assert first[0].id not in _ids(RC.recommend(st, RC.Context(), d2))
    # the same patch 10 % bigger stays dismissed; 50 % bigger comes back (and stays back)
    r0 = first[0]
    assert r0.id == 'ground:blind'
    small, big = dict(r0.evidence), dict(r0.evidence)
    small['blind_m2'] = round(small['blind_m2'] * 1.1, 4)
    big['blind_m2'] = round(big['blind_m2'] * 1.5, 4)
    assert not RC.material(r0.evidence, small) and RC.material(r0.evidence, big)
    assert RC.material({'status': 'fail'}, {'status': 'pass'}) and RC.material({'a': 1}, {'a': 1, 'b': 2})
    assert not RC.material({'run': 'x', 'n': 0.0}, {'run': 'x', 'n': 0.0005})
    res2 = copy.copy(sample.res)
    res2.metrics = dict(sample.res.metrics, blind_m2=sample.res.metrics['blind_m2'] * 1.6)
    st2 = RC.State(sample.p, res2, list(sample.issues), sample.worst)
    every = RC.recommend(st2, RC.Context())
    d.prune(every)
    assert r0.id not in d.data and r0.id in _ids(RC.recommend(st2, RC.Context(), d))


def test_suggest_next_items_are_validated_deduplicated_and_never_run():
    local = [RC.Recommendation('density:QR', 'Raise QR\'s minimum density', '', 'ask', prompt='How can I raise QR?')]
    items = [{'label': 'Check QR\'s tolerance', 'prompt': 'Run the mounting tolerance study for QR.', 'kind': 'ask'},
             {'label': 'raise qr s minimum density', 'prompt': 'x', 'kind': 'ask'},           # the local one
             {'label': 'Check QR\'s tolerance', 'prompt': 'again', 'kind': 'ask'},           # a duplicate
             {'label': '', 'prompt': 'no label', 'kind': 'ask'},
             {'label': 'Drive it', 'prompt': 'a 10 s drive in Town10 at dusk', 'kind': 'plan'},
             {'label': 'Fix it all', 'prompt': 'Get the blind ground to zero.', 'kind': 'run_now'},
             {'label': 'Fourth', 'prompt': 'Fourth one.', 'kind': 'goal'}, 'junk']
    out = RC.clean_next(items, local)
    assert [r.title for r in out] == ['Check QR\'s tolerance', 'Drive it', 'Fix it all']
    assert [r.kind for r in out] == ['ask', 'plan', 'ask']                # an unknown kind is a question
    assert out[1].prompt.startswith('Plan this for my approval with propose_plan:')
    assert len(RC.clean_next([{'label': 'L' * 200, 'prompt': 'P' * 900, 'kind': 'ask'}])[0].title) == 60
    assert not RC.clean_next('nope') and not RC.clean_next([])
    # the tool: shown through the callback, an error without items, and nothing else happens
    shown = []
    tc = T.ToolContext(samples.tigor_sample_project(), None, None, on_next=shown.append)
    ok = T.suggest_next(tc, items=items[:1])
    assert ok['shown'] == 1 and [r.title for r in shown[0]] == ['Check QR\'s tolerance']
    assert 'error' in T.suggest_next(tc, items=[])
    assert T.BY_NAME['suggest_next'].kind == 'ui' and T.validate_args('suggest_next', {'items': items[:1]}) == []
    assert T.validate_args('suggest_next', {'items': [{'label': 'x', 'prompt': 'y', 'kind': 'start_drive'}]})
    assert 'suggest_next' in A.SYSTEM and 'NEXT STEPS' in A.SYSTEM


def test_recommend_is_fast_on_the_sample(sample, capsys):
    """Under 50 ms on the sample: the whole list, and with a context (a selection, an edit, a job, the CARLA page)."""
    st = _state(sample, carla={'installed': True, 'port': 2000, 'server_up': False, 'runs': 2, 'render':
                               {'mpx': 120.0, 'gt_mpx': 180.0, 'budget': 200.0, 'scale': 1.0},
                               'last_run': {'path': '/r/1', 'name': '1', 'status': 'ok', 'ground_truth': True,
                                            'checked': False, 'scored': False}},
                stitched=SimpleNamespace(settings=SimpleNamespace(height=0.9),
                                         seams=[{'tags': ('ML', 'QL'), 'vanish_m2': 0.12}]))
    cx = RC.Context(page='sim', selection=(('camera', 'QR', 'QR'),), edits=[_edit(0.01, 0.07)], undo_index=4,
                    job={'kind': 'carla_run', 'ok': True, 't': time.time()}, now=time.time())
    RC.recommend(st, cx)                                                 # imports warmed
    times = {}
    for name, c in (('plain', RC.Context()), ('context', cx)):
        ts = []
        for _ in range(20):
            t0 = time.perf_counter()
            recs = RC.recommend(st, c)
            ts.append((time.perf_counter() - t0) * 1000)
        times[name] = (float(np.median(ts)), max(ts))
        assert recs
    with capsys.disabled():
        print(f'\nrecommend() on the sample: plain median {times["plain"][0]:.2f} ms (max {times["plain"][1]:.2f}); '
              f'with a context median {times["context"][0]:.2f} ms (max {times["context"][1]:.2f})')
    assert times['plain'][0] < 50 and times['context'][0] < 50


# ============================================================================================ the panel
from test_gui import app, win, pump, settle, _wait  # noqa: E402,F401  (fixtures)


@pytest.fixture()
def groq(monkeypatch):
    s = FakeGroq().start()
    monkeypatch.setenv('VANTAGE_ASSISTANT_URL', s.url)
    yield s
    s.stop()


@pytest.fixture()
def work(win, tmp_path):
    from vantage import workdir
    old = win.ctx.settings.value(workdir.KEY, '')
    workdir.put(win.ctx.settings, str(tmp_path / 'work'))
    yield str(tmp_path / 'work')
    win.ctx.settings.setValue(workdir.KEY, old)
    win.ctx.settings.remove('sim/carla_root')


def _ready(win, provider='groq', key='test-key'):
    st = win.ctx.settings
    K.save(key, provider)
    st.setValue('assistant/enabled', True)
    st.setValue('assistant/provider', provider)
    st.setValue(f'assistant/consent_{provider}', True)
    st.setValue('assistant/model', 'openai/gpt-oss-120b')
    st.setValue('assistant/claude_model', 'claude-opus-5-5')
    st.setValue('assistant/daily_cap', 0)
    win.ctx.assistant.reload()


def _idle(app, win, timeout=60):
    t = time.time()
    while win.ctx.assistant.busy and time.time() - t < timeout:
        pump(app, 0.05)
    pump(app, 0.2)


def _refresh(app, panel):
    pump(app, 0.05)
    panel.recommender.timer.stop()
    panel.recommender.refresh()
    pump(app, 0.05)


def _fake_carla(tmp_path):
    from vantage.sim import server as SV
    root = tmp_path / 'carla'
    root.mkdir(exist_ok=True)
    (root / SV.BINARY).parent.mkdir(parents=True, exist_ok=True)
    (root / SV.BINARY).write_text('#!/bin/sh\n')
    return str(root)


def test_empty_state_lists_recommendations_with_their_buttons_and_the_header_chip(app, win, groq, work):
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest
    from vantage.ui.assistant import RecList, RecRow
    _ready(win)
    win.show_assistant()
    panel = win.assistant_panel
    _refresh(app, panel)
    rc = panel.recommender
    assert rc.recs and rc.ms['recommend'] < 50
    assert isinstance(panel.empty_list, RecList) and panel.empty_list.isVisible()
    below = panel._below_strip()                                       # what the strip shows is not repeated
    strip = set(panel.strip.list.rows)
    assert _ids(below) == [r.id for r in rc.recs if r.id not in strip] and strip
    rows = [panel.empty_list.rows[r.id] for r in below[:RC.SHOWN]]
    assert [r.title.text() for r in rows] == [r.title for r in below[:RC.SHOWN]]
    assert all(r.b.text() == RC.BUTTON[r.rec.kind] for r in rows)
    assert {r.rec.kind for r in rows} >= {'goal', 'ask'}
    assert rows[0].why.text() and rows[0].ico.pixmap() is not None
    more = len(below) - RC.SHOWN
    assert panel.empty_list.more.isVisible() == (more > 0)
    # the header chip opens every one, any time
    assert panel.b_recs.text() == f'Recommended ({len(rc.recs)})'
    QTest.mouseClick(panel.b_recs, Qt.MouseButton.LeftButton)
    pump(app, 0.1)
    pop = panel.popup
    assert pop.isVisible() and [w.rec.id for w in pop.findChildren(RecRow) if w.isVisible()] == \
        _ids(rc.recs[:RC.SHOWN])
    pop.hide()
    # dismiss: gone from the list, kept for this project file only (QSettings, not the project)
    first = below[0]
    QTest.mouseClick(panel.empty_list.rows[first.id].x, Qt.MouseButton.LeftButton)
    pump(app, 0.1)
    assert first.id not in _ids(rc.recs) and first.id not in panel.empty_list.rows
    assert first.id in json.loads(win.ctx.settings.value(rc._key()))
    assert 'dismiss' not in json.dumps(win.store.project.analysis.assistant or {})
    rc.dismissed.data.clear()
    rc._save_dismissed()
    # computing all this sent nothing
    assert not groq.raw and not groq.requests


def test_ask_sends_the_prompt_and_goal_starts_agent_mode(app, win, groq, work):
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest
    _ready(win)
    win.show_assistant()
    panel = win.assistant_panel
    _refresh(app, panel)
    rec = next(r for r in panel._below_strip() if r.kind == 'ask')
    groq.script.append(groq.answer('Here is why.'))
    QTest.mouseClick(panel.empty_list.rows[rec.id].b, Qt.MouseButton.LeftButton)
    _idle(app, win)
    assert len(groq.requests) == 1
    user = groq.requests[0]['messages'][-1]['content']
    assert user.startswith(A.mask(rec.prompt)) and '[Agent mode' not in user
    assert 'Here is why' in win.ctx.assistant.transcript()
    win.ctx.assistant.new_chat()
    _refresh(app, panel)
    goal = next(r for r in panel.recommender.recs if r.kind == 'goal')
    groq.script.append(groq.answer('Done.'))
    panel.run_cta(goal)
    _idle(app, win)
    assert '[Agent mode' in groq.requests[-1]['messages'][-1]['content']
    assert 'AGENT MODE' in groq.requests[-1]['messages'][0]['content']


def test_plan_it_gives_a_plan_card_that_is_not_approved(app, win, groq, work, tmp_path):
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest
    from test_copilot import DRIVE, _propose
    from vantage.ui.assistant import CopilotCard
    win.ctx.settings.setValue('sim/carla_root', _fake_carla(tmp_path))
    _ready(win)
    win.show_assistant()
    panel = win.assistant_panel
    _refresh(app, panel)
    rec = next(r for r in panel._below_strip() if r.id == 'carla:first')
    if rec.id not in panel.empty_list.rows:
        panel.empty_list.more.click()                                   # the rest are one click away
    assert rec.kind == 'plan' and panel.empty_list.rows[rec.id].b.text() == 'Plan it'
    groq.script += [groq.tool_call('propose_plan', _propose([{'kind': 'carla_run', 'label': 'Drive',
                                                              'values': DRIVE}], 'A drive with ground truth'), 'c1'),
                    groq.answer('The plan is on the card; nothing runs until you approve it.')]
    QTest.mouseClick(panel.empty_list.rows[rec.id].b, Qt.MouseButton.LeftButton)
    _idle(app, win)
    assert 'propose_plan' in groq.requests[0]['messages'][-1]['content']
    card = panel.findChildren(CopilotCard)[-1]
    assert card.plan.status == 'draft' and card.b_approve.isVisible()
    assert not CP.records(win.store.project)                             # nothing approved, nothing recorded
    assert not win.page('sim').busy and win.ctx.assistant.runner is None
    # the waiting plan is now a recommendation that opens its card
    _refresh(app, panel)
    rv = next(r for r in panel.recommender.recs if r.id == f'plan:review:{card.plan.id}')
    assert rv.kind == 'open' and panel.open_target(rv.target)
    assert ('plan', card.plan.id, f'plan {card.plan.id}') in panel.recommender.context.selection


def test_open_switches_the_page_and_runs_nothing(app, win, groq, work):
    v = next(v for v in win.store.project.variants if v.sensors)

    def fn(p):
        p.active_uid = v.uid
    win.store.edit(['variants'], fn, 'data rig')
    settle(app, win)
    _ready(win)
    win.show_assistant()
    panel = win.assistant_panel
    _refresh(app, panel)
    rec = next(r for r in panel.recommender.recs if r.id == 'lidar_calib:plan')
    n, rev = win.store.undo.index(), win.store.revision
    panel.run_cta(rec)
    pump(app, 0.2)
    pg = win.page('lidar_calib')
    assert win.page_name() == 'lidar_calib' and pg.result is None          # no calibration was run
    assert win.store.undo.index() == n and win.store.revision == rev and not groq.raw
    # a stale analysis: Open points at Run now and runs nothing
    stale = RC.Recommendation('analysis:stale', 'Update the analysis', '', 'open',
                              target={'page': 'viewports', 'point': 'analysis'})
    an = win.ctx.analysis
    started = []
    an.started.connect(lambda: started.append(1))
    panel.run_cta(stale)
    pump(app, 0.3)
    assert win.page_name() == 'viewports' and not started


def test_the_live_strip_follows_the_selection_and_an_edit_and_undo_it_only_shows(app, win, groq, work):
    _ready(win)
    win.show_assistant()
    panel, st = win.assistant_panel, win.store
    _refresh(app, panel)
    qr = st.variant.by_tag('QR')
    st.select_cams([qr.uid])
    t0 = time.time()
    while 'For QR' not in panel.strip.head.text() and time.time() - t0 < 5:
        pump(app, 0.05)
    assert panel.strip.isVisible() and panel.strip.head.text() == 'For QR · 3D and top views'
    assert set(panel.strip.list.rows) <= {r.id for r in panel.recommender.recs}
    assert 'unique:QR' in panel.strip.list.rows and len(panel.strip.list.rows) <= RC.STRIP
    t0 = time.time()                                                     # the simulated pitch arrives off-thread
    while panel.recommender._hint_busy is not None and time.time() - t0 < 30:
        pump(app, 0.1)
    _refresh(app, panel)
    d = next(r for r in panel.recommender.recs if r.id == 'density:QR')
    key = ('pitch', 'QR', st.revision, st.project.active_uid)
    assert key in panel.recommender.hints
    assert ('try pitch' in d.title) == (panel.recommender.hints[key] is not None)
    # an edit that adds blind ground: the analysis before and after it
    st.select_cams([])
    f = st.variant.by_tag('F')
    st.set_cam(f.uid, yaw=f.yaw + 40.0)
    settle(app, win)
    t0 = time.time()
    while not any(k.endswith(':undo') for k in panel.strip.list.rows) and time.time() - t0 < 10:
        pump(app, 0.1)
    e = panel.recommender.tracker.edits[-1]
    assert e['text'].startswith('Changed F') and e['before'] and e['after']
    assert e['after']['blind_m2'] == pytest.approx(float(win.ctx.analysis.result.metrics['blind_m2']))
    added = e['after']['blind_m2'] - e['before']['blind_m2']
    assert added > RC.EDIT_BLIND_M2
    uid = f'edit:{e["index"]}:undo'
    row = panel.strip.list.rows[uid]
    assert row.rec.title == f'Your last change added {added:.3f} m² blind ground: undo it'
    assert panel.strip.head.text().startswith('After: Changed F')
    n = st.undo.index()
    yaw = st.variant.by_tag('F').yaw
    row.b.click()
    pump(app, 0.2)
    assert st.undo.index() == n and st.variant.by_tag('F').yaw == pytest.approx(yaw)      # shown, not undone
    assert win.docks['bottom'].isVisible() and win.bottom.keys[win.bottom.tabs.currentIndex()] == 'history'
    st.undo.undo()
    settle(app, win)
    _refresh(app, panel)
    assert uid not in {r.id for r in panel.recommender.recs}                           # undone: gone
    assert not groq.raw


def test_refresh_is_debounced_and_keeps_its_order(app, win, groq, work):
    _ready(win)
    win.show_assistant()
    panel = win.assistant_panel
    rc = panel.recommender
    _refresh(app, panel)
    order = _ids(rc.recs)
    n = rc.refreshes
    for _ in range(8):
        rc.schedule()
        pump(app, 0.02)
    assert rc.refreshes == n
    pump(app, rc.DEBOUNCE_MS / 1000 + 0.3)
    assert rc.refreshes == n + 1 and _ids(rc.recs) == order
    emitted = []
    rc.updated.connect(lambda: emitted.append(1))
    rc.refresh()
    assert not emitted                                                   # the same list: no repaint
    rows = dict(panel.empty_list.rows)
    rc.refresh()
    assert all(panel.empty_list.rows[k] is w for k, w in rows.items())   # rows kept, not rebuilt


def test_next_steps_under_an_answer_with_groq(app, win, groq, work):
    from vantage.ui.assistant import AssistantTurn, NextSteps
    _ready(win)
    win.show_assistant()
    panel = win.assistant_panel
    _refresh(app, panel)
    items = [{'label': 'Check QR\'s tolerance', 'prompt': 'Run the mounting tolerance study for QR.', 'kind': 'ask'},
             {'label': 'Drive it in rain', 'prompt': 'a 10 s drive in Town10 in rain', 'kind': 'plan'}]
    groq.script += [groq.tool_call('suggest_next', {'items': items}, 'n1'),
                    groq.answer('QR is the weak camera.')]
    panel.ask('Which camera is weakest?')
    _idle(app, win)
    turn = panel.findChildren(AssistantTurn)[-1]
    assert not turn.tools                                                # suggest_next is not a lookup row
    ns = turn.findChild(NextSteps)
    assert ns is not None and ns.isVisible()
    titles = [b.text() for b in ns.chips.btns]
    assert titles[:2] == ['Check QR\'s tolerance', 'Drive it in rain'] and 3 <= len(titles) <= 5
    local = {r.title for r in panel.recommender.recs}
    assert set(titles[2:]) <= local
    result = json.loads(groq.requests[1]['messages'][-1]['content'])
    assert result['shown'] == 2
    n = len(groq.requests)
    groq.script.append(groq.answer('Studied.'))
    ns.chips.btns[0].click()
    _idle(app, win)
    assert len(groq.requests) == n + 1 and 'mounting tolerance study for QR' in \
        groq.requests[-1]['messages'][-1]['content']


def test_plan_card_revision_chips_follow_up_the_plan(app, win, groq, work):
    from vantage.ui.assistant import CopilotCard
    _ready(win)
    win.show_assistant()
    panel, ctl = win.assistant_panel, win.ctx.assistant
    pl = _draft(['MemAvailable is 6.0 GB; the run needs about 9.0 GB with the server.'])
    ctl._add_plan(pl)
    card = CopilotCard(ctl, pl)
    panel._add(card)
    pump(app, 0.1)
    assert card.chips is not None and [b.text() for b in card.chips.chips.btns] == \
        ['Halve the duration: 20 → 10 s', 'Use half-size images: scale 1 → 0.5']
    groq.script.append(groq.answer('Revised.'))
    card.chips.chips.btns[0].click()
    _idle(app, win)
    assert 'halve the duration to 10 s' in groq.requests[-1]['messages'][-1]['content']
    assert pl.status == 'draft' and not CP.records(win.store.project)    # a revision asks; nothing approved


def test_the_carla_page_context(app, win, groq, work, tmp_path):
    from vantage.core import simrig as SR
    win.ctx.settings.setValue('sim/carla_root', _fake_carla(tmp_path))
    p = samples.tigor_sample_project()                                   # opened as it is: no edit to undo
    p.active_uid = next(v for v in p.variants if v.sensors).uid
    p.analysis.sim = dict(SR.settings_of(p).to_dict(), image_scale=1.0,
                          ground_truth={'semantic': True, 'instance': True, 'range': True})
    win.store.set_project(p)
    win.store.select_cams([])
    settle(app, win)
    _ready(win)
    win.show_assistant()
    win.set_page('sim')
    pg = win.page('sim')
    t0 = time.time()
    while pg.rig is None and time.time() - t0 < 30:
        pump(app, 0.1)
    panel = win.assistant_panel
    _refresh(app, panel)
    ids = list(panel.strip.list.rows)
    assert panel.strip.head.text() == 'CARLA simulator' and panel.strip.isVisible()
    d = pg.rig.doc
    total = sum(c['render_mpx'] for c in d['cameras']) + float(d['ground_truth']['gt_render_mpx'])
    from vantage.sim import carla_bridge as CB
    assert total > CB.RENDER_BUDGET_MPX
    assert {'carla:budget', 'carla:server'} <= set(ids), ids
    b = next(r for r in panel.recommender.recs if r.id == 'carla:budget')
    assert f'{total:.0f} Mpx' in b.why and b.kind == 'plan' and 'Image scale 1 with ground truth' in b.title
    s = next(r for r in panel.recommender.recs if r.id == 'carla:server')
    panel.run_cta(s)
    pump(app, 0.1)
    assert win.page_name() == 'sim' and pg.server is None and not pg.busy and not groq.raw            # pointed at Start; nothing started


# ============================================================================================ Claude
def test_next_steps_with_claude(app, win, work, monkeypatch):
    pytest.importorskip('anthropic', reason='the anthropic SDK is not installed in this Python (optional)')
    from fake_anthropic import FakeAnthropic
    from vantage.assistant import claude as CL
    from vantage.ui.assistant import AssistantTurn, NextSteps
    s = FakeAnthropic(key='sk-ant-test-key').start()
    monkeypatch.setenv('VANTAGE_CLAUDE_URL', s.url)
    try:
        _ready(win, 'claude', 'sk-ant-test-key')
        win.show_assistant()
        panel = win.assistant_panel
        _refresh(app, panel)
        items = [{'label': 'Plan a dusk drive', 'prompt': 'a 10 s drive in Town10 at dusk', 'kind': 'plan'}]
        s.script += [s.reply(text='ML limits the density.', tools=[('suggest_next', {'items': items})]),
                     s.reply(text='That is all.')]
        panel.ask('Which camera limits the density?')
        _idle(app, win)
        tools = {t['name']: t for t in s.requests[0]['tools']}
        sn = tools['suggest_next']
        assert sn['strict'] and sn['input_schema']['additionalProperties'] is False
        assert sn['input_schema']['properties']['items']['items']['additionalProperties'] is False
        assert len(CL.tool_defs()) == len(T.TOOLS)
        turn = panel.findChildren(AssistantTurn)[-1]
        ns = turn.findChild(NextSteps)
        assert ns is not None and ns.chips.btns[0].text() == 'Plan a dusk drive'
        assert ns.recs[0].kind == 'plan' and ns.recs[0].prompt.startswith('Plan this for my approval')
    finally:
        s.stop()
        K.delete('claude')
