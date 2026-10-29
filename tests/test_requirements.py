"""Requirements scorecard: every metric of the registry against its study called directly on the sample data rig
(with an INS and two GNSS antennas), pass/fail and margin signs for each operator, metrics that cannot be evaluated
and why, requirements implied by the plan's settings, the brief parser, the review pack PDF, the CI gate's exit codes,
the Issues list, re-evaluation after edits (stale study values), cancel, and the page (offscreen)."""
import json
import os
import shutil
import subprocess

import numpy as np
import pytest

from vantage.core import perception as PR
from vantage.core import recording as RB
from vantage.core import requirements as RQ
from vantage.core import rules, samples, sensors
from vantage.core import sensor_coverage as SC
from vantage.core import stitched_objects as SO
from vantage.core import timing as TM
from vantage.core import volume as VOL
from vantage.core.analysis import Engine
from vantage.core.project import Project, Sensor
from vantage.core.tolerance import TolSettings, run_monte_carlo, worst_single_camera

from test_gui import app, win, pump, settle, _wait  # noqa: F401 (fixtures)


def _rig():
    """The sample's data rig (6 cameras, a roof lidar, 5 radars) made active, with an INS and two GNSS antennas."""
    p = samples.tigor_sample_project()
    v = next(v for v in p.variants if v.sensors)
    p.active_uid = v.uid
    mesh = p.vehicle.mesh
    for kind, mid in (('imu', 'ins_dual'), ('gnss', 'gnss_survey'), ('gnss', 'gnss_survey')):
        m = p.sensor_models[mid]
        tag = sensors.next_tag(kind, {s.tag for s in v.sensors} | {c.tag for c in v.cameras})
        v.sensors.append(Sensor(kind=kind, model_id=mid, tag=tag,
                                **sensors.default_pose(kind, m, mesh, p.rig, v.sensors)))
    return p


def _item(metric, target, op=None, priority='must', **params):
    m = RQ.METRICS[metric]
    return {'metric': metric, 'params': params, 'op': op or m.better, 'target': target, 'priority': priority}


def _with(p, items, **st):
    p.analysis.requirements = {'items': items, 'settings': st}
    return p


def _every_metric_items(p):
    """One requirement per metric of this build (default parameters), plus parameter variants."""
    items = [_item(mid, 0.0) for mid in RQ.METRICS if RQ.metric_available(mid)]
    items += [_item('perception.range', 40.0, cls='Pedestrian', sector=''),
              _item('perception.range', 80.0, cls='Car', sector='front'),
              _item('sensor_coverage.range', 100.0, modality='lidar', cls='Car', sector='front'),
              _item('sensor_coverage.range', 30.0, modality='radar', cls='Car', sector='rear'),
              _item('sensor_coverage.gaps', 0.0, cls='Pedestrian'),
              _item('sensor_coverage.lidar_blocked_pct', 10.0, lidar='L1'),
              _item('sensor_coverage.ground_blind_m', 3.0, lidar='L1'),
              _item('sensor_coverage.overlap_pct', 50.0, camera='F'),
              _item('issues.count', 0.0, severity='warning'),
              _item('timing.misalignment_ms', 1.0, lines='suggested'),
              _item('volume.undetected_m2', 0.1, object='Adult 1.8 m'),
              _item('tolerance.mc_blind_m2', 0.05, stat='max')]
    return items


@pytest.fixture(scope='module')
def done():
    """The data rig evaluated once with a requirement on every metric (shared by the registry tests)."""
    p = _rig()
    _with_critical(p)
    _with(p, _every_metric_items(p), mc_samples=120)
    eng = Engine()
    res = RQ.run(p, eng)
    return p, eng, res


def _with_critical(p):
    """A child 2 m ahead (resolved) and a sign 400 m ahead (no camera resolves it) as critical targets."""
    from vantage.core import critical as CR
    st = CR.settings_of(p)
    st.targets.append(CR.preset(p, 'Child 2 m ahead', st.targets))
    far = CR.preset(p, 'sign', st.targets)
    far['x'], far['name'] = 400.0, 'Far sign'
    st.targets.append(far)
    p.analysis.critical = st.to_dict()
    return p


def _val(res, metric, **params):
    rs = [r for r in res.items if r.item['metric'] == metric and
          all(r.item['params'].get(k) == v for k, v in params.items())]
    assert rs, (metric, params)
    assert rs[0].status in ('pass', 'fail'), (metric, params, rs[0].reason)
    return rs[0].value


# ======================================================================= the registry against the studies
def test_registry_ground_coverage_and_issues(done):
    p, eng, res = done
    cov = Engine().analyse(p)
    for key in ('blind_m2', 'min_density', 'covered_pct', 'multi_pct', 'min_ok_pct', 'target_ok_pct'):
        assert _val(res, f'coverage.{key}') == pytest.approx(cov.metrics[key], abs=1e-9)
    vg = eng.vehicle_geom(p)
    mounts = {c.uid: rules.mount_info(p, c, vg) for c in p.variant.cameras}
    iss = rules.collect_issues(p, cov, vg, mounts)
    so = SO.issue(SO.analyse(p, eng, cov))
    iss = [i for i in iss if i.action != 'requirement'] + ([so] if so is not None else [])
    assert _val(res, 'issues.count', severity='critical') == sum(i.severity == 'critical' for i in iss)
    assert _val(res, 'issues.count', severity='warning') == len(iss)
    assert res.issues is not None and len(res.issues) == len(iss)


def test_registry_perception_range(done):
    p, eng, res = done
    pr = PR.analyse(p, Engine(), None)
    sec = PR._sector_of(pr.az)
    k_ped = [c.name for c in pr.classes].index('Pedestrian')
    k_car = [c.name for c in pr.classes].index('Car')
    front = [n for n, _ in PR.SECTORS].index('front')
    # independent: the farthest-range arrays, not the sector summary the metric reads
    assert _val(res, 'perception.range', cls='Pedestrian', sector='') == pytest.approx(pr.far[k_ped].min())
    assert _val(res, 'perception.range', cls='Car', sector='front') == pytest.approx(pr.far[k_car, sec == front].min())
    assert _val(res, 'perception.range', cls='', sector='') == pytest.approx(pr.far.min())
    short = sum(not d['pass'] for s in pr.sectors for d in s['classes'].values())
    assert _val(res, 'perception.short') == short == pr.metrics['checks'] - pr.metrics['passed']


def test_registry_lidar_and_radar_coverage(done):
    p, eng, res = done
    sc = SC.run(p, Engine())
    gaps = sum(d['verdict'] == 'gap' for s in sc.sectors for d in s['classes'].values())
    assert _val(res, 'sensor_coverage.gaps', cls='') == gaps == sc.metrics['gap']
    assert _val(res, 'sensor_coverage.redundant_pct', cls='') == pytest.approx(
        100 * sc.metrics['redundant'] / sc.metrics['checks'])
    (L,) = sc.lidars
    assert _val(res, 'sensor_coverage.lidar_blocked_pct', lidar='L1') == pytest.approx(100 * L.blocked)
    assert _val(res, 'sensor_coverage.lidar_blocked_pct', lidar='') == pytest.approx(100 * L.blocked)
    b = np.asarray(L.blind, float)
    assert _val(res, 'sensor_coverage.ground_blind_m', lidar='L1') == pytest.approx(np.median(b[np.isfinite(b)]),
                                                                                    abs=0.006)
    k = [c.name for c in sc.classes].index('Car')
    sec = PR._sector_of(sc.az)
    front = [n for n, _ in PR.SECTORS].index('front')
    assert _val(res, 'sensor_coverage.range', modality='lidar', cls='Car', sector='front') == pytest.approx(
        np.min(sc.lidar[k, sec == front]))
    rear = [n for n, _ in PR.SECTORS].index('rear')
    rr = sc.radar[k, sec == rear]
    assert _val(res, 'sensor_coverage.range', modality='radar', cls='Car', sector='rear') == pytest.approx(
        rr.min() if np.isfinite(rr).all() else 0.0)
    f = next(o for o in sc.overlap if o['tag'] == 'F')
    assert _val(res, 'sensor_coverage.overlap_pct', camera='F') == pytest.approx(100 * f['share']['20'])
    assert _val(res, 'sensor_coverage.overlap_pct', camera='') == pytest.approx(
        100 * min(o['share']['20'] for o in sc.overlap))


def test_registry_timing_recording_power(done):
    p, eng, res = done
    tp = TM.plan(p)
    assert _val(res, 'timing.misalignment_ms', lines='planned') == pytest.approx(tp.metrics['worst_ms'])
    assert _val(res, 'timing.misalignment_ms', lines='suggested') == pytest.approx(tp.suggestion['worst_ms'])
    assert _val(res, 'timing.blur_px') == pytest.approx(max(c['blur_worst_px'] for c in tp.cams))
    rb = RB.budget(p)
    t = rb.totals
    assert _val(res, 'recording.hours_per_set') == pytest.approx(rb.settings.storage_tb * 1e6 / t['mb_s'] / 3600)
    assert _val(res, 'recording.write_load_pct') == pytest.approx(
        100 * t['mb_s'] * rb.settings.write_headroom / rb.settings.write_mb_s)
    assert _val(res, 'recording.uplink_load_pct') == pytest.approx(100 * t['uplink_load'])
    assert _val(res, 'recording.gb_per_hour') == pytest.approx(t['mb_s'] * 3.6)
    assert _val(res, 'recording.link_load_pct') == pytest.approx(
        100 * max(r['load'] for r in rb.rows if r['load'] is not None))
    if RQ.available('power'):
        from vantage.core import power as PW
        pw = PW.run(p)
        tt, st = pw.totals, pw.settings
        assert _val(res, 'power.runtime_min') == pytest.approx(
            st.battery_wh * st.battery_dod / tt['dcdc_input_w'] * 60.0)
        assert _val(res, 'power.dcdc_load_pct') == pytest.approx(100 * tt['dcdc_output_w'] / st.dcdc_w)
        assert _val(res, 'power.alternator_margin_w') == pytest.approx(
            st.alternator_a * st.alternator_v - tt['dcdc_input_w'])
        assert _val(res, 'power.steady_w') == pytest.approx(sum(r['steady_w'] for r in pw.rows))
        assert _val(res, 'power.circuit_load_pct') == pytest.approx(
            100 * max(c['steady_a'] / c['fuse_a'] for c in pw.circuits))
        assert _val(res, 'power.cable_drop_pct') == pytest.approx(
            max(r['drop_pct'] for r in pw.rows if r.get('drop_pct') is not None))


def test_registry_volume_stitched_tolerance(done):
    p, eng, res = done
    e2 = Engine()
    cov = e2.analyse(p)
    st = VOL.VolumeSettings.from_dict(p.analysis.volume)
    vr = VOL.analyse_volume(p, e2, cov, st.height, st.dz, st.cell or None)
    assert _val(res, 'volume.blind_m3') == pytest.approx(vr.metrics['blind_m3'])
    assert _val(res, 'volume.covered_to_m') == pytest.approx(vr.metrics['covered_to_m'])
    assert _val(res, 'volume.undetected_m2', object='Adult 1.8 m') == pytest.approx(
        vr.metrics['objects']['Adult 1.8 m']['undetected_m2'])
    so = SO.analyse(p, e2, cov)
    assert _val(res, 'stitched.vanish_m2') == pytest.approx(so.metrics['vanish_m2'])
    assert _val(res, 'stitched.shown_pct') == pytest.approx(100 * so.metrics['shown_mean'])
    a = p.analysis
    w = worst_single_camera(e2, p, a.tol_deg, a.tol_mm / 1000.0, cell=max(a.cell_m, 0.05))
    assert _val(res, 'tolerance.single_blind_m2') == pytest.approx(w[0])
    mc = run_monte_carlo(e2, p, TolSettings(yaw=a.tol_deg, pitch=a.tol_deg, roll=a.tol_deg, pos_mm=10.0, samples=120,
                                            cell=max(a.cell_m, 0.05)))
    assert _val(res, 'tolerance.mc_blind_m2', stat='p95') == pytest.approx(np.percentile(mc.blind, 95))
    assert _val(res, 'tolerance.mc_blind_m2', stat='max') == pytest.approx(mc.blind.max())


def test_registry_gnss(done):
    p, eng, res = done
    if not RQ.available('gnss'):
        pytest.skip('the GNSS and IMU study is not part of this build')
    from vantage.core import gnss as GN
    gr = GN.run(p, Engine())
    assert len(gr.antennas) == 2 and gr.imu is not None and gr.baseline is not None
    assert _val(res, 'gnss.sky_pct') == pytest.approx(100 * min(a.visible for a in gr.antennas))
    assert _val(res, 'gnss.expected_sats') == pytest.approx(min(a.expected_sats for a in gr.antennas))
    b = gr.baseline
    assert _val(res, 'gnss.baseline_m') == pytest.approx(b['horizontal_m'])
    m = p.sensor_models['ins_dual']
    assert _val(res, 'gnss.heading_sigma_deg') == pytest.approx(m.heading_sigma(b['horizontal_m']))
    assert _val(res, 'gnss.pose_sigma_mm') == pytest.approx(max(r['total_mm'] for r in gr.budget['rows']))
    # without antennas: why not
    q = _rig()
    q.variant.sensors = [s for s in q.variant.sensors if s.kind != 'gnss']
    r = RQ.run(_with(q, [_item('gnss.sky_pct', 90.0), _item('gnss.heading_sigma_deg', 0.2)]), Engine())
    assert [x.status for x in r.items] == ['not_evaluated'] * 2
    assert 'No visible GNSS antenna' in r.items[0].reason and 'two antenna inputs' in r.items[1].reason


def test_registry_exposure(done):
    p, eng, res = done
    if not RQ.available('exposure'):
        pytest.skip('the sun glare and dirt study is not part of this build')
    from vantage.core import exposure as XP
    xr = XP.run(p, Engine())
    cams = [i for i in xr.items if i.kind == 'camera']
    assert _val(res, 'exposure.sun_pct') == pytest.approx(100 * max(i.sun['share'] for i in cams))
    assert _val(res, 'exposure.near_pct') == pytest.approx(100 * max(i.sun['near'] for i in cams))
    assert _val(res, 'exposure.low_pct') == pytest.approx(100 * max(i.sun['low'] for i in cams))
    assert _val(res, 'exposure.lidar_sun_pct') == pytest.approx(
        100 * max(i.sun['share'] for i in xr.items if i.kind == 'lidar'))
    assert _val(res, 'exposure.dirt_score') == pytest.approx(
        max(i.dirt['score'] for i in xr.items if i.kind in ('camera', 'lidar')))
    r = RQ.run(_with(_rig(), [_item('exposure.sun_pct', 5.0, camera='F'), _item('exposure.dirt_score', 0.5,
                                                                               sensor='R')]), Engine())
    f = next(i for i in cams if i.tag == 'F')
    b = next(i for i in xr.items if i.tag == 'R')
    assert r.items[0].value == pytest.approx(100 * f.sun['share'])
    assert r.items[1].value == pytest.approx(b.dirt['score'])


def test_registry_critical_targets_and_observability(done):
    """critical.unresolved and observability.* against the studies called directly (the scorecard runs them with the
    envelopes, the board suggestions, the lidar pairs and the fusion density left out: the figures are the same)."""
    from vantage.core import critical as CR
    from vantage.core import observability as OB
    p, eng, res = done
    cr = CR.run(p, Engine())
    miss = [r.name for r in cr.targets if not r.met]
    assert miss == ['Far sign'] and _val(res, 'critical.unresolved') == 1.0
    r = next(x for x in res.items if x.item['metric'] == 'critical.unresolved')
    assert r.status == 'fail' and 'Far sign' in r.evidence['detail'] and r.evidence['study'] == 'critical'
    ob = OB.run(p, Engine())
    cams = [c for c in ob.cameras if c['sigma'] is not None]
    assert len(cams) == len(ob.cameras) == sum(c.visible for c in p.variant.cameras)
    assert _val(res, 'observability.worst_ratio') == pytest.approx(max(c['ratio'] for c in cams), rel=1e-9)
    assert _val(res, 'observability.worst_deg') == pytest.approx(
        max(c['sigma'][k] for c in cams for k in ('yaw_deg', 'pitch_deg', 'roll_deg')), rel=1e-9)
    assert _val(res, 'observability.worst_mm') == pytest.approx(
        max(c['sigma'][k] for c in cams for k in ('x_mm', 'y_mm', 'z_mm')), rel=1e-9)
    # no critical targets, no boards: not evaluated, with the reason
    q = _rig()
    q.targets = []
    _with(q, [_item('critical.unresolved', 0.0), _item('observability.worst_ratio', 1.0)])
    out = {r.item['metric']: r for r in RQ.run(q, Engine()).items}
    assert out['critical.unresolved'].status == 'not_evaluated' and 'No critical target' in \
        out['critical.unresolved'].reason
    assert out['observability.worst_ratio'].status == 'not_evaluated' and 'No calibration targets' in \
        out['observability.worst_ratio'].reason


def test_registry_fleet():
    if not RQ.available('fleet'):
        pytest.skip('the fleet study is not part of this build')
    from vantage.core import fleet as FL
    p = _rig()
    q = _with(_rig(), [_item('fleet.out_of_tolerance', 0.0)])
    r = RQ.run(q, Engine(), issues_list=False)
    assert r.items[0].status == 'not_evaluated' and 'No vehicle' in r.items[0].reason
    fleet = {'vehicles': [], 'settings': {}}
    # V1 as planned, V2 with F 25 mm forward (2.5× the 10 mm tolerance), V3 with F turned 3.5° (3.5× the 1°)
    for k, (dx, dyaw) in enumerate([(0.0, 0.0), (0.025, 0.0), (0.0, 3.5)]):
        veh = FL.add_vehicle(fleet, f'V{k + 1}')
        rec = FL.record_from_variant(p, p.variant, date=f'2026-05-0{k + 1}')
        rec['poses']['F']['x'] += dx
        rec['poses']['F']['yaw'] += dyaw
        FL.add_record(fleet, veh['id'], rec)
    p.fleet = fleet
    fr = FL.run(p)
    cells = list(fr.cells.values())
    res = RQ.run(_with(p, [_item('fleet.out_of_tolerance', 0.0), _item('fleet.remount', 0.0),
                           _item('fleet.worst_pct', 100.0), _item('fleet.drifting', 0.0),
                           _item('fleet.outliers', 0.0)]), Engine(), issues_list=False)
    v = [x.value for x in res.items]
    assert v[0] == sum(c['verdict'] in ('adjust', 're-mount') for c in cells) == 2
    assert v[1] == sum(c['verdict'] == 're-mount' for c in cells) == 1
    assert v[2] == pytest.approx(100 * max(c['ratio'] for c in cells)) and v[2] == pytest.approx(350, rel=0.01)
    assert v[3] == len(fr.flagged('drift')) and v[4] == sum(len(f['outliers']) for f in fr.fleet)
    assert [x.status for x in res.items][:3] == ['fail', 'fail', 'fail']
    plan = {i['metric'] for i in RQ.from_plan(p)}
    assert {'fleet.out_of_tolerance', 'fleet.remount'} <= plan


def test_registry_brackets():
    if not RQ.available('brackets'):
        pytest.skip('the bracket generator is not part of this build')
    from vantage.core import brackets as BR
    p = _rig()
    eng = Engine()
    plan = BR.plan(p, eng)
    bad = [b for b in plan.brackets if b.status in ('standoff', 'floating', 'inside', 'face')]
    res = RQ.run(_with(p, [_item('brackets.not_buildable', 0.0), _item('brackets.fov_blocked_pct', 0.5, camera='QL'),
                           _item('brackets.overhang_mm', 60.0, camera='QL'),
                           _item('brackets.fov_blocked_pct', 0.5, camera='F')]), eng, issues_list=False)
    v = res.items
    assert v[0].value == len(bad) and len(bad) >= 1 and v[0].status == 'fail'
    assert 'standoff' in v[0].evidence['detail']
    ql = BR.run(p, Engine(), cams=['QL']).by_tag('QL')
    assert ql.generated
    assert v[1].value == pytest.approx(ql.params['fov_blocked_pct'], abs=1e-6)
    assert v[2].value == pytest.approx(ql.params['lever_mm'], abs=1e-6)
    f = plan.by_tag('F')
    if f.status != 'ok':                                # a camera that needs a standoff gets no bracket: why not
        assert v[3].status == 'not_evaluated' and "'F'" in v[3].reason
    assert 'brackets_full' in res.studies and res.studies['brackets_full']['ran']


def test_registry_drivelog(tmp_path):
    if not RQ.available('drivelog'):
        pytest.skip('the drive-log check is not part of this build')
    import importlib
    td = importlib.import_module('test_drivelog')       # its synthetic bag with injected faults
    from vantage.core import drivelog as DL
    p, v = td.make_project()
    eng = Engine()
    items = [_item('drivelog.faulty_streams', 0.0), _item('drivelog.missing_streams', 0.0),
             _item('drivelog.drop_pct', 0.1), _item('drivelog.sync_ms', 1.0), _item('drivelog.rotation_deg', 0.3),
             _item('drivelog.lidar_phase_deg', 1.0)]
    r = RQ.run(_with(p, items), eng, issues_list=False)
    assert all(x.status == 'not_evaluated' and 'No drive log checked yet' in x.reason for x in r.items)
    path = tmp_path / 'drive'
    td.make_drive(path, p, v, eng)
    p.analysis.drivelog = {'path': str(path), 'camera_stamp': 'trigger', 'frames_per_camera': td.N_FRAMES}
    dr = DL.run(p, None, DL.DriveLogSettings.from_dict(p.analysis.drivelog), engine=eng)
    m = dr.metrics
    res = RQ.run(p, eng, issues_list=False)
    val = [x.value for x in res.items]
    assert val[0] == m['streams_faulty'] >= 1 and val[1] == len(m['missing']) == 1
    assert val[2] == pytest.approx(100 * m['worst_drop_share']) and m['worst_drop'] == 'MR'
    assert val[3] == pytest.approx(m['worst_sync_ms']) and m['worst_sync_tag'] == 'R'
    assert val[4] == pytest.approx(m['worst_rotation_deg']) and m['worst_rotation_tag'] == 'ML'
    assert val[5] == pytest.approx(abs(m['lidar_phase_error_deg']))
    assert [x.status for x in res.items] == ['fail'] * 6
    assert res.items[0].evidence['study'] == 'drivelog' and res.items[0].evidence['figures']


def test_registry_lidar_camera_calibration(tmp_path):
    if not RQ.available('lidar_plan'):
        pytest.skip('the lidar-camera calibration is not part of this build')
    from vantage.core import lidar_calib as LC
    p = _rig()
    eng = Engine()
    st = LC.LidarCalibSettings.from_dict({'camera': 'F', 'lidar': 'L1'})
    p.analysis.lidar_calib = st.to_dict()
    cam, lid, _ = LC.pick_pair(p, p.variant, st, eng)
    poses, _ = LC.plan_poses(p, cam, lid, st, eng)
    sg = LC.predicted_sigma(p, cam, lid, poses, st)
    items = [_item('lidar_calib.planned_sigma_mm', 3.0), _item('lidar_calib.planned_sigma_deg', 0.1),
             _item('lidar_calib.delta_mm', 10.0), _item('lidar_calib.delta_deg', 0.3),
             _item('lidar_calib.sigma_mm', 5.0)]
    res = RQ.run(_with(p, items), eng, issues_list=False)
    assert res.items[0].value == pytest.approx(max(sg[k] for k in ('x_mm', 'y_mm', 'z_mm')))
    assert res.items[1].value == pytest.approx(max(sg[k] for k in ('roll_deg', 'pitch_deg', 'yaw_deg')))
    assert all(r.status == 'not_evaluated' and 'No captured board data' in r.reason for r in res.items[2:])
    # real captures: the synthetic pairs of a lidar mounted 15 mm and 0.5° off, written as files
    r0 = LC.run(p, eng, LC.LidarCalibSettings.from_dict(dict(st.to_dict(), err_x_mm=15.0, err_yaw_deg=0.5)))
    LC.save_captures(r0.captures, str(tmp_path), '.pcd')
    p.analysis.lidar_calib = dict(st.to_dict(), source='folder', folder=str(tmp_path))
    res = RQ.run(p, eng, issues_list=False)
    direct = LC.run(p, Engine(), LC.LidarCalibSettings.from_dict(p.analysis.lidar_calib))
    assert direct.T is not None
    v = [r.value for r in res.items]
    assert v[2] == pytest.approx(direct.delta['position_mm'], abs=1e-6) and v[2] == pytest.approx(15.0, abs=3.0)
    assert v[3] == pytest.approx(direct.delta['rotation_deg'], abs=1e-9) and v[3] == pytest.approx(0.5, abs=0.1)
    assert v[4] == pytest.approx(direct.sigma['position_mm'], abs=1e-6)
    assert [r.status for r in res.items[2:4]] == ['fail', 'fail']


def test_every_metric_evaluates_and_studies_run_once(done):
    p, eng, res = done
    bad = [(r.item['metric'], r.reason) for r in res.items if r.status not in ('pass', 'fail')]
    # the data rig has no fleet records, no drive log and no captured calibration boards (each is tested with data
    # of its own): they say so
    assert all((m.startswith('fleet.') and 'No vehicle with a measured record' in why) or
               (m.startswith('drivelog.') and 'No drive log checked yet' in why) or
               (m.startswith('lidar_calib.') and 'No captured board data' in why) for m, why in bad)
    used = {RQ.METRICS[r.item['metric']].study for r in res.items}
    assert used <= set(res.studies) and all(s['ran'] for k, s in res.studies.items()
                                            if k not in ('fleet', 'drivelog', 'lidar_calib'))
    assert not RQ.available('fleet') or (not res.studies['fleet']['ran'] and res.studies['fleet']['error'])
    # evidence names the study, the variant and the figures
    r = next(x for x in res.items if x.item['metric'] == 'perception.range')
    assert r.evidence['study'] == 'perception' and r.evidence['variant'] == p.variant.name
    assert r.evidence['figures'] and 'Shortest' in r.evidence['detail']


# ======================================================================= operators, margins, verdicts
@pytest.mark.parametrize('op,value,target,ok,margin', [
    ('>=', 50.0, 40.0, True, 10.0), ('>=', 40.0, 40.0, True, 0.0), ('>=', 30.0, 40.0, False, -10.0),
    ('>', 40.0, 40.0, False, 0.0), ('>', 41.0, 40.0, True, 1.0),
    ('<=', 0.5, 1.0, True, 0.5), ('<=', 1.0, 1.0, True, 0.0), ('<=', 1.5, 1.0, False, -0.5),
    ('<', 1.0, 1.0, False, 0.0), ('<', 0.9, 1.0, True, 0.1),
    ('==', 3.0, 3.0, True, 0.0), ('==', 2.0, 3.0, False, -1.0), ('==', 4.0, 3.0, False, -1.0)])
def test_compare_signs(op, value, target, ok, margin):
    got_ok, got_m = RQ.compare(value, op, target)
    assert got_ok is ok and got_m == pytest.approx(margin)


def test_items_pass_and_fail_with_margins_and_verdict():
    p = _rig()
    hours = RB.budget(p).totals['hours_per_set']
    items = [_item('recording.hours_per_set', hours - 1.0), _item('recording.hours_per_set', hours + 1.0),
             _item('recording.hours_per_set', hours + 1.0, op='<='), _item('recording.hours_per_set', 99.0,
                                                                           priority='should')]
    res = RQ.run(_with(p, items), cheap_only=True)
    st = [r.status for r in res.items]
    assert st == ['pass', 'fail', 'pass', 'fail']
    assert res.items[0].margin == pytest.approx(1.0) and res.items[1].margin == pytest.approx(-1.0)
    assert res.items[2].margin == pytest.approx(1.0)
    assert res.verdict == 'fail'
    c = res.counts()
    assert c['must'] == {'total': 3, 'pass': 2, 'fail': 1, 'not_evaluated': 0}
    assert c['should'] == {'total': 1, 'pass': 0, 'fail': 1, 'not_evaluated': 0}
    res = RQ.run(_with(p, [items[0], items[3]]), cheap_only=True)
    assert res.verdict == 'pass'                         # a failing should requirement never decides
    assert RQ.run(_with(p, []), cheap_only=True).verdict == 'empty'


def test_not_evaluated_says_why():
    p = _rig()
    v = p.variant
    v.sensors = [s for s in v.sensors if s.kind != 'lidar']          # no lidar any more
    items = [_item('timing.misalignment_ms', 1.0), _item('sensor_coverage.lidar_blocked_pct', 10.0),
             _item('sensor_coverage.range', 50.0, modality='lidar', cls='Car', sector='front'),
             _item('sensor_coverage.gaps', 0.0, cls='Unicorn'),
             _item('perception.range', 40.0, cls='Unicorn', sector=''),
             {'metric': 'no.such.metric', 'op': '>=', 'target': 1, 'priority': 'must'},
             _item('recording.hours_per_set', float('nan'))]
    res = RQ.run(_with(p, items), Engine())
    r = {i: x for i, x in enumerate(res.items)}
    assert all(x.status == 'not_evaluated' and x.reason for x in res.items)
    assert 'spinning lidar' in r[0].reason
    assert 'No lidar' in r[1].reason and 'No lidar range' in r[2].reason and 'no lidar in the variant' in r[2].reason
    assert "'Unicorn'" in r[3].reason
    assert "'Unicorn'" in r[4].reason
    assert 'Unknown metric' in r[5].reason
    assert 'not a finite number' in r[6].reason
    assert res.verdict == 'incomplete'
    # no vehicle body: the geometry studies cannot run, the cheap ones can
    q = Project()
    q.variant.cameras.extend(p.variant.cameras[:2])
    res = RQ.run(_with(q, [_item('coverage.blind_m2', 0.0), _item('recording.gb_per_hour', 1e6)]))
    assert res.items[0].status == 'not_evaluated' and 'No vehicle body' in res.items[0].reason
    assert res.items[1].status == 'pass'


def test_another_variant_and_a_missing_one():
    p = _rig()
    four = next(v for v in p.variants if v.name == '4-cam')
    items = [_item('recording.gb_per_hour', 1e6)]
    res = RQ.run(_with(p, items, variant=four.uid), cheap_only=True)
    assert res.variant == '4-cam' and res.items[0].evidence['variant'] == '4-cam'
    assert res.items[0].value == pytest.approx(RB.budget(p, four).totals['gb_h'])
    assert p.variant.name.startswith('Data rig')                     # the project's active variant is left alone
    res = RQ.run(_with(p, items, variant='gone'), cheap_only=True)
    assert res.variant == p.variant.name and 'gone' in res.items[0].evidence['note']


# ======================================================================= items, validation, settings
def test_settings_and_items_are_tolerant():
    st = RQ.RequirementsSettings.from_dict({
        'items': [{'metric': 'coverage.blind_m2', 'target': '0.5', 'op': '≤', 'priority': 'maybe', 'id': 'A'},
                  {'metric': 'perception.range', 'params': {'cls': 'Car', 'junk': 1}, 'target': 'x', 'id': 'A'},
                  'junk', None, {'metric': 'timing.blur_px', 'target': 1, 'op': '<'}],
        'settings': {'mc_samples': 1e9, 'mc_pos_mm': 'abc', 'variant': 7, 'title': '  '}})
    a, b, c = st.items
    assert a['op'] == '<=' and a['target'] == 0.5 and a['priority'] == 'must' and a['id'] == 'A'
    assert a['text'] == 'Blind ground in the ROI ≤ 0.5 m²'
    assert b['id'] != 'A' and b['params'] == {'cls': 'Car', 'sector': ''} and b['target'] is None
    assert c['op'] == '<' and c['id'] not in ('A', b['id'])
    assert st.mc_samples == 5000 and st.mc_pos_mm == 10.0 and st.variant == '' and st.title == 'Design review'
    again = RQ.RequirementsSettings.from_dict(st.to_dict())
    assert again.to_dict() == st.to_dict()
    p = _rig()
    ok, msg = RQ.validate_item(_item('sensor_coverage.lidar_blocked_pct', 10, lidar='L1'), p)
    assert ok and msg == 'OK'
    ok, msg = RQ.validate_item(_item('sensor_coverage.lidar_blocked_pct', 10, lidar='L7'), p)
    assert not ok and "'L7'" in msg
    assert not RQ.validate_item(_item('timing.blur_px', 1, op='~'))[0]
    assert not RQ.validate_item(dict(_item('timing.blur_px', 1), params={'x': 1}))[0]
    assert not RQ.validate_item({'metric': 'nope'})[0]


def test_catalogue_is_json_with_the_projects_choices():
    p = _rig()
    cat = RQ.catalogue(p)
    json.dumps(cat)
    m = {x['id']: x for x in cat['metrics']}
    blocked = m['sensor_coverage.lidar_blocked_pct']
    assert blocked['study'] == 'sensor_coverage' and blocked['cost'] == 'study' and blocked['unit'] == '%'
    assert [c['value'] for c in blocked['params'][0]['choices']] == ['', 'L1']
    assert m['recording.hours_per_set']['cost'] == 'cheap' and m['recording.hours_per_set']['better'] == '>='
    sec = m['perception.range']['params'][1]
    assert [c['value'] for c in sec['choices']][:3] == ['', 'front', 'front-left']
    assert set(cat['operators']) == set(RQ.OPS)


# ======================================================================= defaults from the plan and templates
def test_from_the_plan_follows_the_settings():
    p = _rig()
    p.analysis.min_density = 4.5
    p.analysis.perception = {'required': {'front': 95.0, 'rear': 0.0}}
    p.analysis.recording = {'shift_hours': 10.0, 'link_warn': 0.6}
    p.analysis.timing = {'blur_limit_px': 2.5}
    items = RQ.from_plan(p)
    by = {}
    for it in items:
        by.setdefault(it['metric'], []).append(it)
    assert by['coverage.min_density'][0]['target'] == 4.5
    front = [i for i in by['perception.range'] if i['params']['sector'] == 'front']
    assert front[0]['target'] == 95.0 and front[0]['params']['cls'] == ''
    assert not [i for i in by['perception.range'] if i['params']['sector'] == 'rear']     # 0 m: nothing required
    assert by['recording.hours_per_set'][0]['target'] == 10.0
    assert by['recording.uplink_load_pct'][0]['target'] == pytest.approx(60.0)
    assert by['timing.blur_px'][0]['target'] == 2.5
    assert by['issues.count'][0]['params'] == {'severity': 'critical'} and by['issues.count'][0]['target'] == 0
    assert all(it['note'] for it in items)
    if RQ.available('power'):
        p.analysis.power = {'runtime_target_min': 45}
        assert next(i for i in RQ.from_plan(p) if i['metric'] == 'power.runtime_min')['target'] == 45
    assert all(RQ.validate_item(RQ.clean_item(i, p), p)[0] for i in RQ.from_plan(p))
    # a variant without lidars or radars gets no lidar requirement
    four = next(v for v in p.variants if v.name == '4-cam')
    metrics = {i['metric'] for i in RQ.from_plan(p, four)}
    assert 'sensor_coverage.gaps' not in metrics and 'timing.misalignment_ms' not in metrics
    assert 'coverage.blind_m2' in metrics


def test_templates_are_valid_for_the_project():
    p = _rig()
    p.analysis.perception = {'classes': [{'name': 'Car', 'height': 1.5, 'width': 1.8, 'min_px': 20}]}
    ts = RQ.templates(p)
    assert len(ts) >= 20
    for t in ts:
        ok, msg = RQ.validate_item(RQ.clean_item(t, p), p)
        assert ok, (t['text'], msg)
    ped = next(t for t in ts if t['text'].startswith('Pedestrians'))
    assert ped['params']['cls'] == ''                  # no pedestrian class here: every class


# ======================================================================= the brief
BRIEF_CASES = [
    ('Pedestrians resolved to 40 m in every sector.',
     [('perception.range', {'cls': 'Pedestrian', 'sector': ''}, '>=', 40.0, 'must')]),
    ('We need 8 hours of driving per disk set',
     [('recording.hours_per_set', {}, '>=', 8.0, 'must')]),
    ('≤ 1 ms sync between the cameras and the lidar',
     [('timing.misalignment_ms', {'lines': 'planned'}, '<=', 1.0, 'must')]),
    ('Cars to 80 m ahead, pedestrians and cyclists to 30 m to the sides',
     [('perception.range', {'cls': 'Car', 'sector': 'front'}, '>=', 80.0, 'must'),
      ('perception.range', {'cls': 'Pedestrian', 'sector': 'left'}, '>=', 30.0, 'must'),
      ('perception.range', {'cls': 'Pedestrian', 'sector': 'right'}, '>=', 30.0, 'must'),
      ('perception.range', {'cls': 'Cyclist', 'sector': 'left'}, '>=', 30.0, 'must'),
      ('perception.range', {'cls': 'Cyclist', 'sector': 'right'}, '>=', 30.0, 'must')]),
    ('Ground should be resolved at 3 px/cm at least',
     [('coverage.min_density', {}, '>=', 3.0, 'should')]),
    ('Motion blur below 1.5 px; the uplink under 70 %',
     [('timing.blur_px', {}, '<', 1.5, 'must'), ('recording.uplink_load_pct', {}, '<', 70.0, 'must')]),
    ('The lidar must detect cars to 0.12 km ahead while we drive at 50 km/h',
     [('sensor_coverage.range', {'modality': 'lidar', 'cls': 'Car', 'sector': 'front'}, '>=', 120.0, 'must')]),
    ('No critical issues.', [('issues.count', {'severity': 'critical'}, '<=', 0.0, 'must')]),
]


@pytest.mark.parametrize('text,want', BRIEF_CASES)
def test_brief_phrasings(text, want):
    p = _rig()
    got = RQ.parse_brief(text, p)
    assert [(o['item']['metric'], o['item']['params'], o['item']['op'], o['item']['target'], o['item']['priority'])
            for o in got] == want
    for o in got:
        assert RQ.validate_item(o['item'], p)[0]
        assert o['source'] and o['item']['text'] == o['source'][:1].upper() + o['source'][1:]
        assert 'brief' in o['item']['note']


def test_brief_power_and_gnss_phrasings():
    p = _rig()
    if RQ.metric_available('power.runtime_min'):
        (o,) = RQ.parse_brief('30 min engine off for calibration', p)
        assert (o['item']['metric'], o['item']['target'], o['item']['op']) == ('power.runtime_min', 30.0, '>=')
        (o,) = RQ.parse_brief('Engine-off runtime of 1.5 h', p)
        assert o['item']['target'] == 90.0
    if RQ.metric_available('gnss.sky_pct'):
        (o,) = RQ.parse_brief('≥ 90 % sky for each GNSS antenna', p)
        assert (o['item']['metric'], o['item']['target'], o['item']['op']) == ('gnss.sky_pct', 90.0, '>=')


def test_brief_junk_yields_nothing():
    p = _rig()
    for junk in ('The weather in Munich is lovely in May.', 'Our team of 4 engineers meets on Mondays at 9.',
                 'Budget: 40 k EUR.', '', '   ', None, 'Drive 300 km a day around the city.'):
        assert RQ.parse_brief(junk, p) == []
    # a brief mixing junk and requirements: only the requirements, each once
    got = RQ.parse_brief('Hello team. Pedestrians to 40 m. Pedestrians to 40 m! Thanks, the team', p)
    assert len(got) == 1


# ======================================================================= outputs
def _pdf_pages(data: bytes) -> int:
    import re
    return len(re.findall(rb'/Type\s*/Page[^s]', data))


def test_review_pack_pdf(done, tmp_path):
    p, eng, res = done
    data, pages, sections = RQ.review_pack(res, p)
    assert data.startswith(b'%PDF') and data.rstrip().endswith(b'%%EOF')
    assert _pdf_pages(data) == pages
    studies = [s for s in res.studies if s != 'issues']
    assert [k for _, k in sections] == ['title', 'scorecard'] + ['study'] * len(studies) + ['issues']
    assert pages >= len(sections)                         # every section starts on a page of its own
    # the requirement texts and the verdict travel in the XMP metadata (uncompressed) …
    for r in res.items:
        assert r.item['text'].replace('&', '&amp;').encode('utf-8') in data
    assert f'Verdict: {RQ.VERDICT_TEXT[res.verdict]}'.encode() in data
    # … and are the PDF's text (when poppler's pdftotext is installed)
    if shutil.which('pdftotext'):
        path = tmp_path / 'pack.pdf'
        path.write_bytes(data)
        text = subprocess.run(['pdftotext', '-layout', str(path), '-'], capture_output=True, text=True,
                              timeout=60).stdout
        flat = ' '.join(text.split())
        assert 'DESIGN REVIEW PACK' in text and 'Scorecard' in text and 'Issues' in text
        for title, _ in sections[2:-1]:
            assert title in flat
        assert f'{len(res.items)}' in text


def test_json_csv_files_and_export(done, tmp_path):
    p, eng, res = done
    doc = RQ.to_doc(res, p)
    assert doc['schema'] == RQ.SCHEMA and doc['variant'] == p.variant.name and doc['frame'].startswith('vehicle')
    assert len(doc['requirements']) == len(res.items)
    back = json.loads(json.dumps(doc))
    for d, r in zip(back['requirements'], res.items):
        assert d['status'] == r.status and d['id'] == r.item['id']
        # a requirement whose metric was not computed has no value (pytest 9.2 refuses approx(None))
        assert d['value'] == (None if r.value is None else pytest.approx(r.value, rel=1e-6, abs=1e-6))
        assert d['evidence']['study'] == RQ.METRICS[r.item['metric']].study
    assert {s['id'] for s in back['studies']} == set(res.studies)
    assert all(s['figures'] for s in back['studies'] if s['ran'])
    assert back['exit_code'] == (5 if res.verdict in ('fail', 'incomplete') else 0)
    import csv
    import io
    rows = list(csv.DictReader(io.StringIO(RQ.to_csv(res))))
    assert len(rows) == len(res.items) and rows[0]['id'] == res.items[0].item['id']
    assert float(rows[0]['value']) == pytest.approx(res.items[0].value, rel=1e-5)
    paths = RQ.export(res, p, str(tmp_path / 'out'))
    assert [os.path.basename(x) for x in paths] == ['requirements.json', 'requirements.csv', 'review_pack.pdf']
    s = RQ.summary(res)
    json.dumps(s)
    assert s['verdict'] == res.verdict and len(s['items']) == len(res.items)


# ======================================================================= the CI gate and the Issues list
def test_check_exit_codes(tmp_path):
    p = _rig()
    gbh = RB.budget(p).totals['gb_h']
    _with(p, [_item('recording.gb_per_hour', gbh + 1)])
    doc, code = RQ.check(p, out_path=str(tmp_path / 'ok.json'))
    assert code == 0 and doc['verdict'] == 'pass'
    assert json.loads((tmp_path / 'ok.json').read_text())['verdict'] == 'pass'
    _with(p, [_item('recording.gb_per_hour', gbh + 1), _item('recording.gb_per_hour', gbh - 1, priority='should')])
    assert RQ.check(p)[1] == 0                               # a should failing does not gate
    _with(p, [_item('recording.gb_per_hour', gbh - 1)])
    doc, code = RQ.check(p, out_path=str(tmp_path / 'pack'))
    assert code == RQ.EXIT_FAIL == 5 and doc['verdict'] == 'fail'
    assert sorted(os.listdir(tmp_path / 'pack')) == ['requirements.csv', 'requirements.json', 'review_pack.pdf']
    _with(p, [_item('recording.gb_per_hour', gbh + 1), _item('sensor_coverage.lidar_blocked_pct', 50, lidar='L9')])
    doc, code = RQ.check(p)
    assert code == 5 and doc['verdict'] == 'incomplete'
    four = next(v for v in p.variants if v.name == '4-cam')        # batch --variant: that variant, whatever is set
    _with(p, [_item('recording.gb_per_hour', 1e6)], variant=p.variant.uid)
    doc, code = RQ.check(p, variant=four.uid)
    assert code == 0 and doc['variant'] == '4-cam'
    _with(p, [])
    doc, code = RQ.check(p)
    assert code == 0 and doc['verdict'] == 'empty'


def test_saved_with_the_project(tmp_path):
    p = _rig()
    hours = RB.budget(p).totals['hours_per_set']
    _with(p, [dict(_item('recording.hours_per_set', hours + 1), text='A full shift per disk set', note='CI'),
              _item('perception.range', 40.0, cls='Pedestrian', sector='')], mc_samples=200, title='Gate 2')
    path = str(tmp_path / 'rig.vantage')
    p.save(path)
    q = Project.load(path)
    assert RQ.settings_of(q).to_dict() == RQ.settings_of(p).to_dict()
    doc, code = RQ.check(q)
    assert code == 5 and [r['status'] for r in doc['requirements']] == ['fail', 'pass']
    assert doc['requirements'][0]['text'] == 'A full shift per disk set' and doc['requirements'][0]['note'] == 'CI'


def test_issues_only_cheap_failing_musts():
    p = _rig()
    hours = RB.budget(p).totals['hours_per_set']
    _with(p, [_item('recording.hours_per_set', hours + 5),                    # cheap must, fails: an issue
              _item('recording.hours_per_set', hours + 5, priority='should'),  # a should: none
              _item('recording.hours_per_set', hours - 5),                     # passes: none
              _item('coverage.blind_m2', 0.0)])                                # a study metric: not run here
    iss = RQ.issues(p)
    assert len(iss) == 1 and iss[0].severity == 'serious' and iss[0].action == 'requirement'
    assert 'Requirement not met' in iss[0].message and 'h' in iss[0].message
    import time
    t0 = time.perf_counter()
    for _ in range(10):
        RQ.issues(p)
    assert (time.perf_counter() - t0) / 10 < 0.03
    p.analysis.requirements = {}
    assert RQ.issues(p) == []


# ======================================================================= re-evaluation, staleness, cancel
def test_reevaluate_reuses_studies_and_marks_stale():
    p = _rig()
    _with(p, [_item('coverage.covered_pct', 99.0), _item('recording.gb_per_hour', 1e6)])
    eng = Engine()
    full = RQ.run(p, eng)
    assert full.items[0].status == 'pass' and 'coverage' in full.cache
    # a requirement edited (the design is the same): the coverage is reused at once, not stale
    _with(p, [_item('coverage.covered_pct', 100.0), _item('recording.gb_per_hour', 1e6),
              _item('coverage.multi_pct', 10.0)])
    r = RQ.reevaluate(p, full)
    assert [x.status for x in r.items] == ['fail', 'pass', 'pass'] and not any(x.stale for x in r.items)
    assert r.items[0].value == full.items[0].value
    # the design changes: the study value is kept but stale, the cheap one is new
    c = p.variant.by_tag('F')
    c.yaw += 30.0
    r2 = RQ.reevaluate(p, full)
    assert r2.items[0].stale and r2.items[0].value == full.items[0].value and not r2.items[1].stale
    assert r2.items[2].status == 'not_run'                  # never run: needs a run
    assert r2.verdict == 'fail'
    # a new run on the new design
    r3 = RQ.run(p, eng, reuse=full)
    assert not any(x.stale for x in r3.items) and r3.inputs_key != full.inputs_key


def test_cancel():
    p = _rig()
    _with(p, [_item('coverage.covered_pct', 99.0), _item('sensor_coverage.gaps', 0.0)])
    assert RQ.run(p, Engine(), cancel=lambda: True) is None
    seen = []

    def prog(f, text=''):
        seen.append((f, text))
    calls = [0]

    def cancel():
        calls[0] += 1
        return any(t == 'Lidar and radar coverage' for _, t in seen)
    assert RQ.run(p, Engine(), progress=prog, cancel=cancel) is None
    assert seen and all(0.0 <= f <= 1.0 for f, _ in seen)


# ======================================================================= the page
def _open_page(win):
    from vantage.ui.workspaces.requirements_ws import RequirementsPage
    if 'requirements' not in win._page_widgets:
        pg = RequirementsPage(win.ctx)
        win.pages.addWidget(pg)
        win._page_widgets['requirements'] = pg
    win.set_page('requirements')
    return win._page_widgets['requirements']


def test_page_add_run_undo_export(app, win, tmp_path):
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest
    st = win.store
    rig = next(v for v in st.project.variants if v.sensors)
    st.edit('variants', lambda q: setattr(q, 'active_uid', rig.uid), 'Data rig')
    settle(app, win)
    pg = _open_page(win)
    pump(app, 0.2)
    assert pg.isVisible() and pg.table.rowCount() == 0 and pg.empty.isVisible()
    assert not pg.b_run.isEnabled()
    # a new requirement written in the editor: the shortest perception range of a pedestrian, ≥ 40 m
    QTest.mouseClick(pg.b_new, Qt.MouseButton.LeftButton)
    pg.ed_metric.setCurrentIndex(pg.ed_metric.findData('perception.range'))
    pump(app, 0.05)
    assert set(pg.param_boxes) == {'cls', 'sector'} and pg.ed_unit.text() == 'm'
    pg.param_boxes['cls'].setCurrentIndex(pg.param_boxes['cls'].findData('Pedestrian'))
    QTest.mouseClick(pg.ed_target.lineEdit(), Qt.MouseButton.LeftButton)
    QTest.keyClick(pg.ed_target.lineEdit(), Qt.Key.Key_A, Qt.KeyboardModifier.ControlModifier)
    QTest.keyClicks(pg.ed_target.lineEdit(), '40')
    QTest.keyClicks(pg.ed_text, 'Pedestrians resolved to 40 m in every sector')
    n = st.undo.count()
    QTest.mouseClick(pg.b_save, Qt.MouseButton.LeftButton)
    pump(app, 0.2)
    assert st.undo.count() == n + 1
    items = RQ.settings_of(st.project).items
    assert len(items) == 1 and items[0]['target'] == 40.0 and items[0]['params']['cls'] == 'Pedestrian'
    assert pg.table.rowCount() == 1 and pg.table.item(0, 6).text() == 'Not run'
    # a cheap one from the templates: evaluated at once
    k = next(i for i, t in enumerate(RQ.templates(st.project)) if t['metric'] == 'recording.hours_per_set')
    pg.add_template(k)
    pump(app, 0.2)
    assert pg.table.rowCount() == 2 and pg.table.item(1, 6).text() == 'Fail'      # 2 h on a disk set, not 8
    assert pg.tiles['verdict'].value == 'Fail'
    # run: the perception range is computed in a background job
    assert pg.b_run.isEnabled()
    QTest.mouseClick(pg.b_run, Qt.MouseButton.LeftButton)
    _wait(app, lambda: not pg.busy and pg.full is not None, 120)
    pump(app, 0.2)
    assert pg.table.item(0, 6).text() == 'Pass'
    assert 'perception' in pg.full.studies and float(pg.table.item(0, 4).text().split()[0]) > 40
    # the evidence of the selected row, and the study's page
    pg.table.selectRow(0)
    pump(app, 0.1)
    assert pg.sel_id == items[0]['id'] and 'Perception range' in pg.ev_head.text()
    assert pg.ev_table.rowCount() > 0 and pg.b_open.isEnabled()
    # an edit of the target in the editor: compared again at once, no run needed
    pg.ed_target.set_value(500.0)
    QTest.mouseClick(pg.b_save, Qt.MouseButton.LeftButton)
    pump(app, 0.2)
    assert RQ.settings_of(st.project).items[0]['target'] == 500.0
    assert pg.table.item(0, 6).text() == 'Fail' and not pg.res.items[0].stale
    st.undo.undo()
    pump(app, 0.2)
    assert RQ.settings_of(st.project).items[0]['target'] == 40.0 and pg.table.item(0, 6).text() == 'Pass'
    # export what is shown
    paths = pg.export(str(tmp_path))
    assert [os.path.basename(x) for x in paths] == ['requirements.json', 'requirements.csv', 'review_pack.pdf']
    doc = json.loads((tmp_path / 'requirements.json').read_text())
    assert [r['status'] for r in doc['requirements']] == ['pass', 'fail'] and doc['verdict'] == 'fail'
    assert (tmp_path / 'review_pack.pdf').read_bytes().startswith(b'%PDF')
    # the design changes: the study value goes stale, the cheap one follows
    c = st.variant.by_tag('F')
    st.set_cam(c.uid, yaw=c.yaw + 10.0)
    pump(app, 0.2)
    assert pg.res.items[0].stale and pg.table.item(0, 4).text().endswith('*')
    st.undo.undo()
    pump(app, 0.2)
    assert not pg.res.items[0].stale
    # from the plan, then the brief (review first, nothing added until accepted)
    n_before = len(RQ.settings_of(st.project).items)
    ids = pg.add_from_plan()
    assert ids and len(RQ.settings_of(st.project).items) == n_before + len(ids)
    st.undo.undo()
    pump(app, 0.2)
    assert len(RQ.settings_of(st.project).items) == n_before
    st.edit('analysis', lambda q: setattr(q.analysis, 'assistant', {'brief': 'Cyclists to 30 m to the rear. '
                                                                             'Uplink below 50 %.'}), 'brief')
    props = pg.suggest_from_brief()
    assert len(props) == 2 and pg.prop_box.isVisible()
    assert len(RQ.settings_of(st.project).items) == n_before
    pg.prop_checks[1].setChecked(False)
    QTest.mouseClick(pg.b_accept, Qt.MouseButton.LeftButton)
    pump(app, 0.2)
    items = RQ.settings_of(st.project).items
    assert len(items) == n_before + 1 and items[-1]['params'] == {'cls': 'Cyclist', 'sector': 'rear'}
    assert not pg.prop_box.isVisible()
    # proposals from the assistant: only the valid ones, for review
    assert pg.show_proposals([{'metric': 'timing.blur_px', 'target': 2, 'text': 'Blur at most 2 px'},
                              {'metric': 'nope', 'target': 1}, {'metric': 'timing.blur_px', 'target': 'x'}]) == 1
    assert pg.prop_box.isVisible() and len(RQ.settings_of(st.project).items) == n_before + 1
    QTest.mouseClick(pg.b_dismiss, Qt.MouseButton.LeftButton)
    pump(app, 0.1)
    assert not pg.prop_box.isVisible() and len(RQ.settings_of(st.project).items) == n_before + 1
    # delete, undo
    pg.select(items[-1]['id'])
    QTest.mouseClick(pg.b_del, Qt.MouseButton.LeftButton)
    pump(app, 0.2)
    assert len(RQ.settings_of(st.project).items) == n_before
    st.undo.undo()
    pump(app, 0.2)
    assert len(RQ.settings_of(st.project).items) == n_before + 1
    # an invalid requirement is refused with the reason
    QTest.mouseClick(pg.b_new, Qt.MouseButton.LeftButton)
    pg.ed_metric.setCurrentIndex(pg.ed_metric.findData('sensor_coverage.lidar_blocked_pct'))
    pg.param_boxes['lidar'].addItem('L9', 'L9')
    pg.param_boxes['lidar'].setCurrentIndex(pg.param_boxes['lidar'].count() - 1)
    n = st.undo.count()
    assert pg.save_editor() is None and "'L9'" in pg.ed_msg.text() and st.undo.count() == n
    # the project replaced while the page shows
    win.store.set_project(samples.tigor_sample_project())
    pump(app, 0.3)
    assert pg.full is None and pg.table.rowCount() == 0
