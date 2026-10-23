"""What the assistant can look at and suggest: tools over a snapshot of the project and its analysis.

Every tool returns plain JSON in metres, degrees, m² and px/cm, in the ISO 8855 vehicle frame (x forward, y left,
z up, origin at the rear-axle centre on the ground). Only text summaries leave the machine: camera poses, lens and
module names, analysis figures. Never the vehicle model, images or files.

The model cannot change the project. simulate_changes runs the analysis on a modified copy; propose_changes does the
same and hands the result to the UI as a Proposal, which the user applies (one undo step) or discards."""
from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass, field

import numpy as np

from ..core.analysis import camera_frames
from ..core.lens import resolve
from ..core.project import mirror_props

POSE = ('x', 'y', 'z', 'yaw', 'pitch', 'roll')
DELTAS = {'dx': 'x', 'dy': 'y', 'dz': 'z', 'dyaw': 'yaw', 'dpitch': 'pitch', 'droll': 'roll'}
METRICS = [  # key, label, unit, higher is better
    ('blind_m2', 'Blind ground', 'm²', False), ('covered_pct', 'Covered', '%', True),
    ('multi_pct', 'Seen by ≥ 2 cameras', '%', True), ('min_density', 'Minimum density', 'px/cm', True),
    ('min_ok_pct', 'Above minimum density', '%', True), ('in_focus_pct', 'In focus', '%', True)]
MAX_CHARS = 14000


@dataclass
class ToolContext:
    """Everything the tools read, gathered on the UI thread when a question is asked."""
    project: object                      # a snapshot (Project.snapshot())
    result: object | None                # AnalysisResult of the live project (may be a run behind)
    engine: object                       # an engine safe to use from the assistant's thread (Engine.share())
    issues: list = field(default_factory=list)
    mounts: dict = field(default_factory=dict)
    worst: tuple | None = None
    stitched: object | None = None       # StitchedObjects
    advice: object | None = None
    volume: object | None = None         # VolumeResult
    guidelines: dict | None = None
    selection: list = field(default_factory=list)   # selected camera tags
    page: str = ''
    revision: int = 0
    brief: str = ''                      # the user's project brief (requirements, notes)
    agent: bool = False                  # working towards a goal on its own (bigger budget, a plan)
    on_proposal: object = None           # callable(Proposal)
    on_plan: object = None               # callable(list of {'step', 'status'})
    on_ui: object = None                 # callable(dict): show something in the app (runs on the UI thread)
    on_status: object = None             # callable(str): progress of a long tool
    cancel: object = None                # threading.Event
    cache: dict = field(default_factory=dict)
    rig_health: dict | None = None       # the Live rig page's snapshot while it streams (core.rig_health)
    drive_log: dict | None = None        # the Drive-log check page's last result (core.drivelog.summary)
    _proposals: int = 0

    def status(self, text):
        if self.on_status is not None:
            self.on_status(text)

    def cancelled(self):
        return self.cancel is not None and self.cancel.is_set()


@dataclass
class Proposal:
    id: int
    title: str
    rationale: str
    targets: dict                        # camera uid -> {prop: value} (partners' mirrored values included)
    rows: list                           # [{'kind', 'camera'/'label', 'slot', 'prop', 'before', 'after', 'unit', ...}]
    effects: list                        # [{'key', 'label', 'unit', 'before', 'after', 'better'}]
    issues_before: int
    issues_after: int
    revision: int
    before: dict                         # camera uid -> {prop: value} as it was when proposed (staleness check)
    seams: dict = field(default_factory=dict)            # 'uidA|uidB' -> angle deg
    seams_before: dict = field(default_factory=dict)
    settings: dict = field(default_factory=dict)         # 'min_density' | 'target_density' | 'roi_distance' -> value
    settings_before: dict = field(default_factory=dict)
    variant_uid: str = ''


# =========================================================================== helpers
def _r(v, n=3):
    if v is None:
        return None
    try:
        v = float(v)
    except (TypeError, ValueError):
        return v
    if not math.isfinite(v):
        return None
    return round(v, n)


def _cams(p, visible_only=True):
    return [c for c in p.variant.cameras if c.visible or not visible_only]


def _by_tag(p, tag):
    t = str(tag or '').strip().lower()
    for c in p.variant.cameras:
        if c.tag.lower() == t:
            return c
    for c in p.variant.cameras:          # a name instead of a tag
        if c.name.lower() == t:
            return c
    return None


def _tags(p):
    return {c.uid: c.tag for c in p.variant.cameras}


def _fov(intr):
    """Horizontal and vertical field of view (degrees) inside the image circle."""
    lim = intr.max_theta
    _, th = intr.unproject(np.array([0.0, float(intr.width), intr.cx, intr.cx]),
                           np.array([intr.cy, intr.cy, 0.0, float(intr.height)]))
    th = np.minimum(np.asarray(th, float), lim)
    return math.degrees(th[0] + th[1]), math.degrees(th[2] + th[3])


def _vehicle_box(tc):
    vg = tc.engine.vehicle_geom(tc.project)
    if vg is None:
        return None, None
    lo, hi = vg.bbox
    return np.asarray(lo, float), np.asarray(hi, float)


def _where(tc, x, y):
    """'front-left corner', 'rear', 'left side', 'under the body' ..."""
    lo, hi = _vehicle_box(tc)
    if lo is None:
        return ''
    fx = 'front' if x > hi[0] else 'rear' if x < lo[0] else ''
    sy = 'left' if y > hi[1] else 'right' if y < lo[1] else ''
    if fx and sy:
        return f'{fx}-{sy} corner'
    if fx:
        return fx
    if sy:
        return f'{sy} side'
    return 'under the body'


def dump(obj) -> str:
    """Compact JSON, cut down if it would flood the conversation."""
    s = json.dumps(obj, separators=(',', ':'), ensure_ascii=False, default=str)
    if len(s) <= MAX_CHARS:
        return s
    if isinstance(obj, dict):
        o = dict(obj)
        for k, v in sorted(o.items(), key=lambda kv: -len(json.dumps(kv[1], default=str))):
            if isinstance(v, list) and len(v) > 4:
                o[k] = v[:max(4, len(v) // 3)] + [f'… {len(v) - max(4, len(v) // 3)} more']
                s = json.dumps(o, separators=(',', ':'), ensure_ascii=False, default=str)
                if len(s) <= MAX_CHARS:
                    return s
    return s[:MAX_CHARS] + '…(cut)'


# =========================================================================== read tools
def get_overview(tc, **_):
    p = tc.project
    lo, hi = _vehicle_box(tc)
    wc = p.vehicle.wheel_centres or {}
    wb = abs(wc['FL'][0] - wc['RL'][0]) if 'FL' in wc and 'RL' in wc else None
    track = abs(wc['FL'][1] - wc['FR'][1]) if 'FL' in wc and 'FR' in wc else None
    tags = _tags(p)
    cams = []
    for c in p.variant.cameras:
        lens, mod = p.lens_for(c), p.module_for(c)
        intr = resolve(mod, lens)
        h, v = _fov(intr)
        cams.append({'tag': c.tag, 'name': c.name, 'visible': c.visible, 'locked': c.locked,
                     'x': _r(c.x), 'y': _r(c.y), 'z': _r(c.z), 'yaw': _r(c.yaw, 1), 'pitch': _r(c.pitch, 1),
                     'roll': _r(c.roll, 1), 'module': mod.name, 'lens': lens.name, 'projection': intr.projection,
                     'hfov_deg': _r(h, 1), 'vfov_deg': _r(v, 1), 'unit_calibration': bool(c.calib),
                     'mirrored_with': tags.get(c.link_uid) or None, 'port': c.port or None})
    a = p.analysis
    res = tc.result
    return {
        'project': p.name, 'variant': p.variant.name, 'variants': [v.name for v in p.variants],
        'vehicle': {'name': p.vehicle.name,
                    'length_m': _r(hi[0] - lo[0]) if lo is not None else None,
                    'width_m': _r(hi[1] - lo[1]) if lo is not None else None,
                    'height_m': _r(hi[2] - lo[2]) if lo is not None else None,
                    'front_x': _r(hi[0]) if lo is not None else None, 'rear_x': _r(lo[0]) if lo is not None else None,
                    'wheelbase_m': _r(wb), 'track_m': _r(track)},
        'cameras': cams,
        'rig': [{'name': q.name, 'kind': q.kind, 'x': _r(q.x), 'y': _r(q.y), 'z': _r(q.z),
                 'size_m': {'length': _r(q.length), 'width': _r(q.width), 'height': _r(q.height)} if q.kind == 'box' else
                 {'radius': _r(q.radius), ('height' if q.kind == 'cylinder' else 'length'):
                  _r(q.height if q.kind == 'cylinder' else q.length)},
                 'yaw': _r(q.yaw, 1), 'occludes': q.occludes, 'mountable': q.mountable, 'visible': q.visible}
                for q in getattr(p, 'rig', ())],
        'sensors': [_sensor_row(p, s) for s in p.variant.sensors],
        'roi': {'mode': p.roi.mode, 'distance_m': _r(p.roi.distance), 'measured_from': p.roi.measured_from},
        'analysis': {'ground_cell_m': _r(a.cell_m), 'min_density_px_cm': _r(a.min_density, 2),
                     'target_density_px_cm': _r(a.target_density, 2), 'evaluated_at': a.obstacle_mode,
                     'height_m': _r(res.z) if res is not None else None},
        'selected_cameras': tc.selection, 'open_page': tc.page,
        'frame': 'ISO 8855: x forward, y left, z up; origin rear-axle centre on the ground; yaw + left, pitch + down',
    }


def _sensor_row(p, s) -> dict:
    """A lidar, radar, GNSS antenna or IMU for the model: pose, product and field (no network addresses)."""
    m = p.sensor_model_for(s)
    d = {'tag': s.tag, 'kind': s.kind, 'model': f'{m.vendor} {m.name}'.strip(), 'visible': s.visible,
         'x': _r(s.x), 'y': _r(s.y), 'z': _r(s.z), 'yaw': _r(s.yaw, 1), 'pitch': _r(s.pitch, 1), 'roll': _r(s.roll, 1)}
    if m.is_lidar:
        lo, hi = m.v_fov()
        d.update(channels=m.n_channels(), elevation_deg=[_r(lo, 2), _r(hi, 2)], horizontal_fov_deg=_r(m.h_fov, 1),
                 azimuth_step_deg=_r(m.h_res, 4), rate_hz=_r(m.rate_hz, 2), range_10pct_m=_r(m.range_m, 1),
                 points_per_s=int(m.pps()), scan=m.scan)
        if m.spinning:
            d.update(rotation=m.rotation, sync_phase_deg=_r(s.phase_deg, 1))
    elif m.kind == 'gnss':
        d.update(elevation_mask_deg=_r(m.mask_deg, 1), position='phase centre')
    elif m.kind == 'imu':
        d.update(rate_hz=_r(m.rate_hz, 1), antenna_inputs=m.antennas, position='centre of the IMU axes')
        if m.heading_sigma_deg > 0:
            d.update(heading_sigma_deg=_r(m.heading_sigma_deg, 3), heading_baseline_m=_r(m.heading_baseline_m, 2))
        if m.pos_sigma_cm > 0:
            d['rtk_position_sigma_cm'] = _r(m.pos_sigma_cm, 2)
        if m.attitude_sigma_deg > 0:
            d['roll_pitch_sigma_deg'] = _r(m.attitude_sigma_deg, 4)
    else:
        d['zones'] = [{'range_car_m': _r(r, 1), 'azimuth_half_deg': _r(a, 1), 'elevation_half_deg': _r(e, 1)}
                      for r, a, e in m.zones]
    return d


def get_coverage(tc, **_):
    res = tc.result
    if res is None:
        return {'error': 'No analysis result yet (the vehicle or the ROI may be missing).'}
    p = tc.project
    tags = _tags(p)
    m = res.metrics
    per = []
    for ce in res.cams:
        d = res.per_cam.get(ce.uid, {})
        per.append({'camera': ce.tag, 'seen_m2': _r(d.get('seen_m2'), 2), 'only_this_camera_m2': _r(d.get('unique_m2'), 2),
                    'best_camera_m2': _r(d.get('best_m2'), 2), 'min_density_where_best': _r(d.get('min_dens'), 2),
                    'blind_if_removed_m2': _r(d.get('if_removed_blind'), 2),
                    'below_min_if_removed_m2': _r(d.get('if_removed_below'), 2),
                    'overlap_m2': {tags.get(u, u): _r(v, 2) for u, v in sorted(d.get('overlaps', {}).items(),
                                                                                 key=lambda kv: -kv[1])[:4]}})
    out = {'roi_area_m2': _r(m.get('roi_area'), 2), 'blind_m2': _r(m.get('blind_m2'), 3),
           'covered_pct': _r(m.get('covered_pct'), 2), 'seen_by_2_or_more_pct': _r(m.get('multi_pct'), 1),
           'min_density_px_cm': _r(m.get('min_density'), 2), 'in_focus_pct': _r(m.get('in_focus_pct'), 1),
           'above_min_density_pct': _r(m.get('min_ok_pct'), 1), 'above_target_density_pct': _r(m.get('target_ok_pct'), 1),
           'evaluated_height_m': _r(m.get('z')), 'cell_m': _r(m.get('cell')), 'per_camera': per}
    if tc.worst and tc.worst[1]:
        b, uid, pert = tc.worst
        out['worst_single_camera_error'] = {'blind_m2': _r(b, 3), 'camera': tags.get(uid, uid), 'perturbation': str(pert),
                                            'tolerance': {'deg': p.analysis.tol_deg, 'mm': p.analysis.tol_mm}}
    return out


def get_issues(tc, **_):
    tags = _tags(tc.project)
    return {'issues': [{'severity': i.severity, 'message': i.message, 'objects': i.objects,
                        'cameras': [tags.get(u, u) for u in i.uids],
                        'where': [_r(v) for v in i.where] if i.where else None} for i in tc.issues],
            'count': len(tc.issues)}


def get_camera(tc, camera='', **_):
    p = tc.project
    c = _by_tag(p, camera)
    if c is None:
        return {'error': f'No camera {camera!r}. Cameras: {", ".join(x.tag for x in p.variant.cameras)}.'}
    ov = next(x for x in get_overview(tc)['cameras'] if x['tag'] == c.tag)
    intr = resolve(p.module_for(c), p.lens_for(c))
    lens, mod = p.lens_for(c), p.module_for(c)
    out = {**ov, 'sensor': {'width_px': intr.width, 'height_px': intr.height, 'pixel_um': _r(mod.pixel_um, 2)},
           'lens_model': {'focal_mm': _r(intr.f_mm, 3), 'max_half_angle_deg': _r(math.degrees(intr.max_theta), 1),
                          'aspect_fy_fx': _r(intr.aspect, 4), 'focus_m': [_r(intr.focus_near, 2), _r(intr.focus_far, 2)]},
           'library_lens_id': lens.id}
    mi = tc.mounts.get(c.uid)
    if mi is not None:
        out['mount'] = {'surface': mi.surface, 'zone': mi.zone_kind or None, 'gap_to_body_mm': _r(mi.gap_mm, 1),
                        'edge_clearance_mm': _r(mi.edge_mm, 1), 'off_the_body': bool(mi.floating)}
    res = tc.result
    if res is not None:
        cov = get_coverage(tc)
        out['coverage'] = next((x for x in cov.get('per_camera', []) if x['camera'] == c.tag), None)
        tags = _tags(p)
        out['seams'] = [{'with': tags.get(s.b if s.a == c.uid else s.a), 'angle_deg': _r(s.angle, 1),
                         'density_px_cm': _r(s.density, 2)} for s in res.seams if c.uid in (s.a, s.b)]
        ce = next((e for e in res.cams if e.uid == c.uid), None)
        if ce is not None and ce.vis.any():
            g = res.grid
            jj, ii = np.nonzero(ce.vis & g.roi)
            if len(jj):
                r = np.hypot(g.xs[ii] - c.x, g.ys[jj] - c.y)
                out['ground_reach_m'] = {'nearest': _r(r.min(), 2), 'farthest_in_roi': _r(r.max(), 2)}
    out['issues'] = [i.message for i in tc.issues if c.uid in i.uids]
    return out


def _blocker(vg, o, d) -> str:
    """What the ray meets first: 'the body' or a named rig part."""
    m = vg.occ_mesh
    _, f = m.raycast(o[None], d[None])
    part = m.parts[int(m.face_part[f[0]])] if f[0] >= 0 and m.parts else None
    return f'the rig part "{part.name}"' if part is not None and part.material == 'rig' else 'the body'


def _explain(tc, P):
    p, eng = tc.project, tc.engine
    vg = eng.vehicle_geom(p)
    rows = []
    for c in _cams(p):
        intr = resolve(p.module_for(c), p.lens_for(c))
        _, W2O = camera_frames(c)
        d = P - c.pos()
        dist = float(np.linalg.norm(d))
        opt = W2O @ d
        u, v, th, ok = intr.project_dirs(opt[None])
        u, v, th = float(u[0]), float(v[0]), float(th[0])
        row = {'camera': c.tag, 'range_m': _r(dist, 2), 'off_axis_deg': _r(math.degrees(th), 1)}
        if th > intr.max_theta:
            row.update(seen=False, reason='outside the lens field of view',
                       detail=f'{math.degrees(th):.0f}° off axis; this lens reaches {math.degrees(intr.max_theta):.0f}°')
        elif intr.projection == 'rectilinear' and opt[2] <= 1e-6:
            row.update(seen=False, reason='behind the camera')
        elif not (0 <= u <= intr.width and 0 <= v <= intr.height):
            edge = ('left' if u < 0 else 'right') if not 0 <= u <= intr.width else ('top' if v < 0 else 'bottom')
            row.update(seen=False, reason='outside the image frame', detail=f'past the {edge} edge of the sensor')
        else:
            depth = np.inf
            if vg is not None:
                depth = float(eng.cube(c.pos()).lookup(d[None])[0][0])
            if depth < dist - (0.02 + 0.01 * dist):
                row.update(seen=False, reason='hidden by the body',
                           detail=f'{_blocker(vg, c.pos(), d / dist)} is {depth:.2f} m along the ray, '
                                  f'the point {dist:.2f} m')
            else:
                dens = float(intr.density_px_per_cm(th, dist))
                row.update(seen=True, density_px_cm=_r(dens, 2), pixel=[_r(u, 0), _r(v, 0)],
                           in_focus=bool(intr.focus_near <= dist <= intr.focus_far))
        rows.append(row)
    return rows


def explain_point(tc, x=0.0, y=0.0, z=0.0, **_):
    try:
        P = np.array([float(x), float(y), float(z)])
    except (TypeError, ValueError):
        return {'error': 'x, y and z must be numbers (metres).'}
    if not np.isfinite(P).all():
        return {'error': 'x, y and z must be finite.'}
    rows = _explain(tc, P)
    seen = [r for r in rows if r.get('seen')]
    out = {'point': [_r(v) for v in P], 'where': _where(tc, P[0], P[1]), 'seen_by': [r['camera'] for r in seen],
           'cameras': rows}
    res = tc.result
    if res is not None and abs(P[2] - res.z) < 1e-6:
        cc = res.grid.cell_of(P[0], P[1])
        if cc is not None:
            i, j = cc
            out['in_roi'] = bool(res.grid.roi[j, i])
            out['distance_to_body_m'] = _r(res.grid.dist[j, i], 2)
    if seen:
        best = max(seen, key=lambda r: r['density_px_cm'])
        out['best_camera'] = best['camera']
        out['best_density_px_cm'] = best['density_px_cm']
    return out


def _regions(tc, mask, min_area, limit):
    from scipy import ndimage
    res = tc.result
    g = res.grid
    a = g.cell ** 2
    lab, n = ndimage.label(mask, structure=np.ones((3, 3)))
    if n == 0:
        return []
    sizes = np.bincount(lab.ravel())[1:]
    order = np.argsort(-sizes)
    out = []
    for k in order[:max(limit * 3, limit)]:
        area = float(sizes[k] * a)
        if area < min_area or len(out) >= limit:
            continue
        jj, ii = np.nonzero(lab == k + 1)
        xs, ys = g.xs[ii], g.ys[jj]
        cx, cy = float(xs.mean()), float(ys.mean())
        q = int(np.argmin((xs - cx) ** 2 + (ys - cy) ** 2))       # a cell of the region nearest its centre
        px, py = float(xs[q]), float(ys[q])
        why = _explain(tc, np.array([px, py, float(res.z)]))
        out.append({'area_m2': _r(area, 3), 'centre': [_r(cx, 2), _r(cy, 2)], 'where': _where(tc, cx, cy),
                    'extent': {'x': [_r(xs.min(), 2), _r(xs.max(), 2)], 'y': [_r(ys.min(), 2), _r(ys.max(), 2)]},
                    'distance_to_body_m': [_r(g.dist[jj, ii].min(), 2), _r(g.dist[jj, ii].max(), 2)],
                    'sample_point': [_r(px, 2), _r(py, 2)],
                    'per_camera_at_sample': {r['camera']: (f"seen, {r['density_px_cm']} px/cm" if r.get('seen')
                                                           else r['reason']) for r in why}})
    return out


def find_blind_regions(tc, min_area_m2=0.002, **_):
    res = tc.result
    if res is None:
        return {'error': 'No analysis result yet.'}
    g = res.grid
    try:
        min_area = max(0.0, float(min_area_m2))
    except (TypeError, ValueError):
        min_area = 0.002
    blind = g.roi & (res.count == 0)
    low = g.roi & (res.count > 0) & (res.best_dens < tc.project.analysis.min_density)
    return {'evaluated_height_m': _r(res.z), 'blind_total_m2': _r(blind.sum() * g.cell ** 2, 3),
            'blind_regions': _regions(tc, blind, min_area, 8),
            'below_min_density_total_m2': _r(low.sum() * g.cell ** 2, 3),
            'below_min_density_regions': _regions(tc, low, max(min_area, 0.01), 5),
            'min_density_px_cm': tc.project.analysis.min_density}


def get_seams(tc, **_):
    res = tc.result
    if res is None:
        return {'error': 'No analysis result yet.'}
    tags = _tags(tc.project)
    sv = {}
    if tc.stitched is not None:
        sv = {(s['a'], s['b']): s for s in tc.stitched.seams}
    adv = {}
    if tc.advice is not None:
        adv = {(s.a, s.b): s for s in tc.advice.seams}
    out = []
    for s in res.seams:
        row = {'between': [tags.get(s.a, s.a), tags.get(s.b, s.b)], 'angle_deg': _r(s.angle, 1),
               'placed': 'automatic' if s.auto else 'by hand', 'density_px_cm': _r(s.density, 2)}
        if (s.a, s.b) in sv:
            row['object_vanishes_m2'] = _r(sv[(s.a, s.b)]['vanish_m2'], 3)
        if (s.a, s.b) in adv and adv[(s.a, s.b)].moved:
            row['advised_angle_deg'] = _r(adv[(s.a, s.b)].best, 1)
        out.append(row)
    return {'seams': out, 'azimuth': 'degrees about the body centre, counter-clockwise from +x (forward)',
            'centre': [_r(v) for v in res.centre]}


def get_studies(tc, **_):
    tags = _tags(tc.project)
    out = {}
    s = tc.stitched
    if s is not None:
        m = s.metrics
        out['objects_in_stitched_view'] = {
            'object_height_m': _r(s.settings.height, 2), 'vanish_threshold': _r(s.settings.threshold, 2),
            'vanishes_at_seams_m2': _r(m.get('vanish_m2'), 3), 'cut_off_m2': _r(m.get('cut_m2'), 3),
            'not_seen_m2': _r(m.get('blind_m2'), 3), 'mean_shown_fraction': _r(m.get('shown_mean'), 3),
            'hotspots': [{'centre': [_r(h.x, 2), _r(h.y, 2)], 'area_m2': _r(h.area_m2, 3), 'where': _where(tc, h.x, h.y),
                          'seam': list(s.seams[h.seam]['tags']) if 0 <= h.seam < len(s.seams) else None,
                          'seen_raw_by': h.seen_by} for h in s.hotspots[:6]],
            'meaning': 'the stitched top view projects everything onto the ground; a standing object vanishes where '
                       'both cameras smear it into the other camera\'s sector, and is cut off where its smear leaves '
                       'the view'}
        if tc.advice is not None and tc.advice.changes:
            out['objects_in_stitched_view']['seam_advice'] = {
                'vanish_now_m2': _r(tc.advice.vanish_now, 3), 'vanish_after_m2': _r(tc.advice.vanish_after, 3),
                'moves': [{'seam': [tags.get(x.a), tags.get(x.b)], 'from_deg': _r(x.angle, 1), 'to_deg': _r(x.best, 1)}
                          for x in tc.advice.seams if x.moved]}
    else:
        out['objects_in_stitched_view'] = 'not computed in the app yet: call run_study(study="stitched_objects")'
    v = tc.volume
    if v is not None:
        m = v.metrics
        out['blind_volume'] = {
            'blind_m3': _r(m.get('blind_m3'), 2), 'blind_pct_of_space': _r(m.get('blind_pct'), 1),
            'space_height_m': _r(m.get('height'), 2), 'ground_blind_m2': _r(m.get('ground_blind_m2'), 3),
            'covered_up_to_m': _r(m.get('covered_to_m'), 2), 'never_seen_at_any_height_m2': _r(m.get('never_seen_m2'), 3),
            'objects': {n: {'height_m': _r(o.get('height'), 2), 'undetected_m2': _r(o.get('undetected_m2'), 3),
                            'hides_up_to_m_from_body': _r(o.get('reach_m'), 2)} for n, o in m.get('objects', {}).items()}}
    else:
        out['blind_volume'] = 'not run in the app yet: call run_study(study="blind_volume")'
    if tc.guidelines is not None:
        out['parking_guidelines'] = tc.guidelines
    if tc.worst and tc.worst[1]:
        b, uid, pert = tc.worst
        out['worst_single_camera_error'] = {'blind_m2': _r(b, 3), 'camera': tags.get(uid, uid), 'perturbation': str(pert)}
    return out


_URL_IN_TEXT = re.compile(r'[a-z][a-z0-9+.-]*://\S+', re.IGNORECASE)


def get_rig_health(tc, **_):
    h = tc.rig_health
    streams = getattr(tc.project, 'streams', {}) or {}
    if not h:
        from ..core.rig_health import describe
        return {'streaming': False,
                'note': 'not streaming: the health is measured on Calibrate ▸ Live rig… while it runs (Start)',
                'configured_streams': {c.tag: describe(streams.get(c.tag, '')) or 'none' for c in _cams(tc.project)}}
    keep = ('status', 'reasons', 'kind', 'fps', 'nominal_fps', 'jitter_ms', 'dropped', 'dropped_share', 'age_s', 'size',
            'expected_size', 'mean', 'clip_bright', 'clip_dark', 'sharpness', 'frozen_frames', 'drift_px', 'drift_checked')
    cams = {k: {f: (_r(v, 3) if isinstance(v, float) else v) for f, v in c.items() if f in keep}
            for k, c in h['cameras'].items()}
    for c in cams.values():                # a stream error names its URL: hosts and user names stay on this machine
        c['reasons'] = [_URL_IN_TEXT.sub('<stream URL>', r) for r in c.get('reasons', [])]
    rig = {k: (_r(v, 2) if isinstance(v, float) else v) for k, v in h['rig'].items()}
    return {'streaming': True, 'rig': rig, 'cameras': cams,
            'meaning': 'fps from frame arrival times; dropped = gaps longer than 1.5 frame intervals; sharpness = '
                       'variance of the Laplacian (low: blurred or covered lens); drift_px = median distance of the '
                       'planned body edge to the edge in the image, an indicator that a camera moved, not a '
                       'measurement (null with drift_checked: the body edge was not found in the image)'}


# =========================================================================== changes
def get_sync_plan(tc, **_):
    from ..core import timing as TM
    return {**TM.summary(TM.plan(tc.project)),
            'meaning': 'times in ms after the reference spinning lidar\'s sync pulse: when its sweep crosses each '
                       'camera\'s view, the trigger that centres the middle row\'s exposure on it, the misalignment of '
                       'each camera\'s trigger line (cameras with one sync letter fire together) and the vehicle\'s '
                       'travel in that time; suggested_groups is the best split over the available trigger lines. '
                       'Blur and skew are px for a static object at object_m while driving straight at speed_kmh. '
                       'The user applies groups and phases on the Sync and timing page.'}


def get_gnss_imu(tc, antenna=None, **_):
    from ..core import gnss as GN
    res = GN.run(tc.project, tc.engine)
    out = {**GN.summary(res),
           'meaning': 'per GNSS antenna the open share of the sky above its elevation mask (the body, the rig and the '
                      'other sensors traced; buildings not modelled), expected satellites (an even sky assumed), what '
                      'blocks it and near-field multipath; the dual-antenna heading baseline and its heading error; '
                      'the lever arms from the IMU to every antenna and sensor in mm (IMU frame); the 1-sigma position '
                      'error of every sensor\'s world pose; the user edits the settings on Analysis ▸ GNSS and IMU'}
    if antenna:
        a = res.antenna(str(antenna).strip())
        if a is None:
            out['error'] = f'no visible GNSS antenna {antenna}'
        else:
            out['antenna_detail'] = GN.to_doc(res)['antennas'][res.antennas.index(a)]
    return out


def get_sun_and_dirt(tc, lat=None, lon=None, start=None, end=None, hour_from=None, hour_to=None, **_):
    from ..core import exposure as XP
    over = {k: v for k, v in dict(lat=lat, lon=lon, start=start, end=end, hour_from=hour_from,
                                  hour_to=hour_to).items() if v is not None}
    st = XP.ExposureSettings.from_dict({**(tc.project.analysis.exposure or {}), **over})
    tc.status('Sun glare and dirt…')
    res = XP.run(tc.project, tc.engine, st, cancel=tc.cancelled)
    if res is None:
        return {'error': 'cancelled'}
    return {**XP.summary(res), 'meaning': (
        'sun_in_image_pct: share of all driving time in the window with the sun inside the camera image (lens model, '
        'not hidden by the body); near_image_pct: within the flare margin outside it; low_sun_ahead_pct: in the image '
        'and low. Lidars: share with the sun inside the field (an indicator of solar background). dirt_score 0-1 is a '
        'heuristic (wheel spray, rear wake, frontal impact, lens attitude, height); radars matter less. Clear sky, '
        'flat horizon.')}


def get_fleet(tc, **_):
    from ..core import fleet as FL
    return {**FL.summary(FL.run(tc.project)),
            'meaning': 'the same rig on many vehicles: each vehicle\'s latest measured sensor poses against the plan '
                       '(verdict ok / adjust / re-mount), sensors drifting over time or that jumped, vehicles that '
                       'differ from the fleet median, and sensors the whole fleet has built off the plan'}


def get_brackets(tc, **_):
    from ..core import brackets as BR
    return {**BR.summary(BR.plan(tc.project, tc.engine)),
            'meaning': 'per camera whether a 3D-printed bracket can hold it at its planned pose: the mount (flat base '
                       'or tube clamp on a rig tube), the housing face the plate holds, the standoff and the standoff a '
                       'bracket needs where the housing would go into the surface, cameras too far from any surface; '
                       'base size, height and plate angle where it can. Meshes and the image, wall and stiffness checks '
                       'come from Analysis ▸ Mount brackets (Generate)'}


def get_sample_dataset(tc, **_):
    from ..core import dataset as DSET
    st = DSET.DatasetSettings.from_dict(tc.project.analysis.dataset)
    est = DSET.estimate(tc.project, st)
    return {'settings': st.to_dict(), 'estimate': {k: est.get(k) for k in (
                'ok', 'reason', 'frames', 'samples', 'keyframe_hz', 'images', 'sweeps', 'scans', 'bytes', 'seconds',
                'notes')},
            'meaning': 'what a synthetic sample drive with these settings would record: frames per nuScenes channel '
                       'and bag topic, keyframes, approximate size and time; the user writes it on Export ▸ Sample '
                       'dataset (a nuScenes version folder and a ROS 2 bag). The scene is synthetic, for testing '
                       'pipelines, not for training'}


def get_drive_log_check(tc, **_):
    if not tc.drive_log:
        return {'checked': False, 'note': 'no drive log checked yet: Calibrate ▸ Drive-log check… (open a ROS 2 bag or '
                                         'a nuScenes folder, Run)'}
    return {'checked': True, **tc.drive_log,
            'meaning': 'a recorded drive against the plan: stream rates and drops, data rate against the recording '
                       'budget, camera trigger offsets against the lidar sweep (own error vs the lidar\'s phase), each '
                       'camera\'s rotation relative to the lidar from edge alignment (confidence, pinned-down axes), '
                       'IMU at rest and GNSS fix shares; stale = the plan changed since'}


def get_requirements(tc, run=False, **_):
    from ..core import requirements as RQ
    if run:
        tc.status('Evaluating the requirements…')
        res = RQ.run(tc.project, tc.engine, cancel=tc.cancelled)
        if res is None:
            return {'error': 'cancelled'}
    else:
        res = RQ.run(tc.project, None, cheap_only=True, issues_list=False)
    return {**RQ.summary(res), 'meaning': (
        'the project\'s requirements evaluated on the app\'s studies: value against target, status pass / fail / '
        'not_evaluated (with the reason) / not_run (a study metric: call again with run=true, takes seconds); the '
        'verdict passes when every must requirement passes; the user edits them on Analysis ▸ Requirements scorecard')}


def get_requirement_metrics(tc, **_):
    from ..core import requirements as RQ
    return RQ.catalogue(tc.project)


def propose_requirements(tc, items=None, text=None, **_):
    from ..core import requirements as RQ
    p = tc.project
    if items is None:                      # parse the brief (or the given text)
        items = [o['item'] for o in RQ.parse_brief(text if text is not None else tc.brief, p)]
    ok, bad = [], []
    for it in items or []:
        c = RQ.clean_item(it, p)
        good, why = RQ.validate_item(c, p) if c is not None else (False, 'not a mapping')
        if good:
            ok.append(c)
        else:
            bad.append({'item': it, 'error': why})
    if ok and tc.on_ui is not None:
        tc.on_ui({'action': 'propose_requirements', 'items': ok})
    return {'proposed': ok, 'rejected': bad, 'note': 'shown on Analysis ▸ Requirements scorecard for the user to tick '
                                                     'and add; nothing is added until they do'}


def _lidar_pair(tc, camera, lidar):
    from ..core import lidar_calib as LC
    st = LC.LidarCalibSettings.from_dict(tc.project.analysis.lidar_calib)
    return LC, LC.LidarCalibSettings.from_dict(dict(st.to_dict(), **{k: v for k, v in
                                                                      (('camera', camera), ('lidar', lidar)) if v}))


def plan_lidar_calibration(tc, camera=None, lidar=None, **_):
    LC, st = _lidar_pair(tc, camera, lidar)
    p = tc.project
    pairs = LC.pairs(p, engine=tc.engine)
    if not pairs:
        return {'error': 'no camera and lidar whose fields overlap in this variant'}
    try:
        cam, lid, share = LC.pick_pair(p, p.variant, st, tc.engine)
    except ValueError as e:
        return {'error': str(e), 'pairs': pairs[:8]}
    poses, notes = LC.plan_poses(p, cam, lid, st, engine=tc.engine)
    return {'pairs': [{k: v for k, v in q.items() if not k.endswith('uid')} for q in pairs[:8]],
            'camera': cam.tag, 'lidar': lid.tag, 'overlap_share': share, 'board': st.board_size(),
            'poses': [b.to_dict() for b in poses], 'notes': notes,
            'predicted_sigma': LC.predicted_sigma(p, cam, lid, poses, st) if poses else None,
            'meaning': 'the camera-lidar pairs ranked by how much of the camera image the lidar covers, and for the '
                       'chosen pair the planned checkerboard poses (in both fields, enough rings across each board) '
                       'with the 1-sigma uncertainty the calibration would reach; Calibrate ▸ Lidar–camera '
                       'calibration runs it'}


def calibrate_lidar_camera(tc, camera=None, lidar=None, **_):
    LC, st = _lidar_pair(tc, camera, lidar)
    tc.status('Lidar–camera calibration (synthetic captures)…')
    try:
        res = LC.run(tc.project, tc.engine, st, cancel=tc.cancelled)
    except ValueError as e:
        return {'error': str(e)}
    if res is None:
        return {'error': 'cancelled'}
    return {**LC.summary(res), 'meaning': (
        'a calibration of the lidar to the camera from simulated board captures of the plan (the settings\' injected '
        'mounting error, if any, is what it should find): verdict against the tolerance, the offset from the plan in '
        'the vehicle frame, the 1-sigma uncertainty and the directions the poses did not determine. Real captures are '
        'calibrated on the page or with batch --lidar-calib-folder')}


def get_power_budget(tc, **_):
    from ..core import power as PW
    return {**PW.summary(PW.run(tc.project)),
            'meaning': 'steady and peak power per device (library figures nominal, peak factors typical), the DC-DC '
                       'converter\'s load and input, the alternator margin while driving, the engine-off runtime on '
                       'the auxiliary battery, the worst circuit against its fuse, power-cable drops and the heat the '
                       'enclosure must shed; the user edits the settings on Analysis ▸ Power budget'}


def get_recording_budget(tc, **_):
    from ..core import recording as RB
    return {**RB.summary(RB.budget(tc.project)),
            'meaning': 'data rate per camera, lidar and radar as recorded (camera bits per pixel and the compressed '
                       'formats are the project\'s assumptions), each sensor link\'s load, the uplink to the recorder, '
                       'the sustained write speed needed with headroom, and hours of driving per disk set'}


def _norm_angle(a):
    a = (float(a) + 180.0) % 360.0 - 180.0
    return 180.0 if a == -180.0 else a


def _lens(p, name):
    t = str(name or '').strip().lower()
    for l in p.lenses.values():
        if l.id.lower() == t or l.name.lower() == t:
            return l
    hits = [l for l in p.lenses.values() if t and (t in l.name.lower() or t in l.id.lower())]
    return hits[0] if len(hits) == 1 else None


def apply_changes(p, changes):
    """Apply change requests to project p in place. Returns (rows, targets, before, errors)."""
    errors, rows = [], []
    targets, before = {}, {}
    if not isinstance(changes, list) or not changes:
        return rows, targets, before, ['changes must be a non-empty list']
    named = {str(ch.get('camera', '')).lower() for ch in changes if isinstance(ch, dict)}
    for ch in changes:
        if not isinstance(ch, dict):
            errors.append(f'not a change: {ch!r}')
            continue
        c = _by_tag(p, ch.get('camera'))
        if c is None:
            errors.append(f'no camera {ch.get("camera")!r}; cameras: {", ".join(x.tag for x in p.variant.cameras)}')
            continue
        if c.locked:
            errors.append(f'{c.tag} is locked; the user has to unlock it first')
            continue
        props = {}
        try:
            for k in POSE:
                if ch.get(k) is not None:
                    props[k] = float(ch[k])
            for dk, k in DELTAS.items():
                if ch.get(dk) is not None:
                    props[k] = props.get(k, getattr(c, k)) + float(ch[dk])
        except (TypeError, ValueError):
            errors.append(f'{c.tag}: positions and angles must be numbers')
            continue
        if ch.get('lens') is not None:
            lens = _lens(p, ch['lens'])
            if lens is None:
                errors.append(f'{c.tag}: no single lens matches {ch["lens"]!r}; lenses: '
                              + ', '.join(sorted(l.name for l in p.lenses.values())))
                continue
            props['lens_id'] = lens.id
        if ch.get('visible') is not None:
            props['visible'] = bool(ch['visible'])
        if not props:
            errors.append(f'{c.tag}: nothing to change')
            continue
        bad = [k for k, v in props.items() if isinstance(v, float) and not math.isfinite(v)]
        if bad:
            errors.append(f'{c.tag}: {", ".join(bad)} not finite')
            continue
        for k in ('yaw', 'roll'):
            if k in props:
                props[k] = _norm_angle(props[k])
        if 'pitch' in props and not -90.0 <= props['pitch'] <= 90.0:
            errors.append(f'{c.tag}: pitch must stay within ±90°')
            continue
        if 'z' in props and not 0.05 <= props['z'] <= 5.0:
            errors.append(f'{c.tag}: height z must be 0.05–5 m')
            continue
        if any(k in props and abs(props[k]) > 20.0 for k in ('x', 'y')):
            errors.append(f'{c.tag}: x and y must be within 20 m of the origin')
            continue
        old = {k: getattr(c, k) for k in props}
        before.setdefault(c.uid, {}).update({k: v for k, v in old.items() if k not in before.get(c.uid, {})})
        for k, v in props.items():
            setattr(c, k, v)
        targets.setdefault(c.uid, {}).update(props)
        for k in props:
            rows.append(_row(p, c, k, old[k], props[k]))
        partner = p.variant.cam(c.link_uid) if c.link_uid else None
        if partner is not None and ch.get('mirror_partner', True) and partner.tag.lower() not in named:
            keys = [k for k in props if k != 'visible']
            if partner.locked:
                errors.append(f'{partner.tag} (mirrored with {c.tag}) is locked and was not moved')
            elif keys:
                pb = {k: getattr(partner, k) for k in keys}
                before.setdefault(partner.uid, {}).update({k: v for k, v in pb.items()
                                                           if k not in before.get(partner.uid, {})})
                mirror_props(c, partner, keys)
                targets.setdefault(partner.uid, {}).update({k: getattr(partner, k) for k in keys})
                for k in keys:
                    rows.append(_row(p, partner, k, pb[k], getattr(partner, k), mirrored_from=c.tag))
    return rows, targets, before, errors


def _row(p, c, k, a, b, mirrored_from=None):
    unit = 'm' if k in ('x', 'y', 'z') else '°' if k in ('yaw', 'pitch', 'roll') else ''
    if k == 'lens_id':
        a = p.lenses[a].name if a in p.lenses else a
        b = p.lenses[b].name if b in p.lenses else b
    return {'camera': c.tag, 'slot': c.slot, 'prop': 'lens' if k == 'lens_id' else k, 'before': a, 'after': b,
            'unit': unit, 'mirrored_from': mirrored_from}


def _evaluate(tc, p):
    from ..core import rules
    eng = tc.engine
    res = eng.analyse(p)
    if res is None:
        return None, None
    vg = eng.vehicle_geom(p)
    mounts = {}
    for c in p.variant.cameras:
        try:
            mounts[c.uid] = rules.mount_info(p, c, vg)
        except Exception:
            pass
    try:
        issues = rules.collect_issues(p, res, vg, mounts)
    except Exception:
        issues = []
    return res, issues


SETTINGS = {  # proposal key -> (label, unit, getter, setter, min, max)
    'min_density': ('Minimum density', 'px/cm', lambda p: p.analysis.min_density,
                    lambda p, v: setattr(p.analysis, 'min_density', v), 0.5, 50.0),
    'target_density': ('Target density', 'px/cm', lambda p: p.analysis.target_density,
                       lambda p, v: setattr(p.analysis, 'target_density', v), 0.5, 80.0),
    'roi_distance': ('ROI distance', 'm', lambda p: p.roi.distance, lambda p, v: setattr(p.roi, 'distance', v), 0.3, 10.0),
}


def _base(tc, cell=None):
    """The analysis of the snapshot itself (the live result can be a run behind)."""
    key = ('base', cell)
    if key not in tc.cache:
        tc.engine.vehicle_geom(tc.project)
        tc.cache[key] = tc.engine.analyse(tc.project, cell=cell)
    return tc.cache[key]


def _apply_seams(tc, p, seams):
    rows, targets, before, errors = [], {}, {}, []
    res = _base(tc)
    if res is None:
        return rows, targets, before, ['no analysis to take the seams from']
    tags = _tags(p)
    for sp in seams if isinstance(seams, list) else []:
        pair = [str(t).lower() for t in (sp.get('between') or [])] if isinstance(sp, dict) else []
        seam = next((x for x in res.seams if sorted([tags.get(x.a, '').lower(), tags.get(x.b, '').lower()]) == sorted(pair)),
                    None) if len(pair) == 2 else None
        if seam is None:
            errors.append(f'no seam between {sp.get("between") if isinstance(sp, dict) else sp}; seams: '
                          + ', '.join(f'{tags.get(x.a)}|{tags.get(x.b)}' for x in res.seams))
            continue
        try:
            ang = _norm_angle(float(sp.get('angle_deg')))
        except (TypeError, ValueError):
            errors.append('a seam angle must be a number (degrees about the body centre)')
            continue
        if abs((ang - seam.angle + 180) % 360 - 180) > 45:
            errors.append(f'{tags.get(seam.a)}|{tags.get(seam.b)}: move a seam by at most 45°')
            continue
        key = f'{seam.a}|{seam.b}'
        before[key] = p.variant.seams.get(key, seam.angle)
        p.variant.seams[key] = ang
        targets[key] = ang
        rows.append({'kind': 'seam', 'label': f'{tags.get(seam.a)} | {tags.get(seam.b)}', 'camera': '', 'slot': -1,
                     'prop': 'seam angle', 'before': float(seam.angle), 'after': ang, 'unit': '°', 'mirrored_from': None})
    return rows, targets, before, errors


def _apply_settings(p, settings):
    rows, targets, before, errors = [], {}, {}, []
    for k, v in (settings or {}).items() if isinstance(settings, dict) else []:
        if k not in SETTINGS:
            errors.append(f'unknown setting {k!r}; settings: {", ".join(SETTINGS)}')
            continue
        lab, unit, get, put, lo, hi = SETTINGS[k]
        try:
            v = float(v)
        except (TypeError, ValueError):
            errors.append(f'{k} must be a number')
            continue
        if not lo <= v <= hi:
            errors.append(f'{k} must be {lo:g}–{hi:g} {unit}')
            continue
        before[k] = get(p)
        put(p, v)
        targets[k] = v
        rows.append({'kind': 'setting', 'label': lab, 'camera': '', 'slot': -1, 'prop': lab, 'before': before[k],
                     'after': v, 'unit': unit, 'mirrored_from': None})
    if 'min_density' in targets or 'target_density' in targets:
        if p.analysis.target_density < p.analysis.min_density:
            errors.append('the target density must not be below the minimum density')
            return [], {}, {}, errors
    return rows, targets, before, errors


def _simulate(tc, changes=None, seams=None, settings=None):
    t0 = time.perf_counter()
    base = tc.project.snapshot()
    trial = tc.project.snapshot()
    rows, targets, before, errors = apply_changes(trial, changes) if changes else ([], {}, {}, [])
    for r in rows:
        r.setdefault('kind', 'camera')
    srows, stargets, sbefore, serr = _apply_seams(tc, trial, seams) if seams else ([], {}, {}, [])
    grows, gtargets, gbefore, gerr = _apply_settings(trial, settings) if settings else ([], {}, {}, [])
    errors += serr + gerr
    if not (targets or stargets or gtargets):
        return None, {'error': '; '.join(errors) or 'nothing to change: give changes, seams or settings'}
    r0, i0 = _evaluate(tc, base)
    r1, i1 = _evaluate(tc, trial)
    if r0 is None or r1 is None:
        return None, {'error': 'The analysis cannot run (no vehicle or ROI).', 'notes': errors}
    effects = []
    for key, lab, unit, up in METRICS:
        a, b = float(r0.metrics.get(key, 0.0)), float(r1.metrics.get(key, 0.0))
        effects.append({'key': key, 'label': lab, 'unit': unit, 'before': a, 'after': b,
                        'better': (b > a) == up if abs(b - a) > 1e-9 else None})
    tags = _tags(trial)
    per = []
    for ce in r1.cams:
        d0, d1 = r0.per_cam.get(ce.uid, {}), r1.per_cam.get(ce.uid, {})
        if ce.uid in targets:
            per.append({'camera': tags.get(ce.uid), 'seen_m2': [_r(d0.get('seen_m2'), 2), _r(d1.get('seen_m2'), 2)],
                        'only_this_camera_m2': [_r(d0.get('unique_m2'), 2), _r(d1.get('unique_m2'), 2)]})
    sev = lambda lst: {s: sum(1 for i in lst if i.severity == s) for s in ('critical', 'serious', 'warning')}
    allrows = rows + srows + grows
    report = {
        'changes': [{({'camera': 'camera', 'seam': 'seam'}.get(r.get('kind', 'camera'), 'setting')): r['camera'] or r['label'],
                     'property': r['prop'],
                     'from': _r(r['before'], 4) if not isinstance(r['before'], str) else r['before'],
                     'to': _r(r['after'], 4) if not isinstance(r['after'], str) else r['after'],
                     **({'mirrored_from': r['mirrored_from']} if r.get('mirrored_from') else {})} for r in allrows],
        'effect': {e['key']: {'before': _r(e['before'], 3), 'after': _r(e['after'], 3)} for e in effects},
        'cameras_changed': per,
        'issues': {'before': sev(i0), 'after': sev(i1),
                   'new': [i.message for i in i1 if i.message not in {j.message for j in i0}][:6],
                   'resolved': [i.message for i in i0 if i.message not in {j.message for j in i1}][:6]},
        'seconds': _r(time.perf_counter() - t0, 2)}
    if errors:
        report['notes'] = errors
    sim = {'rows': allrows, 'targets': targets, 'before': before, 'seams': stargets, 'seams_before': sbefore,
           'settings': gtargets, 'settings_before': gbefore, 'effects': effects, 'n0': len(i0), 'n1': len(i1)}
    return sim, report


def simulate_changes(tc, changes=None, seams=None, settings=None, **_):
    _, report = _simulate(tc, changes, seams, settings)
    return report


def propose_changes(tc, changes=None, seams=None, settings=None, title='', rationale='', **_):
    sim, report = _simulate(tc, changes, seams, settings)
    if sim is None:
        return report
    tc._proposals += 1
    prop = Proposal(tc._proposals, str(title or 'Proposed change')[:80], str(rationale or '')[:600], sim['targets'],
                    sim['rows'], sim['effects'], sim['n0'], sim['n1'], tc.revision, sim['before'],
                    seams=sim['seams'], seams_before=sim['seams_before'], settings=sim['settings'],
                    settings_before=sim['settings_before'], variant_uid=tc.project.active_uid)
    if tc.on_proposal is not None:
        tc.on_proposal(prop)
    return {'proposal': prop.id, 'status': 'shown to the user with its predicted effect; they decide whether to apply it',
            **report}


# =========================================================================== studies and search
def compare_variants(tc, variants=None, **_):
    p = tc.project
    wanted = variants if isinstance(variants, list) and variants else [v.name for v in p.variants]
    out = []
    for name in wanted[:8]:
        if tc.cancelled():
            break
        v = next((x for x in p.variants if x.name.lower() == str(name).lower()), None)
        if v is None:
            out.append({'variant': str(name), 'error': 'no such variant'})
            continue
        q = p.snapshot()
        q.active_uid = v.uid
        tc.status(f'Analysing variant {v.name}…')
        res, issues = _evaluate(tc, q)
        if res is None:
            out.append({'variant': v.name, 'error': 'cannot be analysed'})
            continue
        m = res.metrics
        out.append({'variant': v.name, 'active': v.uid == p.active_uid,
                    'cameras': [c.tag for c in v.cameras if c.visible],
                    'blind_m2': _r(m['blind_m2'], 3), 'covered_pct': _r(m['covered_pct'], 2),
                    'seen_by_2_or_more_pct': _r(m['multi_pct'], 1), 'min_density_px_cm': _r(m['min_density'], 2),
                    'above_min_density_pct': _r(m['min_ok_pct'], 1),
                    'issues': {s: sum(1 for i in issues if i.severity == s) for s in ('critical', 'serious', 'warning')}})
    return {'variants': out}


STUDIES = ('blind_volume', 'stitched_objects', 'load_cases', 'mounting_tolerance', 'perception_range', 'sensor_coverage')


def run_study(tc, study='', height=None, object_height=None, tolerance_deg=None, tolerance_mm=None, **_):
    p, eng = tc.project, tc.engine
    if study not in STUDIES:
        return {'error': f'study must be one of {", ".join(STUDIES)}'}
    res = _base(tc)
    if res is None:
        return {'error': 'The analysis cannot run (no vehicle or ROI).'}
    tags = _tags(p)
    try:
        if study == 'blind_volume':
            from ..core import volume as V
            h = min(max(float(height or 2.0), 0.3), 4.0)
            tc.status(f'Blind volume up to {h:.1f} m…')
            vr = V.analyse_volume(p, eng, res, height=h, dz=0.1, cell=max(res.cell, 0.1),
                                  cancel=tc.cancelled)                                     # a callable
            if vr is None:
                return {'error': 'cancelled'}
            m = vr.metrics
            return {'study': study, 'grid': '10 cm columns, 10 cm levels', 'height_m': _r(h, 2),
                    'blind_m3': _r(m['blind_m3'], 2), 'blind_pct_of_space': _r(m['blind_pct'], 1),
                    'ground_blind_m2': _r(m['ground_blind_m2'], 3), 'covered_up_to_m': _r(m['covered_to_m'], 2),
                    'worst_level': {'z_m': _r(m['worst_level_z'], 2), 'blind_m2': _r(m['worst_level_m2'], 2)},
                    'never_seen_at_any_height_m2': _r(m['never_seen_m2'], 3),
                    'objects': {n: {'height_m': _r(o['height'], 2), 'undetected_m2': _r(o['undetected_m2'], 3),
                                    'hides_up_to_m_from_body': _r(o['reach_m'], 2)} for n, o in m['objects'].items()}}
        if study == 'stitched_objects':
            from ..core import stitched_objects as so
            st = so.Settings.from_dict(p.analysis.stitched)
            h = min(max(float(object_height or st.height), 0.1), 2.5)
            tc.status(f'Objects {h:.1f} m tall in the stitched view…')
            sres = so.analyse(p, eng, res, height=h, cell=max(res.cell, 0.1), cancel=tc.cancel)   # an Event
            if sres is None:
                return {'error': 'cancelled'}
            tc.status('Searching seam angles…')
            adv = so.suggest_seams(sres, cancel=tc.cancel)
            m = sres.metrics
            out = {'study': study, 'object_height_m': _r(h, 2), 'grid': '10 cm cells',
                   'vanishes_at_seams_m2': _r(m['vanish_m2'], 3), 'cut_off_m2': _r(m['cut_m2'], 3),
                   'not_seen_m2': _r(m['blind_m2'], 3), 'mean_shown_fraction': _r(m['shown_mean'], 3),
                   'per_seam': [{'seam': list(x['tags']), 'angle_deg': _r(x['angle'], 1), 'vanish_m2': _r(x['vanish_m2'], 3)}
                                for x in sres.seams],
                   'hotspots': [{'centre': [_r(hh.x, 2), _r(hh.y, 2)], 'area_m2': _r(hh.area_m2, 3),
                                 'where': _where(tc, hh.x, hh.y)} for hh in sres.hotspots[:6]]}
            if adv is not None and adv.changes:
                out['seam_advice'] = {'vanish_now_m2': _r(adv.vanish_now, 3), 'vanish_after_m2': _r(adv.vanish_after, 3),
                                      'moves': [{'between': [tags.get(x.a), tags.get(x.b)], 'from_deg': _r(x.angle, 1),
                                                 'to_deg': _r(x.best, 1)} for x in adv.seams if x.moved],
                                      'note': 'propose these with propose_changes(seams=[...])'}
            return out
        if study == 'perception_range':
            from ..core import perception as PR
            st = PR.PerceptionSettings.from_dict(p.analysis.perception)
            tc.status('Perception range…')
            pr = PR.analyse(p, eng, res, st, cancel=tc.cancelled)
            if pr is None:
                return {'error': 'cancelled'}
            return {'study': study, **PR.summary(pr),
                    'objects': [{'name': c.name, 'height_m': c.height, 'width_m': c.width, 'min_px': c.min_px,
                                 'above_ground_m': c.base} for c in pr.classes],
                    'meaning': 'min_m is the shortest range, over the directions of the sector, out to which some '
                               'camera resolves the object (in the image, not hidden by the body, at least min_px tall '
                               'through the real lens); a sector passes when min_m reaches required_m. Stereo pairs: '
                               'cameras that both see a car 20 m out; depth_error_m is z^2*0.25px/(f*baseline).'}
        if study == 'sensor_coverage':
            from ..core import sensor_coverage as SC
            if not p.variant.sensors:
                return {'error': 'this variant has no lidar or radar (Vehicle ▸ Add lidar / Add radar)'}
            tc.status('Lidar and radar coverage…')
            cr = SC.run(p, eng, cancel=tc.cancelled)
            if cr is None:
                return {'error': 'cancelled'}
            return {'study': study, **SC.summary(cr),
                    'meaning': 'per object class and sector the shortest range out to which the cameras, the lidars '
                               '(points on the object at least the class minimum, summed over lidars, within range on '
                               'a 10 % target, not blocked by the vehicle) and the radars (object centre in a zone, '
                               'range scaled by the fourth root of its cross-section) detect it; verdict redundant = '
                               'two or more modalities reach the required range, single = one, gap = none. '
                               'blocked_pct: share of a lidar\'s points that land on the vehicle or the rig. '
                               'ground_blind_from_body: how far from the body the nearest lidar ground point lies.'}
        if study == 'load_cases':
            from ..core import loadcases as LC
            tc.status('Load cases…')
            rows = LC.evaluate(eng, p, cell=max(res.cell, 0.1))
            return {'study': study, 'note': 'the axle drops are the project\'s load cases (presets are examples, not '
                                            'this vehicle\'s suspension data)',
                    'cases': [{'name': r.name, 'front_drop_mm': _r(r.front_mm, 0), 'rear_drop_mm': _r(r.rear_mm, 0),
                               'pitch_deg': _r(r.pitch_deg, 2), 'blind_m2': _r(r.blind_m2, 3),
                               'blind_change_m2': _r(r.delta_blind, 3), 'min_density_px_cm': _r(r.min_density, 2),
                               'ground_shift_cm': _r(r.ground_shift_cm, 1), 'seam_ghost_cm': _r(r.seam_ghost_cm, 1),
                               'status': r.status} for r in rows]}
        from ..core.tolerance import worst_single_camera
        deg = float(tolerance_deg if tolerance_deg is not None else p.analysis.tol_deg)
        mm = float(tolerance_mm if tolerance_mm is not None else p.analysis.tol_mm)
        tc.status(f'Worst single-camera error at {deg:g}° / {mm:g} mm…')
        eng.vehicle_geom(p)
        b, uid, pert = worst_single_camera(eng, p, deg, mm / 1000.0, cell=max(res.cell, 0.05))
        return {'study': study, 'tolerance': {'deg': deg, 'mm': mm}, 'worst_blind_m2': _r(b, 3),
                'camera': tags.get(uid, uid) or None, 'perturbation': str(pert) if pert is not None else None,
                'baseline_blind_m2': _r(res.metrics['blind_m2'], 3)}
    except (TypeError, ValueError) as e:
        return {'error': f'bad option: {e}'}


VARY = {'pitch': (4.0, 0.25, 15.0), 'yaw': (6.0, 0.5, 25.0), 'roll': (4.0, 0.5, 10.0),   # first step, last step,
        'z': (0.04, 0.005, 0.15), 'x': (0.06, 0.005, 0.20), 'y': (0.04, 0.005, 0.10)}     # default range (±)


def optimise_camera(tc, camera='', vary=None, ranges=None, objective='blind', min_density=None, mirror_partner=True,
                    budget_s=20, **_):
    """Pattern search over one camera's pose (relative to where it is) at the project's ground resolution (5 cm at
    least, so small blind patches count). Nothing changes in the design."""
    p = tc.project
    c = _by_tag(p, camera)
    if c is None:
        return {'error': f'no camera {camera!r}'}
    if c.locked:
        return {'error': f'{c.tag} is locked'}
    keys = [k for k in (vary or ['pitch', 'yaw']) if k in VARY]
    if not keys:
        return {'error': f'vary must name some of {", ".join(VARY)}'}
    if objective not in ('blind', 'density', 'redundancy'):
        return {'error': "objective must be 'blind', 'density' or 'redundancy'"}
    try:
        rng = {k: abs(float((ranges or {}).get(k, VARY[k][2]))) for k in keys}
        need = float(min_density) if min_density is not None else None
        budget = min(max(float(budget_s or 20), 3.0), 60.0)
    except (TypeError, ValueError, AttributeError):
        return {'error': 'ranges, min_density and budget_s must be numbers'}
    cell = max(p.analysis.cell_m, 0.05)
    t0 = time.perf_counter()
    blind0 = [None]

    def score(off):
        trial = p.snapshot()
        ch = {'camera': c.tag, 'mirror_partner': bool(mirror_partner), **{f'd{k}': v for k, v in off.items() if v}}
        _, targets, _, errs = apply_changes(trial, [ch]) if len(ch) > 2 else ([], {'-': 1}, {}, [])
        if not targets:
            return None
        res = tc.engine.analyse(trial, cell=cell)
        if res is None:
            return None
        m = res.metrics
        b, d, r = m['blind_m2'], m['min_density'], m['multi_pct']
        if blind0[0] is None:
            blind0[0] = b
        worse = max(0.0, b - blind0[0] - 0.005) * 1000
        key = {'blind': (round(b, 4), -d), 'density': (worse, -round(d, 3), b), 'redundancy': (worse, -round(r, 2), b)}
        s = key[objective]
        if need is not None:
            s = (round(max(0.0, need - d), 3),) + s
        return s, m

    cur = {k: 0.0 for k in keys}
    first = score(cur)
    if first is None:
        return {'error': 'The analysis cannot run (no vehicle or ROI).'}
    best_s, best_m = first
    start_m = best_m
    steps = {k: VARY[k][0] for k in keys}
    n = 1
    while time.perf_counter() - t0 < budget and n < 160 and not tc.cancelled():
        moved = False
        for k in keys:
            for sgn in (1, -1):
                cand = dict(cur)
                cand[k] = max(-rng[k], min(rng[k], cur[k] + sgn * steps[k]))
                if abs(cand[k] - cur[k]) < 1e-9:
                    continue
                r = score(cand)
                n += 1
                if r is not None and r[0] < best_s:
                    cur, (best_s, best_m), moved = cand, r, True
                    break
            if moved:
                break
        if not moved:
            steps = {k: v / 2 for k, v in steps.items()}
            if all(steps[k] < VARY[k][1] for k in keys):
                break
        tc.status(f'Optimising {c.tag}: {n} trials · blind {best_m["blind_m2"]:.3f} m² · '
                  f'min density {best_m["min_density"]:.2f} px/cm')
    change = {'camera': c.tag, **{f'd{k}': round(v, 1 if k in ('pitch', 'yaw', 'roll') else 3)
                                  for k, v in cur.items() if abs(v) > 1e-9}}
    out = {'camera': c.tag, 'objective': objective, 'varied': {k: f'±{rng[k]:g} {"°" if k in VARY and k not in "xyz" else "m"}'
                                                                for k in keys},
           'search': {'trials': n, 'seconds': _r(time.perf_counter() - t0, 1), 'grid_cell_m': cell,
                      'stopped': 'cancelled' if tc.cancelled() else 'budget' if time.perf_counter() - t0 >= budget
                      else 'converged'},
           'on_search_grid': {'blind_m2': [_r(start_m['blind_m2'], 3), _r(best_m['blind_m2'], 3)],
                              'min_density_px_cm': [_r(start_m['min_density'], 2), _r(best_m['min_density'], 2)],
                              'seen_by_2_or_more_pct': [_r(start_m['multi_pct'], 1), _r(best_m['multi_pct'], 1)]}}
    if len(change) == 1:
        out['result'] = 'no better pose within the ranges; the camera is already at a local optimum for this objective'
        return out
    change['mirror_partner'] = bool(mirror_partner)
    _, check = _simulate(tc, [change])
    eff = check.get('effect', {})
    out['best_change'] = change
    out['checked_at_project_resolution'] = eff or check
    g = lambda k: (eff.get(k, {}).get('after') or 0) - (eff.get(k, {}).get('before') or 0)
    worth = g('blind_m2') <= -0.002 or g('min_density') >= 0.05 or g('multi_pct') >= 0.5
    out['worth_proposing'] = bool(worth)
    out['next'] = ('offer it with propose_changes(changes=[best_change])' if worth else
                   'the gain is negligible (under 0.002 m² blind, 0.05 px/cm, 0.5 % redundancy): do not propose it')
    return out


def update_plan(tc, steps=None, **_):
    good = []
    for st in steps if isinstance(steps, list) else []:
        if isinstance(st, dict) and str(st.get('step', '')).strip():
            status = st.get('status', 'pending')
            good.append({'step': str(st['step']).strip()[:140],
                         'status': status if status in ('pending', 'in_progress', 'done', 'skipped') else 'pending'})
    if not good:
        return {'error': 'steps must be a list of {step, status}'}
    if tc.on_plan is not None:
        tc.on_plan(good[:12])
    return {'plan': 'shown to the user', 'steps': len(good)}


UI_ACTIONS = ('select_cameras', 'frame_point', 'frame_all', 'heatmap', 'overlay', 'open_page')


def show_in_app(tc, action='', cameras=None, x=None, y=None, heatmap=None, overlay=None, on=True, page=None, **_):
    if action not in UI_ACTIONS:
        return {'error': f'action must be one of {", ".join(UI_ACTIONS)}'}
    req = {'action': action}
    if action == 'select_cameras':
        found = [c.tag for c in (_by_tag(tc.project, t) for t in (cameras or [])) if c is not None]
        if not found:
            return {'error': 'name the cameras by tag'}
        req['cameras'] = found
    elif action == 'frame_point':
        try:
            req['x'], req['y'] = float(x), float(y)
        except (TypeError, ValueError):
            return {'error': 'x and y (metres) are needed'}
    elif action == 'heatmap':
        if heatmap not in ('none', 'coverage', 'density', 'redundancy', 'stitched'):
            return {'error': "heatmap must be none, coverage, density, redundancy or stitched"}
        req['heatmap'] = heatmap
    elif action == 'overlay':
        req['overlay'], req['on'] = str(overlay or ''), bool(on)
    elif action == 'open_page':
        req['page'] = str(page or '')
    if tc.on_ui is not None:
        tc.on_ui(req)
    return {'done': action, **{k: v for k, v in req.items() if k != 'action'}}


# =========================================================================== registry
_CHANGE_ITEM = {
    'type': 'object',
    'properties': {
        'camera': {'type': 'string', 'description': 'camera tag, e.g. "ML"'},
        **{k: {'type': 'number', 'description': f'new absolute {k} ({"m" if k in "xyz" else "deg"})'} for k in POSE},
        **{dk: {'type': 'number', 'description': f'change of {k} ({"m" if k in "xyz" else "deg"})'}
           for dk, k in DELTAS.items()},
        'lens': {'type': 'string', 'description': 'library lens name or id to fit instead'},
        'visible': {'type': 'boolean', 'description': 'false switches the camera off in the analysis'},
        'mirror_partner': {'type': 'boolean',
                           'description': 'also mirror the change onto the linked left/right partner (default true)'},
    },
    'required': ['camera'],
}
_SEAM_ITEM = {'type': 'object', 'properties': {
    'between': {'type': 'array', 'items': {'type': 'string'}, 'description': 'the two camera tags, e.g. ["QR", "R"]'},
    'angle_deg': {'type': 'number', 'description': 'azimuth about the body centre, counter-clockwise from +x'}},
    'required': ['between', 'angle_deg']}
_SETTINGS = {'type': 'object', 'properties': {
    'min_density': {'type': 'number', 'description': 'px/cm'}, 'target_density': {'type': 'number', 'description': 'px/cm'},
    'roi_distance': {'type': 'number', 'description': 'm around the body'}}}
_CHANGE_PARAMS = {'changes': {'type': 'array', 'items': _CHANGE_ITEM, 'description': 'camera changes'},
                  'seams': {'type': 'array', 'items': _SEAM_ITEM, 'description': 'stitch seam angles'},
                  'settings': _SETTINGS}


@dataclass
class Tool:
    name: str
    label: str              # what the UI shows while it runs: "Reading coverage"
    description: str
    params: dict
    fn: object
    kind: str = 'read'      # read | compute | simulate | propose | plan | ui
    required: tuple = ()


TOOLS = [
    Tool('get_overview', 'Read the design overview',
         'The project, vehicle size, every camera (tag, pose, lens, module, field of view, locked, mirrored partner), '
         'the lidars and radars (tag, pose, model, channels and field, range, rate), the rig on the vehicle (roof '
         'racks, masts, enclosures: centre, size, whether they block cameras or carry them), the ROI and the analysis '
         'thresholds. Start here.', {}, get_overview),
    Tool('get_coverage', 'Read coverage figures',
         'Ground coverage figures: blind area, coverage, redundancy, minimum pixel density, and per camera the area it '
         'sees, sees alone, and what would go blind without it; the worst single-camera mounting error.', {},
         get_coverage),
    Tool('get_issues', 'Read the issues list', 'The Issues list (severity, message, cameras involved).', {}, get_issues),
    Tool('get_camera', 'Read camera details',
         'One camera in detail: pose, lens model, sensor, field of view, mount clearance, coverage, seams, issues.',
         {'camera': {'type': 'string', 'description': 'camera tag'}}, get_camera, required=('camera',)),
    Tool('explain_point', 'Check what sees a point',
         'For one point (metres, vehicle frame), which cameras see it and why not: outside the lens field of view, '
         'outside the image frame, hidden by the body; with pixel density and range.',
         {'x': {'type': 'number'}, 'y': {'type': 'number'},
          'z': {'type': 'number', 'description': 'height, default 0 (ground)'}}, explain_point, required=('x', 'y')),
    Tool('find_blind_regions', 'Find blind regions',
         'Connected blind regions and regions below the minimum density at the analysis height, largest first, each '
         'with where it is and why each camera misses its centre.',
         {'min_area_m2': {'type': 'number', 'description': 'ignore smaller regions (default 0.002)'}},
         find_blind_regions),
    Tool('get_seams', 'Read the stitch seams',
         'The bird\'s-eye stitch seams: which cameras, angle, density at the seam, where standing objects vanish, '
         'advised angles.', {}, get_seams),
    Tool('get_studies', 'Read the studies',
         'Results of studies already run in the app: objects in the stitched view, the 3D blind volume, parking '
         'guidelines and the worst single-camera error. Use run_study to compute one that has not been run.', {},
         get_studies),
    Tool('get_rig_health', 'Read the live rig health',
         'While the Live rig page streams the cameras: per camera the measured frame rate against the module\'s, '
         'jitter, dropped frames, resolution, exposure, sharpness, frozen images and a drift indicator, with an '
         'ok / warning / fault verdict and reasons; for the rig the cameras up, the worst frame rate and the sync '
         'skew (software timestamps). Otherwise says it is not streaming and which cameras have a stream set.', {},
         get_rig_health),
    Tool('get_sync_plan', 'Read the sync and timing plan',
         'Camera triggers against the spinning lidar\'s sweep: per camera the crossing time, trigger time, '
         'misalignment of its trigger line and the travel at speed, suggested trigger lines and the worst '
         'misalignment they give, phases for other lidars, and motion blur and rolling-shutter skew at speed with '
         'the longest exposure for the blur limit. Cheap.', {}, get_sync_plan),
    Tool('get_recording_budget', 'Read the recording budget',
         'How much data the rig records: per camera, lidar and radar the data rate and its link load; the totals '
         '(Mbit/s, GB per hour, TB per shift), the write speed the recorder needs against what it has, the network '
         'uplink, hours of driving per disk set, and findings with severities. Cheap.', {}, get_recording_budget),
    Tool('get_power_budget', 'Read the power budget',
         'What the rig draws: per device the steady and peak power, circuit, current and power-cable drop; the totals '
         '(steady and peak W, DC-DC output and input), the alternator margin, the engine-off runtime on the auxiliary '
         'battery, the worst circuit against its fuse, the enclosure heat, and findings with severities. Cheap.',
         {}, get_power_budget),
    Tool('get_gnss_imu', 'Read the GNSS and IMU study',
         'Per GNSS antenna the open sky above its mask, expected satellites, blockers and multipath; the dual-antenna '
         'heading baseline (length, direction, heading error); the IMU lever arms to every antenna and sensor; each '
         'sensor\'s pose error (mm, 1 sigma). Give an antenna tag for its blocked sectors and horizon profile. Cheap.',
         {'antenna': {'type': 'string', 'description': 'optional GNSS antenna tag for the full sky detail'}},
         get_gnss_imu),
    Tool('get_sun_and_dirt', 'Sun glare and dirt',
         'Per camera how often over the campaign the sun is in its image, near it (flare) and low ahead, with the '
         'worst months, hours and headings; per lidar the share with the sun in its field; per camera, lidar window '
         'and radar fascia a dirt and spray score with cleaning advice (a heuristic). The campaign is the project\'s '
         '(location, dates, driving hours, headings); lat, lon, start, end (YYYY-MM-DD), hour_from, hour_to override '
         'it for this answer only. Under a second.',
         {'lat': {'type': 'number'}, 'lon': {'type': 'number'}, 'start': {'type': 'string'},
          'end': {'type': 'string'}, 'hour_from': {'type': 'number'}, 'hour_to': {'type': 'number'}},
         get_sun_and_dirt, 'compute'),
    Tool('get_fleet', 'Read the fleet',
         'The vehicles built to this design and their calibrations: per vehicle and sensor the latest offset from the '
         'plan and its verdict, drifting sensors (trend per 30 days, days until out of tolerance), jumps, outlier '
         'vehicles against the fleet median, systematic offsets, unknown tags, and findings. Cheap.', {}, get_fleet),
    Tool('get_brackets', 'Read the mount brackets',
         'Per camera whether a 3D-printable bracket can hold it at its planned position and angles: mount surface, '
         'flat base or tube clamp, face, standoff and the standoff needed, base size, height and plate angle. About a '
         'second.', {}, get_brackets),
    Tool('get_sample_dataset', 'Read the sample dataset plan',
         'The synthetic drive the rig would record for pipeline tests: settings, frames per channel, keyframes, '
         'approximate size and generation time. Cheap; writes nothing.', {}, get_sample_dataset),
    Tool('get_drive_log_check', 'Read the drive-log check',
         'The last recorded drive checked against the plan (Calibrate ▸ Drive-log check): streams OK/missing, dropped '
         'frames, data rate against the budget, lidar phase lock, worst camera trigger offset, camera rotations with '
         'their confidence, and the findings. Reads the page\'s result; does not read the log. Cheap.', {},
         get_drive_log_check),
    Tool('plan_lidar_calibration', 'Plan a lidar–camera calibration',
         'The camera–lidar pairs ranked by shared field; for one pair (camera, lidar tags; default the best) the '
         'checkerboard poses to capture and the uncertainty they would give. About half a second.',
         {'camera': {'type': 'string'}, 'lidar': {'type': 'string'}}, plan_lidar_calibration),
    Tool('calibrate_lidar_camera', 'Run a lidar–camera calibration',
         'Calibrates the lidar to the camera from simulated captures of the planned board poses: verdict, offset from '
         'the plan, 1-sigma uncertainty, undetermined directions. About 8 seconds.',
         {'camera': {'type': 'string'}, 'lidar': {'type': 'string'}}, calibrate_lidar_camera, 'compute'),
    Tool('get_requirements', 'Read the requirements scorecard',
         'The project\'s requirements with value, target, margin and pass/fail, and the verdict. Cheap metrics at once; '
         'run=true also runs the studies the others need (seconds).',
         {'run': {'type': 'boolean'}}, get_requirements, 'compute'),
    Tool('get_requirement_metrics', 'List the requirement metrics',
         'The metrics a requirement can use: id, label, unit, study, parameters with the project\'s choices.',
         {}, get_requirement_metrics),
    Tool('propose_requirements', 'Propose requirements',
         'Propose requirements for the user to review on the scorecard page: items [{metric, params, op, target, '
         'priority, text}] (see get_requirement_metrics), or none to read them from the project brief (or text).',
         {'items': {'type': 'array', 'items': {'type': 'object'}}, 'text': {'type': 'string'}},
         propose_requirements, 'propose'),
    Tool('compare_variants', 'Compare variants',
         'Analyse layout variants of this project side by side (blind area, coverage, redundancy, density, issues). '
         'All variants unless named.', {'variants': {'type': 'array', 'items': {'type': 'string'}}},
         compare_variants, 'compute'),
    Tool('run_study', 'Run a study',
         'Compute a study now (10 cm grids, a few seconds): blind_volume (where a pole, child or adult goes unseen, '
         'up to `height`), stitched_objects (where an object of `object_height` vanishes at the seams, with seam '
         'advice), load_cases (the project\'s axle drops), mounting_tolerance (worst blind area when one camera is off '
         'by tolerance_deg / tolerance_mm), perception_range (how far out cars, pedestrians, cyclists and signs are '
         'resolved per sector against the required ranges, and the stereo camera pairs), sensor_coverage (with '
         'lidars or radars: per object and sector the camera, lidar and radar ranges and whether two, one or no '
         'modality reaches the required range; the share of each lidar\'s points the vehicle blocks, its ground blind '
         'zone, and how much of each camera image a lidar covers).',
         {'study': {'type': 'string', 'enum': list(STUDIES)}, 'height': {'type': 'number'},
          'object_height': {'type': 'number'}, 'tolerance_deg': {'type': 'number'}, 'tolerance_mm': {'type': 'number'}},
         run_study, 'compute', ('study',)),
    Tool('optimise_camera', 'Optimise a camera',
         'Search one camera\'s pose (relative to where it is now, within ranges) for the best blind area, minimum '
         'density or redundancy, optionally keeping a minimum density; the mirrored partner follows. Returns the best '
         'change, checked at the project\'s resolution, ready for propose_changes. Takes up to budget_s seconds.',
         {'camera': {'type': 'string'},
          'vary': {'type': 'array', 'items': {'type': 'string', 'enum': list(VARY)},
                   'description': 'default ["pitch", "yaw"]'},
          'ranges': {'type': 'object', 'description': 'max change per variable, e.g. {"pitch": 10, "z": 0.05} (deg, m)'},
          'objective': {'type': 'string', 'enum': ['blind', 'density', 'redundancy']},
          'min_density': {'type': 'number', 'description': 'px/cm the result must keep everywhere'},
          'mirror_partner': {'type': 'boolean'}, 'budget_s': {'type': 'number', 'description': '3–60, default 20'}},
         optimise_camera, 'compute', ('camera',)),
    Tool('simulate_changes', 'Simulate a change',
         'Run the analysis with camera, seam or threshold changes applied to a copy of the design and report the '
         'effect (blind area, coverage, density, issues). Nothing changes in the user\'s design.',
         dict(_CHANGE_PARAMS), simulate_changes, 'simulate'),
    Tool('propose_changes', 'Propose a change',
         'Offer changes to the user: they see them with the predicted effect and apply them as one undo step (or as '
         'a new variant), or discard them. Simulate first. Give a short title and a one-sentence rationale.',
         {**_CHANGE_PARAMS, 'title': {'type': 'string', 'description': 'e.g. "Lower the mirror cameras 20 mm"'},
          'rationale': {'type': 'string'}}, propose_changes, 'propose'),
    Tool('update_plan', 'Update the plan',
         'Show your plan to the user as a checklist and keep it current: call it first with the steps, then again '
         'whenever a step starts (in_progress) or ends (done / skipped).',
         {'steps': {'type': 'array', 'items': {'type': 'object', 'properties': {
             'step': {'type': 'string'}, 'status': {'type': 'string', 'enum': ['pending', 'in_progress', 'done', 'skipped']}},
             'required': ['step', 'status']}}}, update_plan, 'plan', ('steps',)),
    Tool('show_in_app', 'Show in Vantage',
         'Show something to the user in the app (does not change the design): select_cameras (cameras), '
         'frame_point (x, y: frames the views on a ground point and marks it), frame_all, heatmap '
         '(none | coverage | density | redundancy | stitched), overlay (overlay key such as camlight, volume, '
         'guidelines, frustums, sensors; on), open_page (viewports | camviews | volume | perception | tolerance | '
         'compare | rig | coverage | timing | recording | power | gnss | exposure | fleet | brackets | dataset | drivelog | requirements | lidar_calib).',
         {'action': {'type': 'string', 'enum': list(UI_ACTIONS)}, 'cameras': {'type': 'array', 'items': {'type': 'string'}},
          'x': {'type': 'number'}, 'y': {'type': 'number'}, 'heatmap': {'type': 'string'}, 'overlay': {'type': 'string'},
          'on': {'type': 'boolean'}, 'page': {'type': 'string'}}, show_in_app, 'ui', ('action',)),
]
BY_NAME = {t.name: t for t in TOOLS}


def schemas() -> list:
    return [{'type': 'function', 'function': {
        'name': t.name, 'description': t.description,
        'parameters': {'type': 'object', 'properties': t.params, **({'required': list(t.required)} if t.required else {})}}}
        for t in TOOLS]


def run(tc, name: str, args: dict) -> tuple[str, bool]:
    """(JSON text for the model, ok)."""
    t = BY_NAME.get(name)
    if t is None:
        return dump({'error': f'unknown tool {name!r}'}), False
    try:
        out = t.fn(tc, **(args if isinstance(args, dict) else {}))
    except Exception as e:                          # a tool must never take the conversation down
        return dump({'error': f'{type(e).__name__}: {e}'}), False
    return dump(out), not (isinstance(out, dict) and 'error' in out and len(out) <= 2)
