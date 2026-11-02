"""Changes to a project as a list of operations: what Claude's edit_project tool applies, in the app (one undo step,
vantage.ui.mcp_bridge) or on a project file (vantage.mcp.session).

An operation names a target and what to do with it:

    {"op": "set", "target": "camera", "id": "F", "props": {"pitch": 35}}
    {"op": "add", "target": "sensor", "props": {"kind": "radar"}}
    {"op": "add", "target": "rig_part", "preset": "sensor_mast"}
    {"op": "remove", "target": "obstacle", "id": "Pole 1.0 m"}
    {"op": "activate", "target": "variant", "id": "Data rig · 6-cam + lidar + radars"}
    {"op": "link", "target": "camera", "id": "ML", "partner": "MR"}
    {"op": "set", "target": "seam", "id": "QR|MR", "props": {"angle_deg": -107}}
    {"op": "set", "target": "seam", "id": "QR|MR", "props": {"auto": true}}

apply() runs them in order on a project and returns the project sections they touched (the undo keys) and a line per
operation. Every value is checked against the field it replaces (its type, its range, the library for a lens or a
model), so a wrong operation raises EditError and changes nothing: the caller runs the list on a snapshot first.

A seam is named by its two cameras' tags as get_seams lists them. It is set the way the Camera views ▸ Stitch seams
page sets it (variant.seams, the angle by hand to 0.1°) or handed back to the automatic placement, so apply() takes
the analysis' seams to find it.

What Claude cannot change here: a locked camera or sensor (the user locked it; Claude may lock, never unlock), a unit
calibration (measured data), stream sources (they may hold passwords), the fleet, photos, the vehicle model, the
Copilot's approval records and the automation settings (a command never comes from the model)."""
from __future__ import annotations

import math

from ..core import project as P
from ..core import rig as R
from ..core import sensors as S

OPS = ('set', 'add', 'remove', 'activate', 'link', 'unlink')
TARGETS = ('camera', 'sensor', 'rig_part', 'obstacle', 'target', 'variant', 'roi', 'ecu', 'analysis',
           'target_settings', 'project', 'log', 'seam')
LISTED = ('camera', 'sensor', 'rig_part', 'obstacle', 'target', 'variant')      # targets named by an id

CAMERA_FIELDS = ('tag', 'name', 'location', 'x', 'y', 'z', 'yaw', 'pitch', 'roll', 'module_id', 'lens_id', 'port',
                 'sync', 'cable_m', 'standoff_mm', 'bracket', 'snap_normal', 'locked', 'visible', 'route_extra_m',
                 'route_note')
SENSOR_FIELDS = ('tag', 'name', 'model_id', 'x', 'y', 'z', 'yaw', 'pitch', 'roll', 'phase_deg', 'port', 'visible',
                 'locked')
RIG_FIELDS = ('name', 'kind', 'x', 'y', 'z', 'length', 'width', 'height', 'radius', 'yaw', 'occludes', 'mountable',
              'visible')
OBSTACLE_FIELDS = ('name', 'x', 'y', 'height', 'radius')
TARGET_FIELDS = ('name', 'x', 'y', 'yaw', 'pattern', 'rows', 'cols', 'square_mm')
VARIANT_FIELDS = ('name', 'note', 'blend_m')
ANALYSIS_LOCKED = ('copilot', 'assistant_log', 'assistant', 'automation')       # audit trail, log, brief, commands
CHOICES = {'sync': tuple('ABCDEFGH'), 'kind@rig_part': ('box', 'cylinder', 'bar'),
           'pattern': ('checkerboard', 'charuco', 'aprilgrid'), 'mode@roi': ('offset', 'polygon'),
           'measured_from': ('outline', 'bbox'), 'corners': ('round', 'square'),
           'heatmap': ('none', 'coverage', 'density', 'redundancy', 'stitched'),
           'obstacle_mode': ('ground', 'pole05', 'pole10', 'child'), 'frame': tuple(P.FRAME_PRESETS),
           'units': ('m', 'mm')}
RANGES = {'x': (-50.0, 50.0), 'y': (-50.0, 50.0), 'z': (-5.0, 20.0), 'pitch': (-90.0, 90.0), 'roll': (-180.0, 180.0), 'cable_m': (0.0, 100.0), 'standoff_mm': (0.0, 1000.0),
          'route_extra_m': (0.0, 100.0), 'length': (0.001, 20.0), 'width': (0.001, 20.0), 'height': (0.001, 20.0),
          'radius': (0.001, 5.0), 'blend_m': (0.0, 2.0), 'rows': (2, 40), 'cols': (2, 40), 'square_mm': (5.0, 1000.0),
          'cell_m': (0.01, 0.5), 'min_density': (0.1, 100.0), 'target_density': (0.1, 100.0), 'tol_deg': (0.0, 45.0),
          'tol_mm': (0.0, 1000.0), 'distance@roi': (0.1, 50.0), 'ports': (1, 16)}
TAG_MAX = 12
SEAM_FIELDS = ('angle_deg', 'auto')
SEAM_GAP_DEG = 1.0          # neighbouring seams stay at least this far apart, in their order round the vehicle


class EditError(ValueError):
    pass


# =========================================================================== values
def _range(key, target):
    return RANGES.get(f'{key}@{target}') or RANGES.get(key)


def _value(target, key, new, cur):
    """new checked and coerced against the field's current value cur (its type) and the field's rules."""
    where = f'{target}.{key}'
    choices = CHOICES.get(f'{key}@{target}') or CHOICES.get(key)
    if isinstance(cur, bool):
        if not isinstance(new, bool):
            raise EditError(f'{where}: true or false, not {new!r}')
        return new
    if isinstance(cur, int) and not isinstance(cur, bool):
        if isinstance(new, bool) or not isinstance(new, (int, float)) or float(new) != int(new):
            raise EditError(f'{where}: a whole number, not {new!r}')
        new = int(new)
    elif isinstance(cur, float):
        if isinstance(new, bool) or not isinstance(new, (int, float)) or not math.isfinite(float(new)):
            raise EditError(f'{where}: a number, not {new!r}')
        new = float(new)
    elif isinstance(cur, str):
        if not isinstance(new, str):
            raise EditError(f'{where}: text, not {new!r}')
        new = new.strip()
        if len(new) > 400:
            raise EditError(f'{where}: at most 400 characters')
    elif isinstance(cur, (list, dict)):
        if not isinstance(new, type(cur)):
            raise EditError(f'{where}: a{"n object" if isinstance(cur, dict) else " list"} is expected')
    if choices is not None and new not in choices:
        raise EditError(f'{where}: one of {", ".join(map(str, choices))}')
    lim = _range(key, target)
    if lim is not None and isinstance(new, (int, float)) and not lo_hi(new, lim):
        raise EditError(f'{where}: between {lim[0]:g} and {lim[1]:g}')
    return new


def lo_hi(v, lim):
    return lim[0] <= v <= lim[1]


def _wrap(a):
    return (a + 180.0) % 360.0 - 180.0


def _fmt(v):
    if isinstance(v, float):
        return f'{v:.4g}'
    if isinstance(v, str):
        return repr(v) if len(v) <= 30 else repr(v[:27] + '…')
    return str(v)


def _set_fields(obj, props, allowed, target, label):
    """Set props on a dataclass object after checking each one; a line saying what changed."""
    if not isinstance(props, dict) or not props:
        raise EditError(f'{label}: props with the values to set are needed')
    bad = [k for k in props if k not in allowed]
    if bad:
        raise EditError(f'{label}: cannot set {", ".join(bad)} (settable: {", ".join(allowed)})')
    done = []
    for k, v in props.items():
        cur = getattr(obj, k)
        v = _value(target, k, v, cur)
        if k in ('yaw',):
            v = _wrap(v)
        if v != cur:
            setattr(obj, k, v)
            done.append(f'{k} {_fmt(cur)} → {_fmt(v)}')
    return f'{label}: ' + (', '.join(done) if done else 'no change')


# =========================================================================== finding things
def _variant(p, ref):
    if ref in (None, '', 'active'):
        return p.variant
    for i, v in enumerate(p.variants):
        if ref in (v.name, v.uid, str(i + 1)):
            return v
    raise EditError(f'no variant {ref!r} (variants: {", ".join(v.name for v in p.variants)})')


def _named(items, ref, what, attr='name'):
    for it in items:
        if ref in (getattr(it, attr), it.uid):
            return it
    names = ', '.join(getattr(it, attr) for it in items) or 'none'
    raise EditError(f'no {what} {ref!r} ({what}s: {names})')


def _variant_key(p, v):
    return 'variants' if v is None else f'variant:{v.uid}'


def _check_tag(v, tag, own_uid):
    if not tag or len(tag) > TAG_MAX or any(ch.isspace() for ch in tag):
        raise EditError(f'tag {tag!r}: 1–{TAG_MAX} characters, no spaces')
    for o in list(v.cameras) + list(v.sensors):
        if o.tag == tag and o.uid != own_uid:
            raise EditError(f'tag {tag!r} is taken in {v.name!r}')


def _check_camera(p, c):
    if c.module_id not in p.modules:
        raise EditError(f'module {c.module_id!r} is not in the library ({", ".join(p.modules)})')
    lens = p.lenses.get(c.lens_id)
    if lens is None:
        raise EditError(f'lens {c.lens_id!r} is not in the library')
    if lens.module_id and lens.module_id != c.module_id:
        fits = [l.id for l in p.lenses.values() if l.module_id in ('', c.module_id)]
        raise EditError(f'lens {c.lens_id!r} is for {lens.module_id}, not {c.module_id} (lenses that fit: '
                        f'{", ".join(fits)})')
    if not 0 <= c.port <= p.ecu.ports:
        raise EditError(f'port {c.port}: 0 (unassigned) to {p.ecu.ports}')


def _check_sensor(p, s, kind):
    m = p.sensor_models.get(s.model_id)
    if m is None or m.kind != kind:
        ok = [k for k, mm in p.sensor_models.items() if mm.kind == kind]
        raise EditError(f'{s.model_id!r} is not a {kind} model in the library ({", ".join(ok)})')


# =========================================================================== operations
def _set(p, op, touched):
    t, ref, props = op['target'], op.get('id'), op.get('props')
    if t == 'camera':
        v = _variant(p, op.get('variant'))
        c = _named(v.cameras, ref, 'camera', 'tag')
        if c.locked:
            raise EditError(f'camera {c.tag} is locked: the user unlocks it in Vantage')
        if isinstance(props, dict) and props.get('locked') is False:
            raise EditError('only the user unlocks a camera')
        line = _set_fields(c, props, CAMERA_FIELDS, 'camera', f'camera {ref}')
        if 'tag' in props:
            _check_tag(v, c.tag, c.uid)
        _check_camera(p, c)
        if c.link_uid and op.get('mirror', True):
            partner = v.cam(c.link_uid)
            if partner is not None and not partner.locked:
                P.mirror_props(c, partner, props.keys())
                _check_camera(p, partner)
                line += f' (mirrored to {partner.tag})'
        touched.add(_variant_key(p, v))
        return line
    if t == 'sensor':
        v = _variant(p, op.get('variant'))
        s = _named(v.sensors, ref, 'sensor', 'tag')
        if s.locked:
            raise EditError(f'sensor {s.tag} is locked: the user unlocks it in Vantage')
        if isinstance(props, dict) and props.get('locked') is False:
            raise EditError('only the user unlocks a sensor')
        line = _set_fields(s, props, SENSOR_FIELDS, 'sensor', f'sensor {ref}')
        if 'tag' in props:
            _check_tag(v, s.tag, s.uid)
        _check_sensor(p, s, s.kind)
        s.phase_deg %= 360.0
        touched.add(_variant_key(p, v))
        return line
    if t == 'rig_part':
        touched.add('rig')
        return _set_fields(_named(p.rig, ref, 'rig part'), props, RIG_FIELDS, 'rig_part', f'rig part {ref}')
    if t == 'obstacle':
        touched.add('obstacles')
        return _set_fields(_named(p.obstacles, ref, 'obstacle'), props, OBSTACLE_FIELDS, 'obstacle', f'obstacle {ref}')
    if t == 'target':
        touched.add('targets')
        return _set_fields(_named(p.targets, ref, 'target'), props, TARGET_FIELDS, 'target', f'target {ref}')
    if t == 'variant':
        v = _variant(p, ref)
        if isinstance(props, dict) and 'seams' in props:
            raise EditError('seams are set one at a time: {"op": "set", "target": "seam", "id": "QR|MR", "props": '
                            '{"angle_deg": -107}} or {"auto": true} for the automatic placement (get_seams lists them)')
        touched.add('variants')
        line = _set_fields(v, props, VARIANT_FIELDS, 'variant', f'variant {v.name!r}')
        if not v.name:
            raise EditError('a variant needs a name')
        return line
    if t == 'roi':
        touched.add('roi')
        allowed = tuple(f for f in P.ROI.__dataclass_fields__)
        line = _set_fields(p.roi, props, allowed, 'roi', 'ROI')
        if p.roi.mode == 'polygon':
            poly = p.roi.polygon
            if len(poly) < 3 or not all(isinstance(q, (list, tuple)) and len(q) == 2 and
                                        all(isinstance(x, (int, float)) and math.isfinite(x) for x in q) for q in poly):
                raise EditError('roi.polygon: at least 3 [x, y] points (metres) for a polygon ROI')
            p.roi.polygon = [(float(x), float(y)) for x, y in poly]
        return line
    if t == 'ecu':
        touched.add('ecu')
        return _set_fields(p.ecu, props, tuple(P.ECU.__dataclass_fields__), 'ecu', 'ECU')
    if t == 'analysis':
        touched.add('analysis')
        allowed = tuple(f for f in P.AnalysisSettings.__dataclass_fields__ if f not in ANALYSIS_LOCKED)
        if isinstance(props, dict):
            props = {k: ({**getattr(p.analysis, k), **v} if isinstance(v, dict) and
                         isinstance(getattr(p.analysis, k, None), dict) else v) for k, v in props.items()}
        return _set_fields(p.analysis, props, allowed, 'analysis', 'analysis settings')
    if t == 'target_settings':
        touched.add('targets')
        return _set_fields(p.target_settings, props, tuple(P.TargetSettings.__dataclass_fields__), 'target_settings',
                           'target settings')
    if t == 'project':
        if not isinstance(props, dict) or not props:
            raise EditError('project: props are needed (name, frame, custom_origin, units, brief)')
        bad = [k for k in props if k not in ('name', 'frame', 'custom_origin', 'units', 'brief')]
        if bad:
            raise EditError(f'project: cannot set {", ".join(bad)} (settable: name, frame, custom_origin, units, brief)')
        done = []
        if 'name' in props:
            name = _value('project', 'name', props['name'], p.name)
            if not name:
                raise EditError('project.name: empty')
            p.name = name
            touched.add('name')
            done.append(f'name → {name!r}')
        for k in ('frame', 'units'):
            if k in props:
                setattr(p, k, _value('project', k, props[k], getattr(p, k)))
                touched.add('frame')
                done.append(f'{k} → {getattr(p, k)}')
        if 'custom_origin' in props:
            o = props['custom_origin']
            if not (isinstance(o, list) and len(o) == 3 and all(isinstance(x, (int, float)) and math.isfinite(x)
                                                                 for x in o)):
                raise EditError('project.custom_origin: [x, y, z] in metres')
            p.custom_origin = [float(x) for x in o]
            touched.add('frame')
            done.append(f'custom origin → {p.custom_origin}')
        if 'brief' in props:
            text = props['brief']
            if not isinstance(text, str) or len(text) > 8000:
                raise EditError('project.brief: text up to 8000 characters')
            p.analysis.assistant = {**p.analysis.assistant, 'brief': text.strip()}
            touched.add('analysis')
            done.append('brief updated')
        return 'project: ' + ', '.join(done)
    raise EditError(f'set: target {t!r} cannot be set')


def _seam(p, op, touched, seams, done):
    """Set one stitch seam of the active variant as the Stitch seams page does: an angle by hand (variant.seams,
    'uidA|uidB' -> degrees to 0.1), or back to the automatic placement (the key removed). done collects
    key -> angle (None: automatic) for the order check."""
    if op['op'] != 'set':
        raise EditError('a seam is set (props angle_deg, or auto: true for the automatic placement); it is not '
                        'added, removed or linked')
    if op.get('variant') not in (None, '', 'active') and _variant(p, op['variant']).uid != p.active_uid:
        raise EditError('seams are set on the active variant: activate the other one first')
    v = p.variant
    if not seams:
        raise EditError('this variant has no seams in the analysis (fewer than two cameras, or no vehicle or ROI)')
    tags = {c.uid: c.tag for c in v.cameras}
    names = ', '.join(f'{tags.get(s.a, "?")}|{tags.get(s.b, "?")}' for s in seams)
    ref = op.get('id')
    parts = [x.strip().lower() for x in (ref if isinstance(ref, list) else str(ref or '').replace(',', '|').split('|'))
             if str(x).strip()]
    seam = next((s for s in seams if len(parts) == 2 and
                 sorted([tags.get(s.a, '').lower(), tags.get(s.b, '').lower()]) == sorted(parts)), None)
    if seam is None:
        raise EditError(f'no seam {ref!r}: name it by its two cameras\' tags, e.g. "{names.split(", ")[0]}" '
                        f'(seams: {names})')
    props = op.get('props')
    if not isinstance(props, dict) or not props:
        raise EditError('props: angle_deg (degrees about the body centre, counter-clockwise from +x) or auto: true')
    bad = [k for k in props if k not in SEAM_FIELDS]
    if bad:
        raise EditError(f'cannot set {", ".join(bad)} (settable: {", ".join(SEAM_FIELDS)})')
    key = f'{seam.a}|{seam.b}'
    label = f'seam {tags[seam.a]} | {tags[seam.b]}'
    cur = v.seams.get(key)
    was = f'{cur:.1f}° by hand' if cur is not None else (f'{seam.angle:.1f}° automatic' if seam.auto else 'automatic')
    auto = props.get('auto')
    if 'auto' in props and not isinstance(auto, bool):
        raise EditError(f'{label}: auto is true or false, not {auto!r}')
    if 'angle_deg' in props:
        if auto:
            raise EditError(f'{label}: an angle or auto: true, not both')
        a = props['angle_deg']
        if isinstance(a, bool) or not isinstance(a, (int, float)) or not math.isfinite(float(a)):
            raise EditError(f'{label}: angle_deg is a number of degrees, not {a!r}')
        new = round(_wrap(float(a)), 1)
    elif auto:
        new = None
    else:                                       # auto: false keeps the seam where it is, as placed by hand
        new = cur if cur is not None else round(seam.angle, 1)
    done[key] = new
    touched.add(_variant_key(p, v))
    if new == cur:
        return f'{label}: no change'
    if new is None:
        v.seams.pop(key, None)
        return f'{label}: {was} → automatic'
    v.seams[key] = new
    return f'{label}: {was} → {new:.1f}° by hand'


def _check_seam_order(p, seams, done):
    """A seam set by hand stays between the seams either side of it (and SEAM_GAP_DEG from them), as the stitch
    gives each camera the sector between its two seams."""
    n = len(seams)
    if n < 2:
        return
    tags = {c.uid: c.tag for c in p.variant.cameras}
    keys = [f'{s.a}|{s.b}' for s in seams]
    ang = [done[k] if k in done else s.angle for k, s in zip(keys, seams)]
    for k in range(n):
        if keys[k] not in done or ang[k] is None:
            continue
        a, i, j = ang[k], (k - 1) % n, (k + 1) % n
        lo, hi = ang[i], ang[j]
        if lo is None or hi is None:
            continue                            # a neighbour going back to automatic is placed between the two
        span = (hi - lo) % 360.0 or 360.0
        if not SEAM_GAP_DEG <= (a - lo) % 360.0 <= span - SEAM_GAP_DEG:
            s, sp, sn = seams[k], seams[i], seams[j]
            raise EditError(f'seam {tags.get(s.a)} | {tags.get(s.b)} at {a:.1f}° is not between the seams either side '
                            f'of it ({tags.get(sp.a)} | {tags.get(sp.b)} at {lo:.1f}°, {tags.get(sn.a)} | '
                            f'{tags.get(sn.b)} at {hi:.1f}°): each camera keeps the sector between its two seams')


def _free_tag(v, base):
    taken = v.tags()
    if base not in taken:
        return base
    i = 2
    while f'{base}{i}' in taken:
        i += 1
    return f'{base}{i}'


def _add(p, op, touched):
    t, props = op['target'], op.get('props') or {}
    if not isinstance(props, dict):
        raise EditError('props must be an object')
    if t == 'camera':
        v = _variant(p, op.get('variant'))
        if len(v.cameras) >= P.MAX_CAMERAS:
            raise EditError(f'{v.name!r} already has {P.MAX_CAMERAS} cameras, the most a variant holds')
        c = P.Camera(tag=_free_tag(v, 'C'), slot=v.free_slot())
        props = {k: x for k, x in props.items() if k != 'locked' or x is True}
        _set_fields(c, props, CAMERA_FIELDS, 'camera', 'new camera')
        _check_tag(v, c.tag, c.uid)
        if c.port == 0:
            used = {o.port for o in v.cameras}
            c.port = next((i for i in range(1, p.ecu.ports + 1) if i not in used), 0)
        _check_camera(p, c)
        v.cameras.append(c)
        touched.add(_variant_key(p, v))
        return (f'added camera {c.tag} at ({c.x:.3f}, {c.y:.3f}, {c.z:.3f}), yaw {c.yaw:g}°, pitch {c.pitch:g}°, '
                f'{c.module_id}/{c.lens_id}, port {c.port}')
    if t == 'sensor':
        v = _variant(p, op.get('variant'))
        kind = props.get('kind', 'lidar')
        if kind not in S.KINDS:
            raise EditError(f'sensor kind: one of {", ".join(S.KINDS)}')
        mid = props.get('model_id') or S.default_model_id(kind)
        if mid not in p.sensor_models:
            mid = next((k for k, m in p.sensor_models.items() if m.kind == kind), mid)
        model = p.sensor_models.get(mid)
        if model is None or model.kind != kind:
            raise EditError(f'{mid!r} is not a {kind} model in the library')
        pose = S.default_pose(kind, model, p.vehicle.mesh, p.rig, v.sensors)
        s = P.Sensor(kind=kind, model_id=mid, tag=S.next_tag(kind, v.tags()), name=f'{model.vendor} {model.name}'.strip(),
                     **pose)
        rest = {k: x for k, x in props.items() if k not in ('kind', 'model_id') and (k != 'locked' or x is True)}
        if rest:
            _set_fields(s, rest, SENSOR_FIELDS, 'sensor', 'new sensor')
        _check_tag(v, s.tag, s.uid)
        v.sensors.append(s)
        touched.add(_variant_key(p, v))
        return f'added {S.KIND_NOUN.get(kind, kind)} {s.tag} ({model.name}) at ({s.x:.3f}, {s.y:.3f}, {s.z:.3f})'
    if t == 'rig_part':
        preset = op.get('preset')
        if preset:
            if preset not in dict(R.PRESETS):
                raise EditError(f'preset: one of {", ".join(dict(R.PRESETS))}')
            if p.vehicle.mesh is None:
                raise EditError('a preset is sized from the vehicle model, and this project has none')
            plan = R.fit_plan(p, preset)
            parts, dz, moved = plan
            R.apply_fit(p, plan)
            touched.add('rig')
            if moved:
                touched.add('variants')
            line = f'added {dict(R.PRESETS)[preset].lower()} ({", ".join(q.name for q in parts)})'
            if dz:
                line += f'; lifted {len(dz)} part(s) onto it by up to {max(dz.values()) * 1000:.0f} mm'
            return line
        part = P.RigPart()
        _set_fields(part, props, RIG_FIELDS, 'rig_part', 'new rig part')
        p.rig.append(part)
        touched.add('rig')
        return f'added rig part {part.name!r} ({part.kind}) at ({part.x:.3f}, {part.y:.3f}, {part.z:.3f})'
    if t == 'obstacle':
        o = P.Obstacle(name=f'Pole {len(p.obstacles) + 1}')
        _set_fields(o, props, OBSTACLE_FIELDS, 'obstacle', 'new obstacle')
        p.obstacles.append(o)
        touched.add('obstacles')
        return f'added obstacle {o.name!r} at ({o.x:.3f}, {o.y:.3f}), {o.height:g} m tall'
    if t == 'target':
        g = P.Target(name=f'T{len(p.targets) + 1}')
        _set_fields(g, props, TARGET_FIELDS, 'target', 'new target')
        p.targets.append(g)
        touched.add('targets')
        return f'added calibration target {g.name!r} at ({g.x:.3f}, {g.y:.3f})'
    if t == 'variant':
        name = props.get('name') or f'Variant {len(p.variants) + 1}'
        if not isinstance(name, str) or not name.strip():
            raise EditError('variant name: text')
        src = _variant(p, op.get('id')) if op.get('copy', True) else None
        v = _copy_variant(src) if src is not None else P.Variant()
        v.name = name.strip()[:64]
        if 'note' in props:
            v.note = _value('variant', 'note', props['note'], v.note)
        p.variants.append(v)
        if op.get('activate', True):
            p.active_uid = v.uid
        touched.add('variants')
        return f'added variant {v.name!r}' + (f' (a copy of {src.name!r})' if src is not None else ' (empty)') + \
            (', now active' if op.get('activate', True) else '')
    if t == 'log':
        from ..assistant import memory as M
        e = M.new_entry(props.get('kind', 'note'), props.get('text', ''), props.get('about') or (), source='assistant')
        if e is None:
            raise EditError(f'log: kind one of {", ".join(M.KINDS)} and a text are needed')
        if M.add(p, e) is None:
            raise EditError('log: the entry was not added (empty or a duplicate)')
        touched.add('analysis')
        return f'design log: {e["kind"]}: {e["text"][:80]}'
    raise EditError(f'add: target {t!r} cannot be added')


def _copy_variant(src):
    """A copy of a variant with its own uids (linked partners and seams follow)."""
    v = P.Variant.from_dict(src.to_dict())
    v.uid = P.new_uid()
    remap = {}
    for c in v.cameras:
        remap[c.uid] = P.new_uid()
        c.uid = remap[c.uid]
    for c in v.cameras:
        c.link_uid = remap.get(c.link_uid, '')
    for s in v.sensors:
        s.uid = P.new_uid()
    v.seams = {'|'.join(remap.get(u, u) for u in k.split('|')): a for k, a in v.seams.items()}
    return v


def _remove(p, op, touched):
    t, ref = op['target'], op.get('id')
    if t == 'camera':
        v = _variant(p, op.get('variant'))
        c = _named(v.cameras, ref, 'camera', 'tag')
        if c.locked:
            raise EditError(f'camera {c.tag} is locked: the user unlocks it in Vantage')
        v.cameras = [o for o in v.cameras if o.uid != c.uid]
        for o in v.cameras:
            if o.link_uid == c.uid:
                o.link_uid = ''
        v.seams = {k: a for k, a in v.seams.items() if c.uid not in k.split('|')}
        touched.add(_variant_key(p, v))
        return f'removed camera {c.tag} from {v.name!r}'
    if t == 'sensor':
        v = _variant(p, op.get('variant'))
        s = _named(v.sensors, ref, 'sensor', 'tag')
        if s.locked:
            raise EditError(f'sensor {s.tag} is locked: the user unlocks it in Vantage')
        v.sensors = [o for o in v.sensors if o.uid != s.uid]
        touched.add(_variant_key(p, v))
        return f'removed sensor {s.tag} from {v.name!r}'
    if t in ('rig_part', 'obstacle', 'target'):
        attr, key, what = {'rig_part': ('rig', 'rig', 'rig part'), 'obstacle': ('obstacles', 'obstacles', 'obstacle'),
                           'target': ('targets', 'targets', 'target')}[t]
        it = _named(getattr(p, attr), ref, what)
        setattr(p, attr, [o for o in getattr(p, attr) if o.uid != it.uid])
        touched.add(key)
        return f'removed {what} {it.name!r}'
    if t == 'variant':
        v = _variant(p, ref)
        if len(p.variants) == 1:
            raise EditError('the last variant cannot be removed')
        p.variants = [o for o in p.variants if o.uid != v.uid]
        if p.active_uid == v.uid:
            p.active_uid = p.variants[0].uid
        touched.add('variants')
        return f'removed variant {v.name!r}; active: {p.variant.name!r}'
    raise EditError(f'remove: target {t!r} cannot be removed')


def _link(p, op, touched, on):
    if op['target'] != 'camera':
        raise EditError('link and unlink are for cameras (mirrored partners)')
    v = _variant(p, op.get('variant'))
    a = _named(v.cameras, op.get('id'), 'camera', 'tag')
    if not on:
        b = v.cam(a.link_uid)
        if b is None:
            raise EditError(f'camera {a.tag} has no partner')
        a.link_uid = b.link_uid = ''
        touched.add(_variant_key(p, v))
        return f'unlinked {a.tag} and {b.tag}'
    b = _named(v.cameras, op.get('partner'), 'camera', 'tag')
    if a.uid == b.uid:
        raise EditError('a camera cannot be its own partner')
    for c in (a, b):
        old = v.cam(c.link_uid)
        if old is not None and old.uid not in (a.uid, b.uid):
            old.link_uid = ''
    a.link_uid, b.link_uid = b.uid, a.uid
    touched.add(_variant_key(p, v))
    return f'linked {a.tag} and {b.tag} as mirrored partners (later changes to one mirror to the other)'


def apply(project, ops, seams=None) -> tuple[list, list]:
    """Run the operations in order on project: (undo keys, a line per operation). EditError on the first bad one (the
    project may then be half-changed: run on a snapshot first). seams: the analysis' seams of the active variant
    (AnalysisResult.seams), which seam operations name."""
    if not isinstance(ops, list) or not ops:
        raise EditError('ops: a list of at least one operation')
    if len(ops) > 100:
        raise EditError('ops: at most 100 operations at a time')
    touched, lines, done = set(), [], {}
    for i, op in enumerate(ops):
        if not isinstance(op, dict):
            raise EditError(f'ops[{i}]: an object')
        kind, t = op.get('op'), op.get('target')
        if kind not in OPS:
            raise EditError(f'ops[{i}].op: one of {", ".join(OPS)}')
        if t not in TARGETS:
            raise EditError(f'ops[{i}].target: one of {", ".join(TARGETS)}')
        if kind in ('set', 'remove') and t in LISTED and t != 'variant' and not op.get('id'):
            raise EditError(f'ops[{i}]: id (the {t}\'s tag or name) is needed')
        try:
            if t == 'seam':
                lines.append(_seam(project, op, touched, seams, done))
            elif kind == 'set':
                lines.append(_set(project, op, touched))
            elif kind == 'add':
                lines.append(_add(project, op, touched))
            elif kind == 'remove':
                lines.append(_remove(project, op, touched))
            elif kind == 'activate':
                if t != 'variant':
                    raise EditError('activate is for variants')
                v = _variant(project, op.get('id'))
                project.active_uid = v.uid
                touched.add('variants')
                lines.append(f'active variant: {v.name!r}')
            else:
                lines.append(_link(project, op, touched, kind == 'link'))
        except EditError as e:
            raise EditError(f'ops[{i}] ({kind} {t}): {e}') from None
    if done:
        _check_seam_order(project, seams, done)
    if 'variants' in touched:                       # the whole list covers each variant's own key
        touched = {k for k in touched if not k.startswith('variant:')}
    return sorted(touched), lines


def title(lines) -> str:
    """The undo step's text: 'Claude · ' and what changed, short."""
    t = '; '.join(lines)
    return 'Claude · ' + (t if len(t) <= 90 else t[:89] + '…')


# =========================================================================== reading
def data(project, sections=None) -> dict:
    """The project's editable data (what edit_project sets), section by section: every field a camera, sensor, rig part
    and so on has, the library ids to choose from, and the variants."""
    p = project
    want = set(sections or ('cameras', 'sensors', 'rig', 'variants'))
    known = {'cameras', 'sensors', 'rig', 'obstacles', 'targets', 'variants', 'roi', 'ecu', 'analysis', 'project',
             'library'}
    bad = want - known
    if bad:
        return {'error': f'unknown section(s) {", ".join(sorted(bad))}; sections: {", ".join(sorted(known))}'}
    v = p.variant
    out = {'variant': v.name}
    if 'cameras' in want:
        out['cameras'] = [{k: getattr(c, k) for k in CAMERA_FIELDS} | {'mirrored_with': (v.cam(c.link_uid).tag if
                          v.cam(c.link_uid) else None), 'unit_calibration': bool(c.calib)} for c in v.cameras]
    if 'sensors' in want:
        out['sensors'] = [{'kind': s.kind, **{k: getattr(s, k) for k in SENSOR_FIELDS}} for s in v.sensors]
    if 'rig' in want:
        out['rig'] = [{k: getattr(r, k) for k in RIG_FIELDS} for r in p.rig]
    if 'obstacles' in want:
        out['obstacles'] = [{k: getattr(o, k) for k in OBSTACLE_FIELDS} for o in p.obstacles]
    if 'targets' in want:
        out['targets'] = [{k: getattr(g, k) for k in TARGET_FIELDS} for g in p.targets]
        out['target_settings'] = p.target_settings.to_dict()
    if 'variants' in want:
        out['variants'] = [{'name': x.name, 'note': x.note, 'active': x.uid == p.active_uid,
                            'cameras': [c.tag for c in x.cameras], 'sensors': [s.tag for s in x.sensors]}
                           for x in p.variants]
    if 'roi' in want:
        out['roi'] = p.roi.to_dict()
    if 'ecu' in want:
        out['ecu'] = p.ecu.to_dict()
    if 'analysis' in want:
        a = p.analysis
        out['analysis'] = {k: getattr(a, k) for k in P.AnalysisSettings.__dataclass_fields__
                           if k not in ANALYSIS_LOCKED and not isinstance(getattr(a, k), dict)}
        out['analysis']['study_settings'] = {k: getattr(a, k) for k in P.AnalysisSettings.__dataclass_fields__
                                             if k not in ANALYSIS_LOCKED and isinstance(getattr(a, k), dict)
                                             and getattr(a, k)}
    if 'project' in want:
        out['project'] = {'name': p.name, 'frame': p.frame, 'custom_origin': p.custom_origin, 'units': p.units,
                          'brief': p.analysis.assistant.get('brief', ''),
                          'vehicle': p.vehicle.name or p.vehicle.spec_name}
    if 'library' in want:
        out['library'] = {
            'modules': [{'id': m.id, 'name': m.name} for m in p.modules.values()],
            'lenses': [{'id': l.id, 'name': l.name, 'for_module': l.module_id or 'any'} for l in p.lenses.values()],
            'sensor_models': [{'id': m.id, 'name': m.name, 'kind': m.kind, 'vendor': m.vendor}
                              for m in p.sensor_models.values()],
            'rig_presets': [k for k, _ in R.PRESETS]}
    return out
