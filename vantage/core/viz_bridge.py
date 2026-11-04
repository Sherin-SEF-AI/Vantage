"""Vantage's scene for the visualisers AV researchers live in: Foxglove (an MCAP file of Foxglove's well-known JSON
schemas, or a live WebSocket server when the foxglove-sdk package is installed) and Rerun (when rerun-sdk is
installed: spawn a viewer, connect to one, or save an .rrd file).

What goes out, in the vehicle frame (ISO 8855, rear-axle origin on the ground, metres; frame id 'vehicle'):

- the vehicle mesh, one entity per part (paint, glass, wheels, ...);
- every visible camera: its pose (a transform vehicle -> '<tag>_optical', OpenCV optical axes: x right, y down,
  z forward), its intrinsics (Kannala–Brandt k1..k4 for a fisheye, plumb_bob for a rectilinear lens, OpenCV pixel
  centres at integers), its true field outline (the image border unprojected through its real lens, clipped to the
  image circle, out to `frustum_m`), its ground footprint (the analysis's outline) and its ego mask (the pixels the
  body, the rig or another housing covers, from the engine's occlusion cube maps; white = ego);
- every visible lidar, radar, GNSS antenna and IMU: its housing as a box, a transform vehicle -> '<tag>', a lidar's
  rings where its beams meet the ground, a radar's zones in plan;
- the 3D blind volume (core.volume): its surface as triangles and its blind voxels (pooled to 0.2 m) as boxes.

Foxglove: MCAP (https://mcap.dev/spec) with message encoding 'json' and schema encoding 'jsonschema', the schemas
foxglove.SceneUpdate, foxglove.FrameTransforms, foxglove.CameraCalibration, foxglove.CompressedImage and
foxglove.PointCloud embedded verbatim (from https://github.com/foxglove/foxglove-sdk/tree/main/schemas/jsonschema,
read 2026-11-04; bytes fields are base64 strings as those schemas' contentEncoding says). Topics: /tf,
/vantage/scene, /<tag>/camera_info, /<tag>/ego_mask, /<lidar>/ground_rings. Open the file in Foxglove (File ▸ Open
local file) and add a 3D panel and Image panels. The MCAP records reuse core.rosbag's writer helpers (no
dependency). Live: foxglove.start_server(port=8765) and foxglove.Channel(topic, schema=<JSON schema dict>) from the
foxglove-sdk Python package (0.30; https://github.com/foxglove/foxglove-sdk, python/foxglove-sdk/python/foxglove/
__init__.py and channel.py, read 2026-11-04), feature-detected: never required.

Rerun: rerun-sdk 0.41 (https://ref.rerun.io/docs/python/0.41.0/rerun/ and
https://rerun.io/docs/reference/types/archetypes, read 2026-11-04): rr.init, rr.spawn / rr.connect_grpc / rr.save,
rr.log(path, archetype, static=True) with ViewCoordinates.RIGHT_HAND_Z_UP on 'vehicle', Mesh3D, LineStrips3D,
Transform3D(translation, mat3x3), Pinhole(image_from_camera, resolution, camera_xyz=RDF), Image, Points3D and
Boxes3D. Rerun's Pinhole is a pinhole: a fisheye's frustum there is the pinhole with its K (approximate); the true
field outline is logged beside it as lines, and its Kannala–Brandt terms go on the entity as text.
"""
from __future__ import annotations

import base64
import json
import math
import os
import struct
import sys
import threading
import time
import zlib

import numpy as np

FRAME = 'vehicle'
T0_NS = 1_000_000_000           # every message's log time: 1 s (a static scene)
# the cameras' colours (ui.theme.CAMERA['dark'], copied: core imports no Qt)
CAMERA_HEX = ['#3987e5', '#d95926', '#199e70', '#c98500', '#d55181', '#008300', '#9085e9', '#e66767']
SENSOR_RGBA = {'lidar': (0.30, 0.75, 0.95, 0.9), 'radar': (0.95, 0.55, 0.20, 0.9), 'gnss': (0.75, 0.75, 0.30, 0.9),
               'imu': (0.70, 0.70, 0.70, 0.9)}
PART_RGBA = {'glass': (0.10, 0.12, 0.14, 0.75), 'wheels': (0.08, 0.08, 0.08, 1.0), 'paint': (0.78, 0.78, 0.76, 1.0)}

_SCHEMAS_Z = ''.join((
    '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'
))
_SCHEMAS = None


def schemas() -> dict:
    """{name: JSON schema} of the Foxglove schemas used (embedded verbatim, zlib + base64)."""
    global _SCHEMAS
    if _SCHEMAS is None:
        _SCHEMAS = json.loads(zlib.decompress(base64.b64decode(_SCHEMAS_Z)).decode())
    return _SCHEMAS


def _hex_rgba(h, a=1.0):
    h = h.lstrip('#')
    return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)) + (a,)


def _quat(R) -> tuple:
    """(x, y, z, w) of a rotation matrix."""
    R = np.asarray(R, float)
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        w, x, y, z = 0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w, x, y, z = (R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w, x, y, z = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w, x, y, z = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s
    q = np.array([x, y, z, w])
    q /= np.linalg.norm(q)
    return tuple(float(v) for v in (q if q[3] >= 0 else -q))


# ---------------------------------------------------------------- the scene
def _distortion(it):
    """(model, D, approximate) of a resolved lens for CameraCalibration / OpenCV."""
    from .lens import r_of_theta
    if it.projection == 'rectilinear':
        return 'plumb_bob', [0.0] * 5, False
    if it.projection in ('kannala_brandt', 'equidistant'):
        kb = list(it.kb) if it.projection == 'kannala_brandt' else [0.0] * 4
        return 'kannala_brandt', [float(v) for v in kb], False
    # another base projection: the Kannala–Brandt terms that best reproduce r(θ) over the field
    th = np.linspace(1e-3, max(it.max_theta, 0.05), 200)
    td = r_of_theta(th, it.f_mm, it.projection, it.kb) / it.f_mm
    A = np.stack([th ** 3, th ** 5, th ** 7, th ** 9], 1)
    k, *_ = np.linalg.lstsq(A, td - th, rcond=None)
    return 'kannala_brandt', [float(v) for v in k], True


def _field_outline(it, n=32):
    """Unit optical-frame directions round the image's border (clockwise from the top left), each clipped to the
    image circle: the camera's true field outline."""
    W, H = it.width, it.height
    t = np.linspace(0, 1, n, endpoint=False)
    u = np.concatenate([0.5 + t * (W - 1), np.full(n, W - 0.5), W - 0.5 - t * (W - 1), np.full(n, 0.5)])
    v = np.concatenate([np.full(n, 0.5), 0.5 + t * (H - 1), np.full(n, H - 0.5), H - 0.5 - t * (H - 1)])
    d, th = it.unproject(u, v)
    d = np.asarray(d, float)
    th = np.asarray(th, float)
    out = ~np.isfinite(th) | (th > it.max_theta)
    if out.any():                                   # beyond the image circle: the circle's edge, same azimuth
        az = np.arctan2((v - it.cy) / it.aspect, u - it.cx)
        m = it.max_theta
        d[out] = np.c_[np.sin(m) * np.cos(az[out]), np.sin(m) * np.sin(az[out]), np.full(out.sum(), math.cos(m))]
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def _ego_mask(project, engine, cam, it, W2O, width):
    """(h, w) bool: the pixels the body, the rig or a housing covers (the engine's occlusion cube at the camera)."""
    w = int(min(width, it.width))
    h = int(round(w * it.height / it.width))
    us = (np.arange(w) + 0.5) / w * it.width
    vs = (np.arange(h) + 0.5) / h * it.height
    U, V = np.meshgrid(us, vs)
    d, th = it.unproject(U.ravel(), V.ravel())
    d = np.asarray(d, float)
    valid = np.isfinite(th) & (np.asarray(th) <= it.max_theta)
    dw = d @ W2O
    pos = cam.pos()
    with np.errstate(divide='ignore', invalid='ignore'):
        tg = np.where(dw[:, 2] < -1e-6, -pos[2] / dw[:, 2], np.inf)
    body_d, _ = engine.cube(pos).lookup(dw)
    ego = valid & np.isfinite(body_d) & (body_d < tg)
    return ego.reshape(h, w)


def scene_entities(project, engine, result=None, frustum_m=2.0, mask_width=480, volume=True) -> dict:
    """The scene, pure geometry in the vehicle frame (metres): {'frame', 'mesh': [{name, kind, vertices (N,3),
    faces (M,3), rgba}], 'cameras': [{tag, uid, slot, pos, R (optical -> vehicle), width, height, K (3,3), D,
    distortion_model, approximate, outline (K,3), edges [(2,3)], footprint (S,2,3), mask (h,w) bool, rgba}],
    'sensors': [{tag, uid, kind, model, pos, R, box (centre, half, R), rings [(channel, (N,3))], wedges [(N,3)],
    rgba}], 'blind': {mesh (V, F), centres (K,3), half (3,)} or None, 'transforms': [{parent, child, translation,
    R}]}."""
    from .analysis import camera_frames
    from .lens import resolve
    from . import sensors as S
    if result is None:
        result = engine.analyse(project)
    v = project.variant
    out = {'frame': FRAME, 'mesh': [], 'cameras': [], 'sensors': [], 'blind': None, 'transforms': []}
    m = project.vehicle.mesh
    if m is not None:
        for i, part in enumerate(m.parts):
            sel = m.face_part == i
            if not sel.any() or part.role == 'ignored':
                continue
            used = np.unique(m.faces[sel])
            remap = np.full(len(m.vertices), -1, np.int64)
            remap[used] = np.arange(len(used))
            name = part.name.lower()
            kind = 'glass' if part.role == 'glass' or 'glass' in name else \
                'wheels' if part.role == 'wheels' else 'paint'
            out['mesh'].append({'name': part.name, 'kind': kind, 'vertices': m.vertices[used].astype(np.float32),
                                'faces': remap[m.faces[sel]].astype(np.int32), 'rgba': PART_RGBA[kind]})
    segs = {c.uid: np.asarray(c.segments, float) for c in getattr(result, 'cams', [])} if result is not None else {}
    gz = float(getattr(result, 'z', 0.0) or 0.0) if result is not None else 0.0
    for cam in v.cameras:
        if not cam.visible:
            continue
        it = resolve(project.module_for(cam), project.lens_for(cam))
        R, W2O = camera_frames(cam)
        Ro = np.asarray(W2O, float).T                               # optical -> vehicle
        pos = np.asarray(cam.pos(), float)
        model, D, approx = _distortion(it)
        K = np.array([[it.f_px, 0, it.cx - 0.5], [0, it.f_px * it.aspect, it.cy - 0.5], [0, 0, 1.0]])
        ol = pos + (_field_outline(it) @ Ro.T) * frustum_m
        n4 = len(ol) // 4
        edges = [np.array([pos, ol[k * n4]]) for k in range(4)]
        fp = segs.get(cam.uid, np.zeros((0, 2, 2)))
        fp3 = np.concatenate([fp, np.full(fp.shape[:2] + (1,), gz)], axis=2) if len(fp) else np.zeros((0, 2, 3))
        try:
            mask = _ego_mask(project, engine, cam, it, W2O, mask_width) if mask_width else None
        except Exception:                                          # no vehicle geometry: no mask
            mask = None
        out['cameras'].append({'tag': cam.tag, 'uid': cam.uid, 'slot': cam.slot, 'pos': pos, 'R': Ro,
                               'width': it.width, 'height': it.height, 'K': K, 'D': D, 'distortion_model': model,
                               'approximate': approx, 'outline': np.vstack([ol, ol[:1]]), 'edges': edges,
                               'footprint': fp3, 'mask': mask,
                               'rgba': _hex_rgba(CAMERA_HEX[cam.slot % len(CAMERA_HEX)])})
        out['transforms'].append({'parent': FRAME, 'child': f'{cam.tag}_optical', 'translation': pos, 'R': Ro})
    for s in v.sensors:
        if not s.visible:
            continue
        md = project.sensor_model_for(s)
        Rs = S.rotation(s)
        c, half, Rb, _ = S.body_box(s, md)
        e = {'tag': s.tag, 'uid': s.uid, 'kind': s.kind, 'model': md.id, 'pos': s.pos(), 'R': Rs,
             'box': (c, half, Rb), 'rings': [], 'wedges': [], 'rgba': SENSOR_RGBA.get(s.kind, (0.8, 0.8, 0.8, 0.9))}
        if s.kind == 'lidar':
            e['rings'] = S.ground_rings(s, md, az_step=1.0, far=min(md.range_m or 30.0, 30.0))
        elif s.kind == 'radar':
            e['wedges'] = [np.c_[w, np.full(len(w), 0.02)] for w in S.radar_wedges(s, md)]
        out['sensors'].append(e)
        out['transforms'].append({'parent': FRAME, 'child': s.tag, 'translation': np.asarray(s.pos(), float),
                                  'R': Rs})
    if volume and m is not None:
        try:
            from . import volume as VO
            vr = VO.analyse_volume(project, engine, res=result, height=2.0, dz=0.1, cell=0.1)
            if vr is not None:
                V, F = VO.blind_surface(vr, pool=1, caps=False, body_walls=False)
                b = np.asarray(vr.blind, bool)
                nz, ny, nx = b.shape
                bp = np.zeros((nz, (ny + 1) // 2 * 2, (nx + 1) // 2 * 2), bool)
                bp[:, :ny, :nx] = b
                bp = bp.reshape(nz, bp.shape[1] // 2, 2, bp.shape[2] // 2, 2).any(axis=(2, 4))   # 0.2 m columns
                kz, ky, kx = np.nonzero(bp)
                cen = np.c_[vr.x0 + (kx * 2 + 1.0) * vr.cell, vr.y0 + (ky * 2 + 1.0) * vr.cell,
                            (kz + 0.5) * vr.dz]
                out['blind'] = {'mesh': (np.asarray(V, np.float32), np.asarray(F, np.int32)), 'centres': cen,
                                'half': np.array([vr.cell, vr.cell, vr.dz / 2])}
        except Exception as ex:                    # the blind volume is optional
            out['blind_error'] = str(ex)
    return out


# ---------------------------------------------------------------- Foxglove messages
def _t():
    return {'sec': T0_NS // 1_000_000_000, 'nsec': T0_NS % 1_000_000_000}


def _xyz(p):
    return {'x': float(p[0]), 'y': float(p[1]), 'z': float(p[2])}


def _rgba(c):
    return {'r': float(c[0]), 'g': float(c[1]), 'b': float(c[2]), 'a': float(c[3])}


_IDENT = {'position': {'x': 0.0, 'y': 0.0, 'z': 0.0}, 'orientation': {'x': 0.0, 'y': 0.0, 'z': 0.0, 'w': 1.0}}


def _entity(eid, **prims):
    e = {'timestamp': _t(), 'frame_id': FRAME, 'id': eid, 'lifetime': {'sec': 0, 'nsec': 0}, 'frame_locked': False,
         'metadata': [], 'arrows': [], 'cubes': [], 'spheres': [], 'cylinders': [], 'lines': [], 'triangles': [],
         'texts': [], 'models': []}
    for k, v in prims.items():
        if k == 'metadata':
            e['metadata'] = [{'key': a, 'value': str(b)} for a, b in v.items()]
        else:
            e[k] = v
    return e


def _line(points, rgba, kind=0, thickness=2.0):
    """kind 0 LINE_STRIP, 1 LINE_LOOP, 2 LINE_LIST (foxglove.LineType); thickness in pixels (scale invariant)."""
    return {'type': kind, 'pose': _IDENT, 'thickness': thickness, 'scale_invariant': True,
            'points': [_xyz(p) for p in np.asarray(points, float).reshape(-1, 3)], 'color': _rgba(rgba),
            'colors': [], 'indices': []}


def _cube(centre, half, R, rgba):
    return {'pose': {'position': _xyz(centre), 'orientation': dict(zip('xyzw', _quat(R)))},
            'size': _xyz(2 * np.asarray(half, float)), 'color': _rgba(rgba)}


def _tris(V, F, rgba):
    return {'pose': _IDENT, 'points': [_xyz(p) for p in np.asarray(V, float)], 'color': _rgba(rgba), 'colors': [],
            'indices': [int(i) for i in np.asarray(F).ravel()]}


def foxglove_messages(ents) -> list:
    """[(topic, schema name, message dict)] of a scene_entities() dict, in Foxglove's JSON schemas."""
    msgs = []
    msgs.append(('/tf', 'foxglove.FrameTransforms', {'transforms': [
        {'timestamp': _t(), 'parent_frame_id': tf['parent'], 'child_frame_id': tf['child'],
         'translation': _xyz(tf['translation']), 'rotation': dict(zip('xyzw', _quat(tf['R'])))}
        for tf in ents['transforms']]}))
    E = []
    for part in ents['mesh']:
        E.append(_entity(f'vehicle/{part["name"]}', triangles=[_tris(part['vertices'], part['faces'], part['rgba'])],
                         metadata={'kind': part['kind']}))
    for c in ents['cameras']:
        lines = [_line(c['outline'], c['rgba'], 0, 2.0)]
        lines.append(_line(np.concatenate(c['edges']), c['rgba'], 2, 1.0))
        texts = [{'pose': {'position': _xyz(c['pos'] + np.array([0, 0, 0.12])), 'orientation': _IDENT['orientation']},
                  'billboard': True, 'font_size': 14.0, 'scale_invariant': True, 'color': _rgba(c['rgba']),
                  'text': c['tag']}]
        E.append(_entity(f'cameras/{c["tag"]}/field', lines=lines, texts=texts,
                         metadata={'distortion_model': c['distortion_model'], 'approximate': c['approximate']}))
        if len(c['footprint']):
            E.append(_entity(f'cameras/{c["tag"]}/footprint', lines=[_line(c['footprint'], c['rgba'], 2, 1.5)]))
    for s in ents['sensors']:
        cen, half, Rb = s['box']
        lines = [_line(r, s['rgba'], 0, 1.0) for _, r in s['rings']] + [_line(w, s['rgba'], 1, 1.5)
                                                                        for w in s['wedges']]
        E.append(_entity(f'sensors/{s["tag"]}', cubes=[_cube(cen, half, Rb, s['rgba'])], lines=lines,
                         metadata={'kind': s['kind'], 'model': s['model']}))
    b = ents.get('blind')
    if b:
        V, F = b['mesh']
        if len(F):
            E.append(_entity('blind_volume/surface', triangles=[_tris(V, F, (0.82, 0.23, 0.23, 0.45))]))
        E.append(_entity('blind_volume/voxels', cubes=[
            {'pose': {'position': _xyz(p), 'orientation': _IDENT['orientation']}, 'size': _xyz(2 * b['half']),
             'color': _rgba((0.82, 0.23, 0.23, 0.25))} for p in b['centres']]))
    msgs.append(('/vantage/scene', 'foxglove.SceneUpdate', {'deletions': [], 'entities': E}))
    for c in ents['cameras']:
        K = np.asarray(c['K'], float)
        P = np.c_[K, np.zeros(3)]
        msgs.append((f'/{c["tag"]}/camera_info', 'foxglove.CameraCalibration', {
            'timestamp': _t(), 'frame_id': f'{c["tag"]}_optical', 'width': int(c['width']), 'height': int(c['height']),
            'distortion_model': c['distortion_model'], 'D': [float(x) for x in c['D']],
            'K': [float(x) for x in K.ravel()], 'R': [1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0],
            'P': [float(x) for x in P.ravel()]}))
        if c['mask'] is not None:
            msgs.append((f'/{c["tag"]}/ego_mask', 'foxglove.CompressedImage', {
                'timestamp': _t(), 'frame_id': f'{c["tag"]}_optical', 'format': 'png',
                'data': base64.b64encode(_png(c['mask'])).decode('ascii')}))
    for s in ents['sensors']:
        if s['rings']:
            P = np.concatenate([r for _, r in s['rings']]).astype('<f4')
            msgs.append((f'/{s["tag"]}/ground_rings', 'foxglove.PointCloud', {
                'timestamp': _t(), 'frame_id': FRAME, 'pose': _IDENT, 'point_stride': 12,
                'fields': [{'name': n, 'offset': 4 * i, 'type': 7} for i, n in enumerate('xyz')],     # 7 = FLOAT32
                'data': base64.b64encode(P.tobytes()).decode('ascii')}))
    return msgs


def _png(mask) -> bytes:
    """A mask as an 8-bit greyscale PNG (zlib, no imaging library needed)."""
    a = (np.asarray(mask, bool).astype(np.uint8) * 255)
    h, w = a.shape
    raw = b''.join(b'\x00' + a[y].tobytes() for y in range(h))

    def chunk(t, d):
        return struct.pack('>I', len(d)) + t + d + struct.pack('>I', zlib.crc32(t + d) & 0xffffffff)
    return (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 0, 0, 0, 0))
            + chunk(b'IDAT', zlib.compress(raw, 6)) + chunk(b'IEND', b''))


def write_mcap(path, msgs, library='vantage') -> str:
    """An MCAP file of [(topic, schema name, message dict)] with JSON messages and jsonschema schemas (data section,
    then a summary of the schemas, channels and statistics so viewers index it)."""
    from .rosbag import MCAP_MAGIC, OP_HEADER, OP_FOOTER, OP_SCHEMA, OP_CHANNEL, OP_MESSAGE, OP_DATA_END, \
        OP_STATISTICS, OP_SUMMARY_OFFSET, _rec, _s, _map
    sch = schemas()
    sids, cids, recs_s, recs_c = {}, {}, [], []
    body = bytearray(MCAP_MAGIC)
    body += _rec(OP_HEADER, _s('') + _s(library))
    seq = {}
    for topic, name, msg in msgs:
        if name not in sids:
            sids[name] = len(sids) + 1
            data = json.dumps(sch[name], separators=(',', ':')).encode()
            r = _rec(OP_SCHEMA, struct.pack('<H', sids[name]) + _s(name) + _s('jsonschema')
                     + struct.pack('<I', len(data)) + data)
            recs_s.append(r)
            body += r
        if topic not in cids:
            cids[topic] = len(cids) + 1
            r = _rec(OP_CHANNEL, struct.pack('<HH', cids[topic], sids[name]) + _s(topic) + _s('json') + _map({}))
            recs_c.append(r)
            body += r
        data = json.dumps(msg, separators=(',', ':')).encode()
        n = seq.get(topic, 0)
        seq[topic] = n + 1
        body += _rec(OP_MESSAGE, struct.pack('<HIQQ', cids[topic], n, T0_NS, T0_NS) + data)
    body += _rec(OP_DATA_END, struct.pack('<I', 0))
    summary_start = len(body)
    groups = []
    for op, recs in ((OP_SCHEMA, recs_s), (OP_CHANNEL, recs_c)):
        g0 = len(body)
        for r in recs:
            body += r
        groups.append((op, g0, len(body) - g0))
    g0 = len(body)
    counts = b''.join(struct.pack('<HQ', cid, seq[t]) for t, cid in cids.items())
    stats = struct.pack('<QHIIIIQQ', sum(seq.values()), len(sids), len(cids), 0, 0, 0, T0_NS, T0_NS) + \
        struct.pack('<I', len(counts)) + counts
    body += _rec(OP_STATISTICS, stats)
    groups.append((OP_STATISTICS, g0, len(body) - g0))
    offsets_start = len(body)
    for op, a, ln in groups:
        body += _rec(OP_SUMMARY_OFFSET, struct.pack('<BQQ', op, a, ln))
    body += _rec(OP_FOOTER, struct.pack('<QQI', summary_start, offsets_start, 0))
    body += MCAP_MAGIC
    os.makedirs(os.path.dirname(os.path.abspath(path)) or '.', exist_ok=True)
    with open(path, 'wb') as f:
        f.write(body)
    return path


def read_mcap(path) -> list:
    """[(topic, schema name, schema encoding, message encoding, message dict)] of a JSON MCAP file (for checks)."""
    from .rosbag import MCAP_MAGIC, OP_SCHEMA, OP_CHANNEL, OP_MESSAGE, OP_FOOTER, _records, _get_s, _get_map
    buf = open(path, 'rb').read()
    if not (buf.startswith(MCAP_MAGIC) and buf.endswith(MCAP_MAGIC)):
        raise ValueError('not an MCAP file')
    data = buf[len(MCAP_MAGIC):-len(MCAP_MAGIC)]
    schemas_, channels, out = {}, {}, []
    for op, p, ln in _records(data):
        if op == OP_SCHEMA:
            sid, = struct.unpack_from('<H', data, p)
            name, q = _get_s(data, p + 2)
            enc, q = _get_s(data, q)
            schemas_[sid] = (name, enc)
        elif op == OP_CHANNEL:
            cid, sid = struct.unpack_from('<HH', data, p)
            topic, q = _get_s(data, p + 4)
            menc, q = _get_s(data, q)
            channels[cid] = (topic, sid, menc)
        elif op == OP_MESSAGE:
            cid, = struct.unpack_from('<H', data, p)
            topic, sid, menc = channels[cid]
            msg = json.loads(bytes(data[p + 22:p + ln]).decode())
            out.append((topic, schemas_[sid][0], schemas_[sid][1], menc, msg))
        elif op == OP_FOOTER:
            break
    return out


def export_foxglove_mcap(project, engine, path, result=None, **kw) -> str:
    """Write the scene as an MCAP file for Foxglove (File ▸ Open local file). Returns the path."""
    return write_mcap(path, foxglove_messages(scene_entities(project, engine, result=result, **kw)))


def foxglove_available() -> bool:
    try:
        import foxglove                                    # noqa: F401  (the foxglove-sdk package)
        return hasattr(foxglove, 'start_server') and hasattr(foxglove, 'Channel')
    except Exception:
        return False


class FoxgloveLive:
    """A live Foxglove WebSocket server (foxglove-sdk) publishing the scene, sent again every `period_s` so a client
    that connects later gets it. stop() ends it."""

    def __init__(self, server, channels, msgs, period_s=2.0):
        self.server, self.channels, self.msgs = server, channels, msgs
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._run, args=(period_s,), daemon=True)
        self.thread.start()

    def _run(self, period):
        while not self._stop.is_set():
            now = time.time_ns()
            for topic, _, msg in self.msgs:
                try:
                    self.channels[topic].log(msg, log_time=now)
                except Exception:
                    pass
            self._stop.wait(period)

    def stop(self):
        self._stop.set()
        try:
            self.server.stop()
        except Exception:
            pass


def serve_foxglove(project, engine, port=8765, result=None, period_s=2.0, **kw):
    """(FoxgloveLive, message) with the foxglove-sdk package installed; (None, how to install it) without."""
    if not foxglove_available():
        return None, ('Live Foxglove needs the foxglove-sdk package (pip install foxglove-sdk); without it, export '
                      'the MCAP file and open it in Foxglove.')
    import foxglove
    msgs = foxglove_messages(scene_entities(project, engine, result=result, **kw))
    server = foxglove.start_server(port=int(port))
    sch = schemas()
    channels = {}
    for topic, name, _ in msgs:
        if topic not in channels:
            channels[topic] = foxglove.Channel(topic, schema=sch[name])
    return FoxgloveLive(server, channels, msgs, period_s), f'Foxglove server on ws://127.0.0.1:{int(port)}'


# ---------------------------------------------------------------- Rerun
def rerun_available() -> bool:
    try:
        import rerun                                       # noqa: F401  (the rerun-sdk package)
        return hasattr(rerun, 'init') and hasattr(rerun, 'log')
    except Exception:
        return False


def _u8(rgba):
    return [int(round(255 * x)) for x in rgba]


def log_rerun(rr, ents, root=FRAME) -> list:
    """Log a scene_entities() dict to an initialised rerun module; returns the entity paths logged."""
    paths = []

    def log(path, arch):
        rr.log(path, arch, static=True)
        paths.append(path)
    log(root, rr.ViewCoordinates.RIGHT_HAND_Z_UP)
    for part in ents['mesh']:
        log(f'{root}/mesh/{part["name"]}', rr.Mesh3D(vertex_positions=part['vertices'],
                                                     triangle_indices=part['faces'],
                                                     albedo_factor=_u8(part['rgba'])))
    for c in ents['cameras']:
        base = f'{root}/cameras/{c["tag"]}'
        log(base, rr.Transform3D(translation=np.asarray(c['pos'], float), mat3x3=np.asarray(c['R'], float)))
        log(f'{base}/image', rr.Pinhole(image_from_camera=np.asarray(c['K'], float),
                                        resolution=[int(c['width']), int(c['height'])],
                                        camera_xyz=rr.ViewCoordinates.RDF))
        log(f'{base}/lens', rr.TextDocument(f'{c["distortion_model"]} D = {[round(x, 6) for x in c["D"]]}'
                                            + (' (fitted)' if c['approximate'] else '')
                                            + '; the Pinhole frustum is approximate for a fisheye: see field'))
        if c['mask'] is not None:
            log(f'{base}/image/ego_mask', rr.Image(c['mask'].astype(np.uint8) * 255))
        strips = [np.asarray(c['outline'], float)] + [np.asarray(e, float) for e in c['edges']]
        log(f'{root}/fields/{c["tag"]}', rr.LineStrips3D(strips, colors=[_u8(c['rgba'])] * len(strips)))
        if len(c['footprint']):
            log(f'{root}/footprints/{c["tag"]}', rr.LineStrips3D(list(np.asarray(c['footprint'], float)),
                                                                 colors=[_u8(c['rgba'])] * len(c['footprint'])))
    for s in ents['sensors']:
        base = f'{root}/sensors/{s["tag"]}'
        cen, half, Rb = s['box']
        log(base, rr.Boxes3D(centers=[cen], half_sizes=[half], quaternions=[rr.Quaternion(xyzw=_quat(Rb))],
                             colors=[_u8(s['rgba'])], labels=[s['tag']]))
        if s['rings']:
            log(f'{base}/ground_rings', rr.LineStrips3D([r for _, r in s['rings']], colors=[_u8(s['rgba'])]))
        if s['wedges']:
            log(f'{base}/zones', rr.LineStrips3D([np.vstack([w, w[:1]]) for w in s['wedges']],
                                                 colors=[_u8(s['rgba'])]))
    b = ents.get('blind')
    if b:
        V, F = b['mesh']
        if len(F):
            log(f'{root}/blind_volume/surface', rr.Mesh3D(vertex_positions=V, triangle_indices=F,
                                                          albedo_factor=[209, 59, 59, 110]))
        if len(b['centres']):
            log(f'{root}/blind_volume/voxels', rr.Boxes3D(centers=b['centres'],
                                                          half_sizes=np.tile(b['half'], (len(b['centres']), 1)),
                                                          colors=[[209, 59, 59, 60]]))
    return paths


def stream_rerun(project, engine, mode='spawn', path=None, result=None, url=None, **kw) -> str:
    """Send the scene to Rerun: mode 'spawn' (start a viewer), 'connect' (a running viewer at url, else Rerun's
    default rerun+http://127.0.0.1:9876/proxy) or 'save' (an .rrd file at path). Returns a status line; without
    rerun-sdk, how to install it."""
    if not rerun_available():
        return 'Rerun is not installed: pip install rerun-sdk (then try again)'
    import rerun as rr
    if mode == 'save' and not path:
        raise ValueError('mode save needs a path for the .rrd file')
    ents = scene_entities(project, engine, result=result, **kw)
    rr.init('vantage', spawn=False)
    if mode == 'spawn':
        rr.spawn()
    elif mode == 'connect':
        rr.connect_grpc(url) if url else rr.connect_grpc()
    elif mode == 'save':
        rr.save(path)
    else:
        raise ValueError(f'unknown mode {mode!r}: spawn, connect or save')
    n = len(log_rerun(rr, ents))
    where = {'spawn': 'a new Rerun viewer', 'connect': url or 'the Rerun viewer', 'save': path}[mode]
    return f'Sent {n} entities to {where}'
