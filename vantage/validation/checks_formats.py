"""Checks of the dataset calibration files: each file read the way the dataset's own tools read it and points projected
through the numbers written, by the format's documented formula, must land where Vantage's lens model puts them."""
from __future__ import annotations

import json
import math
import os

import numpy as np

from .suite import Outcome, check

TOL_PX = 0.05
ORIGIN = np.array([1.0, 0.2, 0.3])          # a custom export origin, so every file must honour it
IDS = ['nuscenes_calib', 'kitti_calib', 'ros_camera_info', 'kalibr_camchain']


def _reference_rig(ctx):
    """The sample's camera rig with one camera unit-calibrated (fx ≠ fy, an offset principal point), one rectilinear
    and one a 210° heated fisheye (a Kalibr wide-angle model), so every path of the writers is exercised."""
    from ..core import calib, exporters as ex
    from ..core.lens import Lens
    q = ctx.sample()
    cams = q.variant.cameras
    base = ex.camera_model(q, cams[1])
    cams[1].calib = calib.make_record(base.module.width_px, base.module.height_px, base.fx * 1.004, base.fx * 1.011,
                                      base.cx + 3.7, base.cy - 6.3,
                                      [base.k[0] + 0.002, base.k[1] - 0.004, base.k[2], base.k[3]], rms_px=0.21,
                                      serial='SN-VAL-1', source='validation')
    q.lenses['val_rect'] = Lens(id='val_rect', name='RECT60', focal_mm=4.2, projection='rectilinear',
                                sheet_fov=(60.0, 48.0, 75.0))
    cams[4].lens_id = 'val_rect'
    cams[3].module_id, cams[3].lens_id = 'sturdecam31fsh', 'sturdecam31fsh_heated'
    return q


def _cases(ctx):
    """[(label, project, export folder)]: the reference rig, and the project given to the suite when it has cameras.
    Exported once (custom origin) and shared by the format checks."""
    if 'formats' in ctx.cache:
        return ctx.cache['formats']
    from ..core import exporters as ex
    out = []
    todo = [('reference rig', _reference_rig(ctx))]
    if ctx.given and ctx.project.variant.cameras and ctx.engine.vehicle_geom(ctx.project) is not None:
        todo.append(('project', ctx.project.snapshot()))
    for k, (label, q) in enumerate(todo):
        q.custom_origin = list(ORIGIN)
        res = ctx.engine.analyse(q)
        d = os.path.join(ctx.tmp(), f'formats{k}')
        ex.export(IDS, q, res, ex.ExportOptions(out_dir=d, frame='custom'))
        out.append((label, q, d))
    ctx.cache['formats'] = out
    return out


def _views(q, cam, rng, n=60):
    """Vehicle-frame points over the camera's view (pixels spread over the image, up to 80° off the axis, 60° for a
    rectilinear lens, 1–20 m out), their optical-frame coordinates and the app's pixels in OpenCV's convention."""
    from ..core.analysis import camera_frames
    from ..core.lens import resolve
    it = resolve(q.module_for(cam), q.lens_for(cam))
    _, W2O = camera_frames(cam)
    lim = math.radians(60.0 if it.projection == 'rectilinear' else 80.0)
    u = rng.uniform(0.02, 0.98, 4 * n) * it.width
    v = rng.uniform(0.02, 0.98, 4 * n) * it.height
    d, th = it.unproject(u, v)
    keep = (th < min(lim, it.max_theta - math.radians(0.5)))
    d = d[keep][:n]
    Pc = d * rng.uniform(1.0, 20.0, len(d))[:, None]
    P = cam.pos() + Pc @ W2O
    uu, vv, _, ok = it.project_dirs(Pc)
    return P[ok], Pc[ok], np.stack([uu[ok], vv[ok]], 1) - 0.5


def _project(Pc, K, D, model):
    """Optical-frame points to pixels as a dataset loader does: cv2.fisheye for Kannala-Brandt (equidistant), the
    pinhole for an undistorted lens."""
    import cv2
    K = np.asarray(K, float).reshape(3, 3)
    if model in ('equidistant', 'kannala_brandt'):
        uv, _ = cv2.fisheye.projectPoints(Pc.reshape(-1, 1, 3).astype(np.float64), np.zeros(3), np.zeros(3), K,
                                          np.asarray(D, float).ravel()[:4])
        return uv.reshape(-1, 2)
    if np.any(np.asarray(D, float)):
        raise ValueError(f'unexpected distortion for a {model} lens')
    h = Pc @ K.T
    return h[:, :2] / h[:, 2:]


def _quat_matrix(w, x, y, z):
    """Rotation matrix of a unit quaternion given w first (pyquaternion's and nuScenes' order)."""
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _kitti_read(path):
    """pykitti's read_calib_file: 'key: numbers' lines; values that are not numbers are kept apart."""
    data, raw = {}, {}
    with open(path, encoding='utf-8') as fh:
        for line in fh:
            if ':' not in line:
                continue
            key, value = line.split(':', 1)
            raw[key] = value.strip()
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass
    return data, raw


def _run(ctx, cid, fn):
    """fn(q, folder, rng) -> {camera: px error} per case (the reference rig, the project given); the worst over
    them. The points come from the check's own generator."""
    rows, worst = {}, 0.0
    rng = ctx.rng(cid)
    for label, q, d in _cases(ctx):
        errs = fn(q, d, rng)
        rows[label] = {k: round(v, 6) for k, v in errs.items()}
        worst = max(worst, max(errs.values(), default=0.0))
    return worst, rows


@check('formats.nuscenes', 'formats', 'nuScenes calibrated_sensor rows',
       'largest pixel distance between where a point lands through a camera\'s calibrated_sensor row (translation, '
       'the w-first sensor-to-ego quaternion transposed, camera_intrinsic and the Kannala–Brandt coefficients through '
       'cv2.fisheye) and where Vantage\'s lens model puts it, every camera, custom export origin',
       'the nuScenes devkit\'s convention (points into the sensor frame by −translation, then the rotation '
       'transposed); OpenCV\'s fisheye projection', TOL_PX, 'px', refs=('nuscenes2020', 'opencv', 'rep103'),
       cost='study')
def _nuscenes(ctx):
    def fn(q, d, rng):
        rows = json.load(open(os.path.join(d, 'nuscenes', 'calibrated_sensor.json'), encoding='utf-8'))
        by_tag = {r['vantage_distortion']['camera']: r for r in rows if 'vantage_distortion' in r}
        out = {}
        for c in q.variant.cameras:
            r = by_tag[c.tag]
            P, _, uv_app = _views(q, c, rng)
            if not len(P):
                continue
            w, x, y, z = r['rotation']
            if w < 0 or abs(w * w + x * x + y * y + z * z - 1) > 1e-8:
                raise ValueError(f'{c.tag}: the quaternion is not unit and w-first')
            Pc = (P - ORIGIN - np.array(r['translation'])) @ _quat_matrix(w, x, y, z)
            dist = r['vantage_distortion']
            uv = _project(Pc, r['camera_intrinsic'], dist['coefficients'] or [0.0] * 4, dist['model'])
            out[c.tag] = float(np.linalg.norm(uv - uv_app, axis=1).max())
        return out
    worst, rows = _run(ctx, 'formats.nuscenes', fn)
    return Outcome(worst, rows)


@check('formats.kitti', 'formats', 'KITTI calib_cam_to_cam and calib_cams_to_vehicle',
       'largest pixel distance between where a point lands through KITTI\'s files (vehicle → camera 00 by the inverse '
       'of Tr_cam_00_to_vehicle, camera 00 → camera n by R_n, T_n, then K_n and D_n by the stated distortion model) '
       'and '
       'where Vantage\'s lens model puts it; P_rect_n must equal K_n [R_n | T_n]',
       'the KITTI raw devkit\'s (and pykitti\'s) reading of the files', TOL_PX, 'px',
       refs=('kitti2013', 'opencv'), cost='study')
def _kitti(ctx):
    prect = [0.0]

    def fn(q, d, rng):
        data, raw = _kitti_read(os.path.join(d, 'kitti', 'calib_cam_to_cam.txt'))
        veh, _ = _kitti_read(os.path.join(d, 'kitti', 'calib_cams_to_vehicle.txt'))
        T00 = np.r_[veh['Tr_cam_00_to_vehicle'].reshape(3, 4), [[0, 0, 0, 1]]]
        names = raw['camera_names'].split()
        out = {}
        for c in q.variant.cameras:
            n = f'{names.index(c.tag):02d}'
            P, Pc_app, uv_app = _views(q, c, rng)
            if not len(P):
                continue
            K, R, T = data[f'K_{n}'].reshape(3, 3), data[f'R_{n}'].reshape(3, 3), data[f'T_{n}']
            prect[0] = max(prect[0], float(np.abs(data[f'P_rect_{n}'].reshape(3, 4) - K @ np.c_[R, T]).max()
                                     / max(float(np.abs(K).max()), 1.0)))
            X0 = (np.linalg.inv(T00) @ np.c_[P - ORIGIN, np.ones(len(P))].T)[:3].T
            X = X0 @ R.T + T
            D, model = data[f'D_{n}'], raw[f'DISTORTION_MODEL_{n}']
            uv = _project(X, K, D[:4] if model == 'equidistant' else D, model)
            out[c.tag] = float(np.linalg.norm(uv - uv_app, axis=1).max())
        return out
    worst, rows = _run(ctx, 'formats.kitti', fn)
    return Outcome(worst, dict(rows, p_rect_rel_difference=prect[0]), ok=prect[0] < 1e-6)


@check('formats.ros_camera_info', 'formats', 'ROS camera_info files',
       'largest pixel distance between where a point (in the camera\'s optical frame, REP 103) lands through '
       'camera_matrix and distortion_coefficients by the stated distortion_model (equidistant through cv2.fisheye, '
       'plumb_bob through the pinhole) and where Vantage\'s lens model puts it; the projection matrix must be [K | 0] '
       'and the rectification the identity',
       'sensor_msgs/CameraInfo and the camera_calibration_parsers YAML layout; OpenCV', TOL_PX, 'px',
       refs=('ros_camera_info', 'rep103', 'opencv'), cost='study')
def _ros(ctx):
    import yaml
    from ..core import exporters as ex
    layout = [True]

    def fn(q, d, rng):
        tags = ex.safe_tags(q.variant)
        out = {}
        for c in q.variant.cameras:
            doc = yaml.safe_load(open(os.path.join(d, 'camera_info', f'{tags[c.uid]}.yaml'), encoding='utf-8'))
            K = np.array(doc['camera_matrix']['data']).reshape(3, 3)
            D = doc['distortion_coefficients']['data']
            layout[0] &= np.allclose(np.array(doc['rectification_matrix']['data']).reshape(3, 3), np.eye(3))
            layout[0] &= np.allclose(np.array(doc['projection_matrix']['data']).reshape(3, 4), np.c_[K, np.zeros(3)])
            _, Pc, uv_app = _views(q, c, rng)
            if not len(Pc):
                continue
            uv = _project(Pc, K, D, doc['distortion_model'])
            out[c.tag] = float(np.linalg.norm(uv - uv_app, axis=1).max())
        return out
    worst, rows = _run(ctx, 'formats.ros_camera_info', fn)
    return Outcome(worst, rows, ok=layout[0])


def _kalibr_reference(model, P):
    """Pixels of optical-frame points through a Kalibr camera, from the papers' equations, written for the check:
    pinhole; omni (Mei & Rives 2007: the unit sphere, then the plane at ξ); double sphere (Usenko et al. 2018);
    extended unified (Khomutenko et al. 2016); then equidistant (Kannala–Brandt) or radtan distortion."""
    q = [float(a) for a in model['intrinsics']]
    k = [float(a) for a in model['distortion_coeffs']]
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    cam = model['camera_model']
    if cam == 'pinhole':
        fu, fv, cu, cv = q
        mx, my = x / z, y / z
    elif cam == 'omni':
        xi, fu, fv, cu, cv = q
        n = np.sqrt(x * x + y * y + z * z)
        mx, my = (x / n) / (z / n + xi), (y / n) / (z / n + xi)
    elif cam == 'ds':
        xi, al, fu, fv, cu, cv = q
        d1 = np.sqrt(x * x + y * y + z * z)
        d2 = np.sqrt(x * x + y * y + (xi * d1 + z) ** 2)
        den = al * d2 + (1 - al) * (xi * d1 + z)
        mx, my = x / den, y / den
    elif cam == 'eucm':
        al, be, fu, fv, cu, cv = q
        d = np.sqrt(be * (x * x + y * y) + z * z)
        den = al * d + (1 - al) * z
        mx, my = x / den, y / den
    else:
        raise ValueError(f'unknown Kalibr camera model {cam}')
    dist = model['distortion_model']
    if dist == 'equidistant':
        r = np.hypot(mx, my)
        th = np.arctan(r)
        thd = th * (1 + k[0] * th ** 2 + k[1] * th ** 4 + k[2] * th ** 6 + k[3] * th ** 8)
        s = np.where(r > 1e-12, thd / np.where(r > 1e-12, r, 1.0), 1.0)
        mx, my = mx * s, my * s
    elif dist == 'radtan':
        k1, k2, p1, p2 = k[:4]
        r2 = mx * mx + my * my
        rad = 1 + k1 * r2 + k2 * r2 * r2
        mx, my = mx * rad + 2 * p1 * mx * my + p2 * (r2 + 2 * mx * mx), my * rad + p1 * (r2 + 2 * my * my) \
            + 2 * p2 * mx * my
    return np.stack([fu * mx + cu, fv * my + cv], 1)


@check('formats.kalibr', 'formats', 'Kalibr camchain',
       'largest pixel distance between where a point lands through each camera\'s Kalibr model (projected by the '
       'model\'s published equations, written for the check) and where Vantage\'s lens model puts it, for pinhole '
       'models; for a fitted wide-angle model (omni, double sphere, extended unified) the largest distance must not '
       'exceed the fit error the file states for it, and a fit outside the export\'s limits (0.25 px RMS, 1 px max) '
       'must carry its warning; T_cn_cnm1 must carry the previous camera\'s coordinates into this one (to 1 µm)',
       'the Kalibr camera models\' equations (Mei & Rives; Usenko et al.; Khomutenko et al.; Kannala–Brandt) and '
       'Kalibr\'s camchain.yaml conventions', TOL_PX, 'px', refs=('kalibr2013', 'kannala2006'), cost='study')
def _kalibr(ctx):
    import re
    import yaml
    from ..core.analysis import camera_frames
    extra = {'chain_m': 0.0, 'wide': {}, 'ok': True}
    line = re.compile(r'^#\s+cam(\d+) (\S+): (\S+), ([\d.]+) px RMS, ([\d.]+) px max(.*)$')

    def fn(q, d, rng):
        text = open(os.path.join(d, 'kalibr', 'camchain.yaml'), encoding='utf-8').read()
        doc = yaml.safe_load(text)
        stated = {}
        for ln in text.splitlines():
            mm = line.match(ln)
            if mm:
                stated[int(mm.group(1))] = (float(mm.group(4)), float(mm.group(5)), 'WARNING' in mm.group(6))
        cams = q.variant.cameras
        out = {}
        for i, c in enumerate(cams):
            m = doc[f'cam{i}']
            P, Pc_app, uv_app = _views(q, c, rng)
            if not len(P):
                continue
            if i:
                T = np.array(m['T_cn_cnm1'])
                _, W2O_prev = camera_frames(cams[i - 1])
                Xp = (P - cams[i - 1].pos()) @ W2O_prev.T
                extra['chain_m'] = max(extra['chain_m'],
                                       float(np.abs(Xp @ T[:3, :3].T + T[:3, 3] - Pc_app).max()))
            uv = _kalibr_reference(m, Pc_app)
            e = np.linalg.norm(uv - uv_app, axis=1)
            if m['camera_model'] == 'pinhole':
                out[c.tag] = float(e.max())
                continue
            if i not in stated:
                raise ValueError(f'{c.tag}: the camchain states no fit error for its {m["camera_model"]} model')
            rms, mx, warned = stated[i]
            within = rms <= 0.25 and mx <= 1.0
            extra['ok'] &= float(e.max()) <= mx + 0.01 and (within or warned)
            extra['wide'][c.tag] = {'model': m['camera_model'], 'max_px': round(float(e.max()), 4),
                                    'stated_rms_px': rms, 'stated_max_px': mx, 'warned': warned}
        return out
    worst, rows = _run(ctx, 'formats.kalibr', fn)
    return Outcome(worst, dict(rows, chain_error_m=extra['chain_m'], wide_models=extra['wide']),
                   ok=extra['chain_m'] < 1e-6 and extra['ok'],
                   note='A wide-angle Kalibr model is a fit to the lens, not the lens: the check holds it to the error '
                        'the file states for it.')
