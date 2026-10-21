"""Reference photo sets: loading frames (folder or .zip) and solving camera poses.

A frame pose is world->camera (OpenCV optical: x right, y down, z forward) in the vehicle
frame (ISO 8855, rear-axle origin). Intrinsics are shared by all frames of a set.
"""
from __future__ import annotations

import math
import os
import time
import zipfile

import numpy as np

from .project import PhotoSet, PhotoFrame

IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff')
MAX_IMAGE_BYTES = 80 * 1024 * 1024


def list_frames(source: str) -> list[str]:
    """Image names in a folder or zip, in natural order."""
    import re

    def natural(s):
        return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

    if os.path.isdir(source):
        names = [n for n in os.listdir(source) if n.lower().endswith(IMAGE_EXTS)]
    elif zipfile.is_zipfile(source):
        with zipfile.ZipFile(source) as z:
            names = [i.filename for i in z.infolist() if i.filename.lower().endswith(IMAGE_EXTS)
                     and not i.filename.startswith('__MACOSX') and i.file_size < MAX_IMAGE_BYTES]
    else:
        raise ValueError('Choose a folder of images or a .zip of images.')
    return sorted(names, key=natural)


def read_frame_bytes(source: str, name: str) -> bytes | None:
    try:
        if os.path.isdir(source):
            path = os.path.realpath(os.path.join(source, name))
            if not path.startswith(os.path.realpath(source) + os.sep):
                return None
            if os.path.getsize(path) > MAX_IMAGE_BYTES:
                return None
            with open(path, 'rb') as fh:
                return fh.read()
        with zipfile.ZipFile(source) as z:
            info = z.getinfo(name)
            if info.file_size > MAX_IMAGE_BYTES:
                return None
            return z.read(info)
    except (OSError, KeyError, zipfile.BadZipFile):
        return None


def frame_pose(ps: PhotoSet, k: int):
    fr = ps.frames[k]
    if fr.pose is None:
        return None
    M = np.asarray(fr.pose, float)
    return M[:3, :3], M[:3, 3]


def project_points(ps: PhotoSet, k: int, P):
    """World points (N,3) -> pixel (N,2) for frame k, with NaN behind the camera."""
    pose = frame_pose(ps, k)
    P = np.atleast_2d(np.asarray(P, float))
    if pose is None:
        return np.full((len(P), 2), np.nan)
    R, t = pose
    c = P @ R.T + t
    z = c[:, 2]
    x = c[:, 0] / np.where(np.abs(z) < 1e-9, 1e-9, z)
    y = c[:, 1] / np.where(np.abs(z) < 1e-9, 1e-9, z)
    r2 = x * x + y * y
    s = 1 + ps.k1 * r2
    uv = np.stack([ps.cx + ps.f_px * x * s, ps.cy + ps.f_px * y * s], axis=1)
    uv[z < 0.05] = np.nan
    return uv


def camera_centre(ps: PhotoSet, k: int):
    pose = frame_pose(ps, k)
    if pose is None:
        return None
    R, t = pose
    return -R.T @ t


def refresh_residuals(ps: PhotoSet):
    errs_all = []
    for k, fr in enumerate(ps.frames):
        errs = []
        for name, u, v in fr.points:
            if name not in ps.landmarks or fr.pose is None:
                continue
            q = project_points(ps, k, [ps.landmarks[name]])[0]
            if np.all(np.isfinite(q)):
                errs.append(math.hypot(q[0] - u, q[1] - v))
        fr.rms = float(math.sqrt(np.mean(np.square(errs)))) if errs else 0.0
        errs_all += errs
    ps.rms = float(math.sqrt(np.mean(np.square(errs_all)))) if errs_all else 0.0


def point_errors(ps: PhotoSet, k: int):
    out = []
    fr = ps.frames[k]
    for name, u, v in fr.points:
        if name in ps.landmarks and fr.pose is not None:
            q = project_points(ps, k, [ps.landmarks[name]])[0]
            out.append((name, u, v, q, math.hypot(q[0] - u, q[1] - v) if np.all(np.isfinite(q)) else float('nan')))
        else:
            out.append((name, u, v, None, float('nan')))
    return out


def calibrate(ps: PhotoSet, image_size=None) -> str:
    """Solve shared intrinsics and per-frame poses from landmark correspondences.

    Frames with >= 6 points join a joint OpenCV calibration (focal length, principal point fixed
    at the centre, k1); frames with 4-5 points are solved with PnP afterwards. Returns a summary."""
    import cv2

    if image_size:
        ps.width, ps.height = image_size
    w, h = ps.width or 1920, ps.height or 1080
    obj, img, idx = [], [], []
    for k, fr in enumerate(ps.frames):
        pts = [(ps.landmarks[n], u, v) for n, u, v in fr.points if n in ps.landmarks]
        if len(pts) >= 6:
            obj.append(np.array([p for p, _, _ in pts], np.float32))
            img.append(np.array([[u, v] for _, u, v in pts], np.float32))
            idx.append(k)
    f0 = ps.f_px or 1.2 * max(w, h)
    K = np.array([[f0, 0, w / 2], [0, f0, h / 2], [0, 0, 1]], np.float64)
    dist = np.zeros(5)
    solved = 0
    if len(obj) >= 2:
        flags = (cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_FIX_ASPECT_RATIO |
                 cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3)
        rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj, img, (w, h), K, dist, flags=flags)
        for k, rv, tv in zip(idx, rvecs, tvecs):
            R, _ = cv2.Rodrigues(rv)
            M = np.eye(4)
            M[:3, :3], M[:3, 3] = R, tv.ravel()
            ps.frames[k].pose = M.tolist()
            solved += 1
    if len(obj) >= 2:
        ps.f_px, ps.cx, ps.cy, ps.k1 = float(K[0, 0]), float(K[0, 2]), float(K[1, 2]), float(dist.ravel()[0])
    Kp = np.array([[ps.f_px, 0, ps.cx], [0, ps.f_px, ps.cy], [0, 0, 1]])
    dp = np.array([ps.k1, 0, 0, 0, 0], float)
    for k, fr in enumerate(ps.frames):
        if k in idx:
            continue
        pts = [(ps.landmarks[n], u, v) for n, u, v in fr.points if n in ps.landmarks]
        if len(pts) >= 4:
            o = np.array([p for p, _, _ in pts], np.float64)
            i = np.array([[u, v] for _, u, v in pts], np.float64)
            ok, rv, tv = cv2.solvePnP(o, i, Kp, dp, flags=cv2.SOLVEPNP_SQPNP if len(pts) >= 3 else 0)
            if ok:
                R, _ = cv2.Rodrigues(rv)
                M = np.eye(4)
                M[:3, :3], M[:3, 3] = R, tv.ravel()
                fr.pose = M.tolist()
                solved += 1
    refresh_residuals(ps)
    if solved == 0:
        return 'Not enough correspondences: each frame needs 4 or more landmarks (6 for the joint fit). Existing poses were kept.'
    ps.method = 'OpenCV calibration from landmarks'
    ps.solved_at = time.strftime('%H:%M')
    return f'Solved {solved} of {len(ps.frames)} frames · {ps.rms:.2f} px RMS'


def worst_frame(ps: PhotoSet):
    best = (-1, 0.0)
    for k, fr in enumerate(ps.frames):
        if fr.rms > best[1]:
            best = (k, fr.rms)
    return best


def photo_focal_35mm(ps: PhotoSet):
    if not ps.width:
        return 0.0
    return ps.f_px / ps.width * 36.0
