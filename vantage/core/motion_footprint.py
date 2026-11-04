"""Where the sensors sample the ground while the vehicle moves: camera footprints and lidar ground rings warped by ego
motion and each sample's time, and the "unobserved wedges" where a spinning lidar's sweep and a camera's exposure
are too far apart in time for late or early fusion of moving objects.

Frames and units. The vehicle frame is ISO 8855 with its origin at the rear-axle centre on the ground (x forward,
y left, z up); points are in metres, angles in degrees, times in milliseconds, speeds in m/s, yaw rates in deg/s
(positive turning left, counter-clockwise seen from above).

Times follow core.timing: milliseconds after the sync pulse. A camera triggered at t_trigger reads its rows top to
bottom over the readout, each row exposed for exposure_ms before it is read, so a ground point seen on image row v
(through the real lens model) is sampled at mid-exposure t = t_trigger + v / H · readout + exposure / 2. A spinning
lidar samples a point when its head faces the point's sensor-frame azimuth (timing.Sweep.time_of). A camera fires
every frame; the frame used is the one nearest the reference time ref_ms (its trigger folded into ref ± half a frame).
A lidar rotation is centred on ref_ms (its times folded into ref ± half a period).

Ego motion. The vehicle drives at constant speed along its own +x with a constant yaw rate: at Δt seconds after the
reference it stands at (v/ω · sin ωΔt, v/ω · (1 − cos ωΔt)) with heading ωΔt in the reference-time vehicle frame
(straight, (vΔt, 0) and heading 0, when ω = 0). A sensor fixed to the vehicle samples, at Δt, the ground point P of its
own frame then; in the reference-time frame that point is R(ωΔt) P + position(Δt). That is where the warped footprint
and rings are drawn. Speed and yaw rate both zero give the static footprint exactly.

Approximations: pitch, roll and suspension motion are left out (the ground stays z = 0); the readout and exposure are
the sync plan's settings for every camera; a camera's image-border polyline beyond the analysis distance is clipped
to it horizontally (its row there is that of the clipped point); a free-running camera (no hardware trigger, or on no
trigger line) is drawn at the trigger the plan would give it and flagged, its real timing is unknown.
"""
from __future__ import annotations

import math

import numpy as np

from . import sensors as SN
from . import timing as TM
from .analysis import camera_frames
from .lens import resolve


# ---------------------------------------------------------------- ego motion
def ego_pose(dt_s, speed_mps: float, yaw_rate_dps: float):
    """(heading rad (K,), x (K,), y (K,)) of the vehicle dt_s seconds after the reference, in the reference-time
    vehicle frame: constant speed along its own +x and constant yaw rate (a circular arc)."""
    dt = np.asarray(dt_s, float)
    w = math.radians(yaw_rate_dps)
    psi = w * dt
    if abs(w) < 1e-12:
        return psi, speed_mps * dt, np.zeros_like(dt)
    return psi, speed_mps / w * np.sin(psi), speed_mps / w * (1.0 - np.cos(psi))


def carry(P, dt_ms, speed_mps: float, yaw_rate_dps: float) -> np.ndarray:
    """Vehicle-frame points P (K, 3) sampled dt_ms (K,) after the reference, in the reference-time vehicle frame."""
    P = np.asarray(P, float).reshape(-1, 3)
    if speed_mps == 0.0 and yaw_rate_dps == 0.0:
        return P.copy()
    psi, x, y = ego_pose(np.asarray(dt_ms, float).reshape(-1) / 1000.0, speed_mps, yaw_rate_dps)
    c, s = np.cos(psi), np.sin(psi)
    out = P.copy()
    out[:, 0] = c * P[:, 0] - s * P[:, 1] + x
    out[:, 1] = s * P[:, 0] + c * P[:, 1] + y
    return out


def _fold(t, ref, cycle):
    """t folded into [ref − cycle/2, ref + cycle/2)."""
    return ref + TM._wrap(np.asarray(t, float) - ref, cycle)


def _settings(project, settings):
    return settings if settings is not None else TM.TimingSettings.from_dict(getattr(project.analysis, 'timing', None))


def _plan(project, st, plan=None):
    return plan if plan is not None else TM.plan(project, settings=st)


def camera_trigger(e: dict, ref_ms: float = 0.0) -> tuple[float, bool]:
    """(trigger ms of the frame nearest ref_ms, free_running) of a sync-plan camera entry: its matched trigger pulse,
    else the trigger the plan would give it (ideal), else the reference itself."""
    t = e.get('trigger_ms')
    free = t is None
    if t is None:
        t = e.get('ideal_ms')
    if t is None:
        t = ref_ms
    frame = e.get('frame_ms') or 0.0
    return (float(_fold(t, ref_ms, frame)) if frame > 0 else float(t)), free


def sample_times(cam, intr, P, trigger_ms: float, st, ref_ms: float = 0.0):
    """(dt ms (K,), image row (K,)) of vehicle-frame ground points P (K, 3) seen by camera cam: the mid-exposure time of
    the row each point is imaged on (through the real lens; rows clipped to the image), after ref_ms."""
    P = np.asarray(P, float).reshape(-1, 3)
    _, W2O = camera_frames(cam)
    opt = (P - cam.pos()) @ W2O.T
    _, v, _, _ = intr.project_dirs(opt)
    v = np.clip(np.nan_to_num(v, nan=intr.height / 2), 0.0, intr.height)
    t = trigger_ms + v / intr.height * st.readout_ms + st.exposure_ms / 2.0
    return t - ref_ms, v


# ---------------------------------------------------------------- camera footprints
def warp_footprints(project, engine, speed_mps: float, yaw_rate_dps: float, result=None, settings=None,
                    ref_ms: float = 0.0, plan=None) -> dict:
    """Each visible camera's ground footprint, static and as sampled while driving. Returns {camera uid: entry}:

    tag, trigger_ms (the frame nearest ref_ms), free_running (no hardware trigger or no trigger line: timing assumed),
    segments (S, 2, 3) and segments_warped (S, 2, 3): the analysis's footprint outline segments (z = 0);
    segments_dt_ms (S, 2): each end's sample time after ref_ms; border / border_warped / border_dt_ms: the image
    border projected onto the ground, lists of polylines (K, 3) and their times; max_shift_m: the largest distance a
    footprint point moves; speed_mps, yaw_rate_dps, ref_ms, readout_ms, exposure_ms."""
    st = _settings(project, settings)
    res = result if result is not None else engine.analyse(project)
    pl = _plan(project, st, plan)
    var = project.variant
    out = {}
    if res is None:
        return out
    by_uid = {c.uid: c for c in var.cameras}
    for ce in res.cams:
        cam = by_uid.get(ce.uid)
        e = pl.cam(ce.uid) if pl is not None else None
        if cam is None or e is None:
            continue
        intr = resolve(project.module_for(cam), project.lens_for(cam))
        trig, free = camera_trigger(e, ref_ms)
        seg = np.asarray(ce.segments, float).reshape(-1, 2, 2)
        seg3 = np.concatenate([seg, np.zeros(seg.shape[:2] + (1,))], axis=2)
        flat = seg3.reshape(-1, 3)
        dt, _ = sample_times(cam, intr, flat, trig, st, ref_ms)
        warped = carry(flat, dt, speed_mps, yaw_rate_dps)
        borders, bw, bt = [], [], []
        for b in ce.border or []:
            b = np.asarray(b, float).reshape(-1, 3).copy()
            b[:, 2] = 0.0
            d2, _ = sample_times(cam, intr, b, trig, st, ref_ms)
            borders.append(b)
            bt.append(d2)
            bw.append(carry(b, d2, speed_mps, yaw_rate_dps))
        shifts = [np.linalg.norm(warped - flat, axis=1)] + [np.linalg.norm(w - b, axis=1) for w, b in zip(bw, borders)]
        shifts = np.concatenate(shifts) if shifts else np.zeros(0)
        out[ce.uid] = {'tag': ce.tag, 'trigger_ms': trig, 'free_running': bool(free or not e.get('triggered')),
                       'segments': seg3, 'segments_warped': warped.reshape(seg3.shape),
                       'segments_dt_ms': dt.reshape(seg3.shape[:2]), 'border': borders, 'border_warped': bw,
                       'border_dt_ms': bt, 'max_shift_m': float(shifts.max()) if len(shifts) else 0.0,
                       'speed_mps': float(speed_mps), 'yaw_rate_dps': float(yaw_rate_dps), 'ref_ms': float(ref_ms),
                       'readout_ms': float(st.readout_ms), 'exposure_ms': float(st.exposure_ms)}
    return out


# ---------------------------------------------------------------- lidar sweeps
def _spinning(project):
    return [s for s in project.variant.sensors if s.visible and s.kind == 'lidar'
            and project.sensor_model_for(s).spinning and project.sensor_model_for(s).rate_hz > 0]


def lidar_sweep(project, engine, speed_mps: float, yaw_rate_dps: float, settings=None, ref_ms: float = 0.0,
                az_step: float = 1.0, far: float | None = None) -> list:
    """Each visible spinning lidar's beam–ground rings, static and as swept while driving. One entry per lidar:

    uid, tag, period_ms, rotation ('cw' / 'ccw'), phase_deg, pos (3,), and rings: [{channel, static (N, 3),
    warped (N, 3), dt_ms (N,)}], dt_ms the time each ring point is sampled after ref_ms (the rotation centred on it).
    far limits the rings' horizontal reach (default the model's range)."""
    out = []
    for s in _spinning(project):
        m = project.sensor_model_for(s)
        sw = TM.Sweep.of(s, m)
        rings = []
        for ch, P in SN.ground_rings(s, m, az_step, far):
            _, az, _ = SN.angles(s, P)
            t = sw.time_of(az)
            dt = _fold(t, ref_ms, sw.period_ms) - ref_ms
            rings.append({'channel': int(ch), 'static': P, 'warped': carry(P, dt, speed_mps, yaw_rate_dps),
                          'dt_ms': dt})
        out.append({'uid': s.uid, 'tag': s.tag, 'period_ms': sw.period_ms, 'rotation': sw.rotation,
                    'phase_deg': sw.phase_deg, 'pos': np.asarray(s.pos(), float), 'rings': rings})
    return out


# ---------------------------------------------------------------- temporal wedges
def _runs(mask):
    """[(start, end)] index runs (inclusive) where mask is True."""
    out, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            out.append((i, j))
            i = j + 1
        else:
            i += 1
    return out


def temporal_wedges(project, engine, threshold_ms: float = 10.0, settings=None, plan=None,
                    step_deg: float = 0.25) -> list:
    """For each visible spinning lidar and each visible camera, the parts of the camera's horizontal field (as the
    lidar sees it: the image's middle row 20 m out, timing's convention) where the lidar's sample time is more than
    threshold_ms from the camera's mid-exposure of the middle row, for the camera frame fused with that sweep (the one
    nearest the head's crossing of the camera's axis). Moving objects there ghost when the lidar and the image are
    fused. One entry per pair:

    lidar_uid, lidar_tag, lidar_pos (3,), camera_uid, camera_tag, t_camera_ms (the fused frame's middle-row mid-exposure,
    after the sync pulse), frame_ms, free_running (no trigger: the lag is unknown, taken as half a frame over the
    whole field), drifts (the frames slip against the sweep: this is one moment), field_deg (left, right) and az_deg / dt_ms (the sampled field, vehicle-frame azimuth
    counter-clockwise from +x, and the signed lidar-minus-camera time), wedges [(az from, az to)] vehicle-frame
    degrees in sweep order, wedge_deg (their total), field_span_deg, max_dt_ms (largest |Δt| over the field)."""
    st = _settings(project, settings)
    pl = _plan(project, st, plan)
    out = []
    spinning = _spinning(project)
    if not spinning or pl is None:
        return out
    by_uid = {c.uid: c for c in project.variant.cameras}
    for s in spinning:
        m = project.sensor_model_for(s)
        sw = TM.Sweep.of(s, m)
        R = SN.rotation(s)
        for e in pl.cams:
            cam = by_uid.get(e['uid'])
            if cam is None:
                continue
            intr = resolve(project.module_for(cam), project.lens_for(cam))
            D, i0 = TM._field_line(cam, intr)
            P = cam.pos() + TM.AXIS_M * D
            _, az, _ = SN.angles(s, P)
            off = np.degrees(np.unwrap(np.radians(az)))
            centre = float(az[i0])
            off = off - off[i0]
            lo, hi = float(off.min()), float(off.max())
            if hi - lo > 360.0:
                lo, hi = -180.0, 180.0
            n = max(2, int(math.ceil((hi - lo) / step_deg)) + 1)
            phi = centre + np.linspace(lo, hi, n)                    # sensor-frame azimuths across the field
            # the frame fused with this sweep: the one whose middle-row mid-exposure is nearest the moment the head
            # crosses the camera's axis point
            free = e.get('trigger_ms') is None or not e.get('triggered')
            t_cross = float(sw.time_of(centre))
            half = st.readout_ms / 2.0 + st.exposure_ms / 2.0
            trig, _ = camera_trigger(e, t_cross - half)
            t_cam = trig + half
            how = e.get('alignment')
            frame = float(e.get('frame_ms') or sw.period_ms)
            dt = TM._wrap(sw.time_of(phi) - t_cam, sw.period_ms)
            if free:                                                 # unknown timing: any lag up to half a frame
                dt = np.full_like(phi, frame / 2.0)
            a = np.radians(phi)
            dv = np.stack([np.cos(a), np.sin(a), np.zeros_like(a)], -1) @ R.T
            az_v = np.degrees(np.arctan2(dv[:, 1], dv[:, 0]))
            order = np.arange(n) if sw.sense > 0 else np.arange(n)[::-1]     # the head's direction of travel
            bad = np.abs(dt) > threshold_ms
            wedges = []
            for i, j in _runs(bad[order]):
                wedges.append((float(az_v[order][i]), float(az_v[order][j])))
            seg = abs(hi - lo) / max(n - 1, 1)
            out.append({'lidar_uid': s.uid, 'lidar_tag': s.tag, 'lidar_pos': np.asarray(s.pos(), float),
                        'camera_uid': e['uid'], 'camera_tag': e['tag'], 't_camera_ms': float(t_cam % sw.period_ms),
                        'frame_ms': frame, 'free_running': bool(free), 'drifts': how == 'drifts',
                        'field_deg': (float(az_v[-1]), float(az_v[0])) if lo < hi else (float(az_v[0]),) * 2,
                        'az_deg': az_v, 'dt_ms': dt, 'wedges': wedges,
                        'wedge_deg': float(bad.sum() * seg), 'field_span_deg': float(hi - lo),
                        'max_dt_ms': float(np.abs(dt).max()) if len(dt) else 0.0,
                        'threshold_ms': float(threshold_ms)})
    return out
