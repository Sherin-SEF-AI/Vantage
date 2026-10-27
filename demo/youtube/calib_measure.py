"""The before/after figures chapter 47 ("A more accurate calibration") shows: 0.1.1's and 0.2.0's joint solve and
lidar–camera solve on the same simulated captures of the sample, measured here and written to
demo/build/youtube/data/calib_measure.json (the chapter reads it and checks every figure it narrates against it).

    python3 demo/youtube/calib_measure.py [--joint-trials 40] [--lidar-trials 300]     (about 8 minutes)

Run it under the memory cap, offscreen, with a throw-away settings folder (see AUTHORING.md). 0.1.1's modules are read
from the git tag v0.1.1 (`git show`, nothing in the work tree or the repository changes) into a temporary folder and
loaded beside 0.2.0's as vantage.core.joint_011 and vantage.core.lidar_calib_011.

Joint solve (the installation check's): the six-camera sample; each trial mounts every camera off its plan by up to
1 deg and 10 mm, lays each floor board off its place (20 mm, 1 deg, the placement prior), adds 0.5 px of corner noise
to every board view a camera sees whole, starts each camera from install.solve_pose and runs joint_solve; an
independent statement of the same least squares, solved by Gauss-Newton to convergence, is the reference optimum.
Reported: the solves that stopped short (more than 0.5 of chi-squared left above the optimum), the scatter over the
uncertainty the optimum predicts (per camera axis), the time per solve.

Lidar–camera (the data rig's F and L1, the page's own pose plan): each trial adds 0.3 px corner noise and the lidar's
range noise along each beam, and solves with the board planes only (edges and checker off, as the planes-only
prediction assumes). Reported: predicted_sigma against the scatter, the mean error (bias) in standard errors, the time
per solve. (After the reproduction scripts of the release's calibration fix.)
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(os.path.dirname(HERE))]
import common  # noqa: E402

import numpy as np  # noqa: E402

OUT = os.path.join(common.BUILD, 'data', 'calib_measure.json')
OLD_TAG = 'v0.1.1'


def load_old(tmp):
    """0.1.1's joint.py and lidar_calib.py from the git tag, as vantage.core.<name>_011."""
    import vantage.core  # noqa: F401
    mods = {}
    for name in ('joint', 'lidar_calib'):
        src = subprocess.run(['git', 'show', f'{OLD_TAG}:vantage/core/{name}.py'], cwd=common.ROOT, check=True,
                             capture_output=True).stdout
        path = os.path.join(tmp, f'{name}_011.py')
        with open(path, 'wb') as fh:
            fh.write(src)
        spec = importlib.util.spec_from_file_location(f'vantage.core.{name}_011', path)
        m = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = m
        spec.loader.exec_module(m)
        mods[name] = m
    return mods


# ====================================================================== the joint problem, stated independently
def _expm(w):
    th = float(np.linalg.norm(w))
    if th < 1e-15:
        return np.eye(3)
    k = w / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K


def _logm(R):
    c = max(-1.0, min(1.0, (np.trace(R) - 1) / 2))
    th = math.acos(c)
    if th < 1e-12:
        return np.zeros(3)
    return th / (2 * math.sin(th)) * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])


class Rig:
    """The sample's six cameras and eight floor boards; the board views each camera sees whole (in the image, in
    front of it, nothing of the vehicle in the way, squares of 8 px or more)."""

    def __init__(self, min_px=8.0):
        from vantage.core import samples
        from vantage.core.analysis import Engine
        from vantage.core.install import board_outline, inner_corners
        from vantage.core.lens import resolve
        from vantage.core.mathutil import BODY_TO_OPTICAL, rot_zyx
        self.p = samples.tigor_sample_project()
        e = Engine()
        vg = e.vehicle_geom(self.p)
        self.cams = list(self.p.variant.cameras)
        self.targets = list(self.p.targets)
        self.intr = [resolve(self.p.module_for(c), self.p.lens_for(c)) for c in self.cams]
        self.Rp = [rot_zyx(c.yaw, c.pitch, c.roll) for c in self.cams]
        self.Pp = [c.pos().astype(float) for c in self.cams]
        self.W = [inner_corners(t).reshape(-1, 3) for t in self.targets]
        self.ctr = [np.array([t.x, t.y, 0.0]) for t in self.targets]
        self.B2O = BODY_TO_OPTICAL
        self.seen = []
        for ci in range(len(self.cams)):
            cube = e.cube(self.Pp[ci], vg)
            for bi, t in enumerate(self.targets):
                P = np.vstack([self.W[bi], board_outline(t)])
                d = (P - self.Pp[ci]) @ (BODY_TO_OPTICAL @ self.Rp[ci].T).T
                u, v, _, ok = self.intr[ci].project_dirs(d)
                if not ok.all() or (d[:, 2] / np.linalg.norm(d, axis=1) < 0.05).any():
                    continue
                dd = P - self.Pp[ci]
                dist = np.linalg.norm(dd, axis=1)
                if not (cube.lookup(dd)[0] > dist - (0.02 + 0.01 * dist)).all():
                    continue
                G = np.stack([u, v], -1)[:len(self.W[bi])].reshape(t.rows - 1, t.cols - 1, 2)
                sq = min(np.linalg.norm(np.diff(G, axis=0), axis=-1).min(),
                         np.linalg.norm(np.diff(G, axis=1), axis=-1).min())
                if sq >= min_px:
                    self.seen.append((ci, bi))

    def project(self, ci, R, pos, P):
        d = (P - pos) @ (self.B2O @ R.T).T
        u, v, _, _ = self.intr[ci].project_dirs(d)
        return np.stack([u, v], -1)

    def board_world(self, bi, b):
        from vantage.core.mathutil import rot_z
        dx, dy, dyaw = b
        return (self.W[bi] - self.ctr[bi]) @ rot_z(dyaw).T + self.ctr[bi] + np.array([dx, dy, 0.0])

    def trial(self, rng, sb_mm=20.0, sb_deg=1.0, px=0.5, mount_deg=1.0, mount_mm=10.0):
        from vantage.core.install import BoardObs
        from vantage.core.mathutil import rot_zyx
        nc, nb = len(self.cams), len(self.targets)
        tR, tP = [], []
        for ci in range(nc):
            c = self.cams[ci]
            d = rng.uniform(-mount_deg, mount_deg, 3)
            tR.append(rot_zyx(c.yaw + d[0], c.pitch + d[1], c.roll + d[2]))
            tP.append(self.Pp[ci] + rng.uniform(-mount_mm, mount_mm, 3) / 1000)
        boards = np.c_[rng.normal(0, sb_mm / 1000, nb), rng.normal(0, sb_mm / 1000, nb),
                       rng.normal(0, math.radians(sb_deg), nb)]
        obs, data = {}, []
        for ci, bi in self.seen:
            uv = self.project(ci, tR[ci], tP[ci], self.board_world(bi, boards[bi])) + rng.normal(0, px, (len(self.W[bi]), 2))
            rays, _ = self.intr[ci].unproject(uv[:, 0], uv[:, 1])
            t = self.targets[bi]
            obs.setdefault(self.cams[ci].uid, []).append(BoardObs(t.uid, t.name, self.W[bi], rays, uv, uv.copy(),
                                                                  np.zeros((4, 2))))
            data.append((ci, bi, uv))
        return tR, tP, boards, obs, data

    def shapes(self):
        return {c.uid: (it.height, it.width) for c, it in zip(self.cams, self.intr)}

    def init(self, obs):
        from vantage.core.install import solve_pose
        return {c.uid: solve_pose(self.p, c, obs[c.uid], (self.intr[ci].height, self.intr[ci].width))[:2]
                for ci, c in enumerate(self.cams) if c.uid in obs}


class Ref:
    """The joint least squares in its own parameters (camera i at R_i exp(w_i), p_i + t_i; board b moved by dx, dy,
    dyaw), corners weighted 1/px and the boards' prior 1/sigma, solved by Gauss-Newton to convergence."""

    def __init__(self, rig, data, base_R, base_P, sb_mm=20.0, sb_deg=1.0, px=0.5):
        self.rig, self.data, self.bR, self.bP = rig, data, base_R, base_P
        self.nc, self.nb = len(rig.cams), len(rig.targets)
        self.sb, self.sy, self.px = sb_mm / 1000, math.radians(sb_deg), px

    def resid(self, x):
        cp, bp = x[:6 * self.nc].reshape(self.nc, 6), x[6 * self.nc:].reshape(self.nb, 3)
        out = [((self.rig.project(ci, self.bR[ci] @ _expm(cp[ci, :3]), self.bP[ci] + cp[ci, 3:],
                                  self.rig.board_world(bi, bp[bi])) - uv) / self.px).ravel() for ci, bi, uv in self.data]
        out.append((bp / np.array([self.sb, self.sb, self.sy])).ravel())
        return np.concatenate(out)

    def jac(self, x, h=1e-7):
        J = np.zeros((len(self.resid(x)), len(x)))
        for k in range(len(x)):
            d = np.zeros(len(x))
            d[k] = h
            J[:, k] = (self.resid(x + d) - self.resid(x - d)) / (2 * h)
        return J

    def x_of(self, poses, boards):
        x = np.zeros(6 * self.nc + 3 * self.nb)
        for ci in range(self.nc):
            R, pos = poses[ci]
            x[6 * ci:6 * ci + 3] = _logm(self.bR[ci].T @ np.asarray(R, float))
            x[6 * ci + 3:6 * ci + 6] = np.asarray(pos, float) - self.bP[ci]
        x[6 * self.nc:] = np.asarray(boards, float).ravel()
        return x

    def cost(self, x):
        r = self.resid(x)
        return float(r @ r)

    def solve(self, x0):
        x = np.array(x0, float)
        c = self.cost(x)
        for _ in range(30):
            step = np.linalg.lstsq(self.jac(x), -self.resid(x), rcond=None)[0]
            lam = 1.0
            while lam > 1e-4:
                cn = self.cost(x + lam * step)
                if cn <= c:
                    break
                lam /= 2
            else:
                break
            x, c = x + lam * step, cn
            if np.max(np.abs(lam * step)) < 1e-12:
                break
        return x, c


def joint(mods, trials, seed=1):
    rig = Rig()
    nc = len(rig.cams)
    out = {}
    for label, J in mods.items():
        rng = np.random.default_rng(seed)
        left, times, errs, covs = [], [], [], []
        for _ in range(trials):
            tR, tP, boards, obs, data = rig.trial(rng)
            init = rig.init(obs)
            t0 = time.perf_counter()
            jr = J.joint_solve(rig.p, rig.cams, obs, init, rig.shapes())
            times.append(time.perf_counter() - t0)
            ref = Ref(rig, data, tR, tP)
            x_true = ref.x_of({ci: (tR[ci], tP[ci]) for ci in range(nc)}, boards)
            xr, cr = ref.solve(x_true)
            byuid = {b.uid: b for b in jr.boards}
            bj = np.array([[byuid[t.uid].dx_mm / 1000, byuid[t.uid].dy_mm / 1000, math.radians(byuid[t.uid].dyaw_deg)]
                           if t.uid in byuid else [0.0, 0.0, 0.0] for t in rig.targets])
            xj = ref.x_of({ci: jr.poses[c.uid] for ci, c in enumerate(rig.cams)}, bj)
            left.append(ref.cost(xj) - cr)
            Jm = ref.jac(xr)
            covs.append(np.linalg.inv(Jm.T @ Jm))
            errs.append(xj - x_true)
        psd = np.sqrt(np.mean([np.diag(C) for C in covs], axis=0))[:6 * nc]
        ratio = np.sqrt(np.mean(np.array(errs) ** 2, axis=0))[:6 * nc] / psd
        left = np.array(left)
        out[label] = {'trials': trials, 'stalled': int((left > 0.5).sum()), 'median_left': float(np.median(left)),
                      'ratio_min': float(ratio.min()), 'ratio_max': float(ratio.max()),
                      'seconds_per_solve': float(np.mean(times))}
        print(label, 'joint', out[label], flush=True)
    return out


def lidar(mods, trials, seed=1):
    from vantage.core import samples, sensors as S
    from vantage.core.analysis import Engine
    from vantage.core.lens import resolve
    from vantage.core.mathutil import euler_zyx, wrap180
    out = {}
    for label, LC in mods.items():
        p = samples.tigor_sample_project()
        p.active_uid = next(v for v in p.variants if v.sensors).uid
        cam = p.variant.by_tag('F')
        lid = next(s for s in p.variant.sensors if s.tag == 'L1')
        st = LC.LidarCalibSettings.from_dict({'camera': 'F', 'lidar': 'L1'})
        poses, _ = LC.plan_poses(p, cam, lid, st, Engine())
        m = p.sensor_model_for(lid)
        intr = resolve(p.module_for(cam), p.lens_for(cam))
        W2O, C = LC._w2o(cam), cam.pos().astype(float)
        T0 = LC.plan_T(cam, lid)
        R0, t0 = T0[:3, :3], T0[:3, 3]
        model = LC.model_corners(st)
        Wb, Hb = st.board_size()
        Ds = S.directions(m.elevations(), S.scan_azimuths(m)).reshape(-1, 3)
        Dopt = Ds @ R0.T
        sr = st.range_sigma_mm / 1000.0
        geo = []
        for bp in poses:
            Rb, cb = W2O @ bp.R, W2O @ (bp.centre - C)
            nrm = Rb[:, 2]
            den = Dopt @ nrm
            with np.errstate(divide='ignore', invalid='ignore'):
                s = np.where(np.abs(den) > 1e-9, float(nrm @ (cb - t0)) / den, -1.0)
            X = t0 + Dopt * s[:, None]
            on = ((s >= m.range_min_m) & (s <= m.range_m) & (np.abs((X - cb) @ Rb[:, 0]) <= Wb / 2)
                  & (np.abs((X - cb) @ Rb[:, 1]) <= Hb / 2))
            cosi = np.abs(Dopt[on] @ nrm)
            geo.append((Rb, cb, np.flatnonzero(on), s[on], max(sr * float(np.sqrt(np.mean(cosi ** 2))), 0.002)))
        pred = LC.predicted_sigma(p, cam, lid, poses, st)
        Rl0, pl0 = LC.lidar_pose(cam, T0)
        e0 = euler_zyx(Rl0)
        rng = np.random.default_rng(seed)
        errs, times = [], []
        for trial in range(trials):
            obs = []
            for k, (Rb, cb, beams, s, rms) in enumerate(geo):
                u, v, _, _ = intr.project_dirs(model @ Rb.T + cb)
                uv = np.stack([u, v], -1) + rng.normal(0, 0.3, (len(model), 2))
                rays, _ = intr.unproject(uv[:, 0], uv[:, 1])
                Rs, cs, rpx, cov = LC._board_pnp(intr, 1.0, 1.0, model, rays, uv)
                pts = Ds[beams] * (s + rng.normal(0, sr, len(beams)))[:, None]
                nl = R0.T @ Rb[:, 2]
                obs.append(LC.PoseObs(f'pose{k + 1:02d}', cam=LC.CamBoard(uv, rays, Rs, cs, rpx, cov),
                                      lid=LC.LidarBoard(pts, nl, float(nl @ pts.mean(0)), rms, np.zeros((0, 3)),
                                                        np.zeros((0, 3)), 8)))
            Ts = LC._T(LC._exp(rng.normal(0, math.radians(0.3), 3)) @ R0, t0 + rng.normal(0, 0.01, 3))
            t1 = time.perf_counter()
            sol = LC._solve(intr, model, Wb, Hb, obs, Ts, use_edges=False, seed=trial, st=st, use_checker=False)
            times.append(time.perf_counter() - t1)
            Rl, pl = LC.lidar_pose(cam, sol['T'])
            e = euler_zyx(Rl)
            errs.append([(pl[0] - pl0[0]) * 1e3, (pl[1] - pl0[1]) * 1e3, (pl[2] - pl0[2]) * 1e3,
                         wrap180(e[2] - e0[2]), e[1] - e0[1], wrap180(e[0] - e0[0])])
        E = np.array(errs)
        sd, mean = E.std(0, ddof=1), E.mean(0)
        keys = ['x_mm', 'y_mm', 'z_mm', 'roll_deg', 'pitch_deg', 'yaw_deg']
        out[label] = {'trials': trials, 'seconds_per_solve': float(np.mean(times)),
                      'axes': {k: {'predicted': float(pred[k]), 'scatter': float(sd[i]), 'ratio': float(sd[i] / pred[k]),
                                   'bias': float(mean[i]), 'bias_se': float(mean[i] / sd[i] * math.sqrt(len(E)))}
                               for i, k in enumerate(keys)}}
        print(label, 'lidar', json.dumps(out[label]), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--joint-trials', type=int, default=40)
    ap.add_argument('--lidar-trials', type=int, default=300)
    a = ap.parse_args()
    from PyQt6.QtWidgets import QApplication
    app = QApplication(sys.argv[:1])  # noqa: F841  (the engine's cube maps need Qt)
    import vantage.core.joint as J
    import vantage.core.lidar_calib as LC
    with tempfile.TemporaryDirectory() as tmp:
        old = load_old(tmp)
        t0 = time.time()
        res = {'old': OLD_TAG, 'joint': joint({'0.1.1': old['joint'], '0.2.0': J}, a.joint_trials),
               'lidar': lidar({'0.1.1': old['lidar_calib'], '0.2.0': LC}, a.lidar_trials),
               'measured': time.strftime('%Y-%m-%d %H:%M'), 'seconds': round(time.time() - t0)}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w', encoding='utf-8') as fh:
        json.dump(res, fh, indent=1)
    print(OUT)


if __name__ == '__main__':
    main()
