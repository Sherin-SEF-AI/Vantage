"""Writes a simulated drive in exactly the layout core.dataset writes: <out>/nuscenes/ (version v1.0-vantage: the 13
tables, samples/<CHANNEL>/ keyframes, sweeps/<CHANNEL>/ the rest, maps/<token>.png), <out>/rosbag/<name>/ (a rosbag2
MCAP bag, core.rosbag's writer), <out>/dataset.json (schema vantage.dataset/1, so the drive-log check judges the
cameras at the size and rate written) and README.md. Pure numpy and the standard library (plus core.rosbag and, for
the JPEG and PNG files, OpenCV): it runs in CARLA's Python environment.

Tokens are the dataset's (md5 of the parts joined by '|'); the calibrated_sensor and sensor rows come from the rig
file (core.simrig: the nuscenes_calib export's, intrinsics scaled to the images). Keyframes (nuScenes samples) are
every n-th sweep of the first lidar (n = its rate / keyframe_hz), else every n-th frame of the first camera; each
channel's keyframe is its planned sample_data nearest in time, as core.dataset assigns them.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import io
import json
import os
import shutil

import numpy as np

from ..core import groundtruth as GT
from ..core import rosbag as RB
from . import frames as FR

VERSION = 'v1.0-vantage'
MARKER = '.vantage-dataset'
DATASET_SCHEMA = 'vantage.dataset/1'
MAP_RES = 0.1
TF_HZ = 50.0
LIDAR_PC2 = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('intensity', '<f4'), ('ring', '<u2'), ('t', '<f4')])
RADAR_PC2 = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('v_r', '<f4'), ('rcs', '<f4')])
RADAR_DTYPE = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('dyn_prop', 'i1'), ('id', '<i2'), ('rcs', '<f4'),
                        ('vx', '<f4'), ('vy', '<f4'), ('vx_comp', '<f4'), ('vy_comp', '<f4'),
                        ('is_quality_valid', 'i1'), ('ambig_state', 'i1'), ('x_rms', 'i1'), ('y_rms', 'i1'),
                        ('invalid_state', 'i1'), ('pdh0', 'i1'), ('vx_rms', 'i1'), ('vy_rms', 'i1')])
PCD_HEAD = ('# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\n'
            'FIELDS x y z dyn_prop id rcs vx vy vx_comp vy_comp is_quality_valid ambig_state x_rms y_rms '
            'invalid_state pdh0 vx_rms vy_rms\nSIZE 4 4 4 1 2 4 4 4 4 4 1 1 1 1 1 1 1 1\n'
            'TYPE F F F I I F F F F F I I I I I I I I\nCOUNT 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1\n'
            'WIDTH {n}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {n}\nDATA binary\n')
MARKER_RGBA = {'vehicle.car': (0.20, 0.55, 0.95, 0.45), 'vehicle.truck': (0.55, 0.35, 0.95, 0.45),
               'human.pedestrian.adult': (0.95, 0.35, 0.20, 0.55), 'vehicle.bicycle': (0.95, 0.75, 0.10, 0.55)}


def token(*parts) -> str:
    """A stable 32-hex token like nuScenes', the same as core.exporters._token."""
    return hashlib.md5('|'.join(str(x) for x in parts).encode('utf-8')).hexdigest()


def _r(v, n=9):
    x = round(float(v), n) + 0.0
    return 0.0 if x == 0 else x


def check_out_dir(out_dir, bag_name, nuscenes=True, rosbag=True):
    """(nuscenes folder, bag folder); ValueError when either holds files Vantage did not write."""
    nus = os.path.join(out_dir, 'nuscenes')
    bag = os.path.join(out_dir, 'rosbag', bag_name)
    if os.path.exists(out_dir) and not os.path.isdir(out_dir):
        raise ValueError(f'{out_dir} is a file: choose a folder.')
    if nuscenes and os.path.isdir(nus) and os.listdir(nus) and not os.path.exists(os.path.join(nus, MARKER)):
        raise ValueError(f'{nus} already holds files Vantage did not write: choose an empty folder.')
    if rosbag and os.path.isdir(bag):
        if [f for f in os.listdir(bag) if not (f == 'metadata.yaml' or f.endswith('.mcap'))]:
            raise ValueError(f'{bag} holds other files: choose an empty folder.')
    return nus, bag


class Channel:
    """One recorded stream: its planned stamps (s), tokens and keyframe assignment."""

    def __init__(self, kind, doc, channel, stamps, rate):
        self.kind, self.doc, self.channel = kind, doc, channel
        self.tag = doc['safe_tag']
        self.topic = doc['topic']
        self.stamps = np.asarray(stamps, float)
        self.rate = rate
        self.keys = {}
        self.tokens = []
        self.written = {}              # planned index -> (timestamp us, filename, is_key, width, height, ego token)
        self.frames = 0
        self.bytes = 0
        # injected faults (carla_bridge FAULT_KINDS): planned indices never written, a clock offset, a stuck window
        self.dropped = set()
        self.offset_s = 0.0
        self.freeze = None             # (from_s, to_s)
        self.range_bias = 0.0

    def frozen(self, k: int) -> bool:
        return self.freeze is not None and self.freeze[0] - 1e-9 <= float(self.stamps[k]) < self.freeze[1] - 1e-9


class Recorder:
    """Writes one drive. events(): planned stamps per channel are given up front (the simulation's schedule)."""

    def __init__(self, rig: dict, out_dir: str, plan: dict, run_key: str, extra: dict, nuscenes=True, rosbag=True,
                 gt=None):
        self.rig = rig
        self.gt = gt or {}
        self.gt_frames = {}                           # camera tag -> [frame entry] (ground_truth/index.json)
        self.gt_sweeps = {}                           # lidar tag -> [sweep entry]
        self.coco_items = {}                          # camera tag -> [(frame entry, [RLE per object])]
        self.lidarseg_rows, self.panoptic_rows = [], []
        self.trajectory_t, self.trajectory_T = [], []   # every step's ego pose (clock-offset stamps' ego poses)
        self.out_dir = os.path.abspath(out_dir)
        out = rig['output']
        self.epoch_us = int(out['epoch_us'])
        self.bag_name = out['bag_name']
        self.nus, self.bag_dir = check_out_dir(self.out_dir, self.bag_name, nuscenes, rosbag)
        self.do_nus, self.do_bag = nuscenes, rosbag
        self.run_key = run_key
        self.extra = extra
        self.notes = []
        self.created = []
        self.chans = plan['chans']                    # [Channel]
        self.samples = np.asarray(plan['samples'], float)
        self.keyframe_hz = plan['keyframe_hz']
        self.sample_tok = [token('vantage', 'sample', run_key, self.us(t)) for t in self.samples]
        self.sample_us = [self.us(t) for t in self.samples]
        for c in self.chans:
            if c.channel:
                c.tokens = [token('vantage', 'sample_data', run_key, c.channel, self.us(t)) for t in c.stamps]
        cal_by_sensor = {r['sensor_token']: r['token'] for r in extra['cal_rows']}
        self.cs_token = {r['channel']: cal_by_sensor.get(r['token'], '') for r in extra['sensor_rows']}
        self.sd_rows, self.ego_rows = [], []
        self.nbytes = {'nuscenes': 0, 'rosbag': 0}
        self.logname = 'vantage-' + _dt.datetime.fromtimestamp(self.epoch_us / 1e6, _dt.timezone.utc).strftime(
            '%Y-%m-%d-%H-%M-%S') + '+0000'
        self.bag = None
        self._prepare()

    # ------------------------------------------------------------------ time
    def us(self, t: float) -> int:
        return self.epoch_us + int(round(t * 1e6))

    def ns(self, t: float) -> int:
        return self.epoch_us * 1000 + int(round(t * 1e9))

    # ------------------------------------------------------------------ folders
    def _prepare(self):
        os.makedirs(self.out_dir, exist_ok=True)
        gt_dir = os.path.join(self.out_dir, 'ground_truth')
        if os.path.isdir(gt_dir) and os.path.isfile(os.path.join(gt_dir, 'index.json')):
            shutil.rmtree(gt_dir, ignore_errors=True)   # (only one Vantage wrote: it holds its index)
        if self.gt.get('enabled'):
            os.makedirs(gt_dir, exist_ok=True)
            self.created.append(gt_dir)
        if self.do_nus:
            if os.path.isdir(self.nus):
                for sub in ('samples', 'sweeps', 'maps', VERSION, GT.LIDARSEG_DIR, GT.PANOPTIC_DIR):
                    shutil.rmtree(os.path.join(self.nus, sub), ignore_errors=True)
            os.makedirs(os.path.join(self.nus, VERSION), exist_ok=True)
            with open(os.path.join(self.nus, MARKER), 'w') as fh:
                fh.write('Written by Vantage (vantage.sim, a CARLA drive); a new run clears samples/, sweeps/, maps/ '
                         f'and {VERSION}/ here.\n')
            self.created.append(self.nus)
        if self.do_bag:
            if os.path.isdir(self.bag_dir):
                shutil.rmtree(self.bag_dir)
            self.created.append(self.bag_dir)
            self.bag = RB.BagWriter(self.bag_dir, overwrite=True)
            b = self.bag
            b.add_topic('/tf_static', 'tf2_msgs/msg/TFMessage')
            b.add_topic('/tf', 'tf2_msgs/msg/TFMessage')
            b.add_topic('/odom', 'nav_msgs/msg/Odometry')
            b.add_topic('/ground_truth/markers', 'visualization_msgs/msg/MarkerArray')
            b.add_topic('/ground_truth/boxes', 'std_msgs/msg/String')
            for c in self.chans:
                if c.kind == 'camera':
                    b.add_topic(c.topic + '/image_raw/compressed', 'sensor_msgs/msg/CompressedImage')
                    b.add_topic(c.topic + '/camera_info', 'sensor_msgs/msg/CameraInfo')
                elif c.kind == 'lidar':
                    b.add_topic(c.topic + '/points', 'sensor_msgs/msg/PointCloud2')
                elif c.kind == 'radar':
                    b.add_topic(c.topic + '/detections', 'sensor_msgs/msg/PointCloud2')
                elif c.kind == 'imu':
                    b.add_topic(c.topic + '/imu', 'sensor_msgs/msg/Imu')
                elif c.kind == 'gnss':
                    b.add_topic(c.topic + '/fix', 'sensor_msgs/msg/NavSatFix')
            t0 = self.ns(0.0)
            b.write('/tf_static', t0, {'transforms': [RB.transform_stamped(np.array(x['matrix'], float), t0,
                                                                            x['parent'], x['child'])
                                                       for x in self.rig['tf_static']]})

    def cleanup(self):
        """Remove what this run wrote (after a cancel or a failure)."""
        if self.bag is not None and not self.bag.closed:
            try:
                self.bag.close()
            except Exception:
                pass
        for path in self.created:
            if os.path.basename(path) == 'nuscenes':
                for sub in ('samples', 'sweeps', 'maps', VERSION, MARKER, GT.LIDARSEG_DIR, GT.PANOPTIC_DIR):
                    q = os.path.join(path, sub)
                    if os.path.isdir(q):
                        shutil.rmtree(q, ignore_errors=True)
                    elif os.path.exists(q):
                        os.remove(q)
                try:
                    os.rmdir(path)
                except OSError:
                    pass
            elif os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
        try:
            os.rmdir(os.path.join(self.out_dir, 'rosbag'))
        except OSError:
            pass

    def _file(self, rel, data):
        path = os.path.join(self.nus, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as fh:
            fh.write(data)
        self.nbytes['nuscenes'] += len(data)

    def _out_file(self, rel, data):
        """A file under the output folder (the ground truth), counted apart."""
        path = os.path.join(self.out_dir, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as fh:
            fh.write(data)
        self.nbytes['ground_truth'] = self.nbytes.get('ground_truth', 0) + len(data)

    # ------------------------------------------------------------------ the ego's trajectory
    def pose(self, t: float, T_ego):
        """The ego pose (global <- ego) at every step, for the ego poses of clock-offset stamps."""
        self.trajectory_t.append(float(t))
        self.trajectory_T.append(np.asarray(T_ego, float).copy())

    def pose_at(self, t: float) -> np.ndarray:
        """The ego pose at t, interpolated between the steps around it (position linearly, rotation by normalised
        quaternion interpolation); the end poses beyond the trajectory."""
        ts = self.trajectory_t
        if not ts:
            return np.eye(4)
        j = int(np.searchsorted(ts, t))
        if j <= 0:
            return self.trajectory_T[0].copy()
        if j >= len(ts):
            return self.trajectory_T[-1].copy()
        a = (t - ts[j - 1]) / max(ts[j] - ts[j - 1], 1e-12)
        A, B = self.trajectory_T[j - 1], self.trajectory_T[j]
        qa, qb = np.array(FR.quat_wxyz(A[:3, :3])), np.array(FR.quat_wxyz(B[:3, :3]))
        if qa @ qb < 0:
            qb = -qb
        q = (1 - a) * qa + a * qb
        q /= np.linalg.norm(q)
        w, x, y, z = q
        T = np.eye(4)
        T[:3, :3] = [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]
        T[:3, 3] = (1 - a) * A[:3, 3] + a * B[:3, 3]
        return T

    # ------------------------------------------------------------------ sample_data
    def _sd(self, c: Channel, k: int, t: float, ext: str, data: bytes, T_ego, width=0, height=0) -> bool:
        """One sample_data file and its rows; T_ego: 4x4 global <- ego at t. True for a keyframe."""
        if not (self.do_nus and c.channel and 0 <= k < len(c.tokens)):
            return False
        us = self.us(t)
        key = c.keys.get(k)
        is_key = key is not None
        rel = f'{"samples" if is_key else "sweeps"}/{c.channel}/{self.logname}__{c.channel}__{us}.{ext}'
        self._file(rel, data)
        tok = c.tokens[k]
        if T_ego is None:                             # a clock-offset stamp: the ego pose at the stamp, at the end
            self.ego_rows.append({'token': tok, 'timestamp': us, '_t': float(t)})
        else:
            self.ego_rows.append({'token': tok, 'timestamp': us,
                                  'rotation': [_r(q, 10) for q in FR.quat_wxyz(T_ego[:3, :3])],
                                  'translation': [_r(q, 6) for q in T_ego[:3, 3]]})
        c.written[k] = (us, rel, is_key, int(width), int(height), ext)
        return is_key

    def camera(self, c: Channel, k: int, t: float, jpeg: bytes, T_ego) -> str | None:
        """Writes a frame; returns its image's path from the output folder (None without nuScenes)."""
        c.frames += 1
        c.bytes += len(jpeg)
        im = c.doc['image']
        self._sd(c, k, t, 'jpg', jpeg, T_ego, im['width'], im['height'])
        rel = c.written.get(k, (None, None))[1] if self.do_nus else None
        if self.bag is not None:
            t_ns = self.ns(t)
            frame = c.doc['frame_id']
            self.bag.write(c.topic + '/image_raw/compressed', t_ns, RB.compressed_image(jpeg, t_ns, frame))
            self.bag.write(c.topic + '/camera_info', t_ns, RB.camera_info(
                im['width'], im['height'], im['distortion_model'], im['D'], np.array(im['K'], float), None, None,
                t_ns, frame))
        return f'nuscenes/{rel}' if rel else None

    # ------------------------------------------------------------------ ground truth
    def gt_camera(self, c: Channel, k: int, t: float, gt: dict, image_rel):
        """A camera frame's label images and objects (ground_truth/<CHANNEL>/<name>.<kind>.png)."""
        us = self.us(t)
        chan = c.channel or ('CAM_' + c.doc['safe_tag'].upper())
        base = os.path.splitext(os.path.basename(image_rel))[0] if image_rel else f'{self.logname}__{chan}__{us}'
        paths = {}
        for kind in ('semantic', 'instance', 'range'):
            data = gt.get(kind)
            if data is None:
                paths[kind] = None
                continue
            rel = f'ground_truth/{chan}/{base}.{kind}.png'
            self._out_file(rel, data)
            paths[kind] = rel
        frame = {'sample_data_token': c.tokens[k] if (self.do_nus and c.tokens) else None, 'timestamp': us,
                 'image': image_rel, 'semantic': paths['semantic'], 'instance': paths['instance'],
                 'range': paths['range'], 'objects': gt['objects']}
        if self.bag is not None:
            frame['bag'] = {'topic': c.topic + '/image_raw/compressed', 'stamp_ns': self.ns(t)}
        if c.frozen(k):
            frame['frozen'] = True                    # the image repeats an earlier frame; the labels are the truth
        self.gt_frames.setdefault(c.doc['tag'], []).append(frame)
        self.coco_items.setdefault(c.doc['tag'], []).append((frame, gt.get('coco') or []))

    def gt_lidar(self, c: Channel, k: int, lidarseg: np.ndarray, panoptic: np.ndarray):
        """A lidar sweep's nuScenes-lidarseg and -panoptic files (keyframes and sweeps alike)."""
        if not (self.do_nus and c.channel and k in c.written):
            return
        tok = c.tokens[k]
        rel_l = f'{GT.LIDARSEG_DIR}/{VERSION}/{tok}_lidarseg.bin'
        rel_p = f'{GT.PANOPTIC_DIR}/{VERSION}/{tok}_panoptic.npz'
        self._file(rel_l, np.ascontiguousarray(lidarseg, np.uint8).tobytes())
        buf = io.BytesIO()
        np.savez_compressed(buf, data=np.ascontiguousarray(panoptic, np.uint16))
        self._file(rel_p, buf.getvalue())
        self.lidarseg_rows.append({'token': tok, 'sample_data_token': tok, 'filename': rel_l})
        self.panoptic_rows.append({'token': tok, 'sample_data_token': tok, 'filename': rel_p})
        us, _, is_key = c.written[k][:3]
        self.gt_sweeps.setdefault(c.doc['tag'], []).append({
            'sample_data_token': tok, 'timestamp': us, 'is_key_frame': bool(is_key), 'points': int(len(lidarseg)),
            'lidarseg': f'nuscenes/{rel_l}', 'panoptic': f'nuscenes/{rel_p}'})

    def lidar(self, c: Channel, k: int, t_end: float, xyz, intensity, ring, tpt, T_ego) -> bool:
        """A sweep stamped at its end; points in the sensor frame, tpt seconds from the sweep start."""
        c.frames += 1
        n = len(xyz)
        c.bytes += n * 20
        a = np.empty((n, 5), np.float32)
        a[:, :3] = xyz
        a[:, 3] = intensity
        a[:, 4] = ring
        is_key = self._sd(c, k, t_end, 'pcd.bin', a.tobytes(), T_ego)
        if self.bag is not None:
            b = np.empty(n, LIDAR_PC2)
            b['x'], b['y'], b['z'] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
            b['intensity'], b['ring'], b['t'] = intensity, ring, tpt
            t_ns = self.ns(t_end)
            self.bag.write(c.topic + '/points', t_ns, RB.pointcloud2(b, t_ns, c.doc['frame_id']))
        return is_key

    def radar(self, c: Channel, k: int, t: float, det: dict, T_ego) -> bool:
        c.frames += 1
        n = len(det['x'])
        c.bytes += n * 43
        a = np.zeros(max(n, 1), RADAR_DTYPE)
        if n:
            for f in ('x', 'y', 'z', 'rcs', 'vx', 'vy', 'vx_comp', 'vy_comp'):
                a[f] = det[f]
            a['dyn_prop'] = det['dyn_prop']
            a['id'] = np.arange(n)
            a['is_quality_valid'] = 1
            a['ambig_state'] = 3
            a['invalid_state'] = 0
            a['pdh0'] = 1
            a['x_rms'] = a['y_rms'] = a['vx_rms'] = a['vy_rms'] = 3
        else:                                         # nuScenes writes an empty scan as one NaN point
            for f in ('x', 'y', 'z', 'rcs', 'vx', 'vy', 'vx_comp', 'vy_comp'):
                a[f] = np.nan
        pcd = PCD_HEAD.format(n=len(a)).encode('ascii') + a.tobytes() + b'\n'
        is_key = self._sd(c, k, t, 'pcd', pcd, T_ego)
        if self.bag is not None:
            b = np.empty(n, RADAR_PC2)
            for f in ('x', 'y', 'z', 'v_r', 'rcs'):
                b[f] = det[f]
            t_ns = self.ns(t)
            self.bag.write(c.topic + '/detections', t_ns, RB.pointcloud2(b, t_ns, c.doc['frame_id']))
        return is_key

    def imu(self, c: Channel, t: float, acc, gyro, R_orient=None, att_sigma=0.0, head_sigma=0.0, noise=None):
        c.frames += 1
        if self.bag is None:
            return
        t_ns = self.ns(t)
        sa, sg = (noise or {}).get('accel_sigma', 0.02), (noise or {}).get('gyro_sigma', 5e-4)
        msg = {'header': RB.header(t_ns, c.doc['frame_id']),
               'angular_velocity': {'x': float(gyro[0]), 'y': float(gyro[1]), 'z': float(gyro[2])},
               'angular_velocity_covariance': [sg * sg, 0, 0, 0, sg * sg, 0, 0, 0, sg * sg],
               'linear_acceleration': {'x': float(acc[0]), 'y': float(acc[1]), 'z': float(acc[2])},
               'linear_acceleration_covariance': [sa * sa, 0, 0, 0, sa * sa, 0, 0, 0, sa * sa]}
        if R_orient is not None:
            msg['orientation'] = RB.quat_msg(R_orient)
            msg['orientation_covariance'] = [att_sigma ** 2, 0, 0, 0, att_sigma ** 2, 0, 0, 0, head_sigma ** 2]
        else:
            msg['orientation'] = {'x': 0.0, 'y': 0.0, 'z': 0.0, 'w': 0.0}
            msg['orientation_covariance'] = [-1.0, 0, 0, 0, 0, 0, 0, 0, 0]       # REP 145: no orientation
        self.bag.write(c.topic + '/imu', t_ns, msg)

    def gnss(self, c: Channel, t: float, lat, lon, alt, sig_h):
        c.frames += 1
        if self.bag is None:
            return
        t_ns = self.ns(t)
        self.bag.write(c.topic + '/fix', t_ns, {
            'header': RB.header(t_ns, c.doc['frame_id']), 'status': {'status': 2, 'service': 15},
            'latitude': float(lat), 'longitude': float(lon), 'altitude': float(alt),
            'position_covariance': [sig_h ** 2, 0.0, 0.0, 0.0, sig_h ** 2, 0.0, 0.0, 0.0, (2 * sig_h) ** 2],
            'position_covariance_type': 2})

    def tf(self, t: float, T_ego, v_body, w_body):
        if self.bag is None:
            return
        t_ns = self.ns(t)
        self.bag.write('/tf', t_ns, {'transforms': [RB.transform_stamped(T_ego, t_ns, 'odom', 'base_link')]})
        self.bag.write('/odom', t_ns, {
            'header': RB.header(t_ns, 'odom'), 'child_frame_id': 'base_link',
            'pose': {'pose': {'position': {'x': float(T_ego[0, 3]), 'y': float(T_ego[1, 3]), 'z': float(T_ego[2, 3])},
                              'orientation': RB.quat_msg(T_ego[:3, :3])}},
            'twist': {'twist': {'linear': {'x': float(v_body[0]), 'y': float(v_body[1]), 'z': float(v_body[2])},
                                'angular': {'x': float(w_body[0]), 'y': float(w_body[1]), 'z': float(w_body[2])}}}})

    def ground_truth(self, t: float, boxes: list, lifetime_s: float):
        """boxes: [{'id', 'category', 'name', 'centre' (global), 'size_lwh', 'R' (3x3 global), 'yaw', 'velocity'}]."""
        if self.bag is None:
            return
        t_ns = self.ns(t)
        markers = [RB.box_marker(b['centre'], b['R'], b['size_lwh'], t_ns, 'odom', 'ground_truth', int(b['id']),
                                 MARKER_RGBA.get(b['category'], (0.8, 0.8, 0.8, 0.5)), int(lifetime_s * 1e9))
                   for b in boxes]
        self.bag.write('/ground_truth/markers', t_ns, {'markers': markers})
        doc = {'stamp_ns': t_ns, 'frame_id': 'odom', 'boxes': [
            {'id': int(b['id']), 'instance_token': token('vantage', 'instance', self.run_key, b['id']),
             'category': b['category'], 'name': b['name'], 'centre': [round(float(q), 4) for q in b['centre']],
             'size_lwh': [round(float(q), 4) for q in b['size_lwh']], 'yaw': round(float(b['yaw']), 6),
             'velocity': [round(float(q), 4) for q in b['velocity']],
             **({'sumo_id': b['sumo_id']} if b.get('sumo_id') is not None else {})} for b in boxes]}
        self.bag.write('/ground_truth/boxes', t_ns, {'data': json.dumps(doc, separators=(',', ':'))})

    # ------------------------------------------------------------------ the end
    def close_bag(self) -> dict | None:
        if self.bag is None:
            return None
        res = self.bag.close()
        self.nbytes['rosbag'] = int(res['bytes']) + os.path.getsize(os.path.join(self.bag_dir, 'metadata.yaml'))
        return res

    def finish(self, annotations: list, instances: list, map_png: bytes | None, scene_doc: dict, run_doc: dict,
               bag_res: dict | None) -> dict:
        """The tables, dataset.json and README.md. Returns the dataset.json document."""
        counts = {'samples': len(self.samples), 'sample_data': 0, 'ego_poses': len(self.ego_rows),
                  'annotations': len(annotations), 'instances': len(instances),
                  'bag_messages': bag_res['messages'] if bag_res else 0}
        for row in self.ego_rows:                     # clock-offset stamps: the ego pose at the stamp
            if '_t' in row:
                T = self.pose_at(row.pop('_t'))
                row['rotation'] = [_r(q, 10) for q in FR.quat_wxyz(T[:3, :3])]
                row['translation'] = [_r(q, 6) for q in T[:3, 3]]
        gt_doc = self._write_ground_truth(instances) if self.gt.get('enabled') else None
        if gt_doc is not None:
            counts['gt_frames'] = sum(len(v) for v in self.gt_frames.values())
            counts['gt_objects'] = sum(len(f['objects']) for v in self.gt_frames.values() for f in v)
            counts['lidarseg'] = len(self.lidarseg_rows)
        if self.do_nus:
            sd = []
            for c in self.chans:
                if not c.channel:
                    continue
                ks = sorted(c.written)
                for i, k in enumerate(ks):
                    us, rel, is_key, w, h, ext = c.written[k]
                    key = c.keys.get(k)
                    sample = key if key is not None else int(min(np.searchsorted(self.sample_us, us, 'left'),
                                                                  len(self.samples) - 1))
                    sd.append({'token': c.tokens[k], 'sample_token': self.sample_tok[sample],
                               'ego_pose_token': c.tokens[k], 'calibrated_sensor_token': self.cs_token[c.channel],
                               'timestamp': us, 'fileformat': ext.split('.')[0], 'is_key_frame': is_key,
                               'height': h, 'width': w, 'filename': rel,
                               'prev': c.tokens[ks[i - 1]] if i else '',
                               'next': c.tokens[ks[i + 1]] if i + 1 < len(ks) else ''})
            counts['sample_data'] = len(sd)
            nus = self.rig['nuscenes']
            log_tok = token('vantage', 'log', self.run_key)
            scene_tok = token('vantage', 'scene', self.run_key)
            map_tok = token('vantage', 'map', self.run_key)
            samples = [{'token': tok, 'timestamp': self.sample_us[i], 'prev': self.sample_tok[i - 1] if i else '',
                        'next': self.sample_tok[i + 1] if i + 1 < len(self.sample_tok) else '',
                        'scene_token': scene_tok} for i, tok in enumerate(self.sample_tok)]
            if map_png is not None:
                self._file(f'maps/{map_tok}.png', map_png)
            sc = self.rig['scenario']
            v = self.rig['vehicle']
            tables = {
                'attribute': [{'token': token('vantage', 'attribute', n), 'name': n, 'description': d}
                              for n, d in nus['attributes']],
                'calibrated_sensor': self.extra['cal_rows'],
                'category': GT.nuscenes_categories(lambda n: token('vantage', 'category', n), nus['categories']),
                'ego_pose': sorted(self.ego_rows, key=lambda r: r['timestamp']),
                'instance': instances,
                'log': [{'token': log_tok, 'logfile': self.logname,
                         'vehicle': f'{scene_doc.get("vehicle") or v.get("blueprint", "")} (rig of '
                                    f'{v.get("name") or "the project"})',
                         'date_captured': _dt.datetime.fromtimestamp(self.epoch_us / 1e6,
                                                                     _dt.timezone.utc).strftime('%Y-%m-%d'),
                         'location': f'carla-{scene_doc.get("town", "")}'}],
                'map': [{'token': map_tok, 'log_tokens': [log_tok], 'category': 'semantic_prior',
                         'filename': f'maps/{map_tok}.png'}],
                'sample': samples,
                'sample_annotation': annotations,
                'sample_data': sorted(sd, key=lambda r: r['timestamp']),
                'scene': [{'token': scene_tok, 'log_token': log_tok, 'nbr_samples': len(samples),
                           'first_sample_token': self.sample_tok[0], 'last_sample_token': self.sample_tok[-1],
                           'name': 'scene-0001',
                           'description': f'CARLA {scene_doc.get("town", "")} drive, {sc["duration_s"]:g} s, '
                                          f'{sc["traffic"]} vehicles, seed {sc["seed"]}; '
                                          f'{self.rig["project"]} / {self.rig["variant"]}'}],
                'sensor': self.extra['sensor_rows'],
                'visibility': [{'token': t, 'level': lv, 'description': d} for t, lv, d in nus['visibility']],
            }
            if self.lidarseg_rows:                    # nuScenes-lidarseg and -panoptic (devkit 1.1+)
                tables['lidarseg'] = self.lidarseg_rows
                tables['panoptic'] = self.panoptic_rows
            for name, rows in tables.items():
                self._file(f'{VERSION}/{name}.json', (json.dumps(rows, indent=1) + '\n').encode('utf-8'))
        channels = [{'kind': c.kind, 'tag': c.doc['tag'], 'channel': c.channel, 'topic': c.topic, 'frames': c.frames,
                     'keyframes': sum(1 for x in c.written.values() if x[2]), 'bytes': c.bytes,
                     'planned_frames': len(c.stamps)} for c in self.chans]
        out = self.rig['output']
        settings = {'duration_s': self.rig['scenario']['duration_s'], 'seed': self.rig['scenario']['seed'],
                    'image_scale': out['image_scale'], 'camera_rate': out['camera_rate'],
                    'channel_names': out['channel_names'], 'keyframe_hz': out['keyframe_hz'],
                    'jpeg_quality': out['jpeg_quality'], 'nuscenes': self.do_nus, 'rosbag': self.do_bag,
                    'lidar_noise_cm': out['lidar_noise_cm'], 'drop_self_hits': out['drop_self_hits'],
                    'undistort': out['undistort'], 'scene': 'carla'}
        doc = {
            'schema': DATASET_SCHEMA, 'generator': 'carla', 'project': self.rig['project'],
            'variant': self.rig['variant'],
            'written': {'nuscenes': 'nuscenes' if self.do_nus else None,
                        'nuscenes_version': VERSION if self.do_nus else None,
                        'rosbag': os.path.relpath(self.bag_dir, self.out_dir) if bag_res is not None else None},
            'settings': settings, 'scene': scene_doc, 'counts': counts,
            'bytes': {'nuscenes': self.nbytes['nuscenes'], 'rosbag': self.nbytes['rosbag'],
                      'ground_truth': self.nbytes.get('ground_truth', 0),
                      'total': self.nbytes['nuscenes'] + self.nbytes['rosbag'] + self.nbytes.get('ground_truth', 0)},
            'ground_truth': ({'index': GT.INDEX, 'schema': GT.SCHEMA, 'cameras': sorted(self.gt_frames),
                              'lidars': sorted(self.gt_sweeps)} if gt_doc is not None else None),
            'faults_injected': run_doc.get('faults_injected', []),
            'channels': channels, 'bag_topics': (bag_res or {}).get('topics', {}),
            'conventions': {
                'global_frame': 'CARLA\'s world with y mirrored (x east, y north, z up, metres), shifted by '
                                'scene.global_offset_m so the map mask starts at its bottom-left corner; the nuScenes '
                                'global frame and the bag\'s odom frame',
                'ego_frame': 'vehicle, ISO 8855: x forward, y left, z up, origin at the rear-axle centre on the ground '
                             '(nuScenes ego, bag base_link); in CARLA\'s vehicle at scene.ego_origin_in_actor_m',
                'time': f'microseconds (nuScenes) / nanoseconds (bag) since the Unix epoch; t = 0 of the drive is '
                        f'{self.epoch_us} us, a sync pulse; simulation time, one step = 1/tick_hz',
                'camera_timestamp': 'the simulation step the frame was rendered at: its planned exposure centre '
                                    'rounded to a step',
                'lidar_timestamp': 'end of the sweep (nuScenes); points in the sensor frame, each step\'s slice of '
                                   'the rotation at that step\'s world state; bag field t = seconds from the sweep '
                                   'start (the firing time from the point\'s azimuth)',
                'lidar_fields': 'nuScenes .pcd.bin float32 x y z intensity ring (ring 0 = the lowest beam; intensity '
                                '= 255 × CARLA\'s atmospheric attenuation, not a reflectivity); bag PointCloud2 x y z '
                                'intensity ring t',
                'radar_fields': 'nuScenes .pcd 18 fields (vx, vy: the radial velocity along the ray in the radar '
                                'frame; _comp: with the radar\'s own motion removed; rcs 0: CARLA gives none); bag '
                                'PointCloud2 x y z v_r rcs',
                'images': 'rectified pinhole, camera_intrinsic exact' if out['undistort'] else
                          'the real lens (CARLA pinhole views remapped through Vantage\'s lens model): '
                          'camera_intrinsic + vantage_distortion (Kannala-Brandt) as the nuscenes_calib export, '
                          'scaled to the image size',
                'bag_frames': '<tag>_link (sensor bodies, REP 103), <tag>_optical_frame (camera images)'},
            'assumptions': self.rig['assumptions'],
            'carla': run_doc,
            'notes': self.notes,
        }
        with open(os.path.join(self.out_dir, 'dataset.json'), 'w', encoding='utf-8') as fh:
            fh.write(json.dumps(doc, indent=1, default=str) + '\n')
        with open(os.path.join(self.out_dir, 'README.md'), 'w', encoding='utf-8') as fh:
            fh.write(_readme(doc, self))
        return doc


    def _write_ground_truth(self, instances) -> dict:
        """ground_truth/index.json (vantage.ground_truth/1) and the COCO files; an object's instance token is kept
        only when nuScenes has the instance (it was annotated at a keyframe)."""
        valid = {r['token'] for r in instances}
        cams = {}
        chans = {c.doc['tag']: c for c in self.chans}
        for tag, frames in self.gt_frames.items():
            for f in frames:
                for o in f['objects']:
                    if o.get('instance_token') not in valid:
                        o['instance_token'] = None
            c = chans[tag]
            im = c.doc['image']
            cams[tag] = {'channel': c.channel or ('CAM_' + c.doc['safe_tag'].upper()), 'width': im['width'],
                         'height': im['height'], 'frames': frames}
        lids = {tag: {'channel': chans[tag].channel, 'sweeps': sw} for tag, sw in self.gt_sweeps.items()}
        notes = list(self.gt.get('notes', []))
        doc = {'schema': GT.SCHEMA, 'project': self.rig['project'], 'variant': self.rig['variant'],
               'classes': GT.classes_doc(), 'cameras': cams, 'lidars': lids,
               'nuscenes_lidarseg_categories': [{'index': i, 'name': n} for i, n, _ in GT.NUSCENES_LIDARSEG],
               'conventions': {
                   'paths': 'relative to this drive\'s folder (the one holding ground_truth/ and nuscenes/)',
                   'pixels': 'output image pixel edges: pixel (i, j) covers [i, i+1) × [j, j+1); bbox [x0, y0, x1, y1]',
                   'semantic': 'uint8 PNG, the class id per pixel (classes)',
                   'instance': 'uint16 PNG, 0 = no object, else the object\'s id (objects[].id: stable over the drive; '
                               'an actor the bridge spawned 1, 2, … in spawn order, a map object 1000 + its index)',
                   'range': f'uint16 PNG = round(range_m · {GT.RANGE_SCALE:g}) along the pixel\'s own ray from the '
                            f'camera\'s optical centre, 0 = nothing within {GT.RANGE_MAX_M:.1f} m; the views\' planar '
                            'depth read at the ray\'s exact position (inverse depth interpolated within one surface, the ground\'s '
                            'meshes (road, lines, sidewalk) one: exact on planes; the nearest view pixel\'s across '
                            'an object\'s edge)',
                   'box3d': 'vehicle frame (ISO 8855, rear axle on the ground) at the frame\'s capture time: centre, '
                            'size [length, width, height] m, yaw degrees',
                   'bbox_full': 'the 3D box\'s edges through the lens model, not clipped to the image (null when '
                                'none of the box is within the lens\'s field)',
                   'truncated': 'share of the 3D box\'s surface outside the image or the lens\'s field',
                   'occluded': 'of the pixels whose ray meets the 3D box, the share hidden by something nearer (from '
                               'the range image): hidden / (visible + hidden)',
                   'range_m': 'camera optical centre to the 3D box\'s centre',
                   'mount': 'the camera\'s actual mount: with a rotation or translation fault the labels match the '
                            'image (the calibration states the plan)',
                   'lidarseg': 'nuScenes-lidarseg: uint8 category index per point of the sweep\'s .pcd.bin (its '
                               'order); panoptic: uint16 index · 1000 + instance (a spawned actor its spawn index, a '
                               'map\'s parked vehicle 500 + its index, stuff 0)'},
               'notes': notes}
        self._out_file(GT.INDEX, (json.dumps(doc, indent=1, default=_json_default) + '\n').encode('utf-8'))
        cats = [{'id': c[0], 'name': c[1], 'supercategory': c[3].split('.')[0]} for c in GT.CLASSES if c[0] > 0]
        for tag, items in self.coco_items.items():
            images, anns = [], []
            for i, (f, rles) in enumerate(items):
                images.append({'id': i + 1, 'file_name': f['image'] or f['semantic'], 'width': cams[tag]['width'],
                               'height': cams[tag]['height'], 'sample_data_token': f['sample_data_token'],
                               'timestamp': f['timestamp']})
                for o, rle in zip(f['objects'], rles):
                    x0, y0, x1, y1 = o['bbox']
                    anns.append({'id': len(anns) + 1, 'image_id': i + 1, 'category_id': GT.CLASS_ID[o['class']],
                                 'bbox': [x0, y0, x1 - x0, y1 - y0], 'area': o['pixels'], 'iscrowd': 0,
                                 'segmentation': rle, 'object_id': o['id'], 'actor_id': o['actor_id'],
                                 'instance_token': o['instance_token'], 'nuscenes_category': o['category'],
                                 'occluded': o['occluded'], 'truncated': o['truncated']})
            coco = {'info': {'description': f'Vantage CARLA drive {self.rig["project"]} / {self.rig["variant"]}, '
                                            f'{cams[tag]["channel"]}', 'version': GT.SCHEMA},
                    'images': images, 'annotations': anns, 'categories': cats}
            self._out_file(f'ground_truth/coco/{cams[tag]["channel"]}.json',
                           (json.dumps(coco, separators=(',', ':'), default=_json_default) + '\n').encode('utf-8'))
        return doc


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return str(o)


def _readme(doc, rec: Recorder) -> str:
    rows = '\n'.join(f'| {c["channel"] or c["topic"]} | {c["kind"]} | {c["frames"]} | {c["keyframes"]} |'
                     for c in doc['channels'])
    sc = doc['scene']
    lines = []
    faults = doc.get('faults_injected') or []
    if faults:
        lines.append('**This drive is deliberately faulty** (fault injection with known answers): '
                     + '; '.join(_fault_text(f) for f in faults) + '. The calibration (nuScenes calibrated_sensor, '
                     'tf, dataset.json, the bag) states the PLAN, as a real faulty rig\'s would; run.json '
                     '`faults_injected` has each fault as applied.\n')
    if doc.get('ground_truth'):
        lines.append(f'- Ground truth: `{doc["ground_truth"]["index"]}` (schema `{doc["ground_truth"]["schema"]}`): per '
                     'camera frame semantic, instance and range images and the objects with their boxes; COCO files '
                     'in `ground_truth/coco/`; nuScenes-lidarseg and -panoptic per lidar sweep.')
    if rec.do_nus:
        lines.append(f'- nuScenes: `nuscenes/` version `{VERSION}`: {doc["counts"]["samples"]} samples, '
                     f'{doc["counts"]["sample_data"]} sample_data, {doc["counts"]["annotations"]} annotations of '
                     f'{doc["counts"]["instances"]} instances. Load: `NuScenes(version=\'{VERSION}\', '
                     f'dataroot=\'<this folder>/nuscenes\')`')
    if doc['written']['rosbag']:
        lines.append(f'- ROS 2 bag: `{doc["written"]["rosbag"]}/` ({doc["counts"]["bag_messages"]} messages): '
                     f'`ros2 bag info {doc["written"]["rosbag"]}`')
    scen = (doc.get('carla') or {}).get('scenario') or {}
    if scen.get('name') and scen['name'] != 'free_drive':
        lines.append(f'- Scenario: {scen["name"]} ({", ".join(f"{k} {v}" for k, v in scen.get("params", {}).items())}); '
                     f'events in run.json `scenario.events`.')
    tr = (doc.get('carla') or {}).get('traffic_source')
    if tr:
        vc = tr.get('vehicles_in_carla') or {}
        lines.append(f'- Traffic: Eclipse SUMO {tr.get("sumo_version") or ""} co-simulated in lockstep '
                     f'({(tr.get("net") or {}).get("kind", "")} net {(tr.get("net") or {}).get("file", "")}, '
                     f'{tr.get("vehicles_spawned", 0)} vehicles spawned, {vc.get("mean", 0):g} on average in CARLA, '
                     f'traffic lights run by {(tr.get("tls") or {}).get("owner", "carla").upper()}); each SUMO '
                     'vehicle\'s id and type in run.json `traffic_source.actors`.')
    if (doc.get('carla') or {}).get('actors'):
        lines.append(f'- An actors scene: {len(doc["carla"]["actors"])} actors around the ego, which stands still '
                     '(run.json `actors`).')
    return f"""# Vantage CARLA drive · {doc['project']} / {doc['variant']}

A {sc.get('duration_s', 0):g} s drive in CARLA {sc.get('carla_version', '')} ({sc.get('town', '')}, weather
{sc.get('weather', '')}, {sc.get('traffic', 0)} vehicles, seed {sc.get('seed', 0)}) recorded by the planned rig on a
{sc.get('vehicle', '')} (the CARLA vehicle nearest the project's by size: its body, not the project's, is in the
images), written as Vantage's sample dataset writes a drive.

{chr(10).join(lines)}

| Channel | Kind | Frames | Keyframes |
|---|---|---:|---:|
{rows}

Conventions: see dataset.json (`conventions`); run.json has the simulation's own report (steps, timings, what was
measured of the vehicle, every note).

{doc['assumptions']}
"""


def _fault_text(f) -> str:
    k = f.get('kind')
    if k == 'rotation':
        return f'{f["tag"]} rotated by {f["deg"]} ° (yaw, pitch, roll in its own frame)'
    if k == 'translation':
        return f'{f["tag"]} moved by {f["m"]} m'
    if k == 'clock_offset':
        return f'{f["tag"]} stamped {f["ms"]:+g} ms off'
    if k == 'drop':
        return f'{f["tag"]} dropped {len(f.get("dropped_frames", f.get("frames", [])))} frames'
    if k == 'freeze':
        return f'{f["tag"]} frozen from {f["from_s"]:g} s to {f["to_s"]:g} s'
    if k == 'range_bias':
        return f'{f["tag"]} ranges {f["m"]:+g} m'
    return str(k)
