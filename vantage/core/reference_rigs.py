"""Reference rigs: how published autonomous-driving fleets and datasets place their cameras and sensors.

Every figure carries its provenance:

- 'measured': read from a dataset's published calibration file (nuScenes calibrated_sensor.json, KITTI calib_*.txt);
- 'stated': from the company's, dataset's or paper's own text or tables;
- 'estimated': from photos, diagrams or community measurements, with how (never presented as a datasheet value).

Unknown fields are left empty and listed per rig. A rig whose layout was never published (counts and roles only:
Waymo's 5th and 6th generation, Mobileye SuperVision, Wayve) is kept for comparison of counts and roles but cannot be
turned into a variant: Vantage does not invent positions for it.

Sources read 2026-11-04:

- nuScenes v1.0-mini `calibrated_sensor.json` + `sensor.json` (the first of the dataset's two calibration sets; two
  independent public copies compared byte for byte, e.g.
  https://github.com/1904jonathan/PardesLine/blob/f3f9317/v1.0-mini/calibrated_sensor.json and
  https://github.com/YYH-GPU/FastBev_infer_DNN/blob/bb8f78c/data/v1.0-mini/calibrated_sensor.json); nuScenes data is
  CC BY-NC-SA 4.0 (https://www.nuscenes.org/terms-of-use): these calibration figures are quoted with attribution.
  The ego frame is the midpoint of the rear axle (https://www.nuscenes.org/nuscenes#data-format); its height is not
  published: the front radar at z = 0.50 m and the cameras at about 1.5 m on a 1.56 m ZOE put it at road level, as
  Vantage's frame is (an inference, noted on the rig).
- KITTI: A. Geiger et al., "Vision meets Robotics: The KITTI Dataset", IJRR 2013, Sec. II and Fig. 3
  (https://www.cvlibs.net/publications/Geiger2013IJRR.pdf) and the raw-data calibration 2011_09_26
  (https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_calib.zip: calib_cam_to_cam.txt,
  calib_velo_to_cam.txt).
- Waymo Open Dataset: P. Sun et al., "Scalability in Perception for Autonomous Driving: Waymo Open Dataset", CVPR
  2020, Tables 2 and 3 (https://arxiv.org/abs/1912.04838); https://waymo.com/open/about/.
- Waymo Driver 5th generation: https://waymo.com/blog/2020/03/introducing-5th-generation-waymo-driver/ ;
  6th generation: https://waymo.com/blog/2024/08/meet-the-6th-generation-waymo-driver/.
- Tesla Autopilot cameras (HW3): ranges and the wide camera's 120° fisheye as Tesla's Autopilot page states them
  (https://www.tesla.com/autopilot, quoted by https://www.notateslaapp.com/tesla-reference/1452/ and
  https://electrek.co/2017/05/16/tesla-autopilot-2-0-can-see/; Tesla's page itself was not reachable from here);
  the main and narrow cameras' fields measured by the community from Tesla's published image
  (https://teslamotorsclub.com/tmc/threads/ap2-0-cameras-capabilities-and-limitations.86430).
- Argoverse 2: https://www.argoverse.org/av2 and https://argoverse.github.io/user-guide/datasets/sensor.html.
- Mobileye SuperVision (Zeekr 001): https://www.mobileye.com/news/geely-holding-group-expands-mobileye-collaboration/.
- Wayve: https://www.wired.com/story/wayve-ai-self-driving-system/ (as indexed; no camera count published).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

CHECKED = '2026-11-04'
PROVENANCE = ('measured', 'stated', 'estimated')


@dataclass
class RefSensor:
    """A camera, lidar or radar of a reference rig. `prov` maps a field name to 'measured' | 'stated' |
    'estimated'; a field without a value is unknown. Position in ISO 8855 (x forward, y left, z up, metres, origin
    at the rear-axle centre on the road), angles in degrees with Vantage's conventions (yaw left positive, pitch
    positive down)."""
    name: str
    kind: str = 'camera'                 # camera | lidar | radar
    role: str = ''
    hfov_deg: float | None = None
    vfov_deg: float | None = None
    range_m: float | None = None
    resolution: tuple | None = None      # (width, height) px
    position_m: tuple | None = None      # (x, y, z)
    yaw: float | None = None
    pitch: float | None = None
    roll: float | None = None
    model: str = ''                      # the product, as named by the source
    library_model: str = ''              # the closest Vantage library model (core.sensors / core.lens), when matched
    prov: dict = field(default_factory=dict)
    note: str = ''

    def missing(self) -> list[str]:
        return [k for k in ('hfov_deg', 'vfov_deg', 'range_m', 'resolution', 'position_m', 'yaw', 'pitch')
                if getattr(self, k) is None]


@dataclass
class ReferenceRig:
    id: str
    company: str
    name: str
    vehicle: str
    year: str
    sources: list                         # [(what, url)]
    cameras: list = field(default_factory=list)
    lidars: list = field(default_factory=list)
    radars: list = field(default_factory=list)
    other: str = ''
    counts: dict = field(default_factory=dict)      # {'cameras': n, 'lidars': n, 'radars': n} as stated (None: not)
    counts_prov: str = 'stated'
    notes: str = ''
    library_vehicle_id: str | None = None
    layout: str = 'none'                 # measured | mixed | estimated | none (no published positions)

    @property
    def sensors(self):
        return self.cameras + self.lidars + self.radars

    @property
    def placed(self) -> bool:
        return self.layout != 'none' and all(s.position_m is not None for s in self.sensors)

    def provenance_mix(self) -> dict:
        out = {k: 0 for k in PROVENANCE}
        for s in self.sensors:
            for v in s.prov.values():
                out[v] = out.get(v, 0) + 1
        return out


def _q_to_R(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def camera_ypr_from_quaternion(q) -> tuple:
    """Vantage (yaw, pitch, roll) of a camera whose nuScenes rotation quaternion (w, x, y, z) takes its OpenCV optical
    frame (x right, y down, z forward) to the ego frame."""
    from .mathutil import BODY_TO_OPTICAL, euler_zyx
    return tuple(float(v) for v in euler_zyx(_q_to_R(q) @ BODY_TO_OPTICAL))


def sensor_ypr_from_quaternion(q) -> tuple:
    """(yaw, pitch, roll) of a lidar or radar whose quaternion takes its own frame (x forward, y left, z up) to the
    ego frame."""
    from .mathutil import euler_zyx
    return tuple(float(v) for v in euler_zyx(_q_to_R(q)))


# --------------------------------------------------------------------------------------------------- the data
# nuScenes v1.0-mini calibrated_sensor.json, first calibration set: (channel, translation, rotation wxyz, fx, fy, cx,
# cy) for the cameras (1600 × 900), (channel, translation, rotation) for the lidar and radars. Quoted to the file's
# digits as rounded here (6 decimals).
NUSCENES_CAMERAS = [
    ('CAM_FRONT', (1.700791, 0.015946, 1.510958), (0.499801543, -0.503031616, 0.499779811, -0.497370838),
     1266.417, 491.507, 816.267),
    ('CAM_FRONT_RIGHT', (1.550848, -0.493405, 1.495748), (0.206034797, -0.202694058, 0.682450782, -0.671361088),
     1260.847, 495.334, 807.968),
    ('CAM_BACK_RIGHT', (1.014878, -0.480568, 1.562395), (0.122809801, -0.132400843, -0.700430582, 0.690496031),
     1259.514, 501.196, 807.253),
    ('CAM_BACK', (0.028326, 0.003451, 1.579103), (0.503787267, -0.497402498, -0.494185022, 0.50454961),
     809.221, 481.778, 829.22),
    ('CAM_BACK_LEFT', (1.035691, 0.484795, 1.59097), (0.692418559, -0.703161942, -0.116483428, 0.112033179),
     1256.741, 492.776, 792.113),
    ('CAM_FRONT_LEFT', (1.523878, 0.494631, 1.509328), (0.675726503, -0.673626652, 0.21214015, -0.211228271),
     1272.598, 479.752, 826.615),
]
NUSCENES_LIDAR = ('LIDAR_TOP', (0.943713, 0.0, 1.84023), (0.707795512, -0.006492242, 0.010646215, -0.706307314))
NUSCENES_RADARS = [
    ('RADAR_FRONT', (3.412, 0.0, 0.5), (0.999998477, 0.0, 0.0, 0.001745328)),
    ('RADAR_FRONT_LEFT', (2.422, 0.8, 0.78), (0.71715392, 0.0, 0.0, 0.696914811)),
    ('RADAR_FRONT_RIGHT', (2.422, -0.8, 0.77), (0.701033739, 0.0, 0.0, -0.713128107)),
    ('RADAR_BACK_LEFT', (-0.562, 0.628, 0.53), (0.048762607, 0.0, 0.0, 0.998810396)),
    ('RADAR_BACK_RIGHT', (-0.562, -0.618, 0.53), (0.033940134, 0.0, 0.0, -0.999423868)),
]

# KITTI raw calibration 2011_09_26: each camera's translation from calib_cam_to_cam.txt (T_0i, cam0 frame, m) and the
# Velodyne → cam0 transform from calib_velo_to_cam.txt.
KITTI_T = {'cam0': (0.0, 0.0, 0.0), 'cam1': (-5.370000e-01, 4.822061e-03, -1.252488e-02),
           'cam2': (5.956621e-02, 2.900141e-04, 2.577209e-03), 'cam3': (-4.731050e-01, 5.551470e-03, -5.250882e-03)}
KITTI_VELO_R = (7.533745e-03, -9.999714e-01, -6.166020e-04, 1.480249e-02, 7.280733e-04, -9.998902e-01, 9.998621e-01,
                7.523790e-03, 1.480755e-02)
KITTI_VELO_T = (-4.069766e-03, -7.631618e-02, -2.717806e-01)
KITTI_CAM_H, KITTI_VELO_H, KITTI_IMU_H = 1.65, 1.73, 0.93          # stated, Fig. 3 (heights above the road)
KITTI_VELO_X_EST = 1.64            # estimated: Fig. 3's 2.71 m wheelbase, cameras 0.80 m behind the front axle


def _kitti_offsets():
    """Each KITTI camera's position relative to cam0 in the vehicle frame (x forward, y left, z up), and the
    Velodyne's, from the measured calibration (x_cam = R x_velo + T, x_i = R_0i x_0 + T_0i): a camera's origin in
    cam0's optical frame is -R_0i^T T_0i ≈ -T_0i (R_0i within a degree of I); the Velodyne's origin is T."""
    o2v = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])   # optical (x right, y down, z fwd) → vehicle
    cams = {k: o2v @ (-np.array(t)) for k, t in KITTI_T.items()}
    velo = o2v @ np.array(KITTI_VELO_T)
    return cams, velo


def _nuscenes() -> ReferenceRig:
    src = [('nuScenes v1.0-mini calibrated_sensor.json (first calibration set)',
            'https://github.com/1904jonathan/PardesLine/blob/f3f9317/v1.0-mini/calibrated_sensor.json'),
           ('nuScenes data format (ego frame: rear-axle midpoint)', 'https://www.nuscenes.org/nuscenes#data-format'),
           ('nuScenes sensor setup (cameras 1600 × 900 Basler, Velodyne HDL-32E, Continental ARS 408-21)',
            'https://www.nuscenes.org/nuscenes#data-collection')]
    cams = []
    for ch, t, q, fx, cy, cx in NUSCENES_CAMERAS:
        yaw, pitch, roll = camera_ypr_from_quaternion(q)
        cams.append(RefSensor(ch, 'camera', role={'CAM_BACK': 'rear'}.get(ch, 'surround'),
                              hfov_deg=round(2 * math.degrees(math.atan(800.0 / fx)), 2),
                              vfov_deg=round(2 * math.degrees(math.atan(450.0 / fx)), 2), resolution=(1600, 900),
                              position_m=t, yaw=round(yaw, 3), pitch=round(pitch, 3), roll=round(roll, 3),
                              model='Basler acA1600-60gc', library_model='basler_aca1600_60gc',
                              prov={'hfov_deg': 'measured', 'vfov_deg': 'measured', 'resolution': 'stated',
                                    'position_m': 'measured', 'yaw': 'measured', 'pitch': 'measured', 'roll': 'measured',
                                    'model': 'stated'},
                              note=f'fx {fx:.1f} px (calibrated_sensor camera_intrinsic); field of the pinhole model '
                                   'over 1600 px'))
    ch, t, q = NUSCENES_LIDAR
    y, p, r = sensor_ypr_from_quaternion(q)
    lid = [RefSensor(ch, 'lidar', 'roof 360°', position_m=t, yaw=round(y, 3), pitch=round(p, 3), roll=round(r, 3),
                     model='Velodyne HDL-32E', library_model='velodyne_hdl32e',
                     prov={'position_m': 'measured', 'yaw': 'measured', 'pitch': 'measured', 'roll': 'measured',
                           'model': 'stated'},
                     note='its frame has x to the right (yaw −89.9°), as nuScenes defines LIDAR_TOP')]
    rads = []
    for ch, t, q in NUSCENES_RADARS:
        y, p, r = sensor_ypr_from_quaternion(q)
        rads.append(RefSensor(ch, 'radar', 'corner' if 'LEFT' in ch or 'RIGHT' in ch else 'front long-range',
                              position_m=t, yaw=round(y, 3), pitch=round(p, 3), roll=round(r, 3),
                              model='Continental ARS 408-21', library_model='conti_ars408_21',
                              prov={'position_m': 'measured', 'yaw': 'measured', 'pitch': 'measured', 'roll': 'measured',
                                    'model': 'stated'}))
    return ReferenceRig('nuscenes', 'Motional (nuTonomy)', 'nuScenes data-collection rig', 'Renault ZOE (Phase 1)',
                        '2018', src, cams, lid, rads, other='GPS/IMU (not in calibrated_sensor)',
                        counts={'cameras': 6, 'lidars': 1, 'radars': 5}, layout='measured',
                        library_vehicle_id='renault_zoe_ph1',
                        notes='Positions and angles measured (the dataset\'s calibration). The ego frame\'s height is '
                              'not published; it is taken at road level, which the front radar at 0.50 m and the '
                              'cameras at about 1.5 m on a 1.56 m car support. nuScenes data is CC BY-NC-SA 4.0.')


def _kitti() -> ReferenceRig:
    src = [('Geiger et al., Vision meets Robotics: The KITTI Dataset, IJRR 2013, Sec. II, Fig. 3',
            'https://www.cvlibs.net/publications/Geiger2013IJRR.pdf'),
           ('KITTI raw calibration 2011_09_26 (calib_cam_to_cam.txt, calib_velo_to_cam.txt)',
            'https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_calib.zip')]
    rel, velo = _kitti_offsets()
    # the Velodyne on the centreline (estimated) at Fig. 3's height; the cameras relative to it from the calibration
    vx = KITTI_VELO_X_EST
    cam0 = np.array([vx, 0.0, KITTI_CAM_H]) - np.array([velo[0], velo[1], 0.0])
    names = {'cam0': ('Cam 0 (grey, left)', 'FL2-14S3M-C', 'flir_fl2_14s3m'),
             'cam1': ('Cam 1 (grey, right)', 'FL2-14S3M-C', 'flir_fl2_14s3m'),
             'cam2': ('Cam 2 (colour, left)', 'FL2-14S3C-C', 'flir_fl2_14s3c'),
             'cam3': ('Cam 3 (colour, right)', 'FL2-14S3C-C', 'flir_fl2_14s3c')}
    cams = []
    for k, (nm, model, lib) in names.items():
        pos = cam0 + np.array([rel[k][0], rel[k][1], 0.0])
        cams.append(RefSensor(nm, 'camera', 'forward stereo', hfov_deg=90.0, vfov_deg=35.0, resolution=(1392, 512),
                              position_m=(round(float(pos[0]), 4), round(float(pos[1]), 4), KITTI_CAM_H),
                              yaw=0.0, pitch=0.0, roll=0.0, model=f'Point Grey Flea2 {model}, 4 mm Edmund Optics',
                              library_model=lib,
                              prov={'hfov_deg': 'stated', 'vfov_deg': 'stated', 'resolution': 'stated',
                                    'position_m': 'estimated', 'yaw': 'estimated', 'pitch': 'estimated',
                                    'roll': 'estimated', 'model': 'stated'},
                              note='lateral spacing measured (calib_cam_to_cam T_0i), height stated (1.65 m), fore-aft '
                                   'estimated from Fig. 3; "opening angle ~90°" and the ROI\'s ~35° vertical are the '
                                   'paper\'s (a diagonal-ish figure for a 4 mm lens on the 1/2" ICX267); facing '
                                   'forward, level (the calibration rotations are within 1°)'))
    lid = [RefSensor('Velodyne', 'lidar', 'roof 360°', vfov_deg=26.8, range_m=120.0,
                     position_m=(vx, 0.0, KITTI_VELO_H), yaw=0.0, pitch=0.0, roll=0.0, model='Velodyne HDL-64E',
                     library_model='velodyne_hdl64e_s2',
                     prov={'vfov_deg': 'stated', 'range_m': 'stated', 'position_m': 'estimated', 'yaw': 'estimated',
                           'pitch': 'estimated', 'roll': 'estimated', 'model': 'stated'},
                     note='0.27 m behind the cameras (measured, velo_to_cam); height 1.73 m stated; on the centreline '
                          'and 1.64 m ahead of the rear axle estimated from Fig. 3')]
    return ReferenceRig('kitti', 'KIT / Toyota TI Chicago', 'KITTI recording platform', 'VW Passat B6 Variant', '2011',
                        src, cams, lid, [], other='OXTS RT 3003 GPS/IMU at 0.93 m (stated)',
                        counts={'cameras': 4, 'lidars': 1, 'radars': 0}, layout='mixed',
                        library_vehicle_id='vw_passat_b6_variant',
                        notes='Relative poses measured from the raw calibration; heights stated (Fig. 3); the '
                              'Velodyne\'s fore-aft position and the centreline estimated from Fig. 3.')


def _tesla_hw3() -> ReferenceRig:
    src = [('Tesla Autopilot page: camera ranges, wide camera 120° fisheye (as quoted)',
            'https://www.notateslaapp.com/tesla-reference/1452/'),
           ('Tesla Autopilot 2.0 camera ranges (Electrek, from Tesla)',
            'https://electrek.co/2017/05/16/tesla-autopilot-2-0-can-see/'),
           ('Community measurement of the forward cameras\' fields from Tesla\'s image (120°, 46°, 34°)',
            'https://teslamotorsclub.com/tmc/threads/ap2-0-cameras-capabilities-and-limitations.86430')]
    E = 'estimated'

    def cam(nm, role, hfov, hp, rng, place, yaw, pitch, note):
        return RefSensor(nm, 'camera', role, hfov_deg=hfov, range_m=rng, position_m=place, yaw=yaw, pitch=pitch,
                         roll=0.0, prov=dict({'range_m': 'stated', 'position_m': E, 'yaw': E, 'pitch': E, 'roll': E},
                                             **({'hfov_deg': hp} if hfov is not None else {})), note=note)
    cams = [
        cam('Narrow forward', 'forward long range', 34.0, E, 250.0, 'windscreen', 0.0, 0.0,
            'behind the windscreen top, in the triple-camera cluster; field from the community measurement'),
        cam('Main forward', 'forward', 46.0, E, 150.0, 'windscreen', 0.0, 0.0,
            'behind the windscreen top; field from the community measurement'),
        cam('Wide forward', 'forward wide', 120.0, 'stated', 60.0, 'windscreen', 0.0, 3.0,
            'behind the windscreen top; Tesla: a 120° fisheye'),
        cam('Forward-looking side left', 'side forward', None, '', 80.0, 'b_pillar_left', 60.0, 10.0,
            'in the left B-pillar; its field is not published'),
        cam('Forward-looking side right', 'side forward', None, '', 80.0, 'b_pillar_right', -60.0, 10.0,
            'in the right B-pillar; its field is not published'),
        cam('Rearward-looking side left', 'side rearward', None, '', 100.0, 'repeater_left', 135.0, 10.0,
            'in the left front-fender repeater; its field is not published'),
        cam('Rearward-looking side right', 'side rearward', None, '', 100.0, 'repeater_right', -135.0, 10.0,
            'in the right front-fender repeater; its field is not published'),
        cam('Rear view', 'rear', None, '', 50.0, 'rear', 180.0, 25.0,
            'above the rear licence plate; its field is not published'),
    ]
    return ReferenceRig('tesla_hw3', 'Tesla', 'Autopilot / FSD camera suite (HW3)', 'Tesla Model 3', '2019–2023', src,
                        cams, [], [], other='HW3 (12 ultrasonic sensors and a forward radar on earlier cars; removed '
                                            'from 2021–2022 builds, "Tesla Vision"). HW4: higher-resolution cameras; '
                                            'some models add a front-bumper camera (not modelled here)',
                        counts={'cameras': 8, 'lidars': 0, 'radars': 0}, layout='estimated',
                        library_vehicle_id='tesla_model3_2024',
                        notes='Ranges as Tesla states them; only the wide camera\'s field is Tesla\'s own; the main and '
                              'narrow fields are a community measurement; the other fields are not published. Every '
                              'position and angle is estimated from Tesla\'s diagrams and placed on the template body; '
                              'drawn on the 2024 Model 3 body (the HW3 cars are the 2019–2023 Model 3).')


def _counts_only(rid, company, name, vehicle, year, src, counts, notes, cams=None, lidars=None):
    return ReferenceRig(rid, company, name, vehicle, year, src, cams or [], lidars or [], [], counts=counts,
                        notes=notes, layout='none')


def library() -> list[ReferenceRig]:
    """The reference rigs, measured and estimated layouts first, then those published as counts and roles only."""
    wod_cams = [RefSensor(n, 'camera', r, hfov_deg=50.4, resolution=res, prov={'hfov_deg': 'stated',
                                                                                 'resolution': 'stated'},
                          note='HFOV ±25.2° (Table 3); positions not published')
                for n, r, res in (('FRONT', 'forward', (1920, 1280)), ('FRONT_LEFT', 'front side', (1920, 1280)),
                                  ('FRONT_RIGHT', 'front side', (1920, 1280)), ('SIDE_LEFT', 'side', (1920, 1040)),
                                  ('SIDE_RIGHT', 'side', (1920, 1040)))]
    wod_lid = [RefSensor('TOP', 'lidar', 'mid-range 360°', vfov_deg=20.0, range_m=75.0,
                         prov={'vfov_deg': 'stated', 'range_m': 'stated'},
                         note='VFOV −17.6° to +2.4°; range as released (restricted)')] + \
        [RefSensor(n, 'lidar', 'short range', vfov_deg=120.0, range_m=20.0,
                   prov={'vfov_deg': 'stated', 'range_m': 'stated'}, note='VFOV −90° to +30°; range as released')
         for n in ('FRONT', 'SIDE_LEFT', 'SIDE_RIGHT', 'REAR')]
    return [
        _nuscenes(), _kitti(), _tesla_hw3(),
        _counts_only('waymo_open_dataset', 'Waymo', 'Waymo Open Dataset (perception) sensors', 'not stated in the paper',
                     '2019', [('Sun et al., CVPR 2020, Tables 2 and 3', 'https://arxiv.org/abs/1912.04838'),
                              ('Waymo Open Dataset: about', 'https://waymo.com/open/about/')],
                     {'cameras': 5, 'lidars': 5, 'radars': 0},
                     'Five pinhole cameras with rolling shutter and five lidars, as released; mounting positions are '
                     'in each segment\'s calibration, not in the paper.', wod_cams, wod_lid),
        _counts_only('waymo_5th_gen', 'Waymo', '5th-generation Waymo Driver', 'Jaguar I-PACE', '2020',
                     [('Introducing the 5th-generation Waymo Driver',
                       'https://waymo.com/blog/2020/03/introducing-5th-generation-waymo-driver/')],
                     {'cameras': 29, 'lidars': None, 'radars': 6},
                     '29 cameras around the body (long-range cameras see beyond 500 m), a 360° roof lidar with a range '
                     'beyond 300 m plus perimeter lidars at the corners (no lidar total in the blog), 6 radars.'),
        _counts_only('waymo_6th_gen', 'Waymo', '6th-generation Waymo Driver', 'Zeekr (and Hyundai IONIQ 5 later)',
                     '2024', [('Meet the 6th-generation Waymo Driver',
                               'https://waymo.com/blog/2024/08/meet-the-6th-generation-waymo-driver/')],
                     {'cameras': 13, 'lidars': 4, 'radars': 6},
                     '13 cameras, 4 lidars, 6 radars and external audio receivers, as Waymo states.'),
        _counts_only('argoverse2', 'Argo AI', 'Argoverse 2 sensor dataset rig', 'Ford Fusion Hybrid', '2021',
                     [('Argoverse 2 overview', 'https://www.argoverse.org/av2'),
                      ('Argoverse 2 sensor dataset', 'https://argoverse.github.io/user-guide/datasets/sensor.html')],
                     {'cameras': 9, 'lidars': 2, 'radars': 0},
                     '7 ring cameras 2048 × 1550 at 20 Hz (the front-centre one in portrait) and 2 forward stereo '
                     'cameras; 2 roof Velodyne VLP-32C lidars (10 Hz, 200 m); per-log calibration ships with the data.'),
        _counts_only('mobileye_supervision', 'Mobileye', 'SuperVision (Zeekr 001)', 'Zeekr 001', '2021',
                     [('Geely Holding Group expands Mobileye collaboration',
                       'https://www.mobileye.com/news/geely-holding-group-expands-mobileye-collaboration/')],
                     {'cameras': 11, 'lidars': 0, 'radars': None},
                     '11 cameras: 7 long-range 8 MP and 4 parking cameras, two EyeQ5; radar not listed in the release.'),
        _counts_only('wayve', 'Wayve', 'Wayve test vehicles', 'Jaguar I-PACE, Ford Mustang Mach-E', '2023–2025',
                     [('WIRED on Wayve\'s system', 'https://www.wired.com/story/wayve-ai-self-driving-system/')],
                     {'cameras': None, 'lidars': None, 'radars': None},
                     'Camera-first; the Jaguars carry small cameras above the front and rear windscreens and the Fords '
                     'a slim box of cameras and radar (press). Wayve publishes no camera count or layout.'),
    ]


def by_id() -> dict[str, ReferenceRig]:
    return {r.id: r for r in library()}


# --------------------------------------------------------------------------------------------------- variants
def _lens_candidates(project):
    """[(module, lens, hfov, width_px)] of every built-in camera module and lens of the project's library."""
    from .lens import model_fov
    out = []
    for ln in project.lenses.values():
        m = project.modules.get(ln.module_id)
        if m is None:
            continue
        try:
            h = float(model_fov(m, ln, circle=True)[0])
        except Exception:
            continue
        out.append((m, ln, h, m.width_px))
    return out


def closest_camera(project, ref: RefSensor):
    """(module id, lens id, hfov of that lens, note): the rig's own module when the library has it (its lens closest
    in field), else the library camera whose field is closest to the stated or estimated one (resolution as the
    tie-break); a camera whose field is not published gets the library's widest automotive lens as a stand-in."""
    cands = _lens_candidates(project)
    target = ref.hfov_deg
    if ref.library_model:
        own = [c for c in cands if c[0].id == ref.library_model]
        if own:
            m, ln, h, _ = min(own, key=lambda c: abs(c[2] - (target or c[2])))
            return m.id, ln.id, h, f'the rig\'s own {m.vendor} {m.name} with {ln.name} (H {h:.1f}°' + (
                f', the rig {target:.1f}°)' if target else ')')
    if target is None:
        auto = [c for c in cands if c[0].vendor == 'e-con Systems'] or cands
        m, ln, h, _ = max(auto, key=lambda c: (c[2] if c[2] < 200 else 0))
        return m.id, ln.id, h, f'field not published: {m.vendor} {m.name} with {ln.name} (H {h:.1f}°) as a stand-in'
    res = ref.resolution[0] if ref.resolution else None
    m, ln, h, _ = min(cands, key=lambda c: (abs(c[2] - target), abs(c[3] - res) if res else 0))
    return m.id, ln.id, h, f'closest library field: {m.vendor} {m.name} with {ln.name} (H {h:.1f}° against ' \
                           f'{target:.1f}°)'


def _placements(mesh, vehicle_id):
    """Estimated mount points on a template body: (point, direction to snap along) by name."""
    from .vehicles import by_id as vehicles, body_spec
    spec = body_spec(vehicles()[vehicle_id])
    L, W = spec.length, spec.width
    xr = -spec.rear_overhang
    xf = xr + L
    P = mesh.vertices
    glass = np.array([p.name == 'Glass' for p in mesh.parts])[mesh.face_part]
    gc = mesh.centroids[glass]
    ws = gc[(mesh.normals[glass][:, 0] > 0.3) & (np.abs(gc[:, 1]) < 0.3)]
    if len(ws):
        top = ws[np.argmax(ws[:, 2])]
        wsx, wsz = float(top[0]) - 0.03, float(top[2]) - 0.04
    else:
        wsx, wsz = xr + spec.glass_u[1] * L, spec.height - 0.1
    hw = W / 2
    return {'windscreen': ((wsx, 0.0, wsz), (1.0, 0.0, 0.6)),
            'b_pillar_left': ((xr + 0.50 * L, hw, spec.belt_z + 0.12), (0.0, 1.0, 0.0)),
            'b_pillar_right': ((xr + 0.50 * L, -hw, spec.belt_z + 0.12), (0.0, -1.0, 0.0)),
            # the side repeaters: on the front fenders, behind the front wheel arch
            'repeater_left': ((spec.wheelbase - spec.wheel_r - 0.35, hw, 0.80), (0.0, 1.0, 0.0)),
            'repeater_right': ((spec.wheelbase - spec.wheel_r - 0.35, -hw, 0.80), (0.0, -1.0, 0.0)),
            'rear': ((xr, 0.0, spec.plate_rear_z + 0.08), (-1.0, 0.0, 0.0))}


def to_variant(project, rig: ReferenceRig, name=None):
    """A Vantage Variant of a placed reference rig: each camera the closest library module and lens (closest_camera),
    each lidar and radar its library model; measured and stated poses kept exact, estimated ones snapped onto the
    project's body. Raises ValueError for a rig whose layout was never published."""
    from .project import Camera, Sensor, Variant
    from .samples import snap_point, link_pairs  # noqa: F401
    if rig.layout == 'none':
        raise ValueError(f'{rig.name}: its layout is not published (counts and roles only); Vantage does not invent '
                         'positions for it')
    mesh = project.vehicle.mesh
    places = _placements(mesh, project.vehicle.template) if any(
        isinstance(s.position_m, str) for s in rig.sensors) else {}
    v = Variant(name=name or f'{rig.company} · {rig.name}')
    lines = [f'Reference rig: {rig.name} ({rig.company}, {rig.vehicle}, {rig.year}). Layout: {rig.layout}.',
             'Sources: ' + '; '.join(u for _, u in rig.sources)]
    for i, c in enumerate(rig.cameras):
        mid, lid, h, why = closest_camera(project, c)
        pos = c.position_m
        if isinstance(pos, str):
            hint, d = places[pos]
            p, _n = snap_point(mesh, hint, d, 0.012)
            pos = (float(p[0]), float(p[1]), float(p[2]))
        cam = Camera(tag=f'C{i + 1}', name=c.name, location=c.role, slot=i % 8, x=round(pos[0], 4),
                     y=round(pos[1], 4), z=round(pos[2], 4), yaw=float(c.yaw or 0.0), pitch=float(c.pitch or 0.0),
                     roll=float(c.roll or 0.0), module_id=mid, lens_id=lid, port=i + 1 if i < 8 else 0,
                     snap_normal=False)
        v.cameras.append(cam)
        prov = ', '.join(f'{k} {p}' for k, p in sorted(c.prov.items()))
        lines.append(f'{cam.tag} {c.name}: {why}; {prov}' + (f'; unknown: {", ".join(c.missing())}'
                                                              if c.missing() else ''))
    k_l = k_r = 0
    for s in rig.lidars + rig.radars:
        if s.library_model:
            model = s.library_model
        else:
            model = 'ouster_os1_128' if s.kind == 'lidar' else 'conti_ars408_21'
        if s.kind == 'lidar':
            k_l += 1
            tag = f'L{k_l}'
        else:
            k_r += 1
            tag = f'RD{k_r}'
        x, y, z = s.position_m
        v.sensors.append(Sensor(tag=tag, name=s.name, kind=s.kind, model_id=model, x=float(x), y=float(y),
                                z=float(z), yaw=float(s.yaw or 0.0), pitch=float(s.pitch or 0.0),
                                roll=float(s.roll or 0.0)))
        lines.append(f'{tag} {s.name}: {model}' + (' (as the rig)' if s.library_model else ' (stand-in)') + '; ' +
                     ', '.join(f'{k} {p}' for k, p in sorted(s.prov.items())))
    lines.append(rig.notes)
    v.note = '\n'.join(lines)
    return v


def reference_project(rig_id):
    """A project on the rig's library vehicle (core.vehicles) with the reference variant added and active; the
    template's own camera layout stays as the first variant for comparison."""
    from . import samples
    from .vehicles import by_id as vehicles
    rig = by_id()[rig_id]
    vid = rig.library_vehicle_id
    fallback = ''
    if vid not in vehicles():
        vid, fallback = 'lincoln_mkz_2017', f' (no library vehicle for {rig.vehicle}: drawn on the Lincoln MKZ)'
    p = samples.template_project(vid)
    v = to_variant(p, rig)
    if fallback:
        v.note += fallback
    p.variants.append(v)
    p.active_uid = v.uid
    p.name = f'{rig.id}-reference'
    p.add_log('library', f'Reference rig {rig.name} ({rig.company}) on the {p.vehicle.name}{fallback}')
    return p


def summary_table() -> list[dict]:
    """One row per rig for the docs: counts, layout and provenance mix, the main sources."""
    out = []
    for r in library():
        out.append({'id': r.id, 'company': r.company, 'rig': r.name, 'vehicle': r.vehicle, 'year': r.year,
                    'cameras': r.counts.get('cameras'), 'lidars': r.counts.get('lidars'),
                    'radars': r.counts.get('radars'), 'layout': r.layout, 'provenance': r.provenance_mix(),
                    'library_vehicle': r.library_vehicle_id, 'sources': [u for _, u in r.sources]})
    return out
