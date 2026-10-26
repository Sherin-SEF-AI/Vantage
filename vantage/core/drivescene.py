"""A synthetic drive for the sample dataset: a seeded street scene, the ego vehicle's drive along it, and what every
sensor of the planned rig records of it (camera images through the real lens model, lidar sweeps fired column by
column, radar detections, IMU and GNSS output) together with the ground truth (boxes, point counts, visibility).

Frames
------
* World: x east, y north, z up, metres; the origin is the dataset's geodetic origin (core.dataset origin_lat/lon/alt),
  and the scene is laid out at positive x and y, where a nuScenes map's pixel grid starts.
* Ego (vehicle): the app's ISO 8855 frame, x forward, y left, z up, origin at the rear-axle centre on the ground. The
  road is flat and the body does not pitch or roll (no suspension), so the ego pose is (x, y, 0, yaw, 0, 0).

The street ('urban')
--------------------
One carriageway along a centreline made of a straight, a circular arc (a left or right bend, chosen by the seed) and a
straight. Across it (lateral offset d, positive to the left of the direction of travel): a 3.5 m lane each way with a
dashed centre line, a 1.5 m cycle lane behind a solid line, a 2.2 m parking strip, a 0.14 m kerb, a 3.5 m pavement of
slabs, a grass verge with trees, and buildings (boxes with windows) set back from 13 m. Street lights and traffic signs
(post and plate) stand on the pavement; a zebra crossing lies ahead of where the drive ends. 'highway' is a dual
carriageway (three lanes each way, a median barrier, hard shoulders, noise walls) with faster traffic.
Moving actors follow lanes at constant speeds: cars and a truck ahead, behind and oncoming, cyclists in the cycle
lanes, pedestrians walking or standing on the pavements and one crossing at the zebra; their start positions are
shifted until no two of them (or the ego) meet. Parked cars fill the parking strips.

The ego drive
-------------
The ego follows the centre of its lane (right-hand traffic) with clothoid transitions into and out of the bend
(curvature ramps over RAMP_M), starting at 80 % of the set speed and reaching it with a smooth (cubic) acceleration
over the first ACCEL_S seconds. Its pose, velocity, acceleration, yaw rate and yaw acceleration are analytic.

Sensors
-------
* Cameras: every pixel's ray (the real lens model, unit calibrations included, or an exact pinhole for rectified
  images) is cast into the scene from the camera's position at the pixel's exposure time: the frame's middle-row time
  (global), or with a rolling shutter each band of 16 rows at its own time (rows read top to bottom over the readout,
  each exposed for the exposure time). The ego body, its rig and the housings of the other sensors and cameras hide
  what lies behind them (the same cube depth maps as the analysis; core.housings); housings are drawn as dark plastic
  and metal. Shading: a fixed sun (Lambert), ground shadows, painted markings antialiased on the pixel footprint,
  a little sensor noise. Rendering culls objects per 16 × 16 pixel tile.
* Lidar: each column of channels fires at its own time (a spinning lidar's head faces phase ∓ 360°·f·t after each
  sync pulse, as core.timing) from the sensor's position at that time, so a sweep is not motion compensated, as a
  driver delivers it. Rays meet the ground, the objects and the own body, rig and the other sensors' and cameras'
  housings (self hits, flagged). A return needs range_min ≤ r ≤ R_max(ρ·cos i) with R_max(ρ) = range_m·√(ρ/0.10)
  (capped at range_max_m when the model gives one), ρ the material's typical reflectivity and i the incidence angle;
  Gaussian range noise; intensity = 255·ρ·cos i.
* Radar: one detection per target object (cars, trucks, pedestrians, cyclists, poles and sign posts) whose centre is
  in a zone at the range its cross-section allows (sensors.in_field with rcs), with nothing of the body beyond the
  fascia allowance nor a nearer object on the line of sight; the reflection point is the object's nearest point;
  range, azimuth, elevation, Doppler and RCS noise; a few clutter detections from building faces.
* IMU: specific force at the IMU's position (lever arm: centripetal and Euler terms) minus gravity, and angular rate,
  in the IMU's own axes, with white noise and a small constant bias (IMU_NOISE). GNSS: each antenna's phase centre as
  WGS84 latitude, longitude and ellipsoidal height, with white RTK-like noise.

Ground truth: per actor and time its box (nuScenes wlh), yaw and velocity in the world frame; lidar and radar points
counted by the object each hit; nuScenes visibility from rays between the cameras and points on the box's faces that
face the ego (the share visible in any camera, past nearer objects and the body).

Weather (core.weather; the simulators take a SceneWeather, None for clear, which runs exactly as without it)
* Cameras: by day an overcast sky drops the sun's shading and shadows, rain darkens the road; at night every surface
  is lit by the ambient light and the ego's two headlamps (their typical beam pattern, inverse square, the cosine on
  the surface, shadows cast by the scene's objects), its luminance mapped through the camera's photon budget (signal
  in electrons with shot and read noise; auto-exposure puts the frame's 99th percentile at full scale with the gain
  capped at NIGHT_FULL_SCALE × the camera's minimum luminance; highlights compressed as 1 − exp(−1.2x), gamma 2.2).
  Fog, rain and snow blend each pixel toward the airlight with the transmission exp(−σd) over its ray's length. Drops
  sit on the lens (their share of the image from core.weather's heuristic, blurred, drifting slowly down) and flakes
  fall in snow.
* Lidar: a return needs r ≤ the weather's derated R_max(ρ) (two-way extinction, Lambert W), its intensity falls by
  exp(−2αr), and a share of the beams return early from droplets or flakes (object id P_WEATHER).
* Radar: a target's cross-section counts reduced by the two-way rain and fog loss at its distance (so its range
  shrinks as the radar equation says) and its reported RCS is lower by that loss.

Modelling limits: synthetic geometry (boxes, cylinders, spheres), simple materials, one fixed sun, no weather, no
multipath or beam divergence, one lidar return per beam, one radar detection per object, flat road, no suspension
motion, actors at constant speeds that do not react to each other. The scene is for testing a pipeline (ingestion,
labelling tools, loaders), not for training.
"""
from __future__ import annotations

import math
import zlib
from dataclasses import dataclass, replace

import numpy as np

from . import sensors as SN
from .lens import resolve
from .mathutil import BODY_TO_OPTICAL, rot_zyx
from .synth import NEAR_CLIP, pixel_rays

G = 9.80665                          # m/s², standard gravity
SUN = np.array([0.32, 0.42, 0.85]) / np.linalg.norm([0.32, 0.42, 0.85])
EPS = 1e-4
RAMP_M = 12.0                        # clothoid transition into and out of the bend, metres of path
ACCEL_S = 3.0                        # the ego reaches the set speed after this long
CAM_CULL_M = 160.0                   # objects farther than this from a camera are not drawn
TILE = 16                            # pixels per tile side (culling) and rows per rolling-shutter band
OWN_MOUNT_M = 0.05                   # lidar: body hits this close to the optical centre are its own mount
RADAR_FASCIA_M = 0.15                # radar: body hits this close are the bumper fascia it sees through
ANNOT_RANGE_M = 60.0                 # actors annotated within this horizontal distance of the ego
TARGET_RCS = {'car': 10.0, 'truck': 20.0, 'pedestrian': 0.5, 'cyclist': 1.0, 'sign': 1.0, 'lamp': 1.0}
# typical figures, not a product's datasheet: per-sample white noise (1 σ) and the spread of the constant bias
IMU_NOISE = {'accel_sigma': 0.02, 'gyro_sigma': 5e-4, 'accel_bias': 0.01, 'gyro_bias': 5e-5}
RADAR_NOISE = {'range_m': 0.10, 'azimuth_deg': 0.25, 'elevation_deg': 0.5, 'velocity_mps': 0.10, 'rcs_db': 1.0}

# materials (what a hit is shaded and reflects like)
M_SKY, M_GROUND, M_CAR, M_BUILDING, M_PLAIN, M_PERSON, M_SIGN, M_BODY = range(8)
P_GROUND, P_BODY, P_SKY = -3, -2, -1          # prim index codes in the buffers
P_WEATHER = -4                                 # a lidar return from fog droplets, rain or snow (core.weather)
NIGHT_FULL_SCALE = 12.0                        # night: full scale at this many times the camera's minimum luminance
BOX, CYL, SPH = 0, 1, 2
# typical lidar reflectivity at 850-905 nm (assumptions)
REFL = {'asphalt': 0.08, 'paint': 0.55, 'cycle': 0.10, 'pavement': 0.25, 'grass': 0.45, 'kerb': 0.30,
        'facade': 0.30, 'window': 0.06, 'roof': 0.15, 'car': 0.30, 'car_glass': 0.05, 'tyre': 0.04, 'person': 0.25,
        'pole': 0.35, 'bark': 0.20, 'leaves': 0.50, 'sign': 0.90, 'sign_back': 0.30, 'body': 0.30, 'barrier': 0.40}

# cross-section of the street (lateral offsets, metres, from the centreline)
URBAN = {'lanes': 1, 'lane_w': 3.5, 'cycle_w': 1.5, 'park_w': 2.2, 'kerb_w': 0.25, 'pave_w': 3.5, 'verge_w': 1.6,
         'build_d': 13.0, 'median_w': 0.0}
HIGHWAY = {'lanes': 3, 'lane_w': 3.6, 'cycle_w': 0.0, 'park_w': 3.0, 'kerb_w': 0.0, 'pave_w': 0.0, 'verge_w': 6.0,
           'build_d': 0.0, 'median_w': 1.2}
KERB_H = 0.14


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


# =========================================================================== the road
class Road:
    """Centreline: a straight of length L1 from (x0, y0) on heading psi0, an arc of radius R turning `turn` (+1 left,
    -1 right) through theta radians, then a straight on. s is the arc length along it (negative before the start, the
    straights extend for ever); d the lateral offset, positive to the left."""

    def __init__(self, x0, y0, psi0, L1, R, turn, theta, profile):
        self.x0, self.y0, self.psi0 = float(x0), float(y0), float(psi0)
        self.L1, self.R, self.turn, self.theta = float(L1), float(R), int(turn), float(theta)
        self.p = dict(profile)
        c, s = math.cos(psi0), math.sin(psi0)
        self.P1 = np.array([x0 + L1 * c, y0 + L1 * s])
        self.C = self.P1 + self.turn * R * np.array([-s, c])
        self.Larc = R * theta
        self.psi2 = psi0 + self.turn * theta
        self.P2 = self.C + self.turn * R * np.array([math.sin(self.psi2), -math.cos(self.psi2)])
        lw = self.p['lane_w'] * self.p['lanes']
        m = self.p['median_w'] / 2
        self.e_lanes = m + lw                                   # outer edge of the driving lanes
        self.e_cycle = self.e_lanes + self.p['cycle_w']
        self.e_kerb = self.e_cycle + self.p['park_w']           # road edge (kerb face, or the shoulder's edge)
        self.e_pave = self.e_kerb + self.p['kerb_w'] + self.p['pave_w']
        self.e_verge = self.e_pave + self.p['verge_w']

    def frame(self, s):
        """(x, y, heading) of the centreline at arc length s."""
        s = np.asarray(s, float)
        L1, La = self.L1, self.Larc
        c0, s0 = math.cos(self.psi0), math.sin(self.psi0)
        psi_a = self.psi0 + self.turn * (s - L1) / self.R
        xa = self.C[0] + self.turn * self.R * np.sin(psi_a)
        ya = self.C[1] - self.turn * self.R * np.cos(psi_a)
        c2, s2 = math.cos(self.psi2), math.sin(self.psi2)
        x = np.where(s <= L1, self.x0 + s * c0, np.where(s <= L1 + La, xa, self.P2[0] + (s - L1 - La) * c2))
        y = np.where(s <= L1, self.y0 + s * s0, np.where(s <= L1 + La, ya, self.P2[1] + (s - L1 - La) * s2))
        psi = np.where(s <= L1, self.psi0, np.where(s <= L1 + La, psi_a, self.psi2))
        return x, y, psi

    def to_world(self, s, d):
        x, y, psi = self.frame(s)
        d = np.asarray(d, float)
        return x - d * np.sin(psi), y + d * np.cos(psi), psi

    def to_sd(self, x, y):
        """(s, d) of world points: the nearest point of the centreline's piece the point lies beside."""
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        vx, vy = x - self.C[0], y - self.C[1]
        psi = np.arctan2(vy, vx) + self.turn * math.pi / 2
        u = _wrap(self.turn * (psi - self.psi0))
        r = np.hypot(vx, vy)
        c0, s0 = math.cos(self.psi0), math.sin(self.psi0)
        c2, s2 = math.cos(self.psi2), math.sin(self.psi2)
        s1 = (x - self.x0) * c0 + (y - self.y0) * s0
        d1 = -(x - self.x0) * s0 + (y - self.y0) * c0
        s3 = self.L1 + self.Larc + (x - self.P2[0]) * c2 + (y - self.P2[1]) * s2
        d3 = -(x - self.P2[0]) * s2 + (y - self.P2[1]) * c2
        sa = self.L1 + u * self.R
        da = self.turn * (self.R - r)
        s = np.where(u < 0, s1, np.where(u > self.theta, s3, sa))
        d = np.where(u < 0, d1, np.where(u > self.theta, d3, da))
        return s, d

    def kappa(self, s):
        s = np.asarray(s, float)
        return np.where((s > self.L1) & (s < self.L1 + self.Larc), self.turn / self.R, 0.0)


# =========================================================================== the ego drive
class EgoDrive:
    """The ego's lane centre (offset d from the road centreline) with clothoid transitions, and its speed profile."""

    def __init__(self, road: Road, speed_kmh: float, duration: float, d: float, ramp: float = RAMP_M):
        self.road, self.d = road, float(d)
        self.v1 = max(float(speed_kmh), 0.0) / 3.6
        self.v0 = 0.8 * self.v1
        self.Ta = max(min(ACCEL_S, 0.4 * max(duration, 0.1)), 0.05)
        Re = road.R - road.turn * self.d                   # the lane centre's radius in the bend
        ke = road.turn / Re
        a, b = road.L1, road.L1 + road.theta * Re
        ramp = max(min(ramp, 0.9 * (b - a), 0.9 * 2 * a), 1e-6)       # a bend from s = 0 starts at once
        self.knots_s = np.array([-1e5, a - ramp / 2, a + ramp / 2, b - ramp / 2, b + ramp / 2, 1e7])
        self.knots_k = np.array([0.0, 0.0, ke, ke, 0.0, 0.0])
        ds = np.diff(self.knots_s)
        area = 0.5 * (self.knots_k[:-1] + self.knots_k[1:]) * ds
        self.knots_a = np.r_[0.0, np.cumsum(area)]
        self.knots_a -= self._area_raw(0.0)                 # heading measured from the start (path s = 0)
        # positions on a fine grid (Simpson), start of the path at road s = 0, lane offset d
        x0, y0, _ = road.to_world(0.0, self.d)
        self.p0 = np.array([float(x0), float(y0)])
        self.h = 0.05
        self.g0 = -60.0
        n = int((self.travel(duration) + 160.0 - self.g0) / self.h) + 2
        g = self.g0 + np.arange(n) * self.h
        psi_a, psi_m, psi_b = self.psi(g[:-1]), self.psi(g[:-1] + self.h / 2), self.psi(g[1:])
        dx = self.h / 6 * (np.cos(psi_a) + 4 * np.cos(psi_m) + np.cos(psi_b))
        dy = self.h / 6 * (np.sin(psi_a) + 4 * np.sin(psi_m) + np.sin(psi_b))
        gx = np.r_[0.0, np.cumsum(dx)]
        gy = np.r_[0.0, np.cumsum(dy)]
        i0 = int(round(-self.g0 / self.h))                  # grid index of path s = 0
        self.gx = gx - gx[i0] + self.p0[0]
        self.gy = gy - gy[i0] + self.p0[1]
        self.n = n

    # ---- path geometry (path arc length p)
    def _area_raw(self, p):
        S, K = self.knots_s, self.knots_k
        p = np.asarray(p, float)
        i = np.clip(np.searchsorted(S, p, 'right') - 1, 0, len(S) - 2)
        ds = np.diff(S)
        area = np.r_[0.0, np.cumsum(0.5 * (K[:-1] + K[1:]) * ds)]
        u = p - S[i]
        slope = (K[i + 1] - K[i]) / ds[i]
        return area[i] + K[i] * u + 0.5 * slope * u * u

    def kappa(self, p):
        return np.interp(np.asarray(p, float), self.knots_s, self.knots_k)

    def dkappa(self, p):
        """dκ/ds."""
        S, K = self.knots_s, self.knots_k
        p = np.asarray(p, float)
        i = np.clip(np.searchsorted(S, p, 'right') - 1, 0, len(S) - 2)
        return (K[i + 1] - K[i]) / np.diff(S)[i]

    def psi(self, p):
        p = np.asarray(p, float)
        S, K = self.knots_s, self.knots_k
        i = np.clip(np.searchsorted(S, p, 'right') - 1, 0, len(S) - 2)
        u = p - S[i]
        slope = (K[i + 1] - K[i]) / np.diff(S)[i]
        return self.road.psi0 + self.knots_a[i] + K[i] * u + 0.5 * slope * u * u

    def xy(self, p):
        p = np.asarray(p, float)
        k = np.clip(((p - self.g0) / self.h).astype(int), 0, self.n - 2)
        a = self.g0 + k * self.h
        u = p - a
        pa, pm, pb = self.psi(a), self.psi(a + u / 2), self.psi(p)
        x = self.gx[k] + u / 6 * (np.cos(pa) + 4 * np.cos(pm) + np.cos(pb))
        y = self.gy[k] + u / 6 * (np.sin(pa) + 4 * np.sin(pm) + np.sin(pb))
        return x, y

    # ---- speed profile
    def speed(self, t):
        t = np.asarray(t, float)
        u = np.clip(t / self.Ta, 0.0, 1.0)
        return self.v0 + (self.v1 - self.v0) * (3 * u * u - 2 * u ** 3)

    def accel(self, t):
        t = np.asarray(t, float)
        u = t / self.Ta
        return np.where((u > 0) & (u < 1), (self.v1 - self.v0) / self.Ta * 6 * u * (1 - u), 0.0)

    def travel(self, t):
        """Path arc length driven by time t (negative before 0: at the start speed)."""
        t = np.asarray(t, float)
        u = np.clip(t / self.Ta, 0.0, 1.0)
        acc = self.v0 * np.minimum(np.maximum(t, 0), self.Ta) + (self.v1 - self.v0) * self.Ta * (u ** 3 - u ** 4 / 2)
        return np.where(t < 0, self.v0 * t, acc + self.v1 * np.maximum(t - self.Ta, 0.0))

    # ---- the pose and its derivatives
    def pose(self, t):
        """(x, y, yaw) of the ego frame in the world at time(s) t."""
        p = self.travel(t)
        x, y = self.xy(p)
        return x, y, self.psi(p)

    def kinematics(self, t) -> dict:
        """Speed v, longitudinal acceleration a, curvature κ, yaw rate ω = vκ and yaw acceleration α = aκ + v²κ'
        (κ' = dκ/ds) at time(s) t; the rear-axle centre moves along the vehicle's x (no side slip)."""
        p = self.travel(t)
        v, a = self.speed(t), self.accel(t)
        k, dk = self.kappa(p), self.dkappa(p)
        return {'v': v, 'a': a, 'kappa': k, 'omega': v * k, 'alpha': a * k + v * v * dk, 'p': p}

    def velocity_world(self, t):
        x, y, psi = self.pose(t)
        v = self.speed(t)
        return np.stack([v * np.cos(psi), v * np.sin(psi), np.zeros_like(psi)], -1)

    def R(self, t) -> np.ndarray:
        _, _, psi = self.pose(t)
        return _rz(float(psi))


def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


# =========================================================================== objects
@dataclass
class Motion:
    """Along the road: s(t) = s0 + vs·t, d(t) = d0 + vd·t (d held within d_lim)."""
    s0: float
    d0: float
    vs: float = 0.0
    vd: float = 0.0
    d_lim: tuple | None = None
    yaw_off: float = 0.0             # extra heading for a standing actor (radians)

    def sd(self, t):
        t = np.asarray(t, float)
        s = self.s0 + self.vs * t
        d = self.d0 + self.vd * t
        if self.d_lim is not None:
            d = np.clip(d, *self.d_lim)
        return s, d


@dataclass
class Obj:
    """A scene object: its parts (prims, in the object's own frame: x along its length, y across, z up from the
    ground), the box around it (length, width, height above `base`), its pose (static) or motion (along the road)."""
    id: int
    kind: str                        # car truck pedestrian cyclist building tree lamp sign kerb barrier wall
    size: tuple                      # (length, width, height)
    prims: list                      # [(shape, mat, rgb, refl, geometry)] local
    pose: tuple = (0.0, 0.0, 0.0)    # static: (x, y, yaw) world
    motion: Motion | None = None
    base: float = 0.0
    category: str = ''               # nuScenes category ('' = not annotated)
    rcs: float = 0.0                 # radar cross-section, m² (0: not a radar target)
    name: str = ''

    @property
    def moving(self):
        return self.motion is not None


class Scene:
    """A road, the ego drive and the objects; prims_at(t) gives every object's parts in the world at time t."""

    def __init__(self, road: Road, drive: EgoDrive, objects: list, kind: str = 'urban', seed: int = 0,
                 extent=None, crossing_s=None):
        self.road, self.drive, self.objects, self.kind, self.seed = road, drive, list(objects), kind, seed
        self.extent = extent                     # (xmin, ymin, xmax, ymax) world, for the map
        self.crossing_s = crossing_s
        self.by_id = {o.id: o for o in self.objects}
        stat = [o for o in self.objects if not o.moving]
        self._static = _PrimSet.build([(o, o.pose) for o in stat], {o.id: (0.0, 0.0) for o in stat})
        self._moving = [o for o in self.objects if o.moving]

    # ---- actors
    def actor_pose(self, o: Obj, t):
        """(x, y, yaw) of an object at time(s) t."""
        if not o.moving:
            x, y, yaw = o.pose
            return np.asarray(x, float), np.asarray(y, float), np.asarray(yaw, float)
        m = o.motion
        s, d = m.sd(t)
        x, y, psi = self.road.to_world(s, d)
        if m.vs < 0:
            yaw = psi + math.pi
        elif m.vs == 0 and m.vd != 0:
            yaw = psi + math.copysign(math.pi / 2, m.vd)
        else:
            yaw = psi + m.yaw_off
        return x, y, yaw

    def actor_velocity(self, o: Obj, t, h=1e-3):
        if not o.moving:
            return np.zeros(3)
        xa, ya, _ = self.actor_pose(o, t - h)
        xb, yb, _ = self.actor_pose(o, t + h)
        return np.array([(float(xb) - float(xa)) / (2 * h), (float(yb) - float(ya)) / (2 * h), 0.0])

    def prims_at(self, t) -> '_PrimSet':
        if not self._moving:
            return self._static
        poses, vel = [], {}
        for o in self._moving:
            x, y, yaw = self.actor_pose(o, t)
            poses.append((o, (float(x), float(y), float(yaw))))
            v = self.actor_velocity(o, t)
            vel[o.id] = (float(v[0]), float(v[1]))
        return _PrimSet.concat(self._static, _PrimSet.build(poses, vel))

    def boxes_at(self, t, near=None, rng_m=ANNOT_RANGE_M) -> list:
        """Ground truth of every annotated object at time t (within rng_m of the point `near` (x, y) when given):
        id, category, centre (world), size (length, width, height), yaw, velocity (world)."""
        out = []
        for o in self.objects:
            if not o.category:
                continue
            x, y, yaw = (float(v) for v in self.actor_pose(o, t))
            if near is not None and math.hypot(x - near[0], y - near[1]) > rng_m:
                continue
            L, W, H = o.size
            out.append({'id': o.id, 'kind': o.kind, 'category': o.category, 'name': o.name,
                        'centre': (x, y, o.base + H / 2), 'size': (L, W, H), 'yaw': yaw,
                        'velocity': tuple(float(v) for v in self.actor_velocity(o, t)), 'moving': o.moving})
        return out


class _PrimSet:
    """Every part of the scene's objects at one instant, as arrays."""
    FIELDS = ('shape', 'obj', 'mat', 'rgb', 'refl', 'cx', 'cy', 'cz', 'yaw', 'L', 'W', 'z0', 'z1', 'r', 'vx', 'vy',
              'bc', 'br')

    @staticmethod
    def build(items, vel) -> '_PrimSet':
        rows = []
        for o, (x, y, yaw) in items:
            c, s = math.cos(yaw), math.sin(yaw)
            vx, vy = vel.get(o.id, (0.0, 0.0))
            for shape, mat, rgb, refl, g in o.prims:
                if shape == BOX:
                    ox, oy, oyaw, L, W, z0, z1 = g
                    cx, cy = x + ox * c - oy * s, y + ox * s + oy * c
                    bc = (cx, cy, (z0 + z1) / 2)
                    br = 0.5 * math.sqrt(L * L + W * W + (z1 - z0) ** 2)
                    rows.append((BOX, o.id, mat, rgb, refl, cx, cy, 0.0, yaw + oyaw, L, W, z0, z1, 0.0, vx, vy, bc, br))
                elif shape == CYL:
                    ox, oy, r, z0, z1 = g
                    cx, cy = x + ox * c - oy * s, y + ox * s + oy * c
                    bc = (cx, cy, (z0 + z1) / 2)
                    br = math.sqrt(r * r + ((z1 - z0) / 2) ** 2)
                    rows.append((CYL, o.id, mat, rgb, refl, cx, cy, 0.0, 0.0, 0.0, 0.0, z0, z1, r, vx, vy, bc, br))
                else:
                    ox, oy, oz, r = g
                    cx, cy = x + ox * c - oy * s, y + ox * s + oy * c
                    rows.append((SPH, o.id, mat, rgb, refl, cx, cy, oz, 0.0, 0.0, 0.0, oz - r, oz + r, r, vx, vy,
                                 (cx, cy, oz), r))
        ps = _PrimSet()
        n = len(rows)
        cols = list(zip(*rows)) if rows else [()] * len(_PrimSet.FIELDS)
        for k, f in enumerate(_PrimSet.FIELDS):
            if f == 'rgb':
                ps.rgb = np.array(cols[k], float).reshape(n, 3)
            elif f == 'bc':
                ps.bc = np.array(cols[k], float).reshape(n, 3)
            elif f in ('shape', 'obj', 'mat'):
                setattr(ps, f, np.array(cols[k], int).reshape(n))
            else:
                setattr(ps, f, np.array(cols[k], float).reshape(n))
        ps.n = n
        return ps

    @staticmethod
    def concat(a: '_PrimSet', b: '_PrimSet') -> '_PrimSet':
        ps = _PrimSet()
        for f in _PrimSet.FIELDS:
            setattr(ps, f, np.concatenate([getattr(a, f), getattr(b, f)]))
        ps.n = a.n + b.n
        return ps

    def moving(self, i) -> bool:
        return self.vx[i] != 0.0 or self.vy[i] != 0.0

    def footprint(self, i):
        """Corners (4, 2) of a box's plan, or None for round parts."""
        if self.shape[i] != BOX:
            return None
        c, s = math.cos(self.yaw[i]), math.sin(self.yaw[i])
        hl, hw = self.L[i] / 2, self.W[i] / 2
        q = np.array([[hl, hw], [-hl, hw], [-hl, -hw], [hl, -hw]])
        return np.c_[self.cx[i] + q[:, 0] * c - q[:, 1] * s, self.cy[i] + q[:, 0] * s + q[:, 1] * c]

    def hit(self, i, O, D):
        """Distance along unit rays D (n, 3) from O ((3,) or (n, 3)) to part i; inf for a miss."""
        k = self.shape[i]
        if k == BOX:
            return _ray_box(O, D, self.cx[i], self.cy[i], self.yaw[i], self.L[i], self.W[i], self.z0[i], self.z1[i])
        if k == CYL:
            return _ray_cyl(O, D, self.cx[i], self.cy[i], self.r[i], self.z0[i], self.z1[i])
        return _ray_sph(O, D, self.cx[i], self.cy[i], self.cz[i], self.r[i])

    def normals(self, idx, P):
        """Outward unit normals (n, 3) of parts idx (n,) at points P (n, 3) on their surfaces."""
        idx = np.asarray(idx, int)
        out = np.zeros((len(idx), 3))
        sh = self.shape[idx]
        m = sh == BOX
        if m.any():
            ii = idx[m]
            c, s = np.cos(self.yaw[ii]), np.sin(self.yaw[ii])
            dx, dy = P[m, 0] - self.cx[ii], P[m, 1] - self.cy[ii]
            loc = np.stack([dx * c + dy * s, -dx * s + dy * c, P[m, 2] - (self.z0[ii] + self.z1[ii]) / 2], 1)
            half = np.stack([self.L[ii], self.W[ii], self.z1[ii] - self.z0[ii]], 1) / 2
            ax = np.argmax(np.abs(loc) / np.maximum(half, 1e-9), axis=1)
            k = np.arange(len(ii))
            nl = np.zeros((len(ii), 3))
            nl[k, ax] = np.sign(loc[k, ax])
            out[m] = np.stack([nl[:, 0] * c - nl[:, 1] * s, nl[:, 0] * s + nl[:, 1] * c, nl[:, 2]], 1)
        m = sh == CYL
        if m.any():
            ii = idx[m]
            dx, dy, z = P[m, 0] - self.cx[ii], P[m, 1] - self.cy[ii], P[m, 2]
            rr = np.maximum(np.hypot(dx, dy), 1e-9)
            dside = np.abs(rr - self.r[ii])
            dtop, dbot = np.abs(z - self.z1[ii]), np.abs(z - self.z0[ii])
            side = (dside <= dtop) & (dside <= dbot)
            nn = np.where((dtop < dbot)[:, None], [0.0, 0.0, 1.0], [0.0, 0.0, -1.0])
            nn[side] = np.c_[dx[side] / rr[side], dy[side] / rr[side], np.zeros(side.sum())]
            out[m] = nn
        m = sh == SPH
        if m.any():
            ii = idx[m]
            out[m] = (P[m] - np.c_[self.cx[ii], self.cy[ii], self.cz[ii]]) / np.maximum(self.r[ii], 1e-9)[:, None]
        return out


# =========================================================================== ray casting
def _ray_box(O, D, cx, cy, yaw, L, W, z0, z1):
    """Entry distance of rays into a yawed box (slabs; a ray along a face plane counts as a miss)."""
    c, s = math.cos(yaw), math.sin(yaw)
    O = np.asarray(O, float)
    ox = (O[..., 0] - cx) * c + (O[..., 1] - cy) * s
    oy = -(O[..., 0] - cx) * s + (O[..., 1] - cy) * c
    oz = O[..., 2]
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        ix = 1.0 / (D[:, 0] * c + D[:, 1] * s)
        t1, t2 = (-L / 2 - ox) * ix, (L / 2 - ox) * ix
        tmin, tmax = np.minimum(t1, t2), np.maximum(t1, t2)
        iy = 1.0 / (-D[:, 0] * s + D[:, 1] * c)
        t1, t2 = (-W / 2 - oy) * iy, (W / 2 - oy) * iy
        tmin = np.maximum(tmin, np.minimum(t1, t2))
        tmax = np.minimum(tmax, np.maximum(t1, t2))
        iz = 1.0 / D[:, 2]
        t1, t2 = (z0 - oz) * iz, (z1 - oz) * iz
        tmin = np.maximum(tmin, np.minimum(t1, t2))
        tmax = np.minimum(tmax, np.maximum(t1, t2))
        return np.where((tmax >= tmin) & (tmin > EPS), tmin, np.inf)


def _ray_cyl(O, D, cx, cy, r, z0, z1):
    """Entry distance of rays into a vertical cylinder (side, top and bottom caps)."""
    O = np.asarray(O, float)
    ox, oy, oz = O[..., 0] - cx, O[..., 1] - cy, O[..., 2]
    dx, dy, dz = D[:, 0], D[:, 1], D[:, 2]
    a = dx * dx + dy * dy
    b = 2 * (ox * dx + oy * dy)
    c = ox * ox + oy * oy - r * r
    disc = b * b - 4 * a * c
    with np.errstate(divide='ignore', invalid='ignore'):
        ts = (-b - np.sqrt(np.maximum(disc, 0))) / (2 * np.maximum(a, 1e-12))
        zs = oz + ts * dz
        t = np.where((disc > 0) & (a > 1e-12) & (ts > EPS) & (zs >= z0) & (zs <= z1), ts, np.inf)
        for zc, ok in ((z1, (oz > z1) & (dz < 0)), (z0, (oz < z0) & (dz > 0))):
            if not np.any(ok):
                continue
            tc = (zc - oz) / dz
            inside = (ox + tc * dx) ** 2 + (oy + tc * dy) ** 2 <= r * r
            t = np.minimum(t, np.where(ok & inside & (tc > EPS), tc, np.inf))
    return t


def _ray_sph(O, D, cx, cy, cz, r):
    O = np.asarray(O, float)
    oc = O - np.array([cx, cy, cz])
    b = np.sum(oc * D, axis=-1) if oc.ndim > 1 else D @ oc
    c = (np.sum(oc * oc, axis=-1) if oc.ndim > 1 else float(oc @ oc)) - r * r
    disc = b * b - c
    t = -b - np.sqrt(np.maximum(disc, 0.0))
    return np.where((disc > 0) & (t > EPS), t, np.inf)


def _ranges(starts, lens):
    """Indices of the concatenated ranges [start, start + len)."""
    lens = np.asarray(lens, int)
    total = int(lens.sum())
    if total == 0:
        return np.zeros(0, int)
    before = np.cumsum(lens) - lens
    return np.arange(total) + np.repeat(np.asarray(starts, int) - before, lens)


# =========================================================================== ground and materials
def ground_material(road: Road, s, d, fw=None, crossing_s=None):
    """(rgb (n, 3) in 0..1, lidar reflectivity (n,)) of ground points at road coordinates (s, d); fw is each point's
    footprint (metres) for antialiasing the paint (None: hard edges, for the lidar)."""
    p = road.p
    n = len(s)
    a = np.abs(d)
    f = np.full(n, 1e-3, d.dtype) if fw is None else np.maximum(fw, 1e-3)
    rgb = np.empty((n, 3), d.dtype)
    refl = np.empty(n, d.dtype)
    mott = 0.012 * np.sin(s * 2.3 + d * 1.9) + 0.008 * np.sin(s * 0.7 - d * 1.1)
    asphalt = 0.33 + mott
    rgb[:] = asphalt[:, None] * np.array([1.0, 1.0, 1.03])
    refl[:] = REFL['asphalt']
    paint = np.zeros(n)

    def line(centre, width, dash=0.0, on=0.0):
        c = np.clip((width / 2 - np.abs(a - centre)) / f + 0.5, 0, 1)
        if dash:
            ph = (s % dash)
            c = c * np.clip((on - ph) / f + 0.5, 0, 1) * np.clip(ph / f + 0.5, 0, 1)
        return c
    m = p['median_w'] / 2
    if p['median_w'] > 0:                                        # highway: lane lines, solid edge lines
        for k in range(1, p['lanes']):
            paint = np.maximum(paint, line(m + k * p['lane_w'], 0.15, 12.0, 4.0))
        paint = np.maximum(paint, line(m + 0.2, 0.2))
        paint = np.maximum(paint, line(road.e_lanes - 0.1, 0.2))
    else:
        paint = np.maximum(paint, np.clip((0.06 - a) / f + 0.5, 0, 1) *
                           np.clip((3.0 - s % 9.0) / f + 0.5, 0, 1) * np.clip((s % 9.0) / f + 0.5, 0, 1))
        if p['cycle_w'] > 0:
            paint = np.maximum(paint, line(road.e_lanes, 0.15))
            cyc = (a > road.e_lanes + 0.075) & (a < road.e_cycle)
            rgb[cyc] = np.c_[0.42 + mott[cyc], 0.22 + mott[cyc], 0.20 + mott[cyc]]
            refl[cyc] = REFL['cycle']
        if crossing_s is not None:                               # zebra: stripes along the road, 0.5 m on 0.5 m off
            on = (np.abs(s - crossing_s) < 2.0) & (a < road.e_kerb)
            zc = np.clip((0.25 - np.abs((d % 1.0) - 0.5)) / f + 0.5, 0, 1)
            zc = zc * np.clip((2.0 - np.abs(s - crossing_s)) / f + 0.5, 0, 1)
            paint = np.maximum(paint, np.where(on, zc, 0.0))
    kerb = (a >= road.e_kerb) & (a < road.e_kerb + p['kerb_w'])
    rgb[kerb] = 0.62
    refl[kerb] = REFL['kerb']
    if p['pave_w'] > 0:                                          # slabs with joints every metre
        pv = (a >= road.e_kerb + p['kerb_w']) & (a < road.e_pave)
        js = np.clip((0.02 - np.abs((s[pv] % 1.0) - 0.5) + 0.48) / f[pv] + 0.5, 0, 1)
        jd = np.clip((0.02 - np.abs(((a[pv] - road.e_kerb) % 1.0) - 0.5) + 0.48) / f[pv] + 0.5, 0, 1)
        g = 0.56 + 0.5 * mott[pv] - 0.12 * np.maximum(js, jd)
        rgb[pv] = np.c_[g, g * 0.99, g * 0.96]
        refl[pv] = REFL['pavement']
    out = a >= road.e_pave
    if p['median_w'] > 0:
        out = a >= road.e_kerb
    gr = 0.36 + 2.5 * mott[out]
    rgb[out] = np.c_[0.25 * gr / 0.36, 0.38 * gr / 0.36, 0.17 * gr / 0.36]
    refl[out] = REFL['grass']
    pm = paint > 0
    rgb[pm] = rgb[pm] * (1 - paint[pm, None]) + paint[pm, None] * np.array([0.90, 0.90, 0.86])
    refl[pm] = refl[pm] * (1 - paint[pm]) + REFL['paint'] * paint[pm]
    return rgb, refl


def surface_shade(ps: _PrimSet, idx, P, nrm):
    """(rgb (n, 3), reflectivity (n,)) of hits at points P (n, 3) with normals on parts idx (n,): materials before
    lighting."""
    n = len(idx)
    rgb = ps.rgb[idx].copy()
    refl = ps.refl[idx].copy()
    mat = ps.mat[idx]
    z = P[:, 2]
    s = mat == M_CAR
    if s.any():
        h0, h1 = ps.z0[idx[s]], ps.z1[idx[s]]
        frac = (z[s] - h0) / np.maximum(h1 - h0, 1e-6)
        top = nrm[s, 2] > 0.5
        glass = (frac > 0.6) & ~top
        tyre = (z[s] - h0) < 0.32
        c = rgb[s]
        c = np.where(glass[:, None], [0.10, 0.12, 0.14], np.where(tyre[:, None], [0.07, 0.07, 0.07], c))
        c = np.where(top[:, None], rgb[s] * 1.08, c)
        rgb[s] = c
        refl[s] = np.where(glass, REFL['car_glass'], np.where(tyre, REFL['tyre'], refl[s]))
    s = mat == M_BUILDING
    if s.any():
        ii = idx[s]
        c, sn = np.cos(ps.yaw[ii]), np.sin(ps.yaw[ii])
        lx = (P[s, 0] - ps.cx[ii]) * c + (P[s, 1] - ps.cy[ii]) * sn
        ly = -(P[s, 0] - ps.cx[ii]) * sn + (P[s, 1] - ps.cy[ii]) * c
        nlx = nrm[s, 0] * c + nrm[s, 1] * sn
        u = np.where(np.abs(nlx) > 0.5, ly, lx)
        zz = z[s]
        roof = nrm[s, 2] > 0.5
        win = (((u % 2.6) > 0.6) & ((u % 2.6) < 1.9) & ((zz % 3.2) > 0.9) & ((zz % 3.2) < 2.3) & (zz > 3.2))
        shop = ((zz > 0.3) & (zz < 2.6) & ((u % 5.0) > 0.4) & ((u % 5.0) < 4.4))
        glass = (win | shop) & ~roof
        cc = rgb[s]
        cc = np.where(glass[:, None], np.array([0.13, 0.16, 0.20]) + 0.04 * np.sin(zz * 0.9)[:, None], cc)
        cc = np.where(roof[:, None], [0.30, 0.29, 0.28], cc)
        rgb[s] = cc
        refl[s] = np.where(glass, REFL['window'], np.where(roof, REFL['roof'], refl[s]))
    s = mat == M_PERSON
    if s.any():
        ii = idx[s]
        rel = z[s] - ps.z0[ii]
        legs = rel < 0.85
        head = rel > 1.50
        cc = rgb[s]
        cc = np.where(legs[:, None], [0.16, 0.18, 0.26], cc)
        cc = np.where(head[:, None], [0.62, 0.46, 0.36], cc)
        rgb[s] = cc
    s = mat == M_SIGN
    if s.any():
        ii = idx[s]
        c, sn = np.cos(ps.yaw[ii]), np.sin(ps.yaw[ii])
        ly = -(P[s, 0] - ps.cx[ii]) * sn + (P[s, 1] - ps.cy[ii]) * c
        lz = z[s] - (ps.z0[ii] + ps.z1[ii]) / 2
        nlx = nrm[s, 0] * c + nrm[s, 1] * sn
        face = nlx < -0.5                                    # the face looks along -x of the plate (at traffic)
        border = np.maximum(np.abs(ly) / (ps.W[ii] / 2), np.abs(lz) / ((ps.z1[ii] - ps.z0[ii]) / 2)) > 0.72
        cc = np.where(face[:, None], np.where(border[:, None], rgb[s], [0.92, 0.92, 0.90]), [0.55, 0.56, 0.57])
        rgb[s] = cc
        refl[s] = np.where(face, REFL['sign'], REFL['sign_back'])
    return rgb, refl


# =========================================================================== scene generation
def _box(ox, oy, oyaw, L, W, z0, z1, mat, rgb, refl):
    return (BOX, mat, tuple(rgb), float(refl), (float(ox), float(oy), float(oyaw), float(L), float(W), float(z0),
                                                float(z1)))


def _cyl(ox, oy, r, z0, z1, mat, rgb, refl):
    return (CYL, mat, tuple(rgb), float(refl), (float(ox), float(oy), float(r), float(z0), float(z1)))


def _sph(ox, oy, oz, r, mat, rgb, refl):
    return (SPH, mat, tuple(rgb), float(refl), (float(ox), float(oy), float(oz), float(r)))


CAR_COLOURS = [(0.16, 0.24, 0.42), (0.58, 0.12, 0.10), (0.80, 0.80, 0.78), (0.20, 0.20, 0.21), (0.55, 0.56, 0.58),
               (0.12, 0.30, 0.20), (0.70, 0.55, 0.20), (0.08, 0.08, 0.10)]
CAR_SIZES = [(4.0, 1.75, 1.48), (4.5, 1.85, 1.52), (4.9, 1.95, 1.75), (3.7, 1.70, 1.50)]
FACADES = [(0.72, 0.66, 0.58), (0.62, 0.58, 0.55), (0.78, 0.74, 0.66), (0.55, 0.42, 0.36), (0.66, 0.68, 0.70),
           (0.80, 0.78, 0.72)]
CLOTHES = [(0.62, 0.16, 0.14), (0.20, 0.36, 0.62), (0.85, 0.80, 0.70), (0.24, 0.50, 0.30), (0.75, 0.58, 0.15),
           (0.30, 0.30, 0.32)]


def car_object(oid, size, rgb, kind='car', category='vehicle.car', **kw) -> Obj:
    L, W, H = size
    return Obj(oid, kind, (L, W, H), [_box(0, 0, 0, L, W, 0.0, H, M_CAR, rgb, REFL['car'])], category=category,
               rcs=TARGET_RCS['truck' if kind == 'truck' else 'car'], **kw)


def pedestrian_object(oid, rgb, height=1.75, **kw) -> Obj:
    return Obj(oid, 'pedestrian', (0.6, 0.6, height), [_cyl(0, 0, 0.27, 0.0, height, M_PERSON, rgb, REFL['person'])],
               category='human.pedestrian.adult', rcs=TARGET_RCS['pedestrian'], **kw)


def cyclist_object(oid, rgb, **kw) -> Obj:
    return Obj(oid, 'cyclist', (1.75, 0.62, 1.75),
               [_box(0, 0, 0, 1.70, 0.12, 0.05, 1.00, M_PLAIN, (0.12, 0.12, 0.13), REFL['tyre'] * 2),
                _cyl(-0.12, 0, 0.24, 0.85, 1.75, M_PERSON, rgb, REFL['person'])],
               category='vehicle.bicycle', rcs=TARGET_RCS['cyclist'], **kw)


def building_object(oid, pose, L, W, H, rgb, **kw) -> Obj:
    return Obj(oid, 'building', (L, W, H), [_box(0, 0, 0, L, W, 0.0, H, M_BUILDING, rgb, REFL['facade'])], pose=pose,
               **kw)


def _obb_overlap(a, b, margin=0.0) -> bool:
    """Do two plan rectangles (cx, cy, yaw, L, W) overlap (separating axis test)?"""
    def corners(r):
        cx, cy, yaw, L, W = r
        c, s = math.cos(yaw), math.sin(yaw)
        q = np.array([[L / 2 + margin, W / 2 + margin], [-L / 2 - margin, W / 2 + margin],
                      [-L / 2 - margin, -W / 2 - margin], [L / 2 + margin, -W / 2 - margin]])
        return np.c_[cx + q[:, 0] * c - q[:, 1] * s, cy + q[:, 0] * s + q[:, 1] * c], (c, s)
    pa, (ca, sa) = corners(a)
    pb, (cb, sb) = corners(b)
    for ax in ((ca, sa), (-sa, ca), (cb, sb), (-sb, cb)):
        da, db = pa @ ax, pb @ ax
        if da.max() < db.min() or db.max() < da.min():
            return False
    return True


def build_scene(kind: str = 'urban', speed_kmh: float = 30.0, duration: float = 8.0, seed: int = 1,
                ego_length: float = 4.0, ego_rear: float = -0.8) -> Scene:
    """The seeded scene for a drive of `duration` s at `speed_kmh`. ego_length / ego_rear: the ego's length and the
    x of its rear end in its own frame (to keep traffic clear of it)."""
    rng = np.random.default_rng([int(seed), 7741])
    prof = HIGHWAY if kind == 'highway' else URBAN
    v1 = max(speed_kmh, 0.0) / 3.6
    probe = EgoDrive(Road(0, 0, 0, 100, 100, 1, 0.5, prof), speed_kmh, duration, 0.0)
    D = float(probe.travel(duration))
    if kind == 'highway':
        L1 = max(30.0, 0.35 * D)
        R = 400.0
        theta = float(np.clip((D - L1 + 120.0) / R, math.radians(10), math.radians(35)))
    else:
        L1 = max(14.0, 0.3 * D)
        R = 50.0
        theta = float(np.clip((D - L1 + 30.0) / R, math.radians(40), math.radians(90)))
    turn = 1 if rng.random() < 0.5 else -1
    psi0 = float(rng.uniform(0, 2 * math.pi))
    road0 = Road(0.0, 0.0, psi0, L1, R, turn, theta, prof)
    lane_w = prof['lane_w']
    m = prof['median_w'] / 2
    ego_d = -(m + lane_w / 2) if kind != 'highway' else -(m + lane_w * 1.5)       # highway: the middle lane
    s_lo, s_hi = -70.0, D + 110.0
    # the extent of everything near the road, then shift the scene to positive x, y
    ss = np.linspace(s_lo, s_hi, 400)
    reach = road0.e_verge + 30.0 if kind != 'highway' else road0.e_verge + 12.0
    xs, ys = [], []
    for dd in (-reach, reach):
        x, y, _ = road0.to_world(ss, dd)
        xs.append(x)
        ys.append(y)
    xs, ys = np.concatenate(xs), np.concatenate(ys)
    shift = np.array([20.0 - xs.min(), 20.0 - ys.min()])
    road = Road(shift[0], shift[1], psi0, L1, R, turn, theta, prof)
    extent = (0.0, 0.0, float(xs.max() + shift[0] + 20.0), float(ys.max() + shift[1] + 20.0))
    drive = EgoDrive(road, speed_kmh, duration, ego_d)
    objs: list[Obj] = []
    nid = [0]

    def new_id():
        nid[0] += 1
        return nid[0]

    def place_static(s, d, yaw_rel=0.0):
        x, y, psi = road.to_world(s, d)
        return (float(x), float(y), float(psi) + yaw_rel)
    crossing = 1.05 * D + 36.0 if kind != 'highway' else None
    if kind == 'highway':
        _highway_static(road, rng, s_lo, s_hi, objs, new_id, place_static)
    else:
        _urban_static(road, rng, s_lo, s_hi, crossing, objs, new_id, place_static)
    # ---- moving traffic, checked against each other and the ego
    T = float(duration)
    tt = np.linspace(0.0, T, max(int(T / 0.1) + 1, 2))
    ex, ey, _ = drive.pose(tt)
    es, ed = road.to_sd(ex, ey)
    tracks = [(es, ed, ego_length, 2.0, ego_rear + ego_length / 2)]          # (s(t), d(t), length, width, centre off)

    def clear(mo: Motion, L, W) -> bool:
        s, d = mo.sd(tt)
        for (s2, d2, L2, W2, off) in tracks:
            if np.any((np.abs(s - (s2 + off)) < (L + L2) / 2 + 1.0) & (np.abs(d - d2) < (W + W2) / 2 + 0.3)):
                return False
        return True

    def add_actor(obj: Obj, mo: Motion, tries=12, step=9.0):
        """Place a moving actor, shifting its start along the road until it meets nothing; dropped otherwise."""
        for k in range(tries):
            off = 0.0 if k == 0 else step * ((k + 1) // 2) * (1 if k % 2 else -1)
            cand = Motion(mo.s0 + off, mo.d0, mo.vs, mo.vd, mo.d_lim, mo.yaw_off)
            if clear(cand, obj.size[0], obj.size[1]):
                obj.motion = cand
                objs.append(obj)
                tracks.append((*cand.sd(tt), obj.size[0], obj.size[1], 0.0))
                return True
        return False
    v_on = max(v1, 25 / 3.6 if kind != 'highway' else 80 / 3.6)
    if kind == 'highway':
        _highway_traffic(rng, v1, v_on, D, m, lane_w, add_actor, new_id)
    else:
        _urban_traffic(road, rng, v1, v_on, D, crossing, add_actor, new_id)
    for o in objs:
        if not o.name:
            o.name = f'{o.kind} {o.id}'
    return Scene(road, drive, objs, kind, seed, extent, crossing)


def _urban_static(road, rng, s_lo, s_hi, crossing, objs, new_id, place_static):
    p = road.p
    # buildings on both sides, kept apart (the inside of the bend squeezes them)
    plots = []
    for side in (1, -1):
        s = s_lo + rng.uniform(0, 6)
        while s < s_hi:
            L = rng.uniform(10, 22)
            W = rng.uniform(8, 14)
            H = rng.uniform(6, 22)
            d0 = p['build_d'] + rng.uniform(0, 2.0)
            sc = s + L / 2
            pose = place_static(sc, side * (d0 + W / 2))
            r = (pose[0], pose[1], pose[2], L, W)
            if not any(_obb_overlap(r, q, 0.3) for q in plots):
                plots.append(r)
                objs.append(building_object(new_id(), pose, L, W, H, FACADES[rng.integers(len(FACADES))]))
            s += L + rng.uniform(1.0, 5.0)
    # kerbs, in pieces of at most 8 m (4 m in the bend)
    s = s_lo
    while s < s_hi:
        inbend = road.L1 - 8 < s < road.L1 + road.Larc
        step = 4.0 if inbend else 8.0
        for side in (1, -1):
            dk = road.e_kerb + p['kerb_w'] / 2
            pose = place_static(s + step / 2, side * dk)
            objs.append(Obj(new_id(), 'kerb', (step, p['kerb_w'], KERB_H),
                            [_box(0, 0, 0, step + 0.02, p['kerb_w'], 0.0, KERB_H, M_PLAIN, (0.62, 0.62, 0.60),
                                  REFL['kerb'])], pose=pose))
        s += step
    # street lights, staggered, the arm over the road
    dl = road.e_kerb + p['kerb_w'] + 0.5
    for side, off in ((-1, 0.0), (1, 12.5)):
        s = s_lo + off
        while s < s_hi:
            objs.append(Obj(new_id(), 'lamp', (0.3, 1.8, 7.5),
                            [_cyl(0, 0, 0.09, 0.0, 7.5, M_PLAIN, (0.45, 0.46, 0.47), REFL['pole']),
                             _box(0.0, -side * 0.8, 0.0, 0.18, 1.6, 7.25, 7.45, M_PLAIN, (0.40, 0.41, 0.42),
                                  REFL['pole'])],
                            pose=place_static(s, side * dl), rcs=TARGET_RCS['lamp']))
            s += 25.0
    # trees on the verges
    dv = road.e_pave + p['verge_w'] / 2
    for side in (1, -1):
        s = s_lo + rng.uniform(0, 8)
        while s < s_hi:
            rr = rng.uniform(1.5, 2.3)
            green = (0.18 + rng.uniform(-0.03, 0.03), 0.33 + rng.uniform(-0.05, 0.05), 0.12)
            objs.append(Obj(new_id(), 'tree', (2 * rr, 2 * rr, 2.6 + 1.8 * rr),
                            [_cyl(0, 0, 0.15, 0.0, 2.7, M_PLAIN, (0.30, 0.22, 0.15), REFL['bark']),
                             _sph(0, 0, 2.6 + 0.8 * rr, rr, M_PLAIN, green, REFL['leaves'])],
                            pose=place_static(s, side * dv)))
            s += rng.uniform(10, 15)
    # traffic signs facing the traffic on their side
    ds = road.e_kerb + p['kerb_w'] + 0.35
    kinds = [(0.78, 0.10, 0.10), (0.10, 0.25, 0.62), (0.78, 0.10, 0.10)]
    for i, s in enumerate(np.arange(s_lo + 18.0, s_hi, 34.0)):
        side = -1 if i % 2 == 0 else 1
        pose = place_static(s, side * ds, 0.0 if side < 0 else math.pi)
        objs.append(Obj(new_id(), 'sign', (0.06, 0.66, 2.7),
                        [_cyl(0, 0, 0.04, 0.0, 2.05, M_PLAIN, (0.55, 0.56, 0.57), REFL['pole']),
                         _box(0.0, 0, 0, 0.05, 0.66, 2.0, 2.66, M_SIGN, kinds[i % 3], REFL['sign'])], pose=pose,
                        rcs=TARGET_RCS['sign']))
    # parked cars in the parking strips, none at the crossing
    dp = road.e_cycle + p['park_w'] / 2
    for side in (1, -1):
        s = s_lo + rng.uniform(0, 6)
        while s < s_hi:
            L, W, H = CAR_SIZES[rng.integers(len(CAR_SIZES))]
            if rng.random() < 0.6 and (crossing is None or abs(s + L / 2 - crossing) > 6.0):
                pose = place_static(s + L / 2, side * dp, 0.0 if side < 0 else math.pi)
                objs.append(car_object(new_id(), (L, W - 0.05, H), CAR_COLOURS[rng.integers(len(CAR_COLOURS))],
                                       pose=pose))
                objs[-1].name = f'parked car {objs[-1].id}'
            s += L + rng.uniform(1.2, 4.0)


def _urban_traffic(road, rng, v1, v_on, D, crossing, add_actor, new_id):
    lane_w = road.p['lane_w']
    dr, dl = -lane_w / 2, lane_w / 2
    col = lambda: CAR_COLOURS[rng.integers(len(CAR_COLOURS))]           # noqa: E731
    size = lambda: CAR_SIZES[rng.integers(len(CAR_SIZES))]              # noqa: E731
    add_actor(car_object(new_id(), size(), col()), Motion(24.0, dr, 1.05 * v1))                     # ahead
    add_actor(car_object(new_id(), size(), col()), Motion(-15.0, dr, 0.9 * v1))                     # behind
    if crossing is not None:                  # one pedestrian crossing at the zebra, mid-road half way through
        lim = road.e_kerb + road.p['kerb_w'] + 1.5
        T = (D / max(v1, 1e-6)) if v1 > 0 else 8.0
        add_actor(pedestrian_object(new_id(), CLOTHES[rng.integers(len(CLOTHES))], 1.72),
                  Motion(crossing, max(-lim, -1.3 * T / 2), 0.0, 1.3, (-lim, lim)), tries=1)
    for k in range(4):                                                                              # oncoming
        add_actor(car_object(new_id(), size(), col()), Motion(D * 0.6 + 20.0 + 32.0 * k + rng.uniform(-4, 4), dl,
                                                              -v_on * rng.uniform(0.9, 1.1)))
    add_actor(car_object(new_id(), (8.5, 2.5, 3.3), (0.85, 0.85, 0.82), 'truck', 'vehicle.truck'),
              Motion(D + 90.0, dl, -v_on * 0.85))
    dc = road.e_lanes + road.p['cycle_w'] / 2
    add_actor(cyclist_object(new_id(), CLOTHES[rng.integers(len(CLOTHES))]), Motion(30.0, -dc, 4.5))
    add_actor(cyclist_object(new_id(), CLOTHES[rng.integers(len(CLOTHES))]), Motion(D + 40.0, dc, -4.2))
    dw = road.e_kerb + road.p['kerb_w'] + road.p['pave_w'] * 0.6
    for k in range(7):
        side = 1 if k % 2 else -1
        s0 = -10.0 + (D + 70.0) * (k + 0.5) / 7 + rng.uniform(-5, 5)
        v = 0.0 if k % 3 == 2 else rng.uniform(1.1, 1.5) * (1 if rng.random() < 0.5 else -1)
        add_actor(pedestrian_object(new_id(), CLOTHES[rng.integers(len(CLOTHES))], rng.uniform(1.62, 1.88)),
                  Motion(s0, side * (dw + rng.uniform(-0.6, 0.6)), v, yaw_off=float(rng.uniform(-1, 1))))


def _highway_static(road, rng, s_lo, s_hi, objs, new_id, place_static):
    p = road.p
    s = s_lo
    while s < s_hi:                                            # median barrier and noise walls in pieces
        inbend = road.L1 - 10 < s < road.L1 + road.Larc
        step = 10.0 if inbend else 20.0
        pose = place_static(s + step / 2, 0.0)
        objs.append(Obj(new_id(), 'barrier', (step, 0.5, 0.9),
                        [_box(0, 0, 0, step + 0.02, 0.5, 0.0, 0.9, M_PLAIN, (0.66, 0.66, 0.64), REFL['barrier'])],
                        pose=pose))
        for side in (1, -1):
            pose = place_static(s + step / 2, side * (road.e_verge + 0.2))
            objs.append(Obj(new_id(), 'wall', (step, 0.3, 4.0),
                            [_box(0, 0, 0, step + 0.02, 0.3, 0.0, 4.0, M_PLAIN, (0.52, 0.55, 0.50), REFL['barrier'])],
                            pose=pose))
        s += step
    for i, s in enumerate(np.arange(s_lo + 30.0, s_hi, 60.0)):
        side = -1 if i % 2 == 0 else 1
        pose = place_static(s, side * (road.e_kerb + 1.0), 0.0 if side < 0 else math.pi)
        objs.append(Obj(new_id(), 'sign', (0.08, 1.6, 3.8),
                        [_cyl(0, 0, 0.06, 0.0, 2.6, M_PLAIN, (0.55, 0.56, 0.57), REFL['pole']),
                         _box(0, 0, 0, 0.06, 1.6, 2.5, 3.8, M_SIGN, (0.05, 0.35, 0.15), REFL['sign'])], pose=pose,
                        rcs=TARGET_RCS['sign']))


def _highway_traffic(rng, v1, v_on, D, m, lane_w, add_actor, new_id):
    col = lambda: CAR_COLOURS[rng.integers(len(CAR_COLOURS))]           # noqa: E731
    size = lambda: CAR_SIZES[rng.integers(len(CAR_SIZES))]              # noqa: E731
    lanes_r = [-(m + lane_w * (k + 0.5)) for k in range(3)]
    lanes_l = [m + lane_w * (k + 0.5) for k in range(3)]
    add_actor(car_object(new_id(), size(), col()), Motion(40.0, lanes_r[1], 1.03 * v1))
    add_actor(car_object(new_id(), size(), col()), Motion(-30.0, lanes_r[1], 0.95 * v1))
    add_actor(car_object(new_id(), size(), col()), Motion(-20.0, lanes_r[0], 1.15 * v1 + 2.0))
    add_actor(car_object(new_id(), (12.0, 2.55, 3.8), (0.82, 0.82, 0.80), 'truck', 'vehicle.truck'),
              Motion(25.0, lanes_r[2], 0.85 * v1))
    add_actor(car_object(new_id(), size(), col()), Motion(70.0, lanes_r[2], 0.9 * v1))
    for k in range(6):
        add_actor(car_object(new_id(), size(), col()),
                  Motion(D * 0.5 + 40.0 + 45.0 * k, lanes_l[k % 3], -v_on * rng.uniform(0.9, 1.1)))


# =========================================================================== geodetic
WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2 - WGS84_F)


def geodetic_to_ecef(lat, lon, alt):
    lat, lon = np.radians(lat), np.radians(lon)
    N = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(lat) ** 2)
    return np.stack([(N + alt) * np.cos(lat) * np.cos(lon), (N + alt) * np.cos(lat) * np.sin(lon),
                     (N * (1 - WGS84_E2) + alt) * np.sin(lat)], -1)


def ecef_to_geodetic(P):
    """(lat, lon degrees, ellipsoidal height m) of ECEF points (n, 3): Bowring's start, three Newton refinements
    (sub-millimetre anywhere near the surface)."""
    P = np.asarray(P, float)
    X, Y, Z = P[..., 0], P[..., 1], P[..., 2]
    lon = np.arctan2(Y, X)
    p = np.hypot(X, Y)
    b = WGS84_A * (1 - WGS84_F)
    ep2 = (WGS84_A ** 2 - b ** 2) / b ** 2
    th = np.arctan2(Z * WGS84_A, p * b)
    lat = np.arctan2(Z + ep2 * b * np.sin(th) ** 3, p - WGS84_E2 * WGS84_A * np.cos(th) ** 3)
    for _ in range(3):
        N = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(lat) ** 2)
        h = p / np.cos(lat) - N
        lat = np.arctan2(Z, p * (1 - WGS84_E2 * N / (N + h)))
    N = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(lat) ** 2)
    h = p / np.cos(lat) - N
    return np.degrees(lat), np.degrees(lon), h


def _enu_basis(lat0, lon0):
    la, lo = math.radians(lat0), math.radians(lon0)
    return np.array([[-math.sin(lo), math.cos(lo), 0.0],
                     [-math.sin(la) * math.cos(lo), -math.sin(la) * math.sin(lo), math.cos(la)],
                     [math.cos(la) * math.cos(lo), math.cos(la) * math.sin(lo), math.sin(la)]])


def enu_to_geodetic(E, lat0, lon0, alt0):
    """World (east, north, up) points (n, 3) about the origin -> (lat, lon, height)."""
    E = np.atleast_2d(np.asarray(E, float))
    X = geodetic_to_ecef(lat0, lon0, alt0) + E @ _enu_basis(lat0, lon0)
    return ecef_to_geodetic(X)


def geodetic_to_enu(lat, lon, alt, lat0, lon0, alt0):
    X = geodetic_to_ecef(np.asarray(lat, float), np.asarray(lon, float), np.asarray(alt, float))
    return (X - geodetic_to_ecef(lat0, lon0, alt0)) @ _enu_basis(lat0, lon0).T


# =========================================================================== cameras
class CameraSim:
    """One camera's pixels: their rays in the vehicle frame (sorted by 16 × 16 tile), what of the body each sees, and
    the image size. rectified: an exact pinhole image of the scene through K (the rays outside the lens's image
    circle stay black), else the real lens."""

    def __init__(self, project, engine, cam, w: int, h: int, rectified: bool = False, K=None):
        self.cam, self.w, self.h, self.rectified = cam, int(w), int(h), bool(rectified)
        self.intr = intr = resolve(project.module_for(cam), project.lens_for(cam))
        self.module, self.lens = project.module_for(cam), project.lens_for(cam)     # the night photon budget
        self.sx, self.sy = self.w / intr.width, self.h / intr.height
        R = rot_zyx(cam.yaw, cam.pitch, cam.roll)
        self.R_vc = R                                  # camera body -> vehicle
        self.W2O = BODY_TO_OPTICAL @ R.T               # vehicle -> optical
        self.pos = cam.pos().astype(float)
        if rectified:
            self.K = np.asarray(K, float)
            U, V = np.meshgrid(np.arange(self.w, dtype=float), np.arange(self.h, dtype=float))
            x = (U.ravel() - self.K[0, 2]) / self.K[0, 0]
            y = (V.ravel() - self.K[1, 2]) / self.K[1, 1]
            d = np.stack([x, y, np.ones_like(x)], 1)
            d /= np.linalg.norm(d, axis=1, keepdims=True)
            th = np.arccos(np.clip(d[:, 2], -1, 1))
            self.pix_ang = 1.0 / self.K[0, 0]
        else:
            self.K = None
            d, th = pixel_rays(intr, self.w, self.h)
            self.pix_ang = intr.pixel_mm / max(intr.f_mm, 1e-6) / self.sx
        valid = th <= intr.max_theta
        rows = np.repeat(np.arange(self.h), self.w)
        cols = np.tile(np.arange(self.w), self.h)
        ntx = (self.w + TILE - 1) // TILE
        tile = (rows // TILE) * ntx + cols // TILE
        idx = np.nonzero(valid)[0]
        order = idx[np.argsort(tile[idx], kind='stable')]
        self.pix = order                               # flat pixel index of each ray, tile order
        self.row = rows[order]
        dv = d[order] @ self.W2O                       # optical -> vehicle
        self.dv = dv
        t_of = tile[order]
        tiles, start, counts = np.unique(t_of, return_index=True, return_counts=True)
        self.tile_start, self.tile_len = start, counts
        self.tile_row = (tiles // ntx)
        ax = np.add.reduceat(dv, start, axis=0)
        ax /= np.maximum(np.linalg.norm(ax, axis=1, keepdims=True), 1e-12)
        self.tile_axis = ax
        cosang = np.sum(dv * np.repeat(ax, counts, axis=0), axis=1)
        self.tile_cone = np.maximum.reduceat(np.arccos(np.clip(cosang, -1, 1)), start) + 1e-3
        self.ray_tile = np.repeat(np.arange(len(start)), counts)
        self.body_d = np.full(len(order), np.inf)
        self.body_shade = np.zeros(len(order))
        self.house = None                              # per ray: index into house_rgb of the housing seen, -1 none
        self.house_rgb = np.zeros((0, 3))
        self._cube = None
        if engine is not None and engine.vehicle_geom(project) is not None:
            self._cube = engine.cube(self.pos)
            bd, sh, hid = self._cube.lookup_ids(dv)
            ok = np.isfinite(bd) & ((bd > NEAR_CLIP) | (hid >= 0))      # other housings show however close
            self.body_d = np.where(ok, bd, np.inf)
            self.body_shade = sh.astype(float)
            if (ok & (hid >= 0)).any():
                from .housings import RGB as H_RGB
                self.house = np.where(ok, hid, -1).astype(np.int16)
                self.house_rgb = np.array([H_RGB.get(h.what, (0.14, 0.14, 0.15)) for h in self._cube.housings])

    # ---- projection of vehicle-frame directions into this image (for visibility and checks)
    def project(self, dirs_v):
        """Pixel (u, v) in this image's pixel coordinates (pixel centres on i + 0.5) and in-image flag."""
        d_opt = np.asarray(dirs_v, float) @ self.W2O.T
        if self.rectified:
            z = d_opt[:, 2]
            with np.errstate(divide='ignore', invalid='ignore'):
                u = self.K[0, 0] * d_opt[:, 0] / z + self.K[0, 2] + 0.5
                v = self.K[1, 1] * d_opt[:, 1] / z + self.K[1, 2] + 0.5
            th = np.arccos(np.clip(d_opt[:, 2] / np.linalg.norm(d_opt, axis=1), -1, 1))
            ok = (z > 1e-6) & (u >= 0) & (u <= self.w) & (v >= 0) & (v <= self.h) & (th <= self.intr.max_theta)
            return u, v, ok
        u, v, _, ok = self.intr.project_dirs(d_opt)
        return u * self.sx, v * self.sy, ok

    def times(self, t_trigger, readout_s, exposure_s, rolling):
        """(per-band exposure-centre times or None, the frame's time): the middle row's exposure centre for a global
        capture; with a rolling shutter each 16-row band's own (rows read top to bottom over the readout)."""
        t_mid = t_trigger + readout_s / 2 + exposure_s / 2
        if not rolling or readout_s <= 0:
            return None, t_mid
        nb = (self.h + TILE - 1) // TILE
        centre = np.minimum(np.arange(nb) * TILE + TILE / 2, self.h) / self.h
        return t_trigger + readout_s * centre + exposure_s / 2, t_mid

    def render(self, scene: Scene, t_trigger: float, readout_s=0.0, exposure_s=0.0, rolling=False, noise=2.0,
               seed=0, want_ids=False, wx=None):
        """RGB uint8 (h, w, 3) image of the scene for a frame triggered at t_trigger (and the object id per pixel when
        want_ids: -3 ground, -2 body, -1 sky or outside the image circle). wx: a core.weather SceneWeather (None:
        clear)."""
        tb, t_mid = self.times(t_trigger, readout_s, exposure_s, rolling)
        drive = scene.drive
        ps = scene.prims_at(t_mid)
        n = len(self.dv)
        dv = self.dv
        xm, ym, pm = (float(v) for v in drive.pose(t_mid))
        if tb is None:
            c, s = math.cos(pm), math.sin(pm)
            D = np.stack([c * dv[:, 0] - s * dv[:, 1], s * dv[:, 0] + c * dv[:, 1], dv[:, 2]], 1)
            O = np.array([xm + c * self.pos[0] - s * self.pos[1], ym + s * self.pos[0] + c * self.pos[1], self.pos[2]])
            tile_ax = self.tile_axis @ _rz(pm).T
            dt = None
            grow = 0.0
        else:
            xb, yb, pb = drive.pose(tb)
            band = self.row // TILE
            c, s = np.cos(pb)[band], np.sin(pb)[band]
            D = np.stack([c * dv[:, 0] - s * dv[:, 1], s * dv[:, 0] + c * dv[:, 1], dv[:, 2]], 1)
            O = np.stack([xb[band] + c * self.pos[0] - s * self.pos[1], yb[band] + s * self.pos[0] + c * self.pos[1],
                          np.full(n, self.pos[2])], 1)
            tband = self.tile_row
            ct, st = np.cos(pb)[tband], np.sin(pb)[tband]
            a = self.tile_axis
            tile_ax = np.stack([ct * a[:, 0] - st * a[:, 1], st * a[:, 0] + ct * a[:, 1], a[:, 2]], 1)
            dt = (tb - t_mid)[band]
            grow = float(drive.speed(t_mid)) * (readout_s + exposure_s) + 0.05
        cam_w = np.array([xm + math.cos(pm) * self.pos[0] - math.sin(pm) * self.pos[1],
                          ym + math.sin(pm) * self.pos[0] + math.cos(pm) * self.pos[1], self.pos[2]])
        best = self.body_d.copy()
        pid = np.where(np.isfinite(best), P_BODY, P_SKY).astype(np.int32)
        Oz = O[:, 2] if O.ndim > 1 else O[2]
        with np.errstate(divide='ignore', invalid='ignore'):
            tg = np.where(D[:, 2] < -1e-9, -Oz / D[:, 2], np.inf)
        sel = tg < best
        best[sel], pid[sel] = tg[sel], P_GROUND
        span = (readout_s + exposure_s) if tb is not None else 0.0
        for i, tl in self._cull(ps, cam_w, tile_ax, grow, span):
            idx = _ranges(self.tile_start[tl], self.tile_len[tl])
            Oi = O if O.ndim == 1 else O[idx]
            if dt is not None and ps.moving(i):
                Oi = Oi - np.c_[ps.vx[i] * dt[idx], ps.vy[i] * dt[idx], np.zeros(len(idx))]
            t = ps.hit(i, Oi, D[idx])
            closer = t < best[idx]
            if closer.any():
                j = idx[closer]
                best[j] = t[closer]
                pid[j] = i
        if wx is not None:
            out = self._weather_image(scene, ps, O, D, best, pid, dt, wx, noise, seed, t_mid)
        else:
            col = self._shade(scene, ps, O, D, best, pid, dt)
            img = np.full((self.h * self.w, 3), 0.16 * 255.0, np.float32)
            img[self.pix] = col * 255.0
            if noise:
                img += np.random.default_rng(seed).standard_normal(img.shape, dtype=np.float32) * noise
            out = np.clip(img, 0, 255).astype(np.uint8).reshape(self.h, self.w, 3)
        if not want_ids:
            return out
        ids = np.full(self.h * self.w, P_SKY, np.int32)
        oid = pid.copy()
        oid[pid >= 0] = ps.obj[pid[pid >= 0]]
        ids[self.pix] = oid
        return out, ids.reshape(self.h, self.w)

    def _cull(self, ps, cam_w, tile_ax, grow, span):
        """[(part, tiles whose cone its bounding sphere meets)] for the parts within CAM_CULL_M of the camera."""
        if not ps.n:
            return []
        w = ps.bc - cam_w
        dd = np.linalg.norm(w, axis=1)
        rad = ps.br + np.hypot(ps.vx, ps.vy) * span + grow + 0.02
        cand = np.nonzero(dd < CAM_CULL_M + rad)[0]
        if not len(cand):
            return []
        wn = w[cand] / np.maximum(dd[cand], 1e-9)[:, None]
        alpha = np.arcsin(np.clip(rad[cand] / np.maximum(dd[cand], 1e-9), 0, 1))
        ang = np.arccos(np.clip(wn @ tile_ax.T, -1, 1))
        hit = ang <= alpha[:, None] + self.tile_cone[None, :]
        hit |= (dd[cand] <= rad[cand])[:, None]
        out = []
        for k in np.nonzero(hit.any(1))[0]:
            out.append((int(cand[k]), np.nonzero(hit[k])[0]))
        return out

    def _shade(self, scene, ps, O, D, best, pid, dt):
        col = np.empty((len(D), 3), np.float32)

        def hits(s):
            return (O[s] if O.ndim > 1 else O[None, :]) + D[s] * best[s][:, None]
        s = np.nonzero(pid == P_SKY)[0]
        if len(s):
            el = np.clip(D[s, 2], 0, 1)[:, None]
            col[s] = np.array([0.80, 0.84, 0.88]) * (1 - el) + np.array([0.55, 0.66, 0.80]) * el
        s = np.nonzero(pid == P_BODY)[0]
        if len(s):
            col[s] = (0.22 + 0.74 * self.body_shade[s, None] / 255.0) * np.array([0.80, 0.82, 0.86])
            if self.house is not None:                  # the rig's own sensors and cameras: dark plastic and metal
                from .housings import tone
                hs = s[self.house[s] >= 0]
                col[hs] = self.house_rgb[self.house[hs]] * tone(self.body_shade[hs])[:, None]
        s = np.nonzero(pid >= 0)[0]
        if len(s):
            Ps = hits(s)
            if dt is not None:                      # the hit on a moving part, in its frame-time position
                mv = ps.vx[pid[s]] != 0
                Ps[mv, 0] -= ps.vx[pid[s][mv]] * dt[s][mv]
                Ps[mv, 1] -= ps.vy[pid[s][mv]] * dt[s][mv]
            nrm = ps.normals(pid[s], Ps)
            rgb, _ = surface_shade(ps, pid[s], Ps, nrm)
            col[s] = rgb * (0.50 + 0.55 * np.clip(nrm @ SUN, 0, 1))[:, None]
        s = np.nonzero(pid == P_GROUND)[0]
        if len(s):
            g = hits(s)
            sd_s, sd_d = scene.road.to_sd(g[:, 0], g[:, 1])
            fw = best[s] * self.pix_ang / np.clip(np.abs(D[s, 2]), 0.05, 1.0)
            rgb, _ = ground_material(scene.road, sd_s.astype(np.float32), sd_d.astype(np.float32),
                                     fw.astype(np.float32), scene.crossing_s)
            lit = self._shadows(ps, g, s, None if dt is None else dt[s])
            col[s] = rgb * (0.62 + 0.38 * lit)[:, None]
        return col

    def _shadows(self, ps, g, sidx, dt=None):
        """1 where a ground point sees the sun, 0 in an object's shadow (dt: each point's time from the frame's, for
        the parts that move during a rolling-shutter readout)."""
        lit = np.ones(len(g))
        if not len(g) or not ps.n:
            return lit
        tiles = self.ray_tile[sidx]
        _, start = np.unique(tiles, return_index=True)
        ends = np.r_[start[1:], len(g)]
        lo = np.minimum.reduceat(g[:, :2], start, axis=0)
        hi = np.maximum.reduceat(g[:, :2], start, axis=0)
        cen = g[:, :2].mean(0)
        # each part's shadow lies between its plan and the plan moved away from the sun by its height
        r = ps.br + (np.hypot(ps.vx, ps.vy) * float(np.abs(dt).max()) if dt is not None and len(dt) else 0.0)
        ext = ps.z1[:, None] * (SUN[:2] / SUN[2])[None, :]
        b0 = np.minimum(ps.bc[:, :2] - r[:, None], ps.bc[:, :2] - r[:, None] - ext)
        b1 = np.maximum(ps.bc[:, :2] + r[:, None], ps.bc[:, :2] + r[:, None] - ext)
        near = (np.hypot(ps.bc[:, 0] - cen[0], ps.bc[:, 1] - cen[1]) < 90.0) & (ps.z1 > 0.3)
        cand = np.nonzero(near)[0]
        if not len(cand):
            return lit
        m = ((hi[None, :, 0] >= b0[cand, None, 0]) & (lo[None, :, 0] <= b1[cand, None, 0]) &
             (hi[None, :, 1] >= b0[cand, None, 1]) & (lo[None, :, 1] <= b1[cand, None, 1]))
        Pg = g.copy()
        Pg[:, 2] = 1e-3
        for k in np.nonzero(m.any(1))[0]:
            tl = np.nonzero(m[k])[0]
            idx = _ranges(start[tl], ends[tl] - start[tl])
            idx = idx[lit[idx] > 0]
            if not len(idx):
                continue
            i = int(cand[k])
            Pi = Pg[idx]
            if dt is not None and ps.moving(i):
                Pi = Pi - np.c_[ps.vx[i] * dt[idx], ps.vy[i] * dt[idx], np.zeros(len(idx))]
            t = ps.hit(i, Pi, np.broadcast_to(SUN, (len(idx), 3)))
            lit[idx[np.isfinite(t)]] = 0.0
        return lit


    # ---- weather (core.weather)
    def _surfaces(self, scene, ps, O, D, best, pid, dt):
        """Per ray what it meets, before lighting: kind (0 sky, 1 own body, 2 object, 3 ground), colour (the material's
        for objects and the ground, the clear shading for the own body), the world point, its normal, and for the
        ground whether the sun reaches it."""
        n = len(D)
        kind = np.zeros(n, np.int8)
        rgb = np.zeros((n, 3), np.float32)
        P = O + D * np.where(np.isfinite(best), best, 0.0)[:, None]
        nrm = np.zeros((n, 3))
        sun = np.ones(n)
        s = np.nonzero(pid == P_BODY)[0]
        if len(s):
            kind[s] = 1
            rgb[s] = (0.22 + 0.74 * self.body_shade[s, None] / 255.0) * np.array([0.80, 0.82, 0.86])
            if self.house is not None:
                from .housings import tone
                hs = s[self.house[s] >= 0]
                rgb[hs] = self.house_rgb[self.house[hs]] * tone(self.body_shade[hs])[:, None]
        s = np.nonzero(pid >= 0)[0]
        if len(s):
            kind[s] = 2
            Ps = P[s].copy()
            if dt is not None:
                mv = ps.vx[pid[s]] != 0
                Ps[mv, 0] -= ps.vx[pid[s][mv]] * dt[s][mv]
                Ps[mv, 1] -= ps.vy[pid[s][mv]] * dt[s][mv]
            nrm[s] = ps.normals(pid[s], Ps)
            rgb[s], _ = surface_shade(ps, pid[s], Ps, nrm[s])
        s = np.nonzero(pid == P_GROUND)[0]
        if len(s):
            kind[s] = 3
            g = P[s]
            sd_s, sd_d = scene.road.to_sd(g[:, 0], g[:, 1])
            fw = best[s] * self.pix_ang / np.clip(np.abs(D[s, 2]), 0.05, 1.0)
            rgb[s], _ = ground_material(scene.road, sd_s.astype(np.float32), sd_d.astype(np.float32),
                                        fw.astype(np.float32), scene.crossing_s)
            nrm[s] = (0.0, 0.0, 1.0)
            sun[s] = self._shadows(ps, g, s, None if dt is None else dt[s])
        return kind, rgb, P, nrm, sun

    def _lamp_shadows(self, ps, P, lamp_w, pose):
        """1 where a world point P (m, 3) sees the headlamp at lamp_w, 0 where an object of the scene is in between
        (parts ahead of the lamp only: the beam does not reach behind it)."""
        lit = np.ones(len(P))
        if not len(P) or not ps.n:
            return lit
        v = P - lamp_w
        dist = np.maximum(np.linalg.norm(v, axis=1), 1e-9)
        u = v / dist[:, None]
        w = ps.bc - lamp_w
        dc = np.linalg.norm(w, axis=1)
        xm, ym, pm = pose
        ahead = (w[:, 0] * math.cos(pm) + w[:, 1] * math.sin(pm)) > -ps.br
        cand = np.nonzero(ahead & (dc < 90.0) & (dc > ps.br) & (ps.z1 > 0.25))[0]
        for i in cand:
            uc = w[i] / dc[i]
            cosr = math.sqrt(max(1.0 - (ps.br[i] / dc[i]) ** 2, 0.0))
            sel = np.nonzero((u @ uc >= cosr - 1e-9) & (dist > dc[i] - ps.br[i]) & (lit > 0))[0]
            if not len(sel):
                continue
            t = ps.hit(i, lamp_w, u[sel])
            lit[sel[t < dist[sel] - 0.05]] = 0.0
        return lit

    def _weather_image(self, scene, ps, O, D, best, pid, dt, wx, noise, seed, t_mid):
        """The frame under a weather condition (see the module docstring): uint8 (h, w, 3)."""
        from . import weather as W
        c = wx.condition
        kind, rgb, P, nrm, sun = self._surfaces(scene, ps, O, D, best, pid, dt)
        n = len(D)
        T = np.exp(-wx.sigma_vis * np.where(np.isfinite(best), best, np.inf)) if wx.sigma_vis > 0 else \
            np.ones(len(D))                                                    # the sky: 0 in fog, haze, rain
        rain = c.rain_mm_h > 0
        if not c.night:
            sunny = c.ambient_lux >= 20000.0
            col = np.zeros((n, 3), np.float32)
            s = kind == 0
            el = np.clip(D[s, 2], 0, 1)[:, None]
            sky = np.array([0.80, 0.84, 0.88]) * (1 - el) + np.array([0.55, 0.66, 0.80]) * el
            col[s] = sky if sunny else 0.25 * sky + 0.75 * np.array([0.76, 0.77, 0.79])
            s = kind == 1
            col[s] = rgb[s] * (1.0 if sunny else 0.85)
            s = kind == 2
            col[s] = rgb[s] * ((0.50 + 0.55 * np.clip(nrm[s] @ SUN, 0, 1)) if sunny else
                               (0.66 + 0.16 * np.clip(nrm[s, 2], 0, 1)))[:, None]
            s = kind == 3
            col[s] = rgb[s] * ((0.62 + 0.38 * sun[s]) if sunny else np.full(int(s.sum()), 0.80))[:, None] * \
                (0.72 if rain else 1.0)
            air = np.array([0.80, 0.81, 0.83]) * (1.0 if sunny else 0.92)
            col = col * T[:, None] + air * (1.0 - T[:, None])
            img = np.full((self.h * self.w, 3), 0.16 * 255.0, np.float32)
            img[self.pix] = col * 255.0
            if noise:
                img += np.random.default_rng(seed).standard_normal(img.shape, dtype=np.float32) * noise
            out = np.clip(img, 0, 255).astype(np.uint8).reshape(self.h, self.w, 3)
        else:
            xm, ym, pm = (float(v) for v in scene.drive.pose(t_mid))
            Rz = _rz(pm)
            st = wx.settings
            lum = np.zeros((n, 3))
            lit = (kind == 2) | (kind == 3)
            Pv = (P[lit] - np.array([xm, ym, 0.0])) @ Rz
            nv = nrm[lit] @ Rz
            E = np.full(int(lit.sum()), float(c.ambient_lux))
            for Lv in wx.lamps:
                Lw = np.array([xm, ym, 0.0]) + Rz @ Lv
                El = W.illuminance(Pv, replace(c, ambient_lux=0.0), st, Lv[None, :], normals=nv)
                on = El > 1e-3
                if on.any():
                    El[on] *= self._lamp_shadows(ps, P[lit][on], Lw, (xm, ym, pm))
                E = E + El
            wet = np.where(kind[lit] == 3, 0.72 if rain else 1.0, 1.0)              # a wet road is darker
            lum[lit] = rgb[lit] * (E * wet / math.pi)[:, None]
            s = kind == 1
            lum[s] = rgb[s] * c.ambient_lux / math.pi
            s = kind == 0
            lum[s] = (0.003 + 0.01 * c.ambient_lux) * np.array([0.60, 0.66, 0.85])
            air = 0.5 * c.ambient_lux / math.pi
            lum = lum * T[:, None] + air * (1.0 - T[:, None])
            fl = W.camera_floor(self.module, self.lens, st)
            k_e = fl['electrons_per_cd_m2']
            S = lum * k_e
            # auto-exposure: the frame's bright end (99th percentile) at full scale, the gain at most NIGHT_FULL_SCALE
            # times the camera's minimum luminance
            L_fs = max(NIGHT_FULL_SCALE * fl['l_min'], float(np.percentile(lum.mean(1), 99)) if n else 0.0)
            S_fs = L_fs * k_e
            if noise:
                rng = np.random.default_rng(seed)
                S = S + rng.standard_normal(S.shape) * np.sqrt(np.maximum(S, 0.0) + st.read_noise_e ** 2)
            lin = 1.0 - np.exp(-1.2 * np.maximum(S / S_fs, 0.0))          # highlights compressed, as an HDR camera
            img = np.zeros((self.h * self.w, 3), np.float32)
            img[self.pix] = np.power(lin, 1 / 2.2) * 255.0
            out = np.clip(img, 0, 255).astype(np.uint8).reshape(self.h, self.w, 3)
        base = [int(np.atleast_1d(seed)[0])] if np.ndim(seed) else [int(seed)]
        lens_seed = base + [zlib.crc32(str(self.cam.tag).encode()), 4093]
        share = wx.drops.get(self.cam.uid, 0.0)
        if share > 0:
            out = _lens_drops(out, share, lens_seed, t_mid)
        if c.snow_mm_h > 0:
            out = _flakes(out, c.snow_mm_h, list(np.atleast_1d(seed).astype(np.int64)) + [7919], c.night)
        return out

def _lens_drops(img, share, seed, t):
    """Drops on the lens covering about `share` of the image: each a blurred, slightly brighter view of what lies
    behind it with a soft edge, seeded per camera (the same drops all drive), drifting down slowly."""
    import cv2
    h, w = img.shape[:2]
    rng = np.random.default_rng(seed)
    a, b = 0.012 * w, 0.040 * w
    mean_area = math.pi * (a * a + a * b + b * b) / 3.0
    n = int(round(share * w * h / mean_area))
    if n <= 0:
        return img
    r = rng.uniform(a, b, n)
    x = rng.uniform(0, w, n)
    y = (rng.uniform(0, h, n) + 0.01 * h * float(t) * rng.uniform(0.5, 1.5, n)) % h
    mask = np.zeros((h, w), np.float32)
    for xi, yi, ri in zip(x, y, r):
        cv2.circle(mask, (int(xi), int(yi)), max(int(ri), 1), 1.0, -1, cv2.LINE_AA)
    mask = cv2.GaussianBlur(mask, (0, 0), 0.002 * w + 0.8) * 0.92
    blur = cv2.GaussianBlur(img.astype(np.float32), (0, 0), 0.012 * w + 1.0)
    out = img.astype(np.float32) * (1.0 - mask[..., None]) + np.minimum(blur * 1.06, 255.0) * mask[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


def _flakes(img, rate, seed, night):
    """Falling snow in front of the lens: bright flakes, more for a heavier snowfall (a new set every frame)."""
    import cv2
    h, w = img.shape[:2]
    rng = np.random.default_rng(seed)
    n = int(min(rate, 30.0) * 120 * (w * h) / (640 * 512))
    over = img.copy()
    k = w / 640.0
    tone = 120 if night else 235
    for xi, yi, ri in zip(rng.uniform(0, w, n), rng.uniform(0, h, n), rng.uniform(0.6, 2.4, n) * k):
        cv2.circle(over, (int(xi), int(yi)), max(int(round(ri)), 1), (tone, tone, tone), -1, cv2.LINE_AA)
    return cv2.addWeighted(over, 0.85, img, 0.15, 0)


# =========================================================================== lidar
class LidarSim:
    """One lidar's beams: channels × columns in firing order. A spinning lidar's column j fires j/J of a period after
    each sync pulse, facing phase ∓ 360·j/J (cw/ccw); a fixed-field or non-repetitive scanner's columns sweep its
    field from one edge to the other over the frame."""

    def __init__(self, project, engine, sensor, model=None):
        m = model or project.sensor_model_for(sensor)
        self.s, self.m = sensor, m
        self.E = m.elevations()
        self.C = len(self.E)
        self.period = 1.0 / m.rate_hz
        if m.spinning:
            J = m.columns()
            sense = 1 if m.rotation == 'ccw' else -1
            ph = float(sensor.phase_deg or 0.0)
            self.az = ph + sense * 360.0 * np.arange(J) / J
        else:
            self.az = SN.scan_azimuths(m)
            J = len(self.az)
        self.J = J
        self.t_col = np.arange(J) * self.period / J
        self.dir_s = SN.directions(self.E, self.az)                    # (C, J, 3) sensor frame
        self.R_vs = SN.rotation(sensor)
        self.pos = sensor.pos().astype(float)
        self.dir_v = self.dir_s @ self.R_vs.T
        self.level = abs(sensor.pitch) < 1e-6 and abs(sensor.roll) < 1e-6
        self.body_d = np.full((self.C, J), np.inf)
        if engine is not None and engine.vehicle_geom(project) is not None:
            bd = engine.cube(self.pos).lookup(self.dir_v.reshape(-1, 3))[0].reshape(self.C, J).astype(float)
            self.body_d = np.where(np.isfinite(bd) & (bd > OWN_MOUNT_M), bd, np.inf)
        self.r_max_cap = m.range_max_m if m.range_max_m > 0 else math.inf

    def _columns(self, a_lo, a_hi, margin):
        """Columns whose azimuth (sensor frame, degrees) lies within [a_lo, a_hi] (a_hi - a_lo < 360)."""
        mid = (a_lo + a_hi) / 2
        off = (self.az - mid + 180.0) % 360.0 - 180.0
        return np.nonzero(np.abs(off) <= (a_hi - a_lo) / 2 + margin)[0]

    def _cull(self, ps, i, O_mid, yaw_mid, grow):
        """(channel slice, column indices) of the beams that may meet part i this sweep."""
        allc = slice(0, self.C)
        step = 360.0 / max(self.J, 1) + 0.05
        if self.level:
            # plan bounds in the yawed sensor frame, elevation bounds from the nearest and farthest plan distances
            zs = O_mid[2]
            yaw = yaw_mid + math.atan2(self.R_vs[1, 0], self.R_vs[0, 0])
            corners = ps.footprint(i)
            if corners is not None:
                ctr = np.array([ps.cx[i], ps.cy[i]])
                v = corners - ctr
                nv = np.linalg.norm(v, axis=1, keepdims=True)
                corners = ctr + v * (1 + grow / np.maximum(nv, 1e-9))
                rel = corners - O_mid[:2]
                c, s = math.cos(ps.yaw[i]), math.sin(ps.yaw[i])
                lx = (O_mid[0] - ps.cx[i]) * c + (O_mid[1] - ps.cy[i]) * s
                ly = -(O_mid[0] - ps.cx[i]) * s + (O_mid[1] - ps.cy[i]) * c
                dx = max(abs(lx) - ps.L[i] / 2 - grow, 0.0)
                dy = max(abs(ly) - ps.W[i] / 2 - grow, 0.0)
                rmin = math.hypot(dx, dy)
                rmax = float(np.linalg.norm(rel, axis=1).max())
            else:
                dc = math.hypot(ps.cx[i] - O_mid[0], ps.cy[i] - O_mid[1])
                rr = ps.r[i] + grow
                rmin, rmax = max(dc - rr, 0.0), dc + rr
                rel = None
            z0, z1 = ps.z0[i] - 0.05, ps.z1[i] + 0.05
            el_hi = math.atan2(z1 - zs, rmin if z1 > zs else rmax)
            el_lo = math.atan2(z0 - zs, rmin if z0 < zs else rmax)
            c0 = int(np.searchsorted(self.E, math.degrees(el_lo) - 0.3, 'left'))
            c1 = int(np.searchsorted(self.E, math.degrees(el_hi) + 0.3, 'right'))
            if c1 <= c0:
                return None, None
            if rmin <= 1e-6:
                return slice(c0, c1), np.arange(self.J)
            if rel is not None:
                az = np.degrees(np.arctan2(rel[:, 1], rel[:, 0]) - yaw)
                az0 = math.degrees(math.atan2(ps.cy[i] - O_mid[1], ps.cx[i] - O_mid[0]) - yaw)
                off = (az - az0 + 180.0) % 360.0 - 180.0
                lo, hi = az0 + off.min(), az0 + off.max()
            else:
                az0 = math.degrees(math.atan2(ps.cy[i] - O_mid[1], ps.cx[i] - O_mid[0]) - yaw)
                half = math.degrees(math.asin(min(1.0, (ps.r[i] + grow) / max(rmin + ps.r[i] + grow, 1e-9))))
                lo, hi = az0 - half, az0 + half
            if hi - lo >= 359.0:
                return slice(c0, c1), np.arange(self.J)
            return slice(c0, c1), self._columns(lo, hi, step)
        # a tilted lidar: the part's bounding sphere seen in the sensor frame
        R_ws = _rz(yaw_mid) @ self.R_vs
        q = (ps.bc[i] - O_mid) @ R_ws
        dd = float(np.linalg.norm(q))
        rad = ps.br[i] + grow
        if dd <= rad:
            return allc, np.arange(self.J)
        a = math.degrees(math.asin(min(1.0, rad / dd)))
        el = math.degrees(math.asin(q[2] / dd))
        c0 = int(np.searchsorted(self.E, el - a - 0.3, 'left'))
        c1 = int(np.searchsorted(self.E, el + a + 0.3, 'right'))
        if c1 <= c0:
            return None, None
        if abs(el) + a >= 89.0:
            return slice(c0, c1), np.arange(self.J)
        azc = math.degrees(math.atan2(q[1], q[0]))
        half = a / max(math.cos(math.radians(abs(el) + a)), 1e-3)
        if half >= 179.0:
            return slice(c0, c1), np.arange(self.J)
        return slice(c0, c1), self._columns(azc - half, azc + half, step)

    def sweep(self, scene: Scene, t0: float, noise_m=0.015, drop_self=True, seed=0, wx=None) -> dict:
        """One sweep starting at t0 (a sync pulse): every return in the sensor frame at its own firing time. wx: a
        core.weather SceneWeather (None: clear)."""
        drive = scene.drive
        tc = t0 + self.t_col
        x, y, psi = drive.pose(tc)
        c, s = np.cos(psi), np.sin(psi)
        O = np.stack([x + c * self.pos[0] - s * self.pos[1], y + s * self.pos[0] + c * self.pos[1],
                      np.full(self.J, self.pos[2])], 1)                    # (J, 3)
        dv = self.dir_v
        D = np.stack([c * dv[..., 0] - s * dv[..., 1], s * dv[..., 0] + c * dv[..., 1], dv[..., 2]], -1)  # (C, J, 3)
        t_mid = t0 + self.period / 2
        ps = scene.prims_at(t_mid)
        xm, ym, pm = (float(v) for v in drive.pose(t_mid))
        O_mid = np.array([xm + math.cos(pm) * self.pos[0] - math.sin(pm) * self.pos[1],
                          ym + math.sin(pm) * self.pos[0] + math.cos(pm) * self.pos[1], self.pos[2]])
        ego_travel = float(drive.speed(t_mid)) * self.period
        best = np.full((self.C, self.J), np.inf)
        pid = np.full((self.C, self.J), P_SKY, np.int32)
        with np.errstate(divide='ignore', invalid='ignore'):
            tg = np.where(D[..., 2] < -1e-9, -self.pos[2] / D[..., 2], np.inf)
        sel = tg < best
        best[sel], pid[sel] = tg[sel], P_GROUND
        far = self.m.range_m * math.sqrt(1.0 / 0.10) + 5.0
        dist = np.linalg.norm(ps.bc[:, :2] - O_mid[:2], axis=1)
        for i in np.nonzero(dist < min(far, 400.0) + ps.br)[0]:
            v = math.hypot(ps.vx[i], ps.vy[i])
            grow = (ego_travel + v * self.period) / 2 + 0.05
            cs, cols = self._cull(ps, i, O_mid, pm, grow)
            if cs is None or not len(cols):
                continue
            Dsub = D[cs][:, cols]                                     # (nc, nj, 3)
            nc, nj = Dsub.shape[:2]
            Osub = np.broadcast_to(O[cols][None], (nc, nj, 3))
            if v > 0:
                shift = np.c_[ps.vx[i], ps.vy[i], 0.0] * (tc[cols] - t_mid)[:, None]
                Osub = Osub - shift[None]
            t = ps.hit(i, Osub.reshape(-1, 3), Dsub.reshape(-1, 3)).reshape(nc, nj)
            cur = best[cs][:, cols]
            closer = t < cur
            if closer.any():
                rows, cc = np.nonzero(closer)
                ci = np.arange(self.C)[cs][rows]
                cj = cols[cc]
                best[ci, cj] = t[rows, cc]
                pid[ci, cj] = i
        # the own body and rig
        selfhit = self.body_d < best
        best = np.where(selfhit, self.body_d, best)
        pid = np.where(selfhit, P_BODY, pid)
        # materials, range limit, noise
        fin = np.isfinite(best)
        ci, cj = np.nonzero(fin)
        r = best[ci, cj]
        d = D[ci, cj]
        P = O[cj] + d * r[:, None]
        pk = pid[ci, cj]
        refl = np.full(len(r), REFL['body'])
        nr = np.zeros((len(r), 3))
        nr[pk == P_GROUND] = (0.0, 0.0, 1.0)
        o = pk >= 0
        if o.any():                                   # the hit on a moving part, in its mid-sweep position
            Po = P[o].copy()
            mv = ps.vx[pk[o]] != 0
            dtc = (tc[cj[o]] - t_mid)[mv]
            Po[mv, 0] -= ps.vx[pk[o][mv]] * dtc
            Po[mv, 1] -= ps.vy[pk[o][mv]] * dtc
            nr[o] = ps.normals(pk[o], Po)
        g = pk == P_GROUND
        if g.any():
            ss, dd = scene.road.to_sd(P[g, 0], P[g, 1])
            _, refl[g] = ground_material(scene.road, ss, dd, None, scene.crossing_s)
        if o.any():
            _, refl[o] = surface_shade(ps, pk[o], Po, nr[o])
        b = pk == P_BODY
        cosi = np.abs(np.sum(nr * d, axis=1))
        cosi = np.where(b, 0.8, cosi)                               # the body's normal is not known here
        rho = refl * np.clip(cosi, 0.02, 1.0)
        if wx is None:
            rmax = np.minimum(self.m.range_m * np.sqrt(rho / 0.10), self.r_max_cap)
        else:                                   # the weather's two-way extinction (core.weather)
            rmax = np.minimum(wx.lidar_rmax(self.m.range_m * np.sqrt(rho / 0.10)), self.r_max_cap)
        keep = (r >= self.m.range_min_m) & (r <= rmax)
        if drop_self:
            keep &= ~b
        ci, cj, r, rho, pk = ci[keep], cj[keep], r[keep], rho[keep], pk[keep]
        if wx is not None:
            rho = rho * np.exp(-2.0 * wx.sigma_lidar * r)
            ci, cj, r, rho, pk = self._false_returns(ci, cj, r, rho, pk, wx, seed)
        if noise_m > 0 and len(r):
            r = r + np.random.default_rng(seed).normal(0.0, noise_m, len(r))
        order = np.lexsort((ci, cj))                                 # firing order: column by column
        ci, cj, r, rho, pk = ci[order], cj[order], r[order], rho[order], pk[order]
        xyz = self.dir_s[ci, cj] * r[:, None]
        obj = pk.copy()
        obj[pk >= 0] = ps.obj[pk[pk >= 0]]
        return {'xyz': xyz.astype(np.float32), 'intensity': np.clip(np.round(255.0 * rho), 0, 255).astype(np.float32),
                'ring': ci.astype(np.uint16), 't': self.t_col[cj].astype(np.float32), 'obj': obj.astype(np.int32),
                'self': pk == P_BODY, 'rays': self.C * self.J}


    def _false_returns(self, ci, cj, r, rho, pk, wx, seed):
        """A share of the beams (core.weather's heuristic) return early from droplets or flakes, in front of what they
        would have hit: fog close in (2 m mean beyond the minimum range), rain and snow anywhere in the near zone."""
        from .weather import FALSE_NEAR_M
        p = wx.false_share
        if p <= 0:
            return ci, cj, r, rho, pk
        rng = np.random.default_rng([*(int(v) for v in np.atleast_1d(seed)), 2718])
        hit = rng.random((self.C, self.J)) < p
        bi, bj = np.nonzero(hit)
        r0 = self.m.range_min_m
        if wx.condition.visibility_m > 0:
            rf = r0 + np.minimum(rng.exponential(2.0, len(bi)), 2 * FALSE_NEAR_M)
        else:
            rf = rng.uniform(r0, max(FALSE_NEAR_M, r0 + 0.5), len(bi))
        rt = np.full((self.C, self.J), np.inf)
        rt[ci, cj] = r
        win = rf < rt[bi, bj]
        bi, bj, rf = bi[win], bj[win], rf[win]
        gone = np.zeros((self.C, self.J), bool)
        gone[bi, bj] = True
        keep = ~gone[ci, cj]
        return (np.r_[ci[keep], bi], np.r_[cj[keep], bj], np.r_[r[keep], rf],
                np.r_[rho[keep], rng.uniform(0.01, 0.06, len(bi))], np.r_[pk[keep], np.full(len(bi), P_WEATHER)])

# =========================================================================== radar
class RadarSim:
    def __init__(self, project, engine, sensor, model=None):
        self.s = sensor
        self.m = model or project.sensor_model_for(sensor)
        self.pos = sensor.pos().astype(float)
        self.R_vs = SN.rotation(sensor)
        self.cube = None
        if engine is not None and engine.vehicle_geom(project) is not None:
            self.cube = engine.cube(self.pos)

    def _body_blocks(self, dirs_v, dist):
        if self.cube is None:
            return np.zeros(len(dist), bool)
        depth = self.cube.lookup(dirs_v)[0]
        return (depth < dist - (0.02 + 0.01 * dist)) & (depth > RADAR_FASCIA_M)

    def scan(self, scene: Scene, t: float, seed=0, clutter=4, wx=None) -> dict:
        """Detections at time t in the radar frame: x y z (m), radial velocity relative to the radar v_r (m/s), the
        same compensated for the ego's motion, RCS (dBsm), dynamic property, the object hit (-1 clutter). wx: a
        core.weather SceneWeather (None: clear): each target's cross-section counts reduced by the two-way loss."""
        rng = np.random.default_rng(seed)
        drive = scene.drive
        x, y, psi = (float(v) for v in drive.pose(t))
        Rw = _rz(psi)
        ego = np.array([x, y, 0.0])
        o_w = ego + Rw @ self.pos
        kin = drive.kinematics(t)
        v_ego = float(kin['v']) * np.array([math.cos(psi), math.sin(psi), 0.0])
        w = float(kin['omega'])
        lever = Rw @ self.pos
        v_rad = v_ego + np.array([-w * lever[1], w * lever[0], 0.0])
        ps = scene.prims_at(t)
        R_ws = Rw @ self.R_vs
        reach = self.m.max_range() * SN.radar_scale(max(TARGET_RCS.values())) + 5.0
        tg, P, ids, rc = [], [], [], []
        for o in scene.objects:
            if o.rcs <= 0:
                continue
            ox, oy, oyaw = (float(v) for v in scene.actor_pose(o, t))
            if math.hypot(ox - o_w[0], oy - o_w[1]) > reach:
                continue
            cen = np.array([ox, oy, o.base + o.size[2] / 2])
            rcs = o.rcs if wx is None else \
                o.rcs * 10.0 ** (-0.2 * wx.radar_db_km * float(np.linalg.norm(cen - o_w)) / 1000.0)
            if not SN.in_field(self.s, self.m, (Rw.T @ (cen - ego))[None], rcs=rcs)[0]:
                continue
            # the reflection: the object's nearest point at the radar's height (within its box)
            c, s_ = math.cos(oyaw), math.sin(oyaw)
            lx = (o_w[0] - ox) * c + (o_w[1] - oy) * s_
            ly = -(o_w[0] - ox) * s_ + (o_w[1] - oy) * c
            L, W, H = o.size
            if o.kind in ('pedestrian', 'lamp', 'sign'):
                rr = 0.27 if o.kind == 'pedestrian' else 0.08
                dd = math.hypot(lx, ly)
                px, py = (lx / dd * rr, ly / dd * rr) if dd > 1e-6 else (0.0, 0.0)
            else:
                px, py = float(np.clip(lx, -L / 2, L / 2)), float(np.clip(ly, -W / 2, W / 2))
            pz = float(np.clip(o_w[2], o.base + 0.2 * H, o.base + 0.8 * H))
            tg.append(o)
            rc.append(rcs)
            P.append((ox + px * c - py * s_, oy + px * s_ + py * c, pz))
            ids.append(o.id)
        rows = []
        if tg:
            P = np.array(P)
            ids = np.array(ids)
            v = P - o_w
            dist = np.linalg.norm(v, axis=1)
            U = v / np.maximum(dist, 1e-9)[:, None]
            ok = (dist >= self.m.range_min_m) & ~self._body_blocks(U @ Rw, dist)
            near = np.linalg.norm(ps.bc - o_w, axis=1) - ps.br <= dist.max()
            for i in np.nonzero(near)[0]:
                if not ok.any():
                    break
                k = np.nonzero(ok)[0]
                tt = ps.hit(i, o_w, U[k])
                ok[k[(tt < dist[k] - 0.05) & (ids[k] != ps.obj[i])]] = False
            for k in np.nonzero(ok)[0]:
                rows.append((P[k], U[k], scene.actor_velocity(tg[k], t), tg[k].rcs if wx is None else rc[k],
                             tg[k].id))
        # clutter: building and wall faces in the widest zone
        if clutter and self.m.zones:
            rng_m, ah, _ = max(self.m.zones, key=lambda z: z[1])
            az = np.radians(rng.uniform(-ah, ah, 24))
            dirs = np.stack([np.cos(az), np.sin(az), np.zeros_like(az)], 1) @ R_ws.T
            best = np.full(len(dirs), np.inf)
            bobj = np.full(len(dirs), -1)
            for i in np.nonzero(np.linalg.norm(ps.bc[:, :2] - o_w[:2], axis=1) - ps.br <= rng_m)[0]:
                tt = ps.hit(i, o_w, dirs)
                cl = tt < best
                best[cl], bobj[cl] = tt[cl], ps.obj[i]
            reach_c = rng_m * SN.radar_scale(3.0)
            if wx is not None:
                from .weather import radar_range
                reach_c = radar_range(reach_c, wx.radar_db_km)
            ok = np.isfinite(best) & (best < reach_c) & (best > self.m.range_min_m)
            ok &= np.array([b >= 0 and scene.by_id[int(b)].kind in ('building', 'wall') for b in bobj])
            for k in np.nonzero(ok)[0][:clutter]:
                rows.append((o_w + dirs[k] * best[k], dirs[k], np.zeros(3), float(rng.uniform(0.3, 3.0)), -1))
        n = len(rows)
        out = {k: np.zeros(n, np.float32) for k in ('x', 'y', 'z', 'v_r', 'v_r_comp', 'vx', 'vy', 'vx_comp', 'vy_comp',
                                                     'rcs')}
        out['dyn_prop'] = np.zeros(n, np.int8)
        out['obj'] = np.zeros(n, np.int32)
        fwd = np.array([math.cos(psi), math.sin(psi), 0.0])
        for k, (Pk, u, vt, rcs, oid) in enumerate(rows):
            q = R_ws.T @ (Pk - o_w)
            rq = float(np.linalg.norm(q))
            r = rq + rng.normal(0, RADAR_NOISE['range_m'])
            az = math.atan2(q[1], q[0]) + math.radians(rng.normal(0, RADAR_NOISE['azimuth_deg']))
            el = math.asin(float(np.clip(q[2] / max(rq, 1e-9), -1, 1))) + \
                math.radians(rng.normal(0, RADAR_NOISE['elevation_deg']))
            out['x'][k] = r * math.cos(el) * math.cos(az)
            out['y'][k] = r * math.cos(el) * math.sin(az)
            out['z'][k] = r * math.sin(el)
            vr = float((vt - v_rad) @ u) + rng.normal(0, RADAR_NOISE['velocity_mps'])
            vc = float(vt @ u)
            us = R_ws.T @ u
            out['v_r'][k], out['v_r_comp'][k] = vr, vc
            out['vx'][k], out['vy'][k] = vr * us[0], vr * us[1]
            out['vx_comp'][k], out['vy_comp'][k] = vc * us[0], vc * us[1]
            out['rcs'][k] = 10 * math.log10(max(rcs, 1e-6)) + rng.normal(0, RADAR_NOISE['rcs_db'])
            out['dyn_prop'][k] = 1 if float(np.linalg.norm(vt)) < 0.3 else (2 if vt @ fwd < 0 else 0)
            out['obj'][k] = oid
        return out


# =========================================================================== IMU and GNSS
def imu_truth(drive: EgoDrive, sensor, t):
    """(specific force (n, 3) m/s², angular rate (n, 3) rad/s) in the IMU's own axes at times t, without errors:
    f = a_origin + α × r + ω × (ω × r) − g at lever arm r (the IMU's position in the vehicle frame)."""
    t = np.atleast_1d(np.asarray(t, float))
    k = drive.kinematics(t)
    r = sensor.pos().astype(float)
    w, al = k['omega'], k['alpha']
    a0 = np.stack([k['a'], k['v'] ** 2 * k['kappa'], np.zeros_like(t)], 1)
    euler = np.stack([-al * r[1], al * r[0], np.zeros_like(t)], 1)            # α ẑ × r
    cent = np.stack([-w * w * r[0], -w * w * r[1], np.zeros_like(t)], 1)       # ω ẑ × (ω ẑ × r)
    f_v = a0 + euler + cent + np.array([0.0, 0.0, G])
    R_vi = SN.rotation(sensor)
    om_v = np.stack([np.zeros_like(t), np.zeros_like(t), w], 1)
    return f_v @ R_vi, om_v @ R_vi


def imu_errors(seed, n):
    """Constant biases (accel (3,), gyro (3,)) and white noise (n, 3) each, IMU_NOISE figures."""
    rng = np.random.default_rng([int(seed), 991])
    ba = rng.normal(0, IMU_NOISE['accel_bias'], 3)
    bg = rng.normal(0, IMU_NOISE['gyro_bias'], 3)
    return ba, bg, rng.normal(0, IMU_NOISE['accel_sigma'], (n, 3)), rng.normal(0, IMU_NOISE['gyro_sigma'], (n, 3))


def antenna_world(drive: EgoDrive, sensor, t):
    """World positions (n, 3) of a GNSS antenna's phase centre at times t."""
    t = np.atleast_1d(np.asarray(t, float))
    x, y, psi = drive.pose(t)
    p = sensor.pos().astype(float)
    c, s = np.cos(psi), np.sin(psi)
    return np.stack([x + c * p[0] - s * p[1], y + s * p[0] + c * p[1], np.full(len(t), p[2])], 1)


# =========================================================================== visibility
def _box_points(L, W, H, n=4):
    """Points (k, 3) and outward normals on the faces of a box centred on its base centre (z from 0 to H)."""
    u = (np.arange(n) + 0.5) / n - 0.5
    P, N = [], []
    for ax, half, sz in ((0, L / 2, (W, H)), (1, W / 2, (L, H))):
        a, b = np.meshgrid(u * sz[0], (u + 0.5) * sz[1])
        for sgn in (1, -1):
            q = np.zeros((a.size, 3))
            q[:, ax] = sgn * half
            q[:, 1 - ax] = a.ravel()
            q[:, 2] = b.ravel()
            P.append(q)
            nn = np.zeros((a.size, 3))
            nn[:, ax] = sgn
            N.append(nn)
    a, b = np.meshgrid(u * L, u * W)
    P.append(np.c_[a.ravel(), b.ravel(), np.full(a.size, H)])
    N.append(np.tile([0.0, 0.0, 1.0], (a.size, 1)))
    return np.vstack(P), np.vstack(N)


def visibility(scene: Scene, t: float, box: dict, cams: list) -> float:
    """Share (0..1) of the points on the box's faces that face the ego which some camera sees: in its image, and
    neither a nearer object nor the body in between."""
    drive = scene.drive
    x, y, psi = (float(v) for v in drive.pose(t))
    Rw = _rz(psi)
    ego_c = np.array([x, y, 0.0]) + Rw @ np.array([1.2, 0.0, 1.0])
    L, W, H = box['size']
    P, N = _box_points(L, W, H)
    c, s = math.cos(box['yaw']), math.sin(box['yaw'])
    Rb = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    base = np.array([box['centre'][0], box['centre'][1], box['centre'][2] - H / 2])
    Pw = base + P @ Rb.T
    Nw = N @ Rb.T
    facing = np.sum(Nw * (ego_c - Pw), axis=1) > 0
    if not facing.any():
        return 0.0
    Pw, Nw = Pw[facing], Nw[facing]
    seen = np.zeros(len(Pw), bool)
    ps = scene.prims_at(t)
    oid = box['id']
    for cs in cams:
        cam_w = np.array([x, y, 0.0]) + Rw @ cs.pos
        v = Pw - cam_w
        dist = np.linalg.norm(v, axis=1)
        u = v / dist[:, None]
        front = np.sum(Nw * -v, axis=1) > 0
        dv = u @ Rw                                           # vehicle frame
        _, _, inimg = cs.project(dv)
        ok = front & inimg & ~seen
        if not ok.any():
            continue
        if cs._cube is not None:
            from_body = _body_depth(cs, dv[ok])
            ok[np.nonzero(ok)[0][from_body < dist[ok] - 0.05]] = False
        if not ok.any():
            continue
        # nearer objects on the lines of sight: only those in the cone from the camera towards the box
        bc = np.array(box['centre'])
        diag = 0.5 * math.sqrt(L * L + W * W + H * H)
        dbox = float(np.linalg.norm(bc - cam_w))
        abox = math.asin(min(1.0, diag / max(dbox, 1e-6)))
        axis = (bc - cam_w) / max(dbox, 1e-9)
        wv = ps.bc - cam_w
        dd = np.linalg.norm(wv, axis=1)
        ang = np.arccos(np.clip(wv @ axis / np.maximum(dd, 1e-9), -1, 1))
        cand = (ps.obj != oid) & (dd - ps.br <= dbox + diag) & \
            ((dd <= ps.br) | (ang <= abox + np.arcsin(np.clip(ps.br / np.maximum(dd, 1e-9), 0, 1))))
        idx = np.nonzero(ok)[0]
        for i in np.nonzero(cand)[0]:
            tt = ps.hit(i, cam_w, u[idx])
            idx = idx[~(tt < dist[idx] - 0.05)]
            if not len(idx):
                break
        seen[idx] = True
    return float(seen.mean())


def _body_depth(cs: CameraSim, dirs_v):
    """The body's distance along vehicle-frame directions from a camera (the camera's cube depth map)."""
    cube = getattr(cs, '_cube', None)
    if cube is None:
        return np.full(len(dirs_v), np.inf)
    d, _ = cube.lookup(dirs_v)
    return np.where(np.isfinite(d) & (d > NEAR_CLIP), d, np.inf)


def visibility_bin(v: float) -> str:
    """nuScenes visibility token: '1' 0-40 %, '2' 40-60 %, '3' 60-80 %, '4' 80-100 %."""
    return '1' if v < 0.4 else '2' if v < 0.6 else '3' if v < 0.8 else '4'
