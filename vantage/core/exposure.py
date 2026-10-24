"""Sun glare and dirt exposure: which cameras will have the sun in their image (washed-out frames, flare) and which
lenses and sensor windows will collect dirt, spray and rain, when, and why.

Sun glare. The sun's position is the NOAA solar position algorithm (Meeus' low-accuracy solar coordinates, good to
about 0.01°, with the NOAA refraction formula; within 0.003° of the SPA paper's worked example), vectorised: azimuth
clockwise from north and elevation with standard atmospheric refraction, for a latitude, longitude and UTC instants. A campaign is
a date range, a daily driving window in local time, a step in minutes and in days, and the route's headings (every
heading equally likely, or a weighted list from a GPX or CSV track: the time spent on each heading). Every
(time, heading) sample with the sun above the horizon puts the sun in the vehicle frame (x forward, y left, z up;
the vehicle stands level, driving straight on that heading), then in each camera's optical frame
(BODY_TO_OPTICAL @ R.T). The sun is *in the image* when the exact lens model maps it inside the sensor rectangle and
the lens' image circle (Intrinsics.project_dirs, max_theta) and the body does not hide it (the engine's occlusion
cube from the camera's centre: a finite depth of 5 cm or more in the sun's direction is the body or a rig part;
nearer is the lens' own housing). *Near the image* is outside the field but within flare_margin_deg of its edge
(the angle to the nearest point of the field's outline): flare risk. *Low sun ahead* is in the image with the sun
less than low_sun_deg above the horizon: the worst case, where it is also in the scene's brightest part. Shares
are of all driving time (night included) and weighted by the hours each sample stands for.

A lidar's indicator is the share of driving time with the sun inside its field (its elevation band and, for a
fixed-field lidar, its azimuth band) and not hidden from it: solar background lowers its range on dark targets in
that direction. Radars do not see sunlight.

Dirt and spray is an empirical heuristic, not a flow simulation (see DIRT_MODEL): a 0–1 score per lens, lidar window
facet, radar fascia and body face from wheel spray, the rear wake, frontal impact, the window's attitude and its
height above the road.
"""
from __future__ import annotations

import csv
import datetime as _dt
import io
import json
import math
import os
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field

import numpy as np

from . import sensors as S
from .analysis import camera_frames
from .contour import iso_segments
from .lens import r_of_theta, resolve
from .mathutil import BODY_TO_OPTICAL, rot_zyx
from .mesh import ROLE_GLASS, ROLE_IGNORED, ROLE_MOUNTABLE, ROLE_WHEELS

SCHEMA = 'vantage.exposure/1'
FRAME = 'vehicle (ISO 8855, rear axle): x forward, y left, z up'
MONTHS = ('Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec')
COMPASS = ('N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW')
OWN_MOUNT_M = 0.05                  # occluder hits nearer than this are the lens' or window's own housing
MAX_SAMPLES = 1_500_000             # (time, heading) samples per study; the day step is raised beyond it
MAX_HEADINGS = 360
SKY_STEP = 2.0                      # sky chart cells, degrees (azimuth and elevation)
SKY_AZ = -180.0 + SKY_STEP * (np.arange(int(360 / SKY_STEP)) + 0.5)      # vehicle frame, ccw from +x
SKY_EL = SKY_STEP * (np.arange(int(90 / SKY_STEP)) + 0.5)
TRACK_MINUTES = 5.0
TRACK_HEADINGS = tuple(range(0, 360, 45))                                 # plus each camera's worst heading
IMAGE_MASK = (64, 48)               # body mask over the image, columns × rows
WORST_SHARE = 0.5                   # a month or hour is among the worst at half the worst one's share or more
SUN_WARN, SUN_SERIOUS, LOW_SERIOUS = 0.01, 0.10, 0.05
FLARE_WARN = 0.20
LIDAR_WARN = 0.40
SEVERITIES = ('critical', 'serious', 'warning', 'good')

ASSUMPTIONS = ('Clear sky every day; a flat horizon (no buildings, trees or terrain); the sun as a point; the '
               'vehicle level, driving straight with the given heading distribution at every hour of the window; '
               'local time is UTC + the offset all year unless a time zone is given (then its daylight saving time '
               'is followed). The dirt and spray score is an empirical heuristic, not a flow simulation or a '
               'measurement.')
DIRT_MODEL = ('Score = min(1, Σ weight × factor), each factor 0–1. Wheel spray (weight 0.8): behind and beside each '
              'wheel, decaying over 1.5 m behind the tyre and 0.4 m ahead of it, with height above 0.3 m (most spray '
              'stays below 0.8 m) and inboard of the tyre, slightly stronger outboard. Rear wake (0.5): windows and '
              'faces facing backwards (normal x < −0.6), more on a boxy rear and lower down. Frontal impact (0.4): '
              'facing forwards (normal x > 0.6): insects, rain and stone chips, most at bumper height, more at the '
              'roof\'s leading edge than on the windscreen. Lens attitude (0.15): a window facing upwards holds drops '
              'and dust, a level one some, one tilted 25° or more down sheds water. Height (0.15): exp(−z / 0.7 m). '
              'A recessed or hooded lens scores 0.6 ×. Low below 0.3, moderate below 0.5, high from 0.5. A camera '
              'takes the worse of its optical axis and the surface it is mounted on for wake and impact.')
DIRT_WEIGHTS = {'spray': 0.8, 'wake': 0.5, 'frontal': 0.4, 'attitude': 0.15, 'height': 0.15}
DIRT_LABEL = {'spray': 'wheel spray', 'wake': 'rear wake', 'frontal': 'frontal impact', 'attitude': 'lens attitude',
              'height': 'low mounting'}
DIRT_CLASSES = ('low', 'moderate', 'high')
DIRT_LIMITS = (0.30, 0.50)
RECESS_FACTOR = 0.6
SPRAY_BEHIND_M, SPRAY_AHEAD_M, SPRAY_TOP_M, SPRAY_FALL_M, SPRAY_INBOARD_M = 1.5, 0.4, 0.3, 0.45, 0.35
TYRE_HALF_M = 0.10
HEIGHT_SCALE_M = 0.7
MOUNT_SEARCH_M = 0.25               # a camera this close to the body takes its surface's normal too
SUN_RAMP = ['#fff4cc', '#ffe39a', '#fdcb5c', '#f9ab31', '#ee861f', '#d9601b', '#b5401a', '#8a2b14']
DIRT_RAMP = ['#ece7da', '#ddcda6', '#cbb07a', '#b78f52', '#9c6e35', '#7c5024', '#5a3618']
CLASS_TONE = {'low': 'good', 'moderate': 'warning', 'high': 'critical'}
STATUS_HEX = {'good': '#0ca30c', 'warning': '#fab219', 'serious': '#ec835a', 'critical': '#d03b3b'}

# accepted range of every number in the settings (values outside are clamped)
LIMITS = {'lat': (-90.0, 90.0), 'lon': (-180.0, 180.0), 'utc_offset_h': (-12.0, 14.0), 'hour_from': (0.0, 23.75),
          'hour_to': (0.25, 24.0), 'minutes': (1.0, 120.0), 'days': (1.0, 60.0), 'heading_step': (1.0, 90.0),
          'flare_margin_deg': (0.0, 45.0), 'low_sun_deg': (0.0, 45.0)}
DEFAULT_START, DEFAULT_END = '2026-01-01', '2026-12-31'
MAX_SPAN_DAYS = 5 * 366
_TZ = re.compile(r'^[A-Za-z0-9_+\-/]{1,64}$')


# =========================================================================== settings
def _clean_headings(v):
    """'uniform', or [[heading_deg 0..360, weight ≥ 0], ...] with some weight; None when not usable."""
    if isinstance(v, str):
        return 'uniform' if v.strip().lower() == 'uniform' else None
    if not isinstance(v, (list, tuple)):
        return None
    out = []
    for e in v[:MAX_HEADINGS]:
        try:
            h, w = float(e[0]), float(e[1])
        except (TypeError, ValueError, IndexError, KeyError):
            continue
        if math.isfinite(h) and math.isfinite(w) and w >= 0:
            out.append([round(h % 360.0, 3), round(w, 6)])
    return out if out and sum(w for _, w in out) > 0 else None


def _date(v, default: str) -> _dt.date:
    try:
        return _dt.date.fromisoformat(str(v).strip()[:10])
    except (TypeError, ValueError):
        return _dt.date.fromisoformat(default)


@dataclass
class ExposureSettings:
    lat: float = 48.14                 # degrees north (Munich)
    lon: float = 11.58                 # degrees east
    utc_offset_h: float = 1.0          # local time = UTC + this (all year, unless timezone is given)
    timezone: str = ''                 # an IANA zone (Europe/Berlin) to follow daylight saving time; '' = fixed offset
    start: str = DEFAULT_START         # ISO dates, both included
    end: str = DEFAULT_END
    hour_from: float = 8.0             # daily driving window, local time (to < from: over midnight)
    hour_to: float = 18.0
    minutes: float = 15.0              # time step within the day
    days: float = 7.0                  # step between sampled days
    headings: object = 'uniform'       # 'uniform' or [[heading_deg (clockwise from north), weight], ...]
    heading_step: float = 10.0         # 'uniform': one heading every this many degrees
    route: dict = field(default_factory=dict)   # the last imported track: {name, headings, lat, lon, km, hours}
    flare_margin_deg: float = 10.0     # outside the field but this close to its edge: flare risk
    low_sun_deg: float = 15.0          # in the image below this elevation: low sun ahead
    recessed: list = field(default_factory=list)  # uids of cameras and sensors whose lens is recessed or hooded

    def to_dict(self):
        d = asdict(self)
        d['headings'] = self.headings if isinstance(self.headings, str) else [list(e) for e in self.headings]
        return d

    @staticmethod
    def from_dict(d: dict | None) -> 'ExposureSettings':
        s = ExposureSettings()
        d = d if isinstance(d, dict) else {}
        for k, (lo, hi) in LIMITS.items():
            if k in d:
                try:
                    v = float(d[k])
                except (TypeError, ValueError):
                    continue
                if math.isfinite(v):
                    setattr(s, k, min(max(v, lo), hi))
        tz = d.get('timezone')
        if isinstance(tz, str) and (tz == '' or _TZ.match(tz.strip())):
            s.timezone = tz.strip()
        a = _date(d.get('start', DEFAULT_START), DEFAULT_START)
        b = _date(d.get('end', DEFAULT_END), DEFAULT_END)
        if b < a:
            a, b = b, a
        if (b - a).days >= MAX_SPAN_DAYS:
            b = a + _dt.timedelta(days=MAX_SPAN_DAYS - 1)
        s.start, s.end = a.isoformat(), b.isoformat()
        h = _clean_headings(d.get('headings', 'uniform'))
        s.headings = h if h is not None else 'uniform'
        r = d.get('route')
        if isinstance(r, dict):
            rh = _clean_headings(r.get('headings'))
            if isinstance(rh, list):
                s.route = {'name': str(r.get('name') or 'route')[:80], 'headings': rh}
                for k in ('lat', 'lon', 'km', 'hours'):
                    try:
                        v = float(r.get(k))
                    except (TypeError, ValueError):
                        continue
                    if math.isfinite(v):
                        s.route[k] = v
        rec = d.get('recessed')
        if isinstance(rec, (list, tuple)):
            s.recessed = sorted({str(u)[:40] for u in rec if isinstance(u, str) and u})[:256]
        return s

    def dates(self) -> tuple[_dt.date, _dt.date]:
        return _date(self.start, DEFAULT_START), _date(self.end, DEFAULT_END)

    def window_h(self) -> float:
        """Hours of the daily driving window (24 when from and to are the same time)."""
        span = (self.hour_to - self.hour_from) % 24.0
        return span if span > 1e-9 else 24.0

    def heading_table(self) -> tuple[np.ndarray, np.ndarray]:
        """(headings in degrees clockwise from north, weights summing to 1)."""
        if isinstance(self.headings, list):
            h = np.array([e[0] for e in self.headings], float)
            w = np.array([e[1] for e in self.headings], float)
            return h, w / w.sum()
        n = max(int(round(360.0 / self.heading_step)), 1)
        return np.arange(n) * 360.0 / n, np.full(n, 1.0 / n)


# =========================================================================== solar position (NOAA)
_EPOCH = _dt.date(1970, 1, 1).toordinal()


def julian_day(when) -> np.ndarray:
    """Julian day of UTC instants: a datetime (naive = UTC), an ISO string, numpy datetime64 or arrays of them."""
    if isinstance(when, _dt.datetime):
        if when.tzinfo is not None:
            when = when.astimezone(_dt.timezone.utc).replace(tzinfo=None)
        when = np.datetime64(when)
    t = np.asarray(when, 'datetime64[ns]').astype('int64')
    return t / 86400e9 + 2440587.5


def _jd0(day: _dt.date) -> float:
    """Julian day of 00:00 UTC on a date."""
    return day.toordinal() - _EPOCH + 2440587.5


def solar_ephemeris(jd) -> dict:
    """The sun's apparent coordinates at Julian day(s) jd (UT; ΔT ignored, under 0.001°): declination and right
    ascension (degrees), equation of time (minutes), distance (AU), apparent longitude and obliquity (degrees)."""
    jd = np.asarray(jd, float)
    T = (jd - 2451545.0) / 36525.0
    L0 = np.mod(280.46646 + T * (36000.76983 + T * 0.0003032), 360.0)
    M = 357.52911 + T * (35999.05029 - 0.0001537 * T)
    e = 0.016708634 - T * (0.000042037 + 0.0000001267 * T)
    Mr = np.radians(M)
    C = (np.sin(Mr) * (1.914602 - T * (0.004817 + 0.000014 * T)) + np.sin(2 * Mr) * (0.019993 - 0.000101 * T)
         + np.sin(3 * Mr) * 0.000289)
    true_long = L0 + C
    R = 1.000001018 * (1 - e * e) / (1 + e * np.cos(np.radians(M + C)))
    om = np.radians(125.04 - 1934.136 * T)
    lam = true_long - 0.00569 - 0.00478 * np.sin(om)
    eps0 = 23.0 + (26.0 + (21.448 - T * (46.815 + T * (0.00059 - T * 0.001813))) / 60.0) / 60.0
    eps = eps0 + 0.00256 * np.cos(om)
    lr, er = np.radians(lam), np.radians(eps)
    ra = np.mod(np.degrees(np.arctan2(np.cos(er) * np.sin(lr), np.cos(lr))), 360.0)
    dec = np.degrees(np.arcsin(np.sin(er) * np.sin(lr)))
    y = np.tan(er / 2) ** 2
    L0r = np.radians(L0)
    eot = 4.0 * np.degrees(y * np.sin(2 * L0r) - 2 * e * np.sin(Mr) + 4 * e * y * np.sin(Mr) * np.cos(2 * L0r)
                           - 0.5 * y * y * np.sin(4 * L0r) - 1.25 * e * e * np.sin(2 * Mr))
    return {'declination': dec, 'right_ascension': ra, 'equation_of_time_min': eot, 'distance_au': R,
            'apparent_longitude': np.mod(lam, 360.0), 'obliquity': eps}


def refraction(elev_deg) -> np.ndarray:
    """Atmospheric refraction (degrees) for a true elevation, the NOAA formula (standard pressure and temperature)."""
    el = np.asarray(elev_deg, float)
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        te = np.tan(np.radians(el))
        r = np.where(el > 85.0, 0.0,
                     np.where(el > 5.0, 58.1 / te - 0.07 / te ** 3 + 0.000086 / te ** 5,
                              np.where(el > -0.575, 1735.0 + el * (-518.2 + el * (103.4 + el * (-12.79 + el * 0.711))),
                                       -20.772 / te)))
    return np.where(np.isfinite(r), r, 0.0) / 3600.0


def sun_position(jd, lat, lon, refract=True) -> tuple[np.ndarray, np.ndarray]:
    """(azimuth clockwise from north, elevation) in degrees of the sun at Julian day(s) jd (UTC) seen from latitude
    lat (north positive) and longitude lon (east positive); the elevation includes refraction unless refract=False."""
    jd = np.asarray(jd, float)
    eph = solar_ephemeris(jd)
    minutes = np.mod(jd - 0.5, 1.0) * 1440.0                       # UTC minutes of the day
    tst = np.mod(minutes + eph['equation_of_time_min'] + 4.0 * lon, 1440.0)
    ha = np.radians(tst / 4.0 - 180.0)
    la, de = math.radians(lat), np.radians(eph['declination'])
    cz = np.clip(math.sin(la) * np.sin(de) + math.cos(la) * np.cos(de) * np.cos(ha), -1.0, 1.0)
    el = 90.0 - np.degrees(np.arccos(cz))
    # Meeus 13.5 with atan2 (well defined at the poles, where the NOAA acos form divides by cos(lat) = 0)
    az = np.mod(np.degrees(np.arctan2(np.sin(ha), np.cos(ha) * math.sin(la) - np.tan(de) * math.cos(la))) + 180.0,
                360.0)
    if refract:
        el = el + refraction(el)
    return az, el


def sun_at(when, lat, lon, refract=True) -> tuple[float, float]:
    """(azimuth, elevation) in degrees at one UTC instant (datetime, ISO string or datetime64)."""
    az, el = sun_position(julian_day(when), lat, lon, refract)
    return float(az), float(el)


def vehicle_dirs(az_deg, el_deg, heading_deg) -> np.ndarray:
    """Unit vectors to the sun in the vehicle frame (x forward, y left, z up) for sun azimuths/elevations (N,) and
    vehicle headings (K,) clockwise from north: shape (N, K, 3). The sun is at H − A counter-clockwise from +x."""
    rel = np.radians(np.asarray(heading_deg, float)[None, :] - np.asarray(az_deg, float)[:, None])
    e = np.radians(np.asarray(el_deg, float))[:, None]
    ce = np.cos(e)
    return np.stack([ce * np.cos(rel), ce * np.sin(rel), np.broadcast_to(np.sin(e), rel.shape)], axis=-1)


# =========================================================================== the campaign
@dataclass
class Campaign:
    jd: np.ndarray          # (N,) UTC Julian day of each time sample
    w: np.ndarray           # (N,) hours of driving the sample stands for
    month: np.ndarray       # (N,) 0..11, local
    hour: np.ndarray        # (N,) 0..23, local
    local_h: np.ndarray     # (N,) local hour, fractional
    day_step: int
    n_days: int             # sampled days
    total_days: int
    step_h: float
    note: str = ''


def _tz_offsets(st: ExposureSettings, days) -> tuple[np.ndarray, str]:
    """UTC offset (hours) at local noon of each day: the time zone's when one is given and known, else the fixed
    offset."""
    if st.timezone:
        try:
            from zoneinfo import ZoneInfo
            z = ZoneInfo(st.timezone)
            return np.array([_dt.datetime(d.year, d.month, d.day, 12, tzinfo=z).utcoffset().total_seconds() / 3600.0
                             for d in days]), ''
        except Exception:                              # unknown zone, or no time zone data on this system
            return (np.full(len(days), st.utc_offset_h),
                    f'Time zone "{st.timezone}" is not known here: UTC{st.utc_offset_h:+g} h is used all year.')
    return np.full(len(days), st.utc_offset_h), ''


def campaign(st: ExposureSettings, n_headings: int = 1) -> Campaign:
    """The time samples of the campaign: every day_step days from start to end, every `minutes` through the driving
    window (sample centres). The day step is raised when days × times × headings would exceed MAX_SAMPLES."""
    d0, d1 = st.dates()
    total = (d1 - d0).days + 1
    span = st.window_h()
    n_t = max(1, int(round(span * 60.0 / st.minutes)))
    step_h = span / n_t
    step_d = max(1, int(round(st.days)))
    note = ''
    need = math.ceil(total / step_d) * n_t * max(n_headings, 1)
    if need > MAX_SAMPLES:
        step_d = max(step_d, math.ceil(total * n_t * max(n_headings, 1) / MAX_SAMPLES))
        while step_d < total and math.ceil(total / step_d) * n_t * max(n_headings, 1) > MAX_SAMPLES:
            step_d += 1
        note = f'Day step raised to {step_d} d to keep the study under {MAX_SAMPLES / 1e6:g} M samples.'
    starts = list(range(0, total, step_d))
    days = [d0 + _dt.timedelta(days=i) for i in starts]
    day_w = np.array([min(step_d, total - i) for i in starts], float)
    off, tz_note = _tz_offsets(st, days)
    note = ' '.join(t for t in (note, tz_note) if t)
    t = st.hour_from + (np.arange(n_t) + 0.5) * step_h                 # local hours from the day's midnight
    jd = (np.array([_jd0(d) for d in days])[:, None] + (t[None, :] - off[:, None]) / 24.0).ravel()
    w = np.repeat(day_w * step_h, n_t)
    roll = np.floor(t / 24.0).astype(int)                                 # a window past midnight: the next day
    local_h = np.tile(np.mod(t, 24.0), len(days))
    m0 = np.array([d.month - 1 for d in days])                             # the window lasts 24 h at most
    m1 = np.array([(d + _dt.timedelta(days=1)).month - 1 for d in days])
    month = np.where(roll[None, :] > 0, m1[:, None], m0[:, None]).ravel()
    return Campaign(jd, w, month, np.floor(local_h).astype(int) % 24, local_h, step_d, len(days), total, step_h, note)


# =========================================================================== geometry helpers
def _hidden(cube, dirs) -> np.ndarray:
    """Directions the body or a rig part blocks from the cube's centre (its own housing, nearer than 5 cm, aside)."""
    if cube is None or len(dirs) == 0:
        return np.zeros(len(dirs), bool)
    depth, _ = cube.lookup(dirs)
    return np.isfinite(depth) & (depth >= OWN_MOUNT_M)


def field_outline(intr, n=96) -> np.ndarray:
    """Optical-frame unit vectors along the outline of a camera's field: the sensor rectangle's border inside the
    image circle and the image circle inside the rectangle."""
    w, h = float(intr.width), float(intr.height)
    t = (np.arange(n) + 0.5) / n
    us = np.concatenate([t * w, np.full(n, w), (1 - t) * w, np.zeros(n)])
    vs = np.concatenate([np.zeros(n), t * h, np.full(n, h), (1 - t) * h])
    d, th = intr.unproject(us, vs)
    keep = th <= intr.max_theta + 1e-9
    phi = np.radians(np.arange(0.0, 360.0, 1.0))
    mt = intr.max_theta
    c = np.stack([math.sin(mt) * np.cos(phi), math.sin(mt) * np.sin(phi), np.full(len(phi), math.cos(mt))], axis=-1)
    u, v, _, _ = intr.project_dirs(c)
    inside = (u >= 0) & (u <= w) & (v >= 0) & (v <= h)
    return np.vstack([d[keep], c[inside]])


def _near_field(dirs, cand, B, margin_rad) -> np.ndarray:
    """dirs within margin_rad of the field outline B (both unit vectors in the same frame), for the candidates."""
    out = np.zeros(len(dirs), bool)
    idx = np.nonzero(cand)[0]
    if len(idx) == 0 or len(B) == 0:
        return out
    cmin = math.cos(margin_rad)
    Bt = np.ascontiguousarray(B.T)
    for s in range(0, len(idx), 16384):
        j = idx[s:s + 16384]
        out[j] = (dirs[j] @ Bt).max(axis=1) >= cmin
    return out


def _sky_dirs() -> np.ndarray:
    a, e = np.radians(SKY_AZ)[None, :], np.radians(SKY_EL)[:, None]
    return np.stack([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.broadcast_to(np.sin(e), (len(SKY_EL),
                                                                                                len(SKY_AZ)))], -1)


def _camera_margin(intr, opt) -> np.ndarray:
    """The lens' signed margin to its field's edge (positive inside; −1 behind a rectilinear lens)."""
    u, v, th, ok = intr.project_dirs(opt)
    m = np.asarray(intr.margin(u, v, th), float)
    if intr.projection == 'rectilinear':
        m = np.where(opt[..., 2] > 1e-6, m, -1.0)
    return np.where(np.isfinite(m), m, -1.0)


def _lidar_angles(s, dirs):
    """(elevation, azimuth) in degrees of vehicle-frame directions in a sensor's own frame."""
    q = dirs @ S.rotation(s)
    el = np.degrees(np.arcsin(np.clip(q[..., 2], -1.0, 1.0)))
    az = np.degrees(np.arctan2(q[..., 1], q[..., 0]))
    return el, az


def _lidar_margin(model, el, az) -> np.ndarray:
    """Signed margin (degrees) to a lidar's field: its elevation band and, for a fixed field, its azimuth band."""
    lo, hi = model.v_fov()
    m = np.minimum(el - lo, hi - el)
    half = model.az_half()
    return m if half >= 180.0 else np.minimum(m, half - np.abs(az))


def _finite_pose(o) -> bool:
    return all(math.isfinite(float(getattr(o, k))) for k in ('x', 'y', 'z', 'yaw', 'pitch', 'roll'))


# =========================================================================== result
@dataclass
class ItemExposure:
    """One camera, lidar or radar."""
    uid: str
    tag: str
    kind: str                     # camera | lidar | radar
    model: str
    slot: int                     # a camera's colour slot; a sensor's index among its kind
    pos: tuple
    ypr: tuple
    sun: dict | None = None       # see _sun_stats; None for a radar
    dirt: dict = field(default_factory=dict)
    sky_margin: np.ndarray | None = None     # (len(SKY_EL), len(SKY_AZ)) margin to the field (> 0 inside)
    sky_blocked: np.ndarray | None = None    # (len(SKY_EL), len(SKY_AZ)) bool: the body hides that direction
    outline: np.ndarray | None = None        # (S, 2, 2) field outline segments in (azimuth, elevation) degrees
    image: dict | None = None     # camera: {'w', 'h', 'cx', 'cy', 'rx', 'ry', 'body' (rows, cols) bool, 'horizon'}
    tracks: dict = field(default_factory=dict)   # heading key ('worst', '0', '45', …) -> [track dict]
    worst_heading: float | None = None


@dataclass
class ExposureResult:
    settings: ExposureSettings
    items: list                   # [ItemExposure]
    sky_density: np.ndarray       # (len(SKY_EL), len(SKY_AZ)) share of driving time with the sun in each cell
    hours_cols: list              # local hours of the day with driving (the heatmap's columns, in window order)
    month_hours: np.ndarray       # (12, 24) driving hours in each cell
    dirt_faces: dict              # {'tri': (F, 3, 3) float32, 'score': (F,) float32 (nan: wheels), 'glass': (F,)}
    dirt_info: dict               # wheel centres, body height, boxiness …
    findings: list
    metrics: dict
    variant: str = ''
    variant_uid: str = ''

    def item(self, tag_or_uid):
        return next((i for i in self.items if tag_or_uid in (i.tag, i.uid)), None)

    def worst(self) -> str:
        return self.findings[0]['severity'] if self.findings else 'good'


# =========================================================================== sun statistics
def _runs(sel, circular=True) -> list:
    """Runs of True in a 1-D mask as (first, last) index pairs, joining the ends when circular."""
    sel = np.asarray(sel, bool)
    n = len(sel)
    if not sel.any():
        return []
    if sel.all():
        return [(0, n - 1)]
    runs, start = [], None
    for i in range(n):
        if sel[i] and start is None:
            start = i
        if not sel[i] and start is not None:
            runs.append([start, i - 1])
            start = None
    if start is not None:
        runs.append([start, n - 1])
    if circular and len(runs) > 1 and runs[0][0] == 0 and runs[-1][1] == n - 1:
        runs[0][0] = runs[-1][0]
        runs.pop()
    return [tuple(r) for r in runs]


def _run_members(r, n):
    a, b = r
    return list(range(a, b + 1)) if a <= b else list(range(a, n)) + list(range(0, b + 1))


def _worst_runs(share, n_keep):
    """The runs of cells at WORST_SHARE of the worst cell's share or more, the n_keep with the most share."""
    s = np.nan_to_num(np.asarray(share, float))
    if s.max(initial=0.0) <= 0:
        return []
    runs = _runs(s >= WORST_SHARE * s.max())
    runs.sort(key=lambda r: -sum(s[i] for i in _run_members(r, len(s))))
    return runs[:n_keep]


def months_text(runs) -> str:
    return ' and '.join(MONTHS[a] if a == b else f'{MONTHS[a]}–{MONTHS[b]}' for a, b in runs) or '—'


def hours_text(runs) -> str:
    return ' and '.join(f'{a:02d}–{(b + 1) % 24:02d} h' for a, b in runs) or '—'


def _heading_span(headings, shares, step):
    """(first, last, best) heading of the run of neighbouring headings (no more than max(15°, step) apart) around
    the worst one whose share is at WORST_SHARE of it or more; None when no heading has any."""
    s = np.nan_to_num(np.asarray(shares, float))
    if not len(s) or s.max() <= 0:
        return None
    order = np.argsort(headings)
    h, v = np.asarray(headings, float)[order], s[order]
    n, k = len(h), int(np.argmax(v))
    gap = max(15.0, step) + 1e-6
    sel = v >= WORST_SHARE * v[k]
    lo = hi = k
    for _ in range(n - 1):                              # walk down, then up, while neighbours qualify
        j = (lo - 1) % n
        if j == hi or not sel[j] or (h[lo] - h[j]) % 360.0 > gap:
            break
        lo = j
    for _ in range(n - 1):
        j = (hi + 1) % n
        if j == lo or not sel[j] or (h[j] - h[hi]) % 360.0 > gap:
            break
        hi = j
    if (hi - lo) % n == n - 1:
        return 0.0, 360.0, float(h[k])                  # every heading
    return float(h[lo]), float(h[hi]), float(h[k])


def heading_span_text(span) -> str:
    if span is None:
        return '—'
    lo, hi, best = span
    if hi - lo >= 360.0:
        return 'any heading'
    return f'{best:.0f}° ({compass(best)})' if lo == hi else f'{lo:.0f}–{hi:.0f}° ({compass(best)})'


def compass(deg) -> str:
    return COMPASS[int(round((float(deg) % 360.0) / 45.0)) % 8]


def _sun_stats(mask, ctx, low=None, near=None, blocked=None) -> dict:
    """Shares of driving time for a boolean mask over the samples above the horizon."""
    W, total = ctx['W'], ctx['total_h']
    w = np.where(mask, W, 0.0)
    cell = np.bincount(ctx['cell'], w, 288)
    with np.errstate(invalid='ignore', divide='ignore'):
        mh = np.where(ctx['cell_h'] > 0, cell / ctx['cell_h'], np.nan).reshape(12, 24)
        months = np.where(ctx['month_h'] > 0, cell.reshape(12, 24).sum(1) / ctx['month_h'], np.nan)
        hours = np.where(ctx['hour_h'] > 0, cell.reshape(12, 24).sum(0) / ctx['hour_h'], np.nan)
        heads = np.bincount(ctx['head'], w, len(ctx['headings'])) / np.maximum(ctx['head_h'], 1e-12)
    share = float(w.sum() / total) if total > 0 else 0.0
    order = [int(k) for k in np.argsort(-heads) if heads[k] > 0][:3]
    mruns, hruns = _worst_runs(months, 2), _worst_runs(hours, 2)
    out = {'share': share, 'hours': float(w.sum()), 'month_hour': mh, 'months': months, 'hours_of_day': hours,
           'headings': heads, 'worst_months': mruns, 'worst_hours': hruns,
           'worst_headings': [(float(ctx['headings'][k]), float(heads[k])) for k in order],
           'worst_span': _heading_span(ctx['headings'], heads, ctx['step'])}
    for name, m in (('low', low), ('near', near), ('blocked', blocked)):
        out[name] = float(np.where(m, W, 0.0).sum() / total) if m is not None and total > 0 else 0.0
    return out


# =========================================================================== sun tracks
def _track_days(st: ExposureSettings) -> list:
    y = st.dates()[0].year
    return [('Mar 20 · equinox', _dt.date(y, 3, 20)), ('Jun 21 · solstice', _dt.date(y, 6, 21)),
            ('Dec 21 · solstice', _dt.date(y, 12, 21))]


def _day_samples(st: ExposureSettings):
    """For the track days: [(label, date, local hours (T,), sun azimuth (T,), elevation (T,))]."""
    out = []
    for label, d in _track_days(st):
        off, _ = _tz_offsets(st, [d])
        t = np.arange(0.0, 24.0, TRACK_MINUTES / 60.0)
        az, el = sun_position(_jd0(d) + (t - off[0]) / 24.0, st.lat, st.lon)
        out.append((label, d, t, az, el))
    return out


def _in_window(st, hours) -> np.ndarray:
    return np.mod(hours - st.hour_from, 24.0) < st.window_h() - 1e-9


def _tracks(st, days, heading, intr, W2O, cube) -> list:
    """The sun's path through a camera's image on each track day at one heading: per day the samples with the sun
    above the horizon, their pixel (u, v), local hour, whether the lens maps them into the image, the body hides
    them, and they fall in the driving window."""
    out = []
    for label, d, t, az, el in days:
        up = el > 0
        if not up.any():
            out.append({'label': label, 'date': d.isoformat(), 'heading': float(heading), 'u': np.zeros(0),
                        'v': np.zeros(0), 'hour': np.zeros(0), 'el': np.zeros(0), 'ok': np.zeros(0, bool),
                        'hidden': np.zeros(0, bool), 'in_window': np.zeros(0, bool)})
            continue
        dirs = vehicle_dirs(az[up], el[up], [heading])[:, 0, :]
        u, v, _, ok = intr.project_dirs(dirs @ W2O.T)
        out.append({'label': label, 'date': d.isoformat(), 'heading': float(heading), 'u': u, 'v': v,
                    'hour': t[up], 'el': el[up], 'ok': ok, 'hidden': _hidden(cube, dirs),
                    'in_window': _in_window(st, t[up])})
    return out


def _image_info(intr, W2O, cube) -> dict:
    """The image frame for drawing: size, image circle (centre, radii in px), where the body covers the image
    (a coarse mask) and the horizon's pixel polylines."""
    rad = float(r_of_theta(intr.max_theta, intr.f_mm, intr.projection, intr.kb)) / intr.pixel_mm
    nc, nr = IMAGE_MASK
    us = (np.arange(nc) + 0.5) / nc * intr.width
    vs = (np.arange(nr) + 0.5) / nr * intr.height
    U, V = np.meshgrid(us, vs)
    d, th = intr.unproject(U.ravel(), V.ravel())
    dw = d @ W2O                                           # optical -> vehicle
    body = (_hidden(cube, dw) & (th <= intr.max_theta)).reshape(nr, nc)
    a = np.radians(np.arange(-180.0, 180.0, 1.0))
    hz = np.stack([np.cos(a), np.sin(a), np.zeros(len(a))], -1)
    u, v, _, ok = intr.project_dirs(hz @ W2O.T)
    lines, cur = [], []
    for i in range(len(a)):
        if ok[i]:
            cur.append((float(u[i]), float(v[i])))
        elif cur:
            lines.append(cur)
            cur = []
    if cur:
        if lines and ok[0] and lines[0]:
            lines[0] = cur + lines[0]                    # the run wraps through ±180°
        else:
            lines.append(cur)
    return {'w': intr.width, 'h': intr.height, 'cx': intr.cx, 'cy': intr.cy, 'rx': rad, 'ry': rad * intr.aspect,
            'body': body, 'horizon': [ln for ln in lines if len(ln) > 1]}


# =========================================================================== dirt and spray
def wheel_centres(project, mesh=None) -> tuple[list, str]:
    """[(x, y, z, tyre radius)] of the wheels and where they come from: the vehicle's wheel centres, else the
    mesh's wheel parts, else a typical car of the mesh's size. The tyre radius is the centre's height."""
    wc = getattr(project.vehicle, 'wheel_centres', {}) or {}
    out = []
    for k in ('FL', 'FR', 'RL', 'RR'):
        v = wc.get(k)
        try:
            x, y, z = (float(c) for c in v[:3])
        except (TypeError, ValueError, IndexError):
            continue
        if all(math.isfinite(c) for c in (x, y, z)):
            out.append((x, y, z, max(z, 0.2)))
    if out:
        return out, 'the vehicle\'s wheel centres'
    if mesh is not None and mesh.n_tris:
        wm = mesh.part_mask({ROLE_WHEELS})
        if wm.any() and any(p.role == ROLE_WHEELS for p in mesh.parts):
            c = mesh.centroids[wm]
            xm = 0.5 * (c[:, 0].min() + c[:, 0].max())
            for front in (True, False):
                for left in (True, False):
                    q = c[((c[:, 0] > xm) == front) & ((c[:, 1] > 0) == left)]
                    if len(q):
                        z = 0.5 * (q[:, 2].min() + q[:, 2].max())
                        out.append((float(q[:, 0].mean()), float(q[:, 1].mean()), float(z), max(float(z), 0.2)))
            if out:
                return out, 'the mesh\'s wheel parts'
        lo, hi = mesh.bbox
        L, Wd = float(hi[0] - lo[0]), float(hi[1] - lo[1])
        xs = (float(lo[0]) + 0.19 * L, float(hi[0]) - 0.19 * L)
        ys = 0.5 * Wd - 0.12
        return [(x, s * ys, 0.31, 0.31) for x in xs for s in (1, -1)], 'a typical car of the body\'s size (assumed)'
    return [(x, s * 0.78, 0.31, 0.31) for x in (0.0, 2.7) for s in (1, -1)], 'a typical car (assumed: no body)'


def _body_info(mesh) -> dict:
    """Body height, length and width, and how boxy its rear is: the frontal area of rear-facing faces (normal x below
    −0.6) in its rear 30 %, over its width × height."""
    if mesh is None or not mesh.n_tris:
        return {'height': 1.5, 'boxiness': 0.4, 'box_factor': 0.55 + 0.45 * min(0.4 / 0.6, 1.0), 'length': 4.5,
                'width': 1.8, 'assumed': True}
    keep = mesh.part_mask({ROLE_MOUNTABLE, ROLE_GLASS})
    lo, hi = mesh.bbox
    L, Wd, H = float(hi[0] - lo[0]), float(hi[1] - lo[1]), float(hi[2])
    n, c, a = mesh.normals, mesh.centroids, mesh.areas
    rear = keep & (c[:, 0] < lo[0] + 0.3 * L) & (n[:, 0] < -0.6)
    box = float((a[rear] * -n[rear, 0]).sum() / max(Wd * max(H, 0.3), 1e-6))
    box = min(max(box, 0.0), 1.0)
    return {'height': max(H, 0.3), 'boxiness': box, 'box_factor': 0.55 + 0.45 * min(box / 0.6, 1.0),
            'length': L, 'width': Wd, 'assumed': False}


def dirt_factors(P, normals, attitude, wheels, body) -> dict:
    """Each dirt factor (0–1) for points P (K, 3) whose windows face `normals` (a list of (K, 3) arrays: the worst
    counts for wake and impact) and whose lens or window points along `attitude` (K, 3)."""
    P = np.atleast_2d(np.asarray(P, float))
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    spray = np.zeros(len(P))
    for xw, yw, zw, r in wheels:
        dx = x - xw
        f_long = np.where(dx < -r, np.exp((dx + r) / SPRAY_BEHIND_M),
                          np.where(dx > r, np.exp(-(dx - r) / SPRAY_AHEAD_M), 1.0))
        same = (np.sign(y) == np.sign(yw)) | (np.abs(y) < 1e-9)
        delta = np.where(same, np.abs(y) - abs(yw), -np.abs(y) - abs(yw))      # outboard of the wheel's centre line
        edge = delta + TYRE_HALF_M                                              # from the tyre's inner edge
        f_lat = np.where(edge >= 0, 1.0, np.exp(edge / SPRAY_INBOARD_M)) * (0.75 + 0.25 * np.clip(edge / 0.4, 0, 1))
        f_h = np.where(z <= SPRAY_TOP_M, 1.0, np.exp(-((z - SPRAY_TOP_M) / SPRAY_FALL_M) ** 2))
        spray = np.maximum(spray, f_long * f_lat * f_h)
    H = body['height']
    rear = np.zeros(len(P))
    front = np.zeros(len(P))
    for n in normals:
        n = np.atleast_2d(np.asarray(n, float))
        ok = np.isfinite(n).all(axis=1)
        nx = np.where(ok, n[:, 0], 0.0)
        rear = np.maximum(rear, np.clip((-nx - 0.6) / 0.4, 0.0, 1.0))
        front = np.maximum(front, np.clip((nx - 0.6) / 0.4, 0.0, 1.0))
    wake = rear * (1.0 - 0.6 * np.clip(z / H, 0.0, 1.0)) * body['box_factor']
    bumper = np.exp(-((z - 0.5) / 0.4) ** 2)
    roof_edge = 0.55 * np.clip((z - 0.85 * H) / (0.15 * H), 0.0, 1.0)
    frontal = front * np.maximum(np.maximum(bumper, roof_edge), 0.35)
    az = np.atleast_2d(np.asarray(attitude, float))[:, 2]
    att = np.clip(0.3 + 0.7 * az / math.sin(math.radians(45.0)), 0.0, 1.0)
    height = np.exp(-np.maximum(z, 0.0) / HEIGHT_SCALE_M)
    return {'spray': spray, 'wake': wake, 'frontal': frontal, 'attitude': att, 'height': height}


def dirt_score(f: dict) -> np.ndarray:
    return np.minimum(1.0, sum(DIRT_WEIGHTS[k] * np.asarray(f[k], float) for k in DIRT_WEIGHTS))


def dirt_class(score: float) -> str:
    return 'low' if score < DIRT_LIMITS[0] else 'moderate' if score < DIRT_LIMITS[1] else 'high'


def _mount_normal(mesh, p) -> np.ndarray:
    """The outward normal of the body surface a lens sits on (nan when there is none within MOUNT_SEARCH_M)."""
    if mesh is None or not mesh.n_tris:
        return np.full(3, np.nan)
    keep = mesh.part_mask({ROLE_MOUNTABLE, ROLE_GLASS})
    q, f, dist = mesh.closest_point(p, keep if not keep.all() else None)
    if f < 0 or not math.isfinite(dist) or dist > MOUNT_SEARCH_M:
        return np.full(3, np.nan)
    n = np.array(mesh.normals[f], float)
    out = np.asarray(p, float) - q
    if dist > 0.005 and float(n @ out) < 0:            # a face wound inwards: the side the lens is on is outward
        n = -n
    return n


def _advice(kind, cls, dominant, facet='') -> str:
    if kind == 'radar':
        return ('Radar: dirt and water films matter less at 77 GHz; keep the fascia clear of snow and ice (a heated '
                'fascia in winter).')
    what = 'window' if kind == 'lidar' else 'lens'
    where = f' (its {facet})' if facet else ''
    if cls == 'high':
        fix = {'spray': 'mount it higher than 0.8 m or ahead of the wheel arch, or shield it from the wheel',
               'wake': 'move it out of the rear face\'s wake (higher, or on the side), or add a deflector',
               'frontal': 'a protective hood helps against insects and stone chips',
               'attitude': 'tilt it down so water runs off',
               'height': 'mount it higher'}.get(dominant, '')
        tool = 'washer nozzle and air knife on the window' if kind == 'lidar' else 'washer jet or air knife on the lens'
        return f'Clean while driving{where}: {tool}' + (f'; or {fix}.' if fix else '.')
    if cls == 'moderate':
        return f'Wipe the {what}{where} at every stop; a hydrophobic coating helps.'
    return 'No cleaning beyond the daily check.'


def _dirt_entry(kind, P, normals, attitude, wheels, body, recessed, facets=None) -> dict:
    f = dirt_factors(P, normals, attitude, wheels, body)
    sc = dirt_score(f) * (RECESS_FACTOR if recessed else 1.0)
    k = int(np.argmax(sc))
    contrib = {n: float(DIRT_WEIGHTS[n] * f[n][k]) for n in DIRT_WEIGHTS}
    dominant = max(contrib, key=contrib.get)
    cls = dirt_class(float(sc[k]))
    facet = facets[k] if facets else ''
    d = {'score': float(sc[k]), 'class': cls, 'factors': {n: float(f[n][k]) for n in DIRT_WEIGHTS},
         'contributions': contrib, 'dominant': dominant, 'recessed': bool(recessed), 'facet': facet,
         'advice': _advice(kind, cls, dominant, facet)}
    if facets:
        d['facets'] = {fc: float(s) for fc, s in zip(facets, sc)}
    return d


FACET_NAMES = ('front', 'front-left', 'left', 'rear-left', 'rear', 'rear-right', 'right', 'front-right')


def _sensor_dirt(s, model, mesh, wheels, body, recessed) -> dict:
    R = S.rotation(s)
    c = np.array([s.x, s.y, s.z], float)
    if model.is_lidar and model.az_half() >= 180.0:
        a = np.radians(np.arange(8) * 45.0)
        n = np.stack([np.cos(a), np.sin(a), np.zeros(8)], -1) @ R.T
        rad = max(float(model.size_mm[0]), 1.0) / 2000.0
        return _dirt_entry('lidar', c + rad * n, [n], n, wheels, body, recessed, list(FACET_NAMES))
    n = R[:, 0][None, :]
    normals = [n] if model.is_lidar else [n, _mount_normal(mesh, c)[None, :]]
    return _dirt_entry(model.kind, c[None, :], normals, n, wheels, body, recessed)


def _face_dirt(mesh, wheels, body) -> dict:
    if mesh is None or not mesh.n_tris:
        return {'tri': np.zeros((0, 3, 3), np.float32), 'score': np.zeros(0, np.float32), 'glass': np.zeros(0, bool)}
    n = mesh.normals
    f = dirt_factors(mesh.centroids, [n], n, wheels, body)
    sc = dirt_score(f).astype(np.float32)
    roles = np.array([p.role for p in mesh.parts] or [ROLE_MOUNTABLE], object)[mesh.face_part]
    sc[(roles == ROLE_WHEELS) | (roles == ROLE_IGNORED)] = np.nan
    glass = (roles == ROLE_GLASS)
    return {'tri': mesh.tri.astype(np.float32), 'score': sc, 'glass': glass, 'factors': f}


# =========================================================================== the study
def _project_for(project, variant):
    if variant is None or variant is project.variant:
        return project
    import copy
    q = copy.copy(project)
    q.variants = list(project.variants) + ([] if any(v is variant for v in project.variants) else [variant])
    q.active_uid = variant.uid
    return q


def run(project, engine=None, settings: ExposureSettings | None = None, variant=None, progress=None,
        cancel=None) -> ExposureResult | None:
    """Sun glare and dirt exposure of a variant's visible cameras, lidars and radars (the active variant by default).
    engine: the analysis engine for occlusion by the body, rig and housings (None: nothing hides). Returns None when
    cancelled."""
    t0 = time.perf_counter()
    p = _project_for(project, variant)
    v = p.variant
    st = settings or ExposureSettings.from_dict(getattr(p.analysis, 'exposure', None))

    def prog(f, text=''):
        if progress is not None:
            progress(min(max(float(f), 0.0), 1.0), text)

    def stop():
        return cancel is not None and cancel()

    vg = engine.vehicle_geom(p) if engine is not None else None
    mesh = vg.mesh if vg is not None else p.vehicle.mesh
    cube_of = (lambda pos: engine.cube(pos)) if vg is not None else (lambda pos: None)
    headings, wk = st.heading_table()
    cp = campaign(st, len(headings))
    az, el = sun_position(cp.jd, st.lat, st.lon)
    up = el > 0
    nu = np.nonzero(up)[0]
    K = len(headings)
    dirs = vehicle_dirs(az[nu], el[nu], headings).reshape(-1, 3)
    el_rep = np.repeat(el[nu], K)
    W = (cp.w[nu][:, None] * wk[None, :]).ravel()
    cell_all = cp.month * 24 + cp.hour
    cell_h = np.bincount(cell_all, cp.w, 288)
    total_h = float(cp.w.sum())
    ctx = {'W': W, 'total_h': total_h, 'cell': np.repeat(cell_all[nu], K), 'cell_h': cell_h,
           'month_h': cell_h.reshape(12, 24).sum(1), 'hour_h': cell_h.reshape(12, 24).sum(0),
           'head': np.tile(np.arange(K), len(nu)), 'head_h': total_h * wk, 'headings': headings,
           'step': 360.0 / K if isinstance(st.headings, str) else 15.0}
    # the sun's positions in the vehicle frame (the same for every sensor)
    if len(dirs):
        vaz = np.degrees(np.arctan2(dirs[:, 1], dirs[:, 0]))
        vel = np.degrees(np.arcsin(np.clip(dirs[:, 2], -1, 1)))
        dens, _, _ = np.histogram2d(vel, vaz, bins=[np.arange(0, 90.0 + 1e-9, SKY_STEP),
                                                    np.arange(-180.0, 180.0 + 1e-9, SKY_STEP)], weights=W)
        dens = dens / max(total_h, 1e-12)
        if isinstance(st.headings, str):        # every heading equally likely: spread over the heading step
            k = max(int(round(360.0 / K / SKY_STEP)), 1)
            if k > 1:
                dens = sum(np.roll(dens, s, axis=1) for s in range(-(k // 2), k - k // 2)) / k
    else:
        dens = np.zeros((len(SKY_EL), len(SKY_AZ)))
    sky = _sky_dirs()
    skyf = sky.reshape(-1, 3)
    days = _day_samples(st)
    cams = [c for c in v.cameras if c.visible and _finite_pose(c)]
    sens = [s for s in v.sensors if s.visible and _finite_pose(s) and s.kind in ('lidar', 'radar')]
    skipped = [o.tag for o in list(v.cameras) + [s for s in v.sensors if s.kind in ('lidar', 'radar')]
               if o.visible and not _finite_pose(o)]
    wheels, wheel_src = wheel_centres(p, mesh)
    body = _body_info(p.vehicle.mesh if p.vehicle.mesh is not None else mesh)
    rec = set(st.recessed)
    margin = math.radians(st.flare_margin_deg)
    n_items = max(len(cams) + len(sens), 1)
    items = []
    prog(0.02, 'sun positions')
    for i, cam in enumerate(cams):
        if stop():
            return None
        prog(0.05 + 0.85 * i / n_items, cam.tag)
        intr = resolve(p.module_for(cam), p.lens_for(cam))
        _, W2O = camera_frames(cam)
        cube = cube_of(cam.pos())
        hid = _hidden(cube, dirs)
        _, _, th, ok = intr.project_dirs(dirs @ W2O.T)
        img = ok & ~hid
        B = field_outline(intr) @ W2O                  # optical -> vehicle
        near = _near_field(dirs, ~ok & ~hid & (th <= intr.max_theta + margin), B, margin)
        sun = _sun_stats(img, ctx, low=img & (el_rep < st.low_sun_deg), near=near, blocked=ok & hid)
        sun['pitch_down'] = {}
        if sun['share'] > 0:
            for dp in (5.0, 10.0):
                R2 = rot_zyx(cam.yaw, cam.pitch + dp, cam.roll)
                ok2 = intr.project_dirs(dirs @ (BODY_TO_OPTICAL @ R2.T).T)[3]
                sun['pitch_down'][dp] = float(np.where(ok2 & ~hid, W, 0.0).sum() / total_h)
        worst_h = sun['worst_headings'][0][0] if sun['worst_headings'] else None
        tracks = {str(h): _tracks(st, days, h, intr, W2O, cube) for h in TRACK_HEADINGS}
        if worst_h is not None:
            tracks['worst'] = _tracks(st, days, worst_h, intr, W2O, cube)
        # the sky chart: the field and what the body hides, in the vehicle frame
        sky_m = _camera_margin(intr, skyf @ W2O.T).reshape(sky.shape[:2])
        blk = _hidden(cube, skyf).reshape(sky.shape[:2])
        axis = (W2O.T @ np.array([0.0, 0.0, 1.0]))[None, :]                # optical axis in the vehicle frame
        dirt = _dirt_entry('camera', cam.pos()[None, :], [axis, _mount_normal(mesh, cam.pos())[None, :]], axis,
                           wheels, body, cam.uid in rec)
        items.append(ItemExposure(
            cam.uid, cam.tag, 'camera', f'{p.module_for(cam).name} · {p.lens_for(cam).name}', int(cam.slot),
            tuple(map(float, cam.pos())), (float(cam.yaw), float(cam.pitch), float(cam.roll)), sun, dirt,
            sky_m.astype(np.float32), blk, iso_segments(sky_m, 0.0, -180.0, 0.0, SKY_STEP),
            _image_info(intr, W2O, cube), tracks, worst_h))
    count = {'lidar': 0, 'radar': 0}
    for j, s in enumerate(sens):
        if stop():
            return None
        prog(0.05 + 0.85 * (len(cams) + j) / n_items, s.tag)
        m = p.sensor_model_for(s)
        kind = 'lidar' if m.is_lidar else 'radar'
        it = ItemExposure(s.uid, s.tag, kind, f'{m.vendor} {m.name}'.strip(), count[kind],
                          (float(s.x), float(s.y), float(s.z)), (float(s.yaw), float(s.pitch), float(s.roll)))
        count[kind] += 1
        it.dirt = _sensor_dirt(s, m, mesh, wheels, body, s.uid in rec)
        if m.is_lidar:
            cube = cube_of(s.pos())
            hid = _hidden(cube, dirs)
            e_s, a_s = _lidar_angles(s, dirs)
            inf = _lidar_margin(m, e_s, a_s) >= 0
            it.sun = _sun_stats(inf & ~hid, ctx, low=inf & ~hid & (el_rep < st.low_sun_deg), blocked=inf & hid)
            it.worst_heading = it.sun['worst_headings'][0][0] if it.sun['worst_headings'] else None
            e_k, a_k = _lidar_angles(s, sky)
            it.sky_margin = _lidar_margin(m, e_k, a_k).astype(np.float32)
            it.sky_blocked = _hidden(cube, skyf).reshape(sky.shape[:2])
            it.outline = iso_segments(it.sky_margin, 0.0, -180.0, 0.0, SKY_STEP)
        items.append(it)
    if stop():
        return None
    prog(0.92, 'dirt map')
    faces = _face_dirt(mesh, wheels, body)
    cols = [int(h) for h in _window_hours(st)]
    res = ExposureResult(st, items, dens.astype(np.float32), cols, cell_h.reshape(12, 24), faces,
                         {'wheels': [tuple(round(c, 3) for c in w) for w in wheels], 'wheels_from': wheel_src, **body},
                         [], {}, v.name, v.uid)
    res.metrics = {'seconds': time.perf_counter() - t0, 'time_samples': int(len(cp.jd)), 'headings': int(K),
                   'samples': int(len(cp.jd) * K), 'days_sampled': cp.n_days, 'day_step': cp.day_step,
                   'campaign_days': cp.total_days, 'driving_hours': total_h,
                   'daylight_share': float(cp.w[up].sum() / total_h) if total_h > 0 else 0.0,
                   'note': cp.note, 'skipped': skipped}
    res.findings = _findings(res)
    prog(1.0, 'done')
    return res


def _window_hours(st: ExposureSettings) -> list:
    """Local hours of the day the driving window touches, in window order."""
    a = int(math.floor(st.hour_from))
    n = int(math.ceil(st.hour_from + st.window_h() - 1e-9)) - a
    return [(a + i) % 24 for i in range(min(max(n, 1), 24))]


# =========================================================================== findings
def _pct(v, d=1) -> str:
    return f'{100.0 * v:.{d}f} %'


def _finding(out, sev, kind, message, items=()):
    out.append({'severity': sev, 'kind': kind, 'message': message, 'tags': [i.tag for i in items],
                'uids': [i.uid for i in items]})


def _findings(res: ExposureResult) -> list:
    st, out = res.settings, []
    m = res.metrics
    cams = [i for i in res.items if i.kind == 'camera']
    if not res.items:
        _finding(out, 'warning', 'empty', 'The variant has no visible camera, lidar or radar to check.')
    if m['daylight_share'] <= 0:
        _finding(out, 'good', 'daylight', 'The sun never rises above the horizon in the driving window: no glare.')
    for it in cams:
        s = it.sun
        if s['share'] < SUN_WARN and s['low'] < SUN_WARN:
            continue
        sev = 'serious' if s['share'] >= SUN_SERIOUS or s['low'] >= LOW_SERIOUS else 'warning'
        when = f'mostly {months_text(s["worst_months"])} at {hours_text(s["worst_hours"])}'
        heads = heading_span_text(s['worst_span']) if s['worst_span'] else ''
        msg = (f'{it.tag} has the sun in its image {_pct(s["share"])} of the driving time ({s["hours"]:,.0f} h), '
               f'{when}' + (f', driving {heads}' if heads else '') + f'; low sun ahead {_pct(s["low"])}. Expect '
               'washed-out frames and flare.')
        pd = s.get('pitch_down') or {}
        if 10.0 in pd and pd[10.0] < s['share'] * 0.8:
            msg += (f' Pitched 5° further down it is {_pct(pd[5.0])}, 10° down {_pct(pd[10.0])}; a lens hood also '
                    'helps.')
        else:
            msg += ' A lens hood or a sun shade above the lens helps.'
        _finding(out, sev, 'sun', msg, [it])
    for it in cams:
        s = it.sun
        if s['near'] >= FLARE_WARN:
            _finding(out, 'warning', 'flare', f'{it.tag}: the sun is within {st.flare_margin_deg:g}° of its image '
                                              f'{_pct(s["near"])} of the driving time: flare risk, a lens hood helps.',
                     [it])
    calm = [i for i in cams if i.sun['share'] < SUN_WARN and i.sun['low'] < SUN_WARN]
    if calm:
        _finding(out, 'good', 'sun', ', '.join(i.tag for i in calm) + (' has' if len(calm) == 1 else ' have')
                 + f' the sun in the image under {_pct(SUN_WARN, 0)} of the driving time.', calm)
    for it in res.items:
        if it.kind == 'lidar' and it.sun is not None and it.sun['share'] >= LIDAR_WARN:
            _finding(out, 'warning', 'lidar_sun',
                     f'{it.tag}: the sun is inside its field {_pct(it.sun["share"])} of the driving time (indicator): '
                     'solar background shortens its range on dark targets towards the sun.', [it])
    for it in res.items:
        d = it.dirt
        if it.kind == 'radar' or d.get('class') != 'high':
            continue
        _finding(out, 'warning', 'dirt',
                 f'{it.tag}: high dirt exposure ({d["score"]:.2f}, mostly {DIRT_LABEL[d["dominant"]]}'
                 + (f' on its {d["facet"]}' if d.get('facet') else '') + f'). {d["advice"]}', [it])
    clean = [i for i in res.items if i.kind != 'radar' and i.dirt.get('class') == 'low']
    if clean:
        _finding(out, 'good', 'dirt', ', '.join(i.tag for i in clean) + ': low dirt exposure.', clean)
    if m.get('skipped'):
        _finding(out, 'warning', 'pose', f'{", ".join(m["skipped"])}: the pose is not a number; left out.')
    order = {s: i for i, s in enumerate(SEVERITIES)}
    out.sort(key=lambda f: order.get(f['severity'], 9))
    return out


# =========================================================================== the Issues list (dirt only: cheap)
_issue_cache: dict = {}


def issues(project, vg=None) -> list:
    """High dirt exposure of a visible camera or lidar for the Issues list (warnings). The sun glare needs the full
    study (a few hundred ms) and is shown on the page only. Cached on the inputs."""
    from . import rules as R
    v = project.variant
    mesh = vg.mesh if vg is not None else project.vehicle.mesh
    st = ExposureSettings.from_dict(getattr(project.analysis, 'exposure', None))
    objs = [o for o in list(v.cameras) + [s for s in v.sensors if s.kind == 'lidar'] if o.visible and _finite_pose(o)]
    key = (id(mesh), getattr(mesh, 'n_tris', 0), json.dumps(project.vehicle.wheel_centres, sort_keys=True,
                                                             default=str),
           tuple((o.uid, o.tag, getattr(o, 'model_id', ''), getattr(o, 'module_id', ''), o.x, o.y, o.z, o.yaw,
                  o.pitch, o.roll) for o in objs), tuple(st.recessed))
    hit = _issue_cache.get('key')
    if hit is not None and hit == key:
        return list(_issue_cache['out'])
    wheels, _ = wheel_centres(project, mesh)
    body = _body_info(project.vehicle.mesh if project.vehicle.mesh is not None else mesh)
    out = []
    rec = set(st.recessed)
    for o in objs:
        if hasattr(o, 'module_id'):
            _, W2O = camera_frames(o)
            axis = (W2O.T @ np.array([0.0, 0.0, 1.0]))[None, :]
            d = _dirt_entry('camera', o.pos()[None, :], [axis, _mount_normal(mesh, o.pos())[None, :]], axis, wheels,
                            body, o.uid in rec)
        else:
            d = _sensor_dirt(o, project.sensor_model_for(o), mesh, wheels, body, o.uid in rec)
        if d['class'] == 'high':
            out.append(R.Issue('warning', f'{o.tag}: high dirt exposure ({d["score"]:.2f}, mostly '
                                          f'{DIRT_LABEL[d["dominant"]]}; a heuristic). {d["advice"]}', o.tag,
                               [o.uid]))
    _issue_cache.clear()
    _issue_cache.update(key=key, out=out)
    return list(out)


# =========================================================================== route headings (GPX / CSV)
def _parse_time(text):
    t = str(text or '').strip()
    if not t:
        return None
    try:
        return float(t)                                  # seconds
    except ValueError:
        pass
    try:
        d = _dt.datetime.fromisoformat(t.replace('Z', '+00:00'))
    except ValueError:
        return None
    if d.tzinfo is None:
        d = d.replace(tzinfo=_dt.timezone.utc)
    return d.timestamp()


def parse_gpx(data) -> list:
    """Track segments of a GPX file (bytes or text): [(lat (N,), lon (N,), times (N,) seconds or nan)]. Segments of
    each <trkseg> (and each <rte>) stay apart."""
    if isinstance(data, str):
        data = data.encode('utf-8')
    root = ET.fromstring(data)
    local = lambda e: e.tag.rsplit('}', 1)[-1]          # noqa: E731

    def points(parent, name):
        lat, lon, ts = [], [], []
        for e in parent.iter():
            if local(e) != name:
                continue
            try:
                a, b = float(e.get('lat')), float(e.get('lon'))
            except (TypeError, ValueError):
                continue
            if not (math.isfinite(a) and math.isfinite(b) and -90 <= a <= 90):
                continue
            tt = next((c.text for c in e if local(c) == 'time'), None)
            t = _parse_time(tt)
            lat.append(a)
            lon.append(b)
            ts.append(np.nan if t is None else t)
        return (np.array(lat), np.array(lon), np.array(ts, float)) if lat else None
    segs = [points(e, 'trkpt') for e in root.iter() if local(e) == 'trkseg']
    segs += [points(e, 'rtept') for e in root.iter() if local(e) == 'rte']
    segs = [s for s in segs if s is not None]
    if not segs:
        s = points(root, 'trkpt')
        segs = [s] if s is not None else []
    return segs


def parse_csv_track(text) -> list:
    """A CSV track: rows lat,lon[,time] (time in seconds or ISO 8601); a header row is skipped. One segment."""
    if isinstance(text, bytes):
        text = text.decode('utf-8', 'replace')
    lat, lon, ts = [], [], []
    for row in csv.reader(io.StringIO(text)):
        if len(row) < 2:
            continue
        try:
            a, b = float(row[0]), float(row[1])
        except ValueError:
            continue                                    # a header or a note
        if not (math.isfinite(a) and math.isfinite(b) and -90 <= a <= 90):
            continue
        t = _parse_time(row[2]) if len(row) > 2 else None
        lat.append(a)
        lon.append(b)
        ts.append(np.nan if t is None else t)
    return [(np.array(lat), np.array(lon), np.array(ts, float))] if lat else []


def route_headings(segments, bin_deg=10.0, name='route') -> dict:
    """The heading distribution of a track: each segment's initial bearing (clockwise from north) weighted by the
    time spent on it when every point has a time (segments slower than 1 m/s are stops and dropped; a gap longer than
    ten typical steps counts as ten), else by its length (segments under 1 m dropped). Binned to bin_deg; also the
    track's centre (lat, lon), length and duration."""
    hs, ws, lats, lons = [], [], [], []
    km, hours = 0.0, 0.0
    timed = all(np.isfinite(t).all() for _, _, t in segments) and any(len(t) > 1 for _, _, t in segments)
    for la, lo, t in segments:
        lats.append(la)
        lons.append(lo)
        if len(la) < 2:
            continue
        p1, p2 = np.radians(la[:-1]), np.radians(la[1:])
        dl = np.radians(lo[1:] - lo[:-1])
        brg = np.mod(np.degrees(np.arctan2(np.sin(dl) * np.cos(p2),
                                           np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl))), 360.0)
        a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
        dist = 2 * 6371008.8 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
        km += float(dist.sum()) / 1000.0
        if timed:
            dt = np.diff(t)
            good = dt > 0
            if good.any():
                dt = np.minimum(dt, 10.0 * float(np.median(dt[good])))
            keep = good & (dist >= 1.0) & (dist / np.where(good, dt, 1.0) >= 1.0)
            w = np.where(keep, dt, 0.0)
            hours += float(np.where(good, dt, 0.0).sum()) / 3600.0
        else:
            keep = dist >= 1.0
            w = np.where(keep, dist, 0.0)
        hs.append(brg[keep])
        ws.append(w[keep])
    h = np.concatenate(hs) if hs else np.zeros(0)
    w = np.concatenate(ws) if ws else np.zeros(0)
    if not len(h) or w.sum() <= 0:
        raise ValueError('The track has no moving segment to take a heading from.')
    nb = max(int(round(360.0 / bin_deg)), 1)
    idx = np.rint(h / (360.0 / nb)).astype(int) % nb
    acc = np.bincount(idx, w, nb)
    acc = acc / acc.sum()
    la, lo = np.concatenate(lats), np.radians(np.concatenate(lons))
    return {'name': str(name)[:80], 'headings': [[round(k * 360.0 / nb, 3), round(float(acc[k]), 6)]
                                                 for k in range(nb) if acc[k] > 0],
            'lat': round(float(la.mean()), 6),
            'lon': round(float(np.degrees(np.arctan2(np.sin(lo).mean(), np.cos(lo).mean()))), 6),
            'km': round(km, 3), 'hours': round(hours, 3) if timed else None, 'weighting': 'time' if timed else
            'distance', 'segments': int(len(h))}


def read_route(path, bin_deg=10.0) -> dict:
    """route_headings of a GPX (.gpx) or CSV file."""
    if os.path.getsize(path) > 100_000_000:
        raise ValueError('The track file is over 100 MB.')
    with open(path, 'rb') as fh:
        data = fh.read()
    name = os.path.basename(path)
    if path.lower().endswith('.gpx') or data.lstrip()[:5] in (b'<?xml', b'<gpx '):
        try:
            segs = parse_gpx(data)
        except ET.ParseError as e:
            raise ValueError(f'Not a readable GPX file: {e}') from None
    else:
        segs = parse_csv_track(data)
    if not segs:
        raise ValueError('No track points found (GPX <trkpt lat lon>, or CSV rows lat,lon[,time]).')
    return route_headings(segs, bin_deg, name)


def parse_heading_text(text) -> list | None:
    """'0:3, 90:1, 180:3' (heading in degrees : weight; the weight may be left out for 1) as a weighted list."""
    out = []
    for part in re.split(r'[,;\s]+', str(text or '').strip()):
        if not part:
            continue
        m = re.fullmatch(r'(-?\d+(?:\.\d+)?)°?(?::(\d+(?:\.\d+)?))?', part)
        if not m:
            return None
        out.append([float(m.group(1)) % 360.0, float(m.group(2)) if m.group(2) else 1.0])
    return _clean_headings(out)


def heading_text(headings) -> str:
    if not isinstance(headings, list):
        return ''
    return ', '.join(f'{h:g}:{w:.3g}' for h, w in headings)


# =========================================================================== output
def _r(v, d=4):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return round(v, d) + 0.0 if math.isfinite(v) else None


def _sun_doc(s) -> dict | None:
    if s is None:
        return None
    return {'in_field_pct': _r(100 * s['share'], 3), 'hours': _r(s['hours'], 1),
            'near_pct': _r(100 * s['near'], 3), 'low_sun_pct': _r(100 * s['low'], 3),
            'blocked_by_body_pct': _r(100 * s['blocked'], 3),
            'worst_months': months_text(s['worst_months']), 'worst_hours': hours_text(s['worst_hours']),
            'worst_headings': [{'heading_deg': _r(h, 1), 'pct': _r(100 * v, 3)} for h, v in s['worst_headings']],
            'worst_heading_span_deg': [_r(s['worst_span'][0], 1), _r(s['worst_span'][1], 1)] if s['worst_span']
            else None,
            'months_pct': [_r(100 * v, 3) for v in s['months']],
            'hours_of_day_pct': [_r(100 * v, 3) for v in s['hours_of_day']],
            'month_hour_pct': [[_r(100 * v, 3) for v in row] for row in s['month_hour']],
            'pitched_down_pct': {f'{k:g}': _r(100 * v, 3) for k, v in (s.get('pitch_down') or {}).items()}}


def _dirt_doc(d) -> dict:
    return {'score': _r(d['score'], 3), 'class': d['class'], 'dominant': DIRT_LABEL.get(d.get('dominant'), ''),
            'factors': {k: _r(v, 3) for k, v in d['factors'].items()},
            'contributions': {k: _r(v, 3) for k, v in d['contributions'].items()},
            'recessed': d['recessed'], 'worst_facet': d.get('facet') or None,
            'facets': {k: _r(v, 3) for k, v in d['facets'].items()} if d.get('facets') else None,
            'advice': d['advice']}


def _track_doc(tracks) -> list:
    out = []
    for t in tracks:
        pts = [[_r(u, 1), _r(v, 1), _r(h, 3), bool(hd), bool(iw)]
               for u, v, h, ok, hd, iw in zip(t['u'], t['v'], t['hour'], t['ok'], t['hidden'], t['in_window']) if ok]
        out.append({'day': t['label'], 'date': t['date'], 'heading_deg': _r(t['heading'], 1),
                    'points': pts})
    return out


def summary(res: ExposureResult) -> dict:
    """Compact figures for the assistant and the batch log."""
    st = res.settings
    return {'location': {'lat': st.lat, 'lon': st.lon}, 'dates': [st.start, st.end],
            'driving_window_local_h': [st.hour_from, st.hour_to],
            'headings': 'uniform' if isinstance(st.headings, str) else f'{len(st.headings)} weighted'
                                                                       + (f' ({st.route.get("name")})'
                                                                          if st.route else ''),
            'driving_hours': _r(res.metrics['driving_hours'], 1),
            'daylight_pct': _r(100 * res.metrics['daylight_share'], 1),
            'items': [{'tag': i.tag, 'kind': i.kind,
                       'sun_in_image_pct' if i.kind == 'camera' else 'sun_in_field_pct':
                           _r(100 * i.sun['share'], 2) if i.sun else None,
                       **({'near_image_pct': _r(100 * i.sun['near'], 2), 'low_sun_ahead_pct': _r(100 * i.sun['low'], 2),
                           'worst_months': months_text(i.sun['worst_months']),
                           'worst_hours': hours_text(i.sun['worst_hours']),
                           'worst_headings': heading_span_text(i.sun['worst_span'])} if i.kind == 'camera' else {}),
                       'dirt_score': _r(i.dirt['score'], 2), 'dirt_class': i.dirt['class'],
                       'dirt_mostly': DIRT_LABEL.get(i.dirt['dominant'], ''), 'advice': i.dirt['advice']}
                      for i in res.items],
            'findings': [{'severity': f['severity'], 'message': f['message']} for f in res.findings
                         if f['severity'] != 'good'],
            'assumptions': ASSUMPTIONS}


def to_doc(res: ExposureResult, project=None) -> dict:
    """The JSON export."""
    st, m = res.settings, res.metrics

    def pose(i):
        return {'position': [_r(c, 4) for c in i.pos], 'yaw': _r(i.ypr[0], 3), 'pitch': _r(i.ypr[1], 3),
                'roll': _r(i.ypr[2], 3)}
    entries = {'cameras': [], 'lidars': [], 'radars': []}
    for i in res.items:
        e = {'tag': i.tag, 'model': i.model, **pose(i), 'sun': _sun_doc(i.sun), 'dirt': _dirt_doc(i.dirt)}
        if i.kind == 'camera':
            e['sun_tracks'] = {'heading_deg': _r(i.worst_heading, 1),
                               'image_px': [i.image['w'], i.image['h']] if i.image else None,
                               'days': _track_doc(i.tracks.get('worst') or i.tracks.get('0') or [])}
        elif i.kind == 'radar':
            e['note'] = 'Radars do not see sunlight; dirt and water on the fascia matter less than on a lens.'
        entries[i.kind + 's'].append(e)
    sc = res.dirt_faces['score']
    ok = np.isfinite(sc)
    area = None
    if len(sc):
        tri = res.dirt_faces['tri'].astype(float)
        a = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
        k = np.digitize(np.nan_to_num(sc), DIRT_LIMITS)
        area = {c: _r(a[ok & (k == j)].sum(), 3) for j, c in enumerate(DIRT_CLASSES)}
    return {
        'schema': SCHEMA, 'project': getattr(project, 'name', ''), 'variant': res.variant,
        'created': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'frame': FRAME,
        'units': {'length': 'm', 'angles': 'degrees', 'azimuth': 'sun: clockwise from north; vehicle frame: '
                                                                 'counter-clockwise from +x',
                  'heading': 'degrees clockwise from north', 'shares': 'percent of all driving time in the window',
                  'time': 'local hours (UTC + offset, or the time zone)', 'pixels': 'sensor pixels, origin top left'},
        'settings': st.to_dict(),
        'assumptions': ASSUMPTIONS,
        'campaign': {'driving_hours': _r(m['driving_hours'], 1), 'daylight_pct': _r(100 * m['daylight_share'], 2),
                     'campaign_days': m['campaign_days'], 'days_sampled': m['days_sampled'],
                     'day_step': m['day_step'], 'time_samples': m['time_samples'], 'headings': m['headings'],
                     'note': m['note'] or None,
                     'heading_table': [[_r(h, 3), _r(w, 6)] for h, w in zip(*st.heading_table())]},
        'sun_model': 'NOAA solar position (Meeus) with NOAA refraction; the sun as a point',
        'cameras': entries['cameras'], 'lidars': entries['lidars'], 'radars': entries['radars'],
        'dirt_model': {'text': DIRT_MODEL, 'weights': DIRT_WEIGHTS, 'classes': {'low': f'< {DIRT_LIMITS[0]}',
                                                                               'moderate': f'< {DIRT_LIMITS[1]}',
                                                                               'high': f'≥ {DIRT_LIMITS[1]}'},
                       'recessed_factor': RECESS_FACTOR, 'wheels': [list(w) for w in res.dirt_info['wheels']],
                       'wheels_from': res.dirt_info['wheels_from'],
                       'body_height': _r(res.dirt_info['height'], 3), 'rear_boxiness': _r(res.dirt_info['boxiness'], 3),
                       'body_area_m2_by_class': area},
        'findings': [{'severity': f['severity'], 'kind': f['kind'], 'message': f['message'], 'sensors': f['tags']}
                     for f in res.findings],
    }


# =========================================================================== the image
def _hex_bgr(h):
    return int(h[5:7], 16), int(h[3:5], 16), int(h[1:3], 16)


def ramp(colors, t) -> np.ndarray:
    """(…, 3) RGB 0–255 of values t in 0..1 along a list of hex colours (linear between stops)."""
    stops = np.array([[int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16)] for c in colors], float)
    t = np.clip(np.nan_to_num(np.asarray(t, float)), 0.0, 1.0) * (len(stops) - 1)
    i = np.minimum(np.floor(t).astype(int), len(stops) - 2)
    f = (t - i)[..., None]
    return stops[i] * (1 - f) + stops[i + 1] * f


# the dirt map's views: (name, title, screen x, screen y (up), depth (larger = nearer the viewer))
DIRT_VIEWS = (('side', 'Left side', lambda P: (-P[..., 0], P[..., 2], P[..., 1])),
              ('top', 'Top', lambda P: (-P[..., 1], P[..., 0], P[..., 2])),
              ('front', 'Front', lambda P: (P[..., 1], P[..., 2], P[..., 0])),
              ('rear', 'Rear', lambda P: (-P[..., 1], P[..., 2], -P[..., 0])))


def view_faces(res: ExposureResult, view: str):
    """(screen polygons (F, 3, 2), scores (F,), glass (F,), shade (F,)) of the dirt map in one view, far to near."""
    fn = dict((n, f) for n, _, f in DIRT_VIEWS)[view]
    tri = res.dirt_faces['tri']
    if not len(tri):
        return np.zeros((0, 3, 2)), np.zeros(0), np.zeros(0, bool), np.zeros(0)
    sx, sy, dep = fn(tri.astype(float))
    order = np.argsort(dep.mean(axis=1))
    poly = np.stack([sx, sy], -1)[order]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]).astype(float)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    vd = {'side': [0, 1, 0], 'top': [0, 0, 1], 'front': [1, 0, 0], 'rear': [-1, 0, 0]}[view]
    shade = 0.78 + 0.22 * np.abs(n @ np.array(vd, float))
    return poly, res.dirt_faces['score'][order], res.dirt_faces['glass'][order], shade[order]


def marker_shown(res: ExposureResult, view: str, pos) -> bool:
    """Whether a sensor at pos is on the half of the vehicle a view looks at (the far half's markers are hidden)."""
    tri = res.dirt_faces['tri']
    if view == 'top' or not len(tri):
        return True
    if view == 'side':
        return pos[1] >= -0.05
    mid = 0.5 * (float(tri[..., 0].min()) + float(tri[..., 0].max()))
    return pos[0] >= mid if view == 'front' else pos[0] <= mid


_ASCII = str.maketrans({'·': '-', '–': '-', '—': '-', '°': ' deg', '×': 'x', '≥': '>=', '≤': '<=', '²': '2'})


def _ascii(text) -> str:
    """OpenCV's fonts draw ASCII only."""
    return str(text).translate(_ASCII).encode('ascii', 'replace').decode('ascii')


def png(res: ExposureResult, width=1400) -> bytes:
    """Per camera and lidar the month × hour share of driving time with the sun in its image (field), and the dirt
    map of the body in four views with the cameras and sensors marked."""
    import cv2
    font, aa = cv2.FONT_HERSHEY_SIMPLEX, cv2.LINE_AA
    bg, ink, ink2, muted = (30, 29, 28), (236, 235, 230), (200, 199, 192), (140, 140, 135)
    st = res.settings
    cols = res.hours_cols
    with_sun = [i for i in res.items if i.sun is not None]
    cw, chh, lab_w = 16, 12, 30
    pw = lab_w + cw * len(cols) + 24
    per_row = max(1, (width - 40) // pw)
    rows = math.ceil(len(with_sun) / per_row) if with_sun else 0
    ph = 34 + 12 * chh + 22
    top = 70
    heat_h = rows * ph + (40 if rows else 0)
    view_h = 300
    H = top + heat_h + 40 + view_h + 60
    img = np.full((H, width, 3), bg, np.uint8)

    def text(t, x, y, col=ink, s=0.45, th=1):
        cv2.putText(img, _ascii(t), (int(x), int(y)), font, s, col, th, aa)
    text(f'Sun glare and dirt exposure  {res.variant}', 20, 30, ink, 0.65, 1)
    hd = 'uniform headings' if isinstance(st.headings, str) else f'{len(st.headings)} weighted headings'
    text(f'{st.lat:.3f}, {st.lon:.3f}  {st.start} to {st.end}  {st.hour_from:g}-{st.hour_to:g} h local  '
         f'every {st.minutes:g} min, {res.metrics["day_step"]} d  {hd}  {res.metrics["driving_hours"]:,.0f} h driving',
         20, 54, ink2, 0.45)
    def top_share(kind):
        return max([float(np.nanmax(i.sun['month_hour'])) for i in with_sun
                    if i.kind == kind and np.isfinite(i.sun['month_hour']).any()] + [0.05])
    vmaxes = {'camera': top_share('camera'), 'lidar': top_share('lidar')}
    for k, it in enumerate(with_sun):
        vmax = vmaxes[it.kind]
        x0 = 20 + (k % per_row) * pw
        y0 = top + (k // per_row) * ph
        word = 'image' if it.kind == 'camera' else 'field'
        text(f'{it.tag}  {100 * it.sun["share"]:.1f} % sun in {word}', x0, y0 + 16, ink, 0.45)
        for r in range(12):
            text(MONTHS[r], x0, y0 + 30 + r * chh + 9, muted, 0.32)
            for c, h in enumerate(cols):
                v = it.sun['month_hour'][r, h]
                x, y = x0 + lab_w + c * cw, y0 + 26 + r * chh
                if not math.isfinite(v):
                    continue
                col = ramp(SUN_RAMP, v / vmax)[::-1] if v > 0 else (52, 51, 49)
                cv2.rectangle(img, (x, y), (x + cw - 2, y + chh - 2), tuple(int(q) for q in col), -1)
        for c, h in enumerate(cols):
            if c % 2 == 0:
                text(f'{h:02d}', x0 + lab_w + c * cw, y0 + 26 + 12 * chh + 10, muted, 0.3)
    if rows:
        y = top + rows * ph + 10
        for j in range(100):
            col = ramp(SUN_RAMP, j / 99)[::-1]
            cv2.rectangle(img, (20 + 2 * j, y), (22 + 2 * j, y + 10), tuple(int(q) for q in col), -1)
        scales = f'cameras 0 to {100 * vmaxes["camera"]:.0f} %' + (
            f', lidars 0 to {100 * vmaxes["lidar"]:.0f} %' if any(i.kind == 'lidar' for i in with_sun) else '')
        text(f'Share of the driving time in that month and hour with the sun in the image (field): {scales}; '
             'dark grey: never', 230, y + 10, muted, 0.4)
    # the dirt map
    y0 = top + heat_h + 30
    text(f'Dirt and spray exposure (heuristic): sand low to brown high; classes at {DIRT_LIMITS[0]:g} and '
         f'{DIRT_LIMITS[1]:g}', 20, y0, ink, 0.5)
    y0 += 14
    boxes = (('side', 20, 420), ('top', 460, 300), ('front', 780, 280), ('rear', 1080, 280))
    sc_all = []
    for name, _, w_ in boxes:
        poly, _, _, _ = view_faces(res, name)
        if len(poly):
            sc_all.append(max(np.ptp(poly[..., 0]) / (w_ - 20), np.ptp(poly[..., 1]) / (view_h - 50)))
    s = 1.0 / max(sc_all) if sc_all else 100.0
    for name, x0, w_ in boxes:
        title = dict((n, t) for n, t, _ in DIRT_VIEWS)[name]
        text(title, x0, y0 + 14, ink2, 0.42)
        poly, sc, glass, shade = view_faces(res, name)
        cx = x0 + w_ / 2
        if len(poly):
            mx = 0.5 * (poly[..., 0].min() + poly[..., 0].max())
            my = 0.5 * (poly[..., 1].min() + poly[..., 1].max())
        else:
            mx = my = 0.0
        cy = y0 + 24 + (view_h - 30) / 2

        def scr(sx, sy):
            return cx + (sx - mx) * s, cy - (sy - my) * s
        if len(poly):
            col = ramp(DIRT_RAMP, np.where(np.isfinite(sc), sc, 0.0))
            col = np.where(np.isfinite(sc)[:, None], col, 70.0) * shade[:, None]
            pts = np.stack(scr(poly[..., 0], poly[..., 1]), -1)
            ip = np.round(pts * 16).astype(np.int32)
            for j in range(len(ip)):
                c = col[j]
                cv2.fillPoly(img, [ip[j]], (int(c[2]), int(c[1]), int(c[0])), aa, 4)
        fn = dict((n, f) for n, _, f in DIRT_VIEWS)[name]
        placed = []                                     # label boxes (x0, y0, x1, y1) in this view
        for it in res.items:
            if not marker_shown(res, name, it.pos):
                continue
            sx, sy, _ = fn(np.array(it.pos, float))
            x, y = scr(float(sx), float(sy))
            tone = _hex_bgr(STATUS_HEX[CLASS_TONE[it.dirt['class']]])
            cv2.circle(img, (int(x), int(y)), 5, (20, 20, 20), -1, aa)
            cv2.circle(img, (int(x), int(y)), 4, tone, -1, aa)
            tw = cv2.getTextSize(_ascii(it.tag), font, 0.38, 1)[0][0]
            for dy in (-4, 12, -17, 25, -30):          # the first free place above, below, further out
                box = (x + 6, y + dy - 10, x + 6 + tw, y + dy + 1)
                if not any(box[0] < b[2] and b[0] < box[2] and box[1] < b[3] and b[1] < box[3] for b in placed):
                    break
            placed.append(box)
            text(it.tag, box[0], box[3] - 1, ink, 0.38)
    y = y0 + view_h + 6
    for j in range(100):
        col = ramp(DIRT_RAMP, j / 99)[::-1]
        cv2.rectangle(img, (20 + 2 * j, y), (22 + 2 * j, y + 10), tuple(int(q) for q in col), -1)
    text('score 0 to 1;  markers: sensor class (green low, amber moderate, red high)', 230, y + 10, muted, 0.4)
    ok, buf = cv2.imencode('.png', img)
    return buf.tobytes()


def files(res: ExposureResult, project=None) -> list:
    """[(name, bytes)]: exposure.json and exposure.png."""
    return [('exposure.json', (json.dumps(to_doc(res, project), indent=1, ensure_ascii=False) + '\n').encode('utf-8')),
            ('exposure.png', png(res))]


def export(res: ExposureResult, project, out_dir) -> list[str]:
    """Write exposure.json and exposure.png into out_dir; returns their paths."""
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for name, data in files(res, project):
        path = os.path.join(out_dir, name)
        with open(path, 'wb') as fh:
            fh.write(data)
        paths.append(path)
    return paths
