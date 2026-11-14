"""The edge watchdog's configuration file (schema vantage.edge_config/1) and the lens model it carries.

The file is written by Vantage (core.edge_daemon.export_config) from a project and read here with the standard library
and NumPy only, so the watchdog runs on a computer that has no Vantage project, Qt or OpenGL. Every value has a default:
a hand-written file needs only its cameras (tag and source).

The lens is carried as a radial table: the angle from the optical axis theta (radians) against the distance from the
principal point r (pixels at the sensor's full resolution), sampled over the image circle, with the principal point
(cx, cy in the sensor's own coordinates, pixel centre at i + 0.5) and the pixel aspect fy / fx. Every radially
symmetric projection Vantage models (rectilinear, equidistant, equisolid, stereographic, orthographic, Kannala-Brandt)
is such a table, so one form serves them all; a stream at another resolution is scaled to it. Linear interpolation
between 256 samples of a smooth r(theta) is exact to well under 0.01 px.
"""
from __future__ import annotations

import json
import math

import numpy as np

SCHEMA = 'vantage.edge_config/1'

# what each section holds when the file leaves it out (the same defaults as core.edge_daemon.EdgeSettings)
KNOCK_DEFAULTS = {
    'enabled': True,
    'analysis_hz': 5.0,         # analysed frames per camera per second, at most
    'analysis_width': 640,      # px: frames are analysed at this width at most
    'threshold_deg': 1.0,       # a step in the direction of travel at least this big is a knock
    'body_threshold_deg': 0.5,  # a step of the body's edge at least this big is a knock (it scatters far less)
    'z_min': 4.0,               # ... and this many standard errors of the window median away from the baseline
    'baseline_s': 3.0,          # the reference is learned over this long (seconds of analysed frames)
    'window_s': 3.0,            # the current pointing is the median over this long
    'hold_s': 1.0,              # a step must stay this long to be a knock
    'pair_s': 0.1,              # the travel direction is measured between a frame and the one about this long before
    'features_per_cell': 5,     # tracked corners per cell of a 10 x 8 grid over the image, at most
    'min_parallax_px': 1.0,     # median feature motion (analysis px) below this: the vehicle stands, no estimate
    'min_inlier_share': 0.7,    # fewer pairs fitting one straight motion than this: turning or traffic, no estimate
    'max_theta_deg': 70.0,      # rays further than this from the optical axis are not used for the travel direction
    'body_coherence': 0.85,     # a pixel whose gradient keeps its direction this well over the learning time is static
    'body_min_gradient': 20.0,  # ... and whose mean gradient (Sobel, grey levels) is at least this: an edge
    'body_flow_cos': 0.35,      # static edges whose gradient is this close to across the flow are the world's (lanes)
    'body_plan_margin_deg': 8.0,  # static edges further than this from where the plan puts the body are left out
    'body_min_share': 0.003,    # the static body edges must cover at least this share of the image
    'body_search_deg': 4.0,     # the body edge is searched this far
    'body_min_score': 0.5,      # normalised correlation below this: no body estimate for that frame
}
ALERT_DEFAULTS = {
    'raise_s': 1.0, 'clear_s': 3.0,
    'grace_s': 3.0,             # a stream's frame-rate, drop, stall and sync checks wait this long after it (re)starts
    'http': {'bind': '127.0.0.1', 'port': 8088},
    'mqtt': {'host': '', 'port': 1883, 'topic': 'vantage/edge', 'user': '', 'keepalive_s': 60},
    'webhook': {'url': '', 'timeout_s': 3.0},
    'log': {'path': '', 'max_mb': 10.0, 'keep': 5},
    'gpio': {'backend': 'none', 'chip': 'gpiochip0', 'line': -1, 'active_high': True, 'path': '', 'pattern': 'beep'},
}


def _num(v, default, lo=-math.inf, hi=math.inf):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return default
    return min(max(x, lo), hi) if math.isfinite(x) else default


class RadialLens:
    """Pixel <-> optical-frame ray for one camera, at the sensor's full resolution (width x height), from the radial
    table. Optical frame: x right, y down, z along the axis (OpenCV's camera frame)."""

    def __init__(self, d: dict):
        self.width = int(d['width'])
        self.height = int(d['height'])
        self.cx = float(d['cx'])
        self.cy = float(d['cy'])
        self.aspect = float(d.get('aspect', 1.0) or 1.0)
        th = np.asarray(d['theta'], float)
        r = np.asarray(d['r_px'], float)
        if th.ndim != 1 or len(th) < 2 or len(th) != len(r) or np.any(np.diff(r) <= 0) or np.any(np.diff(th) <= 0):
            raise ValueError('the lens table needs increasing theta and r_px of the same length')
        self.theta, self.r = th, r
        self.max_theta = float(d.get('max_theta', th[-1]))
        self.f_px = float(d.get('f_px', r[1] / th[1] if th[1] > 0 else 1.0))

    @staticmethod
    def from_intrinsics(intr, samples: int = 256) -> dict:
        """The table of a vantage.core.lens.Intrinsics (needs Vantage; the watchdog itself only reads tables)."""
        from ..core.lens import r_of_theta
        top = float(intr.max_theta)
        if intr.projection == 'rectilinear':
            top = min(top, math.radians(85.0))
        th = np.linspace(0.0, top, samples)
        r = np.asarray(r_of_theta(th, intr.f_mm, intr.projection, intr.kb), float) / intr.pixel_mm
        keep = np.concatenate([[True], np.diff(r) > 1e-9])             # a lens whose r(theta) folds back is cut there
        bad = np.where(~keep)[0]
        n = int(bad[0]) if len(bad) else samples
        return {'width': int(intr.width), 'height': int(intr.height), 'cx': float(intr.cx), 'cy': float(intr.cy),
                'aspect': float(intr.aspect), 'f_px': float(intr.f_px), 'projection': intr.projection,
                'max_theta': float(min(intr.max_theta, th[n - 1])),
                'theta': [round(float(x), 7) for x in th[:n]], 'r_px': [round(float(x), 5) for x in r[:n]]}

    def to_dict(self) -> dict:
        return {'width': self.width, 'height': self.height, 'cx': self.cx, 'cy': self.cy, 'aspect': self.aspect,
                'f_px': self.f_px, 'max_theta': self.max_theta, 'theta': self.theta.tolist(), 'r_px': self.r.tolist()}

    def scale(self, w: int, h: int) -> tuple:
        """(sx, sy): stream pixels per sensor pixel for a w x h stream."""
        return w / self.width, h / self.height

    def unproject(self, u, v, w=None, h=None):
        """OpenCV pixel coordinates (centre at integer i) of a w x h stream (the sensor's size when None) -> unit rays
        (N, 3) and theta (N,)."""
        sx, sy = self.scale(w or self.width, h or self.height)
        x = (np.asarray(u, float) + 0.5) / sx - self.cx
        y = ((np.asarray(v, float) + 0.5) / sy - self.cy) / self.aspect
        rr = np.hypot(x, y)
        th = np.interp(rr, self.r, self.theta, right=np.nan)
        s = np.where(rr > 1e-12, np.sin(np.nan_to_num(th)) / np.where(rr > 1e-12, rr, 1.0), 0.0)
        d = np.stack([x * s, y * s, np.cos(np.nan_to_num(th))], -1)
        return d, th

    def project(self, d, w=None, h=None):
        """Optical-frame directions (N, 3) -> OpenCV pixel coordinates (u, v) of a w x h stream and whether each lands
        inside the image circle and the frame."""
        d = np.asarray(d, float)
        sx, sy = self.scale(w or self.width, h or self.height)
        rho = np.hypot(d[..., 0], d[..., 1])
        th = np.arctan2(rho, d[..., 2])
        rr = np.interp(th, self.theta, self.r, right=np.nan)
        s = np.where(rho > 1e-12, rr / np.where(rho > 1e-12, rho, 1.0), 0.0)
        u = (self.cx + d[..., 0] * s) * sx - 0.5
        v = (self.cy + d[..., 1] * s * self.aspect) * sy - 0.5
        W, H = (w or self.width), (h or self.height)
        ok = np.isfinite(rr) & (th <= self.max_theta) & (u >= -0.5) & (u <= W - 0.5) & (v >= -0.5) & (v <= H - 0.5)
        return u, v, ok

    def deg_per_px(self, theta: float, w=None) -> float:
        """Degrees per stream pixel along the radius at angle theta from the axis (the local image scale)."""
        sx = (w or self.width) / self.width
        dr = np.gradient(self.r, self.theta)
        return float(math.degrees(1.0 / (np.interp(theta, self.theta, dr) * sx)))

    def inside(self, w, h, max_theta=None):
        """Mask (h, w) of the pixels whose ray is within max_theta (the image circle when None)."""
        uu, vv = np.meshgrid(np.arange(w, dtype=float), np.arange(h, dtype=float))
        _, th = self.unproject(uu.ravel(), vv.ravel(), w, h)
        lim = self.max_theta if max_theta is None else min(self.max_theta, max_theta)
        return (np.nan_to_num(th, nan=99.0) <= lim).reshape(h, w)


def load(path_or_doc) -> dict:
    """A configuration with every default filled in, from a path or a dict. Raises ValueError with the reason when
    the file cannot be used (no cameras, a camera without a tag)."""
    if isinstance(path_or_doc, dict):
        doc = path_or_doc
    else:
        with open(path_or_doc, encoding='utf-8') as fh:
            doc = json.load(fh)
    if not isinstance(doc, dict):
        raise ValueError('the configuration is not a JSON object')
    schema = str(doc.get('schema', SCHEMA))
    if not schema.startswith('vantage.edge_config/'):
        raise ValueError(f'not an edge watchdog configuration (schema {schema!r})')
    out = {k: v for k, v in doc.items()}
    out['schema'] = SCHEMA
    out['rig'] = str(doc.get('rig') or doc.get('project') or 'rig')
    knock = dict(KNOCK_DEFAULTS)
    for k, v in (doc.get('knock') or {}).items():
        if k in knock:
            knock[k] = bool(v) if isinstance(knock[k], bool) else type(knock[k])(_num(v, knock[k], 0.0))
    out['knock'] = knock
    al = json.loads(json.dumps(ALERT_DEFAULTS))
    for k, v in (doc.get('alerts') or {}).items():
        if isinstance(al.get(k), dict) and isinstance(v, dict):
            for kk, vv in v.items():
                if kk in al[k]:
                    d0 = al[k][kk]
                    al[k][kk] = (bool(vv) if isinstance(d0, bool) else int(_num(vv, d0)) if isinstance(d0, int)
                                 else _num(vv, d0) if isinstance(d0, float) else str(vv))
        elif k in ('raise_s', 'clear_s', 'grace_s'):
            al[k] = _num(v, al[k], 0.0, 3600.0)
    out['alerts'] = al
    out['health'] = dict(doc.get('health') or {})
    cams = []
    for c in doc.get('cameras') or []:
        if not isinstance(c, dict) or not str(c.get('tag', '')).strip():
            raise ValueError('every camera needs a tag')
        cc = dict(c)
        cc['tag'] = str(c['tag']).strip()
        cc['source'] = str(c.get('source') or '').strip()
        cc['fps'] = _num(c.get('fps'), 30.0, 0.1, 1000.0)
        if c.get('lens'):
            RadialLens(c['lens'])                       # raises on a broken table
        cams.append(cc)
    if not cams:
        raise ValueError('the configuration lists no cameras')
    if len({c['tag'] for c in cams}) != len(cams):
        raise ValueError('camera tags must be unique')
    out['cameras'] = cams
    po = {}
    for k, v in (doc.get('planned_offsets') or {}).items():
        try:
            po[str(k)] = (float(v[0]), float(v[1]))
        except (TypeError, ValueError, IndexError):
            continue
    out['planned_offsets'] = po
    return out
