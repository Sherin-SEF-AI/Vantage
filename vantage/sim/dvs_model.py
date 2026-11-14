"""The event model and CARLA's DVS camera, the core core.event_camera builds on: an event stream (Events), the
per-frame simulator (EventSim, SimConfig: carla=True is CARLA's DVSCamera.cpp exactly), CARLA's and linear log
luminance, and the comparison of CARLA's DVS stream with the model's.

It lives in vantage/sim because the CARLA bridge runs it in the loop inside CARLA's own Python, where only
vantage/sim and a few plain files are shipped (packaging/release.BRIDGE_SOURCES): NumPy and the standard library
only."""
from __future__ import annotations

import io
import math
from dataclasses import dataclass

import numpy as np

EVENTS_SCHEMA = 'vantage.events/1'            # a recording's events/index.json
COMPARE_SCHEMA = 'vantage.dvs_compare/1'


# CARLA's carla.DVSEventArray raw_data (packed: x, y uint16, t int64 ns, pol bool), as its PythonAPI examples read it
CARLA_DVS_DTYPE = np.dtype([('x', np.uint16), ('y', np.uint16), ('t', np.int64), ('pol', bool)])


# CARLA DVSCamera.cpp: grey = 0.2989 R + 0.5870 G + 0.1140 B of the 8-bit image; L = ln(log_eps + grey)
CARLA_GREY = (0.2989, 0.5870, 0.1140)          # R, G, B


CARLA_DVS_DEFAULTS = {'positive_threshold': 0.3, 'negative_threshold': 0.3, 'sigma_positive_threshold': 0.0,
                      'sigma_negative_threshold': 0.0, 'refractory_period_ns': 0, 'use_log': True, 'log_eps': 0.001}


SHOT_INTEN_FACTOR = 0.25                      # v2e: shot noise at full intensity is this share of the dark rate


MIN_THRESHOLD = 0.01                          # ln units: a mismatched threshold never falls below (v2e, CARLA)


NEVER = -1e30


REC709 = (0.2126, 0.7152, 0.0722)             # linear R, G, B -> luminance


# =========================================================================== events
@dataclass
class Events:
    """An event stream: x, y (pixels), t (µs from the stream's origin, int64), p (+1 ON, −1 OFF)."""
    x: np.ndarray
    y: np.ndarray
    t: np.ndarray
    p: np.ndarray
    width: int = 0
    height: int = 0

    @staticmethod
    def empty(width=0, height=0) -> 'Events':
        return Events(np.zeros(0, np.uint16), np.zeros(0, np.uint16), np.zeros(0, np.int64), np.zeros(0, np.int8),
                      width, height)

    def __len__(self):
        return int(len(self.t))

    @staticmethod
    def concat(items, width=0, height=0) -> 'Events':
        items = [e for e in items if e is not None and len(e)]
        if not items:
            return Events.empty(width, height)
        return Events(np.concatenate([e.x for e in items]).astype(np.uint16),
                      np.concatenate([e.y for e in items]).astype(np.uint16),
                      np.concatenate([e.t for e in items]).astype(np.int64),
                      np.concatenate([e.p for e in items]).astype(np.int8),
                      width or items[0].width, height or items[0].height)

    def sorted(self) -> 'Events':
        o = np.lexsort((self.x, self.y, self.t))
        return Events(self.x[o], self.y[o], self.t[o], self.p[o], self.width, self.height)

    def take(self, sel) -> 'Events':
        return Events(self.x[sel], self.y[sel], self.t[sel], self.p[sel], self.width, self.height)

    def between(self, t0_us, t1_us) -> 'Events':
        return self.take((self.t >= t0_us) & (self.t < t1_us))

    def to_npz(self) -> bytes:
        buf = io.BytesIO()
        np.savez_compressed(buf, x=self.x.astype(np.uint16), y=self.y.astype(np.uint16), t=self.t.astype(np.int64),
                            p=self.p.astype(np.int8), size=np.array([self.width, self.height], np.int32))
        return buf.getvalue()

    @staticmethod
    def from_npz(src) -> 'Events':
        with np.load(io.BytesIO(src) if isinstance(src, (bytes, bytearray)) else src, allow_pickle=False) as z:
            size = z['size'] if 'size' in z.files else np.array([0, 0])
            return Events(z['x'].astype(np.uint16), z['y'].astype(np.uint16), z['t'].astype(np.int64),
                          z['p'].astype(np.int8), int(size[0]), int(size[1]))

    @staticmethod
    def from_carla(raw, t0_ns=0, width=0, height=0) -> 'Events':
        """A carla.DVSEventArray's raw_data (CARLA_DVS_DTYPE): t in µs from t0_ns (floor)."""
        a = np.frombuffer(bytes(raw), CARLA_DVS_DTYPE)
        return Events(a['x'].astype(np.uint16), a['y'].astype(np.uint16), (a['t'].astype(np.int64) - int(t0_ns)) // 1000,
                      np.where(a['pol'], 1, -1).astype(np.int8), width, height)


# =========================================================================== the event model
@dataclass
class SimConfig:
    """What EventSim does. carla=True reproduces CARLA's DVS camera exactly (float32 crossings, no low-pass, no noise,
    the threshold's spread drawn per pixel and frame)."""
    threshold_on: float = 0.3
    threshold_off: float = 0.3
    sigma: float = 0.0
    refractory_s: float = 0.0
    lowpass: bool = False
    latency: object = None             # EventParams (its latency_at) or None
    lux_range: tuple | None = None     # (cut-off, saturation) lux: clamp and dead pixels (needs lux in step)
    leak_hz: float = 0.0
    shot_hz: float = 0.0
    carla: bool = False
    tol: float = 1e-6                  # |ΔL| below this between frames: no events (CARLA's tolerance)

    @staticmethod
    def for_params(ep, st, lowpass=None, noise=True) -> 'SimConfig':
        """From core.event_camera's EventParams (ep) and EventSettings (st)."""
        return SimConfig(ep.threshold, ep.threshold, ep.sigma, ep.refractory_s,
                         st.photoreceptor if lowpass is None else bool(lowpass), ep,
                         (ep.cutoff_lux, ep.dr_lux[1]), st.leak_hz if noise else 0.0,
                         st.shot_noise_hz if noise else 0.0)

    @staticmethod
    def carla_dvs(attrs: dict | None = None) -> 'SimConfig':
        a = dict(CARLA_DVS_DEFAULTS, **(attrs or {}))
        return SimConfig(float(a['positive_threshold']), float(a['negative_threshold']),
                         float(a.get('sigma_positive_threshold', 0.0) or 0.0),
                         float(a.get('refractory_period_ns', 0) or 0) * 1e-9, carla=True)


def carla_log_grey(bgr, log_eps=0.001, use_log=True) -> np.ndarray:
    """CARLA's DVS input from an 8-bit B, G, R(, A) image: ln(log_eps + 0.2989 R + 0.5870 G + 0.1140 B) as float32
    (DVSCamera.cpp: computed in double, kept as float)."""
    a = np.asarray(bgr)
    g = CARLA_GREY[0] * a[..., 2].astype(np.float64) + CARLA_GREY[1] * a[..., 1] + CARLA_GREY[2] * a[..., 0]
    return (np.log(log_eps + g) if use_log else g).astype(np.float32)


def linear_log_luminance(bgr, response_ln=None) -> np.ndarray:
    """ln of linear relative luminance of an 8-bit B, G, R image through a camera response (ln exposure per level,
    per channel B, G, R; default CARLA's measured response, vantage.sim.tonemap.RESPONSE_LN)."""
    if response_ln is None:
        from .tonemap import RESPONSE_LN as response_ln
    r = np.asarray(response_ln, np.float64)
    a = np.asarray(bgr)[..., :3].astype(np.intp)
    lin = REC709[2] * np.exp(r[0][a[..., 0]]) + REC709[1] * np.exp(r[1][a[..., 1]]) + REC709[0] * np.exp(r[2][a[..., 2]])
    return np.log(np.maximum(lin, 1e-6))


class EventSim:
    """The event model, frame by frame: feed log intensity (h, w) at increasing times; each step returns the events
    since the previous frame (t in µs from the first frame's time, floored to the timestamp grid)."""

    def __init__(self, shape, cfg: SimConfig, seed=0):
        self.h, self.w = int(shape[0]), int(shape[1])
        self.cfg = cfg
        self.rng = np.random.default_rng([int(seed), 1729])
        self.n = self.h * self.w
        self.started = False
        self.dt_ = np.float32 if cfg.carla else np.float64
        if cfg.carla or cfg.sigma <= 0:
            self.th_on = np.full(self.n, cfg.threshold_on, self.dt_)
            self.th_off = np.full(self.n, cfg.threshold_off, self.dt_)
        else:                                          # fixed-pattern mismatch, drawn once
            self.th_on = np.maximum(self.rng.normal(cfg.threshold_on, cfg.sigma, self.n), MIN_THRESHOLD)
            self.th_off = np.maximum(self.rng.normal(cfg.threshold_off, cfg.sigma, self.n), MIN_THRESHOLD)
        self.counts = {'signal': 0, 'refractory_dropped': 0, 'leak_steps': 0, 'shot': 0, 'frames': 0}

    def _prep(self, logI, lux):
        Lr = np.asarray(logI, self.dt_).reshape(-1)
        dead = None
        inten = None
        if lux is not None and self.cfg.lux_range is not None:
            lo, hi = self.cfg.lux_range
            E = np.broadcast_to(np.asarray(lux, float), (self.h, self.w)).reshape(-1)
            # saturation: the signal stops rising above the range's top (in the same log units as the input)
            over = E > hi
            if over.any():
                Lr = np.where(over, Lr - np.log(E / hi), Lr)
            dead = E < lo
            inten = np.clip(np.log(np.maximum(E, lo) / lo) / math.log(hi / lo), 0.0, 1.0)
            return Lr, E, dead, inten
        return Lr, None, dead, inten

    def reset(self, logI, t_s, lux=None):
        Lr, E, dead, inten = self._prep(logI, lux)
        self.Lf = Lr.copy()                            # the photoreceptor's (filtered) signal
        self.ref = Lr.copy()
        self.last = np.full(self.n, NEVER)
        self.t0 = float(t_s)
        self.t = float(t_s)
        self.started = True
        self.counts['frames'] = 1

    def step(self, logI, t_s, lux=None) -> Events:
        sig, shot = self.step_split(logI, t_s, lux)
        return sig if not len(shot) else Events.concat([sig, shot], self.w, self.h)

    def step_split(self, logI, t_s, lux=None) -> tuple:
        """(signal events, shot-noise events) since the previous frame."""
        empty = Events.empty(self.w, self.h)
        if not self.started:
            self.reset(logI, t_s, lux)
            return empty, empty
        cfg = self.cfg
        t_s = float(t_s)
        dt = t_s - self.t
        if dt <= 0:
            return empty, empty
        Lr, E, dead, inten = self._prep(logI, lux)
        it = self.Lf
        if cfg.lowpass and cfg.latency is not None and E is not None:
            tau = np.asarray(cfg.latency.latency_at(np.maximum(E, 1e-9)), float)
            eps = np.clip(dt / np.maximum(tau, 1e-9), 0.0, 1.0)
            if dead is not None:
                eps = np.where(dead, 0.0, eps)
            itdt = it + eps * (Lr - it)
        elif dead is not None:
            itdt = np.where(dead, it, Lr)
        else:
            itdt = Lr
        if cfg.leak_hz > 0:                            # v2e: the reference leaks down -> ON events
            self.ref = self.ref - cfg.leak_hz * self.th_on * dt
        ev = self._crossings(it, itdt, dt)
        if cfg.leak_hz > 0:                            # a reference leaked a threshold below the signal: ON at t
            ev = Events.concat([ev, self._leaked(itdt, t_s)], self.w, self.h)
        self.Lf = itdt.astype(self.dt_, copy=False)
        self.t = t_s
        self.counts['frames'] += 1
        return ev, (self._shot(dt, inten) if cfg.shot_hz > 0 else empty)

    def _crossings(self, it, itdt, dt) -> Events:
        """ESIM / CARLA DVSCamera.cpp: from the pixel's reference, step by its threshold towards the new value; each
        level strictly beyond the previous frame's value and within the new one is a crossing, timed by linear
        interpolation; the refractory period drops an event (the reference still moves)."""
        cfg = self.cfg
        f = self.dt_
        d = itdt - it
        idx = np.flatnonzero(np.abs(d) > cfg.tol)
        if not len(idx):
            return Events.empty(self.w, self.h)
        a, b = it[idx].astype(f), itdt[idx].astype(f)
        pos = b >= a
        if cfg.carla:
            C = np.where(pos, self.th_on[idx], self.th_off[idx]).astype(np.float64)
            if cfg.sigma > 0:                          # CARLA: a fresh threshold per pixel and frame
                C = np.maximum(C + self.rng.normal(0.0, cfg.sigma, len(C)), MIN_THRESHOLD)
            C = C.astype(f)
        else:
            C = np.where(pos, self.th_on[idx], self.th_off[idx]).astype(f)
        sgn = np.where(pos, 1, -1).astype(f)
        cur = self.ref[idx].astype(f)
        last = self.last[idx]
        t_prev = self.t
        xs, ys, ts, ps = [], [], [], []
        alive = np.ones(len(idx), bool)
        dropped = 0
        span = (b - a)
        while True:
            nxt = (cur + sgn * C).astype(f)
            ok = alive & np.where(pos, (nxt > a) & (nxt <= b), (nxt < a) & (nxt >= b))
            if not ok.any():
                break
            k = np.flatnonzero(ok)
            frac = ((nxt[k] - a[k]).astype(np.float64) / span[k].astype(np.float64))
            te = t_prev + frac * dt
            if cfg.refractory_s > 0:
                emit = (last[k] <= NEVER / 2) | (te - last[k] >= cfg.refractory_s - 1e-12)
            else:
                emit = np.ones(len(k), bool)
            ke = k[emit]
            last[ke] = te[emit]
            dropped += int((~emit).sum())
            pix = idx[ke]
            xs.append(pix % self.w)
            ys.append(pix // self.w)
            ts.append(te[emit])
            ps.append(np.where(pos[ke], 1, -1))
            cur[k] = nxt[k]
            alive = ok
        self.ref[idx] = cur
        self.last[idx] = last
        self.counts['refractory_dropped'] += dropped
        if not ts:
            return Events.empty(self.w, self.h)
        t = np.concatenate(ts)
        ev = Events(np.concatenate(xs).astype(np.uint16), np.concatenate(ys).astype(np.uint16),
                    np.floor((t - self.t0) * 1e6 + 1e-6).astype(np.int64), np.concatenate(ps).astype(np.int8),
                    self.w, self.h)
        self.counts['signal'] += len(ev)
        return ev

    def _leaked(self, L, t_s) -> Events:
        """Pixels whose reference leaked θ or more below the signal (a still pixel never crosses between frames):
        their ON events at the frame's time, the reference moved up by them (v2e's leak events)."""
        th = self.th_on
        n = np.floor((L - self.ref) / th + 1e-9).astype(np.int64)
        n = np.where(n > 0, n, 0)
        idx = np.flatnonzero(n)
        if not len(idx):
            return Events.empty(self.w, self.h)
        self.ref[idx] = self.ref[idx] + n[idx] * th[idx]
        pix = np.repeat(idx, n[idx])
        self.counts['leak_steps'] += len(pix)
        return Events((pix % self.w).astype(np.uint16), (pix // self.w).astype(np.uint16),
                      np.full(len(pix), int(math.floor((t_s - self.t0) * 1e6 + 1e-6)), np.int64),
                      np.ones(len(pix), np.int8), self.w, self.h)

    def _shot(self, dt, inten) -> Events:
        """v2e's shot noise: ON and OFF at shot_hz / 2 each, scaled by (f − 1)·i + 1 (dark pixels noisiest)."""
        cfg = self.cfg
        i01 = np.zeros(self.n) if inten is None else inten
        lam = cfg.shot_hz / 2.0 * dt * ((SHOT_INTEN_FACTOR - 1.0) * i01 + 1.0)
        out = []
        for pol in (1, -1):
            k = self.rng.poisson(lam)
            if not k.any():
                continue
            pix = np.repeat(np.arange(self.n), k)
            te = self.t - dt + self.rng.random(len(pix)) * dt
            out.append(Events((pix % self.w).astype(np.uint16), (pix // self.w).astype(np.uint16),
                              np.floor((te - self.t0) * 1e6).astype(np.int64), np.full(len(pix), pol, np.int8),
                              self.w, self.h))
        ev = Events.concat(out, self.w, self.h)
        self.counts['shot'] += len(ev)
        return ev


def rate_series(ev: Events, bin_us=1000, t0_us=None, t1_us=None) -> tuple[np.ndarray, np.ndarray]:
    """(bin starts µs, events per second in each bin)."""
    if not len(ev) and (t0_us is None or t1_us is None):
        return np.zeros(0), np.zeros(0)
    a = int(ev.t.min()) if t0_us is None else int(t0_us)
    b = int(ev.t.max()) + 1 if t1_us is None else int(t1_us)
    edges = np.arange(a, b + bin_us, bin_us)
    h, _ = np.histogram(ev.t, edges)
    return edges[:-1], h / (bin_us * 1e-6)


def compare_dvs(carla_ev: Events, model_ev: Events, width, height, t0_us=None, t1_us=None, bin_us=1000) -> dict:
    """CARLA's DVS events against the model's on the same frames: totals, the per-pixel counts' correlation, the
    events matched exactly (same pixel, polarity and µs, ±1 µs), the rate series' correlation."""
    a, b = carla_ev.sorted(), model_ev.sorted()
    if t0_us is not None:
        a, b = a.between(t0_us, t1_us), b.between(t0_us, t1_us)
    ca = np.zeros(width * height)
    cb = np.zeros(width * height)
    np.add.at(ca, a.y.astype(np.intp) * width + a.x.astype(np.intp), 1)
    np.add.at(cb, b.y.astype(np.intp) * width + b.x.astype(np.intp), 1)
    r = float(np.corrcoef(ca, cb)[0, 1]) if ca.std() > 0 and cb.std() > 0 else None

    def keys(e, dt=0):
        return (e.t.astype(np.int64) + dt) * (2 * width * height) + (e.y.astype(np.int64) * width + e.x) * 2 + \
            (e.p > 0)
    ka = keys(a)
    kb = np.unique(np.concatenate([keys(b, d) for d in (-1, 0, 1)]))
    exact = float(np.isin(ka, kb).mean()) if len(ka) else None
    lo = min([int(e.t.min()) for e in (a, b) if len(e)] or [0]) if t0_us is None else int(t0_us)
    hi = max([int(e.t.max()) + 1 for e in (a, b) if len(e)] or [1]) if t1_us is None else int(t1_us)
    _, ra = rate_series(a, bin_us, lo, hi)
    _, rb = rate_series(b, bin_us, lo, hi)
    rr = float(np.corrcoef(ra, rb)[0, 1]) if len(ra) > 2 and ra.std() > 0 and rb.std() > 0 else None
    na, nb = len(a), len(b)
    return {'schema': COMPARE_SCHEMA, 'carla_events': na, 'model_events': nb,
            'count_ratio': nb / na if na else None, 'count_error': abs(nb - na) / na if na else None,
            'on_share_carla': float((a.p > 0).mean()) if na else None,
            'on_share_model': float((b.p > 0).mean()) if nb else None,
            'pixel_count_r': r, 'pixel_count_mae': float(np.abs(ca - cb).mean()),
            'exact_match_share': exact, 'rate_series_r': rr, 'bin_us': bin_us,
            'pixels_with_events_carla': int((ca > 0).sum()), 'pixels_with_events_model': int((cb > 0).sum())}


EVT2_CD_OFF, EVT2_CD_ON, EVT2_TIME_HIGH = 0x0, 0x1, 0x8     # Prophesee EVT 2.0 event types


def raw_header(fmt, width, height, extra=None) -> bytes:
    """A Metavision-style RAW header ('% key value' lines, '% end')."""
    name = {'EVT2.0': ('2.0', 'EVT2'), 'EVT3.0': ('3.0', 'EVT3'), 'EVT2.1': ('2.1', 'EVT21')}[fmt]
    lines = [f'% evt {name[0]}', f'% format {name[1]};height={int(height)};width={int(width)}',
             f'% geometry {int(width)}x{int(height)}']
    for k, v in (extra or {}).items():
        lines.append(f'% {k} {v}')
    lines.append('% end')
    return ('\n'.join(lines) + '\n').encode('ascii', 'replace')


def encode_evt2(ev: Events) -> bytes:
    """EVT 2.0: per event a 32-bit word (type CD_ON/CD_OFF << 28 | t & 63 << 22 | x << 11 | y), a TIME_HIGH word
    (0x8 << 28 | t >> 6) whenever t >> 6 changes; little-endian."""
    ev = ev.sorted()
    if not len(ev):
        return b''
    t = ev.t.astype(np.int64)
    if (t < 0).any():
        raise ValueError('EVT 2.0 needs times from 0.')
    hi = t >> 6
    new = np.r_[True, hi[1:] != hi[:-1]]
    cd = ((np.where(ev.p > 0, EVT2_CD_ON, EVT2_CD_OFF).astype(np.uint64) << 28) | ((t & 63).astype(np.uint64) << 22)
          | (ev.x.astype(np.uint64) << 11) | ev.y.astype(np.uint64))
    th = (np.uint64(EVT2_TIME_HIGH) << np.uint64(28)) | (hi.astype(np.uint64) & np.uint64(0x0FFFFFFF))
    n_th = int(new.sum())
    words = np.zeros(len(t) + n_th, np.uint64)
    pos = np.arange(len(t)) + np.cumsum(new)          # each event after its TIME_HIGH (if any)
    words[pos] = cd
    words[pos[new] - 1] = th[new]
    return words.astype('<u4').tobytes()
