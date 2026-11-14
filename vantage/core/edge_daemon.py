"""The edge watchdog, from the project's side: its configuration exported from a design, its systemd unit, and dry runs
of the watchdog (vantage.edge) on a recording with faults written into it.

The watchdog itself (vantage.edge) is a headless service for the car's computer: it reads the rig's camera streams
(V4L2, GStreamer, RTSP, files, ROS 2 topics), runs rig health's checks on every frame (frame rate, drops, stalls,
frozen images, exposure, sharpness, resolution, the capture skew against the sync plan) and knock detection on a few
frames a second (the direction of travel in each camera's frame and, where the camera sees the own body, the body's
edge), and alerts locally: a status page and JSON on the car's network, MQTT, a webhook, a log file, a buzzer line. It
has no Qt and no OpenGL and only reads and alerts. Its configuration is one JSON file (schema vantage.edge_config/1)
written here: the cameras (tag, stream source, nominal frame rate, resolution, the lens as a radial table, the planned
pose and the direction of travel it implies in the camera's optical frame, where the plan puts the own body in its
image), the sync plan's capture offsets, rig health's thresholds, the knock settings and the alert channels.

Dry runs (dry_run) replay a recording (a nuScenes folder, image folders, video files or a ROS 2 bag) through the
watchdog's own code, deterministically on the recording's stamps, with faults written in (vantage.edge.inject: a knock
re-rendered through the lens model, a frozen stream, dropped frames, a stall, a dark image), then score every fault:
caught or not, by which check and how long after it began, and every alert no fault explains (a false alarm). Without a
recording, synthetic_drive renders a straight drive past textured ground and walls through each camera's lens, with the
own body as a box, so the checks can be tried on any project.

Settings (project.analysis.edge_daemon, EdgeSettings): the stream sources per camera (else project.streams, else the
V4L2 device of its GMSL port), the knock thresholds, the alert channels, the status page's address and the service's
limits. Passwords are never stored: the MQTT password comes from the environment of the service.

Limits: see vantage.edge.knock (travel direction: straight driving, a mostly static world; body edge: the body in
view). A dry run measures the watchdog's logic and its cost on this computer, not a Jetson's: the cost on the car
scales with its CPU (the status reports it live). The synthetic drive is a test scene, not a rendering of the world.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, field

import numpy as np

SCHEMA = 'vantage.edge_daemon/1'
CONFIG_SCHEMA = 'vantage.edge_config/1'
FRAME = 'vehicle (ISO 8855, rear axle)'
LIMITS = {
    'analysis_hz': (0.5, 30.0), 'analysis_width': (160, 1920), 'knock_deg': (0.1, 20.0),
    'body_knock_deg': (0.1, 20.0), 'knock_z': (1.0, 50.0),
    'baseline_s': (1.0, 600.0), 'window_s': (0.3, 60.0), 'hold_s': (0.0, 60.0), 'min_inlier_share': (0.3, 0.99),
    'raise_s': (0.0, 600.0), 'clear_s': (0.0, 3600.0), 'grace_s': (0.0, 600.0), 'http_port': (0, 65535),
    'mqtt_port': (1, 65535),
    'gpio_line': (-1, 1023), 'memory_mb': (64, 16384), 'cpu_percent': (5, 800), 'log_mb': (0.1, 1000.0),
}
INTS = ('analysis_width', 'http_port', 'mqtt_port', 'gpio_line', 'memory_mb', 'cpu_percent')
GPIO_BACKENDS = ('none', 'auto', 'gpiod', 'gpioset', 'file')
FAULT_CHECKS = {'knock': ('knock',), 'freeze': ('frozen',), 'drop': ('drops', 'frame_rate', 'host'),
                'stall': ('stall', 'stream', 'frame_rate', 'host'), 'dark': ('dark',)}   # host: every camera at once
ASSUMPTIONS = (
    'Rig health is vantage.core.rig_health\'s checks with its thresholds; capture times are the car computer\'s clock '
    'when a frame arrived (live) or the recording\'s stamps (dry run).',
    'Knock detection compares each camera\'s pointing with its own baseline learned over the first seconds: the '
    'direction of travel in the camera\'s frame (straight driving, a mostly static world) and, where the body is in '
    'view, the body\'s edge. It is an alarm, not a calibration.',
    'A knock in a dry run is the camera rotated about its optical centre, re-rendered through the lens model (no '
    'parallax from the millimetres a real knock also shifts it).',
    'Cost figures are this computer\'s; on the car they scale with its CPU.',
)


@dataclass
class EdgeSettings:
    sources: dict = field(default_factory=dict)      # tag -> stream source; empty = project.streams, else /dev/videoN
    analysis_hz: float = 5.0
    analysis_width: int = 640
    knock_deg: float = 1.0
    body_knock_deg: float = 0.5
    knock_z: float = 4.0
    baseline_s: float = 3.0
    window_s: float = 3.0
    hold_s: float = 1.0
    min_inlier_share: float = 0.7
    knock_enabled: bool = True
    raise_s: float = 1.0
    clear_s: float = 3.0
    grace_s: float = 3.0
    http_bind: str = '127.0.0.1'
    http_port: int = 8088
    mqtt_host: str = ''
    mqtt_port: int = 1883
    mqtt_topic: str = ''                               # '' = vantage/<rig>
    mqtt_user: str = ''
    webhook_url: str = ''
    log_path: str = '/var/log/vantage/edge.log'
    log_mb: float = 10.0
    gpio_backend: str = 'none'
    gpio_chip: str = 'gpiochip0'
    gpio_line: int = -1
    gpio_path: str = ''
    service_user: str = 'vantage'
    python: str = '/usr/bin/python3'
    config_path: str = '/etc/vantage/edge.json'
    memory_mb: int = 512
    cpu_percent: int = 100
    health: dict = field(default_factory=dict)        # rig_health.HealthSettings overrides

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d) -> 'EdgeSettings':
        s = EdgeSettings()
        d = d if isinstance(d, dict) else {}
        for k, (lo, hi) in LIMITS.items():
            if k in d:
                try:
                    v = float(d[k])
                except (TypeError, ValueError):
                    continue
                if math.isfinite(v):
                    v = min(max(v, lo), hi)
                    setattr(s, k, int(round(v)) if k in INTS else v)
        for k in ('http_bind', 'mqtt_host', 'mqtt_topic', 'mqtt_user', 'webhook_url', 'log_path', 'gpio_chip',
                  'gpio_path', 'service_user', 'python', 'config_path'):
            if isinstance(d.get(k), str):
                setattr(s, k, d[k].strip()[:400])
        if isinstance(d.get('knock_enabled'), bool):
            s.knock_enabled = d['knock_enabled']
        if d.get('gpio_backend') in GPIO_BACKENDS:
            s.gpio_backend = d['gpio_backend']
        if isinstance(d.get('sources'), dict):
            s.sources = {str(k)[:40]: str(v).strip()[:1000] for k, v in d['sources'].items() if str(v).strip()}
        if isinstance(d.get('health'), dict):
            from .rig_health import HealthSettings
            base = HealthSettings().to_dict()
            hs = HealthSettings.from_dict({k: v for k, v in d['health'].items() if k in base}).to_dict()
            s.health = {k: v for k, v in hs.items() if v != base[k]}
        if s.webhook_url and not s.webhook_url.lower().startswith(('http://', 'https://')):
            s.webhook_url = ''
        return s


def settings_of(project) -> EdgeSettings:
    return EdgeSettings.from_dict(getattr(project.analysis, 'edge_daemon', None))


def _variant(project, variant):
    if isinstance(variant, str):
        return project.variant_by_uid(variant) or project.variant
    return variant or project.variant


def source_of(project, cam, st: EdgeSettings) -> tuple[str, str]:
    """(source spec, where it came from) of a camera's stream on the car."""
    if st.sources.get(cam.tag):
        return st.sources[cam.tag], 'settings'
    s = (getattr(project, 'streams', {}) or {}).get(cam.tag, '').strip()
    if s and not s.startswith('synthetic'):
        return s, 'live view'
    if getattr(cam, 'port', 0):
        from .capture import default_spec
        return default_spec(int(cam.port)), f'GMSL port {cam.port}'
    return '', 'none'


def travel_dir_optical(cam) -> list:
    """The vehicle's forward axis (+x) in the camera's optical frame (x right, y down, z ahead) at its planned pose."""
    from .mathutil import BODY_TO_OPTICAL, rot_zyx
    d = BODY_TO_OPTICAL @ rot_zyx(cam.yaw, cam.pitch, cam.roll).T @ np.array([1.0, 0.0, 0.0])
    return [round(float(x), 9) for x in d]


def export_config(project, variant=None, settings: EdgeSettings | None = None, fps: dict | None = None,
                  engine=None, body_masks: bool = True) -> dict:
    """The watchdog's configuration (schema vantage.edge_config/1) for a variant's visible cameras. body_masks: carry
    where the plan puts the own body in each image (rig_health.expected_view, 160 px wide), so the knock check's body
    edge is looked for there only (needs the analysis engine: given, or made here)."""
    import datetime as _dt
    from .. import __version__
    from ..edge.config import RadialLens
    from .lens import resolve
    from .rig_health import HealthSettings, planned_offsets
    st = settings or settings_of(project)
    var = _variant(project, variant)
    rig = f'{project.name} · {var.name}'
    cams = []
    masks = {}
    if body_masks:
        try:
            from ..edge.knock import encode_mask
            from .rig_health import expected_view
            if engine is None:
                from .analysis import Engine
                engine = Engine()
            for c in (c for c in var.cameras if c.visible):
                ev = expected_view(project, engine, c, width=160)
                if ev.body.any():
                    masks[c.tag] = encode_mask(ev.body)
        except Exception:                              # noqa: BLE001 - without the plan's body the check still runs
            masks = {}
    for c in (c for c in var.cameras if c.visible):
        mod = project.module_for(c)
        intr = resolve(mod, project.lens_for(c))
        src, where = source_of(project, c, st)
        cams.append({'tag': c.tag, 'uid': c.uid, 'name': c.name, 'source': src, 'source_from': where,
                     'fps': float((fps or {}).get(c.tag, mod.fps) or 30.0), 'width': int(mod.width_px),
                     'height': int(mod.height_px), 'module': mod.id, 'port': int(getattr(c, 'port', 0) or 0),
                     'pose': {'x': round(float(c.x), 5), 'y': round(float(c.y), 5), 'z': round(float(c.z), 5),
                              'yaw': float(c.yaw), 'pitch': float(c.pitch), 'roll': float(c.roll), 'frame': FRAME},
                     'travel_dir_optical': travel_dir_optical(c),
                     'lens': RadialLens.from_intrinsics(intr),
                     **({'body_mask': masks[c.tag]} if c.tag in masks else {})})
    try:
        po = planned_offsets(project, var)
    except Exception:                                  # noqa: BLE001 - a plan that cannot be read: no offsets
        po = {}
    health = HealthSettings.from_dict(st.health).to_dict()
    tag_ok = ''.join(ch if ch.isalnum() or ch in '-_' else '-' for ch in project.name.lower()).strip('-') or 'rig'
    return {
        'schema': CONFIG_SCHEMA,
        'created': _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat(),
        'vantage': __version__, 'project': project.name, 'variant': var.name, 'rig': rig,
        'conventions': {'pixels': 'lens tables at the sensor\'s full resolution, its own coordinates (pixel centre at '
                                  'i + 0.5); streams at another size are scaled',
                        'optical_frame': 'x right, y down, z along the optical axis',
                        'pose': f'{FRAME}; yaw, pitch (positive down), roll in degrees, intrinsic Z-Y-X',
                        'planned_offsets': 'tag -> [capture phase ms, frame cycle ms] from the sync plan',
                        'times': 'seconds'},
        'cameras': cams,
        'planned_offsets': {k: [round(v[0], 6), round(v[1], 6)] for k, v in po.items()},
        'health': health,
        'knock': {'enabled': st.knock_enabled, 'analysis_hz': st.analysis_hz, 'analysis_width': st.analysis_width,
                  'threshold_deg': st.knock_deg, 'body_threshold_deg': st.body_knock_deg, 'z_min': st.knock_z,
                  'baseline_s': st.baseline_s, 'window_s': st.window_s, 'hold_s': st.hold_s,
                  'min_inlier_share': st.min_inlier_share},
        'alerts': {'raise_s': st.raise_s, 'clear_s': st.clear_s, 'grace_s': st.grace_s,
                   'http': {'bind': st.http_bind, 'port': st.http_port},
                   'mqtt': {'host': st.mqtt_host, 'port': st.mqtt_port,
                            'topic': st.mqtt_topic or f'vantage/{tag_ok}', 'user': st.mqtt_user},
                   'webhook': {'url': st.webhook_url},
                   'log': {'path': st.log_path, 'max_mb': st.log_mb, 'keep': 5},
                   'gpio': {'backend': st.gpio_backend, 'chip': st.gpio_chip, 'line': st.gpio_line,
                            'path': st.gpio_path, 'active_high': True, 'pattern': 'beep'}},
    }


def config_json(cfg: dict) -> str:
    return json.dumps(cfg, indent=1, ensure_ascii=False) + '\n'


def systemd_unit(settings: EdgeSettings | None = None, rig: str = '') -> str:
    """The service's systemd unit: runs the watchdog as an unprivileged user with the camera and GPIO groups,
    restarts it on failure, bounds its memory and CPU, and keeps it from writing anywhere but its log folder."""
    st = settings or EdgeSettings()
    log_dir = os.path.dirname(st.log_path) if st.log_path else ''
    rw = f'ReadWritePaths={log_dir}\n' if log_dir else ''
    logs = f'LogsDirectory={os.path.basename(log_dir)}\n' if log_dir.startswith('/var/log/') else ''
    rw_dev = ''
    if st.gpio_backend == 'file' and st.gpio_path:
        rw_dev = f'ReadWritePaths={os.path.dirname(st.gpio_path)}\n'
    return (
        '# Vantage edge watchdog (vantage.edge): rig health and knock alerts. It only reads streams and alerts.\n'
        f'# Install: copy to /etc/systemd/system/vantage-edge.service, the configuration to {st.config_path},\n'
        '# then: sudo systemctl daemon-reload && sudo systemctl enable --now vantage-edge\n'
        '[Unit]\n'
        f'Description=Vantage edge watchdog{" (" + rig + ")" if rig else ""}\n'
        'Documentation=https://github.com/Sherin-SEF-AI/Vantage\n'
        'After=network-online.target\nWants=network-online.target\n\n'
        '[Service]\nType=simple\n'
        f'User={st.service_user}\nGroup={st.service_user}\nSupplementaryGroups=video gpio\n'
        f'ExecStart={st.python} -m vantage.edge --config {st.config_path}\n'
        '# the MQTT password, if the broker needs one: Environment=VANTAGE_EDGE_MQTT_PASSWORD=… in a drop-in\n'
        'Restart=on-failure\nRestartSec=5\n'
        f'MemoryMax={st.memory_mb}M\nCPUQuota={st.cpu_percent}%\nNice=5\n'
        'NoNewPrivileges=yes\nProtectSystem=strict\nProtectHome=read-only\nPrivateTmp=yes\n'
        'ProtectKernelTunables=yes\nProtectControlGroups=yes\nRestrictSUIDSGID=yes\n'
        f'{logs}{rw}{rw_dev}'
        '\n[Install]\nWantedBy=multi-user.target\n')


def readme(cfg: dict, st: EdgeSettings) -> str:
    cams = '\n'.join(f'| {c["tag"]} | `{c["source"] or "(set a source)"}` | {c["fps"]:g} | '
                     f'{c["width"]}×{c["height"]} |' for c in cfg['cameras'])
    return f"""# Vantage edge watchdog · {cfg['rig']}

The watchdog reads the rig's camera streams on the car's computer, checks their health and whether a camera has been
knocked, and alerts on a local status page (http://{st.http_bind}:{st.http_port}/), MQTT, a webhook, its log file and
an optional buzzer line. It only reads and alerts.

## Install (Jetson Orin / Ubuntu, x86 or ARM64)

    sudo apt install python3-numpy python3-opencv      # or: pip install numpy opencv-python-headless
    pip install --no-deps vantage                      # the package without its desktop dependencies (Qt, OpenGL)
    sudo useradd --system --groups video,gpio vantage
    sudo install -D -m 644 edge.json {st.config_path}
    sudo install -m 644 vantage-edge.service /etc/systemd/system/
    sudo systemctl daemon-reload && sudo systemctl enable --now vantage-edge
    python3 -m vantage.edge --config {st.config_path} --check

ROS 2 topics (sources `ros2:/topic`) need ROS 2 sourced in the service's environment (an ExecStart through
`bash -c 'source /opt/ros/humble/setup.bash && exec python3 -m vantage.edge …'`).

## Cameras

| Camera | Source | fps | Size |
|---|---|---:|---|
{cams}

## Try it on a recording

    python3 -m vantage.edge --config edge.json --dry-run DRIVE_FOLDER --report run.json
"""


def install_files(project, variant=None, settings: EdgeSettings | None = None) -> list:
    st = settings or settings_of(project)
    cfg = export_config(project, variant, st)
    return [('edge.json', config_json(cfg).encode('utf-8')),
            ('vantage-edge.service', systemd_unit(st, cfg['rig']).encode('utf-8')),
            ('README.md', readme(cfg, st).encode('utf-8'))]


# ====================================================================================== the synthetic drive
class FrameSeries:
    """A camera's frames from a generator function (the synthetic drive) with known stamps."""

    def __init__(self, times, render):
        self.times = list(times)
        self.render = render
        self.stamped = True

    def __len__(self):
        return len(self.times)

    def frames(self):
        for i, t in enumerate(self.times):
            yield t, self.render(i, t)


def _texture(rng, n=1024, sigma=1.6):
    t = rng.normal(0.0, 1.0, (n, n)).astype(np.float32)
    t = _wrap_blur(t, sigma)                        # tileable: the road repeats every n texels without a seam
    t = (t - t.mean()) / (t.std() + 1e-9)
    return np.clip(128 + 45 * t, 0, 255).astype(np.float32)


def _wrap_blur(t, sigma):
    """A Gaussian blur that wraps round the edges (OpenCV's filters do not wrap)."""
    import cv2
    p = int(math.ceil(4 * sigma)) + 1
    big = np.pad(t, p, mode='wrap')
    return cv2.GaussianBlur(big, (0, 0), sigma)[p:-p, p:-p]


def _levels(tex, n=5):
    """The texture pre-filtered for 1, 2, 4 … texels per pixel (a mip chain kept at full size, so a far surface is not
    aliased into flicker that no camera would see)."""
    return [tex] + [_wrap_blur(tex, 0.6 * 2 ** k) for k in range(1, n)]


def synthetic_drive(cfg: dict, seconds: float = 10.0, fps: float = 10.0, speed_kmh: float = 30.0, width: int = 480,
                    seed: int = 0, body: tuple | None = None, noise: float = 1.5) -> dict:
    """{tag: FrameSeries}: a straight drive at speed_kmh along a road (textured ground, textured walls 7 m to either
    side, a bright sky), seen by each configured camera through its lens table at its planned pose, at width px. body:
    the own vehicle as a box (x0, x1, y0, y1, z0, z1) in the vehicle frame (None: no body), panelled so it has edges.
    Textures are filtered by distance (no aliasing). Sensor noise of noise grey levels (seeded) keeps consecutive
    frames different, as a real camera's are."""
    import cv2
    from ..edge.config import RadialLens
    from ..edge.inject import BODY_TO_OPTICAL, rot_zyx
    rng = np.random.default_rng(seed)
    lv_g, lv_w = _levels(_texture(rng)), _levels(_texture(rng, sigma=2.5))
    m_px = 0.03                                     # metres per texture pixel
    n = lv_g[0].shape[0]
    v = speed_kmh / 3.6
    times = np.arange(int(round(seconds * fps))) / fps
    out = {}
    for ci, c in enumerate(cfg['cameras']):
        lens = RadialLens(c['lens'])
        w = int(width)
        h = max(1, round(w * lens.height / lens.width))
        f_s = lens.f_px * w / lens.width
        uu, vv = np.meshgrid(np.arange(w, dtype=float), np.arange(h, dtype=float))
        d, th = lens.unproject(uu.ravel(), vv.ravel(), w, h)
        inside = np.isfinite(th) & (np.nan_to_num(th, nan=9.0) <= lens.max_theta)
        p = c['pose']
        Rc = rot_zyx(p['yaw'], p['pitch'], p['roll'])
        D = d @ (Rc @ BODY_TO_OPTICAL.T).T                 # vehicle-frame directions
        C = np.array([p['x'], p['y'], p['z']], float)
        inf = np.full(len(D), np.inf)
        with np.errstate(divide='ignore', invalid='ignore'):
            tg = np.where(D[:, 2] < -1e-6, -C[2] / D[:, 2], inf)
            tw = np.where(np.abs(D[:, 1]) > 1e-6, (np.sign(D[:, 1]) * 7.0 - C[1]) / D[:, 1], inf)
        tw = np.where(tw > 0, tw, inf)
        tb = inf.copy()
        if body is not None:
            lo, hi = np.array(body[0::2], float), np.array(body[1::2], float)
            with np.errstate(divide='ignore', invalid='ignore'):
                t1, t2 = (lo - C) / D, (hi - C) / D
            tn = np.nanmax(np.minimum(t1, t2), 1)
            tf = np.nanmin(np.maximum(t1, t2), 1)
            tb = np.where((tf >= tn) & (tf > 0), np.maximum(tn, 0.0), inf)
        tg = np.where(tg < 80.0, tg, inf)
        tw = np.where(tw < 80.0, tw, inf)
        T = np.stack([tg, tw, tb])
        which, tmin = np.argmin(T, 0), np.min(T, 0)
        hit = np.isfinite(tmin) & inside
        P = C + D * np.where(np.isfinite(tmin), tmin, 0.0)[:, None]
        # footprint of a pixel on the surface, in texels (stretched by the grazing angle), picks the filtered level
        with np.errstate(divide='ignore', invalid='ignore'):
            graze_g = np.abs(D[:, 2])
            graze_w = np.abs(D[:, 1])
            fp = np.where(which == 0, tmin / f_s / np.maximum(graze_g, 0.05), tmin / f_s / np.maximum(graze_w, 0.05))
        lvl = np.clip(np.floor(np.log2(np.maximum(fp / m_px, 1.0))), 0, len(lv_g) - 1).astype(np.int8).reshape(h, w)
        shade_b = 70 + 35 * ((np.floor(P[:, 0] / 0.35) + np.floor(P[:, 2] / 0.25)) % 2) + 25 * (P[:, 2] > 1.1)
        base = np.full(len(D), 215.0, np.float32)            # sky
        base[~inside] = 0.0
        isb = hit & (which == 2)
        base[isb] = shade_b[isb]
        static = base.reshape(h, w)
        isg = (hit & (which == 0)).reshape(h, w)
        isw = (hit & (which == 1)).reshape(h, w)
        Pr = P.reshape(h, w, 3).astype(np.float32)
        crng = np.random.default_rng([seed, ci])

        def sample(levels, mx, my, lvl, sel):
            mx = np.mod(mx, n).astype(np.float32)
            my = np.mod(my, n).astype(np.float32)
            out_ = np.zeros(mx.shape, np.float32)
            for k in np.unique(lvl[sel]):
                m = sel & (lvl == k)
                out_[m] = cv2.remap(levels[int(k)], mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)[m]
            return out_

        def render(i, t, Pr=Pr, isg=isg, isw=isw, static=static, crng=crng, lvl=lvl):
            x0 = v * t
            img = static.copy()
            if isg.any():
                g = sample(lv_g, (Pr[..., 0] + x0) / m_px, Pr[..., 1] / m_px, lvl, isg)
                img[isg] = g[isg]
            if isw.any():
                ww = sample(lv_w, (Pr[..., 0] + x0) / m_px, Pr[..., 2] / m_px + 7.0 * (Pr[..., 1] > 0) / m_px, lvl,
                            isw)
                img[isw] = 0.8 * ww[isw] + 20
            if noise:
                img = img + crng.normal(0.0, noise, img.shape).astype(np.float32)
            g8 = np.clip(img, 0, 255).astype(np.uint8)
            return np.repeat(g8[..., None], 3, 2)
        out[c['tag']] = FrameSeries(times, render)
    return out


def body_box(project) -> tuple:
    """The own vehicle as a box (x0, x1, y0, y1, z0, z1), vehicle frame: its mesh's extent without the mirrors (the
    width from the vehicle's spec when it has one) and from 0.25 m up."""
    V = np.asarray(project.vehicle.mesh.vertices, float)
    lo, hi = V.min(0), V.max(0)
    width = float(getattr(project.vehicle.spec, 'width', 0) or 0) if getattr(project.vehicle, 'spec', None) else 0.0
    half = width / 2 if width > 0 else 0.82 * max(hi[1], -lo[1])
    return (float(lo[0]), float(hi[0]), -half, half, 0.25, float(hi[2]))


# ====================================================================================== dry runs
@dataclass
class DryRunResult:
    report: dict                     # vantage.edge's run record
    faults: list                     # the faults written in, each with how it was caught
    false_alarms: list               # alerts no fault explains
    metrics: dict
    settings: EdgeSettings
    variant_uid: str = ''
    recording: str = ''


def _knock_matrix(f) -> np.ndarray:
    from ..edge.inject import BODY_TO_OPTICAL, rot_zyx
    return BODY_TO_OPTICAL @ rot_zyx(f['yaw'], f['pitch'], f['roll']) @ BODY_TO_OPTICAL.T


def _deflection(M, ray) -> float:
    r = np.asarray(ray, float)
    r = r / np.linalg.norm(r)
    return float(math.degrees(math.acos(max(-1.0, min(1.0, float(r @ (M.T @ r)))))))


def score(report: dict, faults: list, plans: dict | None = None) -> tuple[list, list]:
    """(faults, each with 'caught', 'check', 'delay_s' and, for a knock, the size each method measured against how
    far the knock really moved what that method looks at (the direction of travel: plans[tag], the planned direction
    in the optical frame; the body edge: the median over its edge pixels' rays)), false alarms."""
    from ..edge.inject import clean
    alerts = report.get('alerts', [])
    t0 = min([tl['t'] for tl in report.get('timeline', [])[:1]] or [0.0])
    cams = report.get('cameras', {})
    used = set()
    out = []
    for f in clean(faults):
        want = FAULT_CHECKS[f['kind']]
        start = t0 + f['at_s']
        hits = [a for a in alerts if (f['camera'] in ('*', a['camera'])) and a['check'] in want
                and a['raised_t'] >= start - 1e-6]
        g = dict(f, caught=bool(hits), check=None, delay_s=None)
        if hits:
            a = min(hits, key=lambda a: a['raised_t'])
            g.update(check=a['check'], delay_s=round(float(a['raised_t'] - start), 3), message=a['message'])
            used.update(id(x) for x in hits)
        if f['kind'] == 'knock':
            k = (cams.get(f['camera']) or {}).get('knock') or {}
            M = _knock_matrix(f)
            rot = math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(M) - 1) / 2))))
            g.update(rotation_deg=round(rot, 4), size_deg=k.get('size_deg'), method=k.get('method'))
            plan = (plans or {}).get(f['camera'])
            kt, kb = k.get('travel') or {}, k.get('body') or {}
            for key, rays, meas in (('travel', None if plan is None else [plan], kt.get('size_deg')),
                                    ('body', kb.get('edge_rays'), kb.get('size_deg'))):
                if not rays:
                    continue
                true = float(np.median([_deflection(M, r) for r in rays]))   # how far the knock moves what it sees
                g[f'{key}_true_deg'] = round(true, 4)
                g[f'{key}_size_deg'] = meas
                g[f'{key}_error_deg'] = None if meas is None else round(meas - true, 4)
            errs = [g[f'{m}_error_deg'] for m in ('travel', 'body') if g.get(f'{m}_error_deg') is not None]
            g['size_error_deg'] = None if not errs else (g.get('travel_error_deg') if
                                                         g.get('travel_error_deg') is not None else errs[0])
        out.append(g)
    explained = set()
    for a in alerts:
        if id(a) in used:
            explained.add(id(a))
            continue
        for f in clean(faults):                   # a fault's side effects on the same camera are explained by it
            if f['camera'] in ('*', a['camera']) and a['raised_t'] >= t0 + f['at_s'] - 1e-6 and \
                    f['kind'] in ('stall', 'drop', 'freeze', 'dark'):
                explained.add(id(a))
            if a['camera'] == 'rig' and a['check'] in ('sync', 'host') and f['kind'] in ('stall', 'drop', 'freeze'):
                explained.add(id(a))
    false = [a for a in alerts if id(a) not in explained]
    return out, false


def dry_run(project, recording=None, faults=None, settings: EdgeSettings | None = None, variant=None,
            cfg: dict | None = None, seconds: float = 12.0, fps: float = 10.0, speed_kmh: float = 30.0,
            width: int = 480, seed: int = 0, realtime: bool = False, progress=None, cancel=None) -> DryRunResult:
    """Run the watchdog over a recording (a path: nuScenes folder, image folders, video files, a ROS 2 bag) or, with
    recording None, over a synthetic drive of the project's cameras; faults: vantage.edge.inject's dicts."""
    from ..edge.daemon import dry_run as _dry
    st = settings or (settings_of(project) if project is not None else EdgeSettings())
    var = _variant(project, variant) if project is not None else None
    if cfg is None:
        cfg = export_config(project, var, st)
    cfg = json.loads(json.dumps(cfg))
    cfg['alerts']['http']['port'] = 0
    cfg['alerts']['log']['path'] = ''
    if recording is None:
        box = body_box(project) if project is not None else None
        rec = synthetic_drive(cfg, seconds, fps, speed_kmh, width, seed, box)
        where = f'synthetic drive ({seconds:g} s at {speed_kmh:g} km/h, {fps:g} fps, {width} px wide)'
    else:
        rec, where = recording, os.path.abspath(recording)
    rep = _dry(cfg, rec, faults, realtime=realtime, notify=False, tail_s=1.0, progress=progress, cancel=cancel)
    plans = {c['tag']: c.get('travel_dir_optical') for c in cfg['cameras']}
    scored, false = score(rep, faults or [], plans)
    caught = sum(f['caught'] for f in scored)
    delays = [f['delay_s'] for f in scored if f['delay_s'] is not None]
    kn = [f for f in scored if f['kind'] == 'knock' and f.get('size_error_deg') is not None]
    cost = rep.get('cost', {})
    frames = rep['recording']['frames']
    secs = float(rep.get('seconds') or 0.0)
    metrics = {'faults': len(scored), 'caught': caught, 'missed': len(scored) - caught, 'false_alarms': len(false),
               'worst_delay_s': max(delays) if delays else None,
               'knock_size_error_deg': max((abs(f['size_error_deg']) for f in kn), default=None),
               'frames': frames, 'recording_s': secs, 'cameras': len(rep['recording']['cameras']),
               'cpu_s': cost.get('cpu_s'), 'wall_s': cost.get('wall_s'),
               'cpu_ms_per_frame': round(1000 * cost['cpu_s'] / frames, 2) if frames and cost.get('cpu_s') else None,
               'cpu_percent_at_recorded_rate': (round(100 * cost['cpu_s'] / secs, 1) if secs and cost.get('cpu_s')
                                                else None),
               'peak_rss_mb': cost.get('peak_rss_mb'), 'cancelled': rep.get('cancelled', False)}
    return DryRunResult(rep, scored, false, metrics, st, var.uid if var is not None else '', where)


def exit_code(res: DryRunResult | None) -> int:
    """0: every injected fault caught and no false alarm; 1: a fault missed or a false alarm; 2: no result."""
    if res is None or res.metrics.get('cancelled'):
        return 2
    return 0 if res.metrics['missed'] == 0 and res.metrics['false_alarms'] == 0 else 1


def summary(res: DryRunResult) -> dict:
    m = res.metrics
    return {'recording': res.recording, 'cameras': res.report['recording']['cameras'], **m,
            'faults': [{k: f.get(k) for k in ('kind', 'camera', 'at_s', 'caught', 'check', 'delay_s', 'method',
                                              'size_deg', 'travel_true_deg', 'travel_size_deg', 'body_true_deg',
                                              'body_size_deg', 'size_error_deg') if k in f} for f in res.faults],
            'false_alarm_list': [{'camera': a['camera'], 'check': a['check'], 'at_s': round(a['raised_t'], 2),
                                  'message': a['message']} for a in res.false_alarms],
            'notes': res.report.get('notes', [])}


def to_doc(res: DryRunResult, project=None) -> dict:
    var = project.variant_by_uid(res.variant_uid) if project is not None and res.variant_uid else None
    rep = res.report
    return {'schema': SCHEMA, 'project': getattr(project, 'name', ''), 'variant': var.name if var else '',
            'frame': FRAME, 'units': {'time': 's', 'angles': 'deg', 'image': 'px'},
            'recording': res.recording, 'assumptions': list(ASSUMPTIONS), 'settings': res.settings.to_dict(),
            'metrics': res.metrics, 'faults': res.faults,
            'false_alarms': res.false_alarms, 'alerts': rep.get('alerts', []), 'notes': rep.get('notes', []),
            'cameras': {t: {k: c.get(k) for k in ('status', 'reasons', 'fps', 'nominal_fps', 'frames', 'dropped',
                                                   'frozen_frames', 'mean', 'knock')}
                        for t, c in rep.get('cameras', {}).items()},
            'timeline': rep.get('timeline', [])}


CSV_COLUMNS = ('id', 'camera', 'check', 'severity', 'raised_s', 'cleared_s', 'message', 'explained_by')


def to_csv(res: DryRunResult) -> str:
    import csv
    import io
    false = {id(a) for a in res.false_alarms}
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(CSV_COLUMNS)
    for a in res.report.get('alerts', []):
        w.writerow([a['id'], a['camera'], a['check'], a['severity'], round(a['raised_t'], 3),
                    '' if a.get('cleared_t') is None else round(a['cleared_t'], 3), a['message'],
                    'false alarm' if id(a) in false else 'injected fault'])
    return buf.getvalue()


def files(res: DryRunResult, project=None) -> list:
    return [('edge_dry_run.json', (json.dumps(to_doc(res, project), indent=1, default=str) + '\n').encode('utf-8')),
            ('edge_alerts.csv', to_csv(res).encode('utf-8'))]


def export(res: DryRunResult, project, out_dir: str) -> list:
    """edge_dry_run.json, edge_alerts.csv and the install files (edge.json, vantage-edge.service, README.md)."""
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    items = files(res, project)
    if project is not None:
        items += install_files(project, res.variant_uid or None, res.settings)
    for name, data in items:
        p = os.path.join(out_dir, name)
        with open(p, 'wb') as fh:
            fh.write(data)
        paths.append(p)
    return paths


def export_install(project, out_dir: str, variant=None, settings=None) -> list:
    os.makedirs(out_dir, exist_ok=True)
    out = []
    for name, data in install_files(project, variant, settings):
        p = os.path.join(out_dir, name)
        with open(p, 'wb') as fh:
            fh.write(data)
        out.append(p)
    return out


# ====================================================================================== validation
def _v_travel(ctx):
    """The travel-direction solver against known motions: rays of random static points (a road and walls 3-60 m
    away) seen from two positions of a camera moving along a known direction, with 0.5 px noise and 15 % of the
    matches replaced by wrong ones (moving traffic, mistracks)."""
    from ..edge.knock import translation_from_rays
    from ..validation.suite import Outcome
    rng = ctx.rng('edge.travel_direction')
    errs = []
    f = 300.0
    for k in range(30 if ctx.quick else 100):
        d = rng.normal(size=3)
        d[2] = abs(d[2]) + 0.2
        d /= np.linalg.norm(d)
        P = np.column_stack([rng.uniform(-15, 15, 300), rng.uniform(-3, 3, 300), rng.uniform(3, 60, 300)])
        X2 = P - d * 0.7
        ok = X2[:, 2] > 1.0
        P, X2 = P[ok], X2[ok]
        n1 = P[:, :2] / P[:, 2:3] + rng.normal(0, 0.5 / f, (len(P), 2))
        n2 = X2[:, :2] / X2[:, 2:3] + rng.normal(0, 0.5 / f, (len(P), 2))
        bad = rng.uniform(size=len(P)) < 0.15
        n2[bad] += rng.normal(0, 20 / f, (bad.sum(), 2))
        r1 = np.column_stack([n1, np.ones(len(P))])
        r2 = np.column_stack([n2, np.ones(len(P))])
        res = translation_from_rays(r1 / np.linalg.norm(r1, axis=1, keepdims=True),
                                    r2 / np.linalg.norm(r2, axis=1, keepdims=True), thresh=1.0 / f, seed=k)
        if res is None:
            errs.append(180.0)
            continue
        errs.append(math.degrees(math.acos(min(1.0, float(res[0] @ d)))))
    e = np.array(errs)
    return Outcome(float(np.percentile(e, 95)), {'pairs': len(e), 'median_deg': float(np.median(e)),
                                                 'p95_deg': float(np.percentile(e, 95)), 'max_deg': float(e.max())})


def _v_faults(ctx):
    """Faults written into a synthetic drive of the sample's data rig: each must be caught by its check, and nothing
    else raised."""
    from ..validation.suite import Outcome
    p = ctx.sample()
    var = next(v for v in p.variants if v.sensors)
    faults = [{'kind': 'knock', 'camera': 'ML', 'at_s': 6.0, 'yaw': 2.0},
              {'kind': 'freeze', 'camera': 'F', 'at_s': 5.0, 'dur_s': 3.0},
              {'kind': 'stall', 'camera': 'QR', 'at_s': 7.0, 'dur_s': 0.0},
              {'kind': 'dark', 'camera': 'R', 'at_s': 4.0, 'dur_s': 4.0},
              {'kind': 'drop', 'camera': 'MR', 'at_s': 4.0, 'dur_s': 5.0, 'keep_every': 3}]
    res = dry_run(p, None, faults, variant=var, seconds=12.0, fps=10.0, width=320 if ctx.quick else 400)
    m = res.metrics
    return Outcome(float(m['missed'] + m['false_alarms']),
                   {'faults': m['faults'], 'caught': m['caught'], 'false alarms': m['false_alarms'],
                    'worst delay s': m['worst_delay_s'], 'knock size error deg': m['knock_size_error_deg']},
                   ok=m['caught'] == len(faults))


def _v_inject(ctx):
    """The knock injection is a rotation: a pixel's ray after the re-rendering is the original ray rotated by the
    injected angles (checked through the lens table on a grid)."""
    from ..edge.config import RadialLens
    from ..edge.inject import BODY_TO_OPTICAL, rot_zyx
    from ..validation.suite import Outcome
    from .lens import resolve
    p = ctx.sample()
    c = next(v for v in p.variants if v.sensors).cameras[0]
    lens = RadialLens(RadialLens.from_intrinsics(resolve(p.module_for(c), p.lens_for(c))))
    w, h = 480, 384
    uu, vv = np.meshgrid(np.linspace(20, w - 20, 25), np.linspace(20, h - 20, 20))
    d, th = lens.unproject(uu.ravel(), vv.ravel(), w, h)
    ok = np.isfinite(th) & (th < lens.max_theta - 0.1)
    M = BODY_TO_OPTICAL @ rot_zyx(1.5, -0.7, 0.4) @ BODY_TO_OPTICAL.T
    u2, v2, ok2 = lens.project(d[ok] @ M.T, w, h)
    d3, _ = lens.unproject(u2[ok2], v2[ok2], w, h)
    ang = np.degrees(np.arccos(np.clip(np.sum(d3 * (d[ok][ok2] @ M.T), 1), -1, 1)))
    return Outcome(float(ang.max()), {'rays': int(ok2.sum()), 'max_deg': float(ang.max())})


def validation_checks() -> list:
    """Spec-compatible dicts for vantage.validation (the lead registers them under a model 'edge')."""
    return [
        {'id': 'edge.travel_direction', 'model': 'edge',
         'title': 'The direction of travel from two frames is the camera\'s true motion',
         'quantity': '95th percentile angle between the recovered and the true displacement direction (sign '
                     'included) over random scenes of 300 static points 3–60 m away, a 0.7 m step, 0.5 px noise and '
                     '15 % wrong matches',
         'reference': 'the simulated camera motion (known exactly)', 'tolerance': 1.2, 'unit': 'deg',
         'refs': ('hartley2004',), 'fn': _v_travel, 'cost': 'cheap'},
        {'id': 'edge.knock_injection', 'model': 'edge',
         'title': 'An injected knock re-renders the image as the rotated camera sees it',
         'quantity': 'largest angle between a re-rendered pixel\'s ray and the original ray rotated by the knock',
         'reference': 'the rotation applied to the rays in closed form', 'tolerance': 0.01, 'unit': 'deg',
         'refs': (), 'fn': _v_inject, 'cost': 'cheap'},
        {'id': 'edge.fault_detection', 'model': 'edge',
         'title': 'The watchdog catches every injected fault and raises nothing else',
         'quantity': 'missed faults plus false alarms on a 12 s synthetic drive of the sample\'s six cameras with a '
                     '2° knock, a frozen stream, a stall, a dark image and dropped frames',
         'reference': 'the injected faults (where and when they were written in)', 'tolerance': 0.0,
         'unit': 'alerts', 'refs': (), 'fn': _v_faults, 'cost': 'study'},
    ]
