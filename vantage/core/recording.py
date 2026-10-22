"""Recording budget: how much data the rig records, whether the links, the network and the recorder's disks keep up,
and how many hours of driving fit on one disk set.

Every visible camera, lidar and radar of the variant records at a data rate:
- a camera: width × height × frames per second × the bits per pixel of the recorded format: raw (the ECU's pixel
  format, ecu.bpp: 16 = YUV 4:2:2, the same figure as the GMSL link budget), JPEG or H.265 at an average bits per pixel.
  The compressed figures are assumptions (typical for driving scenes at good quality); replace them with ones measured
  on your encoder and your scenes.
- a lidar: its points per second × bytes per point, either as the driver delivers the packets (the model's
  bytes_per_point, headers included) or as a decoded cloud (point_bytes: x y z intensity ring time).
- a radar: its stated output rate.
On top come GNSS, IMU and CAN (extra_mbps) and the container's own overhead (message headers, indexes) as a share of
all of it.

Links: a camera's GMSL2 link carries its raw pixels whatever is recorded (rules.camera_gbps against ecu.link_gbps;
rules.link_budget raises the issues for those). A lidar or radar sends its packets over the link its interface names
(1000BASE-T1 …, the fastest of 'A / B'), and every recorded lidar and radar shares the uplink from the switch to the
recorder.

Units are decimal, as disks and links are sold: Mbit/s = 10⁶ bit/s, MB = 10⁶ bytes, GB = 10⁹, TB = 10¹².
Cheap (no geometry): computed again on every edit.
"""
from __future__ import annotations

import csv
import io
import json
import math
import os
import re
from dataclasses import asdict, dataclass

from . import rules as R

SCHEMA = 'vantage.recording_budget/1'
CAMERA_FORMATS = ('raw', 'jpeg', 'h265')
LIDAR_FORMATS = ('packets', 'points')
FORMAT_LABEL = {'raw': 'Raw', 'jpeg': 'JPEG', 'h265': 'H.265', 'packets': 'Packets', 'points': 'Point cloud'}
SEVERITIES = ('critical', 'serious', 'warning', 'good')
# findings that go to the Issues list; a camera's GMSL link is rules.link_budget's, an unknown interface only a note
ISSUE_KINDS = ('write', 'disk', 'uplink', 'link')
ASSUMPTIONS = ('JPEG and H.265 bits per pixel are typical averages for driving scenes, not measurements: replace them '
               'with figures from your encoder. Lidar packet sizes are the library models\' nominal figures.')
# accepted range of every number in the settings (values outside are clamped, as the perception settings do)
LIMITS = {'jpeg_bpp': (0.01, 32.0), 'h265_bpp': (0.001, 32.0), 'point_bytes': (1.0, 256.0),
          'extra_mbps': (0.0, 100000.0), 'container_overhead': (0.0, 1.0), 'storage_tb': (0.01, 10000.0),
          'write_mb_s': (1.0, 1e6), 'write_headroom': (1.0, 5.0), 'shift_hours': (0.25, 24.0),
          'uplink_gbps': (0.01, 1000.0), 'link_warn': (0.1, 1.0)}


@dataclass
class RecordingSettings:
    camera_format: str = 'raw'         # raw (ecu.bpp) | jpeg | h265
    jpeg_bpp: float = 2.0              # bits per pixel, an assumption
    h265_bpp: float = 0.15             # bits per pixel, an assumption
    lidar_format: str = 'packets'      # packets (the model's bytes_per_point) | points (point_bytes, decoded)
    point_bytes: float = 32.0          # x y z intensity ring time
    include_radars: bool = True
    extra_mbps: float = 2.0            # GNSS, IMU, CAN
    container_overhead: float = 0.03   # message headers and indexes, a share of everything recorded
    storage_tb: float = 8.0            # one disk set
    write_mb_s: float = 1500.0         # sustained write, after any SLC cache is full
    write_headroom: float = 1.3        # the disks must sustain this many times the data rate
    shift_hours: float = 8.0
    uplink_gbps: float = 10.0          # switch to recorder, shared by the lidars and radars
    link_warn: float = 0.8             # a link busier than this share is flagged

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d: dict | None) -> 'RecordingSettings':
        s = RecordingSettings()
        d = d if isinstance(d, dict) else {}
        if d.get('camera_format') in CAMERA_FORMATS:
            s.camera_format = d['camera_format']
        if d.get('lidar_format') in LIDAR_FORMATS:
            s.lidar_format = d['lidar_format']
        v = d.get('include_radars')
        if isinstance(v, (bool, int)):
            s.include_radars = bool(v)
        elif isinstance(v, str):
            s.include_radars = v.strip().lower() in ('1', 'true', 'yes', 'on')
        for k, (lo, hi) in LIMITS.items():
            if k in d:
                try:
                    v = float(d[k])
                except (TypeError, ValueError):
                    continue
                if math.isfinite(v):
                    setattr(s, k, min(max(v, lo), hi))
        return s

    def camera_bpp(self, ecu) -> float:
        """Recorded bits per pixel of a camera image."""
        return {'raw': float(ecu.bpp), 'jpeg': self.jpeg_bpp, 'h265': self.h265_bpp}[self.camera_format]


@dataclass
class RecordingResult:
    settings: RecordingSettings
    rows: list                  # per visible camera, lidar and radar: dict (see budget)
    totals: dict
    findings: list              # [{'severity', 'kind', 'message', 'tags', 'uids'}], worst first
    raw_bpp: float = 16.0       # the ECU's pixel format the raw figures use
    variant: str = ''
    variant_uid: str = ''

    def row(self, tag):
        return next((r for r in self.rows if r['tag'] == tag), None)

    def worst(self) -> str:
        return self.findings[0]['severity'] if self.findings else 'good'


# =========================================================================== links
_BASE = re.compile(r'(\d+(?:\.\d+)?)\s*(G?)BASE', re.I)


def link_capacity_mbps(interface: str) -> float | None:
    """Capacity of a sensor's link in Mbit/s from its interface: 10GBASE-T → 10000, 2.5GBASE-T → 2500,
    1000BASE-T/T1 → 1000, 100BASE-TX/T1 → 100, CAN-FD → 5 (data phase), CAN → 1. 'A / B' takes the fastest; None
    when no part names a known link."""
    best = None
    for part in re.split(r'[/,|+]|\bor\b', str(interface or '')):
        part = part.strip()
        if not part:
            continue
        m = _BASE.search(part)
        if m:
            cap = float(m.group(1)) * (1000.0 if m.group(2) else 1.0)
        elif re.search(r'CAN[\s-]*FD', part, re.I):
            cap = 5.0
        elif re.search(r'\bCAN\b', part, re.I):
            cap = 1.0
        else:
            continue
        best = cap if best is None else max(best, cap)
    return best


def _num(v, default=0.0) -> float:
    """A model figure as a finite, non-negative float (a hand-edited library entry may hold anything)."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) and v >= 0 else default


def _link_severity(load, warn, over='critical'):
    if load is None:
        return None
    return over if load > 1.0 + 1e-9 else 'warning' if load > warn + 1e-9 else 'good'


# =========================================================================== the budget
def budget(project, variant=None, settings: RecordingSettings | None = None) -> RecordingResult:
    """The recording budget of a variant (default the active one). Rows: every visible camera, lidar and radar with
    tag, kind, model, slot (a camera's colour slot; a sensor's index among its kind), format, recorded, mbps, mb_s,
    gb_h (the sensor's own data, before the container overhead), link, port, link_mbps (what its link carries),
    link_capacity_mbps (None when the interface is not known), load and link_severity."""
    st = settings or RecordingSettings.from_dict(getattr(project.analysis, 'recording', None))
    v = variant or project.variant
    e = project.ecu
    rows = []
    bpp = st.camera_bpp(e)
    gmsl = _num(e.link_gbps) * 1000.0
    for c in v.cameras:
        if not c.visible:
            continue
        m = project.module_for(c)
        mbps = _num(m.width_px) * _num(m.height_px) * _num(m.fps) * bpp / 1e6
        wire = R.camera_gbps(project, c) * 1000.0
        load = wire / gmsl if gmsl > 0 else None
        fmt = f'{FORMAT_LABEL[st.camera_format]} · {bpp:g} bit/px · {m.width_px} × {m.height_px} @ {m.fps:g}'
        rows.append(_row(c.uid, c.tag, 'camera', m.name, c.slot, fmt, True, mbps, m.interface or 'GMSL2',
                         f'{c.port}' if c.port else '', wire, gmsl if gmsl > 0 else None, load,
                         _link_severity(load, st.link_warn, 'serious')))
    count = {'lidar': 0, 'radar': 0}
    for s in v.sensors:
        if not s.visible:
            continue
        m = project.sensor_model_for(s)
        wire = _safe(m.data_bps) / 1e6
        if m.is_lidar:
            recorded = True
            pps = _safe(m.pps)
            if st.lidar_format == 'points':
                mbps = pps * st.point_bytes * 8 / 1e6
                fmt = f'Point cloud · {st.point_bytes:g} B/pt · {pps / 1e6:.2f} Mpts/s'
            else:
                mbps = wire
                fmt = f'Packets · {_num(m.bytes_per_point):g} B/pt · {pps / 1e6:.2f} Mpts/s'
        else:
            recorded = st.include_radars
            mbps = wire if recorded else 0.0
            fmt = f'Stated output · {_num(m.data_mbps):g} Mbit/s' if recorded else 'Not recorded'
        cap = link_capacity_mbps(m.interface)
        load = wire / cap if cap else None
        kind = 'lidar' if m.is_lidar else 'radar'
        rows.append(_row(s.uid, s.tag, kind, f'{m.vendor} {m.name}'.strip(), count[kind], fmt, recorded, mbps,
                         m.interface or '—', s.port or '', wire, cap, load, _link_severity(load, st.link_warn)))
        count[kind] += 1
    totals = _totals(st, rows)
    res = RecordingResult(st, rows, totals, [], float(e.bpp), v.name, v.uid)
    res.findings = _findings(res, e)
    return res


def _safe(fn) -> float:
    """A model's derived figure (points/s, bit/s); 0 when its fields do not compute."""
    try:
        return _num(fn())
    except (TypeError, ValueError, ZeroDivisionError):
        return 0.0


def _row(uid, tag, kind, model, slot, fmt, recorded, mbps, link, port, link_mbps, cap, load, sev):
    return {'uid': uid, 'tag': tag, 'kind': kind, 'model': model, 'slot': int(slot), 'format': fmt,
            'recorded': bool(recorded), 'mbps': mbps, 'mb_s': mbps / 8.0, 'gb_h': mbps / 8.0 * 3600.0 / 1000.0,
            'link': link, 'port': port, 'link_mbps': link_mbps, 'link_capacity_mbps': cap, 'load': load,
            'link_severity': sev}


def _totals(st: RecordingSettings, rows) -> dict:
    by_kind = {k: sum(r['mbps'] for r in rows if r['kind'] == k) for k in ('camera', 'lidar', 'radar')}
    sensors = sum(by_kind.values())
    payload = sensors + st.extra_mbps
    overhead = payload * st.container_overhead
    mbps = payload + overhead
    mb_s = mbps / 8.0
    gb_h = mb_s * 3600.0 / 1000.0
    tb_shift = gb_h * st.shift_hours / 1000.0
    hours = st.storage_tb * 1e6 / mb_s / 3600.0 if mb_s > 0 else math.inf
    net = [r for r in rows if r['kind'] != 'camera' and r['recorded']]
    uplink = sum(r['link_mbps'] for r in net)
    up_cap = st.uplink_gbps * 1000.0
    need = mb_s * st.write_headroom
    return {'mbps': mbps, 'mb_s': mb_s, 'gb_h': gb_h, 'tb_shift': tb_shift, 'by_kind_mbps': by_kind,
            'sensor_mbps': sensors, 'extra_mbps': st.extra_mbps, 'overhead_mbps': overhead,
            'hours_per_set': hours, 'sets_per_shift': max(1, math.ceil(tb_shift / st.storage_tb - 1e-9)),
            'write_needed_mb_s': need, 'write_mb_s': st.write_mb_s, 'write_load': need / st.write_mb_s,
            'uplink_mbps': uplink, 'uplink_capacity_mbps': up_cap, 'uplink_load': uplink / up_cap,
            'networked': len(net),
            'counts': {k: sum(1 for r in rows if r['kind'] == k) for k in ('camera', 'lidar', 'radar')}}


def _finding(out, sev, kind, message, rows=()):
    out.append({'severity': sev, 'kind': kind, 'message': message, 'tags': [r['tag'] for r in rows],
                'uids': [r['uid'] for r in rows]})


def _findings(res: RecordingResult, ecu) -> list:
    st, t, rows = res.settings, res.totals, res.rows
    out = []
    warn = st.link_warn * 100
    # ---- the recorder's disks: sustained write and capacity
    if t['mb_s'] > 0:
        if t['write_needed_mb_s'] > st.write_mb_s + 1e-9:
            plain = t['mb_s'] > st.write_mb_s + 1e-9
            _finding(out, 'critical', 'write',
                     f'The recorder cannot write the data: {t["mb_s"]:,.0f} MB/s'
                     + (f' against {st.write_mb_s:,.0f} MB/s sustained write, even before the {st.write_headroom:g}× '
                        'headroom.' if plain else
                        f' × {st.write_headroom:g} headroom needs {t["write_needed_mb_s"]:,.0f} MB/s sustained, the '
                        f'disks write {st.write_mb_s:,.0f} MB/s.')
                     + ' Use faster disks (or more of them striped), compress the cameras or lower a frame rate.')
        else:
            _finding(out, 'good', 'write',
                     f'The disks keep up: {t["write_needed_mb_s"]:,.0f} of {st.write_mb_s:,.0f} MB/s sustained write '
                     f'needed ({t["mb_s"]:,.0f} MB/s × {st.write_headroom:g} headroom, '
                     f'{t["write_load"] * 100:.0f} %).')
        h = t['hours_per_set']
        if h < st.shift_hours - 1e-9:
            _finding(out, 'warning', 'disk',
                     f'One {st.storage_tb:g} TB disk set holds {h:.1f} h of driving, less than the '
                     f'{st.shift_hours:g} h shift: {t["sets_per_shift"]} sets ({t["tb_shift"]:.1f} TB) per shift.')
        else:
            _finding(out, 'good', 'disk',
                     f'One {st.storage_tb:g} TB disk set holds {h:.1f} h of driving, a whole '
                     f'{st.shift_hours:g} h shift ({t["tb_shift"]:.1f} TB).')
    # ---- the uplink from the switch to the recorder
    if t['networked']:
        load = t['uplink_load']
        net = [r for r in rows if r['kind'] != 'camera' and r['recorded']]
        what = f'The lidars and radars send {t["uplink_mbps"]:,.1f} Mbit/s to the recorder'
        cap = f'{st.uplink_gbps:g} Gbit/s uplink'
        if load > 1.0 + 1e-9:
            _finding(out, 'serious', 'uplink', f'{what}, over the {cap} ({load * 100:.0f} %). Use a faster uplink or '
                                               'a second recorder port.', net)
        elif load > st.link_warn + 1e-9:
            _finding(out, 'warning', 'uplink', f'{what}: {load * 100:.0f} % of the {cap}, above the {warn:.0f} % '
                                               'margin.', net)
        else:
            _finding(out, 'good', 'uplink', f'{what}: {load * 100:.0f} % of the {cap}.')
    # ---- each lidar's and radar's own link
    sens = [r for r in rows if r['kind'] != 'camera']
    for r in sens:
        cap, load = r['link_capacity_mbps'], r['load']
        if cap is None:
            _finding(out, 'warning', 'link_unknown', f'{r["tag"]}: the link rate of "{r["link"]}" is not known, so its '
                                                     'load is not checked.', [r])
        elif r['link_severity'] == 'critical':
            _finding(out, 'critical', 'link',
                     f'{r["tag"]} sends {r["link_mbps"]:,.1f} Mbit/s over its {r["link"]} link ({cap:g} Mbit/s): '
                     f'{load * 100:.0f} %. Use a faster link or a lower point rate.', [r])
        elif r['link_severity'] == 'warning':
            _finding(out, 'warning', 'link',
                     f'{r["tag"]} uses {load * 100:.0f} % of its {r["link"]} link ({r["link_mbps"]:,.1f} of '
                     f'{cap:g} Mbit/s), above the {warn:.0f} % margin.', [r])
    known = [r for r in sens if r['load'] is not None]
    if known and all(r['link_severity'] == 'good' for r in known):
        b = max(known, key=lambda r: r['load'])
        _finding(out, 'good', 'link', f'Every lidar and radar link is under {warn:.0f} %: the busiest, {b["tag"]}, '
                                      f'uses {b["load"] * 100:.0f} % of its {b["link"]} link.')
    # ---- the cameras' GMSL links (raw pixels whatever is recorded)
    cams = [r for r in rows if r['kind'] == 'camera' and r['load'] is not None]
    for r in cams:
        if r['link_severity'] == 'serious':
            _finding(out, 'serious', 'camera_link',
                     f'{r["tag"]} needs {r["link_mbps"]:,.0f} Mbit/s on its {r["link"]} link, over its '
                     f'{r["link_capacity_mbps"]:,.0f} Mbit/s.', [r])
        elif r['link_severity'] == 'warning':
            _finding(out, 'warning', 'camera_link',
                     f'{r["tag"]} uses {r["load"] * 100:.0f} % of its {r["link"]} link, above the {warn:.0f} % '
                     'margin.', [r])
    if cams and all(r['link_severity'] == 'good' for r in cams):
        b = max(cams, key=lambda r: r['load'])
        _finding(out, 'good', 'camera_link',
                 f'Every camera link is under {warn:.0f} %: the busiest, {b["tag"]}, carries {b["link_mbps"]:,.0f} of '
                 f'{b["link_capacity_mbps"]:,.0f} Mbit/s ({b["load"] * 100:.0f} %, {ecu.bpp} bit/px raw with '
                 f'{(ecu.overhead - 1) * 100:.0f} % overhead).')
    order = {s: i for i, s in enumerate(SEVERITIES)}
    out.sort(key=lambda f: order.get(f['severity'], 9))          # stable: the order above within a severity
    return out


# =========================================================================== the Issues list
def issues(project) -> list:
    """The budget's problems for the Issues list (cheap: no geometry): the recorder cannot write the data, a lidar's
    or radar's link over capacity or over the margin, the uplink over capacity or the margin, the disk set full before
    the shift ends. A camera's GMSL link is left to rules.link_budget."""
    res = budget(project)
    out = []
    for f in res.findings:
        if f['severity'] == 'good' or f['kind'] not in ISSUE_KINDS:
            continue
        out.append(R.Issue(f['severity'], f['message'], ', '.join(f['tags']) or f'{res.variant} · recording',
                           list(f['uids'])))
    return out


# =========================================================================== output
def _r(v, d=3):
    return None if v is None or not math.isfinite(float(v)) else round(float(v), d)


def summary(res: RecordingResult) -> dict:
    """Compact figures (the assistant, the batch report header)."""
    t, st = res.totals, res.settings
    return {'camera_format': st.camera_format, 'lidar_format': st.lidar_format,
            'data_rate_mbit_s': _r(t['mbps'], 1), 'data_rate_mb_s': _r(t['mb_s'], 1), 'gb_per_hour': _r(t['gb_h'], 1),
            'tb_per_shift': _r(t['tb_shift'], 2), 'shift_hours': st.shift_hours,
            'hours_per_disk_set': _r(t['hours_per_set'], 2), 'disk_set_tb': st.storage_tb,
            'disk_sets_per_shift': t['sets_per_shift'],
            'write_needed_mb_s': _r(t['write_needed_mb_s'], 1), 'write_available_mb_s': st.write_mb_s,
            'uplink_mbit_s': _r(t['uplink_mbps'], 1), 'uplink_load_pct': _r(t['uplink_load'] * 100, 1),
            'per_sensor': [{'tag': r['tag'], 'kind': r['kind'], 'mbit_s': _r(r['mbps'], 1),
                            'gb_per_hour': _r(r['gb_h'], 1), 'link': r['link'],
                            'link_load_pct': _r(r['load'] * 100, 1) if r['load'] is not None else None}
                           for r in res.rows],
            'findings': [{'severity': f['severity'], 'message': f['message']} for f in res.findings
                         if f['severity'] != 'good'],
            'assumptions': ASSUMPTIONS}


def _row_doc(r, total):
    return {'tag': r['tag'], 'kind': r['kind'], 'model': r['model'], 'format': r['format'], 'recorded': r['recorded'],
            'data_mbit_s': _r(r['mbps']), 'data_mb_s': _r(r['mb_s']), 'gb_per_hour': _r(r['gb_h']),
            'share_pct': _r(100 * r['mbps'] / total, 2) if total > 0 else 0.0,
            'link': r['link'], 'port': r['port'], 'link_mbit_s': _r(r['link_mbps']),
            'link_capacity_mbit_s': _r(r['link_capacity_mbps']),
            'link_load_pct': _r(r['load'] * 100, 2) if r['load'] is not None else None,
            'link_status': r['link_severity']}


def to_doc(res: RecordingResult, project=None) -> dict:
    """The JSON export: every sensor's data rate and link, the totals, the recorder and the findings."""
    t, st = res.totals, res.settings
    return {
        'schema': SCHEMA,
        'project': getattr(project, 'name', ''), 'variant': res.variant,
        'units': 'decimal: Mbit/s = 1e6 bit/s, MB = 1e6 bytes, GB = 1e9 bytes, TB = 1e12 bytes',
        'settings': dict(st.to_dict(), raw_bits_per_pixel=res.raw_bpp),
        'assumptions': ASSUMPTIONS,
        'sensors': [_row_doc(r, t['mbps']) for r in res.rows],
        'other': {'gnss_imu_can_mbit_s': _r(t['extra_mbps']), 'container_overhead_mbit_s': _r(t['overhead_mbps'])},
        'totals': {'data_mbit_s': _r(t['mbps']), 'data_mb_s': _r(t['mb_s']), 'gb_per_hour': _r(t['gb_h']),
                   'tb_per_shift': _r(t['tb_shift']), 'shift_hours': st.shift_hours,
                   'by_kind_mbit_s': {k: _r(v) for k, v in t['by_kind_mbps'].items()},
                   'hours_per_disk_set': _r(t['hours_per_set']), 'disk_set_tb': st.storage_tb,
                   'disk_sets_per_shift': t['sets_per_shift'],
                   'write_needed_mb_s': _r(t['write_needed_mb_s']), 'write_available_mb_s': st.write_mb_s,
                   'write_load_pct': _r(t['write_load'] * 100, 2),
                   'uplink_mbit_s': _r(t['uplink_mbps']), 'uplink_capacity_mbit_s': _r(t['uplink_capacity_mbps']),
                   'uplink_load_pct': _r(t['uplink_load'] * 100, 2)},
        'findings': [{'severity': f['severity'], 'kind': f['kind'], 'message': f['message'], 'sensors': f['tags']}
                     for f in res.findings],
    }


CSV_COLUMNS = ('tag', 'kind', 'model', 'format', 'recorded', 'data_mbit_s', 'data_mb_s', 'gb_per_hour', 'share_pct',
               'link', 'port', 'link_mbit_s', 'link_capacity_mbit_s', 'link_load_pct')


def to_csv(res: RecordingResult) -> str:
    """One row per sensor and a total row (the total includes GNSS/IMU/CAN and the container overhead; its link
    columns are the uplink from the switch to the recorder)."""
    t, st = res.totals, res.settings
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator='\n')
    w.writerow(CSV_COLUMNS)
    blank = lambda v: '' if v is None else v          # noqa: E731
    for r in res.rows:
        d = _row_doc(r, t['mbps'])
        w.writerow([blank(d[k]) for k in CSV_COLUMNS])
    w.writerow(['total', '', '', f'includes {st.extra_mbps:g} Mbit/s GNSS/IMU/CAN and '
                                 f'{st.container_overhead * 100:g} % container overhead', '',
                _r(t['mbps']), _r(t['mb_s']), _r(t['gb_h']), 100.0 if t['mbps'] > 0 else 0.0, 'uplink', '',
                _r(t['uplink_mbps']), _r(t['uplink_capacity_mbps']), _r(t['uplink_load'] * 100, 2)])
    return buf.getvalue()


def files(res: RecordingResult, project=None) -> list:
    """[(name, bytes)]: recording_budget.json and recording_budget.csv."""
    return [('recording_budget.json', (json.dumps(to_doc(res, project), indent=1) + '\n').encode('utf-8')),
            ('recording_budget.csv', to_csv(res).encode('utf-8'))]


def export(res: RecordingResult, project, out_dir) -> list[str]:
    """Write recording_budget.json and recording_budget.csv into out_dir; returns their paths."""
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for name, data in files(res, project):
        path = os.path.join(out_dir, name)
        with open(path, 'wb') as fh:
            fh.write(data)
        paths.append(path)
    return paths
