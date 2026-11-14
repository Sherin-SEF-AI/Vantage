"""Analysis ▸ Event cameras: neuromorphic sensors on the rig (Prophesee EVK4 HD with the IMX636, EVK5 HD with the
IMX646, the GenX320): each event camera's event rate and bandwidth over a rendered drive, its motion blur against the
RGB camera's smear, and a tunnel exit with a stopped car ahead, where the RGB camera's exposure catches up and the
event camera's dynamic range keeps the contrast (core.event_camera). A variant without an event camera is studied with
a virtual one beside its front RGB camera. A CARLA drive recorded with event cameras (events/index.json) loads beside
it: CARLA's DVS camera against the model. The study runs in the background (about 30 s on the sample's data rig)
when the page shows and again after edits; Cancel stops it."""
from __future__ import annotations

import json
import math
import os
import threading
import traceback

import numpy as np
from PyQt6.QtCore import QObject, QPointF, QRectF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QImage, QPainter, QPen
from PyQt6.QtWidgets import (QAbstractItemView, QFileDialog, QGridLayout, QHBoxLayout, QHeaderView, QProgressBar,
                             QScrollArea, QSplitter, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from ...core import event_camera as E
from ..charts import ChartBase, nice_ticks
from ..theme import STATUS, THEME, mono, sans
from ..widgets import ROW_H, ScrubSpinBox, SegmentedControl, StatusLine, button, fit_width, hline, label

# (group, key, label, unit, decimals, scrub step, shown / stored, tooltip)
FIELDS = (
    ('Drive', 'speed_kmh', 'Speed', 'km/h', 0, 5.0, 1.0, 'Driving speed of the event-rate, blur and tunnel studies'),
    ('Drive', 'object_m', 'Blur object', 'm', 1, 0.5, 1.0,
     'Distance of the static object the blur is measured on (as the sync plan\'s)'),
    ('Drive', 'duration_s', 'Drive', 's', 2, 0.05, 1.0, 'Length of the rendered drive the event rate is taken over'),
    ('Drive', 'render_scale', 'Render scale', '%', 0, 5.0, 100.0,
     'The rendered drive\'s resolution as a share of the sensor\'s (the rate is scaled by the pixel count)'),
    ('Drive', 'scene_white_cd', 'White', 'cd/m²', 0, 500.0, 1.0,
     'Luminance of a white surface in the rendered scene: the scene\'s brightness (daylight about 10 000)'),
    ('Sensor', 'threshold_pct', 'Threshold', '%', 1, 1.0, 1.0,
     'Contrast threshold used when the module states none (the IMX636\'s nominal is 25 %)'),
    ('Sensor', 'sigma_ln', 'Mismatch σ', '', 3, 0.005, 1.0,
     'Pixel-to-pixel threshold spread, ln units (v2e\'s default 0.03; no maker states it)'),
    ('Sensor', 'refractory_us', 'Refractory', 'µs', 0, 10.0, 1.0,
     'Dead time after an event when the module states none (makers: adjustable)'),
    ('Sensor', 'latency_us', 'Latency', 'µs', 0, 50.0, 1.0, 'Latency used when the module states none'),
    ('Sensor', 'shot_noise_hz', 'Shot noise', 'Hz', 2, 0.5, 1.0,
     'ON + OFF noise events per pixel per second in the dark (v2e\'s noisy preset: 5)'),
    ('Sensor', 'leak_hz', 'Leak', 'Hz', 2, 0.05, 1.0, 'Leak (ON) events per pixel per second (v2e: 0.1)'),
    ('Readout', 'erc_window_us', 'ERC window', 'µs', 0, 50.0, 1.0,
     'The event-rate controller drops evenly within windows this long that exceed the cap'),
    ('Readout', 'fifo_events', 'FIFO', 'ev', 0, 1024.0, 1.0, 'The FIFO readout\'s buffer, events'),
    ('Tunnel', 'tunnel_pre_s', 'Before portal', 's', 2, 0.1, 1.0, 'The tunnel exit starts this long before the portal'),
    ('Tunnel', 'tunnel_post_s', 'After portal', 's', 2, 0.1, 1.0, '… and ends this long after it'),
    ('Tunnel', 'tunnel_hz', 'Event frames', 'Hz', 0, 50.0, 1.0, 'Frames the event model runs on through the exit'),
    ('Tunnel', 'ae_tau_s', 'AE time const.', 's', 2, 0.05, 1.0,
     'The RGB auto exposure follows its metering with this time constant (assumed)'),
    ('Tunnel', 'target_m', 'Car past portal', 'm', 0, 5.0, 1.0, 'A stopped car this far past the portal in the lane'),
    ('Tunnel', 'target_albedo', 'Car albedo', '', 2, 0.01, 1.0, 'Its reflectance (a dark car about 0.05)'),
    ('Tunnel', 'min_contrast', 'Min contrast', '%', 0, 1.0, 100.0,
     'Weber contrast the RGB camera must keep between the car and the road to see it'),
    ('Tunnel', 'min_target_events', 'Min events', 'ev', 0, 5.0, 1.0,
     'Events on the car per window the event camera needs to see it'),
    ('Tunnel', 'lamp_cd', 'Luminaires', 'cd/m²', 0, 100.0, 1.0, 'The tunnel\'s ceiling lights (every 10 m)'),
)
COLS = (('Camera', 'The event camera (a virtual one is marked *)'),
        ('Module', 'The event sensor'),
        ('Mev/s', 'Mean event rate over the rendered drive, million events per second (full sensor)'),
        ('Peak', 'Busiest millisecond, Mev/s'),
        ('Mbit/s', 'Data rate in the chosen format'),
        ('Link', 'Share of the camera\'s link the data fills, %'),
        ('Dropped', 'Events the readout drops at its cap, %'),
        ('Edge', 'A moving edge\'s placement spread from its events (10–90 %), px'),
        ('Smear', 'The compared RGB camera\'s motion smear at its exposure, px'),
        ('Tunnel ev', 'Time the event camera misses the stopped car while it is in view, ms'),
        ('Tunnel RGB', 'Time the RGB camera\'s ISP output misses it (HDR signal in the tooltip), ms'))
VIEWS = (('tunnel', 'Tunnel exit'), ('rate', 'Event rate'))


def inputs_key(project) -> str:
    """Everything the study reads: a result made from other inputs is stale."""
    m = project.vehicle.mesh
    return json.dumps({'mesh': [id(m), getattr(m, 'n_tris', 0)], 'variant': project.variant.to_dict(),
                       'rig': [r.to_dict() for r in project.rig], 'ev': E.settings_of(project).to_dict(),
                       'isp': project.analysis.isp, 'timing': project.analysis.timing,
                       'library': project.section('library')}, sort_keys=True, default=str)


class _Sig(QObject):
    done = pyqtSignal(int, object)
    failed = pyqtSignal(int, str)
    progress = pyqtSignal(int, float, str)


def _num(text, tip='', color=None):
    it = QTableWidgetItem(text)
    it.setFont(mono(12))
    it.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    it.setFlags(it.flags() & ~Qt.ItemFlag.ItemIsEditable)
    if tip:
        it.setToolTip(tip)
    if color is not None:
        it.setForeground(color)
    return it


def _fmt(v, dp=1):
    return '—' if v is None or (isinstance(v, float) and not math.isfinite(v)) else f'{v:.{dp}f}'


def _qimage(rgb) -> QImage:
    rgb = np.ascontiguousarray(np.asarray(rgb, np.uint8))
    if rgb.ndim == 2:
        rgb = np.ascontiguousarray(np.repeat(rgb[..., None], 3, 2))
    h, w = rgb.shape[:2]
    return QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()


class Previews(QWidget):
    """The selected camera's pictures: the events over the rendered drive, the tunnel exit through the event camera
    and through the RGB camera's ISP output, and a moving edge's time surface."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.items = []                  # [(title, QImage)]
        self.setMinimumHeight(240)
        THEME.changed.connect(lambda *_: self.update())

    def set_camera(self, cam):
        self.items = []
        if cam is not None:
            r, tn = cam['rate'], cam['tunnel']
            if r.get('preview') is not None:
                self.items.append((f'{cam["tag"]} · 10 ms of events on the rendered drive', _qimage(r['preview'])))
            if tn.get('preview_event') is not None:
                self.items.append((f'Tunnel exit · {cam["tag"]} events at '
                                   f'{1000 * tn["event"].get("preview_t_s", 0):+.0f} ms', _qimage(tn['preview_event'])))
            if tn.get('preview_rgb') is not None:
                self.items.append((f'Tunnel exit · {tn["rgb"]["tag"]} ISP output', _qimage(tn['preview_rgb'])))
            ts = cam['blur']['edge'].get('time_surface')
            if ts is not None:
                g = (np.clip(ts, 0, 1) * 255).astype(np.uint8)
                self.items.append(('Time surface of a moving edge', _qimage(g)))
        self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        p.setFont(sans(11))
        p.setPen(THEME.color('muted'))
        if not self.items:
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, 'Run the study to see the events')
            return
        n = len(self.items)
        cols = 2 if n > 1 else 1
        rows = int(math.ceil(n / cols))
        cw, ch = self.width() / cols, self.height() / rows
        for k, (title, img) in enumerate(self.items):
            r, q = divmod(k, cols)
            cell = QRectF(q * cw + 3, r * ch + 3, cw - 6, ch - 6)
            p.setFont(sans(10))
            p.setPen(THEME.color('ink2'))
            p.drawText(QRectF(cell.left(), cell.top(), cell.width(), 14), Qt.AlignmentFlag.AlignLeft, title)
            s = min(cell.width() / img.width(), (cell.height() - 16) / img.height())
            tr = QRectF(cell.left() + (cell.width() - img.width() * s) / 2, cell.top() + 16, img.width() * s,
                        img.height() * s)
            p.drawImage(tr, img)


class TunnelChart(ChartBase):
    """The tunnel exit over time (t = 0: the event camera passes the portal): the RGB camera's contrast of the car
    against the road (solid: its ISP output; dashed: its merged HDR signal) and the event camera's events on the car
    per window (bars), each against its threshold."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.cam = None
        self.settings = E.EventSettings()
        self.title = 'Tunnel exit: the stopped car'
        self.setMinimumHeight(220)
        self._marks = []

    def set_data(self, cam, settings):
        self.cam, self.settings = cam, settings
        self.tip = None
        self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.paint_title(p)
        self._marks = []
        cam = self.cam
        if cam is None:
            return
        tn = cam['tunnel']
        ev, rg = tn['event'], tn.get('rgb')
        ts = ev['window_t_s']
        if not ts:
            return
        t0, t1 = ts[0], ts[-1] + ev['effective_window_ms'] / 1000.0
        W, H = self.width(), self.height()
        top_r = QRectF(48, 26, W - 60, (H - 60) * 0.5)
        bot_r = QRectF(48, top_r.bottom() + 18, W - 60, (H - 60) * 0.5 - 4)

        def X(t):
            return top_r.left() + top_r.width() * (t - t0) / max(t1 - t0, 1e-9)
        # RGB contrast
        self.paint_grid_y(p, top_r, [0, 0.25, 0.5, 0.75, 1.0], 1.0, fmt=lambda v: f'{100 * v:.0f}')
        p.setFont(mono(10))
        p.setPen(THEME.color('muted'))
        p.drawText(QRectF(0, top_r.top() - 16, 46, 14), Qt.AlignmentFlag.AlignRight, '%')
        if rg is not None:
            col = THEME.cam(int(cam['slot']) - 4 if cam['slot'] >= 4 else 0)
            for key, style in (('output_contrast', Qt.PenStyle.SolidLine), ('contrast', Qt.PenStyle.DashLine)):
                pts = [(r['t_s'], r[key]) for r in rg['rows'] if r[key] is not None]
                p.setPen(QPen(col, 2 if style == Qt.PenStyle.SolidLine else 1.3, style))
                for (ta, a), (tb, b) in zip(pts[:-1], pts[1:]):
                    p.drawLine(QPointF(X(ta), top_r.bottom() - top_r.height() * min(a, 1.0)),
                               QPointF(X(tb), top_r.bottom() - top_r.height() * min(b, 1.0)))
                for t, v in pts:
                    self._marks.append((QPointF(X(t), top_r.bottom() - top_r.height() * min(v, 1.0)), 'rgb', t, v, key))
            y = top_r.bottom() - top_r.height() * self.settings.min_contrast
            p.setPen(QPen(QColor(STATUS['critical']), 1, Qt.PenStyle.DashLine))
            p.drawLine(QPointF(top_r.left(), y), QPointF(top_r.right(), y))
            p.setFont(sans(10))
            p.setPen(THEME.color('ink2'))
            p.drawText(QRectF(top_r.left() + 4, top_r.top() - 2, top_r.width(), 14), Qt.AlignmentFlag.AlignLeft,
                       f'{rg["tag"]}: contrast of the car (solid: ISP output · dashed: {rg["dr_db"]:.0f} dB HDR signal)')
        # event bars
        cnt = ev['target_events']
        vmax = max([self.settings.min_target_events * 2] + list(cnt))
        ticks = nice_ticks(vmax, 3)
        vtop = ticks[-1] if ticks else vmax
        self.paint_grid_y(p, bot_r, ticks, vtop, fmt=lambda v: f'{v:,.0f}')
        bw = max(bot_r.width() / max(len(cnt), 1) - 1, 1)
        col = THEME.cam(int(cam['slot']))
        for t, c, seen, iv in zip(ts, cnt, ev['target_seen'], ev['target_in_view']):
            h = bot_r.height() * min(c / vtop, 1.0)
            r = QRectF(X(t), bot_r.bottom() - h, bw, h)
            p.fillRect(r, col if seen else THEME.color('muted', 0.6))
            self._marks.append((QPointF(X(t) + bw / 2, bot_r.bottom() - h), 'ev', t, c, iv))
        y = bot_r.bottom() - bot_r.height() * self.settings.min_target_events / vtop
        p.setPen(QPen(QColor(STATUS['critical']), 1, Qt.PenStyle.DashLine))
        p.drawLine(QPointF(bot_r.left(), y), QPointF(bot_r.right(), y))
        p.setFont(sans(10))
        p.setPen(THEME.color('ink2'))
        p.drawText(QRectF(bot_r.left() + 4, bot_r.top() - 14, bot_r.width(), 14), Qt.AlignmentFlag.AlignLeft,
                   f'{ev["tag"]}: events on the car per {ev["window_ms"]:g} ms')
        # the portal and the time axis
        p.setPen(QPen(THEME.color('strong'), 1, Qt.PenStyle.DotLine))
        p.drawLine(QPointF(X(0.0), top_r.top()), QPointF(X(0.0), bot_r.bottom()))
        p.setFont(mono(10))
        p.setPen(THEME.color('muted'))
        for t in np.arange(math.ceil(t0 * 2) / 2, t1 + 1e-9, 0.5):
            p.drawText(QRectF(X(t) - 20, bot_r.bottom() + 2, 40, 14), Qt.AlignmentFlag.AlignCenter, f'{t:+.1f}')
        p.drawText(QRectF(0, H - 14, W - 8, 14), Qt.AlignmentFlag.AlignRight, 's from the portal')
        self.paint_tip(p)

    def mouseMoveEvent(self, e):
        best, bd = None, 14.0
        for m in self._marks:
            d = math.hypot(m[0].x() - e.position().x(), m[0].y() - e.position().y())
            if d < bd:
                best, bd = m, d
        if best is None:
            self.tip = None
        elif best[1] == 'rgb':
            self.set_tip(best[0], f'{100 * best[3]:.0f} %', 'ISP output contrast' if best[4] == 'output_contrast'
                         else 'HDR signal contrast', f't = {best[2]:+.3f} s')
        else:
            self.set_tip(best[0], f'{best[3]:,.0f} events', f't = {best[2]:+.3f} s',
                         'the car in view' if best[4] else 'the car not in view')
        self.update()


class RateChart(ChartBase):
    """Events per second over the rendered drive (1 ms bins, the full sensor) against the readout's cap."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.cam = None
        self.title = 'Event rate over the drive'
        self.setMinimumHeight(220)

    def set_data(self, cam):
        self.cam = cam
        self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        cam = self.cam
        self.subtitle = '' if cam is None else f'{cam["tag"]} · {cam["rate"]["frames"]} rendered frames'
        self.paint_title(p)
        if cam is None or not cam['rate']['series_evs']:
            return
        r = cam['rate']
        ys = np.asarray(r['series_evs']) / 1e6
        xs = np.asarray(r['series_t_ms'])
        cap = (r['cap_evs'] or 0) / 1e6
        vmax = max(float(ys.max()), cap * 1.05 if cap else 0.0, 1e-6)
        ticks = nice_ticks(vmax, 4)
        top = ticks[-1] if ticks else vmax
        R = QRectF(48, 26, self.width() - 60, self.height() - 26 - 30)
        self.paint_grid_y(p, R, ticks, top, fmt=lambda v: f'{v:g}')
        p.setFont(mono(10))
        p.setPen(THEME.color('muted'))
        p.drawText(QRectF(0, R.top() - 16, 46, 14), Qt.AlignmentFlag.AlignRight, 'Mev/s')
        x1 = max(float(xs[-1]) + 1.0, 1.0)

        def X(t):
            return R.left() + R.width() * t / x1
        bw = max(R.width() / len(ys) - 0.5, 0.8)
        col = THEME.cam(int(cam['slot']))
        for t, v in zip(xs, ys):
            h = R.height() * min(v / top, 1.0)
            p.fillRect(QRectF(X(t), R.bottom() - h, bw, h), col)
        if cap:
            y = R.bottom() - R.height() * min(cap / top, 1.0)
            p.setPen(QPen(QColor(STATUS['critical']), 1, Qt.PenStyle.DashLine))
            p.drawLine(QPointF(R.left(), y), QPointF(R.right(), y))
            p.setFont(sans(10))
            p.drawText(QRectF(R.left(), y - 14, R.width() - 4, 13), Qt.AlignmentFlag.AlignRight,
                       f'cap {cap:.0f} Mev/s ({r["cap_source"]})')
        p.setFont(mono(10))
        p.setPen(THEME.color('muted'))
        for t in nice_ticks(x1, 4):
            p.drawText(QRectF(X(t) - 20, R.bottom() + 2, 40, 14), Qt.AlignmentFlag.AlignCenter, f'{t:g}')
        p.drawText(QRectF(0, self.height() - 14, self.width() - 8, 14), Qt.AlignmentFlag.AlignRight, 'ms')


# =========================================================================== the page
class EventPage(QWidget):
    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        self.res: E.EventResult | None = None
        self.busy = False
        self.stale = False
        self.status_text = ''
        self.sel_uid = ''
        self.drive = None                # a loaded CARLA drive: {'path', 'index', 'rows'}
        self._gen = 0
        self._cancel = threading.Event()
        self._for = None
        self._syncing = False
        self.sig = _Sig()
        self.sig.done.connect(self._done)
        self.sig.failed.connect(self._failed)
        self.sig.progress.connect(self._progress)
        self.setObjectName('base')
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        split = QSplitter(Qt.Orientation.Horizontal)
        split.setHandleWidth(2)
        split.addWidget(self._left())
        cw = QWidget()
        cw.setObjectName('panel')
        cl = QVBoxLayout(cw)
        cl.setContentsMargins(8, 6, 8, 6)
        cl.setSpacing(6)
        self.head = label('', ink2=True, wrap=True)
        cl.addWidget(self.head)
        self.previews = Previews()
        cl.addWidget(self.previews, 1)
        split.addWidget(cw)
        right = QSplitter(Qt.Orientation.Vertical)
        right.setHandleWidth(2)
        sw = QWidget()
        sw.setObjectName('panel')
        sl = QVBoxLayout(sw)
        sl.setContentsMargins(8, 6, 8, 6)
        sl.setSpacing(4)
        row = QHBoxLayout()
        self.view = SegmentedControl(VIEWS, 'tunnel')
        self.view.setToolTip('The chart: the tunnel exit over time, or the event rate over the rendered drive')
        self.view.changed.connect(lambda k: self.refresh())
        row.addWidget(self.view)
        row.addStretch(1)
        sl.addLayout(row)
        self.tunnel_chart = TunnelChart()
        self.rate_chart = RateChart()
        sl.addWidget(self.tunnel_chart, 1)
        sl.addWidget(self.rate_chart, 1)
        self.chart_note = label('', small=True, wrap=True, ink2=True)
        sl.addWidget(self.chart_note)
        right.addWidget(sw)
        tw = QWidget()
        tw.setObjectName('panel')
        tl = QVBoxLayout(tw)
        tl.setContentsMargins(8, 6, 8, 6)
        tl.setSpacing(4)
        tl.addWidget(label('Event cameras', bold=True))
        t = self.table = QTableWidget(0, len(COLS))
        t.verticalHeader().setVisible(False)
        t.verticalHeader().setDefaultSectionSize(ROW_H)
        t.horizontalHeader().setFixedHeight(ROW_H)
        t.setShowGrid(False)
        t.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        t.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        for c, (head, tip) in enumerate(COLS):
            hi = QTableWidgetItem(head)
            hi.setToolTip(tip)
            t.setHorizontalHeaderItem(c, hi)
        hh = t.horizontalHeader()
        hh.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        for c in [0] + list(range(2, len(COLS))):
            hh.setSectionResizeMode(c, QHeaderView.ResizeMode.ResizeToContents)
        t.itemSelectionChanged.connect(self._row_picked)
        tl.addWidget(t, 1)
        right.addWidget(tw)
        fw = QWidget()
        fw.setObjectName('panel')
        fl = QVBoxLayout(fw)
        fl.setContentsMargins(10, 6, 10, 8)
        fl.setSpacing(4)
        fl.addWidget(label('Findings', bold=True))
        self.find_host = QWidget()
        self.find_lay = QVBoxLayout(self.find_host)
        self.find_lay.setContentsMargins(0, 0, 0, 0)
        self.find_lay.setSpacing(4)
        self.find_lay.addStretch(1)
        sc = QScrollArea()
        sc.setWidgetResizable(True)
        sc.setWidget(self.find_host)
        fl.addWidget(sc, 1)
        self.findings: list[StatusLine] = []
        right.addWidget(fw)
        right.setSizes([320, 200, 280])
        split.addWidget(right)
        split.setSizes([300, 560, 700])
        lay.addWidget(split)
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(500)
        self.timer.timeout.connect(self.run)
        ctx.store.changed.connect(self._on_edit)
        ctx.store.project_replaced.connect(self._reset)
        self._sync_fields()
        self.refresh()

    # ------------------------------------------------------------------ settings
    def _left(self):
        w = QWidget()
        w.setObjectName('panel')
        w.setMinimumWidth(280)
        outer = QVBoxLayout(w)
        outer.setContentsMargins(0, 0, 0, 0)
        sc = QScrollArea()
        sc.setWidgetResizable(True)
        host = QWidget()
        ll = QVBoxLayout(host)
        ll.setContentsMargins(8, 6, 8, 8)
        ll.setSpacing(4)
        top = QHBoxLayout()
        top.addWidget(label('Event cameras'))
        top.addStretch(1)
        top.addWidget(label('rate, blur, tunnel exit', muted=True, small=True))
        ll.addLayout(top)
        ll.addWidget(hline())
        ll.addWidget(label('Neuromorphic sensors record per-pixel brightness changes as events instead of frames. '
                           'How many events the rig\'s event cameras send and whether their links keep up, how '
                           'sharply they place a moving edge where the RGB cameras smear it, and a tunnel exit with a '
                           'stopped car ahead. The sensors\' figures are their makers\'; the event model is ESIM with '
                           'v2e\'s noise.', muted=True, small=True, wrap=True))
        ll.addSpacing(4)
        grid = QGridLayout()
        grid.setVerticalSpacing(2)
        grid.setHorizontalSpacing(6)
        grid.setColumnStretch(1, 1)

        def row(text, wdg, tip):
            a = label(text, ink2=True)
            a.setFixedWidth(104)
            a.setToolTip(tip)
            wdg.setToolTip(tip)
            i = grid.rowCount()
            grid.addWidget(a, i, 0)
            grid.addWidget(wdg, i, 1)
        self.fmt = SegmentedControl([(f, E.FORMAT_LABEL[f]) for f in E.FORMATS], 'EVT3.0')
        self.fmt.changed.connect(lambda k: self.set_option('event_format', k))
        row('Format', self.fmt, 'The event format the data rate is given in (Prophesee EVT 3.0, 2.1 or 2.0)')
        self.bwmode = SegmentedControl([('erc', 'ERC'), ('fifo', 'FIFO')], 'erc')
        self.bwmode.changed.connect(lambda k: self.set_option('bandwidth_mode', k))
        row('Readout', self.bwmode, 'At the cap: the event-rate controller drops evenly (ERC), or a FIFO delays and '
                                    'drops what overflows it')
        self.lowpass = SegmentedControl([('on', 'On'), ('off', 'Ideal')], 'on')
        self.lowpass.changed.connect(lambda k: self.set_option('photoreceptor', k == 'on'))
        row('Latency', self.lowpass, 'On: each pixel responds with its stated latency at its illuminance; Ideal: '
                                     'instantly')
        ll.addLayout(grid)
        self.fields = {}
        group = None
        g2 = None
        for grp, key, text, unit, dec, step, mul, tip in FIELDS:
            if grp != group:
                if g2 is not None:
                    ll.addLayout(g2)
                ll.addSpacing(4)
                ll.addWidget(label(grp, bold=True))
                g2 = QGridLayout()
                g2.setVerticalSpacing(2)
                g2.setHorizontalSpacing(6)
                g2.setColumnStretch(2, 1)
                group = grp
            a = label(text, ink2=True)
            a.setFixedWidth(104)
            a.setToolTip(tip)
            f = ScrubSpinBox('', 'plain', dec, unit, step=step, width=110)
            f.setToolTip(tip)
            f.committed.connect(lambda v, k=key, m=mul: self.set_option(k, v / m))
            i = g2.rowCount()
            g2.addWidget(a, i, 0)
            g2.addWidget(f, i, 1)
            self.fields[key] = (f, mul)
        if g2 is not None:
            ll.addLayout(g2)
        ll.addSpacing(6)
        ll.addWidget(label('CARLA drive', bold=True))
        self.drive_text = label('No drive loaded. A CARLA drive recorded with event cameras (Simulate ▸ CARLA) '
                                'writes events/index.json: load its folder to compare CARLA\'s DVS camera with the '
                                'model and to read its event rate.', muted=True, small=True, wrap=True)
        ll.addWidget(self.drive_text)
        ll.addStretch(1)
        sc.setWidget(host)
        fit_width(sc)
        outer.addWidget(sc, 1)
        bot = QVBoxLayout()
        bot.setContentsMargins(8, 4, 8, 8)
        bot.setSpacing(4)
        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.setTextVisible(False)
        self.bar.setFixedHeight(4)
        bot.addWidget(self.bar)
        self.status = label('', muted=True, small=True, wrap=True)
        bot.addWidget(self.status)
        bl = QHBoxLayout()
        self.b_run = button('Run', 'primary', 'play', 'Run the event camera study now')
        self.b_run.clicked.connect(lambda: self.run())
        self.b_cancel = button('Cancel', None, 'stop', 'Stop the running study')
        self.b_cancel.clicked.connect(self.cancel)
        self.b_load = button('Drive…', None, 'folder', 'Load a CARLA drive recorded with event cameras')
        self.b_load.clicked.connect(lambda: self.load_drive())
        self.b_export = button('Export…', None, 'export', 'The study (JSON), the cameras and the tunnel exit (CSV), '
                                                          'the event previews (PNG)')
        self.b_export.clicked.connect(lambda: self.export())
        bl.addWidget(self.b_run)
        bl.addWidget(self.b_cancel)
        bl.addStretch(1)
        bot.addLayout(bl)
        bl2 = QHBoxLayout()                          # a second row: four buttons did not fit a narrow panel
        bl2.addWidget(self.b_load)
        bl2.addWidget(self.b_export)
        bl2.addStretch(1)
        bot.addLayout(bl2)
        outer.addLayout(bot)
        return w

    @property
    def settings(self) -> E.EventSettings:
        return E.settings_of(self.ctx.store.project)

    def set_option(self, key, value) -> bool:
        """Change one setting, saved with the project as one undo step; False when nothing changes."""
        if self._syncing:
            return False
        st = self.settings
        old = st.to_dict()
        setattr(st, key, value)
        new = E.EventSettings.from_dict(st.to_dict()).to_dict()
        if new == old:
            self._sync_fields()
            return False
        shown = new[key]
        name = next((f[2].lower() for f in FIELDS if f[1] == key), key.replace('_', ' '))
        text = ('on' if shown else 'ideal') if key == 'photoreceptor' else (
            f'{shown:g}' if isinstance(shown, (int, float)) else str(shown))

        def fn(p):
            p.analysis.event_camera = new
        self.ctx.store.edit('analysis', fn, f'Event cameras · {name} {text}')
        return True

    def _sync_fields(self):
        st = self.settings
        self._syncing = True
        try:
            for key, (f, mul) in self.fields.items():
                f.set_value(float(getattr(st, key)) * mul)
            self.fmt.set_value(st.event_format)
            self.bwmode.set_value(st.bandwidth_mode)
            self.lowpass.set_value('on' if st.photoreceptor else 'off')
        finally:
            self._syncing = False

    # ------------------------------------------------------------------ the job
    def _on_edit(self, *_):
        self._sync_fields()
        if self.res is not None and self._for is not None:
            self.stale = inputs_key(self.ctx.store.project) != self._for[1]
        if self.stale and self.isVisible():
            self.timer.start()
        self.refresh()

    def _reset(self, *_):
        self.cancel()
        self.res, self._for, self.stale, self.sel_uid, self.drive = None, None, False, '', None
        self._sync_fields()
        self.refresh()
        if self.isVisible():
            self.timer.start()

    def showEvent(self, e):
        super().showEvent(e)
        if (self.res is None or self.stale) and not self.busy:
            self.timer.start()

    def activate(self):
        if (self.res is None or self.stale) and not self.busy:
            self.run()

    def _engine(self):
        win = self.ctx.window
        if win is not None and hasattr(win, '_shared_engine'):
            return win._shared_engine()
        return self.ctx.analysis.engine.share()

    def run(self) -> bool:
        """Start the study on a snapshot of the project (a running one is cancelled)."""
        self.timer.stop()
        p = self.ctx.store.project
        project = p.snapshot()
        st = E.settings_of(project)
        eng = self._engine()
        self._cancel.set()
        self._cancel = threading.Event()
        cancel = self._cancel
        self._gen += 1
        gen = self._gen
        key = (p, inputs_key(p))
        self.busy = True
        self.status_text = 'Rendering the drive and running the event model…'
        self.refresh()

        def work():
            try:
                res = E.run(project, eng, st, progress=lambda f, t='': self.sig.progress.emit(gen, f, t),
                            cancel=cancel.is_set)
                self.sig.done.emit(gen, (key, res))
            except RuntimeError:                  # the page went while computing
                pass
            except Exception as e:
                try:
                    self.sig.failed.emit(gen, f'{type(e).__name__}: {e}\n{traceback.format_exc()}')
                except RuntimeError:
                    pass
        threading.Thread(target=work, daemon=True).start()
        return True

    def cancel(self):
        self._cancel.set()
        if self.busy:
            self._gen += 1
            self.busy = False
            self.status_text = 'Cancelled'
            self.refresh()

    def _progress(self, gen, f, text):
        if gen == self._gen:
            self.bar.setValue(int(f * 1000))
            if text:
                self.status.setText(f'{100 * f:.0f} % · {text}')

    def _done(self, gen, out):
        if gen != self._gen:
            return
        self.busy = False
        key, res = out
        if res is None or key[0] is not self.ctx.store.project:
            self.status_text = 'Cancelled' if res is None else ''
            self.refresh()
            return
        self.res, self._for = res, key
        self.stale = key[1] != inputs_key(self.ctx.store.project)
        self.status_text = (f'{len(res.cams)} event camera{"s" if len(res.cams) != 1 else ""}'
                            + (' (virtual)' if res.metrics.get('virtual') else '')
                            + f' · {res.metrics["seconds"]:.1f} s')
        if not self.sel_uid or res.cam(self.sel_uid) is None:
            self.sel_uid = res.cams[0]['uid'] if res.cams else ''
        self.refresh()
        if self.stale and self.isVisible():
            self.timer.start()

    def _failed(self, gen, msg):
        if gen != self._gen:
            return
        self.busy = False
        self.status_text = 'Failed: ' + msg.splitlines()[0]
        self.refresh()

    # ------------------------------------------------------------------ a CARLA drive
    def load_drive(self, folder=None) -> bool:
        if folder is None:
            folder = QFileDialog.getExistingDirectory(self, 'A CARLA drive recorded with event cameras')
            if not folder:
                return False
        idx = E.load_recording(folder)
        if idx is None:
            self.ctx.msg(f'{folder} holds no events/index.json (a CARLA drive recorded with event cameras).', 6000)
            return False
        self.drive = {'path': folder, 'index': idx, 'rows': E.recording_summary(idx)}
        self.ctx.msg(f'Loaded the event cameras of {os.path.basename(folder.rstrip(os.sep))}', 4000)
        self.refresh()
        return True

    def _drive_text(self) -> str:
        d = self.drive
        if d is None:
            return ''
        out = [f'{os.path.basename(d["path"].rstrip(os.sep))}:']
        for r in d['rows']:
            parts = [f'{r["tag"]} {r["width"]} × {r["height"]}, {r["duration_s"]:g} s']
            if r.get('rate_evs') is not None:
                parts.append(f'{r["rate_evs"] / 1e6:.2f} Mev/s (peak {r["peak_1ms_evs"] / 1e6:.2f}), '
                             f'{r["activity_hz"]:.1f} events per pixel per second')
            c = r.get('compare')
            if c:
                parts.append(f'CARLA\'s DVS against the model: {c["carla_events"]:,} and {c["model_events"]:,} '
                             f'events, {100 * (c["exact_match_share"] or 0):.1f} % matched exactly, per-pixel '
                             f'r = {_fmt(c["pixel_count_r"], 3)}')
            cl = r.get('compare_linear')
            if cl:
                parts.append(f'in linear light the model makes {_fmt(cl["count_ratio"], 2)}× the events')
            out.append('; '.join(parts) + '.')
        return ' '.join(out)

    # ------------------------------------------------------------------ the views
    def select(self, uid):
        self.sel_uid = uid
        self.refresh()

    def _row_picked(self):
        if self._syncing:
            return
        rows = self.table.selectionModel().selectedRows() if self.table.selectionModel() else []
        if rows and self.res is not None and rows[0].row() < len(self.res.cams):
            uid = self.res.cams[rows[0].row()]['uid']
            if uid != self.sel_uid:
                self.select(uid)

    def refresh(self):
        res = self.res
        self.b_cancel.setEnabled(self.busy)
        self.b_export.setEnabled(res is not None)
        self.bar.setVisible(self.busy)
        self.status.setText(self.status_text + (' · out of date: re-running' if self.stale and res is not None and
                                                not self.busy and self.status_text and not
                                                self.status_text.startswith('Cancelled') else ''))
        if self.drive is not None:
            self.drive_text.setText(self._drive_text())
        cam = res.cam(self.sel_uid) if res is not None else None
        view = self.view.value() or 'tunnel'
        self.tunnel_chart.setVisible(view == 'tunnel')
        self.rate_chart.setVisible(view == 'rate')
        self.tunnel_chart.set_data(cam, res.settings if res is not None else self.settings)
        self.rate_chart.set_data(cam)
        self.previews.set_camera(cam)
        if res is None or cam is None:
            self.head.setText('No result yet.' if res is None else 'No event camera.')
            self.chart_note.setText('')
            self._fill_table()
            self._fill_findings()
            return
        ep = cam['params']
        self.head.setText(f'{res.variant} · {cam["tag"]}{" (virtual)" if cam["virtual"] else ""}: {cam["module"]}, '
                          f'{ep["width"]} × {ep["height"]}, threshold {ep["threshold_pct"]:g} % '
                          f'({ep["threshold_source"]}), latency {ep["latency_source"]}, dynamic range '
                          f'{ep["dr_db"]:g} dB')
        tn = cam['tunnel']
        rg, ev = tn.get('rgb'), tn['event']
        if view == 'tunnel':
            sc = tn['scene']
            txt = (f'Inside {sc["interior_cd"]:g} cd/m², the road outside {sc["outside_road_cd"]:.0f} cd/m² and the '
                   f'car {sc["target_cd"]:.0f} cd/m² ({sc["scene_db"]:.0f} dB), {sc["speed_kmh"]:g} km/h. ')
            if rg is not None:
                txt += (f'{rg["tag"]}\'s ISP output misses the car for {1000 * rg["output_blind_s"]:.0f} ms, its HDR '
                        f'signal for {1000 * rg["blind_s"]:.0f} ms; ')
            txt += (f'{ev["tag"]} for {1000 * ev["blind_s"]:.0f} ms (judged over {ev["effective_window_ms"]:.0f} ms at '
                    f'the rendered resolution; its latency on the car {ev["latency_target_us"]:.0f} µs, inside '
                    f'{ev["latency_interior_us"]:.0f} µs).')
        else:
            r = cam['rate']
            txt = (f'{r["rate_evs"] / 1e6:.1f} Mev/s on average ({r["activity_hz"]:.1f} per pixel per second), '
                   f'{r["mbps"][r["format"]]:.0f} Mbit/s of {E.FORMAT_LABEL[r["format"]]} '
                   f'({r["bits_per_event"][r["format"]]:.1f} bit/event at the rendered resolution, '
                   f'{r["bits_per_event_dense"][r["format"]]:.1f} at the sensor\'s event density); EVT 2.0 '
                   f'{r["mbps"]["EVT2.0"]:.0f}, EVT 2.1 {r["mbps"]["EVT2.1"]:.0f} Mbit/s.')
        self.chart_note.setText(txt)
        self._fill_table()
        self._fill_findings()

    def _fill_table(self):
        res = self.res
        t = self.table
        self._syncing = True
        try:
            if res is None:
                t.setRowCount(0)
                return
            t.setRowCount(len(res.cams))
            for r, c in enumerate(res.cams):
                it = QTableWidgetItem(c['tag'] + (' *' if c['virtual'] else ''))
                it.setForeground(THEME.cam(int(c['slot'])))
                it.setFlags(it.flags() & ~Qt.ItemFlag.ItemIsEditable)
                t.setItem(r, 0, it)
                mi = QTableWidgetItem(c['module'])
                mi.setFlags(mi.flags() & ~Qt.ItemFlag.ItemIsEditable)
                t.setItem(r, 1, mi)
                ra, b, tn = c['rate'], c['blur'], c['tunnel']
                f = ra['format']
                t.setItem(r, 2, _num(_fmt(ra['rate_evs'] / 1e6, 1)))
                t.setItem(r, 3, _num(_fmt(ra['peak_1ms_evs'] / 1e6, 1)))
                t.setItem(r, 4, _num(_fmt(ra['mbps'][f], 0)))
                ld = ra['link_load']
                t.setItem(r, 5, _num('—' if ld is None else f'{100 * ld:.0f}',
                                     color=QColor(STATUS['serious'] if ld and ld > 1 else STATUS['warning'])
                                     if ld and ld > res.settings.link_warn else None))
                dr = ra['bandwidth']['dropped_share']
                t.setItem(r, 6, _num(f'{100 * dr:.1f}', color=QColor(STATUS['warning']) if dr > 0.001 else None))
                t.setItem(r, 7, _num(_fmt(b['edge'].get('spread_px'), 2)))
                t.setItem(r, 8, _num(_fmt(b.get('rgb_smear_px'), 1), b.get('rgb_tag') or ''))
                t.setItem(r, 9, _num(_fmt(1000 * tn['event']['blind_s'], 0)))
                rg = tn.get('rgb')
                t.setItem(r, 10, _num('—' if rg is None else _fmt(1000 * rg['output_blind_s'], 0),
                                      '' if rg is None else f'HDR signal: {1000 * rg["blind_s"]:.0f} ms'))
            sel = next((i for i, c in enumerate(res.cams) if c['uid'] == self.sel_uid), -1)
            if sel >= 0:
                t.selectRow(sel)
        finally:
            self._syncing = False

    def _fill_findings(self):
        for w in self.findings:
            w.hide()
            w.deleteLater()
        self.findings = []
        if self.res is None:
            return
        for f in self.res.findings:
            ln = StatusLine(f['severity'] if f['severity'] in STATUS else '', f['message'])
            ln.text.setProperty('small', True)
            self.find_lay.insertWidget(self.find_lay.count() - 1, ln)
            self.findings.append(ln)

    # ------------------------------------------------------------------ export
    def export(self, folder=None):
        if self.res is None:
            return []
        if folder is None:
            folder = QFileDialog.getExistingDirectory(self, 'Export the event camera study into')
            if not folder:
                return []
        from ...core import provenance as PV
        paths = PV.stamp_paths(E.export(self.res, self.ctx.store.project, folder), self.ctx.store.project, folder)
        self.ctx.msg(f'Exported the event camera study to {folder}', 5000)
        return paths
