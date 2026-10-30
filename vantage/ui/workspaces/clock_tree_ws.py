"""Analysis ▸ Clock tree and time sync: the rig's clock distribution from GNSS through the grandmaster, the switch and
the deserializers to every sensor; a fault scenario (PPS dropout, grandmaster loss, link asymmetry, missed or late
FSYNC pulses) run as a seeded Monte Carlo; every sensor's timestamp error against true time and the lidar; and the
projection error it makes in each camera's image. Runs in the background after every edit while the page shows."""
from __future__ import annotations

import math
import threading
import traceback

import numpy as np
from PyQt6.QtCore import QObject, QPointF, QRectF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QImage, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import (QAbstractItemView, QComboBox, QFileDialog, QGridLayout, QHBoxLayout, QHeaderView, QMenu,
                             QProgressBar, QScrollArea, QSplitter, QTableWidget, QTableWidgetItem, QTabWidget,
                             QVBoxLayout, QWidget)

from ...core import clock_tree as CT
from ...core.mathutil import parse_number
from .. import colormaps
from ..charts import ChartBase
from ..theme import STATUS, THEME, mono, sans
from ..widgets import ROW_H, ScrubSpinBox, SegmentedControl, StatusLine, ToggleSwitch, button, hline, label

# key, label, kind, decimals, unit, scrub step, tooltip
RUN_FIELDS = (('runs', 'Runs', 'plain', 0, '', 10.0, 'Monte Carlo runs over the timeline'),
              ('seed', 'Seed', 'plain', 0, '', 1.0, 'Monte Carlo seed: the same seed gives the same numbers'),
              ('duration_s', 'Duration', 'plain', 0, 's', 10.0, 'Length of the scenario timeline'))
MOTION_FIELDS = (('speed_kmh', 'Speed', 'plain', 0, 'km/h', 5.0, 'Vehicle speed for the projection error'),
                 ('yaw_rate_dps', 'Yaw rate', 'plain', 1, '°/s', 1.0, 'Yaw rate for the projection error (a turn; '
                                                                     'positive to the left)'),
                 ('range_m', 'Range', 'plain', 1, 'm', 1.0, 'Points this far from the camera along each pixel\'s ray'))
LIMIT_FIELDS = (('threshold_px', 'Max error', 'plain', 2, 'px', 0.1, 'The most projection error the dataset accepts '
                                                                    '(worst over the image, 95 % of frames)'),
                ('time_budget_us', 'Time budget', 'plain', 0, 'µs', 10.0, 'How far from UTC the rig\'s time may drift'),
                ('holdover_need_s', 'Ride through', 'plain', 0, 's', 10.0, 'The longest PPS outage the rig must ride '
                                                                          'through within the time budget'),
                ('bmca_s', 'Announce t/o', 'plain', 1, 's', 0.5, 'Announce timeout before a backup grandmaster takes '
                                                                 'over (802.1AS: 3 announce intervals of 1 s)'))
NODE_COLS = (('Node', 'The clock tree: GNSS, grandmaster, switch, compute, deserializers and sensors'),
             ('Stamps', 'Where the node\'s timestamps or time come from'),
             ('p95 true', 'Timestamp (or clock) error against true time, 95 % of frames'),
             ('p95 lidar', 'Timestamp error against the reference lidar, 95 % of frames'),
             ('Max', 'Worst timestamp error against the lidar (true time for the lidar and the clocks)'))
PARAM_COLS = (('Parameter', ''), ('Value', 'Double-click to edit; your edits are kept with the project'),
              ('Typical', 'The typical value and where it comes from'))
EVENT_COLS = (('Event', ''), ('At s', 'When it starts on the timeline'),
              ('For / size', 'Duration (s) of a dropout or loss (0 = for good), or the size (ns, ms)'),
              ('Where', 'The node it hits'))
PROJ_COLS = (('Camera', ''), ('Δt p95', 'Camera stamp minus lidar stamp, ms, 95 % of frames'),
             ('Δt max', 'The worst frame, ms'), ('Centre p95', 'Error at the image centre, px, for Δt p95'),
             ('Worst p95', 'Worst error over the image, px, for Δt p95'),
             ('Worst max', 'Worst error over the image, px, for the worst frame'),
             ('Plan', 'The sync plan\'s own misalignment (ms) and the error it makes if the pipeline does not '
                      'compensate it, px'))
ERR_RAMP = ['#184f95', '#3987e5', '#86b6ef', '#fab219', '#ec835a', '#d03b3b']


class _Sig(QObject):
    done = pyqtSignal(int, object)
    failed = pyqtSignal(int, str)
    progress = pyqtSignal(int, float)


def _table(cols, stretch=0):
    t = QTableWidget(0, len(cols))
    t.verticalHeader().setVisible(False)
    t.verticalHeader().setDefaultSectionSize(ROW_H)
    t.horizontalHeader().setFixedHeight(ROW_H)
    t.setShowGrid(False)
    t.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
    t.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    t.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    for c, (head, tip) in enumerate(cols):
        it = QTableWidgetItem(head)
        if tip:
            it.setToolTip(tip)
        t.setHorizontalHeaderItem(c, it)
    hh = t.horizontalHeader()
    for c in range(len(cols)):
        hh.setSectionResizeMode(c, QHeaderView.ResizeMode.Stretch if c == stretch else
                                QHeaderView.ResizeMode.ResizeToContents)
    return t


def _item(text, tip='', color=None, num=False, editable=False, data=None):
    it = QTableWidgetItem(text)
    if num:
        it.setFont(mono(12))
        it.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    if tip:
        it.setToolTip(tip)
    if color is not None:
        it.setForeground(color)
    if editable:
        it.setFlags(it.flags() | Qt.ItemFlag.ItemIsEditable)
    else:
        it.setFlags(it.flags() & ~Qt.ItemFlag.ItemIsEditable)
    if data is not None:
        it.setData(Qt.ItemDataRole.UserRole, data)
    return it


def _px(v, dp=2):
    return '—' if v is None or not math.isfinite(v) else f'{v:.{dp}f}'


def _value_text(key, v) -> str:
    if key in CT.CHOICES:
        return dict(CT.CHOICES[key]).get(v, str(v))
    unit = CT.PARAMS[key][1]
    dp = CT.PARAMS[key][4]
    return f'{v:.{dp}f}'.rstrip('0').rstrip('.') + (f' {unit}' if unit else '') if isinstance(v, float) else str(v)


def _tone(px, limit):
    if px is None:
        return None
    return 'good' if px <= limit else 'warning' if px <= 3 * limit else 'serious'


# =========================================================================== the tree
class TopologyView(QWidget):
    """The clock tree drawn top down: GNSS at the top, the grandmaster, the switch and the compute, the endpoints and
    the deserializers with their cameras at the bottom. Each box says what the node is and its p95 timestamp error
    against the lidar (true time for clocks); the grandmaster has a ring. Click a node to edit it."""
    selected = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setMinimumHeight(220)
        self.topo = None
        self.res = None
        self.sel = ''
        self.hover = ''
        self.boxes = {}
        THEME.changed.connect(self._retheme)

    def _retheme(self, *_):
        self.update()

    def set_tree(self, topo, res=None):
        self.topo, self.res = topo, res
        self.update()

    def select(self, nid):
        self.sel = nid or ''
        self.update()

    def _layout(self):
        topo = self.topo
        kids = {}
        for n in topo.nodes:
            kids.setdefault(n.parent, []).append(n)
        depth, order = {}, []

        def walk(n, d):
            depth[n.id] = d
            ch = kids.get(n.id, [])
            if not ch:
                order.append(n.id)
            for c in ch:
                walk(c, d + 1)
        for r in kids.get('', []):
            walk(r, 0)
        nl = max(len(order), 1)
        levels = max(depth.values()) + 1 if depth else 1
        W, H = self.width(), self.height()
        bw = min(118.0, (W - 16) / nl - 6)
        bh = 34.0
        gap_y = max((H - 16 - bh) / max(levels - 1, 1), bh + 8)
        x = {}
        for i, nid in enumerate(order):
            x[nid] = 8 + (i + 0.5) * (W - 16) / nl

        def place(n):
            ch = kids.get(n.id, [])
            if ch:
                xs = [place(c) for c in ch]
                x[n.id] = (min(xs) + max(xs)) / 2
            return x[n.id]
        for r in kids.get('', []):
            place(r)
        self.boxes = {nid: QRectF(x[nid] - bw / 2, 8 + depth[nid] * gap_y, bw, bh) for nid in x}

    def _figure(self, n):
        res = self.res
        if res is None:
            return ''
        s = res.sensor(n.id)
        if s is not None:
            st = s['vs_lidar'] or s['true']
            return CT.fmt_time(st['p95'])
        c = next((c for c in res.clocks if c['id'] == n.id), None)
        return CT.fmt_time(c['true']['p95']) + ' UTC' if c is not None else ''

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), THEME.color('panel'))
        if self.topo is None or not self.topo.nodes:
            p.setPen(THEME.color('muted'))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, 'No clock tree')
            p.end()
            return
        self._layout()
        topo = self.topo
        p.setPen(QPen(THEME.color('strong'), 1))
        for n in topo.nodes:
            if n.parent and n.parent in self.boxes and n.id in self.boxes:
                a, b = self.boxes[n.parent], self.boxes[n.id]
                ym = (a.bottom() + b.top()) / 2
                path = QPainterPath(QPointF(a.center().x(), a.bottom()))
                path.lineTo(a.center().x(), ym)
                path.lineTo(b.center().x(), ym)
                path.lineTo(b.center().x(), b.top())
                dashed = n.parent == 'gnss'                 # a PPS cable, not a PTP link
                p.setPen(QPen(THEME.color('muted' if dashed else 'strong'), 1,
                              Qt.PenStyle.DashLine if dashed else Qt.PenStyle.SolidLine))
                p.drawPath(path)
        f1, f2 = sans(11, QFont.Weight.DemiBold), mono(10)
        for n in topo.nodes:
            r = self.boxes.get(n.id)
            if r is None:
                continue
            sel, hov = n.id == self.sel, n.id == self.hover
            p.setPen(QPen(THEME.color('ink' if sel else 'strong'), 1.5 if sel else 1))
            p.setBrush(THEME.color('sel' if hov or sel else 'raised'))
            p.drawRoundedRect(r, 4, 4)
            if n.id == topo.gm:
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.setPen(QPen(THEME.color('ink2'), 1))
                p.drawRoundedRect(r.adjusted(-3, -3, 3, 3), 6, 6)
            if n.kind == 'camera':
                p.fillRect(QRectF(r.left() + 5, r.top() + 6, 7, 7), THEME.cam(n.slot))
            dx = 15 if n.kind == 'camera' else 6
            name = n.info.get('tag') or n.label.split(' · ')[0]
            if n.kind in ('server', 'compute', 'switch', 'gnss'):
                name = {'gnss': 'GNSS', 'server': 'Time server', 'switch': 'Switch'}.get(n.kind, n.label)
            elif n.kind == 'deser':
                name = n.label.replace('Deserializer', 'Deser')
            p.setFont(f1)
            p.setPen(THEME.color('ink'))
            p.drawText(QRectF(r.left() + dx, r.top() + 2, r.width() - dx - 3, 16),
                       Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                       p.fontMetrics().elidedText(name, Qt.TextElideMode.ElideRight, int(r.width() - dx - 3)))
            fig = self._figure(n)
            p.setFont(f2)
            tone = None
            if self.res is not None:
                s = self.res.sensor(n.id)
                if s is not None and s['vs_lidar'] is not None:
                    v = s['vs_lidar']['p95'] * 1e6
                    tone = 'good' if v <= CT.ACCURACY_US else 'warning' if v <= 100 else 'serious'
            p.setPen(QColor(STATUS[tone]) if tone else THEME.color('muted'))
            p.drawText(QRectF(r.left() + 6, r.top() + 17, r.width() - 9, 15),
                       Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, fig or CT.KIND_LABEL[n.kind])
        p.end()

    def node_at(self, pos):
        return next((nid for nid, r in self.boxes.items() if r.contains(QPointF(pos))), '')

    def mouseMoveEvent(self, e):
        h = self.node_at(e.position())
        if h != self.hover:
            self.hover = h
            n = self.topo.node(h) if (h and self.topo) else None
            self.setToolTip(f'{n.label}\n{CT.KIND_LABEL[n.kind]}' + (f'\n{n.note}' if n.note else '') if n else '')
            self.update()

    def leaveEvent(self, e):
        self.hover = ''
        self.update()

    def mousePressEvent(self, e):
        h = self.node_at(e.position())
        if h:
            self.sel = h
            self.update()
            self.selected.emit(h)


# =========================================================================== charts
def _log_y(v, lo, hi, top, bottom):
    v = math.log10(max(v, 10 ** lo))
    return bottom - (bottom - top) * (min(max(v, lo), hi) - lo) / (hi - lo)


_TIME_TICKS = ((-9, '1 ns'), (-8, '10 ns'), (-7, '100 ns'), (-6, '1 µs'), (-5, '10 µs'), (-4, '100 µs'),
               (-3, '1 ms'), (-2, '10 ms'), (-1, '100 ms'))


class DistChart(ChartBase):
    """Per sensor, its timestamp error on a log scale: a dot at the median, a bar to the 95th percentile and a tick at
    the worst frame; against the reference lidar or true time."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.rows = []
        self.mode = 'lidar'
        self.title = 'Timestamp error'
        self.setMinimumHeight(180)

    def set_result(self, res, mode='lidar'):
        self.mode = mode
        self.rows = []
        if res is not None:
            for s in res.sensors:
                st = s['vs_lidar'] if mode == 'lidar' else s['true']
                if st is None:
                    continue
                self.rows.append((s['tag'], s['kind'], s['slot'], st))
        self.subtitle = 'against the lidar' if mode == 'lidar' else 'against true time'
        self.update()

    def _geom(self):
        return QRectF(64, 26, max(self.width() - 76, 10), max(self.height() - 48, 10))

    def _range(self):
        vals = [v for *_, st in self.rows for v in st.values() if v]
        hi = math.ceil(math.log10(max(vals))) if vals else -3
        lo = math.floor(math.log10(max(min(vals), 1e-10))) if vals else -9
        return max(lo, -10), max(hi, lo + 2)

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.paint_title(p)
        r = self._geom()
        if not self.rows:
            p.setPen(THEME.color('muted'))
            p.drawText(r, Qt.AlignmentFlag.AlignCenter, 'Run the study')
            p.end()
            return
        lo, hi = self._range()
        p.setFont(mono(10))
        for k in range(lo, hi + 1):
            x = r.left() + r.width() * (k - lo) / (hi - lo)
            p.setPen(QPen(THEME.color('hair'), 1))
            p.drawLine(QPointF(x, r.top()), QPointF(x, r.bottom()))
            p.setPen(THEME.color('muted'))
            lbl = dict(_TIME_TICKS).get(k, f'1e{k} s')
            p.drawText(QRectF(x - 30, r.bottom() + 3, 60, 14), Qt.AlignmentFlag.AlignCenter, lbl)
        n = len(self.rows)
        rh = r.height() / n
        X = lambda v: r.left() + r.width() * (min(max(math.log10(max(v, 1e-12)), lo), hi) - lo) / (hi - lo)
        for i, (tag, kind, slot, st) in enumerate(self.rows):
            y = r.top() + (i + 0.5) * rh
            col = THEME.cam(slot) if kind == 'camera' else THEME.color('ink2')
            p.setFont(sans(10))
            p.setPen(col)
            p.drawText(QRectF(0, y - 7, r.left() - 6, 14), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                       tag)
            hov = self.hover == i
            bh = max(min(rh * 0.5, 10), 3)
            p.setPen(Qt.PenStyle.NoPen)
            c2 = QColor(col)
            c2.setAlphaF(0.85 if hov else 0.55)
            p.setBrush(c2)
            x0, x1 = X(st['p50'] or 1e-12), X(st['p95'] or 1e-12)
            p.drawRoundedRect(QRectF(x0, y - bh / 2, max(x1 - x0, 2), bh), 2, 2)
            p.setPen(QPen(col, 1))
            xm = X(st['max'] or 1e-12)
            p.drawLine(QPointF(x1, y), QPointF(xm, y))
            p.drawLine(QPointF(xm, y - bh / 2 - 2), QPointF(xm, y + bh / 2 + 2))
            p.setBrush(THEME.color('ink'))
            p.drawEllipse(QPointF(x0, y), 2.5, 2.5)
        self.paint_tip(p)
        p.end()

    def mouseMoveEvent(self, e):
        r = self._geom()
        if not self.rows:
            return
        i = int((e.position().y() - r.top()) / (r.height() / len(self.rows)))
        if 0 <= i < len(self.rows) and r.top() <= e.position().y() <= r.bottom():
            tag, kind, slot, st = self.rows[i]
            self.hover = i
            self.set_tip(e.position(), f'{CT.fmt_time(st["p95"])} p95', f'{tag}: median {CT.fmt_time(st["p50"])}, '
                                                                        f'worst {CT.fmt_time(st["max"])}',
                         self.subtitle)
        else:
            self.hover, self.tip = None, None
        self.update()


class TimelineChart(ChartBase):
    """The scenario on its timeline: the events as bands, and the chosen node's timestamp error over time (log scale):
    the median line, the band to the 95th percentile and the worst frame dotted."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.res = None
        self.nid = ''
        self.title = 'Timeline'
        self.setMinimumHeight(170)

    def set_result(self, res, nid=''):
        self.res, self.nid = res, nid
        self.update()

    def _series(self):
        res = self.res
        if res is None or not self.nid:
            return None, ''
        for key, what in (('rel:' + self.nid, 'against the lidar'), (self.nid, 'against true time'),
                          ('clock:' + self.nid, 'clock against true time')):
            if key in res.series:
                return res.series[key], what
        return None, ''

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        res = self.res
        ser, what = self._series()
        n = res.topology.node(self.nid) if res is not None and self.nid else None
        self.subtitle = (f'{n.label} · {what}' if n is not None and ser is not None else
                         'pick a node' if res is not None else '')
        self.paint_title(p)
        r = QRectF(52, 24, max(self.width() - 60, 10), max(self.height() - 44, 10))
        p.setPen(QPen(THEME.color('strong'), 1))
        p.drawRect(r)
        if res is None:
            p.end()
            return
        D = res.settings.duration_s
        X = lambda t: r.left() + r.width() * t / D
        p.setFont(sans(9))
        for ev in res.events:
            a = X(ev['t_s'])
            band = ev['kind'] in ('pps_dropout', 'gm_loss')
            end = ev['t_s'] + ev['dur_s'] if ev['dur_s'] > 0 else D
            b = X(min(end, D)) if band else a + 2
            c = THEME.color('serious' if ev['kind'] in ('gm_loss', 'fsync_missed') else 'warning', 0.16)
            p.fillRect(QRectF(a, r.top(), max(b - a, 2), r.height()), c)
            p.setPen(THEME.color('muted'))
            p.drawText(QRectF(a + 3, r.top() + 2, 160, 12), Qt.AlignmentFlag.AlignLeft, CT.EVENT_LABEL[ev['kind']])
        vals = [] if ser is None else [v for k in ('p50', 'p95', 'max') for v in ser[k] if math.isfinite(v) and v > 0]
        hi = math.ceil(math.log10(max(vals))) if vals else -3
        lo = max(math.floor(math.log10(min(vals))) if vals else -9, hi - 7)
        hi = max(hi, lo + 1)
        p.setFont(mono(9))
        for k in range(lo, hi + 1):
            y = _log_y(10.0 ** k, lo, hi, r.top(), r.bottom())
            p.setPen(QPen(THEME.color('hair'), 1))
            p.drawLine(QPointF(r.left(), y), QPointF(r.right(), y))
            p.setPen(THEME.color('muted'))
            p.drawText(QRectF(0, y - 7, r.left() - 4, 14), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                       dict(_TIME_TICKS).get(k, f'1e{k}'))
        for i in range(7):
            t = D * i / 6
            p.drawText(QRectF(X(t) - 24, r.bottom() + 2, 48, 14), Qt.AlignmentFlag.AlignCenter, f'{t:.0f} s')
        if ser is None:
            p.end()
            return
        ts = res.series['t']
        col = THEME.cam(n.slot) if n is not None and n.kind == 'camera' else THEME.color('ink2')
        band = QPainterPath()
        top = [(X(t), _log_y(v, lo, hi, r.top(), r.bottom())) for t, v in zip(ts, ser['p95']) if math.isfinite(v)]
        bot = [(X(t), _log_y(v, lo, hi, r.top(), r.bottom())) for t, v in zip(ts, ser['p50']) if math.isfinite(v)]
        if top and bot:
            band.moveTo(*top[0])
            for q in top[1:]:
                band.lineTo(*q)
            for q in bot[::-1]:
                band.lineTo(*q)
            band.closeSubpath()
            c2 = QColor(col)
            c2.setAlphaF(0.25)
            p.fillPath(band, c2)
        for key, style, width in (('p50', Qt.PenStyle.SolidLine, 1.5), ('max', Qt.PenStyle.DotLine, 1.0)):
            pts = [QPointF(X(t), _log_y(v, lo, hi, r.top(), r.bottom())) for t, v in zip(ts, ser[key])
                   if math.isfinite(v)]
            p.setPen(QPen(col, width, style))
            for a, b in zip(pts, pts[1:]):
                p.drawLine(a, b)
        self.paint_tip(p)
        p.end()

    def mouseMoveEvent(self, e):
        ser, what = self._series()
        if ser is None:
            return
        D = self.res.settings.duration_s
        r = QRectF(52, 24, max(self.width() - 60, 10), max(self.height() - 44, 10))
        t = (e.position().x() - r.left()) / r.width() * D
        ts = self.res.series['t']
        if not r.contains(e.position()):
            self.tip = None
        else:
            i = int(np.argmin(np.abs(ts - t)))
            self.set_tip(e.position(), f'{CT.fmt_time(ser["p95"][i])} p95',
                         f'at {ts[i]:.1f} s: median {CT.fmt_time(ser["p50"][i])}, worst {CT.fmt_time(ser["max"][i])}',
                         what)
        self.update()


class HeatMap(QWidget):
    """A camera's image: the projection error at each sample over the image (colour, log scale about the limit) and
    arrows from where a lidar point is to where the slipped stamps put it (lengthened to be seen)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.pe = None
        self.segs = []
        self.limit = 1.0
        self.scale = 1.0
        self.caption = ''
        self.setMinimumHeight(200)
        self.setMouseTracking(True)
        THEME.changed.connect(self._retheme)

    def _retheme(self, *_):
        self.update()

    def set_map(self, pe, segs, limit, scale, caption):
        self.pe, self.segs, self.limit, self.scale, self.caption = pe, segs, limit, scale, caption
        self.update()

    def _frame(self):
        if self.pe is None:
            return None, 1.0
        W, H = self.pe['width'], self.pe['height']
        avail = QRectF(8, 22, self.width() - 70, self.height() - 30)
        k = min(avail.width() / W, avail.height() / H)
        return QRectF(avail.left(), avail.top(), W * k, H * k), k

    def colour(self, v):
        lut = colormaps.ramp(ERR_RAMP)
        lo, hi = math.log10(self.limit / 100.0), math.log10(self.limit * 10.0)
        t = (math.log10(max(v, 1e-9)) - lo) / (hi - lo)
        r_, g, b = lut[int(min(max(t, 0.0), 1.0) * 255)]
        return QColor(int(r_), int(g), int(b))

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), THEME.color('panel'))
        p.setFont(sans(10))
        p.setPen(THEME.color('ink2'))
        p.drawText(QRectF(8, 2, self.width() - 16, 18), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                   self.caption)
        fr, k = self._frame()
        if fr is None:
            p.end()
            return
        pe = self.pe
        gu, gv = pe['grid']
        err = pe['err'][1:].reshape(gv, gu)
        lut = colormaps.ramp(ERR_RAMP)
        lo, hi = math.log10(self.limit / 100.0), math.log10(self.limit * 10.0)
        with np.errstate(divide='ignore', invalid='ignore'):
            t = np.clip((np.log10(np.maximum(err, 1e-9)) - lo) / (hi - lo), 0.0, 1.0)
        rgb = lut[np.nan_to_num(t * 255).astype(np.intp)]
        a = np.where(np.isfinite(err), 255, 0).astype(np.uint8)
        buf = np.ascontiguousarray(np.dstack([rgb, a]).astype(np.uint8))
        img = QImage(buf.data, gu, gv, gu * 4, QImage.Format.Format_RGBA8888).copy()
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        p.drawImage(fr, img)
        p.setPen(QPen(THEME.color('strong'), 1))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRect(fr)
        p.setPen(QPen(THEME.color('ink'), 1.2))
        for (u0, v0), (u1, v1), _ in self.segs:
            a_ = QPointF(fr.left() + u0 * k, fr.top() + v0 * k)
            b_ = QPointF(fr.left() + u1 * k, fr.top() + v1 * k)
            p.drawLine(a_, b_)
            p.drawEllipse(b_, 1.5, 1.5)
        # colour bar
        bar = QRectF(fr.right() + 14, fr.top(), 10, fr.height())
        for i in range(64):
            v = 10 ** (lo + (hi - lo) * (1 - (i + 0.5) / 64))
            p.fillRect(QRectF(bar.left(), bar.top() + bar.height() * i / 64, bar.width(), bar.height() / 64 + 1),
                       self.colour(v))
        p.setFont(mono(9))
        p.setPen(THEME.color('muted'))
        for v in (self.limit / 100, self.limit / 10, self.limit, self.limit * 10):
            y = bar.top() + bar.height() * (1 - (math.log10(v) - lo) / (hi - lo))
            p.drawText(QRectF(bar.right() + 2, y - 7, 40, 14), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                       f'{v:g}')
        p.drawText(QRectF(bar.left() - 4, bar.bottom() + 1, 50, 12), Qt.AlignmentFlag.AlignLeft, 'px')
        p.end()

    def mouseMoveEvent(self, e):
        fr, k = self._frame()
        if fr is None or not fr.contains(e.position()):
            self.setToolTip('')
            return
        u = (e.position().x() - fr.left()) / k
        v = (e.position().y() - fr.top()) / k
        pe = self.pe
        i = 1 + int(np.argmin((pe['u'][1:] - u) ** 2 + (pe['v'][1:] - v) ** 2))
        val = pe['err'][i]
        self.setToolTip(f'{val:.3f} px at ({u:.0f}, {v:.0f})' if math.isfinite(val) else 'No point here')


# =========================================================================== the page
class ClockTreePage(QWidget):
    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        self.res: CT.ClockTreeResult | None = None
        self.topo: CT.Topology | None = None
        self.sel = ''                       # the node picked in the tree or the table
        self.busy = False
        self.status_text = ''
        self._gen = 0
        self._dirty = True
        self._syncing = False
        self.sig = _Sig()
        self.sig.done.connect(self._done)
        self.sig.failed.connect(self._failed)
        self.sig.progress.connect(lambda g, f: self.bar.setValue(int(f * 1000)) if g == self._gen else None)
        self.setObjectName('base')
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        split = QSplitter(Qt.Orientation.Horizontal)
        split.setHandleWidth(2)
        split.addWidget(self._left())
        split.addWidget(self._centre())
        split.addWidget(self._right())
        split.setSizes([310, 760, 620])
        lay.addWidget(split)
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(400)
        self.timer.timeout.connect(self.run)
        ctx.store.changed.connect(self._changed)
        ctx.store.project_replaced.connect(self._changed)
        self._sync_fields()
        self._refresh_tree()

    # ------------------------------------------------------------------ panels
    def _grid(self, fields, ll):
        grid = QGridLayout()
        grid.setVerticalSpacing(2)
        grid.setHorizontalSpacing(6)
        grid.setColumnStretch(2, 1)
        for i, (key, text, kind, dec, unit, step, tip) in enumerate(fields):
            a = label(text, ink2=True)
            a.setFixedWidth(84)
            f = ScrubSpinBox('', kind, dec, unit, step=step, width=96)
            f.setToolTip(tip)
            f.committed.connect(lambda v, k=key: self.set_option(k, v))
            grid.addWidget(a, i, 0)
            grid.addWidget(f, i, 1)
            self.fields[key] = f
        ll.addLayout(grid)
        return grid

    def _left(self):
        w = QWidget()
        w.setObjectName('panel')
        ll = QVBoxLayout(w)
        ll.setContentsMargins(8, 6, 8, 8)
        ll.setSpacing(4)
        top = QHBoxLayout()
        top.addWidget(label('Clock tree and time sync'))
        top.addStretch(1)
        ll.addLayout(top)
        ll.addWidget(hline())
        ll.addWidget(label('Time flows from the GNSS PPS through the grandmaster and the switch to every sensor, and '
                           'to the cameras through the deserializers\' FSYNC. Add faults to the timeline: the Monte '
                           'Carlo gives each sensor\'s timestamp error and how far it moves lidar points in the '
                           'images.', muted=True, small=True, wrap=True))
        self.fields = {}
        ll.addSpacing(4)
        ll.addWidget(label('Monte Carlo', bold=True))
        self._grid(RUN_FIELDS, ll)
        ll.addSpacing(4)
        row = QHBoxLayout()
        row.addWidget(label('Scenario', bold=True))
        row.addStretch(1)
        self.preset = QComboBox()
        self.preset.setToolTip('Fill the timeline with a scenario\'s faults')
        self.preset.addItem('Custom', '')
        for key, text in CT.PRESETS:
            self.preset.addItem(text, key)
        self.preset.activated.connect(lambda i: self.apply_preset(self.preset.itemData(i)))
        row.addWidget(self.preset)
        ll.addLayout(row)
        self.events = _table(EVENT_COLS, stretch=0)
        self.events.setEditTriggers(QAbstractItemView.EditTrigger.DoubleClicked |
                                    QAbstractItemView.EditTrigger.EditKeyPressed)
        self.events.itemChanged.connect(self._event_edited)
        self.events.setMinimumHeight(ROW_H * 4)
        ll.addWidget(self.events)
        row = QHBoxLayout()
        self.b_add = button('Add fault', None, 'plus', 'Add a fault to the timeline')
        menu = QMenu(self.b_add)
        for k in CT.EVENT_KINDS:
            a = menu.addAction(CT.EVENT_LABEL[k])
            a.setToolTip(CT.EVENT_HELP[k])
            a.triggered.connect(lambda _=False, k=k: self.add_event(k))
        self.b_add.setMenu(menu)
        self.b_del = button('Remove', None, 'minus', 'Remove the selected fault')
        self.b_del.clicked.connect(lambda: self.remove_event(self.events.currentRow()))
        row.addWidget(self.b_add)
        row.addWidget(self.b_del)
        row.addStretch(1)
        ll.addLayout(row)
        ll.addSpacing(4)
        ll.addWidget(label('Projection', bold=True))
        self._grid(MOTION_FIELDS, ll)
        self.surface = SegmentedControl([('range', 'At range'), ('ground', 'On the ground')], 'range')
        self.surface.setToolTip('Points at the range from each camera, or where its rays meet the ground')
        self.surface.changed.connect(lambda k: self.set_option('surface', k))
        ll.addWidget(self.surface)
        self.rows = ToggleSwitch('One stamp per image (rolling shutter)')
        self.rows.setToolTip('On: the pipeline uses one timestamp per image, so each row is off by (row/H − ½) × '
                             'the readout as well. Off: rows are compensated')
        self.rows.toggled.connect(lambda on: self.set_option('rows_compensated', not on))
        ll.addWidget(self.rows)
        ll.addSpacing(4)
        ll.addWidget(label('Limits', bold=True))
        self._grid(LIMIT_FIELDS, ll)
        ll.addSpacing(4)
        ll.addWidget(label('Slip', bold=True))
        grid = QGridLayout()
        grid.setHorizontalSpacing(6)
        a = label('Time error', ink2=True)
        a.setFixedWidth(84)
        f = ScrubSpinBox('', 'plain', 2, 'ms', step=0.1, width=96)
        f.setToolTip('A time error between a lidar point and a camera image')
        f.committed.connect(lambda v: self.set_option('slip_ms', v))
        self.fields['slip_ms'] = f
        grid.addWidget(a, 0, 0)
        grid.addWidget(f, 0, 1)
        self.slip_cam = QComboBox()
        self.slip_cam.setToolTip('The camera the slip is projected into')
        self.slip_cam.activated.connect(lambda *_: self._fill_slip())
        grid.addWidget(self.slip_cam, 1, 0, 1, 3)
        ll.addLayout(grid)
        self.slip_text = label('', small=True, wrap=True, ink2=True)
        ll.addWidget(self.slip_text)
        ll.addStretch(1)
        self.status = label('', muted=True, small=True, wrap=True)
        ll.addWidget(self.status)
        self.bar = QProgressBar()
        self.bar.setTextVisible(False)
        self.bar.setRange(0, 1000)
        self.bar.hide()
        ll.addWidget(self.bar)
        bl = QHBoxLayout()
        self.b_export = button('Export…', None, 'export', 'The tree, the scenario and the errors as JSON and CSV, '
                                                          'and the timeline as PNG')
        self.b_export.clicked.connect(lambda: self.export())
        self.b_cancel = button('Cancel', None, 'stop', 'Stop the run')
        self.b_cancel.clicked.connect(self.cancel)
        self.b_run = button('Run', 'primary', None, 'Run the Monte Carlo again')
        self.b_run.clicked.connect(self.run)
        bl.addWidget(self.b_export)
        bl.addStretch(1)
        bl.addWidget(self.b_cancel)
        bl.addWidget(self.b_run)
        ll.addLayout(bl)
        sc = QScrollArea()
        sc.setWidgetResizable(True)
        sc.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        sc.setWidget(w)
        sc.setMinimumWidth(300)
        return sc

    def _centre(self):
        sp = QSplitter(Qt.Orientation.Vertical)
        sp.setHandleWidth(2)
        tw = QWidget()
        tw.setObjectName('panel')
        tl = QVBoxLayout(tw)
        tl.setContentsMargins(8, 6, 8, 6)
        self.head = label('', ink2=True, wrap=True)
        tl.addWidget(self.head)
        self.tree = TopologyView()
        self.tree.selected.connect(self.select_node)
        tl.addWidget(self.tree, 1)
        sp.addWidget(tw)
        bw = QWidget()
        bw.setObjectName('panel')
        bl = QHBoxLayout(bw)
        bl.setContentsMargins(8, 6, 8, 6)
        left = QVBoxLayout()
        left.addWidget(label('Nodes', bold=True))
        self.nodes = _table(NODE_COLS)
        self.nodes.itemSelectionChanged.connect(self._node_row)
        left.addWidget(self.nodes, 1)
        bl.addLayout(left, 3)
        right = QVBoxLayout()
        row = QHBoxLayout()
        self.p_title = label('Parameters', bold=True)
        row.addWidget(self.p_title)
        row.addStretch(1)
        self.b_reset = button('Typical', 'ghost', None, 'Back to the typical values for this node')
        self.b_reset.clicked.connect(lambda: self.reset_node(self.sel))
        row.addWidget(self.b_reset)
        right.addLayout(row)
        self.params = _table(PARAM_COLS, stretch=2)
        self.params.setEditTriggers(QAbstractItemView.EditTrigger.DoubleClicked |
                                    QAbstractItemView.EditTrigger.EditKeyPressed)
        self.params.itemChanged.connect(self._param_edited)
        right.addWidget(self.params, 1)
        self.p_note = label('', small=True, wrap=True, muted=True)
        right.addWidget(self.p_note)
        bl.addLayout(right, 4)
        sp.addWidget(bw)
        sp.setSizes([420, 380])
        return sp

    def _right(self):
        sp = QSplitter(Qt.Orientation.Vertical)
        sp.setHandleWidth(2)
        tabs = QTabWidget()
        self.tabs = tabs
        # distributions
        w = QWidget()
        l_ = QVBoxLayout(w)
        l_.setContentsMargins(8, 6, 8, 6)
        self.dist_mode = SegmentedControl([('lidar', 'Against the lidar'), ('true', 'Against true time')], 'lidar')
        self.dist_mode.changed.connect(lambda k: self.dist.set_result(self.res, k))
        l_.addWidget(self.dist_mode)
        self.dist = DistChart()
        l_.addWidget(self.dist, 1)
        self.timeline = TimelineChart()
        l_.addWidget(self.timeline, 1)
        tabs.addTab(w, 'Errors')
        # camera image
        w = QWidget()
        l_ = QVBoxLayout(w)
        l_.setContentsMargins(8, 6, 8, 6)
        row = QHBoxLayout()
        self.map_cam = QComboBox()
        self.map_cam.setToolTip('The camera whose image is shown')
        self.map_cam.activated.connect(lambda *_: self._fill_map())
        row.addWidget(self.map_cam)
        self.map_which = SegmentedControl([('p95', 'Δt p95'), ('max', 'Δt worst'), ('slip', 'Slip')], 'max')
        self.map_which.changed.connect(lambda *_: self._fill_map())
        row.addWidget(self.map_which)
        row.addStretch(1)
        l_.addLayout(row)
        self.heat = HeatMap()
        l_.addWidget(self.heat, 1)
        tabs.addTab(w, 'Camera image')
        # worst case
        w = QWidget()
        l_ = QVBoxLayout(w)
        l_.setContentsMargins(8, 6, 8, 6)
        self.proj = _table(PROJ_COLS)
        l_.addWidget(self.proj, 1)
        self.proj_note = label('', small=True, wrap=True, muted=True)
        l_.addWidget(self.proj_note)
        tabs.addTab(w, 'Worst case')
        sp.addWidget(tabs)
        fw = QWidget()
        fw.setObjectName('panel')
        fl = QVBoxLayout(fw)
        fl.setContentsMargins(8, 6, 8, 6)
        fl.addWidget(label('Findings', bold=True))
        self.findings_box = QVBoxLayout()
        self.findings_box.setSpacing(3)
        fl.addLayout(self.findings_box)
        fl.addStretch(1)
        sc = QScrollArea()
        sc.setWidgetResizable(True)
        sc.setWidget(fw)
        sp.addWidget(sc)
        sp.setSizes([560, 300])
        return sp

    # ------------------------------------------------------------------ settings
    @property
    def settings(self) -> CT.ClockTreeSettings:
        return CT.settings_of(self.ctx.store.project)

    def _store(self, st: CT.ClockTreeSettings, text) -> bool:
        new = CT.ClockTreeSettings.from_dict(st.to_dict()).to_dict()
        if new == self.settings.to_dict():
            self._sync_fields()
            return False

        def fn(p):
            p.analysis.clock_tree = new
        self.ctx.store.edit('analysis', fn, f'Clock tree · {text}')
        return True

    def set_option(self, key, value) -> bool:
        """Change one setting, saved with the project as one undo step; False when nothing changes."""
        if self._syncing:
            return False
        st = self.settings
        setattr(st, key, value)
        name = next((f[1].lower() for f in RUN_FIELDS + MOTION_FIELDS + LIMIT_FIELDS if f[0] == key),
                    key.replace('_', ' '))
        return self._store(st, f'{name} {value:g}' if isinstance(value, float) else f'{name} {value}')

    def apply_preset(self, key) -> bool:
        if not key or self.topo is None:
            return False
        st = self.settings
        st.events = CT.preset_events(key, self.topo, st.duration_s)
        st.preset = key
        return self._store(st, f'scenario {dict(CT.PRESETS)[key].lower()}')

    def add_event(self, kind) -> bool:
        st = self.settings
        if len(st.events) >= CT.MAX_EVENTS:
            return False
        node = ''
        if kind in ('fsync_missed', 'fsync_late'):
            node = next((n.id for n in self.topo.nodes if n.kind == 'deser'), '') if self.topo else ''
        elif kind == 'asymmetry':
            node = self.topo.lidar if self.topo else ''
        st.events.append(CT.clean_event({'kind': kind, 't_s': round(st.duration_s / 2, 1), 'node': node}))
        st.preset = ''
        return self._store(st, f'add {CT.EVENT_LABEL[kind].lower()}')

    def remove_event(self, i) -> bool:
        st = self.settings
        if not 0 <= i < len(st.events):
            return False
        e = st.events.pop(i)
        st.preset = ''
        return self._store(st, f'remove {CT.EVENT_LABEL[e["kind"]].lower()}')

    def set_event(self, i, **kw) -> bool:
        st = self.settings
        if not 0 <= i < len(st.events):
            return False
        st.events[i].update(kw)
        st.preset = ''
        return self._store(st, f'{CT.EVENT_LABEL[st.events[i]["kind"]].lower()} edited')

    def set_param(self, nid, key, value) -> bool:
        """Set one node parameter (an edit over its typical value), one undo step."""
        v = CT.clean_value(key, value)
        if v is None:
            self._fill_params()
            return False
        st = self.settings
        st.nodes.setdefault(nid, {})[key] = v
        n = self.topo.node(nid) if self.topo else None
        return self._store(st, f'{n.label if n else nid} {CT.PARAMS[key][0].lower()} {_value_text(key, v)}')

    def reset_node(self, nid) -> bool:
        st = self.settings
        if nid not in st.nodes:
            return False
        st.nodes.pop(nid)
        n = self.topo.node(nid) if self.topo else None
        return self._store(st, f'{n.label if n else nid} back to typical')

    def _sync_fields(self):
        st = self.settings
        self._syncing = True
        try:
            for key, f in self.fields.items():
                f.set_value(float(getattr(st, key)))
            self.surface.set_value(st.surface)
            self.rows.blockSignals(True)
            self.rows.setChecked(not st.rows_compensated)
            self.rows.blockSignals(False)
            self.fields['range_m'].setEnabled(st.surface == 'range')
            i = self.preset.findData(st.preset if st.preset else '')
            self.preset.setCurrentIndex(max(i, 0))
        finally:
            self._syncing = False

    # ------------------------------------------------------------------ running
    def _changed(self, *_):
        self._dirty = True
        self._sync_fields()
        self._refresh_tree()
        if self.isVisible():
            self.timer.start()

    def showEvent(self, e):
        super().showEvent(e)
        if self._dirty:
            self.timer.start()

    def activate(self):
        if self._dirty:
            self.run()

    def _refresh_tree(self):
        try:
            self.topo = CT.build_topology(self.ctx.store.project)
        except Exception as e:                 # keep the page alive; say what went wrong
            self.status.setText(f'Failed: {type(e).__name__}: {e}')
            traceback.print_exc()
            return
        if self.sel and self.topo.node(self.sel) is None:
            self.sel = ''
        self.refresh()

    def run(self) -> bool:
        """Run the study in the background (a newer run replaces an older one)."""
        self.timer.stop()
        self._dirty = False
        project = self.ctx.store.project.snapshot() if hasattr(self.ctx.store.project, 'snapshot') else \
            self.ctx.store.project
        self._gen += 1
        gen = self._gen
        self.busy = True
        self.bar.setValue(0)
        self.bar.show()
        self.b_cancel.setEnabled(True)
        self.status.setText('Running the Monte Carlo…')

        def thread():
            try:
                res = CT.run(project, progress=lambda f: self.sig.progress.emit(gen, f),
                             cancel=lambda: gen != self._gen)
                self.sig.done.emit(gen, res)
            except RuntimeError:            # the page closed while computing
                pass
            except Exception as e:
                try:
                    self.sig.failed.emit(gen, f'{type(e).__name__}: {e}\n{traceback.format_exc()}')
                except RuntimeError:
                    pass
        threading.Thread(target=thread, daemon=True).start()
        return True

    def cancel(self):
        self._gen += 1
        self.busy = False
        self.bar.hide()
        self.b_cancel.setEnabled(False)
        self.status.setText('Cancelled')

    def _done(self, gen, res):
        if gen != self._gen:
            return
        self.busy = False
        self.bar.hide()
        self.b_cancel.setEnabled(False)
        if res is None:
            self.status.setText('Cancelled')
            return
        self.res = res
        self.topo = res.topology
        bad = sum(f['severity'] in ('critical', 'serious', 'warning') for f in res.findings)
        self.status.setText(f'{res.variant}: {res.settings.runs} runs × {res.settings.duration_s:g} s · '
                            + (f'{bad} finding{"s" * (bad != 1)} to look at' if bad else 'nothing to flag')
                            + f' · {res.seconds:.1f} s')
        self.refresh()

    def _failed(self, gen, msg):
        if gen != self._gen:
            return
        self.busy = False
        self.bar.hide()
        self.b_cancel.setEnabled(False)
        self.status.setText('Failed: ' + msg.splitlines()[0])

    def wait(self, timeout=60.0) -> bool:
        """Block until the current run is done (tests and scripts)."""
        import time
        from PyQt6.QtWidgets import QApplication
        t = time.time()
        while (self.busy or self.timer.isActive()) and time.time() - t < timeout:
            QApplication.processEvents()
            time.sleep(0.01)
        QApplication.processEvents()
        return not self.busy

    # ------------------------------------------------------------------ the views
    def select_node(self, nid):
        self.sel = nid
        self.tree.select(nid)
        self._syncing = True
        try:
            for r in range(self.nodes.rowCount()):
                if self.nodes.item(r, 0).data(Qt.ItemDataRole.UserRole) == nid:
                    self.nodes.selectRow(r)
                    break
        finally:
            self._syncing = False
        self._fill_params()
        self.timeline.set_result(self.res, nid)

    def _node_row(self):
        if self._syncing:
            return
        r = self.nodes.currentRow()
        it = self.nodes.item(r, 0) if r >= 0 else None
        if it is not None:
            self.select_node(it.data(Qt.ItemDataRole.UserRole))

    def refresh(self):
        res, topo = self.res, self.topo
        self.b_export.setEnabled(res is not None)
        valid = res is not None and topo is not None and res.topology.variant_uid == topo.variant_uid
        self.tree.set_tree(topo, res if valid else None)
        if topo is not None and not self.sel:
            self.sel = topo.lidar or topo.gm
            self.tree.select(self.sel)
        if res is not None:
            m, st = res.metrics, res.settings
            w = m['worst_max_px']
            self.head.setText(
                f'Grandmaster {m["gm"] or "none"} · holds UTC within {st.time_budget_us:g} µs for '
                f'{m["holdover_s"]:.0f} s without PPS · camera–lidar Δt {CT.fmt_time((m["worst_dt_p95_ms"] or 0) / 1e3)}'
                f' p95 · worst projection ' + (f'{w:.2f} px ({m["worst_max_tag"]})' if w is not None else '—'))
        elif topo is not None:
            self.head.setText(f'{len(topo.nodes)} nodes · grandmaster '
                              f'{topo.node(topo.gm).label if topo.gm else "none"}')
        self._fill_events()
        self._fill_nodes()
        self._fill_params()
        self.dist.set_result(res, self.dist_mode.value() or 'lidar')
        self.timeline.set_result(res, self.sel)
        self._fill_cams()
        self._fill_map()
        self._fill_proj()
        self._fill_slip()
        self._fill_findings()

    def _fill_events(self):
        st = self.settings
        t = self.events
        self._syncing = True
        try:
            t.setRowCount(len(st.events))
            for r, e in enumerate(st.events):
                t.setItem(r, 0, _item(CT.EVENT_LABEL[e['kind']], CT.EVENT_HELP[e['kind']], data=r))
                t.setItem(r, 1, _item(f'{e["t_s"]:g}', num=True, editable=True, data=r))
                if e['kind'] in ('pps_dropout', 'gm_loss'):
                    t.setItem(r, 2, _item(f'{e["dur_s"]:g} s' if e['dur_s'] > 0 else 'for good', num=True,
                                          editable=True, data=r))
                elif e['kind'] in CT.EVENT_UNIT:
                    t.setItem(r, 2, _item(f'{e["value"]:g} {CT.EVENT_UNIT[e["kind"]]}', num=True, editable=True,
                                          data=r))
                else:
                    t.setItem(r, 2, _item('', data=r))
                if 'node' in CT.EVENT_FIELDS[e['kind']]:
                    cb = QComboBox()
                    opts = self._event_nodes(e['kind'])
                    for nid, text in opts:
                        cb.addItem(text, nid)
                    i = cb.findData(e['node'])
                    if i < 0:
                        cb.addItem(f'{e["node"] or "—"} (not in this variant)', e['node'])
                        i = cb.count() - 1
                    cb.setCurrentIndex(i)
                    cb.activated.connect(lambda k, r=r, cb=cb: QTimer.singleShot(
                        0, lambda v=cb.itemData(k): self.set_event(r, node=v)))
                    t.setCellWidget(r, 3, cb)
                else:
                    t.removeCellWidget(r, 3)
                    t.setItem(r, 3, _item('', data=r))
        finally:
            self._syncing = False

    def _event_nodes(self, kind):
        if self.topo is None:
            return []
        if kind == 'asymmetry':
            return [(n.id, n.label) for n in self.topo.nodes
                    if n.parent and n.parent != 'gnss' and 'asym_ns' in n.params()]
        return [(n.id, n.label) for n in self.topo.nodes if n.kind in ('deser', 'camera')]

    def _event_edited(self, item):
        if self._syncing:
            return
        r, c = item.row(), item.column()
        st = self.settings
        if not 0 <= r < len(st.events) or c not in (1, 2):
            return
        e = st.events[r]
        text = item.text().replace('for good', '0').strip()
        try:
            v = float(parse_number(text.split()[0] if text else '', 'plain'))
        except Exception:                      # parse_number raises on anything that is not a number
            QTimer.singleShot(0, self._fill_events)
            return
        key = 't_s' if c == 1 else 'dur_s' if e['kind'] in ('pps_dropout', 'gm_loss') else \
            'value' if e['kind'] in CT.EVENT_UNIT else None
        if key:                                # later: the item is still being committed
            QTimer.singleShot(0, lambda: self.set_event(r, **{key: v}))

    def _fill_nodes(self):
        topo, res = self.topo, self.res
        t = self.nodes
        if topo is None:
            t.setRowCount(0)
            return
        self._syncing = True
        try:
            t.setRowCount(len(topo.nodes))
            for r, n in enumerate(topo.nodes):
                it = _item(('  ' * len(topo.ancestors(n.id)[1:])) + n.label, n.note or CT.KIND_LABEL[n.kind],
                           THEME.cam(n.slot) if n.kind == 'camera' else None, data=n.id)
                t.setItem(r, 0, it)
                how = (dict(CT.CHOICES['stamping']).get(n.p.get('stamping'), '') if 'stamping' in n.p else
                       'grandmaster' if n.id == topo.gm else
                       {'gnss': 'PPS', 'switch': 'relays', 'deser': f'FSYNC {n.p.get("fsync", "")}'}.get(
                           n.kind, 'PTP' if n.parent != 'gnss' else 'PPS'))
                t.setItem(r, 1, _item(how))
                s = res.sensor(n.id) if res is not None else None
                c = next((c for c in res.clocks if c['id'] == n.id), None) if res is not None else None
                tr = (s['true'] if s else c['true'] if c else None)
                rel = s['vs_lidar'] if s else None
                t.setItem(r, 2, _item(CT.fmt_time(tr['p95']) if tr else '—', num=True))
                tone = None
                if rel is not None:
                    tone = 'good' if rel['p95'] * 1e6 <= CT.ACCURACY_US else 'warning' if rel['p95'] < 1e-4 else \
                        'serious'
                t.setItem(r, 3, _item(CT.fmt_time(rel['p95']) if rel else '—', num=True,
                                      color=QColor(STATUS[tone]) if tone else None))
                mx = (rel or tr or {}).get('max')
                t.setItem(r, 4, _item(CT.fmt_time(mx) if mx is not None else '—', num=True))
                if n.id == self.sel:
                    t.selectRow(r)
        finally:
            self._syncing = False

    def _fill_params(self):
        t = self.params
        n = self.topo.node(self.sel) if self.topo is not None and self.sel else None
        self._syncing = True
        try:
            if n is None:
                t.setRowCount(0)
                self.p_title.setText('Parameters')
                self.p_note.setText('Pick a node in the tree.')
                self.b_reset.setEnabled(False)
                return
            self.p_title.setText(n.label)
            keys = list(n.params())
            t.setRowCount(len(keys))
            for r, k in enumerate(keys):
                lab, unit, *_rest, tip = CT.PARAMS[k]
                t.setItem(r, 0, _item(lab, tip, data=k))
                ed = k in n.edited
                if k in CT.CHOICES:
                    cb = QComboBox()
                    allowed = CT.STAMPING.get(n.kind) if k == 'stamping' else [c for c, _ in CT.CHOICES[k]]
                    for key, text in CT.CHOICES[k]:
                        if key in allowed:
                            cb.addItem(text, key)
                    cb.setCurrentIndex(max(cb.findData(n.p[k]), 0))
                    if n.kind == 'camera' and k == 'stamping' and not n.info.get('triggered'):
                        cb.setEnabled(False)
                        cb.setToolTip('The camera free-runs: it is stamped on arrival')
                    cb.activated.connect(lambda i, k=k, cb=cb, nid=n.id: QTimer.singleShot(
                        0, lambda v=cb.itemData(i): self.set_param(nid, k, v)))
                    t.setCellWidget(r, 1, cb)
                    t.setItem(r, 1, _item('', data=k))
                else:
                    t.removeCellWidget(r, 1)
                    t.setItem(r, 1, _item(_value_text(k, n.p[k]), 'Your value' if ed else 'Typical', num=True,
                                          editable=True, data=k))
                    if ed:
                        f = t.item(r, 1).font()
                        f.setWeight(QFont.Weight.DemiBold)
                        t.item(r, 1).setFont(f)
                typ = n.typical.get(k)
                t.setItem(r, 2, _item(f'{_value_text(k, typ[0])} · {typ[1]}' if typ else '', typ[1] if typ else ''))
            facts = [f'{k} {v:g}' if isinstance(v, float) else f'{k} {v}' for k, v in n.info.items()
                     if v not in ('', None)]
            sv = (self.res.sim.servo.get(n.id) if self.res is not None and self.res.sim is not None else None)
            extra = ''
            if sv and sv['capped']:
                extra = (f' The servo bandwidth is capped at {sv["bandwidth_hz"]:.2g} Hz by its update interval '
                         f'({sv["ts_s"]:g} s).')
            self.p_note.setText(' · '.join(facts) + ('. ' if facts else '') + (n.note or '') + extra)
            self.b_reset.setEnabled(bool(n.edited))
        finally:
            self._syncing = False

    def _param_edited(self, item):
        if self._syncing or item.column() != 1:
            return
        k = item.data(Qt.ItemDataRole.UserRole)
        if k not in CT.PARAMS or k in CT.CHOICES:
            return
        text = item.text().strip()
        unit = CT.PARAMS[k][1]
        if unit and text.endswith(unit):
            text = text[:-len(unit)].strip()
        try:
            v = float(parse_number(text, 'plain'))
        except Exception:                      # parse_number raises on anything that is not a number
            QTimer.singleShot(0, self._fill_params)
            return
        QTimer.singleShot(0, lambda nid=self.sel: self.set_param(nid, k, v))

    def _fill_cams(self):
        cams = [(n.id, n.uids[0], n.label) for n in (self.topo.nodes if self.topo else []) if n.kind == 'camera']
        for cb in (self.map_cam, self.slip_cam):
            cur = cb.currentData()
            if [(cb.itemData(i), cb.itemText(i)) for i in range(cb.count())] != [(u, t) for _, u, t in cams]:
                cb.blockSignals(True)
                cb.clear()
                for _, u, t in cams:
                    cb.addItem(t, u)
                cb.blockSignals(False)
            i = cb.findData(cur)
            cb.setCurrentIndex(i if i >= 0 else 0)

    def _cam(self, uid):
        return next((c for c in self.ctx.store.project.variant.cameras if c.uid == uid), None)

    def _fill_slip(self):
        st = self.settings
        cam = self._cam(self.slip_cam.currentData())
        if cam is None:
            self.slip_text.setText('No camera.')
            return
        sl = CT.slip(self.ctx.store.project, cam, st.slip_ms, st.speed_kmh, st.yaw_rate_dps, st.surface, st.range_m)
        where = 'on the ground' if st.surface == 'ground' else f'{st.range_m:g} m away'
        self.slip_text.setText(
            f'{st.slip_ms:g} ms turning {st.yaw_rate_dps:g}°/s at {st.speed_kmh:g} km/h: {sl["rotation_deg"]:.3f}° '
            f'and {sl["translation_mm"]:.1f} mm. {cam.tag}: {_px(sl["centre_px"])} px at the image centre, '
            f'{_px(sl["worst_px"])} px at worst ({where}).')

    def _fill_map(self):
        res = self.res
        cam = self._cam(self.map_cam.currentData())
        if cam is None:
            self.heat.set_map(None, [], 1.0, 1.0, '')
            return
        st = res.settings if res is not None else self.settings
        which = self.map_which.value() or 'max'
        row = res.camera(cam.uid) if res is not None else None
        if which == 'slip' or row is None:
            dt, what = st.slip_ms / 1000.0, f'slip {st.slip_ms:g} ms'
        else:
            dt = row[f'dt_{which}_ms'] / 1000.0
            what = f'Δt {"p95" if which == "p95" else "worst"} {row[f"dt_{which}_ms"]:.3g} ms'
        node = res.topology.node(f'cam:{cam.uid}') if res is not None else None
        ro = node.p['readout_ms'] if node is not None else 0.0
        pe = CT.projection_error(self.ctx.store.project, cam, dt, st.speed_mps, st.yaw_rate_dps, st.surface,
                                 st.range_m, grid=(61, 49), readout_ms=ro, rows_compensated=st.rows_compensated)
        worst = pe['worst_px'] or 0.0
        scale = min(max(40.0 / worst, 1.0), 1e5) if worst > 0 else 1.0     # arrows about 40 px long at worst
        scale = 10 ** math.floor(math.log10(scale)) if scale > 1 else 1.0
        segs = CT.image_segments(self.ctx.store.project, res, cam.uid, dt_s=dt, scale=scale) if res is not None \
            else []
        self.heat.set_map(pe, segs, st.threshold_px, scale,
                          f'{cam.tag} · {what} · worst {_px(pe["worst_px"])} px, centre {_px(pe["centre_px"])} px'
                          + (f' · arrows × {scale:g}' if scale > 1 else ''))

    def _fill_proj(self):
        res = self.res
        t = self.proj
        rows = res.projection if res is not None else []
        t.setRowCount(len(rows))
        lim = res.settings.threshold_px if res is not None else 1.0
        for r, c in enumerate(rows):
            t.setItem(r, 0, _item(c['tag'], f'against {c["against"]}', THEME.cam(c['slot'])))
            t.setItem(r, 1, _item(f'{c["dt_p95_ms"]:.3g}', num=True))
            t.setItem(r, 2, _item(f'{c["dt_max_ms"]:.3g}', num=True))
            for j, k in ((3, 'centre_p95_px'), (4, 'worst_p95_px'), (5, 'worst_max_px')):
                tone = _tone(c[k], lim)
                t.setItem(r, j, _item(_px(c[k]), num=True, color=QColor(STATUS[tone]) if tone else None))
            plan = (f'{c["plan_misalign_ms"]:+.1f} ms · {_px(c["plan_worst_px"], 1)} px'
                    if c['plan_misalign_ms'] is not None else '—')
            t.setItem(r, 6, _item(plan, num=True))
        if res is not None:
            st = res.settings
            self.proj_note.setText(f'At {st.speed_kmh:g} km/h turning {st.yaw_rate_dps:g}°/s, points '
                                   + ('on the ground' if st.surface == 'ground' else f'{st.range_m:g} m away')
                                   + f'; limit {st.threshold_px:g} px. Plan: the sync plan\'s own camera–lidar '
                                     f'misalignment, for a pipeline that does not compensate it.')

    def _fill_findings(self):
        while self.findings_box.count():
            w = self.findings_box.takeAt(0).widget()
            if w is not None:
                w.hide()
                w.deleteLater()
        res = self.res
        items = [(f['severity'], f['message']) for f in res.findings] if res is not None else []
        if self.topo is not None:
            items += [('note', n) for n in self.topo.notes if res is None]
        for sev, text in items or [('good', 'Run the study to see the findings.')]:
            line = StatusLine(sev if sev != 'note' else 'good', text)
            if sev == 'note':
                line.icon.clear()
            line.text.setProperty('small', True)
            self.findings_box.addWidget(line)

    # ------------------------------------------------------------------ export
    def export(self, folder=None):
        if self.res is None:
            return []
        if folder is None:
            folder = QFileDialog.getExistingDirectory(self, 'Export the clock tree study into')
            if not folder:
                return []
        from ...core import provenance as PV          # where the JSON came from
        paths = PV.stamp_paths(CT.export(self.res, self.ctx.store.project, folder), self.ctx.store.project, folder)
        self.ctx.msg(f'Exported the clock tree study to {folder}', 5000)
        return paths
