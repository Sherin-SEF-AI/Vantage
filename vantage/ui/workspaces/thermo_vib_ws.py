"""Analysis ▸ Thermal and vibration drift: how the midday sun warps the rig and how the road shakes it.

Left: the sun and heat inputs, the calibration tolerance, the road and the events. Centre: each camera's and sensor's
drift from its calibration over the day (turn, move, or the pairs' drift in pixels), and per device the acceleration
PSD at its mount and the transfer functions. Right: the devices (first mode, shaking, blur, worst drift), the pairs and
the findings. The study (core.thermo_vib.run) takes a fraction of a second: it runs in a background thread after every
edit while the page shows, with a progress bar and Cancel."""
from __future__ import annotations

import math
import threading
import traceback

import numpy as np
from PyQt6.QtCore import QObject, QPointF, QRectF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import (QAbstractItemView, QComboBox, QFileDialog, QHBoxLayout, QHeaderView, QLineEdit,
                             QProgressBar, QScrollArea, QSplitter, QTableWidget, QTableWidgetItem, QTabWidget,
                             QVBoxLayout, QWidget)

from ...core import thermo_vib as TV
from ..charts import ChartBase, nice_ticks, tick_label
from ..theme import STATUS, THEME, mono, sans
from ..widgets import ROW_H, ScrubSpinBox, Section, SegmentedControl, StatusLine, button, hline, label

AUTO_DELAY_MS = 300
# key, label, decimals, unit, scrub step, tooltip
HEAT_FIELDS = (
    ('t_cal_c', 'Calibrated at', 1, '°C', 1.0, 'Temperature of the garage the cameras are calibrated in: every part '
                                               'at it is the reference'),
    ('t_max_c', 'Peak air', 1, '°C', 1.0, 'The day\'s highest air temperature (in full sun)'),
    ('daily_range_c', 'Daily range', 1, 'K', 1.0, 'The peak minus the night\'s low (typical 10-15 K on a clear day)'),
    ('peak_hour', 'Peak at', 1, 'h', 0.5, 'Local time of the peak air temperature (typical 15:00)'),
    ('air_kmh', 'Airflow', 0, 'km/h', 5.0, 'Air speed over the rig while driving: the convection that cools it '
                                           '(0 = parked)'),
)
TOL_FIELDS = (
    ('tol_deg', 'Angle', 3, '°', 0.1, 'Angle tolerance of a calibration (Custom)'),
    ('tol_mm', 'Position', 1, 'mm', 1.0, 'Position tolerance of a calibration (Custom)'),
    ('px_limit', 'Pair limit', 2, 'px', 0.1, 'The most relative drift of a camera-lidar or stereo pair in the '
                                             'camera\'s image (typical 1 px)'),
    ('range_m', 'Parallax at', 0, 'm', 1.0, 'A pair\'s relative move is turned into pixels at this range'),
)
ROAD_FIELDS = (
    ('speed_kmh', 'Speed', 0, 'km/h', 5.0, 'Driving speed on the road class'),
    ('zeta_body', 'Body damping', 2, 'ζ', 0.05, 'Suspension damping as a share of critical (typical 0.2-0.4)'),
    ('zeta_mount', 'Metal mounts', 3, 'ζ', 0.005, 'Damping of bolted aluminium mounts and rig tubes (typical '
                                                 '0.01-0.05)'),
    ('zeta_printed', 'Printed', 3, 'ζ', 0.005, 'Damping of a printed camera bracket (typical 0.03-0.08)'),
)
EVENT_FIELDS = (
    ('bump_h_mm', 'Bump height', 0, 'mm', 5.0, 'Speed bump height (a half-sine)'),
    ('bump_len_m', 'Bump length', 2, 'm', 0.1, 'Speed bump length along the road'),
    ('bump_kmh', 'Bump speed', 0, 'km/h', 5.0, 'Speed over the bump'),
    ('joint_mm', 'Joint step', 0, 'mm', 5.0, 'Expansion joint as a sharp step (at the road speed)'),
    ('cobble_kmh', 'Cobble speed', 0, 'km/h', 5.0, 'Speed on cobblestones'),
    ('sett_m', 'Sett pitch', 2, 'm', 0.01, 'Length of one sett along the road: the cobblestones\' beat V / pitch'),
)
FIELD_LABEL = {k: t for k, t, *_ in HEAT_FIELDS + TOL_FIELDS + ROAD_FIELDS + EVENT_FIELDS}
FIELD_LABEL.update(material='rig material', finish='finish', roof_paint='roof paint', feet='rack feet',
                   tol_source='tolerance', road_class='road class', cobble_class='cobblestone class', date='date',
                   heading_deg='heading')
DEV_COLS = (('Device', 'The camera or sensor, and what it rides on'),
            ('f1 Hz', 'First mode of its softest mount (bracket, mast or crossbar)'),
            ('Shakes', 'Why it resonates: in the body band, the wheel-hop band, at the excitation peak or the '
                       'cobblestones\' beat'),
            ('Flex °', 'RMS angle of its mount\'s flex against the body on the road class at the speed'),
            ('Blur px', 'RMS motion blur of the body\'s and the mount\'s shaking in one exposure (core.timing\'s)'),
            ('Drift °', 'Worst thermal turn from the calibration over the day'),
            ('Drift mm', 'Worst thermal move from the calibration over the day'),
            ('Bump °', 'Peak flex angle over the speed bump'))
PAIR_COLS = (('Pair', 'Devices that share a calibration'),
             ('Drift px', 'Worst relative drift over the day in the camera\'s image (rotation over the image plus the '
                          'parallax of the relative move)'),
             ('Drift °', 'The relative turn at that hour'),
             ('At', 'The hour of the worst drift'),
             ('Vib px', 'RMS relative flex of their mounts on the road, in the camera\'s image'),
             ('Baseline mm', 'RMS change of the distance between them on the road (stereo baseline flex)'))
METRICS = (('angle', 'Turn °'), ('move', 'Move mm'), ('pairs', 'Pairs px'))


class _Sig(QObject):
    done = pyqtSignal(int, object)
    failed = pyqtSignal(int, str)
    progress = pyqtSignal(int, float)


def device_color(d, i=0) -> QColor:
    """A camera's own colour; sensors in ink tones (lidar strong, others muted)."""
    if d.get('slot') is not None:
        return THEME.cam(d['slot'])
    return THEME.color('ink') if d['kind'] == 'lidar' else THEME.color('ink2' if i % 2 else 'muted')


# =========================================================================== a line chart with log axes
class LineChart(ChartBase):
    """Lines over a linear or logarithmic x and y, shaded x bands, horizontal and vertical marker lines, and a hover
    tooltip on the nearest point."""

    def __init__(self, title='', log_x=False, log_y=False, x_fmt=None, unit='', parent=None):
        super().__init__(parent)
        self.title = title
        self.subtitle = unit
        self.log_x, self.log_y = log_x, log_y
        self.x_fmt = x_fmt or tick_label
        self.series, self.bands, self.hlines, self.vlines = [], [], [], []
        self.empty = 'No data'
        self.setMinimumHeight(170)

    def set_data(self, series, bands=(), hlines=(), vlines=(), empty='No data'):
        """series: [{'name', 'x', 'y', 'color' (QColor), 'dash' (bool), 'width'}]; bands: [(x0, x1, text)]; hlines:
        [(y, text, QColor)]; vlines: [(x, text, QColor)]."""
        self.series = [s for s in series if len(s['x'])]
        self.bands, self.hlines, self.vlines = list(bands), list(hlines), list(vlines)
        self.empty = empty
        self.tip = None
        self.update()

    def _ranges(self):
        xs = np.concatenate([np.asarray(s['x'], float) for s in self.series]) if self.series else np.array([0, 1.0])
        ys = np.concatenate([np.asarray(s['y'], float) for s in self.series]) if self.series else np.array([0, 1.0])
        ys = np.r_[ys, [h[0] for h in self.hlines]]
        if self.log_x:
            xs = xs[xs > 0]
        x0, x1 = (float(xs.min()), float(xs.max())) if len(xs) else (0.0, 1.0)
        if x1 <= x0:
            x1 = x0 + 1.0
        if self.log_y:
            pos = ys[np.isfinite(ys) & (ys > 0)]
            hi = float(pos.max()) if len(pos) else 1.0
            lo = max(float(pos.min()) if len(pos) else hi * 1e-3, hi * 1e-8)
            y0, y1 = 10 ** math.floor(math.log10(lo)), 10 ** math.ceil(math.log10(hi))
            if y1 <= y0:
                y1 = y0 * 10
        else:
            fin = ys[np.isfinite(ys)]
            top = float(fin.max()) if len(fin) else 1.0
            ticks = nice_ticks(max(top, 1e-9))
            y0, y1 = 0.0, ticks[-1]
        return x0, x1, y0, y1

    def _plot(self):
        return QRectF(56, 24, max(self.width() - 66, 10), max(self.height() - 48, 10))

    def _map(self, r, rng, x, y):
        x0, x1, y0, y1 = rng
        if self.log_x:
            fx = (math.log10(max(x, 1e-30)) - math.log10(x0)) / (math.log10(x1) - math.log10(x0))
        else:
            fx = (x - x0) / (x1 - x0)
        if self.log_y:
            fy = (math.log10(max(y, y0 * 1e-3)) - math.log10(y0)) / (math.log10(y1) - math.log10(y0))
        else:
            fy = (y - y0) / (y1 - y0)
        return QPointF(r.left() + fx * r.width(), r.bottom() - min(max(fy, -0.02), 1.02) * r.height())

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.paint_title(p)
        r = self._plot()
        if not self.series:
            p.setFont(sans(11))
            p.setPen(THEME.color('muted'))
            p.drawText(r, Qt.AlignmentFlag.AlignCenter, self.empty)
            p.end()
            return
        rng = self._ranges()
        x0, x1, y0, y1 = rng
        # bands
        for a, b, text in self.bands:
            pa, pb = self._map(r, rng, max(a, x0), y0), self._map(r, rng, min(b, x1), y0)
            if pb.x() <= r.left() or pa.x() >= r.right():
                continue
            p.fillRect(QRectF(pa.x(), r.top(), pb.x() - pa.x(), r.height()), THEME.color('sel'))
            p.setFont(sans(10))
            p.setPen(THEME.color('muted'))
            p.drawText(QRectF(pa.x(), r.top() + 1, max(pb.x() - pa.x(), 40), 14), Qt.AlignmentFlag.AlignLeft, text)
        # y grid
        p.setFont(mono(10))
        if self.log_y:
            ticks = [10 ** k for k in range(int(round(math.log10(y0))), int(round(math.log10(y1))) + 1)]
            step = max(1, len(ticks) // 5)
            ticks = ticks[::step]
            fmt = (lambda v: f'1e{int(round(math.log10(v)))}')
        else:
            ticks = nice_ticks(y1)
            fmt = tick_label
        for v in ticks:
            y = self._map(r, rng, x0, v).y()
            p.setPen(QPen(THEME.color('hair'), 1))
            p.drawLine(QPointF(r.left(), y), QPointF(r.right(), y))
            p.setPen(THEME.color('muted'))
            p.drawText(QRectF(0, y - 7, r.left() - 6, 14), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                       fmt(v))
        # x ticks
        if self.log_x:
            xt = [10 ** k for k in range(math.ceil(math.log10(x0) - 1e-9), math.floor(math.log10(x1) + 1e-9) + 1)]
        else:                                          # whole units (hours), about six ticks
            step = max(1, math.ceil((x1 - x0) / 6))
            xt = [float(v) for v in range(math.ceil(x0), math.floor(x1) + 1, step)] or [x0, x1]
        for v in xt:
            x = self._map(r, rng, v, y0).x()
            p.setPen(QPen(THEME.color('hair'), 1))
            p.drawLine(QPointF(x, r.top()), QPointF(x, r.bottom()))
            p.setPen(THEME.color('muted'))
            p.drawText(QRectF(x - 30, r.bottom() + 3, 60, 14), Qt.AlignmentFlag.AlignCenter, self.x_fmt(v))
        p.setPen(QPen(THEME.color('strong'), 1))
        p.drawLine(r.bottomLeft(), r.bottomRight())
        # markers
        for k, (v, text, col) in enumerate(sorted(self.vlines, key=lambda m: m[0])):
            if not (x0 <= v <= x1):
                continue
            x = self._map(r, rng, v, y0).x()
            p.setPen(QPen(col, 1, Qt.PenStyle.DashLine))
            p.drawLine(QPointF(x, r.top()), QPointF(x, r.bottom()))
            p.setFont(sans(10))
            w = 130
            left = x + w + 3 > r.right()               # near the right edge: the label goes left of the line
            p.drawText(QRectF(x - w - 3 if left else x + 3, r.top() + 14 + 14 * k, w, 14),
                       Qt.AlignmentFlag.AlignRight if left else Qt.AlignmentFlag.AlignLeft, text)
        for v, text, col in self.hlines:
            if not (y0 <= v <= y1):
                continue
            y = self._map(r, rng, x0, v).y()
            p.setPen(QPen(col, 1, Qt.PenStyle.DashLine))
            p.drawLine(QPointF(r.left(), y), QPointF(r.right(), y))
            p.setFont(sans(10))
            p.drawText(QRectF(r.right() - 160, y - 15, 158, 14), Qt.AlignmentFlag.AlignRight, text)
        # lines
        p.save()
        p.setClipRect(r.adjusted(-2, -2, 2, 2))
        for s in self.series:
            path = QPainterPath()
            first = True
            for x, y in zip(s['x'], s['y']):
                if not (math.isfinite(x) and math.isfinite(y)) or (self.log_y and y <= 0) or (self.log_x and x <= 0):
                    first = True
                    continue
                pt = self._map(r, rng, x, y)
                if first:
                    path.moveTo(pt)
                    first = False
                else:
                    path.lineTo(pt)
            pen = QPen(s['color'], s.get('width', 1.6))
            if s.get('dash'):
                pen.setStyle(Qt.PenStyle.DashLine)
            if self.hover is not None and self.hover[0] is s:
                pen.setWidthF(pen.widthF() + 1.2)
            p.setPen(pen)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawPath(path)
        p.restore()
        if self.hover is not None:
            s, i = self.hover
            pt = self._map(r, rng, s['x'][i], s['y'][i])
            p.setPen(QPen(THEME.color('ink'), 1))
            p.setBrush(s['color'])
            p.drawEllipse(pt, 3.5, 3.5)
        self.paint_tip(p)
        p.end()

    def mouseMoveEvent(self, e):
        if not self.series:
            return
        r = self._plot()
        rng = self._ranges()
        pos = e.position()
        best, bd = None, 18.0 ** 2
        for s in self.series:
            for i, (x, y) in enumerate(zip(s['x'], s['y'])):
                if not (math.isfinite(x) and math.isfinite(y)) or (self.log_y and y <= 0):
                    continue
                pt = self._map(r, rng, x, y)
                d = (pt.x() - pos.x()) ** 2 + (pt.y() - pos.y()) ** 2
                if d < bd:
                    best, bd = (s, i), d
        self.hover = best
        if best is None:
            self.tip = None
        else:
            s, i = best
            y = s['y'][i]
            val = f'{y:.3g}' if self.log_y or abs(y) < 0.01 else f'{y:.3f}'
            self.set_tip(self._map(r, rng, s['x'][i], y), f'{val} {self.subtitle}'.strip(), s['name'],
                         f'at {self.x_fmt(s["x"][i])}')
        self.update()


def _table(cols):
    t = QTableWidget(0, len(cols))
    t.verticalHeader().setVisible(False)
    t.verticalHeader().setDefaultSectionSize(ROW_H)
    t.horizontalHeader().setFixedHeight(ROW_H)
    t.setShowGrid(False)
    t.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    t.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    t.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
    for c, (head, tip) in enumerate(cols):
        it = QTableWidgetItem(head)
        it.setToolTip(tip)
        t.setHorizontalHeaderItem(c, it)
    hh = t.horizontalHeader()
    hh.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
    for c in range(1, len(cols)):
        hh.setSectionResizeMode(c, QHeaderView.ResizeMode.ResizeToContents)
    return t


def _num(text, tip='', color=None):
    it = QTableWidgetItem(text)
    it.setFont(mono(12))
    it.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    if tip:
        it.setToolTip(tip)
    if color is not None:
        it.setForeground(color)
    return it


def _f(v, dp=2):
    return '—' if v is None or not math.isfinite(v) else f'{v:.{dp}f}'


# =========================================================================== the page
class ThermoVibPage(QWidget):
    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        self.result: TV.ThermoVibResult | None = None
        self.status_text = ''
        self.busy = False
        self.stale = False
        self._gen = 0
        self._for = None
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
        split.setSizes([300, 720, 620])
        lay.addWidget(split)
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(AUTO_DELAY_MS)
        self.timer.timeout.connect(lambda: self.run(quiet=True))
        ctx.store.changed.connect(self._on_store)
        ctx.store.project_replaced.connect(self._replaced)
        THEME.changed.connect(self._retheme)
        self._sync_fields()
        self.refresh()

    # ------------------------------------------------------------------ settings panel
    def _spin(self, key, dec, unit, step, tip):
        f = ScrubSpinBox('', 'plain', dec, unit, step=step, width=96)
        f.setToolTip(tip)
        f.committed.connect(lambda v, k=key: self.set_option(k, v))
        self.fields[key] = f
        return f

    def _combo(self, key, options, tip):
        c = QComboBox()
        c.setToolTip(tip)
        for k, text in options:
            c.addItem(text, k)
        c.activated.connect(lambda i, k=key, c=c: self.set_option(k, c.itemData(i)))
        self.combos[key] = c
        return c

    def _left(self):
        outer = QWidget()
        outer.setObjectName('panel')
        outer.setMinimumWidth(280)
        ol = QVBoxLayout(outer)
        ol.setContentsMargins(0, 0, 0, 0)
        ol.setSpacing(0)
        head = QWidget()
        hl = QVBoxLayout(head)
        hl.setContentsMargins(8, 6, 8, 4)
        hl.setSpacing(4)
        top = QHBoxLayout()
        top.addWidget(label('Thermal and vibration drift'))
        top.addStretch(1)
        hl.addLayout(top)
        hl.addWidget(hline())
        hl.addWidget(label('The midday sun warps the rig and the road shakes it. Every camera and sensor is checked '
                           'against its calibration from the garage: hour by hour in full sun, on the road class at '
                           'speed, and over a bump.', muted=True, small=True, wrap=True))
        ol.addWidget(head)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        body = QWidget()
        body.setObjectName('panel')
        bl = QVBoxLayout(body)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.setSpacing(0)
        self.fields, self.combos, self.segs = {}, {}, {}
        # ---- sun and heat
        s = Section('Sun and heat')
        self.f_date = QLineEdit()
        self.f_date.setPlaceholderText('auto: the highest sun')
        self.f_date.setToolTip('The day (YYYY-MM-DD). Empty: the day of the exposure campaign with the highest noon '
                               'sun, at the exposure study\'s location and driving window')
        self.f_date.editingFinished.connect(lambda: self.set_option('date', self.f_date.text().strip()))
        s.row('Day', self.f_date)
        heads = [(None, 'Worst of 12')] + [(float(h), f'{h}° {_compass(h)}') for h in range(0, 360, 30)]
        self.c_heading = QComboBox()
        self.c_heading.setToolTip('The vehicle\'s heading, clockwise from north: it turns the sun round the rig. '
                                  'Worst of 12: the heading with the largest drift')
        for k, text in heads:
            self.c_heading.addItem(text, k)
        self.c_heading.activated.connect(lambda i: self.set_option('heading_deg', self.c_heading.itemData(i)))
        s.row('Heading', self.c_heading)
        for key, text, dec, unit, step, tip in HEAT_FIELDS:
            s.row(text, self._spin(key, dec, unit, step, tip), tip=tip)
        s.row('Rig', self._combo('material', [(k, TV.MATERIALS[k]['label']) for k in TV.RIG_MATERIALS],
                                 'Material of the rig parts (masts, crossbars, enclosures)'))
        s.row('Finish', self._combo('finish', [(k, v['label']) for k, v in TV.FINISHES.items()],
                                    'Surface of the rig parts: how much sun they absorb'))
        s.row('Roof paint', self._combo('roof_paint', [(k, v['label']) for k, v in TV.ROOF_PAINTS.items()],
                                        'The car\'s paint on the roof the rack stands on'))
        seg = SegmentedControl([('sliding', 'Sliding'), ('clamped', 'Clamped')])
        seg.setToolTip('The rack\'s feet: one slides along the roof (the bar expands freely), or both are clamped (the '
                       'bar\'s length follows the steel roof and the difference becomes stress)')
        seg.changed.connect(lambda k: self.set_option('feet', k))
        self.segs['feet'] = seg
        s.row('Rack feet', seg)
        bl.addWidget(s)
        # ---- tolerance
        s = Section('Calibration tolerance')
        seg = SegmentedControl([('install', 'Install'), ('observability', 'Observ.'), ('custom', 'Custom')])
        seg.setToolTip('What the drift is checked against: the installation check\'s 1° / 10 mm, the calibration '
                       'observability study\'s requirement, or the figures below')
        seg.changed.connect(lambda k: self.set_option('tol_source', k))
        self.segs['tol_source'] = seg
        s.row('Against', seg)
        for key, text, dec, unit, step, tip in TOL_FIELDS:
            s.row(text, self._spin(key, dec, unit, step, tip), tip=tip)
        self.tol_note = label('', muted=True, small=True, wrap=True)
        s.add(self.tol_note)
        bl.addWidget(s)
        # ---- road
        s = Section('Road')
        seg = SegmentedControl([(c, c) for c in TV.ISO_CLASSES])
        seg.setToolTip('ISO 8608 road class: A very good … C average … E very poor (H the roughest)')
        seg.changed.connect(lambda k: self.set_option('road_class', k))
        self.segs['road_class'] = seg
        s.row('Class', seg)
        for key, text, dec, unit, step, tip in ROAD_FIELDS:
            s.row(text, self._spin(key, dec, unit, step, tip), tip=tip)
        bl.addWidget(s)
        # ---- events
        s = Section('Events', expanded=False)
        for key, text, dec, unit, step, tip in EVENT_FIELDS:
            s.row(text, self._spin(key, dec, unit, step, tip), tip=tip)
        s.row('Cobble class', self._combo('cobble_class', [(c, TV.ISO_LABEL[c]) for c in TV.ISO_CLASSES],
                                          'The cobblestones as an ISO 8608 class (typical D)'))
        bl.addWidget(s)
        bl.addStretch(1)
        scroll.setWidget(body)
        ol.addWidget(scroll, 1)
        foot = QWidget()
        fl = QVBoxLayout(foot)
        fl.setContentsMargins(8, 4, 8, 8)
        fl.setSpacing(4)
        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.setTextVisible(False)
        self.bar.setFixedHeight(4)
        fl.addWidget(self.bar)
        self.status = label('', muted=True, small=True, wrap=True)
        fl.addWidget(self.status)
        row = QHBoxLayout()
        self.b_run = button('Run', 'primary', 'play', 'Run the study now')
        self.b_run.clicked.connect(lambda: self.run())
        self.b_cancel = button('Cancel', None, 'stop', 'Stop the running study')
        self.b_cancel.clicked.connect(self.cancel)
        self.b_export = button('Export…', None, 'export', 'The study as JSON (vehicle frame, mm and degrees), the '
                                                          'hour-by-hour drift, pairs, vibration and PSD as CSV and a '
                                                          'PNG')
        self.b_export.clicked.connect(lambda: self.export())
        row.addWidget(self.b_run)
        row.addWidget(self.b_cancel)
        row.addStretch(1)
        row.addWidget(self.b_export)
        fl.addLayout(row)
        ol.addWidget(foot)
        return outer

    def _centre(self):
        w = QWidget()
        w.setObjectName('panel')
        cl = QVBoxLayout(w)
        cl.setContentsMargins(8, 4, 8, 6)
        cl.setSpacing(4)
        self.head = label('', ink2=True, wrap=True)
        cl.addWidget(self.head)
        self.tabs = QTabWidget()
        # over the day
        day = QWidget()
        dl = QVBoxLayout(day)
        dl.setContentsMargins(0, 4, 0, 0)
        dl.setSpacing(4)
        row = QHBoxLayout()
        self.metric = SegmentedControl(METRICS, 'angle')
        self.metric.changed.connect(lambda k: self._fill_day())
        row.addWidget(self.metric)
        row.addStretch(1)
        self.day_note = label('', muted=True, small=True)
        row.addWidget(self.day_note)
        dl.addLayout(row)
        self.day_chart = LineChart('Drift from the calibration', x_fmt=_hh)
        dl.addWidget(self.day_chart, 2)
        self.temp_chart = LineChart('Temperatures', x_fmt=_hh, unit='°C')
        dl.addWidget(self.temp_chart, 1)
        self.tabs.addTab(day, 'Over the day')
        # vibration
        vib = QWidget()
        vl = QVBoxLayout(vib)
        vl.setContentsMargins(0, 4, 0, 0)
        vl.setSpacing(4)
        row = QHBoxLayout()
        row.addWidget(label('Device', ink2=True))
        self.c_dev = QComboBox()
        self.c_dev.setToolTip('The camera or sensor whose mount the plots show')
        self.c_dev.activated.connect(lambda _: self._fill_vib())
        row.addWidget(self.c_dev, 1)
        self.vib_note = label('', muted=True, small=True)
        row.addWidget(self.vib_note)
        vl.addLayout(row)
        self.psd_chart = LineChart('Acceleration PSD at the mount', log_x=True, log_y=True, x_fmt=_hz,
                                   unit='(m/s²)²/Hz')
        vl.addWidget(self.psd_chart, 1)
        self.tf_chart = LineChart('Transfer functions', log_x=True, log_y=True, x_fmt=_hz, unit='×')
        vl.addWidget(self.tf_chart, 1)
        self.tabs.addTab(vib, 'Vibration')
        cl.addWidget(self.tabs, 1)
        return w

    def _right(self):
        right = QSplitter(Qt.Orientation.Vertical)
        right.setHandleWidth(2)
        tw = QWidget()
        tw.setObjectName('panel')
        tl = QVBoxLayout(tw)
        tl.setContentsMargins(8, 6, 8, 6)
        tl.setSpacing(4)
        tl.addWidget(label('Cameras and sensors', bold=True))
        self.dev_table = _table(DEV_COLS)
        self.dev_table.itemSelectionChanged.connect(self._dev_selected)
        tl.addWidget(self.dev_table, 1)
        tl.addWidget(label('Pairs that share a calibration', bold=True))
        self.pair_table = _table(PAIR_COLS)
        tl.addWidget(self.pair_table, 1)
        right.addWidget(tw)
        fw = QWidget()
        fw.setObjectName('panel')
        fl = QVBoxLayout(fw)
        fl.setContentsMargins(8, 6, 8, 6)
        fl.setSpacing(3)
        fl.addWidget(label('Findings', bold=True))
        sc = QScrollArea()
        sc.setWidgetResizable(True)
        sc.setFrameShape(QScrollArea.Shape.NoFrame)
        box = QWidget()
        self.find_box = QVBoxLayout(box)
        self.find_box.setContentsMargins(0, 0, 0, 0)
        self.find_box.setSpacing(4)
        self.find_box.addStretch(1)
        sc.setWidget(box)
        fl.addWidget(sc, 1)
        right.addWidget(fw)
        right.setSizes([560, 300])
        return right

    # ------------------------------------------------------------------ settings
    @property
    def settings(self) -> TV.ThermoVibSettings:
        return TV.settings_of(self.ctx.store.project)

    def set_option(self, key, value) -> bool:
        """Change one setting, saved with the project as one undo step; False when nothing changes."""
        st = self.settings
        old = st.to_dict()
        d = dict(old)
        d[key] = value
        new = TV.ThermoVibSettings.from_dict(d).to_dict()
        if new == old:
            self._sync_fields()              # a refused or clamped value: show what is stored
            return False
        shown = new[key]
        if key == 'heading_deg':
            shown = 'worst of 12' if shown is None else f'{shown:g}°'
        elif key == 'date':
            shown = shown or 'auto'
        elif isinstance(shown, float):
            shown = f'{shown:g}'

        def fn(p):
            p.analysis.thermo_vib = new
        self.ctx.store.edit('analysis', fn, f'Thermal and vibration · {FIELD_LABEL.get(key, key)} {shown}')
        return True

    def _sync_fields(self):
        st = self.settings
        self._syncing = True
        try:
            for key, f in self.fields.items():
                f.set_value(float(getattr(st, key)))
            for key, c in self.combos.items():
                i = c.findData(getattr(st, key))
                if i >= 0 and c.currentIndex() != i:
                    c.setCurrentIndex(i)
            for key, sg in self.segs.items():
                sg.set_value(getattr(st, key))
            if not self.f_date.hasFocus():
                self.f_date.setText(st.date)
            i = next((k for k in range(self.c_heading.count()) if self.c_heading.itemData(k) == st.heading_deg), 0)
            self.c_heading.setCurrentIndex(i)
            custom = st.tol_source == 'custom'
            for k in ('tol_deg', 'tol_mm'):
                self.fields[k].setEnabled(custom)
            tol = TV.tolerance(self.ctx.store.project, st)
            self.tol_note.setText(f'Checked against {tol["label"]}.')
        finally:
            self._syncing = False

    # ------------------------------------------------------------------ running
    def _on_store(self, *_):
        self._sync_fields()
        if self.result is not None or self.isVisible():
            if self._for is None or TV.inputs_key(self.ctx.store.project) != self._for[1]:
                self.stale = True
                self.refresh_status()
                if self.isVisible():
                    self.timer.start()

    def _replaced(self, *_):
        self._gen += 1
        self.result, self._for, self.stale, self.busy = None, None, False, False
        self._sync_fields()
        self.refresh()
        if self.isVisible():
            self.timer.start()

    def showEvent(self, e):
        super().showEvent(e)
        if self.result is None or self.stale:
            self.timer.start()

    def activate(self):
        if self.result is None or self.stale:
            self.run(quiet=True)

    def run(self, quiet=False) -> bool:
        """Start the study in a background thread (a running one is replaced)."""
        p = self.ctx.store.project
        project = p.snapshot()
        key = (p, TV.inputs_key(p))
        self._gen += 1
        gen = self._gen
        self.busy = True
        self.status_text = 'Working out the sun, the rig\'s temperatures and the road…'
        self.bar.setValue(0)
        self.refresh_status()

        def thread():
            try:
                res = TV.run(project, progress=lambda f, *_: self.sig.progress.emit(gen, f),
                             cancel=lambda: gen != self._gen)
                self.sig.done.emit(gen, (key, res))
            except RuntimeError:              # the window closed while computing
                pass
            except Exception as e:
                try:
                    self.sig.failed.emit(gen, f'{type(e).__name__}: {e}\n{traceback.format_exc()}')
                except RuntimeError:
                    pass
        threading.Thread(target=thread, daemon=True).start()
        return True

    def cancel(self):
        if self.busy:
            self._gen += 1
            self.busy = False
            self.status_text = 'Cancelled'
            self.refresh_status()

    def _done(self, gen, out):
        if gen != self._gen:
            return
        self.busy = False
        key, res = out
        if res is None or key[0] is not self.ctx.store.project:
            self.status_text = 'Cancelled' if res is None else ''
            self.refresh_status()
            return
        self.result, self._for = res, key
        self.stale = key[1] != TV.inputs_key(self.ctx.store.project)
        bad = sum(f['severity'] in TV.ISSUE_SEVERITIES for f in res.findings)
        self.status_text = (f'{res.variant}: {len(res.devices)} devices, {len(res.pairs)} pairs · '
                            + (f'{bad} finding{"s" * (bad != 1)} to look at' if bad else 'nothing to flag')
                            + f' · {res.seconds:.2f} s')
        self.bar.setValue(1000)
        self.refresh()
        if self.stale and self.isVisible():
            self.timer.start()

    def _failed(self, gen, msg):
        if gen != self._gen:
            return
        self.busy = False
        self.status_text = 'Failed: ' + msg.splitlines()[0]
        self.refresh_status()

    # ------------------------------------------------------------------ views
    def _retheme(self, *_):
        self.refresh()

    def refresh_status(self):
        txt = self.status_text
        if self.stale and not self.busy and self.result is not None:
            txt = (txt + ' · ' if txt else '') + 'out of date'
        self.status.setText(txt)
        self.b_cancel.setEnabled(self.busy)
        self.b_export.setEnabled(self.result is not None)

    def refresh(self):
        self.refresh_status()
        res = self.result
        if res is None:
            self.head.setText('Run the study to see the drift over the day and the shaking on the road.')
            self.day_chart.set_data([], empty='No result yet')
            self.temp_chart.set_data([], empty='')
            self.psd_chart.set_data([], empty='No result yet')
            self.tf_chart.set_data([], empty='')
            self.dev_table.setRowCount(0)
            self.pair_table.setRowCount(0)
            self._fill_findings([])
            return
        st = res.settings
        hd = f'{res.heading_deg:.0f}° {_compass(res.heading_deg)}' + (' (the worst)' if res.heading_auto else '')
        modes = res.vehicle['modes']
        self.head.setText(f'{res.day["date"]}' + (' (highest sun)' if res.day['auto_date'] else '') +
                          f', heading {hd}, calibrated at {st.t_cal_c:g} °C · body {modes["bounce_hz"]:.2f} / '
                          f'{modes["pitch_hz"]:.2f} Hz, wheel hop {modes["hop_hz"][0]:.1f} Hz · class '
                          f'{st.road_class} at {st.speed_kmh:g} km/h ({res.vehicle["road_rms_mm"]:.0f} mm RMS road)')
        self._fill_day()
        cur = self.c_dev.currentData()
        self.c_dev.blockSignals(True)
        self.c_dev.clear()
        for d in res.devices:
            self.c_dev.addItem(f'{d["tag"]} · {d["name"]} (on {d["on"]})', d['uid'])
        i = self.c_dev.findData(cur)
        self.c_dev.setCurrentIndex(i if i >= 0 else 0)
        self.c_dev.blockSignals(False)
        self._fill_vib()
        self._fill_tables()
        self._fill_findings(res.findings)

    def _fill_day(self):
        res = self.result
        if res is None:
            return
        metric = self.metric.value() or 'angle'
        tol = res.tol
        hrs = [h['hour'] for h in res.hours]
        series = []
        if metric == 'pairs':
            by = {d['uid']: d for d in res.devices}
            for pr in res.pairs:
                ys = [s['px'] if s['px'] is not None else float('nan') for s in pr['thermal']]
                series.append({'name': f'{pr["label"]} ({pr["kind"]})', 'x': hrs, 'y': ys,
                               'color': device_color(by[pr['a']]), 'dash': pr['kind'] == 'stereo'})
            hl = [(tol['px'], f'limit {tol["px"]:g} px', THEME.color('critical'))]
            self.day_chart.subtitle = 'px'
            self.day_chart.title = 'Pairs\' relative drift in the camera\'s image'
        else:
            k = 'turn_deg' if metric == 'angle' else 'move_mm'
            for i, d in enumerate(res.devices):
                series.append({'name': f'{d["tag"]} on {d["on"]}', 'x': hrs, 'y': [s[k] for s in d['thermal']],
                               'color': device_color(d, i), 'dash': d['kind'] != 'camera'})
            lim = tol['deg'] if metric == 'angle' else tol['mm']
            unit = '°' if metric == 'angle' else 'mm'
            hl = [(lim, f'tolerance {lim:g} {unit}', THEME.color('critical'))]
            self.day_chart.subtitle = unit
            self.day_chart.title = 'Turn from the calibration' if metric == 'angle' else 'Move from the calibration'
        top = max([max((y for y in s['y'] if math.isfinite(y)), default=0.0) for s in series] + [0.0])
        if top < hl[0][0] * 0.05:                      # far below the line: do not squash the curves
            hl = []
        self.day_chart.set_data(series, hlines=hl, empty='No camera-lidar or stereo pairs' if metric == 'pairs'
                                else 'No cameras or sensors')
        tser = [{'name': 'Air', 'x': hrs, 'y': [h['t_air_c'] for h in res.hours], 'color': THEME.color('muted')},
                {'name': 'Roof', 'x': hrs, 'y': [h['t_roof_c'] for h in res.hours], 'color': THEME.color('ink2'),
                 'dash': True}]
        for i, pt in enumerate(res.parts):
            tser.append({'name': pt['name'], 'x': hrs, 'y': [h['parts'][pt['uid']]['mean_c'] for h in res.hours],
                         'color': THEME.cam(i + 3)})
        self.temp_chart.set_data(tser)
        hottest = max(res.parts, key=lambda pt: pt['max_mean_c'] or -1e9, default=None)
        self.day_note.setText(f'Roof up to {max(h["t_roof_c"] for h in res.hours):.0f} °C' +
                              (f', {hottest["name"]} {hottest["max_mean_c"]:.0f} °C' if hottest else '') +
                              f' · air {min(h["t_air_c"] for h in res.hours):.0f}-'
                              f'{max(h["t_air_c"] for h in res.hours):.0f} °C')

    def _fill_vib(self):
        res = self.result
        if res is None:
            return
        uid = self.c_dev.currentData()
        d = res.device(uid) if uid else None
        if d is None:
            self.psd_chart.set_data([], empty='No device')
            self.tf_chart.set_data([], empty='')
            return
        f = res.psd['f']
        pd = res.psd['devices'][uid]
        col = device_color(d)
        bands = [(TV.BODY_BAND[0], TV.BODY_BAND[1], 'body'), (TV.HOP_BAND[0], TV.HOP_BAND[1], 'wheel hop')]
        vl = [(m['f1_hz'], f'{m["name"]} {m["f1_hz"]:.0f} Hz', THEME.color('critical' if d['resonance'] else 'ink2'))
              for m in d['modes']]
        ev = res.events.get('cobble', {})
        if ev.get('sett_hz'):
            vl.append((ev['sett_hz'], f'setts {ev["sett_hz"]:.0f} Hz', THEME.color('muted')))
        self.psd_chart.set_data([{'name': f'{d["tag"]} mount, class {res.settings.road_class} at '
                                          f'{res.settings.speed_kmh:g} km/h', 'x': f, 'y': pd['accel'], 'color': col}],
                                bands=bands, vlines=vl)
        self.tf_chart.set_data([{'name': 'Body: mount height per road height', 'x': f, 'y': pd['h_body'],
                                 'color': THEME.color('ink2')},
                                {'name': f'{d["modes"][0]["name"] if d["modes"] else "mount"}: tip per base '
                                         'acceleration', 'x': f, 'y': pd['trans'], 'color': col}],
                               bands=bands, vlines=vl[:1])
        vb = d['vib']
        self.vib_note.setText(f'{vb["accel_rms"]:.2f} m/s² RMS at the mount · flex {vb["flex_deg_rms"]:.4f}° RMS · '
                              f'first mode {d["f1_hz"]:.0f} Hz')

    def _dev_selected(self):
        rows = self.dev_table.selectionModel().selectedRows() if self.dev_table.selectionModel() else []
        if not rows:
            return
        uid = self.dev_table.item(rows[0].row(), 0).data(Qt.ItemDataRole.UserRole)
        i = self.c_dev.findData(uid)
        if i >= 0 and i != self.c_dev.currentIndex():
            self.c_dev.setCurrentIndex(i)
            self._fill_vib()

    def _fill_tables(self):
        res = self.result
        tol = res.tol
        t = self.dev_table
        t.setRowCount(len(res.devices))
        for r, d in enumerate(res.devices):
            it = QTableWidgetItem(f'{d["tag"]} · {d["on"]}')
            it.setForeground(device_color(d, r))
            it.setData(Qt.ItemDataRole.UserRole, d['uid'])
            it.setToolTip('\n'.join(f'{m["name"]}: {m["f1_hz"]:.1f} Hz ({m["source"]})' for m in d['modes']))
            t.setItem(r, 0, it)
            flags = sorted({w for fl in d['resonance'] for w in fl['why']})
            t.setItem(r, 1, _num(_f(d['f1_hz'], 1), color=QColor(STATUS['critical']) if flags else None))
            t.setItem(r, 2, _num(', '.join(flags) if flags else '—',
                                 color=QColor(STATUS['critical']) if flags else None))
            vb = d['vib']
            t.setItem(r, 3, _num(_f(vb['flex_deg_rms'], 4)))
            t.setItem(r, 4, _num(_f(vb['blur_px'], 3) if vb['blur_px'] is not None else '—'))
            w = d['thermal_worst']
            t.setItem(r, 5, _num(_f(w['deg'], 4), color=QColor(STATUS['critical']) if w['deg'] > tol['deg'] else None,
                                 tip=f'at {_hh(w["hour"])}'))
            t.setItem(r, 6, _num(_f(w['mm'], 2), color=QColor(STATUS['critical']) if w['mm'] > tol['mm'] else None,
                                 tip=f'at {_hh(w["hour"])}'))
            bump = (d.get('events') or {}).get('bump')
            t.setItem(r, 7, _num(_f(bump['peak_flex_deg'], 4) if bump else '—'))
        p = self.pair_table
        p.setRowCount(len(res.pairs))
        for r, pr in enumerate(res.pairs):
            it = QTableWidgetItem(f'{pr["label"]} · {pr["kind"]}')
            p.setItem(r, 0, it)
            wk = pr['thermal_worst'] or {}
            px = wk.get('px')
            p.setItem(r, 1, _num(_f(px, 2), color=QColor(STATUS['critical']) if px is not None and px > tol['px']
                                 else None))
            p.setItem(r, 2, _num(_f(wk.get('deg'), 4)))
            p.setItem(r, 3, _num(_hh(wk.get('hour'))))
            vp = pr['vib'].get('px_rms')
            p.setItem(r, 4, _num(_f(vp, 3), color=QColor(STATUS['warning']) if vp is not None and vp > tol['px']
                                 else None))
            p.setItem(r, 5, _num(_f(pr['vib'].get('baseline_mm_rms'), 4)))

    def _fill_findings(self, findings):
        while self.find_box.count() > 1:
            w = self.find_box.takeAt(0).widget()
            if w is not None:
                w.hide()
                w.deleteLater()
        for i, f in enumerate(findings):
            line = StatusLine(f['severity'] if f['severity'] in STATUS else None, f['message'])
            line.text.setProperty('small', True)
            self.find_box.insertWidget(i, line)

    # ------------------------------------------------------------------ export
    def export(self, folder=None):
        res = self.result
        if res is None:
            return []
        if folder is None:
            folder = QFileDialog.getExistingDirectory(self, 'Export the thermal and vibration study into')
            if not folder:
                return []
        from ...core import provenance as PV          # where the JSON came from
        paths = PV.stamp_paths(TV.export(res, self.ctx.store.project, folder), self.ctx.store.project, folder)
        self.ctx.msg(f'Exported the thermal and vibration study to {folder}', 5000)
        return paths


def _hh(h):
    if h is None or not math.isfinite(h):
        return '—'
    return TV._hh(h)


def _hz(v):
    return f'{v:g} Hz' if v >= 1 else f'{v:.2g} Hz'


def _compass(h):
    return ('N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW')[int(((h % 360) + 22.5) // 45) % 8]
