"""Analysis ▸ Thermal cameras at night: how far each thermal (LWIR) camera detects, recognises and identifies a
pedestrian, a cyclist and a car on a clear night, in fog, in heavy rain and on a hot summer road (where the contrast
inverts), next to the visible cameras in the light of UN R112 headlamps (low and high beam): which camera sees the
pedestrian first. The study (core.thermal_ir) runs as a background job (about a second) and again by itself after
edits while the page shows."""
from __future__ import annotations

import json
import math
import threading
import traceback

import numpy as np
from PyQt6.QtCore import QObject, QPointF, QRectF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QPainter, QPen, QPolygonF
from PyQt6.QtWidgets import (QAbstractItemView, QComboBox, QFileDialog, QGridLayout, QHBoxLayout, QHeaderView,
                             QProgressBar, QScrollArea, QSplitter, QTableWidget, QTableWidgetItem, QVBoxLayout,
                             QWidget)

from ...core import thermal_ir as TH
from ..charts import ChartBase
from ..theme import STATUS, THEME, mono, sans
from ..widgets import (fit_width, ROW_H, ScrubSpinBox, SegmentedControl, StatusLine, ToggleSwitch, button, hline,
                       label)
from .recording_ws import Tile

JOB = 'thermal'
AUTO_MAX_S = 4.0
COND_COLORS = ['#3987e5', '#9085e9', '#2fb5c7', '#ec835a']
# (key, label, unit, decimals, step, scale, tip): ThermalSettings fields; stored × scale = shown
FIELDS = (
    ('snr_min', 'SNR needed', '', 1, 0.5, 1.0,
     'Contrast-to-noise (thermal) or signal-to-noise (visible) ratio a camera needs to detect the target (Rose\'s '
     'criterion: 5)'),
    ('lateral_m', 'Pedestrian at', 'm', 2, 0.25, 1.0,
     'Where the target stands: this far to the left of each camera\'s axis (negative: to the right; −1.5 m is UN '
     'R112\'s 50R/75R line, the kerb side of right-hand traffic)'),
    ('skin_share', 'Exposed skin', '%', 0, 1.0, 100.0,
     'Share of the pedestrian\'s silhouette that is bare skin (face, hands): it is warmer than the clothing'),
    ('max_range_m', 'Ranges to', 'm', 0, 50.0, 1.0, 'The farthest range the study looks at'),
)
# (key, label, unit, decimals, step, tip): ThermalCondition fields of the shown condition
COND_FIELDS = (
    ('t_air_c', 'Air', '°C', 1, 1.0, 'Air temperature (the far scene is taken at it)'),
    ('rh_pct', 'Humidity', '%', 0, 5.0, 'Relative humidity: water vapour absorbs in the LWIR'),
    ('t_road_c', 'Road', '°C', 1, 1.0, 'Road surface temperature: the background below the camera\'s height'),
    ('visibility_m', 'Fog', 'm', 0, 10.0, 'Fog visibility (meteorological optical range); 0 = no fog'),
    ('rain_mm_h', 'Rain', 'mm/h', 1, 1.0, 'Rain rate'),
    ('clo', 'Clothing', 'clo', 2, 0.1, 'The pedestrian\'s clothing insulation (1 clo: a suit; 0.5: summer clothes)'),
)
WINDOW_ITEMS = [(k, v[0]) for k, v in TH.WINDOWS.items() if k != 'glass']


def _num(text, tip='', color=None, bold=False):
    it = QTableWidgetItem(text)
    it.setFont(mono(12, QFont.Weight.DemiBold if bold else None))
    it.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    it.setFlags(it.flags() & ~Qt.ItemFlag.ItemIsEditable)
    if tip:
        it.setToolTip(tip)
    if color is not None:
        it.setForeground(color)
    return it


def _text(text, tip='', color=None):
    it = QTableWidgetItem(text)
    it.setFlags(it.flags() & ~Qt.ItemFlag.ItemIsEditable)
    if tip:
        it.setToolTip(tip)
    if color is not None:
        it.setForeground(color)
    return it


def _table(cols):
    t = QTableWidget(0, len(cols))
    t.setHorizontalHeaderLabels(cols)
    t.verticalHeader().setVisible(False)
    t.verticalHeader().setDefaultSectionSize(ROW_H)
    t.horizontalHeader().setFixedHeight(ROW_H)
    t.setShowGrid(False)
    t.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    t.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
    return t


def row_keys(res) -> list:
    """(tag, beam, modality, slot) per chart row: a thermal camera once, a visible one per beam."""
    out = []
    for c in res.cams:
        for b in ([''] if c['modality'] == 'lwir' else list(TH.BEAMS)):
            out.append((c['tag'], b, c['modality'], c['slot']))
    return out


def row_label(tag, beam, mod) -> str:
    return f'{tag} · ' + ('thermal' if mod == 'lwir' else f'{beam} beam')


# =========================================================================== charts
class RangeBars(ChartBase):
    """Per camera (and beam) one bar per condition: the range for the chosen class and level; hover for the figure
    and what limits it."""
    BAR, GAP, GROUP_GAP, LEFT = 9, 2, 8, 112

    def __init__(self, parent=None):
        super().__init__(parent)
        self.title = 'Range per camera and condition'
        self.res = None
        self.cls, self.level = 'pedestrian', 'detection'
        self._hits = []
        self.setMinimumHeight(160)

    def set_data(self, res, cls, level):
        self.res, self.cls, self.level = res, cls, level
        n = len(row_keys(res)) if res is not None else 0
        k = len(res.conditions) if res is not None else 0
        self.setMinimumHeight(max(160, 46 + n * (k * (self.BAR + self.GAP) + self.GROUP_GAP) + 22))
        self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        res = self.res
        self.paint_title(p, f'{TH.CLASS_LABEL.get(self.cls, self.cls)} · {self.level}')
        self._hits = []
        if res is None or not res.rows:
            p.setFont(sans(11))
            p.setPen(THEME.color('muted'))
            p.drawText(QRectF(0, 24, self.width(), 20), Qt.AlignmentFlag.AlignVCenter, 'No camera to study.')
            p.end()
            return
        vals = [r['range_m'] for r in res.rows if r['class'] == self.cls and r['level'] == self.level]
        vmax = max(vals + [10.0])
        from ..charts import nice_ticks, tick_label
        ticks = nice_ticks(vmax, 5)
        top = ticks[-1]
        x0, x1 = self.LEFT, self.width() - 34
        y = 26
        # legend
        p.setFont(sans(11))
        lx = x0
        for i, c in enumerate(res.conditions):
            p.fillRect(QRectF(lx, y + 4, 8, 8), QColor(COND_COLORS[i % len(COND_COLORS)]))
            p.setPen(THEME.color('ink2'))
            p.drawText(QRectF(lx + 12, y, 200, 16), Qt.AlignmentFlag.AlignVCenter, c.name)
            lx += 12 + p.fontMetrics().horizontalAdvance(c.name) + 16
        y += 22
        y_top = y
        keys = row_keys(res)
        group = len(res.conditions) * (self.BAR + self.GAP) + self.GROUP_GAP
        y_bot = y_top + len(keys) * group
        p.setFont(mono(10))
        for t in ticks:
            x = x0 + (x1 - x0) * t / top
            p.setPen(QPen(THEME.color('hair'), 1))
            p.drawLine(QPointF(x, y_top - 2), QPointF(x, y_bot))
            p.setPen(THEME.color('muted'))
            p.drawText(QRectF(x - 30, y_bot + 2, 60, 14), Qt.AlignmentFlag.AlignCenter, tick_label(t))
        p.drawText(QRectF(x1 + 4, y_bot + 2, 30, 14), Qt.AlignmentFlag.AlignLeft, 'm')
        for gi, (tag, beam, mod, slot) in enumerate(keys):
            gy = y_top + gi * group
            p.setFont(sans(11))
            p.fillRect(QRectF(0, gy + group / 2 - 6, 8, 8), THEME.cam(slot))
            p.setPen(THEME.color('ink'))
            p.drawText(QRectF(12, gy, x0 - 16, group - self.GROUP_GAP), Qt.AlignmentFlag.AlignVCenter,
                       row_label(tag, beam, mod))
            for ci, c in enumerate(res.conditions):
                r = res.row(tag, c.id, self.cls, self.level, beam)
                v = r['range_m'] if r else 0.0
                by = gy + ci * (self.BAR + self.GAP)
                w = max((x1 - x0) * v / top, 0.0)
                col = QColor(COND_COLORS[ci % len(COND_COLORS)])
                key = (gi, ci)
                if self.hover is not None and self.hover != key:
                    col.setAlphaF(0.45)
                rect = QRectF(x0, by, w, self.BAR)
                if w >= 1:
                    p.fillPath(self.bar_path(rect, 2.0, 'right'), col)
                self._hits.append((QRectF(x0, by - 1, max(w, 24), self.BAR + 2), key,
                                   (QPointF(x0 + w, by), f'{v:,.0f} m',
                                    f'{row_label(tag, beam, mod)} · {c.name}',
                                    f'limited by {r["limited_by"]}' if r else '')))
        self.paint_tip(p)
        p.end()

    def mouseMoveEvent(self, e):
        pos = e.position()
        hit = next((h for h in self._hits if h[0].contains(pos)), None)
        key = hit[1] if hit else None
        if key != self.hover:
            self.hover = key
            if hit:
                self.set_tip(*hit[2])
            else:
                self.tip = None
            self.update()


class SnrPlot(ChartBase):
    """The pedestrian's contrast-to-noise ratio (thermal) or signal-to-noise ratio (visible) against range, log-log,
    for the shown condition, with the SNR needed as a line."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.title = 'Pedestrian contrast-to-noise against range'
        self.res, self.cond, self.beam = None, '', 'low'
        self.setMinimumHeight(220)

    def set_data(self, res, cond, beam='low'):
        self.res, self.cond, self.beam = res, cond, beam
        self.update()

    def _plot(self):
        return QRectF(46, 40, self.width() - 60, self.height() - 66)

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.paint_title(p, f'visible: {self.beam} beam · log scales')
        res = self.res
        R = self._plot()
        if res is None or not res.curves:
            p.end()
            return
        xs = np.log10(res.ranges)
        xa, xb = float(xs[0]), float(xs[-1])
        ya, yb = -1.0, 4.0

        def m(x, y):
            return QPointF(R.left() + (x - xa) / (xb - xa) * R.width(),
                           R.bottom() - (min(max(y, ya), yb) - ya) / (yb - ya) * R.height())
        p.setFont(mono(10))
        for k in range(int(ya), int(yb) + 1):
            q = m(xa, k)
            p.setPen(QPen(THEME.color('hair'), 1))
            p.drawLine(q, QPointF(R.right(), q.y()))
            p.setPen(THEME.color('muted'))
            p.drawText(QRectF(0, q.y() - 7, R.left() - 6, 14), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                       f'{10 ** k:g}')
        for k in range(0, int(xb) + 1):
            for mm in (1, 2, 5):
                v = math.log10(mm * 10 ** k)
                if xa <= v <= xb:
                    q = m(v, ya)
                    p.setPen(QPen(THEME.color('hair'), 1))
                    p.drawLine(QPointF(q.x(), R.top()), q)
                    p.setPen(THEME.color('muted'))
                    p.drawText(QRectF(q.x() - 20, R.bottom() + 2, 40, 12), Qt.AlignmentFlag.AlignCenter, f'{mm * 10 ** k:g}')
        p.setFont(sans(10))
        p.drawText(QRectF(R.left(), R.bottom() + 12, R.width(), 12), Qt.AlignmentFlag.AlignRight, 'range, m')
        yl = m(xa, math.log10(res.settings.snr_min)).y()
        pen = QPen(QColor(STATUS['critical']), 1, Qt.PenStyle.CustomDashLine)
        pen.setDashPattern([3, 3])
        p.setPen(pen)
        p.drawLine(QPointF(R.left(), yl), QPointF(R.right(), yl))
        slots = {c['tag']: c['slot'] for c in res.cams}
        lx = R.left()
        p.setClipRect(R.adjusted(-1, -1, 1, 1))
        legend = []
        for (tag, cid, beam), d in sorted(res.curves.items()):
            if cid != self.cond or (beam and beam != self.beam):
                continue
            y = np.log10(np.maximum(np.asarray(d['snr'], float), 10 ** (ya - 1)))
            col = THEME.cam(slots.get(tag, 0))
            pen = QPen(col, 2.0 if beam == '' else 1.4)
            if beam:
                pen.setStyle(Qt.PenStyle.CustomDashLine)
                pen.setDashPattern([4, 2])
            p.setPen(pen)
            p.drawPolyline(QPolygonF([m(a, b) for a, b in zip(xs, y)]))
            legend.append((tag + (' thermal' if beam == '' else ' visible'), col))
        p.setClipping(False)
        p.setFont(sans(10))
        for text, col in legend:
            p.fillRect(QRectF(lx, 24, 8, 8), col)
            p.setPen(THEME.color('ink2'))
            p.drawText(QRectF(lx + 11, 20, 140, 14), Qt.AlignmentFlag.AlignVCenter, text)
            lx += 11 + p.fontMetrics().horizontalAdvance(text) + 12
        p.end()


# =========================================================================== the controller
class _Sig(QObject):
    done = pyqtSignal(int, object)
    failed = pyqtSignal(int, str)


def inputs_key(project) -> str:
    v = project.variant
    return json.dumps({'variant': v.to_dict(), 'thermal': project.analysis.thermal,
                       'weather': project.analysis.weather, 'materials': project.analysis.materials,
                       'library': project.section('library'), 'mesh': id(project.vehicle.mesh)},
                      sort_keys=True, default=str)


# =========================================================================== the page
class ThermalPage(QWidget):
    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        self.result: TH.ThermalResult | None = None
        self.busy = False
        self.stale = False
        self.status_text = ''
        self._gen = 0
        self._for = None
        self._shown_cond = TH.COND_IDS[0]
        self._win_boxes = {}
        self.sig = _Sig()
        self.sig.done.connect(self._done)
        self.sig.failed.connect(self._failed)
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(300)
        self.timer.timeout.connect(lambda: self.run(quiet=True))
        self.setObjectName('base')
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        split = QSplitter(Qt.Orientation.Horizontal)
        split.setHandleWidth(2)
        split.addWidget(self._left())
        split.addWidget(self._right())
        split.setSizes([340, 1260])
        lay.addWidget(split)
        ctx.store.changed.connect(self._on_edit)
        ctx.store.project_replaced.connect(self._reset)
        THEME.changed.connect(lambda *_: self._show())

    # ------------------------------------------------------------------ settings panel
    def _left(self):
        w = QWidget()
        w.setObjectName('panel')
        ll = QVBoxLayout(w)
        ll.setContentsMargins(8, 6, 8, 8)
        ll.setSpacing(4)
        top = QHBoxLayout()
        top.addWidget(label('Thermal cameras at night'))
        top.addStretch(1)
        top.addWidget(label('LWIR · R112 headlamps', muted=True, small=True))
        ll.addLayout(top)
        ll.addWidget(hline())
        ll.addWidget(label('How far each thermal camera detects a pedestrian, a cyclist and a car at night, in fog, '
                           'in rain and on a hot road, next to the visible cameras in the headlamps\' light. Typical '
                           'models with cited figures; ranges are what the camera resolves, an upper bound for a '
                           'detector.', muted=True, small=True, wrap=True))
        ll.addSpacing(6)
        ll.addWidget(label('Conditions', bold=True))
        self.cond_toggles = {}
        for cid in TH.COND_IDS:
            t = ToggleSwitch(TH.COND[cid].name)
            t.setToolTip(TH.COND_NOTES[cid])
            t.clicked.connect(lambda on, c=cid: self._toggle_cond(c, on))
            ll.addWidget(t)
            self.cond_toggles[cid] = t
        ll.addSpacing(4)
        ll.addWidget(label('Shown condition', ink2=True))
        self.cond_pick = SegmentedControl([(c, TH.COND[c].name.split()[0]) for c in TH.COND_IDS], TH.COND_IDS[0])
        self.cond_pick.setToolTip('The condition the tiles, the curves and the fields below show and edit')
        self.cond_pick.changed.connect(self._pick_cond)
        ll.addWidget(self.cond_pick)
        grid = QGridLayout()
        grid.setVerticalSpacing(2)
        grid.setHorizontalSpacing(6)
        grid.setColumnStretch(2, 1)
        self.cond_fields = {}
        for key, text, unit, dec, step, tip in COND_FIELDS:
            f = ScrubSpinBox('', 'plain', dec, unit, step=step, width=110)
            f.setToolTip(tip)
            f.committed.connect(lambda v, k=key, t=text, u=unit: self._set_cond(k, v, f'{t} {v:g} {u}'))
            a = label(text, ink2=True)
            a.setFixedWidth(118)
            a.setToolTip(tip)
            i = grid.rowCount()
            grid.addWidget(a, i, 0)
            grid.addWidget(f, i, 1)
            self.cond_fields[key] = f
        ll.addLayout(grid)
        self.fog_kind = SegmentedControl([(k, 'Radiation fog' if k == 'radiation' else 'Advection fog')
                                          for k in TH.FOGS], 'radiation')
        self.fog_kind.setToolTip('Radiation fog (small drops, clear nights inland): the LWIR passes it far better '
                                 'than light. Advection fog (large drops, sea fog): about as opaque as to light')
        self.fog_kind.changed.connect(lambda k: self._set_cond('fog', k, f'{k} fog'))
        ll.addWidget(self.fog_kind)
        ll.addSpacing(6)
        ll.addWidget(label('Targets and criteria', bold=True))
        self.crit = SegmentedControl([(k, 'DRI pixels' if k == 'dri' else 'Johnson 1958') for k in TH.CRITERIA],
                                     'dri')
        self.crit.setToolTip('\n'.join(TH.CRITERIA_NOTE.values()))
        self.crit.changed.connect(lambda k: self._set(f'{TH.CRITERIA_LABEL[k]}', criteria=k))
        ll.addWidget(self.crit)
        self.cls_pick = SegmentedControl([(k, TH.CLASS_LABEL[k]) for k in TH.CLASSES], 'pedestrian')
        self.cls_pick.setToolTip(TH.CLASS_NOTE)
        self.cls_pick.changed.connect(lambda _: self._show())
        ll.addWidget(self.cls_pick)
        self.lvl_pick = SegmentedControl([(k, k.capitalize()) for k in TH.LEVELS], 'detection')
        self.lvl_pick.changed.connect(lambda _: self._show())
        ll.addWidget(self.lvl_pick)
        grid = QGridLayout()
        grid.setVerticalSpacing(2)
        grid.setHorizontalSpacing(6)
        grid.setColumnStretch(2, 1)
        self.fields = {}
        for key, text, unit, dec, step, scale, tip in FIELDS:
            f = ScrubSpinBox('', 'plain', dec, unit, step=step, width=110)
            f.setToolTip(tip)
            f.committed.connect(lambda v, k=key, s=scale, t=text, u=unit: self._set(f'{t} {v:g} {u}', **{k: v / s}))
            a = label(text, ink2=True)
            a.setFixedWidth(118)
            a.setToolTip(tip)
            i = grid.rowCount()
            grid.addWidget(a, i, 0)
            grid.addWidget(f, i, 1)
            self.fields[key] = (f, scale)
        ll.addLayout(grid)
        ll.addSpacing(6)
        ll.addWidget(label('Visible cameras at night', bold=True))
        self.beam_model = SegmentedControl([(k, TH.BEAM_MODEL_LABEL[k]) for k in TH.BEAM_MODELS], 'typical')
        self.beam_model.setToolTip('Typical lamp: core.weather\'s R112-shaped pattern (it passes R112). R112 '
                                   'minimum: that pattern dimmed until its weakest test point sits at the '
                                   'regulation\'s minimum, the dimmest lamp R112 approves')
        self.beam_model.changed.connect(lambda k: self._set(TH.BEAM_MODEL_LABEL[k], beam_model=k))
        ll.addWidget(self.beam_model)
        self.clothing = SegmentedControl([(k, TH.CLOTHING_LABEL[k]) for k in TH.CLOTHING], 'medium')
        self.clothing.setToolTip('The pedestrian\'s clothing by its Munsell value: reflectance 4.5 %, 19 % and 58 % '
                                 '(ASTM D1535)')
        self.clothing.changed.connect(lambda k: self._set(f'{TH.CLOTHING_LABEL[k]} clothing', clothing=k))
        ll.addWidget(self.clothing)
        ll.addSpacing(6)
        ll.addWidget(label('Thermal cameras', bold=True))
        self.win_host = QWidget()
        self.win_lay = QGridLayout(self.win_host)
        self.win_lay.setContentsMargins(0, 0, 0, 0)
        self.win_lay.setVerticalSpacing(2)
        ll.addWidget(self.win_host)
        self.b_add = button('Add a thermal camera', None, 'plus',
                            'A Teledyne FLIR Boson+ 640 with its 32° shutterless lens on the front bumper, level, '
                            'looking ahead (one undo step)')
        self.b_add.clicked.connect(self.add_thermal)
        ll.addWidget(self.b_add)
        ll.addStretch(1)
        self.bar = QProgressBar()
        self.bar.setRange(0, 0)
        self.bar.setFixedHeight(3)
        self.bar.setTextVisible(False)
        self.bar.hide()
        ll.addWidget(self.bar)
        self.status = label('', muted=True, small=True, wrap=True)
        ll.addWidget(self.status)
        bl = QHBoxLayout()
        b = self.b_defaults = button('Defaults', 'ghost', None, 'Back to the default thermal settings')
        b.clicked.connect(self._defaults)
        bl.addWidget(b)
        self.b_run = button('Run', None, 'play', 'Compute the study again now')
        self.b_run.clicked.connect(lambda: self.run())
        bl.addWidget(self.b_run)
        bl.addStretch(1)
        self.b_export = button('Export…', None, 'export', 'The ranges, curves, R112 table and findings as JSON, CSV '
                                                          'and PNG')
        self.b_export.clicked.connect(lambda: self.export())
        bl.addWidget(self.b_export)
        ll.addLayout(bl)
        sc = QScrollArea()
        sc.setWidgetResizable(True)
        sc.setWidget(w)
        fit_width(sc)
        sc.setMinimumWidth(320)
        return sc

    def _right(self):
        right = QSplitter(Qt.Orientation.Vertical)
        right.setHandleWidth(2)
        tw = QWidget()
        tw.setObjectName('panel')
        tl = QVBoxLayout(tw)
        tl.setContentsMargins(10, 8, 10, 8)
        tl.setSpacing(8)
        tiles = QHBoxLayout()
        tiles.setSpacing(8)
        self.tiles = {}
        for key, caption, tip in (
                ('thermal', 'Thermal', 'The farthest a forward thermal camera detects the class under the shown '
                                       'condition'),
                ('low', 'Visible, low beam', 'The farthest a forward visible camera detects it in the low beam'),
                ('high', 'Visible, high beam', 'The same in the high beam'),
                ('dT', 'Apparent ΔT', 'The pedestrian\'s radiance difference against its background at 0 m, as a '
                                      'temperature difference (negative: colder than the background)')):
            t = self.tiles[key] = Tile(caption, tip)
            tiles.addWidget(t, 1)
        tl.addLayout(tiles)
        charts = QHBoxLayout()
        self.bars = RangeBars()
        self.snr = SnrPlot()
        sc = QScrollArea()
        sc.setWidgetResizable(True)
        sc.setWidget(self.bars)
        charts.addWidget(sc, 1)
        charts.addWidget(self.snr, 1)
        tl.addLayout(charts, 1)
        tl.addWidget(label('Range per camera (m) · the class and level chosen on the left', bold=True))
        self.table = _table(['Camera', 'Module'] + [TH.COND[c].name for c in TH.COND_IDS])
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        tl.addWidget(self.table, 1)
        right.addWidget(tw)
        bw = QWidget()
        bw.setObjectName('panel')
        bl = QHBoxLayout(bw)
        bl.setContentsMargins(10, 6, 10, 8)
        col = QVBoxLayout()
        col.addWidget(label('UN R112 class B test points (per headlamp)', bold=True))
        self.r112 = _table(['Beam', 'Point', 'h °', 'v °', 'cd', 'At', 'lx'])
        self.r112.setToolTip(TH.R112_SOURCE + '. lx: the illuminance the point\'s intensity puts on a surface at the '
                                              'distance it is named for (E = I/d²)')
        self.r112.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        col.addWidget(self.r112, 1)
        bl.addLayout(col, 1)
        col = QVBoxLayout()
        col.addWidget(label('Findings', bold=True))
        self.find_host = QWidget()
        self.find_lay = QVBoxLayout(self.find_host)
        self.find_lay.setContentsMargins(0, 0, 0, 0)
        self.find_lay.setSpacing(4)
        self.find_lay.addStretch(1)
        sc = QScrollArea()
        sc.setWidgetResizable(True)
        sc.setWidget(self.find_host)
        col.addWidget(sc, 1)
        bl.addLayout(col, 2)
        self.findings: list[StatusLine] = []
        right.addWidget(bw)
        right.setSizes([640, 240])
        return right

    # ------------------------------------------------------------------ settings
    @property
    def settings(self) -> TH.ThermalSettings:
        return TH.settings_of(self.ctx.store.project)

    def _store(self, st: TH.ThermalSettings, text) -> bool:
        new = TH.ThermalSettings.from_dict(st.to_dict()).to_dict()
        if new == self.settings.to_dict():
            self._sync()
            return False

        def fn(p):
            p.analysis.thermal = new
        self.ctx.store.edit('analysis', fn, f'Thermal cameras · {text}')
        return True

    def _set(self, text, **kw):
        st = self.settings
        for k, v in kw.items():
            setattr(st, k, v)
        return self._store(st, text)

    def _toggle_cond(self, cid, on):
        st = self.settings
        keep = set(st.conditions) | {cid} if on else set(st.conditions) - {cid}
        st.conditions = [c for c in TH.COND_IDS if c in keep]
        if not st.conditions:
            self._sync()                       # one condition at least
            return False
        return self._store(st, f'{TH.COND[cid].name} {"on" if on else "off"}')

    def _pick_cond(self, cid):
        self._shown_cond = cid
        self._sync()
        self._show()

    def _set_cond(self, key, value, text):
        st = self.settings
        cid = self._shown_cond
        p = dict(st.params.get(cid, {}))
        p[key] = value
        st.params = dict(st.params, **{cid: p})
        return self._store(st, f'{TH.COND[cid].name}: {text}')

    def _set_window(self, uid, tag, key):
        st = self.settings
        w = dict(st.windows)
        if key == 'none':
            w.pop(uid, None)
        else:
            w[uid] = key
        st.windows = w
        return self._store(st, f'{tag} behind {TH.WINDOWS[key][0].lower()}')

    def _defaults(self):
        if self.ctx.store.project.analysis.thermal:
            def fn(p):
                p.analysis.thermal = {}
            self.ctx.store.edit('analysis', fn, 'Thermal cameras · defaults')

    def add_thermal(self):
        """A Boson+ 640 on the front bumper (core.thermal_ir.add_thermal_camera) as one undo step."""
        p = self.ctx.store.project
        from ...core.project import MAX_CAMERAS
        if len(p.variant.cameras) >= MAX_CAMERAS:
            self.ctx.msg(f'Up to {MAX_CAMERAS} cameras per variant.')
            return None
        eng = self.ctx.analysis.engine if getattr(self.ctx, 'analysis', None) is not None else None

        def fn(q):
            return TH.add_thermal_camera(q, engine=eng).uid
        uid = self.ctx.store.edit(['variant', 'library'], fn, 'Added a thermal camera (Boson+ 640, front bumper)')
        self.ctx.msg('Added a Teledyne FLIR Boson+ 640 on the front bumper', 4000)
        return uid

    def _sync(self):
        st = self.settings
        d = st.to_dict()
        for key, (f, scale) in self.fields.items():
            f.set_value(d[key] * scale)
        for cid, t in self.cond_toggles.items():
            t.blockSignals(True)
            t.setChecked(cid in st.conditions)
            t.blockSignals(False)
        c = st.condition(self._shown_cond)
        for key, f in self.cond_fields.items():
            f.set_value(getattr(c, key))
        self.fog_kind.set_value(c.fog)
        self.cond_pick.set_value(self._shown_cond)
        self.crit.set_value(st.criteria)
        self.beam_model.set_value(st.beam_model)
        self.clothing.set_value(st.clothing)
        self._sync_windows(st)

    def _sync_windows(self, st):
        p = self.ctx.store.project
        cams = [c for c in p.variant.cameras if c.visible and TH.is_thermal(p.module_for(c))]
        want = [(c.uid, c.tag) for c in cams]
        if [(u, t) for u, (t, _) in self._win_boxes.items()] != want:
            for _, (_, (a, box)) in list(self._win_boxes.items()):
                for w_ in (a, box):
                    self.win_lay.removeWidget(w_)
                    w_.hide()
                    w_.deleteLater()
            self._win_boxes = {}
            if not cams:
                a = label('No thermal camera in this variant.', muted=True, small=True)
                box = QWidget()
                self.win_lay.addWidget(a, 0, 0, 1, 2)
                self._win_boxes[''] = ('', (a, box))
            for i, c in enumerate(cams):
                a = label(f'{c.tag} window', ink2=True)
                a.setFixedWidth(118)
                box = QComboBox()
                for k, text in WINDOW_ITEMS:
                    box.addItem(text, k)
                box.setToolTip('What the thermal camera looks through. A camera behind the windscreen or a side '
                               'window sees nothing (glass is opaque from 8 to 14 µm) unless the body has a '
                               'germanium or ZnSe window cut in front of it')
                box.currentIndexChanged.connect(
                    lambda _i, b=box, u=c.uid, t=c.tag: self._set_window(u, t, b.currentData()))
                self.win_lay.addWidget(a, i, 0)
                self.win_lay.addWidget(box, i, 1)
                self._win_boxes[c.uid] = (c.tag, (a, box))
        for uid, (tag, (_, box)) in self._win_boxes.items():
            if isinstance(box, QComboBox):
                k = st.windows.get(uid, 'none')
                box.blockSignals(True)
                box.setCurrentIndex(max(0, box.findData(k)))
                box.blockSignals(False)

    # ------------------------------------------------------------------ running
    def _on_edit(self, *_):
        if self._for is not None and inputs_key(self.ctx.store.project) == self._for[1]:
            return
        if self.result is not None:
            self.stale = True
        if self.isVisible():
            self._sync()
            if self._quick():
                self.timer.start()
            self._show_status()

    def _reset(self, *_):
        self._gen += 1
        self.result, self._for, self.stale, self.busy, self.status_text = None, None, False, False, ''
        if self.isVisible():
            self._sync()
            self.timer.start()
        self._show()

    def _quick(self):
        return self.result is None or self.result.metrics.get('seconds', 99) < AUTO_MAX_S

    def showEvent(self, e):
        super().showEvent(e)
        self.activate()

    def activate(self):
        self._sync()
        if (self.result is None or self.stale or self._for is None
                or self._for[0] is not self.ctx.store.project) and not self.busy:
            self.timer.start()
        self._show()

    def run(self, quiet=False) -> bool:
        ctx = self.ctx
        p = ctx.store.project
        win = ctx.window
        if self.busy:
            if quiet or (win is not None and JOB in getattr(win, '_jobs', {})):
                return False
            self._gen += 1
        project = p.snapshot()
        key = (p, inputs_key(p))
        eng = win._shared_engine() if win is not None and hasattr(win, '_shared_engine') else \
            ctx.analysis.engine.share()
        self._gen += 1
        gen = self._gen
        self.busy = True
        self.status_text = 'Computing the thermal study…'
        self._show_status()

        def work(progress, cancel):
            return key, TH.run(project, eng, progress=progress, cancel=cancel)
        if not quiet and win is not None and hasattr(win, 'run_job'):
            return win.run_job(JOB, 'Thermal cameras', work, lambda out: self._done(gen, out),
                               lambda m: self._failed(gen, m))

        def thread():
            try:
                out = work(lambda *_: None, lambda: gen != self._gen)
                self.sig.done.emit(gen, out)
            except RuntimeError:
                pass
            except Exception as e:
                try:
                    self.sig.failed.emit(gen, f'{type(e).__name__}: {e}\n{traceback.format_exc()}')
                except RuntimeError:
                    pass
        threading.Thread(target=thread, daemon=True).start()
        return True

    def _done(self, gen, out):
        if gen != self._gen:
            return
        self.busy = False
        key, res = out if out is not None else (None, None)
        if res is None or key[0] is not self.ctx.store.project:
            self.status_text = 'Cancelled' if res is None else ''
            self._show_status()
            return
        self.result, self._for = res, key
        self.stale = key[1] != inputs_key(self.ctx.store.project)
        m = res.metrics
        self.status_text = (f'{res.variant}: {m["cameras"]} cameras ({m["thermal_cameras"]} thermal) × '
                            f'{m["conditions"]} conditions · {m["seconds"]:.1f} s')
        self._show()
        if self.stale and self.isVisible() and self._quick():
            self.timer.start()

    def _failed(self, gen, msg):
        if gen != self._gen:
            return
        self.busy = False
        self.status_text = 'Failed: ' + msg.splitlines()[0]
        self._show_status()

    # ------------------------------------------------------------------ showing
    def _show_status(self):
        self.bar.setVisible(self.busy)
        self.b_export.setEnabled(self.result is not None)
        self.status.setText(self.status_text + (' · stale: the design or the settings changed'
                                                if self.stale and self.result is not None else ''))

    def _show(self):
        self._show_status()
        res = self.result
        cls, lvl = self.cls_pick.value() or 'pedestrian', self.lvl_pick.value() or 'detection'
        cid = self._shown_cond
        if res is not None and cid not in {c.id for c in res.conditions} and res.conditions:
            cid = res.conditions[0].id
        self.bars.set_data(res, cls, lvl)
        self.snr.set_data(res, cid, 'low')
        self.snr.title = f'Pedestrian contrast-to-noise: {TH.COND[cid].name.lower()}'
        if res is None:
            for t in self.tiles.values():
                t.set('—')
            self.table.setRowCount(0)
            self.r112.setRowCount(0)
            self._set_findings([])
            return
        fwd = {c['tag'] for c in res.cams if c['forward']}

        def best(lwir, beam):
            rs = [r for r in res.rows if r['condition'] == cid and r['class'] == cls and r['level'] == lvl
                  and r['tag'] in fwd and (r['modality'] == 'lwir') == lwir and r['beam'] == beam]
            return max(rs, key=lambda r: r['range_m'], default=None)
        th, lo, hi = best(True, ''), best(False, 'low'), best(False, 'high')
        win = max((r for r in (th, lo) if r is not None), key=lambda r: r['range_m'], default=None)
        for key, r in (('thermal', th), ('low', lo), ('high', hi)):
            if r is None:
                self.tiles[key].set('—', '', 'no forward camera of this kind')
            else:
                sev = None if key == 'high' else ('good' if r is win else 'warning')
                self.tiles[key].set(f'{r["range_m"]:,.0f}', 'm', f'{r["tag"]} · limited by {r["limited_by"]}', sev)
        dts = res.scenes.get(cid, {}).get('pedestrian_dT0_k', {})
        if dts:
            tag, dT = max(dts.items(), key=lambda kv: abs(kv[1]))
            self.tiles['dT'].set(f'{dT:+.2f}', 'K', f'{tag} · ' + ('inverted: colder than the background' if dT < 0
                                                                   else 'warmer than the background'),
                                 'warning' if dT < 0 else None)
        else:
            self.tiles['dT'].set('—', '', 'no thermal camera')
        # the range table
        keys = row_keys(res)
        t = self.table
        t.setRowCount(len(keys))
        conds = {c.id: c for c in res.conditions}
        for i, (tag, beam, mod, slot) in enumerate(keys):
            cam = res.cam(tag)
            it = _text(row_label(tag, beam, mod))
            it.setData(Qt.ItemDataRole.DecorationRole, THEME.cam(slot))
            t.setItem(i, 0, it)
            need = cam.get('needed') or []
            t.setItem(i, 1, _text(f'{cam["module_name"]} · {cam["lens_name"]}',
                                  ('Not stated on the datasheet: ' + ', '.join(need)) if need else ''))
            for j, c in enumerate(TH.COND_IDS, 2):
                if c not in conds:
                    t.setItem(i, j, _text('off', color=THEME.color('muted')))
                    continue
                r = res.row(tag, c, cls, lvl, beam)
                if r is None:
                    t.setItem(i, j, _text('—'))
                    continue
                col = QColor(STATUS['critical']) if r['limited_by'] == 'glass' else None
                t.setItem(i, j, _num(f'{r["range_m"]:,.0f}', f'{row_label(tag, beam, mod)}, {conds[c].label()}: '
                                     f'{r["range_m"]:.1f} m, limited by {r["limited_by"]}', col))
        # R112
        rows = res.r112
        t = self.r112
        t.setRowCount(len(rows))
        for i, r in enumerate(rows):
            t.setItem(i, 0, _text('low (passing)' if r['beam'] == 'low' else 'high (driving)'))
            t.setItem(i, 1, _text(r['point']))
            t.setItem(i, 2, _num('' if r['h_deg'] is None else f'{r["h_deg"]:+.2f}'))
            t.setItem(i, 3, _num('' if r['v_deg'] is None else f'{r["v_deg"]:+.2f}'))
            t.setItem(i, 4, _num(f'{"≥" if r["limit"] == "min" else "≤"} {r["cd"]:,.0f}'))
            t.setItem(i, 5, _num('' if r['range_m'] is None else f'{r["range_m"]:g} m'))
            t.setItem(i, 6, _num('' if r['lux_per_lamp'] is None else f'{r["lux_per_lamp"]:.2f}'))
        self._set_findings(res.findings)

    def _set_findings(self, findings):
        for w in self.findings:
            self.find_lay.removeWidget(w)
            w.hide()
            w.deleteLater()
        self.findings = []
        for f in findings:
            w = StatusLine(f['severity'], f['message'])
            self.find_lay.insertWidget(self.find_lay.count() - 1, w)
            self.findings.append(w)

    # ------------------------------------------------------------------ export
    def export(self, folder=None):
        res = self.result
        if res is None:
            return []
        if folder is None:
            folder = QFileDialog.getExistingDirectory(self, 'Export the thermal study into')
            if not folder:
                return []
        from ...core import provenance as PV
        paths = PV.stamp_paths(TH.export(res, self.ctx.store.project, folder), self.ctx.store.project, folder)
        self.ctx.msg(f'Exported the thermal study to {folder}', 5000)
        return paths
