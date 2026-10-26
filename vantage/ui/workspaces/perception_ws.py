"""Analysis ▸ Perception range: how far out the rig resolves cars, pedestrians, cyclists and signs, per direction,
against the ranges the dataset needs; and how the cameras overlap for stereo. The weather condition (core.weather)
applies to the envelopes; every condition is compared in a table with how each lidar and radar degrades."""
from __future__ import annotations

import json
import math
import os
import threading
import traceback

from PyQt6.QtCore import QObject, QRectF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QPainter, QPen
from PyQt6.QtWidgets import (QAbstractItemView, QComboBox, QFileDialog, QGridLayout, QHBoxLayout, QHeaderView,
                             QProgressBar, QSplitter, QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget)

from ...core import perception as PR
from ...core import weather as W
from ...core.mathutil import parse_number
from ..perceptionview import EnvelopeMap, RoseGrid, _envelope, class_color, controller
from ..theme import THEME, STATUS, mono, sans
from ..widgets import ROW_H, ScrubSpinBox, SegmentedControl, button, hline, label

CLASS_COLS = (('name', 'Object', None, 'The object class'),
              ('height', 'Height', 2, 'Height of the object, m'),
              ('width', 'Width', 2, 'Width of the object, m'),
              ('min_px', 'Min px', 0, 'The height in the image, in pixels, your detector needs to find the object'),
              ('base', 'Raised', 2, 'Height of its bottom above the ground, m (a sign on a post)'))
SHORT = {'front': 'Front', 'front-left': 'FL', 'left': 'Left', 'rear-left': 'RL', 'rear': 'Rear', 'rear-right': 'RR',
         'right': 'Right', 'front-right': 'FR'}


WX_FIELDS = (('visibility_m', 'Visibility', 'm', 0, 50.0,
               'Fog or haze: the meteorological optical range (where a black object\'s contrast falls to 5 %); '
               '0 = none'),
              ('rain_mm_h', 'Rain', 'mm/h', 1, 1.0, 'Rain rate (2.5 light, 7.6 heavy, 25 a downpour)'),
              ('snow_mm_h', 'Snow', 'mm/h', 1, 0.5, 'Snowfall as water (1–2.5 moderate)'),
              ('ambient_lux', 'Ambient', 'lx', 1, 10.0,
               'Light on a vertical surface from the sky, the moon or street lights (full moon 0.1–0.3, street '
               'lighting 2–50, an overcast day thousands)'))
SENSOR_COLS = ('Sensor', 'Kind', 'Clear', 'Now', 'Change', 'Note')


def _num_item(text, tone=None, tip=''):
    it = QTableWidgetItem(text)
    it.setFont(mono(12))
    it.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    if tone:
        it.setForeground(QColor(STATUS[tone]) if tone in STATUS else THEME.color(tone))
    if tip:
        it.setToolTip(tip)
    return it


def _m(v, d=0):
    return '∞' if v is None or not math.isfinite(v) else f'{v:.{d}f}'


class WeatherMap(EnvelopeMap):
    """The envelope map, with the clear envelope of the chosen class dashed while a weather condition applies."""

    def paintEvent(self, e):
        super().paintEvent(e)
        pr = self.pr
        w = getattr(pr, 'weather', None) or {}
        ref, cond = w.get('clear_far'), w.get('condition')
        if pr is None or ref is None or cond is None or cond.clear or not pr.classes:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        c, k = self._geom()
        p.setPen(QPen(THEME.color('ink2'), 1.2, Qt.PenStyle.DashLine))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPolygon(_envelope(c, k, ref[min(self.k, len(ref) - 1)], pr.az))
        p.setFont(sans(10, QFont.Weight.DemiBold))
        p.setPen(THEME.color('ink'))
        p.drawText(QRectF(0, self.height() - 36, self.width() - 10, 16), Qt.AlignmentFlag.AlignRight,
                   f'{cond.label()} · dashed grey: clear')
        p.end()


class _WxSig(QObject):
    done = pyqtSignal(int, object)
    failed = pyqtSignal(int, str)


def _table(cols, rows=0):
    t = QTableWidget(rows, cols)
    t.verticalHeader().setVisible(False)
    t.verticalHeader().setDefaultSectionSize(ROW_H)
    t.horizontalHeader().setFixedHeight(ROW_H)
    t.setShowGrid(False)
    t.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
    return t


class PerceptionPage(QWidget):
    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        self.ctl = controller(ctx)
        self.setObjectName('base')
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        split = QSplitter(Qt.Orientation.Horizontal)
        split.setHandleWidth(2)
        split.addWidget(self._left())
        # ---------------- centre: the envelopes around the vehicle
        cw = QWidget()
        cw.setObjectName('panel')
        cl = QVBoxLayout(cw)
        cl.setContentsMargins(8, 4, 8, 6)
        cl.setSpacing(4)
        head = self._head = QHBoxLayout()
        self.pick = SegmentedControl([('0', 'Car')], '0')
        self.pick.changed.connect(lambda k: self.map.set_class(int(k)))
        head.addWidget(self.pick)
        head.addStretch(1)
        self.mode = SegmentedControl([('classes', 'All objects'), ('cameras', 'By camera')], 'classes')
        self.mode.changed.connect(lambda m: self.map.set_mode(m))
        head.addWidget(self.mode)
        cl.addLayout(head)
        self.map = WeatherMap()
        cl.addWidget(self.map, 1)
        split.addWidget(cw)
        # ---------------- right: roses, verdicts, stereo pairs
        right = QSplitter(Qt.Orientation.Vertical)
        right.setHandleWidth(2)
        self.roses = RoseGrid()
        rw = QWidget()
        rw.setObjectName('panel')
        rl = QVBoxLayout(rw)
        rl.setContentsMargins(8, 6, 8, 6)
        rl.addWidget(label('Range per object · required range per sector', bold=True))
        rl.addWidget(self.roses, 1)
        right.addWidget(rw)
        tw = QWidget()
        tw.setObjectName('panel')
        tl = QVBoxLayout(tw)
        tl.setContentsMargins(8, 6, 8, 6)
        tl.setSpacing(4)
        tl.addWidget(label('Worst direction in each sector', bold=True))
        self.verdicts = _table(len(PR.SECTORS) + 1)
        self.verdicts.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.verdicts.setToolTip('The shortest range in the sector at which the rig still resolves the object; green '
                                 'when it reaches the required range')
        tl.addWidget(self.verdicts)
        tl.addWidget(label('Camera pairs that overlap (stereo)', bold=True))
        self.pairs = _table(6)
        self.pairs.setHorizontalHeaderLabels(['Pair', 'Overlap °', 'Baseline m', 'Δz 10 m', 'Δz 20 m', 'Δz 50 m'])
        self.pairs.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.pairs.setToolTip(f'Overlap: the directions in which both cameras see a car {PR.OVERLAP_AT:g} m away. '
                              f'Δz: the depth resolution of the pair as stereo for a {PR.DISPARITY_PX:g} px disparity '
                              f'error, z²·Δd / (f·B), with the lower angular resolution of the two')
        hh = self.pairs.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for c in range(1, 6):
            hh.setSectionResizeMode(c, QHeaderView.ResizeMode.ResizeToContents)
        tl.addWidget(self.pairs, 1)
        right.addWidget(tw)
        right.addWidget(self._weather_panel())
        right.setSizes([330, 360, 380])
        split.addWidget(right)
        split.setSizes([340, 760, 600])
        lay.addWidget(split)
        self.ctl.changed.connect(self.refresh)
        ctx.store.changed.connect(lambda *_: self._sync_fields())
        ctx.store.project_replaced.connect(self._sync_fields)
        ctx.store.changed.connect(self._weather_edited)
        ctx.store.project_replaced.connect(self._weather_reset)
        self._shown = None
        self._syncing = False
        # the weather comparison: its own background run (core.weather.run) while the page shows
        self.wres = None
        self._wgen = 0
        self._wkey = None
        self._wbusy = False
        self._wsig = _WxSig()
        self._wsig.done.connect(self._weather_done)
        self._wsig.failed.connect(self._weather_failed)
        self._wtimer = QTimer(self)
        self._wtimer.setSingleShot(True)
        self._wtimer.setInterval(400)
        self._wtimer.timeout.connect(self.run_weather)
        self._sync_fields()
        self.refresh()

    # ------------------------------------------------------------------ settings panel
    def _left(self):
        w = QWidget()
        w.setObjectName('panel')
        w.setMinimumWidth(340)
        ll = QVBoxLayout(w)
        ll.setContentsMargins(8, 6, 8, 8)
        ll.setSpacing(4)
        top = QHBoxLayout()
        top.addWidget(label('Perception range'))
        top.addStretch(1)
        top.addWidget(label('every camera, every direction', muted=True, small=True))
        ll.addLayout(top)
        ll.addWidget(hline())
        ll.addWidget(label('How far out the rig resolves the objects your dataset needs. An object stands on the ground; '
                           'a camera resolves it when it is in the image, not hidden by the body, and at least the '
                           'minimum pixel height tall through the camera\'s real lens. Set the minimums your detector '
                           'needs; the defaults are typical.', muted=True, small=True, wrap=True))
        ll.addSpacing(4)
        ll.addWidget(label('Objects', bold=True))
        self.classes = _table(len(CLASS_COLS))
        for c, (_, head, _, tip) in enumerate(CLASS_COLS):
            it = QTableWidgetItem(head)
            it.setToolTip(tip)
            self.classes.setHorizontalHeaderItem(c, it)
        self.classes.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.classes.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        hh = self.classes.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for c in range(1, len(CLASS_COLS)):
            hh.setSectionResizeMode(c, QHeaderView.ResizeMode.Fixed)
            hh.resizeSection(c, 52)
        self.classes.itemChanged.connect(self._class_edited)
        ll.addWidget(self.classes)
        row = QHBoxLayout()
        b_add = self.b_add = button('Add', 'ghost', 'plus', 'Add an object class')
        b_add.clicked.connect(self._add_class)
        b_del = button('Remove', 'ghost', 'minus', 'Remove the selected object class')
        b_del.clicked.connect(self._remove_class)
        b_def = button('Defaults', 'ghost', None, 'Back to the default object classes')
        b_def.clicked.connect(lambda: self.ctl.set_classes(PR.DEFAULT_CLASSES, 'Perception range · default objects'))
        for b in (b_add, b_del, b_def):
            row.addWidget(b)
        row.addStretch(1)
        ll.addLayout(row)
        ll.addSpacing(6)
        ll.addWidget(label('Required range per sector', bold=True))
        grid = QGridLayout()
        grid.setVerticalSpacing(2)
        grid.setHorizontalSpacing(6)
        grid.setColumnStretch(4, 1)
        self.req = {}
        for i, (name, _) in enumerate(PR.SECTORS):
            f = ScrubSpinBox('', 'plain', 0, 'm', step=5.0, width=76)
            f.setToolTip(f'The {name} sector must resolve every object out to this range')
            f.committed.connect(lambda v, n=name: self.ctl.set_required(n, v) or self._sync_fields())
            a = label(name.replace('-', ' ').capitalize(), ink2=True)
            a.setFixedWidth(64 if i % 2 == 0 else 86)
            if i % 2:
                a.setContentsMargins(10, 0, 0, 0)
            grid.addWidget(a, i // 2, (i % 2) * 2)
            grid.addWidget(f, i // 2, (i % 2) * 2 + 1)
            self.req[name] = f
        ll.addLayout(grid)
        ll.addSpacing(4)
        mr = QHBoxLayout()
        a = label('Out to', ink2=True)
        a.setFixedWidth(64)
        mr.addWidget(a)
        self.max_range = ScrubSpinBox('', 'plain', 0, 'm', step=10.0, width=76)
        self.max_range.setToolTip('The farthest range analysed')
        self.max_range.committed.connect(lambda v: self.ctl.set_option(max_range=float(v)) or self._sync_fields())
        mr.addWidget(self.max_range)
        mr.addStretch(1)
        ll.addLayout(mr)
        ll.addSpacing(6)
        ll.addLayout(self._weather_fields())
        ll.addStretch(1)
        self.status = label('', muted=True, small=True, wrap=True)
        ll.addWidget(self.status)
        self.bar = QProgressBar()
        self.bar.setTextVisible(False)
        self.bar.setRange(0, 1000)
        self.bar.hide()
        self.ctl.progress.connect(lambda f: self.bar.setValue(int(f * 1000)))
        ll.addWidget(self.bar)
        bl = QHBoxLayout()
        bl.addStretch(1)
        self.b_export = button('Export…', None, 'export', 'The ranges, verdicts and stereo pairs as JSON, and the plan as PNG')
        self.b_export.clicked.connect(lambda: self.export())
        self.b_run = button('Run', 'primary')
        self.b_run.clicked.connect(lambda: self.ctl.run())
        bl.addWidget(self.b_export)
        bl.addWidget(self.b_run)
        ll.addLayout(bl)
        return w

    def _class_edited(self, item):
        if self._syncing:
            return
        rows = [dict(c) for c in self.ctl.settings.classes]      # the stored values, not the rounded text
        r, (key, _, dec, _) = item.row(), CLASS_COLS[item.column()]
        txt = item.text().strip()
        try:
            value = (txt or rows[r]['name']) if dec is None else \
                float(parse_number(txt, 'plain' if key == 'min_px' else 'length'))     # '150 cm' works too
            rows[r][key] = value
        except Exception:                      # parse_number raises on anything that is not a number
            value = None
        what = {'name': 'name', 'min_px': 'minimum pixels', 'base': 'height above ground'}.get(key, key)
        if value is None or not self.ctl.set_classes(rows, f'Perception range · {rows[r]["name"]} {what}'):
            self._sync_fields()                # not a number, or not a usable size: show what is stored

    def _add_class(self):
        rows = self.ctl.settings.classes
        names, n = {c['name'] for c in rows}, 1
        while (name := 'Object' + (f' {n}' if n > 1 else '')) in names:
            n += 1
        rows = rows + [{'name': name, 'height': 1.0, 'width': 0.5, 'min_px': 16.0, 'base': 0.0}]
        if self.ctl.set_classes(rows, f'Perception range · add {name}'):
            self.classes.setCurrentCell(len(rows) - 1, 0)

    def _remove_class(self):
        r = self.classes.currentRow()
        rows = self.ctl.settings.classes
        if 0 <= r < len(rows) and len(rows) > 1:
            name = rows[r]['name']
            self.ctl.set_classes(rows[:r] + rows[r + 1:], f'Perception range · remove {name}')

    def _sync_fields(self):
        st = self.ctl.settings
        self._syncing = True
        try:
            self.classes.setRowCount(len(st.classes))
            for r, c in enumerate(st.classes):
                for j, (key, _, dec, _) in enumerate(CLASS_COLS):
                    v = c[key]
                    text = str(v) if dec is None else f'{v:.{dec}f}'
                    it = self.classes.item(r, j)
                    if it is None:             # items are kept (an edit lands here while its item is still in use)
                        it = QTableWidgetItem()
                        if dec is not None:
                            it.setFont(mono(12))
                            it.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                        self.classes.setItem(r, j, it)
                    if it.text() != text:
                        it.setText(text)
                    if dec is None:
                        it.setForeground(class_color(r))
            self.classes.setFixedHeight(ROW_H * (len(st.classes) + 1) + 6)
            for name, f in self.req.items():
                f.set_value(st.required.get(name, 0.0))     # leaves a field the user is typing in alone
            self.max_range.set_value(st.max_range)
        finally:
            self._syncing = False
        self._sync_weather()

    # ------------------------------------------------------------------ weather: settings
    def _weather_fields(self):
        lay = QVBoxLayout()
        lay.setSpacing(3)
        top = QHBoxLayout()
        top.addWidget(label('Weather', bold=True))
        top.addStretch(1)
        top.addWidget(label('the sample dataset uses it too', muted=True, small=True))
        lay.addLayout(top)
        row = QHBoxLayout()
        self.cond = QComboBox()
        self.cond.setToolTip('The condition the envelopes, the roses and the verdicts show (core.weather). Clear is '
                             'the reference: no weather model. Saved with the project; the sample dataset renders '
                             'it unless its own setting names another.')
        for c in W.PRESETS:
            self.cond.addItem(c.name, c.id)
        self.cond.currentIndexChanged.connect(self._pick_condition)
        row.addWidget(self.cond, 1)
        self.b_typical = button('Typical', 'ghost', None, 'Back to this condition\'s typical figures')
        self.b_typical.clicked.connect(self._typical)
        row.addWidget(self.b_typical)
        lay.addLayout(row)
        grid = QGridLayout()
        grid.setVerticalSpacing(2)
        grid.setHorizontalSpacing(6)
        grid.setColumnStretch(4, 1)
        self.light = SegmentedControl([('day', 'Day'), ('night', 'Night')], 'day')
        self.light.setToolTip('Night: the vehicle\'s headlamps light what is ahead; the ambient light the rest')
        self.light.changed.connect(lambda v: self._set_param('light', v))
        self.beam = SegmentedControl([('low', 'Low'), ('high', 'High')], 'low')
        self.beam.setToolTip('Headlamps at night: low beam (cut-off below the horizon) or high beam')
        self.beam.changed.connect(lambda v: self._set_param('beam', v))
        a = label('Light', ink2=True)
        a.setFixedWidth(64)
        grid.addWidget(a, 0, 0)
        grid.addWidget(self.light, 0, 1)
        b = label('Beam', ink2=True)
        b.setFixedWidth(86)
        b.setContentsMargins(10, 0, 0, 0)
        grid.addWidget(b, 0, 2)
        grid.addWidget(self.beam, 0, 3)
        self.wx_fields = {}
        for i, (key, name, unit, dec, step, tip) in enumerate(WX_FIELDS):
            f = ScrubSpinBox('', 'plain', dec, unit, step=step, width=86)
            f.setToolTip(tip)
            f.committed.connect(lambda v, k=key: self._set_param(k, float(v)) or self._sync_weather())
            a = label(name, ink2=True)
            a.setFixedWidth(64 if i % 2 == 0 else 86)
            a.setToolTip(tip)
            if i % 2:
                a.setContentsMargins(10, 0, 0, 0)
            grid.addWidget(a, 1 + i // 2, (i % 2) * 2)
            grid.addWidget(f, 1 + i // 2, (i % 2) * 2 + 1)
            self.wx_fields[key] = f
        lay.addLayout(grid)
        self.wx_note = label('', muted=True, small=True, wrap=True)
        lay.addWidget(self.wx_note)
        return lay

    def _wst(self) -> W.WeatherSettings:
        return W.settings_of(self.ctx.store.project)

    def _store_weather(self, st: W.WeatherSettings, text) -> bool:
        new = W.WeatherSettings.from_dict(st.to_dict()).to_dict()
        if new == self._wst().to_dict():
            return False

        def fn(p):
            p.analysis.weather = new
        self.ctx.store.edit('analysis', fn, text)
        return True

    def _pick_condition(self, *_):
        if self._syncing:
            return
        cid = self.cond.currentData()
        st = self._wst()
        if cid not in W.IDS or cid == st.condition:
            return
        st.condition = cid
        self._store_weather(st, f'Weather · {W.PRESET[cid].name.lower()}')
        self._sync_weather()

    def _set_param(self, key, value):
        """An edit of the shown condition's figure (undoable; clear has none)."""
        if self._syncing:
            return False
        st = self._wst()
        cid = st.condition
        if cid == 'clear':
            return False
        if isinstance(value, float) and not math.isfinite(value):
            return False
        p = dict(st.params.get(cid, {}))
        p[key] = value
        st.params = dict(st.params, **{cid: p})
        what = {'visibility_m': 'visibility', 'rain_mm_h': 'rain', 'snow_mm_h': 'snow', 'ambient_lux': 'ambient light',
                'light': 'light', 'beam': 'beam'}.get(key, key)
        done = self._store_weather(st, f'Weather · {W.PRESET[cid].name.lower()} {what}')
        if not done:
            self._sync_weather()
        return done

    def _typical(self):
        st = self._wst()
        if st.condition in st.params:
            st.params = {k: v for k, v in st.params.items() if k != st.condition}
            self._store_weather(st, f'Weather · {W.PRESET[st.condition].name.lower()} typical')

    def _sync_weather(self):
        st = self._wst()
        c = st.selected()
        self._syncing = True
        try:
            i = self.cond.findData(c.id)
            if i != self.cond.currentIndex():
                self.cond.setCurrentIndex(i)
            for j, cc in enumerate(st.conditions()):
                self.cond.setItemText(j, cc.label() if not cc.clear else 'Clear (reference: no weather)')
            editable = not c.clear
            self.light.set_value(c.light)
            self.beam.set_value(c.beam)
            for key, f in self.wx_fields.items():
                f.set_value(getattr(c, key))
                f.setEnabled(editable)
            self.light.setEnabled(editable)
            self.beam.setEnabled(editable and c.night)
            self.b_typical.setEnabled(c.id in st.params)
            self.wx_note.setText(W.NOTES.get(c.id, '') + ('' if c.clear else ' Typical models: ' +
                                 'see the Weather tab\'s notes.'))
        finally:
            self._syncing = False

    # ------------------------------------------------------------------ weather: the comparison
    def _weather_panel(self):
        w = QWidget()
        w.setObjectName('panel')
        lay = QVBoxLayout(w)
        lay.setContentsMargins(8, 6, 8, 6)
        lay.setSpacing(4)
        top = QHBoxLayout()
        top.addWidget(label('Weather · every condition compared', bold=True))
        top.addWidget(label('range straight ahead, m', muted=True, small=True))
        top.addStretch(1)
        self.b_wexport = button('Export…', 'ghost', 'export', 'The comparison, each sensor per condition and the '
                                                             'findings as JSON and CSV, and a plan PNG')
        self.b_wexport.clicked.connect(lambda: self.export_weather())
        top.addWidget(self.b_wexport)
        lay.addLayout(top)
        self.wtabs = QTabWidget()
        self.wtabs.setDocumentMode(True)
        self.wcond = _table(3)
        self.wcond.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.wcond.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.wcond.setToolTip('The farthest range straight ahead each object is resolved at under each condition, '
                              'and the sector checks that pass. Pick a row to see its sensors.')
        self.wcond.currentCellChanged.connect(lambda *_: self._show_sensors())
        self.wtabs.addTab(self.wcond, 'Ranges')
        self.wsens = _table(len(SENSOR_COLS))
        self.wsens.setHorizontalHeaderLabels(list(SENSOR_COLS))
        self.wsens.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        hh = self.wsens.horizontalHeader()
        for c in range(len(SENSOR_COLS) - 1):
            hh.setSectionResizeMode(c, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(len(SENSOR_COLS) - 1, QHeaderView.ResizeMode.Stretch)
        self.wsens.setToolTip('Cameras: the farthest range within 20° of their heading at which they resolve the first '
                              'object; lidars: range on a 10 % target; radars: their farthest zone on a car')
        self.wtabs.addTab(self.wsens, 'Sensors')
        self.wfind = _table(2)
        self.wfind.setHorizontalHeaderLabels(['', 'Finding'])
        self.wfind.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.wfind.setWordWrap(True)
        self.wfind.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.wfind.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.wtabs.addTab(self.wfind, 'Findings')
        notes = label(W.ASSUMPTIONS, muted=True, small=True, wrap=True)
        nw = QWidget()
        nl = QVBoxLayout(nw)
        nl.setContentsMargins(4, 6, 4, 4)
        nl.addWidget(notes)
        nl.addStretch(1)
        self.wtabs.addTab(nw, 'Notes')
        lay.addWidget(self.wtabs, 1)
        self.wstatus = label('', muted=True, small=True, wrap=True)
        lay.addWidget(self.wstatus)
        return w

    def _weather_key(self):
        st = self.ctx.store
        return (id(st.project), st.revision, id(self.ctx.analysis.result))

    def _weather_edited(self, *_):
        """A weather edit changes the perception range too: the controller only watches its own settings."""
        self._sync_weather()
        self._check_weather_stale()
        if self.ctl.active:
            self._wtimer.start()

    def _check_weather_stale(self):
        pr = self.ctl.result
        if pr is not None and (pr.weather or {}).get('settings') != self._wst().to_dict():
            self.ctl._mark_stale()

    def _weather_reset(self, *_):
        self._wgen += 1
        self.wres, self._wkey, self._wbusy = None, None, False
        self._sync_weather()
        self._show_weather()
        if self.ctl.active:
            self._wtimer.start()

    def run_weather(self):
        """core.weather.run on a background thread (geometry once, every condition's thresholds on top)."""
        if self._wbusy or not self.ctl.active and self.isHidden():
            return False
        key = self._weather_key()
        if key == self._wkey and self.wres is not None:
            return False
        ctx = self.ctx
        win = ctx.window
        project = ctx.store.project.snapshot()
        eng = win._shared_engine() if win is not None else ctx.analysis.engine.share()
        self._wgen += 1
        gen = self._wgen
        self._wbusy = True
        self.wstatus.setText('Comparing the conditions…')

        def thread():
            try:
                out = W.run(project, eng, cancel=lambda: gen != self._wgen)
                self._wsig.done.emit(gen, (key, out))
            except RuntimeError:            # the window closed while computing
                pass
            except Exception as e:
                try:
                    self._wsig.failed.emit(gen, f'{type(e).__name__}: {e}\n{traceback.format_exc()}')
                except RuntimeError:
                    pass
        threading.Thread(target=thread, daemon=True).start()
        return True

    def _weather_done(self, gen, out):
        if gen != self._wgen:
            return
        self._wbusy = False
        key, res = out
        if res is None or key[0] != id(self.ctx.store.project):
            return
        self.wres, self._wkey = res, key
        self._show_weather()
        if key != self._weather_key() and self.ctl.active:
            self._wtimer.start()                # edited while it ran

    def _weather_failed(self, gen, msg):
        if gen != self._wgen:
            return
        self._wbusy = False
        self.wstatus.setText('Failed: ' + msg.splitlines()[0])

    def _show_weather(self):
        res = self.wres
        self.b_wexport.setEnabled(res is not None)
        t = self.wcond
        if res is None:
            t.setRowCount(0)
            self.wsens.setRowCount(0)
            self.wfind.setRowCount(0)
            self.wstatus.setText('')
            return
        pr = res.perception
        names = [k.name for k in pr.classes]
        t.setColumnCount(len(names) + 2)
        t.setHorizontalHeaderLabels(['Condition'] + names + ['Sectors'])
        for j, n in enumerate(names, 1):
            t.horizontalHeaderItem(j).setToolTip(f'How far straight ahead a {n.lower()} is resolved, m')
        hh = t.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for c in range(1, len(names) + 2):
            hh.setSectionResizeMode(c, QHeaderView.ResizeMode.ResizeToContents)
        clear = res.compare_row('clear')
        t.setRowCount(len(res.conditions))
        cur = None
        for r, c in enumerate(res.conditions):
            row = res.compare_row(c.id)
            it = QTableWidgetItem(c.name if not c.clear else 'Clear (reference)')
            it.setData(Qt.ItemDataRole.UserRole, c.id)
            it.setToolTip(f'{c.label()}: {W.NOTES.get(c.id, "")}')
            if c.id == res.selected:
                f = it.font()
                f.setBold(True)
                it.setFont(f)
                cur = r
            t.setItem(r, 0, it)
            for j, n in enumerate(names, 1):
                v = row['per_class'][n]['ahead_m']
                ref = clear['per_class'][n]['ahead_m'] if clear else v
                tone = None if ref <= 0 or v >= 0.97 * ref else 'warning' if v >= 0.6 * ref else 'critical'
                t.setItem(r, j, _num_item(f'{v:.0f}', tone, f'{n} straight ahead: {v:.1f} m (clear {ref:.1f} m)'))
            tone = 'good' if row['passed'] == row['checks'] else \
                'warning' if clear is None or row['passed'] >= clear['passed'] else 'critical'
            t.setItem(r, len(names) + 1, _num_item(f'{row["passed"]}/{row["checks"]}', tone,
                                                   'Sector checks that pass under this condition'))
        t.setFixedHeight(ROW_H * (len(res.conditions) + 1) + 6)
        if cur is not None and t.currentRow() < 0:
            t.setCurrentCell(cur, 0)
        self._show_sensors()
        f = self.wfind
        f.setRowCount(len(res.findings))
        for r, fd in enumerate(res.findings):
            it = QTableWidgetItem(fd['severity'])
            it.setForeground(QColor(STATUS.get(fd['severity'], '#888888')))
            f.setItem(r, 0, it)
            f.setItem(r, 1, QTableWidgetItem(fd['message']))
        f.resizeRowsToContents()
        m = res.metrics
        self.wstatus.setText(f'{m["conditions"]} conditions · {len(res.findings)} findings · {m["seconds"]:.1f} s · '
                             'typical models (Notes)')
        if self.wtabs.tabText(2).split(' ')[0] == 'Findings':
            self.wtabs.setTabText(2, f'Findings {len(res.findings)}' if res.findings else 'Findings')

    def _show_sensors(self):
        res = self.wres
        if res is None:
            return
        r = self.wcond.currentRow()
        it = self.wcond.item(r, 0) if r >= 0 else None
        cid = it.data(Qt.ItemDataRole.UserRole) if it is not None else res.selected
        rows = res.sensors.get(cid, [])
        cmp_, cmp0 = res.compare_row(cid), res.compare_row('clear')
        first = res.perception.classes[0].name if res.perception.classes else None
        t = self.wsens
        t.setRowCount(len(rows))
        for i, x in enumerate(rows):
            it = QTableWidgetItem(x['tag'])
            if x['kind'] == 'camera':
                it.setForeground(THEME.cam(x.get('slot', 0)))
            t.setItem(i, 0, it)
            t.setItem(i, 1, QTableWidgetItem(x['kind']))
            note, now, was = '', None, None
            if x['kind'] == 'camera':
                if first and cmp_ and cmp0:
                    now = cmp_['per_camera'].get(x['tag'], {}).get(first)
                    was = cmp0['per_camera'].get(x['tag'], {}).get(first)
                bits = []
                if 'lit_range_m' in x:
                    bits.append(f'a pedestrian lit to {_m(x["lit_range_m"])} m · L_min {x["min_luminance_cd_m2"]:.2f} '
                                'cd/m²')
                if x['lens_drops'] > 0:
                    bits.append(f'drops ≈ {x["lens_drops"] * 100:.0f} % of the image (heuristic)')
                if x['hdr_short']:
                    bits.append(f'{x["dynamic_range_db"]:g} dB < {W.HDR_NIGHT_DB:g} dB')
                note = ' · '.join(bits)
            elif x['kind'] in ('lidar', 'radar'):
                now, was = x['range_m'], x['clear_range_m']
                note = (f'α {x["extinction_per_km"]:.2f}/km at {x["wavelength_nm"]:g} nm'
                        + (f' · false returns ≈ {x["false_returns"] * 100:.1f} % (heuristic)'
                           if x['false_returns'] > 0 else '')) if x['kind'] == 'lidar' else \
                    f'{x["attenuation_db_km"]:.2f} dB/km at {x["frequency_ghz"]:g} GHz'
            else:
                note = x.get('note', '')
            t.setItem(i, 2, _num_item('—' if was is None else _m(was)))
            t.setItem(i, 3, _num_item('—' if now is None else _m(now)))
            ch = None if now is None or was is None or not was or not math.isfinite(was) else 100 * (now - was) / was
            tone = None if ch is None or ch > -3 else 'warning' if ch > -25 else 'critical'
            t.setItem(i, 4, _num_item('—' if ch is None else f'{ch:+.0f} %', tone))
            t.setItem(i, 5, QTableWidgetItem(note))
            t.item(i, 5).setToolTip(note)

    def export_weather(self, folder=None):
        res = self.wres
        if res is None:
            return []
        if folder is None:
            folder = QFileDialog.getExistingDirectory(self, 'Export the weather comparison into')
            if not folder:
                return []
        paths = W.export(res, self.ctx.store.project, folder)
        self.ctx.msg(f'Exported the weather comparison to {folder}', 5000)
        return paths

    # ------------------------------------------------------------------ state
    def showEvent(self, e):
        super().showEvent(e)
        self.ctl.set_active(True)
        self._check_weather_stale()             # the weather may have changed while the page was hidden
        self._wtimer.start()

    def hideEvent(self, e):
        super().hideEvent(e)
        self.ctl.set_active(False)
        self._wtimer.stop()

    def activate(self):
        self.ctl.set_active(True)
        self._check_weather_stale()
        self._wtimer.start()

    def refresh(self):
        ctl = self.ctl
        pr = ctl.result
        win = self.ctx.window
        self.b_run.setEnabled(not (ctl.busy and win is not None and 'perception' in win._jobs))
        self.bar.setVisible(ctl.busy)
        self.b_export.setEnabled(pr is not None)
        cond = PR._condition(pr) if pr is not None else None
        self.status.setText(ctl.status + (f' · {cond.label()}' if cond is not None and not cond.clear else '')
                            + (' · stale: the design or the settings changed' if ctl.stale and pr else ''))
        self._check_weather_stale()             # the weather changed while it ran
        if pr is self._shown:
            return
        self._shown = pr
        self.map.set_result(pr)
        self.roses.set_result(pr)
        if pr is None:
            self.verdicts.setRowCount(0)
            self.pairs.setRowCount(0)
            return
        opts = [(str(i), c.name) for i, c in enumerate(pr.classes)]
        if [(k, b.text()) for k, b in self.pick.buttons.items()] != opts:
            cur = self.pick.value()
            old = self.pick
            self.pick = SegmentedControl(opts, cur if cur in dict(opts) else '0')
            self.pick.changed.connect(lambda k: self.map.set_class(int(k)))
            self._head.replaceWidget(old, self.pick)
            old.hide()
            old.setParent(None)
            old.deleteLater()
        self.map.set_class(int(self.pick.value() or 0))
        # verdicts: classes x sectors
        v = self.verdicts
        v.setRowCount(len(pr.classes))
        v.setHorizontalHeaderLabels(['Object'] + [f'{SHORT.get(s["name"], s["name"])} {s["required_m"]:g}'
                                                  for s in pr.sectors])
        hh = v.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for c in range(1, len(pr.sectors) + 1):
            hh.setSectionResizeMode(c, QHeaderView.ResizeMode.ResizeToContents)
        for r, cl in enumerate(pr.classes):
            it = QTableWidgetItem(cl.name)
            it.setForeground(class_color(r))
            v.setItem(r, 0, it)
            for j, s in enumerate(pr.sectors, 1):
                d = s['classes'][cl.name]
                it = QTableWidgetItem(f'{d["min_m"]:.0f}')
                it.setFont(mono(12))
                it.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                it.setForeground(QColor(STATUS['good' if d['pass'] else 'critical']))
                it.setToolTip(f'{cl.name}, {s["name"]}: resolved out to {d["min_m"]:.1f} m in every direction of the '
                              f'sector (median {d["median_m"]:.1f} m); required {s["required_m"]:g} m'
                              + ('' if d['pass'] else f' · {d["shortfall_m"]:.1f} m short'))
                v.setItem(r, j, it)
        v.setFixedHeight(ROW_H * (len(pr.classes) + 1) + 6)
        # stereo pairs
        t = self.pairs
        t.setRowCount(len(pr.pairs))
        slots = {uid: slot for uid, _, slot in pr.cams}
        for r, pp in enumerate(pr.pairs):
            a, b = pp['tags']
            it = QTableWidgetItem(f'{a} · {b}')
            it.setForeground(THEME.cam(slots.get(pp['a'], 0)))
            t.setItem(r, 0, it)
            vals = [f'{pp["overlap_deg"]:.0f}', f'{pp["baseline_m"]:.2f}'] + \
                   [(f'{pp["depth_error_m"][f"{z:g}"]:.2f}' if math.isfinite(pp['depth_error_m'][f'{z:g}']) else '—')
                    for z in PR.STEREO_AT]
            for j, val in enumerate(vals, 1):
                it = QTableWidgetItem(val)
                it.setFont(mono(12))
                it.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                t.setItem(r, j, it)

    # ------------------------------------------------------------------ export
    def export(self, folder=None):
        pr = self.ctl.result
        if pr is None:
            return []
        if folder is None:
            folder = QFileDialog.getExistingDirectory(self, 'Export the perception range into')
            if not folder:
                return []
        paths = [os.path.join(folder, 'perception_range.json'), os.path.join(folder, 'perception_range.png')]
        with open(paths[0], 'w', encoding='utf-8') as fh:
            fh.write(json.dumps(PR.to_doc(pr, self.ctx.store.project), indent=1) + '\n')
        with open(paths[1], 'wb') as fh:
            fh.write(PR.png(pr))
        self.ctx.msg(f'Exported the perception range to {folder}', 5000)
        return paths
