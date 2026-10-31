"""Analysis ▸ Power budget: what every device of the rig draws, whether the DC-DC converter, the fuses and the power
cables carry it, the alternator's margin while driving, the engine-off runtime on the auxiliary battery and the heat
the enclosure must shed. Cheap: computed again on every edit while the page shows."""
from __future__ import annotations

import re

from PyQt6.QtCore import QRectF, Qt, QTimer
from PyQt6.QtGui import QColor, QFont, QPainter
from PyQt6.QtWidgets import (QAbstractItemView, QComboBox, QFileDialog, QGridLayout, QHBoxLayout, QHeaderView,
                             QScrollArea, QSplitter, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from ...core import power as PW
from ...core.mathutil import parse_number
from .. import icons
from ..charts import ChartBase
from ..theme import THEME, STATUS, mono, sans
from ..widgets import fit_width, ROW_H, ScrubSpinBox, SegmentedControl, StatusLine, ToggleSwitch, button, hline, label
from .recording_ws import SENSOR_COLORS, SEV_WORD, ShareBar, Tile, _table

# hue families apart from the eight camera colours: lidar and radar as on the recording page, GNSS and IMU from their
# 3D colours (ui.sensorview.COLORS), the loads at the distribution in neutral greys, other loads in warm greys
FAMILY = {'lidar': SENSOR_COLORS['lidar'], 'radar': SENSOR_COLORS['radar'],
          'gnss': ['#7f9a1c', '#5d7310', '#a3bd45'], 'imu': ['#6e7f96', '#4f5f75', '#93a3b8'],
          'fixed': ['#8a8780', '#6c6a64', '#a7a49c', '#57554f'], 'other': ['#a39787', '#7f7466', '#c2b8aa']}
FIXED_KINDS = ('compute', 'recorder', 'switch', 'ptp')

# (group, key, label, unit, decimals, step, scale, tip); scale: stored value × scale = shown value
FIELDS = (
    ('Supply', 'dcdc_w', 'DC-DC rating', 'W', 0, 10.0, 1.0,
     'The DC-DC converter\'s rated output power: it feeds every circuit of the rig'),
    ('Supply', 'dcdc_eff', 'Efficiency', '%', 1, 0.5, 100.0,
     'The converter\'s efficiency: the vehicle supplies the output power ÷ this'),
    ('Vehicle', 'alternator_a', 'Spare current', 'A', 1, 1.0, 1.0,
     'What the alternator has left for the rig after the vehicle\'s own loads (your figure; lower at idle). On an '
     'EV: the spare current of its own HV-to-12 V converter'),
    ('Vehicle', 'alternator_v', 'at', 'V', 1, 0.1, 1.0, 'The vehicle\'s system voltage while charging'),
    ('Vehicle', 'battery_wh', 'Aux battery', 'Wh', 0, 50.0, 1.0,
     'The auxiliary battery that runs the rig with the engine off (0 = none)'),
    ('Vehicle', 'battery_dod', 'Usable', '%', 0, 5.0, 100.0,
     'The share of the battery you may use (about 50 % for lead-acid, 80–90 % for LiFePO4)'),
    ('Vehicle', 'runtime_target_min', 'Engine-off target', 'min', 0, 5.0, 1.0,
     'How long the rig must run with the engine off: a parked calibration session'),
    ('Loads', 'camera_w', 'Camera, typical', 'W', 2, 0.1, 1.0,
     'A camera whose module gives no power figure: a GMSL camera powered over its coax by the deserializer'),
    ('Loads', 'compute_w', 'Compute', 'W', 0, 5.0, 1.0,
     'The ECU (Jetson AGX Orin: 60 W in its top power mode); 0 = left out'),
    ('Loads', 'recorder_w', 'Recorder', 'W', 0, 5.0, 1.0, 'The data recorder with its disks; 0 = left out'),
    ('Loads', 'switch_w', 'Network switch', 'W', 0, 1.0, 1.0,
     'The switch the lidars and radars plug into; 0 = left out'),
    ('Loads', 'ptp_w', 'Time server', 'W', 0, 1.0, 1.0, 'Time server / PTP grandmaster; 0 = left out'),
    ('Peaks', 'lidar_peak', 'Lidar spin-up', '×', 2, 0.1, 1.0,
     'A lidar draws this many times its steady power for a few seconds while its motor spins up (typical)'),
    ('Peaks', 'peak_factor', 'Others at switch-on', '×', 2, 0.05, 1.0,
     'Every other device draws this many times its steady power at switch-on (typical)'),
    ('Peaks', 'lidar_heater_w', 'Lidar heater', 'W', 0, 1.0, 1.0,
     'A lidar\'s cold-start heater, on top of its spin-up (0 = none)'),
    ('Cables', 'cable_awg', 'Gauge', 'AWG', 0, 1.0, 1.0,
     'The gauge of every lidar\'s, radar\'s and IMU\'s power cable unless set in its row (copper, 20 °C)'),
    ('Cables', 'max_drop_pct', 'Max drop', '%', 1, 0.5, 1.0,
     'The largest voltage drop over a power cable (there and back) at the steady current, % of the bus'),
    ('Cables', 'min_input_v', 'Min input', 'V', 1, 0.5, 1.0,
     'The lowest input voltage the devices accept (9 V for most 12 V automotive sensors: check each datasheet)'),
)

DEVICE_COLS = ('Device', 'Kind', 'Model', 'Steady W', 'Peak W', 'Circuit', 'Current A', 'Cable m', 'AWG', 'Drop',
               'At peak', 'Enclosure')
C_CIRCUIT, C_CABLE, C_AWG, C_DROP, C_PEAK_DROP, C_ENCL = 5, 7, 8, 9, 10, 11
CIRCUIT_COLS = ('Circuit', 'Devices', 'Fuse A', 'Steady A', 'Peak A', 'Load', 'Peak')
C_FUSE = 2
OTHER_COLS = ('Name', 'Steady W', 'Peak W', 'Circuit')
KEY = Qt.ItemDataRole.UserRole


def row_color(row) -> QColor:
    """A camera's colour slot; any other device a step of its kind's hue family."""
    if row['kind'] == 'camera':
        return THEME.cam(row['slot'])
    fam = FAMILY.get(row['kind']) or FAMILY['fixed' if row['kind'] in FIXED_KINDS else 'other']
    return QColor(fam[row['slot'] % len(fam)])


def _parse(text):
    """A typed number ('12', '7.5 A', '3,2'), or None for an empty or 'auto' cell; raises ValueError for junk."""
    t = re.sub(r'(?i)\(?auto\)?|awg|[av]\b|w\b|%|\bm\b', '', str(text)).strip()
    if not t:
        return None
    v = parse_number(t, 'plain')
    if v != v or v in (float('inf'), float('-inf')):
        raise ValueError(text)
    return v


# ---------------------------------------------------------------------- share bar
class PowerBar(ShareBar):
    """Where the vehicle's power goes: each device's steady share, the cables' losses and the converter's loss, as one
    stacked bar (the recording page's, in W)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.title = 'Where the power goes'

    def paint_title(self, p, right=''):
        ChartBase.paint_title(self, p, f'{self.total:,.0f} W from the vehicle' if self.total > 0 else '')

    def paintEvent(self, e):
        if self.segs and self.total > 0:
            return super().paintEvent(e)
        p = QPainter(self)
        self.paint_title(p)
        p.setFont(sans(11))
        p.setPen(THEME.color('muted'))
        p.drawText(QRectF(0, self.BAR_Y, self.width(), self.BAR_H), Qt.AlignmentFlag.AlignVCenter,
                   'Nothing draws power: no visible device and every fixed load at 0 W.')
        p.end()


class _Combo(QComboBox):
    """A circuit picker in a table cell: the wheel scrolls the table, it never changes the circuit."""

    def wheelEvent(self, e):
        e.ignore()


def _combo(value, tip=''):
    cb = _Combo()
    for k in PW.CIRCUITS:
        cb.addItem(PW.CIRCUIT_LABEL[k], k)
    cb.setCurrentIndex(max(0, cb.findData(value)))
    cb.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
    if tip:
        cb.setToolTip(tip)
    return cb


# ---------------------------------------------------------------------- the page
class PowerPage(QWidget):
    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        self.result: PW.PowerResult | None = None
        self._dirty = True
        self._busy = False                     # filling the tables: their edits are not the user's
        self.setObjectName('base')
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        split = QSplitter(Qt.Orientation.Horizontal)
        split.setHandleWidth(2)
        split.addWidget(self._left())
        # ---------------- right: totals, shares, the device table; circuits and findings under it
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
                ('load', 'Rig load', 'Every device\'s steady power, and the peak when everything switches on at once'),
                ('dcdc', 'DC-DC converter', 'The converter\'s output (the loads and their cables\' losses) against '
                                            'its rating, and what it draws from the vehicle'),
                ('alternator', 'Alternator margin', 'The alternator\'s spare power less what the converter draws while '
                                                    'driving'),
                ('runtime', 'Engine-off runtime', 'How long the auxiliary battery runs the rig with the engine off'),
                ('circuit', 'Worst circuit', 'The circuit nearest its fuse: steady current against the fuse rating'),
                ('heat', 'Enclosure heat', 'Steady power of the devices inside the enclosure (and the converter\'s '
                                           'loss when it is inside): the heat to shed')):
            t = self.tiles[key] = Tile(caption, tip)
            tiles.addWidget(t, 1)
        tl.addLayout(tiles)
        self.bar = PowerBar()
        tl.addWidget(self.bar)
        tl.addWidget(label('Every device', bold=True))
        self.table = _table(DEVICE_COLS)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.DoubleClicked
                                   | QAbstractItemView.EditTrigger.EditKeyPressed)
        self.table.setToolTip('Double-click a lidar\'s, radar\'s or IMU\'s cable length or gauge to set it (clear it '
                              'to route again); pick a circuit; tick the devices inside the enclosure')
        self.table.itemChanged.connect(self._device_edited)
        hh = self.table.horizontalHeader()
        for c in range(len(DEVICE_COLS)):
            hh.setSectionResizeMode(c, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        tl.addWidget(self.table, 1)
        right.addWidget(tw)
        low = QSplitter(Qt.Orientation.Horizontal)
        low.setHandleWidth(2)
        cw = QWidget()
        cw.setObjectName('panel')
        cl = QVBoxLayout(cw)
        cl.setContentsMargins(10, 6, 10, 8)
        cl.setSpacing(4)
        cl.addWidget(label('Circuits', bold=True))
        self.circ = _table(CIRCUIT_COLS)
        self.circ.setEditTriggers(QAbstractItemView.EditTrigger.DoubleClicked
                                  | QAbstractItemView.EditTrigger.EditKeyPressed)
        self.circ.setToolTip('Double-click a fuse to enter the rating you fit (clear it for the automatic size: the '
                             f'smallest standard fuse with the steady current within {PW.CONTINUOUS * 100:.0f} % and '
                             'the peak within the rating)')
        self.circ.itemChanged.connect(self._circuit_edited)
        ch = self.circ.horizontalHeader()
        for c in range(len(CIRCUIT_COLS)):
            ch.setSectionResizeMode(c, QHeaderView.ResizeMode.ResizeToContents)
        ch.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        cl.addWidget(self.circ, 1)
        low.addWidget(cw)
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
        low.addWidget(fw)
        low.setSizes([560, 700])
        right.addWidget(low)
        right.setSizes([600, 300])
        split.addWidget(right)
        split.setSizes([350, 1250])
        lay.addWidget(split)
        ctx.store.changed.connect(self._on_change)
        ctx.store.project_replaced.connect(self._on_change)
        THEME.changed.connect(self._on_change)
        self.refresh()

    # ------------------------------------------------------------------ settings panel
    def _left(self):
        w = QWidget()
        w.setObjectName('panel')
        ll = QVBoxLayout(w)
        ll.setContentsMargins(8, 6, 8, 8)
        ll.setSpacing(4)
        top = QHBoxLayout()
        top.addWidget(label('Power budget'))
        top.addStretch(1)
        top.addWidget(label('every device, the supply', muted=True, small=True))
        ll.addLayout(top)
        ll.addWidget(hline())
        ll.addWidget(label('What every device draws, whether the DC-DC converter, the fuses and the power cables carry '
                           'it, the alternator\'s margin while driving and how long the auxiliary battery runs the '
                           'rig with the engine off. Library power figures are nominal; the peak factors typical.',
                           muted=True, small=True, wrap=True))
        self.fields = {}
        group = None
        grid = None
        for g, key, text, unit, dec, step, scale, tip in FIELDS:
            if g != group:
                if grid is not None:
                    ll.addLayout(grid)
                    if group == 'Supply':
                        self.dcdc_inside = ToggleSwitch('DC-DC inside the enclosure')
                        self.dcdc_inside.setToolTip('The converter\'s loss heats the enclosure')
                        self.dcdc_inside.clicked.connect(
                            lambda on: self._set('DC-DC inside the enclosure' if on else 'DC-DC outside the enclosure',
                                                 dcdc_in_enclosure=on))
                        ll.addWidget(self.dcdc_inside)
                group = g
                ll.addSpacing(6)
                ll.addWidget(label(g, bold=True))
                if g == 'Supply':
                    self.bus = SegmentedControl([(f'{v:g}', f'{v:g} V') for v in PW.BUS_VOLTAGES], '12')
                    self.bus.setToolTip('The rig\'s bus: the DC-DC converter\'s output that feeds every circuit. A '
                                        'higher voltage carries the same power at less current')
                    self.bus.changed.connect(lambda k: self._set(f'{k} V bus', bus_v=float(k)))
                    ll.addWidget(self.bus)
                grid = QGridLayout()
                grid.setVerticalSpacing(2)
                grid.setHorizontalSpacing(6)
                grid.setColumnStretch(2, 1)
            f = ScrubSpinBox('', 'plain', dec, unit, step=step, width=110)
            f.setToolTip(tip)
            f.committed.connect(lambda v, k=key, s=scale, t=text, u=unit: self._commit(k, v / s, f'{t} {v:g} {u}'))
            a = label(text, ink2=True)
            a.setFixedWidth(128)
            a.setToolTip(tip)
            i = grid.rowCount()
            grid.addWidget(a, i, 0)
            grid.addWidget(f, i, 1)
            self.fields[key] = (f, scale)
        ll.addLayout(grid)
        # ---- other loads
        ll.addSpacing(6)
        ll.addWidget(label('Other loads', bold=True))
        ll.addWidget(label('Fans, displays, a laptop: anything else on the rig\'s bus. Double-click to edit.',
                           muted=True, small=True, wrap=True))
        self.other = QTableWidget(0, len(OTHER_COLS))
        self.other.setHorizontalHeaderLabels(OTHER_COLS)
        self.other.verticalHeader().setVisible(False)
        self.other.verticalHeader().setDefaultSectionSize(ROW_H)
        self.other.horizontalHeader().setFixedHeight(ROW_H)
        self.other.setShowGrid(False)
        self.other.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.other.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.other.setEditTriggers(QAbstractItemView.EditTrigger.DoubleClicked
                                   | QAbstractItemView.EditTrigger.EditKeyPressed)
        oh = self.other.horizontalHeader()
        oh.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for c in (1, 2, 3):
            oh.setSectionResizeMode(c, QHeaderView.ResizeMode.ResizeToContents)
        self.other.itemChanged.connect(self._other_edited)
        self.other.itemSelectionChanged.connect(self._other_buttons)
        ll.addWidget(self.other)
        ob = QHBoxLayout()
        self.b_add = button('Add load', None, 'plus', 'Add a load to the rig\'s bus')
        self.b_add.clicked.connect(self.add_other)
        self.b_remove = button('Remove', 'ghost', 'trash', 'Remove the selected load (the last one when none is '
                                                           'selected)')
        self.b_remove.clicked.connect(self.remove_other)
        ob.addWidget(self.b_add)
        ob.addWidget(self.b_remove)
        ob.addStretch(1)
        ll.addLayout(ob)
        ll.addStretch(1)
        sc = QScrollArea()
        sc.setWidgetResizable(True)
        sc.setWidget(w)
        fit_width(sc)
        # the status and the buttons stay in view under the scrolling settings
        out = QWidget()
        out.setObjectName('panel')
        ol = QVBoxLayout(out)
        ol.setContentsMargins(0, 0, 0, 8)
        ol.setSpacing(4)
        ol.addWidget(sc, 1)
        self.status = label('', muted=True, small=True, wrap=True)
        self.status.setContentsMargins(8, 0, 8, 0)
        ol.addWidget(self.status)
        bl = QHBoxLayout()
        bl.setContentsMargins(8, 0, 8, 0)
        b_def = self.b_defaults = button('Defaults', 'ghost', None, 'Back to the default power settings')
        b_def.clicked.connect(self._defaults)
        bl.addWidget(b_def)
        bl.addStretch(1)
        self.b_export = button('Export…', None, 'export', 'Every device\'s power, current and cable, the circuits and '
                                                           'the findings as JSON and CSV')
        self.b_export.clicked.connect(lambda: self.export())
        bl.addWidget(self.b_export)
        ol.addLayout(bl)
        out.setMinimumWidth(320)
        return out

    @property
    def settings(self) -> PW.PowerSettings:
        return PW.PowerSettings.from_dict(self.ctx.store.project.analysis.power)

    def _apply(self, text, change):
        """change(settings) edits a copy of the settings; the result (clamped) is saved as one undo step. False when
        nothing changes."""
        st = self.settings
        change(st)
        new = PW.PowerSettings.from_dict(st.to_dict()).to_dict()
        if new == self.settings.to_dict():
            self._sync_fields()                # a clamped value: show what is stored
            return False

        def fn(p):
            p.analysis.power = new
        self.ctx.store.edit('analysis', fn, f'Power budget · {text}')
        return True

    def _set(self, text, **kw):
        return self._apply(text, lambda st: [setattr(st, k, v) for k, v in kw.items()])

    def _commit(self, key, value, text):
        return self._set(text, **{key: int(round(value)) if key == 'cable_awg' else value})

    def _defaults(self):
        if self.ctx.store.project.analysis.power:
            def fn(p):
                p.analysis.power = {}
            self.ctx.store.edit('analysis', fn, 'Power budget · defaults')

    def _sync_fields(self):
        st = self.settings
        d = st.to_dict()
        for key, (f, scale) in self.fields.items():
            f.set_value(d[key] * scale)        # leaves a field the user is typing in alone
        self.bus.set_value(f'{st.bus_v:g}')
        self.dcdc_inside.blockSignals(True)
        self.dcdc_inside.setChecked(st.dcdc_in_enclosure)
        self.dcdc_inside.blockSignals(False)
        self._fill_other(st)

    # ------------------------------------------------------------------ other loads
    def _fill_other(self, st):
        t = self.other
        sel = t.currentRow()
        self._busy = True
        t.setRowCount(len(st.other))
        for i, (name, steady, peak, circ, _) in enumerate(st.other):
            for c, text in enumerate((name, f'{steady:g}', f'{peak:g}')):
                it = QTableWidgetItem(text)
                if c:
                    it.setFont(mono(12))
                    it.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                t.setItem(i, c, it)
            cb = _combo(circ, 'The circuit this load is on')
            cb.currentIndexChanged.connect(
                lambda _, i=i, cb=cb: self._later(self._set_other, i, 3, cb.currentData()))
            t.setCellWidget(i, 3, cb)
        if 0 <= sel < len(st.other):
            t.selectRow(sel)
        t.setFixedHeight(ROW_H * (max(len(st.other), 2) + 1) + 4)
        self._busy = False
        self._other_buttons()

    def _other_buttons(self):
        self.b_remove.setEnabled(self.other.rowCount() > 0)

    def _later(self, fn, *args):
        """Run an edit after the signal that asked for it returns (the tables are rebuilt by the edit)."""
        QTimer.singleShot(0, lambda: fn(*args))

    def _other_edited(self, it):
        if self._busy:
            return
        self._later(self._set_other, it.row(), it.column(), it.text())

    def _set_other(self, i, col, value):
        st = self.settings
        if not 0 <= i < len(st.other):
            return False
        name = st.other[i][0]
        if col == 0:
            text = str(value).strip()
            if not text:
                self._sync_fields()
                return False
            what = f'renamed {name} to {text}'
        elif col in (1, 2):
            try:
                value = _parse(value)
            except ValueError:
                value = None
            if value is None:
                self._sync_fields()
                return False
            what = f'{name} {"steady" if col == 1 else "peak"} {value:g} W'
        else:
            what = f'{name} on the {PW.CIRCUIT_NAME.get(value, "other")} circuit'

        def change(s):
            row = s.other[i]
            if col == 0:
                row[0] = text
            elif col == 1:
                row[1] = value
                row[2] = max(row[2], value)    # a higher steady figure lifts the peak with it
            elif col == 2:
                row[2] = max(value, row[1])
            else:
                row[3] = value
        return self._apply(what, change)

    def add_other(self):
        st = self.settings
        names = {r[0] for r in st.other}
        name = next(f'Load {k}' for k in range(1, len(names) + 2) if f'Load {k}' not in names)
        steady = 10.0
        if self._apply(f'added {name}',
                       lambda s: s.other.append([name, steady, round(steady * s.peak_factor, 2), 'other', False])):
            self.other.selectRow(self.other.rowCount() - 1)
            return True
        return False

    def remove_other(self):
        st = self.settings
        if not st.other:
            return False
        i = self.other.currentRow()
        i = i if 0 <= i < len(st.other) else len(st.other) - 1
        return self._apply(f'removed {st.other[i][0]}', lambda s: s.other.pop(i))

    # ------------------------------------------------------------------ the device and circuit tables
    def _device_edited(self, it):
        if self._busy:
            return
        key = it.data(KEY)
        if key is None:
            return
        if it.column() == C_ENCL:
            self._later(self.set_enclosure, key, it.checkState() == Qt.CheckState.Checked)
        elif it.column() in (C_CABLE, C_AWG):
            self._later(self.set_cable, key, 'm' if it.column() == C_CABLE else 'awg', it.text())

    def set_enclosure(self, key, inside):
        res = self.result
        r = next((r for r in res.rows if r['key'] == key), None) if res else None
        if r is None:
            return False

        def change(s):
            if key.startswith('other:'):
                i = int(key[6:])
                if i < len(s.other):
                    s.other[i][4] = inside
            elif inside and key not in s.enclosure:
                s.enclosure.append(key)
            elif not inside and key in s.enclosure:
                s.enclosure.remove(key)
        return self._apply(f'{r["tag"]} {"inside" if inside else "outside"} the enclosure', change)

    def set_cable(self, key, what, text):
        """Set a device's cable length ('m') or gauge ('awg') from typed text; an empty cell routes it again (or uses
        the default gauge)."""
        res = self.result
        r = next((r for r in res.rows if r['key'] == key), None) if res else None
        if r is None or r['cable_m'] is None:
            return False
        try:
            v = _parse(text)
        except ValueError:
            self.refresh()
            return False

        def change(s):
            own = dict(s.cables.get(key, {}))
            if v is None:
                own.pop(what, None)
            else:
                own[what] = int(round(v)) if what == 'awg' else v
            if own:
                s.cables[key] = own
            else:
                s.cables.pop(key, None)
        word = 'cable' if what == 'm' else 'cable gauge'
        text = (f'{r["tag"]} {word} {"routed" if what == "m" else "default"}' if v is None else
                f'{r["tag"]} cable {v:g} m' if what == 'm' else f'{r["tag"]} cable {int(round(v))} AWG')
        if not self._apply(text, change):
            self.refresh()
            return False
        return True

    def set_circuit(self, key, circuit):
        res = self.result
        r = next((r for r in res.rows if r['key'] == key), None) if res else None
        if r is None or circuit not in PW.CIRCUITS or circuit == r['circuit']:
            return False

        def change(s):
            if key.startswith('other:'):
                i = int(key[6:])
                if i < len(s.other):
                    s.other[i][3] = circuit
            elif circuit == PW.DEFAULT_CIRCUIT[r['kind']]:
                s.circuit_of.pop(key, None)
            else:
                s.circuit_of[key] = circuit
        return self._apply(f'{r["tag"]} on the {PW.CIRCUIT_NAME[circuit]} circuit', change)

    def _circuit_edited(self, it):
        if self._busy or it.column() != C_FUSE:
            return
        self._later(self.set_fuse, it.data(KEY), it.text())

    def set_fuse(self, circuit, text):
        """The fitted fuse of a circuit from typed text; an empty cell or 'auto' sizes it automatically."""
        if circuit not in PW.CIRCUITS:
            return False
        try:
            v = _parse(text)
        except ValueError:
            self.refresh()
            return False
        if v is not None and v <= 0:
            v = None

        def change(s):
            if v is None:
                s.fuses.pop(circuit, None)
            else:
                s.fuses[circuit] = v
        name = PW.CIRCUIT_NAME[circuit]
        if not self._apply(f'{name} fuse automatic' if v is None else f'{name} fuse {v:g} A', change):
            self.refresh()
            return False
        return True

    # ------------------------------------------------------------------ state
    def _on_change(self, *_):
        if self.isVisible():
            self.refresh()
        else:
            self._dirty = True                 # computed when the page shows again

    def showEvent(self, e):
        super().showEvent(e)
        if self._dirty:
            self.refresh()

    def activate(self):
        if self._dirty:
            self.refresh()

    def refresh(self):
        p = self.ctx.store.project
        res = self.result = PW.run(p)
        self._dirty = False
        self._sync_fields()
        st, t = res.settings, res.totals
        sev = {f['kind']: f['severity'] for f in reversed(res.findings)}       # the worst of each kind
        self._tiles(res, sev)
        # ---- shares of the vehicle's power
        segs = [(r['tag'], row_color(r), r['steady_w'],
                 f'{r["tag"]} · {PW.KIND_LABEL[r["kind"]].lower()} · {r["model"]}\n{r["steady_w"]:,.1f} W steady · '
                 f'{r["peak_w"]:,.1f} W peak · {r["steady_a"]:.2f} A · {PW.CIRCUIT_LABEL[r["circuit"]]} circuit')
                for r in res.rows if r['steady_w'] > 0]
        tiny = max(0.05, 0.001 * t['dcdc_input_w'])                          # under 0.1 %: left out of the bar
        if t['cable_loss_w'] >= tiny:
            segs.append(('Cables', THEME.color('strong'), t['cable_loss_w'],
                         f'Power cables\' loss\n{t["cable_loss_w"]:,.1f} W'))
        if t['dcdc_loss_w'] >= tiny:
            segs.append(('DC-DC loss', THEME.color('muted'), t['dcdc_loss_w'],
                         f'The DC-DC converter\'s loss at {st.dcdc_eff * 100:g} % efficiency\n'
                         f'{t["dcdc_loss_w"]:,.1f} W'))
        self.bar.set_data(segs, t['dcdc_input_w'] if segs else 0.0)
        self._busy = True
        try:
            self._fill_devices(res)
            self._fill_circuits(res)
        finally:
            self._busy = False
        # ---- findings
        for w in self.findings:
            self.find_lay.removeWidget(w)
            w.hide()                    # until the deferred delete runs it would still paint over the new rows
            w.deleteLater()
        self.findings = []
        for f in res.findings:
            w = StatusLine('good' if f['severity'] == 'note' else f['severity'], f['message'])
            if f['severity'] == 'note':
                w.icon.setPixmap(icons.pixmap('help', THEME.color('muted'), 12))
            self.find_lay.insertWidget(self.find_lay.count() - 1, w)
            self.findings.append(w)
        worst = res.worst()
        bad = sum(f['severity'] in PW.ISSUE_SEVERITIES for f in res.findings)
        self.status.setText(f'{res.variant}: ' + ('every check passes' if worst == 'good' else
                                                   f'{bad} finding{"s" if bad > 1 else ""} to look at'))

    def _tiles(self, res, sev):
        st, t = res.settings, res.totals
        n = len(res.rows)
        self.tiles['load'].set(f'{t["steady_w"]:,.0f}', 'W', f'{t["peak_w"]:,.0f} W peak · {n} '
                               f'device{"" if n == 1 else "s"}')
        self.tiles['dcdc'].set(f'{t["dcdc_load"] * 100:.0f}', '%', f'{t["dcdc_output_w"]:,.0f} of {st.dcdc_w:,.0f} W · '
                               f'{t["dcdc_input_w"]:,.0f} W in', sev.get('dcdc'))
        m = t['alternator_margin_w']
        if m < 0:
            note = (f'battery empty in {t["battery_drain_min"]:,.0f} min' if t['battery_usable_wh'] > 0 else
                    'drains the vehicle\'s battery')
        else:
            note = f'{t["dcdc_input_w"]:,.0f} of {t["alternator_spare_w"]:,.0f} W spare'
        self.tiles['alternator'].set(f'{m:+,.0f}', 'W', note, sev.get('alternator'))
        rt = t['runtime_min']
        if t['dcdc_input_w'] <= 0:
            self.tiles['runtime'].set('—', '', 'nothing draws power')
        else:
            self.tiles['runtime'].set(f'{rt:,.0f}', 'min', f'{t["battery_usable_wh"]:,.0f} Wh usable · target '
                                      f'{st.runtime_target_min:g} min', sev.get('runtime', 'good'))
        c = res.circuit(t['worst_circuit']) if t['worst_circuit'] else None
        if c is None:
            self.tiles['circuit'].set('—', '', 'no circuit has a load')
        elif c['fuse_a'] is None:
            self.tiles['circuit'].set('—', '', f'{c["label"]}: {c["steady_a"]:.1f} A, no fuse fits', c['severity'])
        else:
            self.tiles['circuit'].set(f'{c["load"] * 100:.0f}', '%', f'{c["label"]}: {c["steady_a"]:.1f} of '
                                      f'{c["fuse_a"]:g} A · peak {c["peak_a"]:.1f} A', c['severity'])
        inside = len(t['heat_tags'])
        self.tiles['heat'].set(f'{t["heat_w"]:,.0f}', 'W', f'{inside} device{"" if inside == 1 else "s"}'
                               + (' + the DC-DC loss' if st.dcdc_in_enclosure else ''))

    def _cell(self, tb, r, c, text, num=False, color=None, tip=None, bold=False, swatch=None, key=None,
              editable=False, check=None):
        it = QTableWidgetItem(text)
        flags = Qt.ItemFlag.ItemIsEnabled
        if editable:
            flags |= Qt.ItemFlag.ItemIsEditable | Qt.ItemFlag.ItemIsSelectable
        if check is not None:
            flags |= Qt.ItemFlag.ItemIsUserCheckable
            it.setCheckState(Qt.CheckState.Checked if check else Qt.CheckState.Unchecked)
        it.setFlags(flags)
        if num:
            it.setFont(mono(12, QFont.Weight.DemiBold if bold else None))
            it.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        elif bold:
            it.setFont(sans(12, QFont.Weight.DemiBold))
        if color is not None:
            it.setForeground(color)
        if swatch is not None:
            it.setData(Qt.ItemDataRole.DecorationRole, swatch)
        if tip:
            it.setToolTip(tip)
        if key is not None:
            it.setData(KEY, key)
        tb.setItem(r, c, it)
        return it

    def _fill_devices(self, res):
        st, t = res.settings, res.totals
        tb = self.table
        tb.setRowCount(len(res.rows) + 1)
        cell = lambda *a, **k: self._cell(tb, *a, **k)          # noqa: E731
        muted = THEME.color('muted')
        for i, r in enumerate(res.rows):
            key = r['key']
            src = {'typical': f'typical camera figure ({st.camera_w:g} W): the module gives none',
                   'missing': 'the model gives no power figure: 0 W', 'library': 'library figure (nominal)',
                   'module': 'the module\'s figure', 'setting': 'the setting on the left',
                   'user': 'your figure (Other loads)'}[r['source']]
            cell(i, 0, r['tag'], swatch=row_color(r), tip=f'{r["tag"]}: {r["supply"]}')
            cell(i, 1, PW.KIND_LABEL[r['kind']])
            cell(i, 2, r['model'])
            cell(i, 3, f'{r["steady_w"]:,.1f}', True, muted if r['source'] in ('typical', 'missing') else None,
                 tip=src)
            cell(i, 4, f'{r["peak_w"]:,.1f}', True,
                 tip='Spin-up' + (f' and {st.lidar_heater_w:g} W heater' if st.lidar_heater_w else '')
                 if r['kind'] == 'lidar' else 'At switch-on')
            cb = _combo(r['circuit'], 'The fused circuit this device is on')
            cb.currentIndexChanged.connect(lambda _, k=key, cb=cb: self._later(self.set_circuit, k, cb.currentData()))
            tb.setCellWidget(i, C_CIRCUIT, cb)
            cell(i, 6, f'{r["steady_a"]:.2f}', True, tip=f'{r["peak_a"]:.2f} A at switch-on')
            if r['cable_m'] is None:
                cell(i, C_CABLE, '—', True, muted, tip=f'No power cable modelled: {r["supply"]}')
                for c in (C_AWG, C_DROP, C_PEAK_DROP):
                    cell(i, c, '', True)
            else:
                cell(i, C_CABLE, f'{r["cable_m"]:.2f}', True, None if r['cable_set'] else muted, key=key,
                     editable=True, tip=('Set length' if r['cable_set'] else
                                         'Routed from the ECU: the Manhattan length × the routing slack + the service '
                                         'loop') + '. Double-click to type a length; clear it to route again')
                cell(i, C_AWG, f'{r["awg"]}', True, None if 'awg' in st.cables.get(key, {}) else muted, key=key,
                     editable=True, tip=f'{PW.awg_ohm_per_km(r["awg"]):.2f} Ω/km, {r["cable_ohm"]:.3f} Ω there and '
                                        'back. Double-click to set; clear it for the default gauge')
                limit = st.max_drop_pct + 1e-9
                if r['drop_pct'] is None:
                    cell(i, C_DROP, 'collapses', True, QColor(STATUS['serious']),
                         tip=f'The cable cannot deliver {r["steady_w"]:.0f} W at {st.bus_v:g} V')
                else:
                    sev = 'good' if r['drop_pct'] <= limit else 'serious'
                    cell(i, C_DROP, f'{r["drop_pct"]:.1f} %', True, QColor(STATUS[sev]),
                         tip=f'{SEV_WORD[sev]}: {r["drop_v"]:.3f} V at {r["steady_a"]:.2f} A steady, '
                             f'{r["v_steady"]:.2f} V at the device (limit {st.max_drop_pct:g} %)')
                if r['peak_drop_pct'] is None:
                    cell(i, C_PEAK_DROP, 'collapses', True, QColor(STATUS['serious']),
                         tip=f'At switch-on the cable cannot deliver {r["peak_w"]:.0f} W at {st.bus_v:g} V')
                else:
                    sev = ('serious' if r['v_peak'] < st.min_input_v - 1e-9 else
                           'warning' if r['peak_drop_pct'] > limit else 'good')
                    cell(i, C_PEAK_DROP, f'{r["peak_drop_pct"]:.1f} %', True, QColor(STATUS[sev]),
                         tip=f'{SEV_WORD[sev]}: at switch-on {r["peak_drop_v"]:.3f} V at {r["peak_a"]:.2f} A, '
                             f'{r["v_peak"]:.2f} V at the device (it needs {st.min_input_v:g} V)')
            cell(i, C_ENCL, '', key=key, check=r['enclosure'],
                 tip='Inside the enclosure: its steady power heats it')
        j = len(res.rows)
        cell(j, 0, 'Total', bold=True)
        cell(j, 1, '')
        cell(j, 2, f'+ {t["cable_loss_w"]:,.1f} W in the cables', color=muted)
        cell(j, 3, f'{t["steady_w"]:,.1f}', True, bold=True)
        cell(j, 4, f'{t["peak_w"]:,.1f}', True, bold=True)
        tb.removeCellWidget(j, C_CIRCUIT)
        cell(j, C_CIRCUIT, f'{st.bus_v:g} V bus')
        cell(j, 6, f'{t["bus_a"]:.2f}', True, bold=True, tip=f'{t["bus_peak_a"]:.2f} A at switch-on')
        for c in (C_CABLE, C_AWG, C_DROP, C_PEAK_DROP, C_ENCL):
            cell(j, c, '')

    def _fill_circuits(self, res):
        tb = self.circ
        tb.setRowCount(len(res.circuits))
        cell = lambda *a, **k: self._cell(tb, *a, **k)          # noqa: E731
        muted = THEME.color('muted')
        for i, c in enumerate(res.circuits):
            cell(i, 0, c['label'])
            cell(i, 1, ' '.join(c['tags']), tip=' '.join(c['tags']))
            if c['fuse_a'] is None:
                cell(i, C_FUSE, 'none fits', True, QColor(STATUS['critical']), key=c['circuit'], editable=True,
                     tip=f'More than a {PW.STANDARD_FUSES_A[-1]:g} A fuse: split the circuit')
            else:
                fits = (f'the smallest standard fuse that holds it: {c["suggested_fuse_a"]:g} A'
                        if c['suggested_fuse_a'] else 'no standard fuse holds it')
                cell(i, C_FUSE, f'{c["fuse_a"]:g}' + (' auto' if c['fuse_auto'] else ''), True,
                     muted if c['fuse_auto'] else None, key=c['circuit'], editable=True,
                     tip=f'{"Sized automatically" if c["fuse_auto"] else "The fuse you fit"}; {fits}. Double-click to '
                         'type the rating you fit; clear it for the automatic size')
            cell(i, 3, f'{c["steady_a"]:.2f}', True, tip=f'{c["steady_w"]:,.1f} W')
            cell(i, 4, f'{c["peak_a"]:.2f}', True, tip=f'{c["peak_w"]:,.1f} W')
            if c['load'] is None:
                cell(i, 5, '—', True)
                cell(i, 6, '—', True)
            else:
                ls = 'critical' if c['load'] > 1 + 1e-9 else 'warning' if c['load'] > PW.CONTINUOUS + 1e-9 else 'good'
                ps = ('critical' if c['peak_load'] > PW.INRUSH_BLOW + 1e-9 else
                      'warning' if c['peak_load'] > 1 + 1e-9 else 'good')
                cell(i, 5, f'{c["load"] * 100:.0f} %', True, QColor(STATUS[ls]),
                     tip=f'{SEV_WORD[ls]}: the steady current should stay within {PW.CONTINUOUS * 100:.0f} % of '
                         'the fuse')
                cell(i, 6, f'{c["peak_load"] * 100:.0f} %', True, QColor(STATUS[ps]),
                     tip=f'{SEV_WORD[ps]}: the peak should stay within the fuse; fuses carry short inrush, a blade '
                         f'fuse blows within seconds at {PW.INRUSH_BLOW:g}×')

    # ------------------------------------------------------------------ export
    def export(self, folder=None):
        res = self.result
        if res is None:
            return []
        if folder is None:
            folder = QFileDialog.getExistingDirectory(self, 'Export the power budget into')
            if not folder:
                return []
        from ...core import provenance as PV          # where the JSON came from
        paths = PV.stamp_paths(PW.export(res, self.ctx.store.project, folder), self.ctx.store.project, folder)
        self.ctx.msg(f'Exported the power budget to {folder}', 5000)
        return paths
