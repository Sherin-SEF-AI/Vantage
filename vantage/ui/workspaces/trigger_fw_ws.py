"""Embedded ▸ Trigger firmware: the trigger box that fires every camera line at the sync plan's times, locked to the
GNSS PPS. Choose the target and its pins, preview the schedule, generate the four projects and prove each one's
scheduling core on this machine (core.trigger_firmware). Generating and checking runs as a background job."""
from __future__ import annotations

import os
import threading
import time
import traceback

from PyQt6.QtCore import QObject, QRectF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QPainter, QPen
from PyQt6.QtWidgets import (QAbstractItemView, QComboBox, QFileDialog, QGridLayout, QHBoxLayout, QHeaderView,
                             QLineEdit, QListWidget, QPlainTextEdit, QScrollArea, QSplitter, QTableWidget,
                             QTableWidgetItem, QVBoxLayout, QWidget)

from ...core import trigger_firmware as FW
from ..charts import ChartBase
from ..theme import STATUS, THEME, mono, sans
from ..widgets import (ROW_H, ScrubSpinBox, SegmentedControl, StatusLine, ToggleSwitch, button, fit_width, hline,
                       label)

JOB = 'trigger_fw'
SEV = {'good': 'good', 'note': None, 'warning': 'warning', 'serious': 'serious', 'critical': 'critical'}


class _Sig(QObject):
    done = pyqtSignal(int, object)
    failed = pyqtSignal(int, str)
    progress = pyqtSignal(int, float, str)


# ---------------------------------------------------------------------- the schedule
class ScheduleChart(ChartBase):
    """Each trigger line's pulses over one lidar rotation after the PPS edge (or one period of the slowest line),
    one row per line, the camera tags on the left and a time axis in ms under them."""
    ROW, TOP, LEFT, AXIS = 22, 24, 96, 22

    def __init__(self, parent=None):
        super().__init__(parent)
        self.title = 'Schedule after the PPS edge'
        self.lines, self.window_ms, self.width_ms = [], 100.0, 0.1
        self.setMinimumHeight(self.TOP + self.AXIS + self.ROW)

    def set_plan(self, plan: FW.FwPlan | None):
        self.lines = list(plan.lines) if plan else []
        if plan and plan.lidar:
            self.window_ms = 1000.0 / plan.lidar['rate_hz']
            self.subtitle = f'one rotation of {plan.lidar["tag"]} ({self.window_ms:g} ms)'
        elif self.lines:
            self.window_ms = max(1000.0 / ln.rate for ln in self.lines)
            self.subtitle = f'{self.window_ms:g} ms'
        else:
            self.subtitle = ''
        self.width_ms = (plan.width_ns / 1e6) if plan else 0.1
        self.setMinimumHeight(self.TOP + self.AXIS + self.ROW * max(1, len(self.lines)))
        self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.paint_title(p)
        W = float(self.width()) - self.LEFT - 8
        if not self.lines:
            p.setFont(sans(11))
            p.setPen(THEME.color('muted'))
            p.drawText(QRectF(0, self.TOP, self.width(), self.ROW), Qt.AlignmentFlag.AlignVCenter,
                       'No trigger line to schedule.')
            p.end()
            return
        x_of = lambda t: self.LEFT + W * t / self.window_ms
        for i, ln in enumerate(self.lines):
            y = self.TOP + i * self.ROW
            p.setPen(QPen(THEME.color('hair'), 1))
            p.drawLine(int(self.LEFT), int(y + self.ROW - 4), int(self.LEFT + W), int(y + self.ROW - 4))
            p.setFont(mono(11, QFont.Weight.DemiBold))
            p.setPen(THEME.cam(i))
            p.drawText(QRectF(0, y, 18, self.ROW), Qt.AlignmentFlag.AlignVCenter, ln.letter)
            p.setFont(sans(11))
            p.setPen(THEME.color('ink2'))
            p.drawText(QRectF(18, y, self.LEFT - 22, self.ROW), Qt.AlignmentFlag.AlignVCenter,
                       p.fontMetrics().elidedText(' '.join(ln.tags), Qt.TextElideMode.ElideRight, self.LEFT - 24))
            per = 1000.0 / ln.rate
            t = float(ln.phase_ns) / 1e6
            col = THEME.cam(i)
            while t < self.window_ms:
                x = x_of(t)
                w = max(2.0, W * self.width_ms / self.window_ms)
                p.fillRect(QRectF(x, y + 4, w, self.ROW - 8), col)
                t += per
        # axis
        y = self.TOP + len(self.lines) * self.ROW + 2
        p.setPen(QPen(THEME.color('strong'), 1))
        p.drawLine(int(self.LEFT), int(y), int(self.LEFT + W), int(y))
        step = next(s for s in (1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000) if self.window_ms / s <= 10)
        p.setFont(mono(10))
        p.setPen(THEME.color('muted'))
        t = 0.0
        while t <= self.window_ms + 1e-9:
            x = x_of(t)
            p.drawLine(int(x), int(y), int(x), int(y + 4))
            p.drawText(QRectF(x - 30, y + 4, 60, 14), Qt.AlignmentFlag.AlignHCenter, f'{t:g}')
            t += step
        p.end()


# ---------------------------------------------------------------------- the page
class TriggerFirmwarePage(QWidget):
    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        self.result = None
        self.plan = None
        self.busy = False
        self._gen = 0
        self._dirty = True
        self.sig = _Sig()
        self.sig.done.connect(self._thread_done)
        self.sig.failed.connect(self._failed)
        self.sig.progress.connect(lambda g, f, t: self._progress(g, t))
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(self._left())
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(10, 8, 10, 8)
        rl.setSpacing(6)
        self.chart = ScheduleChart()
        rl.addWidget(self.chart)
        self.lines_table = self._table(['Line', 'Pin', 'Cameras', 'Rate', 'First pulse', 'Plan\'s worst'])
        self.lines_table.setMaximumHeight(170)
        rl.addWidget(self.lines_table)
        rl.addWidget(label('Host check: each project\'s scheduling core compiled and simulated here', bold=True))
        self.res_table = self._table(['Target', 'Files', 'Exact clock', 'Locked max', 'Locked p99', 'Holdover max',
                                      'More', 'Status'])
        self.res_table.setMaximumHeight(150)
        self.res_table.itemSelectionChanged.connect(self._pick_target_row)
        rl.addWidget(self.res_table)
        files = QSplitter(Qt.Orientation.Horizontal)
        self.file_list = QListWidget()
        self.file_list.currentTextChanged.connect(self._show_file)
        self.file_view = QPlainTextEdit()
        self.file_view.setReadOnly(True)
        self.file_view.setFont(mono(11))
        files.addWidget(self.file_list)
        files.addWidget(self.file_view)
        files.setSizes([220, 700])
        rl.addWidget(files, 1)
        self.find_host = QWidget()
        self.find_lay = QVBoxLayout(self.find_host)
        self.find_lay.setContentsMargins(0, 0, 0, 0)
        self.find_lay.setSpacing(3)
        sc = QScrollArea()
        sc.setWidgetResizable(True)
        sc.setWidget(self.find_host)
        sc.setMaximumHeight(130)
        rl.addWidget(sc)
        self.findings = []
        split.addWidget(right)
        split.setSizes([330, 1270])
        lay.addWidget(split)
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(120)
        self.timer.timeout.connect(self.refresh)
        ctx.store.changed.connect(self._on_change)
        ctx.store.project_replaced.connect(self._on_replaced)
        THEME.changed.connect(lambda *_: self.chart.update())

    @staticmethod
    def _table(cols):
        t = QTableWidget(0, len(cols))
        t.setHorizontalHeaderLabels(cols)
        t.verticalHeader().hide()
        t.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        t.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        t.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        t.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        t.horizontalHeader().setStretchLastSection(True)
        t.verticalHeader().setDefaultSectionSize(ROW_H)
        return t

    # ------------------------------------------------------------------ settings panel
    def _left(self):
        w = QWidget()
        w.setObjectName('panel')
        ll = QVBoxLayout(w)
        ll.setContentsMargins(8, 6, 8, 8)
        ll.setSpacing(4)
        top = QHBoxLayout()
        top.addWidget(label('Trigger firmware'))
        top.addStretch(1)
        top.addWidget(label('embedded', muted=True, small=True))
        ll.addLayout(top)
        ll.addWidget(hline())
        ll.addWidget(label('Firmware for a trigger box that fires every camera trigger line at the sync plan\'s times, '
                           'locked to the GNSS PPS. All four projects are generated; the scheduling core of each is '
                           'compiled and simulated on this machine. Nothing here is flashed, and the boards only '
                           'produce trigger pulses.', muted=True, small=True, wrap=True))
        self.target = SegmentedControl([(t, FW.TARGETS[t]['short']) for t in FW.TARGET_IDS], 'rp2040')
        self.target.setToolTip('The project the page shows; Export writes all four')
        self.target.changed.connect(lambda k: self._set(f'target {FW.TARGETS[k]["short"]}', target=k))
        ll.addWidget(label('Target', bold=True))
        ll.addWidget(self.target)
        self.groups = SegmentedControl([('plan', 'Plan\'s lines'), ('suggested', 'Suggested lines')], 'plan')
        self.groups.setToolTip('The cameras\' own sync letters, or the trigger groups the sync plan suggests')
        self.groups.changed.connect(lambda k: self._set(f'{k} trigger lines', groups=k))
        ll.addWidget(label('Trigger lines', bold=True))
        ll.addWidget(self.groups)
        grid = QGridLayout()
        grid.setVerticalSpacing(2)
        grid.setHorizontalSpacing(6)
        self.fields = {}

        def num(row, key, text, unit, dec, step, tip, g=grid):
            f = ScrubSpinBox('', 'plain', dec, unit, step=step, width=110)
            f.setToolTip(tip)
            f.committed.connect(lambda v, k=key, t=text, u=unit: self._set(f'{t} {v:g} {u}'.strip(), **{k: v}))
            a = label(text, ink2=True)
            a.setFixedWidth(130)
            a.setToolTip(tip)
            g.addWidget(a, row, 0)
            g.addWidget(f, row, 1)
            self.fields[key] = f
        num(0, 'width_us', 'Pulse width', 'µs', 0, 10.0, 'Width of every trigger pulse')
        num(1, 'latency_us', 'Trigger latency', 'µs', 1, 1.0,
            'The camera\'s delay from the trigger edge to the exposure (FSYNC latency): every pulse comes this much '
            'earlier')
        ll.addLayout(grid)
        self.active_low = ToggleSwitch('Active-low pulses')
        self.active_low.setToolTip('The trigger inputs fire on a falling edge (the lines idle high)')
        self.active_low.clicked.connect(lambda on: self._set('active-low pulses' if on else 'active-high pulses',
                                                             active_low=on))
        ll.addWidget(self.active_low)
        # ---- RP2040
        ll.addSpacing(4)
        ll.addWidget(label('RP2040 pins (C and Rust)', bold=True))
        g2 = QGridLayout()
        g2.setVerticalSpacing(2)
        num(0, 'rp_base_pin', 'First line', 'GP', 0, 1.0, 'The first trigger line\'s GPIO; the others follow it '
            '(PIO drives consecutive pins)', g2)
        num(1, 'rp_pps_pin', 'PPS input', 'GP', 0, 1.0, 'The GNSS PPS (3.3 V) on this GPIO', g2)
        num(2, 'rp_selftest_pin', 'Self-test input', 'GP', 0, 1.0, 'Tied low at power-up: the self-test pattern', g2)
        ll.addLayout(g2)
        # ---- STM32
        ll.addWidget(label('STM32F411 pins', bold=True))
        self.stm_board = SegmentedControl([(k, v['name'].split()[0] if k == 'nucleo' else 'Black Pill')
                                           for k, v in FW.STM32_BOARDS.items()], 'nucleo')
        self.stm_board.changed.connect(lambda k: self._set(f'{FW.STM32_BOARDS[k]["name"]}', stm_board=k))
        ll.addWidget(self.stm_board)
        g3 = QGridLayout()
        g3.setVerticalSpacing(2)
        self.stm_pins = QLineEdit()
        self.stm_pins.setToolTip('The lines\' pins in line order, from ' + ', '.join(FW.STM32_PINS))
        self.stm_pins.editingFinished.connect(self._stm_pins_edited)
        self.stm_pps = QComboBox()
        self.stm_pps.addItems(FW.STM32_PPS_PINS)
        self.stm_pps.setToolTip('The PPS on a TIM2_CH1 pin (input capture)')
        self.stm_pps.activated.connect(lambda i: self._set(f'STM32 PPS on {FW.STM32_PPS_PINS[i]}',
                                                           stm_pps_pin=FW.STM32_PPS_PINS[i]))
        for r, (t, wdg) in enumerate((('Line pins', self.stm_pins), ('PPS input', self.stm_pps))):
            a = label(t, ink2=True)
            a.setFixedWidth(130)
            g3.addWidget(a, r, 0)
            g3.addWidget(wdg, r, 1)
        ll.addLayout(g3)
        # ---- Jetson
        ll.addWidget(label('Jetson GPIO', bold=True))
        g4 = QGridLayout()
        g4.setVerticalSpacing(2)
        self.j_chip = QLineEdit()
        self.j_chip.setToolTip('The GPIO chip (gpiodetect): gpiochip0, or a /dev path')
        self.j_chip.editingFinished.connect(lambda: self._set(f'Jetson chip {self.j_chip.text().strip()}',
                                                              jetson_chip=self.j_chip.text().strip()))
        self.j_offs = QLineEdit()
        self.j_offs.setToolTip('The lines\' offsets on that chip (gpioinfo), in line order, e.g. 105, 106')
        self.j_offs.editingFinished.connect(self._j_offs_edited)
        for r, (t, wdg) in enumerate((('Chip', self.j_chip), ('Line offsets', self.j_offs))):
            a = label(t, ink2=True)
            a.setFixedWidth(130)
            g4.addWidget(a, r, 0)
            g4.addWidget(wdg, r, 1)
        ll.addLayout(g4)
        self.j_clock = SegmentedControl([('realtime', 'CLOCK_REALTIME'), ('tai', 'CLOCK_TAI')], 'realtime')
        self.j_clock.setToolTip('The system clock chrony or phc2sys keeps on GNSS')
        self.j_clock.changed.connect(lambda k: self._set(f'Jetson clock {k}', jetson_clock=k))
        ll.addWidget(self.j_clock)
        # ---- host simulation
        ll.addWidget(label('Host simulation', bold=True))
        g5 = QGridLayout()
        g5.setVerticalSpacing(2)
        num(0, 'sim_ppm', 'Crystal error', 'ppm', 1, 1.0, 'The simulated board\'s crystal, this far off', g5)
        num(1, 'sim_jitter_ns', 'PPS jitter', 'ns', 0, 5.0, 'The simulated PPS edge\'s jitter (1 sigma)', g5)
        num(2, 'sim_seconds', 'Length', 's', 0, 10.0, 'Simulated seconds', g5)
        num(3, 'sim_drop_s', 'PPS dropout', 's', 0, 1.0, 'A dropout this long a third of the way in (holdover)', g5)
        ll.addLayout(g5)
        ll.addStretch(1)
        self.status = label('', muted=True, small=True, wrap=True)
        ll.addWidget(self.status)
        bl = QHBoxLayout()
        b_def = button('Defaults', 'ghost', None, 'Back to the default firmware settings')
        b_def.clicked.connect(self._defaults)
        bl.addWidget(b_def)
        bl.addStretch(1)
        self.b_run = button('Generate and check', 'primary', 'play', 'Generate the four projects and run each one\'s '
                                                                     'host check (gcc, cargo) in the background')
        self.b_run.clicked.connect(lambda: self.start())
        bl.addWidget(self.b_run)
        ll.addLayout(bl)
        self.b_export = button('Export…', None, 'export', 'The projects (one folder per target), the schedule and '
                                                          'the host-check results')
        self.b_export.clicked.connect(lambda: self.export())
        ll.addWidget(self.b_export)
        sc = QScrollArea()
        sc.setWidgetResizable(True)
        sc.setWidget(w)
        fit_width(sc)
        sc.setMinimumWidth(310)
        return sc

    @property
    def settings(self) -> FW.FirmwareSettings:
        return FW.settings_of(self.ctx.store.project)

    def _set(self, text, **kw):
        st = self.settings
        for k, v in kw.items():
            setattr(st, k, v)
        new = FW.FirmwareSettings.from_dict(st.to_dict()).to_dict()
        if new == self.settings.to_dict():
            self._sync_fields()
            return False

        def fn(p):
            p.analysis.trigger_fw = new
        self.ctx.store.edit('analysis', fn, f'Trigger firmware · {text}')
        return True

    def _stm_pins_edited(self):
        pins = [s.strip().upper() for s in self.stm_pins.text().replace(';', ',').split(',') if s.strip()]
        self._set('STM32 line pins ' + ', '.join(pins), stm_pins=pins)

    def _j_offs_edited(self):
        try:
            offs = [int(s) for s in self.j_offs.text().replace(';', ',').split(',') if s.strip()]
        except ValueError:
            self._sync_fields()
            return
        self._set('Jetson line offsets', jetson_offsets=offs)

    def _defaults(self):
        if self.ctx.store.project.analysis.trigger_fw:
            def fn(p):
                p.analysis.trigger_fw = {}
            self.ctx.store.edit('analysis', fn, 'Trigger firmware · defaults')

    def _sync_fields(self):
        st = self.settings
        for k, f in self.fields.items():
            f.set_value(getattr(st, k))
        self.target.set_value(st.target)
        self.groups.set_value(st.groups)
        self.stm_board.set_value(st.stm_board)
        self.j_clock.set_value(st.jetson_clock)
        self.active_low.blockSignals(True)
        self.active_low.setChecked(st.active_low)
        self.active_low.blockSignals(False)
        if not self.stm_pins.hasFocus():
            self.stm_pins.setText(', '.join(st.stm_pins))
        self.stm_pps.setCurrentText(st.stm_pps_pin)
        if not self.j_chip.hasFocus():
            self.j_chip.setText(st.jetson_chip)
        if not self.j_offs.hasFocus():
            self.j_offs.setText(', '.join(map(str, st.jetson_offsets)))

    # ------------------------------------------------------------------ state
    def _on_change(self, *_):
        if self.isVisible():
            self.timer.start()
        else:
            self._dirty = True

    def _on_replaced(self, *_):
        self._gen += 1
        self.busy = False
        self.result = None
        self._on_change()

    def showEvent(self, e):
        super().showEvent(e)
        if self._dirty:
            self.refresh()

    def activate(self):
        if self._dirty:
            self.refresh()

    def refresh(self):
        """The plan preview and the pins (cheap); the result tables show the last run until the next one."""
        self._dirty = False
        self._sync_fields()
        p = self.ctx.store.project
        st = self.settings
        try:
            self.plan = FW.firmware_plan(p, None, st)
        except Exception as e:                       # noqa: BLE001 - a project the sync plan cannot read
            self.plan = None
            self.status.setText(f'No plan: {e}')
        self.chart.set_plan(self.plan)
        pins = FW.pin_assignment(self.plan, st.target, st)[0] if self.plan else {}
        rows = self.plan.lines if self.plan else []
        tb = self.lines_table
        tb.setRowCount(len(rows))
        for i, ln in enumerate(rows):
            vals = [ln.letter, pins.get(ln.letter, '—'), ' '.join(ln.tags), f'{ln.rate} Hz',
                    f'{float(ln.phase_ns) / 1e6:.4f} ms',
                    '—' if ln.worst_ms is None else f'{ln.worst_ms:.2f} ms']
            for c, v in enumerate(vals):
                it = QTableWidgetItem(v)
                if c in (3, 4, 5):
                    it.setFont(mono(12))
                    it.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                if c == 0:
                    it.setForeground(THEME.cam(i))
                tb.setItem(i, c, it)
        self._show_result()
        if not self.busy:
            stale = self.result is not None and self.result.settings.to_dict() != st.to_dict()
            self.status.setText((f'{len(rows)} trigger line{"s" * (len(rows) != 1)} from the '
                                 f'{self.plan.source} groups' if self.plan else '') +
                                (' · the results below are for earlier settings' if stale else ''))

    # ------------------------------------------------------------------ results
    def _show_result(self):
        res = self.result
        tb = self.res_table
        targets = list(FW.TARGET_IDS)
        tb.setRowCount(len(targets) if res else 0)
        if res:
            for i, t in enumerate(targets):
                c = res.checks.get(t) or {}
                sc = c.get('scenarios') or {}
                d = sc.get('disciplined', {})
                ex = sc.get('exact', {})
                more = ''
                if c.get('pio'):
                    more = f'PIO: {"cycle-exact" if c["pio"]["ok"] else "MISMATCH"}'
                if c.get('rust'):
                    more = f'Rust = C: {"yes" if c["rust"].get("identical_to_c") else "NO"}'
                if c.get('jetson'):
                    lj = c['jetson']['late_us']
                    more = f'late p50 {lj["p50"]:.0f} µs, p99 {lj["p99"]:.0f} µs (here)'
                ok = c.get('ok')
                status = 'proved' if ok else ('not run' if ok is None else 'FAILED')
                vals = [FW.TARGETS[t]['short'], str(len(res.trees.get(t, {}))),
                        '—' if not ex else f'≤ {ex.get("max_tick_error", 0):.2f} tick',
                        '—' if not d else f'{d["locked"].get("max_ns", 0):.0f} ns',
                        '—' if not d else f'{d["locked"].get("p99_ns", 0):.0f} ns',
                        '—' if not d else f'{d["holdover"].get("max_ns", 0):.0f} ns', more,
                        c.get('error', '')[:80] or c.get('skipped', '') or status]
                for j, v in enumerate(vals):
                    it = QTableWidgetItem(v)
                    if 2 <= j <= 5:
                        it.setFont(mono(12))
                        it.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                    if j == 7:
                        it.setForeground(QColor(STATUS['good' if ok else 'warning' if ok is None else 'critical']))
                    tb.setItem(i, j, it)
        self._fill_files()
        for w in self.findings:
            self.find_lay.removeWidget(w)
            w.hide()
            w.deleteLater()
        self.findings = []
        found = (res.findings if res else []) or ([{'severity': 'critical', 'message': p}
                                                   for p in (self.plan.problems if self.plan else [])])
        for f in found:
            w = StatusLine(SEV.get(f['severity']), f['message'])
            self.find_lay.addWidget(w)
            self.findings.append(w)

    def _pick_target_row(self):
        rows = self.res_table.selectionModel().selectedRows()
        if rows:
            t = FW.TARGET_IDS[rows[0].row()]
            if t != self.settings.target:
                self._set(f'target {FW.TARGETS[t]["short"]}', target=t)

    def _fill_files(self):
        res = self.result
        t = self.settings.target
        names = sorted(res.trees.get(t, {})) if res else []
        cur = self.file_list.currentItem().text() if self.file_list.currentItem() else 'README.md'
        self.file_list.blockSignals(True)
        self.file_list.clear()
        self.file_list.addItems(names)
        self.file_list.blockSignals(False)
        pick = cur if cur in names else ('README.md' if 'README.md' in names else (names[0] if names else ''))
        if pick:
            self.file_list.setCurrentRow(names.index(pick))
        else:
            self.file_view.setPlainText('Generate and check to see the projects.' if not res else '')

    def _show_file(self, name):
        res = self.result
        data = res.trees.get(self.settings.target, {}).get(name) if res and name else None
        if data is None:
            return
        try:
            self.file_view.setPlainText(data.decode('utf-8'))
        except UnicodeDecodeError:
            self.file_view.setPlainText(f'({len(data)} bytes of binary data)')

    # ------------------------------------------------------------------ the job
    def start(self):
        if self.busy:
            return False
        project = self.ctx.store.project.snapshot()
        st = self.settings
        self._gen += 1
        gen = self._gen
        self.busy = True
        self.b_run.setEnabled(False)
        self.status.setText('Generating the projects and running the host checks…')

        def work(progress, cancel):
            return FW.run(project, settings=st, progress=progress, cancel=cancel)
        win = self.ctx.window
        if win is not None and hasattr(win, 'run_job'):
            ok = win.run_job(JOB, 'Trigger firmware', work, lambda out: self._thread_done(gen, out),
                             lambda m: self._failed(gen, m))
            if not ok:
                self.busy = False
                self.b_run.setEnabled(True)
            return ok

        def thread():
            try:
                out = work(lambda f, t='': self.sig.progress.emit(gen, f, t), lambda: gen != self._gen)
                self.sig.done.emit(gen, out)
            except Exception as e:                    # noqa: BLE001
                try:
                    self.sig.failed.emit(gen, f'{e}\n{traceback.format_exc()}')
                except RuntimeError:
                    pass
        threading.Thread(target=thread, daemon=True).start()
        return True

    def _progress(self, gen, text):
        if gen == self._gen and text:
            self.status.setText(text)

    def _thread_done(self, gen, out):
        if gen != self._gen:
            return
        self.busy = False
        self.b_run.setEnabled(True)
        if out is None:
            self.status.setText('Cancelled')
            return
        self.result = out
        m = out.metrics
        self.status.setText(f'{len(m["proved"])} of {len(out.trees)} projects proved on this machine · '
                            f'{m["seconds"]:.1f} s')
        self.ctx.msg(f'Trigger firmware: {len(m["proved"])} of {len(out.trees)} projects proved', 5000)
        self.refresh()

    def _failed(self, gen, message):
        if gen != self._gen:
            return
        self.busy = False
        self.b_run.setEnabled(True)
        self.status.setText('Failed: ' + message.splitlines()[0][:200])

    # ------------------------------------------------------------------ export
    def export(self, folder=None):
        res = self.result
        if res is None:
            self.ctx.msg('Generate and check first', 4000)
            return []
        if folder is None:
            folder = QFileDialog.getExistingDirectory(self, 'Export the trigger firmware into')
            if not folder:
                return []
        paths = FW.export(res, self.ctx.store.project, folder)
        try:
            from ...core import provenance as PV
            paths = PV.stamp_paths(paths, self.ctx.store.project, folder)
        except Exception:                             # noqa: BLE001 - provenance is extra, never a blocker
            pass
        self.ctx.msg(f'Exported the trigger firmware to {folder}', 5000)
        return paths
