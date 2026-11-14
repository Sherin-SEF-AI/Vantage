"""Embedded ▸ Edge watchdog: the in-car service that watches the rig's camera streams (health and knocks) and alerts
locally. The page sets it up from the design (each camera's stream source, the knock check, the alert channels and the
service's limits, all undoable and saved with the project), shows the configuration file and the systemd unit it
writes, and dry-runs the watchdog's own code on a recording or a synthetic drive with faults written in, scoring what
it caught, how fast, and what it raised that no fault explains."""
from __future__ import annotations

import json
import threading
import traceback

from PyQt6.QtCore import QObject, Qt, pyqtSignal
from PyQt6.QtGui import QGuiApplication
from PyQt6.QtWidgets import (QAbstractItemView, QComboBox, QFileDialog, QGridLayout, QHBoxLayout, QHeaderView,
                             QLineEdit, QPlainTextEdit, QProgressBar, QScrollArea, QSplitter, QTableWidget,
                             QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget)

from ...core import edge_daemon as ED
from ..theme import THEME, mono
from ..widgets import (ROW_H, ScrubSpinBox, SegmentedControl, StatusLine, ToggleSwitch, button, fit_width, hline,
                       label)

JOB = 'edge_dry_run'
# (group, key, label, unit, decimals, step, tip)
FIELDS = (
    ('Knock check', 'analysis_hz', 'Analysis rate', 'Hz', 1, 0.5,
     'Frames analysed per camera per second for the knock check (each with the frame about 0.1 s before it)'),
    ('Knock check', 'knock_deg', 'Knock (travel)', '°', 2, 0.1, 'A camera whose direction of travel moves this much '
                                                                 'from its baseline and stays there is knocked'),
    ('Knock check', 'body_knock_deg', 'Knock (body)', '°', 2, 0.1, 'A camera whose view of the own body moves this '
                                                                   'much and stays there is knocked (the body edge '
                                                                   'scatters far less than the travel direction)'),
    ('Knock check', 'knock_z', 'Significance', 'σ', 1, 0.5, 'The step must also be this many standard errors of the '
                                                            'window median from the baseline'),
    ('Knock check', 'baseline_s', 'Baseline', 's', 1, 1.0, 'The camera\'s pointing is learned over this long after '
                                                            'the watchdog starts'),
    ('Knock check', 'window_s', 'Window', 's', 1, 0.5, 'The current pointing is the median over this long'),
    ('Knock check', 'hold_s', 'Hold', 's', 1, 0.5, 'A step must stay this long to be a knock'),
    ('Alerts', 'raise_s', 'Raise after', 's', 1, 0.5, 'A condition must last this long before it is an alert'),
    ('Alerts', 'clear_s', 'Clear after', 's', 1, 0.5, 'A raised alert clears once its condition has been gone this '
                                                       'long'),
    ('Alerts', 'grace_s', 'Start-up grace', 's', 1, 0.5, 'A stream\'s frame-rate, drop, stall and sync checks wait '
                                                         'this long after it starts'),
    ('Service', 'http_port', 'Status page port', '', 0, 1.0, 'The local status page\'s TCP port (0: none)'),
    ('Service', 'memory_mb', 'Memory limit', 'MB', 0, 64.0, 'systemd MemoryMax for the service'),
    ('Service', 'cpu_percent', 'CPU limit', '%', 0, 10.0, 'systemd CPUQuota for the service (100 % = one core)'),
)
TEXT_FIELDS = (
    ('Alerts', 'mqtt_host', 'MQTT broker', 'Host name or address of an MQTT broker on the car\'s or the depot\'s '
                                           'network; empty: no MQTT'),
    ('Alerts', 'mqtt_topic', 'MQTT topic', 'Topic prefix; alerts go to <topic>/alerts, the status to <topic>/status'),
    ('Alerts', 'webhook_url', 'Webhook', 'An http(s) URL that gets every alert as a JSON POST; empty: none'),
    ('Alerts', 'log_path', 'Log file', 'The watchdog\'s log on the car'),
    ('Alerts', 'gpio_chip', 'GPIO chip', 'The GPIO chip of the buzzer or lamp line (gpiochip0 …)'),
    ('Service', 'http_bind', 'Status page on', '127.0.0.1: the car computer only; its network address: the car\'s '
                                               'network (no password: keep it off networks you do not control)'),
    ('Service', 'service_user', 'Run as', 'The unprivileged user the service runs as (in the video and gpio groups)'),
    ('Service', 'config_path', 'Configuration', 'Where the configuration file goes on the car'),
)
FAULT_KINDS = (('knock', 'Knock'), ('freeze', 'Frozen'), ('drop', 'Dropped frames'), ('stall', 'Stall'),
               ('dark', 'Dark'))
PRESET = [{'kind': 'knock', 'camera': 1, 'at_s': 6.0, 'yaw': 2.0}, {'kind': 'freeze', 'camera': 0, 'at_s': 5.0,
                                                                     'dur_s': 3.0},
          {'kind': 'stall', 'camera': -1, 'at_s': 7.0, 'dur_s': 0.0},
          {'kind': 'drop', 'camera': 2, 'at_s': 4.0, 'dur_s': 5.0, 'keep_every': 3}]


class _Sig(QObject):
    config = pyqtSignal(int, object)
    done = pyqtSignal(int, object)
    failed = pyqtSignal(int, str)
    progress = pyqtSignal(int, float, str)


def _table(cols, stretch=None):
    t = QTableWidget(0, len(cols))
    t.setHorizontalHeaderLabels(cols)
    t.verticalHeader().setVisible(False)
    t.verticalHeader().setDefaultSectionSize(ROW_H)
    t.horizontalHeader().setFixedHeight(ROW_H)
    t.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    t.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    hh = t.horizontalHeader()
    for i in range(len(cols)):
        hh.setSectionResizeMode(i, QHeaderView.ResizeMode.Stretch if i == stretch else
                                QHeaderView.ResizeMode.ResizeToContents)
    return t


def _item(text, num=False, tip=None, color=None):
    it = QTableWidgetItem(text)
    if num:
        it.setFont(mono(12))
        it.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    if tip:
        it.setToolTip(tip)
    if color is not None:
        it.setForeground(color)
    it.setFlags(it.flags() & ~Qt.ItemFlag.ItemIsEditable)
    return it


def _text_view():
    t = QPlainTextEdit()
    t.setReadOnly(True)
    t.setFont(mono(11))
    t.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
    return t


class EdgePage(QWidget):
    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        self.setObjectName('base')
        self._dirty = True
        self._gen = 0
        self._cgen = 0
        self.result = None
        self.cfg = None
        self.busy = False
        self.sig = _Sig()
        self.sig.config.connect(self._config_done)
        self.sig.done.connect(self._done)
        self.sig.failed.connect(self._failed)
        self.sig.progress.connect(self._progress)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(self._left())
        self.tabs = QTabWidget()
        self.tabs.addTab(self._cams_tab(), 'Cameras')
        self.cfg_view = _text_view()
        self.unit_view = _text_view()
        self.tabs.addTab(self._text_tab(self.cfg_view, 'edge.json', 'The configuration file the watchdog reads'),
                         'Configuration')
        self.tabs.addTab(self._text_tab(self.unit_view, 'vantage-edge.service', 'The systemd unit that runs it'),
                         'Service')
        self.tabs.addTab(self._dry_tab(), 'Dry run')
        split.addWidget(self.tabs)
        split.setSizes([330, 1270])
        lay.addWidget(split)
        ctx.store.changed.connect(self._on_change)
        ctx.store.project_replaced.connect(self._on_replaced)

    # ------------------------------------------------------------------ settings panel
    def _left(self):
        w = QWidget()
        w.setObjectName('panel')
        ll = QVBoxLayout(w)
        ll.setContentsMargins(8, 6, 8, 8)
        ll.setSpacing(4)
        top = QHBoxLayout()
        top.addWidget(label('Edge watchdog'))
        top.addStretch(1)
        top.addWidget(label('in the car', muted=True, small=True))
        ll.addLayout(top)
        ll.addWidget(hline())
        ll.addWidget(label('A service for the car\'s computer (a Jetson or any Linux box) that reads the cameras\' '
                           'streams, checks their health and whether a camera has been knocked, and alerts on a local '
                           'status page, MQTT, a webhook, its log and a buzzer line. It only reads and alerts.',
                           muted=True, small=True, wrap=True))
        self.fields, self.texts = {}, {}
        groups = {}
        for g in ('Knock check', 'Alerts', 'Service'):
            ll.addSpacing(6)
            ll.addWidget(label(g, bold=True))
            if g == 'Knock check':
                self.knock_on = ToggleSwitch('Check for knocks')
                self.knock_on.setToolTip('The direction of travel in each camera\'s frame and, where the camera sees '
                                         'the own body, its edge, against their own baseline')
                self.knock_on.clicked.connect(lambda on: self._set('knock check ' + ('on' if on else 'off'),
                                                                    knock_enabled=bool(on)))
                ll.addWidget(self.knock_on)
            if g == 'Alerts':
                self.gpio = SegmentedControl([(k, k.capitalize() if k != 'gpioset' else 'gpioset')
                                              for k in ED.GPIO_BACKENDS], 'none')
                self.gpio.setToolTip('The buzzer or lamp line: libgpiod\'s Python module, its gpioset tool, a file '
                                     '(sysfs value file) or none')
                self.gpio.changed.connect(lambda k: self._set(f'GPIO {k}', gpio_backend=k))
            grid = QGridLayout()
            grid.setVerticalSpacing(2)
            grid.setHorizontalSpacing(6)
            grid.setColumnStretch(1, 1)
            groups[g] = grid
            for fg, key, text, unit, dec, step, tip in FIELDS:
                if fg != g:
                    continue
                f = ScrubSpinBox('', 'plain', dec, unit, step=step, width=120)
                f.setToolTip(tip)
                f.committed.connect(lambda v, k=key, t=text, u=unit: self._set(f'{t} {v:g} {u}'.strip(), **{k: v}))
                a = label(text, ink2=True)
                a.setToolTip(tip)
                i = grid.rowCount()
                grid.addWidget(a, i, 0)
                grid.addWidget(f, i, 1)
                self.fields[key] = f
            for fg, key, text, tip in TEXT_FIELDS:
                if fg != g:
                    continue
                e = QLineEdit()
                e.setToolTip(tip)
                e.setFixedHeight(ROW_H)
                e.editingFinished.connect(lambda k=key, e=e, t=text: self._set(f'{t} {e.text().strip() or "none"}',
                                                                               **{k: e.text().strip()}))
                a = label(text, ink2=True)
                a.setToolTip(tip)
                i = grid.rowCount()
                grid.addWidget(a, i, 0)
                grid.addWidget(e, i, 1)
                self.texts[key] = e
            ll.addLayout(grid)
            if g == 'Alerts':
                ll.addWidget(self.gpio)
                gl = QHBoxLayout()
                gl.addWidget(label('GPIO line', ink2=True))
                self.gpio_line = ScrubSpinBox('', 'plain', 0, '', step=1.0, width=120)
                self.gpio_line.setToolTip('The line\'s offset on its chip (-1: none)')
                self.gpio_line.committed.connect(lambda v: self._set(f'GPIO line {int(v)}', gpio_line=v))
                gl.addStretch(1)
                gl.addWidget(self.gpio_line)
                ll.addLayout(gl)
        ll.addStretch(1)
        self.status = label('', muted=True, small=True, wrap=True)
        ll.addWidget(self.status)
        bl = QHBoxLayout()
        b = self.b_defaults = button('Defaults', 'ghost', None, 'Back to the default watchdog settings')
        b.clicked.connect(self._defaults)
        bl.addWidget(b)
        bl.addStretch(1)
        self.b_install = button('Export…', None, 'export', 'The configuration, the systemd unit and a README with '
                                                           'the install steps')
        self.b_install.clicked.connect(lambda: self.export_install())
        bl.addWidget(self.b_install)
        ll.addLayout(bl)
        sc = QScrollArea()
        sc.setWidgetResizable(True)
        sc.setWidget(w)
        fit_width(sc)
        sc.setMinimumWidth(300)
        return sc

    def _cams_tab(self):
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(8, 8, 8, 8)
        v.addWidget(label('Each camera\'s stream on the car: a V4L2 device (/dev/video0), a GStreamer pipeline, an '
                          'RTSP or HTTP URL, a video file, or a ROS 2 topic (ros2:/camera/image_raw/compressed). '
                          'Double-click a source to change it; empty takes the live view\'s source, else the device of '
                          'the '
                          'camera\'s GMSL port.', muted=True, small=True, wrap=True))
        self.cam_table = _table(('Camera', 'Source', 'From', 'fps', 'Size', 'Body in view'), stretch=1)
        self.cam_table.setEditTriggers(QAbstractItemView.EditTrigger.DoubleClicked |
                                       QAbstractItemView.EditTrigger.EditKeyPressed)
        self.cam_table.itemChanged.connect(self._source_edited)
        v.addWidget(self.cam_table, 1)
        return w

    def _text_tab(self, view, name, tip):
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(8, 8, 8, 8)
        h = QHBoxLayout()
        h.addWidget(label(f'{name} · {tip}', muted=True, small=True))
        h.addStretch(1)
        b = button('Copy', 'ghost', 'copy', f'Copy {name} to the clipboard')
        b.clicked.connect(lambda: (QGuiApplication.clipboard().setText(view.toPlainText()),
                                   self.ctx.msg(f'Copied {name}', 3000)))
        h.addWidget(b)
        v.addLayout(h)
        v.addWidget(view, 1)
        return w

    def _dry_tab(self):
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(8, 8, 8, 8)
        v.addWidget(label('Run the watchdog\'s own code over a recording (a nuScenes folder, a folder of image folders '
                          'or video files, a ROS 2 bag) on its stamps, with faults written in; without a recording, '
                          'over a synthetic drive of this design\'s cameras. Each fault is scored: caught or not, by '
                          'which check and how long after it began; anything else raised is a false alarm.',
                          muted=True, small=True, wrap=True))
        h = QHBoxLayout()
        self.rec = QLineEdit()
        self.rec.setPlaceholderText('Synthetic drive (no recording)')
        self.rec.setFixedHeight(ROW_H)
        h.addWidget(self.rec, 1)
        bf = button('Folder…', 'ghost', 'folder', 'A nuScenes folder, a folder of per-camera image folders or video '
                                                  'files, or a ROS 2 bag folder')
        bf.clicked.connect(self._pick_folder)
        bb = button('File…', 'ghost', 'file', 'A video file or a single .mcap / .db3 bag')
        bb.clicked.connect(self._pick_file)
        h.addWidget(bf)
        h.addWidget(bb)
        v.addLayout(h)
        fl = QHBoxLayout()
        fl.addWidget(label('Faults', bold=True))
        fl.addStretch(1)
        b_add = button('Add', 'ghost', 'plus', 'Add a fault')
        b_add.clicked.connect(lambda: self._add_fault({'kind': 'knock', 'camera': 0, 'at_s': 5.0, 'yaw': 2.0}))
        b_del = button('Remove', 'ghost', 'minus', 'Remove the selected fault')
        b_del.clicked.connect(self._remove_fault)
        b_pre = button('Typical', 'ghost', None, 'A knock, a frozen stream, a stall and dropped frames')
        b_pre.clicked.connect(self._preset)
        for b in (b_add, b_del, b_pre):
            fl.addWidget(b)
        v.addLayout(fl)
        self.fault_table = _table(('Fault', 'Camera', 'At s', 'Size', 'For s'))
        self.fault_table.setEditTriggers(QAbstractItemView.EditTrigger.DoubleClicked |
                                         QAbstractItemView.EditTrigger.EditKeyPressed)
        self.fault_table.setFixedHeight(ROW_H * 6)
        v.addWidget(self.fault_table)
        rl = QHBoxLayout()
        self.b_run = button('Dry run', 'primary', 'play', 'Run the watchdog on the recording with the faults')
        self.b_run.clicked.connect(lambda: self.run())
        rl.addWidget(self.b_run)
        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.setTextVisible(False)
        self.bar.setFixedHeight(4)
        self.bar.hide()
        rl.addWidget(self.bar, 1)
        self.b_export = button('Export…', None, 'export', 'The dry run (JSON, alerts CSV) and the install files')
        self.b_export.clicked.connect(lambda: self.export())
        self.b_export.setEnabled(False)
        rl.addWidget(self.b_export)
        v.addLayout(rl)
        self.run_status = StatusLine(None, 'Not run yet.')
        v.addWidget(self.run_status)
        self.score_table = _table(('Fault', 'Camera', 'At s', 'Caught', 'By', 'After s', 'Size measured',
                                   'Moved'), stretch=None)
        v.addWidget(self.score_table, 1)
        self.alert_table = _table(('Raised s', 'Camera', 'Check', 'Severity', 'Cleared s', 'Message', 'Explained'),
                                  stretch=5)
        v.addWidget(self.alert_table, 1)
        self.cost = label('', muted=True, small=True, wrap=True)
        v.addWidget(self.cost)
        return w

    # ------------------------------------------------------------------ settings
    @property
    def settings(self) -> ED.EdgeSettings:
        return ED.settings_of(self.ctx.store.project)

    def _set(self, text, **kw):
        st = self.settings
        for k, v in kw.items():
            setattr(st, k, v)
        new = ED.EdgeSettings.from_dict(st.to_dict()).to_dict()
        if new == self.settings.to_dict():
            self._sync_fields()
            return False

        def fn(p):
            p.analysis.edge_daemon = new
        self.ctx.store.edit('analysis', fn, f'Edge watchdog · {text}')
        return True

    def _defaults(self):
        if self.ctx.store.project.analysis.edge_daemon:
            def fn(p):
                p.analysis.edge_daemon = {}
            self.ctx.store.edit('analysis', fn, 'Edge watchdog · defaults')

    def _sync_fields(self):
        st = self.settings
        for k, f in self.fields.items():
            f.set_value(float(getattr(st, k)))
        for k, e in self.texts.items():
            if not e.hasFocus():
                e.setText(str(getattr(st, k)))
        self.gpio.set_value(st.gpio_backend)
        self.gpio_line.set_value(float(st.gpio_line))
        self.knock_on.blockSignals(True)
        self.knock_on.setChecked(st.knock_enabled)
        self.knock_on.blockSignals(False)

    def _source_edited(self, it):
        if it.column() != 1 or self._filling:
            return
        tag = self.cam_table.item(it.row(), 0).text()
        src = dict(self.settings.sources)
        val = it.text().strip()
        cur = (self.cfg or {}).get('cameras', [])
        shown = next((c['source'] for c in cur if c['tag'] == tag), '')
        if val == shown and tag not in src:
            return
        if val:
            src[tag] = val
        else:
            src.pop(tag, None)
        self._set(f'{tag} source {val or "default"}', sources=src)

    # ------------------------------------------------------------------ state
    def _on_change(self, *_):
        if self.isVisible():
            self.refresh()
        else:
            self._dirty = True

    def _on_replaced(self, *_):
        self.result = None
        self._gen += 1
        self.busy = False
        self._show_result()
        self._on_change()

    def showEvent(self, e):
        super().showEvent(e)
        if self._dirty:
            self.refresh()

    def activate(self):
        if self._dirty:
            self.refresh()

    def refresh(self):
        """The settings into the fields, then the configuration and the unit (in the background: the planned body
        masks need the analysis engine)."""
        self._dirty = False
        self._sync_fields()
        p = self.ctx.store.project
        st = self.settings
        self.unit_view.setPlainText(ED.systemd_unit(st, f'{p.name} · {p.variant.name}'))
        self._cgen += 1
        gen = self._cgen
        snap = p.snapshot()
        var_uid = p.variant.uid
        win = self.ctx.window
        eng = win._shared_engine() if win is not None and hasattr(win, '_shared_engine') else \
            self.ctx.analysis.engine.share()
        self.status.setText('Writing the configuration…')

        def work():
            try:
                cfg = ED.export_config(snap, var_uid, st, engine=eng)
            except Exception as e:                                # noqa: BLE001
                cfg = {'error': f'{type(e).__name__}: {e}'}
            try:
                self.sig.config.emit(gen, cfg)
            except RuntimeError:                                  # the page is gone
                pass
        threading.Thread(target=work, daemon=True).start()

    def _config_done(self, gen, cfg):
        if gen != self._cgen:
            return
        if 'error' in cfg:
            self.status.setText(f'The configuration could not be written: {cfg["error"]}')
            return
        self.cfg = cfg
        self.cfg_view.setPlainText(ED.config_json(cfg))
        self._fill_cams()
        n = len(cfg['cameras'])
        nos = [c['tag'] for c in cfg['cameras'] if not c['source']]
        self.status.setText(f'{n} camera{"s" * (n != 1)} watched' + (f'; no source for {", ".join(nos)}' if nos else '')
                            + f' · status page {cfg["alerts"]["http"]["bind"]}:{cfg["alerts"]["http"]["port"]}')

    _filling = False

    def _fill_cams(self):
        t = self.cam_table
        self._filling = True
        cams = self.cfg['cameras'] if self.cfg else []
        t.setRowCount(len(cams))
        for i, c in enumerate(cams):
            t.setItem(i, 0, _item(c['tag']))
            src = QTableWidgetItem(c['source'])
            src.setFont(mono(11))
            src.setToolTip('Double-click to change')
            t.setItem(i, 1, src)
            t.setItem(i, 2, _item(c.get('source_from', '')))
            t.setItem(i, 3, _item(f'{c["fps"]:g}', True))
            t.setItem(i, 4, _item(f'{c["width"]}×{c["height"]}', True))
            t.setItem(i, 5, _item('yes' if c.get('body_mask') else 'no',
                                  tip='Whether the plan puts part of the own body in this camera\'s image: the knock '
                                      'check then also watches the body\'s edge'))
        self._filling = False
        self._fill_fault_cams()

    # ------------------------------------------------------------------ faults
    def _tags(self):
        return [c['tag'] for c in (self.cfg or {}).get('cameras', [])] or \
            [c.tag for c in self.ctx.store.project.variant.cameras if c.visible]

    def _fill_fault_cams(self):
        tags = self._tags()
        for r in range(self.fault_table.rowCount()):
            cb = self.fault_table.cellWidget(r, 1)
            if isinstance(cb, QComboBox):
                cur = cb.currentText()
                cb.blockSignals(True)
                cb.clear()
                cb.addItems(tags + ['*'])
                cb.setCurrentText(cur if cur in tags + ['*'] else (tags[0] if tags else '*'))
                cb.blockSignals(False)

    def _add_fault(self, f):
        tags = self._tags()
        t = self.fault_table
        r = t.rowCount()
        t.insertRow(r)
        kind = QComboBox()
        for k, text in FAULT_KINDS:
            kind.addItem(text, k)
        kind.setCurrentIndex([k for k, _ in FAULT_KINDS].index(f['kind']))
        t.setCellWidget(r, 0, kind)
        cam = QComboBox()
        cam.addItems(tags + ['*'])
        ci = f.get('camera', 0)
        cam.setCurrentText(ci if isinstance(ci, str) else (tags[ci % len(tags)] if tags else '*'))
        t.setCellWidget(r, 1, cam)
        size = f.get('yaw', f.get('gain', f.get('keep_every', 0.0)))
        for c, val in ((2, f.get('at_s', 5.0)), (3, size), (4, f.get('dur_s', 2.0))):
            it = QTableWidgetItem(f'{val:g}')
            it.setFont(mono(12))
            it.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            it.setToolTip({2: 'Seconds after the recording\'s first frame', 3: 'Knock: yaw degrees; dark: the '
                           'brightness gain; dropped frames: one in this many is delivered', 4: 'How long (0: to '
                           'the end); a knock stays'}[c])
            t.setItem(r, c, it)

    def _remove_fault(self):
        r = self.fault_table.currentRow()
        if r >= 0:
            self.fault_table.removeRow(r)

    def _preset(self):
        self.fault_table.setRowCount(0)
        for f in PRESET:
            self._add_fault(f)

    def faults(self) -> list:
        out = []
        t = self.fault_table
        for r in range(t.rowCount()):
            kind = t.cellWidget(r, 0).currentData()
            cam = t.cellWidget(r, 1).currentText()

            def num(c, d):
                try:
                    return float(t.item(r, c).text())
                except (AttributeError, ValueError):
                    return d
            f = {'kind': kind, 'camera': cam, 'at_s': num(2, 5.0)}
            size, dur = num(3, 0.0), num(4, 2.0)
            if kind == 'knock':
                f['yaw'] = size
            else:
                f['dur_s'] = dur
            if kind == 'dark':
                f['gain'] = size if size > 0 else 0.06
            if kind == 'drop':
                f['keep_every'] = int(size) if size >= 2 else 3
            out.append(f)
        return out

    def _pick_folder(self):
        d = QFileDialog.getExistingDirectory(self, 'A recording folder')
        if d:
            self.rec.setText(d)

    def _pick_file(self):
        f, _ = QFileDialog.getOpenFileName(self, 'A recording', '', 'Recordings (*.avi *.mp4 *.mkv *.mov *.mcap *.db3)')
        if f:
            self.rec.setText(f)

    # ------------------------------------------------------------------ the dry run
    def run(self):
        if self.busy:
            return False
        p = self.ctx.store.project
        snap = p.snapshot()
        var_uid = p.variant.uid
        st = self.settings
        rec = self.rec.text().strip() or None
        faults = self.faults()
        cfg = json.loads(json.dumps(self.cfg)) if self.cfg else None
        self._gen += 1
        gen = self._gen
        self.busy = True
        self.b_run.setEnabled(False)
        self.run_status.set(None, 'Running the watchdog…')

        def work(progress, cancel):
            return ED.dry_run(snap, rec, faults, settings=st, variant=var_uid, cfg=cfg, progress=progress,
                              cancel=cancel)
        win = self.ctx.window
        if win is not None and hasattr(win, 'run_job'):
            ok = win.run_job(JOB, 'Edge watchdog dry run', work, lambda r: self._done(gen, r),
                             lambda m: self._failed(gen, m))
            if not ok:
                self.busy = False
                self.b_run.setEnabled(True)
            return ok

        def thread():
            try:
                r = work(lambda f, t='': self.sig.progress.emit(gen, f, t), lambda: gen != self._gen)
                self.sig.done.emit(gen, r)
            except Exception as e:                            # noqa: BLE001
                try:
                    self.sig.failed.emit(gen, f'{e}\n{traceback.format_exc()}')
                except RuntimeError:
                    pass
        self.bar.show()
        threading.Thread(target=thread, daemon=True).start()
        return True

    def _progress(self, gen, f, text):
        if gen == self._gen:
            self.bar.setValue(int(f * 1000))
            if text:
                self.run_status.set(None, f'Running the watchdog… {text}')

    def _done(self, gen, res):
        if gen != self._gen:
            return
        self.busy = False
        self.bar.hide()
        self.b_run.setEnabled(True)
        self.result = res
        self._show_result()

    def _failed(self, gen, msg):
        if gen != self._gen:
            return
        self.busy = False
        self.bar.hide()
        self.b_run.setEnabled(True)
        self.run_status.set('serious', 'The dry run failed: ' + msg.splitlines()[0])

    def _show_result(self):
        res = self.result
        self.b_export.setEnabled(res is not None)
        self.score_table.setRowCount(0)
        self.alert_table.setRowCount(0)
        if res is None:
            self.run_status.set(None, 'Not run yet.')
            self.cost.setText('')
            return
        m = res.metrics
        sev = 'good' if ED.exit_code(res) == 0 else 'serious'
        self.run_status.set(sev, f'{m["caught"]} of {m["faults"]} fault{"s" * (m["faults"] != 1)} caught · '
                                 f'{m["false_alarms"]} false alarm{"s" * (m["false_alarms"] != 1)}'
                                 + (f' · slowest {m["worst_delay_s"]:.1f} s' if m['worst_delay_s'] is not None
                                    else '') + f' · {res.recording}')
        st = self.score_table
        st.setRowCount(len(res.faults))
        for i, f in enumerate(res.faults):
            ok = f['caught']
            col = THEME.color('ink') if ok else THEME.color('muted')
            st.setItem(i, 0, _item(dict(FAULT_KINDS).get(f['kind'], f['kind'])))
            st.setItem(i, 1, _item(f['camera']))
            st.setItem(i, 2, _item(f'{f["at_s"]:g}', True))
            st.setItem(i, 3, _item('yes' if ok else 'missed', color=col))
            st.setItem(i, 4, _item(f.get('method') or f.get('check') or ''))
            st.setItem(i, 5, _item('' if f['delay_s'] is None else f'{f["delay_s"]:.1f}', True))
            st.setItem(i, 6, _item('' if f.get('size_deg') is None else f'{f["size_deg"]:.2f}°', True))
            true = f.get('travel_true_deg') if f.get('travel_size_deg') is not None else f.get('body_true_deg')
            st.setItem(i, 7, _item('' if f['kind'] != 'knock' or true is None else f'{true:.2f}°', True,
                                   tip='How far the knock really moved what the method that caught it looks at'))
        false = {id(a) for a in res.false_alarms}
        alerts = res.report.get('alerts', [])
        at = self.alert_table
        at.setRowCount(len(alerts))
        for i, a in enumerate(alerts):
            at.setItem(i, 0, _item(f'{a.get("raised_s", a["raised_t"]):.2f}', True))
            at.setItem(i, 1, _item(a['camera']))
            at.setItem(i, 2, _item(a['check']))
            at.setItem(i, 3, _item(a['severity']))
            at.setItem(i, 4, _item('' if a.get('cleared_s') is None else f'{a["cleared_s"]:.2f}', True))
            at.setItem(i, 5, _item(a['message']))
            at.setItem(i, 6, _item('false alarm' if id(a) in false else 'injected'))
        self.cost.setText(f'{m["frames"]} frames of {m["cameras"]} cameras over {m["recording_s"]:.1f} s · '
                          f'{m["cpu_ms_per_frame"] or 0:.1f} ms CPU per frame on this computer · '
                          + ' · '.join(res.report.get('notes', [])[:3]))

    # ------------------------------------------------------------------ export
    def export_install(self, folder=None):
        if folder is None:
            folder = QFileDialog.getExistingDirectory(self, 'Export the watchdog\'s install files into')
            if not folder:
                return []
        paths = ED.export_install(self.ctx.store.project, folder)
        self.ctx.msg(f'Exported the edge watchdog\'s files to {folder}', 5000)
        return paths

    def export(self, folder=None):
        if self.result is None:
            return []
        if folder is None:
            folder = QFileDialog.getExistingDirectory(self, 'Export the dry run into')
            if not folder:
                return []
        from ...core import provenance as PV
        p = self.ctx.store.project
        paths = PV.stamp_paths(ED.export(self.result, p, folder), p, folder)
        self.ctx.msg(f'Exported the dry run to {folder}', 5000)
        return paths

