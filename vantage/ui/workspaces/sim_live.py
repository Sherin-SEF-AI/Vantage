"""Simulate ▸ CARLA ▸ Live rig: the planned cameras driven in CARLA in real time and streamed to the Live rig page, with
faults injected while it runs (a camera turned on its mount, a frozen stream, dropped or late frames), so the rig
health checks are seen reacting to known faults.

A panel the CARLA page embeds (it also works on its own: SimLivePanel(ctx)). Start live starts the CARLA server when
none answers (through the CARLA page, capped in its own systemd scope), writes the live rig (vantage.sim.live.build_rig:
the chosen cameras at their modules' frame rates), runs the live server (vantage.sim.live in CARLA's Python, capped),
points the cameras' streams in the project at its MJPEG URLs as one undoable edit, and opens the Live rig page with
a small fault window. Stop live tidies CARLA up, stops what it started and puts the streams back (another undoable
edit) when they still hold the live URLs. Render size, served size and the cameras are per machine (they depend on the
GPU), kept in the settings, not in the project.

The panel's own table runs core.rig_health on the same streams while it is shown (the Live rig page runs its own while
that page is shown), so a fault and the verdict it causes are side by side here."""
from __future__ import annotations

import copy
import datetime as _dt
import json
import os
import signal
import subprocess
import threading
import time

from PyQt6.QtCore import QObject, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (QAbstractItemView, QApplication, QCheckBox, QComboBox, QDialog, QGridLayout, QHBoxLayout,
                             QHeaderView, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from ... import workdir
from ...core import rig_health as rh
from ...core import simrig as SR
from ...core import carla_live as CL
from ...sim import live as LV
from ...sim import server as SV
from ..theme import STATUS, mono
from ..widgets import ROW_H, ScrubSpinBox, SegmentedControl, StatusLine, button, hline, label

KEY_ROOT = 'sim/carla_root'            # the CARLA page's keys: the same installation
KEY_PYTHON = 'sim/python'
KEY_SCALE = 'sim/live_scale'
KEY_SERVE = 'sim/live_serve'
KEY_CAMS = 'sim/live_cameras'
DEFAULT_SCALE = 0.25
SERVER_WAIT_S = 300.0                  # a CARLA server takes up to a minute or two to answer
START_WAIT_S = 600.0                   # loading a town, the warm-up and the first frames
COLS = (('Camera', ''),
        ('Served', 'Frames per second the live server serves (what CARLA reaches in real time)'),
        ('Latency', 'Median milliseconds from the start of the step that rendered a frame to it being served'),
        ('Health', 'Rig health\'s verdict on the stream, as the Live rig page gives it (hover for the reasons)'),
        ('fps', 'The frame rate rig health measures from the frames\' arrival times'),
        ('Drift', 'Rig health\'s body-edge indicator in sensor pixels: how far the body edge in the image lies from '
                  'where it should be. Measured against CARLA\'s own car as the camera sees it at its planned mount '
                  '(the live server renders it once at the start), since CARLA\'s car is not the project\'s body; '
                  'against the project\'s body when the server has no mask'),
        ('Fault', 'Faults injected on the camera now'))
HEALTH = {'ok': 'good', 'warning': 'warning', 'fault': 'critical'}


class _Sig(QObject):
    progress = pyqtSignal(int, str)
    started = pyqtSignal(int, object)
    failed = pyqtSignal(int, str)
    stopped = pyqtSignal(int, str)
    event = pyqtSignal(int, object)
    status = pyqtSignal(int, object)
    reply = pyqtSignal(str, object)
    view = pyqtSignal(object)
    body = pyqtSignal(int, str, object)


def _item(text, tip='', num=False, color=None):
    it = QTableWidgetItem(text)
    if tip:
        it.setToolTip(tip)
    if num:
        it.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        it.setFont(mono(12))
    if color:
        it.setForeground(QColor(color))
    return it


# ---------------------------------------------------------------------- fault controls
class FaultControls(QWidget):
    """Inject and clear faults on one camera of the live rig (in the panel, and in the fault window)."""

    def __init__(self, panel):
        super().__init__()
        self.panel = panel
        g = QGridLayout(self)
        g.setContentsMargins(0, 0, 0, 0)
        g.setHorizontalSpacing(6)
        g.setVerticalSpacing(4)
        self.cam = QComboBox()
        self.cam.setToolTip('The camera the fault goes to')
        g.addWidget(label('Camera', ink2=True), 0, 0)
        g.addWidget(self.cam, 0, 1)
        rot = QHBoxLayout()
        rot.setSpacing(4)
        self.turn = {}
        for k, v, tip in (('yaw', 0.0, 'Turn about the camera\'s own up axis (+: to its left)'),
                          ('pitch', 1.0, 'Tilt about the camera\'s own left axis (+: down)'),
                          ('roll', 0.0, 'Roll about the camera\'s view axis')):
            f = ScrubSpinBox('', 'plain', 2, '°', step=0.1, width=64)
            f.set_value(v)
            f.setToolTip(tip)
            rot.addWidget(label(k, muted=True, small=True))
            rot.addWidget(f)
            self.turn[k] = f
        self.b_rotate = button('Rotate', None, 'rotate', 'Turn the camera on its mount by these angles (about its own '
                                                        'axes): CARLA renders it from the turned mount')
        self.b_rotate.clicked.connect(self.rotate)
        rot.addWidget(self.b_rotate)
        rot.addStretch(1)
        g.addLayout(rot, 0, 2)
        row = QHBoxLayout()
        row.setSpacing(4)
        self.b_freeze = button('Freeze', None, 'pause', 'The stream repeats its last frame (a frozen camera)')
        self.b_freeze.clicked.connect(lambda: self.inject({'kind': 'freeze'}))
        row.addWidget(self.b_freeze)
        self.rate = ScrubSpinBox('', 'plain', 0, '%', step=5.0, width=58)
        self.rate.set_value(30.0)
        self.rate.setToolTip('Share of the frames left out')
        self.b_drop = button('Drop', None, 'minus', 'Leave out this share of the frames at random')
        self.b_drop.clicked.connect(lambda: self.inject({'kind': 'drop', 'rate': min(max(self.rate.value(), 0.0),
                                                                                       100.0) / 100.0}))
        row.addWidget(self.b_drop)
        row.addWidget(self.rate)
        self.ms = ScrubSpinBox('', 'plain', 0, 'ms', step=10.0, width=66)
        self.ms.set_value(100.0)
        self.ms.setToolTip('How late every frame is served')
        self.b_delay = button('Delay', None, 'sync', 'Serve every frame this much later (the cameras then '
                                                            'disagree in time)')
        self.b_delay.clicked.connect(lambda: self.inject({'kind': 'delay', 'ms': min(max(self.ms.value(), 0.0),
                                                                                    LV.MAX_MS)}))
        row.addWidget(self.b_delay)
        row.addWidget(self.ms)
        row.addStretch(1)
        self.b_clear = button('Clear', 'ghost', 'close', 'Clear every fault on this camera')
        self.b_clear.clicked.connect(lambda: self.panel.clear_faults(self.cam.currentData()))
        row.addWidget(self.b_clear)
        self.b_clear_all = button('Clear all', 'ghost', None, 'Clear every fault on every camera')
        self.b_clear_all.clicked.connect(lambda: self.panel.clear_faults(None))
        row.addWidget(self.b_clear_all)
        g.addWidget(label('Fault', ink2=True), 1, 0)
        g.addLayout(row, 1, 1, 1, 2)
        self.active = label('', muted=True, small=True, wrap=True)
        g.addWidget(self.active, 2, 0, 1, 3)
        g.setColumnStretch(2, 1)
        self.sync([], [])

    def inject(self, fault: dict):
        tag = self.cam.currentData()
        if tag:
            self.panel.inject(dict(fault, tag=tag))

    def rotate(self):
        self.inject({'kind': 'rotation', 'deg': [round(self.turn[k].value(), 4) for k in ('yaw', 'pitch', 'roll')]})

    def sync(self, tags, faults):
        """The cameras that stream now and the active faults."""
        cur = self.cam.currentData()
        if [self.cam.itemData(i) for i in range(self.cam.count())] != list(tags):
            self.cam.blockSignals(True)
            self.cam.clear()
            for t in tags:
                self.cam.addItem(t, t)
            i = self.cam.findData(cur)
            self.cam.setCurrentIndex(max(i, 0))
            self.cam.blockSignals(False)
        on = bool(tags) and self.panel.state == 'streaming'
        for w in (self.cam, self.b_rotate, self.b_freeze, self.b_drop, self.b_delay, self.b_clear, self.b_clear_all,
                  self.rate, self.ms, *self.turn.values()):
            w.setEnabled(on)
        self.active.setText('Active: ' + '; '.join(f'{f["tag"]} {f.get("text", f["kind"])}'
                                                   + (' (mounting…)' if f.get('state') == 'mounting' else '')
                                                   for f in faults) if faults else
                            ('No fault injected.' if on else 'Faults can be injected while live.'))


class FaultWindow(QDialog):
    """The fault controls in a small window that stays up while the Live rig page is shown."""

    def __init__(self, panel, parent=None):
        super().__init__(parent)
        self.setWindowTitle('CARLA live · faults')
        self.setWindowFlag(Qt.WindowType.Tool, True)
        self.setModal(False)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 8, 10, 10)
        lay.setSpacing(6)
        self.status = label('', muted=True, small=True, wrap=True)
        lay.addWidget(self.status)
        self.controls = FaultControls(panel)
        lay.addWidget(self.controls)
        self.resize(560, 150)


# ---------------------------------------------------------------------- the panel
class SimLivePanel(QWidget):
    def __init__(self, ctx, host=None):
        super().__init__()
        self.ctx = ctx
        self.host = host                  # the CARLA page (its installation, server and settings), or None
        self.setObjectName('panel')
        self.sig = _Sig()
        self.sig.progress.connect(self._on_progress)
        self.sig.started.connect(self._on_started)
        self.sig.failed.connect(self._on_failed)
        self.sig.stopped.connect(self._on_stopped)
        self.sig.event.connect(self._on_event)
        self.sig.status.connect(self._on_status)
        self.sig.reply.connect(self._on_reply)
        self.sig.view.connect(self._on_view)
        self.sig.body.connect(self._on_body)
        self.state = 'off'                # off | starting | streaming | stopping | failed
        self.extra_pythonpath = ''        # tests: a fake carla module ahead of the real one
        self.proc = None
        self.port = None
        self.client = None
        self.folder = ''
        self.info = {}                    # what the start reported: port, tags, rig summary
        self.live = None                  # the live server's last /status
        self.report = None                # live.json of the last session
        self.message = ''
        self._gen = 0
        self._cancel = threading.Event()
        self._plock = threading.Lock()
        self._for_project = None
        self._saved = None                # tag -> the stream it had before (None: none)
        self._urls = {}                   # tag -> the live URL put in its place
        self._server_started = False
        self._own_server = None
        self._polling = False
        self._fault_window = None
        self.monitor = rh.RigMonitor()
        self.snap = None
        self.views, self.view_fail = {}, {}
        self.carla_views = {}             # tag -> rig_health.ExpectedView of CARLA's own car (the live server's mask)
        self._fetching = set()
        self._drift_busy = False
        self._drift_t = 0.0
        self._rr = 0
        self._build()
        self.poll_timer = QTimer(self)
        self.poll_timer.setInterval(1000)
        self.poll_timer.timeout.connect(self._poll)
        self.health_timer = QTimer(self)
        self.health_timer.setInterval(300)
        self.health_timer.timeout.connect(self._health_tick)
        ctx.store.changed.connect(self._on_store_changed)
        ctx.store.project_replaced.connect(self._on_project_replaced)
        app = QApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self.shutdown)
        self._rebuild_cams()
        self._sync()

    # ------------------------------------------------------------------ layout
    def _build(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 6, 10, 8)
        lay.setSpacing(4)
        top = QHBoxLayout()
        top.addWidget(label('Live rig', bold=True))
        top.addStretch(1)
        self.where = label('', muted=True, small=True)
        top.addWidget(self.where)
        lay.addLayout(top)
        lay.addWidget(label('Drive the planned cameras in CARLA in real time, watch them on the Live rig page as a '
                            'real '
                            'rig, and inject faults to see rig health react.', muted=True, small=True, wrap=True))
        self.cam_row = QHBoxLayout()
        self.cam_row.setSpacing(8)
        self.cam_row.addWidget(label('Cameras', ink2=True))
        self.cam_boxes = {}
        self.cam_host = QHBoxLayout()
        self.cam_host.setSpacing(8)
        self.cam_row.addLayout(self.cam_host)
        self.cam_row.addStretch(1)
        lay.addLayout(self.cam_row)
        row = QHBoxLayout()
        row.setSpacing(6)
        row.addWidget(label('Render', ink2=True))
        self.scale = ScrubSpinBox('', 'plain', 0, '%', step=5.0, width=70)
        self.scale.setToolTip('The views CARLA renders, at this share of each camera\'s sensor size (smaller: faster, '
                              'closer to real time)')
        self.scale.committed.connect(self._scale_changed)
        row.addWidget(self.scale)
        row.addSpacing(8)
        row.addWidget(label('Served at', ink2=True))
        self.serve = SegmentedControl([('sensor', 'Sensor size'), ('render', 'As rendered')], 'sensor')
        self.serve.setToolTip('Sensor size: frames resized to the module\'s resolution, as rig health expects them. As '
                              'rendered: smaller frames, less to encode and decode')
        self.serve.changed.connect(lambda k: self.ctx.settings.setValue(KEY_SERVE, k))
        row.addWidget(self.serve)
        row.addStretch(1)
        lay.addLayout(row)
        brow = QHBoxLayout()
        brow.setSpacing(6)
        self.b_start = button('Start live', 'primary', 'play', 'Start the CARLA server if none answers, drive the rig '
                                                              'in '
                                                              'real time, point the cameras\' streams at it (undoable) '
                                                              'and open the Live rig page')
        self.b_start.clicked.connect(lambda: self.start())
        brow.addWidget(self.b_start)
        self.b_stop = button('Stop live', 'ghost', 'stop', 'Stop streaming, tidy CARLA up and put the streams back')
        self.b_stop.clicked.connect(lambda: self.stop())
        brow.addWidget(self.b_stop)
        self.b_page = button('Live rig page', 'ghost', 'external', 'Open the Live rig page with the fault window')
        self.b_page.clicked.connect(lambda: self.open_rig_page())
        brow.addWidget(self.b_page)
        self.b_faults = button('Fault window', 'ghost', 'warning', 'The fault controls in a small window that stays up '
                                                                  'on other pages')
        self.b_faults.clicked.connect(self.show_fault_window)
        brow.addWidget(self.b_faults)
        brow.addStretch(1)
        lay.addLayout(brow)
        self.status_line = StatusLine(None, '')
        lay.addWidget(self.status_line)
        self.table = QTableWidget(0, len(COLS))
        for c, (h, tip) in enumerate(COLS):
            it = QTableWidgetItem(h)
            if tip:
                it.setToolTip(tip)
            self.table.setHorizontalHeaderItem(c, it)
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(ROW_H)
        self.table.horizontalHeader().setFixedHeight(ROW_H)
        self.table.setShowGrid(False)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        hh = self.table.horizontalHeader()
        for c in range(len(COLS) - 1):
            hh.setSectionResizeMode(c, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(len(COLS) - 1, QHeaderView.ResizeMode.Stretch)
        self.table.setMinimumHeight(ROW_H * 4)
        lay.addWidget(self.table, 1)
        lay.addWidget(hline())
        self.controls = FaultControls(self)
        lay.addWidget(self.controls)
        self.limits = label(LV.LIMITS_TEXT, muted=True, small=True, wrap=True)
        lay.addWidget(self.limits)
        s = self.ctx.settings
        try:
            sc = float(s.value(KEY_SCALE, DEFAULT_SCALE) or DEFAULT_SCALE)
        except (TypeError, ValueError):
            sc = DEFAULT_SCALE
        self.scale.set_value(100.0 * min(max(sc, SR.LIMITS['image_scale'][0]), 1.0))
        self.serve.set_value('render' if str(s.value(KEY_SERVE, 'sensor')) == 'render' else 'sensor')

    # ------------------------------------------------------------------ settings and cameras
    @property
    def running(self) -> bool:
        return self.state in ('starting', 'streaming', 'stopping')

    def _hattr(self, name, default=None):
        return getattr(self.host, name, default) if self.host is not None else default

    @property
    def carla_root(self) -> str:
        r = self._hattr('carla_root') or str(self.ctx.settings.value(KEY_ROOT, '') or '')
        if not SV.is_carla_root(r):
            found = SV.candidate_roots()
            r = found[0] if found else ''
        return r

    @property
    def python(self) -> str:
        return self._hattr('python') or str(self.ctx.settings.value(KEY_PYTHON, '') or '') or \
            SV.client_python(self.carla_root)

    def scale_value(self) -> float:
        return min(max(self.scale.value() / 100.0, SR.LIMITS['image_scale'][0]), 1.0)

    def _scale_changed(self, v):
        self.ctx.settings.setValue(KEY_SCALE, self.scale_value())
        self.scale.set_value(100.0 * self.scale_value())

    def _cams(self):
        return [c for c in self.ctx.store.variant.cameras if c.visible]

    def _rebuild_cams(self):
        saved = str(self.ctx.settings.value(KEY_CAMS, '') or '')
        off = set(t for t in saved.split(',') if t) if saved else set()
        for b in self.cam_boxes.values():
            b.hide()
            b.setParent(None)
            b.deleteLater()
        self.cam_boxes = {}
        for c in self._cams():
            b = QCheckBox(c.tag)
            b.setChecked(c.tag not in off)
            b.setToolTip(f'{c.name}: stream this camera live')
            b.toggled.connect(self._cams_changed)
            self.cam_host.addWidget(b)
            self.cam_boxes[c.tag] = b

    def _cams_changed(self, *_):
        off = [t for t, b in self.cam_boxes.items() if not b.isChecked()]
        self.ctx.settings.setValue(KEY_CAMS, ','.join(off))
        self._sync()

    def chosen(self) -> list:
        return [t for t, b in self.cam_boxes.items() if b.isChecked()]

    def set_cameras(self, tags):
        """Choose the cameras to stream (tags; the others off)."""
        for t, b in self.cam_boxes.items():
            b.setChecked(t in set(tags))

    # ------------------------------------------------------------------ start
    def start(self, plan=None) -> bool:
        """Start streaming. plan: the Copilot's ticket when an approved plan starts it: refused unless the project
        holds the plan's approval record for these cameras at this scale."""
        if self.running:
            return False
        if isinstance(plan, dict):              # reached from a Copilot plan: only with its approval record
            from ...assistant import copilot as CP
            ok, why = CP.gate(self.ctx.store.project, plan, 'carla_live', {'cameras': self.chosen(), 'scale': round(self.scale_value(), 4)})
            if not ok:
                return self._refuse(why)
        p = self.ctx.store.project
        st = SR.settings_of(p)
        tags = self.chosen()
        if not tags:
            return self._refuse('Choose at least one camera to stream.')
        py = self.python
        if not py:
            return self._refuse('Choose CARLA\'s client Python first (Installation, above).')
        if self._hattr('busy', False):
            return self._refuse('A CARLA recording is running: start live when it is done.')
        win = self.ctx.window
        other = win.carla_user('live') if win is not None and hasattr(win, 'carla_user') else ''
        if other:                                     # a campaign or critical targets on the same server
            return self._refuse(other)
        if self.host is not None and getattr(self.host, 'no_render', None) is not None and \
                self.host.no_render.isChecked() and not SV.port_open('127.0.0.1', st.port):
            return self._refuse('Live needs a server that renders: turn "No rendering" off.')
        srv = self._hattr('server')
        if srv is not None and getattr(srv, 'running', False) and getattr(srv, 'no_render', False):
            return self._refuse('The CARLA server runs without rendering: the cameras need one that renders. Stop it '
                                'and start it again with rendering.')
        self._gen += 1
        gen = self._gen
        self._cancel = threading.Event()
        self._for_project = p
        self._server_started = False
        self.report = None
        self.live = None
        if not SV.port_open('127.0.0.1', st.port):
            if not self._start_server(st):
                return False
            self._server_started = True
        folder = self._new_dir()
        self.folder = folder
        args = ['--port', st.port, '--tm-port', st.tm_port, '--timeout', f'{st.timeout_s:g}',
                '--serve-size', self.serve.value() or 'sensor', '--parent-pid', os.getpid(),
                '--report', os.path.join(folder, 'live.json')]
        snap = p.snapshot()
        self._set_state('starting', 'Starting…')
        threading.Thread(target=self._work, name='carla-live-start', daemon=True,
                         args=(gen, snap, st, tags, self.scale_value(), folder, py, self.extra_pythonpath or
                               self._hattr('extra_pythonpath', '') or '', args, self._cancel,
                               self._server_started)).start()
        return True

    def _refuse(self, text) -> bool:
        self.status_line.set('warning', text)
        self.ctx.msg(text, 6000)
        return False

    def _start_server(self, st) -> bool:
        root = self.carla_root
        if not SV.is_carla_root(root):
            return self._refuse(f'No CARLA server answers on port {st.port} and no CARLA folder is chosen: choose it '
                                'above (Installation).')
        if self.host is not None and hasattr(self.host, 'start_server'):
            if not self.host.start_server():
                return self._refuse('The CARLA server did not start: see the Server line above.')
            return True
        log = os.path.join(workdir.project_dir(self.ctx.store.project, workdir.get(self.ctx.settings), 'carla'),
                           'server.log')
        try:
            os.makedirs(os.path.dirname(log), exist_ok=True)
            self._own_server = SV.Server(root, st.port, st.memory_gb, st.quality, False, log)
            self._own_server.start()
        except Exception as e:                        # noqa: BLE001 (shown)
            self._own_server = None
            return self._refuse(f'The CARLA server did not start: {e}')
        return True

    def _new_dir(self) -> str:
        p = self.ctx.store.project
        root = workdir.project_dir(p, workdir.get(self.ctx.settings), 'carla')
        stamp = _dt.datetime.now().strftime('%Y%m%d-%H%M%S')
        out = os.path.join(root, f'live-{workdir.slug(p.variant.name)}-{stamp}')
        k = 2
        while os.path.exists(out):
            out = os.path.join(root, f'live-{workdir.slug(p.variant.name)}-{stamp}-{k}')
            k += 1
        return out

    def _server_died(self) -> bool:
        srv = self._own_server or self._hattr('server')
        return srv is not None and getattr(srv, 'proc', None) is not None and srv.proc.poll() is not None

    def _work(self, gen, snap, st, tags, scale, folder, py, extra, args, cancel, waited_server):
        """(a worker thread) The server answering, the live rig, the live server streaming: started(info) or
        failed(message); stopped('cancelled') when Stop came first."""
        def prog(text):
            try:
                self.sig.progress.emit(gen, text)
            except RuntimeError:
                pass
        proc = None
        port = None
        try:
            t0 = time.monotonic()
            while not SV.port_open('127.0.0.1', st.port):
                if cancel.is_set():
                    raise LV.Cancelled()
                if self._server_died():
                    raise RuntimeError('The CARLA server stopped while starting: see its log (server.log in the '
                                       'project\'s carla folder).')
                if time.monotonic() - t0 > SERVER_WAIT_S:
                    raise RuntimeError(f'No CARLA server answered on port {st.port} within {SERVER_WAIT_S:.0f} s.')
                prog(f'Waiting for the CARLA server · {time.monotonic() - t0:.0f} s')
                cancel.wait(1.0)
            if waited_server:
                cancel.wait(5.0)                      # the port opens before the server takes clients
            prog('Writing the live rig')
            rig = CL.build_rig(snap, st, tags, scale, cancel=cancel.is_set)
            if rig is None or cancel.is_set():
                raise LV.Cancelled()
            rig_dir = os.path.join(folder, 'rig')
            SR.export(rig, snap, rig_dir)
            cmd, env = CL.command(py, os.path.join(rig_dir, 'sim_rig.json'), 0, args, extra_path=extra)
            log = open(os.path.join(folder, 'live.log'), 'w', encoding='utf-8')
            with self._plock:
                if cancel.is_set():
                    log.close()
                    raise LV.Cancelled()
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=log, stdin=subprocess.DEVNULL, text=True,
                                        env=env, start_new_session=True, cwd=folder)
                self.proc = proc
            log.close()                               # the child keeps its own descriptor
            events = []
            done = threading.Event()

            def reader():
                try:
                    for line in proc.stdout:
                        ev = LV.parse_event(line)
                        if ev is None:
                            continue
                        events.append(ev)
                        try:
                            self.sig.event.emit(gen, ev)
                        except RuntimeError:
                            return
                finally:
                    done.set()
                    try:
                        self.sig.event.emit(gen, {'event': 'exit', 'code': proc.wait()})
                    except RuntimeError:
                        pass
            threading.Thread(target=reader, name='carla-live-out', daemon=True).start()
            t0 = time.monotonic()
            last = ''
            while True:
                if cancel.is_set():
                    raise LV.Cancelled()
                if port is None:
                    port = next((e['port'] for e in events if e.get('event') == 'listening'), None)
                bad = next((e for e in events if e.get('state') == 'failed' or e.get('event') == 'stopped'), None)
                if bad is not None or done.is_set():
                    raise RuntimeError(self._why(bad, folder))
                if port is not None:
                    try:
                        s = LV.LiveClient(port, timeout=2.0).status()
                    except OSError:
                        s = {}
                    if s.get('state') == 'streaming' and any(c['served'] for c in s.get('cameras', {}).values()):
                        info = {'port': port, 'tags': list(s['cameras']), 'status': s, 'folder': folder,
                                'tick_hz': rig.doc['simulation']['tick_hz'],
                                'schedule': rh.schedule_offsets(rig.doc),
                                'render_mpx': round(sum(c['render_mpx'] for c in rig.doc['cameras']), 2)}
                        self.sig.started.emit(gen, info)
                        return
                    state = {'connecting': 'Connecting to CARLA', 'warming up': 'Traffic settling',
                             'streaming': 'First frames'}.get(s.get('state'), 'Starting the live server')
                    if state != last:
                        prog(state)
                        last = state
                if time.monotonic() - t0 > START_WAIT_S:
                    raise RuntimeError(f'The live server did not stream within {START_WAIT_S:.0f} s.')
                cancel.wait(0.25)
        except LV.Cancelled:
            self._end_proc(proc, port)
            try:
                self.sig.stopped.emit(gen, 'cancelled')
            except RuntimeError:
                pass
        except Exception as e:                        # noqa: BLE001 (shown)
            self._end_proc(proc, port)
            try:
                self.sig.failed.emit(gen, str(e) if isinstance(e, RuntimeError) else f'{type(e).__name__}: {e}')
            except RuntimeError:
                pass

    @staticmethod
    def _why(ev, folder) -> str:
        if ev is not None and ev.get('error'):
            return f'The live server failed: {ev["error"]}'
        try:
            with open(os.path.join(folder, 'live.log'), encoding='utf-8', errors='replace') as fh:
                tail = [x.strip() for x in fh.read().splitlines() if x.strip()][-3:]
        except OSError:
            tail = []
        return 'The live server stopped' + (f': {" · ".join(tail)}' if tail else ' without a word') + \
            f' (see {os.path.join(folder, "live.log")}).'

    def _end_proc(self, proc, port):
        """(a worker thread) Ask the live server to stop and tidy up; signal it, then kill it, when it does not."""
        if proc is None:
            return
        if port is not None and proc.poll() is None:
            try:
                LV.LiveClient(port, timeout=3.0).stop()
            except (OSError, LV.LiveError):
                pass
        for sig, wait in ((None, 30.0), (signal.SIGINT, 20.0), (signal.SIGKILL, 10.0)):
            if sig is not None and proc.poll() is None:
                try:
                    os.killpg(proc.pid, sig)
                except (ProcessLookupError, PermissionError):
                    try:
                        proc.send_signal(sig)
                    except ProcessLookupError:
                        pass
            try:
                proc.wait(wait if port is not None or sig is not None else 0.1)
                return
            except subprocess.TimeoutExpired:
                continue

    # ------------------------------------------------------------------ started, stopped
    def _on_progress(self, gen, text):
        if gen == self._gen and self.state == 'starting':
            self.status_line.set(None, f'Starting live · {text}…')

    def _on_started(self, gen, info):
        if gen != self._gen or self.state != 'starting':
            return
        if self.ctx.store.project is not self._for_project:
            self.info = info
            self.port = info['port']
            self.state = 'streaming'                  # (no stream of the new project points at it): stop it now
            self.stop('the project was replaced')
            return
        self.info = info
        self.port = info['port']
        self.client = LV.LiveClient(self.port)
        self.live = info.get('status')
        self.carla_views, self._fetching = {}, set()
        self._fetch_bodies()
        self._point_streams(info['tags'])
        self._set_state('streaming', '')
        self.poll_timer.start()
        if self.isVisible():
            self._start_monitor()
        self.ctx.msg(f'CARLA live: {len(info["tags"])} camera(s) streaming; the Live rig page shows them', 6000)
        self.open_rig_page()

    def _on_failed(self, gen, msg):
        if gen != self._gen:
            return
        with self._plock:
            self.proc = None
        self._restore_streams()
        self._stop_server_if_ours()
        self._close_fault_window()
        self.message = msg
        self._set_state('failed', msg)
        self.ctx.msg('CARLA live failed: ' + msg.splitlines()[0], 8000)

    def _on_event(self, gen, ev):
        if gen != self._gen:
            return
        kind = ev.get('event')
        if kind == 'stopped':
            self.message = ev.get('error') or ev.get('reason') or ''
        if kind == 'exit' and self.state == 'streaming':    # the live server ended by itself (CARLA died, …)
            self.poll_timer.stop()
            self._stop_monitor()
            self._stop_rig_page()
            self._restore_streams()
            self._stop_server_if_ours()
            self._close_fault_window()
            with self._plock:
                self.proc = None
            self._read_report()
            why = self.message or f'exit code {ev.get("code")}'
            self._set_state('failed' if ev.get('code') else 'off', f'The live server stopped: {why}')

    def stop(self, why='stopped') -> bool:
        """Stop streaming: the readers first, the streams put back, then the live server tidies CARLA up (in the
        background) and a server this panel started is stopped."""
        if self.state not in ('starting', 'streaming'):
            return False
        gen = self._gen
        self._cancel.set()
        was = self.state
        self._set_state('stopping', 'Stopping: CARLA is being tidied up…')
        self.poll_timer.stop()
        self._stop_monitor()
        self._stop_rig_page()
        self._restore_streams()
        self._close_fault_window()
        if was == 'starting':
            return True                               # the start's worker sees the cancel and reports stopped
        with self._plock:
            proc, port = self.proc, self.port

        def work():
            self._end_proc(proc, port)
            try:
                self.sig.stopped.emit(gen, why)
            except RuntimeError:
                pass
        threading.Thread(target=work, name='carla-live-stop', daemon=True).start()
        return True

    def _on_stopped(self, gen, why):
        if gen != self._gen:
            return
        with self._plock:
            self.proc = None
        self.port = None
        self.client = None
        self._stop_server_if_ours()
        self._read_report()
        rep = self.report or {}
        cams = rep.get('cameras') or {}
        txt = 'Stopped' + (f' ({why})' if why not in ('stopped', '') else '') + '; CARLA tidied up'
        if cams:
            txt += ' · served ' + ', '.join(f'{t} {c.get("served", 0)} frames' for t, c in cams.items())
        self._set_state('off', txt + '.')

    def _read_report(self):
        try:
            with open(os.path.join(self.folder, 'live.json'), encoding='utf-8') as fh:
                self.report = json.load(fh)
        except (OSError, ValueError):
            self.report = None

    def _stop_server_if_ours(self):
        if not self._server_started:
            return
        self._server_started = False
        if self._own_server is not None:
            srv, self._own_server = self._own_server, None
            threading.Thread(target=srv.stop, daemon=True).start()
        elif self.host is not None and hasattr(self.host, 'stop_server'):
            self.host.stop_server()

    def shutdown(self):
        """On quit: the live server and a server this panel started go with the app (it would stop by itself when the
        app is gone, through --parent-pid)."""
        self._cancel.set()
        with self._plock:
            proc, port = self.proc, self.port
        if proc is not None and proc.poll() is None:
            try:
                if port:
                    LV.LiveClient(port, timeout=1.0).stop()
                proc.wait(10.0)
            except Exception:                         # noqa: BLE001
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                    proc.wait(5.0)
                except Exception:                     # noqa: BLE001
                    pass
        if self._own_server is not None:
            self._own_server.stop(10.0)
            self._own_server = None

    # ------------------------------------------------------------------ the streams
    def _point_streams(self, tags):
        p = self.ctx.store.project
        urls = {t: LV.stream_url(self.port, t) for t in tags}
        saved = {t: p.streams.get(t) or None for t in tags}

        def fn(pj):
            for t, u in urls.items():
                pj.streams[t] = u
        self.ctx.store.edit('streams', fn, f'CARLA live: {", ".join(tags)} from the live server')
        self._saved, self._urls = saved, urls

    def _restore_streams(self):
        """Put the streams back (one undoable edit) where they still hold the live URLs."""
        urls, saved = self._urls, self._saved or {}
        self._saved, self._urls = None, {}
        p = self.ctx.store.project
        if not urls or p is not self._for_project:
            return
        todo = {t: saved.get(t) for t, u in urls.items() if p.streams.get(t) == u}
        if not todo:
            return

        def fn(pj):
            for t, old in todo.items():
                if old:
                    pj.streams[t] = old
                else:
                    pj.streams.pop(t, None)
        self.ctx.store.edit('streams', fn, 'CARLA live stopped: streams put back')

    def _is_ours(self, spec) -> bool:
        return bool(spec) and spec in self._urls.values()

    # ------------------------------------------------------------------ the Live rig page and the fault window
    def open_rig_page(self, faults=True) -> bool:
        w = self.ctx.window
        if w is None or not hasattr(w, 'set_page'):
            return False
        w.set_page('rig')
        pg = w.page('rig')
        if self.state == 'streaming' and hasattr(pg, 'start') and not getattr(pg, 'running', False):
            pg.start()
        if faults and self.state == 'streaming':
            self.show_fault_window()
        return True

    def _stop_rig_page(self):
        """The Live rig page stops reading the live streams before they go (else it retries a dead server)."""
        w = self.ctx.window
        pg = getattr(w, '_page_widgets', {}).get('rig') if w is not None else None
        if pg is None or not getattr(pg, 'running', False):
            return
        streams = self.ctx.store.project.streams
        if any(self._is_ours(streams.get(k, '')) for k in getattr(pg.monitor, 'streams', {})):
            pg.stop()
            pg._resume = False

    def show_fault_window(self):
        if self.state != 'streaming':
            self.ctx.msg('Faults can be injected while live.', 4000)
            return None
        if self._fault_window is None:
            self._fault_window = FaultWindow(self, self.ctx.window if isinstance(self.ctx.window, QWidget) else None)
        self._sync()
        self._fault_window.show()
        self._fault_window.raise_()
        return self._fault_window

    def _close_fault_window(self):
        if self._fault_window is not None:
            self._fault_window.hide()

    # ------------------------------------------------------------------ faults
    def inject(self, fault: dict):
        """Send a fault to the live server (in the background); the reply comes back as a message and in the table."""
        client = self.client
        if client is None or self.state != 'streaming':
            self.ctx.msg('Faults can be injected while live.', 4000)
            return False

        def work():
            try:
                res = client.fault(fault)
            except (OSError, LV.LiveError) as e:
                res = e
            try:
                self.sig.reply.emit('fault', res)
            except RuntimeError:
                pass
        threading.Thread(target=work, daemon=True).start()
        return True

    def clear_faults(self, tag=None):
        client = self.client
        if client is None or self.state != 'streaming':
            return False

        def work():
            try:
                res = client.clear(tag=tag)
            except (OSError, LV.LiveError) as e:
                res = e
            try:
                self.sig.reply.emit('clear', res)
            except RuntimeError:
                pass
        threading.Thread(target=work, daemon=True).start()
        return True

    def _on_reply(self, what, res):
        if isinstance(res, Exception):
            self.ctx.msg(f'CARLA live: {res}', 6000)
            return
        if what == 'fault':
            self.ctx.msg(f'CARLA live · {res["tag"]}: {res.get("text", res["kind"])}', 5000)
        else:
            self.ctx.msg('CARLA live: ' + (f'cleared {len(res)} fault(s)' if res else 'no fault to clear'), 4000)
        self._poll()

    # ------------------------------------------------------------------ status and health
    def _poll(self):
        client = self.client
        if client is None or self._polling:
            return
        self._polling = True
        gen = self._gen

        def work():
            try:
                s = client.status()
            except (OSError, LV.LiveError, ValueError):
                s = None
            try:
                self.sig.status.emit(gen, s)
            except RuntimeError:
                pass
        threading.Thread(target=work, daemon=True).start()

    def _on_status(self, gen, s):
        self._polling = False
        if gen != self._gen or self.state != 'streaming' or s is None:
            return
        self.live = s
        self._fetch_bodies()
        self._sync()

    def _fetch_bodies(self):
        """Fetch the live server's body masks (CARLA's own car per camera) not fetched yet, in the background."""
        client, live, gen = self.client, self.live or {}, self._gen
        if client is None:
            return
        for tag, c in (live.get('cameras') or {}).items():
            if not c.get('body_mask') or tag in self.carla_views or tag in self._fetching:
                continue
            self._fetching.add(tag)

            def work(tag=tag, size=tuple(c['size'])):
                try:
                    ev = CL.body_view(client.body(tag), size, ('carla', client.port, tag))
                except Exception as e:                # noqa: BLE001 (the project's body is used instead)
                    ev = e
                try:
                    self.sig.body.emit(gen, tag, ev)
                except RuntimeError:
                    pass
            threading.Thread(target=work, name='carla-live-body', daemon=True).start()

    def _on_body(self, gen, tag, ev):
        self._fetching.discard(tag)
        if gen != self._gen or isinstance(ev, Exception):
            return
        self.carla_views[tag] = ev
        s = self.monitor.streams.get(tag)
        if s is not None:
            s.drift, s.drift_checked = None, False          # measured again against CARLA's car

    def _start_monitor(self):
        if self.state != 'streaming' or self.monitor.running:
            return
        cams = [c for c in self._cams() if c.tag in self._urls]
        # the skew is judged against the live rig's own schedule: it steps at the cameras' rate and cannot honour a
        # stagger finer than its step, so what it delivers is its plan
        self.monitor.configure(self.ctx.store.project, cams, self._engine, planned=self.info.get('schedule') or {})
        self.monitor.start()
        self.health_timer.start()

    def _stop_monitor(self):
        self.health_timer.stop()
        if self.monitor.running:
            self.monitor.stop(wait=False)

    def _engine(self):
        w = self.ctx.window
        if w is not None and hasattr(w, '_shared_engine'):
            return w._shared_engine()
        from ...core.analysis import Engine
        return Engine()

    def summary(self) -> dict:
        """The live session in brief, for the assistant: state, real-time factor, per camera what is served, its
        latency, its faults and rig health's verdict and drift here (no URLs)."""
        live = self.live or {}
        h = self.health() or {}
        cams = {}
        for t, c in (live.get('cameras') or {}).items() if self.state == 'streaming' else ():
            hc = (h.get('cameras') or {}).get(t, {})
            cams[t] = {'served_fps': c.get('fps'), 'latency_ms': (c.get('latency_ms') or {}).get('median'),
                       'size': c.get('size'), 'render_size': c.get('render_size'), 'views': c.get('views'),
                       'faults': [f.get('text', f['kind']) for f in c.get('faults', [])],
                       'health': hc.get('status'), 'reasons': hc.get('reasons'), 'fps': hc.get('fps'),
                       'drift_px': hc.get('drift_px'), 'drift_against': hc.get('drift_against')}
        return {'state': self.state, 'message': self.status_line.text.text(),
                'realtime_factor': live.get('realtime_factor') if self.state == 'streaming' else None,
                'town': live.get('town'), 'tick_hz': live.get('tick_hz'), 'cameras': cams,
                'limits': LV.LIMITS_TEXT}

    def health(self) -> dict | None:
        """Rig health's snapshot of the live streams (JSON-ready) while the panel watches them, else None; each
        camera's drift says what it was measured against."""
        if not self.monitor.running:
            return None
        snap = rh.jsonable(self.monitor.snapshot())
        for t, c in snap['cameras'].items():
            c['drift_against'] = 'carla_body' if t in self.carla_views else 'project_body'
        return snap

    def _health_tick(self):
        if not self.monitor.running:
            self.health_timer.stop()
            return
        self.monitor.poll()
        self.snap = self.monitor.snapshot()
        self._fill()
        self._drift_step()

    def _drift_step(self):
        """One camera per call, on a worker thread: its planned view (when missing or the plan changed) and the
        alignment of its newest frame with it, as the Live rig page does it."""
        if self._drift_busy or not self.isVisible() or not self.monitor.running:
            return
        cams = [c for c in self._cams() if c.tag in self.monitor.streams]
        p = self.ctx.store.project
        if not cams or p.vehicle.mesh is None:
            return
        now = time.monotonic()
        todo = next((c for c in cams if c.tag not in self.carla_views and
                     (self.views.get(c.tag) is None or self.views[c.tag].key != rh.view_key(p, c))
                     and self.view_fail.get(c.tag) != rh.view_key(p, c)), None)
        if todo is None:
            if now - self._drift_t < 1.0 / len(cams):
                return
            todo = cams[self._rr % len(cams)]
            self._rr += 1
        frame = self.monitor.preview(todo.tag)
        ev = self.carla_views.get(todo.tag)
        key = ev.key if ev is not None else rh.view_key(p, todo)
        if ev is None:
            ev = self.views.get(todo.tag)
            if ev is not None and ev.key != key:
                ev = None
        snap = p.snapshot() if ev is None else None
        eng = self._engine() if ev is None else None
        cam = copy.deepcopy(todo)
        self._drift_busy = True
        self._drift_t = now

        def work():
            out = (cam.tag, None, None, False, key)
            try:
                v = ev or rh.expected_view(snap, eng, cam)
                out = (cam.tag, v, rh.alignment(frame, v) if frame is not None else None, frame is not None, key)
            except Exception:                         # noqa: BLE001 (no indicator for this camera)
                pass
            try:
                self.sig.view.emit(out)
            except RuntimeError:
                pass
        threading.Thread(target=work, name='carla-live-drift', daemon=True).start()

    def _on_view(self, out):
        self._drift_busy = False
        tag, ev, d, scored, key = out
        if ev is None:
            self.view_fail[tag] = key
            return
        if ev is not self.carla_views.get(tag):
            self.views[tag] = ev
        s = self.monitor.streams.get(tag)
        if scored and s is not None:
            s.drift, s.drift_checked = d, True

    def _fill(self):
        live = (self.live or {}).get('cameras', {}) if self.state == 'streaming' else {}
        health = (self.snap or {}).get('cameras', {}) if self.monitor.running else {}
        tags = list(live) or list(self._urls)
        self.table.setRowCount(len(tags))
        for i, t in enumerate(tags):
            L, h = live.get(t, {}), health.get(t, {})
            lat = (L.get('latency_ms') or {}).get('median')
            verdict = h.get('status', '—') if h else '—'
            reasons = '; '.join(h.get('reasons', [])) if h else ''
            drift = h.get('drift_px') if h else None
            vals = [(t, '', False, None),
                    (f'{L["fps"]:.1f}' if L.get('fps') is not None else '—', '', True, None),
                    (f'{lat:.0f} ms' if lat is not None else '—', '', True, None),
                    (verdict, reasons, False, STATUS.get(HEALTH.get(verdict, ''), None)),
                    (f'{h["fps"]:.1f}' if h.get('fps') else '—', '', True, None),
                    (f'{drift:.0f} px' if drift is not None else ('no edge' if h.get('drift_checked') else '—'),
                     'against CARLA\'s own car' if t in self.carla_views else 'against the project\'s body', True,
                     None),
                    ('; '.join(f.get('text', f['kind']) for f in L.get('faults', [])) or '', '', False, None)]
            for c, (text, tip, num, color) in enumerate(vals):
                self.table.setItem(i, c, _item(text, tip, num, color))

    # ------------------------------------------------------------------ state
    def _set_state(self, state, text):
        self.state = state
        sev = {'failed': 'serious', 'streaming': 'good'}.get(state)
        self.status_line.set(sev, text)
        self._sync()

    def _sync(self):
        on = self.state == 'streaming'
        self.b_start.setEnabled(not self.running and bool(self.chosen()))
        self.b_stop.setEnabled(self.state in ('starting', 'streaming'))
        self.b_page.setEnabled(on)
        self.b_faults.setEnabled(on)
        for b in self.cam_boxes.values():
            b.setEnabled(not self.running)
        self.scale.setEnabled(not self.running)
        self.serve.setEnabled(not self.running)
        live = self.live or {}
        tags = list((live.get('cameras') or {})) if on else []
        faults = live.get('faults', []) if on else []
        self.controls.sync(tags, faults)
        if self._fault_window is not None:
            self._fault_window.controls.sync(tags, faults)
        if on:
            rtf = live.get('realtime_factor')
            txt = (f'Streaming {len(tags)} camera{"s" * (len(tags) != 1)} at '
                   + (f'{rtf:.2f}× real time' if rtf is not None else 'the start')
                   + f' · {live.get("town", "")} · {self.info.get("render_mpx", 0):g} Mpx per frame set')
            if live.get('lag_resets'):
                txt += (' · CARLA is slower than real time: frames come less often (stream fewer cameras or render '
                        'smaller: CARLA renders every pinhole view in turn, a fisheye is 5)')
            self.status_line.set('good' if (rtf or 1.0) > 0.9 else 'warning', txt)
            if self._fault_window is not None:
                self._fault_window.status.setText(txt)
            self.where.setText(f'http://127.0.0.1:{self.port}/')
        elif not self.running:
            self.where.setText('')
        self._fill()

    # ------------------------------------------------------------------ the project
    def _on_store_changed(self, sections):
        if sections & {'variant', 'variants', 'library'} and not self.running:
            if [c.tag for c in self._cams()] != list(self.cam_boxes):
                self._rebuild_cams()
                self._sync()
        if 'streams' in sections and self.monitor.running:
            cams = [c for c in self._cams() if c.tag in self._urls]
            self.monitor.update(self.ctx.store.project, cams, self._engine)

    def _on_project_replaced(self, *_):
        if self.running:
            self._for_project = None                  # the new project's streams are not touched
            self._urls, self._saved = {}, None
            self.stop('the project was replaced')
        self.views, self.view_fail = {}, {}
        self.carla_views, self._fetching = {}, set()
        self.monitor = rh.RigMonitor()
        self._rebuild_cams()
        self._sync()

    def showEvent(self, e):
        super().showEvent(e)
        if self.state == 'streaming':
            self._start_monitor()

    def hideEvent(self, e):
        self._stop_monitor()                          # the Live rig page reads the streams while it is shown
        super().hideEvent(e)
