"""Calibrate ▸ Live rig: every camera's stream at once, with its health (frame rate, drops, exposure, sharpness, sync)
and a check that each camera still looks where the plan says it does; and the calibration coach (ui.live_calib_coach,
core.calib_coach): while a board is held in front of one camera, how firmly the views so far pin its lens (or its
lidar), where to hold the board next (a ghost board over the live image) and when to stop."""
from __future__ import annotations

import copy
import json
import os
import threading
import time

from PyQt6.QtCore import Qt, QTimer, QObject, QRectF, QPointF, pyqtSignal
from PyQt6.QtGui import QPainter, QPixmap, QPen, QColor, QFont, QPainterPath
from PyQt6.QtWidgets import (QWidget, QHBoxLayout, QVBoxLayout, QGridLayout, QSplitter, QCheckBox, QComboBox,
                             QFileDialog, QScrollArea, QFrame, QLineEdit, QSizePolicy, QMenu)

from ...core import capture as cap
from ...core import rig_health as rh
from ..theme import THEME, STATUS, mono, sans
from ..widgets import label, button, tool_button, Chip, SegmentedControl
from ..live_calib_coach import CoachPanel
from .install_ws import to_qimage

KINDS = [('none', 'Not set'), ('synthetic', 'Synthetic'), ('stream', 'Stream')]
MODES = [('health', 'Health'), ('coach', 'Calibration coach')]
DOT = {'ok': STATUS['good'], 'warning': STATUS['warning'], 'fault': STATUS['critical']}
VIDEO_FILTER = 'Video (*.mp4 *.mkv *.avi *.mov *.webm *.h264 *.h265 *.mjpeg);;All files (*)'


def _kind(spec):
    k = rh.describe(spec)
    return 'none' if not k else 'synthetic' if k == 'synthetic' else 'stream'


class _Signals(QObject):
    view = pyqtSignal(object)            # (tag, ExpectedView | None, drift px | None, scored)


# ---------------------------------------------------------------------- tiles
class RigTile(QWidget):
    clicked = pyqtSignal(str)
    double_clicked = pyqtSignal(str)
    HEAD = 22
    FOOT = 34

    def __init__(self, page, uid, tag):
        super().__init__()
        self.page = page
        self.uid = uid
        self.tag = tag
        self.setMinimumSize(200, 150)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._pix = None
        self._pix_key = None

    def _fit(self, w, h):
        top, bot = self.HEAD, self.FOOT
        s = min(self.width() / w, (self.height() - top - bot) / h)
        dw, dh = w * s, h * s
        return QRectF((self.width() - dw) / 2, top + (self.height() - top - bot - dh) / 2, dw, dh)

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), THEME.color('viewport'))
        cam = self.page.ctx.store.cam(self.uid)
        sel = self.page.selected == self.uid
        p.setPen(QPen(THEME.color('ink2' if sel else 'hair'), 1))
        p.drawRect(QRectF(0.5, 0.5, self.width() - 1, self.height() - 1))
        if cam is None:
            p.end()
            return
        st = self.page.cam_state(cam.tag)
        status = st.get('status', 'off') if self.page.running else 'stopped'
        # header: status dot, colour chip, tag, name, source
        dot = QColor(DOT[status]) if status in DOT else THEME.color('muted' if status == 'starting' else 'hair')
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(dot)
        p.drawEllipse(QPointF(self.width() - 12, 11), 4.5, 4.5)
        p.fillRect(QRectF(8, 7, 8, 8), THEME.cam(cam.slot))
        p.setFont(mono(11, QFont.Weight.DemiBold))
        p.setPen(THEME.color('ink'))
        fm = p.fontMetrics()
        p.drawText(QRectF(21, 0, 60, self.HEAD), Qt.AlignmentFlag.AlignVCenter, cam.tag)
        x = 21 + fm.horizontalAdvance(cam.tag) + 8
        p.setFont(sans(11))
        spec = self.page.ctx.store.project.streams.get(cam.tag, '')
        src = {'': 'no stream', 'synthetic': 'synthetic'}.get(rh.describe(spec), rh.shown(spec))
        if not self.page.running and spec:
            src += ' · stopped'
        p.setPen(THEME.color('muted'))
        p.drawText(QRectF(x, 0, self.width() - x - 24, self.HEAD), Qt.AlignmentFlag.AlignVCenter,
                   p.fontMetrics().elidedText(f'{cam.name} · {src}', Qt.TextElideMode.ElideMiddle,
                                              int(self.width() - x - 26)))
        img = self.page.monitor.preview(cam.tag)
        body = QRectF(0, self.HEAD, self.width(), self.height() - self.HEAD - self.FOOT)
        if img is None:
            p.setPen(THEME.color('muted'))
            msg = ('No stream set · right-click to set one' if not spec else
                   '; '.join(st.get('reasons', [])) or ('Connecting…' if self.page.running else 'Press Start'))
            p.drawText(body.adjusted(8, 0, -8, 0), Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap, msg)
        else:
            h, w = img.shape[:2]
            rect = self._fit(w, h)
            key = (id(img), int(rect.width()), int(rect.height()))
            if self._pix_key != key:
                self._pix = QPixmap.fromImage(to_qimage(img)).scaled(
                    int(rect.width()), int(rect.height()), Qt.AspectRatioMode.IgnoreAspectRatio,
                    Qt.TransformationMode.SmoothTransformation)
                self._pix_key = key
            p.drawPixmap(rect.topLeft(), self._pix)
            self._overlay(p, cam, rect)
        self._footer(p, st)
        p.end()

    def _overlay(self, p, cam, rect):
        if self.page.mode == 'coach':
            if self.uid == self.page.coach.cam_uid():
                self.page.coach.paint(p, rect)
            return
        ev = self.page.views.get(cam.tag)
        if ev is None:
            return
        sx, sy = rect.width() / ev.sensor[0], rect.height() / ev.sensor[1]

        def path(pts, closed):
            pp = QPainterPath()
            for k, (u, v) in enumerate(pts):
                pt = QPointF(rect.left() + u * sx, rect.top() + v * sy)
                pp.moveTo(pt) if k == 0 else pp.lineTo(pt)
            if closed:
                pp.closeSubpath()
            return pp
        p.setBrush(Qt.BrushStyle.NoBrush)
        if self.page.show_edge.isChecked():
            p.setPen(QPen(QColor(THEME.cam_hex(cam.slot)), 1.6, Qt.PenStyle.DashLine))
            for c in ev.contours:
                p.drawPath(path(c, False))
        if self.page.show_horizon.isChecked():
            p.setPen(QPen(QColor(255, 255, 255, 170), 1.0, Qt.PenStyle.DotLine))
            for hz in ev.horizon:
                p.drawPath(path(hz, False))

    def _footer(self, p, st):
        y = self.height() - self.FOOT
        p.setFont(mono(10))
        p.setPen(THEME.color('ink2'))
        line1, line2 = '', ''
        if st.get('frames'):
            size = st.get('size') or [0, 0]
            line1 = (f'{st["fps"]:5.1f}/{st["nominal_fps"]:g} fps · jitter {st["jitter_ms"]:.1f} ms · '
                     f'drop {st["dropped"]} · {size[0]}×{size[1]}')
            line2 = f'mean {st.get("mean", 0):.0f} · clip {st.get("clip_bright", 0) * 100:.0f} % · ' \
                    f'sharp {st.get("sharpness", 0):.0f}'
            if st.get('drift_px') is not None:
                line2 += f' · edge {st["drift_px"]:.0f} px'
            elif st.get('drift_checked'):
                line2 += ' · edge not found'
        p.drawText(QRectF(8, y + 1, self.width() - 16, 16), Qt.AlignmentFlag.AlignVCenter,
                   p.fontMetrics().elidedText(line1, Qt.TextElideMode.ElideRight, self.width() - 16))
        reasons = st.get('reasons') or []
        status = st.get('status') if self.page.running else 'stopped'
        if reasons and status in ('warning', 'fault') and st.get('frames'):
            p.setPen(QColor(DOT[status]))
            line2 = '; '.join(reasons)
        p.drawText(QRectF(8, y + 17, self.width() - 16, 16), Qt.AlignmentFlag.AlignVCenter,
                   p.fontMetrics().elidedText(line2, Qt.TextElideMode.ElideRight, self.width() - 16))

    def mousePressEvent(self, e):
        self.clicked.emit(self.uid)

    def mouseDoubleClickEvent(self, e):
        self.double_clicked.emit(self.uid)

    def contextMenuEvent(self, e):
        m = QMenu(self)
        m.addAction('Stream source…', lambda: self.page.edit_source(self.uid))
        m.addAction('Synthetic', lambda: self.page.set_kind(self.uid, 'synthetic'))
        m.addAction('No stream', lambda: self.page.set_kind(self.uid, 'none'))
        m.addSeparator()
        m.addAction('Show all' if self.page.single else 'Enlarge', lambda: self.page.toggle_single(self.uid))
        m.exec(e.globalPos())


# ---------------------------------------------------------------------- page
class RigPage(QWidget):
    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        self.setObjectName('base')
        self.monitor = rh.RigMonitor()
        self.snap = None                  # the monitor's last snapshot
        self.views = {}                   # tag -> rig_health.ExpectedView of the current plan
        self.view_fail = {}               # tag -> view key that could not be computed
        self.rows = {}
        self.tiles = {}
        self.selected = None
        self.single = False
        self._resume = False              # streaming when the page was hidden: start again when shown
        self._drift_busy = False
        self._drift_t = 0.0
        self._rr = 0
        self.mode = 'health'              # health | coach
        self.sig = _Signals()
        self.sig.view.connect(self._on_view)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        split = QSplitter(Qt.Orientation.Horizontal)
        split.setHandleWidth(2)
        split.addWidget(self._build_left())
        split.addWidget(self._build_centre())
        split.setSizes([340, 1400])
        split.setStretchFactor(1, 1)
        lay.addWidget(split)

        self.timer = QTimer(self)
        self.timer.setInterval(100)
        self.timer.timeout.connect(self._tick)
        ctx.store.changed.connect(self._on_store_changed)
        ctx.store.project_replaced.connect(self._on_project_replaced)
        self._rebuild()
        self.destroyed.connect(self.coach.shutdown)

    # ------------------------------------------------------------------ layout
    def _build_left(self):
        left = QWidget()
        left.setObjectName('panel')
        ll = QVBoxLayout(left)
        ll.setContentsMargins(10, 8, 10, 10)
        ll.setSpacing(4)
        ll.addWidget(label('Live rig', bold=True))
        ll.addWidget(label('Every camera\'s stream at once: frame rate, drops, exposure, sharpness and how far apart the '
                           'cameras capture. The dashed outline is where the plan puts the body in each image; a '
                           'camera that has moved shows its body edge somewhere else.', muted=True, small=True,
                           wrap=True))
        self.mode_seg = SegmentedControl(MODES, 'health')
        self.mode_seg.changed.connect(self.set_mode)
        self.mode_seg.setToolTip('Health: every stream\'s statistics. Calibration coach: guide a board held in front '
                                 'of one camera until its calibration is pinned')
        ll.addWidget(self.mode_seg)
        ll.addSpacing(4)
        ll.addWidget(label('Streams', bold=True))
        self.src_grid = QGridLayout()
        self.src_grid.setHorizontalSpacing(6)
        self.src_grid.setVerticalSpacing(3)
        self.src_grid.setColumnStretch(2, 1)
        ll.addLayout(self.src_grid)
        ll.addWidget(label('Stream: rtsp://…, http://…, /dev/video0, a GStreamer pipeline or a video file. Synthetic: '
                           'the parking scene through the camera\'s lens at the module\'s frame rate (or type one). '
                           'Saved with the project and shared with the installation check.', muted=True, small=True,
                           wrap=True))
        ll.addSpacing(6)
        self.health_box = QWidget()
        hl = QVBoxLayout(self.health_box)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.setSpacing(4)
        hl.addWidget(label('Overlay', bold=True))
        self.show_edge = QCheckBox('Planned body outline')
        self.show_edge.setChecked(True)
        self.show_horizon = QCheckBox('Planned horizon')
        self.show_horizon.setChecked(True)
        self.show_score = QCheckBox('Edge alignment indicator')
        self.show_score.setChecked(True)
        self.show_score.setToolTip('Median distance from the planned body edge to the nearest edge in the image, in sensor '
                                   'pixels. An indicator that a camera moved, not a measurement: the installation check '
                                   'measures the pose.')
        for cb in (self.show_edge, self.show_horizon, self.show_score):
            cb.toggled.connect(self._overlay_changed)
            hl.addWidget(cb)
        hl.addWidget(label('Indicator only: a high value means the body edge is not where it was planned (a moved camera, '
                           'or a scene with few edges). Measure with the installation check.', muted=True, small=True,
                           wrap=True))
        ll.addWidget(self.health_box)
        self.coach = CoachPanel(self)
        self.coach.setVisible(False)
        ll.addWidget(self.coach)
        ll.addStretch(1)
        self.status = label('Stopped', muted=True, small=True, wrap=True)
        ll.addWidget(self.status)
        g = QGridLayout()
        g.setHorizontalSpacing(6)
        g.setVerticalSpacing(4)
        self.b_save = button('Save frames…', None, 'file', 'Write the newest frame of every camera as <tag>.png and the '
                                                           'statistics as rig_health.json')
        self.b_save.clicked.connect(lambda: self.save_frames())
        self.b_check = button('Check installation', None, 'target', 'Freeze the newest frame of every camera and open '
                                                                  'the installation check with them')
        self.b_check.clicked.connect(self.check_installation)
        self.b_start = button('Start', 'primary', 'play', 'Read every camera\'s stream')
        self.b_start.clicked.connect(lambda: self.stop() if self.running else self.start())
        g.addWidget(self.b_save, 0, 0)
        g.addWidget(self.b_check, 0, 1)
        g.addWidget(self.b_start, 1, 0, 1, 2)
        ll.addLayout(g)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(left)
        scroll.setMinimumWidth(310)
        return scroll

    def _build_centre(self):
        w = QWidget()
        w.setObjectName('panel')
        cl = QVBoxLayout(w)
        cl.setContentsMargins(8, 4, 8, 6)
        cl.setSpacing(4)
        h = QHBoxLayout()
        self.summary = label('', mono_=True)
        h.addWidget(self.summary)
        h.addStretch(1)
        self.clock = label('sync from software timestamps (arrival at Vantage), not sensor time', muted=True, small=True)
        self.clock.setToolTip(rh.CLOCK_NOTE)
        h.addWidget(self.clock)
        cl.addLayout(h)
        self.tile_host = QWidget()
        self.tile_grid = QGridLayout(self.tile_host)
        self.tile_grid.setContentsMargins(0, 0, 0, 0)
        self.tile_grid.setSpacing(6)
        cl.addWidget(self.tile_host, 1)
        return w

    # ------------------------------------------------------------------ cameras / sources
    @property
    def running(self):
        return self.monitor.running

    def _cams(self):
        return [c for c in self.ctx.store.variant.cameras if c.visible]

    def _engine(self):
        w = self.ctx.window
        if w is not None and hasattr(w, '_shared_engine'):
            return w._shared_engine()
        from ...core.analysis import Engine
        return Engine()

    def cam_state(self, tag) -> dict:
        if self.snap is not None and tag in self.snap['cameras']:
            return self.snap['cameras'][tag]
        return {'status': 'off' if not self.ctx.store.project.streams.get(tag) else 'stopped'}

    def _on_store_changed(self, sections):
        if sections & {'variant', 'variants', 'streams', 'library'}:
            if [c.uid for c in self._cams()] != list(self.tiles) or \
                    [c.tag for c in self._cams()] != [t.tag for t in self.tiles.values()]:
                self._rebuild()
            else:
                for uid in self.rows:
                    self._sync_row(uid)
            if self.running:
                self._judge_against()
                self.monitor.update(self.ctx.store.project, self._cams(), self._engine)
                self.coach.install_technician()
            self._repaint()
        self.coach.on_store_changed(sections)

    def _on_project_replaced(self, *_):
        self.stop()
        self._resume = False
        self.coach.remove_technician()
        self.monitor = rh.RigMonitor()
        self.snap = None
        self.views, self.view_fail = {}, {}
        self._rebuild()
        self.coach.sync_fields()
        if self.mode == 'coach':
            self.coach.rebuild()

    def _rebuild(self):
        cams = self._cams()
        while self.src_grid.count():
            w = self.src_grid.takeAt(0).widget()
            if w:
                w.deleteLater()
        self.rows = {}
        for i, c in enumerate(cams):
            chip = Chip(c.slot, c.tag)
            combo = QComboBox()
            for k, t in KINDS:
                combo.addItem(t, k)
            combo.setFixedWidth(92)
            combo.activated.connect(lambda _=0, u=c.uid, cb=combo: self.set_kind(u, cb.currentData()))
            edit = QLineEdit()
            edit.setFont(mono(11))
            edit.editingFinished.connect(lambda u=c.uid, ed=edit: self._edit_done(u, ed.text()))
            browse = tool_button('folder', 'Choose a video file')
            browse.clicked.connect(lambda _=False, u=c.uid: self.browse(u))
            self.src_grid.addWidget(chip, i, 0)
            self.src_grid.addWidget(combo, i, 1)
            self.src_grid.addWidget(edit, i, 2)
            self.src_grid.addWidget(browse, i, 3)
            self.rows[c.uid] = (combo, edit, browse)
            self._sync_row(c.uid)
        for t in self.tiles.values():
            t.setParent(None)
            t.deleteLater()
        self.tiles = {}
        for c in cams:
            t = RigTile(self, c.uid, c.tag)
            t.clicked.connect(self.select)
            t.double_clicked.connect(self.toggle_single)
            self.tiles[c.uid] = t
        if self.selected not in self.tiles:
            self.selected = cams[0].uid if cams else None
            self.single = False
        self._layout_tiles()
        self._summary()

    def _layout_tiles(self):
        while self.tile_grid.count():
            self.tile_grid.takeAt(0)
        tiles = list(self.tiles.values())
        cu = self.coach.cam_uid() if self.mode == 'coach' else None
        if cu in self.tiles:
            for t in tiles:
                t.setVisible(t.uid == cu)
            self.tile_grid.addWidget(self.tiles[cu], 0, 0)
            return
        if self.single and self.selected in self.tiles:
            for t in tiles:
                t.setVisible(t.uid == self.selected)
            self.tile_grid.addWidget(self.tiles[self.selected], 0, 0)
            return
        n = len(tiles)
        cols = 1 if n <= 1 else 2 if n <= 4 else 3 if n <= 9 else 4
        for i, t in enumerate(tiles):
            t.setVisible(True)
            self.tile_grid.addWidget(t, i // cols, i % cols)

    def _sync_row(self, uid):
        cam = self.ctx.store.cam(uid)
        if cam is None or uid not in self.rows:
            return
        combo, edit, browse = self.rows[uid]
        spec = self.ctx.store.project.streams.get(cam.tag, '')
        kind = _kind(spec)
        combo.setCurrentIndex([k for k, _ in KINDS].index(kind))
        edit.blockSignals(True)
        edit.setEnabled(kind != 'none')
        browse.setVisible(kind == 'stream')
        if kind == 'synthetic':
            edit.setText(spec.split(':', 1)[1] if ':' in spec else '')
            edit.setPlaceholderText(f'{self.ctx.store.project.module_for(cam).fps:g} fps (the module\'s)')
        elif kind == 'stream':
            edit.setText(spec)
            edit.setPlaceholderText('rtsp://…, /dev/video0, pipeline, file')
        else:
            edit.setText('')
            edit.setPlaceholderText('')
        edit.blockSignals(False)
        edit.setToolTip(spec)

    def _set_stream(self, tag, spec):
        spec = (spec or '').strip()
        p = self.ctx.store.project
        if p.streams.get(tag, '') == spec:
            return

        def fn(p):
            if spec:
                p.streams[tag] = spec
            else:
                p.streams.pop(tag, None)
        self.ctx.store.edit('streams', fn, f'Stream of {tag}: {rh.shown(spec)}' if spec else f'Removed the stream of {tag}')

    def set_kind(self, uid, kind):
        cam = self.ctx.store.cam(uid)
        if cam is None or not kind:
            return
        cur = self.ctx.store.project.streams.get(cam.tag, '')
        if kind == _kind(cur):
            return
        if kind == 'none':
            self._set_stream(cam.tag, '')
        elif kind == 'synthetic':
            self._set_stream(cam.tag, 'synthetic')
        else:
            self._set_stream(cam.tag, cap.default_spec(cam.port))
        self._sync_row(uid)

    def _edit_done(self, uid, text):
        cam = self.ctx.store.cam(uid)
        if cam is None:
            return
        text = text.strip()
        kind = _kind(self.ctx.store.project.streams.get(cam.tag, ''))
        if kind == 'synthetic':
            try:
                fps = float(text) if text else 0.0
            except ValueError:
                self.ctx.msg(f'{cam.tag}: type a frame rate for the synthetic stream, e.g. 30', 5000)
                self._sync_row(uid)
                return
            self._set_stream(cam.tag, f'synthetic:{fps:g}' if fps > 0 else 'synthetic')
        elif kind == 'stream' and text:
            self._set_stream(cam.tag, text)

    def edit_source(self, uid):
        """Per-tile source editing: give the camera a stream if it has none and put the cursor in its field."""
        cam = self.ctx.store.cam(uid)
        if cam is None or uid not in self.rows:
            return
        if not self.ctx.store.project.streams.get(cam.tag):
            self.set_kind(uid, 'stream')
        edit = self.rows[uid][1]
        edit.setFocus()
        edit.selectAll()

    def browse(self, uid):
        cam = self.ctx.store.cam(uid)
        if cam is None:
            return
        cur = self.ctx.store.project.streams.get(cam.tag, '')
        path, _ = QFileDialog.getOpenFileName(self, f'Video for {cam.tag}', cur if os.path.exists(cur) else '',
                                              VIDEO_FILTER)
        if path:
            self._set_stream(cam.tag, path)

    # ------------------------------------------------------------------ running
    def _planned(self):
        """What the capture-time skew is judged against (rig_health.RigMonitor.configure): the CARLA live rig's own
        schedule (rig_health.schedule_offsets) when every camera with a stream is the live rig's, since it steps at the
        cameras' rate and fires them as that schedule says whatever the plan's stagger; else the project's sync plan."""
        w = self.ctx.window
        sim = getattr(w, '_page_widgets', {}).get('sim') if w is not None else None
        panel = getattr(sim, 'live', None)
        if panel is None or panel.state != 'streaming':
            return 'plan'
        streams = self.ctx.store.project.streams
        urls = getattr(panel, '_urls', {}) or {}
        tags = [c.tag for c in self._cams() if streams.get(c.tag)]
        if tags and all(urls.get(t) and streams.get(t) == urls[t] for t in tags):
            return dict((getattr(panel, 'info', None) or {}).get('schedule') or {})
        return 'plan'

    def _judge_against(self):
        """Follow _planned while streaming (a stream pointed at or away from the live rig), without a restart."""
        pl = self._planned()
        if pl == 'plan':
            self.monitor.plan_mode = 'plan'                  # the monitor's update reads the plan again
        else:
            self.monitor.plan_mode, self.monitor.planned = 'given', pl

    def start(self):
        cams = self._cams()
        self.monitor.configure(self.ctx.store.project, cams, self._engine, planned=self._planned())
        if not self.monitor.streams:
            self.ctx.msg('No camera has a stream yet: set one per camera (Synthetic, a URL, a device or a video file)',
                         6000)
            self.snap = self.monitor.snapshot()
            self._summary()
            self._repaint()
            return
        self.monitor.start()
        if self.mode == 'coach':
            self.coach.install_technician()
        self.snap = self.monitor.snapshot()
        self.timer.start()
        self._sync_buttons()
        self._summary()
        self._repaint()

    def _sync_tip(self, r) -> str:
        """The sync figures behind the summary: what the skew is judged against and each camera's offset to the
        reference camera, its planned offset and its skew."""
        if r.get('sync_reference') is None:
            return 'Sync: fewer than two cameras streaming'
        against = ('the CARLA live rig\'s schedule' if self.monitor.plan_mode == 'given' and self.monitor.planned else
                   'the sync plan\'s trigger offsets') if r.get('sync_against') == 'plan' else \
            'no plan (every camera expected to fire together)'
        lines = [f'Sync skew judged against {against}; reference {r["sync_reference"]}']
        for k, c in (self.snap or {}).get('cameras', {}).items():
            if c.get('sync_skew_ms') is None or k == r['sync_reference']:
                continue
            pl = c.get('sync_planned_ms')
            lines.append(f'{k}: offset {c["sync_offset_ms"]:+.1f} ms'
                         + (f', planned {pl:+.1f} ms' if pl is not None else '')
                         + f', skew {c["sync_skew_ms"]:+.1f} ms')
        return '\n'.join(lines)

    def stop(self):
        """Stop every reader without waiting for it (a reader blocked in a network read exits when the read returns)."""
        self.timer.stop()
        was = self.monitor.running
        self.monitor.stop(wait=False)
        if was:
            self.snap = self.monitor.snapshot()
        self._sync_buttons()
        self._summary()
        self._repaint()

    def health(self):
        """The current statistics (JSON-ready) while streaming, else None (for the assistant)."""
        if not self.running:
            return None
        snap = rh.jsonable(self.monitor.snapshot())
        live = self._live_views()
        for t, c in snap['cameras'].items():           # what the drift indicator compared the frame with
            c['drift_against'] = 'carla_body' if t in live else 'project_body'
        return snap

    def _tick(self):
        if not self.monitor.running:
            self.timer.stop()
            return
        self.monitor.poll()
        self.snap = self.monitor.snapshot()
        self._repaint()
        self._summary()
        if self.mode == 'coach':
            self.coach.tick()
        else:
            self._drift_step()

    def _repaint(self):
        for t in self.tiles.values():
            t.update()

    def _sync_buttons(self):
        on = self.running
        self.b_start.setText('Stop' if on else 'Start')
        from .. import icons
        self.b_start.setIcon(icons.icon('stop' if on else 'play', 12))

    def _summary(self):
        cams = self._cams()
        streams = self.ctx.store.project.streams
        n_set = sum(1 for c in cams if streams.get(c.tag))
        if not self.running or self.snap is None:
            self.summary.setText(f'Stopped · {n_set} of {len(cams)} cameras have a stream')
            self.summary.setToolTip('')
            self.status.setText('Stopped' if n_set else 'Set a stream per camera, then Start.')
            return
        r = self.snap['rig']
        parts = [f'{r["streaming"]} of {r["cameras"]} up']
        if r['worst_fps'] is not None:
            parts.append(f'worst {r["worst_fps"]:.1f} fps ({r["worst_fps_camera"]})')
        if r['sync_skew_ms'] is not None:
            parts.append(f'sync skew {r["sync_skew_ms"]:.0f} ms ({r["sync_skew_camera"]})')
        parts.append(f'{r["faults"]} fault{"s" if r["faults"] != 1 else ""}')
        if r['warnings']:
            parts.append(f'{r["warnings"]} warning{"s" if r["warnings"] != 1 else ""}')
        self.summary.setText(' · '.join(parts))
        self.summary.setToolTip(self._sync_tip(r))
        self.status.setText(f'Streaming for {r["seconds"]:.0f} s')

    # ------------------------------------------------------------------ drift overlay
    def _overlay_changed(self, *_):
        if not self.show_score.isChecked():
            for s in self.monitor.streams.values():
                s.drift, s.drift_checked = None, False
        self._drift_step()
        self._repaint()

    def _live_views(self) -> dict:
        """tag -> rig_health.ExpectedView of CARLA's own car, for the cameras whose stream is the CARLA live rig's
        (Simulate ▸ CARLA ▸ Live rig): CARLA renders its own car, not the project's body, so the drift indicator is
        measured against what the camera sees of that car at its planned mount (the live server renders it once)."""
        w = self.ctx.window
        sim = getattr(w, '_page_widgets', {}).get('sim') if w is not None else None
        panel = getattr(sim, 'live', None)
        if panel is None or panel.state != 'streaming':
            return {}
        streams = self.ctx.store.project.streams
        urls = getattr(panel, '_urls', {}) or {}
        return {t: ev for t, ev in dict(panel.carla_views).items() if urls.get(t) and streams.get(t) == urls[t]}

    def _key_of(self, p, cam, live) -> tuple:
        ev = live.get(cam.tag)
        return ev.key if ev is not None else rh.view_key(p, cam)

    def _drift_step(self):
        """One camera per call, on a worker thread: its planned view (when missing or the plan changed) and, while
        streaming, the alignment of its newest preview with it. A camera streamed by the CARLA live rig is compared
        with CARLA's own car instead (its body mask from the live server)."""
        if self._drift_busy or not self.isVisible():
            return
        if not (self.show_edge.isChecked() or self.show_horizon.isChecked() or self.show_score.isChecked()):
            return
        cams = self._cams()
        if not cams:
            return
        p = self.ctx.store.project
        live = self._live_views()
        if p.vehicle.mesh is None and not live:
            return
        todo = None
        for c in cams:
            if c.tag not in live and p.vehicle.mesh is None:
                continue
            k = self._key_of(p, c, live)
            ev = self.views.get(c.tag)
            if (ev is None or ev.key != k) and self.view_fail.get(c.tag) != k:
                todo = c
                break
        score = self.show_score.isChecked() and self.running
        now = time.monotonic()
        if todo is None:
            if not score or now - self._drift_t < 1.0 / len(cams):
                return
            todo = cams[self._rr % len(cams)]
            self._rr += 1
            if todo.tag not in live and p.vehicle.mesh is None:
                return
        frame = self.monitor.preview(todo.tag) if score else None
        key = self._key_of(p, todo, live)
        ev = live.get(todo.tag) or self.views.get(todo.tag)
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
                d = rh.alignment(frame, v) if frame is not None else None
                out = (cam.tag, v, d, frame is not None, key)
            except Exception:
                pass
            try:
                self.sig.view.emit(out)
            except RuntimeError:            # the page was deleted meanwhile
                pass
        threading.Thread(target=work, name='rig-drift', daemon=True).start()

    def _on_view(self, out):
        self._drift_busy = False
        tag, ev, d, scored, key = out
        if ev is None:
            self.view_fail[tag] = key
            return
        if ev.key != key:
            return
        self.views[tag] = ev
        s = self.monitor.streams.get(tag)
        if scored and s is not None and self.show_score.isChecked():
            s.drift, s.drift_checked = d, True
        for t in self.tiles.values():
            if t.tag == tag:
                t.update()

    # ------------------------------------------------------------------ calibration coach
    def set_mode(self, mode):
        """'health' (every stream) or 'coach' (the calibration coach on one camera)."""
        if mode not in dict(MODES) or mode == self.mode:
            self.mode_seg.set_value(self.mode)
            return
        self.mode = mode
        self.mode_seg.set_value(mode)
        coach = mode == 'coach'
        self.health_box.setVisible(not coach)
        self.coach.setVisible(coach)
        if coach:
            self.coach.activate()
        else:
            self.coach.deactivate()
        self._layout_tiles()
        self._repaint()

    def coach_camera_changed(self):
        self._layout_tiles()
        self._repaint()

    def repaint_coach(self):
        uid = self.coach.cam_uid()
        if uid in self.tiles:
            self.tiles[uid].update()

    def coach_state(self):
        """The coach's state (JSON-ready) while it is on, else None (for the assistant)."""
        if self.mode != 'coach' or not self.coach.state.get('built'):
            return None
        from ...core import calib_coach as CC
        st = self.coach.state
        adv = st.get('advice')
        return CC._clean({'camera': st['cam_tag'], 'lidar': st.get('lidar'), 'views': st['views'],
                          'status': st.get('status'), 'next': adv.text if adv is not None else None,
                          'next_vehicle': adv.text_vehicle if adv is not None else None})

    # ------------------------------------------------------------------ selection
    def select(self, uid):
        self.selected = uid
        self._repaint()

    def toggle_single(self, uid):
        self.selected = uid
        self.single = not self.single
        self._layout_tiles()
        self._repaint()

    # ------------------------------------------------------------------ frames out
    def _frames(self):
        """uid -> newest full-resolution frame of every camera that has one."""
        out = {}
        for c in self._cams():
            f = self.monitor.latest(c.tag)
            if f is not None:
                out[c.uid] = f
        return out

    def save_frames(self, folder=None):
        frames = self._frames()
        if not frames:
            self.ctx.msg('No frames yet: start the streams first', 5000)
            return []
        if not folder:
            folder = QFileDialog.getExistingDirectory(self, 'Folder for the frames (one <tag>.png per camera)')
        if not folder:
            return []
        os.makedirs(folder, exist_ok=True)
        paths, files = [], {}
        for uid, f in frames.items():
            tag = self.ctx.store.cam(uid).tag
            path = os.path.join(folder, f'{tag}.png')
            cap.save_image(path, f)
            paths.append(path)
            files[tag] = os.path.basename(path)
        doc = rh.jsonable(self.monitor.snapshot())
        doc.update(project=self.ctx.store.project.name, variant=self.ctx.store.variant.name, frames=files,
                   time=time.strftime('%Y-%m-%dT%H:%M:%S'), settings=self.monitor.settings.to_dict())
        jp = os.path.join(folder, 'rig_health.json')
        with open(jp, 'w', encoding='utf-8') as fh:
            json.dump(doc, fh, indent=1, ensure_ascii=False)
        paths.append(jp)
        self.ctx.msg(f'Saved {len(frames)} frame(s) and rig_health.json to {folder}', 6000)
        return paths

    def check_installation(self):
        frames = self._frames()
        if not frames:
            self.ctx.msg('No frames yet: start the streams first', 5000)
            return False
        w = self.ctx.window
        if w is None:
            return False
        ip = w.page('install')
        ip.use_frames(frames)
        w.set_page('install')
        self.ctx.msg(f'Installation check: using {len(frames)} frame(s) from the live rig · Run check', 6000)
        return True

    # ------------------------------------------------------------------ visibility
    def activate(self):
        self._drift_step()

    def showEvent(self, e):
        super().showEvent(e)
        if self._resume:
            self._resume = False
            self.start()

    def hideEvent(self, e):
        # free the cameras for other programs (and the installation check) while the page is not shown
        if self.running:
            self._resume = True
            self.stop()
        super().hideEvent(e)
