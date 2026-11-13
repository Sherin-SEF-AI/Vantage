"""Calibrate ▸ Install guide: a rig fitted to a car, taken from its live streams to a measured and verified
calibration and coverage, stage by stage (core.install_guide), with Claude as a coach (assistant.install_coach).

Left: the eight stages, each closed by a test, and what to do in the current one. Centre: every camera live, with what
the stage needs drawn over it: the port check's brightness, the planned and the measured body outline with the turn and
the shim that corrects the aim, the boards the plan expects (yellow) and the corners found (green), the corners'
residuals after the solve. Right: the ground coverage of the measured rig, robust (green), uncertain (amber) or blind
(red), and the coach. Heavy work (aim, detection, the solve) runs on one worker thread, a camera at a time.

Sources: Live reads project.streams (as Calibrate ▸ Live rig); Rehearse renders every camera's view of the planned
floor and boards from a pose a little off its plan, to practise the guide (and test it) without a car."""
from __future__ import annotations

import json
import math
import os
import threading
import time

import numpy as np
from PyQt6.QtCore import Qt, QTimer, QObject, QRectF, QPointF, pyqtSignal
from PyQt6.QtGui import QPainter, QPixmap, QPen, QColor, QFont, QImage, QPolygonF
from PyQt6.QtWidgets import (QWidget, QHBoxLayout, QVBoxLayout, QGridLayout, QSplitter, QCheckBox,
                             QFrame, QLineEdit, QSizePolicy, QTextBrowser, QFileDialog)

from ...core import install_guide as G
from ...core import rig_health as rh
from ...core.project import Variant, new_uid
from ..theme import THEME, STATUS, mono, sans
from ..widgets import label, button, SegmentedControl, ScrubSpinBox
from .install_ws import to_qimage

SOURCES = [('live', 'Live'), ('rehearse', 'Rehearse')]
REHEARSE_FPS = 10.0
STAGE_HELP = {
    'connect': 'Every camera streaming at its frame rate. Streams are set in Calibrate ▸ Live rig (a V4L2 device '
               'or a GStreamer pipeline per camera).',
    'identify': 'Cover one lens at a time with your hand, in the order asked: the stream that goes dark is that '
                'camera\'s. A swapped GMSL cable shows here, before anything is measured.',
    'lens': 'Each camera\'s own lens calibration (Calibrate ▸ Calibration coach, or Lens calibration). Without one the '
            'library model of the lens is used: accept that only for a lens of the same batch.',
    'aim': 'Each camera\'s turn on its mount from the car\'s own outline in its image (dashed: where the plan puts '
           'the outline; green: where it is). Correct the mount until every axis is within tolerance.',
    'boards': 'Lay the boards where the plan puts them (Plan boards makes a plan that meets the requirement; Export '
              'layout lists every board\'s place). Yellow: where each camera should see a board; green: corners found.',
    'solve': 'Every camera\'s pose from the boards, all cameras and boards in one fit, with its 1σ per axis. A camera '
             'meets the requirement when σ × safety is within it on every axis.',
    'coverage': 'The ground the measured rig sees: robust (green) where a camera sees it with 3σ to spare, uncertain '
                '(amber) near an image edge, blind (red). Compared with the plan.',
    'verify': 'Tests the solve did not use: the check board seen by two cameras must land on the same floor points; '
              'the seams; the fit\'s residuals; the coverage.',
}


def _fmt(v, dp=2, unit=''):
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return '—'
    return f'{v:.{dp}f}{unit}'.replace('-', '−')


class _Signals(QObject):
    done = pyqtSignal(str, object)          # job name, result
    coach = pyqtSignal(object)


# ---------------------------------------------------------------------- sources
class GuideMonitor(rh.RigMonitor):
    """rig_health's monitor; in rehearsal every camera's source renders its view of the planned floor and boards from
    a pose a little off the plan (rehearsal: tag -> (R, position))."""

    def __init__(self):
        super().__init__()
        self.rehearsal = {}
        self.engine = None

    def _make(self, s, project, cam, engine_factory):
        if s.spec != 'rehearse':
            return super()._make(s, project, cam, engine_factory)

        def on_frame(frame, t, s=s):
            s.stats.add(frame, t)
            if t - s.preview_t >= 1.0 / rh.PREVIEW_HZ:
                import cv2
                h, w = frame.shape[:2]
                pw = min(rh.PREVIEW_W, w)
                pv = cv2.resize(frame, (pw, max(1, round(h * pw / w))), interpolation=cv2.INTER_AREA)
                with s.plock:
                    s.preview, s.preview_t = pv, t
        pose = self.rehearsal.get(cam.tag)
        snap = project.snapshot()
        eng = self.engine
        mod = project.module_for(cam)

        def render(cam=cam, pose=pose):
            img = G.render_view(snap, eng, cam, pose, w=mod.width_px, seed=cam.slot, noise=0.0)
            import cv2
            return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) if img.ndim == 2 else img
        fps = min(REHEARSE_FPS, mod.fps or REHEARSE_FPS)
        s.stats.nominal = fps                       # judged against the rehearsal's rate, not the camera's
        return rh.SyntheticSource(render, fps, (mod.width_px, mod.height_px), noise=2, spec='rehearse',
                                  on_frame=on_frame)


# ---------------------------------------------------------------------- tiles
class GuideTile(QWidget):
    HEAD = 22
    FOOT = 38
    clicked = pyqtSignal(str)

    def __init__(self, page, tag):
        super().__init__()
        self.page = page
        self.tag = tag
        self.setMinimumSize(220, 170)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._pix = None
        self._pix_key = None

    def _fit(self, w, h):
        s = min(self.width() / w, (self.height() - self.HEAD - self.FOOT) / h)
        dw, dh = w * s, h * s
        return QRectF((self.width() - dw) / 2, self.HEAD + (self.height() - self.HEAD - self.FOOT - dh) / 2, dw, dh)

    def mousePressEvent(self, e):
        self.clicked.emit(self.tag)

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), THEME.color('viewport'))
        page = self.page
        sel = page.selected == self.tag
        p.setPen(QPen(THEME.color('ink2' if sel else 'hair'), 1))
        p.drawRect(QRectF(0.5, 0.5, self.width() - 1, self.height() - 1))
        cam = page.session.cam(self.tag) if page.session else None
        st = page.cam_health(self.tag)
        status = st.get('status', 'off') if page.running else 'stopped'
        dot = {'ok': STATUS['good'], 'warning': STATUS['warning'], 'fault': STATUS['critical']}.get(status)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(dot) if dot else THEME.color('hair'))
        p.drawEllipse(QPointF(self.width() - 12, 11), 4.5, 4.5)
        if cam is not None:
            p.fillRect(QRectF(8, 7, 8, 8), THEME.cam(cam.slot))
        p.setFont(mono(11, QFont.Weight.DemiBold))
        p.setPen(THEME.color('ink'))
        p.drawText(QRectF(21, 0, 60, self.HEAD), Qt.AlignmentFlag.AlignVCenter, self.tag)
        p.setFont(sans(11))
        p.setPen(THEME.color('muted'))
        fps = st.get('fps')
        head = (cam.name if cam else '') + (f' · {fps:.0f} fps' if fps else '')
        p.drawText(QRectF(56, 0, self.width() - 80, self.HEAD), Qt.AlignmentFlag.AlignVCenter,
                   p.fontMetrics().elidedText(head, Qt.TextElideMode.ElideRight, int(self.width() - 82)))
        img = page.monitor.preview(self.tag)
        body = QRectF(0, self.HEAD, self.width(), self.height() - self.HEAD - self.FOOT)
        if img is None:
            p.setPen(THEME.color('muted'))
            msg = ('Press Start' if not page.running else '; '.join(st.get('reasons', [])) or 'Connecting…')
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
            if cam is not None:
                try:
                    self._overlay(p, cam, rect)
                except Exception:
                    pass
        self._footer(p)
        p.end()

    # sensor pixels -> widget
    def _map(self, cam, rect):
        intr = G._intr(self.page.session.project, cam)
        sx, sy = rect.width() / intr.width, rect.height() / intr.height
        return lambda u, v: QPointF(rect.left() + u * sx, rect.top() + v * sy)

    def _poly(self, p, pts, to, closed=True, color='#f2c84b', width=1.6, dash=False):
        pen = QPen(QColor(color), width)
        if dash:
            pen.setStyle(Qt.PenStyle.DashLine)
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        poly = QPolygonF([to(u, v) for u, v in pts])
        if closed:
            p.drawPolygon(poly)
        else:
            p.drawPolyline(poly)

    def _overlay(self, p, cam, rect):
        page = self.page
        s = page.session
        stage = s.stage
        to = self._map(cam, rect)
        intr = G._intr(s.project, cam)
        if stage == 'identify':
            idn = s.ident
            b = page.brightness.get(self.tag)
            base = idn['baseline'].get(self.tag)
            if b is not None and base:
                frac = max(0.0, min(1.0, b / base))
                p.fillRect(QRectF(rect.left() + 6, rect.bottom() - 12, (rect.width() - 12) * frac, 6),
                           QColor(STATUS['good'] if frac > 0.5 else STATUS['critical']))
            for tag, key in idn['done'].items():
                if key == self.tag:
                    p.setFont(sans(18, QFont.Weight.Bold))
                    p.setPen(QColor(STATUS['good'] if key == tag else STATUS['warning']))
                    p.drawText(rect, Qt.AlignmentFlag.AlignCenter, f'this is {tag}')
        if stage == 'aim':
            am = s.aim_models.get(self.tag)
            a = s.aim.get(self.tag)
            if am is not None:
                u, v, _, ok = intr.project_dirs(am.dirs)
                self._dots(p, u[ok], v[ok], to, '#f2c84b', 1.2)
                if a and a.get('ok'):
                    from ...core.joint import _rotvec_to_matrix
                    Rc = _rotvec_to_matrix(np.radians(a['optical_deg']))
                    u, v, _, ok = intr.project_dirs(am.dirs @ Rc)
                    self._dots(p, u[ok], v[ok], to, STATUS['good'], 1.6)
        if stage in ('boards', 'solve', 'verify', 'aim'):
            pose = s.current_pose(cam)
            from ...core.install import board_outline
            exp = set(page.expected.get(self.tag, []))
            for t in s.project.targets:
                if t.name not in exp:
                    continue
                uv, ok = G.project_pts(intr, pose, board_outline(t))
                if ok.all():
                    chk = t.name in s.st.check_boards
                    self._poly(p, uv, to, color='#7fb2ff' if chk else '#f2c84b', dash=True)
                    p.setFont(sans(10, QFont.Weight.DemiBold))
                    p.drawText(to(*uv[0]) + QPointF(3, -3), t.name + (' (check)' if chk else ''))
            for b in s.stands:
                if self.tag in (b.parent, b.child):
                    uv, ok = G.project_pts(intr, pose, b.outline())
                    if ok.all():
                        self._poly(p, uv, to, color='#f2c84b', dash=True)
                        p.drawText(to(*uv[0]) + QPointF(3, -3), b.name)
            if stage != 'aim':
                shape = s.shapes.get(cam.uid)
                if shape is not None:
                    fs = (intr.width / shape[1], intr.height / shape[0])
                    for lst in list(s.det.get(cam.uid, {}).values()):
                        if lst:
                            pts = lst[-1].image_pts
                            self._dots(p, pts[:, 0] * fs[0], pts[:, 1] * fs[1], to, STATUS['good'], 2.2)
            if stage in ('solve', 'verify'):
                sol = s.solution
                pose_s = sol.get('poses', {}).get(cam.uid)
                shape = s.shapes.get(cam.uid)
                if pose_s is not None and shape is not None:
                    fs = (intr.width / shape[1], intr.height / shape[0])
                    p.setPen(QPen(QColor(STATUS['critical']), 1.2))
                    for o in (sol.get('used') or {}).get(cam.uid, []):
                        uv, _ = G.project_pts(intr, pose_s, o.world)
                        for (a, b_), (c, d) in zip(uv, o.image_pts * fs):
                            # residuals drawn 20× so a tenth of a pixel shows
                            p.drawLine(to(a, b_), to(a + 20 * (c - a), b_ + 20 * (d - b_)))

    def _dots(self, p, u, v, to, color, r):
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(color))
        step = max(1, len(u) // 600)
        for a, b in zip(u[::step], v[::step]):
            p.drawEllipse(to(a, b), r, r)

    def _footer(self, p):
        page = self.page
        y = self.height() - self.FOOT
        p.setFont(sans(11))
        p.setPen(THEME.color('ink2'))
        lines = page.tile_lines(self.tag)
        for i, (text, kind) in enumerate(lines[:2]):
            col = {'good': STATUS['good'], 'bad': STATUS['critical'], 'warn': STATUS['warning']}.get(kind)
            p.setPen(QColor(col) if col else THEME.color('ink2'))
            p.drawText(QRectF(8, y + 2 + 17 * i, self.width() - 16, 16), Qt.AlignmentFlag.AlignVCenter,
                       p.fontMetrics().elidedText(text, Qt.TextElideMode.ElideRight, self.width() - 16))


# ---------------------------------------------------------------------- coverage map
class CoverageMap(QWidget):
    COLORS = {1: STATUS['good'], 2: STATUS['warning'], 3: STATUS['critical']}

    def __init__(self, page):
        super().__init__()
        self.page = page
        self.setMinimumSize(260, 260)
        self._img = None
        self._key = None

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), THEME.color('viewport'))
        s = self.page.session
        cov = s.cover if s else {}
        grid = cov.get('grid')
        if not cov.get('ok') or grid is None:
            p.setPen(THEME.color('muted'))
            p.setFont(sans(11))
            p.drawText(self.rect().adjusted(10, 0, -10, 0), Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap,
                       'The coverage of the measured rig shows here after the poses are measured.')
            p.end()
            return
        st = cov['status']
        ny, nx = st.shape
        if self._key is not id(st):
            rgba = np.zeros((ny, nx, 4), np.uint8)
            for k, c in self.COLORS.items():
                q = QColor(c)
                rgba[st == k] = (q.red(), q.green(), q.blue(), 190)
            rgba = np.ascontiguousarray(rgba[::-1])            # y up
            self._img = QImage(rgba.data, nx, ny, 4 * nx, QImage.Format.Format_RGBA8888).copy()
            self._key = id(st)
        x0, y0, cell = grid['x0'], grid['y0'], grid['cell']
        W, H = nx * cell, ny * cell
        sc = min((self.width() - 16) / W, (self.height() - 16) / H)
        ox = (self.width() - W * sc) / 2
        oy = (self.height() - H * sc) / 2
        p.drawImage(QRectF(ox, oy, W * sc, H * sc), self._img)

        def to(x, y):
            return QPointF(ox + (x - x0) * sc, oy + (y0 + H - y) * sc)
        # boards and cameras
        from ...core.install import board_outline
        p.setPen(QPen(THEME.color('ink2'), 1))
        for t in s.project.targets:
            q = QPolygonF([to(a, b) for a, b, _ in board_outline(t)])
            p.drawPolygon(q)
        for c in s.cams:
            pose = s.solution.get('poses', {}).get(c.uid) or G.planned_pose(c)
            pt = to(pose[1][0], pose[1][1])
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(THEME.cam(c.slot))
            p.drawEllipse(pt, 4, 4)
            p.setPen(THEME.color('ink'))
            p.setFont(mono(9))
            p.drawText(pt + QPointF(5, -4), c.tag)
        p.end()


# ---------------------------------------------------------------------- page
class GuidePage(QWidget):
    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        self.setObjectName('base')
        self.monitor = GuideMonitor()
        self.session: G.GuideSession | None = None
        self.running = False
        self.source = 'live'
        self.selected = None
        self.tiles = {}
        self.brightness = {}
        self.expected = {}
        self.snap = None
        self.plan = None
        self._busy = False
        self._rr = 0
        self._last_health = 0.0
        self.sig = _Signals()
        self.sig.done.connect(self._job_done)
        self.sig.coach.connect(self._coach_done)
        self.coach = None
        self._coach_thread = None

        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        split = QSplitter(Qt.Orientation.Horizontal)
        split.setHandleWidth(2)
        split.addWidget(self._build_left())
        split.addWidget(self._build_centre())
        split.addWidget(self._build_right())
        split.setSizes([380, 1100, 380])
        split.setStretchFactor(1, 1)
        lay.addWidget(split)

        self.timer = QTimer(self)
        self.timer.setInterval(100)
        self.timer.timeout.connect(self._tick)
        ctx.store.project_replaced.connect(lambda *_: self.reset())
        ctx.store.changed.connect(self._store_changed)
        self.reset()

    # ------------------------------------------------------------------ layout
    def _build_left(self):
        w = QWidget()
        w.setObjectName('panel')
        ll = QVBoxLayout(w)
        ll.setContentsMargins(10, 8, 10, 10)
        ll.setSpacing(5)
        ll.addWidget(label('Install guide', bold=True))
        self.variant_lbl = label('', muted=True, small=True, wrap=True)
        ll.addWidget(self.variant_lbl)
        row = QHBoxLayout()
        self.src_seg = SegmentedControl(SOURCES, 'live')
        self.src_seg.changed.connect(self.set_source)
        self.src_seg.setToolTip('Live: the cameras\' streams (Calibrate ▸ Live rig). Rehearse: every camera\'s view of '
                                'the planned boards rendered from a pose a little off the plan')
        row.addWidget(self.src_seg)
        self.start_btn = button('Start', 'primary')
        self.start_btn.clicked.connect(self.toggle)
        row.addWidget(self.start_btn)
        ll.addLayout(row)
        ll.addSpacing(4)
        self.stage_box = QVBoxLayout()
        self.stage_box.setSpacing(2)
        self.stage_btns = {}
        for s in G.STAGES:
            b = button(G.STAGE_TITLES[s])
            b.setCheckable(True)
            b.setStyleSheet('text-align: left; padding-left: 8px;')
            b.clicked.connect(lambda _=False, s=s: self.set_stage(s))
            self.stage_btns[s] = b
            self.stage_box.addWidget(b)
        ll.addLayout(self.stage_box)
        ll.addSpacing(6)
        self.help_lbl = label('', muted=True, small=True, wrap=True)
        ll.addWidget(self.help_lbl)
        self.actions = QHBoxLayout()
        self.actions.setSpacing(4)
        self.act_btns = {}
        for key, text in (('identify', 'Start port check'), ('skip_identify', 'Ports are labelled'),
                          ('accept_lens', 'Use library lenses'), ('plan', 'Plan boards'),
                          ('measure', 'Measure now'), ('verify', 'Run the checks')):
            b = button(text)
            b.clicked.connect(lambda _=False, k=key: self.action(k))
            self.act_btns[key] = b
            self.actions.addWidget(b)
        ll.addLayout(self.actions)
        self.detail = QTextBrowser()
        self.detail.setOpenExternalLinks(False)
        self.detail.setFrameShape(QFrame.Shape.NoFrame)
        ll.addWidget(self.detail, 1)
        row2 = QHBoxLayout()
        self.save_btn = button('Save as-built')
        self.save_btn.setToolTip('A new variant with the measured poses')
        self.save_btn.clicked.connect(self.save_as_built)
        self.report_btn = button('Report…')
        self.report_btn.clicked.connect(self.save_report)
        self.layout_btn = button('Export layout…')
        self.layout_btn.setToolTip('Every board\'s place (CSV), to lay them out with a tape')
        self.layout_btn.clicked.connect(self.export_layout)
        for b in (self.save_btn, self.report_btn, self.layout_btn):
            row2.addWidget(b)
        ll.addLayout(row2)
        # settings
        grid = QGridLayout()
        grid.setHorizontalSpacing(6)
        grid.setColumnStretch(0, 1)
        self.spins = {}
        for i, (key, text, lo, hi, step, dp) in enumerate((
                ('need_deg', 'Requirement, 1σ angle (°)', 0.01, 2.0, 0.01, 3),
                ('need_mm', 'Requirement, 1σ position (mm)', 0.5, 100.0, 0.5, 1),
                ('board_sigma_mm', 'Boards laid to (mm)', 0.5, 50.0, 0.5, 1),
                ('pixel_sigma', 'Corner noise (px)', 0.05, 2.0, 0.05, 2))):
            sp = ScrubSpinBox('', 'plain', dp, '', step=step, width=90)
            sp.setRange(lo, hi)
            sp.committed.connect(lambda v, k=key: self._setting(k, v))
            grid.addWidget(label(text, small=True), i, 0)
            grid.addWidget(sp, i, 1)
            self.spins[key] = sp
        ll.addLayout(grid)
        return w

    def _build_centre(self):
        w = QWidget()
        self.grid = QGridLayout(w)
        self.grid.setContentsMargins(4, 4, 4, 4)
        self.grid.setSpacing(4)
        return w

    def _build_right(self):
        w = QWidget()
        w.setObjectName('panel')
        rl = QVBoxLayout(w)
        rl.setContentsMargins(10, 8, 10, 10)
        rl.setSpacing(5)
        rl.addWidget(label('Coverage of the measured rig', bold=True))
        self.map = CoverageMap(self)
        rl.addWidget(self.map, 3)
        self.cov_lbl = label('', small=True, wrap=True)
        rl.addWidget(self.cov_lbl)
        rl.addSpacing(6)
        head = QHBoxLayout()
        head.addWidget(label('Coach', bold=True))
        self.auto_coach = QCheckBox('On events')
        self.auto_coach.setChecked(True)
        self.auto_coach.setToolTip('Claude comments when a stage is done or fails, a camera drops out, or the '
                                   'installer is stuck (never more often than every 20 s)')
        head.addStretch(1)
        head.addWidget(self.auto_coach)
        rl.addLayout(head)
        self.coach_view = QTextBrowser()
        self.coach_view.setFrameShape(QFrame.Shape.NoFrame)
        rl.addWidget(self.coach_view, 4)
        row = QHBoxLayout()
        self.ask_edit = QLineEdit()
        self.ask_edit.setPlaceholderText('Ask the coach…')
        self.ask_edit.returnPressed.connect(self.ask)
        row.addWidget(self.ask_edit)
        b = button('Ask')
        b.clicked.connect(self.ask)
        row.addWidget(b)
        rl.addLayout(row)
        self.coach_note = label('', muted=True, small=True, wrap=True)
        rl.addWidget(self.coach_note)
        return w

    # ------------------------------------------------------------------ state
    def _engine(self):
        w = self.ctx.window
        if w is not None and hasattr(w, '_shared_engine'):
            return w._shared_engine()
        from ...core.analysis import Engine
        return Engine()

    def reset(self):
        """A new session on the project as it is now."""
        self.stop()
        p = self.ctx.store.project
        self.session = G.GuideSession(p, self._engine())
        self.monitor.engine = self.session.engine
        for k, sp in self.spins.items():
            sp.blockSignals(True)
            sp.set_value(float(getattr(self.session.st, k)))
            sp.blockSignals(False)
        self.variant_lbl.setText(f'{p.name} · {p.variant.name} · {len(self.session.cams)} cameras · '
                                 f'{len(p.targets)} floor boards, {len(self.session.stands)} on stands')
        self.expected = {}                 # computed on Start (the boards each camera should see: a few seconds)
        self._rebuild_tiles()
        from ...assistant.install_coach import Coach
        s = self.session
        self.coach = Coach(self._coach_state, self._coach_frame, overlay=None,
                           whatif=lambda tag, a, b, c: G.coverage_whatif(s.project, s.engine, tag, a, b, c))
        ok, why = self.coach.available()
        self.coach_note.setText('Claude ' + self.coach.model + (' · ready' if ok else f' · {why}'))
        self._refresh()

    def _store_changed(self, sections):
        if self.running or self._busy:
            return
        if sections & {'variant', 'variants', 'targets', 'install_guide', 'library', 'vehicle', 'streams'}:
            self.reset()

    def _setting(self, key, v):
        if self.session is not None:
            setattr(self.session.st, key, float(v))
            self._refresh()

    def _rebuild_tiles(self):
        for t in self.tiles.values():
            t.setParent(None)
            t.deleteLater()
        self.tiles = {}
        cams = self.session.cams
        cols = 3 if len(cams) > 4 else 2 if len(cams) > 1 else 1
        for i, c in enumerate(cams):
            t = GuideTile(self, c.tag)
            t.clicked.connect(self.select)
            self.grid.addWidget(t, i // cols, i % cols)
            self.tiles[c.tag] = t

    def cam_health(self, tag) -> dict:
        return ((self.snap or {}).get('cameras') or {}).get(tag, {})

    def select(self, tag):
        self.selected = None if self.selected == tag else tag
        for t in self.tiles.values():
            t.update()

    def set_stage(self, s):
        if self.session is not None:
            self.session.set_stage(s)
        self._refresh()

    # ------------------------------------------------------------------ sources
    def set_source(self, kind):
        was = self.running
        self.stop()
        self.source = kind
        if was:
            self.start()

    def _configure(self):
        p = self.ctx.store.project
        s = self.session
        if self.source == 'rehearse':
            rng = np.random.default_rng(int(time.time()) % 1000)
            from ...core.mathutil import rot_zyx
            self.monitor.rehearsal = {}
            for c in s.cams:
                d = rng.normal(0, 0.6, 3)
                self.monitor.rehearsal[c.tag] = (rot_zyx(c.yaw + d[0], c.pitch + d[1], c.roll + d[2]),
                                                 c.pos() + rng.normal(0, 0.004, 3))
            q = p.snapshot()
            q.install_guide = s.project.install_guide
            q.streams = {c.tag: 'rehearse' for c in s.cams}
            self.monitor.configure(q, s.cams, self._engine, planned=None)
            s.nominal_fps = {c.tag: min(REHEARSE_FPS, p.module_for(c).fps or REHEARSE_FPS) for c in s.cams}
        else:
            self.monitor.configure(p, s.cams, self._engine)
            s.nominal_fps = {}

    def toggle(self):
        self.stop() if self.running else self.start()

    def start(self):
        if self.session is None:
            return
        if not self.expected:
            self.expected = self.session.expected_boards()
        self._configure()
        self.monitor.start()
        self.running = True
        self.timer.start()
        self.start_btn.setText('Stop')
        self._refresh()

    def stop(self):
        if getattr(self, 'monitor', None) is not None:
            self.monitor.stop(wait=False)
        self.running = False
        if hasattr(self, 'timer'):
            self.timer.stop()
        if hasattr(self, 'start_btn'):
            self.start_btn.setText('Start')

    def hideEvent(self, e):
        self.stop()
        super().hideEvent(e)

    # ------------------------------------------------------------------ the loop
    def _tick(self):
        if not self.running or self.session is None:
            return
        self.monitor.poll()
        now = time.time()
        s = self.session
        if now - self._last_health > 0.5:
            self._last_health = now
            self.snap = self.monitor.snapshot()
            s.update_health(self.snap)
            self.brightness = {}
            for c in s.cams:
                pv = self.monitor.preview(c.tag)
                if pv is not None:
                    self.brightness[c.tag] = G.mean_brightness(pv)
            if s.stage == 'identify' and s.ident['active'] and self.brightness:
                s.identify_feed(self.brightness)
            self._refresh()
            self._maybe_coach()
        if not self._busy:
            self._work()
        for t in self.tiles.values():
            t.update()
        self.map.update()

    def _work(self):
        """The stage's heavy step on the worker thread, a camera at a time."""
        s = self.session
        if s.stage not in ('aim', 'boards'):
            return
        cams = s.cams
        if not cams:
            return
        cam = cams[self._rr % len(cams)]
        self._rr += 1
        frame = self.monitor.latest(cam.tag)
        if frame is None:
            return
        self._busy = True
        job = 'aim' if s.stage == 'aim' else 'feed'

        def run():
            try:
                r = s.aim_camera(cam.tag, frame) if job == 'aim' else s.feed(cam.tag, frame)
                self.sig.done.emit(job, r)
            except Exception as e:
                self.sig.done.emit('error', f'{job} {cam.tag}: {e}')
        threading.Thread(target=run, daemon=True, name='guide-work').start()

    def _job_done(self, job, r):
        self._busy = False
        s = self.session
        if job == 'error':
            s.log('error', str(r))
        elif job == 'plan':
            self.plan = r
            self._apply_plan(r)
        elif job == 'measure':
            if s.solution.get('ok') and s.stage in ('boards', 'solve'):
                s.set_stage('coverage')
        elif job == 'feed':
            if s.boards_status()['ok'] and not s.solution:
                self.action('measure')
        self._refresh()

    # ------------------------------------------------------------------ actions
    def action(self, key):
        s = self.session
        if s is None:
            return
        if key == 'identify':
            if not self.running:
                self.start()
            s.set_stage('identify')
            s.identify_start(self.brightness or {c.tag: 128.0 for c in s.cams})
        elif key == 'skip_identify':
            s.identify_skip()
        elif key == 'accept_lens':
            s.lens_accept = not s.lens_accept
        elif key == 'plan':
            if self._busy:
                return
            self._busy = True
            proj = self.ctx.store.project
            eng = s.engine
            st = s.st

            def run():
                try:
                    self.sig.done.emit('plan', G.plan_boards(proj, eng, st))
                except Exception as e:
                    self.sig.done.emit('error', f'plan: {e}')
            threading.Thread(target=run, daemon=True, name='guide-plan').start()
            s.log('plan', 'planning the boards…')
        elif key == 'measure':
            if self._busy:
                return
            self._busy = True

            def run():
                try:
                    r = s.solve()
                    if r.get('poses'):
                        s.coverage()
                        s.verify()
                    self.sig.done.emit('measure', r)
                except Exception as e:
                    self.sig.done.emit('error', f'solve: {e}')
            threading.Thread(target=run, daemon=True, name='guide-solve').start()
        elif key == 'verify':
            if self._busy:
                return
            self._busy = True

            def run():
                try:
                    s.coverage()
                    self.sig.done.emit('verify', s.verify())
                except Exception as e:
                    self.sig.done.emit('error', f'verify: {e}')
            threading.Thread(target=run, daemon=True, name='guide-verify').start()
        self._refresh()

    def _apply_plan(self, plan):
        def fn(p):
            G.apply_plan(p, plan)
        n = len(plan['targets'])
        self.ctx.store.edit(['targets', 'install_guide'], fn, f'Install guide: board plan ({n} floor boards, '
                                                              f'{len(plan["stands"])} on stands)')
        self.reset()
        s = self.session
        for line in plan.get('log', []):
            s.log('plan', line)
        bad = [r['camera'] for r in plan['cameras'] if not r.get('ok')]
        s.log('plan', 'every camera meets the requirement' if not bad else 'short of the requirement: ' + ', '.join(bad))

    def save_as_built(self):
        s = self.session
        if s is None or not s.solution.get('poses'):
            self.ctx.msg('Measure the poses first', 4000)
            return
        v = s.as_built()

        def fn(p):
            nv = Variant.from_dict(v.to_dict())
            nv.uid = new_uid()
            for c in nv.cameras:
                c.link_uid = ''
            p.variants.append(nv)
            p.active_uid = nv.uid
            return nv
        self.ctx.store.edit('variants', fn, f'Install guide: {v.name}')
        if self.ctx.window is not None and hasattr(self.ctx.window, '_fill_variant_combo'):
            self.ctx.window._fill_variant_combo()
        self.ctx.msg(f'Created variant {v.name}', 6000)

    def report(self) -> dict:
        doc = self.session.report() if self.session else {}
        if self.coach is not None:
            doc['coach'] = [{k: v for k, v in e.items() if k in ('t', 'reason', 'question', 'text', 'removed')}
                            for e in self.coach.log]
        return doc

    def save_report(self, path=None):
        if not path:
            path, _ = QFileDialog.getSaveFileName(self, 'Save the install report', 'install-report.json',
                                                  'JSON (*.json)')
        if not path:
            return
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.report(), f, indent=1, ensure_ascii=False)
        self.ctx.msg(f'Saved {os.path.basename(path)}', 4000)

    def export_layout(self, path=None):
        p = self.ctx.store.project
        if not path:
            path, _ = QFileDialog.getSaveFileName(self, 'Export the board layout', 'board-layout.csv', 'CSV (*.csv)')
        if not path:
            return
        rows = ['name,kind,x_m,y_m,z_m,yaw_deg,rows,cols,square_mm,check,seen_by']
        exp = self.expected or {}
        chk = set(self.session.st.check_boards) if self.session else set()
        for t in p.targets:
            seen = ' '.join(k for k, v in exp.items() if t.name in v)
            rows.append(f'{t.name},floor,{t.x:.3f},{t.y:.3f},0.000,{t.yaw:.1f},{t.rows},{t.cols},{t.square_mm:g},'
                        f'{"yes" if t.name in chk else ""},{seen}')
        for b in G.stands_of(p):
            rows.append(f'{b.name},stand,{b.x:.3f},{b.y:.3f},{b.z:.3f},{b.yaw:.1f},{b.rows},{b.cols},'
                        f'{b.square_mm:g},,{b.parent} {b.child}')
        with open(path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(rows) + '\n')
        self.ctx.msg(f'Exported {len(rows) - 1} boards to {os.path.basename(path)}', 4000)

    # ------------------------------------------------------------------ coach
    def _coach_state(self):
        return self.session.state()

    def _coach_frame(self, tag):
        return self.monitor.latest(tag)

    def _maybe_coach(self):
        if not self.auto_coach.isChecked() or self.coach is None or self._coach_thread is not None:
            return
        if not self.coach.available()[0]:
            return
        reason = self.coach.should_ask(self.session.state())
        if reason:
            self._run_coach(reason=reason)

    def ask(self):
        q = self.ask_edit.text().strip()
        if not q or self.coach is None or self._coach_thread is not None:
            return
        self.ask_edit.clear()
        self._append_coach(f'<p><b>You:</b> {q}</p>')
        self._run_coach(question=q)

    def _run_coach(self, reason='', question=None):
        def run():
            self.sig.coach.emit(self.coach.ask(reason=reason, question=question))
        self._coach_thread = threading.Thread(target=run, daemon=True, name='guide-coach')
        self._coach_thread.start()

    def _coach_done(self, r):
        self._coach_thread = None
        if r.get('error'):
            if r['error'] != 'busy':
                self.coach_note.setText(r['error'])
            return
        text = (r.get('text') or '').replace('\n', '<br>')
        extra = (f'<br><span style="color:{STATUS["warning"]}">{len(r["removed"])} sentence(s) removed: figures the '
                 'guide did not give</span>') if r.get('removed') else ''
        why = f'<span style="color:gray">{r.get("reason")}</span><br>' if r.get('reason') else ''
        self._append_coach(f'<p>{why}{text}{extra}</p>')
        self.coach_note.setText(f'Claude {r.get("model")} · {r.get("seconds", 0):.1f} s')

    def _append_coach(self, html):
        self.coach_view.append(html)

    # ------------------------------------------------------------------ text
    def tile_lines(self, tag) -> list:
        s = self.session
        if s is None:
            return []
        stage = s.stage
        c = s.cam(tag)
        if stage == 'connect':
            h = self.cam_health(tag)
            if not self.running:
                return [('stopped', '')]
            return [(f"{_fmt(h.get('fps'), 1)} fps · {h.get('status', 'off')}", 'good' if h.get('status') == 'ok'
                     else 'warn'), ('; '.join(h.get('reasons', [])), 'bad')]
        if stage == 'identify':
            idn = s.ident
            if idn['asking'] == tag:
                return [(f'Cover the lens of {tag} ({c.location or c.name})', 'warn')]
            mapped = next((t for t, k in idn['done'].items() if k == tag), None)
            return [(f'stream of {mapped}' if mapped else '', 'good' if mapped == tag else 'warn')]
        if stage == 'lens':
            r = next((x for x in s.lens_rows() if x['camera'] == tag), {})
            return [(r.get('source', ''), 'good' if r.get('ok') else 'warn')]
        if stage == 'aim':
            a = s.aim.get(tag)
            if not a:
                return [('measuring…', '')]
            if not a.get('ok'):
                return [(a.get('why', ''), 'warn')]
            head = (f"yaw {_fmt(a['d_yaw_deg'], 1, '°')} pitch {_fmt(a['d_pitch_deg'], 1, '°')} "
                    f"roll {_fmt(a['d_roll_deg'], 1, '°')}")
            todo = [x for x in a['instructions'] if 'within' not in x]
            return [(head, 'good' if a.get('within') else 'warn'), (todo[0] if todo else 'within tolerance',
                                                                    'warn' if todo else 'good')]
        if stage == 'boards':
            row = next((r for r in s.boards_status()['cameras'] if r['camera'] == tag), {})
            exp, found = row.get('expected', []), row.get('found', [])
            return [(f"{len(found)} of {len(exp)} boards found", 'good' if found and not row.get('missing') else 'warn'),
                    (('missing ' + ', '.join(row['missing'])) if row.get('missing') else '', 'warn')]
        if stage in ('solve', 'coverage', 'verify'):
            row = next((r for r in s.solution.get('cameras', []) if r['camera'] == tag), None)
            if not row or not row.get('solved'):
                return [('not measured', 'warn')]
            sg = row['sigma']
            return [(f"σ yaw {_fmt(sg['yaw_deg'], 3, '°')} pitch {_fmt(sg['pitch_deg'], 3, '°')} "
                     f"x {_fmt(sg['x_mm'], 1, ' mm')}", 'good' if row['meets'] else 'bad'),
                    (f"{_fmt(row['ratio'], 2)}× the requirement ({row['axis']})", 'good' if row['meets'] else 'bad')]
        return []

    def _refresh(self):
        s = self.session
        if s is None:
            return
        rows = s.stage_rows()
        for r in rows:
            b = self.stage_btns[r['stage']]
            b.setChecked(r['current'])
            b.setText(('✓ ' if r['done'] else '○ ') + r['title'])
        self.help_lbl.setText(STAGE_HELP.get(s.stage, ''))
        vis = {'identify': ('identify', 'skip_identify'), 'lens': ('accept_lens',), 'boards': ('plan', 'measure'),
               'solve': ('measure',), 'coverage': ('verify',), 'verify': ('verify',)}.get(s.stage, ())
        for k, b in self.act_btns.items():
            b.setVisible(k in vis)
        self.act_btns['accept_lens'].setText('Library lenses: accepted' if s.lens_accept else 'Use library lenses')
        self.detail.setHtml(self._detail_html())
        cov = s.cover
        if cov.get('ok'):
            self.cov_lbl.setText(
                f"Robust {cov['robust_pct']:.1f} % · uncertain {cov['uncertain_pct']:.1f} % · blind "
                f"{cov['blind_m2']:.2f} m² of {cov['roi_m2']:.1f} m² · plan {cov['plan_covered_pct']:.1f} %"
                + (''.join(f" · {t}: a car is 20 px wide at {v['car']:.0f} m" for t, v in cov.get('far_field_m', {})
                           .items())))
        else:
            self.cov_lbl.setText('')

    def _detail_html(self) -> str:
        s = self.session
        st = s.state()
        stage = s.stage
        out = []

        def row(cells, head=False):
            tag = 'th' if head else 'td'
            return '<tr>' + ''.join(f'<{tag} style="padding:1px 6px">{c}</{tag}>' for c in cells) + '</tr>'
        if stage == 'connect':
            c = st['connect']
            out.append('<table>' + row(['camera', 'fps', 'of', 'status'], True) + ''.join(
                row([r['camera'], _fmt(r['fps'], 1), _fmt(r['nominal_fps'], 0), r['status']]) for r in c['cameras'])
                + '</table>')
            if c.get('sync_skew_ms') is not None:
                out.append(f"<p>Capture-time spread {_fmt(c['sync_skew_ms'], 1)} ms ({c['sync_status']})</p>")
        elif stage == 'identify':
            i = st['identify']
            if i['asking']:
                cam = s.cam(i['asking'])
                out.append(f"<p><b>Cover the lens of {i['asking']}</b> ({cam.location or cam.name}) with your hand.</p>")
            out.append(f"<p>{i['why']}</p>")
            out.append('<table>' + ''.join(row([t, '→ stream of ' + k + (' ✓' if k == t else ' (swapped)')])
                                           for t, k in i['done'].items()) + '</table>')
        elif stage == 'lens':
            out.append('<table>' + row(['camera', 'lens model', 'RMS px'], True) + ''.join(
                row([r['camera'], r['source'] + (' — ' + '; '.join(r['problems']) if r['problems'] else ''),
                     _fmt(r.get('rms_px'), 2)]) for r in st['lens']['cameras']) + '</table>')
        elif stage == 'aim':
            for r in st['aim']['cameras']:
                ins = r['instructions'] or ([r['why']] if r['why'] else ['measuring…'])
                out.append(f"<p><b>{r['camera']}</b><br>" + '<br>'.join(ins) + '</p>')
        elif stage == 'boards':
            b = st['boards']
            out.append('<table>' + row(['camera', 'found', 'missing'], True) + ''.join(
                row([r['camera'], ', '.join(r['found']), ', '.join(r['missing'])]) for r in b['cameras']) + '</table>')
            if self.plan:
                out.append('<p><b>Board plan</b><br>' + '<br>'.join(self.plan.get('log', [])[-12:]) + '</p>')
                out.append('<table>' + row(['camera', 'kind', 'predicted ×req'], True) + ''.join(
                    row([r['camera'], r['kind'], _fmt(r.get('ratio'), 2)]) for r in self.plan['cameras']) + '</table>')
        elif stage == 'solve':
            sol = st['solve']
            out.append('<table>' + row(['camera', 'σ yaw°', 'σ pitch°', 'σ roll°', 'σ x mm', 'σ y mm', 'σ z mm',
                                        '×req'], True) + ''.join(
                row([r['camera']] + ([_fmt(r['sigma'][k], 3 if k.endswith('deg') else 1) for k in G.SIGMA_KEYS]
                                     + [_fmt(r['ratio'], 2)] if r.get('solved') else ['not measured'] * 7))
                for r in sol.get('cameras', [])) + '</table>')
            fc = sol.get('floor_chi2')
            if fc:
                out.append(f"<p>Fit residuals: χ² {_fmt(fc['chi2'], 1)} on {fc['dof']} degrees of freedom, p = "
                           f"{_fmt(fc['p'], 3)} ({'consistent with' if fc['pass'] else 'more than'} the corner noise)</p>")
            if sol.get('why'):
                out.append(f"<p style='color:{STATUS['critical']}'>{sol['why']}</p>")
        elif stage == 'coverage':
            c = st['coverage']
            if c.get('ok'):
                out.append(f"<p>Robust {_fmt(c['robust_pct'], 1)} %, uncertain {_fmt(c['uncertain_pct'], 1)} %, "
                           f"blind {_fmt(c['blind_m2'], 2)} m²; the plan covers {_fmt(c['plan_covered_pct'], 1)} %.</p>")
                if c.get('holes'):
                    out.append('<p>Blind patches: ' + '; '.join(
                        f"{_fmt(h['area_m2'], 2)} m² at x {_fmt(h['x'], 2)} m, y {_fmt(h['y'], 2)} m"
                        for h in c['holes'][:6]) + '</p>')
        elif stage == 'verify':
            v = st['verify']
            for t in v.get('tests', []):
                ok = t.get('pass')
                out.append(f"<p><span style='color:{STATUS['good'] if ok else STATUS['critical']}'>"
                           f"{'pass' if ok else 'fail'}</span> {t['test']}"
                           + (f" — {t.get('why')}" if t.get('why') else '') + '</p>')
            if v.get('note'):
                out.append(f"<p>{v['note']}</p>")
        ev = st.get('events', [])[-6:]
        if ev:
            out.append('<p style="color:gray">' + '<br>'.join(e['text'] for e in ev) + '</p>')
        return ''.join(out)

    def activate(self):
        self._refresh()
