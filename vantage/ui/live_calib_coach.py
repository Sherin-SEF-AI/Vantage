"""The Live rig page's calibration coach: the panel (settings, lock indicators, the next view in words, finish), the
AR overlay drawn over the coached camera's live image (the ghost board where to hold it next, through the exact lens
model; the board found in the frame; the corners collected so far) and a simulated technician that shows a rendered
board to a synthetic stream. The coach itself (core.calib_coach) runs on one background thread; detections are
throttled to the settings' rate, one at a time, so the live view stays smooth."""
from __future__ import annotations

import concurrent.futures as cf
import json
import math
import os
import threading
import time
import traceback

import numpy as np
from PyQt6.QtCore import Qt, QObject, QRectF, QPointF, pyqtSignal
from PyQt6.QtGui import QPainter, QPen, QColor, QFont, QPainterPath, QBrush
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QComboBox, QFileDialog, QSizePolicy)

from ..core import calib_coach as CC
from ..core import rig_health as rh
from .theme import THEME, STATUS, mono, sans
from .widgets import label, button, ScrubSpinBox, SegmentedControl, ToggleSwitch, StatusLine, hline

MODE_OPTIONS = [('lens', 'Lens'), ('lidar', 'Lidar–camera')]
TECH_OPTIONS = [('off', 'Off'), ('coached', 'Coached'), ('random', 'Random')]
BLUR_S = 0.4                         # the simulated technician's frames are blurred this long after a move
LOCK_FLOOR = 100.0                   # a lock bar is empty at this many times the requirement


def _ratio_colour(r) -> QColor:
    if r <= 1.0:
        return QColor(STATUS['good'])
    return QColor(STATUS['warning'] if r <= 2.0 else STATUS['serious'] if r < LOCK_FLOOR else STATUS['critical'])


# ---------------------------------------------------------------------- lock indicators
class LockBars(QWidget):
    """One bar per lock: full and green when its 1σ is within the requirement, emptier the further it is (log scale,
    empty at 100×)."""
    ROW = 19

    def __init__(self):
        super().__init__()
        self.locks = {}
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._resize()

    def set_locks(self, locks: dict):
        self.locks = dict(locks or {})
        self._resize()
        self.update()

    def _resize(self):
        self.setFixedHeight(max(1, len(self.locks)) * self.ROW + 2)

    def fill_of(self, ratio) -> float:
        if not math.isfinite(ratio) or ratio >= LOCK_FLOOR:
            return 0.0
        return float(min(1.0, max(0.0, 1.0 - math.log10(max(ratio, 1.0)) / math.log10(LOCK_FLOOR))))

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        lab_w, val_w = 74, 112
        bar_w = max(20, self.width() - lab_w - val_w - 8)
        for i, (k, x) in enumerate(self.locks.items()):
            y = i * self.ROW + 1
            p.setFont(sans(11))
            p.setPen(THEME.color('ink2'))
            p.drawText(QRectF(0, y, lab_w, self.ROW), Qt.AlignmentFlag.AlignVCenter, x.get('label', k))
            r = QRectF(lab_w, y + 5, bar_w, self.ROW - 10)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(THEME.color('raised'))
            p.drawRoundedRect(r, 2, 2)
            f = self.fill_of(x['ratio'])
            if f > 0:
                p.setBrush(_ratio_colour(x['ratio']))
                p.drawRoundedRect(QRectF(r.left(), r.top(), r.width() * f, r.height()), 2, 2)
            p.setPen(THEME.color('ink' if x['locked'] else 'muted'))
            p.setFont(mono(10))
            unit = '°' if x['unit'] == 'deg' else ' mm'
            dp = 3 if x['unit'] == 'deg' else 1
            sig = '—' if not math.isfinite(x['sigma']) else f'{x["sigma"]:.{dp}f}'
            p.drawText(QRectF(lab_w + bar_w + 8, y, val_w, self.ROW), Qt.AlignmentFlag.AlignVCenter,
                       f'{sig} / {x["need"]:g}{unit}')
        p.end()


# ---------------------------------------------------------------------- the overlay
def paint_overlay(p: QPainter, rect: QRectF, sensor, ov: dict, slot=0):
    """Draw the coach's overlay over a camera image shown in rect (sensor = (width, height) in sensor pixels):
    ov['ghost'] (core.calib_coach.Coach.ghost, lens-model pixels), ov['det'] (corners found, OpenCV pixels, and whether
    the view was accepted) and ov['coverage'] (the accepted views' corners, OpenCV pixels)."""
    sx, sy = rect.width() / sensor[0], rect.height() / sensor[1]

    def pt(u, v):
        return QPointF(rect.left() + u * sx, rect.top() + v * sy)

    def poly(pts, closed=False):
        pp = QPainterPath()
        for k, (u, v) in enumerate(pts):
            pp.moveTo(pt(u, v)) if k == 0 else pp.lineTo(pt(u, v))
        if closed:
            pp.closeSubpath()
        return pp
    cov = ov.get('coverage')
    if cov is not None and len(cov):
        p.setPen(Qt.PenStyle.NoPen)
        c = QColor(STATUS['good'])
        c.setAlpha(110)
        p.setBrush(c)
        for u, v in cov:
            p.drawEllipse(pt(u + 0.5, v + 0.5), 1.6, 1.6)
    g = ov.get('ghost')
    if g is not None:
        col = QColor(THEME.cam_hex(slot))
        fill = QColor(col)
        fill.setAlpha(40)
        p.setBrush(QBrush(fill))
        p.setPen(QPen(col, 2.0, Qt.PenStyle.DashLine))
        p.drawPath(poly(g['outline'], True))
        p.setBrush(Qt.BrushStyle.NoBrush)
        thin = QColor(col)
        thin.setAlpha(150)
        p.setPen(QPen(thin, 0.8))
        for ln in g['grid']:
            p.drawPath(poly(ln))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(col)
        for u, v in g['corners']:
            p.drawEllipse(pt(u, v), 1.8, 1.8)
    d = ov.get('det')
    if d is not None and d.get('uv') is not None:
        col = QColor(STATUS['good'] if d.get('accepted') else STATUS['warning'])
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(col, 1.4))
        for u, v in d['uv']:
            p.drawEllipse(pt(u + 0.5, v + 0.5), 3.0, 3.0)


# ---------------------------------------------------------------------- simulated technician
class TechnicianSource(rh.SyntheticSource):
    """A synthetic stream that shows a simulated technician's board: rendered through the camera's lens model at the
    pose the technician holds (show()), blurred for BLUR_S after each move (the board in motion), otherwise like
    rig_health.SyntheticSource (fresh noise per frame at the stream's rate)."""

    def __init__(self, tech: CC.Technician, fps=10.0, size=None, on_frame=None, scale=0.5):
        super().__init__(lambda: None, fps, size, spec='synthetic', on_frame=on_frame)
        self.tech = tech
        self.scale = scale
        self.pose = None
        self.version = 0
        self.moved = 0.0
        self._plock = threading.Lock()

    def show(self, pose):
        with self._plock:
            self.pose = pose
            self.version += 1
            self.moved = time.monotonic()

    def _render(self, pose, version):
        import cv2
        c = self.tech.coach
        if pose is None:
            w, h = int(round(c.intr.width * self.scale)), int(round(c.intr.height * self.scale))
            base = np.full((h, w, 3), 118, np.uint8)
        else:
            base = self.tech.render(pose, self.scale, seed=version, noise=0.0)
        if self.size_wanted and (base.shape[1], base.shape[0]) != self.size_wanted:
            base = cv2.resize(base, self.size_wanted, interpolation=cv2.INTER_LINEAR)
        n = self.noise
        base = np.minimum(base, 254 - 2 * n).astype(np.uint8)
        k = max(3, int(base.shape[1] * 0.02)) | 1
        return base, cv2.blur(base, (k, 3))

    def _run(self, stop):
        rng = np.random.default_rng(7)
        cur, base, blurred = -1, None, None
        period = 1.0 / self.fps
        nxt = time.monotonic()
        k = 0
        while not stop.is_set():
            with self._plock:
                pose, version, moved = self.pose, self.version, self.moved
            if version != cur:
                try:
                    base, blurred = self._render(pose, version)
                except Exception as e:                      # noqa: BLE001  (shown as the stream's error)
                    self.error = f'cannot render the technician\'s board: {e}'
                    return
                cur = version
            img = blurred if (pose is not None and time.monotonic() - moved < BLUR_S) else base
            f = img + rng.integers(0, 2 * self.noise + 1, img.shape[:2], dtype=np.uint8)[..., None]
            k += 1
            t = min(nxt, time.monotonic())
            with self._lock:
                self._frame = f
                self.frames += 1
                self.size = (f.shape[1], f.shape[0])
                self.stamp = t
            cb = self.on_frame
            if cb is not None:
                try:
                    cb(f, t)
                except Exception:                           # noqa: BLE001
                    pass
            nxt += period
            delay = nxt - time.monotonic()
            if delay < -period:
                nxt = time.monotonic()
            elif delay > 0:
                stop.wait(delay)


# ---------------------------------------------------------------------- the panel
class _Sig(QObject):
    done = pyqtSignal(object)            # (generation, kind, payload)
    failed = pyqtSignal(object)          # (generation, text)


class CoachPanel(QWidget):
    """Settings, lock indicators and guidance of the Live rig page's calibration coach. page: the RigPage (its
    monitor, its streams, its context)."""

    def __init__(self, page):
        super().__init__()
        self.page = page
        self.ctx = page.ctx
        self.coach: CC.Coach | None = None              # touched only on the worker thread
        self.tech: CC.Technician | None = None          # likewise
        self.pool = cf.ThreadPoolExecutor(max_workers=1, thread_name_prefix='calib-coach')
        self.gen = 0
        self.active = False
        self.state = {'status': None, 'advice': None, 'det': None, 'coverage': None, 'views': 0, 'cam_uid': None,
                      'cam_tag': '', 'slot': 0, 'error': '', 'built': False, 'lidar': None}
        self._busy = False
        self._last_t = 0.0
        self._last_n = -1
        self._pending = 0
        self._tech_tag = ''               # the camera whose stream the simulated technician has taken over
        self.sig = _Sig()
        self.sig.done.connect(self._done)
        self.sig.failed.connect(self._failed)
        self._build()
        self.sync_fields()

    # ------------------------------------------------------------------ layout
    def _build(self):
        ll = QVBoxLayout(self)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.setSpacing(4)
        ll.addWidget(label('Calibration coach', bold=True))
        ll.addWidget(label('Hold the board where the ghost board shows. Each sharp, new view adds its corners\' '
                           'information; a bar turns green when its 1σ is within the requirement. Lens: the camera\'s '
                           'own intrinsics. Lidar–camera: where the lidar sits relative to the camera.', muted=True,
                           small=True, wrap=True))
        g = QGridLayout()
        g.setHorizontalSpacing(6)
        g.setVerticalSpacing(4)
        g.setColumnStretch(1, 1)
        self.mode = SegmentedControl(MODE_OPTIONS, 'lens')
        self.mode.changed.connect(lambda k: self._set('mode', k, f'Calibration coach: {dict(MODE_OPTIONS)[k]}'))
        self.cam = QComboBox()
        self.cam.activated.connect(lambda _=0: self._set('camera', self.cam.currentData() or '',
                                                         f'Calibration coach on {self.cam.currentText()}'))
        g.addWidget(label('Calibrate', muted=True), 0, 0)
        g.addWidget(self.mode, 0, 1)
        g.addWidget(label('Camera', muted=True), 1, 0)
        g.addWidget(self.cam, 1, 1)
        ll.addLayout(g)
        self.board_box = QWidget()
        bg = QHBoxLayout(self.board_box)
        bg.setContentsMargins(0, 0, 0, 0)
        bg.setSpacing(4)
        self.cols = ScrubSpinBox('', 'plain', 0, '', step=1, width=52)
        self.rows = ScrubSpinBox('', 'plain', 0, '', step=1, width=52)
        self.square = ScrubSpinBox('', 'plain', 0, 'mm', step=1, width=70)
        self.cols.setToolTip('Inner corners across the board')
        self.rows.setToolTip('Inner corners down the board')
        self.square.setToolTip('Square size')
        bg.addWidget(label('Board', muted=True))
        for w_, key, lab in ((self.cols, 'cols', 'inner corners across'), (self.rows, 'rows', 'inner corners down'),
                             (self.square, 'square_mm', 'square size')):
            w_.committed.connect(lambda v, k=key, t=lab: self._set(k, v, f'Calibration coach board: {t} {v:g}'))
            bg.addWidget(w_)
        bg.addStretch(1)
        ll.addWidget(self.board_box)
        self.board_note = label('', muted=True, small=True, wrap=True)
        ll.addWidget(self.board_note)
        self.need = label('', muted=True, small=True, wrap=True)
        ll.addWidget(self.need)
        h = QHBoxLayout()
        h.setSpacing(6)
        self.auto = ToggleSwitch('Accept automatically')
        self.auto.setChecked(True)
        self.auto.setToolTip('Accept every sharp view that is not a repeat of one already taken')
        h.addWidget(self.auto)
        h.addStretch(1)
        ll.addLayout(h)
        h = QHBoxLayout()
        h.setSpacing(6)
        self.b_accept = button('Accept view', None, None, 'Take the newest frame as a view now (sharp and new only)')
        self.b_accept.clicked.connect(self.accept_now)
        self.b_undo = button('Undo view', None, None, 'Drop the last accepted view')
        self.b_undo.clicked.connect(self.undo)
        self.b_reset = button('Reset', None, None, 'Drop every view and start again')
        self.b_reset.clicked.connect(self.reset)
        for b in (self.b_accept, self.b_undo, self.b_reset):
            h.addWidget(b)
        h.addStretch(1)
        ll.addLayout(h)
        h = QHBoxLayout()
        h.addWidget(label('Simulated technician', muted=True))
        self.tech_seg = SegmentedControl(TECH_OPTIONS, 'off')
        self.tech_seg.changed.connect(self.set_technician)
        self.tech_seg.setToolTip('With a synthetic stream: a simulated technician shows a rendered board, where the '
                                 'coach asks (Coached) or at random hand-held poses (Random)')
        h.addWidget(self.tech_seg)
        h.addStretch(1)
        ll.addLayout(h)
        ll.addWidget(hline())
        self.views_lbl = label('No views yet', bold=True)
        ll.addWidget(self.views_lbl)
        self.bars = LockBars()
        ll.addWidget(self.bars)
        ll.addWidget(label('Next view', bold=True))
        self.advice = label('', wrap=True)
        ll.addWidget(self.advice)
        self.advice_v = label('', muted=True, small=True, wrap=True)
        ll.addWidget(self.advice_v)
        self.line = StatusLine('', '')
        ll.addWidget(self.line)
        h = QHBoxLayout()
        h.setSpacing(6)
        self.b_finish = button('Lens calibration…', 'primary', None, 'Hand the accepted views to the calibration')
        self.b_finish.clicked.connect(lambda: self.finish())
        self.b_save = button('Save session…', None, 'file', 'Write the accepted frames and the session (poses, 1σ) '
                                                            'as calib_coach_session.json')
        self.b_save.clicked.connect(lambda: self.save_session())
        h.addWidget(self.b_finish)
        h.addWidget(self.b_save)
        h.addStretch(1)
        ll.addLayout(h)

    # ------------------------------------------------------------------ settings
    @property
    def settings(self) -> CC.CoachSettings:
        return CC.settings_of(self.ctx.store.project)

    def _set(self, key, value, text):
        st = self.settings
        if getattr(st, key) == value:
            return
        d = st.to_dict()
        d[key] = value
        new = CC.CoachSettings.from_dict(d).to_dict()

        def fn(p):
            p.analysis.calib_coach = new
        self.ctx.store.edit('analysis', fn, text)

    def sync_fields(self):
        st = self.settings
        self.mode.set_value(st.mode)
        cams = [c for c in self.ctx.store.variant.cameras if c.visible]
        self.cam.blockSignals(True)
        self.cam.clear()
        self.cam.addItem('Best pair' if st.mode == 'lidar' else (cams[0].tag if cams else '—'), '')
        for c in cams:
            self.cam.addItem(c.tag, c.tag)
        i = self.cam.findData(st.camera)
        self.cam.setCurrentIndex(max(0, i))
        self.cam.blockSignals(False)
        for w_, v in ((self.cols, st.cols), (self.rows, st.rows), (self.square, st.square_mm)):
            w_.blockSignals(True)
            w_.set_value(v)
            w_.blockSignals(False)
        lens = st.mode == 'lens'
        self.board_box.setVisible(lens)
        if lens:
            self.board_note.setText(f'{st.cols} × {st.rows} inner corners of {st.square_mm:g} mm (the lens '
                                    'calibration\'s board), held 0.5–2 m high, '
                                    f'{st.near_m:g}–{st.far_m:g} m from the lens.')
        else:
            from ..core import lidar_calib as LC
            lst = LC.LidarCalibSettings.from_dict(self.ctx.store.project.analysis.lidar_calib)
            self.board_note.setText(f'The lidar–camera calibration\'s board: {lst.cols} × {lst.rows} inner corners of '
                                    f'{lst.square_mm:g} mm with a {lst.margin_mm:g} mm margin, '
                                    f'{lst.near_m:g}–{lst.far_m:g} m from the camera.')
        self.b_finish.setText('Lens calibration…' if lens else 'Lidar–camera calibration…')
        self._sync_buttons()

    def on_store_changed(self, sections):
        if sections & {'analysis', 'variant', 'variants', 'library', 'project', 'vehicle', 'rig', 'targets'}:
            self.sync_fields()
            if self.active:
                self.rebuild()

    def _sync_buttons(self):
        n = self.state['views']
        lens = self.settings.mode == 'lens'
        self.b_finish.setEnabled(self.state['built'] and (n >= 4 if lens else n >= 3))
        self.b_undo.setEnabled(n > 0)
        self.b_reset.setEnabled(n > 0)
        self.b_save.setEnabled(n > 0)
        self.b_accept.setEnabled(self.state['built'])

    # ------------------------------------------------------------------ the coach on its thread
    def _submit(self, kind, fn):
        gen = self.gen
        self._pending += 1

        def job():
            try:
                out = fn()
                self.sig.done.emit((gen, kind, out))
            except RuntimeError:                         # the panel was deleted meanwhile
                pass
            except Exception as e:                       # noqa: BLE001  (shown in the panel)
                try:
                    self.sig.failed.emit((gen, f'{type(e).__name__}: {e}\n{traceback.format_exc()}'))
                except RuntimeError:
                    pass
        try:
            self.pool.submit(job)
        except RuntimeError:                             # the pool was shut down
            self._pending -= 1

    def _snapshot(self, det=None, advice=None) -> dict:
        """What the UI shows, made on the worker thread (immutable for the UI)."""
        c = self.coach
        cov = np.concatenate([v.uv for v in c.views if v.uv is not None]) if c.views else np.zeros((0, 2))
        out = {'status': c.status(), 'views': len(c.views), 'coverage': cov, 'cam_uid': c.cam.uid,
               'cam_tag': c.tag, 'slot': c.cam.slot, 'lidar': c.lidar.tag if c.lidar is not None else None,
               'need': dict(c.need), 'sensor': (c.intr.width, c.intr.height)}
        if det is not None:
            out['det'] = {'uv': det.uv, 'accepted': det.accepted, 'reason': det.reason, 'found': det.found,
                          'sharpness': det.sharpness}
        if advice is not None:
            out['advice'] = advice
        return out

    def rebuild(self):
        """A new coach from the project's settings (the views taken so far are kept when the camera, the mode and the
        board are the same)."""
        self.gen += 1
        project = self.ctx.store.project.snapshot()
        eng = self.page._engine()
        st = CC.settings_of(project)
        kind = self.tech_seg.value()

        def work():
            old = self.coach
            c = CC.Coach(project, eng, st)
            c.candidates()
            c._cand_infos()
            if old is not None and old.cam.uid == c.cam.uid and old.st.mode == c.st.mode and \
                    (old.board.cols, old.board.rows, old.board.square_mm) == \
                    (c.board.cols, c.board.rows, c.board.square_mm):
                for v in old.views:
                    c.add_pose(v.pose, v.frame, v.uv, v.name, v.t, v.sharpness, v.detected)
            self.coach = c
            self.tech = CC.Technician(c, kind if kind in CC.TECHNICIANS else 'coached', st.seed)
            adv = c.next_best()
            return self._snapshot(advice=adv)
        self.state['built'] = False
        self._busy = False
        self._submit('built', work)

    def activate(self):
        """The page switched to the coach (or started streaming in it)."""
        self.active = True
        if self.coach is None or not self.state['built']:
            self.rebuild()
        self.install_technician()

    def deactivate(self):
        self.active = False
        self.remove_technician()

    def tick(self):
        """Called by the page's timer: hand the newest frame of the coached camera to the coach, at most detect_hz
        times a second and one at a time."""
        if not self.active or not self.state['built'] or self._busy:
            return
        tag = self.state['cam_tag']
        s = self.page.monitor.streams.get(tag)
        if s is None:
            return
        now = time.monotonic()
        if now - self._last_t < 1.0 / max(self.settings.detect_hz, 0.1):
            return
        n = s.stats.frames
        if n == self._last_n:
            return
        frame = self.page.monitor.latest(tag)
        if frame is None:
            return
        self._last_t, self._last_n = now, n
        self._offer(frame, self.auto.isChecked(), 'frame')

    def _offer(self, frame, accept, kind):
        self._busy = True
        t = time.monotonic()
        src = self._tech_source()
        moving = src is not None and src.pose is not None and t - src.moved < BLUR_S

        def work():
            import cv2
            c = self.coach
            hint = self.state.get('advice').pose if self.state.get('advice') is not None else None
            det = c.offer(frame, t, hint, accept)
            if det.accepted and frame.ndim == 3:
                c.views[-1].frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)  # a third of the memory
            adv = None
            if det.accepted or kind == 'accept':
                adv = c.next_best(det.pose if det.found else None)
            out = self._snapshot(det, adv)
            # the simulated technician moves on once a view is taken, or when the frame it holds still cannot be taken
            # (no board found, a repeat, too blurred, the lidar not crossing it)
            if src is not None and self.tech is not None and not moving and (accept or not det.found):
                pose = self._tech_pose(adv)
                if pose is not None:
                    src.show(pose)
            return out
        self._submit(kind, work)

    def _tech_pose(self, adv):
        """(worker thread) Where the simulated technician holds the board next."""
        t = self.tech
        if t is None:
            return None
        t.kind = self.tech_seg.value() if self.tech_seg.value() in CC.TECHNICIANS else t.kind
        if t.kind == 'random':
            t.pose = t._random_pose()
        else:
            a = adv or self.coach.next_best(t.pose)
            t.advice = a
            t.pose = None if a is None else t._jittered(a.pose)
        return t.pose

    def accept_now(self):
        if not self.state['built'] or self._busy:
            return False
        frame = self.page.monitor.latest(self.state['cam_tag'])
        if frame is None:
            self.ctx.msg('No frame from this camera yet: start the streams first', 5000)
            return False
        self._offer(frame, True, 'accept')
        return True

    def undo(self):
        if self.coach is None:
            return

        def work():
            self.coach.undo()
            return self._snapshot(advice=self.coach.next_best())
        self._submit('undo', work)

    def reset(self):
        if self.coach is None:
            return

        def work():
            self.coach.reset()
            out = self._snapshot(advice=self.coach.next_best())
            src = self._tech_source()
            if src is not None and self.tech is not None:
                self.tech.pose = None
                pose = self._tech_pose(out['advice'])
                if pose is not None:
                    src.show(pose)
            return out
        self._submit('reset', work)

    def _done(self, out):
        gen, kind, snap = out
        self._pending = max(0, self._pending - 1)
        if kind in ('frame', 'accept'):
            self._busy = False
        if kind == 'finished_lidar':
            self._open_lidar(snap)
            return
        if gen != self.gen:
            return
        prev_tag = self.state['cam_tag']
        self.state.update(snap)
        self.state['built'] = True
        self.state['error'] = ''
        if kind == 'built':
            self.state['det'] = None
            if prev_tag != snap['cam_tag']:
                if self._tech_tag and self._tech_tag != snap['cam_tag']:
                    self.remove_technician()
                self.page.coach_camera_changed()
            self.install_technician()
        self._fill()
        self.page.repaint_coach()

    def _failed(self, out):
        gen, text = out
        self._pending = max(0, self._pending - 1)
        self._busy = False
        if gen != self.gen:
            return
        self.state['error'] = text.splitlines()[0]
        self.line.set('warning', self.state['error'])
        print(text)

    def _fill(self):
        stt = self.state.get('status')
        if stt is None:
            return
        n = self.state['views']
        tag = self.state['cam_tag']
        what = f'{tag}–{self.state["lidar"]}' if self.state.get('lidar') else tag
        if stt['locked']:
            self.views_lbl.setText(f'{what}: {n} views · every indicator within the requirement')
        else:
            w = stt['locks'][stt['worst_key']]
            self.views_lbl.setText(f'{what}: {n} view{"s" if n != 1 else ""} · loosest {w["label"].lower()} '
                                   + ('not determined' if w['ratio'] >= CC.UNDETERMINED else f'{w["ratio"]:.1f}×'))
        self.bars.set_locks(stt['locks'])
        need = self.state.get('need') or {}
        if need:
            txt = f'Requirement 1σ {need["deg"]:g}°' + (f' and {need["mm"]:g} mm' if 'mm' in need else '')
            src = need.get('source_deg', '')
            self.need.setText(f'{txt}, from {src}.')
        adv = self.state.get('advice')
        if stt['locked']:
            self.advice.setText('Done: every indicator is within the requirement. Finish to run the calibration.')
            self.advice_v.setText('')
        elif adv is None:
            self.advice.setText('No hand-held pose left that adds a new view.')
            self.advice_v.setText('')
        else:
            self.advice.setText(adv.text)
            self.advice_v.setText(f'{adv.text_vehicle} Works on {CC.LOCK_LABELS[adv.target_lock].lower()}: worst '
                                  f'{_times(adv.worst_before)} → {_times(adv.worst_after)} the requirement.')
        d = self.state.get('det')
        if d is None:
            self.line.set('', '')
        elif d['accepted']:
            self.line.set('good', f'View {n} taken (sharpness {d["sharpness"]:.0f}).')
        elif not d['found']:
            self.line.set('', 'No board in the image.')
        else:
            self.line.set('warning', d['reason'] or 'Board found: press Accept view to take it.')
        self._sync_buttons()

    # ------------------------------------------------------------------ drawing
    def cam_uid(self):
        if self.state['cam_uid']:
            return self.state['cam_uid']
        st = self.settings
        cams = [c for c in self.ctx.store.variant.cameras if c.visible]
        c = next((c for c in cams if c.tag == st.camera), cams[0] if cams else None)
        return c.uid if c is not None else None

    def overlay(self) -> dict:
        adv = self.state.get('advice')
        stt = self.state.get('status')
        return {'ghost': adv.ghost if adv is not None and not (stt and stt['locked']) else None,
                'det': self.state.get('det'), 'coverage': self.state.get('coverage')}

    def paint(self, p, rect):
        sensor = self.state.get('sensor')
        if not sensor:
            return
        paint_overlay(p, rect, sensor, self.overlay(), self.state.get('slot', 0))

    # ------------------------------------------------------------------ the simulated technician
    def _tech_source(self, tag=None):
        s = self.page.monitor.streams.get(tag or self.state['cam_tag'])
        return s.source if s is not None and isinstance(s.source, TechnicianSource) else None

    def set_technician(self, kind):
        self.tech_seg.set_value(kind)
        if kind == 'off':
            self.remove_technician()
        else:
            src = self._tech_source()
            self.install_technician()                    # a new technician steps in with a board
            if src is not None and self.coach is not None:  # one already there changes how it chooses: move now
                self._submit('move', lambda: (src.show(self._tech_pose(self.state.get('advice'))),
                                              self._snapshot())[1])

    def install_technician(self) -> bool:
        """Swap the coached camera's synthetic stream for the simulated technician's (only a synthetic stream)."""
        kind = self.tech_seg.value()
        if not self.active or kind == 'off' or self.tech is None or not self.state['built']:
            return False
        tag = self.state['cam_tag']
        s = self.page.monitor.streams.get(tag)
        if s is None or rh.describe(s.spec) != 'synthetic':
            if s is not None or kind != 'off':
                self.ctx.msg(f'The simulated technician needs a synthetic stream on {tag}', 5000)
            return False
        if isinstance(s.source, TechnicianSource):
            s.source.tech = self.tech
            return True
        old = s.source
        old.stop(wait=False)
        src = TechnicianSource(self.tech, old.fps, old.size_wanted, old.on_frame)
        s.source = src
        self._tech_tag = tag
        if self.page.running:
            src.start()
        # the technician steps in with the board where the coach asks (no empty frame to search first)
        self._submit('move', lambda: (src.show(self._tech_pose(self.state.get('advice'))), self._snapshot())[1])
        return True

    def remove_technician(self):
        """Back to the camera's own synthetic stream."""
        tag = self._tech_tag or self.state['cam_tag']
        self._tech_tag = ''
        if self._tech_source(tag) is None:
            return
        s = self.page.monitor.streams.pop(tag, None)
        if s is not None:
            s.source.stop(wait=False)
        self.page.monitor.update(self.ctx.store.project, self.page._cams(), self.page._engine)

    # ------------------------------------------------------------------ finish
    def _wait_idle(self, timeout=30.0):
        """Run the jobs queued before (the coach is the worker's): wait for the pool to drain."""
        fut = self.pool.submit(lambda: None)
        fut.result(timeout)

    def finish(self, folder=None, run=True):
        """Hand the accepted views to the calibration: lens → Calibrate ▸ Lens calibration with them as its views;
        lidar → the frames (and, from the simulated technician, the lidar's scans of the same poses) written as
        image + point-cloud pairs, the lidar–camera calibration pointed at that folder and opened."""
        if self.coach is None:
            return None
        self._wait_idle()
        if self.coach.st.mode == 'lens':
            return self.finish_lens()
        return self.finish_lidar(folder, run)

    def finish_lens(self):
        c = self.coach
        views = c.lens_views()
        if len(views) < 4:
            self.ctx.msg('The lens calibration needs at least 4 views', 5000)
            return None
        w = self.ctx.window
        if w is None or not hasattr(w, 'lens_calibration'):
            return None
        d = w.lens_calibration(c.cam.uid)
        if d is None:
            return None
        d.cols.set_value(c.board.cols)
        d.rows.set_value(c.board.rows)
        d.square.set_value(c.board.square_mm)
        d.clear_views()
        d._add([(f'coach_{v[0]}', v[1]) for v in views])
        self.ctx.msg(f'Lens calibration: {len(views)} views from the coach · Run', 6000)
        return d

    def finish_lidar(self, folder=None, run=True):
        """Write the views as pose01.png … (with the simulated technician also the lidar's scans, pose01.pcd …) on the
        worker thread, then point the lidar–camera calibration at the folder and open it (and run it, simulated)."""
        c = self.coach
        n = sum(1 for v in c.views if v.frame is not None)
        if n < 3:
            self.ctx.msg('The lidar–camera calibration needs at least 3 views', 5000)
            return None
        if not folder:
            folder = QFileDialog.getExistingDirectory(self, 'Folder for the views (pose01.png …)')
        if not folder:
            return None
        simulated = self._tech_source() is not None

        def work():
            from ..core import lidar_calib as LC
            from ..core import capture as cap
            views = [v for v in c.views if v.frame is not None]
            os.makedirs(folder, exist_ok=True)
            if simulated:
                # the simulated technician's lidar: its scans of the same poses (the lidar–camera calibration's own
                # simulation), each paired with the coach's frame of that pose
                sim = LC.simulate(c.project, c.engine, c.cam, c.lidar, [v.pose for v in views], c.lst) or []
                for k, (v, cp) in enumerate(zip(views, sim)):
                    cp.image, cp.name = v.frame, f'pose{k + 1:02d}'
                LC.save_captures(sim, folder)
            else:
                for k, v in enumerate(views):
                    cap.save_image(os.path.join(folder, f'pose{k + 1:02d}.png'), v.frame)
            with open(os.path.join(folder, 'calib_coach_session.json'), 'w', encoding='utf-8') as fh:
                json.dump(c.session_doc(), fh, indent=1, ensure_ascii=False)
            return {'folder': folder, 'views': len(views), 'simulated': simulated, 'run': run, 'camera': c.tag,
                    'lidar': c.lidar.tag}
        self.ctx.msg(f'Writing {n} views to {folder}…', 4000)
        self._submit('finished_lidar', work)
        return folder

    def _open_lidar(self, out):
        d = dict(self.ctx.store.project.analysis.lidar_calib or {})
        d.update(source='folder', folder=out['folder'], camera=out['camera'], lidar=out['lidar'])

        def fn(p):
            p.analysis.lidar_calib = d
        self.ctx.store.edit('analysis', fn, f'Lidar–camera calibration from the coach\'s {out["views"]} views')
        w = self.ctx.window
        if w is not None and hasattr(w, 'page'):
            w.set_page('lidar_calib')
            pg = w.page('lidar_calib')
            if out['simulated'] and out['run'] and pg is not None:
                pg.run()
        if not out['simulated']:
            self.ctx.msg(f'Saved {out["views"]} views to {out["folder"]}: put the lidar\'s scan of each beside it '
                         '(pose01.pcd …), then Run', 8000)

    def save_session(self, folder=None):
        if self.coach is None or not self.state['views']:
            return []
        if not folder:
            folder = QFileDialog.getExistingDirectory(self, 'Folder for the coach\'s views and session')
        if not folder:
            return []
        self._wait_idle()
        from ..core import capture as cap
        os.makedirs(folder, exist_ok=True)
        paths = []
        for v in self.coach.views:
            if v.frame is not None:
                pth = os.path.join(folder, f'{v.name}.png')
                cap.save_image(pth, v.frame)
                paths.append(pth)
        jp = os.path.join(folder, 'calib_coach_session.json')
        with open(jp, 'w', encoding='utf-8') as fh:
            json.dump(self.coach.session_doc(), fh, indent=1, ensure_ascii=False)
        paths.append(jp)
        self.ctx.msg(f'Saved {len(paths) - 1} view(s) and calib_coach_session.json to {folder}', 6000)
        return paths

    def shutdown(self):
        self.gen += 1
        try:
            self.remove_technician()
        except (RuntimeError, AttributeError):          # the page is going too
            pass
        self.pool.shutdown(wait=False, cancel_futures=True)


def _times(r) -> str:
    return 'not determined' if not math.isfinite(r) or r >= CC.UNDETERMINED else f'{r:.1f}×'
