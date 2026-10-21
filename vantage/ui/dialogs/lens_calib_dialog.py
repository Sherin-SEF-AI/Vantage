"""Calibrate ▸ Lens calibration: measure one camera's own intrinsics from checkerboard views, and import unit
calibrations made elsewhere (OpenCV, ROS, Kalibr, JSON)."""
from __future__ import annotations

import math
import os
import threading
import traceback

import numpy as np
from PyQt6.QtCore import Qt, QObject, QRectF, QPointF, QTimer, pyqtSignal
from PyQt6.QtGui import QPainter, QPen, QColor, QFont
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QComboBox, QTableWidget, QTableWidgetItem,
                             QHeaderView, QAbstractItemView, QProgressBar, QFileDialog, QLineEdit, QSplitter, QStackedWidget,
                             QMessageBox)

from ...core import calib, capture as cap
from ...core.lens import resolve
from ..theme import THEME, STATUS, mono
from ..widgets import label, button, ScrubSpinBox, SegmentedControl, ROW_H, hline
from ..workspaces.install_ws import to_qimage
from .base import VDialog

VIEW_COLS = ['View', 'Corners', 'RMS px', 'Status']


class _Sig(QObject):
    progress = pyqtSignal(float, str)
    done = pyqtSignal(object)
    failed = pyqtSignal(str)


def _table(cols, stretch=0):
    t = QTableWidget(0, len(cols))
    t.setHorizontalHeaderLabels(cols)
    t.verticalHeader().setVisible(False)
    t.verticalHeader().setDefaultSectionSize(ROW_H + 1)
    t.horizontalHeader().setFixedHeight(ROW_H)
    t.setShowGrid(False)
    t.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    t.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
    t.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    hh = t.horizontalHeader()
    for c in range(len(cols)):
        hh.setSectionResizeMode(c, QHeaderView.ResizeMode.ResizeToContents)
    hh.setSectionResizeMode(stretch, QHeaderView.ResizeMode.Stretch)
    return t


def _cell(text, num=False, color=None):
    it = QTableWidgetItem(text)
    if num:
        it.setFont(mono(12))
        it.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    if color:
        it.setForeground(QColor(color))
    return it


class ViewPreview(QWidget):
    """One calibration view with its detected corners, and the corner coverage of all used views."""

    def __init__(self):
        super().__init__()
        self.img = None
        self.qimg = None
        self.corners = None
        self.coverage = None
        self.caption = ''
        self.setMinimumSize(320, 240)

    def set_view(self, img, corners, coverage, caption=''):
        self.img = img
        self.qimg = to_qimage(img) if img is not None else None
        self.corners = corners
        self.coverage = coverage
        self.caption = caption
        self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), THEME.color('canvas'))
        if self.qimg is None:
            p.setPen(THEME.color('muted'))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, 'Add views, then run the calibration')
            p.end()
            return
        iw, ih = self.qimg.width(), self.qimg.height()
        s = min((self.width() - 8) / iw, (self.height() - 26) / ih)
        w, h = iw * s, ih * s
        r = QRectF((self.width() - w) / 2, 4, w, h)
        p.drawImage(r, self.qimg)
        if self.coverage is not None:
            gy, gx = self.coverage.shape
            mx = max(1.0, float(self.coverage.max()))
            for j in range(gy):
                for i in range(gx):
                    v = self.coverage[j, i]
                    cell = QRectF(r.x() + i * w / gx, r.y() + j * h / gy, w / gx, h / gy)
                    if v <= 0:
                        c = QColor(STATUS['warning'])
                        c.setAlpha(36)
                    else:
                        c = QColor(STATUS['good'])
                        c.setAlpha(int(14 + 40 * min(1.0, v / mx)))
                    p.fillRect(cell, c)
        if self.corners is not None and len(self.corners):
            p.setPen(QPen(QColor('#ff4fd8'), 1.2))
            for u, v in self.corners:
                p.drawEllipse(QPointF(r.x() + (u + 0.5) * s, r.y() + (v + 0.5) * s), 2.4, 2.4)
            p.setPen(QPen(QColor('#ff4fd8'), 1.0))
            a = self.corners[0]
            p.drawEllipse(QPointF(r.x() + (a[0] + 0.5) * s, r.y() + (a[1] + 0.5) * s), 5, 5)
        p.setPen(THEME.color('ink2'))
        p.setFont(QFont(p.font().family(), 9))
        p.drawText(QRectF(0, r.bottom() + 3, self.width(), 18), Qt.AlignmentFlag.AlignCenter, self.caption)
        p.end()


class LensCalibDialog(VDialog):
    def __init__(self, ctx, uid=None, parent=None):
        super().__init__(parent, 'Lens calibration', 'this camera\'s own intrinsics', (1240, 760))
        self.ctx = ctx
        self.views = []              # [(name, image)]
        self.truth = None            # simulated unit record (synthetic source)
        self.result = None
        self.busy = False
        self.live = None
        self.sig = _Sig()
        self.sig.progress.connect(self._progress)
        self.sig.done.connect(self._done)
        self.sig.failed.connect(self._failed)
        split = QSplitter(Qt.Orientation.Horizontal)
        split.setHandleWidth(2)
        split.addWidget(self._build_left())
        split.addWidget(self._build_centre())
        split.addWidget(self._build_right())
        split.setSizes([330, 520, 390])
        lay = QVBoxLayout(self.body)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(split)
        self.hint.setText('cv::fisheye (Kannala–Brandt) model · stored on the camera; it replaces the library lens in the '
                          'analysis, the installation check, the remap tables and every export')
        b_close = button('Close')
        b_close.clicked.connect(self.reject)
        self.b_apply = button('Apply to camera', 'primary')
        self.b_apply.clicked.connect(self.apply)
        self.buttons.addWidget(b_close)
        self.buttons.addWidget(self.b_apply)
        self.live_timer = QTimer(self)
        self.live_timer.setInterval(150)
        self.live_timer.timeout.connect(self._live_tick)
        cams = [c for c in ctx.store.variant.cameras]
        for c in cams:
            self.cam.addItem(f'{c.tag} · {c.location or c.name}', c.uid)
        sel = uid or (ctx.store.selected.uid if ctx.store.selected else (cams[0].uid if cams else None))
        if sel:
            self.cam.setCurrentIndex(max(0, self.cam.findData(sel)))
        self._cam_changed()
        self._sync()

    # ------------------------------------------------------------------ layout
    def _build_left(self):
        w = QWidget()
        w.setObjectName('panel')
        ll = QVBoxLayout(w)
        ll.setContentsMargins(12, 8, 10, 10)
        ll.setSpacing(5)
        ll.addWidget(label('Camera', bold=True))
        self.cam = QComboBox()
        self.cam.currentIndexChanged.connect(self._cam_changed)
        ll.addWidget(self.cam)
        self.cam_info = label('', muted=True, small=True, wrap=True)
        ll.addWidget(self.cam_info)
        ll.addSpacing(4)
        ll.addWidget(label('Board', bold=True))
        g = QGridLayout()
        g.setVerticalSpacing(2)
        self.cols = ScrubSpinBox('', 'plain', 0, '', step=1, width=90)
        self.rows = ScrubSpinBox('', 'plain', 0, '', step=1, width=90)
        self.square = ScrubSpinBox('', 'plain', 1, 'mm', step=1, width=90)
        self.cols.set_value(9)
        self.rows.set_value(6)
        self.square.set_value(60)
        for i, (t, x) in enumerate((('Inner corners ↔', self.cols), ('Inner corners ↕', self.rows), ('Square', self.square))):
            a = label(t, ink2=True)
            a.setFixedWidth(120)
            g.addWidget(a, i, 0)
            g.addWidget(x, i, 1)
        g.setColumnStretch(2, 1)
        ll.addLayout(g)
        ll.addSpacing(4)
        h = QHBoxLayout()
        h.addWidget(label('Views', bold=True))
        h.addStretch(1)
        self.n_views = label('', muted=True, small=True, mono_=True)
        h.addWidget(self.n_views)
        ll.addLayout(h)
        self.source = SegmentedControl([('images', 'Images'), ('synthetic', 'Synthetic'), ('live', 'Live')], 'images')
        self.source.changed.connect(self._source_changed)
        ll.addWidget(self.source)
        self.src_stack = QStackedWidget()
        # images
        pi = QWidget()
        li = QVBoxLayout(pi)
        li.setContentsMargins(0, 4, 0, 0)
        li.addWidget(label('Hold a flat checkerboard in front of the camera at 10 to 20 places, near the edges and corners '
                           'of the image too, tilted a little each time. Every image must be the same size.', muted=True,
                           small=True, wrap=True))
        r = QHBoxLayout()
        b = button('Add images…', None, 'file')
        b.clicked.connect(self.add_images)
        r.addWidget(b)
        b = button('Folder…', None, 'folder')
        b.clicked.connect(self.add_folder)
        r.addWidget(b)
        r.addStretch(1)
        li.addLayout(r)
        li.addStretch(1)
        self.src_stack.addWidget(pi)
        # synthetic
        ps = QWidget()
        ls = QVBoxLayout(ps)
        ls.setContentsMargins(0, 4, 0, 0)
        ls.addWidget(label('Renders a hand-held board seen by a simulated unit that differs from the datasheet lens, so '
                           'the result can be compared with the truth.', muted=True, small=True, wrap=True))
        gs = QGridLayout()
        gs.setVerticalSpacing(2)
        self.syn_n = ScrubSpinBox('', 'plain', 0, '', step=1, width=90)
        self.syn_f = ScrubSpinBox('', 'plain', 1, '%', step=0.1, width=90)
        self.syn_pp = ScrubSpinBox('', 'plain', 0, 'px', step=1, width=90)
        self.syn_seed = ScrubSpinBox('', 'plain', 0, '', step=1, width=90)
        for x, v in ((self.syn_n, 20), (self.syn_f, 1.5), (self.syn_pp, 15), (self.syn_seed, 1)):
            x.set_value(v)
        for i, (t, x) in enumerate((('Views', self.syn_n), ('Focal ±', self.syn_f), ('Centre ±', self.syn_pp),
                                    ('Seed', self.syn_seed))):
            a = label(t, ink2=True)
            a.setFixedWidth(120)
            gs.addWidget(a, i, 0)
            gs.addWidget(x, i, 1)
        gs.setColumnStretch(2, 1)
        ls.addLayout(gs)
        b = button('Render views', None, 'camera-wide')
        b.clicked.connect(self.render_synthetic)
        r = QHBoxLayout()
        r.addWidget(b)
        r.addStretch(1)
        ls.addLayout(r)
        ls.addStretch(1)
        self.src_stack.addWidget(ps)
        # live
        pl = QWidget()
        lv = QVBoxLayout(pl)
        lv.setContentsMargins(0, 4, 0, 0)
        lv.addWidget(label('A V4L2 device, a GStreamer pipeline (contains !), an RTSP URL or a video file. Move the board and '
                           'grab a view at each place.', muted=True, small=True, wrap=True))
        self.live_spec = QLineEdit()
        self.live_spec.setFont(mono(11))
        lv.addWidget(self.live_spec)
        r = QHBoxLayout()
        self.b_live = button('Start', None, 'camera-wide')
        self.b_live.clicked.connect(self.toggle_live)
        self.b_grab = button('Grab view', 'primary')
        self.b_grab.clicked.connect(self.grab_view)
        r.addWidget(self.b_live)
        r.addWidget(self.b_grab)
        r.addStretch(1)
        lv.addLayout(r)
        self.live_state = label('', muted=True, small=True, wrap=True)
        lv.addWidget(self.live_state)
        lv.addStretch(1)
        self.src_stack.addWidget(pl)
        ll.addWidget(self.src_stack)
        r = QHBoxLayout()
        b = button('Clear views', None, 'minus')
        b.clicked.connect(self.clear_views)
        r.addWidget(b)
        r.addStretch(1)
        ll.addLayout(r)
        ll.addSpacing(4)
        ll.addWidget(label('Serial number', bold=True))
        self.serial = QLineEdit()
        self.serial.setPlaceholderText('printed on the camera, optional')
        ll.addWidget(self.serial)
        ll.addStretch(1)
        self.status = label('', muted=True, small=True, wrap=True)
        ll.addWidget(self.status)
        self.bar = QProgressBar()
        self.bar.setTextVisible(False)
        self.bar.setRange(0, 1000)
        ll.addWidget(self.bar)
        r = QHBoxLayout()
        r.addStretch(1)
        self.b_run = button('Calibrate', 'primary')
        self.b_run.clicked.connect(self.run)
        r.addWidget(self.b_run)
        ll.addLayout(r)
        w.setMinimumWidth(300)
        return w

    def _build_centre(self):
        w = QWidget()
        w.setObjectName('panel')
        cl = QVBoxLayout(w)
        cl.setContentsMargins(8, 6, 8, 8)
        cl.setSpacing(4)
        h = QHBoxLayout()
        h.addWidget(label('Views'))
        h.addStretch(1)
        h.addWidget(label('magenta: detected corners (ring: first) · tint: corner coverage of the used views', muted=True,
                          small=True))
        cl.addLayout(h)
        self.preview = ViewPreview()
        cl.addWidget(self.preview, 3)
        self.vtable = _table(VIEW_COLS, stretch=3)
        self.vtable.cellClicked.connect(lambda r, c: self._show_view(r))
        cl.addWidget(self.vtable, 2)
        return w

    def _build_right(self):
        w = QWidget()
        w.setObjectName('panel')
        rl = QVBoxLayout(w)
        rl.setContentsMargins(10, 6, 12, 8)
        rl.setSpacing(5)
        rl.addWidget(label('Result'))
        rl.addWidget(hline())
        self.summary = label('Not calibrated yet.', wrap=True)
        rl.addWidget(self.summary)
        self.advice = label('', small=True, wrap=True)
        rl.addWidget(self.advice)
        self.ptable = _table(['Parameter', 'Library', 'This unit', 'Difference'], stretch=3)
        rl.addWidget(self.ptable, 1)
        self.diff = label('', muted=True, small=True, wrap=True)
        rl.addWidget(self.diff)
        r = QHBoxLayout()
        self.b_save = button('Save…', None, 'export', 'Write this calibration as OpenCV YAML or Vantage JSON')
        self.b_save.clicked.connect(self.save)
        r.addWidget(self.b_save)
        r.addStretch(1)
        rl.addLayout(r)
        return w

    # ------------------------------------------------------------------ state
    def _camera(self):
        return self.ctx.store.cam(self.cam.currentData())

    def _cam_changed(self, *_):
        c = self._camera()
        if c is None:
            return
        p = self.ctx.store.project
        m, l = p.module_for(c), p.nominal_lens_for(c)
        cur = c.calib
        txt = f'{m.name} · {l.name} ({l.label or "lens"}) · {m.width_px} × {m.height_px}'
        if cur:
            txt += (f'\nUnit calibration: {cur.get("serial") or "no serial"} · {cur.get("rms_px", 0):.3f} px · '
                    f'{cur.get("views", 0)} views · {cur.get("date", "")} · from {cur.get("source") or "?"}')
            self.serial.setText(cur.get('serial', ''))
        else:
            txt += '\nNo unit calibration: the library lens is used.'
        self.cam_info.setText(txt)
        self.live_spec.setText(cap.default_spec(c.port))
        self.b_apply.setText(f'Apply to {c.tag}')
        self.result = None
        self._fill_result()

    def _source_changed(self, k):
        self.src_stack.setCurrentIndex({'images': 0, 'synthetic': 1, 'live': 2}[k])
        if k != 'live':
            self.stop_live()

    def _pattern(self):
        return int(max(3, self.cols.value())), int(max(3, self.rows.value()))

    def _sync(self):
        self.n_views.setText(f'{len(self.views)} view' + ('' if len(self.views) == 1 else 's'))
        self.b_run.setEnabled(not self.busy and len(self.views) >= 4)
        ok = self.result is not None and self.result.ok
        self.b_apply.setEnabled(ok and not self.busy)
        self.b_save.setEnabled(ok)

    def _fill_views(self):
        res = {v.name: v for v in self.result.views} if self.result else {}
        self.vtable.setRowCount(len(self.views))
        for i, (name, img) in enumerate(self.views):
            v = res.get(name)
            corners = '–' if v is None or v.corners is None else str(len(v.corners))
            rms = '–' if v is None or not v.used else f'{v.rms:.3f}'
            if v is None:
                st, col = f'{img.shape[1]} × {img.shape[0]}', None
            elif v.used:
                st, col = 'used', STATUS['good']
            else:
                st, col = v.note or 'not used', STATUS['warning']
            for j, t in enumerate((name, corners, rms, st)):
                self.vtable.setItem(i, j, _cell(t, 1 <= j <= 2, col if j == 3 else None))
        if self.views:
            self._show_view(min(max(0, self.vtable.currentRow()), len(self.views) - 1))
        else:
            self.preview.set_view(None, None, None)
        self._sync()

    def _show_view(self, row):
        if not (0 <= row < len(self.views)):
            return
        name, img = self.views[row]
        v = next((x for x in self.result.views if x.name == name), None) if self.result else None
        cov = self.result.coverage if self.result is not None and self.result.ok else None
        cap_ = f'{name} · {img.shape[1]} × {img.shape[0]}'
        if v is not None and v.corners is not None:
            cap_ += f' · {len(v.corners)} corners' + (f' · {v.rms:.3f} px' if v.used else '')
        self.preview.set_view(img, None if v is None else v.corners, cov, cap_)

    def _fill_result(self):
        r = self.result
        self.ptable.setRowCount(0)
        self.diff.setText('')
        self.advice.setText('')
        if r is None:
            self.summary.setText('Not calibrated yet.')
            self._sync()
            return
        if not r.ok:
            self.summary.setText(r.message)
            self._sync()
            return
        c = self._camera()
        p = self.ctx.store.project
        m, base = p.module_for(c), p.nominal_lens_for(c)
        self.summary.setText(r.message)
        if r.advice:
            self.advice.setText('\n'.join('• ' + a for a in r.advice))
            self.advice.setStyleSheet(f'color: {STATUS["warning"]}')
        else:
            self.advice.setText('Good spread of views and a low residual.')
            self.advice.setStyleSheet(f'color: {STATUS["good"]}')
        rows = calib.compare_rows(m, base, r.record)
        rows.append(('RMS', '', f'{r.rms:.3f} px', ''))
        rows.append(('Image', '', f'{r.width} × {r.height}', 'scaled to the sensor' if r.width != m.width_px else ''))
        self.ptable.setRowCount(len(rows))
        for i, row in enumerate(rows):
            for j, t in enumerate(row):
                self.ptable.setItem(i, j, _cell(t, j > 0))
        unit = resolve(m, calib.unit_lens(base, m, r.record))
        field = self._calibrated_theta(unit)
        d = calib.model_difference(resolve(m, base), unit, max_theta=field)
        txt = (f'Within the field the views cover (up to {math.degrees(field):.0f}° off axis), the library lens puts a ray up '
               f'to {d["max_px"]:.1f} px ({d["rms_px"]:.1f} px RMS, {d["max_deg"]:.2f}°) away from where this unit images it.')
        if self.truth is not None:
            t = resolve(m, calib.unit_lens(base, m, self.truth))
            e = calib.model_difference(unit, t, max_theta=field)
            txt += (f'\nAgainst the simulated truth: {e["rms_px"]:.2f} px RMS, {e["max_px"]:.2f} px max, centre '
                    f'{e["centre_px"]:.2f} px.')
        self.diff.setText(txt)
        self._sync()

    def _calibrated_theta(self, unit):
        r = self.result
        pts = np.concatenate([v.corners for v in r.views if v.used])
        s = unit.width / r.width
        _, th = unit.unproject((pts[:, 0] + 0.5) * s, (pts[:, 1] + 0.5) * s)
        return float(th.max())

    # ------------------------------------------------------------------ sources
    def _add(self, items):
        names = {n for n, _ in self.views}
        for n, img in items:
            k, base = 2, n
            while n in names:
                n, k = f'{base} ({k})', k + 1
            names.add(n)
            self.views.append((n, img))
        self.result = None
        self._fill_views()
        self._fill_result()

    def add_images(self, paths=None):
        if not paths:
            paths, _ = QFileDialog.getOpenFileNames(self, 'Calibration images', os.path.expanduser('~'),
                                                    'Images (' + ' '.join('*' + e for e in cap.IMAGE_EXTS) + ')')
        items, bad = [], []
        for pth in paths or []:
            try:
                items.append((os.path.basename(pth), cap.load_image(pth)))
            except Exception as e:
                bad.append(f'{os.path.basename(pth)}: {e}')
        self.truth = None
        self._add(items)
        if bad:
            self.status.setText('Skipped ' + '; '.join(bad[:3]) + (f' and {len(bad) - 3} more' if len(bad) > 3 else ''))

    def add_folder(self, folder=None):
        folder = folder or QFileDialog.getExistingDirectory(self, 'Folder of calibration images')
        if folder:
            self.add_images([os.path.join(folder, f) for f in sorted(os.listdir(folder)) if f.lower().endswith(cap.IMAGE_EXTS)])

    def clear_views(self):
        self.views = []
        self.result = None
        self.truth = None
        self._fill_views()
        self._fill_result()

    def render_synthetic(self):
        if self.busy:
            return
        c = self._camera()
        p = self.ctx.store.project
        m, base = p.module_for(c), p.nominal_lens_for(c)
        seed = int(self.syn_seed.value())
        truth = calib.simulated_unit(m, base, seed=seed, f_pct=abs(self.syn_f.value()), pp_px=abs(self.syn_pp.value()))
        n, pat, sq = int(max(4, self.syn_n.value())), self._pattern(), max(5.0, self.square.value())
        self._start('Rendering…')

        def work():
            try:
                v = calib.board_views(m, base, truth, pat, sq, n=n, seed=seed,
                                      progress=lambda f, t: self.sig.progress.emit(f, t))
                self.sig.done.emit(('views', v, truth))
            except Exception as e:
                self.sig.failed.emit(f'{type(e).__name__}: {e}\n{traceback.format_exc()}')
        threading.Thread(target=work, daemon=True).start()

    def toggle_live(self):
        if self.live is not None:
            self.stop_live()
            return
        self.live = cap.LiveSource(self.live_spec.text().strip())
        self.live.start()
        self.live_timer.start()
        self.b_live.setText('Stop')

    def stop_live(self):
        if self.live is not None:
            self.live.stop()
            self.live = None
        self.live_timer.stop()
        if hasattr(self, 'b_live'):
            self.b_live.setText('Start')

    def _live_tick(self):
        ls = self.live
        if ls is None:
            return
        f = ls.latest()
        if ls.error:
            self.live_state.setText(ls.error)
        elif f is not None:
            self.live_state.setText(f'{ls.size[0]} × {ls.size[1]} · {ls.frames} frames')
            self.preview.set_view(f, None, None, 'live')

    def grab_view(self):
        f = self.live.latest() if self.live else None
        if f is None:
            self.status.setText('No live frame yet.')
            return
        self.truth = None
        self._add([(f'grab_{len(self.views) + 1:02d}', f)])

    # ------------------------------------------------------------------ run
    def _start(self, text):
        self.busy = True
        self.bar.setValue(0)
        self.status.setText(text)
        self._sync()

    def run(self):
        if self.busy or len(self.views) < 4:
            return
        c = self._camera()
        p = self.ctx.store.project
        m, base = p.module_for(c), p.nominal_lens_for(c)
        views, pat, sq = list(self.views), self._pattern(), max(1.0, self.square.value())
        serial = self.serial.text().strip()
        src = 'synthetic' if self.truth is not None else ('live' if self.source.value() == 'live' else 'images')
        self._start('Finding the board…')

        def work():
            try:
                r = calib.calibrate(views, pat, sq, m, base, progress=lambda f, t: self.sig.progress.emit(f, t),
                                    serial=serial, source=src)
                self.sig.done.emit(('result', r, None))
            except Exception as e:
                self.sig.failed.emit(f'{type(e).__name__}: {e}\n{traceback.format_exc()}')
        threading.Thread(target=work, daemon=True).start()

    def _progress(self, f, t):
        self.bar.setValue(int(f * 1000))
        self.status.setText(t)

    def _done(self, out):
        self.busy = False
        self.bar.setValue(1000)
        kind, val, extra = out
        if kind == 'views':
            self.views = []
            self._add(val)
            self.truth = extra
            self.serial.setText(extra.get('serial', ''))
            self.status.setText(f'Rendered {len(val)} views of a simulated unit (focal {extra["fx"]:.1f} px, centre '
                                f'{extra["cx"]:.1f}, {extra["cy"]:.1f})')
        else:
            self.result = val
            self.status.setText(val.message)
            self._fill_views()
            self._fill_result()
        self._sync()

    def _failed(self, m):
        self.busy = False
        self.status.setText('Failed: ' + m.splitlines()[0])
        print(m)
        self._sync()

    # ------------------------------------------------------------------ outputs
    def apply(self):
        r = self.result
        c = self._camera()
        if r is None or not r.ok or c is None:
            return
        rec = calib.clean_record(dict(r.record, serial=self.serial.text().strip() or r.record.get('serial', '')))
        apply_calibrations(self.ctx, {c.uid: rec}, f'Calibrated the lens of {c.tag} ({rec["rms_px"]:.3f} px)')
        self._cam_changed()
        self.result = r
        self._fill_result()
        self.ctx.msg(f'{c.tag} now uses its own lens calibration', 5000)

    def save(self, path=None):
        r = self.result
        if r is None or not r.ok:
            return
        c = self._camera()
        if not path:
            path, _ = QFileDialog.getSaveFileName(self, 'Save lens calibration', f'{c.tag}.yaml',
                                                  'OpenCV YAML (*.yaml);;Vantage JSON (*.json)')
        if not path:
            return
        rec = dict(r.record, serial=self.serial.text().strip() or r.record.get('serial', ''))
        with open(path, 'w', encoding='utf-8') as fh:
            fh.write(calib.to_opencv_yaml(rec, c.tag) if not path.lower().endswith('.json') else
                     calib.to_json(self.ctx.store.project, [_with_calib(c, rec)]))
        self.ctx.msg(f'Saved {path}')

    def closeEvent(self, e):
        self.stop_live()
        super().closeEvent(e)

    def done(self, r):
        self.stop_live()
        super().done(r)


def _with_calib(cam, rec):
    import copy
    c = copy.copy(cam)
    c.calib = calib.clean_record(rec)
    return c


def apply_calibrations(ctx, recs: dict, text: str):
    """Store unit calibrations {uid: record} on the cameras of the active variant (one undo step)."""
    def fn(p):
        for c in p.variant.cameras:
            if c.uid in recs:
                c.calib = calib.clean_record(recs[c.uid]) if recs[c.uid] else {}
    ctx.store.edit('variant', fn, text)
    ctx.analysis.request()


class ImportCalibDialog(VDialog):
    """Read calibration files and assign each unit calibration to a camera."""
    COLS = ['File', 'Name in file', 'Size', 'fx', 'cx', 'cy', 'Camera', 'Check']

    def __init__(self, ctx, paths, parent=None):
        super().__init__(parent, 'Import unit calibrations', 'OpenCV · ROS camera_info · Kalibr · JSON', (1080, 460))
        self.ctx = ctx
        self.records, errors = [], []
        for pth in paths:
            try:
                self.records += calib.read_file(pth)
            except Exception as e:
                errors.append(f'{os.path.basename(pth)}: {e}')
        cams = list(ctx.store.variant.cameras)
        self.cams = cams
        matched, _ = calib.match(self.records, cams)
        by_rec = {id(r): uid for uid, r in matched.items()}
        lay = QVBoxLayout(self.body)
        lay.setContentsMargins(14, 10, 14, 10)
        lay.addWidget(label('Each calibration is matched to a camera by the name or topic in the file, the file name, or the '
                            'serial number. Change the camera where the match is wrong; leave it empty to skip one.',
                            muted=True, small=True, wrap=True))
        self.table = _table(self.COLS, stretch=7)
        self.table.setRowCount(len(self.records))
        self.combos = []
        for i, r in enumerate(self.records):
            vals = [os.path.basename(r.get('file', '')), r.get('hint', ''), f'{r["width"]} × {r["height"]}',
                    f'{r["fx"]:.2f}', f'{r["cx"]:.2f}', f'{r["cy"]:.2f}']
            for j, t in enumerate(vals):
                self.table.setItem(i, j, _cell(t, j >= 3))
            cb = QComboBox()
            cb.addItem('— skip —', '')
            for c in cams:
                cb.addItem(f'{c.tag} · {c.location or c.name}', c.uid)
            cb.setCurrentIndex(max(0, cb.findData(by_rec.get(id(r), ''))))
            cb.currentIndexChanged.connect(lambda _=0, k=i: self._check(k))
            self.table.setCellWidget(i, 6, cb)
            self.combos.append(cb)
            self._check(i)
        lay.addWidget(self.table, 1)
        self.err = label('\n'.join(errors), small=True, wrap=True)
        self.err.setStyleSheet(f'color: {STATUS["serious"]}')
        self.err.setVisible(bool(errors))
        lay.addWidget(self.err)
        self.hint.setText(f'{len(self.records)} calibration(s) in {len(paths)} file(s)')
        b_close = button('Cancel')
        b_close.clicked.connect(self.reject)
        self.b_ok = button('Apply', 'primary')
        self.b_ok.clicked.connect(self.apply)
        self.buttons.addWidget(b_close)
        self.buttons.addWidget(self.b_ok)
        self.b_ok.setEnabled(bool(self.records))

    def _check(self, i):
        uid = self.combos[i].currentData()
        c = self.ctx.store.cam(uid) if uid else None
        if c is None:
            self.table.setItem(i, 7, _cell('skipped', color=THEME.color('muted').name()))
            return
        p = self.ctx.store.project
        probs = calib.problems(self.records[i], p.module_for(c), p.nominal_lens_for(c))
        self.table.setItem(i, 7, _cell('; '.join(probs) if probs else 'fits this camera',
                                       color=STATUS['warning'] if probs else STATUS['good']))

    def assignments(self) -> dict:
        out = {}
        for r, cb in zip(self.records, self.combos):
            if cb.currentData():
                out[cb.currentData()] = r
        return out

    def apply(self):
        a = self.assignments()
        if not a:
            self.reject()
            return
        dup = len(a) < sum(1 for cb in self.combos if cb.currentData())
        if dup and QMessageBox.question(self, 'Import', 'Two calibrations are assigned to the same camera; the last one '
                                        'wins. Continue?') != QMessageBox.StandardButton.Yes:
            return
        tags = ', '.join(self.ctx.store.cam(u).tag for u in a)
        apply_calibrations(self.ctx, a, f'Imported unit calibrations for {tags}')
        self.ctx.msg(f'{len(a)} camera(s) now use their own lens calibration', 5000)
        self.accept()
