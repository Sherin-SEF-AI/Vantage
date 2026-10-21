"""D1 · Installation check: capture each camera, find the floor boards, solve the installed pose,
compare with the plan and stitch the bird's-eye view with planned and measured poses."""
from __future__ import annotations

import json
import math
import os
import threading
import time
import traceback

import numpy as np
from PyQt6.QtCore import Qt, QTimer, QObject, QRectF, QPointF, pyqtSignal
from PyQt6.QtGui import QPainter, QImage, QPixmap, QPen, QColor, QFont, QPainterPath
from PyQt6.QtWidgets import (QWidget, QHBoxLayout, QVBoxLayout, QGridLayout, QSplitter, QCheckBox, QComboBox, QTableWidget,
                             QTableWidgetItem, QHeaderView, QAbstractItemView, QProgressBar, QFileDialog, QScrollArea,
                             QFrame, QLineEdit, QSizePolicy)

from ...core import capture as cap
from ...core.install import InstallSettings, check_camera
from ...core.mathutil import rot_zyx, wrap180
from ...core.project import Variant, new_uid
from ..theme import THEME, STATUS, mono, sans
from ..widgets import label, button, tool_button, ScrubSpinBox, SegmentedControl, Chip, ROW_H, hline

SOURCES = [('synthetic', 'Synthetic'), ('file', 'File'), ('live', 'Live')]


def to_qimage(img: np.ndarray) -> QImage:
    img = np.ascontiguousarray(img)
    h, w = img.shape[:2]
    if img.ndim == 2:
        return QImage(img.data, w, h, w, QImage.Format.Format_Grayscale8).copy()
    return QImage(img.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()


def _fmt_delta(v, dp):
    return f'{v:+.{dp}f}'.replace('-', '−')


class _Signals(QObject):
    progress = pyqtSignal(float, str)
    done = pyqtSignal(object)
    failed = pyqtSignal(str)


# ---------------------------------------------------------------------- image tiles
class CaptureTile(QWidget):
    clicked = pyqtSignal(str)
    double_clicked = pyqtSignal(str)

    def __init__(self, page, uid):
        super().__init__()
        self.page = page
        self.uid = uid
        self.setMinimumSize(160, 120)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._pix = None
        self._pix_key = None

    def _cam(self):
        return self.page.ctx.store.cam(self.uid)


    def _fit(self, w, h):
        top = 22
        s = min(self.width() / w, (self.height() - top) / h)
        dw, dh = w * s, h * s
        return QRectF((self.width() - dw) / 2, top + (self.height() - top - dh) / 2, dw, dh), s

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), THEME.color('viewport'))
        cam = self._cam()
        sel = self.page.selected == self.uid
        p.setPen(QPen(THEME.color('ink2' if sel else 'hair'), 1))
        p.drawRect(QRectF(0.5, 0.5, self.width() - 1, self.height() - 1))
        if cam is None:
            p.end()
            return
        # header: chip + tag + source + status
        p.fillRect(QRectF(8, 7, 8, 8), THEME.cam(cam.slot))
        p.setFont(mono(11, QFont.Weight.DemiBold))
        p.setPen(THEME.color('ink'))
        p.drawText(QRectF(21, 0, 80, 22), Qt.AlignmentFlag.AlignVCenter, cam.tag)
        src = self.page.sources.get(self.uid, {}).get('kind', 'synthetic')
        r = self.page.results.get(self.uid)
        p.setFont(sans(11))
        p.setPen(THEME.color('muted'))
        info = dict(SOURCES)[src]
        img, ikey = self.page.display_image(self.uid)
        if img is not None:
            info += f' · {img.shape[1]} × {img.shape[0]}'
        if src == 'live':
            ls = self.page.live.get(self.uid)
            if ls is not None and ls.error:
                info += ' · ' + ls.error
            elif ls is not None and self.uid not in self.page.images:
                info += ' · live'
        p.drawText(QRectF(60, 0, self.width() - 150, 22), Qt.AlignmentFlag.AlignVCenter, info)
        if r is not None and r.status != 'none':
            txt = {'good': 'OK', 'warning': 'Adjust', 'serious': 'Re-mount'}.get(r.status, r.status)
            fm = p.fontMetrics()
            w = fm.horizontalAdvance(txt) + 14
            box = QRectF(self.width() - w - 6, 4, w, 15)
            c = QColor(STATUS.get(r.status, '#888'))
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(c)
            p.drawRoundedRect(box, 3, 3)
            p.setPen(QColor('#111111'))
            p.drawText(box, Qt.AlignmentFlag.AlignCenter, txt)
        if img is None:
            p.setPen(THEME.color('muted'))
            p.drawText(QRectF(0, 22, self.width(), self.height() - 22), Qt.AlignmentFlag.AlignCenter,
                       'No image yet · Capture or load a file')
            p.end()
            return
        h, w = img.shape[:2]
        rect, s = self._fit(w, h)
        key = (ikey, int(rect.width()), int(rect.height()))
        if self._pix_key != key:
            qi = to_qimage(img)
            self._pix = QPixmap.fromImage(qi).scaled(int(rect.width()), int(rect.height()),
                                                     Qt.AspectRatioMode.IgnoreAspectRatio,
                                                     Qt.TransformationMode.SmoothTransformation)
            self._pix_key = key
        p.drawPixmap(rect.topLeft(), self._pix)
        if r is not None and self.page.show_overlay.isChecked():
            # planned outlines (dashed) and detected corners (status colour)
            sx = s * (w / max(1, r.img_w)) if getattr(r, 'img_w', 0) else s
            plan = QPen(THEME.color('ink', 0.85), 1.2, Qt.PenStyle.DashLine)
            col = QColor(STATUS.get(r.status, STATUS['good']))
            for b in r.boards:
                path = QPainterPath()
                for k, (u, v) in enumerate(b.outline):
                    pt = QPointF(rect.left() + u * sx, rect.top() + v * sx)
                    path.moveTo(pt) if k == 0 else path.lineTo(pt)
                path.closeSubpath()
                p.setPen(plan)
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawPath(path)
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(col)
                rad = max(1.2, min(3.0, 2.0 * s * 2))
                for u, v in b.image_pts:
                    p.drawEllipse(QPointF(rect.left() + u * sx, rect.top() + v * sx), rad, rad)
                p.setPen(THEME.color('ink'))
                p.setFont(sans(10))
                c0 = b.outline.mean(axis=0)
                p.drawText(QPointF(rect.left() + c0[0] * sx + 4, rect.top() + c0[1] * sx - 4), b.name)
        p.end()

    def mousePressEvent(self, e):
        self.clicked.emit(self.uid)

    def mouseDoubleClickEvent(self, e):
        self.double_clicked.emit(self.uid)


class ImageView(QWidget):
    """Fits one RGB image; used for the stitched bird's-eye view."""

    def __init__(self, empty='No stitched view yet'):
        super().__init__()
        self.img = None
        self.empty = empty
        self.caption = ''
        self.setMinimumSize(200, 160)
        self._pix = None
        self._key = None

    def set_image(self, img, caption=''):
        self.img = img
        self.caption = caption
        self._key = None
        self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        p.fillRect(self.rect(), THEME.color('viewport'))
        if self.img is None:
            p.setPen(THEME.color('muted'))
            p.setFont(sans(12))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap, self.empty)
            p.end()
            return
        h, w = self.img.shape[:2]
        s = min(self.width() / w, (self.height() - 18) / h)
        dw, dh = int(w * s), int(h * s)
        key = (id(self.img), dw, dh)
        if self._key != key:
            self._pix = QPixmap.fromImage(to_qimage(self.img)).scaled(dw, dh, Qt.AspectRatioMode.IgnoreAspectRatio,
                                                                       Qt.TransformationMode.SmoothTransformation)
            self._key = key
        x0, y0 = (self.width() - dw) // 2, (self.height() - 18 - dh) // 2
        p.drawPixmap(x0, y0, self._pix)
        if self.caption:
            p.setPen(THEME.color('muted'))
            p.setFont(sans(11))
            p.drawText(QRectF(0, self.height() - 18, self.width(), 18), Qt.AlignmentFlag.AlignCenter, self.caption)
        p.end()


# ---------------------------------------------------------------------- page
class InstallPage(QWidget):
    COLS = ['Camera', 'Source', 'Boards', 'Corners', 'RMS px', 'Δx', 'Δy', 'Δz mm', 'Δψ', 'Δθ', 'Δφ °', 'Status']

    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        self.setObjectName('base')
        self.sources = {}          # uid -> {'kind', 'path', 'spec'}
        self.images = {}           # uid -> captured (frozen) image
        self.truth = {}            # uid -> (dyaw, dpitch, droll, dx, dy, dz) simulated mounting error
        self.results = {}          # uid -> PoseResult
        self.live = {}             # uid -> capture.LiveSource
        self.stitched = {}         # 'planned' | 'measured' -> RGB
        self.luts = {}
        self.version = {}          # uid -> bumped whenever the frozen image changes (tile pixmap cache key)
        self.selected = None
        self.single = False
        self.busy = False
        self.sig = _Signals()
        self.sig.progress.connect(self._progress)
        self.sig.done.connect(self._done)
        self.sig.failed.connect(self._failed)
        self.rows = {}
        self.tiles = {}

        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        split = QSplitter(Qt.Orientation.Horizontal)
        split.setHandleWidth(2)
        split.addWidget(self._build_left())
        split.addWidget(self._build_centre())
        split.addWidget(self._build_right())
        split.setSizes([330, 1000, 520])
        split.setStretchFactor(1, 1)
        lay.addWidget(split)

        self.live_timer = QTimer(self)
        self.live_timer.setInterval(120)
        self.live_timer.timeout.connect(self._live_tick)
        ctx.store.changed.connect(self._on_store_changed)
        ctx.store.project_replaced.connect(self._on_project_replaced)
        self._rebuild()

    # ------------------------------------------------------------------ layout
    def _build_left(self):
        left = QWidget()
        left.setObjectName('panel')
        left.setMinimumWidth(300)
        ll = QVBoxLayout(left)
        ll.setContentsMargins(8, 6, 8, 8)
        ll.setSpacing(4)
        top = QHBoxLayout()
        top.addWidget(label('Installation check'))
        top.addStretch(1)
        top.addWidget(label('end of line', muted=True, small=True))
        ll.addLayout(top)
        ll.addWidget(hline())
        ll.addWidget(label('Lay the planned calibration boards on the floor (Calibrate ▸ Target planner), capture one image '
                           'per camera and run the check. Each camera\'s pose is solved from the boards it sees and compared '
                           'with the plan.', muted=True, small=True, wrap=True))
        ll.addSpacing(4)
        h = QHBoxLayout()
        h.addWidget(label('Sources', bold=True))
        h.addStretch(1)
        self.all_src = QComboBox()
        self.all_src.addItem('Set all…', '')
        for k, t in SOURCES:
            self.all_src.addItem(t, k)
        self.all_src.activated.connect(self._set_all_sources)
        h.addWidget(self.all_src)
        ll.addLayout(h)
        self.src_grid = QGridLayout()
        self.src_grid.setHorizontalSpacing(6)
        self.src_grid.setVerticalSpacing(3)
        self.src_grid.setColumnStretch(2, 1)
        ll.addLayout(self.src_grid)
        b = QHBoxLayout()
        self.b_folder = button('Load folder…', None, 'folder', 'Match image files to cameras by tag: F.png, cam_QR.jpg …')
        self.b_folder.clicked.connect(self.load_folder)
        self.b_save_caps = button('Save captures…', None, 'file', 'Write the current captures as <tag>.png')
        self.b_save_caps.clicked.connect(self.save_captures)
        b.addWidget(self.b_folder)
        b.addWidget(self.b_save_caps)
        b.addStretch(1)
        ll.addLayout(b)
        ll.addSpacing(6)
        ll.addWidget(label('Synthetic captures', bold=True))
        self.sim = QCheckBox('Simulate a mounting error')
        self.sim.setChecked(True)
        ll.addWidget(self.sim)
        grid = QGridLayout()
        grid.setVerticalSpacing(2)
        self.sim_ang = ScrubSpinBox('', 'angle', 1, '°', width=90)
        self.sim_pos = ScrubSpinBox('', 'plain', 0, 'mm', step=1, width=90)
        self.sim_seed = ScrubSpinBox('', 'plain', 0, '', step=1, width=90)
        self.sim_res = QComboBox()
        self.sim_res.addItem('Full', 1.0)
        self.sim_res.addItem('Half', 0.5)
        self.sim_res.setFixedWidth(90)
        for w, v in ((self.sim_ang, 1.5), (self.sim_pos, 15), (self.sim_seed, 7)):
            w.set_value(v)
        for i, (t, w) in enumerate((('Angle ±', self.sim_ang), ('Position ±', self.sim_pos), ('Seed', self.sim_seed),
                                    ('Resolution', self.sim_res))):
            a = label(t, ink2=True)
            a.setFixedWidth(110)
            grid.addWidget(a, i, 0)
            grid.addWidget(w, i, 1)
        grid.setColumnStretch(2, 1)
        ll.addLayout(grid)
        ll.addSpacing(6)
        ll.addWidget(label('Tolerance', bold=True))
        g2 = QGridLayout()
        g2.setVerticalSpacing(2)
        self.tol_pos = ScrubSpinBox('', 'plain', 0, 'mm', step=1, width=90)
        self.tol_ang = ScrubSpinBox('', 'angle', 2, '°', width=90)
        self.min_boards = ScrubSpinBox('', 'plain', 0, '', step=1, width=90)
        st = InstallSettings()
        self.tol_pos.set_value(st.pos_tol_mm)
        self.tol_ang.set_value(st.ang_tol_deg)
        self.min_boards.set_value(st.min_boards)
        for i, (t, w) in enumerate((('Position', self.tol_pos), ('Angle', self.tol_ang), ('Boards needed', self.min_boards))):
            a = label(t, ink2=True)
            a.setFixedWidth(110)
            g2.addWidget(a, i, 0)
            g2.addWidget(w, i, 1)
        g2.setColumnStretch(2, 1)
        ll.addLayout(g2)
        ll.addWidget(label('Within tolerance: OK. Up to 3× : adjust or keep as built. Beyond: re-mount.', muted=True, small=True,
                           wrap=True))
        ll.addStretch(1)
        self.status = label('Not run yet', muted=True, small=True, wrap=True)
        ll.addWidget(self.status)
        self.bar = QProgressBar()
        self.bar.setTextVisible(False)
        self.bar.setRange(0, 1000)
        ll.addWidget(self.bar)
        bl = QHBoxLayout()
        bl.addStretch(1)
        self.b_capture = button('Capture', None, 'camera-wide', 'Grab or render a fresh image for every camera')
        self.b_capture.clicked.connect(lambda: self.run(check=False))
        self.b_run = button('Run check', 'primary')
        self.b_run.clicked.connect(lambda: self.run(check=True))
        bl.addWidget(self.b_capture)
        bl.addWidget(self.b_run)
        ll.addLayout(bl)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(left)
        scroll.setMinimumWidth(300)
        return scroll

    def _build_centre(self):
        centre = QSplitter(Qt.Orientation.Vertical)
        centre.setHandleWidth(2)
        tw = QWidget()
        tw.setObjectName('panel')
        tl = QVBoxLayout(tw)
        tl.setContentsMargins(8, 4, 8, 6)
        tl.setSpacing(4)
        h = QHBoxLayout()
        h.addWidget(label('Captures'))
        h.addStretch(1)
        self.show_overlay = QCheckBox('Boards and corners')
        self.show_overlay.setChecked(True)
        self.show_overlay.toggled.connect(self._repaint_tiles)
        h.addWidget(self.show_overlay)
        h.addWidget(label('dashed: planned board · dots: detected corners · double-click to enlarge', muted=True, small=True))
        tl.addLayout(h)
        self.tile_host = QWidget()
        self.tile_grid = QGridLayout(self.tile_host)
        self.tile_grid.setContentsMargins(0, 0, 0, 0)
        self.tile_grid.setSpacing(6)
        tl.addWidget(self.tile_host, 1)
        centre.addWidget(tw)
        ww = QWidget()
        ww.setObjectName('panel')
        wl = QVBoxLayout(ww)
        wl.setContentsMargins(8, 4, 8, 6)
        h = QHBoxLayout()
        h.addWidget(label('Measured against plan'))
        h.addStretch(1)
        self.summary = label('', muted=True, small=True, mono_=True)
        h.addWidget(self.summary)
        wl.addLayout(h)
        self.table = QTableWidget(0, len(self.COLS))
        self.table.setHorizontalHeaderLabels(self.COLS)
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(ROW_H + 1)
        self.table.horizontalHeader().setFixedHeight(ROW_H)
        self.table.setShowGrid(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        hh = self.table.horizontalHeader()
        for c in range(len(self.COLS)):
            hh.setSectionResizeMode(c, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(len(self.COLS) - 1, QHeaderView.ResizeMode.Stretch)
        self.table.cellClicked.connect(self._row_clicked)
        wl.addWidget(self.table, 1)
        centre.addWidget(ww)
        centre.setSizes([560, 260])
        return centre

    def _build_right(self):
        right = QWidget()
        right.setObjectName('panel')
        rl = QVBoxLayout(right)
        rl.setContentsMargins(8, 4, 8, 8)
        rl.setSpacing(4)
        h = QHBoxLayout()
        h.addWidget(label('Stitched bird\'s-eye'))
        h.addStretch(1)
        self.which = SegmentedControl([('planned', 'Planned poses'), ('measured', 'Measured poses')], 'measured')
        self.which.changed.connect(self._show_bev)
        h.addWidget(self.which)
        rl.addLayout(h)
        self.bev = ImageView('Run the check to stitch the captures into a bird\'s-eye view.')
        rl.addWidget(self.bev, 1)
        rl.addWidget(label('With the planned poses a mounting error shows as broken lines at the seams; the measured poses '
                           'remove it. The remap tables are what the surround-view pipeline on the ECU loads.',
                           muted=True, small=True, wrap=True))
        g = QGridLayout()
        g.setHorizontalSpacing(6)
        g.setVerticalSpacing(4)
        self.b_png = button('Save PNG…', None, 'bev')
        self.b_png.clicked.connect(self.save_png)
        self.b_lut = button('Export remap tables…', None, 'export')
        self.b_lut.clicked.connect(self.export_luts)
        self.b_report = button('Save report…', None, 'report')
        self.b_report.clicked.connect(self.save_report)
        self.b_apply = button('Apply as as-built variant', 'primary', None,
                              'Copy the current variant with every solved camera moved to its measured pose')
        self.b_apply.clicked.connect(self.apply_as_built)
        g.addWidget(self.b_png, 0, 0)
        g.addWidget(self.b_report, 0, 1)
        g.addWidget(self.b_lut, 1, 0, 1, 2)
        g.addWidget(self.b_apply, 2, 0, 1, 2)
        rl.addLayout(g)
        right.setMinimumWidth(320)
        self._sync_buttons()
        return right

    # ------------------------------------------------------------------ cameras / sources
    def _cams(self):
        return [c for c in self.ctx.store.variant.cameras if c.visible]

    def _on_store_changed(self, sections):
        if 'variant' in sections or 'variants' in sections:
            uids = [c.uid for c in self._cams()]
            if uids != list(self.tiles):
                self._rebuild()
            else:
                self._fill_table()
                self._repaint_tiles()

    def _on_project_replaced(self, *_):
        self.stop_live()
        self.sources, self.images, self.truth, self.results, self.stitched, self.luts = {}, {}, {}, {}, {}, {}
        self._rebuild()

    def _rebuild(self):
        cams = self._cams()
        live_uids = set(self.live)
        for uid in live_uids - {c.uid for c in cams}:
            self.live.pop(uid).stop()
        while self.src_grid.count():
            w = self.src_grid.takeAt(0).widget()
            if w:
                w.deleteLater()
        self.rows = {}
        by_tag = {s.get('tag'): (u, s) for u, s in self.sources.items()}
        for i, c in enumerate(cams):
            if c.uid not in self.sources and c.tag in by_tag:
                old, src = by_tag[c.tag]
                self.sources[c.uid] = dict(src)
                if old in self.images:
                    self.set_image(c.uid, self.images[old])
            s = self.sources.setdefault(c.uid, {'kind': 'synthetic', 'path': '', 'spec': cap.default_spec(c.port)})
            s['tag'] = c.tag
            chip = Chip(c.slot, c.tag)
            combo = QComboBox()
            for k, t in SOURCES:
                combo.addItem(t, k)
            combo.setCurrentIndex([k for k, _ in SOURCES].index(s['kind']))
            combo.setFixedWidth(92)
            combo.currentIndexChanged.connect(lambda _=0, u=c.uid, cb=combo: self.set_source(u, cb.currentData()))
            edit = QLineEdit()
            edit.setFont(mono(11))
            edit.editingFinished.connect(lambda u=c.uid, ed=edit: self._edit_done(u, ed.text()))
            browse = tool_button('folder', 'Choose an image file')
            browse.clicked.connect(lambda _=False, u=c.uid: self.browse(u))
            self.src_grid.addWidget(chip, i, 0)
            self.src_grid.addWidget(combo, i, 1)
            self.src_grid.addWidget(edit, i, 2)
            self.src_grid.addWidget(browse, i, 3)
            self.rows[c.uid] = (combo, edit, browse)
            self._sync_row(c.uid)
        # tiles
        for t in self.tiles.values():
            t.setParent(None)
            t.deleteLater()
        self.tiles = {}
        for c in cams:
            t = CaptureTile(self, c.uid)
            t.clicked.connect(self.select)
            t.double_clicked.connect(self.toggle_single)
            self.tiles[c.uid] = t
        if self.selected not in self.tiles:
            self.selected = cams[0].uid if cams else None
            self.single = False
        self._layout_tiles()
        self._fill_table()

    def _layout_tiles(self):
        while self.tile_grid.count():
            self.tile_grid.takeAt(0)
        tiles = list(self.tiles.values())
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
        combo, edit, browse = self.rows[uid]
        s = self.sources[uid]
        kind = s['kind']
        edit.setEnabled(kind != 'synthetic')
        browse.setVisible(kind == 'file')
        edit.blockSignals(True)
        if kind == 'synthetic':
            edit.setText('')
            edit.setPlaceholderText('rendered from the plan')
        elif kind == 'file':
            edit.setText(s['path'])
            edit.setPlaceholderText('image file')
        else:
            edit.setText(s['spec'])
            edit.setPlaceholderText('/dev/video0, 0, rtsp://…, GStreamer pipeline')
        edit.blockSignals(False)
        edit.setToolTip(edit.text())

    def set_source(self, uid, kind):
        s = self.sources.get(uid)
        if s is None or not kind:
            return
        if s['kind'] == 'live' and kind != 'live':
            ls = self.live.pop(uid, None)
            if ls:
                ls.stop()
        s['kind'] = kind
        self.set_image(uid, None)
        self.truth.pop(uid, None)
        self.results.pop(uid, None)
        if uid in self.rows:
            combo = self.rows[uid][0]
            if combo.currentData() != kind:
                combo.blockSignals(True)
                combo.setCurrentIndex([k for k, _ in SOURCES].index(kind))
                combo.blockSignals(False)
            self._sync_row(uid)
        if kind == 'file' and s['path']:
            self._load_file(uid, s['path'])
        if kind == 'live':
            self._start_live(uid)
        self._after_sources()

    def _set_all_sources(self, i):
        kind = self.all_src.itemData(i)
        self.all_src.setCurrentIndex(0)
        if kind:
            for uid in list(self.sources):
                if uid in self.rows:
                    self.set_source(uid, kind)

    def _edit_done(self, uid, text):
        s = self.sources.get(uid)
        if s is None:
            return
        text = text.strip()
        if s['kind'] == 'file' and text != s['path']:
            s['path'] = text
            if text:
                self._load_file(uid, text)
        elif s['kind'] == 'live' and text and text != s['spec']:
            s['spec'] = text
            ls = self.live.pop(uid, None)
            if ls:
                ls.stop()
            self.set_image(uid, None)
            self._start_live(uid)
        self._after_sources()

    def browse(self, uid):
        cam = self.ctx.store.cam(uid)
        path, _ = QFileDialog.getOpenFileName(self, f'Image for {cam.tag if cam else "camera"}', self.sources[uid]['path'] or '',
                                              'Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.pgm *.ppm);;All files (*)')
        if path:
            self.sources[uid]['path'] = path
            self._sync_row(uid)
            self._load_file(uid, path)
            self._after_sources()

    def load_folder(self, folder=None):
        if not folder:
            folder = QFileDialog.getExistingDirectory(self, 'Folder with one image per camera')
        if not folder:
            return
        cams = self._cams()
        found = cap.match_folder(folder, [c.tag for c in cams])
        for c in cams:
            if c.tag in found:
                self.sources[c.uid].update(kind='file', path=found[c.tag])
                if c.uid in self.live:
                    self.live.pop(c.uid).stop()
                combo = self.rows[c.uid][0]
                combo.blockSignals(True)
                combo.setCurrentIndex(1)
                combo.blockSignals(False)
                self._sync_row(c.uid)
                self._load_file(c.uid, found[c.tag])
        missing = [c.tag for c in cams if c.tag not in found]
        self.ctx.msg(f'Loaded {len(found)} image(s) from {os.path.basename(folder)}' +
                     (f' · none for {", ".join(missing)}' if missing else ''), 6000)
        self._after_sources()

    def _load_file(self, uid, path):
        self.results.pop(uid, None)
        self.truth.pop(uid, None)
        try:
            img = cap.load_image(path)
        except Exception as e:
            self.set_image(uid, None)
            self.ctx.msg(f'{e}', 6000)
            return False
        cam = self.ctx.store.cam(uid)
        warn = self._aspect_warning(cam, img)
        if warn:
            self.ctx.msg(warn, 8000)
        self.set_image(uid, img)
        return True

    def _aspect_warning(self, cam, img):
        if cam is None:
            return ''
        from ...core.lens import resolve
        intr = resolve(self.ctx.store.project.module_for(cam), self.ctx.store.project.lens_for(cam))
        a_img = img.shape[1] / img.shape[0]
        a_sen = intr.width / intr.height
        if abs(a_img / a_sen - 1) > 0.01:
            return (f'{cam.tag}: the image is {img.shape[1]} × {img.shape[0]} but the sensor is {intr.width} × {intr.height}. '
                    'Use full-frame captures (no crop) or the pose will be wrong.')
        return ''

    def _start_live(self, uid):
        if not self.isVisible() and not self.busy:
            return
        ls = self.live.get(uid)
        if ls is None:
            ls = cap.LiveSource(self.sources[uid]['spec'])
            self.live[uid] = ls
        ls.start()
        if not self.live_timer.isActive():
            self.live_timer.start()

    def stop_live(self):
        self.live_timer.stop()
        for ls in self.live.values():
            ls.stop()
        self.live = {}

    def _live_tick(self):
        any_live = False
        for uid, ls in list(self.live.items()):
            if uid not in self.images and uid in self.tiles:
                self.tiles[uid].update()
            any_live = any_live or ls.running
        if not any_live and not self.live:
            self.live_timer.stop()

    def set_image(self, uid, img):
        if img is None:
            self.images.pop(uid, None)
        else:
            self.images[uid] = img
        self.version[uid] = self.version.get(uid, 0) + 1

    def display_image(self, uid):
        """(image, cache key): the frozen capture, or the newest live frame while a live source is previewing."""
        img = self.images.get(uid)
        if img is not None:
            return img, ('img', self.version.get(uid, 0))
        ls = self.live.get(uid)
        if ls is not None:
            f = ls.latest()
            if f is not None:
                return f, ('live', ls.frames)
        return None, None

    def _after_sources(self):
        self.stitched, self.luts = {}, {}
        self._fill_table()
        self._repaint_tiles()
        self._show_bev()
        self._sync_buttons()

    # ------------------------------------------------------------------ selection
    def select(self, uid):
        self.selected = uid
        cams = [c.uid for c in self._cams()]
        if uid in cams:
            self.table.selectRow(cams.index(uid))
        self._repaint_tiles()

    def toggle_single(self, uid):
        self.selected = uid
        self.single = not self.single
        self._layout_tiles()
        self._repaint_tiles()

    def _row_clicked(self, r, c):
        cams = self._cams()
        if 0 <= r < len(cams):
            self.select(cams[r].uid)

    def _repaint_tiles(self):
        for t in self.tiles.values():
            t.update()

    def activate(self):
        for uid, s in self.sources.items():
            if s['kind'] == 'live' and uid in self.rows:
                self._start_live(uid)

    def hideEvent(self, e):
        # keep cameras free for other programs while the page is not shown
        for ls in self.live.values():
            ls.stop()
        self.live_timer.stop()
        super().hideEvent(e)

    # ------------------------------------------------------------------ run
    def settings(self) -> InstallSettings:
        return InstallSettings(pos_tol_mm=max(0.1, self.tol_pos.value()), ang_tol_deg=max(0.01, self.tol_ang.value()),
                               min_boards=int(max(1, self.min_boards.value())))

    def run(self, check=True):
        if self.busy:
            return
        cams = self._cams()
        if not cams:
            self.ctx.msg('This variant has no cameras.')
            return
        project = self.ctx.store.project
        if check and not project.targets:
            self.ctx.msg('No calibration boards in the project. Place them with Calibrate ▸ Target planner first.', 6000)
            return
        jobs = []
        for c in cams:
            s = self.sources[c.uid]
            if s['kind'] == 'live':
                ls = self.live.get(c.uid)
                f = ls.latest() if ls else None
                if f is None:
                    self.ctx.msg(f'{c.tag}: no frame from {s["spec"]}' + (f' ({ls.error})' if ls and ls.error else ''), 6000)
                    continue
                self.set_image(c.uid, f)
            if s['kind'] == 'synthetic':
                jobs.append(('render', c.uid))
            elif c.uid in self.images:
                jobs.append(('have', c.uid))
        sim = self.sim.isChecked()
        rng = np.random.default_rng(int(self.sim_seed.value()))
        a, mm = abs(self.sim_ang.value()), abs(self.sim_pos.value()) / 1000
        errs = {}
        for c in cams:
            d = np.r_[rng.uniform(-a, a, 3), rng.uniform(-mm, mm, 3)] if sim else np.zeros(6)
            errs[c.uid] = d
        scale = float(self.sim_res.currentData())
        seed = int(self.sim_seed.value())
        st = self.settings()
        snap = project.snapshot()
        eng = self.ctx.window._shared_engine()
        result = self.ctx.analysis.result
        images = dict(self.images)
        self.busy = True
        self._sync_buttons()
        self.bar.setValue(0)
        self.status.setText('Capturing…')
        self._t0 = time.time()

        def work():
            from ...core.synth import render_capture
            from ...core.lens import resolve
            try:
                out = {'images': {}, 'truth': {}, 'results': {}, 'check': check}
                n = len(jobs) * (2 if check else 1) + (2 if check else 0)
                k = 0
                for kind, uid in jobs:
                    c = snap.variant.cam(uid)
                    if kind == 'render':
                        intr = resolve(snap.module_for(c), snap.lens_for(c))
                        d = errs[uid]
                        R = rot_zyx(c.yaw + d[0], c.pitch + d[1], c.roll + d[2])
                        pos = c.pos() + d[3:]
                        img = render_capture(snap, eng, c, int(intr.width * scale), int(intr.height * scale),
                                             pose=(R, pos), seed=seed * 31 + c.slot)
                        out['images'][uid] = img
                        out['truth'][uid] = d
                    else:
                        out['images'][uid] = images[uid]
                    k += 1
                    self.sig.progress.emit(k / n, f'Captured {c.tag}')
                if check:
                    for kind, uid in jobs:
                        c = snap.variant.cam(uid)
                        img = out['images'][uid]
                        r = check_camera(snap, c, img, st)
                        r.img_w = img.shape[1]
                        out['results'][uid] = r
                        k += 1
                        self.sig.progress.emit(k / n, f'Checked {c.tag}: {len(r.boards)} board(s)')
                    if result is not None:
                        from ...core.stitch import build_lut, stitch
                        lp = build_lut(snap, eng, result, res=0.02)
                        out['luts'] = {'planned': lp}
                        out['stitched'] = {'planned': stitch(lp, out['images'])}
                        k += 1
                        self.sig.progress.emit(k / n, 'Stitched with planned poses')
                        poses = {u: (rot_zyx(r.yaw, r.pitch, r.roll), r.pos) for u, r in out['results'].items()
                                 if r.pos is not None}
                        lm = build_lut(snap, eng, result, res=0.02, poses=poses)
                        out['luts']['measured'] = lm
                        out['stitched']['measured'] = stitch(lm, out['images'])
                        k += 1
                        self.sig.progress.emit(k / n, 'Stitched with measured poses')
                self.sig.done.emit(out)
            except Exception as e:
                self.sig.failed.emit(f'{type(e).__name__}: {e}\n{traceback.format_exc()}')
        self._thread = threading.Thread(target=work, daemon=True)
        self._thread.start()

    def _progress(self, frac, text):
        self.bar.setValue(int(frac * 1000))
        self.status.setText(text)

    def _done(self, out):
        self.busy = False
        self.bar.setValue(1000)
        for uid, img in out['images'].items():
            self.set_image(uid, img)
        for uid, d in out['truth'].items():
            self.truth[uid] = d
        if out['check']:
            self.results = dict(out['results'])
            self.luts = out.get('luts', {})
            self.stitched = out.get('stitched', {})
        else:
            self.results = {}
            self.stitched, self.luts = {}, {}
        secs = time.time() - self._t0
        if out['check']:
            res = list(self.results.values())
            ok = sum(r.status == 'good' for r in res)
            adj = sum(r.status == 'warning' for r in res)
            bad = sum(r.status == 'serious' for r in res)
            self.status.setText(f'Checked {len(res)} camera(s) in {secs:.1f} s · {ok} OK · {adj} adjust · {bad} re-mount')
            self.ctx.store.project.add_log('install', f'{len(res)} cameras · {ok} OK · {adj} adjust · {bad} re-mount')
            if self.ctx.analysis.result is None:
                self.ctx.msg('No analysis result yet, so the bird\'s-eye view was not stitched. Run the analysis (F5).', 6000)
        else:
            self.status.setText(f'Captured {len(out["images"])} image(s) in {secs:.1f} s')
        self._fill_table()
        self._repaint_tiles()
        self._show_bev()
        self._sync_buttons()

    def _failed(self, m):
        self.busy = False
        self._sync_buttons()
        self.status.setText('Failed: ' + m.splitlines()[0])
        print(m)

    def _sync_buttons(self):
        if not hasattr(self, 'b_apply'):
            return
        solved = any(r.pos is not None for r in self.results.values())
        for b in (self.b_run, self.b_capture, self.b_folder):
            b.setEnabled(not self.busy)
        self.b_apply.setEnabled(solved and not self.busy)
        self.b_report.setEnabled(bool(self.results))
        self.b_lut.setEnabled(bool(self.luts))
        self.b_png.setEnabled(bool(self.stitched))
        self.b_save_caps.setEnabled(bool(self.images))

    # ------------------------------------------------------------------ results
    def _fill_table(self):
        cams = self._cams()
        self.table.setRowCount(len(cams))
        worst_resid = []
        for i, c in enumerate(cams):
            r = self.results.get(c.uid)
            src = dict(SOURCES)[self.sources.get(c.uid, {}).get('kind', 'synthetic')]
            vals = [c.tag, src]
            if r is None:
                vals += ['–'] * 9 + ['Not checked' if c.uid in self.images or src == 'Synthetic' else 'No image']
            else:
                vals += [str(len(r.boards)), str(r.n_corners)]
                if r.pos is not None:
                    vals += [f'{r.rms_px:.2f}'] + [_fmt_delta(v, 1) for v in r.dpos_mm] + [_fmt_delta(v, 2) for v in r.dang_deg]
                else:
                    vals += ['–'] * 7
                vals.append(r.message)
            for j, v in enumerate(vals):
                it = QTableWidgetItem(v)
                if 2 <= j <= 10:
                    it.setFont(mono(12))
                    it.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                if j == 0:
                    it.setFont(mono(12, QFont.Weight.DemiBold))
                    it.setForeground(THEME.cam(c.slot))
                if j == 11 and r is not None and r.status in STATUS:
                    it.setForeground(QColor(STATUS[r.status]))
                if r is not None and r.pos is not None:
                    tol = self.settings()
                    if 5 <= j <= 7 and abs(r.dpos_mm[j - 5]) > tol.pos_tol_mm:
                        it.setForeground(QColor(STATUS['warning' if abs(r.dpos_mm[j - 5]) <= 3 * tol.pos_tol_mm else 'serious']))
                    if 8 <= j <= 10 and abs(r.dang_deg[j - 8]) > tol.ang_tol_deg:
                        it.setForeground(QColor(STATUS['warning' if abs(r.dang_deg[j - 8]) <= 3 * tol.ang_tol_deg else 'serious']))
                d = self.truth.get(c.uid)
                if d is not None and r is not None and r.pos is not None:
                    true_a = np.array([d[0], d[1], d[2]])
                    true_p = d[3:] * 1000
                    ra = np.abs(r.dang_deg - true_a).max()
                    rp = np.abs(r.dpos_mm - true_p).max()
                    worst_resid.append((ra, rp))
                    it.setToolTip(f'Simulated error: ψ {d[0]:+.2f}° θ {d[1]:+.2f}° φ {d[2]:+.2f}° · '
                                  f'x {true_p[0]:+.1f} y {true_p[1]:+.1f} z {true_p[2]:+.1f} mm\n'
                                  f'Recovered to within {ra:.2f}° / {rp:.1f} mm')
                self.table.setItem(i, j, it)
        if worst_resid:
            ra = max(a for a, _ in worst_resid)
            rp = max(p for _, p in worst_resid)
            self.summary.setText(f'simulated error recovered to within {ra:.2f}° · {rp:.1f} mm')
        elif self.results:
            solved = sum(r.pos is not None for r in self.results.values())
            self.summary.setText(f'{solved} / {len(self.results)} solved')
        else:
            self.summary.setText('')

    def _show_bev(self, *_):
        which = self.which.value() or 'measured'
        img = self.stitched.get(which)
        lut = self.luts.get(which)
        cap_ = ''
        if lut is not None:
            cap_ = (f'{"Measured" if which == "measured" else "Planned"} poses · {lut.nx} × {lut.ny} px · '
                    f'{lut.res * 1000:.0f} mm/px')
        self.bev.set_image(img, cap_)

    # ------------------------------------------------------------------ outputs
    def save_png(self):
        which = self.which.value() or 'measured'
        img = self.stitched.get(which)
        if img is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, 'Save stitched view', f'birds-eye-{which}.png', 'PNG (*.png)')
        if path:
            cap.save_image(path, img)
            self.ctx.msg(f'Saved {path}')

    def save_captures(self, folder=None):
        if not self.images:
            return
        if not folder:
            folder = QFileDialog.getExistingDirectory(self, 'Save captures into')
        if not folder:
            return
        os.makedirs(folder, exist_ok=True)
        n = 0
        for c in self._cams():
            img = self.images.get(c.uid)
            if img is not None:
                cap.save_image(os.path.join(folder, f'{c.tag}.png'), img)
                n += 1
        self.ctx.msg(f'Saved {n} capture(s) to {folder}')

    def export_luts(self, out_dir=None):
        lut = self.luts.get('measured') or self.luts.get('planned')
        if lut is None:
            return
        if not out_dir:
            out_dir = QFileDialog.getExistingDirectory(self, 'Write remap tables into')
        if not out_dir:
            return
        from ...core.stitch import export_lut
        paths = export_lut(lut, os.path.join(out_dir, 'bev_lut'), self.ctx.store.project)
        which = 'measured' if 'measured' in self.luts else 'planned'
        self.ctx.msg(f'Wrote {len(paths)} files ({which} poses) to {os.path.join(out_dir, "bev_lut")}', 6000)
        return paths

    def report(self) -> dict:
        cams = self._cams()
        st = self.settings()
        rows = []
        for c in cams:
            r = self.results.get(c.uid)
            row = {'tag': c.tag, 'source': self.sources.get(c.uid, {}).get('kind', ''),
                   'planned': {'x': c.x, 'y': c.y, 'z': c.z, 'yaw': c.yaw, 'pitch': c.pitch, 'roll': c.roll}}
            if r is not None:
                row.update(status=r.status, message=r.message, boards=[b.name for b in r.boards], corners=r.n_corners)
                if r.pos is not None:
                    row['measured'] = {'x': round(float(r.pos[0]), 4), 'y': round(float(r.pos[1]), 4),
                                       'z': round(float(r.pos[2]), 4), 'yaw': round(r.yaw, 3), 'pitch': round(r.pitch, 3),
                                       'roll': round(r.roll, 3)}
                    row['delta_mm'] = [round(float(v), 1) for v in r.dpos_mm]
                    row['delta_deg'] = [round(float(v), 3) for v in r.dang_deg]
                    row['rms_px'] = round(r.rms_px, 3)
            if c.uid in self.truth:
                d = self.truth[c.uid]
                row['simulated_error'] = {'deg': [round(float(v), 3) for v in d[:3]], 'mm': [round(float(v) * 1000, 1) for v in d[3:]]}
            rows.append(row)
        p = self.ctx.store.project
        return {'schema': 'vantage.install_check/1', 'project': p.name, 'variant': p.variant.name,
                'time': time.strftime('%Y-%m-%dT%H:%M:%S'),
                'frame': 'ISO 8855, rear-axle origin on the ground; angles yaw-pitch-roll in degrees, pitch positive down',
                'tolerance': {'position_mm': st.pos_tol_mm, 'angle_deg': st.ang_tol_deg, 'boards': st.min_boards},
                'cameras': rows}

    def save_report(self, path=None):
        if not self.results:
            return
        if not path:
            path, _ = QFileDialog.getSaveFileName(self, 'Save installation report', 'install-check.json', 'JSON (*.json)')
        if not path:
            return
        with open(path, 'w', encoding='utf-8') as fh:
            json.dump(self.report(), fh, indent=1, ensure_ascii=False)
        self.ctx.msg(f'Saved {path}')

    def apply_as_built(self):
        res = {u: r for u, r in self.results.items() if r.pos is not None}
        if not res:
            return
        cur = self.ctx.store.project.variant
        name = f'{cur.name} · as-built'

        def fn(p):
            v = Variant.from_dict(cur.to_dict())
            v.uid = new_uid()
            v.name = name
            for c in v.cameras:
                r = res.get(c.uid)
                c.uid = new_uid()
                if r is not None:
                    c.x, c.y, c.z = (round(float(t), 4) for t in r.pos)
                    c.yaw, c.pitch, c.roll = round(wrap180(r.yaw), 3), round(r.pitch, 3), round(wrap180(r.roll), 3)
                c.link_uid = ''            # measured poses are not mirror images of each other
            v.seams = {}
            p.variants.append(v)
            p.active_uid = v.uid
            return v
        self.ctx.store.edit('variants', fn, f'Applied measured poses as {name}')
        self.ctx.window._fill_variant_combo()
        self.ctx.msg(f'Created variant {name} with {len(res)} measured camera pose(s)', 6000)
