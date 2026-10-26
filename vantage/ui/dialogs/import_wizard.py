"""A4 · Import vehicle model wizard: file & units, orientation, wheel centres / ground / origin + spec check,
occluder & mount zones, summary."""
from __future__ import annotations

import copy
import math
import os
import threading

import numpy as np
from PyQt6.QtCore import Qt, QObject, pyqtSignal, QPointF, QRectF, QTimer
from PyQt6.QtGui import QColor, QPen, QPainter, QFont, QBrush
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QListWidget, QListWidgetItem,
                             QStackedWidget, QComboBox, QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView,
                             QSlider, QCheckBox, QLineEdit, QFileDialog, QPushButton, QMessageBox, QStyledItemDelegate, QStyle)

from ...core import loaders
from ...core.mesh import (Mesh, Part, Zone, ROLE_MOUNTABLE, ROLE_GLASS, ROLE_WHEELS, ROLE_IGNORED, ROLES,
                         body_width_points, is_mirror)
from ...core.project import Vehicle
from .. import icons
from ..theme import THEME, mono, sans, STATUS
from ..viewport3d import make_viewport, MeshDraw, LineDraw, TriDraw, shade_faces
from ..widgets import label, button, SegmentedControl, hline, vline, ROW_H, StatusLine, ScrubSpinBox, fix_height
from .base import VDialog

UNITS = [('m', 'Metres', 1.0), ('mm', 'Millimetres', 1e-3), ('cm', 'Centimetres', 1e-2), ('in', 'Inches', 0.0254)]
AXES = ['+X', '-X', '+Y', '-Y', '+Z', '-Z']
STEPS = ['File', 'Orient', 'Scale and origin', 'Body and zones', 'Summary']


def axis_vec(a):
    v = np.zeros(3)
    v['XYZ'.index(a[1])] = 1.0 if a[0] == '+' else -1.0
    return v


def detect_roles(parts):
    out = []
    for p in parts:
        n = (p.name + ' ' + p.material).lower()
        if any(k in n for k in ('glass', 'window', 'windshield', 'windscreen', 'windscr')) or p.alpha < 0.9:
            out.append(ROLE_GLASS)
        elif any(k in n for k in ('wheel', 'tyre', 'tire', 'rim', 'hub')):
            out.append(ROLE_WHEELS)
        elif any(k in n for k in ('interior', 'seat', 'dash', 'steering', 'cabin', 'engine')):
            out.append(ROLE_IGNORED)
        else:
            out.append(ROLE_MOUNTABLE)
    return out


def detect_glass(mesh: Mesh):
    gi = [i for i, p in enumerate(mesh.parts) if p.role == ROLE_GLASS or 'glass' in (p.name + p.material).lower() or p.alpha < 0.9]
    for i in gi:
        mesh.parts[i].role = ROLE_GLASS
    zi = next((i for i, z in enumerate(mesh.zones) if z.auto and z.kind == 'forbidden'), None)
    if zi is None:
        mesh.zones.append(Zone('Glass (auto)', 'forbidden', auto=True))
        zi = len(mesh.zones) - 1
    m = np.isin(mesh.face_part, gi)
    mesh.face_zone = np.where(m, zi, mesh.face_zone).astype(np.int16)
    mesh.zones[zi].name = f'Glass ({int(m.sum())} faces, auto)'
    return int(m.sum())


def guess_units(lo, hi, hint):
    ext = float(np.max(hi - lo))
    readings = [(u, name, ext * f) for u, name, f in UNITS]
    fits = [u for u, _, L in readings if 2.4 <= L <= 25.0]
    if hint in ('m', 'mm', 'cm', 'in') and (hint in fits or not fits):
        return hint, readings, True
    if len(fits) == 1:
        return fits[0], readings, True
    return (fits[0] if fits else 'm'), readings, False


class _LoadSig(QObject):
    loaded = pyqtSignal(object)
    failed = pyqtSignal(str)
    progress = pyqtSignal(float)


class PreviewScene(QObject):
    """Minimal scene for the wizard previews."""
    changed = pyqtSignal()

    def __init__(self, wiz, mode):
        super().__init__()
        self.wiz = wiz
        self.mode = mode            # 'orient' | 'side' | 'zones'
        self._key = None
        self._md = None
        self.hover_hit = None             # (point, normal) under the cursor; hover() is the provider method
        self.painting = False

    def background_color(self):
        return THEME.color('viewport')

    def background_image(self, vp):
        return None

    def mesh(self):
        m = self.wiz.world_mesh()
        if m is None:
            return None
        key = (id(m), self.mode, self.wiz.zone_version, THEME.name)
        if key != self._key:
            body = np.array(QColor(THEME.tok['body']).getRgbF()[:3])
            base = np.tile(body, (m.n_tris, 1))
            alpha = np.ones(m.n_tris)
            roles = [p.role for p in m.parts]
            for i, r in enumerate(roles):
                sel = m.face_part == i
                if r == ROLE_GLASS:
                    base[sel] = np.array(QColor(THEME.tok['glass']).getRgbF()[:3])
                elif r == ROLE_WHEELS:
                    base[sel] = 0.16
                elif r == ROLE_IGNORED:
                    alpha[sel] = 0.15
            if self.mode == 'zones':
                for zi, z in enumerate(m.zones):
                    sel = m.face_zone == zi
                    if z.kind == 'forbidden':
                        base[sel] = np.array([0.82, 0.23, 0.23])
                    else:
                        base[sel] = np.array([0.35, 0.7, 0.45])
            self._md = MeshDraw(m.tri.astype(np.float32), m.normals, base, alpha, version=id(m))
            self._key = key
        return self._md

    def ground(self):
        return None

    def lines(self, vp):
        out = []
        g = []
        for k in np.arange(-6, 6.01, 0.5):
            g.append([[k, -6, 0], [k, 6, 0]])
            g.append([[-6, k, 0], [6, k, 0]])
        out.append(LineDraw(np.array(g), THEME.color('grid1'), 1.0, segments=True))
        L = 0.6
        for v, c in ((np.array([L, 0, 0]), '#e0533f'), (np.array([0, L, 0]), '#3fae49'), (np.array([0, 0, L]), '#3d86e6')):
            out.append(LineDraw(np.array([[0, 0, 0], v]), QColor(c), 2.0, depth=False))
        return out

    def tris_extra(self, vp):
        return []

    def hud_label(self, vp):
        return {'orient': 'Preview · ' + self.wiz.orient_text() + ' · ISO 8855 triad',
                'side': 'Left side · click a wheel to pick its centre' if self.wiz.picking else 'Left side · wheel centres',
                'zones': f'Mount zones · brush {self.wiz.brush_mm():.0f} mm · drag to paint'}[self.mode]

    def overlay(self, p, vp):
        if self.mode == 'side':
            for k, c in self.wiz.wheels.items():
                q, _, ok = vp.project([c])
                if not ok[0]:
                    continue
                x, y = q[0]
                rr = self.wiz.wheel_r.get(k, 0.3) * vp.px_per_m(c)
                p.setPen(QPen(THEME.color('ink'), 1.2))
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawEllipse(QPointF(x, y), 9, 9)
                p.drawLine(QPointF(x - 9, y), QPointF(x + 9, y))
                p.drawLine(QPointF(x, y - 9), QPointF(x, y + 9))
                o = self.wiz.origin_world()
                txt = f'{k} {c[0] - o[0]:.3f} · {c[2] - o[2]:.3f}'
                p.setFont(mono(11))
                fw = p.fontMetrics().horizontalAdvance(txt) + 10
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QColor(0, 0, 0, 170))
                p.drawRect(QRectF(x + 12, y - 9, fw, 18))
                p.setPen(QColor(240, 239, 236))
                p.drawText(QPointF(x + 17, y + 4), txt)
        if self.mode == 'zones' and self.hover_hit is not None:
            hit, n = self.hover_hit
            r = self.wiz.brush_mm() / 1000
            t1 = np.cross([0, 0, 1.0], n)
            if np.linalg.norm(t1) < 1e-6:
                t1 = np.array([1.0, 0, 0])
            t1 /= np.linalg.norm(t1)
            t2 = np.cross(n, t1)
            a = np.linspace(0, 2 * math.pi, 32)
            pts = hit + (np.outer(np.cos(a), t1) + np.outer(np.sin(a), t2)) * r
            sc, _, ok = vp.project(pts)
            if ok.all():
                from PyQt6.QtGui import QPolygonF
                col = QColor(STATUS['critical']) if self.wiz.paint_kind == 'forbidden' else THEME.color('ink')
                p.setPen(QPen(col, 1.4))
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawPolygon(QPolygonF([QPointF(*q) for q in sc]))

    def _hit(self, vp, pos):
        m = self.wiz.world_mesh()
        if m is None:
            return None
        o, d = vp.ray(pos.x(), pos.y())
        t, f = m.raycast(o[None], d[None])
        if f[0] < 0:
            return None
        n = m.normals[f[0]]
        if n @ d > 0:
            n = -n
        return o + d * t[0], n, int(f[0])

    def mouse_press(self, vp, e):
        if e.button() != Qt.MouseButton.LeftButton:
            return False
        if self.mode == 'side' and self.wiz.picking:
            h = self._hit(vp, e.position())
            if h is not None:
                self.wiz.pick_wheel(h[0])
            return True
        if self.mode == 'zones' and self.wiz.paint_kind:
            self.painting = True
            self._paint(vp, e.position())
            return True
        return False

    def _paint(self, vp, pos):
        h = self._hit(vp, pos)
        if h is not None:
            self.wiz.paint_at(h[0], h[1])

    def mouse_move(self, vp, e):
        if self.painting:
            self._paint(vp, e.position())

    def mouse_release(self, vp, e):
        self.painting = False

    def click(self, vp, e):
        pass

    def double_click(self, vp, e):
        pass

    def hover(self, vp, pos):
        if self.mode == 'zones' and pos is not None:
            h = self._hit(vp, pos)
            self.hover_hit = (h[0], h[1]) if h else None
            self.changed.emit()

    def key_press(self, vp, e):
        return False

    def context_menu(self, vp, pos, gpos):
        pass

    def drag_enter(self, vp, e):
        return False

    def drag_move(self, vp, e):
        pass

    def drop(self, vp, e):
        pass


class StepDelegate(QStyledItemDelegate):
    def __init__(self, wiz):
        super().__init__()
        self.wiz = wiz

    def sizeHint(self, opt, idx):
        from PyQt6.QtCore import QSize
        return QSize(opt.rect.width(), 42)

    def paint(self, p, opt, idx):
        i = idx.row()
        r = opt.rect
        cur = i == self.wiz.step
        if cur:
            p.fillRect(r, THEME.color('sel'))
            p.fillRect(QRectF(r.left(), r.top(), 2, r.height()), THEME.color('ink'))
        done = i < self.wiz.step
        x = r.left() + 14
        cy = r.top() + 16
        if done:
            p.drawPixmap(int(x), int(cy - 6), icons.pixmap('good', STATUS['good'], 12))
        else:
            p.setPen(QPen(THEME.color('ink' if cur else 'strong'), 1))
            p.setBrush(THEME.color('ink') if cur else Qt.BrushStyle.NoBrush)
            p.drawRect(QRectF(x - 1, cy - 7, 14, 14))
            p.setFont(mono(10))
            p.setPen(THEME.color('panel') if cur else THEME.color('ink2'))
            p.drawText(QRectF(x - 1, cy - 7, 14, 14), Qt.AlignmentFlag.AlignCenter, str(i + 1))
        p.setFont(sans(12))
        p.setPen(THEME.color('ink' if cur or done else 'ink2'))
        p.drawText(QPointF(x + 22, cy + 4), STEPS[i])
        sub = self.wiz.step_summary(i) if i <= self.wiz.step else ''
        if sub:
            p.setFont(mono(10))
            p.setPen(THEME.color('muted'))
            p.drawText(QPointF(x + 22, cy + 19), sub)


class ImportWizard(VDialog):
    def __init__(self, ctx, parent=None, path=None, step=0, reuse=False, paint=None):
        super().__init__(parent, 'Import vehicle model', 'Step 1 of 5', (980, 680))
        self.ctx = ctx
        self.lm: loaders.LoadedMesh | None = None
        self.path = path or ''
        self.units = 'm'
        self.up, self.fwd = '+Z', '+X'
        self.flip_lr = False
        self.roles = []
        self.wb = {}               # wheel centres in base-mesh coordinates (scaled + rotated, before the origin shift)
        self.wheel_r = {}
        self.picking = None
        self.pick_order = []
        self.ground_mode = 'tyre'
        self.origin_mode = 'rear_axle'
        self.spec = {}
        self.occ_target = 8000
        self.paint_kind = paint
        self.zones = []
        self.face_zone = None
        self.zone_version = 0
        self._world = None
        self._world_key = None
        self.step = 0
        self.reuse = reuse
        self.sig = _LoadSig()
        self.sig.loaded.connect(self._loaded)
        self.sig.failed.connect(self._failed)
        self.setAcceptDrops(True)
        lay = QHBoxLayout(self.body)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        self.steps = QListWidget()
        self.steps.setObjectName('steps')
        self.steps.setFixedWidth(210)
        self.steps.setItemDelegate(StepDelegate(self))
        for s in STEPS:
            self.steps.addItem(s)
        self.steps.itemClicked.connect(lambda it: self.go(self.steps.row(it)) if self.steps.row(it) <= self._max_step() else None)
        lay.addWidget(self.steps)
        self.stack = QStackedWidget()
        lay.addWidget(self.stack, 1)
        self._page_file()
        self._page_orient()
        self._page_scale()
        self._page_zones()
        self._page_summary()
        self.b_cancel = button('Cancel', 'ghost')
        self.b_cancel.clicked.connect(self.reject)
        self.b_back = button('Back')
        self.b_back.clicked.connect(lambda: self.go(self.step - 1))
        self.b_next = button('Next', 'primary')
        self.b_next.clicked.connect(self._next)
        for b in (self.b_cancel, self.b_back, self.b_next):
            self.buttons.addWidget(b)
        if reuse and ctx.store.project.vehicle.mesh is not None:
            self._from_current()
            self.go(step)
        else:
            self.go(0)
            if path:
                QTimer.singleShot(50, lambda: self.load(path))

    # ------------------------------------------------------------------ pages
    def _page_file(self):
        w = QWidget()
        l = QVBoxLayout(w)
        l.setContentsMargins(16, 14, 16, 14)
        l.setSpacing(10)
        self.drop = QPushButton()
        self.drop.setProperty('role', 'ghost')
        self.drop.setStyleSheet(f'QPushButton {{ border: 1px dashed {THEME.tok["strong"]}; border-radius: 2px; '
                                f'min-height: 130px; max-height: 130px; }}')
        dl = QHBoxLayout(self.drop)
        dl.setContentsMargins(24, 0, 24, 0)
        ic = QLabel()
        ic.setPixmap(icons.pixmap('file', THEME.tok['ink2'], 16))
        col = QVBoxLayout()
        self.f_name = QLabel('Drop a vehicle model here')
        self.f_name.setFont(mono(13))
        self.f_meta = label('or press Browse…', muted=True, mono_=True, small=True)
        self.f_note = label('', muted=True, small=True)
        col.addStretch(1)
        col.addWidget(self.f_name)
        col.addWidget(self.f_meta)
        col.addWidget(self.f_note)
        col.addStretch(1)
        br = button('Browse…')
        br.clicked.connect(self._browse)
        dl.addWidget(ic)
        dl.addLayout(col, 1)
        dl.addWidget(br)
        for x in (ic,):
            x.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.drop.clicked.connect(self._browse)
        l.addWidget(self.drop)
        g = QGridLayout()
        g.addWidget(label('Supported', ink2=True), 0, 0)
        chips = QHBoxLayout()
        for e in loaders.SUPPORTED_EXTS[:-1]:
            c = QLabel(e)
            c.setFont(mono(11))
            c.setStyleSheet(f'border: 1px solid {THEME.tok["strong"]}; border-radius: 2px; padding: 1px 6px;')
            chips.addWidget(c)
        chips.addStretch(1)
        g.addLayout(chips, 0, 1)
        g.addWidget(label('Units', ink2=True), 1, 0)
        uh = QHBoxLayout()
        self.u_combo = QComboBox()
        for u, name, _ in UNITS:
            self.u_combo.addItem(name, u)
        self.u_combo.setFixedWidth(150)
        self.u_combo.activated.connect(lambda i: self._set_units(self.u_combo.itemData(i)))
        self.u_note = label('', muted=True, small=True)
        uh.addWidget(self.u_combo)
        uh.addWidget(self.u_note, 1)
        g.addLayout(uh, 1, 1)
        g.setColumnStretch(1, 1)
        l.addLayout(g)
        self.err = QWidget()
        self.err.setObjectName('errorBox')
        el = QVBoxLayout(self.err)
        el.setContentsMargins(12, 8, 12, 8)
        self.err_line = StatusLine('critical', '')
        el.addWidget(self.err_line)
        self.err_tbl = QGridLayout()
        el.addLayout(self.err_tbl)
        eb = QHBoxLayout()
        eb.addStretch(1)
        self.b_mm = button('Use millimetres', 'primary')
        self.b_mm.clicked.connect(lambda: self._set_units('mm'))
        eb.addWidget(self.b_mm)
        el.addLayout(eb)
        self.err.hide()
        l.addWidget(self.err)
        self.parts = QTableWidget(0, 4)
        self.parts.setHorizontalHeaderLabels(['Mesh', 'Triangles', 'Material', 'Role'])
        self.parts.verticalHeader().setVisible(False)
        self.parts.verticalHeader().setDefaultSectionSize(ROW_H + 2)
        self.parts.horizontalHeader().setFixedHeight(ROW_H)
        self.parts.setShowGrid(False)
        self.parts.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        hh = self.parts.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        hh.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        hh.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        l.addWidget(self.parts, 1)
        self.total = label('', mono_=True, ink2=True)
        l.addWidget(self.total)
        self.stack.addWidget(w)

    def _page_orient(self):
        w = QWidget()
        l = QVBoxLayout(w)
        l.setContentsMargins(16, 12, 16, 12)
        top = QHBoxLayout()
        col = QVBoxLayout()
        col.addWidget(label('Up axis', ink2=True))
        self.up_seg = SegmentedControl([(a, a.replace('-', '−')) for a in AXES], self.up, min_w=34)
        self.up_seg.changed.connect(lambda a: self._orient(up=a))
        col.addWidget(self.up_seg)
        top.addLayout(col)
        top.addSpacing(20)
        col = QVBoxLayout()
        col.addWidget(label('Forward axis', ink2=True))
        self.fw_seg = SegmentedControl([(a, a.replace('-', '−')) for a in AXES], self.fwd, min_w=34)
        self.fw_seg.changed.connect(lambda a: self._orient(fwd=a))
        col.addWidget(self.fw_seg)
        top.addLayout(col)
        top.addSpacing(20)
        col = QVBoxLayout()
        col.addWidget(label('Flip', ink2=True))
        fl = QHBoxLayout()
        fl.setSpacing(4)
        b1 = button('Up')
        b1.clicked.connect(lambda: self._orient(up=('-' if self.up[0] == '+' else '+') + self.up[1]))
        b2 = button('Forward')
        b2.clicked.connect(lambda: self._orient(fwd=('-' if self.fwd[0] == '+' else '+') + self.fwd[1]))
        self.b_lr = button('Left ↔ right')
        self.b_lr.setCheckable(True)
        self.b_lr.toggled.connect(lambda on: self._orient(lr=on))
        for b in (b1, b2, self.b_lr):
            fl.addWidget(b)
        col.addLayout(fl)
        top.addLayout(col)
        top.addStretch(1)
        l.addLayout(top)
        self.sc_orient = PreviewScene(self, 'orient')
        self.vp_orient = make_viewport(self.sc_orient)
        l.addWidget(self.vp_orient, 1)
        self.o_info = QHBoxLayout()
        self.bbox_lbl = label('', mono_=True)
        self.axis_lbl = label('', ink2=True)
        self.axis_ok = StatusLine('good', '', wrap=False)
        self.o_info.addWidget(self.bbox_lbl)
        self.o_info.addSpacing(20)
        self.o_info.addWidget(self.axis_lbl)
        self.o_info.addSpacing(20)
        self.o_info.addWidget(self.axis_ok, 1)
        l.addLayout(self.o_info)
        self.stack.addWidget(w)

    def _page_scale(self):
        w = QWidget()
        l = QVBoxLayout(w)
        l.setContentsMargins(16, 12, 16, 12)
        top = QHBoxLayout()
        self.sc_side = PreviewScene(self, 'side')
        self.vp_side = make_viewport(self.sc_side)
        top.addWidget(self.vp_side, 3)
        right = QVBoxLayout()
        right.addWidget(label('Wheel centres', bold=True))
        self.wheel_rows = {}
        g = QGridLayout()
        g.setVerticalSpacing(2)
        for i, k in enumerate(('RL', 'RR', 'FL', 'FR')):
            a = label(k, mono_=True)
            v = label('—', mono_=True, small=True)
            st = QLabel()
            b = button('Pick')
            b.clicked.connect(lambda _=False, k=k: self.start_pick(k))
            g.addWidget(a, i, 0)
            g.addWidget(v, i, 1)
            g.addWidget(st, i, 2)
            g.addWidget(b, i, 3)
            self.wheel_rows[k] = (v, st, b)
        right.addLayout(g)
        ab = button('Auto-detect from wheel parts')
        ab.clicked.connect(self.auto_wheels)
        right.addWidget(ab)
        gg = QGridLayout()
        gg.addWidget(label('Ground', ink2=True), 0, 0)
        self.ground_c = QComboBox()
        self.ground_c.addItem('Tyre contact', 'tyre')
        self.ground_c.addItem('Lowest point', 'lowest')
        self.ground_c.addItem('As in the file', 'file')
        self.ground_c.activated.connect(lambda i: self._set('ground_mode', self.ground_c.itemData(i)))
        gg.addWidget(self.ground_c, 0, 1)
        gg.addWidget(label('Origin', ink2=True), 1, 0)
        self.origin_c = QComboBox()
        for k, t in (('rear_axle', 'ISO 8855 · rear axle'), ('front_axle', 'ISO 8855 · front axle'), ('front_bumper', 'Front bumper'),
                     ('centre', 'Vehicle centre'), ('file', 'As in the file')):
            self.origin_c.addItem(t, k)
        self.origin_c.activated.connect(lambda i: self._set('origin_mode', self.origin_c.itemData(i)))
        gg.addWidget(self.origin_c, 1, 1)
        right.addLayout(gg)
        right.addStretch(1)
        top.addLayout(right, 2)
        l.addLayout(top, 3)
        sh = QHBoxLayout()
        sh.addWidget(label('Spec check', bold=True))
        sh.addStretch(1)
        self.spec_src = label('tolerance ± 5 mm · double-click a spec value to edit', muted=True, small=True)
        sh.addWidget(self.spec_src)
        l.addLayout(sh)
        self.spec_t = QTableWidget(5, 5)
        self.spec_t.setHorizontalHeaderLabels(['Dimension', 'Measured m', 'Spec m', 'Δ mm', 'Status'])
        self.spec_t.verticalHeader().setVisible(False)
        self.spec_t.verticalHeader().setDefaultSectionSize(ROW_H)
        self.spec_t.horizontalHeader().setFixedHeight(ROW_H)
        self.spec_t.setShowGrid(False)
        self.spec_t.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.spec_t.setFixedHeight(ROW_H * 6 + 4)
        self.spec_t.itemChanged.connect(self._spec_edit)
        l.addWidget(self.spec_t)
        self.stack.addWidget(w)

    def _page_zones(self):
        w = QWidget()
        l = QVBoxLayout(w)
        l.setContentsMargins(16, 12, 16, 12)
        oh = QHBoxLayout()
        oh.addWidget(label('Occluder', bold=True))
        oh.addStretch(1)
        oh.addWidget(label('used for ray tests · full mesh stays for display', muted=True, small=True))
        l.addLayout(oh)
        sh = QHBoxLayout()
        sh.addWidget(label('Simplify', ink2=True))
        self.occ = QSlider(Qt.Orientation.Horizontal)
        self.occ.setRange(1000, 60000)
        self.occ.setValue(self.occ_target)
        self.occ.sliderReleased.connect(self._occ)
        self.occ_lbl = label('', mono_=True, small=True)
        sh.addWidget(self.occ, 1)
        sh.addWidget(self.occ_lbl)
        l.addLayout(sh)
        mid = QHBoxLayout()
        self.sc_zones = PreviewScene(self, 'zones')
        self.vp_zones = make_viewport(self.sc_zones)
        mid.addWidget(self.vp_zones, 3)
        right = QVBoxLayout()
        right.addWidget(label('Paint', bold=True))
        self.paint_seg = SegmentedControl([('allowed', 'Allowed'), ('forbidden', 'Forbidden'), ('none', 'Off')], self.paint_kind or 'none')
        self.paint_seg.changed.connect(self._paint_mode)
        right.addWidget(self.paint_seg)
        zr = QHBoxLayout()
        zr.addWidget(label('Zone', ink2=True))
        self.zone_name = QLineEdit('New zone')
        zr.addWidget(self.zone_name, 1)
        right.addLayout(zr)
        br = QHBoxLayout()
        br.addWidget(label('Brush', ink2=True))
        self.brush = ScrubSpinBox('', 'plain', 0, 'mm', step=5)
        self.brush.set_value(40)
        br.addWidget(self.brush)
        right.addLayout(br)
        self.glass_cb = QCheckBox('Mark glass as forbidden')
        self.glass_cb.setChecked(True)
        self.glass_cb.toggled.connect(self._glass)
        right.addWidget(self.glass_cb)
        self.glass_note = label('', muted=True, small=True, wrap=True)
        right.addWidget(self.glass_note)
        self.zt = QTableWidget(0, 3)
        self.zt.setHorizontalHeaderLabels(['Kind', 'Zone', 'Faces'])
        self.zt.verticalHeader().setVisible(False)
        self.zt.verticalHeader().setDefaultSectionSize(ROW_H)
        self.zt.horizontalHeader().setFixedHeight(ROW_H)
        self.zt.setShowGrid(False)
        self.zt.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.zt.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        right.addWidget(self.zt, 1)
        bz = button('Delete selected zone')
        bz.clicked.connect(self._del_zone)
        right.addWidget(bz)
        mid.addLayout(right, 2)
        l.addLayout(mid, 1)
        self.stack.addWidget(w)

    def _page_summary(self):
        w = QWidget()
        l = QVBoxLayout(w)
        l.setContentsMargins(16, 14, 16, 14)
        l.addWidget(label('Summary', bold=True))
        self.sum_t = QTableWidget(0, 2)
        self.sum_t.horizontalHeader().setVisible(False)
        self.sum_t.verticalHeader().setVisible(False)
        self.sum_t.verticalHeader().setDefaultSectionSize(ROW_H + 2)
        self.sum_t.setShowGrid(False)
        self.sum_t.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.sum_t.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.sum_t.setColumnWidth(0, 180)
        l.addWidget(self.sum_t, 1)
        self.keep_cams = QCheckBox('Keep the current cameras (they are re-checked against the new body)')
        self.keep_cams.setChecked(True)
        l.addWidget(self.keep_cams)
        self.stack.addWidget(w)

    # ------------------------------------------------------------------ navigation
    def _max_step(self):
        return 4 if self.lm is not None else 0

    def go(self, i):
        i = max(0, min(i, self._max_step()))
        self.step = i
        self.stack.setCurrentIndex(i)
        self.set_subtitle(f'Step {i + 1} of 5')
        self.steps.viewport().update()
        self.b_back.setEnabled(i > 0)
        self.b_next.setText('Import' if i == 4 else 'Next')
        self.b_next.setEnabled(self.lm is not None)
        self.hint.setText(['Drop a file anywhere on this dialog.', 'Rotations are applied to the mesh, not stored as a transform.',
                           'Pick each wheel on the mesh; the ground plane passes through the four contact points.',
                           'Glass and shut lines are excluded from placement and from the optimiser.',
                           'Import replaces the vehicle; you can undo it from History.'][i])
        if i == 1:
            self._refresh_orient()
            QTimer.singleShot(20, lambda: self._frame(self.vp_orient, 'persp'))
        elif i == 2:
            if not self.wheels:
                self.auto_wheels(quiet=True)
            self._refresh_scale()
            QTimer.singleShot(20, lambda: self._frame(self.vp_side, 'side'))
        elif i == 3:
            self._ensure_zones()
            self._refresh_zones()
            QTimer.singleShot(20, lambda: self._frame(self.vp_zones, 'persp'))
        elif i == 4:
            self._refresh_summary()

    def _next(self):
        if self.step == 4:
            self.finish()
        else:
            self.go(self.step + 1)

    def _frame(self, vp, view):
        m = self.world_mesh()
        if m is None:
            return
        vp.set_view(view)
        if view == 'side':
            vp.cam.az = 90.0
        lo, hi = m.bbox
        vp.frame_box(lo, hi, 1.05 if view == 'side' else 0.9)

    def step_summary(self, i):
        if self.lm is None:
            return ''
        if i == 0:
            return f'{os.path.basename(self.path)} · {self.units}'[:30]
        if i == 1:
            return f'{self.up} up · {self.fwd} forward'
        if i == 2:
            return f'{len(self.wheels)} wheels · {self.origin_mode.replace("_", " ")}'
        if i == 3:
            m = self.world_mesh()
            return f'occluder {self.occ_target:,} tris'.replace(',', ' ')
        return ''

    # ------------------------------------------------------------------ loading
    def _browse(self):
        exts = ' '.join('*' + e for e in loaders.SUPPORTED_EXTS)
        path, _ = QFileDialog.getOpenFileName(self, 'Vehicle model', self.ctx.settings.value('dir/models', os.path.expanduser('~')),
                                              f'3D models ({exts})')
        if path:
            self.load(path)

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        for u in e.mimeData().urls():
            p = u.toLocalFile()
            if os.path.splitext(p)[1].lower() in loaders.SUPPORTED_EXTS:
                self.load(p)
                return

    def load(self, path):
        self.path = path
        self.ctx.settings.setValue('dir/models', os.path.dirname(path))
        self.f_name.setText(os.path.basename(path))
        self.f_meta.setText('Loading…')
        self.err.hide()

        def run():
            try:
                lm = loaders.load_mesh(path)
                self.sig.loaded.emit(lm)
            except loaders.MeshLoadError as e:
                self.sig.failed.emit(str(e))
            except Exception as e:
                self.sig.failed.emit(f'{type(e).__name__}: {e}')
        threading.Thread(target=run, daemon=True).start()

    def _failed(self, msg):
        self.f_meta.setText('')
        self.err_line.set('critical', f'Critical · {msg}')
        while self.err_tbl.count():
            w = self.err_tbl.takeAt(0).widget()
            if w:
                w.deleteLater()
        self.b_mm.hide()
        self.err.show()

    def _loaded(self, lm):
        self.lm = lm
        self.roles = detect_roles(lm.parts)
        self.face_zone = np.full(len(lm.faces), -1, np.int16)
        self.zones = []
        self.wheels, self.wheel_r = {}, {}
        lo, hi = lm.vertices.min(axis=0), lm.vertices.max(axis=0)
        u, readings, sure = guess_units(lo, hi, lm.unit_hint)
        self.units = u
        self.u_combo.setCurrentIndex([x[0] for x in UNITS].index(u))
        self.up = lm.up_hint if lm.up_hint in AXES else self._guess_up(lo, hi)
        self.fwd = lm.forward_hint if lm.forward_hint in AXES else self._guess_fwd(lo, hi)
        if self.fwd[1] == self.up[1]:
            self.fwd = '+X' if self.up[1] != 'X' else '+Y'
        self.up_seg.set_value(self.up)
        self.fw_seg.set_value(self.fwd)
        size = os.path.getsize(self.path) / 1e6 if os.path.exists(self.path) else 0
        self.f_meta.setText(f'{self.path} · {size:.1f} MB · {len(lm.parts)} meshes · {len(lm.faces):,} tris'.replace(',', ' '))
        self.f_note.setText('Drop another file here to replace it.' + (f'  Notes: {"; ".join(lm.notes[:2])}' if lm.notes else ''))
        ext = hi - lo
        f = dict((x[0], x[2]) for x in UNITS)[u]
        dims = ' × '.join(f'{v * f:.3f}' for v in sorted(ext, reverse=True))
        self.u_note.setText(f'{"from file" if lm.unit_hint and sure else "guessed"} · bounding box {dims} m'
                            + (' fits a passenger car' if 3.0 <= max(ext) * f <= 5.6 else ''))
        if not sure:
            self.err_line.set('critical', f'Critical · units could not be determined. {os.path.basename(self.path)} has no unit '
                                          f'metadata; its bounding box is {" × ".join(f"{v:.4g}" for v in ext)}.')
            while self.err_tbl.count():
                w = self.err_tbl.takeAt(0).widget()
                if w:
                    w.deleteLater()
            for i, (uu, name, L) in enumerate(readings):
                self.err_tbl.addWidget(label(f'read as {name.lower()}', ink2=True), i, 0)
                self.err_tbl.addWidget(label(f'{L:.3f} m', mono_=True), i, 1)
                self.err_tbl.addWidget(label('yes' if 2.4 <= L <= 25 else 'no', ink2=True), i, 2)
            self.b_mm.show()
            self.err.show()
        self._fill_parts()
        self._world = None
        self.spec = dict(self.ctx.store.project.vehicle.spec) if self.ctx.store.project.vehicle.spec else {}
        self.go(0)

    def _guess_up(self, lo, hi):
        ext = hi - lo
        # cars: length > width >= height; up is the smallest extent with the body above the wheels
        return '+' + 'XYZ'[int(np.argmin(ext))]

    def _guess_fwd(self, lo, hi):
        return '+' + 'XYZ'[int(np.argmax(hi - lo))]

    def _fill_parts(self):
        lm = self.lm
        counts = np.bincount(lm.face_part, minlength=len(lm.parts))
        self.parts.setRowCount(len(lm.parts))
        for i, p in enumerate(lm.parts):
            self.parts.setItem(i, 0, QTableWidgetItem(p.name or f'Part {i + 1}'))
            it = QTableWidgetItem(f'{counts[i]:,}'.replace(',', ' '))
            it.setFont(mono(12))
            it.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.parts.setItem(i, 1, it)
            m = QTableWidgetItem(p.material or '—')
            m.setFont(mono(11))
            self.parts.setItem(i, 2, m)
            cb = QComboBox()
            for r in ROLES:
                cb.addItem(r + (' (4 found)' if r == ROLE_WHEELS and False else ''), r)
            cb.setCurrentIndex(ROLES.index(self.roles[i]))
            cb.activated.connect(lambda k, i=i, cb=cb: self._role(i, cb.itemData(k)))
            self.parts.setCellWidget(i, 3, cb)
        self.total.setText(f'Total  {len(lm.faces):,} tris'.replace(',', ' '))

    def _role(self, i, r):
        self.roles[i] = r
        self._world = None
        self.zone_version += 1

    def _set_units(self, u):
        self.units = u
        self.u_combo.setCurrentIndex([x[0] for x in UNITS].index(u))
        self.err.hide()
        self._world = None
        self.wheels = {}

    def _set(self, k, v):
        setattr(self, k, v)
        self._world = None
        self._refresh_scale()

    def _orient(self, up=None, fwd=None, lr=None):
        if up:
            self.up = up
            if self.fwd[1] == up[1]:
                self.fwd = '+X' if up[1] != 'X' else '+Y'
        if fwd:
            if fwd[1] == self.up[1]:
                return
            self.fwd = fwd
        if lr is not None:
            self.flip_lr = lr
        self.up_seg.set_value(self.up)
        self.fw_seg.set_value(self.fwd)
        self._world = None
        self.wheels = {}
        self._refresh_orient()

    def orient_text(self):
        return f'{self.up} up · {self.fwd} forward'

    # ------------------------------------------------------------------ geometry
    @property
    def wheels(self):
        if not self.wb:
            return {}
        self.world_mesh()
        off = getattr(self, '_base_off', np.zeros(3))
        return {k: c - off for k, c in self.wb.items()}

    @wheels.setter
    def wheels(self, value):
        # only used to clear, or to load wheels given in a frame whose offset is zero
        self.wb = {k: np.asarray(c, float) for k, c in dict(value).items()}

    def _from_current(self):
        v = self.ctx.store.project.vehicle
        m = v.mesh
        self.lm = loaders.LoadedMesh(m.vertices.copy(), m.faces.copy(), m.face_part.copy(),
                                     [loaders.PartInfo(p.name, p.material, p.alpha, p.color) for p in m.parts], 'vantage', 'm', '+Z', '+X', [])
        self.roles = [p.role for p in m.parts]
        self.zones = [copy.deepcopy(z) for z in m.zones]
        self.face_zone = m.face_zone.copy()
        self.units, self.up, self.fwd = 'm', '+Z', '+X'
        self.path = v.source
        self.origin_mode = 'file'
        self.ground_mode = 'file'
        self.ground_c.setCurrentIndex(2)
        self.origin_c.setCurrentIndex(4)
        self.wheels = {k: np.array(c, float) for k, c in v.wheel_centres.items()}
        self.wheel_r = {k: float(c[2]) for k, c in v.wheel_centres.items()}
        self.spec = dict(v.spec)
        self.occ_target = v.occluder_target
        self.occ.setValue(min(max(self.occ_target, 1000), 60000))
        self.f_name.setText(os.path.basename(v.source) or v.name)
        self.f_meta.setText(f'current vehicle · {m.n_tris:,} tris'.replace(',', ' '))
        self.u_combo.setCurrentIndex(0)
        self._fill_parts()

    def _base_mesh(self):
        """Loaded mesh scaled to metres and rotated to ISO axes (before ground/origin shift)."""
        lm = self.lm
        f = dict((x[0], x[2]) for x in UNITS)[self.units]
        z = axis_vec(self.up)
        x = axis_vec(self.fwd)
        y = np.cross(z, x)
        if self.flip_lr:
            y = -y
        R = np.stack([x, y, z])          # rows: new axes expressed in file coordinates
        parts = [Part(p.name or f'Part {i + 1}', p.material, p.alpha, tuple(p.color), self.roles[i]) for i, p in enumerate(lm.parts)]
        m = Mesh(lm.vertices * f, lm.faces, lm.face_part, parts, self.face_zone, self.zones)
        m = m.transformed(R=R)
        t = m.tri
        vol = float(np.einsum('ij,ij->i', t[:, 0], np.cross(t[:, 1], t[:, 2])).sum() / 6.0)
        self.inside_out = vol < 0
        if self.inside_out:
            # faces wound clockwise (mirrored export): flip so normals point outward
            m = Mesh(m.vertices, m.faces[:, ::-1].copy(), m.face_part, m.parts, m.face_zone, m.zones)
        return m

    def world_mesh(self):
        if self.lm is None:
            return None
        key = (self.units, self.up, self.fwd, self.flip_lr, self.ground_mode, self.origin_mode,
               tuple(sorted((k, tuple(np.round(v, 4))) for k, v in self.wb.items())), tuple(self.roles))
        if self._world is None or self._world_key != key:
            m = self._base_mesh()
            off = self._offset(m)
            m = m.transformed(t=-off)
            m.face_zone = self.face_zone
            m.zones = self.zones
            self._world = m
            self._world_key = key
            self._base_off = off
        return self._world

    def _offset(self, m):
        """Translation that puts the chosen origin at 0 (in base-mesh coordinates)."""
        lo, hi = m.bbox
        wc = self._wheels_base()
        z0 = 0.0
        if self.ground_mode == 'tyre' and len(wc) >= 3:
            z0 = float(np.mean([c[2] - self.wheel_r.get(k, 0.3) for k, c in wc.items()]))
        elif self.ground_mode == 'lowest':
            z0 = float(lo[2])
        x0, y0 = 0.0, 0.0
        if self.origin_mode == 'rear_axle' and 'RL' in wc and 'RR' in wc:
            x0 = (wc['RL'][0] + wc['RR'][0]) / 2
            y0 = (wc['RL'][1] + wc['RR'][1]) / 2
        elif self.origin_mode == 'front_axle' and 'FL' in wc and 'FR' in wc:
            x0 = (wc['FL'][0] + wc['FR'][0]) / 2
            y0 = (wc['FL'][1] + wc['FR'][1]) / 2
        elif self.origin_mode == 'front_bumper':
            x0, y0 = hi[0], (lo[1] + hi[1]) / 2
        elif self.origin_mode == 'centre':
            x0, y0 = (lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2
        elif self.origin_mode == 'rear_axle':
            x0, y0 = lo[0] + 0.2 * (hi[0] - lo[0]), (lo[1] + hi[1]) / 2
        if self.origin_mode == 'file' and self.ground_mode == 'file':
            return np.zeros(3)
        if self.origin_mode == 'file':
            x0 = y0 = 0.0
        return np.array([x0, y0, z0])

    def _wheels_base(self):
        return dict(self.wb)

    def origin_world(self):
        return np.zeros(3)

    def auto_wheels(self, quiet=False):
        m = self.world_mesh()
        if m is None:
            return
        wi = [i for i, r in enumerate(self.roles) if r == ROLE_WHEELS]
        cen = None
        if wi:
            sel = np.isin(m.face_part, wi)
            pts = m.tri[sel].reshape(-1, 3)
            if len(pts) > 20:
                cen = pts
        if cen is None:
            # no wheel parts: take the lowest 12 % of the body, split by quadrant
            v = m.vertices[np.unique(m.faces)]
            lo, hi = v.min(axis=0), v.max(axis=0)
            cen = v[v[:, 2] < lo[2] + 0.12 * (hi[2] - lo[2]) + 0.25]
        lo, hi = m.bbox
        cx = np.median(cen[:, 0])
        cy = (lo[1] + hi[1]) / 2
        found = {}
        for k, sx, sy in (('RL', -1, 1), ('RR', -1, -1), ('FL', 1, 1), ('FR', 1, -1)):
            q = cen[(np.sign(cen[:, 0] - cx) == sx) & (np.sign(cen[:, 1] - cy) == sy)]
            if len(q) < 6:
                continue
            qlo, qhi = q.min(axis=0), q.max(axis=0)
            r = (qhi[2] - qlo[2]) / 2
            if not (0.12 < r < 0.8):
                continue
            yo = qhi[1] if sy > 0 else qlo[1]
            found[k] = np.array([(qlo[0] + qhi[0]) / 2, yo, (qlo[2] + qhi[2]) / 2])
            self.wheel_r[k] = r
        off = getattr(self, '_base_off', np.zeros(3))
        if len(found) == 4:
            self.wb = {k: c + off for k, c in found.items()}
            self._world = None
            if not quiet:
                self.ctx.msg('Found 4 wheels.')
        elif not quiet:
            QMessageBox.information(self, 'Wheels', f'Found {len(found)} of 4 wheels automatically. Pick the others on the mesh.')
            self.wb.update({k: c + off for k, c in found.items()})
            self._world = None
        self._refresh_scale()

    def start_pick(self, k):
        self.picking = k
        self.go(2)
        self.vp_side.update()

    def pick_wheel(self, hit):
        """Wheel centre from vertices near the click on the same side."""
        m = self.world_mesh()
        k = self.picking
        v = m.vertices[np.unique(m.faces)]
        near = v[(np.abs(v[:, 0] - hit[0]) < 0.42) & (np.abs(v[:, 2] - hit[2]) < 0.42) & (np.sign(v[:, 1]) == np.sign(hit[1] or 1))]
        if len(near) < 6:
            self.ctx.msg('Click on the tyre.')
            return
        lo, hi = near.min(axis=0), near.max(axis=0)
        r = (hi[2] - lo[2]) / 2
        c = np.array([(lo[0] + hi[0]) / 2, hi[1] if hit[1] > 0 else lo[1], (lo[2] + hi[2]) / 2])
        self.wb[k] = c + getattr(self, '_base_off', np.zeros(3))
        self.wheel_r[k] = r
        order = ['RL', 'RR', 'FL', 'FR']
        rest = [x for x in order if x not in self.wb]
        self.picking = rest[0] if rest else None
        self._world = None
        self._refresh_scale()

    def measured(self):
        m = self.world_mesh()
        if m is None:
            return {}
        keep = np.array([p.role != ROLE_IGNORED for p in m.parts])[m.face_part]
        mirror = np.array([is_mirror(p.name) for p in m.parts])[m.face_part]
        tri = m.tri[keep] if keep.any() else m.tri
        mir = mirror[keep] if keep.any() else mirror
        v = tri[~mir].reshape(-1, 3) if (~mir).any() else tri.reshape(-1, 3)
        lo, hi = v.min(axis=0), v.max(axis=0)
        w = body_width_points(tri, mir)                 # as datasheets give it: without the mirrors
        out = {'Length': hi[0] - lo[0], 'Width': float(np.ptp(w[:, 1])), 'Height': hi[2] - max(lo[2], 0)}
        wc = self.wheels
        if 'FL' in wc and 'RL' in wc:
            out['Wheelbase'] = abs(wc['FL'][0] - wc['RL'][0])
            nonw = np.array([p.role not in (ROLE_WHEELS, ROLE_IGNORED) for p in m.parts])[m.face_part]
            c = m.centroids[nonw]
            mid = (c[:, 0] > min(wc['RL'][0], wc['FL'][0]) + 0.45) & (c[:, 0] < max(wc['RL'][0], wc['FL'][0]) - 0.45)
            if mid.any():
                out['Ground clearance'] = float(m.tri[nonw][mid][:, :, 2].min())
        return out

    # ------------------------------------------------------------------ refresh
    def _refresh_orient(self):
        m = self.world_mesh()
        if m is None:
            return
        lo, hi = m.bbox
        ext = hi - lo
        self.bbox_lbl.setText(f'Bounding box {ext[0]:.3f} × {ext[1]:.3f} × {ext[2]:.3f} m')
        longest = 'XYZ'[int(np.argmax(ext))]
        self.axis_lbl.setText(f'Longest axis {longest}')
        if longest == 'X' and ext[2] < ext[0]:
            self.axis_ok.set('good', 'Good · longest axis is forward')
        else:
            self.axis_ok.set('warning', 'Warning · the longest axis is not forward; check up and forward')
        self.vp_orient.update()

    def _refresh_scale(self):
        if not hasattr(self, 'wheel_rows'):
            return
        o = np.zeros(3)
        for k, (v, st, b) in self.wheel_rows.items():
            c = self.wheels.get(k)
            if c is not None:
                v.setText(f'{c[0]:.3f}  {c[1]:.3f}  {c[2]:.3f}')
                st.setPixmap(icons.pixmap('good', STATUS['good'], 12))
            else:
                v.setText('—')
                st.setPixmap(icons.pixmap('warning', STATUS['warning'], 12) if self.lm is not None else icons.pixmap('good', '#00000000', 12))
            b.setText('Picking…' if self.picking == k else 'Pick')
        self.ground_c.setCurrentIndex(max(0, self.ground_c.findData(self.ground_mode)))
        self.origin_c.setCurrentIndex(max(0, self.origin_c.findData(self.origin_mode)))
        meas = self.measured()
        rows = ['Length', 'Width', 'Height', 'Wheelbase', 'Ground clearance']
        self.spec_t.blockSignals(True)
        for i, k in enumerate(rows):
            mv = meas.get(k)
            sv = self.spec.get(k)
            a = QTableWidgetItem(k)
            a.setFlags(a.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.spec_t.setItem(i, 0, a)
            for j, val in ((1, f'{mv:.3f}' if mv is not None else '—'), (2, f'{sv:.3f}' if sv else '')):
                it = QTableWidgetItem(val)
                it.setFont(mono(12))
                it.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                if j == 1:
                    it.setFlags(it.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.spec_t.setItem(i, j, it)
            if mv is not None and sv:
                d = (mv - sv) * 1000
                dd = QTableWidgetItem(f'{d:+.0f}'.replace('-', '−'))
                dd.setFont(mono(12))
                dd.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                ok = abs(d) <= 5
                s = QTableWidgetItem('Within tolerance' if ok else ('Warning · check ride height' if k == 'Ground clearance' else 'Warning · check scale'))
                s.setIcon(icons.status_icon('good' if ok else 'warning', 12))
            else:
                dd = QTableWidgetItem('')
                s = QTableWidgetItem('enter a spec value' if mv is not None else '')
                s.setForeground(THEME.color('muted'))
            for it in (dd, s):
                it.setFlags(it.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.spec_t.setItem(i, 3, dd)
            self.spec_t.setItem(i, 4, s)
        self.spec_t.blockSignals(False)
        self.vp_side.update()
        self.steps.viewport().update()

    def _spec_edit(self, it):
        if it.column() != 2:
            return
        k = self.spec_t.item(it.row(), 0).text()
        try:
            from ...core.mathutil import parse_number
            v = parse_number(it.text(), 'length') if it.text().strip() else 0
        except Exception:
            v = 0
        if v > 0:
            self.spec[k] = v
        else:
            self.spec.pop(k, None)
        QTimer.singleShot(0, self._refresh_scale)

    def brush_mm(self):
        return self.brush.value() if hasattr(self, 'brush') else 40.0

    def _ensure_zones(self):
        if self.face_zone is None or len(self.face_zone) != len(self.lm.faces):
            self.face_zone = np.full(len(self.lm.faces), -1, np.int16)
        if self.glass_cb.isChecked() and not any(z.auto for z in self.zones):
            self._glass(True)

    def _glass(self, on):
        if self.lm is None:
            return
        gi = [i for i, r in enumerate(self.roles) if r == ROLE_GLASS]
        zi = next((i for i, z in enumerate(self.zones) if z.auto), None)
        if on:
            if zi is None:
                self.zones.append(Zone('Glass (auto)', 'forbidden', auto=True))
                zi = len(self.zones) - 1
            sel = np.isin(self.lm.face_part, gi)
            self.face_zone = np.where(sel, zi, self.face_zone).astype(np.int16)
            n = int(sel.sum())
            self.zones[zi].name = f'Glass ({len(gi)} part{"s" if len(gi) != 1 else ""}, auto)'
            self.glass_note.setText(f'{len(gi)} glass part{"s" if len(gi) != 1 else ""} found by material and transparency · {n:,} faces'.replace(',', ' '))
        elif zi is not None:
            self._drop_zone(zi)
        self.zone_version += 1
        self._world = None
        self._refresh_zones()

    def _paint_mode(self, k):
        self.paint_kind = None if k == 'none' else k

    def paint_at(self, hit, n):
        m = self.world_mesh()
        name = self.zone_name.text().strip() or 'Zone'
        zi = next((i for i, z in enumerate(self.zones) if z.name == name and z.kind == self.paint_kind), None)
        if zi is None:
            self.zones.append(Zone(name, self.paint_kind))
            zi = len(self.zones) - 1
        r = self.brush_mm() / 1000
        d = np.linalg.norm(m.centroids - hit, axis=1)
        sel = (d < r) & (m.normals @ n > 0.3)
        self.face_zone[sel] = zi
        m.face_zone = self.face_zone
        self.zone_version += 1
        self._refresh_zones()

    def _drop_zone(self, zi):
        fz = self.face_zone
        self.face_zone = np.where(fz == zi, -1, np.where(fz > zi, fz - 1, fz)).astype(np.int16)
        self.zones.pop(zi)

    def _del_zone(self):
        r = self.zt.currentRow()
        if 0 <= r < len(self.zones):
            self._drop_zone(r)
            self.zone_version += 1
            self._world = None
            self._refresh_zones()

    def _occ(self):
        self.occ_target = int(round(self.occ.value() / 500) * 500)
        self._refresh_zones()

    def _refresh_zones(self):
        m = self.world_mesh()
        if m is None:
            return
        n = m.n_tris
        t = min(self.occ_target, n)
        dev = 0.0
        if n > self.occ_target and n < 400_000:
            d, dev = m.decimate(self.occ_target)
            t = d.n_tris
        self.occ_lbl.setText(f'{t:,} tris · {100 * t / max(n, 1):.1f} % · {dev * 1000:.1f} mm max deviation'.replace(',', ' '))
        self.zt.setRowCount(len(self.zones))
        for i, z in enumerate(self.zones):
            k = QTableWidgetItem(z.kind)
            if z.kind == 'forbidden':
                k.setForeground(QColor(STATUS['critical']))
            self.zt.setItem(i, 0, k)
            self.zt.setItem(i, 1, QTableWidgetItem(z.name))
            c = QTableWidgetItem(str(int((self.face_zone == i).sum())))
            c.setFont(mono(12))
            c.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.zt.setItem(i, 2, c)
        self.sc_zones._key = None
        self.vp_zones.update()
        self.steps.viewport().update()

    def _refresh_summary(self):
        m = self.world_mesh()
        meas = self.measured()
        rows = [('File', os.path.basename(self.path) or '—'), ('Units', dict((x[0], x[1]) for x in UNITS)[self.units]),
                ('Normals', 'flipped: the mesh was inside-out' if getattr(self, 'inside_out', False) else 'outward'),
                ('Orientation', self.orient_text() + (' · mirrored left/right' if self.flip_lr else '')),
                ('Triangles', f'{m.n_tris:,}'.replace(',', ' ')),
                ('Parts', ', '.join(f'{p.name} ({p.role})' for p in m.parts[:6]) + (' …' if len(m.parts) > 6 else '')),
                ('Wheels', ', '.join(f'{k} {c[0]:.3f}/{c[1]:.3f}/{c[2]:.3f}' for k, c in sorted(self.wheels.items())) or 'none'),
                ('Origin', self.origin_c.currentText() + ' · ground ' + self.ground_c.currentText().lower()),
                ('Dimensions', ' · '.join(f'{k} {v:.3f}' for k, v in meas.items())),
                ('Occluder', f'{self.occ_target:,} tris target'.replace(',', ' ')),
                ('Mount zones', ', '.join(f'{z.name} ({z.kind})' for z in self.zones) or 'none')]
        self.sum_t.setRowCount(len(rows))
        for i, (a, b) in enumerate(rows):
            x = QTableWidgetItem(a)
            x.setForeground(THEME.color('ink2'))
            self.sum_t.setItem(i, 0, x)
            self.sum_t.setItem(i, 1, QTableWidgetItem(b))

    # ------------------------------------------------------------------ finish
    def finish(self):
        m = self.world_mesh()
        if m is None:
            return
        mesh = m.copy()
        mesh.face_zone = self.face_zone.copy()
        mesh.zones = [copy.deepcopy(z) for z in self.zones]
        name = os.path.splitext(os.path.basename(self.path))[0] if self.path and not self.reuse else self.ctx.store.project.vehicle.name
        wc = {k: [round(float(x), 4) for x in c] for k, c in self.wheels.items()}
        old = self.ctx.store.project.vehicle
        landmarks = {}
        if self.reuse and (self.units, self.up, self.fwd, self.flip_lr) == ('m', '+Z', '+X', False):
            # the current body in its own axes: keep its other landmarks (moved with a new origin, if one was chosen)
            off = getattr(self, '_base_off', np.zeros(3))
            landmarks = {k: tuple(float(x) - float(o) for x, o in zip(v, off)) for k, v in old.landmarks.items()
                         if not k.startswith('Wheel centre ')}
        landmarks.update({f'Wheel centre {k}': tuple(c) for k, c in wc.items()})
        veh = Vehicle(name=name if not self.reuse else old.name, spec_name=old.spec_name if self.reuse else '',
                      source=self.path or old.source, template='' if not self.reuse else old.template,
                      import_units=self.units, import_up=self.up, import_forward=self.fwd, occluder_target=self.occ_target,
                      wheel_centres=wc, spec=dict(self.spec), mesh=mesh, imported_tris=mesh.n_tris,
                      landmarks=landmarks)
        keep = self.keep_cams.isChecked()

        def fn(p):
            p.vehicle = veh
            if not keep:
                p.variant.cameras = []
            if p.photos.landmarks:
                for k, c in wc.items():
                    p.photos.landmarks[f'Wheel centre {k}'] = list(c)
        self.ctx.store.edit(['vehicle', 'variant', 'photos'], fn,
                            f'Imported {os.path.basename(self.path) or name} · {mesh.n_tris:,} tris · units {self.units} · up {self.up} · forward {self.fwd}'.replace(',', ' '))
        self.ctx.store.project.add_log('import', f'{os.path.basename(self.path) or name} · {mesh.n_tris} tris · units {self.units} · up {self.up} · forward {self.fwd}')
        self.ctx.analysis.request(force=True)
        self.ctx.window.viewarea.frame_all()
        self.accept()
