"""Simulate ▸ Rig perception: 3D (bird's-eye) detection of each class in each region around the car, for each rig, on
the same scenes, with intervals; rig against rig, paired. The page picks the recordings (or drives a perception scene
set in CARLA for the chosen variants), the detector backend (the lifted YOLOX, or a team's own command on this
computer), runs the study as a background job with Cancel, and shows polar region maps per class (the reference rig,
the compared rig and their difference, hatched where the difference is not significant), the region table, the
comparisons, the findings, what each scene set yields per region, and every instant's detections over its ground truth
from above. Nothing runs by itself after an edit (the result is marked stale)."""
from __future__ import annotations

import json
import math
import os
import threading
import traceback

from PyQt6.QtCore import QObject, QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QPainter, QPen, QPolygonF
from PyQt6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QFileDialog, QGridLayout, QHBoxLayout,
                             QHeaderView, QInputDialog, QLineEdit, QListWidget, QListWidgetItem, QPlainTextEdit,
                             QProgressBar, QScrollArea, QSlider, QSplitter, QTableWidget, QTableWidgetItem,
                             QTabWidget, QToolTip, QVBoxLayout, QWidget)

from ...core import bev_backends as BB
from ...core import detection as DT
from ...core import perception_scenes as PS
from ...core import rig_perception as RP
from .. import colormaps as CM
from ..theme import STATUS, THEME, mono, sans
from ..widgets import fit_width, ROW_H, ScrubSpinBox, SegmentedControl, StatusLine, button, hline, label
from .recording_ws import Tile

JOB = 'rig_perception'
JOB_DRIVE = 'rig_perception_drive'
JOB_YIELD = 'rig_perception_yield'
EMPTY = ('Add CARLA drives with ground truth of two rig variants on the same scenes (or drive a scene set below), '
         'then Run. The first rig is the reference.')
FIELDS = (
    ('score', 'Score threshold', '', 2, 0.05, 'The operating point: recall, precision and the centre error count '
                                              'detections scoring at least this (AP uses every detection)'),
    ('match_m', 'Match within', 'm', 1, 0.5, 'A detection matches an object when their centres are this close in '
                                             'the bird\'s-eye view (nuScenes\' true-positive threshold is 2 m)'),
    ('frame_step', 'Every n-th instant', '', 0, 1.0, 'Evaluate every n-th instant (quicker, fewer objects)'),
    ('max_range_m', 'Range', 'm', 0, 5.0, 'Evaluate objects up to this far from the vehicle (0: the recordings\' '
                                          'annotation range)'),
    ('pitch_sigma_deg', 'Pitch σ', '°', 2, 0.1, 'The lifted detector\'s pitch uncertainty (the body on its springs '
                                                'and the calibration): its ground ranges\' σ grows as r² · σ / h'),
    ('bootstrap', 'Bootstrap', '', 0, 100.0, 'Resamples for the intervals (0: none)'),
    ('block_s', 'Block', 's', 2, 0.25, 'The bootstrap draws blocks of this many seconds, within each scene'),
    ('confidence', 'Confidence', '', 3, 0.01, 'The intervals\' confidence level'),
    ('min_gt', 'Fewest objects', '', 0, 1.0, 'A region with fewer objects is shown hatched and never a finding'),
)
REGION_COLS = (('Rig', ''), ('Class', ''), ('Region', ''), ('Objects', 'Annotated objects (object-instants), '
                                                                      'whether a camera sees them or not'),
               ('Visible', 'Share some camera of the rig shows (not an ignore region)'),
               ('Recall', 'At the operating score, within the match distance, with its interval'),
               ('Precision', ''), ('mAP', 'nuScenes AP averaged over 0.5, 1, 2, 4 m'), ('ATE', 'Mean centre '
                                                                                         'error of the matches (m)'))
CMP_COLS = (('Class', ''), ('Region', ''), ('Objects', 'Of the compared rig (the reference\'s in brackets)'),
            ('Recall', 'Reference → compared'), ('Δ recall', 'Compared − reference, with its interval'),
            ('mAP', 'Reference → compared'), ('Δ mAP', ''), ('Paired', ''))
YIELD_COLS = (('Class', ''), ('Region', ''), ('Object-instants', 'Annotated objects at the evaluated instants'),
              ('Objects', 'Distinct objects'))
TABS = ('Maps', 'Regions', 'Compare', 'Findings', 'Yield', 'Frames', 'Notes')


def _table(cols):
    t = QTableWidget(0, len(cols))
    for c, (h, tip) in enumerate(cols):
        it = QTableWidgetItem(h)
        if tip:
            it.setToolTip(tip)
        t.setHorizontalHeaderItem(c, it)
    t.verticalHeader().setVisible(False)
    t.verticalHeader().setDefaultSectionSize(ROW_H)
    t.horizontalHeader().setFixedHeight(ROW_H)
    t.setShowGrid(False)
    t.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    t.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    hh = t.horizontalHeader()
    for c in range(len(cols)):
        hh.setSectionResizeMode(c, QHeaderView.ResizeMode.ResizeToContents)
    hh.setStretchLastSection(True)
    return t


def _cell(t, r, c, text, num=False, tip=None, color=None):
    it = QTableWidgetItem(text)
    if num:
        it.setFont(mono(12))
        it.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    if tip:
        it.setToolTip(tip)
    if color is not None:
        it.setForeground(color)
    t.setItem(r, c, it)


def _pc(v, ci=None) -> str:
    if v is None:
        return '—'
    s = f'{v * 100:.1f} %'
    if ci:
        s += f'  [{ci[0] * 100:.1f}–{ci[1] * 100:.1f}]'
    return s


def _dpc(v, ci=None) -> str:
    if v is None:
        return '—'
    s = f'{v * 100:+.1f} pt'
    if ci:
        s += f'  [{ci[0] * 100:+.1f}, {ci[1] * 100:+.1f}]'
    return s


def _qc(rgb) -> QColor:
    return QColor(int(rgb[0]), int(rgb[1]), int(rgb[2]))


# ============================================================================ the polar region map
class PolarMap(QWidget):
    """Regions as sector × range-ring cells around the car (front up, left on the left, rings of equal width per range
    bin), coloured by a rig's figure (sequential blue) or by a difference (diverging orange–blue); grey for regions
    without objects; hatched where a region has too few objects or a difference is not significant. Hover: the
    region, its value, interval and objects."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(220, 250)
        self.setMouseTracking(True)
        self.grid = None
        self.values = {}                # rid -> (value, solid, n, ci)
        self.kind = 'recall'
        self.title = ''
        self.empty = 'Run to see the regions.'
        self._cells = []
        THEME.changed.connect(self._theme_changed)

    def _theme_changed(self, *_):
        self.update()

    def set_data(self, grid, values, kind, title):
        self.grid, self.values, self.kind, self.title = grid, dict(values), kind, title
        self.update()

    def _geom(self):
        top = 22.0
        s = min(self.width(), self.height() - top - 26)
        return self.width() / 2.0, top + s / 2.0, max(s / 2.0 - 6, 10)

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), THEME.color('viewport'))
        p.setFont(sans(11))
        p.setPen(THEME.color('ink2'))
        p.drawText(QRectF(6, 2, self.width() - 12, 18), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                   self.title)
        if self.grid is None:
            p.setPen(THEME.color('muted'))
            p.drawText(QRectF(self.rect()), Qt.AlignmentFlag.AlignCenter, self.empty)
            return
        theme = 'light' if THEME.name == 'light' else 'dark'
        seq = CM.ramp(CM.DENSITY[theme], 256)
        div = CM.diverging(theme)
        grey = QColor(CM.BELOW_MIN[theme])
        hatch = QBrush(QColor(CM.HATCH[theme]), Qt.BrushStyle.BDiagPattern)
        cx, cy, R = self._geom()
        self._cells = []
        for s, r, rid, pts in RP.polar_cells(self.grid):
            poly = QPolygonF([QPointF(cx + x * R, cy - y * R) for x, y in pts])
            v = self.values.get(rid)
            if v is None or v[0] is None:
                col = grey
            elif self.kind == 'difference':
                t = 0.5 + 0.5 * max(-1.0, min(1.0, v[0] / 0.5))
                col = _qc(div[int(round(t * (len(div) - 1)))])
            else:
                col = _qc(seq[int(round(max(0.0, min(1.0, v[0])) * 255))])
            p.setPen(QPen(THEME.color('viewport'), 1))
            p.setBrush(col)
            p.drawPolygon(poly)
            if v is not None and v[0] is not None and not v[1]:
                p.setBrush(hatch)
                p.setPen(Qt.PenStyle.NoPen)
                p.drawPolygon(poly)
            self._cells.append((poly, rid))
        # the car and the front
        w, h = R * 0.07, R * 0.14
        p.setPen(QPen(THEME.color('ink2'), 1))
        p.setBrush(THEME.color('body'))
        p.drawRoundedRect(QRectF(cx - w, cy - h, 2 * w, 2 * h), 2, 2)
        p.drawLine(QPointF(cx, cy - h), QPointF(cx, cy - h - 5))
        p.setFont(sans(9))
        p.setPen(THEME.color('muted'))
        names = self.grid.range_names()
        p.drawText(QRectF(4, self.height() - 22, self.width() - 8, 20), Qt.AlignmentFlag.AlignLeft |
                   Qt.AlignmentFlag.AlignVCenter, 'rings: ' + ' · '.join(names) + ' (not to scale)')

    def mouseMoveEvent(self, e):
        pos = e.position()
        for poly, rid in self._cells:
            if poly.containsPoint(pos, Qt.FillRule.OddEvenFill):
                s, r = self.grid.parse_id(rid)
                v = self.values.get(rid)
                if v is None or v[0] is None:
                    txt = f'{self.grid.region_name(s, r)}: no objects'
                else:
                    val = _dpc(v[0], v[3]) if self.kind == 'difference' else _pc(v[0], v[3])
                    note = '' if v[1] else (' · not significant' if self.kind == 'difference' else ' · thin')
                    txt = f'{self.grid.region_name(s, r)}: {val} · {v[2]} objects{note}'
                QToolTip.showText(e.globalPosition().toPoint(), txt, self)
                return
        QToolTip.hideText()


class BevFrame(QWidget):
    """One instant from above, to scale: the ground truth (rings: found filled, missed hollow, ignore regions grey)
    and the detections at the operating point (crosses: matched blue, false orange), the region rings."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(300, 260)
        self.frame = None
        self.grid = None
        self.score = 0.3
        self.half_m = 60.0
        self.caption = ''
        THEME.changed.connect(self._theme_changed)

    def _theme_changed(self, *_):
        self.update()

    def set_frame(self, frame, grid, score, half_m, caption=''):
        self.frame, self.grid, self.score, self.half_m, self.caption = frame, grid, score, half_m, caption
        self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), THEME.color('viewport'))
        p.setFont(sans(11))
        p.setPen(THEME.color('ink2'))
        p.drawText(QRectF(6, 2, self.width() - 12, 18), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                   self.caption)
        if self.frame is None or self.grid is None:
            p.setPen(THEME.color('muted'))
            p.drawText(QRectF(self.rect()), Qt.AlignmentFlag.AlignCenter, 'Run to see each instant from above.')
            return
        top = 22
        S = min(self.width(), self.height() - top)
        cx, cy = self.width() / 2.0, top + S / 2.0
        k = (S / 2.0 - 4) / self.half_m

        def X(x, y):
            return QPointF(cx - (y - self.grid.origin_xy[1]) * k, cy - (x - self.grid.origin_xy[0]) * k)
        p.setPen(QPen(THEME.color('hair'), 1))
        p.setBrush(Qt.BrushStyle.NoBrush)
        o = X(*self.grid.origin_xy)
        for edge in self.grid.range_edges:
            if edge <= self.half_m:
                p.drawEllipse(o, edge * k, edge * k)
        w = 360.0 / self.grid.sectors
        for s in range(self.grid.sectors):
            b = math.radians(s * w - w / 2)
            p.drawLine(o, QPointF(o.x() - math.sin(b) * self.half_m * k, o.y() - math.cos(b) * self.half_m * k))
        p.setBrush(THEME.color('body'))
        p.drawRect(QRectF(X(4.0, 0.9), X(-0.8, -0.9)).normalized())
        blue, orange = QColor('#3987e5'), QColor('#d45e2f')
        for x, y, e_, vis, sc, ign in self.frame['gt']:
            q = X(x, y)
            if ign:
                p.setPen(QPen(THEME.color('muted'), 1))
                p.setBrush(Qt.BrushStyle.NoBrush)
            elif sc >= self.score:
                p.setPen(QPen(blue, 1))
                p.setBrush(blue)
            else:
                p.setPen(QPen(orange, 1.5))
                p.setBrush(Qt.BrushStyle.NoBrush)
            r = 4.0 if vis != RP.VIS_INVISIBLE else 3.0
            p.drawEllipse(q, r, r)
        for x, y, e_, sc, stt, how, src in self.frame['det']:
            if sc < self.score:
                continue
            q = X(x, y)
            p.setPen(QPen(blue if stt == 1 else (THEME.color('muted') if stt < 0 else orange), 1.5))
            p.drawLine(QPointF(q.x() - 4, q.y() - 4), QPointF(q.x() + 4, q.y() + 4))
            p.drawLine(QPointF(q.x() - 4, q.y() + 4), QPointF(q.x() + 4, q.y() - 4))


class _Sig(QObject):
    done = pyqtSignal(int, object)
    failed = pyqtSignal(int, str)
    progress = pyqtSignal(int, float, str)


# ============================================================================ the page
class RigPerceptionPage(QWidget):
    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        self.result: RP.RigPerceptionResult | None = None
        self.detector = None                       # a 2D detector callable (tests); None: the chosen model
        self.backend = None                        # a Backend instance (tests); None: from the settings
        self.runner = None                         # a bridge runner for the scene-set drives (tests)
        self.yield_doc = None
        self.busy = False
        self.drive_busy = False
        self.status_text = ''
        self._gen = 0
        self._for = None
        self._dirty = True
        self._syncing = False
        self._info = {}
        self.sig = _Sig()
        self.sig.done.connect(self._thread_done)
        self.sig.failed.connect(self._failed)
        self.sig.progress.connect(self._progress)
        self.setObjectName('base')
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        split = QSplitter(Qt.Orientation.Horizontal)
        split.setHandleWidth(2)
        split.addWidget(self._left())
        split.addWidget(self._right())
        split.setSizes([380, 1000])
        lay.addWidget(split)
        ctx.store.changed.connect(self._on_change)
        ctx.store.project_replaced.connect(self._on_replaced)
        THEME.changed.connect(self._retheme)
        self._sync_fields()
        self.refresh()

    # ------------------------------------------------------------------ left
    def _left(self):
        w = QWidget()
        w.setObjectName('panel')
        ll = QVBoxLayout(w)
        ll.setContentsMargins(8, 6, 8, 8)
        ll.setSpacing(4)
        top = QHBoxLayout()
        top.addWidget(label('Rig perception'))
        top.addStretch(1)
        top.addWidget(label('3D detection · regions · rigs', muted=True, small=True))
        ll.addLayout(top)
        ll.addWidget(hline())
        ll.addWidget(label('How well a 3D detector finds each class in each region around the car, per rig, on the '
                           'same CARLA scenes, and rig against rig with paired intervals.', muted=True, small=True,
                           wrap=True))
        # the detector
        ll.addWidget(label('Detector', ink2=True))
        self.backend_pick = SegmentedControl([('lifted', 'Lifted YOLOX'), ('external', 'External command')],
                                             'lifted')
        self.backend_pick.setToolTip('Lifted YOLOX: Vantage\'s own monocular lift of YOLOX boxes, fused across '
                                     'cameras. External: your team\'s detector, run as a command on this computer')
        self.backend_pick.changed.connect(self._backend_picked)
        ll.addWidget(self.backend_pick, 0, Qt.AlignmentFlag.AlignLeft)
        self.model_status = StatusLine('warning', '')
        ll.addWidget(self.model_status)
        self.ext_box = QWidget()
        el = QVBoxLayout(self.ext_box)
        el.setContentsMargins(0, 0, 0, 0)
        el.setSpacing(3)
        row = QHBoxLayout()
        self.ext_pick = QComboBox()
        self.ext_pick.setToolTip('External detectors set on this computer (never stored in the project)')
        self.ext_pick.currentIndexChanged.connect(self._ext_picked)
        row.addWidget(self.ext_pick, 1)
        b = button('New…', 'ghost', 'plus', 'Name a new external detector')
        b.clicked.connect(lambda: self.new_external())
        row.addWidget(b)
        el.addLayout(row)
        self.ext_argv = QPlainTextEdit()
        self.ext_argv.setFont(mono(11))
        self.ext_argv.setMaximumHeight(80)
        self.ext_argv.setPlaceholderText('One argument per line, e.g.\n/usr/bin/python3\n/path/to/detect.py\n'
                                         '--dataroot\n{dataroot}\n--out\n{out}')
        self.ext_argv.setToolTip('The command as an argument list (no shell). Placeholders: {dataroot} the '
                                 'recording\'s nuScenes folder, {version} its tables\' folder, {frames} the frames '
                                 'file Vantage writes, {out} where to write the nuScenes submission JSON')
        el.addWidget(self.ext_argv)
        row = QHBoxLayout()
        b = button('Save', 'ghost', None, 'Keep this argument list for the detector (it then needs approval)')
        b.clicked.connect(lambda: self.save_external())
        row.addWidget(b)
        self.b_approve = button('Approve', None, 'check', 'Allow exactly this argument list to run')
        self.b_approve.clicked.connect(lambda: self.approve_external())
        row.addWidget(self.b_approve)
        row.addStretch(1)
        el.addLayout(row)
        self.ext_status = StatusLine('warning', '')
        el.addWidget(self.ext_status)
        ll.addWidget(self.ext_box)
        ll.addWidget(label(BB.TORCH_NOTE, muted=True, small=True, wrap=True))
        ll.addWidget(hline())
        # recordings
        ll.addWidget(label('Recordings', ink2=True))
        self.recs = QListWidget()
        self.recs.setMinimumHeight(90)
        self.recs.setMaximumHeight(140)
        self.recs.setToolTip('CARLA drives with ground truth (or nuScenes folders with annotations), each with its '
                             'rig; the first rig is the reference')
        ll.addWidget(self.recs)
        rb = QHBoxLayout()
        b = button('Add…', None, 'folder', 'Add a recording folder')
        b.clicked.connect(lambda: self._pick_dir())
        rb.addWidget(b)
        b = button('Rig…', 'ghost', 'pin', 'Name the rig the selected recording was made with (recordings of one '
                                          'rig are pooled)')
        b.clicked.connect(lambda: self.rename_rig())
        rb.addWidget(b)
        b = button('Reference', 'ghost', None, 'Move the selected recording\'s rig to the front (the reference)')
        b.clicked.connect(lambda: self.make_reference())
        rb.addWidget(b)
        b = button('Remove', 'ghost', 'trash', 'Remove the selected recording from the list (nothing is deleted)')
        b.clicked.connect(lambda: self.remove_recording())
        rb.addWidget(b)
        rb.addStretch(1)
        ll.addLayout(rb)
        ll.addWidget(hline())
        # the scene set
        ll.addWidget(label('Scene set (CARLA)', ink2=True))
        self.set_pick = QComboBox()
        for k, v in PS.SCENE_SETS.items():
            self.set_pick.addItem(v['label'], k)
            self.set_pick.setItemData(self.set_pick.count() - 1, v['about'], Qt.ItemDataRole.ToolTipRole)
        self.set_pick.currentIndexChanged.connect(self._set_picked)
        ll.addWidget(self.set_pick)
        self.var_list = QListWidget()
        self.var_list.setMaximumHeight(96)
        self.var_list.setToolTip('The variants to drive the set with (the same seeds for each)')
        self.var_list.itemChanged.connect(self._var_toggled)
        ll.addWidget(self.var_list)
        g = QGridLayout()
        a = label('Seeds', ink2=True)
        g.addWidget(a, 0, 0)
        self.seeds = QLineEdit()
        self.seeds.setFont(mono(12))
        self.seeds.setToolTip('Seeds, comma separated: each a repeat of the whole set (the traffic follows it)')
        self.seeds.editingFinished.connect(self._seeds_edited)
        g.addWidget(self.seeds, 0, 1)
        ll.addLayout(g)
        self.set_text = label('', muted=True, small=True, wrap=True)
        ll.addWidget(self.set_text)
        sb = QHBoxLayout()
        self.b_drive = button('Drive', None, 'play', 'Drive the set in CARLA for the chosen variants (a server must '
                                                     'answer on the project\'s CARLA port; drives already made are '
                                                     'reused)')
        self.b_drive.clicked.connect(lambda: self.drive())
        sb.addWidget(self.b_drive)
        b = button('Use drives', 'ghost', 'import', 'Put the set\'s finished drives in the recordings list')
        b.clicked.connect(lambda: self.use_drives())
        sb.addWidget(b)
        b = button('Yield', 'ghost', 'layers', 'Count the ground-truth objects the recordings yield per class and '
                                              'region (no detector: quick)')
        b.clicked.connect(lambda: self.compute_yield())
        sb.addWidget(b)
        sb.addStretch(1)
        ll.addLayout(sb)
        ll.addWidget(hline())
        # settings
        ll.addWidget(label('Matching and statistics', ink2=True))
        g = QGridLayout()
        g.setHorizontalSpacing(6)
        g.setVerticalSpacing(3)
        self.fields = {}
        for i, (key, text, unit, dec, step, tip) in enumerate(FIELDS):
            f = ScrubSpinBox('', 'plain', dec, unit, step=step, width=110)
            f.setToolTip(tip)
            f.committed.connect(lambda v, k=key, t=text, u=unit: self._set(f'{t.lower()} {v:g}{u}', **{k: v}))
            a = label(text, ink2=True)
            a.setFixedWidth(130)
            a.setToolTip(tip)
            g.addWidget(a, i, 0)
            g.addWidget(f, i, 1)
            self.fields[key] = f
        ll.addLayout(g)
        self.instants = SegmentedControl([('all', 'Every sweep'), ('keyframes', 'Keyframes')], 'all')
        self.instants.setToolTip('Evaluate at every lidar sweep (10 Hz) or at the nuScenes keyframes (2 Hz)')
        self.instants.changed.connect(lambda k: self._set(f'instants {k}', instants=k))
        ll.addWidget(self.instants, 0, Qt.AlignmentFlag.AlignLeft)
        self.views = SegmentedControl([('fisheye', 'Fisheye'), ('pinhole', 'Pinhole views')], 'fisheye')
        self.views.setToolTip('The lifted detector on the fisheye images as recorded, or on pinhole views cut from '
                              'them (detection quality\'s views)')
        self.views.changed.connect(lambda k: self._set(f'images {k}', views=k))
        ll.addWidget(self.views, 0, Qt.AlignmentFlag.AlignLeft)
        cg = QGridLayout()
        self.class_box = {}
        for i, c in enumerate(BB.BEV_CLASSES):
            cb = QCheckBox(DT.CLASS_LABEL[c])
            cb.toggled.connect(lambda on, k=c: self._class_toggled(k, on))
            cg.addWidget(cb, i // 2, i % 2)
            self.class_box[c] = cb
        ll.addLayout(cg)
        ll.addWidget(label(RP.ASSUMPTIONS, muted=True, small=True, wrap=True))
        self.status = label('', muted=True, small=True, wrap=True)
        ll.addWidget(self.status)
        self.bar = QProgressBar()
        self.bar.setTextVisible(False)
        self.bar.setRange(0, 1000)
        self.bar.hide()
        ll.addWidget(self.bar)
        bl = QHBoxLayout()
        b = button('Defaults', 'ghost', None, 'Back to the default settings (the recordings stay)')
        b.clicked.connect(self._defaults)
        bl.addWidget(b)
        bl.addStretch(1)
        self.b_export = button('Export…', None, 'export', 'The study as JSON and CSV, the region maps as PNG, the '
                                                          'findings and the perception records')
        self.b_export.clicked.connect(lambda: self.export())
        bl.addWidget(self.b_export)
        self.b_cancel = button('Cancel', 'ghost', None, 'Stop the run or the drives')
        self.b_cancel.clicked.connect(lambda: self.cancel())
        bl.addWidget(self.b_cancel)
        self.b_run = button('Run', 'primary', 'play', 'Detect and evaluate (a background job)')
        self.b_run.clicked.connect(lambda: self.run())
        bl.addWidget(self.b_run)
        ll.addLayout(bl)
        ll.addStretch(1)
        sc = QScrollArea()
        sc.setWidgetResizable(True)
        sc.setWidget(w)
        fit_width(sc)
        sc.setMinimumWidth(360)
        return sc

    # ------------------------------------------------------------------ right
    def _right(self):
        w = QWidget()
        w.setObjectName('panel')
        rl = QVBoxLayout(w)
        rl.setContentsMargins(10, 8, 10, 8)
        rl.setSpacing(8)
        pick = QHBoxLayout()
        self.cls_pick = QComboBox()
        self.cls_pick.setToolTip('The class shown')
        self.cls_pick.setPlaceholderText('Class')
        self.cls_pick.currentIndexChanged.connect(lambda *_: self._show())
        pick.addWidget(self.cls_pick, 1)
        self.rig_pick = QComboBox()
        self.rig_pick.setToolTip('The rig compared with the reference')
        self.rig_pick.setPlaceholderText('Compared rig')
        self.rig_pick.currentIndexChanged.connect(lambda *_: self._show())
        pick.addWidget(self.rig_pick, 2)
        self.metric_pick = SegmentedControl([('recall', 'Recall'), ('ap', 'mAP'), ('visible_share', 'Visible')],
                                            'recall')
        self.metric_pick.setToolTip('The figure the rigs\' maps show (the difference map is always recall)')
        self.metric_pick.changed.connect(lambda *_: self._show())
        pick.addWidget(self.metric_pick)
        for c in (self.cls_pick, self.rig_pick, self.metric_pick):
            c.setEnabled(False)
        rl.addLayout(pick)
        tiles = QHBoxLayout()
        self.tiles = {}
        for key, cap, tip in (('objects', 'Objects', 'Annotated object-instants of the class within range'),
                              ('ref', 'Reference recall', 'The reference rig, every region'),
                              ('cmp', 'Compared recall', 'The compared rig, every region'),
                              ('diff', 'Δ recall', 'Compared − reference, paired interval'),
                              ('ap', 'mAP', 'The compared rig\'s nuScenes mean AP'),
                              ('ate', 'ATE', 'The compared rig\'s mean centre error of matches'),
                              ('findings', 'Findings', 'Differences whose interval excludes zero')):
            t = self.tiles[key] = Tile(cap, tip)
            tiles.addWidget(t, 1)
        rl.addLayout(tiles)
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        mw = QWidget()
        ml = QHBoxLayout(mw)
        ml.setContentsMargins(0, 6, 0, 0)
        self.map_a, self.map_b, self.map_d = PolarMap(), PolarMap(), PolarMap()
        for m in (self.map_a, self.map_b, self.map_d):
            ml.addWidget(m, 1)
        self.tabs.addTab(mw, 'Maps')
        self.region_table = _table(REGION_COLS)
        self.tabs.addTab(self.region_table, 'Regions')
        self.cmp_table = _table(CMP_COLS)
        self.tabs.addTab(self.cmp_table, 'Compare')
        self.find_list = QListWidget()
        self.find_list.setWordWrap(True)
        self.tabs.addTab(self.find_list, 'Findings')
        yw = QWidget()
        yl = QVBoxLayout(yw)
        yl.setContentsMargins(0, 6, 0, 0)
        self.yield_note = label('Press Yield to count the objects per region.', muted=True, small=True, wrap=True)
        yl.addWidget(self.yield_note)
        self.yield_table = _table(YIELD_COLS)
        yl.addWidget(self.yield_table, 1)
        self.tabs.addTab(yw, 'Yield')
        fw = QWidget()
        fl = QVBoxLayout(fw)
        fl.setContentsMargins(0, 6, 0, 0)
        ctl = QHBoxLayout()
        self.frame_rec = QComboBox()
        self.frame_rec.setToolTip('The recording shown')
        self.frame_rec.currentIndexChanged.connect(lambda *_: self._frames_changed())
        ctl.addWidget(self.frame_rec, 1)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setToolTip('The recording\'s instants')
        self.slider.valueChanged.connect(lambda *_: self.show_frame())
        ctl.addWidget(self.slider, 2)
        fl.addLayout(ctl)
        self.bev = BevFrame()
        fl.addWidget(self.bev, 1)
        fl.addWidget(label('Objects: found (filled blue), missed (orange ring), ignore regions (grey). Detections at '
                           'the operating point: matched (blue cross), false (orange cross). To scale.', muted=True,
                           small=True, wrap=True))
        self.tabs.addTab(fw, 'Frames')
        nw = QWidget()
        nl = QVBoxLayout(nw)
        nl.setContentsMargins(0, 6, 0, 0)
        self.notes_label = label('', muted=True, small=True, wrap=True)
        self.notes_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        sc = QScrollArea()
        sc.setWidgetResizable(True)
        sc.setWidget(self.notes_label)
        nl.addWidget(sc, 1)
        self.tabs.addTab(nw, 'Notes')
        rl.addWidget(self.tabs, 1)
        return w

    # ------------------------------------------------------------------ settings
    @property
    def settings(self) -> RP.RigPerceptionSettings:
        return RP.settings_of(self.ctx.store.project)

    def _store(self, st: RP.RigPerceptionSettings, text) -> bool:
        new = RP.RigPerceptionSettings.from_dict(st.to_dict()).to_dict()
        if new == self.settings.to_dict():
            self._sync_fields()
            return False

        def fn(p):
            p.analysis.rig_perception = new
        self.ctx.store.edit('analysis', fn, f'Rig perception · {text}')
        return True

    def _set(self, text, **kw):
        if self._syncing:
            return False
        st = self.settings
        for k, v in kw.items():
            setattr(st, k, v)
        return self._store(st, text)

    def _defaults(self):
        st = self.settings
        self._store(RP.RigPerceptionSettings(recordings=list(st.recordings), scene_set=dict(st.scene_set)), 'defaults')

    def _class_toggled(self, c, on):
        if self._syncing:
            return
        cl = [k for k in BB.BEV_CLASSES if (k == c and on) or (k != c and k in self.settings.classes)]
        self._set(f'{DT.CLASS_LABEL[c].lower()} {"on" if on else "off"}', classes=cl)

    # ---- the scene set
    def scene_settings(self) -> PS.SceneSetSettings:
        return PS.SceneSetSettings.from_dict(self.settings.scene_set)

    def _set_scene(self, text, **kw):
        if self._syncing:
            return False
        ss = self.scene_settings()
        for k, v in kw.items():
            setattr(ss, k, v)
        return self._set(text, scene_set=ss.to_dict())

    def _set_picked(self, i):
        k = self.set_pick.itemData(i)
        if k:
            self._set_scene(f'scene set {k}', set_id=k)

    def _var_toggled(self, item):
        if self._syncing:
            return
        uids = [self.var_list.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self.var_list.count())
                if self.var_list.item(i).checkState() == Qt.CheckState.Checked]
        self._set_scene('scene set variants', variants=uids)

    def _seeds_edited(self):
        ss = PS.SceneSetSettings.from_dict({'seeds': [x for x in self.seeds.text().replace(';', ',').split(',')
                                                      if x.strip()]})
        self._set_scene(f'seeds {", ".join(map(str, ss.seeds))}', seeds=ss.seeds)
        self._sync_fields()

    # ---- the backend
    def _store_ext(self) -> BB.ExternalStore:
        return BB.ExternalStore(self.ctx.settings)

    def _backend_picked(self, k):
        if self._syncing:
            return
        if k == 'lifted':
            self._set('detector lifted YOLOX', backend='lifted')
        else:
            names = self._store_ext().names()
            if names:
                self._set(f'detector {names[0]}', backend=f'external:{names[0]}')
            else:
                self._sync_fields()
                self.ext_box.show()

    def _ext_picked(self, i):
        name = self.ext_pick.itemData(i)
        if name and not self._syncing:
            self._set(f'detector {name}', backend=f'external:{name}')
        self._sync_ext()

    def new_external(self, name=None) -> bool:
        if name is None:
            name, ok = QInputDialog.getText(self, 'External detector', 'Name (letters, digits, - and _):')
            if not ok:
                return False
        name = BB.clean_name(name)
        if not name:
            return False
        self.ext_argv.setPlainText('')
        self.ext_pick.blockSignals(True)
        if self.ext_pick.findData(name) < 0:
            self.ext_pick.addItem(name, name)
        self.ext_pick.setCurrentIndex(self.ext_pick.findData(name))
        self.ext_pick.blockSignals(False)
        self._sync_ext()
        return True

    def _argv(self) -> list:
        return [ln for ln in (x.strip() for x in self.ext_argv.toPlainText().splitlines()) if ln]

    def save_external(self) -> bool:
        name = self.ext_pick.currentData()
        if not name:
            self.ctx.msg('Name the detector first (New…).', 5000)
            return False
        try:
            self._store_ext().put(name, self._argv())
        except ValueError as e:
            self.ctx.msg(str(e), 6000)
            return False
        self._set(f'detector {name}', backend=f'external:{name}')
        self._sync_ext()
        return True

    def approve_external(self) -> bool:
        name = self.ext_pick.currentData()
        d = self._store_ext().get(name) if name else None
        if d is None:
            return False
        ok = self._store_ext().approve(name, d['argv'])
        self.ctx.msg(f'Approved: {" ".join(d["argv"])}' if ok else 'Not approved.', 6000)
        self._sync_ext()
        return ok

    def _sync_ext(self):
        name = self.ext_pick.currentData()
        es = self._store_ext()
        d = es.get(name) if name else None
        if d is None:
            self.ext_status.set('warning', 'Set the command (one argument per line) and Save.')
            self.b_approve.setEnabled(False)
            return
        if not self.ext_argv.hasFocus() and self._argv() != d['argv']:
            self.ext_argv.setPlainText('\n'.join(d['argv']))
        ok = es.approved(name) is not None
        self.ext_status.set('good' if ok else 'serious', ('Approved: ' if ok else 'Not approved yet: ') +
                            ' '.join(d['argv']))
        self.b_approve.setEnabled(not ok)

    # ---- recordings
    def add_recording(self, path, rig='') -> bool:
        p = os.path.abspath(os.path.expanduser(str(path)))
        info = RP.recording_info(p)
        if not info['ok']:
            self.ctx.msg(info['text'], 6000)
            return False
        st = self.settings
        if any(r['path'] == p for r in st.recordings):
            return False
        self._info[p] = info
        return self._set(f'recording {info.get("label") or os.path.basename(p)}',
                         recordings=st.recordings + [{'path': p, 'rig': rig or info.get('variant') or ''}])

    def remove_recording(self, i=None) -> bool:
        st = self.settings
        i = self.recs.currentRow() if i is None else i
        if not 0 <= i < len(st.recordings):
            return False
        recs = list(st.recordings)
        gone = recs.pop(i)
        return self._set(f'removed {os.path.basename(gone["path"])}', recordings=recs)

    def rename_rig(self, i=None, name=None) -> bool:
        st = self.settings
        i = self.recs.currentRow() if i is None else i
        if not 0 <= i < len(st.recordings):
            return False
        if name is None:
            name, ok = QInputDialog.getText(self, 'Rig', 'The rig this recording was made with:',
                                            text=self._rig_of(st.recordings[i]))
            if not ok:
                return False
        recs = [dict(r) for r in st.recordings]
        recs[i]['rig'] = str(name).strip()[:120]
        return self._set(f'rig {name}', recordings=recs)

    def make_reference(self, i=None) -> bool:
        st = self.settings
        i = self.recs.currentRow() if i is None else i
        if not 0 < i < len(st.recordings):
            return False
        rig = self._rig_of(st.recordings[i])
        recs = [r for r in st.recordings if self._rig_of(r) == rig] + \
            [r for r in st.recordings if self._rig_of(r) != rig]
        return self._set(f'reference {rig}', recordings=recs)

    def _rig_of(self, r) -> str:
        return r.get('rig') or self._rec_info(r['path']).get('variant') or os.path.basename(r['path'])

    def _rec_info(self, p):
        if p not in self._info:
            self._info[p] = RP.recording_info(p)
        return self._info[p]

    def _pick_dir(self):
        d = QFileDialog.getExistingDirectory(self, 'Add a recording with ground truth', '')
        if d:
            self.add_recording(d)

    def open_path(self, path) -> bool:
        ok = self.add_recording(path) or any(r['path'] == os.path.abspath(path) for r in self.settings.recordings)
        self.refresh()
        return ok

    # ------------------------------------------------------------------ state
    def _key(self, p=None):
        p = p or self.ctx.store.project
        d = RP.settings_of(p).to_dict()
        d.pop('scene_set', None)
        return json.dumps(d, sort_keys=True)

    @property
    def stale(self) -> bool:
        return self.result is not None and self._for is not None and self._for[1] != self._key()

    def _on_change(self, *_):
        if not self.isVisible():
            self._dirty = True
            return
        self._sync_fields()
        self.refresh()

    def _on_replaced(self, *_):
        self._gen += 1
        self.result, self._for, self.busy, self.drive_busy = None, None, False, False
        self.yield_doc = None
        self.status_text = ''
        self._info = {}
        self._sync_fields()
        self._fill_pickers()
        self.refresh()

    def _retheme(self, *_):
        self._show()

    def showEvent(self, e):
        super().showEvent(e)
        self.activate()

    def activate(self):
        if self._dirty:
            self._dirty = False
            self._sync_fields()
        self.refresh()

    def _sync_fields(self):
        st = self.settings
        ss = self.scene_settings()
        p = self.ctx.store.project
        self._syncing = True
        try:
            ext = st.backend.startswith('external:')
            self.backend_pick.set_value('external' if ext else 'lifted')
            self.ext_box.setVisible(ext or self.backend_pick.value() == 'external')
            names = self._store_ext().names()
            cur = st.backend[9:] if ext else (self.ext_pick.currentData() or '')
            self.ext_pick.clear()
            for n in names + ([cur] if cur and cur not in names else []):
                self.ext_pick.addItem(n, n)
            if cur:
                self.ext_pick.setCurrentIndex(max(self.ext_pick.findData(cur), 0))
            for k, f in self.fields.items():
                f.set_value(getattr(st, k))
            self.instants.set_value(st.instants)
            self.views.set_value(st.views)
            for c, cb in self.class_box.items():
                cb.setChecked(c in st.classes)
            self.recs.clear()
            ref = self._rig_of(st.recordings[0]) if st.recordings else ''
            for r in st.recordings:
                info = self._rec_info(r['path'])
                rig = self._rig_of(r)
                it = QListWidgetItem(('★ ' if rig == ref else '') + f'{rig} · ' +
                                     (info.get('label') or os.path.basename(r['path'])) +
                                     ('' if info['ok'] else ' · not readable'))
                it.setToolTip(f'{r["path"]}\n{info["text"]}')
                if not info['ok']:
                    it.setForeground(QColor(STATUS['serious']))
                self.recs.addItem(it)
            self.set_pick.setCurrentIndex(max(self.set_pick.findData(ss.set_id), 0))
            self.var_list.clear()
            chosen = ss.variants or [p.variant.uid]
            for v in p.variants:
                it = QListWidgetItem(v.name)
                it.setData(Qt.ItemDataRole.UserRole, v.uid)
                it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                it.setCheckState(Qt.CheckState.Checked if v.uid in chosen else Qt.CheckState.Unchecked)
                self.var_list.addItem(it)
            self.seeds.setText(', '.join(str(s) for s in ss.seeds))
        finally:
            self._syncing = False
        self._sync_model()
        self._sync_ext()
        self._sync_set()

    def _sync_model(self):
        st = self.settings
        if st.backend.startswith('external:'):
            self.model_status.hide()
            return
        self.model_status.show()
        s = DT.status(st.model)
        self.model_status.set('good' if s['verified'] else 'warning',
                              s['text'] if s['verified'] else s['text'] + ' Download it on Simulate ▸ Detection '
                                                                          'quality.')

    def _sync_set(self):
        ss = self.scene_settings()
        try:
            est = PS.estimate(self.ctx.store.project, ss)
            n = est['drives']
            txt = (f'{len(PS.describe(ss))} scenes × {len(ss.seeds)} seed(s) × {len(ss.variants) or 1} variant(s): '
                   f'{n} drives, {est["done"]} done; about {est["remaining_s"] / 60:.0f} min of CARLA to go '
                   f'({est["basis"]}).')
            if est.get('problems'):
                txt += ' ' + ' '.join(est['problems'])
        except Exception as e:                     # noqa: BLE001 — a project the campaign cannot plan
            txt = f'Cannot plan the set: {e}'
        self.set_text.setText(txt)
        self.set_text.setToolTip('\n'.join(f'{d["scenario"]}: {d["puts"]}' for d in PS.describe(ss)))

    # ------------------------------------------------------------------ jobs
    def _start(self, kind, title, work):
        win = self.ctx.window
        gen = self._gen
        if win is not None and hasattr(win, 'run_job'):
            return win.run_job(kind, title, work, lambda out: self._thread_done(gen, (kind, out)),
                               lambda m: self._failed(gen, m))

        def thread():
            try:
                out = work(lambda f, t='': self.sig.progress.emit(gen, f, t), lambda: gen != self._gen)
                self.sig.done.emit(gen, (kind, out))
            except Exception as e:                  # noqa: BLE001
                try:
                    self.sig.failed.emit(gen, f'{type(e).__name__}: {e}\n{traceback.format_exc()}')
                except RuntimeError:
                    pass
        threading.Thread(target=thread, daemon=True).start()
        return True

    def _progress(self, gen, f, text):
        if gen == self._gen:
            self.bar.show()
            self.bar.setValue(int(f * 1000))
            if text:
                self.status.setText(text)

    def _thread_done(self, gen, payload):
        if gen != self._gen:
            return
        kind, out = payload
        if kind == JOB_DRIVE:
            self.drive_busy = False
            if out is None:
                self.status_text = 'Drives cancelled'
            else:
                ok = sum(v in ('ok', 'reused') for v in out['states'].values())
                self.status_text = f'{ok} of {len(out["states"])} drives ready' + (
                    f'; failed: {"; ".join(out["errors"].values())[:200]}' if out.get('errors') else '')
                self.use_drives(out['recordings'])
            self._sync_set()
            self.refresh()
            return
        if kind == JOB_YIELD:
            self.yield_doc = out
            self.status_text = 'Yield counted' if out is not None else 'Cancelled'
            self._show_yield()
            self.refresh()
            return
        self.busy = False
        key, res = out if out is not None else (None, None)
        if res is None or key[0] is not self.ctx.store.project:
            self.status_text = 'Cancelled' if res is None else ''
            self.refresh()
            return
        self.result, self._for = res, key
        self.status_text = f'{res.timing.get("frames", 0)} instants in {res.seconds:.1f} s'
        self.ctx.store.project.add_log('rig_perception', f'{len(res.recordings)} recording(s), '
                                                         f'{len(res.rigs)} rig(s), {res.backend.get("name", "")}')
        self.ctx.msg(f'Rig perception · {self.status_text}', 5000)
        self._fill_pickers()
        self.refresh()

    def _failed(self, gen, msg):
        if gen == self._gen:
            self.busy = self.drive_busy = False
            self.status_text = 'Failed: ' + msg.splitlines()[0]
        self.refresh()

    def run(self) -> bool:
        if self.busy or self.drive_busy:
            return False
        st = self.settings
        if not st.recordings:
            self.status_text = 'Add a recording with ground truth first.'
            self.refresh()
            return False
        if self.detector is None and self.backend is None and not st.backend.startswith('external:'):
            s = DT.status(st.model)
            if not s['verified']:
                self.status_text = s['text'] + ' Download it on Simulate ▸ Detection quality.'
                self.refresh()
                return False
        p = self.ctx.store.project
        key = (p, self._key(p))
        det, be, qs = self.detector, self.backend, self.ctx.settings
        self._gen += 1
        self.busy = True
        self.status_text = 'Running…'
        self.bar.setValue(0)

        def work(progress, cancel):
            return key, RP.run(p, settings=st, progress=progress, cancel=cancel, backend=be, detector=det,
                               qsettings=qs)
        ok = self._start(JOB, 'Rig perception', work)
        if not ok:
            self.busy = False
        self.refresh()
        return ok

    def drive(self) -> bool:
        if self.busy or self.drive_busy:
            return False
        p = self.ctx.store.project
        ss = self.scene_settings()
        runner = self.runner
        self._gen += 1
        self.drive_busy = True
        self.status_text = 'Driving the scene set…'

        def work(progress, cancel):
            from ...core import simcampaign as SC
            try:
                return PS.run(p, ss, progress=progress, cancel=cancel, runner=runner)
            except SC.Cancelled:
                return None
        ok = self._start(JOB_DRIVE, 'Rig perception · scene set', work)
        if not ok:
            self.drive_busy = False
        self.refresh()
        return ok

    def use_drives(self, recs=None) -> int:
        """Put the scene set's finished drives in the recordings list (the variant's name as the rig)."""
        if recs is None:
            recs = PS.recordings(self.ctx.store.project, self.scene_settings())
        st = self.settings
        have = {r['path'] for r in st.recordings}
        new = [r for r in recs if r['path'] not in have]
        if not new:
            self.ctx.msg('No new drives of the scene set.', 4000)
            return 0
        self._set(f'{len(new)} scene-set drives', recordings=(st.recordings + new)[:RP.MAX_RECORDINGS])
        return len(new)

    def compute_yield(self) -> bool:
        st = self.settings
        if not st.recordings or self.busy or self.drive_busy:
            return False
        p = self.ctx.store.project
        from ...core import regions as RG
        grid = RG.default_grid(p)
        self._gen += 1

        def work(progress, cancel):
            return PS.yields(st.recordings, grid, st, progress)
        return self._start(JOB_YIELD, 'Rig perception · yield', work)

    def cancel(self):
        win = self.ctx.window
        for job in (JOB, JOB_DRIVE, JOB_YIELD):
            j = getattr(win, '_jobs', {}).get(job) if win is not None else None
            if j is not None:
                j['cancel'].set()
        if win is None or not hasattr(win, 'run_job'):
            if self.busy or self.drive_busy:
                self._gen += 1
                self.busy = self.drive_busy = False
                self.status_text = 'Cancelled'
        self.refresh()

    def export(self, folder=None) -> list:
        res = self.result
        if res is None:
            return []
        if folder is None:
            folder = QFileDialog.getExistingDirectory(self, 'Export the rig perception study into')
            if not folder:
                return []
        from ...core import provenance as PV
        paths = PV.stamp_paths(RP.export(res, self.ctx.store.project, folder), self.ctx.store.project, folder)
        self.ctx.msg(f'Exported the rig perception study to {folder}', 5000)
        return paths

    def summary(self) -> dict | None:
        return RP.summary(self.result) if self.result is not None else None

    # ------------------------------------------------------------------ showing
    def refresh(self):
        st = self.settings
        busy = self.busy or self.drive_busy
        self.b_run.setEnabled(not busy and bool(st.recordings))
        self.b_drive.setEnabled(not busy)
        self.b_cancel.setVisible(busy)
        self.b_export.setEnabled(self.result is not None and not busy)
        self.bar.setVisible(busy)
        txt = self.status_text
        if self.stale and not busy:
            txt = (txt + ' · ' if txt else '') + 'The settings changed since the run: Run again.'
        if not st.recordings and self.result is None and not txt:
            txt = EMPTY
        self.status.setText(txt)
        self._show()

    def _fill_pickers(self):
        res = self.result
        for cb in (self.cls_pick, self.rig_pick, self.frame_rec):
            cb.blockSignals(True)
            cb.clear()
        if res is not None:
            for c in res.classes:
                self.cls_pick.addItem(DT.CLASS_LABEL.get(c, c), c)
            v = self.cls_pick.findData('vehicle')
            self.cls_pick.setCurrentIndex(v if v >= 0 else 0)
            for r in res.rigs[1:]:
                self.rig_pick.addItem(r['name'], r['name'])
            for r in res.recordings:
                self.frame_rec.addItem(f'{r["rig"]} · {r["label"]}', r['index'])
        for cb in (self.cls_pick, self.rig_pick, self.frame_rec):
            cb.blockSignals(False)
        self.cls_pick.setEnabled(self.cls_pick.count() > 0)
        self.rig_pick.setEnabled(self.rig_pick.count() > 1)
        self.metric_pick.setEnabled(res is not None)
        self._frames_changed()

    def _cls(self):
        return self.cls_pick.currentData() or 'vehicle'

    def _show(self):
        res = self.result
        if res is None:
            for t in self.tiles.values():
                t.set('—')
            for m in (self.map_a, self.map_b, self.map_d):
                m.set_data(None, {}, 'recall', '')
            for t in (self.region_table, self.cmp_table):
                t.setRowCount(0)
            self.find_list.clear()
            self.notes_label.setText('\n\n'.join((RP.ASSUMPTIONS, RP.LIMITS_TEXT, BB.TORCH_NOTE)))
            return
        cls = self._cls()
        ref = res.rigs[0]['name']
        other = self.rig_pick.currentData() or (res.rigs[1]['name'] if len(res.rigs) > 1 else None)
        metric = self.metric_pick.value()
        a = res.cell(ref, cls) or {}
        b = res.cell(other, cls) if other else None
        comp = next((c for c in res.comparisons if c['class'] == cls and c['rig'] == other and c['region'] == 'all'),
                    None)
        self.tiles['objects'].set(f'{a.get("n_gt", 0)}', '', f'{(a.get("visible_share") or 0) * 100:.0f} % visible')
        self.tiles['ref'].set(_pc(a.get('recall')), '', ref)
        self.tiles['cmp'].set(_pc((b or {}).get('recall')) if b else '—', '', other or 'one rig only')
        if comp and comp.get('recall_difference') is not None:
            ci = comp.get('recall_difference_ci')
            sig = bool(ci and (ci[0] > 0 or ci[1] < 0))
            self.tiles['diff'].set(f'{comp["recall_difference"] * 100:+.1f}', 'pt',
                                   (f'[{ci[0] * 100:+.1f}, {ci[1] * 100:+.1f}]' if ci else '') +
                                   ('' if sig else ' · n.s.'), 'good' if sig and comp['recall_difference'] > 0 else
                                   ('serious' if sig else None))
        else:
            self.tiles['diff'].set('—')
        cb = b or a
        self.tiles['ap'].set(_pc(cb.get('ap')), '', 'nuScenes, 0.5–4 m')
        self.tiles['ate'].set('—' if cb.get('ate_m') is None else f'{cb["ate_m"]:.2f}', 'm', 'matches at the score')
        nsig = sum(1 for f in res.findings if f['significant'])
        self.tiles['findings'].set(str(nsig), '', 'significant')
        # maps
        lab = DT.CLASS_LABEL.get(cls, cls)

        def vals(rig):
            out = {}
            for rid, c in ((res.metrics.get(rig) or {}).get(cls) or {}).items():
                if rid == 'all':
                    continue
                v = c.get(metric)
                out[rid] = (v, c['n_gt'] >= res.settings.min_gt, c['n_gt'], c.get(f'{metric}_ci'))
            return out
        mname = {'recall': 'recall', 'ap': 'mAP', 'visible_share': 'visible share'}[metric]
        self.map_a.set_data(res.grid, vals(ref), metric, f'{lab} · {ref} · {mname}')
        if other:
            self.map_b.set_data(res.grid, vals(other), metric, f'{lab} · {other} · {mname}')
            dv = {}
            for c in res.comparisons:
                if c['class'] == cls and c['rig'] == other and c['region'] != 'all':
                    ci = c.get('recall_difference_ci')
                    n = min(c['n_gt'], c['n_gt_reference'])
                    sig = bool(ci and (ci[0] > 0 or ci[1] < 0) and n >= res.settings.min_gt)
                    dv[c['region']] = (c.get('recall_difference'), sig, n, ci)
            self.map_d.set_data(res.grid, dv, 'difference', f'{lab} · Δ recall (hatched: not significant)')
        else:
            self.map_b.set_data(None, {}, metric, 'One rig: add another to compare')
            self.map_d.set_data(None, {}, 'difference', '')
        # tables
        rows = [r for r in RP.region_rows(res) if r['class'] == cls]
        t = self.region_table
        t.setRowCount(len(rows))
        for i, r in enumerate(rows):
            _cell(t, i, 0, r['rig'])
            _cell(t, i, 1, DT.CLASS_LABEL.get(r['class'], r['class']))
            _cell(t, i, 2, r['region_name'])
            _cell(t, i, 3, str(r['n_gt']), True)
            _cell(t, i, 4, _pc(r.get('visible_share')), True)
            _cell(t, i, 5, _pc(r.get('recall'), r.get('recall_ci')), True)
            _cell(t, i, 6, _pc(r.get('precision'), r.get('precision_ci')), True)
            _cell(t, i, 7, _pc(r.get('ap'), r.get('ap_ci')), True)
            _cell(t, i, 8, '—' if r.get('ate_m') is None else f'{r["ate_m"]:.2f}', True)
        crow = [c for c in res.comparisons if c['class'] == cls and (other is None or c['rig'] == other)]
        t = self.cmp_table
        t.setRowCount(len(crow))
        for i, c in enumerate(crow):
            ci = c.get('recall_difference_ci')
            sig = bool(ci and (ci[0] > 0 or ci[1] < 0))
            _cell(t, i, 0, DT.CLASS_LABEL.get(c['class'], c['class']))
            _cell(t, i, 1, c['region_name'])
            _cell(t, i, 2, f'{c["n_gt"]} ({c["n_gt_reference"]})', True)
            _cell(t, i, 3, f'{_pc(c.get("recall_reference"))} → {_pc(c.get("recall"))}', True)
            _cell(t, i, 4, _dpc(c.get('recall_difference'), ci), True,
                  color=QColor(STATUS['good' if (c.get('recall_difference') or 0) > 0 else 'serious']) if sig
                  else None)
            _cell(t, i, 5, f'{_pc(c.get("ap_reference"))} → {_pc(c.get("ap"))}', True)
            _cell(t, i, 6, _dpc(c.get('ap_difference'), c.get('ap_difference_ci')), True)
            _cell(t, i, 7, c['pairing'])
        self.find_list.clear()
        for f in res.findings:
            if f['metric'] == 'note' or f['significant'] or f['class'] == cls:
                it = QListWidgetItem(f['text'])
                if f['significant']:
                    it.setForeground(QColor(STATUS['good' if f['difference'] > 0 else 'serious']))
                self.find_list.addItem(it)
        self.notes_label.setText('\n\n'.join(list(res.notes) + [f'Detector: {res.backend.get("name")} '
                                                                 f'({res.backend.get("version")}).',
                                                                 f'Evaluated range: {res.eval_range_m:.1f} m from the '
                                                                 'grid\'s reference point.', RP.ASSUMPTIONS,
                                                                 RP.LIMITS_TEXT, BB.TORCH_NOTE]))
        self._show_yield()
        self.show_frame()

    def _show_yield(self):
        y = self.yield_doc
        t = self.yield_table
        if not y:
            t.setRowCount(0)
            return
        cls = self._cls() if self.result is not None else None
        rows = []
        from ...core import regions as RG
        grid = RG.RegionGrid.from_dict(y.get('grid') or {})
        for c, d in y['object_instants'].items():
            if cls and c != cls:
                continue
            for rid, n in sorted(d.items(), key=lambda kv: -kv[1]):
                p = grid.parse_id(rid)
                rows.append((c, grid.region_name(*p) if p else rid, n, y['objects'].get(c, {}).get(rid, 0)))
        t.setRowCount(len(rows))
        for i, (c, name, n, k) in enumerate(rows):
            _cell(t, i, 0, DT.CLASS_LABEL.get(c, c))
            _cell(t, i, 1, name)
            _cell(t, i, 2, str(n), True)
            _cell(t, i, 3, str(k), True)
        empty = {c: len(v) for c, v in y.get('regions_empty', {}).items() if c == (cls or c)}
        self.yield_note.setText('Objects the recordings hold per region within '
                                f'{(y.get("evaluated_range_m") or 0):.0f} m. Regions without any: ' +
                                ', '.join(f'{DT.CLASS_LABEL.get(c, c)} {n}' for c, n in empty.items()) + '.')

    def _frames_changed(self):
        res = self.result
        n = len(res.frames.get(self.frame_rec.currentData(), [])) if res is not None else 0
        self.slider.blockSignals(True)
        self.slider.setRange(0, max(n - 1, 0))
        self.slider.blockSignals(False)
        self.show_frame()

    def show_frame(self):
        res = self.result
        if res is None:
            self.bev.set_frame(None, None, 0.3, 60.0)
            return
        ri = self.frame_rec.currentData()
        fr = res.frames.get(ri, [])
        if not fr:
            self.bev.set_frame(None, None, 0.3, 60.0)
            return
        k = min(self.slider.value(), len(fr) - 1)
        f = fr[k]
        t0 = fr[0]['t']
        self.bev.set_frame(f, res.grid, res.settings.score, max(res.eval_range_m, 20.0),
                           f'{res.recordings[ri]["label"]} · t = {(f["t"] - t0) / 1e6:.2f} s · '
                           f'{len(f["gt"])} objects, {sum(1 for d in f["det"] if d[3] >= res.settings.score)} '
                           'detections')
