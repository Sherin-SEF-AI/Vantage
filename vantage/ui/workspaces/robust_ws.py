"""Simulate ▸ Robust perception: the rig's tolerances (mounting, thermal, vibration, calibration) carried to what its
perception sees, with error bars, measured on recordings with ground truth (core.robust_perception).

Left: the recordings, the mode (calibrated per car or the design calibration), each source with its figures from the
app's studies (editable), the budget (builds, Sobol base samples, instants), Run and Cancel. Centre: per region the
band over builds of the chosen class and figure (recall, BEV error, out of view, ghost rate) as a sector × range heat
map, the sensitivity of each figure to each source and camera (first-order and total Sobol indices, or one source at a
time for detection), and the worst build's images. Right: the headline per recording, the sources' figures and the
notes. The study runs in a background thread with a progress bar; settings are saved with the project (undoable)."""
from __future__ import annotations

import math
import os
import threading
import traceback

import numpy as np
from PyQt6.QtCore import QObject, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (QAbstractItemView, QComboBox, QFileDialog, QHBoxLayout, QHeaderView, QLabel,
                             QListWidget, QProgressBar, QScrollArea, QSlider, QSplitter, QTableWidget,
                             QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget)

from ...core import robust_perception as RP
from ..theme import THEME, mono, sans
from ..widgets import ROW_H, ScrubSpinBox, Section, SegmentedControl, ToggleSwitch, button, hline, label

# key, label, decimals, unit, scrub step, tooltip
SOURCE_FIELDS = {
    'mounting': (('mount_deg', 'Angle ±', 2, '°', 0.1, 'Mounting tolerance of each angle (custom)'),
                 ('mount_mm', 'Position ±', 1, 'mm', 1.0, 'Mounting tolerance of each position axis (custom)')),
    'thermal': (('thermal_deg', 'Angle σ', 3, '°', 0.01, 'Thermal drift per angle, 1 σ (custom)'),
                ('thermal_mm', 'Move σ', 2, 'mm', 0.1, 'Thermal move per axis, 1 σ (custom)')),
    'vibration': (('vib_body_deg', 'Body pitch σ', 3, '°', 0.01, 'The body\'s pitch on the suspension, 1 σ (custom)'),
                  ('vib_heave_mm', 'Heave σ', 1, 'mm', 1.0, 'The body\'s heave, 1 σ (custom)'),
                  ('vib_flex_deg', 'Mount flex', 3, '°', 0.005, 'RMS angle of a mount\'s flex (custom)'),
                  ('vib_flex_mm', 'Flex move', 2, 'mm', 0.01, 'RMS move of a mount\'s flex (custom)')),
    'calibration': (('calib_deg', 'Angle σ', 3, '°', 0.01, 'Calibration residual per angle, 1 σ (custom)'),
                    ('calib_mm', 'Position σ', 1, 'mm', 1.0, 'Calibration residual per axis, 1 σ (custom)')),
}
FROM_KEY = {'mounting': 'mount_from', 'thermal': 'thermal_from', 'vibration': 'vib_from', 'calibration': 'calib_from'}
FROM_OPTS = {'mounting': [('install', 'Install'), ('analysis', 'Analysis'), ('fleet_twin', 'Fleet'),
                          ('custom', 'Custom')],
             'thermal': [('study', 'Study'), ('custom', 'Custom')], 'vibration': [('study', 'Study'), ('custom', 'Custom')],
             'calibration': [('observability', 'Study'), ('custom', 'Custom')]}
BUDGET_FIELDS = (('samples', 'Builds', 0, '', 1.0, 'Builds evaluated with the detector (each one pass over every image)'),
                 ('sobol_n', 'Sobol n', 0, '', 8.0, 'Pick-freeze base samples of the BEV figures ((sources + 2) × n '
                                                    'builds, about 20 ms each); 0 = no indices'),
                 ('oat_samples', 'One at a time', 0, '', 1.0, 'Builds per source with only that source on (the '
                                                             'detector\'s sensitivity)'),
                 ('frame_step', 'Detector step', 0, '', 1.0, 'Every n-th instant goes to the detector'),
                 ('bev_frame_step', 'BEV step', 0, '', 1.0, 'Every n-th instant for the BEV figures'),
                 ('gate_m', 'Fusion gate', 1, 'm', 0.5, 'Two lifts of one object closer than max(gate, share × range) '
                                                        'fuse; farther: a ghost'),
                 ('seed', 'Seed', 0, '', 1.0, 'Random seed of the builds'))
FIELD_LABEL = {k: t for v in SOURCE_FIELDS.values() for k, t, *_ in v}
FIELD_LABEL.update({k: t for k, t, *_ in BUDGET_FIELDS})
FIELD_LABEL.update(mode='mode', recordings='recordings', detect='detector', lift='lift', mounting='mounting',
                   thermal='thermal', vibration='vibration', calibration='calibration', mount_from='mounting from',
                   thermal_from='thermal from', vib_from='vibration from', calib_from='calibration from',
                   mount_dist='mounting distribution')
FIGURES = (('recall', 'Recall'), ('bev', 'BEV error'), ('oov', 'Out of view'), ('ghost', 'Ghosts'))


class _Sig(QObject):
    done = pyqtSignal(int, object)
    failed = pyqtSignal(int, str)
    progress = pyqtSignal(int, float, str)
    sources = pyqtSignal(int, object)


def _table(cols):
    t = QTableWidget(0, len(cols))
    t.verticalHeader().setVisible(False)
    t.verticalHeader().setDefaultSectionSize(ROW_H)
    t.horizontalHeader().setFixedHeight(ROW_H)
    t.setShowGrid(False)
    t.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    t.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    for c, (head, tip) in enumerate(cols):
        it = QTableWidgetItem(head)
        it.setToolTip(tip)
        t.setHorizontalHeaderItem(c, it)
    hh = t.horizontalHeader()
    hh.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
    for c in range(1, len(cols)):
        hh.setSectionResizeMode(c, QHeaderView.ResizeMode.ResizeToContents)
    return t


def _f(v, nd=2, unit=''):
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return '—'
    return f'{v:.{nd}f}{unit}'


# =========================================================================== the region heat map
class RegionMap(QWidget):
    """Sectors (columns) × range bins (rows): each cell coloured by a value and labelled with its band."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.cells = {}          # region id -> (value, text, tip)
        self.grid = None
        self.lo, self.hi, self.good_high = 0.0, 1.0, True
        self.title = ''
        self.empty = 'No result yet'
        self.setMinimumHeight(220)
        self.setMouseTracking(True)

    def set_data(self, grid, cells, lo, hi, good_high=True, title='', empty='No data'):
        self.grid, self.cells = grid, cells
        self.lo, self.hi, self.good_high = lo, max(hi, lo + 1e-9), good_high
        self.title, self.empty = title, empty
        self.update()

    def _rect(self):
        return QRectF(96, 26, max(self.width() - 106, 10), max(self.height() - 36, 10))

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.setFont(sans(11))
        p.setPen(THEME.color('ink'))
        p.drawText(QRectF(8, 4, self.width() - 16, 18), Qt.AlignmentFlag.AlignLeft, self.title)
        r = self._rect()
        g = self.grid
        if g is None or not self.cells:
            p.setPen(THEME.color('muted'))
            p.drawText(r, Qt.AlignmentFlag.AlignCenter, self.empty)
            p.end()
            return
        ns, nr = g.sectors, g.n_ranges
        cw, ch = r.width() / ns, r.height() / nr
        names, ranges = g.sector_names(), g.range_names()
        good, bad = THEME.color('good'), THEME.color('critical')
        for s in range(ns):
            p.setPen(THEME.color('muted'))
            p.setFont(sans(10))
            p.drawText(QRectF(r.left() + s * cw, r.top() - 14, cw, 13), Qt.AlignmentFlag.AlignCenter, names[s])
        for k in range(nr):
            p.setPen(THEME.color('muted'))
            p.drawText(QRectF(0, r.top() + k * ch, r.left() - 6, ch),
                       Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, ranges[k])
        for s in range(ns):
            for k in range(nr):
                rid = g.region_id(s, k)
                cell = QRectF(r.left() + s * cw + 1, r.top() + k * ch + 1, cw - 2, ch - 2)
                v = self.cells.get(rid)
                if v is None:
                    p.fillRect(cell, THEME.color('raised'))
                    continue
                val, text, _ = v
                t = 0.5 if val is None else min(max((val - self.lo) / (self.hi - self.lo), 0.0), 1.0)
                if not self.good_high:
                    t = 1.0 - t
                c = QColor(int(bad.red() + (good.red() - bad.red()) * t), int(bad.green() + (good.green() -
                                                                                          bad.green()) * t),
                           int(bad.blue() + (good.blue() - bad.blue()) * t), 150)
                p.fillRect(cell, c)
                p.setPen(THEME.color('ink'))
                p.setFont(mono(10))
                p.drawText(cell, Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap, text)
        p.end()

    def region_at(self, pos):
        g = self.grid
        if g is None:
            return None
        r = self._rect()
        if not r.contains(pos):
            return None
        s = int((pos.x() - r.left()) / (r.width() / g.sectors))
        k = int((pos.y() - r.top()) / (r.height() / g.n_ranges))
        return g.region_id(min(s, g.sectors - 1), min(k, g.n_ranges - 1))

    def mouseMoveEvent(self, e):
        rid = self.region_at(e.position())
        v = self.cells.get(rid) if rid else None
        self.setToolTip(v[2] if v else '')


# =========================================================================== sensitivity bars
class IndexBars(QWidget):
    """First-order (filled) and total (outline) indices per group, with their intervals."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.rows = []           # (label, S, ST, S_ci)
        self.title = ''
        self.note = ''
        self.setMinimumHeight(160)

    def set_rows(self, rows, title='', note=''):
        self.rows, self.title, self.note = rows, title, note
        self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.setFont(sans(11))
        p.setPen(THEME.color('ink'))
        p.drawText(QRectF(8, 4, self.width() - 16, 18), Qt.AlignmentFlag.AlignLeft, self.title)
        if not self.rows:
            p.setPen(THEME.color('muted'))
            p.drawText(QRectF(0, 0, self.width(), self.height()), Qt.AlignmentFlag.AlignCenter, 'No indices')
            p.end()
            return
        x0, top = 150, 28
        wmax = max(self.width() - x0 - 60, 20)
        acc = THEME.cam(0)
        for i, (lab, S, ST, ci) in enumerate(self.rows):
            y = top + i * 22
            p.setPen(THEME.color('ink'))
            p.setFont(mono(10))
            p.drawText(QRectF(6, y, x0 - 10, 20), Qt.AlignmentFlag.AlignVCenter, lab)
            s = min(max(S or 0.0, 0.0), 1.0)
            p.fillRect(QRectF(x0, y + 4, wmax * s, 12), acc)
            if ST is not None:
                p.setPen(QPen(THEME.color('ink2'), 1))
                p.drawRect(QRectF(x0, y + 3, wmax * min(max(ST, 0.0), 1.0), 14))
            if ci:
                a, b = (min(max(v or 0.0, 0.0), 1.0) for v in ci)
                p.setPen(QPen(THEME.color('ink'), 1))
                p.drawLine(int(x0 + wmax * a), y + 10, int(x0 + wmax * b), y + 10)
            p.setPen(THEME.color('muted'))
            p.drawText(QRectF(x0 + wmax + 4, y, 56, 20), Qt.AlignmentFlag.AlignVCenter, _f(S, 2))
        p.setPen(THEME.color('muted'))
        p.setFont(sans(10))
        p.drawText(QRectF(8, top + len(self.rows) * 22 + 2, self.width() - 16, 30),
                   Qt.AlignmentFlag.AlignLeft | Qt.TextFlag.TextWordWrap, self.note)
        p.end()


# =========================================================================== the page
class RobustPage(QWidget):
    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        self.result: RP.RobustResult | None = None
        self.busy = False
        self._gen = 0
        self._syncing = False
        self.detector = None                  # a callable for tests; None: the project's YOLOX
        self.studies = None                   # reused source studies (tests)
        self._sources = None
        self.status_text = ''
        self.sig = _Sig()
        self.sig.done.connect(self._done)
        self.sig.failed.connect(self._failed)
        self.sig.progress.connect(self._progress)
        self.sig.sources.connect(self._sources_done)
        self.setObjectName('base')
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        split = QSplitter(Qt.Orientation.Horizontal)
        split.setHandleWidth(2)
        split.addWidget(self._left())
        split.addWidget(self._centre())
        split.addWidget(self._right())
        split.setSizes([320, 760, 520])
        lay.addWidget(split)
        ctx.store.changed.connect(self._on_store)
        ctx.store.project_replaced.connect(self._replaced)
        THEME.changed.connect(lambda *_: self.refresh())
        self._sync_fields()
        self.refresh()

    # ------------------------------------------------------------------ the settings panel
    def _spin(self, key, dec, unit, step, tip):
        f = ScrubSpinBox('', 'plain', dec, unit, step=step, width=90)
        f.setToolTip(tip)
        f.committed.connect(lambda v, k=key: self.set_option(k, v))
        self.fields[key] = f
        return f

    def _left(self):
        outer = QWidget()
        outer.setObjectName('panel')
        outer.setMinimumWidth(300)
        ol = QVBoxLayout(outer)
        ol.setContentsMargins(0, 0, 0, 0)
        ol.setSpacing(0)
        head = QWidget()
        hl = QVBoxLayout(head)
        hl.setContentsMargins(8, 6, 8, 4)
        hl.setSpacing(4)
        hl.addWidget(label('Robust perception'))
        hl.addWidget(hline())
        hl.addWidget(label('Each build of the car is drawn from the tolerances; its cameras see the recorded drive '
                           'turned and moved accordingly, while the perception keeps its calibration. Recall, BEV '
                           'error and ghosts per region, with the band over builds.', muted=True, small=True,
                           wrap=True))
        ol.addWidget(head)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        body = QWidget()
        body.setObjectName('panel')
        bl = QVBoxLayout(body)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.setSpacing(0)
        self.fields, self.segs, self.toggles = {}, {}, {}
        # ---- recordings
        s = Section('Recordings with ground truth')
        self.rec_list = QListWidget()
        self.rec_list.setFixedHeight(84)
        self.rec_list.setToolTip('CARLA runs with ground truth (Simulate ▸ CARLA); the first is the reference')
        s.add(self.rec_list)
        row = QWidget()
        rl = QHBoxLayout(row)
        rl.setContentsMargins(0, 0, 0, 0)
        b_add = button('Add…', None, 'folder', 'Add a recording folder')
        b_add.clicked.connect(lambda: self.add_recording())
        b_rm = button('Remove', None, 'trash', 'Remove the selected recording')
        b_rm.clicked.connect(self.remove_recording)
        rl.addWidget(b_add)
        rl.addWidget(b_rm)
        rl.addStretch(1)
        s.add(row)
        bl.addWidget(s)
        # ---- mode
        s = Section('Calibration')
        seg = SegmentedControl([('per_vehicle', 'Per car'), ('design', 'Design')])
        seg.setToolTip('Per car: each car is calibrated after it is built (the build\'s turn changes the images; the '
                       'calibration residual reaches the geometry). Design: every car uses the design poses (the '
                       'mounting error reaches the geometry in full)')
        seg.changed.connect(lambda k: self.set_option('mode', k))
        self.segs['mode'] = seg
        s.row('Mode', seg)
        seg = SegmentedControl([('contact', 'Contact'), ('box', 'Box')])
        seg.setToolTip('How a 2D object is lifted to the ground: its lowest visible point, or its box\'s '
                       'bottom-centre')
        seg.changed.connect(lambda k: self.set_option('lift', k))
        self.segs['lift'] = seg
        s.row('Lift', seg)
        bl.addWidget(s)
        # ---- sources
        self.src_notes = {}
        for key in RP.SOURCES:
            s = Section(RP.SOURCE_LABEL[key], expanded=key == 'mounting')
            tg = ToggleSwitch('')
            tg.setToolTip(f'Draw the {RP.SOURCE_LABEL[key].lower()} in every build')
            tg.toggled.connect(lambda on, k=key: self.set_option(k, bool(on)))
            self.toggles[key] = tg
            s.row('On', tg)
            seg = SegmentedControl(FROM_OPTS[key])
            seg.setToolTip('Where the figures come from: the app\'s own study, or the figures below')
            seg.changed.connect(lambda v, k=FROM_KEY[key]: self.set_option(k, v))
            self.segs[FROM_KEY[key]] = seg
            s.row('From', seg)
            if key == 'mounting':
                sg = SegmentedControl([('uniform', 'Uniform'), ('normal', 'Normal')])
                sg.setToolTip('Uniform within ± the tolerance, or normal with ± = 2 σ (as the tolerance study)')
                sg.changed.connect(lambda v: self.set_option('mount_dist', v))
                self.segs['mount_dist'] = sg
                s.row('Spread', sg)
            for k, text, dec, unit, step, tip in SOURCE_FIELDS[key]:
                s.row(text, self._spin(k, dec, unit, step, tip), tip=tip)
            note = label('', muted=True, small=True, wrap=True)
            self.src_notes[key] = note
            s.add(note)
            bl.addWidget(s)
        # ---- budget
        s = Section('Budget', expanded=False)
        tg = ToggleSwitch('')
        tg.setToolTip('Run YOLOX on every build\'s images (recall); off: the BEV figures only')
        tg.toggled.connect(lambda on: self.set_option('detect', bool(on)))
        self.toggles['detect'] = tg
        s.row('Detector', tg)
        for k, text, dec, unit, step, tip in BUDGET_FIELDS:
            s.row(text, self._spin(k, dec, unit, step, tip), tip=tip)
        self.budget_note = label('', muted=True, small=True, wrap=True)
        s.add(self.budget_note)
        bl.addWidget(s)
        bl.addStretch(1)
        scroll.setWidget(body)
        ol.addWidget(scroll, 1)
        foot = QWidget()
        fl = QVBoxLayout(foot)
        fl.setContentsMargins(8, 4, 8, 8)
        fl.setSpacing(4)
        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.setTextVisible(False)
        self.bar.setFixedHeight(4)
        fl.addWidget(self.bar)
        self.status = label('', muted=True, small=True, wrap=True)
        fl.addWidget(self.status)
        row = QHBoxLayout()
        self.b_run = button('Run', 'primary', 'play', 'Run the study on the recordings (minutes with the detector)')
        self.b_run.clicked.connect(lambda: self.run())
        self.b_cancel = button('Cancel', None, 'stop', 'Stop the running study')
        self.b_cancel.clicked.connect(self.cancel)
        self.b_export = button('Export…', None, 'export', 'The study as JSON (per region and class, the builds, the '
                                                          'indices), CSV, PNG and the perception records')
        self.b_export.clicked.connect(lambda: self.export())
        row.addWidget(self.b_run)
        row.addWidget(self.b_cancel)
        row.addStretch(1)
        row.addWidget(self.b_export)
        fl.addLayout(row)
        ol.addWidget(foot)
        return outer

    def _centre(self):
        w = QWidget()
        w.setObjectName('panel')
        cl = QVBoxLayout(w)
        cl.setContentsMargins(8, 4, 8, 6)
        cl.setSpacing(4)
        self.head = label('', ink2=True, wrap=True)
        cl.addWidget(self.head)
        row = QHBoxLayout()
        self.c_rec = QComboBox()
        self.c_rec.setToolTip('The recording the views show')
        self.c_rec.activated.connect(lambda _: self.refresh_views())
        row.addWidget(self.c_rec, 2)
        self.c_cls = QComboBox()
        self.c_cls.setToolTip('The class')
        for c in RP.CLASSES:
            self.c_cls.addItem(c, c)
        self.c_cls.activated.connect(lambda _: self.refresh_views())
        row.addWidget(self.c_cls, 1)
        self.fig = SegmentedControl(FIGURES, 'recall')
        self.fig.setToolTip('The figure the map shows')
        self.fig.changed.connect(lambda _: self.refresh_views())
        row.addWidget(self.fig)
        cl.addLayout(row)
        self.tabs = QTabWidget()
        self.map = RegionMap()
        self.tabs.addTab(self.map, 'Regions')
        sens = QWidget()
        sl = QVBoxLayout(sens)
        sl.setContentsMargins(0, 4, 0, 0)
        self.sens_src = IndexBars()
        self.sens_cam = IndexBars()
        self.sens_det = IndexBars()
        sl.addWidget(self.sens_src, 1)
        sl.addWidget(self.sens_cam, 1)
        sl.addWidget(self.sens_det, 1)
        self.tabs.addTab(sens, 'Sensitivity')
        wb = QWidget()
        wl = QVBoxLayout(wb)
        wl.setContentsMargins(0, 4, 0, 0)
        r2 = QHBoxLayout()
        self.c_cam = QComboBox()
        self.c_cam.setToolTip('The camera')
        self.c_cam.activated.connect(lambda _: self.refresh_image())
        r2.addWidget(self.c_cam)
        self.c_build = SegmentedControl([('worst', 'Worst build'), ('nominal', 'Nominal')], 'worst')
        self.c_build.changed.connect(lambda _: self.refresh_image())
        r2.addWidget(self.c_build)
        self.s_frame = QSlider(Qt.Orientation.Horizontal)
        self.s_frame.setToolTip('The instant')
        self.s_frame.valueChanged.connect(lambda _: self.refresh_image())
        r2.addWidget(self.s_frame, 1)
        wl.addLayout(r2)
        self.img = QLabel()
        self.img.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.img.setMinimumHeight(200)
        wl.addWidget(self.img, 1)
        self.img_note = label('', muted=True, small=True, wrap=True)
        wl.addWidget(self.img_note)
        self.tabs.addTab(wb, 'Worst build')
        cl.addWidget(self.tabs, 1)
        return w

    def _right(self):
        right = QSplitter(Qt.Orientation.Vertical)
        right.setHandleWidth(2)
        tw = QWidget()
        tw.setObjectName('panel')
        tl = QVBoxLayout(tw)
        tl.setContentsMargins(8, 6, 8, 6)
        tl.addWidget(label('Per region', bold=True))
        self.table = _table((('Region', 'Sector · range'), ('Objects', 'Objects (or object instants) in the region'),
                             ('Nominal', 'The nominal build\'s figure'), ('Mean', 'Mean over builds'),
                             ('5–95 %', '90 % of builds within'), ('p95 pooled', 'BEV: the 95th percentile of every '
                                                                              'object in every build')))
        tl.addWidget(self.table, 1)
        right.addWidget(tw)
        fw = QWidget()
        fw.setObjectName('panel')
        fl = QVBoxLayout(fw)
        fl.setContentsMargins(8, 6, 8, 6)
        fl.addWidget(label('Sources and findings', bold=True))
        sc = QScrollArea()
        sc.setWidgetResizable(True)
        sc.setFrameShape(QScrollArea.Shape.NoFrame)
        self.find = label('', small=True, wrap=True)
        self.find.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.find.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        sc.setWidget(self.find)
        fl.addWidget(sc, 1)
        self.b_store = button('Add records to the model', None, 'dataset', 'Hand the nominal build\'s perception '
                              'records (vantage.perception_record/1) to the learned rig model\'s store')
        self.b_store.clicked.connect(self.store_records)
        fl.addWidget(self.b_store)
        right.addWidget(fw)
        right.setSizes([480, 380])
        return right

    def _engine(self):
        """The window's shared engine (its occlusion maps are reused), else a share of the analysis engine."""
        win = getattr(self.ctx, 'window', None)
        try:
            if win is not None and hasattr(win, '_shared_engine'):
                return win._shared_engine()
            return self.ctx.analysis.engine.share()
        except Exception:                                        # noqa: BLE001 — the study makes its own
            return None

    # ------------------------------------------------------------------ settings
    @property
    def settings(self) -> RP.RobustSettings:
        return RP.settings_of(self.ctx.store.project)

    def set_option(self, key, value) -> bool:
        """Change one setting, saved with the project as one undo step; False when nothing changes."""
        if self._syncing:
            return False
        old = self.settings.to_dict()
        d = dict(old)
        d[key] = value
        new = RP.RobustSettings.from_dict(d).to_dict()
        if new == old:
            self._sync_fields()
            return False
        shown = new[key]
        if isinstance(shown, bool):
            shown = 'on' if shown else 'off'
        elif isinstance(shown, float):
            shown = f'{shown:g}'
        elif isinstance(shown, list):
            shown = f'{len(shown)} recording{"s" * (len(shown) != 1)}'

        def fn(p):
            p.analysis.robust_perception = new
        self.ctx.store.edit('analysis', fn, f'Robust perception · {FIELD_LABEL.get(key, key)} {shown}')
        return True

    def add_recording(self, path=None) -> bool:
        if path is None:
            path = QFileDialog.getExistingDirectory(self, 'A recording with ground truth (a CARLA run)')
            if not path:
                return False
        from ...core import detection_eval as DE
        info = DE.recording_info(path)
        if not info.get('ok'):
            self.ctx.msg(info.get('text', 'Not a recording with ground truth'), 6000)
            return False
        recs = [r for r in self.settings.recordings if os.path.normpath(r) != os.path.normpath(path)] + [str(path)]
        return self.set_option('recordings', recs)

    def remove_recording(self):
        i = self.rec_list.currentRow()
        recs = list(self.settings.recordings)
        if 0 <= i < len(recs):
            recs.pop(i)
            self.set_option('recordings', recs)

    def _sync_fields(self):
        st = self.settings
        self._syncing = True
        try:
            for key, f in self.fields.items():
                f.set_value(float(getattr(st, key)))
            for key, sg in self.segs.items():
                sg.set_value(getattr(st, key))
            for key, tg in self.toggles.items():
                tg.setChecked(bool(getattr(st, key)))
            for key in RP.SOURCES:
                custom = getattr(st, FROM_KEY[key]) == 'custom'
                for k, *_ in SOURCE_FIELDS[key]:
                    self.fields[k].setEnabled(custom)
            self.rec_list.clear()
            for r in st.recordings:
                self.rec_list.addItem(os.path.basename(os.path.normpath(r)))
            n_act = len(st.active())
            self.budget_note.setText(
                f'{st.samples} detector builds + {n_act * st.oat_samples if n_act > 1 else 0} one at a time; '
                f'{(n_act + 2) * st.sobol_n + (len(RP.SOURCES) + 4) * max(st.sobol_n // 2, 8)} BEV builds. '
                + ('The calibration source is off in the design mode.' if st.mode == 'design' else ''))
            self._fill_sources()
        finally:
            self._syncing = False

    def _fill_sources(self):
        srcs = self._sources if self._sources is not None else (self.result.sources if self.result else None)
        for key, note in self.src_notes.items():
            s = (srcs or {}).get(key)
            if s is None:
                note.setText('Figures from the study when the page runs.')
                continue
            rows = s.table()
            bits = [s.origin]
            for r in rows[:8]:
                vals = ', '.join(f'{k} {v:g}' for k, v in r.items() if k != 'camera' and isinstance(v, (int, float)))
                bits.append(f'{r["camera"]}: {vals}')
            sh = s.to_doc()['shared']
            if sh:
                bits.append(', '.join(f'{k} {v:g}' if isinstance(v, (int, float)) else f'{k}: {len(v)}'
                                      for k, v in sh.items()))
            bits += s.notes
            note.setText('\n'.join(bits))

    # ------------------------------------------------------------------ running
    def _on_store(self, *_):
        self._sync_fields()
        self.refresh_status()

    def _replaced(self, *_):
        self._gen += 1
        self.result, self.busy, self._sources = None, False, None
        self._sync_fields()
        self.refresh()

    def activate(self):
        self._sync_fields()
        if self._sources is None and self.ctx.store.project is not None:
            self.load_sources()

    def load_sources(self):
        """The sources' figures from the app's studies (a few seconds: the observability study), in the background."""
        p = self.ctx.store.project
        project = p.snapshot()
        st = self.settings
        gen = self._gen
        tags = [c.tag for c in project.variant.cameras if c.visible]
        studies = self.studies
        eng = self._engine()

        def thread():
            try:
                src = RP.resolve_sources(project, st, tags, None, eng, studies)
                self.sig.sources.emit(gen, src)
            except Exception:                                    # noqa: BLE001 — the run reports its own errors
                pass
        threading.Thread(target=thread, daemon=True).start()

    def _sources_done(self, gen, src):
        if gen != self._gen:
            return
        self._sources = src
        self._fill_sources()

    def run(self, quiet=False) -> bool:
        st = self.settings
        if not st.recordings:
            self.status_text = 'Add a recording with ground truth first (a CARLA run with ground truth on).'
            self.refresh_status()
            return False
        project = self.ctx.store.project.snapshot()
        self._gen += 1
        gen = self._gen
        self.busy = True
        self.status_text = 'Reading the ground truth…'
        self.bar.setValue(0)
        self.refresh_status()
        det, studies = self.detector, self.studies
        eng = self._engine()

        def thread():
            try:
                res = RP.run(project, eng, settings=st, detector=det, studies=studies,
                             progress=lambda f, t='': self.sig.progress.emit(gen, f, t),
                             cancel=lambda: gen != self._gen)
                self.sig.done.emit(gen, res)
            except RuntimeError:
                pass
            except Exception as e:                               # noqa: BLE001
                try:
                    self.sig.failed.emit(gen, f'{type(e).__name__}: {e}\n{traceback.format_exc()}')
                except RuntimeError:
                    pass
        threading.Thread(target=thread, daemon=True).start()
        return True

    def cancel(self):
        if self.busy:
            self._gen += 1
            self.busy = False
            self.status_text = 'Cancelled'
            self.refresh_status()

    def _progress(self, gen, f, text):
        if gen == self._gen:
            self.bar.setValue(int(f * 1000))
            if text:
                self.status_text = text
                self.status.setText(text)

    def _done(self, gen, res):
        if gen != self._gen:
            return
        self.busy = False
        if res is None:
            self.status_text = 'Cancelled'
            self.refresh_status()
            return
        self.result = res
        self._sources = res.sources
        n = sum(r.info['objects'] for r in res.recordings)
        self.status_text = (f'{len(res.recordings)} recording{"s" * (len(res.recordings) != 1)}, {n} object views, '
                            f'{res.settings.samples} builds with the detector · {res.seconds:.0f} s')
        if res.records:
            self.status_text += f' · {len(res.records)} perception record(s)'
        self.bar.setValue(1000)
        self._sync_fields()
        self.refresh()

    def _failed(self, gen, msg):
        if gen != self._gen:
            return
        self.busy = False
        self.status_text = 'Failed: ' + msg.splitlines()[0]
        self.refresh_status()

    # ------------------------------------------------------------------ views
    def refresh_status(self):
        self.status.setText(self.status_text)
        self.b_cancel.setEnabled(self.busy)
        self.b_run.setEnabled(not self.busy)
        self.b_export.setEnabled(self.result is not None)
        if hasattr(self, 'b_store'):
            self.b_store.setEnabled(bool(self.result is not None and self.result.records))

    def refresh(self):
        self.refresh_status()
        res = self.result
        cur = self.c_rec.currentIndex()
        self.c_rec.blockSignals(True)
        self.c_rec.clear()
        for rr in (res.recordings if res else []):
            self.c_rec.addItem(rr.label)
        self.c_rec.setCurrentIndex(min(max(cur, 0), self.c_rec.count() - 1))
        self.c_rec.blockSignals(False)
        if res is None:
            self.head.setText('Add a recording with ground truth and run: every build of the car is drawn from the '
                              'tolerances, and its cameras see the drive as they would.')
        else:
            st = res.settings
            det = (f' · {st.samples} builds with {res.detector.get("name", "the detector")}' if st.detect else
                   ' · BEV figures only')
            self.head.setText(f'{RP.MODE_LABEL[st.mode]} · sources: '
                              + (', '.join(RP.SOURCE_LABEL[s].lower() for s in st.active()) or 'none') + det)
        self.refresh_views()
        self._fill_findings()

    def _rr(self):
        res = self.result
        if res is None or not res.recordings:
            return None
        return res.recordings[min(max(self.c_rec.currentIndex(), 0), len(res.recordings) - 1)]

    def refresh_views(self):
        res, rr = self.result, self._rr()
        cls = self.c_cls.currentData() or 'pedestrian'
        fig = self.fig.value() or 'recall'
        self.table.setRowCount(0)
        if rr is None:
            self.map.set_data(None, {}, 0, 1, empty='No result yet')
            self.sens_src.set_rows([])
            self.sens_cam.set_rows([])
            self.sens_det.set_rows([])
            self.refresh_image()
            return
        g = res.grid
        cells, rows = {}, []
        if fig == 'recall':
            src = rr.det.get(cls) or {}
            for rid, d in src.items():
                b = d['recall']
                txt = f'{_f(b["mean"])}\n{_f(b["p05"])}–{_f(b["p95"])}'
                tip = (f'{rid}: recall nominal {_f(d["nominal"])}, over builds {_f(b["mean"])} (90 % of builds '
                       f'{_f(b["p05"])}–{_f(b["p95"])}), {d["n_gt"]} objects')
                if rid != 'all':
                    cells[rid] = (b['mean'], txt, tip)
                rows.append((rid, d['n_gt'], d['nominal'], b['mean'], (b['p05'], b['p95']), None))
            self.map.set_data(g, cells, 0.0, 1.0, True, f'{cls} recall over builds (mean, 5–95 %)',
                              'No objects of this class (or the detector was off)')
        elif fig == 'bev':
            src = rr.bev.get(cls) or {}
            nb = rr.nominal['bev'].get(cls) or {}
            hi = max([d['err_m']['p95'] or 0.0 for d in src.values()] + [0.1])
            for rid, d in src.items():
                b = d['err_m']
                cells[rid] = (b['mean'], f'{_f(b["mean"])} m\np95 {_f(d["err_p95_m"])}',
                              f'{rid}: BEV error over builds {_f(b["mean"])} m (builds {_f(b["p05"])}–'
                              f'{_f(b["p95"])} m; every object p95 {_f(d["err_p95_m"])} m); nominal ATE '
                              f'{_f((nb.get(rid) or {}).get("ate_m"))} m; {d["n"]} object views')
                rows.append((rid, d['n'], (nb.get(rid) or {}).get('ate_m'), b['mean'], (b['p05'], b['p95']),
                             d['err_p95_m']))
            self.map.set_data(g, cells, 0.0, hi, False, f'{cls} BEV error added by the build (m; nominal column: '
                                                        f'the nominal lift against the ground truth)')
        elif fig == 'oov':
            src = rr.bev.get(cls) or {}
            for rid, d in src.items():
                b = d['oov']
                cells[rid] = (b['mean'], f'{_f((b["mean"] or 0) * 100, 1)} %',
                              f'{rid}: share of object views pushed out of the image, mean {_f(b["mean"], 3)}')
                rows.append((rid, d['n'], 0.0, b['mean'], (b['p05'], b['p95']), None))
            self.map.set_data(g, cells, 0.0, 0.2, False, f'{cls} out of view over builds')
        else:
            src = rr.ghost.get(cls) or {}
            for rid, d in src.items():
                b = d['rate']
                if rid != 'all':
                    cells[rid] = (b['mean'], f'{_f(b["mean"])}\nnom {_f(d["nominal"])}',
                                  f'{rid}: ghost rate over builds {_f(b["mean"])} (p95 {_f(b["p95"])}), nominal '
                                  f'{_f(d["nominal"])}; {d["n"]} objects seen twice')
                rows.append((rid, d['n'], d['nominal'], b['mean'], (b['p05'], b['p95']), None))
            self.map.set_data(g, cells, 0.0, 0.5, False, f'{cls} ghost rate after fusion gating (two lifts beyond the '
                                                         f'gate)')
        self.table.setRowCount(len(rows))
        for i, (rid, n, nom, mean, band, p95) in enumerate(rows):
            p = g.parse_id(rid)
            vals = [g.region_name(*p) if p else rid, str(n), _f(nom, 3), _f(mean, 3),
                    f'{_f(band[0], 3)}–{_f(band[1], 3)}', _f(p95, 2)]
            for c, v in enumerate(vals):
                it = QTableWidgetItem(v)
                if c:
                    it.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self.table.setItem(i, c, it)
        self._fill_sens(rr)
        cur = self.c_cam.currentText()
        self.c_cam.blockSignals(True)
        self.c_cam.clear()
        for t in rr.tags:
            self.c_cam.addItem(t)
        i = self.c_cam.findText(cur)
        self.c_cam.setCurrentIndex(i if i >= 0 else 0)
        self.c_cam.blockSignals(False)
        sc = res._scenes.get(res.recordings.index(rr))
        self.s_frame.blockSignals(True)
        self.s_frame.setRange(0, max(len(sc.det_slots) - 1, 0) if sc else 0)
        self.s_frame.blockSignals(False)
        self.refresh_image()

    def _fill_sens(self, rr):
        metric = {'recall': 'err:all', 'bev': 'err:all', 'oov': 'oov:all', 'ghost': 'ghost:all'}[self.fig.value() or
                                                                                              'recall']
        cls = self.c_cls.currentData() or 'pedestrian'
        if self.fig.value() == 'bev' and f'err:{cls}' in RP.SOBOL_METRICS:
            metric = f'err:{cls}'
        for w, kind, title in ((self.sens_src, 'by_source', 'By source'), (self.sens_cam, 'by_camera', 'By camera')):
            so = rr.sobol.get(kind)
            if not so or metric not in so['metrics']:
                w.set_rows([], title)
                continue
            j = so['metrics'].index(metric)
            rows = [(RP.SOURCE_LABEL.get(g, g), r['S'][j], r['ST'][j], (r['S_ci'][j] if r['S_ci'] else None))
                    for g, r in so['groups'].items()]
            rows.sort(key=lambda x: -(x[1] or 0))
            w.set_rows(rows, f'{title}: Sobol indices of {metric} (filled first order, outline total)',
                       f'n = {so["n"]} base samples, {so["evaluations"]} builds; bars: 90 % bootstrap intervals')
        if rr.oat:
            rows = [(RP.SOURCE_LABEL.get(s, s), v['all'].get('share'), None, None) for s, v in rr.oat.items()]
            rows.sort(key=lambda x: -(x[1] or 0))
            self.sens_det.set_rows(rows, 'Detection: share of the rig recall\'s variance, one source at a time',
                                   'One source on at a time: interactions are not counted (a screening, not Sobol).')
        else:
            self.sens_det.set_rows([], 'Detection: one source at a time')

    def refresh_image(self):
        res, rr = self.result, self._rr()
        if rr is None or not self.c_cam.count():
            self.img.clear()
            self.img_note.setText('')
            return
        which = self.c_build.value() or 'worst'
        out = RP.build_image(res, res.recordings.index(rr), which, self.c_cam.currentText(), self.s_frame.value())
        if out is None:
            self.img.clear()
            self.img_note.setText('No images of this build (the detector was off).')
            return
        img, text = out
        h, w = img.shape[:2]
        q = QImage(np.ascontiguousarray(img[:, :, ::-1]).data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
        pm = QPixmap.fromImage(q)
        sz = self.img.size()
        self.img.setPixmap(pm.scaled(max(sz.width(), 50), max(sz.height(), 50), Qt.AspectRatioMode.KeepAspectRatio,
                                     Qt.TransformationMode.SmoothTransformation))
        extra = f' · {rr.worst.get("text", "")}' if which == 'worst' and rr.worst else ''
        self.img_note.setText(text + extra + ' · green: ground truth, orange: detections, purple: no recorded content')

    def _fill_findings(self):
        res = self.result
        lines = []
        if res is None:
            self.find.setText('Run the study to see the figures per region and the dominant sources.')
            return
        for rr in res.recordings:
            lines.append(f'<b>{rr.label}</b>')
            lines += [f'· {h}' for h in RP.headline(rr, res.grid)]
            dom = RP.dominant(rr)
            if dom:
                for k, d in dom.items():
                    lines.append(f'· BEV error driven most by {RP.SOURCE_LABEL.get(d["group"], d["group"])} '
                                 f'({k.replace("by_", "by ")}: S {_f(d["S"])}, total {_f(d["ST"])})')
            g = (rr.ghost.get('all') or {}).get('all')
            if g:
                lines.append(f'· Ghost rate {_f(g["rate"]["mean"], 3)} over builds (nominal {_f(g["nominal"], 3)})')
            tb = rr.z_subpx_m
            lines.append(f'· Translations up to {_f(tb.get("largest_translation_mm"), 1)} mm move image content less '
                         f'than a pixel beyond {_f(max(tb["per_camera_m"].values(), default=None), 1)} m')
        lines.append('<b>Sources</b>')
        for s in res.sources.values():
            lines.append(f'· {RP.SOURCE_LABEL[s.key]} ({"on" if s.key in res.settings.active() else "off"}): '
                         f'{s.origin}')
        if res.records:
            lines.append(f'<b>Perception records</b>: {len(res.records)} (vantage.perception_record/1, in the export)')
        lines += [f'· {n}' for n in res.notes]
        lines.append(f'<span style="color:{THEME.tok.get("muted", "#888")}">{RP.ASSUMPTIONS}</span>')
        self.find.setText('<br>'.join(lines))

    def store_records(self):
        if self.result is None or not self.result.records:
            return None
        info = RP.store_records(self.result)
        self.ctx.msg(f'{info["stored"]} perception record(s): {info["where"]}', 6000)
        return info

    def export(self, folder=None):
        res = self.result
        if res is None:
            return []
        if folder is None:
            folder = QFileDialog.getExistingDirectory(self, 'Export the robust perception study into')
            if not folder:
                return []
        from ...core import provenance as PV
        paths = PV.stamp_paths(RP.export(res, self.ctx.store.project, folder), self.ctx.store.project, folder)
        self.ctx.msg(f'Exported the robust perception study to {folder}', 5000)
        return paths
