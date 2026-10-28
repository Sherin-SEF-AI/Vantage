"""Simulate ▸ Detection quality: an off-the-shelf detector (YOLOX through OpenCV, downloaded on request) run on the
camera images of CARLA drives with ground truth, matched to the labelled objects: recall, precision and average
precision per camera, class and range, the detection range, recall against object height, the rig as a whole, rigs
compared (differences with intervals), the fisheye images against pinhole views cut from them, and every frame's
detections over its ground truth. The model download and the evaluation are background jobs with Cancel; nothing
runs by itself after an edit (the result is marked stale)."""
from __future__ import annotations

import glob
import json
import math
import os
import threading
import traceback

import numpy as np
from PyQt6.QtCore import QObject, QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QImage, QPainter, QPen
from PyQt6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QFileDialog, QGridLayout, QHBoxLayout,
                             QHeaderView, QLineEdit, QListWidget, QListWidgetItem, QProgressBar, QScrollArea, QSlider,
                             QSplitter, QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget)

from ... import workdir
from ...core import detection as DT
from ...core import detection_eval as DE
from ..charts import ChartBase
from ..theme import STATUS, THEME, mono, sans
from ..widgets import ROW_H, ScrubSpinBox, SegmentedControl, StatusLine, button, hline, label
from .recording_ws import Tile

JOB = 'detection'
JOB_DL = 'detection_download'
EMPTY = ('Download the detector, add a CARLA drive recorded with ground truth (Simulate ▸ CARLA, Ground truth on), '
         'then Run. Add more drives of other rig variants with the same scenario and seed to compare them.')
# (key, label, unit, decimals, step, tip)
FIELDS = (
    ('score', 'Score threshold', '', 2, 0.05, 'The operating point: recall and precision count the detections scoring '
                                              'at least this (the average precision uses every detection kept)'),
    ('iou', 'IoU', '', 2, 0.05, 'A detection matches an object when their boxes overlap at least this much (COCO\'s '
                                'AP50 uses 0.5)'),
    ('iou_small', 'IoU, small objects', '', 2, 0.05, 'The IoU for objects less tall than the next value: a box a few '
                                                     'pixels off loses most of its IoU on a small object'),
    ('small_px', 'Small below', 'px', 0, 1.0, 'Objects less tall than this use the small-object IoU'),
    ('min_px', 'Ignore below', 'px', 0, 1.0, 'Objects (and unmatched detections) less tall than this are ignore '
                                             'regions: neither misses nor false detections'),
    ('max_occluded', 'Ignore occluded over', '', 2, 0.05, 'Objects hidden more than this share are ignored'),
    ('max_truncated', 'Ignore truncated over', '', 2, 0.05, 'Objects more than this share outside the image are '
                                                            'ignored'),
    ('frame_step', 'Every n-th frame', '', 0, 1.0, 'Evaluate every n-th camera frame (quicker, fewer objects)'),
    ('bootstrap', 'Bootstrap', '', 0, 100.0, 'Resamples for the intervals (0: none)'),
    ('block_s', 'Block', 's', 2, 0.25, 'The bootstrap draws blocks of this many seconds (frames close in time see the '
                                       'same objects; 0: every frame on its own)'),
    ('confidence', 'Confidence', '', 3, 0.01, 'The intervals\' confidence level'),
)
VIEW_FIELDS = (
    ('view_fov_deg', 'View field', '°', 1, 5.0, 'Each pinhole view\'s horizontal field of view'),
    ('view_pitch_deg', 'View pitch', '°', 1, 1.0, 'The views look this far below the horizon (level, no roll)'),
    ('view_width', 'View width', 'px', 0, 32.0, 'Each view\'s width in pixels (its height: × 0.75)'),
)
CAM_COLS = (('Camera', ''), ('Range', 'Range bin: camera to the object\'s box centre'),
            ('Objects', 'Labelled objects that are not ignored'), ('Found', 'Objects detected at the operating point'),
            ('Recall', 'Found / objects, with its bootstrap interval'),
            ('Detections', 'Detections at the operating point (not ignored)'),
            ('Precision', 'Detections that match an object / detections, with its interval'),
            ('AP50', 'COCO\'s 101-point average precision at the matching IoU, with its interval'),
            ('AP', 'COCO\'s AP averaged over IoU 0.50:0.05:0.95'))
CLASS_COLS = (('Class', ''), ('Objects', ''), ('Recall', ''), ('Precision', ''), ('AP50', ''), ('AP', ''),
              ('Range', 'The farthest range bin with recall ≥ 50 % (bins with enough objects)'),
              ('H50', 'The object height (px) at which recall reaches 50 %'),
              ('Rig recall', 'Objects the whole rig detects (any camera), each counted once per instant'))
CMP_COLS = (('Recording', ''), ('Figure', 'Camera-level (every camera pooled) or the rig (an object once, any camera)'),
            ('Range', ''), ('Objects', 'Objects of the recording (the reference\'s in brackets)'),
            ('Reference', 'The first recording\'s value'), ('Value', ''), ('Difference', 'Value − reference, with '
                                                                                         'its bootstrap interval'),
            ('Paired', 'Same scenario, seed, town and duration: the same time blocks are drawn for both'))
VIEW_COLS = (('Class', ''), ('Range', ''), ('Objects', 'Objects not ignored in the fisheye image'),
             ('In a view', 'Of those, the ones some pinhole view sees'), ('Fisheye', 'Recall on the fisheye images'),
             ('Views', 'Recall on the pinhole views (found in any view)'), ('Difference', 'Views − fisheye, paired'))
TABS = ('Cameras', 'Classes', 'Height', 'Compare', 'Views', 'Frames', 'Notes')


def _table(cols, stretch=None):
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
    t.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
    hh = t.horizontalHeader()
    for c in range(len(cols)):
        hh.setSectionResizeMode(c, QHeaderView.ResizeMode.Stretch if c == stretch else
                                QHeaderView.ResizeMode.ResizeToContents)
    if stretch is None:
        hh.setStretchLastSection(True)
    return t


def _cell(t, r, c, text, num=False, tip=None, color=None, bold=False):
    it = QTableWidgetItem(text)
    if num:
        it.setFont(mono(12))
        it.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    if tip:
        it.setToolTip(tip)
    if color is not None:
        it.setForeground(color)
    if bold:
        f = it.font()
        f.setBold(True)
        it.setFont(f)
    t.setItem(r, c, it)
    return it


def _pc(v, ci=None, nd=0) -> str:
    if v is None:
        return '—'
    s = f'{v * 100:.{nd}f} %'
    if ci:
        s += f'  [{ci[0] * 100:.{nd}f}–{ci[1] * 100:.{nd}f}]'
    return s


def _dpc(v, ci=None) -> str:
    if v is None:
        return '—'
    s = f'{v * 100:+.1f} pt'
    if ci:
        s += f'  [{ci[0] * 100:+.1f}, {ci[1] * 100:+.1f}]'
    return s


def _g(v, nd=3):
    return '—' if v is None else f'{v:.{nd}f}'


def _qimage(bgr) -> QImage:
    rgb = np.ascontiguousarray(bgr[:, :, ::-1])
    h, w = rgb.shape[:2]
    return QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()


# ============================================================================ the chart
class RecallChart(ChartBase):
    """Recall (0–100 %) over categories (range or height bins), one line per series with its intervals as whiskers;
    hover gives the value, its interval and how many objects it rests on."""
    MARGIN = (44, 30, 12, 34)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(200)
        self.labels: list = []
        self.series: list = []             # [(name, QColor, [value|None], [ci|None], [n])]
        self.empty = 'Run to see recall against range.'
        self._pts = []

    def set_data(self, labels, series, title='', subtitle=''):
        self.labels, self.series = list(labels), list(series)
        self.title, self.subtitle = title, subtitle
        self.hover = None
        self.update()

    def _rect(self):
        l, t, r, b = self.MARGIN
        return QRectF(l, t, max(self.width() - l - r, 10), max(self.height() - t - b, 10))

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.paint_title(p)
        r = self._rect()
        if not self.labels or not self.series:
            p.setFont(sans(11))
            p.setPen(THEME.color('muted'))
            p.drawText(r, Qt.AlignmentFlag.AlignCenter, self.empty)
            return
        p.setFont(mono(10))
        for t in (0, 0.25, 0.5, 0.75, 1.0):
            y = r.bottom() - t * r.height()
            p.setPen(QPen(THEME.color('strong' if t == 0 else 'hair'), 1))
            p.drawLine(QPointF(r.left(), y), QPointF(r.right(), y))
            p.setPen(THEME.color('muted'))
            p.drawText(QRectF(0, y - 7, r.left() - 6, 14), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                       f'{t * 100:.0f}%')
        n = len(self.labels)
        X = lambda j: r.left() + (j + 0.5) / n * r.width()        # noqa: E731
        p.setFont(sans(10))
        for j, lab in enumerate(self.labels):
            p.setPen(THEME.color('muted'))
            p.drawText(QRectF(X(j) - r.width() / n / 2, r.bottom() + 4, r.width() / n, 14),
                       Qt.AlignmentFlag.AlignCenter, lab)
        self._pts = []
        ns = len(self.series)
        for si, (name, col, vals, cis, cnt) in enumerate(self.series):
            off = (si - (ns - 1) / 2) * 4.0
            pts = []
            for j, v in enumerate(vals):
                if v is None:
                    continue
                x, y = X(j) + off, r.bottom() - v * r.height()
                ci = cis[j] if j < len(cis) else None
                if ci:
                    p.setPen(QPen(col, 1))
                    p.drawLine(QPointF(x, r.bottom() - ci[0] * r.height()), QPointF(x, r.bottom() - ci[1] * r.height()))
                pts.append((j, QPointF(x, y)))
                self._pts.append((QPointF(x, y), si, j))
            p.setPen(QPen(col, 2))
            for (ja, a), (jb, b) in zip(pts, pts[1:]):
                if jb == ja + 1:                       # a bin without objects breaks the line
                    p.drawLine(a, b)
            for _j, q in pts:
                hov = self.hover == (si, int(round((q.x() - off - r.left()) / r.width() * n - 0.5)))
                p.setBrush(col.lighter(130) if hov else col)
                p.setPen(QPen(THEME.color('ink') if hov else col, 1))
                p.drawEllipse(q, 3.5 if hov else 3.0, 3.5 if hov else 3.0)
        # the key
        p.setFont(sans(10))
        x = r.left()
        for name, col, *_ in self.series:
            p.setPen(QPen(col, 2))
            p.drawLine(QPointF(x, self.height() - 8), QPointF(x + 14, self.height() - 8))
            p.setPen(THEME.color('ink2'))
            w = p.fontMetrics().horizontalAdvance(name)
            p.drawText(QPointF(x + 18, self.height() - 4), name)
            x += 30 + w
        self.paint_tip(p)

    def mouseMoveEvent(self, e):
        pos = e.position()
        best, bd = None, 12.0
        for q, si, j in self._pts:
            d = math.hypot(q.x() - pos.x(), q.y() - pos.y())
            if d < bd:
                best, bd = (q, si, j), d
        if best is None:
            if self.hover is not None:
                self.hover = None
                self.tip = None
                self.update()
            return
        q, si, j = best
        name, col, vals, cis, cnt = self.series[si]
        ci = cis[j] if j < len(cis) else None
        self.hover = (si, j)
        self.set_tip(q, _pc(vals[j], ci), f'{name} · {self.labels[j]}',
                     f'{cnt[j]} objects' if j < len(cnt) else '')
        self.update()


class FrameView(QWidget):
    """One frame with its detections drawn over its ground truth (core.detection_eval.overlay_image), fitted."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(320, 220)
        self.img: QImage | None = None
        self.caption = ''
        self.empty = 'Run to see the detections over the ground truth.'
        THEME.changed.connect(self._theme_changed)

    def _theme_changed(self, *_):
        self.update()

    def set_image(self, img: QImage | None, caption=''):
        self.img, self.caption = img, caption
        self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        p.fillRect(self.rect(), THEME.color('viewport'))
        p.setFont(sans(11))
        p.setPen(THEME.color('ink2'))
        p.drawText(QRectF(8, 2, self.width() - 16, 18), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                   self.caption)
        if self.img is None:
            p.setPen(THEME.color('muted'))
            p.drawText(QRectF(self.rect()), Qt.AlignmentFlag.AlignCenter, self.empty)
            return
        top = 22.0
        W, H = self.width(), self.height() - top - 4
        s = min(W / max(self.img.width(), 1), H / max(self.img.height(), 1))
        w, h = self.img.width() * s, self.img.height() * s
        p.drawImage(QRectF((W - w) / 2, top + (H - h) / 2, w, h), self.img)


class _Sig(QObject):
    done = pyqtSignal(int, object)
    failed = pyqtSignal(int, str)
    progress = pyqtSignal(int, float, str)


# ============================================================================ the page
class DetectionPage(QWidget):
    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        self.result: DE.DetectionEvalResult | None = None
        self.detector = None                      # a callable image -> detections (tests); None: the chosen model
        self.busy = False
        self.dl_busy = False
        self.status_text = ''
        self._gen = 0
        self._dl_gen = 0
        self._for = None                          # (project, settings key) of the result
        self._dirty = True
        self._syncing = False
        self._info_cache = {}
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

    # ------------------------------------------------------------------ left: the detector, recordings, settings
    def _left(self):
        w = QWidget()
        w.setObjectName('panel')
        ll = QVBoxLayout(w)
        ll.setContentsMargins(8, 6, 8, 8)
        ll.setSpacing(4)
        top = QHBoxLayout()
        top.addWidget(label('Detection quality'))
        top.addStretch(1)
        top.addWidget(label('detector · ground truth · rigs', muted=True, small=True))
        ll.addLayout(top)
        ll.addWidget(hline())
        ll.addWidget(label('How well an off-the-shelf detector finds the objects in each camera\'s images of a CARLA '
                           'drive, against the drive\'s own ground truth: per camera, class and range, and rig '
                           'against rig on the same scene.', muted=True, small=True, wrap=True))
        # the detector
        ll.addWidget(label('Detector', ink2=True))
        row = QHBoxLayout()
        self.model = QComboBox()
        for m in DT.MODELS.values():
            self.model.addItem(f'{m.name} · {m.size / 1e6:.0f} MB · COCO AP {m.coco_ap:g}', m.id)
            self.model.setItemData(self.model.count() - 1, f'{m.name}: {m.params_m:g} M parameters, {m.gflops:g} '
                                                           f'GFLOPs at {m.input_px} px; {m.note}',
                                   Qt.ItemDataRole.ToolTipRole)
        self.model.setToolTip('The detector model (YOLOX, Apache-2.0; COCO AP is the publisher\'s figure)')
        self.model.currentIndexChanged.connect(self._model_picked)
        row.addWidget(self.model, 1)
        ll.addLayout(row)
        self.model_status = StatusLine('warning', '')
        ll.addWidget(self.model_status)
        lic = label(DT.LICENCE_TEXT, muted=True, small=True, wrap=True)
        lic.setToolTip(DT.MODELS[DT.DEFAULT_MODEL].licence_url)
        ll.addWidget(lic)
        bl = QHBoxLayout()
        self.b_download = button('Download', None, 'import', 'Download the model from YOLOX\'s GitHub release into '
                                                            'your data folder (checked against its SHA-256)')
        self.b_download.clicked.connect(lambda: self.download())
        bl.addWidget(self.b_download)
        self.b_import = button('Import file…', 'ghost', 'file', 'Use a model file you downloaded yourself (it must be '
                                                               'one of the published files, by SHA-256)')
        self.b_import.clicked.connect(lambda: self.import_model())
        bl.addWidget(self.b_import)
        bl.addStretch(1)
        ll.addLayout(bl)
        g = QGridLayout()
        g.setHorizontalSpacing(6)
        g.setVerticalSpacing(4)
        a = label('Input', ink2=True)
        a.setFixedWidth(118)
        g.addWidget(a, 0, 0)
        self.input_mode = SegmentedControl([('auto', 'Image size'), ('model', 'Model size')], 'auto')
        self.input_mode.setToolTip('Image size: each image at its own resolution (sides up to a multiple of 32, at '
                                   'most 1280 px); model size: the square the model was published at')
        self.input_mode.changed.connect(self._input_picked)
        g.addWidget(self.input_mode, 0, 1, 1, 2, Qt.AlignmentFlag.AlignLeft)
        a = label('Backend', ink2=True)
        g.addWidget(a, 1, 0)
        opts = [('cpu', 'CPU')] + ([('opencl', 'OpenCL')] if DT.opencl_available() else [])
        self.backend = SegmentedControl(opts, 'cpu')
        self.backend.setToolTip('OpenCV\'s DNN target (PyPI\'s OpenCV has no CUDA)')
        self.backend.changed.connect(lambda k: self._set(f'backend {k}', backend=k))
        g.addWidget(self.backend, 1, 1, 1, 2, Qt.AlignmentFlag.AlignLeft)
        ll.addLayout(g)
        ll.addWidget(hline())
        # recordings
        ll.addWidget(label('Recordings', ink2=True))
        self.recs = QListWidget()
        self.recs.setMinimumHeight(90)
        self.recs.setMaximumHeight(120)
        self.recs.setToolTip('CARLA drives recorded with ground truth; the first is the reference the others are '
                             'compared with')
        ll.addWidget(self.recs)
        rb = QHBoxLayout()
        b = button('Add…', None, 'folder', 'Add a CARLA run folder (one with ground_truth/index.json)')
        b.clicked.connect(lambda: self._pick_dir())
        rb.addWidget(b)
        b = button('Latest run', 'ghost', 'history', 'Add the newest CARLA run with ground truth in this project\'s '
                                                   'working folder')
        b.clicked.connect(lambda: self.add_latest())
        rb.addWidget(b)
        b = button('Reference', 'ghost', 'pin', 'Make the selected recording the reference (the first)')
        b.clicked.connect(lambda: self.make_reference())
        rb.addWidget(b)
        b = button('Remove', 'ghost', 'trash', 'Remove the selected recording from the list (nothing is deleted)')
        b.clicked.connect(lambda: self.remove_recording())
        rb.addWidget(b)
        rb.addStretch(1)
        ll.addLayout(rb)
        ll.addWidget(hline())
        # matching
        ll.addWidget(label('Matching and statistics', ink2=True))
        g = QGridLayout()
        g.setHorizontalSpacing(6)
        g.setVerticalSpacing(3)
        self.fields = {}
        for i, (key, text, unit, dec, step, tip) in enumerate(FIELDS + VIEW_FIELDS):
            f = ScrubSpinBox('', 'plain', dec, unit, step=step, width=110)
            f.setToolTip(tip)
            f.committed.connect(lambda v, k=key, t=text, u=unit: self._set(f'{t.lower()} {v:g}{u}', **{k: v}))
            a = label(text, ink2=True)
            a.setFixedWidth(140)
            a.setToolTip(tip)
            g.addWidget(a, i, 0)
            g.addWidget(f, i, 1)
            self.fields[key] = f
        n = len(FIELDS) + len(VIEW_FIELDS)
        self.texts = {}
        for i, (key, text, tip) in enumerate((
                ('range_edges', 'Range bins', 'Range bin edges in metres, comma separated (bins from 0 and beyond '
                                              'the last)'),
                ('height_edges', 'Height bins', 'Object height bin edges in pixels, comma separated'),
                ('view_yaws', 'View headings', 'The pinhole views\' headings from the camera\'s own, degrees, left '
                                               'positive, comma separated')), n):
            e = QLineEdit()
            e.setToolTip(tip)
            e.setFont(mono(12))
            e.editingFinished.connect(lambda k=key, ed=e, t=text: self._text_edited(k, ed, t))
            a = label(text, ink2=True)
            a.setToolTip(tip)
            g.addWidget(a, i, 0)
            g.addWidget(e, i, 1, 1, 2)
            self.texts[key] = e
        ll.addLayout(g)
        a = label('Images', ink2=True)
        ll.addWidget(a)
        self.views = SegmentedControl([('fisheye', 'Fisheye'), ('pinhole', 'Pinhole views'), ('both', 'Both')],
                                      'fisheye')
        self.views.setToolTip('Run the detector on the fisheye images as recorded, on pinhole views cut from them '
                              'through the recording\'s calibration, or on both (and compare them object by object)')
        self.views.changed.connect(lambda k: self._set(f'images {k}', views=k))
        ll.addWidget(self.views, 0, Qt.AlignmentFlag.AlignLeft)
        ll.addWidget(label('Classes', ink2=True))
        cg = QGridLayout()
        cg.setHorizontalSpacing(8)
        cg.setVerticalSpacing(2)
        self.class_box = {}
        for i, c in enumerate(DT.CLASSES):
            cb = QCheckBox(DT.CLASS_LABEL[c])
            if c not in DE.DEFAULT_CLASSES:
                cb.setToolTip('Off by default: COCO knows stop signs only, and the ground truth does not say which '
                              'sign is which, so there is no recall to measure (Notes)')
            cb.toggled.connect(lambda on, k=c: self._class_toggled(k, on))
            cg.addWidget(cb, i // 2, i % 2)
            self.class_box[c] = cb
        ll.addLayout(cg)
        ll.addWidget(label(DE.ASSUMPTIONS, muted=True, small=True, wrap=True))
        self.status = label('', muted=True, small=True, wrap=True)
        ll.addWidget(self.status)
        self.bar = QProgressBar()
        self.bar.setTextVisible(False)
        self.bar.setRange(0, 1000)
        self.bar.hide()
        ll.addWidget(self.bar)
        bl = QHBoxLayout()
        b_def = button('Defaults', 'ghost', None, 'Back to the default detector and matching settings (the '
                                                  'recordings stay)')
        b_def.clicked.connect(self._defaults)
        bl.addWidget(b_def)
        bl.addStretch(1)
        self.b_export = button('Export…', None, 'export', 'The evaluation as JSON and CSV, the recall chart as PNG, '
                                                          'the detections in COCO\'s results format and an overlay per '
                                                          'camera')
        self.b_export.clicked.connect(lambda: self.export())
        bl.addWidget(self.b_export)
        self.b_cancel = button('Cancel', 'ghost', None, 'Stop the run or the download')
        self.b_cancel.clicked.connect(lambda: self.cancel())
        bl.addWidget(self.b_cancel)
        self.b_run = button('Run', 'primary', 'play', 'Detect and evaluate (a background job)')
        self.b_run.clicked.connect(lambda: self.run())
        bl.addWidget(self.b_run)
        ll.addLayout(bl)
        ll.addStretch(1)
        sc = QScrollArea()
        sc.setWidgetResizable(True)
        sc.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        sc.setWidget(w)
        sc.setMinimumWidth(360)
        return sc

    # ------------------------------------------------------------------ right: tiles and tabs
    def _right(self):
        w = QWidget()
        w.setObjectName('panel')
        rl = QVBoxLayout(w)
        rl.setContentsMargins(10, 8, 10, 8)
        rl.setSpacing(8)
        pick = QHBoxLayout()
        self.rec_pick = QComboBox()
        self.rec_pick.setToolTip('The recording shown')
        self.rec_pick.currentIndexChanged.connect(lambda *_: self._show())
        pick.addWidget(self.rec_pick, 2)
        self.cls_pick = QComboBox()
        self.cls_pick.setToolTip('The class shown')
        self.cls_pick.currentIndexChanged.connect(lambda *_: self._show())
        pick.addWidget(self.cls_pick, 1)
        self.mode_pick = SegmentedControl([('fisheye', 'Fisheye'), ('pinhole', 'Views')], 'fisheye')
        self.mode_pick.setToolTip('Results on the fisheye images or on the pinhole views')
        self.mode_pick.changed.connect(lambda *_: self._show())
        pick.addWidget(self.mode_pick)
        rl.addLayout(pick)
        tiles = QHBoxLayout()
        tiles.setSpacing(8)
        self.tiles = {}
        for key, caption, tip in (
                ('objects', 'Objects', 'Labelled objects of the class over every camera frame, not ignored'),
                ('recall', 'Recall', 'Share of the objects detected at the operating point (every camera)'),
                ('precision', 'Precision', 'Share of the detections at the operating point that match an object'),
                ('ap', 'AP50', 'COCO\'s average precision at the matching IoU'),
                ('range', 'Detection range', 'The farthest range bin with recall ≥ 50 %'),
                ('rig', 'Rig recall', 'Objects the rig detects with any camera, each counted once per instant')):
            t = self.tiles[key] = Tile(caption, tip)
            tiles.addWidget(t, 1)
        rl.addLayout(tiles)
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        # cameras
        cw = QSplitter(Qt.Orientation.Vertical)
        cw.setHandleWidth(2)
        self.cam_chart = RecallChart()
        cw.addWidget(self.cam_chart)
        self.cam_table = _table(CAM_COLS)
        cw.addWidget(self.cam_table)
        cw.setSizes([260, 300])
        self.tabs.addTab(cw, 'Cameras')
        # classes
        self.class_table = _table(CLASS_COLS)
        self.tabs.addTab(self.class_table, 'Classes')
        # height
        hw = QSplitter(Qt.Orientation.Vertical)
        hw.setHandleWidth(2)
        self.height_chart = RecallChart()
        self.height_chart.empty = 'Run to see recall against the object\'s height in pixels.'
        hw.addWidget(self.height_chart)
        self.height_note = label('', muted=True, small=True, wrap=True)
        hw.addWidget(self.height_note)
        hw.setSizes([360, 60])
        self.tabs.addTab(hw, 'Height')
        # compare
        mw = QSplitter(Qt.Orientation.Vertical)
        mw.setHandleWidth(2)
        self.rig_chart = RecallChart()
        self.rig_chart.empty = 'Run to see each rig\'s recall against the distance from the vehicle.'
        mw.addWidget(self.rig_chart)
        self.cmp_table = _table(CMP_COLS)
        mw.addWidget(self.cmp_table)
        mw.setSizes([260, 300])
        self.tabs.addTab(mw, 'Compare')
        # views
        vw = QWidget()
        vl = QVBoxLayout(vw)
        vl.setContentsMargins(0, 6, 0, 0)
        self.views_note = label('', muted=True, small=True, wrap=True)
        vl.addWidget(self.views_note)
        self.view_table = _table(VIEW_COLS)
        vl.addWidget(self.view_table, 1)
        self.tabs.addTab(vw, 'Views')
        # frames
        fw = QWidget()
        fl = QVBoxLayout(fw)
        fl.setContentsMargins(0, 6, 0, 0)
        fl.setSpacing(6)
        ctl = QHBoxLayout()
        self.cam_pick = QComboBox()
        self.cam_pick.setToolTip('The camera shown')
        self.cam_pick.currentIndexChanged.connect(lambda *_: self._frames_changed())
        ctl.addWidget(self.cam_pick)
        self.frame_label = label('', mono_=True, small=True)
        ctl.addWidget(self.frame_label)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setToolTip('The camera\'s frames over the drive')
        self.slider.valueChanged.connect(lambda *_: self.show_frame())
        ctl.addWidget(self.slider, 1)
        b = button('Most objects', 'ghost', None, 'Jump to the frame with the most labelled objects')
        b.clicked.connect(lambda: self.show_frame(best=True))
        ctl.addWidget(b)
        fl.addLayout(ctl)
        self.frame_view = FrameView()
        fl.addWidget(self.frame_view, 1)
        self.legend = label('Objects: found (green), missed (orange), ignored (grey). Detections at the operating '
                            'point: matched (blue), false (magenta).', muted=True, small=True, wrap=True)
        fl.addWidget(self.legend)
        self.tabs.addTab(fw, 'Frames')
        # notes
        nw = QWidget()
        nl = QVBoxLayout(nw)
        nl.setContentsMargins(0, 6, 0, 0)
        self.note_host = QWidget()
        self.note_lay = QVBoxLayout(self.note_host)
        self.note_lay.setContentsMargins(0, 0, 0, 0)
        self.note_lay.setSpacing(4)
        self.note_lay.addStretch(1)
        sc = QScrollArea()
        sc.setWidgetResizable(True)
        sc.setWidget(self.note_host)
        nl.addWidget(sc, 1)
        self.notes: list = []
        self.tabs.addTab(nw, 'Notes')
        rl.addWidget(self.tabs, 1)
        return w

    # ------------------------------------------------------------------ settings
    @property
    def settings(self) -> DE.DetectionSettings:
        return DE.settings_of(self.ctx.store.project)

    def _store(self, st: DE.DetectionSettings, text) -> bool:
        new = DE.DetectionSettings.from_dict(st.to_dict()).to_dict()
        if new == self.settings.to_dict():
            self._sync_fields()                # a clamped value: show what is stored
            return False

        def fn(p):
            p.analysis.detection = new
        self.ctx.store.edit('analysis', fn, f'Detection quality · {text}')
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
        self._store(DE.DetectionSettings(recordings=list(st.recordings)), 'defaults')

    def _model_picked(self, i):
        mid = self.model.itemData(i)
        if mid and not self._syncing:
            self._set(f'model {DT.MODELS[mid].name}', model=mid)
        self._sync_model()

    def _input_picked(self, k):
        st = self.settings
        px = DT.MODELS[st.model].input_px if k == 'model' else 0
        self._set('input ' + ('the model\'s size' if px else 'the image\'s size'), input_px=px)

    def _text_edited(self, key, ed, text):
        if self._syncing:
            return
        st = DE.DetectionSettings.from_dict(dict(self.settings.to_dict(), **{key: ed.text()}))
        self._set(f'{text.lower()} {ed.text()}', **{key: getattr(st, key)})
        self._sync_fields()

    def _class_toggled(self, c, on):
        if self._syncing:
            return
        cl = [k for k in DT.CLASSES if (k == c and on) or (k != c and k in self.settings.classes)]
        self._set(f'{DT.CLASS_LABEL[c].lower()} {"on" if on else "off"}', classes=cl)

    # ------------------------------------------------------------------ recordings
    def add_recording(self, path) -> bool:
        """Add a run folder with ground truth to the list (undoable); False when it has none."""
        p = os.path.abspath(os.path.expanduser(str(path)))
        info = DE.recording_info(p)
        if not info['ok']:
            self.ctx.msg(info['text'], 6000)
            return False
        st = self.settings
        if p in st.recordings:
            return False
        if len(st.recordings) >= DE.MAX_RECORDINGS:
            self.ctx.msg(f'At most {DE.MAX_RECORDINGS} recordings.', 5000)
            return False
        self._info_cache[p] = info
        return self._set(f'recording {info.get("label") or os.path.basename(p)}', recordings=st.recordings + [p])

    def remove_recording(self, i=None) -> bool:
        st = self.settings
        i = self.recs.currentRow() if i is None else i
        if not 0 <= i < len(st.recordings):
            return False
        recs = list(st.recordings)
        gone = recs.pop(i)
        return self._set(f'removed {os.path.basename(gone)}', recordings=recs)

    def make_reference(self, i=None) -> bool:
        st = self.settings
        i = self.recs.currentRow() if i is None else i
        if not 0 < i < len(st.recordings):
            return False
        recs = list(st.recordings)
        recs.insert(0, recs.pop(i))
        return self._set(f'reference {os.path.basename(recs[0])}', recordings=recs)

    def latest_run(self) -> str | None:
        """The newest CARLA run with ground truth under <working folder>/<project>/carla/."""
        root = workdir.project_dir(self.ctx.store.project, workdir.get(self.ctx.settings), 'carla')
        found = glob.glob(os.path.join(root, '*', 'ground_truth', 'index.json'))
        if not found:
            return None
        return os.path.dirname(os.path.dirname(max(found, key=os.path.getmtime)))

    def add_latest(self) -> bool:
        p = self.latest_run()
        if p is None:
            self.ctx.msg('No CARLA run with ground truth in this project\'s working folder yet: run one on Simulate ▸ '
                         'CARLA with Ground truth on.', 6000)
            return False
        return self.add_recording(p)

    def _pick_dir(self):
        st = self.settings
        start = os.path.dirname(st.recordings[-1]) if st.recordings else \
            workdir.project_dir(self.ctx.store.project, workdir.get(self.ctx.settings), 'carla')
        d = QFileDialog.getExistingDirectory(self, 'Add a CARLA run with ground truth', start)
        if d:
            self.add_recording(d)

    def open_path(self, path) -> bool:
        """Open a run folder (other pages: the CARLA page after a run): added to the list and shown."""
        ok = self.add_recording(path) or os.path.abspath(path) in self.settings.recordings
        self.refresh()
        return ok

    def _rec_info(self, p):
        if p not in self._info_cache:
            self._info_cache[p] = DE.recording_info(p)
        return self._info_cache[p]

    # ------------------------------------------------------------------ the model
    def _sync_model(self):
        st = self.settings
        s = DT.status(st.model)
        sev = 'good' if s['verified'] else ('serious' if s['present'] else 'warning')
        self.model_status.set(sev, s['text'])
        self.model_status.setToolTip(s['path'])
        self.b_download.setEnabled(not s['verified'] and not self.dl_busy)
        self.b_download.setText('Downloading…' if self.dl_busy else 'Download')
        return s

    def download(self, model=None) -> bool:
        """Download the chosen model (a background job: progress in the job bar, Cancel stops it)."""
        if self.dl_busy:
            return False
        mid = model or self.settings.model
        self._dl_gen += 1
        self.dl_busy = True
        self.status_text = f'Downloading {DT.MODELS[mid].name}…'

        def work(progress, cancel):
            return DT.download(mid, progress=progress, cancel=cancel)
        ok = self._start(JOB_DL, f'Detector · download {DT.MODELS[mid].name}', work, '_dl_gen')
        if not ok:
            self.dl_busy = False
        self.refresh()
        return ok

    def import_model(self, path=None) -> bool:
        if path is None:
            path, _ = QFileDialog.getOpenFileName(self, 'Import a detector model', '', 'ONNX models (*.onnx)')
            if not path:
                return False
        try:
            mid = DT.import_file(path)
        except (OSError, ValueError) as e:
            self.ctx.msg(str(e), 7000)
            return False
        self.ctx.msg(f'{DT.MODELS[mid].name} imported into {DT.models_dir()}', 5000)
        if mid != self.settings.model:
            self._set(f'model {DT.MODELS[mid].name}', model=mid)
        self.refresh()
        return True

    # ------------------------------------------------------------------ state
    def _key(self, p=None):
        p = p or self.ctx.store.project
        return json.dumps(DE.settings_of(p).to_dict(), sort_keys=True)

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
        self._gen += 1                        # a run still going is for the old project
        self.result, self._for, self.busy = None, None, False
        self.status_text = ''
        self._info_cache = {}
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
        self._syncing = True
        try:
            self.model.setCurrentIndex(max(self.model.findData(st.model), 0))
            self.input_mode.set_value('model' if st.input_px else 'auto')
            if st.backend in self.backend.buttons:
                self.backend.set_value(st.backend)
            for k, f in self.fields.items():
                f.set_value(getattr(st, k))
            self.texts['range_edges'].setText(', '.join(f'{x:g}' for x in st.range_edges))
            self.texts['height_edges'].setText(', '.join(f'{x:g}' for x in st.height_edges))
            self.texts['view_yaws'].setText(', '.join(f'{x:g}' for x in st.view_yaws))
            self.views.set_value(st.views)
            for c, cb in self.class_box.items():
                cb.setChecked(c in st.classes)
            for k, _t, _u, _d, _s, _tip in VIEW_FIELDS:
                self.fields[k].setEnabled(st.views != 'fisheye')
            self.texts['view_yaws'].setEnabled(st.views != 'fisheye')
            self.recs.clear()
            for i, p in enumerate(st.recordings):
                info = self._rec_info(p)
                it = QListWidgetItem(('★ ' if i == 0 else '') + (info.get('label') or os.path.basename(p)) +
                                     ('' if info['ok'] else ' · not readable'))
                it.setToolTip(f'{p}\n{info["text"]}')
                if not info['ok']:
                    it.setForeground(QColor(STATUS['serious']))
                self.recs.addItem(it)
        finally:
            self._syncing = False
        self._sync_model()

    # ------------------------------------------------------------------ jobs
    def _start(self, kind, title, work, gen_attr):
        win = self.ctx.window
        gen = getattr(self, gen_attr)
        if win is not None and hasattr(win, 'run_job'):
            return win.run_job(kind, title, work, lambda out: self._thread_done(gen, (kind, out)),
                               lambda m: self._failed(gen, m))

        def thread():
            try:
                out = work(lambda f, t='': self.sig.progress.emit(gen, f, t), lambda: gen != getattr(self, gen_attr))
                self.sig.done.emit(gen, (kind, out))
            except RuntimeError as e:          # the page closed while computing, or a run failure
                try:
                    self.sig.failed.emit(gen, f'{e}')
                except RuntimeError:
                    pass
            except Exception as e:              # noqa: BLE001
                try:
                    self.sig.failed.emit(gen, f'{type(e).__name__}: {e}\n{traceback.format_exc()}')
                except RuntimeError:
                    pass
        threading.Thread(target=thread, daemon=True).start()
        return True

    def _progress(self, gen, f, text):
        if gen in (self._gen, self._dl_gen):
            self.bar.show()
            self.bar.setValue(int(f * 1000))
            if text:
                self.status.setText(text)

    def _thread_done(self, gen, payload):
        kind, out = payload
        if kind == JOB_DL:
            if gen != self._dl_gen:
                return
            self.dl_busy = False
            self.status_text = 'Download cancelled' if out is None else f'Downloaded {os.path.basename(out)}'
            self.ctx.msg(self.status_text, 5000)
            self._sync_model()
            self.refresh()
            return
        if gen != self._gen:
            return
        self.busy = False
        key, res = out if out is not None else (None, None)
        if res is None or key[0] is not self.ctx.store.project:
            self.status_text = 'Cancelled' if res is None else ''
            self.refresh()
            return
        self.result, self._for = res, key
        n_img = sum(v.get('images', 0) for v in res.timing.values() if isinstance(v, dict))
        self.status_text = f'{n_img} images in {res.seconds:.1f} s'
        self.ctx.store.project.add_log('detection', f'{len(res.recordings)} recording(s), {n_img} images, '
                                                    f'{res.detector.get("name", "")}')
        self.ctx.msg(f'Detection quality · {self.status_text}', 5000)
        self._fill_pickers()
        self.refresh()

    def _failed(self, gen, msg):
        if gen == self._dl_gen and self.dl_busy:
            self.dl_busy = False
            self.status_text = 'Download failed: ' + msg.splitlines()[0]
            self._sync_model()
        elif gen == self._gen and self.busy:
            self.busy = False
            self.status_text = 'Failed: ' + msg.splitlines()[0]
        self.refresh()

    def run(self) -> bool:
        if self.busy:
            return False
        st = self.settings
        if not st.recordings:
            self.status_text = 'Add a recording with ground truth first.'
            self.refresh()
            return False
        if self.detector is None:
            s = DT.status(st.model)
            if not s['verified']:
                self.status_text = s['text'] + ' Press Download.'
                self.refresh()
                return False
        p = self.ctx.store.project
        key = (p, self._key(p))
        det = self.detector
        self._gen += 1
        self.busy = True
        self.status_text = 'Running…'
        self.bar.setValue(0)

        def work(progress, cancel):
            return key, DE.run(None, settings=st, progress=progress, cancel=cancel, detector=det)
        ok = self._start(JOB, 'Detection quality', work, '_gen')
        if not ok:
            self.busy = False
            self.status_text = ''
        self.refresh()
        return ok

    def cancel(self):
        win = self.ctx.window
        for job in (JOB, JOB_DL):
            j = getattr(win, '_jobs', {}).get(job) if win is not None else None
            if j is not None:
                j['cancel'].set()
        if win is None or not hasattr(win, 'run_job'):
            if self.busy:
                self._gen += 1
                self.busy = False
                self.status_text = 'Cancelled'
            if self.dl_busy:
                self._dl_gen += 1
                self.dl_busy = False
                self.status_text = 'Download cancelled'
                self._sync_model()
            self.refresh()

    def export(self, folder=None) -> list:
        res = self.result
        if res is None:
            return []
        if folder is None:
            folder = QFileDialog.getExistingDirectory(self, 'Export the detection quality into')
            if not folder:
                return []
        from ...core import provenance as PV
        paths = PV.stamp_paths(DE.export(res, self.ctx.store.project, folder), self.ctx.store.project, folder)
        self.ctx.msg(f'Exported the detection quality to {folder}', 5000)
        return paths

    def summary(self) -> dict | None:
        return DE.summary(self.result) if self.result is not None else None

    # ------------------------------------------------------------------ showing the result
    def refresh(self):
        st = self.settings
        busy = self.busy or self.dl_busy
        self.b_run.setEnabled(not self.busy and bool(st.recordings))
        self.b_cancel.setVisible(busy)
        self.b_export.setEnabled(self.result is not None and not self.busy)
        self.bar.setVisible(busy)
        txt = self.status_text
        if self.stale and not self.busy:
            txt = (txt + ' · ' if txt else '') + 'The settings changed since the run: Run again.'
        if not st.recordings and self.result is None and not txt:
            txt = EMPTY
        self.status.setText(txt)
        self._show()

    def _fill_pickers(self):
        res = self.result
        for cb in (self.rec_pick, self.cls_pick, self.cam_pick):
            cb.blockSignals(True)
            cb.clear()
        if res is not None:
            for i, r in enumerate(res.recordings):
                self.rec_pick.addItem(r['label'], i)
            have = [c for c in DT.EVAL_CLASSES if any(r['class'] == c and r['n_gt'] for r in res.rows)] or \
                [c for c in DT.EVAL_CLASSES if any(r['class'] == c for r in res.rows)]
            for c in have:
                self.cls_pick.addItem(DT.CLASS_LABEL.get(c, c), c)
            v = self.cls_pick.findData('vehicle')
            self.cls_pick.setCurrentIndex(v if v >= 0 else 0)
            self.mode_pick.set_value(res.modes()[0])
            self.mode_pick.setEnabled(len(res.modes()) > 1)
        for cb in (self.rec_pick, self.cls_pick, self.cam_pick):
            cb.blockSignals(False)
        self._fill_cams()

    def _fill_cams(self):
        res = self.result
        self.cam_pick.blockSignals(True)
        self.cam_pick.clear()
        if res is not None:
            ri, mode = self._rec(), self._mode()
            for (r, m, cam) in sorted(k for k in res.frames if k[0] == ri and k[1] == mode):
                self.cam_pick.addItem(cam, cam)
        self.cam_pick.blockSignals(False)
        self._frames_changed()

    def _rec(self) -> int:
        return int(self.rec_pick.currentData() or 0)

    def _cls(self) -> str:
        return self.cls_pick.currentData() or 'vehicle'

    def _mode(self) -> str:
        res = self.result
        m = self.mode_pick.value() if hasattr(self.mode_pick, 'value') else 'fisheye'
        return m if res is None or m in res.modes() else res.modes()[0]

    def _show(self):
        res = self.result
        if res is None:
            for t in self.tiles.values():
                t.set('—')
            for tb in (self.cam_table, self.class_table, self.cmp_table, self.view_table):
                tb.setRowCount(0)
            for ch in (self.cam_chart, self.height_chart, self.rig_chart):
                ch.set_data([], [])
            self.frame_view.set_image(None)
            self.height_note.setText('')
            self.views_note.setText('')
            self._fill_notes(None)
            return
        ri, cls, mode = self._rec(), self._cls(), self._mode()
        if self.cam_pick.count() and self.cam_pick.itemData(0) not in [k[2] for k in res.frames
                                                                         if k[0] == ri and k[1] == mode]:
            self._fill_cams()
        self._fill_tiles(res, ri, cls, mode)
        self._fill_cameras(res, ri, cls, mode)
        self._fill_classes(res, ri, mode)
        self._fill_height(res, cls, mode)
        self._fill_compare(res, cls, mode)
        self._fill_views(res, ri)
        self._fill_notes(res)

    def _fill_tiles(self, res, ri, cls, mode):
        r = res.row(ri, mode, 'all', cls, 'all')
        rg = next((x for x in res.ranges if x['rec'] == ri and x['mode'] == mode and x['camera'] == 'all' and
                   x['class'] == cls), None)
        rig = next((x for x in res.rig if x['rec'] == ri and x['class'] == cls and x['distance_bin'] == 'all'), None)
        if r is None:
            for t in self.tiles.values():
                t.set('—')
            return
        self.tiles['objects'].set(f'{r["n_gt"]:,}', '', f'{r["detected"]:,} found')
        ci = r['recall_ci']
        self.tiles['recall'].set(_pc(r['recall']) if r['recall'] is not None else '—', '',
                                 f'{ci[0] * 100:.0f}–{ci[1] * 100:.0f} %' if ci else '')
        ci = r['precision_ci']
        self.tiles['precision'].set(_pc(r['precision']) if r['precision'] is not None else '—', '',
                                    f'{r["fp"]} false · ' + (f'{ci[0] * 100:.0f}–{ci[1] * 100:.0f} %' if ci else ''))
        ci = r.get('ap50_ci')
        self.tiles['ap'].set(_g(r.get('ap50'), 3), '', f'{ci[0]:.3f}–{ci[1]:.3f}' if ci else '')
        if rg and rg['range_bin']:
            val = f'> {res.settings.range_edges[-1]:g}' if rg['beyond_last_edge'] else f'{rg["range_m"]:g}'
            self.tiles['range'].set(val, 'm', rg['range_bin'])
        else:
            self.tiles['range'].set('—', '', (rg or {}).get('text', '')[:40])
        self.tiles['rig'].set(_pc(rig['recall']) if rig and rig['recall'] is not None else '—', '',
                              f'{rig["n"]} objects' if rig else '')

    def _fill_cameras(self, res, ri, cls, mode):
        labels = DE.bin_labels(res.settings.range_edges)
        rows = [r for r in res.rows if r['rec'] == ri and r['mode'] == mode and r['class'] == cls]
        cams = []
        for r in rows:
            if r['camera'] not in cams:
                cams.append(r['camera'])
        series = []
        for k, cam in enumerate(cams):
            by = {r['range_bin']: r for r in rows if r['camera'] == cam}
            col = THEME.color('ink') if cam == 'all' else THEME.cam(k - 1)
            series.append(('every camera' if cam == 'all' else cam, col,
                           [by[lab]['recall'] if lab in by else None for lab in labels],
                           [by[lab]['recall_ci'] if lab in by else None for lab in labels],
                           [by[lab]['n_gt'] if lab in by else 0 for lab in labels]))
        self.cam_chart.set_data(labels, series, f'Recall against range · {DT.CLASS_LABEL.get(cls, cls)}',
                                f'score ≥ {res.settings.score:g}')
        t = self.cam_table
        t.setRowCount(len(rows))
        for i, r in enumerate(rows):
            _cell(t, i, 0, r['camera'], bold=r['range_bin'] == 'all')
            _cell(t, i, 1, r['range_bin'])
            _cell(t, i, 2, f'{r["n_gt"]:,}', True)
            _cell(t, i, 3, f'{r["detected"]:,}', True)
            _cell(t, i, 4, _pc(r['recall'], r['recall_ci']), True)
            _cell(t, i, 5, f'{r["n_det"]:,}', True)
            _cell(t, i, 6, _pc(r['precision'], r['precision_ci']), True)
            ci = r.get('ap50_ci')
            _cell(t, i, 7, _g(r.get('ap50')) + (f'  [{ci[0]:.3f}–{ci[1]:.3f}]' if ci else ''), True)
            _cell(t, i, 8, _g(r.get('ap')), True)

    def _fill_classes(self, res, ri, mode):
        rows = [r for r in res.rows if r['rec'] == ri and r['mode'] == mode and r['camera'] == 'all' and
                r['range_bin'] == 'all']
        t = self.class_table
        t.setRowCount(len(rows))
        for i, r in enumerate(rows):
            rg = next((x for x in res.ranges if x['rec'] == ri and x['mode'] == mode and x['camera'] == 'all' and
                       x['class'] == r['class']), {})
            hh = next((x for x in res.heights if x['rec'] == ri and x['mode'] == mode and x['class'] == r['class']), {})
            rig = next((x for x in res.rig if x['rec'] == ri and x['class'] == r['class'] and
                        x['distance_bin'] == 'all'), {})
            _cell(t, i, 0, DT.CLASS_LABEL.get(r['class'], r['class']), bold=r['class'] in DT.GROUPS)
            _cell(t, i, 1, f'{r["n_gt"]:,}', True)
            _cell(t, i, 2, _pc(r['recall'], r['recall_ci']), True)
            _cell(t, i, 3, _pc(r['precision'], r['precision_ci']), True)
            _cell(t, i, 4, _g(r.get('ap50')), True)
            _cell(t, i, 5, _g(r.get('ap')), True)
            _cell(t, i, 6, rg.get('range_bin') or '—', tip=rg.get('text'))
            _cell(t, i, 7, f'{hh["h50_px"]:.0f} px' if hh.get('h50_px') else '—', True)
            _cell(t, i, 8, _pc(rig.get('recall'), rig.get('recall_ci')), True)

    def _fill_height(self, res, cls, mode):
        series, labels, notes = [], DE.height_labels(res.settings.height_edges), []
        for ri, info in enumerate(res.recordings):
            h = next((x for x in res.heights if x['rec'] == ri and x['mode'] == mode and x['class'] == cls), None)
            if h is None:
                continue
            series.append((info['label'], THEME.cam(ri), [b['recall'] for b in h['bins']],
                           [b['recall_ci'] for b in h['bins']], [b['n'] for b in h['bins']]))
            if h['h50_px']:
                notes.append(f'{info["label"]}: 50 % recall at about {h["h50_px"]:.0f} px')
        self.height_chart.set_data(labels, series, f'Recall against object height · {DT.CLASS_LABEL.get(cls, cls)}',
                                   'every camera; occluded and truncated objects left out')
        self.height_note.setText('; '.join(notes) or 'Recall does not reach 50 % in any height bin with objects.')

    def _fill_compare(self, res, cls, mode):
        labels = DE.bin_labels(res.settings.range_edges)
        series = []
        for ri, info in enumerate(res.recordings):
            by = {x['distance_bin']: x for x in res.rig if x['rec'] == ri and x['class'] == cls}
            series.append((info['label'], THEME.cam(ri), [by[lab]['recall'] if lab in by else None for lab in labels],
                           [by[lab]['recall_ci'] if lab in by else None for lab in labels],
                           [by[lab]['n'] if lab in by else 0 for lab in labels]))
        self.rig_chart.set_data(labels, series, f'Rig recall against distance · {DT.CLASS_LABEL.get(cls, cls)}',
                                'an object once per instant, any camera')
        rows = [c for c in res.comparisons if c['class'] == cls and c['mode'] in (mode, 'rig')]
        t = self.cmp_table
        t.setRowCount(len(rows))
        for i, c in enumerate(rows):
            d, ci = c.get('recall_difference'), c.get('recall_difference_ci')
            col = None
            if ci:
                col = QColor(STATUS['good']) if ci[0] > 0 else QColor(STATUS['serious']) if ci[1] < 0 else None
            _cell(t, i, 0, c['recording'])
            _cell(t, i, 1, 'Rig recall' if c['mode'] == 'rig' else 'Recall (cameras)')
            _cell(t, i, 2, c['range_bin'])
            _cell(t, i, 3, f'{c["n_gt"]:,} ({c["n_gt_reference"]:,})', True)
            _cell(t, i, 4, _pc(c.get('recall_reference')), True)
            _cell(t, i, 5, _pc(c.get('recall')), True)
            _cell(t, i, 6, _dpc(d, ci), True, color=col,
                  tip='Precision ' + _dpc(c.get('precision_difference'), c.get('precision_difference_ci')) +
                      ' · AP50 ' + (f'{c["ap50_difference"]:+.3f}' if c.get('ap50_difference') is not None else '—')
                  if c['mode'] != 'rig' else c.get('note'))
            _cell(t, i, 7, 'yes' if c['paired'] else 'no')

    def _fill_views(self, res, ri):
        rows = [v for v in res.views if v['rec'] == ri]
        self.views_note.setText(
            ('Object by object: found on the fisheye image, found in any pinhole view (' +
             ', '.join(f'{y:+g}°' for y in res.settings.view_yaws) + f' from the camera\'s heading, '
             f'{res.settings.view_fov_deg:g}° wide, {res.settings.view_pitch_deg:g}° down).')
            if rows else 'Choose Images ▸ Both and Run to compare the fisheye images with pinhole views cut from them.')
        t = self.view_table
        t.setRowCount(len(rows))
        for i, v in enumerate(rows):
            _cell(t, i, 0, DT.CLASS_LABEL.get(v['class'], v['class']), bold=v['range_bin'] == 'all')
            _cell(t, i, 1, v['range_bin'])
            _cell(t, i, 2, f'{v["n"]:,}', True)
            _cell(t, i, 3, f'{v["in_a_view"]:,}', True)
            _cell(t, i, 4, _pc(v['recall_fisheye'], v['recall_fisheye_ci']), True)
            _cell(t, i, 5, _pc(v['recall_views'], v['recall_views_ci']), True)
            _cell(t, i, 6, _dpc(v['difference'], v['difference_ci']), True)

    def _frames_changed(self):
        res = self.result
        n = len(res.frames.get((self._rec(), self._mode(), self.cam_pick.currentData()), [])) if res else 0
        self.slider.blockSignals(True)
        self.slider.setRange(0, max(n - 1, 0))
        self.slider.setValue(0)
        self.slider.blockSignals(False)
        self.show_frame(best=True)

    def show_frame(self, k=None, best=False):
        res = self.result
        cam = self.cam_pick.currentData()
        if res is None or cam is None:
            self.frame_view.set_image(None)
            self.frame_label.setText('')
            return False
        ri, mode = self._rec(), self._mode()
        frs = res.frames.get((ri, mode, cam), [])
        if not frs:
            self.frame_view.set_image(None)
            return False
        if best:
            k = DE.best_frame(res, ri, mode, cam)
            self.slider.blockSignals(True)
            self.slider.setValue(k)
            self.slider.blockSignals(False)
        k = self.slider.value() if k is None else int(k)
        k = min(max(k, 0), len(frs) - 1)
        img = DE.overlay_image(res, ri, mode, cam, k)
        fr = frs[k]
        gt = fr['gt']
        ok = (gt[:, 5] == 0) & (gt[:, 4] >= 0) if len(gt) else np.zeros(0, bool)
        found = int(np.count_nonzero(ok & (gt[:, 6] >= res.settings.score))) if len(gt) else 0
        dets = fr['det']
        fp = int(np.count_nonzero((dets[:, 5] >= res.settings.score) & (dets[:, 6] == 0))) if len(dets) else 0
        view = '' if fr['view'] < 0 else f' · view {fr["image"].split("#")[-1]}'
        t0 = frs[0]['t']
        self.frame_label.setText(f'{k + 1}/{len(frs)} · {(fr["t"] - t0) / 1e6:.2f} s')
        self.frame_view.set_image(_qimage(img) if img is not None else None,
                                  f'{cam}{view} · {int(ok.sum())} objects, {found} found, {fp} false detections')
        return img is not None

    def _fill_notes(self, res):
        for w in self.notes:
            w.hide()
            w.deleteLater()
        self.notes = []
        lines = []
        if res is not None:
            d = res.detector
            how = f' through OpenCV {d["opencv"]} ({d.get("backend") or "cpu"})' if d.get('opencv') else ''
            lines.append(('good', f'{d.get("name", "")}{how}; ' +
                          '; '.join(f'{m}: {v["images"]} images, {v["ms_mean"]:.0f} ms each'
                                    for m, v in res.timing.items() if isinstance(v, dict)) +
                          f'; {res.seconds:.1f} s in all.'))
            lines += [('warning', n) for n in res.notes]
        lines += [('', DE.ASSUMPTIONS), ('', DE.LIMITS_TEXT), ('', DT.LICENCE_TEXT)]
        for sev, text in lines:
            w = StatusLine(sev or 'good', text)
            if not sev:
                w.icon.clear()
            self.note_lay.insertWidget(self.note_lay.count() - 1, w)
            self.notes.append(w)
