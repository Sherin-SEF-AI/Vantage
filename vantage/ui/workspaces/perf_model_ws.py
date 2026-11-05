"""Analysis ▸ Learned detection model: the perception records measured so far (CARLA perception in the loop, re-shot
drives, tolerance runs; imported from other teams), a model trained on them that predicts every rig's detection per
class and region with an uncertainty (core.perf_model: a Gaussian process on the logit of recall over physics
features of the rig), how well its intervals held up in cross-validation, the current variant's prediction as a polar
map of the regions around the car (recall and its std), what moved the prediction against the reference variant, and
which rigs to evaluate for real next (active learning). Training, cross-validation and a campaign are background jobs
with Cancel; nothing trains by itself after an edit (the prediction is marked stale)."""
from __future__ import annotations

import math
import os
import threading
import traceback

from PyQt6.QtCore import QObject, QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QFileDialog, QGridLayout, QHBoxLayout,
                             QHeaderView, QProgressBar, QScrollArea, QSpinBox, QSplitter, QTableWidget,
                             QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget)

from ...core import perf_model as PM
from ...core import regions as RG
from ..charts import ChartBase
from ..theme import SEQ, STATUS, THEME, mono, sans
from ..widgets import ROW_H, ScrubSpinBox, SegmentedControl, StatusLine, button, hline, label
from .recording_ws import Tile

JOB = 'perf_model'
JOB_AL = 'perf_model_campaign'
EMPTY = ('No perception records yet. Run Simulate ▸ Perception in the loop on a few rig variants (each run adds a '
         'record), or import records another team exported, then Train.')
STORE_COLS = (('Source', ''), ('Scene set', 'Town · scenario · seed'), ('Rig', 'The rig\'s label or hash'),
              ('Cameras', ''), ('Detector', ''), ('Objects', 'Ground-truth objects counted, all classes and regions'))
CV_COLS = (('Class', ''), ('Check', 'Leave one scene set out, or leave rigs out (whole rigs per fold)'),
           ('Method', 'The Gaussian process, or a baseline with the same noise model'),
           ('Rows', 'Held-out class × region rows'), ('68 %', 'Share of held-out rows inside the 68 % interval'),
           ('95 %', 'Share inside the 95 % interval'), ('RMSE', 'Recall error on held-out rows'),
           ('NLPD', 'Mean negative log predictive density (lower is better)'))
AL_COLS = (('#', 'Pick order'), ('Variant', ''), ('Objective', 'Predicted weighted recall'), ('± std', ''),
           ('EI', 'Expected improvement over the best evaluated rig'), ('Why', ''))
METHOD_TEXT = {'gp': 'Gaussian process', 'logistic': 'Logistic in pixels', 'constant': 'Constant'}


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
    return it


def _fmt(v, nd=3):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return '—'
    return f'{x:.{nd}f}' if math.isfinite(x) else '—'


class _Sig(QObject):
    done = pyqtSignal(int, object)
    failed = pyqtSignal(int, str)
    progress = pyqtSignal(int, float, str)


# ============================================================================ the polar region map
class RegionMap(ChartBase):
    """The regions around the car as a polar map: sectors around, range bins outwards (equal ring widths, labelled),
    each region filled by its value (recall 0..1, or its std 0..vmax) on the app's sequential ramp; hover a region for
    its name, value and interval."""

    def __init__(self, title='', vmax=1.0):
        super().__init__()
        self.title = title
        self.vmax = vmax
        self.grid = RG.RegionGrid()
        self.values = {}                   # region id -> value
        self.extra = {}                    # region id -> tooltip lines
        self.empty = 'Train the model to see its prediction.'
        self.setMinimumSize(260, 280)
        self._cells = []                   # (path, rid)

    def set(self, grid, values, extra=None, vmax=None):
        self.grid = grid
        self.values = dict(values or {})
        self.extra = dict(extra or {})
        if vmax is not None:
            self.vmax = vmax
        self.tip = None
        self.update()

    def colour(self, v):
        if v is None or not math.isfinite(v):
            return QColor(0, 0, 0, 0)
        t = min(max(v / max(self.vmax, 1e-9), 0.0), 1.0)
        return QColor(SEQ[int(round(t * (len(SEQ) - 1)))])

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.paint_title(p, f'0 … {self.vmax:g}')
        top = 24
        side = min(self.width(), self.height() - top - 18)
        cx, cy = self.width() / 2, top + side / 2
        R = side / 2 - 18
        g = self.grid
        nr = g.n_ranges
        r_in = R * 0.12
        dr = (R - r_in) / nr
        self._cells = []
        if not self.values:
            p.setPen(THEME.color('muted'))
            p.setFont(sans(11))
            p.drawText(QRectF(0, top, self.width(), side), Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap,
                       self.empty)
        w = 360.0 / g.sectors
        for s, r, rid, _ in g.regions():
            a0 = s * w - w / 2              # degrees from straight ahead, positive left; up on the screen is ahead
            path = QPainterPath()
            r0, r1 = r_in + r * dr, r_in + (r + 1) * dr
            pts = []
            for k in range(9):
                a = math.radians(a0 + w * k / 8)
                pts.append(QPointF(cx - r1 * math.sin(a), cy - r1 * math.cos(a)))
            for k in range(8, -1, -1):
                a = math.radians(a0 + w * k / 8)
                pts.append(QPointF(cx - r0 * math.sin(a), cy - r0 * math.cos(a)))
            path.moveTo(pts[0])
            for q in pts[1:]:
                path.lineTo(q)
            path.closeSubpath()
            v = self.values.get(rid)
            p.setPen(QPen(THEME.color('hair'), 1))
            p.setBrush(self.colour(v) if v is not None else THEME.color('raised'))
            if self.hover == rid:
                p.setPen(QPen(THEME.color('ink'), 2))
            p.drawPath(path)
            self._cells.append((path, rid))
        # the car and the ring labels
        p.setPen(QPen(THEME.color('strong'), 1))
        p.setBrush(THEME.color('body'))
        p.drawRoundedRect(QRectF(cx - r_in * 0.45, cy - r_in * 0.8, r_in * 0.9, r_in * 1.6), 3, 3)
        p.setFont(mono(9))
        p.setPen(THEME.color('muted'))
        edges = [0.0] + list(g.range_edges)
        for k, e_m in enumerate(edges[1:], 1):
            p.drawText(QPointF(cx + 3, cy - (r_in + k * dr) + 10), f'{e_m:g}')
        p.drawText(QRectF(0, cy - R - 16, self.width(), 14), Qt.AlignmentFlag.AlignHCenter, 'ahead')
        self.paint_tip(p)

    def mouseMoveEvent(self, e):
        pos = e.position()
        hit = next((rid for path, rid in self._cells if path.contains(pos)), None)
        if hit != self.hover:
            self.hover = hit
            if hit is None:
                self.tip = None
            else:
                sr = self.grid.parse_id(hit)
                v = self.values.get(hit)
                self.set_tip(pos, _fmt(v) if v is not None else 'no prediction',
                             self.grid.region_name(*sr) if sr else hit, *self.extra.get(hit, []))
            self.update()

    def leaveEvent(self, e):
        self.hover, self.tip = None, None
        self.update()


class CalibrationChart(ChartBase):
    """Empirical against nominal interval coverage (pooled over the trained classes): the Gaussian process and the
    two baselines; the diagonal is perfect calibration."""

    COLS = {'gp': '#3987e5', 'logistic': '#0ca30c', 'constant': '#898781'}

    def __init__(self, by='scene'):
        super().__init__()
        self.by = by
        self.title = 'Leave one scene set out' if by == 'scene' else 'Leave rigs out'
        self.curves = {}
        self.empty = 'No cross-validation yet.'
        self.setMinimumSize(240, 240)

    def set(self, model):
        self.curves = {}
        if model is not None:
            for meth in ('gp', 'logistic', 'constant'):
                acc = {}
                for (c, met), cvs in model.cv.items():
                    cv = cvs.get(self.by) or {}
                    if met != 'recall' or not cv.get('ok'):
                        continue
                    for lv, emp in cv[meth]['curve']:
                        a = acc.setdefault(lv, [0.0, 0])
                        a[0] += emp * cv[meth]['n']
                        a[1] += cv[meth]['n']
                if acc:
                    self.curves[meth] = [(lv, s / n) for lv, (s, n) in sorted(acc.items()) if n]
        self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.paint_title(p)
        r = QRectF(34, 26, max(self.width() - 46, 10), max(self.height() - 70, 10))
        p.setPen(QPen(THEME.color('hair'), 1))
        p.drawRect(r)
        p.drawLine(r.bottomLeft(), r.topRight())
        p.setFont(mono(9))
        p.setPen(THEME.color('muted'))
        for t in (0.0, 0.5, 1.0):
            p.drawText(QRectF(0, r.bottom() - t * r.height() - 7, r.left() - 4, 14),
                       Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, f'{t:g}')
            p.drawText(QRectF(r.left() + t * r.width() - 14, r.bottom() + 2, 28, 14), Qt.AlignmentFlag.AlignHCenter,
                       f'{t:g}')
        p.setFont(sans(10))
        p.drawText(QRectF(r.left(), r.bottom() + 14, r.width(), 14), Qt.AlignmentFlag.AlignHCenter,
                   'nominal coverage → empirical ↑')
        if not self.curves:
            p.drawText(r, Qt.AlignmentFlag.AlignCenter, self.empty)
            return
        for i, (meth, pts) in enumerate(self.curves.items()):
            col = QColor(self.COLS[meth])
            p.setPen(QPen(col, 2))
            q = [QPointF(r.left() + lv * r.width(), r.bottom() - v * r.height()) for lv, v in pts]
            for a, b in zip(q, q[1:]):
                p.drawLine(a, b)
            p.setBrush(col)
            for a in q:
                p.drawEllipse(a, 2.5, 2.5)
            p.setPen(col)
            p.drawText(QPointF(r.left() + 6, r.top() + 14 + 13 * i), METHOD_TEXT[meth])


# ============================================================================ the page
class PerfModelPage(QWidget):
    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        self.model: PM.Model | None = None
        self.result: PM.PerfModelResult | None = None
        self.proposals = []
        self.campaign_result = None
        self.evaluator = None                  # tests: evaluator(candidate, progress, cancel) -> records
        self.busy = False
        self.status_text = ''
        self.cls = 'pedestrian'
        self._gen = 0
        self._stale = True
        self._syncing = False
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
        split.setSizes([400, 1000])
        lay.addWidget(split)
        ctx.store.changed.connect(self._on_change)
        ctx.store.project_replaced.connect(self._on_replaced)
        THEME.changed.connect(lambda *_: self.update())
        self.model = PM.load_model()
        self._sync_fields()
        self.refresh()

    # ------------------------------------------------------------------ settings
    @property
    def settings(self) -> PM.PerfModelSettings:
        return PM.settings_of(self.ctx.store.project)

    def _engine(self):
        an = getattr(self.ctx, 'analysis', None)
        return getattr(an, 'engine', None)

    def store(self) -> PM.Store:
        return PM.store_for(self.ctx.store.project)

    def _set(self, text, **kw):
        if self._syncing:
            return

        def fn(p):
            d = dict(PM.settings_dict(p))
            d.update(kw)
            p.analysis.perf_model = PM.PerfModelSettings.from_dict(d).to_dict()
        self.ctx.store.edit('analysis', fn, f'Learned model: {text}')

    # ------------------------------------------------------------------ left: store, training
    def _left(self):
        sc = QScrollArea()
        sc.setWidgetResizable(True)
        w = QWidget()
        w.setObjectName('panel')
        ll = QVBoxLayout(w)
        ll.setContentsMargins(8, 6, 8, 8)
        ll.setSpacing(4)
        top = QHBoxLayout()
        top.addWidget(label('Learned detection model'))
        top.addStretch(1)
        top.addWidget(label('records · model · active learning', muted=True, small=True))
        ll.addLayout(top)
        ll.addWidget(hline())
        ll.addWidget(label('Predicts each rig\'s detection per class and region from its geometry, with an '
                           'uncertainty checked by cross-validation, trained on the perception records measured so '
                           'far; and picks which rigs are worth simulating for real.', muted=True, small=True,
                           wrap=True))
        ll.addWidget(label('Records', ink2=True))
        self.store_text = label('', muted=True, small=True, wrap=True)
        ll.addWidget(self.store_text)
        self.in_project = QCheckBox('Keep records with the project')
        self.in_project.setToolTip('Store this project\'s records in a file beside the project file instead of your '
                                   'data folder (undoable)')
        self.in_project.toggled.connect(lambda on: self._set('records with the project' if on else
                                                             'records in the data folder', store_in_project=on))
        ll.addWidget(self.in_project)
        self.store_table = _table(STORE_COLS, stretch=2)
        self.store_table.setMinimumHeight(160)
        ll.addWidget(self.store_table)
        rb = QHBoxLayout()
        b = button('Import…', None, 'import', 'Add records from a JSON-lines file (another team\'s export); '
                                              'duplicates are recognised by id')
        b.clicked.connect(lambda: self.import_file())
        rb.addWidget(b)
        b = button('Export…', 'ghost', 'export', 'Write the records to a JSON-lines file')
        b.clicked.connect(lambda: self.export_records())
        rb.addWidget(b)
        self.anon = QCheckBox('Anonymised')
        self.anon.setChecked(True)
        self.anon.setToolTip('Drop the vehicle\'s, project\'s and variant\'s names and every path; geometry, metrics '
                             'and the provenance hashes stay')
        rb.addWidget(self.anon)
        rb.addStretch(1)
        ll.addLayout(rb)
        ll.addWidget(hline())
        ll.addWidget(label('Objective', ink2=True))
        g = QGridLayout()
        g.setHorizontalSpacing(6)
        g.setVerticalSpacing(4)
        self.w_ped = ScrubSpinBox('', 'plain', 2, '', step=0.1, width=70)
        self.w_ped.setRange(0.0, 10.0)
        self.w_ped.setToolTip('Weight of pedestrian recall in the objective')
        self.w_ped.committed.connect(lambda v: self._objective())
        self.w_veh = ScrubSpinBox('', 'plain', 2, '', step=0.1, width=70)
        self.w_veh.setRange(0.0, 10.0)
        self.w_veh.setToolTip('Weight of vehicle recall (cars, trucks and buses together) in the objective')
        self.w_veh.committed.connect(lambda v: self._objective())
        self.w_cyc = ScrubSpinBox('', 'plain', 2, '', step=0.1, width=70)
        self.w_cyc.setRange(0.0, 10.0)
        self.w_cyc.setToolTip('Weight of cyclist recall in the objective')
        self.w_cyc.committed.connect(lambda v: self._objective())
        self.max_range = ScrubSpinBox('', 'plain', 0, 'm', step=10.0, width=70)
        self.max_range.setRange(10.0, 200.0)
        self.max_range.setToolTip('Regions starting nearer than this count in the objective')
        self.max_range.committed.connect(lambda v: self._set(f'objective range {v:g} m', max_range_m=v))
        self.kappa = ScrubSpinBox('', 'plain', 2, '', step=0.1, width=70)
        self.kappa.setRange(0.0, 10.0)
        self.kappa.setToolTip('The optimiser\'s term is the predicted objective − κ × its std (κ > 0 prefers rigs the '
                              'model is sure about)')
        self.kappa.committed.connect(lambda v: self._set(f'κ {v:g}', kappa=v))
        for i, (t, wdg) in enumerate((('Pedestrian', self.w_ped), ('Vehicle', self.w_veh), ('Cyclist', self.w_cyc),
                                      ('Regions to', self.max_range), ('κ', self.kappa))):
            a = label(t, ink2=True)
            a.setFixedWidth(100)
            g.addWidget(a, i, 0)
            g.addWidget(wdg, i, 1, Qt.AlignmentFlag.AlignLeft)
        ll.addLayout(g)
        ll.addWidget(hline())
        bl = QHBoxLayout()
        self.b_train = button('Train', 'primary', 'sparkle', 'Fit the model on the records and cross-validate it (a '
                                                             'background job; it is saved for the design agent)')
        self.b_train.clicked.connect(lambda: self.train())
        bl.addWidget(self.b_train)
        self.b_cancel = button('Cancel', 'ghost', 'stop', 'Stop training or the campaign')
        self.b_cancel.clicked.connect(lambda: self.cancel())
        bl.addWidget(self.b_cancel)
        self.b_export = button('Export…', 'ghost', 'export', 'Write the model\'s report: JSON, calibration CSV, '
                                                            'predictions CSV and the calibration plot')
        self.b_export.clicked.connect(lambda: self.export())
        bl.addWidget(self.b_export)
        bl.addStretch(1)
        ll.addLayout(bl)
        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.setTextVisible(False)
        self.bar.setFixedHeight(4)
        self.bar.hide()
        ll.addWidget(self.bar)
        self.status = label('', muted=True, small=True, wrap=True)
        ll.addWidget(self.status)
        self.model_status = StatusLine('warning', '')
        ll.addWidget(self.model_status)
        ll.addStretch(1)
        sc.setWidget(w)
        return sc

    def _objective(self):
        ob = {}
        for c, wdg in (('pedestrian', self.w_ped), ('vehicle', self.w_veh), ('cyclist', self.w_cyc)):
            if wdg.value() > 0:
                ob[c] = round(float(wdg.value()), 3)
        if ob:
            self._set('objective weights', objective=ob)

    # ------------------------------------------------------------------ right: prediction, calibration, AL
    def _right(self):
        w = QWidget()
        rl = QVBoxLayout(w)
        rl.setContentsMargins(8, 6, 8, 8)
        tiles = QHBoxLayout()
        self.t_records = Tile('Records', 'Perception records in the store (rigs · scene sets)')
        self.t_obj = Tile('Objective', 'Predicted weighted recall of the variant ± its std')
        self.t_ped = Tile('Pedestrian', 'Predicted mean pedestrian recall over the regions that count')
        self.t_veh = Tile('Vehicle', 'Predicted mean vehicle recall over the regions that count')
        self.t_cov = Tile('95 % coverage', 'Leave-one-scene-out: share of held-out rows inside the 95 % interval')
        for t in (self.t_records, self.t_obj, self.t_ped, self.t_veh, self.t_cov):
            tiles.addWidget(t)
        rl.addLayout(tiles)
        self.tabs = QTabWidget()
        # prediction
        pw = QWidget()
        pl = QVBoxLayout(pw)
        row = QHBoxLayout()
        row.addWidget(label('Class', ink2=True))
        self.cls_pick = QComboBox()
        self.cls_pick.setToolTip('The class the maps show')
        self.cls_pick.currentIndexChanged.connect(lambda i: self.set_class(self.cls_pick.itemData(i)))
        row.addWidget(self.cls_pick)
        row.addStretch(1)
        self.stale = label('', muted=True, small=True)
        row.addWidget(self.stale)
        pl.addLayout(row)
        maps = QHBoxLayout()
        self.map_mean = RegionMap('Predicted recall', 1.0)
        self.map_std = RegionMap('Its std', 0.3)
        maps.addWidget(self.map_mean)
        maps.addWidget(self.map_std)
        pl.addLayout(maps, 1)
        pl.addWidget(label('What moved the prediction', ink2=True))
        self.why = label('', muted=True, small=True, wrap=True)
        pl.addWidget(self.why)
        self.tabs.addTab(pw, 'Prediction')
        # calibration
        cw = QWidget()
        cl = QVBoxLayout(cw)
        ch = QHBoxLayout()
        self.cal_scene = CalibrationChart('scene')
        self.cal_rig = CalibrationChart('rig')
        ch.addWidget(self.cal_scene)
        ch.addWidget(self.cal_rig)
        cl.addLayout(ch, 1)
        self.cv_table = _table(CV_COLS)
        cl.addWidget(self.cv_table, 1)
        self.tabs.addTab(cw, 'Calibration')
        # active learning
        aw = QWidget()
        al = QVBoxLayout(aw)
        ar = QHBoxLayout()
        ar.addWidget(label('Pick', ink2=True))
        self.k_spin = QSpinBox()
        self.k_spin.setRange(1, 20)
        self.k_spin.setToolTip('How many rigs to propose')
        self.k_spin.valueChanged.connect(lambda v: self._set(f'propose {v}', k=v))
        ar.addWidget(self.k_spin)
        self.mode = SegmentedControl([('ei', 'Expected improvement'), ('std', 'Most uncertain'), ('ucb', 'Optimistic')],
                                     'ei')
        self.mode.setToolTip('Expected improvement over the best evaluated rig; the objective\'s largest std; or '
                             'mean + κ·std')
        self.mode.changed.connect(lambda k: self._set(f'pick by {k}', mode=k))
        ar.addWidget(self.mode)
        self.b_propose = button('Propose', None, 'target', 'Rank the project\'s variants for a real evaluation')
        self.b_propose.clicked.connect(lambda: self.propose())
        ar.addWidget(self.b_propose)
        ar.addWidget(label('Rounds', ink2=True))
        self.rounds = QSpinBox()
        self.rounds.setRange(1, 10)
        self.rounds.setValue(2)
        self.rounds.setToolTip('Campaign rounds: pick, evaluate for real, add the records, refit')
        ar.addWidget(self.rounds)
        self.b_campaign = button('Run campaign', 'ghost', 'play', 'Evaluate the picks for real (perception in the '
                                                                 'loop, or CARLA with this module\'s interim '
                                                                 'evaluator), add their records and refit')
        self.b_campaign.clicked.connect(lambda: self.run_campaign())
        ar.addWidget(self.b_campaign)
        ar.addStretch(1)
        al.addLayout(ar)
        self.al_table = _table(AL_COLS, stretch=5)
        al.addWidget(self.al_table, 1)
        self.al_text = label('', muted=True, small=True, wrap=True)
        al.addWidget(self.al_text)
        self.tabs.addTab(aw, 'Active learning')
        nw = QWidget()
        nl = QVBoxLayout(nw)
        self.notes = label('', muted=True, small=True, wrap=True)
        self.notes.setAlignment(Qt.AlignmentFlag.AlignTop)
        nl.addWidget(self.notes)
        nl.addStretch(1)
        self.tabs.addTab(nw, 'Notes')
        rl.addWidget(self.tabs, 1)
        return w

    # ------------------------------------------------------------------ syncing
    def _sync_fields(self):
        st = self.settings
        self._syncing = True
        try:
            self.in_project.setChecked(bool(st.store_in_project))
            for c, wdg in (('pedestrian', self.w_ped), ('vehicle', self.w_veh), ('cyclist', self.w_cyc)):
                wdg.setValue(float(st.objective.get(c, 0.0)))
            self.max_range.setValue(st.max_range_m)
            self.kappa.setValue(st.kappa)
            self.k_spin.setValue(st.k)
            self.mode.set_value(st.mode)
        finally:
            self._syncing = False

    def activate(self):
        self._sync_fields()
        if self.model is not None and self.model.trained and self._stale and not self.busy:
            self.predict()
        self.refresh()

    def _on_change(self, *_):
        self._stale = True
        if self.isVisible():
            self._sync_fields()
            if self.model is not None and self.model.trained and not self.busy:
                self.predict()
            self.refresh()

    def _on_replaced(self, *_):
        self.result, self.proposals, self._stale = None, [], True
        if self.isVisible():
            self.activate()

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
            except RuntimeError as e:
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
        if gen == self._gen:
            self.bar.show()
            self.bar.setValue(int(f * 1000))
            if text:
                self.status.setText(text)

    def train(self) -> bool:
        if self.busy:
            return False
        p = self.ctx.store.project
        st = self.settings
        S = self.store()
        v = p.variant
        self._gen += 1
        self.busy = True
        self.status_text = 'Training…'
        gen, sig = self._gen, self.sig
        engine = self._engine()

        def work(progress, cancel):
            def both(f, text=''):
                progress(f, text)
                try:
                    sig.progress.emit(gen, f, text)
                except RuntimeError:
                    pass
            res = PM.run(p, engine.share() if engine is not None else None, st, v, both, cancel, store=S)
            if res is not None and res.model.trained:
                PM.save_model(res.model)
            return p, res
        ok = self._start(JOB, 'Learned detection model', work)
        if not ok:
            self.busy = False
        self.refresh()
        return ok

    def predict(self, sync=False):
        """The current variant with the model in hand: features, the GP's prediction and what moved it (no ranking:
        Propose does that). In a background thread unless sync; a newer request supersedes an older one."""
        if self.model is None or not self.model.trained:
            return
        p = self.ctx.store.project
        args = (p, self._engine(), self.settings, p.variant, self.store(), self.model)

        def work():
            eng = args[1].share() if args[1] is not None else None
            return PM.run(args[0], eng, args[2], args[3], store=args[4], model=args[5], rank=False)
        if sync:
            try:
                self.result, self._stale = work(), False
            except Exception as e:              # noqa: BLE001 — a project the model cannot read: say so
                self.status_text = f'Prediction failed: {e}'
            return
        self._pred_gen = getattr(self, '_pred_gen', 0) + 1
        gen = self._pred_gen

        def thread():
            try:
                res = work()
            except Exception as e:              # noqa: BLE001
                res = f'Prediction failed: {e}'
            try:
                self.sig.done.emit(-gen, ('predict', (p, res)))
            except RuntimeError:                # the page closed
                pass
        threading.Thread(target=thread, daemon=True).start()

    def _thread_done(self, gen, payload):
        kind, out = payload
        if kind == 'predict':
            if -gen != getattr(self, '_pred_gen', 0):
                return
            p, res = out
            if isinstance(res, str):
                self.status_text = res
            elif p is self.ctx.store.project and res is not None:
                self.result, self._stale = res, False
            self.refresh()
            return
        if gen != self._gen:
            return
        self.busy = False
        self.bar.hide()
        if kind == JOB:
            p, res = out if out is not None else (None, None)
            if res is None or p is not self.ctx.store.project:
                self.status_text = 'Cancelled' if res is None else ''
                self.refresh()
                return
            self.result, self.model, self._stale = res, res.model, False
            self.proposals = res.proposals
            m = res.model
            self.status_text = (f'Trained on {m.n_records} records in {m.seconds:.1f} s' if m.trained else m.reason())
            p.add_log('perf_model', self.status_text)
            self.ctx.msg(f'Learned model · {self.status_text}', 5000)
        elif kind == JOB_AL:
            self.campaign_result = out
            if out is not None:
                self.model = out['model']
                self.predict(sync=True)
                pts = [c for c in out['curve'] if c.get('prospective')]
                self.status_text = (f'Campaign: {len(out["rounds"])} round(s); ' + '; '.join(
                    f'round {c["round"] + 1}: RMSE {c["prospective"]["rmse"]:.3f} on new rigs' for c in pts))
            else:
                self.status_text = 'Campaign cancelled'
        self.refresh()

    def _failed(self, gen, msg):
        if gen == self._gen and self.busy:
            self.busy = False
            self.bar.hide()
            self.status_text = 'Failed: ' + msg.splitlines()[0]
        self.refresh()

    def cancel(self):
        win = self.ctx.window
        for job in (JOB, JOB_AL):
            j = getattr(win, '_jobs', {}).get(job) if win is not None else None
            if j is not None:
                j['cancel'].set()
        if (win is None or not hasattr(win, 'run_job')) and self.busy:
            self._gen += 1
            self.busy = False
            self.status_text = 'Cancelled'
        self.refresh()

    def propose(self) -> list:
        if self.model is None or not self.model.trained:
            self.status_text = self.model.reason() if self.model is not None else 'Train the model first.'
            self.refresh()
            return []
        p = self.ctx.store.project
        st = self.settings
        try:
            cands = PM.candidates_from_variants(p, None, self._engine(),
                                                RG.RegionGrid.from_dict(self.model.grid) if self.model.grid else None)
            self.proposals = PM.propose(cands, st.k, st.objective, self.model, st.mode, st.kappa)
            self.status_text = f'{len(self.proposals)} of {len(cands)} variants proposed'
        except (RuntimeError, ValueError) as e:
            self.status_text = str(e)
            self.proposals = []
        self.refresh()
        return self.proposals

    def carla_ready(self) -> bool:
        if self.evaluator is not None or PM.rig_perception_available():
            return True
        try:
            from ...core import simrig as SR
            from ...sim import server as SV
            st = SR.settings_of(self.ctx.store.project)
            return bool(SV.port_open('127.0.0.1', st.port))
        except Exception:                       # noqa: BLE001
            return False

    def run_campaign(self, rounds=None, k=None) -> bool:
        if self.busy:
            return False
        if not self.carla_ready():
            self.status_text = ('No evaluator: start CARLA (Simulate ▸ CARLA) or add perception in the loop; '
                                'Propose still ranks the variants.')
            self.refresh()
            return False
        p = self.ctx.store.project
        st = self.settings
        S = self.store()
        rounds = int(rounds or self.rounds.value())
        k = int(k or st.k)
        ev = self.evaluator
        self._gen += 1
        self.busy = True
        self.status_text = 'Campaign…'
        gen, sig = self._gen, self.sig
        engine = self._engine()

        def work(progress, cancel):
            def both(f, text=''):
                progress(f, text)
                try:
                    sig.progress.emit(gen, f, text)
                except RuntimeError:
                    pass
            cands = PM.candidates_from_variants(p, None, engine.share() if engine is not None else None)
            return PM.campaign(p, cands, rounds, k, ev, S, st, progress=both, cancel=cancel)
        ok = self._start(JOB_AL, 'Active-learning campaign', work)
        if not ok:
            self.busy = False
        self.refresh()
        return ok

    # ------------------------------------------------------------------ store actions
    def import_file(self, path=None) -> dict | None:
        if path is None:
            path, _ = QFileDialog.getOpenFileName(self, 'Import perception records', '',
                                                  'Perception records (*.jsonl *.json)')
            if not path:
                return None
        try:
            r = PM.import_records(path, self.store())
        except (OSError, ValueError) as e:
            self.status_text = f'Import failed: {e}'
            self.refresh()
            return None
        self.status_text = (f'Imported {r["added"]} record(s); {r["duplicates"]} already there; {r["rejected"]} '
                            'rejected' + (f' ({r["errors"][0]["errors"][0]})' if r['errors'] else ''))
        self._stale = True
        self.refresh()
        return r

    def export_records(self, path=None, anonymise=None) -> int:
        if path is None:
            path, _ = QFileDialog.getSaveFileName(self, 'Export perception records', 'perception_records.jsonl',
                                                  'Perception records (*.jsonl)')
            if not path:
                return 0
        an = self.anon.isChecked() if anonymise is None else bool(anonymise)
        n = PM.export_records(path, anonymise=an, store=self.store())
        self.status_text = f'Exported {n} record(s){" (anonymised)" if an else ""}'
        self.refresh()
        return n

    def export(self, folder=None) -> list:
        res = self.result
        if res is None:
            return []
        if folder is None:
            folder = QFileDialog.getExistingDirectory(self, 'Export the learned model\'s report into')
            if not folder:
                return []
        paths = PM.export(res, self.ctx.store.project, folder)
        self.ctx.msg(f'Exported {len(paths)} files', 4000)
        return paths

    def set_class(self, cls):
        if cls and cls != self.cls:
            self.cls = cls
            self.refresh()

    # ------------------------------------------------------------------ refresh
    def refresh(self):
        p = self.ctx.store.project
        S = self.store()
        docs = S.records()
        summ = PM.store_summary(docs)
        loc = 'beside the project file' if PM.project_store_path(p) else 'in your data folder'
        self.store_text.setText(f'{summ["records"]} records · {summ["rigs"]} rigs · {summ["scenes"]} scene sets, '
                                f'kept {loc} ({os.path.basename(S.path)})' if docs else EMPTY)
        self.store_table.setRowCount(len(docs))
        for i, d in enumerate(docs):
            sc = d.get('scene') or {}
            n = sum(int(c.get('n_gt') or 0) for regs in (d.get('metrics') or {}).values() for c in regs.values())
            _cell(self.store_table, i, 0, str(d.get('source', '')))
            _cell(self.store_table, i, 1, ' · '.join(str(x) for x in (sc.get('town'), sc.get('scenario'),
                                                                      f'seed {sc.get("seed")}') if x))
            _cell(self.store_table, i, 2, str(d.get('rig_label') or d.get('rig_hash', '')[:12]),
                  tip=f'rig {d.get("rig_hash", "")[:16]} · record {d.get("id", "")[:16]}')
            _cell(self.store_table, i, 3, ' '.join(c.get('tag', '') for c in (d.get('rig') or {}).get('cameras', [])))
            det = d.get('detector') or {}
            _cell(self.store_table, i, 4, f'{det.get("id", "")} · {det.get("backend", "")}',
                  tip=str(det.get('evaluator', '')))
            _cell(self.store_table, i, 5, str(n), num=True)
        self.t_records.set(str(summ['records']), '', f'{summ["rigs"]} rigs · {summ["scenes"]} scene sets')
        m = self.model
        trained = m is not None and m.trained
        self.model_status.set('good' if trained else 'warning',
                              (f'Trained {m.fitted_utc} on {m.n_records} records: ' +
                               '; '.join(f'{c}: {t}' for c, t in m.status.items() if t.startswith('trained')))
                              if trained else (m.reason() if m is not None else 'Not trained yet.'))
        self.status.setText(self.status_text)
        self.b_train.setEnabled(not self.busy)
        self.b_campaign.setEnabled(not self.busy)
        self.b_propose.setEnabled(trained and not self.busy)
        self.b_cancel.setEnabled(self.busy)
        self.b_export.setEnabled(self.result is not None)
        self._fill_prediction()
        self._fill_calibration()
        self._fill_al()
        self.notes.setText('\n\n'.join([PM.ASSUMPTIONS, PM.LIMITS_TEXT] +
                                       ([f'{c}: {t}' for c, t in m.status.items()] if m is not None else []) +
                                       (list(self.result.notes) if self.result is not None else [])))

    def _fill_prediction(self):
        res = self.result
        pred = res.prediction if res is not None else None
        classes = list((pred or {}).get('classes') or {})
        self._syncing = True
        try:
            self.cls_pick.clear()
            for c in classes:
                self.cls_pick.addItem(c.capitalize(), c)
            if classes and self.cls not in classes:
                self.cls = 'pedestrian' if 'pedestrian' in classes else classes[0]
            if classes:
                self.cls_pick.setCurrentIndex(classes.index(self.cls))
        finally:
            self._syncing = False
        self.stale.setText('stale: the project changed since this prediction' if self._stale and pred else '')
        if not pred or not pred.get('trained') or self.cls not in pred['classes']:
            why = (pred or {}).get('reason') or (self.model.reason() if self.model is not None else
                                                 'Train the model to see its prediction.')
            for mp in (self.map_mean, self.map_std):
                mp.empty = why
                mp.set(mp.grid, {})
            self.why.setText('')
            self.t_obj.set('—')
            self.t_ped.set('—')
            self.t_veh.set('—')
            return
        grid = RG.RegionGrid.from_dict(res.model.grid) if res.model.grid else RG.RegionGrid()
        d = pred['classes'][self.cls]
        means = {rid: float(v) for rid, v in zip(pred['regions'], d['mean'])}
        stds = {rid: float(v) for rid, v in zip(pred['regions'], d['std'])}
        extra = {rid: [f'68 % {d["lo68"][i]:.2f}–{d["hi68"][i]:.2f}', f'95 % {d["lo95"][i]:.2f}–{d["hi95"][i]:.2f}',
                       f'std {d["std"][i]:.3f}'] for i, rid in enumerate(pred['regions'])}
        self.map_mean.set(grid, means, extra)
        self.map_std.set(grid, stds, extra, vmax=max(0.05, round(max(stds.values()) * 1.1, 2)))
        mo, so = res.objective
        self.t_obj.set(_fmt(mo), '', f'± {_fmt(so)} · {res.variant}')
        for t, c in ((self.t_ped, 'pedestrian'), (self.t_veh, 'vehicle')):
            r = pred['rig'].get(c)
            t.set(_fmt(r['mean']) if r else '—', '', f'± {_fmt(r["std"])}' if r else 'no model for this class')
        if res.attributions:
            self.why.setText(f'Against {res.reference}: ' + ' '.join(a['text'] for a in res.attributions))
        else:
            self.why.setText('This is the reference variant (choose another to compare).' if not res.reference
                             else 'Nothing moved the prediction measurably.')

    def _fill_calibration(self):
        m = self.model
        self.cal_scene.set(m)
        self.cal_rig.set(m)
        rows = PM.cv_table(m) if m is not None else []
        self.cv_table.setRowCount(len(rows))
        for i, r in enumerate(rows):
            _cell(self.cv_table, i, 0, r['class'])
            _cell(self.cv_table, i, 1, 'scene sets' if r['cv'] == 'scene' else 'rigs')
            _cell(self.cv_table, i, 2, METHOD_TEXT.get(r['method'], r.get('note', '')))
            _cell(self.cv_table, i, 3, str(r.get('n', 0)), num=True)
            for c, k, nd in ((4, 'coverage68', 3), (5, 'coverage95', 3), (6, 'rmse', 3), (7, 'nlpd', 2)):
                col = None
                if k == 'coverage95' and r.get(k) is not None:
                    col = QColor(STATUS['good'] if 0.88 <= r[k] <= 0.99 else STATUS['warning'])
                _cell(self.cv_table, i, c, _fmt(r.get(k), nd), num=True, color=col)
        cov = [r for r in rows if r.get('method') == 'gp' and r['cv'] == 'scene' and r.get('coverage95') is not None]
        if cov:
            n = sum(r['n'] for r in cov)
            v = sum(r['coverage95'] * r['n'] for r in cov) / max(n, 1)
            self.t_cov.set(f'{v * 100:.0f}', '%', f'{n} held-out rows', 'good' if 0.88 <= v <= 0.99 else 'warning')
        else:
            self.t_cov.set('—', '', 'needs two scene sets')

    def _fill_al(self):
        rows = self.proposals or []
        self.al_table.setRowCount(len(rows))
        for i, r in enumerate(rows):
            _cell(self.al_table, i, 0, str(i + 1), num=True)
            _cell(self.al_table, i, 1, r['name'])
            _cell(self.al_table, i, 2, _fmt(r.get('mean')), num=True)
            _cell(self.al_table, i, 3, _fmt(r.get('std')), num=True)
            _cell(self.al_table, i, 4, _fmt(r.get('ei'), 4), num=True)
            _cell(self.al_table, i, 5, r.get('why', ''))
        cr = self.campaign_result
        if cr is not None:
            pts = [c for c in cr['curve'] if 'records' in c]
            self.al_text.setText('Learning curve: ' + '; '.join(
                f'round {c["round"] + 1}: {c["records"]} records' +
                (f', RMSE {c["prospective"]["rmse"]:.3f} and 95 % coverage {c["prospective"]["coverage95"]:.2f} on '
                 f'the new rigs before they joined' if c.get('prospective', {}).get('n') else '') for c in pts))
        else:
            self.al_text.setText('' if self.carla_ready() else
                                 'Run campaign needs CARLA (Simulate ▸ CARLA) or perception in the loop; Propose '
                                 'ranks the variants with the model alone.')
