"""Analysis ▸ Design agent: requirements in (typed, or read from documents), a Pareto set of complete rigs out, each
explained and critiqued; nothing changes in the design until one option is accepted (one undo step, written to the
design log).

Left: the brief, the documents, how to read them (the brief parser, or the assistant's provider with the key from
Preferences), the budget, power, tolerance and camera-count fields, the prices of the hardware, the search settings
and Run. Centre: the requirements and constraints read from the brief (each with its quoted source; accept or reject
each row, edit its target), what Vantage cannot check, the Pareto chart on two chosen axes and the option cards.
Right: the selected option against the current variant, why each sensor is there, its self-critique, and Create
variant (an option that changes many things shows its change set for approval first)."""
from __future__ import annotations

import copy
import math
import os
import threading
import traceback

from PyQt6.QtCore import QObject, QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFrame,
                             QGridLayout, QHBoxLayout, QHeaderView, QLineEdit, QListWidget, QPlainTextEdit,
                             QProgressBar, QPushButton, QScrollArea, QSpinBox, QSplitter, QTableWidget,
                             QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget)

from ...core import docs_ingest as DI
from ...core import requirements as RQ
from ...core import rig_designer as RD
from .. import icons
from ..theme import CAMERA, STATUS, THEME, mono, sans
from ..widgets import ROW_H, button, fit_width, hline, label

JOB = 'designer'
REQ_COLS = ('Use', 'Kind', 'Requirement', 'Check', 'Target', 'Priority', 'Source')
OPT_COLS = ('Requirement', 'Current', 'Option', 'Margin', 'Status')
SEN_COLS = ('Sensor', 'What', 'Serves', 'Why it is there')
METHODS = (('deterministic', 'Brief parser (no LLM)'), ('groq', 'Groq (your key)'), ('claude', 'Claude (your key)'))
SEV_COLOR = {'critical': STATUS['critical'], 'serious': STATUS['serious'], 'warning': STATUS['warning'],
             'note': '#898781'}


class _Sig(QObject):
    done = pyqtSignal(int, object)
    failed = pyqtSignal(int, str)
    progress = pyqtSignal(int, float, str)
    read = pyqtSignal(int, object, str)


def _table(cols):
    t = QTableWidget(0, len(cols))
    t.setHorizontalHeaderLabels(cols)
    t.verticalHeader().setVisible(False)
    t.verticalHeader().setDefaultSectionSize(ROW_H)
    t.horizontalHeader().setFixedHeight(ROW_H)
    t.setShowGrid(False)
    t.setWordWrap(False)
    t.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    t.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    t.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
    t.setTextElideMode(Qt.TextElideMode.ElideRight)
    return t


def _item(text, tip='', mono_=False, color=None, editable=False):
    it = QTableWidgetItem(str(text))
    it.setToolTip(tip or str(text))
    if mono_:
        it.setFont(mono(11))
    if color:
        it.setForeground(QColor(color))
    fl = it.flags() & ~Qt.ItemFlag.ItemIsEditable
    if editable:
        fl |= Qt.ItemFlag.ItemIsEditable
    it.setFlags(fl)
    return it


def _fmt(v, unit=''):
    if v is None:
        return '—'
    if isinstance(v, str):
        return v
    return RD._fmt(v) + (f' {unit}' if unit else '')


# =========================================================================== the Pareto chart
class ParetoChart(QWidget):
    """The options on two axes; the current variant hollow; click an option to select it."""
    picked = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setMinimumHeight(220)
        self.options, self.current, self.axes, self.sel = [], None, ['cost', 'margin'], None
        self._pts = []
        THEME.changed.connect(self.update)

    def set(self, options, current, axes, sel=None):
        self.options, self.current, self.axes, self.sel = list(options), current, list(axes), sel
        self.update()

    def points(self):
        out = []
        for o in ([self.current] if self.current else []) + self.options:
            x, y = RD.axis_value(o, self.axes[0]), RD.axis_value(o, self.axes[1])
            if x is not None and y is not None and math.isfinite(x) and math.isfinite(y):
                out.append((o, float(x), float(y)))
        return out

    def missing_text(self) -> str:
        for a in self.axes:
            if self.options and all(RD.axis_value(o, a) is None for o in self.options):
                return ('Cost incomplete: enter the prices of the hardware (left) to chart it.' if a == 'cost' else
                        f'{RD.OBJECTIVE_LABEL.get(a, a)}: no value.')
        return ''

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), THEME.color('panel'))
        pts = self.points()
        self._pts = []
        W, H = self.width(), self.height()
        L, R, T, B = 64, 18, 14, 34
        p.setFont(sans(11))
        p.setPen(THEME.color('muted'))
        if not pts:
            p.drawText(QRectF(0, 0, W, H), Qt.AlignmentFlag.AlignCenter,
                       self.missing_text() or 'Run the design agent to see its options here.')
            p.end()
            return
        xs = [x for _, x, _ in pts]
        ys = [y for _, _, y in pts]
        x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
        if x1 - x0 < 1e-9:
            x0, x1 = x0 - 1, x1 + 1
        if y1 - y0 < 1e-9:
            y0, y1 = y0 - 1, y1 + 1
        dx, dy = (x1 - x0) * 0.08, (y1 - y0) * 0.1
        x0, x1, y0, y1 = x0 - dx, x1 + dx, y0 - dy, y1 + dy

        def P(x, y):
            return QPointF(L + (x - x0) / (x1 - x0) * (W - L - R), H - B - (y - y0) / (y1 - y0) * (H - T - B))
        p.setPen(QPen(THEME.color('hair'), 1))
        p.drawLine(int(L), int(H - B), int(W - R), int(H - B))
        p.drawLine(int(L), int(T), int(L), int(H - B))
        p.setPen(THEME.color('muted'))
        p.setFont(mono(10))
        for v in (x0 + dx, x1 - dx):
            q = P(v, y0)
            p.drawText(QRectF(q.x() - 40, H - B + 2, 80, 14), Qt.AlignmentFlag.AlignCenter, RD._fmt(v))
        for v in (y0 + dy, y1 - dy):
            q = P(x0, v)
            p.drawText(QRectF(0, q.y() - 7, L - 6, 14), Qt.AlignmentFlag.AlignRight, RD._fmt(v))
        p.setFont(sans(11))
        p.drawText(QRectF(L, H - 16, W - L - R, 14), Qt.AlignmentFlag.AlignCenter,
                   RD.OBJECTIVE_LABEL.get(self.axes[0], self.axes[0]) + ' →')
        p.save()
        p.translate(12, (H - B + T) / 2)
        p.rotate(-90)
        p.drawText(QRectF(-90, -8, 180, 16), Qt.AlignmentFlag.AlignCenter,
                   RD.OBJECTIVE_LABEL.get(self.axes[1], self.axes[1]) + ' →')
        p.restore()
        for o, x, y in pts:
            q = P(x, y)
            self._pts.append((o['label'], q))
            if o['label'] == 'current':
                p.setPen(QPen(THEME.color('ink2'), 1.6))
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawEllipse(q, 6, 6)
            else:
                col = QColor(CAMERA.get(THEME.name, CAMERA['dark'])[0]) if o['feasible'] else \
                    QColor(STATUS['critical'])
                p.setPen(QPen(THEME.color('ink'), 2) if o['label'] == self.sel else Qt.PenStyle.NoPen)
                p.setBrush(col)
                p.drawEllipse(q, 6, 6)
            p.setPen(THEME.color('ink'))
            p.drawText(QPointF(q.x() + 9, q.y() - 7), o['label'])
        p.end()

    def mousePressEvent(self, e):
        pos = e.position()
        best = min(self._pts, key=lambda t: (t[1].x() - pos.x()) ** 2 + (t[1].y() - pos.y()) ** 2, default=None)
        if best is not None and (best[1].x() - pos.x()) ** 2 + (best[1].y() - pos.y()) ** 2 < 14 ** 2 and \
                best[0] != 'current':
            self.picked.emit(best[0])


class OptionCard(QFrame):
    """One option: its verdict, cost, power, margin, detection, what it changes and its worst finding."""

    def __init__(self, page, opt, cur):
        super().__init__()
        self.setObjectName('panel')
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setFixedWidth(250)
        self.label = opt['label']
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 6, 8, 6)
        lay.setSpacing(2)
        head = QHBoxLayout()
        sev = 'good' if opt['verdict'] == 'pass' else 'critical' if opt['verdict'] == 'fail' else 'warning'
        t = label(f'Option {opt["label"]}', bold=True)
        head.addWidget(t)
        head.addStretch(1)
        v = label({'pass': 'meets every must', 'fail': 'misses a must', 'incomplete': 'not fully checked'}
                  [opt['verdict']], small=True)
        v.setStyleSheet(f'color: {STATUS[sev]}')
        head.addWidget(v)
        lay.addLayout(head)
        cur_ = page.settings.currency
        m = opt['margin']
        for text in (f'{RD._money(opt["cost"], cur_)}' + (f' · {len(opt["cost_missing"])} unpriced'
                                                          if opt['cost'] is None else '')
                     + f' · {opt["power_w"]:.0f} W',
                     'margin ' + ('—' if m is None else f'{m:+.2f}') + f' · detection {opt["detection"] * 100:.0f} %'
                     + ('' if opt['detection_kind'] == 'learned' else ' (geometric)'),
                     f'{opt["counts"]["cameras"]} cameras · {opt["counts"]["lidars"]} lidars · '
                     f'{opt["counts"]["radars"]} radars'):
            lay.addWidget(label(text, mono_=True, small=True))
        ch = opt['changes']
        lay.addWidget(label(('; '.join(ch[:3]) + (f' … ({len(ch)} changes)' if len(ch) > 3 else '')) or
                            'no change', muted=True, small=True, wrap=True))
        worst = next((c for c in opt['critique'] if c['severity'] in ('critical', 'serious')), None)
        if worst is not None:
            w = label(worst['message'], small=True, wrap=True)
            w.setStyleSheet(f'color: {SEV_COLOR[worst["severity"]]}')
            lay.addWidget(w)
        lay.addStretch(1)
        row = QHBoxLayout()
        self.b_show = button('Details', 'ghost', None, 'This option against the current variant, why each sensor, '
                                                       'its self-critique')
        self.b_show.clicked.connect(lambda: page.select_option(self.label))
        self.b_create = button('Create variant', 'primary', 'plus', 'Add this option to the project as a new variant '
                                                                     '(one undo step)')
        self.b_create.clicked.connect(lambda: page.create_variant(self.label))
        row.addWidget(self.b_show)
        row.addStretch(1)
        row.addWidget(self.b_create)
        lay.addLayout(row)


# =========================================================================== the page
class DesignerPage(QWidget):
    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        self.res: RD.DesignerResult | None = None
        self.sel = None
        self.pending = None                 # an option waiting for approval
        self.busy = self.reading = False
        self._gen = 0
        self._rgen = 0
        self._cancel = threading.Event()
        self._filling = False
        self._dirty = True
        self.status_text = ''
        self.sig = _Sig()
        self.sig.done.connect(self._done)
        self.sig.failed.connect(self._failed)
        self.sig.progress.connect(self._progress)
        self.sig.read.connect(self._read_done)
        self.setObjectName('base')
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        split = QSplitter(Qt.Orientation.Horizontal)
        split.setHandleWidth(2)
        split.addWidget(self._left())
        split.addWidget(self._centre())
        split.addWidget(self._right())
        split.setSizes([330, 820, 430])
        lay.addWidget(split)
        ctx.store.changed.connect(self._on_change)
        ctx.store.project_replaced.connect(self._reset)
        self.refresh()

    @property
    def settings(self) -> RD.DesignerSettings:
        return RD.settings_of(self.ctx.store.project)

    def _edit(self, fn, text):
        """One undoable edit of the designer's settings: fn(st) changes them."""
        def apply(p):
            st = RD.settings_of(p)
            fn(st)
            p.analysis.designer = st.to_dict()
        self.ctx.store.edit('analysis', apply, text)

    # ------------------------------------------------------------------ layout
    def _left(self):
        w = QWidget()
        w.setObjectName('panel')
        ll = QVBoxLayout(w)
        ll.setContentsMargins(8, 6, 8, 8)
        ll.setSpacing(4)
        top = QHBoxLayout()
        top.addWidget(label('Design agent'))
        top.addStretch(1)
        top.addWidget(label('requirements → rigs', muted=True, small=True))
        ll.addLayout(top)
        ll.addWidget(hline())
        ll.addWidget(label('State what the rig must do, or add the documents that say it. The agent proposes complete '
                           'rigs trading cost, power, margin and detection; nothing changes until you create one.',
                           muted=True, small=True, wrap=True))
        ll.addWidget(label('Brief', bold=True))
        self.brief = QPlainTextEdit()
        self.brief.setPlaceholderText('e.g. Robotaxi, pedestrians at 80 m forward, 360° up to 30 m, budget under $6k, '
                                      'under 300 W, survives a 2 mm mounting tolerance')
        self.brief.setFixedHeight(96)
        self.brief.textChanged.connect(self._brief_typed)
        ll.addWidget(self.brief)
        ll.addWidget(label('Documents', bold=True))
        self.docs = QListWidget()
        self.docs.setFixedHeight(70)
        ll.addWidget(self.docs)
        dr = QHBoxLayout()
        self.b_add_docs = button('Add documents…', None, 'file', 'An ODD specification, a regulation excerpt, a test '
                                                               'catalogue: .txt, .md, .pdf, .docx or .html')
        self.b_add_docs.clicked.connect(lambda: self.add_documents())
        self.b_rm_doc = button('Remove', 'ghost', 'trash', 'Remove the selected document')
        self.b_rm_doc.clicked.connect(self.remove_document)
        dr.addWidget(self.b_add_docs)
        dr.addStretch(1)
        dr.addWidget(self.b_rm_doc)
        ll.addLayout(dr)
        mr = QHBoxLayout()
        self.method = QComboBox()
        for k, t in METHODS:
            self.method.addItem(t, k)
        self.method.setToolTip('How to read the requirements: the brief parser (no network), or the assistant\'s '
                               'provider with your key from Preferences ▸ Assistant (the text of the brief and the '
                               'documents is sent to it). Figures are always checked against the quoted sentence.')
        self.b_read = button('Read requirements', 'primary', 'sparkle', 'Read the brief and the documents into '
                                                                         'requirements and constraints, each with its '
                                                                         'quoted source')
        self.b_read.clicked.connect(lambda: self.read_requirements())
        mr.addWidget(self.method, 1)
        mr.addWidget(self.b_read)
        ll.addLayout(mr)
        ll.addSpacing(6)
        ll.addWidget(label('Limits', bold=True))
        g = QGridLayout()
        g.setVerticalSpacing(3)
        g.setColumnStretch(1, 1)

        def spin(lo, hi, dec, suffix, tip):
            s = QDoubleSpinBox()
            s.setRange(lo, hi)
            s.setDecimals(dec)
            s.setSuffix(suffix)
            s.setSpecialValueText('none')
            s.setToolTip(tip)
            s.setKeyboardTracking(False)
            return s
        self.f_budget = spin(0, 1e8, 0, '', 'The rig\'s budget (0 = none): the sum of the prices below must stay under it')
        self.f_currency = QComboBox()
        for c in ('USD', 'EUR', 'GBP'):
            self.f_currency.addItem(c)
        self.f_power = spin(0, 5000, 0, ' W', 'The rig\'s steady power limit (0 = none), as the power budget counts '
                                              'it (with the compute, recorder, switch and time server)')
        self.f_tol = spin(0, 50, 1, ' mm', 'The mounting tolerance every must requirement has to survive (0 = none)')
        self.f_cams = QSpinBox()
        self.f_cams.setRange(0, RD.MAX_CAMERAS)
        self.f_cams.setSpecialValueText('any')
        self.f_cams.setToolTip('At most this many cameras (any = up to 8)')
        for r, (t, wdg) in enumerate((('Budget', self.f_budget), ('Currency', self.f_currency),
                                      ('Power under', self.f_power), ('Tolerance', self.f_tol),
                                      ('Cameras ≤', self.f_cams))):
            a = label(t, ink2=True)
            a.setFixedWidth(86)
            g.addWidget(a, r, 0)
            g.addWidget(wdg, r, 1)
        ll.addLayout(g)
        self.f_budget.valueChanged.connect(lambda v: self._set_limit('budget', v))
        self.f_power.valueChanged.connect(lambda v: self._set_power(v))
        self.f_tol.valueChanged.connect(lambda v: self._set_limit('tolerance_mm', v))
        self.f_cams.valueChanged.connect(lambda v: self._set_limit('max_cameras', v))
        self.f_currency.activated.connect(lambda _i: self._set_currency(self.f_currency.currentText()))
        ll.addSpacing(6)
        ph = QHBoxLayout()
        ph.addWidget(label('Prices', bold=True))
        ph.addStretch(1)
        self.c_all = QCheckBox('whole library')
        self.c_all.setToolTip('List every module, lens and sensor of the library (else the current rig\'s and the '
                              'options\')')
        self.c_all.toggled.connect(lambda _v: self._fill_prices())
        ph.addWidget(self.c_all)
        ll.addLayout(ph)
        ll.addWidget(label('Vantage has no prices: enter yours. A rig with an unpriced item has no cost (the cost '
                           'objective and the budget are then incomplete for it).', muted=True, small=True, wrap=True))
        self.prices = _table(('Item', 'Kind', 'Price'))
        self.prices.setEditTriggers(QAbstractItemView.EditTrigger.DoubleClicked |
                                    QAbstractItemView.EditTrigger.EditKeyPressed |
                                    QAbstractItemView.EditTrigger.AnyKeyPressed)
        self.prices.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.prices.setFixedHeight(170)
        self.prices.itemChanged.connect(self._price_edited)
        ll.addWidget(self.prices)
        ll.addSpacing(6)
        ll.addWidget(label('Search', bold=True))
        g2 = QGridLayout()
        g2.setVerticalSpacing(3)
        self.s_options = QSpinBox()
        self.s_options.setRange(*RD.LIMITS['options'])
        self.s_evals = QSpinBox()
        self.s_evals.setRange(*RD.LIMITS['evaluations'])
        self.s_budget = QSpinBox()
        self.s_budget.setRange(int(RD.LIMITS['budget_s'][0]), int(RD.LIMITS['budget_s'][1]))
        self.s_budget.setSuffix(' s')
        self.s_budget.setSpecialValueText('none')          # 0: no time limit (the default: reproducible)
        self.s_seed = QSpinBox()
        self.s_seed.setRange(0, 2 ** 31 - 1)
        for r, (t, wdg, key, tip) in enumerate((
                ('Options', self.s_options, 'options', 'How many options to propose (3–7)'),
                ('Rigs to score', self.s_evals, 'evaluations', 'The search\'s budget in rigs scored (the result is '
                                                               'the same for the same seed)'),
                ('Time limit', self.s_budget, 'budget_s', 'None (the default): the rigs to score alone decide, so the '
                                                          'same seed gives the same options on any machine. A limit '
                                                          'stops the search there at the latest, and says so; the '
                                                          'options then depend on the machine\'s speed'),
                ('Seed', self.s_seed, 'seed', 'The random seed: the same seed, the same options'))):
            a = label(t, ink2=True)
            a.setFixedWidth(86)
            wdg.setToolTip(tip)
            wdg.setKeyboardTracking(False)
            g2.addWidget(a, r, 0)
            g2.addWidget(wdg, r, 1)
            wdg.valueChanged.connect(lambda v, k=key: self._set_search(k, v))
        ll.addLayout(g2)
        self.checks = {}
        for key, text, tip in (('vary_lenses', 'Swap lenses and modules', 'Try other lenses and camera modules from '
                                                                          'the library'),
                               ('vary_count', 'Add and remove cameras', 'Cameras (or mirrored pairs) added beside an '
                                                                        'existing mount, or removed'),
                               ('vary_lidar', 'Change lidars', 'Swap, add or remove lidars'),
                               ('vary_radar', 'Change radars', 'Swap, add or remove radars'),
                               ('placement', 'Placement search', 'Re-aim the front\'s cameras with the placement '
                                                                 'optimiser'),
                               ('reference_seeds', 'Start from reference rigs', 'Also start from the published '
                                                                                'reference layouts scaled to this '
                                                                                'vehicle')):
            cb = QCheckBox(text)
            cb.setToolTip(tip)
            cb.toggled.connect(lambda v, k=key: self._set_search(k, bool(v)))
            self.checks[key] = cb
            ll.addWidget(cb)
        ll.addStretch(1)
        self.status = label('', muted=True, small=True, wrap=True)
        ll.addWidget(self.status)
        self.bar = QProgressBar()
        self.bar.setTextVisible(False)
        self.bar.setRange(0, 1000)
        self.bar.hide()
        ll.addWidget(self.bar)
        bl = QHBoxLayout()
        self.b_export = button('Export…', None, 'export', 'The options as JSON and CSV, why each sensor (CSV) and the '
                                                          'chart (PNG)')
        self.b_export.clicked.connect(lambda: self.export())
        self.b_cancel = button('Cancel', 'ghost', 'stop', 'Stop the design run')
        self.b_cancel.clicked.connect(self.cancel)
        self.b_run = button('Run', 'primary', 'optimise', 'Search for rigs that meet the accepted requirements')
        self.b_run.clicked.connect(lambda: self.run())
        bl.addWidget(self.b_export)
        bl.addStretch(1)
        bl.addWidget(self.b_cancel)
        bl.addWidget(self.b_run)
        ll.addLayout(bl)
        sc = QScrollArea()
        sc.setWidgetResizable(True)
        sc.setWidget(w)
        fit_width(sc)
        sc.setMinimumWidth(280)
        return sc

    def _centre(self):
        w = QWidget()
        w.setObjectName('panel')
        cl = QVBoxLayout(w)
        cl.setContentsMargins(10, 8, 10, 8)
        cl.setSpacing(6)
        head = QHBoxLayout()
        head.addWidget(label('Requirements and constraints', bold=True))
        head.addStretch(1)
        self.req_note = label('', muted=True, small=True)
        head.addWidget(self.req_note)
        cl.addLayout(head)
        self.tabs = QTabWidget()
        self.req = _table(REQ_COLS)
        self.req.setEditTriggers(QAbstractItemView.EditTrigger.DoubleClicked |
                                 QAbstractItemView.EditTrigger.EditKeyPressed)
        hh = self.req.horizontalHeader()
        for c in range(len(REQ_COLS)):
            hh.setSectionResizeMode(c, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(2, QHeaderView.ResizeMode.Interactive)
        hh.resizeSection(2, 200)
        hh.setSectionResizeMode(6, QHeaderView.ResizeMode.Stretch)
        self.req.itemChanged.connect(self._req_edited)
        self.tabs.addTab(self.req, 'Checked')
        self.unchecked = _table(('Statement', 'Source', 'Why Vantage cannot check it'))
        self.unchecked.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.tabs.addTab(self.unchecked, 'Not checkable')
        self.rejected = _table(('Row', 'Source', 'Why it was rejected'))
        self.rejected.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.tabs.addTab(self.rejected, 'Rejected')
        cl.addWidget(self.tabs, 3)
        ch = QHBoxLayout()
        ch.addWidget(label('Options', bold=True))
        ch.addStretch(1)
        self.ax = []
        for k in range(2):
            cb = QComboBox()
            for a in RD.AXES:
                cb.addItem(RD.OBJECTIVE_LABEL[a], a)
            cb.setToolTip('The chart\'s ' + ('horizontal' if k == 0 else 'vertical') + ' axis')
            cb.activated.connect(lambda _i: self._axes_picked())
            ch.addWidget(label('x' if k == 0 else 'y', muted=True, small=True))
            ch.addWidget(cb)
            self.ax.append(cb)
        cl.addLayout(ch)
        self.chart = ParetoChart()
        self.chart.picked.connect(self.select_option)
        cl.addWidget(self.chart, 2)
        self.cards_host = QWidget()
        self.cards = QHBoxLayout(self.cards_host)
        self.cards.setContentsMargins(0, 0, 0, 0)
        self.cards.setSpacing(6)
        self.cards.addStretch(1)
        sc = QScrollArea()
        sc.setWidgetResizable(True)
        sc.setWidget(self.cards_host)
        sc.setFixedHeight(200)
        sc.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        cl.addWidget(sc)
        self.card_widgets = []
        self.search_note = label('', muted=True, small=True, wrap=True)
        cl.addWidget(self.search_note)
        return w

    def _right(self):
        w = QWidget()
        w.setObjectName('panel')
        rl = QVBoxLayout(w)
        rl.setContentsMargins(8, 6, 8, 8)
        rl.setSpacing(4)
        self.d_title = label('No option selected', bold=True)
        rl.addWidget(self.d_title)
        self.d_sub = label('', muted=True, small=True, wrap=True)
        rl.addWidget(self.d_sub)
        rl.addWidget(label('Against the current variant', bold=True))
        self.cmp = _table(OPT_COLS)
        self.cmp.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        rl.addWidget(self.cmp, 2)
        rl.addWidget(label('Why each sensor', bold=True))
        self.sens = _table(SEN_COLS)
        self.sens.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        rl.addWidget(self.sens, 2)
        rl.addWidget(label('Self-critique', bold=True))
        self.crit = QListWidget()
        self.crit.setWordWrap(True)
        rl.addWidget(self.crit, 2)
        self.approve_box = QWidget()
        al = QVBoxLayout(self.approve_box)
        al.setContentsMargins(0, 4, 0, 0)
        self.approve_title = label('', bold=True, wrap=True)
        al.addWidget(self.approve_title)
        self.approve_list = QListWidget()
        self.approve_list.setFixedHeight(110)
        al.addWidget(self.approve_list)
        ar = QHBoxLayout()
        self.b_approve_no = button('Not now', 'ghost')
        self.b_approve_no.clicked.connect(self._approve_cancel)
        self.b_approve = button('Approve and create', 'primary', 'check', 'Record the approval of this change set and '
                                                                         'create the variant (one undo step)')
        self.b_approve.clicked.connect(self._approve_and_create)
        ar.addWidget(self.b_approve_no)
        ar.addStretch(1)
        ar.addWidget(self.b_approve)
        al.addLayout(ar)
        self.approve_box.hide()
        rl.addWidget(self.approve_box)
        self.why = QLineEdit()
        self.why.setPlaceholderText('Why this option (for the design log, optional)')
        rl.addWidget(self.why)
        br = QHBoxLayout()
        self.b_compare = button('Open in Compare', 'ghost', 'compare', 'After creating it: compare the new variant '
                                                                       'with the others on the Compare page')
        self.b_compare.clicked.connect(self.open_compare)
        self.b_create = button('Create variant', 'primary', 'plus', 'Add the selected option to the project as a new '
                                                                     'variant (one undo step)')
        self.b_create.clicked.connect(lambda: self.create_variant(self.sel))
        br.addWidget(self.b_compare)
        br.addStretch(1)
        br.addWidget(self.b_create)
        rl.addLayout(br)
        return w

    # ------------------------------------------------------------------ state
    def _on_change(self, sections=None):
        if self.isVisible():
            self.refresh()
        else:
            self._dirty = True

    def _reset(self, *_):
        self._gen += 1
        self._rgen += 1
        self._cancel.set()
        self.busy = self.reading = False
        self.res, self.sel, self.pending = None, None, None
        self.status_text = ''
        self._on_change()

    def showEvent(self, e):
        super().showEvent(e)
        if self._dirty:
            self.refresh()

    def activate(self):
        if self._dirty:
            self.refresh()

    def refresh(self):
        self._dirty = False
        st = self.settings
        self._filling = True
        try:
            if self.brief.toPlainText() != st.brief:
                self.brief.setPlainText(st.brief)
            self.docs.clear()
            for d in st.documents:
                self.docs.addItem(f'{d.get("name", "")} · {d.get("sentences", 0)} sentences'
                                  + (f' · {len(d.get("problems") or [])} problems' if d.get('problems') else ''))
                self.docs.item(self.docs.count() - 1).setToolTip('\n'.join(d.get('problems') or []) or d.get('name'))
            cs = RD.constraint_set(st)
            self.f_budget.setValue(cs['budget']['value'] if 'budget' in cs else 0.0)
            self.f_currency.setCurrentText(st.currency)
            pw = [r for r in st.requirements if r['item']['metric'] == 'power.steady_w' and r.get('accepted', True)]
            self.f_power.setValue(min(float(r['item']['target']) for r in pw) if pw else 0.0)
            self.f_tol.setValue(cs['tolerance_mm']['value'] if 'tolerance_mm' in cs else 0.0)
            self.f_cams.setValue(int(cs['max_cameras']['value']) if 'max_cameras' in cs else 0)
            self.s_options.setValue(st.options)
            self.s_evals.setValue(st.evaluations)
            self.s_budget.setValue(int(st.budget_s))
            self.s_seed.setValue(st.seed)
            for k, cb in self.checks.items():
                cb.setChecked(bool(getattr(st, k)))
            for cb, a in zip(self.ax, st.axes):
                cb.setCurrentIndex(max(0, cb.findData(a)))
        finally:
            self._filling = False
        self._fill_requirements()
        self._fill_prices()
        self._fill_options()
        self._fill_state()

    def _fill_state(self):
        st = self.settings
        n = sum(1 for r in st.requirements if r.get('accepted', True))
        self.b_run.setEnabled(not self.busy and n > 0)
        self.b_cancel.setEnabled(self.busy or self.reading)
        self.b_read.setEnabled(not self.reading and bool(st.brief.strip() or st.documents))
        self.b_export.setEnabled(self.res is not None and not self.busy)
        self.b_rm_doc.setEnabled(bool(st.documents))
        self.bar.setVisible(self.busy or self.reading)
        opt = self.res.option(self.sel) if self.res is not None and self.sel else None
        self.b_create.setEnabled(opt is not None and not self.busy)
        self.b_compare.setEnabled(any(a.get('variant_uid') and self.ctx.store.project.variant_by_uid(a['variant_uid'])
                                      for a in st.accepted))
        stale = self.res is not None and self.res.inputs_key != RD.inputs_key(self.ctx.store.project, st)
        self.status.setText(self.status_text + (' · the inputs changed since this run: run again'
                                                if stale and not self.busy else ''))

    # ------------------------------------------------------------------ brief and documents
    def _brief_typed(self):
        if self._filling:
            return
        text = self.brief.toPlainText()

        def fn(st):
            st.brief = text
        self.ctx.store.edit('analysis', lambda p: self._apply(p, fn), 'Design brief edited', merge_key='designer-brief')

    def _apply(self, p, fn):
        st = RD.settings_of(p)
        fn(st)
        p.analysis.designer = st.to_dict()

    def add_documents(self, paths=None):
        if paths is None:
            paths, _ = QFileDialog.getOpenFileNames(self, 'Add requirement documents', '',
                                                    'Documents (*.txt *.md *.markdown *.pdf *.docx *.html *.htm)')
        if not paths:
            return []
        docs = []
        st = self.settings
        tags = {d.get('tag') for d in st.documents}
        k = 1
        for path in paths:
            while f'D{k}' in tags:
                k += 1
            d = DI.read(path, f'D{k}')
            tags.add(d.tag)
            docs.append({'tag': d.tag, 'name': d.name, 'path': path, 'kind': d.kind, 'pages': d.pages,
                         'sentences': len(d.sentences), 'problems': list(d.problems), 'sha256': d.sha256})

        def fn(st):
            st.documents = st.documents + docs
        self._edit(fn, f'Added {len(docs)} document{"s" * (len(docs) != 1)} to the design brief')
        bad = [f'{d["name"]}: {d["problems"][0]}' for d in docs if d['problems']]
        self.ctx.msg('Added ' + ', '.join(d['name'] for d in docs) + (f' · {"; ".join(bad)}' if bad else ''), 6000)
        return docs

    def remove_document(self):
        i = self.docs.currentRow()
        st = self.settings
        if not 0 <= i < len(st.documents):
            i = len(st.documents) - 1
        if i < 0:
            return
        name = st.documents[i].get('name', '')

        def fn(st):
            st.documents = [d for j, d in enumerate(st.documents) if j != i]
        self._edit(fn, f'Removed {name} from the design brief')

    def _documents(self):
        """The brief and the documents read again (their sentences with ids), and what could not be read."""
        st = self.settings
        out = []
        if st.brief.strip():
            out.append(DI.from_text(st.brief))
        gone = []
        for d in st.documents:
            if d.get('path') and os.path.isfile(d['path']):
                out.append(DI.read(d['path'], d.get('tag') or 'D'))
            else:
                gone.append(d.get('name') or d.get('path') or '?')
        if gone:
            self.ctx.msg(f'Not found any more (moved or deleted): {", ".join(gone)}: read without them.', 6000)
        return out

    def _llm_key(self, provider):
        """(key, model, url) for the provider from Preferences ▸ Assistant; (None, …) when there is none."""
        from ...assistant import secrets as SEC, providers as PR
        key, _src = SEC.load(provider)
        model = str(self.ctx.settings.value(f'assistant/model_{provider}', '') or '') or PR.default_model(provider)
        return key, model, None

    def read_requirements(self, method=None, wait=False):
        """Read the brief and documents into requirement, constraint and not-checkable rows (shown for review; the
        rows replace the ones read before, rows you entered stay)."""
        method = method or self.method.currentData()
        docs = self._documents()
        if not docs:
            self.ctx.msg('Type a brief or add a document first.', 4000)
            return False
        p = self.ctx.store.project
        key = model = url = None
        if method in ('groq', 'claude'):
            key, model, url = self._llm_key(method)
            if not key:
                self.ctx.msg(f'No {method.capitalize()} key in Preferences ▸ Assistant: reading with the brief parser '
                             'instead.', 6000)
                method = 'deterministic'
        self._rgen += 1
        gen = self._rgen
        self.reading = True
        self.status_text = 'Reading the requirements…'
        self._cancel = threading.Event()
        cancel = self._cancel
        self._fill_state()
        snap = p.snapshot()

        def work():
            from ...assistant import designer_llm as DL
            note = ''
            try:
                ex = DL.extract(docs, snap, method if key else None, key, model, url, cancel)
            except Exception as e:
                note = f'{method.capitalize()} could not read them ({e}); read with the brief parser instead.'
                ex = DL.deterministic(docs, snap)
            return ex, note
        if wait:
            ex, note = work()
            self._read_done(gen, ex, note)
            return True

        def thread():
            try:
                ex, note = work()
                self.sig.read.emit(gen, ex, note)
            except RuntimeError:
                pass
            except Exception as e:
                try:
                    self.sig.failed.emit(-gen, f'{type(e).__name__}: {e}')
                except RuntimeError:
                    pass
        threading.Thread(target=thread, daemon=True).start()
        return True

    def _read_done(self, gen, ex, note):
        if gen != self._rgen:
            return
        self.reading = False
        keep_r = [r for r in self.settings.requirements if r.get('origin') == 'user']
        keep_c = [c for c in self.settings.constraints if c.get('origin') == 'user']

        def fn(st):
            st.requirements = keep_r + ex.requirements
            st.constraints = keep_c + ex.constraints
            st.not_checkable = ex.not_checkable
            st.rejected = ex.rejected
            st.extraction = {k: v for k, v in ex.to_dict().items() if k in ('method', 'model', 'usage', 'seconds',
                                                                             'documents', 'notes')}
        self._edit(fn, f'Read {len(ex.requirements)} requirements and {len(ex.constraints)} constraints from the brief')
        self.status_text = (f'Read by {ex.method}' + (f' ({ex.model})' if ex.model else '') + f': '
                            f'{len(ex.requirements)} requirements, {len(ex.constraints)} constraints, '
                            f'{len(ex.not_checkable)} not checkable, {len(ex.rejected)} rejected'
                            + (f' · {note}' if note else ''))
        self.refresh()

    # ------------------------------------------------------------------ requirement rows
    def _rows(self):
        st = self.settings
        return [('req', i, r) for i, r in enumerate(st.requirements)] + \
            [('con', i, c) for i, c in enumerate(st.constraints)]

    def _fill_requirements(self):
        st = self.settings
        self._filling = True
        try:
            rows = self._rows()
            self.req.setRowCount(len(rows))
            for k, (kind, i, r) in enumerate(rows):
                src = r.get('source') or {}
                stext = (f'"{src.get("quote", "")}" — {src.get("doc", "")} {src.get("location", "")}'.strip()
                         if src else 'entered on this page')
                use = _item('', 'Use this row in the design (untick to reject it)')
                use.setFlags(use.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                use.setCheckState(Qt.CheckState.Checked if r.get('accepted', True) else Qt.CheckState.Unchecked)
                self.req.setItem(k, 0, use)
                if kind == 'req':
                    it = r['item']
                    m = RQ.METRICS.get(it['metric'])
                    self.req.setItem(k, 1, _item('requirement', 'Checked by ' + (m.label if m else it['metric'])))
                    self.req.setItem(k, 2, _item(it['text']))
                    self.req.setItem(k, 3, _item(RQ.describe(it, self.ctx.store.project)))
                    self.req.setItem(k, 4, _item(f'{RQ.OP_TEXT.get(it["op"], "")} {it["target"]:g}', 'Double-click '
                                                 'to edit (e.g. ≥ 80 or <= 0.5)', True, editable=True))
                    self.req.setItem(k, 5, _item(it['priority'], 'must or should (double-click to edit)',
                                                 editable=True))
                else:
                    self.req.setItem(k, 1, _item('constraint', RQ.CONSTRAINT_LABEL.get(r['kind'], r['kind'])))
                    self.req.setItem(k, 2, _item(r.get('text') or RQ.CONSTRAINT_LABEL.get(r['kind'], '')))
                    self.req.setItem(k, 3, _item(RQ.CONSTRAINT_LABEL.get(r['kind'], r['kind'])))
                    self.req.setItem(k, 4, _item(f'{RQ.OP_TEXT.get(r["op"], "")} {r["value"]:g} {r.get("unit", "")}'
                                                 .strip(), 'Double-click to edit', True, editable=True))
                    self.req.setItem(k, 5, _item(r.get('priority', 'must'), editable=True))
                self.req.setItem(k, 6, _item(stext, src.get('sentence', stext)))
            ex = st.extraction or {}
            self.req_note.setText((f'read by {ex.get("method")}' + (f' · {ex["model"]}' if ex.get('model') else '')
                                   if ex else '') + f' · {len(rows)} rows')
            self.unchecked.setRowCount(len(st.not_checkable))
            for k, n in enumerate(st.not_checkable):
                src = n.get('source') or {}
                self.unchecked.setItem(k, 0, _item(src.get('quote', ''), src.get('sentence', '')))
                self.unchecked.setItem(k, 1, _item(f'{src.get("doc", "")} {src.get("location", "")}'.strip()))
                self.unchecked.setItem(k, 2, _item(n.get('why', '')))
            self.rejected.setRowCount(len(st.rejected))
            for k, n in enumerate(st.rejected):
                src = n.get('source') or {}
                row = n.get('row')
                txt = (f'{row.get("metric") or row.get("kind")} {row.get("op", "")} '
                       f'{row.get("target", row.get("value", ""))}' if isinstance(row, dict) else str(row))
                self.rejected.setItem(k, 0, _item(txt))
                self.rejected.setItem(k, 1, _item(f'"{src.get("quote", "")}" {src.get("location", "")}' if src else
                                                  '—', src.get('sentence', '')))
                self.rejected.setItem(k, 2, _item(n.get('why', '')))
            self.tabs.setTabText(1, f'Not checkable ({len(st.not_checkable)})')
            self.tabs.setTabText(2, f'Rejected ({len(st.rejected)})')
        finally:
            self._filling = False

    def _req_edited(self, it):
        if self._filling:
            return
        k, col = it.row(), it.column()
        rows = self._rows()
        if not 0 <= k < len(rows):
            return
        kind, i, r = rows[k]
        if col == 0:
            on = it.checkState() == Qt.CheckState.Checked

            def fn(st):
                (st.requirements if kind == 'req' else st.constraints)[i]['accepted'] = on
            self._edit(fn, f'{"Accepted" if on else "Rejected"} "{(r.get("item") or r).get("text", "")[:40]}"')
            return
        if col == 4:
            import re
            mt = re.match(r'\s*(≥|>=|≤|<=|=|==|<|>)?\s*([-+]?\d+(?:[.,]\d+)?)', it.text())
            if not mt:
                self.ctx.msg('Write the target as a number, optionally after ≥, ≤, <, > or =.', 4000)
                self._fill_requirements()
                return
            op = {'≥': '>=', '≤': '<=', '=': '==', None: None}.get(mt.group(1), mt.group(1))
            val = float(mt.group(2).replace(',', '.'))

            def fn(st):
                row = (st.requirements if kind == 'req' else st.constraints)[i]
                tgt = row['item'] if kind == 'req' else row
                if op in RQ.OPS:
                    tgt['op'] = op
                tgt['target' if kind == 'req' else 'value'] = val
                if row.get('source'):
                    row['source'] = dict(row['source'], edited=True)
            self._edit(fn, f'Target of "{(r.get("item") or r).get("text", "")[:40]}" set to {val:g}')
            return
        if col == 5:
            pr = it.text().strip().lower()
            if pr not in RQ.PRIORITIES:
                self.ctx.msg('Priority is must or should.', 4000)
                self._fill_requirements()
                return

            def fn(st):
                row = (st.requirements if kind == 'req' else st.constraints)[i]
                (row['item'] if kind == 'req' else row)['priority'] = pr
            self._edit(fn, f'Priority of "{(r.get("item") or r).get("text", "")[:40]}" set to {pr}')

    # ------------------------------------------------------------------ limits and prices
    def _set_limit(self, kind, value):
        if self._filling:
            return
        value = float(value)
        unit = {'budget': self.settings.currency, 'tolerance_mm': 'mm'}.get(kind, '')
        text = {'budget': f'Budget under {value:g} {unit}', 'tolerance_mm': f'Survives a {value:g} mm mounting '
                                                                           'tolerance',
                'max_cameras': f'At most {value:g} cameras'}[kind]

        def fn(st):
            st.constraints = [c for c in st.constraints if not (c['kind'] == kind and c.get('origin') == 'user')]
            for c in st.constraints:
                if c['kind'] == kind:
                    c['accepted'] = False           # the page's figure replaces what was read
            if value > 0:
                st.constraints.append({'kind': kind, 'op': '<' if kind == 'budget' else
                                       ('==' if kind == 'tolerance_mm' else '<='), 'value': value, 'unit': unit,
                                       'priority': 'must', 'text': text, 'source': {}, 'accepted': True,
                                       'origin': 'user'})
        self._edit(fn, text if value > 0 else f'No {RQ.CONSTRAINT_LABEL[kind].lower()} limit')

    def _set_power(self, value):
        if self._filling:
            return
        value = float(value)

        def fn(st):
            st.requirements = [r for r in st.requirements if not (r['item']['metric'] == 'power.steady_w'
                                                                  and r.get('origin') == 'user')]
            for r in st.requirements:
                if r['item']['metric'] == 'power.steady_w':
                    r['accepted'] = False
            if value > 0:
                it = RQ.clean_item({'metric': 'power.steady_w', 'params': {}, 'op': '<', 'target': value,
                                    'priority': 'must', 'text': f'Under {value:g} W',
                                    'note': 'Entered on the design agent page.'},
                                   taken=[r['item']['id'] for r in st.requirements])
                st.requirements.append({'item': it, 'source': {}, 'accepted': True, 'origin': 'user',
                                        'metric_label': RQ.METRICS['power.steady_w'].label})
        self._edit(fn, f'Power limit {value:g} W' if value > 0 else 'No power limit')

    def _set_currency(self, cur):
        if self._filling or cur == self.settings.currency:
            return

        def fn(st):
            st.currency = cur
        self._edit(fn, f'Design costs in {cur}')

    def _set_search(self, key, value):
        if self._filling or getattr(self.settings, key) == value:
            return

        def fn(st):
            setattr(st, key, value)
        self._edit(fn, f'Design search: {key.replace("_", " ")} {value}')

    def _price_items(self):
        """[(key, name, kind)]: the current rig's hardware, the options', and with 'whole library' every item."""
        p = self.ctx.store.project
        keys = {}
        rigs = [RD.Rig.from_variant(p.variant)]
        if self.res is not None:
            rigs += [RD.Rig(o['rig']['cams'], o['rig']['sensors']) for o in self.res.options]
        for rig in rigs:
            for ln in RD.bill(p, rig, {})['lines']:
                keys[ln['key']] = ln['what']
        if self.c_all.isChecked():
            for m in p.modules.values():
                keys.setdefault(RD.price_key('module', m.id), f'{m.vendor} {m.name}'.strip())
            for ln in p.lenses.values():
                keys.setdefault(RD.price_key('lens', ln.id), ln.name + (f' ({ln.label})' if ln.label else ''))
            for m in p.sensor_models.values():
                keys.setdefault(RD.price_key('sensor', m.id), f'{m.vendor} {m.name}'.strip())
        order = {'module': 0, 'lens': 1, 'sensor': 2}
        return sorted(((k, v, k.split(':', 1)[0]) for k, v in keys.items()), key=lambda t: (order[t[2]], t[1]))

    def _fill_prices(self):
        st = self.settings
        self._filling = True
        try:
            items = self._price_items()
            self.prices.setRowCount(len(items))
            for k, (key, name, kind) in enumerate(items):
                a = _item(name, key)
                a.setData(Qt.ItemDataRole.UserRole, key)
                self.prices.setItem(k, 0, a)
                self.prices.setItem(k, 1, _item(kind))
                v = st.prices.get(key)
                self.prices.setItem(k, 2, _item('' if v is None else f'{v:g}', 'Your price (empty: not priced)',
                                                True, editable=True))
        finally:
            self._filling = False

    def _price_edited(self, it):
        if self._filling or it.column() != 2:
            return
        key = self.prices.item(it.row(), 0).data(Qt.ItemDataRole.UserRole)
        text = it.text().strip().replace(',', '')
        if text:
            try:
                v = float(text)
                if not math.isfinite(v) or v < 0:
                    raise ValueError
            except ValueError:
                self.ctx.msg('A price is a number of at least 0 (empty: not priced).', 4000)
                self._fill_prices()
                return
        else:
            v = None
        self.set_price(key, v)

    def set_price(self, key, value):
        def fn(st):
            if value is None:
                st.prices.pop(key, None)
            else:
                st.prices[key] = float(value)
        self._edit(fn, f'Price of {key} ' + ('cleared' if value is None else f'set to {value:g}'))

    # ------------------------------------------------------------------ running
    def run(self, wait=False) -> bool:
        ctx = self.ctx
        p = ctx.store.project
        st = self.settings
        if self.busy:
            return False
        if not any(r.get('accepted', True) for r in st.requirements):
            ctx.msg('No accepted requirements: read them from the brief or add a limit first.', 5000)
            return False
        win = ctx.window
        snap = p.snapshot()
        eng = win._shared_engine() if win is not None and hasattr(win, '_shared_engine') else \
            ctx.analysis.engine.share()
        self._gen += 1
        gen = self._gen
        self.busy = True
        self.status_text = 'Designing…'
        self.bar.setValue(0)
        self._cancel = threading.Event()
        cancel = self._cancel

        def work(progress, cancel_fn):
            def prog(f, text=''):
                if progress is not None:
                    progress(f, text)
                self.sig.progress.emit(gen, f, text)
            return p, RD.run(snap, eng, st, progress=prog, cancel=lambda: cancel.is_set() or bool(cancel_fn()))
        if wait:
            try:
                self._done(gen, work(None, lambda: False))
            except Exception as e:
                self._failed(gen, f'{type(e).__name__}: {e}')
            return True
        if win is not None and hasattr(win, 'run_job'):
            ok = win.run_job(JOB, 'Design agent', work, lambda out: self._done(gen, out),
                             lambda m: self._failed(gen, m))
            if not ok:
                self.busy = False
                self.status_text = 'The previous run is still finishing: try again in a moment.'
            self._fill_state()
            return ok
        self._fill_state()

        def thread():
            try:
                out = work(None, lambda: gen != self._gen)
                self.sig.done.emit(gen, out)
            except RuntimeError:
                pass
            except Exception as e:
                try:
                    self.sig.failed.emit(gen, f'{type(e).__name__}: {e}\n{traceback.format_exc()}')
                except RuntimeError:
                    pass
        threading.Thread(target=thread, daemon=True).start()
        return True

    def cancel(self):
        self._cancel.set()
        win = self.ctx.window
        if win is not None and JOB in getattr(win, '_jobs', {}):
            win._jobs[JOB]['cancel'].set()

    def _progress(self, gen, f, text):
        if gen == self._gen:
            self.bar.setValue(int(f * 1000))
            if text:
                self.status_text = f'Designing… {text}'
                self.status.setText(self.status_text)

    def _done(self, gen, out):
        if gen != self._gen:
            return
        self.busy = False
        p, res = out if out is not None else (None, None)
        if res is None or p is not self.ctx.store.project:
            self.status_text = 'Cancelled' if res is None else ''
            self._fill_state()
            return
        self.res = res
        self.sel = res.options[0]['label'] if res.options else None
        s = res.search
        self.status_text = (f'{len(res.options)} options · {s["rigs_scored"]} rigs scored in {s["seconds"]:.0f} s'
                            + ('' if s['stopped'] == 'finished' else f' · stopped: {s["stopped"]}'))
        self.ctx.store.project.add_log('designer', self.status_text)
        self._fill_prices()
        self._fill_options()
        self._fill_state()

    def _failed(self, gen, msg):
        if gen < 0:
            if -gen != self._rgen:
                return
            self.reading = False
        elif gen != self._gen:
            return
        self.busy = False
        self.status_text = 'Failed: ' + msg.splitlines()[0]
        self._fill_state()

    # ------------------------------------------------------------------ options
    def _axes_picked(self):
        a = [cb.currentData() for cb in self.ax]
        if a[0] == a[1]:
            self._fill_options()
            return

        def fn(st):
            st.axes = a
        self._edit(fn, f'Design chart: {a[0]} against {a[1]}')

    def _fill_options(self):
        res = self.res
        for w in self.card_widgets:
            w.hide()
            w.deleteLater()
        self.card_widgets = []
        if res is None:
            self.chart.set([], None, self.settings.axes)
            self.search_note.setText('')
            self._fill_detail()
            return
        self.chart.set(res.options, res.current, [cb.currentData() for cb in self.ax], self.sel)
        for o in res.options:
            card = OptionCard(self, o, res.current)
            self.cards.insertWidget(self.cards.count() - 1, card)
            self.card_widgets.append(card)
        s = res.search
        cur = res.current
        self.search_note.setText(
            f'Detection: {res.detection_text}. Current variant: {RD.option_line(dict(cur, label="current"), self.settings.currency)}. '
            f'Screened on {", ".join(s["screening_metrics"]) or "the cheap metrics"}; '
            + (f'{", ".join(s["front_only_metrics"])} on the front only. ' if s['front_only_metrics'] else '')
            + (' '.join(res.notes[:3])))
        self._fill_detail()

    def select_option(self, label):
        self.sel = label
        self.pending = None
        self.approve_box.hide()
        if self.res is not None:
            self.chart.set(self.res.options, self.res.current, [cb.currentData() for cb in self.ax], label)
        self._fill_detail()
        self._fill_state()

    def _fill_detail(self):
        res = self.res
        opt = res.option(self.sel) if res is not None and self.sel else None
        if opt is None:
            self.d_title.setText('No option selected')
            self.d_sub.setText('')
            for t in (self.cmp, self.sens):
                t.setRowCount(0)
            self.crit.clear()
            return
        cur = res.current
        self.d_title.setText(f'Option {opt["label"]} · ' + {'pass': 'meets every must requirement',
                                                             'fail': 'misses a must requirement',
                                                             'incomplete': 'not fully checked'}[opt['verdict']])
        rob = opt.get('robust')
        self.d_sub.setText(f'{opt["origin"]}. ' + ('; '.join(opt['changes'][:8]) or 'no change') + '.'
                           + (f' Robust: {"survives" if rob["ok"] else "fails"} {rob["method"]}.' if rob else ''))
        curv = {r['id']: r for r in cur['items']}
        rows = list(opt['items'])
        rows += [{'id': c['kind'], 'text': c['text'], 'value': c['actual'], 'margin': c['margin'],
                  'status': c['status'], 'unit': c.get('unit', ''), 'kind': 'constraint'} for c in opt['constraints']]
        curc = {c['kind']: c for c in cur['constraints']}
        rows += [{'id': '_cost', 'text': 'Cost', 'value': opt['cost'], 'margin': None, 'status': '',
                  'unit': self.settings.currency, 'kind': 'figure'},
                 {'id': '_power', 'text': 'Power (steady)', 'value': opt['power_w'], 'margin': None, 'status': '',
                  'unit': 'W', 'kind': 'figure'},
                 {'id': '_det', 'text': 'Detection' + (' (geometric)' if opt['detection_kind'] != 'learned' else ''),
                  'value': opt['detection'] * 100, 'margin': None, 'status': '', 'unit': '%', 'kind': 'figure'}]
        figs = {'_cost': cur['cost'], '_power': cur['power_w'], '_det': cur['detection'] * 100}
        self.cmp.setRowCount(len(rows))
        for k, r in enumerate(rows):
            if r.get('kind') == 'constraint':
                cv = curc.get(r['id'], {}).get('actual')
            elif r.get('kind') == 'figure':
                cv = figs[r['id']]
            else:
                cv = curv.get(r['id'], {}).get('value')
            st = r.get('status') or ''
            col = STATUS['good'] if st == 'pass' else STATUS['critical'] if st == 'fail' else None
            self.cmp.setItem(k, 0, _item(r['text']))
            self.cmp.setItem(k, 1, _item(_fmt(cv, r.get('unit', '')), mono_=True))
            self.cmp.setItem(k, 2, _item(_fmt(r.get('value'), r.get('unit', '')), mono_=True))
            self.cmp.setItem(k, 3, _item('' if r.get('margin') is None else f'{r["margin"]:+.3g}', mono_=True))
            self.cmp.setItem(k, 4, _item(RQ.STATUS_TEXT.get(st, st), color=col))
        self.sens.setRowCount(len(opt['sensors']))
        for k, s in enumerate(opt['sensors']):
            self.sens.setItem(k, 0, _item(s['tag'], mono_=True))
            self.sens.setItem(k, 1, _item(s['what']))
            self.sens.setItem(k, 2, _item(s['serves_text'] or '—'))
            self.sens.setItem(k, 3, _item(s['why']))
        self.crit.clear()
        for c in opt['critique']:
            self.crit.addItem(f'{c["severity"]}: {c["message"]}')
            self.crit.item(self.crit.count() - 1).setForeground(QColor(SEV_COLOR.get(c['severity'], '#898781')))
        if not opt['critique']:
            self.crit.addItem('Nothing found.')

    # ------------------------------------------------------------------ accepting an option
    def create_variant(self, label):
        res = self.res
        opt = res.option(label) if res is not None and label else None
        if opt is None:
            return None
        self.select_option(label)
        if RD.needs_approval(opt):
            self.pending = label
            self.approve_title.setText(f'Option {label} changes {len(opt["changes"])} things: approve the change set '
                                       'to create it')
            self.approve_list.clear()
            for ch in opt['changes']:
                self.approve_list.addItem(ch)
            self.approve_box.show()
            return None
        return self._create(label, approve=False)

    def _approve_cancel(self):
        self.pending = None
        self.approve_box.hide()

    def _approve_and_create(self):
        if self.pending is None:
            return None
        label, self.pending = self.pending, None
        self.approve_box.hide()
        return self._create(label, approve=True)

    def _create(self, label, approve):
        res = self.res
        opt = res.option(label)
        why = self.why.text().strip()
        out = {}

        def fn(p):
            if approve:
                RD.approve(p, opt)
            out['v'] = RD.accept(p, res, label, why)
        try:
            self.ctx.store.edit(['variants', 'analysis'], fn, f'Created design option {label} as a variant')
        except ValueError as e:
            self.ctx.msg(str(e), 6000)
            return None
        v = out['v']
        win = self.ctx.window
        if win is not None and hasattr(win, '_fill_variant_combo'):
            win._fill_variant_combo()
        self.why.clear()
        self.ctx.msg(f'Created variant {v.name} from option {label} (Edit ▸ Undo removes it)', 6000)
        self._fill_state()
        return v

    def open_compare(self):
        win = self.ctx.window
        if win is not None and hasattr(win, 'set_page'):
            win.set_page('compare')

    # ------------------------------------------------------------------ export
    def export(self, folder=None):
        if self.res is None:
            return []
        if folder is None:
            folder = QFileDialog.getExistingDirectory(self, 'Export the design options into')
            if not folder:
                return []
        paths = RD.export(self.res, self.ctx.store.project, folder)
        try:
            from ...core import provenance as PV
            paths = PV.stamp_paths(paths, self.ctx.store.project, folder)
        except Exception:
            pass
        self.ctx.msg(f'Exported the design options to {folder}', 5000)
        return paths
