"""Analysis ▸ Requirements scorecard: the programme's written requirements as pass/fail checks on the app's own studies,
one verdict, the evidence behind each value, and the design-review pack.

Cheap metrics (no geometry) are evaluated again on every edit while the page shows; study metrics when the scorecard
runs (a main-window job with Cancel), each study once for all requirements that need it. After an edit to the design a
study metric keeps its last value, marked stale, until the next run; editing a requirement alone keeps the studies
(the target is compared again at once)."""
from __future__ import annotations

import threading
import traceback

from PyQt6.QtCore import QObject, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import (QAbstractItemView, QComboBox, QFileDialog, QGridLayout, QHBoxLayout, QHeaderView,
                             QLineEdit, QMenu, QProgressBar, QScrollArea, QSplitter, QTableWidget, QTableWidgetItem,
                             QVBoxLayout, QWidget, QCheckBox)

from ...core import requirements as RQ
from .. import icons
from ..theme import THEME, STATUS, mono, sans
from ..widgets import ROW_H, ScrubSpinBox, SegmentedControl, button, hline, label
from .recording_ws import Tile

JOB = 'requirements'
COLS = ('ID', 'Requirement', 'Metric', 'Target', 'Value', 'Margin', 'Status', 'Priority')
STATUS_SEV = {'pass': 'good', 'fail': 'critical'}
VERDICT_SEV = {'pass': 'good', 'fail': 'critical', 'incomplete': 'warning'}


class _Sig(QObject):
    done = pyqtSignal(int, object)
    failed = pyqtSignal(int, str)
    progress = pyqtSignal(int, float)


def _combo(min_chars=14) -> QComboBox:
    """A combo box that does not widen its panel to its longest entry."""
    cb = QComboBox()
    cb.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
    cb.setMinimumContentsLength(min_chars)
    return cb


def _table(cols):
    t = QTableWidget(0, len(cols))
    t.setHorizontalHeaderLabels(cols)
    t.verticalHeader().setVisible(False)
    t.verticalHeader().setDefaultSectionSize(ROW_H)
    t.horizontalHeader().setFixedHeight(ROW_H)
    t.setShowGrid(False)
    t.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    return t


class RequirementsPage(QWidget):
    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        self.full: RQ.RequirementsResult | None = None     # the last run (study results inside)
        self.res: RQ.RequirementsResult | None = None      # what is shown: the last run evaluated again after edits
        self.proposals = []                                # from the brief, under review
        self.sel_id = None                                 # the selected requirement
        self.edit_id = None                                # the requirement in the editor (None: a new one)
        self.busy = False
        self.status_text = ''
        self._gen = 0
        self._dirty = True
        self._filling = False
        self.sig = _Sig()
        self.sig.done.connect(self._done)
        self.sig.failed.connect(self._failed)
        self.sig.progress.connect(lambda g, f: self.bar.setValue(int(f * 1000)) if g == self._gen else None)
        self.setObjectName('base')
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        split = QSplitter(Qt.Orientation.Horizontal)
        split.setHandleWidth(2)
        split.addWidget(self._left())
        split.addWidget(self._centre())
        split.addWidget(self._right())
        split.setSizes([320, 900, 380])
        lay.addWidget(split)
        ctx.store.changed.connect(self._on_change)
        ctx.store.project_replaced.connect(self._reset)
        THEME.changed.connect(self._retheme)
        self._new_item()
        self.refresh()

    # ------------------------------------------------------------------ layout
    def _left(self):
        w = QWidget()
        w.setObjectName('panel')
        ll = QVBoxLayout(w)
        ll.setContentsMargins(8, 6, 8, 8)
        ll.setSpacing(4)
        top = QHBoxLayout()
        top.addWidget(label('Requirements'))
        top.addStretch(1)
        top.addWidget(label('scorecard · review pack', muted=True, small=True))
        ll.addLayout(top)
        ll.addWidget(hline())
        ll.addWidget(label('The programme\'s requirements as measurable checks on this app\'s own studies: each with '
                           'its value, its margin (positive is good) and the figures behind it. The verdict passes '
                           'when every must requirement passes.', muted=True, small=True, wrap=True))
        ll.addSpacing(6)
        ll.addWidget(label('Evaluate', bold=True))
        self.variant = _combo()
        self.variant.setToolTip('The variant the requirements are evaluated against')
        self.variant.activated.connect(self._variant_picked)
        ll.addWidget(self.variant)
        ll.addSpacing(6)
        ll.addWidget(label('Add requirements', bold=True))
        self.b_new = button('New requirement', None, 'plus', 'A blank requirement in the editor on the right')
        self.b_new.clicked.connect(self._new_item)
        self.b_plan = button('Add from the plan', None, None,
                             'Requirements implied by what the plan already sets: the perception range\'s required '
                             'ranges, the minimum density, the recording shift, the blur limit …')
        self.b_plan.clicked.connect(self.add_from_plan)
        self.b_templates = button('Templates', None, 'chev-d', 'Typical data-rig requirements to add and edit')
        self.tmenu = QMenu(self.b_templates)
        self.tmenu.aboutToShow.connect(self._fill_templates)
        self.b_templates.setMenu(self.tmenu)
        self.b_brief = button('Suggest from the brief', None, 'sparkle',
                              'Read the project brief (/brief in the assistant) for requirements written with a '
                              'number and a unit, and propose them for review')
        self.b_brief.clicked.connect(self.suggest_from_brief)
        for b in (self.b_new, self.b_plan, self.b_templates, self.b_brief):
            ll.addWidget(b)
        # ---- proposals from the brief, for review
        self.prop_box = QWidget()
        pl = QVBoxLayout(self.prop_box)
        pl.setContentsMargins(0, 6, 0, 0)
        pl.setSpacing(3)
        self.prop_title = label('From the brief', bold=True)
        pl.addWidget(self.prop_title)
        pl.addWidget(label('Tick what to add; nothing is added until you do.', muted=True, small=True, wrap=True))
        self.prop_host = QVBoxLayout()
        self.prop_host.setSpacing(2)
        pl.addLayout(self.prop_host)
        pr = QHBoxLayout()
        self.b_accept = button('Add ticked', 'primary')
        self.b_accept.clicked.connect(self.accept_proposals)
        self.b_dismiss = button('Dismiss', 'ghost')
        self.b_dismiss.clicked.connect(lambda: self._set_proposals([]))
        pr.addWidget(self.b_dismiss)
        pr.addStretch(1)
        pr.addWidget(self.b_accept)
        pl.addLayout(pr)
        self.prop_checks, self.prop_rows, self.prop_origin = [], [], 'the brief'
        self.prop_box.hide()
        ll.addWidget(self.prop_box)
        ll.addStretch(1)
        self.status = label('', muted=True, small=True, wrap=True)
        ll.addWidget(self.status)
        self.bar = QProgressBar()
        self.bar.setTextVisible(False)
        self.bar.setRange(0, 1000)
        self.bar.hide()
        ll.addWidget(self.bar)
        bl = QHBoxLayout()
        bl.addStretch(1)
        self.b_export = button('Export…', None, 'export', 'The scorecard as JSON and CSV, and the design-review pack '
                                                           '(PDF)')
        self.b_export.clicked.connect(lambda: self.export())
        self.b_run = button('Run', 'primary', None, 'Evaluate every requirement: each study a requirement needs runs '
                                                    'once')
        self.b_run.clicked.connect(lambda: self.run())
        bl.addWidget(self.b_export)
        bl.addWidget(self.b_run)
        ll.addLayout(bl)
        sc = QScrollArea()
        sc.setWidgetResizable(True)
        sc.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        sc.setWidget(w)
        sc.setMinimumWidth(260)
        return sc

    def _centre(self):
        w = QWidget()
        w.setObjectName('panel')
        cl = QVBoxLayout(w)
        cl.setContentsMargins(10, 8, 10, 8)
        cl.setSpacing(8)
        tiles = QHBoxLayout()
        tiles.setSpacing(8)
        self.tiles = {}
        for key, caption, tip in (
                ('verdict', 'Verdict', 'Pass when every must requirement passes; fail when one fails; not evaluated '
                                       'when one cannot be evaluated'),
                ('must', 'Must', 'Must requirements passed'),
                ('should', 'Should', 'Should requirements passed (never decisive)'),
                ('open', 'Not evaluated', 'Requirements without a value: not run yet, or the metric cannot be '
                                          'evaluated (the evidence says why)')):
            t = self.tiles[key] = Tile(caption, tip)
            tiles.addWidget(t, 1)
        cl.addLayout(tiles)
        head = QHBoxLayout()
        head.addWidget(label('Scorecard', bold=True))
        head.addStretch(1)
        self.b_dup = button('Duplicate', 'ghost', 'copy', 'A copy of the selected requirement')
        self.b_dup.clicked.connect(self.duplicate_selected)
        self.b_del = button('Delete', 'ghost', 'trash', 'Delete the selected requirement')
        self.b_del.clicked.connect(self.delete_selected)
        head.addWidget(self.b_dup)
        head.addWidget(self.b_del)
        cl.addLayout(head)
        self.table = _table(COLS)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setWordWrap(False)
        hh = self.table.horizontalHeader()
        for c in range(len(COLS)):
            hh.setSectionResizeMode(c, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        hh.setSectionResizeMode(2, QHeaderView.ResizeMode.Interactive)
        hh.resizeSection(2, 250)
        self.table.setTextElideMode(Qt.TextElideMode.ElideRight)
        self.table.itemSelectionChanged.connect(self._row_picked)
        cl.addWidget(self.table, 1)
        self.empty = label('No requirements yet: add from the plan, pick a template, suggest from the brief or write '
                           'a new one.', muted=True, small=True, wrap=True)
        cl.addWidget(self.empty)
        return w

    def _right(self):
        w = QWidget()
        w.setObjectName('panel')
        rl = QVBoxLayout(w)
        rl.setContentsMargins(8, 6, 8, 8)
        rl.setSpacing(4)
        eh = QHBoxLayout()
        self.ed_title = label('New requirement', bold=True)
        eh.addWidget(self.ed_title)
        eh.addStretch(1)
        rl.addLayout(eh)
        g = QGridLayout()
        g.setVerticalSpacing(3)
        g.setHorizontalSpacing(6)
        g.setColumnStretch(1, 1)

        def row(text, wdg, tip=''):
            a = label(text, ink2=True)
            a.setFixedWidth(78)
            if tip:
                a.setToolTip(tip)
                wdg.setToolTip(tip)
            r = g.rowCount()
            g.addWidget(a, r, 0)
            g.addWidget(wdg, r, 1)
        self.ed_text = QLineEdit()
        self.ed_text.setPlaceholderText('As written in the programme (empty: from the metric)')
        row('Text', self.ed_text, 'The requirement as written')
        self.ed_metric = _combo()
        self.ed_metric.setMaxVisibleItems(24)
        self.ed_metric.currentIndexChanged.connect(self._metric_changed)
        row('Metric', self.ed_metric, 'What is measured, and the study that computes it')
        rl.addLayout(g)
        self.param_grid = QGridLayout()
        self.param_grid.setVerticalSpacing(3)
        self.param_grid.setHorizontalSpacing(6)
        self.param_grid.setColumnStretch(1, 1)
        rl.addLayout(self.param_grid)
        self.param_boxes = {}
        g2 = QGridLayout()
        g2.setVerticalSpacing(3)
        g2.setHorizontalSpacing(6)
        g2.setColumnStretch(3, 1)
        a = label('Target', ink2=True)
        a.setFixedWidth(78)
        self.ed_op = QComboBox()
        for op in RQ.OPS:
            self.ed_op.addItem(RQ.OP_TEXT[op], op)
        self.ed_op.setFixedWidth(52)
        self.ed_op.setToolTip('≥ at least · ≤ at most · = exactly · < less than · > more than')
        self.ed_target = ScrubSpinBox('', 'plain', 4, '', width=110)
        self.ed_unit = label('', muted=True, mono_=True)
        g2.addWidget(a, 0, 0)
        g2.addWidget(self.ed_op, 0, 1)
        g2.addWidget(self.ed_target, 0, 2)
        g2.addWidget(self.ed_unit, 0, 3)
        b = label('Priority', ink2=True)
        b.setFixedWidth(78)
        self.ed_prio = SegmentedControl([('must', 'Must'), ('should', 'Should')], 'must')
        self.ed_prio.setToolTip('A must requirement decides the verdict; a should requirement is counted only')
        g2.addWidget(b, 1, 0)
        g2.addWidget(self.ed_prio, 1, 1, 1, 3)
        n = label('Note', ink2=True)
        n.setFixedWidth(78)
        self.ed_note = QLineEdit()
        self.ed_note.setPlaceholderText('Where it comes from, who asked for it')
        g2.addWidget(n, 2, 0)
        g2.addWidget(self.ed_note, 2, 1, 1, 3)
        rl.addLayout(g2)
        self.ed_help = label('', muted=True, small=True, wrap=True)
        rl.addWidget(self.ed_help)
        self.ed_msg = label('', small=True, wrap=True)
        self.ed_msg.hide()
        rl.addWidget(self.ed_msg)
        bl = QHBoxLayout()
        self.b_revert = button('Revert', 'ghost', None, 'Back to what is stored')
        self.b_revert.clicked.connect(lambda: self._load_editor(self.edit_id))
        self.b_save = button('Add', 'primary', None, 'Save the requirement (undoable)')
        self.b_save.clicked.connect(self.save_editor)
        bl.addWidget(self.b_revert)
        bl.addStretch(1)
        bl.addWidget(self.b_save)
        rl.addLayout(bl)
        rl.addSpacing(8)
        rl.addWidget(hline())
        # ---- the evidence of the selected requirement
        ev = QHBoxLayout()
        ev.addWidget(label('Evidence', bold=True))
        ev.addStretch(1)
        self.b_open = button('Open the study', 'ghost', 'external', 'Show the study this value comes from')
        self.b_open.clicked.connect(self.open_study)
        ev.addWidget(self.b_open)
        rl.addLayout(ev)
        self.ev_head = label('Select a requirement to see where its value comes from.', muted=True, small=True,
                             wrap=True)
        rl.addWidget(self.ev_head)
        self.ev_table = _table(('Figure', 'Value'))
        self.ev_table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.ev_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.ev_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.ev_table.setWordWrap(True)
        rl.addWidget(self.ev_table, 1)
        self.ev_detail = label('', small=True, wrap=True)
        rl.addWidget(self.ev_detail)
        sc = QScrollArea()
        sc.setWidgetResizable(True)
        sc.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        sc.setWidget(w)
        sc.setMinimumWidth(320)
        self._fill_metrics()
        return sc

    # ------------------------------------------------------------------ settings (saved with the project, undoable)
    @property
    def settings(self) -> RQ.RequirementsSettings:
        return RQ.settings_of(self.ctx.store.project)

    def _store(self, st: RQ.RequirementsSettings, text) -> bool:
        new = RQ.RequirementsSettings.from_dict(st.to_dict()).to_dict()
        if new == self.settings.to_dict():
            return False

        def fn(p):
            p.analysis.requirements = new
        self.ctx.store.edit('analysis', fn, f'Requirements · {text}')
        return True

    def _add_items(self, items, text) -> list:
        """Add requirements (new ids) in one undo step; returns their ids."""
        st = self.settings
        ids = []
        for it in items:
            c = RQ.clean_item(dict(it, id=''), self.ctx.store.project, [i['id'] for i in st.items])
            if c is None:
                continue
            st.items.append(c)
            ids.append(c['id'])
        if ids:
            self._store(st, text)
        return ids

    def _variant_picked(self, i):
        uid = self.variant.itemData(i) or ''
        st = self.settings
        if uid != st.variant:
            st.variant = uid
            name = self.variant.itemText(i)
            self._store(st, f'evaluate {name}')

    # ------------------------------------------------------------------ adding
    def add_from_plan(self):
        p = self.ctx.store.project
        st = self.settings
        v = p.variant_by_uid(st.variant) or p.variant
        have = {(i['metric'], tuple(sorted(i['params'].items()))) for i in st.items}
        new = [it for it in RQ.from_plan(p, v)
               if (it['metric'], tuple(sorted(RQ.clean_item(it, p)['params'].items()))) not in have]
        n_old = len(RQ.from_plan(p, v)) - len(new)
        ids = self._add_items(new, f'{len(new)} from the plan')
        self.ctx.msg(f'Added {len(ids)} requirement{"s" * (len(ids) != 1)} from the plan'
                     + (f' ({n_old} already there)' if n_old else '') + '.', 5000)
        if ids:
            self.select(ids[0])
        return ids

    def _fill_templates(self):
        self.tmenu.clear()
        group = None
        for i, t in enumerate(RQ.templates(self.ctx.store.project)):
            sid = RQ.METRICS[t['metric']].study
            if sid != group:
                group = sid
                self.tmenu.addSection(RQ.STUDIES[sid].label)
            a = self.tmenu.addAction(t['text'])
            a.triggered.connect(lambda _=False, k=i: self.add_template(k))

    def add_template(self, k):
        ts = RQ.templates(self.ctx.store.project)
        if not 0 <= k < len(ts):
            return None
        ids = self._add_items([ts[k]], f'template "{ts[k]["text"]}"')
        if ids:
            self.select(ids[0])
            self.ctx.msg(f'Added {ids[0]}: {ts[k]["text"]}. Edit its target on the right.', 4000)
        return ids[0] if ids else None

    def suggest_from_brief(self):
        p = self.ctx.store.project
        brief = str((p.analysis.assistant or {}).get('brief', '')).strip()
        if not brief:
            self._set_proposals([])
            self.ctx.msg('No project brief yet: type /brief in the assistant to write one.', 6000)
            return []
        have = {(i['metric'], tuple(sorted(i['params'].items())), i['op'], i['target']) for i in self.settings.items}
        props = [o for o in RQ.parse_brief(brief, p)
                 if (o['item']['metric'], tuple(sorted(o['item']['params'].items())), o['item']['op'],
                     o['item']['target']) not in have]
        self._set_proposals(props)
        self.ctx.msg(f'{len(props)} requirement{"s" * (len(props) != 1)} found in the brief' if props else
                     'Found no requirement with a number and a unit in the brief (or they are all in the list).', 5000)
        return props

    def show_proposals(self, items, source='Proposed by the assistant') -> int:
        """Requirements proposed from elsewhere (the assistant) put up for review like the brief's: only the valid
        ones, nothing added until the user ticks and adds them. Returns how many are shown."""
        p = self.ctx.store.project
        props = []
        for raw in items or []:
            it = RQ.clean_item(raw, p)
            if it is None or not RQ.validate_item(it, p)[0]:
                continue
            it['note'] = it['note'] or source + '.'
            props.append({'item': it, 'source': it['text'], 'metric_label': RQ.METRICS[it['metric']].label})
        self._set_proposals(props, 'the assistant')
        return len(props)

    def _set_proposals(self, props, origin='the brief'):
        self.proposals = list(props)
        self.prop_origin = origin
        self.prop_title.setText(f'From {origin}')
        for row in self.prop_rows:
            self.prop_host.removeWidget(row)
            row.hide()
            row.deleteLater()
        self.prop_rows = []
        self.prop_checks = []
        for o in self.proposals:
            it = o['item']
            m = RQ.METRICS[it['metric']]
            row = QWidget()
            h = QHBoxLayout(row)
            h.setContentsMargins(0, 0, 0, 0)
            h.setSpacing(6)
            cb = QCheckBox()
            cb.setChecked(True)
            prm = RQ._params_text(it, self.ctx.store.project) if it['params'] else ''
            text = label(f'“{o["source"]}”: {m.label}' + (f' ({prm})' if prm else '')
                         + f' {RQ.OP_TEXT[it["op"]]} {RQ.fmt_value(it["target"], m)}, {it["priority"]}',
                         small=True, wrap=True)
            text.mousePressEvent = lambda e, c=cb: c.toggle()          # the text ticks the box as a label does
            h.addWidget(cb, 0, Qt.AlignmentFlag.AlignTop)
            h.addWidget(text, 1)
            self.prop_host.addWidget(row)
            self.prop_checks.append(cb)
            self.prop_rows.append(row)
        self.prop_box.setVisible(bool(self.proposals))

    def accept_proposals(self):
        pick = [o['item'] for o, cb in zip(self.proposals, self.prop_checks) if cb.isChecked()]
        origin = self.prop_origin
        ids = self._add_items(pick, f'{len(pick)} from {origin}') if pick else []
        self._set_proposals([])
        if ids:
            self.select(ids[0])
            self.ctx.msg(f'Added {len(ids)} requirement{"s" * (len(ids) != 1)} from {origin}.', 4000)
        return ids

    def duplicate_selected(self):
        it = next((i for i in self.settings.items if i['id'] == self.sel_id), None)
        if it is None:
            return None
        ids = self._add_items([dict(it, text=it['text'])], f'copy of {it["id"]}')
        if ids:
            self.select(ids[0])
        return ids[0] if ids else None

    def delete_selected(self):
        st = self.settings
        it = next((i for i in st.items if i['id'] == self.sel_id), None)
        if it is None:
            return False
        row = st.items.index(it)
        st.items.remove(it)
        self._store(st, f'delete {it["id"]}')
        if st.items:
            self.select(st.items[min(row, len(st.items) - 1)]['id'])
        else:
            self.sel_id = None
            self._new_item()
        return True

    # ------------------------------------------------------------------ the editor
    def _fill_metrics(self):
        cb = self.ed_metric
        cb.blockSignals(True)
        cb.clear()
        group = None
        for m in RQ.METRICS.values():
            if not RQ.available(m.study):
                continue
            if m.study != group:
                group = m.study
                cb.addItem(RQ.STUDIES[m.study].label.upper(), None)
                cb.model().item(cb.count() - 1).setEnabled(False)
            cb.addItem(f'  {m.label}' + (f' ({m.unit})' if m.unit else ''), m.id)
        cb.blockSignals(False)

    def _metric(self) -> RQ.Metric | None:
        return RQ.METRICS.get(self.ed_metric.currentData())

    def _metric_changed(self, *_):
        if self._filling:
            return
        m = self._metric()
        if m is None:
            return
        self._build_params(m, {})
        self.ed_op.setCurrentIndex(RQ.OPS.index(m.better))
        self._unit(m)

    def _unit(self, m):
        self.ed_unit.setText(m.unit)
        self.ed_target.setDecimals(max(2, m.decimals + 1))
        s = RQ.STUDIES[m.study]
        self.ed_help.setText(f'{m.help}. From the {s.label.lower()} ({s.menu})'
                             + ('; cheap: evaluated on every edit.' if s.cost == 'cheap' else
                                '; evaluated when the scorecard runs.'))

    def _build_params(self, m, values):
        while self.param_grid.count():
            wdg = self.param_grid.takeAt(0).widget()
            if wdg is not None:
                wdg.hide()
                wdg.deleteLater()
        self.param_boxes = {}
        p = self.ctx.store.project
        v = p.variant_by_uid(self.settings.variant) or p.variant
        for r, prm in enumerate(m.params):
            a = label(prm.label, ink2=True)
            a.setFixedWidth(78)
            box = _combo(10)
            opts = prm.options(p, v)
            for val, lab in opts:
                box.addItem(lab, val)
            want = values.get(prm.key, prm.default)
            i = box.findData(want)
            if i < 0:
                low = [str(x).lower() for x, _ in opts]
                if str(want).lower() in low:
                    i = low.index(str(want).lower())
                else:                       # a tag or class the project does not have (any more): keep it, say so
                    box.addItem(f'{want} (not in the project)', want)
                    i = box.count() - 1
            box.setCurrentIndex(i)
            self.param_grid.addWidget(a, r, 0)
            self.param_grid.addWidget(box, r, 1)
            self.param_boxes[prm.key] = box

    def _new_item(self):
        self.edit_id = None
        self.ed_title.setText('New requirement')
        self.b_save.setText('Add')
        self.b_revert.setEnabled(False)
        self._filling = True
        try:
            self.ed_text.setText('')
            self.ed_note.setText('')
            i = next((k for k in range(self.ed_metric.count()) if self.ed_metric.itemData(k)), 0)
            self.ed_metric.setCurrentIndex(i)
        finally:
            self._filling = False
        m = self._metric()
        if m is not None:
            self._build_params(m, {})
            self.ed_op.setCurrentIndex(RQ.OPS.index(m.better))
            self._unit(m)
        self.ed_target.set_value(0.0)
        self.ed_prio.set_value('must')
        self._editor_msg('')

    def _load_editor(self, iid):
        it = next((i for i in self.settings.items if i['id'] == iid), None) if iid else None
        if it is None:
            self._new_item()
            return
        self.edit_id = iid
        self.ed_title.setText(f'Requirement {iid}')
        self.b_save.setText('Save')
        self.b_revert.setEnabled(True)
        self._filling = True
        try:
            self.ed_text.setText(it['text'])
            self.ed_note.setText(it.get('note', ''))
            self.ed_text.setCursorPosition(0)         # a long text shows from its start, not its end
            self.ed_note.setCursorPosition(0)
            i = self.ed_metric.findData(it['metric'])
            if i < 0:                       # a metric this build does not have: shown, not editable into another
                self.ed_metric.addItem(f'{it["metric"]} (not in this build)', it['metric'])
                i = self.ed_metric.count() - 1
            self.ed_metric.setCurrentIndex(i)
        finally:
            self._filling = False
        m = RQ.METRICS.get(it['metric'])
        if m is not None:
            self._build_params(m, it['params'])
            self._unit(m)
        self.ed_op.setCurrentIndex(RQ.OPS.index(it['op']))
        self.ed_target.set_value(it['target'] if it['target'] is not None else 0.0)
        self.ed_prio.set_value(it['priority'])
        self._editor_msg('')

    def editor_item(self) -> dict:
        """The requirement as the editor shows it."""
        m = self._metric()
        return {'id': self.edit_id or '', 'text': self.ed_text.text().strip(),
                'metric': self.ed_metric.currentData() or '',
                'params': {k: b.currentData() if b.currentData() is not None else '' for k, b in
                           self.param_boxes.items()} if m is not None else {},
                'op': self.ed_op.currentData(), 'target': float(self.ed_target.value()),
                'priority': self.ed_prio.value() or 'must', 'note': self.ed_note.text().strip()}

    def _editor_msg(self, text, sev=None):
        self.ed_msg.setText(text)
        self.ed_msg.setStyleSheet(f'color: {STATUS[sev]};' if sev else '')
        self.ed_msg.setVisible(bool(text))

    def save_editor(self):
        """Add the editor's requirement, or replace the one it edits (one undo step). Returns its id, or None when
        it is not a valid requirement (the editor says why)."""
        p = self.ctx.store.project
        self.ed_target.interpretText()
        raw = self.editor_item()
        ok, why = RQ.validate_item(raw, p)
        if not ok:
            self._editor_msg(why, 'critical')
            return None
        st = self.settings
        if self.edit_id is None:
            ids = self._add_items([raw], f'add "{raw["text"] or RQ.describe(raw, p)}"')
            iid = ids[0] if ids else None
        else:
            idx = next((k for k, i in enumerate(st.items) if i['id'] == self.edit_id), None)
            if idx is None:
                self._editor_msg('This requirement is gone (deleted or undone).', 'warning')
                return None
            if not raw['text'] or raw['text'] == RQ.describe(st.items[idx], p):
                raw['text'] = ''           # a generated text follows the metric and target
            st.items[idx] = RQ.clean_item(raw, p)
            iid = self.edit_id
            if not self._store(st, f'edit {iid}'):
                self._editor_msg('Nothing changed.')
                return iid
        self._editor_msg('')
        if iid:
            self.select(iid)
        return iid

    # ------------------------------------------------------------------ selection and evidence
    def select(self, iid):
        self.sel_id = iid
        self._fill_table()
        self._load_editor(iid)
        self._fill_evidence()

    def _row_picked(self):
        if self._filling:
            return
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            return
        it = self.table.item(rows[0].row(), 0)
        iid = it.data(Qt.ItemDataRole.UserRole) if it is not None else None
        if iid and iid != self.sel_id:
            self.sel_id = iid
            self._load_editor(iid)
            self._fill_evidence()

    def _fill_evidence(self):
        r = self.res.item(self.sel_id) if self.res is not None and self.sel_id else None
        t = self.ev_table
        if r is None:
            self.ev_head.setText('Select a requirement to see where its value comes from.')
            t.setRowCount(0)
            self.ev_detail.setText('')
            self.b_open.setEnabled(False)
            return
        e = r.evidence
        m = r.metric
        head = f'{r.item["id"]} · {e.get("study_label", "—")} · {e.get("variant", "")}'
        if r.status in ('pass', 'fail'):
            head += (f'\n{RQ.fmt_value(r.value, m)} against {RQ.OP_TEXT[r.item["op"]]} '
                     f'{RQ.fmt_value(r.item["target"], m)}: margin {RQ.fmt_value(r.margin, m, signed=True)}')
        if r.stale:
            head += '\nStale: from the last run, before the design changed. Run again.'
        self.ev_head.setText(head)
        figs = e.get('figures') or []
        t.setRowCount(len(figs))
        for k, (a, b) in enumerate(figs):
            ia = QTableWidgetItem(str(a))
            ia.setForeground(THEME.color('ink2'))
            ib = QTableWidgetItem(str(b))
            ib.setToolTip(str(b))
            t.setItem(k, 0, ia)
            t.setItem(k, 1, ib)
        t.resizeRowsToContents()
        detail = r.reason or e.get('detail', '')
        if e.get('note'):
            detail = f'{detail} {e["note"]}'.strip()
        self.ev_detail.setText(detail)
        self.ev_detail.setStyleSheet(f'color: {STATUS["warning"]};' if r.reason else '')
        self.b_open.setEnabled(bool(e.get('study')))
        self.b_open.setText(f'Open {e.get("study_label", "the study").lower()}' if e.get('study') else
                            'Open the study')

    def open_study(self):
        r = self.res.item(self.sel_id) if self.res is not None and self.sel_id else None
        if r is None:
            return False
        page, menu = r.evidence.get('page', ''), r.evidence.get('menu', '')
        win = self.ctx.window
        if win is not None and page and page in getattr(win, 'PAGE_DOCKS', {}):
            win.set_page(page)
            return True
        self.ctx.msg(f'{r.evidence.get("study_label", "The study")}: {menu}', 6000)
        return False

    # ------------------------------------------------------------------ state
    def _on_change(self, sections=None):
        if self.isVisible():
            self.refresh()
        else:
            self._dirty = True

    def _reset(self, *_):
        self._gen += 1
        self.busy = False
        self.full = self.res = None
        self.status_text = ''
        self.sel_id = None
        self._set_proposals([])
        self._new_item()
        self._on_change()

    def _retheme(self, *_):
        self._fill_table()
        self._fill_evidence()

    def showEvent(self, e):
        super().showEvent(e)
        if self._dirty:
            self.refresh()

    def activate(self):
        if self._dirty:
            self.refresh()

    def refresh(self):
        """Evaluate again (cheap metrics; study metrics from the last run, stale when the design changed)."""
        self._dirty = False
        p = self.ctx.store.project
        self.res = RQ.reevaluate(p, self.full)
        ids = [i['id'] for i in self.settings.items]
        if self.sel_id not in ids:
            self.sel_id = None
        self._fill_variants()
        self._fill_table()
        self._fill_tiles()
        if self.edit_id is not None:
            if self.edit_id in ids:
                self._load_editor(self.edit_id)
            else:
                self._new_item()
        self._fill_evidence()
        self._fill_state()

    def _fill_variants(self):
        p = self.ctx.store.project
        want = [('', f'Active variant · {p.variant.name}')] + [(v.uid, v.name) for v in p.variants]
        have = [(self.variant.itemData(i), self.variant.itemText(i)) for i in range(self.variant.count())]
        if want != have:
            self.variant.blockSignals(True)
            self.variant.clear()
            for uid, name in want:
                self.variant.addItem(name, uid)
            self.variant.blockSignals(False)
        i = self.variant.findData(self.settings.variant)
        self.variant.setCurrentIndex(max(i, 0))

    def _fill_tiles(self):
        res = self.res
        if res is None or not res.items:
            self.tiles['verdict'].set('—', '', 'no requirements')
            for k in ('must', 'should', 'open'):
                self.tiles[k].set('—', '', '')
            return
        c = res.counts()
        v = res.verdict
        stale = sum(r.stale for r in res.items)
        self.tiles['verdict'].set(RQ.VERDICT_TEXT[v], '', f'{res.variant}' + (f' · {stale} stale' if stale else ''),
                                  VERDICT_SEV.get(v))
        for k in ('must', 'should'):
            d = c[k]
            sev = None if not d['total'] else 'good' if d['pass'] == d['total'] else \
                ('critical' if k == 'must' else 'warning') if d['fail'] else 'warning'
            self.tiles[k].set(f'{d["pass"]}/{d["total"]}', 'passed',
                              f'{d["fail"]} fail · {d["not_evaluated"]} not evaluated', sev if d['total'] else None)
        n_run = sum(r.status == 'not_run' for r in res.items)
        n_ne = sum(r.status == 'not_evaluated' for r in res.items)
        self.tiles['open'].set(str(n_run + n_ne), '', (f'{n_run} not run · ' if n_run else '') + f'{n_ne} cannot be '
                                                                                                f'evaluated')

    def _fill_table(self):
        res = self.res
        rows = res.items if res is not None else []
        t = self.table
        self._filling = True
        try:
            t.setRowCount(len(rows))
            right = Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            for k, r in enumerate(rows):
                m = r.metric
                it = r.item

                def cell(c, text, num=False, color=None, tip=None, icon=None, bold=False):
                    x = QTableWidgetItem(text)
                    if num:
                        x.setFont(mono(12))
                        x.setTextAlignment(right)
                    elif bold:
                        x.setFont(sans(12, QFont.Weight.DemiBold))
                    if color is not None:
                        x.setForeground(color)
                    if tip:
                        x.setToolTip(tip)
                    if icon is not None:
                        x.setIcon(icon)
                    t.setItem(k, c, x)
                    return x
                cell(0, it['id'], bold=True).setData(Qt.ItemDataRole.UserRole, it['id'])
                cell(1, it['text'], tip=it['text'] + (f'\n{it["note"]}' if it.get('note') else ''))
                prm = RQ._params_text(it, self.ctx.store.project) if it['params'] else ''
                cell(2, (m.label if m else it['metric']) + (f' · {prm}' if prm else ''), color=THEME.color('ink2'),
                     tip=(m.help if m else 'Not a metric of this build'))
                cell(3, f'{RQ.OP_TEXT[it["op"]]} {RQ.fmt_value(it["target"], m)}', True)
                muted = THEME.color('muted')
                cell(4, RQ.fmt_value(r.value, m) + (' *' if r.stale else ''), True, muted if r.stale else None,
                     'Stale: from the last run, before the design changed' if r.stale else None)
                mcol = None if r.margin is None else QColor(STATUS['good'] if r.margin >= 0 else STATUS['critical'])
                cell(5, RQ.fmt_value(r.margin, m, signed=True), True, muted if r.stale else mcol)
                sev = STATUS_SEV.get(r.status)
                cell(6, RQ.STATUS_TEXT.get(r.status, r.status), color=QColor(STATUS[sev]) if sev else muted,
                     tip=r.reason or None, icon=icons.status_icon(sev, 12) if sev else None)
                cell(7, it['priority'].capitalize(), color=THEME.color('ink' if it['priority'] == 'must' else 'ink2'))
            sel = next((k for k, r in enumerate(rows) if r.item['id'] == self.sel_id), None)
            if sel is not None:
                t.selectRow(sel)
            else:
                t.clearSelection()
        finally:
            self._filling = False
        self.empty.setVisible(not rows)
        self.b_del.setEnabled(self.sel_id is not None)
        self.b_dup.setEnabled(self.sel_id is not None)

    def _fill_state(self):
        win = self.ctx.window
        running = self.busy and win is not None and JOB in getattr(win, '_jobs', {})
        self.b_run.setEnabled(not running and bool(self.settings.items))
        self.bar.setVisible(self.busy)
        self.b_export.setEnabled(self.res is not None and bool(self.res.items))
        res = self.res
        text = self.status_text
        if res is not None and not self.busy and res.items:
            n_run = sum(r.status == 'not_run' for r in res.items)
            stale = sum(r.stale for r in res.items)
            if n_run or stale:
                text = (text + ' · ' if text else '') + (f'{n_run} need a run' if n_run else '') + \
                       (' · ' if n_run and stale else '') + (f'{stale} stale: the design changed' if stale else '')
        self.status.setText(text)

    # ------------------------------------------------------------------ running
    def run(self, quiet=False) -> bool:
        ctx = self.ctx
        p = ctx.store.project
        st = self.settings
        if not st.items:
            ctx.msg('No requirements to evaluate yet.', 4000)
            return False
        win = ctx.window
        if self.busy and win is not None and JOB in getattr(win, '_jobs', {}):
            return False
        prev = self.full
        uid = st.variant if p.variant_by_uid(st.variant) is not None else p.active_uid
        if prev is not None and prev.variant_uid == uid and prev.inputs_key == RQ.inputs_key(p, uid) and \
                not RQ._plan_studies(st.items + [{'metric': 'issues.count'}], {**prev.cache, **prev.errors}):
            res = RQ.run(p, None, st, reuse=prev, revision=ctx.store.revision)       # every study in hand: at once
            self._gen += 1
            self._done(self._gen, (p, res))
            return True
        project = p.snapshot()
        eng = win._shared_engine() if win is not None and hasattr(win, '_shared_engine') else \
            ctx.analysis.engine.share()
        self._gen += 1
        gen = self._gen
        self.busy = True
        self.status_text = 'Evaluating the requirements…'
        self.bar.setValue(0)
        rev = ctx.store.revision

        def work(progress, cancel):
            def prog(f, text=''):
                if progress is not None:
                    progress(f, text)
                self.sig.progress.emit(gen, f)
            return p, RQ.run(project, eng, st, progress=prog, cancel=cancel, reuse=prev, revision=rev)
        if not quiet and win is not None and hasattr(win, 'run_job'):
            ok = win.run_job(JOB, 'Requirements scorecard', work, lambda out: self._done(gen, out),
                             lambda m: self._failed(gen, m))
            if not ok:                      # an earlier run (for a project since replaced) is still finishing
                self.busy = False
                self.status_text = 'The previous run is still finishing: try again in a moment.'
            self._fill_state()
            return ok
        self._fill_state()

        def thread():
            try:
                out = work(None, lambda: gen != self._gen)
                self.sig.done.emit(gen, out)
            except RuntimeError:            # the window closed while computing
                pass
            except Exception as e:
                try:
                    self.sig.failed.emit(gen, f'{type(e).__name__}: {e}\n{traceback.format_exc()}')
                except RuntimeError:
                    pass
        threading.Thread(target=thread, daemon=True).start()
        return True

    def _done(self, gen, out):
        if gen != self._gen:
            return
        self.busy = False
        p, res = out if out is not None else (None, None)
        if res is None or p is not self.ctx.store.project:
            self.status_text = 'Cancelled' if res is None else ''
            self._fill_state()
            return
        self.full = res
        secs = sum(s['seconds'] for s in res.studies.values())
        bad = [s['label'] for s in res.studies.values() if s['error']]
        self.status_text = (f'{len(res.items)} requirements on {res.variant} · {len(res.studies)} studies in '
                            f'{secs:.1f} s' + (f' · not evaluated: {", ".join(bad)}' if bad else ''))
        self.ctx.store.project.add_log('requirements', f'{RQ.VERDICT_TEXT[res.verdict]} · {self.status_text}')
        self.refresh()

    def _failed(self, gen, msg):
        if gen != self._gen:
            return
        self.busy = False
        self.status_text = 'Failed: ' + msg.splitlines()[0]
        self._fill_state()

    # ------------------------------------------------------------------ export
    def export(self, folder=None):
        res = self.res
        if res is None or not res.items:
            return []
        if folder is None:
            folder = QFileDialog.getExistingDirectory(self, 'Export the requirements scorecard into')
            if not folder:
                return []
        from ...core import provenance as PV          # where the JSON came from
        paths = PV.stamp_paths(RQ.export(res, self.ctx.store.project, folder), self.ctx.store.project, folder)
        open_ = sum(r.status == 'not_run' or r.stale for r in res.items)
        self.ctx.msg(f'Exported the scorecard and the review pack to {folder}'
                     + (f' ({open_} requirements not run or stale: run first for a complete pack)' if open_ else ''),
                     6000)
        return paths
