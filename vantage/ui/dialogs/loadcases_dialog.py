"""Analysis ▸ Load cases: how passengers and luggage (body pitch and drop) move the stitched ground image."""
from __future__ import annotations

import threading
import traceback

from PyQt6.QtCore import Qt, QObject, pyqtSignal
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import (QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView,
                             QProgressBar)

from ...core.loadcases import DEFAULT_CASES, cases_of, evaluate
from ..theme import THEME, STATUS, mono
from ..widgets import label, button, ROW_H
from .base import VDialog

COLS = ['Case', 'Front drop mm', 'Rear drop mm', 'Pitch °', 'Blind m²', 'Δ blind', 'Min px/cm', 'Ground shift cm',
        'Seam ghost cm', 'Status']
EDITABLE = 3


class _Sig(QObject):
    done = pyqtSignal(object)
    failed = pyqtSignal(str)


class LoadCasesDialog(VDialog):
    def __init__(self, ctx, parent=None):
        super().__init__(parent, 'Load cases', 'body drop and pitch under load', (1020, 520))
        self.ctx = ctx
        self.results = None
        self.busy = False
        self.sig = _Sig()
        self.sig.done.connect(self._done)
        self.sig.failed.connect(self._failed)
        lay = QVBoxLayout(self.body)
        lay.setContentsMargins(14, 10, 14, 10)
        lay.setSpacing(6)
        lay.addWidget(label('Enter how far each axle drops from kerb weight in every load case (static suspension deflection, '
                            'positive = lower). The cameras are calibrated at kerb weight, so a loaded body moves the '
                            'stitched ground image and the two cameras at a seam stop agreeing. The preset values are '
                            'examples: replace them with the vehicle\'s suspension data.', muted=True, small=True, wrap=True))
        self.table = QTableWidget(0, len(COLS))
        self.table.setHorizontalHeaderLabels(COLS)
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(ROW_H + 2)
        self.table.horizontalHeader().setFixedHeight(ROW_H)
        self.table.setShowGrid(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for c in range(1, len(COLS)):
            hh.setSectionResizeMode(c, QHeaderView.ResizeMode.ResizeToContents)
        self.table.itemChanged.connect(self._edited)
        lay.addWidget(self.table, 1)
        row = QHBoxLayout()
        b_add = button('Add case', None, 'plus')
        b_add.clicked.connect(self.add_case)
        b_del = button('Remove', None, 'minus')
        b_del.clicked.connect(self.remove_case)
        b_reset = button('Example values')
        b_reset.clicked.connect(lambda: self.set_cases(DEFAULT_CASES))
        row.addWidget(b_add)
        row.addWidget(b_del)
        row.addWidget(b_reset)
        row.addStretch(1)
        self.bar = QProgressBar()
        self.bar.setTextVisible(False)
        self.bar.setRange(0, 0)
        self.bar.setFixedWidth(160)
        self.bar.hide()
        row.addWidget(self.bar)
        lay.addLayout(row)
        self.hint.setText('Ground shift: worst move of a ground point in the stitched view · Seam ghost: worst disagreement '
                          'between the two cameras at a seam')
        b_close = button('Close')
        b_close.clicked.connect(self.accept)
        self.b_run = button('Run', 'primary')
        self.b_run.clicked.connect(self.run)
        self.buttons.addWidget(b_close)
        self.buttons.addWidget(self.b_run)
        self._filling = False
        self._run_for = None           # the project a running evaluation belongs to
        self.set_cases(cases_of(ctx.store.project))
        # File > New / Open: show (and run) the new project's cases, never write the old ones into it
        ctx.store.project_replaced.connect(self._project_replaced)

    def _project_replaced(self):
        self.set_cases(cases_of(self.ctx.store.project))
        self.set_subtitle('body drop and pitch under load')

    # ------------------------------------------------------------------ table
    def set_cases(self, cases):
        self._filling = True
        self.table.setRowCount(0)
        for name, f, r in cases:
            self._append(name, f, r)
        self._filling = False
        self.results = None

    def _append(self, name, f, r):
        i = self.table.rowCount()
        self.table.insertRow(i)
        for j, v in enumerate((str(name), f'{float(f):g}', f'{float(r):g}')):
            it = QTableWidgetItem(v)
            if j:
                it.setFont(mono(12))
                it.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.table.setItem(i, j, it)
        for j in range(EDITABLE, len(COLS)):
            it = QTableWidgetItem('')
            it.setFlags(it.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(i, j, it)

    def add_case(self):
        self._filling = True
        self._append(f'Case {self.table.rowCount() + 1}', 0, 0)
        self._filling = False
        self.table.editItem(self.table.item(self.table.rowCount() - 1, 0))

    def remove_case(self):
        rows = sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True)
        for r in rows:
            self.table.removeRow(r)

    def _edited(self, it):
        if self._filling or it.column() >= EDITABLE:
            return
        if it.column() in (1, 2):
            from ...core.mathutil import parse_number
            try:
                v = parse_number(it.text(), 'length', 'mm')
            except ValueError:
                v = 0.0
            self._filling = True
            it.setText(f'{v:g}')
            self._filling = False
        # results no longer match the inputs
        for j in range(EDITABLE, len(COLS)):
            o = self.table.item(it.row(), j)
            if o:
                o.setText('')

    def cases(self):
        from ...core.mathutil import parse_number
        out = []
        for i in range(self.table.rowCount()):
            name = (self.table.item(i, 0).text() if self.table.item(i, 0) else '').strip() or f'Case {i + 1}'
            vals = []
            for j in (1, 2):
                try:
                    vals.append(float(parse_number(self.table.item(i, j).text(), 'length', 'mm')))
                except (ValueError, AttributeError):
                    vals.append(0.0)
            out.append((name, vals[0], vals[1]))
        return out

    # ------------------------------------------------------------------ run
    def run(self):
        if self.busy:
            return
        cases = self.cases()
        if not cases:
            return
        store = self.ctx.store
        if self.ctx.analysis.result is None or not store.variant.cameras:
            self.set_subtitle('Nothing to evaluate yet: the load cases need a vehicle, cameras and an analysis result.')
            return
        if [list(c) for c in cases] != [[n, float(f), float(r)] for n, f, r in cases_of(store.project)]:
            store.edit('analysis', lambda p: setattr(p.analysis, 'load_cases', [list(c) for c in cases]), 'Edited load cases')
        project = store.project.snapshot()
        eng = self.ctx.window._shared_engine()
        cell = max(project.analysis.cell_m, 0.05)
        self._run_for = store.project
        self.busy = True
        self.b_run.setEnabled(False)
        self.bar.show()

        def work():
            try:
                self.sig.done.emit(evaluate(eng, project, cases, cell=cell))
            except Exception as e:
                self.sig.failed.emit(f'{type(e).__name__}: {e}\n{traceback.format_exc()}')
        threading.Thread(target=work, daemon=True).start()

    def _done(self, results):
        self.busy = False
        self.b_run.setEnabled(True)
        self.bar.hide()
        if self._run_for is not self.ctx.store.project:
            return                         # evaluated for a project that has since been replaced
        self.results = results
        self._filling = True
        for i, r in enumerate(results):
            if i >= self.table.rowCount():
                break
            vals = [f'{r.pitch_deg:+.2f}', f'{r.blind_m2:.3f}', f'{r.delta_blind:+.3f}', f'{r.min_density:.1f}',
                    f'{r.ground_shift_cm:.1f}', f'{r.seam_ghost_cm:.1f}',
                    {'good': 'OK', 'warning': 'Recalibrate per load', 'serious': 'Visible ghosting'}.get(r.status, r.status)]
            for j, v in enumerate(vals, EDITABLE):
                it = self.table.item(i, j)
                it.setText(v.replace('-', '−'))
                if j < len(COLS) - 1:
                    it.setFont(mono(12))
                    it.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                else:
                    it.setForeground(QColor(STATUS.get(r.status, '#888')))
        self._filling = False
        worst = max(results, key=lambda r: r.seam_ghost_cm) if results else None
        if worst:
            self.set_subtitle(f'worst: {worst.name} · {worst.seam_ghost_cm:.1f} cm seam ghost · '
                              f'{worst.ground_shift_cm:.1f} cm ground shift')
        self.ctx.store.project.add_log('loadcases', f'{len(results)} cases' +
                                       (f' · worst seam ghost {worst.seam_ghost_cm:.1f} cm ({worst.name})' if worst else ''))

    def _failed(self, m):
        self.busy = False
        self.b_run.setEnabled(True)
        self.bar.hide()
        print(m)
        self.set_subtitle('Failed: ' + m.splitlines()[0])
