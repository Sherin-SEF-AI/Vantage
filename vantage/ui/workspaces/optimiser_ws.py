"""B3 · Optimise placement."""
from __future__ import annotations

import copy
import time

import numpy as np
from PyQt6.QtCore import Qt, QTimer, QObject, QPointF, QRectF, pyqtSignal
from PyQt6.QtGui import QPainter, QPen, QPolygonF, QColor
from PyQt6.QtWidgets import (QWidget, QHBoxLayout, QVBoxLayout, QGridLayout, QSplitter, QRadioButton, QButtonGroup,
                             QCheckBox, QComboBox, QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView,
                             QProgressBar, QTabWidget, QScrollArea, QFileDialog, QFrame)

from ...core import diff_optim as DO
from ...core.optimiser import OptSettings, default_vars, FreeVar
from ..charts import Scatter, ChartBase
from ..planview import PlanView
from ..theme import THEME, mono, sans, STATUS
from .. import icons
from ..widgets import label, button, ScrubSpinBox, Chip, ROW_H, hline, StatusLine, SegmentedControl


class OptimiserPage(QWidget):
    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        self.opt = None
        self.cands = []
        self.sel = None
        self.sel_result = None
        self.base_result = None
        self.sel_cams = None
        self.t0 = None
        self.setObjectName('base')
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        split = QSplitter(Qt.Orientation.Horizontal)
        split.setHandleWidth(2)
        # ------------------------------------------------ left: settings
        left = QWidget()
        left.setObjectName('panel')
        left.setMinimumWidth(300)
        ll = QVBoxLayout(left)
        ll.setContentsMargins(8, 6, 8, 8)
        ll.setSpacing(4)
        top = QHBoxLayout()
        top.addWidget(label('Optimise placement'))
        top.addStretch(1)
        self.from_lbl = label('', muted=True, small=True)
        top.addWidget(self.from_lbl)
        ll.addLayout(top)
        ll.addWidget(hline())
        ll.addWidget(label('Objective', bold=True))
        self.obj = QButtonGroup(self)
        for i, (k, t) in enumerate((('blind', 'Least blind area'), ('cameras', 'Fewest cameras'), ('density', 'Highest minimum density'))):
            rb = QRadioButton(t)
            rb.setProperty('key', k)
            self.obj.addButton(rb, i)
            ll.addWidget(rb)
            if i == 0:
                rb.setChecked(True)
        self.under_tol = QCheckBox('Score under tolerance')
        self.under_tol.setChecked(True)
        ll.addWidget(self.under_tol)
        self.tol_note = label('± 5°, ± 10 mm, 24 samples per candidate. Tie-break: highest minimum density.', muted=True, small=True, wrap=True)
        ll.addWidget(self.tol_note)
        ll.addSpacing(4)
        ll.addWidget(label('Constraints', bold=True))
        g = QGridLayout()
        g.setVerticalSpacing(2)
        self.zones = QComboBox()
        self.zones.addItem('Allowed only', 'allowed')
        self.zones.addItem('Any surface', 'any')
        self.h_lo = ScrubSpinBox('', 'length', 2, '', step=0.01)
        self.h_hi = ScrubSpinBox('', 'length', 2, 'm', step=0.01)
        self.h_lo.set_value(0.45)
        self.h_hi.set_value(1.05)
        self.cable = ScrubSpinBox('≤', 'length', 2, 'm', step=0.05)
        self.cable.set_value(3.0)
        self.c_lo = ScrubSpinBox('', 'plain', 0, '', step=1)
        self.c_hi = ScrubSpinBox('', 'plain', 0, '', step=1)
        self.module = QComboBox()
        hw = QWidget()
        hl = QHBoxLayout(hw)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.setSpacing(2)
        hl.addWidget(self.h_lo)
        hl.addWidget(label('–', muted=True))
        hl.addWidget(self.h_hi)
        cw = QWidget()
        cl = QHBoxLayout(cw)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.setSpacing(2)
        cl.addWidget(self.c_lo)
        cl.addWidget(label('–', muted=True))
        cl.addWidget(self.c_hi)
        for i, (t, w) in enumerate((('Mount zones', self.zones), ('Height', hw), ('Cable', self.cable), ('Camera count', cw), ('Module', self.module))):
            a = label(t, ink2=True)
            a.setFixedWidth(110)
            g.addWidget(a, i, 0)
            g.addWidget(w, i, 1)
        ll.addLayout(g)
        ll.addSpacing(4)
        ll.addWidget(label('Critical targets', bold=True))
        ch = QHBoxLayout()
        ch.setSpacing(4)
        self.crit = QComboBox()
        self.crit.addItem('Ignore', 'off')
        self.crit.addItem('Every target resolved', 'require')
        self.crit.addItem('Weigh unresolved targets', 'weigh')
        self.crit.setToolTip('The targets of Analysis ▸ Critical targets: required, a candidate that leaves one '
                             'unresolved never reaches the Pareto front; weighed, each unresolved target adds the '
                             'weight to its objective (m² of blind area, cameras or px/cm)')
        self.crit.currentIndexChanged.connect(self._crit_mode)
        self.crit_w = ScrubSpinBox('w', 'plain', 2, '', step=0.1)
        self.crit_w.set_value(1.0)
        self.crit_w.setToolTip('Added to a candidate\'s objective per critical target it leaves unresolved')
        self.crit_w.setFixedWidth(84)
        ch.addWidget(self.crit, 1)
        ch.addWidget(self.crit_w)
        ll.addLayout(ch)
        self.crit_note = label('', muted=True, small=True, wrap=True)
        ll.addWidget(self.crit_note)
        ll.addSpacing(4)
        fh = QHBoxLayout()
        fh.addWidget(label('Free variables', bold=True))
        fh.addStretch(1)
        fh.addWidget(label('L/R pairs move together', muted=True, small=True))
        ll.addLayout(fh)
        self.vars_t = QTableWidget(0, 3)
        self.vars_t.setHorizontalHeaderLabels(['Cam', 'Moves', 'Range'])
        self.vars_t.verticalHeader().setVisible(False)
        self.vars_t.verticalHeader().setDefaultSectionSize(ROW_H)
        self.vars_t.horizontalHeader().setFixedHeight(ROW_H)
        self.vars_t.setShowGrid(False)
        self.vars_t.setEditTriggers(QAbstractItemView.EditTrigger.DoubleClicked)
        self.vars_t.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.vars_t.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.vars_t.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.vars_t.itemChanged.connect(self._var_edit)
        ll.addWidget(self.vars_t)
        ll.addWidget(label('Double-click a range to edit, e.g. “±0.10 · 30–50°” or “50–80° · 35–55°”.', muted=True, small=True, wrap=True))
        gg = QHBoxLayout()
        self.gens = ScrubSpinBox('gen', 'plain', 0, '', step=5)
        self.gens.set_value(60)
        self.pop = ScrubSpinBox('pop', 'plain', 0, '', step=5)
        self.pop.set_value(60)
        gg.addWidget(self.gens)
        gg.addWidget(self.pop)
        ll.addLayout(gg)
        ll.addStretch(1)
        bl = QHBoxLayout()
        bl.addStretch(1)
        self.b_run = button('Start', 'primary')
        self.b_run.clicked.connect(self.toggle)
        bl.addWidget(self.b_run)
        ll.addLayout(bl)
        split.addWidget(left)
        # ------------------------------------------------ centre
        centre = QWidget()
        centre.setObjectName('panel')
        cl = QVBoxLayout(centre)
        cl.setContentsMargins(8, 6, 8, 6)
        cl.setSpacing(6)
        stats = QHBoxLayout()
        stats.setSpacing(28)
        self.s = {}
        for k in ('Generation', 'Evaluations', 'Elapsed', 'Remaining ≈', 'Best worst-case blind'):
            w = QWidget()
            h = QHBoxLayout(w)
            h.setContentsMargins(0, 0, 0, 0)
            h.setSpacing(6)
            h.addWidget(label(k, ink2=True))
            v = label('—', mono_=True)
            h.addWidget(v)
            self.s[k] = v
            stats.addWidget(w)
        self.bar = QProgressBar()
        self.bar.setTextVisible(False)
        self.bar.setRange(0, 1000)
        self.bar.setFixedWidth(200)
        stats.addStretch(1)
        stats.addWidget(self.bar)
        cl.addLayout(stats)
        cl.addWidget(hline())
        self.scatter = Scatter()
        self.scatter.title = 'Blind area vs minimum density · every evaluated layout'
        self.scatter.picked.connect(self.select)
        cl.addWidget(self.scatter, 3)
        cl.addWidget(label('Candidates · Pareto front'))
        self.table = QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels(['', 'Cams', 'Blind m²', 'Worst m²', 'Min px/cm', 'Cable max m',
                                              'Targets', 'Changes vs current'])
        self.table.horizontalHeaderItem(6).setToolTip('Critical targets resolved at the candidate\'s pose (the icon: '
                                                      'still resolved in every tolerance sample)')
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(ROW_H + 1)
        self.table.horizontalHeader().setFixedHeight(ROW_H)
        self.table.setShowGrid(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        hh = self.table.horizontalHeader()
        for c in range(7):
            hh.setSectionResizeMode(c, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(7, QHeaderView.ResizeMode.Stretch)
        self.table.setColumnHidden(6, True)
        self.table.cellClicked.connect(lambda r, c: self.select(r))
        cl.addWidget(self.table, 2)
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.addTab(centre, 'Evolutionary search')
        self.refiner = RefinePanel(ctx, self)
        self.tabs.addTab(self.refiner, 'Refine with gradients')
        split.addWidget(self.tabs)
        # ------------------------------------------------ right
        right = QWidget()
        right.setObjectName('base')
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(2)
        self.plan = PlanView(ctx, 'footprints', interactive=False)
        self.plan.legend = False
        self.plan.result_fn = lambda: self.sel_result
        self.plan.cameras_fn = lambda: self.sel_cams or []
        rl.addWidget(self.plan, 1)
        self.delta = PlanView(ctx, 'delta', interactive=False)
        self.delta.delta_fn = lambda: (self.base_result, self.sel_result)
        self.delta.result_fn = lambda: self.sel_result if self.base_result is not None else None
        self.delta.cameras_fn = lambda: self.sel_cams or []
        self.delta.show_tooltip = False
        rl.addWidget(self.delta, 1)
        foot = QWidget()
        foot.setObjectName('panel')
        fl = QHBoxLayout(foot)
        fl.setContentsMargins(8, 6, 8, 6)
        self.summary = label('', ink2=True, small=True, wrap=True)
        fl.addWidget(self.summary, 1)
        self.b_prev = button('Preview in 3D')
        self.b_prev.clicked.connect(self.preview3d)
        self.b_apply = button('Apply as new variant', 'primary')
        self.b_apply.clicked.connect(self.apply)
        self.b_three = button('Add the best three', None, None, 'Add the first three candidates of the Pareto front '
                                                                'as variants, in one undo step (the active variant '
                                                                'stays)')
        self.b_three.clicked.connect(lambda: self.apply_best())
        self.b_three.setEnabled(False)
        fl.addWidget(self.b_prev)
        fl.addWidget(self.b_three)
        fl.addWidget(self.b_apply)
        rl.addWidget(foot)
        split.addWidget(right)
        split.setSizes([310, 1065, 625])
        lay.addWidget(split)
        self.vars = []
        self._busy = False
        self.timer = QTimer(self)
        self.timer.setInterval(500)
        self.timer.timeout.connect(self._tick)
        ctx.store.project_replaced.connect(self._project_replaced)

    # ------------------------------------------------------------------ setup
    def _project_replaced(self):
        """File > New / Open: candidates of the old project must not be applied to the new one."""
        job = self.ctx.window._jobs.get('optimiser') if self.ctx.window else None
        if job:
            job['cancel'].set()
        self.opt = None
        self.cands = []
        self.sel = None
        self.sel_result = self.base_result = self.sel_cams = None
        self.table.setRowCount(0)
        self.b_three.setEnabled(False)
        self.scatter.set_data([], [], [], None, None)
        self.summary.setText('')
        for v in self.s.values():
            v.setText('—')
        self.bar.setValue(0)
        for pv in (self.plan, self.delta):
            pv._img_key = None
            pv.update()
        self.refiner.project_replaced()
        self.activate()

    def activate(self):
        p = self.ctx.store.project
        v = p.variant
        self.from_lbl.setText(f'from {v.name}')
        n = len(v.cameras)
        self.c_lo.set_value(max(1, min(4, n)))
        self.c_hi.set_value(n)
        self.module.clear()
        self.module.addItem('Keep current lenses', '')
        if self.opt is None:
            self.vars = default_vars(p, v)
            self._fill_vars()
            self.plan.label = 'Select a candidate'
            self.delta.label = ''
        self.cable.set_value(max(max((c.cable_m for c in v.cameras), default=3.0), 3.0))
        self._crit_mode()
        self.refiner.load()

    def _crit_mode(self, *_):
        from ...core.critical import settings_of
        n = sum(t['enabled'] for t in settings_of(self.ctx.store.project).targets)
        mode = self.crit.currentData()
        self.crit_w.setEnabled(mode == 'weigh')
        self.crit.setEnabled(n > 0)
        self.crit_note.setText(f'{n} target{"s" * (n != 1)} from Analysis ▸ Critical targets; the cameras\' '
                               f'requirement only (the lidars do not move).' if n else
                               'No critical targets yet: place them in Analysis ▸ Critical targets.')

    def _fill_vars(self):
        self._busy = True
        st = self.ctx.store
        self.vars_t.setRowCount(len(self.vars))
        for i, fv in enumerate(self.vars):
            c = st.cam(fv.uid)
            it = QTableWidgetItem(fv.tag)
            it.setFont(mono(12))
            it.setFlags(it.flags() & ~Qt.ItemFlag.ItemIsEditable)
            if c:
                it.setIcon(_chip_icon(c.slot))
            self.vars_t.setItem(i, 0, it)
            moves = f'mirrors {self._tag(fv.linked_to)}' if fv.linked_to else \
                (f'along {"bumper" if c and c.tag == "R" else "grille" if c and c.tag == "F" else "body"} ({fv.axis}), ψ, θ' if fv.move else 'ψ, θ')
            m = QTableWidgetItem(moves)
            m.setFlags(m.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.vars_t.setItem(i, 1, m)
            r = QTableWidgetItem(fv.describe())
            r.setFont(mono(11))
            r.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            if fv.linked_to:
                r.setFlags(r.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.vars_t.setItem(i, 2, r)
        self.vars_t.setFixedHeight(ROW_H * (len(self.vars) + 1) + 4)
        self._busy = False

    def _tag(self, uid):
        c = self.ctx.store.cam(uid)
        return c.tag if c else '?'

    def _var_edit(self, it):
        if self._busy or it.column() != 2:
            return
        import re
        fv = self.vars[it.row()]
        txt = it.text().replace('−', '-')
        nums = [float(x) for x in re.findall(r'-?\d+(?:\.\d+)?', txt)]
        try:
            if '±' in txt and nums:
                fv.move = abs(nums[0])
                nums = nums[1:]
            if len(nums) >= 4:
                fv.yaw = (nums[0], nums[1])
                fv.pitch = (nums[2], nums[3])
            elif len(nums) >= 2:
                fv.pitch = (nums[0], nums[1])
        except Exception:
            pass
        self._fill_vars()

    def settings(self) -> OptSettings:
        key = self.obj.checkedButton().property('key')
        crit = self.crit.currentData() if self.crit.isEnabled() else 'off'
        return OptSettings(objective=key, under_tolerance=self.under_tol.isChecked(), zones=self.zones.currentData(),
                           height=(self.h_lo.value(), self.h_hi.value()), cable_max=self.cable.value(),
                           count=(int(self.c_lo.value()), int(self.c_hi.value())), generations=int(max(1, self.gens.value())),
                           population=int(max(6, self.pop.value())), vars=self.vars, critical=crit,
                           critical_weight=max(0.0, float(self.crit_w.value())))

    # ------------------------------------------------------------------ run
    def toggle(self):
        if 'optimiser' in self.ctx.window._jobs:
            self.ctx.window.cancel_jobs()
            self.b_run.setText('Stopping…')
            return
        st = self.settings()
        self.base_result = None
        self.opt = self.ctx.window.run_optimiser(st, self._progress)
        if self.opt is None:
            return
        self.t0 = time.time()
        self.b_run.setText('Stop')
        self.b_run.setIcon(icons.icon('stop', 12))
        self.timer.start()

    def _tick(self):
        if self.t0 is None:
            return
        el = time.time() - self.t0
        self.s['Elapsed'].setText(f'{int(el // 60)}:{int(el % 60):02d}')

    def _progress(self, opt):
        if opt is None or opt is not self.opt:
            return
        st = opt.st
        g = opt.generation
        el = time.time() - (self.t0 or time.time())
        self.s['Generation'].setText(f'{g} / {st.generations}')
        self.s['Evaluations'].setText(f'{opt.evals:,}'.replace(',', ' '))
        rem = el / max(g, 1) * (st.generations - g)
        self.s['Remaining ≈'].setText(f'{int(rem // 60)}:{int(rem % 60):02d}')
        best = min((c.worst for c in opt.front), default=float('nan'))
        self.s['Best worst-case blind'].setText(f'{best:.3f} m²')
        self.bar.setValue(int(g / max(st.generations, 1) * 1000))
        self._update_chart(opt, final=False)

    def _update_chart(self, opt, final):
        ev = opt.evaluated
        if len(ev) > 4000:
            idx = np.linspace(0, len(ev) - 1, 4000).astype(int)
            ev = [ev[i] for i in idx]
        pts = [(c.worst, c.min_density) for c in ev if c.worst < 5]
        front = [(c.worst, c.min_density) for c in opt.front if c.worst < 5]
        self.cands = opt.candidates(5)
        labels = [(c.worst, c.min_density, c.label) for c in self.cands]
        nom = getattr(opt, 'nominal_cand', None)
        self.scatter.set_data(pts, front, labels, (nom.worst, nom.min_density) if nom else None, self.sel)
        self._fill_table(opt)

    def _fill_table(self, opt):
        self.table.setRowCount(len(self.cands))
        self.table.setColumnHidden(6, opt.crit is None)
        self.b_three.setEnabled(len(self.cands) > 0)
        for i, c in enumerate(self.cands):
            vals = [c.label, str(c.cams), f'{c.blind:.3f}', f'{c.worst:.3f}', f'{c.min_density:.1f}',
                    f'{c.cable_max:.2f}', opt.targets_text(c), opt.changes(c)]
            for j, v in enumerate(vals):
                it = QTableWidgetItem(v)
                it.setFont(mono(12) if j < 7 else mono(11))
                if 0 < j < 7:
                    it.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                if j == 5:
                    it.setIcon(icons.status_icon('good' if c.cable_ok else 'serious', 12))
                if j == 6 and opt.crit is not None:
                    pen = opt.crit_penalty(c)
                    it.setIcon(icons.status_icon('good' if pen == 0 else 'warning' if c.crit_unmet == 0 else
                                                 'serious', 12))
                    it.setToolTip('Every critical target resolved' + (' in every tolerance sample' if
                                                                       opt.st.under_tolerance else '')
                                  if pen == 0 else
                                  (f'Unresolved: {", ".join(c.crit_missing)}' if c.crit_missing else
                                   f'{pen} target{"s" * (pen > 1)} lost in a tolerance sample'))
                self.table.setItem(i, j, it)

    def finished(self, opt):
        self.timer.stop()
        self.b_run.setText('Start')
        self.b_run.setIcon(icons.icon('play', 12))
        if opt is not self.opt:
            return                         # a run for a project that has since been replaced
        self.bar.setValue(1000)
        self._update_chart(opt, final=True)
        if self.cands:
            self.select(0)

    # ------------------------------------------------------------------ candidates
    def select(self, i):
        if not (0 <= i < len(self.cands)) or self.opt is None:
            return
        self.sel = i
        cand = self.cands[i]
        cams = []
        for uid, gene in cand.genes.items():
            c = self.opt.camera_for(uid, gene)
            if c is not None:
                cams.append(c)
        self.sel_cams = cams
        eng = self.ctx.analysis.engine
        with self.ctx.analysis._lock:
            snap = self.ctx.store.project.snapshot()
            self.sel_result = eng.analyse(snap, cameras=cams)
            if self.base_result is None:
                self.base_result = eng.analyse(snap)
        cur = self.base_result.metrics['blind_m2'] if self.base_result else 0
        self.plan.label = f'{cand.label} · top · footprints'
        self.delta.label = f'{cand.label} vs current · Δ density'
        for pv in (self.plan, self.delta):
            pv._img_key = None
            pv.fit()
        nom = getattr(self.opt, 'nominal_cand', None)
        crit = '' if self.opt.crit is None else \
            f' It resolves {self.opt.targets_text(cand)} critical targets' + (
                f' (current {self.opt.targets_text(nom)})' if nom else '') + '.'
        self.summary.setText((f'{cand.label} keeps {cand.blind:.3f} m² blind and moves the tolerance worst case from '
                              f'{nom.worst:.3f} to {cand.worst:.3f} m².' if nom else '') + crit)
        self.table.selectRow(i)
        self.scatter.selected = i
        self.scatter.update()

    def preview3d(self):
        if self.sel_cams is None:
            return
        w = self.ctx.window
        w.set_page('viewports')
        pane = w.viewarea.active
        pane.set_kind('persp')
        pane.scene.cameras_override = self.sel_cams
        pane.scene.result_override = self.sel_result
        pane.scene._ground_key = None
        pane.scene.label_prefix = f'Preview {self.cands[self.sel].label} · Esc returns to the optimiser'
        pane.vp3d.update()

        def restore():
            pane.scene.cameras_override = None
            pane.scene.result_override = None
            pane.scene._ground_key = None
            pane.scene.label_prefix = ''
            pane.vp3d.update()
        w._preview_restore = restore
        self.ctx.msg('Previewing the candidate in 3D. Press Esc to go back.', 5000)

    def apply(self):
        if self.sel is None or self.opt is None:
            return
        cand = self.cands[self.sel]
        base = self.ctx.store.project.variant.name
        name = f'{base} · {cand.label}'
        v = self.opt.to_variant(cand, name)

        def fn(p):
            p.variants.append(v)
            p.active_uid = v.uid
        self.ctx.store.edit('variants', fn, f'Applied optimiser {cand.label} as {name}')
        self.ctx.window._fill_variant_combo()
        self.ctx.msg(f'Created variant {name}')

    def apply_best(self, n=3):
        """The first n candidates of the Pareto front (as the table lists them) as new variants, one undo step; the
        active variant stays, so they can be compared."""
        if self.opt is None or not self.cands:
            return []
        base = self.ctx.store.project.variant.name
        taken = {v.name for v in self.ctx.store.project.variants}
        new = []
        for cand in self.cands[:n]:
            name, k = f'{base} · {cand.label}', 2
            while name in taken:
                name, k = f'{base} · {cand.label} ({k})', k + 1
            taken.add(name)
            new.append(self.opt.to_variant(cand, name))

        def fn(p):
            p.variants.extend(new)
        self.ctx.store.edit('variants', fn, f'Added optimiser {", ".join(c.label for c in self.cands[:n])} as '
                                            f'variants')
        self.ctx.window._fill_variant_combo()
        self.ctx.msg(f'Added {len(new)} variant{"s" * (len(new) != 1)}: {", ".join(v.name for v in new)}', 5000)
        return new


def _chip_icon(slot):
    from PyQt6.QtGui import QPixmap, QPainter, QIcon
    pm = QPixmap(8, 8)
    pm.fill(THEME.cam(slot))
    return QIcon(pm)


# ============================================================================ refine with gradients
class _RefineSig(QObject):
    live = pyqtSignal(object)           # (label, history, checks) copies from the worker
    prog = pyqtSignal(float)


def _lim(vals, pad=0.08):
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-12:
        lo, hi = lo - (abs(lo) * 0.1 or 0.5), hi + (abs(hi) * 0.1 or 0.5)
    d = (hi - lo) * pad
    return lo - d, hi + d


def _ticks(lo, hi, n=4):
    span = hi - lo
    step = 10 ** np.floor(np.log10(span / n))
    for m in (1, 2, 2.5, 5, 10):
        if span / (step * m) <= n + 1:
            step *= m
            break
    t0 = np.ceil(lo / step) * step
    return [float(v) for v in np.arange(t0, hi + step * 1e-9, step)]


class IterChart(ChartBase):
    """The surrogate objective at every L-BFGS-B iteration (a line; hairlines where a stage starts and τ sharpens)
    and the exact engine's objective at every stage boundary (dots), for one start."""
    MARGIN = (54, 26, 12, 30)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.hist, self.checks = [], []
        self.setMinimumHeight(150)
        self.title = 'Objective per iteration'

    def set_data(self, hist, checks):
        self.hist, self.checks = list(hist or []), list(checks or [])
        self.tip = None
        self.update()

    def _rect(self):
        l_, t_, r_, b_ = self.MARGIN
        return QRectF(l_, t_, max(self.width() - l_ - r_, 10), max(self.height() - t_ - b_, 10))

    def _xy(self):
        r = self._rect()
        ys = [h['objective'] for h in self.hist] + [c['objective'] for c in self.checks]
        xs = [h['iteration'] for h in self.hist] + [c['iteration'] for c in self.checks]
        xlim = (0.0, float(max(max(xs), 1)))
        ylim = _lim(ys)
        X = lambda v: r.left() + (v - xlim[0]) / ((xlim[1] - xlim[0]) or 1.0) * r.width()        # noqa: E731
        Y = lambda v: r.bottom() - (v - ylim[0]) / ((ylim[1] - ylim[0]) or 1.0) * r.height()     # noqa: E731
        return r, X, Y, xlim, ylim

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.paint_title(p, 'line: surrogate · dots: exact engine')
        if not self.hist and not self.checks:
            p.setFont(sans(12))
            p.setPen(THEME.color('muted'))
            p.drawText(QRectF(0, 0, self.width(), self.height()), Qt.AlignmentFlag.AlignCenter,
                       'Refine a layout to see the objective fall')
            p.end()
            return
        r, X, Y, xlim, ylim = self._xy()
        p.setFont(mono(10))
        for t in _ticks(*ylim):
            y = Y(t)
            p.setPen(QPen(THEME.color('hair'), 1))
            p.drawLine(QPointF(r.left(), y), QPointF(r.right(), y))
            p.setPen(THEME.color('muted'))
            p.drawText(QRectF(0, y - 7, r.left() - 6, 14), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                       f'{t:.4g}')
        for t in _ticks(*xlim):
            p.setPen(THEME.color('muted'))
            p.drawText(QRectF(X(t) - 30, r.bottom() + 3, 60, 13), Qt.AlignmentFlag.AlignCenter, f'{t:.0f}')
        p.setPen(QPen(THEME.color('strong'), 1))
        p.drawLine(QPointF(r.left(), r.bottom()), QPointF(r.right(), r.bottom()))
        p.setFont(sans(10))
        p.setPen(THEME.color('muted'))
        p.drawText(QRectF(r.left(), r.bottom() + 15, r.width(), 13), Qt.AlignmentFlag.AlignRight, 'iteration')
        starts = [h for h in self.hist if h.get('start')]
        for h in starts[1:]:
            x = X(h['iteration'])
            p.setPen(QPen(THEME.color('hair'), 1, Qt.PenStyle.DashLine))
            p.drawLine(QPointF(x, r.top()), QPointF(x, r.bottom()))
        for h in starts:
            p.setPen(THEME.color('muted'))
            p.drawText(QRectF(X(h['iteration']) + 3, r.top(), 60, 12), Qt.AlignmentFlag.AlignLeft,
                       f'τ ×{h["sharpness"]:.2g}')
        stages = {}
        for h in self.hist:
            stages.setdefault(h['stage'], []).append(h)
        p.setPen(QPen(THEME.cam(0), 1.6))
        for hs in stages.values():
            if len(hs) > 1:
                p.drawPolyline(QPolygonF([QPointF(X(h['iteration']), Y(h['objective'])) for h in hs]))
        p.setPen(QPen(THEME.color('panel'), 1.5))
        p.setBrush(THEME.color('ink'))
        for c in self.checks:
            p.drawEllipse(QPointF(X(c['iteration']), Y(c['objective'])), 4, 4)
        self.paint_tip(p)
        p.end()

    def mouseMoveEvent(self, e):
        self.tip = None
        if self.hist or self.checks:
            r, X, Y, _, _ = self._xy()
            pos = e.position()
            pts = [(X(h['iteration']), Y(h['objective']), 'surrogate', h) for h in self.hist] + \
                  [(X(c['iteration']), Y(c['objective']), 'exact', c) for c in self.checks]
            best = min(pts, key=lambda t: (t[0] - pos.x()) ** 2 + (t[1] - pos.y()) ** 2)
            if (best[0] - pos.x()) ** 2 + (best[1] - pos.y()) ** 2 < 400:
                d = best[3]
                if best[2] == 'exact':
                    self.set_tip(QPointF(best[0], best[1]), f'{d["objective"]:.4f}',
                                 f'exact engine after stage {d["stage"]}',
                                 f'blind {d["blind_m2"]:.3f} m² · below minimum {d["low_m2"]:.3f} m²')
                else:
                    self.set_tip(QPointF(best[0], best[1]), f'{d["objective"]:.4f}',
                                 f'surrogate, iteration {d["iteration"]}', f'τ ×{d["sharpness"]:.2g}')
        self.update()


def _num_cell(t, r, c, text, align_right=True):
    it = QTableWidgetItem(text)
    it.setFont(mono(11))
    if align_right:
        it.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    t.setItem(r, c, it)
    return it


def _small_table(cols):
    t = QTableWidget(0, len(cols))
    t.setHorizontalHeaderLabels(cols)
    t.verticalHeader().setVisible(False)
    t.verticalHeader().setDefaultSectionSize(ROW_H)
    t.horizontalHeader().setFixedHeight(ROW_H)
    t.setShowGrid(False)
    t.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    t.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
    hh = t.horizontalHeader()
    for c in range(len(cols)):
        hh.setSectionResizeMode(c, QHeaderView.ResizeMode.ResizeToContents)
    hh.setStretchLastSection(True)
    return t


class RefinePanel(QWidget):
    """Refine with gradients: L-BFGS-B on a smooth surrogate of the coverage objective (core.diff_optim), started
    from the current variant or the selected Pareto candidates, every result checked with the exact engine."""

    METRICS = (('blind_m2', 'Blind ground', 'm²', 3), ('low_m2', 'Below minimum density', 'm²', 3),
               ('unresolved', 'Unresolved critical targets', '', 2), ('seam_near', 'Objects on a seam', '', 2),
               ('min_density', 'Minimum density', 'px/cm', 2), ('objective', 'Objective', '', 4))

    def __init__(self, ctx, page):
        super().__init__()
        self.ctx = ctx
        self.page = page
        self.result = None
        self.base_variant_uid = ''
        self._loading = False
        self._committing = False
        self._live = {}
        self._last_emit = 0.0
        self.sig = _RefineSig()
        self.sig.live.connect(self._on_live)
        self.sig.prog.connect(lambda f: self.bar.setValue(int(f * 1000)))
        self.setObjectName('panel')
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        # ------------------------------------------------ settings
        sc = QScrollArea()
        sc.setWidgetResizable(True)
        sc.setFrameShape(QFrame.Shape.NoFrame)
        sc.setMinimumWidth(236)
        sc.setMaximumWidth(300)
        left = QWidget()
        left.setObjectName('panel')
        ll = QVBoxLayout(left)
        ll.setContentsMargins(8, 6, 8, 8)
        ll.setSpacing(4)
        ll.addWidget(label('Start from', bold=True))
        self.start = SegmentedControl([('variant', 'Current variant'), ('pareto', 'Pareto candidates')], 'variant')
        self.start.changed.connect(lambda _: (self._start_note(), self._commit()))
        ll.addWidget(self.start)
        self.start_note = label('', muted=True, small=True, wrap=True)
        ll.addWidget(self.start_note)
        ll.addSpacing(4)
        ll.addWidget(label('Objective weights', bold=True))
        g = QGridLayout()
        g.setVerticalSpacing(2)
        self.w = {}
        for i, (k, t, tip) in enumerate((
                ('w_blind', 'Blind ground', 'Per m² of ground no camera sees'),
                ('w_density', 'Below minimum', 'Per m² of seen ground below the minimum density'),
                ('w_critical', 'Critical target', 'Per critical target no camera resolves'),
                ('w_seam', 'Object on a seam', 'Per standing critical target or obstacle that a stitch seam passes '
                                               'near'))):
            a = label(t, ink2=True)
            a.setToolTip(tip)
            w = ScrubSpinBox('', 'plain', 2, '', step=0.05)
            w.setToolTip(tip)
            w.committed.connect(lambda _v: self._commit())
            self.w[k] = w
            g.addWidget(a, i, 0)
            g.addWidget(w, i, 1)
        ll.addLayout(g)
        ll.addSpacing(4)
        ll.addWidget(label('Limits and search', bold=True))
        g2 = QGridLayout()
        g2.setVerticalSpacing(2)
        self.lim = {}
        for i, (k, t, unit, dec, step, tip) in enumerate((
                ('max_move_mm', 'Move ±', 'mm', 0, 5, 'Per axis (x, y, z), inside the camera\'s mount zone'),
                ('max_turn_deg', 'Yaw, pitch ±', '°', 1, 0.5, 'Yaw and pitch'),
                ('max_roll_deg', 'Roll ±', '°', 1, 0.5, 'Roll'),
                ('seam_span_deg', 'Seams ±', '°', 1, 1, 'How far a seam angle may move (seam angles as variables)'),
                ('stages', 'Stages', '', 0, 1, 'τ annealed from soft to sharp over this many stages'),
                ('iterations', 'Iterations', '', 0, 5, 'L-BFGS-B iterations per stage at most'),
                ('candidates', 'Candidates', '', 0, 1, 'Pareto candidates refined when no row of the candidate '
                                                     'table is selected'))):
            a = label(t, ink2=True)
            a.setToolTip(tip)
            w = ScrubSpinBox('', 'plain', dec, unit, step=step)
            w.setToolTip(tip)
            w.committed.connect(lambda _v: self._commit())
            self.lim[k] = w
            g2.addWidget(a, i, 0)
            g2.addWidget(w, i, 1)
        ll.addLayout(g2)
        self.move = QCheckBox('Move positions (off: aim only)')
        self.move.toggled.connect(lambda _: self._commit())
        self.seams = QCheckBox('Seam angles as variables')
        self.seams.setToolTip('Each stitch seam may turn within its limit, away from critical targets and obstacles; '
                              'the result is saved as manual seams')
        self.seams.toggled.connect(lambda _: self._commit())
        ll.addWidget(self.move)
        ll.addWidget(self.seams)
        ll.addStretch(1)
        ll.addWidget(label(f'A stage stops when no variable moves more than {DO.STEP_DEG}° / {DO.STEP_MM} mm in an '
                           f'iteration: the solver\'s step tolerance, not how accurate the pose is in the world.',
                           muted=True, small=True, wrap=True))
        bl = QHBoxLayout()
        bl.addStretch(1)
        self.b_run = button('Refine', 'primary', None, 'Refine in the background (Cancel stops it)')
        self.b_run.clicked.connect(self.toggle)
        bl.addWidget(self.b_run)
        ll.addLayout(bl)
        sc.setWidget(left)
        lay.addWidget(sc)
        # ------------------------------------------------ results
        right = QWidget()
        right.setObjectName('panel')
        rl = QVBoxLayout(right)
        rl.setContentsMargins(8, 6, 8, 6)
        rl.setSpacing(6)
        top = QHBoxLayout()
        top.addWidget(label('Start', ink2=True))
        self.which = QComboBox()
        self.which.setMinimumWidth(110)
        self.which.currentIndexChanged.connect(lambda _: self.show_selected())
        top.addWidget(self.which)
        self.status = label('', muted=True, small=True)
        top.addWidget(self.status, 1)
        self.bar = QProgressBar()
        self.bar.setTextVisible(False)
        self.bar.setRange(0, 1000)
        self.bar.setFixedWidth(160)
        top.addWidget(self.bar)
        rl.addLayout(top)
        self.chart = IterChart()
        rl.addWidget(self.chart, 3)
        self.metrics = _small_table(['', 'Exact before', 'Exact after', 'Surrogate before', 'Surrogate after'])
        self.metrics.setRowCount(len(self.METRICS))
        self.metrics.setFixedHeight(ROW_H * (len(self.METRICS) + 1) + 4)
        rl.addWidget(self.metrics)
        self.cams = _small_table(['Cam', 'Moves', 'Δx mm', 'Δy mm', 'Δz mm', 'Δψ °', 'Δθ °', 'Δφ °'])
        self.cams.horizontalHeaderItem(5).setToolTip('Yaw change')
        self.cams.horizontalHeaderItem(6).setToolTip('Pitch change (positive = down)')
        self.cams.horizontalHeaderItem(7).setToolTip('Roll change')
        rl.addWidget(self.cams, 2)
        self.find_box = QVBoxLayout()
        self.find_box.setSpacing(2)
        rl.addLayout(self.find_box)
        self._find_rows = []
        foot = QHBoxLayout()
        foot.addStretch(1)
        self.b_export = button('Export…', None, None, 'The refinement as JSON (settings, poses, history, exact checks) '
                                                     'and two CSV files')
        self.b_export.clicked.connect(lambda: self.export())
        self.b_add = button('Add as variant', 'primary', None, 'Add the refined layout of this start as a new variant, '
                                                               'in one undo step (the active variant stays)')
        self.b_add.clicked.connect(lambda: self.add_variant())
        self.b_export.setEnabled(False)
        self.b_add.setEnabled(False)
        foot.addWidget(self.b_export)
        foot.addWidget(self.b_add)
        rl.addLayout(foot)
        lay.addWidget(right, 1)
        ctx.store.changed.connect(self._store_changed)
        self.load()

    # ------------------------------------------------------------------ settings
    def load(self):
        """The widgets from the project's analysis.diff_optim."""
        st = DO.settings_of(self.ctx.store.project)
        self._loading = True
        self.start.set_value(st.start)
        for k, w in self.w.items():
            w.set_value(getattr(st, k))
        for k, w in self.lim.items():
            w.set_value(getattr(st, k))
        self.move.setChecked(st.move)
        self.seams.setChecked(st.seams)
        self._loading = False
        self._start_note()

    def settings(self) -> DO.DiffSettings:
        d = DO.settings_of(self.ctx.store.project).to_dict()
        d['start'] = self.start.value() or 'variant'
        for k, w in list(self.w.items()) + list(self.lim.items()):
            d[k] = float(w.value())
        d['move'] = self.move.isChecked()
        d['seams'] = self.seams.isChecked()
        return DO.DiffSettings.from_dict(d)

    def _commit(self):
        if self._loading:
            return
        d = self.settings().to_dict()
        if d == DO.settings_of(self.ctx.store.project).to_dict():
            return

        def fn(p):
            p.analysis.diff_optim = d
        self._committing = True
        try:
            self.ctx.store.edit('analysis', fn, 'Gradient refinement settings')
        finally:
            self._committing = False

    def _store_changed(self, sections):
        if 'analysis' in sections and not self._committing:
            self.load()

    def _start_note(self):
        if self.start.value() == 'pareto':
            n = len(self.chosen())
            k = self.settings().candidates
            self.start_note.setText(f'{n} candidate{"s" * (n != 1)}: the rows selected in the candidate table, else '
                                    f'the first {k}.' if n else 'Run the evolutionary search first: its Pareto '
                                                                'candidates are refined.')
        else:
            v = self.ctx.store.project.variant
            self.start_note.setText(f'{v.name}: every visible camera; locked ones stay, mirrored pairs move '
                                    f'together.')

    def chosen(self) -> list:
        """The Pareto candidates to refine: the selected rows of the candidate table, else the first
        `candidates` (settings)."""
        pg = self.page
        if pg.opt is None or not pg.cands:
            return []
        rows = sorted({i.row() for i in pg.table.selectionModel().selectedRows()})
        n = self.settings().candidates
        rows = [r for r in rows if 0 <= r < len(pg.cands)] or list(range(min(n, len(pg.cands))))
        return [pg.cands[r] for r in rows]

    # ------------------------------------------------------------------ run
    def toggle(self):
        w = self.ctx.window
        if w is not None and 'diff_optim' in w._jobs:
            w._jobs['diff_optim']['cancel'].set()
            self.b_run.setText('Stopping…')
            return
        self.start_run()

    def start_run(self) -> bool:
        st = self.settings()
        snap = self.ctx.store.project.snapshot()
        variant = snap.variant
        starts = None
        if st.start == 'pareto':
            cands = self.chosen()
            opt = self.page.opt
            if not cands or opt is None:
                self.ctx.msg('Run the evolutionary search first: its Pareto candidates are refined.', 5000)
                return False
            variant = next((v for v in snap.variants if v.uid == opt.variant.uid), None)
            if variant is None:
                self.ctx.msg('The candidates belong to a variant that is gone. Run the search again.', 5000)
                return False
            starts = DO.starts_from_optimiser(opt, cands)
        eng = self.ctx.analysis.engine.share()
        eng.vehicle_geom(snap, variant)
        self.base_variant_uid = variant.uid
        self._live = {}
        self.result = None
        self.which.clear()
        self.b_add.setEnabled(False)
        sig = self.sig

        def live(label_, hist, checks):
            now = time.perf_counter()
            if now - self._last_emit < 0.15:
                return
            self._last_emit = now
            sig.live.emit((label_, [dict(h) for h in hist], [dict(c) for c in checks]))

        def fn(progress, cancel):
            def prog(f):
                progress(f, f'{f * 100:.0f} %')
                sig.prog.emit(f)
            return DO.run(snap, eng, st, variant, prog, cancel, starts, live)
        ok = self.ctx.window.run_job('diff_optim', 'Gradient refinement', fn, self.finished, self.failed)
        if ok:
            self.b_run.setText('Cancel')
            self.b_run.setIcon(icons.icon('stop', 12))
            self.bar.setValue(0)
            self.status.setText('Refining…')
        return ok

    def _on_live(self, payload):
        label_, hist, checks = payload
        self._live[label_] = (hist, checks)
        if self.which.findText(label_) < 0:
            self.which.blockSignals(True)
            self.which.addItem(label_)
            self.which.blockSignals(False)
        if self.result is None:                 # a run going on: follow the start being refined
            self.chart.set_data(hist, checks)

    def _reset_button(self):
        self.b_run.setText('Refine')
        self.b_run.setIcon(icons.icon('play', 12))

    def failed(self, msg):
        self._reset_button()
        self.status.setText('Stopped.')

    def finished(self, res):
        self._reset_button()
        if res is None or res.variant_uid != self.base_variant_uid:
            return
        self.result = res
        self.bar.setValue(1000)
        self.which.blockSignals(True)
        self.which.clear()
        for r in res.refinements:
            self.which.addItem(r.label)
        self.which.blockSignals(False)
        better = sum(r.improved for r in res.refinements)
        self.status.setText(f'{"Cancelled · " if res.cancelled else ""}{len(res.refinements)} start'
                            f'{"s" * (len(res.refinements) != 1)} refined, {better} better in the exact check · '
                            f'{res.seconds:.1f} s')
        self.b_export.setEnabled(True)
        self._fill_findings(res.findings)
        best = next((i for i, r in enumerate(res.refinements) if r.improved), 0)
        self.which.setCurrentIndex(best)
        self.show_selected()

    def current(self):
        if self.result is None:
            return None
        i = self.which.currentIndex()
        return self.result.refinements[i] if 0 <= i < len(self.result.refinements) else None

    def show_selected(self):
        r = self.current()
        if r is None:
            return
        self.chart.set_data(r.history, r.checks)
        for i, (k, name, unit, dec) in enumerate(self.METRICS):
            it = QTableWidgetItem(name + (f' ({unit})' if unit else ''))
            self.metrics.setItem(i, 0, it)
            vals = [r.exact_before.get(k), r.exact_after.get(k), r.surrogate_before.get(k), r.surrogate_after.get(k)]
            for j, v in enumerate(vals):
                txt = '—' if v is None else (f'{v:.0f}' if k in ('unresolved', 'seam_near') and j < 2 else
                                             f'{v:.{dec}f}')
                c = _num_cell(self.metrics, i, j + 1, txt)
                if j == 1 and v is not None and r.exact_before.get(k) is not None:
                    up_good = k == 'min_density'
                    d = v - r.exact_before[k]
                    if abs(d) > 1e-9:
                        good = (d > 0) == up_good
                        c.setForeground(THEME.color('ink') if good else QColor(STATUS.get('warning', '#d08a2e')))
                        c.setIcon(icons.status_icon('good' if good else 'warning', 10))
        deltas = r.deltas()
        self.cams.setRowCount(len(deltas))
        st = self.result.settings
        for i, d in enumerate(deltas):
            tag = QTableWidgetItem(d['tag'])
            tag.setFont(mono(12))
            cam = next((c for c in r.cams_before if c.uid == d['uid']), None)
            if cam is not None:
                tag.setIcon(_chip_icon(cam.slot))
            self.cams.setItem(i, 0, tag)
            role = {'moves': 'moves', 'mirrors': 'mirrors its pair', 'locked': 'locked', 'hidden': 'hidden'}
            self.cams.setItem(i, 1, QTableWidgetItem(role.get(d['role'], d['role'])))
            lims = [st.max_move_mm] * 3 + [st.max_turn_deg] * 2 + [st.max_roll_deg]
            for j, (v, m) in enumerate(zip(d['delta'], lims)):
                c = _num_cell(self.cams, i, j + 2, f'{v:+.1f}' if j < 3 else f'{v:+.2f}')
                if d['role'] == 'moves' and m > 0 and abs(v) >= m - (0.5 if j < 3 else 0.05):
                    c.setToolTip('At its limit')
                    c.setForeground(QColor(STATUS.get('warning', '#d08a2e')))
        self.b_add.setEnabled(r.improved)
        self.b_add.setToolTip('Add the refined layout of this start as a new variant, in one undo step (the active '
                              'variant stays)' if r.improved else 'The exact check kept the start: nothing to add')

    def _fill_findings(self, findings):
        for w in self._find_rows:
            w.hide()
            w.deleteLater()
        self._find_rows = []
        for f in findings[:8]:
            row = StatusLine(f['severity'] if f['severity'] != 'note' else '', f['message'])
            row.text.setProperty('ink2', True)
            self.find_box.addWidget(row)
            self._find_rows.append(row)

    # ------------------------------------------------------------------ results out
    def add_variant(self):
        """The selected start's refined layout as a new variant (one undo step); returns it (None: nothing to add)."""
        r = self.current()
        if r is None or not r.improved:
            return None
        p = self.ctx.store.project
        base = next((v for v in p.variants if v.uid == self.result.variant_uid), None)
        if base is None:
            self.ctx.msg('The variant this refinement started from is gone.', 5000)
            return None
        taken = {v.name for v in p.variants}
        stem = f'{base.name} · refined' if r.label == 'current' else f'{base.name} · {r.label} refined'
        name, k = stem, 2
        while name in taken:
            name, k = f'{stem} ({k})', k + 1
        nv = DO.to_variant(r, base, name)

        def fn(pj):
            pj.variants.append(nv)
        self.ctx.store.edit('variants', fn, f'Added the gradient refinement as {name}')
        if self.ctx.window is not None:
            self.ctx.window._fill_variant_combo()
        self.ctx.msg(f'Added variant {name}', 5000)
        return nv

    def export(self, folder=None):
        if self.result is None:
            return []
        if folder is None:
            folder = QFileDialog.getExistingDirectory(self, 'Export the gradient refinement into')
            if not folder:
                return []
        import os
        p = self.ctx.store.project
        paths = []
        for name, data in DO.files(self.result, p):
            path = os.path.join(folder, name)
            with open(path, 'wb') as fh:
                fh.write(data)
            paths.append(path)
        from ...core import provenance as PV
        paths = PV.stamp_paths(paths, p, folder)
        self.ctx.msg(f'Exported the gradient refinement to {folder}', 5000)
        return paths

    def project_replaced(self):
        w = self.ctx.window
        job = w._jobs.get('diff_optim') if w is not None else None
        if job:
            job['cancel'].set()
        self.result = None
        self.base_variant_uid = ''
        self._live = {}
        self.which.clear()
        self.chart.set_data([], [])
        self.metrics.clearContents()
        self.cams.setRowCount(0)
        self._fill_findings([])
        self.status.setText('')
        self.bar.setValue(0)
        self.b_add.setEnabled(False)
        self.b_export.setEnabled(False)
        self.load()
