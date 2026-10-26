"""B3 · Optimise placement."""
from __future__ import annotations

import copy
import time

import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (QWidget, QHBoxLayout, QVBoxLayout, QGridLayout, QSplitter, QRadioButton, QButtonGroup,
                             QCheckBox, QComboBox, QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView,
                             QProgressBar)

from ...core.optimiser import OptSettings, default_vars, FreeVar
from ..charts import Scatter
from ..planview import PlanView
from ..theme import THEME, mono, STATUS
from .. import icons
from ..widgets import label, button, ScrubSpinBox, Chip, ROW_H, hline, StatusLine


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
        split.addWidget(centre)
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
