"""B2 · Tolerance study (Monte Carlo + sensitivity + pinned worst cases)."""
from __future__ import annotations

import copy
import os
import time

import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (QWidget, QHBoxLayout, QVBoxLayout, QGridLayout, QSplitter, QCheckBox, QComboBox, QTableWidget,
                             QTableWidgetItem, QHeaderView, QAbstractItemView, QPushButton, QProgressBar, QFileDialog,
                             QScrollArea, QFrame)

from ...core.tolerance import TolSettings
from ..charts import Histogram, Tornado
from ..planview import PlanView
from ..scene import MainScene
from ..theme import THEME, mono, sans
from ..viewport3d import make_viewport
from ..widgets import label, button, ScrubSpinBox, Chip, ROW_H, LABEL_W, hline


def describe(tag, d):
    parts = []
    syms = ('ψ', 'θ', 'φ')
    for k in range(3):
        if abs(d[k]) >= 0.05:
            parts.append(f'{syms[k]} {d[k]:+.1f}°')
    for k, ax in zip(range(3, 6), 'xyz'):
        if abs(d[k]) >= 0.0005:
            parts.append(f'{ax} {d[k] * 1000:+.0f} mm')
    return (f'{tag}  ' + '  '.join(parts)).replace('-', '−')


class TolerancePage(QWidget):
    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        self.result = None
        self.setObjectName('base')
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        split = QSplitter(Qt.Orientation.Horizontal)
        split.setHandleWidth(2)
        # ---------------- left settings
        left = QWidget()
        left.setObjectName('panel')
        left.setMinimumWidth(300)
        ll = QVBoxLayout(left)
        ll.setContentsMargins(8, 6, 8, 8)
        ll.setSpacing(4)
        top = QHBoxLayout()
        top.addWidget(label('Tolerance study'))
        top.addStretch(1)
        top.addWidget(label('Monte Carlo', muted=True, small=True))
        ll.addLayout(top)
        ll.addWidget(hline())
        ll.addWidget(label('Perturb', bold=True))
        self.cam_box = QGridLayout()
        self.cam_box.setHorizontalSpacing(12)
        ll.addLayout(self.cam_box)
        self.cam_checks = {}
        grid = QGridLayout()
        grid.setVerticalSpacing(2)
        self.yaw = ScrubSpinBox('', 'angle', 1, '°', width=100)
        self.pitch = ScrubSpinBox('', 'angle', 1, '°', width=100)
        self.roll = ScrubSpinBox('', 'angle', 1, '°', width=100)
        self.pos = ScrubSpinBox('', 'plain', 0, 'mm', step=1, width=100)
        self.dist = QComboBox()
        self.dist.addItem('Uniform', 'uniform')
        self.dist.addItem('Normal (± = 2σ)', 'normal')
        self.dist.setFixedWidth(100)
        self.samples = ScrubSpinBox('', 'plain', 0, '', step=50, width=100)
        self.seed = ScrubSpinBox('', 'plain', 0, '', step=1, width=100)
        self.cells = ScrubSpinBox('', 'plain', 0, 'cm', step=1, width=100)
        for w, v in ((self.yaw, 5.0), (self.pitch, 5.0), (self.roll, 5.0), (self.pos, 10), (self.samples, 2000), (self.seed, 1729), (self.cells, 5)):
            w.set_value(v)
        for i, (t, w) in enumerate((('Yaw ±', self.yaw), ('Pitch ±', self.pitch), ('Roll ±', self.roll), ('Position ±', self.pos),
                                    ('Distribution', self.dist), ('Samples', self.samples), ('Seed', self.seed), ('Ground cells', self.cells))):
            a = label(t, ink2=True)
            a.setFixedWidth(120)
            grid.addWidget(a, i, 0)
            grid.addWidget(w, i, 1)
        grid.setColumnStretch(2, 1)
        ll.addLayout(grid)
        self.corr = QCheckBox('Correlate L/R pairs')
        ll.addWidget(self.corr)
        ll.addWidget(label('Off: ML and MR err independently, as mirror brackets do.', muted=True, small=True, wrap=True))
        ll.addSpacing(6)
        ll.addWidget(label('Metrics', bold=True))
        self.m_blind = QCheckBox('Blind area in ROI')
        self.m_dens = QCheckBox('Minimum density')
        self.m_seam = QCheckBox('Seam position shift')
        for c in (self.m_blind, self.m_dens, self.m_seam):
            c.setChecked(True)
            ll.addWidget(c)
        ll.addStretch(1)
        foot = QHBoxLayout()
        self.last = label('No run yet', muted=True, small=True)
        self.count = label('', muted=True, small=True, mono_=True)
        foot.addWidget(self.last)
        foot.addStretch(1)
        foot.addWidget(self.count)
        ll.addLayout(foot)
        self.bar = QProgressBar()
        self.bar.setTextVisible(False)
        self.bar.setRange(0, 1000)
        ll.addWidget(self.bar)
        bl = QHBoxLayout()
        bl.addStretch(1)
        self.b_csv = button('Export CSV')
        self.b_csv.clicked.connect(self.export_csv)
        self.b_run = button('Run study', 'primary')
        self.b_run.clicked.connect(self.run)
        bl.addWidget(self.b_csv)
        bl.addWidget(self.b_run)
        ll.addLayout(bl)
        split.addWidget(left)
        # ---------------- centre
        centre = QSplitter(Qt.Orientation.Vertical)
        centre.setHandleWidth(2)
        charts = QSplitter(Qt.Orientation.Horizontal)
        charts.setHandleWidth(2)
        hw = QWidget()
        hw.setObjectName('panel')
        hl = QVBoxLayout(hw)
        hl.setContentsMargins(8, 4, 8, 6)
        self.hist = Histogram(log=True)
        self.hist.title = 'Blind area in ROI'
        hl.addWidget(self.hist, 1)
        self.stats = label('', mono_=True)
        hl.addWidget(self.stats)
        charts.addWidget(hw)
        tw = QWidget()
        tw.setObjectName('panel')
        tl = QVBoxLayout(tw)
        tl.setContentsMargins(8, 4, 8, 6)
        self.tornado = Tornado()
        self.tornado.title = 'Sensitivity · one parameter at a time'
        tl.addWidget(self.tornado)
        charts.addWidget(tw)
        centre.addWidget(charts)
        ww = QWidget()
        ww.setObjectName('panel')
        wl = QVBoxLayout(ww)
        wl.setContentsMargins(8, 4, 8, 6)
        h = QHBoxLayout()
        h.addWidget(label('Worst cases'))
        h.addStretch(1)
        h.addWidget(label('click a row to preview · Pin keeps it in the right-hand view', muted=True, small=True))
        wl.addLayout(h)
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(['#', 'Perturbation', 'Blind m²', 'Min px/cm', 'Where', ''])
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(ROW_H + 1)
        self.table.horizontalHeader().setFixedHeight(ROW_H)
        self.table.setShowGrid(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        hh.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        for c in (2, 3, 5):
            hh.setSectionResizeMode(c, QHeaderView.ResizeMode.ResizeToContents)
        self.table.cellClicked.connect(lambda r, c: self.preview(r))
        wl.addWidget(self.table, 1)
        centre.addWidget(ww)
        centre.setSizes([430, 400])
        split.addWidget(centre)
        # ---------------- right: pinned views
        right = QSplitter(Qt.Orientation.Vertical)
        right.setHandleWidth(2)
        self.pin_plan = PlanView(ctx, 'footprints', interactive=False)
        self.pin_plan.legend = False
        self.pin_plan.result_fn = lambda: self.pinned_result
        self.pin_plan.cameras_fn = lambda: self.pinned_cams or ctx.store.variant.cameras
        right.addWidget(self.pin_plan)
        self.pin_scene = MainScene(ctx)
        self.pin_scene.show_tooltip = False
        self.pin_3d = make_viewport(self.pin_scene)
        right.addWidget(self.pin_3d)
        split.addWidget(right)
        split.setSizes([315, 1060, 625])
        lay.addWidget(split)
        self.pinned_result = None
        self.pinned_cams = None
        self.pinned_index = None
        ctx.store.changed.connect(lambda *_: self._cams())
        ctx.store.project_replaced.connect(self._cams)
        ctx.window.jobs.progress.connect(self._progress)
        ctx.window.jobs.failed.connect(lambda k, m: self.b_run.setEnabled(True) if k == 'tolerance' else None)
        self._cams()

    def activate(self):
        self._cams()
        QTimer.singleShot(30, lambda: self.pin_3d.frame_box(*self.ctx.window.viewarea._bbox()))

    def _cams(self):
        while self.cam_box.count():
            w = self.cam_box.takeAt(0).widget()
            if w:
                w.deleteLater()
        prev = {u: c.isChecked() for u, c in self.cam_checks.items()}
        self.cam_checks = {}
        for i, c in enumerate(self.ctx.store.variant.cameras):
            w = QWidget()
            h = QHBoxLayout(w)
            h.setContentsMargins(0, 0, 0, 0)
            h.setSpacing(4)
            cb = QCheckBox()
            cb.setChecked(prev.get(c.uid, True))
            h.addWidget(cb)
            h.addWidget(Chip(c.slot, c.tag))
            self.cam_box.addWidget(w, i // 3, i % 3)
            self.cam_checks[c.uid] = cb

    def settings(self) -> TolSettings:
        return TolSettings(yaw=self.yaw.value(), pitch=self.pitch.value(), roll=self.roll.value(), pos_mm=self.pos.value(),
                           distribution=self.dist.currentData(), samples=int(max(10, self.samples.value())),
                           seed=int(self.seed.value()), cell=max(0.01, self.cells.value() / 100),
                           cameras=[u for u, c in self.cam_checks.items() if c.isChecked()], correlate_pairs=self.corr.isChecked(),
                           metric_blind=self.m_blind.isChecked(), metric_density=self.m_dens.isChecked(),
                           metric_seam=self.m_seam.isChecked())

    def run(self):
        self.b_run.setEnabled(False)
        self.ctx.window.run_tolerance(self.settings())

    def _progress(self, kind, frac, text):
        if kind == 'tolerance':
            self.bar.setValue(int(frac * 1000))
            st = self.settings()
            self.count.setText(f'{int(frac * st.samples)} / {st.samples}')

    def show_result(self, r):
        self.result = r
        self.b_run.setEnabled(True)
        st = r.settings
        self.bar.setValue(int(1000 * len(r.blind) / max(st.samples, 1)))
        self.count.setText(f'{len(r.blind)} / {st.samples}' + (' · cancelled' if r.cancelled else ''))
        self.last.setText(f'Last run {time.strftime("%H:%M", time.localtime(r.finished_at))} · {r.seconds:.1f} s · {r.threads} threads'
                          + (f' · cancelled after {len(r.blind)} samples' if r.cancelled else ''))
        self.hist.title = f'Blind area in ROI · {len(r.blind)} samples' + (' (cancelled)' if r.cancelled else '')
        self.hist.set_data(r.blind, bins=20, markers=[(r.pct(95), 'P95', True), (r.pct(99), 'P99', True), (float(r.blind.max()), 'max', False)],
                           xlabel='blind area, m²')
        self.stats.setText(f'zero {r.zero_pct:.1f} %    P95 {r.pct(95):.3f}    P99 {r.pct(99):.3f}    max {r.blind.max():.3f} m²'
                           f'    nominal {r.nominal_blind:.3f} m²')
        self.tornado.set_rows([(lab, m, p) for lab, uid, sym, m, p in r.sensitivity],
                              left=f'−{st.yaw:g}° / −{st.pos_mm:g} mm', right=f'+{st.yaw:g}° / +{st.pos_mm:g} mm')
        st_ = self.ctx.store
        self.table.setRowCount(len(r.worst))
        for i, case in enumerate(r.worst):
            cam = st_.cam(case.culprit)
            d = case.perturb.get(case.culprit, (0,) * 6)
            vals = [str(i + 1), describe(cam.tag if cam else '?', d), f'{case.blind:.3f}', f'{case.min_density:.1f}', case.where]
            for j, v in enumerate(vals):
                it = QTableWidgetItem(v)
                if j in (1, 2, 3):
                    it.setFont(mono(12))
                if j in (2, 3):
                    it.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                if j == 0:
                    it.setForeground(THEME.color('muted'))
                self.table.setItem(i, j, it)
            b = QPushButton('Pin')
            b.clicked.connect(lambda _=False, i=i: self.pin(i))
            self.table.setCellWidget(i, 5, b)
        if r.worst:
            self.pin(0)

    def _perturbed_cams(self, case):
        from ...core.mathutil import wrap180
        cams = []
        for c in self.ctx.store.variant.cameras:
            c2 = copy.deepcopy(c)
            d = case.perturb.get(c.uid)
            if d is not None:
                c2.yaw = wrap180(c.yaw + d[0])
                c2.pitch += d[1]
                c2.roll += d[2]
                c2.x += d[3]
                c2.y += d[4]
                c2.z += d[5]
            cams.append(c2)
        return cams

    def preview(self, i, pin=False):
        if self.result is None or not (0 <= i < len(self.result.worst)):
            return
        case = self.result.worst[i]
        cams = self._perturbed_cams(case)
        eng = self.ctx.analysis.engine
        with self.ctx.analysis._lock:
            res = eng.analyse(self.ctx.store.project.snapshot(), cameras=cams)
        self.pinned_result = res
        self.pinned_cams = cams
        self.pin_scene.result_override = res
        self.pin_scene.cameras_override = cams
        self.pin_scene._ground_key = None
        cam = self.ctx.store.cam(case.culprit)
        d = case.perturb.get(case.culprit, (0,) * 6)
        self.pin_plan.label = f'{"Pinned" if pin else "Preview"} #{i + 1} · top · blind {case.blind:.3f} m² {case.where}'
        self.pin_scene.label_prefix = f'{"Pinned" if pin else "Preview"} #{i + 1} · {describe(cam.tag if cam else "?", d)}'
        if case.centroid is not None:
            x, y = case.centroid
            self.pin_plan.zoom_rect = (x - 1.0, y - 0.7, x + 1.0, y + 0.7)
            self.pin_plan.pinned_marker = (x, y)
        else:
            self.pin_plan.zoom_rect = None
        self.pin_plan._img_key = None
        self.pin_plan.fit()
        self.pin_3d.update()
        if pin:
            self.pinned_index = i
            for k in range(self.table.rowCount()):
                w = self.table.cellWidget(k, 5)
                if w:
                    w.setText('Pinned' if k == i else 'Pin')
            self.ctx.window.sel_label.setText(f'pinned case #{i + 1}')

    def pin(self, i):
        self.preview(i, pin=True)

    def export_csv(self):
        if self.result is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, 'Export tolerance samples', 'tolerance.csv', 'CSV (*.csv)')
        if not path:
            return
        r = self.result
        with open(path, 'w') as fh:
            fh.write('sample,blind_m2,min_density_px_cm,seam_shift_deg\n')
            for i in range(len(r.blind)):
                fh.write(f'{i},{r.blind[i]:.5f},{r.min_density[i]:.3f},{r.seam_shift[i]:.2f}\n')
            fh.write('\n# sensitivity: parameter,delta_minus_m2,delta_plus_m2\n')
            for lab, uid, sym, m, p in r.sensitivity:
                fh.write(f'# {lab},{m:.5f},{p:.5f}\n')
        self.ctx.msg(f'Exported {len(r.blind)} samples to {path}')
