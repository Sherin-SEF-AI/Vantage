"""Analysis ▸ Blind volume: where around the vehicle something can stand unseen by every camera, in 3D."""
from __future__ import annotations

import json
import os

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (QWidget, QHBoxLayout, QVBoxLayout, QGridLayout, QSplitter, QTableWidget, QTableWidgetItem,
                             QHeaderView, QAbstractItemView, QProgressBar, QSlider, QFileDialog)

from ...core import volume as vol
from ..scene import MainScene
from ..theme import THEME, STATUS, mono
from ..viewport3d import make_viewport
from ..volumeview import RoseChart, VolumeMap, SERIES
from ..widgets import label, button, ScrubSpinBox, SegmentedControl, ROW_H, hline

FIELDS = (('height', 'Height', 'm', 2, 0.1, 1.0, 'Top of the analysed space above the ground'),
          ('dz', 'Level step', 'cm', 0, 1.0, 100.0, 'Spacing of the analysed heights'),
          ('cell', 'Columns', 'cm', 0, 1.0, 100.0, 'Spacing of the analysed columns on the ground; 0 = the analysis grid '
                                                    '(at least 5 cm)'))


class VolumePage(QWidget):
    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        self.ctl = ctx.volume
        self.setObjectName('base')
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        split = QSplitter(Qt.Orientation.Horizontal)
        split.setHandleWidth(2)
        split.addWidget(self._left())
        # ---------------- centre: the blind space in 3D
        self.scene = MainScene(ctx)
        self.scene.show_volume = True
        self.view = make_viewport(self.scene)
        self.view.cam.az, self.view.cam.el = 225.0, 32.0
        cw = QWidget()
        cw.setObjectName('panel')
        cl = QVBoxLayout(cw)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.setSpacing(0)
        cl.addWidget(self.view, 1)
        split.addWidget(cw)
        # ---------------- right: rose and plan
        right = QSplitter(Qt.Orientation.Vertical)
        right.setHandleWidth(2)
        rw = QWidget()
        rw.setObjectName('panel')
        rl = QVBoxLayout(rw)
        rl.setContentsMargins(8, 4, 8, 6)
        self.rose = RoseChart()
        rl.addWidget(self.rose, 1)
        right.addWidget(rw)
        mw = QWidget()
        mw.setObjectName('panel')
        ml = QVBoxLayout(mw)
        ml.setContentsMargins(8, 4, 8, 6)
        ml.setSpacing(4)
        head = QHBoxLayout()
        self.mode = SegmentedControl([('slice', 'Blind at height'), ('detect', 'Smallest object seen')], 'slice')
        self.mode.changed.connect(self._set_mode)
        head.addWidget(self.mode)
        head.addStretch(1)
        ml.addLayout(head)
        srow = QHBoxLayout()
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 0)
        self.slider.valueChanged.connect(self._set_level)
        self.z_label = label('', mono_=True)
        self.z_label.setFixedWidth(150)
        srow.addWidget(label('Height', ink2=True))
        srow.addWidget(self.slider, 1)
        srow.addWidget(self.z_label)
        self.slider_row = QWidget()
        self.slider_row.setLayout(srow)
        ml.addWidget(self.slider_row)
        self.map = VolumeMap()
        ml.addWidget(self.map, 1)
        right.addWidget(mw)
        right.setSizes([420, 480])
        split.addWidget(right)
        split.setSizes([320, 900, 560])
        lay.addWidget(split)
        self.ctl.changed.connect(self.refresh)
        ctx.store.changed.connect(lambda *_: self._sync_fields())
        ctx.store.project_replaced.connect(self._sync_fields)
        self._shown = None
        self._framed = False
        self._picked = False                # the user has moved the height slider
        self.slider.sliderMoved.connect(lambda *_: setattr(self, '_picked', True))
        self._sync_fields()
        self.refresh()

    def _left(self):
        w = QWidget()
        w.setObjectName('panel')
        w.setMinimumWidth(300)
        ll = QVBoxLayout(w)
        ll.setContentsMargins(8, 6, 8, 8)
        ll.setSpacing(4)
        top = QHBoxLayout()
        top.addWidget(label('Blind volume'))
        top.addStretch(1)
        top.addWidget(label('3D, every camera', muted=True, small=True))
        ll.addLayout(top)
        ll.addWidget(hline())
        ll.addWidget(label('The space around the vehicle, from the ground up, that no camera sees: the same test as the '
                           'ground analysis (lens model, sensor edges, the body, rig and housings) at every height. An '
                           'object standing on the ground is detected when a camera sees any part of it.', muted=True,
                           small=True, wrap=True))
        grid = QGridLayout()
        grid.setVerticalSpacing(2)
        self.fields = {}
        for i, (key, text, unit, dec, step, scale, tip) in enumerate(FIELDS):
            f = ScrubSpinBox('', 'plain', dec, unit, step=step, width=100)
            f.setToolTip(tip)
            f.committed.connect(lambda v, k=key, s=scale: self.ctl.set_settings(**{k: v / s}))
            a = label(text, ink2=True)
            a.setFixedWidth(110)
            grid.addWidget(a, i, 0)
            grid.addWidget(f, i, 1)
            if key == 'cell':
                grid.addWidget(label('0 = analysis grid', muted=True, small=True), i, 2)
            self.fields[key] = (f, scale)
        grid.setColumnStretch(3, 1)
        ll.addLayout(grid)
        ll.addSpacing(6)
        ll.addWidget(label('Result', bold=True))
        mg = QGridLayout()
        mg.setVerticalSpacing(2)
        self.metric = {}
        for i, (key, text) in enumerate((('blind', 'Blind volume'), ('pct', 'Share of the space'),
                                         ('ground', 'Blind on the ground'), ('never', 'Nothing seen up to the top'),
                                         ('covered', 'Covered up to (≤ 1 % blind)'))):
            a = label(text, ink2=True)
            a.setFixedWidth(170)
            v = label('—', mono_=True)
            mg.addWidget(a, i, 0)
            mg.addWidget(v, i, 1)
            self.metric[key] = v
        mg.setColumnStretch(2, 1)
        ll.addLayout(mg)
        ll.addSpacing(6)
        ll.addWidget(label('Objects standing around the vehicle', bold=True))
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(['Object', 'Height m', 'Unseen m²', 'Reach m'])
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(ROW_H)
        self.table.horizontalHeader().setFixedHeight(ROW_H)
        self.table.setShowGrid(False)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for c in range(1, 4):
            hh.setSectionResizeMode(c, QHeaderView.ResizeMode.ResizeToContents)
        self.table.setFixedHeight(ROW_H * 4 + 6)
        self.table.setToolTip('Unseen: ground area where an object of that height is not seen by any camera. '
                              'Reach: the farthest such place from the body.')
        ll.addWidget(self.table)
        ll.addStretch(1)
        self.status = label('', muted=True, small=True)
        ll.addWidget(self.status)
        self.bar = QProgressBar()
        self.bar.setTextVisible(False)
        self.bar.setRange(0, 1000)
        self.bar.hide()
        self.ctl.progress.connect(lambda f: self.bar.setValue(int(f * 1000)))
        ll.addWidget(self.bar)
        bl = QHBoxLayout()
        bl.addStretch(1)
        self.b_export = button('Export…', None, 'export', 'The blind surface as OBJ and the figures as JSON')
        self.b_export.clicked.connect(lambda: self.export())
        self.b_run = button('Run', 'primary')
        self.b_run.clicked.connect(lambda: self.ctl.run())
        bl.addWidget(self.b_export)
        bl.addWidget(self.b_run)
        ll.addLayout(bl)
        return w

    # ------------------------------------------------------------------ state
    def showEvent(self, e):
        super().showEvent(e)
        self.ctl.set_active(True)

    def hideEvent(self, e):
        super().hideEvent(e)
        self.ctl.set_active(False)

    def activate(self):
        self.ctl.set_active(True)

    def _sync_fields(self):
        st = self.ctl.settings.to_dict()
        for key, (f, scale) in self.fields.items():
            if not f.hasFocus():
                f.set_value(st[key] * scale)

    def _set_mode(self, m):
        self.map.set_mode(m)
        self.slider_row.setVisible(m == 'slice')

    def _set_level(self, k):
        self.map.set_level(k)
        vr = self.ctl.result
        if vr is not None and 0 <= k < len(vr.z):
            self.z_label.setText(f'{vr.z[k]:.2f} m · {vr.level_blind_m2[k]:.2f} m²')
            self.ctl.set_cut(float(vr.z[k]))       # the 3D views show the blind space up to here
        self._rose()

    def refresh(self):
        ctl = self.ctl
        vr = ctl.result
        self.b_run.setEnabled(not (ctl.busy and self.ctx.window is not None and 'volume' in self.ctx.window._jobs))
        self.bar.setVisible(ctl.busy)
        self.b_export.setEnabled(vr is not None)
        st = ctl.status + (' · stale: the analysis or the settings changed' if ctl.stale and vr is not None else '')
        self.status.setText(st)
        if vr is self._shown:
            return
        self._shown = vr
        self.map.set_result(vr)
        self.scene.changed.emit()
        if vr is not None and not self._framed:
            self._framed = True
            self.view.frame_box((vr.x0, vr.y0, 0.0), (vr.x0 + vr.nx * vr.cell, vr.y0 + vr.ny * vr.cell, 1.0), 0.9)
        if vr is None:
            for v in self.metric.values():
                v.setText('—')
            self.table.setRowCount(0)
            self.rose.set_data([], [], 1.0)
            self.slider.setRange(0, 0)
            return
        m = vr.metrics
        self.metric['blind'].setText(f'{m["blind_m3"]:.2f} m³')
        self.metric['pct'].setText(f'{m["blind_pct"]:.1f} % of {m["roi_m3"]:.1f} m³')
        self.metric['ground'].setText(f'{m["ground_blind_m2"]:.3f} m²')
        self.metric['never'].setText(f'{m["never_seen_m2"]:.3f} m²')
        self.metric['covered'].setText(f'{m["covered_to_m"]:.2f} m')
        self.table.setRowCount(len(vr.objects))
        for i, (name, h) in enumerate(vr.objects):
            o = m['objects'][name]
            vals = [name, f'{h:.2f}', f'{o["undetected_m2"]:.3f}', f'{o["reach_m"]:.2f}']
            for j, v in enumerate(vals):
                it = QTableWidgetItem(v)
                if j:
                    it.setFont(mono(12))
                    it.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                if j == 2:
                    a = o['undetected_m2']
                    it.setForeground(QColor(STATUS['good' if a < 1e-9 else 'warning' if a <= 0.02 else 'critical']))
                self.table.setItem(i, j, it)
        k = min(self.slider.value(), len(vr.z) - 1) if self._picked else vr.level_index(self.ctl.cut_z)
        self.slider.blockSignals(True)
        self.slider.setRange(0, len(vr.z) - 1)
        self.slider.setValue(k)
        self.slider.blockSignals(False)
        self._set_level(k)

    def _rose(self):
        vr = self.ctl.result
        if vr is None:
            return
        series = [(name, vr.reach_object[name], QColor(SERIES[i % len(SERIES)])) for i, (name, _) in enumerate(vr.objects)]
        k = self.slider.value()
        series.append((f'Anything at {vr.z[k]:.2f} m', vr.reach_level[k], THEME.color('ink2')))
        rmax = float(max(vr.dist[vr.roi].max() if vr.roi.any() else 1.0, 0.5))
        self.rose.set_data(vr.sector_deg, series, rmax)

    # ------------------------------------------------------------------ export
    def export(self, folder=None):
        vr = self.ctl.result
        if vr is None:
            return []
        if folder is None:
            folder = QFileDialog.getExistingDirectory(self, 'Export the blind volume into')
            if not folder:
                return []
        paths = [os.path.join(folder, 'blind_volume.json'), os.path.join(folder, 'blind_volume.obj')]
        with open(paths[0], 'w', encoding='utf-8') as fh:
            fh.write(json.dumps(vol.to_doc(vr, self.ctx.store.project), indent=1) + '\n')
        with open(paths[1], 'w', encoding='utf-8') as fh:
            fh.write(vol.to_obj(vr))
        from ...core import provenance as PV          # where the JSON came from
        PV.stamp_paths(paths, self.ctx.store.project, folder)
        self.ctx.msg(f'Exported the blind volume to {folder}', 5000)
        return paths
