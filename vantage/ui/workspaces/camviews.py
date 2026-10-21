"""B1 · Camera views, stitched bird's-eye and bowl view."""
from __future__ import annotations

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (QWidget, QHBoxLayout, QVBoxLayout, QGridLayout, QSplitter, QTableWidget, QTableWidgetItem,
                             QHeaderView, QAbstractItemView)

from ..camview import CamImageView
from ..planview import PlanView
from ..scene import MainScene
from ..theme import THEME, mono
from ..viewport3d import make_viewport
from ..widgets import label, ROW_H


class CamViewsPage(QWidget):
    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        self.setObjectName('base')
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        split = QSplitter(Qt.Orientation.Horizontal)
        split.setHandleWidth(2)
        left = QWidget()
        left.setObjectName('base')
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.setSpacing(2)
        self.grid_w = QWidget()
        self.grid_w.setObjectName('base')
        self.grid = QGridLayout(self.grid_w)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setSpacing(2)
        ll.addWidget(self.grid_w, 1)
        seams = QWidget()
        seams.setObjectName('panel')
        sl = QVBoxLayout(seams)
        sl.setContentsMargins(8, 4, 8, 4)
        sl.setSpacing(2)
        head = QHBoxLayout()
        head.addWidget(label('Stitch seams'))
        head.addStretch(1)
        self.seam_note = label('', muted=True, small=True)
        head.addWidget(self.seam_note)
        sl.addLayout(head)
        self.table = QTableWidget(3, 1)
        self.table.verticalHeader().setDefaultSectionSize(ROW_H)
        self.table.horizontalHeader().setFixedHeight(ROW_H)
        self.table.setShowGrid(False)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.DoubleClicked)
        self.table.verticalHeader().setFixedWidth(120)
        self.table.setVerticalHeaderLabels(['Angle °', 'Blend m', 'Density at seam px/cm'])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setFixedHeight(ROW_H * 4 + 4)
        self.table.itemChanged.connect(self._edited)
        sl.addWidget(self.table)
        ll.addWidget(seams)
        split.addWidget(left)
        right = QSplitter(Qt.Orientation.Vertical)
        right.setHandleWidth(2)
        self.bev = PlanView(ctx, 'bev')
        self.bev.legend = False
        self.bev.seamMoved.connect(lambda: None)
        right.addWidget(self.bev)
        self.bowl_scene = MainScene(ctx, mode='bowl')
        self.bowl = make_viewport(self.bowl_scene)
        self.bowl.cam.az, self.bowl.cam.el = 235.0, 38.0
        right.addWidget(self.bowl)
        split.addWidget(right)
        split.setSizes([1230, 770])
        lay.addWidget(split)
        self.views = []
        self._uids = None
        self._busy = False
        ctx.store.changed.connect(lambda *_: self.rebuild())
        ctx.store.project_replaced.connect(self.rebuild)
        ctx.analysis.finished.connect(lambda *_: self.refresh())

    def activate(self):
        self.rebuild()
        self.bev.fit()
        res = self.ctx.analysis.result
        if res is not None:
            import numpy as np
            g = res.grid
            lo = np.array([g.x0, g.y0, 0.0])
            hi = np.array([g.x0 + g.nx * g.cell, g.y0 + g.ny * g.cell, 1.0])
            QTimer.singleShot(30, lambda: self.bowl.frame_box(lo, hi, 0.95))

    def rebuild(self):
        cams = self.ctx.store.variant.cameras
        uids = [c.uid for c in cams]
        if uids == self._uids:
            self.refresh()
            return
        self._uids = uids
        for v in self.views:
            self.grid.removeWidget(v)
            v.deleteLater()
        self.views = []
        cols = 3 if len(cams) <= 6 else 4
        for i, c in enumerate(cams):
            v = CamImageView(self.ctx, uid=c.uid, header=True, legend=False)
            self.grid.addWidget(v, i // cols, i % cols)
            self.views.append(v)
        self.refresh()

    def refresh(self):
        res = self.ctx.analysis.result
        n = len(self.ctx.store.variant.cameras)
        self.bev.label = f'Bird’s-eye · stitched from {n} cameras · drag a handle to move a seam'
        self.bev.update()
        if res is None:
            return
        self._busy = True
        st = self.ctx.store
        seams = res.seams
        self.table.setColumnCount(len(seams))
        heads = []
        for j, s in enumerate(seams):
            a, b = st.cam(s.a), st.cam(s.b)
            heads.append(f'{a.tag if a else "?"} | {b.tag if b else "?"}')
            for i, val in enumerate((f'{s.angle:.1f}', f'{st.variant.blend_m:.2f}', f'{s.density:.1f}')):
                it = QTableWidgetItem(val.replace('-', '−'))
                it.setFont(mono(12))
                it.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                if i == 2:
                    it.setFlags(it.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.table.setItem(i, j, it)
        self.table.setHorizontalHeaderLabels(heads)
        cx, cy = res.centre
        o = self.ctx.store.project.origin_offset()
        self.seam_note.setText(f'angles from the body centre ({cx - o[0]:.3f}, {cy - o[1]:.3f}) · drag the ■ handles in the bird’s-eye view')
        self._busy = False

    def _edited(self, it):
        if self._busy:
            return
        res = self.ctx.analysis.result
        if res is None or it.column() >= len(res.seams):
            return
        s = res.seams[it.column()]
        try:
            v = float(it.text().replace('−', '-'))
        except ValueError:
            self.refresh()
            return
        st = self.ctx.store
        if it.row() == 0:
            st.edit('variant', lambda p: p.variant.seams.__setitem__(f'{s.a}|{s.b}', round(v, 1)), 'Seam angle')
        elif it.row() == 1:
            st.edit('variant', lambda p: setattr(p.variant, 'blend_m', max(0.0, v)), 'Seam blend width')
        self.ctx.analysis.request()
