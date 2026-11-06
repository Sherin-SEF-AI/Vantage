"""Analysis ▸ Compare rigs…: the cross-rig domain gap (core.rig_gap) between two variants of the project, or a variant
and a rig read from a nuScenes or KITTI calibration folder (or a reference rig, when core.reference_rigs is present):
the components, the total, the camera matching and the findings, with the comparison figure."""
from __future__ import annotations

import threading

from PyQt6.QtCore import QObject, Qt, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (QComboBox, QFileDialog, QGridLayout, QHBoxLayout, QLabel, QTableWidget, QTableWidgetItem,
                             QVBoxLayout, QWidget, QHeaderView)

from ..widgets import button, label
from .base import VDialog

COMPONENTS = [('coverage', 'Viewing directions (pixels per steradian)'), ('resolution', 'Resolution where both see'),
              ('height_pitch', 'Camera heights and depressions'), ('fov', 'Fields of view'),
              ('topology', 'Camera count and overlaps'), ('bev', 'Ground seen to 50 m')]


class _Sig(QObject):
    done = pyqtSignal(object)


class RigGapDialog(VDialog):
    def __init__(self, ctx, parent=None):
        super().__init__(parent, 'Compare rigs', 'Cross-rig domain gap', (900, 640))
        self.ctx = ctx
        self.ext = None                         # a rig read from a folder
        self.sig = _Sig()
        self.sig.done.connect(self._done)
        p = ctx.store.project
        lay = QVBoxLayout(self.body)
        top = QGridLayout()
        self.a = QComboBox()
        self.b = QComboBox()
        for v in p.variants:
            if v.cameras:
                self.a.addItem(v.name, ('variant', v.uid))
                self.b.addItem(v.name, ('variant', v.uid))
        # findData cannot match a Python tuple stored as item data: look the active variant up by hand
        self.a.setCurrentIndex(next((i for i in range(self.a.count())
                                     if self.a.itemData(i) == ('variant', p.variant.uid)), 0))
        for rid, name in self._reference_rigs():
            self.b.addItem(f'Reference · {name}', ('reference', rid))
        self.b.addItem('From a nuScenes or KITTI calibration folder…', ('folder', ''))
        if self.b.count() > 1:
            self.b.setCurrentIndex(1 if self.b.itemData(0) == self.a.currentData() else 0)
        top.addWidget(label('Rig A', ink2=True), 0, 0)
        top.addWidget(self.a, 0, 1)
        top.addWidget(label('Rig B', ink2=True), 1, 0)
        top.addWidget(self.b, 1, 1)
        self.b_run = button('Compare', 'primary')
        self.b_run.clicked.connect(self.run)
        top.addWidget(self.b_run, 0, 2, 2, 1)
        lay.addLayout(top)
        self.headline = label('', wrap=True)
        lay.addWidget(self.headline)
        mid = QHBoxLayout()
        self.table = QTableWidget(len(COMPONENTS) + 1, 3)
        self.table.setHorizontalHeaderLabels(['Component', 'Gap (0–1)', 'Weight'])
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        mid.addWidget(self.table, 1)
        self.fig = QLabel()
        self.fig.setMinimumSize(380, 300)
        self.fig.setAlignment(Qt.AlignmentFlag.AlignCenter)
        mid.addWidget(self.fig, 1)
        lay.addLayout(mid, 1)
        self.findings = label('', wrap=True, muted=True)
        lay.addWidget(self.findings)
        lay.addWidget(label('An index of the geometric shift between the two rigs, not a predicted mAP: its transfer '
                            'band is a heuristic. Directions are compared as seen from each camera\'s centre; a rig read '
                            'from a folder has no body to hide its views.', muted=True, wrap=True))
        self.result = None
        b_close = button('Close', 'ghost')
        b_close.clicked.connect(self.reject)
        self.buttons.addWidget(b_close)

    @staticmethod
    def _reference_rigs():
        try:
            from ...core import reference_rigs as RR
            return [(r.id, f'{r.company} {r.name}') for r in RR.library() if r.placed]   # layouts with poses only
        except Exception:                       # not installed or failing: the variants and folders still work
            return []

    def _rig(self, data):
        from ...core import rig_gap as RG
        kind, ident = data
        p = self.ctx.store.project
        eng = self.ctx.analysis.engine
        if kind == 'variant':
            return RG.rig_from_variant(p, p.variant_by_uid(ident), engine=eng)
        if kind == 'reference':
            from ...core import reference_rigs as RR
            q = RR.reference_project(ident)
            return RG.rig_from_variant(q, q.variant, engine=None)
        return self.ext

    def run(self):
        if self.b.currentData()[0] == 'folder' and self.ext is None:
            path = QFileDialog.getExistingDirectory(self, 'nuScenes or KITTI calibration folder')
            if not path:
                return
            from ...core import rig_gap as RG
            try:
                self.ext = RG.load_rig(path)
            except Exception as e:
                self.headline.setText(f'Could not read that folder: {e}')
                return
        self.b_run.setEnabled(False)
        self.headline.setText('Comparing…')
        da, db = self.a.currentData(), self.b.currentData()

        def work():
            from ...core import rig_gap as RG
            try:
                A, B = self._rig(da), self._rig(db)
                gap = RG.rig_gap(A, B)
                img = RG.figure(gap)
                self.sig.done.emit((gap, img, None))
            except Exception as e:
                self.sig.done.emit((None, None, str(e)))

        threading.Thread(target=work, daemon=True).start()

    def _done(self, out):
        gap, img, err = out
        self.b_run.setEnabled(True)
        if err:
            self.headline.setText(f'Could not compare: {err}')
            return
        self.result = gap
        self.headline.setText(f'Rig contrastive distance {gap.total:.3f} · transfer band: {gap.transfer_band} '
                              f'(heuristic) · {len(gap.matches)} cameras matched')
        for i, (k, name) in enumerate(COMPONENTS):
            self.table.setItem(i, 0, QTableWidgetItem(name))
            self.table.setItem(i, 1, QTableWidgetItem(f'{gap.components.get(k, float("nan")):.3f}'))
            self.table.setItem(i, 2, QTableWidgetItem(f'{gap.weights.get(k, 0.0):.2f}'))
        n = len(COMPONENTS)
        self.table.setItem(n, 0, QTableWidgetItem('Total'))
        self.table.setItem(n, 1, QTableWidgetItem(f'{gap.total:.3f}'))
        self.table.setItem(n, 2, QTableWidgetItem('1.00'))
        self.findings.setText('\n'.join(f'· {f}' for f in gap.findings[:6]))
        if img is not None:
            import numpy as np
            rgb = np.ascontiguousarray(img[..., ::-1])
            q = QImage(rgb.data, rgb.shape[1], rgb.shape[0], 3 * rgb.shape[1], QImage.Format.Format_RGB888).copy()
            self.fig.setPixmap(QPixmap.fromImage(q).scaled(self.fig.width(), self.fig.height(),
                                                           Qt.AspectRatioMode.KeepAspectRatio,
                                                           Qt.TransformationMode.SmoothTransformation))
