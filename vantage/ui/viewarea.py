"""Central viewport area: single / side-by-side / 2x2 panes, each a 3D view, a plan view or a camera view."""
from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal, QEvent
from PyQt6.QtGui import QPainter, QPen
from PyQt6.QtWidgets import QWidget, QSplitter, QVBoxLayout, QStackedLayout, QHBoxLayout

from .camview import CamImageView
from .planview import PlanView
from .scene import MainScene
from .theme import THEME
from .viewport3d import make_viewport
from .widgets import tool_button

PLAN_LABELS = {'coverage': 'Coverage by best camera', 'density': 'Pixel density · best camera', 'redundancy': 'Redundancy',
               'footprints': 'Footprints', 'bev': 'Bird’s-eye · stitched', 'targets': 'Floor plan · redundancy',
               'stitched': 'Objects in the stitched view · hover to see a pole’s smears'}


class Pane(QWidget):
    activated = pyqtSignal(object)

    def __init__(self, ctx, kind='persp'):
        super().__init__()
        self.ctx = ctx
        self.kind = kind
        self.active = False
        self.stack = QStackedLayout(self)
        self.stack.setContentsMargins(0, 0, 0, 0)
        self.scene = MainScene(ctx)
        self.vp3d = make_viewport(self.scene)
        self.vp3d.nav_style = ctx.nav_style
        self.plan = PlanView(ctx, 'coverage')
        self.cam = CamImageView(ctx, legend=True)
        for w in (self.vp3d, self.plan, self.cam):
            self.stack.addWidget(w)
            w.installEventFilter(self)
        # hover controls (split / maximise)
        self.ctrl = QWidget(self)
        hl = QHBoxLayout(self.ctrl)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.setSpacing(2)
        self.b_max = tool_button('maximise', 'Maximise viewport (Ctrl+Space)', size=20, icon_size=12)
        self.b_max.clicked.connect(lambda: ctx.window.toggle_maximise())
        hl.addWidget(self.b_max)
        self.ctrl.hide()
        ctx.overlays_changed.connect(self._labels)
        ctx.store.selection_changed.connect(self._labels)
        ctx.analysis.finished.connect(lambda *_: self._labels())
        self.set_kind(kind)

    def eventFilter(self, obj, e):
        if e.type() == QEvent.Type.MouseButtonPress:
            self.activated.emit(self)
        elif e.type() == QEvent.Type.Enter:
            self.ctrl.show()
            self.ctrl.raise_()
        elif e.type() == QEvent.Type.Leave:
            self.ctrl.hide()
        return False

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self.ctrl.adjustSize()
        self.ctrl.move(self.width() - self.ctrl.width() - 6, 4)

    def set_kind(self, kind):
        self.kind = kind
        if kind in ('persp', 'side', 'front', 'rear', 'top3d', 'bowl'):
            self.stack.setCurrentWidget(self.vp3d)
            self.scene.mode = 'bowl' if kind == 'bowl' else 'scene'
            if kind in ('side', 'front', 'rear'):
                self.vp3d.set_view(kind)
            elif kind == 'persp':
                if self.vp3d.cam.ortho:
                    self.vp3d.set_view('persp')
            self.vp3d.update()
        elif kind in ('top', 'bev', 'targets'):
            self.stack.setCurrentWidget(self.plan)
            self.plan.mode = {'top': self._plan_mode(), 'bev': 'bev', 'targets': 'targets'}[kind]
            self.plan.fit()
        elif kind == 'through':
            self.stack.setCurrentWidget(self.cam)
            self.cam.schedule()
        self._labels()

    def _plan_mode(self):
        h = self.ctx.heatmap
        return h if h != 'none' else 'coverage'

    def _labels(self):
        res = self.ctx.analysis.result
        if self.kind == 'top':
            self.plan.mode = self._plan_mode()
            cell = f' · {res.cell * 100:.0f} cm cells' if res and self.plan.mode == 'density' else ''
            self.plan.label = f'Top · {PLAN_LABELS.get(self.plan.mode, "")}{cell}'
        elif self.kind == 'bev':
            n = len(res.cams) if res else 0
            self.plan.label = f'Bird’s-eye · stitched from {n} cameras · drag a handle to move a seam'
        elif self.kind == 'targets':
            self.plan.label = 'Top · floor plan · redundancy'
        elif self.kind == 'through':
            c = self.ctx.store.selected
            if c:
                p = self.ctx.store.project
                lens, m = p.lens_for(c), p.module_for(c)
                from ..core.lens import PROJECTION_LABEL
                self.cam.label = f'Through {c.tag} · {PROJECTION_LABEL[lens.projection].lower()} · {m.width_px} × {m.height_px}'
        self.update()

    def current_widget(self):
        return self.stack.currentWidget()

    def paintEvent(self, e):
        pass


class ViewArea(QWidget):
    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        self.layout_ = QVBoxLayout(self)
        self.layout_.setContentsMargins(0, 0, 0, 0)
        self.panes = [Pane(ctx, 'persp'), Pane(ctx, 'top'), Pane(ctx, 'through'), Pane(ctx, 'bev')]
        for p in self.panes:
            p.activated.connect(self.set_active)
        self.mode = 'single'
        self.active = self.panes[0]
        self.panes[0].active = True
        self.split = None
        self.set_mode('single')

    def set_active(self, pane):
        for p in self.panes:
            p.active = p is pane
        self.active = pane
        if self.ctx.window:
            self.ctx.window.sync_view_buttons()

    def set_mode(self, mode):
        self.mode = mode
        for p in self.panes:
            p.setParent(None)
        if self.split is not None:
            self.split.setParent(None)
            self.split.deleteLater()
        if mode == 'single':
            self.split = QSplitter(Qt.Orientation.Horizontal)
            self.split.addWidget(self.active if self.active in self.panes else self.panes[0])
        elif mode == 'side':
            self.split = QSplitter(Qt.Orientation.Horizontal)
            self.split.addWidget(self.panes[0])
            self.split.addWidget(self.panes[1])
            self.split.setSizes([1, 1])
        else:
            self.split = QSplitter(Qt.Orientation.Vertical)
            top = QSplitter(Qt.Orientation.Horizontal)
            bot = QSplitter(Qt.Orientation.Horizontal)
            top.addWidget(self.panes[0])
            top.addWidget(self.panes[1])
            bot.addWidget(self.panes[2])
            bot.addWidget(self.panes[3])
            top.setSizes([1, 1])
            bot.setSizes([1, 1])
            self.split.addWidget(top)
            self.split.addWidget(bot)
            self.split.setSizes([1, 1])
        self.split.setHandleWidth(2)
        self.layout_.addWidget(self.split)
        for p in self.panes:
            p.show() if p.parent() is not None else None
        if mode != 'single' and self.active not in self.panes[:2 if mode == 'side' else 4]:
            self.set_active(self.panes[0])

    def visible_panes(self):
        return [p for p in self.panes if p.isVisible() and p.parent() is not None]

    def frame_all(self):
        for p in self.visible_panes():
            p.vp3d.frame_box(*self._bbox(), margin=0.82)
            p.plan.fit()

    def _bbox(self):
        import numpy as np
        res = self.ctx.analysis.result
        m = self.ctx.store.project.vehicle.mesh
        if m is not None:
            lo, hi = m.bbox
            return lo - np.array([1.4, 1.4, 0]), hi + np.array([1.4, 1.4, 0])
        return np.array([-3, -3, 0.0]), np.array([5, 3, 1.5])
