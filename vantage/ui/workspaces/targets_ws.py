"""B6 · Calibration target planner: floor plan with redundancy and boards; settings live in the Inspector."""
from __future__ import annotations

from PyQt6.QtWidgets import QWidget, QVBoxLayout

from ..planview import PlanView


class TargetsPage(QWidget):
    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.plan = PlanView(ctx, 'targets')
        self.plan.label = 'Top · floor plan · redundancy'
        lay.addWidget(self.plan)

    def activate(self):
        self.plan.fit()
        self.ctx.store.select_obj(('targets',))
        insp = self.ctx.window.inspector
        if not self.ctx.analysis.targets and self.ctx.store.project.targets:
            pass
