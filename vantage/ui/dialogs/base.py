"""Frameless dialog with the design's header row (title, muted subtitle, close) and footer (hint + buttons)."""
from __future__ import annotations

from PyQt6.QtCore import Qt, QPoint
from PyQt6.QtGui import QPainter, QPen
from PyQt6.QtWidgets import QDialog, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSizeGrip

from ..theme import THEME, sans
from ..widgets import label, tool_button, hline


class VDialog(QDialog):
    def __init__(self, parent=None, title='', subtitle='', size=(880, 600), resizable=True):
        super().__init__(parent)
        self.setWindowFlags(Qt.WindowType.Dialog | Qt.WindowType.FramelessWindowHint)
        self.setWindowTitle(title)
        self.resize(*size)
        self.setObjectName('vdialog')
        outer = QVBoxLayout(self)
        outer.setContentsMargins(1, 1, 1, 1)
        outer.setSpacing(0)
        head = QWidget()
        head.setFixedHeight(36)
        head.setObjectName('panel')
        h = QHBoxLayout(head)
        h.setContentsMargins(14, 0, 8, 0)
        h.setSpacing(8)
        self.title_lbl = QLabel(title)
        self.title_lbl.setProperty('title', True)
        self.sub_lbl = label(subtitle, muted=True)
        h.addWidget(self.title_lbl)
        h.addWidget(self.sub_lbl)
        h.addStretch(1)
        self.head_right = QHBoxLayout()
        h.addLayout(self.head_right)
        x = tool_button('close', 'Close  Esc', size=22, icon_size=12)
        x.clicked.connect(self.reject)
        h.addWidget(x)
        outer.addWidget(head)
        outer.addWidget(hline())
        self.body = QWidget()
        self.body.setObjectName('panel')
        outer.addWidget(self.body, 1)
        outer.addWidget(hline())
        foot = QWidget()
        foot.setObjectName('panel')
        foot.setFixedHeight(44)
        f = QHBoxLayout(foot)
        f.setContentsMargins(14, 0, 14, 0)
        f.setSpacing(6)
        self.hint = label('', muted=True, small=True)
        f.addWidget(self.hint, 1)
        self.buttons = QHBoxLayout()
        self.buttons.setSpacing(6)
        f.addLayout(self.buttons)
        if resizable:
            grip = QSizeGrip(foot)
            grip.setFixedSize(10, 10)
            f.addWidget(grip, 0, Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignRight)
        outer.addWidget(foot)
        self._drag = None
        head.mousePressEvent = self._press
        head.mouseMoveEvent = self._move
        head.mouseReleaseEvent = lambda e: setattr(self, '_drag', None)

    def set_subtitle(self, s):
        self.sub_lbl.setText(s)

    def _press(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self._drag = e.globalPosition().toPoint() - self.frameGeometry().topLeft()

    def _move(self, e):
        if self._drag is not None:
            self.move(e.globalPosition().toPoint() - self._drag)

    def paintEvent(self, e):
        p = QPainter(self)
        p.fillRect(self.rect(), THEME.color('panel'))
        p.setPen(QPen(THEME.color('strong'), 1))
        p.drawRect(self.rect().adjusted(0, 0, -1, -1))
        p.end()

    def showEvent(self, e):
        super().showEvent(e)
        par = self.parentWidget()
        if par is not None:
            g = par.window().geometry()
            self.move(g.center() - self.rect().center())
