"""Small custom widgets used across the app."""
from __future__ import annotations

import math

from PyQt6.QtCore import Qt, pyqtSignal, QSize, QRectF, QPointF, QEvent, QTimer
from PyQt6.QtGui import QPainter, QColor, QPen, QFont, QFontMetrics, QCursor, QBrush, QPainterPath
from PyQt6.QtWidgets import (QWidget, QLabel, QDoubleSpinBox, QAbstractSpinBox, QAbstractButton, QHBoxLayout,
                             QVBoxLayout, QPushButton, QButtonGroup, QToolButton, QSizePolicy, QFrame, QGridLayout,
                             QLineEdit, QComboBox, QApplication)

from ..core.mathutil import parse_number, fmt_num
from .theme import THEME, mono, sans, STATUS
from . import icons

LABEL_W = 96
ROW_H = 22


def fix_height(w, h):
    """Fixed height that survives the global QSS min-height rules."""
    w.setFixedHeight(h)
    cls = type(w).__name__ if type(w).__module__.startswith('PyQt6') else type(w).__mro__[1].__name__
    w.setStyleSheet((w.styleSheet() or '') + f' {cls} {{ min-height: {h}px; max-height: {h}px; }}')
    return w


def hline():
    f = QFrame()
    f.setFixedHeight(1)
    f.setProperty('hair', True)
    return f


def vline():
    f = QFrame()
    f.setFixedWidth(1)
    f.setProperty('hair', True)
    return f


def label(text='', muted=False, mono_=False, small=False, bold=False, ink2=False, wrap=False):
    w = QLabel(text)
    if muted:
        w.setProperty('muted', True)
    if mono_:
        w.setProperty('mono', True)
    if small:
        w.setProperty('small', True)
    if bold:
        w.setProperty('h', True)
    if ink2:
        w.setProperty('ink2', True)
    if wrap:
        w.setWordWrap(True)
    return w


def button(text, role=None, icon_name=None, tip=None):
    b = QPushButton(text)
    if role:
        b.setProperty('role', role)
    if icon_name:
        sev = icon_name if icon_name in STATUS else None
        b.setIcon(icons.status_icon(sev, 12) if sev else icons.icon(icon_name, 12))
    if tip:
        b.setToolTip(tip)
    b.setCursor(Qt.CursorShape.PointingHandCursor)
    return b


def tool_button(icon_name, tip='', checkable=False, size=22, icon_size=16, text=None):
    b = QToolButton()
    b.setIcon(icons.icon(icon_name, icon_size))
    b.setIconSize(QSize(icon_size, icon_size))
    b.setFixedSize(size, size)
    b.setCheckable(checkable)
    b.setAutoRaise(True)
    if tip:
        b.setToolTip(tip)
    if text:
        b.setText(text)
    return b


# ---------------------------------------------------------------------- scrub spin box
class _ScrubHandle(QLabel):
    def __init__(self, box, text):
        super().__init__(text, box)
        self.box = box
        self.setCursor(Qt.CursorShape.SizeHorCursor)
        self.setProperty('muted', True)
        self.setFont(mono(12))
        self._press = None

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton and self.box.isEnabled() and not self.box.isReadOnly():
            self._press = e.globalPosition().x()
            self._start = self.box.value()
            self._moved = False
            self.box.scrub_started.emit()
        e.accept()

    def mouseMoveEvent(self, e):
        if self._press is None:
            return
        dx = e.globalPosition().x() - self._press
        if abs(dx) < 1 and not self._moved:
            return
        self._moved = True
        mods = QApplication.keyboardModifiers()
        step = self.box.scrub_step
        if mods & Qt.KeyboardModifier.ShiftModifier:
            step *= 0.1
        elif mods & Qt.KeyboardModifier.ControlModifier:
            step *= 10
        v = self._start + round(dx) * step
        self.box.set_mixed(False)
        self.box.setValue(v)
        self.box.scrubbed.emit(self.box.value())

    def mouseReleaseEvent(self, e):
        if self._press is not None:
            self._press = None
            if self._moved:
                self.box.committed.emit(self.box.value())
            else:
                self.box.setFocus()
                self.box.selectAll()
            self.box.scrub_finished.emit()
        e.accept()


class ScrubSpinBox(QDoubleSpinBox):
    """Number field without arrows. Drag the axis letter to scrub (Shift fine, Ctrl coarse).
    Accepts expressions and units: '1.88+0.05', '950 mm', '65deg'."""
    committed = pyqtSignal(float)
    scrubbed = pyqtSignal(float)
    scrub_started = pyqtSignal()
    scrub_finished = pyqtSignal()

    def __init__(self, axis='', kind='length', decimals=3, unit='', step=None, parent=None, width=None):
        super().__init__(parent)
        self.kind = kind
        self.unit_label = unit
        self._mixed = False
        self._display_unit = 'm'
        self.setDecimals(decimals)
        self.setRange(-1e7, 1e7)
        self.setKeyboardTracking(False)
        self.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.setFont(mono(12))
        self.scrub_step = step if step is not None else (10 ** -decimals if kind != 'angle' else 0.1)
        self.setSingleStep(self.scrub_step if kind == 'angle' else max(self.scrub_step, 10 ** -decimals))
        self.setCorrectionMode(QAbstractSpinBox.CorrectionMode.CorrectToPreviousValue)
        self.handle = _ScrubHandle(self, axis) if axis else None
        self.unit_w = QLabel(unit, self) if unit else None
        if self.unit_w:
            self.unit_w.setProperty('muted', True)
            self.unit_w.setFont(mono(11))
            self.unit_w.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._last = None
        self._shown = None             # the text set_value / a commit last put in the field
        self._typed = False            # the user has edited the text since it was last set or committed
        self.lineEdit().textEdited.connect(lambda _: setattr(self, '_typed', True))
        self.editingFinished.connect(self._on_finished)
        self.setFixedHeight(ROW_H)
        if width:
            self.setFixedWidth(width)
        self._layout_children()

    def set_display_unit(self, u):
        self._display_unit = u

    def setDecimals(self, d):
        if d != self.decimals():       # Qt rewrites the text even for the same value (a refresh would clobber typing)
            super().setDecimals(d)

    def _layout_children(self):
        left = 6
        right = 6
        if self.handle:
            fm = QFontMetrics(self.handle.font())
            w = fm.horizontalAdvance(self.handle.text()) + 8
            self.handle.setGeometry(1, 1, w, self.height() - 2)
            self.handle.setAlignment(Qt.AlignmentFlag.AlignCenter)
            left = w
        if self.unit_w:
            fm = QFontMetrics(self.unit_w.font())
            w = fm.horizontalAdvance(self.unit_w.text()) + 6
            self.unit_w.setGeometry(self.width() - w - 2, 1, w, self.height() - 2)
            self.unit_w.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
            right = w + 2
        le = self.lineEdit()
        le.setTextMargins(max(0, left - 6), 0, max(0, right - 6), 0)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._layout_children()

    def set_unit(self, unit):
        if self.unit_w:
            self.unit_w.setText(unit)
            self._layout_children()

    def set_mixed(self, m: bool):
        if m != self._mixed:
            self._mixed = m
            self.setProperty('mixed', m)
            self.style().unpolish(self)
            self.style().polish(self)
            self.lineEdit().setText(self.textFromValue(self.value()))

    def set_value(self, v, mixed=False):
        if self._typed and self.hasFocus():
            return                     # a refresh must not overwrite what the user is typing
        self.blockSignals(True)
        self._mixed = mixed
        self.setProperty('mixed', mixed)
        if v is not None and not (isinstance(v, float) and math.isnan(v)):
            self.setValue(v)
        self.lineEdit().setText(self.textFromValue(self.value()))
        # the unrounded value: typing exactly what the field shows (rounded) is still a change
        self._last = float(v) if v is not None and math.isfinite(v) else self.value()
        self._shown = self.lineEdit().text()
        self._typed = False
        self.blockSignals(False)

    def textFromValue(self, v):
        if self._mixed:
            return '—'
        return fmt_num(v, self.decimals())

    def valueFromText(self, text):
        try:
            if self.kind == 'length':
                return parse_number(text, 'length', self._display_unit)
            if self.kind == 'angle':
                return parse_number(text, 'angle')
            return parse_number(text, 'plain')
        except Exception:
            return self.value()

    def validate(self, text, pos):
        from PyQt6.QtGui import QValidator
        return QValidator.State.Intermediate if text.strip() in ('', '-', '—', '−') else QValidator.State.Acceptable, text, pos

    def fixup(self, text):
        return self.textFromValue(self.value())

    def focusInEvent(self, e):
        super().focusInEvent(e)
        if self._mixed:
            self.lineEdit().setText('')

    def keyPressEvent(self, e):
        if e.key() == Qt.Key.Key_Escape:
            self._typed = False
            self.set_value(self._last if self._last is not None else self.value(), self._mixed)
            self.clearFocus()
            return
        if e.key() in (Qt.Key.Key_Up, Qt.Key.Key_Down):
            mods = e.modifiers()
            step = self.scrub_step * (10 if self.kind != 'angle' else 1)
            if mods & Qt.KeyboardModifier.ShiftModifier:
                step *= 0.1
            elif mods & Qt.KeyboardModifier.ControlModifier:
                step *= 10
            self.setValue(self.value() + (step if e.key() == Qt.Key.Key_Up else -step))
            self._mixed = False
            self._typed = False
            self.committed.emit(self.value())
            self._last = self.value()
            self._shown = self.lineEdit().text()
            return
        super().keyPressEvent(e)

    def wheelEvent(self, e):
        e.ignore()

    def _on_finished(self):
        text = self.lineEdit().text()
        if self._mixed and text.strip() in ('', '—'):
            self.lineEdit().setText('—')
            return
        # focus leaving an untouched field is not an edit (its text is the rounded value)
        edited = self._typed or text != self._shown
        self._typed = False
        v = self.valueFromText(text)
        self.blockSignals(True)
        self.setValue(v)
        self.blockSignals(False)
        if edited and (self._last is None or abs(v - self._last) > 10 ** -(self.decimals() + 2)) or self._mixed:
            self._mixed = False
            self.setProperty('mixed', False)
            self._last = v
            self.committed.emit(v)
        self.lineEdit().setText(self.textFromValue(self.value()))
        self._shown = self.lineEdit().text()


# ---------------------------------------------------------------------- toggle switch
class ToggleSwitch(QAbstractButton):
    def __init__(self, text='', parent=None):
        super().__init__(parent)
        self.setCheckable(True)
        self.setText(text)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def sizeHint(self):
        fm = self.fontMetrics()
        return QSize(26 + (8 + fm.horizontalAdvance(self.text()) if self.text() else 0), ROW_H)

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = QRectF(0.5, (self.height() - 14) / 2 + 0.5, 26, 13)
        on = self.isChecked()
        en = self.isEnabled()
        if on and en:
            p.setBrush(THEME.color('ink'))
            p.setPen(QPen(THEME.color('ink'), 1))
        else:
            p.setBrush(THEME.color('raised'))
            p.setPen(QPen(THEME.color('strong' if en else 'hair'), 1))
        p.drawRoundedRect(r, 2, 2)
        k = QRectF(r.right() - 11 if on else r.left() + 2.5, r.top() + 2.5, 8.5, 8)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(THEME.color('panel') if on and en else THEME.color('ink2' if en else 'hair'))
        p.drawRoundedRect(k, 1.5, 1.5)
        if self.hasFocus():
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.setPen(QPen(THEME.color('ink'), 1))
            p.drawRoundedRect(r.adjusted(-2, -2, 2, 2), 3, 3)
        if self.text():
            p.setPen(THEME.color('ink' if en else 'muted'))
            p.drawText(QRectF(34, 0, self.width() - 34, self.height()), Qt.AlignmentFlag.AlignVCenter, self.text())
        p.end()


# ---------------------------------------------------------------------- segmented control
class SegmentedControl(QWidget):
    changed = pyqtSignal(str)

    def __init__(self, options, value=None, parent=None, min_w=0):
        """options: list of (key, label)."""
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        self.group = QButtonGroup(self)
        self.group.setExclusive(True)
        self.buttons = {}
        n = len(options)
        for i, (key, text) in enumerate(options):
            b = QPushButton(text)
            b.setCheckable(True)
            b.setProperty('seg', True)
            b.setProperty('segpos', 'only' if n == 1 else 'first' if i == 0 else 'last' if i == n - 1 else 'mid')
            b.setFixedHeight(ROW_H)
            if min_w:
                b.setMinimumWidth(min_w)
            b.setFocusPolicy(Qt.FocusPolicy.TabFocus)
            self.group.addButton(b)
            self.buttons[key] = b
            lay.addWidget(b)
            b.clicked.connect(lambda _=False, k=key: self.changed.emit(k))
        if value is not None:
            self.set_value(value)

    def set_value(self, key):
        b = self.buttons.get(key)
        if b:
            b.blockSignals(True)
            b.setChecked(True)
            b.blockSignals(False)

    def value(self):
        for k, b in self.buttons.items():
            if b.isChecked():
                return k
        return None


# ---------------------------------------------------------------------- chip
class Chip(QWidget):
    """8 x 8 identity square + tag in ink (mono semibold)."""

    def __init__(self, slot=0, tag='', parent=None, size=8):
        super().__init__(parent)
        self.slot, self.tag, self.sz = slot, tag, size
        self.setFixedHeight(ROW_H)
        self._update_width()

    def _update_width(self):
        fm = QFontMetrics(mono(11, QFont.Weight.DemiBold))
        self.setFixedWidth(self.sz + 6 + fm.horizontalAdvance(self.tag) + 2)

    def set(self, slot, tag):
        self.slot, self.tag = slot, tag
        self._update_width()
        self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        y = (self.height() - self.sz) / 2
        p.fillRect(QRectF(0, y, self.sz, self.sz), THEME.cam(self.slot))
        p.setFont(mono(11, QFont.Weight.DemiBold))
        p.setPen(THEME.color('ink'))
        p.drawText(QRectF(self.sz + 5, 0, self.width(), self.height()), Qt.AlignmentFlag.AlignVCenter, self.tag)
        p.end()


def paint_chip(p: QPainter, x, y_mid, slot, tag, size=8, font=None, color=None):
    p.fillRect(QRectF(x, y_mid - size / 2, size, size), THEME.cam(slot))
    p.setFont(font or mono(11, QFont.Weight.DemiBold))
    p.setPen(color or THEME.color('ink'))
    fm = p.fontMetrics()
    p.drawText(QPointF(x + size + 5, y_mid + fm.ascent() / 2 - 1), tag)
    return x + size + 5 + fm.horizontalAdvance(tag)


# ---------------------------------------------------------------------- sections
class SectionHeader(QWidget):
    toggled = pyqtSignal(bool)

    def __init__(self, title, expanded=True, parent=None):
        super().__init__(parent)
        self.setFixedHeight(ROW_H)
        self.expanded = expanded
        lay = QHBoxLayout(self)
        lay.setContentsMargins(6, 0, 6, 0)
        lay.setSpacing(6)
        self.chev = QLabel()
        self.chev.setFixedWidth(12)
        self.title = QLabel(title)
        self.title.setProperty('h', True)
        self.summary = QLabel('')
        self.summary.setProperty('muted', True)
        self.summary.setFont(mono(11))
        self.summary.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.summary.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.right = QHBoxLayout()
        self.right.setSpacing(2)
        lay.addWidget(self.chev)
        lay.addWidget(self.title)
        lay.addWidget(self.summary, 1)
        lay.addLayout(self.right)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._sync()
        THEME.changed.connect(self._theme_changed)      # a bound slot: disconnected when the widget goes

    def _theme_changed(self, *_):
        self._sync()

    def _sync(self):
        self.chev.setPixmap(icons.pixmap('chev-d' if self.expanded else 'chev-r', THEME.tok['ink2'], 12))
        self.summary.setVisible(not self.expanded)

    def set_summary(self, text):
        self.summary.setText(text)
        self.summary.setToolTip(text)

    def add_right(self, w):
        self.right.addWidget(w)

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self.set_expanded(not self.expanded)
            self.toggled.emit(self.expanded)

    def set_expanded(self, ex):
        self.expanded = ex
        self._sync()

    def paintEvent(self, e):
        if self.underMouse():
            p = QPainter(self)
            p.fillRect(self.rect(), THEME.color('raised'))
            p.end()

    def enterEvent(self, e):
        self.update()

    def leaveEvent(self, e):
        self.update()


class Section(QWidget):
    def __init__(self, title, expanded=True, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        self.header = SectionHeader(title, expanded)
        self.body = QWidget()
        self.blay = QVBoxLayout(self.body)
        self.blay.setContentsMargins(8, 2, 8, 6)
        self.blay.setSpacing(2)
        lay.addWidget(self.header)
        lay.addWidget(self.body)
        self.body.setVisible(expanded)
        self.header.toggled.connect(self.body.setVisible)

    def row(self, text, *widgets, stretch_last=True, label_w=LABEL_W, tip=None):
        r = QWidget()
        r.setFixedHeight(ROW_H)
        r.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        h = QHBoxLayout(r)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(2)
        lab = QLabel(text)
        lab.setProperty('ink2', True)
        lab.setFixedWidth(label_w)
        if tip:
            lab.setToolTip(tip)
        h.addWidget(lab)
        for i, w in enumerate(widgets):
            if isinstance(w, str):
                w = label(w, muted=True, mono_=True, small=True)
            if isinstance(w, (ScrubSpinBox, QComboBox, QLineEdit)):
                w.setMinimumWidth(30)
                w.setSizePolicy(QSizePolicy.Policy.Ignored, w.sizePolicy().verticalPolicy())
            h.addWidget(w, 1 if (stretch_last and i == len(widgets) - 1) or isinstance(w, (ScrubSpinBox, QComboBox, QLineEdit)) else 0)
        self.blay.addWidget(r)
        r.label = lab
        return r

    def add(self, w):
        self.blay.addWidget(w)
        return w

    def set_expanded(self, ex):
        self.header.set_expanded(ex)
        self.body.setVisible(ex)


class ValueLabel(QWidget):
    """Mono value + muted unit + optional status icon + muted note, as in the Inspector readouts."""

    def __init__(self, parent=None, align_right=False):
        super().__init__(parent)
        h = QHBoxLayout(self)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(5)
        self.v = QLabel('')
        self.v.setFont(mono(12))
        self.u = QLabel('')
        self.u.setProperty('muted', True)
        self.u.setFont(mono(11))
        self.i = QLabel()
        self.n = QLabel('')
        self.n.setProperty('ink2', True)
        if align_right:
            h.addStretch(1)
        h.addWidget(self.v)
        h.addWidget(self.u)
        h.addWidget(self.i)
        h.addWidget(self.n)
        if not align_right:
            h.addStretch(1)
        self.setFixedHeight(ROW_H)

    def set(self, value, unit='', status=None, note=''):
        self.v.setText(value)
        self.u.setText(unit)
        self.u.setVisible(bool(unit))
        if status:
            self.i.setPixmap(icons.pixmap(status, STATUS[status], 12))
            self.i.show()
        else:
            self.i.clear()
            self.i.hide()
        self.n.setText(note)
        self.n.setVisible(bool(note))


class StatusLine(QWidget):
    def __init__(self, sev='warning', text='', parent=None, wrap=True):
        super().__init__(parent)
        h = QHBoxLayout(self)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(6)
        self.icon = QLabel()
        self.icon.setFixedWidth(14)
        self.icon.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.text = QLabel(text)
        self.text.setWordWrap(wrap)
        h.addWidget(self.icon, 0, Qt.AlignmentFlag.AlignTop)
        h.addWidget(self.text, 1)
        self.set(sev, text)

    def set(self, sev, text):
        self.icon.setPixmap(icons.pixmap(sev, STATUS.get(sev, '#888'), 12) if sev else icons.pixmap('good', '#00000000', 12))
        self.text.setText(text)


class DockTitle(QWidget):
    """Dock title bar: label left, muted context right, float/close on hover."""

    def __init__(self, dock, title, right=''):
        super().__init__(dock)
        self.dock = dock
        self.setFixedHeight(ROW_H)
        h = QHBoxLayout(self)
        h.setContentsMargins(8, 0, 4, 0)
        h.setSpacing(4)
        self.title = QLabel(title)
        self.right = QLabel(right)
        self.right.setProperty('muted', True)
        self.extra = QHBoxLayout()
        self.extra.setSpacing(2)
        self.b_float = tool_button('float', 'Float', size=18, icon_size=12)
        self.b_close = tool_button('close', 'Close', size=18, icon_size=12)
        self.b_float.clicked.connect(lambda: dock.setFloating(not dock.isFloating()))
        self.b_close.clicked.connect(dock.close)
        h.addWidget(self.title)
        h.addStretch(1)
        h.addWidget(self.right)
        h.addLayout(self.extra)
        h.addWidget(self.b_float)
        h.addWidget(self.b_close)
        self._hover(False)

    def _hover(self, on):
        self.b_float.setVisible(on)
        self.b_close.setVisible(on)

    def enterEvent(self, e):
        self._hover(True)

    def leaveEvent(self, e):
        self._hover(False)

    def paintEvent(self, e):
        p = QPainter(self)
        p.fillRect(self.rect(), THEME.color('panel'))
        p.end()


class ImageCircleDiagram(QWidget):
    """Image circle over the sensor rectangle (inspector + module editor)."""

    def __init__(self, parent=None, detailed=False):
        super().__init__(parent)
        self.detailed = detailed
        self.data = None
        self.setMinimumSize(64, 64)

    def set(self, sensor_w_mm, sensor_h_mm, circle_r_mm, max_half_deg=0, width_px=0, height_px=0):
        self.data = (sensor_w_mm, sensor_h_mm, circle_r_mm, max_half_deg, width_px, height_px)
        self.update()

    def paintEvent(self, e):
        if not self.data:
            return
        sw, sh, cr, mh, wp, hp = self.data
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        if self.detailed:
            p.fillRect(self.rect(), THEME.color('raised'))
        W, H = self.width(), self.height()
        span = max(2 * cr, math.hypot(sw, sh)) * 1.08
        s = min(W, H) / span
        cx, cy = W / 2, H / 2
        pen = QPen(THEME.color('ink2'), 1, Qt.PenStyle.DashLine)
        pen.setDashPattern([3, 3])
        p.setPen(pen)
        p.drawEllipse(QPointF(cx, cy), cr * s, cr * s)
        p.setPen(QPen(THEME.color('ink2'), 1))
        p.drawRect(QRectF(cx - sw / 2 * s, cy - sh / 2 * s, sw * s, sh * s))
        p.drawLine(QPointF(cx - 4, cy), QPointF(cx + 4, cy))
        p.drawLine(QPointF(cx, cy - 4), QPointF(cx, cy + 4))
        if self.detailed:
            p.setFont(mono(11))
            p.setPen(THEME.color('ink2'))
            p.drawLine(QPointF(cx, cy), QPointF(cx + sw / 2 * s, cy - sh / 2 * s))
            hd = math.hypot(sw, sh) / 2
            p.drawText(QPointF(cx + sw / 8 * s, cy - sh / 8 * s - 4), f'{hd:.3f}')
            p.drawText(QRectF(cx - sw / 2 * s, cy + sh / 2 * s + 2, sw * s, 16), Qt.AlignmentFlag.AlignCenter,
                       f'{sw:.3f} mm · {wp} px')
            p.save()
            p.translate(cx - sw / 2 * s - 4, cy)
            p.rotate(-90)
            p.drawText(QRectF(-40, -14, 80, 14), Qt.AlignmentFlag.AlignCenter, f'{sh:.3f}')
            p.restore()
            p.drawText(QRectF(cx, cy - cr * s - 2, cr * s, 16), Qt.AlignmentFlag.AlignLeft,
                       f'θ {mh:.0f}° · r {cr:.2f} mm')
        p.end()
