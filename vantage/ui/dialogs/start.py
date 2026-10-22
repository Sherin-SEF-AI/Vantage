"""C1 · Start window: recent projects, new, open, templates; drop a project or model anywhere."""
from __future__ import annotations

import os
import time

from PyQt6.QtCore import Qt, QRectF, QPointF, QSize
from PyQt6.QtGui import QPainter, QPixmap, QFont, QPen, QColor
from PyQt6.QtWidgets import (QDialog, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem,
                             QStyledItemDelegate, QStyle, QPushButton, QFileDialog, QFrame)

from ...core.project import read_thumbnail
from ...core.loaders import SUPPORTED_EXTS
from .. import icons
from ..theme import THEME, sans, mono
from ..widgets import label, hline, vline, fix_height

TEMPLATES = [('tigor', 'Tata Tigor EV', 'measured from 360° photos · 3.993 × 1.677 × 1.532 m', '6-cam', 'sedan'),
             ('sedan', 'Sedan', '4.0 × 1.7 × 1.5 m placeholder body', '6-cam', 'sedan'),
             ('hatchback', 'Hatchback', '3.8 × 1.7 × 1.5 m', '4-cam', 'hatchback'),
             ('suv', 'SUV', '4.4 × 1.8 × 1.7 m', '6-cam', 'suv'),
             ('van', 'Van', '5.3 × 2.0 × 2.3 m', '8-cam', 'van'),
             ('truck', 'Truck + trailer', 'articulated, two bodies', '8-cam', 'truck')]


class RecentDelegate(QStyledItemDelegate):
    def sizeHint(self, opt, idx):
        return QSize(opt.rect.width(), 92)

    def paint(self, p, opt, idx):
        d = idx.data(Qt.ItemDataRole.UserRole)
        r = opt.rect
        sel = opt.state & QStyle.StateFlag.State_Selected
        hov = opt.state & QStyle.StateFlag.State_MouseOver
        if sel or hov:
            p.fillRect(r, THEME.color('sel' if sel else 'raised'))
        if sel:
            p.fillRect(QRectF(r.left(), r.top(), 2, r.height()), THEME.color('ink'))
        p.setPen(QPen(THEME.color('hair'), 1))
        p.drawLine(r.bottomLeft(), r.bottomRight())
        th = QRectF(r.left() + 24, r.top() + 14, 88, 64)
        p.fillRect(th, THEME.color('base'))
        pm = d.get('thumb')
        if pm is not None and not pm.isNull():
            p.drawPixmap(th.toRect(), pm)
        else:
            p.drawPixmap(int(th.center().x() - 8), int(th.center().y() - 8), icons.pixmap('car-top', THEME.tok['muted'], 16))
        p.setPen(QPen(THEME.color('strong'), 1))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRect(th)
        x = th.right() + 18
        p.setFont(sans(15, QFont.Weight.DemiBold))
        p.setPen(THEME.color('ink'))
        p.drawText(QPointF(x, r.top() + 32), d['name'])
        p.setFont(sans(12))
        p.setPen(THEME.color('ink2'))
        p.drawText(QPointF(x, r.top() + 54), d['meta'])
        p.setFont(mono(11))
        p.setPen(THEME.color('muted'))
        p.drawText(QPointF(x, r.top() + 74), d['path'])
        p.setFont(sans(12))
        p.drawText(QRectF(r.left(), r.top(), r.width() - 24, 56), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, d['when'])


class StartWindow(QDialog):
    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.choice = None
        self.setWindowTitle('Vantage')
        self.resize(1200, 760)
        self.setAcceptDrops(True)
        self.setObjectName('panel')
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        head = QWidget()
        head.setFixedHeight(46)
        h = QHBoxLayout(head)
        h.setContentsMargins(20, 0, 20, 0)
        ic = QLabel()
        ic.setPixmap(icons.pixmap('app', THEME.tok['ink'], 16))
        t = QLabel('Vantage')
        t.setFont(sans(15, QFont.Weight.DemiBold))
        st = label('sensor rig planner for data collection', muted=True)
        h.addWidget(ic)
        h.addWidget(t)
        h.addWidget(st)
        h.addStretch(1)
        pref = QPushButton('Preferences')
        pref.setProperty('role', 'ghost')
        pref.clicked.connect(self._prefs)
        h.addWidget(pref)
        lay.addWidget(head)
        lay.addWidget(hline())
        body = QHBoxLayout()
        body.setSpacing(0)
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 10, 0, 0)
        ll.setSpacing(6)
        lh = QHBoxLayout()
        lh.setContentsMargins(20, 0, 20, 0)
        lh.addWidget(label('Recent', bold=True))
        lh.addStretch(1)
        self.filter = QLineEdit()
        self.filter.setPlaceholderText('Filter')
        self.filter.setFixedWidth(200)
        self.filter.addAction(icons.icon('search', 12), QLineEdit.ActionPosition.LeadingPosition)
        self.filter.textChanged.connect(self._fill)
        lh.addWidget(self.filter)
        ll.addLayout(lh)
        self.list = QListWidget()
        self.list.setItemDelegate(RecentDelegate())
        self.list.setMouseTracking(True)
        self.list.itemActivated.connect(self._open_item)
        self.list.itemDoubleClicked.connect(self._open_item)
        ll.addWidget(self.list, 1)
        self.empty = label('No recent projects yet. Open the Tata Tigor EV sample from the templates on the right, or open a .vantage file.',
                           muted=True, wrap=True)
        self.empty.setContentsMargins(20, 10, 20, 10)
        ll.addWidget(self.empty)
        body.addWidget(left, 65)
        body.addWidget(vline())
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(20, 10, 20, 10)
        rl.setSpacing(6)
        rl.addWidget(label('Start', bold=True))
        b_new = self._big('plus', 'New blank project', 'Ctrl+N', primary=True)
        b_new.clicked.connect(lambda: self._done('new', None))
        b_open = self._big('folder', 'Open…', 'Ctrl+O')
        b_open.clicked.connect(self._open_file)
        rl.addWidget(b_new)
        rl.addWidget(b_open)
        rl.addSpacing(14)
        rl.addWidget(label('New from template', ink2=True))
        rl.addWidget(hline())
        for key, name, meta, cams, ic in TEMPLATES:
            row = QPushButton()
            row.setProperty('role', 'ghost')
            fix_height(row, 48)
            row.setCursor(Qt.CursorShape.PointingHandCursor)
            rlay = QHBoxLayout(row)
            rlay.setContentsMargins(0, 0, 0, 0)
            i = QLabel()
            i.setPixmap(icons.pixmap(ic, THEME.tok['ink2'], 16))
            i.setFixedWidth(26)
            col = QVBoxLayout()
            col.setSpacing(0)
            col.addWidget(QLabel(name))
            col.addWidget(label(meta, muted=True))
            c = label(cams, mono_=True, ink2=True)
            rlay.addWidget(i)
            rlay.addLayout(col, 1)
            rlay.addWidget(c)
            for w in (i, c):
                w.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
            row.clicked.connect(lambda _=False, k=key: self._done('template', k))
            rl.addWidget(row)
            rl.addWidget(hline())
        rl.addWidget(label('Templates bring a placeholder body, mount zones and a camera preset. Replace the body with '
                           'Vehicle ▸ Replace model.', muted=True, wrap=True))
        rl.addStretch(1)
        body.addWidget(right, 35)
        lay.addLayout(body, 1)
        lay.addWidget(hline())
        foot = QWidget()
        foot.setFixedHeight(36)
        fl = QHBoxLayout(foot)
        fl.setContentsMargins(20, 0, 20, 0)
        ic2 = QLabel()
        ic2.setPixmap(icons.pixmap('export', THEME.tok['muted'], 12))
        fl.addWidget(ic2)
        fl.addWidget(label('Drop a .vantage project or a vehicle model (.glb .gltf .obj .stl .fbx .ply .step) anywhere on this window.', muted=True))
        fl.addStretch(1)
        lay.addWidget(foot)
        from PyQt6.QtGui import QShortcut, QKeySequence
        QShortcut(QKeySequence('Ctrl+N'), self, lambda: self._done('new', None))
        QShortcut(QKeySequence('Ctrl+O'), self, self._open_file)
        self._fill()

    def _big(self, ic, text, sc, primary=False):
        b = QPushButton()
        fix_height(b, 40)
        if primary:
            b.setProperty('role', 'primary')
        l = QHBoxLayout(b)
        l.setContentsMargins(14, 0, 14, 0)
        i = QLabel()
        i.setPixmap(icons.pixmap(ic, THEME.tok['panel'] if primary else THEME.tok['ink'], 16))
        t = QLabel(text)
        t.setFont(sans(13))
        if primary:
            t.setStyleSheet(f'color: {THEME.tok["panel"]};')
        s = QLabel(sc)
        s.setFont(mono(12))
        s.setStyleSheet(f'color: {THEME.tok["panel"] if primary else THEME.tok["muted"]};')
        for w in (i, t, s):
            w.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        l.addWidget(i)
        l.addWidget(t, 1)
        l.addWidget(s)
        return b

    def _fill(self):
        self.list.clear()
        q = self.filter.text().lower()
        recent = [p for p in (self.settings.value('recent', [], type=list) or []) if os.path.exists(p)]
        for path in recent:
            name = os.path.splitext(os.path.basename(path))[0]
            if q and q not in (name + path).lower():
                continue
            png, info = read_thumbnail(path)
            pm = QPixmap()
            if png:
                pm.loadFromData(png)
            meta = f"{info.get('vehicle', '')} · {info.get('cameras', 0)} cameras" + (f" · {info['variants']} variants" if info.get('variants', 0) > 1 else '')
            mt = os.path.getmtime(path)
            when = time.strftime('today %H:%M', time.localtime(mt)) if time.time() - mt < 86400 and time.localtime(mt).tm_mday == time.localtime().tm_mday \
                else time.strftime('%d %b %Y', time.localtime(mt))
            it = QListWidgetItem()
            it.setData(Qt.ItemDataRole.UserRole, {'name': name, 'meta': meta, 'path': path, 'when': when, 'thumb': pm})
            self.list.addItem(it)
        self.empty.setVisible(self.list.count() == 0)

    def _open_item(self, it):
        self._done('open', it.data(Qt.ItemDataRole.UserRole)['path'])

    def _open_file(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Open project or vehicle model', os.path.expanduser('~'),
                                              'Projects and models (*.vantage ' + ' '.join('*' + e for e in SUPPORTED_EXTS) + ')')
        if path:
            self._done('open' if path.endswith('.vantage') else 'model', path)

    def _done(self, kind, val):
        self.choice = (kind, val)
        self.accept()

    def _prefs(self):
        from .misc import PreferencesDialog
        PreferencesDialog(None, self, settings=self.settings).exec()

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        for u in e.mimeData().urls():
            p = u.toLocalFile()
            if p.endswith('.vantage'):
                self._done('open', p)
                return
            if os.path.splitext(p)[1].lower() in SUPPORTED_EXTS:
                self._done('model', p)
                return
