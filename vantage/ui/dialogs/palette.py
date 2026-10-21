"""C2 · Command palette: fuzzy search over every command and object."""
from __future__ import annotations

from PyQt6.QtCore import Qt, QRectF, QPointF, QSize
from PyQt6.QtGui import QFont, QPen, QKeySequence
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLineEdit, QListWidget, QListWidgetItem,
                             QStyledItemDelegate, QStyle, QLabel, QWidget)

from .. import icons
from ..theme import THEME, sans, mono
from ..widgets import label, hline, paint_chip

RECENT = []


def fuzzy(q, text):
    """Score of q as an in-order subsequence of text; 0 = no match. Prefix and word starts score higher."""
    if not q:
        return 1.0
    t = text.lower()
    q = q.lower()
    if t.startswith(q):
        return 100 + 1 / len(t)
    i = t.find(q)
    if i >= 0:
        return 60 + (10 if i == 0 or not t[i - 1].isalnum() else 0) + 1 / len(t)
    pos = 0
    score = 0
    for ch in q:
        j = t.find(ch, pos)
        if j < 0:
            return 0
        score += 3 if j == 0 or not t[j - 1].isalnum() else 1
        pos = j + 1
    return score / len(q)


class PaletteDelegate(QStyledItemDelegate):
    def __init__(self, palette):
        super().__init__()
        self.palette = palette

    def sizeHint(self, opt, idx):
        d = idx.data(Qt.ItemDataRole.UserRole)
        return QSize(opt.rect.width(), 24 if d and d.get('header') else 27)

    def paint(self, p, opt, idx):
        d = idx.data(Qt.ItemDataRole.UserRole)
        r = opt.rect
        if d.get('header'):
            p.setFont(sans(11))
            p.setPen(THEME.color('muted'))
            p.drawText(QRectF(r.left() + 12, r.top(), r.width(), r.height()), Qt.AlignmentFlag.AlignBottom, d['header'])
            return
        sel = opt.state & QStyle.StateFlag.State_Selected
        if sel:
            p.fillRect(r, THEME.color('sel'))
            p.fillRect(QRectF(r.left(), r.top(), 2, r.height()), THEME.color('ink'))
        x = r.left() + 22
        cy = r.center().y()
        if d.get('chip'):
            x = paint_chip(p, x, cy + 0.5, *d['chip']) + 10
        elif d.get('icon'):
            p.drawPixmap(int(x), int(cy - 6), icons.pixmap(d['icon'], THEME.tok['ink2'], 12))
            x += 20
        else:
            x += 20
        q = self.palette.query().lower()
        text = d['text']
        p.setFont(sans(12))
        fm = p.fontMetrics()
        # bold the matched prefix
        if q and text.lower().startswith(q):
            p.setFont(sans(12, QFont.Weight.DemiBold))
            p.setPen(THEME.color('ink'))
            p.drawText(QPointF(x, cy + 5), text[:len(q)])
            w = p.fontMetrics().horizontalAdvance(text[:len(q)])
            p.setFont(sans(12))
            p.drawText(QPointF(x + w, cy + 5), text[len(q):])
        else:
            p.setPen(THEME.color('ink'))
            p.drawText(QPointF(x, cy + 5), text)
        if d.get('detail'):
            p.setFont(mono(11))
            p.setPen(THEME.color('ink2'))
            p.drawText(QPointF(x + fm.horizontalAdvance(text) + 8, cy + 5), d['detail'])
        p.setFont(sans(11))
        p.setPen(THEME.color('muted'))
        p.drawText(QRectF(r.left(), r.top(), r.width() - 120, r.height()), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, d.get('cat', ''))
        if d.get('shortcut'):
            p.setFont(mono(11))
            p.drawText(QRectF(r.left(), r.top(), r.width() - 14, r.height()), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, d['shortcut'])


class CommandPalette(QDialog):
    def __init__(self, ctx, window, prefix=''):
        super().__init__(window)
        self.ctx = ctx
        self.win = window
        self.setWindowFlags(Qt.WindowType.Popup | Qt.WindowType.FramelessWindowHint)
        self.setFixedWidth(660)
        self.setObjectName('card')
        lay = QVBoxLayout(self)
        lay.setContentsMargins(1, 1, 1, 1)
        lay.setSpacing(0)
        top = QWidget()
        h = QHBoxLayout(top)
        h.setContentsMargins(12, 6, 12, 6)
        ic = QLabel()
        ic.setPixmap(icons.pixmap('search', THEME.tok['ink2'], 16))
        self.edit = QLineEdit()
        self.edit.setStyleSheet('QLineEdit { border: none; background: transparent; font-size: 13px; }')
        self.edit.setPlaceholderText('Type a command or an object name')
        hint = label('↑ ↓ move · Enter run · Tab scope', muted=True, small=True)
        h.addWidget(ic)
        h.addWidget(self.edit, 1)
        h.addWidget(hint)
        lay.addWidget(top)
        lay.addWidget(hline())
        self.list = QListWidget()
        self.list.setItemDelegate(PaletteDelegate(self))
        self.list.setFixedHeight(380)
        self.list.itemActivated.connect(self.run)
        self.list.itemClicked.connect(self.run)
        lay.addWidget(self.list)
        self.scope = 'all'
        self.edit.textChanged.connect(self.fill)
        self.edit.installEventFilter(self)
        self.items = self._collect()
        self.edit.setText(prefix.replace('obj ', ''))
        if prefix.startswith('obj'):
            self.scope = 'objects'
        self.fill()
        g = window.geometry()
        self.move(g.center().x() - self.width() // 2, g.top() + 90)

    def query(self):
        return self.edit.text().strip()

    def _collect(self):
        out = []
        for aid, a in self.win.actions_by_id.items():
            d = a.data() or {}
            if not a.text():
                continue
            out.append({'kind': 'cmd', 'text': a.text().replace('&', ''), 'cat': d.get('category', ''), 'icon': d.get('icon'),
                        'shortcut': a.shortcut().toString(QKeySequence.SequenceFormat.NativeText), 'action': a})
        st = self.ctx.store
        p = st.project
        o = p.origin_offset()
        for c in p.variant.cameras:
            out.append({'kind': 'obj', 'text': f'{c.tag} · {c.name}', 'cat': 'Cameras', 'chip': (c.slot, c.tag),
                        'detail': f'{c.x - o[0]:.3f}, {c.y - o[1]:.3f}', 'run': lambda c=c: (st.select_cams([c.uid]), self.win.frame_selection())})
        for v in p.variants:
            out.append({'kind': 'obj', 'text': f'{v.name}', 'cat': 'Layout ▸ Variants', 'icon': 'camera-wide',
                        'run': lambda v=v: self.win.switch_variant(v.uid)})
        for t in p.targets:
            out.append({'kind': 'obj', 'text': f'{t.name} · target', 'cat': 'Calibration targets', 'icon': 'checker',
                        'detail': f'{t.x - o[0]:.3f}, {t.y - o[1]:.3f}', 'run': lambda t=t: (self.win.set_page('targets'), st.select_obj(('target', t.uid)))})
        for ob in p.obstacles:
            out.append({'kind': 'obj', 'text': f'{ob.name}', 'cat': 'Obstacles', 'icon': 'pole', 'run': lambda ob=ob: st.select_obj(('obstacle', ob.uid))})
        if p.photos.frames:
            for k in range(len(p.photos.frames)):
                out.append({'kind': 'obj', 'text': f'Reference photos · frame {k:02d}', 'cat': 'Vehicle', 'icon': 'photo-spin',
                            'run': lambda k=k: (self.win.set_page('photo'), self.win.page('photo').set_frame(k))})
        out.append({'kind': 'obj', 'text': 'Ground plane', 'cat': 'Vehicle', 'icon': 'ground', 'run': lambda: self.win.import_vehicle(step=2, reuse=True)})
        out.append({'kind': 'obj', 'text': 'ROI', 'cat': 'Analysis', 'icon': 'roi', 'run': self.win.focus_roi})
        if p.vehicle.mesh is not None:
            out.append({'kind': 'obj', 'text': f'{p.vehicle.name} · body mesh', 'cat': 'Vehicle', 'icon': 'mesh', 'run': lambda: st.select_obj(('vehicle',))})
        return out

    def fill(self):
        self.list.clear()
        q = self.query()
        groups = {'cmd': [], 'obj': []}
        for it in self.items:
            if self.scope == 'objects' and it['kind'] != 'obj':
                continue
            if self.scope == 'commands' and it['kind'] != 'cmd':
                continue
            if it['kind'] == 'cmd' and not it['action'].isEnabled():
                continue
            s = max(fuzzy(q, it['text']), 0.8 * fuzzy(q, it.get('cat', '') + ' ' + it['text']) if q else 0)
            if s > 0:
                groups[it['kind']].append((s, it))
        rec = [it for it in RECENT if it in self.items] if not q else []
        first = None
        for title, key in (('Commands', 'cmd'), ('Objects', 'obj')):
            rows = sorted(groups[key], key=lambda x: -x[0])[:12 if q else 8]
            if not rows:
                continue
            self._header(title)
            for _, it in rows:
                li = self._add(it)
                first = first or li
        if rec:
            self._header('Recent')
            for it in rec[:4]:
                self._add(it)
        if first:
            self.list.setCurrentItem(first)

    def _header(self, t):
        li = QListWidgetItem()
        li.setData(Qt.ItemDataRole.UserRole, {'header': t})
        li.setFlags(Qt.ItemFlag.NoItemFlags)
        self.list.addItem(li)

    def _add(self, it):
        li = QListWidgetItem()
        li.setData(Qt.ItemDataRole.UserRole, it)
        self.list.addItem(li)
        return li

    def eventFilter(self, obj, e):
        from PyQt6.QtCore import QEvent
        if obj is self.edit and e.type() == QEvent.Type.KeyPress:
            k = e.key()
            if k in (Qt.Key.Key_Down, Qt.Key.Key_Up):
                n = self.list.count()
                i = self.list.currentRow()
                step = 1 if k == Qt.Key.Key_Down else -1
                for _ in range(n):
                    i = (i + step) % max(n, 1)
                    d = self.list.item(i).data(Qt.ItemDataRole.UserRole)
                    if not d.get('header'):
                        break
                self.list.setCurrentRow(i)
                return True
            if k in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                self.run(self.list.currentItem())
                return True
            if k == Qt.Key.Key_Tab:
                self.scope = {'all': 'commands', 'commands': 'objects', 'objects': 'all'}[self.scope]
                self.edit.setPlaceholderText({'all': 'Type a command or an object name', 'commands': 'Commands only',
                                              'objects': 'Objects only'}[self.scope])
                self.fill()
                return True
            if k == Qt.Key.Key_Escape:
                self.reject()
                return True
        return False

    def run(self, li):
        if li is None:
            return
        d = li.data(Qt.ItemDataRole.UserRole)
        if not d or d.get('header'):
            return
        self.accept()
        if d in RECENT:
            RECENT.remove(d)
        RECENT.insert(0, d)
        del RECENT[8:]
        if d['kind'] == 'cmd':
            a = d['action']
            if a.isCheckable():
                a.trigger()
            else:
                a.trigger()
        else:
            d['run']()
