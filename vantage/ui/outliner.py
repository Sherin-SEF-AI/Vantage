"""Outliner (scene tree) and Library docks."""
from __future__ import annotations

from PyQt6.QtCore import Qt, QSize, QRect, QRectF, QPointF, QMimeData, pyqtSignal, QModelIndex, QTimer
from PyQt6.QtGui import QPainter, QColor, QFont, QPen, QDrag, QFontMetrics
from PyQt6.QtWidgets import (QTreeWidget, QTreeWidgetItem, QStyledItemDelegate, QStyle, QWidget, QVBoxLayout,
                             QHBoxLayout, QLineEdit, QComboBox, QLabel, QAbstractItemView, QMenu, QStyleOptionViewItem)

from ..core.rules import route_length
from . import icons, ops
from .theme import THEME, mono, sans, STATUS
from .widgets import paint_chip, tool_button, label

ROLE_KIND = Qt.ItemDataRole.UserRole + 1
ROLE_ID = Qt.ItemDataRole.UserRole + 2
ROLE_META = Qt.ItemDataRole.UserRole + 3
ROLE_ICON = Qt.ItemDataRole.UserRole + 4
ROLE_SEV = Qt.ItemDataRole.UserRole + 5
ROLE_FLAGS = Qt.ItemDataRole.UserRole + 6
GUTTER = 24


class OutlinerDelegate(QStyledItemDelegate):
    def __init__(self, tree):
        super().__init__(tree)
        self.tree = tree

    def sizeHint(self, option, index):
        return QSize(option.rect.width(), 22)

    def paint(self, p: QPainter, opt: QStyleOptionViewItem, index: QModelIndex):
        p.save()
        r = opt.rect
        sel = bool(opt.state & QStyle.StateFlag.State_Selected)
        hov = bool(opt.state & QStyle.StateFlag.State_MouseOver)
        full = QRect(0, r.top(), self.tree.viewport().width(), r.height())
        if sel:
            p.fillRect(full, THEME.color('sel'))
            p.fillRect(QRect(0, r.top(), 2, r.height()), THEME.color('ink'))
        elif hov:
            p.fillRect(full, THEME.color('raised'))
        kind = index.data(ROLE_KIND)
        flags = index.data(ROLE_FLAGS) or {}
        # gutter: eye / lock (shown on hover or when not default)
        if kind in ('camera', 'rig'):
            gy = r.top() + (r.height() - 12) // 2
            hidden, locked = flags.get('hidden'), flags.get('locked')
            if hov or hidden:
                p.drawPixmap(4, gy, icons.pixmap('eye-off' if hidden else 'eye', THEME.tok['muted' if hidden else 'ink2'], 12))
            elif locked:
                p.drawPixmap(4, gy, icons.pixmap('lock', THEME.tok['muted'], 12))
        x = r.left() + 2
        cy = r.top() + r.height() / 2
        text = index.data(Qt.ItemDataRole.DisplayRole) or ''
        meta = index.data(ROLE_META) or ''
        dim = flags.get('hidden')
        if kind == 'camera':
            slot, tag = index.data(ROLE_ICON)
            x = paint_chip(p, x + 2, cy, slot, tag) + 8
        else:
            ic = index.data(ROLE_ICON)
            if ic:
                p.drawPixmap(int(x), int(cy - 6), icons.pixmap(ic, THEME.tok['ink2'], 12))
                x += 18
        bold = kind in ('group',)
        p.setFont(sans(12, QFont.Weight.DemiBold if bold else QFont.Weight.Normal))
        p.setPen(THEME.color('muted' if dim else ('ink' if bold or sel or kind == 'camera' else 'ink2')))
        sev = index.data(ROLE_SEV)
        right = r.right() - 6
        if sev:
            p.drawPixmap(right - 12, int(cy - 6), icons.pixmap(sev, STATUS[sev], 12))
            right -= 16
        fm = QFontMetrics(mono(11))
        fmt = p.fontMetrics()
        tw = fmt.horizontalAdvance(text)
        space = right - x
        mw = fm.horizontalAdvance(meta) if meta else 0
        if meta and tw + 10 + mw > space:
            mw = max(0, space - tw - 10)
            meta = fm.elidedText(meta, Qt.TextElideMode.ElideLeft, int(mw)) if mw > 24 else ''
            mw = fm.horizontalAdvance(meta) if meta else 0
        avail = space - (mw + 10 if meta else 0)
        p.drawText(QRectF(x, r.top(), max(avail, 10), r.height()), Qt.AlignmentFlag.AlignVCenter,
                   fmt.elidedText(text, Qt.TextElideMode.ElideRight, int(max(avail, 10))))
        if meta:
            p.setFont(mono(11))
            p.setPen(THEME.color('muted'))
            p.drawText(QRectF(right - mw, r.top(), mw, r.height()), Qt.AlignmentFlag.AlignVCenter, meta)
        p.restore()


class Outliner(QTreeWidget):
    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        self.setHeaderHidden(True)
        self.setIndentation(12)
        self.setUniformRowHeights(True)
        self.setRootIsDecorated(True)
        self.setMouseTracking(True)
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.setItemDelegate(OutlinerDelegate(self))
        self.setEditTriggers(QAbstractItemView.EditTrigger.EditKeyPressed | QAbstractItemView.EditTrigger.DoubleClicked)
        self.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._menu)
        self.itemSelectionChanged.connect(self._on_select)
        self.itemChanged.connect(self._on_rename)
        self._block = False
        self.expanded_state = {'vehicle': True, 'cameras': True, 'compute': True, 'targets': False, 'analysis': True,
                               'obstacles': False, 'ports': False, 'rig': True, 'sensors': True}
        st = ctx.store
        st.changed.connect(lambda *_: self.rebuild())
        st.project_replaced.connect(self.rebuild)
        st.selection_changed.connect(self.sync_selection)
        ctx.analysis.finished.connect(lambda *_: self.rebuild())
        ctx.analysis.secondary.connect(lambda k, v: self.rebuild() if k in ('issues', 'targets') else None)
        THEME.changed.connect(self._theme_changed)      # a bound slot: disconnected when the widget goes
        self.rebuild()

    def _theme_changed(self, *_):
        self.viewport().update()

    def closeEditor(self, editor, hint):
        super().closeEditor(editor, hint)
        if getattr(self, '_rebuild_pending', False):
            QTimer.singleShot(0, self.rebuild)

    def count(self):
        n = 0
        it = self.invisibleRootItem()
        stack = [it]
        while stack:
            x = stack.pop()
            for i in range(x.childCount()):
                n += 1
                stack.append(x.child(i))
        return n

    def _item(self, parent, text, kind, ident=None, icon=None, meta='', sev=None, flags=None, editable=False):
        it = QTreeWidgetItem(parent, [text])
        it.setData(0, ROLE_KIND, kind)
        it.setData(0, ROLE_ID, ident)
        it.setData(0, ROLE_ICON, icon)
        it.setData(0, ROLE_META, meta)
        it.setData(0, ROLE_SEV, sev)
        it.setData(0, ROLE_FLAGS, flags or {})
        f = Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled
        if editable:
            f |= Qt.ItemFlag.ItemIsEditable | Qt.ItemFlag.ItemIsDragEnabled
        if kind == 'group' and ident == 'cameras':
            f |= Qt.ItemFlag.ItemIsDropEnabled
        it.setFlags(f)
        return it

    def rebuild(self):
        if self.state() == QAbstractItemView.State.EditingState:
            self._rebuild_pending = True    # rebuilding now would close the editor and lose what is being typed
            return
        self._rebuild_pending = False
        for i in range(self.topLevelItemCount()):
            it = self.topLevelItem(i)
            self.expanded_state[it.data(0, ROLE_ID)] = it.isExpanded()
        self._block = True
        self.clear()
        p = self.ctx.store.project
        v = p.variant
        res = self.ctx.analysis.result
        issues = self.ctx.analysis.issues
        sev_by_uid = {}
        order = {'critical': 0, 'serious': 1, 'warning': 2}
        for iss in issues:
            for u in iss.uids:
                cur = sev_by_uid.get(u)
                if cur is None or order.get(iss.severity, 9) < order.get(cur, 9):
                    sev_by_uid[u] = iss.severity
        veh = p.vehicle
        m = veh.mesh
        g = self._item(self, f'Vehicle — {veh.name if m is not None else "none"}', 'group', 'vehicle', 'car-side')
        if m is not None:
            vg = self.ctx.analysis.engine.vg
            occ = vg.body.n_tris if vg is not None else m.n_tris
            self._item(g, 'Body mesh', 'vehicle-part', 'vehicle', 'mesh', f'{_k(veh.imported_tris or m.n_tris)} · occ {_k(occ)}')
            if veh.wheel_centres:
                r = next(iter(veh.wheel_centres.values()))[2]
                self._item(g, f'Wheels ×{len(veh.wheel_centres)}', 'vehicle-part', 'wheels', 'axle', f'R {r:.3f}')
            self._item(g, 'Mount zones', 'vehicle-part', 'zones', 'mount-zone', f'{len(m.zones)} zones')
            if p.photos.frames:
                self._item(g, f'Reference photos ({len(p.photos.frames)})', 'vehicle-part', 'photos', 'photo-spin')
        if p.rig:
            rg = self._item(self, f'Rig ({len(p.rig)})', 'group', 'rig', 'layers',
                            f'{sum(1 for q in p.rig if q.visible and q.occludes)} blocking')
            for q in p.rig:
                self._item(rg, q.name, 'rig', q.uid, 'cylinder' if q.kind == 'cylinder' else 'cube',
                           q.kind if q.occludes else f'{q.kind} · see-through', flags={'hidden': not q.visible})
        if v.sensors:
            kinds = {q.kind for q in v.sensors}
            title = 'Lidars and radars' if kinds <= {'lidar', 'radar'} else 'Sensors'
            sg = self._item(self, f'{title} ({len(v.sensors)})', 'group', 'sensors', 'sensor', v.name)
            for q in v.sensors:
                m = p.sensor_model_for(q)
                it = self._item(sg, f'{q.tag} · {q.name}', 'sensor', q.uid, 'sensor', m.name,
                                sev_by_uid.get(q.uid), {'hidden': not q.visible, 'locked': q.locked})
                it.setToolTip(0, m.summary())
        cg = self._item(self, f'Cameras ({len(v.cameras)})', 'group', 'cameras', 'camera-wide', v.name)
        for c in v.cameras:
            it = self._item(cg, c.name, 'camera', c.uid, (c.slot, c.tag), f'port {c.port}' if c.port else 'no port',
                            sev_by_uid.get(c.uid), {'hidden': not c.visible, 'locked': c.locked}, editable=True)
            it.setToolTip(0, f'{c.tag} · {c.location}')
        e = p.ecu
        comp = self._item(self, f'Compute — {e.name}', 'group', 'compute', 'ecu')
        used = sum(1 for c in v.cameras if c.port)
        pr = self._item(comp, f'GMSL ports 1–{e.ports}', 'ecu', 'ports', 'cable', f'{used} / {e.ports}')
        for port in range(1, e.ports + 1):
            cam = next((c for c in v.cameras if c.port == port), None)
            sev = None
            if cam and route_length(p, cam) > cam.cable_m:
                sev = 'serious'
            self._item(pr, f'Port {port}' + (f' · {cam.tag}' if cam else ''), 'ecu', 'ports', None,
                       f'{route_length(p, cam):.2f} m' if cam else '—', sev)
        tg = self._item(self, f'Calibration targets ({len(p.targets)})', 'group', 'targets', 'checker')
        tst = {s.uid: s for s in self.ctx.analysis.targets}
        for t in p.targets:
            s = tst.get(t.uid)
            self._item(tg, t.name, 'target', t.uid, 'checker', ' '.join(x[1] for x in s.seen_by) if s else '',
                       None if (s is None or s.ok) else 'critical')
        an = self._item(self, 'Analysis', 'group', 'analysis', 'coverage')
        roi_meta = f'{res.metrics["roi_area"]:.3f} m²' if res else ''
        self._item(an, f'ROI {p.roi.distance:.1f} m' if p.roi.mode == 'offset' else 'ROI polygon', 'roi', 'roi', 'roi', roi_meta)
        ob = self._item(an, f'Obstacles ({len(p.obstacles)})', 'group2', 'obstacles', 'pole', 'poles')
        for o in p.obstacles:
            seen = res.obstacles.get(o.uid) if res else None
            self._item(ob, o.name, 'obstacle', o.uid, 'pole', f'{o.height:.1f} m', 'critical' if seen is not None and not seen else None)
        for i in range(self.topLevelItemCount()):
            it = self.topLevelItem(i)
            it.setExpanded(self.expanded_state.get(it.data(0, ROLE_ID), True))
        ob.setExpanded(self.expanded_state.get('obstacles', False))
        pr.setExpanded(self.expanded_state.get('ports', False))
        self._block = False
        self.sync_selection()
        if self.ctx.window:
            self.ctx.window.set_dock_count('outliner', str(self.count()))

    def sync_selection(self):
        self._block = True
        st = self.ctx.store
        sel_ids = set(st.sel_cams)
        obj = st.sel_obj
        self.clearSelection()
        stack = [self.invisibleRootItem()]
        while stack:
            x = stack.pop()
            for i in range(x.childCount()):
                ch = x.child(i)
                k, ident = ch.data(0, ROLE_KIND), ch.data(0, ROLE_ID)
                if (k == 'camera' and ident in sel_ids) or (obj and _obj_match(obj, k, ident)):
                    ch.setSelected(True)
                stack.append(ch)
        self._block = False
        self.viewport().update()

    def _on_select(self):
        if self._block:
            return
        items = self.selectedItems()
        cams = [it.data(0, ROLE_ID) for it in items if it.data(0, ROLE_KIND) == 'camera']
        st = self.ctx.store
        if cams:
            st.select_cams(cams)
            return
        if not items:
            return
        it = items[0]
        k, ident = it.data(0, ROLE_KIND), it.data(0, ROLE_ID)
        if k == 'vehicle-part':
            st.select_obj((ident,))
        elif k == 'target':
            st.select_obj(('target', ident))
        elif k == 'obstacle':
            st.select_obj(('obstacle', ident))
        elif k == 'rig':
            st.select_obj(('rig', ident))
        elif k == 'sensor':
            st.select_obj(('sensor', ident))
        elif k == 'roi':
            st.select_obj(('roi',))
        elif k == 'ecu':
            st.select_obj(('ecu',))
        elif k == 'group':
            st.select_obj({'cameras': ('variant',), 'targets': ('targets',), 'compute': ('ecu',),
                           'vehicle': ('vehicle',), 'analysis': ('roi',), 'rig': ('rig',),
                           'sensors': ('sensors',)}.get(ident, ()))
        elif k == 'group2':
            st.select_obj(('obstacles',))

    def _on_rename(self, it, col):
        if self._block or it.data(0, ROLE_KIND) != 'camera':
            return
        uid = it.data(0, ROLE_ID)
        name = it.text(0).strip()
        c = self.ctx.store.cam(uid)
        if c and name and name != c.name:
            self.ctx.store.edit('variant', lambda p: setattr(p.variant.cam(uid), 'name', name), f'Renamed {c.tag} to {name}')

    def mousePressEvent(self, e):
        it = self.itemAt(e.position().toPoint())
        if it is not None and e.position().x() < GUTTER and it.data(0, ROLE_KIND) == 'camera':
            uid = it.data(0, ROLE_ID)
            c = self.ctx.store.cam(uid)
            if c:
                ops.set_visible(self.ctx, [uid], not c.visible)
            return
        if it is not None and e.position().x() < GUTTER and it.data(0, ROLE_KIND) == 'sensor':
            from .sensorview import set_visible as sensor_visible
            sensor_visible(self.ctx, it.data(0, ROLE_ID), bool((it.data(0, ROLE_FLAGS) or {}).get('hidden')))
            return
        if it is not None and e.position().x() < GUTTER and it.data(0, ROLE_KIND) == 'rig':
            from .rigview import set_visible
            set_visible(self.ctx, it.data(0, ROLE_ID), bool((it.data(0, ROLE_FLAGS) or {}).get('hidden')))
            return
        super().mousePressEvent(e)

    def keyPressEvent(self, e):
        if e.key() == Qt.Key.Key_F2:
            it = self.currentItem()
            if it is not None and it.data(0, ROLE_KIND) == 'camera':
                self.editItem(it, 0)
                return
        if e.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace) and self.ctx.store.sel_cams:
            ops.delete_cameras(self.ctx, list(self.ctx.store.sel_cams))
            return
        if e.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            sens = [it.data(0, ROLE_ID) for it in self.selectedItems() if it.data(0, ROLE_KIND) == 'sensor']
            if sens:
                from .sensorview import delete_sensors
                delete_sensors(self.ctx, sens)
                return
            rig = [it.data(0, ROLE_ID) for it in self.selectedItems() if it.data(0, ROLE_KIND) == 'rig']
            if rig:
                from .rigview import delete_parts
                delete_parts(self.ctx, rig)
                return
        super().keyPressEvent(e)

    def dropEvent(self, e):
        src = [it for it in self.selectedItems() if it.data(0, ROLE_KIND) == 'camera']
        tgt = self.itemAt(e.position().toPoint())
        e.ignore()
        if not src or tgt is None or tgt.data(0, ROLE_KIND) != 'camera':
            return
        uid_t = tgt.data(0, ROLE_ID)
        moving = [it.data(0, ROLE_ID) for it in src]

        def fn(p):
            cams = p.variant.cameras
            mv = [c for c in cams if c.uid in moving]
            rest = [c for c in cams if c.uid not in moving]
            idx = next((i for i, c in enumerate(rest) if c.uid == uid_t), len(rest))
            p.variant.cameras = rest[:idx] + mv + rest[idx:]
        self.ctx.store.edit('variant', fn, 'Reordered cameras')

    def _menu(self, pos):
        it = self.itemAt(pos)
        if it is None or not self.ctx.window:
            return
        k = it.data(0, ROLE_KIND)
        gpos = self.viewport().mapToGlobal(pos)
        w = self.ctx.window
        if k == 'camera':
            if it.data(0, ROLE_ID) not in self.ctx.store.sel_cams:
                self.ctx.store.select_cams([it.data(0, ROLE_ID)])
            w.camera_menu(self.ctx.store.sel_cams, gpos)
        else:
            w.object_menu(k, it.data(0, ROLE_ID), gpos)


def _obj_match(obj, kind, ident):
    if obj[0] in ('target', 'obstacle'):
        return kind == obj[0] and ident == obj[1]
    if obj[0] == 'rig':
        return (kind == 'rig' and ident == obj[1]) if len(obj) > 1 else (kind == 'group' and ident == 'rig')
    if obj[0] == 'sensor':
        return kind == 'sensor' and ident == obj[1]
    if obj[0] == 'sensors':
        return kind == 'group' and ident == 'sensors'
    if obj[0] == 'roi':
        return kind == 'roi'
    if obj[0] in ('vehicle', 'wheels', 'zones', 'photos'):
        return kind == 'vehicle-part' and ident == obj[0]
    if obj[0] == 'ecu':
        return kind == 'ecu' and ident == 'ports'
    if obj[0] == 'variant':
        return kind == 'group' and ident == 'cameras'
    if obj[0] == 'targets':
        return kind == 'group' and ident == 'targets'
    if obj[0] == 'obstacles':
        return kind == 'group2'
    return False


def _k(n):
    return f'{n / 1000:.0f}k' if n >= 10000 else f'{n / 1000:.1f}k' if n >= 1000 else str(n)


# ------------------------------------------------------------------------------------------- library
class LibraryTree(QTreeWidget):
    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        self.setHeaderHidden(True)
        self.setIndentation(12)
        self.setRootIsDecorated(True)
        self.setMouseTracking(True)
        self.setItemDelegate(OutlinerDelegate(self))
        self.setDragEnabled(True)
        self.setDragDropMode(QAbstractItemView.DragDropMode.DragOnly)
        self.itemDoubleClicked.connect(self._edit)
        self.itemClicked.connect(self._pick)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._menu)

    def startDrag(self, actions):
        it = self.currentItem()
        if it is None or it.data(0, ROLE_KIND) != 'lens':
            return
        mid, lid = it.data(0, ROLE_ID)
        md = QMimeData()
        md.setData('application/x-vantage-lens', f'{mid}|{lid}'.encode())
        d = QDrag(self)
        d.setMimeData(md)
        d.setPixmap(icons.pixmap('camera-wide', THEME.tok['ink'], 16))
        d.exec(Qt.DropAction.CopyAction)

    def _pick(self, it, col):
        if it.data(0, ROLE_KIND) == 'lens':
            mid, lid = it.data(0, ROLE_ID)
            self.ctx.place_lens = (mid, lid)
            self.ctx.library_pick.emit(mid, lid)

    def _edit(self, it, col):
        if it.data(0, ROLE_KIND) == 'lens' and self.ctx.window:
            mid, lid = it.data(0, ROLE_ID)
            self.ctx.window.edit_module(mid, lid)
        elif it.data(0, ROLE_KIND) == 'template' and self.ctx.window:
            self.ctx.window.new_from_template(it.data(0, ROLE_ID))

    def _menu(self, pos):
        it = self.itemAt(pos)
        m = QMenu(self)
        if it is not None and it.data(0, ROLE_KIND) == 'lens':
            mid, lid = it.data(0, ROLE_ID)
            m.addAction('Edit module…', lambda: self.ctx.window.edit_module(mid, lid))
            m.addAction('Use for new cameras', lambda: self._pick(it, 0))
            sel = self.ctx.store.sel_cams
            if sel:
                m.addAction(f'Apply to selected ({len(sel)})', lambda: ops.set_lens(self.ctx, list(sel), mid, lid))
            if not self.ctx.store.project.lenses[lid].builtin:
                m.addSeparator()
                m.addAction('Delete lens', lambda: self.ctx.window.delete_lens(lid))
        m.addSeparator()
        m.addAction('Import camera module…', lambda: self.ctx.window.import_module())
        m.exec(self.viewport().mapToGlobal(pos))


class LibraryPanel(QWidget):
    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        self.setObjectName('panel')
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 6, 8, 6)
        lay.setSpacing(6)
        top = QHBoxLayout()
        top.setSpacing(4)
        self.search = QLineEdit()
        self.search.setPlaceholderText('Search')
        self.search.addAction(icons.icon('search', 12), QLineEdit.ActionPosition.LeadingPosition)
        self.kind = QComboBox()
        self.kind.addItems(['Camera modules', 'Lenses', 'Vehicles', 'Brackets'])
        self.kind.setFixedWidth(126)
        top.addWidget(self.search, 1)
        top.addWidget(self.kind)
        lay.addLayout(top)
        self.tree = LibraryTree(ctx)
        lay.addWidget(self.tree, 1)
        self.hint = label('Drag a module onto the body to place it.\nDouble-click to edit.', muted=True, small=True)
        lay.addWidget(self.hint)
        self.search.textChanged.connect(self.rebuild)
        self.kind.currentIndexChanged.connect(self.rebuild)
        ctx.store.changed.connect(lambda s: self.rebuild() if 'library' in s else None)
        ctx.store.project_replaced.connect(self.rebuild)
        THEME.changed.connect(self._theme_changed)      # a bound slot: disconnected when the widget goes
        self.rebuild()

    def _theme_changed(self, *_):
        self.tree.viewport().update()

    def rebuild(self):
        t = self.tree
        t.clear()
        q = self.search.text().lower().strip()
        p = self.ctx.store.project
        kind = self.kind.currentIndex()
        from ..core.lens import model_fov
        if kind in (0, 1):
            used = {c.module_id for c in p.variant.cameras}
            for mid, m in p.modules.items():
                hay = f'{m.vendor} {m.name} {m.sensor} {m.interface}'
                shown = [(lid, l) for lid, l in p.lenses.items() if (not l.module_id or l.module_id == mid)
                         and (not q or q in f'{l.name} {l.label} {hay}'.lower())]
                if q and not shown:
                    continue                                    # a search hides the modules it does not match
                grp = QTreeWidgetItem(t, [f'{m.name} · {m.sensor.replace("Sony ", "")}'.strip(' ·')])
                grp.setData(0, ROLE_KIND, 'group2')
                grp.setData(0, ROLE_ICON, None)
                grp.setFlags(Qt.ItemFlag.ItemIsEnabled)
                grp.setToolTip(0, m.summary() + (f'\n\n{m.note}' if m.note else ''))
                for lid, l in shown:
                    fh, fv, fd = l.sheet_fov if l.sheet_fov[0] else model_fov(m, l)
                    it = QTreeWidgetItem(grp, [l.name])
                    it.setData(0, ROLE_KIND, 'lens')
                    it.setData(0, ROLE_ID, (mid, lid))
                    it.setData(0, ROLE_ICON, 'camera-wide' if fh > 90 else 'camera-narrow')
                    it.setData(0, ROLE_META, f'{fh:.1f}° × {fv:.1f}° · {l.focal_mm:.3g}')
                    it.setToolTip(0, f'{m.name} + {l.summary()}' + (f'\n\n{l.note}' if l.note else ''))
                    it.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsDragEnabled)
                grp.setExpanded(bool(q) or mid in used)        # the modules in use open; the rest of the library folds
            custom = [l for l in p.lenses.values() if not l.builtin]
            proj = QTreeWidgetItem(t, [f'Project modules ({len(custom)})'])
            proj.setData(0, ROLE_KIND, 'group2')
            proj.setFlags(Qt.ItemFlag.ItemIsEnabled)
        elif kind == 2:
            from ..core.bodygen import template_specs
            items = [('tigor', 'Tata Tigor EV', 'measured from photos', 'sedan')] + \
                    [(k, s.name, f'{s.length:.1f} × {s.width:.1f} × {s.height:.1f} m', k if k != 'truck' else 'truck')
                     for k, s in template_specs().items()]
            for key, name, meta, ic in items:
                if q and q not in name.lower():
                    continue
                it = QTreeWidgetItem(t, [name])
                it.setData(0, ROLE_KIND, 'template')
                it.setData(0, ROLE_ID, key)
                it.setData(0, ROLE_ICON, {'sedan': 'sedan', 'hatchback': 'hatchback', 'suv': 'suv', 'van': 'van', 'truck': 'truck', 'tigor': 'sedan'}.get(key, 'car-side'))
                it.setData(0, ROLE_META, meta)
                it.setToolTip(0, 'Double-click to start a new project from this body')
        else:
            for name, meta in (('Flat plate · 25 mm', 'bumper, grille'), ('Mirror underside clip', 'mirror housing'),
                               ('Quarter-panel pod · 40 mm', 'body side'), ('Boot-lid wedge 10°', 'boot lid')):
                if q and q not in name.lower():
                    continue
                it = QTreeWidgetItem(t, [name])
                it.setData(0, ROLE_KIND, 'bracket')
                it.setData(0, ROLE_ICON, 'bumper-mount')
                it.setData(0, ROLE_META, meta)
        self.hint.setText('Drag a module onto the body to place it.\nDouble-click to edit.' if kind < 2 else
                          'Double-click a body to start a project from it.' if kind == 2 else 'Brackets set the camera standoff.')
