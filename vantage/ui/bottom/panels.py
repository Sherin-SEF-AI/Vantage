"""Camera view dock and the bottom analysis panels (Coverage, Cameras, Tolerance, Issues, History, Log)."""
from __future__ import annotations

import math
import time

import numpy as np
from PyQt6.QtCore import Qt, QAbstractTableModel, QModelIndex, QTimer, pyqtSignal, QRectF, QPointF, QSize
from PyQt6.QtGui import QColor, QFont, QPainter, QPen, QKeySequence, QGuiApplication
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QTabBar, QStackedWidget,
                             QTableView, QHeaderView, QAbstractItemView, QListWidget, QListWidgetItem, QStyledItemDelegate,
                             QStyle, QPushButton, QToolButton, QMenu, QPlainTextEdit, QSizePolicy, QFrame, QComboBox,
                             QCheckBox)

from ...core.lens import resolve, PROJECTION_LABEL
from ...core.mathutil import parse_number
from ...core.rules import route_length
from .. import icons, ops
from ..camview import CamImageView
from ..charts import BarList, DensityHistogram
from ..theme import THEME, mono, sans, STATUS
from ..widgets import (label, hline, vline, ValueLabel, tool_button, button, paint_chip, ScrubSpinBox, ROW_H,
                       StatusLine)


# ---------------------------------------------------------------------------------------- camera view dock
class CameraDock(QWidget):
    titleChanged = pyqtSignal(str)

    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        self.setObjectName('panel')
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 4, 8, 6)
        lay.setSpacing(6)
        top = QHBoxLayout()
        top.setSpacing(8)
        self.view = CamImageView(ctx, legend=False)
        self.view.setMinimumSize(150, 120)
        top.addWidget(self.view, 3)
        g = QGridLayout()
        g.setSpacing(0)
        g.setVerticalSpacing(1)
        self.vals = {}
        for i, k in enumerate(('Body in view', 'Ground', 'Min in ROI', 'Seams', 'Grid')):
            a = label(k, ink2=True, small=True)
            v = QLabel('')
            v.setFont(mono(11))
            v.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            g.addWidget(a, i, 0)
            g.addWidget(v, i, 1)
            self.vals[k] = v
        g.setRowStretch(5, 1)
        gw = QWidget()
        gw.setLayout(g)
        gw.setFixedWidth(138)
        top.addWidget(gw)
        lay.addLayout(top, 1)
        sw = QWidget()
        sw.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        sw.setFixedHeight(37)
        self.strip = QHBoxLayout(sw)
        self.strip.setContentsMargins(0, 0, 0, 0)
        self.strip.setSpacing(4)
        lay.addWidget(sw)
        self.thumbs = []
        st = ctx.store
        st.changed.connect(lambda *_: self.rebuild_thumbs())
        st.project_replaced.connect(self.rebuild_thumbs)
        st.selection_changed.connect(self.update_text)
        ctx.analysis.finished.connect(lambda *_: QTimer.singleShot(80, self.update_text))
        self.view.timer.timeout.connect(lambda: QTimer.singleShot(5, self.update_text))
        self.rebuild_thumbs()

    def rebuild_thumbs(self):
        cams = self.ctx.store.variant.cameras
        uids = [c.uid for c in cams]
        if [t.uid for t in self.thumbs] == uids:
            for t in self.thumbs:
                t.update()
            self.update_text()
            return
        for t in self.thumbs:
            self.strip.removeWidget(t)
            t.deleteLater()
        self.thumbs = []
        while self.strip.count():
            it = self.strip.takeAt(0)
            if it.widget():
                it.widget().setParent(None)
        for c in cams:
            t = Thumb(self.ctx, c.uid)
            self.strip.addWidget(t)
            self.thumbs.append(t)
        self.strip.addStretch(1)
        self.update_text()

    def update_text(self):
        c = self.ctx.store.selected
        p = self.ctx.store.project
        for t in self.thumbs:
            t.update()
        if c is None:
            self.titleChanged.emit('')
            for v in self.vals.values():
                v.setText('')
            return
        lens = p.lens_for(c)
        m = p.module_for(c)
        self.titleChanged.emit(f'{c.tag} · {PROJECTION_LABEL[lens.projection].lower()} · {m.width_px} × {m.height_px}')
        st = self.view.stats
        self.vals['Body in view'].setText(f'{st.get("body_pct", 0):.1f} %' if st else '…')
        self.vals['Ground'].setText(f'{st.get("ground_pct", 0):.1f} %' if st else '…')
        res = self.ctx.analysis.result
        if res and c.uid in res.per_cam:
            self.vals['Min in ROI'].setText(f'{res.per_cam[c.uid]["min_dens"]:.1f} px/cm')
            nb = []
            for s in res.seams:
                if s.a == c.uid:
                    nb.append(s.b)
                elif s.b == c.uid:
                    nb.append(s.a)
            self.vals['Seams'].setText(' · '.join(self.ctx.store.cam(u).tag for u in nb if self.ctx.store.cam(u)))
        self.vals['Grid'].setText('0.5 m')


class Thumb(CamImageView):
    def __init__(self, ctx, uid):
        super().__init__(ctx, uid=uid, legend=False, res_scale=1.0)
        self.setFixedSize(44, 35)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def paintEvent(self, e):
        super().paintEvent(e)
        c = self.camera()
        if c is None:
            return
        p = QPainter(self)
        r = self.rect()
        p.fillRect(QRectF(0, r.height() - 13, r.width(), 13), QColor(0, 0, 0, 150))
        paint_chip(p, 3, r.height() - 6.5, c.slot, c.tag, size=6, font=mono(9, QFont.Weight.DemiBold), color=QColor(240, 239, 236))
        if c.uid in self.ctx.store.sel_cams:
            p.setPen(QPen(THEME.color('ink'), 1))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRect(QRectF(0.5, 0.5, r.width() - 1, r.height() - 1))
        p.end()


# ---------------------------------------------------------------------------------------- coverage tab
class CoverageTab(QWidget):
    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        self.setObjectName('panel')
        lay = QHBoxLayout(self)
        lay.setContentsMargins(8, 4, 8, 6)
        lay.setSpacing(12)
        left = QWidget()
        left.setFixedWidth(340)
        g = QGridLayout(left)
        g.setContentsMargins(0, 0, 0, 0)
        g.setVerticalSpacing(1)
        g.setHorizontalSpacing(6)
        t = label('Coverage', bold=True)
        self.head = label('', muted=True, small=True)
        self.head.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        g.addWidget(t, 0, 0)
        g.addWidget(self.head, 0, 1, 1, 2)
        self.rows = {}
        keys = ['Blind in ROI', 'Covered', 'Seen by ≥ 2 cameras', 'Minimum density', 'In focus', 'Worst single-camera error',
                'Object vanishes at seams', 'ROI area']
        for i, k in enumerate(keys, start=1):
            a = label(k, ink2=True)
            v = ValueLabel(align_right=True)
            g.addWidget(a, i, 0)
            g.addWidget(v, i, 1, 1, 2)
            self.rows[k] = v
        g.setRowStretch(len(keys) + 1, 1)
        lay.addWidget(left)
        lay.addWidget(vline())
        self.bars = BarList()
        self.bars.title = 'Ground seen only by this camera'
        self.bars.setFixedWidth(520)
        self.bars.picked.connect(lambda uid: ctx.store.select_cams([uid]))
        lay.addWidget(self.bars)
        lay.addWidget(vline())
        self.hist = DensityHistogram()
        lay.addWidget(self.hist, 1)
        ctx.analysis.finished.connect(lambda *_: self.refresh())
        ctx.analysis.secondary.connect(lambda k, v: self.refresh() if k == 'worst' else None)
        ctx.analysis.stale_changed.connect(lambda s: self._stale(s))
        ctx.store.selection_changed.connect(self.refresh)
        if getattr(ctx, 'stitched', None) is not None:
            ctx.stitched.ready.connect(self.refresh)
        self.refresh()

    def _stale(self, s):
        eff = self.graphicsEffect()
        from PyQt6.QtWidgets import QGraphicsOpacityEffect
        if s:
            ef = QGraphicsOpacityEffect(self)
            ef.setOpacity(0.45)
            self.setGraphicsEffect(ef)
            self.head.setText('stale · recomputes on release' if self.ctx.store.project.analysis.live else 'stale · F5 to run')
        else:
            self.setGraphicsEffect(None)
            self.refresh()

    def refresh(self):
        res = self.ctx.analysis.result
        p = self.ctx.store.project
        if res is None:
            for v in self.rows.values():
                v.set('—')
            self.bars.set_rows([])
            return
        m = res.metrics
        if not self.ctx.analysis.stale:
            self.head.setText(f'ROI {p.roi.distance:.1f} m · {res.cell * 100:.0f} cm cells · {res.ms:.0f} ms')
        self.rows['Blind in ROI'].set(f'{m["blind_m2"]:.3f}', 'm²', 'good' if m['blind_m2'] < 0.0005 else 'critical')
        self.rows['Covered'].set(f'{m["covered_pct"]:.1f}', '%')
        self.rows['Seen by ≥ 2 cameras'].set(f'{m["multi_pct"]:.1f}', '%')
        md = m['min_density']
        self.rows['Minimum density'].set(f'{md:.1f}', 'px/cm', 'good' if md >= p.analysis.min_density else 'warning')
        self.rows['In focus'].set(f'{m["in_focus_pct"]:.1f}', '%')
        w = self.ctx.analysis.worst
        tol = f'{p.analysis.tol_deg:g}° / {p.analysis.tol_mm / 10:g} cm'
        self.rows['Worst single-camera error'].set(f'{w[0]:.3f}' if w else '…', 'm²', None, tol)
        self.rows['ROI area'].set(f'{m["roi_area"]:.3f}', 'm²')
        sres = getattr(getattr(self.ctx, 'stitched', None), 'result', None)
        if sres is None:
            self.rows['Object vanishes at seams'].set('…', 'm²')
        else:
            v = sres.metrics['vanish_m2']
            from ...core.stitched_objects import ISSUE_M2
            self.rows['Object vanishes at seams'].set(f'{v:.2f}', 'm²', 'good' if v <= ISSUE_M2 else 'warning',
                                                      sres.settings.label)
        cams = {c.uid: c for c in p.variant.cameras}
        rows = [(c.uid, c.slot, c.tag, res.per_cam[c.uid]['unique_m2']) for c in res.cams if c.uid in cams]
        self.bars.set_rows(rows, self.ctx.store.sel_cams)
        edges, areas, _ = res.hist
        self.hist.set_data(edges, areas, p.analysis.min_density, p.analysis.target_density)


# ---------------------------------------------------------------------------------------- cameras table
COLS = [('tag', 'Tag'), ('location', 'Name'), ('x', 'X'), ('y', 'Y'), ('z', 'Z'), ('yaw', 'Yaw °'), ('pitch', 'Pitch °'),
        ('roll', 'Roll °'), ('module', 'Module'), ('lens', 'Lens'), ('port', 'Port'), ('cable', 'Cable m'),
        ('unique', 'Unique m²'), ('mind', 'Min px/cm')]


class CamerasModel(QAbstractTableModel):
    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        ctx.store.changed.connect(lambda *_: self.reset())
        ctx.store.project_replaced.connect(self.reset)
        ctx.analysis.finished.connect(lambda *_: self.reset())

    def reset(self):
        self.beginResetModel()
        self.endResetModel()

    @property
    def cams(self):
        return self.ctx.store.variant.cameras

    def rowCount(self, parent=QModelIndex()):
        return len(self.cams)

    def columnCount(self, parent=QModelIndex()):
        return len(COLS)

    def headerData(self, s, o, role=Qt.ItemDataRole.DisplayRole):
        if o == Qt.Orientation.Horizontal:
            if role == Qt.ItemDataRole.DisplayRole:
                k, t = COLS[s]
                if k in ('x', 'y', 'z'):
                    return f'{t} {self.ctx.store.project.units}'
                return t
            if role == Qt.ItemDataRole.TextAlignmentRole:
                return Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter if s >= 2 and COLS[s][0] not in ('module', 'lens') else Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        return None

    def data(self, idx, role=Qt.ItemDataRole.DisplayRole):
        c = self.cams[idx.row()]
        k = COLS[idx.column()][0]
        p = self.ctx.store.project
        o = p.origin_offset()
        unit = 1000.0 if p.units == 'mm' else 1.0
        dp = 0 if p.units == 'mm' else 3
        if role in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.EditRole):
            res = self.ctx.analysis.result
            if k == 'tag':
                return c.tag
            if k == 'location':
                return c.location or c.name
            if k in ('x', 'y', 'z'):
                i = 'xyz'.index(k)
                v = (getattr(c, k) - o[i]) * unit
                return f'{v:.{dp}f}'.replace('-', '−') if role == Qt.ItemDataRole.DisplayRole else f'{v:.{dp + 1}f}'
            if k in ('yaw', 'pitch', 'roll'):
                return f'{getattr(c, k):.1f}'.replace('-', '−')
            if k == 'module':
                return p.module_for(c).name
            if k == 'lens':
                return p.lens_for(c).name
            if k == 'port':
                return str(c.port) if c.port else '—'
            if k == 'cable':
                return f'{route_length(p, c):.2f} / {c.cable_m:.2f}'
            if k == 'unique':
                return f'{res.per_cam[c.uid]["unique_m2"]:.3f}' if res and c.uid in res.per_cam else ''
            if k == 'mind':
                return f'{res.per_cam[c.uid]["min_dens"]:.1f}' if res and c.uid in res.per_cam else ''
        if role == Qt.ItemDataRole.FontRole and k not in ('location', 'module', 'lens'):
            return mono(12, QFont.Weight.DemiBold if k == 'tag' else None)
        if role == Qt.ItemDataRole.TextAlignmentRole:
            if k in ('tag', 'location', 'module', 'lens'):
                return Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
            return Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        if role == Qt.ItemDataRole.DecorationRole and k == 'cable':
            return icons.status_icon('serious' if route_length(p, c) > c.cable_m else 'good', 12)
        if role == Qt.ItemDataRole.UserRole:
            return c
        return None

    def flags(self, idx):
        k = COLS[idx.column()][0]
        f = Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled
        if k in ('tag', 'location', 'x', 'y', 'z', 'yaw', 'pitch', 'roll', 'port') and not self.cams[idx.row()].locked:
            f |= Qt.ItemFlag.ItemIsEditable
        return f

    def setData(self, idx, value, role=Qt.ItemDataRole.EditRole):
        return self.set_cell(idx.row(), idx.column(), str(value))

    def set_cell(self, r, col, text, merge=False, check=False):
        """Apply a typed or pasted cell; with check=True only report whether it can be read."""
        c = self.cams[r]
        k = COLS[col][0]
        p = self.ctx.store.project
        o = p.origin_offset()
        st = self.ctx.store
        try:
            if k in ('x', 'y', 'z'):
                v = parse_number(text, 'length', p.units)
                v = v / (1000.0 if p.units == 'mm' else 1.0) + o['xyz'.index(k)]
                edit = lambda: st.set_cam(c.uid, **{k: round(v, 4)})
            elif k in ('yaw', 'pitch', 'roll'):
                v = round(parse_number(text, 'angle'), 2)
                edit = lambda: st.set_cam(c.uid, **{k: v})
            elif k == 'tag':
                t = text.strip()[:4]
                edit = lambda: st.set_cam(c.uid, tag=t, text=f'Renamed tag {c.tag} to {t}') if t else None
            elif k == 'location':
                edit = lambda: st.set_cam(c.uid, location=text.strip(), text=f'Renamed {c.tag}')
            elif k == 'port':
                port = 0 if text.strip() in ('—', '-', '') else float(text)
                if port != int(port) or not 0 <= port <= p.ecu.ports:
                    if not check:
                        self.ctx.msg(f'GMSL port must be 1 to {p.ecu.ports}, or 0 to unassign.')
                    return False
                port = int(port)
                # 0 unassigns: no other camera takes this one's port
                edit = (lambda: ops.assign_port(self.ctx, c.uid, port)) if port else \
                    (lambda: st.set_cam(c.uid, port=0, text=f'Unassigned {c.tag} from GMSL port {c.port}'))
            else:
                return False
        except Exception:
            if not check:
                self.ctx.msg(f'Could not read “{text}”. Use a number with an optional unit, for example 950 mm or 1.88+0.05.')
            return False
        if check:
            return True
        edit()
        self.ctx.analysis.request()
        return True


class ChipDelegate(QStyledItemDelegate):
    def paint(self, p, opt, idx):
        sel = bool(opt.state & QStyle.StateFlag.State_Selected)
        if sel:
            p.fillRect(opt.rect, THEME.color('sel'))
            p.fillRect(QRectF(opt.rect.left(), opt.rect.top(), 2, opt.rect.height()), THEME.color('ink'))
        elif opt.state & QStyle.StateFlag.State_MouseOver:
            p.fillRect(opt.rect, THEME.color('raised'))
        c = idx.data(Qt.ItemDataRole.UserRole)
        paint_chip(p, opt.rect.left() + 8, opt.rect.center().y() + 0.5, c.slot, c.tag)


class CamerasTab(QWidget):
    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        self.setObjectName('panel')
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        self.model = CamerasModel(ctx)
        self.view = QTableView()
        self.view.setModel(self.model)
        self.view.verticalHeader().setVisible(False)
        self.view.verticalHeader().setDefaultSectionSize(ROW_H)
        self.view.horizontalHeader().setFixedHeight(ROW_H)
        self.view.setShowGrid(False)
        self.view.setMouseTracking(True)
        self.view.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.view.setEditTriggers(QAbstractItemView.EditTrigger.DoubleClicked | QAbstractItemView.EditTrigger.EditKeyPressed
                                  | QAbstractItemView.EditTrigger.AnyKeyPressed)
        hh = self.view.horizontalHeader()
        hh.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        hh.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        self.view.setColumnWidth(0, 70)
        self.view.setItemDelegateForColumn(0, ChipDelegate(self.view))
        self.view.selectionModel().selectionChanged.connect(self._sel)
        self.view.installEventFilter(self)
        lay.addWidget(self.view, 1)
        foot = label('  Paste rows from a spreadsheet to update positions. Enter commits, Esc reverts. Expressions and units '
                     'are accepted, for example 950 mm or 1.88+0.05.', muted=True, small=True)
        foot.setFixedHeight(20)
        lay.addWidget(foot)
        ctx.store.selection_changed.connect(self._sync)
        self.model.modelReset.connect(self._sync)
        self._block = False

    def _sel(self, *_):
        if self._block:
            return
        rows = sorted({i.row() for i in self.view.selectionModel().selectedRows()})
        uids = [self.model.cams[r].uid for r in rows if r < len(self.model.cams)]
        if uids and set(uids) != set(self.ctx.store.sel_cams):
            self.ctx.store.select_cams(uids)

    def _sync(self):
        self._block = True
        sm = self.view.selectionModel()
        sm.clearSelection()
        from PyQt6.QtCore import QItemSelection, QItemSelectionModel
        for r, c in enumerate(self.model.cams):
            if c.uid in self.ctx.store.sel_cams:
                sm.select(self.model.index(r, 0), QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows)
        self._block = False

    def eventFilter(self, obj, e):
        from PyQt6.QtCore import QEvent
        if e.type() == QEvent.Type.KeyPress and (e.matches(QKeySequence.StandardKey.Paste)):
            self.paste()
            return True
        if e.type() == QEvent.Type.KeyPress and e.matches(QKeySequence.StandardKey.Copy):
            self.copy()
            return True
        return False

    def copy(self):
        rows = sorted({i.row() for i in self.view.selectionModel().selectedRows()}) or range(self.model.rowCount())
        lines = ['\t'.join(t for _, t in COLS)]
        for r in rows:
            lines.append('\t'.join(str(self.model.data(self.model.index(r, c))) for c in range(len(COLS))))
        QGuiApplication.clipboard().setText('\n'.join(lines))
        self.ctx.msg(f'Copied {len(rows)} rows')

    def paste(self):
        text = QGuiApplication.clipboard().text().strip()
        if not text:
            return
        idx = self.view.currentIndex()
        r0 = idx.row() if idx.isValid() else 0
        c0 = idx.column() if idx.isValid() else 0
        lines = [l.split('\t') for l in text.splitlines() if l.strip()]
        if lines and lines[0] and lines[0][0].strip() in ('Tag', 'tag'):
            lines = lines[1:]
            c0 = 0
        cams = self.model.cams
        tags = {c.tag: i for i, c in enumerate(cams)}
        cells, locked, unknown, bad = [], [], [], 0
        for k, row in enumerate(lines):
            r = r0 + k
            if c0 == 0 and row and row[0].strip():
                if row[0].strip() not in tags:
                    unknown.append(row[0].strip())       # a row for a camera this variant does not have
                    continue
                r = tags[row[0].strip()]
            if r >= len(cams):
                break
            if cams[r].locked:
                locked.append(cams[r].tag)
                continue
            for j, val in enumerate(row):
                col = c0 + j
                if col >= len(COLS) or COLS[col][0] in ('tag', 'module', 'lens', 'cable', 'unique', 'mind'):
                    continue
                if val.strip():
                    if self.model.set_cell(r, col, val.strip(), check=True):
                        cells.append((r, col, val.strip()))
                    else:
                        bad += 1
        if cells:
            # one undo step for the whole paste
            st = self.ctx.store
            st.undo.beginMacro(f'Pasted {len(cells)} values')
            try:
                for r, col, val in cells:
                    self.model.set_cell(r, col, val)
            finally:
                st.undo.endMacro()
        skipped = ([f'locked {", ".join(locked)}'] if locked else []) + ([f'unknown {", ".join(unknown)}'] if unknown else []) + \
            ([f'{bad} unreadable'] if bad else [])
        self.ctx.msg(f'Pasted {len(cells)} values' + (f' · skipped {"; ".join(skipped)}' if skipped else ''))


# ---------------------------------------------------------------------------------------- tolerance (compact)
class ToleranceTab(QWidget):
    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        self.setObjectName('panel')
        lay = QHBoxLayout(self)
        lay.setContentsMargins(8, 6, 8, 6)
        lay.setSpacing(12)
        col = QVBoxLayout()
        row = QHBoxLayout()
        row.setSpacing(6)
        self.deg = ScrubSpinBox('±', 'angle', 1, '°', width=86)
        self.mm = ScrubSpinBox('±', 'plain', 0, 'mm', step=1, width=86)
        self.n = ScrubSpinBox('N', 'plain', 0, '', step=50, width=86)
        self.seed = ScrubSpinBox('seed', 'plain', 0, '', step=1, width=100)
        self.deg.set_value(5.0)
        self.mm.set_value(10)
        self.n.set_value(2000)
        self.seed.set_value(1729)
        for w in (label('Yaw/pitch/roll', ink2=True), self.deg, label('Position', ink2=True), self.mm, label('Samples', ink2=True),
                  self.n, self.seed):
            row.addWidget(w)
        row.addStretch(1)
        col.addLayout(row)
        self.summary = label('No study run yet. Run a Monte Carlo study with the settings above, or open the full study view.',
                             ink2=True, wrap=True)
        col.addWidget(self.summary)
        col.addStretch(1)
        bl = QHBoxLayout()
        self.b_run = button('Run study', 'primary')
        self.b_run.clicked.connect(self.run)
        b_open = button('Open study view')
        b_open.clicked.connect(lambda: ctx.window.set_page('tolerance'))
        bl.addWidget(self.b_run)
        bl.addWidget(b_open)
        bl.addStretch(1)
        col.addLayout(bl)
        lay.addLayout(col, 1)

    def settings(self):
        from ...core.tolerance import TolSettings
        d = self.deg.value()
        return TolSettings(yaw=d, pitch=d, roll=d, pos_mm=self.mm.value(), samples=int(self.n.value()), seed=int(self.seed.value()))

    def run(self):
        self.ctx.window.run_tolerance(self.settings())

    def show_result(self, r):
        ran = f'cancelled after {len(r.blind)} of {r.settings.samples} samples' if r.cancelled else f'{len(r.blind)} samples'
        self.summary.setText(f'{ran} in {r.seconds:.1f} s · zero blind in {r.zero_pct:.1f} % · '
                             f'P95 {r.pct(95):.3f} m² · P99 {r.pct(99):.3f} m² · max {r.blind.max():.3f} m² · '
                             f'nominal {r.nominal_blind:.3f} m²')


# ---------------------------------------------------------------------------------------- issues
class IssuesTab(QWidget):
    countChanged = pyqtSignal(int, str)

    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        self.setObjectName('panel')
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        from PyQt6.QtWidgets import QTableWidget, QTableWidgetItem
        self.t = QTableWidget(0, 4)
        self.t.setHorizontalHeaderLabels(['Severity', 'Message', 'Object', ''])
        self.t.verticalHeader().setVisible(False)
        self.t.verticalHeader().setDefaultSectionSize(ROW_H + 2)
        self.t.horizontalHeader().setFixedHeight(ROW_H)
        self.t.setShowGrid(False)
        self.t.setMouseTracking(True)
        self.t.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.t.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        hh = self.t.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        hh.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        hh.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(3, QHeaderView.ResizeMode.Fixed)
        self.t.setColumnWidth(0, 130)
        self.t.setColumnWidth(3, 112)
        self.t.cellClicked.connect(self._show)
        self.t.cellEntered.connect(self._hover)
        self.t.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.t.customContextMenuRequested.connect(self._menu)
        lay.addWidget(self.t)
        ctx.analysis.secondary.connect(lambda k, v: self.refresh() if k == 'issues' else None)
        self._hover_row = -1

    def refresh(self):
        from PyQt6.QtWidgets import QTableWidgetItem
        iss = self.ctx.analysis.issues
        self.t.setRowCount(len(iss))
        for i, x in enumerate(iss):
            a = QTableWidgetItem(x.severity.capitalize())
            a.setIcon(icons.status_icon(x.severity, 12))
            self.t.setItem(i, 0, a)
            self.t.setItem(i, 1, QTableWidgetItem(x.message))
            o = QTableWidgetItem(x.objects)
            o.setFont(mono(11))
            o.setForeground(THEME.color('ink2'))
            self.t.setItem(i, 2, o)
            cell = QWidget()
            h = QHBoxLayout(cell)
            h.setContentsMargins(0, 0, 4, 0)
            h.setSpacing(4)
            ask = QPushButton('Ask')
            ask.setIcon(icons.icon('sparkle', 12))
            ask.setToolTip('Ask the assistant to explain this issue and suggest a fix')
            ask.clicked.connect(lambda _=False, i=i: self._ask(i))
            b = QPushButton('Show')
            b.clicked.connect(lambda _=False, i=i: self._show(i, 0))
            h.addWidget(ask)
            h.addWidget(b)
            cell.setVisible(False)
            self.t.setCellWidget(i, 3, cell)
        worst = iss[0].severity if iss else ''
        self.countChanged.emit(len(iss), worst)

    def _hover(self, r, c):
        if r == self._hover_row:
            return
        for k in range(self.t.rowCount()):
            w = self.t.cellWidget(k, 3)
            if w:
                w.setVisible(k == r)
        self._hover_row = r

    def _ask(self, r):
        iss = self.ctx.analysis.issues
        if 0 <= r < len(iss) and self.ctx.window is not None:
            x = iss[r]
            self.ctx.window.ask_assistant(f'Explain this issue and suggest a fix: "{x.message}" '
                                          f'({x.severity}; {x.objects}).')

    def _menu(self, pos):
        r = self.t.rowAt(pos.y())
        if r < 0:
            return
        m = QMenu(self)
        m.addAction(icons.icon('frame', 16), 'Show on the vehicle', lambda: self._show(r, 0))
        m.addAction(icons.icon('sparkle', 16), 'Ask the assistant', lambda: self._ask(r))
        m.exec(self.t.viewport().mapToGlobal(pos))

    def _show(self, r, c):
        iss = self.ctx.analysis.issues
        if not (0 <= r < len(iss)):
            return
        x = iss[r]
        if x.uids:
            self.ctx.store.select_cams(x.uids)
            self.ctx.window.frame_selection()
        elif x.where is not None:
            self.ctx.window.frame_point(x.where)


# ---------------------------------------------------------------------------------------- history / log
class HistoryTab(QListWidget):
    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        self.stack = ctx.store.undo
        self.stack.indexChanged.connect(lambda *_: self.refresh())
        self.itemClicked.connect(self._jump)
        self.setUniformItemSizes(True)
        self.refresh()

    def refresh(self):
        self.clear()
        st = self.stack
        n = st.count()
        for i in range(n, -1, -1):
            if i == 0:
                text = 'Opened project'
                t = ''
            else:
                cmd = st.command(i - 1)
                text = cmd.text()
                t = time.strftime('%H:%M:%S', time.localtime(getattr(cmd, 't', time.time())))
            it = QListWidgetItem(f'{t:>8}   {text}' if t else f'{"":>8}   {text}')
            it.setFont(mono(11) if False else sans(12))
            it.setData(Qt.ItemDataRole.UserRole, i)
            if i > st.index():
                it.setForeground(THEME.color('muted'))
            if i == st.index():
                it.setForeground(THEME.color('ink'))
                f = sans(12, QFont.Weight.DemiBold)
                it.setFont(f)
            self.addItem(it)

    def _jump(self, it):
        self.stack.setIndex(int(it.data(Qt.ItemDataRole.UserRole)))
        self.ctx.analysis.request()


class LogTab(QPlainTextEdit):
    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        self.setReadOnly(True)
        self.setFont(mono(11))
        self._n = 0
        t = QTimer(self)
        t.setInterval(700)
        t.timeout.connect(self.refresh)
        t.start()
        ctx.store.project_replaced.connect(lambda: (setattr(self, '_n', 0), self.refresh()))

    def refresh(self):
        log = self.ctx.store.project.log
        if len(log) == self._n:
            return
        self._n = len(log)
        lines = []
        for e in reversed(log[-300:]):
            ms = int((e.t % 1) * 1000)
            lines.append(f'{time.strftime("%H:%M:%S", time.localtime(e.t))}.{ms:03d}  {e.kind:<10} {e.text}')
        self.setPlainText('\n'.join(lines))


# ---------------------------------------------------------------------------------------- container
class BottomPanel(QWidget):
    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        self.setObjectName('base')
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        bar = QWidget()
        bar.setObjectName('base')
        bl = QHBoxLayout(bar)
        bl.setContentsMargins(0, 0, 4, 0)
        bl.setSpacing(0)
        self.tabs = QTabBar()
        self.tabs.setExpanding(False)
        self.tabs.setDrawBase(False)
        self.tabs.setIconSize(QSize(12, 12))
        bl.addWidget(self.tabs)
        bl.addStretch(1)
        self.more = tool_button('more', 'Panel options', size=20, icon_size=12)
        bl.addWidget(self.more)
        lay.addWidget(bar)
        self.stack = QStackedWidget()
        lay.addWidget(self.stack, 1)
        self.coverage = CoverageTab(ctx)
        self.cameras = CamerasTab(ctx)
        self.tolerance = ToleranceTab(ctx)
        self.issues = IssuesTab(ctx)
        self.history = HistoryTab(ctx)
        self.log = LogTab(ctx)
        self.keys = []
        for key, title, w in (('coverage', 'Coverage', self.coverage), ('cameras', 'Cameras', self.cameras),
                              ('tolerance', 'Tolerance', self.tolerance), ('issues', 'Issues', self.issues),
                              ('history', 'History', self.history), ('log', 'Log', self.log)):
            self.tabs.addTab(title)
            self.stack.addWidget(w)
            self.keys.append(key)
        self.tabs.currentChanged.connect(self.stack.setCurrentIndex)
        self.issues.countChanged.connect(self._issues_count)
        m = QMenu(self.more)
        m.addAction('Copy coverage summary', self._copy_summary)
        m.addAction('Copy camera table', self.cameras.copy)
        self.more.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.more.setMenu(m)

    def _issues_count(self, n, worst):
        i = self.keys.index('issues')
        self.tabs.setTabText(i, f'Issues {n}' if n else 'Issues')
        self.tabs.setTabIcon(i, icons.status_icon(worst, 12) if worst else icons.icon('good', 12, '#00000000'))

    def show_tab(self, key):
        if key in self.keys:
            self.tabs.setCurrentIndex(self.keys.index(key))

    def _copy_summary(self):
        res = self.ctx.analysis.result
        if res is None:
            return
        m = res.metrics
        txt = (f'Blind in ROI {m["blind_m2"]:.3f} m²\nCovered {m["covered_pct"]:.1f} %\nSeen by ≥ 2 cameras {m["multi_pct"]:.1f} %\n'
               f'Minimum density {m["min_density"]:.1f} px/cm\nIn focus {m["in_focus_pct"]:.1f} %\nROI area {m["roi_area"]:.3f} m²')
        QGuiApplication.clipboard().setText(txt)
        self.ctx.msg('Copied the coverage summary')
