"""Analysis ▸ Frame: the T-slot aluminium frame for the roof rig (core.framing). Left: the profiles, the joints, the
plates and the cut settings. Centre: the frame in 3D with the sensors it carries (drag to orbit, wheel to zoom,
double-click to reset) and its totals. Right: the cut list, the stock bars, the bill of materials, each device's
mount and the findings. Cheap: worked out again on every edit while the page shows. "Add to rig" puts the frame into
the rig as parts (one undo step), so it occludes the cameras like the rig it replaces; Export writes the cut list,
the BOM (CSV and PDF), the DXF plates, an STL per part and the STEP assembly."""
from __future__ import annotations

import math

import numpy as np
from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtGui import QColor, QImage, QPainter
from PyQt6.QtWidgets import (QAbstractItemView, QComboBox, QFileDialog, QGridLayout, QHBoxLayout, QHeaderView,
                             QScrollArea, QSplitter, QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget)

from ...core import framing as FR
from ...core import meshing as M
from ..theme import THEME, mono, sans
from ..widgets import ROW_H, ScrubSpinBox, SegmentedControl, StatusLine, ToggleSwitch, button, fit_width, hline, label

HOME = (-130.0, 24.0)
# key, label, unit, decimals, step, tip
FIELDS = (
    ('plate_mm', 'Plate', 'mm', 1, 0.5, 'Thickness of every mount and top plate (6061-T6)'),
    ('plate_margin_mm', 'Plate margin', 'mm', 1, 1.0, 'How far a plate reaches past the device\'s footprint'),
    ('min_plate_mm', 'Smallest plate', 'mm', 0, 5.0, 'The smallest plate side: room for two bolts into the slot'),
    ('rail_inset_mm', 'Rail inset', 'mm', 0, 5.0, 'Rails this far in from the platform\'s edges (or the feet)'),
    ('kerf_mm', 'Saw kerf', 'mm', 1, 0.5, 'Material lost per cut when the stock bars are planned'),
    ('stock_m', 'Stock bar', 'm', 2, 0.5, 'Length of the stock bars (0 = the profile\'s own stock length)'),
)


def _rgb(c: QColor):
    return (c.red(), c.green(), c.blue())


class FrameView(QWidget):
    """A software-rendered orbit view of mesh layers (core.meshing.render): [(V, F, rgb, alpha)] from layers_fn().
    Drag to orbit, wheel to zoom, double-click to reset."""

    def __init__(self, layers_fn=None, message='', parent=None):
        super().__init__(parent)
        self.setMinimumSize(260, 220)
        self.layers_fn = layers_fn or (lambda: [])
        self.message = message
        self.overlay = ''
        self.yaw, self.elev = HOME
        self.zoom = 1.0
        self._img = None
        self._key = None
        self._drag = None
        self.revision = 0
        THEME.changed.connect(self._retheme)

    def _retheme(self, *_):
        self._key = None
        self.update()

    def changed(self):
        self.revision += 1
        self._key = None
        self.update()

    def reset_view(self):
        self.yaw, self.elev = HOME
        self.zoom = 1.0
        self.changed()

    def render_image(self, w, h):
        layers = [L for L in self.layers_fn() if len(L[1])]
        if not layers:
            return None
        allv = np.vstack([L[0] for L in layers])
        c = (allv.min(axis=0) + allv.max(axis=0)) / 2
        rad = float(np.linalg.norm(allv - c, axis=1).max()) or 1.0
        bg = _rgb(THEME.color('viewport'))
        img = M.render(layers, M.view_basis(self.yaw, self.elev), (w, h), 0.45 * min(w, h) / rad * self.zoom, c,
                       bg=bg, ss=1)
        img = np.ascontiguousarray(img)
        return QImage(img.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()

    def paintEvent(self, e):
        p = QPainter(self)
        W, H = self.width(), self.height()
        p.fillRect(self.rect(), THEME.color('viewport'))
        key = (W, H, round(self.yaw, 2), round(self.elev, 2), round(self.zoom, 3), self.revision)
        if key != self._key:
            try:
                self._img = self.render_image(W, H)
            except Exception:                       # a degenerate mesh: show the message, never crash the paint
                self._img = None
            self._key = key
        if self._img is None:
            p.setFont(sans(12))
            p.setPen(THEME.color('muted'))
            p.drawText(QRectF(20, 0, W - 40, H), Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap, self.message)
        else:
            p.drawImage(0, 0, self._img)
            p.setFont(sans(11))
            p.setPen(THEME.color('muted'))
            p.drawText(QRectF(10, H - 20, W - 20, 16), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                       'Drag to orbit · wheel to zoom · double-click to reset')
            if self.overlay:
                p.setFont(mono(11))
                p.setPen(THEME.color('ink2'))
                p.drawText(QRectF(10, 6, W - 20, 16), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                           self.overlay)
        p.end()

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self._drag = (e.position(), self.yaw, self.elev)

    def mouseMoveEvent(self, e):
        if self._drag is None:
            return
        pos, yaw, elev = self._drag
        d = e.position() - pos
        self.yaw = yaw - d.x() * 0.5
        self.elev = max(-89.0, min(89.0, elev + d.y() * 0.5))
        self.update()

    def mouseReleaseEvent(self, e):
        self._drag = None

    def wheelEvent(self, e):
        step = e.angleDelta().y() / 120.0
        self.zoom = max(0.3, min(8.0, self.zoom * math.pow(1.15, step)))
        self.update()

    def mouseDoubleClickEvent(self, e):
        self.reset_view()


def table(cols, stretch=0):
    t = QTableWidget(0, len(cols))
    t.setHorizontalHeaderLabels([c[0] for c in cols])
    for i, (_, tip) in enumerate(cols):
        t.horizontalHeaderItem(i).setToolTip(tip)
    t.verticalHeader().setVisible(False)
    t.verticalHeader().setDefaultSectionSize(ROW_H)
    t.horizontalHeader().setFixedHeight(ROW_H)
    t.setShowGrid(False)
    t.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    t.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    hh = t.horizontalHeader()
    for c in range(len(cols)):
        hh.setSectionResizeMode(c, QHeaderView.ResizeMode.ResizeToContents)
    hh.setSectionResizeMode(stretch, QHeaderView.ResizeMode.Stretch)
    return t


def fill(t: QTableWidget, rows, numeric=()):
    """Put rows (lists of text) into the table; columns in numeric right-aligned in mono."""
    t.setRowCount(len(rows))
    for r, row in enumerate(rows):
        for c, val in enumerate(row):
            it = QTableWidgetItem('' if val is None else str(val))
            if c in numeric:
                it.setFont(mono(12))
                it.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            t.setItem(r, c, it)


def findings_box():
    host = QWidget()
    lay = QVBoxLayout(host)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(4)
    lay.addStretch(1)
    sc = QScrollArea()
    sc.setWidgetResizable(True)
    sc.setWidget(host)
    return sc, lay


def set_findings(lay, findings):
    while lay.count() > 1:
        w = lay.takeAt(0).widget()
        if w is not None:
            w.hide()
            w.deleteLater()
    for i, f in enumerate(findings):
        sev = f['severity'] if f['severity'] in ('good', 'warning', 'serious', 'critical') else None
        lay.insertWidget(i, StatusLine(sev, f['message']))


class FramingPage(QWidget):
    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        self.result: FR.FramingResult | None = None
        self._dirty = True
        self._layers = []
        self.setObjectName('base')
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        split = QSplitter(Qt.Orientation.Horizontal)
        split.setHandleWidth(2)
        split.addWidget(self._left())
        # centre: the view and the totals
        cw = QWidget()
        cw.setObjectName('panel')
        cl = QVBoxLayout(cw)
        cl.setContentsMargins(8, 6, 8, 8)
        self.view = FrameView(lambda: self._layers, 'Nothing on the roof to frame: fit a roof rack or a mast '
                                                    '(Rig) or place a sensor above the roof.')
        cl.addWidget(self.view, 1)
        self.totals = label('', ink2=True, wrap=True)
        cl.addWidget(self.totals)
        split.addWidget(cw)
        # right: tables and findings
        rw = QWidget()
        rw.setObjectName('panel')
        rl = QVBoxLayout(rw)
        rl.setContentsMargins(8, 6, 8, 8)
        self.tabs = QTabWidget()
        self.t_cut = table([('Item', 'Cut list item'), ('Profile', 'Catalogue profile'), ('Length mm', 'Cut length'),
                            ('Ends', 'End machining'), ('Qty', 'How many'), ('kg each', 'Mass of one cut'),
                            ('Members', 'Which members')], 3)
        self.t_stock = table([('Profile', ''), ('Stock mm', 'Stock bar length'), ('Bars', 'Stock bars to buy'),
                              ('Offcut %', 'Share of the bought length left over'),
                              ('Cuts per bar', 'mm, in the order to cut them')], 4)
        self.t_bom = table([('Category', ''), ('Item', ''), ('Spec', ''), ('Qty', ''), ('kg', 'Total mass')], 2)
        self.t_mount = table([('Device', ''), ('Kind', ''), ('Mount', 'How it is carried'),
                              ('Plate', 'Its plate (DXF plate_<id>.dxf)'), ('On', 'The member under it'),
                              ('Gap mm', 'Its base above its plate (negative: it sinks into the frame)')], 2)
        self.tabs.addTab(self.t_cut, 'Cut list')
        self.tabs.addTab(self.t_stock, 'Stock')
        self.tabs.addTab(self.t_bom, 'BOM')
        self.tabs.addTab(self.t_mount, 'Mounts')
        fsc, self.find_lay = findings_box()
        self.tabs.addTab(fsc, 'Findings')
        rl.addWidget(self.tabs, 1)
        split.addWidget(rw)
        split.setSizes([300, 640, 660])
        lay.addWidget(split)
        ctx.store.changed.connect(self._on_change)
        ctx.store.project_replaced.connect(self._on_change)
        THEME.changed.connect(self._on_theme)

    # ------------------------------------------------------------------ settings
    def _left(self):
        w = QWidget()
        w.setObjectName('panel')
        ll = QVBoxLayout(w)
        ll.setContentsMargins(8, 6, 8, 8)
        ll.setSpacing(4)
        ll.addWidget(label('Frame'))
        ll.addWidget(hline())
        ll.addWidget(label('A T-slot aluminium frame for the rig on the roof: members, plates under every sensor, '
                           'fixings and feet, with a cut list, a BOM, DXF plates, STL parts and a STEP assembly.',
                           muted=True, small=True, wrap=True))
        ll.addSpacing(6)
        ll.addWidget(label('Profiles', bold=True))
        self.combos = {}
        g = QGridLayout()
        for i, (key, text, tip) in enumerate((('main_profile', 'Members', 'Crossbars, rails, cross members, arms'),
                                              ('post_profile', 'Posts', 'Masts and standoffs'))):
            c = QComboBox()
            c.setToolTip(tip)
            for pid, pr in FR.PROFILES.items():
                c.addItem(f'{pr.label} ({pr.maker} {pr.part})', pid)
            c.activated.connect(lambda idx, k=key, c=c: self._set(f'{k.split("_")[0]} profile {c.itemData(idx)}',
                                                                  **{k: c.itemData(idx)}))
            self.combos[key] = c
            g.addWidget(label(text, ink2=True), i, 0)
            g.addWidget(c, i, 1)
        ll.addLayout(g)
        ll.addSpacing(6)
        ll.addWidget(label('Joints and feet', bold=True))
        self.seg_joint = SegmentedControl([(k, v) for k, v in FR.JOINT_LABEL.items()], 'brackets')
        self.seg_joint.setToolTip('Corner brackets (square cuts) or anchor fasteners (counterbores, access holes, '
                                  'tapped post ends)')
        self.seg_joint.changed.connect(lambda k: self._set(f'joints: {FR.JOINT_LABEL[k].lower()}', joint=k))
        ll.addWidget(self.seg_joint)
        self.seg_feet = SegmentedControl([(k, v) for k, v in FR.FEET_LABEL.items()], 'rails')
        self.seg_feet.changed.connect(lambda k: self._set(f'feet: {FR.FEET_LABEL[k].lower()}', feet=k))
        ll.addWidget(self.seg_feet)
        self.t_cams = ToggleSwitch('Plates for cameras on the rig')
        self.t_cams.setToolTip('Cameras above the roof get a plate too (their printed bracket sets the angle)')
        self.t_cams.clicked.connect(lambda on: self._set('camera plates ' + ('on' if on else 'off'),
                                                         frame_cameras=bool(on)))
        ll.addWidget(self.t_cams)
        ll.addSpacing(6)
        ll.addWidget(label('Plates and cuts', bold=True))
        grid = QGridLayout()
        self.fields = {}
        for i, (key, text, unit, dec, step, tip) in enumerate(FIELDS):
            f = ScrubSpinBox('', 'plain', dec, unit, step=step, width=100)
            f.setToolTip(tip)
            f.committed.connect(lambda v, k=key, t=text, u=unit: self._set(f'{t.lower()} {v:g} {u}', **{k: v}))
            a = label(text, ink2=True)
            a.setToolTip(tip)
            grid.addWidget(a, i, 0)
            grid.addWidget(f, i, 1)
            self.fields[key] = f
        ll.addLayout(grid)
        ll.addStretch(1)
        self.status = label('', muted=True, small=True, wrap=True)
        ll.addWidget(self.status)
        bl = QHBoxLayout()
        b = button('Defaults', 'ghost', None, 'Back to the default frame settings')
        b.clicked.connect(self._defaults)
        bl.addWidget(b)
        self.b_apply = button('Add to rig', None, 'frame', 'Put the frame into the rig as parts (one undo step): the '
                                                           'parts it stands for are hidden, the frame occludes and '
                                                           'carries like them')
        self.b_apply.clicked.connect(self.apply_to_rig)
        bl.addWidget(self.b_apply)
        bl.addStretch(1)
        self.b_export = button('Export…', None, 'export', 'Cut list, BOM (CSV, PDF), DXF plates, STL parts, STEP '
                                                          'assembly, JSON')
        self.b_export.clicked.connect(lambda: self.export())
        bl.addWidget(self.b_export)
        ll.addLayout(bl)
        sc = QScrollArea()
        sc.setWidgetResizable(True)
        sc.setWidget(w)
        fit_width(sc)
        sc.setMinimumWidth(290)
        return sc

    @property
    def settings(self) -> FR.FramingSettings:
        return FR.settings_of(self.ctx.store.project)

    def _set(self, text, **kw) -> bool:
        st = self.settings
        for k, v in kw.items():
            setattr(st, k, v)
        new = FR.FramingSettings.from_dict(st.to_dict()).to_dict()
        if new == self.settings.to_dict():
            self._sync()
            return False

        def fn(p):
            p.analysis.framing = new
        self.ctx.store.edit('analysis', fn, f'Frame · {text}')
        return True

    def _defaults(self):
        rep = self.settings.replaced
        if self.ctx.store.project.analysis.framing != FR.FramingSettings(replaced=rep).to_dict():
            def fn(p):
                p.analysis.framing = FR.FramingSettings(replaced=rep).to_dict() if rep else {}
            self.ctx.store.edit('analysis', fn, 'Frame · defaults')

    def _sync(self):
        st = self.settings
        for k, c in self.combos.items():
            i = c.findData(getattr(st, k))
            if i >= 0:
                c.setCurrentIndex(i)
        for k, f in self.fields.items():
            f.set_value(float(getattr(st, k)))
        self.seg_joint.set_value(st.joint)
        self.seg_feet.set_value(st.feet)
        self.t_cams.blockSignals(True)
        self.t_cams.setChecked(st.frame_cameras)
        self.t_cams.blockSignals(False)

    # ------------------------------------------------------------------ state
    def _on_change(self, *_):
        if self.isVisible():
            self.refresh()
        else:
            self._dirty = True

    def _on_theme(self, *_):
        if self.result is not None:
            self._build_layers()

    def showEvent(self, e):
        super().showEvent(e)
        if self._dirty:
            self.refresh()

    def activate(self):
        if self._dirty:
            self.refresh()

    def refresh(self):
        p = self.ctx.store.project
        self._dirty = False
        self._sync()
        try:
            res = self.result = FR.synthesise(p)
        except Exception as e:                           # a broken input: say so, keep the page alive
            self.result = None
            self.status.setText(f'Failed: {type(e).__name__}: {e}')
            self._layers = []
            self.view.changed()
            return
        self._build_layers()
        m = res.mass
        self.totals.setText(f'{len(res.members)} members · {m["length_m"]:.2f} m of profile · {len(res.plates)} '
                            f'plates · {len(res.feet)} feet · {m["frame_kg"]:.1f} kg '
                            f'(members {m["members_kg"]:.1f}, plates {m["plates_kg"]:.1f}, fixings '
                            f'{m["fixings_kg"]:.1f}, feet {m["feet_kg"]:.1f})' if res.members else
                            'No frame.')
        fill(self.t_cut, [[r['item'], r['label'], f'{r["length_mm"]:.1f}',
                           r['ends'] + (f'; {r["access_holes"]} access holes' if r['access_holes'] else ''),
                           r['quantity'], f'{r["kg_each"]:.3f}', ' '.join(r['members'])] for r in res.cut_list],
             numeric=(2, 4, 5))
        fill(self.t_stock, [[s['label'], f'{s["stock_mm"]:.0f}', s['bars'], f'{s["waste_pct"]:g}',
                             ' | '.join(' + '.join(f'{c:g}' for c in b) for b in s['cuts_mm'])] for s in res.stock],
             numeric=(1, 2, 3))
        fill(self.t_bom, [[r['category'], r['item'], r['spec'], f'{r["quantity"]:g} {r["unit"]}', f'{r["kg"]:.3f}']
                          for r in res.bom], numeric=(3, 4))
        fill(self.t_mount, [[mt['tag'], mt['kind'], mt['status'], mt['plate'], mt['member'],
                             '' if mt['gap_mm'] is None else f'{mt["gap_mm"]:.1f}'] for mt in res.mounts],
             numeric=(5,))
        set_findings(self.find_lay, res.findings)
        bad = sum(f['severity'] in ('critical', 'serious', 'warning') for f in res.findings)
        self.status.setText(f'{res.variant}: ' + (f'{bad} finding{"s" * (bad != 1)} to look at' if bad else
                                                  'nothing to flag'))
        self.b_export.setEnabled(bool(res.members))
        self.b_apply.setEnabled(bool(res.members))

    def _build_layers(self):
        res = self.result
        layers = []
        dark = THEME.color('viewport').lightness() < 128
        if res is not None:
            for s in FR.solids(res):
                V, F = s.mesh()
                c = tuple(int(255 * x) for x in s.colour)
                layers.append((V, F, c if not dark else tuple(min(255, int(x * 1.05)) for x in c), 1.0))
            if res.members:
                for V, F, uid in FR.device_boxes(self.ctx.store.project, res):
                    layers.append((V, F, (10, 165, 184), 1.0))
        self._layers = layers
        self.view.overlay = (f'{res.variant} · {len(res.members)} members · {res.mass.get("frame_kg", 0):.1f} kg'
                             if res is not None and res.members else '')
        self.view.changed()

    # ------------------------------------------------------------------ actions
    def apply_to_rig(self) -> bool:
        res = self.result
        if res is None or not res.members:
            return False
        self.ctx.store.edit(['rig', 'analysis'], lambda q: FR.apply_to_rig(q, FR.synthesise(q)),
                            f'Frame · added to the rig ({len(res.members)} members, {len(res.plates)} plates)')
        self.ctx.msg('The frame is in the rig: Undo takes it out again', 5000)
        return True

    def export(self, folder=None):
        res = self.result
        if res is None:
            return []
        if folder is None:
            folder = QFileDialog.getExistingDirectory(self, 'Export the frame into')
            if not folder:
                return []
        from ...core import provenance as PV
        paths = PV.stamp_paths(FR.export(res, self.ctx.store.project, folder), self.ctx.store.project, folder)
        self.ctx.msg(f'Exported the frame to {folder}', 5000)
        return paths
