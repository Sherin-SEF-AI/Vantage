"""Inspector: context-sensitive property pages."""
from __future__ import annotations

import math

import numpy as np
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QComboBox, QStackedWidget,
                             QScrollArea, QFrame, QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView,
                             QCheckBox, QSlider, QMenu, QSizePolicy, QPushButton, QToolButton)

from ..core.analysis import surface_frame
from ..core.lens import resolve, derived, model_fov, PROJECTIONS, PROJECTION_LABEL
from ..core.mathutil import parse_number
from ..core.project import FRAME_PRESETS, Obstacle, Target
from ..core.rules import route_length, route_text, EDGE_RULE_MM, camera_gbps, link_budget
from . import icons, ops
from .store import mirror_props
from .theme import THEME, mono, sans, STATUS
from .widgets import (Section, ScrubSpinBox, ToggleSwitch, SegmentedControl, Chip, ValueLabel, StatusLine, label,
                      button, tool_button, hline, ImageCircleDiagram, ROW_H, LABEL_W)

SYNC_LINES = list('ABCDEFGH')        # hardware trigger lines a camera can be put on
BRACKETS = ['Flat plate · 25 mm', 'Mirror underside clip', 'Quarter-panel pod · 40 mm', 'Boot-lid wedge 10°', 'None']


def _table(cols, stretch=1):
    t = QTableWidget(0, len(cols))
    t.setHorizontalHeaderLabels(cols)
    t.verticalHeader().setVisible(False)
    t.verticalHeader().setDefaultSectionSize(ROW_H)
    t.horizontalHeader().setFixedHeight(ROW_H)
    t.setShowGrid(False)
    t.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    t.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    t.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
    t.horizontalHeader().setSectionResizeMode(stretch, QHeaderView.ResizeMode.Stretch)
    t.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    t.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    t.setFocusPolicy(Qt.FocusPolicy.NoFocus)
    return t


def _fit_table(t):
    t.setFixedHeight(ROW_H * (t.rowCount() + 1) + 2)


def _cell(text, mono_=False, right=False, muted=False, color=None):
    it = QTableWidgetItem(text)
    if mono_:
        it.setFont(mono(12))
    if right:
        it.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    if muted:
        it.setForeground(THEME.color('muted'))
    if color is not None:
        it.setForeground(color)
    return it


class Page(QWidget):
    title = ''

    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        self.lay = QVBoxLayout(self)
        self.lay.setContentsMargins(0, 0, 0, 8)
        self.lay.setSpacing(0)

    def refresh(self):
        pass

    @property
    def project(self):
        return self.ctx.store.project


# ------------------------------------------------------------------------------------------------ camera
class CameraPage(Page):
    title = 'Camera'

    def __init__(self, ctx):
        super().__init__(ctx)
        self._busy = False
        head = QWidget()
        head.setFixedHeight(30)
        h = QHBoxLayout(head)
        h.setContentsMargins(8, 4, 6, 4)
        h.setSpacing(4)
        self.chip = Chip()
        self.name = QLineEdit()
        self.name.editingFinished.connect(self._rename)
        self.b_link = tool_button('link', 'Link L/R pair (mirrored edits)', checkable=True)
        self.b_link.clicked.connect(lambda: ops.toggle_link(ctx, ctx.store.selected.uid) if ctx.store.selected else None)
        self.b_lock = tool_button('lock', 'Lock transform', checkable=True)
        self.b_lock.clicked.connect(lambda: ops.set_lock(ctx, list(ctx.store.sel_cams)))
        self.b_more = tool_button('more', 'More')
        self.b_more.clicked.connect(lambda: ctx.window.camera_menu(ctx.store.sel_cams, self.b_more.mapToGlobal(self.b_more.rect().bottomLeft())))
        h.addWidget(self.chip)
        h.addWidget(self.name, 1)
        h.addWidget(self.b_link)
        h.addWidget(self.b_lock)
        h.addWidget(self.b_more)
        self.lay.addWidget(head)

        # Transform
        self.s_tr = Section('Transform')
        self.tr_link = QLabel()
        self.s_tr.header.add_right(self.tr_link)
        self.px, self.py, self.pz = (ScrubSpinBox(a, 'length', 3) for a in ('X', 'Y', 'Z'))
        self.ry, self.rp, self.rr = (ScrubSpinBox(a, 'angle', 1) for a in ('ψ', 'θ', 'φ'))
        self.row_pos = self.s_tr.row('Position m', self.px, self.py, self.pz)
        self.s_tr.row('Rotation °', self.ry, self.rp, self.rr)
        for w, k in ((self.px, 'x'), (self.py, 'y'), (self.pz, 'z'), (self.ry, 'yaw'), (self.rp, 'pitch'), (self.rr, 'roll')):
            w.committed.connect(lambda v, k=k: self._set(k, v))
            w.scrubbed.connect(lambda v, k=k: self._set(k, v, merge=True))
            w.scrub_started.connect(lambda: setattr(ctx.store, 'dragging', True))
            w.scrub_finished.connect(self._scrub_done)
        self.frame = QComboBox()
        for k, v in FRAME_PRESETS.items():
            self.frame.addItem(v, k)
        self.frame.activated.connect(self._frame)
        self.s_tr.row('Frame', self.frame)
        self.mirror_note = label('', muted=True, small=True, wrap=True)
        self.mirror_note.setContentsMargins(LABEL_W + 2, 0, 0, 0)
        self.s_tr.add(self.mirror_note)
        self.lay.addWidget(self.s_tr)
        self.lay.addWidget(hline())

        # Mount
        self.s_mount = Section('Mount')
        self.mount_icon = QLabel()
        self.s_mount.header.add_right(self.mount_icon)
        self.surface = QComboBox()
        self.surface.activated.connect(self._surface)
        self.s_mount.row('Surface', self.surface)
        self.snap_n = ToggleSwitch()
        self.snap_n.toggled.connect(lambda on: self._set('snap_normal', on) if not self._busy else None)
        self.s_mount.row('Snap to normal', self.snap_n, QWidget())
        self.standoff = ScrubSpinBox('', 'plain', 0, 'mm', step=1)
        self.standoff.setFixedWidth(100)
        self.standoff.committed.connect(lambda v: self._set('standoff_mm', max(0.0, v)))
        self.s_mount.row('Standoff', self.standoff, QWidget())
        self.bracket = QComboBox()
        self.bracket.addItems(BRACKETS)
        self.bracket.activated.connect(lambda i: self._set('bracket', self.bracket.currentText()))
        self.s_mount.row('Bracket', self.bracket)
        self.height_v = ValueLabel()
        self.s_mount.row('Height', self.height_v)
        self.edge_v = ValueLabel()
        self.s_mount.row('Body edge', self.edge_v)
        self.house_v = ValueLabel()
        self.s_mount.row('Housings', self.house_v, tip='Housings in view: the share of this camera\'s image each other '
                                                       'sensor\'s housing or camera module covers (the body and rig '
                                                       'in front of one hide it)')
        self.float_box = QWidget()
        fb = QVBoxLayout(self.float_box)
        fb.setContentsMargins(0, 4, 0, 2)
        fb.setSpacing(4)
        self.float_msg = StatusLine('critical', '')
        fb.addWidget(self.float_msg)
        bl = QHBoxLayout()
        self.b_snap = button('Snap to surface', 'primary')
        self.b_snap.clicked.connect(lambda: ops.snap_to_surface(ctx, list(ctx.store.sel_cams)))
        self.b_keep = button('Keep as bracket')
        self.b_keep.clicked.connect(lambda: ops.snap_to_surface(ctx, list(ctx.store.sel_cams), keep_as_bracket=True))
        bl.addWidget(self.b_snap)
        bl.addWidget(self.b_keep)
        bl.addStretch(1)
        fb.addLayout(bl)
        self.s_mount.add(self.float_box)
        self.lay.addWidget(self.s_mount)
        self.lay.addWidget(hline())

        # Optics
        self.s_opt = Section('Optics', expanded=False)
        self.module = QComboBox()
        self.module.activated.connect(self._module)
        self.lens = QComboBox()
        self.lens.activated.connect(self._lens)
        self.proj = QComboBox()
        for k in PROJECTIONS:
            self.proj.addItem(PROJECTION_LABEL[k] if k != 'kannala_brandt' else 'Measured (Kannala–Brandt)', k)
        self.proj.activated.connect(self._projection)
        self.s_opt.row('Module', self.module)
        self.s_opt.row('Lens', self.lens)
        self.s_opt.row('Projection', self.proj)
        self.focal_v = ValueLabel()
        self.s_opt.row('Focal · aperture', self.focal_v)
        self.fov_v = ValueLabel()
        self.s_opt.row('FOV H · V · D', self.fov_v)
        self.focus_v = ValueLabel()
        self.s_opt.row('Focus', self.focus_v)
        self.calib_v = ValueLabel()
        self.calib_v.setToolTip('A unit calibration (Calibrate ▸ Lens calibration) replaces the library lens for this camera')
        self.s_opt.row('Unit calibration', self.calib_v)
        ic = QWidget()
        icl = QHBoxLayout(ic)
        icl.setContentsMargins(0, 0, 0, 0)
        icl.setSpacing(8)
        lab = label('Image circle', ink2=True)
        lab.setFixedWidth(LABEL_W)
        self.circle = ImageCircleDiagram()
        self.circle.setFixedSize(76, 66)
        self.circle_txt = label('', muted=True, mono_=True, small=True, wrap=True)
        icl.addWidget(lab)
        icl.addWidget(self.circle)
        icl.addWidget(self.circle_txt, 1)
        ic.setFixedHeight(70)
        self.s_opt.add(ic)
        self.b_edit_mod = button('Edit module…')
        self.b_edit_mod.clicked.connect(lambda: ctx.window.edit_module(ctx.store.selected.module_id, ctx.store.selected.lens_id))
        self.b_calib = button('Calibrate…', None, 'lens', 'Measure this camera\'s own lens from checkerboard views')
        self.b_calib.clicked.connect(lambda: ctx.window.lens_calibration(ctx.store.selected.uid if ctx.store.selected else None))
        row = QHBoxLayout()
        row.addSpacing(LABEL_W + 2)
        row.addWidget(self.b_edit_mod)
        row.addWidget(self.b_calib)
        row.addStretch(1)
        self.s_opt.blay.addLayout(row)
        self.lay.addWidget(self.s_opt)
        self.lay.addWidget(hline())

        # Sensor
        self.s_sen = Section('Sensor', expanded=False)
        self.sen_vals = {}
        for k in ('Sensor', 'Resolution', 'Pixel pitch', 'Optical format', 'Interface', 'Frame rate'):
            v = ValueLabel()
            self.sen_vals[k] = v
            self.s_sen.row(k, v)
        self.lay.addWidget(self.s_sen)
        self.lay.addWidget(hline())

        # Link
        self.s_link = Section('Link', expanded=False)
        self.link_icon = QLabel()
        self.s_link.header.add_right(self.link_icon)
        self.port = QComboBox()
        self.port.activated.connect(self._port)
        self.port_note = label('', muted=True, small=True)
        self.s_link.row('GMSL port', self.port, self.port_note)
        self.sync = QComboBox()
        self.sync.addItems([f'Line {x}' for x in SYNC_LINES] + ['free-running'])
        self.sync.activated.connect(lambda i: self._set('sync', (SYNC_LINES + [''])[i]))
        self.s_link.row('Trigger line', self.sync, tip='Cameras on one hardware trigger line fire together; '
                                                         'Analysis ▸ Sync and timing… suggests the lines.')
        self.cable = QComboBox()
        self.cable.activated.connect(lambda i: self._set('cable_m', float(self.cable.currentData())))
        self.s_link.row('Cable', self.cable)
        self.routed_v = ValueLabel()
        self.s_link.row('Routed', self.routed_v)
        self.route_txt = label('', ink2=True, small=True, wrap=True)
        self.s_link.add(self.route_txt)
        bl = QHBoxLayout()
        self.b_cable = button('Use 15 m cable')
        self.b_cable.clicked.connect(self._longer_cable)
        self.b_route = button('Edit route…')
        self.b_route.clicked.connect(lambda: ctx.window.edit_route(ctx.store.selected.uid))
        bl.addWidget(self.b_cable)
        bl.addWidget(self.b_route)
        bl.addStretch(1)
        self.s_link.blay.addLayout(bl)
        self.lay.addWidget(self.s_link)
        self.lay.addWidget(hline())

        # This camera
        self.s_me = Section('This camera')
        self.me_icon = QLabel()
        self.s_me.header.add_right(self.me_icon)
        self.me_rows = []
        for _ in range(6):
            v = ValueLabel()
            r = self.s_me.row('', v, label_w=128)
            self.me_rows.append((r, v))
        self.lay.addWidget(self.s_me)
        self.lay.addStretch(1)

    # ---------------------------------------------------------------- edits
    def _cams(self):
        return self.ctx.store.selected_cams()

    def _set(self, key, val, merge=False):
        if self._busy:
            return
        st = self.ctx.store
        p = self.project
        if key in ('x', 'y', 'z'):
            unit = 1000.0 if p.units == 'mm' else 1.0
            val = round(val / unit + p.origin_offset()['xyz'.index(key)], 4)
        elif key in ('yaw', 'pitch', 'roll'):
            val = round(val, 2)
        cams = self._cams()
        if len(cams) == 1:
            st.set_cam(cams[0].uid, merge=merge, **{key: val})
        elif cams:
            uids = [c.uid for c in cams]

            def fn(pj):
                done = set()
                for u in uids:
                    c = pj.variant.cam(u)
                    if c is None or u in done:
                        continue           # a linked partner already mirrored from its pair
                    setattr(c, key, val)
                    done.add(u)
                    partner = pj.variant.cam(c.link_uid) if c.link_uid else None
                    if partner is not None:
                        mirror_props(c, partner, [key])
                        done.add(partner.uid)
            # one undo step for all selected cameras (one merge key while scrubbing)
            st.edit('variant', fn, f"{'Moved' if key in ('x', 'y', 'z') else 'Changed'} {', '.join(c.tag for c in cams)}",
                    merge_key=(tuple(uids), key) if merge else None)
        if not merge:
            self.ctx.analysis.request()

    def _scrub_done(self):
        self.ctx.store.dragging = False
        self.ctx.analysis.request()

    def _rename(self):
        c = self.ctx.store.selected
        t = self.name.text().strip()
        if c and t and t != c.name and len(self._cams()) == 1:
            self.ctx.store.edit('variant', lambda p: setattr(p.variant.cam(c.uid), 'name', t), f'Renamed {c.tag} to {t}')

    def _frame(self, i):
        key = self.frame.itemData(i)
        if key == 'custom':
            self.ctx.window.custom_frame()
            return
        self.ctx.store.edit('frame', lambda p: setattr(p, 'frame', key), f'Frame set to {FRAME_PRESETS[key]}')

    def _surface(self, i):
        zi = self.surface.itemData(i)
        vg = self.ctx.analysis.engine.vehicle_geom(self.project)
        if vg is None or zi is None or zi < 0:
            return
        m = vg.occ_mesh
        faces = np.nonzero(m.face_zone == zi)[0]
        if not len(faces):
            return
        for c in self._cams():
            cen = m.centroids[faces]
            k = faces[int(np.argmin(np.linalg.norm(cen - c.pos(), axis=1)))]
            q = m.centroids[k]
            n = m.normals[k]
            p = q + n * c.standoff_mm / 1000
            self.ctx.store.set_cam(c.uid, x=round(float(p[0]), 4), y=round(float(p[1]), 4), z=round(float(p[2]), 4),
                                   text=f'Moved {c.tag} onto {m.zones[zi].name}')

    def _module(self, i):
        mid = self.module.itemData(i)
        lenses = [lid for lid, l in self.project.lenses.items() if not l.module_id or l.module_id == mid]
        if lenses:
            ops.set_lens(self.ctx, list(self.ctx.store.sel_cams), mid, lenses[0])

    def _lens(self, i):
        lid = self.lens.itemData(i)
        c = self.ctx.store.selected
        if c:
            ops.set_lens(self.ctx, list(self.ctx.store.sel_cams), c.module_id, lid)

    def _projection(self, i):
        c = self.ctx.store.selected
        if not c:
            return
        proj = self.proj.itemData(i)
        lid = c.lens_id

        def fn(p):
            p.lenses[lid].projection = proj
        n = sum(1 for x in self.project.variant.cameras if x.lens_id == lid)
        self.ctx.store.edit('library', fn, f'{self.project.lenses[lid].name} projection set to {PROJECTION_LABEL[proj]} ({n} cameras)')

    def _port(self, i):
        c = self.ctx.store.selected
        if c:
            ops.assign_port(self.ctx, c.uid, int(self.port.itemData(i)))

    def _longer_cable(self):
        c = self.ctx.store.selected
        if not c:
            return
        rl = route_length(self.project, c)
        opts = [x for x in self.project.module_for(c).cables_m if x >= rl] or [max(self.project.module_for(c).cables_m)]
        self._set('cable_m', float(opts[0]))

    # ---------------------------------------------------------------- refresh
    def refresh(self):
        cams = self._cams()
        if not cams:
            return
        self._busy = True
        c = cams[0]
        multi = len(cams) > 1
        p = self.project
        st = self.ctx.store
        self.chip.set(c.slot, c.tag if not multi else f'{len(cams)}')
        self.name.setText(c.name if not multi else ', '.join(x.tag for x in cams))
        self.name.setEnabled(not multi)
        self.b_link.setChecked(bool(c.link_uid))
        self.b_link.setIcon(icons.icon('link' if c.link_uid else 'unlink', 16))
        self.b_lock.setChecked(c.locked)
        self.b_lock.setIcon(icons.icon('lock' if c.locked else 'unlock', 16))
        o = p.origin_offset()
        unit = 1000.0 if p.units == 'mm' else 1.0
        dp = 0 if p.units == 'mm' else 3
        self.row_pos.label.setText(f'Position {p.units}')
        for w, k, axis in ((self.px, 'x', 0), (self.py, 'y', 1), (self.pz, 'z', 2)):
            vals = [(getattr(x, k) - o[axis]) * unit for x in cams]
            w.setDecimals(dp)
            w.set_display_unit(p.units)
            w.scrub_step = 1.0 if p.units == 'mm' else 0.001
            w.set_value(vals[0], mixed=len(set(round(v, 6) for v in vals)) > 1)
            w.setEnabled(not c.locked)
        for w, k in ((self.ry, 'yaw'), (self.rp, 'pitch'), (self.rr, 'roll')):
            vals = [getattr(x, k) for x in cams]
            w.set_value(vals[0], mixed=len(set(round(v, 4) for v in vals)) > 1)
            w.setEnabled(not c.locked)
        i = self.frame.findData(p.frame)
        self.frame.setCurrentIndex(max(i, 0))
        partner = st.cam(c.link_uid) if c.link_uid else None
        self.tr_link.setPixmap(icons.pixmap('link', THEME.tok['ink2'], 12) if partner else icons.pixmap('link', '#00000000', 12))
        self.mirror_note.setText(f'Mirrored from {partner.tag} · Y and ψ negated' if partner and not multi else '')
        self.mirror_note.setVisible(bool(partner) and not multi)
        # mount
        mi = self.ctx.analysis.mounts.get(c.uid)
        self.surface.clear()
        vg = self.ctx.analysis.engine.vg
        if vg is not None:
            for zi, z in enumerate(vg.occ_mesh.zones):
                if z.kind == 'allowed':
                    self.surface.addItem(z.name, zi)
        if mi:
            idx = self.surface.findText(mi.surface)
            if idx < 0:
                self.surface.insertItem(0, f'{mi.surface} · auto', -1)
                idx = 0
            self.surface.setCurrentIndex(idx)
        self.snap_n.setChecked(c.snap_normal)
        self.standoff.set_value(c.standoff_mm)
        bi = self.bracket.findText(c.bracket)
        if bi < 0:
            self.bracket.addItem(c.bracket)
            bi = self.bracket.findText(c.bracket)
        self.bracket.setCurrentIndex(bi)
        self.height_v.set(f'{c.z - o[2]:.3f}' if p.units == 'm' else f'{(c.z - o[2]) * 1000:.0f}', p.units)
        sev_mount = None
        if mi:
            if mi.edge_mm < EDGE_RULE_MM:
                self.edge_v.set(f'{mi.edge_mm:.0f}', 'mm', 'warning', f'Warning · min {EDGE_RULE_MM:.0f} mm')
                sev_mount = 'warning'
            else:
                self.edge_v.set(f'{mi.edge_mm:.0f}' if math.isfinite(mi.edge_mm) else '—', 'mm')
            self.float_box.setVisible(bool(mi.floating))
            if mi.floating:
                self.float_msg.set('critical', f'Critical · {c.tag} is {mi.gap_mm:.0f} mm off the body. Snap to surface or move onto a mount zone.')
                sev_mount = 'critical'
            if mi.zone_kind == 'forbidden':
                sev_mount = sev_mount or 'serious'
        else:
            self.edge_v.set('…', '')
            self.float_box.setVisible(False)
        # housings in view (core.housings, the same figures as the Issues list)
        sev_house = None
        if multi:
            self.house_v.set('—')
            self.house_v.setToolTip('')
        else:
            from ..core import housings as HS
            iv = HS.view_shares(p, c, self.ctx.analysis.engine.vehicle_geom(p))
            txt, sev_house = HS.in_view_text(iv)
            self.house_v.set(txt, '', sev_house)
            self.house_v.setToolTip('\n'.join(f"{x['label'][0].upper()}{x['label'][1:]}: {x['share'] * 100:.1f} % "
                                              f"of the image, {x['where']}" for x in iv.shares) or
                                    'No other sensor\'s housing or camera module is in this camera\'s image.')
        order = ('critical', 'serious', 'warning')
        sev_mount = min((x for x in (sev_mount, sev_house) if x), key=order.index, default=None)
        self.mount_icon.setPixmap(icons.pixmap(sev_mount, STATUS[sev_mount], 12) if sev_mount else icons.pixmap('good', '#00000000', 12))
        mods = ', '.join(sorted({x.module_id for x in cams}))
        self.s_mount.header.set_summary(f'{mi.surface if mi else "—"} · {c.standoff_mm:.0f} mm · edge {mi.edge_mm:.0f} mm' if mi else '')
        # optics
        mod, lens = p.module_for(c), p.lens_for(c)
        self.module.clear()
        for mid, m in p.modules.items():
            self.module.addItem(m.name, mid)
        self.module.setCurrentIndex(max(0, self.module.findData(c.module_id)))
        self.lens.clear()
        for lid, l in p.lenses.items():
            if not l.module_id or l.module_id == c.module_id:
                self.lens.addItem(f'{l.name} · {l.label}' if l.label else l.name, lid)
        self.lens.setCurrentIndex(max(0, self.lens.findData(c.lens_id)))
        self.proj.setCurrentIndex(max(0, self.proj.findData(lens.projection)))
        self.proj.setEnabled(not c.calib)
        if c.calib:
            cal = c.calib
            self.calib_v.set(cal.get('serial') or 'unit', '', 'good',
                             f'{cal.get("rms_px", 0):.2f} px · {cal.get("views", 0) or "imported"}'
                             f'{" views" if cal.get("views") else ""} · {cal.get("date", "")}')
        else:
            self.calib_v.set('none', '', None, 'library lens')
        d = derived(mod, lens)
        seen = model_fov(mod, lens, circle=True)          # what the image shows (a fisheye's image circle)
        self.focal_v.set(f'{lens.focal_mm:.2f}', 'mm', None, f'F{lens.f_number:g}' if lens.f_number > 0 else 'F not given')
        self.fov_v.set(f'{seen[0]:.1f}  {seen[1]:.1f}  {seen[2]:.1f}', '°')
        if lens.focus_near_m > 0:
            self.focus_v.set(f'{lens.focus_near_m:.2f}', 'm', None,
                             '– ∞' if math.isinf(lens.focus_far_m) else f'– {lens.focus_far_m:.1f} m')
        else:
            self.focus_v.set('—', '', None, 'not given')
        self.circle.set(mod.sensor_w_mm, mod.sensor_h_mm, d['image_circle_r_mm'], d['max_half_deg'])
        self.circle_txt.setText(f'{mod.sensor_w_mm:.3f} × {mod.sensor_h_mm:.3f} mm\n{mod.optical_format} · {mod.sensor.replace("Sony ", "")}')
        self.s_opt.header.set_summary(f'{lens.name} · {PROJECTION_LABEL[lens.projection].lower()} · {lens.focal_mm:.3g} mm'
                                      if not c.calib else f'{lens.name} · unit calibration · {lens.focal_mm:.3g} mm')
        # sensor
        self.sen_vals['Sensor'].set(mod.sensor)
        self.sen_vals['Resolution'].set(f'{mod.width_px} × {mod.height_px}', 'px')
        self.sen_vals['Pixel pitch'].set(f'{mod.pixel_um:.1f}', 'µm')
        self.sen_vals['Optical format'].set(mod.optical_format)
        self.sen_vals['Interface'].set(mod.interface)
        self.sen_vals['Frame rate'].set(f'{mod.fps:g}', 'fps', None, 'hw sync' if mod.hw_sync else '')
        self.s_sen.header.set_summary(f'{mod.sensor.replace("Sony ", "")} · {mod.width_px} × {mod.height_px} · {mod.pixel_um:g} µm')
        # link
        self.port.clear()
        used = {x.port: x.tag for x in p.variant.cameras if x.uid != c.uid and x.port}
        for k in range(1, p.ecu.ports + 1):
            self.port.addItem(f'{k}' + (f' · {used[k]} (swap)' if k in used else ''), k)
        if c.port:
            self.port.setCurrentIndex(c.port - 1)
        else:
            self.port.setCurrentIndex(-1)
        self.port_note.setText(p.ecu.deserializer(c.port))
        line = (c.sync or '').strip().upper()
        self.sync.setCurrentIndex(SYNC_LINES.index(line) if line in SYNC_LINES else len(SYNC_LINES))
        self.cable.clear()
        for x in mod.cables_m:
            self.cable.addItem(f'{x:.2f} m', x)
        ci = self.cable.findData(c.cable_m)
        if ci < 0:
            self.cable.addItem(f'{c.cable_m:.2f} m', c.cable_m)
            ci = self.cable.count() - 1
        self.cable.setCurrentIndex(ci)
        rl = route_length(p, c)
        over = rl > c.cable_m
        self.routed_v.set(f'{rl:.2f}', 'm', 'serious' if over else 'good',
                          f'Serious · {rl - c.cable_m:.2f} m short' if over else f'{c.cable_m - rl:.2f} m spare')
        self.route_txt.setText('Route: ' + route_text(p, c))
        alt = [x for x in mod.cables_m if x >= rl and x != c.cable_m]
        self.b_cable.setVisible(bool(over and alt))
        if alt:
            self.b_cable.setText(f'Use {alt[0]:g} m cable')
        self.link_icon.setPixmap(icons.pixmap('serious' if over else 'good', STATUS['serious' if over else 'good'], 12))
        sync = (c.sync or '').strip().upper()
        sync = f'line {sync}' if sync in SYNC_LINES else 'free-running'
        self.s_link.header.set_summary(f'port {c.port or "—"} · {sync} · {rl:.2f} / {c.cable_m:.2f} m')
        # this camera
        res = self.ctx.analysis.result
        rows = []
        if res and c.uid in res.per_cam and not multi:
            pc = res.per_cam[c.uid]
            rows.append((f'Seen only by {c.tag}', f'{pc["unique_m2"]:.3f}', 'm²', None, ''))
            ovs = sorted(pc['overlaps'].items(), key=lambda kv: -kv[1])[:2]
            for uid, a in ovs:
                o_c = st.cam(uid)
                rows.append((f'Overlap · {o_c.tag if o_c else "?"}', f'{a:.3f}', 'm²', None, ''))
            mn = p.analysis.min_density
            rows.append(('Min in ROI', f'{pc["min_dens"]:.1f}', 'px/cm', 'warning' if 0 < pc['min_dens'] < mn else None, ''))
            rows.append(('If removed', f'{pc["if_removed_blind"]:.3f}', 'm²', None, 'blind'))
            rows.append(('', f'{pc["if_removed_below"]:.3f}', 'm²', None, f'below {mn:.1f} px/cm'))
            worst = pc['if_removed_blind'] > 0.5
            self.me_icon.setPixmap(icons.pixmap('warning', STATUS['warning'], 12) if worst else icons.pixmap('good', '#00000000', 12))
            self.me_icon.setToolTip('Removing this camera leaves blind ground' if worst else '')
        for i, (r, v) in enumerate(self.me_rows):
            if i < len(rows):
                lab, val, unit, sev, note = rows[i]
                r.label.setText(lab)
                v.set(val, unit, sev, note)
                r.show()
            else:
                r.hide()
        self.s_me.setVisible(bool(rows))
        self._busy = False


# ------------------------------------------------------------------------------------------------ ROI
class RoiPage(Page):
    title = 'Region of interest'

    def __init__(self, ctx):
        super().__init__(ctx)
        head = QWidget()
        head.setFixedHeight(30)
        h = QHBoxLayout(head)
        h.setContentsMargins(8, 4, 8, 4)
        ic = QLabel()
        ic.setPixmap(icons.pixmap('roi', THEME.tok['ink2'], 16))
        self.name = QLineEdit()
        self.name.setReadOnly(True)
        h.addWidget(ic)
        h.addWidget(self.name, 1)
        self.lay.addWidget(head)
        s = Section('Shape')
        self.mode = SegmentedControl([('offset', 'Offset'), ('polygon', 'Polygon')], min_w=56)
        self.mode.changed.connect(self._mode)
        s.row('Mode', self.mode, QWidget())
        self.dist = ScrubSpinBox('', 'length', 1, 'm', step=0.05)
        self.dist.setFixedWidth(96)
        self.dist.committed.connect(lambda v: self._edit(lambda r: setattr(r, 'distance', max(0.2, round(v, 3))), 'ROI distance'))
        s.row('Distance', self.dist, QWidget())
        self.measured = QComboBox()
        self.measured.addItem('Body outline, plan view', 'outline')
        self.measured.addItem('Bounding box', 'bbox')
        self.measured.activated.connect(lambda i: self._edit(lambda r: setattr(r, 'measured_from', self.measured.itemData(i)), 'ROI measured from'))
        s.row('Measured from', self.measured)
        self.corners = QComboBox()
        self.corners.addItem('Round (true offset)', 'round')
        self.corners.addItem('Square', 'square')
        self.corners.activated.connect(lambda i: self._edit(lambda r: setattr(r, 'corners', self.corners.itemData(i)), 'ROI corners'))
        s.row('Corners', self.corners)
        self.exclude = QCheckBox('Ground under the body')
        self.exclude.toggled.connect(lambda on: self._edit(lambda r: setattr(r, 'exclude_under_body', on), 'ROI exclusion') if not self._busy else None)
        s.row('Exclude', self.exclude)
        self.lay.addWidget(s)
        self.lay.addWidget(hline())
        sp = Section('Polygon')
        sp.header.set_summary('used in polygon mode')
        self.poly = _table(['#', 'X m', 'Y m'], 1)
        self.poly.setEditTriggers(QAbstractItemView.EditTrigger.DoubleClicked | QAbstractItemView.EditTrigger.EditKeyPressed)
        self.poly.itemChanged.connect(self._poly_edit)
        sp.add(self.poly)
        bl = QHBoxLayout()
        self.b_edit = button('Edit on map')
        self.b_edit.setCheckable(True)
        self.b_edit.toggled.connect(self._edit_on_map)
        self.b_from = button('From offset shape')
        self.b_from.clicked.connect(self._poly_from_offset)
        bl.addWidget(self.b_edit)
        bl.addWidget(self.b_from)
        bl.addStretch(1)
        sp.blay.addLayout(bl)
        self.lay.addWidget(sp)
        self.lay.addWidget(hline())
        sr = Section('Result')
        self.r_area, self.r_cells, self.r_blind, self.r_min, self.r_time = (ValueLabel() for _ in range(5))
        sr.row('Area', self.r_area)
        self.cells_row = sr.row('Cells at 5 cm', self.r_cells)
        sr.row('Blind', self.r_blind)
        sr.row('Min density', self.r_min)
        sr.row('Analysis time', self.r_time)
        self.lay.addWidget(sr)
        self.lay.addWidget(hline())
        st = Section('Density thresholds')
        self.t_min = ScrubSpinBox('', 'plain', 1, 'px/cm', step=0.1)
        self.t_min.committed.connect(lambda v: self._an(lambda a: setattr(a, 'min_density', max(0.1, v)), 'Minimum density'))
        self.t_min_pct = ValueLabel()
        st.row('Minimum', self.t_min, self.t_min_pct)
        self.t_tgt = ScrubSpinBox('', 'plain', 1, 'px/cm', step=0.1)
        self.t_tgt.committed.connect(lambda v: self._an(lambda a: setattr(a, 'target_density', max(0.1, v)), 'Target density'))
        self.t_tgt_pct = ValueLabel()
        st.row('Target', self.t_tgt, self.t_tgt_pct)
        st.add(label('Share of ROI at or above each threshold. Density is the best camera per cell: '
                     '√(radial × tangential sampling) / range, in px/cm.', muted=True, small=True, wrap=True))
        self.lay.addWidget(st)
        self.lay.addStretch(1)
        self._busy = False

    def _edit(self, fn, text):
        if self._busy:
            return
        self.ctx.store.edit('roi', lambda p: fn(p.roi), text)
        self.ctx.analysis.request()

    def _an(self, fn, text):
        self.ctx.store.edit('analysis', lambda p: fn(p.analysis), text)
        self.ctx.analysis.request()

    def _mode(self, m):
        def fn(r):
            r.mode = m
            if m == 'polygon' and len(r.polygon) < 3:
                self._seed_polygon(r)
        self._edit(fn, f'ROI mode {m}')

    def _seed_polygon(self, r):
        res = self.ctx.analysis.result
        if res is not None:
            x0, y0, x1, y1 = (res.grid.x0 + 0.25, res.grid.y0 + 0.25, res.grid.x0 + res.grid.nx * res.grid.cell - 0.25,
                              res.grid.y0 + res.grid.ny * res.grid.cell - 0.25)
        else:
            x0, y0, x1, y1 = -2.8, -2.8, 5.2, 2.8
        r.polygon = [(round(x1, 3), round(y1, 3)), (round(x0, 3), round(y1, 3)), (round(x0, 3), round(y0, 3)), (round(x1, 3), round(y0, 3))]

    def _poly_from_offset(self):
        res = self.ctx.analysis.result
        if res is None or not res.grid.boundary:
            return
        b = max(res.grid.boundary, key=len)
        step = max(1, len(b) // 24)
        pts = [(round(float(x), 3), round(float(y), 3)) for x, y in b[::step]]
        self._edit(lambda r: setattr(r, 'polygon', pts), 'ROI polygon from offset shape')

    def _poly_edit(self, it):
        if self._busy:
            return
        r, c = it.row(), it.column()
        if c == 0:
            return
        try:
            v = parse_number(it.text(), 'length')
        except Exception:
            self.refresh()
            return
        o = self.project.origin_offset()

        def fn(roi, r=r, c=c, v=v):
            poly = [list(p) for p in roi.polygon]
            poly[r][c - 1] = round(v + o[c - 1], 3)
            roi.polygon = [tuple(p) for p in poly]
        self._edit(fn, 'Edited ROI vertex')

    def _edit_on_map(self, on):
        if self.ctx.window:
            self.ctx.window.edit_roi_on_map(on)

    def refresh(self):
        self._busy = True
        p = self.project
        r = p.roi
        self.name.setText(f'ROI {r.distance:.1f} m' if r.mode == 'offset' else 'ROI polygon')
        self.mode.set_value(r.mode)
        self.dist.set_value(r.distance)
        self.measured.setCurrentIndex(max(0, self.measured.findData(r.measured_from)))
        self.corners.setCurrentIndex(max(0, self.corners.findData(r.corners)))
        self.exclude.setChecked(r.exclude_under_body)
        o = p.origin_offset()
        self.poly.setRowCount(len(r.polygon))
        for i, (x, y) in enumerate(r.polygon):
            self.poly.setItem(i, 0, _cell(str(i + 1), muted=True))
            self.poly.setItem(i, 1, _cell(f'{x - o[0]:.3f}', True, True))
            self.poly.setItem(i, 2, _cell(f'{y - o[1]:.3f}', True, True))
        _fit_table(self.poly)
        res = self.ctx.analysis.result
        if res:
            m = res.metrics
            self.r_area.set(f'{m["roi_area"]:.3f}', 'm²')
            self.cells_row.label.setText(f'Cells at {res.cell * 100:.0f} cm')
            self.r_cells.set(f'{m["cells"]:,}'.replace(',', ' '))
            self.r_blind.set(f'{m["blind_m2"]:.3f}', 'm²', 'good' if m['blind_m2'] < 1e-3 else 'critical')
            self.r_min.set(f'{m["min_density"]:.1f}', 'px/cm')
            self.r_time.set(f'{res.ms:.0f}', 'ms')
            self.t_min_pct.set(f'{m["min_ok_pct"]:.1f}', '%')
            self.t_tgt_pct.set(f'{m["target_ok_pct"]:.1f}', '%')
        self.t_min.set_value(p.analysis.min_density)
        self.t_tgt.set_value(p.analysis.target_density)
        self._busy = False


# ------------------------------------------------------------------------------------------------ vehicle
class VehiclePage(Page):
    title = 'Vehicle'

    def __init__(self, ctx):
        super().__init__(ctx)
        s = Section('Body')
        self.name = QLineEdit()
        self.name.editingFinished.connect(self._rename)
        s.row('Name', self.name)
        self.src = ValueLabel()
        s.row('Source', self.src)
        self.tris = ValueLabel()
        s.row('Triangles', self.tris)
        self.occ = QSlider(Qt.Orientation.Horizontal)
        self.occ.setRange(1000, 60000)
        self.occ.setSingleStep(500)
        self.occ.sliderReleased.connect(self._occ)
        self.occ_v = ValueLabel()
        s.row('Occluder', self.occ, self.occ_v)
        s.add(label('Line-of-sight tests use the simplified occluder; the full mesh stays for display and picking.',
                    muted=True, small=True, wrap=True))
        self.lay.addWidget(s)
        self.lay.addWidget(hline())
        sc = Section('Spec check')
        self.spec = _table(['Dimension', 'Measured m', 'Spec m', 'Δ mm', 'Status'], 4)
        sc.add(self.spec)
        bl = QHBoxLayout()
        b1 = button('Replace model…')
        b1.clicked.connect(lambda: ctx.window.import_vehicle())
        b2 = button('Edit spec…')
        b2.clicked.connect(lambda: ctx.window.spec_check())
        bl.addWidget(b1)
        bl.addWidget(b2)
        bl.addStretch(1)
        sc.blay.addLayout(bl)
        self.lay.addWidget(sc)
        self.lay.addStretch(1)

    def _rename(self):
        t = self.name.text().strip()
        if t and t != self.project.vehicle.name:
            self.ctx.store.edit('vehicle', lambda p: setattr(p.vehicle, 'name', t), f'Vehicle renamed to {t}')

    def _occ(self):
        v = int(round(self.occ.value() / 500) * 500)
        self.ctx.store.edit('vehicle', lambda p: setattr(p.vehicle, 'occluder_target', v), f'Occluder {v} tris')
        self.ctx.analysis.request()

    def refresh(self):
        v = self.project.vehicle
        self.name.setText(v.name)
        self.src.set(v.source.replace('template:', 'template · ') or '—')
        m = v.mesh
        vg = self.ctx.analysis.engine.vg
        self.tris.set(f'{(v.imported_tris or (m.n_tris if m else 0)):,}'.replace(',', ' '))
        self.occ.blockSignals(True)
        self.occ.setValue(v.occluder_target)
        self.occ.blockSignals(False)
        if vg is not None:
            self.occ_v.set(f'{vg.body.n_tris:,}'.replace(',', ' '), 'tris', None, f'{vg.max_dev * 1000:.1f} mm dev')
        fill_spec(self.spec, v)


def fill_spec(table, v):
    meas = v.measured()
    rows = [k for k in ('Length', 'Width', 'Height', 'Wheelbase', 'Ground clearance') if k in meas or k in v.spec]
    table.setRowCount(len(rows))
    for i, k in enumerate(rows):
        mv = meas.get(k)
        sv = v.spec.get(k)
        table.setItem(i, 0, _cell(k))
        table.setItem(i, 1, _cell(f'{mv:.3f}' if mv is not None else '—', True, True))
        table.setItem(i, 2, _cell(f'{sv:.3f}' if sv else '—', True, True))
        if mv is not None and sv:
            d = (mv - sv) * 1000
            table.setItem(i, 3, _cell(f'{d:+.0f}'.replace('-', '−'), True, True))
            sev = 'good' if abs(d) <= 5 else 'warning'
            it = _cell('Within tolerance' if sev == 'good' else ('Warning · check ride height' if k == 'Ground clearance' else 'Warning · check scale'))
            it.setIcon(icons.status_icon(sev, 12))
            table.setItem(i, 4, it)
        else:
            table.setItem(i, 3, _cell('—', True, True, True))
            table.setItem(i, 4, _cell('no spec value', muted=True))
    _fit_table(table)


class WheelsPage(Page):
    title = 'Wheels'

    def __init__(self, ctx):
        super().__init__(ctx)
        s = Section('Wheel centres')
        self.t = _table(['Wheel', 'X m', 'Y m', 'Z m'], 0)
        s.add(self.t)
        self.wb, self.tr = ValueLabel(), ValueLabel()
        s.row('Wheelbase', self.wb)
        s.row('Track', self.tr)
        b = button('Pick wheel centres…')
        b.clicked.connect(lambda: ctx.window.import_vehicle(step=2, reuse=True))
        bl = QHBoxLayout()
        bl.addWidget(b)
        bl.addStretch(1)
        s.blay.addLayout(bl)
        self.lay.addWidget(s)
        self.lay.addStretch(1)

    def refresh(self):
        wc = self.project.vehicle.wheel_centres
        o = self.project.origin_offset()
        keys = [k for k in ('FL', 'FR', 'RL', 'RR') if k in wc]
        self.t.setRowCount(len(keys))
        for i, k in enumerate(keys):
            self.t.setItem(i, 0, _cell(k))
            for j in range(3):
                self.t.setItem(i, j + 1, _cell(f'{wc[k][j] - o[j]:.3f}', True, True))
        _fit_table(self.t)
        if 'FL' in wc and 'RL' in wc:
            self.wb.set(f'{abs(wc["FL"][0] - wc["RL"][0]):.3f}', 'm')
        if 'FL' in wc and 'FR' in wc:
            self.tr.set(f'{abs(wc["FL"][1] - wc["FR"][1]):.3f}', 'm', None, 'hub faces')


class ZonesPage(Page):
    title = 'Mount zones'

    def __init__(self, ctx):
        super().__init__(ctx)
        s = Section('Zones')
        self.t = _table(['Kind', 'Zone', 'Faces'], 1)
        self.t.setEditTriggers(QAbstractItemView.EditTrigger.DoubleClicked)
        self.t.itemChanged.connect(self._renamed)
        self.t.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.t.customContextMenuRequested.connect(self._menu)
        s.add(self.t)
        bl = QHBoxLayout()
        self.b_allow = button('Paint allowed')
        self.b_allow.setCheckable(True)
        self.b_forbid = button('Paint forbidden')
        self.b_forbid.setCheckable(True)
        self.b_allow.toggled.connect(lambda on: self._paint('allowed', on))
        self.b_forbid.toggled.connect(lambda on: self._paint('forbidden', on))
        bl.addWidget(self.b_allow)
        bl.addWidget(self.b_forbid)
        bl.addStretch(1)
        s.blay.addLayout(bl)
        s.add(label('Painting opens the vehicle wizard on the zones step; glass is found from materials and transparency.',
                    muted=True, small=True, wrap=True))
        self.lay.addWidget(s)
        self.lay.addStretch(1)
        self._busy = False

    def _paint(self, kind, on):
        if on and self.ctx.window:
            self.b_allow.setChecked(False)
            self.b_forbid.setChecked(False)
            self.ctx.window.import_vehicle(step=3, reuse=True, paint=kind)

    def _renamed(self, it):
        if self._busy or it.column() != 1:
            return
        i = it.row()
        name = it.text().strip()
        m = self.project.vehicle.mesh
        if m is None or not name:
            return

        def fn(p, i=i, name=name):
            p.vehicle.mesh.zones[i].name = name
        self.ctx.store.edit('vehicle', fn, f'Renamed zone to {name}')

    def _menu(self, pos):
        it = self.t.itemAt(pos)
        if it is None:
            return
        i = it.row()
        m = QMenu(self)

        def toggle(p, i=i):
            z = p.vehicle.mesh.zones[i]
            z.kind = 'forbidden' if z.kind == 'allowed' else 'allowed'

        def delete(p, i=i):
            mesh = p.vehicle.mesh
            mesh.face_zone = np.where(mesh.face_zone == i, -1, np.where(mesh.face_zone > i, mesh.face_zone - 1, mesh.face_zone)).astype(np.int16)
            mesh.zones.pop(i)
        m.addAction('Toggle allowed / forbidden', lambda: (self.ctx.store.edit('vehicle', toggle, 'Changed zone kind'), self.ctx.analysis.request()))
        m.addAction('Delete zone', lambda: (self.ctx.store.edit('vehicle', delete, 'Deleted zone'), self.ctx.analysis.request()))
        m.exec(self.t.viewport().mapToGlobal(pos))

    def refresh(self):
        self._busy = True
        m = self.project.vehicle.mesh
        zones = m.zones if m is not None else []
        self.t.setRowCount(len(zones))
        for i, z in enumerate(zones):
            n = int((m.face_zone == i).sum())
            k = _cell(z.kind, color=None if z.kind == 'allowed' else QColorHex(STATUS['critical']))
            k.setFlags(k.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.t.setItem(i, 0, k)
            self.t.setItem(i, 1, QTableWidgetItem(z.name))
            c = _cell(str(n), True, True)
            c.setFlags(c.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.t.setItem(i, 2, c)
        _fit_table(self.t)
        self._busy = False


def QColorHex(h):
    from PyQt6.QtGui import QColor
    return QColor(h)


# ------------------------------------------------------------------------------------------------ ECU
class EcuPage(Page):
    title = 'Compute'

    def __init__(self, ctx):
        super().__init__(ctx)
        s = Section('Deserializer')
        self.name = QLineEdit()
        self.name.editingFinished.connect(lambda: self._set('name', self.name.text()))
        s.row('Name', self.name)
        self.loc = QLineEdit()
        self.loc.editingFinished.connect(lambda: self._set('location', self.loc.text()))
        s.row('Location', self.loc)
        self.ex, self.ey, self.ez = (ScrubSpinBox(a, 'length', 3) for a in 'XYZ')
        for w, k in ((self.ex, 'x'), (self.ey, 'y'), (self.ez, 'z')):
            w.committed.connect(lambda v, k=k: self._set(k, v + self.project.origin_offset()['xyz'.index(k)]))
        s.row('Position m', self.ex, self.ey, self.ez)
        self.factor = ScrubSpinBox('', 'plain', 2, '×', step=0.01)
        self.factor.committed.connect(lambda v: self._set('route_factor', max(1.0, v)))
        self.loop = ScrubSpinBox('', 'length', 2, 'm', step=0.05)
        self.loop.committed.connect(lambda v: self._set('service_loop_m', max(0.0, v)))
        s.row('Routing slack', self.factor, self.loop)
        self.lay.addWidget(s)
        self.lay.addWidget(hline())
        sp = Section('GMSL ports · Jetson AGX Orin')
        self.t = _table(['Port', 'Cam', 'Deser.', 'Routed m', 'Gbps'], 2)
        sp.add(self.t)
        self.lay.addWidget(sp)
        self.lay.addWidget(hline())
        sb = Section('Link budget')
        self.link = ScrubSpinBox('', 'plain', 1, 'Gbps', step=0.5)
        self.link.committed.connect(lambda v: self._set('link_gbps', max(0.1, v)))
        self.csi = ScrubSpinBox('', 'plain', 1, 'Gbps', step=0.5)
        self.csi.committed.connect(lambda v: self._set('csi_gbps_per_deser', max(0.1, v)))
        sb.row('Link / deser.', self.link, self.csi)
        self.bpp = ScrubSpinBox('', 'plain', 0, 'bit/px', step=1)
        self.bpp.committed.connect(lambda v: self._set('bpp', int(max(1, round(v)))))
        self.ovh = ScrubSpinBox('', 'plain', 2, '×', step=0.01)
        self.ovh.committed.connect(lambda v: self._set('overhead', max(1.0, v)))
        sb.row('Pixel format', self.bpp, self.ovh)
        self.bt = _table(['Deser.', 'Cameras', 'Gbps', 'Load'], 1)
        sb.add(self.bt)
        self.budget_note = label('', muted=True, small=True, wrap=True)
        sb.add(self.budget_note)
        self.lay.addWidget(sb)
        self.lay.addStretch(1)

    def _set(self, k, v):
        if getattr(self.project.ecu, k) == v:
            return                         # focus left a field without a change: no undo step
        self.ctx.store.edit('ecu', lambda p: setattr(p.ecu, k, v), f'ECU {k}')

    def refresh(self):
        p = self.project
        e = p.ecu
        o = p.origin_offset()
        self.name.setText(e.name)
        self.loc.setText(e.location)
        self.ex.set_value(e.x - o[0])
        self.ey.set_value(e.y - o[1])
        self.ez.set_value(e.z - o[2])
        self.factor.set_value(e.route_factor)
        self.loop.set_value(e.service_loop_m)
        self.t.setRowCount(e.ports)
        for k in range(1, e.ports + 1):
            cam = next((c for c in p.variant.cameras if c.port == k), None)
            self.t.setItem(k - 1, 0, _cell(str(k), True))
            if cam:
                it = _cell(cam.tag, True)
                self.t.setItem(k - 1, 1, it)
                rl = route_length(p, cam)
                r = _cell(f'{rl:.2f}', True, True)
                r.setIcon(icons.status_icon('serious' if rl > cam.cable_m else 'good', 12))
                self.t.setItem(k - 1, 3, r)
                g = camera_gbps(p, cam)
                gi = _cell(f'{g:.2f}', True, True)
                if g > e.link_gbps:
                    gi.setForeground(THEME.color('serious'))
                self.t.setItem(k - 1, 4, gi)
            else:
                self.t.setItem(k - 1, 1, _cell('—', muted=True))
                self.t.setItem(k - 1, 3, _cell('—', True, True, True))
                self.t.setItem(k - 1, 4, _cell('—', True, True, True))
            dz = _cell(f"{'A' if k <= 4 else 'B'} · link {(k - 1) % 4 + 1}", muted=True)
            dz.setToolTip(e.deserializer(k))
            self.t.setItem(k - 1, 2, dz)
        _fit_table(self.t)
        self.link.set_value(e.link_gbps)
        self.csi.set_value(e.csi_gbps_per_deser)
        self.bpp.set_value(e.bpp)
        self.ovh.set_value(e.overhead)
        rows = link_budget(p)
        self.bt.setRowCount(len(rows))
        notes = []
        for i, (name, ports, tags, used, cap, worst, link) in enumerate(rows):
            load = used / cap if cap else 0.0
            st = 'serious' if load > 1 else 'warning' if load > 0.85 else 'good'
            ni = _cell(name, True)
            ni.setToolTip(f'Deserializer {name} · ports {ports}')
            self.bt.setItem(i, 0, ni)
            self.bt.setItem(i, 1, _cell(' '.join(tags) or '—', True, muted=not tags))
            self.bt.setItem(i, 2, _cell(f'{used:.2f} / {cap:g}', True, True))
            li = _cell(f'{load * 100:.0f} %', True, True, color=THEME.color(st) if tags else None)
            li.setIcon(icons.status_icon(st, 12))
            self.bt.setItem(i, 3, li)
            if load > 1:
                notes.append(f'Deserializer {name} needs {used:.1f} of {cap:.1f} Gbps: move a camera to the other one '
                             'or lower the frame rate.')
            if worst > link:
                notes.append(f'A camera on {name} needs {worst:.2f} Gbps, more than one {link:.1f} Gbps link carries.')
        _fit_table(self.bt)
        self.budget_note.setText(' '.join(notes) if notes else
                                 f'Per camera: width × height × fps × {e.bpp} bit/px × {e.overhead:.2f}. '
                                 'Edit the values to match the serializer and pixel format.')


class ObstaclePage(Page):
    title = 'Obstacle'

    def __init__(self, ctx):
        super().__init__(ctx)
        s = Section('Pole')
        self.name = QLineEdit()
        self.name.editingFinished.connect(lambda: self._set('name', self.name.text()))
        s.row('Name', self.name)
        self.ox, self.oy = ScrubSpinBox('X', 'length', 3), ScrubSpinBox('Y', 'length', 3)
        self.ox.committed.connect(lambda v: self._set('x', v + self.project.origin_offset()[0]))
        self.oy.committed.connect(lambda v: self._set('y', v + self.project.origin_offset()[1]))
        s.row('Position m', self.ox, self.oy)
        self.h = ScrubSpinBox('', 'length', 2, 'm', step=0.05)
        self.h.committed.connect(lambda v: self._set('height', max(0.05, v)))
        s.row('Height', self.h, QWidget())
        self.r = ScrubSpinBox('', 'length', 3, 'm', step=0.005)
        self.r.committed.connect(lambda v: self._set('radius', max(0.005, v)))
        s.row('Radius', self.r, QWidget())
        self.lay.addWidget(s)
        self.lay.addWidget(hline())
        sv = Section('Seen by')
        self.t = _table(['Cam', 'Visible', 'Max px/cm'], 0)
        sv.add(self.t)
        self.none = StatusLine('critical', 'No camera sees this pole.')
        sv.add(self.none)
        bl = QHBoxLayout()
        b = button('Delete obstacle', 'destructive', 'critical')
        b.clicked.connect(self._delete)
        bl.addWidget(b)
        bl.addStretch(1)
        sv.blay.addLayout(bl)
        self.lay.addWidget(sv)
        self.lay.addStretch(1)

    def _uid(self):
        o = self.ctx.store.sel_obj
        return o[1] if o and o[0] == 'obstacle' else None

    def _set(self, k, v):
        uid = self._uid()
        ob = next((o for o in self.project.obstacles if o.uid == uid), None)
        if ob is None or getattr(ob, k) == v:
            return                         # focus left a field without a change: no undo step

        def fn(p):
            for o in p.obstacles:
                if o.uid == uid:
                    setattr(o, k, v)
        self.ctx.store.edit('obstacles', fn, f'Obstacle {k}')
        self.ctx.analysis.request()

    def _delete(self):
        uid = self._uid()
        self.ctx.store.edit('obstacles', lambda p: setattr(p, 'obstacles', [o for o in p.obstacles if o.uid != uid]), 'Deleted obstacle')
        self.ctx.store.select_obj(('obstacles',))
        self.ctx.analysis.request()

    def refresh(self):
        uid = self._uid()
        ob = next((o for o in self.project.obstacles if o.uid == uid), None)
        if ob is None:
            return
        o = self.project.origin_offset()
        self.name.setText(ob.name)
        self.ox.set_value(ob.x - o[0])
        self.oy.set_value(ob.y - o[1])
        self.h.set_value(ob.height)
        self.r.set_value(ob.radius)
        res = self.ctx.analysis.result
        seen = res.obstacles.get(uid, {}) if res else {}
        self.t.setRowCount(len(seen))
        for i, (cu, (frac, dens)) in enumerate(seen.items()):
            c = self.ctx.store.cam(cu)
            self.t.setItem(i, 0, _cell(c.tag if c else '?', True))
            self.t.setItem(i, 1, _cell(f'{frac * 100:.0f} %', True, True))
            self.t.setItem(i, 2, _cell(f'{dens:.1f}', True, True))
        _fit_table(self.t)
        self.none.setVisible(res is not None and not seen)


class ObstaclesPage(Page):
    title = 'Obstacles'

    def __init__(self, ctx):
        super().__init__(ctx)
        s = Section('Poles')
        self.t = _table(['Name', 'X m', 'Y m', 'H m', 'Seen by'], 4)
        self.t.cellClicked.connect(lambda r, c: ctx.store.select_obj(('obstacle', self.project.obstacles[r].uid)))
        s.add(self.t)
        bl = QHBoxLayout()
        b = button('Add pole', icon_name='plus')
        b.clicked.connect(self._add)
        bl.addWidget(b)
        bl.addStretch(1)
        s.blay.addLayout(bl)
        mode = QComboBox()
        for k, t in (('ground', 'Ground only (z = 0)'), ('pole05', 'Vertical targets at 0.5 m'), ('pole10', 'Vertical targets at 1.0 m'),
                     ('child', 'Child cylinder (0.9 m)')):
            mode.addItem(t, k)
        mode.activated.connect(lambda i: (ctx.store.edit('analysis', lambda p: setattr(p.analysis, 'obstacle_mode', mode.itemData(i)),
                                                         'Obstacle test mode'), ctx.analysis.request()))
        self.mode = mode
        s.row('Coverage height', mode)
        self.lay.addWidget(s)
        self.lay.addStretch(1)

    def _add(self):
        n = len(self.project.obstacles) + 1

        def fn(p):
            p.obstacles.append(Obstacle(name=f'Pole {n}', x=4.0, y=-1.2, height=1.0))
        self.ctx.store.edit('obstacles', fn, 'Added pole')
        self.ctx.analysis.request()

    def refresh(self):
        p = self.project
        o = p.origin_offset()
        res = self.ctx.analysis.result
        self.t.setRowCount(len(p.obstacles))
        for i, ob in enumerate(p.obstacles):
            self.t.setItem(i, 0, _cell(ob.name))
            self.t.setItem(i, 1, _cell(f'{ob.x - o[0]:.2f}', True, True))
            self.t.setItem(i, 2, _cell(f'{ob.y - o[1]:.2f}', True, True))
            self.t.setItem(i, 3, _cell(f'{ob.height:.2f}', True, True))
            seen = res.obstacles.get(ob.uid, {}) if res else {}
            tags = ' '.join(self.ctx.store.cam(u).tag for u in seen if self.ctx.store.cam(u))
            self.t.setItem(i, 4, _cell(tags or 'none', True, color=None if tags else QColorHex(STATUS['critical'])))
        _fit_table(self.t)
        self.mode.setCurrentIndex(max(0, self.mode.findData(p.analysis.obstacle_mode)))


# ------------------------------------------------------------------------------------------------ variant / seams
class VariantPage(Page):
    title = 'Variant'

    def __init__(self, ctx):
        super().__init__(ctx)
        s = Section('Variant')
        self.name = QLineEdit()
        self.name.editingFinished.connect(self._rename)
        s.row('Name', self.name)
        self.note = QLineEdit()
        self.note.editingFinished.connect(self._note)
        s.row('Note', self.note)
        self.ncam = ValueLabel()
        s.row('Cameras', self.ncam)
        self.blend = ScrubSpinBox('', 'length', 2, 'm', step=0.01)
        self.blend.committed.connect(lambda v: (ctx.store.edit('variant', lambda p: setattr(p.variant, 'blend_m', max(0.0, v)), 'Seam blend width'),
                                                ctx.analysis.request()))
        s.row('Seam blend', self.blend, QWidget())
        self.lay.addWidget(s)
        self.lay.addWidget(hline())
        ss = Section('Stitch seams')
        ss.header.set_summary('angle from the body centre')
        self.t = _table(['Seam', 'Angle °', '', 'px/cm'], 0)
        self.t.setEditTriggers(QAbstractItemView.EditTrigger.DoubleClicked)
        self.t.itemChanged.connect(self._angle)
        ss.add(self.t)
        bl = QHBoxLayout()
        b = button('Reset to auto')
        b.clicked.connect(lambda: (ctx.store.edit('variant', lambda p: setattr(p.variant, 'seams', {}), 'Seams reset to auto'), ctx.analysis.request()))
        bl.addWidget(b)
        bl.addStretch(1)
        ss.blay.addLayout(bl)
        self.lay.addWidget(ss)
        self.lay.addStretch(1)
        self._busy = False

    def _rename(self):
        t = self.name.text().strip()
        if t and t != self.project.variant.name:
            self.ctx.store.edit('variants', lambda p: setattr(p.variant, 'name', t), f'Variant renamed to {t}')

    def _note(self):
        t = self.note.text()
        if t != self.project.variant.note:
            self.ctx.store.edit('variant', lambda p: setattr(p.variant, 'note', t), 'Variant note')

    def _angle(self, it):
        if self._busy or it.column() != 1:
            return
        res = self.ctx.analysis.result
        if res is None:
            return
        s = res.seams[it.row()]
        try:
            v = parse_number(it.text(), 'angle')        # rejects nan / inf
        except ValueError:
            self.refresh()
            return
        self.ctx.store.edit('variant', lambda p: p.variant.seams.__setitem__(f'{s.a}|{s.b}', round(v, 1)), 'Seam angle')
        self.ctx.analysis.request()

    def refresh(self):
        self._busy = True
        v = self.project.variant
        self.name.setText(v.name)
        self.note.setText(v.note)
        self.ncam.set(str(len(v.cameras)), '', None, f'{len(v.cameras)} of 8 GMSL ports')
        self.blend.set_value(v.blend_m)
        res = self.ctx.analysis.result
        seams = res.seams if res else []
        self.t.setRowCount(len(seams))
        for i, s in enumerate(seams):
            a, b = self.ctx.store.cam(s.a), self.ctx.store.cam(s.b)
            c0 = _cell(f'{a.tag if a else "?"} | {b.tag if b else "?"}', True)
            c0.setFlags(c0.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.t.setItem(i, 0, c0)
            self.t.setItem(i, 1, _cell(f'{s.angle:.1f}', True, True))
            c2 = _cell('auto' if s.auto else 'manual', muted=True)
            c2.setFlags(c2.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.t.setItem(i, 2, c2)
            c3 = _cell(f'{s.density:.1f}', True, True)
            c3.setFlags(c3.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.t.setItem(i, 3, c3)
        _fit_table(self.t)
        self._busy = False


class EmptyPage(Page):
    title = ''

    def __init__(self, ctx):
        super().__init__(ctx)
        w = label('Nothing selected.\n\nClick a camera in the viewport or the outliner.\nPress C to place a camera on the body, '
                  'or drag a module from the Library.', muted=True, wrap=True)
        w.setContentsMargins(10, 10, 10, 10)
        self.lay.addWidget(w)
        self.lay.addStretch(1)


class Inspector(QWidget):
    titleChanged = pyqtSignal(str)

    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        self.setObjectName('panel')
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.stack = QStackedWidget()
        self.stack.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll.setWidget(self.stack)
        lay.addWidget(self.scroll)
        from .inspector_extra import PhotosPage, TargetsPage
        from .rigview import RigPage
        from .sensorview import SensorPage
        self.pages = {
            'empty': EmptyPage(ctx), 'camera': CameraPage(ctx), 'roi': RoiPage(ctx), 'vehicle': VehiclePage(ctx),
            'wheels': WheelsPage(ctx), 'zones': ZonesPage(ctx), 'ecu': EcuPage(ctx), 'obstacle': ObstaclePage(ctx),
            'obstacles': ObstaclesPage(ctx), 'variant': VariantPage(ctx), 'photos': PhotosPage(ctx), 'targets': TargetsPage(ctx),
            'rig': RigPage(ctx), 'sensor': SensorPage(ctx),
        }
        for pg in self.pages.values():
            pg.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
            pg.setMinimumWidth(0)
            self.stack.addWidget(pg)
        self.current = 'empty'
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(15)
        self.timer.timeout.connect(self.refresh)
        st = ctx.store
        st.selection_changed.connect(self.timer.start)
        st.changed.connect(lambda *_: self.timer.start())
        st.project_replaced.connect(self.timer.start)
        ctx.analysis.finished.connect(lambda *_: self.timer.start())
        ctx.analysis.secondary.connect(lambda *_: self.timer.start())
        self.refresh()

    def page_for_selection(self):
        st = self.ctx.store
        if st.sel_cams:
            return 'camera'
        o = st.sel_obj
        if not o:
            return 'empty'
        k = o[0]
        return {'target': 'targets', 'targets': 'targets', 'roi': 'roi', 'vehicle': 'vehicle', 'wheels': 'wheels',
                'zones': 'zones', 'ecu': 'ecu', 'obstacle': 'obstacle', 'obstacles': 'obstacles', 'variant': 'variant',
                'photos': 'photos', 'rig': 'rig', 'sensor': 'sensor', 'sensors': 'sensor'}.get(k, 'empty')

    def refresh(self):
        key = self.page_for_selection()
        pg = self.pages[key]
        if key != self.current:
            self.stack.setCurrentWidget(pg)
            self.current = key
        try:
            pg.refresh()
        except Exception as ex:
            import traceback
            traceback.print_exc()
        self.titleChanged.emit(pg.title)
