"""Smaller dialogs: preferences, shortcuts, spec check, cable route, custom frame, project settings, help, about."""
from __future__ import annotations

import copy
import csv
import json
import math
import os
import re

import numpy as np
from PyQt6.QtCore import Qt, QSettings
from PyQt6.QtGui import QKeySequence
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QLineEdit, QComboBox, QCheckBox,
                             QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView, QPushButton, QTextBrowser,
                             QApplication)

from ...core.mathutil import parse_number
from ...core.project import Camera, FRAME_PRESETS
from ...core.rules import route_length, route_text
from ..theme import THEME, mono, STATUS
from ..widgets import label, button, ScrubSpinBox, SegmentedControl, hline, ROW_H, StatusLine
from .. import icons
from .base import VDialog


def confirm(parent, title, text, ok_text='OK', destructive=True) -> bool:
    d = VDialog(parent, title, '', (460, 190), resizable=False)
    l = QVBoxLayout(d.body)
    l.setContentsMargins(14, 12, 14, 12)
    l.addWidget(label(text, wrap=True))
    l.addStretch(1)
    c = button('Cancel', 'ghost')
    c.clicked.connect(d.reject)
    o = button(ok_text, 'destructive' if destructive else 'primary', 'critical' if destructive else None)
    o.clicked.connect(d.accept)
    d.buttons.addWidget(c)
    d.buttons.addWidget(o)
    return bool(d.exec())


def ask_text(parent, title, lab, default='') -> str | None:
    d = VDialog(parent, title, '', (420, 160), resizable=False)
    l = QHBoxLayout(d.body)
    l.setContentsMargins(14, 12, 14, 12)
    l.addWidget(label(lab, ink2=True))
    e = QLineEdit(default)
    e.selectAll()
    l.addWidget(e, 1)
    c = button('Cancel', 'ghost')
    c.clicked.connect(d.reject)
    o = button('OK', 'primary')
    o.clicked.connect(d.accept)
    e.returnPressed.connect(d.accept)
    d.buttons.addWidget(c)
    d.buttons.addWidget(o)
    return e.text().strip() if d.exec() else None


# ------------------------------------------------------------------------------------- preferences
class PreferencesDialog(VDialog):
    def __init__(self, ctx, parent=None, settings=None, tab='general'):
        super().__init__(parent, 'Preferences', '', (660, 520), resizable=False)
        self.ctx = ctx
        self.settings = settings or (ctx.settings if ctx else QSettings('Vantage', 'Vantage'))
        from PyQt6.QtWidgets import QTabWidget
        from ..assistant import AssistantPrefs
        outer = QVBoxLayout(self.body)
        outer.setContentsMargins(0, 0, 0, 0)
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        general = QWidget()
        general.setObjectName('panel')
        self.tabs.addTab(general, 'General')
        self.assistant = AssistantPrefs(self.settings)
        self.assistant.setObjectName('panel')
        self.tabs.addTab(self.assistant, 'Assistant')
        if tab == 'assistant':
            self.tabs.setCurrentWidget(self.assistant)
        outer.addWidget(self.tabs)
        g = QGridLayout(general)
        g.setContentsMargins(14, 12, 14, 12)
        g.setVerticalSpacing(6)
        s = self.settings
        r = 0

        def row(text, w, note=''):
            nonlocal r
            a = label(text, ink2=True)
            a.setFixedWidth(150)
            g.addWidget(a, r, 0)
            g.addWidget(w, r, 1)
            if note:
                g.addWidget(label(note, muted=True, small=True), r, 2)
            r += 1
        self.theme = SegmentedControl([('dark', 'Dark'), ('light', 'Light'), ('system', 'System')], s.value('ui/theme', 'dark'))
        row('Theme', self.theme)
        self.scale = SegmentedControl([('0.9', '90 %'), ('1.0', '100 %'), ('1.15', '115 %'), ('1.3', '130 %')],
                                      str(s.value('ui/scale', '1.0')))
        row('UI scale', self.scale, 'icons re-rasterised')
        self.nav = QComboBox()
        self.nav.addItem('Blender (middle-drag orbits, Shift pans)', 'blender')
        self.nav.addItem('CAD (right-drag orbits, middle pans)', 'cad')
        self.nav.setCurrentIndex(max(0, self.nav.findData(s.value('nav/style', 'blender'))))
        row('3D navigation', self.nav, 'left-drag on empty space orbits; or the control top right')
        self.units = SegmentedControl([('m', 'm'), ('mm', 'mm')], s.value('units/default', 'm'))
        row('Default units', self.units, 'new projects')
        self.live = QCheckBox('Recompute while editing')
        self.live.setChecked(s.value('analysis/live', True, type=bool))
        row('Live analysis', self.live)
        self.cell = QComboBox()
        for cm in (5, 2, 1):
            self.cell.addItem(f'{cm} cm', cm / 100)
        row('Ground resolution', self.cell, 'finer is slower')
        if ctx is not None:
            self.cell.setCurrentIndex(max(0, self.cell.findData(ctx.store.project.analysis.cell_m)))
        from ... import workdir
        wrow = QWidget()
        wl = QHBoxLayout(wrow)
        wl.setContentsMargins(0, 0, 0, 0)
        wl.setSpacing(6)
        self.working = QLineEdit(workdir.get(s))
        self.working.setPlaceholderText(workdir.default())
        self.working.setToolTip('Sample datasets, the assistant\'s outputs and exports of unsaved projects go here, '
                                'in a folder per project')
        wl.addWidget(self.working, 1)
        wb = button('Browse…', 'ghost')
        wb.clicked.connect(self._browse_working)
        wl.addWidget(wb)
        row('Working folder', wrow)
        self.soft = QCheckBox('Draw the 3D view with QPainter (no OpenGL)')
        self.soft.setChecked(s.value('render/software', False, type=bool))
        row('Renderer', self.soft, 'applies on restart')
        g.setRowStretch(r, 1)
        g.setColumnStretch(2, 1)
        c = button('Cancel', 'ghost')
        c.clicked.connect(self.reject)
        o = button('Apply', 'primary')
        o.clicked.connect(self.apply)
        self.buttons.addWidget(c)
        self.buttons.addWidget(o)

    def _browse_working(self):
        from PyQt6.QtWidgets import QFileDialog
        d = QFileDialog.getExistingDirectory(self, 'Working folder', self.working.text())
        if d:
            self.working.setText(d)

    def apply(self):
        from ..theme import THEME
        s = self.settings
        s.setValue('ui/theme', self.theme.value())
        s.setValue('ui/scale', self.scale.value())
        s.setValue('nav/style', self.nav.currentData())
        s.setValue('units/default', self.units.value())
        s.setValue('analysis/live', self.live.isChecked())
        s.setValue('render/software', self.soft.isChecked())
        from ... import workdir
        workdir.put(s, self.working.text())
        THEME.apply(QApplication.instance(), self.theme.value(), float(self.scale.value()))
        if self.ctx is not None:
            self.ctx.nav_style = self.nav.currentData()
            w = self.ctx.window
            for p in w.viewarea.panes:
                p.vp3d.nav_style = self.ctx.nav_style
            proj = self.ctx.store.project
            if abs(proj.analysis.cell_m - self.cell.currentData()) > 1e-9:
                w.set_cell(self.cell.currentData())
            if proj.analysis.live != self.live.isChecked():
                w.actions_by_id['live'].setChecked(self.live.isChecked())
        self.assistant.apply()
        if self.ctx is not None and getattr(self.ctx, 'assistant', None) is not None:
            self.ctx.assistant.reload()
        self.accept()


# ------------------------------------------------------------------------------------- shortcuts
class ShortcutsDialog(VDialog):
    def __init__(self, window, parent=None):
        super().__init__(parent, 'Keyboard shortcuts', 'searchable', (720, 600))
        l = QVBoxLayout(self.body)
        l.setContentsMargins(14, 10, 14, 10)
        self.q = QLineEdit()
        self.q.setPlaceholderText('Filter')
        self.q.addAction(icons.icon('search', 12), QLineEdit.ActionPosition.LeadingPosition)
        l.addWidget(self.q)
        self.t = QTableWidget(0, 3)
        self.t.setHorizontalHeaderLabels(['Command', 'Menu', 'Shortcut'])
        self.t.verticalHeader().setVisible(False)
        self.t.verticalHeader().setDefaultSectionSize(ROW_H)
        self.t.setShowGrid(False)
        self.t.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        hh = self.t.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        hh.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        l.addWidget(self.t)
        rows = []
        for aid, a in window.actions_by_id.items():
            sc = a.shortcut().toString(QKeySequence.SequenceFormat.NativeText)
            if sc:
                rows.append((a.text().replace('&', ''), (a.data() or {}).get('category', ''), sc))
        rows += [('Orbit (about the point under the cursor)', 'Viewport',
                  'Middle-drag or left-drag on empty space · touchpad: two fingers · ← → ↑ ↓'),
                 ('Pan', 'Viewport', 'Shift + middle-drag, right-drag · touchpad: Shift + two fingers · Shift + arrows'),
                 ('Zoom (towards the cursor)', 'Viewport', 'Wheel · pinch or Ctrl + two fingers · + / −'),
                 ('Navigation control', 'Viewport', 'Top right: drag the ball, click an axis, drag the hand or magnifier'),
                 ('Top / front / rear / side view', 'Viewport', '7 / 1 / Ctrl+1 / 3 (number row or numpad)'),
                 ('Default view', 'Viewport', '0'),
                 ('Toggle ortho', 'Viewport', '5'), ('Look through camera', 'Viewport', 'Numpad 0'),
                 ('Move or turn the selection', 'Viewport',
                  'Drag an arrow, the square or a ring · Ctrl snaps 1 cm / 5° · Shift fine · Esc cancels'),
                 ('Scrub a number', 'Inspector', 'Drag the axis letter · Shift fine · Ctrl coarse'),
                 ('Step a number', 'Inspector', '↑ / ↓'), ('Rename', 'Outliner', 'F2'), ('Back to viewports', 'Workspaces', 'Esc'),
                 ('Previous / next photo', 'Reference photos', '← / →'), ('Play the spin', 'Reference photos', 'Space')]
        self.rows = sorted(rows, key=lambda r: (r[1], r[0]))
        self.q.textChanged.connect(self.fill)
        self.fill()
        o = button('Close', 'primary')
        o.clicked.connect(self.accept)
        self.buttons.addWidget(o)

    def fill(self):
        q = self.q.text().lower()
        rows = [r for r in self.rows if not q or q in ' '.join(r).lower()]
        self.t.setRowCount(len(rows))
        for i, (a, b, c) in enumerate(rows):
            self.t.setItem(i, 0, QTableWidgetItem(a))
            it = QTableWidgetItem(b)
            it.setForeground(THEME.color('muted'))
            self.t.setItem(i, 1, it)
            s = QTableWidgetItem(c)
            s.setFont(mono(12))
            self.t.setItem(i, 2, s)


# ------------------------------------------------------------------------------------- spec check
class SpecDialog(VDialog):
    def __init__(self, ctx, parent=None):
        p = ctx.store.project
        super().__init__(parent, 'Spec check', f'{p.vehicle.name} · tolerance ± 5 mm', (640, 360))
        self.ctx = ctx
        l = QVBoxLayout(self.body)
        l.setContentsMargins(14, 10, 14, 10)
        from ..inspector import fill_spec, _table
        self.t = _table(['Dimension', 'Measured m', 'Spec m', 'Δ mm', 'Status'], 4)
        self.t.setEditTriggers(QAbstractItemView.EditTrigger.DoubleClicked)
        fill_spec(self.t, p.vehicle)
        for i in range(self.t.rowCount()):
            for c in (0, 1, 3, 4):
                it = self.t.item(i, c)
                if it:
                    it.setFlags(it.flags() & ~Qt.ItemFlag.ItemIsEditable)
        l.addWidget(self.t)
        l.addWidget(label('Double-click a spec value to edit it (metres). Measured values come from the mesh and the wheel centres.',
                          muted=True, small=True, wrap=True))
        l.addStretch(1)
        c = button('Cancel', 'ghost')
        c.clicked.connect(self.reject)
        o = button('Save spec', 'primary')
        o.clicked.connect(self.save)
        self.buttons.addWidget(c)
        self.buttons.addWidget(o)

    def save(self):
        spec = {}
        for i in range(self.t.rowCount()):
            k = self.t.item(i, 0).text()
            try:
                v = parse_number(self.t.item(i, 2).text(), 'length')
                if v > 0:
                    spec[k] = round(v, 4)
            except Exception:
                pass
        self.ctx.store.edit('vehicle', lambda p: setattr(p.vehicle, 'spec', spec), 'Edited vehicle spec')
        self.accept()


# ------------------------------------------------------------------------------------- cable route
class RouteDialog(VDialog):
    def __init__(self, ctx, uid, parent=None):
        c = ctx.store.cam(uid)
        super().__init__(parent, 'Cable route', f'{c.tag} · port {c.port or "—"}', (560, 330), resizable=False)
        self.ctx, self.uid = ctx, uid
        p = ctx.store.project
        g = QGridLayout(self.body)
        g.setContentsMargins(14, 12, 14, 12)
        g.setVerticalSpacing(6)
        self.note = QLineEdit(c.route_note or route_text(p, c))
        self.extra = ScrubSpinBox('', 'length', 2, 'm', step=0.05)
        self.extra.set_value(c.route_extra_m)
        self.cable = QComboBox()
        for x in p.module_for(c).cables_m:
            self.cable.addItem(f'{x:.2f} m', x)
        self.cable.setCurrentIndex(max(0, self.cable.findData(c.cable_m)))
        self.result = StatusLine('good', '')
        for i, (t, w) in enumerate((('Route', self.note), ('Extra length', self.extra), ('Module cable', self.cable))):
            a = label(t, ink2=True)
            a.setFixedWidth(110)
            g.addWidget(a, i, 0)
            g.addWidget(w, i, 1)
        g.addWidget(label(f'Routed length = Manhattan distance to the ECU × {p.ecu.route_factor:.2f} + {p.ecu.service_loop_m:.2f} m service loop '
                          '+ extra length.', muted=True, small=True, wrap=True), 3, 0, 1, 2)
        g.addWidget(self.result, 4, 0, 1, 2)
        g.setRowStretch(5, 1)
        self.extra.committed.connect(lambda v: self._upd())
        self.cable.currentIndexChanged.connect(lambda i: self._upd())
        self._upd()
        cb = button('Cancel', 'ghost')
        cb.clicked.connect(self.reject)
        o = button('Apply', 'primary')
        o.clicked.connect(self.apply)
        self.buttons.addWidget(cb)
        self.buttons.addWidget(o)

    def _upd(self):
        import copy
        c = copy.deepcopy(self.ctx.store.cam(self.uid))
        c.route_extra_m = self.extra.value()
        rl = route_length(self.ctx.store.project, c)
        cab = self.cable.currentData()
        if rl > cab:
            self.result.set('serious', f'Serious · routed {rl:.2f} m on a {cab:.2f} m cable, {rl - cab:.2f} m short.')
        else:
            self.result.set('good', f'Routed {rl:.2f} m on a {cab:.2f} m cable · {cab - rl:.2f} m spare.')

    def apply(self):
        self.ctx.store.set_cam(self.uid, route_note=self.note.text(), route_extra_m=self.extra.value(), cable_m=float(self.cable.currentData()),
                               text='Edited cable route')
        self.accept()


# ------------------------------------------------------------------------------------- custom frame
class CustomFrameDialog(VDialog):
    def __init__(self, ctx, parent=None):
        super().__init__(parent, 'Custom coordinate frame', 'origin in ISO 8855 rear-axle coordinates', (520, 230), resizable=False)
        self.ctx = ctx
        p = ctx.store.project
        h = QHBoxLayout(self.body)
        h.setContentsMargins(14, 14, 14, 14)
        self.f = [ScrubSpinBox(a, 'length', 3) for a in 'XYZ']
        for w, v in zip(self.f, p.custom_origin):
            w.set_value(v)
            h.addWidget(w)
        cb = button('Cancel', 'ghost')
        cb.clicked.connect(self.reject)
        o = button('Use this origin', 'primary')
        o.clicked.connect(self.apply)
        self.buttons.addWidget(cb)
        self.buttons.addWidget(o)
        self.hint.setText('Axes stay ISO 8855 (x forward, y left, z up).')

    def apply(self):
        vals = [w.value() for w in self.f]

        def fn(p):
            p.custom_origin = vals
            p.frame = 'custom'
        self.ctx.store.edit('frame', fn, 'Custom frame origin')
        self.accept()


# ------------------------------------------------------------------------------------- project settings
class ProjectSettingsDialog(VDialog):
    def __init__(self, ctx, parent=None):
        p = ctx.store.project
        super().__init__(parent, 'Project settings', p.name, (560, 300), resizable=False)
        self.ctx = ctx
        g = QGridLayout(self.body)
        g.setContentsMargins(14, 12, 14, 12)
        self.name = QLineEdit(p.name)
        self.tol_deg = ScrubSpinBox('', 'angle', 1, '°')
        self.tol_deg.set_value(p.analysis.tol_deg)
        self.tol_mm = ScrubSpinBox('', 'plain', 0, 'mm', step=5)
        self.tol_mm.set_value(p.analysis.tol_mm)
        for i, (t, w) in enumerate((('Name', self.name), ('Tolerance angle', self.tol_deg), ('Tolerance slide', self.tol_mm))):
            a = label(t, ink2=True)
            a.setFixedWidth(130)
            g.addWidget(a, i, 0)
            g.addWidget(w, i, 1)
        g.addWidget(label('The tolerance values drive “Worst single-camera error” in the Coverage panel.', muted=True, small=True, wrap=True), 3, 0, 1, 2)
        g.setRowStretch(4, 1)
        cb = button('Cancel', 'ghost')
        cb.clicked.connect(self.reject)
        o = button('Apply', 'primary')
        o.clicked.connect(self.apply)
        self.buttons.addWidget(cb)
        self.buttons.addWidget(o)

    def apply(self):
        n = self.name.text().strip() or 'untitled'
        d, mm = self.tol_deg.value(), self.tol_mm.value()

        def fn(p):
            p.name = n
            p.analysis.tol_deg = d
            p.analysis.tol_mm = mm
        self.ctx.store.edit(['analysis', 'name'], fn, 'Project settings')
        self.ctx.window.set_project_meta()
        self.ctx.analysis.request(force=True)
        self.accept()


# ------------------------------------------------------------------------------------- help / about
HELP = {
    'frames': ('Coordinate frames', """
<h3>ISO 8855</h3>
<p>x forward, y left, z up. Lengths in metres; angles in degrees.</p>
<p>The internal origin is the <b>rear-axle centre on the ground plane</b>. The display frame can be moved to the front axle,
the front bumper, the vehicle centre or a custom point (View ▸ Coordinate frame); axes never change.</p>
<h3>Camera orientation</h3>
<p>Intrinsic Z-Y-X: yaw ψ about z (positive turns left), then pitch θ about the new y (positive tilts <b>down</b>),
then roll φ about the optical axis. The camera body frame has the optical axis along +x.</p>
<p>Exports convert to OpenCV (x right, y down, z forward) for calibration files, and to REP 103 for ROS 2 (identical axes to ISO 8855).</p>"""),
    'lens': ('Lens projection models', """
<p>Image height r for a ray θ off the optical axis, with focal length f:</p>
<table cellpadding=4>
<tr><td>Rectilinear</td><td>r = f · tan θ</td></tr>
<tr><td>Equidistant</td><td>r = f · θ</td></tr>
<tr><td>Equisolid</td><td>r = 2f · sin(θ/2)</td></tr>
<tr><td>Stereographic</td><td>r = 2f · tan(θ/2)</td></tr>
<tr><td>Orthographic</td><td>r = f · sin θ</td></tr>
<tr><td>Measured (Kannala–Brandt)</td><td>r = f · θ (1 + k1θ² + k2θ⁴ + k3θ⁶ + k4θ⁸)</td></tr>
</table>
<p>The built-in STURDeCAM31 lenses use a Kannala–Brandt curve fitted to the datasheet's horizontal, vertical and diagonal FOV
with the datasheet focal length. Pixel density is √(radial × tangential sampling) divided by range, in px/cm.</p>"""),
}


class HelpDialog(VDialog):
    def __init__(self, topic, parent=None):
        title, html = HELP.get(topic, ('Help', ''))
        super().__init__(parent, title, '', (640, 460))
        l = QVBoxLayout(self.body)
        l.setContentsMargins(14, 10, 14, 10)
        b = QTextBrowser()
        b.setHtml(f'<div style="font-family: IBM Plex Sans; font-size: 12px;">{html}</div>')
        b.setStyleSheet('QTextBrowser { font-family: "IBM Plex Sans"; }')
        l.addWidget(b)
        o = button('Close', 'primary')
        o.clicked.connect(self.accept)
        self.buttons.addWidget(o)


class AboutDialog(VDialog):
    def __init__(self, parent=None):
        super().__init__(parent, 'About Vantage', '', (520, 300), resizable=False)
        l = QVBoxLayout(self.body)
        l.setContentsMargins(14, 12, 14, 12)
        from ... import __version__
        l.addWidget(label(f'Vantage {__version__} · sensor rig planner for autonomous-driving data collection', bold=True))
        l.addWidget(label('Plans surround-view camera rigs: placement on the body, ground coverage, pixel density, stitch seams, '
                          'tolerance studies, optimisation, calibration targets and exports.', wrap=True))
        l.addWidget(label('Typeface: IBM Plex Sans and Plex Mono, SIL Open Font License 1.1.', muted=True, small=True, wrap=True))
        l.addWidget(label('Built with PyQt6, NumPy, SciPy and OpenCV.', muted=True, small=True))
        l.addStretch(1)
        o = button('Close', 'primary')
        o.clicked.connect(self.accept)
        self.buttons.addWidget(o)


# ------------------------------------------------------------------------------------- layout import
def import_layout_file(path, project) -> list[Camera]:
    """CSV with columns tag,name,x,y,z,yaw,pitch[,roll,lens,port] or JSON as written by the extrinsics export.
    Units, angles and frame follow the file: the JSON 'frame' block, CSV headers such as x_mm / yaw_rad and a 'frame'
    column; without them, metres and degrees in the current display frame."""
    cams = []
    if path.lower().endswith('.json'):
        with open(path) as fh:
            data = json.load(fh)
        fr = data.get('frame') if isinstance(data, dict) else None
        fr = fr if isinstance(fr, dict) else {}
        L = _LENGTH.get(fr.get('units'), 1.0)
        A = _ANGLE.get(fr.get('angles'), 1.0)
        if isinstance(fr.get('origin_from_rear_axle'), list) and len(fr['origin_from_rear_axle']) == 3:
            o = np.asarray(fr['origin_from_rear_axle'], float) * L
        else:
            o = _frame_origin(project, fr.get('id'))
        rows = data.get('cameras', []) if isinstance(data, dict) else data if isinstance(data, list) else []
        for i, r in enumerate(rows):
            pos = r.get('position', [r.get('x', 0), r.get('y', 0), r.get('z', 1 / L)])
            cams.append(Camera(tag=str(r.get('tag', f'C{i + 1}'))[:4], name=r.get('name', r.get('tag', 'Camera')),
                               location=r.get('name', ''), slot=i % 8, x=float(pos[0]) * L + o[0], y=float(pos[1]) * L + o[1],
                               z=float(pos[2]) * L + o[2], yaw=float(r.get('yaw', 0)) * A, pitch=float(r.get('pitch', 30 / A)) * A,
                               roll=float(r.get('roll', 0)) * A, lens_id=_lens_id(project, r.get('lens')),
                               port=int(r.get('gmsl_port', r.get('port', i + 1)) or 0)))
    else:
        with open(path, newline='') as fh:
            rd = csv.DictReader(fh)
            for i, r in enumerate(rd):
                cols = dict(_column(k, v) for k, v in r.items() if k)

                def text(key, default=''):
                    return cols.get(key, ('', ''))[0].strip() or default
                o = _frame_origin(project, text('frame'))
                port = next((cols[k][0].strip() for k in ('port', 'gmsl_port', 'gmsl') if k in cols), None)
                cams.append(Camera(tag=text('tag', f'C{i + 1}')[:4], name=text('name') or text('tag') or 'Camera',
                                   location=text('name'), slot=i % 8,
                                   x=_metres(cols, 'x', '0') + o[0], y=_metres(cols, 'y', '0') + o[1],
                                   z=_metres(cols, 'z', '1') + o[2], yaw=_degrees(cols, 'yaw', '0'),
                                   pitch=_degrees(cols, 'pitch', '30'), roll=_degrees(cols, 'roll', '0'),
                                   lens_id=_lens_id(project, text('lens')),
                                   port=int(float(port)) if port else 0 if port == '' else i + 1))
    if not cams:
        raise ValueError('No cameras found in the file.')
    if len(cams) > 8:
        raise ValueError(f'{len(cams)} cameras; the maximum is 8.')
    return cams


_LENGTH = {'m': 1.0, 'mm': 1e-3, 'cm': 1e-2, 'in': 0.0254}
_ANGLE = {'deg': 1.0, '°': 1.0, 'rad': 180.0 / math.pi}


def _column(header, value):
    """CSV header and cell -> (name, (text, unit)): 'x_mm' / 'X (mm)' / 'yaw_rad' / 'Yaw °' / 'gmsl_port' give
    x mm, x mm, yaw rad, yaw °, gmsl_port without a unit."""
    h = header.strip().lower()
    mo = re.fullmatch(r'(.+?)(?:[\s_]+\(?|\s*\()(mm|cm|m|in|deg|rad|°)\)?', h)
    name, unit = (mo.group(1).strip(), mo.group(2)) if mo else (h, '')
    return name, (value or '', unit)


def _metres(cols, key, default):
    text, unit = cols.get(key, ('', ''))
    unit = unit if unit in _LENGTH else 'm'
    return parse_number(text.strip() or default, 'length', unit) * _LENGTH[unit]


def _degrees(cols, key, default):
    text, unit = cols.get(key, ('', ''))
    return parse_number(text.strip() or default, 'angle') * _ANGLE.get(unit, 1.0)


def _frame_origin(project, frame_id):
    """World (rear-axle) position of a named frame's origin; the current display frame when the name is unknown."""
    if frame_id not in FRAME_PRESETS:
        return project.origin_offset()
    q = copy.copy(project)          # shallow: only the frame differs
    q.frame = frame_id
    return q.origin_offset()


def _lens_id(project, name):
    if not name:
        return next(iter(project.lenses))
    for lid, l in project.lenses.items():
        if l.name.lower() == str(name).lower() or lid == str(name).lower():
            return lid
    return next(iter(project.lenses))
