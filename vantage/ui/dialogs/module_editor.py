"""A5 · Camera module editor: datasheet, projection model, distortion, image circle, derived values.

The Preset list holds the built-in library (e-con Systems modules, each lens under its module): picking one shows that
module and lens, and applying switches the cameras being edited to them. The Source block shows where a module's and
lens's figures come from and which figures the source does not give."""
from __future__ import annotations

import copy
import csv
import html
import json
import math
import os
import re

import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QLineEdit, QComboBox, QPushButton,
                             QMenu, QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView, QFileDialog, QMessageBox,
                             QScrollArea, QFrame, QCheckBox)

from ...core.lens import (Module, Lens, PROJECTIONS, PROJECTION_LABEL, SHUTTERS, LIBRARY_NOTE, r_of_theta, model_fov,
                          derived, fit_sheet as fit_sheet_kb, best_base, half_sizes, builtin_modules, builtin_lenses,
                          builtin_module, builtin_lens, resolve)
from ...core.mathutil import parse_number
from ...core.project import new_uid
from ..charts import ProjectionPlot
from .. import icons
from ..theme import THEME, mono, STATUS
from ..widgets import (fit_width, label, button, SegmentedControl, ScrubSpinBox, hline, vline, ImageCircleDiagram,
                       ROW_H, StatusLine)
from .base import VDialog

# figures that describe a product: an import of another camera starts them blank instead of inheriting the old ones
_PRODUCT_FIELDS = {'vendor': '', 'shutter': '', 'dynamic_range_db': 0.0, 'ip_rating': '', 'temp_c': (), 'chroma': '',
                   'url': '', 'note': ''}


class PresetCombo(QComboBox):
    """The built-in lenses grouped under their modules (module rows are headings), then Manual. The SegmentedControl
    API the editor used before: changed(key), set_value(key), value()."""
    changed = pyqtSignal(str)

    def __init__(self, value='manual', parent=None):
        super().__init__(parent)
        lenses = builtin_lenses()
        for m in builtin_modules():
            mine = [l for l in lenses if l.module_id == m.id]
            if not mine:
                continue
            self.addItem(f'{m.name} · {m.sensor}', None)
            head = self.model().item(self.count() - 1)
            head.setEnabled(False)
            head.setFont(mono(11))
            for l in mine:
                h, v, d = l.sheet_fov
                self.addItem(f'    {m.name} · {l.name} · {l.label} · {h:g}° H', l.id)
                self.setItemData(self.count() - 1, f'{m.name} + {l.name}: {h:g}° H × {v:g}° V × {d:g}° D (e-con)',
                                 Qt.ItemDataRole.ToolTipRole)
        self.addItem('Manual', 'manual')
        self.setMaxVisibleItems(24)
        self.setMinimumWidth(320)
        self.set_value(value)
        self.activated.connect(lambda i: self.changed.emit(self.itemData(i)) if self.itemData(i) else None)

    def keys(self) -> list[str]:
        return [self.itemData(i) for i in range(self.count()) if self.itemData(i)]

    def set_value(self, key):
        i = self.findData(key)
        if i >= 0:
            self.blockSignals(True)
            self.setCurrentIndex(i)
            self.blockSignals(False)

    def value(self):
        return self.currentData()


def _num_or_dash(v, fmt='{:g}'):
    return fmt.format(v) if v and v > 0 else '—'


def _parse_size(text):
    """'25 × 25 × 43.8' (a dimension may be '—': not given) -> (w, h, l) or None when it is not three parts."""
    parts = [p.strip() for p in re.split(r'[×x*,]', text.replace('mm', ''))]
    if len(parts) != 3:
        return None
    out = []
    for p in parts:
        try:
            v = float(p)
        except ValueError:
            v = 0.0                       # '—' / blank: not given
        out.append(v if math.isfinite(v) and v > 0 else 0.0)
    return tuple(out)


def _parse_temp(text):
    """'−40 … 85' -> (-40.0, 85.0); blank or '—' -> () (not given); anything else -> None (left unchanged)."""
    t = text.replace('−', '-').replace('°C', '').replace('°', '').strip()
    if not t or t == '—':
        return ()
    nums = [float(x) for x in re.findall(r'-?\d+(?:\.\d+)?', t)]
    return (min(nums), max(nums)) if len(nums) == 2 else None


class ModuleEditor(VDialog):
    def __init__(self, ctx, mid, lid, parent=None, start_import=False):
        p = ctx.store.project
        self.ctx = ctx
        self.mid, self.lid = mid, lid
        self.mod = copy.deepcopy(p.modules[mid])
        self.lens = copy.deepcopy(p.lenses[lid])
        self.is_new = False
        users = [c for c in p.variant.cameras if c.lens_id == lid and c.module_id == mid]
        super().__init__(parent, 'Camera module', '', (1380, 900))
        self.users = users
        body = QVBoxLayout(self.body)
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        bar = QHBoxLayout()
        bar.setContentsMargins(14, 6, 14, 6)
        self.b_imp = QPushButton('Import')
        m = QMenu(self.b_imp)
        m.addAction('JSON…', lambda: self.import_file('json'))
        m.addAction('CSV…', lambda: self.import_file('csv'))
        m.addAction('OpenCV / ROS / Kalibr YAML…', lambda: self.import_file('yaml'))
        self.b_imp.setMenu(m)
        bar.addWidget(self.b_imp)
        bar.addWidget(label('JSON · CSV · OpenCV YAML · or edit the fields below', muted=True))
        bar.addStretch(1)
        bar.addWidget(label('Preset', ink2=True))
        self.preset = PresetCombo('manual')
        self.preset.set_value(lid if lid in self.preset.keys() else 'manual')
        self.preset.setToolTip('The built-in library: picking a lens shows it with its module; applying switches the '
                               'cameras being edited to them.')
        self.preset.changed.connect(self.apply_preset)
        bar.addWidget(self.preset)
        body.addLayout(bar)
        body.addWidget(hline())
        cols = QHBoxLayout()
        cols.setSpacing(0)
        # ---- datasheet
        ds = QWidget()
        dl = QGridLayout(ds)
        dl.setContentsMargins(14, 10, 14, 10)
        dl.setVerticalSpacing(3)
        dl.addWidget(label('Datasheet', bold=True), 0, 0, 1, 2)
        self.f = {}
        r = 1

        def head(t):
            nonlocal r
            dl.addWidget(label(t, muted=True, small=True), r, 0, 1, 2)
            r += 1

        def field(key, text, w):
            nonlocal r
            a = label(text, ink2=True)
            a.setFixedWidth(108)
            dl.addWidget(a, r, 0)
            dl.addWidget(w, r, 1)
            self.f[key] = w
            r += 1
            return w

        def line(key, text, tip=''):
            w = QLineEdit()
            w.editingFinished.connect(self.read_fields)
            if tip:
                w.setToolTip(tip)
            return field(key, text, w)

        def num(key, text, unit, dp=2, step=0.01, tip=''):
            w = ScrubSpinBox('', 'plain', dp, unit, step=step)
            w.committed.connect(lambda v, key=key: self.read_fields(key))
            if tip:
                w.setToolTip(tip)
            return field(key, text, w)
        head('Module')
        line('m_name', 'Name')
        line('m_vendor', 'Vendor')
        line('m_sensor', 'Sensor')
        line('m_res', 'Resolution')
        num('m_pixel', 'Pixel pitch', 'µm', 2, 0.05)
        line('m_format', 'Optical format')
        line('m_iface', 'Interface')
        num('m_fps', 'Frame rate', 'fps', 0, 1)
        sh = QComboBox()
        for k in SHUTTERS:
            sh.addItem({'': 'Not given', 'rolling': 'Rolling', 'global': 'Global'}[k], k)
        sh.activated.connect(lambda _i: self.read_fields('m_shutter'))
        field('m_shutter', 'Shutter', sh)
        num('m_dr', 'Dynamic range', 'dB', 1, 1, '0 = not given')
        line('m_chroma', 'Colour', 'colour, monochrome or RGB-IR; blank = not given')
        line('m_size', 'Size mm', 'Width × height × length along the optical axis; — = not given')
        num('m_weight', 'Weight', 'g', 1, 1, '0 = not given')
        num('m_power', 'Power', 'W', 3, 0.1, '0 = not given (the power budget then uses its typical camera figure)')
        line('m_ip', 'IP rating', 'e.g. IP67; blank = not given or no enclosure')
        line('m_temp', 'Temperature °C', 'Operating range, e.g. −40 … 85; blank = not given')
        line('m_cables', 'Cable m')
        r += 1
        head('Lens')
        line('l_name', 'Name')
        line('l_label', 'Label')
        num('l_focal', 'Focal length', 'mm', 3, 0.01)
        num('l_fno', 'Aperture', 'F', 1, 0.1, '0 = not given')
        num('l_near', 'Focus near', 'm', 2, 0.01, '0 = not given (no near limit)')
        line('l_far', 'Focus far m', '∞ or a distance in metres')
        line('l_sheet', 'Sheet FOV H V D')
        num('l_maxhalf', 'Image circle θ', '°', 1, 0.5)
        r += 1
        head('Source')
        self.source = label('', muted=True, small=True, wrap=True)
        self.source.setTextFormat(Qt.TextFormat.RichText)
        self.source.setOpenExternalLinks(True)
        self.source.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
        self.source.setToolTip(LIBRARY_NOTE)
        dl.addWidget(self.source, r, 0, 1, 2)
        r += 1
        dl.setRowStretch(r, 1)
        sc = QScrollArea()
        sc.setWidgetResizable(True)
        sc.setFrameShape(QFrame.Shape.NoFrame)
        sc.setWidget(ds)
        fit_width(sc)
        sc.setFixedWidth(356)
        cols.addWidget(sc)
        cols.addWidget(vline())
        # ---- projection
        pw = QWidget()
        pl = QVBoxLayout(pw)
        pl.setContentsMargins(14, 10, 14, 10)
        ph = QHBoxLayout()
        ph.addWidget(label('Projection model', bold=True))
        ph.addStretch(1)
        self.proj = SegmentedControl([(k, PROJECTION_LABEL[k]) for k in PROJECTIONS if k != 'orthographic'], self.lens.projection)
        self.proj.changed.connect(self._proj)
        ph.addWidget(self.proj)
        pl.addLayout(ph)
        self.plot = ProjectionPlot()
        pl.addWidget(self.plot, 1)
        self.legend = label('', muted=True, small=True)
        pl.addWidget(self.legend)
        dh = QHBoxLayout()
        dh.addWidget(label('Distortion · Kannala–Brandt k1–k4', bold=True))
        dh.addStretch(1)
        self.fit_lbl = label('', muted=True, mono_=True, small=True)
        dh.addWidget(self.fit_lbl)
        pl.addLayout(dh)
        self.kb = QTableWidget(1, 8)
        self.kb.setHorizontalHeaderLabels(['k1', 'k2', 'k3', 'k4', 'fx px', 'fy px', 'cx px', 'cy px'])
        self.kb.verticalHeader().setVisible(False)
        self.kb.verticalHeader().setDefaultSectionSize(ROW_H)
        self.kb.horizontalHeader().setFixedHeight(ROW_H)
        self.kb.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.kb.setFixedHeight(ROW_H * 2 + 4)
        self.kb.setShowGrid(False)
        self.kb.itemChanged.connect(self._kb_edit)
        pl.addWidget(self.kb)
        bl = QHBoxLayout()
        b_fit = button('Fit to datasheet FOV')
        b_fit.clicked.connect(self.fit_sheet)
        bl.addWidget(b_fit)
        bl.addWidget(label('Fits k1–k4 to the sheet’s H, V and D field of view at this focal length.', muted=True,
                           small=True, wrap=True), 1)
        bl.addStretch(1)
        pl.addLayout(bl)
        cols.addWidget(pw, 1)
        cols.addWidget(vline())
        # ---- image circle + derived
        rw = QWidget()
        rw.setFixedWidth(340)
        rl = QVBoxLayout(rw)
        rl.setContentsMargins(14, 10, 14, 10)
        rl.addWidget(label('Image circle on sensor', bold=True))
        self.circle = ImageCircleDiagram(detailed=True)
        self.circle.setFixedHeight(270)
        rl.addWidget(self.circle)
        self.circle_note = label('', muted=True, small=True, wrap=True)
        rl.addWidget(self.circle_note)
        rl.addSpacing(6)
        rl.addWidget(label('Derived', bold=True))
        self.der = QGridLayout()
        self.der.setVerticalSpacing(2)
        self.der_v = {}
        for i, k in enumerate(('FOV H · V · D', 'iFOV at centre', 'px/° at centre', 'px/° at H edge', 'f in pixels', 'Principal point',
                               'Hyperfocal at F')):
            a = label(k, ink2=True)
            v = label('', mono_=True)
            v.setAlignment(Qt.AlignmentFlag.AlignRight)
            self.der.addWidget(a, i, 0)
            self.der.addWidget(v, i, 1)
            self.der_v[k] = (a, v)
        rl.addLayout(self.der)
        rl.addSpacing(6)
        rl.addWidget(label('Datasheet vs model', bold=True))
        self.cmp = QTableWidget(3, 4)
        self.cmp.setHorizontalHeaderLabels(['FOV', 'sheet', 'model', 'Δ'])
        self.cmp.verticalHeader().setVisible(False)
        self.cmp.verticalHeader().setDefaultSectionSize(ROW_H)
        self.cmp.horizontalHeader().setFixedHeight(ROW_H)
        self.cmp.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.cmp.setShowGrid(False)
        self.cmp.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.cmp.setFixedHeight(ROW_H * 4 + 4)
        rl.addWidget(self.cmp)
        self.warn = StatusLine('good', '')
        rl.addWidget(self.warn)
        rl.addStretch(1)
        cols.addWidget(rw)
        body.addLayout(cols, 1)
        self.hint.setText('Changes apply to every camera using this module. Esc cancels, Enter applies.')
        c = button('Cancel', 'ghost')
        c.clicked.connect(self.reject)
        sp = button('Save as preset…')
        sp.clicked.connect(self.save_preset)
        self.ok = button('', 'primary')
        self.ok.clicked.connect(self.apply)
        self.ok.setDefault(True)
        self.buttons.addWidget(c)
        self.buttons.addWidget(sp)
        self.buttons.addWidget(self.ok)
        self._busy = False
        self.write_fields()
        self.update_all()
        if start_import:
            self.is_new = True
            self.import_file('json')

    # ------------------------------------------------------------------ fields
    def write_fields(self):
        self._busy = True
        m, l = self.mod, self.lens
        f = self.f
        f['m_name'].setText(m.name)
        f['m_vendor'].setText(m.vendor)
        f['m_sensor'].setText(m.sensor)
        f['m_res'].setText(f'{m.width_px} × {m.height_px}')
        f['m_pixel'].set_value(m.pixel_um)
        f['m_format'].setText(m.optical_format)
        f['m_iface'].setText(m.interface)
        f['m_fps'].set_value(m.fps)
        f['m_shutter'].setCurrentIndex(max(0, f['m_shutter'].findData(m.shutter if m.shutter in SHUTTERS else '')))
        f['m_dr'].set_value(m.dynamic_range_db)
        f['m_chroma'].setText(m.chroma)
        f['m_size'].setText(' × '.join(_num_or_dash(v) for v in m.size_mm))
        f['m_weight'].set_value(m.weight_g)
        f['m_power'].set_value(m.power_w)
        f['m_ip'].setText(m.ip_rating)
        f['m_temp'].setText(f'{m.temp_c[0]:g} … {m.temp_c[1]:g}'.replace('-', '−') if len(m.temp_c) == 2 else '')
        f['m_cables'].setText(' · '.join(f'{v:g}' for v in m.cables_m))
        f['l_name'].setText(l.name)
        f['l_label'].setText(l.label)
        f['l_focal'].set_value(l.focal_mm)
        f['l_fno'].set_value(l.f_number)
        f['l_near'].set_value(l.focus_near_m)
        f['l_far'].setText('∞' if math.isinf(l.focus_far_m) else f'{l.focus_far_m:g}')
        f['l_sheet'].setText(' · '.join(f'{v:g}' for v in l.sheet_fov))
        f['l_maxhalf'].set_value(derived(m, l)['max_half_deg'])
        self.proj.set_value(l.projection if l.projection != 'orthographic' else 'equisolid')
        self._busy = False

    def read_fields(self, edited=None):
        if self._busy:
            return
        m, l = self.mod, self.lens
        f = self.f
        try:
            m.name = f['m_name'].text().strip() or m.name
            m.vendor = f['m_vendor'].text().strip()
            m.sensor = f['m_sensor'].text().strip()
            nums = [int(x) for x in re.findall(r'\d+', f['m_res'].text())]
            if len(nums) >= 2:
                m.width_px, m.height_px = nums[0], nums[1]
            m.pixel_um = max(0.1, f['m_pixel'].value())
            m.optical_format = f['m_format'].text().strip()
            m.interface = f['m_iface'].text().strip()
            m.fps = max(1.0, f['m_fps'].value())
            m.shutter = f['m_shutter'].currentData() or ''
            m.dynamic_range_db = max(0.0, f['m_dr'].value())
            m.chroma = f['m_chroma'].text().strip()
            sz = _parse_size(f['m_size'].text())
            if sz is not None:
                m.size_mm = sz
            m.weight_g = max(0.0, f['m_weight'].value())
            m.power_w = max(0.0, f['m_power'].value())
            m.ip_rating = f['m_ip'].text().strip()
            tc = _parse_temp(f['m_temp'].text())
            if tc is not None:
                m.temp_c = tc
            cab = [float(x) for x in re.findall(r'\d+(?:\.\d+)?', f['m_cables'].text())]
            if cab:
                m.cables_m = tuple(cab)
            l.name = f['l_name'].text().strip() or l.name
            l.label = f['l_label'].text().strip()
            l.focal_mm = max(0.1, f['l_focal'].value())
            fno = f['l_fno'].value()
            l.f_number = 0.0 if fno <= 0 else max(0.5, fno)         # 0 = not given
            l.focus_near_m = max(0.0, f['l_near'].value())
            far = re.findall(r'\d+(?:\.\d+)?', f['l_far'].text())          # '∞' / blank: infinity
            l.focus_far_m = float(far[0]) if far and float(far[0]) > l.focus_near_m else math.inf
            sh = [float(x) for x in re.findall(r'\d+(?:\.\d+)?', f['l_sheet'].text())]
            l.sheet_fov = tuple((sh + [0, 0, 0])[:3])
            if edited == 'l_maxhalf':
                # the field otherwise shows the derived limit, rounded: storing it would freeze the limit
                l.max_half_deg = f['l_maxhalf'].value()
        except Exception as e:
            QMessageBox.warning(self, 'Could not read a field', str(e))
        self.preset.set_value('manual')
        self.update_all()

    def _proj(self, k):
        self.lens.projection = k
        self.update_all()

    def _kb_edit(self, it):
        if self._busy:
            return
        c = it.column()
        try:
            v = float(it.text().replace('−', '-'))
        except ValueError:
            self.update_all()
            return
        if not math.isfinite(v) or (c in (4, 5) and v <= 0):
            self.update_all()
            return
        l, m = self.lens, self.mod
        kb = list(l.kb)
        if c < 4:
            kb[c] = v
            l.kb = tuple(kb)
            l.projection = 'kannala_brandt'
        elif c == 4:
            l.focal_mm = v * m.pixel_um * 1e-3
        elif c == 5:
            l.aspect = v / (l.focal_mm / (m.pixel_um * 1e-3))       # fy / fx
        elif c == 6:
            l.pp_offset_px = (v - m.width_px / 2, l.pp_offset_px[1])
        elif c == 7:
            l.pp_offset_px = (l.pp_offset_px[0], v - m.height_px / 2)
        self.write_fields()
        self.update_all()

    def fit_sheet(self):
        m, l = self.mod, self.lens
        if not l.sheet_fov[0]:
            QMessageBox.information(self, 'No datasheet FOV', 'Enter the sheet FOV (H V D) first.')
            return
        hs = half_sizes(m)
        sheet = tuple(l.sheet_fov)
        if l.max_half_deg > 0 and sheet[2] and sheet[2] / 2 >= l.max_half_deg - 1e-6:
            sheet = (sheet[0], sheet[1], 0.0)        # the stated D is the image circle (it limits the image)
        l.kb = fit_sheet_kb(l.focal_mm, sheet, hs, base=best_base(sheet, hs)[0])
        l.projection = 'kannala_brandt'
        self.proj.set_value('kannala_brandt')
        self.update_all()

    def apply_preset(self, key):
        """Show a library lens and its module (the project's own copies when it has them, else the built-in ones).
        Applying then switches the cameras being edited to them (_commit)."""
        if key == 'manual':
            return
        p = self.ctx.store.project
        lens = p.lenses.get(key) or builtin_lens(key)
        if lens is None:
            return
        mod = p.modules.get(lens.module_id) or builtin_module(lens.module_id)
        if mod is None:
            return
        self.lens, self.mod = copy.deepcopy(lens), copy.deepcopy(mod)
        self.write_fields()
        self.update_all()

    def switching(self) -> bool:
        """True when applying moves the cameras to another library module / lens (a preset was picked)."""
        return not self.is_new and (self.mod.id, self.lens.id) != (self.mid, self.lid)

    # ------------------------------------------------------------------ views
    def update_all(self):
        self._busy = True
        m, l = self.mod, self.lens
        p = self.ctx.store.project
        n = len(self.users)
        s = 's' if n != 1 else ''
        self.set_subtitle(f'{m.name} + {l.name} · used by {n} camera{s} in {p.variant.name}')
        if self.is_new or not n:
            self.ok.setText('Add to library')
        elif self.switching():
            self.ok.setText(f'Switch {n} camera{s} to {m.name} + {l.name}')
        else:
            self.ok.setText(f'Apply to {n} camera{s}')
        d = derived(m, l)
        seen = model_fov(m, l, circle=True)          # what the image shows: the image circle may cut the corners
        circle_cut = any(abs(a - b) > 0.05 for a, b in zip(seen, d['fov']))
        hw, hh = m.sensor_w_mm / 2, m.sensor_h_mm / 2
        hd = math.hypot(hw, hh)
        tmax = min(120.0, max(d['max_half_deg'], seen[2] / 2) + 6)
        th = np.linspace(0, tmax, 160)
        curves = []
        for k in PROJECTIONS:
            if k == 'orthographic':
                continue
            if k == 'kannala_brandt' and not any(l.kb):
                continue
            r = r_of_theta(np.radians(th), l.focal_mm, k, l.kb)
            curves.append((PROJECTION_LABEL[k], th, np.where(np.abs(r) < 50, r, np.nan), 'sel' if k == l.projection else 'other'))
        pts = [(t, r) for t, r in l.measured]
        if not pts and l.sheet_fov[0]:
            # a stated D the image circle stands for (the corners are dark) is not a point on the curve
            pts = [(f / 2, h) for i, (f, h) in enumerate(zip(l.sheet_fov, (hw, hh, hd)))
                   if f and not (i == 2 and circle_cut)]
        ymax = max(hd * 1.3, d['image_circle_r_mm'] * 1.1)
        self.plot.set_data(curves, pts, [(hw, 'H'), (hh, 'V'), (hd, 'D')], (d['fov'][0] / 2, 'HFOV/2'), tmax, ymax)
        src = f'measured · {len(l.measured)} samples' if l.measured else ('datasheet FOV points' if l.sheet_fov[0] else '')
        self.legend.setText(f'——  {PROJECTION_LABEL[l.projection]} · selected     ■  {src}     ----  other models     |  HFOV / 2 = {d["fov"][0] / 2:.2f}°')
        intr = resolve(m, l)
        vals = list(l.kb) + [intr.f_px, intr.f_px * intr.aspect, intr.cx, intr.cy]
        for i, v in enumerate(vals):
            it = QTableWidgetItem(f'{v:.4f}' if i < 4 else f'{v:.1f}')
            it.setFont(mono(12))
            it.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.kb.setItem(0, i, it)
        self.fit_lbl.setText(f'fit RMS {l.fit_rms_px:.2f} px · {m.width_px} × {m.height_px}' if l.fit_rms_px else f'{m.width_px} × {m.height_px}')
        self.circle.set(m.sensor_w_mm, m.sensor_h_mm, d['image_circle_r_mm'], d['max_half_deg'], m.width_px, m.height_px)
        clears = d['image_circle_r_mm'] >= hd
        self.circle_note.setText(f'The image circle {"clears" if clears else "does not cover"} the {hd:.3f} mm half-diagonal'
                                 + ('.' if clears else ': the sensor corners are dark.'))
        self.der_v['FOV H · V · D'][0].setText('FOV H · V · D, circle' if circle_cut else 'FOV H · V · D')
        self.der_v['FOV H · V · D'][1].setText(f'{seen[0]:.1f} · {seen[1]:.1f} · {seen[2]:.1f} °')
        self.der_v['FOV H · V · D'][1].setToolTip(
            f'The projection reaches {d["fov"][0]:.1f} · {d["fov"][1]:.1f} · {d["fov"][2]:.1f}° at the sensor edges; '
            f'the image circle ({2 * d["max_half_deg"]:.1f}°) limits what the image shows.' if circle_cut else '')
        self.der_v['iFOV at centre'][1].setText(f'{d["ifov_deg"]:.4f} °/px')
        self.der_v['px/° at centre'][1].setText(f'{d["px_per_deg_centre"]:.2f}')
        self.der_v['px/° at H edge'][0].setText(f'px/° at H edge ({d["h_edge_deg"]:.2f}°)')
        self.der_v['px/° at H edge'][1].setText(f'{d["px_per_deg_edge"]:.2f}')
        self.der_v['f in pixels'][1].setText(f'{d["f_px"]:.1f} px')
        self.der_v['Principal point'][1].setText(f'{d["cx"]:.1f} · {d["cy"]:.1f} px')
        if d['hyperfocal_m'] is None:
            self.der_v['Hyperfocal at F'][0].setText('Hyperfocal')
            self.der_v['Hyperfocal at F'][1].setText('— (no aperture)')
        else:
            self.der_v['Hyperfocal at F'][0].setText(f'Hyperfocal at F{l.f_number:g}')
            self.der_v['Hyperfocal at F'][1].setText(f'{d["hyperfocal_m"]:.2f} m')
        self._source_text()
        worst = 'good'
        msgs = []
        for i, name in enumerate(('Horizontal', 'Vertical', 'Diagonal')):
            sv = l.sheet_fov[i] if i < len(l.sheet_fov) else 0
            mv = seen[i]
            a = QTableWidgetItem(name)
            if sv:
                dlt = mv - sv
                sev = 'good' if abs(dlt) < 1.0 else 'warning' if abs(dlt) < 5 else 'serious'
                if sev != 'good':
                    msgs.append(name.lower())
                if ['good', 'warning', 'serious'].index(sev) > ['good', 'warning', 'serious'].index(worst):
                    worst = sev
                a.setIcon(icons.status_icon(sev, 12))
                vals = [f'{sv:.1f}', f'{mv:.1f}', f'{dlt:+.1f}'.replace('-', '−')]
            else:
                vals = ['—', f'{mv:.1f}', '—']
            self.cmp.setItem(i, 0, a)
            for j, v in enumerate(vals):
                it = QTableWidgetItem(v)
                it.setFont(mono(12))
                it.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self.cmp.setItem(i, j + 1, it)
        if msgs:
            self.warn.set(worst, f'{worst.capitalize()} · the {" and ".join(msgs)} FOV differ from the sheet by more than 1°. '
                                 'Analysis uses the selected model; try Measured and Fit to datasheet FOV.')
        else:
            self.warn.set('good', 'The model matches the datasheet FOV within 1°.' if l.sheet_fov[0] else 'No datasheet FOV to compare.')
        self._busy = False

    # ------------------------------------------------------------------ import / apply
    def import_file(self, kind):
        filt = {'json': 'JSON (*.json)', 'csv': 'CSV (*.csv)', 'yaml': 'Calibration YAML (*.yaml *.yml *.xml)'}[kind]
        path, _ = QFileDialog.getOpenFileName(self, 'Import camera module', os.path.expanduser('~'), filt)
        if not path:
            if self.is_new and not self.isVisible():
                pass
            return
        try:
            if kind == 'json':
                self._import_json(path)
            elif kind == 'csv':
                self._import_csv(path)
            else:
                self._import_cv(path)
        except Exception as e:
            QMessageBox.critical(self, 'Import failed', f'{os.path.basename(path)}: {e}')
            return
        self.is_new = True
        self.preset.set_value('manual')
        self.write_fields()
        self.update_all()

    def _apply_dict(self, d):
        mfields = Module.__dataclass_fields__
        lfields = Lens.__dataclass_fields__
        md = d.get('module', d)
        ld = d.get('lens', d)
        md = md if isinstance(md, dict) else {}
        ld = ld if isinstance(ld, dict) else {}
        if md.get('name') and md.get('name') != self.mod.name:
            # another product: what the old one's source said about it (vendor, shutter, IP, note...) does not carry over
            for k, v in _PRODUCT_FIELDS.items():
                setattr(self.mod, k, v)
        if ld.get('name') and ld.get('name') != self.lens.name:
            self.lens.note = ''
        mod = self.mod.to_dict()
        mod.update({k: v for k, v in md.items() if k in mfields and k not in ('id', 'builtin')})
        self.mod = Module.from_dict(mod)              # coerces the file's lists and drops junk in the newer fields
        for k, v in ld.items():
            if k in lfields and k not in ('id', 'builtin', 'module_id'):
                setattr(self.lens, k, tuple(v) if isinstance(v, list) and k != 'measured' else v)
        intr = d.get('intrinsics')
        if isinstance(intr, dict):
            self._apply_intrinsics(intr.get('width'), intr.get('height'), intr.get('fx'), intr.get('cx'), intr.get('cy'), intr.get('k'), intr.get('fy'))

    def _apply_intrinsics(self, w, h, fx, cx, cy, k, fy=None):
        m, l = self.mod, self.lens
        if w and h:
            m.width_px, m.height_px = int(w), int(h)
        if fx:
            l.focal_mm = float(fx) * m.pixel_um * 1e-3
            l.aspect = float(fy) / float(fx) if fy else 1.0
        if cx is not None and cy is not None:
            # OpenCV pixel coordinates: the centre of a 1920-wide sensor is 959.5
            l.pp_offset_px = (float(cx) + 0.5 - m.width_px / 2, float(cy) + 0.5 - m.height_px / 2)
        if k is not None and len(k) >= 4:
            l.kb = tuple(float(x) for x in list(k)[:4])
            l.projection = 'kannala_brandt'

    def _import_json(self, path):
        with open(path) as fh:
            d = json.load(fh)
        if 'cameras' in d and d['cameras']:
            d = d['cameras'][0]
        self._apply_dict(d)

    def _import_csv(self, path):
        d = {}
        with open(path, newline='') as fh:
            for row in csv.reader(fh):
                if len(row) >= 2 and row[0].strip() and not row[0].startswith('#'):
                    k, v = row[0].strip(), row[1].strip()
                    try:
                        v = json.loads(v)
                    except Exception:
                        pass
                    d[k] = v
        self._apply_dict(d)

    def _import_cv(self, path):
        from ...core.calib import read_file
        r = read_file(path)[0]
        self._apply_intrinsics(r['width'], r['height'], r['fx'], r['cx'], r['cy'], r['k'], r['fy'])
        self.lens.name = os.path.splitext(os.path.basename(path))[0]
        self.lens.label = 'calibrated'
        self.lens.note = f'Imported from {os.path.basename(path)}.'

    def _commit(self, new_ids=False, name=None):
        mod, lens = copy.deepcopy(self.mod), copy.deepcopy(self.lens)
        switch = not new_ids and self.switching()
        if new_ids:
            lens.id = new_uid()
            if name:
                lens.name = name
            # an imported or edited module is added beside the one it started from, never written over it
            old = self.ctx.store.project.modules.get(mod.id)
            if old is not None and old.to_dict() != mod.to_dict():
                mod.id = new_uid()
            lens.module_id = mod.id
            mod.builtin = lens.builtin = False
        lens.module_id = lens.module_id or self.mid
        mid, lid = self.mid, self.lid

        def fn(p):
            if not (new_ids and mod.id in p.modules):      # unchanged: the new lens joins the module as it is
                p.modules[mod.id] = mod
            p.lenses[lens.id] = lens
            if switch:                                    # a preset: every camera on (mid, lid) moves to it
                for v in p.variants:
                    for c in v.cameras:
                        if c.module_id == mid and c.lens_id == lid:
                            c.module_id, c.lens_id = mod.id, lens.id
        if switch:
            n = len(self.users)
            self.ctx.store.edit(['library', 'variants'], fn, f'Switched {n} camera{"s" if n != 1 else ""} to '
                                f'{mod.name} + {lens.name}', sections={'library', 'variants', 'variant'})
        else:
            self.ctx.store.edit('library', fn, f'{"Added" if new_ids else "Edited"} {mod.name} + {lens.name}')
        self.ctx.analysis.request()
        return lens.id

    def _source_text(self):
        """Where the module's and lens's figures come from and what the source does not give."""
        m, l = self.mod, self.lens

        def link(mt):
            url = mt.group(1).rstrip('.,')
            short = re.sub(r'^https?://(www\.)?', '', url)
            if short.count('/') > 1:
                short = short.split('/')[0] + '/…/' + short.rsplit('/', 1)[1]
            return f'<a href="{url}" style="color: {THEME.tok["ink2"]}">{short}</a>' + mt.group(1)[len(url):]

        def para(text):
            return re.sub(r'(https?://[^\s)<]+)', link, html.escape(text, quote=False))
        parts = []
        if m.note:
            parts.append(para(m.note))
        if l.note:
            parts.append(para(l.note))
        if not parts:
            parts.append('No source recorded for this module and lens.')
        miss = m.missing() + l.missing()
        if miss:
            parts.append('Not given: ' + html.escape(', '.join(miss)) + '.')
        self.source.setText('<br><br>'.join(parts))

    def apply(self):
        self._commit(new_ids=self.is_new)
        self.accept()

    def save_preset(self):
        from .misc import ask_text
        name = ask_text(self, 'Save as preset', 'Lens name', f'{self.lens.name} copy')
        if not name:
            return
        self._commit(new_ids=True, name=name)
        self.accept()
