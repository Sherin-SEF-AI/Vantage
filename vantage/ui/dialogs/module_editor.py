"""A5 · Camera module editor: datasheet, projection model, distortion, image circle, derived values."""
from __future__ import annotations

import copy
import csv
import json
import math
import os

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QLineEdit, QComboBox, QPushButton,
                             QMenu, QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView, QFileDialog, QMessageBox,
                             QScrollArea, QFrame, QCheckBox)

from ...core.lens import (Module, Lens, PROJECTIONS, PROJECTION_LABEL, r_of_theta, model_fov, derived, fit_kb_to_fov,
                          builtin_modules, builtin_lenses, resolve)
from ...core.mathutil import parse_number
from ...core.project import new_uid
from ..charts import ProjectionPlot
from .. import icons
from ..theme import THEME, mono, STATUS
from ..widgets import label, button, SegmentedControl, ScrubSpinBox, hline, vline, ImageCircleDiagram, ROW_H, StatusLine
from .base import VDialog


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
        m.addAction('OpenCV YAML…', lambda: self.import_file('yaml'))
        self.b_imp.setMenu(m)
        bar.addWidget(self.b_imp)
        bar.addWidget(label('JSON · CSV · OpenCV YAML · or edit the fields below', muted=True))
        bar.addStretch(1)
        bar.addWidget(label('Preset', ink2=True))
        presets = [(l.id, f'{l.name} {l.label}'.strip()) for l in builtin_lenses()] + [('manual', 'Manual')]
        self.preset = SegmentedControl(presets, lid if lid in [x for x, _ in presets] else 'manual')
        self.preset.changed.connect(self.apply_preset)
        bar.addWidget(self.preset)
        body.addLayout(bar)
        body.addWidget(hline())
        cols = QHBoxLayout()
        cols.setSpacing(0)
        # ---- datasheet
        ds = QWidget()
        ds.setFixedWidth(340)
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

        def line(key, text):
            w = QLineEdit()
            w.editingFinished.connect(self.read_fields)
            return field(key, text, w)

        def num(key, text, unit, dp=2, step=0.01):
            w = ScrubSpinBox('', 'plain', dp, unit, step=step)
            w.committed.connect(lambda v: self.read_fields())
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
        line('m_size', 'Size mm')
        num('m_weight', 'Weight', 'g', 0, 1)
        line('m_cables', 'Cable m')
        r += 1
        head('Lens')
        line('l_name', 'Name')
        line('l_label', 'Label')
        num('l_focal', 'Focal length', 'mm', 3, 0.01)
        num('l_fno', 'Aperture', 'F', 1, 0.1)
        num('l_near', 'Focus near', 'm', 2, 0.01)
        line('l_sheet', 'Sheet FOV H V D')
        num('l_maxhalf', 'Image circle θ', '°', 1, 0.5)
        dl.setRowStretch(r, 1)
        cols.addWidget(ds)
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
        bl.addWidget(label('Least-squares fit of k1–k4 to the sheet’s H, V and D field of view at the stated focal length.', muted=True, small=True))
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
        f['m_size'].setText(' × '.join(f'{v:g}' for v in m.size_mm))
        f['m_weight'].set_value(m.weight_g)
        f['m_cables'].setText(' · '.join(f'{v:g}' for v in m.cables_m))
        f['l_name'].setText(l.name)
        f['l_label'].setText(l.label)
        f['l_focal'].set_value(l.focal_mm)
        f['l_fno'].set_value(l.f_number)
        f['l_near'].set_value(l.focus_near_m)
        f['l_sheet'].setText(' · '.join(f'{v:g}' for v in l.sheet_fov))
        f['l_maxhalf'].set_value(derived(m, l)['max_half_deg'])
        self.proj.set_value(l.projection if l.projection != 'orthographic' else 'equisolid')
        self._busy = False

    def read_fields(self):
        if self._busy:
            return
        import re
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
            sz = [float(x) for x in re.findall(r'\d+(?:\.\d+)?', f['m_size'].text())]
            if len(sz) == 3:
                m.size_mm = tuple(sz)
            m.weight_g = f['m_weight'].value()
            cab = [float(x) for x in re.findall(r'\d+(?:\.\d+)?', f['m_cables'].text())]
            if cab:
                m.cables_m = tuple(cab)
            l.name = f['l_name'].text().strip() or l.name
            l.label = f['l_label'].text().strip()
            l.focal_mm = max(0.1, f['l_focal'].value())
            l.f_number = max(0.5, f['l_fno'].value())
            l.focus_near_m = max(0.0, f['l_near'].value())
            sh = [float(x) for x in re.findall(r'\d+(?:\.\d+)?', f['l_sheet'].text())]
            l.sheet_fov = tuple((sh + [0, 0, 0])[:3])
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
        l, m = self.lens, self.mod
        kb = list(l.kb)
        if c < 4:
            kb[c] = v
            l.kb = tuple(kb)
            l.projection = 'kannala_brandt'
        elif c in (4, 5):
            l.focal_mm = v * m.pixel_um * 1e-3
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
        hs = (m.sensor_w_mm / 2, m.sensor_h_mm / 2, math.hypot(m.sensor_w_mm, m.sensor_h_mm) / 2)
        l.kb = fit_kb_to_fov(l.focal_mm, l.sheet_fov, hs, base='equisolid')
        l.projection = 'kannala_brandt'
        self.proj.set_value('kannala_brandt')
        self.update_all()

    def apply_preset(self, key):
        if key == 'manual':
            return
        for lb in builtin_lenses():
            if lb.id == key:
                keep_id = self.lens.id
                self.lens = copy.deepcopy(lb)
                self.lens.id = keep_id
                self.lens.builtin = False
        self.mod = copy.deepcopy(builtin_modules()[0]) if self.mod.builtin or not self.mod.name else self.mod
        self.write_fields()
        self.update_all()

    # ------------------------------------------------------------------ views
    def update_all(self):
        self._busy = True
        m, l = self.mod, self.lens
        p = self.ctx.store.project
        n = len(self.users)
        self.set_subtitle(f'{m.name} + {l.name} · used by {n} camera{"s" if n != 1 else ""} in {p.variant.name}')
        self.ok.setText(f'Apply to {n} camera{"s" if n != 1 else ""}' if n and not self.is_new else 'Add to library')
        d = derived(m, l)
        hw, hh = m.sensor_w_mm / 2, m.sensor_h_mm / 2
        hd = math.hypot(hw, hh)
        tmax = min(95.0, max(d['max_half_deg'], d['fov'][2] / 2) + 6)
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
            pts = [(l.sheet_fov[0] / 2, hw), (l.sheet_fov[1] / 2, hh), (l.sheet_fov[2] / 2, hd)]
        ymax = max(hd * 1.3, d['image_circle_r_mm'] * 1.1)
        self.plot.set_data(curves, pts, [(hw, 'H'), (hh, 'V'), (hd, 'D')], (d['fov'][0] / 2, 'HFOV/2'), tmax, ymax)
        src = f'measured · {len(l.measured)} samples' if l.measured else ('datasheet FOV points' if l.sheet_fov[0] else '')
        self.legend.setText(f'——  {PROJECTION_LABEL[l.projection]} · selected     ■  {src}     ----  other models     |  HFOV / 2 = {d["fov"][0] / 2:.2f}°')
        intr = resolve(m, l)
        vals = list(l.kb) + [intr.f_px, intr.f_px, intr.cx, intr.cy]
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
        self.der_v['FOV H · V · D'][1].setText(f'{d["fov"][0]:.1f} · {d["fov"][1]:.1f} · {d["fov"][2]:.1f} °')
        self.der_v['iFOV at centre'][1].setText(f'{d["ifov_deg"]:.4f} °/px')
        self.der_v['px/° at centre'][1].setText(f'{d["px_per_deg_centre"]:.2f}')
        self.der_v['px/° at H edge'][0].setText(f'px/° at H edge ({d["h_edge_deg"]:.2f}°)')
        self.der_v['px/° at H edge'][1].setText(f'{d["px_per_deg_edge"]:.2f}')
        self.der_v['f in pixels'][1].setText(f'{d["f_px"]:.1f} px')
        self.der_v['Principal point'][1].setText(f'{d["cx"]:.1f} · {d["cy"]:.1f} px')
        self.der_v['Hyperfocal at F'][0].setText(f'Hyperfocal at F{l.f_number:g}')
        self.der_v['Hyperfocal at F'][1].setText(f'{d["hyperfocal_m"]:.2f} m')
        worst = 'good'
        msgs = []
        for i, name in enumerate(('Horizontal', 'Vertical', 'Diagonal')):
            sv = l.sheet_fov[i] if i < len(l.sheet_fov) else 0
            mv = d['fov'][i]
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
        filt = {'json': 'JSON (*.json)', 'csv': 'CSV (*.csv)', 'yaml': 'OpenCV YAML (*.yaml *.yml *.xml)'}[kind]
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
        for k, v in md.items():
            if k in mfields and k not in ('id', 'builtin'):
                setattr(self.mod, k, tuple(v) if isinstance(v, list) else v)
        for k, v in ld.items():
            if k in lfields and k not in ('id', 'builtin', 'module_id'):
                setattr(self.lens, k, tuple(v) if isinstance(v, list) and k != 'measured' else v)
        intr = d.get('intrinsics')
        if isinstance(intr, dict):
            self._apply_intrinsics(intr.get('width'), intr.get('height'), intr.get('fx'), intr.get('cx'), intr.get('cy'), intr.get('k'))

    def _apply_intrinsics(self, w, h, fx, cx, cy, k):
        m, l = self.mod, self.lens
        if w and h:
            m.width_px, m.height_px = int(w), int(h)
        if fx:
            l.focal_mm = float(fx) * m.pixel_um * 1e-3
        if cx is not None and cy is not None:
            l.pp_offset_px = (float(cx) - m.width_px / 2, float(cy) - m.height_px / 2)
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
        import cv2
        fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
        if not fs.isOpened():
            raise ValueError('not an OpenCV FileStorage file')
        K = fs.getNode('camera_matrix').mat()
        D = fs.getNode('distortion_coefficients').mat()
        w = fs.getNode('image_width').real()
        h = fs.getNode('image_height').real()
        model = fs.getNode('distortion_model').string()
        fs.release()
        if K is None:
            raise ValueError('camera_matrix missing')
        k = D.ravel().tolist() if D is not None else None
        if model and model.lower() not in ('fisheye', 'kannala_brandt', 'equidistant') and k is not None:
            raise ValueError(f'distortion model “{model}” is not fisheye; only fisheye (Kannala–Brandt) is supported')
        self._apply_intrinsics(w, h, K[0, 0], K[0, 2], K[1, 2], k)
        self.lens.name = os.path.splitext(os.path.basename(path))[0]
        self.lens.label = 'calibrated'

    def _commit(self, new_ids=False, name=None):
        mod, lens = copy.deepcopy(self.mod), copy.deepcopy(self.lens)
        mod.builtin = False if new_ids else mod.builtin
        lens.builtin = False if new_ids else lens.builtin
        if new_ids:
            lens.id = new_uid()
            if name:
                lens.name = name
            lens.module_id = mod.id if mod.id in self.ctx.store.project.modules else self.mid
        lens.module_id = lens.module_id or self.mid

        def fn(p):
            if mod.id not in p.modules or not new_ids:
                p.modules[mod.id] = mod
            else:
                p.modules[mod.id] = mod
            p.lenses[lens.id] = lens
        self.ctx.store.edit('library', fn, f'{"Added" if new_ids else "Edited"} {mod.name} + {lens.name}')
        self.ctx.analysis.request()
        return lens.id

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
