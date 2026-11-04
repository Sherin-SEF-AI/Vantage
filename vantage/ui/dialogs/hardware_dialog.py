"""New custom hardware…: a spec sheet for a camera module, lens, lidar, radar, GNSS antenna, GNSS/INS, IMU or a piece of
rig hardware (compute, recorder, switch, time server, DC-DC converter), built from core/hardware.py's registry: each
field with its unit, whether it is required and what in Vantage reads it; live validation (errors block Add, warnings
are shown); load a filled spec sheet (JSON), save a blank one, duplicate a library item; Add puts it into the project
as one undo step (a rig item is selected for its role, so the studies use it at once)."""
from __future__ import annotations

import json

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QFileDialog, QFormLayout, QHBoxLayout, QLineEdit, QScrollArea,
                             QVBoxLayout, QWidget)

from ...core import hardware as HW
from ..widgets import button, label
from .base import VDialog


def _fmt(v):
    if v is None:
        return ''
    if isinstance(v, (list, tuple)):
        return json.dumps(list(v)) if v and isinstance(v[0], (list, tuple)) else ', '.join(f'{x:g}' if isinstance(
            x, (int, float)) else str(x) for x in v)
    if isinstance(v, float):
        return f'{v:g}'
    return str(v)


class HardwareDialog(VDialog):
    def __init__(self, ctx, parent=None, kind='lidar', spec=None):
        super().__init__(parent, 'New custom hardware', 'The figures Vantage reads, with who reads them', (760, 720))
        self.ctx = ctx
        self.kind = kind
        self.inputs = {}
        lay = QVBoxLayout(self.body)
        lay.setContentsMargins(14, 10, 14, 10)
        top = QHBoxLayout()
        top.addWidget(label('Kind', ink2=True))
        self.kind_box = QComboBox()
        for k in HW.kinds():
            self.kind_box.addItem(HW.KIND_LABEL[k], k)
        self.kind_box.setCurrentIndex(max(0, self.kind_box.findData(kind)))
        self.kind_box.currentIndexChanged.connect(lambda i: self.set_kind(self.kind_box.itemData(i)))
        top.addWidget(self.kind_box, 1)
        self.dup_box = QComboBox()
        self.dup_box.setToolTip('Start from a library item: its figures are copied into the sheet')
        self.dup_box.activated.connect(lambda i: self.duplicate(self.dup_box.itemData(i)) if i > 0 else None)
        top.addWidget(self.dup_box, 1)
        lay.addLayout(top)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        lay.addWidget(self.scroll, 1)
        self.msg = label('', wrap=True)
        self.msg.setTextFormat(Qt.TextFormat.PlainText)
        lay.addWidget(self.msg)
        b_load = button('Load spec sheet…', tip='A JSON sheet as Save blank sheet writes it, filled in')
        b_load.clicked.connect(lambda: self.load_sheet())
        b_save = button('Save blank sheet…', tip='Every field with its unit, range and who reads it')
        b_save.clicked.connect(lambda: self.save_template())
        self.b_add = button('Add to project', 'primary')
        self.b_add.clicked.connect(self.add)
        b_close = button('Close', 'ghost')
        b_close.clicked.connect(self.reject)
        for b in (b_load, b_save):
            self.buttons.addWidget(b)
        self.buttons.addStretch(1)
        for b in (b_close, self.b_add):
            self.buttons.addWidget(b)
        self.set_kind(kind)
        if spec:
            self.set_values(spec)

    # ------------------------------------------------------------------ the form
    def set_kind(self, kind):
        self.kind = kind
        if self.kind_box.itemData(self.kind_box.currentIndex()) != kind:
            self.kind_box.blockSignals(True)
            self.kind_box.setCurrentIndex(self.kind_box.findData(kind))
            self.kind_box.blockSignals(False)
        w = QWidget()
        form = QFormLayout(w)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self.inputs = {}
        for f in HW.fields(kind):
            if f.type == 'bool':
                ed = QCheckBox()
                ed.setChecked(bool(f.default))
                ed.toggled.connect(self._check)
            elif f.type == 'choice':
                ed = QComboBox()
                for c in f.choices:
                    ed.addItem(c or '(not stated)', c)
                if f.default is not None:
                    ed.setCurrentIndex(max(0, ed.findData(f.default)))
                ed.currentIndexChanged.connect(self._check)
            else:
                ed = QLineEdit()
                ed.setPlaceholderText(_fmt(f.default) if f.default is not None else
                                      ('required' if f.required else 'not stated'))
                ed.textChanged.connect(self._check)
            tip = f.help + ('\n' if f.help else '') + 'Used by: ' + ', '.join(f.used_by or ('the library',))
            if f.lo is not None or f.hi is not None:
                tip += f'\nRange: {f.lo:g} … {f.hi:g} {f.unit}'
            ed.setToolTip(tip)
            name = f'{f.label}{" *" if f.required else ""}' + (f' ({f.unit})' if f.unit else '')
            lab = label(name, ink2=not f.required)
            lab.setToolTip(tip)
            form.addRow(lab, ed)
            self.inputs[f.key] = (f, ed)
        self.scroll.setWidget(w)
        self._fill_duplicates()
        self._check()

    def _fill_duplicates(self):
        self.dup_box.blockSignals(True)
        self.dup_box.clear()
        self.dup_box.addItem('Duplicate from the library…', None)
        for iid, name in self._library():
            self.dup_box.addItem(name, iid)
        self.dup_box.blockSignals(False)
        self.dup_box.setVisible(self.dup_box.count() > 1)

    def _library(self):
        from ...core.lens import builtin_lenses, builtin_modules
        from ...core.sensors import builtin_sensor_models
        k = self.kind
        if k == 'camera_module':
            return [(m.id, f'{m.vendor} {m.name}') for m in builtin_modules()]
        if k == 'lens':
            return [(ln.id, ln.name) for ln in builtin_lenses()]
        sk = {'ins': 'imu'}.get(k, k)
        out = []
        for m in builtin_sensor_models():
            if m.kind != sk or (k == 'ins' and m.antennas == 0) or (k == 'imu' and m.antennas > 0):
                continue
            out.append((m.id, f'{m.vendor} {m.name}'))
        return out

    def duplicate(self, iid):
        from ...core.lens import builtin_lenses, builtin_modules
        from ...core.sensors import builtin_sensor_models
        pool = {m.id: m for m in builtin_modules()} if self.kind == 'camera_module' else \
            {ln.id: ln for ln in builtin_lenses()} if self.kind == 'lens' else \
            {m.id: m for m in builtin_sensor_models()}
        obj = pool.get(iid)
        if obj is None:
            return
        spec = HW.to_spec(obj, self.kind)
        spec['id'] = ''
        spec['name'] = f'{spec.get("name")} (custom)'
        self.set_values(spec)

    def set_values(self, spec):
        _, spec = HW.unwrap(spec)
        for key, (f, ed) in self.inputs.items():
            v = spec.get(key)
            if isinstance(ed, QCheckBox):
                ed.setChecked(bool(v) if v is not None else bool(f.default))
            elif isinstance(ed, QComboBox):
                ed.setCurrentIndex(max(0, ed.findData(v if v is not None else (f.default or ''))))
            else:
                ed.setText(_fmt(v))
        self._check()

    def values(self) -> dict:
        out = {}
        for key, (f, ed) in self.inputs.items():
            if isinstance(ed, QCheckBox):
                out[key] = ed.isChecked()
                continue
            if isinstance(ed, QComboBox):
                out[key] = ed.currentData()
                continue
            t = ed.text().strip()
            if not t:
                out[key] = None
            elif f.type in ('float', 'int'):
                try:
                    x = float(t.replace(',', ''))
                    out[key] = int(x) if f.type == 'int' and x == int(x) else x
                except ValueError:
                    out[key] = t
            elif f.type in ('vec2', 'vec3', 'list'):
                try:
                    out[key] = [float(x) for x in t.replace(';', ',').split(',') if x.strip()]
                except ValueError:
                    out[key] = t
            elif f.type == 'zones':
                try:
                    out[key] = json.loads(t)
                except ValueError:
                    out[key] = t
            else:
                out[key] = t
        return out

    def _check(self, *_):
        errs, warns = HW.validate(self.kind, self.values())
        self.b_add.setEnabled(not errs)
        lines = [f'✗ {e}' for e in errs] + [f'! {w}' for w in warns]
        self.msg.setText('\n'.join(lines) if lines else 'Ready to add.')

    # ------------------------------------------------------------------ files
    def save_template(self, path=None):
        if not path:
            path, _ = QFileDialog.getSaveFileName(self, 'Save blank spec sheet', f'{self.kind}.json', 'JSON (*.json)')
        if path:
            with open(path, 'w', encoding='utf-8') as fh:
                fh.write(HW.dump_template(self.kind))
        return path

    def load_sheet(self, path=None):
        if not path:
            path, _ = QFileDialog.getOpenFileName(self, 'Load spec sheet', '', 'JSON (*.json)')
        if not path:
            return
        with open(path, encoding='utf-8') as fh:
            data = json.load(fh)
        kind, spec = HW.unwrap(data)
        if kind and kind != self.kind and kind in HW.kinds():
            self.set_kind(kind)
        self.set_values(spec)

    # ------------------------------------------------------------------ add
    def add(self):
        spec = self.values()
        errs, _ = HW.validate(self.kind, spec)
        if errs:
            self._check()
            return None
        st = self.ctx.store
        kind = self.kind
        box = {}

        def fn(p):
            box['id'] = HW.add_to_project(p, kind, spec)
        keys = ['library'] if kind in HW.SENSOR_KINDS else ['hardware', 'ecu', 'analysis']
        st.edit(keys, fn, f'Added {HW.KIND_LABEL[kind].lower()} {spec.get("name")}')
        self.ctx.analysis.request(force=True)
        self.msg.setText(f'Added {spec.get("name")} ({box["id"]}).' +
                         (' Selected for the rig: the studies use its figures now.' if kind in HW.RIG_ROLES else
                          ' It is in the Library; pick it for a sensor or camera.'))
        return box['id']


def open_hardware_dialog(window, kind=None, spec=None):
    """Show the dialog for the main window (modeless, so the studies can be watched while filling it)."""
    d = HardwareDialog(window.ctx, window, kind=kind or 'lidar', spec=spec)
    d.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
    d.show()
    return d
