"""Analysis ▸ Load cases: how the load (occupants, luggage, the rig itself) moves the body on its suspension, and so
every camera and sensor of the rig, and what that does to the stitched ground image.

Tabs: Cases (the load cases from the suspension: their items, the corner drops, pitch and roll, every camera's and
sensor's move against the calibration load and, after Run, the ground figures), Loads (the load items and the rig's
masses), Suspension (corners, tyres, bump stops, measured curves, anti-roll bars, kerb loads; import and export) and
Typed drops (front/rear axle drops typed per case, as in earlier versions). Edits are undoable steps on
analysis.suspension; the typed drops are saved to analysis.load_cases when they are run.
"""
from __future__ import annotations

import math
import os
import threading
import traceback

from PyQt6.QtCore import Qt, QObject, QTimer, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (QVBoxLayout, QHBoxLayout, QGridLayout, QTableWidget, QTableWidgetItem, QHeaderView,
                             QAbstractItemView, QProgressBar, QTabWidget, QWidget, QComboBox, QLineEdit, QMenu,
                             QSplitter, QFileDialog)

from ...core import suspension as SU
from ...core.loadcases import DEFAULT_CASES, cases_of, evaluate
from ..theme import STATUS, mono
from ..widgets import label, button, ROW_H, SegmentedControl, ToggleSwitch
from .base import VDialog

# ---- the typed drops (earlier versions' table: the column layout is kept)
COLS = ['Case', 'Front drop mm', 'Rear drop mm', 'Pitch °', 'Blind m²', 'Δ blind', 'Min px/cm', 'Ground shift cm',
        'Seam ghost cm', 'Status']
EDITABLE = 3
STATUS_TEXT = {'good': 'OK', 'warning': 'Recalibrate per load', 'serious': 'Visible ghosting'}
CASE_STATUS = {'good': 'OK', 'warning': 'Recalibrate', 'serious': 'Ghosting'}

# ---- the suspension cases
CASE_COLS = ['Case', 'Added kg', 'FL mm', 'FR mm', 'RL mm', 'RR mm', 'Pitch °', 'Roll °', 'Bump stop', 'Move mm',
             'Turn °', 'Δ blind m²', 'Shift cm', 'Ghost cm', 'Status']
CASE_TIPS = ['Double-click to rename', 'Mass the case adds to the empty vehicle (kerb)',
             'Drop of the body at the front-left wheel from kerb (positive = lower)',
             'Drop at the front-right wheel from kerb', 'Drop at the rear-left wheel from kerb',
             'Drop at the rear-right wheel from kerb', 'Body pitch from kerb (positive = nose down)',
             'Body roll from kerb (positive = right side lower)', 'Corners resting on their bump stop at this load',
             'Largest move of a camera or sensor relative to the calibration load',
             'The body\'s rotation relative to the calibration load (the same for every device: the body is rigid)',
             'Blind area change against the calibration load (after Run)',
             'Worst move of a ground point in the stitched view calibrated at the calibration load (after Run)',
             'Worst disagreement of the two cameras at a seam (after Run)', 'Ground figures (after Run)']
C_IMG = 11
ITEM_COLS = ['In this case', 'Mass kg']
DEV_COLS = ['Camera / sensor', 'Δx mm', 'Δy mm', 'Δz mm', 'Move mm', 'Δyaw °', 'Δpitch °', 'Δroll °']
LOAD_COLS = ['Item', 'Kind', 'Mass kg', 'x m', 'y m', 'z m']
RIG_COLS = ['Tag', 'Part', 'Kind', 'Mass kg', 'Estimate', 'x m', 'y m', 'z m']
SUSP_COLS = ['Corner', 'Spring N/mm', 'Motion ratio', 'Wheel rate N/mm', 'Tyre N/mm', 'Kerb load kg', 'Ride height mm',
             'Bump gap mm', 'Bump rate N/mm', 'Bump curve', 'Measured curve']
SUSP_FIELDS = [None, ('spring_rate', 'rate', 'N/mm', 1), ('motion_ratio', 'ratio', '', 3),
               ('wheel_rate', 'rate', 'N/mm', 1), ('tyre_rate', 'rate', 'N/mm', 0), ('kerb_load', 'mass', 'kg', 1),
               ('ride_height', 'length', 'mm', 1), ('bump_gap', 'length', 'mm', 1), ('bump_rate', 'rate', 'N/mm', 1),
               None, None]
SUSP_TIPS = ['', 'Spring rate at the spring; units such as 140 lbf/in or 25 kN/m are converted',
             'Spring travel per wheel travel (wheel rate = spring rate × ratio²)',
             'Rate at the wheel: spring rate × motion ratio² (typing it sets the spring rate)',
             'Vertical rate of the tyre (in series with the suspension)', 'Wheel load at kerb (the empty vehicle)',
             'Wheel centre to wheel-arch lip at kerb, as the model is (0 = not given)',
             'Wheel travel from kerb until the bump stop touches',
             'Bump-stop rate at the wheel once touching (used when there is no bump curve)',
             'The progressive bump-stop curve (from a file)',
             'A measured wheel load vs travel curve (K&C or CAD) replaces the spring and bump stop']
MIRRORED = {'spring_rate', 'motion_ratio', 'wheel_rate', 'tyre_rate', 'bump_gap', 'bump_rate'}
FILE_FILTER = 'Suspension data (*.json *.yaml *.yml *.csv);;JSON (*.json);;YAML (*.yaml *.yml);;CSV (*.csv)'


class _Sig(QObject):
    done = pyqtSignal(object)
    failed = pyqtSignal(str)


def _table(cols, tips=None, stretch=0):
    t = QTableWidget(0, len(cols))
    t.setHorizontalHeaderLabels(cols)
    if tips:
        for i, tip in enumerate(tips):
            if tip:
                t.horizontalHeaderItem(i).setToolTip(tip)
    t.verticalHeader().setVisible(False)
    t.verticalHeader().setDefaultSectionSize(ROW_H + 2)
    t.horizontalHeader().setFixedHeight(ROW_H)
    t.setShowGrid(False)
    t.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    t.setEditTriggers(QAbstractItemView.EditTrigger.DoubleClicked | QAbstractItemView.EditTrigger.EditKeyPressed
                      | QAbstractItemView.EditTrigger.AnyKeyPressed)
    hh = t.horizontalHeader()
    for c in range(len(cols)):
        hh.setSectionResizeMode(c, QHeaderView.ResizeMode.Stretch if c == stretch else
                                QHeaderView.ResizeMode.ResizeToContents)
    return t


def _item(text, num=False, editable=False, tip=''):
    it = QTableWidgetItem(str(text).replace('-', '−') if num else str(text))
    if num:
        it.setFont(mono(12))
        it.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    if not editable:
        it.setFlags(it.flags() & ~Qt.ItemFlag.ItemIsEditable)
    if tip:
        it.setToolTip(tip)
    return it


def _f(v, dp=1, sign=False):
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return ''
    if abs(v) < 0.5 * 10 ** -dp:
        v = 0.0
    return f'{v:+.{dp}f}' if sign else f'{v:.{dp}f}'


def _text(it) -> str:
    return (it.text() if it is not None else '').replace('−', '-').strip()


class _QtyEdit(QLineEdit):
    """A number field that reads units ('1230 kg', '2.7 kN', '140 lbf/in'): committed(value in the base unit)."""
    committed = pyqtSignal(float)
    error = pyqtSignal(str)

    def __init__(self, kind, unit, dec=1, width=96, tip='', empty=None):
        super().__init__()
        self.kind, self.unit, self.dec = kind, unit, dec
        self.empty = empty                      # the value an emptied field means (None: a number is needed)
        self.setFixedWidth(width)
        self.setFixedHeight(ROW_H)
        self.setFont(mono(12))
        self.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        if tip:
            self.setToolTip(tip)
        self._v = None
        self.editingFinished.connect(self._finish)

    def set_value(self, v):
        if self.hasFocus() and self.isModified():
            return                                   # never overwrite what is being typed
        self._v = v
        self.setText(f'{v:.{self.dec}f} {self.unit}'.strip() if v is not None else '')
        self.setModified(False)

    def _finish(self):
        if not self.isModified():
            return
        self.setModified(False)
        if not self.text().strip() and self.empty is not None:
            self.committed.emit(float(self.empty))
            return
        try:
            v = SU.quantity(self.text(), self.kind, self.unit if self.kind not in ('ratio',) else None)
        except ValueError as e:
            self.set_value(self._v)
            self.error.emit(str(e))
            return
        self.committed.emit(v)


class LoadCasesDialog(VDialog):
    def __init__(self, ctx, parent=None):
        super().__init__(parent, 'Load cases', 'body drop, pitch and roll under load', (1240, 760))
        self.ctx = ctx
        self.results = None            # the typed drops' LoadResults (Typed drops tab)
        self.sres = None               # the suspension cases with their ground figures, after Run
        self.statics = None            # the suspension cases' static solve (always current)
        self.busy = False
        self.sig = _Sig()
        self.sig.done.connect(self._done)
        self.sig.failed.connect(self._failed)
        self._filling = False
        self._run_for = None           # the project a running evaluation belongs to
        self._img_rev = None           # store revision the ground figures belong to
        self._stale = False
        self._timer = QTimer(self)             # store changes from elsewhere (a camera drag) refresh once per burst
        self._timer.setSingleShot(True)
        self._timer.setInterval(40)
        self._timer.timeout.connect(self.refresh)
        lay = QVBoxLayout(self.body)
        lay.setContentsMargins(14, 10, 14, 10)
        lay.setSpacing(6)
        lay.addWidget(label('A data-collection vehicle carries its rig, a crew and equipment, and its cameras are '
                            'calibrated at one load. The suspension and the loads give each corner\'s drop, the body\'s '
                            'pitch and roll and so how far every camera and sensor moves relative to the road between '
                            'the calibration load and each case: what the calibration has to absorb. Run adds the '
                            'ground figures: blind area, the stitched ground\'s shift and the ghosting at the seams.',
                            muted=True, small=True, wrap=True))
        self.typical_lbl = label('', small=True, wrap=True)
        lay.addWidget(self.typical_lbl)
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.addTab(self._cases_tab(), 'Cases')
        self.tabs.addTab(self._loads_tab(), 'Loads')
        self.tabs.addTab(self._susp_tab(), 'Suspension')
        self.tabs.addTab(self._typed_tab(), 'Typed drops')
        lay.addWidget(self.tabs, 1)
        row = QHBoxLayout()
        self.msg_lbl = label('', muted=True, small=True, wrap=True)
        row.addWidget(self.msg_lbl, 1)
        self.bar = QProgressBar()
        self.bar.setTextVisible(False)
        self.bar.setRange(0, 0)
        self.bar.setFixedWidth(160)
        self.bar.hide()
        row.addWidget(self.bar)
        lay.addLayout(row)
        self.hint.setText('Drops from kerb (the empty vehicle as modelled) · moves, shift and ghost relative to the '
                          'calibration load · pitch + nose down, roll + right side down')
        b_close = button('Close')
        b_close.clicked.connect(self.accept)
        self.b_run = button('Run', 'primary', None, 'Evaluate the ground figures of every case (suspension cases and '
                                                   'typed drops) in the background')
        self.b_run.clicked.connect(self.run)
        self.buttons.addWidget(b_close)
        self.buttons.addWidget(self.b_run)
        self.set_cases(cases_of(ctx.store.project))
        self.refresh()
        # File > New / Open: show (and run) the new project's cases, never write the old ones into it
        ctx.store.project_replaced.connect(self._project_replaced)
        ctx.store.changed.connect(self._store_changed)

    # ================================================================== layout
    def _cases_tab(self):
        w = QWidget()
        w.setObjectName('panel')
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 8, 0, 0)
        v.setSpacing(6)
        top = QHBoxLayout()
        top.addWidget(label('Calibrated at', ink2=True))
        self.cal_combo = QComboBox()
        self.cal_combo.setMinimumWidth(260)
        self.cal_combo.setToolTip('The load the cameras and sensors are calibrated at: every move, the ground shift and '
                                  'the ghosting are relative to it')
        self.cal_combo.currentIndexChanged.connect(self._cal_picked)
        top.addWidget(self.cal_combo)
        top.addSpacing(12)
        b = button('Add case', None, 'plus', 'A new case with the calibration load\'s items')
        b.clicked.connect(lambda: self.add_load_case())
        top.addWidget(b)
        b = button('Duplicate', None, 'copy', 'A copy of the selected case')
        b.clicked.connect(lambda: self.duplicate_load_case())
        top.addWidget(b)
        b = button('Remove', 'ghost', 'trash', 'Remove the selected case')
        b.clicked.connect(lambda: self.remove_load_case())
        top.addWidget(b)
        top.addStretch(1)
        v.addLayout(top)
        self.find_lbl = label('', small=True, wrap=True)
        self.find_lbl.setToolTip('The static solve\'s findings (and, after Run, the ghosting): the worst first')
        v.addWidget(self.find_lbl)
        split = QSplitter(Qt.Orientation.Vertical)
        split.setHandleWidth(4)
        self.ctable = _table(CASE_COLS, CASE_TIPS, stretch=-1)
        self.ctable.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Interactive)
        self.ctable.setColumnWidth(0, 210)
        self.ctable.horizontalHeader().setSectionResizeMode(len(CASE_COLS) - 1, QHeaderView.ResizeMode.Stretch)
        self.ctable.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.ctable.itemChanged.connect(self._case_edited)
        self.ctable.itemSelectionChanged.connect(self._case_selected)
        split.addWidget(self.ctable)
        low = QSplitter(Qt.Orientation.Horizontal)
        low.setHandleWidth(4)
        box = QWidget()
        bl = QVBoxLayout(box)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.setSpacing(4)
        self.items_lbl = label('Loads in this case', bold=True)
        bl.addWidget(self.items_lbl)
        bl.addWidget(label('Tick what the case carries; a mass here overrides the item\'s own for this case (clear it '
                           'to use the item\'s).', muted=True, small=True, wrap=True))
        self.mtable = _table(ITEM_COLS)
        self.mtable.itemChanged.connect(self._case_item_edited)
        bl.addWidget(self.mtable, 1)
        low.addWidget(box)
        dev = QWidget()
        ll = QVBoxLayout(dev)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.setSpacing(4)
        self.dev_lbl = label('Cameras and sensors relative to the calibration load', bold=True)
        self.dev_lbl.setWordWrap(True)
        ll.addWidget(self.dev_lbl)
        self.dtable = _table(DEV_COLS, ['', 'Along x (forward), vehicle frame', 'Along y (left)', 'Along z (up)',
                                        'Distance moved', 'Change of the device\'s own yaw',
                                        'Change of its pitch (positive down)', 'Change of its roll'])
        self.dtable.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        ll.addWidget(self.dtable, 1)
        low.addWidget(dev)
        low.setSizes([340, 860])
        split.addWidget(low)
        split.setSizes([250, 330])
        v.addWidget(split, 1)
        return w

    def _loads_tab(self):
        w = QWidget()
        w.setObjectName('panel')
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 8, 0, 0)
        v.setSpacing(6)
        top = QHBoxLayout()
        top.addWidget(label('Load items', bold=True))
        top.addWidget(label('masses at points in the vehicle frame (x forward, y left, z up, m from the rear-axle '
                            'centre on the ground); double-click to edit', muted=True, small=True))
        top.addStretch(1)
        b = button('Add item', None, 'plus', 'A custom item (10 kg in the boot)')
        b.clicked.connect(lambda: self.add_item())
        top.addWidget(b)
        self.b_preset = button('Add preset', None, 'plus', 'A seat, the boot, the roof or the recording equipment, '
                                                           'placed on the vehicle')
        menu = QMenu(self.b_preset)
        for pid, name, _, _ in SU.PRESETS:
            menu.addAction(name, lambda p=pid: self.add_item(p))
        self.b_preset.setMenu(menu)
        top.addWidget(self.b_preset)
        b = button('Place on the vehicle', None, None, 'Estimate the seats\', boot\'s, roof\'s and equipment\'s '
                                                       'positions again from the vehicle model and the driver side')
        b.clicked.connect(lambda: self.place_items())
        top.addWidget(b)
        b = button('Remove', 'ghost', 'trash', 'Remove the selected item (from every case too)')
        b.clicked.connect(lambda: self.remove_item())
        top.addWidget(b)
        v.addLayout(top)
        self.itable = _table(LOAD_COLS)
        self.itable.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.itable.itemChanged.connect(self._item_edited)
        v.addWidget(self.itable, 3)
        top = QHBoxLayout()
        top.addWidget(label('The rig', bold=True))
        self.rig_lbl = label('', muted=True, small=True)
        top.addWidget(self.rig_lbl)
        top.addStretch(1)
        b = button('Estimates', 'ghost', 'refresh', 'Back to the estimated masses')
        b.clicked.connect(lambda: self.reset_rig_masses())
        top.addWidget(b)
        v.addLayout(top)
        v.addWidget(label('The visible cameras, sensors and rig parts of the variant and the ECU, where the design puts '
                          'them. Cameras weigh their module\'s weight plus a 15 g bracket; sensors are estimated from '
                          'their housing size and rig parts as 3 mm aluminium shells: typical estimates. Double-click '
                          'a mass to enter the real one (clear it for the estimate).', muted=True, small=True,
                          wrap=True))
        self.rtable = _table(RIG_COLS, stretch=1)
        self.rtable.itemChanged.connect(self._rig_edited)
        v.addWidget(self.rtable, 2)
        return w

    def _susp_tab(self):
        w = QWidget()
        w.setObjectName('panel')
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 8, 0, 0)
        v.setSpacing(6)
        self.source_lbl = label('', small=True, wrap=True)
        v.addWidget(self.source_lbl)
        g = QGridLayout()
        g.setHorizontalSpacing(8)
        g.setVerticalSpacing(4)
        self.fields = {}
        spec = [('kerb_mass', 'Kerb mass', 'mass', 'kg', 0, 'The empty vehicle (no rig, no occupants); spread over '
                 'the corners by the shares; a weight in N or kN is converted'),
                ('front_pct', 'Front axle', 'percent', '%', 1, 'Share of the kerb mass on the front axle'),
                ('left_pct', 'Left side', 'percent', '%', 1, 'Share of the kerb mass on the left wheels'),
                ('gross_mass', 'Gross mass', 'mass', 'kg', 0, 'Gross vehicle mass the cases are checked against '
                 '(0 = not checked)'),
                ('occupant_mass', 'Occupant', 'mass', 'kg', 0, 'Mass of a new occupant item'),
                ('track_front', 'Track front', 'position', 'm', 3, 'Between the front tyre centres (0 = measured on '
                 'the vehicle model)'),
                ('track_rear', 'Track rear', 'position', 'm', 3, 'Between the rear tyre centres (0 = measured)'),
                ('arb_front', 'Anti-roll front', 'rate', 'N/mm', 1, 'The front bar\'s contribution to each wheel\'s '
                 'rate in roll (0 = none)'),
                ('arb_rear', 'Anti-roll rear', 'rate', 'N/mm', 1, 'The rear bar\'s (or twist beam\'s) contribution '
                 'to each wheel\'s rate in roll'),
                ('tol_mm', 'Tolerance', 'length', 'mm', 1, 'A camera or sensor moving more than this from the '
                 'calibration load is flagged'),
                ('tol_deg', 'Tolerance', 'angle', '°', 2, 'A body rotation larger than this from the calibration load '
                 'is flagged')]
        for i, (key, text, kind, unit, dec, tip) in enumerate(spec):
            f = _QtyEdit(kind, unit, dec, tip=tip, empty=0.0 if key in ('gross_mass', 'track_front', 'track_rear')
                         else None)
            f.committed.connect(lambda val, k=key: self.set_field(k, val))
            f.error.connect(self._say)
            a = label(text, ink2=True)
            a.setToolTip(tip)
            r, c = divmod(i, 6)
            g.addWidget(a, r, 2 * c)
            g.addWidget(f, r, 2 * c + 1)
            self.fields[key] = f
        g.addWidget(label('Driver', ink2=True), 1, 10)
        self.side = SegmentedControl([('left', 'Left'), ('right', 'Right')], 'left')
        self.side.setToolTip('Which front seat the driver sits in (the driver and front passenger items swap sides)')
        self.side.changed.connect(lambda k: self.set_field('driver_side', k))
        g.addWidget(self.side, 1, 11)
        g.setColumnStretch(12, 1)
        v.addLayout(g)
        self.stable = _table(SUSP_COLS, SUSP_TIPS)
        self.stable.itemChanged.connect(self._corner_edited)
        v.addWidget(self.stable)
        row = QHBoxLayout()
        self.same = ToggleSwitch('Same left and right')
        self.same.setChecked(True)
        self.same.setToolTip('Rates, motion ratios and bump stops typed for one wheel go to both wheels of the axle')
        row.addWidget(self.same)
        row.addStretch(1)
        b = button('Import…', None, 'import', 'A suspension file (JSON or YAML), a "parameter,value,unit" CSV sheet '
                                             'or a ride-height table (load, travel; per corner or for the selected '
                                             'corners)')
        b.clicked.connect(lambda: self.import_file())
        row.addWidget(b)
        b = button('Export…', None, 'export', 'The suspension and loads as JSON or YAML, or the suspension as a CSV '
                                             'sheet')
        b.clicked.connect(lambda: self.export_file())
        row.addWidget(b)
        b = button('Clear curve', 'ghost', None, 'Drop the measured curve of the selected corners')
        b.clicked.connect(lambda: self.clear_curve())
        row.addWidget(b)
        b = button('Typical values', 'ghost', 'refresh', 'Back to typical values for a compact sedan scaled to the '
                                                        'model (keeps the loads and cases)')
        b.clicked.connect(lambda: self.typical_values())
        row.addWidget(b)
        v.addLayout(row)
        v.addWidget(label(SU.ASSUMPTIONS, muted=True, small=True, wrap=True))
        v.addStretch(1)
        return w

    def _typed_tab(self):
        w = QWidget()
        w.setObjectName('panel')
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 8, 0, 0)
        v.setSpacing(6)
        v.addWidget(label('Enter how far each axle drops from kerb weight in every load case (static suspension '
                          'deflection, positive = lower) when you have the drops but not the suspension. The cameras '
                          'are calibrated at kerb weight, so a loaded body moves the stitched ground image and the two '
                          'cameras at a seam stop agreeing. The preset values are examples: replace them with the '
                          'vehicle\'s suspension data.', muted=True, small=True, wrap=True))
        self.table = QTableWidget(0, len(COLS))
        self.table.setHorizontalHeaderLabels(COLS)
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(ROW_H + 2)
        self.table.horizontalHeader().setFixedHeight(ROW_H)
        self.table.setShowGrid(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for c in range(1, len(COLS)):
            hh.setSectionResizeMode(c, QHeaderView.ResizeMode.ResizeToContents)
        self.table.itemChanged.connect(self._edited)
        v.addWidget(self.table, 1)
        row = QHBoxLayout()
        b_add = button('Add case', None, 'plus')
        b_add.clicked.connect(self.add_case)
        b_del = button('Remove', None, 'minus')
        b_del.clicked.connect(self.remove_case)
        b_reset = button('Example values')
        b_reset.clicked.connect(lambda: self.set_cases(DEFAULT_CASES))
        row.addWidget(b_add)
        row.addWidget(b_del)
        row.addWidget(b_reset)
        row.addStretch(1)
        v.addLayout(row)
        return w

    # ================================================================== store
    def _project_replaced(self):
        self._timer.stop()
        self.set_cases(cases_of(self.ctx.store.project))
        self.sres = None
        self._img_rev = None
        self.set_subtitle('body drop, pitch and roll under load')
        self.refresh()

    def _store_changed(self, sections=None):
        if not self.isVisible():
            self._stale = True
            return
        self._timer.start()

    def showEvent(self, e):
        super().showEvent(e)
        if self._stale:
            self._stale = False
            self.refresh()

    def _say(self, text, ms=6000):
        self.msg_lbl.setText(text)
        if getattr(self.ctx, 'msg', None) is not None and self.ctx.window is not None:
            self.ctx.msg(text.splitlines()[0], ms)

    def settings(self) -> SU.SuspensionSettings:
        return SU.settings_for(self.ctx.store.project)

    def _apply(self, text, change, source=None):
        """change(settings) edits a copy of the settings; the result (clamped) is saved as one undo step. False when
        nothing changes."""
        p = self.ctx.store.project
        old = self.settings()
        st = self.settings()
        change(st)
        if source:
            st.source = source
        new = SU.SuspensionSettings.from_dict(st.to_dict(), base=SU.typical(p, st.driver_side)).to_dict()
        if new == old.to_dict() and p.analysis.suspension:
            self.refresh()                      # a clamped value: show what is stored
            return False

        def fn(q):
            q.analysis.suspension = new
        self.ctx.store.edit('analysis', fn, f'Load cases · {text}')
        self._timer.stop()
        self.refresh()                          # the dialog's own edit shows at once
        return True

    # ================================================================== refresh
    def refresh(self):
        p = self.ctx.store.project
        st = self.settings()
        try:
            self.statics = SU.solve_cases(p, st)
        except Exception as e:                  # never leave the dialog half-filled
            traceback.print_exc()
            self.statics = None
            self._say(f'The static solve failed: {type(e).__name__}: {e}')
        self._filling = True
        try:
            self._fill_typical(st)
            self._fill_findings(st)
            self._fill_cal(st)
            self._fill_cases(st)
            self._fill_loads(st)
            self._fill_susp(st)
        finally:
            self._filling = False
        self._case_selected()

    def _fill_typical(self, st):
        if st.is_typical:
            self.typical_lbl.setText(f'⚠ {SU.TYPICAL_NOTE} Kerb mass estimated from the model\'s size.')
            self.typical_lbl.setStyleSheet(f'color: {STATUS["warning"]};')
            self.source_lbl.setText('Typical values: the rates follow typical ride frequencies (1.25 Hz front, '
                                    '1.5 Hz rear) at the kerb mass estimated from the model\'s size, tyres 200 N/mm, '
                                    'a generic progressive bump stop and typical anti-roll bars. Replace them with '
                                    'your vehicle\'s data: type them (units such as lbf/in or kN/m are converted) or '
                                    'import a file.')
            self.source_lbl.setStyleSheet(f'color: {STATUS["warning"]};')
        else:
            self.typical_lbl.setText('')
            self.typical_lbl.setStyleSheet('')
            self.source_lbl.setText(f'Suspension: {st.source}' + (f' · {st.note}' if st.note else ''))
            self.source_lbl.setStyleSheet('')
        self.typical_lbl.setVisible(bool(self.typical_lbl.text()))

    def _fill_findings(self, st):
        res = self.statics
        if self.sres is not None and self._img_rev == self.ctx.store.revision:
            res = self.sres                      # the same statics with the ground figures
        fs = [f for f in (res.findings if res is not None else []) if f['severity'] in SU.ISSUE_SEVERITIES]
        if not fs:
            good = [f for f in (res.findings if res is not None else []) if f['severity'] == 'good']
            self.find_lbl.setText(good[0]['message'] if good else '')
            self.find_lbl.setStyleSheet(f'color: {STATUS["good"]};' if good else '')
        else:
            self.find_lbl.setText('\n'.join(f'{f["severity"].capitalize()}: {f["message"]}' for f in fs[:4]) +
                                  (f'\n… and {len(fs) - 4} more' if len(fs) > 4 else ''))
            self.find_lbl.setStyleSheet(f'color: {STATUS[fs[0]["severity"]]};')
        self.find_lbl.setVisible(bool(self.find_lbl.text()))

    def _fill_cal(self, st):
        cb = self.cal_combo
        cb.blockSignals(True)
        cb.clear()
        cb.addItem('Kerb (empty, as modelled)', '')
        for c in st.cases:
            cb.addItem(c.name, c.id)
        i = cb.findData(st.calibrated_at)
        cb.setCurrentIndex(max(i, 0))
        cb.blockSignals(False)

    def _selected_case(self):
        st = self.statics.settings if self.statics is not None else None
        r = self.ctable.currentRow()
        if st is None or not (0 <= r < len(st.cases)):
            return None
        return st.cases[r]

    def _fill_cases(self, st):
        t = self.ctable
        sel = t.currentRow()
        res = self.statics
        img = {}
        if self.sres is not None and self._img_rev == self.ctx.store.revision:
            img = {c.id: c.image for c in self.sres.cases if c.image is not None}
        t.setRowCount(len(st.cases))
        for i, case in enumerate(st.cases):
            cr = res.cases[i] if res is not None and i < len(res.cases) else None
            nm = _item(case.name, editable=True, tip='The cameras are calibrated at this load (bold)'
                       if case.id == st.calibrated_at else 'Double-click to rename')
            if case.id == st.calibrated_at:
                fnt = nm.font()
                fnt.setBold(True)
                nm.setFont(fnt)
            nm.setData(Qt.ItemDataRole.UserRole, case.id)
            t.setItem(i, 0, nm)
            vals = [''] * (len(CASE_COLS) - 1)
            if cr is not None:
                s = cr.statics
                vals[:10] = [_f(s.added_kg, 1)] + [_f(s.corners[k].drop_mm, 1) for k in SU.CORNERS] + \
                    [_f(s.pitch_deg, 2, True), _f(s.roll_deg, 2, True), ' '.join(s.bump) or '—', _f(cr.worst_mm, 1),
                     _f(cr.turn_deg, 2)]
                r = img.get(case.id)
                if r is not None:
                    vals[10:] = [_f(r.delta_blind, 3, True), _f(r.ground_shift_cm, 1), _f(r.seam_ghost_cm, 1),
                                 CASE_STATUS.get(r.status, r.status)]
            for j, v in enumerate(vals, 1):
                it = _item(v, num=j < len(CASE_COLS) - 1)
                if 2 <= j <= 5 and cr is not None:
                    k = SU.CORNERS[j - 2]
                    cs = cr.statics.corners[k]
                    it.setToolTip(f'{k}: suspension {cs.travel_mm:.1f} mm + tyre {cs.tyre_mm:.1f} mm · wheel load '
                                  f'{cs.load_kg:.0f} kg' +
                                  (f' · ride height {cs.ride_mm:.0f} mm' if cs.ride_mm is not None else '') +
                                  (f' · {cs.bump_mm:.1f} mm into the bump stop' if cs.on_bump else '') +
                                  (' · beyond the measured curve' if cs.beyond_curve else '') +
                                  (' · the wheel would lift' if cs.lifted else ''))
                    if cs.lifted:
                        it.setForeground(QColor(STATUS['serious']))
                if j == 1 and cr is not None and cr.over_gross > 0:
                    it.setForeground(QColor(STATUS['serious']))
                    it.setToolTip(f'{cr.over_gross:.0f} kg over the gross vehicle mass')
                if j == 8 and cr is not None and cr.statics.bump:
                    it.setForeground(QColor(STATUS['warning']))
                if cr is not None and ((j == 9 and cr.worst_mm > st.tol_mm) or (j == 10 and cr.turn_deg > st.tol_deg)):
                    it.setForeground(QColor(STATUS['warning']))
                if j == len(CASE_COLS) - 1 and img.get(case.id) is not None:
                    it.setForeground(QColor(STATUS.get(img[case.id].status, '#888')))
                    it.setToolTip(STATUS_TEXT.get(img[case.id].status, ''))
                t.setItem(i, j, it)
        if st.cases and t.currentRow() != min(max(sel, 0), len(st.cases) - 1):
            t.setCurrentCell(min(max(sel, 0), len(st.cases) - 1), 0)

    def _case_selected(self):
        if self._filling:
            return
        self._filling = True
        try:
            st = self.statics.settings if self.statics is not None else None
            case = self._selected_case()
            m = self.mtable
            m.setRowCount(0)
            d = self.dtable
            d.setRowCount(0)
            if st is None or case is None:
                self.items_lbl.setText('Loads in this case')
                self.dev_lbl.setText('Cameras and sensors relative to the calibration load')
                return
            self.items_lbl.setText(f'Loads in "{case.name}"')
            comps = self.statics.components
            m.setRowCount(len(st.items))
            for i, it in enumerate(st.items):
                c = _item(it.name, editable=False, tip=SU.KIND_LABEL.get(it.kind, it.kind))
                c.setFlags(c.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                c.setCheckState(Qt.CheckState.Checked if it.id in case.items else Qt.CheckState.Unchecked)
                c.setData(Qt.ItemDataRole.UserRole, it.id)
                m.setItem(i, 0, c)
                own = SU.item_mass(self.ctx.store.project, st, it, comps)[0]
                over = case.items.get(it.id)
                mi = _item(_f(over if over is not None else own, 1), num=True, editable=True,
                           tip='This case\'s mass' if over is not None else 'The item\'s own mass')
                if over is None:
                    mi.setForeground(QColor('#888'))
                m.setItem(i, 1, mi)
            cr = next((c for c in self.statics.cases if c.id == case.id), None)
            if cr is None:
                return
            self.dev_lbl.setText(f'Cameras and sensors in "{case.name}" relative to "{self.statics.calibration}" · '
                                 f'body {cr.rel["heave_mm"]:+.1f} mm at the rear axle, pitch {cr.rel["pitch_deg"]:+.2f}°, '
                                 f'roll {cr.rel["roll_deg"]:+.2f}°'.replace('-', '−'))
            d.setRowCount(len(cr.devices))
            for i, dv in enumerate(cr.devices):
                d.setItem(i, 0, _item(f'{dv["tag"]} · {dv["kind"]}'))
                for j, k in enumerate(('dx_mm', 'dy_mm', 'dz_mm', 'move_mm'), 1):
                    d.setItem(i, j, _item(_f(dv[k], 1, k != 'move_mm'), num=True))
                for j, k in enumerate(('d_yaw_deg', 'd_pitch_deg', 'd_roll_deg'), 5):
                    d.setItem(i, j, _item(_f(dv[k], 3, True), num=True))
                if dv['move_mm'] > st.tol_mm:
                    d.item(i, 4).setForeground(QColor(STATUS['warning']))
        finally:
            self._filling = False

    def _fill_loads(self, st):
        p = self.ctx.store.project
        comps = self.statics.components if self.statics is not None else SU.rig_components(p, st)
        t = self.itable
        sel = t.currentRow()
        t.setRowCount(len(st.items))
        for i, it in enumerate(st.items):
            rig = it.kind == 'rig'
            m, pos = SU.item_mass(p, st, it, comps)
            t.setItem(i, 0, _item(it.name, editable=True))
            t.item(i, 0).setData(Qt.ItemDataRole.UserRole, it.id)
            t.setItem(i, 1, _item(SU.KIND_LABEL.get(it.kind, it.kind)))
            for j, v in enumerate([m, *pos], 2):
                c = _item(_f(v, 1 if j == 2 else 3), num=True, editable=not rig,
                          tip='From the rig\'s parts (the table below)' if rig else '')
                if rig:
                    c.setForeground(QColor('#888'))
                t.setItem(i, j, c)
        if 0 <= sel < len(st.items) and t.currentRow() != sel:
            t.setCurrentCell(sel, 0)
        r = self.rtable
        r.setRowCount(len(comps))
        for i, c in enumerate(comps):
            r.setItem(i, 0, _item(c['tag']))
            r.item(i, 0).setData(Qt.ItemDataRole.UserRole, c['uid'])
            r.setItem(i, 1, _item(c['name']))
            r.setItem(i, 2, _item(c['kind']))
            mi = _item(_f(c['mass'], 3), num=True, editable=True, tip=c['source'])
            if c['source'] != 'entered':
                mi.setForeground(QColor('#888'))
            r.setItem(i, 3, mi)
            r.setItem(i, 4, _item(c['source'] if c['source'] != 'entered' else f'entered (estimate {c["estimate"]:.3f})'))
            for j, k in enumerate(('x', 'y', 'z'), 5):
                r.setItem(i, j, _item(_f(c[k], 3), num=True))
        tot = sum(c['mass'] for c in comps)
        self.rig_lbl.setText(f'{len(comps)} parts · {tot:.1f} kg')

    def _fill_susp(self, st):
        vals = {'kerb_mass': st.kerb_mass, 'front_pct': st.front_pct, 'left_pct': st.left_pct,
                'gross_mass': st.gross_mass, 'occupant_mass': st.occupant_mass, 'track_front': st.track_front,
                'track_rear': st.track_rear, 'arb_front': st.arb_front, 'arb_rear': st.arb_rear, 'tol_mm': st.tol_mm,
                'tol_deg': st.tol_deg}
        for k, f in self.fields.items():
            f.set_value(None if (k in ('gross_mass', 'track_front', 'track_rear') and not vals[k]) else vals[k])
        self.fields['gross_mass'].setPlaceholderText('not checked')
        if self.statics is not None:
            xy = self.statics.xy
            for ax in ('front', 'rear'):
                l, r = SU.AXLE_CORNERS[ax]
                f = self.fields[f'track_{ax}']
                f.setPlaceholderText(f'{xy[l][1] - xy[r][1]:.3f} m')
                f.setToolTip(f'Between the {ax} tyre centres' + (
                    '' if getattr(st, f'track_{ax}') else f': {xy[l][1] - xy[r][1]:.3f} m measured on the model') +
                    ' (type a value to set it, clear it to measure on the model)')
        self.side.set_value(st.driver_side)
        t = self.stable
        t.setRowCount(len(SU.CORNERS))
        for i, c in enumerate(SU.CORNERS):
            co = st.corners[c]
            t.setItem(i, 0, _item(f'{c} · {SU.CORNER_LABEL[c].lower()}'))
            for j, spec in enumerate(SUSP_FIELDS):
                if spec is None:
                    continue
                key, _, _, dec = spec
                v = co.wheel_rate if key == 'wheel_rate' else getattr(co, key)
                t.setItem(i, j, _item(_f(v, dec), num=True, editable=not (co.curve and key in (
                    'spring_rate', 'wheel_rate', 'bump_rate'))))
            bc = (f'progressive · {len(co.bump_curve)} points' if co.bump_curve else
                  f'{co.bump_rate:g} N/mm' if co.bump_rate > 0 else 'none')
            t.setItem(i, 9, _item(bc))
            mc = _item(f'{len(co.curve)} points · {co.curve[0][0]:.0f}–{co.curve[-1][0]:.0f} N' if co.curve else '—',
                       tip='Replaces the spring and the bump stop' if co.curve else '')
            t.setItem(i, 10, mc)
            if co.curve:
                for j in (1, 3, 8):
                    t.item(i, j).setForeground(QColor('#888'))
        t.setFixedHeight((ROW_H + 2) * len(SU.CORNERS) + ROW_H + 6)

    # ================================================================== cases
    def _cal_picked(self, i):
        if self._filling:
            return
        self.set_calibration(self.cal_combo.itemData(i) or '')

    def set_calibration(self, case_id):
        name = next((c.name for c in self.settings().cases if c.id == case_id), 'kerb')
        return self._apply(f'calibrated at {name}', lambda st: setattr(st, 'calibrated_at', case_id or ''))

    def _case_edited(self, it):
        if self._filling or it.column() != 0:
            return
        cid = it.data(Qt.ItemDataRole.UserRole)
        name = it.text().strip()
        if not name:
            self.refresh()
            return

        def ch(st):
            c = st.case(cid)
            if c is not None:
                c.name = name[:80]
        self._apply(f'renamed a case to {name}', ch)

    def add_load_case(self, name=None, items=None):
        """A new case (default: the calibration load's items, else the rig only) after the selected one."""
        def ch(st):
            base = st.case(st.calibrated_at)
            its = dict(items) if items is not None else (dict(base.items) if base is not None else
                                                         {i.id: None for i in st.items if i.kind == 'rig'})
            nm = name or f'Case {len(st.cases) + 1}'
            st.cases.append(SU.LoadCase(st.new_id(nm, 'case'), nm, its))
        if len(self.settings().cases) >= SU.MAX_CASES:
            self._say(f'At most {SU.MAX_CASES} cases.')
            return False
        ok = self._apply('added a case', ch)
        if ok:
            self.ctable.setCurrentCell(self.ctable.rowCount() - 1, 0)
            self._case_selected()
        return ok

    def duplicate_load_case(self):
        case = self._selected_case()
        if case is None:
            return False
        return self.add_load_case(f'{case.name} (copy)', case.items)

    def remove_load_case(self):
        case = self._selected_case()
        if case is None:
            return False

        def ch(st):
            st.cases = [c for c in st.cases if c.id != case.id]
            if st.calibrated_at == case.id:
                st.calibrated_at = ''
        return self._apply(f'removed {case.name}', ch)

    def _case_item_edited(self, it):
        if self._filling:
            return
        case = self._selected_case()
        if case is None:
            return
        iid = self.mtable.item(it.row(), 0).data(Qt.ItemDataRole.UserRole)
        if it.column() == 0:
            self.set_case_item(case.id, iid, it.checkState() == Qt.CheckState.Checked)
        else:
            txt = _text(it)
            if not txt:
                self.set_case_item(case.id, iid, True, None)
                return
            try:
                m = SU.quantity(txt, 'mass', 'kg', 'mass')
            except ValueError as e:
                self._say(str(e))
                self.refresh()
                return
            self.set_case_item(case.id, iid, True, max(m, 0.0))

    def set_case_item(self, case_id, item_id, included, mass=None):
        """Include an item in a case (with this case's mass, or None for the item's own) or leave it out."""
        st = self.settings()
        case, item = st.case(case_id), st.item(item_id)
        if case is None or item is None:
            return False

        def ch(s):
            c = s.case(case_id)
            if included:
                c.items[item_id] = mass
            else:
                c.items.pop(item_id, None)
        what = (f'{item.name} in {case.name}' + (f' at {mass:g} kg' if mass is not None else '')) if included else \
            f'{item.name} out of {case.name}'
        return self._apply(what, ch)

    # ================================================================== loads
    def add_item(self, preset=None):
        """A preset item (SU.PRESETS id) or a custom one (10 kg in the boot); it joins the selected case. Returns the
        new item's id (None when nothing was added)."""
        p = self.ctx.store.project
        st = self.settings()
        if len(st.items) >= SU.MAX_ITEMS:
            self._say(f'At most {SU.MAX_ITEMS} load items.')
            return None
        case = self._selected_case()
        cid = case.id if case is not None else None
        if preset is not None:
            new = SU.preset_item(p, st, preset)
        else:
            x, y, z = SU.seat_positions(p, st.driver_side)['luggage']
            n = sum(1 for i in st.items if i.name.startswith('Item ')) + 1
            new = SU.LoadItem(st.new_id(f'item {n}'), f'Item {n}', 'other', 10.0, x, y, z)

        def ch(s):
            s.items.append(new)
            c = s.case(cid) if cid else None
            if c is not None:
                c.items[new.id] = None
        if not self._apply(f'added {new.name}' + (f' to {case.name}' if case is not None else ''), ch):
            return None
        self.itable.setCurrentCell(self.itable.rowCount() - 1, 0)
        return new.id

    def remove_item(self, item_id=None):
        st = self.settings()
        if item_id is None:
            r = self.itable.currentRow()
            if not (0 <= r < len(st.items)):
                return False
            item_id = st.items[r].id
        item = st.item(item_id)
        if item is None:
            return False

        def ch(s):
            s.items = [i for i in s.items if i.id != item_id]
            for c in s.cases:
                c.items.pop(item_id, None)
        return self._apply(f'removed {item.name}', ch)

    def place_items(self):
        """The seats', boot's, roof's and equipment's positions estimated again from the vehicle and the driver side."""
        p = self.ctx.store.project

        def ch(st):
            pos = SU.seat_positions(p, st.driver_side)
            for it in st.items:
                stem = it.id.rsplit('_', 1)[0] if it.id not in pos and it.id.rsplit('_', 1)[0] in pos else it.id
                if stem in pos:
                    it.x, it.y, it.z = pos[stem]
        return self._apply('placed the items on the vehicle', ch)

    def _item_edited(self, it):
        if self._filling:
            return
        iid = self.itable.item(it.row(), 0).data(Qt.ItemDataRole.UserRole)
        col = it.column()
        if col == 0:
            name = it.text().strip()
            if not name:
                self.refresh()
                return
            self.set_item(iid, name=name[:80])
            return
        key = {2: 'mass', 3: 'x', 4: 'y', 5: 'z'}.get(col)
        if key is None:
            return
        try:
            v = SU.quantity(_text(it), 'mass' if key == 'mass' else 'position', 'kg' if key == 'mass' else 'm', key)
        except ValueError as e:
            self._say(str(e))
            self.refresh()
            return
        self.set_item(iid, **{key: v})

    def set_item(self, item_id, **kw):
        st = self.settings()
        item = st.item(item_id)
        if item is None:
            return False

        def ch(s):
            i = s.item(item_id)
            for k, v in kw.items():
                if k == 'mass':
                    v = SU._clamp(v, 'item_mass')
                elif k in ('x', 'y', 'z'):
                    v = SU._clamp(v, 'position')
                setattr(i, k, v)
        what = ', '.join(f'{k} {v:g}' if isinstance(v, float) else f'{k} {v}' for k, v in kw.items())
        return self._apply(f'{item.name} {what}', ch)

    def _rig_edited(self, it):
        if self._filling or it.column() != 3:
            return
        uid = self.rtable.item(it.row(), 0).data(Qt.ItemDataRole.UserRole)
        txt = _text(it)
        if not txt:
            self.set_rig_mass(uid, None)
            return
        try:
            v = SU.quantity(txt, 'mass', 'kg', 'mass')
        except ValueError as e:
            self._say(str(e))
            self.refresh()
            return
        self.set_rig_mass(uid, v)

    def set_rig_mass(self, uid, kg):
        """A camera's, sensor's or rig part's real mass (None: back to the estimate)."""
        def ch(st):
            if kg is None:
                st.rig_masses.pop(uid, None)
            else:
                st.rig_masses[uid] = SU._clamp(float(kg), 'rig_mass')
        tag = next((c['tag'] for c in (self.statics.components if self.statics else []) if c['uid'] == uid), uid)
        return self._apply(f'{tag} mass ' + (f'{kg:g} kg' if kg is not None else 'estimated'), ch)

    def reset_rig_masses(self):
        if not self.settings().rig_masses:
            return False
        return self._apply('rig masses estimated', lambda st: st.rig_masses.clear())

    # ================================================================== suspension
    def set_field(self, key, value):
        """A vehicle-level figure: kerb_mass, front_pct, left_pct, gross_mass, occupant_mass, track_front/rear,
        arb_front/rear, tol_mm, tol_deg, driver_side."""
        susp = key in ('kerb_mass', 'front_pct', 'left_pct', 'track_front', 'track_rear', 'arb_front', 'arb_rear')

        def ch(st):
            if key == 'kerb_mass':
                st.set_kerb(mass=SU._clamp(value, 'kerb_mass'))
            elif key in ('front_pct', 'left_pct'):
                st.set_kerb(**{key: SU._clamp(value, 'pct')})
            elif key in ('track_front', 'track_rear'):
                v = SU._clamp(value, 'track')
                setattr(st, key, 0.0 if v < 0.3 else v)
            elif key in ('arb_front', 'arb_rear'):
                setattr(st, key, SU._clamp(value, 'arb_rate'))
            elif key == 'gross_mass':
                st.gross_mass = SU._clamp(value, 'gross_mass')
            elif key == 'occupant_mass':
                st.occupant_mass = SU._clamp(value, 'occupant_mass')
            elif key == 'tol_mm':
                st.tol_mm = SU._clamp(value, 'tol_mm')
            elif key == 'tol_deg':
                st.tol_deg = SU._clamp(value, 'tol_deg')
            elif key == 'driver_side' and value in ('left', 'right'):
                st.driver_side = value
                sd = 1.0 if value == 'left' else -1.0
                for iid, sign in (('driver', sd), ('passenger', -sd)):     # the front occupants change seats
                    it = st.item(iid)
                    if it is not None:
                        it.y = sign * abs(it.y)
        text = {'driver_side': f'driver on the {value}'}.get(key) or \
            f'{key.replace("_", " ")} {value:g}'
        return self._apply(text, ch, source='entered' if susp else None)

    def _corner_edited(self, it):
        if self._filling:
            return
        spec = SUSP_FIELDS[it.column()] if it.column() < len(SUSP_FIELDS) else None
        if spec is None:
            return
        key, kind, unit, _ = spec
        corner = SU.CORNERS[it.row()]
        try:
            v = SU.quantity(_text(it), kind, unit or None, f'{corner} {key.replace("_", " ")}')
        except ValueError as e:
            self._say(str(e))
            self.refresh()
            return
        self.set_corner(corner, key, v)

    def set_corner(self, corner, key, value, both=None):
        """One corner's figure (and its partner's when 'Same left and right' is on, for the rates and bump stops)."""
        both = self.same.isChecked() if both is None else both
        targets = [corner] + ([SU.PARTNER[corner]] if both and key in MIRRORED else [])

        def ch(st):
            for c in targets:
                co = st.corners[c]
                if key == 'wheel_rate':
                    co.spring_rate = SU._clamp(value / co.motion_ratio ** 2, 'spring_rate')
                else:
                    setattr(co, key, SU._clamp(value, key))
        return self._apply(f'{"/".join(targets)} {key.replace("_", " ")} {value:g}', ch, source='entered')

    def _selected_corners(self):
        rows = sorted({i.row() for i in self.stable.selectedIndexes()})
        return [SU.CORNERS[r] for r in rows if 0 <= r < len(SU.CORNERS)]

    def clear_curve(self, corners=None):
        corners = corners or self._selected_corners()
        st = self.settings()
        corners = [c for c in corners if st.corners[c].curve]
        if not corners:
            self._say('Select the corners whose measured curve to drop.')
            return False

        def ch(s):
            for c in corners:
                s.corners[c].curve = []
        return self._apply(f'cleared the {"/".join(corners)} curve', ch, source='entered')

    def typical_values(self):
        """The suspension back to typical values for the model (the loads, rig masses and cases are kept)."""
        p = self.ctx.store.project

        def ch(st):
            t = SU.typical(p, st.driver_side)
            st.corners = t.corners
            st.arb_front, st.arb_rear = t.arb_front, t.arb_rear
            st.track_front = st.track_rear = 0.0
            st.note = ''
        return self._apply('typical suspension values', ch, source='typical')

    def import_file(self, path=None, corner=None):
        """Read a suspension file (JSON / YAML), a CSV parameter sheet or a CSV ride-height table (for the corners it
        names, else `corner` or the selected corners) as one undo step."""
        if path is None:
            path, _ = QFileDialog.getOpenFileName(self, 'Import suspension data', os.path.expanduser('~'), FILE_FILTER)
            if not path:
                return False
        if corner is None:
            sel = set(self._selected_corners())
            corner = ('all' if sel == set(SU.CORNERS) else 'front' if sel == {'FL', 'FR'} else
                      'rear' if sel == {'RL', 'RR'} else (sorted(sel, key=SU.CORNERS.index)[0] if sel else None))
        try:
            new, what, warnings = SU.load_file(path, self.settings(), corner)
        except ValueError as e:
            self._say(f'Not imported: {e}', 9000)
            return False

        def ch(st):
            for k in new.__dataclass_fields__:
                setattr(st, k, getattr(new, k))
        self._apply(f'imported {what} from {os.path.basename(path)}', ch, source='imported')
        self._say(f'Imported the {what} from {os.path.basename(path)}.' +
                  (' ' + ' '.join(warnings[:4]) + (f' (+{len(warnings) - 4} more)' if len(warnings) > 4 else '')
                   if warnings else ''))
        return True

    def export_file(self, path=None):
        if path is None:
            path, _ = QFileDialog.getSaveFileName(self, 'Export suspension data', 'suspension.json', FILE_FILTER)
            if not path:
                return []
        try:
            paths = SU.save_file(self.settings(), path)
        except (OSError, ValueError) as e:
            self._say(f'Not exported: {e}')
            return []
        self._say('Exported ' + ', '.join(os.path.basename(x) for x in paths) + f' to {os.path.dirname(path)}.')
        return paths

    # ================================================================== typed drops (earlier versions' table)
    def set_cases(self, cases):
        self._filling = True
        self.table.setRowCount(0)
        for name, f, r in cases:
            self._append(name, f, r)
        self._filling = False
        self.results = None

    def _append(self, name, f, r):
        i = self.table.rowCount()
        self.table.insertRow(i)
        for j, v in enumerate((str(name), f'{float(f):g}', f'{float(r):g}')):
            it = QTableWidgetItem(v)
            if j:
                it.setFont(mono(12))
                it.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.table.setItem(i, j, it)
        for j in range(EDITABLE, len(COLS)):
            it = QTableWidgetItem('')
            it.setFlags(it.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(i, j, it)

    def add_case(self):
        self._filling = True
        self._append(f'Case {self.table.rowCount() + 1}', 0, 0)
        self._filling = False
        self.table.editItem(self.table.item(self.table.rowCount() - 1, 0))

    def remove_case(self):
        rows = sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True)
        for r in rows:
            self.table.removeRow(r)

    def _edited(self, it):
        if self._filling or it.column() >= EDITABLE:
            return
        if it.column() in (1, 2):
            from ...core.mathutil import parse_number
            try:
                v = parse_number(it.text(), 'length', 'mm')
            except ValueError:
                v = 0.0
            self._filling = True
            it.setText(f'{v:g}')
            self._filling = False
        # results no longer match the inputs
        for j in range(EDITABLE, len(COLS)):
            o = self.table.item(it.row(), j)
            if o:
                o.setText('')

    def cases(self):
        from ...core.mathutil import parse_number
        out = []
        for i in range(self.table.rowCount()):
            name = (self.table.item(i, 0).text() if self.table.item(i, 0) else '').strip() or f'Case {i + 1}'
            vals = []
            for j in (1, 2):
                try:
                    vals.append(float(parse_number(self.table.item(i, j).text(), 'length', 'mm')))
                except (ValueError, AttributeError):
                    vals.append(0.0)
            out.append((name, vals[0], vals[1]))
        return out

    # ================================================================== run
    def run(self):
        if self.busy:
            return
        cases = self.cases()
        st = self.settings()
        if not cases and not st.cases:
            return
        store = self.ctx.store
        if self.ctx.analysis.result is None or not store.variant.cameras:
            self.set_subtitle('Nothing to evaluate yet: the load cases need a vehicle, cameras and an analysis result.')
            return
        if [list(c) for c in cases] != [[n, float(f), float(r)] for n, f, r in cases_of(store.project)]:
            store.edit('analysis', lambda p: setattr(p.analysis, 'load_cases', [list(c) for c in cases]), 'Edited load cases')
        project = store.project.snapshot()
        eng = self.ctx.window._shared_engine()
        cell = max(project.analysis.cell_m, 0.05)
        self._run_for = store.project
        self._run_rev = store.revision
        self.busy = True
        self.b_run.setEnabled(False)
        self.bar.show()

        def work():
            try:
                typed = evaluate(eng, project, cases, cell=cell) if cases else []
                sus = SU.run(project, eng, cell=cell) if SU.settings_for(project).cases else None
                self.sig.done.emit((typed, sus))
            except Exception as e:
                self.sig.failed.emit(f'{type(e).__name__}: {e}\n{traceback.format_exc()}')
        threading.Thread(target=work, daemon=True).start()

    def _done(self, payload):
        self.busy = False
        self.b_run.setEnabled(True)
        self.bar.hide()
        if self._run_for is not self.ctx.store.project:
            return                         # evaluated for a project that has since been replaced
        results, sus = payload
        self.results = results
        self._filling = True
        for i, r in enumerate(results):
            if i >= self.table.rowCount():
                break
            vals = [f'{r.pitch_deg:+.2f}', f'{r.blind_m2:.3f}', f'{r.delta_blind:+.3f}', f'{r.min_density:.1f}',
                    f'{r.ground_shift_cm:.1f}', f'{r.seam_ghost_cm:.1f}', STATUS_TEXT.get(r.status, r.status)]
            for j, v in enumerate(vals, EDITABLE):
                it = self.table.item(i, j)
                it.setText(v.replace('-', '−'))
                if j < len(COLS) - 1:
                    it.setFont(mono(12))
                    it.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                else:
                    it.setForeground(QColor(STATUS.get(r.status, '#888')))
        self._filling = False
        self.sres = sus
        stale = sus is not None and self._run_rev != self.ctx.store.revision
        self._img_rev = None if stale else self.ctx.store.revision
        self.refresh()
        every = list(results) + ([c.image for c in sus.cases if c.image is not None] if sus is not None else [])
        worst = max(every, key=lambda r: r.seam_ghost_cm) if every else None
        if worst:
            self.set_subtitle(f'worst: {worst.name} · {worst.seam_ghost_cm:.1f} cm seam ghost · '
                              f'{worst.ground_shift_cm:.1f} cm ground shift')
        if stale:
            self._say('The inputs changed during the run: Run again for their ground figures.')
        elif sus is not None and not sus.images and sus.image_note:
            self._say(sus.image_note)
        self.ctx.store.project.add_log('loadcases', f'{len(every)} cases' +
                                       (f' · worst seam ghost {worst.seam_ghost_cm:.1f} cm ({worst.name})' if worst else ''))

    def _failed(self, m):
        self.busy = False
        self.b_run.setEnabled(True)
        self.bar.hide()
        print(m)
        self.set_subtitle('Failed: ' + m.splitlines()[0])
