"""Calibrate ▸ Fleet: the same rig built on many vehicles. Every vehicle's measured (as-built) sensor poses, imported
from installation checks, extrinsics, nuScenes tables, Kalibr camchains or variants (or typed in), against the plan;
how each sensor moves over months; which vehicle differs from the rest. Cheap: computed again on every edit while the
page shows. Every edit is one undo step (project.fleet, saved with the project)."""
from __future__ import annotations

import os
import pickle
import threading
from types import SimpleNamespace

from PyQt6.QtCore import QObject, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import (QAbstractItemView, QComboBox, QFileDialog, QGridLayout, QHBoxLayout, QHeaderView,
                             QLineEdit, QMenu, QScrollArea, QSplitter, QTableWidget, QTableWidgetItem, QVBoxLayout,
                             QWidget)

from ...core import fleet as FL
from ...core.mathutil import fmt_num, parse_number
from ...core.project import Variant
from ..fleetview import DriftChart, FleetMatrix, sensor_color
from ..theme import STATUS, THEME, mono, sans
from ..widgets import ROW_H, ScrubSpinBox, StatusLine, button, hline, label, tool_button
from .recording_ws import Tile

# (key, label, unit, decimals, step, tooltip)
FIELDS = (
    ('pos_tol_mm', 'Position', 'mm', 1, 1.0, 'Largest offset along any axis that is still within tolerance (as the '
                                              'installation check)'),
    ('ang_tol_deg', 'Angle', '°', 2, 0.1, 'Largest rotation between planned and measured orientation within tolerance'),
    ('drift_window_days', 'Drift window', 'days', 0, 5.0,
     'A sensor whose trend takes it out of tolerance within this many days is flagged as drifting'),
    ('outlier_z', 'Outlier z', '', 1, 0.5,
     'Robust z-score (median absolute deviation, never under a quarter of the tolerance) beyond which a vehicle '
     'differs from the fleet'),
    ('jump_frac', 'Jump', '× tol', 2, 0.05,
     'A latest step this share of the tolerance off the sensor\'s earlier course is a jump'),
)
POSE_COLS = (('x', 'x', 'm', 3), ('y', 'y', 'm', 3), ('z', 'z', 'm', 3), ('yaw', 'Yaw', '°', 2),
             ('pitch', 'Pitch', '°', 2), ('roll', 'Roll', '°', 2))
FLEET_COLS = ('Sensor', 'Vehicles', 'Median Δ mm', 'Median Δ°', 'Spread', 'Beyond', 'Outliers', 'Median / tol')
SEV_OF = {'ok': 'good', 'adjust': 'warning', 're-mount': 'serious', 'missing': None}
ASYNC_POSES = 5000      # a fleet with more poses than this is analysed on a worker thread (about 0.2 s per 10 000)


def _table(cols, editable=False):
    t = QTableWidget(0, len(cols))
    t.setHorizontalHeaderLabels(list(cols))
    t.verticalHeader().setVisible(False)
    t.verticalHeader().setDefaultSectionSize(ROW_H)
    t.horizontalHeader().setFixedHeight(ROW_H)
    t.setShowGrid(False)
    t.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    t.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
    t.setEditTriggers(QAbstractItemView.EditTrigger.DoubleClicked | QAbstractItemView.EditTrigger.EditKeyPressed
                      if editable else QAbstractItemView.EditTrigger.NoEditTriggers)
    return t


def _item(text, num=False, editable=False, tip=None, color=None, bold=False, data=None):
    it = QTableWidgetItem(text)
    flags = Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled
    if editable:
        flags |= Qt.ItemFlag.ItemIsEditable
    it.setFlags(flags)
    if num:
        it.setFont(mono(12, QFont.Weight.DemiBold if bold else None))
        it.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    elif bold:
        it.setFont(sans(12, QFont.Weight.DemiBold))
    if tip:
        it.setToolTip(tip)
    if color is not None:
        it.setForeground(color)
    if data is not None:
        it.setData(Qt.ItemDataRole.UserRole, data)
    return it


class _Sig(QObject):
    done = pyqtSignal(int, object)           # generation, FleetResult or the exception


def _pose_count(p) -> int:
    """Poses in all records (how big the analysis is), from the raw data."""
    vs = p.fleet.get('vehicles') if isinstance(p.fleet, dict) else None
    return sum(len(r['poses']) for v in (vs if isinstance(vs, list) else []) if isinstance(v, dict)
               for r in (v.get('records') if isinstance(v.get('records'), list) else [])
               if isinstance(r, dict) and isinstance(r.get('poses'), dict))


def _fit(t, rows, most=6):
    t.setFixedHeight(ROW_H * (min(max(rows, 1), most) + 1) + 6)


class FleetPage(QWidget):
    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        self.result: FL.FleetResult | None = None
        self._dirty = True
        self._filling = False
        self.sel_vehicle = ''              # vehicle id
        self.sel_record = ''               # record id (of the selected vehicle)
        self.sel_cell = None               # (vehicle id, plan tag)
        self._edits_vid = ''               # the vehicle the name, VIN and note fields show
        self._gen = 0                      # bumped by every refresh; a worker's result for an older one is dropped
        self.busy = False                  # a big fleet is being analysed on a worker thread
        self._sig = _Sig()
        self._sig.done.connect(self._done)
        self.setObjectName('base')
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        split = QSplitter(Qt.Orientation.Horizontal)
        split.setHandleWidth(2)
        split.addWidget(self._left())
        right = QSplitter(Qt.Orientation.Vertical)
        right.setHandleWidth(2)
        right.addWidget(self._top())
        right.addWidget(self._bottom())
        right.setSizes([640, 300])
        split.addWidget(right)
        split.setSizes([430, 1170])
        lay.addWidget(split)
        ctx.store.changed.connect(self._on_change)
        ctx.store.project_replaced.connect(self._on_replaced)
        THEME.changed.connect(self._on_change)
        self.refresh()

    # ------------------------------------------------------------------ the left panel
    def _left(self):
        w = QWidget()
        w.setObjectName('panel')
        ll = QVBoxLayout(w)
        ll.setContentsMargins(8, 6, 8, 8)
        ll.setSpacing(4)
        top = QHBoxLayout()
        top.addWidget(label('Fleet'))
        top.addStretch(1)
        top.addWidget(label('the rig on every vehicle', muted=True, small=True))
        ll.addLayout(top)
        ll.addWidget(hline())
        ll.addWidget(label('Each vehicle built to this design and its measurements (records): the measured poses '
                           'of its sensors on a date, from an installation check, an extrinsics export, nuScenes '
                           'tables, a Kalibr camchain or a variant. Compared with the plan, over time and across the '
                           'fleet.',
                           muted=True, small=True, wrap=True))
        # ---- the plan and the tolerances
        ll.addSpacing(4)
        ll.addWidget(label('Plan', bold=True))
        self.plan_combo = QComboBox()
        self.plan_combo.setToolTip('The variant the vehicles are compared with: the active one, or a chosen variant')
        self.plan_combo.activated.connect(self._plan_chosen)
        ll.addWidget(self.plan_combo)
        ll.addSpacing(4)
        ll.addWidget(label('Tolerance', bold=True))
        grid = QGridLayout()
        grid.setVerticalSpacing(2)
        grid.setHorizontalSpacing(6)
        grid.setColumnStretch(2, 1)
        self.fields = {}
        for i, (key, text, unit, dec, step, tip) in enumerate(FIELDS):
            f = ScrubSpinBox('', 'plain', dec, unit, step=step, width=110)
            f.setToolTip(tip)
            f.committed.connect(lambda v, k=key, t=text, u=unit: self._set(f'{t.lower()} {v:g} {u}'.strip(), **{k: v}))
            a = label(text, ink2=True)
            a.setFixedWidth(96)
            a.setToolTip(tip)
            grid.addWidget(a, i // 1, 0)
            grid.addWidget(f, i // 1, 1)
            self.fields[key] = f
        ll.addLayout(grid)
        # ---- vehicles
        ll.addSpacing(6)
        hv = QHBoxLayout()
        hv.addWidget(label('Vehicles', bold=True))
        hv.addStretch(1)
        self.b_add_vehicle = tool_button('plus', 'Add a vehicle built to this design')
        self.b_add_vehicle.clicked.connect(lambda: self.add_vehicle())
        self.b_del_vehicle = tool_button('trash', 'Delete the selected vehicle and its records')
        self.b_del_vehicle.clicked.connect(lambda: self.delete_vehicle())
        hv.addWidget(self.b_add_vehicle)
        hv.addWidget(self.b_del_vehicle)
        ll.addLayout(hv)
        self.vtable = _table(('Vehicle', 'VIN', 'Records', 'Latest', 'Status'))
        hh = self.vtable.horizontalHeader()
        for c in range(5):
            hh.setSectionResizeMode(c, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.vtable.itemSelectionChanged.connect(self._vehicle_picked)
        ll.addWidget(self.vtable)
        vg = QGridLayout()
        vg.setVerticalSpacing(2)
        vg.setHorizontalSpacing(6)
        self.v_edits = {}
        for i, (key, text, tip) in enumerate((('name', 'Name', 'The vehicle\'s name in this page (V1, a fleet number)'),
                                               ('vin', 'VIN', 'Vehicle identification number'),
                                               ('note', 'Note', 'Anything worth knowing: build date, rig revision'))):
            e = QLineEdit()
            e.setToolTip(tip)
            e.editingFinished.connect(lambda k=key: self._vehicle_field(k))
            a = label(text, ink2=True)
            a.setFixedWidth(40)
            vg.addWidget(a, i, 0)
            vg.addWidget(e, i, 1)
            self.v_edits[key] = e
        ll.addLayout(vg)
        # ---- records of the selected vehicle
        ll.addSpacing(6)
        hr = QHBoxLayout()
        self.rec_title = label('Records', bold=True)
        hr.addWidget(self.rec_title)
        hr.addStretch(1)
        self.b_import = button('Import…', None, 'import',
                               'Import measured poses for the selected vehicle: an installation check report, '
                               'extrinsics JSON/YAML/CSV, nuScenes calibrated_sensor.json or a Kalibr camchain')
        self.b_import.clicked.connect(lambda: self.import_files())
        self.b_from_variant = button('Variant…', 'ghost', None, 'Take the poses of a variant of this project as a '
                                                                'record (an as-built variant, a survey typed in)')
        self.b_from_variant.clicked.connect(self._variant_menu)
        self.b_new_record = tool_button('plus', 'A new record to type poses into')
        self.b_new_record.clicked.connect(lambda: self.new_record())
        self.b_del_record = tool_button('trash', 'Delete the selected record')
        self.b_del_record.clicked.connect(lambda: self.delete_record())
        for b in (self.b_import, self.b_from_variant, self.b_new_record, self.b_del_record):
            hr.addWidget(b)
        ll.addLayout(hr)
        self.rtable = _table(('Date', 'Kind', 'Source', 'Poses'), editable=True)
        self.rtable.setToolTip('Double-click a date to change it (YYYY-MM-DD)')
        hh = self.rtable.horizontalHeader()
        for c in range(4):
            hh.setSectionResizeMode(c, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.rtable.itemSelectionChanged.connect(self._record_picked)
        self.rtable.itemChanged.connect(self._record_edited)
        ll.addWidget(self.rtable)
        self.rec_note = label('', muted=True, small=True, wrap=True)
        ll.addWidget(self.rec_note)
        # ---- poses of the selected record
        hp = QHBoxLayout()
        hp.addWidget(label('Poses', bold=True))
        hp.addWidget(label('vehicle frame, m and °', muted=True, small=True))
        hp.addStretch(1)
        self.add_tag = QComboBox()
        self.add_tag.setToolTip('A sensor of the plan this record has no pose for: Add puts in its planned pose to '
                                'edit')
        self.add_tag.setMinimumWidth(70)
        hp.addWidget(self.add_tag)
        self.b_add_pose = button('Add', 'ghost', None,
                                 'Add the chosen sensor to the record (its planned pose, to edit)')
        self.b_add_pose.clicked.connect(lambda: self.add_pose())
        hp.addWidget(self.b_add_pose)
        self.b_del_pose = tool_button('trash', 'Remove the selected pose from the record')
        self.b_del_pose.clicked.connect(lambda: self.delete_pose())
        hp.addWidget(self.b_del_pose)
        ll.addLayout(hp)
        self.ptable = _table(('Sensor',) + tuple(f'{t} {u}' for _, t, u, _ in POSE_COLS), editable=True)
        self.ptable.setToolTip('Double-click a number to change it; units and sums work (950 mm, 1.2+0.05)')
        hh = self.ptable.horizontalHeader()
        for c in range(7):
            hh.setSectionResizeMode(c, QHeaderView.ResizeMode.Stretch if c else QHeaderView.ResizeMode.ResizeToContents)
        self.ptable.itemChanged.connect(self._pose_edited)
        ll.addWidget(self.ptable)
        ll.addStretch(1)
        self.status = StatusLine(None, '')
        ll.addWidget(self.status)
        bl = QHBoxLayout()
        self.b_median = button('Variant from the fleet median', 'ghost', None,
                               'Copy the plan with every sensor measured on two or more vehicles moved to the fleet\'s '
                               'median as-built pose, as one undo step')
        self.b_median.clicked.connect(lambda: self.make_median_variant())
        bl.addWidget(self.b_median)
        bl.addStretch(1)
        self.b_export = button('Export…', None, 'export', 'The fleet as JSON (every record and the analysis), the '
                                                           'matrix as CSV and a PNG of the matrix and drift charts')
        self.b_export.clicked.connect(lambda: self.export())
        bl.addWidget(self.b_export)
        ll.addLayout(bl)
        sc = QScrollArea()
        sc.setWidgetResizable(True)
        sc.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        sc.setWidget(w)
        sc.setMinimumWidth(380)
        return sc

    # ------------------------------------------------------------------ the right side
    def _top(self):
        tw = QWidget()
        tw.setObjectName('panel')
        tl = QVBoxLayout(tw)
        tl.setContentsMargins(10, 8, 10, 8)
        tl.setSpacing(8)
        tiles = QHBoxLayout()
        tiles.setSpacing(8)
        self.tiles = {}
        for key, caption, tip in (
                ('vehicles', 'Vehicles', 'Vehicles in the fleet and their records'),
                ('within', 'Within tolerance', 'Measured sensors within the tolerance in their latest record'),
                ('off', 'Adjust · re-mount', 'Measured sensors beyond the tolerance (adjust) or three times it '
                                             '(re-mount) in their latest record'),
                ('drift', 'Drifting · jumps', 'Sensors whose trend leaves the tolerance within the drift window, and '
                                              'latest steps off the earlier course'),
                ('fleet', 'Fleet outliers', 'Vehicles whose sensor stands out from the fleet median, and sensors the '
                                            'whole fleet has off the plan')):
            t = self.tiles[key] = Tile(caption, tip)
            tiles.addWidget(t, 1)
        tl.addLayout(tiles)
        self.matrix = FleetMatrix()
        self.matrix.cell_clicked.connect(self.select_cell)
        self.matrix.vehicle_clicked.connect(self.select_vehicle)
        msc = self.matrix_scroll = QScrollArea()
        msc.setWidgetResizable(True)
        msc.setWidget(self.matrix)
        msc.setFrameShape(QScrollArea.Shape.NoFrame)
        tl.addWidget(msc, 3)
        self.detail = StatusLine(None, '')
        tl.addWidget(self.detail)
        self.chart = DriftChart()
        tl.addWidget(self.chart, 4)
        return tw

    def _bottom(self):
        sp = QSplitter(Qt.Orientation.Horizontal)
        sp.setHandleWidth(2)
        fw = QWidget()
        fw.setObjectName('panel')
        fl = QVBoxLayout(fw)
        fl.setContentsMargins(10, 6, 10, 8)
        fl.setSpacing(4)
        fl.addWidget(label('Every sensor across the fleet', bold=True))
        self.ftable = _table(FLEET_COLS)
        self.ftable.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.ftable.setToolTip('Median Δ: the fleet median of the vehicles\' latest offsets from the plan. Spread: the '
                               'robust standard deviation (1.4826 × median absolute deviation) of the worst axis')
        hh = self.ftable.horizontalHeader()
        for c in range(len(FLEET_COLS)):
            hh.setSectionResizeMode(c, QHeaderView.ResizeMode.ResizeToContents)
        hh.setStretchLastSection(True)
        fl.addWidget(self.ftable, 1)
        sp.addWidget(fw)
        nw = QWidget()
        nw.setObjectName('panel')
        nl = QVBoxLayout(nw)
        nl.setContentsMargins(10, 6, 10, 8)
        nl.setSpacing(4)
        nl.addWidget(label('Findings', bold=True))
        self.find_host = QWidget()
        self.find_lay = QVBoxLayout(self.find_host)
        self.find_lay.setContentsMargins(0, 0, 0, 0)
        self.find_lay.setSpacing(4)
        self.find_lay.addStretch(1)
        sc = QScrollArea()
        sc.setWidgetResizable(True)
        sc.setWidget(self.find_host)
        nl.addWidget(sc, 1)
        self.findings: list[StatusLine] = []
        sp.addWidget(nw)
        sp.setSizes([780, 420])
        return sp

    # ------------------------------------------------------------------ edits (each one undo step)
    @property
    def settings(self) -> FL.FleetSettings:
        return FL.settings_of(self.ctx.store.project)

    def _edit(self, text, fn):
        def run(p):
            if not isinstance(p.fleet, dict):
                p.fleet = {}
            return fn(p.fleet)
        return self.ctx.store.edit('fleet', run, f'Fleet · {text}')

    def _set(self, text, **kw):
        """Change settings as one undo step; False when nothing changes (a clamped value shows what is stored)."""
        st = self.settings
        for k, v in kw.items():
            setattr(st, k, v)
        new = FL.FleetSettings.from_dict(st.to_dict()).to_dict()
        if new == self.settings.to_dict():
            self._sync_settings()
            return False

        def fn(f):
            f['settings'] = new
        self._edit(text, fn)
        return True

    def _plan_chosen(self, i):
        uid = self.plan_combo.itemData(i) or ''
        name = self.plan_combo.itemText(i)
        self._set(f'plan {name}', variant=uid)

    def add_vehicle(self, name='', vin='', note=''):
        v = self._edit('added a vehicle', lambda f: FL.add_vehicle(f, name, vin, note))
        self.sel_vehicle, self.sel_record = v['id'], ''
        self._reselect()
        self.v_edits['name'].setFocus()
        self.v_edits['name'].selectAll()
        return v['id']

    def delete_vehicle(self, vid=None):
        vid = vid or self.sel_vehicle
        v = FL.vehicle({'vehicles': FL.vehicles(self.ctx.store.project)}, vid) if vid else None
        if v is None:
            return False
        self._edit(f'deleted {v["name"]}', lambda f: FL.remove_vehicle(f, vid))
        self.ctx.msg(f'Deleted {v["name"]} and its {len(v["records"])} record(s) (Ctrl+Z brings them back)', 5000)
        return True

    def _vehicle_field(self, key):
        if self._filling or not self._edits_vid:
            return
        v = FL.vehicle({'vehicles': FL.vehicles(self.ctx.store.project)}, self._edits_vid)
        text = self.v_edits[key].text()
        if v is None or v.get(key, '') == (text.strip() if key != 'note' else text):
            return
        self._edit(f'{v["name"]} {key}', lambda f: FL.set_vehicle(f, v['id'], **{key: text}))

    def _variant_menu(self):
        m = QMenu(self)
        for v in self.ctx.store.project.variants:
            m.addAction(v.name, lambda uid=v.uid: self.import_variant(uid))
        m.exec(self.b_from_variant.mapToGlobal(self.b_from_variant.rect().bottomLeft()))

    def _add_records(self, recs, text):
        """Add records to the selected vehicle (a new one when there is none) as one undo step."""
        vid = self.sel_vehicle if self._vehicle() is not None else ''

        def fn(f):
            v = FL.vehicle(f, vid) if vid else None
            if v is None:
                v = FL.add_vehicle(f)
            ids = [FL.add_record(f, v['id'], r)['id'] for r in recs]
            return v['id'], ids
        vid, ids = self._edit(text, fn)
        self.sel_vehicle, self.sel_record = vid, ids[-1] if ids else ''
        self._reselect()
        return ids

    def import_files(self, paths=None):
        """Read records from files (a dialog when none are given) into the selected vehicle; reports what matched the
        plan and what could not be read."""
        if paths is None:
            v = self._vehicle()
            paths, _ = QFileDialog.getOpenFileNames(
                self, f'Import measured poses{" for " + v["name"] if v else ""}', '',
                'Calibrations (*.json *.yaml *.yml *.csv);;All files (*)')
        if not paths:
            return []
        p = self.ctx.store.project
        recs, notes, errors = [], [], []
        for path in paths:
            try:
                r = FL.read_file(path, p)
            except FL.FleetImportError as e:
                errors.append(f'{os.path.basename(path)}: {e}')
                continue
            except Exception as e:                     # a broken file must not take the page down
                errors.append(f'{os.path.basename(path)}: {type(e).__name__}: {e}')
                continue
            recs.append(r)
            notes.append(FL.import_report(r, p)['text'])
        ids = self._add_records(recs, f'imported {len(recs)} record{"" if len(recs) == 1 else "s"}') if recs else []
        v = self._vehicle()
        if recs:
            self.status.set('warning' if errors or any('not in the plan' in n for n in notes) else 'good',
                            f'Imported into {v["name"]}: ' + '; '.join(notes) +
                            ('. Not read: ' + '; '.join(errors) if errors else ''))
            self.ctx.msg(f'Imported {len(recs)} record(s) into {v["name"]}', 5000)
        else:
            self.status.set('serious', 'Nothing imported. ' + '; '.join(errors))
            self.ctx.msg('Nothing imported: ' + (errors[0] if errors else ''), 6000)
        return ids

    def import_variant(self, uid):
        p = self.ctx.store.project
        var = p.variant_by_uid(uid)
        if var is None:
            return []
        r = FL.record_from_variant(p, var)
        ids = self._add_records([r], f'record from {var.name}')
        self.status.set('good', 'Added ' + FL.import_report(r, p)['text'])
        return ids

    def new_record(self, date=''):
        ids = self._add_records([FL.manual_record(date)], 'new record')
        self.status.set('good', 'A new record dated today: add sensors with Add, then type in their measured poses.')
        return ids[0] if ids else ''

    def delete_record(self, rid=None):
        rid = rid or self.sel_record
        v = self._vehicle()
        r = next((x for x in v['records'] if x['id'] == rid), None) if v else None
        if r is None:
            return False
        self._edit(f'deleted the {r["date"] or "undated"} record of {v["name"]}',
                   lambda f: FL.remove_record(f, v['id'], rid))
        self.sel_record = ''
        self._reselect()
        return True

    def set_record_date(self, rid, text):
        v = self._vehicle()
        if v is None:
            return False
        try:
            date = FL.normalise_date(text)
        except ValueError as e:
            self.status.set('warning', str(e))
            self.ctx.msg(str(e), 5000)
            self._fill_records()
            return False
        self._edit(f'record of {v["name"]} dated {date}', lambda f: FL.set_record(f, v['id'], rid, date=date))
        return True

    def add_pose(self, tag=None):
        rec = self._record()
        tag = tag or self.add_tag.currentText()
        if rec is None or not tag:
            return False
        plan = {ps['tag']: ps for ps in self.result.plan} if self.result else {}
        ps = plan.get(tag)
        if ps is None:
            return False
        v = self._vehicle()
        self._edit(f'{tag} added to the record of {v["name"]}',
                   lambda f: FL.set_pose(f, v['id'], rec['id'], tag, dict(ps['pose'])))
        return True

    def delete_pose(self, tag=None):
        rec = self._record()
        if rec is None:
            return False
        if tag is None:
            rows = self.ptable.selectionModel().selectedRows()
            if not rows:
                return False
            tag = self.ptable.item(rows[0].row(), 0).text()
        v = self._vehicle()
        self._edit(f'{tag} removed from the record of {v["name"]}', lambda f: FL.set_pose(f, v['id'], rec['id'], tag,
                                                                                           None))
        return True

    def set_pose_value(self, tag, key, value):
        rec, v = self._record(), self._vehicle()
        if rec is None or tag not in rec['poses']:
            return False
        q = dict(rec['poses'][tag])
        q[key] = float(value)
        if FL.clean_pose(q) is None or q == rec['poses'][tag]:
            self._fill_poses()
            return False
        self._edit(f'{tag} {key} in the record of {v["name"]}', lambda f: FL.set_pose(f, v['id'], rec['id'], tag, q))
        return True

    def make_median_variant(self):
        res = self.result
        if self.busy:
            self.ctx.msg('The fleet is still being analysed: try again in a moment.', 4000)
            return None
        if res is None or not FL.median_poses(res):
            self.ctx.msg('No sensor is measured on two or more vehicles yet.', 5000)
            return None
        n = len(FL.median_poses(res))
        if self.ctx.store.project.variant_by_uid(res.variant_uid) is None:
            return None
        v = self.ctx.store.edit('variants', lambda p: FL.median_variant(p, res),
                                'Fleet · variant from the fleet median')
        win = getattr(self.ctx, 'window', None)
        if win is not None and hasattr(win, '_fill_variant_combo'):
            win._fill_variant_combo()
        self.ctx.msg(f'Created variant {v.name}: {n} sensor(s) at the fleet median as-built pose', 6000)
        return v

    # ------------------------------------------------------------------ table edits
    def _record_edited(self, it):
        if self._filling or it.column() != 0:
            return
        rid, text = it.data(Qt.ItemDataRole.UserRole), it.text()
        if rid:                                   # after the table's own signal: the edit rebuilds the table
            QTimer.singleShot(0, lambda: self.set_record_date(rid, text))

    def _pose_edited(self, it):
        if self._filling or it.column() == 0:
            return
        tag = self.ptable.item(it.row(), 0).text()
        key, _, unit, _ = POSE_COLS[it.column() - 1]
        try:
            val = parse_number(it.text(), 'length' if unit == 'm' else 'angle', 'm')
        except ValueError:
            self.ctx.msg(f'"{it.text()}" is not a number', 4000)
            QTimer.singleShot(0, self._fill_poses)
            return
        QTimer.singleShot(0, lambda: self.set_pose_value(tag, key, val))   # after the table's own signal

    # ------------------------------------------------------------------ selection
    def _vehicle(self):
        return FL.vehicle({'vehicles': FL.vehicles(self.ctx.store.project)}, self.sel_vehicle) \
            if self.sel_vehicle else None

    def _record(self):
        v = self._vehicle()
        return next((r for r in v['records'] if r['id'] == self.sel_record), None) if v else None

    def _vehicle_picked(self):
        if self._filling:
            return
        rows = self.vtable.selectionModel().selectedRows()
        if rows:                                  # after the table's own signal: selecting refills the table
            vid = self.vtable.item(rows[0].row(), 0).data(Qt.ItemDataRole.UserRole)
            QTimer.singleShot(0, lambda: self.select_vehicle(vid))

    def _record_picked(self):
        if self._filling:
            return
        rows = self.rtable.selectionModel().selectedRows()
        if rows:
            self.sel_record = self.rtable.item(rows[0].row(), 0).data(Qt.ItemDataRole.UserRole)
            self._fill_record_detail()

    def select_vehicle(self, vid):
        if vid != self.sel_vehicle:
            self.sel_vehicle, self.sel_record = vid, ''
            if self.sel_cell and self.sel_cell[0] != vid:
                self.sel_cell = None
        self._fill_selection()

    def select_cell(self, vid, tag):
        self.sel_cell = (vid, tag)
        if vid != self.sel_vehicle:
            self.sel_vehicle, self.sel_record = vid, ''
        c = self.result.cell(vid, tag) if self.result else None
        if c is not None:
            self.sel_record = c['record']
        self._fill_selection()

    # ------------------------------------------------------------------ state
    def _reselect(self):
        """After an edit that moved the selection: the edit's own refresh ran already while the page shows."""
        if self._dirty or self.result is None:
            self.refresh()
        else:
            self._fill_selection()

    def _on_change(self, *_):
        if self.isVisible():
            self.refresh()
        else:
            self._dirty = True

    def _on_replaced(self):
        self.sel_vehicle, self.sel_record, self.sel_cell = '', '', None
        self._on_change()

    def showEvent(self, e):
        super().showEvent(e)
        if self._dirty:
            self.refresh()

    def activate(self):
        if self._dirty:
            self.refresh()

    def refresh(self):
        """Compute the fleet again: at once for a usual fleet, on a worker thread for a big one (a copy of the fleet
        and the plan, so edits meanwhile cannot disturb it; a newer refresh supersedes it)."""
        p = self.ctx.store.project
        self._dirty = False
        self._gen += 1
        if _pose_count(p) <= ASYNC_POSES:
            self.busy = False
            self._show(FL.run(p))
            return
        gen = self._gen
        snap = SimpleNamespace(fleet=pickle.loads(pickle.dumps(p.fleet, pickle.HIGHEST_PROTOCOL)))
        plan = Variant.from_dict(FL.plan_variant(p).to_dict())
        self.busy = True
        self.detail.set(None, 'Analysing the fleet…')
        sig = self._sig

        def work():
            try:
                res = FL.run(snap, variant=plan)
            except Exception as e:                  # reported on the page, never lost in a thread
                res = e
            try:
                sig.done.emit(gen, res)
            except RuntimeError:                    # the page was closed meanwhile
                pass
        threading.Thread(target=work, daemon=True).start()

    def _done(self, gen, res):
        if gen != self._gen:                        # an edit since: its own run follows
            return
        self.busy = False
        if isinstance(res, Exception):
            self.status.set('serious', f'The fleet analysis failed: {type(res).__name__}: {res}')
            return
        self._show(res)

    def _show(self, res):
        self.result = res
        vs = res.vehicles
        ids = [v['id'] for v in vs]
        if self.sel_vehicle not in ids:
            self.sel_vehicle, self.sel_record = (ids[0] if ids else ''), ''
        if self.sel_cell and (self.sel_cell[0] not in ids or self.sel_cell[1] not in [ps['tag'] for ps in res.plan]):
            self.sel_cell = None
        st = res.settings
        bad = sum(f['severity'] != 'good' for f in res.findings)
        self._summary = (f'{len(vs)} vehicle{"" if len(vs) == 1 else "s"} against {res.variant}: '
                         + ('nothing to look at' if not bad else f'{bad} finding{"s" if bad > 1 else ""}')
                         + f' (tolerance {st.pos_tol_mm:g} mm / {st.ang_tol_deg:g}°)')
        self._sync_settings()
        self._fill_tiles()
        self._fill_fleet()
        self._fill_findings()
        self._fill_selection()
        self.b_median.setEnabled(bool(FL.median_poses(res)))
        self.b_median.setProperty('role', 'primary' if any(f['systematic'] for f in res.fleet) else 'ghost')
        self.b_median.style().unpolish(self.b_median)
        self.b_median.style().polish(self.b_median)
        self.b_export.setEnabled(bool(vs))

    def _sync_settings(self):
        p = self.ctx.store.project
        st = self.settings
        for key, f in self.fields.items():
            f.set_value(getattr(st, key))
        self.plan_combo.blockSignals(True)
        self.plan_combo.clear()
        self.plan_combo.addItem(f'Active variant · {p.variant.name}', '')
        for v in p.variants:
            self.plan_combo.addItem(v.name, v.uid)
        i = self.plan_combo.findData(st.variant) if st.variant and p.variant_by_uid(st.variant) else 0
        self.plan_combo.setCurrentIndex(max(i, 0))
        self.plan_combo.blockSignals(False)

    def _fill_tiles(self):
        res = self.result
        cells = list(res.cells.values())
        n = len(cells)
        cnt = {k: sum(c['verdict'] == k for c in cells) for k in FL.VERDICTS}
        recs = sum(v['records'] for v in res.vehicles)
        self.tiles['vehicles'].set(f'{len(res.vehicles)}', '', f'{recs} record{"" if recs == 1 else "s"} · '
                                                               f'{len(res.plan)} sensors in the plan')
        if n:
            self.tiles['within'].set(f'{cnt["ok"]}', f'of {n}', 'latest record within tolerance',
                                     'good' if cnt['ok'] == n else 'warning')
            off = cnt['adjust'] + cnt['re-mount']
            self.tiles['off'].set(f'{cnt["adjust"]} · {cnt["re-mount"]}', '', 'adjust · re-mount',
                                  'serious' if cnt['re-mount'] else 'warning' if off else 'good')
        else:
            self.tiles['within'].set('—', '', 'nothing measured')
            self.tiles['off'].set('—', '', 'nothing measured')
        nd, nj = len(res.flagged('drift')), len(res.flagged('jump'))
        self.tiles['drift'].set(f'{nd} · {nj}', '', 'drifting · jumps' if n else 'needs dated records',
                                ('warning' if nd or nj else 'good') if n else None)
        no = sum(1 for f in res.fleet for o in f['outliers'] if not o['cause'])
        ns = sum(f['systematic'] for f in res.fleet)
        self.tiles['fleet'].set(f'{no}', '', f'vehicles stand out · {ns} sensor{"" if ns == 1 else "s"} off the plan '
                                             'fleet-wide' if ns else 'vehicles stand out from the fleet',
                                ('warning' if no or ns else 'good') if any(f['n'] >= FL.MIN_FLEET for f in res.fleet)
                                else None)

    def _fill_vehicles(self):
        res = self.result
        t = self.vtable
        self._filling = True
        t.setRowCount(len(res.vehicles))
        sel_row = -1
        for i, v in enumerate(res.vehicles):
            sev = SEV_OF.get(v['verdict']) if v['records'] else None
            t.setItem(i, 0, _item(v['name'], bold=v['id'] == self.sel_vehicle, data=v['id'], tip=v['note'] or None))
            t.setItem(i, 1, _item(v['vin']))
            t.setItem(i, 2, _item(str(v['records']), num=True))
            t.setItem(i, 3, _item(v['latest'][:10] or '—', num=True))
            word = (FL.VERDICT_LABEL[v['verdict']] if v['records'] and v['worst'] is not None else
                    'Not measured' if v['records'] else 'No records')
            flags = sum(len(x) for x in v['flags'].values())
            st_it = _item(word + (f' · {flags} flag{"" if flags == 1 else "s"}' if flags else ''),
                          color=QColor(STATUS[sev]) if sev else THEME.color('muted'))
            t.setItem(i, 4, st_it)
            if v['id'] == self.sel_vehicle:
                sel_row = i
        if sel_row >= 0:
            t.selectRow(sel_row)
        else:
            t.clearSelection()
        _fit(t, len(res.vehicles), 6)
        self._filling = False
        v = self._vehicle()
        self._filling = True
        other = (v['id'] if v else '') != self._edits_vid
        for key, e in self.v_edits.items():
            if other or not e.hasFocus():          # leave a field being typed in alone, unless the vehicle changed
                e.setText(v[key] if v else '')
            e.setEnabled(v is not None)
        self._edits_vid = v['id'] if v else ''
        self._filling = False
        self.b_del_vehicle.setEnabled(v is not None)

    def _fill_records(self):
        v = self._vehicle()
        t = self.rtable
        self._filling = True
        recs = sorted(v['records'], key=lambda r: (FL.date_days(r['date']) is None, FL.date_days(r['date']) or 0)) \
            if v else []
        if v and self.sel_record not in [r['id'] for r in recs]:
            self.sel_record = recs[-1]['id'] if recs else ''
        t.setRowCount(len(recs))
        sel_row = -1
        p = self.ctx.store.project
        for i, r in enumerate(recs):
            rep = FL.import_report(r, p)
            t.setItem(i, 0, _item(r['date'] or '—', num=True, editable=True, data=r['id'],
                                  tip='Double-click to change the date (YYYY-MM-DD)',
                                  color=None if FL.parse_date(r['date']) else QColor(STATUS['warning'])))
            t.setItem(i, 1, _item(FL.KIND_LABEL.get(r['kind'], r['kind'])))
            t.setItem(i, 2, _item(r['source'], tip=r['note'] or None))
            extra = len(rep['unknown']) + len(rep['hidden']) + len(rep['duplicates'])
            t.setItem(i, 3, _item(f'{len(r["poses"])}' + (f' · {extra} ?' if extra else ''), num=True,
                                  tip=rep['text'], color=QColor(STATUS['warning']) if extra else None))
            if r['id'] == self.sel_record:
                sel_row = i
        if sel_row >= 0:
            t.selectRow(sel_row)
        _fit(t, len(recs), 6)
        self._filling = False
        self.rec_title.setText(f'Records of {v["name"]}' if v else 'Records')
        for b in (self.b_new_record, self.b_from_variant):
            b.setEnabled(True)
        self.b_del_record.setEnabled(bool(recs))
        self._fill_record_detail()

    def _fill_record_detail(self):
        r = self._record()
        p = self.ctx.store.project
        if r is None:
            self.rec_note.setText('')
        else:
            rep = FL.import_report(r, p)
            self.rec_note.setText((r['note'] + ' · ' if r['note'] else '') + rep['text'])
        self._fill_poses()

    def _fill_poses(self):
        r = self._record()
        t = self.ptable
        self._filling = True
        tags = list(r['poses']) if r else []
        t.setRowCount(len(tags))
        plan = {ps['tag'].lower(): ps for ps in self.result.plan} if self.result else {}
        for i, tag in enumerate(tags):
            q = r['poses'][tag]
            ps = plan.get(tag.lower())
            it = _item(tag, bold=True, tip='Not in the plan' if ps is None else None,
                       color=QColor(STATUS['warning']) if ps is None else None)
            if ps is not None:
                it.setData(Qt.ItemDataRole.DecorationRole, sensor_color(ps))
            t.setItem(i, 0, it)
            for j, (key, _, unit, dec) in enumerate(POSE_COLS, 1):
                t.setItem(i, j, _item(fmt_num(q[key], dec), num=True, editable=True,
                                      tip=f'{q[key]:.6f} {unit}'))
        _fit(t, len(tags), 8)
        self._filling = False
        have = {x.lower() for x in tags}
        cur = self.add_tag.currentText()
        self.add_tag.clear()
        if r is not None and self.result is not None:
            self.add_tag.addItems([ps['tag'] for ps in self.result.plan if ps['tag'].lower() not in have])
        if cur:
            self.add_tag.setCurrentText(cur)
        self.add_tag.setEnabled(r is not None and self.add_tag.count() > 0)
        self.b_add_pose.setEnabled(r is not None and self.add_tag.count() > 0)
        self.b_del_pose.setEnabled(bool(tags))

    def _fill_selection(self):
        self._fill_vehicles()
        self._fill_records()
        res = self.result
        self.matrix.set_result(res, self.sel_cell, self.sel_vehicle)
        self.matrix_scroll.setMaximumHeight(self.matrix.minimumHeight() + 4)     # no empty band under a small fleet
        c = res.cell(*self.sel_cell) if self.sel_cell else None
        v = res.vehicle(self.sel_cell[0]) if self.sel_cell else None
        self.chart.set_data(res, c, v['name'] if v else '')
        if self.sel_cell and c is None and v is not None:
            self.detail.set('warning', f'{v["name"]} · {self.sel_cell[1]}: not measured in any of its records.')
        elif c is not None:
            d = c['d']
            dr = c['drift']
            text = (f'{v["name"]} · {c["tag"]} · {FL.VERDICT_LABEL[c["verdict"]]} in the latest record '
                    f'({c["date"] or "undated"}, {c["source"] or FL.KIND_LABEL[c["record_kind"]]}): '
                    f'Δx {d[0]:+.1f}, Δy {d[1]:+.1f}, Δz {d[2]:+.1f} mm; Δyaw {d[3]:+.2f}°, Δpitch {d[4]:+.2f}°, '
                    f'Δroll {d[5]:+.2f}° ({c["rot_deg"]:.2f}° rotation, {c["ratio"]:.2f}× the tolerance)')
            if dr and dr['trend']:
                text += '. Drifting: ' + ('already out of tolerance' if dr['days_to_tol'] == 0 else
                                          f'out of tolerance in about {dr["days_to_tol"]:.0f} days')
            if dr and dr['jump']:
                text += f'. The latest step is a jump ({dr["jump_ratio"]:.1f}× the tolerance off its course)'
            if c['outlier']:
                o = c['outlier']
                text += f'. Fleet outlier in {o["component"]} (robust z {abs(o["z"]):.1f})'
            self.detail.set(SEV_OF[c['verdict']], text + '.')
        else:
            self.detail.set(None, self._summary + ('. Click a cell for its detail and drift.' if res.cells else ''))

    def _fill_fleet(self):
        res = self.result
        t = self.ftable
        rows = [f for f in res.fleet if f['n']]
        t.setRowCount(len(rows))
        st = res.settings
        for i, f in enumerate(rows):
            ps = next(x for x in res.plan if x['tag'] == f['tag'])
            it = _item(f['tag'], bold=True)
            it.setData(Qt.ItemDataRole.DecorationRole, sensor_color(ps))
            t.setItem(i, 0, it)
            t.setItem(i, 1, _item(str(f['n']), num=True))
            m, s = f['median'], f['sigma']
            t.setItem(i, 2, _item(' '.join(f'{x:+.1f}' for x in m[:3]), num=True, tip='Δx Δy Δz, mm'))
            t.setItem(i, 3, _item(' '.join(f'{x:+.2f}' for x in m[3:]), num=True, tip='Δyaw Δpitch Δroll, °'))
            t.setItem(i, 4, _item(f'{max(s[:3]):.1f} mm · {max(s[3:]):.2f}°', num=True))
            t.setItem(i, 5, _item(f'{f["beyond"]} of {f["n"]}', num=True,
                                  color=QColor(STATUS['warning']) if f['beyond'] else None))
            names = ', '.join(o['name'] + (f' ({o["cause"]})' if o['cause'] else '') for o in f['outliers'])
            tip = '\n'.join(f'{o["name"]}: {o["component"]} {o["from_median"]:+.{0 if o["component"] in "xyz" else 2}f}'
                            f'{" mm" if o["component"] in "xyz" else "°"} from the median, robust z {abs(o["z"]):.1f}'
                            + (f' ({o["cause"]})' if o['cause'] else '') for o in f['outliers'])
            t.setItem(i, 6, _item(names or ('—' if f['n'] >= FL.MIN_FLEET else f'needs {FL.MIN_FLEET}'),
                                  tip=tip or None,
                                  color=QColor(STATUS['warning']) if f['outliers'] else THEME.color('muted')))
            t.setItem(i, 7, _item('off the plan' if f['systematic'] else
                                  f'{f["median_ratio"]:.2f}×' if f['median_ratio'] is not None else '—',
                                  color=QColor(STATUS['warning']) if f['systematic'] else None,
                                  tip=f'The fleet median is {f["median_ratio"]:.2f}× the tolerance '
                                      f'({st.pos_tol_mm:g} mm / {st.ang_tol_deg:g}°) from the plan'
                                      if f['median_ratio'] is not None else None))

    def _fill_findings(self):
        for w in self.findings:
            self.find_lay.removeWidget(w)
            w.hide()                    # until the deferred delete runs it would still paint over the new rows
            w.deleteLater()
        self.findings = []
        for f in self.result.findings:
            w = StatusLine(f['severity'], f['message'])
            self.find_lay.insertWidget(self.find_lay.count() - 1, w)
            self.findings.append(w)

    # ------------------------------------------------------------------ export
    def export(self, folder=None):
        res = self.result
        if res is None or not res.vehicles:
            return []
        if self.busy:
            self.ctx.msg('The fleet is still being analysed: try again in a moment.', 4000)
            return []
        if folder is None:
            folder = QFileDialog.getExistingDirectory(self, 'Export the fleet into')
            if not folder:
                return []
        cells = [self.sel_cell] if self.sel_cell else None
        paths = FL.export(res, self.ctx.store.project, folder, cells=cells)
        self.ctx.msg(f'Exported the fleet to {folder}', 5000)
        return paths
