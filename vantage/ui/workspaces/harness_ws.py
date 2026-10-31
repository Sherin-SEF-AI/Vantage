"""Analysis ▸ Cable harness: every camera's and sensor's cable routed along the body to an entry point and inside to
the compute box (core.harness), drawn on the body from above, from the side or in 3D; each run's length against what
its link allows, the GNSS cables' RF margin and the power pairs' drop; the entry points (grommets, door seals) to
edit; the harness sheet as JSON, CSV and PNG. Worked out on a thread after every edit while the page shows."""
from __future__ import annotations

import re

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import (QAbstractItemView, QComboBox, QFileDialog, QGridLayout, QHBoxLayout, QHeaderView,
                             QScrollArea, QSplitter, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from ...core import harness as HN
from ...core.mathutil import parse_number
from .. import icons
from ..harnessview import HarnessView, controller, run_qcolor
from ..theme import THEME, STATUS, mono, sans
from ..widgets import fit_width, ROW_H, ScrubSpinBox, SegmentedControl, StatusLine, ToggleSwitch, button, hline, label
from .recording_ws import SEV_WORD, Tile, _table

# (group, key, label, unit, decimals, step, scale, tip); scale: stored value × scale = shown value
FIELDS = (
    ('Routing', 'slack', 'Surface slack', '×', 2, 0.01, 1.0,
     'The path along the body × this: clips, bends and drip loops'),
    ('Routing', 'glass_cost', 'Avoid glass', '×', 0, 1.0, 1.0,
     'A metre on glass costs this many metres elsewhere: routes keep off the glass unless nothing else reaches'),
    ('Routing', 'moving_cost', 'Avoid moving parts', '×', 0, 1.0, 1.0,
     'A metre over a door, the bonnet, the tailgate or a shut line (by part or mount-zone name) costs this many'),
    ('Routing', 'free_cost', 'Free air', '×', 1, 0.1, 1.0,
     'A metre across a gap between separate shells (a mirror, a rig part) costs this many'),
    ('GNSS antenna cable', 'gnss_lna_db', 'Antenna LNA gain', 'dB', 1, 1.0, 1.0,
     'The active antenna\'s LNA gain (typical 28 dB; check your antenna\'s data sheet)'),
    ('GNSS antenna cable', 'gnss_min_db', 'Receiver needs', 'dB', 1, 1.0, 1.0,
     'The gain the receiver needs at its input: the LNA gain less the cable\'s loss (typical; check its manual)'),
    ('GNSS antenna cable', 'connector_db', 'Per connector', 'dB', 2, 0.05, 1.0,
     'Loss of each RF connector (two per cable)'),
    ('Margins', 'warn_share', 'Flag above', '%', 0, 1.0, 100.0,
     'A run longer than this share of what its link allows is flagged'),
)
ECU_FIELDS = (('route_factor', 'Inside slack', '×', 2, 0.01,
               'The inside path (Manhattan, along the trim) × this; the compute box\'s routing slack, as the estimate '
               'elsewhere in the app uses'),
              ('service_loop_m', 'Service loop', 'm', 2, 0.05,
               'Added to every run: the loop left at the compute box (the compute box\'s setting)'))
RUN_COLS = ('Run', 'Length m', 'Use', 'Status', 'Cable', 'Entry', 'To', 'Limit', 'Manhattan m', 'Extra m')
C_RUN, C_LEN, C_USE, C_STATUS, C_CABLE, C_ENTRY, C_TO, C_LIMIT, C_MAN, C_EXTRA = range(10)
ENTRY_COLS = ('Name', 'Kind', 'X m', 'Y m', 'Z m', 'On', 'Max')
E_NAME, E_KIND, E_X, E_Y, E_Z, E_ON, E_CAP = range(7)
KEY = Qt.ItemDataRole.UserRole
MODES = [('top', 'Top'), ('side', 'Side'), ('3d', '3D')]


class _Combo(QComboBox):
    """A picker in a table cell: the wheel scrolls the table, it never changes the value."""

    def wheelEvent(self, e):
        e.ignore()


def _use_sev(r, st) -> str:
    """The status of a run's share of what bounds it (the Use column and the tile)."""
    use = r.get('use')
    if use is None:
        return 'good'
    if r['what'] == 'rf':
        return 'serious' if r['margin_db'] < 0 else 'warning' if r['margin_db'] < HN.RF_WARN_DB else 'good'
    return 'serious' if use > 1 + 1e-9 else 'warning' if use > st.warn_share + 1e-9 else 'good'


def _hide_widgets(table):
    """Hide the pickers in a table's cells before it is filled again: Qt deletes the replaced ones later, and until
    then they would still paint where their cells were."""
    for r in range(table.rowCount()):
        for c in range(table.columnCount()):
            w = table.cellWidget(r, c)
            if w is not None:
                w.hide()


def _parse(text):
    """A typed number ('1.5', '1,5 m'); None for an empty cell; raises ValueError for junk."""
    t = re.sub(r'(?i)\bm\b|db|×|%', '', str(text)).strip()
    if not t:
        return None
    v = parse_number(t, 'plain')
    if v != v or v in (float('inf'), float('-inf')):
        raise ValueError(text)
    return v


class HarnessPage(QWidget):
    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        self.ctl = controller(ctx)
        self._busy = False                     # filling the tables: their edits are not the user's
        self.selected = ''                     # run id
        self.setObjectName('base')
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        split = QSplitter(Qt.Orientation.Horizontal)
        split.setHandleWidth(2)
        split.addWidget(self._left())
        right = QSplitter(Qt.Orientation.Vertical)
        right.setHandleWidth(2)
        tw = QWidget()
        tw.setObjectName('panel')
        tl = QVBoxLayout(tw)
        tl.setContentsMargins(10, 8, 10, 8)
        tl.setSpacing(8)
        tiles = QHBoxLayout()
        tiles.setSpacing(8)
        self.tiles = {}
        for key, caption, tip in (
                ('cables', 'Cables', 'Every cable of the rig: each camera\'s link, each sensor\'s data or RF cable and '
                                     'each lidar\'s, radar\'s and IMU\'s power pair'),
                ('total', 'Total length', 'All cables routed, against the Manhattan estimate of the data and RF runs'),
                ('longest', 'Longest run', 'The longest data or RF run'),
                ('tightest', 'Tightest run', 'The run nearest what bounds it: a camera\'s fitted cable, a link\'s '
                                             'length limit, a GNSS cable\'s RF budget'),
                ('entries', 'Entry points', 'Entry points the routes use, of those switched on')):
            t = self.tiles[key] = Tile(caption, tip)
            tiles.addWidget(t, 1)
        tl.addLayout(tiles)
        vh = QHBoxLayout()
        vh.addWidget(label('The routes', bold=True))
        vh.addStretch(1)
        self.mode = SegmentedControl(MODES, 'top')
        self.mode.setToolTip('From above, from the side or in 3D (drag to orbit)')
        self.mode.changed.connect(self._set_mode)
        vh.addWidget(self.mode)
        tl.addLayout(vh)
        self.view = HarnessView()
        self.view.run_clicked.connect(self.select_run)
        self.view.entry_clicked.connect(self._select_entry)
        tl.addWidget(self.view, 1)
        right.addWidget(tw)
        low = QSplitter(Qt.Orientation.Horizontal)
        low.setHandleWidth(2)
        rw = QWidget()
        rw.setObjectName('panel')
        rl = QVBoxLayout(rw)
        rl.setContentsMargins(10, 6, 10, 8)
        rl.setSpacing(4)
        rl.addWidget(label('Every cable', bold=True))
        self.table = _table(RUN_COLS)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.DoubleClicked
                                   | QAbstractItemView.EditTrigger.EditKeyPressed)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setToolTip('Pick a run\'s cable type and entry point; double-click Extra to add length (a camera\'s '
                              'route extra, or a sensor\'s); select a run to see it on the body')
        self.table.itemChanged.connect(self._run_edited)
        self.table.itemSelectionChanged.connect(self._row_selected)
        hh = self.table.horizontalHeader()
        for c in range(len(RUN_COLS)):
            hh.setSectionResizeMode(c, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(C_TO, QHeaderView.ResizeMode.Stretch)
        rl.addWidget(self.table, 1)
        low.addWidget(rw)
        fw = QWidget()
        fw.setObjectName('panel')
        fl = QVBoxLayout(fw)
        fl.setContentsMargins(10, 6, 10, 8)
        fl.setSpacing(4)
        fl.addWidget(label('Findings', bold=True))
        self.find_host = QWidget()
        self.find_lay = QVBoxLayout(self.find_host)
        self.find_lay.setContentsMargins(0, 0, 0, 0)
        self.find_lay.setSpacing(4)
        self.find_lay.addStretch(1)
        sc = QScrollArea()
        sc.setWidgetResizable(True)
        sc.setWidget(self.find_host)
        fl.addWidget(sc, 1)
        self.findings: list[StatusLine] = []
        low.addWidget(fw)
        low.setSizes([820, 440])
        right.addWidget(low)
        right.setSizes([560, 380])
        split.addWidget(right)
        split.setSizes([470, 1130])
        lay.addWidget(split)
        self.ctl.changed.connect(self._on_result)
        ctx.store.changed.connect(self._on_store)
        ctx.store.project_replaced.connect(self._on_store)
        THEME.changed.connect(self._on_theme)
        self._sync_fields()
        self._on_result()

    # ------------------------------------------------------------------ settings panel
    def _left(self):
        w = QWidget()
        w.setObjectName('panel')
        ll = QVBoxLayout(w)
        ll.setContentsMargins(8, 6, 8, 8)
        ll.setSpacing(4)
        top = QHBoxLayout()
        top.addWidget(label('Cable harness'))
        top.addStretch(1)
        top.addWidget(label('every cable, routed on the body', muted=True, small=True))
        ll.addLayout(top)
        ll.addWidget(hline())
        ll.addWidget(label('Each cable runs from its device along the body to an entry point, then inside to the '
                           'compute box along the trim. Lengths are checked against what each link allows (typical '
                           'limits: each cable type gives its basis), the GNSS cables\' loss against the RF margin and '
                           'the power pairs\' drop as in the power budget.', muted=True, small=True, wrap=True))
        self.enabled = ToggleSwitch('Use the routed lengths everywhere')
        self.enabled.setToolTip('The routed lengths replace the Manhattan estimate in the Issues list, the camera '
                                'cable check, the power budget and the exports')
        self.enabled.clicked.connect(lambda on: self.ctl.set('routed lengths used everywhere' if on else
                                                             'Manhattan estimate used elsewhere', enabled=on))
        ll.addSpacing(4)
        ll.addWidget(self.enabled)
        self.fields = {}
        group, grid = None, None
        for g, key, text, unit, dec, step, scale, tip in FIELDS:
            if g != group:
                if grid is not None:
                    ll.addLayout(grid)
                    if group == 'Routing':
                        self._compute_row(ll)
                group = g
                ll.addSpacing(6)
                ll.addWidget(label(g, bold=True))
                if g == 'GNSS antenna cable':
                    self.rf = QComboBox()
                    for k in HN.RF_CABLES:
                        self.rf.addItem(f'{HN.CABLES[k].label} · {HN.CABLES[k].db_per_m:g} dB/m', k)
                    self.rf.setToolTip('The coax of every GNSS antenna unless picked per run (typical loss at L1, '
                                       '1.575 GHz)')
                    self.rf.currentIndexChanged.connect(
                        lambda _: self._later(self.ctl.set, f'GNSS cable {self.rf.currentText()}',
                                              gnss_cable=self.rf.currentData()))
                    ll.addWidget(self.rf)
                grid = QGridLayout()
                grid.setVerticalSpacing(2)
                grid.setHorizontalSpacing(6)
                grid.setColumnStretch(2, 1)
            f = ScrubSpinBox('', 'plain', dec, unit, step=step, width=110)
            f.setToolTip(tip)
            f.committed.connect(lambda v, k=key, s=scale, t=text, u=unit: self.ctl.set(f'{t} {v:g} {u}', **{k: v / s}))
            self._grid_row(grid, text, f, tip)
            self.fields[key] = (f, scale)
        ll.addLayout(grid)
        # ---- entry points
        ll.addSpacing(6)
        ll.addWidget(label('Entry points', bold=True))
        ll.addWidget(label('Grommets and seals where cables go inside: placed from the body\'s shape until you edit '
                           'them. Positions in metres in the project\'s frame, taken onto the body; Max: the cables it '
                           'takes (empty: no limit). Double-click to edit.', muted=True, small=True, wrap=True))
        self.entries = QTableWidget(0, len(ENTRY_COLS))
        self.entries.setHorizontalHeaderLabels(ENTRY_COLS)
        self.entries.verticalHeader().setVisible(False)
        self.entries.verticalHeader().setDefaultSectionSize(ROW_H)
        self.entries.horizontalHeader().setFixedHeight(ROW_H)
        self.entries.setShowGrid(False)
        self.entries.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.entries.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.entries.setEditTriggers(QAbstractItemView.EditTrigger.DoubleClicked
                                     | QAbstractItemView.EditTrigger.EditKeyPressed)
        eh = self.entries.horizontalHeader()
        eh.setSectionResizeMode(E_NAME, QHeaderView.ResizeMode.Stretch)
        for c in range(1, len(ENTRY_COLS)):
            eh.setSectionResizeMode(c, QHeaderView.ResizeMode.ResizeToContents)
        eh.setSectionResizeMode(E_KIND, QHeaderView.ResizeMode.Fixed)
        self.entries.setColumnWidth(E_KIND, 96)
        self.entries.itemChanged.connect(self._entry_edited)
        self.entries.itemSelectionChanged.connect(self._entry_selected)
        ll.addWidget(self.entries)
        eb = QHBoxLayout()
        self.b_add = button('Add', None, 'plus', 'Add an entry point on the roof (then type where it is)')
        self.b_add.clicked.connect(self.add_entry)
        self.b_remove = button('Remove', 'ghost', 'trash', 'Remove the selected entry point (the last one when none '
                                                           'is selected)')
        self.b_remove.clicked.connect(self.remove_entry)
        self.b_reset = button('From the body', 'ghost', None, 'Back to the entry points placed from the body\'s shape')
        self.b_reset.clicked.connect(self.reset_entries)
        for b in (self.b_add, self.b_remove, self.b_reset):
            eb.addWidget(b)
        eb.addStretch(1)
        ll.addLayout(eb)
        ll.addSpacing(6)
        self.assumptions = label(HN.ASSUMPTIONS + ' ' + HN.LIMITS_TEXT, muted=True, small=True, wrap=True)
        ll.addWidget(self.assumptions)
        ll.addStretch(1)
        sc = QScrollArea()
        sc.setWidgetResizable(True)
        sc.setWidget(w)
        fit_width(sc)
        out = QWidget()
        out.setObjectName('panel')
        ol = QVBoxLayout(out)
        ol.setContentsMargins(0, 0, 0, 8)
        ol.setSpacing(4)
        ol.addWidget(sc, 1)
        self.status = label('', muted=True, small=True, wrap=True)
        self.status.setContentsMargins(8, 0, 8, 0)
        ol.addWidget(self.status)
        bl = QHBoxLayout()
        bl.setContentsMargins(8, 0, 8, 0)
        self.b_defaults = button('Defaults', 'ghost', None, 'Back to the default harness settings and entry points')
        self.b_defaults.clicked.connect(self.ctl.defaults)
        bl.addWidget(self.b_defaults)
        bl.addStretch(1)
        self.b_export = button('Export…', None, 'export', 'Every run with its ends, cable, connectors, length, limit '
                                                           'and path as JSON and CSV, and the drawing as PNG')
        self.b_export.clicked.connect(lambda: self.export())
        bl.addWidget(self.b_export)
        ol.addLayout(bl)
        out.setMinimumWidth(440)
        return out

    def _grid_row(self, grid, text, f, tip):
        a = label(text, ink2=True)
        a.setFixedWidth(128)
        a.setToolTip(tip)
        i = grid.rowCount()
        grid.addWidget(a, i, 0)
        grid.addWidget(f, i, 1)

    def _compute_row(self, ll):
        self.compute = SegmentedControl([('auto', 'Auto'), ('inside', 'Inside'), ('outside', 'Outside')], 'auto')
        self.compute.setToolTip('Where the compute box is: inside the body (the cables go in through an entry '
                                'point) or outside (on a rack: reached along the surface); Auto decides from the body')
        self.compute.changed.connect(lambda k: self.ctl.set(f'compute box {k}', compute=k))
        row = QHBoxLayout()
        a = label('Compute box', ink2=True)
        a.setFixedWidth(128)
        row.addWidget(a)
        row.addWidget(self.compute)
        row.addStretch(1)
        ll.addLayout(row)
        grid = QGridLayout()
        grid.setVerticalSpacing(2)
        grid.setHorizontalSpacing(6)
        grid.setColumnStretch(2, 1)
        self.ecu_fields = {}
        for key, text, unit, dec, step, tip in ECU_FIELDS:
            f = ScrubSpinBox('', 'plain', dec, unit, step=step, width=110)
            f.setToolTip(tip)
            f.committed.connect(lambda v, k=key, t=text, u=unit: self.set_ecu(k, v, f'{t} {v:g} {u}'))
            self._grid_row(grid, text, f, tip)
            self.ecu_fields[key] = f
        ll.addLayout(grid)

    def set_ecu(self, key, value, text) -> bool:
        lim = {'route_factor': (1.0, 3.0), 'service_loop_m': (0.0, 5.0)}[key]
        v = min(max(float(value), lim[0]), lim[1])
        if abs(getattr(self.ctx.store.project.ecu, key) - v) < 1e-12:
            self._sync_fields()
            return False

        def fn(p):
            setattr(p.ecu, key, v)
        self.ctx.store.edit('ecu', fn, f'Compute box · {text}')
        return True

    def _sync_fields(self):
        st = self.ctl.settings
        d = st.to_dict()
        for key, (f, scale) in self.fields.items():
            f.set_value(d[key] * scale)
        e = self.ctx.store.project.ecu
        for key, f in self.ecu_fields.items():
            f.set_value(getattr(e, key))
        for w, v in ((self.enabled, st.enabled),):
            w.blockSignals(True)
            w.setChecked(v)
            w.blockSignals(False)
        self.compute.set_value(st.compute)
        self.rf.blockSignals(True)
        self.rf.setCurrentIndex(max(0, self.rf.findData(st.gnss_cable)))
        self.rf.blockSignals(False)

    def _later(self, fn, *args, **kw):
        """Run an edit after the signal that asked for it returns (the tables are rebuilt by the edit)."""
        QTimer.singleShot(0, lambda: fn(*args, **kw))

    # ------------------------------------------------------------------ entry points
    def _entries_now(self) -> list | None:
        """The entry points as stored (the defaults from the body until edited); None until the first routing has
        placed the defaults (they are worked out on the thread, never here)."""
        res = self.ctl.result
        st = self.ctl.settings
        if st.entries:
            return [dict(e) for e in st.entries]
        if res is not None and res.routed and not res.settings.entries:
            return [{'uid': e['uid'], 'name': e['name'], 'kind': e['kind'], 'x': float(e['asked'][0]),
                     'y': float(e['asked'][1]), 'z': float(e['asked'][2]), 'enabled': e['enabled'],
                     'capacity': e['capacity']} for e in res.entries]
        return None

    def _store_entries(self, ents, text) -> bool:
        st = self.ctl.settings
        st.entries = ents
        live = {e['uid'] for e in ents}
        st.assign = {u: e for u, e in st.assign.items() if e in live}
        return self.ctl.store(st, text)

    def _fill_entries(self):
        res = self.ctl.result
        ents = self._entries_now()
        ready = ents is not None
        ents = ents or []
        uses = {e['uid']: len(e['cables']) for e in res.entries} if res is not None else {}
        snapped = {e['uid']: e['pos'] for e in res.entries} if res is not None else {}
        t = self.entries
        sel = t.currentRow()
        busy, self._busy = self._busy, True
        try:
            _hide_widgets(t)
            t.setRowCount(len(ents))
            p = self.ctx.store.project
            for i, e in enumerate(ents):
                d = p.to_display([e['x'], e['y'], e['z']])
                it = QTableWidgetItem(e['name'])
                it.setData(KEY, e['uid'])
                n = uses.get(e['uid'], 0)
                tip = f'{n} cable{"s" * (n != 1)} through it'
                if e['uid'] in snapped:
                    q = p.to_display(snapped[e['uid']])
                    tip += f'; on the body at {q[0]:.3f}, {q[1]:.3f}, {q[2]:.3f}'
                it.setToolTip(tip)
                t.setItem(i, E_NAME, it)
                cb = _Combo()
                for k in HN.ENTRY_KINDS:
                    cb.addItem(HN.ENTRY_LABEL[k], k)
                cb.setCurrentIndex(max(0, cb.findData(e['kind'])))
                cb.currentIndexChanged.connect(lambda _, i=i, cb=cb: self._later(self.set_entry, i, 'kind',
                                                                                  cb.currentData()))
                t.setCellWidget(i, E_KIND, cb)
                for c, v in ((E_X, d[0]), (E_Y, d[1]), (E_Z, d[2])):
                    xi = QTableWidgetItem(f'{v:.3f}')
                    xi.setFont(mono(12))
                    xi.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                    t.setItem(i, c, xi)
                on = QTableWidgetItem('')
                on.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsSelectable)
                on.setCheckState(Qt.CheckState.Checked if e['enabled'] else Qt.CheckState.Unchecked)
                on.setToolTip('Cables may go in here')
                t.setItem(i, E_ON, on)
                ci = QTableWidgetItem(str(e['capacity']) if e['capacity'] else '')
                ci.setFont(mono(12))
                ci.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                ci.setToolTip('The most cables it takes (empty: no limit)')
                t.setItem(i, E_CAP, ci)
            if 0 <= sel < len(ents):
                t.selectRow(sel)
            t.setFixedHeight(ROW_H * (max(len(ents), 3) + 1) + 4)
        finally:
            self._busy = busy
        self.b_add.setEnabled(ready and len(ents) < HN.MAX_ENTRIES)
        self.b_remove.setEnabled(ready and bool(ents))
        self.b_reset.setEnabled(bool(self.ctl.settings.entries))

    def _entry_edited(self, it):
        if self._busy:
            return
        c = it.column()
        if c == E_ON:
            self._later(self.set_entry, it.row(), 'enabled', it.checkState() == Qt.CheckState.Checked)
        elif c == E_NAME:
            self._later(self.set_entry, it.row(), 'name', it.text())
        elif c in (E_X, E_Y, E_Z):
            self._later(self.set_entry, it.row(), 'xyz'[c - E_X], it.text())
        elif c == E_CAP:
            self._later(self.set_entry, it.row(), 'capacity', it.text())

    def set_entry(self, i, what, value) -> bool:
        """Change entry point i: 'name', 'kind', 'enabled', 'capacity' or a coordinate 'x'/'y'/'z' (typed, in the
        project's frame)."""
        ents = self._entries_now()
        if ents is None or not 0 <= i < len(ents):
            return False
        e = ents[i]
        name = e['name']
        if what == 'name':
            text = str(value).strip()
            if not text or text == name:
                self._fill_entries()
                return False
            e['name'], label_ = text, f'renamed {name} to {text}'
        elif what == 'kind':
            e['kind'], label_ = value, f'{name} is a {HN.ENTRY_LABEL.get(value, "entry point").lower()}'
        elif what == 'enabled':
            e['enabled'], label_ = bool(value), f'{name} {"on" if value else "off"}'
        elif what == 'capacity':
            try:
                v = _parse(value)
            except ValueError:
                self._fill_entries()
                return False
            e['capacity'] = 0 if v is None else int(round(max(v, 0)))
            label_ = f'{name} takes {e["capacity"]} cables' if e['capacity'] else f'{name} takes any number of cables'
        else:
            try:
                v = _parse(value)
            except ValueError:
                v = None
            if v is None:
                self._fill_entries()
                return False
            p = self.ctx.store.project
            d = p.to_display([e['x'], e['y'], e['z']])
            d['xyz'.index(what)] = v
            w = p.from_display(d)
            e['x'], e['y'], e['z'] = (float(q) for q in w)
            label_ = f'{name} moved to {d[0]:.3f}, {d[1]:.3f}, {d[2]:.3f}'
        if not self._store_entries(ents, label_):
            self._fill_entries()
            return False
        return True

    def add_entry(self) -> bool:
        from ...core.rig import roof
        ents = self._entries_now()
        if ents is None:
            return False
        if len(ents) >= HN.MAX_ENTRIES:
            self.ctx.msg(f'At most {HN.MAX_ENTRIES} entry points', 3000)
            return False
        xf, xr, hw, top = roof(self.ctx.store.project.vehicle.mesh)
        names = {e['name'] for e in ents}
        name = next(f'Entry point {k}' for k in range(1, len(ents) + 2) if f'Entry point {k}' not in names)
        uids = {e['uid'] for e in ents}
        uid = next(f'e{k}' for k in range(1, len(ents) + 2) if f'e{k}' not in uids)
        ents.append({'uid': uid, 'name': name, 'kind': 'other', 'x': (xf + xr) / 2, 'y': 0.0, 'z': top,
                     'enabled': True, 'capacity': 0})
        if self._store_entries(ents, f'added {name}'):
            self.entries.selectRow(len(ents) - 1)
            return True
        return False

    def remove_entry(self) -> bool:
        ents = self._entries_now()
        if not ents:
            return False
        i = self.entries.currentRow()
        i = i if 0 <= i < len(ents) else len(ents) - 1
        name = ents.pop(i)['name']
        return self._store_entries(ents, f'removed {name}')

    def reset_entries(self) -> bool:
        if not self.ctl.settings.entries:
            return False
        return self._store_entries([], 'entry points from the body')

    def _entry_selected(self):
        rows = self.entries.selectionModel().selectedRows()
        it = self.entries.item(rows[0].row(), E_NAME) if rows else None
        self.view.set_selected(self.selected, it.data(KEY) if it is not None else '')

    def _select_entry(self, uid):
        for i in range(self.entries.rowCount()):
            it = self.entries.item(i, E_NAME)
            if it is not None and it.data(KEY) == uid:
                self.entries.selectRow(i)
                return

    # ------------------------------------------------------------------ runs
    def _cell(self, r, c, text, num=False, color=None, tip=None, bold=False, swatch=None, key=None, editable=False):
        tb = self.table
        w = tb.cellWidget(r, c)
        if w is not None:                              # a picker there before (the row was another run's)
            w.hide()
            tb.removeCellWidget(r, c)
        it = QTableWidgetItem(text)
        flags = Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
        if editable:
            flags |= Qt.ItemFlag.ItemIsEditable
        it.setFlags(flags)
        if num:
            it.setFont(mono(12, QFont.Weight.DemiBold if bold else None))
            it.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        elif bold:
            it.setFont(sans(12, QFont.Weight.DemiBold))
        if color is not None:
            it.setForeground(color)
        if swatch is not None:
            it.setData(Qt.ItemDataRole.DecorationRole, swatch)
        if tip:
            it.setToolTip(tip)
        if key is not None:
            it.setData(KEY, key)
        tb.setItem(r, c, it)
        return it

    def _fill_runs(self, res):
        tb = self.table
        _hide_widgets(tb)
        tb.clearContents()
        tb.setRowCount(len(res.runs))
        muted = THEME.color('muted')
        p = self.ctx.store.project
        st = res.settings
        for i, r in enumerate(res.runs):
            what = {'data': 'data', 'rf': 'RF', 'power': 'power'}[r['what']]
            self._cell(i, C_RUN, f'{r["tag"]} · {what}', swatch=run_qcolor(r), key=r['id'],
                       tip=f'{r["tag"]} ({r["model"]}) → {r["to"]}')
            if r['what'] == 'power':
                self._cell(i, C_CABLE, r['cable_label'], tip=HN.CABLES['power'].basis)
            else:
                cb = _Combo()
                opts = HN.RF_CABLES if r['what'] == 'rf' else HN.DATA_CABLES
                for k in opts:
                    c = HN.CABLES[k]
                    cb.addItem(c.label + (f' · {c.limit_m:g} m' if c.limit_m else ''), k)
                cb.setCurrentIndex(max(0, cb.findData(r['cable'])))
                own = st.cables.get(r['uid'])
                cb.setToolTip(f'{HN.CABLES[r["cable"]].basis}. ' +
                              ('Picked for this run.' if own else 'From the device\'s interface.'))
                cb.currentIndexChanged.connect(lambda _, u=r['uid'], cb=cb, t=r['tag']:
                                               self._later(self.set_cable, u, cb.currentData(), t))
                tb.setCellWidget(i, C_CABLE, cb)
            self._cell(i, C_TO, r['to'])
            if r['what'] == 'power' or not res.routed or (r['mode'] == 'in_in' and not st.assign.get(r['uid'])):
                txt = r['entry_name'] or ('inside' if r['device_inside'] else '—')
                self._cell(i, C_ENTRY, txt, color=muted,
                           tip='Runs with its data cable' if r['what'] == 'power' else
                           'Starts inside the body: no entry point' if r['device_inside'] else 'No entry point needed')
            else:
                cb = _Combo()
                cb.addItem('Shortest' + (f' ({r["entry_name"]})' if r['entry_name'] and not st.assign.get(r['uid'])
                                         else ''), '')
                for e in res.entries:
                    if e['enabled']:
                        cb.addItem(e['name'], e['uid'])
                cb.setCurrentIndex(max(0, cb.findData(st.assign.get(r['uid'], ''))))
                cb.setToolTip('The entry point this cable goes in through (Shortest: the one that makes the run '
                              'shortest)')
                cb.currentIndexChanged.connect(lambda _, u=r['uid'], cb=cb, t=r['tag']:
                                               self._later(self.set_assign, u, cb.currentData(), t))
                tb.setCellWidget(i, C_ENTRY, cb)
            parts = r['parts']
            self._cell(i, C_LEN, f'{r["length_m"]:.2f}', True, bold=True,
                       tip=f'along the body {parts["drop"] + parts["outside"]:.2f} m · inside {parts["inside"]:.2f} m '
                           f'· service loop {parts["service_loop"]:.2f} m · extra {parts["extra"]:.2f} m')
            self._cell(i, C_MAN, f'{r["manhattan_m"]:.2f}', True, muted,
                       tip='The Manhattan estimate (× the inside slack + the service loop) the app used before')
            if r['what'] == 'power':
                self._cell(i, C_LIMIT, f'{r["awg"]} AWG', True, muted, tip='The power budget\'s gauge for this device')
                if r['drop_pct'] is None:
                    self._cell(i, C_USE, 'collapses', True, QColor(STATUS['serious']))
                else:
                    self._cell(i, C_USE, f'{r["drop_pct"]:.1f} %', True, QColor(STATUS[r['severity']]),
                               tip=f'{SEV_WORD[r["severity"]]}: voltage drop at the steady current')
            else:
                if r['what'] == 'rf':
                    lim_text = f'{st.gnss_lna_db - st.gnss_min_db:g} dB'
                    lim_tip = (f'The RF budget: the {st.gnss_lna_db:g} dB LNA gain less the {st.gnss_min_db:g} dB the '
                               f'receiver needs; {r["loss_db"]:.1f} dB lost, {r["margin_db"]:.1f} dB left')
                elif r['kind'] == 'camera' and r['fitted_m']:
                    lim_text = f'{r["fitted_m"]:g} m cable'
                    lim_tip = (f'The fitted cable; the module offers {", ".join(f"{o:g}" for o in r["options"])} m'
                               + (f' and the link allows {r["limit_m"]:g} m' if r['limit_m'] else ''))
                else:
                    lim_text = f'{r["limit_m"]:g} m' if r['limit_m'] else '—'
                    lim_tip = r['limit_basis']
                self._cell(i, C_LIMIT, lim_text, True, muted, tip=lim_tip)
                use = r.get('use')
                sev = _use_sev(r, st)
                self._cell(i, C_USE, '—' if use is None else f'{use * 100:.0f} %', True,
                           None if use is None else QColor(STATUS[sev]),
                           tip='' if use is None else f'{SEV_WORD[sev]}: {use * 100:.0f} % of {r["use_of"]}')
            if r['what'] != 'power':
                ex = r['parts']['extra']
                self._cell(i, C_EXTRA, f'{ex:.2f}' if ex else '', True, key=r['uid'], editable=True,
                           tip='Length added to this run (a camera\'s route extra, a sensor\'s extra). Double-click to '
                               'type it; clear it for none')
            else:
                self._cell(i, C_EXTRA, '', True)
            sev = r['severity']
            it = self._cell(i, C_STATUS, SEV_WORD.get(sev, sev), color=QColor(STATUS[sev]) if sev in STATUS else None,
                            tip='\n'.join(m for s, k, m in r['problems']) or 'Within every limit')
            if sev in STATUS and sev != 'good':
                it.setData(Qt.ItemDataRole.DecorationRole, icons.pixmap(sev, STATUS[sev], 12))
            if r['id'] == self.selected:
                tb.selectRow(i)

    def _run_edited(self, it):
        if self._busy or it.column() != C_EXTRA:
            return
        self._later(self.set_extra, it.data(KEY), it.text())

    def set_extra(self, uid, text) -> bool:
        """A run's extra length: a camera's route_extra_m (its own field), a sensor's in the harness settings."""
        try:
            v = _parse(text)
        except ValueError:
            self._on_result()
            return False
        v = 0.0 if v is None else min(max(v, HN.EXTRA_M[0]), HN.EXTRA_M[1])
        p = self.ctx.store.project
        cam = p.variant.cam(uid)
        if cam is not None:
            if abs(cam.route_extra_m - v) < 1e-9:
                return False
            self.ctx.store.set_cam(uid, text=f'{cam.tag} route extra {v:g} m', route_extra_m=v)
            return True
        s = p.variant.sensor(uid)
        if s is None:
            return False
        st = self.ctl.settings
        if v > 0:
            st.extra[uid] = v
        else:
            st.extra.pop(uid, None)
        return self.ctl.store(st, f'{s.tag} extra {v:g} m')

    def set_cable(self, uid, cid, tag='') -> bool:
        st = self.ctl.settings
        res = self.ctl.result
        r = res.data_run(uid) if res is not None else None
        if cid not in HN.CABLES:
            return False
        # the type the device's interface gives is the default: picking it again clears the choice
        dev_default = None
        if r is not None:
            st0 = HN.HarnessSettings.from_dict(dict(st.to_dict(), cables={}))
            devs, _ = HN._devices(self.ctx.store.project, self.ctx.store.project.variant, st0)
            dev_default = next((d['cable'] for d in devs if d['uid'] == uid), None)
        if cid == dev_default:
            st.cables.pop(uid, None)
        else:
            st.cables[uid] = cid
        return self.ctl.store(st, f'{tag or "run"} on {HN.CABLES[cid].label}')

    def set_assign(self, uid, entry_uid, tag='') -> bool:
        st = self.ctl.settings
        if entry_uid:
            st.assign[uid] = entry_uid
        else:
            st.assign.pop(uid, None)
        res = self.ctl.result
        ent = res.entry(entry_uid) if res is not None and entry_uid else None
        name = ent['name'] if ent else 'the shortest'
        return self.ctl.store(st, f'{tag or "run"} through {name}')

    def _row_selected(self):
        rows = self.table.selectionModel().selectedRows()
        it = self.table.item(rows[0].row(), 0) if rows else None
        rid = it.data(KEY) if it is not None else ''
        if rid != self.selected:
            self.selected = rid or ''
            self.view.set_selected(self.selected)

    def select_run(self, rid):
        """Select a run (from the view): its row in the table and its route on the body."""
        self.selected = rid or ''
        self.view.set_selected(self.selected)
        if not rid:
            self.table.blockSignals(True)
            self.table.clearSelection()
            self.table.blockSignals(False)
            return
        for i in range(self.table.rowCount()):
            it = self.table.item(i, 0)
            if it is not None and it.data(KEY) == rid:
                self.table.blockSignals(True)
                self.table.selectRow(i)
                self.table.blockSignals(False)
                self.table.scrollToItem(it)
                return

    # ------------------------------------------------------------------ state
    def _set_mode(self, mode):
        self.view.set_mode(mode)

    def _on_theme(self, *_):
        if self.isVisible():
            self._on_result()

    def _on_store(self, *_):
        if self.isVisible():
            self._sync_fields()

    def showEvent(self, e):
        super().showEvent(e)
        self.activate()

    def hideEvent(self, e):
        super().hideEvent(e)
        self.ctl.set_active(False)

    def activate(self):
        self._sync_fields()
        self.ctl.set_active(True)
        self._on_result()

    def refresh(self):
        """Route again now (the result shows when the thread is done)."""
        self.ctl.run()

    def _on_result(self):
        ctl = self.ctl
        res = ctl.result
        msg = ''
        if ctl.busy and res is None:
            msg = 'Routing the cables…'
        elif ctl.stale:
            msg = 'Routing again…' if ctl.wanted() else 'Out of date'
        self.view.set_result(res, ctl.display, msg if res is not None else (msg or ctl.status or 'Routing the cables…'))
        self.status.setText(ctl.status)
        self.b_export.setEnabled(res is not None)
        self._busy = True
        try:
            self._fill_entries()
            if res is None:
                self.table.setRowCount(0)
                for t in self.tiles.values():
                    t.set('—', '', '')
            else:
                self._tiles(res)
                self._fill_runs(res)
        finally:
            self._busy = False
        for w in self.findings:
            self.find_lay.removeWidget(w)
            w.hide()                    # until the deferred delete runs it would still paint over the new rows
            w.deleteLater()
        self.findings = []
        for f in (res.findings if res is not None else []):
            w = StatusLine('good' if f['severity'] == 'note' else f['severity'], f['message'])
            if f['severity'] == 'note':
                w.icon.setPixmap(icons.pixmap('help', THEME.color('muted'), 12))
            self.find_lay.insertWidget(self.find_lay.count() - 1, w)
            self.findings.append(w)

    def _tiles(self, res):
        t = res.totals
        st = res.settings
        n_data = sum(r['what'] != 'power' for r in res.runs)
        self.tiles['cables'].set(f'{t["runs"]}', '', f'{n_data} data and RF · {t["runs"] - n_data} power')
        man = t['manhattan_m']
        diff = (t['data_length_m'] / man - 1) * 100 if man > 0 else 0.0
        self.tiles['total'].set(f'{t["length_m"]:.1f}', 'm', f'data and RF {t["data_length_m"]:.1f} m · {diff:+.0f} % '
                                f'on the estimate' if res.routed else 'the Manhattan estimate (no body)')
        lg = t['longest']
        self.tiles['longest'].set('—' if lg is None else f'{lg["length_m"]:.2f}', '' if lg is None else 'm',
                                  '' if lg is None else f'{lg["tag"]} · {lg["cable"]}')
        tg = t['tightest']
        if tg is None:
            self.tiles['tightest'].set('—', '', 'no run has a limit')
        else:
            r = res.run(tg['id'])
            self.tiles['tightest'].set(f'{tg["use"] * 100:.0f}', '%', f'{r["tag"]} · {r["length_m"]:.2f} m of '
                                       f'{tg["of"]}', _use_sev(r, st))
        on = [e for e in res.entries if e['enabled']]
        used = [e for e in res.entries if e['cables']]
        cap = 'capacity' in {f['kind'] for f in res.findings}
        self.tiles['entries'].set(f'{len(used)}', f'of {len(on)}', ', '.join(e['name'] for e in used) or
                                  ('no cable goes in' if res.routed else 'no body'), 'warning' if cap else None)

    # ------------------------------------------------------------------ export
    def export(self, folder=None):
        res = self.ctl.result
        if res is None:
            return []
        if folder is None:
            folder = QFileDialog.getExistingDirectory(self, 'Export the cable harness into')
            if not folder:
                return []
        from ...core import provenance as PV          # where the JSON came from
        paths = PV.stamp_paths(HN.export(res, self.ctx.store.project, folder), self.ctx.store.project, folder)
        self.ctx.msg(f'Exported the cable harness to {folder}', 5000)
        return paths
