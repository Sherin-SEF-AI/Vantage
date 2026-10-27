"""Analysis ▸ GNSS and IMU: the sky each GNSS antenna sees (a polar plot per antenna, what blocks it, the expected
satellites, near-field multipath), the dual-antenna heading baseline, the lever arms from the IMU to every antenna and
sensor in both frames, and the pose error budget: how well each sensor's world pose will be known. Computed again after
every edit while the page shows (each antenna's sky is cached, so most edits cost nothing); a slow first run goes to a
worker thread."""
from __future__ import annotations

import math
import threading
import traceback

from PyQt6.QtCore import QObject, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import (QAbstractItemView, QComboBox, QFileDialog, QGridLayout, QHBoxLayout, QHeaderView,
                             QScrollArea, QSplitter, QStackedWidget, QTableWidget, QTableWidgetItem, QVBoxLayout,
                             QWidget)

from ...core import gnss as G
from .. import icons
from ..gnssview import SkyPlot, blocker_color
from ..theme import THEME, STATUS, mono, sans
from ..widgets import ROW_H, ScrubSpinBox, StatusLine, ToggleSwitch, button, hline, label
from .recording_ws import Tile

QUICK_S = 0.15                       # a study expected to take longer runs on a worker thread
EMPTY = 'Add a GNSS antenna or an IMU: Vehicle ▸ Add GNSS antenna / Add IMU.'
SEV_TONE = {'critical': 'critical', 'serious': 'serious', 'warning': 'warning', 'good': 'good'}

# (group, key, label, unit, decimals, step, tip)
FIELDS = (
    ('Sky view', 'mask_deg', 'Mask', '°', 1, 1.0,
     'One elevation mask for every antenna (typing a value turns it on); off, each antenna uses its model\'s mask'),
    ('Sky view', 'az_step_deg', 'Azimuth step', '°', 1, 0.5, 'Width of a sky cell; finer is slower'),
    ('Sky view', 'el_step_deg', 'Elevation step', '°', 2, 0.25, 'Height of a sky cell; finer is slower'),
    ('Sky view', 'n_sats', 'Satellites', '', 0, 1.0,
     'Satellites above the mask in open sky: about 32 for a multi-constellation receiver (GPS, Galileo, GLONASS, '
     'BeiDou) above 10°, an assumption. Expected satellites = this × the visible share of the sky'),
    ('Sky view', 'near_m', 'Near field', 'm', 2, 0.05,
     'A structure this close to the antenna that rises above its horizon is flagged for multipath'),
    ('Error budget', 'pos_sigma_cm', 'Position σ', 'cm', 1, 0.5,
     'The INS\'s horizontal position error (1σ, with RTK), applied to each horizontal axis; 0 = the INS model\'s'),
    ('Error budget', 'vertical_ratio', 'Vertical ×', '×', 1, 0.1, 'Vertical position σ = horizontal σ × this'),
    ('Error budget', 'attitude_sigma_deg', 'Roll, pitch σ', '°', 3, 0.005,
     'The INS\'s roll and pitch error (1σ); 0 = the INS model\'s'),
    ('Error budget', 'heading_sigma_deg', 'Heading σ', '°', 3, 0.01,
     'The INS\'s heading error (1σ); 0 = from the dual-antenna baseline (the model\'s figure scaled by its rated '
     'baseline over the horizontal baseline length)'),
    ('Error budget', 'lever_arm_sigma_mm', 'Lever arm σ', 'mm', 1, 1.0,
     'How well each lever arm is measured, per axis (the antenna\'s in the INS and each sensor\'s own)'),
    ('Error budget', 'label_range_m', 'Label range', 'm', 0, 5.0,
     'The budget also gives the horizontal error of a point this far from each sensor (the heading error swings it)'),
)
ANT_COLS = (('Antenna', 'The GNSS antenna (its phase centre)'),
            ('Open', 'Share of the sky above its elevation mask that nothing of the vehicle blocks (solid-angle '
                     'weighted)'),
            ('Sats', 'Expected satellites: satellites above the mask × the open share (an even sky assumed)'),
            ('Worst sector', 'The blocked azimuth run (counter-clockwise from forward) that hides most sky, and how '
                             'high it reaches'),
            ('Blockers', 'What blocks its sky above the mask, with its share'),
            ('Near field', 'Structures within the near-field distance rising above the antenna\'s horizon '
                           '(multipath)'),
            ('Ground plane', 'Whether its base sits on a surface reaching 10 cm round it'))
SEC_COLS = (('From', 'Start of the blocked run, degrees counter-clockwise from forward'),
            ('To', 'End of the run'), ('Up to', 'Highest blocked elevation in the run'),
            ('Sky', 'Share of the sky above the mask the run hides'), ('Blockers', 'What blocks it, largest first'))
LEVER_COLS = (('Target', 'The sensor, antenna or the vehicle frame origin (the nuScenes ego frame)'),
              ('Kind', ''),
              ('IMU x', 'Lever arm in the IMU\'s own frame (x forward, y left, z up as printed on it), mm'),
              ('IMU y', ''), ('IMU z', ''),
              ('Veh x', 'Lever arm in the vehicle frame (ISO 8855: x forward, y left, z up), mm'),
              ('Veh y', ''), ('Veh z', ''),
              ('Length', 'Length of the lever arm, mm'))
BUDGET_COLS = (('Target', 'The sensor or the vehicle frame origin'),
               ('Lever', 'Distance from the IMU, m'),
               ('σx', 'Position error along the vehicle\'s x, mm (1σ)'), ('σy', 'Along y, mm'), ('σz', 'Along z, mm'),
               ('Horiz.', 'Horizontal position error, mm (1σ)'),
               ('3D', 'Total position error, mm (1σ)'),
               ('Far', 'Horizontal error of a point at the label range from the sensor, mm (the heading error '
                       'swings it)'))
KIND_LABEL = {'origin': 'Origin', 'gnss': 'Antenna', 'camera': 'Camera', 'lidar': 'Lidar', 'radar': 'Radar',
              'imu': 'IMU'}


def _table(cols, select=False):
    t = QTableWidget(0, len(cols))
    for c, (head, tip) in enumerate(cols):
        it = QTableWidgetItem(head)
        if tip:
            it.setToolTip(tip)
        t.setHorizontalHeaderItem(c, it)
    t.verticalHeader().setVisible(False)
    t.verticalHeader().setDefaultSectionSize(ROW_H)
    t.horizontalHeader().setFixedHeight(ROW_H)
    t.setShowGrid(False)
    t.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    if select:
        t.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        t.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    else:
        t.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
    hh = t.horizontalHeader()
    for c in range(len(cols)):
        hh.setSectionResizeMode(c, QHeaderView.ResizeMode.ResizeToContents)
    return t


def _fit(t, rows, most=6):
    t.setFixedHeight(ROW_H * (min(max(rows, 1), most) + 1) + 6)


def _cell(t, r, c, text, num=False, color=None, tip=None, bold=False, swatch=None):
    it = QTableWidgetItem(text)
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
    t.setItem(r, c, it)
    return it


def _mm(v, nd=0):
    return '—' if v is None or not math.isfinite(v) else f'{v:,.{nd}f}'


def _geo_key(p):
    """The vehicle body as the engine keys it (not the rig: a rig change reuses the simplified body)."""
    m = p.vehicle.mesh
    return (id(m), getattr(m, 'n_tris', 0), p.vehicle.occluder_target,
            tuple(x.role for x in m.parts) if m is not None else ())


class _Sig(QObject):
    done = pyqtSignal(int, object)
    failed = pyqtSignal(int, str)


class GnssPage(QWidget):
    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        self.result: G.GnssResult | None = None
        self.sel_uid = None                 # the antenna picked on the page
        self.busy = False
        self._dirty = True
        self._pending = False
        self._gen = 0
        self._estimate = None               # seconds an uncached run took last time
        self._geo = None                    # the vehicle body that run was made for
        self._eng = None
        self._syncing = False
        self.sig = _Sig()
        self.sig.done.connect(self._done)
        self.sig.failed.connect(self._failed)
        self.setObjectName('base')
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        split = QSplitter(Qt.Orientation.Horizontal)
        split.setHandleWidth(2)
        split.addWidget(self._left())
        self.stack = QStackedWidget()
        self.stack.addWidget(self._content())
        self.stack.addWidget(self._empty())
        split.addWidget(self.stack)
        split.setSizes([310, 1290])
        lay.addWidget(split)
        ctx.store.changed.connect(self._on_change)
        ctx.store.project_replaced.connect(self._on_replaced)
        ctx.store.selection_changed.connect(self._on_selection)
        THEME.changed.connect(self._on_theme)
        self._sync_fields()

    # ------------------------------------------------------------------ layout
    def _left(self):
        w = QWidget()
        w.setObjectName('panel')
        ll = QVBoxLayout(w)
        ll.setContentsMargins(8, 6, 8, 8)
        ll.setSpacing(4)
        top = QHBoxLayout()
        top.addWidget(label('GNSS and IMU'))
        top.addStretch(1)
        top.addWidget(label('sky, heading, lever arms', muted=True, small=True))
        ll.addLayout(top)
        ll.addWidget(hline())
        ll.addWidget(label('What sky each GNSS antenna sees past the body, the rig and the other sensors, the '
                           'dual-antenna heading baseline, the lever arms the INS needs, and how well each sensor\'s '
                           'world pose will be known. Azimuths counter-clockwise from forward.', muted=True, small=True,
                           wrap=True))
        self.fields = {}
        self.combos = {}
        group = grid = None
        for g, key, text, unit, dec, step, tip in FIELDS:
            if g != group:
                if grid is not None:
                    ll.addLayout(grid)
                    if group == 'Sky view':
                        self._heading_group(ll)
                group = g
                ll.addSpacing(6)
                ll.addWidget(label(g, bold=True))
                if g == 'Sky view':
                    self.mask_on = ToggleSwitch('One mask for every antenna')
                    self.mask_on.setToolTip('Off: each antenna uses its model\'s elevation mask')
                    self.mask_on.clicked.connect(
                        lambda on: self._set('one mask for every antenna' if on else 'each antenna\'s own mask',
                                             mask_override=on))
                    ll.addWidget(self.mask_on)
                grid = QGridLayout()
                grid.setVerticalSpacing(2)
                grid.setHorizontalSpacing(6)
                grid.setColumnStretch(2, 1)
            f = ScrubSpinBox('', 'plain', dec, unit, step=step, width=110)
            f.setToolTip(tip)
            f.committed.connect(lambda v, k=key, t=text, u=unit: self._commit(k, v, f'{t} {v:g}{" " + u if u else ""}'))
            a = label(text, ink2=True)
            a.setFixedWidth(104)
            a.setToolTip(tip)
            i = grid.rowCount()
            grid.addWidget(a, i, 0)
            grid.addWidget(f, i, 1)
            self.fields[key] = f
        ll.addLayout(grid)
        self.inputs = label('', muted=True, small=True, wrap=True)
        ll.addWidget(self.inputs)
        ll.addStretch(1)
        self.status = label('', muted=True, small=True, wrap=True)
        ll.addWidget(self.status)
        bl = QHBoxLayout()
        b_def = self.b_defaults = button('Defaults', 'ghost', None, 'Back to the default GNSS and IMU settings')
        b_def.clicked.connect(self._defaults)
        bl.addWidget(b_def)
        bl.addStretch(1)
        self.b_export = button('Export…', None, 'export',
                               'The sky views, baseline, lever arms, error budget and findings as JSON, the INS lever '
                               'arms as text and the sky plots as PNG')
        self.b_export.clicked.connect(lambda: self.export())
        bl.addWidget(self.b_export)
        ll.addLayout(bl)
        sc = QScrollArea()
        sc.setWidgetResizable(True)
        sc.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        sc.setWidget(w)
        sc.setMinimumWidth(290)
        return sc

    def _heading_group(self, ll):
        ll.addSpacing(6)
        ll.addWidget(label('Heading and lever arms', bold=True))
        grid = QGridLayout()
        grid.setVerticalSpacing(2)
        grid.setHorizontalSpacing(6)
        grid.setColumnStretch(2, 1)
        for i, (key, text, tip) in enumerate((
                ('imu', 'Reference IMU', 'The IMU the lever arms and the error budget are taken from (Auto: the '
                                         'first GNSS/INS by tag, else the first IMU)'),
                ('primary', 'Primary antenna', 'The heading baseline runs from the primary to the secondary antenna '
                                               '(Auto: the first two by tag)'),
                ('secondary', 'Secondary', 'The second antenna of the heading baseline'))):
            cb = QComboBox()
            cb.setFixedWidth(110)
            cb.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
            cb.setToolTip(tip)
            cb.activated.connect(lambda _=0, k=key, c=cb, t=text: self._set(f'{t.lower()} {c.currentData() or "auto"}',
                                                                             **{k: c.currentData() or ''}))
            a = label(text, ink2=True)
            a.setFixedWidth(104)
            a.setToolTip(tip)
            grid.addWidget(a, i, 0)
            grid.addWidget(cb, i, 1)
            self.combos[key] = cb
        ll.addLayout(grid)

    def _content(self):
        split = QSplitter(Qt.Orientation.Horizontal)
        split.setHandleWidth(2)
        # ---------------- centre: tiles, sky plots, antennas, the picked antenna's blocked sectors
        cw = QWidget()
        cw.setObjectName('panel')
        cl = QVBoxLayout(cw)
        cl.setContentsMargins(10, 8, 10, 8)
        cl.setSpacing(6)
        tiles = QHBoxLayout()
        tiles.setSpacing(8)
        self.tiles = {}
        for key, caption, tip in (
                ('sky', 'Worst sky', 'The open share of the sky above the mask of the antenna that sees '
                                              'least'),
                ('heading', 'Baseline', 'Horizontal length of the dual-antenna baseline and the heading σ '
                                                'it gives'),
                ('imu', 'IMU to origin', 'Lever arm from the reference IMU to the vehicle frame origin (the nuScenes '
                                         'ego frame)'),
                ('pose', 'Sensor position σ', 'The largest horizontal position error (1σ) of any camera, lidar or '
                                              'radar')):
            t = self.tiles[key] = Tile(caption, tip)
            tiles.addWidget(t, 1)
        cl.addLayout(tiles)
        self.plots_host = QWidget()
        self.plots_lay = QHBoxLayout(self.plots_host)
        self.plots_lay.setContentsMargins(0, 0, 0, 0)
        self.plots_lay.setSpacing(8)
        self.plots: list[SkyPlot] = []
        sc = QScrollArea()
        sc.setWidgetResizable(True)
        sc.setFrameShape(QScrollArea.Shape.NoFrame)
        sc.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        sc.setWidget(self.plots_host)
        sc.setMinimumHeight(340)
        cl.addWidget(sc, 3)
        cl.addWidget(label('Antennas', bold=True))
        self.ants = _table(ANT_COLS, select=True)
        hh = self.ants.horizontalHeader()
        hh.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        hh.setSectionResizeMode(5, QHeaderView.ResizeMode.Interactive)
        hh.resizeSection(5, 130)
        self.ants.itemSelectionChanged.connect(self._table_pick)
        cl.addWidget(self.ants)
        self.sec_head = label('Blocked sectors', bold=True)
        cl.addWidget(self.sec_head)
        self.secs = _table(SEC_COLS)
        self.secs.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        cl.addWidget(self.secs, 1)
        split.addWidget(cw)
        # ---------------- right: baseline, lever arms, the error budget, findings
        right = QSplitter(Qt.Orientation.Vertical)
        right.setHandleWidth(2)
        rw = QWidget()
        rw.setObjectName('panel')
        rl = QVBoxLayout(rw)
        rl.setContentsMargins(10, 8, 10, 8)
        rl.setSpacing(4)
        rl.addWidget(label('Heading baseline', bold=True))
        self.base_text = label('', wrap=True)
        self.base_text.setTextFormat(Qt.TextFormat.PlainText)
        rl.addWidget(self.base_text)
        self.lever_head = label('Lever arms', bold=True)
        rl.addWidget(self.lever_head)
        self.levers = _table(LEVER_COLS)
        self.levers.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        rl.addWidget(self.levers, 1)
        rl.addWidget(label('Pose error budget', bold=True))
        self.budget_note = label('', muted=True, small=True, wrap=True)
        rl.addWidget(self.budget_note)
        self.budget = _table(BUDGET_COLS)
        self.budget.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        rl.addWidget(self.budget, 1)
        right.addWidget(rw)
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
        fsc = QScrollArea()
        fsc.setWidgetResizable(True)
        fsc.setWidget(self.find_host)
        fl.addWidget(fsc, 1)
        self.findings: list[StatusLine] = []
        right.addWidget(fw)
        right.setSizes([620, 260])
        split.addWidget(right)
        split.setSizes([700, 590])
        return split

    def _empty(self):
        w = QWidget()
        w.setObjectName('panel')
        l = QVBoxLayout(w)
        l.addStretch(1)
        self.empty_text = label(EMPTY, muted=True, wrap=True)
        self.empty_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        l.addWidget(self.empty_text)
        row = QHBoxLayout()
        row.addStretch(1)
        self.b_add_gnss = button('Add GNSS antenna', None, 'sensor', 'A GNSS antenna on the roof')
        self.b_add_gnss.clicked.connect(lambda: self._add('gnss'))
        self.b_add_imu = button('Add IMU', None, 'sensor', 'A dual-antenna GNSS/INS above the rear axle')
        self.b_add_imu.clicked.connect(lambda: self._add('imu'))
        row.addWidget(self.b_add_gnss)
        row.addWidget(self.b_add_imu)
        row.addStretch(1)
        l.addLayout(row)
        l.addStretch(2)
        return w

    def _add(self, kind):
        from ..sensorview import add_sensor
        add_sensor(self.ctx, kind)

    # ------------------------------------------------------------------ settings
    @property
    def settings(self) -> G.GnssSettings:
        return G.GnssSettings.from_dict(self.ctx.store.project.analysis.gnss)

    def _set(self, text, **kw):
        """Change settings as one undo step (saved with the project); False when nothing changes."""
        if self._syncing:
            return False
        st = self.settings
        for k, v in kw.items():
            setattr(st, k, v)
        new = G.GnssSettings.from_dict(st.to_dict()).to_dict()
        if new == self.settings.to_dict():
            self._sync_fields()                # a clamped value: show what is stored
            return False

        def fn(p):
            p.analysis.gnss = new
        self.ctx.store.edit('analysis', fn, f'GNSS and IMU · {text}')
        return True

    def _commit(self, key, value, text):
        if key == 'mask_deg':
            return self._set(text, mask_override=True, mask_deg=value)
        return self._set(text, **{key: value})

    def _defaults(self):
        if self.ctx.store.project.analysis.gnss:
            def fn(p):
                p.analysis.gnss = {}
            self.ctx.store.edit('analysis', fn, 'GNSS and IMU · defaults')

    def _sync_fields(self):
        self._syncing = True
        try:
            st = self.settings.to_dict()
            for key, f in self.fields.items():
                f.set_value(st[key])           # leaves a field the user is typing in alone
            self.mask_on.blockSignals(True)
            self.mask_on.setChecked(st['mask_override'])
            self.mask_on.blockSignals(False)
            v = self.ctx.store.project.variant
            res = self.result
            auto = {'imu': res.imu['tag'] if res is not None and res.imu else '',
                    'primary': res.baseline['primary'] if res is not None and res.baseline else '',
                    'secondary': res.baseline['secondary'] if res is not None and res.baseline else ''}
            for key, cb in self.combos.items():
                tags = [s.tag for s in G.visible_of(v, 'imu' if key == 'imu' else 'gnss')]
                cb.blockSignals(True)
                cb.clear()
                cb.addItem(f'Auto · {auto[key]}' if auto[key] else 'Auto', '')
                for t in tags:
                    cb.addItem(t, t)
                i = cb.findData(st[key]) if st[key] in tags else 0
                cb.setCurrentIndex(max(i, 0))
                cb.blockSignals(False)
        finally:
            self._syncing = False

    # ------------------------------------------------------------------ state
    def _engine(self):
        if self._eng is None:
            self._eng = self.ctx.analysis.engine.share()
        return self._eng

    def _on_change(self, *_):
        if self.isVisible():
            self.refresh()
        else:
            self._dirty = True                 # computed when the page shows again

    def _on_replaced(self, *_):
        self._gen += 1                         # a run still going is for the old project
        self.busy = False
        self.result = None
        self.sel_uid = None
        self._estimate = self._geo = None
        self._on_change()

    def _on_theme(self, *_):
        if self.result is not None:
            self._show(self.result)

    def showEvent(self, e):
        super().showEvent(e)
        if self._dirty:
            self.refresh()

    def activate(self):
        if self._dirty:
            self.refresh()

    def refresh(self):
        """Compute the study for the project as it is now: at once when it is quick, else on a worker thread."""
        p = self.ctx.store.project
        self._dirty = False
        self._sync_fields()
        if not G.has_inputs(p):
            self._gen += 1
            self.busy = False
            self.result = None
            self.stack.setCurrentIndex(1)
            self.status.setText(EMPTY)
            self.b_export.setEnabled(False)
            return
        self.stack.setCurrentIndex(0)
        if self._estimate is not None and self._estimate < QUICK_S and self._geo == _geo_key(p):
            self._gen += 1
            self.busy = False
            res = G.run(p, self._engine())
            self._learn(res, _geo_key(p))
            self._show(res)
            return
        if self.busy:
            self._pending = True               # run again with the latest edits when this run ends
            return
        self._gen += 1
        gen = self._gen
        self.busy = True
        self._pending = False
        self.status.setText('Tracing the antennas\' sky…')
        snap, eng, geo = p.snapshot(), self._engine(), _geo_key(p)

        def work():
            try:
                res = G.run(snap, eng, cancel=lambda: gen != self._gen)
                self.sig.done.emit(gen, (p, geo, res))
            except RuntimeError:               # the page went while computing
                pass
            except Exception as e:
                try:
                    self.sig.failed.emit(gen, f'{type(e).__name__}: {e}\n{traceback.format_exc()}')
                except RuntimeError:
                    pass
        threading.Thread(target=work, daemon=True).start()

    def _learn(self, res, geo):
        """How long a run that misses the sky cache takes, and the vehicle body it was made for (a new body is
        simplified on the first run: a worker's job)."""
        if res is not None:
            self._estimate = sum(a.ms for a in res.antennas) / 1000.0 + 0.01
            self._geo = geo

    def _done(self, gen, out):
        if gen != self._gen:
            return
        self.busy = False
        p, geo, res = out
        if res is None or p is not self.ctx.store.project:
            return
        self._learn(res, geo)
        if self._pending or not self.isVisible():
            self._pending = False
            self._on_change()                  # edits arrived while it ran: once more with them
            if not self.isVisible():
                self._show(res)
            return
        self._show(res)

    def _failed(self, gen, msg):
        if gen != self._gen:
            return
        self.busy = False
        self.status.setText('Failed: ' + msg.splitlines()[0])

    # ------------------------------------------------------------------ picking an antenna
    def pick(self, uid, select=True):
        """Show an antenna's blocked sectors, highlight it, and (select) select it in the views."""
        res = self.result
        if res is None or not any(a.uid == uid for a in res.antennas):
            return
        self.sel_uid = uid
        for pl in self.plots:
            pl.set_sky(pl.sky, pl.sky is not None and pl.sky.uid == uid)
        row = next(i for i, a in enumerate(res.antennas) if a.uid == uid)
        if self.ants.currentRow() != row:
            self.ants.blockSignals(True)
            self.ants.selectRow(row)
            self.ants.blockSignals(False)
        self._fill_sectors()
        if select:
            sel = self.ctx.store.sel_obj
            if not (sel and sel[0] == 'sensor' and len(sel) > 1 and sel[1] == uid):
                self.ctx.store.select_obj(('sensor', uid))

    def _table_pick(self):
        res = self.result
        r = self.ants.currentRow()
        if res is not None and 0 <= r < len(res.antennas):
            self.pick(res.antennas[r].uid)

    def _on_selection(self):
        sel = self.ctx.store.sel_obj
        if self.isVisible() and sel and sel[0] == 'sensor' and len(sel) > 1 and sel[1] != self.sel_uid:
            self.pick(sel[1], select=False)

    # ------------------------------------------------------------------ showing a result
    def _show(self, res: G.GnssResult):
        self.result = res
        self._sync_fields()
        self.b_export.setEnabled(True)
        if self.sel_uid is None or not any(a.uid == self.sel_uid for a in res.antennas):
            self.sel_uid = res.antennas[0].uid if res.antennas else None
        self._fill_tiles()
        self._fill_plots()
        self._fill_antennas()
        self._fill_sectors()
        self._fill_baseline()
        self._fill_levers()
        self._fill_budget()
        self._fill_findings()
        bad = sum(f['severity'] in G.ISSUE_SEVERITIES for f in res.findings)
        n_a, n_i = len(res.antennas), len(res.imus)
        self.status.setText(f'{res.variant}: {n_a} antenna{"s" * (n_a != 1)}, {n_i} IMU{"s" * (n_i != 1)} · '
                            + ('every check passes' if not bad else f'{bad} finding{"s" * (bad != 1)} to look at'))

    def _fill_tiles(self):
        res = self.result
        sev = {}
        for f in reversed(res.findings):       # the worst of each kind
            sev[f['kind']] = f['severity']
        tone = lambda s: SEV_TONE.get(s)        # noqa: E731
        if res.antennas:
            w = min(res.antennas, key=lambda a: a.visible)
            worst = next((f['severity'] for f in res.findings if f['kind'] in ('no_sky', 'sky') and w.tag in f['tags']),
                         None)
            self.tiles['sky'].set(f'{w.visible * 100:.1f}', '% open',
                                  f'{w.tag} · ~{w.expected_sats:.0f} of {w.n_sats:.0f} satellites', tone(worst))
        else:
            self.tiles['sky'].set('—', '', 'no GNSS antenna')
        b = res.baseline
        if b is not None:
            hs = b['heading_sigma_deg']
            self.tiles['heading'].set(f'{b["horizontal_m"]:.2f}', 'm',
                                      f'{b["primary"]} → {b["secondary"]} · '
                                      + (f'heading σ {hs:.3f}°' if math.isfinite(hs) else 'no σ in the model'),
                                      tone(sev.get('baseline')))
        else:
            why = 'no dual-antenna INS' if not res.imu or res.imu['antennas'] < 2 else 'needs two antennas'
            self.tiles['heading'].set('—', '', why, tone(sev.get('no_heading')))
        o = next((r for r in res.lever_arms if r['kind'] == 'origin'), None)
        if res.imu is not None and o is not None:
            self.tiles['imu'].set(f'{o["length_mm"] / 1000:.2f}', 'm', f'{res.imu["tag"]} · {res.imu["model"]}')
        else:
            self.tiles['imu'].set('—', '', 'no IMU', tone(sev.get('no_receiver')))
        rows = [r for r in (res.budget or {}).get('rows', []) if r['kind'] != 'origin']
        if rows:
            w = max(rows, key=lambda r: r['horizontal_mm'])
            self.tiles['pose'].set(f'{w["horizontal_mm"]:.0f}', 'mm',
                                   f'{w["tag"]} · {w["far_mm"]:.0f} mm at {res.settings.label_range_m:g} m')
        else:
            self.tiles['pose'].set('—', '', 'no IMU' if res.imu is None else 'no camera, lidar or radar')

    def _fill_plots(self):
        res = self.result
        while len(self.plots) > len(res.antennas):
            pl = self.plots.pop()
            self.plots_lay.removeWidget(pl)
            pl.hide()
            pl.deleteLater()
        while len(self.plots) < len(res.antennas):
            pl = SkyPlot()
            pl.picked.connect(self.pick)
            self.plots_lay.addWidget(pl, 1)
            self.plots.append(pl)
        for pl, a in zip(self.plots, res.antennas):
            pl.set_sky(a, a.uid == self.sel_uid)
            pl.setMinimumWidth(260)

    def _fill_antennas(self):
        res = self.result
        t = self.ants
        t.blockSignals(True)
        t.setRowCount(len(res.antennas))
        for i, a in enumerate(res.antennas):
            sev = 'critical' if a.no_sky else 'serious' if a.blocked > G.BLOCKED_SERIOUS else \
                'warning' if a.blocked > G.BLOCKED_WARN else 'good'
            _cell(t, i, 0, a.tag, tip=f'{a.model}\nphase centre x {a.pos[0]:.3f} y {a.pos[1]:.3f} z {a.pos[2]:.3f} m')
            _cell(t, i, 1, f'{a.visible * 100:.1f} %', True, QColor(STATUS[sev]),
                  f'{a.visible * 100:.2f} % of the sky above {a.mask_deg:g}° open; {a.visible_all * 100:.2f} % above '
                  'the horizon')
            _cell(t, i, 2, f'{a.expected_sats:.0f}', True, tip=f'of {a.n_sats:.0f} above the mask in open sky')
            ws = a.worst_sector()
            _cell(t, i, 3, G.sector_text(ws) if ws else '—')
            blk = [b for b in a.blockers if b['share'] > 0.0005]
            _cell(t, i, 4, ', '.join(f'{b["name"]} {b["share"] * 100:.1f} %' for b in blk[:4]) or 'none',
                  tip='\n'.join(f'{b["name"]} ({G.KIND_WORD.get(b["kind"], b["kind"])}): {b["share"] * 100:.2f} % '
                                f'above the mask, up to {b["max_el"]:.0f}°, nearest {b["nearest_m"]:.2f} m'
                                for b in a.blockers) or None,
                  swatch=blocker_color(blk[0]) if blk else None)
            near = [f'{n["name"]} {n["dist_m"]:.2f} m, rises to {n["el_deg"]:.0f}°' for n in a.near]
            more = f' +{len(near) - 1}' if len(near) > 1 else ''
            short = f'{a.near[0]["name"]} {a.near[0]["dist_m"]:.2f} m{more}' if near else 'clear'
            _cell(t, i, 5, short, color=QColor(STATUS['warning']) if near else None, tip='\n'.join(near) or None)
            g = a.ground
            _cell(t, i, 6, ('on ' + g['surface']) if g['ok'] else 'missing',
                  color=None if g['ok'] else THEME.color('muted'),
                  tip=f'stands on {g["surface"] or "nothing"}; {g["points"] - g["missing"]} of {g["points"]} points '
                      f'{G.GROUND_PLANE_M * 100:.0f} cm round its base on a surface')
        _fit(t, len(res.antennas), 4)
        row = next((i for i, a in enumerate(res.antennas) if a.uid == self.sel_uid), -1)
        if row >= 0:
            t.selectRow(row)
        t.blockSignals(False)

    def _fill_sectors(self):
        res = self.result
        a = next((x for x in res.antennas if x.uid == self.sel_uid), None) if res is not None else None
        t = self.secs
        if a is None:
            self.sec_head.setText('Blocked sectors')
            t.setRowCount(0)
            return
        self.sec_head.setText(f'Blocked sectors · {a.tag} (above its {a.mask_deg:g}° mask)')
        t.setRowCount(max(len(a.sectors), 1))
        if not a.sectors:
            _cell(t, 0, 0, '—')
            for c in range(1, 4):
                _cell(t, 0, c, '')
            _cell(t, 0, 4, 'nothing blocks the sky above the mask', color=THEME.color('muted'))
        for i, s in enumerate(a.sectors):
            _cell(t, i, 0, f'{s["az_from"]:.0f}°', True)
            _cell(t, i, 1, f'{s["az_to"]:.0f}°', True)
            _cell(t, i, 2, f'{s["max_el"]:.0f}°', True)
            _cell(t, i, 3, f'{s["share"] * 100:.1f} %', True)
            _cell(t, i, 4, ', '.join(s['blockers']))

    def _fill_baseline(self):
        res = self.result
        b = res.baseline
        if b is None:
            ins = res.imu
            if ins is None:
                txt = 'No IMU or GNSS/INS in this variant.'
            elif ins['antennas'] < 2:
                txt = f'{ins["tag"]} ({ins["model"]}) takes {ins["antennas"]} antenna{"s" * (ins["antennas"] != 1)}: ' \
                      'no dual-antenna heading.'
            else:
                txt = f'{ins["tag"]} takes two antennas; the rig has {len(res.antennas)}: no heading baseline.'
            self.base_text.setText(txt)
            return
        hs = b['heading_sigma_deg']
        self.base_text.setText(
            f'{b["primary"]} (primary) → {b["secondary"]} (secondary): {b["length_m"]:.3f} m, '
            f'{b["horizontal_m"]:.3f} m horizontally.\n'
            f'Vehicle frame: azimuth {b["azimuth_deg"]:.2f}° from forward, vertical angle {b["vertical_deg"]:.2f}°.\n'
            f'{b["ins"]} frame (tell the INS): azimuth {b["imu_azimuth_deg"]:.2f}°, vertical angle '
            f'{b["imu_vertical_deg"]:.2f}°.\n'
            + (f'Heading σ {hs:.3f}° ({b["rated_sigma_deg"]:g}° at {b["rated_baseline_m"]:g} m rated, scaled by the '
               'baseline).' if math.isfinite(hs) else 'The INS model gives no heading σ.'))

    def _fill_levers(self):
        res = self.result
        t = self.levers
        ins = res.imu
        self.lever_head.setText(f'Lever arms from {ins["tag"]}, mm' if ins else 'Lever arms (no IMU)')
        rows = res.lever_arms
        t.setRowCount(len(rows))
        for i, r in enumerate(rows):
            sw = None
            if r['kind'] == 'camera':
                c = self.ctx.store.project.variant.cam(r['uid'])
                sw = THEME.cam(c.slot) if c is not None else None
            _cell(t, i, 0, r['tag'] + (f' · {r["role"]}' if r['role'] else ''), swatch=sw,
                  bold=r['kind'] == 'origin')
            _cell(t, i, 1, KIND_LABEL.get(r['kind'], r['kind']))
            for c in range(3):
                _cell(t, i, 2 + c, _mm(r['imu_mm'][c]), True)
                _cell(t, i, 5 + c, _mm(r['vehicle_mm'][c]), True, THEME.color('ink2'))
            _cell(t, i, 8, _mm(r['length_mm']), True)

    def _fill_budget(self):
        res = self.result
        t = self.budget
        bud = res.budget
        if bud is None:
            self.budget_note.setText('Add an IMU or GNSS/INS for the pose error budget.')
            t.setRowCount(0)
            return
        inp = bud['inputs']
        hs = inp['heading_sigma_deg']
        src = {'model': 'model', 'setting': 'set', 'baseline': 'baseline', 'none': 'not known'}
        self.budget_note.setText(
            f'{res.imu["tag"]}: position {inp["pos_sigma_cm"]:g} cm ({src[inp["pos_source"]]}), vertical × '
            f'{inp["vertical_ratio"]:g}; roll and pitch {inp["attitude_sigma_deg"]:g}° ({src[inp["attitude_source"]]});'
            ' heading ' + (f'{hs:.3f}°' if math.isfinite(hs) else '—') + f' ({src[inp["heading_source"]]}); lever '
            f'arms {inp["lever_arm_sigma_mm"]:g} mm per axis × {inp["lever_arms_measured"]}. All 1σ.')
        rows = bud['rows']
        t.setRowCount(len(rows))
        worst = max((r['horizontal_mm'] for r in rows if r['kind'] != 'origin'), default=None)
        for i, r in enumerate(rows):
            _cell(t, i, 0, r['tag'], bold=r['kind'] == 'origin',
                  tip=f'INS {r["ins_mm"][0]:.1f}/{r["ins_mm"][1]:.1f}/{r["ins_mm"][2]:.1f} mm, attitude on the lever '
                      f'arm {r["rotation_mm"][0]:.1f}/{r["rotation_mm"][1]:.1f}/{r["rotation_mm"][2]:.1f} mm, '
                      f'lever-arm measurement {r["measure_mm"][0]:.1f} mm per axis')
            _cell(t, i, 1, f'{r["lever_m"]:.2f}', True)
            for c in range(3):
                _cell(t, i, 2 + c, _mm(r['sigma_mm'][c], 1), True)
            hl = worst is not None and r['kind'] != 'origin' and abs(r['horizontal_mm'] - worst) < 1e-9
            _cell(t, i, 5, _mm(r['horizontal_mm'], 1), True, bold=hl)
            _cell(t, i, 6, _mm(r['total_mm'], 1), True)
            _cell(t, i, 7, _mm(r['far_mm'], 0), True, THEME.color('ink2'))

    def _fill_findings(self):
        res = self.result
        for w in self.findings:
            self.find_lay.removeWidget(w)
            w.hide()                    # until the deferred delete runs it would still paint over the new rows
            w.deleteLater()
        self.findings = []
        for f in res.findings:
            sev = f['severity']
            w = StatusLine(sev if sev in STATUS else 'good', f['message'])
            if sev not in STATUS:       # a note: no verdict
                w.icon.setPixmap(icons.pixmap('dash', THEME.color('muted'), 12))
            self.find_lay.insertWidget(self.find_lay.count() - 1, w)
            self.findings.append(w)

    # ------------------------------------------------------------------ export
    def export(self, folder=None):
        res = self.result
        if res is None:
            return []
        if folder is None:
            folder = QFileDialog.getExistingDirectory(self, 'Export the GNSS and IMU study into')
            if not folder:
                return []
        from ...core import provenance as PV          # where the JSON came from
        paths = PV.stamp_paths(G.export(res, self.ctx.store.project, folder), self.ctx.store.project, folder)
        self.ctx.msg(f'Exported the GNSS and IMU study to {folder}', 5000)
        return paths
