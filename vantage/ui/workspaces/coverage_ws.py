"""Analysis ▸ Lidar and radar coverage: where the lidars and radars reach and what of them the vehicle blocks, how far
out cameras, lidars and radars detect each object per direction against the perception range's required ranges, and
whether the lidar points can be projected into each camera."""
from __future__ import annotations

import math

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (QAbstractItemView, QFileDialog, QGridLayout, QHBoxLayout, QHeaderView, QProgressBar,
                             QSplitter, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from ...core import perception as PR
from ...core import sensor_coverage as SC
from ...core import sensors as S
from ...core.mathutil import parse_number
from ..coverageview import (EMPTY, SHORT, CoverageMap, VerdictGrid, controller, has_sensors, modality_color,
                            verdict_color)
from ..theme import THEME, STATUS, mono
from ..widgets import ROW_H, SegmentedControl, button, hline, label

# (key, header, decimals or None, editable, tooltip)
CLASS_COLS = (('name', 'Object', None, False, 'The object class (sizes as in the perception range)'),
              ('size', 'Size', None, False, 'Height × width of the object, m; a raised one (a sign) stands on a post'),
              ('min_points', 'Min pts', 0, True, 'Lidar points per frame your detector needs on the object'),
              ('rcs', 'RCS m²', 1, True, 'Radar cross-section of the object, m² (0: radars are not relied on for it)'))
REACH = (('0', 'All'), ('60', '60 m'), ('20', '20 m'))


def _table(cols, rows=0):
    t = QTableWidget(rows, cols)
    t.verticalHeader().setVisible(False)
    t.verticalHeader().setDefaultSectionSize(ROW_H)
    t.horizontalHeader().setFixedHeight(ROW_H)
    t.setShowGrid(False)
    t.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
    return t


def _num(text, tip=None, color=None):
    it = QTableWidgetItem(text)
    it.setFont(mono(12))
    it.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    if tip:
        it.setToolTip(tip)
    if color is not None:
        it.setForeground(color)
    return it


def _fit(t, rows, most=5):
    """A table as tall as its rows (up to most; it scrolls beyond)."""
    t.setFixedHeight(ROW_H * (min(max(rows, 1), most) + 1) + 6)


def _m(v, nd=0, limit=None):
    """A range as text: '—' for none, '≥250' where it reaches the end of the study."""
    if v is None or not math.isfinite(v):
        return '—'
    return f'≥{limit:.0f}' if limit is not None and v >= limit - 1e-6 else f'{v:.{nd}f}'


class SensorCoveragePage(QWidget):
    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        self.ctl = controller(ctx)
        self.setObjectName('base')
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        split = QSplitter(Qt.Orientation.Horizontal)
        split.setHandleWidth(2)
        split.addWidget(self._left())
        # ---------------- centre: the plan
        cw = QWidget()
        cw.setObjectName('panel')
        cl = QVBoxLayout(cw)
        cl.setContentsMargins(8, 4, 8, 6)
        cl.setSpacing(4)
        head = self._head = QHBoxLayout()
        self.pick = SegmentedControl([('0', 'Car')], '0')
        self.pick.changed.connect(lambda k: self._set_class(int(k)))
        head.addWidget(self.pick)
        head.addStretch(1)
        self.layer = SegmentedControl([('density', 'Density'), ('rings', 'Rings')], 'density')
        self.layer.setToolTip('Lidar ground points: points per m² per frame summed over the lidars, or each channel\'s '
                              'ground ring where the vehicle does not block it')
        self.layer.changed.connect(lambda m: self.map.set_layer(m))
        head.addWidget(self.layer)
        head.addSpacing(8)
        self.reach = SegmentedControl(list(REACH), '0')
        self.reach.setToolTip('How far out the plan reaches')
        self.reach.changed.connect(lambda v: self.map.set_reach(float(v)))
        head.addWidget(self.reach)
        cl.addLayout(head)
        self.map = CoverageMap()
        cl.addWidget(self.map, 1)
        split.addWidget(cw)
        # ---------------- right: verdicts, lidars, cameras
        right = QSplitter(Qt.Orientation.Vertical)
        right.setHandleWidth(2)
        tw = QWidget()
        tw.setObjectName('panel')
        tl = QVBoxLayout(tw)
        tl.setContentsMargins(8, 6, 8, 6)
        tl.setSpacing(4)
        tl.addWidget(label('Sensor types reaching the required range', bold=True))
        self.grid = VerdictGrid()
        self.grid.setToolTip('Per object and sector, how many of camera, lidar and radar reach the required range in '
                             'every direction of the sector: green two or more, amber one, red none')
        self.grid.picked.connect(lambda k: self._set_class(k))
        tl.addWidget(self.grid)
        self.sector_head = label('Car · shortest range in each sector', bold=True)
        tl.addWidget(self.sector_head)
        self.sectors = _table(6)
        self.sectors.setHorizontalHeaderLabels(['Sector', 'Needed', 'Camera', 'Lidar', 'Radar', 'Verdict'])
        self.sectors.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        hh = self.sectors.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for c in range(1, 6):
            hh.setSectionResizeMode(c, QHeaderView.ResizeMode.ResizeToContents)
        self.sectors.setFixedHeight(ROW_H * (len(PR.SECTORS) + 1) + 6)
        tl.addWidget(self.sectors)
        tl.addStretch(1)
        right.addWidget(tw)
        lw = QWidget()
        lw.setObjectName('panel')
        ll = QVBoxLayout(lw)
        ll.setContentsMargins(8, 6, 8, 6)
        ll.setSpacing(4)
        ll.addWidget(label('Lidars', bold=True))
        self.lidars = _table(7)
        heads = [('Lidar', 'The lidar and its model'),
                 ('On car %', 'Share of each frame\'s points that land on the own vehicle or rig'),
                 ('Ring m', 'Nearest ground ring: how far from the lidar its lowest beam meets the ground'),
                 ('Blind m', 'Median over the directions the lidar reaches the ground: how far from the body the '
                             'nearest ground point it reaches lies (the vehicle\'s blocking included)')] + \
                [(f'{d:g} m', f'Points per frame on a car {d:g} m ahead of the lidar') for d in SC.CAR_AT]
        for c, (h, tip) in enumerate(heads):
            it = QTableWidgetItem(h)
            it.setToolTip(tip)
            self.lidars.setHorizontalHeaderItem(c, it)
        self.lidars.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        hh = self.lidars.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for c in range(1, 7):
            hh.setSectionResizeMode(c, QHeaderView.ResizeMode.ResizeToContents)
        _fit(self.lidars, 1)
        ll.addWidget(self.lidars)
        ll.addWidget(label('Radars', bold=True))
        self.radars = _table(2)
        self.radars.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.radars.setToolTip('Per radar, the farthest range at which it detects each object (from the vehicle '
                               'centre, in any direction): its zones scaled by (RCS / 10 m²)^¼, the vehicle\'s '
                               'blocking included')
        _fit(self.radars, 1)
        ll.addWidget(self.radars)
        ll.addWidget(label('Lidar points in each camera', bold=True))
        self.overlap = _table(6)
        heads = [('Camera', 'The camera')] + \
                [(f'{d:g} m %', f'Share of the camera\'s image (within the lens circle, off the body) whose scene '
                                f'point {d:g} m out (or on the ground, where nearer) some lidar covers: in its field '
                                f'and range, not hidden from it') for d in SC.OVERLAP_AT] + \
                [('Best', 'The lidar that covers most of the image at 20 m'),
                 ('Rings', f'Rings of that lidar crossing the image\'s centre column {SC.RINGS_AT:g} m out')]
        for c, (h, tip) in enumerate(heads):
            it = QTableWidgetItem(h)
            it.setToolTip(tip)
            self.overlap.setHorizontalHeaderItem(c, it)
        self.overlap.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        hh = self.overlap.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for c in range(1, 6):
            hh.setSectionResizeMode(c, QHeaderView.ResizeMode.ResizeToContents)
        ll.addWidget(self.overlap, 1)
        right.addWidget(lw)
        right.setSizes([370, 510])
        split.addWidget(right)
        split.setSizes([330, 760, 600])
        lay.addWidget(split)
        self.ctl.changed.connect(self.refresh)
        ctx.store.changed.connect(lambda *_: self._sync_fields())
        ctx.store.project_replaced.connect(self._sync_fields)
        THEME.changed.connect(self._retheme)           # a bound slot: disconnected when the page goes
        self._shown = None
        self._syncing = False
        self._sync_fields()
        self.refresh()

    def _retheme(self, *_):
        """The tables' colours are set when they are filled: fill them again in the new theme."""
        self._shown = None
        self._sync_fields()
        self.refresh()

    # ------------------------------------------------------------------ settings panel
    def _left(self):
        w = QWidget()
        w.setObjectName('panel')
        w.setMinimumWidth(330)
        ll = QVBoxLayout(w)
        ll.setContentsMargins(8, 6, 8, 8)
        ll.setSpacing(4)
        top = QHBoxLayout()
        top.addWidget(label('Lidar and radar coverage'))
        top.addStretch(1)
        top.addWidget(label('every sensor, every direction', muted=True, small=True))
        ll.addLayout(top)
        ll.addWidget(hline())
        ll.addWidget(label('Where the lidars and radars reach, what of the lidar beams lands on the own vehicle, and '
                           'how far out each object is detected next to the cameras. A lidar detects an object when '
                           'its points on it in one frame reach the minimum; a radar when the object is in a zone at '
                           'the range its cross-section allows and nothing of the vehicle is in between. The cameras '
                           'are the perception range\'s.', muted=True, small=True, wrap=True))
        ll.addSpacing(4)
        ll.addWidget(label('Objects', bold=True))
        self.classes = _table(len(CLASS_COLS))
        for c, (_, head, _, _, tip) in enumerate(CLASS_COLS):
            it = QTableWidgetItem(head)
            it.setToolTip(tip)
            self.classes.setHorizontalHeaderItem(c, it)
        self.classes.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.classes.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        hh = self.classes.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for c, wd in ((1, 76), (2, 58), (3, 58)):
            hh.setSectionResizeMode(c, QHeaderView.ResizeMode.Fixed)
            hh.resizeSection(c, wd)
        self.classes.itemChanged.connect(self._class_edited)
        ll.addWidget(self.classes)
        row = QHBoxLayout()
        b_per = self.b_from = button('From perception', 'ghost', None,
                                     'The perception range\'s objects (names and sizes), keeping the minimum points '
                                     'and cross-sections of the objects of the same name')
        b_per.clicked.connect(lambda: self.ctl.classes_from_perception())
        b_def = button('Defaults', 'ghost', None, 'Back to the default objects')
        b_def.clicked.connect(lambda: self.ctl.set_classes(SC.DEFAULT_CLASSES, 'Lidar and radar coverage · default '
                                                                                'objects'))
        row.addWidget(b_per)
        row.addWidget(b_def)
        row.addStretch(1)
        ll.addLayout(row)
        ll.addSpacing(6)
        rh = QHBoxLayout()
        rh.addWidget(label('Required range per sector', bold=True))
        rh.addStretch(1)
        b_req = button('Edit…', 'ghost', None, 'The required ranges are the perception range\'s: every sensor type is '
                                               'judged against the same requirement')
        b_req.clicked.connect(self._edit_required)
        rh.addWidget(b_req)
        ll.addLayout(rh)
        grid = QGridLayout()
        grid.setVerticalSpacing(2)
        grid.setHorizontalSpacing(6)
        grid.setColumnStretch(4, 1)
        self.req = {}
        for i, (name, _) in enumerate(PR.SECTORS):
            a = label(name.replace('-', ' ').capitalize(), ink2=True)
            a.setFixedWidth(64 if i % 2 == 0 else 86)
            if i % 2:
                a.setContentsMargins(10, 0, 0, 0)
            v = label('', mono_=True)
            v.setFixedWidth(60)
            v.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            grid.addWidget(a, i // 2, (i % 2) * 2)
            grid.addWidget(v, i // 2, (i % 2) * 2 + 1)
            self.req[name] = v
        ll.addLayout(grid)
        ll.addWidget(label('From Analysis ▸ Perception range.', muted=True, small=True))
        ll.addStretch(1)
        self.status = label('', muted=True, small=True, wrap=True)
        ll.addWidget(self.status)
        self.bar = QProgressBar()
        self.bar.setTextVisible(False)
        self.bar.setRange(0, 1000)
        self.bar.hide()
        self.ctl.progress.connect(lambda f: self.bar.setValue(int(f * 1000)))
        ll.addWidget(self.bar)
        bl = QHBoxLayout()
        bl.addStretch(1)
        self.b_export = button('Export…', None, 'export', 'The ranges, verdicts, lidar figures and camera overlap as '
                                                          'JSON, and the plan as PNG')
        self.b_export.clicked.connect(lambda: self.export())
        self.b_run = button('Run', 'primary')
        self.b_run.clicked.connect(lambda: self.ctl.run())
        bl.addWidget(self.b_export)
        bl.addWidget(self.b_run)
        ll.addLayout(bl)
        return w

    def _edit_required(self):
        win = self.ctx.window
        if win is not None:
            win.set_page('perception')

    def _class_edited(self, item):
        if self._syncing:
            return
        rows = [dict(c) for c in self.ctl.settings.classes]      # the stored values, not the rounded text
        r, (key, _, _, editable, _) = item.row(), CLASS_COLS[item.column()]
        if not editable or not 0 <= r < len(rows):
            return
        try:
            value = float(parse_number(item.text().strip(), 'plain'))
            rows[r][key] = value
        except Exception:                      # parse_number raises on anything that is not a number
            value = None
        what = {'min_points': 'minimum points', 'rcs': 'radar cross-section'}[key]
        if value is None or not self.ctl.set_classes(rows, f'Lidar and radar coverage · {rows[r]["name"]} {what}'):
            self._sync_fields()                # not a number, or not a usable value: show what is stored

    def _sync_fields(self):
        st = self.ctl.settings
        self._syncing = True
        try:
            self.classes.setRowCount(len(st.classes))
            for r, c in enumerate(st.classes):
                for j, (key, _, dec, editable, _) in enumerate(CLASS_COLS):
                    if key == 'size':
                        text = f'{c["height"]:.1f} × {c["width"]:.1f}'
                    else:
                        v = c[key]
                        text = str(v) if dec is None else f'{v:.{dec}f}'
                    it = self.classes.item(r, j)
                    if it is None:             # items are kept (an edit lands here while its item is still in use)
                        it = QTableWidgetItem()
                        if dec is not None or key == 'size':
                            it.setFont(mono(12))
                            it.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                        if not editable:
                            it.setFlags(it.flags() & ~Qt.ItemFlag.ItemIsEditable)
                        self.classes.setItem(r, j, it)
                    if it.text() != text:
                        it.setText(text)
                    if key == 'size':
                        it.setToolTip(f'{c["height"]:g} m high, {c["width"]:g} m wide'
                                      + (f', {c["base"]:g} m above the ground' if c['base'] > 0 else ''))
                    if key == 'rcs':
                        it.setForeground(THEME.color('muted') if c['rcs'] <= 0 else THEME.color('ink'))
                        it.setToolTip('Radars are not relied on for this object' if c['rcs'] <= 0 else
                                      f'A radar sees it to {(c["rcs"] / 10.0) ** 0.25 * 100:.0f} % of its range '
                                      f'on a car')
            self.classes.setFixedHeight(ROW_H * (len(st.classes) + 1) + 6)
            req = PR.PerceptionSettings.from_dict(self.ctx.store.project.analysis.perception).required
            for name, v in self.req.items():
                v.setText(f'{req.get(name, 0.0):g} m')
        finally:
            self._syncing = False

    # ------------------------------------------------------------------ state
    def showEvent(self, e):
        super().showEvent(e)
        self.ctl.set_active(True)

    def hideEvent(self, e):
        super().hideEvent(e)
        self.ctl.set_active(False)

    def activate(self):
        self.ctl.set_active(True)

    def _set_class(self, k):
        res = self.ctl.result
        if res is None or not 0 <= k < len(res.classes):
            return
        self.pick.set_value(str(k))
        self.map.set_class(k)
        self.grid.set_class(k)
        self._fill_sectors(res, k)

    def refresh(self):
        ctl = self.ctl
        res = ctl.result
        win = self.ctx.window
        empty = not has_sensors(self.ctx.store.project)
        self.b_run.setEnabled(not empty and not (ctl.busy and win is not None and 'sensor_coverage' in win._jobs))
        self.bar.setVisible(ctl.busy)
        self.b_export.setEnabled(res is not None)
        status = EMPTY if empty else ctl.status
        self.status.setText(status + (' · stale: the design or the settings changed' if ctl.stale and res else ''))
        self.map.set_empty(EMPTY if empty else 'Run the lidar and radar coverage')
        if res is self._shown:
            return
        self._shown = res
        self.map.set_result(res)
        self.grid.set_result(res)
        if res is None:
            for t in (self.sectors, self.lidars, self.radars, self.overlap):
                t.setRowCount(0)
            return
        opts = [(str(i), c.name) for i, c in enumerate(res.classes)]
        if [(k, b.text()) for k, b in self.pick.buttons.items()] != opts:
            cur = self.pick.value()
            old = self.pick
            self.pick = SegmentedControl(opts, cur if cur in dict(opts) else '0')
            self.pick.changed.connect(lambda k: self._set_class(int(k)))
            self._head.replaceWidget(old, self.pick)
            old.hide()
            old.setParent(None)
            old.deleteLater()
        k = int(self.pick.value() or 0)
        self.map.set_class(k)
        self.grid.set_class(k)
        self._fill_sectors(res, k)
        self._fill_lidars(res)
        self._fill_radars(res)
        self._fill_overlap(res)

    def _fill_sectors(self, res, k):
        cl = res.classes[k]
        self.sector_head.setText(f'{cl.name} · shortest range in each sector')
        t = self.sectors
        t.setRowCount(len(res.sectors))
        for r, s in enumerate(res.sectors):
            d = s['classes'][cl.name]
            t.setItem(r, 0, QTableWidgetItem(s['name'].replace('-', ' ').capitalize()))
            t.setItem(r, 1, _num(f'{s["required_m"]:g}'))
            for j, mod in enumerate(SC.MODALITIES, 2):
                v = d[f'{mod}_m']
                none = 'not relied on (RCS 0)' if mod == 'radar' and cl.rcs <= 0 and res.radars else f'no {mod}'
                col = THEME.color('muted') if v is None else QColor(STATUS['good']) if d['meets'][mod] else \
                    THEME.color('ink2')
                t.setItem(r, j, _num(_m(v, limit=float(res.ranges[-1])), f'{cl.name}, {s["name"]}: ' + (
                    none if v is None else f'the {mod}s detect it out to {v:.1f} m in every direction of the sector'),
                    col))
            it = QTableWidgetItem(d['verdict'].capitalize())
            it.setForeground(verdict_color(d['verdict']))
            it.setToolTip(f'{d["modalities"]} of 3 sensor types reach the required {s["required_m"]:g} m')
            t.setItem(r, 5, it)

    def _fill_lidars(self, res):
        t = self.lidars
        t.setRowCount(len(res.lidars))
        _fit(t, len(res.lidars))
        for r, li in enumerate(res.lidars):
            it = QTableWidgetItem(f'{li.tag} · {li.spec.name}')
            it.setForeground(modality_color('lidar'))
            it.setToolTip(f'{li.spec.vendor} {li.spec.name}: {li.spec.summary()}'.strip())
            t.setItem(r, 0, it)
            sec = ', '.join(f'{SHORT.get(n, n)} {100 * v:.0f} %' for n, v in li.blocked_sectors.items() if v)
            tone = 'good' if li.blocked < 0.05 else 'warning' if li.blocked < 0.25 else 'critical'
            t.setItem(r, 1, _num(f'{100 * li.blocked:.1f}', 'On the own vehicle or rig, by the beam\'s heading: '
                                 + (sec or 'none'), QColor(STATUS[tone])))
            t.setItem(r, 2, _num(_m(li.ring_m, 1)))
            st = SC._stats(li.blind)
            t.setItem(r, 3, _num(_m(st['median_m'], 1) if st['median_m'] is not None else '—',
                                 (f'From the body outline: {st["min_m"]:.1f} m at best, {st["max_m"]:.1f} m at worst; '
                                  f'reaches the ground in {100 * st["directions_reached"]:.0f} % of the directions')
                                 if st['median_m'] is not None else 'Reaches the ground in no direction'))
            for j, d in enumerate(SC.CAR_AT, 4):
                v = li.car_points[d]
                car = next((c for c in res.classes if c.name.lower() == 'car'), None)
                need = car.min_points if car is not None else SC.DEFAULT_CLASSES[0]['min_points']
                t.setItem(r, j, _num(f'{v:.0f}', f'{v:.1f} points per frame on a car {d:g} m ahead of {li.tag}',
                                     QColor(STATUS['good']) if v >= need else THEME.color('ink2')))

    def _fill_radars(self, res):
        t = self.radars
        ks = [k for k, c in enumerate(res.classes) if c.rcs > 0]
        t.setColumnCount(1 + len(ks))
        t.setHorizontalHeaderLabels(['Radar'] + [f'{res.classes[k].name} m' for k in ks])
        hh = t.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for c in range(1, 1 + len(ks)):
            hh.setSectionResizeMode(c, QHeaderView.ResizeMode.ResizeToContents)
        t.setRowCount(len(res.radars))
        _fit(t, len(res.radars))
        for r, rd in enumerate(res.radars):
            it = QTableWidgetItem(f'{rd.tag} · {rd.spec.name}')
            it.setForeground(modality_color('radar'))
            it.setToolTip(f'{rd.spec.name}: {rd.spec.summary()} · heading {S.heading(rd.sensor):.0f}°')
            t.setItem(r, 0, it)
            for j, k in enumerate(ks, 1):
                v = rd.far[k]
                far = float(np.nanmax(v)) if np.isfinite(v).any() else None
                a = float(res.az[int(np.nanargmax(v))]) if far else None
                cl = res.classes[k]
                t.setItem(r, j, _num(_m(far, limit=float(res.ranges[-1])),
                                     f'{cl.name} (RCS {cl.rcs:g} m²): ' + (f'to {far:.0f} m, at {a:.0f}°' if far
                                                                           else 'not detected')))

    def _fill_overlap(self, res):
        t = self.overlap
        t.setRowCount(len(res.overlap))
        for r, o in enumerate(res.overlap):
            it = QTableWidgetItem(o['tag'])
            it.setForeground(THEME.cam(o['slot']))
            t.setItem(r, 0, it)
            for j, d in enumerate(SC.OVERLAP_AT, 1):
                t.setItem(r, j, _num(f'{100 * o["share"][f"{d:g}"]:.0f}'))
            per = ', '.join(f'{k} {100 * v:.0f} %' for k, v in o['per_lidar'].items())
            t.setItem(r, 4, QTableWidgetItem(o['best'] or '—'))
            t.item(r, 4).setToolTip(f'At 20 m: {per}' if per else 'No lidar covers the image')
            t.setItem(r, 5, _num(str(o['rings'])))

    # ------------------------------------------------------------------ export
    def export(self, folder=None):
        res = self.ctl.result
        if res is None:
            return []
        if folder is None:
            folder = QFileDialog.getExistingDirectory(self, 'Export the lidar and radar coverage into')
            if not folder:
                return []
        paths = SC.export(res, self.ctx.store.project, folder, k_sel=int(self.pick.value() or 0))
        self.ctx.msg(f'Exported the lidar and radar coverage to {folder}', 5000)
        return paths
