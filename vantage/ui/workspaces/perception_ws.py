"""Analysis ▸ Perception range: how far out the rig resolves cars, pedestrians, cyclists and signs, per direction,
against the ranges the dataset needs; and how the cameras overlap for stereo."""
from __future__ import annotations

import json
import math
import os

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (QAbstractItemView, QFileDialog, QGridLayout, QHBoxLayout, QHeaderView, QProgressBar,
                             QSplitter, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from ...core import perception as PR
from ...core.mathutil import parse_number
from ..perceptionview import EnvelopeMap, RoseGrid, class_color, controller
from ..theme import THEME, STATUS, mono
from ..widgets import ROW_H, ScrubSpinBox, SegmentedControl, button, hline, label

CLASS_COLS = (('name', 'Object', None, 'The object class'),
              ('height', 'Height', 2, 'Height of the object, m'),
              ('width', 'Width', 2, 'Width of the object, m'),
              ('min_px', 'Min px', 0, 'The height in the image, in pixels, your detector needs to find the object'),
              ('base', 'Raised', 2, 'Height of its bottom above the ground, m (a sign on a post)'))
SHORT = {'front': 'Front', 'front-left': 'FL', 'left': 'Left', 'rear-left': 'RL', 'rear': 'Rear', 'rear-right': 'RR',
         'right': 'Right', 'front-right': 'FR'}


def _table(cols, rows=0):
    t = QTableWidget(rows, cols)
    t.verticalHeader().setVisible(False)
    t.verticalHeader().setDefaultSectionSize(ROW_H)
    t.horizontalHeader().setFixedHeight(ROW_H)
    t.setShowGrid(False)
    t.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
    return t


class PerceptionPage(QWidget):
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
        # ---------------- centre: the envelopes around the vehicle
        cw = QWidget()
        cw.setObjectName('panel')
        cl = QVBoxLayout(cw)
        cl.setContentsMargins(8, 4, 8, 6)
        cl.setSpacing(4)
        head = self._head = QHBoxLayout()
        self.pick = SegmentedControl([('0', 'Car')], '0')
        self.pick.changed.connect(lambda k: self.map.set_class(int(k)))
        head.addWidget(self.pick)
        head.addStretch(1)
        self.mode = SegmentedControl([('classes', 'All objects'), ('cameras', 'By camera')], 'classes')
        self.mode.changed.connect(lambda m: self.map.set_mode(m))
        head.addWidget(self.mode)
        cl.addLayout(head)
        self.map = EnvelopeMap()
        cl.addWidget(self.map, 1)
        split.addWidget(cw)
        # ---------------- right: roses, verdicts, stereo pairs
        right = QSplitter(Qt.Orientation.Vertical)
        right.setHandleWidth(2)
        self.roses = RoseGrid()
        rw = QWidget()
        rw.setObjectName('panel')
        rl = QVBoxLayout(rw)
        rl.setContentsMargins(8, 6, 8, 6)
        rl.addWidget(label('Range per object · required range per sector', bold=True))
        rl.addWidget(self.roses, 1)
        right.addWidget(rw)
        tw = QWidget()
        tw.setObjectName('panel')
        tl = QVBoxLayout(tw)
        tl.setContentsMargins(8, 6, 8, 6)
        tl.setSpacing(4)
        tl.addWidget(label('Worst direction in each sector', bold=True))
        self.verdicts = _table(len(PR.SECTORS) + 1)
        self.verdicts.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.verdicts.setToolTip('The shortest range in the sector at which the rig still resolves the object; green '
                                 'when it reaches the required range')
        tl.addWidget(self.verdicts)
        tl.addWidget(label('Camera pairs that overlap (stereo)', bold=True))
        self.pairs = _table(6)
        self.pairs.setHorizontalHeaderLabels(['Pair', 'Overlap °', 'Baseline m', 'Δz 10 m', 'Δz 20 m', 'Δz 50 m'])
        self.pairs.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.pairs.setToolTip(f'Overlap: the directions in which both cameras see a car {PR.OVERLAP_AT:g} m away. '
                              f'Δz: the depth resolution of the pair as stereo for a {PR.DISPARITY_PX:g} px disparity '
                              f'error, z²·Δd / (f·B), with the lower angular resolution of the two')
        hh = self.pairs.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for c in range(1, 6):
            hh.setSectionResizeMode(c, QHeaderView.ResizeMode.ResizeToContents)
        tl.addWidget(self.pairs, 1)
        right.addWidget(tw)
        right.setSizes([420, 460])
        split.addWidget(right)
        split.setSizes([340, 760, 600])
        lay.addWidget(split)
        self.ctl.changed.connect(self.refresh)
        ctx.store.changed.connect(lambda *_: self._sync_fields())
        ctx.store.project_replaced.connect(self._sync_fields)
        self._shown = None
        self._syncing = False
        self._sync_fields()
        self.refresh()

    # ------------------------------------------------------------------ settings panel
    def _left(self):
        w = QWidget()
        w.setObjectName('panel')
        w.setMinimumWidth(340)
        ll = QVBoxLayout(w)
        ll.setContentsMargins(8, 6, 8, 8)
        ll.setSpacing(4)
        top = QHBoxLayout()
        top.addWidget(label('Perception range'))
        top.addStretch(1)
        top.addWidget(label('every camera, every direction', muted=True, small=True))
        ll.addLayout(top)
        ll.addWidget(hline())
        ll.addWidget(label('How far out the rig resolves the objects your dataset needs. An object stands on the ground; '
                           'a camera resolves it when it is in the image, not hidden by the body, and at least the '
                           'minimum pixel height tall through the camera\'s real lens. Set the minimums your detector '
                           'needs; the defaults are typical.', muted=True, small=True, wrap=True))
        ll.addSpacing(4)
        ll.addWidget(label('Objects', bold=True))
        self.classes = _table(len(CLASS_COLS))
        for c, (_, head, _, tip) in enumerate(CLASS_COLS):
            it = QTableWidgetItem(head)
            it.setToolTip(tip)
            self.classes.setHorizontalHeaderItem(c, it)
        self.classes.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.classes.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        hh = self.classes.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for c in range(1, len(CLASS_COLS)):
            hh.setSectionResizeMode(c, QHeaderView.ResizeMode.Fixed)
            hh.resizeSection(c, 52)
        self.classes.itemChanged.connect(self._class_edited)
        ll.addWidget(self.classes)
        row = QHBoxLayout()
        b_add = self.b_add = button('Add', 'ghost', 'plus', 'Add an object class')
        b_add.clicked.connect(self._add_class)
        b_del = button('Remove', 'ghost', 'minus', 'Remove the selected object class')
        b_del.clicked.connect(self._remove_class)
        b_def = button('Defaults', 'ghost', None, 'Back to the default object classes')
        b_def.clicked.connect(lambda: self.ctl.set_classes(PR.DEFAULT_CLASSES, 'Perception range · default objects'))
        for b in (b_add, b_del, b_def):
            row.addWidget(b)
        row.addStretch(1)
        ll.addLayout(row)
        ll.addSpacing(6)
        ll.addWidget(label('Required range per sector', bold=True))
        grid = QGridLayout()
        grid.setVerticalSpacing(2)
        grid.setHorizontalSpacing(6)
        grid.setColumnStretch(4, 1)
        self.req = {}
        for i, (name, _) in enumerate(PR.SECTORS):
            f = ScrubSpinBox('', 'plain', 0, 'm', step=5.0, width=76)
            f.setToolTip(f'The {name} sector must resolve every object out to this range')
            f.committed.connect(lambda v, n=name: self.ctl.set_required(n, v) or self._sync_fields())
            a = label(name.replace('-', ' ').capitalize(), ink2=True)
            a.setFixedWidth(64 if i % 2 == 0 else 86)
            if i % 2:
                a.setContentsMargins(10, 0, 0, 0)
            grid.addWidget(a, i // 2, (i % 2) * 2)
            grid.addWidget(f, i // 2, (i % 2) * 2 + 1)
            self.req[name] = f
        ll.addLayout(grid)
        ll.addSpacing(4)
        mr = QHBoxLayout()
        a = label('Out to', ink2=True)
        a.setFixedWidth(64)
        mr.addWidget(a)
        self.max_range = ScrubSpinBox('', 'plain', 0, 'm', step=10.0, width=76)
        self.max_range.setToolTip('The farthest range analysed')
        self.max_range.committed.connect(lambda v: self.ctl.set_option(max_range=float(v)) or self._sync_fields())
        mr.addWidget(self.max_range)
        mr.addStretch(1)
        ll.addLayout(mr)
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
        self.b_export = button('Export…', None, 'export', 'The ranges, verdicts and stereo pairs as JSON, and the plan as PNG')
        self.b_export.clicked.connect(lambda: self.export())
        self.b_run = button('Run', 'primary')
        self.b_run.clicked.connect(lambda: self.ctl.run())
        bl.addWidget(self.b_export)
        bl.addWidget(self.b_run)
        ll.addLayout(bl)
        return w

    def _class_edited(self, item):
        if self._syncing:
            return
        rows = [dict(c) for c in self.ctl.settings.classes]      # the stored values, not the rounded text
        r, (key, _, dec, _) = item.row(), CLASS_COLS[item.column()]
        txt = item.text().strip()
        try:
            value = (txt or rows[r]['name']) if dec is None else \
                float(parse_number(txt, 'plain' if key == 'min_px' else 'length'))     # '150 cm' works too
            rows[r][key] = value
        except Exception:                      # parse_number raises on anything that is not a number
            value = None
        what = {'name': 'name', 'min_px': 'minimum pixels', 'base': 'height above ground'}.get(key, key)
        if value is None or not self.ctl.set_classes(rows, f'Perception range · {rows[r]["name"]} {what}'):
            self._sync_fields()                # not a number, or not a usable size: show what is stored

    def _add_class(self):
        rows = self.ctl.settings.classes
        names, n = {c['name'] for c in rows}, 1
        while (name := 'Object' + (f' {n}' if n > 1 else '')) in names:
            n += 1
        rows = rows + [{'name': name, 'height': 1.0, 'width': 0.5, 'min_px': 16.0, 'base': 0.0}]
        if self.ctl.set_classes(rows, f'Perception range · add {name}'):
            self.classes.setCurrentCell(len(rows) - 1, 0)

    def _remove_class(self):
        r = self.classes.currentRow()
        rows = self.ctl.settings.classes
        if 0 <= r < len(rows) and len(rows) > 1:
            name = rows[r]['name']
            self.ctl.set_classes(rows[:r] + rows[r + 1:], f'Perception range · remove {name}')

    def _sync_fields(self):
        st = self.ctl.settings
        self._syncing = True
        try:
            self.classes.setRowCount(len(st.classes))
            for r, c in enumerate(st.classes):
                for j, (key, _, dec, _) in enumerate(CLASS_COLS):
                    v = c[key]
                    text = str(v) if dec is None else f'{v:.{dec}f}'
                    it = self.classes.item(r, j)
                    if it is None:             # items are kept (an edit lands here while its item is still in use)
                        it = QTableWidgetItem()
                        if dec is not None:
                            it.setFont(mono(12))
                            it.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                        self.classes.setItem(r, j, it)
                    if it.text() != text:
                        it.setText(text)
                    if dec is None:
                        it.setForeground(class_color(r))
            self.classes.setFixedHeight(ROW_H * (len(st.classes) + 1) + 6)
            for name, f in self.req.items():
                f.set_value(st.required.get(name, 0.0))     # leaves a field the user is typing in alone
            self.max_range.set_value(st.max_range)
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

    def refresh(self):
        ctl = self.ctl
        pr = ctl.result
        win = self.ctx.window
        self.b_run.setEnabled(not (ctl.busy and win is not None and 'perception' in win._jobs))
        self.bar.setVisible(ctl.busy)
        self.b_export.setEnabled(pr is not None)
        self.status.setText(ctl.status + (' · stale: the design or the settings changed' if ctl.stale and pr else ''))
        if pr is self._shown:
            return
        self._shown = pr
        self.map.set_result(pr)
        self.roses.set_result(pr)
        if pr is None:
            self.verdicts.setRowCount(0)
            self.pairs.setRowCount(0)
            return
        opts = [(str(i), c.name) for i, c in enumerate(pr.classes)]
        if [(k, b.text()) for k, b in self.pick.buttons.items()] != opts:
            cur = self.pick.value()
            old = self.pick
            self.pick = SegmentedControl(opts, cur if cur in dict(opts) else '0')
            self.pick.changed.connect(lambda k: self.map.set_class(int(k)))
            self._head.replaceWidget(old, self.pick)
            old.hide()
            old.setParent(None)
            old.deleteLater()
        self.map.set_class(int(self.pick.value() or 0))
        # verdicts: classes x sectors
        v = self.verdicts
        v.setRowCount(len(pr.classes))
        v.setHorizontalHeaderLabels(['Object'] + [f'{SHORT.get(s["name"], s["name"])} {s["required_m"]:g}'
                                                  for s in pr.sectors])
        hh = v.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for c in range(1, len(pr.sectors) + 1):
            hh.setSectionResizeMode(c, QHeaderView.ResizeMode.ResizeToContents)
        for r, cl in enumerate(pr.classes):
            it = QTableWidgetItem(cl.name)
            it.setForeground(class_color(r))
            v.setItem(r, 0, it)
            for j, s in enumerate(pr.sectors, 1):
                d = s['classes'][cl.name]
                it = QTableWidgetItem(f'{d["min_m"]:.0f}')
                it.setFont(mono(12))
                it.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                it.setForeground(QColor(STATUS['good' if d['pass'] else 'critical']))
                it.setToolTip(f'{cl.name}, {s["name"]}: resolved out to {d["min_m"]:.1f} m in every direction of the '
                              f'sector (median {d["median_m"]:.1f} m); required {s["required_m"]:g} m'
                              + ('' if d['pass'] else f' · {d["shortfall_m"]:.1f} m short'))
                v.setItem(r, j, it)
        v.setFixedHeight(ROW_H * (len(pr.classes) + 1) + 6)
        # stereo pairs
        t = self.pairs
        t.setRowCount(len(pr.pairs))
        slots = {uid: slot for uid, _, slot in pr.cams}
        for r, pp in enumerate(pr.pairs):
            a, b = pp['tags']
            it = QTableWidgetItem(f'{a} · {b}')
            it.setForeground(THEME.cam(slots.get(pp['a'], 0)))
            t.setItem(r, 0, it)
            vals = [f'{pp["overlap_deg"]:.0f}', f'{pp["baseline_m"]:.2f}'] + \
                   [(f'{pp["depth_error_m"][f"{z:g}"]:.2f}' if math.isfinite(pp['depth_error_m'][f'{z:g}']) else '—')
                    for z in PR.STEREO_AT]
            for j, val in enumerate(vals, 1):
                it = QTableWidgetItem(val)
                it.setFont(mono(12))
                it.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                t.setItem(r, j, it)

    # ------------------------------------------------------------------ export
    def export(self, folder=None):
        pr = self.ctl.result
        if pr is None:
            return []
        if folder is None:
            folder = QFileDialog.getExistingDirectory(self, 'Export the perception range into')
            if not folder:
                return []
        paths = [os.path.join(folder, 'perception_range.json'), os.path.join(folder, 'perception_range.png')]
        with open(paths[0], 'w', encoding='utf-8') as fh:
            fh.write(json.dumps(PR.to_doc(pr, self.ctx.store.project), indent=1) + '\n')
        with open(paths[1], 'wb') as fh:
            fh.write(PR.png(pr))
        self.ctx.msg(f'Exported the perception range to {folder}', 5000)
        return paths
