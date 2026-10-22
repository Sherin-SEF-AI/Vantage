"""Analysis ▸ Sync and timing: when each camera fires against the spinning lidar's sweep, which cameras share a
hardware trigger line, the phases of the other spinning lidars, and the motion blur and rolling-shutter skew at
driving speed. The plan is cheap: it is worked out again on every edit while the page shows."""
from __future__ import annotations

import math
import traceback

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import (QAbstractItemView, QComboBox, QFileDialog, QGridLayout, QHBoxLayout, QHeaderView,
                             QSplitter, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from ...core import timing as TM
from ...core.mathutil import parse_number
from ..theme import STATUS, THEME, mono
from ..timingview import TimingDial
from ..widgets import ROW_H, ScrubSpinBox, StatusLine, button, hline, label

# key, label, number kind, decimals, unit, scrub step, tooltip
FIELDS = (('speed_kmh', 'Speed', 'plain', 0, 'km/h', 5.0, 'Driving speed: the motion blur, and how far the vehicle '
                                                             'travels during a misalignment'),
          ('exposure_ms', 'Exposure', 'plain', 2, 'ms', 0.1, 'Exposure time of each row'),
          ('readout_ms', 'Readout', 'plain', 1, 'ms', 1.0, 'Rolling-shutter readout from the first row to the last: '
                                                            'enter your sensor\'s (0 for a global shutter)'),
          ('object_m', 'Object at', 'length', 1, 'm', 0.5, 'Distance of the static object the blur is measured on'),
          ('blur_limit_px', 'Blur limit', 'plain', 2, 'px', 0.1, 'The most motion blur the dataset accepts'),
          ('trigger_lines', 'Trigger lines', 'plain', 0, '', 1.0, 'Hardware trigger outputs available for the '
                                                                  'cameras'))
CAM_COLS = (('Camera', 'The camera; its frame rate as a multiple of the lidar\'s in the tooltip'),
            ('Line', 'Trigger line (the camera\'s sync group): cameras on one line fire together. Type a letter to '
                     'move the camera to another line'),
            ('Az °', 'Where the lidar sees the camera\'s view: the azimuth, in the lidar\'s frame, of a point 20 m '
                     'along the camera\'s optical axis'),
            ('Cross', 'ms after the sync pulse at which the sweep crosses the camera\'s view'),
            ('Trigger', 'ms after the sync pulse at which the camera\'s line triggers the frame matched to the sweep'),
            ('Off', 'Misalignment, ms: the middle row\'s exposure centre minus the sweep crossing'),
            ('Travel', 'How far the vehicle moves during the misalignment, cm'),
            ('Blur', 'Worst motion blur over the image, px, for a static object at the set distance'),
            ('Max exp', 'Longest exposure, ms, that keeps the worst blur within the limit'),
            ('Skew', 'Rolling-shutter skew between the first and the last row, px, worst over the image'))
LIDAR_COLS = (('Lidar', 'The spinning lidars; the first is the reference the cameras follow'),
              ('Hz', 'Rotation rate'), ('Turns', 'Rotation seen from the lidar\'s top: cw or ccw'),
              ('Phase °', 'The azimuth (its own frame) the lidar faces at the top of each sync period. Type to change'),
              ('Suggested °', 'The phase that makes it face the same direction as the reference at the same time'),
              ('Apart', 'ms between the reference and this lidar sweeping the same direction now (worst direction)'),
              ('After', 'The same with the suggested phase'))
GOOD_CM, WARN_CM = 5.0, 20.0         # travel during a misalignment: green up to this, amber up to that, then orange


def _table(cols):
    t = QTableWidget(0, len(cols))
    t.verticalHeader().setVisible(False)
    t.verticalHeader().setDefaultSectionSize(ROW_H)
    t.horizontalHeader().setFixedHeight(ROW_H)
    t.setShowGrid(False)
    t.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
    for c, (head, tip) in enumerate(cols):
        it = QTableWidgetItem(head)
        it.setToolTip(tip)
        t.setHorizontalHeaderItem(c, it)
    hh = t.horizontalHeader()
    hh.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
    for c in range(1, len(cols)):
        hh.setSectionResizeMode(c, QHeaderView.ResizeMode.ResizeToContents)
    return t


def _num(text, tip='', color=None, editable=False, bold=False):
    it = QTableWidgetItem(text)
    it.setFont(mono(12, QFont.Weight.DemiBold if bold else None))
    it.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    if tip:
        it.setToolTip(tip)
    if color is not None:
        it.setForeground(color)
    if not editable:
        it.setFlags(it.flags() & ~Qt.ItemFlag.ItemIsEditable)
    return it


def _f(v, dp=1):
    return '—' if v is None or not math.isfinite(v) else f'{v:.{dp}f}'


class TimingPage(QWidget):
    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        self.res: TM.TimingResult | None = None
        self.proposing = False             # the suggested trigger letters are shown in the table
        self._dirty = True
        self._syncing = False
        self.setObjectName('base')
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        split = QSplitter(Qt.Orientation.Horizontal)
        split.setHandleWidth(2)
        split.addWidget(self._left())
        # ---------------- centre: the dial
        cw = QWidget()
        cw.setObjectName('panel')
        cl = QVBoxLayout(cw)
        cl.setContentsMargins(8, 4, 8, 6)
        cl.setSpacing(4)
        self.head = label('', ink2=True, wrap=True)
        cl.addWidget(self.head)
        self.dial = TimingDial()
        cl.addWidget(self.dial, 1)
        split.addWidget(cw)
        # ---------------- right: cameras, groups, lidars, what needs attention
        right = QSplitter(Qt.Orientation.Vertical)
        right.setHandleWidth(2)
        tw = QWidget()
        tw.setObjectName('panel')
        tl = QVBoxLayout(tw)
        tl.setContentsMargins(8, 6, 8, 6)
        tl.setSpacing(4)
        tl.addWidget(label('Cameras against the sweep', bold=True))
        self.table = _table(CAM_COLS)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.DoubleClicked |
                                   QAbstractItemView.EditTrigger.EditKeyPressed)
        self.table.itemChanged.connect(self._cam_edited)
        tl.addWidget(self.table, 1)
        row = QHBoxLayout()
        self.b_suggest = button('Suggest groups', None, 'sync', 'Split the cameras over the trigger lines so the worst '
                                                               'misalignment is smallest')
        self.b_suggest.clicked.connect(lambda: self.set_proposing(not self.proposing))
        self.b_apply = button('Apply', 'primary', None, 'Set the cameras\' sync groups to the suggested lines (one '
                                                        'undo step)')
        self.b_apply.clicked.connect(self.apply_groups)
        row.addWidget(self.b_suggest)
        row.addStretch(1)
        row.addWidget(self.b_apply)
        tl.addLayout(row)
        self.proposal = label('', small=True, wrap=True)
        tl.addWidget(self.proposal)
        right.addWidget(tw)
        lw = QWidget()
        lw.setObjectName('panel')
        ll = QVBoxLayout(lw)
        ll.setContentsMargins(8, 6, 8, 6)
        ll.setSpacing(4)
        ll.addWidget(label('Spinning lidars', bold=True))
        self.lidars = _table(LIDAR_COLS)
        self.lidars.setEditTriggers(QAbstractItemView.EditTrigger.DoubleClicked |
                                    QAbstractItemView.EditTrigger.EditKeyPressed)
        self.lidars.itemChanged.connect(self._phase_edited)
        ll.addWidget(self.lidars)
        row = QHBoxLayout()
        row.addStretch(1)
        self.b_phases = button('Apply phases', None, 'check', 'Set the other lidars to the suggested phases (one undo '
                                                              'step)')
        self.b_phases.clicked.connect(self.apply_phases)
        row.addWidget(self.b_phases)
        ll.addLayout(row)
        ll.addWidget(hline())
        ll.addWidget(label('To check', bold=True))
        self.notes_box = QVBoxLayout()
        self.notes_box.setSpacing(3)
        ll.addLayout(self.notes_box)
        ll.addStretch(1)
        right.addWidget(lw)
        right.setSizes([520, 380])
        split.addWidget(right)
        split.setSizes([300, 680, 720])
        lay.addWidget(split)
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(30)
        self.timer.timeout.connect(self.recompute)
        ctx.store.changed.connect(self._changed)
        ctx.store.project_replaced.connect(self._replaced)
        self.recompute()

    # ------------------------------------------------------------------ settings panel
    def _left(self):
        w = QWidget()
        w.setObjectName('panel')
        w.setMinimumWidth(280)
        ll = QVBoxLayout(w)
        ll.setContentsMargins(8, 6, 8, 8)
        ll.setSpacing(4)
        top = QHBoxLayout()
        top.addWidget(label('Sync and timing'))
        top.addStretch(1)
        top.addWidget(label('cameras against the lidar', muted=True, small=True))
        ll.addLayout(top)
        ll.addWidget(hline())
        ll.addWidget(label('Every camera should expose its image while the spinning lidar sweeps across its view. '
                           'Cameras on one trigger line fire together; a rolling shutter reads its rows over the '
                           'readout, so the middle row is aimed at the sweep. Enter your sensor\'s readout and '
                           'exposure.', muted=True, small=True, wrap=True))
        ll.addSpacing(4)
        grid = QGridLayout()
        grid.setVerticalSpacing(2)
        grid.setHorizontalSpacing(6)
        grid.setColumnStretch(2, 1)
        a = label('Reference', ink2=True)
        a.setFixedWidth(84)
        grid.addWidget(a, 0, 0)
        self.ref = QComboBox()
        self.ref.setToolTip('The spinning lidar the cameras follow')
        self.ref.activated.connect(lambda i: self.set_option('lidar_uid', self.ref.itemData(i) or ''))
        grid.addWidget(self.ref, 0, 1, 1, 2)
        self.fields = {}
        for i, (key, text, kind, dec, unit, step, tip) in enumerate(FIELDS, 1):
            a = label(text, ink2=True)
            a.setFixedWidth(84)
            f = ScrubSpinBox('', kind, dec, unit, step=step, width=96)
            f.setToolTip(tip)
            f.committed.connect(lambda v, k=key: self.set_option(k, v))
            grid.addWidget(a, i, 0)
            grid.addWidget(f, i, 1)
            self.fields[key] = f
        ll.addLayout(grid)
        ll.addSpacing(6)
        self.figures = label('', small=True, wrap=True, ink2=True)
        ll.addWidget(self.figures)
        ll.addStretch(1)
        self.status = label('', muted=True, small=True, wrap=True)
        ll.addWidget(self.status)
        bl = QHBoxLayout()
        bl.addStretch(1)
        self.b_export = button('Export…', None, 'export', 'The plan as JSON (in the project\'s frame and units) and '
                                                          'the dial as PNG')
        self.b_export.clicked.connect(lambda: self.export())
        bl.addWidget(self.b_export)
        ll.addLayout(bl)
        return w

    @property
    def settings(self) -> TM.TimingSettings:
        return TM.TimingSettings.from_dict(self.ctx.store.project.analysis.timing)

    def set_option(self, key, value) -> bool:
        """Change one setting, saved with the project as one undo step; False when nothing changes."""
        st = self.settings
        old = st.to_dict()
        setattr(st, key, value)
        new = TM.TimingSettings.from_dict(st.to_dict()).to_dict()
        if new == old:
            self._sync_fields()                # a refused or rounded value: show what is stored
            return False
        name = next((f[1].lower() for f in FIELDS if f[0] == key), 'reference lidar')
        shown = new[key] if key != 'lidar_uid' else (self.ref.currentText() if value else 'first spinning lidar')

        def fn(p):
            p.analysis.timing = new
        self.ctx.store.edit('analysis', fn, f'Sync and timing · {name} {shown:g}' if isinstance(shown, float)
                            else f'Sync and timing · {name} {shown}')
        return True

    def _sync_fields(self):
        st = self.settings
        self._syncing = True
        try:
            for key, f in self.fields.items():
                f.set_value(float(getattr(st, key)))
            p = self.ctx.store.project
            opts = [('First spinning lidar', '')] + [
                (f'{s.tag} · {p.sensor_model_for(s).name}', s.uid) for s in p.variant.sensors
                if s.visible and s.kind == 'lidar' and p.sensor_model_for(s).spinning]
            if [(self.ref.itemText(i), self.ref.itemData(i)) for i in range(self.ref.count())] != opts:
                self.ref.blockSignals(True)
                self.ref.clear()
                for text, uid in opts:
                    self.ref.addItem(text, uid)
                self.ref.blockSignals(False)
            i = next((i for i, (_, u) in enumerate(opts) if u == st.lidar_uid), 0)
            if self.ref.currentIndex() != i:
                self.ref.setCurrentIndex(i)
        finally:
            self._syncing = False

    # ------------------------------------------------------------------ state
    def _changed(self, *_):
        self._dirty = True
        if self.isVisible():
            self.timer.start()

    def _replaced(self, *_):
        self.proposing = False
        self._changed()

    def showEvent(self, e):
        super().showEvent(e)
        if self._dirty:
            self.timer.start()

    def activate(self):
        if self._dirty:
            self.recompute()

    def recompute(self):
        self.timer.stop()
        self._dirty = False
        try:
            self.res = TM.plan(self.ctx.store.project)
            self.status.setText(f'Worked out in {self.res.metrics["seconds"] * 1000:.0f} ms')
        except Exception as e:                 # keep the last plan on the page; say what went wrong
            self.status.setText(f'Failed: {type(e).__name__}: {e}')
            traceback.print_exc()
        self._sync_fields()
        self.refresh()

    def set_proposing(self, on: bool):
        self.proposing = bool(on) and self.res is not None and self.res.lidar is not None
        self.refresh()

    # ------------------------------------------------------------------ the views
    def refresh(self):
        res = self.res
        self.dial.set_result(res)
        self.b_export.setEnabled(res is not None)
        if res is None:
            return
        st, m, lid = res.settings, res.metrics, res.lidar
        if lid is None:
            self.proposing = False
            self.head.setText(f'No sweep plan: motion blur and rolling-shutter skew of {len(res.cams)} '
                              f'camera{"s" if len(res.cams) != 1 else ""} only')
        else:
            w = m['worst_ms']
            self.head.setText(
                (f'Worst misalignment {w:.1f} ms ({m["worst_travel_cm"]:.0f} cm at {st.speed_kmh:g} km/h, '
                 f'{m["worst_tag"]})' if w is not None else 'No camera on a trigger line')
                + f' · {m["lines_used"]} of {st.trigger_lines} trigger lines · period {lid.period_ms:g} ms · '
                  f'{m["rotation_travel_m"]:.2f} m travelled per rotation (deskew)')
        mx = m['max_exposure_ms']
        self.figures.setText(f'Worst blur {m["worst_blur_px"]:.1f} px ({m["worst_blur_tag"] or "—"}) for an object '
                             f'{st.object_m:g} m away at {st.speed_kmh:g} km/h; exposures up to '
                             + (f'{mx:.2f} ms' if mx is not None else 'any length') +
                             f' keep every camera within {st.blur_limit_px:g} px.' if res.cams else 'No cameras.')
        self._fill_cameras()
        self._fill_lidars()
        self._fill_notes()

    def _fill_cameras(self):
        res = self.res
        sug = res.suggestion['letters'] if self.proposing else {}
        t = self.table
        self._syncing = True
        try:
            t.setRowCount(len(res.cams))
            for r, c in enumerate(res.cams):
                it = QTableWidgetItem(c['tag'])
                it.setForeground(THEME.cam(c['slot']))
                it.setFlags(it.flags() & ~Qt.ItemFlag.ItemIsEditable)
                it.setData(Qt.ItemDataRole.UserRole, c['uid'])
                tip = f'{c["name"]} · {c["fps"]:g} fps'
                if c.get('ratio') is not None:
                    tip += f', {c["ratio"]:.3g}× the lidar\'s rate' + (
                        ': the frames drift against the sweep' if c['alignment'] == 'drifts' else '')
                    if c['alignment'] == 'drifts':
                        it.setIcon(_warn_icon())
                it.setToolTip(tip)
                t.setItem(r, 0, it)
                now = c['sync'] or '—'
                new = sug.get(c['uid'])
                text = f'{now} → {new}' if new is not None and new != c['sync'] else now
                it = _num(text, editable=not self.proposing and c['hw_sync'], bold=new is not None and new != c['sync'])
                it.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                it.setData(Qt.ItemDataRole.UserRole, c['uid'])
                if not c['hw_sync']:
                    it.setToolTip('The module has no hardware trigger input')
                t.setItem(r, 1, it)
                tone = None
                if c.get('travel_cm') is not None:
                    tone = ('good' if c['travel_cm'] <= GOOD_CM else 'warning' if c['travel_cm'] <= WARN_CM
                            else 'serious')
                vals = [(_f(c.get('azimuth_deg'), 0), None), (_f(c.get('cross_ms')), None),
                        (_f(c.get('trigger_ms')), None),
                        ((f'{c["misalign_ms"]:+.1f}' if c.get('misalign_ms') is not None else '—'),
                         QColor(STATUS[tone]) if tone else None),
                        (_f(c.get('travel_cm'), 0), QColor(STATUS[tone]) if tone else None),
                        (_f(c['blur_worst_px'], 2),
                         QColor(STATUS['good' if c['blur_worst_px'] <= res.settings.blur_limit_px else 'critical'])),
                        (_f(c['max_exposure_ms'], 2), None), (_f(c['skew_px'], 0), None)]
                for j, (val, col) in enumerate(vals, 2):
                    t.setItem(r, j, _num(val, color=col))
                t.item(r, 7).setToolTip(f'{c["blur_worst_px"]:.2f} px at worst, {c["blur_centre_px"]:.2f} px at the '
                                        f'image centre')
                t.item(r, 9).setToolTip(f'{c["skew_px"]:.1f} px at worst, {c["skew_centre_px"]:.1f} px at the image '
                                        f'centre')
        finally:
            self._syncing = False
        # the suggestion
        lid = res.lidar
        self.b_suggest.setEnabled(lid is not None and bool(res.suggestion['letters']))
        self.b_suggest.setText('Hide suggestion' if self.proposing else 'Suggest groups')
        same = all(res.cam(u)['sync'] == L for u, L in res.suggestion['letters'].items())
        self.b_apply.setVisible(self.proposing)
        self.b_apply.setEnabled(self.proposing and not same)
        if not self.proposing or lid is None:
            self.proposal.setText('')
            return
        before, after = res.metrics['worst_ms'], res.suggestion['worst_ms']
        v = res.settings.speed_mps / 10.0           # cm per ms
        if same:
            self.proposal.setText(f'The cameras already use the best grouping for {res.settings.trigger_lines} '
                                  f'trigger lines: worst misalignment {after:.1f} ms.')
        else:
            self.proposal.setText(
                f'Suggested: {res.suggestion["lines"]} trigger line{"s" if res.suggestion["lines"] != 1 else ""}; '
                f'worst misalignment ' + (f'{before:.1f} → ' if before is not None else '') + f'{after:.1f} ms ('
                + (f'{before * v:.0f} → ' if before is not None else '') + f'{after * v:.0f} cm at '
                f'{res.settings.speed_kmh:g} km/h). Apply sets the cameras\' sync groups.')

    def _fill_lidars(self):
        res = self.res
        t = self.lidars
        lid = res.lidar
        rows = []
        if lid is not None:
            rows.append((lid.uid, f'{lid.tag} · reference', lid.rate_hz, lid.rotation, lid.phase_deg, None, None, None))
            rows += [(d['uid'], d['tag'], d['rate_hz'], d['rotation'], d['phase_deg'], d['suggested_phase_deg'],
                      d['misalign_ms'], d['after_ms']) for d in res.lidars]
        self._syncing = True
        try:
            t.setRowCount(len(rows))
            for r, (uid, tag, hz, turns, ph, sug, now, after) in enumerate(rows):
                it = QTableWidgetItem(tag)
                it.setFlags(it.flags() & ~Qt.ItemFlag.ItemIsEditable)
                it.setData(Qt.ItemDataRole.UserRole, uid)
                note = next((d['note'] for d in res.lidars if d['uid'] == uid), '')
                if note:
                    it.setToolTip(note)
                t.setItem(r, 0, it)
                t.setItem(r, 1, _num(f'{hz:g}'))
                t.setItem(r, 2, _num(turns))
                ph_it = _num(f'{ph:.1f}', editable=True)
                ph_it.setData(Qt.ItemDataRole.UserRole, uid)
                t.setItem(r, 3, ph_it)
                t.setItem(r, 4, _num(_f(sug)))
                t.setItem(r, 5, _num(_f(now)))
                t.setItem(r, 6, _num(_f(after)))
        finally:
            self._syncing = False
        t.setFixedHeight(ROW_H * (max(len(rows), 1) + 1) + 6)
        self.b_phases.setEnabled(any(abs(((d['suggested_phase_deg'] - d['phase_deg']) + 180) % 360 - 180) > 0.01
                                     for d in res.lidars))

    def _fill_notes(self):
        while self.notes_box.count():
            w = self.notes_box.takeAt(0).widget()
            if w is not None:
                w.hide()
                w.deleteLater()
        notes = list(self.res.notes)                # (without a lidar the dial says why there is no sweep plan)
        for n in notes or ['Nothing to flag.']:
            line = StatusLine('warning' if notes else 'good', n)
            line.text.setProperty('small', True)
            self.notes_box.addWidget(line)

    # ------------------------------------------------------------------ edits
    def _cam_edited(self, item):
        if self._syncing or item.column() != 1:
            return
        uid = item.data(Qt.ItemDataRole.UserRole)
        cam = self.ctx.store.cam(uid)
        letter = item.text().strip().upper()
        bad = bool(letter) and not (letter.isalnum() and len(letter) <= 2)
        if cam is None or bad or letter == (cam.sync or '').upper():
            QTimer.singleShot(0, self.refresh)     # nothing to change, or not a line name: show what is stored (later:
            return                                 # the item is still being committed)

        def fn(p):
            c = p.variant.cam(uid)
            if c is not None:
                c.sync = letter
        self.ctx.store.edit('variant', fn, f'{cam.tag} on trigger line {letter}' if letter else
                            f'{cam.tag} off the trigger lines')

    def _phase_edited(self, item):
        if self._syncing or item.column() != 3:
            return
        uid = item.data(Qt.ItemDataRole.UserRole)
        s = self.ctx.store.variant.sensor(uid)
        try:
            v = float(parse_number(item.text(), 'angle')) % 360.0
        except Exception:                      # parse_number raises on anything that is not an angle
            v = None
        if s is None or v is None or abs(((v - s.phase_deg) + 180) % 360 - 180) < 1e-6:
            QTimer.singleShot(0, self.refresh)
            return

        def fn(p):
            x = p.variant.sensor(uid)
            if x is not None:
                x.phase_deg = round(v, 3)
        self.ctx.store.edit('variant', fn, f'{s.tag} phase {v:.1f}°')

    def apply_groups(self) -> bool:
        """Set the cameras' sync letters to the suggestion, as one undo step."""
        res = self.res
        if res is None or not res.suggestion['letters']:
            return False
        letters = dict(res.suggestion['letters'])
        if all(res.cam(u)['sync'] == L for u, L in letters.items()):
            return False
        before, after = res.metrics['worst_ms'], res.suggestion['worst_ms']

        def fn(p):
            for c in p.variant.cameras:
                if c.uid in letters:
                    c.sync = letters[c.uid]
        self.proposing = False
        self.ctx.store.edit('variant', fn, f'Trigger groups from the timing plan · worst '
                            + (f'{before:.1f} → ' if before is not None else '') + f'{after:.1f} ms')
        return True

    def apply_phases(self) -> bool:
        """Set the other spinning lidars to the suggested phases, as one undo step."""
        if self.res is None or not self.res.lidars:
            return False
        phases = TM.suggested_phases(self.res)

        def fn(p):
            for s in p.variant.sensors:
                if s.uid in phases:
                    s.phase_deg = phases[s.uid]
        self.ctx.store.edit('variant', fn, f'Lidar phases from the timing plan · {self.res.lidar.tag} the reference')
        return True

    # ------------------------------------------------------------------ export
    def export(self, folder=None):
        if self.res is None:
            return []
        if folder is None:
            folder = QFileDialog.getExistingDirectory(self, 'Export the sync and timing plan into')
            if not folder:
                return []
        paths = TM.export(self.res, self.ctx.store.project, folder)
        self.ctx.msg(f'Exported the sync and timing plan to {folder}', 5000)
        return paths


def _warn_icon():
    from .. import icons
    return icons.status_icon('warning', 12)
