"""Analysis ▸ SOTIF matrix: the ISO 21448 triggering conditions against the rig's functional insufficiencies, measured
from the app's own studies (core.sotif), and the ODD's speed bound.

Left: the ODD (target speed, the stop model, the sensor basis, the critical classes, the stop margin and the lane
corridor) and the triggering conditions it includes with their road surface. Right: the ODD speed limit and what binds
it, the matrix (one row per condition; per critical class the detection range ahead against the stopping distance,
the child hidden ahead, the sensors that lose their view; each cell's tooltip names the study that produced it), the
ODD table, the known/unknown scenario areas, the requirements it traces to with the proposals (added to the scorecard
as one undo step) and the findings. The study takes a few seconds: it runs in a background thread after an edit while
the page shows. Supports a safety case; not a certification (the page says so)."""
from __future__ import annotations

import threading
import traceback

from PyQt6.QtCore import QObject, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (QAbstractItemView, QComboBox, QFileDialog, QHBoxLayout, QHeaderView, QProgressBar,
                             QScrollArea, QSplitter, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from ...core import sotif as SO
from ..theme import STATUS, THEME
from ..widgets import (ROW_H, ScrubSpinBox, Section, SegmentedControl, StatusLine, ToggleSwitch, button, fit_width,
                       hline, label)
from .recording_ws import Tile

AUTO_DELAY_MS = 500
FIELDS = (('target_kmh', 'Target speed', 0, 'km/h', 5.0, 'The ODD\'s target speed: every cell\'s verdict is at it'),
          ('margin_m', 'Stop margin', 1, 'm', 0.5, 'Subtracted from every range (2 m: R157 Annex 3 Table 3\'s d1)'),
          ('lane_half_m', 'Lane half width', 2, 'm', 0.05, 'The ego lane corridor ahead: this far each side of the '
                                                         'centre line'))
CUSTOM = (('perception_s', 'Perception', 2, 's', 0.05), ('reaction_s', 'Reaction', 2, 's', 0.05),
          ('decel_g', 'Deceleration', 3, 'g', 0.01), ('jerk_ms3', 'Jerk', 1, 'm/s³', 0.5))
MORE = (('sun_min_share', 'Sun share above', 1, '%', 0.5, 100.0, 'A camera with the low sun in its image more than '
                                                                  'this share of driving time is blinded'),
        ('dirt_min_score', 'Dirt score from', 2, '', 0.05, 1.0, 'core.exposure\'s dirt and spray score from which a '
                                                                'lens counts as obscured (0.5: high)'),
        ('fov_max_share', 'View lost above', 0, '%', 1.0, 100.0, 'A sensor losing more of its view than this is a '
                                                                 'field-of-view insufficiency'),
        ('ae_ev_per_s', 'AE adapts at', 1, 'EV/s', 0.5, 1.0, 'Auto-exposure adaptation rate at a tunnel exit '
                                                             '(assumed)'))
LABEL = {k: t for k, t, *_ in FIELDS + CUSTOM + MORE}
LABEL.update(stop_model='stop model', basis='basis', classes='critical classes')
VERDICT_SEV = {'ok': 'good', 'insufficient': 'critical', 'not_evaluated': 'warning'}


class _Sig(QObject):
    done = pyqtSignal(int, object)
    failed = pyqtSignal(int, str)
    progress = pyqtSignal(int, float)


def _table(cols, stretch=0):
    t = QTableWidget(0, len(cols))
    t.verticalHeader().setVisible(False)
    t.verticalHeader().setDefaultSectionSize(ROW_H)
    t.horizontalHeader().setFixedHeight(ROW_H)
    t.setShowGrid(False)
    t.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    t.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    for c, (head, tip) in enumerate(cols):
        it = QTableWidgetItem(head)
        it.setToolTip(tip)
        t.setHorizontalHeaderItem(c, it)
    hh = t.horizontalHeader()
    for c in range(len(cols)):
        hh.setSectionResizeMode(c, QHeaderView.ResizeMode.Stretch if c == stretch else
                                QHeaderView.ResizeMode.ResizeToContents)
    return t


def _cell_tip(c) -> str:
    src = '; '.join(f'{s["study"]} ({s["module"]}.{s["function"]}): {s["figure"]}' for s in c.get('sources') or [])
    bits = [SO.INSUFFICIENCY_LABEL[c['insufficiency']]]
    if c['insufficiency'] == 'range' and c.get('value') is not None:
        bits.append(f'{c["class"]}: {c["value"]:.1f} m ahead of the bumper; needed {c["threshold"]:.1f} m; safe up '
                    f'to {c["max_kmh"]:.0f} km/h')
        bt = c.get('by_type_m') or {}
        bits.append('By sensor type: ' + ', '.join(f'{k} {"—" if v is None else f"{v:.0f} m"}' for k, v in bt.items()))
    if c.get('lost'):
        bits.append(f'Without {c["lost"]}')
    if c.get('reason'):
        bits.append(c['reason'])
    if c.get('note'):
        bits.append(c['note'])
    if src:
        bits.append('Source: ' + src)
    return '\n'.join(bits)


def _cell_text(c) -> str:
    if c['verdict'] == 'not_evaluated' or c.get('value') is None:
        return '—'
    if c['insufficiency'] == 'range':
        return f'{c["value"]:.0f} / {c["threshold"]:.0f} m'
    if c['insufficiency'] == 'child':
        return f'{c["value"]:.2f} m'
    return f'{c["sensor"]} {100 * c["value"]:.0f} %'


class SotifPage(QWidget):
    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        self.result: SO.SotifResult | None = None
        self.busy = False
        self.stale = False
        self._gen = 0
        self._for = None
        self._syncing = False
        self.status_text = ''
        self.findings: list[StatusLine] = []
        self.sig = _Sig()
        self.sig.done.connect(self._done)
        self.sig.failed.connect(self._failed)
        self.sig.progress.connect(lambda g, f: self.bar.setValue(int(f * 1000)) if g == self._gen else None)
        self.setObjectName('base')
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        split = QSplitter(Qt.Orientation.Horizontal)
        split.setHandleWidth(2)
        split.addWidget(self._left())
        split.addWidget(self._right())
        split.setSizes([340, 1260])
        lay.addWidget(split)
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(AUTO_DELAY_MS)
        self.timer.timeout.connect(lambda: self.run(quiet=True))
        ctx.store.changed.connect(self._on_store)
        ctx.store.project_replaced.connect(self._replaced)
        THEME.changed.connect(self._theme_changed)
        self._sync_fields()
        self.refresh()

    def _theme_changed(self, *_):
        self.refresh()

    # ------------------------------------------------------------------ panels
    def _left(self):
        outer = QWidget()
        outer.setObjectName('panel')
        outer.setMinimumWidth(300)
        ol = QVBoxLayout(outer)
        ol.setContentsMargins(0, 0, 0, 0)
        ol.setSpacing(0)
        head = QWidget()
        hl = QVBoxLayout(head)
        hl.setContentsMargins(8, 6, 8, 4)
        hl.addWidget(label('SOTIF matrix'))
        hl.addWidget(hline())
        hl.addWidget(label('ISO 21448: which triggering conditions make the rig\'s perception insufficient, measured '
                           'from the app\'s studies, and the highest speed the rig supports in each.',
                           muted=True, small=True, wrap=True))
        ol.addWidget(head)
        body = QWidget()
        body.setObjectName('panel')
        bl = QVBoxLayout(body)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.setSpacing(0)
        self.fields, self.segs = {}, {}
        s = Section('Operational design domain')
        for key, text, dec, unit, step, tip in FIELDS:
            f = ScrubSpinBox('', 'plain', dec, unit, step=step, width=90)
            f.setToolTip(tip)
            f.committed.connect(lambda v, k=key: self.set_option(k, v))
            self.fields[key] = (f, 1.0)
            s.row(text, f, tip=tip)
        seg = SegmentedControl([('r157', 'R157'), ('r157_fsm', 'ALKS'), ('aashto', 'AASHTO'), ('custom', 'Custom')])
        seg.setToolTip('R157: UN R157 Annex 3 careful driver (0.4 + 0.75 s, 0.774 g in 0.6 s); ALKS: its fuzzy safety '
                       'model (0.75 s, 6 m/s², 12.65 m/s³); AASHTO: design sight distance (2.5 s, 3.4 m/s²); Custom: '
                       'the figures below')
        seg.changed.connect(lambda k: self.set_option('stop_model', k))
        self.segs['stop_model'] = seg
        s.row('Stop model', seg)
        for key, text, dec, unit, step in CUSTOM:
            f = ScrubSpinBox('', 'plain', dec, unit, step=step, width=90)
            f.setToolTip('The custom stop model\'s figure (given by you)')
            f.committed.connect(lambda v, k=key: self.set_option(k, v))
            self.fields[key] = (f, 1.0)
            s.row(text, f)
        seg = SegmentedControl([(b, SO.BASIS_LABEL[b].split()[0] if b != 'two' else 'Two') for b in SO.BASES])
        seg.setToolTip('Which sensors make the detection range: the second-best sensor type at each azimuth (two kinds '
                       'agree), the best one, or one kind alone')
        seg.changed.connect(lambda k: self.set_option('basis', k))
        self.segs['basis'] = seg
        s.row('Basis', seg)
        self.class_box = QWidget()
        self.class_lay = QHBoxLayout(self.class_box)
        self.class_lay.setContentsMargins(0, 0, 0, 0)
        self.class_toggles = {}
        s.row('Classes', self.class_box)
        self.model_note = label('', muted=True, small=True, wrap=True)
        s.add(self.model_note)
        bl.addWidget(s)
        s = Section('Triggering conditions')
        self.t_cond = _table((('Condition', 'In the ODD: its maximum safe speed bounds the ODD\'s'),
                              ('Road', 'The road surface: its adhesion caps the deceleration')))
        self.t_cond.setMinimumHeight(ROW_H * 12)
        s.add(self.t_cond)
        bl.addWidget(s)
        s = Section('Thresholds', expanded=False)
        for key, text, dec, unit, step, scale, tip in MORE:
            f = ScrubSpinBox('', 'plain', dec, unit, step=step, width=90)
            f.setToolTip(tip)
            f.committed.connect(lambda v, k=key, sc=scale: self.set_option(k, v / sc))
            self.fields[key] = (f, scale)
            s.row(text, f, tip=tip)
        bl.addWidget(s)
        bl.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setWidget(body)
        fit_width(scroll)
        ol.addWidget(scroll, 1)
        foot = QWidget()
        fl = QVBoxLayout(foot)
        fl.setContentsMargins(8, 4, 8, 8)
        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.setTextVisible(False)
        self.bar.setFixedHeight(4)
        fl.addWidget(self.bar)
        self.status = label('', muted=True, small=True, wrap=True)
        fl.addWidget(self.status)
        row = QHBoxLayout()
        self.b_run = button('Run', 'primary', 'play', 'Run the study now')
        self.b_run.clicked.connect(lambda: self.run())
        self.b_defaults = button('Defaults', 'ghost', None, 'Back to the default ODD and conditions')
        self.b_defaults.clicked.connect(self._defaults)
        self.b_export = button('Export…', None, 'export', 'The matrix and the ODD table as JSON (every cell with its '
                                                          'source and provenance), CSV, an HTML and a PDF report')
        self.b_export.clicked.connect(lambda: self.export())
        row.addWidget(self.b_run)
        row.addWidget(self.b_defaults)
        row.addStretch(1)
        row.addWidget(self.b_export)
        fl.addLayout(row)
        ol.addWidget(foot)
        return outer

    def _right(self):
        right = QSplitter(Qt.Orientation.Vertical)
        right.setHandleWidth(2)
        a = QWidget()
        a.setObjectName('panel')
        al = QVBoxLayout(a)
        al.setContentsMargins(10, 8, 10, 6)
        al.setSpacing(6)
        self.disclaimer = label(SO.DISCLAIMER, small=True, wrap=True)
        self.disclaimer.setStyleSheet(f'color:{STATUS["warning"]}')
        al.addWidget(self.disclaimer)
        tiles = QHBoxLayout()
        self.tiles = {}
        for key, caption, tip in (('odd', 'ODD speed limit', 'The lowest maximum safe speed over the conditions the '
                                                             'ODD includes'),
                                  ('binding', 'Bound by', 'The condition and class that set the limit'),
                                  ('bad', 'Insufficient cells', 'Cells of conditions in the ODD that fail at the '
                                                                'target speed'),
                                  ('ne', 'Not evaluated', 'Cells a study could not compute (area 3)')):
            self.tiles[key] = Tile(caption, tip)
            tiles.addWidget(self.tiles[key], 1)
        al.addLayout(tiles)
        al.addWidget(label('Insufficiency matrix', bold=True))
        self.t_mat = _table((('Condition', 'Triggering condition'),), stretch=0)
        al.addWidget(self.t_mat, 1)
        right.addWidget(a)
        b = QWidget()
        b.setObjectName('panel')
        blay = QVBoxLayout(b)
        blay.setContentsMargins(10, 6, 10, 6)
        blay.addWidget(label('ODD table', bold=True))
        self.t_odd = _table((('Condition', ''), ('In ODD', ''), ('Road (μ)', 'Peak adhesion (Wong, Table 1.3)'),
                             ('Decel.', 'Full deceleration, m/s²'), ('Delay', 'Perception + reaction + blind time'),
                             ('Bound by', 'The class with the shortest range'), ('Range', 'Corridor range ahead'),
                             ('Max speed', 'The largest speed whose stopping distance fits')))
        blay.addWidget(self.t_odd, 1)
        right.addWidget(b)
        c = QWidget()
        c.setObjectName('panel')
        cl = QVBoxLayout(c)
        cl.setContentsMargins(10, 6, 10, 8)
        cl.setSpacing(4)
        cl.addWidget(label('Known and unknown scenarios', bold=True))
        self.areas = label('', small=True, wrap=True)
        self.areas.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        cl.addWidget(self.areas)
        rr = QHBoxLayout()
        rr.addWidget(label('Requirements', bold=True))
        rr.addStretch(1)
        self.b_props = button('Add the proposals', None, 'plus', 'Add the proposed requirements to the scorecard '
                                                                 '(one undo step)')
        self.b_props.clicked.connect(self.add_proposals)
        rr.addWidget(self.b_props)
        cl.addLayout(rr)
        self.req = label('', small=True, wrap=True)
        self.req.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        cl.addWidget(self.req)
        cl.addWidget(label('Findings', bold=True))
        self.find_host = QWidget()
        self.find_lay = QVBoxLayout(self.find_host)
        self.find_lay.setContentsMargins(0, 0, 0, 0)
        self.find_lay.setSpacing(4)
        self.find_lay.addStretch(1)
        sc = QScrollArea()
        sc.setWidgetResizable(True)
        sc.setWidget(self.find_host)
        cl.addWidget(sc, 1)
        right.addWidget(c)
        right.setSizes([480, 220, 320])
        return right

    def _engine(self):
        win = getattr(self.ctx, 'window', None)
        try:
            if win is not None and hasattr(win, '_shared_engine'):
                return win._shared_engine()
            return self.ctx.analysis.engine.share()
        except Exception:                                        # noqa: BLE001 — the study makes its own
            return None

    # ------------------------------------------------------------------ settings
    @property
    def settings(self) -> SO.SotifSettings:
        return SO.settings_of(self.ctx.store.project)

    def _commit(self, new, text) -> bool:
        if new == self.settings.to_dict():
            self._sync_fields()
            return False

        def fn(p):
            p.analysis.sotif = new
        self.ctx.store.edit('analysis', fn, f'SOTIF · {text}')
        return True

    def set_option(self, key, value) -> bool:
        if self._syncing:
            return False
        d = self.settings.to_dict()
        d[key] = value
        new = SO.SotifSettings.from_dict(d).to_dict()
        shown = new.get(key)
        shown = f'{shown:g}' if isinstance(shown, float) else shown
        return self._commit(new, f'{LABEL.get(key, key)} {shown}')

    def set_in_odd(self, tid, on) -> bool:
        if self._syncing:
            return False
        d = self.settings.to_dict()
        odd = [t for t in d['odd'] if t != tid] + ([tid] if on else [])
        d['odd'] = odd
        new = SO.SotifSettings.from_dict(d).to_dict()
        return self._commit(new, f'{SO.TRIGGER[tid][1]} {"in" if on else "out of"} the ODD')

    def set_road(self, tid, road) -> bool:
        if self._syncing:
            return False
        d = self.settings.to_dict()
        d['roads'] = dict(d['roads'], **{tid: road})
        new = SO.SotifSettings.from_dict(d).to_dict()
        return self._commit(new, f'{SO.TRIGGER[tid][1]}: {SO.ROAD_LABEL[road].lower()} road')

    def set_class(self, name, on) -> bool:
        if self._syncing:
            return False
        d = self.settings.to_dict()
        cls = [c for c in d['classes'] if c != name] + ([name] if on else [])
        if not cls:
            self._sync_fields()
            return False
        d['classes'] = cls
        return self._commit(SO.SotifSettings.from_dict(d).to_dict(), f'{name} {"critical" if on else "not critical"}')

    def _defaults(self):
        if self.ctx.store.project.analysis.sotif:
            def fn(p):
                p.analysis.sotif = {}
            self.ctx.store.edit('analysis', fn, 'SOTIF · defaults')

    def _class_names(self):
        from ...core import perception as PR
        st = PR.PerceptionSettings.from_dict(getattr(self.ctx.store.project.analysis, 'perception', None))
        return [c['name'] for c in st.classes]

    def _sync_fields(self):
        st = self.settings
        self._syncing = True
        try:
            for k, (f, scale) in self.fields.items():
                f.set_value(float(getattr(st, k)) * scale)
                if k in dict((c[0], 1) for c in CUSTOM):
                    f.setEnabled(st.stop_model == 'custom')
            for k, sg in self.segs.items():
                sg.set_value(getattr(st, k))
            m = st.model()
            self.model_note.setText(m['source'])
            names = self._class_names()
            if list(self.class_toggles) != names:
                for w in self.class_toggles.values():
                    self.class_lay.removeWidget(w)
                    w.hide()
                    w.deleteLater()
                self.class_toggles = {}
                for n in names:
                    t = ToggleSwitch(n)
                    t.setToolTip(f'{n}s are a critical class: the ODD bound needs them detected within the stopping '
                                 'distance')
                    t.clicked.connect(lambda on, k=n: self.set_class(k, on))
                    self.class_lay.addWidget(t)
                    self.class_toggles[n] = t
            for n, t in self.class_toggles.items():
                t.blockSignals(True)
                t.setChecked(n in st.classes)
                t.blockSignals(False)
            self._fill_conditions(st)
        finally:
            self._syncing = False

    def _fill_conditions(self, st):
        ids = [t for t in SO.TRIGGER_IDS if t not in SO.OPTIONAL or self._has_optional(t)]
        tb = self.t_cond
        if tb.rowCount() != len(ids):
            tb.setRowCount(len(ids))
            for i, tid in enumerate(ids):
                cb = ToggleSwitch(SO.TRIGGER[tid][1])
                cb.setToolTip(SO.TRIGGER[tid][5])
                cb.clicked.connect(lambda on, k=tid: self.set_in_odd(k, on))
                tb.setCellWidget(i, 0, cb)
                combo = QComboBox()
                for r in SO.ROADS:
                    combo.addItem(SO.ROAD_LABEL[r], r)
                combo.currentIndexChanged.connect(lambda _i, k=tid, c=combo: self.set_road(k, c.currentData()))
                tb.setCellWidget(i, 1, combo)
            self._cond_ids = ids
        for i, tid in enumerate(self._cond_ids):
            cb, combo = tb.cellWidget(i, 0), tb.cellWidget(i, 1)
            cb.blockSignals(True)
            cb.setChecked(tid in st.odd)
            cb.blockSignals(False)
            combo.blockSignals(True)
            combo.setCurrentIndex(SO.ROADS.index(st.road(tid)))
            combo.blockSignals(False)

    def _has_optional(self, tid) -> bool:
        p = self.ctx.store.project
        want = 'lwir' if tid == 'thermal_night' else 'event'
        return any(c.visible and getattr(p.module_for(c), 'modality', 'rgb') == want for c in p.variant.cameras)

    # ------------------------------------------------------------------ running
    def _key(self):
        try:
            return SO.cache_key(self.ctx.store.project)
        except Exception:                                        # noqa: BLE001
            return None

    def _on_store(self, *_):
        if self.isVisible():
            self._sync_fields()
        else:
            self._fields_dirty = True
        if self.result is not None or self.isVisible():
            if self._for != self._key():
                self.stale = True
                self.refresh_status()
                if self.isVisible():
                    self.timer.start()

    def _replaced(self, *_):
        self._gen += 1
        self.result, self._for, self.stale, self.busy = None, None, False, False
        self.t_cond.setRowCount(0)
        self._sync_fields()
        self.refresh()
        if self.isVisible():
            self.timer.start()

    def showEvent(self, e):
        super().showEvent(e)
        if getattr(self, '_fields_dirty', False):
            self._fields_dirty = False
            self._sync_fields()
        if self.result is None or self.stale:
            self.timer.start()

    def activate(self):
        if self.result is None or self.stale:
            self.run(quiet=True)

    def run(self, quiet=False) -> bool:
        project = self.ctx.store.project.snapshot()
        key = self._key()
        self._gen += 1
        gen = self._gen
        self.busy = True
        self.status_text = 'Running the studies and filling the matrix…'
        self.bar.setValue(0)
        self.refresh_status()
        eng = self._engine()

        def thread():
            try:
                res = SO.run(project, eng, progress=lambda f, *_: self.sig.progress.emit(gen, f),
                             cancel=lambda: gen != self._gen)
                self.sig.done.emit(gen, (key, res))
            except RuntimeError:
                pass
            except Exception as e:                               # noqa: BLE001
                try:
                    self.sig.failed.emit(gen, f'{type(e).__name__}: {e}\n{traceback.format_exc()}')
                except RuntimeError:
                    pass
        threading.Thread(target=thread, daemon=True).start()
        return True

    def _done(self, gen, out):
        if gen != self._gen:
            return
        self.busy = False
        key, res = out
        if res is None:
            self.status_text = 'Cancelled'
            self.refresh_status()
            return
        self.result, self._for = res, key
        self.stale = key != self._key()
        m = res.metrics
        self.status_text = f'{res.variant}: {m["rows"]} conditions · {m["cells"]} cells · {m["seconds"]:.1f} s'
        self.bar.setValue(1000)
        self.refresh()
        if self.stale and self.isVisible():
            self.timer.start()

    def _failed(self, gen, msg):
        if gen != self._gen:
            return
        self.busy = False
        self.status_text = 'Failed: ' + msg.splitlines()[0]
        self.refresh_status()

    # ------------------------------------------------------------------ views
    def refresh_status(self):
        txt = self.status_text
        if self.stale and not self.busy and self.result is not None:
            txt = (txt + ' · ' if txt else '') + 'out of date'
        self.status.setText(txt)
        self.b_export.setEnabled(self.result is not None)
        self.b_props.setEnabled(self.result is not None and bool(self.result.proposals))

    def refresh(self):
        self.refresh_status()
        res = self.result
        for w in self.findings:
            self.find_lay.removeWidget(w)
            w.hide()
            w.deleteLater()
        self.findings = []
        if res is None:
            for t in self.tiles.values():
                t.set('—', '', 'runs when the page shows')
            self.t_mat.setRowCount(0)
            self.t_odd.setRowCount(0)
            self.areas.setText('')
            self.req.setText('')
            return
        st, odd = res.settings, res.odd
        sev = None if odd['meets_target'] is None else ('good' if odd['meets_target'] else 'serious')
        self.tiles['odd'].set('—' if odd['speed_kmh'] is None else f'{odd["speed_kmh"]:.0f}', 'km/h',
                              f'target {st.target_kmh:g} km/h · {odd["model"]}', sev)
        self.tiles['binding'].set(odd['binding_class'] or '—', '', odd['binding_label'] or 'nothing binds')
        self.tiles['bad'].set(f'{res.metrics["insufficient"]}', 'cells',
                              f'of {res.metrics["cells"]} · {len(res.rows)} conditions',
                              'serious' if res.metrics['insufficient'] else 'good')
        self.tiles['ne'].set(f'{res.metrics["not_evaluated"]}', 'cells', 'what the studies cannot say',
                             'warning' if res.metrics['not_evaluated'] else None)
        self._fill_matrix(res)
        self._fill_odd(res)
        lines = []
        for a in res.areas:
            items = '; '.join(i['label'] for i in a['items'][:8]) + (' …' if len(a['items']) > 8 else '')
            lines.append(f'<b>Area {a["area"]}, {a["label"].lower()}</b> ({len(a["items"])}): '
                         f'{items or a["text"]}')
        self.areas.setText('<br>'.join(lines))
        tr = ('Traced: ' + '; '.join(f'{t["requirement"]} {t["text"]} ({len(t["cells"])} cells)' for t in res.trace)
              if res.trace else 'No scorecard requirement checks a figure of the matrix yet.')
        self.req.setText(tr + '<br>Proposed: ' + '; '.join(p['text'] for p in res.proposals) + '.')
        for f in res.findings:
            w = StatusLine(f['severity'], f['message'])
            self.find_lay.insertWidget(self.find_lay.count() - 1, w)
            self.findings.append(w)

    def _fill_matrix(self, res):
        cls = res.classes
        heads = [('Condition', 'Triggering condition (italic: outside the ODD)')] + \
            [(f'{c}', f'{c}: detection range ahead / needed at the target speed') for c in cls] + \
            [('Child', 'A child 0.9 m tall hidden ahead of the bumper'), ('View lost', 'Sensors losing their view'),
             ('Max', 'Maximum safe speed, km/h')]
        tb = self.t_mat
        tb.setColumnCount(len(heads))
        for i, (h, tip) in enumerate(heads):
            it = QTableWidgetItem(h)
            it.setToolTip(tip)
            tb.setHorizontalHeaderItem(i, it)
        hh = tb.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for i in range(1, len(heads)):
            hh.setSectionResizeMode(i, QHeaderView.ResizeMode.ResizeToContents)
        tb.setRowCount(len(res.rows))
        for r_i, r in enumerate(res.rows):
            it = QTableWidgetItem(r['label'])
            it.setToolTip(f'{r["note"]}\n{r["detail"]}'.strip())
            if not r['in_odd']:
                f = it.font()
                f.setItalic(True)
                it.setFont(f)
                it.setForeground(THEME.color('muted'))
            tb.setItem(r_i, 0, it)
            rng = {c['class']: c for c in r['cells'] if c['insufficiency'] == 'range'}
            for j, c in enumerate(cls, 1):
                cell = rng.get(c)
                tb.setItem(r_i, j, self._item(cell))
            child = next((c for c in r['cells'] if c['insufficiency'] == 'child'), None)
            tb.setItem(r_i, len(cls) + 1, self._item(child))
            fov = [c for c in r['cells'] if c['insufficiency'] == 'fov']
            bad = [c for c in fov if c['verdict'] == 'insufficient']
            it = QTableWidgetItem(', '.join(f'{c["sensor"]} {100 * c["value"]:.0f} %' for c in bad[:4])
                                  + (' …' if len(bad) > 4 else '') if bad else ('—' if not fov else 'none'))
            it.setToolTip('\n\n'.join(_cell_tip(c) for c in fov[:8]) or 'No sensor loses its view here.')
            if bad:
                it.setForeground(QColor(STATUS['warning']))
            tb.setItem(r_i, len(cls) + 2, it)
            mk = r.get('max_kmh')
            it = QTableWidgetItem('—' if mk is None else f'{mk:.0f}')
            it.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            if mk is not None and mk < res.settings.target_kmh:
                it.setForeground(QColor(STATUS['critical']))
            tb.setItem(r_i, len(cls) + 3, it)

    def _item(self, cell):
        if cell is None:
            it = QTableWidgetItem('')
            return it
        it = QTableWidgetItem(_cell_text(cell))
        it.setToolTip(_cell_tip(cell))
        it.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        sev = VERDICT_SEV.get(cell['verdict'])
        if sev:
            c = QColor(STATUS[sev])
            c.setAlphaF(0.28)
            it.setBackground(c)
        return it

    def _fill_odd(self, res):
        tb = self.t_odd
        tb.setRowCount(len(res.rows))
        for i, r in enumerate(res.rows):
            vals = [r['label'], 'yes' if r['in_odd'] else 'no', f'{SO.ROAD_LABEL[r["road"]]} ({SO.ROAD_MU[r["road"]]:g})',
                    f'{r.get("decel_ms2", 0):.2f}', f'{r.get("delay_s", 0):.2f} s', r.get('binding') or '—',
                    '—' if r.get('binding_range_m') is None else f'{r["binding_range_m"]:.1f} m',
                    '—' if r.get('max_kmh') is None else f'{r["max_kmh"]:.0f} km/h']
            for c, v in enumerate(vals):
                it = QTableWidgetItem(v)
                if c >= 3:
                    it.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                if c == 7 and r.get('max_kmh') is not None and r['max_kmh'] < res.settings.target_kmh and r['in_odd']:
                    it.setForeground(QColor(STATUS['critical']))
                tb.setItem(i, c, it)

    # ------------------------------------------------------------------ actions
    def add_proposals(self) -> int:
        """Add the proposed requirements the scorecard does not have yet (one undo step); returns how many."""
        res = self.result
        if res is None:
            return 0
        from ...core import requirements as RQ
        st = RQ.settings_of(self.ctx.store.project)
        have = {(i['metric'], json_key(i.get('params'))) for i in st.items}
        taken = [i['id'] for i in st.items]
        new = []
        for p_ in res.proposals:
            if (p_['metric'], json_key(p_['params'])) in have:
                continue
            it = RQ.clean_item(dict(p_, note='Proposed by the SOTIF matrix (ODD target speed).'), self.ctx.store.project,
                               taken)
            if it is not None:
                taken.append(it['id'])
                new.append(it)
        if not new:
            self.ctx.msg('The scorecard already has every proposed requirement', 4000)
            return 0
        d = st.to_dict()
        d['items'] = d['items'] + new

        def fn(p):
            p.analysis.requirements = d
        self.ctx.store.edit('analysis', fn, f'SOTIF · {len(new)} requirement{"s" * (len(new) != 1)} proposed')
        self.ctx.msg(f'Added {len(new)} requirement{"s" * (len(new) != 1)} to the scorecard', 4000)
        return len(new)

    def export(self, folder=None):
        res = self.result
        if res is None:
            return []
        if folder is None:
            folder = QFileDialog.getExistingDirectory(self, 'Export the SOTIF matrix into')
            if not folder:
                return []
        from ...core import provenance as PV
        paths = PV.stamp_paths(SO.export(res, self.ctx.store.project, folder), self.ctx.store.project, folder)
        self.ctx.msg(f'Exported the SOTIF matrix to {folder}', 5000)
        return paths


def json_key(d) -> str:
    import json
    return json.dumps({k: str(v).lower() for k, v in (d or {}).items()}, sort_keys=True)
