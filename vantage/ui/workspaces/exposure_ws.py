"""Analysis ▸ Sun glare and dirt: which cameras will have the sun in their image over a data-collection campaign
(where, when, at which headings), the flare risk just outside their field, the lidars' solar background, and how
exposed every lens, lidar window and radar fascia is to dirt, spray and rain (a heuristic), with cleaning advice.
The study runs as a background job (well under a second for the sample's data rig) and again by itself after edits
while the page shows."""
from __future__ import annotations

import datetime
import json
import threading
import traceback

import numpy as np
from PyQt6.QtCore import QObject, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (QAbstractItemView, QComboBox, QFileDialog, QGridLayout, QHBoxLayout, QHeaderView,
                             QLineEdit, QProgressBar, QScrollArea, QSplitter, QTableWidget, QTableWidgetItem,
                             QVBoxLayout, QWidget)

from ...core import exposure as X
from ..exposureview import DirtMap, HeadingRose, ImageTracks, SkyChart, SunHeatmap, item_color
from ..theme import STATUS, THEME, mono
from ..widgets import fit_width, ROW_H, ScrubSpinBox, SegmentedControl, StatusLine, button, hline, label

JOB = 'exposure'
AUTO_MAX_S = 3.0
# (group, key, label, unit, decimals, scrub step, tooltip)
FIELDS = (
    ('Campaign', 'lat', 'Latitude', '° N', 4, 0.1, 'Where the data is collected: degrees north (south negative)'),
    ('Campaign', 'lon', 'Longitude', '° E', 4, 0.1, 'Degrees east (west negative)'),
    ('Campaign', 'utc_offset_h', 'UTC offset', 'h', 2, 0.5,
     'Local time = UTC + this, all year (a time zone below follows daylight saving time instead)'),
    ('Campaign', 'hour_from', 'Driving from', 'h', 2, 0.25, 'Start of the daily driving window, local time'),
    ('Campaign', 'hour_to', 'Driving to', 'h', 2, 0.25,
     'End of the daily driving window, local time (earlier than the start: the window runs over midnight)'),
    ('Campaign', 'minutes', 'Time step', 'min', 0, 5.0, 'Time between samples through the day'),
    ('Campaign', 'days', 'Day step', 'd', 0, 1.0, 'Days between sampled days'),
    ('Glare', 'flare_margin_deg', 'Flare margin', '°', 1, 1.0,
     'The sun outside the image but this close to its edge counts as flare risk'),
    ('Glare', 'low_sun_deg', 'Low sun below', '°', 1, 1.0,
     'The sun in the image and lower than this above the horizon: low sun ahead, the worst case'),
)
TEXT_FIELDS = (('timezone', 'Time zone', 'fixed offset', 'An IANA time zone (Europe/Berlin, Asia/Kolkata) to follow '
                                                         'daylight saving time; empty: the fixed UTC offset'),
               ('start', 'From date', 'YYYY-MM-DD', 'First day of the campaign'),
               ('end', 'To date', 'YYYY-MM-DD', 'Last day of the campaign'))
PRESETS = (('Munich', 48.14, 11.58, 1.0, 'Europe/Berlin'), ('Chennai', 13.08, 80.27, 5.5, 'Asia/Kolkata'),
           ('Bengaluru', 12.97, 77.59, 5.5, 'Asia/Kolkata'), ('Detroit', 42.33, -83.05, -5.0, 'America/Detroit'),
           ('Phoenix', 33.45, -112.07, -7.0, 'America/Phoenix'),
           ('San Francisco', 37.77, -122.42, -8.0, 'America/Los_Angeles'), ('Tokyo', 35.68, 139.69, 9.0, 'Asia/Tokyo'),
           ('Shanghai', 31.23, 121.47, 8.0, 'Asia/Shanghai'), ('Dubai', 25.20, 55.27, 4.0, 'Asia/Dubai'),
           ('Sydney', -33.87, 151.21, 10.0, 'Australia/Sydney'), ('Oslo', 59.91, 10.75, 1.0, 'Europe/Oslo'))
TABLE_COLS = (('Sensor', 'The camera, lidar or radar'),
              ('Sun %', 'Share of all driving time with the sun in the image (a lidar: inside its field), not hidden '
                        'by the body'),
              ('Near %', 'The sun outside the image but within the flare margin of its edge: flare risk'),
              ('Low %', 'The sun in the image and low (below the set elevation): the worst case'),
              ('Worst months', 'Months with at least half the worst month\'s share'),
              ('Worst hours', 'Local hours with at least half the worst hour\'s share'),
              ('Heading', 'The headings (clockwise from north) on which the sun is most often in the image'),
              ('Dirt', 'Dirt and spray score 0–1 (a heuristic); hover for the factors'),
              ('Class', 'Low below 0.3, moderate below 0.5, high from 0.5'),
              ('Recessed', 'Tick when the lens or window is recessed or hooded: its dirt score is 0.6 ×'),
              ('Cleaning', 'What the score suggests'))
HEAD_KEYS = [('worst', 'Worst')] + [(str(h), X.compass(h)) for h in X.TRACK_HEADINGS]


def inputs_key(project) -> str:
    """Everything the study reads, as text: a result made from other inputs is stale."""
    m = project.vehicle.mesh
    v = project.variant
    return json.dumps({'mesh': [id(m), getattr(m, 'n_tris', 0), project.vehicle.occluder_target],
                       'wheels': project.vehicle.wheel_centres, 'variant': v.to_dict(),
                       'rig': [r.to_dict() for r in project.rig], 'exposure': project.analysis.exposure,
                       'library': project.section('library')}, sort_keys=True, default=str)


class _Sig(QObject):
    done = pyqtSignal(int, object)
    failed = pyqtSignal(int, str)
    progress = pyqtSignal(int, float)


def controller(ctx) -> 'ExposureController':
    c = getattr(ctx, 'exposure', None)
    if c is None:
        c = ctx.exposure = ExposureController(ctx)
    return c


class ExposureController(QObject):
    """The project's sun glare and dirt study: run on demand (Run: a main-window job with Cancel) and again by itself
    after edits while the page shows it and a run is quick."""
    changed = pyqtSignal()
    progress = pyqtSignal(float)

    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        self.result: X.ExposureResult | None = None
        self.stale = False
        self.busy = False
        self.status = ''
        self.active = False
        self._gen = 0
        self._for = None                 # (project, inputs key) of the result
        self.sig = _Sig()
        self.sig.done.connect(self._done)
        self.sig.failed.connect(self._failed)
        self.sig.progress.connect(lambda g, f: self.progress.emit(f) if g == self._gen else None)
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(300)
        self.timer.timeout.connect(lambda: self.run(quiet=True))
        ctx.store.changed.connect(self._on_edit)
        ctx.store.project_replaced.connect(self._reset)

    @property
    def settings(self) -> X.ExposureSettings:
        return X.ExposureSettings.from_dict(self.ctx.store.project.analysis.exposure)

    def store(self, st: X.ExposureSettings, text) -> bool:
        """Settings as one undo step (saved with the project); False when nothing changes."""
        new = X.ExposureSettings.from_dict(st.to_dict()).to_dict()
        if new == self.settings.to_dict():
            return False

        def fn(p):
            p.analysis.exposure = new
        self.ctx.store.edit('analysis', fn, f'Sun glare and dirt · {text}')
        return True

    def _quick(self):
        return self.result is None or self.result.metrics.get('seconds', 99) < AUTO_MAX_S

    def _on_edit(self, *_):
        if not (self.active or self.result is not None):
            return                         # never shown: nothing to keep up to date
        if self._for is None or inputs_key(self.ctx.store.project) != self._for[1]:
            if self.result is not None and not self.stale:
                self.stale = True
                self.changed.emit()
            if self.active and self._quick() and not self.busy:
                self.timer.start()

    def set_active(self, on: bool):
        self.active = on
        if on and (self.result is None or self.stale) and self._quick() and not self.busy:
            self.timer.start()

    def _reset(self, *_):
        self._gen += 1
        self.result, self._for, self.stale, self.busy, self.status = None, None, False, False, ''
        self.changed.emit()
        if self.active:
            self.timer.start()

    def run(self, quiet=False) -> bool:
        ctx = self.ctx
        p = ctx.store.project
        win = ctx.window
        if self.busy:
            if quiet or (win is not None and JOB in win._jobs):
                return False
            self._gen += 1
        project = p.snapshot()
        st = X.ExposureSettings.from_dict(project.analysis.exposure)
        eng = win._shared_engine() if win is not None else ctx.analysis.engine.share()
        self._gen += 1
        gen = self._gen
        self.busy = True
        self.status = 'Computing the sun glare and dirt exposure…'
        key = (p, inputs_key(p))

        def work(progress, cancel):
            return key, X.run(project, eng, st, progress=progress, cancel=cancel)
        if not quiet and win is not None and hasattr(win, 'run_job'):
            ok = win.run_job(JOB, 'Sun glare and dirt', work, lambda out: self._done(gen, out),
                             lambda m: self._failed(gen, m))
            self.changed.emit()
            return ok
        self.changed.emit()

        def thread():
            try:
                out = work(lambda f, *_: self.sig.progress.emit(gen, f), lambda: gen != self._gen)
                self.sig.done.emit(gen, out)
            except RuntimeError:            # the window closed while computing
                pass
            except Exception as e:
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
        key, res = out if out is not None else (None, None)
        if res is None or key[0] is not self.ctx.store.project:
            self.status = 'Cancelled' if res is None else ''
            self.changed.emit()
            return
        self.result, self._for = res, key
        self.stale = key[1] != inputs_key(self.ctx.store.project)
        m = res.metrics
        n = {k: sum(1 for i in res.items if i.kind == k) for k in ('camera', 'lidar', 'radar')}
        self.status = (', '.join(f'{v} {k}{"s" * (v != 1)}' for k, v in n.items() if v or k == 'camera')
                       + f' · {m["samples"] / 1000:,.0f} k samples ({m["days_sampled"]} days × '
                         f'{m["time_samples"] // max(m["days_sampled"], 1)} times × {m["headings"]} headings) · '
                         f'{m["seconds"]:.1f} s' + (f' · {m["note"]}' if m['note'] else ''))
        self.changed.emit()
        if self.stale and self.active and self._quick():
            self.timer.start()

    def _failed(self, gen, msg):
        if gen != self._gen:
            return
        self.busy = False
        self.status = 'Failed: ' + msg.splitlines()[0]
        self.changed.emit()


# =========================================================================== the page
def _num(text, tip='', color=None):
    it = QTableWidgetItem(text)
    it.setFont(mono(12))
    it.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    it.setFlags(it.flags() & ~Qt.ItemFlag.ItemIsEditable)
    if tip:
        it.setToolTip(tip)
    if color is not None:
        it.setForeground(color)
    return it


def _text(text, tip='', color=None):
    it = QTableWidgetItem(text)
    it.setFlags(it.flags() & ~Qt.ItemFlag.ItemIsEditable)
    if tip:
        it.setToolTip(tip)
    if color is not None:
        it.setForeground(color)
    return it


def _pct(v):
    return '—' if v is None else f'{100 * v:.1f}'


class ExposurePage(QWidget):
    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        self.ctl = controller(ctx)
        self.sel_uid = ''
        self._shown = None
        self._syncing = False
        self.setObjectName('base')
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        split = QSplitter(Qt.Orientation.Horizontal)
        split.setHandleWidth(2)
        split.addWidget(self._left())
        right = QSplitter(Qt.Orientation.Vertical)
        right.setHandleWidth(2)
        # ---------------- the table
        tw = QWidget()
        tw.setObjectName('panel')
        tl = QVBoxLayout(tw)
        tl.setContentsMargins(8, 6, 8, 6)
        tl.setSpacing(4)
        tl.addWidget(label('Every camera, lidar and radar', bold=True))
        t = self.table = QTableWidget(0, len(TABLE_COLS))
        t.verticalHeader().setVisible(False)
        t.verticalHeader().setDefaultSectionSize(ROW_H)
        t.horizontalHeader().setFixedHeight(ROW_H)
        t.setShowGrid(False)
        t.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        t.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        for c, (head, tip) in enumerate(TABLE_COLS):
            hi = QTableWidgetItem(head)
            hi.setToolTip(tip)
            t.setHorizontalHeaderItem(c, hi)
        hh = t.horizontalHeader()
        for c in range(len(TABLE_COLS)):
            hh.setSectionResizeMode(c, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(len(TABLE_COLS) - 1, QHeaderView.ResizeMode.Stretch)
        t.itemSelectionChanged.connect(self._row_picked)
        t.itemChanged.connect(self._item_changed)
        tl.addWidget(t, 1)
        right.addWidget(tw)
        # ---------------- the selected camera: heatmap, sky, tracks
        cw = QWidget()
        cw.setObjectName('panel')
        cl = QHBoxLayout(cw)
        cl.setContentsMargins(8, 6, 8, 6)
        cl.setSpacing(12)
        self.heat = SunHeatmap()
        self.sky = SkyChart()
        cl.addWidget(self.heat, 3)
        cl.addWidget(self.sky, 3)
        trk = QWidget()
        tv = QVBoxLayout(trk)
        tv.setContentsMargins(0, 0, 0, 0)
        tv.setSpacing(4)
        self.head = SegmentedControl(HEAD_KEYS, 'worst')
        self.head.setToolTip('The heading the sun\'s path is drawn for: the camera\'s worst, or a compass heading')
        self.head.changed.connect(lambda k: self.tracks.set_data(self._item(), k))
        tv.addWidget(self.head)
        self.tracks = ImageTracks()
        tv.addWidget(self.tracks, 1)
        cl.addWidget(trk, 4)
        right.addWidget(cw)
        # ---------------- dirt map and findings
        bottom = QSplitter(Qt.Orientation.Horizontal)
        bottom.setHandleWidth(2)
        dw = QWidget()
        dw.setObjectName('panel')
        dl = QVBoxLayout(dw)
        dl.setContentsMargins(8, 6, 8, 6)
        self.dirt = DirtMap()
        self.dirt.picked.connect(self.select)
        dl.addWidget(self.dirt)
        bottom.addWidget(dw)
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
        bottom.addWidget(fw)
        bottom.setSizes([760, 520])
        right.addWidget(bottom)
        right.setSizes([250, 330, 330])
        split.addWidget(right)
        split.setSizes([330, 1300])
        lay.addWidget(split)
        self.ctl.changed.connect(self.refresh)
        ctx.store.changed.connect(lambda *_: self._sync_fields())
        ctx.store.project_replaced.connect(self._sync_fields)
        THEME.changed.connect(self._retheme)
        self._sync_fields()
        self.refresh()

    # ------------------------------------------------------------------ settings panel
    def _left(self):
        w = QWidget()
        w.setObjectName('panel')
        ll = QVBoxLayout(w)
        ll.setContentsMargins(8, 6, 8, 8)
        ll.setSpacing(4)
        top = QHBoxLayout()
        top.addWidget(label('Sun glare and dirt'))
        top.addStretch(1)
        top.addWidget(label('every camera, a campaign', muted=True, small=True))
        ll.addLayout(top)
        ll.addWidget(hline())
        ll.addWidget(label('When the sun shines into each camera over a data-collection campaign (washed-out frames, '
                           'flare), from the sun\'s position at every sampled time and heading through the real lens '
                           'and the body; and how exposed each lens, lidar window and radar fascia is to dirt, spray '
                           'and rain. Clear sky, a flat horizon; the dirt score is a heuristic.',
                           muted=True, small=True, wrap=True))
        self.fields = {}
        self.texts = {}
        grid = None
        group = None

        def new_group(g):
            nonlocal grid
            if grid is not None:
                ll.addLayout(grid)
            ll.addSpacing(6)
            ll.addWidget(label(g, bold=True))
            grid = QGridLayout()
            grid.setVerticalSpacing(2)
            grid.setHorizontalSpacing(6)
            grid.setColumnStretch(2, 1)

        def row(text, wdg, tip):
            a = label(text, ink2=True)
            a.setFixedWidth(104)
            a.setToolTip(tip)
            wdg.setToolTip(tip)
            i = grid.rowCount()
            grid.addWidget(a, i, 0)
            grid.addWidget(wdg, i, 1, 1, 2 if isinstance(wdg, (QLineEdit, QComboBox)) else 1)
        for g, key, text, unit, dec, step, tip in FIELDS:
            if g != group:
                if g == 'Glare':
                    self._headings_group(ll, new_group)
                new_group(g)
                group = g
                if g == 'Campaign':
                    self.preset = QComboBox()
                    self.preset.addItem('Place…')
                    for name, *_ in PRESETS:
                        self.preset.addItem(name)
                    self.preset.activated.connect(self._preset)
                    row('Preset', self.preset, 'Fill in a city\'s latitude, longitude, UTC offset and time zone')
            f = ScrubSpinBox('', 'plain', dec, unit, step=step, width=120)
            f.committed.connect(lambda v, k=key, t=text, u=unit: self._commit(k, v, f'{t.lower()} {v:g} {u}'.strip()))
            row(text, f, tip)
            self.fields[key] = f
            if key == 'utc_offset_h':
                for tkey, ttext, ph, ttip in TEXT_FIELDS:
                    e = QLineEdit()
                    e.setPlaceholderText(ph)
                    e.setFixedHeight(ROW_H)
                    e.editingFinished.connect(lambda k=tkey, ed=e, t=ttext: self._commit_text(k, ed, t))
                    row(ttext, e, ttip)
                    self.texts[tkey] = e
        ll.addLayout(grid)
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
        b_def = self.b_defaults = button('Defaults', 'ghost', None, 'Back to the default campaign and glare settings')
        b_def.clicked.connect(self._defaults)
        bl.addWidget(b_def)
        bl.addStretch(1)
        self.b_export = button('Export…', None, 'export', 'Every camera\'s, lidar\'s and radar\'s figures, the sun '
                                                           'tracks and the findings as JSON, the month × hour heatmaps '
                                                           'and the dirt map as PNG')
        self.b_export.clicked.connect(lambda: self.export())
        self.b_run = button('Run', 'primary')
        self.b_run.clicked.connect(lambda: self.ctl.run())
        bl.addWidget(self.b_export)
        bl.addWidget(self.b_run)
        ll.addLayout(bl)
        sc = QScrollArea()
        sc.setWidgetResizable(True)
        sc.setWidget(w)
        fit_width(sc)
        sc.setMinimumWidth(320)
        return sc

    def _headings_group(self, ll, new_group):
        new_group('Route headings')
        self.mode = SegmentedControl([('uniform', 'Uniform'), ('route', 'Route')], 'uniform')
        self.mode.setToolTip('Uniform: every heading equally likely. Route: the time spent on each heading of an '
                             'imported track, or a typed list')
        self.mode.changed.connect(self._set_mode)
        hl = QHBoxLayout()
        col = QVBoxLayout()
        col.setSpacing(4)
        col.addWidget(self.mode)
        self.route_lbl = label('', muted=True, small=True, wrap=True)
        col.addWidget(self.route_lbl)
        b_imp = self.b_import = button('Import GPX / CSV…', 'ghost', 'import',
                                       'Headings from a track: GPX <trkpt lat lon> (with <time>: weighted by time) or '
                                       'CSV rows lat,lon[,time]; also sets the location to the track\'s centre')
        b_imp.clicked.connect(lambda: self.import_route())
        col.addWidget(b_imp)
        col.addStretch(1)
        hl.addLayout(col, 1)
        self.rose = HeadingRose()
        hl.addWidget(self.rose)
        ll.addLayout(hl)
        g = QGridLayout()
        g.setVerticalSpacing(2)
        g.setHorizontalSpacing(6)
        g.setColumnStretch(2, 1)
        a = label('Headings', ink2=True)
        a.setFixedWidth(104)
        self.head_text = QLineEdit()
        self.head_text.setPlaceholderText('0:3, 90:1, 180:3, 270:1')
        self.head_text.setFixedHeight(ROW_H)
        tip = ('Heading in degrees clockwise from north, a colon and its weight, separated by commas: '
               '"0:3, 180:3, 90:1, 270:1" is mostly north–south')
        a.setToolTip(tip)
        self.head_text.setToolTip(tip)
        self.head_text.editingFinished.connect(self._commit_headings)
        g.addWidget(a, 0, 0)
        g.addWidget(self.head_text, 0, 1, 1, 2)
        a = label('Heading step', ink2=True)
        a.setFixedWidth(104)
        self.step_f = ScrubSpinBox('', 'plain', 0, '°', step=5.0, width=120)
        self.step_f.setToolTip('Uniform: one heading every this many degrees')
        self.step_f.committed.connect(lambda v: self._commit('heading_step', v, f'heading step {v:g}°'))
        g.addWidget(a, 1, 0)
        g.addWidget(self.step_f, 1, 1)
        self.fields['heading_step'] = self.step_f
        ll.addLayout(g)

    @property
    def settings(self) -> X.ExposureSettings:
        return self.ctl.settings

    def _set(self, text, **kw) -> bool:
        st = self.settings
        for k, v in kw.items():
            setattr(st, k, v)
        if not self.ctl.store(st, text):
            self._sync_fields()                 # nothing changed (a clamped value): show what is stored
            return False
        return True

    def _commit(self, key, value, text):
        return self._set(text, **{key: value})

    def _commit_text(self, key, edit, text):
        if self._syncing or not edit.isModified():
            return                              # focus leaving an untouched field is not an edit
        edit.setModified(False)
        value = edit.text().strip()
        if key in ('start', 'end'):
            try:
                datetime.date.fromisoformat(value[:10])
            except ValueError:
                edit.setText(getattr(self.settings, key))           # what is stored (the field keeps the focus)
                self.ctx.msg(f'{text}: not a date (YYYY-MM-DD)', 4000)
                return
        if key == 'timezone' and value:
            try:
                from zoneinfo import ZoneInfo
                ZoneInfo(value)
            except Exception:
                self.ctx.msg(f'"{value}" is not a time zone known here: the fixed UTC offset is used', 6000)
        if getattr(self.settings, key) == value:
            return
        if not self._set(f'{text.lower()} {value or "none"}', **{key: value}):
            edit.setText(getattr(self.settings, key))           # not usable: what is stored

    def _commit_headings(self):
        if self._syncing or not self.head_text.isModified():
            return
        self.head_text.setModified(False)
        text = self.head_text.text().strip()
        if not text:
            self._sync_fields()
            return
        h = X.parse_heading_text(text)
        if h is None:
            self.ctx.msg('Headings: type "heading:weight" pairs, e.g. 0:3, 180:3, 90:1', 5000)
            st = self.settings
            self.head_text.setText(X.heading_text(st.headings) if isinstance(st.headings, list) else '')
            return
        self._set('typed headings', headings=h)

    def _set_mode(self, mode):
        st = self.settings
        if mode == 'uniform':
            self._set('uniform headings', headings='uniform')
        elif isinstance(st.headings, list):
            return
        elif st.route.get('headings'):
            self._set(f'headings of {st.route.get("name", "the route")}', headings=st.route['headings'])
        else:
            self.mode.set_value('uniform')
            self.ctx.msg('Import a GPX or CSV track, or type headings, for route headings', 5000)

    def _preset(self, i):
        if i <= 0:
            return
        name, lat, lon, off, tz = PRESETS[i - 1]
        self.preset.setCurrentIndex(0)
        self._set(f'location {name}', lat=lat, lon=lon, utc_offset_h=off, timezone=tz)

    def _defaults(self):
        if self.ctx.store.project.analysis.exposure:
            def fn(p):
                p.analysis.exposure = {}
            self.ctx.store.edit('analysis', fn, 'Sun glare and dirt · defaults')

    def import_route(self, path=None):
        if path is None:
            path, _ = QFileDialog.getOpenFileName(self, 'Import a route for its headings', '',
                                                  'Tracks (*.gpx *.csv *.txt);;All files (*)')
            if not path:
                return None
        try:
            r = X.read_route(path)
        except (OSError, ValueError) as e:
            self.ctx.msg(f'Could not read the route: {e}', 6000)
            return None
        st = self.settings
        st.route = {k: r[k] for k in ('name', 'headings', 'lat', 'lon', 'km', 'hours') if r.get(k) is not None}
        st.headings = r['headings']
        st.lat, st.lon = r['lat'], r['lon']
        self.ctl.store(st, f'headings from {r["name"]}')
        self.ctx.msg(f'{r["name"]}: {r["km"]:.1f} km, {len(r["headings"])} headings weighted by {r["weighting"]}; '
                     f'location set to its centre ({r["lat"]:.3f}, {r["lon"]:.3f})', 6000)
        return r

    def _sync_fields(self):
        st = self.settings
        self._syncing = True
        try:
            d = st.to_dict()
            for key, f in self.fields.items():
                f.set_value(d[key])
            for key, e in self.texts.items():
                if not e.isModified() and e.text() != d[key]:          # leaves what the user is typing alone
                    e.setText(d[key])
            route = isinstance(st.headings, list)
            self.mode.set_value('route' if route else 'uniform')
            text = X.heading_text(st.headings) if route else ''
            if not self.head_text.isModified() and self.head_text.text() != text:
                self.head_text.setText(text)
            self.head_text.setPlaceholderText('0:3, 90:1, 180:3, 270:1')
            self.step_f.setEnabled(not route)
            h, w = st.heading_table()
            self.rose.set_data(h, w, not route)
            r = st.route
            if route and r and st.headings == r.get('headings'):
                bits = [r.get('name', 'route')] + ([f'{r["km"]:.1f} km'] if r.get('km') else []) + \
                       ([f'{r["hours"]:.1f} h'] if r.get('hours') else [])
                self.route_lbl.setText(' · '.join(bits) + f' · {len(st.headings)} headings')
            elif route:
                self.route_lbl.setText(f'{len(st.headings)} typed headings')
            else:
                self.route_lbl.setText(f'Every {360 / len(h):g}° equally likely')
        finally:
            self._syncing = False

    # ------------------------------------------------------------------ state
    def _retheme(self, *_):
        self._shown = None
        self.refresh()

    def showEvent(self, e):
        super().showEvent(e)
        self.ctl.set_active(True)

    def hideEvent(self, e):
        super().hideEvent(e)
        self.ctl.set_active(False)

    def activate(self):
        self.ctl.set_active(True)

    def _item(self):
        res = self.ctl.result
        return res.item(self.sel_uid) if res is not None and self.sel_uid else None

    def select(self, uid):
        """Pick a camera or sensor: the table row, the charts and the dirt map's marker follow."""
        res = self.ctl.result
        if res is None or res.item(uid) is None:
            return
        self.sel_uid = uid
        r = next((k for k, i in enumerate(res.items) if i.uid == uid), -1)
        if r >= 0 and self.table.currentRow() != r:
            self.table.blockSignals(True)
            self.table.selectRow(r)
            self.table.blockSignals(False)
        self._show_item()

    def _row_picked(self):
        res = self.ctl.result
        r = self.table.currentRow()
        if res is not None and 0 <= r < len(res.items):
            self.sel_uid = res.items[r].uid
            self._show_item()

    def _item_changed(self, item):
        if self._syncing or item.column() != 9:
            return
        res = self.ctl.result
        r = item.row()
        if res is None or not 0 <= r < len(res.items):
            return
        it = res.items[r]
        on = item.checkState() == Qt.CheckState.Checked
        st = self.settings
        rec = set(st.recessed)
        if on == (it.uid in rec):
            return
        rec = rec | {it.uid} if on else rec - {it.uid}
        self._set(f'{it.tag} ' + ('recessed' if on else 'not recessed'), recessed=sorted(rec))

    def refresh(self):
        ctl = self.ctl
        res = ctl.result
        win = self.ctx.window
        self.b_run.setEnabled(not (ctl.busy and win is not None and JOB in win._jobs))
        self.bar.setVisible(ctl.busy)
        self.b_export.setEnabled(res is not None)
        self.status.setText((ctl.status or 'Run the study') +
                            (' · stale: the design or the settings changed' if ctl.stale and res else ''))
        if res is self._shown:
            return
        self._shown = res
        self._fill_table(res)
        self._fill_findings(res)
        self.dirt.set_data(res, self.sel_uid)
        if res is None:
            self.heat.set_data(None, [], 0.05)
            self.sky.set_data(None, None)
            self.tracks.set_data(None)
            return
        if res.item(self.sel_uid) is None:
            first = next((i for i in res.items if i.kind == 'camera'), res.items[0] if res.items else None)
            self.sel_uid = first.uid if first is not None else ''
        if self.sel_uid:
            self.select(self.sel_uid)
        else:
            self._show_item()

    def _show_item(self):
        res = self.ctl.result
        it = self._item()
        if res is None:
            return
        peers = [i for i in res.items if i.sun is not None and (it is None or i.kind == it.kind)]
        vmax = max([float(np.nanmax(i.sun['month_hour'])) for i in peers
                    if np.isfinite(i.sun['month_hour']).any()] + [0.05])
        self.heat.set_data(it, res.hours_cols, vmax)
        self.sky.set_data(res, it)
        self.tracks.set_data(it, self.head.value() or 'worst')
        self.head.setEnabled(it is not None and it.kind == 'camera')
        self.dirt.set_data(res, self.sel_uid)

    def _fill_table(self, res):
        t = self.table
        self._syncing = True
        t.blockSignals(True)
        try:
            items = res.items if res is not None else []
            t.setRowCount(len(items))
            st = res.settings if res is not None else None
            for r, it in enumerate(items):
                name = _text(f'{it.tag} · {it.model}', f'{it.kind.capitalize()} {it.tag}: {it.model}')
                name.setData(Qt.ItemDataRole.DecorationRole, item_color(it))
                t.setItem(r, 0, name)
                s = it.sun
                if s is None:
                    for c in range(1, 7):
                        t.setItem(r, c, _num('—', 'Radars do not see sunlight', THEME.color('muted')) if c <= 3
                                  else _text('', 'Radars do not see sunlight'))
                else:
                    cam = it.kind == 'camera'
                    sev = ('serious' if s['share'] >= X.SUN_SERIOUS or s['low'] >= X.LOW_SERIOUS else
                           'warning' if s['share'] >= X.SUN_WARN or s['low'] >= X.SUN_WARN else 'good') if cam \
                        else ('warning' if s['share'] >= X.LIDAR_WARN else None)
                    t.setItem(r, 1, _num(_pct(s['share']),
                                         f'{s["hours"]:,.0f} h of {res.metrics["driving_hours"]:,.0f} h of driving'
                                         + (f'; the body hides it another {100 * s["blocked"]:.1f} %'
                                            if s['blocked'] > 0 else '')
                                         + ('' if cam else ' (inside the lidar\'s field: an indicator of solar '
                                                           'background)'),
                                         QColor(STATUS[sev]) if sev and sev != 'good' else None))
                    t.setItem(r, 2, _num(_pct(s['near']) if cam else '—',
                                         f'Within {st.flare_margin_deg:g}° outside the image' if cam else ''))
                    t.setItem(r, 3, _num(_pct(s['low']), f'In the image below {st.low_sun_deg:g}° elevation'))
                    t.setItem(r, 4, _text(X.months_text(s['worst_months'])))
                    t.setItem(r, 5, _text(X.hours_text(s['worst_hours'])))
                    t.setItem(r, 6, _text(X.heading_span_text(s['worst_span']),
                                          '; '.join(f'{h:.0f}°: {100 * v:.1f} % of the time on that heading'
                                                    for h, v in s['worst_headings'])))
                d = it.dirt
                tip = (f'Mostly {X.DIRT_LABEL[d["dominant"]]}' + (f' on its {d["facet"]}' if d.get('facet') else '')
                       + '\n' + '\n'.join(f'{X.DIRT_LABEL[k]}: {v:.2f} × {X.DIRT_WEIGHTS[k]:g} = '
                                          f'{d["contributions"][k]:.2f}' for k, v in d['factors'].items())
                       + ('\nrecessed or hooded: × 0.6' if d['recessed'] else ''))
                t.setItem(r, 7, _num(f'{d["score"]:.2f}', tip))
                tone = X.CLASS_TONE[d['class']]
                t.setItem(r, 8, _text(d['class'].capitalize(), tip,
                                      QColor(STATUS[tone]) if it.kind != 'radar' else THEME.color('ink2')))
                chk = QTableWidgetItem('')
                chk.setFlags((chk.flags() | Qt.ItemFlag.ItemIsUserCheckable) & ~Qt.ItemFlag.ItemIsEditable)
                chk.setCheckState(Qt.CheckState.Checked if d['recessed'] else Qt.CheckState.Unchecked)
                chk.setToolTip(TABLE_COLS[9][1])
                t.setItem(r, 9, chk)
                t.setItem(r, 10, _text(d['advice'], d['advice']))
        finally:
            t.blockSignals(False)
            self._syncing = False

    def _fill_findings(self, res):
        for w in self.findings:
            self.find_lay.removeWidget(w)
            w.hide()                    # until the deferred delete runs it would still paint over the new rows
            w.deleteLater()
        self.findings = []
        for f in (res.findings if res is not None else []):
            w = StatusLine(f['severity'], f['message'])
            self.find_lay.insertWidget(self.find_lay.count() - 1, w)
            self.findings.append(w)

    # ------------------------------------------------------------------ export
    def export(self, folder=None):
        res = self.ctl.result
        if res is None:
            return []
        if folder is None:
            folder = QFileDialog.getExistingDirectory(self, 'Export the sun glare and dirt exposure into')
            if not folder:
                return []
        from ...core import provenance as PV          # where the JSON came from
        paths = PV.stamp_paths(X.export(res, self.ctx.store.project, folder), self.ctx.store.project, folder)
        self.ctx.msg(f'Exported the sun glare and dirt exposure to {folder}', 5000)
        return paths
