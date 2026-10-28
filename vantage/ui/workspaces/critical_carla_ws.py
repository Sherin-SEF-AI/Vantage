"""Analysis ▸ Critical targets in CARLA: the project's critical targets stood in CARLA around the still ego (each as
the actor nearest it in kind and size), recorded with ground truth, and compared camera by camera with the critical-
targets study (core.critical_carla): the agreement, the pixel error, every disagreement explained, and crops of each
camera's view of each target with the actor's own pixels outlined and the study's rectangle drawn. The run is a
background job with Cancel (drives already recorded from the same rig are reused); nothing runs by itself after an
edit (the result is marked stale)."""
from __future__ import annotations

import json
import math
import threading
import traceback

import numpy as np
from PyQt6.QtCore import QObject, QPointF, QRectF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QImage, QPainter, QPen
from PyQt6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QFileDialog, QGridLayout, QHBoxLayout,
                             QHeaderView, QLineEdit, QProgressBar, QScrollArea, QSplitter, QTableWidget,
                             QTableWidgetItem, QTabWidget, QTextEdit, QVBoxLayout, QWidget)

from ... import workdir
from ...core import critical as CR
from ...core import critical_carla as CC
from ..charts import ChartBase
from ..theme import STATUS, THEME, mono, sans
from ..widgets import ROW_H, ScrubSpinBox, StatusLine, button, hline, label
from .recording_ws import Tile

JOB = 'critical_carla'
EMPTY = ('Add critical targets on Analysis ▸ Critical targets, start a CARLA server (Simulate ▸ CARLA), then Run: '
         'every target is stood in CARLA as the actor nearest it and each camera\'s view of it is compared with the '
         'study.')
# (key, label, unit, decimals, step, tip)
FIELDS = (
    ('seed', 'Seed', '', 0, 1.0, 'Chooses the spawn point when it is −1 (a straight road)'),
    ('duration_s', 'Duration', 's', 1, 0.5, 'Seconds recorded per drive (the scene stands still: one second is ten '
                                             'frames at the lidar\'s rate)'),
    ('image_scale', 'Image scale', '', 2, 0.05, 'Image size against the sensor\'s (1: the study\'s own pixels; six '
                                                'full-size fisheyes with ground truth need more than the bridge\'s 8 GB)'),
    ('frames', 'Frames', '', 0, 1.0, 'The last frames of each drive measured; their spread is stated'),
)
TARGET_COLS = (('Target', ''), ('Kind', ''), ('CARLA actor', 'What stands in for it in CARLA (the nearest in kind and '
                                                              'size, measured on CARLA 0.9.16)'),
               ('Size', 'The actor\'s height (rendered, for a walker) against the target\'s'))
ROW_COLS = (('Target', ''), ('Camera', ''), ('Scene', 'The drive: scene and spawn point'),
            ('Study', 'The study\'s look at the nominal target: pixels across its height · share seen'),
            ('At actor size', 'The study\'s look at a target of the actor\'s size and place'),
            ('CARLA', 'Measured on CARLA\'s ground truth: the actor\'s own pixels\' extent along its height · share '
                      'seen (range image), with half the frames\' spread'),
            ('CARLA car hides', 'Share of the target in view that CARLA\'s own car hides (its pixels, by its actor id)'),
            ('Agree', 'The camera meets the target in CARLA as the prediction at the actor\'s size says'),
            ('Why', 'The reason found for a disagreement (or a pixel difference over 15 %)'))
RES_COLS = (('Target', ''), ('Actor', 'The blueprint and its family'), ('Nominal', 'The target\'s height × width'),
            ('Actor size', 'Its box as spawned: length × width × height'),
            ('Height', 'The actor\'s height the comparison measures (a walker\'s rendered height, a board\'s band, '
                       'else its box) against the target\'s'),
            ('Study', 'Resolved by the study (nominal)'), ('CARLA', 'Resolved in CARLA, per scene'),
            ('Agree', 'Resolved alike in CARLA and the prediction at the actor\'s size, per scene'))
TABS = ('Agreement', 'Targets', 'Crops', 'Chart', 'Notes')


def _table(cols, stretch=None):
    t = QTableWidget(0, len(cols))
    for c, (h, tip) in enumerate(cols):
        it = QTableWidgetItem(h)
        if tip:
            it.setToolTip(tip)
        t.setHorizontalHeaderItem(c, it)
    t.verticalHeader().setVisible(False)
    t.verticalHeader().setDefaultSectionSize(ROW_H)
    t.horizontalHeader().setFixedHeight(ROW_H)
    t.setShowGrid(False)
    t.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    t.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    t.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
    hh = t.horizontalHeader()
    for c in range(len(cols)):
        hh.setSectionResizeMode(c, QHeaderView.ResizeMode.Stretch if c == stretch else
                                QHeaderView.ResizeMode.ResizeToContents)
    if stretch is None:
        hh.setStretchLastSection(True)
    return t


def _cell(t, r, c, text, num=False, tip=None, color=None):
    it = QTableWidgetItem(text)
    if num:
        it.setFont(mono(12))
        it.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    if tip:
        it.setToolTip(tip)
    if color is not None:
        it.setForeground(color)
    t.setItem(r, c, it)
    return it


def _qimage(bgr) -> QImage:
    rgb = np.ascontiguousarray(bgr[:, :, ::-1])
    h, w = rgb.shape[:2]
    return QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()


def _look_text(d) -> str:
    return '—' if not d else f'{d["px"]:.0f} px · {100 * d["seen"]:.0f} %'


def _yes(v) -> str:
    return '—' if v is None else ('yes' if v else 'no')


def candidates(kind: str, name: str = '') -> list:
    """(choice, label) the actor combo offers for a target kind: Automatic, the blueprints of its kind nearest first,
    a stand-in prop, Not in CARLA."""
    out = [('auto', 'Automatic')]
    if kind in ('child', 'pedestrian') or (kind == 'object' and CC._object_kind(name) in ('child', 'adult')):
        ages = ('child', 'teenager') if kind == 'child' else ('adult', 'teenager', 'child')
        seen = set()
        for k, r in sorted(CC.WALKERS.items(), key=lambda kr: (ages.index(kr[1][0]) if kr[1][0] in ages else 9,
                                                                kr[1][2])):
            if r[0] in ages and (r[0], r[2]) not in seen:
                seen.add((r[0], r[2]))
                out.append((f'blueprint:walker.pedestrian.{k}', f'{r[0]} walker {k} ({r[2]:.2f} m)'))
    elif kind == 'car' or (kind == 'object' and CC._object_kind(name) in ('car', 'truck', 'van', 'bus')):
        from ...core import simrig as SR
        for k, r in sorted(SR.CARLA_VEHICLES.items(), key=lambda kr: kr[1][3]):
            if r[0] in ('car', 'van', 'truck', 'bus'):
                out.append((f'blueprint:{k}', f'{k.split(".", 1)[1]} ({r[3]:.2f} × {r[2]:.2f} m)'))
    elif kind == 'sign':
        for k, b in CC.BOARDS.items():
            out.append((f'blueprint:static.prop.{k}', f'board {k} ({b["width"]:.2f} × {b["z1"] - b["z0"]:.2f} m)'))
    elif kind == 'object':
        for k, r in sorted(CC.TWO_WHEELERS.items()):
            out.append((f'blueprint:{k}', f'{k.split(".", 1)[1]} with its rider'))
    if kind not in ('kerb', 'lane'):
        out.append(('standin', 'Stand-in prop'))
    out.append(('none', 'Not in CARLA'))
    return out


# ============================================================================ the chart
class PxChart(ChartBase):
    """CARLA's pixels against the prediction at the actor's size (log scales), one dot per compared camera view:
    green agreeing, red not; the 1:1 line; hover names the target, camera and scene."""
    MARGIN = (48, 28, 14, 34)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(260)
        self.pts = []                       # [(x px, y px, agree, label)]
        self.empty = 'Run to see CARLA\'s pixels against the study\'s.'
        self._screen = []

    def set_data(self, pts, title='', subtitle=''):
        self.pts = list(pts)
        self.title, self.subtitle = title, subtitle
        self.hover = None
        self.tip = None
        self.update()

    def _rect(self):
        l, t, r, b = self.MARGIN
        return QRectF(l, t, max(self.width() - l - r, 10), max(self.height() - t - b, 10))

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.paint_title(p)
        r = self._rect()
        if not self.pts:
            p.setFont(sans(11))
            p.setPen(THEME.color('muted'))
            p.drawText(r, Qt.AlignmentFlag.AlignCenter, self.empty)
            return
        vals = [x for a, b, _, _ in self.pts for x in (a, b) if x > 0]
        hi = max(max(vals) * 1.3, 50.0)
        lo = 10.0 ** math.floor(math.log10(max(min(vals) / 1.5, 1.0)))      # the decade below the smallest
        L0, L1 = math.log10(lo), math.log10(hi)

        def X(v):
            return r.left() + (math.log10(max(v, lo)) - L0) / (L1 - L0) * r.width()

        def Y(v):
            return r.bottom() - (math.log10(max(v, lo)) - L0) / (L1 - L0) * r.height()
        p.setFont(mono(10))
        for k in range(0, int(math.ceil(L1)) + 1):
            for m in (1, 2, 5):
                v = m * 10 ** k
                if lo <= v <= hi:
                    p.setPen(QPen(THEME.color('hair'), 1))
                    p.drawLine(QPointF(X(v), r.top()), QPointF(X(v), r.bottom()))
                    p.drawLine(QPointF(r.left(), Y(v)), QPointF(r.right(), Y(v)))
                    p.setPen(THEME.color('muted'))
                    p.drawText(QRectF(X(v) - 20, r.bottom() + 3, 40, 14), Qt.AlignmentFlag.AlignCenter, f'{v:g}')
                    p.drawText(QRectF(0, Y(v) - 7, r.left() - 6, 14),
                               Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, f'{v:g}')
        p.setPen(QPen(THEME.color('strong'), 1, Qt.PenStyle.DashLine))
        p.drawLine(QPointF(X(lo), Y(lo)), QPointF(X(hi), Y(hi)))
        self._screen = []
        for i, (a, b, ok, lab) in enumerate(self.pts):
            q = QPointF(X(a), Y(b))
            col = QColor(STATUS['good'] if ok else STATUS['serious'])
            hov = self.hover == i
            p.setPen(QPen(THEME.color('ink') if hov else col, 1))
            p.setBrush(col.lighter(130) if hov else col)
            p.drawEllipse(q, 4.0 if hov else 3.0, 4.0 if hov else 3.0)
            self._screen.append(q)
        p.setFont(sans(10))
        p.setPen(THEME.color('ink2'))
        p.drawText(QRectF(r.left(), self.height() - 16, r.width(), 14), Qt.AlignmentFlag.AlignRight,
                   'the study at the actor\'s size (sensor px, log)  ·  up: CARLA')
        self.paint_tip(p)

    def mouseMoveEvent(self, e):
        pos = e.position()
        best, bd = None, 10.0
        for i, q in enumerate(self._screen):
            d = math.hypot(q.x() - pos.x(), q.y() - pos.y())
            if d < bd:
                best, bd = i, d
        if best is None:
            if self.hover is not None:
                self.hover, self.tip = None, None
                self.update()
            return
        a, b, ok, lab = self.pts[best]
        self.hover = best
        self.set_tip(self._screen[best], f'CARLA {b:.0f} px · study {a:.0f} px', lab,
                     'agree' if ok else 'disagree')
        self.update()


class CropGrid(QWidget):
    """Crops of one target, one per camera and scene, in rows that fill the width."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.items = []                       # [(QImage, caption)]
        self.empty = 'Run, then pick a target to see each camera\'s view of it.'
        self.setMinimumHeight(240)
        THEME.changed.connect(self._theme_changed)

    def _theme_changed(self, *_):
        self.update()

    def set_items(self, items):
        self.items = list(items)
        self._relayout()
        self.update()

    def _relayout(self):
        x, y, rowh = 6, 6, 0
        W = max(self.width(), 200)
        self._pos = []
        for img, cap in self.items:
            w, h = img.width(), img.height() + 16
            if x + w > W - 6 and x > 6:
                x, y = 6, y + rowh + 8
                rowh = 0
            self._pos.append((x, y))
            x += w + 8
            rowh = max(rowh, h)
        self.setMinimumHeight(max(y + rowh + 8, 240))

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._relayout()

    def paintEvent(self, e):
        p = QPainter(self)
        p.fillRect(self.rect(), THEME.color('base'))
        if not self.items:
            p.setFont(sans(11))
            p.setPen(THEME.color('muted'))
            p.drawText(QRectF(self.rect()), Qt.AlignmentFlag.AlignCenter, self.empty)
            return
        if len(getattr(self, '_pos', [])) != len(self.items):
            self._relayout()
        p.setFont(sans(10))
        for (img, cap), (x, y) in zip(self.items, self._pos):
            p.setPen(THEME.color('ink2'))
            p.drawText(QRectF(x, y, img.width(), 14), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, cap)
            p.drawImage(QPointF(x, y + 16), img)


class _Sig(QObject):
    done = pyqtSignal(int, object)
    failed = pyqtSignal(int, str)
    progress = pyqtSignal(int, float, str)


# ============================================================================ the page
class CriticalCarlaPage(QWidget):
    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        self.result: CC.CriticalCarlaResult | None = None
        self.busy = False
        self.status_text = ''
        self.runner = None                    # runner(rig_json, out_dir, progress, cancel): the tests' bridge
        self.python = None                    # CARLA's Python (None: the CARLA page's choice)
        self.out_dir = None                   # where the drives go (None: <working folder>/<project>/carla/critical)
        self._gen = 0
        self._for = None                      # (project, settings key) of the result
        self._dirty = True
        self._syncing = False
        self._plan = None
        self.sig = _Sig()
        self.sig.done.connect(self._thread_done)
        self.sig.failed.connect(self._failed)
        self.sig.progress.connect(self._progress)
        self.setObjectName('base')
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        split = QSplitter(Qt.Orientation.Horizontal)
        split.setHandleWidth(2)
        split.addWidget(self._left())
        split.addWidget(self._right())
        split.setSizes([400, 1000])
        lay.addWidget(split)
        self.srv_timer = QTimer(self)
        self.srv_timer.setInterval(3000)
        self.srv_timer.timeout.connect(self._refresh_server)
        ctx.store.changed.connect(self._on_change)
        ctx.store.project_replaced.connect(self._on_replaced)
        THEME.changed.connect(self._retheme)
        self._sync_fields()
        self.refresh()

    # ------------------------------------------------------------------ left: CARLA, the scene, the targets
    def _left(self):
        w = QWidget()
        w.setObjectName('panel')
        ll = QVBoxLayout(w)
        ll.setContentsMargins(8, 6, 8, 8)
        ll.setSpacing(6)
        ll.addWidget(label('Critical targets in CARLA', bold=True))
        ll.addWidget(label('Each critical target stood in CARLA as the actor nearest it, the ego parked, recorded '
                           'with ground truth; what each camera sees of it compared with the critical-targets study.',
                           muted=True, small=True, wrap=True))
        ll.addWidget(hline())
        # ---- CARLA
        ll.addWidget(label('CARLA', bold=True))
        self.srv_status = StatusLine(None, '')
        ll.addWidget(self.srv_status)
        sr = QHBoxLayout()
        sr.addStretch(1)
        self.b_server = button('Start server', 'ghost', 'play', 'Start the CARLA server the CARLA page is set up for '
                                                                '(headless, capped)')
        self.b_server.clicked.connect(self.start_server)
        sr.addWidget(self.b_server)
        ll.addLayout(sr)
        ll.addWidget(hline())
        # ---- the scene
        ll.addWidget(label('Scene', bold=True))
        g = QGridLayout()
        g.setHorizontalSpacing(8)
        g.setVerticalSpacing(4)
        a = label('Town', ink2=True)
        a.setFixedWidth(104)
        self.town = QLineEdit()
        self.town.setPlaceholderText('the CARLA page\'s')
        self.town.setToolTip('The CARLA map (empty: the town set on Simulate ▸ CARLA)')
        self.town.editingFinished.connect(self._town_edited)
        g.addWidget(a, 0, 0)
        g.addWidget(self.town, 0, 1)
        g.addWidget(label('Weather', ink2=True), 1, 0)
        from ...core import simrig as SR
        self.weather = QComboBox()
        for wx in SR.CARLA_WEATHERS:
            self.weather.addItem(wx, wx)
        self.weather.setToolTip('CARLA\'s weather preset for the scenes (its light and the sun\'s shadows)')
        self.weather.currentIndexChanged.connect(self._weather_picked)
        g.addWidget(self.weather, 1, 1)
        g.addWidget(label('Spawn points', ink2=True), 2, 0)
        self.spawns = QLineEdit()
        self.spawns.setToolTip('Where the ego parks: CARLA spawn point numbers, comma separated (−1: one chosen by the '
                               'seed on a straight road); every scene is driven at each (other backgrounds and light)')
        self.spawns.editingFinished.connect(self._spawns_edited)
        g.addWidget(self.spawns, 2, 1)
        self.fields = {}
        for key, text, unit, dec, step, tip in FIELDS:
            f = ScrubSpinBox('', 'plain', dec, unit, step=step, width=118)
            f.setToolTip(tip)
            f.committed.connect(lambda v, k=key, t=text, u=unit: self._set(f'{t.lower()} {v:g} {u}'.strip(),
                                                                           **{k: v}))
            a = label(text, ink2=True)
            a.setToolTip(tip)
            i = g.rowCount()
            g.addWidget(a, i, 0)
            g.addWidget(f, i, 1)
            self.fields[key] = f
        ll.addLayout(g)
        self.group = QCheckBox('Targets that cannot see each other share a drive')
        self.group.setToolTip('Off: every target in a drive of its own (the study looks at each target alone)')
        self.group.toggled.connect(lambda on: None if self._syncing else self._set(
            'targets ' + ('grouped' if on else 'one per drive'), group=bool(on)))
        ll.addWidget(self.group)
        ll.addWidget(hline())
        # ---- the targets
        ll.addWidget(label('Targets', bold=True))
        self.tg_table = _table(TARGET_COLS, stretch=0)
        self.tg_table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.tg_table.setMinimumHeight(6 * ROW_H)
        self.tg_table.verticalHeader().setDefaultSectionSize(ROW_H + 2)
        ll.addWidget(self.tg_table)
        self.plan_lbl = label('', muted=True, small=True, wrap=True)
        ll.addWidget(self.plan_lbl)
        ll.addSpacing(4)
        ll.addWidget(label('Limits', bold=True))
        ll.addWidget(label(CC.LIMITS_TEXT, muted=True, small=True, wrap=True))
        ll.addStretch(1)
        self.status = label('', muted=True, small=True, wrap=True)
        ll.addWidget(self.status)
        self.bar = QProgressBar()
        self.bar.setTextVisible(False)
        self.bar.setRange(0, 1000)
        self.bar.hide()
        ll.addWidget(self.bar)
        bl = QHBoxLayout()
        self.b_open = button('Open drives…', 'ghost', 'folder', 'Compare drives recorded before without CARLA: one '
                                                                 'drive, or a folder holding drives at any depth (per '
                                                                 'seed, per scene)')
        self.b_open.clicked.connect(lambda: self.open_drives())
        bl.addWidget(self.b_open)
        bl.addStretch(1)
        self.b_export = button('Export…', None, 'export', 'The comparison as JSON and CSV, the targets\' sizes, the '
                                                          'pixel chart and every crop')
        self.b_export.clicked.connect(lambda: self.export())
        self.b_cancel = button('Cancel', 'ghost')
        self.b_cancel.clicked.connect(self.cancel)
        self.b_run = button('Run', 'primary')
        self.b_run.clicked.connect(lambda: self.run())
        bl.addWidget(self.b_export)
        bl.addWidget(self.b_cancel)
        bl.addWidget(self.b_run)
        ll.addLayout(bl)
        sc = QScrollArea()
        sc.setWidgetResizable(True)
        sc.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        sc.setWidget(w)
        sc.setMinimumWidth(380)
        return sc

    # ------------------------------------------------------------------ right: tiles, tabs
    def _right(self):
        w = QWidget()
        w.setObjectName('panel')
        rl = QVBoxLayout(w)
        rl.setContentsMargins(10, 8, 10, 8)
        rl.setSpacing(6)
        tl = QHBoxLayout()
        self.tiles = {}
        for k, cap, tip in (('agree', 'Agreement', 'Camera views where CARLA and the prediction at the actor\'s size '
                                                   'give the same verdict'),
                            ('nominal', 'With the study', 'The same against the study\'s nominal targets'),
                            ('targets', 'Targets', 'Targets resolved alike (some camera meets them), per scene'),
                            ('px', 'Pixel error', 'Median of CARLA − prediction at the actor\'s size, views where '
                                                  'both see over 90 % of it'),
                            ('drives', 'Drives', 'Scenes × spawn points driven')):
            t = Tile(cap, tip)
            self.tiles[k] = t
            tl.addWidget(t)
        rl.addLayout(tl)
        self.tabs = QTabWidget()
        # agreement
        aw = QWidget()
        al = QVBoxLayout(aw)
        al.setContentsMargins(0, 4, 0, 0)
        self.only_dis = QCheckBox('Only disagreements and pixel differences over 15 %')
        self.only_dis.toggled.connect(lambda _: self._fill_rows())
        al.addWidget(self.only_dis)
        self.rows = _table(ROW_COLS, stretch=8)
        self.rows.itemSelectionChanged.connect(self._row_picked)
        al.addWidget(self.rows)
        self.tabs.addTab(aw, TABS[0])
        # targets
        self.res_table = _table(RES_COLS)
        self.tabs.addTab(self.res_table, TABS[1])
        # crops
        cw = QWidget()
        cl = QVBoxLayout(cw)
        cl.setContentsMargins(0, 4, 0, 0)
        cr = QHBoxLayout()
        cr.addWidget(label('Target', ink2=True))
        self.crop_target = QComboBox()
        self.crop_target.currentIndexChanged.connect(lambda _: self._fill_crops())
        cr.addWidget(self.crop_target, 1)
        cr.addWidget(label('Scene', ink2=True))
        self.crop_drive = QComboBox()
        self.crop_drive.currentIndexChanged.connect(lambda _: self._fill_crops())
        cr.addWidget(self.crop_drive, 1)
        cl.addLayout(cr)
        cl.addWidget(label('Green: the actor\'s own pixels in CARLA. Amber: the rectangle the study measures at the '
                           'actor\'s size. Tinted: CARLA\'s car.', muted=True, small=True, wrap=True))
        self.crops = CropGrid()
        csc = QScrollArea()
        csc.setWidgetResizable(True)
        csc.setWidget(self.crops)
        cl.addWidget(csc, 1)
        self.tabs.addTab(cw, TABS[2])
        # chart
        self.chart = PxChart()
        self.tabs.addTab(self.chart, TABS[3])
        # notes
        self.notes = QTextEdit()
        self.notes.setReadOnly(True)
        self.tabs.addTab(self.notes, TABS[4])
        rl.addWidget(self.tabs, 1)
        self.findings = label('', small=True, wrap=True)
        self.findings.setTextFormat(Qt.TextFormat.PlainText)
        rl.addWidget(self.findings)
        return w

    # ------------------------------------------------------------------ settings
    @property
    def settings(self) -> CC.CriticalCarlaSettings:
        return CC.settings_of(self.ctx.store.project)

    def _set(self, text, **kw) -> bool:
        if self._syncing:
            return False
        st = self.settings
        new = CC.CriticalCarlaSettings.from_dict(dict(st.to_dict(), **kw))
        if new.to_dict() == st.to_dict():
            self._sync_fields()
            return False

        def fn(p):
            p.analysis.critical_carla = new.to_dict()
        self.ctx.store.edit('analysis', fn, f'CARLA critical targets: {text}')
        return True

    def set_actor(self, uid, choice) -> bool:
        acts = dict(self.settings.actors)
        if choice == 'auto':
            acts.pop(uid, None)
        else:
            acts[uid] = choice
        t = CR.settings_of(self.ctx.store.project).target(uid)
        return self._set(f'{t["name"] if t else "target"} as {CC.CHOICE_LABEL.get(choice, choice.split(":", 1)[-1])}',
                         actors=acts)

    def _town_edited(self):
        self._set(f'town {self.town.text().strip() or "the CARLA page\'s"}', town=self.town.text().strip())

    def _weather_picked(self, i):
        if not self._syncing and i >= 0:
            self._set(f'weather {self.weather.itemData(i)}', weather=self.weather.itemData(i))

    def _spawns_edited(self):
        self._set(f'spawn points {self.spawns.text().strip()}', spawn_points=self.spawns.text())

    def _sync_fields(self):
        st = self.settings
        self._syncing = True
        try:
            self.town.setText(st.town)
            self.weather.setCurrentIndex(max(self.weather.findData(st.weather), 0))
            self.spawns.setText(', '.join(str(x) for x in st.spawn_points))
            for k, f in self.fields.items():
                f.set_value(getattr(st, k))
            self.group.setChecked(st.group)
            self._fill_targets()
        finally:
            self._syncing = False

    def _fill_targets(self):
        p = self.ctx.store.project
        try:
            self._plan = CC.plan(p, self.settings)
        except Exception:                            # noqa: BLE001 (a project the plan cannot read: shown empty)
            self._plan = None
        rows = self._plan.targets if self._plan else []
        self.tg_table.setRowCount(len(rows))
        st = self.settings
        for i, r in enumerate(rows):
            a = r['actor']
            _cell(self.tg_table, i, 0, r['name'], tip=r['place'])
            _cell(self.tg_table, i, 1, CR.KIND_LABEL.get(r['kind'], r['kind']))
            combo = QComboBox()
            combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(10)
            for choice, lab in candidates(r['kind'], r['name']):
                combo.addItem(lab, choice)
            cur = st.actors.get(r['uid'], 'auto')
            k = combo.findData(cur)
            if k < 0:
                combo.addItem(cur.split(':', 1)[-1], cur)
                k = combo.count() - 1
            combo.setCurrentIndex(k)
            combo.setToolTip((a['note'] if a['spawnable'] else a['why']) or '')
            combo.currentIndexChanged.connect(lambda j, c=combo, uid=r['uid']: None if self._syncing else
                                              self.set_actor(uid, c.itemData(j)))
            self.tg_table.setCellWidget(i, 2, combo)
            hh = CC.actor_height(a)
            size = (f'{hh:.2f}/{a["nominal"]["height"]:.2f} m' if a['spawnable'] and hh else
                    'not in CARLA' if not a['spawnable'] else '—')
            _cell(self.tg_table, i, 3, size, num=True, tip=(a['note'] if a['spawnable'] else a['why']) or None,
                  color=None if a['spawnable'] else QColor(STATUS['warning']))
        if self._plan is None or not rows:
            self.plan_lbl.setText('No critical target yet: add them on Analysis ▸ Critical targets.')
        else:
            n_live = sum(1 for r in rows if r['actor']['spawnable'])
            sp = len(self.settings.spawn_points)
            nd = len(self._plan.drives)
            ran = [d['seconds'] for d in (self.result.drives if self.result is not None else [])
                   if not d.get('reused') and d.get('seconds')]
            each = float(np.mean(ran)) if ran else CC.SECONDS_PER_DRIVE
            cost = (f' (about {each * nd / 60:.0f} min: {each:.0f} s a drive, ' +
                    ('the last run' if ran else 'measured: six fisheyes at half size on an RTX 5080') + ')') if each else ''
            self.plan_lbl.setText(f'{n_live} of {len(rows)} targets stood in CARLA in {len(self._plan.groups)} '
                                  f'scene{"s" * (len(self._plan.groups) != 1)} × {sp} spawn point{"s" * (sp != 1)}: '
                                  f'{nd} drive{"s" * (nd != 1)}{cost}.' +
                                  ('' if not self._plan.notes else ' ' + ' '.join(self._plan.notes)))

    # ------------------------------------------------------------------ the server
    def _sim(self):
        from ...core import simrig as SR
        return SR.settings_of(self.ctx.store.project)

    def _refresh_server(self):
        from ...sim import server as SV
        sim = self._sim()
        if self.runner is not None:
            self.srv_status.set('good', 'A test bridge stands in for CARLA.')
        elif SV.port_open('127.0.0.1', sim.port):
            self.srv_status.set('good', f'A CARLA server answers on port {sim.port}.')
        else:
            self.srv_status.set('warning', f'No CARLA server on port {sim.port}: start one here or on Simulate ▸ '
                                           'CARLA.')

    def start_server(self) -> bool:
        """The CARLA page starts the server it is set up for (its CARLA folder, headless, capped)."""
        win = self.ctx.window
        try:
            sim_page = win.page('sim') if win is not None else None
        except Exception:                            # noqa: BLE001 (no CARLA page in this build)
            sim_page = None
        if sim_page is None or not hasattr(sim_page, 'start_server'):
            self.ctx.msg('Open Simulate ▸ CARLA to start a server.', 5000)
            return False
        ok = bool(sim_page.start_server())
        if not ok:
            self.ctx.msg('The CARLA page could not start a server: choose the CARLA folder there.', 6000)
        QTimer.singleShot(1000, self._refresh_server)
        return ok

    # ------------------------------------------------------------------ state
    def _key(self, p=None):
        p = p or self.ctx.store.project
        return json.dumps({'cc': CC.settings_of(p).to_dict(), 'targets': CR.settings_of(p).to_dict()['targets'],
                           'sim': self._sim().to_dict() if p is self.ctx.store.project else None},
                          sort_keys=True, default=str)

    @property
    def stale(self) -> bool:
        return self.result is not None and self._for is not None and self._for[1] != self._key()

    def _on_change(self, *_):
        if not self.isVisible():
            self._dirty = True
            return
        self._sync_fields()
        self.refresh()

    def _on_replaced(self, *_):
        self._gen += 1                                # a run still going is for the old project
        self.result, self._for, self.busy = None, None, False
        self.status_text = ''
        self._sync_fields()
        self.refresh()

    def _retheme(self, *_):
        self._show()

    def showEvent(self, e):
        super().showEvent(e)
        self.activate()

    def hideEvent(self, e):
        super().hideEvent(e)
        self.srv_timer.stop()

    def activate(self):
        if self._dirty:
            self._dirty = False
            self._sync_fields()
        self._refresh_server()
        self.srv_timer.start()
        self.refresh()

    # ------------------------------------------------------------------ jobs
    def _start(self, title, work):
        win = self.ctx.window
        gen = self._gen
        if win is not None and hasattr(win, 'run_job'):
            return win.run_job(JOB, title, work, lambda out: self._thread_done(gen, out),
                               lambda m: self._failed(gen, m))

        def thread():
            try:
                out = work(lambda f, t='': self.sig.progress.emit(gen, f, t), lambda: gen != self._gen)
                self.sig.done.emit(gen, out)
            except Exception as e:                    # noqa: BLE001
                try:
                    self.sig.failed.emit(gen, f'{e}\n{traceback.format_exc()}')
                except RuntimeError:
                    pass
        threading.Thread(target=thread, daemon=True).start()
        return True

    def _progress(self, gen, f, text):
        if gen == self._gen:
            self.bar.show()
            self.bar.setValue(int(f * 1000))
            if text:
                self.status.setText(text)

    def _thread_done(self, gen, out):
        if gen != self._gen:
            return
        self.busy = False
        key, res = out if out is not None else (None, None)
        if res is None or key[0] is not self.ctx.store.project:
            self.status_text = 'Cancelled: drives already recorded are kept and reused.' if res is None else ''
            self.refresh()
            return
        self.result, self._for = res, key
        m = res.metrics
        self.status_text = (f'{m["pairs"]} camera views compared in {res.seconds:.0f} s, '
                            f'{sum(1 for d in res.drives if d.get("reused"))} of {len(res.drives)} drives reused')
        self.ctx.store.project.add_log('critical_carla', f'{m["agree"]} of {m["pairs"]} camera views agree with the '
                                                         f'critical-targets study in CARLA')
        self.ctx.msg(f'Critical targets in CARLA · {self.status_text}', 5000)
        self._fill_pickers()
        self.refresh()

    def _failed(self, gen, msg):
        if gen == self._gen and self.busy:
            self.busy = False
            self.status_text = 'Failed: ' + msg.splitlines()[0]
        self.refresh()

    def run(self) -> bool:
        """Every scene driven in CARLA (drives already there from the same rig reused) and compared: a background
        job."""
        if self.busy:
            return False
        from ...sim import server as SV
        win = self.ctx.window
        other = win.carla_user('critical_carla') if win is not None and hasattr(win, 'carla_user') else ''
        if other:                                     # one client steps a synchronous CARLA world
            self.status_text = other
            self.refresh()
            return False
        p = self.ctx.store.project
        if not any(t['enabled'] for t in CR.settings_of(p).targets):
            self.status_text = 'Add critical targets first (Analysis ▸ Critical targets).'
            self.refresh()
            return False
        if self._plan is not None and not self._plan.drives:
            self.status_text = 'None of the targets can be stood in CARLA (see the Targets table).'
            self.refresh()
            return False
        sim = self._sim()
        python = self.python
        if self.runner is None:
            python = python or CC.carla_python()
            if not python:
                self.status_text = 'Choose CARLA\'s client Python on Simulate ▸ CARLA first.'
                self.refresh()
                return False
            if not SV.port_open('127.0.0.1', sim.port):
                self.status_text = f'No CARLA server answers on port {sim.port}: Start server (or Simulate ▸ CARLA).'
                self.refresh()
                return False
        st = self.settings
        snap = p.snapshot()
        out = self.out_dir or workdir.project_dir(p, workdir.get(self.ctx.settings), 'carla', 'critical',
                                                  workdir.slug(p.variant.name))
        key = (p, self._key(p))
        runner = self.runner
        self._gen += 1
        self.busy = True
        self.status_text = 'Running…'
        self.bar.setValue(0)

        def work(progress, cancel):
            res = CC.run(snap, None, st, None, lambda f, t='': progress(0.97 * f, t), cancel, sim, out, python,
                         runner)
            if res is not None and not CC.make_crops(res, progress=lambda f, t='': progress(0.97 + 0.03 * f, t),
                                                     cancel=cancel):
                return key, None
            return key, res
        ok = self._start('Critical targets in CARLA', work)
        if not ok:
            self.busy = False
            self.status_text = 'Another run is going.'
        self.refresh()
        return ok

    def open_drives(self, folder=None) -> bool:
        """Compare drives recorded before (a folder of drives, or one drive) without CARLA: a background job."""
        if self.busy:
            return False
        if folder is None:
            folder = QFileDialog.getExistingDirectory(self, 'Open CARLA drives of the critical targets',
                                                      self.out_dir or workdir.get(self.ctx.settings))
            if not folder:
                return False
        drives = [folder] if CC.is_drive(folder) else CC.drives_in(folder)
        if not drives:
            self.status_text = f'No CARLA drive with ground truth in {folder}.'
            self.refresh()
            return False
        p = self.ctx.store.project
        st = self.settings
        snap = p.snapshot()
        key = (p, self._key(p))
        self._gen += 1
        self.busy = True
        self.status_text = f'Comparing {len(drives)} drive{"s" * (len(drives) != 1)}…'

        keys = CC.drive_keys(drives)              # per-seed subfolders named alike stay apart ('seed2/scene1-…')

        def work(progress, cancel):
            res = CC.analyse(snap, [{'dir': d, 'key': k, 'group': None, 'spawn_point': None, 'reused': True}
                                    for d, k in zip(drives, keys)], st, progress=progress, cancel=cancel)
            if res is not None and not CC.make_crops(res, cancel=cancel):
                return key, None
            return key, res
        ok = self._start('Critical targets in CARLA', work)
        if not ok:
            self.busy = False
        self.refresh()
        return ok

    def cancel(self):
        win = self.ctx.window
        j = getattr(win, '_jobs', {}).get(JOB) if win is not None else None
        if j is not None:
            j['cancel'].set()
        elif self.busy:
            self._gen += 1
            self.busy = False
            self.status_text = 'Cancelled'
        self.refresh()

    def export(self, folder=None) -> list:
        res = self.result
        if res is None:
            return []
        if folder is None:
            folder = QFileDialog.getExistingDirectory(self, 'Export the critical targets in CARLA into')
            if not folder:
                return []
        from ...core import provenance as PV
        paths = PV.stamp_paths(CC.export(res, self.ctx.store.project, folder), self.ctx.store.project, folder)
        self.ctx.msg(f'Exported the critical targets in CARLA to {folder}', 5000)
        return paths

    def summary(self) -> dict | None:
        return CC.summary(self.result) if self.result is not None else None

    # ------------------------------------------------------------------ showing the result
    def refresh(self):
        st_txt = self.status_text
        if self.result is not None and self.stale and not self.busy:
            st_txt = (st_txt + ' · ' if st_txt else '') + ('Settings or targets changed since the run: Run again '
                                                           '(drives of the same rig are reused).')
        self.status.setText(st_txt)
        self.bar.setVisible(self.busy)
        self.b_run.setEnabled(not self.busy)
        self.b_open.setEnabled(not self.busy)
        self.b_cancel.setEnabled(self.busy)
        self.b_export.setEnabled(self.result is not None and not self.busy)
        self._show()

    def _fill_pickers(self):
        res = self.result
        self.crop_target.blockSignals(True)
        self.crop_drive.blockSignals(True)
        cur_t, cur_d = self.crop_target.currentData(), self.crop_drive.currentData()
        self.crop_target.clear()
        self.crop_drive.clear()
        if res is not None:
            for t in res.targets:
                if t['actor']['spawnable']:
                    self.crop_target.addItem(t['name'], t['uid'])
            self.crop_drive.addItem('Every scene', '')
            for d in res.drives:
                self.crop_drive.addItem(d['key'], d['key'])
        k = self.crop_target.findData(cur_t)
        self.crop_target.setCurrentIndex(max(k, 0))
        k = self.crop_drive.findData(cur_d)
        self.crop_drive.setCurrentIndex(max(k, 0))
        self.crop_target.blockSignals(False)
        self.crop_drive.blockSignals(False)

    def _show(self):
        res = self.result
        for t in self.tiles.values():
            t.set('—')
        if res is None:
            self.rows.setRowCount(0)
            self.res_table.setRowCount(0)
            self.chart.set_data([], 'CARLA against the study')
            self.crops.set_items([])
            self.notes.setPlainText(EMPTY + '\n\n' + CC.ASSUMPTIONS + '\n\n' + CC.LIMITS_TEXT)
            self.findings.setText('')
            return
        m = res.metrics
        if m['pairs']:
            ci = m['agreement_ci']
            self.tiles['agree'].set(f'{100 * m["agreement"]:.0f}', '%', f'{m["agree"]} of {m["pairs"]} views · '
                                                                        f'{100 * ci[0]:.0f}–{100 * ci[1]:.0f} %',
                                    'good' if m['agreement'] >= 0.9 else 'warning')
        if m['agreement_nominal'] is not None:
            self.tiles['nominal'].set(f'{100 * m["agreement_nominal"]:.0f}', '%',
                                      f'{m["agree_nominal"]} of {m["pairs_nominal"]} views')
        if m['target_agreement'] is not None:
            self.tiles['targets'].set(f'{m["targets_agree"]}/{m["targets_scenes"]}', '',
                                      'resolved alike, per scene')
        pe = m['px_error_rel_clear'] or m['px_error_rel']
        if pe:
            self.tiles['px'].set(f'{100 * pe["median"]:+.1f}', '%', f'{pe["n"]} views · '
                                                                    f'RMS {100 * pe["rms"]:.1f} %')
        self.tiles['drives'].set(str(len(res.drives)), '', f'{sum(1 for d in res.drives if d.get("reused"))} reused')
        self._fill_rows()
        self._fill_res()
        self._fill_crops()
        self.chart.set_data([(r['vantage_actor']['px'], r['carla']['px'], bool(r['agree']),
                              f'{r["target"]} · {r["camera"]} · {r["drive"]}') for r in res.rows_of(counted=True)
                             if r['vantage_actor']['px'] > 0 or r['carla']['px'] > 0],
                            'CARLA against the study', f'{m["pairs"]} views')
        self._fill_notes()
        bad = [f for f in res.findings if f['severity'] in ('serious', 'warning')]
        self.findings.setText('\n'.join(f'• {f["message"]}' for f in bad[:4]) +
                              (f'\n… and {len(bad) - 4} more (Agreement tab)' if len(bad) > 4 else ''))

    def _fill_rows(self):
        res = self.result
        if res is None:
            return
        rows = [r for r in res.rows_of(counted=True)
                if not self.only_dis.isChecked() or not r['agree'] or r['reason']]
        self._row_list = rows
        self.rows.setRowCount(len(rows))
        for i, r in enumerate(rows):
            c = r['carla']
            _cell(self.rows, i, 0, r['target'])
            _cell(self.rows, i, 1, r['camera'])
            _cell(self.rows, i, 2, r['drive'], tip=f'spawn point {r["spawn_point"]}')
            _cell(self.rows, i, 3, _look_text(r['vantage']), num=True,
                  color=QColor(STATUS['good' if (r['vantage'] or {}).get('met') else 'serious']))
            _cell(self.rows, i, 4, _look_text(r['vantage_actor']), num=True,
                  color=QColor(STATUS['good' if r['vantage_actor']['met'] else 'serious']))
            _cell(self.rows, i, 5, _look_text(c) + (f' ±{c["px_spread"] / 2:.0f}' if c['frames'] > 1 and
                                                    c['px_spread'] >= 1 else ''), num=True,
                  color=QColor(STATUS['good' if c['met'] else 'serious']),
                  tip='; '.join(f'{k} hides {100 * v:.0f} %' for k, v in c['hidden'].items()) or None)
            _cell(self.rows, i, 6, f'{100 * c["ego_hidden"]:.0f} %', num=True)
            _cell(self.rows, i, 7, _yes(r['agree']), color=QColor(STATUS['good' if r['agree'] else 'serious']))
            _cell(self.rows, i, 8, r['why'] or '', tip='\n'.join(x['text'] for x in r['reasons']) or None)

    def _row_picked(self):
        rows = getattr(self, '_row_list', [])
        i = self.rows.currentRow()
        if 0 <= i < len(rows):
            k = self.crop_target.findData(rows[i]['uid'])
            if k >= 0:
                self.crop_target.setCurrentIndex(k)

    def _fill_res(self):
        res = self.result
        self.res_table.setRowCount(len(res.targets))
        for i, t in enumerate(res.targets):
            a = t['actor']
            sz = next((s['size_lwh'] for s in t['spawned'] if s.get('size_lwh')), None)
            _cell(self.res_table, i, 0, t['name'], tip=t['place'])
            _cell(self.res_table, i, 1, (a['blueprint'] or 'not in CARLA') + (' (stand-in)' if a['stand_in'] else ''),
                  tip=(a['note'] if a['spawnable'] else a['why']) or None,
                  color=None if a['spawnable'] else QColor(STATUS['warning']))
            _cell(self.res_table, i, 2, f'{t["nominal"]["height"]:.2f} × {t["nominal"]["width"]:.2f} m', num=True)
            _cell(self.res_table, i, 3, ' × '.join(f'{x:.2f}' for x in sz) + ' m' if sz else '—', num=True)
            hh = t['actor_height']
            _cell(self.res_table, i, 4, f'{hh:.2f} m ({100 * (hh / t["nominal"]["height"] - 1):+.0f} %)' if hh else '—',
                  num=True)
            _cell(self.res_table, i, 5, _yes(t['resolved_study']))
            _cell(self.res_table, i, 6, ' '.join(f'{_yes(v)}' for v in t['resolved_carla'].values()) or '—',
                  tip='\n'.join(f'{k}: {_yes(v)}' for k, v in t['resolved_carla'].items()) or None)
            ag = list(t['agree'].values())
            _cell(self.res_table, i, 7, f'{sum(ag)}/{len(ag)}' if ag else '—',
                  color=QColor(STATUS['good' if ag and all(ag) else 'serious']) if ag else None)

    def _fill_crops(self):
        res = self.result
        if res is None:
            self.crops.set_items([])
            return
        uid, drv = self.crop_target.currentData(), self.crop_drive.currentData()
        items = []
        for r in res.rows_of(uid=uid, counted=True):
            if drv and r['drive'] != drv:
                continue
            im = res.crops.get(CC.row_key(r))
            if im is None:
                continue
            items.append((_qimage(im), f'{r["camera"]} · {r["drive"]}' + ('' if r['agree'] else ' · disagree')))
        self.crops.empty = ('No camera of this scene sees it.' if uid else
                            'Run, then pick a target to see each camera\'s view of it.')
        self.crops.set_items(items)

    def _fill_notes(self):
        res = self.result
        m = res.metrics
        L = [f'{res.variant}: {m["targets"]} targets, {m["spawned"]} stood in CARLA, {m["drives"]} drives, '
             f'{m["pairs"]} camera views compared in {res.seconds:.0f} s.', '']
        for k, v in CC.METHOD.items():
            L.append(f'{k}: {v}')
        L += ['', 'Drives:']
        for d in res.drives:
            L.append(f'  {d["key"]}: {d.get("town") or ""} spawn point {(d.get("start") or {}).get("spawn_point")}, '
                     f'{d.get("weather") or ""}, ego {d.get("ego") or ""}' + (' (reused)' if d.get('reused') else '') +
                     (f'; cameras inside CARLA\'s car: {", ".join(d["mounts_inside"])}' if d.get('mounts_inside')
                      else ''))
        pe = m['px_error']
        if pe:
            L += ['', f'Pixel error (CARLA − at the actor\'s size): median {pe["median"]:+.1f} px, RMS {pe["rms"]:.1f} '
                      f'px over {pe["n"]} views; frame-to-frame spread median '
                      f'{(m["frames_spread_px"] or {}).get("median", 0):.1f} px.']
        if m['disagreements_by_reason']:
            L.append('Disagreements by reason: ' + ', '.join(f'{CC.REASONS.get(k, k)} {n}' for k, n in
                                                             m['disagreements_by_reason'].items()))
        L += ['', 'Findings:'] + [f'  [{f["severity"]}] {f["message"]}' for f in res.findings]
        L += ['', 'Assumptions: ' + CC.ASSUMPTIONS, '', 'Limits: ' + CC.LIMITS_TEXT, '',
              f'Actor sizes measured on {CC.MEASURED_ON}.'] + [f'Note: {n}' for n in res.notes]
        self.notes.setPlainText('\n'.join(L))
