"""Analysis ▸ Cleaning and de-icing: washer nozzles, air nozzles and lens heaters on the rig's cameras and lidars
(core.cleaning): what each spray covers, how long a clean takes and how often one is needed, how long the tank lasts
over a shift, whether the heaters clear ice at speed in the cold, and the power and harness they add.

Left: the devices (add one on a camera or lidar, propose a starting layout, remove), the selected device's place and
figures (each figure's source shown: datasheet, measured, fitted, assumed or yours) and the system (pump, tank, shift,
dirt, ambient, speed). Right: the totals, a 3D view of the spray cones and the windows (covered points green, missed
red) or a flat map of every window, the per-device table, the clearing time per dirt kind, the power budget with the
cleaning's loads, the tubing and wiring runs, and the findings. The study is cheap; it runs in a background thread
shortly after every edit while the page shows (core.exposure's dirt scores are part of it)."""
from __future__ import annotations

import math
import threading
import traceback

import numpy as np
from PyQt6.QtCore import QObject, QPointF, QRectF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import (QAbstractItemView, QComboBox, QFileDialog, QHBoxLayout, QHeaderView, QScrollArea,
                             QSplitter, QStackedWidget, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from ...core import cleaning as CL
from ..charts import ChartBase
from ..scene import MainScene
from ..theme import STATUS, THEME, sans
from ..widgets import (ROW_H, ScrubSpinBox, Section, SegmentedControl, StatusLine, button, fit_width, hline, label)
from .recording_ws import Tile

AUTO_DELAY_MS = 150
SYSTEM = (('pump_w', 'Pump power', 0, 'W', 5.0, 1.0, 'The pump\'s power while it runs (HELLA states none: assumed '
                                                     '40 W until you enter yours)'),
          ('tank_l', 'Tank', 2, 'l', 0.1, 1.0, 'Washer tank volume (HELLA: 1.5 l)'),
          ('shift_h', 'Shift', 1, 'h', 0.5, 1.0, 'Hours of driving between refills'),
          ('max_obscured', 'Clean at', 0, '%', 1.0, 100.0, 'A window is cleaned when this share of it is obscured'),
          ('event_s', 'Spray at least', 1, 's', 0.1, 1.0, 'Shortest spray per cleaning event'),
          ('ambient_c', 'Ambient', 0, '°C', 1.0, 1.0, 'Air temperature for the heaters'),
          ('speed_kmh', 'Speed', 0, 'km/h', 5.0, 1.0, 'Vehicle speed: the airflow over the heated lens'),
          ('wind_ms', 'Headwind', 1, 'm/s', 1.0, 1.0, 'Added to the speed'),
          ('ice_mm', 'Ice', 1, 'mm', 0.1, 1.0, 'Ice thickness on the window to clear'),
          ('heater_on_c', 'Heaters on below', 0, '°C', 1.0, 1.0, 'The heaters run at and below this ambient'),
          ('shed_ms', 'Air sheds drops at', 0, 'm/s', 1.0, 1.0, 'Air speed that sheds drops from a window (assumed)'),
          ('min_cover', 'Cover at least', 0, '%', 5.0, 100.0, 'A window covered less than this is a finding'))
DEVICE = (('standoff_mm', 'Standoff', 0, 'mm', 1.0, 'The nozzle\'s distance beyond the window\'s edge'),
          ('ahead_mm', 'Ahead', 0, 'mm', 1.0, 'The nozzle\'s height in front of the window (along its normal; on a '
                                              'round lidar, above or below the band)'),
          ('aim_mm', 'Aim offset', 0, 'mm', 1.0, 'Aim this far past the window\'s centre, towards the far side'),
          ('azimuth_deg', 'Azimuth', 0, '°', 5.0, 'Round lidar: where around the band the nozzle stands'),
          ('window_mm', 'Window', 1, 'mm', 1.0, 'Window diameter (0: the lens glass as drawn)'),
          ('reach_mm', 'Reach', 0, 'mm', 10.0, 'The spray reaches this far'))
FIG = {'flow_ml_min': ('Flow', 0, 'ml/min'), 'ref_bar': ('At', 1, 'bar'), 'spray_deg': ('Spray angle', 0, '°'),
       'orifice_mm': ('Orifice', 2, 'mm'), 'supply_bar': ('Supply', 1, 'bar'), 'valve_w': ('Valve', 1, 'W'),
       'power_w': ('Power', 1, 'W'), 'g0_w_k': ('Still-air loss', 4, 'W/K'), 'c_j_k': ('Heat capacity', 1, 'J/K')}
LABEL = {k: t for k, t, *_ in SYSTEM + DEVICE}
LABEL.update({k: v[0] for k, v in FIG.items()}, pump='pump', dirt='dirt', regime='airflow')
COVER = np.array([0.05, 0.64, 0.05])
AIRC = np.array([0.16, 0.53, 0.90])
MISS = np.array([0.82, 0.23, 0.23])


class _Sig(QObject):
    done = pyqtSignal(int, object)
    failed = pyqtSignal(int, str)


def _table(cols, stretch=0):
    t = QTableWidget(0, len(cols))
    t.verticalHeader().setVisible(False)
    t.verticalHeader().setDefaultSectionSize(ROW_H)
    t.horizontalHeader().setFixedHeight(ROW_H)
    t.setShowGrid(False)
    t.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    t.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    t.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
    for c, (head, tip) in enumerate(cols):
        it = QTableWidgetItem(head)
        it.setToolTip(tip)
        t.setHorizontalHeaderItem(c, it)
    hh = t.horizontalHeader()
    for c in range(len(cols)):
        hh.setSectionResizeMode(c, QHeaderView.ResizeMode.Stretch if c == stretch else
                                QHeaderView.ResizeMode.ResizeToContents)
    return t


def _fill(t, rows):
    t.setRowCount(len(rows))
    for i, r in enumerate(rows):
        for c, v in enumerate(r):
            it = QTableWidgetItem(str(v))
            it.setToolTip(str(v))
            t.setItem(i, c, it)


def _fmt(v, d=1, unit=''):
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return '—'
    return f'{v:.{d}f}{(" " + unit) if unit else ""}'


# =========================================================================== 3D
def device_draws(project, res, gl=False, selected=None) -> tuple[list, list]:
    """(lines, points) for the 3D view: each nozzle's cone (generators and the rim where it reaches the window) in
    green when its window is covered enough, else red; each cleaned window's sample points, covered green (air: blue)
    and missed red; heaters as a ring at the lens."""
    from ..viewport3d import LineDraw, PointDraw
    if res is None:
        return [], []
    lines, pts, cols = [], [], []
    st = res.settings
    by_t = {t['uid']: t for t in res.targets}
    for d in res.devices:
        if d['problem'] or d['tag'] is None:
            continue
        w = res.windows.get(d['target'])
        if w is None:
            continue
        win = w[0]
        if d['kind'] in ('washer', 'air'):
            n, a = np.array(d['nozzle']), np.array(d['aim'])
            t = by_t.get(d['target'], {})
            cov = t.get('wash_cover' if d['kind'] == 'washer' else 'air_cover') or 0.0
            ok = cov >= st.min_cover
            col = QColor(STATUS['good'] if ok else STATUS['critical'])
            if selected == d['id']:
                col = THEME.color('ink')
            L = float(np.linalg.norm(win.centre - n)) * 1.15
            u = np.cross(a, [0.0, 0.0, 1.0])
            if np.linalg.norm(u) < 1e-6:
                u = np.cross(a, [0.0, 1.0, 0.0])
            u /= np.linalg.norm(u)
            v = np.cross(a, u)
            r = L * math.tan(math.radians(d['half_deg']))
            ang = np.linspace(0, 2 * math.pi, 33)
            rim = n + a * L + r * (np.cos(ang)[:, None] * u + np.sin(ang)[:, None] * v)
            lines.append(LineDraw(rim, col, 1.2))
            for k in range(0, 32, 8):
                lines.append(LineDraw(np.array([n, rim[k]]), col, 1.2))
        else:
            ang = np.linspace(0, 2 * math.pi, 33)
            rad = win.size[0] * 1.15
            ring = win.centre + rad * (np.cos(ang)[:, None] * win.R[:, 0] + np.sin(ang)[:, None] * win.R[:, 1]) \
                + win.R[:, 2] * 0.001 if win.kind != 'band' else None
            if ring is not None:
                lines.append(LineDraw(ring, QColor(STATUS['serious']), 2.0, dash=True, dash_len=0.004))
    for t in res.targets:
        w = res.windows.get(t['uid'])
        if w is None or not (t['washers'] or t['airs']):
            continue
        win, cov, air = w
        step = max(len(win.points) // 400, 1)
        P = win.points[::step] + win.normals[::step] * 0.0015
        c = np.where(cov[::step, None], COVER, np.where(air[::step, None], AIRC, MISS))
        pts.append(P)
        cols.append(c)
    if not pts:
        return lines, []
    P = np.concatenate(pts).astype(np.float32)
    C = np.c_[np.concatenate(cols), np.ones(len(P))].astype(np.float32)
    return lines, [PointDraw(P, C, 4.0, True)]


def scene_lines(ctx) -> list:
    """For the main 3D view (View ▸ Overlays ▸ Cleaning, wired by the lead): the cones and heater rings from the
    cached result of the project as it is."""
    res = CL.cached(ctx.store.project)
    return device_draws(ctx.store.project, res)[0] if res is not None else []


def scene_points(ctx) -> list:
    res = CL.cached(ctx.store.project)
    return device_draws(ctx.store.project, res)[1] if res is not None else []


class CleaningScene(MainScene):
    """The page's 3D view: the scene with the spray cones and the windows' covered and missed points."""

    def __init__(self, ctx, page):
        super().__init__(ctx)
        self.page = page

    def lines(self, vp):
        return super().lines(vp) + device_draws(self.ctx.store.project, self.page.result,
                                                getattr(vp, 'renderer', None) is not None, self.page.selected)[0]

    def points(self, vp):
        return super().points(vp) + device_draws(self.ctx.store.project, self.page.result,
                                                 getattr(vp, 'renderer', None) is not None)[1]


# =========================================================================== flat window map
class WindowMap(ChartBase):
    """Every cleaned window flat: a lens or face as seen from in front (its top up), a round lidar's band unrolled
    (azimuth across, front in the middle); covered points green, cleared by air blue, missed red."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.title = 'Windows'
        self.items = []                 # (title, uv (N, 2), colours (N, 3), kind)
        self.setMinimumHeight(160)

    def set_data(self, project, res):
        self.items = []
        if res is not None:
            for t in res.targets:
                w = res.windows.get(t['uid'])
                if w is None or not (t['washers'] or t['airs']):
                    continue
                win, cov, air = w
                uv = win.uv.copy()
                if win.kind == 'band':
                    uv[:, 0] = -((uv[:, 0] + math.pi) % (2 * math.pi) - math.pi)
                c = np.where(cov[:, None], COVER, np.where(air[:, None], AIRC, MISS))
                wc = t['wash_cover'] if t['wash_cover'] is not None else t['air_cover']
                self.items.append((f'{t["tag"]} · {100 * (wc or 0):.0f} %', uv, c, win.kind))
        self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.paint_title(p, 'green: washed · blue: air · red: missed')
        if not self.items:
            p.setFont(sans(11))
            p.setPen(THEME.color('muted'))
            p.drawText(QRectF(0, 24, self.width(), 20), Qt.AlignmentFlag.AlignLeft,
                       'No washer or air nozzle yet: add one, or propose a layout.')
            p.end()
            return
        n = len(self.items)
        cw = max(self.width() / max(n, 1), 60.0)
        H = self.height() - 44
        for i, (title, uv, c, kind) in enumerate(self.items):
            x0 = i * cw
            box = QRectF(x0 + 6, 26, cw - 12, H)
            ext = np.abs(uv).max(0)
            ext = np.maximum(ext, 1e-6)
            if kind == 'band':
                sx, sy = (box.width() / 2) / ext[0], (box.height() / 2) / ext[1] * 0.5
            else:
                s_ = min(box.width(), box.height()) / 2 / ext.max()
                sx = sy = s_
            cx, cy = box.center().x(), box.center().y()
            zc = (uv[:, 1].max() + uv[:, 1].min()) / 2 if kind == 'band' else 0.0
            p.setPen(Qt.PenStyle.NoPen)
            step = max(len(uv) // 900, 1)
            for (u, v), col in zip(uv[::step], c[::step]):
                p.setBrush(QColor(int(col[0] * 255), int(col[1] * 255), int(col[2] * 255)))
                p.drawEllipse(QPointF(cx + u * sx, cy - (v - zc) * sy), 1.6, 1.6)
            p.setPen(QPen(THEME.color('ink2')))
            p.setFont(sans(11))
            p.drawText(QRectF(x0, self.height() - 18, cw, 16), Qt.AlignmentFlag.AlignCenter, title)
        p.end()


# =========================================================================== the page
class CleaningPage(QWidget):
    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        self.result: CL.CleaningResult | None = None
        self.busy = False
        self.stale = False
        self._gen = 0
        self._for = None
        self._syncing = False
        self.selected = None
        self.findings: list[StatusLine] = []
        self.view3d = None
        self.sig = _Sig()
        self.sig.done.connect(self._done)
        self.sig.failed.connect(self._failed)
        self.setObjectName('base')
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        split = QSplitter(Qt.Orientation.Horizontal)
        split.setHandleWidth(2)
        split.addWidget(self._left())
        split.addWidget(self._right())
        split.setSizes([360, 1240])
        lay.addWidget(split)
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(AUTO_DELAY_MS)
        self.timer.timeout.connect(self.run)
        ctx.store.changed.connect(self._on_store)
        ctx.store.project_replaced.connect(self._replaced)
        THEME.changed.connect(self._theme_changed)
        self._sync_fields()
        self.refresh()

    def _theme_changed(self, *_):
        self.refresh()

    # ------------------------------------------------------------------ left
    def _left(self):
        outer = QWidget()
        outer.setObjectName('panel')
        outer.setMinimumWidth(320)
        ol = QVBoxLayout(outer)
        ol.setContentsMargins(0, 0, 0, 0)
        ol.setSpacing(0)
        head = QWidget()
        hl = QVBoxLayout(head)
        hl.setContentsMargins(8, 6, 8, 4)
        hl.addWidget(label('Cleaning and de-icing'))
        hl.addWidget(hline())
        hl.addWidget(label('Washer and air nozzles and lens heaters on the cameras and lidars: what they cover, how '
                           'often they clean, the tank over a shift, ice at speed, the power they add.',
                           muted=True, small=True, wrap=True))
        ol.addWidget(head)
        body = QWidget()
        body.setObjectName('panel')
        bl = QVBoxLayout(body)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.setSpacing(0)
        s = Section('Devices')
        self.t_dev = _table((('Device', 'Id and kind'), ('On', 'The camera or lidar it cleans'),
                             ('Product', 'Library product or your figures')), stretch=2)
        self.t_dev.setMinimumHeight(ROW_H * 7)
        self.t_dev.itemSelectionChanged.connect(self._dev_selected)
        s.add(self.t_dev)
        self.c_kind = QComboBox()
        for k in CL.KINDS:
            self.c_kind.addItem(CL.KIND_LABEL[k], k)
        self.c_kind.currentIndexChanged.connect(lambda _i: self._fill_products())
        self.c_target = QComboBox()
        self.c_target.setToolTip('The camera or lidar the device cleans')
        self.c_prod = QComboBox()
        s.row('Kind', self.c_kind)
        s.row('On', self.c_target)
        s.row('Product', self.c_prod)
        r = QWidget()
        rl = QHBoxLayout(r)
        rl.setContentsMargins(0, 0, 0, 0)
        self.b_add = button('Add', None, 'plus', 'Add this device (one undo step)')
        self.b_add.clicked.connect(lambda: self.add_device())
        self.b_remove = button('Remove', 'ghost', 'trash', 'Remove the selected device')
        self.b_remove.clicked.connect(lambda: self.remove_device())
        self.b_propose = button('Propose a layout', 'ghost', 'sparkle', 'Washers on every camera and lidar with a '
                                'moderate dirt score or more (four round a spinning lidar), an e-con heater on every '
                                'camera: replaces the devices (one undo step)')
        self.b_propose.clicked.connect(lambda: self.propose())
        rl.addWidget(self.b_add)
        rl.addWidget(self.b_remove)
        rl.addStretch(1)
        rl.addWidget(self.b_propose)
        s.add(r)
        bl.addWidget(s)
        s = self.sec_dev = Section('Selected device')
        self.c_dprod = QComboBox()
        self.c_dprod.currentIndexChanged.connect(lambda _i: self._set_device_key('product', self.c_dprod.currentData()))
        s.row('Product', self.c_dprod)
        self.seg_side = SegmentedControl([(k, k.capitalize()) for k in CL.SIDES])
        self.seg_side.setToolTip('Which side of the window the nozzle sits on (as the image is seen, its top up)')
        self.seg_side.changed.connect(lambda k: self._set_device_key('side', k))
        s.row('Side', self.seg_side)
        self.dev_fields = {}
        for key, text, dec, unit, step, tip in DEVICE:
            f = ScrubSpinBox('', 'plain', dec, unit, step=step, width=90)
            f.setToolTip(tip)
            f.committed.connect(lambda v, k=key: self._set_device_key(k, v))
            self.dev_fields[key] = (f, s.row(text, f, tip=tip))
        self.fig_fields = {}
        for key, (text, dec, unit) in FIG.items():
            f = ScrubSpinBox('', 'plain', dec, unit, step=10 ** -dec if dec else 1.0, width=90)
            f.committed.connect(lambda v, k=key: self._set_device_param(k, v))
            self.fig_fields[key] = (f, s.row(text, f))
        self.dev_note = label('', muted=True, small=True, wrap=True)
        s.add(self.dev_note)
        bl.addWidget(s)
        s = Section('System')
        self.c_pump = QComboBox()
        for k, v in CL.PUMPS.items():
            self.c_pump.addItem(v['name'], k)
        self.c_pump.currentIndexChanged.connect(lambda _i: self.set_option('pump', self.c_pump.currentData()))
        s.row('Pump', self.c_pump)
        self.c_dirt = QComboBox()
        for k, v in CL.DIRT.items():
            self.c_dirt.addItem(v[0], k)
        self.c_dirt.setToolTip('The dirt the duty cycle and the tank are worked out for (rates and doses assumed)')
        self.c_dirt.currentIndexChanged.connect(lambda _i: self.set_option('dirt', self.c_dirt.currentData()))
        s.row('Dirt', self.c_dirt)
        self.seg_regime = SegmentedControl([('turbulent', 'Turbulent'), ('mixed', 'Mixed'), ('laminar', 'Laminar')])
        self.seg_regime.setToolTip('The airflow over a heated lens: turbulent from its leading edge (default, '
                                   'conservative), laminar turning turbulent at Re 5·10⁵, or laminar')
        self.seg_regime.changed.connect(lambda k: self.set_option('regime', k))
        s.row('Airflow', self.seg_regime)
        self.fields = {}
        for key, text, dec, unit, step, scale, tip in SYSTEM:
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
        self.status = label('', muted=True, small=True, wrap=True)
        fl.addWidget(self.status)
        row = QHBoxLayout()
        self.b_defaults = button('Defaults', 'ghost', None, 'No devices, the default system')
        self.b_defaults.clicked.connect(self._defaults)
        self.b_export = button('Export…', None, 'export', 'Every device, window, clean, the tank, power and harness '
                                                          'runs as JSON (with provenance), CSV and a plan PNG')
        self.b_export.clicked.connect(lambda: self.export())
        row.addWidget(self.b_defaults)
        row.addStretch(1)
        row.addWidget(self.b_export)
        fl.addLayout(row)
        ol.addWidget(foot)
        return outer

    # ------------------------------------------------------------------ right
    def _right(self):
        right = QSplitter(Qt.Orientation.Vertical)
        right.setHandleWidth(2)
        a = QWidget()
        a.setObjectName('panel')
        al = QVBoxLayout(a)
        al.setContentsMargins(10, 8, 10, 6)
        al.setSpacing(6)
        tiles = QHBoxLayout()
        self.tiles = {}
        for key, caption, tip in (('cover', 'Worst window', 'The window its washers and air nozzles cover least'),
                                  ('tank', 'Tank lasts', 'Driving hours on one tank with the chosen dirt'),
                                  ('power', 'Average power', 'Heaters, pump, valves and compressor over the duty'),
                                  ('ice', 'Ice cleared in', 'The slowest heater at the ambient and speed')):
            self.tiles[key] = Tile(caption, tip)
            tiles.addWidget(self.tiles[key], 1)
        al.addLayout(tiles)
        vr = QHBoxLayout()
        self.view_mode = SegmentedControl([('3d', '3D'), ('map', 'Windows')], 'map')
        self.view_mode.changed.connect(self._set_view)
        vr.addWidget(self.view_mode)
        vr.addStretch(1)
        al.addLayout(vr)
        self.stack = QStackedWidget()
        self.map = WindowMap()
        self.stack.addWidget(self.map)
        al.addWidget(self.stack, 1)
        right.addWidget(a)
        b = QWidget()
        b.setObjectName('panel')
        blay = QVBoxLayout(b)
        blay.setContentsMargins(10, 6, 10, 6)
        blay.addWidget(label('Devices', bold=True))
        self.t_res = _table((('Device', ''), ('On', ''), ('Covers', 'Share of the window its spray or jet covers'),
                             ('Missed', 'Where the rest is'), ('Flow', 'ml/min at the pump\'s pressure'),
                             ('Lands', 'Share of the spray that lands on the window'),
                             ('Clean', 'Seconds per clean with the chosen dirt'),
                             ('Power', 'W while on'), ('Lens', 'Steady lens temperature (heater)'),
                             ('Ice', 'Time to clear the ice (heater)')), stretch=0)
        blay.addWidget(self.t_res, 1)
        blay.addWidget(label('Clearing per dirt kind (rates and doses assumed)', bold=True))
        self.t_clear = _table((('Washer', ''), ('Dirt', ''), ('Clean', 's per clean'),
                               ('Events', 'Cleans per hour at the window\'s dirt score'),
                               ('Duty', 'Share of the time spraying')), stretch=1)
        blay.addWidget(self.t_clear, 1)
        right.addWidget(b)
        c = QWidget()
        c.setObjectName('panel')
        cl = QVBoxLayout(c)
        cl.setContentsMargins(10, 6, 10, 8)
        cl.addWidget(label('Power and harness', bold=True))
        self.power = label('', small=True, wrap=True)
        cl.addWidget(self.power)
        self.t_harn = _table((('Run', ''), ('From', ''), ('To', ''), ('Length', 'm'), ('Route', '')), stretch=0)
        cl.addWidget(self.t_harn, 1)
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
        right.setSizes([440, 330, 300])
        return right

    def _set_view(self, key):
        if key == '3d' and self.view3d is None:
            from ..viewport3d import make_viewport
            self.scene3d = CleaningScene(self.ctx, self)
            self.view3d = make_viewport(self.scene3d)
            self.view3d.cam.az, self.view3d.cam.el = 210.0, 25.0
            self.stack.addWidget(self.view3d)
        self.stack.setCurrentWidget(self.view3d if key == '3d' else self.map)
        self.view_mode.set_value(key)
        if key == '3d':
            self._frame()

    def _frame(self):
        res = self.result
        if self.view3d is None or res is None:
            return
        pts = [np.array(d['nozzle']) for d in res.devices if d.get('nozzle')]
        objs = {o.uid: o for o in list(self.ctx.store.project.variant.cameras) +
                list(self.ctx.store.project.variant.sensors)}
        pts += [objs[t['uid']].pos() for t in res.targets if t['uid'] in objs]
        if pts:
            P = np.array(pts, float)
            self.view3d.frame_box(P.min(0) - 0.3, P.max(0) + 0.3, 1.1)

    # ------------------------------------------------------------------ settings
    @property
    def settings(self) -> CL.CleaningSettings:
        return CL.settings_of(self.ctx.store.project)

    def _commit(self, new, text) -> bool:
        if new == self.settings.to_dict():
            self._sync_fields()
            return False

        def fn(p):
            p.analysis.cleaning = new
        self.ctx.store.edit('analysis', fn, f'Cleaning · {text}')
        return True

    def set_option(self, key, value) -> bool:
        if self._syncing:
            return False
        d = self.settings.to_dict()
        d[key] = value
        new = CL.CleaningSettings.from_dict(d).to_dict()
        shown = new.get(key)
        shown = f'{shown:g}' if isinstance(shown, float) else shown
        return self._commit(new, f'{LABEL.get(key, key)} {shown}')

    def _targets(self):
        p = self.ctx.store.project
        out = [(c.uid, c.tag) for c in p.variant.cameras]
        out += [(s.uid, s.tag) for s in p.variant.sensors if p.sensor_model_for(s).is_lidar]
        return out

    def add_device(self, kind=None, target=None, product=None) -> str | None:
        """Add a device (the pickers' choices unless given); returns its id."""
        kind = kind or self.c_kind.currentData()
        target = target or self.c_target.currentData()
        product = product or self.c_prod.currentData()
        if not target:
            self.ctx.msg('No camera or lidar to put it on', 3000)
            return None
        d = self.settings.to_dict()
        ids = {x['id'] for x in d['devices']}
        n = 1
        while f'C{n}' in ids:
            n += 1
        did = f'C{n}'
        d['devices'] = d['devices'] + [{'id': did, 'kind': kind, 'product': product, 'target': target}]
        tag = dict(self._targets()).get(target, target)
        before = self.selected
        self.selected = did                      # selected as the edit lands, so the editor shows it at once
        if self._commit(CL.CleaningSettings.from_dict(d).to_dict(), f'{CL.KIND_LABEL[kind].lower()} on {tag}'):
            return did
        self.selected = before
        return None

    def remove_device(self, did=None) -> bool:
        did = did or self.selected
        d = self.settings.to_dict()
        keep = [x for x in d['devices'] if x['id'] != did]
        if len(keep) == len(d['devices']):
            return False
        d['devices'] = keep
        self.selected = None
        return self._commit(CL.CleaningSettings.from_dict(d).to_dict(), f'{did} removed')

    def propose(self) -> int:
        lay = CL.propose_layout(self.ctx.store.project)
        d = self.settings.to_dict()
        d['devices'] = lay
        self.selected = None
        self._commit(CL.CleaningSettings.from_dict(d).to_dict(), f'proposed layout ({len(lay)} devices)')
        return len(lay)

    def _cur(self):
        return next((x for x in self.settings.devices if x['id'] == self.selected), None)

    def _set_device_key(self, key, value) -> bool:
        if self._syncing or self.selected is None:
            return False
        d = self.settings.to_dict()
        for x in d['devices']:
            if x['id'] == self.selected:
                x[key] = value
                if key == 'product':
                    x['params'] = {}
        new = CL.CleaningSettings.from_dict(d).to_dict()
        shown = f'{value:g}' if isinstance(value, float) else (CL.PRODUCTS[value]['name'] if key == 'product'
                                                               else value)
        return self._commit(new, f'{self.selected} {LABEL.get(key, key)} {shown}')

    def _set_device_param(self, key, value) -> bool:
        if self._syncing or self.selected is None:
            return False
        d = self.settings.to_dict()
        for x in d['devices']:
            if x['id'] == self.selected:
                x['params'] = dict(x.get('params') or {}, **{key: value})
        return self._commit(CL.CleaningSettings.from_dict(d).to_dict(),
                            f'{self.selected} {FIG[key][0].lower()} {value:g} {FIG[key][2]}')

    def _defaults(self):
        if self.ctx.store.project.analysis.cleaning:
            def fn(p):
                p.analysis.cleaning = {}
            self.selected = None
            self.ctx.store.edit('analysis', fn, 'Cleaning · defaults')

    def _fill_products(self):
        kind = self.c_kind.currentData()
        self.c_prod.blockSignals(True)
        self.c_prod.clear()
        for k, v in CL.PRODUCTS.items():
            if v['kind'] == kind:
                self.c_prod.addItem(f'{v["maker"]} {v["name"]}'.strip(), k)
        self.c_prod.blockSignals(False)

    def _dev_selected(self):
        if self._syncing:
            return
        rows = self.t_dev.selectionModel().selectedRows()
        st = self.settings
        self.selected = st.devices[rows[0].row()]['id'] if rows and rows[0].row() < len(st.devices) else None
        self._sync_device()
        if self.view3d is not None:
            self.view3d.update()

    def _sync_fields(self):
        st = self.settings
        self._syncing = True
        try:
            for k, (f, scale) in self.fields.items():
                f.set_value(float(getattr(st, k)) * scale)
            self.c_pump.setCurrentIndex(list(CL.PUMPS).index(st.pump))
            self.c_dirt.setCurrentIndex(list(CL.DIRT).index(st.dirt))
            self.seg_regime.set_value(st.regime)
            tg = self._targets()
            if [self.c_target.itemData(i) for i in range(self.c_target.count())] != [u for u, _ in tg]:
                self.c_target.clear()
                for u, t in tg:
                    self.c_target.addItem(t, u)
            if self.c_prod.count() == 0:
                self._fill_products()
            tags = dict(tg)
            self.t_dev.setRowCount(len(st.devices))
            sel_row = None
            for i, x in enumerate(st.devices):
                prod = CL.PRODUCTS[x['product']]
                vals = [f'{x["id"]} · {CL.KIND_LABEL[x["kind"]].lower()}' + ('' if x['enabled'] else ' (off)'),
                        tags.get(x['target'], '?'), f'{prod["maker"]} {prod["name"]}'.strip()]
                for c, v in enumerate(vals):
                    it = QTableWidgetItem(v)
                    it.setToolTip(prod['cite'])
                    self.t_dev.setItem(i, c, it)
                if x['id'] == self.selected:
                    sel_row = i
            if sel_row is not None:
                self.t_dev.selectRow(sel_row)
            else:
                self.t_dev.clearSelection()
                if self.selected and not any(x['id'] == self.selected for x in st.devices):
                    self.selected = None
        finally:
            self._syncing = False
        self._sync_device()
        self.b_remove.setEnabled(self.selected is not None)

    def _sync_device(self):
        x = self._cur()
        self.sec_dev.setEnabled(x is not None)
        self.b_remove.setEnabled(x is not None)
        if x is None:
            self.dev_note.setText('Select a device to edit its place and figures.')
            return
        self._syncing = True
        try:
            self.c_dprod.clear()
            for k, v in CL.PRODUCTS.items():
                if v['kind'] == x['kind']:
                    self.c_dprod.addItem(f'{v["maker"]} {v["name"]}'.strip(), k)
            self.c_dprod.setCurrentIndex(max(self.c_dprod.findData(x['product']), 0))
            self.seg_side.set_value(x['side'])
            nozzle = x['kind'] in ('washer', 'air')
            self.seg_side.setEnabled(nozzle)
            for k, (f, row) in self.dev_fields.items():
                f.set_value(float(x[k]))
                row.setVisible(nozzle or k == 'window_mm')
            srcs = []
            for k, (f, row) in self.fig_fields.items():
                val, src = CL.figure(x, k)
                row.setVisible(val is not None)
                if val is not None:
                    f.set_value(float(val))
                    f.setToolTip(f'Source: {src}')
                    row.label.setText(f'{FIG[k][0]} ({src})')
                    srcs.append(src)
            self.dev_note.setText(CL.PRODUCTS[x['product']]['cite'])
        finally:
            self._syncing = False

    # ------------------------------------------------------------------ running
    def _key(self):
        try:
            return CL.cache_key(self.ctx.store.project)
        except Exception:                                        # noqa: BLE001
            return None

    def _on_store(self, *_):
        if self.isVisible():
            self._sync_fields()
        else:
            self._fields_dirty = True
        if self._for != self._key():
            self.stale = True
            if self.isVisible():
                self.timer.start()

    def _replaced(self, *_):
        self._gen += 1
        self.result, self._for, self.stale, self.busy, self.selected = None, None, False, False, None
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
            self.run()

    def _engine(self):
        win = getattr(self.ctx, 'window', None)
        try:
            if win is not None and hasattr(win, '_shared_engine'):
                return win._shared_engine()
            return self.ctx.analysis.engine.share()
        except Exception:                                        # noqa: BLE001
            return None

    def run(self) -> bool:
        project = self.ctx.store.project.snapshot()
        key = self._key()
        self._gen += 1
        gen = self._gen
        self.busy = True
        eng = self._engine()

        def thread():
            try:
                res = CL.run(project, eng, cancel=lambda: gen != self._gen)
                pb = CL.power_budget(project, res) if res is not None else None
                self.sig.done.emit(gen, (key, res, pb))
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
        key, res, pb = out
        if res is None:
            return
        self.result, self._for, self.power_res = res, key, pb
        self.stale = key != self._key()
        self.refresh()
        if self.view3d is not None:
            self.view3d.update()
        if self.stale and self.isVisible():
            self.timer.start()

    def _failed(self, gen, msg):
        if gen != self._gen:
            return
        self.busy = False
        self.status.setText('Failed: ' + msg.splitlines()[0])

    # ------------------------------------------------------------------ views
    def refresh(self):
        res = self.result
        for w in self.findings:
            self.find_lay.removeWidget(w)
            w.hide()
            w.deleteLater()
        self.findings = []
        self.b_export.setEnabled(res is not None)
        self.map.set_data(self.ctx.store.project, res)
        if res is None:
            for t in self.tiles.values():
                t.set('—', '', '')
            for t in (self.t_res, self.t_clear, self.t_harn):
                t.setRowCount(0)
            self.power.setText('')
            self.status.setText('')
            return
        st, tot = res.settings, res.totals
        covs = [(t['wash_cover'] if t['wash_cover'] is not None else t['air_cover'], t['tag']) for t in res.targets
                if t['washers'] or t['airs']]
        if covs:
            c, tag = min(covs)
            self.tiles['cover'].set(f'{100 * c:.0f}', '%', f'{tag} covered', 'good' if c >= st.min_cover else 'warning')
        else:
            self.tiles['cover'].set('—', '', 'no washer or air nozzle')
        if tot['washers']:
            rt = tot['runtime_h']
            self.tiles['tank'].set('∞' if not math.isfinite(rt) else f'{rt:.1f}', 'h',
                                   f'{tot["tank_ml"] / 1000:g} l · {tot["fluid_ml_h"]:.0f} ml/h · {tot["dirt_label"]}',
                                   'good' if tot['tank_ok'] else 'serious')
        else:
            self.tiles['tank'].set('—', '', 'no washer')
        self.tiles['power'].set(f'{tot["average_w"]:.1f}', 'W', f'peak {tot["peak_w"]:.0f} W · heaters '
                                + ('on' if tot['heaters_on'] else 'off') + f' at {st.ambient_c:g} °C')
        hs = [d for d in res.devices if d['kind'] == 'heater' and not d['problem']]
        if hs and st.ambient_c < 0:
            worst = max(hs, key=lambda d: d['clear_s'] if d['clear_s'] is not None else math.inf)
            if worst['clear_s'] is None:
                self.tiles['ice'].set('never', '', f'{worst["tag"]} at {worst["steady_c"]:.1f} °C', 'serious')
            else:
                self.tiles['ice'].set(f'{worst["clear_s"] / 60:.1f}', 'min', f'{st.ice_mm:g} mm at {st.ambient_c:g} °C,'
                                      f' {st.speed_kmh:g} km/h', 'good')
        else:
            self.tiles['ice'].set('—', '', 'no heater' if not hs else f'above 0 °C ({st.ambient_c:g} °C)')
        rows = []
        for d in res.devices:
            rows.append([f'{d["id"]} · {CL.KIND_LABEL[d["kind"]].lower()}', d['tag'] or '?',
                         '—' if d.get('cover') is None else f'{100 * d["cover"]:.0f} %', d.get('missed_where') or '',
                         _fmt(d.get('flow_ml_min'), 0), '—' if d.get('capture') is None else
                         f'{100 * d["capture"]:.0f} %', _fmt(d.get('clear_s') if d['kind'] == 'washer' else None, 2),
                         _fmt(d.get('power_w') or d.get('compressor_w'), 1),
                         _fmt(d.get('steady_c'), 1, '°C'), '—' if d['kind'] != 'heater' else
                         ('never' if d.get('clear_s') is None else f'{d["clear_s"] / 60:.1f} min')])
            if d['problem']:
                rows[-1][3] = d['problem']
        _fill(self.t_res, rows)
        _fill(self.t_clear, [[f'{r["device"]} · {r["tag"]}', r['label'], _fmt(r['clear_s'], 2),
                              _fmt(r['events_per_h'], 2), '—' if r['duty'] is None else f'{100 * r["duty"]:.2f} %']
                             for r in CL.clearing_table(res)])
        pb = getattr(self, 'power_res', None)
        ptxt = ('; '.join(f'{r[0].split(": ")[1]} {r[1]:.1f} W (peak {r[2]:.0f} W)' for r in res.power_rows)
                or 'No load.')
        if pb is not None:
            ptxt += (f'<br>With them the rig draws {pb.totals["steady_w"]:.0f} W steady; the auxiliary battery runs it '
                     f'{pb.totals["runtime_min"]:.0f} min with the engine off (core.power, the cleaning as other '
                     'loads).')
        ptxt += f'<br><span style="color:{THEME.tok.get("muted", "#888")}">{tot["pump_cite"]} {tot["valve_cite"]}</span>'
        self.power.setText(ptxt)
        _fill(self.t_harn, [[f'{h["device"]} {h["what"]}', h['from'], h['to'], f'{h["length_m"]:.2f}', h['route']]
                            for h in res.harness])
        for f in res.findings:
            w = StatusLine(f['severity'], f['message'])
            self.find_lay.insertWidget(self.find_lay.count() - 1, w)
            self.findings.append(w)
        n = len(res.devices)
        self.status.setText(f'{res.variant}: {n} device{"s" * (n != 1)} · ' + (
            'nothing to look at' if not res.findings else f'{len(res.findings)} finding'
                                                          f'{"s" * (len(res.findings) != 1)}'))

    def export(self, folder=None):
        res = self.result
        if res is None:
            return []
        if folder is None:
            folder = QFileDialog.getExistingDirectory(self, 'Export the cleaning layout into')
            if not folder:
                return []
        from ...core import provenance as PV
        paths = PV.stamp_paths(CL.export(res, self.ctx.store.project, folder), self.ctx.store.project, folder)
        self.ctx.msg(f'Exported the cleaning layout to {folder}', 5000)
        return paths
