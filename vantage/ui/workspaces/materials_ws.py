"""Analysis ▸ Materials: the vehicle's surface materials, what every radar looks through (loss, range, multipath
risk), the cameras behind glass (light, apparent shift, dashboard reflection) and the windscreen wipers' sweep.

The study (core.materials, core.wipers) takes a few tenths of a second on a full rig: it runs in a background thread
on a snapshot, again by itself a moment after each edit while the page shows (once when it shows again after edits
made elsewhere). Settings are saved with the project (analysis.materials) and every edit is one undo step; a part's
material and thickness are the vehicle's (mesh.Part.surface / surface_mm, undone with the vehicle).
"""
from __future__ import annotations

import math
import threading
import traceback

import numpy as np
from PyQt6.QtCore import QObject, QPointF, QRectF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPolygonF
from PyQt6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QFileDialog, QGridLayout, QHBoxLayout,
                             QHeaderView, QScrollArea, QSplitter, QTableWidget, QTableWidgetItem, QTabWidget,
                             QVBoxLayout, QWidget)

from ...core import materials as M
from ...core import wipers as W
from ..charts import ChartBase
from ..theme import STATUS, THEME, mono, sans
from ..widgets import ROW_H, ScrubSpinBox, SegmentedControl, StatusLine, ToggleSwitch, button, hline, label

RISK_TONE = {'low': 'good', 'moderate': 'warning', 'high': 'serious'}
SEV_WORD = {'good': 'OK', 'warning': 'Warning', 'serious': 'Serious', 'critical': 'Critical'}
MOUNT_WORD = {'exposed': 'nothing (in front of the body)', 'enclosed': 'the body, further out', 'blocked': 'blocked',
              'behind': '', 'deep': 'deep behind '}
DEBOUNCE_MS = 250

# (group, key, label, unit, decimals, step, scale, tip); scale: stored value × scale = shown value
FIELDS = (
    ('Radar', 'radar_ghz', 'Frequency', 'GHz', 1, 0.5, 1.0,
     'The radars\' carrier: 76–77 GHz for long-range radars, 77–81 GHz for corner radars'),
    ('Radar', 'loss_warn_db', 'Loss warning', 'dB', 1, 0.5, 1.0,
     'Two-way loss through the fascia from which a radar gets a warning (a typical design target is 3 dB: '
     '16 % less range)'),
    ('Radar', 'loss_serious_db', 'Loss serious', 'dB', 1, 0.5, 1.0, 'Two-way loss from which it is serious'),
    ('Radar', 'refl_warn_db', 'Reflection target', 'dB', 1, 1.0, 1.0,
     'The fascia should reflect less than this back into the radar (a typical target is -10 dB)'),
    ('Radar', 'min_gap_mm', 'Closest fascia', 'mm', 1, 0.5, 1.0,
     'A fascia nearer than this to the radar sits in its antenna\'s near field (about one wavelength)'),
    ('Automatic regions', 'fascia_depth_m', 'Bumper depth', 'm', 2, 0.05, 1.0,
     'A body modelled as one painted part: its front and rear ends this deep count as the plastic bumper fascia'),
    ('Automatic regions', 'fascia_top_m', 'Bumper top', 'm', 2, 0.05, 1.0,
     'Up to this height (0: 55 % of the body\'s height, at most 0.9 m)'),
    ('Glass', 'dash_reflectance', 'Dashboard', '%', 0, 1.0, 100.0,
     'Reflectance of the dashboard under the windscreen (a dark one about 5 %)'),
    ('Glass', 'road_reflectance', 'Road', '%', 0, 1.0, 100.0, 'Reflectance of the sunlit road (asphalt 7–15 %)'),
    ('Glass', 'glare_warn', 'Glare warning', '%', 0, 1.0, 100.0,
     'The glass reflecting more than this share of the dashboard into the image is a warning'),
    ('Glass', 'transmission_warn', 'Light warning', '%', 0, 1.0, 100.0,
     'The glass passing less than this along a camera\'s axis is a warning'),
    ('Glass', 'calib_m', 'Calibration at', 'm', 1, 0.5, 1.0,
     'Distance of a calibration target seen through the glass: the intrinsics absorb the glass\'s effect there'),
)
ARM_COLS = (('name', 'Wiper', None), ('pivot_w', 'Pivot across', 'm'), ('pivot_h', 'Pivot up', 'm'),
            ('arm_m', 'Arm', 'm'), ('blade_m', 'Blade', 'm'), ('park_deg', 'Rest', '°'), ('sweep_deg', 'Sweep', '°'))


# =========================================================================== the controller
class _Sig(QObject):
    done = pyqtSignal(int, object)
    failed = pyqtSignal(int, str)


def controller(ctx) -> 'MaterialsController':
    c = getattr(ctx, 'materials', None)
    if c is None:
        c = ctx.materials = MaterialsController(ctx)
    return c


class MaterialsController(QObject):
    """The project's materials study, computed in a background thread on a snapshot; kept up to date while the page
    shows (or the 3D view draws the wipers), lazily otherwise."""
    changed = pyqtSignal()

    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        self.result: M.MaterialsResult | None = None
        self.busy = False
        self.stale = True
        self.status = ''
        self.active = False
        self._gen = 0
        self._for = None                   # (project, revision) the result was made for
        self._lines = (None, [])           # (result, its wiper lines for the 3D view)
        self.sig = _Sig()
        self.sig.done.connect(self._done)
        self.sig.failed.connect(self._failed)
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(DEBOUNCE_MS)
        self.timer.timeout.connect(self.run)
        ctx.store.changed.connect(self._on_edit)
        ctx.store.project_replaced.connect(self._reset)

    @property
    def settings(self) -> M.MaterialSettings:
        return M.settings_of(self.ctx.store.project)

    def set_active(self, on: bool):
        self.active = on
        if on and (self.result is None or self.stale):
            self.timer.start()

    def _on_edit(self, *_):
        self.stale = True
        if self.active:
            self.timer.start()

    def _reset(self, *_):
        self._gen += 1
        self.result, self._for, self.stale, self.busy, self.status = None, None, True, False, ''
        self.changed.emit()
        if self.active:
            self.timer.start()

    def request(self):
        """Compute soon when there is no current result (the 3D view asks this way)."""
        if (self.result is None or self.stale) and not self.busy and not self.timer.isActive():
            self.timer.start()

    def run(self):
        st = self.ctx.store
        p = st.project
        if p.vehicle.mesh is None:
            self._gen += 1
            self.result, self.busy, self.stale = None, False, False
            self.status = 'No vehicle model: import one first.'
            self.changed.emit()
            return
        self._gen += 1
        gen = self._gen
        project = p.snapshot()
        key = (p, st.revision)
        eng = self.ctx.analysis.engine.share()
        self.busy = True
        self.status = 'Computing the materials study…'
        self.changed.emit()

        def work():
            try:
                res = M.run(project, eng, cancel=lambda: gen != self._gen)
                self.sig.done.emit(gen, (key, res))
            except RuntimeError:            # the window closed while computing
                pass
            except Exception as e:
                try:
                    self.sig.failed.emit(gen, f'{type(e).__name__}: {e}\n{traceback.format_exc()}')
                except RuntimeError:
                    pass
        threading.Thread(target=work, daemon=True).start()

    def _done(self, gen, out):
        if gen != self._gen:
            return
        self.busy = False
        key, res = out
        if res is None or key[0] is not self.ctx.store.project:
            self.status = '' if res is not None else 'Cancelled'
            self.changed.emit()
            return
        self.result, self._for = res, key
        self.stale = key[1] != self.ctx.store.revision
        m = res.metrics
        self.status = (f'{m.get("radars", 0)} radar{"s" * (m.get("radars", 0) != 1)} · '
                       f'{m.get("glass_cameras", 0)} camera{"s" * (m.get("glass_cameras", 0) != 1)} behind glass · '
                       f'{m.get("seconds", 0):.2f} s')
        self.changed.emit()
        if self.ctx.overlays.get('wipers'):
            self.ctx.overlays_changed.emit()          # the 3D view draws the new sweep
        if self.stale and self.active:
            self.timer.start()

    def _failed(self, gen, msg):
        if gen != self._gen:
            return
        self.busy = False
        self.status = 'Failed: ' + msg.splitlines()[0]
        self.changed.emit()


def scene_lines(ctx) -> list:
    """For the 3D view (View ▸ Overlays ▸ Wiper sweep): each wiper's swept area outlined on the windscreen and its
    blade at rest, from the last study (asks for one when there is none)."""
    from ..viewport3d import LineDraw
    ctl = controller(ctx)
    res = ctl.result
    if res is None or ctl._for is None or ctl._for[0] is not ctx.store.project:
        ctl.request()
        return []
    if ctl.stale:
        ctl.request()
    if res.wipers is None or res.wipers.windscreen is None:
        return []
    if ctl._lines[0] is not res:                       # carried onto the glass once per result
        mesh, _ = M._geometry(ctx.store.project, None)
        ctl._lines = (res, W.sweep_lines(res.wipers, mesh) if mesh is not None else [])
    out = []
    ink = THEME.color('ink', 0.9)
    for kind, i, P in ctl._lines[1]:
        out.append(LineDraw(P, ink if kind == 'sweep' else THEME.color('ink2'), 1.4 if kind == 'sweep' else 2.4,
                            dash=kind == 'sweep', dash_len=0.03))
    return out


# =========================================================================== widgets
def _qc(rgb) -> QColor:
    return QColor(int(rgb[0]), int(rgb[1]), int(rgb[2]))


class BeamMap(ChartBase):
    """One radar's two-way loss over its field: azimuth across (its left on the left, as seen from behind it),
    elevation up; dark red where something opaque or the vehicle blocks the ray; the zones outlined."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.title = 'Two-way loss over the field'
        self.rep = None
        self.setMinimumHeight(170)

    def set_report(self, rep):
        self.rep = rep
        self.tip = None
        self.update()

    def _rect(self):
        return QRectF(44, 24, max(self.width() - 44 - 70, 40), max(self.height() - 24 - 34, 40))

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        rep = self.rep
        self.paint_title(p, f'{rep["tag"]} · 0–10 dB' if rep else '')
        r = self._rect()
        if rep is None:
            p.setFont(sans(11))
            p.setPen(THEME.color('muted'))
            p.drawText(r, Qt.AlignmentFlag.AlignCenter, 'Pick a radar in the table')
            p.end()
            return
        mp = rep['map']
        az, el, L = mp['az'], mp['el'], mp['loss_db']
        rgb = M.loss_rgb(L)[::-1, ::-1].astype(np.uint8)          # rows: elevation up; columns: +az on the left
        h, w = rgb.shape[:2]
        rgba = np.dstack([rgb, np.full((h, w), 255, np.uint8)]).copy()
        img = QImage(rgba.data, w, h, 4 * w, QImage.Format.Format_RGBA8888).copy()
        p.drawImage(r, img)
        p.setPen(QPen(THEME.color('strong'), 1))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRect(r)
        A, E = float(az[-1]), float(el[-1])
        p.setPen(QPen(THEME.color('ink'), 1, Qt.PenStyle.DashLine))
        for z in rep['zones']:
            x0 = r.left() + (A - z['az_half_deg']) / (2 * A) * r.width() if A > 0 else r.left()
            x1 = r.left() + (A + z['az_half_deg']) / (2 * A) * r.width() if A > 0 else r.right()
            y0 = r.top() + (E - z['el_half_deg']) / (2 * E) * r.height() if E > 0 else r.top()
            y1 = r.top() + (E + z['el_half_deg']) / (2 * E) * r.height() if E > 0 else r.bottom()
            p.drawRect(QRectF(x0, y0, x1 - x0, y1 - y0))
        p.setFont(mono(10))
        p.setPen(THEME.color('muted'))
        p.drawText(QRectF(r.left(), r.bottom() + 2, 60, 14), Qt.AlignmentFlag.AlignLeft, f'{A:+.0f}°')
        p.drawText(QRectF(r.right() - 60, r.bottom() + 2, 60, 14), Qt.AlignmentFlag.AlignRight, f'{-A:+.0f}°')
        p.drawText(QRectF(r.left(), r.bottom() + 2, r.width(), 14), Qt.AlignmentFlag.AlignHCenter, 'azimuth')
        p.drawText(QRectF(0, r.top() - 7, r.left() - 4, 14), Qt.AlignmentFlag.AlignRight, f'{E:+.0f}°')
        p.drawText(QRectF(0, r.bottom() - 7, r.left() - 4, 14), Qt.AlignmentFlag.AlignRight, f'{-E:+.0f}°')
        # legend
        lx = r.right() + 14
        for k in range(60):
            v = 10.0 * (59 - k) / 59
            p.fillRect(QRectF(lx, r.top() + k * r.height() / 60, 10, r.height() / 60 + 1), _qc(M.loss_rgb(v)))
        p.setPen(THEME.color('muted'))
        p.drawText(QRectF(lx + 13, r.top() - 6, 50, 14), Qt.AlignmentFlag.AlignLeft, '10 dB')
        p.drawText(QRectF(lx + 13, r.bottom() - 8, 50, 14), Qt.AlignmentFlag.AlignLeft, '0 dB')
        p.fillRect(QRectF(lx, r.bottom() + 6, 10, 8), _qc(M.loss_rgb(np.inf)))
        p.drawText(QRectF(lx + 13, r.bottom() + 3, 60, 14), Qt.AlignmentFlag.AlignLeft, 'blocked')
        self.paint_tip(p)
        p.end()

    def mouseMoveEvent(self, e):
        rep = self.rep
        r = self._rect()
        pos = e.position()
        if rep is None or not r.contains(pos):
            if self.tip is not None:
                self.tip = None
                self.update()
            return
        mp = rep['map']
        az, el, L = mp['az'], mp['el'], mp['loss_db']
        i = int(np.clip((pos.x() - r.left()) / r.width() * len(az), 0, len(az) - 1))
        j = int(np.clip((pos.y() - r.top()) / r.height() * len(el), 0, len(el) - 1))
        a, b = float(az[::-1][i]), float(el[::-1][j])
        v = float(L[len(el) - 1 - j, len(az) - 1 - i])
        self.set_tip(pos, 'blocked' if not math.isfinite(v) else f'{v:.2f} dB two-way',
                     f'azimuth {a:+.1f}° · elevation {b:+.1f}°')
        self.update()

    def leaveEvent(self, e):
        self.tip = None
        self.update()


class GlassView(ChartBase):
    """The windscreen seen from inside, looking forward (the vehicle's left on the left): the glass, the area the
    wipers clear, each arm's sweep, its blade at rest and pivot, and the cameras' footprints on the glass (filled where
    wiped, red rings where not)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.title = 'Windscreen from inside'
        self.res = None
        self.empty = ''
        self._cams = []                         # (QPointF centre, camera record)
        self.setMinimumHeight(220)

    def set_result(self, wr, empty=''):
        self.res = wr
        self.empty = empty
        self.tip = None
        self.update()

    def _map(self):
        ws = self.res.windscreen
        w0, h0, cell, nw, nh = ws.grid
        r = QRectF(8, 24, max(self.width() - 16, 40), max(self.height() - 34, 40))
        s = min(r.width() / (nw * cell), r.height() / (nh * cell))
        ox = r.left() + (r.width() - nw * cell * s) / 2
        oy = r.top() + (r.height() - nh * cell * s) / 2
        wmax, hmax = w0 + nw * cell, h0 + nh * cell

        def to(Q):
            Q = np.atleast_2d(np.asarray(Q, float))
            return np.c_[ox + (wmax - Q[:, 0]) * s, oy + (hmax - Q[:, 1]) * s]
        return to, s, (ox, oy)

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        wr = self.res
        ws = wr.windscreen if wr is not None else None
        self.paint_title(p, f'{wr.wiped_share * 100:.0f} % wiped' if ws is not None else '')
        if ws is None:
            p.setFont(sans(11))
            p.setPen(THEME.color('muted'))
            p.drawText(QRectF(0, 24, self.width(), self.height() - 24), Qt.AlignmentFlag.AlignCenter,
                       'No windscreen found in the model' if wr is not None else self.empty)
            p.end()
            return
        to, s, (ox, oy) = self._map()
        w0, h0, cell, nw, nh = ws.grid
        glass = THEME.color('glass')
        wiped = QColor(THEME.color('ink'))
        wiped.setAlphaF(0.20)
        rgba = np.zeros((nh, nw, 4), np.uint8)
        g = glass
        rgba[ws.mask] = (g.red(), g.green(), g.blue(), 150)
        rgba[wr.wiped] = (wiped.red(), wiped.green(), wiped.blue(), 110)
        rgba = np.ascontiguousarray(rgba[::-1, ::-1])
        img = QImage(rgba.data, nw, nh, 4 * nw, QImage.Format.Format_RGBA8888).copy()
        p.drawImage(QRectF(ox, oy, nw * cell * s, nh * cell * s), img)
        p.setPen(QPen(THEME.color('ink2'), 1))
        p.setBrush(Qt.BrushStyle.NoBrush)
        for poly in ws.outline:
            P = to(poly)
            p.drawPolygon(QPolygonF([QPointF(float(x), float(y)) for x, y in P]))
        for a in wr.arms:
            P = to(a.outline(40))
            p.setPen(QPen(THEME.color('ink', 0.7), 1, Qt.PenStyle.DashLine))
            p.drawPolygon(QPolygonF([QPointF(float(x), float(y)) for x, y in P]))
            b = to(a.blade_at(a.park_deg % 360.0))
            p.setPen(QPen(THEME.color('ink'), 2.5))
            p.drawLine(QPointF(*b[0]), QPointF(*b[1]))
            c = to([[a.pivot_w, a.pivot_h]])[0]
            p.setBrush(THEME.color('ink'))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawEllipse(QPointF(*c), 3.5, 3.5)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.setFont(sans(10))
            p.setPen(THEME.color('ink2'))
            p.drawText(QPointF(c[0] + 6, c[1] - 4), a.name)
        self._cams = []
        bad = QColor(STATUS['critical'])
        for c in wr.cameras:
            col = THEME.cam(c['slot'])
            P = to(c['points'])
            for (x, y), ok in zip(P, c['wiped']):
                if ok:
                    p.setPen(Qt.PenStyle.NoPen)
                    p.setBrush(col)
                    p.drawEllipse(QPointF(x, y), 2.2, 2.2)
                else:
                    p.setBrush(Qt.BrushStyle.NoBrush)
                    p.setPen(QPen(bad, 1.2))
                    p.drawEllipse(QPointF(x, y), 2.6, 2.6)
            x1, y0 = float(P[:, 0].max()), float(P[:, 1].min())
            p.setFont(mono(11, QFont.Weight.DemiBold))
            p.setPen(col)
            p.drawText(QPointF(x1 + 6, y0 + 10), f'{c["tag"]} {c["share_wiped"] * 100:.0f} %')
            ctr = to([c['centre']])[0]
            self._cams.append((QPointF(*ctr), c))
        self.paint_tip(p)
        p.end()

    def mouseMoveEvent(self, e):
        pos = e.position()
        hit = min(self._cams, key=lambda t: (t[0] - pos).manhattanLength(), default=None)
        if hit is None or (hit[0] - pos).manhattanLength() > 60:
            if self.tip is not None:
                self.tip = None
                self.update()
            return
        c = hit[1]
        d = c['dirt_unwiped']
        self.set_tip(pos, f'{c["tag"]} · {c["share_wiped"] * 100:.0f} % wiped',
                     f'not wiped: {c["where_unwiped"]}' if c['where_unwiped'] else '',
                     f'dirt there {d["score"]:.2f} ({d["class"]})' if d else '')
        self.update()

    def leaveEvent(self, e):
        self.tip = None
        self.update()


def _table(cols, tips=None):
    t = QTableWidget(0, len(cols))
    for i, c in enumerate(cols):
        it = QTableWidgetItem(c)
        if tips and i < len(tips) and tips[i]:
            it.setToolTip(tips[i])
        t.setHorizontalHeaderItem(i, it)
    t.verticalHeader().setVisible(False)
    t.verticalHeader().setDefaultSectionSize(ROW_H)
    t.horizontalHeader().setFixedHeight(ROW_H)
    t.setShowGrid(False)
    t.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    t.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    t.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
    hh = t.horizontalHeader()
    for c in range(len(cols)):
        hh.setSectionResizeMode(c, QHeaderView.ResizeMode.ResizeToContents)
    hh.setStretchLastSection(True)
    return t


def _fit(t, rows, most=8):
    """A table just tall enough for its rows (scrolling beyond `most`)."""
    t.setFixedHeight(ROW_H * (min(max(rows, 1), most) + 1) + 6)


def _f(v, fmt='{:.1f}', none='—'):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return none
    return fmt.format(x) if math.isfinite(x) else none


# =========================================================================== the page
class MaterialsPage(QWidget):
    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        self.ctl = controller(ctx)
        self._shown = None
        self._syncing = False
        self.setObjectName('base')
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        split = QSplitter(Qt.Orientation.Horizontal)
        split.setHandleWidth(2)
        split.addWidget(self._left())
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.addTab(self._radar_tab(), 'Radars')
        self.tabs.addTab(self._glass_tab(), 'Glass and wipers')
        self.tabs.addTab(self._parts_tab(), 'Parts')
        self.tabs.addTab(self._findings_tab(), 'Findings')
        split.addWidget(self.tabs)
        split.setSizes([330, 1270])
        lay.addWidget(split)
        self.ctl.changed.connect(self.refresh)
        ctx.store.changed.connect(self._on_store)
        ctx.store.project_replaced.connect(self._on_store)
        THEME.changed.connect(self._retheme)          # a bound slot: disconnected when the page goes
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
        top.addWidget(label('Materials'))
        top.addStretch(1)
        top.addWidget(label('radar, glass, wipers', muted=True, small=True))
        ll.addLayout(top)
        ll.addWidget(hline())
        ll.addWidget(label('What each radar looks through and what it costs in range, the cameras behind glass, and '
                           'the wipers\' sweep. Material figures are typical values from the literature (see the '
                           'export): measure your fascia and set its class and thickness under Parts.',
                           muted=True, small=True, wrap=True))
        ll.addSpacing(6)
        ll.addWidget(label('Vehicle paint', bold=True))
        self.paint = SegmentedControl([(k, {'none': 'None', 'repaired': 'Repair'}.get(k, M.FINISH_LABEL[k]))
                                       for k in M.FINISHES], 'solid')
        self.paint.setToolTip('The paint on the painted parts (body and bumpers): metallic and pearl basecoats '
                              'reflect 77 GHz far more than a solid one; a repaired (repainted) metallic bumper most')
        self.paint.changed.connect(lambda k: self._set(f'{M.FINISH_LABEL[k].lower()} paint', paint=k))
        ll.addWidget(self.paint)
        self.fields = {}
        group, grid = None, None
        for g, key, text, unit, dec, step, scale, tip in FIELDS:
            if g != group:
                if grid is not None:
                    ll.addLayout(grid)
                group = g
                ll.addSpacing(6)
                ll.addWidget(label(g, bold=True))
                if g == 'Automatic regions':
                    self.auto = ToggleSwitch('Split a one-part body')
                    self.auto.setToolTip('A body modelled as one painted part: its bumper ends are the plastic fascia '
                                         'and, without forward-facing glass, the band between the side windows and '
                                         'the bonnet is the windscreen')
                    self.auto.clicked.connect(lambda on: self._set('automatic regions ' + ('on' if on else 'off'),
                                                                   auto_regions=on))
                    ll.addWidget(self.auto)
                grid = QGridLayout()
                grid.setVerticalSpacing(2)
                grid.setHorizontalSpacing(6)
                grid.setColumnStretch(2, 1)
            f = ScrubSpinBox('', 'plain', dec, unit, step=step, width=110)
            f.setToolTip(tip)
            f.committed.connect(lambda v, k=key, s=scale, t=text, u=unit: self._set(f'{t} {v:g} {u}', **{k: v / s}))
            a = label(text, ink2=True)
            a.setFixedWidth(118)
            a.setToolTip(tip)
            i = grid.rowCount()
            grid.addWidget(a, i, 0)
            grid.addWidget(f, i, 1)
            self.fields[key] = (f, scale)
        ll.addLayout(grid)
        ll.addSpacing(6)
        ll.addWidget(label('Wipers', bold=True))
        self.w_on = ToggleSwitch('Windscreen wipers')
        self.w_on.setToolTip('Map the wipers\' sweep on the windscreen and check the cameras behind it')
        self.w_on.clicked.connect(lambda on: self._set_wipers('wipers ' + ('on' if on else 'off'), enabled=on))
        ll.addWidget(self.w_on)
        self.layout_seg = SegmentedControl([(k, W.LAYOUT_LABEL[k]) for k in W.LAYOUTS], 'tandem')
        self.layout_seg.setToolTip('Typical arms sized from the windscreen; Custom: the arms in the table')
        self.layout_seg.changed.connect(self._layout)
        ll.addWidget(self.layout_seg)
        self.drive = SegmentedControl([(k, k.upper()) for k in W.DRIVES], 'lhd')
        self.drive.setToolTip('Left- or right-hand drive: the driver\'s wiper sweeps the driver\'s side')
        self.drive.changed.connect(lambda k: self._set_wipers(W.DRIVE_LABEL[k].lower(), drive=k))
        ll.addWidget(self.drive)
        g2 = QGridLayout()
        g2.setHorizontalSpacing(6)
        self.min_wiped = ScrubSpinBox('', 'plain', 0, '%', step=1.0, width=110)
        self.min_wiped.setToolTip('A camera with less of its view on the windscreen inside the sweep gets a warning')
        self.min_wiped.committed.connect(lambda v: self._set_wipers(f'wiped share {v:g} %', min_wiped=v / 100.0))
        a = label('Camera wiped', ink2=True)
        a.setFixedWidth(118)
        g2.addWidget(a, 0, 0)
        g2.addWidget(self.min_wiped, 0, 1)
        g2.setColumnStretch(2, 1)
        ll.addLayout(g2)
        ll.addStretch(1)
        self.status = label('', muted=True, small=True, wrap=True)
        ll.addWidget(self.status)
        bl = QHBoxLayout()
        self.b_defaults = button('Defaults', 'ghost', None, 'Back to the default materials settings (part materials '
                                                            'set under Parts stay)')
        self.b_defaults.clicked.connect(self._defaults)
        bl.addWidget(self.b_defaults)
        bl.addStretch(1)
        self.b_export = button('Export…', None, 'export', 'Materials, radars, glass and wipers as JSON, CSV and PNG')
        self.b_export.clicked.connect(lambda: self.export())
        bl.addWidget(self.b_export)
        ll.addLayout(bl)
        sc = QScrollArea()
        sc.setWidgetResizable(True)
        sc.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        sc.setWidget(w)
        sc.setMinimumWidth(300)
        return sc

    def _radar_tab(self):
        w = QWidget()
        w.setObjectName('panel')
        l = QVBoxLayout(w)
        l.setContentsMargins(10, 8, 10, 8)
        l.setSpacing(6)
        l.addWidget(label('What each radar looks through', bold=True))
        self.radars = _table(('Radar', 'Looks through', 'Gap', 'Thickness', 'Incidence', 'Loss', 'Range', 'Reflection',
                              'Multipath'),
                             ('', 'The surface its beam leaves the vehicle through (the paint over it in brackets)',
                              'Radar to that surface along its boresight, mm (and in half wavelengths in air)',
                              'The fascia\'s substrate, mm (and in half wavelengths inside it: least reflection at '
                              'whole numbers)', 'Angle between the boresight and the surface\'s normal',
                              'Two-way loss, the first zone\'s mean (dB)',
                              'Each zone\'s range on a car, derated by its loss: range ∝ (two-way transmission)^¼',
                              'Power the fascia reflects back into the radar along its boresight (dB)',
                              'Ghost and multipath risk: reflection, cavity resonance, metal nearby'))
        self.radars.itemSelectionChanged.connect(self._radar_picked)
        _fit(self.radars, 1)
        l.addWidget(self.radars)
        self.beam = BeamMap()
        l.addWidget(self.beam, 1)
        self.radar_text = label('', wrap=True)
        self.radar_text.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        l.addWidget(self.radar_text)
        return w

    def _glass_tab(self):
        w = QWidget()
        w.setObjectName('panel')
        l = QVBoxLayout(w)
        l.setContentsMargins(10, 8, 10, 8)
        l.setSpacing(6)
        l.addWidget(label('Cameras behind glass', bold=True))
        self.cams = _table(('Camera', 'Glass', 'Incidence', 'Light', 'Centre shift', 'Error at 3 m', 'Dashboard',
                            'Wiped', 'Glare shield'),
                           ('', 'The glass its view leaves through', 'Angle between its axis and the glass\'s normal',
                            'Visible light the glass passes along its axis (and the stops lost)',
                            'Where its centre appears through the glass (the best single shift of all its rays), mm',
                            'Pixel error left at 3 m after that shift (rms)',
                            'Share of the dashboard\'s brightness the glass reflects into the image',
                            'Share of its view on the windscreen the wipers clear',
                            'A black glare shield under the camera stops the dashboard\'s reflection'))
        hh = self.cams.horizontalHeader()
        hh.setStretchLastSection(False)
        hh.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        _fit(self.cams, 1)
        l.addWidget(self.cams)
        split = QSplitter(Qt.Orientation.Horizontal)
        split.setHandleWidth(2)
        self.glass = GlassView()
        split.addWidget(self.glass)
        aw = QWidget()
        al = QVBoxLayout(aw)
        al.setContentsMargins(0, 0, 0, 0)
        al.setSpacing(4)
        self.arms_head = label('Wipers', bold=True)
        al.addWidget(self.arms_head)
        self.arms = _table([c[1] for c in ARM_COLS],
                           ['', 'Pivot across the glass from the middle of its bottom edge, + to the vehicle\'s left',
                            'Pivot up the glass from its bottom edge (below it: in the cowl)', 'Pivot to the blade\'s '
                            'centre', 'Blade length (along the arm)', 'The arm at rest, degrees from the vehicle\'s '
                            'left towards the top', 'How far it turns from rest towards the top'])
        self.arms.setEditTriggers(QAbstractItemView.EditTrigger.DoubleClicked |
                                  QAbstractItemView.EditTrigger.EditKeyPressed |
                                  QAbstractItemView.EditTrigger.AnyKeyPressed)
        self.arms.itemChanged.connect(self._arm_edited)
        al.addWidget(self.arms, 1)
        self.ws_text = label('', muted=True, small=True, wrap=True)
        al.addWidget(self.ws_text)
        split.addWidget(aw)
        split.setSizes([620, 520])
        l.addWidget(split, 1)
        return w

    def _parts_tab(self):
        w = QWidget()
        w.setObjectName('panel')
        l = QVBoxLayout(w)
        l.setContentsMargins(10, 8, 10, 8)
        l.setSpacing(6)
        l.addWidget(label('Material of each part', bold=True))
        self.parts = _table(('Part', 'Role', 'Imported as', 'Material', 'Thickness', 'Chosen', 'Area'),
                            ('', 'Its role in the vehicle model', 'The material name in the imported file',
                             'The surface material (Automatic: from its role and name, and by position on a one-part '
                             'body)', 'Thickness, mm (empty: the material\'s typical one)', 'How the material was '
                             'chosen', 'Surface area, m²'))
        self.parts.setEditTriggers(QAbstractItemView.EditTrigger.DoubleClicked |
                                   QAbstractItemView.EditTrigger.EditKeyPressed |
                                   QAbstractItemView.EditTrigger.AnyKeyPressed)
        self.parts.itemChanged.connect(self._part_edited)
        _fit(self.parts, 4)
        l.addWidget(self.parts)
        self.regions_text = label('', muted=True, small=True, wrap=True)
        l.addWidget(self.regions_text)
        self.lib = _table(('Material', 'Kind', 'Layers', 'Two-way', 'Reflection', 'Note'),
                          ('', '', 'Inner to outer at the typical thickness; painted ones take the vehicle\'s paint',
                           'Two-way loss at normal incidence, with the vehicle\'s paint (dB)',
                           'Reflection at normal incidence (dB)', 'Where the figures come from'))
        self.lib.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        l.addWidget(label('Library (typical values)', bold=True))
        l.addWidget(self.lib, 1)
        return w

    def _findings_tab(self):
        w = QWidget()
        w.setObjectName('panel')
        l = QVBoxLayout(w)
        l.setContentsMargins(10, 8, 10, 8)
        l.setSpacing(4)
        self.find_host = QWidget()
        self.find_lay = QVBoxLayout(self.find_host)
        self.find_lay.setContentsMargins(0, 0, 0, 0)
        self.find_lay.setSpacing(4)
        self.find_lay.addStretch(1)
        sc = QScrollArea()
        sc.setWidgetResizable(True)
        sc.setWidget(self.find_host)
        l.addWidget(sc, 1)
        self.limits_text = label(M.ASSUMPTIONS + ' ' + M.LIMITS_TEXT + ' ' + W.ASSUMPTIONS, muted=True, small=True,
                                 wrap=True)
        l.addWidget(self.limits_text)
        self.findings: list[StatusLine] = []
        return w

    # ------------------------------------------------------------------ edits
    @property
    def settings(self) -> M.MaterialSettings:
        return M.settings_of(self.ctx.store.project)

    def _store(self, st: M.MaterialSettings, text) -> bool:
        new = M.MaterialSettings.from_dict(st.to_dict()).to_dict()
        if new == self.settings.to_dict():
            self._sync_fields()                # a clamped value: show what is stored
            return False

        def fn(p):
            p.analysis.materials = new
        self.ctx.store.edit('analysis', fn, f'Materials · {text}')
        return True

    def _set(self, text, **kw) -> bool:
        st = self.settings
        for k, v in kw.items():
            setattr(st, k, v)
        return self._store(st, text)

    def _set_wipers(self, text, **kw) -> bool:
        st = self.settings
        ws = W.WiperSettings.from_dict(st.wipers)
        for k, v in kw.items():
            setattr(ws, k, v)
        st.wipers = ws.to_dict()
        return self._store(st, text)

    def _layout(self, k):
        if k == 'custom':
            if W.WiperSettings.from_dict(self.settings.wipers).arms:      # your arms from before: back to them
                return self._set_wipers('custom wipers', layout='custom')
            res = self.ctl.result
            arms = res.wipers.arms if res is not None and res.wipers is not None else []
            if not arms:
                self._sync_fields()
                self.ctx.msg('No windscreen to size wipers from: import a model with glass.', 4000)
                return False
            return self._set_wipers('custom wipers', layout='custom', arms=[a.to_dict() for a in arms])
        return self._set_wipers(f'{W.LAYOUT_LABEL[k].lower()} wipers', layout=k)

    def _arm_edited(self, item):
        if self._syncing:
            return
        res = self.ctl.result
        if res is None or res.wipers is None:
            return
        r, c = item.row(), item.column()
        arms = [a.to_dict() for a in res.wipers.arms]
        if not 0 <= r < len(arms):
            return
        key, name, unit = ARM_COLS[c]
        if key == 'name':
            arms[r]['name'] = item.text().strip() or arms[r]['name']
        else:
            from ...core.mathutil import parse_number
            try:
                angle = unit == '°'
                v = float(parse_number(item.text(), 'angle' if angle else 'length', 'deg' if angle else 'm'))
            except Exception:
                v = float('nan')
            if not math.isfinite(v):
                self._fill_arms(res)
                return
            arms[r][key] = v
        if not self._set_wipers(f'{arms[r]["name"]} {name.lower()}', layout='custom', arms=arms):
            self._fill_arms(res)

    def _part_edited(self, item):
        if self._syncing or item.column() != 4:
            return
        i = item.row()
        m = self.ctx.store.project.vehicle.mesh
        if m is None or not 0 <= i < len(m.parts):
            return
        t = item.text().strip().replace('mm', '').strip()
        try:
            mm = float(t) if t else 0.0
        except ValueError:
            mm = float('nan')
        if not math.isfinite(mm) or mm < 0 or mm > 100:
            self._fill_parts(self.ctl.result)
            return
        if abs(mm - m.parts[i].surface_mm) < 1e-9:
            return
        name = m.parts[i].name

        def fn(p, i=i, mm=mm):
            p.vehicle.mesh.parts[i].surface_mm = mm
        self.ctx.store.edit('vehicle', fn, f'Material thickness of {name}: ' + (f'{mm:g} mm' if mm else 'typical'))

    def _part_class(self, i, cid):
        m = self.ctx.store.project.vehicle.mesh
        if self._syncing or m is None or not 0 <= i < len(m.parts) or m.parts[i].surface == cid:
            return
        name = m.parts[i].name

        def fn(p, i=i, cid=cid):
            p.vehicle.mesh.parts[i].surface = cid
        self.ctx.store.edit('vehicle', fn, f'Material of {name}: ' + (M.CLASSES[cid].name if cid else 'automatic'))

    def _shield(self, uid, on):
        st = self.settings
        s = set(st.glare_shield)
        (s.add if on else s.discard)(uid)
        st.glare_shield = sorted(s)
        cam = self.ctx.store.project.variant.cam(uid)
        self._store(st, f'glare shield {"on" if on else "off"} for {cam.tag if cam else uid}')

    def _defaults(self):
        if self.ctx.store.project.analysis.materials:
            def fn(p):
                p.analysis.materials = {}
            self.ctx.store.edit('analysis', fn, 'Materials · defaults')

    # ------------------------------------------------------------------ state
    def _on_store(self, *_):
        self._sync_fields()

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

    def _sync_fields(self):
        self._syncing = True
        try:
            st = self.settings
            d = st.to_dict()
            for key, (f, scale) in self.fields.items():
                f.set_value(d[key] * scale)
            self.paint.set_value(st.paint)
            for sw, on in ((self.auto, st.auto_regions),):
                sw.blockSignals(True)
                sw.setChecked(on)
                sw.blockSignals(False)
            ws = W.WiperSettings.from_dict(st.wipers)
            self.w_on.blockSignals(True)
            self.w_on.setChecked(ws.enabled)
            self.w_on.blockSignals(False)
            self.layout_seg.set_value(ws.layout)
            self.drive.set_value(ws.drive)
            self.min_wiped.set_value(ws.min_wiped * 100)
        finally:
            self._syncing = False

    def refresh(self):
        ctl = self.ctl
        res = ctl.result
        self.status.setText(ctl.status + (' · updating…' if ctl.busy and res is not None else ''))
        self.b_export.setEnabled(res is not None)
        if res is self._shown:
            return
        self._shown = res
        self._fill_radars(res)
        self._fill_cams(res)
        self.glass.set_result(res.wipers if res is not None else None,
                              'Wipers are switched off' if res is not None else ctl.status or 'Computing…')
        self._fill_arms(res)
        self._fill_parts(res)
        self._fill_findings(res)

    # ------------------------------------------------------------------ filling
    def _item(self, text, num=False, color=None, tip=None, bold=False, editable=False, swatch=None):
        it = QTableWidgetItem(text)
        if num:
            it.setFont(mono(12, QFont.Weight.DemiBold if bold else None))
            it.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        if color is not None:
            it.setForeground(color)
        if tip:
            it.setToolTip(tip)
        if swatch is not None:
            it.setData(Qt.ItemDataRole.DecorationRole, swatch)
        flags = it.flags()
        it.setFlags(flags | Qt.ItemFlag.ItemIsEditable if editable else flags & ~Qt.ItemFlag.ItemIsEditable)
        return it

    def _fill_radars(self, res):
        t = self.radars
        rows = res.radars if res is not None else []
        keep = self._picked_radar()
        t.blockSignals(True)
        t.setRowCount(len(rows))
        _fit(t, len(rows))
        st = res.settings if res is not None else M.MaterialSettings()
        radar_col = QColor('#a0743c')
        for i, r in enumerate(rows):
            z0 = r['zones'][0]
            through = MOUNT_WORD.get(r['mount'], '') + (r['class_name'] or '')
            if r['mount'] in ('exposed', 'enclosed'):
                through = MOUNT_WORD[r['mount']]
            if r['paint'] != 'none' and r['class_name']:
                through += f' ({M.FINISH_LABEL.get(r["paint"], r["paint"]).lower()})'
            t.setItem(i, 0, self._item(r['tag'], swatch=radar_col, tip=r['model']))
            t.setItem(i, 1, self._item(through, tip='; '.join(r['reasons'])))
            gap = r['gap_mm']
            t.setItem(i, 2, self._item(_f(gap, '{:.1f} mm') + (f'  {r["gap_half_waves"]:.2f} λ/2'
                                                                if r['gap_half_waves'] is not None else ''), True))
            t.setItem(i, 3, self._item(_f(r['thickness_mm'] or None, '{:.2f} mm') +
                                       (f'  {r["half_waves"]:.2f}' if r['half_waves'] is not None else ''), True))
            t.setItem(i, 4, self._item(_f(r['incidence_deg'], '{:.1f}°'), True))
            loss = z0['loss_db']
            sev = 'good' if not math.isfinite(loss) or loss < st.loss_warn_db else \
                'warning' if loss < st.loss_serious_db else 'serious'
            t.setItem(i, 5, self._item('blocked' if not math.isfinite(loss) else f'{loss:.2f} dB', True,
                                       QColor(STATUS[sev]) if sev != 'good' else None))
            t.setItem(i, 6, self._item(' · '.join(f'{z["derated_m"]:.0f}' for z in r['zones']) + ' m', True,
                                       tip='\n'.join(f'zone ±{z["az_half_deg"]:g}°: {z["range_m"]:.0f} → '
                                                     f'{z["derated_m"]:.0f} m ({(z["factor"] - 1) * 100:+.0f} %)'
                                                     for z in r['zones'])))
            t.setItem(i, 7, self._item(_f(r['reflection_db'], '{:.1f} dB'), True))
            t.setItem(i, 8, self._item(r['risk'], color=QColor(STATUS[RISK_TONE[r['risk']]])
                                       if r['risk'] != 'low' else None, tip='\n'.join(r['reasons'])))
        t.blockSignals(False)
        tags = [r['tag'] for r in rows]
        pick = tags.index(keep) if keep in tags else (0 if rows else -1)
        if pick >= 0:
            t.selectRow(pick)
        self._radar_picked()

    def _picked_radar(self):
        r = self.radars.currentRow()
        it = self.radars.item(r, 0) if r >= 0 else None
        return it.text() if it is not None else None

    def _radar_picked(self):
        res = self.ctl.result
        tag = self._picked_radar()
        rep = res.radar(tag) if (res is not None and tag) else None
        self.beam.set_report(rep)
        if rep is None:
            self.radar_text.setText('No radar in this variant.' if res is not None and not res.radars else '')
            return
        parts = []
        if rep['mount'] in ('behind', 'deep') and rep['cavity']:
            cav = rep['cavity']
            parts.append(f'Cavity between the radar and the fascia: {cav["buildup_db"]:+.1f} dB build-up at this gap '
                         f'({cav["min_db"]:+.1f} to {cav["max_db"]:+.1f} dB as the gap changes by λ/2 = '
                         f'{cav["wavelength_mm"] / 2:.2f} mm); between resonances at {cav["best_gap_mm"]:.1f} mm.')
        bt = rep['best_thickness']
        if bt is not None:
            parts.append(f'Least reflection near this thickness: {bt["thickness_mm"]:.2f} mm '
                         f'({bt["reflection_db"]:.1f} dB, {bt["loss_db"]:.2f} dB two-way).')
        mn = rep['metal_near']
        if mn is not None:
            parts.append(f'Nearest metal: {M._lc(mn["name"])} {mn["distance_mm"]:.0f} mm away, '
                         + ('inside the field' if mn['outside_deg'] <= 0 else f'{mn["outside_deg"]:.0f}° beside it')
                         + f' ({mn["where"]}).')
        if rep['blocked_by']:
            parts.append('Blocked: ' + ', '.join(f'{"the vehicle" if k == "vehicle" else M._lc(M.CLASSES[k].name)} '
                                                 f'{v * 100:.0f} %' for k, v in rep['blocked_by'].items()) + '.')
        if rep['reasons']:
            parts.append(f'Multipath risk {rep["risk"]}: ' + '; '.join(rep['reasons']) + '.')
        self.radar_text.setText(' '.join(parts) or f'{rep["tag"]} looks through nothing near it.')

    def _fill_cams(self, res):
        t = self.cams
        rows = res.cameras if res is not None else []
        wiped = {c['uid']: c['share_wiped'] for c in (res.wipers.cameras if res is not None and res.wipers else [])}
        t.setRowCount(len(rows))
        _fit(t, len(rows), 4)
        st = res.settings if res is not None else M.MaterialSettings()
        for i, c in enumerate(rows):
            t.setItem(i, 0, self._item(c['tag'], swatch=THEME.cam(c['slot'])))
            t.setItem(i, 1, self._item(c['class_name'] + f' · {c["thickness_mm"]:.2f} mm'))
            t.setItem(i, 2, self._item(f'{c["incidence_deg"]:.1f}°', True,
                                       tip=f'{c["incidence_range_deg"][0]:.0f}–{c["incidence_range_deg"][1]:.0f}° '
                                           'over the image'))
            low = c['transmission'] < st.transmission_warn
            t.setItem(i, 3, self._item(f'{c["transmission"] * 100:.0f} %  −{c["stops_lost"]:.2f} EV', True,
                                       QColor(STATUS['warning']) if low else None,
                                       tip=f'lowest over the image {c["transmission_min"] * 100:.0f} %'))
            sh = c['apparent_shift_mm']
            t.setItem(i, 4, self._item(f'{c["apparent_shift_norm_mm"]:.2f} mm', True,
                                       tip=f'x {sh[0]:+.2f} · y {sh[1]:+.2f} · z {sh[2]:+.2f} mm (vehicle frame)'))
            e3 = c['errors']['3']
            ie = c['intrinsics_equivalent']
            t.setItem(i, 5, self._item(f'{e3["fitted_rms_px"]:.2f} px', True,
                                       tip=f'without the shift {e3["naive_rms_px"]:.2f} px rms; at 1 m '
                                           f'{c["errors"]["1"]["fitted_rms_px"]:.2f} px, at 10 m '
                                           f'{c["errors"]["10"]["fitted_rms_px"]:.2f} px'
                                           + (f'\ncalibrated through the glass at {ie["distance_m"]:g} m the '
                                              f'intrinsics absorb {ie["focal_ppm"]:+.0f} ppm focal length and '
                                              f'({ie["principal_shift_px"][0]:+.2f}, '
                                              f'{ie["principal_shift_px"][1]:+.2f})'
                                              ' px principal point' if ie else '')))
            glare = c['dash_reflectance'] > st.glare_warn and not c['shield']
            t.setItem(i, 6, self._item(f'{c["dash_reflectance"] * 100:.0f} %', True,
                                       QColor(STATUS['warning']) if glare else None,
                                       tip=f'{c["dash_share"] * 100:.0f} % of the view; about '
                                           f'{(1 - c["contrast_kept"]) * 100:.0f} % contrast lost on a sunlit road'))
            sw = wiped.get(c['uid'])
            bad = sw is not None and sw < W.WiperSettings.from_dict(st.wipers).min_wiped
            t.setItem(i, 7, self._item('—' if sw is None else f'{sw * 100:.0f} %', True,
                                       QColor(STATUS['serious' if sw is not None and sw < 0.5 else 'warning'])
                                       if bad else None))
            box = QCheckBox()
            box.setChecked(c['shield'])
            box.setToolTip('A black, textured glare shield is fitted under this camera')
            box.toggled.connect(lambda on, u=c['uid']: self._shield(u, on))
            t.setCellWidget(i, 8, box)
        if not rows:
            t.setRowCount(0)

    def _fill_arms(self, res):
        wr = res.wipers if res is not None else None
        self._syncing = True
        try:
            t = self.arms
            arms = wr.arms if wr is not None else []
            t.setRowCount(len(arms))
            for i, a in enumerate(arms):
                d = a.to_dict()
                for c, (key, _, unit) in enumerate(ARM_COLS):
                    v = d[key]
                    text = v if key == 'name' else (f'{v:.1f}' if unit == '°' else f'{v:.3f}')
                    t.setItem(i, c, self._item(str(text), key != 'name', editable=True))
            if wr is None:
                self.arms_head.setText('Wipers (switched off)' if res is not None else 'Wipers')
                self.ws_text.setText('')
            else:
                ws = wr.windscreen
                self.arms_head.setText(('Your wipers' if wr.arms_source == 'yours' else
                                        f'Typical {W.LAYOUT_LABEL[wr.settings.layout].lower()} wipers') +
                                       ' · double-click a value to edit')
                self.ws_text.setText('No windscreen: the model has no forward-facing glass and no body band between '
                                     'side windows and a bonnet.' if ws is None else
                                     f'Windscreen: {ws.width_m:.2f} m across the bottom, {ws.height_m:.2f} m up the '
                                     f'glass, {ws.incline_deg:.0f}° from horizontal, {ws.area_m2:.2f} m² '
                                     f'({ws.source}). '
                                     f'The wipers clear {wr.wiped_share * 100:.0f} % of it. ' + W.ASSUMPTIONS)
        finally:
            self._syncing = False

    def _fill_parts(self, res):
        self._syncing = True
        try:
            t = self.parts
            m = self.ctx.store.project.vehicle.mesh
            infos = {pi['index']: pi for pi in (res.parts if res is not None else [])}
            parts = m.parts if m is not None else []
            t.setRowCount(len(parts))
            _fit(t, len(parts), 10)
            for i, part in enumerate(parts):
                info = infos.get(i, {})
                t.setItem(i, 0, self._item(part.name))
                t.setItem(i, 1, self._item(part.role))
                t.setItem(i, 2, self._item(part.material or '—'))
                box = QComboBox()
                auto = info.get('classes') or []
                auto_text = M.CLASSES[auto[0]].name if len(auto) == 1 else \
                    f'{len(auto)} by position' if auto else '—'
                box.addItem(f'Automatic · {auto_text}' if not part.surface else 'Automatic', '')
                box.setItemData(0, ', '.join(M.CLASSES[c].name for c in auto), Qt.ItemDataRole.ToolTipRole)
                for cid, mc in M.CLASSES.items():
                    box.addItem(mc.name, cid)
                k = box.findData(part.surface if part.surface in M.CLASSES else '')
                box.setCurrentIndex(max(k, 0))
                box.setToolTip(M.CLASSES[part.surface].note if part.surface in M.CLASSES else
                               'From the part\'s role and name (and by position on a one-part body)')
                box.currentIndexChanged.connect(lambda _=0, i=i, b=box: self._part_class(i, b.currentData()))
                t.setCellWidget(i, 3, box)
                t.setItem(i, 4, self._item(f'{part.surface_mm:g}' if part.surface_mm > 0 else '', True,
                                           editable=True, tip='mm; empty: the material\'s typical thickness'))
                t.setItem(i, 5, self._item(M.WHY_TEXT.get(info.get('why', ''), '')))
                t.setItem(i, 6, self._item(_f(info.get('area_m2'), '{:.2f} m²'), True))
            reg = res.regions if res is not None else {}
            bits = []
            if reg.get('fascia'):
                bits.append(f'bumper fascia: {reg["fascia"]} faces (ends below {reg["fascia_top_m"]:.2f} m)')
            if reg.get('windscreen'):
                bits.append(f'windscreen: {reg["windscreen"]} faces')
            if reg.get('rear_window'):
                bits.append(f'rear window: {reg["rear_window"]} faces')
            self.regions_text.setText(('Automatic regions on the painted body: ' + '; '.join(bits) + '.')
                                      if bits else '')
            st = res.settings if res is not None else self.settings
            self.lib.setRowCount(len(M.CLASSES))
            for i, mc in enumerate(M.CLASSES.values()):
                lay = ' + '.join(f'{M.DIELECTRICS[d][0]} {mm:g} mm' for d, mm in mc.layers) or '—'
                fig = M.class_figures(mc, st.paint, st.radar_ghz)
                self.lib.setItem(i, 0, self._item(mc.name, swatch=QColor(mc.color)))
                self.lib.setItem(i, 1, self._item(mc.kind))
                self.lib.setItem(i, 2, self._item(lay + (f' + {M.FINISH_LABEL[st.paint].lower()} paint'
                                                         if mc.painted and mc.layers and st.paint != 'none' else '')))
                self.lib.setItem(i, 3, self._item(_f(fig['two_way_loss_db'], '{:.2f} dB', 'blocks'), True))
                self.lib.setItem(i, 4, self._item(_f(fig['reflection_db'], '{:.1f} dB'), True))
                self.lib.setItem(i, 5, self._item(mc.note, tip=mc.note))
        finally:
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
        bad = sum(f['severity'] != 'good' for f in (res.findings if res is not None else []))
        self.tabs.setTabText(3, f'Findings · {bad}' if bad else 'Findings')

    # ------------------------------------------------------------------ export
    def export(self, folder=None):
        res = self.ctl.result
        if res is None:
            return []
        if folder is None:
            folder = QFileDialog.getExistingDirectory(self, 'Export the materials study into')
            if not folder:
                return []
        paths = M.export(res, self.ctx.store.project, folder)
        self.ctx.msg(f'Exported the materials study to {folder}', 5000)
        return paths
