"""Analysis ▸ Road and body motion: the road under the car (ramps, crests, garage ramps, speed bumps, kerbs, a crowned
road) and what it changes against flat ground; how braking, accelerating and cornering move the cameras.

The coverage analysis (heat maps, metrics, Issues) uses the road as soon as it has a primitive and is switched on; the
plan and 3D views draw it. This page runs the study (core.terrain.run: the road ahead and behind, the bumps' and
kerbs' first-seen distances, clearance, the body motion cases with their ground figures) as a main-window job with
Cancel, and again by itself after edits while it shows and a run is quick. The motion sliders update their read-out
live (statics only) and are saved when let go."""
from __future__ import annotations

import json
import math
import threading
import traceback

import numpy as np
from PyQt6.QtCore import QObject, QPointF, QRectF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import (QAbstractItemView, QComboBox, QFileDialog, QGridLayout, QHBoxLayout, QHeaderView,
                             QProgressBar, QScrollArea, QSlider, QSplitter, QTableWidget, QTableWidgetItem, QTabWidget,
                             QVBoxLayout, QWidget)

from ...core import suspension as SU
from ...core import terrain as TR
from ..charts import ChartBase
from ..planview import PlanView
from ..theme import THEME, STATUS, mono, sans
from ..widgets import (ROW_H, ScrubSpinBox, SegmentedControl, StatusLine, ToggleSwitch, button, hline, label,
                       tool_button)
from .recording_ws import Tile

JOB = 'terrain'
AUTO_MAX_S = 3.0
EMPTY = 'Load a vehicle model to study the road and the body motion.'
CHOICE_LABEL = {'ahead': 'Ahead', 'behind': 'Behind', 'left': 'Left', 'right': 'Right'}
REACH = (('15', '15 m'), ('30', '30 m'), ('60', '60 m'))
# (key, label, unit, decimals, step, scale, tip): the body-motion inputs (core.suspension.Dynamics)
DYN_FIELDS = (
    ('cg_height', 'CG height', 'm', 3, 0.01, 1.0,
     'The empty car\'s centre of gravity above the road. 0: estimated as 0.37 of the model\'s height (typical)'),
    ('anti_dive', 'Anti-dive', '%', 0, 1.0, 1.0,
     'Share of the front axle\'s braking load transfer its linkage takes (typical 20 %)'),
    ('anti_lift', 'Anti-lift', '%', 0, 1.0, 1.0,
     'Share of the rear axle\'s braking load transfer its linkage takes (typical 10 %)'),
    ('anti_squat', 'Anti-squat', '%', 0, 1.0, 1.0,
     'Share of the driven axle\'s load transfer under acceleration its linkage takes (typical 15 %)'),
    ('rc_front', 'Roll centre F', 'm', 3, 0.01, 1.0, 'Front roll-centre height (typical MacPherson strut 0.08 m)'),
    ('rc_rear', 'Roll centre R', 'm', 3, 0.01, 1.0, 'Rear roll-centre height (typical twist beam 0.20 m)'),
    ('unsprung_kg', 'Unsprung', 'kg', 0, 1.0, 1.0, 'Unsprung mass per corner: wheel, tyre, brake, hub (typical 35 kg)'),
)
PRESETS = (('Hard braking', -0.8, 0.0), ('Accelerate', 0.3, 0.0), ('Corner left', 0.0, 0.5),
           ('Corner right', 0.0, -0.5))


class _Sig(QObject):
    done = pyqtSignal(int, object)
    failed = pyqtSignal(int, str)
    progress = pyqtSignal(int, float)


def controller(ctx) -> 'TerrainController':
    c = getattr(ctx, 'terrain_ctl', None)
    if c is None:
        c = ctx.terrain_ctl = TerrainController(ctx)
    return c


def inputs_key(project) -> str:
    """Everything the study reads, as text: a result made from other inputs is stale."""
    m = project.vehicle.mesh
    a = project.analysis
    return json.dumps({'mesh': [id(m), getattr(m, 'n_tris', 0), project.vehicle.occluder_target],
                       'wheels': project.vehicle.wheel_centres, 'variant': project.variant.to_dict(),
                       'rig': [r.to_dict() for r in project.rig], 'roi': project.roi.to_dict(), 'cell': a.cell_m,
                       'obstacle_mode': a.obstacle_mode, 'terrain': a.terrain, 'suspension': a.suspension,
                       'housings': a.housings, 'library': project.section('library')}, sort_keys=True, default=str)


class TerrainController(QObject):
    """The project's road and body motion study: settings edits (undoable, saved with the project), runs on demand and
    again by itself after edits while the page shows it and a run is quick."""
    changed = pyqtSignal()
    progress = pyqtSignal(float)

    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        self.result: TR.TerrainResult | None = None
        self.stale = False
        self.busy = False
        self.status = ''
        self.active = False
        self._gen = 0
        self._for = None
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

    # ---- settings
    @property
    def settings(self) -> TR.TerrainSettings:
        return TR.TerrainSettings.from_dict(self.ctx.store.project.analysis.terrain)

    def _store(self, st: TR.TerrainSettings, text) -> bool:
        new = TR.TerrainSettings.from_dict(st.to_dict()).to_dict()
        if new == self.settings.to_dict():
            return False                   # nothing changes (a clamped value: the fields show what is stored)

        def fn(p):
            p.analysis.terrain = new
        self.ctx.store.edit('analysis', fn, f'Road and body motion · {text}')
        return True

    def set_enabled(self, on: bool) -> bool:
        st = self.settings
        st.enabled = bool(on)
        return self._store(st, 'road used in the coverage analysis' if on else 'flat ground in the coverage analysis')

    def add_feature(self, kind: str) -> bool:
        st = self.settings
        if kind not in TR.PARAMS or len(st.features) >= TR.MAX_FEATURES:
            return False
        st.features.append(TR.default_feature(kind))
        return self._store(st, f'add {TR.KIND_LABEL[kind].lower()}')

    def remove_feature(self, i: int) -> bool:
        st = self.settings
        if not 0 <= i < len(st.features):
            return False
        f = st.features.pop(i)
        return self._store(st, f'remove {TR.KIND_LABEL[f["kind"]].lower()}')

    def set_feature(self, i: int, text: str, **kw) -> bool:
        st = self.settings
        if not 0 <= i < len(st.features):
            return False
        st.features[i] = TR.clean_feature(dict(st.features[i], **kw)) or st.features[i]
        return self._store(st, f'{TR.KIND_LABEL[st.features[i]["kind"]].lower()} {text}')

    def set_motion(self, text: str, **kw) -> bool:
        st = self.settings
        for k, v in kw.items():
            setattr(st, k, v)
        return self._store(st, text)

    def set_dynamics(self, text: str, **kw) -> bool:
        st = self.settings
        st.dynamics = SU.Dynamics.from_dict(dict(st.dynamics.to_dict(), **kw))
        return self._store(st, text)

    def motion_defaults(self) -> bool:
        st = self.settings
        st.dynamics = SU.Dynamics()
        return self._store(st, 'typical body-motion inputs')

    # ---- state
    def _quick(self):
        return self.result is None or self.result.seconds < AUTO_MAX_S

    def _mark_stale(self):
        if self.result is not None and not self.stale:
            self.stale = True
            self.changed.emit()
        if self.active and self._quick() and not self.busy:
            self.timer.start()

    def _on_edit(self, *_):
        if not (self.active or self.result is not None):
            return
        if self._for is None or inputs_key(self.ctx.store.project) != self._for[1]:
            self._mark_stale()

    def set_active(self, on: bool):
        self.active = on
        if on and (self.result is None or self.stale) and self._quick():
            self.timer.start()

    def _reset(self, *_):
        self._gen += 1
        self.result, self._for, self.stale, self.busy, self.status = None, None, False, False, ''
        self.changed.emit()
        if self.active:
            self.timer.start()

    # ---- run
    def run(self, quiet=False) -> bool:
        ctx = self.ctx
        p = ctx.store.project
        win = ctx.window
        if p.vehicle.mesh is None or getattr(p.vehicle.mesh, 'n_tris', 0) == 0:
            if not (win is not None and JOB in win._jobs):
                self._gen += 1
                self.busy = False
            self.result, self._for, self.stale, self.status = None, None, False, EMPTY
            self.changed.emit()
            return False
        if self.busy:
            if quiet or (win is not None and JOB in win._jobs):
                return False
            self._gen += 1
        project = p.snapshot()
        eng = win._shared_engine() if win is not None else ctx.analysis.engine.share()
        self._gen += 1
        gen = self._gen
        self.busy = True
        self.status = 'Studying the road and the body motion…'
        key = (p, inputs_key(p))

        def work(progress, cancel):
            return key, TR.run(project, eng, progress=progress, cancel=cancel)
        if not quiet and win is not None and hasattr(win, 'run_job'):
            ok = win.run_job(JOB, 'Road and body motion', work, lambda out: self._done(gen, out),
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
        bad = sum(f['severity'] in TR.ISSUE_SEVERITIES for f in res.findings)
        road = f'{len(res.features)} road primitive{"s" * (len(res.features) != 1)}' if res.road else 'flat ground'
        self.status = (f'{res.variant}: {road} · {len(res.motion["cases"])} motion cases · '
                       + (f'{bad} finding{"s" * (bad != 1)} to look at' if bad else 'nothing to flag')
                       + f' · {res.seconds:.1f} s')
        self.changed.emit()
        if self.stale and self.active and self._quick():
            self.timer.start()

    def _failed(self, gen, msg):
        if gen != self._gen:
            return
        self.busy = False
        self.status = 'Failed: ' + msg.splitlines()[0]
        self.changed.emit()


# =========================================================================== the road's profile, side on
class ProfileChart(ChartBase):
    """The road along the centre line behind and ahead of the car, side on (heights exaggerated, the car true to
    scale): green where some camera sees the road, red where only flat ground would be seen (the road hides it), grey
    where no camera looks; the primitives marked under the axis. Hover for the figures at a distance."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.title = 'Road along the centre line'
        self.res = None
        self.project = None
        self.reach = 30.0
        self.setMinimumHeight(170)
        self._map = None

    def set_result(self, res, project):
        self.res, self.project = res, project
        self.tip = None
        self.update()

    def set_reach(self, r):
        self.reach = float(r)
        self.update()

    def _geometry(self):
        res, project = self.res, self.project
        ref = TR.references(project)
        x_lo, x_hi = ref['rear_x'] - self.reach, ref['front_x'] + self.reach
        hs = [0.0]
        for pr in res.profiles.values():
            sel = pr['d'] <= self.reach
            hs += [float(pr['h'][sel].min()), float(pr['h'][sel].max())]
        W, H = self.width(), self.height()
        left, right, top, bot = 10.0, W - 10.0, 26.0, H - 40.0
        sx = (right - left) / (x_hi - x_lo)
        zl, zh = min(hs) - 0.1, max(hs) + 0.1
        car_h = 1.6 * sx                                    # the car stands true to scale
        sz = min((bot - top - car_h - 8) / max(zh - zl, 1e-3), sx * 20.0)
        sz = max(sz, sx)
        z0 = bot - (0.0 - zl) * sz if zh - zl > 1e-3 else bot - 6

        def P(x, z):
            return QPointF(left + (x - x_lo) * sx, z0 - z * sz)
        return ref, x_lo, x_hi, sx, sz, P, (left, right, top, bot)

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        res = self.res
        if res is None or not res.profiles or self.project is None:
            self.paint_title(p)
            p.setFont(sans(11))
            p.setPen(THEME.color('muted'))
            p.drawText(QRectF(0, 24, self.width(), self.height() - 24), Qt.AlignmentFlag.AlignCenter,
                       'Flat ground: add a ramp, crest, garage ramp, speed bump or kerb to see the road side on.'
                       if res is not None else 'Run the study to see the road side on.')
            p.end()
            return
        ref, x_lo, x_hi, sx, sz, P, (left, right, top, bot) = self._geometry()
        self.paint_title(p, f'm behind and ahead of the bumpers · heights × {sz / sx:.1f} · the car true to scale')
        p.setPen(QPen(THEME.color('hair'), 1))
        p.drawLine(P(x_lo, 0), P(x_hi, 0))
        seen_c, hid_c, out_c = QColor(STATUS['good']), QColor(STATUS['critical']), THEME.color('muted')
        self._map = []
        for name, pr in res.profiles.items():
            sgn, x0 = (1.0, ref['front_x']) if name == 'ahead' else (-1.0, ref['rear_x'])
            sel = pr['d'] <= self.reach
            xs, zs = x0 + sgn * pr['d'][sel], pr['h'][sel]
            seen, flat = pr['seen'][sel], pr['seen_flat'][sel]
            for k in range(len(xs) - 1):
                col = seen_c if seen[k] else hid_c if flat[k] else out_c
                p.setPen(QPen(col, 3.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.FlatCap))
                p.drawLine(P(xs[k], zs[k]), P(xs[k + 1], zs[k + 1]))
            self._map.append((name, xs, zs, pr, sel))
        # the car (true to scale, on its stance) and its cameras
        st = res.stance or {}
        pitch = math.radians(st.get('pitch_deg', 0.0) or 0.0)
        heave = (st.get('heave_mm', 0.0) or 0.0) / 1000.0
        hgt = SU.vehicle_geometry(self.project)['dims'].get('Height', 1.5)
        base = P(ref['rear_x'], 0.0)

        def body(x, z):
            xw = x * math.cos(pitch) + z * math.sin(pitch)
            zw = -x * math.sin(pitch) + z * math.cos(pitch) - heave
            return QPointF(base.x() + (xw - ref['rear_x']) * sx, base.y() - zw * sx)
        xr, xf = ref['rear_x'], ref['front_x']
        path = QPainterPath(body(xr, 0.2))
        for x, z in ((xf, 0.2), (xf, hgt * 0.55), (xr + 0.62 * (xf - xr), hgt), (xr + 0.25 * (xf - xr), hgt),
                     (xr, hgt * 0.62)):
            path.lineTo(body(x, z))
        path.closeSubpath()
        p.setPen(QPen(THEME.color('ink2'), 1.2))
        p.setBrush(THEME.color('raised'))
        p.drawPath(path)
        for c in self.project.variant.cameras:
            if c.visible:
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(THEME.cam(c.slot))
                p.drawEllipse(body(c.x, c.z), 3.0, 3.0)
        # primitives under the axis, distance ticks
        p.setFont(sans(10))
        for f in TR.Terrain(res.features, ref).features:
            if not f.along:
                continue
            x = f.origin
            if x_lo <= x <= x_hi:
                q = QPointF(left + (x - x_lo) * sx, bot + 2)
                p.setPen(QPen(THEME.color('ink2'), 1))
                p.drawLine(q, QPointF(q.x(), q.y() + 4))
                name = {'bump': 'bump', 'kerb': 'kerb', 'ramp': 'ramp', 'crest': 'crest', 'garage': 'garage'}[f.kind]
                p.drawText(QRectF(q.x() - 30, q.y() + 3, 60, 13), Qt.AlignmentFlag.AlignCenter, name)
        p.setFont(mono(10))
        p.setPen(THEME.color('muted'))
        step = 5.0 if self.reach <= 30 else 10.0
        for sgn, x0 in ((1.0, ref['front_x']), (-1.0, ref['rear_x'])):
            for d in np.arange(step, self.reach + 1e-6, step):
                q = P(x0 + sgn * d, 0.0)
                p.drawText(QRectF(q.x() - 20, bot + 22, 40, 12), Qt.AlignmentFlag.AlignCenter, f'{d:g}')

        self.paint_tip(p)
        p.end()

    def mouseMoveEvent(self, e):
        if not self._map or self.res is None:
            return
        x = e.position().x()
        best = None
        ref, x_lo, x_hi, sx, sz, P, _ = self._geometry()
        for name, xs, zs, pr, sel in self._map:
            sx_ = np.array([P(a, b).x() for a, b in zip(xs, zs)]) if len(xs) else np.zeros(0)
            if not len(sx_):
                continue
            k = int(np.argmin(np.abs(sx_ - x)))
            if best is None or abs(sx_[k] - x) < best[0]:
                best = (abs(sx_[k] - x), name, k, xs, zs, pr, sel)
        if best is None or best[0] > 12:
            if self.tip is not None:
                self.tip = None
                self.update()
            return
        _, name, k, xs, zs, pr, sel = best
        d = float(pr['d'][sel][k])
        seen, flat = bool(pr['seen'][sel][k]), bool(pr['seen_flat'][sel][k])
        dens = float(pr['density'][sel][k])
        what = (f'seen, best {dens:.2f} px/cm' if seen else 'hidden by the road (seen on flat ground)' if flat else
                'no camera looks here')
        self.set_tip(P(xs[k], zs[k]), f'{d:.1f} m {name}', f'road {zs[k]:+.2f} m', what)
        self.update()

    def leaveEvent(self, e):
        self.tip = None
        self.update()


# =========================================================================== one road primitive
class FeatureRow(QWidget):
    """A primitive's fields: its parameters (ScrubSpinBoxes), its side and a remove button; every edit is one undo
    step."""

    def __init__(self, page, i, f):
        super().__init__()
        self.page, self.i, self.kind = page, i, f['kind']
        self.setToolTip(TR.TYPICAL[self.kind])
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 4, 0, 4)
        lay.setSpacing(2)
        head = QHBoxLayout()
        head.setSpacing(6)
        head.addWidget(label(TR.KIND_LABEL[self.kind], bold=True))
        self.desc = label('', muted=True, small=True)
        head.addWidget(self.desc, 1)
        self.b_remove = tool_button('trash', f'Remove this {TR.KIND_LABEL[self.kind].lower()}')
        self.b_remove.clicked.connect(lambda: page.ctl.remove_feature(self.i))
        head.addWidget(self.b_remove)
        lay.addLayout(head)
        grid = QGridLayout()
        grid.setHorizontalSpacing(6)
        grid.setVerticalSpacing(2)
        self.fields = {}
        for n, (key, text, unit, _, lo, hi, dec, step, scale, tip) in enumerate(TR.PARAMS[self.kind]):
            fl = ScrubSpinBox('', 'plain', dec, unit, step=step, width=96)
            fl.setToolTip(tip)
            fl.committed.connect(lambda v, k=key, s=scale, t=text, u=unit:
                                 page.ctl.set_feature(self.i, f'{t.lower()} {v:g} {u}', **{k: v / s}))
            a = label(text, ink2=True)
            a.setToolTip(tip)
            grid.addWidget(a, n // 2, (n % 2) * 2)
            grid.addWidget(fl, n // 2, (n % 2) * 2 + 1)
            self.fields[key] = (fl, scale)
        grid.setColumnStretch(4, 1)
        lay.addLayout(grid)
        self.choice = None
        if self.kind in TR.CHOICE:
            k, opts, _ = TR.CHOICE[self.kind]
            self.choice = SegmentedControl([(o, CHOICE_LABEL[o]) for o in opts], f[k])
            self.choice.setToolTip('Where it is: ahead of the front bumper or behind the rear one' if k == 'where' else
                                   'Beside the car (left, right) or across the road (ahead, behind)')
            self.choice.changed.connect(lambda v, k=k: page.ctl.set_feature(self.i, CHOICE_LABEL[v].lower(), **{k: v}))
            lay.addWidget(self.choice)
        self.sync(f)

    def sync(self, f):
        self.desc.setText(TR.describe(f))
        for key, (fl, scale) in self.fields.items():
            fl.set_value(f[key] * scale)
        if self.choice is not None:
            self.choice.set_value(f[TR.CHOICE[self.kind][0]])


def _table(cols, tips=None):
    t = QTableWidget(0, len(cols))
    for c, h in enumerate(cols):
        it = QTableWidgetItem(h)
        if tips and tips.get(h):
            it.setToolTip(tips[h])
        t.setHorizontalHeaderItem(c, it)
    t.verticalHeader().setVisible(False)
    t.verticalHeader().setDefaultSectionSize(ROW_H)
    t.horizontalHeader().setFixedHeight(ROW_H)
    t.setShowGrid(False)
    t.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    t.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
    hh = t.horizontalHeader()
    for c in range(len(cols)):
        hh.setSectionResizeMode(c, QHeaderView.ResizeMode.ResizeToContents)
    hh.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
    return t


def _cell(t, r, c, text, num=False, color=None, tip=None, swatch=None):
    it = QTableWidgetItem(text)
    if num:
        it.setFont(mono(12))
        it.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    if color is not None:
        it.setForeground(color)
    if tip:
        it.setToolTip(tip)
    if swatch is not None:
        it.setData(Qt.ItemDataRole.DecorationRole, swatch)
    t.setItem(r, c, it)


def _m(v, nd=1):
    return '—' if v is None else f'{v:.{nd}f}'


# =========================================================================== the page
class TerrainPage(QWidget):
    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        self.ctl = controller(ctx)
        self.setObjectName('base')
        self.rows: list[FeatureRow] = []
        self.findings: list[StatusLine] = []
        self._shown = None
        self._syncing = False
        self._dirty = True
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        split = QSplitter(Qt.Orientation.Horizontal)
        split.setHandleWidth(2)
        split.addWidget(self._left())
        # ---------------- centre: the plan on the road, the profile
        centre = QSplitter(Qt.Orientation.Vertical)
        centre.setHandleWidth(2)
        pw = QWidget()
        pw.setObjectName('panel')
        pl = QVBoxLayout(pw)
        pl.setContentsMargins(8, 4, 8, 6)
        pl.addWidget(label('Coverage on this road', bold=True))
        self.plan = PlanView(ctx, 'coverage', interactive=False)
        self.plan.label = 'Coverage analysis · the road\'s edges dashed, a kerb\'s face solid, height contours thin'
        pl.addWidget(self.plan, 1)
        centre.addWidget(pw)
        cw = QWidget()
        cw.setObjectName('panel')
        cl = QVBoxLayout(cw)
        cl.setContentsMargins(8, 4, 8, 6)
        head = QHBoxLayout()
        head.addStretch(1)
        self.reach = SegmentedControl(list(REACH), '30')
        self.reach.setToolTip('How far behind and ahead of the car the profile reaches')
        self.reach.changed.connect(lambda v: self.chart.set_reach(float(v)))
        head.addWidget(self.reach)
        cl.addLayout(head)
        self.chart = ProfileChart()
        cl.addWidget(self.chart, 1)
        centre.addWidget(cw)
        centre.setSizes([560, 300])
        split.addWidget(centre)
        # ---------------- right: tiles, tables, findings
        right = QSplitter(Qt.Orientation.Vertical)
        right.setHandleWidth(2)
        tw = QWidget()
        tw.setObjectName('panel')
        tl = QVBoxLayout(tw)
        tl.setContentsMargins(10, 8, 10, 8)
        tl.setSpacing(8)
        tiles = QGridLayout()
        tiles.setSpacing(8)
        self.tiles = {}
        for n, (key, caption, tip) in enumerate((
                ('blind', 'Blind ground', 'Blind ground in the region of interest on this road, against flat ground'),
                ('ahead', 'Road ahead in view', 'How far ahead of the front bumper the road\'s centre line stays in '
                                                'view without a break, against flat ground'),
                ('seen', 'First seen', 'The least visible bump or kerb: the distance at which a camera first '
                                       'resolves it as the car approaches'),
                ('clear', 'Clearance', 'The smallest gap between the body\'s underside and the road: where it stands '
                                       'and driving over each primitive across the road'),
                ('brake', 'Hard braking', 'The body\'s pitch against the calibration load at 0.8 g, and how far the '
                                          'horizon of the forward camera moves'),
                ('corner', 'Cornering 0.5 g', 'The body\'s roll in a 0.5 g left turn against the calibration load'))):
            t = self.tiles[key] = Tile(caption, tip)
            tiles.addWidget(t, n // 3, n % 3)
        tl.addLayout(tiles)
        self.tabs = QTabWidget()
        self.t_road = _table(('Primitive', 'First seen', 'By', 'Lost', 'Gaps', 'Clearance driving over'),
                             {'First seen': 'The farthest distance (m) at which some camera resolves it as the car '
                                            'approaches: its top in view and its height spanning the minimum pixels',
                              'Lost': 'The nearest distance it is still resolved at; closer it is out of view',
                              'Gaps': 'Distances in between at which no camera resolves it (hidden by the road)',
                              'Clearance driving over': 'The smallest body clearance (cm) met driving over it'})
        self.t_cams = _table(('Camera', 'Seen flat m²', 'Seen road m²', 'Hidden m²', 'Density flat', 'Density road'),
                             {'Hidden m²': 'Ground in the region of interest the camera sees on flat ground but not '
                                           'on this road',
                              'Density flat': 'Mean pixel density on the ground it sees, px/cm'})
        self.t_motion = _table(('Case', 'Pitch °', 'Roll °', 'Heave mm', 'Horizon', 'Blind Δ m²', 'Shift cm',
                                'Ghost cm'),
                               {'Pitch °': 'Against the calibration load: positive nose down',
                                'Roll °': 'Against the calibration load: positive right side down',
                                'Heave mm': 'Against the calibration load: positive lower (at the rear axle)',
                                'Horizon': 'How far the horizon moves in each forward camera\'s image (px, up or '
                                           'down)',
                                'Blind Δ m²': 'Change of the blind ground with the body moved (flat ground)',
                                'Shift cm': 'Worst shift of the stitched ground calibrated at the calibration load',
                                'Ghost cm': 'Worst disagreement of two cameras at a seam'})
        self.tabs.addTab(self.t_motion, 'Body motion')
        self.tabs.addTab(self.t_road, 'Road primitives')
        self.tabs.addTab(self.t_cams, 'Cameras on the road')
        tl.addWidget(self.tabs, 1)
        right.addWidget(tw)
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
        self.assume = label(TR.ASSUMPTIONS, muted=True, small=True, wrap=True)
        fl.addWidget(self.assume)
        right.addWidget(fw)
        right.setSizes([560, 300])
        split.addWidget(right)
        split.setSizes([360, 640, 640])
        lay.addWidget(split)
        self.ctl.changed.connect(self.refresh)
        ctx.store.changed.connect(self._on_store)
        ctx.store.project_replaced.connect(self._on_store)
        THEME.changed.connect(self._retheme)          # a bound slot: disconnected when the page goes
        self._sync_fields()
        self.refresh()

    def _retheme(self, *_):
        self._shown = None
        self.refresh()

    # ------------------------------------------------------------------ settings panel
    def _left(self):
        w = QWidget()
        w.setObjectName('panel')
        ll = QVBoxLayout(w)
        ll.setContentsMargins(8, 6, 8, 8)
        ll.setSpacing(4)
        top = QHBoxLayout()
        top.addWidget(label('Road and body motion'))
        top.addStretch(1)
        top.addWidget(label('terrain, dive, squat, roll', muted=True, small=True))
        ll.addLayout(top)
        ll.addWidget(hline())
        ll.addWidget(label('Put the car on a real road: ground points sit on it, the car stands on it, and the road '
                           'itself can hide ground from a camera. Distances are from the bumper (or the body\'s side '
                           'for a kerb beside the car). With no primitive the ground is flat and every result is as '
                           'before.', muted=True, small=True, wrap=True))
        self.enabled = ToggleSwitch('Use the road in the coverage analysis')
        self.enabled.setToolTip('The heat maps, metrics and Issues use the road; off: flat ground (the road is kept)')
        self.enabled.clicked.connect(lambda on: self.ctl.set_enabled(on))
        ll.addWidget(self.enabled)
        ll.addSpacing(4)
        ll.addWidget(label('Road', bold=True))
        self.rows_host = QWidget()
        self.rows_lay = QVBoxLayout(self.rows_host)
        self.rows_lay.setContentsMargins(0, 0, 0, 0)
        self.rows_lay.setSpacing(2)
        ll.addWidget(self.rows_host)
        self.no_rows = label('Flat ground.', muted=True, small=True)
        self.rows_lay.addWidget(self.no_rows)
        add = QGridLayout()
        add.setSpacing(4)
        self.add_buttons = {}
        for n, kind in enumerate(TR.KINDS):
            b = button(TR.KIND_LABEL[kind], 'ghost', 'plus', f'Add a {TR.KIND_LABEL[kind].lower()}. '
                                                           f'{TR.TYPICAL[kind]}')
            b.clicked.connect(lambda _=False, k=kind: self.ctl.add_feature(k))
            add.addWidget(b, n // 3, n % 3)
            self.add_buttons[kind] = b
        ll.addLayout(add)
        g = QGridLayout()
        g.setHorizontalSpacing(6)
        g.setVerticalSpacing(2)
        self.f_range = ScrubSpinBox('', 'plain', 0, 'm', step=5.0, width=96)
        self.f_range.setToolTip('How far ahead and behind the road is checked (the profile and the bumps\' and '
                                'kerbs\' first-seen distances)')
        self.f_range.committed.connect(lambda v: self.ctl.set_motion(f'checked to {v:g} m', range_m=v))
        self.f_minpx = ScrubSpinBox('', 'plain', 1, 'px', step=0.5, width=96)
        self.f_minpx.setToolTip('A bump or kerb counts as seen when its height spans this many pixels in one image '
                                'with its top in view')
        self.f_minpx.committed.connect(lambda v: self.ctl.set_motion(f'seen at {v:g} px', min_px=v))
        g.addWidget(label('Check to', ink2=True), 0, 0)
        g.addWidget(self.f_range, 0, 1)
        g.addWidget(label('Seen at', ink2=True), 0, 2)
        g.addWidget(self.f_minpx, 0, 3)
        g.setColumnStretch(4, 1)
        ll.addLayout(g)
        ll.addSpacing(8)
        ll.addWidget(hline())
        ll.addWidget(label('Body motion', bold=True))
        ll.addWidget(label(TR.MOTION_NOTE, muted=True, small=True, wrap=True))
        row = QHBoxLayout()
        row.addWidget(label('Driving load', ink2=True))
        self.load = QComboBox()
        self.load.setToolTip('What the car carries while driving (a suspension load case); the results are against '
                             'the load the cameras are calibrated at')
        self.load.currentIndexChanged.connect(self._load_picked)
        row.addWidget(self.load, 1)
        ll.addLayout(row)
        self.sl_long = QSlider(Qt.Orientation.Horizontal)
        self.sl_long.setRange(-120, 80)
        self.sl_long.setToolTip('Longitudinal acceleration: left braking, right accelerating')
        self.sl_lat = QSlider(Qt.Orientation.Horizontal)
        self.sl_lat.setRange(-100, 100)
        self.sl_lat.setToolTip('Lateral acceleration: left a right turn, right a left turn')
        self.lab_long = label('', mono_=True)
        self.lab_lat = label('', mono_=True)
        sg = QGridLayout()
        sg.setHorizontalSpacing(6)
        sg.addWidget(label('Longitudinal', ink2=True), 0, 0)
        sg.addWidget(self.sl_long, 0, 1)
        sg.addWidget(self.lab_long, 0, 2)
        sg.addWidget(label('Lateral', ink2=True), 1, 0)
        sg.addWidget(self.sl_lat, 1, 1)
        sg.addWidget(self.lab_lat, 1, 2)
        sg.setColumnStretch(1, 1)
        ll.addLayout(sg)
        self._slide = QTimer(self)
        self._slide.setSingleShot(True)
        self._slide.setInterval(400)
        self._slide.timeout.connect(self._commit_sliders)
        for s in (self.sl_long, self.sl_lat):
            s.valueChanged.connect(self._slid)
            s.sliderReleased.connect(self._commit_sliders)
        pr = QHBoxLayout()
        pr.setSpacing(4)
        for name, ax, ay in PRESETS:
            b = button(name, 'ghost', None, f'{ax:+g} g longitudinal, {ay:+g} g lateral')
            b.clicked.connect(lambda _=False, n=name, a=ax, l_=ay: self.ctl.set_motion(f'sliders at {n.lower()}',
                                                                                      long_g=a, lat_g=l_))
            pr.addWidget(b)
        pr.addStretch(1)
        ll.addLayout(pr)
        self.readout = label('', small=True, wrap=True)
        ll.addWidget(self.readout)
        dg = QGridLayout()
        dg.setHorizontalSpacing(6)
        dg.setVerticalSpacing(2)
        self.dyn = {}
        for n, (key, text, unit, dec, step, scale, tip) in enumerate(DYN_FIELDS):
            fl = ScrubSpinBox('', 'plain', dec, unit, step=step, width=96)
            fl.setToolTip(tip)
            fl.committed.connect(lambda v, k=key, t=text, u=unit: self.ctl.set_dynamics(f'{t.lower()} {v:g} {u}',
                                                                                        **{k: v}))
            a = label(text, ink2=True)
            a.setToolTip(tip)
            dg.addWidget(a, n // 2, (n % 2) * 2)
            dg.addWidget(fl, n // 2, (n % 2) * 2 + 1)
            self.dyn[key] = fl
        dg.setColumnStretch(4, 1)
        ll.addLayout(dg)
        self.cg_note = label('', muted=True, small=True, wrap=True)
        ll.addWidget(self.cg_note)
        self.drive = SegmentedControl([(k, {'front': 'Front drive', 'rear': 'Rear drive', 'awd': 'AWD'}[k])
                                       for k in SU.DRIVES], 'front')
        self.drive.setToolTip('Which axle drives: the anti-squat percentage applies to it under acceleration '
                              '(all-wheel drive: half to each)')
        self.drive.changed.connect(lambda k: self.ctl.set_dynamics(SU.DRIVE_LABEL[k].lower(), drive=k))
        ll.addWidget(self.drive)
        b_typ = button('Typical inputs', 'ghost', None, 'Back to the typical body-motion inputs')
        b_typ.clicked.connect(lambda: self.ctl.motion_defaults())
        rr = QHBoxLayout()
        rr.addWidget(b_typ)
        rr.addStretch(1)
        ll.addLayout(rr)
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
        self.b_export = button('Export…', None, 'export', 'The road, the coverage against flat ground, the profile, '
                                                           'the first-seen distances, clearance and the motion cases '
                                                           'as JSON and CSV, and the profile as PNG')
        self.b_export.clicked.connect(lambda: self.export())
        self.b_run = button('Run', 'primary')
        self.b_run.clicked.connect(lambda: self.ctl.run())
        bl.addWidget(self.b_export)
        bl.addWidget(self.b_run)
        ll.addLayout(bl)
        sc = QScrollArea()
        sc.setWidgetResizable(True)
        sc.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        sc.setWidget(w)
        sc.setMinimumWidth(340)
        return sc

    # ------------------------------------------------------------------ edits
    def _slid(self, *_):
        if self._syncing:
            return
        self._preview()
        self._slide.start()

    def _commit_sliders(self):
        self._slide.stop()
        ax, ay = self.sl_long.value() / 100.0, self.sl_lat.value() / 100.0
        st = self.ctl.settings
        if abs(st.long_g - ax) > 1e-9 or abs(st.lat_g - ay) > 1e-9:
            self.ctl.set_motion(f'sliders at {ax:+.2f} g, {ay:+.2f} g', long_g=ax, lat_g=ay)

    def _load_picked(self, i):
        if self._syncing or i < 0:
            return
        cid = self.load.itemData(i) or ''
        if cid != self.ctl.settings.load_case:
            self.ctl.set_motion(f'driving with {self.load.itemText(i)}', load_case=cid)

    def _preview(self):
        """The slider case's attitude and forward cameras' horizon, live (the statics only)."""
        p = self.ctx.store.project
        st = self.ctl.settings
        st.long_g, st.lat_g = self.sl_long.value() / 100.0, self.sl_lat.value() / 100.0
        self.lab_long.setText(f'{st.long_g:+.2f} g')
        self.lab_lat.setText(f'{st.lat_g:+.2f} g')
        try:
            c = TR.motion_cases(p, st)['cases'][-1]
        except Exception as ex:                     # a broken suspension file: say so, keep the page
            self.readout.setText(f'Cannot solve the body motion: {ex}')
            return
        hz = [h for h in c['horizon'] if h['facing'] == 'forward']
        parts = [f'Pitch {c["rel"]["pitch_deg"]:+.2f}° · roll {c["rel"]["roll_deg"]:+.2f}° · heave '
                 f'{c["rel"]["heave_mm"]:+.0f} mm against the calibration load']
        parts += [f'{h["tag"]} horizon {abs(h["dv_px"]):.0f} px {"up" if h["dv_px"] < 0 else "down"} '
                  f'({h["d_deg"]:+.2f}°), tilt {h["d_tilt_deg"]:+.2f}°' for h in hz]
        if c['lifted']:
            parts.append(f'the {", ".join(c["lifted"])} wheel would lift')
        self.readout.setText('\n'.join(parts))

    def _sync_fields(self):
        st = self.ctl.settings
        p = self.ctx.store.project
        self._syncing = True
        try:
            self.enabled.setChecked(st.enabled)
            kinds = [f['kind'] for f in st.features]
            if kinds != [r.kind for r in self.rows]:
                for r in self.rows:
                    self.rows_lay.removeWidget(r)
                    r.hide()                        # until the deferred delete runs it would still paint
                    r.deleteLater()
                self.rows = []
                for i, f in enumerate(st.features):
                    r = FeatureRow(self, i, f)
                    self.rows_lay.addWidget(r)
                    self.rows.append(r)
            else:
                for r, f in zip(self.rows, st.features):
                    r.sync(f)
            self.no_rows.setVisible(not st.features)
            for b in self.add_buttons.values():
                b.setEnabled(len(st.features) < TR.MAX_FEATURES)
            self.f_range.set_value(st.range_m)
            self.f_minpx.set_value(st.min_px)
            d = st.dynamics
            for key, fl in self.dyn.items():
                fl.set_value(getattr(d, key))
            self.drive.set_value(d.drive)
            est = SU.cg_height_estimate(p)
            self.cg_note.setText(f'CG height {"entered" if d.cg_height > 0 else "estimated"}: '
                                 f'{d.cg_height if d.cg_height > 0 else est:.2f} m for the empty car. The anti-dive, '
                                 'anti-lift, anti-squat, roll-centre and unsprung figures are typical unless entered; '
                                 'wheel rates and anti-roll bars come from Vehicle ▸ Load cases.')
            su = SU.settings_for(p)
            cal = su.case(su.calibrated_at)
            items = [('', f'Calibration load ({cal.name if cal else "kerb"})')] + \
                [(c.id, c.name) for c in su.cases if cal is None or c.id != cal.id]
            if [(self.load.itemData(i), self.load.itemText(i)) for i in range(self.load.count())] != items:
                self.load.clear()
                for cid, name in items:
                    self.load.addItem(name, cid)
            k = next((i for i, (cid, _) in enumerate(items) if cid == st.load_case), 0)
            self.load.setCurrentIndex(k)
            self.sl_long.setValue(int(round(st.long_g * 100)))
            self.sl_lat.setValue(int(round(st.lat_g * 100)))
        finally:
            self._syncing = False
        self._preview()

    # ------------------------------------------------------------------ state
    def _on_store(self, *_):
        if self.isVisible():
            self._sync_fields()
        else:
            self._dirty = True

    def showEvent(self, e):
        super().showEvent(e)
        if self._dirty:
            self._dirty = False
            self._sync_fields()
        self.ctl.set_active(True)

    def hideEvent(self, e):
        super().hideEvent(e)
        self.ctl.set_active(False)

    def activate(self):
        if self._dirty:
            self._dirty = False
            self._sync_fields()
        self.ctl.set_active(True)

    def refresh(self):
        ctl = self.ctl
        res = ctl.result
        win = self.ctx.window
        p = self.ctx.store.project
        empty = p.vehicle.mesh is None
        self.b_run.setEnabled(not empty and not (ctl.busy and win is not None and JOB in win._jobs))
        self.bar.setVisible(ctl.busy)
        self.b_export.setEnabled(res is not None)
        self.status.setText((EMPTY if empty else ctl.status) +
                            (' · stale: the design or the settings changed' if ctl.stale and res else ''))
        if res is self._shown:
            return
        self._shown = res
        self.chart.set_result(res, p)
        for t in (self.t_road, self.t_cams, self.t_motion):
            t.setRowCount(0)
        for w in self.findings:
            self.find_lay.removeWidget(w)
            w.hide()
            w.deleteLater()
        self.findings = []
        for t in self.tiles.values():
            t.set('—', '', 'run the study')
        if res is None:
            return
        self._fill_tiles(res)
        self._fill_road(res)
        self._fill_cams(res)
        self._fill_motion(res)
        for f in res.findings:
            w = StatusLine(f['severity'] if f['severity'] in STATUS else None, f['message'])
            self.find_lay.insertWidget(self.find_lay.count() - 1, w)
            self.findings.append(w)

    def _fill_tiles(self, res):
        cov = res.coverage
        if cov is not None:
            d = cov['road']['blind_m2'] - cov['flat']['blind_m2']
            self.tiles['blind'].set(f'{cov["road"]["blind_m2"]:.2f}', 'm²', f'flat ground {cov["flat"]["blind_m2"]:.2f} m²',
                                    'warning' if d > 0.05 else 'good')
        else:
            self.tiles['blind'].set('—', '', 'flat ground' if not res.road else 'no cameras')
        pr = res.profiles.get('ahead')
        if pr is not None:
            far = pr['seen_to_m']
            short = [h for h in pr['hidden'] if far is None or h[0] < far]
            self.tiles['ahead'].set(_m(far), 'm', f'flat ground {_m(pr["seen_to_flat_m"])} m'
                                    + (f' · {len(short)} short shadow{"s" * (len(short) != 1)}' if short else ''))
        else:
            self.tiles['ahead'].set('—', '', 'flat ground')
        # the least visible bump or kerb, across the road first (its distance is along the road; a kerb beside the
        # car is checked out to the side, which does not compare)
        pool = [s for s in res.sweeps if s['across']] or res.sweeps
        if pool:
            worst = min(pool, key=lambda s: -1 if s['first_seen_m'] is None else s['first_seen_m'])
            if worst['first_seen_m'] is None:
                self.tiles['seen'].set('never', '', worst['label'], 'serious')
            else:
                top = worst['first_seen_m'] >= worst['distances'][-1][0] - 1e-6
                self.tiles['seen'].set(f'{"≥" if top else ""}{worst["first_seen_m"]:.1f}', 'm',
                                       f'{worst["label"]} ({worst["by"]})',
                                       'serious' if any(s['first_seen_m'] is None for s in res.sweeps) else 'good')
        else:
            self.tiles['seen'].set('—', '', 'no bump or kerb')
        cl = [v for v in ([res.clearance['body_m']] if res.clearance and res.clearance['body_m'] is not None else [])
              + [d['body_m'] for d in res.drive_over if d['body_m'] is not None]]
        if cl:
            m = min(cl)
            self.tiles['clear'].set(f'{m * 100:.1f}', 'cm', 'the body\'s underside over the road',
                                    'serious' if m < 0 else 'warning' if m < 0.03 else 'good')
        else:
            self.tiles['clear'].set('—', '', 'flat ground')
        cases = {c['name']: c for c in res.motion.get('cases', [])}
        hb = cases.get('Hard braking')
        if hb is not None:
            hz = [h for h in hb['horizon'] if h['facing'] == 'forward']
            note = ', '.join(f'{h["tag"]} horizon {abs(h["dv_px"]):.0f} px' for h in hz) or 'no forward camera'
            self.tiles['brake'].set(f'{hb["rel"]["pitch_deg"]:+.2f}', '° pitch', note)
        cl_ = cases.get('Cornering left')
        if cl_ is not None:
            self.tiles['corner'].set(f'{cl_["rel"]["roll_deg"]:+.2f}', '° roll',
                                     f'{cl_["rel"]["roll_deg"] / 0.5:.1f}°/g roll gradient')

    def _fill_road(self, res):
        t = self.t_road
        t.setRowCount(len(res.features))
        sweeps = {s['index']: s for s in res.sweeps}
        over = {d['index']: d for d in res.drive_over}
        for i, f in enumerate(res.features):
            _cell(t, i, 0, TR.describe(f), tip=TR.TYPICAL[f['kind']])
            sw = sweeps.get(i)
            if sw is not None:
                none = sw['first_seen_m'] is None
                _cell(t, i, 1, 'never' if none else _m(sw['first_seen_m']), True,
                      QColor(STATUS['serious']) if none else None)
                _cell(t, i, 2, sw['by'] or '—')
                lost = sw['lost_m']
                _cell(t, i, 3, '—' if lost is None else ('to bumper' if sw['across'] else 'to side') if lost <= 0
                      else _m(lost), True)
                _cell(t, i, 4, ', '.join(f'{a:.1f}' if a == b else f'{a:.1f}-{b:.1f}' for a, b in sw['gaps']) or '—')
            else:
                for c in range(1, 5):
                    _cell(t, i, c, '—', c in (1, 3))
            d = over.get(i)
            if d is not None and d['body_m'] is not None:
                col = QColor(STATUS['serious' if d['body_m'] < 0 else 'warning' if d['body_m'] < 0.03 else 'good'])
                _cell(t, i, 5, f'{d["body_m"] * 100:.1f}', True, col, f'{d["body_where"]}, with it '
                      f'{d["body_at_m"]:.1f} m from the bumper; devices: {d["device"]} '
                      f'{_m(d["device_m"] * 100 if d["device_m"] is not None else None)} cm')
            else:
                _cell(t, i, 5, '—', True)

    def _fill_cams(self, res):
        t = self.t_cams
        t.setRowCount(len(res.per_camera))
        for r, c in enumerate(res.per_camera):
            _cell(t, r, 0, c['tag'], swatch=THEME.cam(c['slot']))
            _cell(t, r, 1, f'{c["seen_flat_m2"]:.2f}', True)
            _cell(t, r, 2, f'{c["seen_road_m2"]:.2f}', True)
            _cell(t, r, 3, f'{c["hidden_m2"]:.2f}', True, QColor(STATUS['warning']) if c['hidden_m2'] > 0.05 else None)
            _cell(t, r, 4, f'{c["mean_density_flat"]:.1f}', True)
            _cell(t, r, 5, f'{c["mean_density_road"]:.1f}', True)

    def _fill_motion(self, res):
        t = self.t_motion
        cases = res.motion.get('cases', [])
        t.setRowCount(len(cases))
        for r, c in enumerate(cases):
            name = c['name'] if c['name'] != 'Sliders' else f'Sliders {c["long_g"]:+.2f} g, {c["lat_g"]:+.2f} g'
            _cell(t, r, 0, name, tip=f'{c["long_g"]:+g} g longitudinal (+ accelerating), {c["lat_g"]:+g} g lateral '
                                     '(+ turning left)'
                  + (f'; the {", ".join(c["lifted"])} wheel would lift' if c['lifted'] else '')
                  + (f'; on the bump stops: {", ".join(c["bump_stops"])}' if c['bump_stops'] else ''))
            _cell(t, r, 1, f'{c["rel"]["pitch_deg"]:+.2f}', True)
            _cell(t, r, 2, f'{c["rel"]["roll_deg"]:+.2f}', True)
            _cell(t, r, 3, f'{c["rel"]["heave_mm"]:+.0f}', True)
            hz = [h for h in c['horizon'] if h['facing'] == 'forward']
            _cell(t, r, 4, ' · '.join(f'{h["tag"]} {abs(h["dv_px"]):.0f} {"↑" if h["dv_px"] < 0 else "↓"}'
                                      for h in hz) or '—',
                  tip='\n'.join(f'{h["tag"]}: {h["dv_px"]:+.1f} px ({h["d_deg"]:+.3f}°, positive up), tilt '
                                f'{h["d_tilt_deg"]:+.2f}°' for h in c['horizon']))
            im = c['image']
            if im is not None:
                col = QColor(STATUS['good' if im.status == 'good' else 'warning' if im.status == 'warning'
                                    else 'serious'])
                _cell(t, r, 5, f'{im.delta_blind:+.2f}', True)
                _cell(t, r, 6, f'{im.ground_shift_cm:.1f}', True, col)
                _cell(t, r, 7, f'{im.seam_ghost_cm:.1f}', True, col)
            else:
                for k in (5, 6, 7):
                    _cell(t, r, k, '—', True)

    # ------------------------------------------------------------------ export
    def export(self, folder=None):
        res = self.ctl.result
        if res is None:
            return []
        if folder is None:
            folder = QFileDialog.getExistingDirectory(self, 'Export the road and body motion study into')
            if not folder:
                return []
        paths = TR.export(res, self.ctx.store.project, folder)
        self.ctx.msg(f'Exported the road and body motion study to {folder}', 5000)
        return paths
