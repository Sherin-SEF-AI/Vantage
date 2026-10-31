"""Analysis ▸ Critical targets: the objects the rig must resolve at known places (a child in front of the bumper, a
traffic light, a kerb, a lane marking on a curve), which cameras resolve them, and how far each camera's mount may be
off before it stops resolving them: the envelope the bracket and the install must hold.

The controller (ctx.critical) keeps the study's result for the project: it runs as a background job (Run, with the
main window's job bar and Cancel) and again by itself after edits while the page shows. The plan and the 3D view of
the page draw the targets (green resolved, red not, the cameras that resolve one joined to it); paint_plan,
scene_tris and scene_lines draw the same in the main views (View ▸ Overlays ▸ Critical targets)."""
from __future__ import annotations

import json
import math
import threading
import traceback

import numpy as np
from PyQt6.QtCore import QObject, QPointF, QRectF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen, QPolygonF
from PyQt6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QFileDialog, QGridLayout, QHBoxLayout,
                             QHeaderView, QLineEdit, QMenu, QProgressBar, QScrollArea, QSplitter, QStackedWidget,
                             QTableWidget, QTableWidgetItem, QToolTip, QVBoxLayout, QWidget)

from ...core import critical as CR
from ..charts import ChartBase
from ..planview import PlanView
from ..scene import MainScene
from ..theme import STATUS, THEME, mono, sans
from ..widgets import fit_width, ROW_H, ScrubSpinBox, SegmentedControl, StatusLine, button, hline, label

JOB = 'critical'
OVERLAY = 'critical'                 # View ▸ Overlays key the main views draw the targets under
AUTO_MAX_S = 8.0                     # re-run by itself after an edit while the last run took less than this
EMPTY = 'Add a critical target: Add ▸ Child 2 m ahead, a traffic light, a kerb, a lane marking…'
AXIS_LABEL = {'yaw': 'Yaw ψ', 'pitch': 'Pitch θ', 'roll': 'Roll φ', 'x': 'X', 'y': 'Y', 'z': 'Z'}
WHY_TEXT = {'pixels': 'too few pixels across it', 'seen': 'too little of it seen', 'focus': 'out of focus',
            'surface': 'the lens would be inside the body, the rig or a housing', 'open': 'no loss within the search'}
STANDING = tuple(k for k in CR.KINDS if k not in CR.LINEAR)
ALL = CR.KINDS
# (key, label, unit, decimals, step, scale, kinds, tip); scale: stored value × scale = shown value
FIELDS = (
    ('x', 'X', 'm', 3, 0.05, 1.0, ALL, 'Where it stands (a kerb or a marking: where it starts), vehicle frame: x '
                                       'forward from the rear axle'),
    ('y', 'Y', 'm', 3, 0.05, 1.0, ALL, 'Vehicle frame: y to the left of the centre line'),
    ('yaw', 'Heading', '°', 1, 1.0, 1.0, CR.LINEAR, 'The direction it runs from its start, counter-clockwise from '
                                                     'forward'),
    ('height', 'Height', 'm', 2, 0.05, 1.0, STANDING + ('kerb',), 'Its height (a kerb: the face above the road)'),
    ('width', 'Width', 'm', 2, 0.05, 1.0, STANDING + ('lane',), 'Its width (a lane marking: the painted line\'s)'),
    ('base', 'Above ground', 'm', 2, 0.1, 1.0, STANDING, 'Height of its bottom above the ground (a sign or a traffic '
                                                         'light head on its post)'),
    ('length', 'Length', 'm', 1, 0.5, 1.0, CR.LINEAR, 'Its length from the start along its heading'),
    ('radius', 'Curve radius', 'm', 0, 5.0, 1.0, ('lane',), 'The marking\'s curve: positive bends left, negative '
                                                            'right, 0 straight'),
    ('min_px', 'Needs', 'px', 1, 1.0, 1.0, ALL, 'Pixels a camera must put across its height (a lane marking: across '
                                                'its width) for the detector you use'),
    ('min_share', 'Seen at least', '%', 0, 5.0, 100.0, ALL, 'Share of its face (a kerb or a marking: of its length) '
                                                            'a camera must see'),
    ('min_points', 'Lidar points', 'pts', 0, 1.0, 1.0, ALL, 'Lidar points per frame it also needs (0: the lidars '
                                                            'are counted, not required)'),
    ('rcs', 'Radar RCS', 'm²', 2, 0.1, 1.0, STANDING, 'Its radar cross-section, to report which radars detect it '
                                                      '(0: radars not relied on)'),
)
TARGET_COLS = ('', 'Target', 'Needs', 'Status')
LOOK_COLS = (('Camera', 'The camera'), ('View %', 'Share of the target inside the image and the lens circle'),
             ('Seen %', 'Share of it in view and not hidden by the body, the rig or a housing'),
             ('Px', 'Pixels across its height (a lane marking: its width) of the part seen'),
             ('Dist m', 'From the camera to the target\'s middle'), ('Focus', 'Within the lens\'s focus range'),
             ('Meets', 'Whether this camera meets the target\'s requirement'))


# =========================================================================== controller
class _Sig(QObject):
    done = pyqtSignal(int, object)
    failed = pyqtSignal(int, str)
    progress = pyqtSignal(int, float)


def controller(ctx) -> 'CriticalController':
    c = getattr(ctx, 'critical', None)
    if c is None:
        c = ctx.critical = CriticalController(ctx)
    return c


def inputs_key(project) -> str:
    """Everything the study reads, as text: a result made from other inputs is stale."""
    m = project.vehicle.mesh
    return json.dumps({'mesh': [id(m), getattr(m, 'n_tris', 0), project.vehicle.occluder_target],
                       'variant': project.variant.to_dict(), 'rig': [r.to_dict() for r in project.rig],
                       'critical': project.analysis.critical, 'housings': project.analysis.housings,
                       'fleet': (project.fleet or {}).get('settings') if isinstance(project.fleet, dict) else None,
                       'library': project.section('library')}, sort_keys=True, default=str)


class CriticalController(QObject):
    """The project's critical targets study and the target selected on the page."""
    changed = pyqtSignal()
    progress = pyqtSignal(float)
    selected = pyqtSignal(str)

    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        self.result: CR.CriticalResult | None = None
        self.stale = False
        self.busy = False
        self.status = ''
        self.active = False
        self.sel = ''                        # uid of the selected target
        self.placing = False                 # a click in the plan or the 3D view moves the selected target there
        self._gen = 0
        self._for = None                     # (project, inputs key) of the result
        self.sig = _Sig()
        self.sig.done.connect(self._done)
        self.sig.failed.connect(self._failed)
        self.sig.progress.connect(lambda g, f: self.progress.emit(f) if g == self._gen else None)
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(400)
        self.timer.timeout.connect(lambda: self.run(quiet=True))
        ctx.store.changed.connect(self._on_edit)
        ctx.store.project_replaced.connect(self._reset)

    # ---- settings (saved with the project; edits are undoable)
    @property
    def settings(self) -> CR.CriticalSettings:
        return CR.settings_of(self.ctx.store.project)

    def _store(self, st: CR.CriticalSettings, text, merge_key=None) -> bool:
        new = CR.CriticalSettings.from_dict(st.to_dict()).to_dict()
        if new == self.settings.to_dict():
            return False

        def fn(p):
            p.analysis.critical = new
        self.ctx.store.edit('analysis', fn, text, merge_key=merge_key)
        return True

    def target(self, uid=None) -> dict | None:
        return self.settings.target(uid or self.sel)

    def select(self, uid):
        uid = uid or ''
        if uid != self.sel:
            self.sel = uid
            self.selected.emit(uid)

    def add(self, label) -> str:
        """Add a preset target (a PRESET_PLACES label or a kind); returns its uid (selected)."""
        st = self.settings
        if len(st.targets) >= CR.MAX_TARGETS:
            self.ctx.msg(f'At most {CR.MAX_TARGETS} critical targets.', 4000)
            return ''
        t = CR.preset(self.ctx.store.project, label, st.targets)
        st.targets.append(t)
        self._store(st, f'Critical targets · added {t["name"]}')
        self.select(t['uid'])
        return t['uid']

    def remove(self, uid=None) -> bool:
        uid = uid or self.sel
        st = self.settings
        t = st.target(uid)
        if t is None:
            return False
        st.targets = [x for x in st.targets if x['uid'] != uid]
        self._store(st, f'Critical targets · removed {t["name"]}')
        if self.sel == uid:
            self.select(st.targets[-1]['uid'] if st.targets else '')
        return True

    def update(self, uid, text=None, merge_key=None, **fields) -> bool:
        """Change fields of a target as one undo step; False when nothing changes (or the value is not usable)."""
        st = self.settings
        t = st.target(uid)
        if t is None:
            return False
        if 'kind' in fields and fields['kind'] != t['kind'] and fields['kind'] in CR.KINDS:
            t.update({k: v for k, v in CR.PRESETS[fields['kind']].items()})      # the new kind's typical figures
        t.update(fields)
        what = text or ', '.join(f'{k} {v:g}' if isinstance(v, float) else f'{k} {v}' for k, v in fields.items())
        return self._store(st, f'Critical targets · {t["name"]} {what}', merge_key)

    def move(self, uid, x, y, merge=True) -> bool:
        return self.update(uid, 'moved', ('critical-move', uid) if merge else None, x=round(float(x), 3),
                           y=round(float(y), 3))

    def set_tolerance(self, **kw) -> bool:
        st = self.settings
        for k, v in kw.items():
            setattr(st, k, v)
        return self._store(st, 'Critical targets · ' + ', '.join(
            {'install_mm': 'install tolerance {v:g} mm', 'install_deg': 'install tolerance {v:g}°',
             'search_deg': 'search ±{v:g}°', 'search_mm': 'search ±{v:g} mm'}.get(k, k + ' {v}').format(v=v)
            for k, v in kw.items()))

    # ---- state
    def _quick(self):
        return self.result is None or self.result.metrics.get('seconds', 99) < AUTO_MAX_S

    def _on_edit(self, *_):
        if not (self.active or self.result is not None):
            return
        if self.sel and self.settings.target(self.sel) is None:
            self.select('')
        if self._for is None or inputs_key(self.ctx.store.project) != self._for[1]:
            if self.result is not None and not self.stale:
                self.stale = True
                self.changed.emit()
            if self.active and self._quick():
                self.timer.start()

    def set_active(self, on: bool):
        self.active = on
        if on and (self.result is None or self.stale) and self._quick():
            self.timer.start()

    def _reset(self, *_):
        self._gen += 1
        self.result, self._for, self.stale, self.busy, self.status = None, None, False, False, ''
        self.placing = False
        self.select('')
        self.changed.emit()
        if self.active:
            self.timer.start()

    # ---- run
    def run(self, quiet=False) -> bool:
        ctx = self.ctx
        p = ctx.store.project
        win = ctx.window
        if not any(t['enabled'] for t in CR.settings_of(p).targets):
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
        self.status = 'Measuring the critical targets and the mounting envelopes…'
        key = (p, inputs_key(p))

        def work(progress, cancel):
            return key, CR.run(project, eng, progress=progress, cancel=cancel)
        if not quiet and win is not None and hasattr(win, 'run_job'):
            ok = win.run_job(JOB, 'Critical targets', work, lambda out: self._done(gen, out),
                             lambda m: self._failed(gen, m))
            self.changed.emit()
            return ok
        self.changed.emit()

        def thread():
            try:
                out = work(lambda f, *_: self.sig.progress.emit(gen, f), lambda: gen != self._gen)
                self.sig.done.emit(gen, out)
            except RuntimeError:             # the window closed while computing
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
        self.status = (f'{m["resolved"]} of {m["targets"]} target{"s" * (m["targets"] != 1)} resolved · '
                       f'{m["seconds"]:.1f} s')
        if self.sel and res.target(self.sel) is None:
            self.select('')
        if not self.sel and res.targets:
            self.select(res.targets[0].target['uid'])
        self.changed.emit()
        if self.ctx.window is not None and any(f['kind'] == 'tight' for f in res.findings):
            self.ctx.analysis.request()          # the Issues list picks up the envelope findings (cached result)
        if self.stale and self.active and self._quick():
            self.timer.start()

    def _failed(self, gen, msg):
        if gen != self._gen:
            return
        self.busy = False
        self.status = 'Failed: ' + msg.splitlines()[0]
        self.changed.emit()


# =========================================================================== drawing (page and main views)
def _targets(ctx):
    st = CR.settings_of(ctx.store.project)
    ctl = getattr(ctx, 'critical', None)
    return [t for t in st.targets if t['enabled']], ctl


def _result(ctx, ctl):
    """The verdicts to draw: the page's result while it is up to date, else the newest one cached for exactly this
    design (the Issues list's check), else None."""
    if ctl is not None and ctl.result is not None and not ctl.stale:
        return ctl.result
    try:
        return CR.latest(ctx.store.project, ctx.analysis.engine.vg)
    except Exception:                    # a design being replaced: no verdicts this frame
        return None


def _tone(res, t) -> QColor:
    r = res.target(t['uid']) if res is not None else None
    if r is None:
        return THEME.color('ink2')
    return QColor(STATUS['good' if r.met else 'critical'])


def _footprint(t) -> np.ndarray:
    """The target's outline on the ground (n, 2): a standing target as a square of its width, a kerb a thin strip
    along its line, a marking its painted strip."""
    if t['kind'] in CR.LINEAR:
        L = float(t['length'])
        s = np.linspace(0.0, L, max(int(L / 0.25), 2) + 1)
        C, n = CR._arc(t, s)
        w = t['width'] if t['kind'] == 'lane' else 0.15
        return np.vstack([C + n * w / 2, (C - n * w / 2)[::-1]])
    h = max(t['width'], 0.1) / 2
    a = math.radians(t['yaw'])
    R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    return np.array([t['x'], t['y']]) + np.array([[-h, -h], [h, -h], [h, h], [-h, h]]) @ R.T


def _centre(t):
    if t['kind'] in CR.LINEAR:
        C, _ = CR._arc(t, np.array([t['length'] / 2]))
        return float(C[0, 0]), float(C[0, 1])
    return t['x'], t['y']


def paint_plan(view, p: QPainter, ctl=None, force=False):
    """The critical targets in a plan view (PlanView: w2s): footprints coloured by verdict, a line from each camera
    that resolves one, names. Drawn when View ▸ Overlays ▸ Critical targets is on (force: always)."""
    ctx = view.ctx
    if not (force or ctx.overlays.get(OVERLAY)):
        return
    targets, c0 = _targets(ctx)
    ctl = ctl or c0
    if not targets:
        return
    res = _result(ctx, ctl)
    cams = {c.uid: c for c in ctx.store.project.variant.cameras}
    p.save()
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    sel = ctl.sel if ctl is not None else ''
    placed = []
    for t in targets:
        col = _tone(res, t)
        cx, cy = _centre(t)
        q = view.w2s(cx, cy)
        r = res.target(t['uid']) if res is not None else None
        if r is not None:
            for lk in r.looks:
                c = cams.get(lk.uid)
                if lk.met and c is not None:
                    pen = QPen(THEME.cam(c.slot, 0.75), 1.2, Qt.PenStyle.DashLine)
                    p.setPen(pen)
                    p.drawLine(view.w2s(c.x, c.y), q)
        poly = QPolygonF([view.w2s(x, y) for x, y in _footprint(t)])
        fill = QColor(col)
        fill.setAlphaF(0.35)
        p.setBrush(fill)
        p.setPen(QPen(col, 2.2 if t['uid'] == sel else 1.4))
        p.drawPolygon(poly)
        if t['kind'] not in CR.LINEAR:
            p.drawEllipse(q, 3.0, 3.0)
        if t['uid'] == sel:
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.setPen(QPen(QColor(STATUS['warning']), 1.5))
            p.drawEllipse(q, 11.0, 11.0)
        p.setFont(sans(11, QFont.Weight.DemiBold if t['uid'] == sel else None))
        p.setPen(THEME.color('ink'))
        fm = p.fontMetrics()
        w, h = fm.horizontalAdvance(t['name']), fm.height()
        for dx, dy in ((13, -6 - h), (13, 6), (-13 - w, -6 - h), (-13 - w, 6), (13, -6 - 2 * h), (13, 6 + h)):
            r = QRectF(q.x() + dx, q.y() + dy, w, h)               # the first place free of the labels drawn
            if not any(r.intersects(o) for o in placed):
                break
        placed.append(r)
        p.drawText(r, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, t['name'])
    p.restore()


def pick_plan(view, pos, ctx, r_px=12.0) -> str:
    """uid of the target under screen position pos in a plan view ('' none)."""
    targets, _ = _targets(ctx)
    best, bd = '', r_px
    for t in targets:
        pts = [(view.w2s(x, y)) for x, y in _footprint(t)] + [view.w2s(*_centre(t))]
        d = min(math.hypot(q.x() - pos.x(), q.y() - pos.y()) for q in pts)
        if t['kind'] in CR.LINEAR:
            poly = QPolygonF(pts[:-1])
            if poly.containsPoint(QPointF(pos), Qt.FillRule.OddEvenFill):
                d = 0.0
        if d < bd:
            best, bd = t['uid'], d
    return best


def _oriented_box(c, half, yaw):
    """Triangles and corner normals of a box centred at c (3,) with half sizes (3,), turned yaw about z."""
    from .. import overlays3d as O3
    T, N = O3.box(-np.asarray(half, float), np.asarray(half, float))
    a = math.radians(yaw)
    R = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1.0]])
    return T @ R.T + np.asarray(c, float), N @ R.T


def scene_tris(ctx, gl: bool, ctl=None, force=False) -> list:
    """The critical targets in 3D: standing targets as boxes of their size (a traffic light's or a sign's on a thin
    post), kerbs as their face, lane markings as their painted strip; coloured by verdict."""
    if not (force or ctx.overlays.get(OVERLAY)):
        return []
    targets, c0 = _targets(ctx)
    ctl = ctl or c0
    if not targets:
        return []
    from .. import overlays3d as O3
    from ..viewport3d import LIGHT
    res = _result(ctx, ctl)
    tri, col, nrm = [], [], []
    for t in targets:
        c = _tone(res, t)
        rgb = np.array([c.redF(), c.greenF(), c.blueF()])
        if t['kind'] == 'lane':
            L = float(t['length'])
            s = np.linspace(0.0, L, max(int(L / 0.5), 2) + 1)
            C, n = CR._arc(t, s)
            a, b = C + n * t['width'] / 2, C - n * t['width'] / 2
            z = 0.004
            T = []
            for i in range(len(s) - 1):
                p0, p1, p2, p3 = (np.r_[a[i], z], np.r_[a[i + 1], z], np.r_[b[i + 1], z], np.r_[b[i], z])
                T += [[p0, p1, p2], [p0, p2, p3]]
            T = np.array(T)
            N = np.broadcast_to(np.array([0, 0, 1.0]), T.shape).copy()
            parts = [(T, N)]
        elif t['kind'] == 'kerb':
            L = float(t['length'])
            a = math.radians(t['yaw'])
            mid = np.array([t['x'] + math.cos(a) * L / 2, t['y'] + math.sin(a) * L / 2, t['height'] / 2])
            parts = [_oriented_box(mid, (L / 2, 0.075, t['height'] / 2), t['yaw'])]
        else:
            w = max(t['width'], 0.05)
            d = min(w, 0.3)
            box = CR.vehicle_box(ctx.store.project)
            yaw = math.degrees(math.atan2(t['y'], t['x'] - box['mid_x'])) + 90.0     # broadside to the car
            parts = [_oriented_box((t['x'], t['y'], t['base'] + t['height'] / 2), (w / 2, d / 2, t['height'] / 2),
                                   yaw)]
            if t['base'] > 0.05:
                parts.append(_oriented_box((t['x'], t['y'], t['base'] / 2), (0.04, 0.04, t['base'] / 2), yaw))
        for T, N in parts:
            fn = N[:, 0]
            # the shader lights the faces itself (one row per face either way)
            lit = np.ones(len(T)) if gl else 0.55 + 0.45 * np.clip(fn @ LIGHT, 0, 1)
            shade = rgb[None] * lit[:, None]
            tri.append(T)
            col.append(np.c_[np.clip(shade, 0, 1), np.full(len(T), 0.85)])
            nrm.append(N)
    if not tri:
        return []
    return [O3.tri_draw(np.concatenate(tri), np.concatenate(col), normals=np.concatenate(nrm) if gl else None)]


def scene_lines(ctx, ctl=None, force=False) -> list:
    """3D sight lines from each camera that resolves a target to it (dashed, the camera's colour)."""
    if not (force or ctx.overlays.get(OVERLAY)):
        return []
    targets, c0 = _targets(ctx)
    ctl = ctl or c0
    res = _result(ctx, ctl) if targets else None
    if not targets or res is None:
        return []
    from ..viewport3d import LineDraw
    cams = {c.uid: c for c in ctx.store.project.variant.cameras}
    out = []
    for t in targets:
        r = res.target(t['uid'])
        if r is None:
            continue
        x, y = _centre(t)
        z = t['base'] + t['height'] / 2 if t['kind'] not in CR.LINEAR else 0.01
        for lk in r.looks:
            c = cams.get(lk.uid)
            if lk.met and c is not None:
                out.append(LineDraw(np.array([c.pos(), [x, y, z]], float), THEME.cam(c.slot, 0.8), 1.2, dash=True,
                                    dash_len=0.08))
    return out


# =========================================================================== the page's plan and 3D view
class CriticalPlan(PlanView):
    """The plan of the page: the coverage footprints with the targets; drag a target to move it, or click to put the
    selected one there while placing."""

    def __init__(self, ctx, ctl):
        super().__init__(ctx, 'footprints', interactive=False)
        self.ctl = ctl
        self.legend = False
        self._tdrag = None
        ctl.changed.connect(self.update)
        ctl.selected.connect(lambda *_: self.update())

    def paintEvent(self, e):
        super().paintEvent(e)
        p = QPainter(self)
        paint_plan(self, p, self.ctl, force=True)
        if self.ctl.placing:
            p.setFont(sans(11))
            p.setPen(THEME.color('ink'))
            p.drawText(QRectF(10, self.height() - 26, self.width() - 20, 18), Qt.AlignmentFlag.AlignLeft,
                       'Click where the selected target goes · Esc stops placing')
        p.end()

    def fit_targets(self):
        """Frame the vehicle and every target."""
        targets, _ = _targets(self.ctx)
        b = CR.vehicle_box(self.project)
        pts = [(b['front_x'], b['half_w']), (b['rear_x'], -b['half_w'])]
        for t in targets:
            pts += [tuple(q) for q in _footprint(t)]
        P = np.array(pts)
        lo, hi = P.min(0) - 1.0, P.max(0) + 1.0
        self.fit((lo[0], lo[1], hi[0], hi[1]))

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton and not e.modifiers() & Qt.KeyboardModifier.AltModifier:
            pos = e.position()
            w = self.s2w(pos.x(), pos.y())
            if self.ctl.placing and self.ctl.sel:
                self.ctl.move(self.ctl.sel, w[0], w[1], merge=False)
                self.ctl.placing = False
                self.ctl.changed.emit()
                return
            uid = pick_plan(self, pos, self.ctx)
            if uid:
                self.ctl.select(uid)
                t = self.ctl.target(uid)
                self._tdrag = {'uid': uid, 'start': (t['x'], t['y']), 'w0': w, 'moved': False}
                self.ctx.store.dragging = True
                return
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        d = self._tdrag
        if d is not None:
            pos = e.position()
            w = self.s2w(pos.x(), pos.y())
            fine = 0.1 if e.modifiers() & Qt.KeyboardModifier.ShiftModifier else 1.0
            dw = (w - d['w0']) * fine
            if np.hypot(*dw) > 1e-4:
                d['moved'] = True
                self.ctl.move(d['uid'], d['start'][0] + dw[0], d['start'][1] + dw[1], merge=True)
            return
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):
        if self._tdrag is not None:
            self._tdrag = None
            self.ctx.store.dragging = False
            self.update()
            return
        super().mouseReleaseEvent(e)

    def focusOutEvent(self, e):
        if self._tdrag is not None:
            self._tdrag = None
            self.ctx.store.dragging = False
        super().focusOutEvent(e)

    def keyPressEvent(self, e):
        if e.key() == Qt.Key.Key_Escape and self.ctl.placing:
            self.ctl.placing = False
            self.ctl.changed.emit()
            return
        super().keyPressEvent(e)


class CriticalScene(MainScene):
    """The 3D view of the page: the scene with the targets and the sight lines; a click picks a target, or puts the
    selected one on the ground there while placing."""

    def __init__(self, ctx, ctl):
        super().__init__(ctx)
        self.ctl = ctl
        ctl.changed.connect(self._emit)
        ctl.selected.connect(self._emit)

    def tris_extra(self, vp):
        return super().tris_extra(vp) + scene_tris(self.ctx, getattr(vp, 'renderer', None) is not None, self.ctl,
                                                   force=True)

    def lines(self, vp):
        return super().lines(vp) + scene_lines(self.ctx, self.ctl, force=True)

    def mouse_press(self, vp, e):
        if self.ctl.placing:
            return False                     # the click places
        return super().mouse_press(vp, e)

    def click(self, vp, e):
        pos = e.position()
        g = vp.ground_at(pos.x(), pos.y())
        if self.ctl.placing and self.ctl.sel and g is not None:
            self.ctl.move(self.ctl.sel, g[0], g[1], merge=False)
            self.ctl.placing = False
            self.ctl.changed.emit()
            return
        uid = self.pick_target(vp, pos)
        if uid:
            self.ctl.select(uid)
            return
        super().click(vp, e)

    def pick_target(self, vp, pos) -> str:
        """uid of the target whose footprint (or standing box's middle) is nearest the click, within 14 px."""
        targets, _ = _targets(self.ctx)
        best, bd = '', 14.0
        for t in targets:
            x, y = _centre(t)
            P = [(x, y, 0.0), (x, y, t['base'] + t['height'] / 2)] + [(a, b, 0.0) for a, b in _footprint(t)]
            s, _, ok = vp.project(np.array(P, float))
            for (sx, sy), good in zip(s, ok):
                if good:
                    d = math.hypot(sx - pos.x(), sy - pos.y())
                    if d < bd:
                        best, bd = t['uid'], d
        return best


# =========================================================================== envelope chart
class EnvelopeBars(ChartBase):
    """One camera's envelope: per axis a bar from how far it may be off one way to the other, on a linear scale of
    ± 5 × the install tolerance (a longer bar is clipped with an arrow), the tolerance dashed, green where the limit
    is at least the tolerance, red where tighter; the limit and the target that sets it on the right."""
    ROW, TOP, LEFT, RIGHT = 26, 26, 64, 300

    def __init__(self, parent=None):
        super().__init__(parent)
        self.title = 'Mounting envelope'
        self.rows = []                  # (axis, (lo, name, why), (hi, name, why))
        self.tol = {'mm': 10.0, 'deg': 1.0}
        self.note = ''
        self._hits = []
        self.setMinimumHeight(self.TOP + 6 * self.ROW + 24)

    def set_data(self, rows, tol, title='Mounting envelope', note=''):
        self.rows, self.tol, self.title, self.note = rows, tol, title, note
        self.hover = None
        self.tip = None
        self.update()

    def _unit(self, ax):
        return self.tol['deg'] if ax in CR.ANGLE_AXES else self.tol['mm'] / 1000.0

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.paint_title(p, self.note)
        self._hits = []
        if not self.rows:
            p.setFont(sans(11))
            p.setPen(THEME.color('muted'))
            p.drawText(QRectF(0, self.TOP, self.width(), 40), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop,
                       'Select a camera that resolves a target (after a run).')
            p.end()
            return
        x0, x1 = self.LEFT, max(self.LEFT + 80, self.width() - self.RIGHT)
        cx, half = (x0 + x1) / 2, (x1 - x0) / 2
        for i, (ax, lo, hi) in enumerate(self.rows):
            y = self.TOP + i * self.ROW
            unit = self._unit(ax)
            span = 5.0 * max(unit, 1e-12)
            k = half / span
            p.setFont(sans(11))
            p.setPen(THEME.color('ink2'))
            p.drawText(QRectF(0, y, self.LEFT - 6, self.ROW), Qt.AlignmentFlag.AlignVCenter, AXIS_LABEL[ax])
            p.setPen(QPen(THEME.color('strong'), 1))
            p.drawLine(QPointF(x0, y + self.ROW / 2), QPointF(x1, y + self.ROW / 2))
            for side, (v, name, why) in ((-1, lo), (1, hi)):
                ok = why in ('surface', 'open') or v >= unit - 1e-12
                col = QColor(STATUS['good' if ok else 'critical'])
                if why == 'surface':
                    col = THEME.color('ink2')
                L = min(v, span) * k
                r = QRectF(cx - L, y + 6, L, self.ROW - 12) if side < 0 else QRectF(cx, y + 6, L, self.ROW - 12)
                p.fillRect(r, col)
                if v > span:                    # clipped: an arrow head at the end
                    tip = cx + side * (half + 7)
                    p.setBrush(col)
                    p.setPen(Qt.PenStyle.NoPen)
                    p.drawPolygon(QPolygonF([QPointF(tip, y + self.ROW / 2), QPointF(cx + side * half, y + 5),
                                             QPointF(cx + side * half, y + self.ROW - 5)]))
                self._hits.append((QRectF(min(r.left(), r.right()) - 2, y, max(r.width(), 6) + 4, self.ROW),
                                   (ax, side, v, name, why)))
            p.setPen(QPen(THEME.color('ink'), 1))
            p.drawLine(QPointF(cx, y + 3), QPointF(cx, y + self.ROW - 3))
            pen = QPen(THEME.color('ink2'), 1, Qt.PenStyle.DashLine)
            p.setPen(pen)
            for side in (-1, 1):
                xt = cx + side * unit * k
                p.drawLine(QPointF(xt, y + 2), QPointF(xt, y + self.ROW - 2))
            lost = [s for s in (lo, hi) if s[2] not in ('surface', 'open')]
            who = min(lost, key=lambda s: s[0])[1] if lost else \
                ('body surface' if 'surface' in (lo[2], hi[2]) else 'no loss in the search')
            p.setFont(mono(11))
            p.setPen(THEME.color('ink'))
            p.drawText(QRectF(x1 + 14, y, 160, self.ROW), Qt.AlignmentFlag.AlignVCenter,
                       f'−{_lim(ax, lo[0], lo[2])} +{_lim(ax, hi[0], hi[2])}')
            p.setFont(sans(10))
            p.setPen(THEME.color('muted'))
            p.drawText(QRectF(x1 + 178, y, self.RIGHT - 180, self.ROW), Qt.AlignmentFlag.AlignVCenter,
                       p.fontMetrics().elidedText(who, Qt.TextElideMode.ElideRight, self.RIGHT - 182))
        p.setFont(sans(10))
        p.setPen(THEME.color('muted'))
        p.drawText(QRectF(x0, self.TOP + 6 * self.ROW + 2, self.width() - x0, 16), Qt.AlignmentFlag.AlignLeft,
                   f'dashed: the install tolerance ±{self.tol["deg"]:g}° / ±{self.tol["mm"]:g} mm · each bar spans '
                   f'±5 × the tolerance; grey: the body\'s surface')
        self.paint_tip(p)
        p.end()

    def mouseMoveEvent(self, e):
        pos = e.position()
        hit = next((h for r, h in self._hits if r.contains(pos)), None)
        if hit != self.hover:
            self.hover = hit
            if hit is None:
                self.tip = None
            else:
                ax, side, v, name, why = hit
                self.set_tip(pos, f'{AXIS_LABEL[ax]} {"−" if side < 0 else "+"}{_lim(ax, v, why)}',
                             f'set by {name}' if why not in ('surface', 'open') else '', WHY_TEXT.get(why, why))
            self.update()

    def leaveEvent(self, e):
        self.hover = None
        self.tip = None
        self.update()


def _lim(ax, v, why=''):
    s = f'{v:.2f}°' if ax in CR.ANGLE_AXES else f'{v * 1000:.1f} mm'
    return ('≥' + s) if why == 'open' else s


# =========================================================================== the page
def _table(cols, tips=None):
    t = QTableWidget(0, len(cols))
    t.verticalHeader().setVisible(False)
    t.verticalHeader().setDefaultSectionSize(ROW_H)
    t.horizontalHeader().setFixedHeight(ROW_H)
    t.setShowGrid(False)
    t.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    t.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    t.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
    for c, h in enumerate(cols):
        it = QTableWidgetItem(h)
        if tips:
            it.setToolTip(tips[c])
        t.setHorizontalHeaderItem(c, it)
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


class CriticalPage(QWidget):
    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        self.ctl = controller(ctx)
        self.sel_cam = ''                   # the camera whose envelope is drawn
        self.scope = 'target'               # the envelope for the selected target, or the camera's tightest
        self._syncing = False
        self._shown = None
        self.findings: list[StatusLine] = []
        self.setObjectName('base')
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        split = QSplitter(Qt.Orientation.Horizontal)
        split.setHandleWidth(2)
        split.addWidget(self._left())
        # ---------------- centre: plan or 3D
        cw = QWidget()
        cw.setObjectName('panel')
        cl = QVBoxLayout(cw)
        cl.setContentsMargins(8, 4, 8, 6)
        cl.setSpacing(4)
        head = QHBoxLayout()
        self.view_mode = SegmentedControl([('plan', 'Top'), ('3d', '3D')], 'plan')
        self.view_mode.changed.connect(self._set_view)
        head.addWidget(self.view_mode)
        head.addStretch(1)
        self.b_fit = button('Fit', 'ghost', None, 'Frame the vehicle and every target')
        self.b_fit.clicked.connect(self._fit)
        head.addWidget(self.b_fit)
        cl.addLayout(head)
        self.stack = QStackedWidget()
        self.plan = CriticalPlan(ctx, self.ctl)
        self.stack.addWidget(self.plan)
        self.view3d = None                  # made when first shown
        cl.addWidget(self.stack, 1)
        split.addWidget(cw)
        # ---------------- right: cameras on the target, the envelope, findings
        right = QSplitter(Qt.Orientation.Vertical)
        right.setHandleWidth(2)
        tw = QWidget()
        tw.setObjectName('panel')
        tl = QVBoxLayout(tw)
        tl.setContentsMargins(8, 6, 8, 6)
        tl.setSpacing(4)
        self.look_head = label('Cameras', bold=True)
        tl.addWidget(self.look_head)
        self.looks = _table([h for h, _ in LOOK_COLS], [t for _, t in LOOK_COLS])
        hh = self.looks.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for c in range(1, len(LOOK_COLS)):
            hh.setSectionResizeMode(c, QHeaderView.ResizeMode.ResizeToContents)
        self.looks.cellClicked.connect(self._look_clicked)
        tl.addWidget(self.looks, 1)
        self.sensors_lbl = label('', muted=True, small=True, wrap=True)
        tl.addWidget(self.sensors_lbl)
        right.addWidget(tw)
        ew = QWidget()
        ew.setObjectName('panel')
        el = QVBoxLayout(ew)
        el.setContentsMargins(8, 6, 8, 6)
        el.setSpacing(4)
        eh = QHBoxLayout()
        self.cam_pick = QComboBox()
        self.cam_pick.setToolTip('The camera whose envelope is drawn')
        self.cam_pick.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.cam_pick.setMinimumWidth(130)
        self.cam_pick.activated.connect(self._cam_picked)
        eh.addWidget(self.cam_pick)
        self.scope_ctl = SegmentedControl([('target', 'This target'), ('all', 'All its targets')], 'target')
        self.scope_ctl.setToolTip('The envelope for the selected target, or the tightest over every target the '
                                  'camera resolves (what its bracket must hold)')
        self.scope_ctl.changed.connect(self._set_scope)
        eh.addWidget(self.scope_ctl)
        eh.addStretch(1)
        el.addLayout(eh)
        self.bars = EnvelopeBars()
        el.addWidget(self.bars)
        self.env_note = label('', muted=True, small=True, wrap=True)
        el.addWidget(self.env_note)
        right.addWidget(ew)
        fw = QWidget()
        fw.setObjectName('panel')
        fl = QVBoxLayout(fw)
        fl.setContentsMargins(8, 6, 8, 6)
        fl.setSpacing(4)
        fl.addWidget(label('Findings', bold=True))
        self.find_host = QWidget()
        self.find_lay = QVBoxLayout(self.find_host)
        self.find_lay.setContentsMargins(0, 0, 0, 0)
        self.find_lay.setSpacing(4)
        self.find_lay.addStretch(1)
        self.notes = label(f'Assumptions. {CR.ASSUMPTIONS}\n\nLimits. {CR.LIMITS_TEXT}\n\nTypical figures. '
                           f'{CR.TYPICAL}', muted=True, small=True, wrap=True)
        self.find_lay.addSpacing(8)
        self.find_lay.addWidget(self.notes)
        sc = QScrollArea()
        sc.setWidgetResizable(True)
        sc.setWidget(self.find_host)
        fl.addWidget(sc, 1)
        right.addWidget(fw)
        right.setSizes([300, 300, 260])
        split.addWidget(right)
        split.setSizes([392, 720, 580])
        lay.addWidget(split)
        self.ctl.changed.connect(self.refresh)
        self.ctl.selected.connect(self._on_selected)
        self.ctl.progress.connect(lambda f: self.bar.setValue(int(f * 1000)))
        ctx.store.changed.connect(self._on_store)
        ctx.store.project_replaced.connect(self._on_store)
        THEME.changed.connect(self._retheme)           # a bound slot: disconnected when the page goes
        self._sync_fields()
        self.refresh()

    # ------------------------------------------------------------------ left: targets and settings
    def _left(self):
        w = QWidget()
        w.setObjectName('panel')
        ll = QVBoxLayout(w)
        ll.setContentsMargins(8, 6, 8, 8)
        ll.setSpacing(4)
        top = QHBoxLayout()
        top.addWidget(label('Critical targets'))
        top.addStretch(1)
        top.addWidget(label('what the rig must resolve', muted=True, small=True))
        ll.addLayout(top)
        ll.addWidget(hline())
        ll.addWidget(label('Objects the rig must resolve at known places, which cameras resolve them, and how far each '
                           'camera\'s mount may be off (each axis alone, and all at once) before it stops resolving '
                           'them: the tolerance its bracket and the install must hold. Sizes and needs are typical: '
                           'set your detector\'s.', muted=True, small=True, wrap=True))
        row = QHBoxLayout()
        self.b_add = button('Add', None, 'plus', 'Add a critical target from the presets (placed relative to the body)')
        menu = QMenu(self.b_add)
        for lab, *_ in CR.PRESET_PLACES:
            menu.addAction(lab, lambda lab=lab: self.ctl.add(lab))
        menu.addSeparator()
        menu.addAction('Custom object', lambda: self.ctl.add('object'))
        self.b_add.setMenu(menu)
        self.add_menu = menu
        row.addWidget(self.b_add)
        self.b_place = button('Place', 'ghost', 'place', 'Click in the plan or the 3D view to put the selected target '
                                                          'there (or drag it in the plan)')
        self.b_place.setCheckable(True)
        self.b_place.clicked.connect(self._toggle_place)
        row.addWidget(self.b_place)
        self.b_remove = button('Remove', 'ghost', 'trash', 'Remove the selected target')
        self.b_remove.clicked.connect(lambda: self.ctl.remove())
        row.addWidget(self.b_remove)
        row.addStretch(1)
        ll.addLayout(row)
        self.targets = _table(TARGET_COLS)
        self.targets.setToolTip('Tick to include a target; pick one to edit it and see its cameras')
        hh = self.targets.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        hh.resizeSection(0, 24)
        hh.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        for c in (2, 3):
            hh.setSectionResizeMode(c, QHeaderView.ResizeMode.ResizeToContents)
        self.targets.itemSelectionChanged.connect(self._row_selected)
        self.targets.itemChanged.connect(self._item_changed)
        ll.addWidget(self.targets)
        # ---- the selected target
        self.edit_box = QWidget()
        eg = QGridLayout(self.edit_box)
        eg.setContentsMargins(0, 4, 0, 0)
        eg.setVerticalSpacing(2)
        eg.setHorizontalSpacing(6)
        eg.addWidget(label('Name', ink2=True), 0, 0)
        self.name = QLineEdit()
        self.name.setFixedHeight(ROW_H)
        self.name.editingFinished.connect(self._name_edited)
        eg.addWidget(self.name, 0, 1, 1, 3)
        eg.addWidget(label('Kind', ink2=True), 1, 0)
        self.kind = QComboBox()
        for k in CR.KINDS:
            self.kind.addItem(CR.KIND_LABEL[k], k)
        self.kind.setToolTip('Changing the kind sets its typical size and needs')
        self.kind.activated.connect(self._kind_picked)
        eg.addWidget(self.kind, 1, 1, 1, 3)
        self.fields = {}
        self.field_labels = {}
        for key, text, unit, dec, step, scale, kinds, tip in FIELDS:
            f = ScrubSpinBox('', 'plain', dec, unit, step=step, width=92)
            f.setToolTip(tip)
            f.committed.connect(lambda v, k=key, s=scale, t=text, u=unit: self._commit(k, v / s, f'{t.lower()} '
                                                                                                f'{v:g} {u}'))
            a = label(text, ink2=True)
            a.setToolTip(tip)
            self.fields[key] = (f, scale, kinds)
            self.field_labels[key] = a
        self.edit_grid = eg
        self._laid_kind = None
        self.focus = QCheckBox('In focus')
        self.focus.setToolTip('The target must lie within the lens\'s focus range')
        self.focus.clicked.connect(lambda on: self.ctl.update(self.ctl.sel, 'in focus' if on else 'focus not needed',
                                                              focus=bool(on)))
        self.place_lbl = label('', muted=True, small=True, wrap=True)
        self._layout_fields('child')
        ll.addWidget(self.edit_box)
        # ---- tolerance
        ll.addSpacing(6)
        ll.addWidget(label('Install tolerance', bold=True))
        tg = QGridLayout()
        tg.setVerticalSpacing(2)
        tg.setHorizontalSpacing(6)
        self.tol_mm = ScrubSpinBox('', 'plain', 1, 'mm', step=1.0, width=96)
        self.tol_mm.setToolTip('Per axis; 0: the project\'s (Fleet ▸ tolerance, as the installation check uses)')
        self.tol_mm.committed.connect(lambda v: self.ctl.set_tolerance(install_mm=max(0.0, v)))
        self.tol_deg = ScrubSpinBox('', 'plain', 2, '°', step=0.1, width=96)
        self.tol_deg.setToolTip('Per axis; 0: the project\'s (Fleet ▸ tolerance, as the installation check uses)')
        self.tol_deg.committed.connect(lambda v: self.ctl.set_tolerance(install_deg=max(0.0, v)))
        tg.addWidget(label('Position', ink2=True), 0, 0)
        tg.addWidget(self.tol_mm, 0, 1)
        tg.addWidget(label('Angle', ink2=True), 0, 2)
        tg.addWidget(self.tol_deg, 0, 3)
        self.search_mm = ScrubSpinBox('', 'plain', 0, 'mm', step=10.0, width=96)
        self.search_mm.setToolTip('How far the envelope search moves the camera in x, y and z')
        self.search_mm.committed.connect(lambda v: self.ctl.set_tolerance(search_mm=v))
        self.search_deg = ScrubSpinBox('', 'plain', 1, '°', step=1.0, width=96)
        self.search_deg.setToolTip('How far the envelope search turns the camera in yaw, pitch and roll')
        self.search_deg.committed.connect(lambda v: self.ctl.set_tolerance(search_deg=v))
        tg.addWidget(label('Search', ink2=True), 1, 0)
        tg.addWidget(self.search_mm, 1, 1)
        tg.addWidget(label('Search', ink2=True), 1, 2)
        tg.addWidget(self.search_deg, 1, 3)
        ll.addLayout(tg)
        self.tol_lbl = label('', muted=True, small=True, wrap=True)
        ll.addWidget(self.tol_lbl)
        ll.addStretch(1)
        self.status = label('', muted=True, small=True, wrap=True)
        ll.addWidget(self.status)
        self.bar = QProgressBar()
        self.bar.setTextVisible(False)
        self.bar.setRange(0, 1000)
        self.bar.hide()
        ll.addWidget(self.bar)
        bl = QHBoxLayout()
        bl.addStretch(1)
        self.b_carla = button('In CARLA…', 'ghost', 'play', 'Stand these targets in the CARLA simulator and compare '
                                                           'what each camera sees there with this study (Simulate ▸ '
                                                           'Critical targets in CARLA)')
        self.b_carla.clicked.connect(lambda: self.ctx.window.set_page('critical_carla') if self.ctx.window else None)
        bl.addWidget(self.b_carla)
        self.b_export = button('Export…', None, 'export', 'Every target, what each camera sees of it and its '
                                                          'envelopes as JSON and CSV, the mechanical sheet '
                                                          '(mounting_envelopes.csv, .png) and the plan')
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
        sc.setMinimumWidth(372)
        return sc

    def _layout_fields(self, kind):
        """The fields a kind uses, two to a row in FIELDS order (the others hidden)."""
        if kind == self._laid_kind:
            return
        self._laid_kind = kind
        eg = self.edit_grid
        for key, (f, _, _) in self.fields.items():
            eg.removeWidget(f)
            eg.removeWidget(self.field_labels[key])
        eg.removeWidget(self.focus)
        eg.removeWidget(self.place_lbl)
        i = 0
        for key, (f, _, kinds) in self.fields.items():
            show = kind in kinds
            f.setVisible(show)
            self.field_labels[key].setVisible(show)
            if show:
                r, c = 2 + i // 2, (i % 2) * 2
                eg.addWidget(self.field_labels[key], r, c)
                eg.addWidget(f, r, c + 1)
                i += 1
        n = 2 + (i + 1) // 2
        eg.addWidget(self.focus, n, 0, 1, 2)
        eg.addWidget(self.place_lbl, n + 1, 0, 1, 4)

    # ------------------------------------------------------------------ edits
    def _commit(self, key, value, text):
        uid = self.ctl.sel
        if not uid or self._syncing:
            return
        if not self.ctl.update(uid, text, **{key: float(value)}):
            self._sync_fields()                  # clamped or unchanged: show what is stored

    def _name_edited(self):
        uid = self.ctl.sel
        t = self.ctl.target(uid)
        if t is None or self._syncing:
            return
        name = self.name.text().strip()
        if not name or name == t['name'] or not self.ctl.update(uid, f'renamed to {name}', name=name):
            self._sync_fields()

    def _kind_picked(self, i):
        uid = self.ctl.sel
        k = self.kind.itemData(i)
        if uid and k:
            self.ctl.update(uid, f'kind {CR.KIND_LABEL[k].lower()}', kind=k)

    def _toggle_place(self, on):
        self.ctl.placing = bool(on) and bool(self.ctl.sel)
        self.b_place.setChecked(self.ctl.placing)
        self.ctl.changed.emit()
        if self.ctl.placing:
            self.ctx.msg('Click in the plan or the 3D view where the selected target goes.', 4000)

    def _row_selected(self):
        if self._syncing:
            return
        rows = self.targets.selectionModel().selectedRows()
        st = self.ctl.settings
        if rows and rows[0].row() < len(st.targets):
            self.ctl.select(st.targets[rows[0].row()]['uid'])

    def _item_changed(self, it):
        if self._syncing or it.column() != 0:
            return
        st = self.ctl.settings
        if it.row() < len(st.targets):
            t = st.targets[it.row()]
            on = it.checkState() == Qt.CheckState.Checked
            if on != t['enabled']:
                self.ctl.update(t['uid'], 'included' if on else 'left out', enabled=on)

    def _on_selected(self, *_):
        self._sync_fields()
        self._fill_looks()
        self._fill_envelope()

    def _on_store(self, *_):
        if self.isVisible():
            self._sync_fields()
            self.plan.update()
        else:
            self._fields_dirty = True            # synced when the page shows again

    def _retheme(self, *_):
        self._shown = None
        self._sync_fields()
        self.refresh()

    def _sync_fields(self):
        st = self.ctl.settings
        self._syncing = True
        try:
            res = self.ctl.result if not self.ctl.stale else None
            t = self.targets
            t.setRowCount(len(st.targets))
            box = CR.vehicle_box(self.ctx.store.project)
            for r, tg in enumerate(st.targets):
                chk = t.item(r, 0)
                if chk is None:
                    chk = QTableWidgetItem()
                    chk.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled |
                                 Qt.ItemFlag.ItemIsSelectable)
                    t.setItem(r, 0, chk)
                chk.setCheckState(Qt.CheckState.Checked if tg['enabled'] else Qt.CheckState.Unchecked)
                t.setItem(r, 1, QTableWidgetItem(tg['name']))
                t.item(r, 1).setToolTip(f'{CR.KIND_LABEL[tg["kind"]]}, '
                                        f'{CR.describe_place(self.ctx.store.project, tg, box)}')
                t.setItem(r, 2, _num(f'{tg["min_px"]:g} px', 'Pixels across its ' +
                                     ('width' if tg['kind'] == 'lane' else 'height') +
                                     f', at least {100 * tg["min_share"]:.0f} % seen'))
                rr = res.target(tg['uid']) if res is not None else None
                if not tg['enabled']:
                    st_it = QTableWidgetItem('left out')
                    st_it.setForeground(THEME.color('muted'))
                elif rr is None:
                    st_it = QTableWidgetItem('—')
                    st_it.setForeground(THEME.color('muted'))
                else:
                    st_it = QTableWidgetItem(', '.join(rr.met_by) if rr.met else 'unresolved')
                    st_it.setForeground(QColor(STATUS['good' if rr.met else 'critical']))
                    b = rr.best()
                    st_it.setToolTip(f'Resolved by {", ".join(rr.met_by)}' if rr.met else
                                     f'Closest: {b.tag}, {b.px:.0f} px, {100 * b.seen:.0f} % seen' if b else '')
                t.setItem(r, 3, st_it)
            t.setFixedHeight(ROW_H * (min(max(len(st.targets), 2), 8) + 1) + 6)
            row = next((i for i, x in enumerate(st.targets) if x['uid'] == self.ctl.sel), -1)
            if row >= 0:
                t.selectRow(row)
            else:
                t.clearSelection()
            cur = st.target(self.ctl.sel)
            self.edit_box.setEnabled(cur is not None)
            self.b_remove.setEnabled(cur is not None)
            self.b_place.setEnabled(cur is not None)
            self.b_place.setChecked(self.ctl.placing)
            if cur is not None:
                if self.name.text() != cur['name'] and not self.name.hasFocus():
                    self.name.setText(cur['name'])
                self.kind.setCurrentIndex(CR.KINDS.index(cur['kind']))
                self._layout_fields(cur['kind'])
                for key, (f, scale, kinds) in self.fields.items():
                    f.set_value(cur[key] * scale)
                self.field_labels['min_px'].setText('Needs across' if cur['kind'] == 'lane' else 'Needs')
                self.focus.setChecked(cur['focus'])
                self.place_lbl.setText(CR.describe_place(self.ctx.store.project, cur, box).capitalize() + '.')
            tol = CR.install_tolerance(self.ctx.store.project, st)
            self.tol_mm.set_value(st.install_mm)
            self.tol_deg.set_value(st.install_deg)
            self.search_mm.set_value(st.search_mm)
            self.search_deg.set_value(st.search_deg)
            self.tol_lbl.setText(f'In use: ±{tol["mm"]:g} mm, ±{tol["deg"]:g}° per axis' +
                                 (' (the project\'s, from the fleet settings: 0 keeps it)'
                                  if tol['source'] == 'project (fleet)' else '') + '.')
        finally:
            self._syncing = False

    # ------------------------------------------------------------------ views
    def _set_view(self, key):
        if key == '3d' and self.view3d is None:
            from ..viewport3d import make_viewport
            self.scene3d = CriticalScene(self.ctx, self.ctl)
            self.view3d = make_viewport(self.scene3d)
            self.view3d.cam.az, self.view3d.cam.el = 200.0, 28.0
            self.stack.addWidget(self.view3d)
        self.stack.setCurrentWidget(self.view3d if key == '3d' else self.plan)
        self.view_mode.set_value(key)
        self._fit()

    def _fit(self):
        if self.stack.currentWidget() is self.plan:
            self.plan.fit_targets()
            return
        targets, _ = _targets(self.ctx)
        b = CR.vehicle_box(self.ctx.store.project)
        pts = [(b['front_x'], b['half_w'], 0.0), (b['rear_x'], -b['half_w'], 1.6)]
        for t in targets:
            pts += [(x, y, t['base'] + t['height']) for x, y in _footprint(t)]
        P = np.array(pts, float)
        self.view3d.frame_box(P.min(0), P.max(0), 1.1)

    # ------------------------------------------------------------------ state
    def showEvent(self, e):
        super().showEvent(e)
        if getattr(self, '_fields_dirty', False):
            self._fields_dirty = False
            self._sync_fields()
        self.ctl.set_active(True)

    def hideEvent(self, e):
        super().hideEvent(e)
        self.ctl.set_active(False)
        if self.ctl.placing:
            self.ctl.placing = False

    def activate(self):
        self.ctl.set_active(True)
        if not self.plan._fitted or not getattr(self, '_framed', False):
            self._framed = True
            QTimer.singleShot(0, self._fit)

    def refresh(self):
        ctl = self.ctl
        res = ctl.result
        win = self.ctx.window
        has = any(t['enabled'] for t in ctl.settings.targets)
        self.b_run.setEnabled(has and not (ctl.busy and win is not None and JOB in win._jobs))
        self.bar.setVisible(ctl.busy)
        self.b_export.setEnabled(res is not None)
        status = EMPTY if not ctl.settings.targets else ctl.status
        self.status.setText(status + (' · stale: the design or the targets changed' if ctl.stale and res else ''))
        self.plan.update()
        if self.view3d is not None:
            self.view3d.update()
        self._sync_fields()
        if res is self._shown:
            return
        self._shown = res
        self._fill_cams()
        self._fill_looks()
        self._fill_envelope()
        self._fill_findings()

    def _fill_cams(self):
        res = self.ctl.result
        cams = [c for c in res.cameras] if res is not None else []
        self.cam_pick.clear()
        for c in cams:
            self.cam_pick.addItem(f'{c.tag}' + (f' · {len(c.targets)} target{"s" * (len(c.targets) != 1)}'
                                                if c.targets else ' · none'), c.uid)
        if cams and not any(c.uid == self.sel_cam for c in cams):
            first = next((c for c in cams if c.targets), cams[0])
            self.sel_cam = first.uid
        i = next((k for k, c in enumerate(cams) if c.uid == self.sel_cam), -1)
        if i >= 0:
            self.cam_pick.setCurrentIndex(i)

    def _fill_looks(self):
        res = self.ctl.result
        r = res.target(self.ctl.sel) if res is not None and self.ctl.sel else None
        t = self.looks
        if r is None:
            t.setRowCount(0)
            self.look_head.setText('Cameras' + (' · run the study' if self.ctl.sel else ''))
            self.sensors_lbl.setText('')
            return
        self.look_head.setText(f'Cameras on {r.name}')
        t.setRowCount(len(r.looks))
        need = r.target['min_px']
        for i, lk in enumerate(r.looks):
            it = QTableWidgetItem(lk.tag)
            it.setForeground(THEME.cam(lk.slot))
            it.setData(Qt.ItemDataRole.UserRole, lk.uid)
            hid = ', '.join(f'{k} {100 * v:.0f} %' for k, v in lk.hidden.items())
            it.setToolTip(f'Hidden by {hid}' if hid else 'Nothing of the vehicle hides it')
            t.setItem(i, 0, it)
            t.setItem(i, 1, _num(f'{100 * lk.in_view:.0f}'))
            t.setItem(i, 2, _num(f'{100 * lk.seen:.0f}', f'{100 * lk.unoccluded:.0f} % of the part in view is '
                                                          f'unoccluded' + (f'; hidden by {hid}' if hid else '')))
            t.setItem(i, 3, _num(f'{lk.px:.0f}', f'{lk.px:.1f} px of the {need:g} needed' +
                                 (f'; {lk.px_width:.0f} px across its width, ≈{lk.area:,.0f} px on it'
                                  if r.target['kind'] not in CR.LINEAR else
                                  f'; {100 * lk.resolved:.0f} % of its length resolved'),
                                 QColor(STATUS['good']) if lk.px >= need else THEME.color('ink2')))
            t.setItem(i, 4, _num(f'{lk.dist:.1f}'))
            t.setItem(i, 5, QTableWidgetItem('yes' if lk.focus else 'no'))
            m = QTableWidgetItem('meets' if lk.met else {'seen': 'too little seen', 'pixels': 'too few px',
                                                        'focus': 'out of focus'}.get(lk.fail, 'no'))
            m.setForeground(QColor(STATUS['good']) if lk.met else THEME.color('muted'))
            t.setItem(i, 6, m)
            if lk.uid == self.sel_cam:
                t.selectRow(i)
        bits = []
        if r.lidar is not None:
            per = ', '.join(f'{k} {v:.0f}' for k, v in r.lidar['per'].items())
            bits.append(f'Lidar points per frame: {r.lidar["points"]:.1f}' + (f' ({per})' if per else '') +
                        (f', {r.lidar["needed"]:g} needed' if r.lidar.get('relied_on') else ', not required'))
        if r.radar is not None:
            bits.append(f'Radars detecting it (RCS {r.radar["rcs"]:g} m²): ' + (', '.join(r.radar['by']) or 'none'))
        self.sensors_lbl.setText(' · '.join(bits))

    def _look_clicked(self, row, col):
        it = self.looks.item(row, 0)
        if it is not None:
            self.sel_cam = it.data(Qt.ItemDataRole.UserRole)
            i = self.cam_pick.findData(self.sel_cam)
            if i >= 0:
                self.cam_pick.setCurrentIndex(i)
            self._fill_envelope()

    def _cam_picked(self, i):
        self.sel_cam = self.cam_pick.itemData(i) or ''
        self._fill_envelope()

    def _set_scope(self, key):
        self.scope = key
        self._fill_envelope()

    def envelope_rows(self):
        """(rows, title, note) the chart shows: the selected camera's envelope for the selected target, or its
        tightest over all its targets."""
        res = self.ctl.result
        if res is None or not self.sel_cam:
            return [], 'Mounting envelope', ''
        c = res.camera(self.sel_cam)
        if c is None:
            return [], 'Mounting envelope', ''
        if self.scope == 'all':
            if not c.targets or not c.lo:
                return [], f'{c.tag} · resolves no target', ''
            rows = [(ax, c.lo[ax], c.hi[ax]) for ax in CR.AXES if ax in c.lo and ax in c.hi]
            note = (f'binding: {c.binding}' if c.binding else 'no target lost within the search') + \
                   (f' · joint {100 * c.joint:.0f} %' if c.joint is not None else '')
            return rows, f'{c.tag} · every target it resolves', note
        r = res.target(self.ctl.sel) if self.ctl.sel else None
        lk = next((x for x in r.looks if x.uid == c.uid), None) if r is not None else None
        if lk is None or lk.envelope is None:
            what = r.name if r is not None else 'the selected target'
            return [], f'{c.tag} · does not resolve {what}', ''
        env = lk.envelope
        rows = [(ax, (env.lo[ax], r.name, env.lo_why[ax]), (env.hi[ax], r.name, env.hi_why[ax])) for ax in CR.AXES]
        note = (f'joint {100 * env.joint:.0f} %' if env.joint is not None else '') + \
               ('' if env.tol_ok is None else (' · holds at the tolerance' if env.tol_ok else
                                               ' · fails at the tolerance'))
        return rows, f'{c.tag} · {r.name}', note.strip(' ·')

    def _fill_envelope(self):
        rows, title, note = self.envelope_rows()
        res = self.ctl.result
        tol = res.install if res is not None else CR.install_tolerance(self.ctx.store.project)
        self.bars.set_data(rows, tol, title, note)
        if res is not None and self.sel_cam:
            c = res.camera(self.sel_cam)
            if c is not None and c.tight:
                n = len({(a, s) for a, s, *_ in c.tight})
                self.env_note.setText(f'{c.tag}: {n} limit{"s" * (n != 1)} tighter than the install tolerance' +
                                      (f'; only {c.tag} resolves {", ".join(c.sole)}' if c.sole else '') + '.')
            elif c is not None and c.targets:
                self.env_note.setText(f'{c.tag} holds every target it resolves within the install tolerance, each '
                                      f'axis alone' + ('' if c.tol_ok is None else (
                                          ' and all at once.' if c.tol_ok else ', but not all axes at once.')))
            else:
                self.env_note.setText('')
        else:
            self.env_note.setText('')

    def _fill_findings(self):
        for w in self.findings:
            self.find_lay.removeWidget(w)
            w.hide()                    # until the deferred delete runs it would still paint over the new rows
            w.deleteLater()
        self.findings = []
        res = self.ctl.result
        if res is None:
            return
        for f in res.findings:
            w = StatusLine(f['severity'], f['message'])
            self.find_lay.insertWidget(len(self.findings), w)         # above the stretch and the notes
            self.findings.append(w)

    # ------------------------------------------------------------------ export
    def export(self, folder=None):
        res = self.ctl.result
        if res is None:
            return []
        if folder is None:
            folder = QFileDialog.getExistingDirectory(self, 'Export the critical targets into')
            if not folder:
                return []
        from ...core import provenance as PV          # where the JSON came from
        paths = PV.stamp_paths(CR.export(res, self.ctx.store.project, folder), self.ctx.store.project, folder)
        self.ctx.msg(f'Exported the critical targets and mounting envelopes to {folder}', 5000)
        return paths
