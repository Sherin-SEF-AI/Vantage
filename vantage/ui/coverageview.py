"""Lidar and radar coverage in the UI: the controller that keeps a result for the project, the plan of the coverage
around the vehicle (camera, lidar and radar range envelopes per object, the lidars' ground rings or point density, the
radar zones, the required range per sector) and the verdict grid. Drawn forward up, left to the left, on a square-root
range scale like the perception range (near ranges stay readable next to 200 m)."""
from __future__ import annotations

import json
import math
import threading
import traceback

import numpy as np
from PyQt6.QtCore import QObject, QPointF, QRectF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QImage, QPainter, QPainterPath, QPen, QPolygonF
from PyQt6.QtWidgets import QToolTip

from ..core import perception as PR
from ..core import sensor_coverage as SC
from ..core import sensors as S
from .charts import ChartBase
from .theme import THEME, STATUS, mono, sans

AUTO_MAX_S = 3.0
JOB = 'sensor_coverage'
EMPTY = 'Add a lidar or radar: Vehicle ▸ Add lidar / Add radar'
RINGS = PR.PNG_RINGS
VERDICT_TONE = {'redundant': 'good', 'single': 'warning', 'gap': 'critical'}
SHORT = {'front': 'Front', 'front-left': 'FL', 'left': 'Left', 'rear-left': 'RL', 'rear': 'Rear', 'rear-right': 'RR',
         'right': 'Right', 'front-right': 'FR'}


def modality_color(mod, alpha=None) -> QColor:
    c = QColor(SC.MODALITY_COLORS[mod])
    if alpha is not None:
        c.setAlphaF(alpha)
    return c


def verdict_color(v) -> QColor:
    return QColor(STATUS[VERDICT_TONE.get(v, 'critical')])


class _Sig(QObject):
    done = pyqtSignal(int, object)
    failed = pyqtSignal(int, str)
    progress = pyqtSignal(int, float)


def controller(ctx) -> 'CoverageController':
    c = getattr(ctx, 'sensor_coverage', None)
    if c is None:
        c = ctx.sensor_coverage = CoverageController(ctx)
    return c


def has_sensors(project) -> bool:
    return any(s.visible for s in project.variant.sensors)


def inputs_key(project) -> str:
    """Everything the study reads, as text: a result made from other inputs is stale."""
    m = project.vehicle.mesh
    v = project.variant
    a = project.analysis
    return json.dumps({'mesh': [id(m), getattr(m, 'n_tris', 0), project.vehicle.occluder_target],
                       'variant': v.to_dict(), 'rig': [r.to_dict() for r in project.rig],
                       'coverage': a.sensor_coverage, 'perception': a.perception, 'weather': a.weather,
                       'housings': a.housings, 'materials': a.materials,     # the radars' fascia, the condition
                       'parts': [(q.name, q.role, q.material, getattr(q, 'surface', ''), getattr(q, 'surface_mm', 0.0))
                                 for q in (m.parts if m is not None else [])],
                       'library': project.section('library')}, sort_keys=True, default=str)


class CoverageController(QObject):
    """The project's lidar and radar coverage: run on demand (the page's Run button, a main-window job with Cancel)
    and again by itself after edits while the page shows it and a run is quick."""
    changed = pyqtSignal()
    progress = pyqtSignal(float)

    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        self.result: SC.CoverageResult | None = None
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

    # ---- settings (saved with the project; edits are undoable)
    @property
    def settings(self) -> SC.CoverageSettings:
        return SC.CoverageSettings.from_dict(self.ctx.store.project.analysis.sensor_coverage)

    def _store(self, st: SC.CoverageSettings, text):
        new = SC.CoverageSettings.from_dict(st.to_dict()).to_dict()
        if new == self.settings.to_dict():
            return False

        def fn(p):
            p.analysis.sensor_coverage = new
        self.ctx.store.edit('analysis', fn, text)
        return True

    def set_classes(self, classes, text='Lidar and radar coverage · object classes'):
        clean = [SC.clean_class(c) for c in classes]
        if not clean or any(c is None for c in clean):
            return False                   # not a usable value (a minimum not above zero, a negative RCS): keep it
        st = self.settings
        st.classes = clean
        return self._store(st, text)

    def classes_from_perception(self):
        """The perception range's classes (names and sizes), keeping min_points and rcs of the classes of the same
        name; typical figures for new ones."""
        mine = {c['name']: c for c in self.settings.classes}
        dflt = {c['name']: c for c in SC.DEFAULT_CLASSES}
        out = []
        for c in PR.PerceptionSettings.from_dict(self.ctx.store.project.analysis.perception).classes:
            had = mine.get(c['name']) or dflt.get(c['name']) or {'min_points': 10.0, 'rcs': 1.0}
            out.append({'name': c['name'], 'height': c['height'], 'width': c['width'], 'base': c.get('base', 0.0),
                        'min_points': had['min_points'], 'rcs': had['rcs']})
        return self.set_classes(out, 'Lidar and radar coverage · objects from the perception range')

    # ---- state
    def _quick(self):
        return self.result is None or self.result.metrics.get('seconds', 99) < AUTO_MAX_S

    def _mark_stale(self):
        if self.result is not None and not self.stale:
            self.stale = True
            self.changed.emit()
        if self.active and self._quick() and not self.busy:
            self.timer.start()

    def _on_edit(self, *_):
        if not (self.active or self.result is not None):
            return                         # never shown: nothing to keep up to date
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
        if not has_sensors(p):
            if not (win is not None and JOB in win._jobs):
                self._gen += 1             # a run still going is for sensors that are gone
                self.busy = False
            self.result, self._for, self.stale, self.status = None, None, False, EMPTY
            self.changed.emit()
            return False
        if self.busy:
            if quiet or (win is not None and JOB in win._jobs):
                return False
            self._gen += 1
        project = p.snapshot()
        st = SC.CoverageSettings.from_dict(project.analysis.sensor_coverage)
        eng = win._shared_engine() if win is not None else ctx.analysis.engine.share()
        self._gen += 1
        gen = self._gen
        self.busy = True
        self.status = 'Computing the lidar and radar coverage…'
        key = (p, inputs_key(p))

        def work(progress, cancel):
            return key, SC.run(project, eng, None, st, progress=progress, cancel=cancel)
        if not quiet and win is not None and hasattr(win, 'run_job'):
            ok = win.run_job(JOB, 'Lidar and radar coverage', work, lambda out: self._done(gen, out),
                             lambda m: self._failed(gen, m))
            self.changed.emit()                # after the job is registered: the page disables Run while it runs
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
        self.status = (f'{m["redundant"]} redundant · {m["single"]} single · {m["gap"]} gaps of {m["checks"]} sector '
                       f'checks · {m["lidars"]} lidar{"s" * (m["lidars"] != 1)}, {m["radars"]} '
                       f'radar{"s" * (m["radars"] != 1)} · {m["seconds"]:.1f} s')
        self.changed.emit()
        if self.stale and self.active and self._quick():
            self.timer.start()

    def _failed(self, gen, msg):
        if gen != self._gen:
            return
        self.busy = False
        self.status = 'Failed: ' + msg.splitlines()[0]
        self.changed.emit()


# =========================================================================== drawing helpers
def _pt(c: QPointF, k, r, az_deg) -> QPointF:
    """Forward up, left to the left, radius ∝ √range."""
    s = math.sqrt(max(r, 0.0)) * k
    a = math.radians(az_deg)
    return QPointF(c.x() - math.sin(a) * s, c.y() - math.cos(a) * s)


def _world(c, k, res, x, y) -> QPointF:
    dx, dy = x - res.centre[0], y - res.centre[1]
    return _pt(c, k, math.hypot(dx, dy), math.degrees(math.atan2(dy, dx)))


def _polyline_runs(c, k, ranges, az, closed=True) -> list:
    """Polylines through the finite (range, azimuth) samples; a gap breaks the line."""
    ok = np.isfinite(ranges)
    if ok.all():
        pts = [_pt(c, k, r, a) for r, a in zip(ranges, az)]
        return [QPolygonF(pts + pts[:1] if closed else pts)]
    out, cur = [], []
    for r, a, good in zip(ranges, az, ok):
        if good:
            cur.append(_pt(c, k, r, a))
        elif cur:
            out.append(QPolygonF(cur))
            cur = []
    if cur:
        out.append(QPolygonF(cur))
    return [q for q in out if q.size() > 1]


class CoverageMap(ChartBase):
    """The plan: for one object class the farthest range the cameras, the lidars and the radars detect it per
    direction, the required range per sector coloured by its verdict, the radar zones for the class's cross-section,
    the lidars' ground density or rings, the body, the lidars and radars."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.res: SC.CoverageResult | None = None
        self.k = 0
        self.layer = 'density'
        self.reach = 0.0                   # range drawn at the edge, m (0: the study's)
        self.empty = 'Run the lidar and radar coverage'
        self._heat = None                  # (key, QImage)
        self.setMinimumSize(320, 320)

    def set_result(self, res):
        self.res = res
        self.k = min(self.k, len(res.classes) - 1) if res is not None and res.classes else 0
        self.update()

    def set_class(self, k):
        self.k = int(k)
        self.update()

    def set_layer(self, layer):
        self.layer = layer
        self.update()

    def set_reach(self, metres):
        self.reach = float(metres)
        self.update()

    def set_empty(self, text):
        if text != self.empty:
            self.empty = text
            self.update()

    def rmax(self):
        full = float(self.res.ranges[-1])
        return min(self.reach, full) if self.reach > 0 else full

    def _geom(self):
        r = QRectF(self.rect()).adjusted(16, 30, -16, -22)
        c = r.center()
        rad = max(min(r.width(), r.height()) / 2 - 6, 10.0)
        return c, rad, rad / math.sqrt(max(self.rmax(), 1.0))

    def _heat_image(self, rad):
        n = int(min(2 * rad, 720))
        key = (id(self.res), n, self.rmax())
        if self._heat is None or self._heat[0] != key:
            rgba = np.ascontiguousarray(SC.heat_rgba(self.res, n, self.rmax()))
            img = QImage(rgba.data, n, n, 4 * n, QImage.Format.Format_RGBA8888).copy()
            self._heat = (key, img)
        return self._heat[1]

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), THEME.color('viewport'))
        res = self.res
        if res is None:
            p.setPen(THEME.color('muted'))
            p.setFont(sans(12))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self.empty)
            return
        c, rad, k = self._geom()
        rmax = self.rmax()
        cl = res.classes[self.k]
        clip = QPainterPath()
        clip.addEllipse(c, rad + 2, rad + 2)
        # ground density under everything
        if self.layer == 'density' and res.lidars:
            p.save()
            p.setClipPath(clip)
            p.setOpacity(0.85)
            p.drawImage(QRectF(c.x() - rad, c.y() - rad, 2 * rad, 2 * rad), self._heat_image(rad))
            p.restore()
        # range rings and sector spokes
        p.setFont(mono(10))
        for ring in RINGS:
            if ring > rmax * 1.001:
                break
            rr = math.sqrt(ring) * k
            p.setPen(QPen(THEME.color('grid2'), 1))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawEllipse(c, rr, rr)
            p.setPen(THEME.color('muted'))
            p.drawText(QPointF(c.x() + 4, c.y() - rr + 13), f'{ring} m')
        for s in res.sectors:
            p.setPen(QPen(THEME.color('grid2'), 1, Qt.PenStyle.DotLine))
            p.drawLine(c, _pt(c, k, rmax, s['centre_deg'] - 22.5))
        p.save()
        p.setClipPath(clip)
        if self.layer == 'rings':
            p.setPen(QPen(modality_color('lidar', 0.42), 0.9))
            p.setBrush(Qt.BrushStyle.NoBrush)
            for li in res.lidars:
                for _, P in li.rings:
                    p.drawPolyline(QPolygonF([_world(c, k, res, x, y) for x, y in P]))
        # radar zones for the class's cross-section
        if cl.rcs > 0:
            p.setPen(QPen(modality_color('radar', 0.6), 1.0, Qt.PenStyle.DashLine))
            p.setBrush(modality_color('radar', 0.08))
            for r_ in res.radars:
                for w in S.radar_wedges(r_.sensor, r_.spec, cl.rcs, samples=48, zone_scale=r_.zone_scale):
                    p.drawPolygon(QPolygonF([_world(c, k, res, x, y) for x, y in w]))
        # required ranges: green two or more sensor types, amber one, red none
        p.setBrush(Qt.BrushStyle.NoBrush)
        for s in res.sectors:
            a0 = s['centre_deg'] - 22.5
            pts = [_pt(c, k, s['required_m'], a0 + t * 45.0 / 12) for t in range(13)]
            p.setPen(QPen(verdict_color(s['classes'][cl.name]['verdict']), 1.8, Qt.PenStyle.DashLine))
            p.drawPolyline(QPolygonF(pts))
        # the nearest ground any lidar reaches
        if res.lidars:
            p.setPen(QPen(modality_color('lidar', 0.9), 1.1, Qt.PenStyle.DotLine))
            for q in _polyline_runs(c, k, res.near, res.az):
                p.drawPolyline(q)
        for mod, arr in (('radar', res.radar), ('lidar', res.lidar), ('camera', res.camera)):
            v = arr[self.k]
            if not np.isfinite(v).all():
                continue
            fill = modality_color(mod, 0.07)
            p.setBrush(fill)
            p.setPen(QPen(modality_color(mod), 2.0))
            p.drawPolygon(QPolygonF([_pt(c, k, r, a) for r, a in zip(v, res.az)]))
        p.restore()
        # body, lidars, radars
        if res.body is not None:
            p.setPen(QPen(THEME.color('strong'), 1))
            p.setBrush(THEME.color('body'))
            p.drawPolygon(QPolygonF([_pt(c, k, r, a) for r, a in zip(res.outline, res.az)]))
        p.setFont(sans(10, QFont.Weight.DemiBold))
        tags = []
        for li in res.lidars:
            q = _world(c, k, res, li.sensor.x, li.sensor.y)
            p.setPen(QPen(THEME.color('viewport'), 1.2))
            p.setBrush(modality_color('lidar'))
            p.drawEllipse(q, 4.5, 4.5)
            tags.append((QPointF(q.x() + 7, q.y() - 12), li.tag))
        for r_ in res.radars:
            q = _world(c, k, res, r_.sensor.x, r_.sensor.y)
            p.setPen(QPen(THEME.color('viewport'), 1.2))
            p.setBrush(modality_color('radar'))
            p.drawRect(QRectF(q.x() - 4, q.y() - 4, 8, 8))
            tags.append((QPointF(q.x() + 6, q.y() + 3), r_.tag))
        fm = p.fontMetrics()
        for at, tag in tags:                 # on a chip: readable over the body and the density alike
            box = QRectF(at.x(), at.y(), fm.horizontalAdvance(tag) + 6, fm.height())
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(THEME.color('raised', 0.85))
            p.drawRoundedRect(box, 2, 2)
            p.setPen(THEME.color('ink'))
            p.drawText(box, Qt.AlignmentFlag.AlignCenter, tag)
        p.setFont(sans(10, QFont.Weight.DemiBold))
        p.setPen(THEME.color('ink2'))
        for s in res.sectors:
            q = _pt(c, k, rmax * 0.995, s['centre_deg'])
            name = s['name'].replace('-', ' ').capitalize()
            w = p.fontMetrics().horizontalAdvance(name)
            p.drawText(QPointF(min(max(q.x() - w / 2, 4), self.width() - w - 4),
                               min(max(q.y() + 4, 16), self.height() - 24)), name)
        # legend
        p.setFont(sans(11))
        x, y = 12.0, 18.0
        p.setPen(THEME.color('ink'))
        p.drawText(QPointF(x, y), cl.name)
        x += p.fontMetrics().horizontalAdvance(cl.name) + 14
        for mod, arr in (('camera', res.camera), ('lidar', res.lidar), ('radar', res.radar)):
            have = np.isfinite(arr[self.k]).all()
            text = mod.capitalize() if have else f'{mod.capitalize()}: ' + (
                'not relied on' if mod == 'radar' and cl.rcs <= 0 and res.radars else 'none')
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(modality_color(mod, 1.0 if have else 0.35))
            p.drawRect(QRectF(x, y - 9, 9, 9))
            p.setPen(THEME.color('ink2' if have else 'muted'))
            p.drawText(QPointF(x + 13, y), text)
            x += p.fontMetrics().horizontalAdvance(text) + 28
        p.setFont(sans(10))
        p.setPen(THEME.color('muted'))
        p.drawText(QRectF(0, self.height() - 18, self.width() - 10, 16), Qt.AlignmentFlag.AlignRight,
                   'dashed: required range (green 2+ sensor types, amber 1, red none) · dotted: nearest lidar ground '
                   '· √ range scale')
        if self.layer == 'density' and res.lidars:          # the density scale, top right
            fm = p.fontMetrics()
            right = f'{SC.DENS_HI:g} points/m² per frame'
            x1 = self.width() - 12 - fm.horizontalAdvance(right)
            p.drawText(QPointF(x1, 18), right)
            x0 = x1 - 6 - len(SC.SEQ) * 7
            for i, col in enumerate(SC.SEQ):
                p.fillRect(QRectF(x0 + i * 7, 10, 7, 8), QColor(col))
            p.drawText(QPointF(x0 - 6 - fm.horizontalAdvance(f'{SC.DENS_LO:g}'), 18), f'{SC.DENS_LO:g}')

    def mouseMoveEvent(self, e):
        res = self.res
        if res is None:
            return
        c, rad, k = self._geom()
        dx, dy = e.position().x() - c.x(), e.position().y() - c.y()
        s = math.hypot(dx, dy)
        if s < 2:
            return
        r = (s / k) ** 2
        az = math.degrees(math.atan2(-dx, -dy)) % 360.0
        cl = res.classes[self.k]
        at = res.at(self.k, az)
        lines = [f'<b>{az:.0f}°</b> · {r:.1f} m from the centre', f'<b>{cl.name}</b>']
        for mod in SC.MODALITIES:
            v = at[mod]
            lines.append(f'{mod.capitalize()}: ' + ('—' if v is None else f'to {v:.0f} m'))
        if res.lidars:
            lines.append(f'Lidar points on it here: {res.points_at(self.k, az, r):.1f} per frame '
                         f'(needs {cl.min_points:g})')
            b = res.blind[res.az_index(az)]
            lines.append('Nearest lidar ground: ' + (f'{b:.1f} m from the body' if math.isfinite(b) else 'none'))
            x = res.centre[0] + r * math.cos(math.radians(az))
            y = res.centre[1] + r * math.sin(math.radians(az))
            x0, y0, cell = res.grid
            i, j = int((x - x0) // cell), int((y - y0) // cell)
            if 0 <= j < res.density.shape[0] and 0 <= i < res.density.shape[1]:
                lines.append(f'Ground density: {res.density[j, i]:.0f} points/m² per frame')
        QToolTip.showText(e.globalPosition().toPoint(), '<br>'.join(lines), self)

    def leaveEvent(self, e):
        QToolTip.hideText()


class VerdictGrid(ChartBase):
    """Object classes × sectors: per cell the number of sensor types that reach the required range, coloured by the
    verdict (green redundant, amber single, red gap). A click picks the class."""
    picked = pyqtSignal(int)

    ROW = 22
    HEAD = 20
    NAME_W = 92

    def __init__(self, parent=None):
        super().__init__(parent)
        self.res = None
        self.k = 0
        self.setMinimumHeight(self.HEAD + self.ROW * 4 + 4)

    def set_result(self, res):
        self.res = res
        n = len(res.classes) if res is not None else 4
        self.setFixedHeight(self.HEAD + self.ROW * n + 4)
        self.update()

    def set_class(self, k):
        self.k = int(k)
        self.update()

    def _cell(self, pos):
        res = self.res
        if res is None:
            return None
        cw = (self.width() - self.NAME_W) / max(len(res.sectors), 1)
        r = int((pos.y() - self.HEAD) // self.ROW)
        s = int((pos.x() - self.NAME_W) // cw) if pos.x() >= self.NAME_W else -1
        if 0 <= r < len(res.classes):
            return r, s
        return None

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), THEME.color('panel'))
        res = self.res
        if res is None:
            return
        cw = (self.width() - self.NAME_W) / max(len(res.sectors), 1)
        p.setFont(sans(10))
        p.setPen(THEME.color('muted'))
        for j, s in enumerate(res.sectors):
            p.drawText(QRectF(self.NAME_W + j * cw, 0, cw, self.HEAD), Qt.AlignmentFlag.AlignCenter,
                       f'{SHORT.get(s["name"], s["name"])} {s["required_m"]:g}')
        for i, cl in enumerate(res.classes):
            y = self.HEAD + i * self.ROW
            if i == self.k:
                p.fillRect(QRectF(0, y, self.width(), self.ROW), THEME.color('sel'))
            p.setFont(sans(11, QFont.Weight.DemiBold if i == self.k else None))
            p.setPen(THEME.color('ink' if i == self.k else 'ink2'))
            p.drawText(QRectF(6, y, self.NAME_W - 8, self.ROW), Qt.AlignmentFlag.AlignVCenter, cl.name)
            p.setFont(mono(11))
            for j, s in enumerate(res.sectors):
                d = s['classes'][cl.name]
                col = verdict_color(d['verdict'])
                box = QRectF(self.NAME_W + j * cw + 2, y + 3, cw - 4, self.ROW - 6)
                fill = QColor(col)
                fill.setAlphaF(0.22)
                p.setPen(QPen(col, 1))
                p.setBrush(fill)
                p.drawRoundedRect(box, 2, 2)
                p.setPen(THEME.color('ink'))
                p.drawText(box, Qt.AlignmentFlag.AlignCenter, str(d['modalities']))

    def mousePressEvent(self, e):
        hit = self._cell(e.position())
        if hit is not None:
            self.picked.emit(hit[0])

    def mouseMoveEvent(self, e):
        hit = self._cell(e.position())
        res = self.res
        if hit is None or hit[1] < 0 or res is None or hit[1] >= len(res.sectors):
            QToolTip.hideText()
            return
        cl, s = res.classes[hit[0]], res.sectors[hit[1]]
        d = s['classes'][cl.name]
        fmt = lambda v: '—' if v is None else f'{v:.0f} m'
        lines = [f'<b>{cl.name}, {s["name"]}</b> · required {s["required_m"]:g} m']
        for mod in SC.MODALITIES:
            v = d[f'{mod}_m']
            lines.append(f'{mod.capitalize()}: {fmt(v)}' + (' ✓' if d['meets'][mod] else ''))
        lines.append(f'{d["verdict"].capitalize()}: {d["modalities"]} of 3 sensor types reach it in every direction')
        QToolTip.showText(e.globalPosition().toPoint(), '<br>'.join(lines), self)

    def leaveEvent(self, e):
        QToolTip.hideText()
