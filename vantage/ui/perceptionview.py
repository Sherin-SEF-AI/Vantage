"""Perception range in the UI: the controller that keeps a result for the project, the plan of the detection
envelopes around the vehicle and the per-class roses. Drawn forward up, left to the left, on a square-root range scale
(near ranges stay readable next to 200 m)."""
from __future__ import annotations

import math
import threading
import traceback

from PyQt6.QtCore import QObject, QPointF, QRectF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QPainter, QPen, QPolygonF
from PyQt6.QtWidgets import QToolTip

from ..core import perception as PR
from .charts import ChartBase
from .theme import THEME, STATUS, mono, sans

AUTO_MAX_S = 2.0
CLASS_COLORS = PR.CLASS_COLORS
RINGS = PR.PNG_RINGS


def class_color(k) -> QColor:
    return QColor(CLASS_COLORS[k % len(CLASS_COLORS)])


class _Sig(QObject):
    done = pyqtSignal(int, object)
    failed = pyqtSignal(int, str)
    progress = pyqtSignal(int, float)


def controller(ctx) -> 'PerceptionController':
    c = getattr(ctx, 'perception', None)
    if c is None:
        c = ctx.perception = PerceptionController(ctx)
    return c


class PerceptionController(QObject):
    """The project's perception-range result: run on demand (the page's Run button, a main-window job with Cancel)
    and again by itself after edits while the page shows it and a run is quick."""
    changed = pyqtSignal()
    progress = pyqtSignal(float)

    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        self.result: PR.PerceptionResult | None = None
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
        ctx.analysis.finished.connect(self._on_analysis)
        ctx.store.changed.connect(self._on_edit)
        ctx.store.project_replaced.connect(self._reset)

    # ---- settings (saved with the project; edits are undoable)
    @property
    def settings(self) -> PR.PerceptionSettings:
        return PR.PerceptionSettings.from_dict(self.ctx.store.project.analysis.perception)

    def _store(self, st: PR.PerceptionSettings, text):
        new = PR.PerceptionSettings.from_dict(st.to_dict()).to_dict()
        if new == self.settings.to_dict():
            return False

        def fn(p):
            p.analysis.perception = new
        self.ctx.store.edit('analysis', fn, text)
        return True

    def set_classes(self, classes, text='Perception range · object classes'):
        clean = [PR.clean_class(c) for c in classes]
        if not clean or any(c is None for c in clean):
            return False                   # not a usable object (a size or minimum not above zero): keep what is stored
        st = self.settings
        st.classes = clean
        return self._store(st, text)

    def set_required(self, sector, metres):
        if not (math.isfinite(metres) and metres >= 0):
            return False
        st = self.settings
        st.required = dict(st.required, **{sector: float(metres)})
        return self._store(st, f'Perception range · required {sector} {float(metres):g} m')

    def set_option(self, **kw):
        st = self.settings
        for k, v in kw.items():
            setattr(st, k, v)
        return self._store(st, 'Perception range · ' + ', '.join(k.replace('_', ' ') for k in kw))

    # ---- state
    def _quick(self):
        return self.result is None or self.result.metrics.get('seconds', 99) < AUTO_MAX_S

    def _mark_stale(self):
        if self.result is not None and not self.stale:
            self.stale = True
            self.changed.emit()
        if self.active and self._quick() and not self.busy:
            self.timer.start()

    def _on_analysis(self, *_):
        if self._for is None or self.ctx.analysis.result is not self._for[2]:
            self._mark_stale()

    def _on_edit(self, *_):
        if self._for is not None and self.settings.to_dict() != self._for[1]:
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
        res = ctx.analysis.result
        if res is None:
            self.status = 'Waiting for the analysis'
            self.changed.emit()
            return False
        win = ctx.window
        if self.busy:
            if quiet or (win is not None and 'perception' in win._jobs):
                return False
            self._gen += 1
        project = ctx.store.project.snapshot()
        st = PR.PerceptionSettings.from_dict(project.analysis.perception)
        eng = win._shared_engine() if win is not None else ctx.analysis.engine.share()
        self._gen += 1
        gen = self._gen
        self.busy = True
        self.status = 'Computing the perception range…'
        key = (ctx.store.project, st.to_dict(), res)

        def work(progress, cancel):
            return key, PR.analyse(project, eng, res, st, progress=progress, cancel=cancel)
        if not quiet and win is not None and hasattr(win, 'run_job'):
            ok = win.run_job('perception', 'Perception range', work, lambda out: self._done(gen, out),
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
        key, pr = out if out is not None else (None, None)
        if pr is None or key[0] is not self.ctx.store.project:
            self.status = 'Cancelled' if pr is None else ''
            self.changed.emit()
            return
        self.result, self._for = pr, key
        self.stale = key[2] is not self.ctx.analysis.result or key[1] != self.settings.to_dict()
        m = pr.metrics
        self.status = (f'{m["passed"]} of {m["checks"]} sector checks pass · {m["points"]:,} object placements · '
                       f'{m["seconds"]:.1f} s').replace(',', ' ')
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
def _scale(rmax, radius_px):
    return radius_px / math.sqrt(max(rmax, 1.0))


def _pt(c: QPointF, k, r, az_deg) -> QPointF:
    """Forward up, left to the left, radius ∝ √range."""
    s = math.sqrt(max(r, 0.0)) * k
    a = math.radians(az_deg)
    return QPointF(c.x() - math.sin(a) * s, c.y() - math.cos(a) * s)


def _rings(p: QPainter, c: QPointF, k, rmax, labels=True, small=False):
    p.setFont(mono(9 if small else 10))
    for ring in RINGS:
        if ring > rmax * 1.001:
            break
        rr = math.sqrt(ring) * k
        p.setPen(QPen(THEME.color('grid2'), 1))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(c, rr, rr)
        if labels:
            p.setPen(THEME.color('muted'))
            p.drawText(QPointF(c.x() + 4, c.y() - rr + (11 if small else 13)), f'{ring} m')


def _envelope(c, k, ranges, az) -> QPolygonF:
    return QPolygonF([_pt(c, k, r, a) for r, a in zip(ranges, az)])


def _required(c, k, pr: PR.PerceptionResult) -> list:
    """(polygon, sector, required) per sector: the arc at the required range."""
    out = []
    for s in pr.sectors:
        a0 = s['centre_deg'] - 22.5
        pts = [_pt(c, k, s['required_m'], a0 + t * 45.0 / 12) for t in range(13)]
        out.append((QPolygonF(pts), s))
    return out


class EnvelopeMap(ChartBase):
    """The detection envelopes around the vehicle: per class (the farthest range any camera resolves it), or for one
    class per camera; the required ranges per sector; the cameras and the body in the middle."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.pr: PR.PerceptionResult | None = None
        self.k = 0
        self.mode = 'classes'
        self.setMinimumSize(320, 320)

    def set_result(self, pr):
        self.pr = pr
        self.k = min(self.k, len(pr.classes) - 1) if pr is not None and pr.classes else 0
        self.update()

    def set_class(self, k):
        self.k = int(k)
        self.update()

    def set_mode(self, m):
        self.mode = m
        self.update()

    def _geom(self):
        pr = self.pr
        r = QRectF(self.rect()).adjusted(16, 30, -16, -16)
        c = r.center()
        rad = min(r.width(), r.height()) / 2 - 6
        return c, _scale(float(pr.ranges[-1]), rad)

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), THEME.color('viewport'))
        pr = self.pr
        if pr is None:
            p.setPen(THEME.color('muted'))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, 'Run the perception range')
            return
        c, k = self._geom()
        _rings(p, c, k, float(pr.ranges[-1]))
        # sector spokes and names
        p.setFont(sans(10))
        for s in pr.sectors:
            a = s['centre_deg'] - 22.5
            p.setPen(QPen(THEME.color('grid2'), 1, Qt.PenStyle.DotLine))
            p.drawLine(c, _pt(c, k, float(pr.ranges[-1]), a))
        # required ranges for the chosen class
        cls = pr.classes[self.k]
        for poly, s in _required(c, k, pr):
            ok = s['classes'][cls.name]['pass']
            pen = QPen(THEME.color('good' if ok else 'critical'), 1.6, Qt.PenStyle.DashLine)
            p.setPen(pen)
            p.drawPolyline(poly)
        if self.mode == 'cameras':
            for ci, (uid, tag, slot) in enumerate(pr.cams):
                col = THEME.cam(slot)
                p.setPen(QPen(col, 1.8))
                fill = QColor(col)
                fill.setAlphaF(0.07)
                p.setBrush(fill)
                p.drawPolygon(_envelope(c, k, pr.cam_far[ci, self.k], pr.az))
        else:
            order = [i for i in range(len(pr.classes)) if i != self.k] + [self.k]
            for i in order:
                col = class_color(i)
                sel = i == self.k
                fill = QColor(col)
                fill.setAlphaF(0.12 if sel else 0.0)
                p.setBrush(fill)
                p.setPen(QPen(col, 2.4 if sel else 1.3, Qt.PenStyle.SolidLine if sel else Qt.PenStyle.SolidLine))
                p.drawPolygon(_envelope(c, k, pr.far[i], pr.az))
        # body and cameras
        if pr.body is not None:
            (x0, y0), (x1, y1) = pr.body
            body = QPolygonF([self._world(c, k, pr, x, y) for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))])
            p.setPen(QPen(THEME.color('strong'), 1))
            p.setBrush(THEME.color('body'))
            p.drawPolygon(body)
        p.setFont(sans(10, QFont.Weight.DemiBold))
        p.setPen(THEME.color('ink2'))
        for s in pr.sectors:
            q = _pt(c, k, float(pr.ranges[-1]) * 0.995, s['centre_deg'])
            name = s['name'].replace('-', ' ').capitalize()
            w = p.fontMetrics().horizontalAdvance(name)
            p.drawText(QPointF(min(max(q.x() - w / 2, 4), self.width() - w - 4), min(max(q.y() + 4, 16), self.height() - 4)),
                       name)
        # legend
        p.setFont(sans(11))
        x, y = 12, 18
        if self.mode == 'cameras':
            p.setPen(THEME.color('ink'))
            text = f'{cls.name} · by camera'
            p.drawText(QPointF(x, y), text)
            x += p.fontMetrics().horizontalAdvance(text) + 14
            for uid, tag, slot in pr.cams:
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(THEME.cam(slot))
                p.drawRect(QRectF(x, y - 9, 9, 9))
                p.setPen(THEME.color('ink2'))
                p.drawText(QPointF(x + 13, y), tag)
                x += p.fontMetrics().horizontalAdvance(tag) + 26
        else:
            for i, cl in enumerate(pr.classes):
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(class_color(i))
                p.drawRect(QRectF(x, y - 9, 9, 9))
                p.setPen(THEME.color('ink' if i == self.k else 'ink2'))
                p.drawText(QPointF(x + 13, y), cl.name)
                x += p.fontMetrics().horizontalAdvance(cl.name) + 28
        p.setPen(THEME.color('muted'))
        p.setFont(sans(10))
        p.drawText(QRectF(0, self.height() - 18, self.width() - 10, 16), Qt.AlignmentFlag.AlignRight,
                   'dashed: required range for ' + cls.name.lower() + ' (green met, red short) · √ range scale')

    @staticmethod
    def _world(c, k, pr, x, y):
        dx, dy = x - pr.centre[0], y - pr.centre[1]
        r = math.hypot(dx, dy)
        return _pt(c, k, r, math.degrees(math.atan2(dy, dx)))

    def mouseMoveEvent(self, e):
        pr = self.pr
        if pr is None:
            return
        c, k = self._geom()
        dx, dy = e.position().x() - c.x(), e.position().y() - c.y()
        s = math.hypot(dx, dy)
        if s < 2:
            return
        r = (s / k) ** 2
        az = math.degrees(math.atan2(-dx, -dy)) % 360.0
        lines = [f'<b>{az:.0f}°</b> · {r:.1f} m from the centre']
        for i, cl in enumerate(pr.classes):
            far, near, tag = pr.at(i, az)
            lines.append(f'{cl.name}: ' + (f'to {far:.0f} m ({tag})' if far > 0 else 'not resolved'))
        QToolTip.showText(e.globalPosition().toPoint(), '<br>'.join(lines), self)

    def leaveEvent(self, e):
        QToolTip.hideText()


class RoseGrid(ChartBase):
    """One small polar rose per class: its envelope and the required range per sector (green met, red short)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.pr = None
        self.setMinimumSize(300, 260)

    def set_result(self, pr):
        self.pr = pr
        self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), THEME.color('panel'))
        pr = self.pr
        if pr is None:
            return
        n = len(pr.classes)
        cols = 2 if n > 1 else 1
        rows = math.ceil(n / cols)
        cw, ch = self.width() / cols, self.height() / rows
        rmax = float(pr.ranges[-1])
        for i, cl in enumerate(pr.classes):
            x0, y0 = (i % cols) * cw, (i // cols) * ch
            c = QPointF(x0 + cw / 2, y0 + ch / 2 + 8)
            rad = min(cw, ch - 22) / 2 - 8
            k = _scale(rmax, rad)
            _rings(p, c, k, rmax, labels=False, small=True)
            col = class_color(i)
            fill = QColor(col)
            fill.setAlphaF(0.18)
            p.setPen(QPen(col, 1.6))
            p.setBrush(fill)
            p.drawPolygon(_envelope(c, k, pr.far[i], pr.az))
            p.setBrush(Qt.BrushStyle.NoBrush)
            met = 0
            for poly, s in _required(c, k, pr):
                ok = s['classes'][cl.name]['pass']
                met += ok
                p.setPen(QPen(THEME.color('good' if ok else 'critical'), 1.4, Qt.PenStyle.DashLine))
                p.drawPolyline(poly)
            p.setFont(sans(11, QFont.Weight.DemiBold))
            p.setPen(THEME.color('ink'))
            p.drawText(QPointF(x0 + 8, y0 + 16), cl.name)
            p.setFont(sans(10))
            tone = 'good' if met == len(pr.sectors) else 'warning' if met >= len(pr.sectors) - 2 else 'critical'
            p.setPen(QColor(STATUS[tone]))
            txt = f'{met}/{len(pr.sectors)} sectors'
            p.drawText(QPointF(x0 + cw - 8 - p.fontMetrics().horizontalAdvance(txt), y0 + 16), txt)
