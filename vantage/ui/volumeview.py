"""3D blind volume in the UI: the controller that keeps a result for the project (shared by the Blind volume page and
the 3D overlay), the reach rose chart and the plan map of the blind space."""
from __future__ import annotations

import math
import threading
import traceback

import numpy as np
from PyQt6.QtCore import Qt, QObject, QPointF, QRectF, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QImage, QPainter, QPainterPath, QPen, QPolygonF
from PyQt6.QtWidgets import QToolTip, QWidget

from ..core import volume as vol
from .charts import ChartBase
from .colormaps import redundancy
from .theme import THEME, STATUS, mono, sans
from .viewport3d import LIGHT, TriDraw

AUTO_MAX_S = 2.0            # re-run by itself after an edit when the last run took less than this
GL_FACES, SOFT_FACES = 60_000, 6_000
SMOOTH, BLUR = 3, 1.0       # the drawn surface: relaxation passes and field blur (level steps terrace otherwise)


class _Sig(QObject):
    done = pyqtSignal(int, object)
    failed = pyqtSignal(int, str)
    progress = pyqtSignal(int, float)


class VolumeController(QObject):
    """The project's blind-volume result.  Runs on demand (the page's Run button: a main-window job with progress and
    Cancel) and again by itself after edits while the page or the 3D overlay shows it and a run is quick."""
    changed = pyqtSignal()
    progress = pyqtSignal(float)

    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        self.result: vol.VolumeResult | None = None
        self.stale = False
        self.busy = False
        self.status = ''
        self.active = False                 # the Blind volume page is showing
        self.cut_z = 0.9                    # the 3D views show the blind space up to this height (child height)
        self._gen = 0
        self._for = None                    # (project, settings dict, analysis result) of the shown result
        self._draw = {}
        self.sig = _Sig()
        self.sig.done.connect(self._done)
        self.sig.failed.connect(self._failed)
        self.sig.progress.connect(lambda g, f: self.progress.emit(f) if g == self._gen else None)
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(400)
        self.timer.timeout.connect(lambda: self.run(quiet=True))
        ctx.analysis.finished.connect(self._on_analysis)
        ctx.store.changed.connect(self._on_edit)
        ctx.store.project_replaced.connect(self._reset)
        ctx.overlays_changed.connect(self._on_overlays)

    # ---- settings
    @property
    def settings(self) -> vol.VolumeSettings:
        return vol.VolumeSettings.from_dict(self.ctx.store.project.analysis.volume)

    def set_settings(self, **kw):
        st = self.settings
        for k, v in kw.items():
            setattr(st, k, float(v))
        new = vol.VolumeSettings.from_dict(st.to_dict()).to_dict()
        if new == self.settings.to_dict():
            return False
        names = {'height': 'height', 'dz': 'level step', 'cell': 'column spacing'}

        def fn(p):
            p.analysis.volume = new
        self.ctx.store.edit('analysis', fn, 'Blind volume ' + ', '.join(names.get(k, k) for k in kw))
        return True

    # ---- state
    def wanted(self) -> bool:
        return self.active or bool(self.ctx.overlays.get('volume'))

    def _quick(self) -> bool:
        return self.result is None or self.result.metrics.get('seconds', 99) < AUTO_MAX_S

    def _mark_stale(self):
        if self.result is not None and not self.stale:
            self.stale = True
            self.changed.emit()
        if self.wanted() and self._quick() and not self.busy:
            self.timer.start()

    def _on_analysis(self, *_):
        if self._for is None or self.ctx.analysis.result is not self._for[2]:
            self._mark_stale()

    def _on_edit(self, *_):
        if self._for is not None and self.settings.to_dict() != self._for[1]:
            self._mark_stale()

    def _on_overlays(self):
        if self.ctx.overlays.get('volume') and (self.result is None or self.stale) and not self.busy:
            self.timer.start()

    def set_active(self, on: bool):
        self.active = on
        if on and (self.result is None or self.stale) and self._quick():
            self.timer.start()

    def set_cut(self, z: float):
        if abs(z - self.cut_z) > 1e-9:
            self.cut_z = float(z)
            self.changed.emit()

    def cut_level(self) -> int:
        """Number of levels the 3D views show."""
        vr = self.result
        return 0 if vr is None else vr.level_index(self.cut_z) + 1

    def _reset(self, *_):
        self._gen += 1
        self.result, self._for, self.stale, self.busy = None, None, False, False
        self._draw = {}
        self.status = ''
        self.changed.emit()
        if self.wanted():
            self.timer.start()

    # ---- run
    def run(self, quiet=False) -> bool:
        """Start a run; quiet runs (after an edit) skip the main window's job bar."""
        ctx = self.ctx
        res = ctx.analysis.result
        if res is None or ctx.store.project.vehicle.mesh is None:
            self.status = 'Waiting for the analysis'
            self.changed.emit()
            return False
        win = ctx.window
        if self.busy:
            if quiet:
                return False
            if win is not None and 'volume' in win._jobs:
                return False
            self._gen += 1                  # an explicit run replaces a quiet one in flight
        project = ctx.store.project.snapshot()
        st = vol.VolumeSettings.from_dict(project.analysis.volume)
        eng = win._shared_engine() if win is not None else ctx.analysis.engine.share()
        self._gen += 1
        gen = self._gen
        self.busy = True
        self.status = 'Computing the blind volume…'
        self.changed.emit()
        live = ctx.store.project
        key = (live, st.to_dict(), res)

        def work(progress, cancel):
            return key, vol.analyse_volume(project, eng, res, st.height, st.dz, st.cell or None,
                                           progress=progress, cancel=cancel)
        if not quiet and win is not None and hasattr(win, 'run_job'):
            return win.run_job('volume', 'Blind volume', work, lambda out: self._done(gen, out),
                               lambda m: self._failed(gen, m))

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
        key, vr = out if out is not None else (None, None)
        if vr is None or key[0] is not self.ctx.store.project:
            self.status = 'Cancelled' if vr is None else ''
            self.changed.emit()
            return
        self.result, self._for = vr, key
        self._draw = {}
        self.stale = key[2] is not self.ctx.analysis.result or key[1] != self.settings.to_dict()
        m = vr.metrics
        self.status = f'{m["points"]:,} points · {m["seconds"]:.1f} s'.replace(',', ' ')
        self.changed.emit()
        if self.stale and self.wanted() and self._quick():
            self.timer.start()

    def _failed(self, gen, msg):
        if gen != self._gen:
            return
        self.busy = False
        self.status = 'Failed: ' + msg.splitlines()[0]
        print(msg)
        self.changed.emit()

    # ---- drawing
    def tri_draw(self, gl: bool) -> TriDraw | None:
        """The blind surface for a 3D view: translucent, lit by the shader renderer (gl) or pre-shaded (software)."""
        vr = self.result
        if vr is None:
            return None
        top = self.cut_level()
        key = (id(vr), gl, THEME.name, top)
        if key not in self._draw:
            if len(self._draw) > 24:
                self._draw.clear()
            limit = GL_FACES if gl else SOFT_FACES
            pool = 1
            P, F = vol.blind_surface(vr, SMOOTH, caps=False, top=top, body_walls=False, blur=BLUR)
            while len(F) > limit and pool < 16:
                pool *= 2
                P, F = vol.blind_surface(vr, SMOOTH, caps=False, pool=pool, top=top, body_walls=False, blur=BLUR)
            tri = P[F]
            base = np.array(QColor(STATUS['critical']).getRgbF()[:3], np.float32)
            fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])       # area-weighted face normals
            nrm = None
            if gl:
                rgb = np.tile(base, (len(tri), 1))
                vn = np.zeros_like(P)
                for k in range(3):
                    np.add.at(vn, F[:, k], fn)
                vn /= np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)
                nrm = vn[F].astype(np.float32)                                 # smooth shading
            else:
                n = fn / np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-12)
                rgb = base[None] * (0.55 + 0.45 * np.clip(n @ LIGHT, 0, 1))[:, None]
            col = np.c_[rgb, np.full(len(tri), 0.34 if gl else 0.42, np.float32)]
            self._draw[key] = TriDraw(tri.astype(np.float32), col, depth=True, normals=nrm) if len(tri) else None
        return self._draw[key]


# ------------------------------------------------------------------------------------------------- charts
SERIES = ['#e66767', '#ec835a', '#fab219', '#3987e5', '#199e70', '#9085e9']


class RoseChart(ChartBase):
    """Polar chart of how far from the body something can hide, per azimuth (forward up, left to the left).
    The radius starts at the body outline; each series is a closed curve."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.title = 'Reach · how far from the body something can hide'
        self.sector_deg = np.zeros(0)
        self.series = []            # (name, values (S,), QColor)
        self.rmax = 1.0
        self.setMinimumSize(240, 240)

    def set_data(self, sector_deg, series, rmax):
        self.sector_deg = np.asarray(sector_deg, float)
        self.series = list(series)
        self.rmax = max(float(rmax), 0.1)
        self.update()

    def _geom(self):
        top = 22
        leg = 18 * max(1, math.ceil(len(self.series) / 2)) + 6
        h = self.height() - top - leg
        R = max(20.0, min(self.width(), h) / 2 - 22)
        c = QPointF(self.width() / 2, top + h / 2)
        return c, R, R * 0.2

    def _pt(self, c, r0, R, az, v):
        r = r0 + (R - r0) * min(max(v, 0.0), self.rmax) / self.rmax
        a = math.radians(az)
        return QPointF(c.x() - math.sin(a) * r, c.y() - math.cos(a) * r)

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.paint_title(p)
        c, R, r0 = self._geom()
        hair = THEME.color('hair')
        # rings every 0.5 m (1 m when far) with labels, spokes every 45°
        step = 0.5 if self.rmax <= 3.0 else 1.0
        rings = []
        k = 1
        while k * step <= self.rmax + 1e-9:
            r = r0 + (R - r0) * k * step / self.rmax
            p.setPen(QPen(hair, 1))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawEllipse(c, r, r)
            rings.append((k * step, r))
            k += 1
        p.setPen(QPen(hair, 1))
        for az in range(0, 360, 45):
            p.drawLine(self._pt(c, r0, R, az, 0), self._pt(c, r0, R, az, self.rmax))
        p.setFont(sans(10))
        p.setPen(THEME.color('ink2'))
        for az, t in ((0, 'Front'), (90, 'Left'), (180, 'Rear'), (270, 'Right')):
            q = self._pt(c, r0, R, az, self.rmax)
            a = math.radians(az)
            q = QPointF(q.x() - math.sin(a) * 12, q.y() - math.cos(a) * 9)
            p.drawText(QRectF(q.x() - 30, q.y() - 8, 60, 16), Qt.AlignmentFlag.AlignCenter, t)
        # the body (radius 0 of the plot is its outline)
        p.setPen(QPen(THEME.color('strong'), 1))
        p.setBrush(THEME.color('body', 0.85))
        p.drawEllipse(c, r0, r0)
        car = QRectF(c.x() - r0 * 0.38, c.y() - r0 * 0.72, r0 * 0.76, r0 * 1.44)
        p.setBrush(THEME.color('raised'))
        p.drawRoundedRect(car, r0 * 0.2, r0 * 0.2)
        p.setBrush(THEME.color('strong'))
        p.drawRoundedRect(QRectF(car.left() + r0 * 0.1, car.top() + r0 * 0.36, car.width() - r0 * 0.2, r0 * 0.24), 2, 2)
        # series
        n = len(self.sector_deg)
        for name, vals, col in self.series:
            if not n:
                break
            poly = QPolygonF([self._pt(c, r0, R, a, v) for a, v in zip(self.sector_deg, vals)])
            fill = QColor(col)
            fill.setAlphaF(0.14)
            p.setBrush(fill)
            p.setPen(QPen(QColor(col), 1.6))
            p.drawPolygon(poly)
        p.setFont(mono(10))
        for v, r in rings:
            q = QPointF(c.x() - r * math.sin(math.radians(-45)) + 2, c.y() - r * math.cos(math.radians(-45)) - 2)
            p.setPen(THEME.color('muted'))
            p.drawText(q, f'{v:g} m')
        # legend
        p.setFont(sans(11))
        y = self.height() - 18 * math.ceil(max(len(self.series), 1) / 2) - 2
        for i, (name, vals, col) in enumerate(self.series):
            x = 4 + (i % 2) * self.width() / 2
            yy = y + (i // 2) * 18
            p.fillRect(QRectF(x, yy + 5, 12, 3), QColor(col))
            p.setPen(THEME.color('ink2'))
            mx = float(np.max(vals)) if len(vals) else 0.0
            p.drawText(QRectF(x + 16, yy, self.width() / 2 - 22, 14), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                       f'{name} · max {mx:.2f} m' if mx > 0 else f'{name} · none')
        if not self.series:
            p.setPen(THEME.color('muted'))
            p.drawText(QRectF(0, 0, self.width(), self.height()), Qt.AlignmentFlag.AlignCenter, 'No result yet')
        p.end()

    def mouseMoveEvent(self, e):
        if not len(self.sector_deg) or not self.series:
            return
        c, R, r0 = self._geom()
        dx, dy = e.position().x() - c.x(), e.position().y() - c.y()
        if math.hypot(dx, dy) > R + 10:
            QToolTip.hideText()
            return
        az = math.degrees(math.atan2(-dx, -dy))
        k = int(((az + 180.0) % 360.0) / (360.0 / len(self.sector_deg))) % len(self.sector_deg)
        lines = [f'azimuth {self.sector_deg[k]:+.0f}°']
        lines += [f'{name}: {vals[k]:.2f} m' if vals[k] > 0 else f'{name}: none' for name, vals, _ in self.series]
        QToolTip.showText(e.globalPosition().toPoint(), '\n'.join(lines), self)


class VolumeMap(QWidget):
    """Plan of the blind space: 'slice' = blind cells at one height (seen cells by camera count), 'detect' = the
    smallest object standing on each cell that a camera still sees (bins at the object classes)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.vr = None
        self.mode = 'slice'
        self.level = 0
        self._img = None
        self._key = None
        self.setMouseTracking(True)
        self.setMinimumSize(200, 160)
        THEME.changed.connect(lambda *_: (setattr(self, '_key', None), self.update()))

    def set_result(self, vr):
        self.vr = vr
        self._key = None
        if vr is not None:
            self.level = min(self.level, len(vr.z) - 1)
        self.update()

    def set_mode(self, mode):
        self.mode = mode
        self._key = None
        self.update()

    def set_level(self, k):
        self.level = int(k)
        self._key = None
        self.update()

    def _image(self):
        vr = self.vr
        key = (id(vr), self.mode, self.level, THEME.name)
        if key == self._key:
            return self._img
        rgba = np.zeros((vr.ny, vr.nx, 4), np.uint8)

        def put(mask, hexcol, a=255):
            c = QColor(hexcol)
            rgba[mask] = (c.red(), c.green(), c.blue(), a)
        put(vr.body, THEME.tok['body'], 200)
        if self.mode == 'slice':
            cnt = vr.count[self.level]
            roi = vr.roi
            r1, r2, r3 = redundancy()               # the plan view's scale, per theme
            put(roi & (cnt == 1), r1, 110)
            put(roi & (cnt == 2), r2, 110)
            put(roi & (cnt >= 3), r3, 110)
            put(roi & (cnt == 0), STATUS['critical'], 235)
        else:
            h = vr.detect_h
            roi = vr.roi
            put(roi & (h <= vr.dz + 1e-9), THEME.tok['ground'], 70)
            objs = sorted(vr.objects, key=lambda o: o[1])
            lo = vr.dz
            cols = [STATUS['warning'], STATUS['serious'], STATUS['critical']]
            for i, (name, hh) in enumerate(objs):
                put(roi & (h > lo + 1e-9) & (h <= hh + 1e-9), cols[min(i, len(cols) - 1)], 225)
                lo = hh
            put(roi & (h > lo + 1e-9), '#7a1f1f', 245)
        img = QImage(rgba.data, vr.nx, vr.ny, vr.nx * 4, QImage.Format.Format_RGBA8888).copy()
        self._img = img.mirrored(False, True)          # row 0 = smallest y; the screen's y points down
        self._key = key
        return self._img

    def _frame(self):
        vr = self.vr
        W, H = self.width(), self.height() - 22
        w, h = vr.nx * vr.cell, vr.ny * vr.cell
        s = min((W - 16) / w, (H - 16) / h)
        ox = (W - w * s) / 2
        oy = 4 + (H - h * s) / 2
        return s, ox, oy

    def w2s(self, x, y):
        vr = self.vr
        s, ox, oy = self._frame()
        return QPointF(ox + (x - vr.x0) * s, oy + (vr.y0 + vr.ny * vr.cell - y) * s)

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        vr = self.vr
        if vr is None:
            p.setPen(THEME.color('muted'))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, 'No result yet')
            p.end()
            return
        s, ox, oy = self._frame()
        img = self._image()
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
        p.drawImage(QRectF(ox, oy, vr.nx * vr.cell * s, vr.ny * vr.cell * s), img)
        p.setPen(QPen(THEME.color('ink'), 1.2))
        p.setBrush(Qt.BrushStyle.NoBrush)
        for line in vr.outline:
            if len(line) > 1:
                p.drawPolyline(QPolygonF([self.w2s(x, y) for x, y in line[:, :2]]))
        # legend
        p.setFont(sans(11))
        y = self.height() - 16
        x = 6
        if self.mode == 'slice':
            r1, r2, r3 = redundancy()
            items = [(STATUS['critical'], 'blind'), (r1, '1 camera'), (r2, '2'), (r3, '3+')]
        else:
            items = [(THEME.tok['ground'], f'≤ {vr.dz * 100:.0f} cm')]
            lo = vr.dz
            cols = [STATUS['warning'], STATUS['serious'], STATUS['critical']]
            for i, (name, hh) in enumerate(sorted(vr.objects, key=lambda o: o[1])):
                items.append((cols[min(i, 2)], f'{lo:g}–{hh:g} m'))
                lo = hh
            items.append(('#7a1f1f', f'> {lo:g} m or none'))
        for col, t in items:
            p.fillRect(QRectF(x, y + 3, 10, 10), QColor(col))
            p.setPen(THEME.color('ink2'))
            tw = p.fontMetrics().horizontalAdvance(t)
            p.drawText(QPointF(x + 14, y + 12), t)
            x += 14 + tw + 12
        p.end()

    def mouseMoveEvent(self, e):
        vr = self.vr
        if vr is None:
            return
        s, ox, oy = self._frame()
        x = vr.x0 + (e.position().x() - ox) / s
        y = vr.y0 + vr.ny * vr.cell - (e.position().y() - oy) / s
        i, j = int((x - vr.x0) / vr.cell), int((y - vr.y0) / vr.cell)
        if not (0 <= i < vr.nx and 0 <= j < vr.ny) or not vr.roi[j, i]:
            QToolTip.hideText()
            return
        h = vr.detect_h[j, i]
        lines = [f'x {x:+.2f} m  y {y:+.2f} m · {vr.dist[j, i]:.2f} m from the body']
        if self.mode == 'slice':
            n = int(vr.count[self.level, j, i])
            lines.append(f'at {vr.z[self.level]:.2f} m: ' + (f'{n} camera(s)' if n else 'blind'))
        lines.append('seen from the ground up' if h <= vr.dz + 1e-9 else
                     (f'lowest seen point {h:.2f} m: smaller objects are missed' if np.isfinite(h) else
                      f'nothing seen up to {vr.height:.2f} m'))
        QToolTip.showText(e.globalPosition().toPoint(), '\n'.join(lines), self)
