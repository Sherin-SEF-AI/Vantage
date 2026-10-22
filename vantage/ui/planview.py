"""Top-down plan views drawn with QPainter: coverage, density, redundancy, footprints, stitched
bird's-eye, calibration targets and variant comparison."""
from __future__ import annotations

import math

import numpy as np
from PyQt6.QtCore import Qt, QPointF, QRectF, pyqtSignal, QLineF
from PyQt6.QtGui import QPainter, QColor, QPen, QImage, QPolygonF, QBrush, QFont, QTransform, QPainterPath
from PyQt6.QtWidgets import QWidget

from ..core.analysis import surface_frame
from ..core.lens import resolve
from ..core.mathutil import rot_zyx
from ..core.mesh import ROLE_GLASS, ROLE_WHEELS, ROLE_IGNORED
from . import guides, maps, ops
from .theme import THEME, mono, sans, STATUS, SEQ, REDUNDANCY, DIVERGING
from .scene import paint_ground_tooltip, _qc
from .widgets import paint_chip

_car_cache = {}


def car_top_image(mesh, ppm=180):
    """Orthographic top render of the body (flat shaded), cached per mesh and theme."""
    key = (id(mesh), mesh.n_tris, ppm, THEME.name)
    if key in _car_cache:
        return _car_cache[key]
    lo, hi = mesh.bbox
    W = int((hi[0] - lo[0]) * ppm) + 4
    H = int((hi[1] - lo[1]) * ppm) + 4
    img = QImage(W, H, QImage.Format.Format_ARGB32_Premultiplied)
    img.fill(Qt.GlobalColor.transparent)
    tri = mesh.tri
    n = mesh.normals
    keepm = np.array([p.role != ROLE_IGNORED for p in mesh.parts] or [True])[mesh.face_part]
    up = n[:, 2] > 0.02
    keep = keepm & up
    tri, n, fp = tri[keep], n[keep], mesh.face_part[keep]
    order = np.argsort(tri[:, :, 2].mean(axis=1))
    body = QColor(THEME.tok['body'])
    glass = QColor(THEME.tok['glass'])
    roles = [p.role for p in mesh.parts]
    names = [p.name.lower() for p in mesh.parts]
    p = QPainter(img)
    p.setRenderHint(QPainter.RenderHint.Antialiasing, False)
    light = np.array([0.25, 0.35, 0.9])
    light /= np.linalg.norm(light)
    for i in order:
        k = fp[i]
        base = glass if (roles[k] == ROLE_GLASS or 'glass' in names[k]) else QColor(40, 40, 40) if roles[k] == ROLE_WHEELS else body
        s = 0.62 + 0.38 * max(0.0, float(n[i] @ light))
        c = QColor.fromRgbF(min(1, base.redF() * s), min(1, base.greenF() * s), min(1, base.blueF() * s))
        p.setBrush(c)
        p.setPen(QPen(c, 0.8))
        a = tri[i]
        p.drawPolygon(QPolygonF([QPointF((a[j, 0] - lo[0]) * ppm + 2, (hi[1] - a[j, 1]) * ppm + 2) for j in range(3)]))
    p.end()
    _car_cache.clear() if len(_car_cache) > 12 else None
    _car_cache[key] = (img, lo, hi, ppm)
    return _car_cache[key]


class PlanView(QWidget):
    """mode: coverage | density | redundancy | footprints | bev | targets | compare | delta | blank"""
    viewChanged = pyqtSignal()
    seamMoved = pyqtSignal()

    def __init__(self, ctx, mode='coverage', parent=None, interactive=True):
        super().__init__(parent)
        self.ctx = ctx
        self.mode = mode
        self.interactive = interactive
        self.result_fn = None           # optional: () -> AnalysisResult (compare/pinned)
        self.cameras_fn = None          # optional: () -> list[Camera]
        self.delta_fn = None            # optional: () -> (res_a, res_b)
        self.label = ''
        self.legend = True
        self.show_tooltip = True
        self.edit_polygon = False
        self.zoom_rect = None           # (x0, y0, x1, y1) to frame
        self.scale = 60.0               # px per metre
        self.centre = np.array([1.2, 0.0])
        self._fitted = False
        self._img_key = None
        self._img = None
        self.hover = None
        self.hover_world = None
        self.drag = None
        self.suggestions = []           # [(uid, x, y)]
        self.pinned_marker = None
        self.show_objects = False       # bird's-eye mode: where standing objects vanish, and their smears
        self._smears = None             # smears of the pole under the cursor (Objects heatmap)
        self._obj_img = self._obj_key = None
        self._hs_smears = (None, [])
        self.setMouseTracking(True)
        self.setMinimumSize(160, 120)
        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        st = ctx.store
        # bound methods (not lambdas) so Qt drops the connections when this view is deleted
        st.changed.connect(self._on_change)
        st.selection_changed.connect(self._on_change)
        st.project_replaced.connect(self._refit)
        ctx.analysis.finished.connect(self._on_result)
        ctx.analysis.secondary.connect(self._on_change)
        ctx.overlays_changed.connect(self._on_change)
        if getattr(ctx, 'stitched', None) is not None:
            ctx.stitched.ready.connect(self._on_change)
        THEME.changed.connect(self._on_theme)

    def _on_change(self, *_):
        self.update()

    def _on_theme(self, *_):
        self._img_key = None
        self.update()

    # ------------------------------------------------------------------ data
    @property
    def project(self):
        return self.ctx.store.project

    def result(self):
        if self.result_fn is not None:
            return self.result_fn()
        return self.ctx.analysis.result

    def cameras(self):
        if self.cameras_fn is not None:
            return self.cameras_fn()
        return self.project.variant.cameras

    def _on_result(self, *_):
        if not self._fitted:
            self.fit()
        self.update()

    def _refit(self):
        self._fitted = False
        self.pinned_marker = None           # an issue's location in the previous project
        self.update()

    # ------------------------------------------------------------------ coordinates
    def w2s(self, x, y):
        return QPointF(self.width() / 2 + (x - self.centre[0]) * self.scale,
                       self.height() / 2 - (y - self.centre[1]) * self.scale)

    def w2s_np(self, P):
        P = np.asarray(P, float)
        return np.stack([self.width() / 2 + (P[..., 0] - self.centre[0]) * self.scale,
                         self.height() / 2 - (P[..., 1] - self.centre[1]) * self.scale], -1)

    def s2w(self, sx, sy):
        return np.array([self.centre[0] + (sx - self.width() / 2) / self.scale,
                         self.centre[1] - (sy - self.height() / 2) / self.scale])

    def fit(self, rect=None, margin=1.06):
        res = self.result()
        explicit = rect is not None          # framing a selection is a view to keep, like a zoom
        if rect is None:
            if self.zoom_rect is not None:
                rect = self.zoom_rect
            elif res is not None:
                rect = maps.extent(res)
            elif self.project.vehicle.mesh is not None:
                lo, hi = self.project.vehicle.mesh.bbox
                rect = (lo[0] - 2.5, lo[1] - 2.5, hi[0] + 2.5, hi[1] + 2.5)
            else:
                rect = (-3, -3, 5, 3)
        x0, y0, x1, y1 = rect
        self.centre = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
        w, h = max(self.width(), 50), max(self.height(), 50)
        self.scale = min((w - 24) / max(x1 - x0, 0.1), (h - 40) / max(y1 - y0, 0.1)) / margin
        self._fitted = res is not None or rect is not None
        self._user_view = explicit
        self.update()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        # keep a view the user zoomed or panned to; otherwise keep everything in frame
        if not self._fitted or e.oldSize().width() < 50 or not getattr(self, '_user_view', False):
            self.fit()

    # ------------------------------------------------------------------ painting
    def paintEvent(self, e):
        p = QPainter(self)
        p.fillRect(self.rect(), THEME.color('viewport'))
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self._grid(p)
        res = self.result()
        if res is not None:
            self._map(p, res)
        self._car(p)
        from .rigview import paint_plan as paint_rig
        paint_rig(self, p)
        if self.mode in ('coverage', 'targets', 'compare', 'footprints'):
            from .sensorview import paint_plan as paint_sensors
            paint_sensors(self, p, fields=self.ctx.overlays.get('sensors', True) and self.mode == 'coverage')
        if res is not None:
            self._lines(p, res)
            self._stitched_marks(p, res)
        guides.paint_plan(self, p)
        self._targets(p)
        self._obstacles(p)
        self._cameras(p, res)
        guides.paint_plan_hits(self, p)
        if res is not None and self.mode == 'bev':
            self._seam_handles(p, res)
        if self.edit_polygon:
            self._polygon_handles(p)
        if self.pinned_marker is not None:
            x, y = self.pinned_marker
            q = self.w2s(x, y)
            p.setPen(QPen(THEME.color('ink'), 1))
            p.drawLine(QPointF(q.x() - 8, q.y()), QPointF(q.x() + 8, q.y()))
            p.drawLine(QPointF(q.x(), q.y() - 8), QPointF(q.x(), q.y() + 8))
        self._hud(p, res)
        if self.hover and self.show_tooltip and self.drag is None:
            paint_ground_tooltip(p, self.hover['pos'], self.hover, self.project, self.cameras())
        p.end()

    def _grid(self, p):
        if not self.ctx.overlays.get('grid', True):
            return
        x0, y1 = self.s2w(0, 0)
        x1, y0 = self.s2w(self.width(), self.height())
        step = 0.5 if self.scale > 25 else 1.0
        pen_a = QPen(THEME.color('grid1'), 1)
        pen_b = QPen(THEME.color('grid2'), 1)
        lines_a, lines_b = [], []
        for x in np.arange(math.floor(x0 / step) * step, x1 + step, step):
            q = self.w2s(x, 0).x()
            (lines_b if abs(x - round(x)) < 1e-6 else lines_a).append(QLineF(q, 0, q, self.height()))
        for y in np.arange(math.floor(y0 / step) * step, y1 + step, step):
            q = self.w2s(0, y).y()
            (lines_b if abs(y - round(y)) < 1e-6 else lines_a).append(QLineF(0, q, self.width(), q))
        p.setPen(pen_a)
        p.drawLines(lines_a)
        p.setPen(pen_b)
        p.drawLines(lines_b)

    def _map_mode(self):
        m = self.mode
        if m == 'footprints':
            return 'footprints' if self.ctx.overlays.get('footprints', True) else 'blank'
        if m == 'targets':
            return 'redundancy'
        if m in ('compare',):
            return 'redundancy'
        return m

    def _stitched(self):
        """The objects-in-the-stitched-view result for this view (not for pinned or compared results)."""
        ctl = getattr(self.ctx, 'stitched', None)
        return ctl.result if ctl is not None and self.result_fn is None else None

    def _map(self, p, res):
        mode = self._map_mode()
        if mode == 'stitched':
            sres = self._stitched()
            if sres is None:
                return
            key = ('stitched', id(sres), THEME.name)
            if key != self._img_key:
                self._img = maps.to_qimage(maps.stitched(sres, k=2))
                self._img_key = key
            x0, y0, x1, y1 = maps.stitched_extent(sres)
            p.drawImage(QRectF(self.w2s(x0, y1), self.w2s(x1, y0)), self._img)
            return
        if self.mode == 'delta' and self.delta_fn:
            a, b = self.delta_fn()
            key = ('delta', id(a), id(b), THEME.name)
            if key != self._img_key:
                img = maps.delta(a, b)
                img = np.clip(img, 0, 255)
                img[..., 3] = np.clip(img[..., 3] * 255, 0, 255)
                self._img = maps.to_qimage(img.astype(np.uint8))
                self._img_key = key
        else:
            key = (id(res), mode, THEME.name, frozenset(self.ctx.isolate) if self.ctx.isolate else None)
            if key != self._img_key:
                arr = maps.make(res, mode, k=2, isolate=self.ctx.isolate or None, shadow=False)
                self._img = maps.to_qimage(arr)
                self._img_key = key
        x0, y0, x1, y1 = maps.extent(res)
        a, b = self.w2s(x0, y1), self.w2s(x1, y0)
        p.save()
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, mode in ('footprints', 'bev', 'blank'))
        p.drawImage(QRectF(a, b), self._img)
        p.restore()

    def _car(self, p):
        m = self.project.vehicle.mesh
        if m is None or m.n_tris == 0:
            return
        vg = self.ctx.analysis.engine.vehicle_geom(self.project)
        src = vg.body if vg is not None else m
        img, lo, hi, ppm = car_top_image(src)
        a = self.w2s(lo[0] - 2 / ppm, hi[1] + 2 / ppm)
        w = img.width() / ppm * self.scale
        h = img.height() / ppm * self.scale
        p.save()
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        p.drawImage(QRectF(a.x(), a.y(), w, h), img)
        p.restore()

    def _poly_path(self, pts):
        s = self.w2s_np(pts)
        return QPolygonF([QPointF(*q) for q in s])

    def _lines(self, p, res):
        ov = self.ctx.overlays
        # footprint outlines
        if self.mode in ('coverage', 'footprints', 'density', 'targets') and ov.get('footprints', True):
            for c in res.cams:
                if self.ctx.isolate and c.uid not in self.ctx.isolate:
                    continue
                if len(c.segments):
                    s = self.w2s_np(c.segments)
                    sel = c.uid in self.ctx.store.sel_cams
                    p.setPen(QPen(THEME.cam(c.slot), 1.6 if sel else 1.0))
                    p.drawLines([QLineF(a[0, 0], a[0, 1], a[1, 0], a[1, 1]) for a in s])
        # ROI boundary
        if ov.get('roi', True):
            pen = QPen(THEME.color('ink', 0.85), 1.2, Qt.PenStyle.CustomDashLine)
            pen.setDashPattern([5, 4])
            p.setPen(pen)
            p.setBrush(Qt.BrushStyle.NoBrush)
            for b in res.grid.boundary:
                p.drawPolyline(self._poly_path(b))
        # seams
        if (ov.get('seams', True) or self.mode == 'bev') and res.seams and self.mode not in ('delta',):
            pen = QPen(THEME.color('ink', 0.7), 1.0, Qt.PenStyle.CustomDashLine)
            pen.setDashPattern([3, 3])
            p.setPen(pen)
            for s in res.seams:
                a, b = self._seam_segment(res, s.angle)
                if a is not None:
                    p.drawLine(self.w2s(*a), self.w2s(*b))

    def _seam_segment(self, res, angle):
        g = res.grid
        cx, cy = res.centre
        a = math.radians(angle)
        rr = np.arange(0, max(g.nx, g.ny) * g.cell, g.cell / 2)
        X, Y = cx + rr * math.cos(a), cy + rr * math.sin(a)
        ii = ((X - g.x0) / g.cell).astype(int)
        jj = ((Y - g.y0) / g.cell).astype(int)
        ok = (ii >= 0) & (ii < g.nx) & (jj >= 0) & (jj < g.ny)
        ok[ok] &= (g.roi | g.body)[jj[ok], ii[ok]]
        idx = np.nonzero(ok)[0]
        if len(idx) < 2:
            return None, None
        # start at the body edge
        ok2 = ok.copy()
        ok2[ok] &= g.roi[jj[ok], ii[ok]]
        idx2 = np.nonzero(ok2)[0]
        if len(idx2) < 2:
            return None, None
        return (X[idx2[0]], Y[idx2[0]]), (X[idx2[-1]], Y[idx2[-1]])

    def _seam_handles(self, p, res):
        for i, s in enumerate(res.seams):
            a, b = self._seam_segment(res, s.angle)
            if b is None:
                continue
            q = self.w2s(*b)
            hot = self.drag and self.drag.get('seam') == i
            p.setPen(QPen(THEME.color('ink'), 1))
            p.setBrush(THEME.color('ink') if hot else THEME.color('panel'))
            p.drawRect(QRectF(q.x() - 4.5, q.y() - 4.5, 9, 9))

    def _polygon_handles(self, p):
        poly = self.project.roi.polygon
        if not poly:
            return
        p.setPen(QPen(THEME.color('ink'), 1))
        for i, (x, y) in enumerate(poly):
            q = self.w2s(x, y)
            p.setBrush(THEME.color('ink') if self.drag and self.drag.get('vertex') == i else THEME.color('panel'))
            p.drawRect(QRectF(q.x() - 4, q.y() - 4, 8, 8))

    def _targets(self, p):
        if not (self.mode == 'targets' or self.ctx.overlays.get('targets')):
            return
        st = {s.uid: s for s in self.ctx.analysis.targets}
        sel = self.ctx.store.sel_obj
        for t in self.project.targets:
            w, h = t.size_m
            c, s = math.cos(math.radians(t.yaw)), math.sin(math.radians(t.yaw))
            sq = t.square_mm / 1000
            for i in range(t.cols):
                for j in range(t.rows):
                    u0, v0 = -w / 2 + i * sq, -h / 2 + j * sq
                    pts = [(u0, v0), (u0 + sq, v0), (u0 + sq, v0 + sq), (u0, v0 + sq)]
                    P = [self.w2s(t.x + u * c - v * s, t.y + u * s + v * c) for u, v in pts]
                    p.setPen(Qt.PenStyle.NoPen)
                    p.setBrush(QColor(20, 20, 20) if (i + j) % 2 == 0 else QColor(200, 198, 190))
                    p.drawPolygon(QPolygonF(P))
            ok = st.get(t.uid).ok if t.uid in st else True
            corners = [(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)]
            P = [self.w2s(t.x + u * c - v * s, t.y + u * s + v * c) for u, v in corners]
            if not ok:
                p.setPen(QPen(QColor(STATUS['critical']), 1.5))
                p.setBrush(Qt.BrushStyle.NoBrush)
                pad = 6
                r = QPolygonF(P).boundingRect().adjusted(-pad, -pad, pad, pad)
                p.drawRect(r)
            if sel == ('target', t.uid):
                p.setPen(QPen(THEME.color('ink'), 1.5))
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawRect(QPolygonF(P).boundingRect().adjusted(-3, -3, 3, 3))
            if self.mode == 'targets':
                q = QPolygonF(P).boundingRect()
                p.setFont(mono(10))
                p.setPen(THEME.color('ink'))
                p.drawText(QPointF(q.right() + 3, q.top() + 9), t.name)
        # suggestions
        for uid, x, y, *_ in self.suggestions:
            t = next((t for t in self.project.targets if t.uid == uid), None)
            if t is None:
                continue
            w, h = t.size_m
            a, b = self.w2s(x - w / 2, y + h / 2), self.w2s(x + w / 2, y - h / 2)
            pen = QPen(THEME.color('ink'), 1, Qt.PenStyle.CustomDashLine)
            pen.setDashPattern([3, 3])
            p.setPen(pen)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRect(QRectF(a, b))
            p.drawLine(self.w2s(t.x, t.y), self.w2s(x, y))

    def _obstacles(self, p):
        for ob in self.project.obstacles:
            q = self.w2s(ob.x, ob.y)
            r = max(3.0, ob.radius * self.scale)
            p.setPen(QPen(THEME.color('ink2'), 1.2))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawEllipse(q, r, r)
            seen = self.ctx.analysis.result.obstacles.get(ob.uid) if self.ctx.analysis.result else None
            if seen is not None and not seen:
                p.setPen(QColor(STATUS['critical']))
                p.drawEllipse(q, r + 3, r + 3)

    def _cameras(self, p, res):
        sel = set(self.ctx.store.sel_cams)
        for c in self.cameras():
            if not c.visible or (self.ctx.isolate and c.uid not in self.ctx.isolate):
                continue
            q = self.w2s(c.x, c.y)
            col = THEME.cam(c.slot)
            # aim wedge: horizontal FOV edges
            intr = resolve(self.project.module_for(c), self.project.lens_for(c))
            hf = math.degrees(math.atan2(intr.width / 2, intr.f_px)) if intr.projection == 'rectilinear' else 55.0
            L = 26 if c.uid not in sel else 34
            for da in (-1, 1):
                a = math.radians(c.yaw + da * min(hf, 64))
                p.setPen(QPen(col, 1.2))
                p.drawLine(q, QPointF(q.x() + math.cos(a) * L, q.y() - math.sin(a) * L))
            s = 8 if c.uid not in sel else 10
            p.setPen(QPen(QColor(0, 0, 0, 150), 1))
            p.setBrush(col)
            p.drawRect(QRectF(q.x() - s / 2, q.y() - s / 2, s, s))
            if c.uid in sel:
                p.setPen(QPen(THEME.color('ink'), 1))
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawRect(QRectF(q.x() - s / 2 - 3, q.y() - s / 2 - 3, s + 6, s + 6))

    def _hud(self, p, res):
        p.setFont(sans(11))
        p.setPen(THEME.color('ink2'))
        if self.label:
            p.drawText(QRectF(10, 6, self.width() - 20, 16), Qt.AlignmentFlag.AlignLeft, self.label)
        # axis marker + scale bar
        p.setPen(QPen(QColor('#e0533f'), 1.4))
        x0, y0 = 22, self.height() - 20
        p.drawLine(QPointF(x0, y0), QPointF(x0 + 18, y0))
        p.setPen(QPen(QColor('#3fae49'), 1.4))
        p.drawLine(QPointF(x0, y0), QPointF(x0, y0 - 18))
        meters = 1.0
        for m in (0.25, 0.5, 1, 2, 5, 10):
            if m * self.scale >= 60:
                meters = m
                break
        w = meters * self.scale
        x1, y = self.width() - 18, self.height() - 14
        p.setPen(QPen(THEME.color('ink2'), 1))
        p.drawLine(QPointF(x1 - w, y), QPointF(x1, y))
        p.drawLine(QPointF(x1 - w, y - 3), QPointF(x1 - w, y))
        p.drawLine(QPointF(x1, y - 3), QPointF(x1, y))
        p.setFont(mono(11))
        p.drawText(QRectF(x1 - w, y - 18, w, 14), Qt.AlignmentFlag.AlignRight, f'{meters:g} m')
        if self.legend and res is not None:
            self._legend(p)

    def _stitched_marks(self, p, res):
        """Objects heatmap: hotspots and the smears of the pole under the cursor. Bird's-eye with show_objects: where
        objects vanish over the sector tints, and the smears of a pole at the worst places."""
        if not (self.mode == 'stitched' or (self.mode == 'bev' and self.show_objects)):
            return
        sres = self._stitched()
        if sres is None:
            return
        if self.mode == 'bev':
            key = (id(sres), THEME.name)
            if key != self._obj_key:
                self._obj_img = maps.to_qimage(maps.stitched_zones(sres, k=2))
                self._obj_key = key
            x0, y0, x1, y1 = maps.stitched_extent(sres)
            p.drawImage(QRectF(self.w2s(x0, y1), self.w2s(x1, y0)), self._obj_img)
        p.setFont(mono(10))
        for h in sres.hotspots[:8]:
            q = self.w2s(h.x, h.y)
            r = max(7.0, math.sqrt(h.area_m2 / math.pi) * self.scale)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.setPen(QPen(THEME.color('ink'), 1.2))
            p.drawEllipse(q, r, r)
            txt = f'{h.area_m2:.2f} m²'
            tw = p.fontMetrics().horizontalAdvance(txt) + 8
            box = QRectF(q.x() + r * 0.72, q.y() - r * 0.72 - 15, tw, 14)
            bg = THEME.color('raised')
            bg.setAlphaF(0.85)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(bg)
            p.drawRoundedRect(box, 3, 3)
            p.setPen(THEME.color('ink'))
            p.drawText(box, Qt.AlignmentFlag.AlignCenter, txt)
        if self.mode == 'stitched':
            self._draw_smears(p, self._smears or [])
        else:
            if self._hs_smears[0] != (id(sres), id(res)):
                from ..core.stitched_objects import smears
                sm = []
                for h in sres.hotspots[:3]:
                    sm += smears(self.project, self.ctx.analysis.engine, res, h.x, h.y)
                self._hs_smears = ((id(sres), id(res)), sm)
            self._draw_smears(p, self._hs_smears[1])

    def _draw_smears(self, p, sm):
        """Each camera draws a pole as a smear away from itself: solid where the stitched view shows that camera
        there, dashed where the seams hand that ground to another camera (the pole is lost there)."""
        for d in sm:
            col = THEME.cam(d['slot'])
            P, seen, w = d['points'], d['seen'], d['weight']
            for i in range(len(P) - 1):
                if not (seen[i] and np.isfinite(P[i]).all() and np.isfinite(P[i + 1]).all()):
                    continue
                shown = w[i] >= 0.5
                c = QColor(col)
                c.setAlphaF(0.95 if shown else 0.55)
                p.setPen(QPen(c, 3.0 if shown else 1.4, Qt.PenStyle.SolidLine if shown else Qt.PenStyle.DashLine,
                              Qt.PenCapStyle.RoundCap))
                p.drawLine(self.w2s(*P[i]), self.w2s(*P[i + 1]))
            p.setPen(QPen(THEME.color('viewport'), 1))
            p.setBrush(col)
            p.drawEllipse(self.w2s(*d['foot']), 3.2, 3.2)

    def _legend(self, p):
        mode = self.mode
        x1 = self.width() - 12
        y = 8
        p.setFont(sans(10))
        if mode == 'density':
            w = min(340, self.width() * 0.4)
            x0 = x1 - w
            p.setPen(THEME.color('ink2'))
            p.drawText(QRectF(x0, y, w, 14), Qt.AlignmentFlag.AlignLeft, 'px/cm · best camera')
            p.drawText(QRectF(x0, y, w, 14), Qt.AlignmentFlag.AlignRight, 'log scale')
            yb = y + 18
            n = len(SEQ)
            for i, c in enumerate(SEQ):
                p.fillRect(QRectF(x0 + i * w / n, yb, w / n + 0.5, 8), QColor(c))
            p.setFont(mono(10))
            for v in (3, 4, 5, 6, 8, 10, 12, 16):
                t = (math.log(v) - math.log(3)) / (math.log(16) - math.log(3))
                xx = x0 + t * w
                p.drawText(QRectF(xx - 14, yb + 10, 28, 12), Qt.AlignmentFlag.AlignCenter, f'{v}+' if v == 16 else f'{v}')
            mn = self.project.analysis.min_density
            tg = self.project.analysis.target_density
            res = self.result()
            ok = res.metrics.get('min_ok_pct', 0) if res else 0
            p.setFont(sans(10))
            yy = yb + 34
            pen = QPen(THEME.color('ink'), 1, Qt.PenStyle.DashLine)
            p.setPen(pen)
            p.drawLine(QPointF(x0, yy - 4), QPointF(x0 + 16, yy - 4))
            p.setPen(THEME.color('ink2'))
            p.drawText(QPointF(x0 + 20, yy), f'min {mn:.1f} · met on {ok:.1f} % of the ROI')
            p.setPen(QPen(THEME.color('ink'), 1))
            p.drawLine(QPointF(x0 + w * 0.62, yy - 4), QPointF(x0 + w * 0.62 + 16, yy - 4))
            p.setPen(THEME.color('ink2'))
            p.drawText(QPointF(x0 + w * 0.62 + 20, yy), f'target {tg:.1f}')
            yy += 16
            p.fillRect(QRectF(x0, yy - 9, 16, 9), QBrush(QColor(STATUS['critical']), Qt.BrushStyle.BDiagPattern))
            p.drawText(QPointF(x0 + 20, yy), 'blind in ROI')
            pen = QPen(THEME.color('ink'), 1, Qt.PenStyle.CustomDashLine)
            pen.setDashPattern([3, 3])
            p.setPen(pen)
            p.drawLine(QPointF(x0 + w * 0.42, yy - 4), QPointF(x0 + w * 0.42 + 16, yy - 4))
            p.setPen(THEME.color('ink2'))
            p.drawText(QPointF(x0 + w * 0.42 + 20, yy), 'ROI · seams')
        elif mode in ('redundancy', 'targets', 'compare'):
            items = [(REDUNDANCY[0], '1 camera'), (REDUNDANCY[1], '2'), (REDUNDANCY[2], '3+')]
            xx = x1 - 260
            for col, txt in items:
                p.fillRect(QRectF(xx, y + 3, 10, 10), QColor(col))
                p.setPen(THEME.color('ink2'))
                p.drawText(QPointF(xx + 14, y + 12), txt)
                xx += 14 + p.fontMetrics().horizontalAdvance(txt) + 12
            if mode == 'targets':
                yy = y + 22
                xx = x1 - 260
                p.setPen(QPen(QColor(STATUS['critical']), 1.2))
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawRect(QRectF(xx, yy, 10, 10))
                p.setPen(THEME.color('ink2'))
                p.drawText(QPointF(xx + 14, yy + 9), 'needs a 2nd camera')
                xx += 130
                pen = QPen(THEME.color('ink'), 1, Qt.PenStyle.CustomDashLine)
                pen.setDashPattern([2, 2])
                p.setPen(pen)
                p.drawRect(QRectF(xx, yy, 10, 10))
                p.setPen(THEME.color('ink2'))
                p.drawText(QPointF(xx + 14, yy + 9), 'suggested position')
        elif mode == 'stitched':
            self._legend_stitched(p, x1, y)
        elif mode == 'delta':
            w = 260
            x0 = x1 - w
            n = len(DIVERGING)
            for i, c in enumerate(DIVERGING):
                p.fillRect(QRectF(x0 + i * w / n, y + 4, w / n + 0.5, 6), QColor(c))

    def _legend_stitched(self, p, x1, y):
        sres = self._stitched()
        w = min(330, self.width() * 0.4)
        x0 = x1 - w
        p.setPen(Qt.PenStyle.NoPen)
        bg = THEME.color('panel')
        bg.setAlphaF(0.9)
        p.setBrush(bg)
        p.drawRoundedRect(QRectF(x0 - 8, y - 4, w + 14, 104), 4, 4)
        p.setFont(sans(10))
        p.setPen(THEME.color('ink2'))
        what = sres.settings.label if sres is not None else 'object'
        p.drawText(QRectF(x0, y, w, 14), Qt.AlignmentFlag.AlignLeft, f'share of a {what} the stitched view shows')
        yb = y + 18
        n = len(SEQ)
        for i, c in enumerate(SEQ):
            p.fillRect(QRectF(x0 + i * w / n, yb, w / n + 0.5, 8), QColor(c))
        p.setFont(mono(10))
        for v in (0, 25, 50, 75, 100):
            xx = x0 + v / 100 * w
            al = Qt.AlignmentFlag.AlignLeft if v == 0 else Qt.AlignmentFlag.AlignRight if v == 100 else Qt.AlignmentFlag.AlignCenter
            p.drawText(QRectF(xx - (0 if v == 0 else 36 if v == 100 else 18), yb + 10, 36, 12), al, f'{v} %')
        p.setFont(sans(10))
        rows = [[('vanish', 'vanishes at a seam'), ('cut', 'cut off: view edge or above the camera')],
                [('blind', 'no camera sees it'), ('hot', 'hotspot · m²')]]
        yy = yb + 38
        for row in rows:
            xx = x0
            for kind, txt in row:
                r = QRectF(xx, yy - 9, 14, 10)
                if kind == 'blind':
                    p.fillRect(r, QBrush(QColor(STATUS['critical']), Qt.BrushStyle.BDiagPattern))
                elif kind == 'vanish':
                    p.fillRect(r, QColor(STATUS['serious']))
                    p.fillRect(r, QBrush(QColor('#fafafa'), Qt.BrushStyle.BDiagPattern))
                elif kind == 'cut':
                    p.fillRect(r, QColor(SEQ[2]))
                    p.fillRect(r, QBrush(QColor('#181816'), Qt.BrushStyle.Dense6Pattern))
                else:
                    p.setPen(QPen(THEME.color('ink'), 1.2))
                    p.setBrush(Qt.BrushStyle.NoBrush)
                    p.drawEllipse(r.center(), 5, 5)
                p.setPen(THEME.color('ink2'))
                p.drawText(QPointF(xx + 19, yy), txt)
                xx += 19 + p.fontMetrics().horizontalAdvance(txt) + 14
            yy += 16
        p.setPen(THEME.color('ink2') if sres is not None else THEME.color('muted'))
        if sres is not None:
            m = sres.metrics
            txt = f'vanishes on {m["vanish_m2"]:.2f} m² · cut off on {m["cut_m2"]:.2f} m² · {m["shown_mean"] * 100:.0f} % shown on average'
        else:
            ctl = getattr(self.ctx, 'stitched', None)
            txt = ctl.status if ctl is not None and ctl.status else 'computing…'
        p.drawText(QPointF(x0, yy + 2), txt)

    # ------------------------------------------------------------------ interaction
    def _pick_camera(self, pos, r=9):
        best = None
        for c in self.cameras():
            if not c.visible:
                continue
            q = self.w2s(c.x, c.y)
            d = math.hypot(q.x() - pos.x(), q.y() - pos.y())
            if d < r and (best is None or d < best[1]):
                best = (c, d)
        return best[0] if best else None

    def _pick_target(self, w):
        for t in self.project.targets:
            tw, th = t.size_m
            ca, sa = math.cos(math.radians(t.yaw)), math.sin(math.radians(t.yaw))
            u = (w[0] - t.x) * ca + (w[1] - t.y) * sa              # into the board's own axes
            v = -(w[0] - t.x) * sa + (w[1] - t.y) * ca
            if abs(u) < tw / 2 and abs(v) < th / 2:
                return t
        return None

    def focusOutEvent(self, e):
        if self.drag and not self.drag.get('pan'):
            self.ctx.store.dragging = False
            self.ctx.analysis.request()
        self.drag = None
        super().focusOutEvent(e)

    def mousePressEvent(self, e):
        if self.drag is not None:
            return                      # another button during a drag: its release must not end someone else's gesture
        self._btn = e.button()
        self.setFocus()
        pos = e.position()
        self._press = pos
        self._last = pos
        w = self.s2w(pos.x(), pos.y())
        if e.button() in (Qt.MouseButton.MiddleButton, Qt.MouseButton.RightButton) or \
                (e.button() == Qt.MouseButton.LeftButton and e.modifiers() & Qt.KeyboardModifier.AltModifier):
            self.drag = {'pan': True, 'button': e.button()}
            return
        if e.button() != Qt.MouseButton.LeftButton or not self.interactive:
            self.drag = {'pan': True, 'button': e.button()}
            return
        res = self.result()
        st = self.ctx.store
        if self.ctx.tool == 'aim' and st.sel_cams and self.result_fn is None:
            for u in list(st.sel_cams):
                ops.aim_at(self.ctx, u, (float(w[0]), float(w[1]), 0.0))
            self.ctx.set_tool('select')
            return
        if self.edit_polygon:
            for i, (x, y) in enumerate(self.project.roi.polygon):
                q = self.w2s(x, y)
                if abs(q.x() - pos.x()) < 6 and abs(q.y() - pos.y()) < 6:
                    self.drag = {'vertex': i}
                    st.dragging = True
                    return
        if self.mode == 'bev' and res is not None:
            for i, s in enumerate(res.seams):
                a, b = self._seam_segment(res, s.angle)
                if b is not None:
                    q = self.w2s(*b)
                    if abs(q.x() - pos.x()) < 7 and abs(q.y() - pos.y()) < 7:
                        self.drag = {'seam': i, 'key': f'{s.a}|{s.b}'}
                        st.dragging = True
                        return
        c = self._pick_camera(pos)
        if self.mode == 'coverage' and self.result_fn is None and self.project.variant.sensors:
            from .sensorview import plan_pick
            s = plan_pick(self, pos)
            if s is not None and c is not None:          # a radar on the bumper under a camera: the nearer one wins
                qc, qs = self.w2s(c.x, c.y), self.w2s(s.x, s.y)
                if math.hypot(qc.x() - pos.x(), qc.y() - pos.y()) <= math.hypot(qs.x() - pos.x(), qs.y() - pos.y()):
                    s = None
            if s is not None:
                st.select_obj(('sensor', s.uid))
                if not s.locked and self.ctx.tool in ('select', 'move'):
                    self.drag = {'sensor': s.uid, 'start': (s.x, s.y), 'w0': w}
                    st.dragging = True
                return
        if c is not None and self.result_fn is None:
            add = bool(e.modifiers() & (Qt.KeyboardModifier.ShiftModifier | Qt.KeyboardModifier.ControlModifier))
            st.select_cams([c.uid], add=add)
            if not c.locked and not add and self.ctx.tool in ('select', 'move'):
                self.drag = {'cam': c.uid, 'start': np.array([c.x, c.y, c.z]), 'w0': w}
                st.dragging = True
            return
        if self.mode == 'targets' or self.ctx.overlays.get('targets'):
            t = self._pick_target(w)
            if t is not None:
                st.select_obj(('target', t.uid))
                self.drag = {'target': t.uid, 'start': (t.x, t.y), 'w0': w}
                st.dragging = True
                return
        self.drag = {'pan': True, 'button': e.button(), 'click': True}

    def mouseMoveEvent(self, e):
        pos = e.position()
        w = self.s2w(pos.x(), pos.y())
        st = self.ctx.store
        d = self.drag
        fine = 0.1 if e.modifiers() & Qt.KeyboardModifier.ShiftModifier else 1.0
        if d and d.get('pan'):
            dl = pos - self._last
            if (pos - self._press).manhattanLength() > 3:
                d['moved'] = True
                self.centre = self.centre - np.array([dl.x(), -dl.y()]) / self.scale
                self.update()
                self.viewChanged.emit()
        elif d and 'cam' in d:
            c = st.cam(d['cam'])
            if c:
                dw = (w - d['w0']) * fine
                nx, ny = d['start'][0] + dw[0], d['start'][1] + dw[1]
                nz = d['start'][2]
                if self.ctx.snap == 'grid':
                    nx, ny = round(nx / 0.05) * 0.05, round(ny / 0.05) * 0.05
                elif self.ctx.snap == 'surface':
                    vg = self.ctx.analysis.engine.vehicle_geom(self.project)
                    if vg is not None:
                        q, n, _, _, _ = surface_frame(vg.occ_mesh, np.array([nx, ny, nz]))
                        pp = q + n * c.standoff_mm / 1000
                        nx, ny, nz = pp
                st.set_cam(c.uid, merge=True, x=round(float(nx), 4), y=round(float(ny), 4), z=round(float(nz), 4))
        elif d and 'sensor' in d:
            dw = (w - d['w0']) * fine
            nx, ny = d['start'][0] + dw[0], d['start'][1] + dw[1]
            if self.ctx.snap == 'grid':
                nx, ny = round(nx / 0.05) * 0.05, round(ny / 0.05) * 0.05
            s = st.variant.sensor(d['sensor'])
            if s is not None:
                from .sensorview import set_props
                set_props(self.ctx, s.uid, f'Moved {s.tag}', merge=True, x=round(float(nx), 4), y=round(float(ny), 4))
        elif d and 'target' in d:
            dw = (w - d['w0']) * fine
            nx, ny = d['start'][0] + dw[0], d['start'][1] + dw[1]
            if self.ctx.snap == 'grid' or self.mode == 'targets':
                nx, ny = round(nx / 0.05) * 0.05, round(ny / 0.05) * 0.05
            uid = d['target']

            def fn(pj, uid=uid, nx=nx, ny=ny):
                for t in pj.targets:
                    if t.uid == uid:
                        t.x, t.y = round(float(nx), 3), round(float(ny), 3)
            st.edit('targets', fn, 'Moved target', merge_key=('target', uid))
        elif d and 'seam' in d:
            res = self.result()
            cx, cy = res.centre
            ang = math.degrees(math.atan2(w[1] - cy, w[0] - cx))
            key = d['key']

            def fn(pj, key=key, ang=ang):
                pj.variant.seams[key] = round(ang, 1)
            st.edit('variant', fn, 'Moved seam', merge_key=('seam', key))
            self.seamMoved.emit()
        elif d and 'vertex' in d:
            i = d['vertex']

            def fn(pj, i=i, w=w):
                poly = list(pj.roi.polygon)
                poly[i] = (round(float(w[0]), 3), round(float(w[1]), 3))
                pj.roi.polygon = poly
            st.edit('roi', fn, 'Edited ROI polygon', merge_key=('roi-vertex', i))
        else:
            self._hover(pos, w)
        self._last = pos

    def _hover(self, pos, w):
        res = self.result()
        self.ctx.cursor_world.emit((w[0], w[1], 0.0))
        info = None
        if res is not None and self.show_tooltip and self._pick_camera(pos) is None:
            q = res.query(w[0], w[1])
            if q and q['in_roi']:
                rows = []
                cams = {c.uid: c for c in self.cameras()}
                from ..core.analysis import camera_frames
                for ce, dens in q['cams'][:4]:
                    cam = cams.get(ce.uid)
                    if cam is None:
                        continue
                    dv = np.array([w[0], w[1], 0.0]) - cam.pos()
                    R, W2O = camera_frames(cam)
                    o = W2O @ dv
                    rows.append((cam, dens, float(np.linalg.norm(dv)), math.degrees(math.atan2(math.hypot(o[0], o[1]), o[2]))))
                info = dict(q, point=(w[0], w[1]), extra=rows, pos=pos)
        self._smears = None
        if info is not None and self.mode == 'stitched':
            sres = self._stitched()
            at = sres.at(w[0], w[1]) if sres is not None else None
            if at is not None:
                info['stitched'] = (at, sres, self.cameras())
                from ..core.stitched_objects import smears
                self._smears = smears(self.project, self.ctx.analysis.engine, res, w[0], w[1])
        had = self.hover is not None
        self.hover = info
        if info or had:
            self.update()
        self.setCursor(Qt.CursorShape.CrossCursor if info else Qt.CursorShape.ArrowCursor)

    def mouseReleaseEvent(self, e):
        if self.drag is not None and e.button() != getattr(self, '_btn', e.button()):
            return
        d = self.drag
        self.drag = None
        st = self.ctx.store
        if d and d.get('pan') and d.get('moved'):
            self._user_view = True
        if d and not d.get('pan'):
            st.dragging = False
            self.ctx.analysis.request()
            self.update()
            return
        if d and d.get('pan') and not d.get('moved'):
            if d.get('button') == Qt.MouseButton.RightButton:
                self._context(e)
            elif d.get('click') and self.interactive and self.result_fn is None:
                if self.hover and self.hover.get('extra'):
                    st.select_cams([self.hover['extra'][0][0].uid])
                else:
                    st.select_cams([])
        self.update()

    def mouseDoubleClickEvent(self, e):
        if self.edit_polygon:
            w = self.s2w(e.position().x(), e.position().y())

            def fn(pj):
                pj.roi.polygon = list(pj.roi.polygon) + [(round(float(w[0]), 3), round(float(w[1]), 3))]
            self.ctx.store.edit('roi', fn, 'Added ROI vertex')
        elif self.ctx.window:
            self.ctx.window.frame_selection()

    def _context(self, e):
        if not self.ctx.window or self.result_fn is not None:
            return
        pos = e.position()
        c = self._pick_camera(pos)
        gpos = e.globalPosition().toPoint()
        if c is not None:
            if c.uid not in self.ctx.store.sel_cams:
                self.ctx.store.select_cams([c.uid])
            self.ctx.window.camera_menu(self.ctx.store.sel_cams, gpos)
            return
        if self.edit_polygon:
            for i, (x, y) in enumerate(self.project.roi.polygon):
                q = self.w2s(x, y)
                if abs(q.x() - pos.x()) < 6 and abs(q.y() - pos.y()) < 6 and len(self.project.roi.polygon) > 3:
                    def fn(pj, i=i):
                        poly = list(pj.roi.polygon)
                        poly.pop(i)
                        pj.roi.polygon = poly
                    self.ctx.store.edit('roi', fn, 'Removed ROI vertex')
                    return
        w = self.s2w(pos.x(), pos.y())
        self.ctx.window.ground_menu(np.array([w[0], w[1], 0.0]), gpos)

    def wheelEvent(self, e):
        dy = e.angleDelta().y()
        if dy == 0:                     # horizontal scrolling is not a zoom
            return
        k = 1.12 ** (dy / 120.0)        # touchpads send many small steps
        self._user_view = True
        pos = e.position()
        w0 = self.s2w(pos.x(), pos.y())
        self.scale = max(5, min(2000, self.scale * k))
        w1 = self.s2w(pos.x(), pos.y())
        self.centre = self.centre + (w0 - w1)
        self.update()
        self.viewChanged.emit()

    def leaveEvent(self, e):
        if self.hover is not None:
            self.hover = None
            self.update()
        self.ctx.cursor_world.emit(None)

    def keyPressEvent(self, e):
        if e.key() in (Qt.Key.Key_Home, Qt.Key.Key_A):
            self.fit()
        elif e.key() == Qt.Key.Key_Escape and self.edit_polygon:
            self.edit_polygon = False
            self.update()
        elif e.key() == Qt.Key.Key_Escape and self.ctx.tool == 'aim':
            self.ctx.set_tool('select')
        else:
            super().keyPressEvent(e)

    def grab_image(self):
        return self.grab().toImage()
