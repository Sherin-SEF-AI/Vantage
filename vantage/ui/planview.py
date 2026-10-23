"""Top-down plan views drawn with QPainter: coverage, density, redundancy, footprints, stitched
bird's-eye, calibration targets and variant comparison.

Over the heatmap raster (ui.maps) the view draws vector layers that stay crisp at any zoom: iso-lines at the minimum
and target density (labelled on the map), screen-space hatches for blind ground and ground below the minimum density,
the ground only one camera sees when its bar is hovered in the Coverage panel, a legend that moves to the corner
where it hides the least data, and a hover read-out with a crosshair (the value under the cursor, the cameras that see
the point and their density, range and off-axis angle)."""
from __future__ import annotations

import math
import weakref

import numpy as np
from PyQt6.QtCore import Qt, QPointF, QRectF, pyqtSignal, QLineF
from PyQt6.QtGui import (QPainter, QColor, QPen, QImage, QPolygonF, QBrush, QFont, QTransform, QPainterPath,
                         QFontMetricsF, QLinearGradient, QPixmap)
from PyQt6.QtWidgets import QWidget

from ..core.analysis import surface_frame
from ..core.lens import resolve
from ..core.mesh import ROLE_GLASS, ROLE_WHEELS, ROLE_IGNORED
from . import colormaps, guides, maps, ops
from .charts import CAMERA_HOVER
from .theme import THEME, mono, sans, STATUS
from .scene import paint_ground_tooltip
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


def _refs(*objs):
    """Weak references to the data a cache was built from (a freed result's successor can reuse its id(), and a
    strong reference would keep an old result and its occlusion cubes alive in a hidden view)."""
    out = []
    for o in objs:
        try:
            out.append(None if o is None else weakref.ref(o))
        except TypeError:
            out.append(lambda o=o: o)
    return tuple(out)


def _same(objs, refs):
    return len(objs) == len(refs) and all((o is None and r is None) or (r is not None and r() is o)
                                          for o, r in zip(objs, refs))


# ---------------------------------------------------------------------- legend
def _sw_fill(hexs):
    def f(p, r):
        p.fillRect(r, QColor(hexs))
    return f


def _sw_hatch(fill, line, pattern):
    def f(p, r):
        if fill is not None:
            p.fillRect(r, fill)
        p.fillRect(r, QBrush(line, pattern))
    return f


def _sw_box(col, dashed):
    def f(p, r):
        pen = QPen(col, 1.2)
        if dashed:
            pen.setStyle(Qt.PenStyle.CustomDashLine)
            pen.setDashPattern([2, 2])
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRect(r.adjusted(1, 0, -1, 0))
    return f


def _sw_dash():
    def f(p, r):
        pen = QPen(THEME.color('ink', 0.85), 1.2, Qt.PenStyle.CustomDashLine)
        pen.setDashPattern([4, 3])
        p.setPen(pen)
        p.drawLine(QPointF(r.left(), r.center().y()), QPointF(r.right() + 2, r.center().y()))
    return f


def _sw_vanish():
    def f(p, r):
        p.fillRect(r, QColor(STATUS['serious']))
        p.fillRect(r, QBrush(QColor('#fafafa'), Qt.BrushStyle.BDiagPattern))
    return f


def _sw_cut():
    def f(p, r):
        p.fillRect(r, QColor(maps.stitched_palette()[4]))
        p.fillRect(r, QBrush(THEME.color('ink'), Qt.BrushStyle.Dense6Pattern))
    return f


def _sw_ring():
    def f(p, r):
        p.setPen(QPen(THEME.color('ink'), 1.2))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(r.center(), 5, 5)
    return f


class _Legend:
    """A plan-view legend card: title, unit line, an optional colour bar (density / change / share shown), swatch
    rows, camera chips and a footer of figures. Measured first so the view can place it where it hides no data."""
    PAD, ROW, FOOT = 8, 16, 14

    def __init__(self, title, sub=''):
        self.title, self.sub = title, sub
        self.bar = None             # ('density', DensityScale, height) | ('diverging', span, height) | ('share', _, _)
        self.rows = []              # (swatch painter(p, QRectF), text)
        self.chips = []             # (slot, tag)
        self.foot = []
        self.size = (0, 0)

    def _bar_labels(self):
        kind, a, hb = self.bar
        if kind == 'density':
            sc = a
            marks = [(0.0, f'{sc.lo:g} min', True)]
            if sc.lo < sc.target < sc.hi:
                marks.append((float(sc.t(sc.target)), f'{sc.target:g} target', True))
            ticks = []
            for v in sc.ticks()[1:][::-1]:           # the top first, then down; keep labels 12 px apart
                t = float(sc.t(v))
                if all(abs(t - m[0]) * hb >= 12 for m in marks + ticks):
                    ticks.append((t, f'{v:g}+' if v == sc.hi else f'{v:g}', False))
            return marks + ticks
        if kind == 'diverging':
            return [(1.0, f'+{a:g} better', True), (0.5, '0 same', False), (0.0, f'−{a:g} worse', True)]
        return []

    def measure(self):
        fs, fb, fm = QFontMetricsF(sans(10)), QFontMetricsF(sans(11, QFont.Weight.DemiBold)), QFontMetricsF(mono(10))
        w = max(fb.horizontalAdvance(self.title), fs.horizontalAdvance(self.sub))
        h = self.PAD + 16 + (13 if self.sub else 0) + 6
        if self.bar is not None:
            kind, _, hb = self.bar
            if kind == 'share':
                w = max(w, 150)
                h += 30
            else:
                lab = max((fm.horizontalAdvance(t) for _, t, _ in self._bar_labels()), default=0)
                w = max(w, 10 + 8 + lab)
                h += hb + 12
        for _, text in self.rows:
            w = max(w, 20 + fs.horizontalAdvance(text))
        h += self.ROW * len(self.rows)
        if self.chips:
            cw = max(fm.horizontalAdvance(t) for _, t in self.chips) + 22
            ncol = max(1, min(len(self.chips), int(max(w, 3 * cw) // cw)))
            w = max(w, ncol * cw)
            self._chip = (cw, ncol)
            h += self.ROW * math.ceil(len(self.chips) / ncol)
        if self.foot:
            w = max(w, max(fs.horizontalAdvance(t) for t in self.foot))
            h += 4 + self.FOOT * len(self.foot)
        self.size = (math.ceil(w) + 2 * self.PAD, math.ceil(h) + self.PAD - 4)
        return self.size

    def paint(self, p, x, y, over_data=False):
        W, H = self.size
        P = self.PAD
        p.save()
        p.setPen(QPen(THEME.color('hair' if not over_data else 'strong'), 1))
        p.setBrush(THEME.color('viewport', 0.9))
        p.drawRoundedRect(QRectF(x + 0.5, y + 0.5, W - 1, H - 1), 4, 4)
        yy = y + P
        p.setFont(sans(11, QFont.Weight.DemiBold))
        p.setPen(THEME.color('ink'))
        p.drawText(QRectF(x + P, yy, W - 2 * P, 16), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, self.title)
        yy += 16
        if self.sub:
            p.setFont(sans(10))
            p.setPen(THEME.color('muted'))
            p.drawText(QRectF(x + P, yy, W - 2 * P, 13), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, self.sub)
            yy += 13
        yy += 6
        if self.bar is not None:
            yy = self._paint_bar(p, x + P, yy, W - 2 * P)
        if self.chips:
            cw, ncol = self._chip
            for k, (slot, tag) in enumerate(self.chips):
                cx = x + P + (k % ncol) * cw
                cy = yy + (k // ncol) * self.ROW + self.ROW / 2
                paint_chip(p, cx + 1, cy, slot, tag, size=10, font=mono(10, QFont.Weight.DemiBold))
            yy += self.ROW * math.ceil(len(self.chips) / ncol)
        p.setFont(sans(10))
        for sw, text in self.rows:
            sw(p, QRectF(x + P, yy + 3, 12, 10))
            p.setPen(THEME.color('ink2'))
            p.drawText(QRectF(x + P + 18, yy, W - 2 * P - 18, self.ROW), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, text)
            yy += self.ROW
        if self.foot:
            yy += 4
            p.setFont(sans(10))
            p.setPen(THEME.color('ink2'))
            for t in self.foot:
                p.drawText(QRectF(x + P, yy, W - 2 * P, self.FOOT), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, t)
                yy += self.FOOT
        p.restore()

    def _paint_bar(self, p, x, y, w):
        kind, a, hb = self.bar
        if kind == 'share':
            pal = maps.stitched_palette()
            g = QLinearGradient(x, 0, x + w, 0)
            for i, c in enumerate(pal):
                g.setColorAt(i / (len(pal) - 1), QColor(c))
            p.fillRect(QRectF(x, y, w, 8), QBrush(g))
            p.setFont(mono(10))
            p.setPen(THEME.color('muted'))
            for v, al in ((0, Qt.AlignmentFlag.AlignLeft), (50, Qt.AlignmentFlag.AlignHCenter), (100, Qt.AlignmentFlag.AlignRight)):
                p.drawText(QRectF(x + v / 100 * w - (0 if v == 0 else 40 if v == 100 else 20), y + 10, 40, 12), al, f'{v} %')
            return y + 30
        bw = 10
        y0 = y + 4
        g = QLinearGradient(0, y0 + hb, 0, y0)
        if kind == 'density':
            for i in range(17):
                t = i / 16
                g.setColorAt(t, QColor(*(int(c) for c in a.lut[int(round(t * (len(a.lut) - 1)))])))
        else:
            lut = colormaps.diverging()
            for i in range(17):
                t = i / 16
                g.setColorAt(t, QColor(*(int(c) for c in lut[int(round(t * (len(lut) - 1)))])))
        p.fillRect(QRectF(x, y0, bw, hb), QBrush(g))
        p.setFont(mono(10))
        fm = p.fontMetrics()
        for t, text, strong in self._bar_labels():
            yy = y0 + (1 - t) * hb
            if strong:
                p.setPen(QPen(THEME.color('viewport', 0.85), 3.2))
                p.drawLine(QPointF(x - 2, yy), QPointF(x + bw + 2, yy))
                pen = QPen(THEME.color('ink'), 1.4)
                if kind == 'density' and t == 0.0:
                    pen = QPen(THEME.color('ink', 0.85), 1.1, Qt.PenStyle.CustomDashLine)
                    pen.setDashPattern([3, 2])
                p.setPen(pen)
                p.drawLine(QPointF(x - 2, yy), QPointF(x + bw + 2, yy))
                p.setPen(THEME.color('ink'))
            else:
                p.setPen(QPen(THEME.color('muted'), 1))
                p.drawLine(QPointF(x + bw, yy), QPointF(x + bw + 3, yy))
                p.setPen(THEME.color('muted'))
            p.drawText(QPointF(x + bw + 8, yy + fm.ascent() / 2 - 1), text)
        return y0 + hb + 8


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
        self._ov_key = self._ov = None      # cached vector layers (world-space paths) of the current result
        self._layer_key = self._layer = None    # raster + vector layers drawn for the current view
        self._layer_refs = self._img_refs = ()
        self._ov_res = ()
        self._uniq = ((), None, None)       # (result ref, uid, path): the hovered camera's only-seen-by-me ground
        self.hl_cam = ''                    # camera hovered here or in a chart (its bar, its ground)
        self.legend_rect = None             # where the legend was drawn (screen), for tests and hit checks
        self._legend_place = self._legend_obj = None
        self._legend_over = False
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
        CAMERA_HOVER.changed.connect(self._on_cam_hover)

    def _on_cam_hover(self, uid):
        if uid != self.hl_cam:
            self.hl_cam = uid
            self.update()

    def _on_change(self, *_):
        self.update()

    def _on_theme(self, *_):
        self._img_key = None
        self._ov_key = self._layer_key = None
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
        if self.hover is not None and self.drag is None:     # the read-out follows the new result
            pos = self.hover['pos']
            self._hover(pos, self.s2w(pos.x(), pos.y()))
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
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        w, h = max(self.width(), 50), max(self.height(), 50)
        dx, dy = max(x1 - x0, 0.1), max(y1 - y0, 0.1)
        s = min((w - 24) / dx, (h - 40) / dy) / margin
        leg = self._legend_spec(res) if self.legend and res is not None else None
        if leg is not None:
            # keep the legend's corner free of data: give up width on the right or height at the top, whichever
            # costs less zoom
            lw, lh = leg.size
            s_r = min((w - 24 - lw - 14) / dx, (h - 40) / dy) / margin
            s_t = min((w - 24) / dx, (h - 40 - lh - 10) / dy) / margin
            if max(s_r, s_t) > s * 0.55:
                if s_r >= s_t:
                    s, cx = s_r, cx + (lw + 14) / 2 / s_r
                else:
                    s, cy = s_t, cy + (lh + 10) / 2 / s_t
        self.centre = np.array([cx, cy])
        self.scale = s
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
        self._legend_place = None
        if res is not None:
            self._legend_place = self._place_legend(res) if self.legend else None
            self._heat(p, res)
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
        if self.hover and self.show_tooltip and self.drag is None:
            self._crosshair(p, res)
        self._hud(p, res)
        if self.hover and self.show_tooltip and self.drag is None:
            if self.hover.get('stitched') is not None:
                paint_ground_tooltip(p, self.hover['pos'], self.hover, self.project, self.cameras())
            else:
                self._readout(p)
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

    def _heat(self, p, res):
        """The heatmap raster and its vector layers, drawn into a pixmap once per view (pan, zoom, result, theme,
        thresholds, highlighted camera): a hover only blits it."""
        mode = self._map_mode()
        dpr = self.devicePixelRatioF()
        lp = self._legend_place
        # the data objects themselves (not their ids, which a freed result's successor can reuse)
        refs = (res, self._stitched() if mode == 'stitched' else None,
                *(self.delta_fn() if self.mode == 'delta' and self.delta_fn else (None, None)))
        key = (mode, self.mode, THEME.name, frozenset(self.ctx.isolate) if self.ctx.isolate else None,
               self._thresholds(), self.hl_cam, (lp.x(), lp.y(), lp.width(), lp.height()) if lp is not None else None,
               float(self.centre[0]), float(self.centre[1]), self.scale, self.width(), self.height(), dpr)
        if key != self._layer_key or not _same(refs, self._layer_refs):
            pm = QPixmap(max(1, round(self.width() * dpr)), max(1, round(self.height() * dpr)))
            pm.setDevicePixelRatio(dpr)
            pm.fill(Qt.GlobalColor.transparent)
            q = QPainter(pm)
            q.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            self._map(q, res)
            self._overlays(q, res)
            q.end()
            self._layer, self._layer_key, self._layer_refs = pm, key, _refs(*refs)
        p.drawPixmap(0, 0, self._layer)

    def _map(self, p, res):
        mode = self._map_mode()
        if mode == 'stitched':
            sres = self._stitched()
            if sres is None:
                return
            key = ('stitched', THEME.name)
            if key != self._img_key or not _same((sres,), self._img_refs):
                self._img = maps.to_qimage(maps.stitched(sres, k=2))
                self._img_key, self._img_refs = key, _refs(sres)
            x0, y0, x1, y1 = maps.stitched_extent(sres)
            p.drawImage(QRectF(self.w2s(x0, y1), self.w2s(x1, y0)), self._img)
            return
        if self.mode == 'delta' and self.delta_fn:
            a, b = self.delta_fn()
            key = ('delta', THEME.name)
            if key != self._img_key or not _same((a, b), self._img_refs):
                img = maps.delta(a, b)
                img = np.clip(img, 0, 255)
                img[..., 3] = np.clip(img[..., 3] * 255, 0, 255)
                self._img = maps.to_qimage(img.astype(np.uint8))
                self._img_key, self._img_refs = key, _refs(a, b)
        else:
            thr = self._thresholds() if mode == 'density' else None
            key = (mode, THEME.name, frozenset(self.ctx.isolate) if self.ctx.isolate else None, thr)
            if key != self._img_key or not _same((res,), self._img_refs):
                # density is continuous: 4 texels per cell, interpolated; categorical maps keep crisp cells.
                # Blind ground is hatched as a vector layer (_overlays), crisp at any zoom.
                arr = maps.make(res, mode, k=4 if mode == 'density' else 2, isolate=self.ctx.isolate or None,
                                shadow=False, blind=False, thresholds=thr)
                self._img = maps.to_qimage(arr)
                self._img_key, self._img_refs = key, _refs(res)
        x0, y0, x1, y1 = maps.extent(res)
        a, b = self.w2s(x0, y1), self.w2s(x1, y0)
        p.save()
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, mode in ('density', 'footprints', 'bev', 'blank'))
        p.drawImage(QRectF(a, b), self._img)
        p.restore()

    def _thresholds(self):
        a = self.project.analysis
        return (float(a.min_density), float(a.target_density))

    # ------------------------------------------------------------------ vector layers over the raster
    def _world_tf(self):
        """World (x fwd, y left, metres) -> screen."""
        s = self.scale
        return QTransform(s, 0, 0, -s, self.width() / 2 - self.centre[0] * s, self.height() / 2 + self.centre[1] * s)

    @staticmethod
    def _path(lines, closed=True):
        path = QPainterPath()
        path.setFillRule(Qt.FillRule.OddEvenFill)
        for ln in lines:
            if len(ln) < 2:
                continue
            path.moveTo(float(ln[0, 0]), float(ln[0, 1]))
            for x, y in ln[1:]:
                path.lineTo(float(x), float(y))
            if closed:
                path.closeSubpath()
        return path

    def _vector_layers(self, res, mode):
        """World-space paths for the current result (cached): blind ground, ground below the minimum density and the
        density iso-lines (density mode)."""
        thr = self._thresholds() if mode == 'density' else None
        key = (mode, thr)
        if key != self._ov_key or not _same((res,), self._ov_res):
            ov = {'blind': self._path(maps.blind_outline(res)), 'below': None, 'iso': {}}
            if thr is not None:
                mn, tg = thr
                below = maps.below_min_outline(res, mn)
                if below:
                    ov['below'] = (self._path(below), self._path(maps.valid_outline(res)))
                for lv, lines in maps.density_contours(res, sorted({mn, tg})).items():
                    longest = max(lines, key=lambda l: float(np.linalg.norm(np.diff(l, axis=0), axis=1).sum()))\
                        if lines else None
                    ov['iso'][lv] = (self._path(lines, closed=False), longest)
            self._ov, self._ov_key, self._ov_res = ov, key, _refs(res)
        return self._ov

    def _overlays(self, p, res):
        mode = self._map_mode()
        if mode == 'stitched' or self.mode == 'delta':
            return
        ov = self._vector_layers(res, mode)
        T = self._world_tf()
        p.save()
        p.setPen(Qt.PenStyle.NoPen)
        th = 'light' if THEME.name == 'light' else 'dark'
        if ov['below'] is not None:                 # hatch the grey, clipped to the seen ROI ground
            below, valid = ov['below']
            p.save()
            p.setClipPath(T.map(valid))
            p.setBrush(QBrush(QColor(colormaps.HATCH[th]), Qt.BrushStyle.FDiagPattern))
            p.drawPath(T.map(below))
            p.restore()
        if not ov['blind'].isEmpty():
            bp = T.map(ov['blind'])
            p.setBrush(THEME.color('critical', 0.30))
            p.drawPath(bp)
            p.setBrush(QBrush(QColor(STATUS['critical']), Qt.BrushStyle.BDiagPattern))
            p.drawPath(bp)
        self._unique_ground(p, res, T)
        if ov['iso']:
            mn, tg = self._thresholds()
            halo = THEME.color('viewport', 0.8)
            for lv, (path, longest) in ov['iso'].items():
                sp = T.map(path)
                strong = lv == tg
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.setPen(QPen(halo, 3.6 if strong else 3.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap,
                              Qt.PenJoinStyle.RoundJoin))
                p.drawPath(sp)
                pen = QPen(THEME.color('ink', 0.95 if strong else 0.8), 1.6 if strong else 1.1, Qt.PenStyle.SolidLine,
                           Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
                if not strong:
                    pen.setStyle(Qt.PenStyle.CustomDashLine)
                    pen.setDashPattern([4, 3])
                p.setPen(pen)
                p.drawPath(sp)
            placed = [] if self._legend_place is None else [self._legend_place.adjusted(-6, -6, 6, 6)]
            cams = [self.w2s(c.x, c.y) for c in self.cameras() if c.visible]
            for lv in sorted(ov['iso'], key=lambda v: v != tg):          # the target's label first
                longest = ov['iso'][lv][1]
                if longest is not None:
                    self._iso_label(p, longest, f'{lv:g} px/cm · ' + ('target' if lv == tg else 'min'), placed, cams)
        p.restore()

    def _iso_label(self, p, line, text, placed, cams):
        """Name an iso-line once: at the point of its longest run nearest the run's middle where the label stays in
        the view, clear of the legend, of labels already placed and of the camera markers."""
        q = self.w2s_np(line)
        W, H = self.width(), self.height()
        p.setFont(mono(10))
        tw = p.fontMetrics().horizontalAdvance(text) + 8
        ok = (q[:, 0] > tw / 2 + 4) & (q[:, 0] < W - tw / 2 - 4) & (q[:, 1] > 46) & (q[:, 1] < H - 40)
        for c in cams:
            ok &= np.hypot(q[:, 0] - c.x(), q[:, 1] - c.y()) > 22 + tw / 2
        idx = np.nonzero(ok)[0]
        if not len(idx):
            return
        box = None
        for i in idx[np.argsort(np.abs(idx - len(q) // 2), kind='stable')][::max(1, len(idx) // 60)]:
            b = QRectF(q[i, 0] - tw / 2, q[i, 1] - 8, tw, 15)
            if not any(b.intersects(r) for r in placed):
                box = b
                break
        if box is None:
            return
        placed.append(box.adjusted(-4, -3, 4, 3))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(THEME.color('viewport', 0.88))
        p.drawRoundedRect(box, 3, 3)
        p.setPen(THEME.color('ink'))
        p.drawText(box, Qt.AlignmentFlag.AlignCenter, text)

    def _unique_ground(self, p, res, T):
        """The ground only the highlighted camera sees (what its bar in the Coverage panel measures)."""
        uid = self.hl_cam
        if not uid or res.cam_index(uid) < 0:
            return
        if not _same((res,), self._uniq[0]) or self._uniq[1] != uid:
            self._uniq = (_refs(res), uid, self._path(maps.unique_outline(res, uid)))
        path = self._uniq[2]
        slot = res.cams[res.cam_index(uid)].slot
        sp = T.map(path)
        p.setBrush(THEME.cam(slot, 0.38))
        p.setPen(QPen(THEME.cam(slot), 1.4))
        p.drawPath(sp)
        p.setPen(Qt.PenStyle.NoPen)

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
                    sel = c.uid in self.ctx.store.sel_cams or c.uid == self.hl_cam
                    p.setPen(QPen(THEME.cam(c.slot), 1.8 if sel else 1.0))
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
            if c.uid == self.hl_cam:            # hovered here or its bar in a chart
                p.setPen(QPen(col, 1.6))
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawEllipse(q, s / 2 + 7, s / 2 + 7)

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
        lr = self._legend_place
        if lr is not None and self._legend_obj is not None:
            self._legend_obj.paint(p, lr.x(), lr.y(), self._legend_over)
        self.legend_rect = lr

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

    # ------------------------------------------------------------------ legend
    def _legend_spec(self, res):
        """What the legend shows for the current mode (a _Legend, measured but not placed), or None."""
        mode = self.mode
        if res is None:
            return None
        compact = self.height() < 420 or self.width() < 640
        hb = int(min(max(self.height() * 0.26, 56), 130)) if not compact else 64
        crit = QColor(STATUS['critical'])
        blind = (_sw_hatch(THEME.color('critical', 0.3), crit, Qt.BrushStyle.BDiagPattern), 'blind · no camera')
        m = res.metrics
        if mode == 'density':
            mn, tg = self._thresholds()
            sc = colormaps.density_scale(mn, tg)
            L = _Legend('Pixel density', 'px/cm · best camera · log scale')
            L.bar = ('density', sc, hb)
            L.rows = [(_sw_hatch(QColor(sc.below_hex), QColor(sc.hatch_hex), Qt.BrushStyle.FDiagPattern),
                       f'below the minimum {mn:g}'), blind]
            L.foot = [f'≥ minimum on {m.get("min_ok_pct", 0):.1f} % of the ROI',
                      f'≥ target on {m.get("target_ok_pct", 0):.1f} %']
        elif mode in ('redundancy', 'targets', 'compare'):
            c1, c2, c3 = colormaps.redundancy()
            L = _Legend('Redundancy', 'cameras that see the ground')
            L.rows = [(_sw_fill(c3), '3 or more'), (_sw_fill(c2), '2'), (_sw_fill(c1), '1'),
                      (blind[0], '0 · blind')]
            if mode == 'targets':
                L.rows += [(_sw_box(crit, False), 'needs a 2nd camera'), (_sw_box(THEME.color('ink'), True), 'suggested position')]
            L.foot = [f'≥ 2 cameras on {m.get("multi_pct", 0):.1f} % of the ROI']
        elif mode in ('coverage', 'footprints', 'bev'):
            iso = self.ctx.isolate if mode == 'footprints' else None
            L = _Legend(*{'coverage': ('Best camera', 'most px/cm at each point'),
                          'footprints': ('Footprints', 'ground each camera sees'),
                          'bev': ('Stitched view', 'camera shown in each sector')}[mode])
            L.chips = [(c.slot, c.tag) for c in res.cams if not iso or c.uid in iso]
            L.rows = [blind]
            if mode == 'bev':
                L.rows.insert(0, (_sw_hatch(None, QColor(STATUS['warning']), Qt.BrushStyle.BDiagPattern),
                                  'its camera cannot see it'))
            if not L.chips:
                return None if mode != 'bev' else L
        elif mode == 'delta':
            L = _Legend('Density change', 'px/cm · selected − current')
            L.bar = ('diverging', 4.0, hb)
        elif mode == 'stitched':
            sres = self._stitched()
            what = sres.settings.label if sres is not None else 'object'
            L = _Legend(f'Share of a {what} shown', 'in the stitched view')
            L.bar = ('share', None, 0)
            L.rows = [(_sw_vanish(), 'vanishes at a seam'), (_sw_cut(), 'cut off: view edge / above'),
                      (_sw_hatch(THEME.color('critical', 0.3), crit, Qt.BrushStyle.BDiagPattern), 'no camera sees it'),
                      (_sw_ring(), 'hotspot · m²')]
            if sres is not None:
                sm = sres.metrics
                L.foot = [f'vanishes on {sm["vanish_m2"]:.2f} m²', f'cut off on {sm["cut_m2"]:.2f} m²',
                          f'{sm["shown_mean"] * 100:.0f} % shown on average']
            else:
                ctl = getattr(self.ctx, 'stitched', None)
                L.foot = [ctl.status if ctl is not None and ctl.status else 'computing…']
        else:
            return None
        if compact:
            L.foot = []
        elif mode not in ('delta', 'stitched'):
            L.rows.append((_sw_dash(), 'ROI edge · seams'))
        L.measure()
        return L

    def _data_rect(self, res):
        """Screen rectangle around the ROI and the body (the data the legend should not cover)."""
        if not _same((res,), getattr(self, '_bbox_res', ())):
            g = res.grid
            jj, ii = np.nonzero(g.roi | g.body)
            self._bbox = (g.x0 + ii.min() * g.cell, g.y0 + jj.min() * g.cell,
                          g.x0 + (ii.max() + 1) * g.cell, g.y0 + (jj.max() + 1) * g.cell) if len(ii) else None
            self._bbox_res = _refs(res)
        if self._bbox is None:
            return QRectF()
        x0, y0, x1, y1 = self._bbox
        return QRectF(self.w2s(x0, y1), self.w2s(x1, y0))

    def _place_legend(self, res):
        """The legend's screen rectangle: the corner (top right, bottom right, top left, bottom left) where it covers
        the least of the ROI. None when the view is too small for it."""
        L = self._legend_spec(res)
        self._legend_obj = L
        if L is None:
            return None
        lw, lh = L.size
        W, H = self.width(), self.height()
        if lw > W * 0.6 or lh > H - 60:
            return None
        data = self._data_rect(res)
        cands = [QRectF(W - lw - 10, 28, lw, lh), QRectF(W - lw - 10, H - lh - 34, lw, lh),
                 QRectF(10, 28, lw, lh), QRectF(10, H - lh - 46, lw, lh)]

        def cover(r):
            i = r.intersected(data)
            return 0.0 if i.isEmpty() else i.width() * i.height()
        best = min(cands, key=cover)
        self._legend_over = cover(best) > 0
        return best

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
        over_cam = self._pick_camera(pos)
        if res is not None and self.show_tooltip and over_cam is None:
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
        # a camera under the pointer lights up its bar in the Coverage panel (and its ground here)
        uid = over_cam.uid if over_cam is not None else ''
        if uid or getattr(self, '_hover_owner', False):
            self._hover_owner = bool(uid)
            CAMERA_HOVER.set(uid)
        if info or had:
            self.update()
        self.setCursor(Qt.CursorShape.CrossCursor if info else Qt.CursorShape.ArrowCursor)

    # ------------------------------------------------------------------ hover read-out
    def readout(self, info=None):
        """The hover read-out as text rows [(role, text)], roles: head (the value under the cursor, with its unit),
        sub, crit (blind), cam (one row per camera: tag, px/cm, range m, off-axis °), more, foot. None without a
        hover point. The figures are the analysis cell's own values (not the interpolated display)."""
        info = info if info is not None else self.hover
        if not info:
            return None
        mode = self._map_mode()
        cams = info.get('cams') or []
        extra = info.get('extra') or []
        n = len(cams)
        out = []
        if n == 0:
            out += [('crit', 'Blind'), ('sub', 'no camera sees this point')]
        else:
            d = float(cams[0][1])
            tag = cams[0][0].tag
            cn = f'{n} camera' + ('' if n == 1 else 's')
            if mode == 'density':
                mn, tg = self._thresholds()
                state = (f'below the minimum {mn:g}' if d < mn else
                         f'meets the minimum, below the target {tg:g}' if d < tg else f'meets the target {tg:g}')
                out += [('head', f'{d:.1f} px/cm'), ('sub', f'best {tag} · {state}')]
            elif mode == 'redundancy':
                out += [('head', cn), ('sub', 'one camera only: no redundancy' if n == 1 else f'best {tag} · {d:.1f} px/cm')]
            elif mode == 'coverage':
                out += [('head', f'best {tag} · {d:.1f} px/cm'), ('sub', f'{cn} see this point' if n > 1 else 'only this camera sees it')]
            else:
                out += [('head', f'{cn} see this point' if n > 1 else '1 camera sees this point'), ('sub', f'best {tag} · {d:.1f} px/cm')]
        for cam, dens, rng, off in extra:
            out.append(('cam', (cam, f'{dens:.1f}', f'{rng:.2f}', f'{off:.1f}')))
        if n > len(extra):
            out.append(('more', f'+ {n - len(extra)} more'))
        gx, gy = info['point']
        o = self.project.origin_offset()
        foot = f'x {gx - o[0]:.3f}  y {gy - o[1]:.3f} m'
        if extra and self.interactive and self.result_fn is None:
            foot += f' · click selects {extra[0][0].tag}'
        out.append(('foot', foot))
        return out

    def _readout(self, p):
        rows = self.readout()
        if not rows:
            return
        pos = self.hover['pos']
        W = 252
        hgt = {'head': 20, 'sub': 15, 'crit': 20, 'cam': 17, 'more': 15, 'foot': 18}
        H = 12 + sum(hgt[r] for r, _ in rows) + (16 if any(r == 'cam' for r, _ in rows) else 0) + 2
        x, y = pos.x() + 18, pos.y() + 16
        if x + W > self.width() - 4:
            x = pos.x() - W - 14
        if y + H > self.height() - 4:
            y = pos.y() - H - 12
        x, y = max(4.0, x), max(4.0, y)
        p.save()
        p.setPen(QPen(THEME.color('strong'), 1))
        p.setBrush(THEME.color('raised'))
        p.drawRoundedRect(QRectF(x, y, W, H), 4, 4)
        cols = [x + 10, x + 112, x + 172, x + W - 10]
        yy = y + 8
        header = False
        for role, val in rows:
            h = hgt[role]
            if role == 'head':
                p.setFont(mono(13, QFont.Weight.DemiBold))
                p.setPen(THEME.color('ink'))
                p.drawText(QRectF(x + 10, yy, W - 20, h), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, val)
            elif role == 'crit':
                p.fillRect(QRectF(x + 10, yy + 5, 12, 10), THEME.color('critical', 0.3))
                p.fillRect(QRectF(x + 10, yy + 5, 12, 10), QBrush(QColor(STATUS['critical']), Qt.BrushStyle.BDiagPattern))
                p.setFont(sans(13, QFont.Weight.DemiBold))
                p.setPen(THEME.color('ink'))
                p.drawText(QRectF(x + 28, yy, W - 38, h), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, val)
            elif role in ('sub', 'more'):
                p.setFont(sans(11))
                p.setPen(THEME.color('ink2' if role == 'sub' else 'muted'))
                p.drawText(QRectF(x + 10, yy, W - 20, h), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, val)
            elif role == 'cam':
                if not header:
                    yy += 4
                    p.setPen(QPen(THEME.color('hair'), 1))
                    p.drawLine(QPointF(x + 10, yy), QPointF(x + W - 10, yy))
                    p.setFont(sans(10))
                    p.setPen(THEME.color('muted'))
                    p.drawText(QRectF(cols[0], yy + 1, 60, 12), Qt.AlignmentFlag.AlignLeft, 'camera')
                    for t, cx in (('px/cm', cols[1]), ('range m', cols[2]), ('off-axis °', cols[3])):
                        p.drawText(QRectF(cx - 70, yy + 1, 70, 12), Qt.AlignmentFlag.AlignRight, t)
                    yy += 12
                    header = True
                cam, dens, rng, off = val
                paint_chip(p, cols[0], yy + h / 2, cam.slot, cam.tag, size=7)
                p.setFont(mono(11))
                p.setPen(THEME.color('ink'))
                for v, cx in ((dens, cols[1]), (rng, cols[2]), (off, cols[3])):
                    p.drawText(QRectF(cx - 60, yy, 60, h), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, v)
            elif role == 'foot':
                yy += 2
                p.setFont(sans(10))
                p.setPen(THEME.color('muted'))
                p.drawText(QRectF(x + 10, yy, W - 20, h - 2), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, val)
            yy += h
        p.restore()

    def _crosshair(self, p, res):
        """Hairlines through the hover point, the analysis cell being read, and sight lines to the cameras that see
        the point (the best one solid)."""
        info = self.hover
        if res is None or not info or info.get('cell') is None:
            return
        pos = info['pos']
        p.save()
        p.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        p.setPen(QPen(THEME.color('ink', 0.22), 1))
        p.drawLine(QPointF(0, pos.y()), QPointF(self.width(), pos.y()))
        p.drawLine(QPointF(pos.x(), 0), QPointF(pos.x(), self.height()))
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        for k, (cam, dens, rng, off) in enumerate(info.get('extra') or []):
            pen = QPen(THEME.cam(cam.slot, 0.9 if k == 0 else 0.6), 1.5 if k == 0 else 1.0)
            if k:
                pen.setStyle(Qt.PenStyle.CustomDashLine)
                pen.setDashPattern([3, 3])
            p.setPen(pen)
            p.drawLine(pos, self.w2s(cam.x, cam.y))
        g = res.grid
        i, j = info['cell']
        a, b = self.w2s(g.x0 + i * g.cell, g.y0 + (j + 1) * g.cell), self.w2s(g.x0 + (i + 1) * g.cell, g.y0 + j * g.cell)
        p.setBrush(Qt.BrushStyle.NoBrush)
        if b.x() - a.x() >= 6:
            p.setPen(QPen(THEME.color('viewport', 0.9), 3))
            p.drawRect(QRectF(a, b))
            p.setPen(QPen(THEME.color('ink'), 1.2))
            p.drawRect(QRectF(a, b))
        else:
            p.setPen(QPen(THEME.color('ink'), 1.2))
            p.drawEllipse(pos, 3.5, 3.5)
        p.restore()

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
        if getattr(self, '_hover_owner', False):
            self._hover_owner = False
            CAMERA_HOVER.set('')
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
