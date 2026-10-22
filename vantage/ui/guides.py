"""Parking guidelines in the views: the shared state (the steering angle while the slider is dragged, cached guideline
sets and camera projections), the drawing for the plan and bird's-eye views, the camera images and the 3D scene, and
the steering bar of the camera views page. Shown while View ▸ Overlays ▸ Parking guidelines is on."""
from __future__ import annotations

import json
import math

import numpy as np
from PyQt6.QtCore import QObject, Qt, QPointF, QRectF, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen, QPolygonF
from PyQt6.QtWidgets import QWidget, QHBoxLayout, QGridLayout, QCheckBox, QSlider, QLineEdit, QPushButton

from ..core import guidelines as G
from .theme import mono, sans

OVERLAY = 'guidelines'
Z_LINES = 0.012                     # above the heatmap ground (0.002) and the swept fill
Z_FILL = 0.006
Z_SOFT_BOWL = 0.04                  # the QPainter renderer paints ground lines (|z| < 0.03) before the bowl floor


def _soft(vp) -> bool:
    """The QPainter fallback renderer: no depth buffer, so no translucent ground fill (it would sort over the car)."""
    return type(vp).__name__ == 'SoftViewport'


def controller(ctx) -> 'GuideController':
    c = getattr(ctx, '_guides', None)
    if c is None:
        c = GuideController(ctx)
        ctx._guides = c
    return c


def enabled(ctx) -> bool:
    return bool(ctx.overlays.get(OVERLAY))


def _hook(obj, ctx, slot):
    """Repaint obj when the guidelines change (connected once; a bound method, so Qt drops it with obj)."""
    if not getattr(obj, '_guides_hooked', False):
        controller(ctx).changed.connect(slot)
        obj._guides_hooked = True


class GuideController(QObject):
    changed = pyqtSignal()

    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        self._preview = None            # road-wheel angle while the steering slider is dragged (not in the project)
        self._outline = (None, None)
        self._set = (None, None)
        self._cams = {}
        self._tris = (None, None)
        ctx.store.changed.connect(self._on_change)
        ctx.store.project_replaced.connect(self._on_change)

    def _on_change(self, *_):
        self._set = (None, None)
        self._cams = {}
        self.changed.emit()

    @property
    def project(self):
        return self.ctx.store.project

    # ---- settings
    def stored(self) -> G.GuidelineSettings:
        return G.GuidelineSettings.from_dict(self.project.analysis.guidelines)

    @property
    def settings(self) -> G.GuidelineSettings:
        st = self.stored()
        if self._preview is not None:
            st.angle_deg = max(-st.max_deg, min(st.max_deg, self._preview))
        return st

    def preview(self, angle):
        """Show this angle while the slider is dragged; nothing is written to the project yet."""
        self._preview = float(angle)
        self._set = (None, None)
        self._cams = {}
        self.changed.emit()

    def commit(self, angle, merge=False):
        """Write the angle as one undo step (merge: consecutive key presses on the slider make one step)."""
        self._preview = None
        st = self.stored()
        a = round(max(-st.max_deg, min(st.max_deg, float(angle))), 1)
        if abs(a - st.angle_deg) < 1e-9:
            self._on_change()
            return
        self.set(angle_deg=a, _merge=merge)

    def set(self, _merge=False, **kw):
        new = G.GuidelineSettings.from_dict({**self.stored().to_dict(), **kw}).to_dict()
        if new == self.stored().to_dict():
            return
        if set(kw) == {'angle_deg'}:
            text = f'Steering {new["angle_deg"]:+.1f}°'
        else:
            text = 'Guidelines · ' + ', '.join(f'{k.replace("_", " ")} {v}' for k, v in kw.items())

        def fn(p):
            p.analysis.guidelines = new
        self.ctx.store.edit('analysis', fn, text, merge_key=('guidelines', tuple(sorted(kw))) if _merge else None)

    # ---- geometry
    def outline(self) -> G.VehicleOutline:
        veh = self.project.vehicle
        key = (id(veh.mesh), veh.mesh.n_tris if veh.mesh is not None else 0,
               json.dumps(veh.wheel_centres, sort_keys=True, default=str), json.dumps(veh.spec, sort_keys=True, default=str))
        if self._outline[0] != key:
            self._outline = (key, G.vehicle_outline(self.project))
        return self._outline[1]

    def ground(self) -> G.GuidelineSet:
        st = self.settings
        o = self.outline()
        obs = tuple((ob.uid, ob.x, ob.y, ob.height, ob.radius) for ob in self.project.obstacles)
        key = (id(o), json.dumps(st.to_dict(), sort_keys=True), obs)
        if self._set[0] != key:
            self._set = (key, G.guideline_set(o, st, obstacles=self.project.obstacles))
            self._cams = {}
        return self._set[1]

    def camera(self, cam) -> G.CamProjection:
        gs = self.ground()
        p = self.project
        lens = json.dumps([p.module_for(cam).to_dict(), p.lens_for(cam).to_dict()], sort_keys=True, default=str)
        key = (cam.x, cam.y, cam.z, cam.yaw, cam.pitch, cam.roll, lens, id(gs), id(p.vehicle.mesh))
        hit = self._cams.get(cam.uid)
        if hit is None or hit[0] != key:
            hit = (key, G.project_ground(p, self.ctx.analysis.engine, cam, gs.lines))
            self._cams[cam.uid] = hit
        return hit[1]

    def swept_tris(self, gs) -> np.ndarray:
        if self._tris[0] is not gs:
            self._tris = (gs, G.triangulate(gs.swept) if gs.swept is not None else np.zeros((0, 3), np.int64))
        return self._tris[1]

    def describe(self, st=None, gs=None) -> str:
        st = st or self.settings
        a = st.angle_deg
        R = G.turn_radius(self.outline(), a)
        parts = [f'{a:+.1f}° road wheel']
        if st.ratio > 0:
            parts.append(f'{a * st.ratio:+.0f}° steering wheel')
        parts.append(f'radius {abs(R):.2f} m' if math.isfinite(R) else 'straight')
        return ' · '.join(parts).replace('-', '−')


# ====================================================================== drawing
def _qc(hexs, a=1.0) -> QColor:
    c = QColor(hexs)
    c.setAlphaF(a)
    return c


def _text(p: QPainter, at: QPointF, text: str, color: QColor, size=10, inside=None):
    """Text with a dark halo, legible on bright ground and on dark asphalt. inside: a rect to keep it in (the text
    moves to the other side of its anchor when it would run off the right or top edge)."""
    font = sans(size, QFont.Weight.DemiBold)
    if inside is not None:
        from PyQt6.QtGui import QFontMetricsF
        fm = QFontMetricsF(font, p.device())
        w = fm.horizontalAdvance(text)
        x, y = at.x(), at.y()
        if x + w > inside.right() - 2:
            x -= w + 10
        if y - fm.ascent() < inside.top() + 2:
            y += fm.ascent() + 8
        at = QPointF(max(inside.left() + 2, x), y)
    path = QPainterPath()
    path.addText(at, font, text)
    p.setPen(QPen(QColor(0, 0, 0, 170), 3.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawPath(path)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(color)
    p.drawPath(path)


def _pen(ln, width, scale=1.0) -> QPen:
    pen = QPen(_qc(ln.hex), width * scale)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    if ln.dash or ln.kind == 'swept':
        pen.setStyle(Qt.PenStyle.CustomDashLine)
        pen.setDashPattern([4, 3] if ln.kind != 'swept' else [6, 3])
    return pen


def _width(ln) -> float:
    return {'static': 1.3, 'trajectory': 2.6, 'distance': 2.2, 'swept': 1.5}.get(ln.kind, 2.0)


def _labelled(gs) -> list:
    """Distance labels from the moving ticks when they are drawn, else from the static ones."""
    kinds = {ln.kind for ln in gs.lines}
    want = 'distance' if 'distance' in kinds else 'static'
    return [ln for ln in gs.lines if ln.kind == want and ln.label]


def paint_plan(view, p: QPainter):
    """Ground lines, the swept path and obstacle hits in a top view (plan or stitched bird's-eye)."""
    ctx = view.ctx
    if not enabled(ctx) or view.mode in ('delta', 'compare'):
        return
    _hook(view, ctx, view.update)
    try:
        gs = controller(ctx).ground()
    except Exception as ex:                 # odd geometry must not break the view
        print('guidelines failed:', ex)
        return
    p.save()
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    if gs.swept is not None:
        poly = QPolygonF([QPointF(*q) for q in view.w2s_np(gs.swept)])
        p.setPen(QPen(_qc(G.COLORS['swept'], 0.95), 1.4))
        p.setBrush(_qc(G.COLORS['swept'], 0.14))
        p.drawPolygon(poly)
    p.setBrush(Qt.BrushStyle.NoBrush)
    for ln in gs.lines:
        if ln.kind == 'swept':
            continue
        p.setPen(_pen(ln, _width(ln)))
        p.drawPolyline(QPolygonF([QPointF(*q) for q in view.w2s_np(ln.pts)]))
    if view.scale > 25:
        for ln in _labelled(gs):
            q = view.w2s_np(ln.pts[:1])[0]
            _text(p, QPointF(q[0] + 5, q[1] - 4), ln.label, _qc(ln.hex))
    p.restore()


def paint_plan_hits(view, p: QPainter):
    """Obstacles the swept path runs into, drawn last so targets and cameras do not cover the labels."""
    ctx = view.ctx
    if not enabled(ctx) or view.mode in ('delta', 'compare'):
        return
    try:
        hits = controller(ctx).ground().hits
    except Exception:
        return
    p.save()
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    axis = view.w2s(0.0, 0.0).y()
    for ob, s in hits:
        c = view.w2s(ob.x, ob.y)
        r = max(6.0, ob.radius * view.scale + 4)
        p.setPen(QPen(_qc(G.COLORS['red']), 2.0))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(c, r, r)
        # on the side away from the car's centre line, where the guidelines do not run
        _chip(p, c, r, f'{ob.name} · hit after {s:.2f} m', _qc(G.COLORS['red']), above=c.y() <= axis,
              inside=QRectF(view.rect()))
    p.restore()


def _chip(p: QPainter, c: QPointF, r: float, text: str, color: QColor, above: bool, inside: QRectF):
    """A label on a dark rounded chip next to a marker, like the area labels of the other top-view overlays."""
    from PyQt6.QtGui import QFontMetricsF
    font = sans(10, QFont.Weight.DemiBold)
    fm = QFontMetricsF(font, p.device())        # the device's resolution, as drawText will use
    w, h = fm.horizontalAdvance(text) + 12, fm.height() + 6
    y = c.y() - r - 6 - h if above else c.y() + r + 6
    x = min(max(c.x() - w / 2, inside.left() + 4), inside.right() - w - 4)      # kept inside the view
    y = min(max(y, inside.top() + 4), inside.bottom() - h - 4)
    box = QRectF(x, y, w, h)
    p.setPen(QPen(color, 1.0))
    p.setBrush(QColor(18, 18, 18, 225))
    p.drawRoundedRect(box, 4, 4)
    p.setFont(font)
    p.setPen(color)
    p.drawText(box, Qt.AlignmentFlag.AlignCenter, text)


def paint_camera(view, p: QPainter, cam, rect):
    """The guidelines a display would draw over this camera's image (rect: where the image is drawn)."""
    ctx = view.ctx
    if not enabled(ctx) or cam is None:
        return
    _hook(view, ctx, view.update)
    try:
        pr = controller(ctx).camera(cam)
    except Exception as ex:
        print('guidelines failed:', ex)
        return
    sx, sy = rect.width() / pr.width, rect.height() / pr.height
    k = max(0.6, min(1.6, rect.width() / 640))
    p.save()
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    p.setClipRect(rect)
    polys = []
    for ln, pieces in pr.pieces:
        for px, _ in pieces:
            polys.append((ln, QPolygonF([QPointF(rect.left() + u * sx, rect.top() + v * sy) for u, v in px])))
    # a dark underlay first, so the colours read on any background
    for ln, poly in polys:
        if ln.kind in ('trajectory', 'distance'):
            p.setPen(QPen(QColor(0, 0, 0, 110), (_width(ln) + 2.2) * k, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap,
                          Qt.PenJoinStyle.RoundJoin))
            p.drawPolyline(poly)
    for ln, poly in polys:
        p.setPen(_pen(ln, _width(ln), k))
        p.drawPolyline(poly)
    if rect.width() >= 240:
        want = {id(ln) for ln in _labelled(controller(ctx).ground())}
        for ln, pieces in pr.pieces:
            if id(ln) in want and pieces:
                u, v = pieces[0][0][0]
                _text(p, QPointF(rect.left() + u * sx + 5, rect.top() + v * sy - 4), ln.label, _qc(ln.hex),
                      size=10 if rect.width() < 640 else 12, inside=rect)
    p.restore()


def scene_lines(scene, vp) -> list:
    ctx = scene.ctx
    if not enabled(ctx) or scene.mode not in ('scene', 'bowl'):
        return []
    _hook(scene, ctx, scene._emit)
    from .viewport3d import LineDraw
    try:
        gs = controller(ctx).ground()
    except Exception as ex:
        print('guidelines failed:', ex)
        return []
    out = []
    z = Z_SOFT_BOWL if (_soft(vp) and scene.mode == 'bowl') else Z_LINES
    for ln in gs.lines:
        pts = np.c_[ln.pts, np.full(len(ln.pts), z)]
        out.append(LineDraw(pts, _qc(ln.hex), _width(ln), dash=ln.dash or ln.kind == 'swept', dash_len=0.08))
    a = np.linspace(0, 2 * math.pi, 25)
    for ob, _ in gs.hits:
        ring = np.c_[ob.x + (ob.radius + 0.06) * np.cos(a), ob.y + (ob.radius + 0.06) * np.sin(a), np.full(25, 0.02)]
        out.append(LineDraw(ring, _qc(G.COLORS['red']), 2.5, depth=False))
    return out


def scene_tris(scene, vp) -> list:
    ctx = scene.ctx
    if not enabled(ctx) or scene.mode not in ('scene', 'bowl') or _soft(vp):
        return []
    from .viewport3d import TriDraw
    try:
        gc = controller(ctx)
        gs = gc.ground()
        if gs.swept is None:
            return []
        tri = gc.swept_tris(gs)
    except Exception as ex:
        print('guidelines failed:', ex)
        return []
    if not len(tri):
        return []
    V = np.c_[gs.swept, np.full(len(gs.swept), Z_FILL)]
    col = np.array(_qc(G.COLORS['swept']).getRgbF()[:3] + (0.2,))
    return [TriDraw(V[tri], np.tile(col, (len(tri), 1)), depth=True)]


# ====================================================================== the steering bar
class GuideBar(QWidget):
    """Camera views page: guidelines on/off, direction, the steering slider (a drag is one undo step) and the
    guideline settings."""

    def __init__(self, ctx, parent=None):
        super().__init__(parent)
        from .widgets import ScrubSpinBox, SegmentedControl, label
        self.ctx = ctx
        self.gc = controller(ctx)
        self.setObjectName('panel')
        g = QGridLayout(self)
        g.setContentsMargins(8, 4, 8, 4)
        g.setHorizontalSpacing(8)
        g.setVerticalSpacing(3)
        self.on = QCheckBox('Parking guidelines')
        self.on.setToolTip('Draw the guidelines over the camera images, the bird\'s-eye view, the bowl and the 3D and top '
                           'views (View ▸ Overlays ▸ Parking guidelines)')
        self.on.toggled.connect(lambda v: ctx.set_overlay(OVERLAY, v))
        g.addWidget(self.on, 0, 0)
        self.dir = SegmentedControl([('reverse', 'Reverse'), ('forward', 'Forward')], 'reverse', min_w=64)
        self.dir.changed.connect(lambda k: self.gc.set(direction=k))
        g.addWidget(self.dir, 0, 1)
        sl = QWidget()
        hl = QHBoxLayout(sl)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.setSpacing(4)
        hl.addWidget(label('L', ink2=True))
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimumWidth(160)
        self.slider.setToolTip('Steering: drag to turn the wheels (left of centre steers left). The views follow while you '
                               'drag; the angle is saved when you let go.')
        self.slider.setSingleStep(10)
        self.slider.setPageStep(50)
        self.slider.valueChanged.connect(self._slid)
        self.slider.sliderReleased.connect(lambda: self.gc.commit(self._angle(self.slider.value())))
        hl.addWidget(self.slider, 1)
        hl.addWidget(label('R', ink2=True))
        self.centre = QPushButton('0°')
        self.centre.setToolTip('Straight ahead')
        self.centre.setFixedWidth(34)
        self.centre.clicked.connect(lambda: self.gc.commit(0.0))
        hl.addWidget(self.centre)
        g.addWidget(sl, 0, 2, 1, 3)
        self.readout = label('', mono_=True, small=True)
        g.addWidget(self.readout, 0, 5, 1, 2)
        self.checks = {}
        row = QWidget()
        rl = QHBoxLayout(row)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(8)
        for key, text, tip in (('static', 'Static', 'Width lines and distance markers straight back or ahead'),
                               ('dynamic', 'Trajectory', 'The paths of the bumper corners at this steering angle'),
                               ('swept', 'Swept path', 'The ground the whole body (with the mirrors) passes over')):
            cb = QCheckBox(text)
            cb.setToolTip(tip)
            cb.toggled.connect(lambda v, k=key: self.gc.set(**{k: v}))
            rl.addWidget(cb)
            self.checks[key] = cb
        g.addWidget(row, 1, 0, 1, 2)
        self.fields = {}
        col = 2
        for key, text, unit, dec, tip in (('path_m', 'Path', 'm', 1, 'Length of the trajectory lines'),
                                          ('margin_m', 'Margin', 'm', 2, 'Added to the body width on each side'),
                                          ('max_deg', 'Full lock', '°', 1, 'Road-wheel angle at full lock'),
                                          ('ratio', 'Ratio', '', 1, 'Steering ratio, for the steering-wheel angle')):
            f = ScrubSpinBox('', 'plain', dec, unit, width=64)
            f.setToolTip(tip)
            f.committed.connect(lambda v, k=key: self.gc.set(**{k: v}))
            box = QWidget()
            hb = QHBoxLayout(box)
            hb.setContentsMargins(0, 0, 0, 0)
            hb.setSpacing(4)
            hb.addWidget(label(text, ink2=True))
            hb.addWidget(f)
            g.addWidget(box, 1, col)
            col += 1
            self.fields[key] = f
        self.marks = QLineEdit()
        self.marks.setToolTip('Distance markers from the bumper, in metres (the first two end the red and yellow bands)')
        self.marks.setFixedWidth(110)
        self.marks.setFont(mono(11))
        self.marks.editingFinished.connect(self._marks_done)
        mb = QWidget()
        ml = QHBoxLayout(mb)
        ml.setContentsMargins(0, 0, 0, 0)
        ml.setSpacing(4)
        ml.addWidget(label('Markers', ink2=True))
        ml.addWidget(self.marks)
        g.addWidget(mb, 1, col)
        self.status = label('', small=True)
        g.addWidget(self.status, 2, 0, 1, 7)
        g.setColumnStretch(4, 1)
        ctx.store.changed.connect(self.sync)
        ctx.store.project_replaced.connect(self.sync)
        ctx.overlays_changed.connect(self.sync)
        self.gc.changed.connect(self._status)
        self.sync()

    @staticmethod
    def _angle(v) -> float:
        return -v / 10.0                    # the slider runs right to left: left of centre is a left (+) turn

    def _slid(self, v):
        if self.slider.isSliderDown():
            self.gc.preview(self._angle(v))
        elif not getattr(self, '_syncing', False):
            self.gc.commit(self._angle(v), merge=True)

    def _marks_done(self):
        try:
            ds = [float(x) for x in self.marks.text().replace(';', ',').replace('m', '').split(',') if x.strip()]
        except ValueError:
            ds = []
        if ds:
            self.gc.set(distances=ds)
        self.sync()

    def sync(self, *_):
        st = self.gc.stored()
        self._syncing = True
        try:
            self.on.blockSignals(True)
            self.on.setChecked(enabled(self.ctx))
            self.on.blockSignals(False)
            self.dir.set_value(st.direction)
            if not self.slider.isSliderDown():
                m = int(round(st.max_deg * 10))
                self.slider.blockSignals(True)
                self.slider.setRange(-m, m)
                self.slider.setValue(int(round(-st.angle_deg * 10)))
                self.slider.blockSignals(False)
            for k, cb in self.checks.items():
                cb.blockSignals(True)
                cb.setChecked(getattr(st, k))
                cb.blockSignals(False)
            for k, f in self.fields.items():
                if not f.hasFocus():
                    f.set_value(getattr(st, k))
            if not self.marks.hasFocus():
                self.marks.setText(', '.join(f'{d:g}' for d in st.distances))
        finally:
            self._syncing = False
        self._status()

    def _status(self, *_):
        st = self.gc.settings
        self.readout.setText(self.gc.describe(st))
        if not enabled(self.ctx):
            self.status.setText('')
            return
        try:
            gs = self.gc.ground()
        except Exception:
            self.status.setText('')
            return
        way = 'reversing' if st.direction == 'reverse' else 'driving forward'
        if gs.hits:
            hits = ', '.join(f'{ob.name} after {s:.2f} m' for ob, s in gs.hits[:3])
            self.status.setText(f'Swept path, {way} {st.length:.1f} m: the body would hit {hits}')
            self.status.setStyleSheet(f'color: {G.COLORS["red"]};')
        else:
            n = len(self.ctx.store.project.obstacles)
            self.status.setText(f'Swept path, {way} {st.length:.1f} m: ' +
                                (f'clear of all {n} obstacle' + ('s' if n != 1 else '') if n else 'no obstacles placed'))
            self.status.setStyleSheet('')
