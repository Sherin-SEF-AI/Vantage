"""Simulated camera images: real lens model, body occlusion from the cube map, ground grid, ROI and seams."""
from __future__ import annotations

import math

import numpy as np
from PyQt6.QtCore import Qt, QPointF, QRectF, QTimer, pyqtSignal, QLineF
from PyQt6.QtGui import QImage, QPainter, QColor, QPen, QFont
from PyQt6.QtWidgets import QWidget

from ..core.analysis import camera_frames
from ..core.lens import resolve
from .theme import THEME, mono, sans
from . import guides, maps

_unproj_cache = {}


def _unproject(intr, w, h):
    key = (intr.width, intr.height, intr.f_mm, intr.projection, tuple(intr.kb), intr.cx, intr.cy, w, h)
    if key in _unproj_cache:
        return _unproj_cache[key]
    us = (np.arange(w) + 0.5) / w * intr.width
    vs = (np.arange(h) + 0.5) / h * intr.height
    U, V = np.meshgrid(us, vs)
    d, th = intr.unproject(U.ravel(), V.ravel())
    if len(_unproj_cache) > 40:
        _unproj_cache.clear()
    _unproj_cache[key] = (d, th)
    return d, th


def render_camera(project, engine, cam, w, h, result=None, grid_step=0.5, draw_roi=True, draw_seams=True):
    """Returns (QImage, stats dict)."""
    intr = resolve(project.module_for(cam), project.lens_for(cam))
    d_opt, th = _unproject(intr, w, h)
    R, W2O = camera_frames(cam)
    dw = d_opt @ W2O          # optical -> world (W2O is orthonormal; row vectors times W2O = O2W applied)
    pos = cam.pos()
    valid = th <= intr.max_theta
    dz = dw[:, 2]
    with np.errstate(divide='ignore', invalid='ignore'):
        tg = np.where(dz < -1e-6, -pos[2] / dz, np.inf)
    ground = valid & np.isfinite(tg) & (tg < 40)
    body_d = np.full(len(dw), np.inf)
    shade = np.zeros(len(dw), np.uint8)
    vg = engine.vehicle_geom(project)
    if vg is not None:
        cube = engine.cube(pos)
        body_d, shade = cube.lookup(dw)
    body = valid & np.isfinite(body_d) & (body_d < np.where(ground, tg, np.inf))
    ground_vis = ground & ~body
    sky = valid & ~body & ~ground
    tok = THEME.tok
    gcol = np.array(QColor(tok['ground']).getRgb()[:3], np.float32)
    scol = np.array(QColor(tok['sky']).getRgb()[:3], np.float32)
    ccol = np.array(QColor(THEME.cam_hex(cam.slot)).getRgb()[:3], np.float32)
    img = np.zeros((len(dw), 3), np.float32)
    fade = np.clip(1.0 - (np.where(ground_vis, tg, 0) - 2.0) / 30.0, 0.55, 1.0)
    img[ground_vis] = gcol * fade[ground_vis, None]
    img[sky] = scol
    s = shade[body].astype(np.float32) / 255.0
    img[body] = (np.array([200, 199, 194], np.float32) * (0.35 + 0.65 * s[:, None])) * 0.55 + ccol * 0.45
    img = img.reshape(h, w, 3)
    rgba = np.concatenate([img, np.full((h, w, 1), 255, np.float32)], axis=2).astype(np.uint8)
    q = maps.to_qimage(rgba)
    stats = {'body_pct': 100.0 * body.sum() / max(valid.sum(), 1), 'ground_pct': 100.0 * ground_vis.sum() / max(valid.sum(), 1)}
    # overlays: ground grid, ROI, seams
    p = QPainter(q)
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    sx, sy = w / intr.width, h / intr.height

    def poly(P, pen):
        if len(P) < 2:
            return
        dv = P - pos
        dist = np.linalg.norm(dv, axis=1)
        opt = dv @ W2O.T
        u, v, t, ok = intr.project_dirs(opt)
        if vg is not None:
            bd, _ = cube.lookup(dv)
            ok &= bd > dist - 0.03
        p.setPen(pen)
        segs = []
        for i in range(len(P) - 1):
            if ok[i] and ok[i + 1] and abs(u[i] - u[i + 1]) * sx < w / 3:
                segs.append(QLineF(u[i] * sx, v[i] * sy, u[i + 1] * sx, v[i + 1] * sy))
        if segs:
            p.drawLines(segs)

    R_ = 7.0
    gx0, gy0 = math.floor((pos[0] - R_) / grid_step) * grid_step, math.floor((pos[1] - R_) / grid_step) * grid_step
    pen = QPen(QColor(235, 235, 230, 120), 1)
    for k in np.arange(0, 2 * R_ + 1e-6, grid_step):
        xs = np.arange(-R_, R_, 0.04)
        poly(np.c_[np.full_like(xs, gx0 + k), pos[1] + xs, np.zeros_like(xs)], pen)
        poly(np.c_[pos[0] + xs, np.full_like(xs, gy0 + k), np.zeros_like(xs)], pen)
    res = result
    if res is not None and draw_roi:
        pen = QPen(QColor(250, 250, 245, 230), 1.4, Qt.PenStyle.CustomDashLine)
        pen.setDashPattern([4, 3])
        for b in res.grid.boundary:
            bb = np.asarray(b)
            # densify
            seg = []
            for a, c in zip(bb[:-1], bb[1:]):
                n = max(2, int(np.linalg.norm(c - a) / 0.04))
                seg.append(np.linspace(a, c, n, endpoint=False))
            if seg:
                P = np.concatenate(seg)
                poly(np.c_[P, np.zeros(len(P))], pen)
    if res is not None and draw_seams and res.seams:
        pen = QPen(QColor(250, 250, 245, 200), 1.2, Qt.PenStyle.CustomDashLine)
        pen.setDashPattern([2, 2])
        cx, cy = res.centre
        for sm in res.seams:
            if cam.uid not in (sm.a, sm.b):
                continue
            a = math.radians(sm.angle)
            rr = np.arange(0.4, 8, 0.03)
            poly(np.c_[cx + rr * math.cos(a), cy + rr * math.sin(a), np.zeros(len(rr))], pen)
    p.end()
    return q, stats


class CamImageView(QWidget):
    """Shows the simulated image of one camera (the selection by default)."""
    clicked = pyqtSignal()

    def __init__(self, ctx, uid=None, parent=None, header=False, legend=True, res_scale=1.0):
        super().__init__(parent)
        self.ctx = ctx
        self.uid = uid
        self.header = header
        self.legend = legend
        self.res_scale = res_scale
        self.label = ''
        self._img = None
        self._key = None
        self.stats = {}
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(40)
        self.timer.timeout.connect(self._render)
        st = ctx.store
        # bound methods (not lambdas) so Qt drops the connections when this widget is deleted
        st.changed.connect(self._on_change)
        st.selection_changed.connect(self._on_change)
        st.project_replaced.connect(self._on_change)
        ctx.analysis.finished.connect(self._on_change)
        THEME.changed.connect(self._on_theme)
        self.setMinimumSize(80, 60)

    def camera(self):
        st = self.ctx.store
        if self.uid:
            return st.cam(self.uid)
        return st.selected

    def _on_change(self, *_):
        self.schedule()

    def _on_theme(self, *_):
        self._key = None
        self.schedule()

    def schedule(self):
        if self.isVisible():
            self.timer.start()

    def showEvent(self, e):
        self.schedule()

    def resizeEvent(self, e):
        self.schedule()

    def _render(self):
        cam = self.camera()
        if cam is None:
            self._img = None
            self.update()
            return
        project = self.ctx.store.project
        intr = resolve(project.module_for(cam), project.lens_for(cam))
        aspect = intr.width / intr.height
        W, H = self.width(), self.height() - (22 if self.header else 0)
        if W / max(H, 1) > aspect:
            h = H
            w = int(h * aspect)
        else:
            w = W
            h = int(w / aspect)
        k = self.res_scale * (0.5 if self.ctx.store.dragging else 1.0)
        w, h = max(int(w * k), 16), max(int(h * k), 12)
        res = self.ctx.analysis.result
        # the resolved intrinsics, not just the lens id: lens edits and unit calibrations change the image too
        key = (cam.x, cam.y, cam.z, cam.yaw, cam.pitch, cam.roll, intr, cam.slot, w, h, id(res), THEME.name,
               id(project.vehicle.mesh))
        if key == self._key:
            return
        try:
            self._img, self.stats = render_camera(project, self.ctx.analysis.engine, cam, w, h, res)
            self._key = key
        except Exception as ex:  # keep the panel alive on odd geometry
            self._img = None
            print('camera render failed:', ex)
        self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        p.fillRect(self.rect(), THEME.color('viewport'))
        top = 0
        cam = self.camera()
        if self.header and cam is not None:
            top = 22
        if self._img is not None:
            img = self._img
            W, H = self.width(), self.height() - top
            s = min(W / img.width(), H / img.height())
            w, h = img.width() * s, img.height() * s
            r = QRectF((W - w) / 2, top + (H - h) / 2, w, h)
            p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
            p.drawImage(r, img)
            guides.paint_camera(self, p, cam, r)
            if self.label:
                p.setFont(sans(11))
                p.setPen(QColor(235, 235, 230))
                p.drawText(QRectF(r.left() + 8, r.top() + 4, r.width(), 16), Qt.AlignmentFlag.AlignLeft, self.label)
            if self.legend and r.width() > 260:
                p.setFont(sans(10))
                p.setPen(THEME.color('ink2'))
                x = r.right() + 8 if W - r.right() > 90 else r.right() - 96
                y = r.bottom() - 46
                rd = self.ctx.store.project.roi.distance
                for i, (txt, style) in enumerate(((f'ROI {rd:.1f} m', [4, 3]), ('seams', [2, 2]), ('grid 0.5 m', None))):
                    yy = y + i * 16
                    if style:
                        pen = QPen(THEME.color('ink2'), 1, Qt.PenStyle.CustomDashLine)
                        pen.setDashPattern(style)
                        p.setPen(pen)
                        p.drawLine(QPointF(x, yy - 4), QPointF(x + 18, yy - 4))
                    p.setPen(THEME.color('ink2'))
                    p.drawText(QPointF(x + 24, yy), txt)
        elif cam is None:
            p.setPen(THEME.color('muted'))
            p.setFont(sans(11))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, 'Select a camera')
        if self.header and cam is not None:
            from .widgets import paint_chip
            p.fillRect(QRectF(0, 0, self.width(), top), THEME.color('panel'))
            x = paint_chip(p, 8, top / 2, cam.slot, cam.tag)
            p.setFont(sans(12))
            p.setPen(THEME.color('ink'))
            p.drawText(QPointF(x + 10, top / 2 + 4), cam.name)
            project = self.ctx.store.project
            m = project.module_for(cam)
            p.setFont(mono(11))
            p.setPen(THEME.color('muted'))
            p.drawText(QRectF(0, 0, self.width() - 8, top), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                       f'port {cam.port or "—"} · {m.fps:g} fps')
        p.end()

    def mousePressEvent(self, e):
        cam = self.camera()
        if cam is not None and self.uid:
            self.ctx.store.select_cams([cam.uid])
        self.clicked.emit()

    def grab_image(self):
        return self._img if self._img is not None else self.grab().toImage()
