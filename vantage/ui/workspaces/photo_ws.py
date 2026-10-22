"""B5 · Reference photo match: spin set, landmark correspondences, residuals, matched 3D view."""
from __future__ import annotations

import math
import os

import numpy as np
from PyQt6.QtCore import Qt, QTimer, QPointF, QRectF, QEvent, pyqtSignal
from PyQt6.QtGui import QPainter, QColor, QPen, QFont, QImage, QLinearGradient
from PyQt6.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QSplitter, QInputDialog, QToolButton

from ...core import photos as ph
from ..scene import MainScene
from ..theme import THEME, mono, sans, STATUS
from ..viewport3d import make_viewport
from ..widgets import label, button, tool_button
from .. import icons


def gl_from_photo(ps, k):
    """(view 4x4, proj(aspect)) for an OpenGL camera that matches photo frame k."""
    pose = ph.frame_pose(ps, k)
    if pose is None or not ps.width:
        return None
    R, t = pose
    M = np.eye(4)
    M[:3, :3], M[:3, 3] = R, t
    flip = np.diag([1.0, -1.0, -1.0, 1.0])
    V = flip @ M
    W, H, f, cx, cy = ps.width, ps.height, ps.f_px, ps.cx, ps.cy
    n, fa = 0.05, 400.0

    def proj(aspect):
        P = np.zeros((4, 4))
        P[0, 0] = 2 * f / W
        P[0, 2] = 1 - 2 * cx / W
        P[1, 1] = 2 * f / H
        P[1, 2] = 2 * cy / H - 1
        P[2, 2] = -(fa + n) / (fa - n)
        P[2, 3] = -2 * fa * n / (fa - n)
        P[3, 2] = -1
        ai = W / H
        if aspect > ai:
            P[0, 0] *= ai / aspect
            P[0, 2] *= ai / aspect
        else:
            P[1, 1] *= aspect / ai
            P[1, 2] *= aspect / ai
        return P
    return {'view': V, 'proj': proj}


class PhotoCanvas(QWidget):
    def __init__(self, page):
        super().__init__()
        self.page = page
        self.ctx = page.ctx
        self.setMouseTracking(True)
        self.setMinimumSize(200, 150)
        self.hover = None

    @property
    def ps(self):
        return self.ctx.store.project.photos

    def _fit(self):
        ps = self.ps
        W, H = (ps.width or 16), (ps.height or 9)
        s = min(self.width() / W, self.height() / H)
        w, h = W * s, H * s
        return QRectF((self.width() - w) / 2, (self.height() - h) / 2, w, h), s

    def to_screen(self, uv):
        r, s = self._fit()
        return QPointF(r.left() + uv[0] * s, r.top() + uv[1] * s)

    def to_image(self, pos):
        r, s = self._fit()
        return ((pos.x() - r.left()) / s, (pos.y() - r.top()) / s)

    def paintEvent(self, e):
        p = QPainter(self)
        p.fillRect(self.rect(), THEME.color('raised'))
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        ps = self.ps
        k = ps.current
        r, s = self._fit()
        img = self.ctx.window.photo_image(k) if self.ctx.window else None
        if img is not None and not img.isNull():
            p.drawImage(r, img)
        else:
            p.fillRect(r, THEME.color('panel'))
            p.setPen(THEME.color('ink2'))
            p.setFont(sans(12))
            msg = 'Photo placeholder · the frame’s pixels draw here'
            if ps.source and not os.path.exists(ps.source):
                msg = f'Photo source not found: {ps.source}'
            elif not ps.source:
                msg = 'No photo source. Vehicle ▸ Reference photos ▸ Load spin set…'
            p.drawText(r.adjusted(10, 8, -10, -8), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop,
                       msg + '\nOverlay: matched model as wire, camera markers, residual vectors × 20')
        if not ps.frames or ph.frame_pose(ps, k) is None:
            p.end()
            return
        # model wire: body creases and silhouette-ish edges
        vg = self.ctx.analysis.engine.vg
        if vg is not None:
            segs = vg.body.crease_segments(28.0)
            if len(segs):
                uv = ph.project_points(ps, k, segs.reshape(-1, 3)).reshape(-1, 2, 2)
                ok = np.isfinite(uv).all(axis=(1, 2))
                p.setPen(QPen(QColor(240, 239, 236, 150), 1))
                from PyQt6.QtCore import QLineF
                lines = []
                for a in uv[ok]:
                    A, B = self.to_screen(a[0]), self.to_screen(a[1])
                    lines.append(QLineF(A, B))
                p.drawLines(lines)
        # camera markers
        cams = self.ctx.store.variant.cameras
        if cams:
            uv = ph.project_points(ps, k, np.array([c.pos() for c in cams]))
            for c, q in zip(cams, uv):
                if np.all(np.isfinite(q)):
                    pt = self.to_screen(q)
                    p.setPen(QPen(QColor(255, 255, 255), 1.2))
                    p.setBrush(Qt.BrushStyle.NoBrush)
                    p.drawEllipse(pt, 7, 7)
                    p.setPen(Qt.PenStyle.NoPen)
                    p.setBrush(THEME.cam(c.slot))
                    p.drawRect(QRectF(pt.x() - 3.5, pt.y() - 3.5, 7, 7))
        # correspondences + residual vectors x20
        for name, u, v, q, err in ph.point_errors(ps, k):
            pt = self.to_screen((u, v))
            p.setPen(QPen(QColor(255, 255, 255), 1))
            p.setBrush(QColor(0, 0, 0, 120))
            p.drawRect(QRectF(pt.x() - 3, pt.y() - 3, 6, 6))
            if q is not None and np.all(np.isfinite(q)):
                tip = self.to_screen((u + (q[0] - u) * 20, v + (q[1] - v) * 20))
                p.setPen(QPen(QColor(STATUS['critical']), 1.4))
                p.drawLine(pt, tip)
            if self.hover == name:
                p.setFont(mono(11))
                p.setPen(QColor(255, 255, 255))
                p.drawText(QPointF(pt.x() + 8, pt.y() - 6), f'{name} · {err:.2f} px')
        # legend
        p.setFont(sans(11))
        p.setPen(THEME.color('ink2'))
        x, y = self.width() - 180, self.height() - 12
        p.setPen(QPen(THEME.color('ink2'), 1))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(QPointF(x, y - 4), 5, 5)
        p.drawText(QPointF(x + 10, y), 'camera marker')
        p.setPen(QPen(QColor(STATUS['critical']), 1.4))
        p.drawLine(QPointF(x + 100, y - 4), QPointF(x + 116, y - 4))
        p.setPen(THEME.color('ink2'))
        p.drawText(QPointF(x + 120, y), 'residual')
        if self.page.landmark_mode:
            p.setPen(THEME.color('ink'))
            p.drawText(QPointF(12, self.height() - 12), f'Click to place “{self.ctx.photo_landmark}” in this frame · right-click removes a point')
        p.end()

    def mouseMoveEvent(self, e):
        ps = self.ps
        if not ps.frames:
            return
        best = None
        for name, u, v in ps.frames[ps.current].points:
            q = self.to_screen((u, v))
            d = math.hypot(q.x() - e.position().x(), q.y() - e.position().y())
            if d < 8 and (best is None or d < best[1]):
                best = (name, d)
        h = best[0] if best else None
        if h != self.hover:
            self.hover = h
            self.update()

    def mousePressEvent(self, e):
        ps = self.ps
        if not ps.frames:
            return
        k = ps.current
        u, v = self.to_image(e.position())
        st = self.ctx.store
        if e.button() == Qt.MouseButton.RightButton:
            pts = ps.frames[k].points
            best = None
            for i, (name, pu, pv) in enumerate(pts):
                q = self.to_screen((pu, pv))
                d = math.hypot(q.x() - e.position().x(), q.y() - e.position().y())
                if d < 10 and (best is None or d < best[1]):
                    best = (i, d)
            if best:
                i = best[0]

                def fn(p, k=k, i=i):
                    pts = list(p.photos.frames[k].points)
                    pts.pop(i)
                    p.photos.frames[k].points = pts
                    ph.refresh_residuals(p.photos)
                st.edit('photos', fn, 'Removed a photo correspondence')
            return
        lm = self.ctx.photo_landmark
        if e.button() == Qt.MouseButton.LeftButton and lm and 0 <= u <= (ps.width or 1e9) and 0 <= v <= (ps.height or 1e9):
            def fn(p, k=k, lm=lm, u=u, v=v):
                pts = [x for x in p.photos.frames[k].points if x[0] != lm]
                pts.append((lm, round(u, 2), round(v, 2)))
                p.photos.frames[k].points = pts
                ph.refresh_residuals(p.photos)
            st.edit('photos', fn, f'Placed {lm} in frame {k:02d}')


class FrameStrip(QWidget):
    picked = pyqtSignal(int)

    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        self.setFixedHeight(84)

    def paintEvent(self, e):
        p = QPainter(self)
        p.fillRect(self.rect(), THEME.color('panel'))
        ps = self.ctx.store.project.photos
        n = len(ps.frames)
        if n == 0:
            p.end()
            return
        W = self.width() - 16
        bw = W / n
        top, base = 10, 54
        mx = max([f.rms for f in ps.frames] + [ps.rms * 1.3, 1e-6])
        for i, f in enumerate(ps.frames):
            x = 8 + i * bw
            h = (f.rms / mx) * (base - top) if f.rms else 3
            cur = i == ps.current
            col = THEME.color('ink') if cur else THEME.color('strong')
            p.fillRect(QRectF(x + bw * 0.2, base - h, bw * 0.6, h), col)
            p.setPen(QPen(THEME.color('ink' if cur else 'hair'), 1))
            p.setBrush(THEME.color('ink') if cur else Qt.BrushStyle.NoBrush)
            p.drawRect(QRectF(x + bw * 0.2, base + 6, bw * 0.6, 12))
        if ps.rms:
            y = base - (ps.rms / mx) * (base - top)
            pen = QPen(THEME.color('ink2'), 1, Qt.PenStyle.DashLine)
            p.setPen(pen)
            p.drawLine(QPointF(8, y), QPointF(self.width() - 8, y))
            p.setFont(mono(10))
            p.drawText(QRectF(0, y - 14, self.width() - 10, 12), Qt.AlignmentFlag.AlignRight, f'RMS {ps.rms:.2f}')
        x = 8 + (ps.current + 0.5) * bw
        p.setPen(QPen(THEME.color('hair'), 1))
        p.drawLine(QPointF(8, base + 22), QPointF(self.width() - 8, base + 22))
        p.setBrush(THEME.color('ink'))
        p.drawRect(QRectF(x - 4, base + 18, 8, 8))
        p.end()

    def mousePressEvent(self, e):
        ps = self.ctx.store.project.photos
        n = len(ps.frames)
        if n:
            i = int((e.position().x() - 8) / ((self.width() - 16) / n))
            self.picked.emit(max(0, min(n - 1, i)))

    def mouseMoveEvent(self, e):
        if e.buttons() & Qt.MouseButton.LeftButton:
            self.mousePressEvent(e)


class PhotoPage(QWidget):
    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        self.landmark_mode = True
        self.picking_model = False
        self.setObjectName('base')
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        split = QSplitter(Qt.Orientation.Horizontal)
        split.setHandleWidth(2)
        left = QWidget()
        left.setObjectName('panel')
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.setSpacing(0)
        head = QWidget()
        head.setFixedHeight(30)
        h = QHBoxLayout(head)
        h.setContentsMargins(8, 2, 8, 2)
        self.title = label('')
        self.meta = label('', muted=True, mono_=True, small=True)
        h.addWidget(self.title)
        h.addWidget(self.meta)
        h.addStretch(1)
        b1 = button('Calibrate…')
        b1.clicked.connect(lambda: ctx.window.calibrate_photos())
        self.b_match = button('Match 3D view to photo', icon_name='frame')
        self.b_match.setCheckable(True)
        self.b_match.toggled.connect(self.match)
        h.addWidget(b1)
        h.addWidget(self.b_match)
        ll.addWidget(head)
        self.canvas = PhotoCanvas(self)
        ll.addWidget(self.canvas, 1)
        ctl = QWidget()
        ctl.setFixedHeight(28)
        c = QHBoxLayout(ctl)
        c.setContentsMargins(8, 0, 8, 0)
        c.setSpacing(2)
        self.b_prev = tool_button('step-back', 'Previous frame  ←')
        self.b_play = tool_button('play', 'Play  Space', checkable=True)
        self.b_next = tool_button('step', 'Next frame  →')
        self.b_prev.clicked.connect(lambda: self.step(-1))
        self.b_next.clicked.connect(lambda: self.step(1))
        self.b_play.toggled.connect(self.play)
        for b in (self.b_prev, self.b_play, self.b_next):
            c.addWidget(b)
        self.spin_lbl = label('', muted=True, small=True)
        c.addSpacing(8)
        c.addWidget(self.spin_lbl)
        c.addStretch(1)
        self.rms_lbl = label('', mono_=True)
        c.addWidget(self.rms_lbl)
        ll.addWidget(ctl)
        self.strip = FrameStrip(ctx)
        self.strip.picked.connect(self.set_frame)
        ll.addWidget(self.strip)
        split.addWidget(left)
        self.scene = MainScene(ctx)
        self.vp = make_viewport(self.scene)
        self.vp.installEventFilter(self)
        split.addWidget(self.vp)
        split.setSizes([720, 720])
        lay.addWidget(split)
        self.timer = QTimer(self)
        self.timer.setInterval(180)
        self.timer.timeout.connect(lambda: self.step(1))
        ctx.store.changed.connect(lambda s: self.refresh() if 'photos' in s else None)
        ctx.store.project_replaced.connect(self.refresh)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def activate(self):
        self.refresh()
        QTimer.singleShot(30, lambda: self.vp.frame_box(*self.ctx.window.viewarea._bbox(), margin=0.85))
        self.setFocus()

    def refresh(self):
        ps = self.ctx.store.project.photos
        n = len(ps.frames)
        k = ps.current if n else 0
        self.title.setText(f'Frame {k:02d} / {n}' if n else 'No photo set')
        name = os.path.basename(ps.frames[k].name) if n else ''
        self.meta.setText(f'  {name} · {ps.width} × {ps.height}' if n else '')
        self.spin_lbl.setText(f'Spin · {n} frames · ← → steps · Space plays')
        self.rms_lbl.setText(f'{ps.rms:.2f} px RMS' if ps.rms else '')
        self.scene.photo_frame = k if n else None
        if self.b_match.isChecked():
            self.match(True)
        self.canvas.update()
        self.strip.update()
        self.ctx.window.sel_label.setText(f'Reference photos · frame {k:02d}') if self.ctx.window else None

    def set_frame(self, k):
        ps = self.ctx.store.project.photos
        if not ps.frames:
            return
        ps.current = k % len(ps.frames)
        self.refresh()
        self.ctx.window.inspector.refresh()

    def step(self, d):
        ps = self.ctx.store.project.photos
        if ps.frames:
            self.set_frame(ps.current + d)

    def play(self, on):
        self.b_play.setIcon(icons.icon('pause' if on else 'play', 16))
        self.timer.start() if on else self.timer.stop()

    def match(self, on):
        ps = self.ctx.store.project.photos
        if on:
            m = gl_from_photo(ps, ps.current)
            if m is None:
                self.ctx.msg('This frame has no solved pose yet. Add landmarks and press Calibrate.')
                self.b_match.setChecked(False)
                return
            self.vp.cam.custom = m
            f35 = ph.photo_focal_35mm(ps)
            self.scene.label_prefix = f'Perspective · matched to frame {ps.current:02d} · f {f35:.0f} mm (35 mm eq.)'
        else:
            self.vp.cam.custom = None
            self.scene.label_prefix = ''
        self.vp.update()

    def keyPressEvent(self, e):
        if e.key() == Qt.Key.Key_Left:
            self.step(-1)
        elif e.key() == Qt.Key.Key_Right:
            self.step(1)
        elif e.key() == Qt.Key.Key_Space:
            self.b_play.toggle()
        else:
            super().keyPressEvent(e)

    # landmark from the model
    def start_landmark_pick(self):
        self.picking_model = True
        self.ctx.msg('Click a point on the model in the 3D view to add it as a landmark.', 6000)

    def eventFilter(self, obj, e):
        if obj is self.vp and self.picking_model and e.type() == QEvent.Type.MouseButtonPress and e.button() == Qt.MouseButton.LeftButton:
            hit = self.scene._mesh_hit(self.vp, e.position())
            self.picking_model = False
            if hit is None:
                self.ctx.msg('No surface under the cursor.')
                return True
            name, ok = QInputDialog.getText(self, 'Add landmark', 'Name', text='Landmark')
            if ok and name.strip():
                pt = [round(float(v), 4) for v in hit[0]]
                self.ctx.store.edit('photos', lambda p: p.photos.landmarks.__setitem__(name.strip(), pt), f'Added landmark {name}')
                self.ctx.photo_landmark = name.strip()
            return True
        return False
