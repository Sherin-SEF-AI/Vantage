"""Lidar–camera calibration in the UI: the pose plan seen from above (the vehicle, the camera, the lidar and each
planned board with its normal and number) and the overlay of the lidar's points on the camera's image through the
plan's or the solved extrinsic, coloured by range, round the board or over the whole image."""
from __future__ import annotations

import math

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap, QPolygonF
from PyQt6.QtWidgets import QToolTip, QWidget

from ..core import lidar_calib as LC
from .theme import THEME, mono, sans

LIDAR_COLOR = '#0aa5b8'              # the lidar's colour in the 3D and plan views (ui.sensorview.COLORS)
RANGE_BINS = 24


def to_qimage(img: np.ndarray) -> QImage:
    img = np.ascontiguousarray(img)
    h, w = img.shape[:2]
    if img.ndim == 2:
        return QImage(img.data, w, h, w, QImage.Format.Format_Grayscale8).copy()
    return QImage(img.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()


def _empty(p: QPainter, rect, text):
    p.setFont(sans(12))
    p.setPen(THEME.color('muted'))
    p.drawText(rect, Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap, text)


# ---------------------------------------------------------------------- the plan from above
class PlanView(QWidget):
    """Forward up, left to the left: the vehicle's outline, the camera (its colour, its heading), the lidar and every
    planned board (its outline seen from above, its normal, its number); the nearest and farthest distance as arcs
    round the camera. Hover a board for its figures, click it to pick it."""
    picked = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(240, 200)
        self.setMouseTracking(True)
        self.project = self.cam = self.lidar = None
        self.poses: list = []
        self.board = (0.9, 0.7)
        self.dist = (3.0, 8.0)
        self.empty = 'No plan yet'
        self.selected = -1
        self.hover = -1
        self._hits = []                  # (QPolygonF, index, tooltip)
        THEME.changed.connect(self._theme_changed)      # a bound slot: disconnected when the widget goes

    def _theme_changed(self, *_):
        self.update()

    def set_data(self, project, cam, lidar, poses, board, dist, empty=''):
        self.project, self.cam, self.lidar = project, cam, lidar
        self.poses, self.board, self.dist = list(poses or []), board, dist
        self.empty = empty or 'No plan yet'
        if self.selected >= len(self.poses):
            self.selected = -1
        self.update()

    def set_selected(self, k):
        if k != self.selected:
            self.selected = k
            self.update()

    def _frame(self):
        """(map vehicle xy -> screen, scale px/m)."""
        pts = [self.cam.pos()[:2], self.lidar.pos()[:2]]
        W, H = self.board
        for b in self.poses:
            pts += list((b.centre + LC.outline_local(W, H) @ b.R.T)[:, :2])
        meas = self.project.vehicle.measured() if self.project.vehicle.mesh is not None else {}
        if meas:
            hw = meas['Width'] / 2
            pts += [(meas['front_x'], hw), (meas['rear_x'], -hw)]
        P = np.array(pts, float)
        lo, hi = P.min(0) - 0.8, P.max(0) + 0.8
        s = min((self.width() - 24) / max(hi[1] - lo[1], 1e-3), (self.height() - 24) / max(hi[0] - lo[0], 1e-3))
        cx = self.width() / 2 + (hi[1] + lo[1]) / 2 * s
        cy = self.height() / 2 + (hi[0] + lo[0]) / 2 * s
        return (lambda x, y: QPointF(cx - y * s, cy - x * s)), s

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), THEME.color('viewport'))
        self._hits = []
        if self.project is None or self.cam is None or self.lidar is None:
            _empty(p, QRectF(self.rect()), self.empty)
            p.end()
            return
        to, s = self._frame()
        # the vehicle
        meas = self.project.vehicle.measured() if self.project.vehicle.mesh is not None else {}
        if meas:
            hw = meas['Width'] / 2
            a, b = to(meas['front_x'], hw), to(meas['rear_x'], -hw)
            p.setPen(QPen(THEME.color('strong'), 1.2))
            p.setBrush(THEME.color('raised'))
            p.drawRoundedRect(QRectF(a, b).normalized(), 0.3 * s, 0.3 * s)
        # nearest and farthest board distance round the camera
        C = self.cam.pos()
        head = math.radians(self.cam.yaw)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(THEME.color('muted'), 1, Qt.PenStyle.DashLine))
        for r in self.dist:
            arc = QPolygonF([to(C[0] + r * math.cos(head + a_), C[1] + r * math.sin(head + a_))
                             for a_ in np.radians(np.linspace(-85, 85, 60))])
            p.drawPolyline(arc)
            q = to(C[0] + r * math.cos(head + math.radians(88)), C[1] + r * math.sin(head + math.radians(88)))
            p.setFont(sans(10))
            p.drawText(q + QPointF(3, -3), f'{r:g} m')
        # the boards
        W, H = self.board
        for k, b in enumerate(self.poses):
            out = b.centre + LC.outline_local(W, H) @ b.R.T
            poly = QPolygonF([to(x, y) for x, y in out[:, :2]])
            sel = k in (self.selected, self.hover)
            p.setPen(QPen(THEME.color('ink' if sel else 'ink2'), 2.4 if sel else 1.6))
            p.setBrush(THEME.color('ink', 0.18 if sel else 0.08))
            p.drawPolygon(poly)
            c0 = to(b.centre[0], b.centre[1])
            n = b.R[:, 2]
            tip = to(b.centre[0] + n[0] * 0.5, b.centre[1] + n[1] * 0.5)
            p.setPen(QPen(THEME.color('muted'), 1))
            p.drawLine(c0, tip)
            p.setFont(mono(11, QFont.Weight.DemiBold))
            p.setPen(THEME.color('ink'))
            back = to(b.centre[0] - n[0] * 0.45, b.centre[1] - n[1] * 0.45)
            p.drawText(QRectF(back.x() - 10, back.y() - 8, 20, 16), Qt.AlignmentFlag.AlignCenter, str(k + 1))
            y_, pi, r_ = b.tilt
            tip_text = (f'Pose {k + 1}: {b.distance:.1f} m from {self.cam.tag}, centre {b.centre[2]:.2f} m high\n'
                        f'turned {y_:+g}° about its vertical, {pi:+g}° back, {r_:+g}° in its plane\n'
                        f'{b.rings} of {self.lidar.tag}\'s channels cross it')
            hit = QPolygonF(poly)
            hit.append(back)
            self._hits.append((hit.boundingRect().adjusted(-6, -6, 6, 6), k, tip_text))
        # the camera and the lidar
        cp = to(C[0], C[1])
        col = THEME.cam(self.cam.slot)
        p.setPen(QPen(col, 2))
        p.drawLine(cp, to(C[0] + 0.7 * math.cos(head), C[1] + 0.7 * math.sin(head)))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(col)
        p.drawRect(QRectF(cp.x() - 4, cp.y() - 4, 8, 8))
        L = self.lidar.pos()
        lp = to(L[0], L[1])
        p.setBrush(QColor(LIDAR_COLOR))
        p.drawEllipse(lp, 5, 5)
        p.setFont(mono(11, QFont.Weight.DemiBold))
        p.setPen(THEME.color('ink'))
        p.drawText(cp + QPointF(7, 14), self.cam.tag)
        p.drawText(lp + QPointF(7, 14), self.lidar.tag)
        p.end()

    def _hit(self, pos):
        return next(((k, t) for r, k, t in self._hits if r.contains(pos)), (-1, ''))

    def mouseMoveEvent(self, e):
        k, tip = self._hit(e.position())
        if k != self.hover:
            self.hover = k
            self.update()
        if k >= 0:
            QToolTip.showText(e.globalPosition().toPoint(), tip, self)
        else:
            QToolTip.hideText()

    def leaveEvent(self, e):
        self.hover = -1
        self.update()

    def mousePressEvent(self, e):
        k, _ = self._hit(e.position())
        if k >= 0:
            self.set_selected(k)
            self.picked.emit(k)


# ---------------------------------------------------------------------- the overlay
class OverlayView(QWidget):
    """One capture's image (grey, dimmed) with the lidar's points drawn through the plan's or the solved extrinsic,
    coloured by range (near warm, far blue), the board's outline as the camera measured it in white; round the board
    or over the whole image. The visual proof of the calibration."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(240, 200)
        self.project = self.result = None
        self.k = 0
        self.mode = 'solved'
        self.zoom = 'board'
        self.empty = 'Capture and solve to see the lidar\'s points in the camera image'
        self._pix = None
        self._pix_key = None
        self._pts = {}
        THEME.changed.connect(self._theme_changed)      # a bound slot: disconnected when the widget goes

    def _theme_changed(self, *_):
        self.update()

    def set_result(self, project, result, empty=''):
        self.project, self.result = project, result
        self.empty = empty or self.empty
        self._pix_key = None
        self._pts = {}
        if result is None or self.k >= len(result.captures):
            self.k = 0
        self.update()

    def set_pose(self, k):
        self.k = int(k)
        self.update()

    def set_mode(self, m):
        self.mode = m
        self.update()

    def set_zoom(self, z):
        self.zoom = z
        self.update()

    def _region(self, img):
        """(x0, y0, w, h) of the image shown."""
        if self.zoom == 'board':
            x0, y0, size = LC._board_box(self.result, self.k, img.shape)
            return x0, y0, size, size
        return 0, 0, img.shape[1], img.shape[0]

    def points(self, which):
        """(uv image pixels, range, reference range) of the capture's lidar points, cached per result and pose."""
        key = (id(self.result), self.k, which)
        if key not in self._pts:
            uv, rng, _ = LC.project_points(self.project, self.result, self.k, which)
            o = self.result.poses[self.k] if self.k < len(self.result.poses) else None
            ref = float(np.linalg.norm(o.lid.points.mean(0))) if o is not None and o.lid is not None else \
                (float(np.median(rng)) if len(rng) else 5.0)
            self._pts[key] = (uv, rng, ref)
        return self._pts[key]

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), THEME.color('viewport'))
        res = self.result
        cap = res.captures[self.k] if res is not None and self.k < len(res.captures) else None
        if cap is None or cap.image is None:
            _empty(p, QRectF(self.rect()).adjusted(20, 0, -20, 0), self.empty if cap is None else
                   f'{cap.name}: {cap.error or "no image"}')
            p.end()
            return
        img = cap.image
        x0, y0, rw, rh = self._region(img)
        top, bottom = 22, 26
        avail = QRectF(6, top, self.width() - 12, self.height() - top - bottom)
        s = min(avail.width() / rw, avail.height() / rh)
        rect = QRectF(avail.center().x() - rw * s / 2, avail.center().y() - rh * s / 2, rw * s, rh * s)
        key = (id(res), self.k, x0, y0, rw, rh, int(rect.width()), int(rect.height()))
        if key != self._pix_key:
            g = LC._gray(img)[y0:y0 + rh, x0:x0 + rw]
            g = (g.astype(np.float32) * 0.6).astype(np.uint8)
            self._pix = QPixmap.fromImage(to_qimage(g)).scaled(int(rect.width()), int(rect.height()),
                                                               Qt.AspectRatioMode.IgnoreAspectRatio,
                                                               Qt.TransformationMode.SmoothTransformation)
            self._pix_key = key
        p.drawPixmap(rect.topLeft(), self._pix)
        which = 'solved' if self.mode == 'solved' and res.T is not None else 'plan'
        uv, rng, ref = self.points(which)
        lo, hi = ref - 2.0, ref + 4.0
        if len(rng):
            q = np.c_[rect.left() + (uv[:, 0] - x0) * s, rect.top() + (uv[:, 1] - y0) * s]
            inside = (q[:, 0] >= rect.left()) & (q[:, 0] <= rect.right()) & (q[:, 1] >= rect.top()) & \
                (q[:, 1] <= rect.bottom())
            q, rr = q[inside], rng[inside]
            t = np.clip((rr - lo) / (hi - lo), 0, 1)
            bins = np.minimum((t * RANGE_BINS).astype(int), RANGE_BINS - 1)
            cols = LC.range_colors(lo + (np.arange(RANGE_BINS) + 0.5) / RANGE_BINS * (hi - lo), lo, hi)
            size = 3.0 if self.zoom == 'board' else 2.0
            for b in np.unique(bins)[::-1]:            # far first, near on top
                sel = q[bins == b]
                c = cols[b]
                pen = QPen(QColor(int(c[0]), int(c[1]), int(c[2])), size)
                pen.setCapStyle(Qt.PenCapStyle.SquareCap)
                p.setPen(pen)
                p.drawPoints(QPolygonF([QPointF(x, y) for x, y in sel]))
        o = res.poses[self.k] if self.k < len(res.poses) else None
        if o is not None and o.cam is not None:
            W, H = res.settings.board_size()
            out = o.cam.c + LC.outline_local(W, H, 12) @ o.cam.R.T
            cam = next((c for v in self.project.variants for c in v.cameras if c.uid == res.camera_uid), None)
            if cam is not None:
                from ..core.lens import resolve
                intr = resolve(self.project.module_for(cam), self.project.lens_for(cam))
                u, v, _, ok = intr.project_dirs(out)
                sx, sy = img.shape[1] / intr.width, img.shape[0] / intr.height
                poly = QPolygonF([QPointF(rect.left() + (a * sx - x0) * s, rect.top() + (b * sy - y0) * s)
                                  for a, b in zip(u, v)])
                p.setPen(QPen(QColor(255, 255, 255, 220), 1.2))
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawPolygon(poly)
        # caption and the range scale
        p.setFont(sans(11, QFont.Weight.DemiBold))
        p.setPen(THEME.color('ink'))
        word = {'plan': 'Plan', 'solved': 'Solved'}[which]
        status = o.status.replace('_', ' ') if o is not None else ''
        p.drawText(QRectF(8, 0, self.width() - 16, top), Qt.AlignmentFlag.AlignVCenter,
                   f'{word} · {cap.name} · {res.lidar_tag} in {res.camera_tag}' + (f' · {status}' if status else ''))
        bar = QRectF(self.width() - 186, self.height() - bottom + 8, 120, 8)
        grad = LC.range_colors(np.linspace(lo, hi, 60), lo, hi)
        for i, c in enumerate(grad):
            p.fillRect(QRectF(bar.left() + i * bar.width() / 60, bar.top(), bar.width() / 60 + 0.5, bar.height()),
                       QColor(int(c[0]), int(c[1]), int(c[2])))
        p.setFont(sans(10))
        p.setPen(THEME.color('muted'))
        p.drawText(QRectF(bar.left() - 70, bar.top() - 4, 66, 16), Qt.AlignmentFlag.AlignRight |
                   Qt.AlignmentFlag.AlignVCenter, f'{max(lo, 0):.0f} m')
        p.drawText(QRectF(bar.right() + 4, bar.top() - 4, 60, 16), Qt.AlignmentFlag.AlignVCenter, f'{hi:.0f} m')
        p.end()
