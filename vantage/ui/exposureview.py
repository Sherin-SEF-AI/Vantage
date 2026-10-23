"""Sun glare and dirt exposure charts: the month × hour share of driving time with the sun in a camera's image, the
sky in the vehicle frame (the camera's field, what the body hides, where the sun is), the sun's path through the
image on the solstices and the equinox, the dirt map of the body in four views, and the route's heading rose."""
from __future__ import annotations

import math

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap, QPolygonF
from PyQt6.QtWidgets import QToolTip

from ..core import exposure as X
from .charts import ChartBase
from .theme import STATUS, THEME, mono, sans

DAY_PEN = {0: Qt.PenStyle.DashLine, 1: Qt.PenStyle.SolidLine, 2: Qt.PenStyle.DotLine}     # equinox, June, December
SUN_LINE = '#ee861f'                 # the sun's path (one hue; the days differ by dash and label)


def ramp_color(colors, t, alpha=None) -> QColor:
    r, g, b = X.ramp(colors, t)
    c = QColor(int(r), int(g), int(b))
    if alpha is not None:
        c.setAlphaF(alpha)
    return c


def item_color(it) -> QColor:
    if it.kind == 'camera':
        return THEME.cam(it.slot)
    return QColor({'lidar': '#0aa5b8', 'radar': '#a0743c'}.get(it.kind, '#0aa5b8'))


def _empty(p, w, h, text):
    p.setFont(sans(11))
    p.setPen(THEME.color('muted'))
    p.drawText(QRectF(0, 24, w, h - 24), Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap, text)


# =========================================================================== month × hour heatmap
class SunHeatmap(ChartBase):
    """Rows: months; columns: the hours of the driving window; a cell's colour is the share of driving time in that
    month and hour with the sun in the image (a lidar: its field), on one scale for all cameras."""
    LEFT, TOP, BOTTOM = 32, 24, 34

    def __init__(self, parent=None):
        super().__init__(parent)
        self.title = 'Sun in the image by month and hour'
        self.item = None
        self.cols = []
        self.vmax = 0.05
        self._cell = (0.0, 0.0)
        self.setMinimumSize(220, 220)

    def set_data(self, item, cols, vmax):
        self.item, self.cols, self.vmax = item, list(cols), max(float(vmax), 1e-6)
        if item is not None:
            self.title = 'Sun in the ' + ('image' if item.kind == 'camera' else 'field') + ' by month and hour'
        self.update()

    def _geom(self):
        n = max(len(self.cols), 1)
        cw = (self.width() - self.LEFT - 4) / n
        ch = (self.height() - self.TOP - self.BOTTOM) / 12
        return cw, ch

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        it = self.item
        self.paint_title(p, f'{it.tag} · {100 * it.sun["share"]:.1f} %' if it is not None and it.sun else '')
        if it is None or it.sun is None:
            _empty(p, self.width(), self.height(), 'Radars do not see sunlight.' if it is not None else
                   'Pick a camera or lidar in the table.')
            p.end()
            return
        cw, ch = self._geom()
        mh = it.sun['month_hour']
        p.setFont(mono(10))
        for r in range(12):
            y = self.TOP + r * ch
            p.setPen(THEME.color('muted'))
            p.drawText(QRectF(0, y, self.LEFT - 4, ch), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                       X.MONTHS[r])
            for c, h in enumerate(self.cols):
                v = mh[r, h]
                if not math.isfinite(v):
                    continue
                rect = QRectF(self.LEFT + c * cw + 1, y + 1, max(cw - 2, 1), max(ch - 2, 1))
                if v > 0:
                    p.fillRect(rect, ramp_color(X.SUN_RAMP, v / self.vmax))
                else:
                    p.fillRect(rect, THEME.color('raised'))
                if self.hover == (r, c):
                    p.setPen(QPen(THEME.color('ink'), 1.5))
                    p.setBrush(Qt.BrushStyle.NoBrush)
                    p.drawRect(rect)
        p.setPen(THEME.color('muted'))
        yb = self.TOP + 12 * ch
        step = 1 if cw >= 22 else 2 if cw >= 11 else 4
        for c, h in enumerate(self.cols):
            if c % step == 0:
                p.drawText(QRectF(self.LEFT + c * cw - 4, yb + 1, cw + 8, 14), Qt.AlignmentFlag.AlignCenter,
                           f'{h:02d}')
        # the scale
        x0, y0, w = self.LEFT, yb + 18, min(140.0, self.width() - self.LEFT - 90)
        for k in range(28):
            p.fillRect(QRectF(x0 + k * w / 28, y0, w / 28 + 0.5, 8), ramp_color(X.SUN_RAMP, k / 27))
        p.setFont(sans(10))
        p.drawText(QRectF(x0 + w + 6, y0 - 4, 120, 16), Qt.AlignmentFlag.AlignVCenter,
                   f'0 – {100 * self.vmax:.0f} % of the time')
        p.end()

    def _hit(self, pos):
        if self.item is None or self.item.sun is None:
            return None
        cw, ch = self._geom()
        c = int((pos.x() - self.LEFT) // cw) if cw > 0 else -1
        r = int((pos.y() - self.TOP) // ch) if ch > 0 else -1
        return (r, c) if 0 <= r < 12 and 0 <= c < len(self.cols) else None

    def mouseMoveEvent(self, e):
        hit = self._hit(e.position())
        if hit != self.hover:
            self.hover = hit
            self.update()
        if hit is None:
            QToolTip.hideText()
            return
        r, c = hit
        h = self.cols[c]
        v = self.item.sun['month_hour'][r, h]
        word = 'image' if self.item.kind == 'camera' else 'field'
        text = (f'{X.MONTHS[r]}, {h:02d}–{(h + 1) % 24:02d} h: ' +
                (f'the sun is in the {word} {100 * v:.1f} % of the driving time' if math.isfinite(v)
                 else 'no driving'))
        QToolTip.showText(e.globalPosition().toPoint(), text, self)


# =========================================================================== sky chart
class SkyChart(ChartBase):
    """The sky in the vehicle frame: zenith in the middle, the horizon at the edge, forward up and left to the left.
    Where the sun is over the campaign (hours of driving per 2° cell), the directions the body hides from the camera
    and the outline of its field."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.title = 'Sky seen from the camera'
        self.res = None
        self.item = None
        self._img = None
        self._key = None
        self.setMinimumSize(220, 220)
        THEME.changed.connect(self._drop)

    def _drop(self, *_):
        self._key = None
        self.update()

    def set_data(self, res, item):
        self.res, self.item = res, item
        self._key = None
        self.update()

    def _disc(self):
        top = 22.0
        size = min(self.width() - 90.0, self.height() - top - 44.0)
        R = max(size / 2, 10.0)
        return QPointF(self.width() / 2, top + 14 + R), R

    def _render(self, n):
        """(n, n) RGBA image of the disc: the sun density, the hidden directions and the field's tint."""
        res, it = self.res, self.item
        yy, xx = np.mgrid[0:n, 0:n] + 0.5
        dx, dy = (xx - n / 2) / (n / 2), (yy - n / 2) / (n / 2)
        r = np.hypot(dx, dy)
        el = 90.0 * (1 - r)
        az = np.degrees(np.arctan2(-dx, -dy))            # forward up, left to the left
        inside = r <= 1.0
        ie = np.clip((el / X.SKY_STEP).astype(int), 0, len(X.SKY_EL) - 1)
        ia = np.clip(((az + 180.0) / X.SKY_STEP).astype(int), 0, len(X.SKY_AZ) - 1)
        out = np.zeros((n, n, 4), np.float32)
        base = THEME.color('raised')
        out[inside] = (base.red(), base.green(), base.blue(), 255)
        dens = res.sky_density
        if dens.size:                    # drawn softened (5 cells round in azimuth, 3 in elevation): the sampled days
            dens = sum(np.roll(dens, s, axis=1) for s in range(-2, 3)) / 5.0         # and times leave speckle
            pad = np.pad(dens, ((1, 1), (0, 0)), mode='edge')
            dens = (pad[:-2] + pad[1:-1] + pad[2:]) / 3.0
        top = float(dens.max()) if dens.size else 0.0
        if top > 0:
            t = np.sqrt(dens[ie, ia] / top)
            col = X.ramp(X.SUN_RAMP, t)
            a = np.where(inside & (dens[ie, ia] > 0), 0.15 + 0.6 * t, 0.0)[..., None]
            out[..., :3] = out[..., :3] * (1 - a) + col * a
        if it is not None and it.sky_margin is not None:
            tint = item_color(it)
            m = it.sky_margin[ie, ia] > 0
            a = np.where(inside & m, 0.16, 0.0)[..., None]
            out[..., :3] = out[..., :3] * (1 - a) + np.array([tint.red(), tint.green(), tint.blue()]) * a
            hid = it.sky_blocked[ie, ia] & inside
            stripe = (((xx + yy) // 4) % 2 == 0)
            g = THEME.color('strong')
            a = np.where(hid, np.where(stripe, 0.85, 0.55), 0.0)[..., None]
            out[..., :3] = out[..., :3] * (1 - a) + np.array([g.red(), g.green(), g.blue()]) * a
        rgba = np.ascontiguousarray(np.clip(out, 0, 255).astype(np.uint8))
        img = QImage(rgba.data, n, n, 4 * n, QImage.Format.Format_RGBA8888).copy()
        return img

    def _pt(self, c, R, az, el):
        r = R * (1 - el / 90.0)
        a = math.radians(az)
        return QPointF(c.x() - math.sin(a) * r, c.y() - math.cos(a) * r)

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        it = self.item
        self.paint_title(p, it.tag if it is not None else '')
        if self.res is None or it is None or it.sky_margin is None:
            _empty(p, self.width(), self.height(), 'Radars do not see sunlight.' if it is not None else
                   'Run the study and pick a camera or lidar.')
            p.end()
            return
        c, R = self._disc()
        n = max(int(2 * R), 8)
        key = (id(self.res), it.uid, n, THEME.name)
        if self._key != key:
            self._img, self._key = self._render(n), key
        p.drawImage(QRectF(c.x() - R, c.y() - R, 2 * R, 2 * R), self._img)
        p.setPen(QPen(THEME.color('hair'), 1))
        p.setBrush(Qt.BrushStyle.NoBrush)
        for el in (30.0, 60.0):
            rr = R * (1 - el / 90.0)
            p.drawEllipse(c, rr, rr)
        for a in range(0, 360, 45):
            p.drawLine(c, self._pt(c, R, a, 0.0))
        p.setPen(QPen(THEME.color('strong'), 1))
        p.drawEllipse(c, R, R)
        # the field's outline, on a halo of the panel colour so it stands off the sun's colours
        segs = [(self._pt(c, R, s[0][0], s[0][1]), self._pt(c, R, s[1][0], s[1][1]))
                for s in (it.outline if it.outline is not None else [])]
        for pen in (QPen(THEME.color('panel'), 4.0), QPen(item_color(it), 1.8)):
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            p.setPen(pen)
            for a, b in segs:
                p.drawLine(a, b)
        p.setFont(sans(10))
        p.setPen(THEME.color('muted'))
        fm = p.fontMetrics()
        for text, a in (('Forward', 0.0), ('Left', 90.0), ('Rear', 180.0), ('Right', -90.0)):
            q = self._pt(c, R, a, 0.0)
            w = fm.horizontalAdvance(text)
            if a == 90.0:
                box = QRectF(q.x() - w - 8, q.y() - 7, w + 4, 14)
            elif a == -90.0:
                box = QRectF(q.x() + 4, q.y() - 7, w + 4, 14)
            else:
                box = QRectF(q.x() - w / 2 - 2, q.y() + (-17 if a == 0.0 else 3), w + 4, 14)
            p.drawText(box, Qt.AlignmentFlag.AlignCenter, text)
        for el in (30, 60):
            q = self._pt(c, R, 45.0, el)
            p.drawText(QRectF(q.x() - 22, q.y() - 7, 22, 14), Qt.AlignmentFlag.AlignRight, f'{el}°')
        # legend
        y = self.height() - 16.0
        x = 4.0
        for k in range(16):
            p.fillRect(QRectF(x + k * 4, y + 3, 4.5, 8), ramp_color(X.SUN_RAMP, k / 15))
        p.drawText(QRectF(x + 68, y, 60, 14), Qt.AlignmentFlag.AlignVCenter, 'sun')
        x += 100
        p.fillRect(QRectF(x, y + 3, 10, 8), THEME.color('strong'))
        p.drawText(QRectF(x + 14, y, 80, 14), Qt.AlignmentFlag.AlignVCenter, 'body')
        x += 54
        p.setPen(QPen(item_color(it), 1.6))
        p.drawLine(QPointF(x, y + 7), QPointF(x + 12, y + 7))
        p.setPen(THEME.color('muted'))
        p.drawText(QRectF(x + 16, y, 80, 14), Qt.AlignmentFlag.AlignVCenter, 'field')
        p.end()

    def mouseMoveEvent(self, e):
        if self.res is None or self.item is None or self.item.sky_margin is None:
            return
        c, R = self._disc()
        dx, dy = (e.position().x() - c.x()) / R, (e.position().y() - c.y()) / R
        r = math.hypot(dx, dy)
        if r > 1.0:
            QToolTip.hideText()
            return
        el = 90.0 * (1 - r)
        az = math.degrees(math.atan2(-dx, -dy))
        ie = min(int(el / X.SKY_STEP), len(X.SKY_EL) - 1)
        ia = min(int((az + 180.0) / X.SKY_STEP), len(X.SKY_AZ) - 1)
        hours = float(self.res.sky_density[ie, ia]) * self.res.metrics['driving_hours']
        side = 'ahead' if abs(az) < 1 else 'behind' if abs(az) > 179 else (f'{abs(az):.0f}° to the '
                                                                          + ('left' if az > 0 else 'right'))
        it = self.item
        what = 'in the image' if it.kind == 'camera' else 'in the field'
        state = ('hidden by the body' if it.sky_blocked[ie, ia] else what if it.sky_margin[ie, ia] > 0
                 else 'outside the ' + ('image' if it.kind == 'camera' else 'field'))
        QToolTip.showText(e.globalPosition().toPoint(),
                          f'{side} of forward, {el:.0f}° up: {state}\nthe sun is here {hours:.1f} h of '
                          f'{self.res.metrics["driving_hours"]:,.0f} h of driving', self)


# =========================================================================== sun tracks in the image
class ImageTracks(ChartBase):
    """The camera's image (sensor rectangle, image circle, where the body covers it, the horizon) with the sun's path
    across it on the equinox and the solstices at one heading; faint where the body hides the sun or outside the
    driving window, a dot with the hour on each full hour."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.title = 'Sun\'s path through the image'
        self.item = None
        self.key = 'worst'
        self._pts = []                       # (QPointF, text) for hover
        self.setMinimumSize(240, 200)

    def set_data(self, item, key='worst'):
        self.item, self.key = item, key
        self.update()

    def tracks(self):
        it = self.item
        if it is None or not it.tracks:
            return []
        return it.tracks.get(self.key) or it.tracks.get('0') or []

    def _frame(self):
        im = self.item.image
        top, bottom = 40.0, 6.0
        s = min((self.width() - 8) / im['w'], (self.height() - top - bottom) / im['h'])
        x0 = (self.width() - im['w'] * s) / 2
        return x0, top, s

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        it = self.item
        trk = self.tracks()
        head = trk[0]['heading'] if trk else None
        self.paint_title(p, f'driving {head:.0f}° ({X.compass(head)})' if head is not None else '')
        self._pts = []
        if it is None or it.image is None:
            _empty(p, self.width(), self.height(), 'Pick a camera in the table.' if it is None else
                   'The sun\'s path is drawn through a camera\'s image: pick a camera.')
            p.end()
            return
        im = it.image
        x0, y0, s = self._frame()
        frame = QRectF(x0, y0, im['w'] * s, im['h'] * s)
        p.fillRect(frame, THEME.color('raised'))
        # the body over the image
        body = im['body']
        if body.any():
            g = THEME.color('strong')
            rgba = np.zeros(body.shape + (4,), np.uint8)
            rgba[body] = (g.red(), g.green(), g.blue(), 200)
            q = QImage(np.ascontiguousarray(rgba).data, body.shape[1], body.shape[0], 4 * body.shape[1],
                       QImage.Format.Format_RGBA8888).copy()
            p.drawImage(frame, q)
        p.save()
        p.setClipRect(frame)
        p.setPen(QPen(THEME.color('muted'), 1, Qt.PenStyle.DashLine))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(QPointF(x0 + im['cx'] * s, y0 + im['cy'] * s), im['rx'] * s, im['ry'] * s)
        p.setPen(QPen(THEME.color('ink2'), 1, Qt.PenStyle.DashDotLine))
        for ln in im['horizon']:
            p.drawPolyline(QPolygonF([QPointF(x0 + u * s, y0 + v * s) for u, v in ln]))
        sun = QColor(SUN_LINE)
        for k, t in enumerate(trk):
            ok = t['ok']
            n = len(ok)
            for i in range(n - 1):
                if not (ok[i] and ok[i + 1]) or t['hour'][i + 1] - t['hour'][i] > 0.2:
                    continue
                strong = t['in_window'][i] and t['in_window'][i + 1] and not (t['hidden'][i] or t['hidden'][i + 1])
                c = QColor(sun)
                c.setAlphaF(1.0 if strong else 0.35)
                p.setPen(QPen(c, 2.0 if strong else 1.2, DAY_PEN.get(k, Qt.PenStyle.SolidLine),
                              Qt.PenCapStyle.RoundCap))
                p.drawLine(QPointF(x0 + t['u'][i] * s, y0 + t['v'][i] * s),
                           QPointF(x0 + t['u'][i + 1] * s, y0 + t['v'][i + 1] * s))
            p.setFont(mono(9))
            for i in range(n):
                if not ok[i]:
                    continue
                q = QPointF(x0 + t['u'][i] * s, y0 + t['v'][i] * s)
                hr = t['hour'][i]
                state = ('hidden by the body' if t['hidden'][i] else 'outside the driving window'
                         if not t['in_window'][i] else 'in the image')
                self._pts.append((q, f'{t["label"]} · {int(hr):02d}:{int(round((hr % 1) * 60)) % 60:02d} · sun '
                                     f'{t["el"][i]:.1f}° up · {state}'))
                if abs(hr - round(hr)) < 1e-6 and int(round(hr)) % 2 == 0:
                    p.setPen(Qt.PenStyle.NoPen)
                    p.setBrush(sun if t['in_window'][i] and not t['hidden'][i] else THEME.color('muted'))
                    p.drawEllipse(q, 2.5, 2.5)
                    p.setPen(THEME.color('ink2'))
                    p.drawText(QRectF(q.x() + 3, q.y() - 13, 30, 12), Qt.AlignmentFlag.AlignLeft,
                               f'{int(round(hr))}')
        p.restore()
        p.setPen(QPen(THEME.color('hair'), 1))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRect(frame)
        # legend: one entry per day
        p.setFont(sans(10))
        x = 0.0
        for k, t in enumerate(trk):
            c = QColor(sun)
            p.setPen(QPen(c, 2.0, DAY_PEN.get(k, Qt.PenStyle.SolidLine)))
            p.drawLine(QPointF(x, 28), QPointF(x + 18, 28))
            p.setPen(THEME.color('ink2'))
            w = p.fontMetrics().horizontalAdvance(t['label'])
            p.drawText(QRectF(x + 22, 21, w + 4, 14), Qt.AlignmentFlag.AlignVCenter, t['label'])
            x += w + 36
        if not any(t['ok'].any() for t in trk):
            p.setPen(THEME.color('muted'))
            p.drawText(frame, Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap,
                       'The sun does not cross the image on these days at this heading.')
        p.end()

    def mouseMoveEvent(self, e):
        pos = e.position()
        best = min(self._pts, key=lambda qt: (qt[0].x() - pos.x()) ** 2 + (qt[0].y() - pos.y()) ** 2, default=None)
        if best is not None and (best[0].x() - pos.x()) ** 2 + (best[0].y() - pos.y()) ** 2 <= 64:
            QToolTip.showText(e.globalPosition().toPoint(), best[1], self)
        else:
            QToolTip.hideText()


# =========================================================================== dirt map
class DirtMap(ChartBase):
    """The body's dirt and spray score per face in four views (left side, top, front, rear) on one scale, the cameras
    and sensors marked in their class's status colour; hover a marker for its factors, click it to pick it."""
    picked = pyqtSignal(str)
    LAYOUT = (('side', 0.0, 0.0, 0.62, 0.5), ('front', 0.0, 0.5, 0.31, 0.5), ('rear', 0.31, 0.5, 0.31, 0.5),
              ('top', 0.62, 0.0, 0.38, 1.0))

    def __init__(self, parent=None):
        super().__init__(parent)
        self.title = 'Dirt and spray exposure (heuristic)'
        self.res = None
        self.selected = ''
        self._pix = None
        self._key = None
        self._marks = []                     # (QPointF, uid, tooltip)
        self._vcache = (None, None)
        self.setMinimumSize(360, 220)
        THEME.changed.connect(self._drop)

    def _drop(self, *_):
        self._key = None
        self.update()

    def set_data(self, res, selected=''):
        if res is not self.res:
            self._key = None
        self.res, self.selected = res, selected
        self.update()

    def _views(self):
        """[(name, rect, scale, (mid x, mid y))] with one scale for every view."""
        key = (id(self.res), self.width(), self.height())
        if self._vcache[0] != key:
            self._vcache = (key, self._layout())
        return self._vcache[1]

    def _layout(self):
        res = self.res
        top, legend = 22.0, 20.0
        W, H = float(self.width()), float(self.height()) - top - legend
        out, polys, scale = [], {}, math.inf
        for name, fx, fy, fw, fh in self.LAYOUT:
            poly = X.view_faces(res, name)[0]
            polys[name] = poly
            rect = QRectF(fx * W + 4, top + fy * H + 14, fw * W - 8, fh * H - 18)
            if len(poly):
                scale = min(scale, rect.width() / max(np.ptp(poly[..., 0]), 1e-6),
                            rect.height() / max(np.ptp(poly[..., 1]), 1e-6))
            out.append((name, rect))
        scale = 50.0 if not math.isfinite(scale) else scale
        res_out = []
        for name, rect in out:
            poly = polys[name]
            mid = ((poly[..., 0].min() + poly[..., 0].max()) / 2, (poly[..., 1].min() + poly[..., 1].max()) / 2) \
                if len(poly) else (0.0, 0.0)
            res_out.append((name, rect, scale, mid))
        return res_out

    @staticmethod
    def _map(rect, scale, mid, sx, sy):
        return (rect.center().x() + (sx - mid[0]) * scale, rect.center().y() - (sy - mid[1]) * scale)

    def _render(self):
        dpr = self.devicePixelRatioF()
        pm = QPixmap(int(self.width() * dpr), int(self.height() * dpr))
        pm.setDevicePixelRatio(dpr)
        pm.fill(Qt.GlobalColor.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        wheel = THEME.color('strong')
        titles = dict((n, t) for n, t, _ in X.DIRT_VIEWS)
        for name, rect, scale, mid in self._views():
            p.setFont(sans(10))
            p.setPen(THEME.color('muted'))
            p.drawText(QRectF(rect.left(), rect.top() - 14, rect.width(), 14), Qt.AlignmentFlag.AlignLeft, titles[name])
            poly, sc, glass, shade = X.view_faces(self.res, name)
            if not len(poly):
                continue
            px, py = self._map(rect, scale, mid, poly[..., 0], poly[..., 1])
            col = X.ramp(X.DIRT_RAMP, np.where(np.isfinite(sc), sc, 0.0)) * shade[:, None]
            p.setPen(Qt.PenStyle.NoPen)
            for j in range(len(poly)):
                if math.isfinite(sc[j]):
                    c = QColor(int(col[j, 0]), int(col[j, 1]), int(col[j, 2]))
                else:
                    c = QColor(wheel)
                p.setBrush(c)
                p.drawPolygon(QPolygonF([QPointF(px[j, 0], py[j, 0]), QPointF(px[j, 1], py[j, 1]),
                                         QPointF(px[j, 2], py[j, 2])]))
        p.end()
        return pm

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.paint_title(p, f'classes at {X.DIRT_LIMITS[0]:g} and {X.DIRT_LIMITS[1]:g}')
        self._marks = []
        if self.res is None or not len(self.res.dirt_faces['tri']):
            _empty(p, self.width(), self.height(), 'No vehicle body: the dirt map needs one.' if self.res else
                   'Run the study.')
            p.end()
            return
        key = (id(self.res), self.width(), self.height(), THEME.name, self.devicePixelRatioF())
        if self._key != key:
            self._pix, self._key = self._render(), key
        p.drawPixmap(0, 0, self._pix)
        p.setFont(mono(10, QFont.Weight.DemiBold))
        fm = p.fontMetrics()
        fns = dict((n, f) for n, _, f in X.DIRT_VIEWS)
        for name, rect, scale, mid in self._views():
            boxes = []                                   # labels placed so far in this view
            for it in self.res.items:
                if not X.marker_shown(self.res, name, it.pos):
                    continue
                sx, sy, _ = fns[name](np.array(it.pos, float))
                x, y = self._map(rect, scale, mid, float(sx), float(sy))
                q = QPointF(x, y)
                sel = it.uid == self.selected
                p.setPen(QPen(THEME.color('ink') if sel else THEME.color('base'), 2.0 if sel else 1.0))
                p.setBrush(QColor(STATUS[X.CLASS_TONE[it.dirt['class']]]))
                p.drawEllipse(q, 5.5 if sel else 4.0, 5.5 if sel else 4.0)
                p.setPen(THEME.color('ink'))
                w = fm.horizontalAdvance(it.tag) + 2
                for dy in (-15, 2, -28, 15, -41):           # the first free slot above, below, further out
                    box = QRectF(x + 6, y + dy, w, 13)
                    if not any(box.intersects(b) for b in boxes):
                        break
                boxes.append(box)
                p.drawText(box, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, it.tag)
                d = it.dirt
                tip = (f'{it.tag}: dirt {d["score"]:.2f} ({d["class"]}), mostly {X.DIRT_LABEL[d["dominant"]]}'
                       + (f' on its {d["facet"]}' if d.get('facet') else '') + '\n'
                       + ' · '.join(f'{X.DIRT_LABEL[k]} {v:.2f}' for k, v in d['factors'].items()) + '\n'
                       + d['advice'])
                self._marks.append((q, it.uid, tip))
        # the scale
        y = self.height() - 14.0
        p.setFont(sans(10))
        for k in range(28):
            p.fillRect(QRectF(4 + k * 4, y, 4.5, 8), ramp_color(X.DIRT_RAMP, k / 27))
        p.setPen(THEME.color('muted'))
        p.drawText(QRectF(122, y - 4, 400, 16), Qt.AlignmentFlag.AlignVCenter,
                   'score 0 – 1 · grey: wheels · markers: low, moderate, high')
        p.end()

    def _mark_at(self, pos):
        return min(((q, uid, tip) for q, uid, tip in self._marks
                    if (q.x() - pos.x()) ** 2 + (q.y() - pos.y()) ** 2 <= 64),
                   key=lambda m: (m[0].x() - pos.x()) ** 2 + (m[0].y() - pos.y()) ** 2, default=None)

    def mouseMoveEvent(self, e):
        m = self._mark_at(e.position())
        if m is not None:
            QToolTip.showText(e.globalPosition().toPoint(), m[2], self)
            self.setCursor(Qt.CursorShape.PointingHandCursor)
        else:
            QToolTip.hideText()
            self.unsetCursor()

    def mousePressEvent(self, e):
        m = self._mark_at(e.position())
        if m is not None:
            self.picked.emit(m[1])


# =========================================================================== heading rose
class HeadingRose(ChartBase):
    """The route's heading distribution: a bar per heading, north up, east to the right."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.table = (np.zeros(0), np.zeros(0))
        self.uniform = True
        self.setFixedSize(96, 96)

    def set_data(self, headings, weights, uniform):
        self.table, self.uniform = (np.asarray(headings, float), np.asarray(weights, float)), uniform
        self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        c = QPointF(self.width() / 2, self.height() / 2)
        R = min(self.width(), self.height()) / 2 - 12
        p.setPen(QPen(THEME.color('hair'), 1))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(c, R, R)
        p.setFont(sans(9))
        p.setPen(THEME.color('muted'))
        for t, a in (('N', 0), ('E', 90), ('S', 180), ('W', 270)):
            q = QPointF(c.x() + math.sin(math.radians(a)) * (R + 6), c.y() - math.cos(math.radians(a)) * (R + 6))
            p.drawText(QRectF(q.x() - 6, q.y() - 6, 12, 12), Qt.AlignmentFlag.AlignCenter, t)
        h, w = self.table
        if not len(w) or w.max() <= 0:
            p.end()
            return
        col = THEME.color('ink2')
        p.setPen(QPen(col, 2.0 if len(h) <= 72 else 1.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        for hh, ww in zip(h, w / w.max()):
            a = math.radians(hh)
            r = R * (0.15 + 0.85 * ww)
            p.drawLine(c, QPointF(c.x() + math.sin(a) * r, c.y() - math.cos(a) * r))
        p.end()

    def mouseMoveEvent(self, e):
        h, w = self.table
        if not len(w):
            return
        c = QPointF(self.width() / 2, self.height() / 2)
        a = math.degrees(math.atan2(e.position().x() - c.x(), c.y() - e.position().y())) % 360.0
        k = int(np.argmin(np.abs((h - a + 180.0) % 360.0 - 180.0)))
        QToolTip.showText(e.globalPosition().toPoint(), 'Every heading equally likely' if self.uniform else
                          f'{h[k]:.0f}° ({X.compass(h[k])}): {100 * w[k]:.1f} % of the driving time', self)
