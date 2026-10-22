"""Sync and timing in the UI: the dial of one lidar rotation. Drawn forward up, left to the left: each point of the
circle is the direction the reference lidar faces at that time after the sync pulse, so the ms ticks run round it in
the sweep's direction. One track per camera shows its field as the sweep crosses it, the crossing, and the frame
matched to it (first to last row, the middle row's exposure centre); the trigger pulses sit on the rim with their line
letters, the vehicle and the cameras in the middle."""
from __future__ import annotations

import math

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QFont, QPainter, QPen, QPolygonF
from PyQt6.QtWidgets import QToolTip

from ..core import timing as TM
from .charts import ChartBase
from .theme import THEME, mono, sans


def _ms(v, dp=1):
    return '—' if v is None else f'{v:.{dp}f}'


class TimingDial(ChartBase):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.res: TM.TimingResult | None = None
        self.setMinimumSize(340, 340)

    def set_result(self, res):
        self.res = res
        self.update()

    # ---- geometry
    def _geom(self):
        r = QRectF(self.rect()).adjusted(10, 26, -10, -22)
        R = max(min(r.width(), r.height()) / 2 - 34, 60.0)
        return r.center(), R

    def _at(self, c, t, r) -> QPointF:
        a = math.radians(float(self.res.lidar.direction(t)))
        return QPointF(c.x() - math.sin(a) * r, c.y() - math.cos(a) * r)

    def _arc(self, c, t0, t1, r) -> QPolygonF:
        T = self.res.lidar.period_ms
        n = max(int(abs(t1 - t0) / T * 360) + 2, 2)
        return QPolygonF([self._at(c, t0 + (t1 - t0) * i / (n - 1), r) for i in range(n)])

    def _hub(self, c, lay, x, y) -> QPointF:
        lid = self.res.lidar
        k = lay['scale']
        return QPointF(c.x() - (y - lid.pos[1]) * k, c.y() - (x - lid.pos[0]) * k)

    # ---- painting
    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), THEME.color('viewport'))
        res = self.res
        if res is None or res.lidar is None:
            p.setPen(THEME.color('muted'))
            p.setFont(sans(12))
            text = 'Computing the plan…' if res is None else res.reason
            p.drawText(QRectF(self.rect()).adjusted(40, 0, -40, 0),
                       Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap, text)
            return
        lid, st = res.lidar, res.settings
        T = lid.period_ms
        c, R = self._geom()
        lay = TM.dial_layout(res, R)
        rim = lay['rim']
        # rim, ticks, the sync pulse and the sweep direction
        p.setPen(QPen(THEME.color('grid2'), 1))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(c, rim, rim)
        step = TM.tick_step(T)
        p.setFont(mono(10))
        k = 0
        while k * step < T - 1e-9:
            t = k * step
            major = k % 2 == 0
            p.setPen(QPen(THEME.color('ink' if k == 0 else 'muted'), 2 if k == 0 else 1))
            p.drawLine(self._at(c, t, rim), self._at(c, t, rim + (8 if major else 4)))
            if major:
                q = self._at(c, t, rim + 18)
                lbl = f'{t:g}' + (' ms' if k == 0 else '')
                w = p.fontMetrics().horizontalAdvance(lbl)
                p.drawText(QPointF(q.x() - w / 2, q.y() + 4), lbl)
            k += 1
        p.setPen(QPen(THEME.color('ink2'), 1.2))
        arrow = self._arc(c, 0.04 * T, 0.12 * T, R + 16)
        p.drawPolyline(arrow)
        tip, back = arrow.at(arrow.count() - 1), arrow.at(max(arrow.count() - 4, 0))
        dx, dy = tip.x() - back.x(), tip.y() - back.y()
        n = math.hypot(dx, dy) or 1.0
        dx, dy = dx / n, dy / n
        p.setBrush(THEME.color('ink2'))
        p.drawPolygon(QPolygonF([tip, QPointF(tip.x() - dx * 8 - dy * 4, tip.y() - dy * 8 + dx * 4),
                                 QPointF(tip.x() - dx * 8 + dy * 4, tip.y() - dy * 8 - dx * 4)]))
        # one track per camera
        tw = lay['track_w']
        ro, ex = st.readout_ms, st.exposure_ms
        for cam, r in zip(res.cams, lay['tracks']):
            col = THEME.cam(cam['slot'])
            pen = QPen(THEME.cam(cam['slot'], 0.32), max(tw * 0.7, 2.0))
            pen.setCapStyle(Qt.PenCapStyle.FlatCap)
            p.setPen(pen)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawPolyline(self._arc(c, cam['enter_ms'], cam['enter_ms'] + cam['sweep_ms'], r))
            pen = QPen(col, 3)
            pen.setCapStyle(Qt.PenCapStyle.FlatCap)
            p.setPen(pen)
            p.drawLine(self._at(c, cam['cross_ms'], r - tw * 0.45), self._at(c, cam['cross_ms'], r + tw * 0.45))
            if cam.get('misalign_ms') is not None:          # the matched frame, centred misalign_ms off the crossing
                t0 = cam['cross_ms'] + cam['misalign_ms'] - ro / 2 - ex / 2
                p.setPen(QPen(THEME.color('ink'), 1))
                p.drawPolyline(self._arc(c, t0, t0 + ro + ex, r))
                p.setBrush(THEME.color('ink'))
                p.setPen(QPen(col, 1.5))
                p.drawEllipse(self._at(c, t0 + ro / 2 + ex / 2, r), 4, 4)
            p.setFont(mono(10, QFont.Weight.DemiBold))
            w = p.fontMetrics().horizontalAdvance(cam['tag'])
            q = self._at(c, cam['enter_ms'] - (w / 2 + 7) / (2 * math.pi * r) * T, r)
            p.setPen(col)
            p.drawText(QPointF(q.x() - w / 2, q.y() + 4), cam['tag'])
        # trigger pulses on the rim
        p.setFont(mono(10, QFont.Weight.DemiBold))
        for g in res.groups:
            t = g['trigger_ms']
            while t < T - 1e-9:
                q = self._at(c, t, rim - 1)
                a = math.radians(float(lid.direction(t)))
                ux, uy = -math.sin(a), -math.cos(a)
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(THEME.color('ink'))
                p.drawPolygon(QPolygonF([q, QPointF(q.x() + ux * 9 - uy * 5, q.y() + uy * 9 + ux * 5),
                                         QPointF(q.x() + ux * 9 + uy * 5, q.y() + uy * 9 - ux * 5)]))
                lq = self._at(c, t, rim - 9)
                p.setPen(THEME.color('ink'))
                w = p.fontMetrics().horizontalAdvance(g['letter'])
                p.drawText(QPointF(lq.x() - w / 2, lq.y() + 4), g['letter'])
                t += g['period_ms']
        # the vehicle and the cameras about the lidar
        if res.outline is not None:
            p.setPen(QPen(THEME.color('muted'), 1))
            p.setBrush(THEME.color('body', 0.28))
            p.drawPolygon(QPolygonF([self._hub(c, lay, x, y) for x, y in res.outline]))
        for cam in res.cams:
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(THEME.cam(cam['slot']))
            p.drawEllipse(self._hub(c, lay, *cam['pos'][:2]), 4, 4)
        q = self._hub(c, lay, *lid.pos[:2])
        p.setPen(QPen(THEME.color('ink'), 1.2))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(q, 6, 6)
        p.setBrush(THEME.color('ink'))
        p.drawEllipse(q, 2, 2)
        # legend
        p.setFont(sans(11))
        p.setPen(THEME.color('ink'))
        p.drawText(QPointF(12, 16), f'{lid.tag} · {lid.name} · {lid.rate_hz:g} Hz {lid.rotation} · '
                                    f'phase {lid.phase_deg:.1f}°')
        p.setFont(sans(10))
        p.setPen(THEME.color('muted'))
        p.drawText(QRectF(0, self.height() - 18, self.width() - 10, 16), Qt.AlignmentFlag.AlignRight,
                   'bar: the sweep crosses the view · dot: middle row exposed · line: first to last row · '
                   '▲ trigger pulse')

    # ---- hover
    def mouseMoveEvent(self, e):
        res = self.res
        if res is None or res.lidar is None:
            return
        lid = res.lidar
        c, R = self._geom()
        lay = TM.dial_layout(res, R)
        dx, dy = e.position().x() - c.x(), e.position().y() - c.y()
        s = math.hypot(dx, dy)
        if s < 2:
            return
        direction = math.degrees(math.atan2(-dx, -dy))
        t = float(lid.time_of(lid.sensor_az(direction)))
        lines = [f'<b>{t:.1f} ms</b> after the sync pulse · {lid.tag} faces {direction % 360:.0f}°']
        tw = lay['track_w']
        for cam, r in zip(res.cams, lay['tracks']):
            if abs(s - r) <= tw / 2:
                lines.append(f'<b>{cam["tag"]}</b> · line {cam["sync"] or "none"} · {cam["fps"]:g} fps '
                             f'({cam["ratio"]:.3g}× the lidar)')
                lines.append(f'sweep enters {cam["enter_ms"]:.1f} ms, crosses {cam["cross_ms"]:.1f} ms, takes '
                             f'{cam["sweep_ms"]:.1f} ms across the image')
                if cam.get('misalign_ms') is not None:
                    lines.append(f'trigger {cam["trigger_ms"]:.1f} ms · middle row {cam["misalign_ms"]:+.1f} ms off '
                                 f'the sweep ({cam["travel_cm"]:.0f} cm at {res.settings.speed_kmh:g} km/h)')
                else:
                    lines.append(f'ideal trigger {_ms(cam["ideal_ms"])} ms · not on a trigger line')
                break
        QToolTip.showText(e.globalPosition().toPoint(), '<br>'.join(lines), self)

    def leaveEvent(self, e):
        QToolTip.hideText()
        super().leaveEvent(e)
