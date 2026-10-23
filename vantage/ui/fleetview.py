"""Fleet in the UI: the matrix (vehicles × the plan's sensors, each cell coloured by its latest offset's share of the
tolerance) and the drift chart of one vehicle's sensor (its offsets over time against the tolerance band, with the
trend carried on over the drift window)."""
from __future__ import annotations

from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QPainter, QPen
from PyQt6.QtWidgets import QToolTip

from ..core import fleet as FL
from .charts import ChartBase
from .theme import STATUS, THEME, mono, sans

SENSOR_COLORS = {'lidar': '#0aa5b8', 'radar': '#a0743c', 'gnss': '#7f9a1c', 'imu': '#6e7f96'}   # ui.sensorview.COLORS
AXIS_COLORS = FL.AXIS_HEX               # x / roll red, y / pitch green, z / yaw blue


def sensor_color(p) -> QColor:
    """A plan sensor's identity colour: a camera's slot, a sensor's kind."""
    if p['kind'] == 'camera':
        return THEME.cam(p['slot'])
    return QColor(SENSOR_COLORS.get(p['kind'], '#888888'))


def cell_fill(c) -> QColor | None:
    """A cell's fill over the panel: the status colour, stronger the nearer (or further beyond) the tolerance."""
    hx, a = FL.ratio_fill(c['ratio'], c['verdict']) if c else (None, 0.0)
    if hx is None:
        return None
    col = QColor(hx)
    col.setAlphaF(a)
    return col


def _ink_on(fill: QColor | None, panel: QColor) -> QColor:
    """Text colour that reads on a translucent fill over the panel."""
    if fill is None:
        return THEME.color('ink')
    a = fill.alphaF()
    r = a * fill.redF() + (1 - a) * panel.redF()
    g = a * fill.greenF() + (1 - a) * panel.greenF()
    b = a * fill.blueF() + (1 - a) * panel.blueF()
    return QColor('#0b0b0b') if 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.5 else QColor('#f0efec')


def cell_tip(res, c, name) -> str:
    """The hover text of a matrix cell."""
    st = res.settings
    d = c['d']
    out = [f'{name} · {c["tag"]} · {FL.VERDICT_LABEL[c["verdict"]]} ({c["ratio"]:.2f}× the tolerance)',
           f'Latest: {c["date"] or "undated"} · {c["source"] or FL.KIND_LABEL[c["record_kind"]]} '
           f'({c["records"]} record{"" if c["records"] == 1 else "s"})',
           f'Δx {d[0]:+.1f}  Δy {d[1]:+.1f}  Δz {d[2]:+.1f} mm  ({c["dpos_mm"]:.1f} mm)',
           f'Δyaw {d[3]:+.2f}  Δpitch {d[4]:+.2f}  Δroll {d[5]:+.2f}°  ({c["rot_deg"]:.2f}° rotation)']
    q = c['quality']
    if q.get('rms_px') is not None:
        out.append(f'RMS {float(q["rms_px"]):.2f} px' + (f' · {q["status"]}' if q.get('status') else ''))
    dr = c['drift']
    if dr:
        if dr['trend']:
            out.append('Drifting: ' + ('already out of tolerance' if dr['days_to_tol'] == 0 else
                                       f'out of tolerance in about {dr["days_to_tol"]:.0f} days'))
        if dr['jump']:
            out.append(f'Jump: the latest step is {dr["jump_ratio"]:.1f}× the tolerance off its course')
    o = c['outlier']
    if o:
        unit = 'mm' if o['component'] in ('x', 'y', 'z') else '°'
        out.append(f'Fleet: {o["from_median"]:+.{0 if unit == "mm" else 2}f} {unit} in {o["component"]} from the '
                   f'median (z {abs(o["z"]):.1f})')
    return '\n'.join(out)


class FleetMatrix(ChartBase):
    """Vehicles down, the plan's sensors across. Each measured cell is filled by its latest offset's share of the
    tolerance and labelled with the part that dominates (14 mm, 2.1°); a dot marks a drifting, jumping or outlying
    sensor; not measured is a hairline box. Click a cell to select it."""
    cell_clicked = pyqtSignal(str, str)          # vehicle id, plan tag
    vehicle_clicked = pyqtSignal(str)
    HEAD, ROW, LAB, GAP = 26, 24, 120, 2

    def __init__(self, parent=None):
        super().__init__(parent)
        self.title = ''
        self.res = None
        self.sel = None                          # (vehicle id, tag)
        self.sel_vehicle = ''
        self._hits = []                          # (QRectF, ('cell', vid, tag) | ('vehicle', vid))
        self.setMinimumHeight(self.HEAD + self.ROW * 2 + 24)

    def set_result(self, res, sel=None, sel_vehicle=''):
        self.res, self.sel, self.sel_vehicle = res, sel, sel_vehicle
        n = len(res.vehicles) if res else 0
        self.setMinimumHeight(self.HEAD + self.ROW * max(n, 2) + 24)
        self.update()

    def _geom(self):
        n = len(self.res.plan) if self.res else 0
        cw = max(30.0, min(76.0, (self.width() - self.LAB - 4) / max(n, 1)))
        return cw

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        self._hits = []
        res = self.res
        panel = THEME.color('panel')
        if res is None or not res.vehicles or not res.plan:
            p.setFont(sans(12))
            p.setPen(THEME.color('muted'))
            msg = ('Add a vehicle and import its installation check or calibration to see it here.' if res is None or
                   not res.vehicles else f'The plan, {res.variant}, has no visible camera or sensor.')
            p.drawText(QRectF(0, 0, self.width(), 40), Qt.AlignmentFlag.AlignVCenter, msg)
            p.end()
            return
        cw = self._geom()
        # header: identity swatch + tag
        p.setFont(mono(11, QFont.Weight.DemiBold))
        fm = p.fontMetrics()
        for j, ps in enumerate(res.plan):
            x = self.LAB + j * cw
            p.fillRect(QRectF(x + 4, 9, 8, 8), sensor_color(ps))
            p.setPen(THEME.color('ink'))
            p.drawText(QRectF(x + 15, 0, cw - 15, self.HEAD), Qt.AlignmentFlag.AlignVCenter,
                       fm.elidedText(ps['tag'], Qt.TextElideMode.ElideRight, int(cw - 16)))
        y = float(self.HEAD)
        for v in res.vehicles:
            r = QRectF(0, y, self.LAB - 6, self.ROW)
            if v['id'] == self.sel_vehicle:
                p.fillRect(r.adjusted(0, 1, 0, -1), THEME.color('sel'))
            p.setFont(sans(12, QFont.Weight.DemiBold if v['id'] == self.sel_vehicle else None))
            p.setPen(THEME.color('ink'))
            p.drawText(r.adjusted(6, 0, 0, 0), Qt.AlignmentFlag.AlignVCenter,
                       p.fontMetrics().elidedText(v['name'], Qt.TextElideMode.ElideRight, int(r.width() - 8)))
            self._hits.append((r, ('vehicle', v['id'])))
            for j, ps in enumerate(res.plan):
                x = self.LAB + j * cw
                box = QRectF(x + 1, y + 1, cw - self.GAP, self.ROW - self.GAP)
                c = res.cells.get((v['id'], ps['tag']))
                if c is None:
                    p.setPen(QPen(THEME.color('hair'), 1))
                    p.setBrush(Qt.BrushStyle.NoBrush)
                    p.drawRect(box.adjusted(0.5, 0.5, -0.5, -0.5))
                    p.setPen(THEME.color('muted'))
                    p.setFont(mono(11))
                    p.drawText(box, Qt.AlignmentFlag.AlignCenter, '—' if v['records'] else '')
                else:
                    fill = cell_fill(c)
                    p.fillRect(box, fill)
                    p.setFont(mono(11, QFont.Weight.DemiBold))
                    p.setPen(_ink_on(fill, panel))
                    p.drawText(box, Qt.AlignmentFlag.AlignCenter,
                               p.fontMetrics().elidedText(FL.cell_text(c, res.settings), Qt.TextElideMode.ElideRight,
                                                          int(box.width() - 4)))
                    d = c['drift']
                    if (d and (d['trend'] or d['jump'])) or c['outlier']:
                        p.setPen(Qt.PenStyle.NoPen)
                        p.setBrush(_ink_on(fill, panel))
                        p.drawEllipse(QPointF(box.right() - 5, box.top() + 5), 2.5, 2.5)
                if self.sel == (v['id'], ps['tag']):
                    p.setPen(QPen(THEME.color('ink'), 1.5))
                    p.setBrush(Qt.BrushStyle.NoBrush)
                    p.drawRect(box.adjusted(-0.5, -0.5, 0.5, 0.5))
                self._hits.append((box, ('cell', v['id'], ps['tag'])))
            y += self.ROW
        p.setFont(sans(11))
        p.setPen(THEME.color('muted'))
        p.drawText(QRectF(0, y + 4, self.width(), 16), Qt.AlignmentFlag.AlignVCenter,
                   f'Fill: the latest offset against the tolerance ({res.settings.pos_tol_mm:g} mm / '
                   f'{res.settings.ang_tol_deg:g}°): green within, amber adjust, orange re-mount. '
                   'Dot: drifting, a jump or a fleet outlier. —: not measured.')
        p.end()

    def _hit(self, pos):
        return next((h for r, h in self._hits if r.contains(pos)), None)

    def mousePressEvent(self, e):
        h = self._hit(e.position())
        if h is None:
            return
        if h[0] == 'cell':
            self.cell_clicked.emit(h[1], h[2])
        else:
            self.vehicle_clicked.emit(h[1])

    def mouseMoveEvent(self, e):
        h = self._hit(e.position())
        if h is not None and h[0] == 'cell' and self.res is not None:
            c = self.res.cells.get((h[1], h[2]))
            v = self.res.vehicle(h[1])
            tip = cell_tip(self.res, c, v['name']) if c else f'{v["name"]} · {h[2]}: not measured in any record'
            QToolTip.showText(e.globalPosition().toPoint(), tip, self)
        else:
            QToolTip.hideText()

    def cell_rect(self, vid, tag) -> QRectF | None:
        """Where a cell is drawn (for tests and the page)."""
        return next((r for r, h in self._hits if h[0] == 'cell' and h[1] == vid and h[2] == tag), None)


class DriftChart(ChartBase):
    """One vehicle's sensor over time: Δ position (mm, x y z) above, Δ angle (°, yaw pitch roll) below, the tolerance
    band shaded, each record a dot, and the trend of a drifting component carried on over the drift window (dashed)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.title = 'Drift'
        self.res = None
        self.cell = None
        self.name = ''
        self._dots = []                     # (QPointF, series index)
        self.setMinimumHeight(220)

    def set_data(self, res, cell, name=''):
        self.res, self.cell, self.name = res, cell, name
        self.title = f'{name} · {cell["tag"]}: offset from the plan over time' if cell else 'Drift'
        self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        self._dots = []
        c, res = self.cell, self.res
        d = c['drift'] if c else None
        right = ''
        if d:
            right = ('drifting' if d['trend'] else 'a jump' if d['jump'] else
                     f'{d["n"]} records over {d["span_days"]:.0f} days')
        self.paint_title(p, right)
        if c is None or res is None:
            p.setFont(sans(11))
            p.setPen(THEME.color('muted'))
            p.drawText(QRectF(0, 24, self.width(), 30), Qt.AlignmentFlag.AlignVCenter,
                       'Click a cell of the matrix to see that sensor over time.')
            p.end()
            return
        st = res.settings
        if d is None:
            p.setFont(sans(11))
            p.setPen(THEME.color('muted'))
            p.drawText(QRectF(0, 24, self.width(), 30), Qt.AlignmentFlag.AlignVCenter,
                       'One dated record of this sensor: a drift needs two or more.')
            p.end()
            return
        ser = d['series']
        ts = [s['days'] for s in ser]
        t0 = ts[0]
        t1 = ts[-1] + (st.drift_window_days if d['trend'] else 0.0)
        span = max(t1 - t0, 1.0)
        top, gap, bottom = 40.0, 26.0, 20.0
        left, rpad = 46.0, 10.0
        H = (self.height() - top - bottom - gap) / 2
        W = self.width() - left - rpad
        X = lambda t: left + (t - t0) / span * W                       # noqa: E731
        for panel in range(2):
            ks = (0, 1, 2) if panel == 0 else (3, 4, 5)
            names = ('x', 'y', 'z') if panel == 0 else ('yaw', 'pitch', 'roll')
            cols = AXIS_COLORS if panel == 0 else (AXIS_COLORS[2], AXIS_COLORS[1], AXIS_COLORS[0])
            tol = st.pos_tol_mm if panel == 0 else st.ang_tol_deg
            vals = [s['d'][k] for s in ser for k in ks]
            lim = max(tol * 1.5, max(abs(v) for v in vals) * 1.15)
            if d['trend']:
                for k in ks:
                    if d['significant'][k]:
                        e = ser[-1]['d'][k] + d['trend_per_30d'][k] / 30.0 * st.drift_window_days
                        lim = max(lim, abs(e) * 1.1)
            y0 = top + panel * (H + gap)
            Y = lambda v, y0=y0, lim=lim: y0 + H / 2 - v / lim * H / 2    # noqa: E731
            band = QColor(STATUS['good'])
            band.setAlphaF(0.10)
            p.fillRect(QRectF(left, Y(tol), W, Y(-tol) - Y(tol)), band)
            p.setPen(QPen(THEME.color('hair'), 1))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRect(QRectF(left, y0, W, H))
            if d['trend']:                                   # the last record: the trend carries on to its right
                p.setPen(QPen(THEME.color('strong'), 1, Qt.PenStyle.DotLine))
                p.drawLine(QPointF(X(ts[-1]), y0), QPointF(X(ts[-1]), y0 + H))
            p.setPen(QPen(THEME.color('strong'), 1))
            p.drawLine(QPointF(left, Y(0)), QPointF(left + W, Y(0)))
            dash = QPen(THEME.color('muted'), 1, Qt.PenStyle.DashLine)
            p.setPen(dash)
            for v in (tol, -tol):
                p.drawLine(QPointF(left, Y(v)), QPointF(left + W, Y(v)))
            p.setFont(mono(10))
            p.setPen(THEME.color('muted'))
            dec = 0 if panel == 0 else (1 if lim >= 1 else 2)
            for v in (lim, tol, 0.0, -tol, -lim):
                if v not in (lim, -lim) and abs(abs(v) - lim) < lim * 0.12:
                    continue
                p.drawText(QRectF(0, Y(v) - 7, left - 5, 14),
                           Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, f'{v:+.{dec}f}' if v else '0')
            # legend, above the panel
            p.setFont(sans(11))
            lx = left
            p.setPen(THEME.color('ink2'))
            cap = 'Position, mm' if panel == 0 else 'Angle, °'
            p.drawText(QPointF(lx, y0 - 6), cap)
            lx += p.fontMetrics().horizontalAdvance(cap) + 10
            for nm, col in zip(names, cols):
                p.fillRect(QRectF(lx, y0 - 14, 8, 8), QColor(col))
                p.setPen(THEME.color('ink2'))
                p.drawText(QPointF(lx + 12, y0 - 6), nm)
                lx += 12 + p.fontMetrics().horizontalAdvance(nm) + 10
            p.setPen(THEME.color('muted'))
            p.drawText(QPointF(lx + 4, y0 - 6), f'band ±{tol:g} {"mm" if panel == 0 else "°"} (the tolerance)')
            for i, (k, col) in enumerate(zip(ks, cols)):
                q = QColor(col)
                pts = [QPointF(X(t), Y(s['d'][k])) for t, s in zip(ts, ser)]
                p.setPen(QPen(q, 1.6))
                for a, b in zip(pts, pts[1:]):
                    p.drawLine(a, b)
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(q)
                for j, pt in enumerate(pts):
                    p.drawEllipse(pt, 3, 3)
                    if i == 0:
                        self._dots.append((QRectF(pt.x() - 6, y0, 12, H), j))
                if d['trend'] and d['significant'][k]:
                    e = ser[-1]['d'][k] + d['trend_per_30d'][k] / 30.0 * st.drift_window_days
                    p.setPen(QPen(q, 1.2, Qt.PenStyle.DashLine))
                    p.drawLine(pts[-1], QPointF(X(t1), Y(e)))
        # date axis
        p.setFont(sans(10))
        p.setPen(THEME.color('muted'))
        yb = self.height() - bottom + 4
        p.drawText(QRectF(left, yb, 120, 14), Qt.AlignmentFlag.AlignLeft, ser[0]['date'][:10])
        last = ser[-1]['date'][:10]
        xl = X(ts[-1])
        p.drawText(QRectF(min(xl - 60, left + W - 120), yb, 120, 14),
                   Qt.AlignmentFlag.AlignRight if xl > left + W - 70 else Qt.AlignmentFlag.AlignHCenter, last)
        if d['trend']:
            p.drawText(QRectF(left + W - 160, yb, 160, 14), Qt.AlignmentFlag.AlignRight,
                       f'+{st.drift_window_days:g} days')
        p.end()

    def mouseMoveEvent(self, e):
        hit = next((j for r, j in self._dots if r.contains(e.position())), None)
        if hit is None or self.cell is None or not self.cell['drift']:
            QToolTip.hideText()
            return
        s = self.cell['drift']['series'][hit]
        d = s['d']
        QToolTip.showText(e.globalPosition().toPoint(),
                          f'{s["date"]} · {s["source"] or FL.KIND_LABEL.get(s["kind"], "")}\n'
                          f'Δx {d[0]:+.1f}  Δy {d[1]:+.1f}  Δz {d[2]:+.1f} mm\n'
                          f'Δyaw {d[3]:+.2f}  Δpitch {d[4]:+.2f}  Δroll {d[5]:+.2f}°\n'
                          f'{s["ratio"]:.2f}× the tolerance', self)
