"""Analysis ▸ Imaging radar: every radar of the rig as a 4D imaging radar (core.imaging_radar): its MIMO virtual array,
the beam (resolution, sidelobes, grating lobes) in cuts, as a map and as a 3D lobe, the detection range of each class
over the road with the two-ray fades, the ghosts the bumper cavity and the road make, the mounting advice, and the
zone model beside it.

Left: the radar shown, the model it is evaluated as (a what-if that leaves the project alone, or a real swap you can
undo), the taper, the road, the cavity and the link budget. Centre: tiles, the cuts, the beam map, the range chart, the
virtual array and the lobe. Right: every radar in a table, the ghosts, the advice and the findings. The study takes
about a second per imaging radar: it runs in a background thread after every edit while the page shows."""
from __future__ import annotations

import math
import threading
import traceback

import numpy as np
from PyQt6.QtCore import QObject, QPointF, QRectF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPolygonF
from PyQt6.QtWidgets import (QAbstractItemView, QComboBox, QFileDialog, QGridLayout, QHBoxLayout, QHeaderView,
                             QProgressBar, QScrollArea, QSplitter, QTableWidget, QTableWidgetItem, QVBoxLayout,
                             QWidget)

from ...core import imaging_radar as IR
from ..charts import ChartBase, nice_ticks
from ..theme import SEQ, STATUS, THEME, mono, sans
from ..widgets import ROW_H, ScrubSpinBox, Section, SegmentedControl, StatusLine, button, hline, label
from .recording_ws import Tile

AUTO_DELAY_MS = 400
FIELDS = (('taylor_sll_db', 'Taylor sidelobes', 0, 'dB', 1.0,
           'The Taylor taper\'s design sidelobe level below the peak'),
          ('taylor_nbar', 'Taylor n̄', 0, '', 1.0, 'Nearly equal sidelobes next to the main lobe (n̄ − 1 of them)'),
          ('roughness_mm', 'Road roughness', 1, 'mm', 0.1, 'rms height of the road texture (Ament\'s factor)'),
          ('face_reflection_db', 'Radar front', 1, 'dB', 1.0,
           'How much the radar\'s own front reflects back into the cavity (0 dB: metal, the worst case)'),
          ('ghost_floor_db', 'Ghost floor', 0, 'dB', 1.0, 'Images weaker than this against the target are not listed'),
          ('max_range_m', 'Search to', 0, 'm', 10.0, 'Detection is searched out to here'),
          ('link_snr_db', 'Threshold SNR', 1, 'dB', 0.5, 'Link budget: the SNR a detection needs (assumed 15 dB)'),
          ('link_loss_db', 'Losses', 1, 'dB', 0.5, 'Link budget: taper and implementation losses (assumed 6 dB)'),
          ('integration_ms', 'Integration', 1, 'ms', 0.5,
           'Link budget: the active frame time integrated coherently (0: the product\'s)'))
FIELD_LABEL = {k: t for k, t, *_ in FIELDS}
FIELD_LABEL.update(window='taper', road='road', polarisation='polarisation', swap_model='evaluated as')
CLASS_COLORS = ('#3987e5', '#d95926', '#199e70', '#c98500', '#d55181', '#9085e9')


class _Sig(QObject):
    done = pyqtSignal(int, object)
    failed = pyqtSignal(int, str)
    progress = pyqtSignal(int, float)


def _fin(v) -> bool:
    try:
        return v is not None and math.isfinite(float(v))
    except (TypeError, ValueError):
        return False


def _ramp(db, lo=-40.0):
    """dB (lo … 0) to RGB on the app's sequential ramp (dark to light)."""
    cols = np.array([[int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)] for h in SEQ[::-1]], float)
    t = np.clip((np.asarray(db, float) - lo) / -lo, 0, 1) * (len(cols) - 1)
    i0 = np.floor(t).astype(int)
    i1 = np.minimum(i0 + 1, len(cols) - 1)
    f = (t - i0)[..., None]
    return cols[i0] * (1 - f) + cols[i1] * f


# =========================================================================== charts
class CutChart(ChartBase):
    """A pattern cut: dB against angle, the taper's and the uniform aperture's, the −3 dB line."""
    PAD_L, PAD_B, PAD_T = 36, 22, 24

    def __init__(self, title):
        super().__init__()
        self.title = title
        self.deg = None
        self.series = []            # (label, dB array, QColor)
        self.half = 90.0
        self.setMinimumHeight(170)

    def set_data(self, deg, series, half, right=''):
        self.deg, self.series, self.half, self.subtitle = deg, series, half, right
        self.update()

    def _rect(self):
        return QRectF(self.PAD_L, self.PAD_T, self.width() - self.PAD_L - 8, self.height() - self.PAD_T - self.PAD_B)

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.paint_title(p)
        r = self._rect()
        if self.deg is None:
            p.setPen(THEME.color('muted'))
            p.setFont(sans(11))
            p.drawText(r, Qt.AlignmentFlag.AlignCenter, 'No array: a zone model states none.')
            p.end()
            return
        p.setFont(mono(10))
        for v in (0, -10, -20, -30, -40, -50):
            y = r.top() + r.height() * (-v / 50)
            p.setPen(QPen(THEME.color('hair' if v else 'strong'), 1))
            p.drawLine(QPointF(r.left(), y), QPointF(r.right(), y))
            p.setPen(THEME.color('muted'))
            p.drawText(QRectF(0, y - 7, r.left() - 4, 14), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                       f'{v}')
        y3 = r.top() + r.height() * 3 / 50
        p.setPen(QPen(THEME.color('strong'), 1, Qt.PenStyle.DashLine))
        p.drawLine(QPointF(r.left(), y3), QPointF(r.right(), y3))
        h = self.half
        for t in (-h, 0.0, h):
            x = r.left() + (t + h) / (2 * h) * r.width()
            p.setPen(THEME.color('muted'))
            p.drawText(QRectF(x - 30, r.bottom() + 3, 60, 14), Qt.AlignmentFlag.AlignCenter, f'{t:+.3g}°')
        m = (self.deg >= -h) & (self.deg <= h)
        xs = r.left() + (self.deg[m] + h) / (2 * h) * r.width()
        for _, ys, col in self.series:
            yy = r.top() + r.height() * np.clip(-ys[m], 0, 50) / 50
            p.setPen(QPen(col, 1.4))
            step = max(1, len(xs) // 1500)
            p.drawPolyline(QPolygonF([QPointF(a, b) for a, b in zip(xs[::step], yy[::step])]))
        x0 = r.left() + 6
        for lab, _, col in self.series:
            p.fillRect(QRectF(x0, r.top() + 4, 10, 3), col)
            p.setPen(THEME.color('ink2'))
            p.setFont(sans(10))
            p.drawText(QPointF(x0 + 14, r.top() + 9), lab)
            x0 += 18 + p.fontMetrics().horizontalAdvance(lab) + 12
        self.paint_tip(p)
        p.end()

    def mouseMoveEvent(self, e):
        r = self._rect()
        if self.deg is None or not r.contains(e.position()):
            self.tip = None
            self.update()
            return
        t = (e.position().x() - r.left()) / r.width() * 2 * self.half - self.half
        i = int(np.argmin(np.abs(self.deg - t)))
        lines = [f'{lab}: {ys[i]:.1f} dB' for lab, ys, _ in self.series]
        self.set_tip(e.position(), f'{self.deg[i]:+.3f}°', *lines)
        self.update()


class BeamMap(QWidget):
    """The beam (array × element pattern) over azimuth × elevation around the field, the field's box, grating lobes."""

    def __init__(self):
        super().__init__()
        self.setMouseTracking(True)
        self.setMinimumSize(260, 170)
        self.img = None
        self.az = self.el = self.db = None
        self.fov = (0.0, 0.0)
        self.marks = []
        self.hover = ''
        THEME.changed.connect(self.update)

    def set_data(self, beam):
        if beam is None or beam.get('map') is None:
            self.img = None
            self.update()
            return
        mp = beam['map']
        ha = min(beam['fov_az_deg'] + 15.0, 90.0)
        he = min(beam['fov_el_deg'] + 15.0, 60.0)
        ia = (mp['az'] >= -ha) & (mp['az'] <= ha)
        ie = (mp['el'] >= -he) & (mp['el'] <= he)
        self.az, self.el = mp['az'][ia], mp['el'][ie]
        self.db = mp['db'][np.ix_(ie, ia)]
        rgb = np.ascontiguousarray(_ramp(self.db[::-1, ::-1]).astype(np.uint8))   # elevation up, left to the left
        h, w = rgb.shape[:2]
        self.img = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
        self.fov = (beam['fov_az_deg'], beam['fov_el_deg'])
        self.marks = [(g['az_deg'], g['el_deg']) for g in beam['grating'] if g['az_deg'] is not None]
        self.update()

    def _rect(self):
        return QRectF(8, 22, self.width() - 16, self.height() - 40)

    def _xy(self, az, el):
        r = self._rect()
        return QPointF(r.left() + (self.az[-1] - az) / (self.az[-1] - self.az[0]) * r.width(),
                       r.top() + (self.el[-1] - el) / (self.el[-1] - self.el[0]) * r.height())

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setFont(sans(12, QFont.Weight.DemiBold))
        p.setPen(THEME.color('ink'))
        p.drawText(QRectF(0, 0, self.width(), 18), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                   'Beam over the field')
        r = self._rect()
        if self.img is None:
            p.setFont(sans(11))
            p.setPen(THEME.color('muted'))
            p.drawText(r, Qt.AlignmentFlag.AlignCenter, 'No array: a zone model states none.')
            p.end()
            return
        p.drawImage(r, self.img)
        a, b = self._xy(self.fov[0], self.fov[1]), self._xy(-self.fov[0], -self.fov[1])
        p.setPen(QPen(THEME.color('ink'), 1))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRect(QRectF(a, b))
        p.setPen(QPen(QColor(STATUS['serious']), 1.5))
        for az, el in self.marks:
            c = self._xy(az, el)
            p.drawEllipse(c, 5, 5)
        p.setFont(sans(10))
        p.setPen(THEME.color('muted'))
        p.drawText(QRectF(8, self.height() - 16, self.width() - 16, 14), Qt.AlignmentFlag.AlignLeft,
                   self.hover or f'±{self.az[-1]:.0f}° az × ±{self.el[-1]:.0f}° el · −40…0 dB · box: the field')
        p.end()

    def mouseMoveEvent(self, e):
        r = self._rect()
        if self.img is None or not r.contains(e.position()):
            self.hover = ''
        else:
            fa = (e.position().x() - r.left()) / r.width()
            fe = (e.position().y() - r.top()) / r.height()
            az = self.az[-1] - fa * (self.az[-1] - self.az[0])
            el = self.el[-1] - fe * (self.el[-1] - self.el[0])
            i = int(np.argmin(np.abs(self.az - az)))
            j = int(np.argmin(np.abs(self.el - el)))
            self.hover = f'{az:+.1f}° az {el:+.1f}° el: {self.db[j, i]:.1f} dB'
        self.update()


class RangeChart(ChartBase):
    """The detection margin of each class against range (log) over the road: above 0 dB it is detected; the zone
    model's range for the class as a tick on the axis."""
    PAD_L, PAD_B, PAD_T = 36, 22, 24

    def __init__(self):
        super().__init__()
        self.title = 'Detection margin against range'
        self.curves = []            # (name, d, margin, QColor, zone range)
        self.far = 400.0
        self.setMinimumHeight(190)

    def set_data(self, radar, far):
        self.curves, self.far = [], far
        if radar is not None:
            for i, (n, g) in enumerate(radar['ground']['classes'].items()):
                self.curves.append((n, g['curve']['d'], g['curve']['margin_db'].astype(float),
                                    QColor(CLASS_COLORS[i % len(CLASS_COLORS)]), radar['zone']['classes'].get(n)))
        self.subtitle = 'dB above the threshold · log range'
        self.update()

    def _rect(self):
        return QRectF(self.PAD_L, self.PAD_T, self.width() - self.PAD_L - 10, self.height() - self.PAD_T - self.PAD_B)

    def _x(self, d):
        r = self._rect()
        return r.left() + (np.log10(np.maximum(d, 1.0))) / math.log10(self.far) * r.width()

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.paint_title(p)
        r = self._rect()
        lo, hi = -20.0, 80.0
        fy = lambda v: r.top() + (hi - np.clip(v, lo, hi)) / (hi - lo) * r.height()      # noqa: E731
        p.setFont(mono(10))
        for v in (-20, 0, 20, 40, 60, 80):
            y = float(fy(v))
            p.setPen(QPen(THEME.color('strong' if v == 0 else 'hair'), 1.4 if v == 0 else 1))
            p.drawLine(QPointF(r.left(), y), QPointF(r.right(), y))
            p.setPen(THEME.color('muted'))
            p.drawText(QRectF(0, y - 7, r.left() - 4, 14), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                       f'{v}')
        for d in (1, 10, 100, 1000):
            if d <= self.far:
                x = float(self._x(d))
                p.setPen(THEME.color('muted'))
                p.drawText(QRectF(x - 25, r.bottom() + 3, 50, 14), Qt.AlignmentFlag.AlignCenter, f'{d} m')
        for name, d, m, col, zr in self.curves:
            xs = self._x(d)
            ys = fy(m)
            step = max(1, len(xs) // 2000)
            p.setPen(QPen(col, 1.4))
            p.drawPolyline(QPolygonF([QPointF(a, b) for a, b in zip(xs[::step], ys[::step])]))
            if _fin(zr) and zr > 0:
                x = float(self._x(zr))
                p.setPen(QPen(col, 2))
                p.drawLine(QPointF(x, r.bottom() - 6), QPointF(x, r.bottom()))
        x0 = r.left() + 6
        p.setFont(sans(10))
        for name, *_, col, _zr in self.curves:
            p.fillRect(QRectF(x0, r.top() + 4, 10, 3), col)
            p.setPen(THEME.color('ink2'))
            p.drawText(QPointF(x0 + 14, r.top() + 9), name)
            x0 += 30 + p.fontMetrics().horizontalAdvance(name)
        p.setPen(THEME.color('muted'))
        p.drawText(QRectF(r.left(), r.bottom() - 20, r.width() - 4, 14), Qt.AlignmentFlag.AlignRight,
                   'ticks on the axis: the zone model\'s range')
        self.paint_tip(p)
        p.end()

    def mouseMoveEvent(self, e):
        r = self._rect()
        if not self.curves or not r.contains(e.position()):
            self.tip = None
            self.update()
            return
        d = 10 ** ((e.position().x() - r.left()) / r.width() * math.log10(self.far))
        lines = []
        for name, dd, m, _, _ in self.curves:
            i = int(np.argmin(np.abs(dd - d)))
            lines.append(f'{name}: {m[i]:+.1f} dB')
        self.set_tip(e.position(), f'{d:.1f} m', *lines)
        self.update()


class ArrayChart(ChartBase):
    """The virtual array (dots), the Tx (squares) and the Rx (circles) in half wavelengths, the row and column the
    beamformer uses outlined."""

    def __init__(self):
        super().__init__()
        self.title = 'Virtual array'
        self.beam = None
        self.setMinimumHeight(150)

    def set_data(self, beam):
        self.beam = beam
        if beam is not None:
            self.subtitle = (f'{beam["n_tx"]} Tx × {beam["n_rx"]} Rx = {beam["n_virtual"]} · {beam["n_unique"]} unique'
                             if beam['layout'] == 'public' else f'assumed {beam["n_az"]} × {beam["n_el"]} aperture')
        self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.paint_title(p)
        b = self.beam
        r = QRectF(10, 24, self.width() - 20, self.height() - 34)
        if b is None:
            p.setFont(sans(11))
            p.setPen(THEME.color('muted'))
            p.drawText(r, Qt.AlignmentFlag.AlignCenter, 'No array: a zone model states none.')
            p.end()
            return
        V = np.asarray(b['virtual'], float)
        pts = [V]
        if b['layout'] == 'public':
            pts += [np.asarray(b['tx'], float), np.asarray(b['rx'], float)]
        allp = np.vstack(pts)
        lo, hi = allp.min(axis=0) - 1, allp.max(axis=0) + 1
        span = np.maximum(hi - lo, 1e-6)
        s = min(r.width() / span[0], r.height() / span[1])
        ox = r.left() + (r.width() - span[0] * s) / 2
        oy = r.bottom() - (r.height() - span[1] * s) / 2

        def xy(q):
            return QPointF(ox + (q[0] - lo[0]) * s, oy - (q[1] - lo[1]) * s)
        dot = max(1.2, min(3.0, s * 0.3))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(THEME.color('ink2'))
        step = max(1, len(V) // 4000)
        for q in V[::step]:
            p.drawEllipse(xy(q), dot, dot)
        if b['layout'] == 'public':
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.setPen(QPen(QColor(CLASS_COLORS[1]), 1.4))
            for q in b['tx']:
                c = xy(q)
                p.drawRect(QRectF(c.x() - 4, c.y() - 4, 8, 8))
            p.setPen(QPen(QColor(CLASS_COLORS[0]), 1.4))
            for q in b['rx']:
                p.drawEllipse(xy(q), 4, 4)
            p.setFont(sans(10))
            p.setPen(THEME.color('muted'))
            p.drawText(QRectF(r.left(), r.bottom() - 12, r.width(), 12), Qt.AlignmentFlag.AlignRight,
                       'squares Tx · circles Rx · dots virtual (λ/2 units)')
        p.end()


class LobeView(QWidget):
    """The radar's one-way beam as a 3D surface (core.imaging_radar.lobe_mesh in the radar's own frame), drawn with
    the painter's algorithm; drag to turn it."""

    def __init__(self):
        super().__init__()
        self.setMinimumSize(220, 190)
        self.tri = self.val = None
        self.yaw, self.pitch = -55.0, 25.0
        self._drag = None
        THEME.changed.connect(self.update)

    def set_data(self, tri, val):
        self.tri, self.val = tri, val
        self.update()

    def mousePressEvent(self, e):
        self._drag = (e.position(), self.yaw, self.pitch)

    def mouseMoveEvent(self, e):
        if self._drag is None:
            return
        p0, y0, t0 = self._drag
        d = e.position() - p0
        self.yaw = y0 + d.x() * 0.5
        self.pitch = min(max(t0 + d.y() * 0.5, -89.0), 89.0)
        self.update()

    def mouseReleaseEvent(self, e):
        self._drag = None

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setFont(sans(12, QFont.Weight.DemiBold))
        p.setPen(THEME.color('ink'))
        p.drawText(QRectF(0, 0, self.width(), 18), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                   '3D lobe')
        if self.tri is None or not len(self.tri):
            p.setFont(sans(11))
            p.setPen(THEME.color('muted'))
            p.drawText(QRectF(0, 20, self.width(), self.height() - 20), Qt.AlignmentFlag.AlignCenter,
                       'No array: a zone model states none.')
            p.end()
            return
        cy, sy = math.cos(math.radians(self.yaw)), math.sin(math.radians(self.yaw))
        cp, sp = math.cos(math.radians(self.pitch)), math.sin(math.radians(self.pitch))
        T = self.tri.reshape(-1, 3)
        x1 = T[:, 0] * cy - T[:, 1] * sy
        y1 = T[:, 0] * sy + T[:, 1] * cy
        sx = y1
        sz = T[:, 2] * cp - x1 * sp
        depth = x1 * cp + T[:, 2] * sp
        R = max(float(np.abs(np.c_[sx, sz]).max()), 1e-6)
        sc = 0.45 * min(self.width(), self.height() - 30) / R
        cxp, cyp = self.width() / 2, 20 + (self.height() - 20) / 2
        X = (cxp + sx * sc).reshape(-1, 3)
        Y = (cyp - sz * sc).reshape(-1, 3)
        D = depth.reshape(-1, 3).mean(axis=1)
        order = np.argsort(D)[::-1]
        cols = _ramp(self.val, -40.0)
        p.setPen(Qt.PenStyle.NoPen)
        for k in order:
            c = cols[k]
            p.setBrush(QColor(int(c[0]), int(c[1]), int(c[2])))
            p.drawPolygon(QPolygonF([QPointF(X[k, i], Y[k, i]) for i in range(3)]))
        p.setFont(sans(10))
        p.setPen(THEME.color('muted'))
        p.drawText(QRectF(4, self.height() - 14, self.width() - 8, 12), Qt.AlignmentFlag.AlignLeft,
                   'radius: dB above −40 · drag to turn')
        p.end()


# =========================================================================== the page
def _table(cols):
    t = QTableWidget(0, len(cols))
    t.setHorizontalHeaderLabels(cols)
    t.verticalHeader().setVisible(False)
    t.verticalHeader().setDefaultSectionSize(ROW_H)
    t.horizontalHeader().setFixedHeight(ROW_H)
    t.setShowGrid(False)
    t.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    t.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    t.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
    for c in range(len(cols)):
        t.horizontalHeader().setSectionResizeMode(c, QHeaderView.ResizeMode.ResizeToContents)
    return t


RADAR_COLS = ('Radar', 'Model', 'Mount', 'Resolution', 'PSL', 'Car', 'Pedestrian', 'Cyclist', 'Ghosts')
GHOST_COLS = ('Kind', 'Target', 'Ghost', 'Offset', 'Δ range', 'Level', 'Seen')


class ImagingRadarPage(QWidget):
    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        self.result: IR.ImagingRadarResult | None = None
        self.busy = False
        self.stale = False
        self._gen = 0
        self._for = None
        self._syncing = False
        self.current = ''                          # the uid of the radar shown
        self.status_text = ''
        self.sig = _Sig()
        self.sig.done.connect(self._done)
        self.sig.failed.connect(self._failed)
        self.sig.progress.connect(lambda g, f: self.bar.setValue(int(f * 1000)) if g == self._gen else None)
        self.setObjectName('base')
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        split = QSplitter(Qt.Orientation.Horizontal)
        split.setHandleWidth(2)
        split.addWidget(self._left())
        split.addWidget(self._centre())
        split.addWidget(self._right())
        split.setSizes([300, 760, 560])
        lay.addWidget(split)
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(AUTO_DELAY_MS)
        self.timer.timeout.connect(lambda: self.run(quiet=True))
        ctx.store.changed.connect(self._on_store)
        ctx.store.project_replaced.connect(self._replaced)
        THEME.changed.connect(lambda *_: self.refresh())
        self._sync_fields()
        self.refresh()

    # ------------------------------------------------------------------ layout
    def _left(self):
        w = QWidget()
        w.setObjectName('panel')
        ll = QVBoxLayout(w)
        ll.setContentsMargins(8, 6, 8, 8)
        ll.setSpacing(4)
        top = QHBoxLayout()
        top.addWidget(label('Imaging radar'))
        top.addStretch(1)
        top.addWidget(label('MIMO array, ghosts, ground', muted=True, small=True))
        ll.addLayout(top)
        ll.addWidget(hline())
        ll.addWidget(label('Every radar as a 4D imaging radar: what its virtual array resolves, what its sidelobes '
                           'hide, how far each class is detected over the road, and the ghosts the bumper cavity and '
                           'the road make. Figures marked assumed are not the maker\'s.', muted=True, small=True,
                           wrap=True))
        sec = Section('Radar')
        self.c_radar = QComboBox()
        self.c_radar.setToolTip('The radar the charts show')
        self.c_radar.currentIndexChanged.connect(self._pick_radar)
        sec.row('Show', self.c_radar)
        self.c_model = QComboBox()
        self.c_model.setToolTip('Evaluate every radar as this imaging radar (the project is not changed), or as its '
                                'own model')
        self.c_model.addItem('Each radar\'s own model', '')
        self.c_model.currentIndexChanged.connect(
            lambda i: None if self._syncing else self.set_option('swap_model', self.c_model.itemData(i) or ''))
        sec.row('Evaluate as', self.c_model)
        bl = QHBoxLayout()
        self.b_swap1 = button('Swap this radar', 'ghost', None, 'Put the chosen imaging radar on the radar shown '
                                                                '(one undo step)')
        self.b_swap1.clicked.connect(lambda: self.swap(all_radars=False))
        self.b_swapall = button('Swap every radar', 'ghost', None, 'Put the chosen imaging radar on every radar '
                                                                   '(one undo step)')
        self.b_swapall.clicked.connect(lambda: self.swap(all_radars=True))
        bl.addWidget(self.b_swap1)
        bl.addWidget(self.b_swapall)
        sec.blay.addLayout(bl)
        ll.addWidget(sec)
        sec = Section('Beam')
        self.seg_window = SegmentedControl([(k, IR.WINDOW_LABEL[k]) for k in IR.WINDOWS], 'taylor')
        self.seg_window.setToolTip('The taper over the array: lower sidelobes for a wider beam')
        self.seg_window.changed.connect(lambda k: self.set_option('window', k))
        sec.blay.addWidget(self.seg_window)
        self.fields = {}
        for key, text, dec, unit, step, tip in FIELDS[:2]:
            self._field(sec, key, text, dec, unit, step, tip)
        ll.addWidget(sec)
        sec = Section('Road and cavity')
        from ...core.materials import ROAD_SURFACES
        self.c_road = QComboBox()
        self.c_road.addItem('No ground reflection', 'none')
        for k, v in ROAD_SURFACES.items():
            self.c_road.addItem(f'{v[0]} (ε′ {v[1]:g})', k)
        self.c_road.setToolTip('The road under the radar (Kurz et al. 2021, measured at 77 GHz)')
        self.c_road.currentIndexChanged.connect(
            lambda i: None if self._syncing else self.set_option('road', self.c_road.itemData(i)))
        sec.row('Road', self.c_road)
        self.seg_pol = SegmentedControl([('auto', 'Product'), ('vertical', 'Vertical'), ('horizontal', 'Horizontal')],
                                        'auto')
        self.seg_pol.setToolTip('The radar\'s polarisation (the product\'s when stated, else vertical)')
        self.seg_pol.changed.connect(lambda k: self.set_option('polarisation', k))
        sec.blay.addWidget(self.seg_pol)
        for key, text, dec, unit, step, tip in FIELDS[2:6]:
            self._field(sec, key, text, dec, unit, step, tip)
        ll.addWidget(sec)
        sec = Section('Link budget', expanded=False)
        for key, text, dec, unit, step, tip in FIELDS[6:]:
            self._field(sec, key, text, dec, unit, step, tip)
        sec.blay.addWidget(label('Used for products whose maker states the chip\'s power and noise figure instead of '
                                 'a range (the AWR2944EVM).', muted=True, small=True, wrap=True))
        ll.addWidget(sec)
        ll.addStretch(1)
        self.status = label('', muted=True, small=True, wrap=True)
        ll.addWidget(self.status)
        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.setTextVisible(False)
        self.bar.setFixedHeight(4)
        ll.addWidget(self.bar)
        bl = QHBoxLayout()
        b = self.b_defaults = button('Defaults', 'ghost', None, 'Back to the default imaging radar settings')
        b.clicked.connect(self._defaults)
        bl.addWidget(b)
        self.b_run = button('Run', None, 'refresh', 'Run the study again now')
        self.b_run.clicked.connect(lambda: self.run())
        bl.addWidget(self.b_run)
        bl.addStretch(1)
        self.b_export = button('Export…', None, 'export', 'Every radar\'s array, beam, ranges, ghosts and advice as '
                                                           'JSON, CSV and PNG')
        self.b_export.clicked.connect(lambda: self.export())
        bl.addWidget(self.b_export)
        ll.addLayout(bl)
        sc = QScrollArea()
        sc.setWidgetResizable(True)
        sc.setWidget(w)
        sc.setMinimumWidth(290)
        return sc

    def _field(self, sec, key, text, dec, unit, step, tip):
        f = ScrubSpinBox('', 'plain', dec, unit, step=step, width=100)
        f.setToolTip(tip)
        f.committed.connect(lambda v, k=key: self.set_option(k, v))
        sec.row(text, f, tip=tip)
        self.fields[key] = f

    def _centre(self):
        w = QWidget()
        w.setObjectName('panel')
        cl = QVBoxLayout(w)
        cl.setContentsMargins(10, 8, 10, 8)
        cl.setSpacing(8)
        self.head = label('', wrap=True)
        cl.addWidget(self.head)
        tiles = QHBoxLayout()
        tiles.setSpacing(8)
        self.tiles = {}
        for key, caption, tip in (
                ('res', 'Resolution', 'The −3 dB beamwidth in azimuth × elevation with the taper (the stated figure '
                                      'in the note)'),
                ('psl', 'Highest sidelobe', 'The highest azimuth sidelobe against the main lobe'),
                ('range', 'Weakest class', 'How far the weakest class is detected without a gap, over the road'),
                ('cavity', 'Fascia image', 'How strong the fascia\'s image of every target is, against the target'),
                ('ghosts', 'Ghosts', 'Images of targets that land inside the field as separate targets')):
            t = self.tiles[key] = Tile(caption, tip)
            tiles.addWidget(t, 1)
        cl.addLayout(tiles)
        grid = QGridLayout()
        grid.setSpacing(8)
        self.cut_az = CutChart('Azimuth cut')
        self.cut_el = CutChart('Elevation cut')
        self.map = BeamMap()
        self.rng = RangeChart()
        self.arr = ArrayChart()
        self.lobe = LobeView()
        grid.addWidget(self.cut_az, 0, 0)
        grid.addWidget(self.cut_el, 0, 1)
        grid.addWidget(self.map, 1, 0)
        grid.addWidget(self.rng, 1, 1)
        grid.addWidget(self.arr, 2, 0)
        grid.addWidget(self.lobe, 2, 1)
        for r_ in range(3):
            grid.setRowStretch(r_, 1)
        cl.addLayout(grid, 1)
        sc = QScrollArea()
        sc.setWidgetResizable(True)
        sc.setWidget(w)
        return sc

    def _right(self):
        w = QWidget()
        w.setObjectName('panel')
        rl = QVBoxLayout(w)
        rl.setContentsMargins(10, 8, 10, 8)
        rl.setSpacing(4)
        rl.addWidget(label('Every radar', bold=True))
        self.t_radars = _table(RADAR_COLS)
        self.t_radars.setToolTip('Ranges: detected without a gap over the road (the zone model\'s in brackets)')
        self.t_radars.itemSelectionChanged.connect(self._table_pick)
        rl.addWidget(self.t_radars, 2)
        rl.addWidget(label('Ghosts of the radar shown', bold=True))
        self.t_ghosts = _table(GHOST_COLS)
        self.t_ghosts.setToolTip('Cavity: the fascia\'s image of a target at that azimuth; ground: the mirror target '
                                 'under the road. Seen: resolved in angle or range (else it merges with the target)')
        rl.addWidget(self.t_ghosts, 2)
        rl.addWidget(label('Mounting advice', bold=True))
        self.advice = label('', small=True, wrap=True)
        rl.addWidget(self.advice)
        rl.addWidget(label('Findings', bold=True))
        self.find_host = QWidget()
        self.find_lay = QVBoxLayout(self.find_host)
        self.find_lay.setContentsMargins(0, 0, 0, 0)
        self.find_lay.setSpacing(4)
        self.find_lay.addStretch(1)
        sc = QScrollArea()
        sc.setWidgetResizable(True)
        sc.setWidget(self.find_host)
        rl.addWidget(sc, 2)
        self.findings: list[StatusLine] = []
        outer = QScrollArea()
        outer.setWidgetResizable(True)
        outer.setWidget(w)
        return outer

    # ------------------------------------------------------------------ settings
    @property
    def settings(self) -> IR.ImagingRadarSettings:
        return IR.settings_of(self.ctx.store.project)

    def set_option(self, key, value) -> bool:
        if self._syncing:
            return False
        old = self.settings.to_dict()
        d = dict(old)
        d[key] = value
        new = IR.ImagingRadarSettings.from_dict(d).to_dict()
        if new == old:
            self._sync_fields()
            return False
        shown = new[key]
        shown = f'{shown:g}' if isinstance(shown, float) else (shown or 'own model')

        def fn(p):
            p.analysis.imaging_radar = new
        self.ctx.store.edit('analysis', fn, f'Imaging radar · {FIELD_LABEL.get(key, key)} {shown}')
        return True

    def _defaults(self):
        if self.ctx.store.project.analysis.imaging_radar:
            def fn(p):
                p.analysis.imaging_radar = {}
            self.ctx.store.edit('analysis', fn, 'Imaging radar · defaults')

    def swap(self, all_radars=False) -> bool:
        """Put the model chosen under 'Evaluate as' on the radar shown (or every visible radar): one undo step."""
        mid = self.c_model.currentData() or ''
        p = self.ctx.store.project
        m = p.sensor_models.get(mid)
        if m is None or m.kind != 'radar':
            self.ctx.msg('Choose an imaging radar under "Evaluate as" first', 4000)
            return False
        uids = [s.uid for s in p.variant.sensors if s.kind == 'radar' and s.visible and not s.locked
                and (all_radars or s.uid == self.current) and s.model_id != mid]
        if not uids:
            self.ctx.msg('Nothing to swap (locked, hidden or already that model)', 4000)
            return False

        def fn(q):
            for s in q.variant.sensors:
                if s.uid in uids:
                    s.model_id = mid
            st = IR.settings_of(q)
            st.swap_model = ''                    # now the project holds it
            q.analysis.imaging_radar = st.to_dict()
        tags = ', '.join(s.tag for s in p.variant.sensors if s.uid in uids)
        self.ctx.store.edit(['variant', 'analysis'], fn, f'{tags} → {m.vendor} {m.name}')
        return True

    def _sync_fields(self):
        st = self.settings
        self._syncing = True
        try:
            d = st.to_dict()
            for k, f in self.fields.items():
                f.set_value(float(d[k]))
            self.seg_window.set_value(st.window)
            self.seg_pol.set_value(st.polarisation)
            i = self.c_road.findData(st.road)
            self.c_road.setCurrentIndex(max(i, 0))
            p = self.ctx.store.project
            want = [('', 'Each radar\'s own model')] + [
                (m.id, f'{m.vendor} {m.name}') for m in p.sensor_models.values() if m.kind == 'radar' and
                (m.ranging == 'mimo' or m.id in IR.ZONE_TO_ARRAY)]
            have = [(self.c_model.itemData(i), self.c_model.itemText(i)) for i in range(self.c_model.count())]
            if want != have:
                self.c_model.clear()
                for k, t in want:
                    self.c_model.addItem(t, k)
            self.c_model.setCurrentIndex(max(self.c_model.findData(st.swap_model), 0))
        finally:
            self._syncing = False

    # ------------------------------------------------------------------ running
    def _key(self):
        return IR.cache_key(self.ctx.store.project)

    def _engine(self):
        win = getattr(self.ctx, 'window', None)
        try:
            if win is not None and hasattr(win, '_shared_engine'):
                return win._shared_engine()
            return self.ctx.analysis.engine.share()
        except Exception:                                        # noqa: BLE001 — the study builds its own
            return None

    def _on_store(self, *_):
        self._sync_fields()
        if self.result is not None or self.isVisible():
            if self._for != self._key():
                self.stale = True
                self.refresh_status()
                if self.isVisible():
                    self.timer.start()

    def _replaced(self, *_):
        self._gen += 1
        self.result, self._for, self.stale, self.busy = None, None, False, False
        self.current = ''
        self._sync_fields()
        self.refresh()
        if self.isVisible():
            self.timer.start()

    def showEvent(self, e):
        super().showEvent(e)
        if self.result is None or self.stale:
            self.timer.start()

    def activate(self):
        if self.result is None or self.stale:
            self.run(quiet=True)

    def run(self, quiet=False) -> bool:
        project = self.ctx.store.project.snapshot()
        key = self._key()
        self._gen += 1
        gen = self._gen
        self.busy = True
        self.status_text = 'Forming the beams and tracing the road and the fascia…'
        self.bar.setValue(0)
        self.refresh_status()
        eng = self._engine()

        def thread():
            try:
                res = IR.run(project, eng, settings=IR.settings_of(project),
                             progress=lambda f: self.sig.progress.emit(gen, f), cancel=lambda: gen != self._gen)
                self.sig.done.emit(gen, (key, res))
            except RuntimeError:
                pass
            except Exception as e:                               # noqa: BLE001
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
        key, res = out
        if res is None:
            self.status_text = 'Cancelled'
            self.refresh_status()
            return
        self.result, self._for = res, key
        self.stale = key != self._key()
        self.status_text = f'{res.variant}: {len(res.radars)} radar{"s" if len(res.radars) != 1 else ""} · ' \
                           f'{res.metrics["seconds"]:.1f} s'
        self.bar.setValue(1000)
        self.refresh()
        if self.stale and self.isVisible():
            self.timer.start()

    def _failed(self, gen, msg):
        if gen != self._gen:
            return
        self.busy = False
        self.status_text = 'Failed: ' + msg.splitlines()[0]
        self.refresh_status()

    # ------------------------------------------------------------------ views
    def refresh_status(self):
        txt = self.status_text
        if self.stale and not self.busy and self.result is not None:
            txt = (txt + ' · ' if txt else '') + 'out of date'
        self.status.setText(txt)
        self.b_export.setEnabled(self.result is not None)

    def _pick_radar(self, i):
        if self._syncing:
            return
        uid = self.c_radar.itemData(i)
        if uid and uid != self.current:
            self.current = uid
            self.show_radar()

    def _table_pick(self):
        rows = self.t_radars.selectionModel().selectedRows() if self.t_radars.selectionModel() else []
        if rows and self.result is not None and rows[0].row() < len(self.result.radars):
            uid = self.result.radars[rows[0].row()]['uid']
            if uid != self.current:
                self.current = uid
                self._syncing = True
                self.c_radar.setCurrentIndex(max(self.c_radar.findData(uid), 0))
                self._syncing = False
                self.show_radar()

    def radar(self):
        res = self.result
        if res is None or not res.radars:
            return None
        return next((r for r in res.radars if r['uid'] == self.current), res.radars[0])

    def refresh(self):
        self.refresh_status()
        res = self.result
        self._syncing = True
        try:
            self.c_radar.clear()
            for r in (res.radars if res else []):
                self.c_radar.addItem(f'{r["tag"]} · {r["model"]}', r['uid'])
            if res and res.radars:
                if not any(r['uid'] == self.current for r in res.radars):
                    self.current = res.radars[0]['uid']
                self.c_radar.setCurrentIndex(max(self.c_radar.findData(self.current), 0))
        finally:
            self._syncing = False
        # every radar
        t = self.t_radars
        rows = res.radars if res else []
        t.setRowCount(len(rows))
        right = Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        for i, r in enumerate(rows):
            b = r['beam']
            g = r['ground']['classes']
            ghosts = [x for x in IR.all_ghosts(r) if x['in_field'] and x['resolved']]
            vals = [r['tag'], r['model'] + (' (as)' if r['swapped'] else ''), r['mount'],
                    (f'{b["az_res_deg"]:.2f}° × ' + (f'{b["el_res_deg"]:.1f}°' if _fin(b['el_res_deg']) else '—'))
                    if b else '—', f'{b["psl_az_db"]:.1f} dB' if b and _fin(b['psl_az_db']) else '—']
            for name in ('Car', 'Pedestrian', 'Cyclist'):
                gc = g.get(name)
                z = r['zone']['classes'].get(name)
                vals.append(f'{gc["continuous_m"]:.0f} m ({z:.0f})' if gc and _fin(gc['continuous_m']) and _fin(z)
                            else '—')
            vals.append(str(len(ghosts)))
            for c, v in enumerate(vals):
                it = QTableWidgetItem(v)
                if c >= 3:
                    it.setFont(mono(11))
                    it.setTextAlignment(right)
                if c == 8 and ghosts:
                    it.setForeground(QColor(STATUS['warning']))
                t.setItem(i, c, it)
        # findings
        for w in self.findings:
            self.find_lay.removeWidget(w)
            w.hide()
            w.deleteLater()
        self.findings = []
        for f in (res.findings if res else []):
            w = StatusLine(f['severity'], f['message'])
            self.find_lay.insertWidget(self.find_lay.count() - 1, w)
            self.findings.append(w)
        if res is None:
            self.head.setText('Runs on its own when the page shows: every visible radar of the active variant.')
        elif not res.radars:
            self.head.setText(f'{res.variant}: no visible radar.')
        self.show_radar()

    def show_radar(self):
        r = self.radar()
        res = self.result
        for k in self.tiles:
            self.tiles[k].set('—', '', '')
        if r is None:
            self.cut_az.set_data(None, [], 90)
            self.cut_el.set_data(None, [], 90)
            self.map.set_data(None)
            self.rng.set_data(None, 400.0)
            self.arr.set_data(None)
            self.lobe.set_data(None, None)
            self.t_ghosts.setRowCount(0)
            self.advice.setText('')
            return
        b = r['beam']
        st = res.settings
        figs = r['figures']
        self.head.setText(f'<b>{r["tag"]}</b> · {r["model"]}' + (' (evaluated as)' if r['swapped'] else '') +
                          f' · {r["carrier_ghz"]:g} GHz · {r["range_ref_m"]:.0f} m on {r["rcs_ref_m2"]:g} m² '
                          f'({"link budget" if r["range_from"] == "link" else r["range_from"]})' +
                          (f' · layout {b["layout"]}' if b else ' · zone model, no array'))
        self.head.setToolTip('\n'.join(f'{k}: {v}' for k, v in figs.items()) or 'The zone model\'s figures')
        if b is not None:
            stated = b['stated']
            note = (f'{stated["az_res_deg"]:g}° stated' if 'az_res_deg' in stated else 'none stated') + \
                f' · uniform {b["az_res_uniform_deg"]:.2f}°'
            self.tiles['res'].set(f'{b["az_res_deg"]:.2f}', '° × ' + (f'{b["el_res_deg"]:.1f}°' if _fin(b['el_res_deg'])
                                                                        else '—'), note)
            gl = len(b['grating'])
            self.tiles['psl'].set(f'{b["psl_az_db"]:.1f}' if _fin(b['psl_az_db']) else '—', 'dB',
                                  f'{IR.WINDOW_LABEL[st.window]} · {gl} grating lobe{"s" if gl != 1 else ""}',
                                  'serious' if gl else ('warning' if _fin(b['psl_az_db']) and b['psl_az_db'] > -20
                                                        else 'good'))
            cu = b['cuts']
            ha = min(max(8 * b['az_res_deg'], 10.0), 90.0) if _fin(b['az_res_deg']) else 90.0
            he = min(max(8 * b['el_res_deg'], 10.0), 90.0) if _fin(b['el_res_deg']) else 90.0
            acc = THEME.cam(1)
            self.cut_az.set_data(cu['deg'], [('uniform', cu['az_uniform'], THEME.color('muted')),
                                             (IR.WINDOW_LABEL[st.window].lower(), cu['az'], acc)], ha,
                                 f'{b["n_az"]} in the row')
            self.cut_el.set_data(cu['deg'], [('uniform', cu['el_uniform'], THEME.color('muted')),
                                             (IR.WINDOW_LABEL[st.window].lower(), cu['el'], acc)], he,
                                 f'{b["n_el"]} in the column')
            self.map.set_data(b)
            self.arr.set_data(b)
            m = self.ctx.store.project.sensor_models.get(r['model_id'])
            from ...core.project import Sensor
            here = Sensor(kind='radar', model_id=r['model_id'], x=0.0, y=0.0, z=0.0)    # the lobe in its own frame
            out = IR.lobe_mesh(here, m, st, scale_m=1.0, n_az=61, n_el=45) if m is not None else None
            self.lobe.set_data(*(out if out else (None, None)))
        else:
            self.cut_az.set_data(None, [], 90)
            self.cut_el.set_data(None, [], 90)
            self.map.set_data(None)
            self.arr.set_data(None)
            self.lobe.set_data(None, None)
        w = r['ground']['classes'].get(r['weakest']) if r['weakest'] else None
        if w is not None:
            self.tiles['range'].set(f'{w["continuous_m"]:.0f}' if _fin(w['continuous_m']) else '—', 'm',
                                    f'{r["weakest"]} · free space {w["free_m"]:.0f} m · {len(w["fades"])} fades'
                                    if _fin(w['free_m']) else r['weakest'])
        c = r['cavity']
        if c is not None:
            self.tiles['cavity'].set(f'{c["image_db"]:.1f}', 'dB', f'{c["class_name"]}, {c["incidence_deg"]:.0f}°, '
                                     f'gap {c["gap_mm"]:.1f} mm')
        else:
            self.tiles['cavity'].set('—', '', 'not behind a fascia')
        gs = [x for x in IR.all_ghosts(r) if x['in_field'] and x['resolved']]
        self.tiles['ghosts'].set(str(len(gs)), '', f'strongest {max(x["level_db"] for x in gs):.1f} dB' if gs else
                                 'none above the floor', 'warning' if gs else 'good')
        self.rng.set_data(r, st.max_range_m)
        # ghosts
        gl = IR.all_ghosts(r)
        self.t_ghosts.setRowCount(len(gl))
        for i, g in enumerate(gl):
            tgt = (f'{g["target_az_deg"]:+.0f}° az' if g['kind'] == 'cavity' else
                   f'{g["class"]} {g["range_m"]:.0f} m')
            vals = [g['kind'] + (f' ×{g["order"]}' if g['kind'] == 'cavity' else ''), tgt,
                    f'{g["ghost_az_deg"]:+.1f}° {g["ghost_el_deg"]:+.1f}°', f'{g["angle_offset_deg"]:.1f}°',
                    f'{g["range_offset_m"] * 100:.1f} cm', f'{g["level_db"]:.1f} dB',
                    ('in field, ' if g['in_field'] else 'outside, ') + (g['resolved'] or 'merges')]
            for c_, v in enumerate(vals):
                it = QTableWidgetItem(v)
                if c_ in (2, 3, 4, 5):
                    it.setFont(mono(11))
                self.t_ghosts.setItem(i, c_, it)
        self.advice.setText('\n'.join('• ' + a for a in IR.advice(r)) or 'Nothing to change.')

    # ------------------------------------------------------------------ export
    def export(self, folder=None):
        res = self.result
        if res is None:
            return []
        if folder is None:
            folder = QFileDialog.getExistingDirectory(self, 'Export the imaging radar study into')
            if not folder:
                return []
        from ...core import provenance as PV
        paths = PV.stamp_paths(IR.export(res, self.ctx.store.project, folder), self.ctx.store.project, folder)
        self.ctx.msg(f'Exported the imaging radar study to {folder}', 5000)
        return paths
