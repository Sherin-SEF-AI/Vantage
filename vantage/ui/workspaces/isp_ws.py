"""Analysis ▸ Exposure metering and HDR seams: each camera's auto-exposure metering mask (what of its image is the
ground, not the body, the rig, the housings, the sky or the dark beyond the lens' image circle), the metering grid
and weighted AE rectangles to load into the ISP, and how far the cameras' exposures drift apart across the
bird's-eye view's seams over the campaign and in hard scenes, before and after the stitcher's gain balancing.
The study runs in the background (about 6 s on the sample's data rig at full sensor resolution) when the page shows
and again after edits; Cancel stops it."""
from __future__ import annotations

import json
import math
import threading
import traceback

import numpy as np
from PyQt6.QtCore import QObject, QPointF, QRectF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPolygonF
from PyQt6.QtWidgets import (QAbstractItemView, QFileDialog, QGridLayout, QHBoxLayout, QHeaderView, QProgressBar,
                             QScrollArea, QSplitter, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from ...core import isp_photometrics as I
from ..charts import ChartBase
from ..theme import SEQ, STATUS, THEME, mono, sans
from ..widgets import ROW_H, fit_width, ScrubSpinBox, SegmentedControl, StatusLine, button, hline, label

# (group, key, label, unit, decimals, scrub step, scale shown / stored, tooltip)
FIELDS = (
    ('Mask', 'horizon_margin_deg', 'Horizon margin', '°', 1, 0.5, 1.0,
     'Rays less than this below the horizon count as sky: a margin for pitch and a hilly horizon'),
    ('Campaign', 'day_step', 'Day step', 'd', 0, 5.0, 1.0, 'Days between sampled days (the Sun glare study\'s dates)'),
    ('Campaign', 'minutes', 'Time step', 'min', 0, 5.0, 1.0, 'Minutes between samples through the driving window'),
    ('Campaign', 'heading_step', 'Heading step', '°', 0, 5.0, 1.0,
     'Degrees between headings when the Sun glare study takes every heading as equally likely'),
    ('Light', 'ground_albedo', 'Ground albedo', '', 2, 0.01, 1.0, 'Reflectance of the road (asphalt about 0.12)'),
    ('Light', 'body_albedo', 'Body albedo', '', 2, 0.05, 1.0, 'Reflectance of the car\'s paint (0.3: mid grey)'),
    ('Light', 'glare_k', 'Glare coefficient', '', 2, 0.1, 1.0,
     'Veiling glare from the sun near the field: k·E/θ² cd/m² (θ in degrees). The eye\'s is 10; 1 is assumed for a '
     'clean coated lens, more for a dirty window'),
    ('Light', 'veil_index', 'Veiling index', '%', 1, 0.5, 100.0,
     'Uniform veil as a share of the image\'s mean luminance (ISO 9358 veiling glare index; 2 % assumed)'),
    ('Scenes', 'low_sun_el_deg', 'Low sun at', '°', 1, 1.0, 1.0, 'Low sun ahead: the sun\'s elevation'),
    ('Scenes', 'exit_m', 'Tunnel exit', 'm', 1, 1.0, 1.0,
     'Tunnel exit: the portal this far ahead of the front bumper (negative: the front is already out)'),
    ('Scenes', 'interior_cd', 'Tunnel road', 'cd/m²', 1, 1.0, 1.0, 'Road luminance inside the tunnel'),
    ('Limits', 'seam_limit_ev', 'Seam step', 'EV', 2, 0.1, 1.0, 'A seam step above this after balancing is a finding'),
    ('Limits', 'body_limit', 'Body share', '%', 0, 5.0, 100.0,
     'A camera whose unmasked metering the body, rig and housings fill above this share is a finding'),
)
CAM_COLS = (('Camera', 'The camera'),
            ('Removed', 'Share of the camera\'s metering the mask removes (body, rig, housings, sky, outside the image '
                        'circle), %'),
            ('Body', 'Share of the unmasked (whole-frame) metering on the body, rig and housings, %'),
            ('EV', 'Exposure value at ISO 100 the masked AE settles on in the scene shown (campaign: the median)'),
            ('Unmasked Δ', 'How far the whole-frame metering would move the exposure, EV (+: the ground darker). '
                           'Campaign: the mean'),
            ('Exposure', 'Exposure time, ms, in the scene shown (campaign: the range of EV)'),
            ('Gain', 'Sensor gain, dB, in the scene shown'),
            ('Veil', 'Veiling glare over the metered ground as a share of mid-grey, % (campaign: the worst)'),
            ('HDR', 'Share of the image within ½ EV of an HDR sub-exposure transition (the SNR dip), %'))
VIEWS = (('after', 'After balancing'), ('before', 'Before'))
TINT_BODY = STATUS['serious']
TINT_SKY = SEQ[4]


def inputs_key(project) -> str:
    """Everything the study reads, as text (the shown scenario aside): a result made from other inputs is stale."""
    m = project.vehicle.mesh
    v = project.variant
    isp = I.settings_of(project).to_dict()            # defaults filled in: a partial dict and its full form agree
    isp.pop('scenario', None)
    return json.dumps({'mesh': [id(m), getattr(m, 'n_tris', 0), project.vehicle.occluder_target],
                       'variant': v.to_dict(), 'rig': [r.to_dict() for r in project.rig], 'isp': isp,
                       'exposure': project.analysis.exposure, 'weather': project.analysis.weather,
                       'housings': project.analysis.housings, 'library': project.section('library')},
                      sort_keys=True, default=str)


class _Sig(QObject):
    done = pyqtSignal(int, object)
    failed = pyqtSignal(int, str)
    progress = pyqtSignal(int, float, str)


def _num(text, tip='', color=None):
    it = QTableWidgetItem(text)
    it.setFont(mono(12))
    it.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    it.setFlags(it.flags() & ~Qt.ItemFlag.ItemIsEditable)
    if tip:
        it.setToolTip(tip)
    if color is not None:
        it.setForeground(color)
    return it


def _fmt(v, dp=1, sign=False):
    if v is None or not math.isfinite(v):
        return '—'
    return f'{v:+.{dp}f}' if sign else f'{v:.{dp}f}'


# =========================================================================== drawing
def _qimage(rgb: np.ndarray) -> QImage:
    rgb = np.ascontiguousarray(rgb, np.uint8)
    h, w = rgb.shape[:2]
    return QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()


def tinted(rgb: np.ndarray, mask: np.ndarray, alpha=0.55) -> np.ndarray:
    """The preview with the excluded classes tinted: body, rig and housings orange, the sky blue."""
    out = rgb.astype(np.float32)
    for cls, hexc in ((I.BODY, TINT_BODY), (I.HOUSING, TINT_BODY), (I.SKY, TINT_SKY)):
        c = np.array([int(hexc[k:k + 2], 16) for k in (1, 3, 5)], np.float32)
        m = mask == cls
        out[m] = out[m] * (1 - alpha) + c * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


class MaskGallery(QWidget):
    """Every camera's view of the synthetic parking scene through its lens, the excluded parts tinted, the horizon
    drawn. Click a view to select its camera."""
    picked = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.res = None
        self.sel = ''
        self.images = {}
        self.setMinimumHeight(220)
        THEME.changed.connect(self._retint)

    def set_result(self, res, sel=''):
        self.res = res
        self.sel = sel
        self._retint()

    def _retint(self, *_):
        self.images = {}
        if self.res is not None:
            for uid, pv in self.res.previews.items():
                self.images[uid] = _qimage(tinted(pv['rgb'], pv['mask']))
        self.update()

    def _cells(self):
        res = self.res
        n = len(res.cams) if res is not None else 0
        if not n:
            return []
        cols = 3 if n > 4 else max(1, min(n, 2))
        rows = int(math.ceil(n / cols))
        W, H = self.width(), self.height() - 18
        cw, ch = W / cols, H / rows
        out = []
        for k, c in enumerate(res.cams):
            r, q = divmod(k, cols)
            out.append((c, QRectF(q * cw + 3, 18 + r * ch + 3, cw - 6, ch - 6)))
        return out

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        p.setFont(sans(11))
        p.setPen(THEME.color('muted'))
        p.drawText(QRectF(0, 0, self.width(), 16), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                   'Metering masks: body, rig and housings orange, sky blue, the horizon dashed')
        if self.res is None:
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, 'Run the study to see the masks')
            return
        for c, r in self._cells():
            img = self.images.get(c['uid'])
            w, h = c['size']
            s = min(r.width() / w, (r.height() - 16) / h)
            tr = QRectF(r.left() + (r.width() - w * s) / 2, r.top() + 16, w * s, h * s)
            if img is not None:
                p.drawImage(tr, img)
            else:
                p.fillRect(tr, THEME.color('raised'))
            g = self.res.geoms.get(c['uid'])
            if g is not None:
                p.setPen(QPen(THEME.color('ink', 0.8), 1, Qt.PenStyle.DashLine))
                for ln in g.horizon:
                    p.drawPolyline(QPolygonF([QPointF(tr.left() + u * s, tr.top() + v * s) for u, v in ln]))
            if c['uid'] == self.sel:
                p.setPen(QPen(THEME.cam(c['slot']), 2))
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawRect(tr.adjusted(-1, -1, 1, 1))
            p.setFont(mono(11, QFont.Weight.DemiBold))
            p.setPen(THEME.cam(c['slot']))
            p.drawText(QRectF(r.left(), r.top(), r.width(), 15), Qt.AlignmentFlag.AlignLeft, c['tag'])
            p.setFont(sans(10))
            p.setPen(THEME.color('ink2'))
            p.drawText(QRectF(r.left(), r.top(), r.width(), 15), Qt.AlignmentFlag.AlignRight,
                       f'{100 * c["removed_share"]:.0f} % removed · body {100 * c["body_share_unmasked"]:.0f} %')

    def mousePressEvent(self, e):
        for c, r in self._cells():
            if r.contains(e.position()):
                self.picked.emit(c['uid'])
                return

    def resizeEvent(self, e):
        self.update()


class GridView(QWidget):
    """The selected camera's metering grid (weights shaded, numbers in the cells) and its AE rectangles."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.cam = None
        self.wmax = 15
        self.setMinimumHeight(160)
        THEME.changed.connect(lambda *_: self.update())

    def set_camera(self, cam, wmax):
        self.cam, self.wmax = cam, max(int(wmax), 1)
        self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.setFont(sans(11))
        p.setPen(THEME.color('muted'))
        c = self.cam
        title = 'Metering grid and AE rectangles' + (f' · {c["tag"]}' if c else '')
        p.drawText(QRectF(0, 0, self.width(), 16), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, title)
        if c is None:
            return
        G = np.asarray(c['grid'])
        rows, cols = G.shape
        W, H = c['size']
        s = min(self.width() / W, (self.height() - 36) / H)
        R = QRectF((self.width() - W * s) / 2, 18, W * s, H * s)
        cw, ch = R.width() / cols, R.height() / rows
        small = mono(9) if cw > 16 else None
        lut = [QColor(x) for x in SEQ]
        for r in range(rows):
            for q in range(cols):
                v = int(G[r, q])
                cell = QRectF(R.left() + q * cw, R.top() + r * ch, cw, ch)
                if v > 0:
                    p.fillRect(cell, lut[min(int(v / self.wmax * (len(lut) - 1)), len(lut) - 1)])
                else:
                    p.fillRect(cell, THEME.color('raised'))
                if small is not None:
                    p.setFont(small)
                    p.setPen(THEME.color('ink') if v < self.wmax * 0.6 else QColor('#ffffff'))
                    p.drawText(cell, Qt.AlignmentFlag.AlignCenter, str(v))
        p.setPen(QPen(THEME.color('hair'), 1))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRect(R)
        p.setPen(QPen(THEME.cam(c['slot']), 2))
        for rc in c['rects']:
            p.drawRect(QRectF(R.left() + rc['left'] * s, R.top() + rc['top'] * s, (rc['right'] - rc['left']) * s,
                              (rc['bottom'] - rc['top']) * s))
        cov = c['rect_cover']
        p.setFont(sans(10))
        p.setPen(THEME.color('ink2'))
        p.drawText(QRectF(0, R.bottom() + 2, self.width(), 16), Qt.AlignmentFlag.AlignLeft,
                   f'{len(c["rects"])} rectangles hold {100 * cov["metered_inside"]:.0f} % of the metered pixels; '
                   f'{100 * cov["spill"]:.1f} % of their area is not ground')


class SeamChart(ChartBase):
    """Campaign: the worst seam step over days and headings at each time of day, one line per seam. A scenario: one
    bar per seam before (hollow) and after (solid) balancing. The dashed line is the limit."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.res = None
        self.scenario = 'campaign'
        self.view = 'after'
        self.title = 'Exposure step across each seam'
        self.setMinimumHeight(200)
        self._marks = []

    def set_data(self, res, scenario, view):
        self.res, self.scenario, self.view = res, scenario, view
        self.tip = None
        self.update()

    def series(self):
        """[(seam key, slot, [(x, value EV)])] as drawn (for the tests and the tooltip)."""
        res = self.res
        if res is None:
            return []
        if self.scenario == 'campaign':
            hrs = res.campaign.get('hours', [])
            k = 'curve_after' if self.view == 'after' else 'curve_before'
            return [(s['key'], s['slot_a'], list(zip(hrs, s.get(k, [])))) for s in res.seams]
        sd = res.scenarios[self.scenario]['seams']
        return [(s['key'], s['slot_a'], [(0, abs(x['before_ev'])), (1, abs(x['after_ev']))])
                for s, x in zip(res.seams, sd)]

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        res = self.res
        sub = I.SCENARIO_LABEL.get(self.scenario, '')
        if self.scenario == 'campaign':
            sub += ' · worst over days and headings, ' + ('after balancing' if self.view == 'after' else 'before')
        self.subtitle = sub
        self.paint_title(p)
        self._marks = []
        ser = self.series()
        if not ser:
            p.setFont(sans(11))
            p.setPen(THEME.color('muted'))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, 'No seams' if res is not None else '')
            return
        lim = res.settings.seam_limit_ev
        top = max([lim * 1.25] + [v for _, _, pts in ser for _, v in pts]) * 1.1
        step = 0.25 if top <= 1.5 else 0.5 if top <= 3 else 1.0
        top = math.ceil(top / step) * step
        r = QRectF(44, 26, self.width() - 56, self.height() - 26 - 34)
        self.paint_grid_y(p, r, [k * step for k in range(int(round(top / step)) + 1)], top,
                          fmt=lambda v: f'{v:g}')
        p.setFont(mono(10))
        p.setPen(THEME.color('muted'))
        p.drawText(QRectF(0, r.top() - 16, 40, 14), Qt.AlignmentFlag.AlignRight, 'EV')

        def Y(v):
            return r.bottom() - r.height() * v / top
        if self.scenario == 'campaign':
            hrs = res.campaign.get('hours', [])
            if not hrs:
                return
            h0, h1 = hrs[0] - 0.5 * (hrs[1] - hrs[0] if len(hrs) > 1 else 1), hrs[-1] + 0.5 * (
                hrs[1] - hrs[0] if len(hrs) > 1 else 1)
            span = (h1 - h0) % 24 or 24

            def X(h):
                return r.left() + r.width() * (((h - h0) % 24) / span)
            p.setFont(mono(10))
            p.setPen(THEME.color('muted'))
            for h in range(int(math.ceil(h0)), int(math.floor(h0 + span)) + 1, 2):
                x = X(h % 24)
                p.drawText(QRectF(x - 16, r.bottom() + 3, 32, 14), Qt.AlignmentFlag.AlignCenter, f'{h % 24:02d}')
            for key, slot, pts in ser:
                pen = QPen(THEME.cam(slot), 2)
                p.setPen(pen)
                poly = QPolygonF([QPointF(X(h), Y(v)) for h, v in pts])
                p.drawPolyline(poly)
                for h, v in pts:
                    self._marks.append((QPointF(X(h), Y(v)), key, h, v))
        else:
            n = len(ser)
            bw = r.width() / max(n, 1)
            for k, (key, slot, pts) in enumerate(ser):
                b, a = pts[0][1], pts[1][1]
                x0 = r.left() + k * bw + bw * 0.18
                w = bw * 0.30
                col = THEME.cam(slot)
                p.setPen(QPen(col, 1.5))
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawPath(self.bar_path(QRectF(x0, Y(b), w, r.bottom() - Y(b))))
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(col)
                p.drawPath(self.bar_path(QRectF(x0 + w + 2, Y(a), w, r.bottom() - Y(a))))
                p.setFont(mono(10))
                p.setPen(THEME.color('ink2'))
                p.drawText(QRectF(r.left() + k * bw, r.bottom() + 3, bw, 14), Qt.AlignmentFlag.AlignCenter, key)
                self._marks.append((QPointF(x0 + w, Y(max(a, b))), key, None, (b, a)))
        p.setPen(QPen(STATUS['critical'], 1, Qt.PenStyle.DashLine))
        p.drawLine(QPointF(r.left(), Y(lim)), QPointF(r.right(), Y(lim)))
        # legend
        if self.scenario == 'campaign':
            x = r.left()
            p.setFont(mono(10))
            for key, slot, _ in ser:
                p.fillRect(QRectF(x, self.height() - 12, 8, 8), THEME.cam(slot))
                p.setPen(THEME.color('ink2'))
                p.drawText(QRectF(x + 11, self.height() - 15, 60, 14), Qt.AlignmentFlag.AlignLeft, key)
                x += 16 + 7 * len(key) + 10
        else:
            p.setFont(sans(10))
            p.setPen(THEME.color('muted'))
            p.drawText(QRectF(0, self.height() - 15, self.width() - 8, 14), Qt.AlignmentFlag.AlignRight,
                       'hollow: before balancing · solid: after')
        self.paint_tip(p)

    def mouseMoveEvent(self, e):
        best, bd = None, 18.0
        for pt, key, h, v in self._marks:
            d = math.hypot(pt.x() - e.position().x(), pt.y() - e.position().y())
            if d < bd:
                best, bd = (pt, key, h, v), d
        if best is None:
            self.tip = None
        else:
            pt, key, h, v = best
            if h is None:
                self.set_tip(pt, f'{v[1]:.2f} EV after', f'{v[0]:.2f} EV before balancing', f'seam {key}')
            else:
                self.set_tip(pt, f'{v:.2f} EV', f'seam {key} at {int(h):02d}:{int(round((h % 1) * 60)) % 60:02d}',
                             'worst over days and headings')
        self.update()


# =========================================================================== the page
class IspPage(QWidget):
    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        self.res: I.IspResult | None = None
        self.busy = False
        self.stale = False
        self.status_text = ''
        self.sel_uid = ''
        self._gen = 0
        self._cancel = threading.Event()
        self._for = None
        self._syncing = False
        self.sig = _Sig()
        self.sig.done.connect(self._done)
        self.sig.failed.connect(self._failed)
        self.sig.progress.connect(self._progress)
        self.setObjectName('base')
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        split = QSplitter(Qt.Orientation.Horizontal)
        split.setHandleWidth(2)
        split.addWidget(self._left())
        # ---------------- centre: masks and the grid
        cw = QWidget()
        cw.setObjectName('panel')
        cl = QVBoxLayout(cw)
        cl.setContentsMargins(8, 6, 8, 6)
        cl.setSpacing(6)
        self.head = label('', ink2=True, wrap=True)
        cl.addWidget(self.head)
        self.gallery = MaskGallery()
        self.gallery.picked.connect(self.select)
        cl.addWidget(self.gallery, 3)
        self.grid_view = GridView()
        cl.addWidget(self.grid_view, 2)
        split.addWidget(cw)
        # ---------------- right: the seam chart, the cameras, the findings
        right = QSplitter(Qt.Orientation.Vertical)
        right.setHandleWidth(2)
        sw = QWidget()
        sw.setObjectName('panel')
        sl = QVBoxLayout(sw)
        sl.setContentsMargins(8, 6, 8, 6)
        sl.setSpacing(4)
        row = QHBoxLayout()
        self.scenario = SegmentedControl([(k, I.SCENARIO_LABEL[k]) for k in I.SCENARIOS], 'campaign')
        self.scenario.setToolTip('The scene the chart and the table show: the campaign\'s daylight, or a hard scene')
        self.scenario.changed.connect(lambda k: self.set_option('scenario', k))
        row.addWidget(self.scenario)
        row.addStretch(1)
        self.view = SegmentedControl(VIEWS, 'after')
        self.view.setToolTip('The campaign curve before or after the bird\'s-eye gain balancing')
        self.view.changed.connect(lambda k: self.refresh())
        row.addWidget(self.view)
        sl.addLayout(row)
        self.chart = SeamChart()
        sl.addWidget(self.chart, 1)
        self.balance = label('', small=True, wrap=True, ink2=True)
        sl.addWidget(self.balance)
        right.addWidget(sw)
        tw = QWidget()
        tw.setObjectName('panel')
        tl = QVBoxLayout(tw)
        tl.setContentsMargins(8, 6, 8, 6)
        tl.setSpacing(4)
        tl.addWidget(label('Cameras', bold=True))
        t = self.table = QTableWidget(0, len(CAM_COLS))
        t.verticalHeader().setVisible(False)
        t.verticalHeader().setDefaultSectionSize(ROW_H)
        t.horizontalHeader().setFixedHeight(ROW_H)
        t.setShowGrid(False)
        t.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        t.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        for c, (head, tip) in enumerate(CAM_COLS):
            hi = QTableWidgetItem(head)
            hi.setToolTip(tip)
            t.setHorizontalHeaderItem(c, hi)
        hh = t.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for c in range(1, len(CAM_COLS)):
            hh.setSectionResizeMode(c, QHeaderView.ResizeMode.ResizeToContents)
        t.itemSelectionChanged.connect(self._row_picked)
        tl.addWidget(t, 1)
        right.addWidget(tw)
        fw = QWidget()
        fw.setObjectName('panel')
        fl = QVBoxLayout(fw)
        fl.setContentsMargins(10, 6, 10, 8)
        fl.setSpacing(4)
        fl.addWidget(label('Findings', bold=True))
        self.find_host = QWidget()
        self.find_lay = QVBoxLayout(self.find_host)
        self.find_lay.setContentsMargins(0, 0, 0, 0)
        self.find_lay.setSpacing(4)
        self.find_lay.addStretch(1)
        sc = QScrollArea()
        sc.setWidgetResizable(True)
        sc.setWidget(self.find_host)
        fl.addWidget(sc, 1)
        self.findings: list[StatusLine] = []
        right.addWidget(fw)
        right.setSizes([330, 260, 260])
        split.addWidget(right)
        split.setSizes([300, 620, 680])
        lay.addWidget(split)
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(400)
        self.timer.timeout.connect(self.run)
        ctx.store.changed.connect(self._on_edit)
        ctx.store.project_replaced.connect(self._reset)
        self._sync_fields()
        self.refresh()

    # ------------------------------------------------------------------ settings panel
    def _left(self):
        w = QWidget()
        w.setObjectName('panel')
        w.setMinimumWidth(280)
        outer = QVBoxLayout(w)
        outer.setContentsMargins(0, 0, 0, 0)
        sc = QScrollArea()
        sc.setWidgetResizable(True)
        host = QWidget()
        ll = QVBoxLayout(host)
        ll.setContentsMargins(8, 6, 8, 8)
        ll.setSpacing(4)
        top = QHBoxLayout()
        top.addWidget(label('Exposure metering'))
        top.addStretch(1)
        top.addWidget(label('masks and HDR seams', muted=True, small=True))
        ll.addLayout(top)
        ll.addWidget(hline())
        ll.addWidget(label('What each camera\'s auto-exposure should meter (the ground, not the body, the rig, the '
                           'housings or the sky), and how far neighbouring cameras\' exposures drift apart across the '
                           'stitch seams. Clear sky and typical photometric figures; the campaign\'s place, dates and '
                           'headings are the Sun glare study\'s.', muted=True, small=True, wrap=True))
        ll.addSpacing(4)
        grid = QGridLayout()
        grid.setVerticalSpacing(2)
        grid.setHorizontalSpacing(6)
        grid.setColumnStretch(1, 1)

        def row(text, wdg, tip):
            a = label(text, ink2=True)
            a.setFixedWidth(104)
            a.setToolTip(tip)
            wdg.setToolTip(tip)
            i = grid.rowCount()
            grid.addWidget(a, i, 0)
            grid.addWidget(wdg, i, 1)
        self.metering = SegmentedControl([('mean_log', 'Mean log'), ('centre', 'Centre')], 'mean_log')
        self.metering.changed.connect(lambda k: self.set_option('metering', k))
        row('Metering', self.metering, 'How the AE averages its metered region: the mean of log luminance, or a '
                                       'centre-weighted linear mean')
        self.gridsize = SegmentedControl([('8', '8 × 8'), ('16', '16 × 16'), ('32', '32 × 32')], '16')
        self.gridsize.changed.connect(lambda k: self.set_grid(int(k)))
        row('Grid', self.gridsize, 'Cells of the exported metering grid (weights 0–15)')
        self.scale = SegmentedControl([('1', 'Full'), ('2', '½'), ('4', '¼')], '1')
        self.scale.changed.connect(lambda k: self.set_option('mask_scale', int(k)))
        row('Mask', self.scale, 'Resolution of the exported masks: the sensor\'s, or a fraction of it')
        ll.addLayout(grid)
        self.fields = {}
        group = None
        g2 = None
        for grp, key, text, unit, dec, step, mul, tip in FIELDS:
            if grp != group:
                if g2 is not None:
                    ll.addLayout(g2)
                ll.addSpacing(4)
                ll.addWidget(label(grp, bold=True))
                g2 = QGridLayout()
                g2.setVerticalSpacing(2)
                g2.setHorizontalSpacing(6)
                g2.setColumnStretch(2, 1)
                group = grp
            a = label(text, ink2=True)
            a.setFixedWidth(104)
            a.setToolTip(tip)
            f = ScrubSpinBox('', 'plain', dec, unit, step=step, width=110)
            f.setToolTip(tip)
            f.committed.connect(lambda v, k=key, m=mul: self.set_option(k, v / m))
            i = g2.rowCount()
            g2.addWidget(a, i, 0)
            g2.addWidget(f, i, 1)
            self.fields[key] = (f, mul)
        if g2 is not None:
            ll.addLayout(g2)
        ll.addStretch(1)
        sc.setWidget(host)
        fit_width(sc)
        outer.addWidget(sc, 1)
        bot = QVBoxLayout()
        bot.setContentsMargins(8, 4, 8, 8)
        bot.setSpacing(4)
        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.setTextVisible(False)
        self.bar.setFixedHeight(4)
        bot.addWidget(self.bar)
        self.status = label('', muted=True, small=True, wrap=True)
        bot.addWidget(self.status)
        bl = QHBoxLayout()
        self.b_run = button('Run', 'primary', 'play', 'Compute the masks and the seam study now')
        self.b_run.clicked.connect(lambda: self.run())
        self.b_cancel = button('Cancel', None, 'stop', 'Stop the running study')
        self.b_cancel.clicked.connect(self.cancel)
        self.b_export = button('Export…', None, 'export', 'Masks (PNG), metering grids and AE rectangles (JSON), the '
                                                          'seam steps over the day (CSV) and the study (JSON)')
        self.b_export.clicked.connect(lambda: self.export())
        bl.addWidget(self.b_run)
        bl.addWidget(self.b_cancel)
        bl.addStretch(1)
        bl.addWidget(self.b_export)
        bot.addLayout(bl)
        outer.addLayout(bot)
        return w

    @property
    def settings(self) -> I.IspSettings:
        return I.settings_of(self.ctx.store.project)

    def set_option(self, key, value) -> bool:
        """Change one setting, saved with the project as one undo step; False when nothing changes."""
        if self._syncing:
            return False
        st = self.settings
        old = st.to_dict()
        setattr(st, key, value)
        new = I.IspSettings.from_dict(st.to_dict()).to_dict()
        if new == old:
            self._sync_fields()
            return False
        shown = new[key]
        name = next((f[2].lower() for f in FIELDS if f[1] == key), key.replace('_', ' '))
        text = I.SCENARIO_LABEL.get(shown, shown) if key == 'scenario' else (
            f'{shown:g}' if isinstance(shown, (int, float)) else str(shown))

        def fn(p):
            p.analysis.isp = new
        self.ctx.store.edit('analysis', fn, f'Exposure metering · {name} {text}')
        return True

    def set_grid(self, n: int) -> bool:
        st = self.settings
        if st.grid_cols == n and st.grid_rows == n:
            return False
        new = dict(st.to_dict(), grid_cols=n, grid_rows=n)

        def fn(p):
            p.analysis.isp = I.IspSettings.from_dict(new).to_dict()
        self.ctx.store.edit('analysis', fn, f'Exposure metering · grid {n} × {n}')
        return True

    def _sync_fields(self):
        st = self.settings
        self._syncing = True
        try:
            for key, (f, mul) in self.fields.items():
                f.set_value(float(getattr(st, key)) * mul)
            self.metering.set_value(st.metering)
            self.scenario.set_value(st.scenario)
            self.gridsize.set_value(str(st.grid_cols) if st.grid_cols == st.grid_rows else '')
            self.scale.set_value(str(st.mask_scale))
        finally:
            self._syncing = False

    # ------------------------------------------------------------------ the job
    def _on_edit(self, *_):
        self._sync_fields()
        if self.res is not None and self._for is not None:
            self.stale = inputs_key(self.ctx.store.project) != self._for[1]
        if self.stale and self.isVisible():
            self.timer.start()
        self.refresh()

    def _reset(self, *_):
        self.cancel()
        self.res, self._for, self.stale, self.sel_uid = None, None, False, ''
        self._sync_fields()
        self.refresh()
        if self.isVisible():
            self.timer.start()

    def showEvent(self, e):
        super().showEvent(e)
        if (self.res is None or self.stale) and not self.busy:
            self.timer.start()

    def activate(self):
        if (self.res is None or self.stale) and not self.busy:
            self.run()

    def _engine(self):
        win = self.ctx.window
        if win is not None and hasattr(win, '_shared_engine'):
            return win._shared_engine()
        return self.ctx.analysis.engine.share()

    def run(self) -> bool:
        """Start the study on a snapshot of the project (a running one is cancelled)."""
        self.timer.stop()
        p = self.ctx.store.project
        project = p.snapshot()
        st = I.settings_of(project)
        eng = self._engine()
        self._cancel.set()
        self._cancel = threading.Event()
        cancel = self._cancel
        self._gen += 1
        gen = self._gen
        key = (p, inputs_key(p))
        self.busy = True
        self.status_text = 'Computing the metering masks and the seam study…'
        self.refresh()

        def work():
            try:
                res = I.run(project, eng, st, progress=lambda f, t='': self.sig.progress.emit(gen, f, t),
                            cancel=cancel.is_set, previews=True)
                self.sig.done.emit(gen, (key, res))
            except RuntimeError:                  # the page went while computing
                pass
            except Exception as e:
                try:
                    self.sig.failed.emit(gen, f'{type(e).__name__}: {e}\n{traceback.format_exc()}')
                except RuntimeError:
                    pass
        threading.Thread(target=work, daemon=True).start()
        return True

    def cancel(self):
        self._cancel.set()
        if self.busy:
            self._gen += 1
            self.busy = False
            self.status_text = 'Cancelled'
            self.refresh()

    def _progress(self, gen, f, text):
        if gen == self._gen:
            self.bar.setValue(int(f * 1000))
            if text:
                self.status.setText(f'{100 * f:.0f} % · {text}')

    def _done(self, gen, out):
        if gen != self._gen:
            return
        self.busy = False
        key, res = out
        if res is None or key[0] is not self.ctx.store.project:
            self.status_text = 'Cancelled' if res is None else ''
            self.refresh()
            return
        self.res, self._for = res, key
        self.stale = key[1] != inputs_key(self.ctx.store.project)
        c = res.campaign
        self.status_text = (f'{len(res.cams)} cameras · {c["samples"]:,} daylight samples ({c["days_sampled"]} days, '
                            f'{len(c["headings"])} headings) · {res.metrics["seconds"]:.1f} s'
                            + (f' · {c["note"]}' if c['note'] else ''))
        if not self.sel_uid or res.cam(self.sel_uid) is None:
            self.sel_uid = res.cams[0]['uid'] if res.cams else ''
        self.refresh()
        if self.stale and self.isVisible():
            self.timer.start()

    def _failed(self, gen, msg):
        if gen != self._gen:
            return
        self.busy = False
        self.status_text = 'Failed: ' + msg.splitlines()[0]
        traceback.print_exc()
        self.refresh()

    # ------------------------------------------------------------------ the views
    def select(self, uid):
        self.sel_uid = uid
        self.refresh()

    def _row_picked(self):
        if self._syncing:
            return
        rows = self.table.selectionModel().selectedRows() if self.table.selectionModel() else []
        if rows and self.res is not None and rows[0].row() < len(self.res.cams):
            uid = self.res.cams[rows[0].row()]['uid']
            if uid != self.sel_uid:
                self.select(uid)

    def refresh(self):
        res = self.res
        self.b_cancel.setEnabled(self.busy)
        self.b_export.setEnabled(res is not None)
        self.bar.setVisible(self.busy)
        self.status.setText(self.status_text + (' · out of date: re-running' if self.stale and res is not None and
                                                not self.busy and self.status_text and not
                                                self.status_text.startswith('Cancelled') else ''))
        st = self.settings
        scen = st.scenario
        self.view.setEnabled(scen == 'campaign')
        self.chart.set_data(res, scen, self.view.value() or 'after')
        self.gallery.set_result(res, self.sel_uid)
        self.grid_view.set_camera(res.cam(self.sel_uid) if res is not None else None,
                                  res.settings.weight_max if res is not None else 15)
        if res is None:
            self.head.setText('No result yet.')
            self.balance.setText('')
            self._fill_table()
            self._fill_findings()
            return
        lo, hi = res.metrics['gain_range']
        ev_lo, ev_hi = res.metrics['gain_range_ev']
        self.head.setText(f'{res.variant} · {I.METERING_LABEL[res.settings.metering]} metering · '
                          f'{len(res.seams)} seams')
        if scen == 'campaign':
            mb = max([s['campaign']['before_max_ev'] for s in res.seams if 'campaign' in s] or [0.0])
            ma = max([s['campaign']['after_max_ev'] for s in res.seams if 'campaign' in s] or [0.0])
            what = f'Over the campaign the seams step by up to {mb:.2f} EV before and {ma:.2f} EV after balancing.'
        else:
            sd = res.scenarios[scen]['seams']
            mb = max([abs(x['before_ev']) for x in sd] or [0.0])
            ma = max([abs(x['after_ev']) for x in sd] or [0.0])
            what = f'{I.SCENARIO_LABEL[scen]}: the seams step by up to {mb:.2f} EV before and {ma:.2f} EV after.'
        self.balance.setText(f'{what} The stitcher\'s gains span {lo:g}–{hi:g}× ({ev_lo:+.1f} to {ev_hi:+.1f} EV '
                             f'through a {res.settings.output_gamma:g} gamma output); the limit is '
                             f'{res.settings.seam_limit_ev:g} EV.')
        self._fill_table()
        self._fill_findings()

    def _fill_table(self):
        res = self.res
        t = self.table
        self._syncing = True
        try:
            if res is None:
                t.setRowCount(0)
                return
            scen = self.settings.scenario
            t.setRowCount(len(res.cams))
            lim = res.settings.body_limit
            for r, c in enumerate(res.cams):
                it = QTableWidgetItem(c['tag'])
                it.setForeground(THEME.cam(c['slot']))
                it.setFlags(it.flags() & ~Qt.ItemFlag.ItemIsEditable)
                it.setToolTip(c['model'])
                t.setItem(r, 0, it)
                t.setItem(r, 1, _num(f'{100 * c["removed_share"]:.0f}'))
                b = c['body_share_unmasked']
                t.setItem(r, 2, _num(f'{100 * b:.0f}', color=QColor(STATUS['warning']) if b > lim else None))
                if scen == 'campaign':
                    cm = c.get('campaign') or {}
                    t.setItem(r, 3, _num(_fmt(cm.get('ev_median'), 1)))
                    t.setItem(r, 4, _num(_fmt(cm.get('unmasked_off_mean_ev'), 2, True),
                                         f'up to {_fmt(cm.get("unmasked_off_max_ev"), 2, True)} EV'))
                    t.setItem(r, 5, _num(f'EV {_fmt(cm.get("ev_min"), 1)}–{_fmt(cm.get("ev_max"), 1)}'))
                    t.setItem(r, 6, _num('—'))
                    t.setItem(r, 7, _num(_fmt(cm.get('veil_max_pct'), 0)))
                    t.setItem(r, 8, _num('—'))
                else:
                    x = res.scenarios[scen]['cams'][c['uid']]
                    t.setItem(r, 3, _num(_fmt(x['ev'], 1), f'wanted {x["ev_wanted"]:.2f}' if x['short_ev'] > 0.05
                                         else ''))
                    t.setItem(r, 4, _num(_fmt(x['unmasked_off_ev'], 2, True)))
                    t.setItem(r, 5, _num(_fmt(x['exposure_ms'], 2)))
                    t.setItem(r, 6, _num(_fmt(x['gain_db'], 1),
                                         color=QColor(STATUS['warning']) if x['short_ev'] > 0.05 else None))
                    t.setItem(r, 7, _num(_fmt(x['veil_pct'], 1)))
                    h = x['hdr']
                    t.setItem(r, 8, _num(_fmt(100 * h['near_transition'], 0),
                                         f'{100 * h["clipped"]:.1f} % clipped, {100 * h["under_floor"]:.1f} % under '
                                         f'the noise floor; scene {h["scene_db"]:.0f} dB'))
            sel = next((i for i, c in enumerate(res.cams) if c['uid'] == self.sel_uid), -1)
            if sel >= 0:
                t.selectRow(sel)
        finally:
            self._syncing = False

    def _fill_findings(self):
        for w in self.findings:
            w.hide()
            w.deleteLater()
        self.findings = []
        if self.res is None:
            return
        for f in self.res.findings:
            ln = StatusLine(f['severity'] if f['severity'] in STATUS else ('good' if f['severity'] == 'good' else ''),
                            f['message'])
            ln.text.setProperty('small', True)
            self.find_lay.insertWidget(self.find_lay.count() - 1, ln)
            self.findings.append(ln)

    # ------------------------------------------------------------------ export
    def export(self, folder=None):
        if self.res is None:
            return []
        if folder is None:
            folder = QFileDialog.getExistingDirectory(self, 'Export the metering masks and seam study into')
            if not folder:
                return []
        from ...core import provenance as PV          # where the JSON came from
        paths = PV.stamp_paths(I.export(self.res, self.ctx.store.project, folder), self.ctx.store.project, folder)
        self.ctx.msg(f'Exported the exposure metering study to {folder}', 5000)
        return paths
